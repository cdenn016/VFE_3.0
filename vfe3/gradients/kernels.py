r"""Optimized hand-derived belief-gradient kernels for VFE_3.0, with oracle fallback.

A family-keyed registry of analytic (grad_mu, grad_sigma) kernels for the QUERY-SIDE
(filtering) gradient. belief_gradients() uses the registered kernel only for the
filtering + gaussian_diagonal + KL (renyi_order=1) + canonical case; every other case
(smoothing, non-KL family, Renyi alpha != 1, surrogate) FALLS BACK to the autograd
oracle -- so a new divergence works immediately and correctly, accelerated later by
registering a kernel. Kernels return RAW Euclidean dF (no preconditioning/retraction).
"""

from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, List, Optional, Tuple

import torch

from vfe3.alpha_i import alpha_gradient_coefficient, alpha_is_per_coord
from vfe3.families.base import get_family
from vfe3.families.policy import family_instance as _family_instance
from vfe3.families.gaussian import diag_kl_unclamped, diag_kl_unclamped_per_coord
from vfe3.free_energy import attention_weights, pairwise_energy, self_divergence_for_alpha
from vfe3.geometry.transport import (
    CompactFactoredTransport,
    FactoredTransport,
    RopeTransport,
    get_transport_registration,
    transport_covariance,
    transport_mean,
)
from vfe3.gradients.oracle import _transport_to_float, belief_gradients_autograd
from vfe3.families.gaussian import FullGaussian
from vfe3.gradients.pairwise_stats import diagonal_kl_pair_stats
from vfe3.numerics import _count_mm_cholesky_fallback, safe_cholesky

_KERNELS: Dict[str, Callable] = {}


@dataclass(frozen=True)
class KernelRegistration:
    r"""A registered closed form and the capabilities it declares.

    Mirrors ``DecodeRegistration.covariance_kinds`` (``vfe3/model/prior_bank.py``): the route
    predicate (``uses_kernel_route``) gates on a declared capability rather than a family-name
    literal, so registering a new family's closed form is sufficient to unlock the route it
    declares -- and ONLY that route.

    ``covariance_kinds`` is the resolved set of family covariance structures ("diagonal" and/or
    "full") this registration covers, validated against the registered family's OWN ``cov_kind`` at
    registration time (unlike decoders, a kernel is keyed by exact family name, so mismatch is
    always a registration bug, not a legitimate multi-family share -- caught here rather than
    mis-routed later).

    ``provides_gradient`` / ``provides_mm_exact`` are DECOUPLED capabilities (audit 2026-08-07): a
    family may register the analytic ``(grad_mu, grad_sigma)`` kernel ``belief_gradients`` fetches
    through ``callable``, the closed-form MM minimizer ``mm_exact_update`` implements as its own
    dedicated branch, or both. Registering one never silently unlocks the other -- this is why
    ``gaussian_full``'s mm_exact fusion below declares ``provides_gradient=False``: it has no
    analytic gradient kernel, and ``e_step_update='gradient'`` must keep routing to the autograd
    oracle exactly as it did before this registration existed.
    """

    callable:           Callable
    covariance_kinds:   FrozenSet[str]
    provides_gradient:  bool = True
    provides_mm_exact:  bool = True


_KERNEL_REGISTRATIONS: Dict[str, KernelRegistration] = {}


def register_kernel(
    name: str,

    *,
    override:          bool                       = False,
    covariance_kinds:  Optional[FrozenSet[str]]    = None,
    provides_gradient: bool                        = True,
    provides_mm_exact: bool                        = True,
) -> Callable:
    """Decorator registering a query-side belief-gradient kernel under family ``name``.

    Duplicate keys fail closed (audit 2026-07-01 round-3): a second registration under an
    existing name silently shadowed the first. Pass ``override=True`` to replace deliberately.

    ``covariance_kinds`` declares the family covariance structure(s) this registration covers
    (mirrors ``register_decode``'s ``covariance_kinds``, ``vfe3/model/prior_bank.py``). OMITTED
    (the common case) resolves to the registered family's OWN ``cov_kind``; SUPPLIED is validated
    against it and rejected if inconsistent, so a registration cannot silently claim a capability
    its own family does not have. ``provides_gradient`` / ``provides_mm_exact`` declare which
    route(s) this registration serves -- see :class:`KernelRegistration`.
    """
    def _wrap(fn: Callable) -> Callable:
        if name in _KERNELS and not override:
            raise KeyError(f"kernel {name!r} already registered; pass override=True to replace")
        family_cov_kind = get_family(name).cov_kind
        resolved_kinds = (frozenset({family_cov_kind}) if covariance_kinds is None
                          else frozenset(covariance_kinds))
        if not resolved_kinds or not resolved_kinds <= {"diagonal", "full"}:
            raise ValueError(
                f"kernel {name!r} covariance_kinds must be a nonempty subset of "
                f"{{'diagonal', 'full'}}, got {sorted(resolved_kinds)}"
            )
        if family_cov_kind not in resolved_kinds:
            raise ValueError(
                f"kernel {name!r} declares covariance_kinds={sorted(resolved_kinds)} but family "
                f"{name!r} has cov_kind={family_cov_kind!r}; a kernel must cover its own family's "
                "covariance structure"
            )
        if not (provides_gradient or provides_mm_exact):
            raise ValueError(
                f"kernel {name!r} must declare at least one of provides_gradient/provides_mm_exact"
            )
        _KERNELS[name] = fn
        _KERNEL_REGISTRATIONS[name] = KernelRegistration(
            callable=fn, covariance_kinds=resolved_kinds,
            provides_gradient=provides_gradient, provides_mm_exact=provides_mm_exact,
        )
    
        return fn
    return _wrap


def has_kernel(name: str) -> bool:
    """Whether a hand kernel is registered for family ``name``."""
    return name in _KERNELS


def get_kernel_registration(name: str) -> KernelRegistration:
    """Return the registration-owned capabilities for kernel family ``name`` (KeyError if absent)."""
    if name not in _KERNELS or name not in _KERNEL_REGISTRATIONS:
        raise KeyError(f"no belief-gradient kernel for family {name!r}; available: {sorted(_KERNELS)}")
    return _KERNEL_REGISTRATIONS[name]


def mm_exact_families() -> Tuple[str, ...]:
    """Sorted family names whose registration declares the mm_exact closed-form capability."""
    return tuple(sorted(
        name for name, reg in _KERNEL_REGISTRATIONS.items() if reg.provides_mm_exact
    ))


def _raw_diag_kl(
    mu_q:    torch.Tensor,             # (N, K) query means
    sigma_q: torch.Tensor,             # (N, K) query variances
    mu_p:    torch.Tensor,             # (N, K) prior means
    sigma_p: torch.Tensor,             # (N, K) prior variances

    *,
    eps:     float = 1e-6,
) -> torch.Tensor:                     # (N,) UNCLAMPED KL(q_i || p_i)
    r"""Unclamped diagonal KL(q||p) = 0.5 Sum_k (s_k/t_k + (mu_p-mu_q)^2/t_k - 1 + log(t_k/s_k)).

    The divergence seam returns the clamped value safe_kl_clamp(D, [0, kl_max]);
    this returns the raw D so the kernel can reproduce the oracle's saturation
    mask (the oracle differentiates THROUGH the clamp, whose gradient is 0 once
    D leaves (0, kl_max)).
    """
    return diag_kl_unclamped(mu_q, sigma_q, mu_p, sigma_p, eps=eps)


def _raw_diag_kl_per_coord(
    mu_q:    torch.Tensor,             # (N, K) query means
    sigma_q: torch.Tensor,             # (N, K) query variances
    mu_p:    torch.Tensor,             # (N, K) prior means
    sigma_p: torch.Tensor,             # (N, K) prior variances

    *,
    eps:     float = 1e-6,
) -> torch.Tensor:                     # (N, K) UNCLAMPED per-coordinate KL D^(k)(q_i || p_i)
    r"""Unclamped per-coordinate diagonal KL D^(k) = 0.5 (s_k/t_k + (mu_p-mu_q)^2/t_k - 1 + log(t_k/s_k)).

    The per-coordinate analog of ``_raw_diag_kl`` (the -K of the summed form becomes -1 per
    coordinate). The per-coordinate self-term differentiates through a PER-COORDINATE
    safe_kl_clamp, so the kernel builds its saturation mask from this so a saturated
    coordinate is gated independently of the others -- matching the filtering oracle.
    """
    return diag_kl_unclamped_per_coord(mu_q, sigma_q, mu_p, sigma_p, eps=eps)


@register_kernel("gaussian_diagonal")
def _diag_kl_filtering_kernel(
    mu_q:       torch.Tensor,             # (N, K)
    sigma_q:    torch.Tensor,             # (N, K)
    mu_p:       torch.Tensor,             # (N, K)
    sigma_p:    torch.Tensor,             # (N, K)
    mu_t:       torch.Tensor,             # (N, N, K) transported key means
    sigma_t:    torch.Tensor,             # (N, N, K) transported key variances
    beta:       torch.Tensor,             # (N, N) or (H, N, N) raw attention weights
    alpha_coef: torch.Tensor,             # (N, 1) or (N, K) self-coupling coefficient

    *,
    kl_max:          float = 100.0,
    eps:             float = 1e-6,
    lambda_beta:     'float | torch.Tensor' = 1.0,   # weight on the belief-coupling (pair) term
    need_sigma_grad: bool  = True,               # False -> skip the sigma pair contraction, return (grad_mu, None)
    irrep_dims:      Optional[List[int]]    = None, # block sizes; maps head h(k) onto coordinate k
    pair_mask:       Optional[torch.Tensor] = None, # destination-energy derivative mask; None means beta is pre-masked
    pair_inv_sigma_t: Optional[torch.Tensor] = None, # precomputed 1 / clamp(sigma_t); None keeps legacy arithmetic
    pair_delta_tq:    Optional[torch.Tensor] = None, # precomputed mu_t - mu_q; paired with pair_inv_sigma_t
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""Diagonal-KL query-side (filtering) gradient (per-head aware).

      grad_mu_i    = m_i a_i (mu_i - mu_p_i)/sigma_p_i + lambda_beta Sum_j beta_ij^(h(k)) (mu_i - mu_t_ij)/sigma_t_ij
      grad_sigma_i = m_i a_i 0.5(1/sigma_p_i - 1/sigma_q_i)
                     + lambda_beta Sum_j beta_ij^(h(k)) 0.5(1/sigma_t_ij - 1/sigma_q_i)

    ``lambda_beta`` (1.0 = pure) scales ONLY the belief-coupling (pair) term, not the alpha
    self-term -- the analytic counterpart of scaling the post-softmax coupling+entropy block in
    ``free_energy`` (the envelope identity makes the pair term equal to d/dtheta of that block at
    beta*, so scaling both by the same factor keeps kernel == oracle).

    ``beta`` is the raw attention weight in its COMPACT per-head form. ``pair_mask`` gates only
    destination-energy derivatives: the one-hop derivative uses ``beta * pair_mask``, while the
    two-hop derivative uses ``(beta.detach() @ beta.detach()) * pair_mask``. Thus a zero or
    saturated intermediate edge may still carry hop mass, but a zero or saturated destination
    energy has the same zero derivative as the scalar clamp. For backward-compatible direct kernel
    calls, ``pair_mask=None`` treats ``beta`` as already masked. The coordinate-resolution weight
    beta_ij^(h(k)) is realized inside ``_pair_contract`` by
    contracting the head axis against a head-shaped VIEW of the pair difference, so the
    (..., N, N, K) ``beta_coord`` broadcast the old kernel consumed is never materialized
    (vram audit 2026-06-10: one full B x N^2 x K tensor saved for backward, for free).

    Self-term saturation mask m_i = 1[0 < D(q_i||p_i) < kl_max]: the oracle differentiates through
    safe_kl_clamp(D, [0, kl_max]), whose gradient is 0 once the raw self-divergence saturates the
    clamp, so the hand kernel zeros its self-term there to stay EXACTLY equal to the filtering
    oracle. The pairwise clamp mask is supplied by the caller from the clamped energy grid.

    The mask is shape-driven by ``alpha_coef``: a per-position coefficient (N,1) carries a summed
    self-divergence clamped as one scalar, so the mask is per-token (N,1); a per-coordinate
    coefficient (N,K) (the ``state_dependent_per_coord`` form) carries a coordinate-wise clamp, so
    the mask is per-coordinate (N,K) and a saturated coordinate is gated WITHOUT killing its
    unsaturated neighbors. The two coincide at K=1.
    """
    if (pair_inv_sigma_t is None) != (pair_delta_tq is None):
        raise ValueError(
            "pair_inv_sigma_t and pair_delta_tq must be provided together"
        )
    sp = sigma_p.clamp(min=eps); sq = sigma_q.clamp(min=eps)
    st = sigma_t.clamp(min=eps) if pair_inv_sigma_t is None else None

    if (mu_q is mu_p and sigma_q is sigma_p and type(mu_q) is torch.Tensor
            and type(sigma_q) is torch.Tensor and type(alpha_coef) is torch.Tensor
            and mu_q.is_floating_point() and sigma_q.is_floating_point()
            and alpha_coef.dtype == mu_q.dtype):
        # Identical tensors give raw KL zero (or NaN), hence the strict mask is zero.
        # Keep coefficient evaluation and ALL self arithmetic below: this preserves
        # self-only gradient edges and NaN propagation without a host finite check.
        self_mask = 0.0
    elif alpha_coef.shape[-1] == 1:                                             # per-position alpha
        raw_self  = _raw_diag_kl(mu_q, sigma_q, mu_p, sigma_p, eps=eps)         # (N,)
        self_mask = ((raw_self > 0.0) & (raw_self < kl_max)).to(mu_q.dtype).unsqueeze(-1)
    else:                                                                       # per-coordinate alpha
        raw_self  = _raw_diag_kl_per_coord(mu_q, sigma_q, mu_p, sigma_p, eps=eps)   # (N, K)
        self_mask = ((raw_self > 0.0) & (raw_self < kl_max)).to(mu_q.dtype)

    beta_pair = beta if pair_mask is None else beta * pair_mask

    if pair_inv_sigma_t is None:
        diff_mu = (mu_q.unsqueeze(-2) - mu_t) / st             # (..., N, N, K)
    else:
        diff_mu = (pair_delta_tq * pair_inv_sigma_t).neg_()     # (..., N, N, K); fresh product
    self_mu  = self_mask * alpha_coef * (mu_q - mu_p) / sp
    pair_mu  = _pair_contract(beta_pair, diff_mu, irrep_dims)
    grad_mu  = self_mu + lambda_beta * pair_mu
    if not need_sigma_grad:
        # skip_belief_sigma_update: the caller freezes sigma, so the (..., N, N, K) sigma pair
        # contraction is dead compute -- skip it entirely rather than discarding its result.
        return grad_mu, None

    if pair_inv_sigma_t is None:
        diff_sig = 0.5 * (1.0 / st - 1.0 / sq.unsqueeze(-2))   # (..., N, N, K)
    else:
        diff_sig = 0.5 * (pair_inv_sigma_t - 1.0 / sq.unsqueeze(-2))
    self_sig = self_mask * alpha_coef * 0.5 * (1.0 / sp - 1.0 / sq)
    pair_sig = _pair_contract(beta_pair, diff_sig, irrep_dims)
    grad_sigma = self_sig + lambda_beta * pair_sig
    return grad_mu, grad_sigma


def _pair_contract(
    beta:       torch.Tensor,             # (..., N, N) single-block OR (..., H, N, N) per-head
    diff:       torch.Tensor,             # (..., N, N, K) pair-difference tensor
    irrep_dims: Optional[List[int]],      # block sizes; None/[K] -> single block
) -> torch.Tensor:                        # (..., N, K) Sum_j beta_ij^(h(k)) diff_ijk
    r"""Pair-term contraction Sum_j beta_ij^(h(k)) diff_ijk WITHOUT the beta_coord broadcast.

    Single block: the legacy ``ij,ijk->ik`` einsum directly. Equal blocks: the per-head beta
    contracts against a (..., N, N, H, d) VIEW of ``diff`` (a reshape of the trailing K axis --
    no copy), the same per-(i, k) sums over j as the old coordinate-broadcast einsum, term for
    term. Unequal blocks (irrep towers): falls back to materializing beta_coord via
    ``_beta_to_coordinate`` (repeat_interleave has no view form).
    """
    if irrep_dims is None or len(irrep_dims) == 1:
        return torch.einsum("...ij,...ijk->...ik", beta, diff)
    if len(set(irrep_dims)) == 1:
        H, d = len(irrep_dims), irrep_dims[0]
        dv = diff.reshape(*diff.shape[:-1], H, d)                 # (..., N, N, H, d) view
        out = torch.einsum("...hij,...ijhd->...ihd", beta, dv)    # (..., N, H, d)
        return out.reshape(*out.shape[:-2], H * d)
    beta_coord = _beta_to_coordinate(beta, irrep_dims, diff.shape[-1])
    return torch.einsum("...ijk,...ijk->...ik", beta_coord, diff)


def _pair_mass(
    w:          torch.Tensor,             # (..., N, N) single-block OR (..., H, N, N) per-head weights
    irrep_dims: Optional[List[int]],      # block sizes; None/[K] -> single block

    K:          int,                      # total belief dimension
) -> torch.Tensor:                        # (..., N, K) Sum_j w_ij^(h(k))
    r"""Row mass Sum_j w_ij^(h(k)) broadcast to coordinate resolution (the mm fusion's pair mass).

    The coordinate map mirrors ``_beta_to_coordinate`` (head h's row sum repeated across its
    d_head coordinates) but sums over the key axis FIRST, so no (..., N, N, K) tensor is built.
    """
    row = w.sum(dim=-1)                                                  # (..., N) or (..., H, N)
    if irrep_dims is None or len(irrep_dims) == 1:
        return row.unsqueeze(-1).expand(*row.shape, K)
    x = row.movedim(-2, -1)                                              # (..., N, H)
    if len(set(irrep_dims)) == 1:                                        # equal blocks: view expand
        d = irrep_dims[0]
        return x.unsqueeze(-1).expand(*x.shape, d).reshape(*x.shape[:-1], x.shape[-1] * d)
    reps = torch.tensor(irrep_dims, device=w.device)                     # unequal blocks: gather
    return torch.repeat_interleave(x, reps, dim=-1)


def uses_kernel_route(
    *,
    renyi_order:               float,
    gradient_mode:             str,
    family:                    str,
    divergence_family:         str,
    include_attention_entropy: bool,
    transport_mode:            str  = "flat",
    decoupled_value_gauge:     bool = False,
    route:                     str  = "gradient",   # "gradient" (belief_gradients) | "mm_exact" (mm_exact_update)
) -> bool:
    r"""Whether the closed-form route is served for ``family`` (else the autograd oracle / a raise).

    The single source of truth for the kernel-coverage predicate, exposed so callers (the
    E-step's unroll-truncation warning, the config-time freeze warning) cannot drift from the
    dispatch below. Capability-driven (audit 2026-08-07), NOT a family-name literal: a family is
    eligible once its registration (``register_kernel``) declares the matching capability for
    ``route`` -- see :class:`KernelRegistration`. ``route="gradient"`` (the default, every
    pre-existing call site) checks ``provides_gradient``; ``route="mm_exact"`` checks
    ``provides_mm_exact``. The two are independent: registering ONE never silently unlocks the
    other, so a family that adds an mm_exact closed form without an analytic gradient kernel (e.g.
    ``gaussian_full``) cannot redirect ``e_step_update='gradient'`` onto an unvalidated path.

    A registration that declares ``needs_mu`` or ``needs_sigma`` excludes both routes (audit
    2026-06-10 F1): a frozen-transport closed form treats transported keys as constants in the
    belief variables, so it would silently omit the derivative of a belief-dependent transport.
    Such registrations route to the autograd oracle, which rebuilds Omega from its differentiation
    leaves through ``omega_builder``.

    ``decoupled_value_gauge`` (RopeTransport.on_value=False) likewise excludes both routes: with the
    attention gauge and value gauge factored apart (GL(K)_attention.tex:1909), beta is the softmax of
    the rotated SCORE energy but the coupling sum uses the un-rotated value energy, so beta is no
    longer that sum's stationary point and the closed-form envelope (which assumes it is) does
    not apply. The oracle differentiates the decoupled F directly and carries the extra d beta/d mu
    term."""
    if route not in ("gradient", "mm_exact"):
        raise ValueError(f"route must be 'gradient' or 'mm_exact', got {route!r}")
    if not has_kernel(family):
        return False
    registration = get_kernel_registration(family)
    capability_ok = (registration.provides_gradient if route == "gradient"
                     else registration.provides_mm_exact)
    if not capability_ok:
        return False
    family_cov_kind = get_family(family).cov_kind
    transport_registration = get_transport_registration(transport_mode)
    return (
        gradient_mode == "filtering"
        and family_cov_kind in registration.covariance_kinds
        and divergence_family == "renyi"
        and abs(renyi_order - 1.0) < 1e-9
        and include_attention_entropy
        and not transport_registration.needs_mu
        and not transport_registration.needs_sigma
        and not decoupled_value_gauge
    )


def _pairwise_stats_reuse_is_sound(
    omega: object,
) -> bool:
    r"""Whether the effective transport representation may reuse diagonal-KL pair statistics."""
    current = omega
    visited: set[int] = set()
    while isinstance(current, RopeTransport):
        identity = id(current)
        if identity in visited:
            return False
        visited.add(identity)
        current = current.base
    return type(current) in (CompactFactoredTransport, FactoredTransport)



def _same_state_masked_alpha_zero(
    mu:      torch.Tensor,             # (..., N, K) shared belief/prior means
    sigma:   torch.Tensor,             # (..., N, K) shared belief/prior variances
    mu_p:    torch.Tensor,
    sigma_p: torch.Tensor,
    b0:      float,
    c0:      float,
    eps:     float,
    kl_max:  float,
    family:            str,
    lambda_alpha_mode: str,
) -> Optional[torch.Tensor]:
    r"""Represent the masked coefficient 0*alpha at q is p without evaluating KL or alpha.

    On the admitted built-in per-coordinate route, the strict self mask is zero.
    Keep a short differentiable zero carrier: its quotient and squared difference
    preserve nonfinite first/second derivatives that a detached zero would erase.
    The clamp is essential at infinite variance. Public/custom coefficient consumers
    still receive the actual alpha; this carrier is private to the masked hand kernel.
    """
    from vfe3.alpha_i import alpha_state_dependent_per_coord, get_alpha
    from vfe3.divergence import get_functional_per_coord, renyi_per_coord
    from vfe3.families.gaussian import DiagonalGaussian

    if (mu is not mu_p or sigma is not sigma_p
            or type(mu) is not torch.Tensor or type(sigma) is not torch.Tensor
            or mu.dtype not in (torch.float32, torch.float64) or sigma.dtype != mu.dtype
            or mu.shape != sigma.shape
            or get_family(family) is not DiagonalGaussian
            or get_kernel(family) is not _diag_kl_filtering_kernel
            or get_alpha(lambda_alpha_mode) is not alpha_state_dependent_per_coord
            or not alpha_is_per_coord(lambda_alpha_mode)
            or get_functional_per_coord("renyi") is not renyi_per_coord
            or type(b0) not in (float, int) or type(c0) not in (float, int)
            or b0 != 1.0 or c0 != 1.0 or eps != 1e-6
            or type(kl_max) not in (float, int) or not 0.0 < kl_max <= 1e6):
        return None
    sp = sigma.clamp(min=eps)
    residual = (mu - mu).square() / sp + sp / sp - 1.0
    return residual.clamp(min=0.0, max=kl_max).nan_to_num(nan=kl_max) * 0.0


def belief_gradients(
    mu:           torch.Tensor,           # (N, K)
    sigma:        torch.Tensor,           # (N, K)
    mu_p:         torch.Tensor,           # (N, K)
    sigma_p:      torch.Tensor,           # (N, K)
    omega:        'torch.Tensor | CompactFactoredTransport | FactoredTransport | RopeTransport | None',

    *,
    tau:          'float | torch.Tensor' = 1.0,
    renyi_order:  float = 1.0,
    kl_max:       float = 100.0,
    eps:          float = 1e-6,
    b0:           float = 1.0,
    c0:           float = 1.0,
    lambda_beta:  'float | torch.Tensor' = 1.0,   # weight on the belief-coupling block (1.0 = pure F)

    include_attention_entropy: bool = True,
    create_graph:              bool = False,   # unroll: oracle returns a differentiable grad (to prior)
    need_sigma_grad:           bool = True,    # False -> kernel skips the sigma pair contraction (grad_sigma None)
    reuse_pairwise_kl_stats:   bool = False,   # reuse graph-live diagonal-KL pair statistics on the canonical route
    gradient_mode:             str  = "filtering",
    family:                    str  = "gaussian_diagonal",
    full_cov_kl_precision:     Optional[str] = None,
    full_cov_kl_backend:       str  = "pairwise",
    divergence_family:         str  = "renyi",
    lambda_alpha_mode:         str  = "constant",
    transport_mode:            str  = "flat",  # registry metadata excludes belief-dependent Omega
    value:                     float = 1.0,

    irrep_dims:                Optional[List[int]]    = None,
    log_prior:                 Optional[torch.Tensor] = None,
    omega_builder:             Optional[Callable]     = None,   # belief-leaf transport rebuild for the oracle
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""Belief gradient: hand kernel for filtering+gaussian_diagonal+KL+canonical+flat, else oracle.

    The closed-form kernel keeps the gradient live (analytic on the live belief), so the unrolled
    E-step signal reaches the prior; ``create_graph=True`` makes the oracle fallback do the same for
    the non-kernel families it serves (else its detached tangent would truncate that signal).

    ``irrep_dims`` (when more than one block) makes attention PER HEAD: the energy/beta carry a
    head axis and the per-coordinate beta the kernel consumes is head h's weight on coordinate k.

    Any transport registration that declares ``needs_mu`` or ``needs_sigma`` routes to the oracle:
    the caller supplies ``omega_builder`` so the oracle rebuilds the belief-dependent transport from
    its differentiation leaves.
    """
    # Value-gauge decoupling (RopeTransport.on_value=False) breaks beta's stationarity for the
    # coupling sum, so the closed-form kernel does not apply -- route to the oracle (which builds the
    # value-gauge coupling energy). The omega object is the source of truth here.
    decoupled_value = isinstance(omega, RopeTransport) and not omega.on_value
    use_kernel = uses_kernel_route(
        renyi_order=renyi_order, gradient_mode=gradient_mode, family=family,
        divergence_family=divergence_family,
        include_attention_entropy=include_attention_entropy,
        transport_mode=transport_mode,
        decoupled_value_gauge=decoupled_value,
        route="gradient",
    )
    if not use_kernel:
        return belief_gradients_autograd(
            mu, sigma, mu_p, sigma_p, omega, tau=tau, renyi_order=renyi_order,
            kl_max=kl_max, eps=eps, b0=b0, c0=c0, value=value, lambda_beta=lambda_beta,
            include_attention_entropy=include_attention_entropy, create_graph=create_graph,
            need_sigma_grad=need_sigma_grad,
            gradient_mode=gradient_mode, family=family,
            full_cov_kl_precision=full_cov_kl_precision,
            full_cov_kl_backend=full_cov_kl_backend,
            divergence_family=divergence_family,
            lambda_alpha_mode=lambda_alpha_mode, irrep_dims=irrep_dims, log_prior=log_prior,
            omega_builder=omega_builder,
        )

    # Reduced-precision island (audit 2026-07-25 F1). Under an outer AMP context the closed-form
    # kernel would otherwise build its transported key moments, energy, beta and derivative at the
    # autocast dtype, while its own reference oracle re-enters fp32 (oracle.py:140). That asymmetry
    # made the two sides of a seam the golden tests pin at fp32 tolerance disagree by ~4e-3 relative,
    # flipped the raw-energy saturation mask that gates the pair derivative, left the structural
    # self-pair energy no longer exactly zero, and silently disabled the reuse_pairwise_kl_stats hoist
    # (whose eligibility test below requires fp32). Re-enter only the kernel body under an explicit
    # fp32 island; the recursion terminates immediately because autocast is disabled in the nested
    # call, and the caller resumes its own autocast context on return. amp_dtype=None is unaffected.
    if torch.is_autocast_enabled(mu.device.type):
        with torch.autocast(device_type=mu.device.type, enabled=False):
            return belief_gradients(
                mu.float(), sigma.float(), mu_p.float(), sigma_p.float(),
                (_transport_to_float(omega) if omega is not None else None),
                tau=(tau.float() if isinstance(tau, torch.Tensor) else tau),
                renyi_order=renyi_order,
                kl_max=kl_max,
                eps=eps,
                b0=b0,
                c0=c0,
                lambda_beta=(lambda_beta.float()
                             if isinstance(lambda_beta, torch.Tensor) else lambda_beta),
                include_attention_entropy=include_attention_entropy,
                create_graph=create_graph,
                need_sigma_grad=need_sigma_grad,
                reuse_pairwise_kl_stats=reuse_pairwise_kl_stats,
                gradient_mode=gradient_mode,
                family=family,
                full_cov_kl_precision=full_cov_kl_precision,
                full_cov_kl_backend=full_cov_kl_backend,
                divergence_family=divergence_family,
                lambda_alpha_mode=lambda_alpha_mode,
                transport_mode=transport_mode,
                value=value,
                irrep_dims=irrep_dims,
                log_prior=(log_prior.float() if log_prior is not None else None),
                omega_builder=omega_builder,
            )

    mu_k, sigma_k = mu.detach(), sigma.detach()
    mu_t = transport_mean(omega, mu_k)                 # rank-agnostic: (N,N,K) or (B,N,N,K)
    # diagonal_out from the BELIEF shape (diagonal iff sigma has the same rank as mu), not the omega
    # rank: a batch-collapsed (N,N,K,K) omega would otherwise misinfer the full
    # sandwich against a batched diagonal sigma. Family-agnostic (covers every diagonal family); a
    # batched dense omega is unaffected (it would infer diagonal anyway). The kernel route is
    # gaussian_diagonal-only, so this is always True here, but the shape check stays robust.
    sigma_t = transport_covariance(omega, sigma_k, diagonal_out=(sigma_k.dim() == mu_k.dim()))
    fam = get_family(family)
    coef = _same_state_masked_alpha_zero(
        mu, sigma, mu_p, sigma_p, b0, c0, eps, kl_max,
        family, lambda_alpha_mode)
    if coef is None:
        sd = self_divergence_for_alpha(fam(mu, sigma), fam(mu_p, sigma_p), alpha=1.0,
            kl_max=kl_max, eps=eps, divergence_family=divergence_family,
            lambda_alpha_mode=lambda_alpha_mode)
    pair_stats = None
    # A raw dense effective base carries no structural same-frame certificate, including through a
    # RoPE wrapper. On the single-block flat route, its numerically composed self links can make the
    # float64 statistics reduction strictly positive where the generic float32 energy and derivative
    # mask are exactly zero. Fail closed there; factored effective bases retain automatic reuse.
    if (reuse_pairwise_kl_stats
            and _pairwise_stats_reuse_is_sound(omega)
            and all(tensor.dtype == torch.float32 for tensor in (mu, sigma, mu_t, sigma_t))):
        pair_stats = diagonal_kl_pair_stats(
            mu,
            sigma,
            mu_t,
            sigma_t,
            kl_max=kl_max,
            eps=eps,
            irrep_dims=irrep_dims,
        )
    if pair_stats is None:
        energy = pairwise_energy(fam(mu, sigma), fam.from_transported(mu_t, sigma_t, sigma_k),
                                 alpha=1.0, kl_max=kl_max, eps=eps,
                                 divergence_family=divergence_family, irrep_dims=irrep_dims)
    else:
        energy = pair_stats.energy
    beta = attention_weights(energy, tau=tau, log_prior=log_prior)   # (N,N) or (H,N,N)
    # Pair-term saturation mask (audit 2026-06-09 P7): the oracle differentiates
    # beta_ij * clamp(E_ij, [0, kl_max]), whose pair gradient VANISHES wherever the raw energy
    # saturates the clamp (the clamp emits the exact bounds, so the equality tests are robust).
    # Without the mask a fully saturated row softmaxes to uniform beta over constant energies and
    # the kernel's transported pair term deviates from autograd-of-F by orders of magnitude.
    # Mirrors the self-term mask inside the kernel body; beta itself (the weights) is unchanged.
    if pair_stats is None:
        pair_mask = ((energy > 0.0) & (energy < kl_max)).to(beta.dtype)
    else:
        pair_mask = pair_stats.pair_mask.to(beta.dtype)
    if coef is None:
        coef = alpha_gradient_coefficient(sd, value=value, b0=b0, c0=c0, mode=lambda_alpha_mode)
    if not alpha_is_per_coord(lambda_alpha_mode):
        coef = coef.unsqueeze(-1)                 # (N,) -> (N,1) per-position broadcast; per-coord sd is already (N,K)
    # The raw beta stays in its COMPACT per-head form; the kernel's _pair_contract realizes
    # beta_ij^(h(k)) against a head-shaped view of the pair difference, so the (..., N, N, K)
    # beta_coord broadcast is never materialized (vram audit 2026-06-10).
    kernel_args   = (mu, sigma, mu_p, sigma_p, mu_t, sigma_t, beta, coef)
    kernel_kwargs = dict(kl_max=kl_max, eps=eps, lambda_beta=lambda_beta,
                         need_sigma_grad=need_sigma_grad,
                         irrep_dims=irrep_dims, pair_mask=pair_mask)
    if pair_stats is not None:
        kernel_kwargs["pair_inv_sigma_t"] = pair_stats.inv_sigma_t
        kernel_kwargs["pair_delta_tq"] = pair_stats.delta_tq
    return get_kernel(family)(*kernel_args, **kernel_kwargs)


def mm_exact_update(
    mu:           torch.Tensor,           # (N, K) or (B, N, K) belief means (query side, live)
    sigma:        torch.Tensor,           # (N, K) or (B, N, K) belief variances (live)
    mu_p:         torch.Tensor,           # (N, K) or (B, N, K) prior means
    sigma_p:      torch.Tensor,           # (N, K) or (B, N, K) prior variances
    omega:        'torch.Tensor | CompactFactoredTransport | FactoredTransport | RopeTransport',

    *,
    tau:          'float | torch.Tensor' = 1.0,
    b0:           'float | torch.Tensor' = 1.0,
    c0:           'float | torch.Tensor' = 1.0,
    lambda_beta:  'float | torch.Tensor' = 1.0,   # weight on the belief-coupling (pair) block

    kl_max:        float = 100.0,
    eps:           float = 1e-6,
    value:         float = 1.0,

    lambda_alpha_mode: str = "constant",
    family:            str = "gaussian_diagonal",
    full_cov_kl_precision: Optional[str] = None,
    full_cov_kl_backend:       str  = "pairwise",
    divergence_family: str = "renyi",

    need_sigma_update:       bool = True,       # False -> omit sigma fusion and return the input sigma exactly
    reuse_pairwise_kl_stats: bool = False,      # reuse graph-live diagonal-KL pair statistics on the canonical route

    irrep_dims:    Optional[List[int]]    = None,
    log_prior:     Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:          # (mu_star, sigma_star), each (..., N, K)
    r"""Closed-form coordinate minimizer of the beta-frozen, strict-pair-masked diagonal-KL
    surrogate (NOT a majorizer of the canonical frozen-attention objective -- the strict pair
    mask below excludes the structural E_ii = 0 self-pairs; see README "mm_exact").

    At frozen attention weights beta (and frozen transported key moments -- the filtering split),
    the per-coordinate stationary point of

        F_hat = Sum_i m_i a_i KL(q_i || p_i)
              + Sum_ij w_ij^(h(k)) KL(q_i || Omega_ij q_j),
        w_ij  = m_ij lambda_beta beta_ij,
        m_ij  = 1[0 < E_ij < kl_max],

    obtained by zeroing the hand kernel's grad_mu / grad_sigma expressions, is the precision fusion

        mu*_ik    = ( m_i a_i mu_p,ik / sigma_p,ik + Sum_j w_ij^(h(k)) mu_t,ijk / sigma_t,ijk ) / P_ik,
        sigma*_ik = ( m_i a_i + Sum_j w_ij^(h(k)) ) / P_ik,
        P_ik      =   m_i a_i / sigma_p,ik + Sum_j w_ij^(h(k)) / sigma_t,ijk,

    with m_i the self-term saturation mask (UPPER gate only, 1[D < kl_max]; 2026-07-10 fix --
    see the comment at the mask) and a_i the same alpha coefficient the kernel
    uses at the CURRENT point (state-dependent envelope frozen). mu* is sigma_q-independent (the
    kernel's grad_mu carries no sigma_q), so the pair is jointly exact in one evaluation. beta and
    the transported moments are the SAME intermediates ``belief_gradients`` builds. The one-hop
    derivative uses ``beta * pair_mask``; the two-hop derivative composes raw detached beta first,
    then applies ``pair_mask`` to the destination energy, matching the scalar functional. Thus the
    fusion is the exact minimizer of the objective the kernel's gradient descends.
    The returned sigma* carries only the ``eps`` floor; the caller applies its damping and the
    sigma_max cap. The graph stays LIVE through (mu*, sigma*) (analytic in mu/sigma via beta, sd),
    matching the kernel's unroll behavior; the caller detaches for straight_through.
    """
    # Reduced-precision island (audit 2026-07-25 F1), for the same reason as ``belief_gradients``
    # above: this closed-form minimizer reuses the same transported moments, energy, beta and
    # saturation mask, and a reduced-dtype pair_mask changes WHICH pairs enter the precision fusion,
    # not merely its rounding. Recursion terminates at once (autocast disabled in the nested call).
    if torch.is_autocast_enabled(mu.device.type):
        with torch.autocast(device_type=mu.device.type, enabled=False):
            return mm_exact_update(
                mu.float(), sigma.float(), mu_p.float(), sigma_p.float(),
                (_transport_to_float(omega) if omega is not None else None),
                tau=(tau.float() if isinstance(tau, torch.Tensor) else tau),
                b0=(b0.float() if isinstance(b0, torch.Tensor) else b0),
                c0=(c0.float() if isinstance(c0, torch.Tensor) else c0),
                lambda_beta=(lambda_beta.float()
                             if isinstance(lambda_beta, torch.Tensor) else lambda_beta),
                kl_max=kl_max,
                eps=eps,
                value=value,
                lambda_alpha_mode=lambda_alpha_mode,
                family=family,
                full_cov_kl_precision=full_cov_kl_precision,
                full_cov_kl_backend=full_cov_kl_backend,
                divergence_family=divergence_family,
                need_sigma_update=need_sigma_update,
                reuse_pairwise_kl_stats=reuse_pairwise_kl_stats,
                irrep_dims=irrep_dims,
                log_prior=(log_prior.float() if log_prior is not None else None),
            )

    # Route predicate (audit 2026-07-26 D-02; capability split 2026-08-07). The fusion below is the
    # stationary point of a Gaussian-KL kernel: alpha = 1 is hardcoded (there is no renyi_order
    # argument at all), the precision expressions are the hand-written Gaussian-KL ones, and the
    # grid is built from ``pairwise_energy`` rather than ``fam.coupling_energy`` -- so a family whose
    # coupling energy is NOT that truncated form silently received the wrong grid
    # (``gaussian_diagonal_exact`` returned a mu* bit-identical to ``gaussian_diagonal``), and a
    # non-Renyi divergence drove beta and pair_mask before being fused with Gaussian-KL algebra.
    # ``route="mm_exact"`` checks the mm_exact CAPABILITY specifically (``KernelRegistration.
    # provides_mm_exact``), decoupled from the sibling ``belief_gradients`` gradient-kernel route at
    # :479 -- a family (e.g. ``gaussian_full``) can register one without the other. There is no
    # oracle for a closed-form minimizer, so this seam fails closed instead. All three live callers
    # (e_step.py, config.py, here) gate on the SAME predicate, so no shipped config changes. The
    # gradient_mode / entropy / renyi_order arguments are the values this derivation assumes rather
    # than caller inputs; ``transport_mode`` is not among this seam's arguments, so the
    # belief-dependent-transport exclusion stays with the callers that own that name.
    if not uses_kernel_route(
            renyi_order=1.0, gradient_mode="filtering", family=family,
            divergence_family=divergence_family, include_attention_entropy=True,
            decoupled_value_gauge=(isinstance(omega, RopeTransport) and not omega.on_value),
            route="mm_exact"):
        raise ValueError(
            f"mm_exact_update is the closed-form minimizer of a Gaussian-KL kernel and does not "
            f"cover family={family!r}, divergence_family={divergence_family!r} (coupled value gauge "
            f"required, and family must register the mm_exact capability -- registered mm_exact "
            f"families: {mm_exact_families()}); its precision fusion would score one objective and "
            f"fuse another. Use e_step_update='gradient' for this configuration."
        )

    mu_k, sigma_k = mu.detach(), sigma.detach()
    mu_t    = transport_mean(omega, mu_k)                        # (..., N, N, K) transported keys
    sigma_t = transport_covariance(omega, sigma_k, diagonal_out=(sigma_k.dim() == mu_k.dim()))
    fam = get_family(family)
    sd = self_divergence_for_alpha(
                                   _family_instance(mu, sigma, fam, full_cov_kl_precision),
                                   _family_instance(mu_p, sigma_p, fam, full_cov_kl_precision), alpha=1.0, kl_max=kl_max, eps=eps,
                                   divergence_family=divergence_family, lambda_alpha_mode=lambda_alpha_mode)
    pair_stats = None
    decoupled_value = isinstance(omega, RopeTransport) and not omega.on_value
    if (reuse_pairwise_kl_stats
            and _pairwise_stats_reuse_is_sound(omega)
            and family == "gaussian_diagonal"
            and divergence_family == "renyi"
            and not decoupled_value
            and all(tensor.dtype == torch.float32 for tensor in (mu, sigma, mu_t, sigma_t))):
        pair_stats = diagonal_kl_pair_stats(
            mu,
            sigma,
            mu_t,
            sigma_t,
            kl_max=kl_max,
            eps=eps,
            irrep_dims=irrep_dims,
        )
    if fam is FullGaussian and full_cov_kl_backend in ("common_frame", "common_frame_factored"):
        # The full-covariance fusion still consumes transported moments, but its attention
        # score follows the configured coupling backend just like the gradient oracle.
        energy = fam.coupling_energy(
            mu, sigma, mu_k, sigma_k, omega,
            alpha=1.0, kl_max=kl_max, eps=eps,
            divergence_family=divergence_family, irrep_dims=irrep_dims,
            precision_policy=full_cov_kl_precision, score_backend=full_cov_kl_backend,
            diagonal_out=False,
        )
    elif pair_stats is None:
        key = (fam.from_transported(mu_t, sigma_t, sigma_k, _precision_policy=full_cov_kl_precision)
               if fam is FullGaussian and full_cov_kl_precision is not None
               else fam.from_transported(mu_t, sigma_t, sigma_k))
        energy = pairwise_energy(_family_instance(mu, sigma, fam, full_cov_kl_precision), key,
                                 alpha=1.0, kl_max=kl_max, eps=eps,
                                 divergence_family=divergence_family, irrep_dims=irrep_dims)
    else:
        energy = pair_stats.energy
    beta = attention_weights(energy, tau=tau, log_prior=log_prior)   # (..., N, N) or (..., H, N, N)
    if pair_stats is None:
        pair_mask = ((energy > 0.0) & (energy < kl_max)).to(beta.dtype)
    else:
        pair_mask = pair_stats.pair_mask.to(beta.dtype)
    coef = alpha_gradient_coefficient(sd, value=value, b0=b0, c0=c0, mode=lambda_alpha_mode)
    if not alpha_is_per_coord(lambda_alpha_mode):
        coef = coef.unsqueeze(-1)                                    # (N,) -> (N,1)

    # Self-term mask: UPPER gate ONLY (2026-07-10 fix). The gradient kernel keeps the lower
    # gate (raw_self > 0) because it mirrors the clamp's zero derivative and dD = 0 at q == p
    # anyway; here the mask is reused as the prior PRECISION WEIGHT in the fusion, and the
    # model enters the E-step with q0 == p EXACTLY (forward_beliefs anchors the belief to the
    # prior), so raw_self == 0 for every token: a lower gate zeroes the prior anchor and snaps
    # mu* to the self-excluded neighbor consensus on the first (often only) inner iteration.
    # The correct MM weight at D = 0 is the envelope alpha* = c0/(b0+0). The upper gate stays:
    # the kl_max clamp flattens the objective in a neighborhood there.
    # docs/2026-07-10-mm-exact-prior-anchor-fix.md
    if fam.cov_kind == "full":
        # Full covariance: reuse the already-computed CLAMPED self-divergence ``sd`` (built above,
        # generic over family) rather than a second family-specific raw-KL implementation.
        # ``safe_kl_clamp`` (families/base.py) maps exactly {raw >= kl_max, NaN, +inf} -> kl_max and
        # leaves every other value untouched, so ``sd < kl_max`` is IDENTICAL to the diagonal
        # branch's ``raw_self < kl_max`` upper gate. ``lambda_alpha_mode='state_dependent_per_coord'``
        # is unreachable here: it needs a per-coordinate self-divergence, which
        # ``self_divergence_for_alpha`` (building ``sd`` above) already raises for a non-diagonal
        # family (free_energy.py ``self_divergence_per_coord``; config.py independently rejects the
        # combination), so ``coef``/``sd`` are always per-position ((..., N, 1) / (..., N)) here.
        self_mask = (sd < kl_max).to(mu.dtype).unsqueeze(-1)         # UPPER gate only (see comment above)
    elif coef.shape[-1] == 1:                                        # per-position alpha
        raw_self  = _raw_diag_kl(mu, sigma, mu_p, sigma_p, eps=eps)
        self_mask = (raw_self < kl_max).to(mu.dtype).unsqueeze(-1)   # UPPER gate only (see comment above)
    else:                                                            # per-coordinate alpha
        raw_self  = _raw_diag_kl_per_coord(mu, sigma, mu_p, sigma_p, eps=eps)
        self_mask = (raw_self < kl_max).to(mu.dtype)                 # UPPER gate only (see comment above)
    a = self_mask * coef                                             # m_i a_i, (..., N, 1) or (..., N, K)

    w = lambda_beta * (beta * pair_mask)

    if fam.cov_kind == "full":
        # Dense K x K precision fusion (WAVE4 audit derivation); see ``_mm_exact_full_covariance``.
        # Returns directly: the scalar-diagonal algebra below does not apply to a (..., K, K) sigma.
        return _mm_exact_full_covariance(
            mu=mu, sigma=sigma, mu_p=mu_p, sigma_p=sigma_p, mu_t=mu_t, sigma_t=sigma_t,
            a=a, w=w, eps=eps, irrep_dims=irrep_dims,
            need_sigma_update=need_sigma_update,
        )

    sp = sigma_p.clamp(min=eps)
    K = mu.shape[-1]
    prior_prec = a / sp                                              # (..., N, K) m a / sigma_p
    if pair_stats is None:
        st = sigma_t.clamp(min=eps)
        pair_prec = _pair_contract(w, 1.0 / st, irrep_dims)          # (..., N, K) Sum_j w / sigma_t
        pair_mean = _pair_contract(w, mu_t / st, irrep_dims)         # (..., N, K) Sum_j w mu_t / sigma_t
    else:
        pair_prec = _pair_contract(w, pair_stats.inv_sigma_t, irrep_dims)
        pair_mean = _pair_contract(w, mu_t * pair_stats.inv_sigma_t, irrep_dims)
    prec = prior_prec + pair_prec                                   # pre-clamp fused precision
    numerator = a * mu_p / sp + pair_mean                           # (..., N, K)
    P = prec.clamp(min=eps)                                         # eps guards the all-saturated row
    mu_star = numerator / P
    # m12: on a fully saturated row (a==0 and all w==0) prec floors to 0; dividing the zero numerator by
    # eps snapped the belief to (0, eps), the OPPOSITE of the gradient route (which stays put). Keep the
    # live belief where prec floors, matching the gradient route and preserving graph liveness.
    degenerate = prec <= eps
    mu_star = torch.where(degenerate, mu, mu_star)
    if not need_sigma_update:
        return mu_star, sigma

    pair_mass  = _pair_mass(w, irrep_dims, K)                        # (..., N, K) Sum_j w
    sigma_star = ((a + pair_mass) / P).clamp(min=eps)
    sigma_star = torch.where(degenerate, sigma, sigma_star)
    return mu_star, sigma_star


def _mm_exact_dense_fusion(
    mu:                torch.Tensor,           # (..., N, d) live belief means (pass-through on a degenerate row)
    sigma:             torch.Tensor,           # (..., N, d, d) live belief covariance (pass-through on a degenerate row)
    mu_p:              torch.Tensor,           # (..., N, d) prior means
    sigma_p:           torch.Tensor,           # (..., N, d, d) prior covariance
    mu_t:              torch.Tensor,           # (..., N, N, d) transported key means
    sigma_t:           torch.Tensor,           # (..., N, N, d, d) transported key covariance (dense sandwich)
    a:                 torch.Tensor,           # (..., N, 1) self/prior coupling weight m_i a_i
    w:                 torch.Tensor,           # (..., N, N) frozen pair weights for THIS block

    *,
    eps:               float,
    need_sigma_update: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # (mu_star, sigma_star, Cholesky-success mask), (..., N, d) / (..., N, d, d) / (..., N)
    r"""Dense d x d precision-fusion minimizer, no irrep/head awareness (WAVE4 audit derivation).

    The full-covariance analogue of ``mm_exact_update``'s diagonal precision fusion, at the SAME
    frozen (a, w) coupling weights the diagonal branch consumes:

        Lambda*      =  a_i Sigma_p_i^{-1}       +  Sum_j w_ij (Omega_ij Sigma_j Omega_ij^T)^{-1},
        Lambda* mu*  =  a_i Sigma_p_i^{-1} mu_p_i +  Sum_j w_ij (Omega_ij Sigma_j Omega_ij^T)^{-1} mu_t,ij,
        Sigma*       =  (a_i + Sum_j w_ij) Lambda*^{-1},

    i.e. the diagonal kernel's scalar ``1/sigma`` divisions become d x d Cholesky-based precisions
    and solves (never an explicit ``torch.linalg.inv``). Per-pair precisions are built with
    ``FullGaussian.mean_fisher_precision`` -- the SAME Cholesky-then-``cholesky_inverse`` route (with
    the SAME ``pinv`` degenerate-factorization fallback) the rest of the full-covariance code already
    uses for "get a precision from a covariance" (``natural_gradient``'s preconditioner, the
    diagnostics in ``metrics.py``), rather than a new convention.

    d is EITHER the full K (the single-irrep-block route, ``w`` shape ``(..., N, N)``) OR one gauge
    head's own ``d_head`` (the multi-head per-block route, called once per head by
    ``_mm_exact_full_covariance`` with every tensor already sliced to that head's coordinates and
    that head's OWN pair weight ``w_ij^(h)``) -- this function has no irrep concept at all, so both
    callers share one implementation and one set of numerics.

    Reused Choleskys: unlike the diagonal branch (whose ``pair_stats`` hoist can share graph-live
    diagonal-KL statistics with ``belief_gradients``), this fusion factors ``sigma_p``/``sigma_t``
    FRESH rather than threading the factor ``FullGaussian.renyi_closed_form`` already computed while
    scoring the self/pair energy above -- the existing factor lives inside a private helper
    (``_full_gaussian_kl_terms``) with three OTHER callers, and reaching into it here would either
    change that shared helper's signature (a wider, riskier edit) or duplicate its algebra. Correct,
    not maximally efficient; a documented opportunity, not a defect.
    """
    fam = get_family("gaussian_full")
    sp_prec = fam.mean_fisher_precision(sigma_p, eps=eps)              # (..., N, d, d)     Sigma_p^{-1}
    st_prec = fam.mean_fisher_precision(sigma_t, eps=eps)              # (..., N, N, d, d)  Sigma_t_ij^{-1}

    prior_prec_term = a.unsqueeze(-1) * sp_prec                                       # (..., N, d, d)
    prior_num_term  = a * torch.einsum("...nkl,...nl->...nk", sp_prec, mu_p)          # (..., N, d)

    pair_prec_term = torch.einsum("...ij,...ijkl->...ikl", w, st_prec)                # (..., N, d, d)
    st_prec_mu     = torch.einsum("...ijkl,...ijl->...ijk", st_prec, mu_t)            # (..., N, N, d)
    pair_num_term  = torch.einsum("...ij,...ijk->...ik", w, st_prec_mu)               # (..., N, d)

    prec      = prior_prec_term + pair_prec_term                       # Lambda*, (..., N, d, d)
    numerator = prior_num_term + pair_num_term                         # (..., N, d)

    mass = a.squeeze(-1) + w.sum(dim=-1)                                # (..., N) total precision mass
    # m12 analogue (see the diagonal branch above): a fully saturated row (a==0, all w==0) leaves
    # Lambda* at (numerically) zero; keep the live belief there instead of solving against a
    # near-singular matrix, matching the diagonal path's "stay put" degenerate convention.
    degenerate = mass <= eps

    L_star, ok_star = safe_cholesky(prec, eps=eps, rounds=5)
    # cholesky_ex leaves a finite PARTIAL factor on failure. Never feed it to a solve/inverse: the
    # value can look finite while representing no factorization. Identity is a safe temporary only;
    # the final selection below retains the incoming belief exactly for every failed row.
    eye = torch.eye(prec.shape[-1], dtype=prec.dtype, device=prec.device)
    L_star_safe = torch.where(ok_star[..., None, None], L_star, eye.expand_as(L_star))
    invalid = degenerate | ~ok_star
    mu_candidate = torch.cholesky_solve(numerator.unsqueeze(-1), L_star_safe).squeeze(-1)
    mu_star = torch.where(invalid.unsqueeze(-1), mu, mu_candidate)

    if not need_sigma_update:
        return mu_star, sigma, ok_star

    lambda_inv = torch.cholesky_inverse(L_star_safe)                   # (..., N, d, d)  Lambda*^{-1}
    sigma_star = mass.unsqueeze(-1).unsqueeze(-1) * lambda_inv
    sigma_star = torch.where(invalid.unsqueeze(-1).unsqueeze(-1), sigma, sigma_star)
    return mu_star, sigma_star, ok_star


_BLOCK_DIAGONAL_RTOL: float = 1e-6   # calibrated against fp32 accumulation noise on a genuinely diagonal input


def _block_diagonal_relative_residual(
    matrix:     torch.Tensor,             # (..., K, K)
    irrep_dims: List[int],
) -> torch.Tensor:                        # () scalar: max|off-block| / max|on-block|, over every batch element
    r"""Cross-block residual of ``matrix`` under the ``irrep_dims`` partition, relative to its own
    on-block scale. The EXECUTED check ``_verify_prior_block_diagonal`` gates the per-head mm_exact
    route on, rather than trusting the gauge-group name."""
    K = matrix.shape[-1]
    on_block = torch.zeros(K, K, dtype=torch.bool, device=matrix.device)
    start = 0
    for d in irrep_dims:
        on_block[start:start + d, start:start + d] = True
        start += d
    on_scale = matrix[..., on_block].abs().amax()
    off_scale = matrix[..., ~on_block].abs().amax() if bool((~on_block).any()) else matrix.new_zeros(())
    return off_scale / on_scale.clamp_min(torch.finfo(matrix.dtype).tiny)


def _verify_prior_block_diagonal(sigma_p: torch.Tensor, irrep_dims: List[int]) -> None:
    r"""Raise unless ``sigma_p`` is (numerically) block-diagonal under ``irrep_dims``.

    EXECUTED, not name-based (2026-08-07 gauge-audit reversal of the earlier blanket multi-head
    ValueError): the per-head fusion below is exact precisely because, and only because, the
    self/prior term's full-matrix inverse ``Sigma_p^{-1}`` is block-diagonal whenever ``Sigma_p``
    is -- true for the encode prior (``prior_bank._encode_prior_sigma``: ``torch.diag_embed(...)``,
    trivially block-diagonal) but NOT for every prior a caller may hand in (a genuinely dense
    packed Cholesky covariance carries no block-zero constraint). Gating on the gauge-group name
    alone would silently mis-fuse that combination; this checks the actual tensor every call.
    """
    residual = _block_diagonal_relative_residual(sigma_p, irrep_dims)
    if bool(residual > _BLOCK_DIAGONAL_RTOL):
        raise ValueError(
            "mm_exact_update's per-head full-covariance fusion "
            f"(irrep_dims={irrep_dims!r}, {len(irrep_dims)} blocks) requires a BLOCK-DIAGONAL "
            "sigma_p: the self/prior term's stationarity condition forces Sigma* off-block entries "
            "to equal Sigma_p^{-1}'s off-block entries (both zero only when Sigma_p is "
            "block-diagonal), while the coupling term never reads Sigma's off-block entries at all "
            "(FullGaussian.coupling_energy slices each head's OWN diagonal submatrix). Measured "
            f"cross-block/on-block residual {float(residual):.3e} > {_BLOCK_DIAGONAL_RTOL:.0e}. "
            "This holds for the encode prior (diag_embed -> diagonal, hence block-diagonal) but NOT "
            "for a genuinely dense packed-Cholesky covariance. Use a block-diagonal prior, a "
            "single-block gauge_group, or e_step_update='gradient'."
        )


def _mm_exact_full_covariance(
    *,
    mu:                torch.Tensor,           # (..., N, K) live belief means (pass-through on a degenerate row)
    sigma:             torch.Tensor,           # (..., N, K, K) live belief covariance (pass-through on a degenerate row)
    mu_p:              torch.Tensor,           # (..., N, K) prior means
    sigma_p:           torch.Tensor,           # (..., N, K, K) prior covariance
    mu_t:              torch.Tensor,           # (..., N, N, K) transported key means
    sigma_t:           torch.Tensor,           # (..., N, N, K, K) transported key covariance (dense sandwich)
    a:                 torch.Tensor,           # (..., N, 1) self/prior coupling weight m_i a_i
    w:                 torch.Tensor,           # (..., N, N) single-block OR (..., H, N, N) per-head pair weights
    eps:               float,
    irrep_dims:        Optional[List[int]],
    need_sigma_update: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:        # (mu_star, sigma_star), each (..., N, K) / (..., N, K, K)
    r"""K x K precision-fusion minimizer for a full-covariance family (WAVE4 audit derivation).

    SINGLE IRREP BLOCK (``irrep_dims`` None or length 1): the dense ``_mm_exact_dense_fusion`` runs
    directly on the whole K.

    MULTI-HEAD (length > 1, e.g. ``gauge_group='block_glk'``): PROVEN exact whenever ``sigma_p`` is
    block-diagonal (2026-08-07 gauge-audit reversal of the earlier blanket ValueError here -- the
    original probe used a DENSE random ``sigma_p``, an input the live encode path never produces;
    measured on the CORRECT construction, ``torch.diag_embed`` as ``prior_bank._encode_prior_sigma``
    builds it, the cross-block ``grad_sigma`` is exactly 0.0 across 6 seeds and H in {2, 3}).
    Derivation: ``FullGaussian.coupling_energy`` scores each head h off ONLY its own diagonal
    K_h x K_h submatrix of Sigma (``.block()``), so as a function of the FULL dense Sigma_i the
    coupling term has EXACTLY ZERO partial derivative with respect to any off-block entry, for every
    Sigma_i (not just a block-diagonal one -- verified: perturbing a dense, non-block-diagonal
    Sigma's off-block entries left grad_sigma's cross-block components matching
    ``-0.5 a (Sigma^{-1})^{(h,h')}`` from the SELF term alone, to the last float). So the total
    gradient's off-block component is exactly ``0.5 a (Sigma_p^{-1} - Sigma*^{-1})^{(h,h')}``; zeroing
    it forces ``(Sigma*^{-1})^{(h,h')} = (Sigma_p^{-1})^{(h,h')} = 0`` whenever Sigma_p is
    block-diagonal (an SPD matrix's inverse is block-diagonal under a partition iff the matrix
    itself is), hence Sigma* itself is exactly block-diagonal, and each diagonal block's own
    stationarity equation is the per-head fusion below -- an EXACT decomposition, not an ansatz.
    ``w`` already carries the per-head axis here (``pairwise_energy(..., irrep_dims=...)`` returns
    ``(..., H, N, N)`` for H > 1, unconditionally), so each head keeps its OWN pair weight w_ij^(h);
    the outer preamble does not need to (and does not) collapse it to one shared weight.

    Any group whose Sigma is genuinely NOT block-diagonal (a dense packed Cholesky covariance, per
    the gauge audit) fails the EXECUTED ``_verify_prior_block_diagonal`` check below rather than
    silently fusing the wrong objective.
    """
    if irrep_dims is None or len(irrep_dims) == 1:
        mu_star, sigma_star, ok_star = _mm_exact_dense_fusion(
            mu, sigma, mu_p, sigma_p, mu_t, sigma_t, a, w,
            eps=eps, need_sigma_update=need_sigma_update,
        )
        _count_mm_cholesky_fallback(ok_star)
        return mu_star, sigma_star
    _verify_prior_block_diagonal(sigma_p, irrep_dims)
    mu_parts:    List[torch.Tensor] = []
    sigma_parts: List[torch.Tensor] = []
    factor_masks: List[torch.Tensor] = []
    start = 0
    for h, d in enumerate(irrep_dims):
        end = start + d
        mu_star_h, sigma_star_h, ok_star_h = _mm_exact_dense_fusion(
            mu[..., start:end], sigma[..., start:end, start:end],
            mu_p[..., start:end], sigma_p[..., start:end, start:end],
            mu_t[..., start:end], sigma_t[..., start:end, start:end],
            a, w[..., h, :, :],
            eps=eps, need_sigma_update=need_sigma_update,
        )
        mu_parts.append(mu_star_h)
        sigma_parts.append(sigma_star_h)
        factor_masks.append(ok_star_h)
        start = end
    mu_star = torch.cat(mu_parts, dim=-1)                               # (..., N, K)
    # Block-diagonal assembly: pad each head's (..., N, d_h, d_h) block with exact zeros on every
    # other head's rows/cols, then sum (the padded blocks never overlap, so sum == placement). Avoids
    # in-place slice writes into a fresh zeros tensor, which is autograd-safe but less obviously so.
    K = mu.shape[-1]
    sigma_star = sigma.new_zeros(sigma.shape)
    start = 0
    for sigma_star_h, d in zip(sigma_parts, irrep_dims):
        end = start + d
        pad = (start, K - end, start, K - end)                          # (left, right, top, bottom)
        sigma_star = sigma_star + torch.nn.functional.pad(sigma_star_h, pad)
        start = end
    ok_star = torch.stack(factor_masks, dim=0).all(dim=0)
    _count_mm_cholesky_fallback(ok_star)
    mu_star = torch.where(ok_star.unsqueeze(-1), mu_star, mu)
    sigma_star = torch.where(ok_star.unsqueeze(-1).unsqueeze(-1), sigma_star, sigma)
    return mu_star, sigma_star


def mm_damped_precision_blend_full(
    mu_old:     torch.Tensor,      # (..., N, K) current belief mean
    sigma_old:  torch.Tensor,      # (..., N, K, K) current belief covariance
    mu_star:    torch.Tensor,      # (..., N, K) mm_exact_update's exact-minimizer mean
    sigma_star: torch.Tensor,      # (..., N, K, K) mm_exact_update's exact-minimizer covariance
    eta:        float,             # mm_damping in (0, 1]

    *,
    eps:        float,
    sigma_max:  Optional[float],
) -> Tuple[torch.Tensor, torch.Tensor]:        # (mu, sigma), each (..., N, K) / (..., N, K, K)
    r"""Full-covariance analogue of the diagonal mm_damping blend (``e_step.py``'s mm_exact branch).

    A mirror-descent step on the Gaussian natural parameters, generalizing the diagonal path's scalar
    ``1/sigma`` to the K x K precision:

        Lambda     <- (1 - eta) Lambda_old + eta Lambda_star,
        Lambda mu  <- (1 - eta) Lambda_old mu_old + eta Lambda_star mu_star,

    with ``Lambda = Sigma^{-1}``. ``eta=1`` reproduces ``mu_star, sigma_star`` (modulo the extra
    factorization round-trip); ``eta=0`` leaves ``(mu_old, sigma_old)`` unchanged -- both match the
    diagonal path's convention at the two damping endpoints. The blended covariance is projected to
    the eigenvalue interval ``[eps, sigma_max]`` with the SAME spectral map ``retract_spd_full`` uses
    for its own ceiling (a natural-parameter blend of two matrices already inside the interval is not
    itself guaranteed to stay there), then hardened with the same public-SPD certificate.
    """
    from vfe3.geometry.retraction import _certify_public_spd, _symmetric_spectral_map
    L_old, ok_old = safe_cholesky(sigma_old, eps=eps, rounds=5)
    L_star, ok_star = safe_cholesky(sigma_star, eps=eps, rounds=5)
    eye = torch.eye(sigma_old.shape[-1], dtype=sigma_old.dtype, device=sigma_old.device)
    L_old_safe = torch.where(ok_old[..., None, None], L_old, eye.expand_as(L_old))
    L_star_safe = torch.where(ok_star[..., None, None], L_star, eye.expand_as(L_star))
    lam_old  = torch.cholesky_inverse(L_old_safe)
    lam_star = torch.cholesky_inverse(L_star_safe)
    lam_new = (1.0 - eta) * lam_old + eta * lam_star
    numerator = ((1.0 - eta) * torch.einsum("...kl,...l->...k", lam_old, mu_old)
                 + eta * torch.einsum("...kl,...l->...k", lam_star, mu_star))
    L_new, ok_new = safe_cholesky(lam_new, eps=eps, rounds=5)
    L_new_safe = torch.where(ok_new[..., None, None], L_new, eye.expand_as(L_new))
    ok = ok_old & ok_star & ok_new
    _count_mm_cholesky_fallback(ok)
    mu_candidate = torch.cholesky_solve(numerator.unsqueeze(-1), L_new_safe).squeeze(-1)
    sigma_candidate = torch.cholesky_inverse(L_new_safe)
    sigma_candidate = 0.5 * (sigma_candidate + sigma_candidate.transpose(-1, -2))
    sigma_candidate = _symmetric_spectral_map(sigma_candidate, "project", lower=eps, upper=sigma_max)
    sigma_candidate = _certify_public_spd(sigma_candidate, eps=eps, sigma_max=sigma_max)
    mu = torch.where(ok.unsqueeze(-1), mu_candidate, mu_old)
    sigma = torch.where(ok.unsqueeze(-1).unsqueeze(-1), sigma_candidate, sigma_old)
    return mu, sigma


def _beta_to_coordinate(
    beta:       torch.Tensor,             # (N, N) single-block OR (H, N, N) per-head
    irrep_dims: Optional[List[int]],      # block sizes; None/[K] -> single block
    K:          int,                      # total belief dimension
) -> torch.Tensor:                        # (N, N, K) per-coordinate attention weight
    r"""Broadcast attention weights to coordinate k via k's irrep block h(k).

    Single block (irrep_dims None or length 1): the one beta_ij repeated across every
    coordinate. Per-head ((H,N,N) with H>1): head h's weight repeated across its d_head
    coordinates, so coordinate k carries beta_ij^(h(k)).
    """
    if irrep_dims is None or len(irrep_dims) == 1:
        return beta.unsqueeze(-1).expand(*beta.shape, K)
    x = beta.movedim(-3, -1)                                             # (N, N, H)
    if len(set(irrep_dims)) == 1:                                        # equal blocks (block_glk):
        d = irrep_dims[0]                                                # expand/reshape, no gather --
        return x.unsqueeze(-1).expand(*x.shape, d).reshape(*x.shape[:-1], x.shape[-1] * d)  # bit-identical
    reps = torch.tensor(irrep_dims, device=beta.device)                 # unequal blocks: gather fallback
    return torch.repeat_interleave(x, reps, dim=-1)                      # (N,N,H)->(N,N,K)


def get_kernel(name: str) -> Callable:
    """Return the registered kernel for family ``name`` (KeyError if absent)."""
    if name not in _KERNELS:
        raise KeyError(f"no belief-gradient kernel for family {name!r}; available: {sorted(_KERNELS)}")
    return _KERNELS[name]


def _gaussian_full_no_gradient_kernel(*_args: object, **_kwargs: object) -> None:
    r"""Marker callable: ``gaussian_full`` declares ONLY the mm_exact capability (no analytic
    (grad_mu, grad_sigma) kernel exists), so ``belief_gradients`` must never actually call this --
    ``uses_kernel_route(route="gradient")`` gates on ``KernelRegistration.provides_gradient=False``
    and keeps routing ``e_step_update='gradient'`` + ``family='gaussian_full'`` to the autograd
    oracle, exactly as before this registration existed. Raises loudly rather than mis-scoring if
    that capability gate ever has a bug and this is reached anyway."""
    raise NotImplementedError(
        "gaussian_full has no analytic (grad_mu, grad_sigma) kernel; its _KERNELS entry exists "
        "only to declare the mm_exact closed-form capability (see mm_exact_update / "
        "_mm_exact_full_covariance). This function must be unreachable through belief_gradients "
        "(uses_kernel_route(route='gradient') checks KernelRegistration.provides_gradient=False for "
        "this family) -- if you are seeing this, that capability gate has a bug."
    )


register_kernel(
    "gaussian_full",
    provides_gradient=False,           # no analytic gradient kernel; belief_gradients keeps using the oracle
    provides_mm_exact=True,            # mm_exact_update's _mm_exact_full_covariance branch covers it
)(_gaussian_full_no_gradient_kernel)
