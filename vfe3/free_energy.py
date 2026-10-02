r"""The single authoritative scalar free energy F = sum_i F_i for VFE_3.0.

F is divergence-agnostic: it assembles the scalar from per-pair energies E_ij and
self-divergences D(q_i||p_i) supplied by the `divergence` registry, so a new
divergence slots in by registration + config, never by editing F. Canonical (with
the attention-entropy term) vs surrogate is a single toggle. The attention prior
is a log-bias B_ij from the `attention_prior` seam; beta* = softmax_j(B - E/tau).
"""

import math
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch

from vfe3.alpha_i import alpha_is_per_coord
from vfe3.divergence import (
    get_functional,
    get_functional_per_coord,
    has_per_coord_functional,
    divergence_functionals_per_coord,
)
from vfe3.families.base import BeliefParams


def _broadcast_tau(tau: 'float | torch.Tensor', energy: torch.Tensor) -> 'float | torch.Tensor':
    r"""Reshape a per-head (H,) tau so it broadcasts against an (..., H, N, N) energy.

    Scalar/0-d tau passes through unchanged (the single-block / scalar-kappa path).
    A 1-d (H,) tau is reshaped to (H, 1, 1): the head axis is always 3rd from the last
    in the energy tensor, so two trailing 1s align correctly for both the unbatched
    (H, N, N) and batched (B, H, N, N) layouts. The reshape also moves tau onto the
    energy's device (no-op when already there): attention_tau builds a CPU (H,) tau
    when a SCALAR kappa meets unequal irrep dims, and this is the one funnel every
    tau-consuming division passes through. A >= 2-d tensor tau passes through unchanged;
    its trailing singleton key axis already right-aligns against the energy.
    """
    if isinstance(tau, torch.Tensor) and tau.dim() == 1:
        # A LENGTH-1 vector is a scalar wearing a head axis, and a single-irrep-block group
        # (glk / so_k / sp, and any cross-coupled block_glk) emits a HEADLESS (..., N, N) energy --
        # there is no head axis for it to align against. Reshaping it to (1, 1, 1) prepended a
        # phantom head that propagated into beta and grad_sigma and broke the attention-map einsum.
        # Collapse a length-one coefficient to 0-d: same number, no invented axis. A
        # genuine multi-head tau (H >= 2) keeps the per-head broadcast unchanged.
        if tau.shape[0] == 1:
            return tau.to(device=energy.device).reshape(())
        return tau.to(device=energy.device).reshape(tau.shape[0], 1, 1)
    return tau


def attention_tau(
    kappa:      'float | torch.Tensor',   # sharpness scalar or (H,) per-head (kappa=1 -> Vaswani recovery)
    irrep_dims: List[int],                # gauge-irrep block sizes; sum == K
) -> 'float | torch.Tensor':
    r"""Softmax temperature tau = kappa * sqrt(d_energy), where d_energy is the dimension the
    per-pair energy E_ij = D(q_i || Omega_ij q_j) accumulates over -- the gauge-irrep BLOCK size.

    Single-block groups (glk / so_k / sp report ``irrep_dims=[K]``) accumulate the divergence over
    the full K, so d_energy = K. Per-head multi-block groups (block_glk: ``irrep_dims=[d_head]*H``)
    accumulate per head, so d_energy = d_head. In both EQUAL-block cases d_energy =
    ``irrep_dims[0]`` and the return is the scalar kappa * sqrt(d) (byte-identical to before).
    UNEQUAL blocks (the so_n/sp_n irrep towers, e.g. the SO(3) spin dims [1, 3, 5, 7]) get a
    per-head (H,) tau with tau_h = kappa_h * sqrt(d_h), so every head's softmax runs at the
    Vaswani temperature of the dimension ITS energy accumulates over; a scalar kappa broadcasts
    across heads, a per-head (H,) kappa is elementwise. The (H,) tau broadcasts downstream via
    ``_broadcast_tau`` (which also handles device placement against the energy).
    """
    if isinstance(kappa, torch.Tensor) and kappa.dim() not in (0, 1):
        raise ValueError(
            f"kappa tensor must be 0-d (scalar) or 1-d (per-head); got {kappa.dim()}-d "
            f"shape {tuple(kappa.shape)}"
        )
    if (isinstance(kappa, torch.Tensor) and kappa.dim() == 1
            and kappa.shape[0] != len(irrep_dims)):
        raise ValueError(
            f"per-head kappa has {kappa.shape[0]} entries but the group has "
            f"{len(irrep_dims)} irrep blocks (irrep_dims={irrep_dims})"
        )
    if len(set(irrep_dims)) == 1:
        return kappa * (irrep_dims[0] ** 0.5)
    sqrt_d = _sqrt_dims(                                           # (H,) per-block sqrt(d_h), cached
        tuple(irrep_dims),
        kappa.device if isinstance(kappa, torch.Tensor) else torch.device("cpu"),
        kappa.dtype if isinstance(kappa, torch.Tensor) else torch.float32,
    )
    return kappa * sqrt_d


_SQRT_D_CACHE: Dict[Tuple[Tuple[int, ...], str, torch.dtype], torch.Tensor] = {}


def _sqrt_dims(dims: Tuple[int, ...], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    r"""Cached per-block sqrt(d_h) vector: attention_tau is called every vfe_block invocation,
    and the unequal-dims branch was allocating this loop-invariant tensor each time (audit
    2026-06-09 overnight F7)."""
    key = (dims, str(device), dtype)
    t = _SQRT_D_CACHE.get(key)
    if t is None:
        t = torch.tensor([float(d) for d in dims], device=device, dtype=dtype).sqrt()
        _SQRT_D_CACHE[key] = t
    return t


def self_divergence(
    q:                 BeliefParams,        # (..., N, K) belief
    p:                 BeliefParams,        # (..., N, K) prior

    *,
    alpha:             float = 1.0,
    kl_max:            float = 100.0,
    eps:               float = 1e-6,
    divergence_family: str   = "renyi",
) -> torch.Tensor:                         # (..., N) D(q_i || p_i)
    r"""Self-coupling divergence via the seam (full-K, not per-head: D(q_i||p_i) is the whole
    belief). ``divergence_family`` selects the functional; the params' subclass the kernel."""
    return get_functional(divergence_family)(
        q, p, alpha=alpha, kl_max=kl_max, eps=eps,
    )


def self_divergence_per_coord(
    q:                 BeliefParams,        # (..., N, K) belief
    p:                 BeliefParams,        # (..., N, K) prior

    *,
    alpha:             float = 1.0,
    kl_max:            float = 100.0,
    eps:               float = 1e-6,
    divergence_family: str   = "renyi",
) -> torch.Tensor:                         # (..., N, K) per-coordinate D^(k)(q_i||p_i)
    r"""Per-coordinate self-divergence, unsummed over the coordinate axis.

    Diagonal families use their coordinate decomposition. Full Gaussians use
    ordered conditional KL contributions, retaining correlations. The full-family
    method rejects Renyi orders other than one; Jeffreys sums both KL directions.
    Non-additive functionals such as squared Hellinger remain unsupported.
    """
    if not callable(getattr(q, "renyi_per_coord", None)):
        raise ValueError(f"{type(q).__name__} has no per-coordinate self-divergence")
    if not has_per_coord_functional(divergence_family):
        raise ValueError(
            f"self_divergence_per_coord has no per-coordinate form for "
            f"divergence_family={divergence_family!r}: only divergences that decompose "
            f"coordinate-wise on a diagonal Gaussian are registered "
            f"({divergence_functionals_per_coord()}). 'squared_hellinger' is excluded (H^2 = "
            f"1 - exp(-D_{{1/2}}/2) is a nonlinear transform of the summed divergence). Use a "
            f"decomposable divergence or a per-position lambda_alpha_mode."
        )
    return get_functional_per_coord(divergence_family)(q, p, alpha=alpha, kl_max=kl_max, eps=eps)


def self_divergence_for_alpha(
    q:                 BeliefParams,        # (..., N, K) belief
    p:                 BeliefParams,        # (..., N, K) prior

    *,
    alpha:             float = 1.0,
    kl_max:            float = 100.0,
    eps:               float = 1e-6,
    divergence_family: str   = "renyi",
    lambda_alpha_mode: str   = "constant",
) -> torch.Tensor:                         # (..., N) summed, or (..., N, K) per-coordinate
    r"""Self-divergence shaped for the selected alpha form: per-coordinate (..., N, K) when the
    form declares ``per_coord=True`` (``alpha_i.alpha_is_per_coord``), else the per-position
    summed (..., N). This is the single routing seam every alpha consumer (the autograd oracle,
    the analytic kernel, the e_step F value, model diagnostics) shares, so a new alpha form's
    divergence-reduction need is honoured by its registration alone, with no consumer edited.
    """
    if alpha_is_per_coord(lambda_alpha_mode):
        return self_divergence_per_coord(
            q, p, alpha=alpha, kl_max=kl_max, eps=eps,
            divergence_family=divergence_family,
        )
    return self_divergence(
        q, p, alpha=alpha, kl_max=kl_max, eps=eps,
        divergence_family=divergence_family,
    )


def attention_weights(
    energy:    torch.Tensor,               # (..., N) or (..., N, N) per-key energies E_ij

    *,
    tau:       'float | torch.Tensor' = 1.0,
    log_prior: Optional[torch.Tensor] = None,   # (..., N/NxN) bias B_ij; None -> 0
) -> torch.Tensor:                         # (...) softmax_j(B - E/tau)
    r"""Attention weights beta*_ij = softmax_j(B_ij - E_ij / tau)."""
    logits = -energy / _broadcast_tau(tau, energy)
    if log_prior is not None:
        if (log_prior.dtype == logits.dtype
                and torch.broadcast_shapes(logits.shape, log_prior.shape) == logits.shape):
            logits.add_(log_prior)                         # fresh logits; no promotion or expansion
        else:
            logits = logits + log_prior
    return torch.softmax(logits, dim=-1)


def log_partition(
    energy:    torch.Tensor,               # (..., N) or (..., N, N)

    *,
    tau:       'float | torch.Tensor' = 1.0,
    log_prior: Optional[torch.Tensor] = None,
) -> torch.Tensor:                         # (...) log Z_i = logsumexp_j(log pi - E/tau)
    r"""Log-partition log Z_i = logsumexp_j(log pi_ij - E_ij / tau), pi = softmax_j(B).

    The partition Z_i = Sum_j pi_ij exp(-E_ij/tau) is built from the NORMALIZED
    prior pi (not the raw log-bias B), so the envelope identity
    Sum_j beta*_ij E_ij + tau Sum_j beta*_ij log(beta*_ij/pi_ij) = -tau log Z_i
    holds for ANY prior the seam emits. Equivalently log Z = logsumexp(B - E/tau)
    - logsumexp(B); using log_softmax(B) subtracts that per-row normalizer in one
    step. With a None prior pi is uniform 1/N, so the bias is -log(N).
    """
    logits = -energy / _broadcast_tau(tau, energy)
    if log_prior is not None:
        logits = logits + torch.log_softmax(log_prior, dim=-1)
    else:
        logits = logits - torch.log(torch.tensor(float(energy.shape[-1]),
                                                  device=energy.device, dtype=energy.dtype))
    return torch.logsumexp(logits, dim=-1)


def reduced_free_energy(
    energy:    torch.Tensor,               # (..., N) or (..., N, N)

    *,
    tau:       'float | torch.Tensor' = 1.0,
    log_prior: Optional[torch.Tensor] = None,
) -> torch.Tensor:                         # (...) F_red,i = -tau log Z_i
    r"""Reduced (envelope) free energy F_red,i = -tau log Z_i; equals the canonical
    beta-block evaluated at beta* for ANY prior (log_partition normalizes the
    prior internally, so the +tau logsumexp(B) per-row offset cannot leak in)."""
    lz = log_partition(energy, tau=tau, log_prior=log_prior)   # (..., N) or (..., H, N)
    # Per-head (H,) tau must broadcast against lz (..., H, N): reshape to (H, 1) so it aligns
    # with the head axis at -2 of lz. (H,1) right-aligns correctly for both (H,N) and (B,H,N).
    # Scalar tau passes through unchanged. The .to() mirrors _broadcast_tau's device hop (a
    # scalar-kappa x unequal-irrep-dims tau is born on CPU). A >= 2-d tensor tau with a trailing
    # singleton key axis drops that axis to align with lz's rows.
    if isinstance(tau, torch.Tensor) and tau.dim() == 1:
        _tau = tau.to(device=lz.device).reshape(tau.shape[0], 1)
    elif isinstance(tau, torch.Tensor) and tau.dim() >= 2:
        _tau = tau.squeeze(-1)
    else:
        _tau = tau
    return -_tau * lz


def free_energy(
    self_div:                  torch.Tensor,        # (..., N) or (..., N, K) D(q_i||p_i)
    energy:                    torch.Tensor,        # (..., N, N) E_ij belief-coupling energies
    alpha:                     torch.Tensor,        # (..., N) or (..., N, K) self-coupling

    *,
    tau:                       'float | torch.Tensor' = 1.0,
    lambda_beta:               'float | torch.Tensor' = 1.0,    # weight on the WHOLE belief-coupling block
    log_eps:                   float = 1e-12,                   # floor for log(beta)/log(pi) in the entropy term
    include_attention_entropy: bool  = True,

    log_prior:                 Optional[torch.Tensor] = None,   # (..., N, N) attention log-prior
    alpha_reg:                 Optional[torch.Tensor] = None,   # (..., N[,K]) R(alpha) if state-dep
    coupling_energy:           Optional[torch.Tensor] = None,   # (..., N, N) VALUE-gauge energy for the coupling sum (None -> energy)
    log_likelihood:            Optional[torch.Tensor] = None,   # (..., N) E_q[log p(o|k)] observation term; GATED STUB (see docstring)
) -> torch.Tensor:                                  # scalar F = sum_i F_i
    r"""Single authoritative scalar free energy.

        F = sum_i [ alpha_i . D(q_i||p_i)            (+ R(alpha_i) if state-dependent)
                  + lambda_beta . ( sum_j beta_ij E_ij
                                    + tau sum_j beta_ij log(beta_ij/pi_ij) )   (entropy: canonical only)
                  - ell_i ]
    beta_ij = softmax_j(log_prior - E/tau); pi = softmax_j(log_prior). ``lambda_beta`` (1.0 = the
    canonical/pure F) scales the COUPLING and ENTROPY together by the SAME factor and leaves beta
    untouched (no lambda inside the softmax), so beta = softmax(-E/tau) stays the stationary point of
    the scaled block and the envelope identity d/dtheta[lambda_beta (coupling+entropy)] =
    lambda_beta sum_j beta* dE/dtheta still holds -- keeping the analytic kernel (which scales its
    pair term by lambda_beta) in agreement with autograd of this F.
    adds the two-hop coupling block F_2 = lambda_2 sum_ik (beta beta)_ik E_ik with DETACHED hop
    weights and no entropy term (see the guarded block below).

    The observation/data term -E_q[log p(o|k)] enters via the optional ``log_likelihood`` arg
    (subtracted below) but is a GATED STUB: no production caller supplies it. The projected
    belief routes never call this function, so wiring it here alone is inert. The term is non-vacuous only once the per-token prior is replaced
    by a top-down cross-scale shadow prior (PIFB:1233): in the current next-token model the input
    token already sets p_i (encode q=p) and the next token already drives the cross-entropy
    (``metrics.py`` books that CE as the data term), so an observation carries no information distinct
    from both. Do NOT "complete the canonical functional" by wiring ``log_likelihood`` here plus one
    caller and believing it live. See docs/2026-06-07-observation-likelihood-term-brainstorm.md.

    Divergence-agnostic: `self_div`/`energy` come from the divergence seam, so a new
    divergence requires no change here.
    """
    beta = attention_weights(energy, tau=tau, log_prior=log_prior)        # (..., N, N) from the SCORE energy

    # self-coupling (sum over coordinate axis too when alpha/self_div are per-coord)
    self_term = alpha * self_div
    if alpha_reg is not None:
        self_term = self_term + alpha_reg
    self_total = self_term.sum()

    # The coupling sum carries the belief-coupling energy beta_ij E_ij. ``coupling_energy`` (None ->
    # the score ``energy``, the coherent single-operator default) lets the VALUE gauge differ from the
    # ATTENTION gauge: beta is still softmax of the score energy, but the energy summed (the pull the
    # belief descends) is the value-gauge energy. With coupling_energy=energy this is byte-identical;
    # decoupled, beta is no longer the stationary point of the summed block, so the envelope theorem
    # fails and only the autograd oracle (not the closed-form kernel) computes the gradient correctly.
    coupling = (beta * (energy if coupling_energy is None else coupling_energy)).sum()

    F = self_total + lambda_beta * coupling
    if include_attention_entropy:
        # Uniform prior (log_prior=None): log pi = -log N is a SCALAR, so no (..., N, N) pi
        # tensor is materialized (audit 2026-06-09 overnight F8 / morning PE7). The max()
        # mirrors the tensor branch's clamp exactly (inert for any real N < 1/log_eps).
        if log_prior is not None:
            # Exact log-prior (m8): the old torch.log(softmax(...).clamp(min=log_eps)) floored a finite
            # deep-tail entry at ~-27.6 nats, so a strong finite prior made F and the autograd oracle
            # deviate from -tau logZ. log_softmax is exact; isfinite neutralizes hard-mask (-inf) entries
            # (beta==0 there, so 0*(.-log_pi) stays the correct 0 with no 0*(-inf)=NaN).
            log_pi = torch.log_softmax(log_prior, dim=-1)
            log_pi = torch.where(torch.isfinite(log_pi), log_pi, torch.zeros_like(log_pi))
        else:
            log_pi = math.log(max(1.0 / beta.shape[-1], log_eps))
        _tau_e = _broadcast_tau(tau, energy)          # (H,1,1) for per-head, scalar otherwise
        # xlogy is the coherent zero-safe extension of beta * log(beta): it returns exactly zero
        # at beta == 0 without flooring any positive tail probability. Flooring log(beta) changes
        # the differentiated scalar below the floor and breaks the Gibbs-envelope derivative.
        beta_for_log = torch.where(beta > 0.0, beta, torch.ones_like(beta))
        entropy = (_tau_e * (torch.special.xlogy(beta, beta_for_log) - beta * log_pi)).sum()
        F = F + lambda_beta * entropy
    if log_likelihood is not None:                              # observation/data term -E_q[log p(o|k)] (gated stub; no live caller)
        F = F - log_likelihood.sum()
    return F


# ===========================================================================
# Typed hierarchical (q/p) decomposition (PB-10).
#
# ``free_energy`` above is the pure q-only training scalar and keeps its byte-for-byte one-shot
# ``.sum()`` reductions; it does NOT delegate to the rows below, because float32 one-shot reductions
# are not bitwise associative with the row-then-query reductions the evaluator performs. The rows and
# the evaluator are the SINGLE typed boundary the diagnostics route their belief assembly through, so
# the belief rows are no longer summed into the total by hand at each call site.
# ===========================================================================

class BeliefFreeEnergyRows(NamedTuple):
    r"""Per-query (..., N) belief-channel rows: already SIGNED and WEIGHTED contributions.

    ``self_coupling`` is alpha_i D(q_i||p_i) (+ R(alpha_i)) reduced over any coordinate axis;
    ``belief_coupling`` is lambda_beta sum_j beta_ij E_ij; ``attention_entropy`` is
    lambda_beta tau sum_j beta_ij log(beta_ij/pi_ij) (an exact zero row when the entropy is gated
    off); ``observation_nll`` is -E_q[log p(o|x)] (an exact zero row when no observation term is
    supplied). Absent blocks are ``torch.zeros_like`` of the self row, never a hidden branch."""
    self_coupling:     torch.Tensor
    belief_coupling:   torch.Tensor
    attention_entropy: torch.Tensor
    observation_nll:   torch.Tensor


class HierarchicalFreeEnergyTerms(NamedTuple):
    r"""The four reduced hierarchical components and their assembled ``total``.

    ``total`` is the field-order sum: self + belief coupling + attention entropy + observation NLL."""
    self_coupling:     torch.Tensor
    belief_coupling:   torch.Tensor
    attention_entropy: torch.Tensor
    observation_nll:   torch.Tensor
    total:             torch.Tensor


def _reduce_row(row: torch.Tensor, how: str) -> torch.Tensor:
    r"""Reduce a per-query row to a scalar by ``"sum"`` or ``"mean"`` over all batch/query entries."""
    if how == "sum":
        return row.sum()
    if how == "mean":
        return row.mean()
    raise ValueError(f"reduction must be 'sum' or 'mean', got {how!r}")


def hierarchical_free_energy_terms(
    self_coupling_rows:     torch.Tensor,  # (..., N), already alpha-weighted and regularized
    belief_coupling_rows:   torch.Tensor,  # (..., N), lambda_beta * sum_j beta_ij E_ij
    attention_entropy_rows: torch.Tensor,  # (..., N), lambda_beta * tau * sum_j beta log(beta/pi)
    observation_nll_rows:   torch.Tensor,  # (..., N), -E_q[log p(o|x)]

    *,
    q_reduction:     str = "sum",
) -> HierarchicalFreeEnergyTerms:
    r"""The single authoritative scalar evaluator over four already-signed, already-weighted rows.

    Every row reduces with ``q_reduction``, which is ``"sum"`` or ``"mean"`` applied directly to every
    batch/query entry after the caller has already collapsed the structural (coordinate/key/head)
    axes. Absent blocks are passed as ``torch.zeros_like`` rows -- there is no hidden branch. The
    evaluator validates matching ``(..., N)`` shapes and devices, reduces each row exactly once, and
    assembles ``total`` in the fixed field order. It NEVER detaches or reweights an input."""
    rows = (self_coupling_rows, belief_coupling_rows, attention_entropy_rows,
            observation_nll_rows)
    for r in rows:
        if not isinstance(r, torch.Tensor):
            raise TypeError(f"every hierarchical row must be a torch.Tensor, got {type(r)!r}")
    ref = self_coupling_rows
    for r in rows:
        if r.shape != ref.shape:
            raise ValueError(
                f"hierarchical rows must share one (..., N) shape; got {tuple(r.shape)} vs "
                f"{tuple(ref.shape)}")
        if r.device != ref.device:
            raise ValueError(
                f"hierarchical rows must share one device; got {r.device} vs {ref.device}")

    self_coupling     = _reduce_row(self_coupling_rows,     q_reduction)
    belief_coupling   = _reduce_row(belief_coupling_rows,   q_reduction)
    attention_entropy = _reduce_row(attention_entropy_rows, q_reduction)
    observation_nll   = _reduce_row(observation_nll_rows,   q_reduction)

    total = self_coupling + belief_coupling + attention_entropy + observation_nll
    return HierarchicalFreeEnergyTerms(
        self_coupling, belief_coupling, attention_entropy, observation_nll, total,
    )


def _belief_free_energy_rows(
    self_div:                  torch.Tensor,        # (..., N) or (..., N, K) D(q_i||p_i)
    energy:                    torch.Tensor,        # (..., N, N) or (..., H, N, N) E_ij
    alpha:                     torch.Tensor,        # (..., N) or (..., N, K) self-coupling

    *,
    tau:                       'float | torch.Tensor' = 1.0,
    lambda_beta:               'float | torch.Tensor' = 1.0,
    log_eps:                   float = 1e-12,
    include_attention_entropy: bool  = True,

    log_prior:                 Optional[torch.Tensor] = None,   # (..., N, N) attention log-prior
    alpha_reg:                 Optional[torch.Tensor] = None,   # (..., N[,K]) R(alpha) if state-dep
    coupling_energy:           Optional[torch.Tensor] = None,   # (..., N, N) VALUE-gauge coupling energy
    log_likelihood:            Optional[torch.Tensor] = None,   # (..., N) E_q[log p(o|k)]
    beta_override:             Optional[torch.Tensor] = None,   # captured beta (avoids recompute)
    per_coord:                 Optional[bool] = None,           # self_div carries a coord axis; None -> infer
) -> BeliefFreeEnergyRows:
    r"""Per-query (..., N) decomposition of the belief channel that :func:`free_energy` sums to a
    scalar -- a NEW hierarchical/diagnostic path, NOT the scalar's reduction order.

    It collapses the coordinate, key, and optional head axes while preserving the batch/query axes.
    The five returned rows are already SIGNED and WEIGHTED (lambda_beta in both beta rows,
    ``observation_nll = -log_likelihood`` when present); the
    attention-entropy row is an exact zero when the entropy is gated off, and the two-hop and
    observation rows are exact zeros when absent. ``beta_override`` supplies a captured beta so
    diagnostic weights are not recomputed. The ONLY detach here forms the fixed two-hop weights
    W2 = beta.detach() @ beta.detach()."""
    beta = (beta_override if beta_override is not None
            else attention_weights(energy, tau=tau, log_prior=log_prior))
    value_energy = energy if coupling_energy is None else coupling_energy

    # Axis bookkeeping. ``energy`` is (..., N, N) or (..., H, N, N) (the head axis, block_glk, is 3rd
    # from last); ``self_div`` is (..., N) or per-coordinate (..., N, K) (state_dependent_per_coord).
    # The rank gap g = energy.dim() - self_div.dim() then reads: g == 2 -> head, no coord; g == 0 ->
    # coord, no head; g == 1 is AMBIGUOUS between (no coord, no head) and (coord AND head), so it is
    # disambiguated by shape -- (no coord, no head) requires self_div.shape == energy.shape[:-1]
    # exactly. Callers that know the alpha form (diagnostics) pass ``per_coord`` explicitly, which
    # also covers the H == N == K collision the shape test cannot separate.
    gap = energy.dim() - self_div.dim()
    if per_coord is None:
        if gap == 0:
            per_coord = True
        elif gap == 2:
            per_coord = False
        elif gap == 1:
            per_coord = self_div.shape != energy.shape[:-1]
        else:
            raise ValueError(
                f"self_div rank {self_div.dim()} is inconsistent with energy rank {energy.dim()}")
    has_head = (gap + (1 if per_coord else 0)) >= 2

    def _collapse(x: torch.Tensor) -> torch.Tensor:                   # (..., [H,] N) -> (..., N)
        return x.sum(dim=-2) if has_head else x

    belief_row = lambda_beta * _collapse((beta * value_energy).sum(dim=-1))

    self_term = alpha * self_div
    if alpha_reg is not None:
        self_term = self_term + alpha_reg
    if per_coord:
        self_term = self_term.sum(dim=-1)                             # collapse the coordinate axis
    self_row = self_term
    if self_row.shape != belief_row.shape:
        raise ValueError(
            f"belief rows disagree on the (..., N) shape: self {tuple(self_row.shape)} vs "
            f"coupling {tuple(belief_row.shape)}; pass per_coord explicitly if the axis "
            f"inference misread the layout")

    if include_attention_entropy:
        if log_prior is not None:
            log_pi = torch.log_softmax(log_prior, dim=-1)
            log_pi = torch.where(torch.isfinite(log_pi), log_pi, torch.zeros_like(log_pi))
        else:
            log_pi = math.log(max(1.0 / beta.shape[-1], log_eps))
        _tau_e = _broadcast_tau(tau, energy)
        beta_for_log = torch.where(beta > 0.0, beta, torch.ones_like(beta))
        entropy_full = _tau_e * (torch.special.xlogy(beta, beta_for_log) - beta * log_pi)
        entropy_row = lambda_beta * _collapse(entropy_full.sum(dim=-1))
    else:
        entropy_row = torch.zeros_like(self_row)

    observation_row = (-log_likelihood if log_likelihood is not None
                       else torch.zeros_like(self_row))

    return BeliefFreeEnergyRows(self_row, belief_row, entropy_row, observation_row)
