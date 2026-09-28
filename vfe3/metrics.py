r"""Diagnostic metrics for VFE_3.0 runs (publication-oriented, registry-backed).

A registry of named metrics over beliefs / transport / attention. Each metric reads
what it needs from a keyword context and returns a plain float (or a small dict), so
new probes slot in by ``register_metric`` without editing call sites; ``compute_metrics``
emits a CSV/JSON-friendly record. The metrics are pure measurements (no gradients, no
side effects) suitable for logging a training run or a figure.

Provided: effective_rank (spectral participation ratio), attention_entropy (row entropy
of beta), holonomy_deviation (cocycle/curvature departure of the transport from flat),
gauge_trace_spread (spread of log|det Omega| = tr embed(phi)), and free_energy_terms
(the per-term F decomposition).
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from vfe3.geometry.groups import GaugeGroup

from vfe3.geometry.lie_ops import CompactBlockElement
from vfe3.numerics import safe_eigvalsh
from vfe3.geometry.transport import CompactFactoredTransport


def effective_rank(
    spectrum: torch.Tensor,              # (..., K) non-negative spectrum (diagonal variances or eigenvalues)

    *,
    eps:      float = 1e-12,
) -> torch.Tensor:                       # (...) participation-ratio effective rank
    r"""Spectral effective rank (sum lam)^2 / sum lam^2 over the last axis.

    Equals K when the spectrum is flat (all equal) and -> 1 when one mode dominates. For a
    full covariance, pass its eigenvalues (``torch.linalg.eigvalsh(Sigma)``); for a diagonal
    belief, the variances ARE the spectrum.
    """
    lam = spectrum.float().clamp(min=0.0)
    s1 = lam.sum(dim=-1)
    s2 = (lam ** 2).sum(dim=-1).clamp(min=eps)
    return (s1 * s1 / s2).to(spectrum.dtype)


def attention_entropy(
    beta: torch.Tensor,                  # (..., N, N) row-stochastic attention weights

    *,
    eps:  float = 1e-12,
) -> torch.Tensor:                       # () mean row entropy
    r"""Mean Shannon entropy -Σ_j β_ij log β_ij over query rows (log N for uniform β)."""
    h = -(beta.clamp(min=eps) * torch.log(beta.clamp(min=eps))).sum(dim=-1)
    return h.mean()


def holonomy_deviation(
    omega: torch.Tensor,                 # (N, N, K, K) pairwise transport Omega_ij

    *,
    max_triangles: int = 512,
) -> torch.Tensor:                       # () mean ||Omega_ij Omega_jk Omega_ki - I||_F
    r"""Curvature proxy: mean Frobenius departure of the triangle holonomy from identity.

    FRAME-DEPENDENT (Frobenius; conjugation-invariant only for orthogonal gauges): under a GL(K)
    frame change the holonomy conjugates, H -> g H g^{-1}, and ||H - I||_F is preserved only for
    orthogonal/unitary g -- use :func:`holonomy_wilson_sampled` for gauge-invariant curvature.

    For a flat (Regime I) cocycle Omega_ij = exp(phi_i)exp(-phi_j) every triangle closes
    (H_ijk = I) so the deviation is ~0; a non-flat / non-cocycle transport gives > 0.
    """
    N, K = omega.shape[0], omega.shape[-1]
    eye = torch.eye(K, device=omega.device, dtype=omega.dtype)

    # Enumerate the first ``max_triangles`` distinct (i, j, k) triples in the same
    # row-major order the nested-loop form used, then evaluate them as ONE batched
    # (T, K, K) matmul rather than T Python-dispatched (K, K) matmuls -- same triangles,
    # same value, one kernel launch instead of T.
    triples: List[Tuple[int, int, int]] = []
    for i in range(N):
        for j in range(N):
            if j == i:
                continue
            for k in range(N):
                if k == i or k == j:
                    continue
                triples.append((i, j, k))
                if len(triples) >= max_triangles:
                    break
            if len(triples) >= max_triangles:
                break
        if len(triples) >= max_triangles:
            break
    if not triples:
        return torch.tensor(0.0, device=omega.device, dtype=omega.dtype)

    idx = torch.tensor(triples, device=omega.device)                      # (T, 3)
    o_ij = omega[idx[:, 0], idx[:, 1]]                                     # (T, K, K)
    o_jk = omega[idx[:, 1], idx[:, 2]]                                     # (T, K, K)
    o_ki = omega[idx[:, 2], idx[:, 0]]                                     # (T, K, K)
    H = o_ij @ o_jk @ o_ki                                                # (T, K, K) holonomy
    return torch.linalg.norm(H - eye, dim=(-2, -1)).mean()


def _finite_quantile(values: torch.Tensor, q: float) -> float:
    """Return a finite-only quantile, or NaN when no finite sample exists."""
    finite = values.detach().reshape(-1).float()
    finite = finite[torch.isfinite(finite)]
    return float(torch.quantile(finite, q)) if finite.numel() else float("nan")


def _diagnostic_embed(
    phi:        torch.Tensor,
    generators: 'torch.Tensor | GaugeGroup',
) -> torch.Tensor:
    r"""Embed $M = \sum_a \phi_a G_a$ at the caller's diagnostic precision."""
    if isinstance(generators, GaugeGroup):
        return generators.embed_algebra(phi, dtype=phi.dtype)
    return torch.einsum("...a,aij->...ij", phi, generators.to(dtype=phi.dtype))


def diagnostic_frame_norm(
    phi:        torch.Tensor,            # (..., n_gen) diagnostic coordinates
    group:      GaugeGroup,

    *,
    max_norm:   float,
    eps:        float,
) -> torch.Tensor:                       # (...) embedded Frobenius norm
    r"""Use $\|\sum_a\phi_a E_a\|_F = \|\phi\|_2$ for a certified elementary basis.

    This measurement-only shortcut keeps the dense reduction near either scale threshold.
    Nonfinite inputs, autograd, autocast, and custom or modified bases retain the original
    embedding, including its zero-times-infinity behavior. Away from thresholds the norm
    may differ by floating-point reduction order.
    """
    if (group.elementary_block_shape is not None
            and phi.dtype in (torch.float32, torch.float64)
            and phi.device == group.generator_device
            and phi.shape[-1] == group.generator_shape[0]
            and not (torch.is_grad_enabled() and phi.requires_grad)
            and not torch.is_autocast_enabled(phi.device.type)):
        norm = torch.linalg.vector_norm(phi, dim=-1)
        # Conservative guard band for both reduction lengths; retain the old reduction
        # when its bound is uninformative or a sum overflows/underflows.
        roundoff = 8 * group.generator_shape[-1] ** 2 * torch.finfo(phi.dtype).eps
        if roundoff < 0.5:
            margin = norm * roundoff + torch.finfo(phi.dtype).tiny ** 0.5
            near_threshold = ((norm - max_norm).abs() <= margin) | ((norm - eps).abs() <= margin)
            if bool((torch.isfinite(norm) & ~near_threshold).all()):
                return norm
    matrix = group.embed_algebra(phi, dtype=phi.dtype)
    return torch.linalg.matrix_norm(matrix, ord="fro", dim=(-2, -1))


def phi_chart_statistics(
    phi:        torch.Tensor,            # (..., n_gen) gauge-frame coordinates
    generators: 'torch.Tensor | GaugeGroup',            # (n_gen, K, K)

    *,
    max_norm:   float               = 20.0,
    eps:        float               = 1e-12,
    block_dims: Optional[List[int]] = None,
) -> Dict[str, float]:
    r"""Summarize embedded chart radius, exponential clamp activation, and conditioning."""
    matrix = _diagnostic_embed(phi.double(), generators)
    norm = torch.linalg.matrix_norm(matrix, ord="fro", dim=(-2, -1))
    scale = (max_norm / norm.clamp(min=eps)).clamp(max=1.0)
    from vfe3.geometry.transport import _blockwise_matrix_exp
    scaled = matrix * scale[..., None, None]
    exp_matrix = (_blockwise_matrix_exp(scaled, block_dims)
                  if block_dims and len(block_dims) > 1
                  else torch.linalg.matrix_exp(scaled))
    singular = torch.linalg.svdvals(exp_matrix)
    condition = singular[..., 0] / singular[..., -1].clamp(min=eps)
    return {
        "phi_matrix_norm_median": _finite_quantile(norm, 0.50),
        "phi_matrix_norm_p95":    _finite_quantile(norm, 0.95),
        "phi_matrix_norm_p99":    _finite_quantile(norm, 0.99),
        "phi_matrix_norm_max":    float(norm.max()) if norm.numel() else float("nan"),
        "phi_exp_clamp_frac":     float((scale < 1.0).float().mean()) if scale.numel() else 0.0,
        "phi_exp_scale_min":      float(scale.min()) if scale.numel() else 1.0,
        "vertex_cond_median":     _finite_quantile(condition, 0.50),
        "vertex_cond_p95":        _finite_quantile(condition, 0.95),
        "vertex_cond_p99":        _finite_quantile(condition, 0.99),
        "vertex_cond_max_ref":    float(condition.max()) if condition.numel() else float("nan"),
        "phi_chart_nonfinite_count": float((~torch.isfinite(matrix)).sum()),
    }




def _deterministic_triples(n_tokens: int, max_triangles: int) -> List[Tuple[int, int, int]]:
    """Return row-major distinct triples without drawing RNG state."""
    triples: List[Tuple[int, int, int]] = []
    for i in range(n_tokens):
        for j in range(n_tokens):
            if i == j:
                continue
            for k in range(n_tokens):
                if k == i or k == j:
                    continue
                triples.append((i, j, k))
                if len(triples) >= max_triangles:
                    return triples
    return triples


def flatness_reference_statistics(
    phi:        torch.Tensor,            # (N, n_gen) or (B, N, n_gen); first batch is sampled
    generators: 'torch.Tensor | GaugeGroup',            # (n_gen, K, K)
    right_phi:  Optional[torch.Tensor] = None,  # exact second vertex factor exp(Y), if present

    *,
    max_triangles: int   = 64,
    max_norm:      float = 20.0,
    eps:           float = 1e-12,
    block_dims:    Optional[List[int]] = None,
) -> Dict[str, float]:
    r"""Compare fp32 and rebuilt-fp64 numerical closure of a flat vertex cocycle.

    With ``right_phi=Y``, the measured vertex and inverse are the actual exact-composition
    operators ``U=exp(X) exp(Y)`` and ``U^{-1}=exp(-Y) exp(-X)``.
    """
    coordinate = phi[0] if phi.dim() == 3 else phi
    matrix64 = _diagnostic_embed(coordinate.double(), generators)
    norm64 = torch.linalg.matrix_norm(matrix64, ord="fro", dim=(-2, -1))
    scale64 = (max_norm / norm64.clamp(min=eps)).clamp(max=1.0)
    matrix64 = matrix64 * scale64[..., None, None]
    right_matrix64 = None
    if right_phi is not None:
        right_coordinate = right_phi[0] if right_phi.dim() == 3 else right_phi
        right_matrix64 = _diagnostic_embed(right_coordinate.double(), generators)
        right_norm64 = torch.linalg.matrix_norm(right_matrix64, ord="fro", dim=(-2, -1))
        right_scale64 = (max_norm / right_norm64.clamp(min=eps)).clamp(max=1.0)
        right_matrix64 = right_matrix64 * right_scale64[..., None, None]
    triples = _deterministic_triples(matrix64.shape[0], max_triangles)
    k_dim = matrix64.shape[-1]

    def _measure(
        matrix:       torch.Tensor,
        right_matrix: Optional[torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        from vfe3.geometry.transport import _blockwise_matrix_exp
        def _exp(value: torch.Tensor) -> torch.Tensor:
            return (_blockwise_matrix_exp(value, block_dims)
                    if block_dims and len(block_dims) > 1
                    else torch.linalg.matrix_exp(value))

        left_pos = _exp(matrix)
        left_neg = _exp(-matrix)
        if right_matrix is None:
            exp_pos = left_pos
            exp_neg = left_neg
        else:
            exp_pos = left_pos @ _exp(right_matrix)
            exp_neg = _exp(-right_matrix) @ left_neg
        eye = torch.eye(k_dim, device=matrix.device, dtype=matrix.dtype)
        inverse = torch.linalg.matrix_norm(exp_pos @ exp_neg - eye, ord="fro", dim=(-2, -1))
        inverse_rel = inverse / torch.linalg.matrix_norm(eye, ord="fro").clamp(min=eps)
        if not triples:
            zero = matrix.new_zeros(())
            return zero, zero, zero, zero, inverse.mean(), inverse_rel.mean()
        index = torch.tensor(triples, device=matrix.device)
        omega_ij = exp_pos[index[:, 0]] @ exp_neg[index[:, 1]]
        omega_jk = exp_pos[index[:, 1]] @ exp_neg[index[:, 2]]
        omega_ki = exp_pos[index[:, 2]] @ exp_neg[index[:, 0]]
        omega_ik = exp_pos[index[:, 0]] @ exp_neg[index[:, 2]]
        holonomy = omega_ij @ omega_jk @ omega_ki
        hol_abs = torch.linalg.matrix_norm(holonomy - eye, ord="fro", dim=(-2, -1))
        hol_rel = hol_abs / torch.linalg.matrix_norm(eye, ord="fro").clamp(min=eps)
        composed = omega_ij @ omega_jk
        cocycle_abs = torch.linalg.matrix_norm(composed - omega_ik, ord="fro", dim=(-2, -1))
        cocycle_scale = torch.maximum(
            torch.linalg.matrix_norm(composed, ord="fro", dim=(-2, -1)),
            torch.linalg.matrix_norm(omega_ik, ord="fro", dim=(-2, -1)),
        ).clamp(min=eps)
        return (
            hol_abs.mean(),
            hol_rel.mean(),
            cocycle_abs.mean(),
            (cocycle_abs / cocycle_scale).mean(),
            inverse.mean(),
            inverse_rel.mean(),
        )

    ha32, hr32, ca32, cr32, ia32, ir32 = _measure(
        matrix64.float(), None if right_matrix64 is None else right_matrix64.float(),
    )
    ha64, hr64, ca64, cr64, ia64, ir64 = _measure(matrix64, right_matrix64)
    return {
        "numerical_holonomy_fp32_abs": float(ha32),
        "numerical_holonomy_fp32_rel": float(hr32),
        "numerical_holonomy_fp64_abs": float(ha64),
        "numerical_holonomy_fp64_rel": float(hr64),
        "numerical_cocycle_fp32_abs":  float(ca32),
        "numerical_cocycle_fp32_rel":  float(cr32),
        "numerical_cocycle_fp64_abs":  float(ca64),
        "numerical_cocycle_fp64_rel":  float(cr64),
        "inverse_consistency_fp32_abs": float(ia32),
        "inverse_consistency_fp32_rel": float(ir32),
        "inverse_consistency_fp64_abs": float(ia64),
        "inverse_consistency_fp64_rel": float(ir64),
        "flatness_reference_triangles": float(len(triples)),
        "flatness_reference_factor_count": 2.0 if right_matrix64 is not None else 1.0,
        "flatness_reference_nonfinite_count": float(sum(
            not torch.isfinite(value)
            for value in (ha32, hr32, ha64, hr64, ca32, cr32, ca64, cr64, ia32, ir32, ia64, ir64)
        )),
    }


def gauge_trace_spread(
    phi:        torch.Tensor,            # (..., n_gen) gauge-frame coordinates
    generators: 'torch.Tensor | GaugeGroup',           # (n_gen, K, K)
) -> torch.Tensor:                       # () std of tr(embed(phi)) = std of log|det exp(embed phi)|
    r"""Spread (std) of log|det Omega| across tokens; 0 at phi = 0 (Omega = I)."""
    traces = (generators.generator_traces() if isinstance(generators, GaugeGroup)
              else generators.diagonal(dim1=-2, dim2=-1).sum(-1))        # (n_gen,) tr(G_a)
    logdet = torch.einsum("...a,a->...", phi, traces)            # tr(embed(phi)) = sum_a phi^a tr(G_a)
    return logdet.flatten().std(unbiased=False)


def free_energy_terms(
    self_div: torch.Tensor,              # (..., N) D(q_i||p_i)
    energy:   torch.Tensor,              # (..., N, N) E_ij
    beta:     torch.Tensor,              # (..., N, N) attention weights
    alpha:    torch.Tensor,              # (..., N) self-coupling

    *,
    tau:                       'float | torch.Tensor' = 1.0,
    lambda_beta:               'float | torch.Tensor' = 1.0,   # weight on the belief-coupling block (1.0 = pure F)
    eps:                       float = 1e-12,
    include_attention_entropy: bool  = True,                   # gate the entropy term in ``total``

    log_prior:                 Optional[torch.Tensor] = None,  # (..., N, N) attention log-prior
    alpha_reg:                 Optional[torch.Tensor] = None,  # (..., N[,K]) R(alpha_i) if state-dependent
    coupling_energy:           Optional[torch.Tensor] = None,  # (..., N, N) VALUE-gauge coupling energy
    log_likelihood:            Optional[torch.Tensor] = None,  # (..., N) E_q[log p(o|k)] observation term
) -> Dict[str, float]:
    r"""Per-term free-energy decomposition: self-coupling, belief-coupling, attention entropy.

    Mirrors ``vfe3.free_energy.free_energy`` so the reported ``total`` equals the scalar that
    function returns for the SAME inputs (within float32 rounding):

        F = sum_i [ alpha_i D(q_i||p_i) + R(alpha_i)
                  + lambda_beta ( sum_j beta_ij E_ij
                                  + tau sum_j beta_ij log(beta_ij/pi_ij) )
                  - ell_i ]   (entropy: canonical only)

    ``belief_coupling``, ``attention_entropy``, and the conditionally present
    ``observation_likelihood`` are RAW (unweighted) block values, so each
    stays individually interpretable; ``total`` is the runtime-realized SCALED free energy
    self_coupling + lambda_beta (belief_coupling + attention_entropy), matching what the E-step
    actually minimizes. At lambda_beta = 1.0 total is byte-identical to the
    unscaled sum.

    ``alpha_reg`` is the per-position regularizer R(alpha_i) (b0 alpha - c0 log alpha under
    ``state_dependent_per_coord``); when supplied it is added elementwise into the self-coupling
    term (and so into ``total``), matching ``free_energy``. Default None leaves the result
    byte-identical to the unregularized path. ``include_attention_entropy`` gates the entropy
    contribution to ``total`` exactly as ``free_energy`` does -- when False the surrogate objective
    the E-step descends omits the entropy term, but its value is still reported under
    ``attention_entropy`` for diagnostics.

    ``coupling_energy`` selects the VALUE-gauge energy for both the one-hop and two-hop coupling
    sums while ``energy`` remains the score energy that produced ``beta``.
    raw detached ``beta @ beta`` weights, matching the scalar objective, and ``log_likelihood`` is
    subtracted from ``total`` when the gated observation seam supplies it. Their defaults preserve
    the original four-term result exactly.
    """
    # RAW public fields (unchanged, unweighted, positive-likelihood): kept byte-for-byte for their
    # reported values -- callers/figures read these individually. ``total`` alone routes through the
    # single typed hierarchical evaluator (weights, entropy gate, negative-likelihood sign). We never
    # divide a weighted row to recover a raw field, so zero weights stay valid.
    self_term = alpha * self_div
    if alpha_reg is not None:
        self_term = self_term + alpha_reg
    self_coupling = float(self_term.sum())
    value_energy = energy if coupling_energy is None else coupling_energy
    belief_coupling = float((beta * value_energy).sum())
    # log pi WITHOUT materializing a full (H,N,N) uniform-pi tensor: with log_prior=None, pi=1/N is
    # uniform so log(pi)=-log(N) is a scalar (audit 2026-06-17 r2 id9; the scalar new_tensor(1/N) keeps
    # the value byte-identical to the old full_like alloc). The log_prior branch is unchanged.
    if log_prior is not None:
        log_pi = torch.log_softmax(log_prior, dim=-1)                       # exact log-prior (m8; was clamped ~-27.6 nats)
        log_pi = torch.where(torch.isfinite(log_pi), log_pi, torch.zeros_like(log_pi))
    else:
        log_pi = torch.log(beta.new_tensor(1.0 / beta.shape[-1]).clamp(min=eps))
    from vfe3.free_energy import (                       # align a per-head (H,) tau to the head axis
        _belief_free_energy_rows, _broadcast_tau, hierarchical_free_energy_terms,
    )
    _tau_e = _broadcast_tau(tau, energy)
    entropy = float((_tau_e * (beta * (torch.log(beta.clamp(min=eps)) - log_pi))).sum())
    observation_likelihood = None
    if log_likelihood is not None:
        observation_likelihood = float(log_likelihood.sum())
    # SIGNED, WEIGHTED belief rows through the evaluator; use only its total.
    rows = _belief_free_energy_rows(
        self_div, energy, alpha, tau=tau, lambda_beta=lambda_beta, log_eps=eps,
        include_attention_entropy=include_attention_entropy,
        log_prior=log_prior, alpha_reg=alpha_reg, coupling_energy=coupling_energy,
        log_likelihood=log_likelihood, beta_override=beta,
    )
    total = float(hierarchical_free_energy_terms(
        rows.self_coupling, rows.belief_coupling, rows.attention_entropy,
        rows.observation_nll, q_reduction="sum",
    ).total)
    terms = {
        "self_coupling":   self_coupling,
        "belief_coupling": belief_coupling,
        "attention_entropy": entropy,
        "total":           total,
    }
    if observation_likelihood is not None:
        terms["observation_likelihood"] = observation_likelihood
    return terms


# ===========================================================================
# Publication-figure metrics (pure measurements; no gradients, no side effects).
# These return per-token / per-head VECTORS or small dicts for figures, rather
# than the single CSV scalar the registry path logs. Runners that reload a
# checkpoint or drive the model live (e_step_belief_trace,
# per_unit_eval_nats, ...) live in vfe3/viz/extract.py, NOT here, so this
# module keeps its "pure measurement" contract.
# ===========================================================================

def _is_full_cov(
    sigma:    torch.Tensor,              # (..., K) diagonal OR (..., K, K) full
    diagonal: Optional[bool] = None,
) -> bool:
    r"""Whether ``sigma`` is a full (..., K, K) covariance vs a diagonal (..., K) variance vector.

    Explicit ``diagonal`` wins; otherwise inferred from a trailing SQUARE pair (last two axes
    equal and ndim >= 2). The diagonal default of this codebase is (..., K), so the only
    ambiguous case is a coincidental K == leading-dim; pass ``diagonal=`` to disambiguate.
    """
    if diagonal is not None:
        return not diagonal
    return sigma.dim() >= 2 and sigma.shape[-1] == sigma.shape[-2]


def _spectrum(
    sigma:    torch.Tensor,              # (..., K) diagonal variances OR (..., K, K) full covariance

    *,
    diagonal: Optional[bool] = None,
    eps:      float          = 1e-12,
    family:   Optional[str]  = None,
) -> torch.Tensor:                       # (..., K) eigenvalue spectrum
    r"""The covariance spectrum: the variances themselves (diagonal) or ``eigvalsh`` (full)."""
    if family is not None:
        from vfe3.families.base import get_family
        return get_family(family).diagnostic_statistics(
            sigma,
            eps=eps,
        )["covariance_spectrum"]
    if _is_full_cov(sigma, diagonal):
        return safe_eigvalsh(0.5 * (sigma + sigma.transpose(-1, -2)))
    return sigma


def _covariance_eigenvalue_floor(
    family: Optional[str],
    eps:    float,
) -> float:
    r"""Covariance-unit floor corresponding to the configured stored-dispersion floor."""
    if family is None:
        return eps
    from vfe3.families.base import get_family
    return get_family(family).covariance_floor(eps)


def _effective_rank_denominator_floor(
    family: Optional[str],
    eps:    float,
) -> float:
    r"""Family-owned denominator stabilizer, including legacy Gaussian behavior."""
    if family is None:
        return eps
    from vfe3.families.base import get_family
    return get_family(family).effective_rank_floor(eps)


def _family_effective_rank(
    spectrum: torch.Tensor,

    *,
    family:   Optional[str],
    eps:      float,
) -> torch.Tensor:
    r"""Participation ratio with opt-in scale normalization for tiny family spectra."""
    if family is not None:
        from vfe3.families.base import get_family
        if get_family(family).effective_rank_rescale:
            lam = spectrum.float().clamp(min=0.0)
            scale = lam.amax(dim=-1, keepdim=True)
            normalized = lam / scale.clamp(min=torch.finfo(lam.dtype).tiny)
            return effective_rank(
                normalized,
                eps=torch.finfo(normalized.dtype).tiny,
            ).to(spectrum.dtype)
    return effective_rank(
        spectrum,
        eps=_effective_rank_denominator_floor(family, eps),
    )


def effective_rank_per_token(
    sigma:    torch.Tensor,              # (..., K) diagonal OR (..., K, K) full covariance

    *,
    diagonal: Optional[bool] = None,
    eps:      float = 1e-12,
    family:   Optional[str] = None,
) -> torch.Tensor:                       # (...) per-token effective rank (NOT mean-reduced)
    r"""Per-token spectral effective rank (sum lam)^2 / sum lam^2, the (...) VECTOR.

    The registered ``effective_rank`` metric reduces this to a single ``.mean()`` per eval; this
    keeps the per-token distribution a single-seed run needs for a ridgeline/violin. Full
    covariances are passed through ``eigvalsh`` first.
    """
    return _family_effective_rank(
        _spectrum(sigma, diagonal=diagonal, eps=eps, family=family),
        family=family,
        eps=eps,
    )


def belief_spectrum(
    sigma:    torch.Tensor,              # (..., K) diagonal OR (..., K, K) full covariance

    *,
    diagonal: Optional[bool] = None,
    eps:      float = 1e-12,
    family:   Optional[str] = None,
) -> Dict[str, Any]:
    r"""Per-token spectral picture of the belief covariances (all PER-TOKEN, not mean-reduced).

    Returns ``eigenvalues`` (..., K) sorted DESCENDING, the spectral condition number
    ``condition`` = lam_max / lam_min (...), the ``effective_rank`` (...), and an explicit
    ``is_positive_definite`` flag. Non-positive eigenvalues are retained for diagnosis and force
    the condition number to +inf; only the effective-rank calculation projects them to zero.
    """
    lam = _spectrum(sigma, diagonal=diagonal, eps=eps, family=family)
    lam_desc = torch.sort(lam, dim=-1, descending=True).values
    lam_max = lam_desc[..., 0]
    lam_min = lam_desc[..., -1]
    is_positive_definite = lam_min > 0.0
    covariance_floor = _covariance_eigenvalue_floor(family, eps)
    condition = torch.where(
        is_positive_definite,
        lam_max / lam_min.clamp(min=covariance_floor),
        torch.full_like(lam_min, float("inf")),
    )
    result: Dict[str, Any] = {
        "eigenvalues":            lam_desc,
        "condition":              condition,
        # lam_min alongside the ratio (audit 2026-08-06 F16). `condition` divides by
        # lam_min.clamp(min=covariance_floor), so once an eigenvalue reaches the floor the ratio
        # stops measuring the belief and starts reporting lam_max/eps -- a readout of the GUARD.
        # Worked example from 183.05's final row: guard_sigma_floor_frac 0.003125 x 1280 = exactly 4
        # floored eigenvalues, and belief_cond_max 1453861.5 x 1e-6 = 1.4538615, i.e. a plausible
        # lam_max divided by the floor. Read `condition` as invalid whenever floor_frac > 0; lam_min
        # is what tells you which regime you are in.
        "lam_min":                lam_min,
        "lam_max":                lam_max,
        # The validity flag FOR ``condition``, carried on the same tensor rather than left to a
        # cross-reference (audit 2026-08-06 F16). ``condition`` is a belief property only where
        # lam_min is strictly above the floor; at or below it the ratio is lam_max/floor and says
        # nothing about the belief. Emitting this beside the ratio means a reader of
        # ``belief_cond_p95`` does not have to know that ``guard_sigma_floor_frac`` exists, is
        # denominated over K*N, and is the column that invalidates this one.
        "condition_is_belief":    lam_min > covariance_floor,
        "effective_rank":         _family_effective_rank(
            lam.clamp(min=0.0),
            family=family,
            eps=eps,
        ),
        "is_positive_definite":   is_positive_definite,
    }
    if family is not None:
        from vfe3.families.base import get_family
        labels = get_family(family).diagnostic_labels()
        result["dispersion_label"] = labels["dispersion"]
        result["spectrum_label"] = labels["covariance_spectrum"]
    return result


def half_fisher_trace(
    sigma:    torch.Tensor,              # (..., K) diagonal OR (..., K, K) full covariance

    *,
    eps:      float = 1e-12,
    diagonal: Optional[bool] = None,
    family:   Optional[str] = None,
) -> torch.Tensor:                       # (...) per-token one-half mean-block Fisher trace
    r"""Per-token KL quadratic coefficient, one-half tr(Sigma^{-1}) in the mean block.

    For a Gaussian N(mu, Sigma), the mean-block Fisher information is Sigma^{-1}. The
    second-order KL coefficient is one-half its trace: sum_k 1 / (2 sigma_k) for diagonal
    covariance and (1/2) tr(Sigma^{-1}) for full covariance. This is a precision diagnostic,
    not the full Fisher trace.
    """
    if family is not None:
        from vfe3.families.base import get_family
        family_cls = get_family(family)
        precision = family_cls.mean_fisher_precision(sigma, eps=eps)
        if family_cls.cov_kind == "full":
            precision_trace = torch.diagonal(
                precision,
                dim1=-2,
                dim2=-1,
            ).sum(dim=-1)
        else:
            precision_trace = precision.sum(dim=-1)
        return 0.5 * precision_trace
    if _is_full_cov(sigma, diagonal):
        sym = 0.5 * (sigma + sigma.transpose(-1, -2))
        # eps ridge floors the inverse like the diagonal branch's clamp (audit 2026-06-09 IG1):
        # a singular belief covariance must not crash the figure path.
        eye = torch.eye(sym.shape[-1], device=sym.device, dtype=sym.dtype)
        inv = torch.linalg.inv(sym + eps * eye)
        return 0.5 * torch.diagonal(inv, dim1=-2, dim2=-1).sum(dim=-1)
    return (0.5 / sigma.clamp(min=eps)).sum(dim=-1)


fisher_trace = half_fisher_trace


def sigma_trace(
    sigma:    torch.Tensor,              # (..., K) diagonal OR (..., K, K) full covariance

    *,
    diagonal: Optional[bool] = None,
    family:   Optional[str] = None,
) -> torch.Tensor:                       # (...) per-token tr(Sigma_q) = sum_k Var_k
    r"""Per-token covariance trace tr(Sigma_q) = sum_k Var_k -- the total belief UNCERTAINTY.

    The complement of ``half_fisher_trace`` (the PRECISION coefficient tr(Sigma^{-1})/2): for the
    Sigma_q-as-calibrated-uncertainty probe (B1/EXP-3) the load-bearing quantity is the variance
    trace itself, whose across-token spread (see ``cv``) gates whether the channel carries any
    decode-time signal. Diagonal: sum_k sigma_k. Full: sum_k Sigma_kk.
    """
    if family is not None:
        from vfe3.families.base import get_family
        return get_family(family).covariance_diagonal(sigma).sum(dim=-1)
    if _is_full_cov(sigma, diagonal):
        return torch.diagonal(sigma, dim1=-2, dim2=-1).sum(dim=-1)
    return sigma.sum(dim=-1)


def covariance_frozen_at_encode(cfg: object) -> bool:
    r"""Whether the stack's output covariance is the encode-time token lookup Sigma(x_i).

    No E-step covariance update runs on a projected ``attention_mode`` (the predictive readout keeps
    ``sigma = belief.sigma`` unless ``predictive_covariance='moment_matched'`` assigns the
    predictive kernel's context-dependent covariance; relational filtering has no sigma arm) or under
    ``skip_belief_sigma_update=True`` on the transported route. After the E-step only the mean-dependent delta-method seams
    rewrite sigma -- a block MLP with ``block_mlp_covariance='delta_full'`` or CG coupling with
    ``cg_covariance_mode='delta_full'``; the head mixer is a fixed per-token congruence and the
    norms act on the mean alone. Without those seams tr Sigma_q(x_i) is a function of the input
    token id x_i only, so the B1/EXP-3 statistics measure input-token variance against next-token
    CE, not context-dependent belief uncertainty.
    """
    attention_mode = getattr(cfg, "attention_mode", "transported")
    if attention_mode == "transported":
        e_step_frozen = bool(getattr(cfg, "skip_belief_sigma_update", False))
    elif attention_mode == "projected_predictive":
        # The predictive readout rewrites sigma only under the moment-matched covariance.
        e_step_frozen = getattr(cfg, "predictive_covariance", "passthrough") != "moment_matched"
    else:
        e_step_frozen = True
    mlp_rewrites_sigma = (bool(getattr(cfg, "use_block_mlp", False))
                          and getattr(cfg, "block_mlp_covariance", "passthrough") == "delta_full")
    cg_rewrites_sigma = (bool(getattr(cfg, "use_cg_coupling", False))
                         and getattr(cfg, "cg_covariance_mode", "passthrough") == "delta_full")
    return e_step_frozen and not (mlp_rewrites_sigma or cg_rewrites_sigma)


def rank_one_residual(
    mu:  torch.Tensor,                   # (..., N, K) per-token belief means of ONE layer

    *,
    eps: float = 1e-12,
) -> torch.Tensor:                       # (...) relative distance of the token cloud from rank-one
    r"""Dong rank-one residual r(X) = ||X - 1 xbar^T||_F / ||X||_F over the (N, K) mean matrix.

    Measures how far the per-token mean cloud X (rows = tokens) sits from the rank-one matrix
    1 xbar^T (every token collapsed to the mean token xbar = mean_n X[n,:]): r -> 0 is total rank
    collapse (all tokens identical), r near 1 is full spread. This is the anti-rank-collapse /
    FFN-brake object of F2/EXP-7, computed on the MEAN stream -- DISTINCT from ``effective_rank``,
    which is the spectral rank of the per-token COVARIANCE Sigma (a different object). Reference:
    Dong et al. 2021, "Attention is not all you need: pure attention loses rank doubly exponentially".
    """
    xbar = mu.mean(dim=-2, keepdim=True)                                  # (..., 1, K) mean token
    num = torch.linalg.norm((mu - xbar).flatten(-2, -1), dim=-1)          # ||X - 1 xbar^T||_F
    den = torch.linalg.norm(mu.flatten(-2, -1), dim=-1).clamp(min=eps)    # ||X||_F
    return num / den


def depth_decay_rate(
    curve: torch.Tensor,                 # (L,) a per-layer scalar (e.g. r(X) by depth)

    *,
    eps:   float = 1e-12,
) -> float:                              # log-linear slope d log(curve) / d layer
    r"""Log-linear decay rate (slope of log(curve) vs layer index) of a per-depth scalar.

    For F2/EXP-7 the per-arm rank-residual curves are compared by their DECAY RATE, not absolute
    level (the no-anchor control plateaus rather than collapsing to rank-one). Fits
    log(curve) ~ a + b * layer by least squares and returns b (negative = decaying with depth).
    """
    y = torch.log(curve.flatten().clamp(min=eps).to(torch.float64))
    n = y.shape[0]
    if n < 2:
        raise ValueError(f"depth_decay_rate needs >= 2 layers, got {n}")
    x = torch.arange(n, dtype=torch.float64, device=y.device)
    xm, ym = x.mean(), y.mean()
    return float(((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum().clamp(min=eps))


def _average_ranks(
    values: torch.Tensor,                # (M,) finite sample
) -> torch.Tensor:                       # (M,) zero-based fractional ranks
    r"""Average ranks for a finite one-dimensional sample, assigning one rank to each tie group."""
    order = torch.argsort(values)
    sorted_values = values[order]
    _, inverse, counts = torch.unique_consecutive(
        sorted_values,
        return_inverse=True,
        return_counts=True,
    )
    ends = counts.cumsum(dim=0).to(torch.float64)
    starts = ends - counts.to(torch.float64)
    average = 0.5 * (starts + ends - 1.0)
    ranks = torch.empty_like(values, dtype=torch.float64)
    ranks[order] = average[inverse]
    return ranks


def spearman_rho(
    x: torch.Tensor,                     # (M,) sample
    y: torch.Tensor,                     # (M,) sample

    *,
    eps: float = 1e-12,
) -> float:                              # Spearman rank correlation in [-1, 1]
    r"""Spearman rank correlation: the Pearson correlation of the RANKS of x and y.

    The headline statistic of the Sigma_q-calibration probe (B1/EXP-3): rho(tr Sigma_q, per-token
    CE). Non-finite observations are removed by a paired mask, and ties receive their fractional
    average rank. Fewer than two paired-finite observations return NaN. A finite constant input has
    zero rank variance and retains the established 0.0 convention.
    """
    x, y = x.flatten().to(torch.float64), y.flatten().to(torch.float64)
    if x.numel() != y.numel():
        raise ValueError(f"spearman_rho needs equal-length inputs, got {x.numel()} vs {y.numel()}")
    paired_finite = torch.isfinite(x) & torch.isfinite(y)
    x, y = x[paired_finite], y[paired_finite]
    if x.numel() < 2:
        return float("nan")
    rx = _average_ranks(x)
    ry = _average_ranks(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denominator = rx.norm() * ry.norm()
    correlation = (rx * ry).sum() / denominator.clamp(min=eps)
    return float(correlation.clamp(min=-1.0, max=1.0))


def cv(
    x: torch.Tensor,                     # (M,) sample

    *,
    eps: float = 1e-12,
) -> float:                              # coefficient of variation std / |mean|
    r"""Coefficient of variation std(x) / |mean(x)| -- the pre-registered spread gate for B1/EXP-3.

    The calibration experiment requires CV(tr Sigma_q) > 0.10: below it the covariance carries no
    across-token signal and the result is reported "covariance inert", not miscoded as "decode
    does not matter". Uses the unbiased std (ddof=1).
    """
    x = x.flatten().to(torch.float64)
    if x.numel() < 2:
        raise ValueError(f"cv needs >= 2 samples, got {x.numel()}")
    return float(x.std(unbiased=True) / x.mean().abs().clamp(min=eps))


def spd_geodesic_distance(
    sigma_a:  torch.Tensor,              # (..., K) diagonal OR (..., K, K) full covariance
    sigma_b:  torch.Tensor,              # (..., K) diagonal OR (..., K, K) full covariance

    *,
    diagonal: Optional[bool] = None,
    eps:      float = 1e-12,
) -> torch.Tensor:                       # (...) affine-invariant SPD distance d_AI(Sigma_a, Sigma_b)
    r"""Affine-invariant Riemannian geodesic distance on the SPD cone.

    d_AI(Sigma_a, Sigma_b) = ||log(Sigma_a^{-1/2} Sigma_b Sigma_a^{-1/2})||_F
                           = sqrt(sum_k (log lambda_k)^2),
    where lambda_k are the generalized eigenvalues of (Sigma_b, Sigma_a). This is the standard
    affine-invariant SPD metric without the Gaussian covariance Fisher block's factor of one half;
    it must therefore not be labeled as the Fisher-Rao distance. For diagonal beliefs
    this reduces to sqrt(sum_k log^2(sigma_b,k / sigma_a,k)). This is the metric the SPD
    retraction itself uses, so belief-trajectory / E-step-residual lengths are measured in the
    geometry the inference actually moves in. Symmetric, zero iff Sigma_a == Sigma_b.

    The full-covariance reduction factors Sigma_a = L L^T in float64 and obtains the generalized
    eigenvalues from the symmetric matrix L^{-1} Sigma_b L^{-T} using triangular solves. It never
    materializes Sigma_a^{-1/2}.
    """
    if not _is_full_cov(sigma_a, diagonal):
        ratio = sigma_b.clamp(min=eps) / sigma_a.clamp(min=eps)
        return torch.sqrt((torch.log(ratio) ** 2).sum(dim=-1).clamp(min=0.0))
    output_dtype = torch.promote_types(sigma_a.dtype, sigma_b.dtype)
    a = sigma_a.to(torch.float64)
    b = sigma_b.to(torch.float64)
    a = 0.5 * (a + a.transpose(-1, -2))
    b = 0.5 * (b + b.transpose(-1, -2))
    chol_a = torch.linalg.cholesky(a)
    left = torch.linalg.solve_triangular(chol_a, b, upper=False)
    generalized = torch.linalg.solve_triangular(
        chol_a, left.transpose(-1, -2), upper=False).transpose(-1, -2)
    generalized = 0.5 * (generalized + generalized.transpose(-1, -2))
    lam = safe_eigvalsh(generalized).clamp(min=eps)
    distance = torch.sqrt((torch.log(lam) ** 2).sum(dim=-1).clamp(min=0.0))
    return distance.to(dtype=output_dtype)


def attention_entropy_rows(
    beta: torch.Tensor,                  # (..., N, N) row-stochastic attention weights

    *,
    eps:  float = 1e-12,
) -> torch.Tensor:                       # (..., N) per-row (per query i) Shannon entropy
    r"""Per-row attention entropy H_i = -sum_j beta_ij log beta_ij, keeping every leading axis.

    The registered ``attention_entropy`` collapses heads, layers, and query rows into one float;
    this keeps the per-(layer, head, query) distribution a single-seed run must lean on. By
    construction ``attention_entropy_rows(beta).mean() == attention_entropy(beta)``. Shared by the
    attention-structure and gauge-specialization figures.
    """
    b = beta.clamp(min=eps)
    return -(b * torch.log(b)).sum(dim=-1)


COLLAPSE_ENTROPY_FRACTION: float = 0.5   # a head is "collapsed" below half its achievable entropy


def attention_head_collapse(
    beta: torch.Tensor,                  # (..., N, N) row-stochastic attention weights

    *,
    eps:      float = 1e-12,
    fraction: float = COLLAPSE_ENTROPY_FRACTION,
) -> Tuple[torch.Tensor, torch.Tensor]:  # (..., ) min normalized row entropy, (...) collapsed flag
    r"""Per-head minimum row entropy as a FRACTION of that row's achievable maximum.

    Row ``i`` of a causal map admits only ``s_i`` keys, so its entropy CANNOT exceed ``ln s_i``.
    Comparing a raw row entropy against a fixed threshold therefore measures the mask, not the head:
    under ``causal_alibi_noself`` the first scorable row admits exactly two keys, so its ceiling
    **is** ``ln 2`` -- the very threshold the previous test used -- and the strict ``<`` fired on
    float noise. Measured on an untrained model whose attention is as uniform as the mask permits:
    row entropies ``0.693146884`` and ``0.693145990`` against ``ln 2 = 0.693147181``, i.e. margins
    of 3e-7 and 1e-6, flagging 2 of 2 heads as collapsed (audit 2026-08-06 F22). The 2026-07-27 and
    2026-08-05 fixes moved that pin from row 0 to the first two-key row rather than removing it.

    Normalizing by ``ln s_i`` makes every scorable row commensurate: 1.0 is the most uniform the
    mask allows and 0.0 is one-hot, so a threshold expresses "this head uses less than ``fraction``
    of the entropy available to it" at any row width. Rows with fewer than two admissible keys carry
    no entropy at all and are excluded rather than scored as collapsed.

    Returns ``(min_normalized_entropy, collapsed)`` with the row axis reduced. A leading axis with
    no scorable row at all yields NaN and False, so ``N == 1`` and fully one-hot maps abstain
    instead of raising a false alarm.
    """
    support = (beta > 0.0).sum(dim=-1)                       # (..., N) admissible keys per row
    scorable = support >= 2
    rows = attention_entropy_rows(beta, eps=eps)             # (..., N) raw nats
    ceiling = torch.log(support.clamp(min=2).to(rows.dtype))
    normalized = (rows / ceiling).masked_fill(~scorable, float("inf"))
    worst = normalized.min(dim=-1).values                    # (...) per head/layer
    has_scorable = scorable.any(dim=-1)
    worst = torch.where(has_scorable, worst, torch.full_like(worst, float("nan")))
    return worst, (worst < fraction) & has_scorable


def _ols_slope(
    y: torch.Tensor,                     # (..., M) response sampled at x = 0..M-1
) -> torch.Tensor:                       # (...) least-squares slope over the last axis
    r"""Ordinary-least-squares slope of ``y`` against the integer index x = 0..M-1 (last axis)."""
    m = y.shape[-1]
    x = torch.arange(m, device=y.device, dtype=y.dtype)
    x = x - x.mean()
    y = y - y.mean(dim=-1, keepdim=True)
    denom = (x * x).sum().clamp(min=1e-12)
    return (y * x).sum(dim=-1) / denom


def causal_sanity(
    beta:   torch.Tensor,                # (..., N, N) attention weights (rows = query i, cols = key j)

    *,
    active: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    r"""Causal-mask and row-stochastic correctness scalars, keeping leading (layer/head) axes.

    Returns ``future_leakage`` = max_{j>i} beta_ij (must be ~0 under a causal mask),
    ``row_sum_error`` = max_i |sum_j beta_ij - 1| (row-stochasticity), and ``active_set_slope``
    = OLS slope of the per-row active-key count (beta_ij > ``active``) against query index i
    (expected ~1 as the causal active set grows by one key per step). Each reduces the (N, N)
    pair block and keeps any leading axes, so a (..., H, N, N) stack yields per-head (...,) values.
    """
    n = beta.shape[-1]
    future = torch.triu(torch.ones(n, n, device=beta.device, dtype=torch.bool), diagonal=1)
    leak = torch.where(future, beta, torch.zeros_like(beta)).amax(dim=(-2, -1))
    row_sum_err = (beta.sum(dim=-1) - 1.0).abs().amax(dim=-1)
    counts = (beta > active).sum(dim=-1).to(beta.dtype)              # (..., N) active keys per row
    return {
        "future_leakage":  leak,
        "row_sum_error":   row_sum_err,
        "active_set_slope": _ols_slope(counts),
    }


def guard_saturation(
    sigma:     torch.Tensor,             # (..., K) diagonal OR (..., K, K) full covariance
    energy:    torch.Tensor,             # (..., N, N) pairwise energies E_ij
    self_div:  torch.Tensor,             # (..., N) self-divergences D(q_i || p_i)

    *,
    diagonal:  Optional[bool] = None,
    eps:       float = 1e-6,
    sigma_max: Optional[float] = 5.0,
    kl_max:    float = 100.0,
    rtol:      float = 1e-3,
    log_prior: Optional[torch.Tensor] = None,   # (..., N, N) additive prior; -inf marks masked pairs
) -> Dict[str, float]:
    r"""Fraction of converged-belief entries pinned at each numerical guard boundary.

    Certifies whether the SPD variance floor/ceiling (``eps``, ``sigma_max``) and the KL clamps
    (``kl_max`` on E_ij and on D(q||p)) are INERT on the pure path or load-bearing. sat_frac =
    mean(|x - boundary| < rtol * |boundary|) over the relevant entries; the variance spectrum is
    the eigenvalues for a full covariance. A near-zero map means the clamps never bind.

    ``energy_klmax_frac`` denominates over the WHOLE ``(..., N, N)`` grid, masked pairs included,
    and that is deliberate: a masked pair is not numerically inert, because ``-inf + NaN`` would
    poison its whole softmax row, and the ``kl_max`` clamp is exactly the firewall this metric
    certifies. Passing ``log_prior`` adds ``energy_klmax_frac_live`` restricted to the pairs the
    causal mask actually admits (``isfinite(log_prior)``), WITHOUT changing the existing column
    (audit 2026-08-06 F17).

    Both are needed because the obvious "the denominator is 2x too big" correction is probably
    sign-wrong: across seven runs on disk ``energy_klmax_frac * 8192`` peaks at 4163 against a
    masked-set cardinality of 4158, the signature of the MASKED triangle saturating while the live
    triangle does not. Re-denominatoring to the live half alone would have doubled an alarm that
    is already overstated. Reporting both is what settles it.
    """
    spec = _spectrum(sigma, diagonal=diagonal)
    # eigvalsh error scales with the largest eigenvalue (~eps_machine * lam_max), so on a FULL
    # covariance the tight relative window rtol*eps at the SPD FLOOR (eps ~ 1e-6) sits far below the
    # eigensolver noise -- which would report sigma_floor as NEVER binding even when it binds on every
    # token. Widen the spectrum-boundary tolerance to that noise floor on the full-cov path only; the
    # diagonal path (_spectrum returns the variances bit-exact) keeps the tight relative window.
    spec_atol = (64.0 * 2.0 ** -23 * float(spec.detach().abs().max())
                 if (_is_full_cov(sigma, diagonal) and spec.numel()) else 0.0)

    def _frac(x: torch.Tensor, boundary: float, *, atol: float = 0.0) -> float:
        if x.numel() == 0:
            return 0.0
        scale = max(abs(boundary), eps)
        return float(((x - boundary).abs() < max(rtol * scale, atol)).float().mean())

    out = {
        "sigma_floor_frac":   _frac(spec, eps, atol=spec_atol),
        "sigma_ceil_frac":    (_frac(spec, sigma_max, atol=spec_atol)
                                if sigma_max is not None else 0.0),
        "energy_klmax_frac":  _frac(energy, kl_max),
        "selfdiv_klmax_frac": _frac(self_div, kl_max),
    }
    if log_prior is not None:
        live = torch.isfinite(log_prior).expand_as(energy)
        out["energy_klmax_frac_live"] = _frac(energy[live], kl_max)
        out["energy_live_pair_frac"] = float(live.float().mean())
    return out


# --- per-token frame class functions (group-dispatched; fixes gauge_trace_spread's det-blindness) ---
#
# NOT gauge invariants of the declared local action. The model's gauge acts on vertex frames on the
# LEFT, U_i -> g_i U_i with mu_i -> g_i mu_i, and on the flat coboundary Omega_ij = U_i U_j^{-1} the
# local choice g_i = U_i^{-1} sends every frame to I (a pure-gauge connection), so no nonconstant
# function of a single vertex frame is invariant under it. The functions below are conjugation-class
# functions: invariant under one GLOBAL conjugation U_i -> g U_i g^{-1}, i.e. statistics of the frame
# in the model's reference trivialization. Their metric keys ('gauge_invariant_spread',
# 'gauge_head_*') are historical names kept for artifact compatibility.

def group_gauge_invariant(
    exp_phi: torch.Tensor,               # (..., K, K) per-token vertex factor exp(embed(phi_i))
    group,                               # GaugeGroup (dispatches the class function)

    *,
    eps:     float = 1e-12,
) -> torch.Tensor:                       # (...) per-token scalar conjugation-class function
    r"""Per-token frame class function, dispatched on the group so it is non-vacuous for every group.

    Invariant under a global conjugation exp(phi_i) -> g exp(phi_i) g^{-1}, NOT under the declared
    local left action U_i -> g_i U_i (see the section note above): it is a reference-trivialization
    frame statistic, not a gauge-invariant observable, despite the historical function name.

    The logged ``gauge_trace_spread`` (= std of log|det Omega| = tr embed(phi)) is identically 0
    for the unimodular groups SO(K) and Sp(2m) (traceless generators, det == 1), so it is blind
    off block_glk. This dispatches the correct invariant: GL volume log|det exp(phi)| for
    glk / block_glk / tied_block_glk, total rotation angle (1/2) sum_k |arg(eig)| for so_k and
    the orthogonal so_n irrep towers, and the eigenvalue-modulus squeeze max_k log|lambda_k| -
    min_k log|lambda_k| for sp / sp_n (whose Sym^p images are traceless, so logdet is blind there
    too; the eigenvalue moduli are conjugation-invariant, unlike the singular values).
    """
    name = getattr(group, "name", "glk")
    if name in ("so_k", "so_n"):
        ang = torch.angle(torch.linalg.eigvals(exp_phi))         # (..., K) eigenphases
        return 0.5 * ang.abs().sum(dim=-1)
    if name in ("sp", "sp_n"):
        # The target symmetry is a GLOBAL conjugation g exp(phi) g^{-1} (see the section note: no
        # vertex-frame function survives the local left action); the singular-value squeeze
        # log(s_max/s_min) is invariant only under ORTHOGONAL conjugation, and
        # Sp(2m,R) is not orthogonal. Use the eigenvalue-MODULUS squeeze instead -- eigenvalues are
        # conjugation-invariant and symplectic spectra come in reciprocal pairs, so this is the natural
        # Ad-invariant squeeze (was frame-dependent under svdvals; audit 2026-06-13 review).
        logmod = torch.log(torch.linalg.eigvals(exp_phi).abs().clamp(min=eps))   # (..., K) log|eig|
        return logmod.amax(dim=-1) - logmod.amin(dim=-1)
    return torch.linalg.slogdet(exp_phi).logabsdet              # GL volume log|det|


def per_head_gauge_invariants(
    exp_phi:    torch.Tensor,            # (..., K, K) per-token vertex factor exp(embed(phi_i))
    irrep_dims: List[int],               # gauge-irrep block sizes; sum == K

    *,
    eps:        float = 1e-12,
) -> Dict[str, torch.Tensor]:
    r"""Per-head, per-token GL(d_head) conjugation-class functions of the converged vertex factor.

    Invariant under a global in-group conjugation, NOT under the declared local left action
    U_i -> g_i U_i (see the section note above), so these are reference-trivialization frame
    statistics; the function name and the ``gauge_head_*`` keys are historical.

    For a block-diagonal group the head-h frame is the (d_head, d_head) diagonal block of
    ``exp_phi``; returns its log-volume ``logdet`` (..., H) and the eigenvalue-modulus squeeze
    ``anisotropy`` = max_k log|lambda_k| - min_k log|lambda_k| (..., H). A single-block group
    (``irrep_dims = [K]``) yields H = 1. Feeds the per-head gauge-specialization ridgelines.

    ``anisotropy`` WAS the singular-value ratio ``s_max / s_min`` (audit 2026-07-27). That is
    invariant only under ORTHOGONAL conjugation, while the symmetry these statistics target is a
    global GL(K) conjugation ``g exp(phi) g^{-1}`` -- the same defect this module already corrected for ``sp`` in
    ``group_gauge_invariant`` above, and for the same reason, but the per-head twin was left
    behind on ``block_glk``, the group the live config actually runs. Measured at block_glk K=4 in
    float64 under an in-group conjugation drawn from the group's own generators: ``logdet`` moved by
    3.7e-15 and the eigenvalue-modulus squeeze by 8.3e-15, while the SVD ratio moved by up to 301%.
    Both returned entries are genuine conjugacy-class functions; that makes them invariant under a
    global conjugation only, which is all the ``gauge_head_*`` keys and the
    ``per_head_gauge_invariants`` name can claim.

    The value is a LOG ratio where the old one was a bare ratio, so the published
    ``gauge_head_aniso_mean`` column changes scale: it is now 0 at an isometric block and grows
    logarithmically, rather than 1 and growing linearly. ``vertex_cond_*`` remains the
    frame-dependent conditioning probe, honestly named, for anyone who wants the old quantity.
    """
    logdets, anisos = [], []
    start = 0
    for d in irrep_dims:
        blk = exp_phi[..., start:start + d, start:start + d]      # (..., d, d) head block
        logdets.append(torch.linalg.slogdet(blk).logabsdet)
        logmod = torch.log(torch.linalg.eigvals(blk).abs().clamp(min=eps))   # (..., d) log|eig|
        anisos.append(logmod.amax(dim=-1) - logmod.amin(dim=-1))
        start += d
    return {
        "logdet":     torch.stack(logdets, dim=-1),              # (..., H)
        "anisotropy": torch.stack(anisos, dim=-1),               # (..., H)
    }


# --- transport / energy directedness (the ln(3) symmetry-breaking signal) ---

def _require_square_token_transport(
    omega:       'torch.Tensor | CompactFactoredTransport',
    metric_name: str,
) -> int:
    r"""Return N after rejecting rectangular query/key transport at graph-metric boundaries."""
    if isinstance(omega, CompactFactoredTransport):
        n_query = omega.exp_blocks.shape[-4]
        n_key = omega.inv_blocks.shape[-4]
    else:
        if omega.dim() < 4:
            raise ValueError(
                f"{metric_name} requires (..., Nq, Nk, K, K) transport layout, got "
                f"{tuple(omega.shape)}")
        if omega.shape[-2] != omega.shape[-1]:
            raise ValueError(
                f"{metric_name} requires trailing square K x K matrix axes, got "
                f"{tuple(omega.shape[-2:])} in shape {tuple(omega.shape)}")
        n_query = omega.shape[-4]
        n_key = omega.shape[-3]
    if n_query != n_key:
        raise ValueError(
            f"{metric_name} requires a square token transport (Nq == Nk); "
            f"got Nq={n_query}, Nk={n_key}")
    return n_query


def _require_unbatched_square_token_transport(
    omega:       'torch.Tensor | CompactFactoredTransport',
    metric_name: str,
) -> int:
    r"""Validate the unbatched square graph consumed by sampled triangle metrics."""
    if isinstance(omega, CompactFactoredTransport):
        if omega.exp_blocks.dim() != 4 or omega.inv_blocks.dim() != 4:
            raise ValueError(
                f"{metric_name} requires unbatched compact (N, H, d, d) factors; got "
                f"{tuple(omega.exp_blocks.shape)} and {tuple(omega.inv_blocks.shape)}")
    elif omega.dim() != 4:
        raise ValueError(
            f"{metric_name} requires an unbatched dense (N, N, K, K) transport; got "
            f"shape {tuple(omega.shape)}")
    return _require_square_token_transport(omega, metric_name)


def transport_asymmetry(
    omega: 'torch.Tensor | CompactFactoredTransport',
) -> torch.Tensor:                       # (..., N, N) A_ij = ||Omega_ij - Omega_ji||_F
    r"""Directedness of the transport: A_ij = ||Omega_ij - Omega_ji||_F.

    Zero when the gauge is off (phi = 0 so every Omega_ij = I, hence symmetric); structured once
    the learned directed transport Omega_ij = exp(phi_i) exp(-phi_j) breaks the i<->j averaging
    symmetry. Omega_ji is the swap of the two token axes (not a matrix transpose).
    """
    _require_square_token_transport(omega, "transport_asymmetry")
    if isinstance(omega, CompactFactoredTransport):
        blocks = torch.einsum(
            "...ihkl,...jhlm->...ijhkm", omega.exp_blocks, omega.inv_blocks)
        delta = blocks - blocks.transpose(-5, -4)                # swap i/j; keep H,d,d
        return delta.square().sum(dim=(-3, -2, -1)).sqrt()
    omega_ji = omega.transpose(-4, -3)                           # swap the i and j token axes
    return torch.linalg.norm(omega - omega_ji, dim=(-2, -1))


def _compact_sampled_pairs(
    omega: CompactFactoredTransport,
    left:  torch.Tensor,                  # (T,) query indices
    right: torch.Tensor,                  # (T,) key indices
) -> torch.Tensor:                        # (T,H,d,d) sampled Omega_left,right blocks
    r"""Sample pairwise block operators without materializing ``(N,N,K,K)`` or dense ``K``."""
    if omega.exp_blocks.dim() != 4:
        raise ValueError("sampled compact transport metrics require unbatched (N,H,d,d) factors")
    return torch.einsum(
        "thkl,thlm->thkm", omega.exp_blocks[left], omega.inv_blocks[right])


def energy_directedness(
    energy: torch.Tensor,                # (..., N, N) pairwise energies E_ij

    *,
    eps:    float = 1e-12,
) -> Dict[str, torch.Tensor]:
    r"""Asymmetry of the pre-softmax pairwise energy E_ij over off-diagonal pairs.

    Returns the mean absolute asymmetry mean|E_ij - E_ji| and its scale-free normalization
    mean(|E_ij - E_ji| / (E_ij + E_ji)); both are 0 for symmetric (frozen-gauge) transport and
    grow once Omega(phi) is directional.
    """
    n = energy.shape[-1]
    off = ~torch.eye(n, dtype=torch.bool, device=energy.device)
    e_t = energy.transpose(-1, -2)
    abs_asym = (energy - e_t).abs()
    rel_asym = abs_asym / (energy + e_t).abs().clamp(min=eps)
    mask = off.expand_as(abs_asym)
    return {
        "abs_asymmetry": abs_asym[mask].mean() if mask.any() else abs_asym.new_zeros(()),
        "rel_asymmetry": rel_asym[mask].mean() if mask.any() else rel_asym.new_zeros(()),
    }


def structured_head_scores(
    beta:        torch.Tensor,           # (..., N, N) attention weights (rows = query i)

    *,
    period:      int = 3,
    band_width:  int = 1,
) -> Dict[str, torch.Tensor]:
    r"""Per-head induction/copy structure: prev-token, period-match, and diagonal-band mass.

    ``prev_token`` = mean_i beta_{i,i-1}; ``period_match`` = mean over causal pairs with i > j and
    (i - j) mod ``period`` == 0 of beta_ij (the period-3 copy structure on the synthetic stream);
    ``diagonal_band`` = mean_i sum_{0 < i-j <= band_width} beta_ij. Each keeps leading (head) axes.
    """
    n = beta.shape[-1]
    ii = torch.arange(n, device=beta.device).unsqueeze(-1)
    jj = torch.arange(n, device=beta.device).unsqueeze(0)
    diff = ii - jj
    causal = diff > 0
    period_mask = causal & (diff % period == 0)
    band_mask = causal & (diff <= band_width)
    prev = torch.diagonal(beta, offset=-1, dim1=-2, dim2=-1).mean(dim=-1)

    def _masked_mean(mask: torch.Tensor) -> torch.Tensor:
        m = mask.to(beta.dtype)
        denom = m.sum().clamp(min=1.0)
        return (beta * m).sum(dim=(-2, -1)) / denom

    return {
        "prev_token":    prev,
        "period_match":  _masked_mean(period_mask),
        "diagonal_band": _masked_mean(band_mask),
    }


# --- attention structure (entropy, head redundancy, distance decay) ---

def head_redundancy_js(
    beta: torch.Tensor,                  # (H, N, N) per-head attention for one layer

    *,
    eps:  float = 1e-12,
) -> torch.Tensor:                       # (H, H) mean row-wise Jensen-Shannon divergence (nats)
    r"""Pairwise Jensen-Shannon divergence between heads' row distributions (head specialization).

    JS(beta^h || beta^h') = mean_i [ (1/2) KL(beta_i^h || m_i) + (1/2) KL(beta_i^h' || m_i) ],
    m_i = (1/2)(beta_i^h + beta_i^h'). High off-diagonal -> specialized heads, low -> redundant.
    """
    b = beta.clamp(min=eps)                                       # (H, N, N)
    bh = b.unsqueeze(1)                                           # (H, 1, N, N)
    bh2 = b.unsqueeze(0)                                          # (1, H, N, N)
    m = 0.5 * (bh + bh2)
    kl1 = (bh * (torch.log(bh) - torch.log(m))).sum(dim=-1)       # (H, H, N)
    kl2 = (bh2 * (torch.log(bh2) - torch.log(m))).sum(dim=-1)
    return (0.5 * (kl1 + kl2)).mean(dim=-1)                       # (H, H)


def attention_distance_decay(
    beta: torch.Tensor,                  # (..., N, N) attention weights
) -> Dict[str, torch.Tensor]:
    r"""Per-head attention-vs-offset profile beta_bar(d) = mean over causal pairs i - j = d.

    Returns ``offsets`` (D,) = 0..N-1 and ``profile`` (..., D), the mean attention at each
    query-key offset d (averaged over the offset-d diagonal). The positional-decay diagnostic;
    figures may bootstrap over rows for a band.
    """
    n = beta.shape[-1]
    profiles = [torch.diagonal(beta, offset=-d, dim1=-2, dim2=-1).mean(dim=-1) for d in range(n)]
    return {
        "offsets": torch.arange(n, device=beta.device),
        "profile": torch.stack(profiles, dim=-1),                # (..., N)
    }


def positional_content_score(
    beta: torch.Tensor,                  # (..., N, N) attention weights

    *,
    eps:  float = 1e-12,
) -> torch.Tensor:                       # (...) per-head R^2 of log beta vs offset |i - j|
    r"""Per-head positional<->content score: R^2 of the OLS fit log(beta_ij) ~ a + b |i - j|.

    Over the causal entries the prior actually ADMITS. R^2 near 1 means attention is explained by
    token DISTANCE (positional, the gauge/RoPE machinery); near 0 means it is content-driven (the
    divergence energy E_ij). Places each head on a positional<->content axis.

    Structurally masked entries are excluded. A prior that forbids a (i, j) pair sends its logit to
    -inf, so beta_ij is exactly 0 and ``log(clamp(0, eps))`` is ``-27.63`` at eps=1e-12 -- an
    outlier two orders of magnitude below any live weight, sitting at a FIXED offset. Including
    them makes the fit measure the mask instead of the attention: with 'causal_alibi_noself' the
    64 masked diagonal entries are 3% of the 2080 causal points but carry 91-96% of ss_tot, and a
    PURE positional prior (R^2 = 1 by construction) scores 1.26e-05 / 0.0315 per head instead of
    0.962 / 0.310. The support is a property of the prior, not of a head, so it is intersected
    across the leading dims to keep the (N, N) mask shape the fit's indexing relies on.
    """
    n = beta.shape[-1]
    ii = torch.arange(n, device=beta.device).unsqueeze(-1)
    jj = torch.arange(n, device=beta.device).unsqueeze(0)
    admitted = (beta > 0).reshape(-1, n, n).all(dim=0)           # prior support, shared (N, N)
    mask = (ii >= jj) & admitted                                 # causal AND not masked out
    if int(mask.sum()) < 2:                                      # degenerate: no slope is defined
        return beta.new_full(beta.shape[:-2], float("nan"))
    x = (ii - jj).to(beta.dtype)[mask]                           # (P,)
    y = torch.log(beta.clamp(min=eps))[..., mask]               # (..., P)
    xc = x - x.mean()
    yc = y - y.mean(dim=-1, keepdim=True)
    slope = (yc * xc).sum(dim=-1) / (xc * xc).sum().clamp(min=eps)
    ss_res = ((yc - slope.unsqueeze(-1) * xc) ** 2).sum(dim=-1)
    ss_tot = (yc ** 2).sum(dim=-1).clamp(min=eps)
    return 1.0 - ss_res / ss_tot


# --- holonomy / curvature (corrected sampling; Regime-II quantity) ---
#
# Every registered, config-reachable transport is the coboundary Omega_ij = U_i U_j^{-1} built from
# one vertex table ('flat', pair_transport_kind='coboundary'). On it the triangle holonomy
# Omega_ij Omega_jk Omega_ki and the cocycle defect Omega_ik - Omega_ij Omega_jk telescope to I and 0
# by construction (pure-gauge links close every Wilson loop), so these diagnostics can detect only
# floating-point roundoff and the conditioning of the stored (U, U^{-1}) pair. They are
# roundoff/consistency monitors of the coboundary construction, NOT flatness certificates; they
# would carry structural information only for a registered non-coboundary transport.

def holonomy_deviation_sampled(
    omega:            'torch.Tensor | CompactFactoredTransport',

    *,
    n_triples:        int  = 512,
    n_boot:           int  = 200,
    seed:             int  = 0,
) -> Dict[str, torch.Tensor]:
    r"""Triangle holonomy ||Omega_ij Omega_jk Omega_ki - I||_F over RANDOM distinct triples.

    Replaces ``holonomy_deviation``'s deterministic first-512 row-major enumeration (which always
    samples the same low-index-token triangles -- a systematically biased curvature estimate)
    with seeded random distinct (i, j, k), returning the full per-triple distribution, a
    bootstrap-over-triples mean CI, and each triple's index span max|.-.|. On the flat phi-cocycle
    every triangle closes (H = I) by construction, so this is ~0 up to roundoff: a numerical
    consistency monitor of the coboundary, not a flatness certificate (see the section note).
    """
    compact = isinstance(omega, CompactFactoredTransport)
    n = _require_unbatched_square_token_transport(omega, "holonomy_deviation_sampled")
    k = omega.K if compact else omega.shape[-1]
    device, dtype = omega.device, omega.dtype
    eye = torch.eye(omega.block_dim if compact else k, device=device, dtype=dtype)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    draw = torch.randint(0, n, (max(n_triples * 3, 12), 3), generator=gen, device=device)
    keep = (draw[:, 0] != draw[:, 1]) & (draw[:, 1] != draw[:, 2]) & (draw[:, 0] != draw[:, 2])
    idx = draw[keep][:n_triples]
    if idx.numel() == 0:
        z = torch.zeros((), device=device, dtype=dtype)
        return {"mean": z, "ci_lo": z, "ci_hi": z,
                "per_triple": torch.zeros(0, device=device, dtype=dtype),
                "span": torch.zeros(0, device=device, dtype=dtype)}
    if compact:
        o_ij = _compact_sampled_pairs(omega, idx[:, 0], idx[:, 1])
        o_jk = _compact_sampled_pairs(omega, idx[:, 1], idx[:, 2])
        o_ki = _compact_sampled_pairs(omega, idx[:, 2], idx[:, 0])
        h = o_ij @ o_jk @ o_ki
        per = (h - eye).square().sum(dim=(-3, -2, -1)).sqrt()
    else:
        h = omega[idx[:, 0], idx[:, 1]] @ omega[idx[:, 1], idx[:, 2]] @ omega[idx[:, 2], idx[:, 0]]
        per = torch.linalg.norm(h - eye, dim=(-2, -1))           # (T,)
    span = (idx.amax(dim=1) - idx.amin(dim=1)).to(dtype)
    ridx = torch.randint(0, per.shape[0], (n_boot, per.shape[0]), generator=gen, device=device)
    boot = per[ridx].mean(dim=1)
    return {
        "mean":       per.mean(),
        "ci_lo":      torch.quantile(boot, 0.025),
        "ci_hi":      torch.quantile(boot, 0.975),
        "per_triple": per,
        "span":       span,
    }


def holonomy_wilson_sampled(
    omega:      'torch.Tensor | CompactFactoredTransport',

    *,
    n_heads:    int                 = 1,
    n_triples:  int                 = 512,
    n_boot:     int                 = 200,
    seed:       int                 = 0,
    irrep_dims: Optional[List[int]] = None,   # per-block dims d_b of an unequal irrep tower (sum d_b = K)
) -> Dict[str, torch.Tensor]:
    r"""Wilson holonomy observable W_ijk = Re Tr(H_ijk) over random distinct triples.

    The gauge-INVARIANT triangle observable (PIFB:862-869): H_ijk = Omega_ij Omega_jk Omega_ki,
    W_ijk = Re Tr(H_ijk), reported normalized as W/K so a closed triangle H = I gives W/K = 1,
    with the Wilson-action density 1 - W/K (flat -> 0) -- the trace complement of
    ``holonomy_deviation_sampled``'s Frobenius ||H - I||_F. When ``n_heads > 1`` the gauge is
    block-diagonal H = oplus_h H^(h), so Tr(H) = sum_h Tr(H^(h)) and the per-head observable is
    W^(h)/d_k with d_k = K / n_heads; the per-head normalized values average back to the full W/K.
    That equal-chunk reshape assumes n_heads EQUAL contiguous blocks; for an UNEQUAL irrep tower
    (so_n/sp_n direct sums with per-block dims like [3, 5]) pass ``irrep_dims`` (audit 2026-07-01
    round-3): the diagonal is split per block and each block b reports W^(b)/d_b = Re Tr(H^(b))/d_b,
    so the d_b-weighted mean sum_b (d_b/K) W^(b)/d_b recovers the full W/K; ``irrep_dims=None``
    keeps the equal-chunk decomposition byte-identical.
    The real part is exact (= Tr) for the real groups built today; it would need ``.real`` only for
    a complex GL(K, C) gauge. On the flat phi-cocycle every triangle closes by construction so
    W/K = 1 (deviation 0) up to roundoff; on the coboundary-only registry this is a roundoff monitor,
    not curvature evidence (see the section note).
    Seeded random triples match
    ``holonomy_deviation_sampled`` so the two observables are directly comparable.
    """
    compact = isinstance(omega, CompactFactoredTransport)
    n = _require_unbatched_square_token_transport(omega, "holonomy_wilson_sampled")
    k = omega.K if compact else omega.shape[-1]
    device, dtype = omega.device, omega.dtype
    if irrep_dims is not None:
        if sum(irrep_dims) != k:
            raise ValueError(f"sum(irrep_dims)={sum(irrep_dims)} must equal K={k}")
        if n_heads > 1 and len(irrep_dims) != n_heads:
            raise ValueError(f"len(irrep_dims)={len(irrep_dims)} must match n_heads={n_heads}")
    elif n_heads > 1 and k % n_heads != 0:
        raise ValueError(f"n_heads={n_heads} must divide K={k} for the per-head Wilson decomposition")
    n_blocks = n_heads if irrep_dims is None else len(irrep_dims)
    if compact and irrep_dims is not None and irrep_dims != omega.irrep_dims:
        raise ValueError(
            f"compact irrep_dims={omega.irrep_dims} do not match requested {irrep_dims}")
    if compact and n_heads > 1 and n_heads != omega.n_blocks:
        raise ValueError(
            f"compact n_heads={n_heads} must match stored blocks={omega.n_blocks}")
    gen = torch.Generator(device=device).manual_seed(int(seed))
    draw = torch.randint(0, n, (max(n_triples * 3, 12), 3), generator=gen, device=device)
    keep = (draw[:, 0] != draw[:, 1]) & (draw[:, 1] != draw[:, 2]) & (draw[:, 0] != draw[:, 2])
    idx = draw[keep][:n_triples]
    if idx.numel() == 0:
        z = torch.zeros((), device=device, dtype=dtype)
        return {"wilson_mean": torch.ones((), device=device, dtype=dtype),
                "deviation_mean": z, "ci_lo": z, "ci_hi": z,
                "per_triple": torch.zeros(0, device=device, dtype=dtype),
                "span": torch.zeros(0, device=device, dtype=dtype),
                "per_head": torch.ones(n_blocks, device=device, dtype=dtype)}
    if compact:
        o_ij = _compact_sampled_pairs(omega, idx[:, 0], idx[:, 1])
        o_jk = _compact_sampled_pairs(omega, idx[:, 1], idx[:, 2])
        o_ki = _compact_sampled_pairs(omega, idx[:, 2], idx[:, 0])
        h = o_ij @ o_jk @ o_ki                                  # (T,H,d,d)
        block_trace = torch.diagonal(h, dim1=-2, dim2=-1).sum(dim=-1)  # (T,H)
        per = block_trace.sum(dim=-1) / k
        diag = None
    else:
        h = omega[idx[:, 0], idx[:, 1]] @ omega[idx[:, 1], idx[:, 2]] @ omega[idx[:, 2], idx[:, 0]]
        diag = torch.diagonal(h, dim1=-2, dim2=-1)               # (T,K)
        per = diag.sum(dim=-1) / k
    span = (idx.amax(dim=1) - idx.amin(dim=1)).to(dtype)
    dev = 1.0 - per                                              # (T,) Wilson-action density 1 - W/K
    ridx = torch.randint(0, dev.shape[0], (n_boot, dev.shape[0]), generator=gen, device=device)
    boot = dev[ridx].mean(dim=1)
    if compact:
        native = block_trace.mean(dim=0) / omega.block_dim
        per_head = native if (irrep_dims is not None or n_heads > 1) else per.mean().reshape(1)
    elif irrep_dims is not None:
        blocks   = torch.split(diag, irrep_dims, dim=-1)             # B chunks, each (T, d_b)
        per_head = torch.stack([b.sum(dim=-1).mean(dim=0) / b.shape[-1] for b in blocks])   # (B,) W^(b)/d_b
    else:
        d_k = k // n_heads
        per_head = diag.reshape(diag.shape[0], n_heads, d_k).sum(dim=-1).mean(dim=0) / d_k   # (n_heads,) W^(h)/d_k
    return {
        "wilson_mean":    per.mean(),
        "deviation_mean": dev.mean(),
        "ci_lo":          torch.quantile(boot, 0.025),
        "ci_hi":          torch.quantile(boot, 0.975),
        "per_triple":     per,
        "span":           span,
        "per_head":       per_head,
    }


def curvature_field(
    omega:  'torch.Tensor | CompactFactoredTransport',
    anchor: int = 0,
) -> torch.Tensor:                       # (N, N) F_ij = ||Omega_ai Omega_ij Omega_ja - I||_F
    r"""Spatial curvature map for a fixed anchor a: F_ij = ||Omega_ai Omega_ij Omega_ja - I||_F.

    Shows where curvature concentrates (a Regime-II quantity; ~0 everywhere on the flat cocycle).
    """
    if isinstance(omega, CompactFactoredTransport):
        exp_blocks = omega.exp_blocks
        inv_blocks = omega.inv_blocks
        o_ai = exp_blocks[anchor] @ inv_blocks
        o_ij = exp_blocks[:, None] @ inv_blocks[None, :]
        o_ja = exp_blocks @ inv_blocks[anchor]
        h = o_ai[:, None] @ o_ij @ o_ja[None, :]
        eye = torch.eye(omega.block_dim, device=omega.device, dtype=omega.dtype)
        return torch.linalg.vector_norm(h - eye, dim=(-3, -2, -1))
    k = omega.shape[-1]
    eye = torch.eye(k, device=omega.device, dtype=omega.dtype)
    o_ai = omega[anchor].unsqueeze(1)                            # (N, 1, K, K) Omega_{a, i} (over i)
    o_ja = omega[:, anchor].unsqueeze(0)                         # (1, N, K, K) Omega_{j, a} (over j)
    h = o_ai @ omega @ o_ja                                      # (N, N, K, K)
    return torch.linalg.norm(h - eye, dim=(-2, -1))


def cocycle_residual_sampled(
    omega:      'torch.Tensor | CompactFactoredTransport',

    *,
    n_triples:  int = 512,
    seed:       int = 0,
) -> torch.Tensor:                       # () mean ||Omega_ik - Omega_ij Omega_jk||_F over random triples
    r"""Cocycle defect mean ||Omega_ik - Omega_ij Omega_jk||_F over random distinct triples.

    A consistency check of the COMPOSITION law itself: a flat phi-cocycle
    Omega_ij = exp(phi_i) exp(-phi_j) satisfies Omega_ik = Omega_ij Omega_jk exactly, so this is ~0
    (cheaper and more fundamental than the triangle holonomy, which is the special case k -> i).
    Because every registered transport is built this way, a nonzero value measures floating-point
    roundoff and the conditioning of the stored frame pair, never a property of the construction:
    a roundoff monitor, not a flatness certificate (see the section note above).
    Seeded random triples (matching
    ``holonomy_deviation_sampled``) avoid the row-major low-index sampling bias.
    """
    compact = isinstance(omega, CompactFactoredTransport)
    n = _require_unbatched_square_token_transport(omega, "cocycle_residual_sampled")
    device, dtype = omega.device, omega.dtype
    gen = torch.Generator(device=device).manual_seed(int(seed))
    draw = torch.randint(0, n, (max(n_triples * 3, 12), 3), generator=gen, device=device)
    keep = (draw[:, 0] != draw[:, 1]) & (draw[:, 1] != draw[:, 2]) & (draw[:, 0] != draw[:, 2])
    idx = draw[keep][:n_triples]
    if idx.numel() == 0:
        return torch.zeros((), device=device, dtype=dtype)
    if compact:
        o_ik = _compact_sampled_pairs(omega, idx[:, 0], idx[:, 2])
        o_ij = _compact_sampled_pairs(omega, idx[:, 0], idx[:, 1])
        o_jk = _compact_sampled_pairs(omega, idx[:, 1], idx[:, 2])
        return (o_ik - o_ij @ o_jk).square().sum(dim=(-3, -2, -1)).sqrt().mean()
    o_ik = omega[idx[:, 0], idx[:, 2]]                           # (T, K, K)
    o_ij = omega[idx[:, 0], idx[:, 1]]
    o_jk = omega[idx[:, 1], idx[:, 2]]
    return torch.linalg.norm(o_ik - o_ij @ o_jk, dim=(-2, -1)).mean()


# --- free-energy closure and per-token profile (the headline F figure) ---

def free_energy_full_decomposition(
    self_coupling:   'float | torch.Tensor',     # F self-term sum_i[alpha_i D(q_i||p_i) + R(alpha_i)] (nats)
    belief_coupling: 'float | torch.Tensor',     # raw sum_ij beta_ij E_ij (nats)
    attention_ent:   'float | torch.Tensor',     # raw tau sum_ij beta_ij log(beta_ij/pi_ij) (nats)
    data_term:       'float | torch.Tensor',     # -E_q[log p(o|x)] = val_ce in nats

    *,
    lambda_beta:     'float | torch.Tensor' = 1.0,
) -> Dict[str, 'float | torch.Tensor']:
    r"""Close the free-energy stack: add the data/likelihood term and the lambda_beta scaling.

    The runtime-realised total is self_coupling + lambda_beta (belief_coupling + attention_entropy)
    + data_term, so the stacked-area figure CLOSES to the F the E-step minimizes (the current
    one-bar snapshot omits the data term and the scaling). Accepts scalars or per-step arrays
    elementwise.
    """
    belief_scaled = lambda_beta * belief_coupling
    entropy_scaled = lambda_beta * attention_ent
    total = self_coupling + belief_scaled + entropy_scaled + data_term
    out: Dict[str, 'float | torch.Tensor'] = {
        "self_coupling":   self_coupling,
        "belief_coupling": belief_scaled,
        "attention_entropy": entropy_scaled,
        "data_term":       data_term,
    }
    out["total"] = total
    return out


def self_coupling_profile(
    self_div: torch.Tensor,              # (N,) or (N, K) per-coordinate D(q_i||p_i)
    alpha:    torch.Tensor,              # (N,) or (N, K) self-coupling alpha_i

    *,
    eps:      float = 1e-12,
) -> Dict[str, torch.Tensor]:
    r"""Surface the per-token (D_i, alpha_i, alpha_i D_i) that diagnostics computes then collapses.

    ``self_divergence_for_alpha`` and ``self_coupling_alpha`` already produce these inside
    model.diagnostics before they are summed into the scalar ``self_coupling``; this returns the
    per-token vectors for the self-divergence violin. Per-coordinate inputs (N, K) are summed over
    the coordinate axis for the per-token totals while the raw arrays are kept. (alpha_i is a flat
    scalar on the default constant-alpha path; only informative under a state-dependent alpha.)
    """
    div_tok = self_div.sum(dim=-1) if self_div.dim() > 1 else self_div
    coupling_tok = (alpha * self_div)
    coupling_tok = coupling_tok.sum(dim=-1) if coupling_tok.dim() > 1 else coupling_tok
    return {
        "self_div":             div_tok,
        "alpha":                alpha,
        "self_coupling_per_token": coupling_tok,
    }


def estep_residuals(
    mu_traj:    torch.Tensor,            # (T+1, N, K) belief means over inner iterations
    sigma_traj: torch.Tensor,            # (T+1, N, K) diagonal OR (T+1, N, K, K) full covariances
    phi_traj:   torch.Tensor,            # (T+1, N, n_gen) gauge frames

    *,
    diagonal:   Optional[bool] = None,
    eps:        float = 1e-12,
) -> Dict[str, torch.Tensor]:
    r"""Per-iteration belief-change residuals (T, N), the covariance step in the SPD metric.

    r_mu(t) = ||mu_t - mu_{t-1}||, r_sigma(t) = spd_geodesic_distance(Sigma_{t-1}, Sigma_t)
    (the affine-invariant step, NOT a Euclidean one), r_phi(t) = ||phi_t - phi_{t-1}||. Shrinking
    residuals certify the E-step converges to a fixed point in the geometry it actually moves in.
    """
    r_mu = torch.linalg.norm(mu_traj[1:] - mu_traj[:-1], dim=-1)              # (T, N)
    r_sigma = spd_geodesic_distance(sigma_traj[:-1], sigma_traj[1:], diagonal=diagonal, eps=eps)
    r_phi = torch.linalg.norm(phi_traj[1:] - phi_traj[:-1], dim=-1)          # (T, N)
    return {"r_mu": r_mu, "r_sigma": r_sigma, "r_phi": r_phi}


# --- single-seed bootstrap bands (resample data, NOT seeds) ---

def bootstrap_ce_band(
    per_seq_nats:   torch.Tensor,        # (S,) summed nats per validation sequence
    per_seq_tokens: torch.Tensor,        # (S,) non-ignored token count per sequence

    *,
    n_boot:         int   = 1000,
    seed:           int   = 0,
    q_lo:           float = 0.025,
    q_hi:           float = 0.975,
) -> Dict[str, float]:
    r"""Token-weighted cross-entropy with a bootstrap-over-SEQUENCES band (single-seed-legitimate).

    Resamples validation sequences with replacement and reports the token-weighted CE
    sum(nats) / sum(tokens) percentiles. This is within-run uncertainty over the eval set, NOT a
    cross-seed confidence interval (the run protocol is single-seed); captions must say so. An
    aggregate with zero evaluated tokens has no cross-entropy estimate and returns NaN for all fields.
    """
    total_tokens = per_seq_tokens.sum()
    if bool(total_tokens == 0):
        nan = float("nan")
        return {"ce": nan, "lo": nan, "hi": nan}
    gen = torch.Generator(device=per_seq_nats.device).manual_seed(int(seed))
    s = per_seq_nats.shape[0]
    idx = torch.randint(0, s, (n_boot, s), generator=gen, device=per_seq_nats.device)
    boot = per_seq_nats[idx].sum(dim=1) / per_seq_tokens[idx].sum(dim=1).clamp(min=1.0)
    point = float(per_seq_nats.sum() / total_tokens)
    return {"ce": point, "lo": float(torch.quantile(boot, q_lo)), "hi": float(torch.quantile(boot, q_hi))}


def bootstrap_token_ce_band(
    arm_token_nats:  torch.Tensor,       # (M,) per-token nats for the ablation arm
    full_token_nats: torch.Tensor,       # (M,) per-token nats for the full model (SAME tokens)

    *,
    n_boot:          int   = 1000,
    seed:            int   = 0,
    q_lo:            float = 0.025,
    q_hi:            float = 0.975,
    block_len:       Optional[int] = None,
) -> Dict[str, float]:
    r"""Paired bootstrap band for the ablation delta (arm minus full) cross-entropy.

    The SAME resample index is applied to both arms (paired), so the band reflects the per-token
    correlation between models, not independent noise. Single-seed-legitimate (within-run over the
    shared token set), not a cross-seed CI.

    ``block_len`` selects the cluster design. The aligned stream is cut into consecutive blocks
    b of ``block_len`` tokens (the evaluation window; the last block may be shorter), with paired
    sums and counts

    .. math:: D_b = \sum_{t \in b} (\ell^{\mathrm{arm}}_t - \ell^{\mathrm{full}}_t), \qquad n_b = |b|,

    and whole blocks are resampled, each replicate being :math:`\sum_b D_b / \sum_b n_b` -- the
    per-sequence ratio design of :func:`bootstrap_ce_band`. Tokens of one sequence share context,
    so resampling them independently (``block_len=None``, kept for API compatibility) undercovers
    whenever the per-token deltas are correlated within a sequence.
    """
    if arm_token_nats.shape != full_token_nats.shape:
        raise ValueError(f"paired bootstrap needs aligned tokens; got {tuple(arm_token_nats.shape)} "
                         f"vs {tuple(full_token_nats.shape)}")
    gen = torch.Generator(device=arm_token_nats.device).manual_seed(int(seed))
    m = arm_token_nats.shape[0]
    point = float(arm_token_nats.mean() - full_token_nats.mean())
    if block_len is None:
        idx = torch.randint(0, m, (n_boot, m), generator=gen, device=arm_token_nats.device)
        delta = arm_token_nats[idx].mean(dim=1) - full_token_nats[idx].mean(dim=1)
    else:
        if isinstance(block_len, bool) or not isinstance(block_len, int) or block_len < 1:
            raise ValueError(f"block_len must be a positive integer, got {block_len!r}")
        paired = (arm_token_nats - full_token_nats).to(torch.float64)                  # (M,)
        blocks = paired.split(block_len)
        block_sums = torch.stack([chunk.sum() for chunk in blocks])                    # (S,) D_b
        block_sizes = torch.tensor([chunk.numel() for chunk in blocks],
                                   dtype=torch.float64, device=paired.device)          # (S,) n_b
        s = block_sums.shape[0]
        idx = torch.randint(0, s, (n_boot, s), generator=gen, device=paired.device)
        delta = block_sums[idx].sum(dim=1) / block_sizes[idx].sum(dim=1)
    return {"delta": point, "lo": float(torch.quantile(delta, q_lo)), "hi": float(torch.quantile(delta, q_hi))}


# --- gauge-equivariance certificate (the symmetry the construction rests on) ---

def gauge_equivariance_residual(
    mu:                torch.Tensor,     # (N, K) converged belief means
    sigma:             torch.Tensor,     # (N, K) diagonal OR (N, K, K) full covariances
    omega:             'torch.Tensor | CompactFactoredTransport',
    group,                               # GaugeGroup (generators, irrep_dims)

    *,
    kappa:             float = 1.0,
    renyi_order:       float = 1.0,
    kl_max:            float = 100.0,
    eps:               float = 1e-6,
    diagonal:          Optional[bool] = None,
    n_samples:         int   = 8,
    scale:             float = 0.5,
    seed:              int   = 0,
    divergence_family: str   = "renyi",
    full_cov_kl_precision: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    r"""Empirical gauge-equivariance certificate of the attention energy and weights.

    The construction's central symmetry is D(rho(g) q_i || rho(g) Omega_ij q_j) = D(q_i||Omega_ij q_j)
    under a GLOBAL structure-group element g (with the transport co-transforming Omega -> g Omega g^{-1}).
    Applies n_samples random IN-group g = exp(sum_a c_a G_a) and a matched OUT-of-group control
    g = exp(scale * randn(K, K)) (generic GL(K), which for a block/skew/symplectic group does NOT
    respect the structure), recomputes E_ij and beta_ij with the FULL Gaussian family (a general g
    makes a diagonal Sigma full), and returns the relative residuals. In-group residuals cluster at
    float32 eps; the out-of-group control sits far above. (For glk the group IS all of GL(K), so the
    control is also invariant -- correct, there is no 'outside'.)

    SCOPE (audit 2026-06-10 F12): this certifies the joint-congruence invariance of the GIVEN
    operator tensor -- (mu, Sigma, Omega) co-transformed together -- which holds for ANY omega.
    It is therefore BLIND to a BUILDER-level equivariance break: the residual is computed on the
    operator it is handed, never on a construction rebuilt from its own inputs. Read
    the in-group panel as "congruence consistency of the converged operator", NOT as a
    certificate that the transport construction is equivariant.
    """
    from vfe3.families.base import get_family
    from vfe3.free_energy import attention_tau, attention_weights, pairwise_energy
    from vfe3.geometry.transport import _factor_dtype, transport_mean

    if _factor_dtype(omega) == torch.float64 and mu.dtype != torch.float64:
        # Escalated float64 frames (audit 2026-09-23 #12): certify the whole congruence at float64,
        # so the random group elements g (built at mu.dtype, from generators cast to it) share the
        # operator's dtype.
        mu, sigma = mu.double(), sigma.double()

    compact_factors: Optional[
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
    ] = None
    if isinstance(omega, CompactFactoredTransport):
        # A generic out-of-group control mixes the compact blocks, but it still acts on the two
        # vertex factors independently. Expanding each vertex to (N,K,K) keeps the exact
        # A_i B_j action without ever allocating the pairwise (N,N,K,K) transport.
        compact_factors = (
            CompactBlockElement(omega.exp_blocks, omega.K).to_dense(),
            CompactBlockElement(omega.inv_blocks, omega.K).to_dense(),
            None,
            None,
        )

    full = get_family("gaussian_full")
    from vfe3.families.gaussian import FullGaussian
    def _instance(mu_: torch.Tensor, sigma_: torch.Tensor):
        if full is FullGaussian and full_cov_kl_precision is not None:
            return full(mu_, sigma_, _precision_policy=full_cov_kl_precision)
        return full(mu_, sigma_)
    def _transport(sigma_: torch.Tensor, omega_, **kwargs) -> torch.Tensor:
        if full is FullGaussian and full_cov_kl_precision is not None:
            kwargs["precision_policy"] = full_cov_kl_precision
        return full.transport_dispersion(sigma_, omega_, **kwargs)
    def _from_transported(mu_: torch.Tensor, sigma_: torch.Tensor, source_: torch.Tensor):
        if full is FullGaussian and full_cov_kl_precision is not None:
            return full.from_transported(mu_, sigma_, source_, _precision_policy=full_cov_kl_precision)
        return full.from_transported(mu_, sigma_, source_)
    is_full = _is_full_cov(sigma, diagonal)
    sigma0 = sigma if is_full else torch.diag_embed(sigma)                    # (N, K, K)
    k = mu.shape[-1]
    _kappa = torch.as_tensor(kappa, dtype=torch.float32, device=mu.device) if isinstance(kappa, (list, tuple)) else kappa
    tau = attention_tau(_kappa, group.irrep_dims)
    off = ~torch.eye(mu.shape[0], dtype=torch.bool, device=mu.device)
    gen = torch.Generator(device=mu.device).manual_seed(int(seed))

    def _energy(
        mu_q:   torch.Tensor,
        sig_q:  torch.Tensor,
        om:     'torch.Tensor | Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(om, tuple):
            left, right, gauge_left, gauge_right = om
            mu_rows: List[torch.Tensor] = []
            sig_rows: List[torch.Tensor] = []
            for i in range(left.shape[0]):
                omega_row = torch.einsum(
                    "ikl,jlm->ijkm", left[i:i + 1], right)                  # (1, N, K, K)
                if gauge_left is not None and gauge_right is not None:
                    omega_row = torch.einsum(
                        "kl,ijlm,mn->ijkn", gauge_left, omega_row, gauge_right)
                mu_rows.append(transport_mean(omega_row, mu_q)[0])
                sig_rows.append(_transport(
                    sig_q,
                    omega_row,
                    diagonal_out=sig_q.dim() == mu_q.dim(),
                )[0])
            mu_t = torch.stack(mu_rows, dim=0)                              # (N, N, K)
            sig_t = torch.stack(sig_rows, dim=0)                            # (N, N, K[, K])
        else:
            mu_t = transport_mean(om.unsqueeze(0), mu_q.unsqueeze(0))[0]          # (N, N, K)
            sig_t = _transport(sig_q.unsqueeze(0), om.unsqueeze(0))[0]  # (N, N, K, K)
        e = pairwise_energy(_instance(mu_q, sig_q), _from_transported(mu_t, sig_t, sig_q),
                            alpha=renyi_order, kl_max=kl_max, eps=eps,
                            divergence_family=divergence_family, irrep_dims=group.irrep_dims)
        return e, attention_weights(e, tau=tau)

    base_transport = compact_factors if compact_factors is not None else omega
    e0, beta0 = _energy(mu, sigma0, base_transport)

    def _residuals(g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        g_inv = torch.linalg.inv(g)
        mu_g = mu @ g.transpose(-1, -2)                                       # (N, K) g mu_i
        sig_g = g @ sigma0 @ g.transpose(-1, -2)                              # (N, K, K) g Sigma g^T
        if compact_factors is None:
            om_g = torch.einsum("kl,ijlm,mn->ijkn", g, omega, g_inv)         # g Omega g^{-1}
        else:
            left, right, _, _ = compact_factors
            om_g = (left, right, g, g_inv)                                    # g (A_i B_j) g^{-1}
        e, beta = _energy(mu_g, sig_g, om_g)
        r_e = ((e - e0).abs() / e0.abs().clamp(min=eps))
        r_b = (beta - beta0).abs()
        return r_e[..., off].flatten(), r_b.flatten()         # off masks the trailing (N, N) axes

    in_e, in_b, out_e, out_b = [], [], [], []
    for _ in range(n_samples):
        c = scale * torch.randn(group.generators.shape[0], generator=gen, device=mu.device, dtype=mu.dtype)
        g_in = torch.linalg.matrix_exp(torch.einsum("a,aij->ij", c, group.generators.to(c.dtype)))
        re, rb = _residuals(g_in)
        in_e.append(re)
        in_b.append(rb)
        m = scale * torch.randn(k, k, generator=gen, device=mu.device, dtype=mu.dtype)
        re2, rb2 = _residuals(torch.linalg.matrix_exp(m))
        out_e.append(re2)
        out_b.append(rb2)
    return {
        "energy_in_group":  torch.cat(in_e),
        "energy_out_group": torch.cat(out_e),
        "beta_in_group":    torch.cat(in_b),
        "beta_out_group":   torch.cat(out_b),
    }


def head_mixer_gauge_residual(
    mu:        torch.Tensor,             # (N, K) converged belief means
    sigma:     torch.Tensor,             # (N, K) diagonal OR (N, K, K) full covariances
    head_mixer,                          # HeadMixer module (the trained Schur-commutant mixer)
    group,                               # GaugeGroup (generators, irrep_dims)

    *,
    n_samples: int            = 8,
    scale:     float          = 0.5,
    seed:      int            = 0,
    eps:       float          = 1e-6,
    diagonal:  Optional[bool] = None,
) -> Dict[str, torch.Tensor]:
    r"""BUILDER-level gauge-equivariance residual of the head mixer (tied vs untied).

    Certifies whether the mixer operation commutes with an IN-group gauge action, i.e.
    mix(g mu, g Sigma g^T) == g mix(mu, Sigma) g^T for g = exp(sum_a c_a G_a) drawn from the
    GROUP's OWN generators. The Schur-commutant mixer M = blockdiag_t(A_t kron I_d) commutes
    with a TIED gauge Omega = kron(I_n, h) (group ``tied_block_glk``, generators kron(I_n, gl(d)))
    -- residual at float32 eps -- but NOT with the UNTIED per-head gauge of ``block_glk`` (each
    head its own gl(d)), where the residual grows as A drifts from I during training.

    This is the complement of :func:`gauge_equivariance_residual`, which co-transforms a SUPPLIED
    Omega (mu, Sigma, Omega together) and is therefore BLIND to a builder-level break: it certifies
    the converged operator's congruence consistency, not whether the construction is equivariant.
    Here the operator is REBUILT under g, so the tied-vs-untied distinction is visible -- this is
    the instrument the A2/EXP-9 ablation needs. Uses the FULL-covariance mixer path (the diagonal
    closed form sigma'[m] = sum_n A[m,n]^2 sigma[n] is equivariant only under DIAGONAL gauges), so a
    diagonal sigma is promoted to full via ``diag_embed``. Returns relative residuals for the mean
    (per token) and the covariance (Frobenius, per token); in-group they cluster at eps for a tied
    gauge and rise with mixer drift for an untied one.
    """
    is_full = _is_full_cov(sigma, diagonal)
    sigma0 = sigma if is_full else torch.diag_embed(sigma)                    # (N, K, K)
    mu_m, sig_m = head_mixer(mu, sigma0)                                      # mix(mu, Sigma)
    gen = torch.Generator(device=mu.device).manual_seed(int(seed))
    r_mu, r_sig = [], []
    for _ in range(n_samples):
        c = scale * torch.randn(group.generators.shape[0], generator=gen, device=mu.device, dtype=mu.dtype)
        g = torch.linalg.matrix_exp(torch.einsum("a,aij->ij", c, group.generators))   # in-group g
        mu_g  = mu @ g.transpose(-1, -2)                                      # (N, K)    g mu_i
        sig_g = torch.einsum("kl,nlm,jm->nkj", g, sigma0, g)                  # (N, K, K) g Sigma g^T
        mu_L, sig_L = head_mixer(mu_g, sig_g)                                 # mix(g .)
        mu_R  = mu_m @ g.transpose(-1, -2)                                    # g . mix (mean)
        sig_R = torch.einsum("kl,nlm,jm->nkj", g, sig_m, g)                   # g . mix (cov)
        r_mu.append((mu_L - mu_R).norm(dim=-1) / mu_R.norm(dim=-1).clamp(min=eps))
        r_sig.append((sig_L - sig_R).flatten(1).norm(dim=-1) / sig_R.flatten(1).norm(dim=-1).clamp(min=eps))
    return {"mu_residual": torch.cat(r_mu), "sigma_residual": torch.cat(r_sig)}


# ---------------------------------------------------------------------------
# Registry: name -> metric(**context). New probes slot in by name.
# ---------------------------------------------------------------------------
_METRICS: Dict[str, Callable] = {}


def register_metric(name: str, *, override: bool = False) -> Callable:
    """Decorator registering a metric that reads its inputs from the context kwargs.

    Duplicate keys fail closed (audit 2026-07-01 round-3): a second registration under an
    existing name silently shadowed the first, so a config-selected probe could dispatch to
    an unintended implementation. Pass ``override=True`` to replace deliberately.
    """
    def _wrap(fn: Callable) -> Callable:
        if name in _METRICS and not override:
            raise KeyError(f"metric {name!r} already registered; pass override=True to replace")
        _METRICS[name] = fn
        return fn
    return _wrap


def get_metric(name: str) -> Callable:
    """Return the registered metric (KeyError if absent)."""
    if name not in _METRICS:
        raise KeyError(f"no metric {name!r}; available: {sorted(_METRICS)}")
    return _METRICS[name]


# Each metric's OWN context key is REQUIRED (no None default): a missing or mis-keyed
# context now raises TypeError at the call instead of an AttributeError deep inside the
# kernel (effective_rank(None) etc.). The trailing **kw stays only to absorb SIBLING
# metrics' context keys, since ``compute_metrics`` floods the full context to every metric.
@register_metric("effective_rank")
def _m_eff_rank(
    *,
    sigma:    torch.Tensor,
    diagonal: bool,
    eps:      float         = 1e-12,
    family:   Optional[str] = None,
    **kw,
) -> float:
    """Mean spectral effective rank of the belief covariances.

    Routes through ``_spectrum`` so a FULL covariance (..., K, K) is reduced to its eigenvalue
    spectrum before the participation ratio (audit 2026-06-13 L15); passing the matrix directly
    treated its rows as a spectrum. A diagonal (..., K) variance tensor is its own spectrum.

    ``diagonal`` is REQUIRED (no auto-inference): a diagonal (N, K) variance table with N == K is
    square in its last two axes, so shape-based auto-inference would misread it as a full (K, K)
    covariance and eigvalsh a variance vector. The caller passes the explicit flag (audit PB-07)."""
    return float(effective_rank_per_token(
        sigma,
        diagonal=diagonal,
        eps=eps,
        family=family,
    ).mean())


@register_metric("attention_entropy")
def _m_attn_entropy(*, beta: torch.Tensor, **kw) -> float:
    """Mean attention row entropy."""
    return float(attention_entropy(beta))


@register_metric("holonomy_deviation")
def _m_holonomy(*, omega: 'torch.Tensor | CompactFactoredTransport', **kw) -> float:
    """Mean triangle-holonomy departure from identity (curvature proxy).

    Routes to the SAMPLED estimator (audit 2026-06-10 F16): the deterministic row-major
    ``holonomy_deviation`` always reads the same low-index-token triangles -- a systematically
    biased curvature estimate -- and ``diagnostics()`` already logs the sampled one, so the
    registry key now agrees with the training log."""
    return float(holonomy_deviation_sampled(omega)["mean"])


@register_metric("gauge_trace_spread")
def _m_gauge_spread(*, phi: torch.Tensor, generators: torch.Tensor, **kw) -> float:
    """Spread of log|det Omega| across tokens."""
    return float(gauge_trace_spread(phi, generators))


@register_metric("free_energy_terms")
def _m_free_energy_terms(
    *,
    self_div:                  torch.Tensor,
    energy:                    torch.Tensor,
    beta:                      torch.Tensor,
    alpha:                     torch.Tensor,

    tau:                       'float | torch.Tensor',
    lambda_beta:               'float | torch.Tensor' = 1.0,

    include_attention_entropy: bool = True,

    log_prior:                 Optional[torch.Tensor] = None,
    alpha_reg:                 Optional[torch.Tensor] = None,
    coupling_energy:           Optional[torch.Tensor] = None,
    log_likelihood:            Optional[torch.Tensor] = None,
    **kw:                      object,
) -> Dict[str, float]:
    """Per-term free-energy decomposition (self-coupling, belief-coupling, attention entropy).

    ``tau`` is REQUIRED (no default): the wrapper has no way to recover the group-aware softmax
    temperature tau = kappa*sqrt(d_head), and a silent tau=1.0 makes the attention-entropy term (and
    so ``total``) wrong for any K>1 (audit 2026-06-13 L16). Callers pass ``attention_tau(...)`` -- as
    the live diagnostics path already does -- matching this module's required-context-key convention.

    Every scalar-objective seam is forwarded. Defaults reproduce the old behavior when a context
    omits the opt-in terms, so existing registry callers remain byte-identical."""
    return free_energy_terms(self_div, energy, beta, alpha, tau=tau, log_prior=log_prior,
                             lambda_beta=lambda_beta,
                             include_attention_entropy=include_attention_entropy,
                             alpha_reg=alpha_reg, coupling_energy=coupling_energy,
                             log_likelihood=log_likelihood)


def compute_metrics(
    names: List[str],

    **context,
) -> Dict[str, 'float | Dict[str, float]']:
    r"""Run the named metrics against the keyword ``context`` (sigma=, beta=, omega=, ...).

    Most metrics return a float; ``free_energy_terms`` returns a nested ``Dict[str, float]``,
    so the value type is widened accordingly."""
    return {n: get_metric(n)(**context) for n in names}


# ---------------------------------------------------------------------------
# Production diagnostics group (audit PB-07): the metric names ``VFEModel.diagnostics()``
# dispatches through the registry, plus the declarative (metric, output_key, flatten) mapping
# the call site walks to assemble its dict. ``attention_entropy`` (row entropy) is aliased to
# ``attn_entropy`` so it does NOT overwrite the free-energy component named ``attention_entropy``
# that ``free_energy_terms`` returns; ``free_energy_terms`` flattens its returned mapping into the
# dict (the call site guards the flatten against overwrite collisions). Holonomy / gauge metrics
# stay OFF this group: their diagnostics consume sibling confidence bounds and active-frame
# branches the scalar registry wrappers do not return.
# ---------------------------------------------------------------------------
DIAGNOSTIC_METRIC_NAMES = (
    "attention_entropy",
    "free_energy_terms",
    "effective_rank",
)
DIAGNOSTIC_METRIC_OUTPUTS = (
    ("attention_entropy", "attn_entropy", False),
    ("free_energy_terms", None, True),
    ("effective_rank", "effective_rank", False),
)
