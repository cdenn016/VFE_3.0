r"""Diagnostic metrics for VFE_3.0 runs (publication-oriented).

Pure measurements over beliefs / transport / attention (no gradients, no side effects), suitable
for logging a training run or a figure: effective rank (spectral participation ratio), attention
row entropy, the sampled coboundary closure residual of Omega_ij = U_i U_j^{-1} (zero in exact
arithmetic, so a roundoff and frame-conditioning monitor, not curvature), the per-term free-energy
decomposition, and content_link_spectrum (similarity invariants of the relational content maps
M_h = O_h T_h).
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from vfe3.geometry.groups import GaugeGroup

from vfe3.numerics import safe_eigvalsh
from vfe3.geometry.transport import CompactFactoredTransport


def eigvals_on_host(
    matrix: torch.Tensor,                # (..., n, n) real or complex, any device
) -> torch.Tensor:                       # (..., n) complex eigenvalues, on ``matrix``'s device
    r"""``torch.linalg.eigvals`` computed by LAPACK on the CPU and returned to the input's device.

    On CUDA, PyTorch evaluates the non-symmetric eigenproblem through MAGMA's hybrid ``geev`` one
    matrix of the batch at a time, with host-device traffic per matrix; for the per-token frame
    batches of the diagnostics (B N matrices) that dominates a logging step. The CPU LAPACK path
    computes the same eigenvalues on the detached matrix (diagnostics only; no gradient).
    """
    return torch.linalg.eigvals(matrix.detach().cpu()).to(matrix.device)


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
    eps:        float               = 1e-12,
    block_dims: Optional[List[int]] = None,
) -> Dict[str, float]:
    r"""Summarize the embedded chart radius and the conditioning of the exact vertex exp(M)."""
    matrix = _diagnostic_embed(phi.double(), generators)
    norm = torch.linalg.matrix_norm(matrix, ord="fro", dim=(-2, -1))
    from vfe3.geometry.transport import _blockwise_matrix_exp
    exp_matrix = (_blockwise_matrix_exp(matrix, block_dims)
                  if block_dims and len(block_dims) > 1
                  else torch.linalg.matrix_exp(matrix))
    singular = torch.linalg.svdvals(exp_matrix)
    condition = singular[..., 0] / singular[..., -1].clamp(min=eps)
    return {
        "phi_matrix_norm_median": _finite_quantile(norm, 0.50),
        "phi_matrix_norm_p95":    _finite_quantile(norm, 0.95),
        "phi_matrix_norm_p99":    _finite_quantile(norm, 0.99),
        "phi_matrix_norm_max":    float(norm.max()) if norm.numel() else float("nan"),
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
    eps:           float = 1e-12,
    block_dims:    Optional[List[int]] = None,
) -> Dict[str, float]:
    r"""Compare fp32 and rebuilt-fp64 numerical closure of a flat vertex cocycle.

    With ``right_phi=Y``, the measured vertex and inverse are the exact-composition operators
    ``U=exp(X) exp(Y)`` and ``U^{-1}=exp(Y)^{-1} exp(X)^{-1}``. The fp32 inverse of each factor is formed as
    the live producer stores it (``_checked_group_inverse_certified``): the float64 inverse of the float32
    exponential, cast to float32, not a separately rounded ``exp(-M)``, which overstated the live residuals
    5 to 26 times (audit 2026-10-07, F65). A frame pair the live certificate escalates to float64 is
    measured here in float32 all the same.
    """
    coordinate = phi[0] if phi.dim() == 3 else phi
    matrix64 = _diagnostic_embed(coordinate.double(), generators)
    right_matrix64 = None
    if right_phi is not None:
        right_coordinate = right_phi[0] if right_phi.dim() == 3 else right_phi
        right_matrix64 = _diagnostic_embed(right_coordinate.double(), generators)
    triples = _deterministic_triples(matrix64.shape[0], max_triangles)
    k_dim = matrix64.shape[-1]

    def _measure(
        matrix:       torch.Tensor,
        right_matrix: Optional[torch.Tensor],
        live_inverse: bool = False,
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

        def _inverse(value: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
            return (torch.linalg.inv(positive.double()).to(positive.dtype) if live_inverse
                    else _exp(-value))

        left_pos = _exp(matrix)
        left_neg = _inverse(matrix, left_pos)
        if right_matrix is None:
            exp_pos = left_pos
            exp_neg = left_neg
        else:
            right_pos = _exp(right_matrix)
            exp_pos = left_pos @ right_pos
            exp_neg = _inverse(right_matrix, right_pos) @ left_neg
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
        matrix64.float(), None if right_matrix64 is None else right_matrix64.float(), live_inverse=True,
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

    A test-only reference, like the scalar it mirrors: no production path calls it, and the projected
    routes' canonical diagnostic assembles its own float64 rows (audit 2026-10-16 F92). It mirrors
    ``vfe3.free_energy.free_energy`` so the reported ``total`` equals the scalar that function returns
    for the SAME inputs (within float32 rounding):

        F = sum_i [ alpha_i D(q_i||p_i) + R(alpha_i)
                  + lambda_beta ( sum_j beta_ij E_ij
                                  + tau sum_j beta_ij log(beta_ij/pi_ij) )
                  - ell_i ]   (entropy: canonical only)

    ``belief_coupling``, ``attention_entropy``, and the conditionally present
    ``observation_likelihood`` are RAW (unweighted) block values, so each
    stays individually interpretable; ``total`` is the runtime-realized SCALED free energy
    self_coupling + lambda_beta (belief_coupling + attention_entropy), the reference scalar of
    ``free_energy``; the projected training routes neither evaluate nor minimize it. At
    lambda_beta = 1.0 total is byte-identical to the unscaled sum.

    ``alpha_reg`` is the per-position regularizer R(alpha_i) (b0 alpha - c0 log alpha under
    ``state_dependent_per_coord``); when supplied it is added elementwise into the self-coupling
    term (and so into ``total``), matching ``free_energy``. Default None leaves the result
    byte-identical to the unregularized path. ``include_attention_entropy`` gates the entropy
    contribution to ``total`` exactly as ``free_energy`` does -- when False the surrogate objective
    omits the entropy term, but its value is still reported under
    ``attention_entropy`` for diagnostics.

    ``coupling_energy`` selects the VALUE-gauge energy for the coupling sum while ``energy`` remains
    the score energy that produced ``beta``, and ``log_likelihood`` is subtracted from ``total``
    when the gated observation seam supplies it. Their defaults preserve
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
    # xlogy, as in the rows that assemble ``total``: zero at beta == 0 with no floor on a positive tail
    # probability, so the reported block is the one ``total`` carries (audit 2026-10-16 F92).
    beta_for_log = torch.where(beta > 0.0, beta, torch.ones_like(beta))
    entropy = float((_tau_e * (torch.special.xlogy(beta, beta_for_log) - beta * log_pi)).sum())
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
# than a single CSV scalar. Runners that reload a
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

    Predictive attention preserves sigma unless ``predictive_covariance='moment_matched'``
    assigns its context-dependent kernel covariance. Projected gradient updates sigma unless
    ``skip_belief_sigma_update=True``. Later seams can also rewrite sigma: a block MLP with any
    non-passthrough ``block_mlp_covariance`` (the delta-method ``'delta_full'`` or the prototype-agent
    ``'posterior'``) or CG coupling with ``cg_covariance_mode='delta_full'``. Normalization acts on the mean alone,
    except the block LayerNorm under ``block_norm_covariance='pushforward'``, which carries Sigma to post-norm units
    through the context-dependent scale of the mean. Without those updates, tr Sigma_q(x_i) depends only
    on the input token, so B1/EXP-3 measures input-token variance against next-token CE.
    """
    attention_mode = getattr(cfg, "attention_mode", "projected_predictive")
    if attention_mode == "projected_gradient":
        e_step_frozen = bool(getattr(cfg, "skip_belief_sigma_update", False))
    else:
        e_step_frozen = getattr(cfg, "predictive_covariance", "passthrough") not in ("moment_matched",
                                                                                        "spread_corrected")
    mlp_rewrites_sigma = (bool(getattr(cfg, "use_block_mlp", False))
                          and getattr(cfg, "block_mlp_covariance", "passthrough") != "passthrough")
    cg_rewrites_sigma = (bool(getattr(cfg, "use_cg_coupling", False))
                         and getattr(cfg, "cg_covariance_mode", "passthrough") == "delta_full")
    # Validation admits 'pushforward' only for a local-coordinate block LayerNorm.
    norm_rewrites_sigma = getattr(cfg, "block_norm_covariance", "passthrough") == "pushforward"
    return e_step_frozen and not (mlp_rewrites_sigma or cg_rewrites_sigma or norm_rewrites_sigma)


def variance_guard_activity(cfg: object) -> Tuple[bool, bool]:
    r"""Whether the ``eps`` variance floor and the ``sigma_max`` ceiling clamp a layer's converged covariance.

    Returns ``(floor, ceiling)``. :func:`guard_saturation` counts the variances within a relative window of
    ``eps`` and of ``sigma_max``; the count certifies a clamp only where the route applies one to the covariance
    it reads. The ceiling is the sigma E-step's SPD bound, absent when ``sigma_max`` is None or
    ``cfg.inactive_fields()`` marks it inert (projected-predictive routes, ``skip_belief_sigma_update``, the
    closed-form full-covariance receiver), the rule ``vfe3.viz.report`` applies to its ceiling row. The floor is
    absent where a moment-matched readout (``predictive_covariance`` ``'moment_matched'`` or ``'spread_corrected'``
    on ``attention_mode='projected_predictive'``, the routes on which ``predictive_covariance_floor`` is active)
    writes the converged covariance: it returns :math:`\Sigma_i^+ = \Sigma_i^{\mathrm{att}} + f\Sigma_{p,i}`, an
    additive floor with no clamp at ``eps``. That readout's one clamp, the Loewner ceiling
    :math:`\Sigma_i^+ \preceq c\Sigma_{p,i}`, is counted in ``covariance_ceiling_elements``. Elsewhere the converged
    covariance is the sigma E-step's retraction onto :math:`[\epsilon, \sigma_{\max}]` or a passthrough of the
    encoded table variance, which :func:`vfe3.numerics.bounded_variance_from_log` clamps at ``eps``.
    """
    inactive = cfg.inactive_fields()
    floor = "predictive_covariance_floor" in inactive
    ceiling = getattr(cfg, "sigma_max", None) is not None and "sigma_max" not in inactive
    return floor, ceiling


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

    r is not invariant under per-token linear maps of the rows. Local-coordinate means mu_i = U_i xi_i carry
    each token's frame U_i, so their r counts frame diversity as spread; the frame-invariant reading of the
    mixed content is r of the reference means xi_i = U_i^{-1} mu_i (the model's
    ``relational_rank_one_residual_reference``, audit 2026-10-16 F45).
    """
    xbar = mu.mean(dim=-2, keepdim=True)                                  # (..., 1, K) mean token
    num = torch.linalg.norm((mu - xbar).flatten(-2, -1), dim=-1)          # ||X - 1 xbar^T||_F
    den = torch.linalg.norm(mu.flatten(-2, -1), dim=-1).clamp(min=eps)    # ||X||_F
    return num / den


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
    support:  Optional[torch.Tensor] = None,   # bool, broadcastable to beta: admissible keys
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

    ``support`` is the prior's admissible-key mask (``isfinite(log_prior)``). Without it the
    support is inferred from ``beta > 0``, which miscounts a key whose weight underflowed to
    exactly zero as masked: an fp32 router with a cost gap above about 103 nats then shrinks the
    row's ceiling, and a fully collapsed head reads as uniform over a one-key row (audit
    2026-10-07 F14).
    """
    admitted = (beta > 0.0) if support is None else support.expand_as(beta)
    support = admitted.sum(dim=-1)                           # (..., N) admissible keys per row
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


# --- per-token frame class functions (group-dispatched; not blind to unimodular groups) ---
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

    A spread of log|det Omega| (= tr embed(phi)) is identically 0 for the unimodular SO(N) towers
    (traceless generators, det == 1), so it is blind off block_glk. This dispatches the correct
    invariant: GL volume log|det exp(phi)| for block_glk, and total rotation angle
    (1/2) sum_k |arg(eig)| for the orthogonal so_n irrep towers.
    """
    if getattr(group, "skew_symmetric", False):                  # so_n, tied or untied
        ang = torch.angle(eigvals_on_host(exp_phi))              # (..., K) eigenphases
        return 0.5 * ang.abs().sum(dim=-1)
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
        logmod = torch.log(eigvals_on_host(blk).abs().clamp(min=eps))        # (..., d) log|eig|
        anisos.append(logmod.amax(dim=-1) - logmod.amin(dim=-1))
        start += d
    return {
        "logdet":     torch.stack(logdets, dim=-1),              # (..., H)
        "anisotropy": torch.stack(anisos, dim=-1),               # (..., H)
    }


def _complex_structure(d: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    r"""The (d, d) complex structure :math:`J = I_{d/2}\otimes[[0,-1],[1,0]]` of the interleaved layout."""
    J = torch.zeros(d, d, dtype=dtype, device=device)
    J[1::2, 0::2] = torch.eye(d // 2, dtype=dtype, device=device)
    J[0::2, 1::2] = -torch.eye(d // 2, dtype=dtype, device=device)
    return J


def per_head_circularity(
    sigma:      torch.Tensor,            # (..., K, K) SPD covariance
    irrep_dims: List[int],               # even head-block sizes; sum == K
) -> torch.Tensor:                       # (..., H, d_max) singular values, descending, NaN-padded
    r"""Per-head impropriety of a covariance against the complex structure J of ``block_glmc``.

    On head block :math:`\Sigma_h` split :math:`R = (\Sigma + J^\top\Sigma J)/2` (commutes with J)
    and :math:`P = (\Sigma - J^\top\Sigma J)/2` (anticommutes). Returns the singular values of
    :math:`R^{-1/2}PR^{-1/2}`, i.e. :math:`|\mathrm{eig}(R^{-1}P)|`. Under
    :math:`\Sigma\to g\Sigma g^\top` with g in GL(m, C) (g and g^T commute with J) R and P move by
    the same congruence, so these are GL(m, C)-invariant per agent. Both :math:`R\pm P` are SPD, so
    they lie in [0, 1); they vanish exactly for proper (circular) :math:`\Sigma`. For diagonal
    :math:`\Sigma` they are :math:`|s_{2k}-s_{2k+1}|/(s_{2k}+s_{2k+1})`, each twice. Float64.
    """
    if any(d % 2 for d in irrep_dims):
        raise ValueError(f"per_head_circularity needs even head blocks; got {irrep_dims}")
    d_max = max(irrep_dims)
    out = []
    start = 0
    for d in irrep_dims:
        S = sigma[..., start:start + d, start:start + d].to(torch.float64)        # (..., d, d)
        J = _complex_structure(d, dtype=torch.float64, device=S.device)
        S_J = J.T @ S @ J
        R, P = 0.5 * (S + S_J), 0.5 * (S - S_J)
        L = torch.linalg.cholesky(R)
        W = torch.linalg.solve_triangular(L, P, upper=False)                      # L^{-1} P
        W = torch.linalg.solve_triangular(L, W.transpose(-1, -2), upper=False)    # L^{-1} P L^{-T}
        sv = torch.linalg.eigvalsh(0.5 * (W + W.transpose(-1, -2))).abs()
        sv = sv.sort(dim=-1, descending=True).values
        out.append(torch.nn.functional.pad(sv, (0, d_max - d), value=float("nan")))
        start += d
    return torch.stack(out, dim=-2)


def per_head_complex_phase(
    exp_phi:    torch.Tensor,            # (..., K, K) realified GL(m, C)^H frame
    irrep_dims: List[int],               # even head-block sizes; sum == K
) -> torch.Tensor:                       # (..., H) arg det_C in (-pi, pi]
    r"""Per-head :math:`\arg\det_{\mathbb{C}}` of a realified complex frame, the U(1) class function
    that :math:`\log|\det|` cannot see (:math:`\det_{\mathbb{R}} = |\det_{\mathbb{C}}|^2`).

    The complex block is :math:`Z = A + iB` with :math:`A = (X_{00} + X_{11})/2` and
    :math:`B = (X_{10} - X_{01})/2` over the pair sub-blocks :math:`X_{ab}` = ``X[a::2, b::2]``, which is
    exact on the image of GL(m, C). Invariant under a global GL(m, C) conjugation. Complex128.
    """
    if any(d % 2 for d in irrep_dims):
        raise ValueError(f"per_head_complex_phase needs even head blocks; got {irrep_dims}")
    phases = []
    start = 0
    for d in irrep_dims:
        X = exp_phi[..., start:start + d, start:start + d].to(torch.float64)
        A = 0.5 * (X[..., 0::2, 0::2] + X[..., 1::2, 1::2])
        B = 0.5 * (X[..., 1::2, 0::2] - X[..., 0::2, 1::2])
        phases.append(torch.angle(torch.linalg.det(torch.complex(A, B))))
        start += d
    return torch.stack(phases, dim=-1)


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
    eps:     float = 1e-12,
    support: Optional[torch.Tensor] = None,   # bool, broadcastable to beta: admissible keys
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
    across the leading dims to keep the (N, N) mask shape the fit's indexing relies on. Pass the
    prior's ``support`` mask when it is available; ``beta > 0`` also drops keys whose weight
    underflowed to zero (audit 2026-10-07 F14).
    """
    n = beta.shape[-1]
    ii = torch.arange(n, device=beta.device).unsqueeze(-1)
    jj = torch.arange(n, device=beta.device).unsqueeze(0)
    live = (beta > 0) if support is None else support.expand_as(beta)
    admitted = live.reshape(-1, n, n).all(dim=0)                 # prior support, shared (N, N)
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

    Uses seeded random distinct (i, j, k) rather than a deterministic first-512 row-major
    enumeration (which always samples the same low-index-token triangles -- a systematically
    biased curvature estimate), returning the full per-triple distribution, a
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


# --- content-link spectrum (similarity invariants of M_h = O_h T_h) ---
#
# The content link is L_ij^h = U_i M_h U_j^{-1}. Its invariants under the passive left action (U_i -> g_i U_i),
# the common reference change (U_i, S, T, O) -> (U_i a, S a, T a, a^{-1} O), and the per-head value basis
# (T_h, O_h) -> (B T_h, O_h B^{-1}) are the similarity invariants of the tuple (M_h). By
# det(z I_K - M_h) = z^{K-d} det(z I_d - W_h), they are read from the d x d W_h = T_hat_h O_hat_h, which removes
# the K-d structural zeros exactly. Singular values and Frobenius norms of M_h (relational_content_max_singular_value)
# are not invariant under the reference change: M_h -> a^{-1} M_h a.
#
# A rank-deficient head (singular W_h) has min|lambda| = 0, which is the expected value, and log|det W_h| = -inf;
# the mean log-modulus is then written as NaN rather than -inf so that metrics.csv never carries an infinity.
#
# With content_link_edges on, the link of edge j -> i is G_ij Pi_h = g_i e^{L} E_ij Pi_h (vfe3/model/edge_links.py),
# context dependent. The keys below are then computed from the shared generator e^{L} (the B = 0 link), except
# the two selfloop keys, which the model replaces by the edge-aware radius of r I + sum_h beta_ii^h G_ii Pi_h; such
# rows carry relational_edge_link_spectrum_shared_generator = 1.0 (posthoc map rows: link_shared_generator_only).

LINK_SPECTRUM_KEYS: Tuple[str, ...] = (
    "relational_link_spectral_radius_max",
    "relational_link_min_abs_eigenvalue",
    "relational_link_mean_log_abs_eigenvalue",
    "relational_link_unit_distance_max",
    "relational_link_idempotency_defect_max",
    "relational_link_trace",
    "relational_link_trace_sq",
    "relational_link_cross_trace",
    "relational_link_selfloop_spectral_radius_max",
    "relational_link_selfloop_spectral_radius_mean",
)

# Token positions per layer at which content_link_spectrum evaluates the self-loop spectral radius.
SELFLOOP_RADIUS_SAMPLE: int = 256


def _head_transfer(
    T_hat: torch.Tensor,                  # (H, d, K) ambient sending maps
    O_hat: torch.Tensor,                  # (H, K, d) ambient receiving maps
) -> torch.Tensor:                        # (H, H, d, d) W_hg = T_hat_h O_hat_g, float64 CPU
    r"""Head-transfer blocks :math:`W_{hg}=\hat T_h\hat O_g`; spec :math:`W_{hh}` is the nonzero spectrum of
    :math:`M_h=\hat O_h\hat T_h` because :math:`\det(zI_K-M_h)=z^{K-d}\det(zI_d-W_{hh})`."""
    T_hat = T_hat.detach().to("cpu", torch.float64)
    O_hat = O_hat.detach().to("cpu", torch.float64)
    return torch.einsum("hsk,gkt->hgst", T_hat, O_hat)


def _head_transfer_full(
    T_runs: Sequence[torch.Tensor],       # per run (n_r, d_r, K) ambient sending maps
    O_runs: Sequence[torch.Tensor],       # per run (n_r, K, d_r) ambient receiving maps
) -> Tuple[torch.Tensor, List[int]]:      # (K, K) W = T O with blocks W_hg = T_hat_h O_hat_g, head sizes
    r"""The K x K head-transfer matrix :math:`\mathsf W=\hat T\hat O` of a mixed-size head partition.

    :math:`\hat T` is the row stack of the :math:`\hat T_h` and :math:`\hat O` the column stack of the
    :math:`\hat O_h`, so the :math:`(h,g)` block of :math:`\mathsf W` is :math:`W_{hg}`, of size
    :math:`d_h\times d_g`, and :math:`W_{hh}` carries the nonzero spectrum of :math:`M_h`.
    """
    K = T_runs[0].shape[-1]
    T_stack = torch.cat([T.detach().to("cpu", torch.float64).reshape(-1, K) for T in T_runs])          # (K, K)
    O_stack = torch.cat([O.detach().to("cpu", torch.float64).permute(1, 0, 2).reshape(K, -1)
                         for O in O_runs], dim=1)                                                       # (K, K)
    dims = [int(T.shape[1]) for T in T_runs for _ in range(T.shape[0])]
    return T_stack @ O_stack, dims


def _mixed_head_invariants(
    W:    torch.Tensor,                   # (K, K) head-transfer matrix
    dims: List[int],                      # head sizes, sum = K
) -> List[Dict[str, float]]:
    r""":func:`content_link_head_invariants` for heads of different sizes, by slicing :math:`\mathsf W`."""
    offsets = [0]
    for d in dims:
        offsets.append(offsets[-1] + d)
    blocks = [slice(offsets[h], offsets[h + 1]) for h in range(len(dims))]
    sum_rho = float(torch.linalg.eigvals(W).abs().max())
    rows = []
    for h, (rows_h, d) in enumerate(zip(blocks, dims)):
        W_head = W[rows_h, rows_h]                                            # (d_h, d_h)
        lam, vec = torch.linalg.eig(W_head)
        vec = vec / torch.linalg.vector_norm(vec, dim=-2, keepdim=True)
        pair = [float((W[rows_h, cols] * W[cols, rows_h].T).sum()) for cols in blocks]   # Tr(W_hg W_gh)
        logdet = float(torch.linalg.slogdet(W_head).logabsdet)
        rows.append({
            "link_spectral_radius": float(lam.abs().max()),
            "link_min_abs_eigenvalue": float(lam.abs().min()),
            "link_mean_log_abs_eigenvalue": logdet / d if math.isfinite(logdet) else float("nan"),
            "link_unit_distance": float((lam - 1.0).abs().max()),
            "link_idempotency_defect": float((lam * (lam - 1.0)).abs().max()),
            "link_trace": float(torch.diagonal(W_head).sum()),
            "link_trace_sq": pair[h],
            "link_cross_trace": sum(pair) - pair[h],
            "link_sum_spectral_radius": sum_rho,
            "link_eigvec_condition": float(torch.linalg.cond(vec)),
        })
    return rows


def content_link_eigenvalues(
    T_hat: torch.Tensor,                  # (H, d, K)
    O_hat: torch.Tensor,                  # (H, K, d)
) -> torch.Tensor:                        # (H, d) complex128, per head sorted by (-|lam|, -Re, -Im)
    r"""Eigenvalues of :math:`\hat T_h\hat O_h` (the nonzero spectrum of :math:`M_h`), in a fixed order.

    Lexicographic order by successive stable sorts: modulus (rounded to 12 decimals so exact conjugate
    pairs tie) descending, then real part descending, then imaginary part descending.
    """
    W = _head_transfer(T_hat, O_hat)
    lam = torch.linalg.eigvals(torch.diagonal(W, dim1=0, dim2=1).permute(2, 0, 1))   # (H, d)
    for part in ("imag", "real", "abs"):                                             # last pass = primary key
        key = (lam.imag if part == "imag" else lam.real if part == "real"
               else torch.round(lam.abs(), decimals=12))
        lam = lam.gather(-1, torch.sort(-key, dim=-1, stable=True).indices)
    return lam


def content_link_head_invariants(
    T_hat: torch.Tensor,                  # (H, d, K)
    O_hat: torch.Tensor,                  # (H, K, d)
) -> List[Dict[str, float]]:
    r"""Per-head similarity invariants of :math:`M_h=\hat O_h\hat T_h` from :math:`W_h=\hat T_h\hat O_h`.

    ``link_spectral_radius`` :math:`\rho(W_h)`, ``link_min_abs_eigenvalue`` (0 for a rank-deficient head),
    ``link_mean_log_abs_eigenvalue`` :math:`\tfrac1d\log|\det W_h|` (NaN for a singular head, where it is
    :math:`-\infty`), ``link_unit_distance`` :math:`\max|\lambda-1|`,
    ``link_idempotency_defect`` :math:`\max|\lambda(\lambda-1)|=\rho(M_h^2-M_h)`, ``link_trace``
    :math:`\operatorname{Tr}M_h`, ``link_trace_sq`` :math:`\operatorname{Tr}M_h^2`, ``link_cross_trace``
    :math:`\sum_{g\ne h}\operatorname{Tr}M_hM_g`, ``link_sum_spectral_radius`` :math:`\rho(\sum_gM_g)` (one
    value per layer), and ``link_eigvec_condition`` :math:`\kappa_2(V_h)`, a numerics certificate that is
    NOT invariant (it depends on the stored per-head basis, and is infinite or huge for a defective
    :math:`W_h`).
    """
    if not torch.is_tensor(T_hat):
        return _mixed_head_invariants(*_head_transfer_full(T_hat, O_hat))
    W = _head_transfer(T_hat, O_hat)                                          # (H, H, d, d)
    H, d = W.shape[0], W.shape[-1]
    W_head = torch.diagonal(W, dim1=0, dim2=1).permute(2, 0, 1)               # (H, d, d)
    lam, vec = torch.linalg.eig(W_head)                                       # (H, d), (H, d, d)
    vec = vec / torch.linalg.vector_norm(vec, dim=-2, keepdim=True)
    pair = (W * W.permute(1, 0, 3, 2)).sum(dim=(-2, -1))                      # (H, H) Tr(W_hg W_gh)
    logdet = torch.linalg.slogdet(W_head).logabsdet                           # (H,), -inf when singular
    logdet = torch.where(torch.isfinite(logdet), logdet, torch.full_like(logdet, float("nan")))
    sum_rho = float(torch.linalg.eigvals(W.permute(0, 2, 1, 3).reshape(H * d, H * d)).abs().max())
    rows = []
    for h in range(H):
        rows.append({
            "link_spectral_radius": float(lam[h].abs().max()),
            "link_min_abs_eigenvalue": float(lam[h].abs().min()),
            "link_mean_log_abs_eigenvalue": float(logdet[h] / d),
            "link_unit_distance": float((lam[h] - 1.0).abs().max()),
            "link_idempotency_defect": float((lam[h] * (lam[h] - 1.0)).abs().max()),
            "link_trace": float(torch.diagonal(W_head[h]).sum()),
            "link_trace_sq": float(pair[h, h]),
            "link_cross_trace": float(pair[h].sum() - pair[h, h]),
            "link_sum_spectral_radius": sum_rho,
            "link_eigvec_condition": float(torch.linalg.cond(vec[h])),
        })
    return rows


def routing_triangle_defect_sums(
    beta:    torch.Tensor,                    # (B, H, N, N) or (B, N, N) row-stochastic routing
    support: Optional[torch.Tensor] = None,   # bool, broadcastable to beta; None -> causal support
) -> torch.Tensor:                            # (H, 3) float64 [path mass, sum w x, sum w x^2]
    r"""Path-weighted moments of the head-sum link curvature :math:`x_{ijk}=\log\rho_{ijk}` per head.

    For the head-sum link :math:`L_{ij}=U_i\hat O B_{ij}\hat T U_j^{-1}`,
    :math:`B_{ij}=\bigoplus_h\beta^h_{ij}I_d`, the triangle holonomy
    :math:`F_{ijk}=L_{ik}^{-1}L_{ij}L_{jk}` (:math:`Hd=K`, invertible maps) is conjugate to
    :math:`R_{ijk}G` with :math:`G=\hat T\hat O` and :math:`R_{ijk}=\bigoplus_h\rho^h_{ijk}I_d`,

    .. math::
        \rho^h_{ijk}=\frac{\beta^h_{ij}\beta^h_{jk}}{\beta^h_{ik}},\qquad
        \log\rho^h_{ijk}=-\frac{D_{ij}+D_{jk}-D_{ik}}{\tau_h}
        +\log\frac{\pi_{ij}\pi_{jk}}{\pi_{ik}}-\log Z_j .

    The routing is flat on strict triangles exactly when every :math:`x_{ijk}=0`
    (:math:`\beta_{ij}=c_i/c_j`). Over strict triangles :math:`k<j<i` with :math:`j\in\mathcal S_i`,
    :math:`k\in\mathcal S_j\cap\mathcal S_i` and :math:`\beta_{ik}>0`, with the two-step path weight
    :math:`w_{ijk}=\beta_{ij}\beta_{jk}`, returns per head (summed over batch and rows)
    :math:`\sum w`, :math:`\sum w x` and :math:`\sum w x^2`, exactly, by six matrix products
    (no :math:`N^3` tensor): with :math:`A=\log\beta` on the strict support,
    :math:`\sum_j w x=(\beta A)\beta+\beta(\beta A)-A_{ik}\,\beta\beta` and similarly for :math:`x^2`.
    """
    b = beta.double()
    if b.dim() == 3:
        b = b.unsqueeze(1)                                                    # (B, 1, N, N)
    N = b.shape[-1]
    strict = torch.ones(N, N, dtype=torch.bool, device=b.device).tril(-1)    # j < i
    edge = strict if support is None else (support.to(b.device) & strict)
    edge = edge.expand_as(b) & (b > 0.0)                                      # (B, H, N, N)
    Bs = torch.where(edge, b, torch.zeros_like(b))
    A = torch.where(edge, torch.log(torch.where(edge, b, torch.ones_like(b))), torch.zeros_like(b))
    BA = Bs * A
    BA2 = BA * A
    S0 = Bs @ Bs                                                              # (B, H, N, N) sum_j w
    Sa = BA @ Bs                                                              # sum_j w log b_ij
    Sb = Bs @ BA                                                              # sum_j w log b_jk
    s1 = Sa + Sb - A * S0
    s2 = BA2 @ Bs + Bs @ BA2 + 2.0 * (BA @ BA) + A * A * S0 - 2.0 * A * (Sa + Sb)
    keep = edge.to(b.dtype)                                                   # (i, k) needs beta_ik > 0
    out = torch.stack([(S0 * keep).sum(dim=(0, 2, 3)),
                       (s1 * keep).sum(dim=(0, 2, 3)),
                       (s2.clamp_min(0.0) * keep).sum(dim=(0, 2, 3))], dim=-1)
    return out.cpu()                                                          # (H, 3)


def content_link_spectrum(
    T_hat:     torch.Tensor,                    # (H, d, K)
    O_hat:     torch.Tensor,                    # (H, K, d)
    beta_self: Optional[torch.Tensor] = None,   # (..., H, N) self weights beta_ii^h; None -> NaN self-loop keys
    residual:  bool = False,                    # attention_residual: A_ii = I + sum_h beta_ii^h M_h
) -> Dict[str, float]:
    r"""The ten ``relational_link_*`` scalars of one layer (keys and order: ``LINK_SPECTRUM_KEYS``).

    Head reductions of :func:`content_link_head_invariants` plus the self-loop spectral radius
    :math:`\rho(rI+\sum_h\beta_{ii}^hM_h)`, computed as :math:`\rho(rI+D_i\mathsf W)` with
    :math:`D_i=\operatorname{diag}(\beta_{ii}^hI_d)` and :math:`\mathsf W=[W_{hg}]` (:math:`K=Hd`). Under a
    causal prior the maximum over tokens is the spectral radius of the whole-sequence fixed-routing operator.
    The mean log-modulus is NaN when any head is singular.

    The self-loop radius is evaluated on at most :data:`SELFLOOP_RADIUS_SAMPLE` tokens: the evenly spaced
    flat indices ``round(linspace(0, BN - 1, S))`` of the ``(B N)`` token axis (sequence-major, so index 0
    is position 0 of sequence 0), every token when :math:`BN \le S`. ``_mean`` is the mean over that
    subsample. ``_max`` is the maximum over the subsample and, when any token has :math:`D_i=I`
    (:math:`\beta_{ii}^h=1` in every head: position 0 under a causal prior with a self key), the radius
    :math:`\rho(rI+\mathsf W)` those tokens share, so it is exact over the subsample plus every
    :math:`D_i=I` token. Over all tokens it is a lower bound: :math:`\rho(rI+D\mathsf W)` is not monotone in
    :math:`D` for a non-normal :math:`\mathsf W` (a nilpotent :math:`\mathsf W` has :math:`\rho=0` while
    :math:`\rho(D\mathsf W)>0` for some diagonal :math:`0\le D\le I`), so an unsampled token can exceed both.
    Evaluating every token cost 4,096 host eigensolves of :math:`128\times128` matrices per layer and log row
    at the paper base (:math:`BN=4096`, :math:`K=128`; audit 2026-10-10 #29).
    """
    heads = content_link_head_invariants(T_hat, O_hat)
    out = {
        "relational_link_spectral_radius_max": max(r["link_spectral_radius"] for r in heads),
        "relational_link_min_abs_eigenvalue": min(r["link_min_abs_eigenvalue"] for r in heads),
        "relational_link_mean_log_abs_eigenvalue": sum(r["link_mean_log_abs_eigenvalue"] for r in heads) / len(heads),
        "relational_link_unit_distance_max": max(r["link_unit_distance"] for r in heads),
        "relational_link_idempotency_defect_max": max(r["link_idempotency_defect"] for r in heads),
        "relational_link_trace": sum(r["link_trace"] for r in heads),
        "relational_link_trace_sq": sum(r["link_trace_sq"] for r in heads),
        "relational_link_cross_trace": 0.5 * sum(r["link_cross_trace"] for r in heads),
        "relational_link_selfloop_spectral_radius_max": float("nan"),
        "relational_link_selfloop_spectral_radius_mean": float("nan"),
    }
    if beta_self is not None:
        if torch.is_tensor(T_hat):
            W = _head_transfer(T_hat, O_hat)
            H, d = W.shape[0], W.shape[-1]
            W_full = W.permute(0, 2, 1, 3).reshape(H * d, H * d)              # (Hd, Hd)
            repeats: 'int | torch.Tensor' = d
        else:                                                                 # mixed head sizes
            W_full, dims = _head_transfer_full(T_hat, O_hat)                  # (K, K)
            H, repeats = len(dims), torch.tensor(dims)
        b = beta_self.detach().to("cpu", torch.float64)
        b = b.reshape(-1, H, b.shape[-1]).transpose(-1, -2).reshape(-1, H)    # (B N, H)
        shift = 1.0 if residual else 0.0
        count = b.shape[0]
        picks = torch.linspace(0, count - 1, min(count, SELFLOOP_RADIUS_SAMPLE),
                               dtype=torch.float64).round().long()            # (S,) flat token indices
        A = b[picks].repeat_interleave(repeats, dim=-1).unsqueeze(-1) * W_full   # (S, K, K) = D_i W
        rho = (eigvals_on_host(A) + shift).abs().amax(dim=-1)                 # (S,)
        rho_max = float(rho.max())
        # Lower bound over all B N tokens, exact over the subsample and the D_i = I tokens (docstring).
        if bool((b == 1.0).all(dim=-1).any()):                               # some token has D_i W = W
            rho_max = max(rho_max, float((eigvals_on_host(W_full) + shift).abs().max()))
        out["relational_link_selfloop_spectral_radius_max"] = rho_max
        out["relational_link_selfloop_spectral_radius_mean"] = float(rho.mean())
    return out


# --- free-energy closure and per-token profile (the headline F figure) ---


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
    per-sequence ratio design. Tokens of one sequence share context,
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
