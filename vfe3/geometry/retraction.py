r"""SPD-manifold retractions + Fisher natural-gradient preconditioner (VFE_3.0).

The SPD retraction keeps Sigma on the SPD manifold under a tangent update; the
Fisher preconditioner converts Euclidean (mu, sigma) gradients to natural
gradients.
"""

import math
from typing import Dict, Optional, Tuple

import torch

from vfe3.numerics import safe_eigh


def _check_sigma_max(sigma_max: Optional[float], eps: float) -> None:
    r"""Reject an eigenvalue ceiling that would violate the SPD/eps invariant."""
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps must be finite and positive; got {eps!r}")
    if sigma_max is None:
        return
    if not math.isfinite(sigma_max) or sigma_max <= eps:
        raise ValueError(
            f"sigma_max must be None or finite and > eps ({eps}); got {sigma_max!r}"
        )


def _public_spd_bounds(
    dtype:     torch.dtype,
    eps:       float,
    sigma_max: Optional[float],
) -> Tuple[float, Optional[float]]:
    """Return strict representable public-dtype bounds or reject an empty interval."""
    _check_sigma_max(sigma_max, eps)
    lower_tensor = torch.tensor(float(eps), dtype=dtype)
    lower_tensor = torch.nextafter(lower_tensor, torch.full_like(lower_tensor, float("inf")))
    lower = float(lower_tensor)
    if not math.isfinite(lower) or lower <= float(eps):
        raise ValueError(f"eps={eps} has no finite representable strict upper neighbor in {dtype}")
    if sigma_max is None:
        return lower, None
    upper_tensor = torch.tensor(float(sigma_max), dtype=dtype)
    upper_tensor = torch.nextafter(upper_tensor, torch.full_like(upper_tensor, -float("inf")))
    upper = float(upper_tensor)
    if not math.isfinite(upper) or upper <= lower:
        raise ValueError(
            f"[{eps}, {sigma_max}] has no representable strict interior in {dtype}"
        )
    return lower, upper


def _spectral_values_and_derivatives(
    eigenvalues: torch.Tensor,
    kind:        str,
    lower:       float,
    upper:       Optional[float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Return ``f(lambda)`` and ``f'(lambda)`` for one bounded scalar matrix function."""
    if kind == "sqrt_floor":
        bounded = eigenvalues.clamp(min=lower)
        values = torch.sqrt(bounded)
        derivatives = torch.where(
            eigenvalues >= lower,
            0.5 / values,
            torch.zeros_like(values),
        )
    elif kind == "inv_sqrt_floor":
        bounded = eigenvalues.clamp(min=lower)
        values = torch.rsqrt(bounded)
        derivatives = torch.where(
            eigenvalues >= lower,
            -0.5 * values / bounded,
            torch.zeros_like(values),
        )
    elif kind == "exp_bounded":
        if upper is None:
            raise ValueError("exp_bounded requires an upper bound")
        bounded = eigenvalues.clamp(min=lower, max=upper)
        values = torch.exp(bounded)
        active = (eigenvalues >= lower) & (eigenvalues <= upper)
        derivatives = torch.where(active, values, torch.zeros_like(values))
    elif kind == "log_floor":
        bounded = eigenvalues.clamp(min=lower)
        values = torch.log(bounded)
        derivatives = torch.where(
            eigenvalues >= lower,
            bounded.reciprocal(),
            torch.zeros_like(values),
        )
    elif kind == "project":
        values = eigenvalues.clamp(min=lower) if upper is None else eigenvalues.clamp(
            min=lower,
            max=upper,
        )
        active = eigenvalues >= lower
        if upper is not None:
            active = active & (eigenvalues <= upper)
        derivatives = active.to(eigenvalues.dtype)
    else:
        raise ValueError(f"unknown symmetric spectral function {kind!r}")
    return values, derivatives


def _loewner_adjoint(
    eigenvalues:  torch.Tensor,
    eigenvectors: torch.Tensor,
    kind:         str,
    lower:        float,
    upper:        Optional[float],
    grad_output:  torch.Tensor,
) -> torch.Tensor:
    r"""Adjoint of ONE symmetric matrix function, given a shared eigendecomposition.

    Factored out of ``_SymmetricSpectralMap.backward`` so the single- and paired-output
    Functions run byte-identical adjoint code (audit 2026-08-05).
    """
    values, _ = _spectral_values_and_derivatives(
        eigenvalues,
        kind,
        lower,
        upper,
    )
    lambda_i = eigenvalues.unsqueeze(-1)
    lambda_j = eigenvalues.unsqueeze(-2)
    gap = lambda_i - lambda_j
    value_gap = values.unsqueeze(-1) - values.unsqueeze(-2)
    scale = torch.maximum(lambda_i.abs(), lambda_j.abs())
    resolution = (
        8.0 * torch.finfo(eigenvalues.dtype).eps * scale
    ).clamp(min=torch.finfo(eigenvalues.dtype).tiny)
    repeated = gap.abs() <= resolution
    safe_gap = torch.where(repeated, torch.ones_like(gap), gap)
    divided_difference = value_gap / safe_gap
    midpoint = 0.5 * (lambda_i + lambda_j)
    _, repeated_limit = _spectral_values_and_derivatives(
        midpoint,
        kind,
        lower,
        upper,
    )
    # Where both eigenvalues lie on the active (unclamped) interval, the divided difference has a
    # cancellation-free closed form; forming it as value_gap / gap instead loses about
    # eps * f / (f' * gap) relative accuracy, which grows as the tangent shrinks (audit 2026-10-07,
    # F33/F67). A pair with a clamped member keeps the generic quotient (the clamp makes the two
    # values differ by a non-cancelling amount), and an unresolved pair keeps the midpoint limit.
    #   sqrt:     (sqrt a - sqrt b)/(a - b) = 1/(sqrt a + sqrt b)
    #   inv_sqrt: (a^-1/2 - b^-1/2)/(a - b) = -1/(sqrt a sqrt b (sqrt a + sqrt b))
    #   exp:      (e^a - e^b)/(a - b) = e^max(a,b) (-expm1(-|a - b|))/|a - b|
    #   log:      (log a - log b)/(a - b) = log1p((a - b)/b)/(a - b)   (close pairs only)
    # The exp form factors out the LARGER exponent, so it is finite wherever the plain quotient is
    # (e^b expm1(a - b) overflowed float32 for a - b > 88.7 inside the [-50, 50] clamp). The log form
    # is used only for |a - b| <= max(a, b)/2: for a wide pair (a - b)/b rounds to -1 in float32 and
    # log1p returns -inf, while the generic quotient has no cancellation there (audit 2026-10-08,
    # general-A-core-math#3, geometer#3).
    active_i = lambda_i >= lower
    active_j = lambda_j >= lower
    if upper is not None and kind in ("exp_bounded", "project"):
        active_i = active_i & (lambda_i <= upper)
        active_j = active_j & (lambda_j <= upper)
    both_active = active_i & active_j & ~repeated
    if kind == "log_floor":
        both_active = both_active & (gap.abs() <= 0.5 * scale)
    if kind in ("sqrt_floor", "inv_sqrt_floor", "exp_bounded", "log_floor"):
        # Placeholders off the active pairs keep every branch finite (torch.where differentiates both).
        safe_i = torch.where(both_active, lambda_i, torch.ones_like(gap))
        safe_j = torch.where(both_active, lambda_j, torch.ones_like(gap))
        safe_gap_active = torch.where(both_active, gap, torch.ones_like(gap))
        if kind == "sqrt_floor":
            closed = (safe_i.sqrt() + safe_j.sqrt()).reciprocal()
        elif kind == "inv_sqrt_floor":
            root_i, root_j = safe_i.sqrt(), safe_j.sqrt()
            closed = -(root_i * root_j * (root_i + root_j)).reciprocal()
        elif kind == "exp_bounded":
            abs_gap = safe_gap_active.abs()
            closed = torch.exp(torch.maximum(safe_i, safe_j)) * (-torch.expm1(-abs_gap)) / abs_gap
        else:
            closed = torch.log1p(safe_gap_active / safe_j) / safe_gap_active
        divided_difference = torch.where(both_active, closed, divided_difference)
    divided_difference = torch.where(repeated, repeated_limit, divided_difference)

    eigenvectors_t = eigenvectors.transpose(-1, -2)
    symmetric_grad = 0.5 * (grad_output + grad_output.transpose(-1, -2))
    inner = eigenvectors_t @ symmetric_grad @ eigenvectors
    grad_matrix = eigenvectors @ (divided_difference * inner) @ eigenvectors_t
    return 0.5 * (grad_matrix + grad_matrix.transpose(-1, -2))


class _SymmetricSpectralMap(torch.autograd.Function):
    r"""Symmetric matrix function :math:`F(A) = V f(\Lambda) V^T` with the exact Loewner adjoint.

    The backward is the Daleckii-Krein form :math:`\bar A = V (D \circ V^T \bar F V) V^T` with
    :math:`D_{ij} = (f(\lambda_i) - f(\lambda_j)) / (\lambda_i - \lambda_j)` and the repeated-eigenvalue
    limit :math:`D_{ij} = f'(\lambda)` (Daleckii and Krein, AMS Transl. 47, 1965; Bhatia, *Matrix
    Analysis*, 1997, Sec. V.3), built from the saved eigenpairs by :func:`_loewner_adjoint`. It never
    differentiates the eigendecomposition, so it is finite and exact at a degenerate spectrum such as
    the isotropic ``Sigma = I`` init, and the forward needs only the plain ``safe_eigh`` (audit
    2026-10-16, F99).
    """

    @staticmethod
    def forward(
        ctx,
        matrix: torch.Tensor,
        kind:   str,
        lower:  float,
        upper:  Optional[float],
    ) -> torch.Tensor:
        symmetric = 0.5 * (matrix + matrix.transpose(-1, -2))
        # safe_eigh: value-identical to torch.linalg.eigh, but chunks the flattened batch below the
        # cuSOLVER batched-eigensolver ceiling (audit 2026-09-23 low; see numerics._EIG_MAX_BATCH).
        eigenvalues, eigenvectors = safe_eigh(symmetric)
        values, _ = _spectral_values_and_derivatives(eigenvalues, kind, lower, upper)
        ctx.save_for_backward(eigenvalues, eigenvectors)
        ctx.kind = kind
        ctx.lower = lower
        ctx.upper = upper
        return eigenvectors * values.unsqueeze(-2) @ eigenvectors.transpose(-1, -2)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        eigenvalues, eigenvectors = ctx.saved_tensors
        grad_matrix = _loewner_adjoint(
            eigenvalues,
            eigenvectors,
            ctx.kind,
            ctx.lower,
            ctx.upper,
            grad_output,
        )
        return grad_matrix, None, None, None


class _SymmetricSpectralMapPair(torch.autograd.Function):
    r"""TWO symmetric matrix functions of ONE matrix, sharing a single eigendecomposition.

    ``retract_spd_full`` needs both ``Sigma^{1/2}`` and ``Sigma^{-1/2}`` of the SAME ``Sigma``.
    Two ``_symmetric_spectral_map`` calls ran the eigendecomposition twice on identical input; the two
    functions share a spectrum, so one decomposition serves both (audit 2026-08-05: measured as a
    redundant ``eigh`` over the live ``(B*N, K, K)`` covariance stack on every E-step).

    Forward is bit-identical to the two separate calls -- same symmetrization, same ``safe_eigh``
    eigendecomposition, same per-kind value map. Backward returns the SUM of the two adjoints,
    which is exactly what autograd accumulates when one tensor feeds two consumers (IEEE-754
    addition of the two contributions is order-independent, so the sum is bit-identical too).
    """

    @staticmethod
    def forward(
        ctx,
        matrix: torch.Tensor,
        kind_a: str,
        kind_b: str,
        lower:  float,
        upper:  Optional[float],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Without this, autograd MATERIALIZES an undefined output gradient as a zero tensor, so the
        # backward's `if grad_output is None: continue` never fires and a full Loewner adjoint runs
        # against an all-zero cotangent -- the saving the skip branch was written for was never
        # taken (audit 2026-08-06 F33). With it, an output that never reached the loss arrives as
        # None and is genuinely skipped. Values are unaffected: the skipped contribution was
        # identically zero either way.
        ctx.set_materialize_grads(False)
        symmetric = 0.5 * (matrix + matrix.transpose(-1, -2))
        eigenvalues, eigenvectors = safe_eigh(symmetric)
        values_a, _ = _spectral_values_and_derivatives(eigenvalues, kind_a, lower, upper)
        values_b, _ = _spectral_values_and_derivatives(eigenvalues, kind_b, lower, upper)
        ctx.save_for_backward(eigenvalues, eigenvectors)
        ctx.kind_a = kind_a
        ctx.kind_b = kind_b
        ctx.lower = lower
        ctx.upper = upper
        return (
            eigenvectors * values_a.unsqueeze(-2) @ eigenvectors.transpose(-1, -2),
            eigenvectors * values_b.unsqueeze(-2) @ eigenvectors.transpose(-1, -2),
        )

    @staticmethod
    def backward(ctx, grad_a: Optional[torch.Tensor], grad_b: Optional[torch.Tensor]):
        eigenvalues, eigenvectors = ctx.saved_tensors
        grad_matrix = None
        for kind, grad_output in ((ctx.kind_a, grad_a), (ctx.kind_b, grad_b)):
            if grad_output is None:                      # that output never reached the loss
                continue
            contribution = _loewner_adjoint(
                eigenvalues,
                eigenvectors,
                kind,
                ctx.lower,
                ctx.upper,
                grad_output,
            )
            grad_matrix = contribution if grad_matrix is None else grad_matrix + contribution
        return grad_matrix, None, None, None, None


def _symmetric_spectral_map(
    matrix: torch.Tensor,
    kind:   str,

    *,
    lower:  float,
    upper:  Optional[float] = None,
) -> torch.Tensor:
    return _SymmetricSpectralMap.apply(matrix, kind, lower, upper)


def _symmetric_spectral_map_pair(
    matrix: torch.Tensor,
    kind_a: str,
    kind_b: str,

    *,
    lower:  float,
    upper:  Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Both matrix functions of ONE matrix from a single shared eigendecomposition.

    Bit-identical to calling :func:`_symmetric_spectral_map` twice on the same input; see
    :class:`_SymmetricSpectralMapPair`."""
    return _SymmetricSpectralMapPair.apply(matrix, kind_a, kind_b, lower, upper)


class _ClippedSpectralRebuild(torch.autograd.Function):
    r"""Per-row spectral clip :math:`V \min(\max(\Lambda, l_r), u_r) V^T` from a supplied eigendecomposition.

    The SPD repair rebuilds one row at several clearances from ONE eigendecomposition of ``matrix``, so
    the eigenpairs arrive precomputed (detached) and ``matrix`` carries the graph. The adjoint in
    ``matrix`` is the Loewner form of :class:`_SymmetricSpectralMap` for :math:`f(\lambda) =
    \mathrm{clip}(\lambda; l_r, u_r)`: :math:`D_{ij} = 1` on a pair of distinct unclipped eigenvalues,
    the quotient :math:`(f_i - f_j)/(\lambda_i - \lambda_j)` on a pair with a clipped member, and the
    limit :math:`f'(\bar\lambda) = [l_r \le \bar\lambda \le u_r]` at the midpoint of an unresolved pair.
    Differentiating the rebuild through the eigendecomposition instead damped the in-cluster term of a
    degenerate unclipped cluster to zero, against an exact value of 1 (audit 2026-10-16, F39). The
    bounds are differentiable as well: :math:`\partial f_k/\partial l_r = [\lambda_k < l_r]` and
    :math:`\partial f_k/\partial u_r = [\lambda_k > u_r]`, the derivative of the clip as written, so a
    clearance that depends on the row keeps its gradient.
    """

    @staticmethod
    def forward(
        ctx,
        matrix:       torch.Tensor,             # (R, K, K) symmetric; the eigenpairs below decompose it
        eigenvalues:  torch.Tensor,             # (R, K) detached
        eigenvectors: torch.Tensor,             # (R, K, K) detached
        lower:        torch.Tensor,             # (R, 1) per-row floor l_r
        upper:        Optional[torch.Tensor],   # (R, 1) per-row ceiling u_r > l_r, or None
    ) -> torch.Tensor:                          # (R, K, K)
        clipped = torch.maximum(eigenvalues, lower)
        if upper is not None:
            clipped = torch.minimum(clipped, upper)
        ctx.save_for_backward(eigenvalues, eigenvectors, clipped, lower,
                              upper if upper is not None else lower)
        ctx.has_upper = upper is not None
        return eigenvectors * clipped.unsqueeze(-2) @ eigenvectors.transpose(-1, -2)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        eigenvalues, eigenvectors, clipped, lower, upper = ctx.saved_tensors
        has_upper = ctx.has_upper
        eigenvectors_t = eigenvectors.transpose(-1, -2)
        symmetric_grad = 0.5 * (grad_output + grad_output.transpose(-1, -2))
        inner = eigenvectors_t @ symmetric_grad @ eigenvectors           # (R, K, K) V^T G V

        lambda_i = eigenvalues.unsqueeze(-1)                             # (R, K, 1)
        lambda_j = eigenvalues.unsqueeze(-2)                             # (R, 1, K)
        gap = lambda_i - lambda_j                                        # (R, K, K)
        scale = torch.maximum(lambda_i.abs(), lambda_j.abs())
        resolution = (
            8.0 * torch.finfo(eigenvalues.dtype).eps * scale
        ).clamp(min=torch.finfo(eigenvalues.dtype).tiny)
        repeated = gap.abs() <= resolution                               # the rule of _loewner_adjoint
        lower_pair = lower.unsqueeze(-1)                                 # (R, 1, 1)
        upper_pair = upper.unsqueeze(-1)
        active = eigenvalues >= lower                                    # (R, K) unclipped eigenvalues
        if has_upper:
            active = active & (eigenvalues <= upper)
        safe_gap = torch.where(repeated, torch.ones_like(gap), gap)
        divided_difference = (clipped.unsqueeze(-1) - clipped.unsqueeze(-2)) / safe_gap
        both_active = active.unsqueeze(-1) & active.unsqueeze(-2) & ~repeated
        divided_difference = torch.where(both_active, torch.ones_like(gap), divided_difference)
        midpoint = 0.5 * (lambda_i + lambda_j)
        limit = midpoint >= lower_pair
        if has_upper:
            limit = limit & (midpoint <= upper_pair)
        divided_difference = torch.where(repeated, limit.to(gap.dtype), divided_difference)
        grad_matrix = eigenvectors @ (divided_difference * inner) @ eigenvectors_t
        grad_matrix = 0.5 * (grad_matrix + grad_matrix.transpose(-1, -2))

        diagonal = inner.diagonal(dim1=-2, dim2=-1)                      # (R, K) cotangent of each clipped value
        grad_lower = grad_upper = None
        if ctx.needs_input_grad[3]:
            grad_lower = torch.where(eigenvalues < lower, diagonal, torch.zeros_like(diagonal)).sum(-1, keepdim=True)
        if has_upper and ctx.needs_input_grad[4]:
            grad_upper = torch.where(eigenvalues > upper, diagonal, torch.zeros_like(diagonal)).sum(-1, keepdim=True)
        return grad_matrix, None, None, grad_lower, grad_upper


def _certify_public_spd(
    matrix: torch.Tensor,                  # (..., K, K) projected covariance in its public dtype

    *,
    eps:       float,
    sigma_max: Optional[float] = None,
) -> torch.Tensor:
    r"""Certify the represented spectrum and repair only rows outside the public interval.

    Cholesky tests of ``Sigma - eps I`` and, when bounded, ``sigma_max I - Sigma`` certify strict
    represented-value headroom at both ends. The ordinary interior path performs no extra
    eigendecomposition and is byte-identical. A failing row alone is spectrally rebuilt with the
    smallest clearance of a ladder ending at the worst-case public-dtype roundoff bound that passes the
    certificates again; a row no rung certifies takes an interior isotropic fallback.
    """
    fallback_value, _public_upper = _public_spd_bounds(matrix.dtype, eps, sigma_max)
    symmetric = 0.5 * (matrix + matrix.transpose(-1, -2))
    check_dtype = torch.float64
    dimension = symmetric.shape[-1]
    eye_check = torch.eye(dimension, device=symmetric.device, dtype=check_dtype)
    flat = symmetric.reshape(-1, dimension, dimension)
    certificate = flat.to(check_dtype)
    _, lower_info = torch.linalg.cholesky_ex(
        certificate - float(eps) * eye_check,
        check_errors=False,
    )
    floor_failed = lower_info.ne(0)
    ceiling_failed = torch.zeros_like(floor_failed)
    if sigma_max is not None:
        _, upper_info = torch.linalg.cholesky_ex(
            float(sigma_max) * eye_check - certificate,
            check_errors=False,
        )
        ceiling_failed = upper_info.ne(0)
    failed = floor_failed | ceiling_failed
    if not bool(failed.any()):
        return symmetric

    # Count the repaired rows (audit 2026-10-07, F34/F64): a floor repair lifts lambda_min to eps plus
    # the public-dtype clearance, so R_Sigma(0) != Sigma for such a row; the count makes that visible.
    # Ceiling contacts are counted separately (audit 2026-10-08, geometer#4): the 'project' clamp leaves
    # eigenvalues exactly at sigma_max, so the strict sigma_max I - Sigma certificate fails on every
    # row that touched the ceiling, and pooling those rows hid the floor lift the counter exposes.
    _count_public_spd_repairs(floor_failed, ceiling_failed & ~floor_failed)
    repair_input = certificate[failed]
    # The eigenpairs are detached: each rebuild below differentiates through the exact Loewner adjoint
    # of the clip (_ClippedSpectralRebuild, audit 2026-10-16 F39), not through the eigendecomposition.
    with torch.no_grad():
        eigenvalues, eigenvectors = safe_eigh(repair_input)
    scale = repair_input.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1.0)
    margin = scale * (8.0 * dimension * torch.finfo(symmetric.dtype).eps)
    margin_scalar = margin.squeeze(-1).squeeze(-1).unsqueeze(-1)
    if sigma_max is not None:
        margin_scalar = torch.minimum(
            margin_scalar,
            torch.full_like(margin_scalar, (float(sigma_max) - float(eps)) / 4.0),
        )
    fallback_scale = torch.full_like(margin, fallback_value)

    def rebuild(clearance: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Clip the spectrum to [eps + c, sigma_max - c], rebuild in the public dtype, and certify it."""
        lower = float(eps) + clearance                                   # (R, 1)
        upper = None if sigma_max is None else float(sigma_max) - clearance
        rebuilt = _ClippedSpectralRebuild.apply(repair_input, eigenvalues, eigenvectors, lower, upper)
        rebuilt = rebuilt.to(symmetric.dtype)
        rebuilt = 0.5 * (rebuilt + rebuilt.transpose(-1, -2))
        represented = rebuilt.to(check_dtype)
        _, lower_info = torch.linalg.cholesky_ex(represented - float(eps) * eye_check, check_errors=False)
        bad = lower_info.ne(0)
        if sigma_max is not None:
            _, upper_info = torch.linalg.cholesky_ex(float(sigma_max) * eye_check - represented,
                                                     check_errors=False)
            bad = bad | upper_info.ne(0)
        return rebuilt, bad

    # Measured clearance (audit 2026-10-08 P2-25): the bound c = 8 K eps_dtype max(1, max|Sigma_ij|) covers
    # the worst-case roundoff of the public-dtype rebuild, but lifted lambda_min up to 568 times eps on
    # measured rows. Each row takes the smallest clearance of the ladder c 2^-10, c 2^-6, c 2^-3, c whose
    # rebuild passes the same certificates; the last rung is the previous clearance, so a row the old
    # repair certified is still certified, and only rows failing every rung reach the fallback below.
    repaired, repair_failed = rebuild(margin_scalar * 2.0 ** -10)
    for factor in (2.0 ** -6, 2.0 ** -3, 1.0):
        if not bool(repair_failed.any()):
            break
        retry, retry_failed = rebuild(margin_scalar * factor)
        take = repair_failed & ~retry_failed
        repaired = torch.where(take[:, None, None], retry, repaired)
        repair_failed = repair_failed & retry_failed
    fallback = fallback_scale.to(symmetric.dtype) * torch.eye(
        dimension,
        device=symmetric.device,
        dtype=symmetric.dtype,
    )
    repaired = torch.where(
        repair_failed.unsqueeze(-1).unsqueeze(-1),
        fallback,
        repaired,
    )
    final_represented = repaired.to(check_dtype)
    _, final_lower_info = torch.linalg.cholesky_ex(
        final_represented - float(eps) * eye_check,
        check_errors=False,
    )
    final_failed = final_lower_info.ne(0)
    if sigma_max is not None:
        _, final_upper_info = torch.linalg.cholesky_ex(
            float(sigma_max) * eye_check - final_represented,
            check_errors=False,
        )
        final_failed = final_failed | final_upper_info.ne(0)
    if bool(final_failed.any()):
        raise ValueError(
            "the requested covariance interval has no certifiable interior fallback "
            f"in public dtype {symmetric.dtype}"
        )
    output = flat.clone()
    output[failed] = repaired
    return output.reshape_as(symmetric)


_NONFINITE_TANGENT_COUNTS: Dict[str, torch.Tensor] = {}
# Covariance rows rebuilt by _certify_public_spd because they failed the eps-floor certificate, per
# device (async add on the repair path only).
_PUBLIC_SPD_REPAIR_COUNTS: Dict[str, torch.Tensor] = {}
# Rows rebuilt ONLY because they failed the strict sigma_max ceiling certificate (ceiling contacts).
_PUBLIC_SPD_CEILING_COUNTS: Dict[str, torch.Tensor] = {}


def _accumulate_device_count(store: Dict[str, torch.Tensor], mask: torch.Tensor) -> None:
    r"""Add ``mask.sum()`` to the per-device counter in ``store`` (async add, no host sync)."""
    key = str(mask.device)
    counter = store.get(key)
    if counter is None:
        counter = torch.zeros((), dtype=torch.int64, device=mask.device)
        store[key] = counter
    counter += mask.sum()


def _count_public_spd_repairs(floor_failed: torch.Tensor, ceiling_only: torch.Tensor) -> None:
    r"""Accumulate the floor repairs and the ceiling-only contacts of :func:`_certify_public_spd`."""
    _accumulate_device_count(_PUBLIC_SPD_REPAIR_COUNTS, floor_failed)
    _accumulate_device_count(_PUBLIC_SPD_CEILING_COUNTS, ceiling_only)


def public_spd_repair_elements() -> int:
    r"""Covariance rows the SPD certificate repaired at the eps FLOOR since the last reset. Host-syncs.

    These are the rows whose :math:`\lambda_{\min}` the repair lifts; rows that failed only the
    ceiling certificate are reported by :func:`public_spd_ceiling_contact_elements`.
    """
    return int(sum(int(count) for count in _PUBLIC_SPD_REPAIR_COUNTS.values()))


def public_spd_ceiling_contact_elements() -> int:
    r"""Covariance rows rebuilt only for failing the strict ``sigma_max`` ceiling certificate. Host-syncs."""
    return int(sum(int(count) for count in _PUBLIC_SPD_CEILING_COUNTS.values()))


def reset_public_spd_repair_elements() -> None:
    r"""Zero the SPD-certificate floor-repair and ceiling-contact counters."""
    for store in (_PUBLIC_SPD_REPAIR_COUNTS, _PUBLIC_SPD_CEILING_COUNTS):
        for count in store.values():
            count.zero_()


def _neutralize_nonfinite_tangent(
    tangent:    torch.Tensor,               # (..., K) diagonal or (..., K, K) full whitened tangent
    event_ndim: int,                        # 1 diagonal, 2 full -- the trailing axes of ONE element
    valid:      Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Zero any tangent ELEMENT that is not entirely finite, and count it (audit 2026-08-06).

    The trust region below bounds the tangent's NORM, which is the guard against a large step. It is
    NOT a guard against a non-finite one: ``||R||_F`` of a matrix holding a single NaN is NaN, so
    ``clamp(trust_region / (NaN + eps), max=1.0)`` is NaN and the clamp MULTIPLIES the poison through
    instead of stopping it. Downstream the two arms then fail differently and both badly -- the full
    arm hands the NaN to ``torch.linalg.eigh``, which raises
    ``_LinAlgError: the algorithm failed to converge`` and kills the whole run mid-training (observed
    2026-08-06 at step 4253 of an ``e_q_mu_lr=0.01`` sweep cell), while the diagonal arm propagates it
    silently, since ``clamp`` and ``exp`` both pass NaN through and the result is a NaN covariance
    that no later clamp repairs.

    Zeroing the offending element is the conservative recovery and the one the geometry already
    means: a zero tangent retracts along ``exp(0) = I``, i.e. that element's covariance is left
    UNCHANGED for this step, exactly as a fully-binding trust region would in the limit. One bad
    token no longer decides the fate of a multi-hour run.

    It is deliberately NOT silent. Masking a non-finite gradient without saying so would turn a
    crash into a quietly wrong number, which is worse in a research setting: the count accumulates
    per device in a plain on-device tensor -- an async add, NO host sync in the hot path -- and
    :func:`nonfinite_tangent_elements` reads it at whatever cadence the caller already syncs on.
    A nonzero count means some belief received a non-finite natural gradient; the retraction is
    where that surfaced, not where it was caused.
    """
    reduce_axes = tuple(range(-event_ndim, 0))
    finite = torch.isfinite(tangent).all(dim=reduce_axes, keepdim=True)
    if valid is not None:
        finite = finite & valid
    key = str(tangent.device)
    counter = _NONFINITE_TANGENT_COUNTS.get(key)
    if counter is None:
        counter = torch.zeros((), dtype=torch.int64, device=tangent.device)
        _NONFINITE_TANGENT_COUNTS[key] = counter
    counter += (~finite).sum()
    return torch.where(finite, tangent, torch.zeros_like(tangent))


def nonfinite_tangent_elements() -> int:
    r"""Total tangent elements neutralized by :func:`_neutralize_nonfinite_tangent` since the last
    reset. Reads the on-device counters, so it host-syncs: call it at an existing logging cadence,
    not inside the E-step."""
    return int(sum(int(count) for count in _NONFINITE_TANGENT_COUNTS.values()))


def reset_nonfinite_tangent_elements() -> None:
    r"""Zero the neutralized-tangent counters (per-run or per-interval accounting)."""
    for count in _NONFINITE_TANGENT_COUNTS.values():
        count.zero_()


def retract_spd_diagonal(
    sigma_diag:   torch.Tensor,             # (..., K) diagonal variances
    delta_sigma:  torch.Tensor,             # (..., K) diagonal tangent

    *,
    step_size:    float          = 1.0,
    trust_region: float          = 5.0,
    eps:          float          = 1e-6,
    sigma_max:    Optional[float] = 10.0,   # matches VFE3Config.sigma_max
) -> torch.Tensor:
    r"""Diagonal SPD retraction sigma_new = sigma * exp(R), R = tau dsigma/sigma.

    ``trust_region`` projects the step-scaled whitened diagonal vector into one L2 ball. Its norm is
    exactly the Frobenius norm of the corresponding diagonal matrix, so the shared trust value has
    the same geometric meaning as the full-SPD arm. Positivity is by construction (exp > 0);
    clamped to [eps, sigma_max].
    When sigma_max is None the eigenvalue ceiling is skipped (pure-path: eps floor only).
    """
    orig_dtype = sigma_diag.dtype
    lower_bound, upper_bound = _public_spd_bounds(orig_dtype, eps, sigma_max)
    with torch.amp.autocast(sigma_diag.device.type, enabled=False):     # tensor-keyed (audit 2026-07-05 m10)
        # float64 stays float64 (audit 2026-07-12 N12, the retract_spd_full F12 policy); half promotes to fp32.
        compute_dtype = torch.float64 if orig_dtype == torch.float64 else torch.float32
        sigma_safe = sigma_diag.to(compute_dtype).clamp(min=eps)
        delta_sigma = delta_sigma.to(compute_dtype)
        whitened = delta_sigma / sigma_safe
        exp_arg = step_size * whitened
        # Before the NORM guard, because a non-finite entry makes that norm NaN and the clamp then
        # multiplies the poison through rather than bounding it (see the helper).
        exp_arg = _neutralize_nonfinite_tangent(exp_arg, event_ndim=1)
        if trust_region is not None and trust_region > 0:
            tangent_norm = torch.linalg.vector_norm(exp_arg, dim=-1, keepdim=True)
            exp_arg = exp_arg * torch.clamp(trust_region / (tangent_norm + eps), max=1.0)
        exp_arg = exp_arg.clamp(-50.0, 50.0)
        sigma_new = sigma_safe * torch.exp(exp_arg)
    sigma_new = (
        sigma_new.clamp(min=lower_bound)
        if upper_bound is None
        else sigma_new.clamp(min=lower_bound, max=upper_bound)
    )
    return sigma_new.to(orig_dtype)


def retract_spd_full(
    sigma:        torch.Tensor,             # (..., K, K) SPD covariances
    delta_sigma:  torch.Tensor,             # (..., K, K) symmetric tangent

    *,
    step_size:    float          = 1.0,
    trust_region: float          = 2.0,
    eps:          float          = 1e-6,
    sigma_max:    Optional[float] = 10.0,   # matches VFE3Config.sigma_max
) -> torch.Tensor:
    r"""Full SPD retraction via the affine-invariant exponential map.

        Sigma_new = S^{1/2} exp(S^{-1/2} (tau dSigma) S^{-1/2}) S^{1/2},
    with one Frobenius-ball trust region on the whole whitened tangent matrix and an eigenvalue
    floor/ceiling [eps, sigma_max] (eigenvalues are variances: the SAME physical ceiling the
    diagonal arm applies to sigma). Every matrix function is differentiated by its exact Loewner
    (Daleckii-Krein) adjoint with the repeated-eigenvalue limit :math:`f'(\lambda)` (see
    :class:`_SymmetricSpectralMap`), so the unrolled backward is finite and exact on a degenerate
    spectrum -- at the isotropic ``Sigma = I`` default gaussian_full init the stock eigh backward is
    100% NaN.
    When sigma_max is None the eigenvalue ceiling is skipped (pure-path: eps floor only).
    """
    _check_sigma_max(sigma_max, eps)
    orig_shape = sigma.shape
    orig_dtype = sigma.dtype
    _public_spd_bounds(orig_dtype, eps, sigma_max)
    if sigma.dim() == 4:
        B, N, K, _ = sigma.shape
        sigma = sigma.reshape(B * N, K, K)
        delta_sigma = delta_sigma.reshape(B * N, K, K)

    with torch.amp.autocast(sigma.device.type, enabled=False):          # tensor-keyed (audit 2026-07-05 m10)
        compute_dtype = torch.float64 if orig_dtype == torch.float64 else torch.float32
        sigma = sigma.to(dtype=compute_dtype)
        delta_sigma = delta_sigma.to(dtype=compute_dtype)
        sigma = 0.5 * (sigma + sigma.transpose(-1, -2))
        delta_sigma = 0.5 * (delta_sigma + delta_sigma.transpose(-1, -2))

        # ONE damped eigendecomposition serves both halves: sqrt and inv-sqrt of the SAME sigma
        # share a spectrum (audit 2026-08-05 -- the two separate calls ran eigh twice over the
        # live (B*N, K, K) stack on every E-step). Bit-identical, forward and backward.
        sigma_sqrt, sigma_inv_sqrt = _symmetric_spectral_map_pair(
            sigma,
            "sqrt_floor",
            "inv_sqrt_floor",
            lower=eps,
        )

        R = sigma_inv_sqrt @ (step_size * delta_sigma) @ sigma_inv_sqrt
        R = 0.5 * (R + R.transpose(-1, -2))
        # Guard R itself, not delta_sigma: sigma_inv_sqrt reaches 1/sqrt(eps) at the variance floor,
        # so a finite-but-extreme tangent can overflow to +-inf HERE, and eigh cannot be handed
        # either that or a NaN. Placed before the norm guard for the reason given in the helper.
        R = _neutralize_nonfinite_tangent(R, event_ndim=2)
        if trust_region is not None and trust_region > 0:
            R_norm = torch.linalg.norm(R, ord='fro', dim=(-2, -1), keepdim=True)
            R = R * torch.clamp(trust_region / (R_norm + eps), max=1.0)

        exp_R = _symmetric_spectral_map(
            R,
            "exp_bounded",
            lower=-50.0,
            upper=50.0,
        )

        sigma_new = sigma_sqrt @ exp_R @ sigma_sqrt
        sigma_new = 0.5 * (sigma_new + sigma_new.transpose(-1, -2))

        sigma_new = _symmetric_spectral_map(
            sigma_new,
            "project",
            lower=eps,
            upper=sigma_max,
        )

    sigma_new = _certify_public_spd(
        sigma_new.to(orig_dtype),
        eps=eps,
        sigma_max=sigma_max,
    )
    if len(orig_shape) == 4:
        sigma_new = sigma_new.reshape(orig_shape)
    return sigma_new


def retract_spd_affine(
    sigma:        torch.Tensor,             # (..., K) diagonal OR (..., K, K) full covariance
    delta_sigma:  torch.Tensor,             # matching rank: the tangent step (e.g. -e_q_sigma_lr * nat_sigma)

    mean_ndim:    int,                      # ndim of the belief mean; full cov iff sigma.dim() == mean_ndim + 1

    *,
    step_size:    float          = 1.0,
    trust_region: float          = 5.0,
    eps:          float          = 1e-6,
    sigma_max:    Optional[float] = 10.0,   # matches VFE3Config.sigma_max
) -> torch.Tensor:                          # (...) same rank as sigma
    r"""Affine-invariant SPD retraction (the default; Pennec, Fillard and Ayache, Int. J. Comput. Vision 66(1), 2006, Sec. 3.3).

    The single home for the rank dispatch the E-step used to perform inline: a full
    covariance (sigma.dim() == mean_ndim + 1) steps along the affine-invariant geodesic via
    ``retract_spd_full``; a diagonal sigma (matching the mean rank) uses ``retract_spd_diagonal``.
    Both are the same affine-invariant exponential map,
        Sigma_new = Sigma^{1/2} exp(Sigma^{-1/2} (step_size dSigma) Sigma^{-1/2}) Sigma^{1/2},
    reduced to sigma_new = sigma exp(step_size dsigma/sigma) on the diagonal cone. The dispatcher
    forwards to the bare functions; the Fisher metric conversion stays in the E-step
    (``natural_gradient``), so the tangent ``delta_sigma`` arrives already preconditioned. The shared
    ``trust_region`` bounds the L2 norm of the step-scaled diagonal tangent and the Frobenius norm of
    the step-scaled full tangent, which are equal when the full tangent is diagonal.
    """
    if sigma.dim() == mean_ndim + 1:                     # full covariance (..., K, K)
        return retract_spd_full(
            sigma, delta_sigma, step_size=step_size, trust_region=trust_region, eps=eps, sigma_max=sigma_max,
        )
    return retract_spd_diagonal(                          # diagonal variances (..., K)
        sigma, delta_sigma, step_size=step_size, trust_region=trust_region, eps=eps, sigma_max=sigma_max,
    )




def natural_gradient(
    grad_mu:    torch.Tensor,                    # (..., K) Euclidean grad wrt mu
    grad_sigma: Optional[torch.Tensor],          # None freezes sigma; else Euclidean grad wrt sigma
    sigma_q:    torch.Tensor,                    # (..., K) diagonal OR (..., K, K) full covariance

    *,
    eps:        float = 1e-6,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""Fisher preconditioner: Euclidean -> natural gradient for a Gaussian.

        nat_mu    = Sigma grad_mu
        nat_sigma = 2 Sigma grad_sigma Sigma   (diagonal: 2 sigma^2 grad_sigma)
    The Fisher metric on Sigma is g(dS1,dS2) = (1/2) tr(S^-1 dS1 S^-1 dS2), so
    g^{kk} = 2 sigma_k^2 in the diagonal case.

    Diagonal vs full is detected by ``sigma_q.dim() == grad_mu.dim()`` (diagonal
    sigma matches the mean rank; full sigma has one extra trailing dim). This is
    correct for the belief shapes used here ((B,N,K) diagonal, (B,N,K,K) full);
    a caller passing other ranks must pass matching mean/sigma ranks.
    """
    is_diagonal = sigma_q.dim() == grad_mu.dim()
    orig_dtype = sigma_q.dtype
    with torch.amp.autocast(sigma_q.device.type, enabled=False):        # tensor-keyed (audit 2026-07-05 m10)
        # float64 stays float64 (audit 2026-07-12 N12; keyed on sigma_q, the metric); half promotes to fp32.
        compute_dtype = torch.float64 if orig_dtype == torch.float64 else torch.float32
        sigma_q    = sigma_q.to(compute_dtype)
        grad_mu    = grad_mu.to(compute_dtype)
        grad_sigma = None if grad_sigma is None else grad_sigma.to(compute_dtype)
        if is_diagonal:
            sigma_safe     = sigma_q.clamp(min=eps)
            nat_grad_mu    = sigma_safe * grad_mu
            nat_grad_sigma = (None if grad_sigma is None
                              else 2.0 * sigma_safe * sigma_safe * grad_sigma)
        else:
            nat_grad_mu    = torch.einsum('...ij,...j->...i', sigma_q, grad_mu)
            if grad_sigma is None:
                nat_grad_sigma = None
            else:
                nat_grad_sigma = 2.0 * torch.einsum('...ij,...jk,...kl->...il', sigma_q, grad_sigma, sigma_q)
                nat_grad_sigma = 0.5 * (nat_grad_sigma + nat_grad_sigma.transpose(-1, -2))
    return nat_grad_mu.to(orig_dtype), (None if nat_grad_sigma is None
                                        else nat_grad_sigma.to(orig_dtype))
