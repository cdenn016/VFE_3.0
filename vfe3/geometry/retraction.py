r"""SPD-manifold retractions + Fisher natural-gradient preconditioner (VFE_3.0).

The SPD retraction keeps Sigma on the SPD manifold under a tangent update; the
Fisher preconditioner converts Euclidean (mu, sigma) gradients to natural
gradients. The phi Lie-algebra retraction (retract_phi) lives here too, dispatching to the
GL(K)/SO(N) retractions in lie_ops.py.
"""

import math
from typing import Callable, Dict, Optional, Tuple

import torch

from vfe3.geometry.groups import GaugeGroup
from vfe3.geometry.lie_ops import (
    clamp_phi_trace,
    project_phi_to_slk,
    retract_glk,
    retract_son,
)
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


_RETRACTIONS: Dict[str, Callable[..., torch.Tensor]] = {}


def register_retraction(name: str, *, override: bool = False) -> Callable:
    """Decorator registering an SPD covariance retraction sigma -> sigma_new.

    Duplicate keys fail closed (audit 2026-07-01 round-3): a second registration under an
    existing name silently shadowed the first. Pass ``override=True`` to replace deliberately.
    """
    def _wrap(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        if name in _RETRACTIONS and not override:
            raise KeyError(f"retraction {name!r} already registered; pass override=True to replace")
        _RETRACTIONS[name] = fn
        return fn
    return _wrap


def get_retraction(name: str) -> Callable[..., torch.Tensor]:
    """Return the registered SPD retraction (KeyError-with-available-list if absent)."""
    if name not in _RETRACTIONS:
        raise KeyError(f"no retraction {name!r}; available: {sorted(_RETRACTIONS)}")
    return _RETRACTIONS[name]


class _EighDamped(torch.autograd.Function):
    r"""``torch.linalg.eigh`` with a gap-regularized (Lorentzian-damped) backward.

    The symmetric-eigendecomposition adjoint carries ``1/(lambda_i - lambda_j)`` gap terms that
    diverge on repeated eigenvalues (Higham, *Functions of Matrices* 2008, Sec 3.2). At a degenerate
    spectrum -- e.g. the isotropic ``Sigma = I`` that is the default ``gaussian_full`` prior init --
    the eigenvectors are arbitrary but the downstream function ``V f(lambda) V^T`` (matrix
    sqrt / inv-sqrt / log / exp here) is smooth, so the *true* gradient is finite; only the
    eigendecomposition's intermediate adjoint blows up, poisoning the whole backward with NaN.

    Replacing ``1/Delta`` by the Lorentzian ``Delta / (Delta^2 + gap_eps)`` leaves a well-separated
    spectrum unchanged (relative error ~ ``gap_eps / Delta^2`` for gaps ``Delta``) while damping the
    degenerate gap term to 0 instead of +/-inf -- it kills the (physically immaterial) gradient
    component within a degenerate eigenspace rather than emitting NaN. Forward is bit-identical to
    ``torch.linalg.eigh`` (only the backward is regularized), so every forward-value contract on the
    retractions is preserved. Validated against the stock ``eigh`` backward on well-separated spectra
    (tests/test_retraction.py).
    """

    @staticmethod
    def forward(ctx, A: torch.Tensor, gap_eps):             # gap_eps: float or 0-d tensor  # noqa: D401
        # safe_eigh: value-identical to torch.linalg.eigh, but chunks the flattened batch below the
        # cuSOLVER batched-eigensolver ceiling (audit 2026-09-23 low; see numerics._EIG_MAX_BATCH).
        w, V = safe_eigh(A)
        ctx.save_for_backward(w, V)
        ctx.gap_eps = gap_eps
        return w, V

    @staticmethod
    def backward(ctx, gw: Optional[torch.Tensor], gV: Optional[torch.Tensor]):
        w, V = ctx.saved_tensors
        Vt = V.transpose(-1, -2)
        # delta_ij = w_j - w_i (the symmetric-eigh adjoint's F_ij = 1/(w_j - w_i); the j index is the
        # eigenvector being perturbed). The diagonal (w_i - w_i = 0) gives F_ii = 0 for gap_eps > 0.
        delta = w.unsqueeze(-2) - w.unsqueeze(-1)            # (..., n, n), delta_ij = w_j - w_i
        F = delta / (delta * delta + ctx.gap_eps)           # Lorentzian-damped 1/(w_j - w_i)
        inner = F * (Vt @ gV) if gV is not None else torch.zeros_like(V)
        if gw is not None:
            inner = inner + torch.diag_embed(gw)
        gA = V @ inner @ Vt
        gA = 0.5 * (gA + gA.transpose(-1, -2))              # A is symmetric -> symmetric cotangent
        return gA, None


def _eigh_damped(A: torch.Tensor, gap_eps) -> Tuple[torch.Tensor, torch.Tensor]:  # gap_eps: float or 0-d tensor
    r"""``(eigenvalues, eigenvectors)`` of symmetric ``A`` with a gap-regularized backward (see
    :class:`_EighDamped`). Drop-in for ``torch.linalg.eigh`` on the full-cov SPD retraction paths so a
    degenerate spectrum (the ``Sigma = I`` default init) yields finite -- not NaN -- gradients on the
    unrolled E-step. ``gap_eps`` bounds the worst-case gap factor at ``1/(2 sqrt(gap_eps))``; the
    retraction call sites pass a spectrum-relative value via :func:`_rel_gap_eps`."""
    return _EighDamped.apply(A, gap_eps)


def _rel_gap_eps(
    A:     torch.Tensor,

    *,
    rel:    Optional[float] = None,
    floor:  Optional[float] = None,
) -> torch.Tensor:                         # (..., 1, 1) on-device tensor; no host-sync
    r"""Spectrum-relative ``gap_eps`` for :func:`_eigh_damped` on the SPD retraction paths (audit
    2026-06-13 L11). The fixed ``gap_eps=1e-8`` over-damps the eigh adjoint ``F_ij = 1/(w_j - w_i)``
    for MEANINGFUL gaps near the variance floor -- a resolvable gap of 1e-4 is biased ~50%. Scaling
    to ``(rel * ||A||_max)^2`` -- the squared fp32 noise floor of the spectrum -- damps only gaps
    below fp32 resolution (true degeneracy stays finite, ``F = 0``) and leaves resolvable gaps
    accurate. When omitted, ``rel`` is derived from the active dtype; ``finfo.tiny`` keeps a
    near-zero spectrum finite without imposing a coordinate-scale floor. Passing an explicit
    nonnegative ``floor`` preserves the diagnostic/test API; ``floor=0.0`` exposes the un-clamped
    relative value. Each matrix receives its own scale so unrelated batch elements cannot change
    its eigendecomposition adjoint. Returns ``(..., 1, 1)`` on A's device/dtype to avoid a CUDA
    host-sync."""
    dtype_info = torch.finfo(A.dtype)
    relative = 8.0 * dtype_info.eps if rel is None else rel
    if not math.isfinite(relative) or relative <= 0.0:
        raise ValueError(f"rel must be None or finite and positive, got {rel!r}")
    minimum = dtype_info.tiny if floor is None else floor
    if not math.isfinite(minimum) or minimum < 0.0:
        raise ValueError(f"floor must be None or finite and nonnegative, got {floor!r}")
    scale = A.detach().abs().amax(dim=(-2, -1), keepdim=True)
    return (relative * scale).pow(2).clamp(min=minimum)


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
    r"""Symmetric matrix function with Loewner divided differences in the adjoint."""

    @staticmethod
    def forward(
        ctx,
        matrix: torch.Tensor,
        kind:   str,
        lower:  float,
        upper:  Optional[float],
    ) -> torch.Tensor:
        symmetric = 0.5 * (matrix + matrix.transpose(-1, -2))
        eigenvalues, eigenvectors = _eigh_damped(symmetric, _rel_gap_eps(symmetric))
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
    Two ``_symmetric_spectral_map`` calls ran ``_eigh_damped`` twice on identical input; the two
    functions share a spectrum, so one decomposition serves both (audit 2026-08-05: measured as a
    redundant ``eigh`` over the live ``(B*N, K, K)`` covariance stack on every E-step).

    Forward is bit-identical to the two separate calls -- same symmetrization, same damped
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
        eigenvalues, eigenvectors = _eigh_damped(symmetric, _rel_gap_eps(symmetric))
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
    eigenvalues, eigenvectors = _eigh_damped(repair_input, _rel_gap_eps(repair_input))
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
        lower = torch.full_like(eigenvalues, float(eps)) + clearance
        clipped = torch.maximum(eigenvalues, lower)
        if sigma_max is not None:
            clipped = torch.minimum(clipped, torch.full_like(eigenvalues, float(sigma_max)) - clearance)
        rebuilt = (eigenvectors * clipped.unsqueeze(-2) @ eigenvectors.transpose(-1, -2)).to(symmetric.dtype)
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
    diagonal arm applies to sigma). Uses the gap-regularized ``_eigh_damped`` eigendecomposition so
    the unrolled backward stays finite on a degenerate spectrum -- the isotropic ``Sigma = I`` default
    gaussian_full init makes the stock eigh backward 100% NaN; forward values are unchanged.
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


@register_retraction("spd_affine")
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

    The single registered home for the rank dispatch the E-step used to perform inline: a full
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


def retract_phi(
    phi:          torch.Tensor,           # (..., n_gen) current gauge frame
    delta_phi:    torch.Tensor,           # (..., n_gen) tangent step (e.g. -grad_phi)
    group:        GaugeGroup,             # supplies generators, skew flag, irrep_dims

    *,
    step_size:      float           = 1.0,
    eps:            float           = 1e-6,
    order:          int             = 4,
    project_slk:    bool            = False,
    compact_blocks: bool            = False,
    mode:           str             = "euclidean",

    trust_region:   Optional[float] = None, # None -> group default (GL:0.1, SO:0.3)
    max_norm:       Optional[float] = None, # None -> group default (GL:5.0, SO:pi)
    trace_clamp:    Optional[float] = None, # soft per-block |tr| cap (GL only)
    residual_max:   Optional[float] = None, # opt-in fail-closed BCH-to-group-product residual gate
) -> torch.Tensor:
    r"""Group-aware phi retraction dispatcher (Gaussian-specialized).

    Skew group (SO(N)) -> retract_son, det control is a no-op (det exp = 1).
    Non-skew (GL(K))   -> retract_glk, then optional det control:
      project_slk=True  hard-projects each block to sl(K) (det Omega_h = 1);
      else trace_clamp soft-bounds |tr| per block. Defaults for trust_region /
      max_norm are taken from the group's compactness when not given.
    ``max_norm`` caps the step, not the base point: a row is capped at
    ``max(max_norm, ||phi||)``, so ``R_phi(0) = phi`` for every frame (see ``_retract_core``).
    ``residual_max`` (None = off) forwards to ``compose_bch``'s fail-closed residual gate under
    ``mode='bch'``.
    """
    G = group.generators
    compact_blocks = (
        compact_blocks
        and mode == "bch"
        and group.phi_coordinate_layout == "block_head_row_major"
    )
    if trust_region is None:
        trust_region = 0.3 if group.skew_symmetric else 0.1
    if max_norm is None:
        max_norm = math.pi if group.skew_symmetric else 5.0

    if group.skew_symmetric:
        return retract_son(
            phi, delta_phi, G, step_size=step_size, trust_region=trust_region,
            max_norm=max_norm, eps=eps, order=order, mode=mode, residual_max=residual_max,
        )

    phi_new = retract_glk(
        phi, delta_phi, G, step_size=step_size, trust_region=trust_region,
        max_norm=max_norm, eps=eps, order=order, mode=mode,
        compact_blocks=compact_blocks,
        block_dims=(group.irrep_dims if compact_blocks else None),
        residual_max=residual_max,
    )
    if project_slk:
        phi_new = project_phi_to_slk(phi_new, G, group.irrep_dims)
    elif trace_clamp is not None:
        phi_new = clamp_phi_trace(phi_new, G, group.irrep_dims, trace_max=trace_clamp)
    return phi_new
