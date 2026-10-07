r"""Frobenius-pullback frame (phi) direction for the ``pullback_group`` M-step (Gaussian-specialized).

Solves the per-block Riemannian direction under the Frobenius-pullback metric of the exponential
map, :math:`G_{ab}(\phi) = \langle d\exp_\phi(T_a), d\exp_\phi(T_b)\rangle_F`, registered as
``pullback_per_block`` in a group-direction registry keyed by a gauge group's
``pullback_capability``. For a block-diagonal group (``block_glk``) the metric is exactly
block-diagonal, since :math:`d\exp_\phi(T_a)` stays in the block of :math:`T_a`; the per-block
solve is therefore the full pullback metric, not an approximation of it. Coordinates in,
coordinates out (..., n_gen). The module also holds the U-space helpers of the ``group_log``
retraction.

The pullback conditioner is not a natural gradient in Amari's sense (Amari 1998, Neural
Computation 10(2)): it is the ambient flat Frobenius form pulled back through d exp, not a Fisher
metric of a statistical model over phi. "Natural gradient" is reserved for the belief-Fisher frame
step (``m_phi_update_mode='fisher_group'``) and the Gaussian-belief step
(``retraction.natural_gradient``).
"""

from collections import OrderedDict
from dataclasses import dataclass
import inspect
import weakref
from typing import Callable, Dict, List, Optional

import torch

from vfe3.geometry.matrix_log import logm_frechet_pair, principal_strip_certificate
from vfe3.numerics import safe_eigvalsh


@dataclass(frozen=True)
class PullbackGroupDirectionResult:
    v_phi:                               torch.Tensor
    xi:                                  torch.Tensor
    min_undamped_generalized_eigenvalue: torch.Tensor
    undamped_generalized_condition:      torch.Tensor
    damped_generalized_condition:        torch.Tensor
    scaled_solve_residual:               torch.Tensor
    series_order:                        int
    # Full (..., d, d), or equal direct-sum blocks (..., H, d, d).
    current_element:                     Optional[torch.Tensor] = None
    # (...) bool, set only under reject_ill_conditioned_rows: rows whose metric passed the
    # undamped-eigenvalue floor and the damped-condition gate. Rejected rows carry v_phi = xi = 0.
    accepted_rows:                       Optional[torch.Tensor] = None


@dataclass(frozen=True)
class _StrictBasisPreparation:
    basis:             torch.Tensor
    structure:         torch.Tensor
    gram:              torch.Tensor
    gram_factor:       torch.Tensor
    gram_inverse_half: torch.Tensor
    gram_is_identity:  bool


@dataclass(frozen=True)
class _StrictBlockPreparation:
    index:               torch.Tensor
    basis:               _StrictBasisPreparation
    matches_first_basis: bool


class _MixedPhiDifferentialOrder(ValueError):
    """Signal that leading strict blocks have different first certified orders."""


_PHI_GROUP_DIRECTIONS: Dict[str, Callable[..., PullbackGroupDirectionResult]] = {}
_PHI_GROUP_DIRECTION_ACCEPTS_COORDINATE_LAYOUT: Dict[str, bool] = {}
_PHI_GROUP_DIRECTION_ACCEPTS_DIFFERENTIAL_MODE: Dict[str, bool] = {}
_PHI_GROUP_DIRECTION_ACCEPTS_ROW_REJECTION: Dict[str, bool] = {}
_PHI_GROUP_DIRECTION_REQUIRES_SOURCE_BASIS: Dict[str, bool] = {}

# Candidate orders at which the tail certificate below is TESTED. The floor is not a precision
# setting -- the certificate is what bounds the error, and it is unchanged -- it only decides how
# early the loop is allowed to notice it has already converged (audit 2026-08-07).
#
# It was 40. Measured against the live block_glk structure constants at phi_scale=0.06, the
# geometric majorant certifies by order 17 (raw term magnitude crosses 1e-12 by order 12), so 40
# ran ~23 redundant matmuls of the full (n_blocks, active_rows, n_gen, n_gen) workspace -- 78.5% of
# the staging call at 256 active rows. In the other direction 40 is not conservative either: at 8x
# the live phi scale the certificate demands 58, and the floor cannot help there because it is a
# floor, not a target. Testing from 8 in steps of 4 costs at most a few extra certificate
# evaluations and returns the first order the SAME majorant accepts.
_PHI_GROUP_MIN_SERIES_ORDER: int   = 8
_PHI_GROUP_MAX_SERIES_ORDER: int   = 128
_PHI_GROUP_SERIES_ORDER_STEP: int  = 4
_PHI_GROUP_TAIL_TOL: float         = 1e-12
_PHI_GROUP_GRAM_RIDGE: float       = 1e-6
_PHI_GROUP_MIN_UNDAMPED_EIG: float = 1e-8
_PHI_GROUP_MAX_DAMPED_COND: float  = 1e6
_PHI_GROUP_SOLVE_RESIDUAL: float   = 1e-10

# How the trivialized exponential differentials Psi_R / Psi_L are evaluated (audit 2026-08-08).
#
# "series"   -- the truncated power series with the geometric-majorant certificate. The historical
#               default, and still the default: it is basis-agnostic, real-arithmetic, and has no
#               conditioning failure mode.
# "spectral" -- the exact Daleckii-Krein closed form. Psi(ad_X) is diagonal in the eigenbasis of
#               ad_X, whose eigenvalues are lam_i - lam_j for lam = eig(X), so ONE d x d
#               eigendecomposition replaces the whole matmul ladder. Exact in exact arithmetic:
#               there is no truncation and hence no order to certify.
#
# OPT-IN, because the closed form trades a truncation error for a CONDITIONING one. It needs
# P = eigenvectors(X), and X is a general real matrix built from an unconstrained gradient: it can
# be defective (P singular, no eigenbasis at all) and is near-defective on a set of positive
# measure. Measured loss is about 2*log10(kappa(P)) digits. At the live phi scale kappa(P) has
# median 11 / p99 70 / max 117 over 400 draws, i.e. ~12 digits retained -- comfortable -- but the
# guards below fall back to the series rather than trusting that in general.
_PHI_GROUP_DIFFERENTIAL_MODES = ("series", "spectral")
_PHI_GROUP_DIFFERENTIAL_MODE: str = "series"
# kappa_F(P) = ||P||_F ||P^-1||_F, an upper bound on kappa_2(P), so the trigger is conservative:
# it can fall back when the true conditioning is fine, never the reverse.
_PHI_GROUP_SPECTRAL_MAX_COND: float = 1e6
# Two basis changes amplify floating-point error by kappa(P)^2. This budget
# supplements invertibility; it is not a forward-error certificate for eig.
_PHI_GROUP_SPECTRAL_ROUNDOFF_BUDGET: float = 1e-10
# Bound the main float64 n_gen-by-n_gen workspaces, including temporaries.
_PHI_GROUP_WORKSPACE_BYTES: int = 64 * 1024 * 1024
_PHI_GROUP_DENSE_WORKSPACES: int = 12
# Psi(ad_X) is a real operator on a real algebra, so its coordinate matrix must come out real. A
# nonneglible imaginary residue is the eigendecomposition telling us it failed -- a free check on
# tensors already computed, and the reason the spectral route can fail closed rather than silently.
_PHI_GROUP_SPECTRAL_MAX_IMAG: float = 1e-8

_STRICT_PULLBACK_PREP_CACHE_MAXSIZE: int = 32
_STRICT_PULLBACK_PREP_CACHE: OrderedDict[
    tuple,
    tuple[
        weakref.ReferenceType[torch.Tensor],
        tuple[_StrictBlockPreparation, ...],
    ],
] = OrderedDict()


def register_phi_group_direction(
    name: str,

    *,
    override:              bool = False,
    preserve_source_basis: bool = False,
) -> Callable:
    """Register one strict M-step pullback-group direction rule."""
    def _wrap(
        fn: Callable[..., PullbackGroupDirectionResult],
    ) -> Callable[..., PullbackGroupDirectionResult]:
        if name in _PHI_GROUP_DIRECTIONS and not override:
            raise KeyError(
                f"phi group direction {name!r} already registered; pass override=True to replace"
            )
        try:
            parameters = tuple(inspect.signature(fn).parameters.values())
            accepts_coordinate_layout = any(
                (
                    parameter.name == "coordinate_layout"
                    and parameter.kind in (
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    )
                )
                or parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters
            )
            accepts_differential_mode = any(
                (
                    parameter.name == "differential_mode"
                    and parameter.kind in (
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    )
                )
                or parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters
            )
            accepts_row_rejection = any(
                parameter.name == "reject_ill_conditioned_rows"
                and parameter.kind in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
                for parameter in parameters
            )
        except (TypeError, ValueError):
            accepts_coordinate_layout = False
            accepts_differential_mode = False
            accepts_row_rejection = False
        _PHI_GROUP_DIRECTIONS[name] = fn
        _PHI_GROUP_DIRECTION_ACCEPTS_COORDINATE_LAYOUT[name] = accepts_coordinate_layout
        _PHI_GROUP_DIRECTION_ACCEPTS_DIFFERENTIAL_MODE[name] = accepts_differential_mode
        _PHI_GROUP_DIRECTION_ACCEPTS_ROW_REJECTION[name] = accepts_row_rejection
        _PHI_GROUP_DIRECTION_REQUIRES_SOURCE_BASIS[name] = bool(preserve_source_basis)
        return fn
    return _wrap


def get_phi_group_direction(
    name: str,
) -> Callable[..., PullbackGroupDirectionResult]:
    """Return a registered strict M-step pullback-group direction rule."""
    if name not in _PHI_GROUP_DIRECTIONS:
        raise KeyError(
            f"no phi group direction {name!r}; available: {sorted(_PHI_GROUP_DIRECTIONS)}"
        )
    return _PHI_GROUP_DIRECTIONS[name]


def pullback_group_direction(
    grad_phi:   torch.Tensor,             # (..., n_gen) processed outer-objective covector
    phi:        torch.Tensor,             # (..., n_gen) current chart coordinates
    generators: torch.Tensor,             # (n_gen, K, K) registered basis

    *,
    mode:              str,
    coordinate_layout: Optional[str]       = None,
    differential_mode: Optional[str]       = None,
    irrep_dims:        Optional[List[int]] = None,
    reject_ill_conditioned_rows: bool      = False,  # per-row rejection instead of a whole-batch raise
) -> PullbackGroupDirectionResult:
    r"""Return the strict M-step direction in an autocast-disabled float64 island.

    The chart covector is solved against the Gram-relative pullback system
    ``(J(phi)^T J(phi) + 1e-6 B) v_phi = grad_phi`` and then converted to the
    left-trivialized group velocity ``xi = Psi_L(ad_phi) v_phi``.

    ``reject_ill_conditioned_rows=True`` (audit 2026-09-23 #10; the optimizer's staging route)
    turns the undamped-eigenvalue floor and the ``1e6`` damped-condition gate into a per-row
    verdict: failing rows return ``v_phi = xi = 0`` and ``accepted_rows`` marks the rest. A
    nonfinite metric spectrum, and a Cholesky or solve-residual failure on an ACCEPTED row, still
    raise. The default keeps the whole-batch fail-closed contract.
    """
    with torch.autocast(device_type=grad_phi.device.type, enabled=False):
        direction = get_phi_group_direction(mode)
        kwargs = {"irrep_dims": irrep_dims}
        if _PHI_GROUP_DIRECTION_ACCEPTS_COORDINATE_LAYOUT.get(mode, False):
            kwargs["coordinate_layout"] = coordinate_layout
        if differential_mode is not None and differential_mode not in _PHI_GROUP_DIFFERENTIAL_MODES:
            raise ValueError(f"invalid phi group differential mode {differential_mode!r}")
        if _PHI_GROUP_DIRECTION_ACCEPTS_DIFFERENTIAL_MODE.get(mode, False):
            kwargs["differential_mode"] = differential_mode
        if reject_ill_conditioned_rows and _PHI_GROUP_DIRECTION_ACCEPTS_ROW_REJECTION.get(mode, False):
            kwargs["reject_ill_conditioned_rows"] = True
        direction_generators = (
            generators
            if _PHI_GROUP_DIRECTION_REQUIRES_SOURCE_BASIS.get(mode, False)
            else generators.to(device=phi.device, dtype=torch.float64)
        )
        return direction(grad_phi, phi, direction_generators, **kwargs)


def _strict_structure_constants(
    generators: torch.Tensor,             # (n_gen, K, K)

    *,
    closure_tol: float = 1e-4,
    max_k:       int   = 12,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Return certified bracket coordinates and the generator Gram in float64.

    For ``B_cd = <G_c, G_d>_F`` and ``[G_a, G_b] = f_ab^c G_c``, the coordinates
    solve ``B_cd f_ab^d = <G_c, [G_a, G_b]>_F`` through the Cholesky factor of
    ``B``. Reconstruction of every bracket certifies closure of the span.
    """
    K = generators.shape[-1]
    if K > max_k:
        raise ValueError(f"strict phi group direction: K={K} exceeds max_k={max_k}")
    basis = generators.to(dtype=torch.float64)
    gram = torch.einsum("aij,bij->ab", basis, basis)
    gram_factor, info = torch.linalg.cholesky_ex(gram)
    if bool((info != 0).any()):
        raise ValueError("strict phi group direction requires an independent generator basis")
    bracket = (
        torch.einsum("aij,bjk->abik", basis, basis)
        - torch.einsum("bij,ajk->abik", basis, basis)
    )
    pairings = torch.einsum("cij,abij->abc", basis, bracket)
    n_gen = basis.shape[0]
    coordinates = torch.cholesky_solve(
        pairings.reshape(-1, n_gen).transpose(0, 1),
        gram_factor,
    ).transpose(0, 1).reshape(n_gen, n_gen, n_gen)
    reconstructed = torch.einsum("abc,cij->abij", coordinates, basis)
    residual = torch.linalg.matrix_norm(reconstructed - bracket, dim=(-2, -1))
    bracket_norm = torch.linalg.matrix_norm(bracket, dim=(-2, -1))
    relative = residual / (bracket_norm + torch.finfo(torch.float64).eps)
    if not bool(torch.isfinite(relative).all()) or float(relative.max()) > closure_tol:
        raise ValueError(
            "strict phi group direction requires a bracket-closed generator span; "
            f"relative reconstruction residual={float(relative.max()):.3e}"
        )
    return coordinates, gram


def _adaptive_phi_differentials(
    ad: torch.Tensor,                     # (..., n_gen, n_gen)

    *,
    require_uniform_leading_batch: bool          = False,
    fixed_order:                   Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    r"""Evaluate certified right and left trivialized exponential differentials.

    The right series is ``Psi_R(z) = (exp(z) - 1) / z = sum z^k/(k+1)!`` and
    the left series is ``Psi_L(z) = (1 - exp(-z)) / z = sum (-z)^k/(k+1)!``.
    A geometric majorant for the first omitted term certifies both series at
    candidate orders from 8 through 128. A fixed order retains the same tail
    certificate and arithmetic when independently evaluated chunks are reunited.
    """
    if fixed_order is not None and (
        fixed_order < _PHI_GROUP_MIN_SERIES_ORDER
        or fixed_order > _PHI_GROUP_MAX_SERIES_ORDER
        or fixed_order % _PHI_GROUP_SERIES_ORDER_STEP
    ):
        raise ValueError("invalid fixed phi differential series order")
    n_gen = ad.shape[-1]
    eye = torch.eye(n_gen, dtype=ad.dtype, device=ad.device).expand_as(ad).clone()
    psi_right = eye.clone()
    psi_left = eye.clone()
    term = eye.clone()
    norm_one = torch.linalg.matrix_norm(ad, ord=1, dim=(-2, -1))
    norm_inf = torch.linalg.matrix_norm(ad, ord=float("inf"), dim=(-2, -1))
    use_one = norm_one <= norm_inf
    alpha = torch.where(use_one, norm_one, norm_inf)
    bound_k = torch.ones_like(alpha)
    for k in range(1, _PHI_GROUP_MAX_SERIES_ORDER):
        term = torch.matmul(term, ad) / float(k + 1)
        psi_right = psi_right + term
        psi_left = psi_left + (term if k % 2 == 0 else -term)
        bound_k = bound_k * alpha / float(k + 1)
        order = k + 1
        if order < _PHI_GROUP_MIN_SERIES_ORDER or order % _PHI_GROUP_SERIES_ORDER_STEP:
            continue
        if fixed_order is not None and order != fixed_order:
            continue
        first_omitted = bound_k * alpha / float(order + 1)
        ratio = alpha / float(order + 2)
        tail = torch.where(
            ratio < 1.0,
            first_omitted / (1.0 - ratio),
            torch.full_like(ratio, torch.inf),
        )
        right_one = torch.linalg.matrix_norm(psi_right, ord=1, dim=(-2, -1))
        right_inf = torch.linalg.matrix_norm(psi_right, ord=float("inf"), dim=(-2, -1))
        left_one = torch.linalg.matrix_norm(psi_left, ord=1, dim=(-2, -1))
        left_inf = torch.linalg.matrix_norm(psi_left, ord=float("inf"), dim=(-2, -1))
        right_norm = torch.where(use_one, right_one, right_inf)
        left_norm = torch.where(use_one, left_one, left_inf)
        right_ok = tail <= _PHI_GROUP_TAIL_TOL * torch.maximum(torch.ones_like(tail), right_norm)
        left_ok = tail <= _PHI_GROUP_TAIL_TOL * torch.maximum(torch.ones_like(tail), left_norm)
        certified = right_ok & left_ok & torch.isfinite(tail)
        if require_uniform_leading_batch:
            leading_certified = certified.reshape(certified.shape[0], -1).all(dim=-1)
            if bool(leading_certified.all()):
                return psi_right, psi_left, order
            # Partial certification still falls back, DELIBERATELY (re-affirmed audit 2026-08-07).
            # The caller catches this to drop out of the batched fast path into the per-block loop,
            # which is what makes the batched result bit-identical to the per-block one -- the
            # contract pinned by
            # test_pullback_group_equal_blocks_with_mixed_series_orders_fall_back_exactly.
            # Continuing to the first UNIFORMLY certified order instead would keep the fast path but
            # evaluate the low-order block at a higher order than the per-block route would, which
            # is equally certified and NOT bit-identical -- so it was rejected.
            #
            # Lowering the floor to 8 does not make this fire more often at the live distribution:
            # both leading blocks draw phi from the same init, so they certify together. Measured at
            # K=20/H=2 over 108 trials (scale 0.06/0.0632/0.2 x 64/256/1024 active rows): ZERO
            # fallbacks under both the old (40, step 8) and new (8, step 4) grids. The mixed state
            # needs blocks with very different ||phi||, which is exactly what that test constructs.
            if bool(leading_certified.any()):
                raise _MixedPhiDifferentialOrder(
                    "strict phi group blocks require different adaptive series orders"
                )
        elif bool(certified.all()):
            return psi_right, psi_left, order
        if fixed_order is not None:
            break
    raise ValueError(
        f"strict phi group differential series certificate failed by order "
        f"{_PHI_GROUP_MAX_SERIES_ORDER if fixed_order is None else fixed_order}"
    )


def set_phi_group_differential_mode(mode: str) -> str:
    r"""Set the process-wide Psi evaluation route; returns the previous value."""
    global _PHI_GROUP_DIFFERENTIAL_MODE
    if mode not in _PHI_GROUP_DIFFERENTIAL_MODES:
        raise ValueError(
            f"phi group differential mode must be one of {_PHI_GROUP_DIFFERENTIAL_MODES}, "
            f"got {mode!r}"
        )
    previous = _PHI_GROUP_DIFFERENTIAL_MODE
    _PHI_GROUP_DIFFERENTIAL_MODE = mode
    return previous


def phi_group_differential_mode() -> str:
    r"""Return the active Psi evaluation route."""
    return _PHI_GROUP_DIFFERENTIAL_MODE


def _sinhc(u: torch.Tensor) -> torch.Tensor:
    r"""``sinh(u)/u`` continued to ``u = 0``, for complex ``u``.

    This is what makes the closed form need no branch at a repeated eigenvalue. The naive quotient
    ``(exp(z) - 1)/z`` is 0/0 exactly where ``lam_i == lam_j``, which is not a rare input -- it is
    the ENTIRE diagonal ``i == j`` of every single evaluation. Rewriting through ``sinh`` moves the
    singularity into a function that is entire, so the limit is a value rather than a special case.
    The small-argument series is for floating point, not for the mathematics: near zero the
    subtraction inside ``sinh`` cancels, so the quotient loses precision long before it loses
    meaning.
    """
    small = u.abs() < 1e-6
    u_safe = torch.where(small, torch.ones_like(u), u)
    series = 1.0 + u * u / 6.0 + (u ** 4) / 120.0
    return torch.where(small, series, torch.sinh(u_safe) / u_safe)


def _spectral_phi_differentials(
    phi:         torch.Tensor,            # (..., n_gen) float64 chart coordinates
    preparation: "_StrictBasisPreparation",
) -> "tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]] | None":
    r"""Exact ``Psi_R(ad_X)`` and ``Psi_L(ad_X)`` in generator coordinates, or None to fall back.

    ``ad_X`` acting on the algebra has eigenvectors ``P E_ij P^-1`` with eigenvalues
    ``lam_i - lam_j``, where ``X = P diag(lam) P^-1``. So for any analytic ``f``,

        f(ad_X)(T) = P [ f(lam_i - lam_j) . (P^-1 T P) ] P^-1              (. = entrywise)

    -- the Daleckii-Krein divided-difference form (Higham, *Functions of Matrices*, 2008, §3.2).
    With ``z = lam_i - lam_j`` and ``u = z/2``:

        Psi_R(z) = (e^z - 1)/z = e^u sinhc(u),   Psi_L(z) = (1 - e^-z)/z = e^-u sinhc(u)

    which is ``Psi_L(z) = Psi_R(-z)``, matching the alternating sign the series branch carries.
    NOTE this is Psi ALONE, not the full differential: the caller multiplies by ``exp(phi)``
    separately, so the ``e^{(lam_i + lam_j)/2}`` factor of the combined ``D exp`` does not belong
    here.

    Returned in the SAME convention the series branch produces -- column ``a`` holds the
    coordinates of ``Psi(ad_X)(G_a)`` -- so the caller is unchanged. Coordinates are recovered
    through the Gram matrix, so a non-orthonormal generator basis is handled rather than assumed.

    Returns ``None`` when the eigendecomposition fails, its conditioning exhausts
    the roundoff budget, or the imaginary residue is too large. The roundoff
    screen uses ``gamma_(8K) kappa_F(P)^2`` for four complex matrix products;
    it is a conservative accuracy screen, not a complete forward-error bound.
    """
    basis = preparation.basis                                        # (n_gen, K, K) float64
    X = torch.einsum("...a,aij->...ij", phi, basis)                  # (..., K, K)
    try:
        lam, P = torch.linalg.eig(X)                                 # complex (..., K), (..., K, K)
        P_inv = torch.linalg.inv(P)
    except torch.linalg.LinAlgError:                                  # defective X: no eigenbasis
        return None
    cond = (torch.linalg.matrix_norm(P, dim=(-2, -1))
            * torch.linalg.matrix_norm(P_inv, dim=(-2, -1))).abs()
    unit_roundoff = torch.finfo(phi.dtype).eps / 2.0
    operations = 8 * X.shape[-1]
    gamma = operations * unit_roundoff / (1.0 - operations * unit_roundoff)
    roundoff = gamma * cond.square()
    eig_residual = torch.linalg.matrix_norm(
        X.to(P.dtype) @ P - P * lam.unsqueeze(-2), dim=(-2, -1),
    )
    eig_scale = (
        torch.linalg.matrix_norm(X, dim=(-2, -1)).clamp_min(1.0)
        * torch.linalg.matrix_norm(P, dim=(-2, -1))
    )
    eig_error = cond.square() * eig_residual / eig_scale
    spectral_ok = (
        torch.isfinite(lam).all()
        & torch.isfinite(P).all()
        & torch.isfinite(P_inv).all()
        & (cond <= _PHI_GROUP_SPECTRAL_MAX_COND).all()
        & (roundoff + eig_error <= _PHI_GROUP_SPECTRAL_ROUNDOFF_BUDGET).all()
    )
    if not bool(spectral_ok):
        return None

    u = 0.5 * (lam.unsqueeze(-1) - lam.unsqueeze(-2))                # (..., K, K), u_ij
    sinhc = _sinhc(u)
    psi_r_entries = torch.exp(u) * sinhc                             # Psi_R(lam_i - lam_j)
    psi_l_entries = torch.exp(-u) * sinhc                            # Psi_L(lam_i - lam_j)

    basis_c = basis.to(P.dtype)

    def _conjugate(entries: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""``P [entries . (P^-1 target P)] P^-1`` over a trailing (..., m, K, K) stack."""
        inner = torch.einsum("...ij,...ajk,...kl->...ail", P_inv, target, P)
        return torch.einsum(
            "...ij,...ajk,...kl->...ail", P, entries.unsqueeze(-3) * inner, P_inv,
        )

    def _real(value: torch.Tensor) -> "torch.Tensor | None":
        scale = value.real.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        if not bool((torch.isfinite(value) &
                     (value.imag.abs() <= _PHI_GROUP_SPECTRAL_MAX_IMAG * scale)).all()):
            return None                                              # eig failed; fail closed
        return value.real.contiguous()

    # RIGHT: return the pushed MATRICES, not coordinates. The caller's next line is
    # ``pushed = einsum("...ca,cij->...aij", psi_right, basis)`` -- it expands the coordinate matrix
    # straight back into matrices, so projecting onto the basis here only to have it undone is the
    # single most expensive step in the routine (n_gen^2 K^2 per row, in complex arithmetic) and it
    # is pure round trip. Handing back the matrices skips both halves.
    pushed = _real(_conjugate(psi_r_entries, basis_c.expand(*P.shape[:-2], *basis_c.shape)))
    if pushed is None:
        return None

    def _apply_left(v_phi: torch.Tensor) -> torch.Tensor:
        r"""``Psi_L(ad_X) v_phi`` in coordinates, WITHOUT materializing Psi_L.

        The series branch builds the full (n_gen, n_gen) operator and then uses it exactly once, on
        one vector. Embedding that vector, conjugating it, and projecting the single result back is
        O(K^3 + n_gen K^2) against O(n_gen^2 K^2) for build-then-apply.
        """
        embedded = torch.einsum("...b,bij->...ij", v_phi.to(P.dtype), basis_c)
        conjugated = _conjugate(psi_l_entries, embedded.unsqueeze(-3)).squeeze(-3)
        proj = torch.einsum("bij,...ij->...b", basis_c, conjugated)
        coeff = (proj if preparation.gram_is_identity
                 else torch.cholesky_solve(
                     proj.unsqueeze(-1), preparation.gram_factor.to(proj.dtype)).squeeze(-1))
        return coeff.real

    return pushed, _apply_left


def _build_strict_basis_preparation(
    basis: torch.Tensor,                  # (n_gen, K, K) float64 target-device basis
) -> _StrictBasisPreparation:
    """Build the immutable tensors shared by strict directions for one basis."""
    structure, gram = _strict_structure_constants(basis)
    gram_factor, gram_info = torch.linalg.cholesky_ex(gram)
    if bool((gram_info != 0).any()):
        raise ValueError("strict phi group direction generator Gram factorization failed")
    identity = torch.eye(gram.shape[-1], dtype=gram.dtype, device=gram.device)
    gram_inverse_half = torch.linalg.solve_triangular(
        gram_factor,
        identity,
        upper=False,
    )
    return _StrictBasisPreparation(
        basis=basis,
        structure=structure,
        gram=gram,
        gram_factor=gram_factor,
        gram_inverse_half=gram_inverse_half,
        gram_is_identity=bool(torch.equal(gram, identity)),
    )


def _strict_preparation_cache_key(
    generators:       torch.Tensor,       # source basis
    irrep_dims:       Optional[List[int]],
    coordinate_layout: Optional[str],

    *,
    target_device: torch.device,
    target_dtype:  torch.dtype,
) -> tuple:
    """Return the complete identity/version key for strict basis preparation."""
    return (
        id(generators),
        int(generators._version),
        tuple(generators.shape),
        generators.device,
        generators.dtype,
        target_device,
        target_dtype,
        None if irrep_dims is None else tuple(irrep_dims),
        coordinate_layout,
    )


def _build_strict_block_preparations(
    generators:        torch.Tensor,      # source basis
    irrep_dims:        Optional[List[int]],
    coordinate_layout: Optional[str],

    *,
    target_device: torch.device,
    target_dtype:  torch.dtype,
    detach:        bool,
) -> tuple[_StrictBlockPreparation, ...]:
    """Build full or direct-sum local basis preparations on the staging device."""
    source = generators.detach() if detach else generators
    basis = source.to(device=target_device, dtype=target_dtype)
    if detach:
        # A cache value must not strongly retain its weakly keyed source, including when
        # source and target device/dtype already match and ``Tensor.to`` returns itself.
        basis = basis.clone()
    if irrep_dims is None or len(irrep_dims) == 1:
        index = torch.arange(basis.shape[0], dtype=torch.long, device=target_device)
        return (
            _StrictBlockPreparation(
                index,
                _build_strict_basis_preparation(basis),
                True,
            ),
        )
    if any(d <= 0 for d in irrep_dims) or sum(irrep_dims) != basis.shape[-1]:
        raise ValueError("pullback_per_block requires positive irrep_dims summing to K")

    if coordinate_layout == "block_head_row_major":
        expected = sum(dimension * dimension for dimension in irrep_dims)
        if basis.shape[0] != expected:
            raise ValueError(
                "block_head_row_major coordinates require one full row-major GL basis "
                f"per block; got {basis.shape[0]} generators, expected {expected}"
            )
        indices = []
        coordinate_start = 0
        for dimension in irrep_dims:
            coordinate_stop = coordinate_start + dimension * dimension
            indices.append(
                torch.arange(
                    coordinate_start,
                    coordinate_stop,
                    dtype=torch.long,
                    device=target_device,
                )
            )
            coordinate_start = coordinate_stop
    else:
        block_of = _generator_block_index(basis, irrep_dims)
        indices = [
            (block_of == block).nonzero(as_tuple=True)[0]
            for block in range(len(irrep_dims))
        ]

    preparations = []
    first_local_basis = None
    matrix_start = 0
    for block, dimension in enumerate(irrep_dims):
        index = indices[block]
        if index.numel() == 0:
            raise ValueError(f"pullback_per_block block {block} has no generators")
        local_basis = basis[index][
            :,
            matrix_start:matrix_start + dimension,
            matrix_start:matrix_start + dimension,
        ].contiguous()
        matches_first_basis = (
            first_local_basis is None
            or torch.equal(local_basis, first_local_basis)
        )
        if first_local_basis is None:
            first_local_basis = local_basis
        preparations.append(
            _StrictBlockPreparation(
                index,
                _build_strict_basis_preparation(local_basis),
                matches_first_basis,
            )
        )
        matrix_start += dimension
    return tuple(preparations)


def _strict_block_preparations(
    generators:        torch.Tensor,      # source basis
    irrep_dims:        Optional[List[int]],
    coordinate_layout: Optional[str],

    *,
    target_device: torch.device,
    target_dtype:  torch.dtype,
) -> tuple[_StrictBlockPreparation, ...]:
    """Return weakly cached immutable strict-basis preparation, or bypass for autograd."""
    if generators.requires_grad:
        return _build_strict_block_preparations(
            generators,
            irrep_dims,
            coordinate_layout,
            target_device=target_device,
            target_dtype=target_dtype,
            detach=False,
        )
    key = _strict_preparation_cache_key(
        generators,
        irrep_dims,
        coordinate_layout,
        target_device=target_device,
        target_dtype=target_dtype,
    )
    cached = _STRICT_PULLBACK_PREP_CACHE.get(key)
    if cached is not None:
        generators_ref, preparations = cached
        if generators_ref() is generators:
            _STRICT_PULLBACK_PREP_CACHE.move_to_end(key)
            return preparations
        _STRICT_PULLBACK_PREP_CACHE.pop(key, None)

    preparations = _build_strict_block_preparations(
        generators,
        irrep_dims,
        coordinate_layout,
        target_device=target_device,
        target_dtype=target_dtype,
        detach=True,
    )

    def _remove_dead_entry(
        generators_ref: weakref.ReferenceType[torch.Tensor],
    ) -> None:
        entry = _STRICT_PULLBACK_PREP_CACHE.get(key)
        if entry is not None and entry[0] is generators_ref:
            _STRICT_PULLBACK_PREP_CACHE.pop(key, None)

    generators_ref = weakref.ref(generators, _remove_dead_entry)
    _STRICT_PULLBACK_PREP_CACHE[key] = (generators_ref, preparations)
    _STRICT_PULLBACK_PREP_CACHE.move_to_end(key)
    while len(_STRICT_PULLBACK_PREP_CACHE) > _STRICT_PULLBACK_PREP_CACHE_MAXSIZE:
        _STRICT_PULLBACK_PREP_CACHE.popitem(last=False)
    return preparations


def _phi_group_rows_per_chunk(n_gen: int) -> int:
    """Bound twelve dense float64 coordinate workspaces to the staging budget."""
    return max(1, _PHI_GROUP_WORKSPACE_BYTES // (_PHI_GROUP_DENSE_WORKSPACES * 8 * n_gen * n_gen))


def _chunked_pullback_group_direction(
    grad_phi:          torch.Tensor,
    phi:               torch.Tensor,
    preparation:       _StrictBasisPreparation,

    *,
    differential_mode:  str,
    rows_per_chunk:     int,
    fixed_series_order: Optional[int] = None,
    reject_ill_conditioned_rows: bool = False,
) -> PullbackGroupDirectionResult:
    r"""Bound dense workspaces while preserving the local batch series order.

    Each chunk first finds its certified order; only chunks below the batch
    maximum are evaluated again. If any spectral chunk declines, all chunks
    use the series, as they would in one unsplit spectral call. Only the small
    result tensors survive each chunk in optimizer no-grad staging. Calls that
    request autograd still retain the graphs needed by their returned tensors.
    """
    leading_shape = phi.shape[:-1]
    n_gen = phi.shape[-1]
    phi_chunks = phi.reshape(-1, n_gen).split(rows_per_chunk)
    grad_chunks = grad_phi.reshape(-1, n_gen).split(rows_per_chunk)

    def evaluate(index: int, mode: str, order: Optional[int]) -> PullbackGroupDirectionResult:
        return _full_pullback_group_direction(
            grad_chunks[index], phi_chunks[index], preparation.basis,
            differential_mode=mode, fixed_series_order=order, preparation=preparation,
            reject_ill_conditioned_rows=reject_ill_conditioned_rows,
        )

    results = [evaluate(i, differential_mode, fixed_series_order) for i in range(len(phi_chunks))]
    if differential_mode == "spectral" and any(result.series_order > 0 for result in results):
        results = [evaluate(i, "series", None) if result.series_order == 0 else result
                   for i, result in enumerate(results)]
    order = max(result.series_order for result in results)
    if order > 0:
        results = [evaluate(i, "series", order) if result.series_order != order else result
                   for i, result in enumerate(results)]

    def combine(name: str, trailing_shape: tuple[int, ...]) -> torch.Tensor:
        return torch.cat([getattr(result, name) for result in results], dim=0).reshape(
            *leading_shape, *trailing_shape,
        )

    dimension = preparation.basis.shape[-1]
    return PullbackGroupDirectionResult(
        v_phi=combine("v_phi", (n_gen,)),
        xi=combine("xi", (n_gen,)),
        min_undamped_generalized_eigenvalue=combine("min_undamped_generalized_eigenvalue", ()),
        undamped_generalized_condition=combine("undamped_generalized_condition", ()),
        damped_generalized_condition=combine("damped_generalized_condition", ()),
        scaled_solve_residual=combine("scaled_solve_residual", ()),
        series_order=order,
        current_element=combine("current_element", (dimension, dimension)),
        accepted_rows=(combine("accepted_rows", ()) if reject_ill_conditioned_rows else None),
    )


def _full_pullback_group_direction(
    grad_phi:   torch.Tensor,             # (..., n_gen)
    phi:        torch.Tensor,             # (..., n_gen)
    generators: torch.Tensor,             # (n_gen, K, K)

    *,
    require_uniform_series_order: bool                             = False,
    differential_mode:            Optional[str]                    = None,
    fixed_series_order:           Optional[int]                    = None,
    preparation:                  Optional[_StrictBasisPreparation] = None,
    reject_ill_conditioned_rows:  bool                             = False,
) -> PullbackGroupDirectionResult:
    r"""Compute one certified full-block pullback direction.

    With ``J_a = D exp_X[G_a]`` and ``M_ab = <J_a, J_b>_F``, this solves
    ``A v_phi = grad_phi`` for ``A = sym(M) + 1e-6 B`` by Cholesky. The returned
    group velocity is ``xi = Psi_L(ad_phi) v_phi``. Generalized eigenvalues are
    those of ``B^{-1/2} sym(M) B^{-1/2}`` and are certified before damping.
    """
    phi_64 = phi.to(dtype=torch.float64)
    grad_64 = grad_phi.to(dtype=torch.float64)
    if preparation is None:
        preparation = _build_strict_basis_preparation(
            generators.to(device=phi.device, dtype=torch.float64)
        )
    evaluation_mode = (_PHI_GROUP_DIFFERENTIAL_MODE
                       if differential_mode is None else differential_mode)
    if evaluation_mode not in _PHI_GROUP_DIFFERENTIAL_MODES:
        raise ValueError(f"invalid phi group differential mode {evaluation_mode!r}")
    rows_per_chunk = _phi_group_rows_per_chunk(phi_64.shape[-1])
    if phi_64.shape == grad_64.shape and phi_64.numel() // phi_64.shape[-1] > rows_per_chunk:
        if require_uniform_series_order:
            raise _MixedPhiDifferentialOrder("large strict blocks require independent chunk staging")
        return _chunked_pullback_group_direction(
            grad_64, phi_64, preparation,
            differential_mode=evaluation_mode,
            fixed_series_order=fixed_series_order,
            rows_per_chunk=rows_per_chunk,
            reject_ill_conditioned_rows=reject_ill_conditioned_rows,
        )
    basis_64 = preparation.basis
    structure = preparation.structure
    gram = preparation.gram
    # Psi_R / Psi_L, either exactly (spectral) or by certified truncation (series). Both return the
    # SAME coordinate convention -- column a holds Psi(ad_X)(G_a) -- so everything below, including
    # every certificate, the ridge and the Cholesky, is untouched by the choice (audit 2026-08-08).
    # series_order = 0 marks the closed form: there is no truncation, so there is no order.
    pushed = apply_left = None
    series_order = 0
    if evaluation_mode == "spectral" and fixed_series_order is None:
        spectral = _spectral_phi_differentials(phi_64, preparation)
        if spectral is not None:
            pushed, apply_left = spectral
    if pushed is None:
        # Also the fallback when the spectral route declines (defective or ill-conditioned P),
        # which is why that helper returns None rather than raising.
        ad = torch.einsum("...a,abc->...cb", phi_64, structure)
        series_kwargs = {"require_uniform_leading_batch": require_uniform_series_order}
        if fixed_series_order is not None:
            series_kwargs["fixed_order"] = fixed_series_order
        psi_right, psi_left, series_order = _adaptive_phi_differentials(ad, **series_kwargs)
        pushed = torch.einsum("...ca,cij->...aij", psi_right, basis_64)
        apply_left = lambda v: torch.einsum("...ab,...b->...a", psi_left, v)   # noqa: E731
    exp_phi = torch.linalg.matrix_exp(torch.einsum("...a,aij->...ij", phi_64, basis_64))
    differential = torch.einsum("...aij,...jk->...aik", pushed, exp_phi)
    metric = torch.einsum("...aij,...bij->...ab", differential, differential)
    metric = 0.5 * (metric + metric.transpose(-1, -2))
    gram_inverse_half = preparation.gram_inverse_half
    whitened = (metric if preparation.gram_is_identity else
                torch.matmul(torch.matmul(gram_inverse_half, metric),
                             gram_inverse_half.transpose(-1, -2)))
    undamped_eigenvalues = safe_eigvalsh(0.5 * (whitened + whitened.transpose(-1, -2)))
    min_undamped = undamped_eigenvalues[..., 0]
    max_undamped = undamped_eigenvalues[..., -1]
    undamped_condition = max_undamped / min_undamped
    damped = metric + _PHI_GROUP_GRAM_RIDGE * gram
    damped_eigenvalues = undamped_eigenvalues + _PHI_GROUP_GRAM_RIDGE
    damped_condition = damped_eigenvalues[..., -1] / damped_eigenvalues[..., 0]
    accepted = None
    if reject_ill_conditioned_rows:
        # Per-row conditioning verdict (audit 2026-09-23 #10). Only a nonfinite spectrum raises.
        if not bool(torch.isfinite(undamped_eigenvalues).all()):
            raise ValueError("strict phi group direction produced a nonfinite metric spectrum")
        accepted = ((min_undamped >= _PHI_GROUP_MIN_UNDAMPED_EIG)
                    & torch.isfinite(damped_condition)
                    & (damped_condition <= _PHI_GROUP_MAX_DAMPED_COND))
    else:
        if not bool(torch.isfinite(undamped_eigenvalues).all()
                    & (min_undamped >= _PHI_GROUP_MIN_UNDAMPED_EIG).all()):
            minimum = float(min_undamped.min())
            raise ValueError(
                "strict phi group direction undamped generalized eigenvalue "
                f"{minimum:.3e} is below {_PHI_GROUP_MIN_UNDAMPED_EIG:.1e}"
            )
        if (
            not bool((torch.isfinite(damped_condition)
                      & (damped_condition <= _PHI_GROUP_MAX_DAMPED_COND)).all())
        ):
            maximum = float(damped_condition.max())
            raise ValueError(
                "strict phi group direction damped generalized condition "
                f"{maximum:.3e} exceeds {_PHI_GROUP_MAX_DAMPED_COND:.1e}"
            )
    factor, info = torch.linalg.cholesky_ex(damped)
    cholesky_failed = info != 0
    if accepted is not None:
        cholesky_failed = cholesky_failed & accepted
    if bool(cholesky_failed.any()):
        raise ValueError("strict phi group direction damped Cholesky factorization failed")
    v_phi = torch.cholesky_solve(grad_64.unsqueeze(-1), factor).squeeze(-1)
    if accepted is not None:
        v_phi = torch.where(accepted.unsqueeze(-1), v_phi, torch.zeros_like(v_phi))
    residual = torch.linalg.vector_norm(
        torch.einsum("...ab,...b->...a", damped, v_phi) - grad_64,
        dim=-1,
    )
    damped_operator_norm = (
        damped_eigenvalues[..., -1]
        if preparation.gram_is_identity
        else torch.linalg.matrix_norm(damped, ord=2, dim=(-2, -1))
    )
    scale = (
        damped_operator_norm
        * torch.linalg.vector_norm(v_phi, dim=-1)
        + torch.linalg.vector_norm(grad_64, dim=-1)
    )
    scaled_residual = residual / scale.clamp_min(torch.finfo(torch.float64).tiny)
    residual_bad = ~(torch.isfinite(scaled_residual)
                     & (scaled_residual <= _PHI_GROUP_SOLVE_RESIDUAL))
    if accepted is not None:
        residual_bad = residual_bad & accepted
    if bool(residual_bad.any()):
        maximum = float(scaled_residual.max())
        raise ValueError(
            "strict phi group direction scaled solve residual "
            f"{maximum:.3e} exceeds {_PHI_GROUP_SOLVE_RESIDUAL:.1e}"
        )
    xi = apply_left(v_phi)
    condition_finite = torch.isfinite(undamped_condition)
    if accepted is not None:
        xi = torch.where(accepted.unsqueeze(-1), xi, torch.zeros_like(xi))
        condition_finite = condition_finite | ~accepted
    if not bool(torch.isfinite(v_phi).all() & torch.isfinite(xi).all()
                & condition_finite.all()):
        raise ValueError("strict phi group direction produced a nonfinite certificate")
    return PullbackGroupDirectionResult(
        v_phi=v_phi,
        xi=xi,
        min_undamped_generalized_eigenvalue=min_undamped,
        undamped_generalized_condition=undamped_condition,
        damped_generalized_condition=damped_condition,
        scaled_solve_residual=scaled_residual,
        series_order=series_order,
        current_element=exp_phi,
        accepted_rows=accepted,
    )


@register_phi_group_direction("pullback_per_block", preserve_source_basis=True)
def _pullback_group_direction_per_block(
    grad_phi:   torch.Tensor,             # (..., n_gen)
    phi:        torch.Tensor,             # (..., n_gen)
    generators: torch.Tensor,             # (n_gen, K, K)

    *,
    coordinate_layout: Optional[str]       = None,
    differential_mode: Optional[str]       = None,
    irrep_dims:        Optional[List[int]] = None,
    reject_ill_conditioned_rows: bool      = False,
) -> PullbackGroupDirectionResult:
    r"""Solve each direct-sum block independently and aggregate global extrema.

    For ``g = direct_sum_h g_h``, each local system is
    ``A_h v_h = grad_h``. The global generalized condition is
    ``max_h(lambda_max,h) / min_h(lambda_min,h)`` rather than the largest local
    condition, while the solve certificate is the maximum local residual.
    Under ``reject_ill_conditioned_rows`` the row verdict uses these GLOBAL extrema; a row
    rejected by any local block is rejected globally, because the global floor and condition
    bound every local one.
    """
    if irrep_dims is None:
        raise ValueError("pullback_per_block requires irrep_dims")
    preparations = _strict_block_preparations(
        generators,
        irrep_dims,
        coordinate_layout,
        target_device=phi.device,
        target_dtype=torch.float64,
    )
    if len(irrep_dims) == 1:
        return _full_pullback_group_direction(
            grad_phi,
            phi,
            generators,
            preparation=preparations[0].basis,
            differential_mode=differential_mode,
            reject_ill_conditioned_rows=reject_ill_conditioned_rows,
        )
    grad_64 = grad_phi.to(dtype=torch.float64)
    phi_64 = phi.to(dtype=torch.float64)
    v_phi = torch.zeros_like(grad_64)
    xi = torch.zeros_like(grad_64)
    equal_blocks_compatible = (
        coordinate_layout == "block_head_row_major"
        and grad_64.ndim > 1
        and grad_64.shape[:-1] == phi_64.shape[:-1]
        and not generators.requires_grad
        and len(preparations) > 1
        and len(set(irrep_dims)) == 1
        and all(preparation.matches_first_basis for preparation in preparations)
    )
    batch_equal_blocks = (
        equal_blocks_compatible
        and phi_64.numel() // preparations[0].basis.basis.shape[0]
            <= _phi_group_rows_per_chunk(preparations[0].basis.basis.shape[0])
    )
    if batch_equal_blocks:
        batched_grad = torch.stack(
            [grad_64[..., preparation.index] for preparation in preparations],
            dim=0,
        )
        batched_phi = torch.stack(
            [phi_64[..., preparation.index] for preparation in preparations],
            dim=0,
        )
        try:
            local = _full_pullback_group_direction(
                batched_grad,
                batched_phi,
                preparations[0].basis.basis,
                require_uniform_series_order=True,
                preparation=preparations[0].basis,
                differential_mode=differential_mode,
                reject_ill_conditioned_rows=reject_ill_conditioned_rows,
            )
        except ValueError:
            batch_equal_blocks = False
    if batch_equal_blocks:
        for block, preparation in enumerate(preparations):
            v_phi = v_phi.index_copy(-1, preparation.index, local.v_phi[block])
            xi = xi.index_copy(-1, preparation.index, local.xi[block])
        local_min_undamped = local.min_undamped_generalized_eigenvalue
        local_max_undamped = (
            local.min_undamped_generalized_eigenvalue
            * local.undamped_generalized_condition
        )
        local_scaled_residual = local.scaled_solve_residual
        series_order = local.series_order
        current_element = local.current_element.movedim(0, -3)
    else:
        local_results = []
        for preparation in preparations:
            index = preparation.index
            local = _full_pullback_group_direction(
                grad_64[..., index],
                phi_64[..., index],
                preparation.basis.basis,
                preparation=preparation.basis,
                differential_mode=differential_mode,
                reject_ill_conditioned_rows=reject_ill_conditioned_rows,
            )
            local_results.append(local)
        evaluation_mode = (_PHI_GROUP_DIFFERENTIAL_MODE
                           if differential_mode is None else differential_mode)
        if (equal_blocks_compatible and evaluation_mode == "spectral"
                and any(result.series_order > 0 for result in local_results)
                and any(result.series_order == 0 for result in local_results)):
            # An unsplit equal-head call declines spectral evaluation jointly.
            # It keeps the all-series result only when every head certifies at
            # the same order; mixed orders/failures use independent head routes.
            try:
                series_results = [
                    result if result.series_order > 0 else _full_pullback_group_direction(
                        grad_64[..., preparation.index], phi_64[..., preparation.index],
                        preparation.basis.basis, preparation=preparation.basis,
                        differential_mode="series",
                        reject_ill_conditioned_rows=reject_ill_conditioned_rows,
                    )
                    for preparation, result in zip(preparations, local_results)
                ]
            except ValueError:
                series_results = []
            if series_results and len({result.series_order for result in series_results}) == 1:
                local_results = series_results
        for preparation, local in zip(preparations, local_results):
            v_phi = v_phi.index_copy(-1, preparation.index, local.v_phi)
            xi = xi.index_copy(-1, preparation.index, local.xi)
        local_min_undamped = torch.stack(
            [result.min_undamped_generalized_eigenvalue for result in local_results],
            dim=0,
        )
        local_max_undamped = torch.stack(
            [
                result.min_undamped_generalized_eigenvalue
                * result.undamped_generalized_condition
                for result in local_results
            ],
            dim=0,
        )
        local_scaled_residual = torch.stack(
            [result.scaled_solve_residual for result in local_results],
            dim=0,
        )
        series_order = max(result.series_order for result in local_results)
        current_element = (
            torch.stack([result.current_element for result in local_results], dim=-3)
            if len(set(irrep_dims)) == 1 else None
        )
    min_undamped = local_min_undamped.amin(dim=0)
    max_undamped = local_max_undamped.amax(dim=0)
    undamped_condition = max_undamped / min_undamped
    local_min_damped = local_min_undamped + _PHI_GROUP_GRAM_RIDGE
    local_max_damped = local_max_undamped + _PHI_GROUP_GRAM_RIDGE
    damped_condition = local_max_damped.amax(dim=0) / local_min_damped.amin(dim=0)
    accepted = None
    if reject_ill_conditioned_rows:
        if not bool(torch.isfinite(min_undamped).all() & torch.isfinite(max_undamped).all()):
            raise ValueError("strict phi group direction produced a nonfinite metric spectrum")
        accepted = ((min_undamped >= _PHI_GROUP_MIN_UNDAMPED_EIG)
                    & torch.isfinite(damped_condition)
                    & (damped_condition <= _PHI_GROUP_MAX_DAMPED_COND))
        v_phi = torch.where(accepted.unsqueeze(-1), v_phi, torch.zeros_like(v_phi))
        xi = torch.where(accepted.unsqueeze(-1), xi, torch.zeros_like(xi))
    else:
        if (
            not bool((torch.isfinite(min_undamped)
                      & (min_undamped >= _PHI_GROUP_MIN_UNDAMPED_EIG)).all())
        ):
            minimum = float(min_undamped.min())
            raise ValueError(
                "strict phi group direction undamped generalized eigenvalue "
                f"{minimum:.3e} is below {_PHI_GROUP_MIN_UNDAMPED_EIG:.1e}"
            )
        if (
            not bool((torch.isfinite(damped_condition)
                      & (damped_condition <= _PHI_GROUP_MAX_DAMPED_COND)).all())
        ):
            maximum = float(damped_condition.max())
            raise ValueError(
                "strict phi group direction damped generalized condition "
                f"{maximum:.3e} exceeds {_PHI_GROUP_MAX_DAMPED_COND:.1e}"
            )

    scaled_residual = local_scaled_residual.amax(dim=0)
    residual_bad = ~(torch.isfinite(scaled_residual)
                     & (scaled_residual <= _PHI_GROUP_SOLVE_RESIDUAL))
    if accepted is not None:
        residual_bad = residual_bad & accepted
    if bool(residual_bad.any()):
        maximum = float(scaled_residual.max())
        raise ValueError(
            "strict phi group direction scaled solve residual "
            f"{maximum:.3e} exceeds {_PHI_GROUP_SOLVE_RESIDUAL:.1e}"
        )

    condition_finite = torch.isfinite(undamped_condition)
    if accepted is not None:
        condition_finite = condition_finite | ~accepted
    if not bool(torch.isfinite(v_phi).all() & torch.isfinite(xi).all()
                & condition_finite.all()):
        raise ValueError("strict phi group direction produced a nonfinite certificate")
    return PullbackGroupDirectionResult(
        v_phi=v_phi,
        xi=xi,
        min_undamped_generalized_eigenvalue=min_undamped,
        undamped_generalized_condition=undamped_condition,
        damped_generalized_condition=damped_condition,
        scaled_solve_residual=scaled_residual,
        series_order=series_order,
        current_element=current_element,
        accepted_rows=accepted,
    )


def _generator_block_index(
    generators: torch.Tensor,             # (n_gen, K, K)
    irrep_dims: List[int],                # block sizes; sum == K

    *,
    tol:        float = 1e-9,
) -> torch.Tensor:                        # (n_gen,) block id per generator
    r"""Block membership of each generator (asserts single-block support)."""
    bounds, start = [], 0
    for d in irrep_dims:
        bounds.append((start, start + d))
        start += d
    n_gen = generators.shape[0]
    block_of = torch.full((n_gen,), -1, dtype=torch.long, device=generators.device)
    for a in range(n_gen):
        mass  = [float(generators[a, s:e, s:e].abs().sum()) for (s, e) in bounds]
        total = float(generators[a].abs().sum())
        hits  = [h for h, m in enumerate(mass) if m > tol]
        if len(hits) != 1 or abs(sum(mass) - total) > tol:
            raise ValueError(f"generator {a} is not supported in a single irrep block")
        block_of[a] = hits[0]
    return block_of


# --- U-space helpers for the group_log retraction (derivation 2026-09-28, Track 1) -----------------
#
# Matrix-level and pure: every argument is a stack of real d x d blocks (..., d, d). For gl and block
# gl with the built-in row-major elementary basis (Gram B = I), a chart row reshapes exactly to its
# blocks X, and the processed phi gradient reshapes to the chart covector block M. Each helper needs
# spec(X) inside the principal strip |Im z| < pi; there the principal-log Frechet derivative inverts
# the exponential differential. Off the strip it inverts the derivative at the wrong logarithm, so
# every helper returns an ``ok`` mask (strip certificate, no log failure, finite) and zeroes the
# blocks it cannot certify. The caller supplies its own off-strip route (the optimizer uses the
# series direction ``pullback_group_direction``).


def uspace_ambient_gradient(
    X: torch.Tensor,                                     # (..., d, d) chart block
    M: torch.Tensor,                                     # (..., d, d) chart covector block

    *,
    U:         Optional[torch.Tensor] = None,            # (..., d, d) exp(X), if already formed
    certified: Optional[torch.Tensor] = None,            # (...) strip certificate, if already formed
) -> tuple[torch.Tensor, torch.Tensor]:                  # (..., d, d) G_bar, (...) ok
    r"""Recover the ambient covector :math:`\bar G = \nabla_U L` from the chart covector.

    With :math:`U = e^X` and :math:`(\nabla_\varphi \ell)_a = \langle G_a,
    \Psi_L(\mathrm{ad}_{X^\top})(U^\top \bar G) \rangle`, the chart covector block is
    :math:`M = \Psi_L(\mathrm{ad}_{X^\top})(U^\top \bar G)`. On the principal strip
    :math:`\Psi_L(\mathrm{ad}_A)^{-1}(M) = e^A L_{\log}(e^A, M)`, hence

    .. math::
        \bar G = L_{\log}(U^\top, M)
        = \Bigl[\mathrm{Log}\begin{pmatrix} U^\top & M \ 0 & U^\top \end{pmatrix}\Bigr]_{12}.

    For block gl this is the block-diagonal part of the ambient gradient, which is all the
    Frobenius pullback reads. The map is linear in :math:`M`, so it commutes with gradient
    accumulation, unscaling and clipping.
    """
    if U is None:
        U = torch.linalg.matrix_exp(X)
    if certified is None:
        certified, _ = principal_strip_certificate(X)
    _, G_bar, failed = logm_frechet_pair(U.transpose(-1, -2), M)
    ok = certified & ~failed & torch.isfinite(G_bar).all(dim=-1).all(dim=-1)
    return torch.where(ok[..., None, None], G_bar, torch.zeros_like(G_bar)), ok


def uspace_spatial_covector(
    X: torch.Tensor,                                     # (..., d, d) chart block
    M: torch.Tensor,                                     # (..., d, d) chart covector block

    *,
    U:         Optional[torch.Tensor] = None,
    certified: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:                  # (..., d, d) G_W, (...) ok
    r"""Right-trivialized (spatial) covector :math:`G_W = \bar G U^\top = \Psi_R(\mathrm{ad}_{X^\top})^{-1}(M)`.

    It pairs with a spatial velocity :math:`W` (:math:`\delta U = W U`) as
    :math:`\langle \bar G, W U \rangle = \langle G_W, W \rangle`.
    """
    if U is None:
        U = torch.linalg.matrix_exp(X)
    G_bar, ok = uspace_ambient_gradient(X, M, U=U, certified=certified)
    return G_bar @ U.transpose(-1, -2), ok


def uspace_chart_velocity(
    X: torch.Tensor,                                     # (..., d, d) chart block
    W: torch.Tensor,                                     # (..., d, d) spatial velocity

    *,
    U:         Optional[torch.Tensor] = None,
    certified: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:                  # (..., d, d) v, (...) ok
    r"""Chart velocity :math:`v = \Psi_R(\mathrm{ad}_X)^{-1}(W) = L_{\log}(U, W U)`.

    By :math:`D_X(v) = \Psi_R(\mathrm{ad}_X)(v)\, U`, the chart step :math:`X + t v` moves the
    frame by :math:`W U` to first order. For a body generator :math:`\xi`,
    :math:`W = U \xi U^{-1}` and :math:`W U = U \xi`.
    """
    if U is None:
        U = torch.linalg.matrix_exp(X)
    if certified is None:
        certified, _ = principal_strip_certificate(X)
    _, v, failed = logm_frechet_pair(U, W @ U)
    ok = certified & ~failed & torch.isfinite(v).all(dim=-1).all(dim=-1)
    return torch.where(ok[..., None, None], v, torch.zeros_like(v)), ok


def uspace_pullback_body_direction(
    X: torch.Tensor,                                     # (..., d, d) chart block
    M: torch.Tensor,                                     # (..., d, d) chart covector block

    *,
    U:         Optional[torch.Tensor] = None,
    certified: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:                  # (..., d, d) xi, (...) ok
    r"""Closed-form left-trivialized Frobenius-pullback gradient :math:`\xi = U^{-1} \bar G`.

    The Frobenius-pullback chart direction :math:`v = G_0^{-1} \nabla_\varphi \ell`,
    :math:`G_0 = D^\top D`, satisfies :math:`D v = P_T \bar G = U \xi` with :math:`\xi` the solution
    of :math:`\Pi_{\mathfrak g}(U^\top U \xi) = \Pi_{\mathfrak g}(U^\top \bar G)`. For gl and block
    gl, :math:`U^\top U` preserves :math:`\mathfrak g`, so :math:`\xi = U^{-1} \bar G` blockwise
    (Prop. 1.5 (i)). It equals ``pullback_group_direction(...).xi`` without the Gram ridge.
    """
    if U is None:
        U = torch.linalg.matrix_exp(X)
    G_bar, ok = uspace_ambient_gradient(X, M, U=U, certified=certified)
    xi = torch.linalg.solve(U, G_bar)
    ok = ok & torch.isfinite(xi).all(dim=-1).all(dim=-1)
    return torch.where(ok[..., None, None], xi, torch.zeros_like(xi)), ok
