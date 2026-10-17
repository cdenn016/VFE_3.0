r"""The exponential-family parameter layer for VFE_3.0.

A ``BeliefParams`` is a batched parameter container plus the family's math
(natural<->moment, log-partition A(theta), entropy, divergences). The divergence
functional ``renyi`` (the KL, alpha = 1; other orders are refused) dispatches on the parameter
object: a family with a ``renyi_closed_form`` method uses it (the pinned Gaussian moment forms);
a family that defines only ``log_partition_at`` (and ``natural``/``expected_statistic``) gets the
generic Bregman KL for free. This is the seam a new exponential
family slots in behind -- by writing-and-registering a subclass, never editing call sites.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, ClassVar, Dict, List, Optional, Tuple, Type

import torch

from vfe3.numerics import safe_eigvalsh

def x_minus_log1p(x: torch.Tensor) -> torch.Tensor:
    r"""Cancellation-free :math:`x - \log(1 + x)` for :math:`x > -1`.

    Inside the band :math:`|x| < u^{1/6}` (u the unit roundoff of ``x.dtype``: 0.0626 for float32, 2.2e-3
    for float64) it is the Taylor series through :math:`x^6/6`. Outside it is ``x - log1p(x)``, whose absolute
    error is about :math:`u|x|` against a value near :math:`x^2/2` (relative error near :math:`2u/|x|`), and
    whose autograd derivative :math:`1 - 1/(1 + x)` carries relative error near :math:`u/|x|`. The series
    truncates at relative :math:`2|x|^5/7` in the value and :math:`|x|^5` in the derivative
    :math:`x - x^2 + \dots + x^5`, so the edge :math:`u^{1/6}` balances the derivative's two errors at
    :math:`u^{5/6}` and keeps the value's below :math:`2u^{5/6}` (about 1e-6 in float32 and 1e-13 in float64).
    The former fixed edge :math:`10^{-3}` left float32 at about 1e-4 relative error just outside it (audit
    2026-10-16, F44). The two-log form :math:`(1 + x) - 1 - \log(1 + x)` loses about :math:`u/x^2` relative
    accuracy instead (audit 2026-10-07, F36).
    """
    near = x.abs() < (torch.finfo(x.dtype).eps / 2.0) ** (1.0 / 6.0)
    small = torch.where(near, x, torch.zeros_like(x))
    series = small.square() * (0.5 + small * (-1.0 / 3.0 + small * (0.25 + small * (-0.2 + small / 6.0))))
    wide = torch.where(near, torch.zeros_like(x), x)
    return torch.where(near, series, wide - torch.log1p(wide))


def ratio_minus_one_minus_log(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    r"""Cancellation-free :math:`r - 1 - \log r` for :math:`r = \text{num}/\text{den} > 0`.

    Near equality (:math:`|x| < 1/2`, :math:`x = (\text{num} - \text{den})/\text{den}`) it is
    :func:`x_minus_log1p`; farther out the value is at least about 0.1 and the two-log form
    :math:`r - 1 + \log\text{den} - \log\text{num}` has no cancellation, and it stays finite when :math:`x` rounds
    to :math:`-1` (a ratio below float32 resolution). Each branch sees placeholder inputs off its own region, so
    neither the value nor the gradient carries a non-finite term from the unselected branch.
    """
    x = (num - den) / den
    near = x.abs() < 0.5
    ones = torch.ones_like(x)
    small = x_minus_log1p(torch.where(near, x, torch.zeros_like(x)))
    far_num, far_den = torch.where(near, ones, num), torch.where(near, ones, den)
    far = far_num / far_den - 1.0 + torch.log(far_den) - torch.log(far_num)
    return torch.where(near, small, far)


def safe_kl_clamp(
    kl:     torch.Tensor,

    *,
    kl_max: float = 100.0,
) -> torch.Tensor:
    r"""Clamp to [0, kl_max]; positive infinity disables the finite upper ceiling."""
    kl = kl.clamp(min=0.0, max=kl_max)
    return kl.nan_to_num(nan=kl_max, posinf=kl_max, neginf=0.0)


def _logdet_chol(L: torch.Tensor) -> torch.Tensor:
    r"""log|Sigma| for SPD Sigma = L Lᵀ from its Cholesky factor L."""
    return 2.0 * torch.log(
        torch.diagonal(L, dim1=-2, dim2=-1).clamp(min=1e-12)
    ).sum(dim=-1)


class BeliefParams(ABC):
    r"""Batched parameters of an exponential family, with the family's behavior.

    Concrete subclasses hold the family's tensors (with arbitrary leading batch dims and a
    trailing coordinate structure) and implement the interface below. ``cov_kind`` is the
    single source of truth for the covariance structure (replacing name sniffing).

    Override tiers (what a new family must provide):
      - ALWAYS required (abstract): ``coordinate_dim``, ``block``, ``broadcast_over_keys``,
        ``natural``, ``log_partition_at``, ``entropy``, and the ``cov_kind`` class attribute.
      - Required ONLY to use the generic divergence path: ``expected_statistic`` (= gradA, the
        mean of the sufficient statistics), consumed by the generic KL (alpha = 1). A family
        that supplies its own ``renyi_closed_form`` never hits the generic path and need not
        override it; the base raises a clear error if the generic path needs it and it is absent.
      - Optional hooks: ``renyi_closed_form(other, *, alpha, kl_max, eps)`` (a pinned closed form
        that bypasses the generic A-path) and ``renyi_per_coord(other, ...)`` (the unsummed
        per-coordinate divergence, defined only for families whose divergence decomposes).

    CAPABILITY DECLARATIONS (audit 2026-07-26 E-03/E-04). A family whose coupling energy is defined
    only on part of the config space declares that scope HERE, so ``VFE3Config.__post_init__`` can
    reject an unsupported pairing at construction instead of letting the family raise mid-forward --
    where ``ablation.py`` misfiles the cell as a training error. Declaring on the class also keeps the
    add-by-registering contract: config validation queries the registry, never a family-name literal.
    """

    cov_kind:                 ClassVar[str]
    dispersion_is_covariance: ClassVar[bool] = False
    effective_rank_rescale:   ClassVar[bool] = False

    # Whether the coupling energy is defined only at the KL point (divergence_family='renyi',
    # renyi_order=1). Set by a family whose transported-divergence identity is KL-specific.
    requires_kl_divergence: ClassVar[bool] = False

    # The weakest pairwise transport this family's coupling energy accepts, as a level of
    # ``vfe3.geometry.transport.PAIR_TRANSPORT_KINDS``:
    #   'any'          -- consumes whatever the transport seams return (the default families),
    #   'pair_inverse' -- needs Omega_ij^{-1} (a certified cocycle, or a dense operator to invert),
    #   'coboundary'   -- needs Omega_ij = U_i U_j^{-1} exactly (frame cancellation).
    transport_requirement: ClassVar[str] = "any"

    # Whether the stored full-layout covariance is constrained to one SPD block per gauge head (zero
    # off-head blocks): the block-diagonal subfamily closed under the block_glk congruence. Such a family
    # keeps ``cov_kind='full'`` (the dense (..., K, K) storage); the moment-matched predictive update
    # returns its head-block M-projection, and config validation scopes it to the routes that preserve it.
    head_block_covariance: ClassVar[bool] = False

    @abstractmethod
    def coordinate_dim(self) -> int:
        r"""K, the number of belief coordinates."""

    @abstractmethod
    def block(self, start: int, end: int) -> "BeliefParams":
        r"""The parameters restricted to coordinate block [start, end) (per-irrep slice)."""

    @abstractmethod
    def broadcast_over_keys(self) -> "BeliefParams":
        r"""Insert a singleton key axis so a query (..., N, K) broadcasts against keys
        (..., N, N, K) in the pairwise energy."""

    @abstractmethod
    def natural(self) -> Tuple[torch.Tensor, ...]:
        r"""Natural parameters theta from these (moment) parameters."""

    @classmethod
    @abstractmethod
    def log_partition_at(cls, theta: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        r"""Log-partition A(theta) at arbitrary natural coordinates theta."""

    @abstractmethod
    def entropy(self) -> torch.Tensor:
        r"""Differential entropy H of this distribution."""

    def expected_statistic(self) -> Tuple[torch.Tensor, ...]:
        r"""E_q[T] = gradA(theta), the mean of the sufficient statistics, aligned with
        ``natural()``. Consumed by the generic KL (alpha = 1) Bregman form. Not abstract:
        a family that supplies ``renyi_closed_form`` never reaches the generic path and need
        not override this. Overriding is required only to use the generic KL path."""
        raise NotImplementedError(
            f"{type(self).__name__} has no renyi_closed_form and does not override "
            f"expected_statistic, which the generic KL (alpha=1) path requires (it is gradA, "
            f"the mean of the sufficient statistics). Provide either method."
        )

    def natural_gradient(
        self,
        grad_mu:    torch.Tensor,                    # (..., K) Euclidean grad wrt mu (location)
        grad_sigma: Optional[torch.Tensor],          # None freezes sigma; else Euclidean grad wrt sigma

        *,
        eps:        float = 1e-6,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:   # (nat_mu, nat_sigma) = Fisher^{-1} grad
        r"""Fisher preconditioner: Euclidean -> natural gradient for THIS family, ``g^{-1} grad``
        with ``g`` the family's Fisher-Rao metric on its (location, scale) coordinates. The single
        seam the E-step's preconditioner step dispatches through (``e_step_iteration``), so a family
        is descended in ITS OWN information geometry rather than a hardcoded Gaussian Fisher (which
        is the wrong DIRECTION on a non-Gaussian product manifold, not a rescalable learning rate).
        Not abstract -- existing families need no edit until they opt in; the base raises because the
        correct metric is family-dependent. ``grad_sigma=None`` requests the family-aware mean
        block only and must return ``nat_sigma=None`` without evaluating the sigma block."""
        raise NotImplementedError(
            f"{type(self).__name__} does not override natural_gradient, the family's Fisher "
            f"preconditioner (Euclidean -> natural gradient). Provide it so the E-step descends in "
            f"this family's information geometry rather than the hardcoded Gaussian Fisher."
        )

    @classmethod
    def covariance_diagonal(
        cls,
        dispersion: torch.Tensor,                    # (..., K) diagonal or (..., K, K) full parameter

        *,
        eps:        float = 1e-12,
    ) -> torch.Tensor:                               # (..., K) marginal covariance diagonal
        r"""Map the stored family dispersion to ``diag(Cov[X])``.

        The belief state's ``sigma`` slot is deliberately family-generic: it stores a Gaussian
        variance/covariance but a Laplace scale. Consumers that need uncertainty, rather than the
        raw parameter, dispatch through this hook. A family must override it because covariance
        semantics cannot be inferred from ``cov_kind`` alone.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not provide covariance_diagonal for its dispersion parameter"
        )

    @classmethod
    def covariance_floor(cls, eps: float) -> float:
        r"""Smallest covariance eigenvalue induced by the stored-dispersion floor ``eps``."""
        raise NotImplementedError(
            f"{cls.__name__} does not provide covariance_floor"
        )

    @classmethod
    def effective_rank_floor(cls, eps: float) -> float:
        r"""Denominator floor used by the family's effective-rank compatibility contract."""
        raise NotImplementedError(
            f"{cls.__name__} does not provide effective_rank_floor"
        )

    @classmethod
    def mean_fisher_precision(
        cls,
        dispersion: torch.Tensor,                    # (..., K) diagonal or (..., K, K) full parameter

        *,
        eps:        float = 1e-12,
    ) -> torch.Tensor:                               # (..., K) diagonal or (..., K, K) precision
        r"""Mean-block Fisher information in the family's stored coordinates."""
        raise NotImplementedError(
            f"{cls.__name__} does not provide mean_fisher_precision"
        )

    @classmethod
    def mean_fisher_quadratic(
        cls,
        mu:         torch.Tensor,                    # (..., K) mean coordinates
        dispersion: torch.Tensor,                    # (..., K) or (..., K, K) family parameter

        *,
        eps:        float = 1e-12,
    ) -> torch.Tensor:                               # (..., K) contributions summing to mu^T I_mu mu
        r"""Per-coordinate contributions to the mean-block Fisher quadratic."""
        precision = cls.mean_fisher_precision(dispersion, eps=eps)
        if cls.cov_kind == "full":
            precision_mu = (precision @ mu.unsqueeze(-1)).squeeze(-1)
            return mu * precision_mu
        return mu ** 2 * precision

    @classmethod
    def trust_region_scale(
        cls,
        dispersion: torch.Tensor,                    # (..., K) diagonal or (..., K, K) full parameter

        *,
        eps:        float = 1e-12,
    ) -> torch.Tensor:                               # (..., K) scale or (..., K, K) whitening factor
        r"""Factor ``L`` used to whiten a mean perturbation in this family's Fisher geometry."""
        raise NotImplementedError(
            f"{cls.__name__} does not provide trust_region_scale"
        )

    @classmethod
    def diagnostic_labels(cls) -> Dict[str, str]:
        r"""Human-readable labels distinguishing stored dispersion from covariance statistics."""
        raise NotImplementedError(
            f"{cls.__name__} does not provide diagnostic_labels"
        )

    @classmethod
    def diagnostic_statistics(
        cls,
        dispersion: torch.Tensor,                    # (..., K) diagonal or (..., K, K) full parameter

        *,
        eps:        float = 1e-12,
    ) -> Dict[str, torch.Tensor]:
        r"""Family-aware covariance spectrum and trace without computing Fisher inverses."""
        covariance_diagonal = cls.covariance_diagonal(dispersion, eps=eps)
        if cls.cov_kind == "full":
            covariance = 0.5 * (dispersion + dispersion.transpose(-1, -2))
            covariance_spectrum = safe_eigvalsh(covariance)
        else:
            covariance_spectrum = covariance_diagonal
        return {
            "covariance_diagonal": covariance_diagonal,
            "covariance_spectrum": covariance_spectrum,
            "covariance_trace":    covariance_diagonal.sum(dim=-1),
        }

    @classmethod
    def stack(cls, parts: List["BeliefParams"], *, dim: int = 0) -> "BeliefParams":
        r"""Stack a list of same-family parts into one ``BeliefParams`` along a NEW axis ``dim``,
        stacking each underlying tensor of the parts. Family-agnostic batching primitive: a single
        functional call over the stacked axis then computes every part's divergence at once. Not
        abstract -- a family that never
        takes the batched path need not override it. Overriding is required only to batch."""
        raise NotImplementedError(
            f"{cls.__name__} does not override stack, the family-agnostic batching primitive "
            f"(stack each part's underlying tensor along a new axis). Provide it to batch the "
            f"per-block loop."
        )

    @classmethod
    def from_covariance(
        cls,
        mu:         torch.Tensor,         # (..., K) locations
        covariance: torch.Tensor,         # (..., K, K) FULL covariance
    ) -> "BeliefParams":
        r"""This family's belief at ``mu`` with the full covariance ``covariance``.

        The family's READOUT of a full second moment into its own stored dispersion, and the hook
        :func:`vfe3.geometry.groups.check_admissible` dispatches through to push a belief forward by
        the GL(K) congruence ``mu -> g mu, Sigma -> g Sigma g^T`` (audit 2026-07-26 E-05). Before this
        hook that verifier -- the project's only executable invariance certificate -- branched on three
        hardcoded family names and raised for everything else, so neither family registered in 2026-07
        could be covered by any invariance test. A family with no full-covariance readout (a location
        family whose dispersion is not a second moment) leaves this unimplemented and the verifier
        reports the gap, which is the documented extension point.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not provide from_covariance, the family's readout of a full "
            f"covariance into its stored dispersion. Provide it to extend the executable "
            f"gauge-invariance check to this family."
        )


_FAMILIES: Dict[str, Type[BeliefParams]] = {}


def register_family(name: str, *, override: bool = False) -> Callable[[Type[BeliefParams]], Type[BeliefParams]]:
    r"""Register a ``BeliefParams`` subclass under ``name`` (the config ``family`` value).

    Duplicate keys fail closed (audit 2026-07-01 round-3): a second registration under an
    existing name silently shadowed the first. Pass ``override=True`` to replace deliberately.
    """
    def _wrap(cls: Type[BeliefParams]) -> Type[BeliefParams]:
        if name in _FAMILIES and not override:
            raise KeyError(f"family {name!r} already registered; pass override=True to replace")
        _FAMILIES[name] = cls
        return cls
    return _wrap


def get_family(name: str) -> Type[BeliefParams]:
    r"""The registered ``BeliefParams`` subclass for ``name`` (KeyError if absent)."""
    if name not in _FAMILIES:
        raise KeyError(f"no family registered under {name!r}; available: {sorted(_FAMILIES)}")
    return _FAMILIES[name]


def family_cov_kind(name: str) -> str:
    r"""Covariance structure ("diagonal" | "full") of family ``name``, from its subclass."""
    return get_family(name).cov_kind


def divergence_families() -> Tuple[str, ...]:
    r"""Registered family names (``cfg.family`` values; the divergence is pinned, see :func:`divergence_functionals`)."""
    return tuple(sorted(_FAMILIES))


@dataclass(frozen=True)
class FunctionalDisplayMetadata:
    """Human-readable metadata carried by a divergence registration."""

    label: str
    units: Optional[str]

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("functional display label must be a nonempty string")
        if self.units is not None and (not isinstance(self.units, str) or not self.units.strip()):
            raise ValueError("functional display units must be a nonempty string or None")


_FUNCTIONALS: Dict[str, Callable[..., torch.Tensor]] = {}
_FUNCTIONAL_DISPLAYS: Dict[str, FunctionalDisplayMetadata] = {}


def register_functional(
    name: str,
    *,
    display: Optional[FunctionalDisplayMetadata] = None,
    override: bool = False,
) -> Callable[[Callable[..., torch.Tensor]], Callable[..., torch.Tensor]]:
    r"""Register a divergence functional (renyi, ...) under ``name`` (the ``divergence_family``).

    Duplicate keys fail closed (audit 2026-07-01 round-3): a second registration under an
    existing name silently shadowed the first. Pass ``override=True`` to replace deliberately.
    """
    def _wrap(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        if name in _FUNCTIONALS and not override:
            raise KeyError(f"divergence functional {name!r} already registered; pass override=True to replace")
        if display is not None:
            _FUNCTIONAL_DISPLAYS[name] = display
        elif name not in _FUNCTIONAL_DISPLAYS:
            _FUNCTIONAL_DISPLAYS[name] = FunctionalDisplayMetadata("divergence", None)
        _FUNCTIONALS[name] = fn
        return fn
    return _wrap


def get_functional(name: str) -> Callable[..., torch.Tensor]:
    r"""The registered divergence functional for ``name`` (KeyError if absent)."""
    if name not in _FUNCTIONALS:
        raise KeyError(f"no functional registered under {name!r}; available: {sorted(_FUNCTIONALS)}")
    return _FUNCTIONALS[name]


def get_functional_display(name: str) -> FunctionalDisplayMetadata:
    """Display metadata for one registered divergence functional."""
    get_functional(name)
    return _FUNCTIONAL_DISPLAYS[name]


def divergence_functionals() -> Tuple[str, ...]:
    r"""Registered functional names, the divergences reachable from tests and post-hoc analysis.

    ``divergence_family`` (and the Renyi order) is pinned by ``VFE3Config`` validation, so training
    runs use only the pinned functional; the other members are not configuration values.
    """
    return tuple(sorted(_FUNCTIONALS))


# Per-COORDINATE divergence registry (the unsummed (..., K) form consumed by the
# state_dependent_per_coord alpha). A divergence has a per-coordinate form ONLY when it decomposes
# as a sum over coordinates: the KL does (renyi_per_coord; ordered conditional terms for the full
# family). A functional with no member here is rejected by ``free_energy.self_divergence_per_coord``.
_FUNCTIONALS_PER_COORD: Dict[str, Callable[..., torch.Tensor]] = {}


def register_functional_per_coord(name: str, *, override: bool = False) -> Callable[[Callable[..., torch.Tensor]], Callable[..., torch.Tensor]]:
    r"""Register a PER-COORDINATE divergence functional under ``name`` (the ``divergence_family``).

    Duplicate keys fail closed (audit 2026-07-01 round-3): a second registration under an
    existing name silently shadowed the first. Pass ``override=True`` to replace deliberately.
    """
    def _wrap(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        if name in _FUNCTIONALS_PER_COORD and not override:
            raise KeyError(f"per-coordinate divergence functional {name!r} already registered; pass override=True to replace")
        _FUNCTIONALS_PER_COORD[name] = fn
        return fn
    return _wrap


def get_functional_per_coord(name: str) -> Callable[..., torch.Tensor]:
    r"""The registered per-coordinate functional for ``name`` (KeyError if absent)."""
    if name not in _FUNCTIONALS_PER_COORD:
        raise KeyError(
            f"no per-coordinate functional registered under {name!r}; available: "
            f"{sorted(_FUNCTIONALS_PER_COORD)}"
        )
    return _FUNCTIONALS_PER_COORD[name]


def has_per_coord_functional(name: str) -> bool:
    r"""Whether divergence ``name`` has a registered per-coordinate (coordinate-decomposing) form."""
    return name in _FUNCTIONALS_PER_COORD


def divergence_functionals_per_coord() -> Tuple[str, ...]:
    r"""Registered per-coordinate functional names (the divergences that decompose coordinate-wise)."""
    return tuple(sorted(_FUNCTIONALS_PER_COORD))


# Renyi orders with |alpha - 1| < _KL_ALPHA_SWITCH are evaluated as the KL (alpha = 1) limit; every
# other order is refused (audit 2026-10-16 F93: the retired divergence_family / renyi_order fields left
# KL as the only configured divergence, so the alpha != 1 blend kernels were removed). Defined once
# here; the family modules import it (they already import base, which imports no family module).
_KL_ALPHA_SWITCH: float = 1e-6


def _require_kl_order(alpha: float, where: str) -> None:
    r"""Refuse a Renyi order outside the KL switch :math:`|\alpha - 1| < 10^{-6}` (only KL is implemented)."""
    if not abs(alpha - 1.0) < _KL_ALPHA_SWITCH:
        raise ValueError(
            f"{where}: only the KL divergence (alpha = 1) is implemented, got alpha={alpha!r}; "
            f"the Renyi alpha != 1 kernels were removed (audit 2026-10-16 F93)")


def _renyi_from_log_partition(
    q:       BeliefParams,
    p:       BeliefParams,

    *,
    alpha:   float,
    kl_max:  float,
    eps:     float,
) -> torch.Tensor:
    r"""Generic KL from the log-partition (for families with no closed form).

    KL = A(tp) - A(tq) - <gradA(tq), tp - tq>, gradA(tq) = E_q[T] (expected_statistic). ``alpha`` is
    kept for signature stability and must lie in the KL switch (ValueError otherwise).

    ``eps`` is accepted for signature symmetry with the closed-form path and is intentionally
    unused here: the A-form is evaluated directly from the natural parameters with no clamp.
    """
    _require_kl_order(alpha, "_renyi_from_log_partition")
    cls = type(q)
    tq = q.natural()
    tp = p.natural()
    grad = q.expected_statistic()                           # E_q[T] = gradA(theta_q)
    A_q = cls.log_partition_at(tq)                           # batch-shaped (parameter axes summed)
    batch_ndim = A_q.dim()
    # Bregman KL = A(tp) - A(tq) - sum_c <gradA_c, (tp - tq)_c>, where each natural-parameter
    # component c is contracted over ITS parameter axes (the trailing dims beyond the batch).
    # A vector statistic (..., K) sums the last axis; a matrix statistic (..., K, K) is
    # Frobenius-contracted over the last two -- so a matrix-parameter family (e.g. the full
    # Gaussian's t2) works through the generic path, not only vector-parameter families.
    inner: 'torch.Tensor | float' = 0.0              # float seed; becomes a Tensor on the first term
    for g, a, b in zip(grad, tq, tp):
        term = g * (b - a)
        param_axes = tuple(range(batch_ndim, term.dim()))
        inner = inner + (term.sum(dim=param_axes) if param_axes else term)
    div = cls.log_partition_at(tp) - A_q - inner
    return safe_kl_clamp(div, kl_max=kl_max)


def renyi(
    q:       BeliefParams,
    p:       BeliefParams,

    *,
    alpha:   float = 1.0,
    kl_max:  float = 100.0,
    eps:     float = 1e-6,
    **kwargs,
) -> torch.Tensor:
    r"""KL(q || p) between two parameter objects, the alpha = 1 Renyi divergence.

    Uses ``q.renyi_closed_form`` when the family provides one (the pinned Gaussian moment
    form); otherwise the generic Bregman path. ``alpha`` is kept so call signatures stay stable;
    any order outside the KL switch :math:`|\alpha - 1| < 10^{-6}` raises ``ValueError`` (audit
    2026-10-16 F93).

    The trailing ``**kwargs`` is the permissive functional contract every divergence-registry
    member shares (a member ignores params it does not use); ``renyi`` consumes ``alpha``.
    """
    _require_kl_order(alpha, "renyi")
    closed = getattr(q, "renyi_closed_form", None)
    if closed is not None:
        return closed(p, alpha=alpha, kl_max=kl_max, eps=eps)
    return _renyi_from_log_partition(q, p, alpha=alpha, kl_max=kl_max, eps=eps)


def kl(
    q:       BeliefParams,
    p:       BeliefParams,

    *,
    kl_max:  float = 100.0,
    eps:     float = 1e-6,
) -> torch.Tensor:
    r"""KL(q || p) = Renyi at alpha = 1."""
    return renyi(q, p, alpha=1.0, kl_max=kl_max, eps=eps)


def renyi_per_coord(
    q:       BeliefParams,
    p:       BeliefParams,

    *,
    alpha:   float = 1.0,
    kl_max:  float = 100.0,
    eps:     float = 1e-6,
    **kwargs,
) -> torch.Tensor:                             # (..., K) per-coordinate Renyi/KL D^(k)
    r"""Per-coordinate KL: the unsummed coordinate terms (family hook).

    Defined for families exposing ``renyi_per_coord``. Diagonal Gaussians give the coordinate KL
    terms and full Gaussians the ordered conditional KL contributions; ``alpha`` is kept for
    signature stability and must lie in the KL switch (ValueError otherwise, audit 2026-10-16 F93).
    """
    _require_kl_order(alpha, "renyi_per_coord")
    return q.renyi_per_coord(p, alpha=alpha, kl_max=kl_max, eps=eps)


register_functional(
    "renyi", display=FunctionalDisplayMetadata("Renyi divergence", "dimensionless"))(renyi)

# The per-coordinate form of the KL (see _FUNCTIONALS_PER_COORD).
register_functional_per_coord("renyi")(renyi_per_coord)
