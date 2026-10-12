r"""Self-coupling coefficient alpha_i for VFE_3.0 (the weight on D(q_i || p_i)).

A config-selected registry of forms:
  state_dependent           alpha*_i = c0 / (b0 + D(q_i||p_i)), the stationary
                            point of alpha*D + R(alpha), R(alpha)=b0*alpha - c0*log alpha.
  state_dependent_per_coord per-coordinate alpha^(k)* = c0^(k)/(b0^(k) + D^(k)).
Pure: a function of the (per-position or per-coordinate) self-divergence D.
"""

from typing import Callable, Dict, Tuple

import torch

_ALPHAS:          Dict[str, Callable] = {}
_ALPHA_PER_COORD: Dict[str, bool]     = {}


def register_alpha(name: str, *, per_coord: bool = False, override: bool = False) -> Callable:
    """Decorator registering an alpha form D -> (alpha, regularizer).

    ``per_coord`` declares whether the form consumes a per-COORDINATE (unsummed) self-
    divergence of shape (..., N, K) rather than the per-position summed (..., N). The
    routing seam ``free_energy.self_divergence_for_alpha`` reads this flag to supply the
    correctly-shaped divergence, so a per-coordinate form slots in by registration alone --
    no consumer call site is edited.

    Duplicate keys fail closed (audit 2026-07-01 F12): a second registration under an
    existing name silently shadowed the first, so a config-selected seam could dispatch to
    an unintended implementation. Pass ``override=True`` to replace deliberately.
    """
    def _wrap(fn: Callable) -> Callable:
        if name in _ALPHAS and not override:
            raise KeyError(f"alpha form {name!r} already registered; pass override=True to replace")
        _ALPHAS[name] = fn
        _ALPHA_PER_COORD[name] = per_coord
        return fn
    return _wrap


def get_alpha(name: str) -> Callable:
    """Return the registered alpha form (KeyError if absent)."""
    if name not in _ALPHAS:
        raise KeyError(f"no alpha form {name!r}; available: {sorted(_ALPHAS)}")
    return _ALPHAS[name]


def alpha_is_per_coord(mode: str) -> bool:
    """Whether alpha form ``mode`` consumes a per-coordinate (unsummed) self-divergence."""
    if mode not in _ALPHA_PER_COORD:
        raise KeyError(f"no alpha form {mode!r}; available: {sorted(_ALPHAS)}")
    return _ALPHA_PER_COORD[mode]


def alpha_regularizer(
    alpha: torch.Tensor,             # (...) coupling coefficient

    *,
    b0:    'float | torch.Tensor' = 1.0,
    c0:    'float | torch.Tensor' = 1.0,
) -> torch.Tensor:
    r"""Precision regularizer R(alpha) = b0*alpha - c0*log(alpha)."""
    return b0 * alpha - c0 * torch.log(alpha)


# The complete set of named params across ALL registered alpha forms. Each form's
# **kwargs may validly contain the other forms' params (the dispatcher forwards a
# single kwargs bag); only keys OUTSIDE this universe are genuine misspellings.
_KNOWN_ALPHA_KWARGS: frozenset = frozenset({"b0", "c0", "eps"})


def _state_dependent_coefficient(
    kl:  torch.Tensor,
    b0:  'float | torch.Tensor',
    c0:  'float | torch.Tensor',
    eps: float = 1e-12,
) -> torch.Tensor:
    r"""Compute alpha* = c0 / max(b0 + D, eps), shared by both public forms."""
    return c0 / (b0 + kl).clamp(min=eps)


@register_alpha("state_dependent")
def alpha_state_dependent(
    kl:    torch.Tensor,             # (..., N) per-position self-divergence

    *,
    b0:    'float | torch.Tensor' = 1.0,
    c0:    'float | torch.Tensor' = 1.0,
    eps:   float = 1e-12,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""State-dependent alpha*_i = c0 / (b0 + D(q_i||p_i)); R(alpha*)."""
    unknown = set(kwargs) - _KNOWN_ALPHA_KWARGS
    if unknown:
        raise TypeError(f"alpha_state_dependent: unexpected kwargs {sorted(unknown)}")
    alpha = _state_dependent_coefficient(kl, b0, c0, eps)
    return alpha, alpha_regularizer(alpha, b0=b0, c0=c0)


@register_alpha("state_dependent_per_coord", per_coord=True)
def alpha_state_dependent_per_coord(
    kl:    torch.Tensor,             # (..., N, K) per-coordinate self-divergence D^(k)

    *,
    b0:    'float | torch.Tensor' = 1.0,   # scalar or (K,)
    c0:    'float | torch.Tensor' = 1.0,   # scalar or (K,)
    eps:   float = 1e-12,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Per-coordinate alpha^(k)* = c0^(k)/(b0^(k) + D^(k)); R summed by the caller's F.

    A pure function of whatever divergence it is given: the per-coordinate self-term
    sum_k alpha^(k) D^(k) results when fed the UNSUMMED per-coordinate divergence of shape
    (..., N, K) that ``free_energy.self_divergence_for_alpha`` routes here for this form
    (declared ``per_coord=True``). The per-coordinate divergence exists only for the
    diagonal family + Renyi functional, enforced by the router and at config construction.
    """
    unknown = set(kwargs) - _KNOWN_ALPHA_KWARGS
    if unknown:
        raise TypeError(f"alpha_state_dependent_per_coord: unexpected kwargs {sorted(unknown)}")
    alpha = _state_dependent_coefficient(kl, b0, c0, eps)
    return alpha, alpha_regularizer(alpha, b0=b0, c0=c0)


def alpha_gradient_coefficient(
    kl:        torch.Tensor,             # (..., N) or (..., N, K) self-divergence

    *,
    b0:        'float | torch.Tensor' = 1.0,
    c0:        'float | torch.Tensor' = 1.0,
    mode:      str,
) -> torch.Tensor:
    r"""Effective coefficient a_i multiplying d D(q_i||p_i) in the belief gradient.

    By the alpha-envelope, at the state-dependent stationary point alpha* the
    coefficient is alpha* itself: d/dx[alpha*(D)*D + R(alpha*(D))] = alpha* dD/dx,
    because alpha + alpha'(D + b0 - c0/alpha) and the bracket vanishes at alpha*.
    So no product-rule correction is needed (R must be present in F).

    The coefficient is the alpha leg of the SAME registered form used by the
    oracle (``self_coupling_alpha``), not a re-derived copy: alpha* = c0/(b0 + D). Sharing the one
    formula makes the envelope-cancellation (kernel coefficient == oracle alpha) a
    structural identity rather than a maintained coincidence.
    """
    form = get_alpha(mode)
    # Built-in forms share the same coefficient formula but need no discarded R(alpha).
    # A replacement registration keeps its complete callable and all dependencies.
    if form is alpha_state_dependent or form is alpha_state_dependent_per_coord:
        return _state_dependent_coefficient(kl, b0, c0)
    return form(kl, b0=b0, c0=c0)[0]


def self_coupling_alpha(
    kl:       torch.Tensor,          # (..., N) or (..., N, K) self-divergence

    *,
    mode:     str,
    **kwargs,                        # forwarded verbatim to the form (b0 / c0 / ...)
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Dispatch to the registered alpha form `mode`; returns (alpha, regularizer).

    Variant params are forwarded via **kwargs (each form declares its own; the
    state-dependent forms take `b0`/`c0`). A new
    form with a novel param slots in by `register_alpha` + config -- the call site
    is never edited (matching the divergence seam, which forwards only what the
    selected leaf declares).
    """
    return get_alpha(mode)(kl, **kwargs)
