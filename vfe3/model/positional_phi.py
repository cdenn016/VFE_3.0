r"""Per-position gauge-frame coordinates for exact group-product encoding.

A registry of per-position Lie-algebra coordinate builders ``pos_phi_i in R^{n_gen}``
used as the right factor in ``exp(phi_i) exp(pos_phi_i)``. ``"none"`` returns no
coordinates. ``"learned"`` slices a model-owned parameter table; ``"frozen"`` is the
parameter-free ``i * scale`` on one generator axis (a Lie-algebra ALiBi).
"""

from typing import Callable, Dict, Optional

import torch

_POS_PHI: Dict[str, Callable[..., Optional[torch.Tensor]]] = {}


def register_pos_phi(name: str, *, override: bool = False) -> Callable:
    """Decorator registering a pos-phi coordinate builder -> (N, n_gen) coords or None.

    Duplicate keys fail closed (audit 2026-07-01 round-3): a second registration under an
    existing name silently shadowed the first. Pass ``override=True`` to replace deliberately.
    """
    def _wrap(fn: Callable[..., Optional[torch.Tensor]]) -> Callable[..., Optional[torch.Tensor]]:
        if name in _POS_PHI and not override:
            raise KeyError(f"positional phi {name!r} already registered; pass override=True to replace")
        _POS_PHI[name] = fn
        return fn
    return _wrap


def get_pos_phi(name: str) -> Callable[..., Optional[torch.Tensor]]:
    """Return the registered pos-phi builder (KeyError-with-available-list if absent)."""
    if name not in _POS_PHI:
        raise KeyError(f"no pos_phi {name!r}; available: {sorted(_POS_PHI)}")
    return _POS_PHI[name]


@register_pos_phi("none")
def _pos_phi_none(
    n:     int,
    n_gen: int,

    *,
    device: torch.device,
    dtype:  torch.dtype = torch.float32,
    **kwargs,
) -> Optional[torch.Tensor]:
    r"""No positional element: returns None (the frame is left unchanged)."""
    return None


@register_pos_phi("learned")
def _pos_phi_learned(
    n:     int,
    n_gen: int,

    *,
    pos_phi_free: Optional[torch.Tensor] = None,   # (max_seq_len, n_gen) model-owned table
    device:       torch.device,
    dtype:        torch.dtype = torch.float32,
    **kwargs,
) -> torch.Tensor:
    r"""Learned absolute positional coords: the first ``n`` rows of the model's table.

    When ``n`` exceeds the trained table length (extrapolation eval beyond ``max_seq_len``; H1/EXP-13)
    the index is CLAMPED to the last learned row rather than sliced -- a bare ``pos_phi_free[:n]``
    silently returns the FULL ``max_seq_len`` table (slicing past the end does not error), which then
    shape-mismatches the ``(n, n_gen)`` frame in ``compose_phi``. Clamping degrades gracefully: unseen
    positions share the boundary offset and are positionally indistinguishable -- the expected
    learned-absolute failure-to-extrapolate, measured rather than crashed."""
    if pos_phi_free is None:
        raise ValueError("pos_phi='learned' requires the model-owned pos_phi_free table")
    t = pos_phi_free.shape[0]
    if n <= t:
        return pos_phi_free[:n]                                   # default path: byte-identical
    idx = torch.arange(n, device=pos_phi_free.device).clamp(max=t - 1)
    return pos_phi_free[idx]


@register_pos_phi("frozen")
def _pos_phi_frozen(
    n:     int,
    n_gen: int,

    *,
    scale:       float = 0.02,
    frozen_axis: int   = 0,
    device:      torch.device,
    dtype:       torch.dtype = torch.float32,
    **kwargs,
) -> torch.Tensor:
    r"""Parameter-free Lie-algebra ALiBi: pos_phi_i = (i * scale) on one generator axis.

    On the projected routes this element is composed into the frame, before the learned comparison maps, and the
    generator's exponential is not orthogonal in general, so neither it nor its relative products commute with
    those maps: the routing cost depends on absolute position (audit 2026-10-07, F31). The relative alternative is
    ``pos_rotation='rope'`` with ``rope_space='comparison'``, which rotates the projected moments after the maps.
    """
    coords = torch.zeros(n, n_gen, device=device, dtype=dtype)
    coords[:, frozen_axis] = torch.arange(n, device=device, dtype=dtype) * scale
    return coords


def positional_phi_coords(
    mode:  str,
    n:     int,
    n_gen: int,

    *,
    scale:        float = 0.02,
    frozen_axis:  int   = 0,
    pos_phi_free: Optional[torch.Tensor] = None,
    device:       torch.device,
    dtype:        torch.dtype = torch.float32,
    **kwargs,                              # variant params flow through (never edit this dispatcher)
) -> Optional[torch.Tensor]:
    r"""Dispatch to the registered pos-phi builder ``mode``; returns (N, n_gen) coords or None."""
    return get_pos_phi(mode)(
        n, n_gen, scale=scale, frozen_axis=frozen_axis,
        pos_phi_free=pos_phi_free, device=device, dtype=dtype, **kwargs,
    )
