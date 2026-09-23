r"""Gauge-RoPE: a block-diagonal positional rotation R(theta) for VFE_3.0 transport.

Realizes the manuscript's rotary positional structure (GL(K)_attention.tex, "Identification with
rotary positional structure"): within each irrep block of size d, coordinate pairs (2k, 2k+1)
rotate by theta_{n,k} = n * base^{-2k/d}, and transport composes as the outer orthogonal similarity
Omega_ij = R(theta_i) Omega_ij^learned R(theta_j)^T. R(theta) coincides with a structure-group
gauge element (the frame U_i = R(theta_i) exp(phi_i)) ONLY on blocks where the irrep image is full
SO(d) (glk/so_k; the l1 defining rep of SO(3)). On higher so_n/sp_n irrep blocks (dim > N) the
coordinate-pair rotation is a generic SO(d) element OUTSIDE the irrep image rho(SO(N))/rho(Sp(2m)),
so R Omega R^T leaves the selected structure group (config.py warns at build); the code never
assumes R in rho(G) -- it enters only as the outer similarity. Block-diagonal on irrep_dims so R is
orthogonal (the GL(K)-congruence divergence invariance and the block-diagonal exp fast path are
unaffected). Parameter-free; default-off via the ``pos_rotation`` registry ("none").
"""

from typing import Callable, Dict, List, Optional

import torch

_POS_ROTATIONS: Dict[str, Callable[..., Optional[torch.Tensor]]] = {}


def register_pos_rotation(name: str, *, override: bool = False) -> Callable:
    """Decorator registering a positional-rotation builder -> (N, K, K) rotation or None.

    Duplicate keys fail closed (audit 2026-07-01 round-3): a second registration under an
    existing name silently shadowed the first. Pass ``override=True`` to replace deliberately.
    """
    def _wrap(fn: Callable[..., Optional[torch.Tensor]]) -> Callable[..., Optional[torch.Tensor]]:
        if name in _POS_ROTATIONS and not override:
            raise KeyError(f"positional rotation {name!r} already registered; pass override=True to replace")
        _POS_ROTATIONS[name] = fn
        return fn
    return _wrap


def get_pos_rotation(name: str) -> Callable[..., Optional[torch.Tensor]]:
    """Return the registered positional-rotation builder (KeyError if absent)."""
    if name not in _POS_ROTATIONS:
        raise KeyError(f"no pos_rotation {name!r}; available: {sorted(_POS_ROTATIONS)}")
    return _POS_ROTATIONS[name]


@register_pos_rotation("none")
def _pos_rotation_none(
    positions:  torch.Tensor,
    irrep_dims: List[int],

    *,
    base:   float = 100.0,
    device: Optional[torch.device] = None,
    dtype:  torch.dtype  = torch.float32,
) -> Optional[torch.Tensor]:
    r"""No rotation: returns None (the transport is left un-rotated)."""
    return None


@register_pos_rotation("rope")
def build_rope_rotation(
    positions:  torch.Tensor,             # (N,) integer token positions
    irrep_dims: List[int],                # block sizes; sum == K

    *,
    base:   float = 100.0,
    device: Optional[torch.device] = None,
    dtype:  torch.dtype  = torch.float32,
) -> torch.Tensor:                        # (N, K, K) block-diagonal orthogonal rotation
    r"""Per-position block-diagonal rotation R(theta) on ``irrep_dims``.

    Within a block of size d at offset s, pairs (s+2k, s+2k+1) rotate by
    theta_{n,k} = n * base^{-2k/d}; an odd-dim block's leftover last coordinate stays identity and is
    therefore POSITIONALLY INERT (it carries zero positional content) -- a future un-gating of rope
    onto an odd-dim block must not assume full positional coverage (audit 2026-06-13 L21). The result
    is orthogonal and block-diagonal, so it preserves the block-diagonal exp fast path.
    """
    pos = positions.to(device=device, dtype=dtype)                 # (N,)
    K = int(sum(irrep_dims))
    N = pos.shape[0]
    R = torch.eye(K, device=device, dtype=dtype).expand(N, K, K).clone()
    start = 0
    for d in irrep_dims:
        n_pairs = d // 2
        for k in range(n_pairs):
            freq = base ** (-2.0 * k / d)
            theta = pos * freq                                     # (N,)
            c, s = torch.cos(theta), torch.sin(theta)
            a, b = start + 2 * k, start + 2 * k + 1
            R[:, a, a] = c;  R[:, a, b] = -s
            R[:, b, a] = s;  R[:, b, b] = c
        start += d
    return R
