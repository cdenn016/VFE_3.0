r"""Gauge-RoPE: a block-diagonal positional rotation R(theta) for VFE_3.0 transport.

Realizes the manuscript's rotary positional structure (GL(K)_attention.tex, "Identification with
rotary positional structure"): within each irrep block of size d, coordinate pairs (2k, 2k+1)
rotate by theta_{n,k} = n * base^{-2k/d}. The executable laws are:

- rope_mode 'frame' / 'frame_value': RIGHT insertion, V_i = U_i R(theta_i)^T, so transport composes as
  Omega_ij = U_i R(theta_i)^T R(theta_j) U_j^{-1} = U_i R(theta_j - theta_i) U_j^{-1} (transport.py; config pins ``rope_insertion='right'``).
  This relative form holds for the flat transport Omega_ij only. On the projected routes the learned maps act
  between the two frames, so the routing comparison reads absolute positions (config.py); under 'frame_value'
  the content link is U_i R(theta_i)^T L R(theta_j) U_j^{-1}, with L the learned content map (O'T in the
  relational operators' chart), and since L does not commute with R in general the content message depends on
  absolute position, not only on theta_j - theta_i (audit 2026-10-20 item 69).
  The retired left insertion R(theta_i) Omega_ij R(theta_j)^T is gauge-breaking and survives only as
  a test reference.
- rope_mode 'comparison' / 'comparison_invariant': the rotation acts only on the projected routing moments,
  a_i -> R(theta_i) a_i and G_i -> R(theta_i) G_i R(theta_i)^T (gradients/relational.py), never on a
  frame.

R(theta) lies in the structure group exactly when the head image contains the coordinate-pair
rotations, which the registration trait ``contains_pair_rotations`` records
(:func:`vfe3.geometry.groups.group_traits`; config.py warns at build when it is False). The image
contains full SO(d) on block_glk, block_so, block_co and block_sl heads and on the l1 defining
block of so_n. block_glmc contains the pair rotations without containing SO(d): each rotation
cI + sJ on a pair is the realification of e^{i theta}, an element of U(1)^m in GL(m, C).
block_borel and block_unipotent contain no rotation, and on the so_n irrep blocks above the
defining one (dim > N) and on the su_n blocks the pair rotation is generally an SO(d) element
outside the irrep image rho(G). The code never assumes R in rho(G). Block-diagonal on irrep_dims so R is
orthogonal (the GL(K)-congruence divergence invariance and the block-diagonal exp fast path are
unaffected). Parameter-free; default-off (config ``rope_mode="none"``).
"""

import math
from typing import List, Optional

import torch

# The pos_rotation builder registry was retired on 2026-10-31 (audit 2026-10-31, dead-code-4): it was keyed by the
# pos_rotation field that _migrate_rope_mode folded into rope_mode on 2026-10-12, its one production lookup used the
# literal key "rope", and the comparison RoPE and the baseline already called build_rope_rotation directly, so a
# registered builder could never be selected by a configuration. Every caller now calls build_rope_rotation.


def build_rope_rotation(
    positions:  torch.Tensor,             # (N,) integer token positions
    irrep_dims: List[int],                # block sizes; sum == K

    *,
    base:     float = 100.0,
    fraction: float = 1.0,
    device:   Optional[torch.device] = None,
    dtype:    torch.dtype  = torch.float32,
) -> torch.Tensor:                        # (N, K, K) block-diagonal orthogonal rotation
    r"""Per-position block-diagonal rotation R(theta) on ``irrep_dims``.

    Within a block of size d at offset s, pairs (s+2k, s+2k+1) rotate by
    theta_{n,k} = n * base^{-2k/d}. With ``fraction`` p < 1 (p-RoPE, Barbero et al., ICLR 2025,
    arXiv:2410.06205) only the floor(p d / 2) fastest pairs k < floor(p d / 2) rotate, at their full-RoPE
    frequencies; the slower pairs are left unrotated (theta = 0), a position-free subspace. p = 1 is RoPE.
    An odd-dim block's leftover last coordinate stays identity and is
    therefore POSITIONALLY INERT (it carries zero positional content) -- a future un-gating of rope
    onto an odd-dim block must not assume full positional coverage (audit 2026-06-13 L21). The result
    is orthogonal and block-diagonal, so it preserves the block-diagonal exp fast path.

    Positions, angles, cosines and sines are formed in float64 and the rotation is cast once to ``dtype``, so a
    float32 result has entrywise error near the float32 rounding of its entries at every position and
    :math:`R_i^\top R_{i+t} = R_t` holds to that floor, instead of to :math:`|\theta| \varepsilon_{32}` from float32
    angles, which grows with the position (audit 2026-10-17 item 44). Callers cache the result, so the float64
    arithmetic is paid once per length.
    """
    pos = positions.to(device=device, dtype=torch.float64)         # (N,) float64 positions
    K = int(sum(irrep_dims))
    N = pos.shape[0]
    R = torch.eye(K, device=device, dtype=torch.float64).expand(N, K, K).clone()   # (N, K, K) float64
    start = 0
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"rope fraction must be in (0, 1], got {fraction!r}")
    for d in irrep_dims:
        n_pairs = rope_rotated_pairs(d, fraction)                  # rotated pairs (all of them at p = 1)
        if d >= 2 and n_pairs == 0:
            raise ValueError(f"rope fraction {fraction} rotates no pair of a block of size {d}")
        for k in range(n_pairs):
            freq = base ** (-2.0 * k / d)
            theta = pos * freq                                     # (N,)
            c, s = torch.cos(theta), torch.sin(theta)
            a, b = start + 2 * k, start + 2 * k + 1
            R[:, a, a] = c;  R[:, a, b] = -s
            R[:, b, a] = s;  R[:, b, b] = c
        start += d
    return R.to(dtype)                                             # one cast of the float64 rotation


def rope_rotated_pairs(d: int, fraction: float) -> int:
    r"""Number of rotated coordinate pairs of a block of size ``d`` under p-RoPE fraction p: floor(p d / 2).

    Pairs (2k, 2k+1) with k below this count rotate; the remaining coordinates (the slower pairs and an odd
    block's last coordinate) are position-free. The floor is that of the configured decimal p: the binary product
    can round below an integer it equals in decimal (0.58 x 100 = 57.99999999999999), which left 5 of the 255,000
    (width, thousandth) pairs at widths 2 to 256 one pair short, so the floor takes an absolute tolerance of 1e-9.
    It exceeds the binary rounding of p d / 2 at any practical width and is smaller than the distance to the next
    integer of p d / 2 for any fraction given to eight decimal places (audit 2026-10-31, type-contracts-2). Every
    caller (the rotation builder, the commutant projection and the configuration checks) reads the count here.
    """
    return int(math.floor(float(fraction) * d / 2.0 + 1e-9))


def rope_commutant_projection(
    G:       torch.Tensor,             # (..., r, r) symmetric comparison covariance
    n_pairs: int,                      # rotated pairs of the block, rope_rotated_pairs(r, fraction)
) -> torch.Tensor:                     # (..., r, r) P(G), commuting with every R(theta) of the block
    r"""Project a covariance onto the commutant of the RoPE torus of one comparison block.

    The rotations of :func:`build_rope_rotation` on a block of size r act on the planes
    :math:`p = (2k, 2k+1)`, :math:`k < m` = ``n_pairs``, and leave the remaining coordinates (index set 0)
    fixed. Averaging over the torus :math:`T^m` with an independent angle per plane gives the
    Frobenius-orthogonal projection onto the commutant,

    .. math::
        P(G) = \int_{T^m} R(\theta) G R(\theta)^T d\theta
             = \bigoplus_{p} \tfrac12 (G_{2k,2k} + G_{2k+1,2k+1}) I_2 \oplus G_{00},

    one isotropic variance per rotating plane, the full non-rotating block :math:`G_{00}` (the identity acts there,
    so every matrix on it commutes), and zero cross-covariance between distinct planes and between a plane and the
    non-rotating coordinates (the plane average of :math:`R_p(\theta_p)` vanishes). Then
    :math:`R P(G) R^T = P(G)` for every torus element, in particular for every positional rotation
    :math:`R_i`. P(G) is SPD when G is (an average of congruences of G).

    The torus commutant is the commutant of the realized rotations when the plane frequencies
    :math:`\mathrm{base}^{-2k/r}` are distinct, which holds for every base other than 1. At base 1 every plane
    shares one frequency and the realized commutant is larger; P(G) still commutes with every :math:`R_i`.
    Built without in-place writes, so autograd reads it directly; a bitwise-symmetric G gives a
    bitwise-symmetric P(G).
    """
    r = G.shape[-1]
    m = 2 * n_pairs
    if not 0 <= m <= r:
        raise ValueError(f"n_pairs={n_pairs} does not fit a block of size {r}")
    diagonal = torch.diagonal(G, dim1=-2, dim2=-1)                                       # (..., r)
    plane = diagonal[..., :m].reshape(*diagonal.shape[:-1], n_pairs, 2).mean(-1)         # (..., m/2)
    isotropic = torch.cat((plane.repeat_interleave(2, dim=-1),
                           torch.zeros_like(diagonal[..., m:])), dim=-1)                 # (..., r)
    fixed = torch.zeros(r, r, dtype=torch.bool, device=G.device)
    fixed[m:, m:] = True                                                                 # non-rotating block
    return torch.diag_embed(isotropic) + G * fixed                                       # (..., r, r)


def rope_commutant_is_diagonal(r: int, n_pairs: int) -> bool:
    r"""Whether :math:`P(G)` of :func:`rope_commutant_projection` is diagonal for every G: at most one coordinate of
    the block of size r stays unrotated (every coordinate of an even block at ``rope_fraction`` = 1), so the
    non-rotating block :math:`G_{00}` is at most 1 x 1."""
    return r - 2 * n_pairs <= 1


def rope_commutant_diagonal(
    g:       torch.Tensor,             # (..., r) diagonal of the comparison covariance G
    n_pairs: int,                      # rotated pairs of the block, with rope_commutant_is_diagonal(r, n_pairs)
) -> torch.Tensor:                     # (..., r) P(G), which is diagonal
    r"""The diagonal :math:`P(G)` of a block whose non-rotating part is at most one coordinate, from diag(G) alone.

    Each rotating plane takes the mean of its two variances and the unrotated coordinate keeps its own; this
    equals ``torch.diagonal(rope_commutant_projection(G, n_pairs), dim1=-2, dim2=-1)`` bit for bit, and the
    off-diagonal entries of :math:`P(G)` vanish structurally.
    """
    r = g.shape[-1]
    m = 2 * n_pairs
    if not (0 <= m <= r and rope_commutant_is_diagonal(r, n_pairs)):
        raise ValueError(f"n_pairs={n_pairs} leaves a non-rotating block of size {r - m} > 1, so P(G) is not diagonal")
    plane = g[..., :m].reshape(*g.shape[:-1], n_pairs, 2).mean(-1)                      # (..., m/2)
    return torch.cat((plane.repeat_interleave(2, dim=-1), g[..., m:]), dim=-1)           # (..., r)
