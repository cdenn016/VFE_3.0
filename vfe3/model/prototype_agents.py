r"""Prototype-agent posterior: the minimal agent model with a static library of template agents.

Source theory: ``Research/manuscripts/Attention/minimal_agent_model.tex`` (Definition ma:def:model,
Proposition ma:prop:minimal (a) and (d), Proposition ma:prop:mean_field), specialized to full-rank
identity query and sending maps, :math:`S = T = I`, with the channel noise absorbed into the template
covariances. Implementation plan: ``docs/implementation/minimal-agent-plan-2026-10-06.md``, Section 3.

A token's working belief is :math:`\mathcal N(\xi_i, C_i)`. Template :math:`a` (one of :math:`M`)
carries the belief :math:`p_a = \mathcal N(\nu_a, \Lambda_a)`, :math:`\Lambda_a = \operatorname{diag}\lambda_a`,
:math:`\lambda_a = e^{\log\lambda_a} > 0`, and the prior weight :math:`\pi_a = \operatorname{softmax}(\log\pi)_a`.
With :math:`W_{ia} = C_i + \Lambda_a` the exact posterior (Proposition ma:prop:minimal (a), (d)) is

.. math::
    w_{ia} = \operatorname{softmax}_a\bigl(\log\pi_a + \log\mathcal N(\nu_a;\xi_i,W_{ia})\bigr),\qquad
    r_{ia} = W_{ia}^{-1}(\nu_a - \xi_i),\qquad u_i = \sum_a w_{ia}r_{ia},

.. math::
    \bar\Lambda_i = \sum_a w_{ia}W_{ia}^{-1},\qquad
    R_i = \sum_a w_{ia}r_{ia}r_{ia}^\top - u_iu_i^\top,\qquad
    \xi_i^+ = \xi_i + C_iu_i,\qquad
    C_i^+ = C_i - C_i(\bar\Lambda_i - R_i)C_i,

the moments of the mixture :math:`\sum_a w_{ia}\mathcal N(m_{ia}, P_{ia})` with
:math:`m_{ia} = \xi_i + C_iW_{ia}^{-1}(\nu_a - \xi_i)` and
:math:`P_{ia} = C_i - C_iW_{ia}^{-1}C_i = (C_i^{-1} + \Lambda_a^{-1})^{-1}`. The covariance is evaluated
in the cancellation-free form :math:`C_i^+ = C_i\Psi_i + C_iR_iC_i` with
:math:`\Psi_i = \sum_a w_{ia}W_{ia}^{-1}\Lambda_a`, which follows from
:math:`C_i - C_iW_{ia}^{-1}C_i = C_iW_{ia}^{-1}(W_{ia} - C_i) = C_iW_{ia}^{-1}\Lambda_a`: both terms are
positive semidefinite by construction and the first is positive definite, so no difference of two
large matrices is formed when a sharp template pins the belief.

The mean-field step (Proposition ma:prop:mean_field) uses the label update at the token's prior and
the power posterior at fixed labels,

.. math::
    r_{ia} = \operatorname{softmax}_a\bigl(\log\pi_a - D_{\mathrm{KL}}(\mathcal N(\xi_i, C_i)\,\|\,p_a)\bigr),\qquad
    D_i = \sum_a r_{ia}\lambda_a^{-1},\qquad \eta_i = \sum_a r_{ia}\Lambda_a^{-1}\nu_a,

.. math::
    P_i = (C_i^{-1} + \operatorname{diag}D_i)^{-1},\qquad
    \xi_i^+ = P_i(C_i^{-1}\xi_i + \eta_i) = \xi_i + P_i(\eta_i - D_i\odot\xi_i),\qquad C_i^+ = P_i,

which omits the between-template spread :math:`C_iR_iC_i` of the exact covariance. That is one coordinate-ascent
sweep; :func:`prototype_mean_field_posterior` runs ``cavi_steps`` of them.

Covariance layouts. ``C`` is diagonal ``(..., K)``, dense ``(..., K, K)``, or compact blocks
``(..., G, b, b)`` with :math:`Gb = K` (a block-diagonal :math:`C_i`), and every covariance output uses the
input layout. Because :math:`\Lambda_a` is diagonal, :math:`W_{ia}` and :math:`C_i^{-1} + \operatorname{diag}D_i`
keep the block structure, so the linear algebra runs block-batched with block size 1 (elementwise),
:math:`b`, or :math:`K`. The exact spread :math:`C_iR_iC_i` couples blocks; a diagonal or block layout
returns the diagonal (the diagonal blocks) of :math:`C_i^+`, the moment-matched member of that family,
which needs only the diagonal (the diagonal blocks) of :math:`R_i`. The dense layout forms the dense spread.

The computing functions (:func:`template_kl_logits` and the two posteriors) run with autocast disabled and
IEEE fp32 GEMMs, at float64 when an input is float64 and float32 otherwise, and return that compute dtype.
"""

from __future__ import annotations

import math
from typing import List, NamedTuple, Optional, Tuple

import torch
from torch.utils.checkpoint import checkpoint

from vfe3.runtime import ieee_fp32_matmul


# Exact-path tile budget: rows * templates * (2 pair tensors of E elements) bytes, E = K (diagonal) or G b^2
# (blocks, dense), as the noisy-query routing tiles (vfe3.relational_scoring.projected_noisy_query_cost)
# budget theirs. It bounds each pair tensor at tile_bytes / 2, not the pair data a tile holds: the diagonal
# tile forms up to nine pair tensors and autograd saves several. At a 16 MiB pair tensor the measured peak
# was about 6, 7 and 11 pair tensors for a forward without gradients, a forward with gradients and an
# unchunked forward and backward on the diagonal layout, and about 5, 5 and 7 on the block layout.
PROTOTYPE_TILE_BYTES: int = 32 * 1024 ** 2
PROTOTYPE_INFERENCE_MODES: Tuple[str, ...] = ("mean_field", "exact")
PROTOTYPE_COVARIANCE_CONTRACTS: Tuple[str, ...] = ("passthrough", "posterior")
COVARIANCE_LAYOUTS: Tuple[str, ...] = ("diagonal", "dense", "blocks")


class ExactPrototypePosterior(NamedTuple):
    r"""Exact template posterior of every token; covariance-shaped fields use the input layout."""

    xi_plus:    torch.Tensor               # (..., K) posterior mean xi^+ = xi + C u
    delta_xi:   torch.Tensor               # (..., K) mean increment C u
    C_plus:     Optional[torch.Tensor]     # posterior covariance C^+ (None unless need_covariance)
    weights:    torch.Tensor               # (..., M) template posterior w
    u:          torch.Tensor               # (..., K) sum_a w_a r_a
    Lambda_bar: Optional[torch.Tensor]     # sum_a w_a W_a^{-1} (None unless need_precision)
    R:          Optional[torch.Tensor]     # between-template spread of r (None unless need_covariance)


class MeanFieldPrototypePosterior(NamedTuple):
    r"""Mean-field template posterior of every token; ``C_plus`` uses the input layout."""

    xi_plus:          torch.Tensor           # (..., K) power-posterior mean
    delta_xi:         torch.Tensor           # (..., K) mean increment P (eta - D xi)
    C_plus:           Optional[torch.Tensor] # P = (C^{-1} + diag D)^{-1} (None unless need_covariance)
    responsibilities: torch.Tensor           # (..., M) label update r at the token's prior
    D:                torch.Tensor           # (..., K) sum_a r_a / lambda_a
    eta:              torch.Tensor           # (..., K) sum_a r_a nu_a / lambda_a


class _TileStats(NamedTuple):
    r"""Template-chunk statistics of a row tile, relative to the detached chunk maximum ``m`` of the logits."""

    m:    torch.Tensor                     # (r,) max_a l_ia (detached stabilizer)
    s:    torch.Tensor                     # (r,) sum_a exp(l_ia - m_i)
    mean: torch.Tensor                     # (r, K) or (r, G, b): normalized weighted mean of r_ia
    M2:   Optional[torch.Tensor]           # (r, K) or (r, G, b, b): sum_a e^{l-m} (r - mean)(r - mean)^T
    Psi:  Optional[torch.Tensor]           # (r, K) or (r, G, b, b): sum_a e^{l-m} W^{-1} Lambda_a
    Lbar: Optional[torch.Tensor]           # (r, K) or (r, G, b, b): sum_a e^{l-m} W^{-1}


def covariance_layout(
    xi: torch.Tensor,                      # (..., K)
    C:  torch.Tensor,                      # (..., K), (..., K, K) or (..., G, b, b)
) -> str:
    r"""Return ``'diagonal'``, ``'dense'`` or ``'blocks'`` for a working covariance next to the mean ``xi``.

    The layout is read from the number of axes: ``C.ndim`` equal to ``xi.ndim`` (diagonal entries),
    ``xi.ndim + 1`` (dense :math:`K \times K`), or ``xi.ndim + 2`` (compact blocks
    :math:`(G, b, b)`, :math:`Gb = K`). The leading axes must equal those of ``xi``.
    """
    K, lead = xi.shape[-1], tuple(xi.shape[:-1])
    if C.dim() == xi.dim() and tuple(C.shape) == tuple(xi.shape):
        return "diagonal"
    if C.dim() == xi.dim() + 1 and tuple(C.shape) == (*lead, K, K):
        return "dense"
    if (C.dim() == xi.dim() + 2 and tuple(C.shape[:-3]) == lead and C.shape[-1] == C.shape[-2]
            and C.shape[-3] * C.shape[-1] == K):
        return "blocks"
    raise ValueError(
        f"working covariance {tuple(C.shape)} does not match the mean {tuple(xi.shape)} as diagonal "
        "(..., K), dense (..., K, K) or compact blocks (..., G, b, b) with G b = K")


def covariance_diagonal(
    C:      torch.Tensor,                  # (..., K), (..., K, K) or (..., G, b, b)
    layout: str,
) -> torch.Tensor:                         # (..., K)
    r"""The diagonal :math:`(C_{kk})_k` of a working covariance in any layout."""
    if layout == "diagonal":
        return C
    return torch.diagonal(C, dim1=-2, dim2=-1).flatten(-2) if layout == "blocks" \
        else torch.diagonal(C, dim1=-2, dim2=-1)


def _compute_dtype(*tensors: torch.Tensor) -> torch.dtype:
    return torch.float64 if any(t.dtype == torch.float64 for t in tensors) else torch.float32


def _broadcast(t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    r"""View a per-row ``(r,)`` factor with trailing singleton axes so it broadcasts against ``like``."""
    return t.reshape(*t.shape, *([1] * (like.dim() - t.dim())))


def _symmetric(A: torch.Tensor) -> torch.Tensor:
    return 0.5 * (A + A.transpose(-1, -2))


@ieee_fp32_matmul()
def template_kl_logits(
    xi:         torch.Tensor,              # (..., K) working means
    c:          torch.Tensor,              # (..., K) working variances diag C_i
    nu:         torch.Tensor,              # (M, K) template means
    log_lambda: torch.Tensor,              # (M, K) template log variances
    log_pi:     torch.Tensor,              # (M,) template log prior weights (unnormalized)
) -> torch.Tensor:                         # (..., M)
    r"""Mean-field label logits :math:`\log\pi_a - D_{\mathrm{KL}}(\mathcal N(\xi_i, C_i)\,\|\,p_a)` up to a row constant.

    .. math::
        \ell_{ia} = \log\pi_a - \tfrac12\sum_k\Bigl[\frac{(C_i)_{kk} + (\xi_{ik} - \nu_{ak})^2}{\lambda_{ak}}
            + \log\lambda_{ak}\Bigr] + \mathrm{const}_i,

    since :math:`\Lambda_a` is diagonal only :math:`\operatorname{diag}C_i` enters, and the entropy of the
    token's belief is template independent. The expanded square is contracted as three matmuls, the same
    expression as ``GaussianTemplateBlockMLP.responsibilities`` (whose softmax is this label step). It
    cancels terms of order :math:`K((C_i)_{kk} + \xi_{ik}^2)/\lambda_{ak}`, so, like the posteriors, it runs
    with autocast disabled at float64 when an input is float64 and float32 otherwise.
    """
    dtype = _compute_dtype(xi, c, nu, log_lambda, log_pi)
    with torch.autocast(device_type=xi.device.type, enabled=False):
        x, cw, n, ll, lp = (t.to(dtype) for t in (xi, c, nu, log_lambda, log_pi))
        precision = torch.exp(-ll)                                           # (M, K)
        quadratic = (
            (cw + x.square()) @ precision.T                                  # (..., M)
            - 2.0 * x @ (n * precision).T
            + (n.square() * precision).sum(dim=-1)
        )
        return lp - 0.5 * (quadratic + ll.sum(dim=-1))


def _tile_statistics(
    logits:          torch.Tensor,          # (r, m)
    r_vec:           torch.Tensor,          # (r, m, K) or (r, m, G, b): W^{-1}(nu - xi)
    W_inv:           Optional[torch.Tensor],  # (r, m, K) or (r, m, G, b, b)
    lam:             torch.Tensor,          # (m, K) or (m, G, b)
    need_covariance: bool,
    need_precision:  bool,
    blocks:          bool,
) -> _TileStats:
    r"""Weighted sums of one tile relative to its own detached maximum :math:`m_i = \max_a\ell_{ia}`.

    With :math:`p_{ia} = e^{\ell_{ia} - m_i}` and :math:`s_i = \sum_a p_{ia}`, returns the normalized mean
    :math:`\bar r_i = \sum_a p_{ia}r_{ia}/s_i`, the centered second moment
    :math:`\sum_a p_{ia}(r_{ia} - \bar r_i)(r_{ia} - \bar r_i)^\top` (two-pass within the tile, so it is a
    sum of positive semidefinite terms), :math:`\sum_a p_{ia}W_{ia}^{-1}\Lambda_a` and
    :math:`\sum_a p_{ia}W_{ia}^{-1}`. The maximum is a stabilizer: every normalized result is independent
    of it, so it carries no gradient.
    """
    m = logits.amax(dim=-1).detach()                                         # (r,)
    p = torch.exp(logits - m.unsqueeze(-1))                                  # (r, m), max entry 1
    s = p.sum(dim=-1)                                                        # (r,) >= 1
    if blocks:
        mean = torch.einsum("ra,ragx->rgx", p, r_vec) / _broadcast(s, r_vec[:, 0])
    else:
        mean = (p.unsqueeze(-2) @ r_vec).squeeze(-2) / s.unsqueeze(-1)       # (r, K)
    M2 = Psi = Lbar = None
    if need_covariance:
        dev = r_vec - mean.unsqueeze(1)                                      # (r, m, ...)
        if blocks:
            M2 = torch.einsum("ra,ragx,ragy->rgxy", p, dev, dev)             # (r, G, b, b)
            Psi = torch.einsum("ra,ragxy,agy->rgxy", p, W_inv, lam)          # sum p W^{-1} Lambda_a
        else:
            M2 = (p.unsqueeze(-2) @ dev.square()).squeeze(-2)                # (r, K)
            Psi = (p.unsqueeze(-2) @ (W_inv * lam)).squeeze(-2)              # (r, K) sum p lambda/(c+lambda)
    if need_precision:
        Lbar = (torch.einsum("ra,ragxy->rgxy", p, W_inv) if blocks
                else (p.unsqueeze(-2) @ W_inv).squeeze(-2))
    return _TileStats(m, s, mean, M2, Psi, Lbar)


# The tiles carry their own IEEE fp32 GEMM policy: under checkpointing they are recomputed in backward,
# outside the caller's context, and must reproduce the forward values.
@ieee_fp32_matmul()
def _exact_diagonal_tile(
    xi:              torch.Tensor,          # (r, K)
    c:               torch.Tensor,          # (r, K) diag C_i
    nu:              torch.Tensor,          # (m, K)
    lam:             torch.Tensor,          # (m, K)
    log_pi:          torch.Tensor,          # (m,)
    need_covariance: bool,
    need_precision:  bool,
) -> Tuple[torch.Tensor, _TileStats, torch.Tensor]:
    r"""One (rows x templates) tile of the exact posterior for diagonal :math:`C_i`, elementwise.

    :math:`W_{iak} = c_{ik} + \lambda_{ak}`, :math:`r_{iak} = (\nu_{ak} - \xi_{ik})/W_{iak}` and
    :math:`\ell_{ia} = \log\pi_a - \tfrac12\sum_k[(\nu_{ak} - \xi_{ik})r_{iak} + \log W_{iak}]`, the log
    density up to the row constant :math:`-\tfrac K2\log 2\pi` and the prior normalizer. Returns the
    logits, the tile statistics and a failure flag (some :math:`W_{iak}` not positive).
    """
    W = c.unsqueeze(-2) + lam                                                # (r, m, K) diag W_ia
    W_inv = W.reciprocal()
    diff = nu - xi.unsqueeze(-2)                                             # (r, m, K) nu_a - xi_i
    r_vec = diff * W_inv                                                     # (r, m, K) r_ia
    logits = log_pi - 0.5 * ((diff * r_vec).sum(dim=-1) + W.log().sum(dim=-1))   # (r, m)
    stats = _tile_statistics(logits, r_vec, W_inv, lam, need_covariance, need_precision, blocks=False)
    return logits, stats, ~(W > 0).all()


@ieee_fp32_matmul()
def _exact_block_tile(
    xi:              torch.Tensor,          # (r, G, b)
    C:               torch.Tensor,          # (r, G, b, b) diagonal blocks of C_i
    nu:              torch.Tensor,          # (m, G, b)
    lam:             torch.Tensor,          # (m, G, b)
    log_pi:          torch.Tensor,          # (m,)
    need_covariance: bool,
    need_precision:  bool,
) -> Tuple[torch.Tensor, _TileStats, torch.Tensor]:
    r"""One (rows x templates) tile of the exact posterior for block-diagonal :math:`C_i` (dense: one block).

    Each pair and block has its own :math:`b \times b` Cholesky factor of
    :math:`W_{iag} = (C_i)_{gg} + \operatorname{diag}\lambda_{ag}`, zero jitter and no repair;
    :math:`\log\mathcal N(\nu_a;\xi_i,W_{ia})` sums over blocks because :math:`W_{ia}` is block diagonal.
    Returns the logits, the tile statistics and a factorization-failure flag.
    """
    W = C.unsqueeze(1) + torch.diag_embed(lam)                               # (r, m, G, b, b), exactly symmetric
    L, info = torch.linalg.cholesky_ex(W, check_errors=False)               # (r, m, G, b, b)
    diff = nu - xi.unsqueeze(1)                                              # (r, m, G, b)
    r_vec = torch.cholesky_solve(diff.unsqueeze(-1), L).squeeze(-1)          # (r, m, G, b) W^{-1}(nu - xi)
    logdet = 2.0 * L.diagonal(dim1=-2, dim2=-1).log().sum(dim=(-2, -1))      # (r, m)
    logits = log_pi - 0.5 * ((diff * r_vec).sum(dim=(-2, -1)) + logdet)      # (r, m)
    W_inv = torch.cholesky_inverse(L) if (need_covariance or need_precision) else None
    stats = _tile_statistics(logits, r_vec, W_inv, lam, need_covariance, need_precision, blocks=True)
    return logits, stats, (info != 0).any()


def _merge_statistics(a: _TileStats, b: _TileStats, outer: bool) -> _TileStats:
    r"""Merge two template chunks of one row tile (online log-sum-exp with Chan's pairwise update).

    With :math:`m = \max(m_a, m_b)`, :math:`S_a = s_ae^{m_a - m}`, :math:`S_b = s_be^{m_b - m}`,
    :math:`S = S_a + S_b` and :math:`\delta = \bar r_b - \bar r_a`,

    .. math::
        \bar r = \bar r_a + \delta\,S_b/S,\qquad
        M_2 = M_{2,a}e^{m_a - m} + M_{2,b}e^{m_b - m} + \frac{S_aS_b}{S}\delta\delta^\top,

    the weighted form of Chan, Golub and LeVeque's parallel variance update, so the merged spread stays
    a sum of positive semidefinite terms. ``outer`` selects the matrix form (block layouts).
    """
    m = torch.maximum(a.m, b.m)                                              # (r,)
    fa, fb = torch.exp(a.m - m), torch.exp(b.m - m)                          # (r,) constants
    Sa, Sb = a.s * fa, b.s * fb
    S = Sa + Sb
    delta = b.mean - a.mean
    mean = a.mean + delta * _broadcast(Sb / S, delta)
    M2 = Psi = Lbar = None
    if a.M2 is not None:
        cross = delta.unsqueeze(-1) * delta.unsqueeze(-2) if outer else delta.square()
        M2 = (a.M2 * _broadcast(fa, a.M2) + b.M2 * _broadcast(fb, b.M2)
              + cross * _broadcast(Sa * Sb / S, cross))
    if a.Psi is not None:
        Psi = a.Psi * _broadcast(fa, a.Psi) + b.Psi * _broadcast(fb, b.Psi)
    if a.Lbar is not None:
        Lbar = a.Lbar * _broadcast(fa, a.Lbar) + b.Lbar * _broadcast(fb, b.Lbar)
    return _TileStats(m, S, mean, M2, Psi, Lbar)


@ieee_fp32_matmul()
def prototype_exact_posterior(
    xi:         torch.Tensor,              # (..., K) working means
    C:          torch.Tensor,              # (..., K), (..., K, K) or (..., G, b, b) working covariance
    nu:         torch.Tensor,              # (M, K) template means
    log_lambda: torch.Tensor,              # (M, K) template log variances
    log_pi:     torch.Tensor,              # (M,) template log prior weights (unnormalized)

    *,
    need_covariance: bool = True,
    need_precision:  bool = False,
    tile_bytes:      int  = PROTOTYPE_TILE_BYTES,
) -> ExactPrototypePosterior:
    r"""Exact posterior of the prototype-agent model (Proposition ma:prop:minimal (a), (d) with :math:`S = T = I`).

    Evaluates the module-docstring formulas for every token without materializing a pair tensor of more
    than ``tile_bytes / 2`` (a tile forms several such tensors; see :data:`PROTOTYPE_TILE_BYTES`): rows
    (tokens) are independent and are tiled, and the templates of a row tile are processed in chunks
    merged by an online log-sum-exp accumulation in ONE pass
    (:func:`_merge_statistics`), so each pair is factorized once in the forward. With gradients on and
    more than one tile, each tile runs under a non-reentrant ``torch.utils.checkpoint`` (it draws no random
    numbers) and is recomputed in backward. Block and dense layouts factor every pair with zero jitter and
    no repair; a failure is collected on the device and raised as one ``ValueError`` after the last tile.

    ``need_covariance`` adds :math:`C^+` and :math:`R`; ``need_precision`` adds :math:`\bar\Lambda`
    (diagnostics and tests; :math:`C^+` does not read it). Covariance-shaped outputs use the input layout:
    the diagonal, the diagonal blocks, or the dense matrix.
    """
    layout = covariance_layout(xi, C)
    if nu.dim() != 2 or nu.shape[-1] != xi.shape[-1] or log_lambda.shape != nu.shape \
            or log_pi.shape != nu.shape[:1]:
        raise ValueError("templates must be nu (M, K), log_lambda (M, K) and log_pi (M,) with the mean's K")
    if tile_bytes < 1:
        raise ValueError(f"tile_bytes must be a positive byte count, got {tile_bytes!r}")
    K, M = xi.shape[-1], nu.shape[0]
    lead = tuple(xi.shape[:-1])
    rows = math.prod(lead)
    if rows == 0:
        raise ValueError("the exact prototype posterior needs at least one token")
    dtype = _compute_dtype(xi, C, nu, log_lambda, log_pi)
    blocks = layout != "diagonal"
    with torch.autocast(device_type=xi.device.type, enabled=False):
        x = xi.to(dtype).reshape(rows, K)                                    # (R, K)
        n, lam, lp = nu.to(dtype), log_lambda.to(dtype).exp(), log_pi.to(dtype)
        if blocks:
            Cb = (C.to(dtype).unsqueeze(-3) if layout == "dense" else C.to(dtype))   # (..., G, b, b)
            G, b = Cb.shape[-3], Cb.shape[-1]
            Cw = Cb.reshape(rows, G, b, b)                                   # (R, G, b, b)
            xw, nw, lw = x.reshape(rows, G, b), n.reshape(M, G, b), lam.reshape(M, G, b)
            tile_fn, elements = _exact_block_tile, G * b * b                 # elements of one pair matrix set
        else:
            Cw = C.to(dtype).reshape(rows, K)                                # (R, K)
            xw, nw, lw = x, n, lam
            tile_fn, elements = _exact_diagonal_tile, K
        pairs = max(1, int(tile_bytes) // ((torch.finfo(dtype).bits // 8) * 2 * elements))
        m_step = min(M, pairs)
        r_step = max(1, min(rows, pairs // m_step))
        tiled = rows > r_step or M > m_step
        failures: List[torch.Tensor] = []
        logit_rows, u_rows, M2_rows, Psi_rows, Lbar_rows, s_rows = [], [], [], [], [], []
        for i in range(0, rows, r_step):
            acc: Optional[_TileStats] = None
            logit_chunks = []
            for j in range(0, M, m_step):
                args = (xw[i:i + r_step], Cw[i:i + r_step], nw[j:j + m_step], lw[j:j + m_step],
                        lp[j:j + m_step])
                if tiled and torch.is_grad_enabled() and any(t.requires_grad for t in args):
                    logits, stats, failed = checkpoint(
                        tile_fn, *args, need_covariance, need_precision,
                        use_reentrant=False, preserve_rng_state=False)
                else:
                    logits, stats, failed = tile_fn(*args, need_covariance, need_precision)
                logit_chunks.append(logits)
                failures.append(failed)
                acc = stats if acc is None else _merge_statistics(acc, stats, outer=blocks)
            logit_rows.append(torch.cat(logit_chunks, dim=-1))               # (r, M)
            u_rows.append(acc.mean)
            s_rows.append(acc.s)
            M2_rows.append(acc.M2)
            Psi_rows.append(acc.Psi)
            Lbar_rows.append(acc.Lbar)
        # One device-side reduction and one host check for every tile.
        if bool(torch.stack(failures).any()):
            raise ValueError(
                "prototype-agent record covariance C_i + Lambda_a is not positive definite: zero-jitter "
                "Cholesky failed; no repair applied" if blocks else
                "prototype-agent record variances c_i + lambda_a must be positive and finite")
        s = torch.cat(s_rows)                                                # (R,)
        u = torch.cat(u_rows)                                                # (R, K) or (R, G, b)
        weights = torch.softmax(torch.cat(logit_rows), dim=-1)               # (R, M)
        delta = (Cw @ u.unsqueeze(-1)).squeeze(-1) if blocks else Cw * u     # (R, ...) C u
        C_plus = R = Lambda_bar = None
        if need_covariance:
            M2, Psi = torch.cat(M2_rows), torch.cat(Psi_rows)
            R, Psi = M2 / _broadcast(s, M2), Psi / _broadcast(s, Psi)        # normalized by sum_a e^{l - m}
            if blocks:
                R = _symmetric(R)
                C_plus = _symmetric(Cw @ Psi + Cw @ R @ Cw)                  # (R, G, b, b)
            else:
                C_plus = Cw * Psi + Cw.square() * R                         # (R, K)
        if need_precision:
            Lbar = torch.cat(Lbar_rows)
            Lambda_bar = Lbar / _broadcast(s, Lbar)
            if blocks:
                Lambda_bar = _symmetric(Lambda_bar)

        def out_layout(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if t is None:
                return None
            if not blocks:
                return t.reshape(*lead, K)
            t = t.reshape(*lead, *t.shape[1:])                               # (..., G, b, b)
            return t.squeeze(-3) if layout == "dense" else t

        delta = delta.reshape(*lead, K)
        return ExactPrototypePosterior(
            xi_plus=x.reshape(*lead, K) + delta,
            delta_xi=delta,
            C_plus=out_layout(C_plus),
            weights=weights.reshape(*lead, M),
            u=u.reshape(*lead, K),
            Lambda_bar=out_layout(Lambda_bar),
            R=out_layout(R),
        )


@ieee_fp32_matmul()
def _exact_em_tile(
    xi:         torch.Tensor,                       # (r, K) or (r, G, b), detached
    C:          torch.Tensor,                       # (r, K) or (r, G, b, b), detached
    weights:    torch.Tensor,                       # (r, m) exact weights w_ia of this chunk, detached
    nu:         torch.Tensor,                       # (m, K) or (m, G, b)
    log_lambda: torch.Tensor,                       # (m, K) or (m, G, b)
    log_pi_hat: torch.Tensor,                       # (m,) normalized log prior
    blocks:     bool,
) -> torch.Tensor:                                  # (r,) sum over the chunk of w_ia [...]
    r"""One (rows x templates) tile of :func:`prototype_exact_em_objective`.

    The component moments are formed without gradients at the current templates,
    :math:`m_{ia} = \xi_i + C_iW_{ia}^{-1}(\nu_a - \xi_i)` and :math:`p_{ia} = \operatorname{diag}(C_iW_{ia}^{-1}\Lambda_a)`
    (elementwise :math:`c\lambda/(c + \lambda)` for a diagonal :math:`C_i`); the bracket is then evaluated at the live
    :math:`(\nu, \log\lambda, \log\hat\pi)`.
    """
    with torch.no_grad():
        lam_old, nu_old = log_lambda.detach().exp(), nu.detach()
        if blocks:
            W = C.unsqueeze(1) + torch.diag_embed(lam_old)                    # (r, m, G, b, b) C + Lambda_a
            L, _ = torch.linalg.cholesky_ex(W)                                # factored by the posterior already
            solved = torch.cholesky_solve((nu_old - xi.unsqueeze(1)).unsqueeze(-1), L)   # (r, m, G, b, 1)
            m_ia = xi.unsqueeze(1) + (C.unsqueeze(1) @ solved).squeeze(-1)    # (r, m, G, b)
            p = (C.unsqueeze(1) * torch.cholesky_inverse(L)).sum(dim=-1) * lam_old   # (r, m, G, b) diag(C W^-1 Lambda)
        else:
            W = C.unsqueeze(-2) + lam_old                                     # (r, m, K) c + lambda
            m_ia = xi.unsqueeze(-2) + C.unsqueeze(-2) * (nu_old - xi.unsqueeze(-2)) / W   # (r, m, K)
            p = C.unsqueeze(-2) * lam_old / W                                 # (r, m, K) c lambda / (c + lambda)
    axes = tuple(range(2, m_ia.dim()))
    quadratic = (((m_ia - nu).square() + p) * torch.exp(-log_lambda)).sum(dim=axes)       # (r, m)
    log_norm = (math.log(2.0 * math.pi) + log_lambda).sum(dim=tuple(range(1, log_lambda.dim())))   # (m,)
    bracket = log_pi_hat - 0.5 * (quadratic + log_norm)                       # (r, m)
    return (weights * bracket).sum(dim=-1)


def prototype_exact_em_objective(
    xi:         torch.Tensor,                       # (..., K) working means (detached)
    C:          torch.Tensor,                       # (..., K), (..., K, K) or (..., G, b, b) working covariance (detached)
    weights:    torch.Tensor,                       # (..., M) exact template weights w_ia (detached)
    nu:         torch.Tensor,                       # (M, K) template means
    log_lambda: torch.Tensor,                       # (M, K) template log variances
    log_pi_hat: torch.Tensor,                       # (M,) normalized log prior log softmax(log pi)
    *,
    tile_bytes: int = PROTOTYPE_TILE_BYTES,
) -> torch.Tensor:                                  # (...) Q_i
    r"""EM objective of the exact posterior: expectations under each template's component posterior.

    For the model :math:`a \sim \hat\pi`, :math:`X \mid a \sim \mathcal N(\nu_a, \Lambda_a)`,
    :math:`\xi_i \mid X \sim \mathcal N(X, C_i)` (evidence :math:`\sum_a\hat\pi_a\mathcal N(\nu_a;\xi_i, W_{ia})`,
    :math:`W_{ia} = C_i + \Lambda_a`), the exact posterior is :math:`\sum_aw_{ia}\mathcal N(m_{ia}, P_{ia})` (module
    docstring) and

    .. math::
        Q_i(\theta) = \sum_aw_{ia}\Bigl[\log\hat\pi_a - \tfrac12\sum_k\Bigl(\log 2\pi\lambda_{ak}
            + \frac{(m_{iak} - \nu_{ak})^2 + (P_{ia})_{kk}}{\lambda_{ak}}\Bigr)\Bigr],

    with :math:`w`, :math:`m` and :math:`P` taken at the current templates and held fixed. It is the expected
    complete-data log-likelihood up to the :math:`\theta`-independent :math:`E_q\log\mathcal N(\xi_i;X, C_i)`, so by
    Fisher's identity :math:`\nabla_\theta Q_i = \nabla_\theta\log p(\xi_i \mid \theta)` at the current templates. Rows and
    templates are tiled as in :func:`prototype_exact_posterior` (each template chunk adds its share, since the weights
    are already normalized); with gradients on and more than one tile, each tile runs under a non-reentrant
    checkpoint. Autocast off, at float64 when an input is float64 and float32 otherwise.
    """
    layout = covariance_layout(xi, C)
    K, M = xi.shape[-1], nu.shape[0]
    lead = tuple(xi.shape[:-1])
    rows = math.prod(lead)
    dtype = _compute_dtype(xi, C, weights, nu, log_lambda, log_pi_hat)
    blocks = layout != "diagonal"
    with torch.autocast(device_type=xi.device.type, enabled=False):
        x = xi.to(dtype).reshape(rows, K)                                    # (R, K)
        w = weights.to(dtype).reshape(rows, M)                               # (R, M)
        n, ll, lp = nu.to(dtype), log_lambda.to(dtype), log_pi_hat.to(dtype)
        if blocks:
            Cb = (C.to(dtype).unsqueeze(-3) if layout == "dense" else C.to(dtype))   # (..., G, b, b)
            G, b = Cb.shape[-3], Cb.shape[-1]
            Cw, xw = Cb.reshape(rows, G, b, b), x.reshape(rows, G, b)
            nw, lw, elements = n.reshape(M, G, b), ll.reshape(M, G, b), G * b * b
        else:
            Cw, xw, nw, lw, elements = C.to(dtype).reshape(rows, K), x, n, ll, K
        pairs = max(1, int(tile_bytes) // ((torch.finfo(dtype).bits // 8) * 2 * elements))
        m_step = min(M, pairs)
        r_step = max(1, min(rows, pairs // m_step))
        tiled = rows > r_step or M > m_step
        out_rows = []
        for i in range(0, rows, r_step):
            total: Optional[torch.Tensor] = None
            for j in range(0, M, m_step):
                args = (xw[i:i + r_step], Cw[i:i + r_step], w[i:i + r_step, j:j + m_step], nw[j:j + m_step],
                        lw[j:j + m_step], lp[j:j + m_step])
                if tiled and torch.is_grad_enabled() and any(t.requires_grad for t in args):
                    part = checkpoint(_exact_em_tile, *args, blocks, use_reentrant=False, preserve_rng_state=False)
                else:
                    part = _exact_em_tile(*args, blocks)
                total = part if total is None else total + part
            out_rows.append(total)
        return torch.cat(out_rows).reshape(lead)


@ieee_fp32_matmul()
def prototype_mean_field_posterior(
    xi:         torch.Tensor,              # (..., K) working means
    C:          torch.Tensor,              # (..., K), (..., K, K) or (..., G, b, b) working covariance
    nu:         torch.Tensor,              # (M, K) template means
    log_lambda: torch.Tensor,              # (M, K) template log variances
    log_pi:     torch.Tensor,              # (M,) template log prior weights (unnormalized)

    *,
    need_covariance: bool = True,
    cavi_steps:      int  = 1,
) -> MeanFieldPrototypePosterior:
    r"""Structured-mean-field posterior of the prototype-agent model (Proposition ma:prop:mean_field), by
    ``cavi_steps`` coordinate-ascent sweeps; one sweep is the first step started at the token's prior.

    The label update at the token's prior is ``softmax`` of :func:`template_kl_logits`; the power
    posterior at those labels has precision :math:`C_i^{-1} + \operatorname{diag}D_i` and is evaluated
    without inverting :math:`C_i`. With :math:`D^{1/2} = \operatorname{diag}(D_i)^{1/2}` and
    :math:`B_i = I + D^{1/2}C_iD^{1/2}` (eigenvalues at least one),

    .. math::
        P_i = (C_i^{-1} + \operatorname{diag}D_i)^{-1} = D^{-1/2}B_i^{-1}D^{1/2}C_i,\qquad
        \xi_i^+ - \xi_i = P_i(\eta_i - D_i\odot\xi_i),

    a product with no difference of large matrices (the push-through identity
    :math:`(I + D^{1/2}CD^{1/2})^{-1}D^{1/2} = D^{1/2}(I + CD)^{-1}` gives
    :math:`D^{-1/2}B^{-1}D^{1/2}C = (I + CD)^{-1}C = (C^{-1} + D)^{-1}`). For a diagonal :math:`C_i` this is
    :math:`P_{ik} = c_{ik}/(1 + c_{ik}D_{ik})`. :math:`D_i > 0` because the responsibilities sum to one and
    every :math:`\lambda_a` is finite. The covariance omits the between-template spread.

    Coordinate ascent. The label step is the exact CAVI update of :math:`q(a_i)` given :math:`q(X_i) = \mathcal N(m, S)`:
    the record of template :math:`a` has the likelihood :math:`\mathcal N(X_i;\nu_a,\Lambda_a)`, and

    .. math::
        E_q[\log\mathcal N(X_i;\nu_a,\Lambda_a)] = -\tfrac12\sum_k\Bigl[\log 2\pi\lambda_{ak}
            + \frac{(m_k - \nu_{ak})^2 + S_{kk}}{\lambda_{ak}}\Bigr]
            = -D_{\mathrm{KL}}(\mathcal N(m, S)\|p_a) - \tfrac K2(1 + \log 2\pi) - \tfrac12\log\det S,

    whose last two terms do not depend on :math:`a` and cancel in the softmax, so
    :math:`q(a_i) = \operatorname{softmax}_a(\log\pi_a - D_{\mathrm{KL}}(\mathcal N(m, S)\|p_a))` and only
    :math:`\operatorname{diag}S` enters. Sweep 1 takes :math:`(m, S) = (\xi_i, C_i)`; sweep :math:`s > 1` takes the
    previous sweep's :math:`(\xi_i^+, C_i^+)` and forms the power posterior from the prior moments
    :math:`(\xi_i, C_i)` with the new responsibilities. Intermediate sweeps form :math:`C_i^+` whatever
    ``need_covariance`` says; the returned responsibilities, :math:`D_i` and :math:`\eta_i` are the last sweep's.
    The sweeps are unrolled under autograd, and the factorizations of every sweep are checked on the host once.
    """
    layout = covariance_layout(xi, C)
    if nu.dim() != 2 or nu.shape[-1] != xi.shape[-1] or log_lambda.shape != nu.shape \
            or log_pi.shape != nu.shape[:1]:
        raise ValueError("templates must be nu (M, K), log_lambda (M, K) and log_pi (M,) with the mean's K")
    if isinstance(cavi_steps, bool) or not isinstance(cavi_steps, int) or cavi_steps < 1:
        raise ValueError(f"cavi_steps must be an int >= 1, got {cavi_steps!r}")
    K = xi.shape[-1]
    lead = tuple(xi.shape[:-1])
    dtype = _compute_dtype(xi, C, nu, log_lambda, log_pi)
    with torch.autocast(device_type=xi.device.type, enabled=False):
        x, Cw = xi.to(dtype), C.to(dtype)
        n, ll, lp = nu.to(dtype), log_lambda.to(dtype), log_pi.to(dtype)
        x_lab, c_lab = x, covariance_diagonal(Cw, layout)                    # (..., K): moments the labels read
        failed: Optional[torch.Tensor] = None
        for sweep in range(cavi_steps):
            last = sweep == cavi_steps - 1
            logits = template_kl_logits(x_lab, c_lab, n, ll, lp)             # (..., M)
            responsibilities = torch.softmax(logits, dim=-1)                 # (..., M)
            precision = torch.exp(-ll)                                       # (M, K) 1 / lambda_a
            D = responsibilities @ precision                                 # (..., K)
            eta = responsibilities @ (n * precision)                         # (..., K)
            g = eta - D * x                                                  # (..., K) sum_a r_a (nu_a - xi)/lambda_a
            C_plus = None
            if layout == "diagonal":
                P = Cw / (1.0 + Cw * D)                                      # (..., K)
                delta = P * g
                C_plus = P
            else:
                Cb = Cw.unsqueeze(-3) if layout == "dense" else Cw           # (..., G, b, b)
                G, b = Cb.shape[-3], Cb.shape[-1]
                Dh = D.sqrt().reshape(*lead, G, b)                           # (..., G, b) D^{1/2}
                eye = torch.eye(b, dtype=dtype, device=Cb.device)
                B = Dh.unsqueeze(-1) * Cb * Dh.unsqueeze(-2) + eye           # (..., G, b, b) I + D^{1/2} C D^{1/2}
                L, info = torch.linalg.cholesky_ex(B, check_errors=False)
                Cg = (Cb @ g.reshape(*lead, G, b).unsqueeze(-1)).squeeze(-1)  # (..., G, b) C g
                delta = (torch.cholesky_solve((Dh * Cg).unsqueeze(-1), L).squeeze(-1) / Dh).reshape(*lead, K)
                if need_covariance or not last:
                    P = torch.cholesky_solve(Dh.unsqueeze(-1) * Cb, L) / Dh.unsqueeze(-1)   # D^{-1/2} B^{-1} D^{1/2} C
                    P = _symmetric(P)
                    C_plus = P.squeeze(-3) if layout == "dense" else P
                failed = (info != 0).any() if failed is None else failed | (info != 0).any()
            if not last:
                x_lab, c_lab = x + delta, covariance_diagonal(C_plus, layout)   # the next labels read (xi^+, diag C^+)
        if failed is not None and bool(failed):
            raise ValueError("prototype-agent power-posterior matrix I + D^{1/2} C D^{1/2} is not positive "
                             "definite: the working covariance is not positive semidefinite")
        return MeanFieldPrototypePosterior(
            xi_plus=x + delta, delta_xi=delta, C_plus=C_plus if need_covariance else None,
            responsibilities=responsibilities, D=D, eta=eta)
