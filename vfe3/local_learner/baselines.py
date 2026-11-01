r"""References for the local-learner test: add-k n-gram models and a bootstrap particle filter."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch

from vfe3.local_learner.core import Params, Tensor, TreeSensor


def _ngram_tables(train: np.ndarray, V: int, order: int) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    r"""Per order :math:`o \le` ``order``: the flat counts of every :math:`o`-gram, ``(V**o,)``, indexed by
    :math:`\sum_j y_j V^{o-1-j}`, and the totals of every context, ``(V**(o-1),)`` (one total for :math:`o = 1`)."""
    tables = {}
    for o in range(1, order + 1):
        idx = np.zeros(len(train) - o + 1, dtype=np.int64)
        for j in range(o):
            idx = idx * V + train[j:len(train) - o + 1 + j]
        counts = np.bincount(idx, minlength=V ** o).astype(np.float64)                 # (V**o,)
        tables[o] = (counts, counts.reshape(-1, V).sum(axis=1))                        # (V**(o-1),) context totals
    return tables


def ngram_nll_grid(train: np.ndarray, streams: np.ndarray, V: int, order: int,
                   ks: Sequence[float]) -> list[tuple[float, int]]:
    r"""``[ngram_nll(train, streams, V, order, k) for k in ks]``, building the tables once.

    Each scored position's target count :math:`c` and context total :math:`n` are looked up once; every constant
    then sums :math:`-\log\frac{c + k}{n + kV}` over the positions in the same order as :func:`ngram_nll`, so the
    sums equal the per-constant ones bit for bit (counts are integers, so the totals are exact in any order).
    """
    train = train.astype(np.int64)
    rows = streams.astype(np.int64)                                                    # (S, L)
    S, L = rows.shape
    tables = _ngram_tables(train, V, order)
    hits = np.zeros((S, max(L - 1, 0)))                                                # (S, L-1) target counts
    totals = np.zeros((S, max(L - 1, 0)))                                              # (S, L-1) context totals
    for t in range(1, L):
        o = min(order, t + 1)                                       # context truncated at the stream start
        idx = np.zeros(S, dtype=np.int64)
        for j in range(o):
            idx = idx * V + rows[:, t - o + 1 + j]
        counts, context_totals = tables[o]
        hits[:, t - 1] = counts[idx]
        totals[:, t - 1] = context_totals[idx // V]                 # o = 1: idx // V = 0, the unigram total
    hits, totals = hits.reshape(-1), totals.reshape(-1)             # stream by stream, position by position
    out = []
    for k in ks:
        total = 0.0
        for p in ((hits + k) / (totals + k * V)).tolist():
            total -= math.log(p)
        out.append((total, int(hits.size)))
    return out


def ngram_nll(train: np.ndarray, streams: np.ndarray, V: int, order: int, k: float) -> tuple[float, int]:
    r"""Summed add-k NLL (nats) of an order-``order`` model on streams (S, L), scoring positions >= 1 of each
    stream with context truncated at the stream start (a missing context symbol is replaced by a lower order).
    Several constants on one training set: :func:`ngram_nll_grid`."""
    return ngram_nll_grid(train, streams, V, order, [k])[0]


@torch.no_grad()
def particle_filter_nll(prm: Params, streams: Tensor, tree: TreeSensor, n_particles: int, seed: int) -> tuple[Tensor, int]:
    r"""Bootstrap particle estimate of the exact model's summed predictive NLL on streams (S, L), per run
    ((R,) on the host).

    :math:`\hat p(y_t\mid y_{<t})=\frac1N\sum_iP(y_t\mid z_t^{(i)})` with :math:`z_t^{(i)}` propagated from
    resampled particles; position 0 conditions the prior particles and is not scored.
    """
    dev = prm.G.device
    g = torch.Generator(device=dev).manual_seed(seed)
    S, L = streams.shape
    R, d = prm.R, prm.d
    cholQt = torch.linalg.cholesky(prm.Q).transpose(-1, -2).unsqueeze(1)   # (R, 1, d, d)
    Gt = prm.G.transpose(-1, -2).unsqueeze(1)                              # (R, 1, d, d)

    def loglik(z: Tensor, y: Tensor) -> Tensor:                       # z (R, S, N, d), y (S,) -> (R, S, N)
        nodes, dirs = tree.nodes[y], tree.dirs[y]                     # (S, D)
        a = prm.A[:, nodes]                                           # (R, S, D, d)
        util = torch.einsum("rsnd,rskd->rsnk", z, a) + prm.b[:, nodes].unsqueeze(2)
        return torch.special.log_ndtr(util * dirs.unsqueeze(1)).sum(-1)

    def resample(z: Tensor, logw: Tensor) -> Tensor:
        w = torch.softmax(logw, -1).reshape(R * S, n_particles)
        idx = torch.multinomial(w, n_particles, replacement=True, generator=g).reshape(R, S, n_particles)
        return torch.gather(z, 2, idx.unsqueeze(-1).expand(-1, -1, -1, d))

    z = torch.randn(R, S, n_particles, d, generator=g, dtype=torch.float64, device=dev)
    z = resample(z, loglik(z, streams[:, 0]))
    total = torch.zeros(R, dtype=torch.float64, device=dev)           # no host read per step
    for t in range(1, L):
        noise = torch.randn(R, S, n_particles, d, generator=g, dtype=torch.float64, device=dev) @ cholQt
        z = z @ Gt + prm.E[:, streams[:, t - 1]].unsqueeze(2) + noise
        ll = loglik(z, streams[:, t])
        total -= (torch.logsumexp(ll, -1) - math.log(n_particles)).sum(-1)
        z = resample(z, ll)
    return total.cpu(), S * (L - 1)
