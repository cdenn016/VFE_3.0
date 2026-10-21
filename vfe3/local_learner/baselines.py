r"""References for the local-learner test: add-k n-gram models and a bootstrap particle filter."""

from __future__ import annotations

import math

import numpy as np
import torch

from vfe3.local_learner.core import Params, Tensor, TreeSensor


def ngram_nll(train: np.ndarray, streams: np.ndarray, V: int, order: int, k: float) -> tuple[float, int]:
    r"""Summed add-k NLL (nats) of an order-``order`` model on streams (S, L), scoring positions >= 1 of each
    stream with context truncated at the stream start (a missing context symbol is replaced by a lower order)."""
    train = train.astype(np.int64)
    tables = {}
    for o in range(1, order + 1):
        idx = np.zeros(len(train) - o + 1, dtype=np.int64)
        for j in range(o):
            idx = idx * V + train[j:len(train) - o + 1 + j]
        tables[o] = np.bincount(idx, minlength=V ** o).reshape((V,) * o).astype(np.float64)
    total, count = 0.0, 0
    for row in streams.astype(np.int64):
        for t in range(1, len(row)):
            o = min(order, t + 1)
            ctx = tuple(row[t - o + 1:t])
            counts = tables[o][ctx] if o > 1 else tables[1]
            total -= math.log((counts[row[t]] + k) / (counts.sum() + k * V))
            count += 1
    return total, count


@torch.no_grad()
def particle_filter_nll(prm: Params, streams: Tensor, tree: TreeSensor, n_particles: int, seed: int) -> tuple[float, int]:
    r"""Bootstrap particle estimate of the exact model's summed predictive NLL on streams (S, L).

    :math:`\hat p(y_t\mid y_{<t})=\frac1N\sum_iP(y_t\mid z_t^{(i)})` with :math:`z_t^{(i)}` propagated from
    resampled particles; position 0 conditions the prior particles and is not scored.
    """
    dev = prm.G.device
    g = torch.Generator(device=dev).manual_seed(seed)
    S, L = streams.shape
    d = prm.d
    cholQ = torch.linalg.cholesky(prm.Q)

    def loglik(z: Tensor, y: Tensor) -> Tensor:                       # z (S, N, d), y (S,) -> (S, N)
        nodes, dirs = tree.nodes[y], tree.dirs[y]                     # (S, D)
        a = prm.A[nodes]                                              # (S, D, d)
        util = torch.einsum("snd,skd->snk", z, a) + prm.b[nodes].unsqueeze(1)
        return torch.special.log_ndtr(util * dirs.unsqueeze(1)).sum(-1)

    def resample(z: Tensor, logw: Tensor) -> Tensor:
        w = torch.softmax(logw, -1)
        idx = torch.multinomial(w, n_particles, replacement=True, generator=g)
        return torch.gather(z, 1, idx.unsqueeze(-1).expand(-1, -1, d))

    z = torch.randn(S, n_particles, d, generator=g, dtype=torch.float64, device=dev)
    z = resample(z, loglik(z, streams[:, 0]))
    total = torch.zeros((), dtype=torch.float64, device=dev)             # no host read per step
    for t in range(1, L):
        noise = torch.randn(S, n_particles, d, generator=g, dtype=torch.float64, device=dev) @ cholQ.T
        z = z @ prm.G.T + prm.E[streams[:, t - 1]].unsqueeze(1) + noise
        ll = loglik(z, streams[:, t])
        total -= (torch.logsumexp(ll, -1) - math.log(n_particles)).sum()
        z = resample(z, ll)
    return float(total), S * (L - 1)
