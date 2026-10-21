r"""Model, probit vocabulary tree and assumed-density filter of the local-learner test.

Generative model (all factors normalized):

.. math::
    z_0\sim\mathcal N(0,I),\qquad z_t\mid z_{t-1},y_{t-1}\sim\mathcal N(Gz_{t-1}+e_{y_{t-1}},Q),\qquad
    P(y_t=v\mid z_t)=\prod_{n\in\operatorname{path}(v)}\Phi\bigl(d_{nv}(a_n^\top z_t+b_n)\bigr).

The filter forecasts :math:`q_t^-=\mathcal N(Gm_{t-1}+e_{y_{t-1}},GP_{t-1}G^\top+Q)`, scores :math:`y_t` by
conditioning sequentially on its path (probit moment matching), and keeps the result as :math:`q_t^+`.
The first token of a stream is assimilated from the prior and not scored.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Callable, Optional

import torch

Tensor = torch.Tensor
LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


@dataclass
class Params:
    r"""Persistent parameters :math:`\theta=(G,E,Q,A,b)`."""
    G: Tensor          # (d, d) transition
    E: Tensor          # (V, d) previous-token channel means e_v
    Q: Tensor          # (d, d) SPD transition noise
    A: Tensor          # (V-1, d) node utility weights a_n
    b: Tensor          # (V-1,) node utility offsets b_n

    def clone(self) -> "Params":
        return Params(**{f.name: getattr(self, f.name).detach().clone() for f in fields(self)})

    def to(self, device: torch.device | str) -> "Params":
        return Params(**{f.name: getattr(self, f.name).detach().to(device) for f in fields(self)})

    @property
    def d(self) -> int:
        return self.G.shape[0]


class TreeSensor:
    r"""Balanced binary vocabulary tree of depth D over V = 2^D tokens (the bits of the token, MSB first)."""

    def __init__(self, depth: int, device: torch.device | str = "cpu") -> None:
        self.depth = depth
        self.V = 2 ** depth
        v = torch.arange(self.V)
        nodes = torch.empty(self.V, depth, dtype=torch.long)
        dirs = torch.empty(self.V, depth, dtype=torch.float64)
        n = torch.ones(self.V, dtype=torch.long)                       # heap index, root = 1
        for k in range(depth):
            bit = (v >> (depth - 1 - k)) & 1
            nodes[:, k] = n - 1
            dirs[:, k] = 2.0 * bit - 1.0
            n = 2 * n + bit
        self.nodes = nodes.to(device)                                  # (V, D) node ids in [0, V-1)
        self.dirs = dirs.to(device)                                    # (V, D) branch signs d_nv


def init_params(d: int, V: int, seed: int, device: torch.device | str = "cpu") -> Params:
    r"""Small random initialization: :math:`G=0.5I`, :math:`Q=0.1I`, Gaussian E and A, zero b (drawn on
    the CPU, so a seed gives the same start on every device)."""
    g = torch.Generator().manual_seed(seed)
    return Params(G=0.5 * torch.eye(d, dtype=torch.float64),
                  E=0.1 * torch.randn(V, d, generator=g, dtype=torch.float64),
                  Q=0.1 * torch.eye(d, dtype=torch.float64),
                  A=torch.randn(V - 1, d, generator=g, dtype=torch.float64) / math.sqrt(d),
                  b=torch.zeros(V - 1, dtype=torch.float64)).to(device)


def forecast(m: Tensor, P: Tensor, y_prev: Tensor, prm: Params) -> tuple[Tensor, Tensor]:
    r"""Forecast moments :math:`(Gm+e_{y_{t-1}},\,GPG^\top+Q)`; m (B, d), P (B, d, d), y_prev (B,)."""
    mm = m @ prm.G.T + prm.E[y_prev]                                   # (B, d)
    Pm = prm.G @ P @ prm.G.T + prm.Q                                   # (B, d, d)
    return mm, 0.5 * (Pm + Pm.transpose(-1, -2))


# collect(n, s, m, P, Pa, s2, sig, mu_u, kappa, lam) receives the node-step quantities (sensor statistics).
NodeHook = Callable[..., None]


def assimilate(m: Tensor, P: Tensor, y: Tensor, prm: Params, tree: TreeSensor,
               collect: Optional[NodeHook] = None) -> tuple[Tensor, Tensor, Tensor]:
    r"""Condition N(m, P) on token y along its path; returns (m+, P+, log forecast probability).

    Per node, with :math:`\sigma^2=1+a^\top Pa`, :math:`\kappa=d(a^\top m+b)/\sigma`,
    :math:`\lambda=\phi(\kappa)/\Phi(\kappa)`:
    :math:`m\leftarrow m+(d\lambda/\sigma)Pa`, :math:`P\leftarrow P-\lambda(\lambda+\kappa)Paa^\top P/\sigma^2`.
    The returned log probability is :math:`\sum_n\log\Phi(\kappa_n)`, a normalized row over tokens.
    """
    logp = torch.zeros(m.shape[0], dtype=m.dtype, device=m.device)
    nodes, dirs = tree.nodes[y], tree.dirs[y]                         # (B, D)
    for k in range(tree.depth):
        n, s = nodes[:, k], dirs[:, k]
        a, bb = prm.A[n], prm.b[n]                                     # (B, d), (B,)
        Pa = (P @ a.unsqueeze(-1)).squeeze(-1)                         # (B, d)
        s2 = 1.0 + (a * Pa).sum(-1)                                    # (B,)
        sig = s2.sqrt()
        mu_u = (a * m).sum(-1) + bb
        kappa = s * mu_u / sig
        log_Phi = torch.special.log_ndtr(kappa)
        lam = torch.exp(-0.5 * kappa ** 2 - LOG_SQRT_2PI - log_Phi)
        if collect is not None:
            collect(n, s, m, P, Pa, s2, sig, mu_u, kappa, lam)
        m = m + (s * lam / sig).unsqueeze(-1) * Pa
        P = P - (lam * (lam + kappa) / s2)[:, None, None] * (Pa.unsqueeze(-1) * Pa.unsqueeze(-2))
        logp = logp + log_Phi
    return m, 0.5 * (P + P.transpose(-1, -2)), logp


def start_state(y0: Tensor, prm: Params, tree: TreeSensor) -> tuple[Tensor, Tensor]:
    r"""Filtered state at t = 0: the prior N(0, I) conditioned on the unscored first token."""
    B, d = y0.shape[0], prm.d
    m = torch.zeros(B, d, dtype=torch.float64, device=prm.G.device)
    P = torch.eye(d, dtype=torch.float64, device=prm.G.device).expand(B, d, d).clone()
    m, P, _ = assimilate(m, P, y0, prm, tree)
    return m, P


@torch.no_grad()
def evaluate_filter(prm: Params, streams: Tensor, tree: TreeSensor) -> tuple[float, int]:
    r"""Summed forecast NLL (nats) and count over streams (S, L); position 0 of each stream is context only."""
    m, P = start_state(streams[:, 0], prm, tree)
    total = torch.zeros((), dtype=torch.float64, device=m.device)        # no host read per step
    for t in range(1, streams.shape[1]):
        mm, Pm = forecast(m, P, streams[:, t - 1], prm)
        m, P, logp = assimilate(mm, Pm, streams[:, t], prm, tree)
        total -= logp.sum()
    return float(total), streams.shape[0] * (streams.shape[1] - 1)


@torch.no_grad()
def sample_sequences(prm: Params, tree: TreeSensor, n_streams: int, length: int, seed: int) -> Tensor:
    r"""Draw (n_streams, length) token streams from the generative model (on the CPU; returns a CPU tensor)."""
    prm, tree = prm.to("cpu"), TreeSensor(tree.depth)
    g = torch.Generator().manual_seed(seed)
    d = prm.d
    cholQ = torch.linalg.cholesky(prm.Q)
    z = torch.randn(n_streams, d, generator=g, dtype=torch.float64)
    out = torch.empty(n_streams, length, dtype=torch.long)
    for t in range(length):
        if t > 0:
            noise = torch.randn(n_streams, d, generator=g, dtype=torch.float64) @ cholQ.T
            z = z @ prm.G.T + prm.E[out[:, t - 1]] + noise
        n = torch.ones(n_streams, dtype=torch.long)
        for _ in range(tree.depth):
            u = (prm.A[n - 1] * z).sum(-1) + prm.b[n - 1] + torch.randn(n_streams, generator=g, dtype=torch.float64)
            n = 2 * n + (u > 0).long()
        out[:, t] = n - tree.V
    return out
