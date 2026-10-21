r"""Learning rules of the local-learner test: stepwise EM on posterior statistics, and backpropagation.

Local arm. Sensor node n is a probit regression of its latent utility :math:`u_n\sim\mathcal N(a_n^\top z+b_n,1)`
truncated to the observed branch (Albert and Chib 1993). At the node's filter state N(m, P), with
:math:`\mu_u=a^\top m+b`, :math:`\sigma^2=1+a^\top Pa`, :math:`w=Pa/\sigma^2`, :math:`c=m-w\mu_u`,

.. math::
    E[u]=\mu_u+d\sigma\lambda,\quad E[u^2]=\sigma^2(1-\lambda(\lambda+\kappa))+E[u]^2,\quad
    E[z]=c+wE[u],\quad E[uz]=cE[u]+wE[u^2],

    E[zz^\top]=P-\sigma^2ww^\top+cc^\top+(cw^\top+wc^\top)E[u]+ww^\top E[u^2],

and :math:`(a_n,b_n)` is the ridge least-squares solve from their running averages. The transition
:math:`[G\ E^\top]` is the ridge regression of :math:`z_t` on :math:`x_t=[z_{t-1};\mathrm{onehot}(y_{t-1})]`
from Rauch-Tung-Striebel smoothed pair moments over blocks of ``window`` steps, and Q is the expected
residual covariance. Averages follow :math:`s\leftarrow(1-\gamma_k)s+\gamma_k\hat s`,
:math:`\gamma_k=(k+1)^{-0.6}` (Cappe and Moulines 2009). No derivative of the filter is taken.

Backprop arm. Adam on the forecast NLL of the same filter, autograd through it, truncated at ``chunk``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Optional

import torch

from vfe3.local_learner.core import Params, Tensor, TreeSensor, assimilate, forecast, start_state

Logger = Callable[[dict], None]


@dataclass
class Stats:
    r"""Per-token averaged sufficient statistics."""
    node_zz: Tensor    # (V-1, d+1, d+1) E[z~ z~^T] per node visit
    node_uz: Tensor    # (V-1, d+1)      E[u z~]
    zz: Tensor         # (d, d)  E[z_t z_t^T]
    zp: Tensor         # (d, d)  E[z_t z_{t-1}^T]
    pp: Tensor         # (d, d)  E[z_{t-1} z_{t-1}^T]
    yp: Tensor         # (V, d)  sum over y_{t-1} = v of E[z_{t-1}]
    yz: Tensor         # (V, d)  sum over y_{t-1} = v of E[z_t]
    cnt: Tensor        # (V,)    frequency of y_{t-1} = v

    @staticmethod
    def zeros(d: int, V: int, device: torch.device | str = "cpu") -> "Stats":
        z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=device)   # noqa: E731
        return Stats(z(V - 1, d + 1, d + 1), z(V - 1, d + 1), z(d, d), z(d, d), z(d, d), z(V, d), z(V, d), z(V))

    def scaled(self, c: float) -> "Stats":
        return Stats(*(getattr(self, f) * c for f in self.__dataclass_fields__))

    def blend(self, other: "Stats", gamma: float) -> "Stats":
        return Stats(*((1 - gamma) * getattr(self, f) + gamma * getattr(other, f)
                       for f in self.__dataclass_fields__))


def node_moments(s: Tensor, m: Tensor, P: Tensor, Pa: Tensor, s2: Tensor, sig: Tensor, mu_u: Tensor,
                 kappa: Tensor, lam: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Truncated-utility moments (E[z], E[zz^T], E[u], E[uz]) at one node; batch (B,)."""
    Eu = mu_u + s * sig * lam                                          # (B,)
    Eu2 = s2 * (1.0 - lam * (lam + kappa)) + Eu ** 2
    w = Pa / s2.unsqueeze(-1)                                          # (B, d)
    c = m - w * mu_u.unsqueeze(-1)
    outer = lambda x, y: x.unsqueeze(-1) * y.unsqueeze(-2)             # noqa: E731
    Ez = c + w * Eu.unsqueeze(-1)
    Ezz = (P - s2[:, None, None] * outer(w, w) + outer(c, c)
           + (outer(c, w) + outer(w, c)) * Eu[:, None, None] + outer(w, w) * Eu2[:, None, None])
    Euz = c * Eu.unsqueeze(-1) + w * Eu2.unsqueeze(-1)
    return Ez, Ezz, Eu, Euz


class SensorCollector:
    r"""Accumulates node statistics :math:`E[\tilde z\tilde z^\top]`, :math:`E[u\tilde z]` with :math:`\tilde z=[z;1]`."""

    def __init__(self, stats: Stats) -> None:
        self.st = stats

    def __call__(self, n: Tensor, s: Tensor, m: Tensor, P: Tensor, Pa: Tensor, s2: Tensor, sig: Tensor,
                 mu_u: Tensor, kappa: Tensor, lam: Tensor) -> None:
        Ez, Ezz, Eu, Euz = node_moments(s, m, P, Pa, s2, sig, mu_u, kappa, lam)
        B, d = Ez.shape
        blk = torch.empty(B, d + 1, d + 1, dtype=torch.float64, device=Ez.device)
        blk[:, :d, :d] = Ezz
        blk[:, :d, d] = Ez
        blk[:, d, :d] = Ez
        blk[:, d, d] = 1.0
        self.st.node_zz.index_add_(0, n, blk)
        self.st.node_uz.index_add_(0, n, torch.cat([Euz, Eu.unsqueeze(-1)], -1))


def accumulate_transition(st: Stats, G: Tensor, m_prev: Tensor, P_prev: Tensor, mf: list[Tensor],
                          Pf: list[Tensor], mm: list[Tensor], Pm: list[Tensor], y_prev: list[Tensor]) -> None:
    r"""RTS-smooth one block (horizon at its last step) and add its pair moments to ``st``.

    For pair (s-1, s): :math:`J=P_{s-1}G^\top(P_s^-)^{-1}`,
    :math:`\hat m_{s-1}=m_{s-1}+J(\hat m_s-m_s^-)`, :math:`\hat P_{s-1}=P_{s-1}+J(\hat P_s-P_s^-)J^\top`,
    :math:`\operatorname{Cov}(z_s,z_{s-1})=\hat P_sJ^\top`.
    """
    outer = lambda x, y: x.unsqueeze(-1) * y.unsqueeze(-2)             # noqa: E731
    ms, Ps = mf[-1], Pf[-1]
    for s in reversed(range(len(mf))):
        mp, Pp = (mf[s - 1], Pf[s - 1]) if s > 0 else (m_prev, P_prev)
        L, _ = torch.linalg.cholesky_ex(Pm[s])                         # no host sync; Pm >= Q is SPD
        J = torch.cholesky_solve(G @ Pp, L).transpose(-1, -2)          # (B, d, d) P_{s-1} G^T (P_s^-)^{-1}
        ms_p = mp + (J @ (ms - mm[s]).unsqueeze(-1)).squeeze(-1)
        Ps_p = Pp + J @ (Ps - Pm[s]) @ J.transpose(-1, -2)
        C = Ps @ J.transpose(-1, -2)                                   # Cov(z_s, z_{s-1})
        st.zz += (Ps + outer(ms, ms)).sum(0)
        st.zp += (C + outer(ms, ms_p)).sum(0)
        st.pp += (Ps_p + outer(ms_p, ms_p)).sum(0)
        st.yp.index_add_(0, y_prev[s], ms_p)
        st.yz.index_add_(0, y_prev[s], ms)
        st.cnt.index_add_(0, y_prev[s], torch.ones_like(y_prev[s], dtype=torch.float64))
        ms, Ps = ms_p, Ps_p


def m_step(st: Stats, prm: Params, ridge: float, q_floor: float, learn_transition: bool = True) -> Params:
    r"""Ridge least-squares solves for every node and for :math:`[G\ E^\top]`; Q from the residual moments."""
    d, V = prm.d, prm.E.shape[0]
    dev = prm.G.device
    eye1 = torch.eye(d + 1, dtype=torch.float64, device=dev)
    ab = torch.linalg.solve(st.node_zz + ridge * eye1, st.node_uz.unsqueeze(-1)).squeeze(-1)   # (V-1, d+1)
    new = prm.clone()
    new.A, new.b = ab[:, :d].contiguous(), ab[:, d].contiguous()
    if not learn_transition:
        return new
    Sxx = torch.zeros(d + V, d + V, dtype=torch.float64, device=dev)
    Sxx[:d, :d], Sxx[:d, d:], Sxx[d:, :d] = st.pp, st.yp.T, st.yp
    Sxx[d:, d:] = torch.diag(st.cnt)
    Szx = torch.cat([st.zp, st.yz.T], 1)                               # (d, d+V)
    W = torch.linalg.solve(Sxx + ridge * torch.eye(d + V, dtype=torch.float64, device=dev), Szx.T).T
    new.G, new.E = W[:, :d].contiguous(), W[:, d:].T.contiguous()
    Q = st.zz - W @ Szx.T
    Q = 0.5 * (Q + Q.T)
    evals, evecs = torch.linalg.eigh(Q)
    new.Q = (evecs * evals.clamp_min(q_floor)) @ evecs.T
    return new


def train_local(prm0: Params, streams: Tensor, tree: TreeSensor, *, chunk: int = 128, window: int = 1,
                blind: bool = False, ridge: float = 1e-6, q_floor: float = 1e-4,
                log: Optional[Logger] = None) -> Params:
    r"""Stepwise variational EM over token streams (B, L). ``blind`` uses the forecast in place of the
    assimilated state in the transition statistics (the target-blind control; requires window = 1)."""
    if blind and window != 1:
        raise ValueError("the target-blind control is defined for window = 1")
    if chunk % window:
        raise ValueError("window must divide chunk")
    prm = prm0.clone()
    d, V = prm.d, prm.E.shape[0]
    B, Ltot = streams.shape
    with torch.no_grad():
        m, P = start_state(streams[:, 0], prm, tree)
        running: Optional[Stats] = None
        dev = prm.G.device
        k, t0, ntok = 0, time.perf_counter(), 0
        nll = torch.zeros((), dtype=torch.float64, device=dev)          # read on the host only when logging
        for c0 in range(1, Ltot, chunk):
            steps = range(c0, min(c0 + chunk, Ltot))
            st = Stats.zeros(d, V, dev)
            hook = SensorCollector(st)
            mf, Pf, mm_l, Pm_l, yp_l = [], [], [], [], []
            m_block, P_block = m, P
            for i, t in enumerate(steps):
                mm, Pm = forecast(m, P, streams[:, t - 1], prm)
                m, P, logp = assimilate(mm, Pm, streams[:, t], prm, tree, collect=hook)
                nll -= logp.sum()
                ntok += B
                post_m, post_P = (mm, Pm) if blind else (m, P)
                mf.append(post_m); Pf.append(post_P); mm_l.append(mm); Pm_l.append(Pm); yp_l.append(streams[:, t - 1])
                if len(mf) == window or i == len(steps) - 1:
                    accumulate_transition(st, prm.G, m_block, P_block, mf, Pf, mm_l, Pm_l, yp_l)
                    m_block, P_block = m, P                     # the state the next forecast is built from
                    mf, Pf, mm_l, Pm_l, yp_l = [], [], [], [], []
            st = st.scaled(1.0 / (B * len(steps)))
            gamma = (k + 1) ** -0.6
            running = st if running is None else running.blend(st, gamma)
            prm = m_step(running, prm, ridge, q_floor)
            k += 1
            if log is not None and (k % 20 == 0 or c0 + chunk >= Ltot):
                log({"arm": "local", "chunk": k, "train_bits": float(nll) / ntok / math.log(2),
                     "G_spectral": float(torch.linalg.matrix_norm(prm.G, 2)), "sec": time.perf_counter() - t0})
                nll, ntok = torch.zeros_like(nll), 0
    return prm


def train_backprop(prm0: Params, streams: Tensor, tree: TreeSensor, *, chunk: int = 128, lr: float = 3e-3,
                   clip: float = 1.0, log: Optional[Logger] = None) -> Params:
    r"""Adam on the filter's forecast NLL, backpropagated through the filter (truncated at ``chunk``)."""
    d = prm0.d
    G = prm0.G.clone().requires_grad_(True)
    E = prm0.E.clone().requires_grad_(True)
    Lq = torch.linalg.cholesky(prm0.Q - 1e-4 * torch.eye(d, dtype=torch.float64, device=prm0.G.device)).requires_grad_(True)
    A = prm0.A.clone().requires_grad_(True)
    b = prm0.b.clone().requires_grad_(True)
    opt = torch.optim.Adam([G, E, Lq, A, b], lr=lr)
    eye = torch.eye(d, dtype=torch.float64, device=prm0.G.device)

    def current() -> Params:
        Lt = torch.tril(Lq)
        return Params(G=G, E=E, Q=Lt @ Lt.T + 1e-4 * eye, A=A, b=b)

    B, Ltot = streams.shape
    with torch.no_grad():
        m, P = start_state(streams[:, 0], current(), tree)
    k, t0, ntok = 0, time.perf_counter(), 0
    nll = torch.zeros((), dtype=torch.float64, device=eye.device)
    for c0 in range(1, Ltot, chunk):
        prm = current()
        m, P = m.detach(), P.detach()
        loss = torch.zeros((), dtype=torch.float64, device=eye.device)
        steps = range(c0, min(c0 + chunk, Ltot))
        for t in steps:
            mm, Pm = forecast(m, P, streams[:, t - 1], prm)
            m, P, logp = assimilate(mm, Pm, streams[:, t], prm, tree)
            loss = loss - logp.sum()
        n = B * len(steps)
        opt.zero_grad()
        (loss / n).backward()
        torch.nn.utils.clip_grad_norm_([G, E, Lq, A, b], clip)
        opt.step()
        nll += loss.detach()
        ntok += n
        k += 1
        if log is not None and (k % 20 == 0 or c0 + chunk >= Ltot):
            log({"arm": "backprop", "chunk": k, "train_bits": float(nll) / ntok / math.log(2),
                 "G_spectral": float(torch.linalg.matrix_norm(G.detach(), 2)), "sec": time.perf_counter() - t0})
            nll, ntok = torch.zeros_like(nll), 0
    with torch.no_grad():
        out = current()
        return Params(G=out.G.detach().clone(), E=out.E.detach().clone(), Q=out.Q.detach().clone(),
                      A=out.A.detach().clone(), b=out.b.detach().clone())
