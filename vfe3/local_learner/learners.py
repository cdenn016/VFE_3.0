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
:math:`\gamma_k=\max((k+1)^{-\alpha},\gamma_{\min})` (Cappe and Moulines 2009; default
:math:`\alpha=0.6`, :math:`\gamma_{\min}=0`), with k optionally restarted every ``reset_every`` updates
(:math:`\gamma=1` then replaces the averages). No derivative of the filter is taken.

Backprop arm. Adam on the forecast NLL of the same filter, autograd through it, truncated at ``chunk``.

All statistics and parameters carry the leading run axis R of :class:`~vfe3.local_learner.core.Params`;
runs never exchange information (Adam is elementwise and gradient clipping is per run).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Optional

import torch

from vfe3.local_learner.core import LOG_SQRT_2PI, Params, Tensor, TreeSensor, assimilate, forecast, start_state

Logger = Callable[[dict], None]


def _outer(x: Tensor, y: Tensor) -> Tensor:
    return x.unsqueeze(-1) * y.unsqueeze(-2)


@dataclass
class Stats:
    r"""Per-token averaged sufficient statistics of R runs."""
    node_zz: Tensor    # (R, V-1, d+1, d+1) E[z~ z~^T] per node visit
    node_uz: Tensor    # (R, V-1, d+1)      E[u z~]
    zz: Tensor         # (R, d, d)  E[z_t z_t^T]
    zp: Tensor         # (R, d, d)  E[z_t z_{t-1}^T]
    pp: Tensor         # (R, d, d)  E[z_{t-1} z_{t-1}^T]
    yp: Tensor         # (R, V, d)  sum over y_{t-1} = v of E[z_{t-1}]
    yz: Tensor         # (R, V, d)  sum over y_{t-1} = v of E[z_t]
    cnt: Tensor        # (R, V)     frequency of y_{t-1} = v
    node_uu: Tensor    # (R, V-1)   E[u^2] per node visit (the residual utility variance of parameter expansion)

    @staticmethod
    def zeros(R: int, d: int, V: int, device: torch.device | str = "cpu") -> "Stats":
        z = lambda *s: torch.zeros(R, *s, dtype=torch.float64, device=device)   # noqa: E731
        return Stats(z(V - 1, d + 1, d + 1), z(V - 1, d + 1), z(d, d), z(d, d), z(d, d), z(V, d), z(V, d), z(V),
                     z(V - 1))

    def scaled(self, c: float) -> "Stats":
        return Stats(*(getattr(self, f) * c for f in self.__dataclass_fields__))

    def blend(self, other: "Stats", gamma: float) -> "Stats":
        return Stats(*((1 - gamma) * getattr(self, f) + gamma * getattr(other, f)
                       for f in self.__dataclass_fields__))


def node_moments(s: Tensor, m: Tensor, P: Tensor, Pa: Tensor, s2: Tensor, sig: Tensor, mu_u: Tensor,
                 kappa: Tensor, lam: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Truncated-utility moments (E[z], E[zz^T], E[u], E[uz]) at one node; any leading batch shape."""
    Eu = mu_u + s * sig * lam
    Eu2 = s2 * (1.0 - lam * (lam + kappa)) + Eu ** 2
    w = Pa / s2.unsqueeze(-1)                                          # (..., d)
    c = m - w * mu_u.unsqueeze(-1)
    Ez = c + w * Eu.unsqueeze(-1)
    Ezz = (P - s2[..., None, None] * _outer(w, w) + _outer(c, c)
           + (_outer(c, w) + _outer(w, c)) * Eu[..., None, None] + _outer(w, w) * Eu2[..., None, None])
    Euz = c * Eu.unsqueeze(-1) + w * Eu2.unsqueeze(-1)
    return Ez, Ezz, Eu, Euz


class SensorCollector:
    r"""Accumulates node statistics :math:`E[\tilde z\tilde z^\top]`, :math:`E[u\tilde z]` with :math:`\tilde z=[z;1]`."""

    def __init__(self, stats: Stats) -> None:
        self.st = stats

    def __call__(self, rows: Tensor, n: Tensor, s: Tensor, m: Tensor, P: Tensor, Pa: Tensor, s2: Tensor,
                 sig: Tensor, mu_u: Tensor, kappa: Tensor, lam: Tensor) -> None:
        Ez, Ezz, Eu, Euz = node_moments(s, m, P, Pa, s2, sig, mu_u, kappa, lam)
        R, B, d = Ez.shape
        nodes = self.st.node_uz.shape[1]
        blk = torch.empty(R, B, d + 1, d + 1, dtype=torch.float64, device=Ez.device)
        blk[..., :d, :d] = Ezz
        blk[..., :d, d] = Ez
        blk[..., d, :d] = Ez
        blk[..., d, d] = 1.0
        idx = (rows * nodes + n).reshape(-1)                           # (R*B,) flat (run, node)
        self.st.node_zz.view(R * nodes, d + 1, d + 1).index_add_(0, idx, blk.reshape(R * B, d + 1, d + 1))
        uz = torch.cat([Euz, Eu.unsqueeze(-1)], -1).reshape(R * B, d + 1)
        self.st.node_uz.view(R * nodes, d + 1).index_add_(0, idx, uz)
        Eu2 = s2 * (1.0 - lam * (lam + kappa)) + Eu ** 2               # E[u^2] at the node (R, B)
        self.st.node_uu.view(R * nodes).index_add_(0, idx, Eu2.reshape(-1))


class CavitySensorCollector:
    r"""Node statistics at the expectation-propagation cavity: the token's assimilated posterior with that node's
    own Gaussian site removed, instead of the sequential filter state that has seen only the earlier nodes.

    During the path pass each node's moment-matching update defines a Gaussian site in the projected utility
    mean :math:`f=a^\top z+b`. With projected cavity mean :math:`\mu_u` and variance :math:`v=a^\top Pa` at the
    node step, the updated projected moments are :math:`\hat v=v-v^2\lambda(\lambda+\kappa)/\sigma^2` and
    :math:`\hat\mu=\mu_u+d\lambda v/\sigma`, so the site has precision
    :math:`\tilde\tau=\hat v^{-1}-v^{-1}=c/(1+v(1-c))` with :math:`c=\lambda(\lambda+\kappa)`, and linear coefficient
    :math:`\tilde\nu=\hat\mu/\hat v-\mu_u/v=d\lambda/\sigma+\tilde\tau\hat\mu`; the right-hand forms avoid the
    cancelling differences. After the pass, with :math:`w=P^+a` and
    :math:`s^+=a^\top P^+a`, removing the site is the rank-one correction

    .. math::
        P_{\setminus n}=P^++\frac{\tilde\tau}{1-\tilde\tau s^+}ww^\top,\qquad
        m_{\setminus n}=m^++\frac{w}{1-\tilde\tau s^+}\bigl(\tilde\tau(a^\top m^++b)-\tilde\nu\bigr),

    and the truncated-utility moments of :func:`node_moments` are taken there (Minka 2001; Rasmussen and
    Williams 2006, Sec. 3.6). For the last node of the path the cavity is the sequential state itself.
    """

    def __init__(self, stats: Stats) -> None:
        self.inner = SensorCollector(stats)
        self.sites: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []

    def __call__(self, rows: Tensor, n: Tensor, s: Tensor, m: Tensor, P: Tensor, Pa: Tensor, s2: Tensor,
                 sig: Tensor, mu_u: Tensor, kappa: Tensor, lam: Tensor) -> None:
        v = s2 - 1.0                                                   # (R, B) projected variance a^T P a
        c = lam * (lam + kappa)                                        # in (0, 1)
        # tau = 1/v_hat - 1/v and nu = mu_hat/v_hat - mu_u/v without the cancelling differences:
        # 1/v_hat = 1/v + tau gives tau = c / (1 + v(1 - c)) and nu = d lambda / sigma + tau mu_hat.
        tau = c / (1.0 + v * (1.0 - c))
        mu_hat = mu_u + s * lam * v / sig
        nu = s * lam / sig + tau * mu_hat
        self.sites.append((n, s, tau, nu))

    def finish(self, rows: Tensor, m_post: Tensor, P_post: Tensor, prm: Params) -> None:
        r"""Accumulate every recorded node at its cavity under the token's final posterior (m+, P+)."""
        for n, s, tau, nu in self.sites:
            a, bb = prm.A[:, n], prm.b[:, n]                           # (R, B, d), (R, B)
            w = (P_post @ a.unsqueeze(-1)).squeeze(-1)                 # (R, B, d)
            s_plus = (a * w).sum(-1)
            denom = (1.0 - tau * s_plus).clamp_min(1e-12)              # = 1/(1 + tau v_cav) > 0 in exact arithmetic
            m_cav = m_post + (w / denom.unsqueeze(-1)) * (tau * ((a * m_post).sum(-1) + bb) - nu).unsqueeze(-1)
            P_cav = P_post + (tau / denom)[..., None, None] * _outer(w, w)
            P_cav = 0.5 * (P_cav + P_cav.transpose(-1, -2))
            Pa = (P_cav @ a.unsqueeze(-1)).squeeze(-1)
            s2 = 1.0 + (a * Pa).sum(-1)
            sig = s2.sqrt()
            mu_u = (a * m_cav).sum(-1) + bb
            kappa = s * mu_u / sig
            log_Phi = torch.special.log_ndtr(kappa)
            lam = torch.exp(-0.5 * kappa ** 2 - LOG_SQRT_2PI - log_Phi)
            self.inner(rows, n, s, m_cav, P_cav, Pa, s2, sig, mu_u, kappa, lam)
        self.sites = []


def accumulate_transition(st: Stats, G: Tensor, m_prev: Tensor, P_prev: Tensor, mf: list[Tensor],
                          Pf: list[Tensor], mm: list[Tensor], Pm: list[Tensor], y_prev: list[Tensor]) -> None:
    r"""RTS-smooth one block (horizon at its last step) and add its pair moments to ``st``.

    For pair (s-1, s): :math:`J=P_{s-1}G^\top(P_s^-)^{-1}`,
    :math:`\hat m_{s-1}=m_{s-1}+J(\hat m_s-m_s^-)`, :math:`\hat P_{s-1}=P_{s-1}+J(\hat P_s-P_s^-)J^\top`,
    :math:`\operatorname{Cov}(z_s,z_{s-1})=\hat P_sJ^\top`. States are (R, B, ...); G is (R, d, d).
    """
    R, V, d = st.yp.shape
    rows = torch.arange(R, device=G.device)[:, None]
    Gb = G.unsqueeze(1)                                                # (R, 1, d, d)
    ms, Ps = mf[-1], Pf[-1]
    for s in reversed(range(len(mf))):
        mp, Pp = (mf[s - 1], Pf[s - 1]) if s > 0 else (m_prev, P_prev)
        L, _ = torch.linalg.cholesky_ex(Pm[s])                         # no host sync; Pm >= Q is SPD
        J = torch.cholesky_solve(Gb @ Pp, L).transpose(-1, -2)         # (R, B, d, d) P_{s-1} G^T (P_s^-)^{-1}
        ms_p = mp + (J @ (ms - mm[s]).unsqueeze(-1)).squeeze(-1)
        Ps_p = Pp + J @ (Ps - Pm[s]) @ J.transpose(-1, -2)
        C = Ps @ J.transpose(-1, -2)                                   # Cov(z_s, z_{s-1})
        st.zz += (Ps + _outer(ms, ms)).sum(1)
        st.zp += (C + _outer(ms, ms_p)).sum(1)
        st.pp += (Ps_p + _outer(ms_p, ms_p)).sum(1)
        idx = (rows * V + y_prev[s]).reshape(-1)                       # (R*B,) flat (run, token)
        st.yp.view(R * V, d).index_add_(0, idx, ms_p.reshape(-1, d))
        st.yz.view(R * V, d).index_add_(0, idx, ms.reshape(-1, d))
        st.cnt.view(R * V).index_add_(0, idx, torch.ones(idx.shape[0], dtype=torch.float64, device=G.device))
        ms, Ps = ms_p, Ps_p


def residual_covariance(Szz: Tensor, Szx: Tensor, Sxx: Tensor, W: Tensor) -> Tensor:
    r"""Expected residual covariance of the regression z ~ W x at a given W,

    .. math:: E[(z - Wx)(z - Wx)^\top] = S_{zz} - W S_{xz} - S_{zx} W^\top + W S_{xx} W^\top,

    with :math:`S_{xz} = S_{zx}^\top`. The shortcut :math:`S_{zz} - W S_{zx}^\top` holds only at the
    unregularized solve; at the ridge solve :math:`W = S_{zx}(S_{xx} + \rho I)^{-1}` it exceeds this by
    :math:`\rho W W^\top`. Shapes: Szz (..., d, d), Szx (..., d, n), Sxx (..., n, n), W (..., d, n)."""
    WSxz = W @ Szx.transpose(-1, -2)                                   # (..., d, d)
    return Szz - WSxz - WSxz.transpose(-1, -2) + W @ Sxx @ W.transpose(-1, -2)


def m_step(st: Stats, prm: Params, ridge: float, q_floor: float, learn_transition: bool = True,
           expand_sensor: bool = False) -> Params:
    r"""Ridge least-squares solves for every node and for :math:`[G\ E^\top]`; Q from the residual moments.

    ``expand_sensor`` applies parameter-expanded EM to each probit node (Liu, Rubin and Wu 1998). The expanded
    model lets the utility have a free variance, :math:`u_n\sim\mathcal N(\tilde a_n^\top\tilde z,\sigma_n^2)`,
    which leaves the branch law unchanged up to the scale of :math:`\tilde a_n`. Its M-step is the same least
    squares for :math:`\tilde a_n` plus :math:`\sigma_n^2=E[(u-\tilde a_n^\top\tilde z)^2]` per visit, and the
    reduction to the unit-variance model is :math:`\tilde a_n\leftarrow\tilde a_n/\sigma_n`. Nodes without
    visits keep :math:`\sigma_n=1`."""
    d, V, R = prm.d, prm.E.shape[1], prm.R
    dev = prm.G.device
    eye1 = torch.eye(d + 1, dtype=torch.float64, device=dev)
    # Both systems are second-moment matrices plus a ridge, hence SPD: Cholesky on contiguous operands (a batched
    # LU solve with a transposed right-hand side fails inside MKL's DLASWP on CPU at d+V = 264).
    Ln = torch.linalg.cholesky(st.node_zz + ridge * eye1)
    ab = torch.cholesky_solve(st.node_uz.unsqueeze(-1).contiguous(), Ln).squeeze(-1)   # (R, V-1, d+1)
    if expand_sensor:
        visits = st.node_zz[..., d, d]                                 # (R, V-1) visit frequency
        resid = (st.node_uu - 2.0 * (ab * st.node_uz).sum(-1)
                 + torch.einsum("rni,rnij,rnj->rn", ab, st.node_zz, ab))
        sigma2 = torch.where(visits > 1e-12, resid / visits.clamp_min(1e-12), torch.ones_like(visits))
        ab = ab / sigma2.clamp_min(1e-6).sqrt().unsqueeze(-1)
    new = prm.clone()
    new.A, new.b = ab[..., :d].contiguous(), ab[..., d].contiguous()
    if not learn_transition:
        return new
    Sxx = torch.zeros(R, d + V, d + V, dtype=torch.float64, device=dev)
    Sxx[:, :d, :d], Sxx[:, :d, d:], Sxx[:, d:, :d] = st.pp, st.yp.transpose(-1, -2), st.yp
    Sxx[:, d:, d:] = torch.diag_embed(st.cnt)
    Szx = torch.cat([st.zp, st.yz.transpose(-1, -2)], -1)              # (R, d, d+V)
    Lx = torch.linalg.cholesky(Sxx + ridge * torch.eye(d + V, dtype=torch.float64, device=dev))
    W = torch.cholesky_solve(Szx.transpose(-1, -2).contiguous(), Lx).transpose(-1, -2)   # (R, d, d+V)
    new.G, new.E = W[..., :d].contiguous(), W[..., d:].transpose(-1, -2).contiguous()
    Q = residual_covariance(st.zz, Szx, Sxx, W)
    Q = 0.5 * (Q + Q.transpose(-1, -2))
    evals, evecs = torch.linalg.eigh(Q)
    new.Q = (evecs * evals.clamp_min(q_floor).unsqueeze(-2)) @ evecs.transpose(-1, -2)
    return new


def spd_logm(G: Tensor) -> Tensor:
    r"""Principal logarithm of a symmetric positive-definite matrix batch (the starting point of the exponential
    link chart; the stage B initialization :math:`G=0.5I` qualifies). Raises on any other input."""
    sym = 0.5 * (G + G.transpose(-1, -2))
    evals, evecs = torch.linalg.eigh(sym)
    if float((G - sym).abs().max()) > 1e-12 or float(evals.min()) <= 0.0:
        raise ValueError("the exponential link chart starts from a symmetric positive-definite G")
    return (evecs * evals.log().unsqueeze(-2)) @ evecs.transpose(-1, -2)


def dexp_matrix(L: Tensor) -> Tensor:
    r"""Matrix of the Frechet derivative :math:`D\exp_L[E]=\int_0^1e^{sL}Ee^{(1-s)L}\,ds` on row-major vec,
    from the upper-right block of :math:`\exp\begin{pmatrix}L&E\\0&L\end{pmatrix}` for each basis matrix E
    (Higham 2008, Sec. 3.2). L (..., d, d) -> (..., d^2, d^2) with column k = vec(Dexp_L[E_k])."""
    *lead, d, _ = L.shape
    n = d * d
    basis = torch.eye(n, dtype=L.dtype, device=L.device).reshape(n, d, d)
    X = torch.zeros(*lead, n, 2 * d, 2 * d, dtype=L.dtype, device=L.device)
    X[..., :d, :d] = L.unsqueeze(-3)
    X[..., d:, d:] = L.unsqueeze(-3)
    X[..., :d, d:] = basis
    D = torch.linalg.matrix_exp(X)[..., :d, d:]                         # (..., n, d, d)
    return D.reshape(*lead, n, n).transpose(-1, -2)


def exp_link_transition(st: Stats, prm: Params, L: Tensor, ridge: float, q_floor: float, step: float,
                        trust: float = 1.0, halvings: int = 10) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Natural-gradient update of the exponential link :math:`G=e^L` at held statistics, then E and Q in
    closed form at the new G.

    Profiling E out of the expected transition energy leaves a quadratic in G whose Gauss-Newton (Fisher) metric
    is :math:`Q^{-1}\otimes\tilde S`. Pulled back through exp (the metric after equation (5) of the October 20
    local-learning note), the natural-gradient direction in L solves :math:`D\exp_L[\delta L]=G_{\rm EM}-G`,
    where :math:`G_{\rm EM}` is the unconstrained regression optimum; ``step`` scales it (1 = full step, exact to
    first order). The step is exact only to first order, so a far-off :math:`G_{\rm EM}` can make
    :math:`e^{L+\delta L}` overshoot: the step is capped at Frobenius norm ``trust`` and then halved, per run, up to
    ``halvings`` times until the profiled energy :math:`\tfrac12\operatorname{tr}(Q^{-1}R(G))` (R the expected
    residual covariance with E refit at G, Q the current noise) is finite and lower than at the current G; a run
    with no such step keeps its L. Returns (L, G, E, Q)."""
    d, V, R = prm.d, prm.E.shape[1], prm.R
    dev = prm.G.device
    Sxx = torch.zeros(R, d + V, d + V, dtype=torch.float64, device=dev)
    Sxx[:, :d, :d], Sxx[:, :d, d:], Sxx[:, d:, :d] = st.pp, st.yp.transpose(-1, -2), st.yp
    Sxx[:, d:, d:] = torch.diag_embed(st.cnt)
    Szx = torch.cat([st.zp, st.yz.transpose(-1, -2)], -1)               # (R, d, d+V)
    Lx = torch.linalg.cholesky(Sxx + ridge * torch.eye(d + V, dtype=torch.float64, device=dev))
    W = torch.cholesky_solve(Szx.transpose(-1, -2).contiguous(), Lx).transpose(-1, -2)
    dG = W[..., :d] - prm.G                                             # (R, d, d)
    # Householder QR, not LU: the batched LU solve has failed inside MKL's pivoting (DLASWP) on CPU here, as it did
    # for the transition solve before that switched to Cholesky; Dexp_L is not symmetric, so QR is used instead.
    Qm, Rm = torch.linalg.qr(dexp_matrix(L).contiguous())
    rhs = Qm.transpose(-1, -2) @ (step * dG).reshape(R, d * d, 1)
    dL = torch.linalg.solve_triangular(Rm, rhs, upper=True).reshape(R, d, d)
    dL = torch.nan_to_num(dL, nan=0.0, posinf=0.0, neginf=0.0)          # a singular Dexp yields no step
    norm = torch.linalg.matrix_norm(dL).clamp_min(1e-300)
    dL = dL * (trust / norm).clamp(max=1.0)[:, None, None]
    Qinv = torch.linalg.inv(prm.Q)

    def profiled(G: Tensor) -> tuple[Tensor, Tensor, Tensor]:          # G (..., R, d, d)
        E = (st.yz - st.yp @ G.transpose(-1, -2)) / (st.cnt + ridge).unsqueeze(-1)       # (..., R, V, d)
        Wg = torch.cat([G, E.transpose(-1, -2)], -1)
        Rc = residual_covariance(st.zz, Szx, Sxx, Wg)
        return 0.5 * (Qinv * Rc).sum((-1, -2)), E, Rc

    h0, _, _ = profiled(prm.G)                                          # (R,)
    scales = 0.5 ** torch.arange(halvings + 1, dtype=torch.float64, device=dev)          # (S,)
    L_try = L + scales[:, None, None, None] * dL                        # (S, R, d, d)
    G_try = torch.linalg.matrix_exp(L_try)
    h_try, _, _ = profiled(G_try)                                       # (S, R)
    ok = torch.isfinite(h_try) & (h_try < h0) & torch.isfinite(G_try).all(-1).all(-1)
    first = torch.where(ok.any(0), ok.float().argmax(0), torch.full_like(h0, halvings + 1, dtype=torch.long))
    take = (first <= halvings)                                          # (R,) runs with an accepted step
    idx = first.clamp(max=halvings)
    L_new = torch.where(take[:, None, None], L_try[idx, torch.arange(R, device=dev)], L)
    G_new = torch.linalg.matrix_exp(L_new)
    _, E_new, Q = profiled(G_new)
    Q = 0.5 * (Q + Q.transpose(-1, -2))
    evals, evecs = torch.linalg.eigh(Q)
    Q = (evecs * evals.clamp_min(q_floor).unsqueeze(-2)) @ evecs.transpose(-1, -2)
    return L_new, G_new, E_new.contiguous(), Q


def train_local(prm0: Params, streams: Tensor, tree: TreeSensor, *, chunk: int = 128, window: int = 1,
                blind: bool = False, ridge: float = 1e-6, q_floor: float = 1e-4,
                gamma_exponent: float = 0.6, gamma_floor: float = 0.0, reset_every: Optional[int] = None,
                sensor_stats: str = "sequential", expand_sensor: bool = False, overrelax: float = 1.0,
                link_chart: str = "matrix", link_step: float = 1.0, log: Optional[Logger] = None,
                log_every: int = 20) -> Params:
    r"""Stepwise variational EM over token streams (B, L), for all runs of ``prm0`` at once. ``blind`` uses the
    forecast in place of the assimilated state in the transition statistics (the target-blind control;
    requires window = 1). The averaging step is :math:`\max((k+1)^{-\alpha},\gamma_{\min})` with
    :math:`\alpha` = ``gamma_exponent``, :math:`\gamma_{\min}` = ``gamma_floor``, and k counted from the
    last restart when ``reset_every`` is set. ``expand_sensor`` uses the parameter-expanded probit M-step of
    :func:`m_step`. ``overrelax`` :math:`\omega\in(0,2)` replaces each M-step result :math:`\theta_{\rm EM}` by
    :math:`\theta+\omega(\theta_{\rm EM}-\theta)` for G, E, A, b and Q (Q's eigenvalues then floored at
    ``q_floor``); :math:`\omega=1` is plain EM and :math:`\omega>1` overrelaxes (Salakhutdinov and Roweis 2003).
    ``link_chart="exp"`` keeps the transition as :math:`G=e^L` and updates L by the natural-gradient step of
    :func:`exp_link_transition` (scaled by ``link_step``), starting from the principal log of ``prm0.G``."""
    if not 0.0 < overrelax < 2.0:
        raise ValueError("overrelax must lie in (0, 2)")
    if link_chart not in ("matrix", "exp"):
        raise ValueError(f"link_chart must be 'matrix' or 'exp', got {link_chart!r}")
    if link_chart == "exp" and (overrelax != 1.0 or blind or not 0.0 < link_step <= 1.0):
        raise ValueError("the exponential link chart takes 0 < link_step <= 1, no overrelaxation, no blind arm")
    if not 0.0 < gamma_exponent <= 1.0 or not 0.0 <= gamma_floor <= 1.0:
        raise ValueError("gamma_exponent must lie in (0, 1] and gamma_floor in [0, 1]")
    if reset_every is not None and reset_every < 1:
        raise ValueError("reset_every must be a positive number of updates")
    if sensor_stats not in ("sequential", "cavity"):
        raise ValueError(f"sensor_stats must be 'sequential' or 'cavity', got {sensor_stats!r}")
    if blind and window != 1:
        raise ValueError("the target-blind control is defined for window = 1")
    if chunk % window:
        raise ValueError("window must divide chunk")
    prm = prm0.clone()
    d, V, R = prm.d, prm.E.shape[1], prm.R
    B, Ltot = streams.shape
    with torch.no_grad():
        L = spd_logm(prm.G) if link_chart == "exp" else None
        m, P = start_state(streams[:, 0], prm, tree)
        running: Optional[Stats] = None
        dev = prm.G.device
        k, t0, ntok = 0, time.perf_counter(), 0
        nll = torch.zeros(R, dtype=torch.float64, device=dev)          # read on the host only when logging
        for c0 in range(1, Ltot, chunk):
            steps = range(c0, min(c0 + chunk, Ltot))
            st = Stats.zeros(R, d, V, dev)
            hook = CavitySensorCollector(st) if sensor_stats == "cavity" else SensorCollector(st)
            rows = torch.arange(R, device=dev)[:, None]
            mf, Pf, mm_l, Pm_l, yp_l = [], [], [], [], []
            m_block, P_block = m, P
            for i, t in enumerate(steps):
                mm, Pm = forecast(m, P, streams[:, t - 1], prm)
                m, P, logp = assimilate(mm, Pm, streams[:, t], prm, tree, collect=hook)
                if sensor_stats == "cavity":
                    hook.finish(rows, m, P, prm)
                nll -= logp.sum(-1)
                ntok += B
                post_m, post_P = (mm, Pm) if blind else (m, P)
                mf.append(post_m); Pf.append(post_P); mm_l.append(mm); Pm_l.append(Pm); yp_l.append(streams[:, t - 1])
                if len(mf) == window or i == len(steps) - 1:
                    accumulate_transition(st, prm.G, m_block, P_block, mf, Pf, mm_l, Pm_l, yp_l)
                    m_block, P_block = m, P                     # the state the next forecast is built from
                    mf, Pf, mm_l, Pm_l, yp_l = [], [], [], [], []
            st = st.scaled(1.0 / (B * len(steps)))
            k_eff = k % reset_every if reset_every else k
            gamma = max((k_eff + 1) ** -gamma_exponent, gamma_floor)
            running = st if running is None or gamma >= 1.0 else running.blend(st, gamma)
            new = m_step(running, prm, ridge, q_floor, learn_transition=link_chart == "matrix",
                         expand_sensor=expand_sensor)
            if link_chart == "exp":
                L, new.G, new.E, new.Q = exp_link_transition(running, prm, L, ridge, q_floor, link_step)
            if overrelax != 1.0:
                for f in ("G", "E", "A", "b", "Q"):
                    old = getattr(prm, f)
                    setattr(new, f, old + overrelax * (getattr(new, f) - old))
                Q = 0.5 * (new.Q + new.Q.transpose(-1, -2))
                evals, evecs = torch.linalg.eigh(Q)
                new.Q = (evecs * evals.clamp_min(q_floor).unsqueeze(-2)) @ evecs.transpose(-1, -2)
            prm = new
            k += 1
            if log is not None and (k % log_every == 0 or c0 + chunk >= Ltot):
                log({"arm": "local", "chunk": k, "train_bits": (nll / ntok / math.log(2)).tolist(),
                     "G_spectral": torch.linalg.matrix_norm(prm.G, 2).tolist(), "sec": time.perf_counter() - t0})
                nll, ntok = torch.zeros_like(nll), 0
    return prm


def train_backprop(prm0: Params, streams: Tensor, tree: TreeSensor, *, chunk: int = 128, lr: float = 3e-3,
                   clip: float = 1.0, link_chart: str = "matrix", log: Optional[Logger] = None,
                   log_every: int = 20) -> Params:
    r"""Adam on the filter's forecast NLL, backpropagated through the filter (truncated at ``chunk``), for all
    runs at once: the loss is the sum of the runs' mean NLLs and each run's gradient is clipped separately.
    ``link_chart="exp"`` learns the transition as :math:`G=e^L` with L the leaf (autograd through matrix_exp),
    starting from the principal log of ``prm0.G``."""
    if link_chart not in ("matrix", "exp"):
        raise ValueError(f"link_chart must be 'matrix' or 'exp', got {link_chart!r}")
    d, R = prm0.d, prm0.R
    dev = prm0.G.device
    eye = torch.eye(d, dtype=torch.float64, device=dev)
    G = (spd_logm(prm0.G) if link_chart == "exp" else prm0.G.clone()).requires_grad_(True)   # L for the exp chart
    E = prm0.E.clone().requires_grad_(True)
    Lq = torch.linalg.cholesky(prm0.Q - 1e-4 * eye).requires_grad_(True)
    A = prm0.A.clone().requires_grad_(True)
    b = prm0.b.clone().requires_grad_(True)
    leaves = [G, E, Lq, A, b]
    opt = torch.optim.Adam(leaves, lr=lr)

    def current() -> Params:
        Lt = torch.tril(Lq)
        Gm = torch.linalg.matrix_exp(G) if link_chart == "exp" else G
        return Params(G=Gm, E=E, Q=Lt @ Lt.transpose(-1, -2) + 1e-4 * eye, A=A, b=b)

    B, Ltot = streams.shape
    with torch.no_grad():
        m, P = start_state(streams[:, 0], current(), tree)
    k, t0, ntok = 0, time.perf_counter(), 0
    nll = torch.zeros(R, dtype=torch.float64, device=dev)
    for c0 in range(1, Ltot, chunk):
        prm = current()
        m, P = m.detach(), P.detach()
        loss = torch.zeros(R, dtype=torch.float64, device=dev)
        steps = range(c0, min(c0 + chunk, Ltot))
        for t in steps:
            mm, Pm = forecast(m, P, streams[:, t - 1], prm)
            m, P, logp = assimilate(mm, Pm, streams[:, t], prm, tree)
            loss = loss - logp.sum(-1)
        n = B * len(steps)
        opt.zero_grad()
        (loss.sum() / n).backward()
        with torch.no_grad():                                          # per-run gradient-norm clipping
            sq = sum(p.grad.reshape(R, -1).pow(2).sum(-1) for p in leaves)
            scale = (clip / (sq.sqrt() + 1e-6)).clamp(max=1.0)
            for p in leaves:
                p.grad.mul_(scale.reshape(R, *([1] * (p.dim() - 1))))
        opt.step()
        nll += loss.detach()
        ntok += n
        k += 1
        if log is not None and (k % log_every == 0 or c0 + chunk >= Ltot):
            log({"arm": "backprop", "chunk": k, "train_bits": (nll / ntok / math.log(2)).tolist(),
                 "G_spectral": torch.linalg.matrix_norm(G.detach(), 2).tolist(), "sec": time.perf_counter() - t0})
            nll, ntok = torch.zeros_like(nll), 0
    with torch.no_grad():
        return current().clone()
