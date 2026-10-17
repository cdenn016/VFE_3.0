r"""Audit 2026-10-12 T1 (per-layer evidence lower bound) and T2 (spread-corrected mean-field covariance).

Every reference here is computed independently of the closed forms under test, on tiny float64 CPU cases:

* T1, minimal agent: the evidence lower bound against :math:`E_q[\log p(X, J, o)] - E_q[\log q]` by tensor-product
  Gauss-Hermite quadrature of the declared log joint (exact for its quadratic log densities), with and without
  comparison rotations and with a receiver that hears no record; the one-head evidence against direct grid
  integration of the declared joint, and the ADF evidence of two decoupled heads against the label-tuple sum; the
  bound under the evidence and nondecreasing over coordinate-ascent sweeps.
* T1, prototype agents: the evidence against grid integration, the bound against quadrature, the bound under the
  evidence and nondecreasing over sweeps, for the diagonal, dense and block layouts.
* T1 plumbing: the per-layer capture, the ``val_*`` keys, the diagnostics CSV rows, nothing when the toggle is off,
  and the forward (so the loss) unchanged by the toggle.
* T2: the covariance against a dense reference built from loops over the labels (with comparison rotations), positive
  definiteness and the Loewner order over the moment-matched covariance, the KL to the exact posterior over 200
  random one-head instances, exact invariance under a non-orthogonal common reference change, causality, and the
  configuration refusals.
"""
import math
import warnings
from typing import Callable, List, Tuple

import numpy as np
import pytest
import torch
from torch.distributions import MultivariateNormal

import vfe3.train as T
from tests.profiles import make_relational_operators, test_config
from vfe3.contracts import CanonicalFrameContext
from vfe3.gradients.relational import relational_predictive_readout
from vfe3.model import minimal_agent as ma
from vfe3.model import prototype_agents as pa
from vfe3.model.model import VFEModel

torch.set_num_threads(1)
F64 = torch.float64
LOG2PI = math.log(2.0 * math.pi)


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _spd(g: torch.Generator, *shape: int) -> torch.Tensor:          # (*batch, K, K)
    K = shape[-1]
    X = torch.randn(*shape, K, generator=g, dtype=F64)
    return X @ X.transpose(-1, -2) / K + 0.5 * torch.eye(K, dtype=F64)


def _orthogonal(g: torch.Generator, N: int, d: int) -> torch.Tensor:  # (N, d, d)
    return torch.linalg.qr(torch.randn(N, d, d, generator=g, dtype=F64))[0]


def _instance(seed: int, N: int = 4, K: int = 2, H: int = 2, d: int = 1, causal: bool = True):
    g = _gen(seed)
    xi = torch.randn(N, K, generator=g, dtype=F64)
    C = _spd(g, N, K)
    S = torch.randn(H, d, K, generator=g, dtype=F64)
    Tm = torch.randn(H, d, K, generator=g, dtype=F64)
    log_noise = torch.log(torch.rand(H, generator=g, dtype=F64) + 0.3)
    lp = 0.5 * torch.randn(H, N, N, generator=g, dtype=F64)
    if causal:
        lp = lp.masked_fill(~torch.ones(N, N, dtype=torch.bool).tril(), -math.inf)
    return xi, C, S, Tm, log_noise, lp


def _expect(m: torch.Tensor, V: torch.Tensor, f: Callable[[torch.Tensor], torch.Tensor], n: int = 3) -> torch.Tensor:
    r""":math:`E_{N(m, V)}[f(X)]` by tensor-product Gauss-Hermite quadrature, exact for polynomials of degree
    :math:`\le 2n - 1` in every coordinate."""
    nodes, weights = np.polynomial.hermite_e.hermegauss(n)
    nodes, weights = torch.tensor(nodes, dtype=F64), torch.tensor(weights / math.sqrt(2.0 * math.pi), dtype=F64)
    K = m.shape[-1]
    grid = torch.cartesian_prod(*([nodes] * K)).reshape(-1, K)                # (n^K, K)
    w = torch.cartesian_prod(*([weights] * K)).reshape(-1, K).prod(-1)       # (n^K,)
    X = m + grid @ torch.linalg.cholesky(V).T                                 # (n^K, K)
    return (w * f(X)).sum()


def _log_normal(x: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    return MultivariateNormal(mean, cov).log_prob(x)


def _brute_force_elbo(xi, C, S, Tm, log_noise, lp, result, rope=None) -> torch.Tensor:
    r""":math:`E_q[\log N(X;\xi_i, C_i) + \sum_h(\log\hat\pi_{J^h} + \log N(R_iS_hX; R_jb_j, R_jB_jR_j^\top))] + H[q]`
    with :math:`q = N(\xi^+, C^+)\prod_h\beta^h`, by quadrature; a receiver whose only admissible source is itself
    under a self-masking prior hears no record of that head."""
    N, K = xi.shape
    H, d, _ = S.shape
    C_plus = C - torch.einsum("nhks,nhst,nhlt->nkl", result.gains, result.M, result.gains)
    eye = torch.eye(d, dtype=F64)
    R = torch.eye(d, dtype=F64).expand(N, d, d) if rope is None else rope
    admissible = torch.isfinite(lp)
    out = torch.empty(N, dtype=F64)
    for i in range(N):
        m, V = result.xi[i], C_plus[i]
        value = _expect(m, V, lambda X: _log_normal(X, xi[i], C[i])) + MultivariateNormal(m, V).entropy()
        for h in range(H):
            row = admissible[h, i]
            if row.sum() == 1 and row[i] and (~admissible[h]).diagonal().any():
                continue                                                      # no record (self-masking prior)
            beta, log_pi = result.weights[h, i], torch.log_softmax(lp[h, i], dim=-1)
            for j in range(N):
                if beta[j] == 0:
                    continue
                b = R[j] @ Tm[h] @ xi[j]
                B = R[j] @ (Tm[h] @ C[j] @ Tm[h].T + log_noise[h].exp() * eye) @ R[j].T
                value = value + beta[j] * (log_pi[j] - beta[j].log()
                                           + _expect(m, V, lambda X: _log_normal(X @ (R[i] @ S[h]).T, b, B)))
        out[i] = value
    return out


@pytest.mark.parametrize("inference,cavi_steps", [("mean_field", 1), ("mean_field", 3), ("exact", 1)])
@pytest.mark.parametrize("variant", ["plain", "rope", "noself"])
def test_minimal_agent_elbo_equals_quadrature_of_the_declared_joint(inference: str, cavi_steps: int,
                                                                    variant: str) -> None:
    d = 2 if variant == "rope" else 1
    xi, C, S, Tm, log_noise, lp = _instance(10, N=4, K=3, H=2, d=d)
    rope = _orthogonal(_gen(11), 4, d) if variant == "rope" else None
    if variant == "noself":                                                   # causal_noself: (0, 0) kept, no record
        lp = lp.masked_fill(torch.eye(4, dtype=torch.bool), -math.inf)
        lp[:, 0, 0] = 0.0
    res = ma.minimal_agent_readout(xi, C, S, Tm, log_noise, lp, inference, rope=rope, cavi_steps=cavi_steps,
                                   elbo=True)
    want = _brute_force_elbo(xi, C, S, Tm, log_noise, lp, res, rope)
    torch.testing.assert_close(res.elbo, want, rtol=1e-10, atol=1e-10)


def _grid_evidence(xi: torch.Tensor, C: torch.Tensor,
                   log_likelihood: Callable[[torch.Tensor], torch.Tensor], n: int = 601, width: float = 9.0) -> float:
    r""":math:`\log\int N(x;\xi, C)\,e^{\ell(x)}dx` by the trapezoid rule on a whitened 2-D grid."""
    z = torch.linspace(-width, width, n, dtype=F64)
    Z = torch.cartesian_prod(z, z)                                            # (n^2, 2)
    L = torch.linalg.cholesky(C)
    X = xi + Z @ L.T
    integrand = torch.exp(-0.5 * Z.square().sum(-1) - LOG2PI + log_likelihood(X)).reshape(n, n)
    h = float(z[1] - z[0])
    return math.log(float(torch.trapezoid(torch.trapezoid(integrand, dx=h), dx=h)))


def test_one_head_evidence_equals_the_integrated_marginal() -> None:
    xi, C, S, Tm, log_noise, lp = _instance(20, N=3, K=2, H=1, d=1, causal=False)
    exact = ma.minimal_agent_readout(xi, C, S, Tm, log_noise, lp, "exact", elbo=True)
    mean_field = ma.minimal_agent_readout(xi, C, S, Tm, log_noise, lp, "mean_field", elbo=True)
    torch.testing.assert_close(mean_field.evidence, exact.evidence, rtol=0, atol=0)   # one exact path for both
    noise = log_noise[0].exp()
    for i in range(3):
        pi_hat = torch.softmax(lp[0, i], dim=-1)

        def likelihood(X: torch.Tensor) -> torch.Tensor:
            terms = [pi_hat[j] * torch.exp(_log_normal(X @ S[0].T, Tm[0] @ xi[j],
                                                       Tm[0] @ C[j] @ Tm[0].T + noise * torch.eye(1, dtype=F64)))
                     for j in range(3)]
            return torch.stack(terms).sum(0).log()

        want = _grid_evidence(xi[i], C[i], likelihood)
        assert abs(float(exact.evidence[i]) - want) < 1e-8
        assert float(exact.elbo[i]) <= want + 1e-10 and float(mean_field.elbo[i]) <= want + 1e-10


def test_adf_evidence_of_decoupled_heads_is_the_label_tuple_sum() -> None:
    r"""With :math:`S_1 = e_1^\top`, :math:`S_2 = e_2^\top` and diagonal :math:`C_i`, :math:`S_1C_iS_2^\top = 0`, so ADF is
    exact and its evidence equals :math:`\log\sum_{j_1,j_2}\hat\pi_{j_1}\hat\pi_{j_2}\int N(x;\xi_i, C_i)\prod_h
    N(S_hx; b^h_{j_h}, B^h_{j_h})dx`."""
    xi, C, _, Tm, log_noise, lp = _instance(21, N=3, K=2, H=2, d=1, causal=False)
    C = torch.diag_embed(torch.diagonal(C, dim1=-2, dim2=-1))
    S = torch.eye(2, dtype=F64).reshape(2, 1, 2)
    res = ma.minimal_agent_readout(xi, C, S, Tm, log_noise, lp, "exact", elbo=True)
    eye = torch.eye(1, dtype=F64)
    for i in range(3):
        pis = [torch.softmax(lp[h, i], dim=-1) for h in range(2)]

        def likelihood(X: torch.Tensor) -> torch.Tensor:
            heads = []
            for h in range(2):
                heads.append(torch.stack([pis[h][j] * torch.exp(_log_normal(
                    X @ S[h].T, Tm[h] @ xi[j], Tm[h] @ C[j] @ Tm[h].T + log_noise[h].exp() * eye))
                    for j in range(3)]).sum(0))
            return (heads[0] * heads[1]).log()                                # = sum over label tuples

        assert abs(float(res.evidence[i]) - _grid_evidence(xi[i], C[i], likelihood)) < 1e-8


@pytest.mark.parametrize("H", [1, 2])
def test_elbo_is_nondecreasing_over_sweeps_and_below_the_one_head_evidence(H: int) -> None:
    xi, C, S, Tm, log_noise, lp = _instance(30, N=5, K=4, H=H, d=2)
    bounds = [ma.minimal_agent_readout(xi, C, S, Tm, log_noise, lp, "mean_field", cavi_steps=n, elbo=True)
              for n in range(1, 9)]
    for before, after in zip(bounds, bounds[1:]):
        assert float((after.elbo - before.elbo).min()) >= -1e-10
    gap_1, gap_8 = (bounds[0].adf_evidence - bounds[0].elbo), (bounds[-1].adf_evidence - bounds[-1].elbo)
    exact = ma.minimal_agent_readout(xi, C, S, Tm, log_noise, lp, "exact", elbo=True)
    if H == 1:                                                                # the ADF evidence is exact
        assert float(gap_1.min()) >= -1e-10 and float(gap_8.min()) >= -1e-10
        assert float((exact.adf_evidence - exact.elbo).min()) >= -1e-10
    print(f"\n[T1] H={H} mean-field gap per receiver (nats): 1 sweep {gap_1.numpy().round(4).tolist()}, "
          f"8 sweeps {gap_8.numpy().round(4).tolist()}; exact-path ELBO gap "
          f"{(exact.adf_evidence - exact.elbo).numpy().round(4).tolist()}")


def test_elbo_on_block_diagonal_covariances_equals_the_dense_call() -> None:
    xi, C, S, Tm, log_noise, lp = _instance(31, N=4, K=4, H=2, d=2)
    blocks = torch.stack([C[:, :2, :2], C[:, 2:, 2:]], dim=1)                 # (N, 2, 2, 2)
    dense = torch.zeros_like(C)
    dense[:, :2, :2], dense[:, 2:, 2:] = blocks[:, 0], blocks[:, 1]
    for inference in ("mean_field", "exact"):
        a = ma.minimal_agent_readout(xi, dense, S, Tm, log_noise, lp, inference, elbo=True)
        b = ma.minimal_agent_readout(xi, blocks, S, Tm, log_noise, lp, inference, elbo=True)
        torch.testing.assert_close(b.elbo, a.elbo, rtol=1e-11, atol=1e-11)
        torch.testing.assert_close(b.adf_evidence, a.adf_evidence, rtol=1e-11, atol=1e-11)
        torch.testing.assert_close(b.is_evidence, a.is_evidence, rtol=1e-11, atol=1e-11)


# ----------------------------------------------------------------------------------------------
# T1: prototype agents
# ----------------------------------------------------------------------------------------------

def _templates(seed: int, M: int, K: int):
    g = _gen(seed)
    nu = 1.5 * torch.randn(M, K, generator=g, dtype=F64)
    log_lambda = torch.log(torch.rand(M, K, generator=g, dtype=F64) + 0.2)
    log_pi = torch.randn(M, generator=g, dtype=F64)
    return nu, log_lambda, log_pi


def _prototype_working(layout: str, R: int, K: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Working covariance in ``layout`` and its dense form; blocks are 2 x 2."""
    g = _gen(seed)
    if layout == "diagonal":
        c = torch.rand(R, K, generator=g, dtype=F64) + 0.3
        return c, torch.diag_embed(c)
    if layout == "dense":
        C = _spd(g, R, K)
        return C, C
    blocks = _spd(g, R, K // 2, 2)                                            # (R, G, 2, 2)
    dense = torch.zeros(R, K, K, dtype=F64)
    for m in range(K // 2):
        dense[:, 2 * m:2 * m + 2, 2 * m:2 * m + 2] = blocks[:, m]
    return blocks, dense


@pytest.mark.parametrize("layout", ["diagonal", "dense", "blocks"])
@pytest.mark.parametrize("inference", ["mean_field", "exact"])
def test_prototype_elbo_and_evidence_against_quadrature_and_integration(layout: str, inference: str) -> None:
    R, K, M = 3, 2, 4
    nu, log_lambda, log_pi = _templates(40, M, K)
    xi = torch.randn(R, K, generator=_gen(41), dtype=F64)
    C, Cd = _prototype_working(layout, R, K, 42)
    if inference == "exact":
        post = pa.prototype_exact_posterior(xi, C, nu, log_lambda, log_pi, need_covariance=True)
        labels = post.weights
    else:
        post = pa.prototype_mean_field_posterior(xi, C, nu, log_lambda, log_pi, cavi_steps=2)
        labels = post.responsibilities
    elbo = pa.prototype_elbo(xi, C, post.xi_plus, post.C_plus, labels, nu, log_lambda, log_pi)
    evidence = pa.prototype_exact_posterior(xi, C, nu, log_lambda, log_pi, need_covariance=False).log_evidence
    pi_hat = torch.softmax(log_pi, dim=-1)
    lam = log_lambda.exp()
    for i in range(R):
        Vq = (torch.diag_embed(post.C_plus[i]) if layout == "diagonal" else
              post.C_plus[i] if layout == "dense" else torch.block_diag(*post.C_plus[i]))
        m = post.xi_plus[i]
        want = _expect(m, Vq, lambda X: _log_normal(X, xi[i], Cd[i])) + MultivariateNormal(m, Vq).entropy()
        for a in range(M):
            want = want + labels[i, a] * (pi_hat[a].log() - labels[i, a].log()
                                          + _expect(m, Vq, lambda X: _log_normal(X, nu[a], torch.diag(lam[a]))))
        assert abs(float(elbo[i]) - float(want)) < 1e-10

        def likelihood(X: torch.Tensor) -> torch.Tensor:
            return torch.stack([pi_hat[a] * torch.exp(_log_normal(X, nu[a], torch.diag(lam[a])))
                                for a in range(M)]).sum(0).log()

        integrated = _grid_evidence(xi[i], Cd[i], likelihood)
        assert abs(float(evidence[i]) - integrated) < 1e-8
        assert float(elbo[i]) <= integrated + 1e-10


@pytest.mark.parametrize("layout", ["diagonal", "dense", "blocks"])
def test_prototype_elbo_is_nondecreasing_over_sweeps(layout: str) -> None:
    R, K, M = 5, 4, 6
    nu, log_lambda, log_pi = _templates(50, M, K)
    xi = 1.5 * torch.randn(R, K, generator=_gen(51), dtype=F64)
    C, _ = _prototype_working(layout, R, K, 52)
    evidence = pa.prototype_exact_posterior(xi, C, nu, log_lambda, log_pi, need_covariance=False).log_evidence
    previous = None
    for n in range(1, 8):
        post = pa.prototype_mean_field_posterior(xi, C, nu, log_lambda, log_pi, cavi_steps=n)
        elbo = pa.prototype_elbo(xi, C, post.xi_plus, post.C_plus, post.responsibilities, nu, log_lambda, log_pi)
        assert float((evidence - elbo).min()) >= -1e-10
        if previous is not None:
            assert float((elbo - previous).min()) >= -1e-10
        previous = elbo
    print(f"\n[T1] prototype {layout}: mean-field gap after 7 sweeps (nats) "
          f"{(evidence - previous).numpy().round(4).tolist()}")


# ----------------------------------------------------------------------------------------------
# T1 plumbing: capture, columns, CSV rows, the toggle
# ----------------------------------------------------------------------------------------------

def _model(elbo: bool, **overrides) -> VFEModel:
    values = dict(attention_mode="projected_predictive", predictive_readout="minimal_agent",
                  minimal_agent_inference="mean_field", block_mlp_mode="prototype_agents",
                  block_mlp_inference="mean_field", block_mlp_expansion=2, n_layers=2, max_seq_len=6,
                  attention_residual=False, predictive_covariance="moment_matched", diagnose_agent_elbo=elbo)
    values.update(overrides)
    torch.manual_seed(0)
    return VFEModel(test_config(**values))


def test_elbo_capture_rows_and_columns_appear_only_with_the_toggle() -> None:
    on, off = _model(True), _model(False)
    windows = torch.randint(0, on.cfg.vocab_size, (3, 6), generator=_gen(60))
    with torch.no_grad():
        logits_on, logits_off = on(windows), off(windows)
    assert torch.equal(logits_on, logits_off)                                 # the toggle never touches the forward
    per_layer = on.diagnostics_per_layer(windows[:1])
    uncoupled = [coupling == 0.0 for coupling in per_layer["relational_minimal_agent_head_coupling"]]
    for key in ma.MINIMAL_AGENT_ELBO_KEYS:
        values = per_layer[key]
        # The exact evidence is reported only where every receiver's heads are uncoupled (round 2, item 12).
        expect = uncoupled if key == "relational_minimal_agent_evidence" else [True] * on.cfg.n_layers
        assert [math.isfinite(v) for v in values] == expect, key
    for row in zip(uncoupled, *(per_layer[key] for key in ma.MINIMAL_AGENT_ELBO_KEYS)):
        exact_layer, elbo, adf, adf_gap, evidence, sampled, sampled_gap, sampled_ess = row
        assert adf_gap == pytest.approx(adf - elbo, rel=1e-6, abs=1e-6)
        assert sampled_gap == pytest.approx(sampled - elbo, rel=1e-6, abs=1e-6)
        assert 1.0 / ma.MINIMAL_AGENT_IS_SAMPLES - 1e-12 <= sampled_ess <= 1.0 + 1e-12   # ESS / n (audit 2026-10-16, F47)
        if exact_layer:                                                       # ADF is exact: the gap is a true gap
            assert evidence == pytest.approx(adf, rel=1e-6, abs=1e-6) and adf_gap >= -1e-6
    assert not any(key in off.diagnostics_per_layer(windows[:1]) for key in ma.MINIMAL_AGENT_ELBO_KEYS)
    for model, expect in ((on, True), (off, False)):
        features: dict = {}
        T._residual_and_attention_rows(model, windows, 5, torch.device("cpu"), features_out=features)
        assert all("forward_moments" not in mlp.__dict__ for mlp in model.block_mlps)
        minimal_rows, prototype_rows = T._elbo_rows(features, 5)
        assert bool(minimal_rows) == expect and bool(prototype_rows) == expect
        for row in minimal_rows:
            # The ADF gap is not sign-definite with coupled heads, and the importance-sampled one is a finite-sample
            # estimate (round 2, item 12 and M1), so neither is asserted nonnegative here.
            assert set(row) == set(T.MINIMAL_AGENT_ELBO_FIELDS) and row["is_samples"] == ma.MINIMAL_AGENT_IS_SAMPLES
            assert math.isfinite(row["adf_elbo_gap"]) and math.isfinite(row["is_elbo_gap"])
            assert 0.0 < row["is_ess"] <= 1.0 + 1e-12                         # the weights' ESS / n beside them
        for row in prototype_rows:
            assert set(row) == set(T.PROTOTYPE_ELBO_FIELDS) and row["n_tokens"] == windows.numel()
            assert row["elbo_gap"] >= -1e-9
    assert set(T._VAL_MINIMAL_AGENT_ELBO_KEYS) == {"val_" + key for key in ma.MINIMAL_AGENT_ELBO_KEYS}


def test_elbo_toggle_is_inactive_without_a_minimal_or_prototype_agent() -> None:
    kernel = test_config(attention_mode="projected_predictive", use_block_mlp=False)
    assert "diagnose_agent_elbo" in kernel.inactive_fields()
    minimal = test_config(attention_mode="projected_predictive", predictive_readout="minimal_agent",
                          attention_residual=False, use_block_mlp=False)
    assert "diagnose_agent_elbo" not in minimal.inactive_fields()
    with pytest.raises(ValueError, match="diagnose_agent_elbo"):
        test_config(diagnose_agent_elbo=1)


# ----------------------------------------------------------------------------------------------
# T2: the spread-corrected mean-field covariance
# ----------------------------------------------------------------------------------------------

def _dense_spread_reference(xi, C, S, Tm, log_noise, lp, rope=None, cavi_steps: int = 1) -> List[torch.Tensor]:
    r""":math:`P_i + \sum_hP_iS_h^\top R'^h_iS_hP_i` from explicit loops: the mean-field labels of the readout, the
    precision :math:`P_i^{-1} = C_i^{-1} + \sum_hS_h^\top\Lambda^hS_h` inverted directly, and every :math:`d^h_{ij}` formed
    one pair at a time in the comparison frame, rotated back by :math:`R_i^\top`."""
    res = ma.minimal_agent_readout(xi, C, S, Tm, log_noise, lp, "mean_field", rope=rope, cavi_steps=cavi_steps)
    N, K = xi.shape
    H, d, _ = S.shape
    R = torch.eye(d, dtype=F64).expand(N, d, d) if rope is None else rope
    out = []
    for i in range(N):
        precision = torch.linalg.inv(C[i])
        Bs = [[R[j] @ (Tm[h] @ C[j] @ Tm[h].T + log_noise[h].exp() * torch.eye(d, dtype=F64)) @ R[j].T
               for j in range(N)] for h in range(H)]
        bs = [[R[j] @ Tm[h] @ xi[j] for j in range(N)] for h in range(H)]
        for h in range(H):
            Lam = sum(res.weights[h, i, j] * torch.linalg.inv(Bs[h][j]) for j in range(N))
            precision = precision + S[h].T @ R[i].T @ Lam @ R[i] @ S[h]
        P = torch.linalg.inv(precision)
        torch.testing.assert_close(res.xi[i], P @ (torch.linalg.solve(C[i], xi[i]) + sum(
            S[h].T @ R[i].T @ sum(res.weights[h, i, j] * torch.linalg.solve(Bs[h][j], bs[h][j]) for j in range(N))
            for h in range(H))), rtol=1e-10, atol=1e-10)                     # the reference fusion is the readout's
        total = P.clone()
        for h in range(H):
            a = R[i] @ S[h] @ res.xi[i]
            g = [torch.linalg.solve(Bs[h][j], bs[h][j] - a) for j in range(N)]
            mean = sum(res.weights[h, i, j] * g[j] for j in range(N))
            spread = sum(res.weights[h, i, j] * torch.outer(g[j] - mean, g[j] - mean) for j in range(N))
            G = P @ S[h].T @ R[i].T
            total = total + G @ spread @ G.T
        out.append(total)
    return out


@pytest.mark.parametrize("rope", [False, True])
@pytest.mark.parametrize("cavi_steps", [1, 3])
def test_spread_corrected_covariance_equals_the_dense_loop_reference(rope: bool, cavi_steps: int) -> None:
    xi, C, S, Tm, log_noise, lp = _instance(70, N=5, K=4, H=2, d=2)
    R = _orthogonal(_gen(71), 5, 2) if rope else None
    res = ma.minimal_agent_readout(xi, C, S, Tm, log_noise, lp, "mean_field", rope=R, cavi_steps=cavi_steps,
                                   spread_corrected=True)
    P = C - torch.einsum("nhks,nhst,nhlt->nkl", res.gains, res.M, res.gains)
    got = P + torch.einsum("nhks,nhst,nhlt->nkl", res.spread_gains, res.label_spread, res.spread_gains)
    for i, want in enumerate(_dense_spread_reference(xi, C, S, Tm, log_noise, lp, R, cavi_steps)):
        torch.testing.assert_close(got[i], want, rtol=1e-10, atol=1e-10)
    tiled_bytes = ma.SPREAD_TILE_BYTES
    try:
        ma.SPREAD_TILE_BYTES = 1                                              # one receiver row per tile
        tiled = ma.minimal_agent_readout(xi, C, S, Tm, log_noise, lp, "mean_field", rope=R, cavi_steps=cavi_steps,
                                         spread_corrected=True)
    finally:
        ma.SPREAD_TILE_BYTES = tiled_bytes
    torch.testing.assert_close(tiled.label_spread, res.label_spread, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("tile_bytes", [ma.SPREAD_TILE_BYTES, 1])
def test_spread_corrected_covariance_gradients_match_central_differences(tile_bytes: int) -> None:
    r"""``torch.autograd.gradcheck`` of :math:`P + \sum_hPS_h^\top R'^hS_hP` in every input, single tile and one
    checkpointed receiver row per tile."""
    xi, C, S, Tm, log_noise, lp = _instance(73, N=3, K=3, H=2, d=2)
    L = torch.linalg.cholesky(C)

    def covariance(xi_, L_, S_, T_, n_):
        C_ = L_ @ L_.transpose(-1, -2)
        res = ma.minimal_agent_readout(xi_, C_, S_, T_, n_, lp, "mean_field", spread_corrected=True, cavi_steps=2)
        P = C_ - torch.einsum("nhks,nhst,nhlt->nkl", res.gains, res.M, res.gains)
        return P + torch.einsum("nhks,nhst,nhlt->nkl", res.spread_gains, res.label_spread, res.spread_gains)

    saved = ma.SPREAD_TILE_BYTES
    try:
        ma.SPREAD_TILE_BYTES = tile_bytes
        inputs = tuple(t.clone().requires_grad_(True) for t in (xi, L, S, Tm, log_noise))
        assert torch.autograd.gradcheck(covariance, inputs, eps=1e-6, atol=1e-7, rtol=1e-6)
    finally:
        ma.SPREAD_TILE_BYTES = saved


def test_spread_corrected_refuses_the_exact_path_and_the_kernel_readout() -> None:
    xi, C, S, Tm, log_noise, lp = _instance(72, N=3, K=2, H=1, d=1)
    with pytest.raises(ValueError, match="mean-field correction"):
        ma.minimal_agent_readout(xi, C, S, Tm, log_noise, lp, "exact", spread_corrected=True)
    with pytest.raises(ValueError, match="spread_corrected"):
        make_relational_operators([2, 2], 0, predictive_covariance="spread_corrected")
    with pytest.raises(ValueError, match="spread_corrected"):
        make_relational_operators([2, 2], 0, predictive_covariance="spread_corrected", attention_residual=False,
                                  predictive_readout="minimal_agent", minimal_agent_inference="exact")
    with pytest.raises(ValueError, match="spread_corrected"):
        test_config(attention_mode="projected_predictive", predictive_covariance="spread_corrected")
    with pytest.raises(ValueError, match="spread_corrected"):
        test_config(attention_mode="projected_predictive", predictive_covariance="spread_corrected",
                    predictive_readout="minimal_agent", minimal_agent_inference="exact", attention_residual=False)
    ok = test_config(attention_mode="projected_predictive", predictive_covariance="spread_corrected",
                     predictive_readout="minimal_agent", minimal_agent_inference="mean_field",
                     attention_residual=False)
    assert "predictive_covariance_floor" not in ok.inactive_fields()
    gradient = test_config(attention_mode="projected_gradient", predictive_covariance="spread_corrected")
    assert "predictive_covariance" in gradient.inactive_fields()


def test_spread_corrected_model_trains_and_the_gap_probe_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""A tiny spread-corrected model reaches the label spread at every layer, takes one finite backward pass, and the
    exact-versus-mean-field probe (which switches the layer to exact inference) runs."""
    from vfe3.posthoc import feature_probes
    model = _model(False, predictive_covariance="spread_corrected", use_block_mlp=False)
    windows = torch.randint(0, model.cfg.vocab_size, (2, 6), generator=_gen(74))
    calls = {"n": 0}
    original = ma.label_spread

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(ma, "label_spread", counting)
    logits = model(windows)
    assert calls["n"] == model.cfg.n_layers
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), windows[:, 1:].reshape(-1))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(bool(torch.isfinite(g).all()) for g in grads)
    assert feature_probes.minimal_agent_gap(model, [windows[:1]], torch.device("cpu")) is not None


def _gaussian_kl(m0: torch.Tensor, S0: torch.Tensor, m1: torch.Tensor, S1: torch.Tensor) -> torch.Tensor:
    return torch.distributions.kl_divergence(MultivariateNormal(m0, S0), MultivariateNormal(m1, S1))


def test_spread_correction_moves_the_mean_field_toward_the_exact_posterior() -> None:
    r"""200 random one-head instances (:math:`N = 8` sources under a uniform prior, :math:`K = 4`, :math:`d = 2`, means
    :math:`2\,N(0, I)`, :math:`C_j = XX^\top/K + I/2`, noise in :math:`[0.1, 0.5]`): the KL from the exact posterior's
    moments to each Gaussian at the last receiver, which is the excess of :math:`\mathrm{KL}(p\,\|\,q)` from the exact
    mixture over its moment match."""
    N, K, d = 8, 4, 2
    before, after = [], []
    for seed in range(200):
        g = _gen(5000 + seed)
        xi = 2.0 * torch.randn(N, K, generator=g, dtype=F64)
        C = _spd(g, N, K)
        S = torch.randn(1, d, K, generator=g, dtype=F64)
        Tm = torch.randn(1, d, K, generator=g, dtype=F64)
        log_noise = torch.log(0.1 + 0.4 * torch.rand(1, generator=g, dtype=F64))
        lp = torch.zeros(1, N, N, dtype=F64)
        ex = ma.minimal_agent_readout(xi, C, S, Tm, log_noise, lp, "exact")
        mf = ma.minimal_agent_readout(xi, C, S, Tm, log_noise, lp, "mean_field", spread_corrected=True)
        C_exact = C - torch.einsum("nhks,nhst,nhlt->nkl", ex.gains, ex.M, ex.gains)
        P = C - torch.einsum("nhks,nhst,nhlt->nkl", mf.gains, mf.M, mf.gains)
        corrected = P + torch.einsum("nhks,nhst,nhlt->nkl", mf.spread_gains, mf.label_spread, mf.spread_gains)
        assert float(torch.linalg.eigvalsh(corrected - P).min()) >= -1e-12   # Loewner: corrected >= P
        before.append(float(_gaussian_kl(ex.xi[-1], C_exact[-1], mf.xi[-1], P[-1])))
        after.append(float(_gaussian_kl(ex.xi[-1], C_exact[-1], mf.xi[-1], corrected[-1])))
    before_t, after_t = torch.tensor(before), torch.tensor(after)
    improved = float((after_t < before_t).double().mean())
    print(f"\n[T2] KL(exact moments || q), 200 instances: median mean field {float(before_t.median()):.4f} -> "
          f"spread corrected {float(after_t.median()):.4f}; improved in {100 * improved:.1f}%")
    assert float(after_t.median()) < 0.6 * float(before_t.median()) and improved >= 0.95


# T2 through RelationalOperators: local layouts, the floor, reference change, causality.
H2, D2, N2 = 2, 2, 4
K2 = H2 * D2
FLOOR = 1e-2


def _operators(covariance: str, seed: int = 0, **extra):
    ops = make_relational_operators([D2] * H2, 11, readout_mode="predictive_content", attention_residual=False,
                                    predictive_covariance=covariance, predictive_covariance_floor=FLOOR,
                                    predictive_readout="minimal_agent", minimal_agent_inference="mean_field",
                                    **extra).double()
    g = _gen(100 + seed)
    with torch.no_grad():
        for p in ops.parameters():
            p.add_(0.3 * torch.randn(p.shape, generator=g, dtype=F64))
    return ops


def _block_dense(blocks: torch.Tensor) -> torch.Tensor:                     # (..., H, d, d) -> (..., K, K)
    out = blocks.new_zeros(*blocks.shape[:-3], K2, K2)
    for h in range(H2):
        out[..., h * D2:(h + 1) * D2, h * D2:(h + 1) * D2] = blocks[..., h, :, :]
    return out


def _local_inputs(family: str, seed: int = 1):
    g = _gen(seed)
    U = torch.eye(D2, dtype=F64) + 0.3 * torch.randn(1, N2, H2, D2, D2, generator=g, dtype=F64)
    mu = 2.0 * torch.randn(1, N2, K2, generator=g, dtype=F64)
    if family == "diagonal":
        sigma = torch.rand(1, N2, K2, generator=g, dtype=F64) + 0.3
        prior = torch.rand(1, N2, K2, generator=g, dtype=F64) + 0.4
    else:
        sigma, prior = _spd(g, 1, N2, K2), _spd(g, 1, N2, K2)
    log_prior = torch.zeros(N2, N2, dtype=F64).masked_fill(~torch.ones(N2, N2, dtype=torch.bool).tril(), -math.inf)
    return U, mu, sigma, prior, log_prior


def _readout(ops, U, mu, sigma, prior, log_prior):
    frame = CanonicalFrameContext(U, torch.linalg.inv(U), block_diagonal=True)
    return relational_predictive_readout(mu, sigma, frame, ops, tau=1.0, log_prior=log_prior,
                                         return_covariance=True, floor_anchor=prior)


@pytest.mark.parametrize("family", ["diagonal", "full"])
def test_operator_output_is_floored_psd_and_above_the_moment_matched_covariance(family: str) -> None:
    U, mu, sigma, prior, log_prior = _local_inputs(family, seed=3)
    mean_mm, cov_mm = _readout(_operators("moment_matched", seed=2), U, mu, sigma, prior, log_prior)
    mean_sc, cov_sc = _readout(_operators("spread_corrected", seed=2), U, mu, sigma, prior, log_prior)
    assert torch.equal(mean_sc, mean_mm)                                      # the mean is not touched
    excess, lift = cov_sc - FLOOR * prior, cov_sc - cov_mm
    if family == "diagonal":
        assert bool((excess > 0).all()) and float(lift.min()) >= -1e-13
    else:
        assert float(torch.linalg.eigvalsh(excess).min()) > 0
        assert float(torch.linalg.eigvalsh(lift).min()) >= -1e-12
    assert float(lift.abs().max()) > 1e-4                                     # the correction is not negligible


@pytest.mark.parametrize("family", ["diagonal", "full"])
def test_spread_corrected_is_exactly_reference_invariant(family: str) -> None:
    r"""Common reference change :math:`U_i\to U_ia`, :math:`S\to Sa`, :math:`T\to Ta` with a non-orthogonal head-block
    :math:`a` of condition number 10 (as in ``tests/test_audit_20261012_floor.py``)."""
    U, mu, sigma, prior, log_prior = _local_inputs(family)
    ops = _operators("spread_corrected")
    mean, cov = _readout(ops, U, mu, sigma, prior, log_prior)
    g = _gen(7)
    Q1 = torch.linalg.qr(torch.randn(H2, D2, D2, generator=g, dtype=F64))[0]
    Q2 = torch.linalg.qr(torch.randn(H2, D2, D2, generator=g, dtype=F64))[0]
    a = Q1 @ torch.diag_embed(torch.logspace(-0.5, 0.5, D2, dtype=F64).expand(H2, D2)) @ Q2.transpose(-1, -2)
    dense = _block_dense(a)
    assert abs(float(torch.linalg.cond(dense)) - 10.0) < 1e-9
    with torch.no_grad():
        for name in ("raw_score_q", "T"):
            getattr(ops, name).copy_(getattr(ops, name) @ dense)
    mean_a, cov_a = _readout(ops, U @ a, mu, sigma, prior, log_prior)
    mean_dev = float((mean_a - mean).abs().max() / mean.abs().max())
    cov_dev = float((cov_a - cov).abs().max() / cov.abs().max())
    print(f"\n[T2] reference change cond(a)=10, {family}: mean {mean_dev:.2e}, covariance {cov_dev:.2e}")
    assert mean_dev < 1e-12 and cov_dev < 1e-12


@pytest.mark.parametrize("family", ["diagonal", "full"])
def test_spread_corrected_is_causal(family: str) -> None:
    U, mu, sigma, prior, log_prior = _local_inputs(family, seed=9)
    results = {}
    for covariance in ("moment_matched", "spread_corrected"):
        ops = _operators(covariance, seed=4)
        base = _readout(ops, U, mu, sigma, prior, log_prior)
        mu2, sigma2 = mu.clone(), sigma.clone()
        mu2[0, 2:] += 2.0
        sigma2[0, 2:] *= 1.7
        moved = _readout(ops, U, mu2, sigma2, prior, log_prior)
        for got, want in zip(moved, base):
            assert float((got[0, 2:] - want[0, 2:]).abs().max()) > 1e-3
        results[covariance] = all(torch.equal(got[0, :2], want[0, :2]) for got, want in zip(moved, base))
    print(f"\n[T2] causality ({family}): earlier receivers bitwise identical: {results}")
    assert results["spread_corrected"]
