r"""Phase 1 of the SO(N) mixed-tower plan (docs/implementation/so-n-mixed-towers-and-frame-tying-plan-2026-10-05.md).

An so_n irrep tower whose blocks differ in size, one block per head, on both projected routes with
one shared (tied) frame per token. Default fixture: SO(3) with [("l1", 2), ("l2", 1)], K = 11, head
sizes [3, 3, 5]. Pins, on CPU at tiny size: config acceptance and the head-count rule; per-run
operator storage (single run = today's tensors); the optimizer and parameter classifiers; head order
on the non-adjacent spec [l1, l2, l1]; the routing score, the moment-matched covariance and the
projected-gradient receiver against brute-force dense float64 references; the per-head
temperatures; train steps, gauge equivariance and a finite-difference phi gradient on the five so_n
routes; the run record; the content-link diagnostics; and a checkpoint/config round trip.
"""
from tests.profiles import make_relational_operators
import math
import warnings
from typing import Dict, List

import pytest
import torch

from tests.profiles import test_config, tiny_config
from tests.test_son_irreps import _SON_ROUTES
from vfe3.config import config_from_serialized, serialize_config
from vfe3.contracts import CanonicalFrameContext
from vfe3.geometry.groups import get_group
from vfe3.model.model import VFEModel
from vfe3.model.relational_operators import RelationalOperators, head_runs

SPEC = [("l1", 2), ("l2", 1)]                                   # head sizes [3, 3, 5], K = 11
MIXED = dict(gauge_group="so_n", group_n=3, irrep_spec=SPEC, embed_dim=11, n_heads=3)


def _model(dtype: torch.dtype = torch.float32, **overrides: object) -> VFEModel:
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(tiny_config(**{**MIXED, **overrides})).to(dtype)
    with torch.no_grad():
        model.prior_bank.phi_embed.normal_(mean=0.0, std=0.3)   # nontrivial SO(3) frames
    return model


def _frames(spec: list, K: int, N: int = 3, seed: int = 0) -> torch.Tensor:   # (1, N, K, K) float64
    group = get_group("so_n")(K, group_n=3, irrep_spec=spec)
    gens = group.generators.to(torch.float64)
    a = torch.randn(1, N, gens.shape[0], generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    return torch.linalg.matrix_exp(torch.einsum("...a,akl->...kl", a, gens))


def _per_head(runs: List[torch.Tensor]) -> List[torch.Tensor]:
    return [head for run in runs for head in run]


def _operators(dims: List[int], **kwargs: object) -> RelationalOperators:
    ops = make_relational_operators(dims, seed=5, **kwargs).double()
    with torch.no_grad():                                        # move off the identity start
        g = torch.Generator().manual_seed(7)
        for p in ops.parameters():
            p.add_(0.2 * torch.randn(p.shape, generator=g, dtype=p.dtype))
    return ops


# ---------------------------------------------------------------- config

def test_config_accepts_a_mixed_tower_with_one_block_per_head() -> None:
    cfg = test_config(**MIXED)
    assert cfg.head_dims == [3, 3, 5] and cfg.head_offsets == [0, 3, 6]
    with pytest.raises(ValueError, match="head_dims"):
        cfg.d_head
    assert math.isclose(cfg.tau, (2 * math.sqrt(3) + math.sqrt(5)) / 3)
    with pytest.raises(ValueError, match="3 blocks"):
        test_config(**{**MIXED, "n_heads": 2})
    assert test_config(**MIXED, kappa_beta=[1.0, 2.0, 0.5]).kappa_beta == [1.0, 2.0, 0.5]
    with pytest.raises(ValueError, match="one entry per irrep block"):
        test_config(**MIXED, kappa_beta=[1.0, 2.0])
    assert test_config(**MIXED, beta_attention_prior="causal_alibi").beta_attention_prior == "causal_alibi"
    # The plan's example tower, K = 3 * 7 + 17 = 38 with an l = 8 head (canonical basis).
    big = test_config(gauge_group="so_n", group_n=3, irrep_spec=[("l3", 3), ("l8", 1)], embed_dim=38, n_heads=4)
    assert big.head_dims == [7, 7, 7, 17]
    # Equal heads keep d_head.
    assert test_config(gauge_group="so_n", group_n=3, irrep_spec=[("l1", 2)], embed_dim=6, n_heads=2).d_head == 3


# ---------------------------------------------------------------- storage

def test_head_runs_are_maximal_contiguous_equal_size_runs() -> None:
    assert head_runs([3, 3, 5]) == [(0, 2, 3, 0), (2, 1, 5, 6)]
    assert head_runs([3, 5, 3]) == [(0, 1, 3, 0), (1, 1, 5, 3), (2, 1, 3, 8)]
    assert head_runs([4, 4]) == [(0, 2, 4, 0)]


def test_single_run_storage_is_unchanged() -> None:
    ops = make_relational_operators([3, 3], seed=1)
    for name, shape in (("raw_score_q", (2, 3, 6)), ("raw_score_k", (2, 3, 6)), ("T", (2, 3, 6)), ("O", (2, 6, 3))):
        assert isinstance(getattr(ops, name), torch.nn.Parameter) and tuple(getattr(ops, name).shape) == shape
    assert list(ops.state_dict()) == ["raw_score_q", "raw_score_k", "T", "O"]
    assert not ops.mixed_heads and ops.score_rank_per_head == 3


def test_mixed_run_storage_keys_shapes_and_initialization() -> None:
    ops = make_relational_operators([3, 3, 5], seed=1, output_init_scale=2.0)
    assert list(ops.state_dict()) == [f"{n}.{r}" for n in ("raw_score_q", "raw_score_k", "T", "O") for r in (0, 1)]
    assert tuple(ops.T[0].shape) == (2, 3, 11) and tuple(ops.T[1].shape) == (1, 5, 11)
    assert tuple(ops.O[0].shape) == (2, 11, 3) and tuple(ops.O[1].shape) == (1, 11, 5)
    T_stack = torch.cat([T.reshape(-1, 11) for T in ops.T])
    O_stack = torch.cat([O.permute(1, 0, 2).reshape(11, -1) for O in ops.O], dim=1)
    assert torch.equal(T_stack, torch.eye(11)) and torch.equal(O_stack, 2.0 * torch.eye(11))
    for S in (*ops.raw_score_q, *ops.raw_score_k):                # orthonormal rows per head
        torch.testing.assert_close(S @ S.transpose(-1, -2), torch.eye(S.shape[1]).expand(S.shape[0], -1, -1),
                                   atol=1e-6, rtol=0)
    with pytest.raises(ValueError, match="per-run accessor"):
        ops.content_factors()


@pytest.mark.parametrize("chart", ["first_order", "exponential"])
def test_charts_on_mixed_runs_equal_the_k_by_k_operator_expressions(chart: str) -> None:
    ops = _operators([3, 3, 5], content_link_chart=chart)
    T_runs, O_runs = ops.content_factor_runs()
    T = torch.cat([t.reshape(-1, 11) for t in T_runs])
    O_stored = torch.cat([o.permute(1, 0, 2).reshape(11, -1) for o in ops.O], dim=1)
    O_prime = torch.cat([o.permute(1, 0, 2).reshape(11, -1) for o in O_runs], dim=1)
    expected = (torch.linalg.inv(T) + O_stored if chart == "first_order"
                else torch.linalg.matrix_exp(O_stored @ T) @ torch.linalg.inv(T))
    torch.testing.assert_close(O_prime, expected, atol=1e-10, rtol=1e-10)


# ---------------------------------------------------------------- optimizer and classifiers

def test_optimizer_covers_every_run_parameter_once_and_classifiers_read_the_attribute() -> None:
    from vfe3.run_resources import parameter_role
    from vfe3.train import _parameter_group, build_optimizer
    model = _model()
    optimizer = build_optimizer(model, model.cfg)
    seen = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(seen) == len(set(seen))
    relational = [p for n, p in model.named_parameters() if n.startswith("relational_operators")]
    assert len(relational) == 8 and all(seen.count(id(p)) == 1 for p in relational)
    assert _parameter_group("relational_operators.T.0") == "relational_content"
    assert _parameter_group("relational_operators.raw_score_k.1") == "relational_score"
    assert parameter_role("relational_operators.O.1", (11, 5)) == ("context_processing", "relational_content")


# ---------------------------------------------------------------- routing score

def _kl(a_q: torch.Tensor, G_q: torch.Tensor, a_k: torch.Tensor, G_k: torch.Tensor) -> torch.Tensor:
    P = torch.linalg.inv(G_k)
    delta = a_q - a_k
    r = G_q.shape[-1]
    return 0.5 * (torch.trace(P @ G_q) + delta @ P @ delta - r
                  + torch.logdet(G_k) - torch.logdet(G_q))


@pytest.mark.parametrize("spec,dims", [(SPEC, [3, 3, 5]), ([("l1", 1), ("l2", 1), ("l1", 1)], [3, 5, 3])])
@pytest.mark.parametrize("full", [False, True])
def test_routing_score_matches_per_head_reference_in_head_order(spec: list, dims: List[int], full: bool) -> None:
    from vfe3.gradients.relational import relational_score
    ops = _operators(dims)
    U = _frames(spec, 11)
    frame = CanonicalFrameContext(U, torch.linalg.inv(U))
    g = torch.Generator().manual_seed(2)
    mu = torch.randn(1, 3, 11, generator=g, dtype=torch.float64)
    if full:
        A = torch.randn(1, 3, 11, 11, generator=g, dtype=torch.float64)
        sigma = A @ A.transpose(-1, -2) / 11 + 0.5 * torch.eye(11, dtype=torch.float64)
    else:
        sigma = torch.rand(1, 3, 11, generator=g, dtype=torch.float64) + 0.5
    capture: Dict[str, object] = {}
    score = relational_score(mu, sigma, mu, sigma, frame, ops, capture)
    assert score.shape == (1, 3, 3, 3)
    S_q = _per_head([q for q, _ in ops.comparison_runs()])
    S_k = _per_head([k for _, k in ops.comparison_runs()])
    Sigma = sigma if full else torch.diag_embed(sigma)
    U_inv = torch.linalg.inv(U)
    for h in range(3):
        for i in range(3):
            for j in range(3):
                Lq, Lk = S_q[h] @ U_inv[0, i], S_k[h] @ U_inv[0, j]
                ref = _kl(Lq @ mu[0, i], Lq @ Sigma[0, i] @ Lq.T, Lk @ mu[0, j], Lk @ Sigma[0, j] @ Lk.T)
                torch.testing.assert_close(score[0, h, i, j], ref, atol=1e-10, rtol=1e-10)
    assert isinstance(capture["projected_covariance_pre"], list)
    assert [G.shape[-1] for G in capture["projected_covariance_pre"]] == [d for _, _, d, _ in ops.runs]


def test_routing_score_float64_recovery_path_agrees_on_a_mixed_tower() -> None:
    import vfe3.relational_scoring as rs
    from vfe3.gradients.relational import relational_score
    ops = make_relational_operators([3, 3, 5], seed=5)
    U = _frames(SPEC, 11).float()
    frame = CanonicalFrameContext(U, torch.linalg.inv(U))
    g = torch.Generator().manual_seed(2)
    mu = torch.randn(1, 3, 11, generator=g)
    sigma = torch.rand(1, 3, 11, generator=g) + 0.5
    healthy = relational_score(mu, sigma, mu, sigma, frame, ops)
    previous = rs.GRAM_CONDITION_TOL
    rs.GRAM_CONDITION_TOL = 1e-30                                 # every float32 entry fails the certificate
    previous_recovery = rs.set_projected_condition_recovery(True)
    before = rs.source_recovery_elements()
    try:
        recovered = relational_score(mu, sigma, mu, sigma, frame, ops)
    finally:
        rs.GRAM_CONDITION_TOL = previous
        rs.set_projected_condition_recovery(previous_recovery)
    assert rs.source_recovery_elements() > before
    torch.testing.assert_close(recovered, healthy, atol=2e-4, rtol=2e-4)


# ---------------------------------------------------------------- moment-matched covariance

def _reference_moment_matched(ops: RelationalOperators, mu: torch.Tensor, Sigma: torch.Tensor,
                              beta: torch.Tensor, U: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
    r"""Sigma_i^+ = U_i C_i^att U_i^T + f Sigma_{p,i} by the defining sums over sources and head pairs (float64
    loops); the prior-anchored floor f Sigma_{p,i} (``prior`` dense, (1, N, K, K)) is added in local coordinates."""
    T_runs, O_runs = ops.content_factor_runs()
    T, O = _per_head(T_runs), _per_head(O_runs)
    M = [o @ t for t, o in zip(T, O)]                                            # (K, K) per head
    H, N, K = len(M), mu.shape[1], mu.shape[-1]
    U_inv = torch.linalg.inv(U[0])
    C = [U_inv[j] @ Sigma[0, j] @ U_inv[j].T for j in range(N)]
    xi = [U_inv[j] @ mu[0, j] for j in range(N)]
    out = []
    for i in range(N):
        b = beta[0, :, i, :]                                                     # (H, N)
        C_att = torch.zeros(K, K, dtype=torch.float64)
        for j in range(N):
            for h in range(H):
                for g in range(H):
                    c = b[h, j] if h == g else b[h, j] * b[g, j]
                    C_att = C_att + c * M[h] @ C[j] @ M[g].T
        for h in range(H):
            n = [M[h] @ xi[j] for j in range(N)]
            mean = sum(b[h, j] * n[j] for j in range(N))
            C_att = C_att + sum(b[h, j] * torch.outer(n[j], n[j]) for j in range(N)) - torch.outer(mean, mean)
        if ops.attention_residual:
            C_att = C_att + C[i] + sum(b[h, i] * (M[h] @ C[i] + C[i] @ M[h].T) for h in range(H))
        out.append(U[0, i] @ C_att @ U[0, i].T + ops.predictive_covariance_floor * prior[0, i])
    return torch.stack(out)[None]


@pytest.mark.parametrize("spec,dims", [(SPEC, [3, 3, 5]), ([("l1", 1), ("l2", 1), ("l1", 1)], [3, 5, 3])])
@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("chart", ["affine", "exponential"])
def test_moment_matched_covariance_matches_the_dense_reference(spec: list, dims: List[int], full: bool,
                                                               chart: str) -> None:
    ops = _operators(dims, predictive_covariance="moment_matched", attention_residual=True,
                     content_link_chart=chart)
    U = _frames(spec, 11)
    frame = CanonicalFrameContext(U, torch.linalg.inv(U))
    g = torch.Generator().manual_seed(3)
    mu = torch.randn(1, 3, 11, generator=g, dtype=torch.float64)
    A = torch.randn(1, 3, 11, 11, generator=g, dtype=torch.float64)
    Sigma = A @ A.transpose(-1, -2) / 11 + 0.5 * torch.eye(11, dtype=torch.float64)
    sigma = Sigma if full else torch.diagonal(Sigma, dim1=-2, dim2=-1)
    beta = torch.randn(1, 3, 3, 3, generator=g, dtype=torch.float64).softmax(-1)
    B = torch.randn(1, 3, 11, 11, generator=torch.Generator().manual_seed(99), dtype=torch.float64)
    P = B @ B.transpose(-1, -2) / 11 + 0.5 * torch.eye(11, dtype=torch.float64)   # the local prior Sigma_p
    prior = P if full else torch.diagonal(P, dim1=-2, dim2=-1)
    got = ops.predictive_covariance_update(mu, sigma, beta, frame, floor_anchor=prior)
    ref = _reference_moment_matched(ops, mu, Sigma if full else torch.diag_embed(sigma), beta, U,
                                    prior if full else torch.diag_embed(prior))
    torch.testing.assert_close(got, ref if full else torch.diagonal(ref, dim1=-2, dim2=-1),
                               atol=1e-10, rtol=1e-10)


def test_predictive_content_matches_the_per_head_sum() -> None:
    ops = _operators([3, 5, 3])
    U = _frames([("l1", 1), ("l2", 1), ("l1", 1)], 11)
    frame = CanonicalFrameContext(U, torch.linalg.inv(U))
    g = torch.Generator().manual_seed(4)
    mu = torch.randn(1, 3, 11, generator=g, dtype=torch.float64)
    beta = torch.randn(1, 3, 3, 3, generator=g, dtype=torch.float64).softmax(-1)
    T_runs, O_runs = ops.content_factor_runs()
    T, O = _per_head(T_runs), _per_head(O_runs)
    xi = torch.linalg.solve(U[0], mu[0].unsqueeze(-1)).squeeze(-1)             # (N, K)
    ref = torch.stack([U[0, i] @ sum(O[h] @ sum(beta[0, h, i, j] * T[h] @ xi[j] for j in range(3))
                                     for h in range(3)) for i in range(3)])[None]
    torch.testing.assert_close(ops.predictive_content(mu, beta, frame), ref, atol=1e-11, rtol=1e-11)


# ---------------------------------------------------------------- projected-gradient receiver

def test_full_link_receiver_matches_the_dense_reference() -> None:
    from vfe3.gradients.full_link import build_full_link_context
    ops = _operators([3, 3, 5], readout_mode="full_link_update", receiver_noise=0.7)
    U = _frames(SPEC, 11)
    frame = CanonicalFrameContext(U, torch.linalg.inv(U))
    g = torch.Generator().manual_seed(5)
    mu = torch.randn(1, 3, 11, generator=g, dtype=torch.float64)
    sigma = torch.rand(1, 3, 11, generator=g, dtype=torch.float64) + 0.5
    mu_p = torch.randn(1, 3, 11, generator=g, dtype=torch.float64)
    sigma_p = torch.rand(1, 3, 11, generator=g, dtype=torch.float64) + 0.5
    lam = torch.tensor([1.0, 0.5, 2.0], dtype=torch.float64)
    context = build_full_link_context(mu, sigma, mu_p, sigma_p, frame, ops, lambda_beta=lam,
                                      c0=0.8, receiver_noise=0.7)
    # alpha*_i = c0/(b0 + KL(q_i||p_i)) at the snapshot, b0 = 1.
    ratio = sigma / sigma_p
    D = 0.5 * (ratio + (mu - mu_p) ** 2 / sigma_p - 1 - ratio.log()).sum(-1)       # (1, N)
    alpha = 0.8 / (1.0 + D)
    T_runs, O_runs = ops.content_factor_runs()
    T, O = _per_head(T_runs), _per_head(O_runs)
    U_inv = torch.linalg.inv(U[0])
    C = [U_inv[j] @ torch.diag(sigma[0, j]) @ U_inv[j].T for j in range(3)]
    xi = [U_inv[j] @ mu[0, j] for j in range(3)]
    for i in range(3):
        P = torch.zeros(11, 11, dtype=torch.float64)
        b = torch.zeros(11, dtype=torch.float64)
        for h in range(3):
            for j in range(3):
                w = lam[h] * context.beta[0, h, i, j]
                R_inv = torch.linalg.inv(T[h] @ C[j] @ T[h].T + 0.7 * torch.eye(T[h].shape[0], dtype=torch.float64))
                P = P + w * O[h] @ R_inv @ O[h].T
                b = b + w * O[h] @ R_inv @ T[h] @ xi[j]
        prior = alpha[0, i] * torch.diag(1.0 / sigma_p[0, i])
        torch.testing.assert_close(context.Lambda[0, i], prior + U_inv[i].T @ P @ U_inv[i], atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(context.eta[0, i], prior @ mu_p[0, i] + U_inv[i].T @ b, atol=1e-10, rtol=1e-10)


# ---------------------------------------------------------------- model routes

def test_per_head_temperatures_follow_the_block_sizes() -> None:
    from vfe3.free_energy import attention_tau
    tau = attention_tau(1.0, _model().group.irrep_dims)
    torch.testing.assert_close(torch.as_tensor(tau, dtype=torch.float64),
                               torch.tensor([3, 3, 5], dtype=torch.float64).sqrt())


@pytest.mark.parametrize("route", sorted(_SON_ROUTES))
def test_mixed_tower_train_step(route: str) -> None:
    from vfe3.train import build_optimizer, train_step
    model = _model(**_SON_ROUTES[route])
    assert model.group.irrep_dims == [3, 3, 5]
    phi0 = model.prior_bank.phi_embed.detach().clone()
    optimizer = build_optimizer(model, model.cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    loss = train_step(model, optimizer, scheduler, torch.tensor([[1, 4, 2]]), torch.tensor([[4, 2, 3]]))
    assert math.isfinite(loss)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(bool(torch.isfinite(g).all()) for g in grads)
    assert all(p.grad is not None for n, p in model.named_parameters() if n.startswith("relational_operators"))
    assert float((model.prior_bank.phi_embed.detach() - phi0).abs().max()) > 0.0


@pytest.mark.parametrize("route", sorted(_SON_ROUTES))
def test_mixed_tower_gauge_equivariance(route: str) -> None:
    from vfe3.viz import extract
    model = _model(**_SON_ROUTES[route]).eval()
    residuals = extract.projected_gauge_equivariance(model, torch.tensor([[1, 4, 2]]), n_draws=2)["residuals"]
    assert residuals
    for key, values in residuals.items():
        assert max(values) < 1e-10, (key, values)


@pytest.mark.parametrize("route", ["predictive_diagonal", "gradient_full_closed_form"])
def test_mixed_tower_phi_gradient_matches_finite_difference(route: str) -> None:
    model = _model(torch.float64, **_SON_ROUTES[route]).eval()
    tokens = torch.tensor([[1, 4, 2]])
    phi = model.prior_bank.phi_embed
    w_mu = torch.randn(1, 3, 11, generator=torch.Generator().manual_seed(3), dtype=torch.float64)

    def objective() -> torch.Tensor:
        belief, _ = model.forward_beliefs(tokens)
        w_sigma = torch.randn(belief.sigma.shape, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
        return (w_mu * belief.mu).sum() + (w_sigma * belief.sigma).sum()

    objective().backward()
    grad = phi.grad.clone()
    h = 1e-5
    for row, col in [(1, 0), (4, 1), (2, 2)]:
        with torch.no_grad():
            phi[row, col] += h
            plus = float(objective())
            phi[row, col] -= 2 * h
            minus = float(objective())
            phi[row, col] += h
        fd = (plus - minus) / (2 * h)
        assert abs(fd - float(grad[row, col])) <= 1e-7 * max(1.0, abs(fd)), (row, col, fd, grad[row, col])


# ---------------------------------------------------------------- records and diagnostics

def test_run_record_counts_and_shapes() -> None:
    from vfe3.run_artifacts import _relational_artifact_fields
    mixed = _relational_artifact_fields(test_config(**MIXED))["relational_operators"]
    assert mixed["per_layer_parameter_counts"] == {"score": 2 * 121, "content": 2 * 121}
    assert mixed["score_rank_per_head"] == [3, 3, 5]
    assert mixed["score_shape"] == [[3, 11], [3, 11], [5, 11]]
    assert mixed["content_shapes"][1] == [[11, 3], [11, 3], [11, 5]]
    equal = _relational_artifact_fields(test_config(embed_dim=6, n_heads=2))["relational_operators"]
    assert equal["per_layer_parameter_counts"] == {"score": 2 * 2 * 6 * 3, "content": 2 * 2 * 6 * 3}
    assert equal["score_rank_per_head"] == 3 and equal["score_shape"] == [2, 3, 6]
    assert equal["content_shapes"] == [[2, 3, 6], [2, 6, 3]]


def test_mixed_link_invariants_reduce_to_the_stacked_path_on_equal_heads() -> None:
    from vfe3 import metrics
    ops = _operators([3, 3])
    T_hat, O_hat = ops._ambient_content_maps(torch.float64)
    beta_self = torch.rand(2, 2, 3, dtype=torch.float64)
    stacked = metrics.content_link_spectrum(T_hat, O_hat, beta_self=beta_self, residual=True)
    as_runs = metrics.content_link_spectrum([T_hat], [O_hat], beta_self=beta_self, residual=True)
    for key, value in stacked.items():
        assert math.isclose(value, as_runs[key], rel_tol=1e-9, abs_tol=1e-12), key


def test_diagnostics_and_scaling_fields_on_a_mixed_tower() -> None:
    import vfe3.run_artifacts as ra
    model = _model()
    d = model.diagnostics(torch.tensor([[1, 4, 2]]))
    assert d["relational_score_rank_per_head"] == pytest.approx(11 / 3)
    assert math.isfinite(d["relational_link_spectral_radius_max"])
    assert math.isfinite(d["relational_projected_cov_max_condition"])
    record = ra._cost_model_fields(model, model.cfg, 1, 0)
    assert record["n_blocks"] == 3


def test_checkpoint_and_config_round_trip() -> None:
    model = _model().eval()
    cfg = config_from_serialized(serialize_config(model.cfg), source="test")
    assert cfg.head_dims == [3, 3, 5]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clone = VFEModel(cfg).eval()
    clone.load_state_dict(model.state_dict())
    tokens = torch.tensor([[1, 4, 2]])
    with torch.no_grad():
        a, _ = model.forward_beliefs(tokens)
        b, _ = clone.forward_beliefs(tokens)
    assert torch.equal(a.mu, b.mu) and torch.equal(a.sigma, b.sigma)


def test_optional_analyses_report_not_applicable() -> None:
    from vfe3.posthoc.covariance import gap_applicability
    from vfe3.posthoc.routing import not_applicable_reason
    model = _model()
    assert "different sizes" in gap_applicability(model)
    assert "different sizes" in not_applicable_reason(model)
