"""Moment-matched predictive covariance (predictive_covariance='moment_matched')."""

from tests.profiles import test_config
import itertools
import re
from dataclasses import asdict, dataclass

import pytest
import torch

from vfe3 import metrics
from vfe3.config import ConfigNotice, VFE3Config, config_from_serialized, serialize_config
from vfe3.contracts import CanonicalFrameContext
from vfe3.gradients.relational import relational_predictive_readout
from vfe3.model.model import VFEModel
from vfe3.model.relational_operators import RelationalOperators
from vfe3.run_artifacts import _relational_artifact_fields

torch.set_num_threads(1)
H, D = 2, 2
K = H * D
N = 3


def _block_dense(blocks: torch.Tensor) -> torch.Tensor:
    """(..., H, d, d) head blocks -> (..., K, K) block-diagonal matrix."""
    out = blocks.new_zeros(*blocks.shape[:-3], K, K)
    for h in range(H):
        out[..., h*D:(h+1)*D, h*D:(h+1)*D] = blocks[..., h, :, :]
    return out


def _setup(input_scope: str, output_scope: str, residual: bool, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    randn = lambda *shape: torch.randn(*shape, generator=generator, dtype=torch.float64)
    ops = RelationalOperators([D] * H, 11, input_scope=input_scope, readout_mode="predictive_content",
                              output_scope=output_scope, attention_residual=residual,
                              predictive_covariance="moment_matched",
                              predictive_covariance_floor=1e-3).double()
    with torch.no_grad():
        ops.T.copy_(randn(*ops.T.shape))
        ops.O.copy_(randn(*ops.O.shape))
    blocks = torch.eye(D, dtype=torch.float64) + 0.4 * randn(1, N, H, D, D)
    mu = randn(1, N, K)
    factor = randn(1, N, K, K)
    sigma_full = factor @ factor.transpose(-1, -2) + 0.5 * torch.eye(K, dtype=torch.float64)
    sigma_diag = torch.rand(1, N, K, generator=generator, dtype=torch.float64) + 0.3
    logits = randn(1, H, N, N)
    causal = torch.ones(N, N, dtype=torch.bool).tril()
    beta = torch.softmax(logits.masked_fill(~causal, -torch.inf), dim=-1)
    return ops, blocks, mu, sigma_full, sigma_diag, beta


def _frame(blocks: torch.Tensor, compact: bool) -> CanonicalFrameContext:
    inverse = torch.linalg.inv(blocks)
    if compact:
        return CanonicalFrameContext(blocks, inverse, block_diagonal=True)
    return CanonicalFrameContext(_block_dense(blocks), _block_dense(inverse))


def _content_maps(ops: RelationalOperators) -> list:
    """Each head's K x K content map M_h = O_h T_h, built independently of the implementation."""
    maps = []
    for h in range(H):
        T_hat = ops.T.new_zeros(D, K)
        O_hat = ops.O.new_zeros(K, D)
        if ops.input_scope == "full":
            T_hat = ops.T[h].detach()
        else:
            T_hat[:, h*D:(h+1)*D] = ops.T[h].detach()
        if ops.output_scope == "full":
            O_hat = ops.O[h].detach()
        else:
            O_hat[h*D:(h+1)*D, :] = ops.O[h].detach()
        maps.append(O_hat @ T_hat)
    return maps


def _enumerated_moments(ops, blocks, mu, sigma, beta, residual, floor):
    """Exact mean and covariance of the declared kernel by enumerating every head-label tuple.

    X = r k_i + sum_h L_{i J^h} k_{J^h} + eps, k_s ~ N(mu_s, Sigma_s) independent,
    J^h ~ beta_i^h independent, eps ~ N(0, floor (tr C_i / K) U_i U_i^T), L_ij^h = U_i M_h U_j^{-1}.
    """
    U = _block_dense(blocks[0])
    U_inv = torch.linalg.inv(U)
    maps = _content_maps(ops)
    means, covariances = [], []
    for i in range(N):
        first = torch.zeros(K, dtype=torch.float64)
        second = torch.zeros(K, K, dtype=torch.float64)
        within = torch.zeros(K, K, dtype=torch.float64)
        for labels in itertools.product(range(N), repeat=H):
            weight = torch.prod(torch.stack([beta[0, h, i, j] for h, j in enumerate(labels)]))
            if weight == 0:
                continue
            A = {s: torch.zeros(K, K, dtype=torch.float64) for s in range(N)}
            if residual:
                A[i] = A[i] + torch.eye(K, dtype=torch.float64)
            for h, j in enumerate(labels):
                A[j] = A[j] + U[i] @ maps[h] @ U_inv[j]
            mean = sum(A[s] @ mu[0, s] for s in range(N))
            cov = sum(A[s] @ sigma[0, s] @ A[s].T for s in range(N))
            first = first + weight * mean
            second = second + weight * torch.outer(mean, mean)
            within = within + weight * cov
        means.append(first)
        C_i = U_inv[i] @ sigma[0, i] @ U_inv[i].T
        noise = floor * torch.trace(C_i) / K * (U[i] @ U[i].T)
        covariances.append(within + second - torch.outer(first, first) + noise)
    return torch.stack(means)[None], torch.stack(covariances)[None]


@pytest.mark.parametrize("input_scope", ["head", "full"])
@pytest.mark.parametrize("output_scope", ["head", "full"])
@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("compact", [True, False])
def test_covariance_matches_enumerated_predictive_kernel(input_scope, output_scope, residual, compact):
    ops, blocks, mu, sigma_full, sigma_diag, beta = _setup(input_scope, output_scope, residual)
    frame = _frame(blocks, compact)
    for sigma in (sigma_full, sigma_diag):
        dense = sigma if sigma.dim() == 4 else torch.diag_embed(sigma)
        mean_ref, cov_ref = _enumerated_moments(ops, blocks, mu, dense, beta, residual, floor=1e-3)
        actual = ops.predictive_covariance_update(mu, sigma, beta, frame)
        expected = cov_ref if sigma.dim() == 4 else torch.diagonal(cov_ref, dim1=-2, dim2=-1)
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
        content = ops.predictive_content(mu, beta, frame) + (mu if residual else 0.0)
        torch.testing.assert_close(content, mean_ref, atol=1e-10, rtol=1e-10)


def test_cross_head_terms_are_present_for_full_covariance() -> None:
    ops, blocks, mu, sigma_full, _, beta = _setup("full", "full", False)
    frame = _frame(blocks, True)
    _, cov_ref = _enumerated_moments(ops, blocks, mu, sigma_full, beta, False, floor=1e-3)
    # Dropping the h != h' terms (independent-heads approximation) is measurably wrong here.
    maps = _content_maps(ops)
    U = _block_dense(blocks[0])
    U_inv = torch.linalg.inv(U)
    i = N - 1
    cross = sum(beta[0, 0, i, j] * beta[0, 1, i, j] * (
        U[i] @ maps[0] @ U_inv[j] @ sigma_full[0, j] @ (U[i] @ maps[1] @ U_inv[j]).T) for j in range(N))
    assert (cross + cross.T).abs().max() > 1e-3
    torch.testing.assert_close(ops.predictive_covariance_update(mu, sigma_full, beta, frame), cov_ref,
                               atol=1e-10, rtol=1e-10)


def test_passive_frame_law() -> None:
    """U -> gU, mu -> g mu, Sigma -> g Sigma g^T maps Sigma^+ -> g Sigma^+ g^T at fixed beta."""
    ops, blocks, mu, sigma_full, _, beta = _setup("full", "full", True, seed=3)
    g_blocks = torch.eye(D, dtype=torch.float64) + 0.5 * torch.randn(
        1, N, H, D, D, generator=torch.Generator().manual_seed(9), dtype=torch.float64)
    g = _block_dense(g_blocks)
    before = ops.predictive_covariance_update(mu, sigma_full, beta, _frame(blocks, True))
    after = ops.predictive_covariance_update(
        (g @ mu.unsqueeze(-1)).squeeze(-1), g @ sigma_full @ g.transpose(-1, -2), beta,
        _frame(g_blocks @ blocks, True))
    torch.testing.assert_close(after, g @ before @ g.transpose(-1, -2), atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("residual", [False, True])
def test_positive_definite_with_sharp_routing_and_rank_deficient_maps(residual) -> None:
    ops, blocks, mu, sigma_full, _, _ = _setup("head", "head", residual, seed=5)
    with torch.no_grad():
        ops.O.zero_()                           # every content map M_h = 0: Cov(m) is exactly zero
    beta = torch.zeros(1, H, N, N, dtype=torch.float64)
    beta[..., torch.arange(N), torch.arange(N)] = 1.0
    out = ops.predictive_covariance_update(mu, sigma_full, beta, _frame(blocks, True))
    assert torch.linalg.eigvalsh(out).min() > 0
    U = _block_dense(blocks)
    U_inv = torch.linalg.inv(U)
    scale = torch.diagonal(U_inv @ sigma_full @ U_inv.transpose(-1, -2), dim1=-2, dim2=-1).mean(-1)
    expected = (1.0 if residual else 0.0) * sigma_full + 1e-3 * scale[..., None, None] * (U @ U.transpose(-1, -2))
    torch.testing.assert_close(out, expected, atol=1e-12, rtol=1e-10)


def test_depth_recursion_stays_well_conditioned_in_float32() -> None:
    """Regression (2026-09-27 sweep): rank-one content maps and no residual over many layers.

    A relative floor floor * Sigma_i compounds as floor**depth and lost float32 definiteness by the
    third layer; the trace-scaled reference floor keeps the condition number bounded.
    """
    torch.manual_seed(0)
    Hq, Dq = 4, 2
    ops = RelationalOperators([Dq] * Hq, 3, input_scope="head", readout_mode="predictive_content",
                              output_scope="full", predictive_covariance="moment_matched")
    with torch.no_grad():
        ops.O.copy_(torch.randn(Hq, Hq * Dq, 1) @ torch.randn(Hq, 1, Dq))    # rank-one O_h
    n = 6
    blocks = torch.eye(Dq) + 0.2 * torch.randn(1, n, Hq, Dq, Dq)
    frame = CanonicalFrameContext(blocks, torch.linalg.inv(blocks), block_diagonal=True)
    mu = torch.randn(1, n, Hq * Dq)
    sigma = torch.diag_embed(torch.rand(1, n, Hq * Dq) + 0.5)
    causal = torch.ones(n, n, dtype=torch.bool).tril()
    beta = torch.softmax(torch.randn(1, Hq, n, n).masked_fill(~causal, -torch.inf), dim=-1)
    for _ in range(12):
        sigma = ops.predictive_covariance_update(mu, sigma, beta, frame)
        eigenvalues = torch.linalg.eigvalsh(sigma.double())
        assert eigenvalues.min() > 0
        assert (eigenvalues.max(-1).values / eigenvalues.min(-1).values).max() < 1e7
        torch.linalg.cholesky(sigma)


def test_gradcheck_inputs_and_content_maps() -> None:
    ops, blocks, mu, _, sigma_diag, beta = _setup("full", "head", True, seed=7)
    frame = _frame(blocks, True)
    mu = mu.clone().requires_grad_(True)
    sigma = sigma_diag.clone().requires_grad_(True)
    beta = beta.clone().requires_grad_(True)
    T = ops.T.detach().clone().requires_grad_(True)
    O = ops.O.detach().clone().requires_grad_(True)

    def _call(mu_in, sigma_in, beta_in, T_in, O_in):
        # Read the content maps from the gradcheck leaves by swapping the parameter slots.
        original = ops._parameters["T"], ops._parameters["O"]
        try:
            ops._parameters["T"], ops._parameters["O"] = T_in, O_in
            return ops.predictive_covariance_update(mu_in, sigma_in, beta_in, frame)
        finally:
            ops._parameters["T"], ops._parameters["O"] = original

    assert torch.autograd.gradcheck(_call, (mu, sigma, beta, T, O))


def _readout_fixture():
    mu = torch.tensor([[[0.2, -0.4, 0.1, 0.7], [1.1, 0.5, -0.3, 0.8],
                        [0.6, 1.2, 0.9, -0.2]]], dtype=torch.float64, requires_grad=True)
    sigma = torch.tensor([[[0.8, 1.2, 0.9, 1.4], [1.1, 0.6, 1.5, 0.7],
                           [0.7, 1.3, 0.8, 1.1]]], dtype=torch.float64)
    U = torch.eye(4, dtype=torch.float64).expand(1, 3, 4, 4).clone()
    U[..., 0, 1] = 0.3
    U[..., 2, 3] = -0.2
    frame = CanonicalFrameContext(U, torch.linalg.inv(U))
    prior = torch.tensor([[0., -torch.inf, -torch.inf], [0., 0., -torch.inf],
                          [0., 0., 0.]], dtype=torch.float64)
    return mu, sigma, frame, prior


def test_readout_returns_the_same_mean_and_the_operator_covariance() -> None:
    mu, sigma, frame, prior = _readout_fixture()
    ops = RelationalOperators([2, 2], 11, input_scope="full", readout_mode="predictive_content",
                              output_scope="full", attention_residual=True,
                              predictive_covariance="moment_matched").double()
    capture = {}
    mean_only = relational_predictive_readout(mu, sigma, frame, ops, tau=.9, log_prior=prior)
    mean, sigma_plus = relational_predictive_readout(
        mu, sigma, frame, ops, tau=.9, log_prior=prior, capture=capture, return_covariance=True)
    torch.testing.assert_close(mean, mean_only, atol=0, rtol=0)
    expected = ops.predictive_covariance_update(mu, sigma, capture["beta_pre"], frame)
    torch.testing.assert_close(sigma_plus, expected, atol=1e-12, rtol=1e-12)
    assert sigma_plus.shape == sigma.shape
    grads = torch.autograd.grad(sigma_plus.sum(), [mu, ops.T, ops.O, ops.raw_score_q])
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_operator_validation() -> None:
    with pytest.raises(ValueError, match="predictive_covariance must be"):
        RelationalOperators([2, 2], 1, readout_mode="predictive_content", predictive_covariance="full")
    with pytest.raises(ValueError, match="requires predictive_content"):
        RelationalOperators([2, 2], 1, predictive_covariance="moment_matched")
    for bad in (0.0, -1.0, float("nan"), True):
        with pytest.raises(ValueError, match="predictive_covariance_floor"):
            RelationalOperators([2, 2], 1, readout_mode="predictive_content",
                                predictive_covariance="moment_matched", predictive_covariance_floor=bad)


def _cfg(**overrides: object) -> VFE3Config:
    values = dict(vocab_size=7, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=2, n_e_steps=1,
                  family="gaussian_diagonal", gauge_group="block_glk",
                  skip_belief_sigma_update=False, e_phi_lr=0.0, oracle_unroll_grad=True,
                  warmup_steps=0, max_steps=2, log_interval=0, eval_interval=0,
                  attention_mode="projected_predictive", decode_mode="linear")
    values.update(overrides)
    return test_config(**values)


def test_config_validation_and_serialization(monkeypatch: pytest.MonkeyPatch) -> None:
    assert test_config(predictive_covariance="passthrough").predictive_covariance == "passthrough"
    for mode in ("transported", "projected_conditional"):
        with pytest.raises(ValueError, match=re.escape(
                "predictive_covariance='moment_matched' requires attention_mode='projected_predictive'")):
            test_config(attention_mode=mode, predictive_covariance="moment_matched")
    with pytest.raises(ValueError, match="predictive_covariance"):
        _cfg(predictive_covariance="full")
    for bad in (0.0, -1e-3, float("inf"), True):
        with pytest.raises(ValueError, match="predictive_covariance_floor"):
            _cfg(predictive_covariance="moment_matched", predictive_covariance_floor=bad)
    cfg = _cfg(predictive_covariance="moment_matched", predictive_covariance_floor=1e-3)
    assert asdict(config_from_serialized(serialize_config(cfg), source="round trip")) == asdict(cfg)
    legacy = serialize_config(_cfg())
    legacy.pop("predictive_covariance")
    legacy.pop("predictive_covariance_floor")
    @dataclass
    class LegacyFixtureConfig(VFE3Config):
        predictive_covariance: str = "passthrough"
        predictive_covariance_floor: float = 1e-4

    monkeypatch.setattr("vfe3.config.VFE3Config", LegacyFixtureConfig)
    restored = config_from_serialized(legacy, source="legacy")
    assert (restored.predictive_covariance, restored.predictive_covariance_floor) == ("passthrough", 1e-4)


def test_inactive_field_reporting() -> None:
    with pytest.warns(ConfigNotice, match="predictive_covariance_floor"):
        cfg = _cfg(predictive_covariance_floor=1e-3)
    assert "predictive_covariance_floor" in cfg.inactive_fields()
    live = _cfg(predictive_covariance="moment_matched", predictive_covariance_floor=1e-3)
    assert "predictive_covariance_floor" not in live.inactive_fields()
    assert "predictive_covariance" not in live.inactive_fields()
    full = _cfg(family="gaussian_full", predictive_covariance="moment_matched")
    assert "family" not in full.inactive_fields()


def test_frozen_covariance_label_and_artifact_fields() -> None:
    assert metrics.covariance_frozen_at_encode(_cfg())
    live = _cfg(predictive_covariance="moment_matched")
    assert not metrics.covariance_frozen_at_encode(live)
    fields = _relational_artifact_fields(live)["relational_operators"]
    assert fields["covariance_update"] == "moment_matched"
    assert fields["content_covariance_contract"] == "exact_predictive_kernel_moment_match_plus_covariant_floor"
    assert _relational_artifact_fields(_cfg())["relational_operators"]["covariance_update"] == "passthrough"


@pytest.mark.parametrize("family,decode_mode", [
    ("gaussian_diagonal", "linear"),
    ("gaussian_diagonal", "diagonal_chunked"),
    ("gaussian_diagonal_exact", "linear"),
    ("gaussian_frame_diagonal", "linear"),
    ("gaussian_full", "linear"),
    ("gaussian_full", "full_chunked"),
])
@pytest.mark.parametrize("residual", [False, True])
def test_model_forward_backward_and_evolving_covariance(family, decode_mode, residual) -> None:
    tokens = torch.tensor([[1, 2, 3, 4]])
    targets = torch.tensor([[2, 3, 4, 5]])
    overrides = dict(family=family, decode_mode=decode_mode, attention_residual=residual,
                     relational_input_scope="full", relational_output_scope="full")
    torch.manual_seed(0)
    base = VFEModel(_cfg(**overrides))
    torch.manual_seed(0)
    model = VFEModel(_cfg(predictive_covariance="moment_matched", **overrides))
    model.load_state_dict(base.state_dict())
    _, loss, _ = model(tokens, targets)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    with torch.no_grad():
        evolved, _ = model.forward_beliefs(tokens)
        frozen, _ = base.forward_beliefs(tokens)
        encoded = base.prior_bank.encode(tokens).sigma
    # Passthrough returns the encoded covariance; moment matching moves it at every layer.
    torch.testing.assert_close(frozen.sigma, encoded.to(frozen.sigma.dtype))
    assert (evolved.sigma - frozen.sigma).abs().max() > 1e-4
