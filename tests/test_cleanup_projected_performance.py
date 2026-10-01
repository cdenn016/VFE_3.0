"""Tiny CPU equivalence and memory contracts for projected-path cleanup."""
from tests.profiles import test_config
from typing import Any

import pytest
import torch

import vfe3.relational_scoring as scoring
import vfe3.model.stack as stack_module
from vfe3.config import VFE3Config
from vfe3.contracts import CanonicalFrameContext
from vfe3.model.model import VFEModel
from vfe3.model.relational_operators import RelationalOperators


def _moments(full: bool, compact: bool, dtype: torch.dtype = torch.float64) -> tuple:
    generator = torch.Generator().manual_seed(49)
    mu = torch.randn(1, 4, 2, 4, generator=generator, dtype=dtype).requires_grad_()
    sigma = torch.rand(mu.shape, generator=generator, dtype=dtype) + 1.
    if full:
        root = torch.randn(1, 4, 2, 4, 4, generator=generator, dtype=dtype) * .1
        sigma = torch.diag_embed(sigma) + root @ root.transpose(-1, -2)
    sigma.requires_grad_()
    size = 2 if compact else 4
    inverse = (torch.eye(size, dtype=dtype).expand(1, 4, 2, size, size)
               + .03 * torch.randn(1, 4, 2, size, size, generator=generator, dtype=dtype)).requires_grad_()
    S = torch.randn(2, 2, 4, generator=generator, dtype=dtype).requires_grad_()
    return mu, sigma, inverse, S


def _dense_moments(mu: torch.Tensor, sigma: torch.Tensor, inverse: torch.Tensor,
                   S: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    L = scoring._projected_linear_map(S, inverse)
    a = (L @ mu.unsqueeze(-1)).squeeze(-1)
    G = ((L * sigma.unsqueeze(-2)) @ L.transpose(-1, -2) if sigma.shape == mu.shape
         else L @ sigma @ L.transpose(-1, -2))
    return a, .5 * G + .5 * G.transpose(-1, -2)


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tiled_moments_outputs_gradients_and_hvp(monkeypatch: pytest.MonkeyPatch,
                                               full: bool, compact: bool, dtype: torch.dtype) -> None:
    monkeypatch.setattr(scoring, "_PROJECTION_TILE_BYTES", 1)
    inputs = _moments(full, compact, dtype)
    actual = scoring.projected_moments(*inputs)
    expected = _dense_moments(*inputs)
    tolerance = 2e-5 if dtype == torch.float32 else 1e-10
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=tolerance, rtol=tolerance)
    actual_grad = torch.autograd.grad(sum(x.square().sum() for x in actual), inputs, create_graph=True)
    expected_grad = torch.autograd.grad(sum(x.square().sum() for x in expected), inputs, create_graph=True)
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b, atol=tolerance, rtol=tolerance)
    actual_hvp = torch.autograd.grad(sum(x.sum() for x in actual_grad), inputs)
    expected_hvp = torch.autograd.grad(sum(x.sum() for x in expected_grad), inputs)
    for a, b in zip(actual_hvp, expected_hvp):
        torch.testing.assert_close(a, b, atol=tolerance * 10, rtol=tolerance)


@pytest.mark.parametrize("full", [False, True])
def test_checkpointed_projection_saves_no_map_storage(monkeypatch: pytest.MonkeyPatch,
                                                      full: bool) -> None:
    monkeypatch.setattr(scoring, "_PROJECTION_TILE_BYTES", 1)
    inputs = _moments(full, True)
    input_storages = {x.untyped_storage().data_ptr() for x in inputs}
    saved = []
    def pack(value: torch.Tensor) -> torch.Tensor:
        if value.untyped_storage().data_ptr() not in input_storages and value.numel():
            saved.append(value)
        return value
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
        _dense_moments(*inputs)
    assert sum(x.numel() * x.element_size() for x in saved) > 0
    saved.clear()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
        values = scoring.projected_moments(*inputs)
    assert saved == []
    assert all(torch.isfinite(x).all() for x in torch.autograd.grad(sum(v.sum() for v in values), inputs))


@pytest.mark.parametrize("full", [False, True])
def test_tiled_projection_keeps_source_factor_recovery(monkeypatch: pytest.MonkeyPatch,
                                                      full: bool) -> None:
    from tests.test_relational_source_factorization import _inputs
    monkeypatch.setattr(scoring, "_PROJECTION_TILE_BYTES", 1)
    mu, sigma, inverse, S = _inputs()
    if full:
        sigma = torch.diag_embed(sigma)
    S.requires_grad_()
    inverse.requires_grad_()
    sigma.requires_grad_()
    identity = torch.eye(3)[None]
    aq, Gq = scoring.projected_moments(mu, sigma, inverse, S)
    ak, Gk = scoring.projected_moments(mu, sigma, inverse, identity)
    actual = scoring.projected_gaussian_kl(aq, Gq, ak, Gk,
                                         projection_q=(sigma, inverse, S, mu),
                                         projection_k=(sigma, inverse, identity, mu))
    aq64, Gq64 = _dense_moments(mu.double(), sigma.double(), inverse.double(), S.double())
    ak64, Gk64 = _dense_moments(mu.double(), sigma.double(), inverse.double(), identity.double())
    expected = scoring.projected_gaussian_kl(aq64, Gq64, ak64, Gk64)
    torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=2e-5)
    for a, b in zip(torch.autograd.grad(actual.sum(), (S, sigma, inverse), retain_graph=True),
                    torch.autograd.grad(expected.sum(), (S, sigma, inverse))):
        torch.testing.assert_close(a, b, rtol=5e-4, atol=5e-3)


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("input_scope", ["full"])
def test_full_output_contraction_matches_expanded_reference(compact: bool, input_scope: str) -> None:
    torch.manual_seed(83)
    operators = RelationalOperators([2, 2], 9,
                                    readout_mode="predictive_content").double()
    with torch.no_grad():
        operators.O.add_(torch.randn_like(operators.O) * .1)
    mu = torch.randn(1, 3, 4, dtype=torch.float64).requires_grad_()
    beta = torch.randn(1, 2, 3, 3, dtype=torch.float64).softmax(-1).requires_grad_()
    frames = torch.eye(2 if compact else 4, dtype=torch.float64)
    shape = (1, 3, 2, 2, 2) if compact else (1, 3, 4, 4)
    frame = CanonicalFrameContext(frames.expand(shape), frames.expand(shape), compact)
    actual = operators.predictive_content(mu, beta, frame)
    canonical = operators.canonical_mean(mu, frame)
    coordinates = canonical.flatten(-2).unsqueeze(-2) if input_scope == "full" else canonical
    values = (operators.T @ coordinates.unsqueeze(-1)).squeeze(-1)
    pooled = torch.einsum("...hij,...jhs->...ihs", beta, values)
    expected = (operators.O @ pooled.unsqueeze(-1)).squeeze(-1).sum(-2)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    variables = (mu, beta, operators.T, operators.O)
    for a, b in zip(torch.autograd.grad(actual.square().sum(), variables),
                    torch.autograd.grad(expected.square().sum(), variables)):
        torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-11)


def _configuration(**overrides: Any) -> VFE3Config:
    settings = dict(vocab_size=7, embed_dim=4, n_heads=2, max_seq_len=3,
                    n_layers=2, n_e_steps=1, family="gaussian_diagonal",
                    gauge_group="block_glk", skip_belief_sigma_update=True,
                    e_phi_lr=0.0, oracle_unroll_grad=True, warmup_steps=0,
                    max_steps=2, log_interval=0, eval_interval=0,
                    decode_temperature_learned=False,
                    use_head_mixer=False, use_block_mlp=False,
                    attention_mode="projected_predictive",
                    prior_handoff_rho=.7, prior_handoff_sigma=.4)
    settings.update(overrides)
    return test_config(**settings)


@pytest.mark.parametrize("family", ["gaussian_diagonal", "gaussian_full"])
@pytest.mark.parametrize("gradient_mode", ["unroll"])
def test_omitted_handoff_preserves_model_outputs_and_gradients(monkeypatch: pytest.MonkeyPatch,
                                                             family: str, gradient_mode: str) -> None:
    torch.manual_seed(54)
    model = VFEModel(_configuration(family=family, e_step_gradient=gradient_mode))
    assert model.prior_bank.decode_log_scale.requires_grad is False
    ids = torch.tensor([[0, 1, 2]])
    targets = torch.tensor([[1, 2, 3]])
    admissions = []
    original = stack_module._can_omit_prior_handoff
    def record(*args: Any) -> bool:
        result = original(*args)
        admissions.append(result)
        return result
    monkeypatch.setattr(stack_module, "_can_omit_prior_handoff", record)
    actual = model(ids, targets)[1]
    assert admissions == [True]
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    actual_grads = torch.autograd.grad(actual, parameters, allow_unused=True)
    monkeypatch.setattr(stack_module, "_can_omit_prior_handoff", lambda *args: False)
    expected = model(ids, targets)[1]
    expected_grads = torch.autograd.grad(expected, parameters, allow_unused=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for a, b in zip(actual_grads, expected_grads):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_handoff_guard_requires_unobserved_all_predictive_layers() -> None:
    cfg = _configuration()
    predictive = RelationalOperators([2, 2], 2, readout_mode="predictive_content")
    gradient = RelationalOperators([2, 2], 2, readout_mode="full_link_update")
    admit = stack_module._can_omit_prior_handoff
    assert admit(cfg, None, None, None, predictive, None)
    assert admit(cfg, None, None, None, None, [predictive, predictive])
    assert not admit(cfg, None, None, None, None, [predictive, gradient])
    assert not admit(cfg, None, None, None, None, None)
    assert not admit(cfg, {}, None, None, predictive, None)
    assert not admit(cfg, None, {}, None, predictive, None)
    assert not admit(cfg, None, None, {}, predictive, None)
    cfg.mstep_self_coupling_weight = .1
    assert not admit(cfg, None, None, None, predictive, None)
    cfg.mstep_self_coupling_weight = 0.
    cfg.cg_energy_weight = .1
    assert not admit(cfg, None, None, None, predictive, None)


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("compact", [False, True])
def test_tiled_projection_broadcast_frames_and_no_grad(monkeypatch: pytest.MonkeyPatch,
                                                       full: bool, compact: bool) -> None:
    monkeypatch.setattr(scoring, "_PROJECTION_TILE_BYTES", 1)
    mu, sigma, inverse, S = _moments(full, compact)
    inverse = inverse[0, 0]  # A shared frame has no token or batch axes to slice.
    with torch.no_grad():
        actual = scoring.projected_moments(mu, sigma, inverse, S)
        expected = _dense_moments(mu, sigma, inverse, S)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)


def test_captured_stack_retains_handoff_values() -> None:
    cfg = _configuration()
    model = VFEModel(cfg)
    capture = {"diagnostic": {}}
    model.forward_beliefs(torch.tensor([[0, 1, 2]]), capture=capture)
    initial_mu, initial_sigma = capture["diagnostic"]["layer_priors"][0]
    first = capture["diagnostic"]["layer_outputs"][0]
    expected_mu = (1 - cfg.prior_handoff_rho) * initial_mu + cfg.prior_handoff_rho * first.mu
    expected_sigma = ((1 - cfg.prior_handoff_sigma) * initial_sigma
                      + cfg.prior_handoff_sigma * first.sigma)
    actual_mu, actual_sigma = capture["final_block_prior"]
    torch.testing.assert_close(actual_mu, expected_mu, rtol=0, atol=0)
    torch.testing.assert_close(actual_sigma, expected_sigma, rtol=0, atol=0)


def test_self_coupling_loss_retains_handoff(monkeypatch: pytest.MonkeyPatch) -> None:
    model = VFEModel(_configuration(mstep_self_coupling_weight=.2))
    assert model.prior_bank.decode_log_scale.requires_grad is False
    admissions = []
    original = stack_module._can_omit_prior_handoff
    def record(*args: Any) -> bool:
        result = original(*args)
        admissions.append(result)
        return result
    monkeypatch.setattr(stack_module, "_can_omit_prior_handoff", record)
    loss = model(torch.tensor([[0, 1, 2]]), torch.tensor([[1, 2, 3]]))[1]
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    assert admissions == [False]
    assert torch.isfinite(loss)
    assert all(g is None or torch.isfinite(g).all() for g in gradients)


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("unbatched", [False, True])
def test_full_input_router_tiling_autocast_and_recomputed_gradients(
    monkeypatch: pytest.MonkeyPatch, full: bool, compact: bool, unbatched: bool,
) -> None:
    from vfe3.gradients.relational import relational_score
    torch.manual_seed(37)
    operators = RelationalOperators([2, 2], 9)
    mu = torch.randn(1, 3, 4) * .1
    sigma = torch.rand(1, 3, 4) + 1.
    if full:
        sigma = torch.diag_embed(sigma)
    inverse = (torch.eye(2).expand(1, 3, 2, 2, 2).clone() if compact
               else torch.eye(4).expand(1, 3, 4, 4).clone())
    if unbatched:
        mu, sigma, inverse = mu[0], sigma[0], inverse[0]
    mu.requires_grad_()
    sigma.requires_grad_()
    inverse.requires_grad_()
    frame = CanonicalFrameContext(inverse, inverse, compact)
    inputs = (mu, sigma, inverse, operators.raw_score_q, operators.raw_score_k)
    monkeypatch.setattr(scoring, "_PROJECTION_TILE_BYTES", 1)
    tile_shapes = []
    original = scoring._projected_moments_tile
    def record(*args: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tile_shapes.append(tuple(args[0].shape))
        return original(*args)
    monkeypatch.setattr(scoring, "_projected_moments_tile", record)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = relational_score(mu, sigma, mu, sigma, frame, operators)
    # Self-scoring batches both comparison maps into one projected_moments call.
    assert len(tile_shapes) == 3
    assert all(shape[-3] == 1 for shape in tile_shapes)
    actual_grad = torch.autograd.grad(actual.square().sum(), inputs)
    assert len(tile_shapes) > 3  # Backward actually recomputed checkpointed tiles.
    monkeypatch.setattr(scoring, "_PROJECTION_TILE_BYTES", 1024 ** 3)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected = relational_score(mu, sigma, mu, sigma, frame, operators)
    expected_grad = torch.autograd.grad(expected.square().sum(), inputs)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-4)
