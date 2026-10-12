"""Receiving-map charts of the content link (content_link_chart, 2026-10-03).

Tiny CPU tensors and models only (K=4, two heads, N=4, vocab 7). Covers the chart identities
(first order O' = T^{-1} + O; exponential O'T = e^{OT}, T O' = e^{TO} with positive determinant),
phi_1 against its series, equal initial content in every chart, gradient checks of the chart maps,
short training runs on both projected routes, the post-hoc map readers, and config validation.
"""
from tests.profiles import make_relational_operators
from tests.profiles import test_config
import logging
import math
from typing import Any, Dict

import pytest
import torch
from torch.utils.data import DataLoader

import vfe3.run_artifacts as ra
import vfe3.train as T
from vfe3.data.datasets import TokenWindows
from vfe3.model.model import VFEModel
from vfe3.model.relational_operators import (CONTENT_LINK_CHARTS, RelationalOperators, _phi1,
                                             chart_receive_map)
from vfe3.posthoc.maps import map_invariants, relational_maps

B, N, V = 2, 4, 7
H, D = 2, 2
K = H * D


def _stack_T(T_: torch.Tensor) -> torch.Tensor:          # (H, d, K) -> (Hd, K)
    return T_.reshape(-1, T_.shape[-1])


def _stack_O(O_: torch.Tensor) -> torch.Tensor:          # (H, K, d) -> (K, Hd)
    return O_.permute(1, 0, 2).reshape(O_.shape[1], -1)


def _random_factors(seed: int) -> tuple:
    g = torch.Generator().manual_seed(seed)
    T_ = torch.eye(K, dtype=torch.float64).reshape(H, D, K) + 0.3 * torch.randn(H, D, K, generator=g,
                                                                                dtype=torch.float64)
    O_ = 0.5 * torch.randn(H, K, D, generator=g, dtype=torch.float64)
    return T_, O_


@pytest.fixture(autouse=True)
def _quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T, "_save_eval_attention_maps", lambda *args, **kwargs: None)
    monkeypatch.setattr(ra, "_run_figures_isolated", lambda *args, **kwargs: True)
    torch.set_num_threads(1)


def test_phi1_matches_its_series() -> None:
    g = torch.Generator().manual_seed(0)
    A = 0.7 * torch.randn(K, K, generator=g, dtype=torch.float64)
    series = torch.zeros_like(A)
    term = torch.eye(K, dtype=torch.float64)
    for n in range(1, 30):
        series = series + term / float(math.factorial(n))
        term = term @ A
    torch.testing.assert_close(_phi1(A), series, rtol=1e-12, atol=1e-12)


def test_chart_identities() -> None:
    T_, O_ = _random_factors(1)
    assert chart_receive_map(T_, O_, "affine") is O_
    Ts, Os = _stack_T(T_), _stack_O(O_)
    first = _stack_O(chart_receive_map(T_, O_, "first_order"))
    torch.testing.assert_close(first, torch.linalg.inv(Ts) + Os, rtol=1e-12, atol=1e-12)
    exp = _stack_O(chart_receive_map(T_, O_, "exponential"))
    torch.testing.assert_close(exp @ Ts, torch.linalg.matrix_exp(Os @ Ts), rtol=1e-10, atol=1e-10)
    transfer = Ts @ exp
    torch.testing.assert_close(transfer, torch.linalg.matrix_exp(Ts @ Os), rtol=1e-10, atol=1e-10)
    assert float(torch.linalg.det(transfer)) > 0.0


def test_exponential_transfer_stays_positive_where_affine_flips_sign() -> None:
    T_ = torch.eye(K, dtype=torch.float64).reshape(H, D, K)
    O_ = torch.eye(K, dtype=torch.float64).reshape(K, H, D).permute(1, 0, 2).clone()
    O_[0, 0, 0] = -1.0                                                  # affine O'T has det < 0
    assert float(torch.linalg.det(_stack_O(O_) @ _stack_T(T_))) < 0.0
    exp = _stack_O(chart_receive_map(T_, O_, "exponential"))
    assert float(torch.linalg.det(_stack_T(T_) @ exp)) > 0.0


@pytest.mark.parametrize("scale", [1.0, 1.0 / math.sqrt(8.0)])
def test_every_chart_starts_at_the_same_content(scale: float) -> None:
    maps = {}
    for chart in CONTENT_LINK_CHARTS:
        ops = make_relational_operators([D] * H, seed=3, output_init_scale=scale, content_link_chart=chart)
        T_, receive = ops.content_factors()
        maps[chart] = (_stack_O(receive.double()) @ _stack_T(T_.double()), ops.raw_score_q.detach().clone())
    for chart in CONTENT_LINK_CHARTS:
        torch.testing.assert_close(maps[chart][0], scale * torch.eye(K, dtype=torch.float64),
                                   rtol=0, atol=1e-6)
        torch.testing.assert_close(maps[chart][1], maps["affine"][1], rtol=0, atol=0)


@pytest.mark.parametrize("chart", ["first_order", "exponential"])
def test_chart_maps_pass_gradcheck(chart: str) -> None:
    T_, O_ = _random_factors(5)
    T_.requires_grad_(True)
    O_.requires_grad_(True)
    assert torch.autograd.gradcheck(lambda t, o: chart_receive_map(t, o, chart), (T_, O_),
                                    eps=1e-6, atol=1e-7)


def _cfg(**overrides: Any) -> Any:
    values: Dict[str, Any] = dict(
        vocab_size=V, embed_dim=K, n_heads=H, batch_size=B, max_seq_len=N, eval_stride=2,
        n_layers=2, n_e_steps=1, warmup_steps=0, max_steps=4, log_interval=0, eval_interval=0,
        checkpoint_interval=0, skip_belief_sigma_update=True,
        generate_embedding_figures=False, pos_phi="learned",
        use_block_mlp=True, block_mlp_expansion=1, beta_attention_prior="causal_alibi",
        family="gaussian_diagonal", gauge_group="block_glk",
        attention_mode="projected_predictive", attention_residual=True,
        predictive_covariance="moment_matched",
        norm_type_block="layernorm", layernorm_affine=True, decode_mode="linear")
    values.update(overrides)
    return test_config(**values)


_GRADIENT = dict(attention_mode="projected_gradient", attention_residual=False,
                 predictive_covariance="passthrough")
_ROUTES = {
    "predictive": {},
    "gradient_closed_form": dict(_GRADIENT, relational_receiver_update="closed_form"),
    "gradient_euler": dict(_GRADIENT, relational_receiver_update="euler"),
}


@pytest.mark.parametrize("route", sorted(_ROUTES))
def test_charts_agree_at_initialization_through_the_model(route: str) -> None:
    ids = (torch.arange(B * N) % V).reshape(B, N)
    outputs = {}
    for chart in CONTENT_LINK_CHARTS:
        torch.manual_seed(5)
        model = VFEModel(_cfg(content_link_chart=chart, **_ROUTES[route])).eval()
        with torch.no_grad():
            belief, logits = model.forward_beliefs(ids, return_logits=True)
        outputs[chart] = (belief.mu, belief.sigma, logits)
    for chart in ("first_order", "exponential"):
        for got, want in zip(outputs[chart], outputs["affine"]):
            torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("route", sorted(_ROUTES))
@pytest.mark.parametrize("chart", ["first_order", "exponential"])
def test_short_training_is_finite_and_moves_the_stored_factors(route: str, chart: str, tmp_path) -> None:
    cfg = _cfg(content_link_chart=chart, **_ROUTES[route])
    torch.manual_seed(5)
    model = VFEModel(cfg)
    before = model.relational_operators.O.detach().clone()
    train_ds = TokenWindows(torch.arange(41) % V, N)
    torch.manual_seed(11)
    losses = T.train(model, DataLoader(train_ds, batch_size=B, shuffle=True), cfg, n_steps=cfg.max_steps,
                     grad_clip=1.0, log_interval=0, eval_interval=0, device=torch.device("cpu"),
                     logger=logging.getLogger("chart-test"), artifacts=None, generate_samples=False)
    assert all(math.isfinite(float(x)) for x in losses)
    assert not torch.equal(model.relational_operators.O.detach(), before)


def test_posthoc_readers_use_the_effective_receiving_map() -> None:
    cfg = _cfg(content_link_chart="exponential")
    torch.manual_seed(5)
    model = VFEModel(cfg)
    with torch.no_grad():
        model.relational_operators.O.add_(0.1 * torch.randn_like(model.relational_operators.O))
    state = model.state_dict()
    maps = relational_maps(state, cfg)[0]
    torch.testing.assert_close(maps["O"], model.relational_operators.content_factors()[1].double(),
                               rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(maps["O_stored"], model.relational_operators.O.double(), rtol=0, atol=0)
    assert "O_stored" not in relational_maps(state)[0]                    # no cfg: stored maps as before
    rows = map_invariants(state, cfg)
    assert all(math.isnan(row["conserved_TO_drift"]) for row in rows)


def test_diagnostics_report_the_effective_content_and_T_conditioning() -> None:
    torch.manual_seed(5)
    model = VFEModel(_cfg(content_link_chart="exponential")).eval()
    d = model.diagnostics((torch.arange(B * N) % V).reshape(B, N))
    assert math.isclose(d["relational_content_T_min_singular_value"], 1.0, rel_tol=1e-6)   # T = I at init
    assert math.isfinite(d["relational_content_max_singular_value"])


def test_config_rejects_an_unknown_chart() -> None:
    with pytest.raises(ValueError, match="content_link_chart"):
        _cfg(content_link_chart="second_order")
