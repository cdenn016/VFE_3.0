r"""M4 routing-score decomposition (:mod:`vfe3.posthoc.routing`) on tiny CPU models.

Models have vocabulary 24, K=4, two heads of dimension 2, two layers and 8-token windows; the exactness
tests run the model in float64 so the component sum is compared with the model's own routing logits and
weights at float64 tolerance.
"""

from __future__ import annotations

from tests.profiles import test_config

import math
import warnings
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.data.datasets import TokenWindows
from vfe3.model.model import VFEModel
from vfe3.posthoc import routing as rt
from vfe3.relational_scoring import projected_gaussian_kl

V = 24
DEVICE = torch.device("cpu")


@pytest.fixture(autouse=True)
def _one_thread() -> None:
    torch.set_num_threads(1)


def _cfg(**over: Any) -> VFE3Config:
    values: Dict[str, Any] = dict(vocab_size=V, embed_dim=4, n_heads=2, max_seq_len=8, eval_stride=4,
                                  n_layers=2, batch_size=2, attention_mode="projected_predictive",
                                  e_phi_lr=0.0, block_mlp_expansion=2)
    values.update(over)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return test_config(**values)


def _model(cfg: VFE3Config, *, seed: int = 0, sigma: float = 0.5, frames: float = 0.3,
           dtype: torch.dtype = torch.float64) -> VFEModel:
    r"""A tiny model moved off its initialization; ``sigma``/``frames`` = 0 give identity covariances/frames."""
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(cfg).eval()
    with torch.no_grad():
        model.prior_bank.mu_embed.normal_(0.0, 1.0)
        model.prior_bank.sigma_log_embed.normal_(0.0, 1.0).mul_(sigma)
        model.prior_bank.phi_embed.normal_(0.0, 1.0).mul_(frames)
        model.pos_phi_free.normal_(0.0, 1.0).mul_(frames)
        for op in model.relational_operators_by_layer:
            for p in op.parameters():
                p.add_(0.3 * torch.randn_like(p))
    return model.to(dtype)


def _windows(n_tokens: int = 30, seed: int = 1) -> TokenWindows:
    stream = torch.randint(0, V, (n_tokens,), generator=torch.Generator().manual_seed(seed))
    return TokenWindows(stream, 8, stride=4, pad_final=True)


def _run(model: VFEModel, batch_size: int = 2) -> Dict[str, Any]:
    dataset = _windows()
    return rt.routing_decomposition(model, dataset, list(range(len(dataset))), batch_size=batch_size,
                                    device=DEVICE)


def _random_moments(B: int, N: int, H: int, r: int, seed: int) -> Dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)

    def spd() -> torch.Tensor:
        X = torch.randn(B, N, H, r, r + 2, generator=g, dtype=torch.float64)
        return X @ X.transpose(-1, -2) / (r + 2) + 0.1 * torch.eye(r, dtype=torch.float64)

    return {"a_q": torch.randn(B, N, H, r, generator=g, dtype=torch.float64),
            "a_k": torch.randn(B, N, H, r, generator=g, dtype=torch.float64),
            "V_q": spd(), "V_k": spd()}


# ----------------------------------------------------------------------------------------------
# Panel A: exact components
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("over", [
    {}, {"kl_max": 1.0},
    {"family": "gaussian_full", "decode_mode": "full_chunked",
     "predictive_covariance": "moment_matched"}])
def test_components_sum_to_the_model_routing_logits(over: Dict[str, Any]) -> None:
    result = _run(_model(_cfg(**over)))
    assert result["status"] == "ok"
    assert result["setup"]["covariance"] == ("full" if "family" in over else "diagonal")
    worst = result["checks_worst"]
    assert worst["identity_max_rel_error"] < 1e-12           # vs projected_gaussian_kl in float64
    assert worst["model_cost_max_abs_error"] < 1e-10         # vs the model's own raw costs
    assert worst["model_logit_max_abs_error"] < 1e-10        # vs -D_bar/tau + B captured from the model
    assert worst["model_beta_max_abs_error"] < 1e-12         # vs the model's routing weights beta_pre
    if over.get("kl_max") == 1.0:
        assert any(abs(row["share_within_cap"]) > 1e-3 for row in result["panel_a"])


def test_logit_components_are_exact_on_constructed_moments_with_cap() -> None:
    B, N, H, r = 2, 5, 3, 3
    m = _random_moments(B, N, H, r, seed=4)
    V_star = m["V_k"].mean(dim=(0, 1))
    W, V_inv, _ = rt.pooled_whitener(V_star)
    log_prior = torch.triu(torch.full((N, N), -math.inf, dtype=torch.float64), 1) - 0.3 * torch.arange(
        N, dtype=torch.float64).view(-1, 1)
    tau = torch.tensor([0.7, 1.3, 2.0], dtype=torch.float64)
    D_ref = projected_gaussian_kl(m["a_q"], m["V_q"], m["a_k"], m["V_k"])            # (B, H, N, N)
    kl_max = float(D_ref.median())                                                  # half the costs capped
    comps = rt.logit_components(m["a_q"], m["V_q"], m["a_k"], m["V_k"], W, V_inv, log_prior, tau, kl_max)
    total = sum(comps[c] for c in rt.COMPONENTS)
    expected = log_prior - D_ref.clamp(0.0, kl_max) / tau.view(1, H, 1, 1)
    support = torch.isfinite(log_prior).expand_as(total)
    assert torch.allclose(total[support], expected[support], rtol=0.0, atol=1e-12)
    assert torch.allclose(comps["D"], D_ref, rtol=1e-12, atol=1e-12)
    assert bool((comps["cap"][support] != 0).any())
    # Row-constant component is constant along j; under V_K,j = V_* the precision term vanishes.
    assert torch.equal(comps["row"], comps["row"][..., :1].expand_as(comps["row"]))
    shared = rt.logit_components(m["a_q"], m["V_q"], m["a_k"], V_star.expand_as(m["V_k"]).contiguous(), W,
                                 V_inv, log_prior, tau, 1e9)
    assert shared["precision"].abs().max() < 1e-12
    assert torch.allclose(shared["logdet"], shared["logdet"][..., :1].expand_as(shared["logdet"]), atol=1e-12)


def test_variance_shares_sum_to_one() -> None:
    result = _run(_model(_cfg()))
    for row in result["panel_a"]:
        assert math.isclose(sum(row[f"share_within_{c}"] for c in rt.COMPONENTS), 1.0, abs_tol=1e-9)
        assert math.isclose(sum(row[f"share_raw_{c}"] for c in rt.COMPONENTS), 1.0, abs_tol=1e-9)
        assert abs(row["share_within_row"]) < 1e-9                    # a row constant cannot route
        assert math.isclose(sum(row[f"cov_within_{c}"] for c in rt.COMPONENTS), row["logit_var_within"],
                            rel_tol=1e-9, abs_tol=1e-12)
        assert row["n_entries"] > row["n_rows"] > 0


# ----------------------------------------------------------------------------------------------
# Panel B: the softmax bound
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("eps", [1e-3, 0.3, 1.0, 4.0])
def test_bound_is_attained_by_the_two_source_extremal_case(eps: float) -> None:
    p1 = 1.0 / (1.0 + math.exp(eps))
    dot = torch.tensor([[math.log(p1), math.log(1.0 - p1)]], dtype=torch.float64)
    key_bias = torch.tensor([[eps, -eps]], dtype=torch.float64)            # offsets +-eps about the midrange
    rows = rt.routing_bound_rows(dot, key_bias - 7.0, torch.ones(1, 2, dtype=torch.bool))
    assert math.isclose(float(rows["epsilon"]), eps, rel_tol=1e-12)
    assert math.isclose(float(rows["l1_shared"]), 2.0 * math.tanh(eps / 2.0), rel_tol=1e-10)
    assert math.isclose(float(rows["bound"]), 2.0 * math.tanh(eps / 2.0), rel_tol=1e-12)


def test_shared_covariance_routing_satisfies_the_bound_on_random_rows() -> None:
    g = torch.Generator().manual_seed(7)
    n_rows, N = 4000, 9
    dot = 3.0 * torch.randn(n_rows, N, generator=g, dtype=torch.float64)
    key_bias = torch.randn(n_rows, N, generator=g, dtype=torch.float64) * torch.rand(
        n_rows, 1, generator=g, dtype=torch.float64) * 5.0
    support = torch.rand(n_rows, N, generator=g) < 0.7
    support[:, 0] = True
    rows = rt.routing_bound_rows(dot, key_bias, support, model_logits=dot + key_bias)
    assert bool((rows["l1_shared"] <= rows["bound"] + 1e-12).all())
    assert bool((rows["bound"] <= torch.minimum(rows["epsilon"], torch.full_like(rows["epsilon"], 2.0)) + 1e-15).all())
    assert torch.allclose(rows["l1_model"], rows["l1_shared"]) and rows["l1_model_shared"].max() < 1e-12
    single = support.sum(-1) == 1
    assert bool((rows["epsilon"][single] == 0).all()) and bool((rows["l1_shared"][single] == 0).all())


def test_model_panel_b_respects_the_theorem_and_the_sample_is_batch_independent() -> None:
    model = _model(_cfg())
    one, three = _run(model, batch_size=1), _run(model, batch_size=3)
    for row in one["panel_b"]:
        assert row["max_shared_excess"] <= rt.BOUND_SLACK
        assert 0.0 <= row["frac_model_within_bound"] <= 1.0
    for key, fields in one["samples"].items():
        order_1, order_3 = np.argsort(fields["epsilon"]), np.argsort(three["samples"][key]["epsilon"])
        for field in rt.SAMPLE_FIELDS:
            np.testing.assert_allclose(fields[field][order_1], three["samples"][key][field][order_3], atol=1e-6)
    for a, b in zip(one["panel_a"], three["panel_a"]):
        assert math.isclose(a["share_within_trace"], b["share_within_trace"], rel_tol=1e-9, abs_tol=1e-12)


# ----------------------------------------------------------------------------------------------
# Panel C: key-covariance dispersion
# ----------------------------------------------------------------------------------------------

def test_logdet_spread_is_zero_for_equal_covariances_and_congruence_invariant() -> None:
    g = torch.Generator().manual_seed(3)
    X = torch.randn(4, 6, generator=g, dtype=torch.float64)
    V0 = X @ X.T + 0.5 * torch.eye(4, dtype=torch.float64)
    n = 11
    equal = rt.logdet_spread(n * V0, n * torch.logdet(V0), n)
    assert abs(float(equal)) < 1e-12
    Vs = torch.stack([V0 + 0.3 * torch.diag(torch.rand(4, generator=g, dtype=torch.float64)) for _ in range(n)])
    A = torch.randn(4, 4, generator=g, dtype=torch.float64) + 3.0 * torch.eye(4, dtype=torch.float64)
    moved = A @ Vs @ A.T
    spread = rt.logdet_spread(Vs.sum(0), torch.logdet(Vs).sum(), n)
    assert float(spread) > 0.0
    assert math.isclose(float(spread), float(rt.logdet_spread(moved.sum(0), torch.logdet(moved).sum(), n)),
                        rel_tol=1e-9)


def test_zero_dispersion_when_all_key_covariances_are_equal() -> None:
    r"""Identity covariances and frames: V_K,j = S_K S_K^T for every source, so every spread vanishes,
    the precision term is identically zero and the trained router equals the shared-covariance router."""
    result = _run(_model(_cfg(), sigma=0.0, frames=0.0))
    assert result["status"] == "ok"
    for row in result["panel_c"]:
        for variant in rt.VARIANTS:
            assert abs(row[f"logdet_spread_{variant}"]) < 1e-12
        assert row["ai_dispersion_model"] < 1e-20
    for row in result["panel_a"]:
        assert row["var_within_precision"] < 1e-24
        assert row["var_within_logdet"] < 1e-24
    for row in result["panel_b"]:
        assert row["l1_model_shared_mean"] < 1e-12


def test_frame_share_is_one_without_covariance_spread_and_zero_without_frames() -> None:
    frames_only = _run(_model(_cfg(), sigma=0.0, frames=0.4))
    for row in frames_only["panel_c"]:
        assert abs(row["logdet_spread_frames_identity"]) < 1e-12 and row["logdet_spread_sigma_mean"] > 1e-6
        assert math.isclose(row["frame_share"], 1.0, abs_tol=1e-9)
    covariances_only = _run(_model(_cfg(), sigma=0.6, frames=0.0))
    for row in covariances_only["panel_c"]:
        assert abs(row["logdet_spread_sigma_mean"]) < 1e-12 and row["logdet_spread_frames_identity"] > 1e-6
        assert math.isclose(row["frame_share"], 0.0, abs_tol=1e-9)
        assert math.isclose(row["logdet_spread_model"], row["logdet_spread_frames_identity"], rel_tol=1e-9)


# ----------------------------------------------------------------------------------------------
# Skips and outputs
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("over, fragment", [({"attention_mode": "projected_gradient", "predictive_covariance": "passthrough"}, "attention_mode"),
                                            ({"relational_routing": "noisy_query"}, "noisy")])
def test_skip_with_reason_on_routes_without_the_projected_gaussian_router(
        over: Dict[str, Any], fragment: str, tmp_path: Path) -> None:
    cfg = _cfg(**over)
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(cfg).eval()
    result = _run(model)
    assert result["status"] == "skipped" and fragment in result["reason"]
    paths = rt.write_routing_outputs(result, tmp_path)
    assert sorted(p.name for p in paths) == ["m4_routing_decomposition.json", "m4_routing_decomposition.md"]
    assert "Skipped" in (tmp_path / "m4_routing_decomposition.md").read_text(encoding="utf-8")


def test_write_outputs_writes_figure_table_and_samples(tmp_path: Path) -> None:
    result = _run(_model(_cfg(), dtype=torch.float32))
    assert result["status"] == "ok"
    paths = rt.write_routing_outputs(result, tmp_path)
    names = {p.name for p in paths}
    assert {"m4_routing_decomposition.md", "m4_routing_decomposition.json", "m4_routing_samples.npz",
            "m4_routing_decomposition.png", "m4_routing_decomposition.pdf"} <= names
    text = (tmp_path / "m4_routing_decomposition.md").read_text(encoding="utf-8")
    assert "Panel A" in text and "Panel B" in text and "Panel C" in text
    with np.load(tmp_path / "m4_routing_samples.npz") as samples:
        assert samples["L0H0_epsilon"].size == result["panel_b"][0]["rows_sampled"]
