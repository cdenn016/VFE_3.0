r"""Remediation of the 2026-10-31 audit, package posthoc-b: post-hoc routing measurements on the live route.

Findings A003 (the S1 map invariants under noisy-key routing), A027 (Table M5 row 10 on ``gaussian_frame_diagonal``),
A047 (the v2:872 hidden temperature under noisy-key routing), A095 (the M4 panel-B scatter), A123 (rows 6 and 10
inside a captured forward), A198 (the M4 component accumulator), A318 (M4 on the live recipe: the noisy-key cost, the
query-shifted clamp and tied comparison maps) and A319 (a Table M5 arm for the comparison-space RoPE).

The live route's switches come from the frozen reference profile at K = 8 (two heads of width 4) with the live
launcher's routing (``relational_routing='noisy_key'``, passthrough covariance, an attention residual), so no test
reads the launcher. Tiny CPU models only: vocabulary 9, two layers, 8-token windows.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pytest
import torch

from tests.profiles import reference_k8_config
from vfe3.config import VFE3Config
from vfe3.data.datasets import TokenWindows
from vfe3.model.model import VFEModel
from vfe3.posthoc import covariance as cv
from vfe3.posthoc import evaluate as ev
from vfe3.posthoc import figures as fg
from vfe3.posthoc import interventions as iv
from vfe3.posthoc import maps as mp
from vfe3.posthoc import report as rp
from vfe3.posthoc import routing as rt
from vfe3.relational_scoring import projected_gaussian_kl

V, N = 9, 8
DEVICE = torch.device("cpu")
LAYER_PREFIXES = ("relational_operators", "relational_operators_extra.0")


@pytest.fixture(autouse=True)
def _quiet() -> Iterator[None]:
    torch.set_num_threads(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _cfg(**over: Any) -> VFE3Config:
    r"""The live route's switches at K = 8: noisy-key routing with learned key noise, the query-shifted clamp,
    comparison-invariant RoPE at ``rope_fraction`` 1, causal ALiBi, passthrough covariance and an attention residual."""
    values: Dict[str, Any] = dict(vocab_size=V, max_seq_len=N, eval_stride=4, relational_routing="noisy_key",
                                  predictive_covariance="passthrough", attention_residual=True)
    values.update(over)
    return reference_k8_config(**values)


def _model(cfg: VFE3Config, *, dtype: torch.dtype = torch.float32, seed: int = 0) -> VFEModel:
    r"""A tiny model moved off its initialization (frames, variances, maps and key noise all perturbed)."""
    torch.manual_seed(seed)
    model = VFEModel(cfg).eval()
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for name, p in model.named_parameters():
            scale = 0.3 if "phi_embed" in name else 0.5 if "sigma" in name else 0.2
            p.add_(scale * torch.randn(p.shape, generator=g, dtype=p.dtype))
    return model.to(dtype)


def _windows() -> TokenWindows:
    stream = torch.randint(0, V, (37,), generator=torch.Generator().manual_seed(1))
    return TokenWindows(stream, N, stride=4, pad_final=True)


def _tokens(seed: int = 2) -> torch.Tensor:
    return torch.randint(0, V, (2, N), generator=torch.Generator().manual_seed(seed))


def _logits(model: VFEModel, tokens: torch.Tensor, capture: Optional[Dict[str, Any]] = None) -> torch.Tensor:
    with torch.no_grad():
        return model.forward_beliefs(tokens, return_logits=True, capture=capture)[1]


def _layer_captures(model: VFEModel, tokens: torch.Tensor) -> List[Dict[str, Any]]:
    capture: Dict[str, Any] = {"diagnostic": {}}
    _logits(model, tokens, capture)
    return capture["diagnostic"]["layer_conditional_captures"]


# ----------------------------------------------------------------------------------------------
# A003: S1 map invariants under noisy-key routing
# ----------------------------------------------------------------------------------------------

def _scaled_maps(state: Dict[str, torch.Tensor], c: float, *, noise: bool) -> Dict[str, torch.Tensor]:
    r""":math:`(S_Q, S_K) \to c(S_Q, S_K)` on every layer, with :math:`n_h \to c^2n_h` when ``noise``."""
    out = dict(state)
    for prefix in LAYER_PREFIXES:
        for leaf in ("raw_score_q", "raw_score_k"):
            out[f"{prefix}.{leaf}"] = c * state[f"{prefix}.{leaf}"]
        if noise:
            out[f"{prefix}.routing_key_log_noise"] = state[f"{prefix}.routing_key_log_noise"] + 2.0 * math.log(c)
    return out


def test_noisy_key_map_rows_report_the_scale_in_noise_units() -> None:
    cfg = _cfg()
    state = {k: v.detach().clone() for k, v in _model(cfg).state_dict().items()}
    assert all("routing_key_log_noise" in maps for maps in mp.relational_maps(state, cfg).values())
    rows = mp.map_invariants(state, cfg)
    assert len(rows) == cfg.n_layers * cfg.n_heads
    for row in rows:
        prefix = LAYER_PREFIXES[row["layer"]]
        n_h = float(state[f"{prefix}.routing_key_log_noise"][row["head"]].double().exp())
        S_Q = state[f"{prefix}.raw_score_q"][row["head"]].double()                     # (r, K)
        S_K = state[f"{prefix}.raw_score_k"][row["head"]].double()
        r = S_Q.shape[0]
        assert row["noisy_key"] == 1.0 and math.isclose(row["key_noise"], n_h, rel_tol=1e-12)
        # The common scale is the routing signal-to-noise coordinate here, not a symmetry coordinate.
        assert math.isnan(row["scale_rms"]) and math.isnan(row["conserved_S_drift"])
        expected = math.sqrt(float(S_Q.square().sum() + S_K.square().sum()) / (2 * r) / n_h)
        assert math.isclose(row["scale_rms_noise_units"], expected, rel_tol=1e-12)
        for name, S in (("S_Q", S_Q), ("S_K", S_K)):
            sv = torch.linalg.svdvals(S) / math.sqrt(n_h)
            assert math.isclose(row[f"{name}_sv_noise_units_min"], float(sv.min()), rel_tol=1e-12)
            assert math.isclose(row[f"{name}_sv_noise_units_max"], float(sv.max()), rel_tol=1e-12)
        assert math.isfinite(row["gain_median"]) and math.isfinite(row["angle_median_deg"])
    plain = _cfg(relational_routing="gaussian_kl")
    for row in mp.map_invariants(_model(plain).state_dict(), plain):
        assert math.isfinite(row["scale_rms"]) and "key_noise" not in row and "scale_rms_noise_units" not in row


def test_the_noise_unit_scale_separates_the_symmetry_from_a_routing_change() -> None:
    r"""The joint move :math:`(S_Q, S_K, n_h) \to (cS_Q, cS_K, c^2n_h)` leaves the router and every noise-unit column
    unchanged; scaling the maps alone changes the router, and the noise-unit scale moves by :math:`c` with it."""
    cfg = _cfg()
    model = _model(cfg)
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    tokens = _tokens()
    base = _logits(model, tokens)
    c = 2.0
    joint, alone = _scaled_maps(state, c, noise=True), _scaled_maps(state, c, noise=False)
    model.load_state_dict(joint)
    joint_logits = _logits(model, tokens)
    model.load_state_dict(alone)
    alone_logits = _logits(model, tokens)
    model.load_state_dict(state)
    assert float((joint_logits - base).abs().max()) < 1e-3 < float((alone_logits - base).abs().max())
    rows = mp.map_invariants(state, cfg)
    for a, b, s in zip(rows, mp.map_invariants(joint, cfg), mp.map_invariants(alone, cfg)):
        for key in ("scale_rms_noise_units", "S_Q_sv_noise_units_min", "S_Q_sv_noise_units_max",
                    "S_K_sv_noise_units_min", "S_K_sv_noise_units_max", "gain_median", "angle_median_deg"):
            assert math.isclose(a[key], b[key], rel_tol=1e-6), key                     # float32 log-noise
        assert math.isclose(s["scale_rms_noise_units"], c * a["scale_rms_noise_units"], rel_tol=1e-6)


def test_map_table_and_figure_panels_show_the_noise_unit_scale(tmp_path: Path) -> None:
    cfg = _cfg()
    rows = mp.map_invariants(_model(cfg).state_dict(), cfg)
    table = rp.map_invariants_table(rows)
    assert "(noise units)" in table and "sqrt(n_h)" in table
    column, label = fg._scale_column(rows)
    assert column == "scale_rms_noise_units" and "n_h" in label
    values, title, _ = fg._invariant_panels(rows)[0]
    np.testing.assert_allclose(values, np.log10(fg._grid(rows, "scale_rms_noise_units")), rtol=1e-12)
    assert "n_h" in title
    fg.plot_map_invariant_curves([dict(r, step=s) for s in (0, 10) for r in rows], tmp_path)
    plain = _cfg(relational_routing="gaussian_kl")
    plain_rows = mp.map_invariants(_model(plain).state_dict(), plain)
    assert fg._scale_column(plain_rows)[0] == "scale_rms"
    assert "(noise units)" not in rp.map_invariants_table(plain_rows)


# ----------------------------------------------------------------------------------------------
# A027: row 10 on the frame-diagonal family
# ----------------------------------------------------------------------------------------------

def test_row_10_is_frame_free_by_construction_on_the_frame_diagonal_family() -> None:
    frame_diagonal = _model(_cfg(family="gaussian_frame_diagonal", relational_routing="gaussian_kl"))
    reason = iv._role_trained_frame_free(frame_diagonal, iv.ROLE_ROUTING_COVARIANCE)
    assert reason is not None and "reference variances" in reason
    with pytest.raises(iv.NotApplicable, match="reference variances"):
        with iv.frames_removed_routing_covariance(frame_diagonal):
            raise AssertionError("row 10 entered its body on the frame-diagonal family")
    for role in (iv.ROLE_INPUT_MOMENTS, iv.ROLE_NORM_MLP):                  # the other roles keep their frames
        assert iv._role_trained_frame_free(frame_diagonal, role) is None
    diagonal = _model(_cfg(relational_routing="gaussian_kl"))
    assert iv._role_trained_frame_free(diagonal, iv.ROLE_ROUTING_COVARIANCE) is None
    dataset = _windows()
    indices = np.arange(len(dataset))
    base = ev.nll_pass(diagonal, dataset, indices, batch_size=2, device=DEVICE)
    with iv.frames_removed_routing_covariance(diagonal) as record:
        arm = ev.nll_pass(diagonal, dataset, indices, batch_size=2, device=DEVICE)
    assert record["intercepted_calls"] > 0 and np.abs(arm.nll - base.nll).max() > 1e-6


# ----------------------------------------------------------------------------------------------
# A047: the hidden temperature under noisy-key routing
# ----------------------------------------------------------------------------------------------

def _mahalanobis(model: VFEModel, variance: float) -> torch.Tensor:
    r"""The mean-dependent part of layer 0's routing cost at a uniform stored variance, on the causal entries.

    ``relational_score(mu, Sigma) - relational_score(0, Sigma)`` with the model's own frames and operators is
    :math:`\tfrac12\delta^\top(G_k + n_hI)^{-1}\delta` (no key noise on the Gaussian-KL route).
    """
    from vfe3.gradients.relational import relational_score

    tokens = _tokens(4)
    K = int(model.cfg.embed_dim)
    with torch.no_grad():
        beliefs = model.prior_bank.encode(tokens)
        beliefs = beliefs._replace(phi=model._apply_pos_phi(beliefs.phi), right_phi=model._pos_phi_right(beliefs.phi))
        _, frame, _ = model._relational_frames(beliefs, model._rope_rotation(N, tokens.device))
        mu = 0.3 * torch.randn(2, N, K, generator=torch.Generator().manual_seed(5))
        sigma = torch.full((2, N, K), variance)
        ops = model.relational_operators
        full = relational_score(mu, sigma, mu, sigma, frame, ops)
        zero = relational_score(torch.zeros_like(mu), sigma, torch.zeros_like(mu), sigma, frame, ops)
    return (full - zero)[..., torch.tril(torch.ones(N, N, dtype=torch.bool))]


def test_hidden_temperature_matches_the_routers_sharpening_only_without_key_noise() -> None:
    final = 1.05
    for routing in ("gaussian_kl", "noisy_key"):
        cfg = _cfg(relational_routing=routing)
        model = _model(cfg)
        with torch.no_grad():
            model.prior_bank.sigma_log_embed.fill_(math.log(final))
            for op in model.relational_operators_by_layer:
                if op.routing_key_log_noise is not None:
                    op.routing_key_log_noise.fill_(math.log(2.0))                    # a learned n_h = 2
        ratio = float((_mahalanobis(model, final) / _mahalanobis(model, float(cfg.sigma_init))).median())
        kappa = float(cfg.sigma_init) / final
        out = cv.hidden_temperature(model)
        if routing == "gaussian_kl":
            assert out["status"] == "ok" and math.isclose(out["factor"], kappa, rel_tol=1e-5)
            assert math.isclose(ratio, out["factor"], rel_tol=1e-3)                  # the 1/c scaling holds
        else:
            assert out["status"] == "not_applicable" and "noisy_key" in out["reason"]
            assert "factor" not in out and "definition" not in out
            assert ratio < 0.9 * kappa                                              # n_h damps the scaling


@pytest.mark.parametrize("covariance, fixed", [("passthrough", True), ("moment_matched", False)])
def test_hidden_temperature_records_whether_the_stack_keeps_the_covariance(covariance: str, fixed: bool) -> None:
    dataset = _windows()
    model = _model(_cfg(relational_routing="gaussian_kl", predictive_covariance=covariance))
    out = cv.covariance_analysis(model, dataset, stream_tokens=dataset.tokens, device=DEVICE, n_windows=2,
                                 n_uniform_windows=0)
    ht = out["hidden_temperature"]
    assert ht["status"] == "ok" and ht["covariance_fixed_through_stack"] is fixed
    assert ("layer-0" in ht["scope"]) is (not fixed)
    assert f"Scope: {ht['scope']}" in cv.covariance_table(out)
    alone = cv.covariance_analysis(model, None, stream_tokens=None, device=DEVICE)
    assert alone["hidden_temperature"]["covariance_fixed_through_stack"] is None


# ----------------------------------------------------------------------------------------------
# A095: the M4 panel-B scatter
# ----------------------------------------------------------------------------------------------

def _synthetic_m4_result(L: int, H: int, S: int) -> Dict[str, Any]:
    r"""A routing result whose rows identify their head: ``l1_shared = 2 eps`` and ``l1_model = 3 eps + h``."""
    rng = np.random.default_rng(0)
    panel_a = [dict(layer=l, head=h, logit_var_within=1.0,
                    **{f"share_within_{c}": 1.0 / len(rt.COMPONENTS) for c in rt.COMPONENTS})
               for l in range(L) for h in range(H)]
    panel_b = [dict(layer=l, head=h, frac_model_within_bound=0.5) for l in range(L) for h in range(H)]
    panel_c = [dict(layer=l, head=h, frame_share=0.5, **{f"logdet_spread_{v}": 0.1 for v in rt.VARIANTS})
               for l in range(L) for h in range(H)]
    samples = {}
    for l in range(L):
        for h in range(H):
            eps = rng.uniform(0.01, 1.0, S).astype(np.float32)
            samples[f"L{l}H{h}"] = {"epsilon": eps, "bound": 2 * np.tanh(eps / 2), "l1_model": 3 * eps + h,
                                    "l1_shared": 2 * eps, "l1_model_shared": np.zeros(S, np.float32),
                                    "n_sources": np.full(S, 5.0, np.float32)}
    return {"panel_a": panel_a, "panel_b": panel_b, "panel_c": panel_c, "samples": samples}


def test_panel_b_scatter_draws_every_head_of_each_layer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.axes

    L, H, S, points = 2, 3, 50, 60
    monkeypatch.setattr(rt, "FIGURE_POINTS_PER_LAYER", points)
    calls: List[Tuple[np.ndarray, np.ndarray, Any]] = []
    original = matplotlib.axes.Axes.scatter

    def recording(self: Any, x: Any, y: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append((np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), kwargs.get("color")))
        return original(self, x, y, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "scatter", recording)
    rt.plot_routing_decomposition(_synthetic_m4_result(L, H, S), tmp_path, name="m4")
    shared = [(x, y) for x, y, color in calls if x.size and isinstance(color, str) and color == "0.6"]
    trained = [(x, y) for x, y, color in calls if x.size and not (isinstance(color, str) and color == "0.6")]
    assert len(shared) == len(trained) == L
    for (x_shared, y_shared), (x_model, y_model) in zip(shared, trained):
        assert x_shared.size == x_model.size == points
        np.testing.assert_array_equal(x_shared, x_model)                    # one row index for both scatters
        np.testing.assert_allclose(y_shared, 2 * x_shared, rtol=1e-6)        # l1_shared of the same rows
        heads = np.rint(y_model - 3 * x_model).astype(int)                   # l1_model = 3 eps + h
        assert set(heads.tolist()) == set(range(H))
    pick = rt._figure_rows(H * S, 0)
    assert np.array_equal(pick, rt._figure_rows(H * S, 0)) and np.array_equal(pick, np.unique(pick))
    assert np.array_equal(rt._figure_rows(points - 1, 1), np.arange(points - 1))  # a small layer is drawn whole


# ----------------------------------------------------------------------------------------------
# A123: rows 6 and 10 inside a captured forward
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("arm", [iv.routing_covariances_identity, iv.frames_removed_routing_covariance],
                         ids=["row6", "row10"])
def test_routing_covariance_rows_run_inside_a_captured_forward(arm: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Both counterfactual scorers take the capture's ``term_support``: the arm runs under a diagnostic capture with
    the values of the uncaptured arm, and the captured term medians of the substituted costs run over the capture's
    causal support (not over every pair, as a dropped support would give)."""
    from vfe3.gradients import relational

    supports: List[Optional[torch.Tensor]] = []
    original = relational.diagonal_kl_term_medians

    def recording(a_q: torch.Tensor, g_q: torch.Tensor, a_k: torch.Tensor, g_k: torch.Tensor,
                  support: Optional[torch.Tensor] = None) -> torch.Tensor:
        supports.append(support)
        return original(a_q, g_q, a_k, g_k, support)

    monkeypatch.setattr(relational, "diagonal_kl_term_medians", recording)
    model = _model(_cfg(relational_routing="gaussian_kl"))
    tokens = _tokens()
    base_layers = _layer_captures(model, tokens)
    base_supports, supports[:] = list(supports), []
    with arm(model) as plain_record:
        plain = _logits(model, tokens)
    capture: Dict[str, Any] = {"diagnostic": {}}
    with arm(model) as record:
        captured = _logits(model, tokens, capture)
    assert torch.equal(plain, captured)
    assert record["intercepted_calls"] == plain_record["intercepted_calls"] > 0
    assert len(supports) == len(base_supports) == model.cfg.n_layers
    for support, base_support in zip(supports, base_supports):
        assert support is not None and base_support is not None and torch.equal(support, base_support)
    layers = capture["diagnostic"]["layer_conditional_captures"]
    for cap, base_cap in zip(layers, base_layers):
        assert cap["score_term_medians_pre"].shape == base_cap["score_term_medians_pre"].shape
        assert not torch.equal(cap["score_term_medians_pre"], base_cap["score_term_medians_pre"])


# ----------------------------------------------------------------------------------------------
# A198: the M4 component accumulator
# ----------------------------------------------------------------------------------------------

def _stacked_panel_a(comps: Dict[str, torch.Tensor], entries: torch.Tensor) -> Dict[str, torch.Tensor]:
    r"""The panel-A sums as accumulated before the fix (all components stacked), the reference of the loop."""
    zero = torch.zeros((), dtype=torch.float64)
    x = torch.stack([torch.where(entries, comps[c], zero) for c in rt.COMPONENTS])   # (C, B, H, N, N)
    total = x.sum(0)
    count = entries.sum(-1).clamp_min(1).to(torch.float64)
    centered = torch.where(entries, x - (x.sum(-1) / count).unsqueeze(-1), zero)
    centered_total = centered.sum(0)
    return {"w_cl": (centered * centered_total).sum(dim=(1, 3, 4)), "w_cc": centered.square().sum(dim=(1, 3, 4)),
            "w_ll": centered_total.square().sum(dim=(0, 2, 3)), "r_c": x.sum(dim=(1, 3, 4)),
            "r_cc": x.square().sum(dim=(1, 3, 4)), "r_cl": (x * total).sum(dim=(1, 3, 4)),
            "r_l": total.sum(dim=(0, 2, 3)), "r_ll": total.square().sum(dim=(0, 2, 3)),
            "entries": entries.sum(dim=(0, 2, 3)).double(), "rows": entries.any(-1).sum(dim=(0, 2)).double()}


def test_component_accumulator_matches_the_stacked_reference() -> None:
    g = torch.Generator().manual_seed(11)
    B, H, n = 2, 3, 6
    comps = {c: torch.randn(B, H, n, n, generator=g, dtype=torch.float64) for c in rt.COMPONENTS}
    comps["row"] = comps["row"][..., :1].expand(B, H, n, n)                     # a row constant, as returned
    entries = torch.tril(torch.ones(n, n, dtype=torch.bool)).expand(B, H, n, n).clone()
    entries[0, 1, 2] = False                                                    # a row without entries
    entries[1, :, 4, :2] = False
    acc = rt._LayerAccumulator(H, 4)
    acc.covariances(comps, entries)
    acc.covariances(comps, entries)                                             # two batches accumulate
    for key, value in _stacked_panel_a(comps, entries).items():
        torch.testing.assert_close(acc.a[key], 2 * value, rtol=1e-12, atol=1e-12, msg=key)


# ----------------------------------------------------------------------------------------------
# A318: M4 on the live recipe
# ----------------------------------------------------------------------------------------------

def _decompose(model: VFEModel) -> Dict[str, Any]:
    dataset = _windows()
    return rt.routing_decomposition(model, dataset, list(range(len(dataset))), batch_size=2, device=DEVICE)


@pytest.mark.parametrize("over", [
    {}, {"kl_max": 1.0}, {"rope_fraction": 0.5}, {"routing_kl_max_reference": "query_shifted_v1"},
    {"tie_comparison_maps": True},
    {"relational_routing": "gaussian_kl", "routing_kl_max_reference": "absolute", "tie_comparison_maps": True}],
    ids=["live", "live_capped", "live_dense_commutant", "query_shifted_v1", "live_tied", "gaussian_kl_absolute_tied"])
def test_m4_decomposition_is_exact_on_the_live_recipe(over: Dict[str, Any], tmp_path: Path) -> None:
    cfg = _cfg(**over)
    result = _decompose(_model(cfg, dtype=torch.float64))
    assert result["status"] == "ok", result.get("reason")
    worst = result["checks_worst"]
    assert worst["identity_max_rel_error"] < 1e-12                 # vs projected_gaussian_kl of the key law
    assert worst["model_cost_max_abs_error"] < 1e-10               # vs the router's raw cost (D_ij - c_i if shifted)
    assert worst["model_logit_max_abs_error"] < 1e-10              # vs -D_bar/tau + B, modulo the row constant
    assert worst["model_beta_max_abs_error"] < 1e-12               # vs the router's beta_pre
    for row in result["panel_a"]:
        assert math.isclose(sum(row[f"share_within_{c}"] for c in rt.COMPONENTS), 1.0, abs_tol=1e-9)
    assert all(row["max_shared_excess"] <= rt.BOUND_SLACK for row in result["panel_b"])
    setup = result["setup"]
    assert setup["relational_routing"] == cfg.relational_routing
    if cfg.relational_routing == "noisy_key":
        assert setup["tau"] == [1.0] * cfg.n_heads                 # the operator's routing temperature
        assert len(setup["key_noise"]) == cfg.n_layers and all(len(h) == cfg.n_heads for h in setup["key_noise"])
        assert "n_h I" in setup["key_covariance"] and "n_h I" in result["definitions"]["whitening"]
    if over.get("kl_max") == 1.0:
        assert any(abs(row["share_within_cap"]) > 1e-3 for row in result["panel_a"])
    if not over:
        assert "routing noisy_key" in rt.routing_table(result)
        names = {p.name for p in rt.write_routing_outputs(result, tmp_path)}
        assert {"m4_routing_decomposition.json", "m4_routing_samples.npz", "m4_routing_decomposition.pdf"} <= names


def test_m4_key_law_is_the_broadened_key_covariance() -> None:
    r"""Under noisy-key routing the pooled :math:`V_*` and panel C read :math:`V_{K,j} + n_hI`: raising every
    :math:`n_h` lowers every log-det spread (the shared noise damps the dispersion)."""
    model = _model(_cfg(), dtype=torch.float64)
    low = _decompose(model)
    with torch.no_grad():
        for op in model.relational_operators_by_layer:
            op.routing_key_log_noise.add_(math.log(10.0))
    high = _decompose(model)
    assert low["status"] == high["status"] == "ok"
    for a, b in zip(low["panel_c"], high["panel_c"]):
        assert b["logdet_spread_model"] < a["logdet_spread_model"]
    assert high["checks_worst"]["model_beta_max_abs_error"] < 1e-12


@pytest.mark.parametrize("reference", ["absolute", "row_min", "query_shifted", "query_shifted_v1"])
def test_components_sum_to_the_clamped_logit_minus_the_row_constant(reference: str) -> None:
    from vfe3.gradients.relational import clamped_routing_cost, query_shifted_row_reference

    B, n, H, r = 2, 5, 3, 3
    g = torch.Generator().manual_seed(4)

    def spd() -> torch.Tensor:
        X = torch.randn(B, n, H, r, r + 2, generator=g, dtype=torch.float64)
        return X @ X.transpose(-1, -2) / (r + 2) + 0.1 * torch.eye(r, dtype=torch.float64)

    a_q, a_k = (torch.randn(B, n, H, r, generator=g, dtype=torch.float64) for _ in range(2))
    V_q, V_k = spd(), spd()
    W, V_inv, _ = rt.pooled_whitener(V_k.mean(dim=(0, 1)))
    log_prior = torch.triu(torch.full((n, n), -math.inf, dtype=torch.float64), 1) - 0.3 * torch.arange(
        n, dtype=torch.float64).view(-1, 1)
    tau = torch.tensor([0.7, 1.3, 2.0], dtype=torch.float64)
    D = projected_gaussian_kl(a_q, V_q, a_k, V_k)                                    # (B, H, N, N)
    c = (-0.5 * (r + torch.logdet(V_q))).movedim(-1, -2).unsqueeze(-1)              # (B, H, N, 1) query constant
    ell = (0.5 * torch.logdet(V_k)).movedim(-1, -2)                                  # (B, H, N) key log-volumes
    ops = SimpleNamespace(kl_max_reference=reference)
    support = torch.isfinite(log_prior).expand_as(D)
    if reference.startswith("query_shifted"):
        kl_max = float((D - c)[support].median())
        volume = ell if reference == "query_shifted" else None
        D_bar = clamped_routing_cost(D - c, ops, kl_max=kl_max, log_prior=log_prior, key_log_volume=volume)
        rho = c + (query_shifted_row_reference(D - c, ell, log_prior) if volume is not None else 0.0)
    else:
        kl_max = float(D[support].median())
        D_bar = clamped_routing_cost(D, ops, kl_max=kl_max, log_prior=log_prior)
        rho = (D.masked_fill(~support, math.inf).amin(-1, keepdim=True) if reference == "row_min"
               else torch.zeros_like(D[..., :1]))
    comps = rt.logit_components(a_q, V_q, a_k, V_k, W, V_inv, log_prior, tau, kl_max, kl_max_reference=reference)
    total = sum(comps[k] for k in rt.COMPONENTS)
    t = tau.view(1, H, 1, 1)
    expected = log_prior - D_bar / t - rho / t                     # the model logit minus rho_i / tau
    assert torch.allclose(total[support], expected[support], rtol=0.0, atol=1e-12)
    torch.testing.assert_close(comps["reference"], rho.expand_as(comps["reference"]), rtol=0.0, atol=1e-12)
    assert bool((comps["cap"][support] != 0).any())


def test_m4_still_refuses_the_costs_it_does_not_decompose() -> None:
    for routing in ("noisy_query", "mean_distance"):
        reason = rt.not_applicable_reason(_model(_cfg(relational_routing=routing)))
        assert reason is not None and routing in reason
    assert "comparison" in rt.not_applicable_reason(_model(_cfg(rope_mode="comparison")))
    assert rt.not_applicable_reason(_model(_cfg())) is None


# ----------------------------------------------------------------------------------------------
# A319: a Table M5 arm for the comparison-space RoPE
# ----------------------------------------------------------------------------------------------

def test_comparison_rope_arm_removes_the_rotation_and_keeps_the_commutant_projection() -> None:
    cfg = _cfg()
    model = _model(cfg)
    dataset = _windows()
    indices = np.arange(len(dataset))
    base = ev.nll_pass(model, dataset, indices, batch_size=2, device=DEVICE)
    with iv.comparison_rope_identity(model) as record:
        arm = ev.nll_pass(model, dataset, indices, batch_size=2, device=DEVICE)
    assert record["layers_changed"] == cfg.n_layers
    assert record["intercepted_calls"] >= cfg.n_layers * math.ceil(len(dataset) / 2)
    assert np.abs(arm.nll - base.nll).max() > 1e-6
    assert np.array_equal(ev.nll_pass(model, dataset, indices, batch_size=2, device=DEVICE).nll, base.nll)
    assert all("comparison_rope_rotation" not in vars(op) for op in model.relational_operators_by_layer)
    tokens = _tokens()
    base_layers = _layer_captures(model, tokens)
    with iv.comparison_rope_identity(model):
        arm_layers = _layer_captures(model, tokens)
    for base_cap, arm_cap in zip(base_layers, arm_layers):        # P(G) as trained, only the means unrotated
        assert torch.equal(base_cap["projected_covariance_pre"], arm_cap["projected_covariance_pre"])
        assert torch.equal(base_cap["projected_key_covariance_pre"], arm_cap["projected_key_covariance_pre"])
        assert not torch.equal(base_cap["beta_pre"], arm_cap["beta_pre"])


def test_comparison_rope_arm_is_not_applicable_without_a_comparison_rope() -> None:
    model = _model(_cfg(rope_mode="none"))
    with pytest.raises(iv.NotApplicable, match="comparison-space RoPE"):
        with iv.comparison_rope_identity(model):
            raise AssertionError("the arm entered its body")


def test_table_m5_runs_the_rope_arm_and_keeps_the_manuscript_rows() -> None:
    assert [i.row for i in iv.INTERVENTIONS] == list(range(1, 13))                # Table M5 of the manuscript
    assert [i.row for i in iv.by_key()] == list(range(1, 14))
    assert [i.key for i in iv.by_key(["comparison_rope_identity"])] == ["baseline", "comparison_rope_identity"]
    model = _model(_cfg())
    dataset = _windows()
    indices = np.arange(len(dataset))
    base = ev.nll_pass(model, dataset, indices, batch_size=2, device=DEVICE)
    rows = rp._run_interventions(iv, model, dataset, indices, base, base.window.copy(), batch_size=2, device=DEVICE,
                                 keys=["comparison_rope_identity"], n_resamples=50, log=lambda message: None,
                                 save_to=None)
    by_key = {row["key"]: row for row in rows}
    assert by_key["comparison_rope_identity"]["status"] == "ok" and by_key["comparison_rope_identity"]["row"] == 13
    assert by_key["restoration_check"]["status"] == "ok" and by_key["restoration_check"]["max_abs_nll_change"] == 0.0
    table = rp.interventions_table(rows, n_units=int(np.unique(base.window).size), unit="window", n_targets=base.n,
                                   subset=False)
    assert iv.by_key(["comparison_rope_identity"])[1].label in table
