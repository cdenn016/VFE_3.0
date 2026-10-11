r"""Run artifacts and analysis remediation of the 2026-10-10 round-2 audit (items 16, 17, 19, 22, 23, 24, 33,
42, 46, 47, 52).

Every model is a tiny CPU profile (``tests/profiles.py``); figures are built in memory and inspected for
their text, never displayed.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tests.profiles import test_config, tiny_config
from vfe3 import run_artifacts as ra
from vfe3.contracts import CanonicalFrameContext
from vfe3.model.model import VFEModel
from vfe3.viz import extract
from vfe3.viz import figures as figs

TOKENS = torch.tensor([[1, 4, 2]])


def _texts(fig: Any) -> str:
    r"""Every title, axis label, tick label, legend entry and figure text of ``fig``, joined."""
    out: List[str] = [t.get_text() for t in fig.texts]
    if getattr(fig, "_suptitle", None) is not None:
        out.append(fig._suptitle.get_text())
    for ax in fig.axes:
        out += [ax.get_title(), ax.get_xlabel(), ax.get_ylabel()]
        out += [t.get_text() for t in ax.get_yticklabels() + ax.get_xticklabels()]
        legend = ax.get_legend()
        if legend is not None:
            out += [t.get_text() for t in legend.get_texts()]
    return "\n".join(out)


def _witness_model(**overrides: object) -> VFEModel:
    torch.manual_seed(0)
    model = VFEModel(tiny_config(**overrides)).eval()
    with torch.no_grad():
        model.prior_bank.phi_embed.normal_(mean=0.0, std=0.3)                 # nontrivial realized frames
    return model


# ------------------------------------------------------------------------------------- item 16

@pytest.mark.parametrize(("family", "decode_mode", "mode", "consumer", "invariant"), [
    ("gaussian_full", "full_chunked", "state_dependent_per_coord", "mstep", False),
    ("gaussian_full", "full_chunked", "state_dependent_per_coord", "canonical", True),   # a diagnostic only
    ("gaussian_full", "full_chunked", "state_dependent_per_coord", None, True),
    ("gaussian_block", "full_chunked", "state_dependent_per_coord", "mstep", False),
    ("gaussian_full", "full_chunked", "state_dependent", "mstep", True),
    ("gaussian_diagonal", "linear", "state_dependent_per_coord", "mstep", True),
])
def test_self_coupling_frame_flag_tracks_family_and_consumer(
    family: str, decode_mode: str, mode: str, consumer: object, invariant: bool,
) -> None:
    cfg = tiny_config(family=family, decode_mode=decode_mode, lambda_alpha_mode=mode,
                      mstep_self_coupling_weight=0.5 if consumer == "mstep" else 0.0,
                      diagnose_canonical_free_energy=consumer == "canonical")
    report = ra._pure_path_report(cfg, [])
    assert report["gauge_flags"]["self_coupling_frame_invariant"] is invariant
    if not invariant:
        assert report["on_gauge_pure_path"] is False
    expected = {"mstep": ["mstep_self_coupling"], "canonical": ["canonical_free_energy_diagnostic"], None: []}
    assert report["config_toggles"]["self_coupling_consumers"] == expected[consumer]


@pytest.mark.parametrize(("mode", "frame_dependent"), [
    ("state_dependent_per_coord", True), ("state_dependent", False), ("constant", False)])
def test_witness_self_term_uses_an_independent_prior(mode: str, frame_dependent: bool) -> None:
    model = _witness_model(family="gaussian_full", decode_mode="full_chunked", lambda_alpha_mode=mode,
                           mstep_self_coupling_weight=0.5)
    residuals = extract.projected_gauge_equivariance(model, TOKENS, n_draws=2)["residuals"]
    worst = max(residuals["self_coupling"])
    assert worst > 1e-3 if frame_dependent else worst < 1e-10
    for key, values in residuals.items():                       # the routing and readout stay covariant
        if key != "self_coupling":
            assert max(values) < 1e-10, (key, values)


def test_witness_reports_no_self_term_without_a_consumer() -> None:
    model = _witness_model(family="gaussian_full", decode_mode="full_chunked",
                           lambda_alpha_mode="state_dependent_per_coord")
    assert "self_coupling" not in extract.projected_gauge_equivariance(model, TOKENS, n_draws=1)["residuals"]


# ------------------------------------------------------------------------------------- item 52

def test_act_on_frame_refuses_a_dense_element_with_off_block_entries() -> None:
    H, d, n = 2, 2, 3
    forward = torch.eye(d, dtype=torch.float64).expand(1, n, H, d, d).clone()      # compact identity frames
    frame = CanonicalFrameContext(forward, forward.clone(), True)
    block = torch.block_diag(*[torch.tensor([[1.0, 0.5], [0.0, 2.0]], dtype=torch.float64)] * H)
    good = block.expand(1, n, H * d, H * d).clone()                                # (1, N, K, K)
    moved = extract._act_on_frame(good, frame)
    assert torch.allclose(moved.forward[0, 0, 1], block[2:, 2:])
    bad = good.clone()
    bad[0, 1, 0, 3] = 1e-3                                                         # one off-block entry
    with pytest.raises(ValueError, match="off-block"):
        extract._act_on_frame(bad, frame)


# ------------------------------------------------------------------------------------- items 19 and 33

def _rope_model(**overrides: object) -> VFEModel:
    torch.manual_seed(4)
    cfg = test_config(vocab_size=11, embed_dim=4, n_heads=2, max_seq_len=6, n_layers=2,
                      attention_mode="projected_predictive", pos_rotation="rope", rope_space="comparison",
                      rope_on_value=False, **overrides)
    return VFEModel(cfg)


def _ce(model: VFEModel, tokens: torch.Tensor, targets: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(tokens)
    return float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1)))


def test_no_rope_arm_keeps_the_commutant_projection(monkeypatch: pytest.MonkeyPatch) -> None:
    import vfe3.geometry.rope as rope_module

    model = _rope_model(rope_comparison_covariance="rotation_invariant")
    tokens, targets = torch.tensor([[0, 1, 2, 3, 4, 5]]), torch.tensor([[1, 2, 3, 4, 5, 6]])
    channel = ra.collect_beta_channel_decomposition(model, tokens, targets)
    operators = model.relational_operators_by_layer
    assert all("comparison_rope_rotation" not in ops.__dict__ for ops in operators)       # restored
    assert channel["ce"]["full"] == pytest.approx(_ce(model, tokens, targets), abs=1e-6)
    # Independent reference: identity rotations from the rotation builder, P(G) still applied.
    monkeypatch.setattr(rope_module, "build_rope_rotation",
                        lambda positions, dims, **kw: torch.eye(sum(dims), device=kw.get("device"),
                                                                dtype=kw.get("dtype")).expand(
                            positions.shape[0], sum(dims), sum(dims)).clone())
    for ops in operators:
        ops.__dict__.pop("_comparison_rope_cache", None)
    identity_with_projection = _ce(model, tokens, targets)
    monkeypatch.undo()
    for ops in operators:
        ops.__dict__.pop("_comparison_rope_cache", None)
        ops.comparison_rope_base = None                     # the former arm: no rotation and no P(G)
    without_projection = _ce(model, tokens, targets)
    assert channel["ce"]["no_rope"] == pytest.approx(identity_with_projection, abs=1e-6)
    assert abs(channel["ce"]["no_rope"] - without_projection) > 1e-5


def test_pure_path_rope_label_and_recorded_covariance_mode() -> None:
    def toggles(**overrides: object) -> Dict[str, object]:
        return ra._pure_path_report(_rope_model(**overrides).cfg, [])["config_toggles"]

    invariant = toggles(rope_comparison_covariance="rotation_invariant")
    assert invariant["rope_pair_energy_exactness"] == (
        "comparison_space_rotated_means_commutant_projected_covariance")
    assert invariant["rope_comparison_covariance"] == "rotation_invariant"
    full = toggles(rope_comparison_covariance="full")
    assert full["rope_pair_energy_exactness"] == "comparison_space_rotated_projected_moments"
    assert full["rope_comparison_covariance"] == "full"
    plain = ra._pure_path_report(tiny_config(), [])["config_toggles"]
    assert plain["rope_comparison_covariance"] is None


# ------------------------------------------------------------------------------------- item 17

def _canonical_history(n: int = 12) -> Dict[str, np.ndarray]:
    step = np.arange(n, dtype=float) * 100.0
    return {"step": step,
            "self_coupling": 24.0 + 0.1 * np.arange(n),        # alpha D + R(alpha), mostly R
            "self_divergence": 6.0 + 0.05 * np.arange(n),
            "belief_coupling": 3.0 - 0.1 * np.arange(n) / n,
            "attention_entropy": 1.0 + 0.01 * np.arange(n),
            "val_ce": 5.0 - 0.1 * np.arange(n),
            "canonical_beta_tv": np.linspace(0.6, 0.4, n)}


def test_canonical_free_energy_figures_are_labeled_as_a_diagnostic() -> None:
    history = _canonical_history()
    for build in (figs.plot_canonical_free_energy_relationship, figs.plot_canonical_free_energy_decomposition):
        fig = build(history, lambda_beta=1.0)
        text = _texts(fig)
        assert "Canonical F" in text and "diagnostic" in text and "optimizes it" in text
        for legacy in ("Inner-objective", "inner", "E-step", "descend", "codescent"):
            assert legacy not in text, (build.__name__, legacy)
        assert r"\beta_{ij}-\beta^*_{ij}" in text                # canonical_beta_tv drawn beside the energy
        figs.plt.close(fig)
    fig = figs.plot_canonical_free_energy_decomposition(history, lambda_beta=1.0)
    text = _texts(fig)
    assert r"D(q_i\|p_i)$ (not summed)" in text and r"+R(\alpha_i)" in text
    bars = [patch for patch in fig.axes[0].patches if patch.get_hatch()]
    assert [bar.get_width() for bar in bars] == [pytest.approx(history["self_divergence"][-1])]
    figs.plt.close(fig)
    assert figs.get_figure("canonical_free_energy_relationship") is figs.plot_canonical_free_energy_relationship
    assert figs.get_figure("free_energy_codescent") is figs.plot_free_energy_relationship    # legacy route only
    assert r"\beta^*_{ij}" in figs.PUB_LABELS["belief_coupling"]


def test_save_figures_routes_the_canonical_diagnostic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    called: List[str] = []
    for name in ("plot_canonical_free_energy_relationship", "plot_canonical_free_energy_decomposition",
                 "plot_free_energy_relationship", "plot_free_energy_decomposition"):
        def fake(history: Dict, *, lambda_beta: float = 1.0, path: str = "", _name: str = name) -> Any:
            called.append(_name)
            assert _name.startswith("plot_free") or "self_divergence" in history
            Path(path).write_bytes(b"png")
            return figs.plt.figure()
        monkeypatch.setattr(figs, name, fake)
    cfg = tiny_config()
    art = ra.RunArtifacts(str(tmp_path), cfg, VFEModel(cfg), dataset="synthetic", device="cpu")
    history = _canonical_history(4)
    art.history = [{key: float(values[i]) for key, values in history.items()}
                   | {"relational_legacy_global_f_available": 1.0} for i in range(4)]
    ra._save_figures(art, None, logging.getLogger("test"))
    assert sorted(called) == ["plot_canonical_free_energy_decomposition", "plot_canonical_free_energy_relationship"]
    assert (tmp_path / "canonical_free_energy_relationship.png").is_file()
    assert not (tmp_path / "free_energy_relationship.png").exists()


# ------------------------------------------------------------------------------------- item 22

def test_routing_heads_withholds_holonomy_of_massless_heads() -> None:
    holonomy = {"step": np.repeat([10.0, 20.0], 3), "layer": np.zeros(6), "head": np.tile([0.0, 1.0, 2.0], 2),
                "path_mass": np.array([0.5, 1.5e-35, 0.2, 0.4, 0.3, 0.3]),
                "log_defect_rms": np.array([2.0, 27.31, 1.0, 3.0, 1.0, 2.0]),
                "prior_log_defect_rms": np.full(6, 3.3)}
    table = figs.routing_head_table(None, None, holonomy)
    np.testing.assert_array_equal(table["path_mass"], holonomy["path_mass"])
    assert math.isnan(table["log_defect_rms"][1]) and table["log_defect_rms"][0] == 2.0
    assert table["prior_log_defect_rms"][1] == 3.3
    fig = figs.plot_routing_heads(table, step=10.0)
    panel = next(ax for ax in fig.axes if ax.get_title().startswith("routing holonomy"))
    assert panel.images[0].get_clim() == pytest.approx((1.0, 2.0))               # 27.31 sets no scale
    assert "withheld for 1" in _texts(fig)
    figs.plt.close(fig)
    assert "withheld" not in _texts(fig := figs.plot_routing_heads(table, step=20.0))
    figs.plt.close(fig)
    fig = figs.plot_routing_heads_training(table)
    panel = next(ax for ax in fig.axes if ax.get_title().startswith("routing holonomy"))
    np.testing.assert_allclose(panel.lines[0].get_ydata(), [1.5, 2.0])         # head mean ignores the cell
    assert "excluded from the head mean and range" in _texts(fig)
    figs.plt.close(fig)


# ------------------------------------------------------------------------------------- item 23

@pytest.mark.parametrize(("source", "additive"), [("config", True), ("config", False), ("posthoc", False)])
def test_figure_s1_panel_a_labels_by_additivity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                source: str, additive: bool) -> None:
    from vfe3.paper import figures as pf
    from vfe3.paper.runs import RunRecord

    diag = tmp_path / "diagnostics"
    diag.mkdir()
    (diag / "residual_ratio.csv").write_text(
        "step,layer,q10,q50,q90,ln_rel_change_q50\n10,0,0.5,1.0,2.0,\n20,0,0.6,1.1,2.1,\n", encoding="utf-8")
    cfg = SimpleNamespace(relational_readout_mode="predictive_content", predictive_readout="kernel",
                          attention_residual=additive)
    run = (RunRecord(run_dir=tmp_path, cfg=cfg, summary={"n_steps": 20}) if source == "config" else
           RunRecord(run_dir=tmp_path, summary={"n_steps": 20},
                     posthoc={"residual": {"final_model.pt": {"residual_additive": additive}}}))
    captured: Dict[str, Any] = {}
    monkeypatch.setattr(pf, "save_figure", lambda fig, out_dir, stem: captured.setdefault("fig", fig) and [])
    record = pf.plot_numerics(run, tmp_path / "out")
    assert record["panels"]["A"]["residual_additive"] is additive
    text = _texts(captured["fig"])
    if additive:
        assert r"$\|m_i\|/\|\mu_i\|$" in text and "Message-to-state ratio" in text
    else:
        assert r"\mu^{\rm out}_i" in text and "Displacement ratio" in text and r"$\|m_i\|" not in text
    figs.plt.close(captured["fig"])


# ------------------------------------------------------------------------------------- item 24

def test_per_layer_aggregation_withholds_per_column(tmp_path: Path) -> None:
    import multiseed_analysis as ms
    from tests.test_multiseed import _write_full_run, _write_request

    _write_request(tmp_path, [1, 2])
    header = "layer,self_coupling,relational_semantics,holonomy_deviation,canonical_beta_tv\n"
    _write_full_run(tmp_path / "a", ppl=1.0, seed=1,
                    per_layer=header + "0,10,relational_predictive_content,nan,0.5\n1,12,relational_predictive_content,nan,0.4\n")
    _write_full_run(tmp_path / "b", ppl=1.0, seed=2,
                    per_layer=header + "0,20,relational_predictive_content,nan,0.3\n1,14,relational_predictive_content,1.0,\n")
    withheld: Dict[str, str] = {}
    out = ms.aggregate_per_layer(tmp_path, withheld=withheld)
    assert out[0]["self_coupling"]["mean"] == pytest.approx(15.0) and out[1]["self_coupling"]["mean"] == 13.0
    assert set(out[0]) == {"self_coupling"}
    assert set(withheld) == {"relational_semantics", "holonomy_deviation", "canonical_beta_tv"}
    assert "text column" in withheld["relational_semantics"]
    assert "at 2 of 2 layer(s)" in withheld["holonomy_deviation"]
    assert "at 1 of 2 layer(s)" in withheld["canonical_beta_tv"]
    assert {"total", "canonical_beta_tv", "self_divergence"} <= set(ms.PER_LAYER_METRICS)


def test_conditional_per_layer_dashboard_shows_the_canonical_diagnostic() -> None:
    n = 3
    per_layer: Dict[str, object] = {
        "attn_entropy": [1.0, 0.9, 0.8], "relational_semantics": ["relational_predictive_content"] * n,
        "self_coupling": [20.0, 22.0, 24.0], "self_divergence": [0.0, 3.0, 6.0],
        "belief_coupling": [6.3, 5.0, 4.0], "attention_entropy": [1.1, 1.0, 0.9],
        "total": [27.4, 28.0, 28.9], "canonical_beta_tv": [0.77, 0.6, 0.5]}
    fig = figs.plot_per_layer_diagnostics(per_layer)
    titles = [ax.get_title() for ax in fig.axes]
    assert "Canonical F at the captured beliefs (diagnostic)" in titles
    assert "Model router against the canonical router" in titles
    figs.plt.close(fig)
    del per_layer["canonical_beta_tv"], per_layer["total"], per_layer["self_coupling"]
    del per_layer["self_divergence"], per_layer["belief_coupling"], per_layer["attention_entropy"]
    fig = figs.plot_per_layer_diagnostics(per_layer)
    assert not any("anonical" in ax.get_title() for ax in fig.axes)
    figs.plt.close(fig)


# ------------------------------------------------------------------------------------- item 42

def test_realized_expected_likelihood_kernel_is_recorded_on_cpu(tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    from vfe3.paper import tables

    monkeypatch.setattr(ra, "_git_code_identity",
                        lambda *a, **k: {"git_sha": "0" * 40, "git_dirty": False, "git_dirty_fingerprint": None})
    for name in ("_write_provenance", "_write_research_artifacts", "_run_figures_isolated"):
        monkeypatch.setattr(ra, name, lambda *a, **k: None)
    cfg = tiny_config(decode_mode="expected_likelihood_chunked", expected_likelihood_kernel="triton",
                      generate_embedding_figures=False)
    model = VFEModel(cfg)
    assert model.prior_bank.realized_expected_likelihood_kernel is None
    model(torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]]))                   # CPU: the torch kernel runs
    assert model.prior_bank.realized_expected_likelihood_kernel == "torch"
    art = ra.RunArtifacts(tmp_path / "r", cfg, model, save_model_artifacts=False)
    ra.finalize_run(model, art, cfg, test_loader=None)
    summary = json.loads((tmp_path / "r" / "summary.json").read_text(encoding="utf-8"))
    assert summary["expected_likelihood_ce_kernel"] == "torch"
    assert "expected_likelihood_kernel" in tables.PRECISION_KEYS
    run = SimpleNamespace(config_raw={"expected_likelihood_kernel": "triton"}, summary=summary, provenance={},
                          config_record={}, raw=lambda key, default=None: {"expected_likelihood_kernel": "triton"}.get(
                              key, default))
    items = tables._numerics_items(run)
    rendered = " ".join(str(value) for _, value in items)
    assert "expected_likelihood_kernel=triton" in rendered and "torch" in rendered


# ------------------------------------------------------------------------------------- item 46

def test_frames_identity_rule_is_shared(tmp_path: Path) -> None:
    from tests.test_semantics_core import _load, _save_run, _tiny_model
    from vfe3.model.frame_actions import frames_identity_reason
    from vfe3.posthoc import interventions as iv

    base = dict(gauge_transport="on", phi_scale=0.01, m_phi_lr=0.01, pos_phi="none", pos_rotation="none")
    assert frames_identity_reason(SimpleNamespace(**base), 8) is None
    assert "gauge_transport='off'" in frames_identity_reason(SimpleNamespace(**{**base, "gauge_transport": "off"}), 8)
    assert "no generators" in frames_identity_reason(SimpleNamespace(**base), 0)
    still = SimpleNamespace(**{**base, "phi_scale": 0.0, "m_phi_lr": 0.0})
    assert "phi_scale=0" in frames_identity_reason(still, 8)
    learned = SimpleNamespace(**{**base, "phi_scale": 0.0, "m_phi_lr": 0.0, "pos_phi": "learned"})
    assert frames_identity_reason(learned, 8) is None and frames_identity_reason(learned, 0) is not None
    assert "phi_scale=0" in frames_identity_reason(learned, 8, positional=False)
    assert frames_identity_reason(SimpleNamespace(**{**base, "gauge_transport": "frozen"}), 8) is None

    cfg, model = _tiny_model(decode_mode="linear", phi_scale=0.0, m_phi_lr=0.0, pos_phi="none")
    run = _load(_save_run(tmp_path, "still", cfg, model))
    assert run.frames_active is False and any("phi_scale=0" in note for note in run.notes)
    assert "phi_scale=0" in iv.frames_inactive_reason(model)
    cfg, model = _tiny_model(decode_mode="linear")
    run = _load(_save_run(tmp_path, "moving", cfg, model))
    assert run.frames_active is True and iv.frames_inactive_reason(model) is None


# ------------------------------------------------------------------------------------- item 47

def test_holonomy_driver_is_click_to_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import inspect

    from vfe3.posthoc import holonomy
    from vfe3.posthoc import streams

    assert "argparse" not in inspect.getsource(holonomy)
    assert list(inspect.signature(holonomy.main).parameters) == []
    assert {"run_dir", "checkpoints", "windows", "batch_size", "triangles", "seed", "device", "out"} == set(
        holonomy.CONFIG)
    monkeypatch.setitem(holonomy.CONFIG, "run_dir", None)
    with pytest.raises(ValueError, match="run_dir"):
        holonomy.main()
    seen: Dict[str, object] = {}
    monkeypatch.setattr(holonomy, "checkpoint_holonomy",
                        lambda run_dir, checkpoint, **kw: seen.setdefault(checkpoint, kw) and {"checkpoint": checkpoint})
    monkeypatch.setattr(holonomy, "snapshot_holonomy", lambda run_dir, cfg: [])
    monkeypatch.setattr(streams, "load_run_info", lambda run_dir, checkpoint: SimpleNamespace(cfg=None))
    out = tmp_path / "holonomy.json"
    for key, value in (("run_dir", str(tmp_path)), ("checkpoints", ["final_model.pt"]), ("windows", 2),
                       ("out", str(out))):
        monkeypatch.setitem(holonomy.CONFIG, key, value)
    report = holonomy.main()
    assert seen["final_model.pt"]["n_windows"] == 2
    assert json.loads(out.read_text(encoding="utf-8"))["checkpoints"] == report["checkpoints"]
