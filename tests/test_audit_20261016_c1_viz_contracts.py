r"""Visualization and metric contracts, 2026-10-16 deep audit, lane C1 (findings 82 to 86).

F82: ``readout_diagnostics.png`` never drew its two canonical-F panels. ``_save_figures`` passed only the
``relational_*`` columns and four routing scalars to the dashboard, and the canonical panel read a ``total``
column that ``metrics.csv`` writes as ``inner_alignment_energy_total``. The readout history now carries the
canonical F terms and router columns, with ``total`` read from ``inner_alignment_energy_total``.

F83: the canonical free-energy figures paired the training-batch F of the log step with the validation CE in
one Pearson correlation, while the held-out ``val_*`` F columns of the same evaluation rows went unplotted; the
held-out chart-radius and vertex-conditioning columns were selected for ``validation_sanity.png`` and drawn by no
panel. The figures now read the held-out columns when they exist and name the population they plot, and the
held-out dashboard draws the chart radius and vertex conditioning.

F84: report mode ran the strict ``generate_figures`` replay before the lenient history re-render, so a run whose
saved config this build refuses lost its history dashboards with the replay. The history now renders whether
or not the replay succeeds.

F85: ``plot_scaling_law`` printed :math:`R^2` for the offset law at four sizes, which leave its three parameters
one residual degree of freedom; the gate now uses ``scaling_min_testable_sizes`` of the fitted form.

F86: ``lr_grid_heatmap_kwargs`` stored a ``None`` baseline that the heatmap then unpacked; the inference-bank
copy and the record decode's device move dropped ``CanonicalFrameContext.block_diagonal``; ``summary.json`` wrote
three ``estep_final_*`` keys that nothing produces, so they were always null.

Tiny CPU models only (K=4, two heads, N=4, vocab 7).
"""
import csv
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import vfe3.run_artifacts as ra
import vfe3.train as T
import vfe3.viz.figures as figs
from tests.profiles import test_config
from tests.test_audit2_20261011_training import _cfg
from vfe3.belief import BeliefState
from vfe3.config import serialize_config
from vfe3.contracts import CanonicalFrameContext
from vfe3.data.datasets import TokenWindows
from vfe3.metric_contracts import scaling_min_testable_sizes
from vfe3.model.model import VFEModel
from vfe3.viz import extract, figure_worker, sweep_adapters

B, N, V = 2, 4, 7
CPU = torch.device("cpu")
LOGGER = logging.getLogger("audit-c1-viz-contracts")


@pytest.fixture(autouse=True)
def _single_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T, "_save_eval_attention_maps", lambda *args, **kwargs: None)
    monkeypatch.setattr(ra, "_run_figures_isolated", lambda *args, **kwargs: True)
    torch.set_num_threads(1)


def _titles(fig: Any) -> List[str]:
    return [ax.get_title() for ax in fig.axes]


def _texts(fig: Any) -> str:
    parts = [text.get_text() for text in fig.texts]
    for ax in fig.axes:
        parts.append(ax.get_title())
        legend = ax.get_legend()
        if legend is not None:
            parts.extend(text.get_text() for text in legend.get_texts())
    return "\n".join(parts)


def _canonical_row(index: int, *, held_out: bool = True, eval_row: bool = True) -> Dict[str, float]:
    r"""One metrics row of a ``diagnose_canonical_free_energy`` run: train-batch F terms on every row, the
    held-out ``val_*`` F terms and ``val_ce`` on the evaluation rows only."""
    nan = float("nan")
    row: Dict[str, float] = {
        "step": float(10 * (index + 1)),
        "relational_legacy_global_f_available": 1.0,
        "relational_score_klmax_frac": 0.01 * index,
        "attn_entropy": 1.2,
        "self_coupling": 20.0 + index,
        "self_divergence": 6.0 + 0.1 * index,
        "belief_coupling": 3.0,
        "attention_entropy": 1.0,
        "inner_alignment_energy_total": 24.0 + index,
        "canonical_total_output": 23.0 + index,
        "canonical_total_delta": -1.0,
        "canonical_beta_tv": 0.6 - 0.01 * index,
        "canonical_self_weight": 0.3,
        "val_ce": (5.0 - 0.1 * index) if eval_row else nan,
    }
    if held_out:
        row.update({
            "val_self_coupling": (2.0 + 0.1 * index) if eval_row else nan,
            "val_self_divergence": 0.5 if eval_row else nan,
            "val_belief_coupling": 0.3 if eval_row else nan,
            "val_attention_entropy": 0.1 if eval_row else nan,
            "val_inner_alignment_energy_total": (2.4 + 0.1 * index) if eval_row else nan,
            "val_canonical_beta_tv": 0.2 if eval_row else nan,
            "val_canonical_self_weight": 0.4 if eval_row else nan,
        })
    return row


# ------------------------------------------------------------------------------------------------ F82

def test_readout_diagnostics_draws_the_canonical_panels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}
    original = figs.plot_readout_diagnostics

    def spy(history: Dict, **kwargs: Any) -> Any:
        fig = original(history, **kwargs)
        captured.update(history=history, titles=_titles(fig))
        return fig

    monkeypatch.setattr(figs, "plot_readout_diagnostics", spy)
    history = [_canonical_row(index) for index in range(4)]
    ra._save_figures(SimpleNamespace(run_dir=tmp_path, cfg=None, history=history), None, LOGGER)
    assert "Canonical F at the captured beliefs (diagnostic)" in captured["titles"]
    assert "Model router against the canonical router" in captured["titles"]
    readout = captured["history"]
    assert readout["total"] == [row["inner_alignment_energy_total"] for row in history]
    for key in ("canonical_total_output", "self_coupling", "self_divergence", "belief_coupling",
                "attention_entropy", "canonical_beta_tv", "canonical_self_weight"):
        assert readout[key] == [row[key] for row in history], key
    assert (tmp_path / "readout_diagnostics.png").is_file()


def test_readout_diagnostics_keeps_no_canonical_panel_without_the_diagnostic(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}
    original = figs.plot_readout_diagnostics

    def spy(history: Dict, **kwargs: Any) -> Any:
        fig = original(history, **kwargs)
        captured.update(history=history, titles=_titles(fig))
        return fig

    monkeypatch.setattr(figs, "plot_readout_diagnostics", spy)
    nan = float("nan")
    history = [{**_canonical_row(index, held_out=False),
                **{key: nan for key in ("self_coupling", "self_divergence", "belief_coupling",
                                        "attention_entropy", "inner_alignment_energy_total")}}
               for index in range(4)]
    for row in history:                                   # the canonical columns exist only under the toggle
        for key in ("canonical_total_output", "canonical_total_delta", "canonical_beta_tv",
                    "canonical_self_weight"):
            del row[key]
    ra._save_figures(SimpleNamespace(run_dir=tmp_path, cfg=None, history=history), None, LOGGER)
    assert "Canonical F at the captured beliefs (diagnostic)" not in captured["titles"]
    assert "total" not in captured["history"]


# ------------------------------------------------------------------------------------------------ F83

def _canonical_spies(monkeypatch: pytest.MonkeyPatch) -> Dict[str, Dict[str, Any]]:
    calls: Dict[str, Dict[str, Any]] = {}
    for name in ("plot_canonical_free_energy_relationship", "plot_canonical_free_energy_decomposition"):
        def fake(history: Dict, *, _name: str = name, **kwargs: Any) -> Any:
            calls[_name] = {"history": history, **kwargs}
            Path(kwargs["path"]).write_bytes(b"png")
            return figs.plt.figure()
        monkeypatch.setattr(figs, name, fake)
    return calls


def test_canonical_figures_read_the_held_out_free_energy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _canonical_spies(monkeypatch)
    history = [_canonical_row(index, eval_row=index % 2 == 1) for index in range(6)]
    ra._save_figures(SimpleNamespace(run_dir=tmp_path, cfg=None, history=history), None, LOGGER)
    evals = [row for row in history if row["val_ce"] == row["val_ce"]]
    assert sorted(calls) == ["plot_canonical_free_energy_decomposition", "plot_canonical_free_energy_relationship"]
    for call in calls.values():
        assert call["population"] == "held_out"
        hist = call["history"]
        assert list(hist["step"]) == [row["step"] for row in evals]
        assert list(hist["val_ce"]) == [row["val_ce"] for row in evals]
        for key in ("self_coupling", "belief_coupling", "attention_entropy", "self_divergence",
                    "canonical_beta_tv", "canonical_self_weight"):
            assert list(hist[key]) == [row[f"val_{key}"] for row in evals], key


def test_canonical_figures_fall_back_to_the_training_batch_and_say_so(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _canonical_spies(monkeypatch)
    history = [_canonical_row(index, held_out=False) for index in range(4)]
    ra._save_figures(SimpleNamespace(run_dir=tmp_path, cfg=None, history=history), None, LOGGER)
    for call in calls.values():
        assert call["population"] == "training_batch"
        assert list(call["history"]["self_coupling"]) == [row["self_coupling"] for row in history]


def _canonical_history(n: int = 12) -> Dict[str, np.ndarray]:
    return {"step": np.arange(n, dtype=float) * 100.0,
            "self_coupling": 24.0 + 0.1 * np.arange(n), "self_divergence": 6.0 + 0.05 * np.arange(n),
            "belief_coupling": 3.0 - 0.1 * np.arange(n) / n, "attention_entropy": 1.0 + 0.01 * np.arange(n),
            "val_ce": 5.0 - 0.1 * np.arange(n), "canonical_beta_tv": np.linspace(0.6, 0.4, n)}


@pytest.mark.parametrize("population, phrase", [("held_out", "held-out diagnostic sequences"),
                                                 ("training_batch", "training batch of each log step")])
def test_canonical_figures_name_their_population(population: str, phrase: str) -> None:
    for build in (figs.plot_canonical_free_energy_relationship, figs.plot_canonical_free_energy_decomposition):
        fig = build(_canonical_history(), lambda_beta=1.0, population=population)
        assert phrase in _texts(fig), build.__name__
        figs.plt.close(fig)
    fig = figs.plot_canonical_free_energy_relationship(_canonical_history(), population=population)
    title = fig.axes[0].get_title()
    assert title.startswith("Held-out canonical F" if population == "held_out" else "Training-batch canonical F")
    figs.plt.close(fig)


def test_canonical_figures_refuse_an_unknown_population() -> None:
    with pytest.raises(ValueError, match="population"):
        figs.plot_canonical_free_energy_relationship(_canonical_history(), population="test")


def test_validation_sanity_draws_the_held_out_chart_radius_and_vertex_conditioning() -> None:
    keys = ("val_phi_matrix_norm_p95", "val_phi_matrix_norm_p99", "val_phi_matrix_norm_max", "val_vertex_cond_p99")
    for key in keys:
        assert key in figs.VALIDATION_SANITY_COLUMNS, key
    history = {"step": np.arange(4, dtype=float) * 10.0,
               **{key: np.linspace(1.0, 2.0, 4) * (index + 1) for index, key in enumerate(keys)}}
    fig = figs.plot_validation_sanity(history)
    titles = _titles(fig)
    assert "Held-out phi chart radius" in titles and "Held-out vertex conditioning" in titles
    figs.plt.close(fig)


# ------------------------------------------------------------------------------------------------ F84

def _report_request(tmp_path: Path, run_dir: Path) -> Path:
    result = run_dir / "figure_result.json"
    result.write_text("{}", encoding="utf-8")
    request = tmp_path / "figure_request.json"
    request.write_text(json.dumps({
        "mode": "report", "run_dir": str(run_dir), "result_path": str(result), "split": "validation",
        "device": "cpu", "max_sequences": None, "n_e_steps": None, "generate_embedding_figures": False,
    }), encoding="utf-8")
    return request


def _history_run(tmp_path: Path, **saved: object) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    raw = {**serialize_config(test_config()), **saved}
    (run_dir / "config.json").write_text(json.dumps({"config": raw}), encoding="utf-8")
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "val_ppl", "grad_norm"])
        writer.writeheader()
        for step in range(1, 9):
            writer.writerow({"step": step, "val_ppl": 50.0 - step, "grad_norm": 1.0 + 0.1 * step})
    return run_dir


def test_report_mode_rebuilds_the_history_when_the_strict_config_is_refused(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = _history_run(tmp_path, lambda_alpha_mode="constant")      # a value this build refuses
    monkeypatch.setenv("VFE3_FIGURE_REQUEST", str(_report_request(tmp_path, run_dir)))
    with pytest.raises(ValueError):
        figure_worker.main()
    assert (run_dir / "val_ppl.png").is_file()
    assert (run_dir / "grad_norm.png").is_file()


def test_report_mode_renders_the_history_after_a_successful_replay(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vfe3.viz.report as report
    run_dir = _history_run(tmp_path)
    order: List[str] = []
    monkeypatch.setattr(report, "generate_figures", lambda *args, **kwargs: order.append("replay") or [])
    monkeypatch.setattr(figure_worker, "_render_persisted_run_figures",
                        lambda *args, **kwargs: order.append("history"))
    monkeypatch.setenv("VFE3_FIGURE_REQUEST", str(_report_request(tmp_path, run_dir)))
    assert figure_worker.main() == 0
    assert order == ["replay", "history"]                  # probes the replay persisted are rendered too
    assert json.loads((run_dir / "figure_result.json").read_text(encoding="utf-8")) == {"paths": []}


# ------------------------------------------------------------------------------------------------ F85

def _scaling_points(sizes: List[float], seeds: int = 3) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(0)
    points = []
    for n_params in sizes:
        mean = 3.0 + 40.0 * n_params ** -0.25
        points.append({"n_params": float(n_params), "route": "embed", "label": f"N{int(n_params)}",
                       "ce_seeds": list(mean * (1.0 + 0.002 * rng.standard_normal(seeds)))})
    return points


def _fit_label(fig: Any) -> str:
    labels = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
    return next(label for label in labels if label.startswith("fit"))


@pytest.mark.parametrize("n_sizes, with_offset, testable", [(4, True, False), (5, True, True), (4, False, True)])
def test_scaling_law_gates_r2_on_the_fitted_form(n_sizes: int, with_offset: bool, testable: bool) -> None:
    sizes = list(np.geomspace(1e5, 1e8, n_sizes))
    fig = figs.plot_scaling_law(_scaling_points(sizes), with_offset=with_offset)
    label = _fit_label(fig)
    figs.plt.close(fig)
    if with_offset:
        assert "offset power law" in label                     # the three-parameter law was fitted
    assert ("$R^2$=" in label) is testable, label
    if not testable:
        assert f"not testable (<{scaling_min_testable_sizes(with_offset)} sizes)" in label


# ------------------------------------------------------------------------------------------------ F86

def _grid_rows() -> List[Dict[str, Any]]:
    return [{"overrides": {"lr_a": x, "lr_b": y}, "primary_val_ppl": 10.0 + x + y}
            for x in (0.1, 0.2) for y in (1.0, 2.0)]


def test_lr_grid_heatmap_without_a_baseline() -> None:
    kwargs = sweep_adapters.lr_grid_heatmap_kwargs(_grid_rows(), "lr_a", "lr_b", (0.1, 0.2), (1.0, 2.0), None)
    assert kwargs is not None and "baseline" not in kwargs["grid"]
    fig = figs.plot_lr_grid_heatmap(**kwargs)
    assert "baseline" not in _texts(fig)
    figs.plt.close(fig)
    fig = figs.plot_lr_grid_heatmap({**kwargs["grid"], "baseline": None})      # an explicit None is tolerated
    figs.plt.close(fig)
    with_baseline = sweep_adapters.lr_grid_heatmap_kwargs(
        _grid_rows(), "lr_a", "lr_b", (0.1, 0.2), (1.0, 2.0), (0.1, 1.0))
    assert with_baseline["grid"]["baseline"] == (0.1, 1.0)
    fig = figs.plot_lr_grid_heatmap(**with_baseline)
    assert "baseline" in _texts(fig)
    figs.plt.close(fig)


def test_bank_copy_keeps_the_compact_frame_layout() -> None:
    forward = torch.eye(2).expand(2, 3, 2, 2, 2).clone()                # (B, N, H, d, d) head blocks
    context = CanonicalFrameContext(forward=forward, inverse=forward.clone(), block_diagonal=True)
    copy = extract._cpu_bank_value(context)
    assert isinstance(copy, CanonicalFrameContext) and copy.block_diagonal is True
    assert torch.equal(copy.forward, forward) and torch.equal(copy.inverse, forward)
    dense = extract._cpu_bank_value(CanonicalFrameContext(forward=torch.eye(4).expand(2, 3, 4, 4).clone(),
                                                          inverse=torch.eye(4).expand(2, 3, 4, 4).clone()))
    assert dense.block_diagonal is False
    decoded: Dict[str, Any] = {}

    def decode(mu: torch.Tensor, sigma: torch.Tensor, *, canonical_frame: Any,
               encoded_sigma: Any = None) -> torch.Tensor:          # the record's encoded covariance (F64)
        decoded["frame"] = canonical_frame
        return mu

    belief = BeliefState(mu=torch.zeros(2, 3, 4), sigma=torch.ones(2, 3, 4), phi=torch.zeros(2, 3, 1))
    extract._decode_inference_record(SimpleNamespace(_decode_belief_with_context=decode),
                                     {"belief": belief, "canonical_frame": copy}, CPU)
    assert decoded["frame"].block_diagonal is True                       # the decode re-host keeps it too


def test_summary_carries_no_always_null_estep_final_keys(tmp_path: Path) -> None:
    cfg = _cfg(max_steps=2)
    torch.manual_seed(5)
    model = VFEModel(cfg)
    artifacts = ra.RunArtifacts(tmp_path / "run", cfg, model)
    loader = DataLoader(TokenWindows((torch.arange(61) * 5 + 2) % V, N), batch_size=B, shuffle=False,
                        drop_last=True)
    losses = T.train(model, loader, cfg, n_steps=2, grad_clip=cfg.grad_clip, log_interval=0, eval_interval=0,
                     device=CPU, logger=LOGGER, artifacts=artifacts, generate_samples=False)
    val_loader = DataLoader(TokenWindows((torch.arange(23) * 3) % V, N, pad_final=True, stride=cfg.eval_stride),
                            batch_size=2, shuffle=False)
    ra.finalize_run(model, artifacts, cfg, final_validation=T.evaluate(model, val_loader), val_loader=val_loader,
                    losses=losses, logger=LOGGER)
    summary = json.loads((tmp_path / "run" / "summary.json").read_text(encoding="utf-8"))
    for key in ("estep_final_iterate_f_per_token", "estep_final_f_per_token", "estep_iterate_evidence"):
        assert key not in summary, key
