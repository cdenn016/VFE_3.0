r"""Figure and run-log contracts, 2026-10-16 deep audit, lane C1 (findings 23, 24 and 25).

F23: ``loss_curve.png`` drew the optimizer objective under the title "Training cross-entropy" (the series is
the CSV's ``train_loss``, which adds the second-order decode, self-coupling, CG and EM terms when those are on),
and a resumed run drew its continuation losses at steps ``1..n`` with continuation-relative epoch guides while
every other trajectory in the same directory used the absolute ``step`` column. The figure request now carries
the absolute step before the first loss (``final_step - len(losses)``), and the curve is labelled as the
training loss.

F24: ``numerical_health`` read layer 0's converged belief for the mu/Sigma/phi finiteness rows and the
largest condition number, but the shared ``conditional_capture`` (the LAST block's, ``stack.py``) for the
energy/beta rows, and ``numerical_trust.png`` presented both as one population; the guard rows come from the
last layer. The extractor now pools every layer (summed non-finite counts over summed entry counts, the
maximum condition over layers, the covariance layout of the family) and the figure names each row group's
layer scope.

F25: ``beta_channel_decomposition.png`` and the run log called ``content_channel`` "content energy" and
dropped ``rope_position_channel``, although the writer flags ``energy_includes_rope_position`` under every
``rope_mode`` other than ``'none'``. Under that flag both now label the bar as the routing cost (content plus
RoPE position) and show the rotation-only arm beside it; a record without the flag keeps the two bars.

Tiny CPU models only (K=4, two heads, N=4, vocab 7; F24 runs ``tests.profiles.live_config`` with three layers).
"""
import csv
import dataclasses
import json
import logging
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import pytest
import torch
from torch.utils.data import DataLoader

import vfe3.run_artifacts as ra
import vfe3.train as T
import vfe3.viz.figures as figs
import matplotlib.pyplot as plt

from tests.profiles import live_config
from tests.test_audit2_20261011_training import _cfg
from vfe3 import metrics
from vfe3.config import VFE3Config
from vfe3.data.datasets import TokenWindows
from vfe3.model.model import VFEModel
from vfe3.viz import extract, figure_worker

B, N, V = 2, 4, 7
CPU = torch.device("cpu")
LOGGER = logging.getLogger("audit-c1-figure-contracts")


@pytest.fixture(autouse=True)
def _single_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T, "_save_eval_attention_maps", lambda *args, **kwargs: None)
    torch.set_num_threads(1)


def _trajectory_spy(monkeypatch: pytest.MonkeyPatch) -> List[Dict[str, Any]]:
    r"""Record every ``plot_trajectory`` call (title, x, epoch guides) without rendering it."""
    calls: List[Dict[str, Any]] = []

    def spy(values: Any, steps: Any = None, **kwargs: Any) -> None:
        calls.append({"values": [float(v) for v in values],
                      "x": None if steps is None else [float(s) for s in steps], **kwargs})
        return None

    monkeypatch.setattr(figs, "plot_trajectory", spy)
    return calls


def _loss_call(calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    # _save_figures renders into a reserved sibling ``.loss_curve.<random>.tmp.png`` and then publishes it.
    matches = [call for call in calls if Path(str(call.get("path", ""))).name.startswith(".loss_curve.")]
    assert len(matches) == 1, [call.get("title") for call in calls]
    return matches[0]


# ------------------------------------------------------------------------------------------------ F23

def test_loss_curve_is_labelled_as_the_training_loss(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _trajectory_spy(monkeypatch)
    history = [{"step": float(s), "steps_per_epoch": 3.0} for s in range(1, 9)]
    artifacts = SimpleNamespace(run_dir=tmp_path, cfg=None, history=history)
    ra._save_figures(artifacts, [2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3], LOGGER)
    call = _loss_call(calls)
    assert call["title"] == "Training loss"
    assert call["ylabel"] == "train loss (nats/token)"
    assert call["x"] == [float(s) for s in range(1, 9)]                    # from scratch: x is 1..n
    assert list(call["epoch_boundaries"]) == [3, 6]


def test_loss_curve_offsets_to_the_absolute_step(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _trajectory_spy(monkeypatch)
    history = [{"step": float(s), "steps_per_epoch": 3.0} for s in range(5, 9)]
    artifacts = SimpleNamespace(run_dir=tmp_path, cfg=None, history=history)
    ra._save_figures(artifacts, [1.7, 1.6, 1.5, 1.4], LOGGER, losses_start_step=4)
    call = _loss_call(calls)
    assert call["x"] == [5.0, 6.0, 7.0, 8.0]
    assert list(call["epoch_boundaries"]) == [6]                           # absolute multiples in (4, 8]


def _loader() -> DataLoader:
    tokens = (torch.arange(61) * 5 + 2) % V                               # 15 windows: 7 batches per epoch
    return DataLoader(TokenWindows(tokens, N), batch_size=B, shuffle=False, drop_last=True)


def _run(cfg: VFE3Config, run_dir: Path, *, n_steps: int,
         resume_from: Any = None) -> Tuple[List[float], ra.RunArtifacts]:
    torch.manual_seed(5)
    model = VFEModel(cfg)
    artifacts = ra.RunArtifacts(run_dir, cfg, model)
    torch.manual_seed(11)
    losses = T.train(model, _loader(), cfg, n_steps=n_steps, grad_clip=cfg.grad_clip,
                     log_interval=1, eval_interval=0, device=CPU, logger=LOGGER,
                     artifacts=artifacts, generate_samples=False, resume_from=resume_from)
    return losses, artifacts


def test_resumed_run_figure_request_places_the_loss_curve_at_absolute_steps(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(checkpoint_interval=6, max_steps=12)
    _run(cfg, tmp_path / "straight", n_steps=12)
    checkpoint = tmp_path / "straight" / "checkpoints" / "step_6.pt"
    losses, artifacts = _run(cfg, tmp_path / "resumed", n_steps=12, resume_from=checkpoint)
    assert len(losses) == 6 and artifacts.final_step == 12

    captured: Dict[str, Any] = {}

    def fake_process(command: List[str], **kwargs: Any) -> subprocess.CompletedProcess:
        request_path = Path(kwargs["env"]["VFE3_FIGURE_REQUEST"])
        captured.update(json.loads(request_path.read_text(encoding="utf-8")))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(ra, "run_process_tree", fake_process)
    assert ra._run_figures_isolated(torch.nn.Linear(1, 1), artifacts, losses, LOGGER,
                                    generate_publication=False)
    assert captured["losses_start_step"] == 6

    validated = figure_worker._validated_finalize_request(captured)
    assert validated[1] == losses and validated[2] == 6
    calls = _trajectory_spy(monkeypatch)
    figure_worker._render_persisted_run_figures(
        artifacts.run_dir, cfg, validated[1], LOGGER, losses_start_step=validated[2])
    call = _loss_call(calls)
    with (artifacts.run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        logged_steps = [float(row["step"]) for row in csv.DictReader(handle)]
    assert logged_steps == [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    assert call["x"] == logged_steps                                       # the CSV's absolute step axis
    assert list(call["epoch_boundaries"]) == [7]                           # 7 batches per epoch


def _finalize_request(**overrides: Any) -> Dict[str, Any]:
    request: Dict[str, Any] = {
        "run_dir": "run", "losses": [1.0, 0.5], "losses_start_step": 3,
        "generate_publication": False, "report_batches_path": None, "model_bundle_path": None,
        "device": "cpu", "max_tokens": 16, "generate_embedding_figures": False,
    }
    request.update(overrides)
    return request


@pytest.mark.parametrize("value", [True, -1, 2.0, "3"])
def test_finalize_request_rejects_a_malformed_loss_start_step(value: Any) -> None:
    with pytest.raises(ValueError, match="losses_start_step"):
        figure_worker._validated_finalize_request(_finalize_request(losses_start_step=value))


def test_finalize_request_requires_the_loss_start_step() -> None:
    request = _finalize_request()
    del request["losses_start_step"]
    with pytest.raises(ValueError, match="losses_start_step"):
        figure_worker._validated_finalize_request(request)


# ------------------------------------------------------------------------------------------------ F24

def _three_layer_model() -> VFEModel:
    torch.manual_seed(0)
    return VFEModel(live_config(n_layers=3)).eval()


def _layer_conditions(model: VFEModel, snapshot: Any) -> List[float]:
    return [float(metrics.belief_spectrum(
        belief.sigma[0], diagonal=model.cfg.diagonal_covariance, eps=model.cfg.eps,
        family=model.cfg.family)["condition"].max()) for belief in snapshot.layer_converged]


def test_numerical_health_pools_every_layer() -> None:
    model = _three_layer_model()
    tokens = torch.randint(0, model.cfg.vocab_size, (1, model.cfg.max_seq_len))
    snapshot = model.build_diagnostic_snapshot(tokens)
    health = extract.numerical_health(model, tokens, snapshot=snapshot)
    conditions = _layer_conditions(model, snapshot)
    assert max(conditions) != conditions[0], "the seed must make a deeper layer the worst conditioned"
    assert health["n_layers"] == 3
    assert health["max_condition"] == pytest.approx(max(conditions), rel=1e-6)
    per_layer = model.diagnostics_per_layer(tokens, snapshot=snapshot)    # B = 1: the same population
    assert health["max_condition"] == pytest.approx(max(per_layer["belief_cond_max"]), rel=1e-6)
    assert health["numel_mu"] == sum(b.mu[0].numel() for b in snapshot.layer_converged)
    assert health["numel_sigma"] == sum(b.sigma[0].numel() for b in snapshot.layer_converged)
    assert health["numel_beta"] == sum(c["beta_pre"][0].numel() for c in snapshot.layer_conditional_captures)
    assert health["numel_energy"] == sum(
        c["score_energy_pre"][0].numel() for c in snapshot.layer_conditional_captures)
    assert all(health[f"nan_{name}"] == 0.0 for name in ("mu", "sigma", "phi", "energy", "beta"))


def test_numerical_health_sees_a_middle_layer_nonfinite_entry() -> None:
    model = _three_layer_model()
    tokens = torch.randint(0, model.cfg.vocab_size, (1, model.cfg.max_seq_len))
    snapshot = model.build_diagnostic_snapshot(tokens)
    beliefs = list(snapshot.layer_converged)
    mu = beliefs[1].mu.clone()
    mu[0, 0, 0] = float("nan")
    beliefs[1] = beliefs[1]._replace(mu=mu)
    captures = [dict(cap) for cap in snapshot.layer_conditional_captures]
    beta = captures[0]["beta_pre"].clone()
    beta[0, 0, -1, 0] = float("inf")
    captures[0]["beta_pre"] = beta
    poisoned = dataclasses.replace(snapshot, layer_converged=tuple(beliefs),
                                   layer_conditional_captures=tuple(captures))
    health = extract.numerical_health(model, tokens, snapshot=poisoned)
    assert health["nan_mu"] * health["numel_mu"] == pytest.approx(1.0)  # one of the pooled entries
    assert health["nan_beta"] * health["numel_beta"] == pytest.approx(1.0)


def test_numerical_trust_names_each_row_groups_layer_scope(tmp_path: Path) -> None:
    guard = {"sigma_floor_frac": 0.25, "sigma_ceil_frac": 0.0, "energy_klmax_frac": 0.0}
    health = {"nan_mu": 0.0, "nan_beta": 0.0, "numel_mu": 12, "numel_beta": 48,
              "max_condition": 3.0, "n_layers": 3}
    fig = figs.plot_numerical_trust(
        guard, health, guard_denominators={"sigma_floor_frac": 8, "sigma_ceil_frac": 8,
                                           "energy_klmax_frac": 16},
        guard_scope="last layer", population="test split, one 4-token sequence",
        path=str(tmp_path / "t.png"))
    cells = {(r, c): cell.get_text().get_text()
             for (r, c), cell in fig.axes[0].tables[0].get_celld().items()}
    labels = [cells[(r, 0)] for r in range(1, 1 + max(key[0] for key in cells))]
    assert r"variance at the floor $\varepsilon$ (last layer)" in labels
    assert r"non-finite $\mu$ (all 3 layers)" in labels
    assert r"non-finite $\beta$ (all 3 layers)" in labels
    assert "largest covariance condition number (all 3 layers): 3" in fig.axes[0].get_title()
    plt.close(fig)


def test_report_marks_the_guard_rows_as_last_layer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vfe3.viz import report

    torch.manual_seed(71)
    model = VFEModel(_cfg(n_layers=2)).eval()
    seen: Dict[str, Any] = {}
    trust = figs.plot_numerical_trust
    monkeypatch.setattr(figs, "plot_numerical_trust",
                        lambda guard, health, *a, **kw: seen.update(health=health, **kw)
                        or trust(guard, health, *a, **kw))
    tokens, targets = torch.tensor([[0, 1, 2, 3]]), torch.tensor([[1, 2, 3, 4]])
    written = report.generate_figures(tmp_path, model=model, loader=[(tokens, targets)],
                                      device=CPU, max_sequences=1)
    assert "numerical_trust.png" in {path.name for path in written}
    assert seen["guard_scope"] == "last layer"
    assert seen["health"]["n_layers"] == 2


# ------------------------------------------------------------------------------------------------ F25

ROPE_RECORD: Dict[str, Any] = {
    "available": True, "energy_includes_rope_position": True,
    "delta_ce": {"content_channel": 0.022, "positional_channel": 0.005, "rope_position_channel": 0.047},
}
NO_ROPE_RECORD: Dict[str, Any] = {
    "available": True, "energy_includes_rope_position": False,
    "delta_ce": {"content_channel": 0.074, "positional_channel": 0.003},
}


def _tick_labels(record: Dict[str, Any], tmp_path: Path) -> List[str]:
    fig = figs.plot_beta_channel_decomposition(record, path=str(tmp_path / "beta.png"))
    labels = [tick.get_text() for tick in fig.axes[0].get_xticklabels()]
    plt.close(fig)
    return labels


def test_beta_channel_figure_shows_the_rope_arm_and_relabels_the_routing_cost(tmp_path: Path) -> None:
    labels = _tick_labels(ROPE_RECORD, tmp_path)
    assert labels == [figs.BETA_CHANNEL_POSITIONAL_LABEL, figs.BETA_CHANNEL_ROUTING_COST_LABEL,
                      figs.BETA_CHANNEL_ROPE_LABEL]
    assert "RoPE position" in labels[1] and "content" in labels[1]
    assert "rotation only" in labels[2]


@pytest.mark.parametrize("record", [NO_ROPE_RECORD, {"available": True, "delta_ce": {
    "content_channel": 0.074, "positional_channel": 0.003}}], ids=["rope-none", "pre-2026-10-10"])
def test_beta_channel_figure_keeps_two_bars_without_rope(record: Dict[str, Any], tmp_path: Path) -> None:
    assert _tick_labels(record, tmp_path) == ["positional prior", "content energy"]


def test_mechanism_log_names_the_rope_arm(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("audit-c1-beta-log")
    with caplog.at_level(logging.INFO, logger="audit-c1-beta-log"):
        ra._log_mechanism_diagnostics(logger, None, ROPE_RECORD, None)
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("Attention channels"))
    assert "content energy" not in line
    assert "routing cost (content + RoPE position) +0.0220" in line
    assert "RoPE position (rotation only) +0.0470" in line
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="audit-c1-beta-log"):
        ra._log_mechanism_diagnostics(logger, None, NO_ROPE_RECORD, None)
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("Attention channels"))
    assert "content energy +0.0740" in line and "RoPE" not in line


@pytest.mark.parametrize("rope_mode", ["comparison_invariant", "none"])
def test_beta_channel_writer_and_figure_agree_on_the_rope_reading(rope_mode: str, tmp_path: Path) -> None:
    torch.manual_seed(0)
    cfg = _cfg(rope_mode=rope_mode, n_layers=2)
    model = VFEModel(cfg)
    tokens = torch.randint(0, V, (2, N))
    targets = torch.randint(0, V, (2, N))
    record = ra.collect_beta_channel_decomposition(model, tokens, targets)
    assert record["available"]
    labels = _tick_labels(record, tmp_path)
    if rope_mode == "none":
        assert labels == ["positional prior", "content energy"]
    else:
        assert record["energy_includes_rope_position"] is True
        assert len(labels) == 3 and labels[2] == figs.BETA_CHANNEL_ROPE_LABEL
