r"""Figure and run-log contracts, 2026-10-16 deep audit, lane C1 (findings 23, 24 and 25).

F23: ``loss_curve.png`` drew the optimizer objective under the title "Training cross-entropy" (the series is
the CSV's ``train_loss``, which adds the second-order decode, self-coupling, CG and EM terms when those are on),
and a resumed run drew its continuation losses at steps ``1..n`` with continuation-relative epoch guides while
every other trajectory in the same directory used the absolute ``step`` column. The figure request now carries
the absolute step before the first loss (``final_step - len(losses)``), and the curve is labelled as the
training loss.

Tiny CPU models only (K=4, two heads, N=4, vocab 7).
"""
import csv
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
from tests.test_audit2_20261011_training import _cfg
from vfe3.config import VFE3Config
from vfe3.data.datasets import TokenWindows
from vfe3.model.model import VFEModel
from vfe3.viz import figure_worker

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
