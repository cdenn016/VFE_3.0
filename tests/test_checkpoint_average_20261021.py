r"""Checkpoint averaging under the run's evaluation protocol (:mod:`vfe3.checkpoint_average`; audit 2026-10-20 item 8).

On a tiny fake run (vocabulary 24, K = 4, two heads, two layers, 8-token windows at stride 4; the run directory of
``tests/test_posthoc_20260928.py``): averaging two identical checkpoints reproduces the original evaluation exactly;
averaging two different ones equals the evaluation of the hand-averaged state dict; non-floating tensors come from
the last checkpoint; incompatible bundles and protected output names are refused; no run artifact is modified.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict

import pytest
import torch

from tests.test_posthoc_20260928 import DEVICE, _fake_run
from vfe3.checkpoint_average import (CheckpointAverageOptions, average_model_states, check_compatible,
                                     load_model_bundle, run_checkpoint_average)
from vfe3.posthoc import evaluate as ev
from vfe3.posthoc import streams as st


@pytest.fixture(autouse=True)
def _one_thread() -> None:
    torch.set_num_threads(1)


def _options(cache_dir: Path, **over: object) -> CheckpointAverageOptions:
    return CheckpointAverageOptions(**{**dict(device="cpu", precision="fp32", cache_dir=str(cache_dir)), **over})


def _reference(run_dir: Path, cache_dir: Path, state: Dict[str, torch.Tensor]) -> Dict[str, float]:
    r"""``vfe3.train.evaluate`` of ``state`` on the run's full validation and test windows."""
    loaded = ev.load_model(run_dir, "final_model.pt", device=DEVICE, precision="fp32")
    loaded.model.load_state_dict(state, strict=True)
    info = st.load_run_info(run_dir, "final_model.pt")
    out = {}
    for split in ("validation", "test"):
        tokens, _ = st.resolve_stream(info, split, cache_dir=cache_dir)
        dataset = st.eval_windows(tokens, info.seq_len, info.eval_stride)
        out[split] = float(ev.protocol_evaluate(loaded.model, dataset, range(len(dataset)), batch_size=2,
                                                device=DEVICE)["ce"])
    return out


def _snapshot(run_dir: Path) -> Dict[str, bytes]:
    return {str(p.relative_to(run_dir)): p.read_bytes() for p in sorted(run_dir.rglob("*")) if p.is_file()}


def test_two_identical_checkpoints_reproduce_the_original_evaluation(tmp_path: Path) -> None:
    run_dir, cache_dir = _fake_run(tmp_path)
    shutil.copy(run_dir / "final_model.pt", run_dir / "final_copy.pt")
    before = _snapshot(run_dir)
    report = run_checkpoint_average(run_dir, options=_options(cache_dir, checkpoints=("final_model.pt",
                                                                                      "final_copy.pt")),
                                    log=lambda _: None)
    state, _ = load_model_bundle(run_dir / "final_model.pt")
    expected = _reference(run_dir, cache_dir, state)
    for split in ("validation", "test"):
        assert report["averaged"][split]["ce"] == expected[split]                 # bitwise: (a + a)/2 = a
        assert report["streams"][split]["status"] == "verified"
    after = _snapshot(run_dir)
    assert set(after) - set(before) == {"averaged_eval.json"}
    assert all(after[name] == data for name, data in before.items())          # no run artifact modified
    written = json.loads((run_dir / "averaged_eval.json").read_text(encoding="utf-8"))
    assert written["checkpoints"] == ["final_model.pt", "final_copy.pt"]
    assert written["protocol"]["seq_len"] == 8 and written["protocol"]["eval_stride"] == 4


def test_two_different_checkpoints_equal_the_hand_averaged_state(tmp_path: Path) -> None:
    run_dir, cache_dir = _fake_run(tmp_path)
    report = run_checkpoint_average(run_dir, options=_options(cache_dir, evaluate_inputs=True), log=lambda _: None)
    best, _ = load_model_bundle(run_dir / "best_model.pt")
    final, _ = load_model_bundle(run_dir / "final_model.pt")
    manual = {key: ((best[key].double() + final[key].double()) / 2).to(final[key].dtype)
              if torch.is_floating_point(final[key]) else final[key].clone() for key in final}
    expected = _reference(run_dir, cache_dir, manual)
    for split in ("validation", "test"):
        assert report["averaged"][split]["ce"] == pytest.approx(expected[split], rel=0.0, abs=1e-6)
    endpoints = {name: _reference(run_dir, cache_dir, state) for name, state in (("best_model.pt", best),
                                                                               ("final_model.pt", final))}
    for name, scores in endpoints.items():
        for split in ("validation", "test"):
            assert report["inputs"][name][split]["ce"] == pytest.approx(scores[split], rel=0.0, abs=1e-6)
    assert report["averaged"]["validation"]["ce"] != endpoints["final_model.pt"]["validation"]


def test_floating_tensors_are_averaged_and_the_rest_come_from_the_last_state() -> None:
    a = {"w": torch.tensor([1.0, 2.0]), "h": torch.tensor([1.0], dtype=torch.float16), "n": torch.tensor([3])}
    b = {"w": torch.tensor([3.0, 6.0]), "h": torch.tensor([2.0], dtype=torch.float16), "n": torch.tensor([7])}
    out = average_model_states([a, b])
    assert torch.equal(out["w"], torch.tensor([2.0, 4.0]))
    assert out["h"].dtype == torch.float16 and float(out["h"]) == 1.5
    assert torch.equal(out["n"], torch.tensor([7]))


def test_incompatible_bundles_are_refused() -> None:
    a = {"w": torch.zeros(2)}
    with pytest.raises(ValueError, match="different config"):
        check_compatible([a, a], [{"embed_dim": 4}, {"embed_dim": 8}], ["x", "y"])
    with pytest.raises(ValueError, match=r"\(3,\)"):
        check_compatible([a, {"w": torch.zeros(3)}], [{}, {}], ["x", "y"])
    with pytest.raises(ValueError, match="different tensors"):
        check_compatible([a, {"v": torch.zeros(2)}], [{}, {}], ["x", "y"])
    with pytest.raises(ValueError, match="at least two"):
        check_compatible([a], [{}], ["x"])
    check_compatible([a, a], [{"resume_from": None}, {"resume_from": "c.pt"}], ["x", "y"])   # bookkeeping only


def test_protected_output_names_are_refused(tmp_path: Path) -> None:
    for name in ("summary.json", "final_model.pt", "sub/x.json"):
        with pytest.raises(ValueError, match="out_name"):
            run_checkpoint_average(tmp_path, options=_options(tmp_path, out_name=name), log=lambda _: None)
