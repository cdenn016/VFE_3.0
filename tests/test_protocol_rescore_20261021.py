r"""Re-scoring checkpoints under one shared evaluation protocol (:mod:`vfe3.protocol_rescore`, 2026-10-21).

On the tiny fake run of ``tests/test_posthoc_20260928.py`` (vocabulary 24, K = 4, two heads, 8-token windows at
stride 4): the run's own protocol reproduces the plain evaluation; a shorter protocol (N = 4, s = 2) equals
``vfe3.train.evaluate`` over those windows and scores the same number of targets; invalid protocols and an existing
output are refused; no run artifact is modified.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pytest
import torch

from tests.test_posthoc_20260928 import DEVICE, _fake_run
from vfe3.posthoc import evaluate as ev
from vfe3.posthoc import streams as st
from vfe3.protocol_rescore import ProtocolRescoreOptions, comparison_table, run_protocol_rescore


@pytest.fixture(autouse=True)
def _one_thread() -> None:
    torch.set_num_threads(1)


def _snapshot(run_dir: Path) -> Dict[str, bytes]:
    return {str(p.relative_to(run_dir)): p.read_bytes() for p in sorted(run_dir.rglob("*")) if p.is_file()}


def _plain(run_dir: Path, cache_dir: Path, checkpoint: str, N: int, s: int, split: str) -> Dict[str, float]:
    loaded = ev.load_model(run_dir, checkpoint, device=DEVICE, precision="fp32")
    info = st.load_run_info(run_dir, checkpoint)
    tokens, _ = st.resolve_stream(info, split, cache_dir=cache_dir)
    dataset = st.eval_windows(tokens, N, s)
    return ev.protocol_evaluate(loaded.model, dataset, range(len(dataset)), batch_size=info.cfg.batch_size,
                                device=DEVICE)


def test_own_and_shorter_protocols_match_plain_evaluation(tmp_path: Path) -> None:
    run_dir, cache_dir = _fake_run(tmp_path)
    before = _snapshot(run_dir)
    entries = ((str(run_dir), "final_model.pt"), (str(run_dir), "best_model.pt"))
    for N, s in ((8, 4), (4, 2)):
        options = ProtocolRescoreOptions(entries=entries, seq_len=N, eval_stride=s, device="cpu", precision="fp32",
                                         cache_dir=str(cache_dir))
        report = run_protocol_rescore(tmp_path / f"out_{N}_{s}.json", options=options, log=lambda _: None)
        for entry in report["results"]:
            for split in ("validation", "test"):
                plain = _plain(run_dir, cache_dir, entry["checkpoint"], N, s, split)
                assert entry["splits"][split]["ce"] == float(plain["ce"])
        assert report["protocol"]["min_context"] == N - s
        assert len(comparison_table(report).splitlines()) == 3
    own = run_protocol_rescore(tmp_path / "own.json", options=ProtocolRescoreOptions(
        entries=entries[:1], seq_len=8, eval_stride=4, device="cpu", precision="fp32", cache_dir=str(cache_dir)),
        log=lambda _: None)
    short = run_protocol_rescore(tmp_path / "short.json", options=ProtocolRescoreOptions(
        entries=entries[:1], seq_len=4, eval_stride=2, device="cpu", precision="fp32", cache_dir=str(cache_dir)),
        log=lambda _: None)
    for split in ("validation", "test"):
        own_split, short_split = own["results"][0]["splits"][split], short["results"][0]["splits"][split]
        targets = [key for key in own_split if "target" in key and "scored" in key]
        for key in targets:                                          # the same targets, scored once each
            assert own_split[key] == short_split[key]
    assert _snapshot(run_dir) == before


def test_invalid_protocols_and_existing_output_are_refused(tmp_path: Path) -> None:
    run_dir, cache_dir = _fake_run(tmp_path)
    entries = ((str(run_dir), "final_model.pt"),)
    for N, s in ((8, 9), (8, 0), (1, 1)):
        with pytest.raises(ValueError):
            run_protocol_rescore(tmp_path / "x.json", options=ProtocolRescoreOptions(
                entries=entries, seq_len=N, eval_stride=s, device="cpu", cache_dir=str(cache_dir)), log=lambda _: None)
    (tmp_path / "taken.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_protocol_rescore(tmp_path / "taken.json", options=ProtocolRescoreOptions(
            entries=entries, device="cpu", cache_dir=str(cache_dir)), log=lambda _: None)
