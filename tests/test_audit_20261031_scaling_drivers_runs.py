r"""Audit 2026-10-31, package scaling-drivers: end-to-end cells of the two drivers at tiny scale.

A real ``scaling.run_cell`` on a K = 4 model (three training steps, in-memory token streams) binds
``scaling_cell.json`` and ``provenance.json`` to one Git snapshot and judges reuse by the package code digest (A050),
validates the final weights although ``max_steps`` is off the evaluation grid (A124), and its cell is accepted by
``scaling_analysis`` through the shared contract (A356). ``scaling.main`` trains the identical muP anchor pair once
(A373). ``parameter_optimization.main`` reruns a single-seed screen as a multi-seed confirmation without retraining the
screened seed (A371) through ablation's cohort reuse, which also serves a relabeled cell from its own sweep. The
ablation cells are stubs; nothing larger than the tiny scaling cells is trained.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
import torch
from torch.utils.data import DataLoader

import scaling
import scaling_analysis
from tests.profiles import config_values
from vfe3 import run_artifacts as ra
from vfe3.config import VFE3Config
from vfe3.data.datasets import TokenWindows

_DATASET = "wikitext-103"
_CODE_A = {"git_sha": "a" * 40, "git_dirty": False, "git_dirty_fingerprint": None}
_CODE_B = {"git_sha": "a" * 40, "git_dirty": True, "git_dirty_fingerprint": "b" * 64}


def _baseline(**overrides: Any) -> Dict[str, Any]:
    values = dict(vocab_size=11, embed_dim=4, n_heads=2, n_layers=1, n_e_steps=1, max_seq_len=8, batch_size=2,
                  max_steps=3, warmup_steps=1, eval_stride=None, eval_interval=2, log_interval=0, kl_max=None,
                  block_mlp_expansion=2, gauge_group="block_glk", m_relational_score_lr=0.01,
                  m_relational_content_lr=0.0025, m_block_mlp_lr=0.001, m_decode_mu_lr=0.02)
    values.update(overrides)
    return config_values(**values)


def _tiny_loaders(vocab_size: int, seq_len: int, batch_size: int) -> Dict[str, DataLoader]:
    r"""In-memory train, validation and test loaders whose datasets carry a cache-like data identity."""
    from vfe3.train import _loader_data_identity

    loaders: Dict[str, DataLoader] = {}
    for split, n_tokens, train in (("train", 97, True), ("validation", 41, False), ("test", 41, False)):
        tokens = (torch.arange(n_tokens) * 7 % vocab_size).long()
        dataset = TokenWindows(tokens, seq_len, pad_final=not train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=train,
                            generator=torch.Generator().manual_seed(0) if train else None)
        identity = _loader_data_identity(loader, vocab_size)
        identity.pop("iterator")
        identity.update({"dataset": _DATASET, "split": split, "max_tokens": None})
        dataset.data_identity = identity
        loaders[split] = loader
    return loaders


@pytest.fixture
def tiny_scaling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Dict[str, Any]:
    r"""``scaling`` wired to tiny in-memory data, a fixed Git identity and no figure worker."""
    baseline = _baseline()
    loaders = _tiny_loaders(baseline["vocab_size"], baseline["max_seq_len"], baseline["batch_size"])
    sources = {split: loader.dataset.data_identity["source"] for split, loader in loaders.items()}
    monkeypatch.setattr(scaling, "BASELINE", baseline)
    monkeypatch.setattr(scaling, "CONFIG", dict(
        routes=["width"], dry_run=False, seeds=[1], data_seed=3, device="cpu", dataset=_DATASET, max_tokens=None,
        max_steps=None, resume=True, kl_max_per_head_dim=None, retry_failed=False, output_dir=str(tmp_path / "out")))
    monkeypatch.setattr(scaling, "get_loader", lambda dataset, seq_len, batch_size, split, **kwargs: loaders[split])
    monkeypatch.setattr(scaling, "_data_source_identities", lambda dataset: json.loads(json.dumps(sources)))
    monkeypatch.setattr(scaling, "_current_code_identity", lambda: dict(_CODE_A))
    monkeypatch.setattr(scaling, "_tokens_per_char", lambda dataset, split: None)
    monkeypatch.setattr(ra, "_git_code_identity", lambda *args, **kwargs: dict(_CODE_A))
    monkeypatch.setattr(ra, "_run_figures_isolated", lambda *args, **kwargs: None)
    return {"sources": sources, "out": tmp_path / "out"}


def test_scaling_cell_binds_one_snapshot_validates_its_final_weights_and_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiny_scaling: Dict[str, Any],
) -> None:
    # The tree changes between the invocation snapshot (A) and the cell start (B), as an edit to a note would.
    monkeypatch.setattr(ra, "_git_code_identity", lambda *args, **kwargs: dict(_CODE_B))
    package = ra._package_code_identity()
    cell = {"label": "K4", "route": "width", "scale_knob": "embed_dim", "overrides": {"embed_dim": 4}}
    root = tmp_path / "cells"
    run_dir = scaling._trusted_scaling_run_dir(root, "width", "K4", 1)
    result = scaling.run_cell(cell, run_dir, 1, dataset=_DATASET, device=torch.device("cpu"),
                              source_identities=tiny_scaling["sources"], code_identity=_CODE_A,
                              package_code_sha256=package, route_key="width")
    assert result["error_kind"] is None and scaling._scaling_result_status(result) == "complete"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    # A124: three steps at eval_interval 2 end off the evaluation grid; the final weights are still validated
    # and offered to the best-model record.
    assert math.isfinite(summary["final_val_ppl"])
    final_validation = json.loads((run_dir / "final_validation.json").read_text(encoding="utf-8"))
    assert final_validation["step"] == 3 and final_validation["ppl"] == summary["final_val_ppl"]
    assert summary["best_val_ppl"] <= summary["final_val_ppl"] and summary["best_step"] in (2, 3)
    # A050: both records describe the snapshot the cell started from, and the package digest binds them.
    cellmeta = json.loads((run_dir / "scaling_cell.json").read_text(encoding="utf-8"))
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    assert cellmeta["code_identity"] == _CODE_B
    assert {key: provenance[key] for key in _CODE_B} == _CODE_B
    assert cellmeta["package_code_sha256"] == provenance["code_identity_sha256"] == package
    cfg = VFE3Config(**scaling._cell_cfg_dict(cell["overrides"], 1, None))
    status = scaling._cell_resume_status(run_dir, cfg, _DATASET, source_identities=tiny_scaling["sources"],
                                         code_identity=_CODE_A, package_code_sha256=package)
    assert status == "current"
    assert scaling._cell_resume_status(run_dir, cfg, _DATASET, source_identities=tiny_scaling["sources"],
                                       code_identity=_CODE_A, package_code_sha256="d" * 64) == "recorded_prior_code"
    # A356: the analysis reader accepts the writer's cell through the shared contract.
    rows = scaling_analysis.harvest(root)
    assert [(row["route"], row["label"], row["seed"]) for row in rows] == [("width", "K4", 1)]
    assert rows[0]["test_ce"] == summary["test_ce"]


def test_identical_mup_anchor_pair_is_trained_once_and_both_cells_are_accepted(
    monkeypatch: pytest.MonkeyPatch, tiny_scaling: Dict[str, Any],
) -> None:
    monkeypatch.setattr(scaling, "ROUTES", {"grow_K_mup": scaling.route_grow_k_mup([4, 8], n_heads=2, anchor_k=4)})
    monkeypatch.setitem(scaling.CONFIG, "routes", ["grow_K_mup"])
    trained: List[str] = []
    real_train = scaling.train

    def counting_train(model: Any, loader: Any, cfg: VFE3Config, **kwargs: Any) -> Any:
        trained.append(f"K{cfg.embed_dim}")
        return real_train(model, loader, cfg, **kwargs)

    monkeypatch.setattr(scaling, "train", counting_train)
    assert scaling.main() == 0
    assert trained == ["K4", "K8", "K8"]                     # K4_fixed, K8_fixed, K8_mup; K4_mup is its twin
    out = tiny_scaling["out"]
    twin = json.loads((out / "grow_K_mup" / "K4_mup" / "s1" / "scaling_cell.json").read_text(encoding="utf-8"))
    assert twin["label"] == "K4_mup" and twin["reused_from"] == "grow_K_mup/K4_fixed/s1"
    design = json.loads((out / "scaling_design.json").read_text(encoding="utf-8"))
    assert design["status"] == "complete"
    assert {cell["label"]: cell.get("reused_from") for cell in design["cells"]}["K4_mup"] == "grow_K_mup/K4_fixed/s1"
    rows = {row["label"]: row for row in scaling_analysis.harvest(out)}
    assert set(rows) == {"K4_fixed", "K4_mup", "K8_fixed", "K8_mup"}
    assert rows["K4_mup"]["effective_run_key"] == rows["K4_fixed"]["effective_run_key"]
    assert rows["K4_mup"]["test_ce"] == rows["K4_fixed"]["test_ce"]
    assert scaling_analysis._requested_design(out, list(rows.values()))["complete"] is True
    assert scaling.main() == 0 and len(trained) == 3          # a resumed invocation serves all four cells


# =============================================================================
# A371: a multi-seed confirmation reuses the screened seed
# =============================================================================

_FIXED_CODE = {"git_sha": "a" * 40, "git_dirty": False, "git_dirty_fingerprint": None}


def _fake_source(dataset: str, split: str, *, cache_dir: Any = None) -> Dict[str, Any]:
    return {"format": "pt", "tokenizer_tag": "tiktoken", "size_bytes": len(split),
            "sha256": "0" * 64 + split, "meta": None, "meta_sha256": None}


def _stub_ablation_cells(monkeypatch: pytest.MonkeyPatch) -> List[Tuple[str, int]]:
    r"""Replace ``ablation.run_single`` by a seed-aware stub that records each trained (label, seed)."""
    import ablation

    monkeypatch.setattr(ablation, "_git_code_identity", lambda: dict(_FIXED_CODE))
    monkeypatch.setattr(ablation, "cache_source_identity", _fake_source)
    monkeypatch.setattr(ablation, "_cleanup", lambda: None)
    monkeypatch.setattr(ablation, "DATA_SEED", None)
    calls: List[Tuple[str, int]] = []

    def fake_run_single(label: str, overrides: Dict[str, Any], run_dir: Path, **kwargs: Any) -> Dict[str, Any]:
        seed = int(kwargs["seed"])
        calls.append((label, seed))
        run_dir.mkdir(parents=True, exist_ok=True)
        ppl = 10.0 + 100.0 * float(overrides.get("m_phi_lr", 0.0)) + 0.01 * seed
        return {"label": label, "error_kind": None, "seed": seed, "overrides": ablation._jsonable(overrides),
                "primary_val_ppl": ppl, "final_val_ppl": ppl, "skipped_steps": 0, "accepted_update_frac": 1.0,
                "_loaded_data_sources": {s: _fake_source(_DATASET, s) for s in ("train", "validation")}}

    monkeypatch.setattr(ablation, "run_single", fake_run_single)
    return calls


def _joint_search(output_dir: Path, seeds: List[int]) -> Dict[str, Any]:
    return {
        "list_only": False, "device": "cpu", "dataset": _DATASET, "max_tokens": None, "max_steps": None,
        "n_trials": 3, "sobol_seed": 5, "seeds": seeds, "resume": True, "output_dir": str(output_dir),
        "parameters": {
            "m_phi_lr": {"enabled": True, "bounds": [0.001, 0.1], "values": None, "scale": "log", "requires": {}},
        },
    }


def test_multiseed_confirmation_reuses_the_screened_seed_and_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ablation
    import parameter_optimization as po

    calls = _stub_ablation_cells(monkeypatch)
    monkeypatch.setattr(ablation, "principled_exact_route", False)
    monkeypatch.setattr(ablation, "fully_gauge_covariant", False)
    monkeypatch.setattr(po, "config", config_values(max_seq_len=8, max_steps=2, warmup_steps=1))
    monkeypatch.setattr(po, "_try_write_sweep_figures", lambda *args, **kwargs: ("skipped", [], [], None))
    output_dir = tmp_path / "search"
    monkeypatch.setattr(po, "SEARCH_CONFIG", _joint_search(output_dir, [6]))
    assert po.main() == 0
    assert sorted(calls) == [("baseline", 6), ("joint_0001", 6), ("joint_0002", 6)]
    calls.clear()
    monkeypatch.setattr(po, "SEARCH_CONFIG", _joint_search(output_dir, [6, 23]))
    assert po.main() == 0                                    # complete: every candidate has both seeds
    assert sorted(calls) == [("baseline__s23", 23), ("joint_0001__s23", 23), ("joint_0002__s23", 23)]
    with open(output_dir / "leaderboard.csv", newline="", encoding="utf-8") as handle:
        board = list(csv.DictReader(handle))
    assert {row["label"]: int(row["successful_seeds"]) for row in board} == {
        "baseline": 2, "joint_0001": 2, "joint_0002": 2}
    status = json.loads((output_dir / "optimization_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete" and status["admitted_results"] == 6


def test_relabeled_cell_is_reused_from_its_own_sweep(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import ablation

    calls = _stub_ablation_cells(monkeypatch)
    monkeypatch.setattr(ablation, "fully_gauge_covariant", False)
    monkeypatch.setattr(ablation, "principled_exact_route", False)
    monkeypatch.setattr(ablation, "BASELINE_CONFIG", config_values(max_seq_len=8, max_steps=2, warmup_steps=1))
    kwargs: Dict[str, Any] = dict(dataset=_DATASET, device=torch.device("cpu"), seed=6, resume=True,
                                  save_model_artifacts=False)
    monkeypatch.setattr(ablation, "SWEEPS", {"probe": {"description": "x", "include_baseline": False,
                                                        "configs": [{"label": "old_name", "lambda_beta": 0.5}]}})
    ablation.run_sweep("probe", tmp_path, **kwargs)
    monkeypatch.setattr(ablation, "SWEEPS", {"probe": {"description": "x", "include_baseline": False,
                                                        "configs": [{"label": "new_name", "lambda_beta": 0.5}]}})
    ablation.run_sweep("probe", tmp_path, **kwargs)
    assert calls == [("old_name", 6)]
    marker = json.loads((tmp_path / "probe" / ablation._sanitize("new_name") / "ablation_result.json")
                        .read_text(encoding="utf-8"))
    assert marker["reused_from"] == f"probe/{ablation._sanitize('old_name')}" and marker["label"] == "new_name"
    assert {row["label"] for row in ablation._collect_sweep_results(tmp_path / "probe")} >= {"new_name"}
