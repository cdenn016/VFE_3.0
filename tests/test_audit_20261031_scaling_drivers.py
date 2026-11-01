r"""Audit 2026-10-31, package scaling-drivers: config-level and record-level checks of the two experiment drivers.

``parameter_optimization.py`` pins ablation.py's module route switches for the whole search and records the cell
ablation executes (A001), forwards ``retry_failed`` and labels candidates by their Sobol draw (A371). ``scaling.py``
names a cell directory by its ROUTES key (A049), judges reuse and invocation drift by the package code digest
(A050), serves a recorded failure only for the experiment that produced it (A051), states the infer_T requirement
(A302), shares one cell contract with ``scaling_analysis.py`` (A356), predicts the parameter count exactly (A372)
and marks the muP anchor twin (A373). ``scaling_analysis.py`` fits the grow_K_mup arms as separate series (A073)
and finds the shipped layout from its default directory (A109). Tiny configurations from ``tests.profiles``; no
model is trained here (the end-to-end cells are in ``test_audit_20261031_scaling_drivers_runs.py``).
"""
from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch

import scaling
import scaling_analysis as sa
from tests.profiles import config_values, test_config
from tests.test_audit_20261008_stats import _write_scaling_run
from vfe3.artifact_utils import canonical_json_sha256
from vfe3.config import VFE3Config, serialize_config
from vfe3.model.model import VFEModel

_CODE_A = {"git_sha": "a" * 40, "git_dirty": False, "git_dirty_fingerprint": None}
_CODE_B = {"git_sha": "a" * 40, "git_dirty": True, "git_dirty_fingerprint": "b" * 64}
_PACKAGE = "c" * 64


def _sources() -> Dict[str, Dict[str, object]]:
    return {split: {"format": "pt", "tokenizer_tag": "tiktoken", "size_bytes": len(split), "sha256": "0" * 64,
                    "meta": None, "meta_sha256": None} for split in ("train", "validation", "test")}


def _scaling_baseline(**overrides: Any) -> Dict[str, Any]:
    values = dict(vocab_size=11, embed_dim=4, n_heads=2, n_layers=1, n_e_steps=1, max_seq_len=8, batch_size=2,
                  max_steps=3, warmup_steps=1, eval_stride=None, eval_interval=2, log_interval=0,
                  kl_max=None, block_mlp_expansion=2, gauge_group="block_glk")
    values.update(overrides)
    return config_values(**values)


def _scaling_settings(output_dir: Path, **overrides: Any) -> Dict[str, Any]:
    settings = dict(routes=["width"], dry_run=False, seeds=[1], data_seed=3, device="cpu",
                    dataset="wikitext-103", max_tokens=None, max_steps=None, resume=True,
                    kl_max_per_head_dim=None, retry_failed=False, output_dir=str(output_dir))
    settings.update(overrides)
    return settings


class _Stop(Exception):
    pass


def _stop(*args: Any, **kwargs: Any) -> Any:
    raise _Stop


# =============================================================================
# A001: the joint search runs, and records, its own route
# =============================================================================

def _po_model_config(**overrides: Any) -> Dict[str, Any]:
    values = dict(vocab_size=11, embed_dim=8, n_heads=2, n_layers=1, max_seq_len=8, eval_stride=None,
                  batch_size=2, max_steps=4, warmup_steps=1, kl_max=None, block_mlp_expansion=2)
    values.update(overrides)
    return config_values(**values)


def _po_search(**overrides: Any) -> Dict[str, Any]:
    search: Dict[str, Any] = {
        "list_only": False, "device": "cpu", "dataset": "wikitext-103", "max_tokens": None, "max_steps": None,
        "n_trials": 3, "sobol_seed": 7, "seeds": [6], "resume": True, "output_dir": "unused-po-output",
        "parameters": {
            "m_phi_lr": {"enabled": True, "bounds": [0.001, 0.1], "values": None, "scale": "log", "requires": {}},
            "m_p_mu_lr": {"enabled": True, "bounds": [0.001, 0.1], "values": None, "scale": "log",
                          "requires": {}},
        },
    }
    search.update(overrides)
    return search


def test_search_scope_pins_ablation_route_switches_to_the_search_values(monkeypatch: pytest.MonkeyPatch) -> None:
    import ablation
    import parameter_optimization as po

    monkeypatch.setattr(ablation, "principled_exact_route", True)
    monkeypatch.setattr(ablation, "fully_gauge_covariant", True)
    declared = po._declared_model_config(_po_model_config())
    for overrides in ({}, {"m_phi_lr": 0.003}):
        recorded = po._candidate_config(overrides, declared, seed=6, max_steps=None)
        with po._ablation_baseline_scope(declared):
            assert (ablation.principled_exact_route, ablation.fully_gauge_covariant) == (False, False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                executed = asdict(VFE3Config(**ablation._cell_cfg_dict(dict(overrides), seed=6)))
        assert executed == recorded
        assert recorded["predictive_readout"] == declared["predictive_readout"] == "kernel"
    assert (ablation.principled_exact_route, ablation.fully_gauge_covariant) == (True, True)


def test_search_named_exact_route_is_both_executed_and_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    import ablation
    import parameter_optimization as po

    monkeypatch.setattr(ablation, "principled_exact_route", False)
    monkeypatch.setattr(ablation, "fully_gauge_covariant", False)
    declared = po._declared_model_config(_po_model_config())
    search = _po_search(principled_exact_route=True)
    switches = po._search_route_switches(search)
    assert switches == {"principled_exact_route": True, "fully_gauge_covariant": False}
    recorded = po._candidate_config({}, declared, seed=6, max_steps=None, route_switches=switches)
    with po._ablation_baseline_scope(declared, route_switches=switches):
        assert ablation.principled_exact_route is True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            executed = asdict(VFE3Config(**ablation._cell_cfg_dict({}, seed=6)))
    assert ablation.principled_exact_route is False
    assert executed == recorded
    assert (recorded["predictive_readout"], recorded["family"]) == ("minimal_agent", "gaussian_frame_diagonal")
    manifest = po._manifest(search, declared, [{"label": "baseline", "overrides": {}}], [6])
    assert manifest["route_switches"] == switches
    assert manifest["model_config"]["predictive_readout"] == "minimal_agent"
    assert manifest["baseline_fingerprint"] == ablation.semantic_config_fingerprint(recorded)
    with pytest.raises(ValueError, match="fully_gauge_covariant"):
        po._search_route_switches(_po_search(fully_gauge_covariant="yes"))


def test_main_runs_the_sweep_under_the_search_switches_and_forwards_retry_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ablation
    import parameter_optimization as po

    monkeypatch.setattr(ablation, "principled_exact_route", True)
    monkeypatch.setattr(ablation, "fully_gauge_covariant", False)
    monkeypatch.setattr(po, "config", _po_model_config())
    monkeypatch.setattr(po, "SEARCH_CONFIG", _po_search(output_dir=str(tmp_path / "po"), retry_failed=True))
    seen: Dict[str, Any] = {}

    def fake_run_sweep(name: str, output_dir: Path, **kwargs: Any) -> List[Dict[str, Any]]:
        seen.update(name=name, retry_failed=kwargs.get("retry_failed"),
                    switches=(ablation.principled_exact_route, ablation.fully_gauge_covariant))
        return []

    monkeypatch.setattr(ablation, "run_sweep", fake_run_sweep)
    monkeypatch.setattr(po, "_try_write_sweep_figures", lambda *args, **kwargs: ("skipped", [], [], None))
    assert po.main() == 1                                      # nothing admitted by the stub
    assert seen["switches"] == (False, False) and seen["retry_failed"] is True
    assert (ablation.principled_exact_route, ablation.fully_gauge_covariant) == (True, False)
    manifest = json.loads((tmp_path / "po" / "optimization_manifest.json").read_text(encoding="utf-8"))
    assert manifest["route_switches"] == {"principled_exact_route": False, "fully_gauge_covariant": False}
    assert manifest["model_config"]["predictive_readout"] == "kernel"


# =============================================================================
# A371: candidate labels are Sobol draw indices
# =============================================================================

def test_candidate_labels_do_not_shift_with_the_search_baseline_anchor() -> None:
    import parameter_optimization as po

    search = _po_search(n_trials=5)
    search["parameters"]["m_block_mlp_lr"] = {"enabled": True, "bounds": [0.001, 0.1], "values": None,
                                              "scale": "log", "requires": {"block_mlp_expansion": 4}}
    labels = [candidate["label"] for candidate in po.generate_candidates(search, _po_model_config())]
    assert labels == ["baseline", "search_baseline", "joint_0001", "joint_0002", "joint_0003"]


def test_candidate_label_is_the_draw_index_when_draws_are_rejected() -> None:
    import parameter_optimization as po

    search = _po_search(n_trials=6, sobol_seed=11)
    search["parameters"] = {
        "m_phi_lr": {"enabled": True, "bounds": [0.001, 0.1], "values": None, "scale": "log", "requires": {}},
        "n_heads": {"enabled": True, "bounds": None, "values": [2, 3], "scale": "linear", "requires": {}},
    }
    labels = [candidate["label"] for candidate in po.generate_candidates(search, _po_model_config())]
    engine = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=11)
    accepted = []
    for draw in range(1, 200):
        point = engine.draw(1, dtype=torch.float64)[0]
        if [2, 3][min(int(float(point[1]) * 2), 1)] == 2:      # n_heads=3 does not divide embed_dim=8
            accepted.append(f"joint_{draw:04d}")
    assert labels[0] == "baseline" and labels[1:] == accepted[:5]
    assert labels[1:] != [f"joint_{index:04d}" for index in range(1, 6)]


# =============================================================================
# A049 / A051: the failure record, its route and its contract
# =============================================================================

@pytest.fixture
def _tiny_scaling(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(scaling, "BASELINE", _scaling_baseline())
    monkeypatch.setattr(scaling, "CONFIG", _scaling_settings(tmp_path / "out"))
    monkeypatch.setattr(scaling, "ROUTES", {"width": [{"label": "K4", "route": "width", "scale_knob": "embed_dim",
                                                        "overrides": {"embed_dim": 4}}]})
    monkeypatch.setattr(scaling, "_current_code_identity", lambda: dict(_CODE_A))
    monkeypatch.setattr(scaling, "_data_source_identities", lambda dataset: _sources())
    monkeypatch.setattr(scaling, "_current_package_code_identity", lambda: _PACKAGE, raising=False)
    return tmp_path / "out"


def _raising_model(calls: List[str]) -> Any:
    def build(cfg: VFE3Config) -> Any:
        calls.append(f"K{cfg.embed_dim}_L{cfg.n_layers}_B{cfg.batch_size}")
        raise RuntimeError("synthetic model build failure")
    return build


def test_legacy_failure_without_a_cell_record_is_retried_not_served(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _tiny_scaling: Path,
) -> None:
    cell = {"label": "K4", "route": "width", "scale_knob": "embed_dim", "overrides": {"embed_dim": 4}}
    run_dir = scaling._trusted_scaling_run_dir(tmp_path / "cells", "width", "K4", 1)
    failure = {"label": "K4", "route": "width", "seed": 1, "status": "failed", "error_kind": "train",
               "error": "synthetic divergence at batch_size 4"}
    for name in ("scaling_result.json", "scaling_failure.json"):
        (run_dir / name).write_text(json.dumps(failure), encoding="utf-8")
    monkeypatch.setattr(scaling, "predict_n_params", _stop)
    with pytest.raises(_Stop):                              # archived and retrained, not reported as terminal
        scaling.run_cell(cell, run_dir, 1, dataset="wikitext-103", device=torch.device("cpu"),
                         source_identities=_sources(), code_identity=_CODE_A)
    assert (run_dir.parent / ".attempts" / "s1__attempt_0001" / "scaling_failure.json").is_file()


def test_failure_is_terminal_only_for_the_experiment_that_produced_it(
    monkeypatch: pytest.MonkeyPatch, _tiny_scaling: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    calls: List[str] = []
    monkeypatch.setattr(scaling, "VFEModel", _raising_model(calls))
    assert scaling.main() == 1 and len(calls) == 1
    failure_path = _tiny_scaling / "width" / "K4" / "s1" / "scaling_failure.json"
    contract = json.loads(failure_path.read_text(encoding="utf-8"))["attempt_contract"]
    assert contract["max_tokens"] is None and contract["package_code_sha256"] == _PACKAGE
    assert scaling.main() == 1                                 # the same experiment: reported, not retrained
    assert len(calls) == 1 and "FAILED earlier, not retried" in capsys.readouterr().out
    monkeypatch.setitem(scaling.CONFIG, "max_tokens", 1000)    # another experiment: archived and retried
    assert scaling.main() == 1 and len(calls) == 2
    assert (failure_path.parent.parent / ".attempts" / "s1__attempt_0001" / "scaling_failure.json").is_file()


def test_failure_whose_cell_record_names_another_corpus_is_not_terminal(tmp_path: Path) -> None:
    cfg = test_config(**_scaling_baseline())
    contract = scaling._scaling_attempt_contract(cfg, dataset="wikitext-103", max_tokens=None,
                                                 source_identities=_sources(), package_code_sha256=_PACKAGE)
    run_dir = tmp_path / "cell"
    run_dir.mkdir()
    failure = {"label": "K4", "route": "width", "seed": int(cfg.seed), "status": "failed", "error_kind": "nonfinite",
               "error": "loss became NaN"}
    (run_dir / "scaling_failure.json").write_text(json.dumps(failure), encoding="utf-8")
    legacy_cell = {"config_sha256": contract["config_sha256"], "dataset": "wikitext-103", "data_seed": 3,
                   "max_tokens": None, "data_sources": _sources(), "code_identity": dict(_CODE_A)}
    (run_dir / "scaling_cell.json").write_text(json.dumps(legacy_cell), encoding="utf-8")

    def terminal() -> object:
        return scaling._terminal_scaling_failure(run_dir, cfg, route="width", label="K4", seed=int(cfg.seed),
                                                 contract={**contract, "data_seed": 3},
                                                 legacy_code_identity=_CODE_A)

    assert terminal() is not None                              # a legacy record judged by its own cell record
    (run_dir / "scaling_cell.json").write_text(json.dumps({**legacy_cell, "dataset": "wiki-ja"}), encoding="utf-8")
    assert terminal() is None
    (run_dir / "scaling_cell.json").write_text(json.dumps({**legacy_cell, "code_identity": _CODE_B}),
                                               encoding="utf-8")
    assert terminal() is None
    (run_dir / "scaling_failure.json").write_text(
        json.dumps({**failure, "attempt_contract": {**contract, "data_seed": 3}}), encoding="utf-8")
    assert terminal() is not None                              # the contract a new record carries decides
    (run_dir / "scaling_failure.json").write_text(
        json.dumps({**failure, "attempt_contract": {**contract, "data_seed": 3, "max_tokens": 1000}}),
        encoding="utf-8")
    assert terminal() is None


def test_route_whose_key_differs_from_its_tag_is_archived_and_retried(
    monkeypatch: pytest.MonkeyPatch, _tiny_scaling: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    calls: List[str] = []
    monkeypatch.setattr(scaling, "ROUTES", {"infer_L": scaling.route_inference_l([1])})
    monkeypatch.setitem(scaling.CONFIG, "routes", ["infer_L"])
    monkeypatch.setattr(scaling, "VFEModel", _raising_model(calls))
    assert scaling.main() == 1 and len(calls) == 1
    cell_dir = _tiny_scaling / "infer_L" / "L1" / "s1"
    first = json.loads((cell_dir / "scaling_result.json").read_text(encoding="utf-8"))
    assert first["route"] == "infer_L" and first["route_tag"] == "depth"
    assert scaling.main() == 1                                 # reported as terminal, the record left as it was
    assert len(calls) == 1 and "FAILED earlier, not retried" in capsys.readouterr().out
    assert json.loads((cell_dir / "scaling_result.json").read_text(encoding="utf-8")) == first
    monkeypatch.setitem(scaling.CONFIG, "retry_failed", True)
    assert scaling.main() == 1 and len(calls) == 2             # archived and retried
    assert (cell_dir.parent / ".attempts" / "s1__attempt_0001" / "scaling_result.json").is_file()
    retried = json.loads((cell_dir / "scaling_result.json").read_text(encoding="utf-8"))
    assert "ownership sentinel" not in str(retried.get("error"))


# =============================================================================
# A050: invocation drift is judged by the package code digest
# =============================================================================

def _complete_cached(cell: Dict[str, Any], run_dir: Path, seed: int, **kwargs: Any) -> Dict[str, Any]:
    ce = 2.0
    return {"label": cell["label"], "route": cell["route"], "scale_knob": cell["scale_knob"], "error_kind": None,
            "seed": seed, "cached": True, "cache_status": "current", "n_params": 10, "test_ce": ce,
            "test_ppl": math.exp(ce), "test_bits_per_token": ce / math.log(2.0), "test_bpc": None, "kl_max": 1.0,
            "router_dead_heads": {}}


def test_tree_edit_during_an_invocation_leaves_its_design_complete(
    monkeypatch: pytest.MonkeyPatch, _tiny_scaling: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    identities = iter([dict(_CODE_A), dict(_CODE_B)])         # invocation start, invocation end
    monkeypatch.setattr(scaling, "_current_code_identity", lambda: next(identities))
    monkeypatch.setattr(scaling, "run_cell", _complete_cached)
    assert scaling.main() == 0
    design = json.loads((_tiny_scaling / "scaling_design.json").read_text(encoding="utf-8"))
    assert design["status"] == "complete" and "error" not in design
    assert design["git_identity_drift"] is True
    out = capsys.readouterr().out
    assert "ALL ROUTES COMPLETE" in out and "input_dir" in out


def test_package_code_edit_during_an_invocation_marks_it_incomplete(
    monkeypatch: pytest.MonkeyPatch, _tiny_scaling: Path,
) -> None:
    packages = iter(["c" * 64, "d" * 64])
    monkeypatch.setattr(scaling, "_current_package_code_identity", lambda: next(packages))
    monkeypatch.setattr(scaling, "run_cell", _complete_cached)
    assert scaling.main() == 1
    design = json.loads((_tiny_scaling / "scaling_design.json").read_text(encoding="utf-8"))
    assert design["status"] == "incomplete"
    assert design["error"] == "code identity drifted during the scaling invocation"


def _write_bound_cell(
    root: Path,
    *,
    route:         str,
    label:         str,
    seed:          int,
    n_params:      int,
    test_ce:       float,
    code_identity: Dict[str, object],
    package:       str,
    provenance_package: str | None = None,
) -> None:
    r"""One schema-2 cell written as ``scaling.run_cell`` now writes it: both records name one snapshot."""
    run = root / route / label / f"s{seed}"
    run.mkdir(parents=True)
    cfg = test_config(seed=seed, vocab_size=16, embed_dim=8, n_heads=2, max_seq_len=8, n_layers=1, n_e_steps=1)
    config = json.loads(json.dumps(serialize_config(cfg), default=str))
    cell = {"schema_version": 2, "route": route, "scale_knob": "embed_dim", "label": label, "seed": seed,
            "dataset": "synthetic", "data_seed": 3, "max_tokens": None, "config_sha256": canonical_json_sha256(config),
            "code_identity": code_identity, "package_code_sha256": package,
            "data_sources": {split: {"format": "pt", "size_bytes": len(split), "sha256": split[0] * 64}
                             for split in ("train", "validation", "test")}}
    digest = canonical_json_sha256(cell)
    cell["reuse_contract_sha256"] = digest
    metrics = {"n_params": n_params, "test_ce": test_ce, "test_ppl": math.exp(test_ce),
               "test_bits_per_token": test_ce / math.log(2.0), "test_bpc": None}
    (run / "summary.json").write_text(json.dumps({
        **metrics, "best_val_ppl": math.exp(test_ce), "scaling_reuse_contract_sha256": digest,
        "scaling_point": {**metrics, "embed_dim": 8, "n_heads": 2, "n_gen": 8, "gauge_group": "block_glk",
                          "n_layers": 1, "n_e_steps": 1, "tokens_seen": 1000}}), encoding="utf-8")
    (run / "config.json").write_text(json.dumps({"dataset": "synthetic", "config": config}), encoding="utf-8")
    (run / "scaling_cell.json").write_text(json.dumps(cell), encoding="utf-8")
    (run / "provenance.json").write_text(json.dumps({
        "seed": seed, **code_identity, "code_identity_sha256": provenance_package or package,
        "train_data_sha256": "train-a", "val_data_sha256": "val-a", "test_data_sha256": "test-a"}),
        encoding="utf-8")


def test_identical_runs_on_either_side_of_a_tree_edit_are_one_run_identity(tmp_path: Path) -> None:
    _write_bound_cell(tmp_path, route="width", label="K8", seed=1, n_params=100, test_ce=4.0,
                      code_identity=_CODE_A, package=_PACKAGE)
    _write_bound_cell(tmp_path, route="depth", label="L1", seed=1, n_params=100, test_ce=4.0,
                      code_identity=_CODE_B, package=_PACKAGE)
    rows = {row["route"]: row for row in sa.harvest(tmp_path)}
    assert set(rows) == {"width", "depth"}
    assert rows["width"]["effective_run_key"] == rows["depth"]["effective_run_key"]


def test_package_digest_that_differs_between_the_two_records_is_rejected(tmp_path: Path) -> None:
    _write_bound_cell(tmp_path, route="width", label="K8", seed=1, n_params=100, test_ce=4.0,
                      code_identity=_CODE_A, package=_PACKAGE, provenance_package="d" * 64)
    rejection: List[str] = []
    run = tmp_path / "width" / "K8" / "s1"
    read = [json.loads((run / name).read_text(encoding="utf-8"))
            for name in ("summary.json", "config.json", "provenance.json", "scaling_cell.json")]
    assert sa._validated_bound_scaling_run(run, *read, rejection=rejection) is None
    assert rejection == ["package code digest differs between scaling_cell and provenance"]


# =============================================================================
# A302 / A356 / A372 / A373: route menu, shared contract, parameter count, muP anchor
# =============================================================================

def test_inference_route_states_and_needs_the_gradient_receiver(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert "projected_gradient" in scaling.route_inference_t.__doc__
    monkeypatch.setattr(scaling, "CONFIG", _scaling_settings(tmp_path))
    monkeypatch.setattr(scaling, "ROUTES", {"infer_T": scaling.route_inference_t([1, 2])})
    monkeypatch.setattr(scaling, "BASELINE", _scaling_baseline(attention_mode="projected_predictive",
                                                               attention_residual=True))
    with pytest.raises(ValueError, match="n_e_steps"):
        scaling.validate_routes(["infer_T"])
    monkeypatch.setattr(scaling, "BASELINE", _scaling_baseline(attention_mode="projected_gradient"))
    scaling.validate_routes(["infer_T"])


def test_scaling_and_its_analysis_share_one_cell_contract() -> None:
    from vfe3 import scaling_contract as contract

    assert scaling._SCALING_CELL_SCHEMA_VERSION == sa._SCALING_CELL_SCHEMA_VERSION == contract.SCALING_CELL_SCHEMA_VERSION
    assert sa._SCALING_SOURCE_SPLITS is contract.SOURCE_SPLITS is scaling._SCALING_SOURCE_SPLITS
    assert scaling._validated_data_source_identities is contract.validated_data_source_identities
    assert scaling._scaling_reuse_contract_sha256 is contract.reuse_contract_sha256
    assert sa._is_sha256 is contract.is_sha256


def _variant_config(variant: str) -> VFE3Config:
    from vfe3.exact_route import principled_exact_route_overrides
    from vfe3.gauge_covariant import fully_gauge_covariant_overrides

    values = _scaling_baseline(embed_dim=8, n_layers=2, pos_phi="learned")
    if variant == "exact_route":
        values.update(principled_exact_route_overrides(dict(values)))
    elif variant == "gauge_covariant_full":
        values.update(attention_mode="projected_predictive", attention_residual=True, family="gaussian_full",
                      decode_mode="full_chunked", use_block_mlp=False, norm_type_block="none",
                      layernorm_affine=False)
        values.update(fully_gauge_covariant_overrides(dict(values)))
    else:
        predictive = {"attention_mode": "projected_predictive", "attention_residual": True,
                      "relational_receiver_measurement": "inverse"}
        values.update({
            "baseline": {},
            "noisy_key": {**predictive, "relational_routing": "noisy_key"},
            "learned_temperature": {"learn_attention_temperature": True},
            "tied_maps": {**predictive, "tie_comparison_maps": True, "beta_attention_prior": "causal_alibi_noself"},
            "lowrank_edges": {**predictive, "content_link_chart": "exponential", "content_link_edges": "lowrank"},
            "untied_kl_decode": {"decode_mode": "diagonal_chunked", "untie_decode_bank": True},
            "per_type_hidden_bias": {"per_type_hidden_bias": True, "block_mlp_expansion": 8},
        }[variant])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VFE3Config(**values)


@pytest.mark.parametrize("variant", ["baseline", "noisy_key", "learned_temperature", "tied_maps", "lowrank_edges",
                                     "untied_kl_decode", "per_type_hidden_bias", "exact_route",
                                     "gauge_covariant_full"])
def test_parameter_prediction_equals_the_built_count(variant: str) -> None:
    cfg = _variant_config(variant)
    predicted, n_gen = scaling.predict_n_params(cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(cfg)
    assert predicted == sum(p.numel() for p in model.parameters())
    assert n_gen == model.group.generator_shape[0]


def test_parameter_prediction_leaves_the_global_rng_and_matmul_policy_alone() -> None:
    torch.manual_seed(3)
    before = torch.get_rng_state()
    precision = torch.get_float32_matmul_precision()
    scaling.predict_n_params(test_config(**{**_scaling_baseline(), "fp32_matmul_precision": "medium"}))
    assert torch.equal(torch.get_rng_state(), before)
    assert torch.get_float32_matmul_precision() == precision
    assert "Exact on the pure path" not in (scaling.predict_n_params.__doc__ or "")


def test_mup_anchor_twin_is_marked_and_the_bias_deviation_is_stated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scaling, "BASELINE", _scaling_baseline(m_relational_score_lr=0.01,
                                                               m_relational_content_lr=0.0025,
                                                               m_block_mlp_lr=0.001, m_decode_mu_lr=0.02))
    cells = {cell["label"]: cell for cell in scaling.route_grow_k_mup([4, 8], n_heads=2, anchor_k=4)}
    twins = {label: cell["twin_of"] for label, cell in cells.items() if "twin_of" in cell}
    assert twins == {"K4_mup": "K4_fixed"}
    assert cells["K4_mup"]["overrides"] == cells["K4_fixed"]["overrides"]
    assert "m_block_mlp_lr" in cells["K8_mup"]["overrides"]
    doc = " ".join(scaling.route_grow_k_mup.__doc__.split())
    assert "output_proj_bias" in doc and "deviat" in doc


# =============================================================================
# A073: the muP arms are separate fit series; the PPL-vs-K fit reads the width routes
# =============================================================================

def _write_mup_sweep(root: Path, *, shift: float) -> None:
    for k in (4, 8, 12, 16, 20):
        n_params = 1000 * k * k
        for arm in ("fixed", "mup"):
            mup_rates = {} if (arm == "fixed" or k == 4) else {"m_block_mlp_lr": 0.001 * 4 / k}
            for seed in (1, 2, 3):
                ce = 3.0 + 40.0 * n_params ** -0.3 + 0.0005 * (seed - 2)
                if arm == "mup" and k > 4:
                    ce -= shift
                _write_scaling_run(root, route="grow_K_mup", label=f"K{k}_{arm}", seed=seed, n_params=n_params,
                                   test_ce=ce, embed_dim=k, **mup_rates)


@pytest.fixture
def _quiet_analysis(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sa.CONFIG, "n_bootstrap", 8)
    monkeypatch.setitem(sa.CONFIG, "with_offset", True)
    monkeypatch.setitem(sa.CONFIG, "min_points", 2)
    monkeypatch.setattr(sa, "_make_figures", lambda *args, **kwargs: None)


def test_mup_arms_are_separate_series_of_the_fits_and_the_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _quiet_analysis: None,
) -> None:
    _write_mup_sweep(tmp_path, shift=0.05)
    monkeypatch.setitem(sa.CONFIG, "input_dir", str(tmp_path))
    sa.analyze()
    summary = json.loads((tmp_path / "scaling_summary.json").read_text(encoding="utf-8"))
    series = {"grow_K_mup/fixed", "grow_K_mup/mup"}
    assert set(summary["per_route"]) == series
    assert set(summary["frontier_collapse"]["routes"]) == series
    lof = summary["pooled_fit_lack_of_fit"]
    # nine distinct (size, arm) groups: the identical K4 anchor pair is counted once
    assert lof["df_lof"] == 9 - lof["n_fit_parameters"]
    assert lof["rejected"] is True


def test_ppl_versus_width_fit_reads_only_the_width_routes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vfe3.viz import figures

    captured: Dict[str, List[Dict[str, Any]]] = {}
    for name in ("plot_scaling_law", "plot_scaling_routes", "plot_kmup_stability", "plot_inference_capacity"):
        monkeypatch.setattr(figures, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(figures, "plot_ppl_offset",
                        lambda points, **kwargs: captured.setdefault("points", list(points)) and None)

    def point(route: str, label: str, knob: str, k: int, layers: int = 1) -> Dict[str, Any]:
        return {"route": route, "label": label, "scale_knob": knob, "embed_dim": float(k), "n_layers": float(layers),
                "n_params": float(100 * k * layers), "ppl_mean": 50.0, "ppl_sem": 0.1, "n_seeds": 3,
                "ce_seeds": [3.9, 3.91, 3.92], "est_flops_6ND": float("nan"), "tokens_seen": 1000.0,
                "n_non_embedding_params": float("nan")}

    points = [point("width", "K4", "embed_dim", 4), point("width", "K8", "embed_dim", 8),
              point("depth", "L2", "n_layers", 8, layers=2), point("width_depth", "K4_L2", "embed_dim+n_layers", 4, 2),
              point("grow_K_mup", "K4_fixed", "embed_dim", 4), point("grow_K_mup", "K8_mup", "embed_dim", 8)]
    sa._make_figures(points, [], tmp_path)
    assert [(p["route"], p["label"]) for p in captured["points"]] == [
        ("width", "K4"), ("width", "K8"), ("grow_K_mup", "K4_fixed")]


# =============================================================================
# A109: the shipped layout is found from the analyzer's default directory
# =============================================================================

def _write_width_sweep(root: Path) -> None:
    for k, n_params in ((4, 100), (8, 220), (12, 470), (16, 1000), (20, 2200)):
        for seed in (1, 2, 3):
            ce = 3.0 + 9.0 * n_params ** -0.4 + 0.0005 * (seed - 2)
            _write_scaling_run(root, route="width", label=f"K{k}", seed=seed, n_params=n_params, test_ce=ce,
                               embed_dim=k)


def test_analysis_finds_the_scaling_layout_below_its_input_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _quiet_analysis: None, capsys: pytest.CaptureFixture[str],
) -> None:
    _write_width_sweep(tmp_path / "standard")
    monkeypatch.setitem(sa.CONFIG, "input_dir", str(tmp_path))
    sa.analyze()
    summary = json.loads((tmp_path / "standard" / "scaling_summary.json").read_text(encoding="utf-8"))
    assert summary["design"]["status"] == "complete" and summary["pooled_fit"] is not None
    assert not (tmp_path / "scaling_points.csv").exists()
    assert str(tmp_path / "standard") in capsys.readouterr().out


def test_unverifiable_design_names_the_manifests_it_searched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _quiet_analysis: None, capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "results"
    _write_width_sweep(root / "deeper" / "standard")            # the manifest lies two levels down
    monkeypatch.setitem(sa.CONFIG, "input_dir", str(root))
    sa.analyze()
    out = capsys.readouterr().out
    assert "no scaling_design.json" in out
    assert str(root / "scaling_design.json") in out and str(root.parent / "scaling_design.json") in out
