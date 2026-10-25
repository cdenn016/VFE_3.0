r"""Regression tests for the October 25, 2026 audit findings on evaluation artifacts (F03, F09, F10, F16, F17).

F03: checkpoint parameter accounting counts a block MLP shared across layers once, as ``named_parameters()`` does.
F17: persistent nonparameter buffers (the additive-control readout, the CG intertwiners, the irrep-basis digest) are
not counted as parameters. F10: the prototype feature probe wraps a tied bank once, attributes each call to its
layer, and restores the method once. F09: a multiseed checkpoint override records the checkpoint of the saved
semantic metrics apart from the checkpoint the stability reads. F16: the scaling preview counts the per-type
hidden-bias table. CPU only, one thread, tiny models.
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest
import torch

from tests.profiles import test_config
from vfe3.model.model import VFEModel
from vfe3.posthoc.maps import load_map_state
from vfe3.run_resources import frozen_parameter_names, parameter_breakdown

torch.set_num_threads(1)

_COORDINATE = dict(attention_mode="projected_predictive", block_mlp_mode="coordinate")
_PROTOTYPE = dict(attention_mode="projected_predictive", block_mlp_mode="prototype_agents")
_MEMORY = dict(attention_mode="projected_predictive", predictive_readout="minimal_agent",
               minimal_agent_inference="exact", family="gaussian_full", decode_mode="full_chunked",
               norm_type_block="none", block_coordinates="reference", block_mlp_mode="cue_result_memory",
               block_mlp_inference="exact", block_mlp_covariance="posterior", block_memory_regression_rank=2)
_ROUTES: Dict[str, dict] = {"coordinate": _COORDINATE, "prototype_agents": _PROTOTYPE,
                            "cue_result_memory": _MEMORY}
_SO3_CG = dict(gauge_group="so_n", group_n=3, irrep_spec=[("l1", 1), ("l2", 1)], embed_dim=8,
               use_cg_coupling=True, use_block_mlp=False, norm_type_block="none")


def _cfg(**values: Any):
    base = dict(vocab_size=7, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=3, block_mlp_expansion=2,
                block_mlp_dropout=0.0)
    base.update(values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return test_config(**base)


def _model(cfg, seed: int = 0) -> VFEModel:
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VFEModel(cfg)


def _state_breakdown(state, cfg) -> Dict[str, Any]:
    return parameter_breakdown(state, cfg, frozen=frozen_parameter_names(cfg, state.keys()))


# ----------------------------------------------------------------------------------------------
# F03: a tied block MLP is counted once from a state dict
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("route", sorted(_ROUTES))
def test_tied_bank_state_dict_counts_match_named_parameters(route: str, tmp_path: Path) -> None:
    cfg = _cfg(use_block_mlp=True, block_mlp_tie_layers=True, **_ROUTES[route])
    model = _model(cfg)
    unique = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bank = sum(p.numel() for p in model.block_mlps[0].parameters())
    named = parameter_breakdown(dict(model.named_parameters()), cfg)
    in_memory = _state_breakdown(model.state_dict(), cfg)
    torch.save({"model_state": model.state_dict()}, tmp_path / "final_model.pt")
    saved_state, _ = load_map_state(tmp_path / "final_model.pt")          # mmap: aliases are separate copies
    saved = _state_breakdown(saved_state, cfg)
    for breakdown in (named, in_memory, saved):
        assert breakdown["total"] == unique
        assert breakdown["context_processing"]["block_mlp"] == bank
    assert len(saved["tied_aliases"]) == (cfg.n_layers - 1) * len(dict(model.block_mlps[0].named_parameters()))
    assert all(name.startswith(("block_mlps.1.", "block_mlps.2.")) for name in saved["tied_aliases"])
    # Every checkpoint key is preserved: the aliases stay in the state dict, only the count changes.
    assert any(key.startswith("block_mlps.2.") for key in model.state_dict())


def test_untied_state_dict_counts_every_layer_and_tied_mismatch_is_refused() -> None:
    cfg = _cfg(use_block_mlp=True, **_COORDINATE)
    model = _model(cfg)
    unique = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert _state_breakdown(model.state_dict(), cfg)["total"] == unique
    tied_cfg = dataclasses.replace(cfg, block_mlp_tie_layers=True)
    with pytest.raises(ValueError, match="not saved from a tied model"):
        _state_breakdown(model.state_dict(), tied_cfg)                  # layers differ: not a tied checkpoint


def test_paper_checkpoint_fallback_counts_a_tied_bank_once(tmp_path: Path) -> None:
    from vfe3.config import serialize_config
    from vfe3.paper.runs import parameter_counts

    cfg = _cfg(use_block_mlp=True, block_mlp_tie_layers=True, **_COORDINATE)
    model = _model(cfg)
    torch.save({"model_state": model.state_dict()}, tmp_path / "final_model.pt")
    raw = serialize_config(cfg)
    run = SimpleNamespace(_cache={}, resources=None, posthoc=None, run_dir=tmp_path, notes=[], summary={},
                          value=lambda key, default=None: raw.get(key, default))
    counts = parameter_counts(run)
    assert run.notes == []
    assert counts["total"] == sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------------------------
# F17: persistent buffers are not parameters
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("case", ("per_token_additive", "so_n_cg_coupling"))
def test_persistent_buffers_are_not_counted_as_parameters(case: str) -> None:
    values = (dict(encode_mode="per_token_additive", pos_phi="none", use_block_mlp=False, n_layers=2)
              if case == "per_token_additive"
              else dict(_SO3_CG, n_layers=2))
    cfg = _cfg(**values)
    model = _model(cfg)
    buffers = {name for name, _ in model.named_buffers()}
    expected = {"per_token_additive": {"prior_bank.additive_R"},
                "so_n_cg_coupling": {"gauge_basis_fingerprint"}}[case]
    assert expected <= buffers
    if case == "so_n_cg_coupling":
        assert any(name.startswith("cg_coupling.cg_") for name in buffers)
    named = parameter_breakdown(dict(model.named_parameters()), cfg)
    state = _state_breakdown(model.state_dict(), cfg)
    assert state["total"] == named["total"]
    assert state["total_including_frozen"] == named["total_including_frozen"] == sum(
        p.numel() for p in model.parameters())
    assert buffers & set(model.state_dict()) <= set(state["excluded_buffers"])


# ----------------------------------------------------------------------------------------------
# F10: the prototype probe on a tied bank
# ----------------------------------------------------------------------------------------------

def test_prototype_probe_wraps_a_tied_bank_once_and_attributes_layers() -> None:
    from vfe3.posthoc.feature_probes import prototype_probe

    cfg = _cfg(use_block_mlp=True, block_mlp_tie_layers=True, n_layers=2, max_seq_len=3, **_PROTOTYPE)
    tied = _model(cfg).eval()
    untied = _model(dataclasses.replace(cfg, block_mlp_tie_layers=False)).eval()
    untied.load_state_dict(tied.state_dict())                           # every layer holds the shared bank
    batches = [torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.tensor([[0, 3, 1], [2, 2, 5]])]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out_tied = prototype_probe(tied, batches, torch.device("cpu"), gap_tokens=4)
        out_untied = prototype_probe(untied, batches, torch.device("cpu"), gap_tokens=4)
    assert [entry["layer"] for entry in out_tied["layers"]] == [0, 1]
    for a, b in zip(out_tied["layers"], out_untied["layers"]):
        assert a["tokens"] == b["tokens"] == 12                          # one call per layer per batch
        assert a["gap_tokens"] == b["gap_tokens"] == 4
        assert a["usage"] == pytest.approx(b["usage"], abs=1e-6)
        assert a["codebook"] == b["codebook"]
        assert a["label_tv"] == pytest.approx(b["label_tv"], abs=1e-6)
    assert out_tied["layers"][0]["usage"] != pytest.approx(out_tied["layers"][1]["usage"], abs=1e-9)
    assert "forward_moments" not in vars(tied.block_mlps[0])            # the class method is restored
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tied.forward_beliefs(batches[0])


# ----------------------------------------------------------------------------------------------
# F09: multiseed checkpoint override provenance
# ----------------------------------------------------------------------------------------------

def _analyzed_run(root: Path, name: str, seed: int) -> Path:
    run_dir = root / name
    (run_dir / "semantics").mkdir(parents=True)
    metrics = {"schema": "vfe3.semantics.report/1", "run": {"checkpoint": str(run_dir / "best_model.pt")},
               "options": {}, "sections": {}}
    (run_dir / "semantics" / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return run_dir


@pytest.mark.parametrize("override", (None, "final_model.pt"))
def test_multiseed_records_metric_and_stability_checkpoints(override, tmp_path: Path,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    from vfe3.semantics import multiseed as M
    from vfe3.semantics import run as semantic_run

    def fake_load(run_dir, *, checkpoint, vocab=None, drop_unknown_config_fields=False):
        seed = int(Path(run_dir).name.split("_")[-1])
        return SimpleNamespace(cfg=_cfg(seed=seed), checkpoint_path=Path(run_dir) / checkpoint, notes=[])

    monkeypatch.setattr(semantic_run, "load_semantic_run", fake_load)
    monkeypatch.setattr(M, "apply_style", lambda: None)
    runs = [_analyzed_run(tmp_path, f"seed_{seed}", seed) for seed in (0, 1)]
    out = tmp_path / "multiseed"
    md_path = M.aggregate_seeds(runs, out_dir=out, checkpoint=override, stability=False, log=lambda msg: None)
    payload = json.loads((out / M.JSON_NAME).read_text(encoding="utf-8"))
    stability = override or "best_model.pt"
    for entry in payload["runs"]:
        assert entry["analysis_checkpoint"] == "best_model.pt"
        assert entry["stability_checkpoint"] == stability
    mismatch = [n for n in payload["notes"] if "different weights" in n]
    assert len(mismatch) == (2 if override else 0)
    text = md_path.read_text(encoding="utf-8")
    assert "metrics checkpoint" in text and "stability checkpoint" in text


# ----------------------------------------------------------------------------------------------
# F16: the scaling preview counts the per-type hidden-bias table
# ----------------------------------------------------------------------------------------------

def test_scaling_preview_counts_the_per_type_hidden_bias() -> None:
    import scaling

    cfg = _cfg(n_layers=2, use_block_mlp=True, block_mlp_mode="coordinate", per_type_hidden_bias=True)
    counts = {}
    for enabled in (False, True):
        cc = dataclasses.replace(cfg, per_type_hidden_bias=enabled)
        predicted, n_gen = scaling.predict_n_params(cc)
        actual = sum(p.numel() for p in _model(cc).parameters())
        counts[enabled] = (predicted, actual, n_gen)
    (pred_off, actual_off, n_gen), (pred_on, actual_on, _) = counts[False], counts[True]
    assert actual_on - actual_off == cfg.vocab_size * n_gen
    assert pred_on - pred_off == cfg.vocab_size * n_gen
    assert pred_on - actual_on == pred_off - actual_off
