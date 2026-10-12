"""Tiny controls for standard scaling grids; never launch a real experiment."""
from tests.profiles import config_values, test_config
import copy

import pytest
import torch

import scaling
from vfe3.model.model import VFEModel


def _baseline() -> dict:
    return config_values(vocab_size=7, embed_dim=4, n_heads=2, n_layers=1, n_e_steps=1,
                max_seq_len=3, batch_size=2, max_steps=12, warmup_steps=1,
                eval_stride=None, gauge_group="block_glk", pos_phi="none",

                attention_mode="projected_predictive",
                use_block_mlp=True, block_mlp_expansion=2, kl_max=32.)


@pytest.fixture(autouse=True)
def _local_scaling_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scaling, "CONFIG", dict(
        routes=["width"], dry_run=True, seeds=[1, 2], data_seed=3,
        device="cpu", dataset="fixture", max_tokens=None, max_steps=None,
        resume=False, output_dir="unused-test-scaling-output"))


def _settings() -> dict:
    return {"widths": [4, 8], "depths": [1, 2], "train_steps": [4, 8]}


def test_grids_preserve_baseline_and_change_only_declared_axes(monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = _baseline()
    original = copy.deepcopy(baseline)
    monkeypatch.setattr(scaling, "BASELINE", baseline)
    routes = scaling.build_standard_routes(_settings())
    assert baseline == original
    assert len(routes["width_depth"]) == 4
    for cell in routes["width"]:
        cfg = scaling._cell_cfg_dict(cell["overrides"], 3, None)
        assert cfg["attention_mode"] == "projected_predictive"
        assert cfg["n_heads"] == 2 and cfg["n_layers"] == 1
        assert cfg["kl_max"] / cfg["embed_dim"] == 8
    for cell in routes["depth"]:
        assert cell["route"] == "depth"
        assert set(cell["overrides"]) == {"n_layers"}
    assert [c["overrides"]["max_steps"] for c in routes["train_steps"]] == [4, 8]


def test_selected_validation_ignores_incompatible_legacy_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scaling, "BASELINE", _baseline())
    routes = scaling.build_standard_routes(_settings())
    routes["legacy"] = [{"label": "bad", "route": "legacy", "scale_knob": "embed_dim",
                         "overrides": {"embed_dim": 3, "n_heads": 2}}]
    monkeypatch.setattr(scaling, "ROUTES", routes)
    monkeypatch.setitem(scaling.CONFIG, "max_steps", None)
    scaling.validate_routes(["width", "depth", "width_depth", "train_steps"])
    with pytest.raises(ValueError, match="legacy"):
        scaling.validate_routes(["legacy"])


def test_budget_override_cannot_collapse_step_sweep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scaling, "BASELINE", _baseline())
    monkeypatch.setattr(scaling, "ROUTES", scaling.build_standard_routes(_settings()))
    monkeypatch.setitem(scaling.CONFIG, "max_steps", 12)
    with pytest.raises(ValueError, match="max_steps"):
        scaling.validate_routes(["train_steps"])


def test_preview_never_reads_corpus_or_builds_model(tmp_path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(scaling, "BASELINE", _baseline())
    monkeypatch.setattr(scaling, "ROUTES", scaling.build_standard_routes(_settings()))
    monkeypatch.setattr(scaling, "CONFIG", dict(scaling.CONFIG, routes=["width"], seeds=[1, 2],
                                               dry_run=True, max_steps=None, output_dir=str(tmp_path / "runs")))
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("preview reached execution")
    monkeypatch.setattr(scaling, "VFEModel", forbidden)
    monkeypatch.setattr(scaling, "_data_source_identities", forbidden)
    assert scaling.main() == 0
    assert not (tmp_path / "runs").exists()
    assert "tokens/run" in capsys.readouterr().out


def test_parameter_estimate_includes_per_layer_relational_maps(monkeypatch: pytest.MonkeyPatch) -> None:
    base = _baseline()
    # Pin norms off to isolate the per-layer map and MLP parameter increment.
    base.update(norm_type_block="none", layernorm_affine=False)
    shallow = test_config(**base)
    deep = test_config(**dict(base, n_layers=2))
    actual = [sum(p.numel() for p in VFEModel(c).parameters()) for c in (shallow, deep)]
    predicted = [scaling.predict_n_params(c)[0] for c in (shallow, deep)]
    assert predicted[1] - predicted[0] == actual[1] - actual[0]
    assert predicted[1] > predicted[0]


@pytest.mark.parametrize("setting, expected", [(3, 3), (None, None), ("model", 7)])
def test_data_seed_policy(setting: object, expected: object, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(scaling.CONFIG, "data_seed", setting)
    assert scaling._effective_data_seed(7) == expected


def test_plan_records_resolved_config_and_joint_axes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scaling, "BASELINE", _baseline())
    monkeypatch.setattr(scaling, "ROUTES", scaling.build_standard_routes(_settings()))
    monkeypatch.setitem(scaling.CONFIG, "max_steps", None)
    monkeypatch.setitem(scaling.CONFIG, "data_seed", 3)
    design = scaling._scaling_design(["width_depth"], [7])
    assert len(design["cells"]) == 4
    cell = design["cells"][-1]
    assert cell["effective_config"]["embed_dim"] == 8
    assert cell["effective_config"]["n_layers"] == 2
    assert cell["effective_config"]["seed"] == 7 and cell["data_seed"] == 3
    cfg = test_config(**cell["effective_config"])
    assert scaling._scaling_knob_report(cfg, "embed_dim+n_layers")["effective_value"] == {
        "embed_dim": 8, "n_layers": 2}


def test_runner_resets_fixed_training_order_across_model_seeds(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scaling, "BASELINE", _baseline())
    monkeypatch.setattr(scaling, "CONFIG", dict(scaling.CONFIG, data_seed=3, resume=False))
    monkeypatch.setattr(scaling, "_data_source_identities", lambda _: {})
    monkeypatch.setattr(scaling, "_current_code_identity", lambda: {})
    monkeypatch.setattr(scaling, "_validated_scaling_code_identity", lambda value: value)
    loaders = []
    observed = []

    class ReachedArtifacts(Exception):
        pass

    def get_loader(dataset: str, length: int, batch: int, split: str, **kwargs: object) -> object:
        generator = None
        if split == "train":
            assert kwargs["data_seed"] == 3
            generator = torch.Generator().manual_seed(3)
        loader = torch.utils.data.DataLoader(torch.arange(8), batch_size=2,
                                            shuffle=split == "train", generator=generator)
        if split == "train":
            loaders.append(loader)
        return loader

    def artifacts(*args: object, **kwargs: object) -> None:
        observed.append(list(iter(loaders[-1]))[0])
        raise ReachedArtifacts

    monkeypatch.setattr(scaling, "get_loader", get_loader)
    monkeypatch.setattr(scaling, "RunArtifacts", artifacts)
    cell = {"label": "tiny", "route": "width", "scale_knob": "embed_dim", "overrides": {}}
    for seed in (1, 2):
        with pytest.raises(ReachedArtifacts):
            scaling.run_cell(cell, tmp_path / str(seed), seed, dataset="fixture", device=torch.device("cpu"))
    torch.testing.assert_close(observed[0], observed[1])
