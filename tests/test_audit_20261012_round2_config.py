r"""Audit 2026-10-12, round 2: configuration, launcher and analysis-loader items.

Item 10: the multi-seed driver and the figure worker never replay a forward, so a run that saved a value
this build refuses is read from its raw saved values (:func:`vfe3.paper.runs.load_config`).
Item 16: C7, the paper arms read the resolved stored-covariance decay; C12, a warmup longer than the run is
refused. H1: the launchers leave ``kl_max`` at None, which resolves to 8 K in every cell.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import multiseed_analysis as ms
from tests.profiles import test_config
from vfe3.config import VFE3Config, serialize_config
from vfe3.viz.figure_worker import _load_worker_config


def _write_config(run_dir: Path, seed: int, **removed: object) -> dict:
    run_dir.mkdir(parents=True)
    raw = {**serialize_config(test_config(seed=seed)), **removed}
    (run_dir / "config.json").write_text(json.dumps({"config": raw}), encoding="utf-8")
    return raw


@pytest.mark.parametrize("removed", [{"norm_type_final": "layernorm"}, {"lambda_alpha_mode": "constant"},
                                     {"beta_attention_prior": "uniform"}, {"decode_bias": False}])
def test_analysis_loaders_fall_back_to_raw_values_for_a_removed_value(tmp_path: Path, removed: dict) -> None:
    raws = [_write_config(tmp_path / f"s{seed}", seed, **removed) for seed in (1, 2)]
    semantic = ms._semantic_config(tmp_path / "s1" / "config.json")
    assert semantic == {key: value for key, value in raws[0].items() if key != "seed"}
    fingerprints = {ms._config_fingerprint(tmp_path / f"s{seed}" / "config.json") for seed in (1, 2)}
    assert len(fingerprints) == 1                                   # the seeds still aggregate together
    cfg = _load_worker_config(tmp_path / "s1")
    assert isinstance(cfg, SimpleNamespace) and vars(cfg) == raws[0]


def test_analysis_loaders_still_build_a_loadable_config(tmp_path: Path) -> None:
    _write_config(tmp_path / "s1", 1)
    assert isinstance(_load_worker_config(tmp_path / "s1"), VFE3Config)
    assert ms._semantic_config(tmp_path / "s1" / "config.json")["family"] == "gaussian_diagonal"


@pytest.mark.parametrize("weight_decay, label", [(0.0, "no_decay"), (0.05, "decayed")])
def test_paper_arms_read_the_resolved_stored_covariance_decay(weight_decay: float, label: str) -> None:
    import paper_arms as pa
    cfg = test_config(sigma_weight_decay=None, weight_decay=weight_decay)    # None inherits weight_decay (C7)
    expected = pa.EVALUATED_NO_DECAY if label == "no_decay" else pa.EVALUATED
    assert pa.evaluated_label(cfg) == expected
    assert (pa._no_decay_already(cfg) is None) == (label == "decayed")
    assert (pa._decay_absent(cfg) is None) == (label == "decayed")


def test_launchers_leave_kl_max_to_its_eight_k_resolution() -> None:
    import ablation
    import parameter_optimization
    import scaling
    import train_vfe3
    default = VFE3Config.__dataclass_fields__["kl_max"].default
    assert default is None and test_config(kl_max=None, embed_dim=4).kl_max == 8.0 * 4   # None resolves to 8 K
    for launcher in (train_vfe3.config, ablation.BASELINE_CONFIG, scaling.config, parameter_optimization.config):
        # A launcher that omits the key runs at the default (audit 2026-10-16 F113: the omitted key raised KeyError).
        assert launcher.get("kl_max", default) is None
    for cell in scaling.build_standard_routes({"widths": [20, 40], "depths": [1], "train_steps": [10]})["width"]:
        assert "kl_max" not in cell["overrides"]


def test_warmup_longer_than_the_run_is_refused() -> None:
    with pytest.raises(ValueError, match="warmup_steps"):
        test_config(max_steps=10, warmup_steps=11)
    assert test_config(max_steps=10, warmup_steps=10).warmup_steps == 10
    assert test_config(max_steps=10, warmup_steps=0).warmup_steps == 0
