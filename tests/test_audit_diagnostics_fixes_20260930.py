"""Tiny CPU regressions for the September 30 diagnostics remediation."""

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import vfe3.run_artifacts as ra
from vfe3.baseline import BaselineConfig, BaselineTransformer
from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.run_resources import throughput_summary


def _cfg(**overrides: object) -> VFE3Config:
    values: dict[str, object] = dict(
        vocab_size=7, embed_dim=4, n_heads=2, n_layers=1, max_seq_len=4,
        batch_size=2, n_e_steps=1, warmup_steps=0, max_steps=2,
        log_interval=0, eval_interval=0, checkpoint_interval=0,
        family="gaussian_full", gauge_group="block_glk",
        attention_mode="projected_predictive", predictive_covariance="moment_matched",
        e_step_gradient="unroll", oracle_unroll_grad=True, e_phi_lr=0.0,
        decode_mode="full_chunked", decode_coordinates="reference",
        pos_phi="none", pos_rotation="none", use_block_mlp=False,
        norm_type_block="none", norm_type_final="none", layernorm_affine=False,
        sigma_max=None, phi_mstep_max_matrix_norm=None, transport_chart_max_norm=12.0,
        generate_embedding_figures=False,
    )
    values.update(overrides)
    return VFE3Config(**values)


@pytest.mark.parametrize("decode_coordinates", ["local", "reference"])
def test_gauge_badge_includes_decoder_coordinate_policy(decode_coordinates: str) -> None:
    report = ra._pure_path_report(_cfg(decode_coordinates=decode_coordinates), [])
    compatible = decode_coordinates == "reference"
    assert report["gauge_flags"]["decoder_gauge_compatible"] is compatible
    assert report["on_gauge_pure_path"] is compatible
    assert report["config_toggles"]["decode_coordinates"] == decode_coordinates


@pytest.mark.parametrize(
    "attention_mode, predictive_covariance, e_step_update, skip_sigma, consumer",
    [
        ("projected_predictive", "moment_matched", "gradient", False, False),
        ("projected_predictive", "passthrough", "gradient", False, False),
        ("projected_conditional", "passthrough", "gradient", False, False),
        ("transported", "passthrough", "gradient", False, True),
        ("transported", "passthrough", "gradient", True, False),
        ("transported", "passthrough", "mm_exact", False, True),
        ("transported", "passthrough", "mm_exact", True, False),
    ],
)
@pytest.mark.parametrize("sigma_max", [None, 100.0])
def test_spectral_cap_facet_tracks_the_live_covariance_consumer(
    attention_mode: str,
    predictive_covariance: str,
    e_step_update: str,
    skip_sigma: bool,
    consumer: bool,
    sigma_max: float | None,
) -> None:
    cfg = _cfg(
        attention_mode=attention_mode, predictive_covariance=predictive_covariance,
        e_step_update=e_step_update, skip_belief_sigma_update=skip_sigma,
        sigma_max=sigma_max,
        decode_coordinates="local" if attention_mode == "transported" else "reference",
    )
    report = ra._pure_path_report(cfg, [])
    expected = sigma_max is None if consumer else "not_applicable"
    assert report["gauge_flags"]["no_fixed_coordinate_spectral_cap"] == expected
    if attention_mode == "projected_predictive":
        assert report["on_gauge_pure_path"] is True


@pytest.mark.parametrize("site", ["provenance", "vfe_scaling", "baseline_scaling"])
@pytest.mark.parametrize("device_type, device_index", [("cpu", None), ("cuda", 3)])
def test_hardware_metadata_follows_execution_device(
    site: str,
    device_type: str,
    device_index: int | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg()
    model = VFEModel(cfg)
    artifacts = None
    if site == "provenance":
        artifacts = ra.RunArtifacts(tmp_path / "run", cfg, model, save_model_artifacts=False)
    elif site == "baseline_scaling":
        baseline_cfg = BaselineConfig(
            vocab_size=7, embed_dim=4, n_heads=2, n_layers=1, mlp_expansion=1,
            max_seq_len=4, batch_size=2, max_steps=2, warmup_steps=0,
            log_interval=0, eval_interval=0, checkpoint_interval=0,
        )
        model = BaselineTransformer(baseline_cfg)

    if device_type == "cuda":
        # Simulate parameter metadata only: no CUDA tensor or workload is constructed.
        parameter = SimpleNamespace(
            device=torch.device("cuda:3"), requires_grad=True, numel=lambda: 1,
        )
        monkeypatch.setattr(model, "parameters", lambda: iter((parameter,)))

    queried: list[torch.device] = []

    def installed_inventory_query() -> int:
        raise AssertionError("execution metadata must not query installed CUDA inventory")

    def device_name(device: object) -> str:
        queried.append(torch.device(device))
        assert device_type == "cuda", "CPU runs must not query CUDA device names"
        return "execution_accelerator_3"

    monkeypatch.setattr(torch.cuda, "device_count", installed_inventory_query)
    monkeypatch.setattr(torch.cuda, "get_device_name", device_name)
    if site == "provenance":
        assert artifacts is not None
        ra._write_provenance(artifacts, cfg, model, logging.getLogger(__name__))
        record = json.loads((artifacts.run_dir / "provenance.json").read_text(encoding="utf-8"))
        assert throughput_summary(artifacts.run_dir)["device_name"] == record["device_name"]
    elif site == "vfe_scaling":
        record = ra._cost_model_fields(model, cfg, 1, 0)
    else:
        record = model.cost_model_fields(1, 0)

    assert record["device_type"] == device_type
    assert record["device_index"] == device_index
    assert record["device"] == ("cpu" if device_type == "cpu" else "cuda:3")
    assert record["device_name"] == ("cpu" if device_type == "cpu" else "execution_accelerator_3")
    assert queried == ([] if device_type == "cpu" else [torch.device("cuda:3")])
