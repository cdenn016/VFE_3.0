"""Behavioral click-to-run coverage for the optional block MLP experiment."""

import ablation
import train_vfe3

from vfe3.config import VFE3Config


_EXPECTED_DEFAULTS = {
    "use_block_mlp": False,
    "block_mlp_expansion": 4,
    "block_mlp_activation": "gelu",
    "block_mlp_dropout": 0.0,
    "m_block_mlp_lr": None,
}


def _label_config(**overrides: object) -> VFE3Config:
    values = {
        "vocab_size": 17,
        "embed_dim": 4,
        "n_heads": 2,
        "max_seq_len": 8,
        "n_layers": 1,
        "gauge_group": "block_glk",
    }
    values.update(overrides)
    return VFE3Config(**values)


def test_config_defaults_and_editable_launchers_construct():
    """Library defaults remain stable while click-to-run launcher choices stay editable."""
    defaults = _label_config()
    assert {key: getattr(defaults, key) for key in _EXPECTED_DEFAULTS} == _EXPECTED_DEFAULTS
    VFE3Config(**ablation.BASELINE_CONFIG)
    VFE3Config(**train_vfe3.config)


def test_block_mlp_ablation_is_opt_in_and_arms_differ_only_by_enable_toggle():
    """The registered comparison is runnable but remains outside the default sweep schedule."""
    runs = dict(ablation.make_run_overrides("block_mlp"))
    ablation.validate_sweeps(["block_mlp"])
    assert "block_mlp" not in ablation.SWEEP_ORDER
    assert runs == {
        "block_mlp_off": {"use_block_mlp": False},
        "block_mlp_on": {"use_block_mlp": True},
    }


def test_block_mlp_hyperparameter_sweeps_are_complete_and_constructible():
    """Every retained MLP registry exposes runnable arms without pinning the active schedule."""
    sweep_names = [
        "block_mlp",
        "block_mlp_mode",
        "block_mlp_covariance",
        "block_mlp_expansion",
        "block_mlp_activation",
        "block_mlp_dropout",
        "m_block_mlp_lr",
    ]

    ablation.validate_sweeps(sweep_names, require_construction=False)

    for sweep_name in sweep_names:
        runs = ablation.make_run_overrides(sweep_name)
        assert runs
        if sweep_name != "block_mlp":
            assert all(overrides["use_block_mlp"] is True for _, overrides in runs)
        for _, overrides in runs:
            VFE3Config(**ablation._cell_cfg_dict({
                "lambda_alpha_mode": "constant",
                "pos_phi": "none",
                **overrides,
            }, seed=0))


def test_active_block_mlp_run_label_is_tagged_without_changing_off_label():
    """Only enabled topology receives the unambiguous MLP run-directory suffix."""
    assert "_mlp" not in train_vfe3._run_label(_label_config(), "synthetic")
    assert "_mlp" in train_vfe3._run_label(_label_config(use_block_mlp=True), "synthetic")
