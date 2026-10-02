from tests.profiles import config_values, test_config, tiny_config
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import ablation
import train_vfe3
from vfe3.contracts import CanonicalFrameContext
from vfe3.model.block import vfe_block
from vfe3.model.block_mlp import CanonicalFrameBlockMLP, GaugeGateBlockMLP
from vfe3.model.model import VFEModel


def _full_cfg(**overrides):
    values = dict(
        vocab_size=17,
        embed_dim=4,
        n_heads=2,
        max_seq_len=4,
        n_layers=1,
        n_e_steps=1,
        e_q_mu_lr=0.1,
        e_q_sigma_lr=0.02,
        e_phi_lr=0.0,
        family="gaussian_full",
        decode_mode="full_chunked",
        gauge_group="block_glk",
        transport_mode="flat",
        pos_phi="none",
        pos_rotation="none",
        norm_type_block="none",
    )
    values.update(overrides)
    return test_config(**values)


def test_block_mlp_coordinate_passthrough_controls():
    cfg = test_config(block_mlp_mode="coordinate", block_mlp_covariance="passthrough",
                      block_mlp_covariance_floor=1e-4)
    assert cfg.block_mlp_mode == "coordinate"
    assert cfg.block_mlp_covariance == "passthrough"
    assert cfg.block_mlp_covariance_floor == 1e-4


@pytest.mark.parametrize("mode", ["coordinate", "gauge_gate", "canonical_frame"])
@pytest.mark.parametrize("covariance", ["passthrough", "delta_full"])
def test_block_mlp_mode_and_covariance_controls_validate(mode, covariance):
    _full_cfg(
        use_block_mlp=True,
        block_mlp_mode=mode,
        block_mlp_covariance=covariance,
    )


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"block_mlp_mode": "unknown"}, "block_mlp_mode"),
        ({"block_mlp_covariance": "unknown"}, "block_mlp_covariance"),
        ({"block_mlp_covariance_floor": 0.0}, "block_mlp_covariance_floor"),
        ({
            # A passthrough gauge gate is admitted on gaussian_diagonal; the dense J Sigma J^T of
            # delta_full still needs the full family.
            "use_block_mlp": True,
            "block_mlp_mode": "gauge_gate",
            "block_mlp_covariance": "delta_full",
            "family": "gaussian_diagonal",
            "decode_mode": "diagonal_chunked",
        }, "full-covariance"),
        ({
            "use_block_mlp": True,
            "block_mlp_mode": "canonical_frame",
            "e_phi_lr": 0.01,
        }, "e_phi_lr"),
        ({
            "use_block_mlp": True,
            "block_mlp_mode": "canonical_frame",
            "pos_rotation": "rope",
            "rope_insertion": "left",
            "rope_full_gauge": True,
            "_allow_legacy_left_rope": True,
        }, "rope_insertion"),
    ],
)
def test_block_mlp_modes_fail_closed_on_incompatible_config(overrides, message):
    with pytest.raises((TypeError, ValueError), match=message):
        _full_cfg(**overrides)


@pytest.mark.parametrize("attention_mode", ["projected_predictive", "projected_gradient"])
@pytest.mark.parametrize("rope_full_gauge, rope_on_value", [(False, False), (False, True), (True, False)])
def test_canonical_rope_split_config_builds_with_visible_caveat(
    attention_mode: str, rope_full_gauge: bool, rope_on_value: bool,
    capsys: pytest.CaptureFixture, recwarn: pytest.WarningsRecorder,
) -> None:
    cfg = tiny_config(
        attention_mode=attention_mode,
        attention_residual=False,


        predictive_covariance="passthrough",
        family="gaussian_full" if rope_full_gauge else "gaussian_diagonal",
        decode_mode="full_chunked" if rope_full_gauge else "diagonal_chunked",
        use_block_mlp=True,
        block_mlp_mode="canonical_frame",
        pos_rotation="rope",
        rope_full_gauge=rope_full_gauge,
        rope_on_value=rope_on_value,
    )
    VFEModel(cfg)
    output = capsys.readouterr().out
    assert not any("not the covariance sandwich" in str(warning.message) for warning in recwarn)
    if not rope_on_value:
        assert output.count("WARNING: canonical_frame + RoPE") == 1
        lines = output.strip().splitlines()
        assert lines[0] == lines[-1] == "-" * 72
        assert "position-dependent MLP" in output
    else:
        assert "WARNING: canonical_frame + RoPE" not in output


def test_canonical_diagonal_predictive_rope_forward_backward() -> None:
    cfg = tiny_config(
        use_block_mlp=True,
        block_mlp_mode="canonical_frame",
        pos_rotation="rope",
        rope_full_gauge=False,
        rope_on_value=False,
        attention_residual=False,
        decode_unigram_prior=False,
    )
    torch.manual_seed(17)
    model = VFEModel(cfg)
    beliefs, logits = model.forward_beliefs(torch.tensor([[0, 1, 2]]), return_logits=True)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 7), torch.tensor([1, 2, 3]))
    loss.backward()
    assert torch.isfinite(beliefs.mu).all()
    assert torch.isfinite(beliefs.sigma).all()
    assert torch.isfinite(logits).all()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert model.block_mlps[0].fc1.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("pos_rotation, rope_full_gauge", [("none", False), ("rope", True)])
def test_canonical_rope_build_caveat_is_absent_without_split(
    pos_rotation: str, rope_full_gauge: bool, capsys: pytest.CaptureFixture,
) -> None:
    VFEModel(_full_cfg(
        vocab_size=7, block_mlp_expansion=2,
        use_block_mlp=True, block_mlp_mode="canonical_frame", pos_rotation=pos_rotation,
        rope_full_gauge=rope_full_gauge, rope_on_value=True,
    ))
    assert "WARNING: canonical_frame + RoPE" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "mode, expected_type",
    [
        ("gauge_gate", GaugeGateBlockMLP),
        ("canonical_frame", CanonicalFrameBlockMLP),
    ],
)
def test_model_builds_selected_untied_block_mlp_type(mode, expected_type):
    model = VFEModel(_full_cfg(
        use_block_mlp=True,
        block_mlp_mode=mode,
        n_layers=2,
    ))
    assert len(model.block_mlps) == 2
    assert all(isinstance(module, expected_type) for module in model.block_mlps)


class _MomentTransform(nn.Module):
    def forward_moments(self, mu, sigma, *, frame_context=None):
        assert frame_context is not None
        return SimpleNamespace(mu=mu + 3.0, sigma=sigma + 5.0)


def test_block_uses_moment_api_and_forwards_canonical_frame():
    cfg = _full_cfg()
    model = VFEModel(cfg)
    belief = model.prior_bank.encode(torch.tensor([[1, 2]]))
    eye = torch.eye(4).expand(1, 2, 4, 4)
    frame = CanonicalFrameContext(eye, eye)
    capture = {}
    out = vfe_block(
        belief,
        belief.mu,
        belief.sigma,
        model.group,
        cfg,
        block_mlp=_MomentTransform(),
        block_mlp_frame=frame,
        capture=capture,
        relational_operators=model.relational_operators_by_layer[0],
        relational_frame=frame,
    )
    assert torch.equal(out.mu, capture["converged"].mu + 3.0)
    assert torch.equal(out.sigma, capture["converged"].sigma + 5.0)


def test_launcher_accepts_mode_and_covariance_controls(monkeypatch):
    selectors = {"block_mlp_mode", "block_mlp_covariance", "block_mlp_covariance_floor"}
    local = config_values(block_mlp_mode="coordinate", block_mlp_covariance="passthrough",
                          block_mlp_covariance_floor=1e-4)
    monkeypatch.setattr(train_vfe3, "config", local)
    assert selectors <= set(train_vfe3.config)
    cfg = test_config(**train_vfe3.config)
    assert cfg.block_mlp_mode == "coordinate"
    assert cfg.block_mlp_covariance == "passthrough"
    assert cfg.block_mlp_covariance_floor == 1e-4


def test_ablation_registers_mode_and_covariance_sweeps(monkeypatch):
    monkeypatch.setattr(ablation, "BASELINE_CONFIG", config_values())
    ablation.validate_sweeps(
        ["block_mlp_mode", "block_mlp_covariance"], require_construction=False,
    )
    modes = dict(ablation.make_run_overrides("block_mlp_mode"))
    covariances = dict(ablation.make_run_overrides("block_mlp_covariance"))
    assert modes == {
        "block_mlp_mode=coordinate": {
            "use_block_mlp": True,
            "block_mlp_mode": "coordinate",
        },
        "block_mlp_mode=gauge_gate": {
            "use_block_mlp": True,
            "block_mlp_mode": "gauge_gate",
            "family": "gaussian_full",
            "decode_mode": "full_chunked",
        },
        "block_mlp_mode=canonical_frame": {
            "use_block_mlp": True,
            "block_mlp_mode": "canonical_frame",
        },
    }
    assert covariances == {
        "block_mlp_covariance=passthrough": {
            "use_block_mlp": True,
            "block_mlp_covariance": "passthrough",
        },
        "block_mlp_covariance=delta_full": {
            "use_block_mlp": True,
            "block_mlp_covariance": "delta_full",
            "family": "gaussian_full",
            "decode_mode": "full_chunked",
        },
    }
    for sweep in ("block_mlp_mode", "block_mlp_covariance"):
        for _, overrides in ablation.make_run_overrides(sweep):
            _full_cfg(**overrides)
