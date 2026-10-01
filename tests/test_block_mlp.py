from tests.profiles import test_config
import re

import pytest
import torch
from torch import nn

from vfe3.config import ConfigNotice
from vfe3.geometry.transport import build_factored_transport
from vfe3.model.block import vfe_block
from vfe3.model.block_mlp import BlockMLP
from vfe3.model.model import VFEModel
from vfe3.model.stack import vfe_stack


def test_block_mlp_known_residual_and_shape():
    """Identity linear maps make ReLU's residual output hand-checkable."""
    mlp = BlockMLP(embed_dim=2, expansion=1, activation="relu", dropout=0.0)
    with torch.no_grad():
        mlp.fc1.weight.copy_(torch.eye(2))
        mlp.fc1.bias.zero_()
        mlp.fc2.weight.copy_(torch.eye(2))
        mlp.fc2.bias.zero_()

    mu = torch.tensor([[-1.0, 2.0]])

    output = mlp(mu)

    assert output.shape == mu.shape
    assert torch.equal(output, torch.tensor([[-1.0, 4.0]]))

@pytest.mark.parametrize("activation", ["gelu", "silu", "relu"])
def test_block_mlp_supported_activations_forward_finite(activation):
    mlp = BlockMLP(embed_dim=3, expansion=2, activation=activation, dropout=0.0)
    mu = torch.tensor([[-1.0, 0.0, 2.0]])
    output = mlp(mu)

    assert output.shape == mu.shape
    assert torch.isfinite(output).all()


def _tiny_cfg(**overrides):
    config = dict(
        vocab_size=17,
        embed_dim=4,
        n_heads=2,
        max_seq_len=4,
        n_layers=1,
        n_e_steps=1,
        e_q_mu_lr=0.1,
        e_q_sigma_lr=0.02,
        e_phi_lr=0.0,
        decode_mode="diagonal_chunked",
        norm_type_block="none",
    )
    config.update(overrides)
    return test_config(**config)


def test_off_mode_registers_no_mlp_state_or_parameters():
    model = VFEModel(_tiny_cfg(use_block_mlp=False))

    assert model.block_mlps is None
    assert not any("block_mlps" in key for key in model.state_dict())
    assert not any("block_mlps" in name for name, _ in model.named_parameters())


@pytest.mark.parametrize(
    "setting",
    [
        {"block_mlp_expansion": 2},
        {"block_mlp_activation": "relu"},
        {"block_mlp_dropout": 0.25},
        {"m_block_mlp_lr": 0.0123},
    ],
)
def test_off_mode_warns_when_mlp_only_setting_is_nondefault(setting):
    # CL01: the structured inactive-field notice names each inert field with its value, and the
    # same structural map feeds sweep admission through inactive_fields().
    (name, value), = setting.items()
    from vfe3.config import VFE3Config
    default = VFE3Config.__dataclass_fields__[name].default
    if value == default:
        value = {"block_mlp_expansion": 3, "block_mlp_activation": "silu",
                 "block_mlp_dropout": 0.125, "m_block_mlp_lr": 0.0234}[name]
        setting = {name: value}
    with pytest.warns(ConfigNotice, match=re.escape(f"{name}={value!r}")):
        cfg = _tiny_cfg(use_block_mlp=False, **setting)
    assert name in cfg.inactive_fields()


def test_active_mlp_is_untied_per_layer():
    model = VFEModel(_tiny_cfg(use_block_mlp=True, n_layers=3))

    assert len(model.block_mlps) == 3
    assert len({id(module) for module in model.block_mlps}) == 3
    assert len({id(parameter) for module in model.block_mlps for parameter in module.parameters()}) == 12


class _Add(nn.Module):
    def __init__(self, amount: float) -> None:
        super().__init__()
        self.amount = amount

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        return mu + self.amount


def test_block_applies_norm_then_mlp_before_handoff():
    cfg = _tiny_cfg(n_layers=2, prior_handoff_rho=1.0)
    model = VFEModel(cfg)
    tokens = torch.tensor([[1, 2, 3, 4]])
    belief = model.prior_bank.encode(tokens)
    block_mlps = nn.ModuleList([_Add(3.0), _Add(5.0)])
    diagnostic = {}
    frame = model._canonical_frame_context(
        build_factored_transport(belief.phi, model.group, compact_blocks=True), compact=True)

    vfe_stack(
        belief,
        belief.mu,
        belief.sigma,
        model.group,
        cfg,
        block_norm=lambda mu, sigma: mu + 2.0,
        block_mlps=block_mlps,
        capture={"diagnostic": diagnostic},
        relational_operators_by_layer=model.relational_operators_by_layer,
        relational_frame=frame,
    )

    first_converged = diagnostic["layer_converged"][0]
    first_output = diagnostic["layer_outputs"][0]
    expected = block_mlps[0](first_converged.mu + 2.0)
    assert torch.equal(first_output.mu, expected)
    assert torch.equal(diagnostic["layer_priors"][1][0], first_output.mu)


def test_coordinate_mlp_preserves_covariance_and_other_belief_fields():
    for family, decode_mode in (("gaussian_diagonal", "diagonal_chunked"),
                                ("gaussian_full", "full_chunked")):
        cfg = _tiny_cfg(family=family, decode_mode=decode_mode, n_heads=2, embed_dim=4)
        model = VFEModel(cfg)
        belief = model.prior_bank.encode(torch.tensor([[1, 2]]))
        capture = {}
        frame = model._canonical_frame_context(
            build_factored_transport(belief.phi, model.group, compact_blocks=True), compact=True)

        out = vfe_block(
            belief,
            belief.mu,
            belief.sigma,
            model.group,
            cfg,
            block_norm=lambda mu, sigma: mu + 2.0,
            block_mlp=_Add(3.0),
            capture=capture,
            relational_operators=model.relational_operators_by_layer[0],
            relational_frame=frame,
        )

        converged = capture["converged"]
        assert torch.equal(out.mu, _Add(3.0)(converged.mu + 2.0))
        assert torch.equal(out.sigma, converged.sigma)
        assert out.phi is converged.phi
        assert out.right_phi is converged.right_phi


@pytest.mark.parametrize("estimator", ["unroll"])
def test_every_active_mlp_layer_receives_finite_gradients(estimator):
    tokens = torch.tensor([[1, 2, 3, 4]])
    targets = torch.tensor([[2, 3, 4, 5]])
    model = VFEModel(_tiny_cfg(
        use_block_mlp=True,
        n_layers=3,
        e_step_gradient=estimator,
    ))

    _, loss, _ = model(tokens, targets)
    loss.backward()

    for layer_index, mlp in enumerate(model.block_mlps):
        for parameter_name, parameter in mlp.named_parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all(), (
                f"non-finite gradient in layer {layer_index} parameter {parameter_name}"
            )


def test_diagnostic_snapshot_with_dropout_is_rng_neutral_and_restores_training_mode():
    model = VFEModel(_tiny_cfg(
        use_block_mlp=True,
        block_mlp_dropout=0.5,
        n_layers=3,
    ))
    model.train()
    tokens = torch.tensor([[1, 2, 3, 4]])

    torch.manual_seed(451)
    expected_next_logits = model(tokens).detach()

    diagnostic_calls = (
        ("snapshot", lambda: model.build_diagnostic_snapshot(tokens)),
        ("summary", lambda: model.diagnostics(tokens)),
        ("attention", lambda: model.attention_maps(tokens)),
        ("per_layer", lambda: model.diagnostics_per_layer(tokens)),
    )
    for diagnostic_name, diagnostic_call in diagnostic_calls:
        torch.manual_seed(451)
        cpu_rng_before = torch.get_rng_state().clone()
        diagnostic_call()

        assert torch.equal(torch.get_rng_state(), cpu_rng_before), diagnostic_name
        assert model.training
        assert all(module.training for module in model.block_mlps.modules())
        actual_next_logits = model(tokens).detach()
        assert torch.equal(actual_next_logits, expected_next_logits), diagnostic_name
