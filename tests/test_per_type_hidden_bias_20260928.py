r"""``per_type_hidden_bias``: the capacity-bearing per-type table on the block-MLP hidden pre-activation.

Tiny CPU models only (one thread). The table :math:`E \in \mathbb R^{V \times n_{\mathrm{gen}}}` is split
over the layers into widths :math:`w_\ell` and enters layer :math:`\ell` as
:math:`h^{(\ell)} = W_1u + b_1 + E^{(\ell)}[x_i]` on the first :math:`w_\ell` hidden units. Checks: zero
initialization leaves logits, loss and every shared gradient bitwise equal to the model without it; a
finite-difference gradient; the parameter count :math:`V n_{\mathrm{gen}}` at diag_K64-like and
full_K64-like tiny shapes; width validation; locality and context dependence; and first-order
non-absorbability: the logit change caused by the table is not in the tangent space of the mean table.
"""

from __future__ import annotations

from tests.profiles import test_config

import warnings
from typing import Any, Dict

import pytest
import torch

from vfe3.config import VFE3Config, config_from_serialized, serialize_config
from vfe3.model.model import VFEModel, per_type_hidden_widths

V = 13


@pytest.fixture(autouse=True)
def _one_thread() -> None:
    torch.set_num_threads(1)


# diag_K64-like and full_K64-like at single-digit width: K = 8 = 2 heads x 4, four layers.
DIAG_LIKE: Dict[str, Any] = dict(
    embed_dim=8, n_heads=2, n_layers=4, family="gaussian_diagonal", decode_mode="linear",
    attention_residual=True, norm_type_block="layernorm", layernorm_affine=True, block_mlp_expansion=8)
FULL_LIKE: Dict[str, Any] = dict(
    embed_dim=8, n_heads=2, n_layers=4, family="gaussian_full", decode_mode="full_chunked",
    predictive_covariance="moment_matched", attention_residual=False, norm_type_block="none",
    layernorm_affine=False, block_mlp_expansion=12)


def _cfg(**over: Any) -> VFE3Config:
    values: Dict[str, Any] = dict(
        vocab_size=V, embed_dim=4, n_heads=2, max_seq_len=6, eval_stride=3, n_layers=2, batch_size=2,
        attention_mode="projected_predictive", e_phi_lr=0.0, block_mlp_expansion=2,
        attention_residual=True, norm_type_block="layernorm", layernorm_affine=True,   # pinned: independent
        skip_belief_sigma_update=False)                                                # of the WIP defaults
    values.update(over)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return test_config(**values)


def _build(cfg: VFE3Config, seed: int = 0, *, dtype: torch.dtype = torch.float32) -> VFEModel:
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(cfg).to(dtype).eval()
    return model


def _perturb(model: VFEModel, seed: int = 0) -> VFEModel:
    g = torch.Generator().manual_seed(seed + 50)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name == "per_type_hidden_bias":
                continue
            if name in ("prior_bank.phi_embed", "pos_phi_free"):
                p.copy_(0.3 * torch.randn(p.shape, generator=g, dtype=p.dtype))
            elif "sigma_log" in name or name.startswith("relational_operators"):
                p.add_(0.3 * torch.randn(p.shape, generator=g, dtype=p.dtype))
    return model


def _tokens(seed: int = 7, B: int = 3, N: int = 6) -> torch.Tensor:
    return torch.randint(0, V, (B, N), generator=torch.Generator().manual_seed(seed))


# ----------------------------------------------------------------------------------------------
# Shapes, widths and zero initialization
# ----------------------------------------------------------------------------------------------

def test_widths_split_n_gen_with_the_remainder_on_the_first_layers() -> None:
    assert per_type_hidden_widths(8192, 4) == (2048,) * 4
    assert per_type_hidden_widths(5120, 6) == (854, 854, 853, 853, 853, 853)
    assert per_type_hidden_widths(32, 3) == (11, 11, 10)
    assert sum(per_type_hidden_widths(4096, 4)) == 4096


@pytest.mark.parametrize("shape", [DIAG_LIKE, FULL_LIKE], ids=["diag_K64_like", "full_K64_like"])
def test_parameter_is_V_by_n_gen_and_zero_init_is_bitwise_the_base(shape: Dict[str, Any]) -> None:
    base = _perturb(_build(_cfg(**shape)))
    table = _perturb(_build(_cfg(per_type_hidden_bias=True, **shape)))
    n_gen = base.prior_bank.phi_embed.shape[-1]
    assert n_gen == shape["n_heads"] * (shape["embed_dim"] // shape["n_heads"]) ** 2
    parameter = dict(table.named_parameters())["per_type_hidden_bias"]
    assert parameter.shape == (V, n_gen) and parameter.numel() == V * n_gen
    assert not bool(parameter.any())
    assert sum(table._per_type_hidden_widths) == n_gen
    assert set(table.state_dict()) - set(base.state_dict()) == {"per_type_hidden_bias"}
    shared = {k: v for k, v in table.state_dict().items() if k != "per_type_hidden_bias"}
    assert all(torch.equal(shared[k], v) for k, v in base.state_dict().items())
    tokens, targets = _tokens(), _tokens(9)
    results = []
    for model in (base, table):
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            logits = model(tokens)
        _, loss, _ = model(tokens, targets)
        loss.backward()
        results.append((logits, loss.detach(), {n: p.grad for n, p in model.named_parameters()}))
    assert torch.equal(results[0][0], results[1][0]) and torch.equal(results[0][1], results[1][1])
    for name, grad in results[0][2].items():
        other = results[1][2][name]
        assert (grad is None and other is None) or torch.equal(grad, other), name
    grad_table = results[1][2]["per_type_hidden_bias"]
    assert grad_table is not None and grad_table.abs().max() > 0          # it trains from step 0
    assert not bool(grad_table[sorted(set(range(V)) - set(tokens.flatten().tolist()))].any())


def test_width_beyond_the_hidden_layer_fails_closed() -> None:
    with pytest.raises(ValueError, match="hidden units"):
        _build(_cfg(per_type_hidden_bias=True, n_layers=1, block_mlp_expansion=1))


def test_nonzero_table_trains_on_both_tiny_bases() -> None:
    for shape in (DIAG_LIKE, FULL_LIKE):
        model = _perturb(_build(_cfg(per_type_hidden_bias=True, **shape))).train()
        with torch.no_grad():
            model.per_type_hidden_bias.normal_(0.0, 0.5)
        _, loss, _ = model(_tokens(), _tokens(9))
        loss.backward()
        assert torch.isfinite(loss) and model.per_type_hidden_bias.grad.abs().max() > 0


# ----------------------------------------------------------------------------------------------
# Gradient
# ----------------------------------------------------------------------------------------------

def test_finite_difference_gradient() -> None:
    r"""Central differences of :math:`L = \langle R, \mu^{(L)}\rangle` in float64 against autograd.

    The objective reads the stack output (float64 end to end); the decode and CE run in the fp32
    island by design, whose rounding would swamp a finite difference.
    """
    model = _perturb(_build(_cfg(per_type_hidden_bias=True, decode_mode="diagonal_chunked"),
                            dtype=torch.float64))
    with torch.no_grad():
        model.per_type_hidden_bias.normal_(0.0, 0.3, generator=torch.Generator().manual_seed(3))
    tokens = _tokens()
    R = torch.randn(*tokens.shape, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(4))

    def objective() -> torch.Tensor:
        return (model.forward_beliefs(tokens)[0].mu * R).sum()

    model.zero_grad(set_to_none=True)
    objective().backward()
    grad = model.per_type_hidden_bias.grad.clone()
    eps = 1e-6
    seen = sorted(set(tokens.flatten().tolist()))
    for v, j in [(seen[0], 0), (seen[1], 3), (seen[-1], 5), (seen[2], 7), (seen[3], 4)]:
        with torch.no_grad():
            model.per_type_hidden_bias[v, j] += eps
            plus = objective().item()
            model.per_type_hidden_bias[v, j] -= 2 * eps
            minus = objective().item()
            model.per_type_hidden_bias[v, j] += eps
        fd = (plus - minus) / (2 * eps)
        assert abs(fd) > 1e-6 or abs(grad[v, j].item()) < 1e-9
        assert abs(fd - grad[v, j].item()) <= 1e-8 + 1e-6 * abs(fd), (v, j, fd, grad[v, j].item())
    assert grad.abs().max() > 1e-3


# ----------------------------------------------------------------------------------------------
# Locality, context dependence, non-absorbability
# ----------------------------------------------------------------------------------------------

def _one_layer_float64() -> VFEModel:
    model = _perturb(_build(_cfg(per_type_hidden_bias=True, n_layers=1, block_mlp_expansion=2,
                                 decode_mode="diagonal_chunked"), dtype=torch.float64))
    return model


def test_table_row_acts_only_at_positions_of_its_type_and_depends_on_context() -> None:
    model = _one_layer_float64()
    v = 5
    tokens = torch.tensor([[1, 2, v, 3, 4, v], [7, 8, v, 3, 4, 9]])          # same type v at position 2
    with torch.no_grad():
        base = model(tokens)
        model.per_type_hidden_bias[v].normal_(0.0, 1.0, generator=torch.Generator().manual_seed(1))
        moved = model(tokens)
    change = (moved - base).abs().amax(-1)                                  # (B, N)
    at_v = tokens == v
    assert bool((change[~at_v] == 0).all())                                 # other types: bitwise unchanged
    assert bool((change[at_v] > 1e-3).all())
    delta = moved - base                                                    # same row, same position:
    assert (delta[0, 2] - delta[1, 2]).abs().max() > 1e-4                   # the effect depends on context


def test_table_is_not_absorbable_into_the_mean_table_to_first_order() -> None:
    r"""The logit change :math:`J_E\,\delta E` is not in the range of :math:`J_\mu` (least-squares residual).

    Absorbing the table into the per-type means would require :math:`J_\mu\,\delta\mu = J_E\,\delta E`
    for some :math:`\delta\mu`; the relative least-squares residual is O(1) for the table and ~0 for a
    direction that is in the range (control).
    """
    from torch.func import functional_call

    model = _one_layer_float64()
    tokens = _tokens(seed=4, B=6)
    mu0 = model.prior_bank.mu_embed.detach().clone()                        # (V, K)
    e0 = model.per_type_hidden_bias.detach().clone()                        # (V, n_gen), zero

    def f_mu(mu: torch.Tensor) -> torch.Tensor:                             # (B N V,)
        return functional_call(model, {"prior_bank.mu_embed": mu}, (tokens,)).flatten()

    def f_e(e: torch.Tensor) -> torch.Tensor:                               # (B N V,)
        return functional_call(model, {"per_type_hidden_bias": e}, (tokens,)).flatten()

    J_mu = torch.autograd.functional.jacobian(f_mu, mu0).reshape(-1, mu0.numel())   # (B N V, V K)
    dE = torch.zeros_like(e0)
    present = sorted(set(tokens.flatten().tolist()))
    dE[present] = torch.randn(len(present), e0.shape[-1], dtype=e0.dtype,
                              generator=torch.Generator().manual_seed(2))
    _, target = torch.autograd.functional.jvp(f_e, e0, dE)                  # (B N V,)
    solution = torch.linalg.lstsq(J_mu, target.unsqueeze(-1)).solution
    residual = (J_mu @ solution).squeeze(-1) - target
    assert target.norm() > 1e-3
    assert residual.norm() / target.norm() > 0.1
    control = J_mu @ torch.randn(mu0.numel(), 1, dtype=mu0.dtype, generator=torch.Generator().manual_seed(5))
    control_fit = J_mu @ torch.linalg.lstsq(J_mu, control).solution
    assert (control_fit - control).norm() / control.norm() < 1e-8
    assert J_mu.shape[0] > J_mu.shape[1]                                   # more logits than mean entries


# ----------------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------------

def test_step_size_resolves_to_the_declared_phi_step_before_coercion() -> None:
    cfg = _cfg(per_type_hidden_bias=True, gauge_transport="off", m_phi_lr=0.0015, pos_phi="none")
    assert cfg.m_phi_lr == 0.0 and cfg.m_per_type_hidden_bias_lr == 0.0015
    again = config_from_serialized(serialize_config(cfg), source="test")
    assert again.m_per_type_hidden_bias_lr == 0.0015 and again.m_phi_lr == 0.0
    assert _cfg(per_type_hidden_bias=True, m_per_type_hidden_bias_lr=0.01).m_per_type_hidden_bias_lr == 0.01
    inactive = cfg.inactive_fields()
    assert "phi_weight_decay" not in inactive and "m_per_type_hidden_bias_lr" not in inactive
    assert "m_per_type_hidden_bias_lr" in _cfg().inactive_fields()
    assert _cfg().m_per_type_hidden_bias_lr is None


def test_row_11_patch_passes_the_table_through() -> None:
    from vfe3.posthoc import interventions as iv

    model = _perturb(_build(_cfg(per_type_hidden_bias=True)))
    with torch.no_grad():
        model.per_type_hidden_bias.normal_(0.0, 0.5, generator=torch.Generator().manual_seed(8))
    tokens = _tokens()
    with torch.no_grad():
        base = model(tokens)
        with iv.frames_removed_norm_mlp(model):
            patched = model(tokens)
    assert torch.isfinite(patched).all() and (patched - base).abs().max() > 1e-4


def test_identity_frame_arm_trains_the_table_and_not_the_frames() -> None:
    model = _build(_cfg(per_type_hidden_bias=True, gauge_transport="off", pos_phi="none")).train()
    _, loss, _ = model(_tokens(), _tokens(9))
    loss.backward()
    assert model.prior_bank.phi_embed.grad is None
    assert model.per_type_hidden_bias.requires_grad and model.per_type_hidden_bias.grad.abs().max() > 0


@pytest.mark.parametrize("over", [
    dict(use_block_mlp=False),
    dict(block_mlp_mode="gaussian_template"),
    dict(block_mlp_mode="gauge_gate", family="gaussian_full", decode_mode="full_chunked"),
])
def test_config_fails_closed_without_a_coordinate_hidden_layer(over: Dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="per_type_hidden_bias"):
        _cfg(per_type_hidden_bias=True, **over)
    with pytest.raises(ValueError, match="m_per_type_hidden_bias_lr"):
        _cfg(per_type_hidden_bias=True, m_per_type_hidden_bias_lr=-1.0)
