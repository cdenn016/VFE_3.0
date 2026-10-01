r"""``block_coordinates='reference'``: LayerNorm and MLP in reference coordinates (M2 arm, M5 row 11 trained).

Tiny CPU models only (one thread). The block computes, with the realized frame :math:`U_i`,

.. math:: \mu_i^{(\ell+1)} = U_i\,\Phi^{(\ell)}(U_i^{-1}\mu_i^{\mathrm{att}}),\qquad
          \Phi^{(\ell)} = (\mathrm{id} + \mathrm{MLP}^{(\ell)})\circ\mathrm{LN}_{\gamma,b},

and a Gaussian-template MLP reads :math:`\operatorname{diag}C_i`. Checks: exact equality with the local
block at identity frames; agreement with the post-hoc row-11 patch and with an independent float64
reference-coordinate forward at the same weights; the helper against a dense float64 formula; passive
covariance under block frame changes :math:`g \in \prod_b GL(d_b)`; float64 gradcheck; identical
parameters and state dicts; fail-closed configuration; row 11 not applicable; a bf16-autocast step.
"""

from __future__ import annotations

from tests.profiles import test_config
import warnings
from typing import Any, Dict

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from vfe3.belief import BeliefState
from vfe3.config import VFE3Config
from vfe3.contracts import CanonicalFrameContext
from vfe3.geometry.norms import AffineLayerNorm
from vfe3.model.block import _reference_coordinate_block
from vfe3.model.block_mlp import BlockMLP, GaussianTemplateBlockMLP
from vfe3.model.model import VFEModel
from vfe3.posthoc import evaluate as ev
from vfe3.posthoc import interventions as iv
from vfe3.posthoc import residual as rs
from vfe3.posthoc import streams as st
from tests.test_posthoc_interventions_roles_20260928 import _reference_logits

V = 24
DEVICE = torch.device("cpu")


@pytest.fixture(autouse=True)
def _one_thread() -> None:
    torch.set_num_threads(1)


def _cfg(**over: Any) -> VFE3Config:
    values: Dict[str, Any] = dict(
        vocab_size=V, embed_dim=4, n_heads=2, max_seq_len=8, eval_stride=4, n_layers=2, batch_size=2,
        attention_mode="projected_predictive", e_phi_lr=0.0, block_mlp_expansion=2,
        attention_residual=True, norm_type_block="layernorm", layernorm_affine=True,   # pinned: independent
        skip_belief_sigma_update=False)                                                # of the WIP defaults
    values.update(over)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return test_config(**values)


def _build(cfg: VFE3Config, seed: int = 0) -> VFEModel:
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VFEModel(cfg).eval()


def _perturb(model: VFEModel, *, identity_frames: bool = False, seed: int = 0) -> VFEModel:
    r"""Move the tiny model off its initialization; ``identity_frames`` zeroes every frame generator."""
    g = torch.Generator().manual_seed(seed + 100)

    def noise(p: torch.Tensor, scale: float) -> torch.Tensor:
        return scale * torch.randn(p.shape, generator=g, dtype=p.dtype)

    with torch.no_grad():
        model.prior_bank.sigma_log_embed.add_(noise(model.prior_bank.sigma_log_embed, 0.5))
        model.prior_bank.phi_embed.copy_(noise(model.prior_bank.phi_embed, 0.3))
        if getattr(model, "pos_phi_free", None) is not None:
            model.pos_phi_free.copy_(noise(model.pos_phi_free, 0.3))
        for op in model.relational_operators_by_layer:
            for p in op.parameters():
                p.add_(noise(p, 0.3))
        for norm in (model.block_norm, model.final_norm):
            if isinstance(norm, torch.nn.Module):
                for p in norm.parameters():
                    p.add_(noise(p, 0.2))
        if identity_frames:
            model.prior_bank.phi_embed.zero_()
            if getattr(model, "pos_phi_free", None) is not None:
                model.pos_phi_free.zero_()
    return model


def _pair(**over: Any):
    r"""A reference-coordinate model and a local model carrying the identical state dict."""
    ref = _perturb(_build(_cfg(block_coordinates="reference", **over)))
    loc = _build(_cfg(**over))
    loc.load_state_dict(ref.state_dict(), strict=True)
    return ref, loc


def _tokens(seed: int = 7, B: int = 3) -> torch.Tensor:
    return torch.randint(0, V, (B, 8), generator=torch.Generator().manual_seed(seed))


def _logits(model: VFEModel, tokens: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model.forward_beliefs(tokens, return_logits=True)[1]


def _dataset(seed: int = 3) -> Any:
    return st.eval_windows(torch.randint(0, V, (61,), generator=torch.Generator().manual_seed(seed)), 8, 4)


LAYERNORM_FULL = dict(norm_type_final="layernorm", )
TEMPLATE = dict(block_mlp_mode="gaussian_template")
MOMENT_MATCHED = dict(predictive_covariance="moment_matched", attention_residual=False)
VARIANTS = {
    "affine_ln_coordinate_passthrough_residual": LAYERNORM_FULL,
    "template_passthrough_residual":             TEMPLATE,
    "affine_ln_coordinate_moment_matched":       dict(LAYERNORM_FULL, **MOMENT_MATCHED),
    "template_moment_matched":                   dict(TEMPLATE, **MOMENT_MATCHED),
    "full_family_template_full_decode":          dict(TEMPLATE, family="gaussian_full", decode_mode="full_chunked"),
}


# ----------------------------------------------------------------------------------------------
# Defaults and identity frames
# ----------------------------------------------------------------------------------------------

def test_defaults_are_local_and_add_nothing() -> None:
    cfg = test_config()
    assert cfg.block_coordinates == "local" and cfg.decode_coordinates == "local"
    assert cfg.per_type_hidden_bias is False and cfg.m_per_type_hidden_bias_lr is None
    model = _build(_cfg())
    assert model.per_type_hidden_bias is None and model.prior_bank.decode_coordinates == "local"
    assert not any(k.startswith("per_type_hidden_bias") for k in model.state_dict())


def test_explicit_local_defaults_are_bitwise_the_implicit_defaults() -> None:
    tokens, targets = _tokens(1), _tokens(2)
    implicit = _perturb(_build(_cfg(**LAYERNORM_FULL)))
    explicit = _build(_cfg(block_coordinates="local", decode_coordinates="local",
                           per_type_hidden_bias=False, **LAYERNORM_FULL))
    explicit.load_state_dict(implicit.state_dict(), strict=True)
    assert torch.equal(_logits(implicit, tokens), _logits(explicit, tokens))
    losses = []
    for model in (implicit, explicit):
        model.zero_grad(set_to_none=True)
        _, loss, _ = model(tokens, targets)
        loss.backward()
        losses.append((loss.detach(), {n: p.grad.clone() for n, p in model.named_parameters()
                                       if p.grad is not None}))
    assert torch.equal(losses[0][0], losses[1][0])
    assert losses[0][1].keys() == losses[1][1].keys()
    assert all(torch.equal(losses[0][1][n], losses[1][1][n]) for n in losses[0][1])


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_reference_equals_local_exactly_at_identity_frames(name: str) -> None:
    r"""With :math:`\phi = 0` every :math:`U_i = I` exactly, so the conjugation is bitwise a no-op."""
    ref = _perturb(_build(_cfg(block_coordinates="reference", **VARIANTS[name])), identity_frames=True)
    loc = _build(_cfg(**VARIANTS[name]))
    loc.load_state_dict(ref.state_dict(), strict=True)
    tokens, targets = _tokens(), _tokens(11)
    assert torch.equal(_logits(ref, tokens), _logits(loc, tokens))
    with torch.no_grad():
        assert torch.equal(ref(tokens, targets)[1], loc(tokens, targets)[1])


# ----------------------------------------------------------------------------------------------
# Same weights: the post-hoc row-11 patch and the independent reference forward
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_reference_model_equals_row_11_patch_at_the_same_weights(name: str) -> None:
    ref, loc = _pair(**VARIANTS[name])
    tokens = _tokens()
    trained = _logits(ref, tokens)
    with iv.frames_removed_norm_mlp(loc):
        patched = _logits(loc, tokens)
    local = _logits(loc, tokens)
    assert (trained - local).abs().max() > 1e-3                     # the frames act on this model
    np.testing.assert_allclose(trained.double().numpy(), patched.double().numpy(), rtol=0, atol=1e-5)


@pytest.mark.parametrize("name", ["affine_ln_coordinate_passthrough_residual", "template_passthrough_residual"])
def test_reference_model_matches_independent_float64_normal_form(name: str) -> None:
    ref, _ = _pair(**VARIANTS[name])
    tokens = _tokens(5)
    expected = _reference_logits(ref, tokens, (iv.ROLE_NORM_MLP,))
    np.testing.assert_allclose(_logits(ref, tokens).double().numpy(), expected.numpy(), rtol=0, atol=5e-5)


# ----------------------------------------------------------------------------------------------
# The block helper: dense float64 formula, passive covariance, gradcheck
# ----------------------------------------------------------------------------------------------

B, N, G, D = 2, 3, 2, 3
K = G * D


def _gl_blocks(scale: float, seed: int) -> torch.Tensor:                   # (B, N, G, d, d)
    g = torch.Generator().manual_seed(seed)
    return torch.linalg.matrix_exp(scale * torch.randn(B, N, G, D, D, generator=g, dtype=torch.float64))


def _frame(blocks: torch.Tensor) -> CanonicalFrameContext:
    return CanonicalFrameContext(blocks, torch.linalg.inv(blocks), block_diagonal=True)


def _dense(blocks: torch.Tensor) -> torch.Tensor:                           # (B, N, K, K)
    out = torch.zeros(*blocks.shape[:-3], K, K, dtype=blocks.dtype)
    for h in range(G):
        out[..., h * D:(h + 1) * D, h * D:(h + 1) * D] = blocks[..., h, :, :]
    return out


def _modules(template: bool, seed: int = 0):
    torch.manual_seed(seed)
    norm = AffineLayerNorm(K).double()
    with torch.no_grad():
        norm.weight.add_(0.3 * torch.randn(K, dtype=torch.float64))
        norm.bias.add_(0.3 * torch.randn(K, dtype=torch.float64))
    mlp = (GaussianTemplateBlockMLP(K, 2, 0.0) if template else BlockMLP(K, 2, "gelu", 0.0)).double()
    return norm, mlp


def _spd(seed: int) -> torch.Tensor:                                        # (B, N, K, K)
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(B, N, K, K, generator=g, dtype=torch.float64)
    return A @ A.transpose(-1, -2) + 0.5 * torch.eye(K, dtype=torch.float64)


def _belief(mu: torch.Tensor, sigma: torch.Tensor) -> BeliefState:
    return BeliefState(mu=mu, sigma=sigma, phi=torch.zeros(B, N, 1, dtype=torch.float64))


@pytest.mark.parametrize("template", [False, True])
def test_helper_matches_dense_float64_formula(template: bool) -> None:
    norm, mlp = _modules(template)
    blocks = _gl_blocks(0.5, 1)
    U = _dense(blocks)
    U_inv = torch.linalg.inv(U)
    mu = torch.randn(B, N, K, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    Sigma = _spd(3)
    out = _reference_coordinate_block(_belief(mu, Sigma), block_norm=norm, block_mlp=mlp,
                                      frame=_frame(blocks), capture=None, hidden_bias=None)
    xi = (U_inv @ mu.unsqueeze(-1)).squeeze(-1)
    v = F.layer_norm(xi, (K,), weight=norm.weight, bias=norm.bias, eps=norm.eps)
    if template:
        C = U_inv @ Sigma @ U_inv.transpose(-1, -2)
        v = mlp.mean_update(v, torch.diagonal(C, dim1=-2, dim2=-1))
    else:
        v = v + mlp.fc2(F.gelu(mlp.fc1(v)))
    expected = (U @ v.unsqueeze(-1)).squeeze(-1)
    assert (out.mu - expected).abs().max() < 1e-12
    assert out.sigma is Sigma                                               # covariance passthrough


@pytest.mark.parametrize("template", [False, True])
def test_passive_block_covariance(template: bool) -> None:
    r""":math:`f_{gU}(g\mu, g\Sigma g^\top) = g f_U(\mu, \Sigma)` for :math:`g \in \prod_b GL(d_b)`."""
    norm, mlp = _modules(template)
    blocks, g_blocks = _gl_blocks(0.5, 4), _gl_blocks(0.6, 5)
    g = _dense(g_blocks)
    mu = torch.randn(B, N, K, dtype=torch.float64, generator=torch.Generator().manual_seed(6))
    Sigma = _spd(7)

    def run(frame_blocks, m, S, *, local: bool = False):
        if local:                                                           # the evaluated local block
            v = norm(m, S)
            if template:
                return mlp.mean_update(v, torch.diagonal(S, dim1=-2, dim2=-1))
            return v + mlp.fc2(F.gelu(mlp.fc1(v)))
        return _reference_coordinate_block(_belief(m, S), block_norm=norm, block_mlp=mlp,
                                           frame=_frame(frame_blocks), capture=None, hidden_bias=None).mu

    gmv = lambda x: (g @ x.unsqueeze(-1)).squeeze(-1)                       # noqa: E731
    moved = run(g_blocks @ blocks, gmv(mu), g @ Sigma @ g.transpose(-1, -2))
    assert (moved - gmv(run(blocks, mu, Sigma))).abs().max() < 1e-11
    local_moved = run(None, gmv(mu), g @ Sigma @ g.transpose(-1, -2), local=True)
    assert (local_moved - gmv(run(None, mu, Sigma, local=True))).abs().max() > 1e-2   # negative control


@pytest.mark.parametrize("template", [False, True])
def test_helper_gradcheck_through_mean_and_frame_generator(template: bool) -> None:
    norm, mlp = _modules(template, seed=3)
    X = 0.3 * torch.randn(B, N, G, D, D, dtype=torch.float64, generator=torch.Generator().manual_seed(8))
    mu = torch.randn(B, N, K, dtype=torch.float64, generator=torch.Generator().manual_seed(9))
    sigma = torch.rand(B, N, K, dtype=torch.float64, generator=torch.Generator().manual_seed(10)) + 0.5

    def f(m: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        frame = CanonicalFrameContext(torch.linalg.matrix_exp(x), torch.linalg.matrix_exp(-x), True)
        return _reference_coordinate_block(_belief(m, sigma), block_norm=norm, block_mlp=mlp,
                                           frame=frame, capture=None, hidden_bias=None).mu

    assert torch.autograd.gradcheck(f, (mu.requires_grad_(), X.requires_grad_()), eps=1e-6, atol=1e-6)


def test_helper_fails_closed_without_a_frame() -> None:
    norm, mlp = _modules(False)
    with pytest.raises(RuntimeError, match="realized relational frame"):
        _reference_coordinate_block(_belief(torch.zeros(B, N, K, dtype=torch.float64),
                                            torch.ones(B, N, K, dtype=torch.float64)),
                                    block_norm=norm, block_mlp=mlp, frame=None, capture=None, hidden_bias=None)


# ----------------------------------------------------------------------------------------------
# Parameters, state dicts and capture coordinates
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["affine_ln_coordinate_passthrough_residual", "template_passthrough_residual"])
def test_parameters_and_state_dicts_are_identical(name: str) -> None:
    ref, loc = _pair(**VARIANTS[name])
    assert [(n, tuple(p.shape)) for n, p in ref.named_parameters()] == \
        [(n, tuple(p.shape)) for n, p in loc.named_parameters()]
    loc.load_state_dict(ref.state_dict(), strict=True)
    ref.load_state_dict(loc.state_dict(), strict=True)
    assert ref.executable_build.block_mlp.parameter_count == loc.executable_build.block_mlp.parameter_count


def test_capture_keeps_local_coordinates() -> None:
    ref, _ = _pair(**LAYERNORM_FULL)
    capture: Dict[str, Any] = {"diagnostic": {}}
    with torch.no_grad():
        belief, _ = ref.forward_beliefs(_tokens(), capture=capture)
    last_output = capture["diagnostic"]["layer_outputs"][-1]
    assert torch.equal(last_output.mu, capture["out"].mu)                  # the stack output is local
    first_in = capture["diagnostic"]["layer_block_mlp_inputs"][0].mu
    # the captured MLP input is U_i LN(U_i^{-1} mu^att), not LN(mu^att)
    converged = capture["diagnostic"]["layer_converged"][0].mu
    assert (first_in - ref.block_norm(converged, None)).abs().max() > 1e-3


# ----------------------------------------------------------------------------------------------
# Configuration envelope
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("over, message", [
    (dict(attention_mode="transported", attention_residual=False), "retired"),
    (dict(attention_mode="projected_conditional", attention_residual=False, n_layers=1), "retired"),
    (dict(pos_rotation="rope"), "pos_rotation"),
    (dict(family="gaussian_frame_diagonal", decode_mode="diagonal_chunked"), "gaussian_frame_diagonal"),
    (dict(block_mlp_mode="canonical_frame"), "block_mlp_mode"),
    (dict(block_mlp_mode="gauge_gate", family="gaussian_full", decode_mode="full_chunked"), "block_mlp_mode"),
    (dict(block_mlp_covariance="delta_full", family="gaussian_full", decode_mode="full_chunked"),
     "block_mlp_covariance"),
    (dict(norm_type_block="mahalanobis"), "norm_type_block"),
    (dict(norm_type_final="mahalanobis"), "norm_type_final"),
    (dict(use_head_mixer=True), "use_head_mixer"),
])
def test_config_fails_closed_outside_the_frame_envelope(over: Dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _cfg(block_coordinates="reference", **over)
    with pytest.raises(ValueError, match="block_coordinates must be one of"):
        _cfg(block_coordinates="canonical")


def test_inactive_marker_for_identity_frames_and_empty_blocks() -> None:
    assert "block_coordinates" in _cfg(block_coordinates="reference", gauge_transport="off").inactive_fields()
    assert "block_coordinates" in _cfg(block_coordinates="reference", norm_type_block="none",
                                       use_block_mlp=False).inactive_fields()
    assert "block_coordinates" not in _cfg(block_coordinates="reference").inactive_fields()


# ----------------------------------------------------------------------------------------------
# Consumers that assume local-coordinate norms
# ----------------------------------------------------------------------------------------------

def test_row_11_not_applicable_while_rows_9_10_12_run() -> None:
    ref, _ = _pair(**LAYERNORM_FULL)
    dataset = _dataset()
    idx = np.arange(len(dataset))
    with pytest.raises(iv.NotApplicable, match="nothing to remove"):
        with iv.frames_removed_norm_mlp(ref):
            ev.nll_pass(ref, dataset, idx, batch_size=2, device=DEVICE)
    for row in (iv.frames_removed_input_moments, iv.frames_removed_routing_covariance,
                iv.frames_removed_decoder):
        with row(ref):
            ev.nll_pass(ref, dataset, idx, batch_size=2, device=DEVICE)
    check = iv.frame_role_composite_check(ref, dataset, idx, batch_size=2, device=DEVICE)
    assert check["frame_free_by_construction"] == [iv.ROLE_NORM_MLP]
    assert check["record"]["roles"] == [iv.ROLE_INPUT_MOMENTS, iv.ROLE_ROUTING_COVARIANCE, iv.ROLE_DECODER]
    assert check["max_abs_nll_difference"] < 5e-6


def test_ln_change_is_undefined_under_reference_coordinates() -> None:
    ref, loc = _pair(**LAYERNORM_FULL)
    assert rs.ln_change_applicability(loc) == (True, None)
    defined, reason = rs.ln_change_applicability(ref)
    assert not defined and "block_coordinates='reference'" in reason


# ----------------------------------------------------------------------------------------------
# Training under bf16 autocast
# ----------------------------------------------------------------------------------------------

def test_one_bf16_autocast_training_step() -> None:
    model = _perturb(_build(_cfg(block_coordinates="reference", amp_dtype="bf16", **LAYERNORM_FULL))).train()
    tokens, targets = _tokens(1), _tokens(2)
    _, loss, _ = model(tokens, targets)
    assert torch.isfinite(loss)
    loss.backward()
    grads = {n: p.grad for n, p in model.named_parameters()}
    assert grads["prior_bank.phi_embed"] is not None and grads["prior_bank.phi_embed"].abs().max() > 0
    assert grads["block_norm.weight"].abs().max() > 0
    assert all(grads[n] is not None and grads[n].abs().max() > 0 for n in grads if n.startswith("block_mlps."))
    capture: Dict[str, Any] = {"diagnostic": {}}
    with torch.no_grad():
        model.forward_beliefs(tokens, capture=capture)
    assert all(state.mu.dtype == torch.float32 for state in capture["diagnostic"]["layer_outputs"])


# ----------------------------------------------------------------------------------------------
# Shared frame actions: compact and dense layouts agree with the float64 formulas
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("full", [False, True])
def test_frame_actions_compact_and_dense_layouts_agree(full: bool) -> None:
    from vfe3.model.frame_actions import frame_matvec, reference_covariance, reference_variances

    blocks = _gl_blocks(0.5, 12)
    compact = _frame(blocks)
    dense = CanonicalFrameContext(_dense(blocks), torch.linalg.inv(_dense(blocks)))
    mu = torch.randn(B, N, K, dtype=torch.float64, generator=torch.Generator().manual_seed(13))
    sigma = _spd(14) if full else torch.rand(B, N, K, dtype=torch.float64) + 0.5
    Sigma = sigma if full else torch.diag_embed(sigma)
    U_inv = dense.inverse
    C = U_inv @ Sigma @ U_inv.transpose(-1, -2)                             # float64 reference formula
    for frame in (compact, dense):
        bd = frame.block_diagonal
        assert (frame_matvec(frame.inverse, mu, bd) - (U_inv @ mu.unsqueeze(-1)).squeeze(-1)).abs().max() < 1e-13
        assert (reference_covariance(mu, sigma, frame) - C).abs().max() < 1e-12
        assert (reference_variances(mu, sigma, frame) - torch.diagonal(C, dim1=-2, dim2=-1)).abs().max() < 1e-12
    Cf = reference_covariance(mu.float(), sigma.float(), compact)          # float64 frame: escalated, float32 out
    assert Cf.dtype == torch.float32 and torch.equal(Cf, Cf.transpose(-1, -2))
