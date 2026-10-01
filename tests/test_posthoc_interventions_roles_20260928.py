r"""Table M5 rows 9-12: frames removed by role (``vfe3.posthoc.interventions.frames_removed``).

Tiny CPU models only. The checks are: the rows are ordinary registry entries; each single-role patch is
exactly (bitwise) the identity map when every frame is already the identity; the patched model equals an
independent implementation of the reference normal form (Corollary ``cor:normal_form``) with
:math:`U_i = I` in the removed roles; all four roles together equal rows 7 and 8 together; every patch
restores the model bit-exactly; and routes outside the normal form raise ``NotApplicable``.
"""

from __future__ import annotations

from tests.profiles import test_config

import math
import warnings
from typing import Any, Sequence

import numpy as np
import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.free_energy import attention_tau
from vfe3.model.model import VFEModel
from vfe3.posthoc import evaluate as ev
from vfe3.posthoc import interventions as iv
from vfe3.posthoc import streams as st

V = 24
DEVICE = torch.device("cpu")
ROW_KEYS = {9: "frames_removed_input_moments", 10: "frames_removed_routing_covariance",
            11: "frames_removed_norm_mlp", 12: "frames_removed_decoder"}
ROLE_OF_ROW = {9: iv.ROLE_INPUT_MOMENTS, 10: iv.ROLE_ROUTING_COVARIANCE, 11: iv.ROLE_NORM_MLP, 12: iv.ROLE_DECODER}


@pytest.fixture(autouse=True)
def _one_thread() -> None:
    torch.set_num_threads(1)


def _cfg(**over: Any) -> VFE3Config:
    values = dict(vocab_size=V, embed_dim=4, n_heads=2, max_seq_len=8, eval_stride=4, n_layers=2,
                  batch_size=2, attention_mode="projected_predictive", e_phi_lr=0.0, block_mlp_expansion=2,
                  attention_residual=True, norm_type_block="none", layernorm_affine=False,   # pinned: independent
                  skip_belief_sigma_update=False)                                          # of the config defaults
    values.update(over)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return test_config(**values)


def _model(cfg: VFE3Config, *, identity_frames: bool = False, seed: int = 0) -> VFEModel:
    r"""A tiny model moved off its initialization; ``identity_frames`` zeroes every frame generator."""
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(cfg).eval()
    with torch.no_grad():
        model.prior_bank.sigma_log_embed.normal_(0.0, 0.5)
        model.prior_bank.phi_embed.normal_(0.0, 0.3)
        if getattr(model, "pos_phi_free", None) is not None:
            model.pos_phi_free.normal_(0.0, 0.3)
        for op in model.relational_operators_by_layer:
            for p in op.parameters():
                p.add_(0.3 * torch.randn_like(p))
        for p in (model.block_norm.parameters() if isinstance(model.block_norm, torch.nn.Module) else ()):
            p.add_(0.2 * torch.randn_like(p))
        if identity_frames:
            model.prior_bank.phi_embed.zero_()
            if getattr(model, "pos_phi_free", None) is not None:
                model.pos_phi_free.zero_()
    return model


def _dataset(seed: int = 3) -> Any:
    return st.eval_windows(torch.randint(0, V, (61,), generator=torch.Generator().manual_seed(seed)), 8, 4)


def _nll(model: VFEModel, dataset: Any) -> np.ndarray:
    return ev.nll_pass(model, dataset, np.arange(len(dataset)), batch_size=2, device=DEVICE).nll


def _logits(model: VFEModel, tokens: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model.forward_beliefs(tokens, return_logits=True)[1]


LAYERNORM = dict(norm_type_block="layernorm", layernorm_affine=True, norm_type_final="layernorm")
CONFIGS = {
    "mlp_only_head_scope":   {},
    "layernorm_full_scope":  dict(LAYERNORM, relational_input_scope="full", relational_output_scope="full"),
    "template_mlp":          dict(norm_type_block="layernorm", norm_type_final="layernorm",
                                  block_mlp_mode="gaussian_template"),
    "full_family_gaussian_decode": dict(family="gaussian_full", decode_mode="full_chunked",
                                        block_mlp_mode="gaussian_template"),
}


# ----------------------------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------------------------

def test_rows_9_to_12_are_registry_entries() -> None:
    rows = {i.row: i for i in iv.INTERVENTIONS}
    assert sorted(rows) == list(range(1, 13)) and iv.NOT_IMPLEMENTED == ()
    labels = {9: "Frames removed from the input moments only",
              10: "Frames removed from the routing covariance only",
              11: "Frames removed from the normalization and MLP conjugation only",
              12: "Frames removed from the decoder only"}
    for row, key in ROW_KEYS.items():
        assert rows[row].key == key and rows[row].label == labels[row]
    assert [i.key for i in iv.by_key([ROW_KEYS[11]])] == ["baseline", ROW_KEYS[11]]
    with pytest.raises(ValueError):
        with iv.frames_removed(_model(_cfg()), ("not_a_role",)):
            pass


# ----------------------------------------------------------------------------------------------
# Exactness at identity frames and restoration
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_single_roles_are_exact_identities_when_frames_are_identity(name: str) -> None:
    r"""With :math:`\phi = 0` every :math:`U_i = I` exactly, so each applicable row is bitwise a no-op."""
    model = _model(_cfg(**CONFIGS[name]), identity_frames=True)
    dataset = _dataset()
    tokens = torch.stack([dataset[i][0] for i in range(4)])
    base_nll, base_logits = _nll(model, dataset), _logits(model, tokens)
    applied = []
    for row in ROW_KEYS:
        try:
            with iv.by_key([ROW_KEYS[row]])[1].apply(model):
                nll, logits = _nll(model, dataset), _logits(model, tokens)
        except iv.NotApplicable:
            continue
        applied.append(row)
        assert np.array_equal(nll, base_nll), (name, row)
        assert torch.equal(logits, base_logits), (name, row)
    expected = [9, 10, 11] if name == "full_family_gaussian_decode" else [9, 10, 11, 12]
    assert applied == expected


@pytest.mark.parametrize("name", ["layernorm_full_scope", "template_mlp"])
def test_rows_change_the_output_and_restore_the_model_bit_exactly(name: str) -> None:
    from vfe3.gradients import qkv, relational
    from vfe3.model import model as model_module
    from vfe3.model import stack as stack_module

    cfg = _cfg(**CONFIGS[name])
    model = _model(cfg)
    dataset = _dataset(5)
    base = _nll(model, dataset)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    final_norm, score = model.final_norm, relational.relational_score
    n_batches = math.ceil(len(dataset) / 2)
    for row in ROW_KEYS:
        with iv.by_key([ROW_KEYS[row]])[1].apply(model) as record:
            arm = _nll(model, dataset)
        assert np.abs(arm - base).max() > 1e-4, row
        assert record["stack_calls"] == n_batches and record["roles"] == [ROLE_OF_ROW[row]]
        if row == 10:
            assert record["intercepted_calls"] == cfg.n_layers * n_batches
        if row == 12:
            assert record["decode_calls"] == n_batches
    assert np.array_equal(_nll(model, dataset), base)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert model_module.vfe_stack is stack_module.vfe_stack
    assert relational.relational_score is score and getattr(qkv, "relational_score", score) is score
    assert "_decode_belief_with_context" not in vars(model) and model.final_norm is final_norm


def test_an_exception_inside_the_patch_still_restores() -> None:
    from vfe3.model import model as model_module
    from vfe3.model import stack as stack_module

    model = _model(_cfg(**CONFIGS["layernorm_full_scope"]))
    dataset = _dataset()
    base = _nll(model, dataset)
    with pytest.raises(KeyboardInterrupt):
        with iv.frames_removed(model, iv.FRAME_ROLES):
            raise KeyboardInterrupt
    assert model_module.vfe_stack is stack_module.vfe_stack and "_decode_belief_with_context" not in vars(model)
    assert np.array_equal(_nll(model, dataset), base)


# ----------------------------------------------------------------------------------------------
# Independent reference-coordinate forward (Corollary cor:normal_form)
# ----------------------------------------------------------------------------------------------

def _gaussian_kl(a: torch.Tensor, Vq: torch.Tensor, b: torch.Tensor, Vk: torch.Tensor) -> torch.Tensor:
    r""":math:`\mathrm{KL}(\mathcal N(a_i,V_{Q,i})\|\mathcal N(b_j,V_{K,j}))` for all pairs, (..., N, N)."""
    r = a.shape[-1]
    P = torch.linalg.inv(Vk)                                                   # (..., N, r, r)
    delta = b.unsqueeze(-3) - a.unsqueeze(-2)                                  # (..., N_i, N_j, r)
    trace = torch.einsum("...jab,...iba->...ij", P, Vq)
    quad = torch.einsum("...ija,...jab,...ijb->...ij", delta, P, delta)
    logdet = torch.logdet(Vk).unsqueeze(-2) - torch.logdet(Vq).unsqueeze(-1)
    return 0.5 * (trace + quad - r + logdet)


def _reference_logits(model: VFEModel, tokens: torch.Tensor, removed: Sequence[str]) -> torch.Tensor:
    r"""Logits of the normal form with :math:`U_i = I` in the ``removed`` roles and the trained frames elsewhere.

    .. math::
        \xi^{(0)} = U_9^{-1}\mu_p,\ C = U_{10}^{-1}\Sigma_p U_{10}^{-\top},\
        \xi^{\mathrm{att}} = \xi + \sum_h\sum_j\beta_{ij}^{(h)}O_hT_h\xi_j,\
        \xi' = U_{11}^{-1}\Phi(U_{11}\xi^{\mathrm{att}}),\
        \ell = W_{\mathrm{out}}U_{12}\,U_{11}^{-1}\mathrm{FN}(U_{11}\xi^{(L)}) + b,

    with :math:`U_r \in \{U_i, I\}` per role; the attention step never sees a frame. A Gaussian-template
    block reads the state covariance in the coordinates it acts in, :math:`U_{11}C^{\mathrm{state}}U_{11}^\top`
    with :math:`C^{\mathrm{state}} = U_i^{-1}\Sigma_p U_i^{-\top}`. Written directly in reference
    coordinates (no model patch), in float64 except the trained LN/MLP modules.
    """
    from vfe3.model.block_mlp import GaussianTemplateBlockMLP

    cfg = model.cfg
    with torch.no_grad():
        beliefs = model.prior_bank.encode(tokens)
        beliefs = beliefs._replace(right_phi=model._pos_phi_right(beliefs.phi))
        frame = model._canonical_frame_context(
            model._build_shared_flat_transport(beliefs, None, require_vertex_factors=True))   # dense (B, N, K, K)
        U, U_inv = frame.forward.double(), frame.inverse.double()
        eye = torch.eye(U.shape[-1], dtype=torch.float64).expand_as(U)
        U9, U10, U11, U12 = ((eye, eye) if role in removed else (U, U_inv) for role in iv.FRAME_ROLES)
        mv = lambda M, x: (M @ x.unsqueeze(-1)).squeeze(-1)                     # noqa: E731

        mu_p = beliefs.mu.double()                                             # (B, N, K)
        Sigma = (beliefs.sigma.double() if beliefs.sigma.dim() == 4
                 else torch.diag_embed(beliefs.sigma.double()))                # (B, N, K, K)
        xi = mv(U9[1], mu_p)                                                   # xi^(0)
        C = U10[1] @ Sigma @ U10[1].transpose(-1, -2)                          # routing covariance
        C_state = U_inv @ Sigma @ U_inv.transpose(-1, -2)                      # the state's reference covariance
        N, K = tokens.shape[1], U.shape[-1]
        log_prior = model._attention_log_prior(N, tokens.device).double()
        tau = float(attention_tau(cfg.kappa_beta, model.group.irrep_dims))
        for layer, op in enumerate(model.relational_operators_by_layer):
            H, d = len(op.irrep_dims), op.irrep_dims[0]
            S_q, S_k = (S.double() for S in op.comparison_pair())             # (H, d, input_dim)
            T_hat, O_hat = op._ambient_content_maps(torch.float64)             # (H, d, K), (H, K, d)
            for _ in range(cfg.n_e_steps):
                scores = []
                for h in range(H):
                    cols = slice(0, K) if op.input_scope == "full" else slice(h * d, (h + 1) * d)
                    x, Ch = xi[..., cols], C[..., cols, cols]
                    a, b = mv(S_q[h], x), mv(S_k[h], x)                        # (B, N, d)
                    Vq = S_q[h] @ Ch @ S_q[h].T
                    Vk = S_k[h] @ Ch @ S_k[h].T
                    scores.append(_gaussian_kl(a, Vq, b, Vk))
                score = torch.stack(scores, dim=-3).clamp(0.0, cfg.kl_max)     # (B, H, N, N)
                beta = torch.softmax(-score / tau + log_prior, dim=-1)
                values = torch.einsum("hsk,bnk->bnhs", T_hat, xi)              # (B, N, H, d)
                message = torch.einsum("hks,bhij,bjhs->bik", O_hat, beta, values)
                xi = xi + message if op.attention_residual else message
            x = mv(U11[0], xi).float()                                         # coordinates the block acts in
            if model.block_norm is not None:
                x = model.block_norm(x, beliefs.sigma)
            if model.block_mlps is not None:
                mlp = model.block_mlps[layer]
                if isinstance(mlp, GaussianTemplateBlockMLP):
                    local = U11[0] @ C_state @ U11[0].transpose(-1, -2)
                    x = mlp.mean_update(x, torch.diagonal(local, dim1=-2, dim2=-1).float())
                else:
                    x = mlp.forward(x)
            xi = mv(U11[1], x.double())
        if model.final_norm is not None:
            xi = mv(U11[1], model.final_norm(mv(U11[0], xi).float(), beliefs.sigma).double())
        x = mv(U12[0], xi)                                                     # decoder W_out U_12
        pb = model.prior_bank
        logits = x @ pb.output_proj_weight.double().T
        if pb.output_proj_bias is not None:
            logits = logits + pb.output_proj_bias.double()
        if pb.decode_unigram_prior:
            logits = logits + pb._unigram_bias().double()
    return logits


@pytest.mark.parametrize("name", ["mlp_only_head_scope", "layernorm_full_scope", "template_mlp"])
def test_patched_rows_match_an_independent_reference_coordinate_forward(name: str) -> None:
    model = _model(_cfg(**CONFIGS[name]))
    dataset = _dataset(7)
    tokens = torch.stack([dataset[i][0] for i in range(4)])
    subsets = [()] + [(role,) for role in iv.FRAME_ROLES] + [iv.FRAME_ROLES]
    for removed in subsets:
        reference = _reference_logits(model, tokens, removed)
        if removed:
            with iv.frames_removed(model, removed):
                logits = _logits(model, tokens)
        else:
            logits = _logits(model, tokens)
        np.testing.assert_allclose(logits.double().numpy(), reference.numpy(), rtol=0, atol=5e-5,
                                   err_msg=f"{name}: roles removed {removed}")
    # the roles are not all no-ops on this model
    for role in iv.FRAME_ROLES:
        assert (_reference_logits(model, tokens, (role,)) - _reference_logits(model, tokens, ())).abs().max() > 1e-3


# ----------------------------------------------------------------------------------------------
# All four roles against rows 7 and 8
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("pos_phi", ["learned", "none"])
def test_all_roles_removed_equals_frames_identity_in_every_role(pos_phi: str) -> None:
    model = _model(_cfg(pos_phi=pos_phi, **LAYERNORM))
    dataset = _dataset(9)
    check = iv.frame_role_composite_check(model, dataset, np.arange(len(dataset)), batch_size=2, device=DEVICE)
    assert check["positional_frames"] is (pos_phi != "none")
    assert check["max_abs_nll_difference"] < 5e-6
    assert abs(check["ce_all_roles_removed"] - float(_nll(model, dataset).mean())) > 1e-3
    if pos_phi == "none":                            # without positional frames rows 7+8 is row 7
        with iv.token_frames_identity(model):
            row7 = _nll(model, dataset)
        assert abs(float(row7.mean()) - check["ce_frames_identity"]) < 1e-12


# ----------------------------------------------------------------------------------------------
# Routes outside the normal form
# ----------------------------------------------------------------------------------------------

def _not_applicable(model: VFEModel, row: int) -> str:
    with pytest.raises(iv.NotApplicable) as caught:
        with iv.by_key([ROW_KEYS[row]])[1].apply(model):
            _nll(model, _dataset())
    return str(caught.value)


def test_not_applicable_routes() -> None:
    transported = _model(_cfg(attention_mode="transported", attention_residual=False, relational_output_scope="head"))
    for row in ROW_KEYS:
        assert "projected predictive" in _not_applicable(transported, row)
    gaussian_decode = _model(_cfg(family="gaussian_full", decode_mode="full_chunked"))
    assert "linear decoder" in _not_applicable(gaussian_decode, 12)
    moment_matched = _model(_cfg(predictive_covariance="moment_matched"))
    assert "moment_matched" in _not_applicable(moment_matched, 10)
    bare = _model(_cfg(use_block_mlp=False))
    assert "nothing to remove" in _not_applicable(bare, 11)
    canonical = _model(_cfg(block_mlp_mode="canonical_frame"))
    assert "nothing to remove" in _not_applicable(canonical, 11)
    for row in (9, 10, 12):                          # the other roles still apply on these models
        for model in (bare, canonical):
            with iv.by_key([ROW_KEYS[row]])[1].apply(model):
                _nll(model, _dataset())


def test_canonical_frame_mlp_is_kept_and_only_the_norm_is_conjugated() -> None:
    model = _model(_cfg(block_mlp_mode="canonical_frame", norm_type_block="layernorm"))
    dataset = _dataset()
    base = _nll(model, dataset)
    with iv.frames_removed_norm_mlp(model) as record:
        arm = _nll(model, dataset)
    assert record["block_norm"] == "conjugated" and record["block_mlp"] == ["covariant"] * model.cfg.n_layers
    assert np.abs(arm - base).max() > 1e-4
