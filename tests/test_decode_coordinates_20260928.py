r"""``decode_coordinates='reference'``: the decoder reads the reference moments (M2 covariant decoder).

Tiny CPU models only (one thread). With the realized frame :math:`U_i` the decoder reads
:math:`\xi_i = U_i^{-1}\mu_i` and :math:`C_i = U_i^{-1}\Sigma_iU_i^{-\top}`:

.. math:: \ell_{iv} = (W_{\mathrm{out}}\xi_i)_v + b_v + \kappa\log p_v \quad\text{(linear)},\qquad
          \ell_{iv} = -\mathrm{KL}(\mathcal N(\xi_i, C_i)\,\|\,p_v)/\tau + \kappa\log p_v \quad\text{(Gaussian)}.

Checks: (a) manual logits for the linear, full and diagonal Gaussian decodes; (b) equality with the
post-hoc row-12 patch at the same weights; (c) fused training CE equals the dense CE, gradients included;
(d) a one-layer functional stack in float64 (attention, Mahalanobis norm, diagonal gauge gate, reference
decoder) is covariant under block frame changes :math:`g \in \prod_b GL(d_b)`: the diagonal run on
:math:`(\mu, \sigma, U)` equals the full-covariance run on :math:`(g\mu, g\Sigma g^\top, gU)`;
(e) fail-closed configuration; (f) row 12 not applicable; plus the relaxed gauge-gate validation.
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
from vfe3.model.block import vfe_block
from vfe3.model.model import VFEModel
from vfe3.model.prior_bank import bounded_variance_from_log
from vfe3.posthoc import evaluate as ev
from vfe3.posthoc import interventions as iv
from vfe3.posthoc import streams as st

V = 24
DEVICE = torch.device("cpu")


@pytest.fixture(autouse=True)
def _one_thread() -> None:
    torch.set_num_threads(1)


def _cfg(**over: Any) -> VFE3Config:
    values: Dict[str, Any] = dict(
        vocab_size=V, embed_dim=4, n_heads=2, max_seq_len=8, eval_stride=4, n_layers=2, batch_size=2,
        attention_mode="projected_predictive", block_mlp_expansion=2,
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
    g = torch.Generator().manual_seed(seed + 100)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name == "prior_bank.phi_embed" or name == "pos_phi_free":
                p.copy_(0.3 * torch.randn(p.shape, generator=g, dtype=p.dtype))
            elif "sigma_log" in name or name.startswith("relational_operators"):
                p.add_(0.3 * torch.randn(p.shape, generator=g, dtype=p.dtype))
            elif name.startswith("block_mlps") or name.startswith("prior_bank.output_proj_bias"):
                p.add_(0.2 * torch.randn(p.shape, generator=g, dtype=p.dtype))
        counts = torch.randint(1, 50, (V,), generator=torch.Generator().manual_seed(1))
        if getattr(model.prior_bank, "decode_unigram_prior", False):
            model.prior_bank.set_unigram_log_prior(counts)
    return model


def _tokens(seed: int = 7, B: int = 3) -> torch.Tensor:
    return torch.randint(0, V, (B, 8), generator=torch.Generator().manual_seed(seed))


def _targets(seed: int = 11, B: int = 3) -> torch.Tensor:
    targets = torch.randint(0, V, (B, 8), generator=torch.Generator().manual_seed(seed))
    targets[0, :2] = -100                                                   # ignored positions too
    return targets


def _dataset(seed: int = 3) -> Any:
    return st.eval_windows(torch.randint(0, V, (61,), generator=torch.Generator().manual_seed(seed)), 8, 4)


def _dense_frame(model: VFEModel, tokens: torch.Tensor) -> CanonicalFrameContext:
    r"""The realized :math:`U_i` rebuilt independently of the forward, as a dense (B, N, K, K) pair."""
    with torch.no_grad():
        beliefs = model.prior_bank.encode(tokens)
        beliefs = beliefs._replace(right_phi=model._pos_phi_right(beliefs.phi))
        return model._canonical_frame_context(
            model._build_shared_flat_transport(beliefs, None, require_vertex_factors=True))


GAUSSIAN_FULL = dict(family="gaussian_full", decode_mode="full_chunked")
GAUSSIAN_DIAG = dict(decode_mode="diagonal_chunked")
MOMENT_MATCHED_FULL = dict(GAUSSIAN_FULL, predictive_covariance="moment_matched", attention_residual=False,
                           norm_type_block="none", layernorm_affine=False, block_mlp_expansion=3)


# ----------------------------------------------------------------------------------------------
# (a) Manual logits
# ----------------------------------------------------------------------------------------------

def test_linear_logits_are_W_xi_plus_bias_plus_unigram() -> None:
    model = _build(_cfg(decode_coordinates="reference"))
    tokens = _tokens()
    with torch.no_grad():
        belief, logits = model.forward_beliefs(tokens, return_logits=True)
    frame = _dense_frame(model, tokens)
    pb = model.prior_bank
    with torch.no_grad():
        xi = (frame.inverse.double() @ belief.mu.double().unsqueeze(-1)).squeeze(-1)   # (B, N, K)
        manual = (xi @ pb.output_proj_weight.double().T + pb.output_proj_bias.double()
                  + pb._unigram_bias().double())
        local = belief.mu.double() @ pb.output_proj_weight.double().T
    np.testing.assert_allclose(logits.double().numpy(), manual.numpy(), rtol=0, atol=1e-5)
    assert (manual - local - pb.output_proj_bias.double() - pb._unigram_bias().double()).abs().max() > 1e-3
    assert torch.equal(model(tokens), logits)                              # inference forward, same path


def _manual_kl_logits(model: VFEModel, belief: BeliefState, frame: CanonicalFrameContext) -> torch.Tensor:
    r""":math:`-\mathrm{KL}(\mathcal N(\xi, C)\|\mathcal N(m_v, \operatorname{diag}s_v))/\tau + \kappa\log p_v` in float64."""
    pb = model.prior_bank
    with torch.no_grad():
        return _manual_kl_body(pb, belief, frame)


def _manual_kl_body(pb: Any, belief: BeliefState, frame: CanonicalFrameContext) -> torch.Tensor:
    U_inv = frame.inverse.double()
    Sigma = belief.sigma.double()
    Sigma = Sigma if Sigma.dim() == 4 else torch.diag_embed(Sigma)
    xi = (U_inv @ belief.mu.double().unsqueeze(-1)).squeeze(-1)             # (B, N, K)
    C = U_inv @ Sigma @ U_inv.transpose(-1, -2)                             # (B, N, K, K)
    m_v = pb._decode_mu_table().double()                                    # (V, K)
    s_v = bounded_variance_from_log(pb._decode_sigma_log_table(), eps=pb.eps).double()
    diag_C = torch.diagonal(C, dim1=-2, dim2=-1)                            # (B, N, K)
    quad = ((xi.unsqueeze(-2) - m_v) ** 2 / s_v).sum(-1)                   # (B, N, V)
    trace = (diag_C.unsqueeze(-2) / s_v).sum(-1)
    kl = 0.5 * (trace + quad - xi.shape[-1] + torch.log(s_v).sum(-1) - torch.logdet(C).unsqueeze(-1))
    return -kl / pb._tau_eff().double() + pb._unigram_bias().double()


@pytest.mark.parametrize("over", [GAUSSIAN_FULL, MOMENT_MATCHED_FULL], ids=["passthrough", "moment_matched"])
def test_full_gaussian_logits_are_kl_of_the_reference_moments(over: Dict[str, Any]) -> None:
    model = _build(_cfg(decode_coordinates="reference", **over))
    tokens = _tokens()
    with torch.no_grad():
        belief, logits = model.forward_beliefs(tokens, return_logits=True)
    manual = _manual_kl_logits(model, belief, _dense_frame(model, tokens))
    np.testing.assert_allclose(logits.double().numpy(), manual.numpy(), rtol=1e-5, atol=1e-4)


def test_diagonal_gaussian_decode_equals_full_C_kl_up_to_a_per_position_constant() -> None:
    model = _build(_cfg(decode_coordinates="reference", **GAUSSIAN_DIAG))
    tokens = _tokens()
    with torch.no_grad():
        belief, logits = model.forward_beliefs(tokens, return_logits=True)
    manual = _manual_kl_logits(model, belief, _dense_frame(model, tokens))
    np.testing.assert_allclose(torch.log_softmax(logits.double(), -1).numpy(),
                               torch.log_softmax(manual, -1).numpy(), rtol=0, atol=1e-5)
    shift = (logits.double() - manual)                                      # per-position constant
    assert (shift - shift[..., :1]).abs().max() < 1e-4 and shift.abs().max() > 1e-6


# ----------------------------------------------------------------------------------------------
# (b) Same weights: the post-hoc row-12 patch
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("block", ["local", "reference"])
def test_linear_reference_decode_equals_row_12_patch(block: str) -> None:
    ref = _build(_cfg(decode_coordinates="reference", block_coordinates=block))
    loc = _build(_cfg(block_coordinates=block), seed=1)
    loc.load_state_dict(ref.state_dict(), strict=True)
    tokens = _tokens(3)
    with torch.no_grad():
        trained = ref.forward_beliefs(tokens, return_logits=True)[1]
        with iv.frames_removed_decoder(loc):
            patched = loc.forward_beliefs(tokens, return_logits=True)[1]
        local = loc.forward_beliefs(tokens, return_logits=True)[1]
    assert (trained - local).abs().max() > 1e-3
    np.testing.assert_allclose(trained.numpy(), patched.numpy(), rtol=0, atol=1e-6)


# ----------------------------------------------------------------------------------------------
# (c) Fused training CE equals the dense CE
# ----------------------------------------------------------------------------------------------

FUSED_ROUTES = {
    "linear":                 {},
    "linear_reference_block": dict(block_coordinates="reference", norm_type_final="layernorm"),
    "diagonal_chunked":       GAUSSIAN_DIAG,
    "full_chunked":           GAUSSIAN_FULL,
    "full_moment_matched":    MOMENT_MATCHED_FULL,
}


@pytest.mark.parametrize("name", sorted(FUSED_ROUTES))
def test_fused_training_ce_equals_dense_ce(name: str) -> None:
    model = _build(_cfg(decode_coordinates="reference", **FUSED_ROUTES[name]))
    tokens, targets = _tokens(), _targets()
    model.zero_grad(set_to_none=True)
    _, loss, stats = model(tokens, targets, return_decode_stats=True)
    loss.backward()
    fused = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    logits = model(tokens)
    dense = F.cross_entropy(logits.reshape(-1, V).double(), targets.reshape(-1), ignore_index=-100)
    dense.backward()
    np.testing.assert_allclose(loss.item(), dense.item(), rtol=0, atol=2e-6)
    assert int(stats.scored_tokens) == int((targets != -100).sum())
    for n, grad in fused.items():
        np.testing.assert_allclose(grad.double().numpy(), model.get_parameter(n).grad.double().numpy(),
                                   rtol=1e-4, atol=2e-6, err_msg=n)



def test_fused_training_loss_matches_dense_logits_under_reference_coordinates() -> None:
    r"""The fused full_chunked CE equals F.cross_entropy on the dense reference-coordinate logits."""
    from vfe3.config import serialize_config

    payload = serialize_config(_cfg(**GAUSSIAN_FULL))
    payload.update(decode_mode="full_chunked", decode_coordinates="reference")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = test_config(**payload)
    model = _build(cfg)
    tokens, targets = _tokens(), _targets()
    with torch.no_grad():
        _, loss, _ = model(tokens, targets)
        logits = model(tokens)
    dense = F.cross_entropy(logits.reshape(-1, V).double(), targets.reshape(-1), ignore_index=-100)
    np.testing.assert_allclose(float(loss), float(dense), rtol=0, atol=2e-6)


def test_decode_consumers_thread_the_frame_and_fail_closed_without_it() -> None:
    model = _build(_cfg(decode_coordinates="reference", **GAUSSIAN_FULL))
    tokens = _tokens()
    capture: Dict[str, Any] = {}
    with torch.no_grad():
        belief, logits = model.forward_beliefs(tokens, return_logits=True, capture=capture)
        again = model._decode_belief_with_context(belief.mu, belief.sigma,
                                                  canonical_frame=capture["canonical_frame"])
        last = model.forward_beliefs(tokens, return_logits=True, decode_last=True)[1]
        degenerate = model.prior_bank.decode_degenerate_positions(
            belief.sigma, canonical_frame=capture["canonical_frame"])
    assert torch.equal(again, logits)
    np.testing.assert_allclose(last[:, 0].numpy(), logits[:, -1].numpy(), rtol=0, atol=1e-6)
    assert degenerate is not None and not bool(degenerate.any())
    with pytest.raises(ValueError, match="realized relational frame"):
        model._decode_belief_with_context(belief.mu, belief.sigma)
    nll = ev.nll_pass(model, _dataset(), np.arange(3), batch_size=2, device=DEVICE)
    assert np.isfinite(nll.nll).all()


# ----------------------------------------------------------------------------------------------
# (d) One-layer functional stack: covariance under block frame changes
# ----------------------------------------------------------------------------------------------

B, N, H, D = 2, 5, 2, 3
K = H * D


def _gl_blocks(scale: float, seed: int) -> torch.Tensor:                   # (B, N, H, d, d)
    g = torch.Generator().manual_seed(seed)
    return torch.linalg.matrix_exp(scale * torch.randn(B, N, H, D, D, generator=g, dtype=torch.float64))


def _block_diag(blocks: torch.Tensor) -> torch.Tensor:                      # (B, N, K, K)
    out = torch.zeros(*blocks.shape[:-3], K, K, dtype=blocks.dtype)
    for h in range(H):
        out[..., h * D:(h + 1) * D, h * D:(h + 1) * D] = blocks[..., h, :, :]
    return out


def _gate_model(family: str, decode_mode: str, decode_coordinates: str = "reference") -> VFEModel:
    cfg = _cfg(vocab_size=V, embed_dim=K, n_heads=H, max_seq_len=N, n_layers=1, family=family,
               decode_mode=decode_mode, decode_coordinates=decode_coordinates, use_block_mlp=True,
               block_mlp_mode="gauge_gate", norm_type_block="mahalanobis", layernorm_affine=False,
               pos_phi="none")
    model = _build(cfg, dtype=torch.float64)
    g = torch.Generator().manual_seed(21)
    with torch.no_grad():
        for p in model.block_mlps.parameters():                            # the gate is the identity at init
            p.add_(0.5 * torch.randn(p.shape, generator=g, dtype=p.dtype))
    return model


def _one_layer(model: VFEModel, mu: torch.Tensor, sigma: torch.Tensor, frame: CanonicalFrameContext):
    n_gen = model.prior_bank.phi_embed.shape[-1]
    belief = BeliefState(mu=mu, sigma=sigma, phi=torch.zeros(B, N, n_gen, dtype=torch.float64))
    with torch.no_grad():
        out = vfe_block(belief, mu, sigma, model.group, model.cfg,
                        log_prior=model._attention_log_prior(N, DEVICE),
                        block_norm=model.block_norm, block_mlp=model.block_mlps[0],
                        relational_operators=model.relational_operators_by_layer[0],
                        relational_frame=frame, relational_value_frame=frame)
        reference = model.cfg.decode_coordinates == "reference"
        logits = model.prior_bank.decode(out.mu, out.sigma, canonical_frame=frame if reference else None)
    return out, logits


@pytest.mark.parametrize("decode", ["linear", "gaussian"])
def test_functional_stack_is_covariant_under_block_frame_changes(decode: str) -> None:
    diag = _gate_model("gaussian_diagonal", "linear" if decode == "linear" else "diagonal_chunked")
    full = _gate_model("gaussian_full", "linear" if decode == "linear" else "full_chunked")
    full.load_state_dict(diag.state_dict(), strict=True)
    gen = torch.Generator().manual_seed(4)
    mu = torch.randn(B, N, K, generator=gen, dtype=torch.float64)
    sigma = torch.rand(B, N, K, generator=gen, dtype=torch.float64) + 0.5
    U_blocks, g_blocks = _gl_blocks(0.4, 5), _gl_blocks(0.5, 6)
    U = CanonicalFrameContext(U_blocks, torch.linalg.inv(U_blocks), block_diagonal=True)
    gU = CanonicalFrameContext(g_blocks @ U_blocks, torch.linalg.inv(g_blocks @ U_blocks), block_diagonal=True)
    g = _block_diag(g_blocks)
    gmv = lambda x: (g @ x.unsqueeze(-1)).squeeze(-1)                       # noqa: E731

    out_d, logits_d = _one_layer(diag, mu, sigma, U)
    out_f, logits_f = _one_layer(full, gmv(mu), g @ torch.diag_embed(sigma) @ g.transpose(-1, -2), gU)
    assert (out_f.mu - gmv(out_d.mu)).abs().max() < 1e-10                   # the stack moves with g
    assert (out_d.mu - mu).abs().max() > 1e-2                               # and does something
    if decode == "linear":
        assert (logits_f - logits_d).abs().max() < 1e-10                    # the logits are invariant
    else:                                                                   # diag C vs full C: equal softmax
        assert (torch.log_softmax(logits_f, -1) - torch.log_softmax(logits_d, -1)).abs().max() < 1e-10

    # negative control: the local decoder (W_out mu, or the KL of (mu, Sigma)) is not invariant
    local = _gate_model("gaussian_full", full.cfg.decode_mode, decode_coordinates="local")
    local.load_state_dict(full.state_dict(), strict=True)
    _, local_logits_f = _one_layer(local, gmv(mu), g @ torch.diag_embed(sigma) @ g.transpose(-1, -2), gU)
    _, local_logits = _one_layer(local, mu, torch.diag_embed(sigma), U)
    assert (torch.log_softmax(local_logits_f, -1) - torch.log_softmax(local_logits, -1)).abs().max() > 1e-3


# ----------------------------------------------------------------------------------------------
# (e) Configuration envelope; (f) row 12
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("over, message", [
    (dict(attention_mode="transported", attention_residual=False,
          decode_mode="diagonal_chunked"), "transported"),
    (dict(rope_mode="frame_value"), "rope_mode"),
    (dict(family="gaussian_frame_diagonal", decode_mode="diagonal_chunked"), "gaussian_frame_diagonal"),
    (dict(decode_mode="expected_likelihood_chunked"), "decode_mode"),
])
def test_config_fails_closed_where_the_decode_frame_is_undefined(over: Dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _cfg(decode_coordinates="reference", **over)
    with pytest.raises(ValueError, match="decode_coordinates must be one of"):
        _cfg(decode_coordinates="canonical")


def test_prior_bank_rejects_reference_decode_outside_its_contract() -> None:
    from vfe3.model.prior_bank import PriorBank

    with pytest.raises(ValueError, match="decode_coordinates"):
        PriorBank(V, 4, 8, decode_mode="expected_likelihood_chunked", decode_coordinates="reference")
    with pytest.raises(ValueError, match="decode_coordinates"):
        PriorBank(V, 4, 8, decode_mode="linear", decode_coordinates="canonical")


def test_row_12_not_applicable_while_rows_9_to_11_run() -> None:
    model = _build(_cfg(decode_coordinates="reference", norm_type_final="layernorm"))
    dataset = _dataset()
    idx = np.arange(len(dataset))
    with pytest.raises(iv.NotApplicable, match="nothing to remove"):
        with iv.frames_removed_decoder(model):
            ev.nll_pass(model, dataset, idx, batch_size=2, device=DEVICE)
    for row in (iv.frames_removed_input_moments, iv.frames_removed_routing_covariance,
                iv.frames_removed_norm_mlp):
        with row(model):
            ev.nll_pass(model, dataset, idx, batch_size=2, device=DEVICE)
    check = iv.frame_role_composite_check(model, dataset, idx, batch_size=2, device=DEVICE)
    assert check["frame_free_by_construction"] == [iv.ROLE_DECODER]
    assert check["max_abs_nll_difference"] < 5e-6


def test_identity_frames_mark_decode_coordinates_inert() -> None:
    assert "decode_coordinates" in _cfg(decode_coordinates="reference", gauge_transport="off").inactive_fields()
    assert "decode_coordinates" not in _cfg(decode_coordinates="reference").inactive_fields()


# ----------------------------------------------------------------------------------------------
# Gauge gate on the diagonal family (relaxed validation) and the inactive-family marker
# ----------------------------------------------------------------------------------------------

def test_gauge_gate_admits_the_diagonal_family_and_delta_full_still_needs_full() -> None:
    _cfg(use_block_mlp=True, block_mlp_mode="gauge_gate", norm_type_block="mahalanobis",
         decode_mode="diagonal_chunked")
    with pytest.raises(ValueError, match="full-covariance"):
        _cfg(use_block_mlp=True, block_mlp_mode="gauge_gate", block_mlp_covariance="delta_full",
             decode_mode="diagonal_chunked")
    with pytest.raises(ValueError, match="retired"):
        _cfg(attention_mode="transported", attention_residual=False,
             use_block_mlp=True, block_mlp_mode="gauge_gate", decode_mode="diagonal_chunked")


def test_family_marker_no_longer_depends_on_a_passthrough_gauge_gate() -> None:
    gate = dict(GAUSSIAN_FULL, use_block_mlp=True, block_mlp_mode="gauge_gate", norm_type_block="none")
    assert "family" in _cfg(**gate).inactive_fields()
    assert "family" not in _cfg(**gate, block_mlp_covariance="delta_full").inactive_fields()
    assert "family" not in _cfg(**gate, decode_coordinates="reference").inactive_fields()
