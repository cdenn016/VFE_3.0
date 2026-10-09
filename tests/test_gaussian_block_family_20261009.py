r"""The ``gaussian_block`` belief family: one SPD :math:`d\times d` covariance block per head (audit 2026-10-09).

block_glk's structure group is :math:`\prod_h GL(d)`, and the block-diagonal SPD matrices are the smallest SPD
family closed under its congruence. The family stores the dense :math:`(\dots, K, K)` layout of the full family
with zero off-head blocks, and the moment-matched predictive update returns the M-projection onto it, the head
blocks :math:`U_{i,h}[C^{\mathrm{att}}_i]_{hh}U_{i,h}^\top`. Checks (float64, tiny dims): the family hooks; the
blockwise update equals the head blocks of the dense update; the next-layer reference covariance is
frame-independent while the diagonal projection is not; logits and the training loss are invariant under the
passive action :math:`(\mu,\Sigma,U)\to(g\mu,g\Sigma g^\top,gU)` on the reference arm; the diagonal decoder's
cross-entropy equals the full decoder's; the configuration scope.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

import pytest
import torch

from tests.profiles import make_relational_operators, tiny_config
from vfe3.contracts import CanonicalFrameContext
from vfe3.families import family_cov_kind, get_family
from vfe3.model.model import VFEModel
from vfe3.viz.extract import _act_on_frame, _dense_frame

H, D = 2, 3
K = H * D
DT = torch.float64


@pytest.fixture(autouse=True)
def _one_thread() -> None:
    torch.set_num_threads(1)


def _block_diag(blocks: torch.Tensor) -> torch.Tensor:          # (..., H, d, d) -> (..., K, K)
    return _dense_frame(blocks, True)


def _random_blocks(batch: tuple, g: torch.Generator, *, spd: bool) -> torch.Tensor:
    A = torch.randn(*batch, H, D, D, generator=g, dtype=DT)
    if spd:
        return A @ A.transpose(-1, -2) + 0.3 * torch.eye(D, dtype=DT)
    return torch.linalg.matrix_exp(0.4 * A)


def _diag_blocks(S: torch.Tensor) -> torch.Tensor:              # (..., K, K) -> (..., H, d, d)
    return torch.diagonal(S.reshape(*S.shape[:-2], H, D, H, D), dim1=-4, dim2=-2).movedim(-1, -3)


# ----------------------------------------------------------------------------------------------
# Family hooks
# ----------------------------------------------------------------------------------------------

def test_family_is_registered_with_the_dense_layout_and_projects_onto_head_blocks() -> None:
    fam = get_family("gaussian_block")
    assert family_cov_kind("gaussian_block") == "full" and fam.head_block_covariance
    assert not get_family("gaussian_full").head_block_covariance
    g = torch.Generator().manual_seed(0)
    A = torch.randn(3, K, K, generator=g, dtype=DT)
    S = A @ A.transpose(-1, -2) + torch.eye(K, dtype=DT)
    q = fam.from_covariance(torch.zeros(3, K, dtype=DT), S, head_dims=[D] * H)
    assert torch.equal(q.sigma, _block_diag(_diag_blocks(S)))
    with pytest.raises(ValueError, match="head_dims"):
        fam.from_covariance(torch.zeros(3, K, dtype=DT), S)
    # The M-projection keeps the marginal blocks: KL(p || q) over block-diagonal q is minimized there, so
    # any move that stays inside the block family (an in-block entry, kept SPD) raises it.
    p = get_family("gaussian_full")(torch.zeros(3, K, dtype=DT), S)
    best = get_family("gaussian_full")(torch.zeros(3, K, dtype=DT), q.sigma)
    kl_best = p.renyi_closed_form(best, kl_max=1e12)
    for (a, b), scale in (((0, 1), 0.05), ((0, 1), -0.05), ((D, D), 0.1), ((D + 1, D + 2), -0.05)):
        moved = q.sigma.clone()
        moved[..., a, b] += scale
        if a != b:
            moved[..., b, a] += scale
        other = get_family("gaussian_full")(torch.zeros(3, K, dtype=DT), moved)
        assert bool((p.renyi_closed_form(other, kl_max=1e12) > kl_best).all())


# ----------------------------------------------------------------------------------------------
# The moment-matched update on the head blocks
# ----------------------------------------------------------------------------------------------

def _operators(block: bool, residual: bool, seed: int = 3):
    ops = make_relational_operators([D] * H, 11, readout_mode="predictive_content",
                                    predictive_covariance="moment_matched", attention_residual=residual,
                                    content_link_chart="exponential", block_covariance=block).double()
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in ops.parameters():
            p.add_(0.3 * torch.randn(p.shape, generator=g, dtype=DT))
    return ops


def _update_inputs(seed: int = 5, B: int = 2, N: int = 4):
    g = torch.Generator().manual_seed(seed)
    mu = torch.randn(B, N, K, generator=g, dtype=DT)
    sigma = _block_diag(_random_blocks((B, N), g, spd=True))
    beta = torch.softmax(torch.randn(B, H, N, N, generator=g, dtype=DT), dim=-1)
    U = _random_blocks((B, N), g, spd=False)
    frame = CanonicalFrameContext(U, torch.linalg.inv(U), True)
    action = _random_blocks((B, N), g, spd=False)
    return mu, sigma, beta, frame, action


def _act(action: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor):
    G = _block_diag(action)
    return (G @ mu.unsqueeze(-1)).squeeze(-1), G @ sigma @ G.transpose(-1, -2)


def _reference(sigma: torch.Tensor, frame: CanonicalFrameContext) -> torch.Tensor:
    inverse = _dense_frame(frame.inverse, True)
    return inverse @ sigma @ inverse.transpose(-1, -2)


@pytest.mark.parametrize("residual", [False, True])
def test_block_update_is_the_head_block_projection_of_the_dense_update(residual: bool) -> None:
    mu, sigma, beta, frame, _ = _update_inputs()
    block = _operators(True, residual).predictive_covariance_update(mu, sigma, beta, frame)
    dense = _operators(False, residual).predictive_covariance_update(mu, sigma, beta, frame)  # full U C U^T
    assert block.shape == sigma.shape
    assert (block - _block_diag(_diag_blocks(block))).abs().max() == 0.0          # exactly head-block diagonal
    assert (dense - _block_diag(_diag_blocks(dense))).abs().max() > 1e-3          # the dense result is not
    assert (block - _block_diag(_diag_blocks(dense))).abs().max() / dense.abs().max() < 1e-12
    assert bool((torch.linalg.eigvalsh(block) > 0).all())


@pytest.mark.parametrize("residual", [False, True])
def test_next_layer_reference_covariance_is_frame_independent_only_for_the_block_projection(residual: bool) -> None:
    r"""Under :math:`U\to gU` the block projection's :math:`C^+` is fixed; the diagonal projection's moves."""
    mu, sigma, beta, frame, action = _update_inputs()
    mu_g, sigma_g = _act(action, mu, sigma)
    frame_g = _act_on_frame(action, frame)
    block_ops, dense_ops = _operators(True, residual), _operators(False, residual)
    C = _reference(block_ops.predictive_covariance_update(mu, sigma, beta, frame), frame)
    C_g = _reference(block_ops.predictive_covariance_update(mu_g, sigma_g, beta, frame_g), frame_g)
    block_deviation = float((C - C_g).abs().max())
    # Diagonal projection of the same moment-matched covariance, read back in reference coordinates.
    full, full_g = (dense_ops.predictive_covariance_update(*args) for args in
                    ((mu, sigma, beta, frame), (mu_g, sigma_g, beta, frame_g)))
    D_ref = _reference(torch.diag_embed(torch.diagonal(full, dim1=-2, dim2=-1)), frame)
    D_ref_g = _reference(torch.diag_embed(torch.diagonal(full_g, dim1=-2, dim2=-1)), frame_g)
    diagonal_deviation = float((D_ref - D_ref_g).abs().max())
    print(f"residual={residual}: block deviation {block_deviation:.3e}, diagonal deviation "
          f"{diagonal_deviation:.3e} (norm {float(D_ref.abs().max()):.3e})")
    assert block_deviation / float(C.abs().max()) < 1e-12
    assert diagonal_deviation / float(D_ref.abs().max()) > 1e-2


def test_block_operators_refuse_routes_they_cannot_serve() -> None:
    with pytest.raises(ValueError, match="block_covariance"):
        make_relational_operators([D, D + 1], 11, readout_mode="predictive_content",
                                  predictive_covariance="moment_matched", block_covariance=True)
    with pytest.raises(ValueError, match="block_covariance"):
        make_relational_operators([D] * H, 11, readout_mode="predictive_content", predictive_readout="minimal_agent",
                                  predictive_covariance="moment_matched", block_covariance=True)
    with pytest.raises(ValueError, match="block_covariance"):
        make_relational_operators([D] * H, 11, readout_mode="predictive_content", content_link_chart="exponential",
                                  content_link_edges="lowrank", predictive_covariance="moment_matched",
                                  block_covariance=True)
    ops = _operators(True, False)
    mu, sigma, beta, frame, _ = _update_inputs()
    dense_frame = CanonicalFrameContext(_dense_frame(frame.forward, True), _dense_frame(frame.inverse, True))
    with pytest.raises(ValueError, match="compact"):
        ops.predictive_covariance_update(mu, sigma, beta, dense_frame)


# ----------------------------------------------------------------------------------------------
# The model on the reference arm
# ----------------------------------------------------------------------------------------------

def _cfg(**over: Any):
    values = dict(family="gaussian_block", embed_dim=K, n_heads=H, vocab_size=9, max_seq_len=5, n_layers=2,
                  decode_mode="full_chunked", block_coordinates="reference", decode_coordinates="reference",
                  norm_type_block="layernorm", layernorm_affine=True, use_block_mlp=True,
                  block_mlp_mode="coordinate", content_link_chart="exponential")
    values.update(over)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tiny_config(**values)


def _model(cfg, seed: int = 0) -> VFEModel:
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(cfg).double().eval()
    g = torch.Generator().manual_seed(seed + 50)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad or name.endswith("phi_embed"):
                scale = 0.3 if name.endswith("phi_embed") else 0.2
                p.add_(scale * torch.randn(p.shape, generator=g, dtype=DT))
    return model


def _forward(model: VFEModel, tokens: torch.Tensor, targets: torch.Tensor,
             action: Optional[torch.Tensor], *, move_frames: bool = True):
    r"""Float64 logits and the training loss with the passive action on the encoded beliefs and the frames.

    The model's decode island casts the final belief to float32 (``_decode_belief_with_context``), so the
    logits are read here by the same prior-bank decode at float64, through the captured decode frame. The
    training loss keeps that float32 island.
    """
    pb = model.prior_bank
    encode, frames = pb.encode, model._relational_frames

    def acted_encode(token_ids: torch.Tensor):
        belief = encode(token_ids)
        mu, sigma = _act(action, belief.mu, belief.sigma)
        return belief._replace(mu=mu, sigma=sigma)

    def acted_frames(beliefs, rope, **kwargs):
        omega, frame, value_frame = frames(beliefs, rope, **kwargs)
        assert value_frame is frame
        moved = _act_on_frame(action, frame) if move_frames else frame
        return omega, moved, moved

    if action is not None:
        pb.encode, model._relational_frames = acted_encode, acted_frames
    try:
        with torch.no_grad():
            capture: dict = {}
            belief, _ = model.forward_beliefs(tokens, capture=capture)
            logits = pb.decode(belief.mu, belief.sigma, canonical_frame=capture.get("canonical_frame"))
            loss = model(tokens, targets)[1]
    finally:
        for owner, name in ((pb, "encode"), (model, "_relational_frames")):
            owner.__dict__.pop(name, None)
    return logits, loss


@pytest.mark.parametrize("decode_mode", ["full_chunked", "diagonal_chunked"])
@pytest.mark.parametrize("covariance", ["moment_matched", "passthrough"])
def test_logits_and_loss_are_invariant_under_the_passive_block_action(decode_mode: str, covariance: str) -> None:
    model = _model(_cfg(decode_mode=decode_mode, predictive_covariance=covariance))
    g = torch.Generator().manual_seed(9)
    tokens = torch.randint(0, 9, (2, 5), generator=g)
    targets = torch.randint(0, 9, (2, 5), generator=g)
    action = _random_blocks((2, 5), g, spd=False)
    logits, loss = _forward(model, tokens, targets, None)
    logits_g, loss_g = _forward(model, tokens, targets, action)
    deviation = float((logits - logits_g).abs().max() / logits.abs().max())
    print(f"{decode_mode}/{covariance}: relative logit deviation {deviation:.3e}")
    assert logits.dtype == DT and deviation < 1e-11
    assert abs(float(loss - loss_g)) < 1e-5 * abs(float(loss))       # float32 decode island
    # Not vacuous: moving the beliefs without their frames changes the logits.
    moved_only, _ = _forward(model, tokens, targets, action, move_frames=False)
    assert float((moved_only - logits).abs().max()) > 1e-3


def test_diagonal_decoder_cross_entropy_equals_the_full_decoder_on_the_block_family() -> None:
    r"""Against a diagonal bank only :math:`\mathrm{diag}\,C_i` and the v-independent :math:`\log\det C_i` enter."""
    full = _model(_cfg(decode_mode="full_chunked"))
    diag = _model(_cfg(decode_mode="diagonal_chunked"))
    diag.load_state_dict(full.state_dict(), strict=True)
    g = torch.Generator().manual_seed(4)
    tokens, targets = torch.randint(0, 9, (2, 5), generator=g), torch.randint(0, 9, (2, 5), generator=g)
    logits_full, loss_full = _forward(full, tokens, targets, None)
    logits_diag, loss_diag = _forward(diag, tokens, targets, None)
    shift = logits_full - logits_diag                               # per position, constant over v
    assert float((shift - shift[..., :1]).abs().max()) < 1e-11
    assert float(shift.abs().max()) > 1e-6                         # log det C differs from log det diag C
    assert abs(float(loss_full - loss_diag)) < 1e-5 * abs(float(loss_full))   # float32 decode island
    assert torch.allclose(torch.log_softmax(logits_full, -1), torch.log_softmax(logits_diag, -1),
                          rtol=0, atol=1e-11)


def test_encoded_prior_is_the_block_diagonal_embedding_of_the_diagonal_table() -> None:
    model = _model(_cfg())
    tokens = torch.tensor([[1, 3, 8]])
    belief = model.prior_bank.encode(tokens)
    assert belief.sigma.shape == (1, 3, K, K)
    assert torch.equal(belief.sigma, torch.diag_embed(torch.diagonal(belief.sigma, dim1=-2, dim2=-1)))
    assert all(isinstance(ops.block_covariance, bool) and ops.block_covariance
               for ops in model.relational_operators_by_layer)


def test_diagonal_and_full_family_paths_do_not_enable_the_block_update() -> None:
    for family, decode in (("gaussian_diagonal", "diagonal_chunked"), ("gaussian_full", "full_chunked")):
        model = _model(_cfg(family=family, decode_mode=decode))
        assert not any(ops.block_covariance for ops in model.relational_operators_by_layer)


@pytest.mark.parametrize("over,match", [
    (dict(attention_mode="projected_gradient", predictive_covariance="passthrough", attention_residual=False),
     "attention_mode"),
    (dict(predictive_readout="minimal_agent", attention_residual=False), "predictive_readout"),
    (dict(content_link_edges="lowrank"), "content_link_edges"),
    (dict(gauge_tying="shared"), "gauge_tying"),
    (dict(decode_mode="family_chunked"), "decode_mode"),
    (dict(decode_mode="expected_likelihood_chunked"), "decode_mode"),
    (dict(block_mlp_covariance="delta_full", block_coordinates="local", norm_type_block="none"),
     "block_mlp_covariance"),
    (dict(block_mlp_mode="prototype_agents", block_mlp_covariance="posterior"), "block_mlp_covariance"),
])
def test_configuration_refuses_routes_outside_the_exact_scope(over: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _cfg(**over)


def test_configuration_admits_the_supported_routes() -> None:
    for over in (dict(), dict(decode_mode="diagonal_chunked"), dict(decode_mode="linear"),
                 dict(predictive_covariance="passthrough"), dict(block_coordinates="local", decode_coordinates="local"),
                 dict(relational_routing="noisy_query"), dict(block_mlp_mode="gaussian_template")):
        assert _cfg(**over).family == "gaussian_block"
