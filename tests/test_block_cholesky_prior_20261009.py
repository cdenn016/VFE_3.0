r"""Frame-free prior table ``prior_covariance_table='block_cholesky'`` (audit 2026-10-09, gauge-theorist).

On the reference-coordinate arm every consumer reads the frame :math:`U` only through
:math:`\xi = U^{-1}\mu` and :math:`C = U^{-1}\Sigma U^{-\top}`. With learned per-type frames and a diagonal
prior table the encoded reference moments are :math:`\xi_v = U_v^{-1}\mu_v` and
:math:`C_v = U_v^{-1}\operatorname{diag}(\sigma_v)U_v^{-\top}`, one full SPD block per head. The frame-free
table stores :math:`\xi_v` (``mu_embed``) and the head blocks :math:`C_{vh} = L_{vh}L_{vh}^\top` (log-diagonal
Cholesky, ``sigma_chol_embed``) with identity frames (``gauge_transport='off'``). Checks (float64, tiny dims):
converting a learned-frame reference-arm model gives the same logits and loss; the table's parameterization;
initialization and parameter accounting; configuration scope.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
import torch

import vfe3.train as T
from tests.profiles import tiny_config
from vfe3.model.model import VFEModel
from vfe3.model.prior_bank import packed_cholesky_covariance

H, D = 2, 3
K = H * D
V = 9
DT = torch.float64


@pytest.fixture(autouse=True)
def _one_thread() -> None:
    torch.set_num_threads(1)


def _cfg(**over: Any):
    values = dict(embed_dim=K, n_heads=H, vocab_size=V, max_seq_len=5, n_layers=2, family="gaussian_block",
                  decode_mode="full_chunked", untie_decode_bank=True, block_coordinates="reference",
                  decode_coordinates="reference", norm_type_block="layernorm", layernorm_affine=True,
                  use_block_mlp=True, block_mlp_mode="coordinate", content_link_chart="exponential",
                  pos_phi="none")
    values.update(over)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tiny_config(**values)


def _build(cfg) -> VFEModel:
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VFEModel(cfg).double().eval()


def _learned_frame_model(**over: Any) -> VFEModel:
    model = _build(_cfg(**over))
    g = torch.Generator().manual_seed(21)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad:
                scale = 0.35 if name.endswith("phi_embed") else 0.2
                p.add_(scale * torch.randn(p.shape, generator=g, dtype=DT))
    return model


def _pack(C: torch.Tensor, eps: float) -> torch.Tensor:    # (..., d, d) SPD -> (..., d(d+1)/2) packed (l, N)
    r"""Inverse of the table map: :math:`C - \epsilon I = LL^\top`, :math:`L = \operatorname{diag}(e^{\ell/2})(I + N)`."""
    L = torch.linalg.cholesky(C - eps * torch.eye(D, dtype=C.dtype))
    scale = torch.diagonal(L, dim1=-2, dim2=-1)                              # (..., d) e^{l/2}
    unit = L / scale.unsqueeze(-1)                                          # I + N
    rows, cols = torch.tril_indices(D, D)
    return torch.where(rows == cols, 2.0 * torch.log(scale)[..., rows], unit[..., rows, cols])


def _convert(learned: VFEModel, **over: Any) -> VFEModel:
    r"""The frame-free model holding :math:`(\xi_v, C_v)` of ``learned`` and every other parameter unchanged."""
    free = _build(_cfg(gauge_transport="off", prior_covariance_table="block_cholesky", **over))
    pb = learned.prior_bank
    with torch.no_grad():
        types = torch.arange(V).unsqueeze(0)                                  # (1, V): every type once
        _, frame, _ = learned._relational_frames(pb.encode(types), None)
        U_inv = torch.linalg.inv(frame.forward[0].double())                    # (V, H, d, d) exact float64 inverse
        mu = pb.mu_embed.reshape(V, H, D, 1)
        xi = (U_inv @ mu).reshape(V, K)                                        # xi_v = U_v^{-1} mu_v
        sigma = torch.exp(pb.sigma_log_embed).reshape(V, H, 1, D)
        C = (U_inv * sigma) @ U_inv.transpose(-1, -2)                          # U_v^{-1} diag(sigma_v) U_v^{-T}
        C = 0.5 * (C + C.transpose(-1, -2))
        state = {k: v for k, v in learned.state_dict().items()
                 if k not in ("prior_bank.sigma_log_embed", "prior_bank.mu_embed", "prior_bank.phi_embed")}
        missing, unexpected = free.load_state_dict(state, strict=False)
        assert set(missing) == {"prior_bank.sigma_chol_embed", "prior_bank.mu_embed", "prior_bank.phi_embed"}
        assert not unexpected
        free.prior_bank.mu_embed.copy_(xi)
        # The learned arm's variances are max(e^s, eps) = e^s here; the table adds eps I, so C - eps I is packed.
        free.prior_bank.sigma_chol_embed.copy_(_pack(C, free.prior_bank.eps))
        assert bool((free.prior_bank.phi_embed == 0).all())
    return free


def _outputs(model: VFEModel, tokens: torch.Tensor, targets: torch.Tensor):
    with torch.no_grad():
        capture: dict = {}
        belief, _ = model.forward_beliefs(tokens, capture=capture)
        logits = model.prior_bank.decode(belief.mu, belief.sigma, canonical_frame=capture.get("canonical_frame"))
        loss = model(tokens, targets)[1]
    return logits, loss


@pytest.mark.parametrize("over", [
    dict(),                                                                     # gaussian_block, moment matched
    dict(decode_mode="diagonal_chunked"),
    dict(family="gaussian_full"),
    dict(predictive_covariance="passthrough"),
    dict(attention_residual=False, block_mlp_mode="gaussian_template"),
])
def test_frame_free_table_reproduces_the_learned_frame_reference_arm(over: dict) -> None:
    learned = _learned_frame_model(**over)
    free = _convert(learned, **over)
    g = torch.Generator().manual_seed(5)
    tokens, targets = torch.randint(0, V, (3, 5), generator=g), torch.randint(0, V, (3, 5), generator=g)
    logits, loss = _outputs(learned, tokens, targets)
    logits_free, loss_free = _outputs(free, tokens, targets)
    deviation = float((logits - logits_free).abs().max() / logits.abs().max())
    print(f"{over}: relative logit deviation {deviation:.3e}, loss {float(loss):.6f} vs {float(loss_free):.6f}")
    assert logits.dtype == DT and deviation < 1e-11
    assert abs(float(loss - loss_free)) < 1e-5 * abs(float(loss))           # float32 decode island
    # Not vacuous: the frames act on the learned model, and the frame-free model with the raw tables differs.
    raw = _build(_cfg(gauge_transport="off", prior_covariance_table="block_cholesky", **over))
    raw.load_state_dict(free.state_dict(), strict=True)
    with torch.no_grad():
        raw.prior_bank.mu_embed.copy_(learned.prior_bank.mu_embed)
    assert float((_outputs(raw, tokens, targets)[0] - logits).abs().max()) > 1e-3


def test_table_parameterization_initialization_and_gradients() -> None:
    model = _build(_cfg(gauge_transport="off", prior_covariance_table="block_cholesky", sigma_init=2.5))
    pb = model.prior_bank
    assert pb.sigma_log_embed is None and "prior_bank.sigma_log_embed" not in model.state_dict()
    assert pb.sigma_chol_embed.shape == (V, H, D * (D + 1) // 2)
    blocks = pb._prior_covariance_blocks()
    expected = (2.5 + pb.eps) * torch.eye(D, dtype=DT).expand(V, H, D, D)  # L = sqrt(sigma_init) I, plus eps I
    assert torch.allclose(blocks, expected, rtol=0, atol=1e-6)               # float32 init
    # The untied decode bank starts where the diagonal table would have started.
    assert torch.allclose(pb.decode_sigma_log_embed, torch.full((V, K), torch.log(torch.tensor(2.5)).item(),
                                                                dtype=DT), rtol=0, atol=1e-6)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        pb.sigma_chol_embed.normal_(0.0, 0.7, generator=g)
    C = pb._prior_covariance_blocks()
    assert bool((torch.linalg.eigvalsh(C) > 0).all())
    # Packed entries round-trip through the Cholesky factor.
    assert torch.allclose(_pack(C, pb.eps), pb.sigma_chol_embed, rtol=0, atol=1e-8)
    encoded = pb.encode(torch.tensor([[1, 4]])).sigma                        # (1, 2, K, K) head-block diagonal
    assert torch.equal(encoded[0, 1, :D, :D], C[4, 0]) and torch.equal(encoded[0, 1, D:, D:], C[4, 1])
    assert bool((encoded[0, 1, :D, D:] == 0).all())
    # Gradient check of the table map in float64.
    packed = pb.sigma_chol_embed[:2].detach().clone().requires_grad_(True)
    assert torch.autograd.gradcheck(lambda x: packed_cholesky_covariance(x, D, eps=pb.eps), (packed,),
                                    eps=1e-6, atol=1e-7)


def test_optimizer_trains_the_table_and_parameter_counts() -> None:
    cfg = _cfg(gauge_transport="off", prior_covariance_table="block_cholesky")
    model = _build(cfg).float()
    optimizer = T.build_optimizer(model, cfg)
    groups = [g for g in optimizer.param_groups if any(p is model.prior_bank.sigma_chol_embed for p in g["params"])]
    assert len(groups) == 1 and groups[0]["role"] == "sigma"
    learned = _build(_cfg())
    count = lambda m: {n: p.numel() for n, p in m.named_parameters()}
    free_counts, learned_counts = count(model), count(learned)
    assert free_counts["prior_bank.sigma_chol_embed"] == V * H * D * (D + 1) // 2
    assert learned_counts["prior_bank.sigma_log_embed"] == V * K
    assert learned_counts["prior_bank.phi_embed"] == V * H * D * D
    assert not model.prior_bank.phi_embed.requires_grad                     # identity frames, frozen


def test_default_table_is_unchanged() -> None:
    model = _build(_cfg())
    assert model.prior_bank.prior_covariance_table == "diagonal"
    assert not hasattr(model.prior_bank, "sigma_chol_embed")
    assert model.prior_bank.sigma_log_embed is not None


@pytest.mark.parametrize("over,match", [
    (dict(family="gaussian_diagonal", decode_mode="diagonal_chunked"), "family"),
    (dict(gauge_transport="on"), "gauge_transport"),
    (dict(untie_decode_bank=False), "untie_decode_bank"),
    (dict(content_link_edges="lowrank"), "content_link_edges"),
    (dict(attention_mode="projected_gradient", predictive_covariance="passthrough", attention_residual=False,
          family="gaussian_full", block_coordinates="local", decode_coordinates="local"), "attention_mode"),
    (dict(prior_covariance_table="cholesky"), "prior_covariance_table"),
])
def test_configuration_scope(over: dict, match: str) -> None:
    values = dict(gauge_transport="off", prior_covariance_table="block_cholesky")
    values.update(over)
    with pytest.raises(ValueError, match=match):
        _cfg(**values)


def test_configuration_admits_the_linear_decode_with_a_tied_bank() -> None:
    cfg = _cfg(gauge_transport="off", prior_covariance_table="block_cholesky", decode_mode="linear",
               untie_decode_bank=False)
    model = _build(cfg)
    assert model.prior_bank.output_proj_weight is not None
