"""Tiny CPU regressions for the September 22 audit remediation."""
import json
import math

import pytest
import torch

from vfe3.config import VFE3Config, serialize_config, config_from_serialized
from vfe3.families.base import safe_kl_clamp
from vfe3.families.gaussian import diag_kl_unclamped, diag_kl_unclamped_per_coord
from vfe3.geometry.groups import get_group
from vfe3.geometry.transport import compute_transport_operators, transport_mean, transport_covariance


def test_near_equal_diagonal_kl_preserves_small_positive_terms_and_gradients() -> None:
    mu = torch.zeros(2, 3, dtype=torch.float64, requires_grad=True)
    sq = torch.tensor([[1., 2., 3.], [1., 2., 3.]], dtype=torch.float64, requires_grad=True)
    sp = sq.detach() * (1 + 1e-7)
    actual = diag_kl_unclamped(mu, sq, mu.detach(), sp)
    x = (sq.detach() - sp) / sp
    expected = (x.square() * (0.5 - x / 3)).sum(-1) / 2
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-25)
    torch.testing.assert_close(actual, diag_kl_unclamped_per_coord(mu, sq, mu.detach(), sp).sum(-1))
    assert torch.autograd.gradcheck(lambda a, b: diag_kl_unclamped(a, b, mu.detach(), sp),
                                    (mu, sq), eps=1e-6, atol=1e-7)


def test_flat_dense_self_links_preserve_moments_and_have_zero_frame_derivative() -> None:
    group = get_group("glk")(3)
    phi = torch.randn(1, 2, 9, generator=torch.Generator().manual_seed(12), requires_grad=True)
    omega = compute_transport_operators(phi, group)["Omega"]
    identity = torch.eye(3).expand(1, 2, 3, 3)
    torch.testing.assert_close(omega[:, torch.arange(2), torch.arange(2)], identity, atol=0, rtol=0)
    mu = torch.randn(1, 2, 3)
    sigma = torch.rand_like(mu) + 0.5
    transported_mu = transport_mean(omega, mu)[:, torch.arange(2), torch.arange(2)]
    transported_sigma = transport_covariance(omega, sigma, diagonal_out=True)[:, torch.arange(2), torch.arange(2)]
    torch.testing.assert_close(transported_mu, mu, atol=0, rtol=0)
    torch.testing.assert_close(transported_sigma, sigma, atol=0, rtol=0)
    derivative, = torch.autograd.grad(transported_mu.sum() + transported_sigma.sum(), phi)
    assert torch.count_nonzero(derivative) == 0


def test_unbounded_kl_ceiling_keeps_gradient_and_json_round_trip() -> None:
    cfg = VFE3Config(kl_max=math.inf)
    payload = json.loads(json.dumps(serialize_config(cfg), allow_nan=False))
    restored = config_from_serialized(payload, source="test")
    assert restored.kl_max == math.inf
    score = torch.tensor([200.], requires_grad=True)
    unclipped = safe_kl_clamp(score, kl_max=restored.kl_max)
    gradient, = torch.autograd.grad(unclipped.sum(), score)
    torch.testing.assert_close(gradient, torch.ones_like(gradient))
    assert safe_kl_clamp(score, kl_max=100.).item() == 100.


@pytest.mark.parametrize("ceiling", [0., -1., -math.inf, math.nan])
def test_invalid_kl_ceiling_is_rejected(ceiling: float) -> None:
    with pytest.raises(ValueError, match="kl_max"):
        VFE3Config(kl_max=ceiling)


def test_exact_congruence_rejects_dense_cross_head_transport() -> None:
    from vfe3.families.exact_congruence import _pullback_query
    mu = torch.zeros(1, 2, 4)
    omega = torch.eye(4).expand(1, 2, 2, 4, 4).clone()
    omega[..., 0, 2] = .1
    with pytest.raises(ValueError, match="block-diagonal"):
        _pullback_query(mu, torch.ones_like(mu), omega, [2, 2])


def test_adamw_reports_nondefault_group_trust_radius() -> None:
    with pytest.warns(UserWarning, match="m_phi_group_trust_radius"):
        VFE3Config(m_phi_update_mode="adamw", m_phi_group_trust_radius=.2)


def test_scaling_rejects_inert_mup_before_creating_model(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scaling
    baseline = dict(scaling.BASELINE, e_step_update="mm_exact", relational_mode="off",
                    relational_readout_mode="conditional_update", relational_output_scope="head",
                    attention_residual=False, eval_stride=None,
                    embed_dim=4, n_heads=2, vocab_size=7, max_seq_len=3, n_layers=1)
    monkeypatch.setattr(scaling, "BASELINE", baseline)
    cell = {"label": "inert", "route": "grow_K_mup", "scale_knob": "embed_dim", "overrides": {}}
    result = scaling.run_cell(cell, tmp_path / "cell", 1, dataset="unused", device=torch.device("cpu"))
    assert result["error_kind"] == "config"
    assert "grow_K_mup varies e_q_mu_lr" in result["error"]
    assert not (tmp_path / "cell").exists()


@pytest.mark.parametrize("mode", ["diagonal_chunked", "full_chunked"])
def test_fp64_decode_policy_preserves_completed_logits(mode: str) -> None:
    from tests.test_priorbank_head_evidence import _reference_bank, _full_query, _diagonal_query
    bank = _reference_bank(mode, (2, 3), enabled=True, unigram=True).float()
    bank.decode_av_precision = "fp64"
    mu, sigma = _full_query() if mode == "full_chunked" else _diagonal_query()
    logits = bank.decode(mu.float(), sigma.float())
    assert logits.dtype == torch.float64
    assert torch.isfinite(logits).all()
