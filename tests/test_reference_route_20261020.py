r"""The reference route pinned at tiny scale, independent of ``train_vfe3.py`` (audit 2026-10-20 item 23).

``tests.profiles.reference_config`` freezes the route switches of the 30.37 run: five heads (the H = 5 ALiBi slopes),
``query_shifted``, ``predictive_covariance_floor_mode='max'``, ``predictive_covariance_spread_squares`` with backward,
comparison-invariant RoPE on the diagonal comparison route, the diagonal family, the exponential content chart and
Fisher table steps. Tiny CPU models only (K = 20, five heads of 4, two layers, N = 4, vocab 7).
"""
import math
from typing import Any, Dict, Tuple

import pytest
import torch

from tests.profiles import _REFERENCE_ROUTE, reference_config
from tests.test_live_route_profile import _assert_router_gradient_nonzero, _richardson_mismatches
from vfe3.attention_prior import _press_slopes
from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.posthoc.klcap import causal_support


def _batch(cfg: VFE3Config, seed: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    shape = (cfg.batch_size, cfg.max_seq_len)
    return (torch.randint(0, cfg.vocab_size, shape, generator=g),
            torch.randint(0, cfg.vocab_size, shape, generator=g))


def _float64_model(**overrides: Any) -> VFEModel:
    r"""The reference route in float64 with untied block-norm gains and a random unigram table (as the live check)."""
    torch.manual_seed(0)
    model = VFEModel(reference_config(**overrides)).double()
    with torch.no_grad():
        generator = torch.Generator().manual_seed(5)
        for name, p in model.named_parameters():
            if name.endswith("norm.weight"):
                p.add_(0.01 * torch.randn(p.shape, dtype=p.dtype, generator=generator))
    bank = model.prior_bank
    if bank.decode_unigram_prior:
        counts = torch.randint(0, 50, (bank.vocab_size,), generator=torch.Generator().manual_seed(6))
        bank.set_unigram_log_prior(counts.double())
    return model


def test_reference_profile_holds_the_reference_switches() -> None:
    cfg = reference_config()
    expected: Dict[str, Any] = {
        "n_heads": 5, "routing_kl_max_reference": "query_shifted", "predictive_covariance_floor_mode": "max",
        "predictive_covariance_spread_squares": True, "rope_mode": "comparison_invariant",
        "family": "gaussian_diagonal", "content_link_chart": "exponential", "m_table_update_mode": "fisher",
        "beta_attention_prior": "causal_alibi", "predictive_readout": "kernel", "relational_routing": "gaussian_kl",
        "attention_mode": "projected_predictive", "predictive_covariance": "moment_matched",
    }
    assert {name: getattr(cfg, name) for name in expected} == expected
    assert all(getattr(cfg, name) == value for name, value in _REFERENCE_ROUTE.items())
    # 4 nats per comparison coordinate, as the reference's 128 at d = 32
    assert cfg.kl_max / (cfg.embed_dim // cfg.n_heads) == 128 / 32


def test_press_slopes_at_five_heads_take_the_non_power_of_two_branch() -> None:
    slopes = _press_slopes(5, 1.0, "cpu", torch.float64)
    four = [2.0 ** (-8.0 * h / 4) for h in range(1, 5)]                         # the lower power-of-two schedule
    eight = [2.0 ** (-8.0 * h / 8) for h in range(1, 9)]
    expected = four + eight[::2][:1]                                            # plus the first even entry of 8
    assert slopes.tolist() == pytest.approx(expected, rel=1e-15)
    assert len(set(slopes.tolist())) == 5


def test_reference_route_runs_the_diagonal_comparison_route_with_five_alibi_heads() -> None:
    torch.manual_seed(0)
    model = VFEModel(reference_config()).eval()
    tokens, _ = _batch(model.cfg)
    snapshot = model.build_diagnostic_snapshot(tokens)
    for layer in range(model.cfg.n_layers):
        cap = model._layer_capture(snapshot, layer)
        assert cap["score_term_medians_pre"].shape == (4, 6)                    # the diagonal route, five heads
        assert "score_row_shift_pre" in cap                                     # the shifted reference ran
        support = causal_support(cap["beta_pre"], snapshot.log_prior)
        capped = (cap["score_energy_pre"] >= model.cfg.kl_max) & support
        assert float(capped.sum()) < float(support.sum())                      # the tiny cap leaves the router live
    prior = snapshot.log_prior                                                  # (H, N, N) ALiBi biases
    assert prior is not None and prior.shape[-3] == 5
    distances = torch.arange(model.cfg.max_seq_len, dtype=prior.dtype)
    slopes = -(prior[..., -1, 0] - prior[..., -1, -1]) / distances[-1]          # B_ij = -slope (i - j)
    torch.testing.assert_close(slopes.double(), _press_slopes(5, model.cfg.alibi_slope, "cpu", torch.float64),
                               rtol=1e-6, atol=0)


def test_reference_backward_matches_float64_richardson_differences() -> None:
    r"""Every trainable gradient of the reference route, spread squares and query_shifted included, matches a float64
    Richardson-extrapolated central difference, and every comparison map receives a nonzero gradient."""
    model = _float64_model()
    assert model.cfg.predictive_covariance_spread_squares
    tokens, targets = _batch(model.cfg)

    def loss() -> torch.Tensor:
        return model(tokens, targets)[1]

    loss().backward()
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    assert trainable and [name for name, p in trainable.items() if p.grad is None] == []
    _assert_router_gradient_nonzero(model)
    assert _richardson_mismatches(loss, trainable) == {}


def test_reference_route_trains_two_fisher_steps() -> None:
    from vfe3.train import train

    cfg = reference_config()
    torch.manual_seed(0)
    model = VFEModel(cfg)
    gen = torch.Generator().manual_seed(1)
    shape = (cfg.batch_size, cfg.max_seq_len)
    batches = [(torch.randint(0, cfg.vocab_size, shape, generator=gen),
                torch.randint(0, cfg.vocab_size, shape, generator=gen)) for _ in range(2)]
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    losses = train(model, batches, cfg, n_steps=2, grad_clip=cfg.grad_clip, log_interval=0, eval_interval=0,
                   generate_samples=False)
    assert len(losses) == 2 and all(math.isfinite(value) for value in losses)
    after = dict(model.named_parameters())
    assert all(bool(torch.isfinite(p).all()) for p in after.values())
    # the Fisher-stepped tables and the comparison maps moved
    for name in ("prior_bank.mu_embed", "prior_bank.sigma_log_embed"):
        assert not torch.equal(after[name].detach(), before[name]), name
    assert any(not torch.equal(p.detach(), before[name]) for name, p in after.items() if "raw_score" in name)
