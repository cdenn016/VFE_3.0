r"""Remediation of the 2026-10-10 deep audit (docs/audits/audit-2026-10-10.md): configuration, decode and
geometry items owned by the coordinator. Tiny CPU models only.

Items: 2 (decode_mu_weight_decay), 3 (float64 decode island), 4 (windowed inactive reason), 13 (weight_decay
and phi_weight_decay inert-field rules), 41 (exact natural parameter), 43 (float64 so_n generators),
61 (deferred warning-only and raise-only host checks).
"""

import math
import warnings

import pytest
import torch

from tests.profiles import test_config
from vfe3.contracts import deferred_host_checks
from vfe3.model.model import VFEModel
from vfe3.train import build_optimizer

BASE = dict(vocab_size=11, embed_dim=4, n_heads=2, max_seq_len=6, n_layers=1)


def _group_of(opt: torch.optim.Optimizer, param: torch.Tensor) -> dict:
    groups = [g for g in opt.param_groups if any(p is param for p in g["params"])]
    assert len(groups) == 1
    return groups[0]


def _wd(opt: torch.optim.Optimizer, param: torch.Tensor) -> float:
    group = _group_of(opt, param)
    return group.get("weight_decay", opt.defaults["weight_decay"])


# ------------------------------------------------------------------------------------- item 2

@pytest.mark.parametrize("decode_mode", ["diagonal_chunked", "linear"])
def test_decode_mu_weight_decay_none_inherits_mu_weight_decay(decode_mode: str) -> None:
    cfg = test_config(**BASE, decode_mode=decode_mode, untie_decode_bank=True,
                      weight_decay=0.05, mu_weight_decay=0.02, decode_mu_weight_decay=None)
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    pb = model.prior_bank
    decode_mean = pb.decode_mu_embed if decode_mode != "linear" else pb.output_proj_weight
    assert _wd(opt, decode_mean) == 0.02
    assert _wd(opt, pb.mu_embed) == 0.02


@pytest.mark.parametrize("decode_mode", ["diagonal_chunked", "linear"])
def test_decode_mu_weight_decay_reaches_only_the_decode_means(decode_mode: str) -> None:
    cfg = test_config(**BASE, decode_mode=decode_mode, untie_decode_bank=True,
                      weight_decay=0.05, mu_weight_decay=0.0, sigma_weight_decay=0.0,
                      decode_mu_weight_decay=0.01)
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    pb = model.prior_bank
    decode_mean = pb.decode_mu_embed if decode_mode != "linear" else pb.output_proj_weight
    assert _wd(opt, decode_mean) == 0.01
    assert _wd(opt, pb.mu_embed) == 0.0
    if decode_mode != "linear":
        assert _wd(opt, pb.decode_sigma_log_embed) == 0.0
    else:
        assert _wd(opt, pb.output_proj_bias) == 0.0
    assert "decode_mu_weight_decay" not in cfg.inactive_fields()


def test_decode_mu_weight_decay_pulls_an_untargeted_decode_row_back() -> None:
    r"""A row that only receives the softmax push walks off without a prior and is pulled back with one."""
    norms = {}
    for wd in (0.0, 0.5):
        torch.manual_seed(0)
        cfg = test_config(**BASE, decode_mode="diagonal_chunked", untie_decode_bank=True,
                          mu_weight_decay=0.0, decode_mu_weight_decay=wd)
        model = VFEModel(cfg)
        opt = build_optimizer(model, cfg)
        table = model.prior_bank.decode_mu_embed
        with torch.no_grad():
            table[7] = 3.0
        for _ in range(20):
            table.grad = torch.zeros_like(table)
            table.grad[7] = -1e-3                    # a small, persistent outward push on row 7
            opt.step()
        norms[wd] = float(table[7].norm())
    assert norms[0.5] < norms[0.0]


def test_decode_mu_weight_decay_is_inert_on_a_tied_bank_and_validated() -> None:
    cfg = test_config(**BASE, decode_mode="diagonal_chunked", untie_decode_bank=False,
                      prior_covariance_table="diagonal", decode_mu_weight_decay=0.01)
    assert "decode_mu_weight_decay" in cfg.inactive_fields()
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="decode_mu_weight_decay"):
            test_config(**BASE, decode_mu_weight_decay=bad)


# ------------------------------------------------------------------------------------- item 13

def test_weight_decay_is_inert_when_every_group_sets_its_own_decay() -> None:
    cfg = test_config(**BASE, use_block_mlp=False, mu_weight_decay=0.0, sigma_weight_decay=0.0,
                      decode_mu_weight_decay=None, weight_decay=0.3)
    assert "weight_decay" in cfg.inactive_fields()
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    assert all("weight_decay" in group for group in opt.param_groups)


def test_weight_decay_is_live_with_a_block_mlp() -> None:
    cfg = test_config(**BASE, use_block_mlp=True, mu_weight_decay=0.0, sigma_weight_decay=0.0)
    assert "weight_decay" not in cfg.inactive_fields()


@pytest.mark.parametrize("mode", ["pullback_group", "fisher_group"])
def test_phi_weight_decay_is_inert_under_manifold_phi_updates(mode: str) -> None:
    cfg = test_config(**BASE, m_phi_update_mode=mode, phi_weight_decay=0.2)
    assert "phi_weight_decay" in cfg.inactive_fields()
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    assert _group_of(opt, model.prior_bank.phi_embed)["weight_decay"] == 0.0


def test_phi_weight_decay_is_live_under_adamw() -> None:
    cfg = test_config(**BASE, m_phi_update_mode="adamw", phi_weight_decay=0.2)
    assert "phi_weight_decay" not in cfg.inactive_fields()


# ------------------------------------------------------------------------------------- item 4

def test_causal_windowed_inactive_reason_names_plain_causal() -> None:
    cfg = test_config(**BASE, beta_attention_prior="causal_windowed", attention_window=64)
    assert "plain causal" in cfg.inactive_fields()["attention_window"]


# ------------------------------------------------------------------------------------- item 3

@pytest.mark.parametrize("decode_mode", ["diagonal_chunked", "linear"])
def test_float64_model_decodes_in_float64(decode_mode: str) -> None:
    torch.manual_seed(1)
    cfg = test_config(**BASE, decode_mode=decode_mode, untie_decode_bank=True, amp_dtype=None)
    model = VFEModel(cfg).double().eval()
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    with torch.no_grad():
        belief, logits = model.forward_beliefs(tokens, return_logits=True)
        direct = model.prior_bank.decode(belief.mu, belief.sigma)
    assert logits.dtype is torch.float64
    torch.testing.assert_close(logits, direct, rtol=1e-12, atol=1e-12)


def test_float64_model_loss_matches_finite_differences() -> None:
    r"""The fused training CE stays in float64, so a central difference at h = 1e-6 resolves the gradient."""
    torch.manual_seed(2)
    cfg = test_config(**BASE, decode_mode="diagonal_chunked", untie_decode_bank=True, amp_dtype=None)
    model = VFEModel(cfg).double()
    tokens = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    targets = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    param = model.prior_bank.decode_mu_embed

    def loss() -> torch.Tensor:
        return model(tokens, targets)[1]                           # (logits, loss, ce)

    loss().backward()
    grad = param.grad[targets[0, 0]].clone()
    h = 1e-6
    index = (int(targets[0, 0]), 0)
    with torch.no_grad():
        param[index] += h
        up = loss().item()
        param[index] -= 2 * h
        down = loss().item()
        param[index] += h
    fd = (up - down) / (2 * h)
    assert math.isclose(fd, float(grad[0]), rel_tol=1e-6, abs_tol=1e-9), (fd, float(grad[0]))


# ------------------------------------------------------------------------------------- item 41

def test_full_gaussian_generic_kl_oracle_is_exact_on_an_ill_conditioned_covariance() -> None:
    from vfe3.families.base import _renyi_from_log_partition
    from vfe3.families.gaussian import FullGaussian

    torch.manual_seed(3)
    K = 6
    Q, _ = torch.linalg.qr(torch.randn(K, K, dtype=torch.float64))
    S = Q @ torch.diag(torch.tensor([1e-4, 0.1, 0.5, 1.0, 2.0, 3.0], dtype=torch.float64)) @ Q.T
    q = FullGaussian(torch.randn(1, K, dtype=torch.float64), S[None])
    p = FullGaussian(torch.zeros(1, K, dtype=torch.float64), 1.5 * torch.eye(K, dtype=torch.float64)[None])
    closed = q.renyi_closed_form(p, alpha=1.0, kl_max=float("inf"), eps=1e-6)
    generic = _renyi_from_log_partition(q, p, alpha=1.0, kl_max=float("inf"), eps=1e-6)
    torch.testing.assert_close(generic, closed, rtol=1e-8, atol=0.0)


# ------------------------------------------------------------------------------------- item 43

def test_so_n_float64_model_carries_the_float64_irrep_basis() -> None:
    cfg = test_config(vocab_size=11, embed_dim=8, n_heads=2, max_seq_len=6, n_layers=1,
                      gauge_group="so_n", group_n=3, irrep_spec=[("l1", 1), ("l2", 1)], gauge_tying=None)
    model = VFEModel(cfg)
    assert model.group.generators.dtype is torch.float32
    model = model.double()
    assert model.group.generators.dtype is torch.float64
    torch.testing.assert_close(model.group.generators, model.group.float64_generators, rtol=0, atol=0)
    # the float32 copy differs from the master only by rounding
    assert float((model.group.float64_generators.float().double() - model.group.float64_generators).abs().max()) > 0
    model = model.float()
    assert model.group.generators.dtype is torch.float32


# ------------------------------------------------------------------------------------- item 61

def test_max_log_warning_rides_the_deferred_host_read() -> None:
    from vfe3 import numerics

    table = torch.nn.Parameter(torch.zeros(4, 3))
    with torch.no_grad():
        table[0, 0] = 99.0
    with deferred_host_checks():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = numerics.bounded_variance_from_log(table)
            assert not caught, "the verdict must wait for the scope's host read"
            numerics.bounded_variance_from_log(table)              # pending: no second queue entry
        assert sum("max_log" in str(w.message) for w in caught) == 0
    torch.testing.assert_close(out[0, 0], torch.exp(torch.tensor(80.0)))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        numerics.bounded_variance_from_log(table)                  # resolved and cached: warns from the cache
    assert any("max_log" in str(w.message) for w in caught)


def test_compact_chart_check_raises_at_the_deferred_read() -> None:
    from vfe3.contracts import host_read
    from vfe3.geometry import transport

    blocks = torch.full((1, 2, 2, 2), float("nan"))
    with pytest.raises(ValueError, match="nonfinite matrix norm"):
        with deferred_host_checks():
            try:
                transport._stable_compact_glk_exp_pair(blocks, validity_max_norm=10.0)
            except ValueError:
                raise AssertionError("the chart verdict must not raise before the host read")
            except Exception:
                pass
            host_read(None)


# ------------------------------------------------------------------------------------- item 51

def test_finalize_wall_time_includes_the_resumed_segment() -> None:
    from types import SimpleNamespace

    from vfe3.run_artifacts import _cumulative_wall_time

    assert _cumulative_wall_time(SimpleNamespace(resumed_wall_clock_s=None), 12.5) == (12.5, None)
    assert _cumulative_wall_time(SimpleNamespace(resumed_wall_clock_s=100.0), 12.5) == (112.5, 100.0)
    assert _cumulative_wall_time(SimpleNamespace(resumed_wall_clock_s=100.0), None) == (None, 100.0)


# ------------------------------------------------------------------------------------- items 5 and 34

def _probe_model(**overrides: object) -> VFEModel:
    torch.manual_seed(4)
    cfg = test_config(**{**BASE, "n_layers": 2, "attention_mode": "projected_predictive", **overrides})
    return VFEModel(cfg)


@pytest.mark.parametrize("rope_mode", ["comparison", "frame"])
def test_beta_channel_decomposition_separates_rope_position(rope_mode: str) -> None:
    from vfe3.run_artifacts import collect_beta_channel_decomposition

    model = _probe_model(rope_mode=rope_mode)
    bases = [ops.comparison_rope_base for ops in model.relational_operators_by_layer]
    tokens, targets = torch.tensor([[0, 1, 2, 3, 4, 5]]), torch.tensor([[1, 2, 3, 4, 5, 6]])
    channel = collect_beta_channel_decomposition(model, tokens, targets)
    assert channel["available"] and channel["energy_includes_rope_position"] is True
    assert math.isfinite(channel["delta_ce"]["rope_position_channel"])
    assert channel["ce"]["no_rope"] != channel["ce"]["full"]
    assert [ops.comparison_rope_base for ops in model.relational_operators_by_layer] == bases
    assert "_rope_rotation" not in model.__dict__


def test_beta_channel_decomposition_without_rope_has_no_rope_arm() -> None:
    from vfe3.run_artifacts import collect_beta_channel_decomposition

    model = _probe_model(rope_mode="none")
    tokens, targets = torch.tensor([[0, 1, 2, 3]]), torch.tensor([[1, 2, 3, 4]])
    channel = collect_beta_channel_decomposition(model, tokens, targets)
    assert channel["energy_includes_rope_position"] is False
    assert "rope_position_channel" not in channel["delta_ce"] and "no_rope" not in channel["ce"]


def test_context_sensitivity_reads_the_final_belief() -> None:
    from vfe3.run_artifacts import collect_context_sensitivity

    model = _probe_model(rope_mode="none")
    tokens = torch.tensor([[0, 1, 2, 3, 4, 5]])
    context = collect_context_sensitivity(model, tokens, n_replicates=2)
    stages = context["stages"]
    assert set(stages) == {"raw_prior", "predictive_content", "layer0_readout"}
    assert stages["raw_prior"] == 0.0
    assert stages["predictive_content"] > 0.0 and stages["layer0_readout"] > 0.0
    assert stages["predictive_content"] != stages["layer0_readout"]


# ------------------------------------------------------------------------------------- float64 source recovery

def test_float64_source_recovery_backpropagates() -> None:
    r"""A failed float64 Gram rebuilt from its source factor must not be written in place into the
    factor autograd saved (``.double()`` is the identity on float64)."""
    from tests.test_relational_source_factorization import _inputs
    from vfe3.relational_scoring import projected_gaussian_kl, projected_moments

    mu, sigma, inverse, S = (t.double() for t in _inputs("cpu"))
    S = S.requires_grad_(True)
    identity = torch.eye(3, dtype=torch.float64)[None]
    aq, Gq = projected_moments(mu, sigma, inverse, S)
    ak, Gk = projected_moments(mu, sigma, inverse, identity)
    evals, evecs = torch.linalg.eigh(Gq.detach())
    v = evecs[..., :, 0]
    indefinite = (Gq - (2.0 * evals[..., 0])[..., None, None] * v[..., :, None] * v[..., None, :]
                  - 1e-12 * torch.eye(3, dtype=torch.float64))
    assert float(torch.linalg.eigvalsh(indefinite.detach()).min()) < 0.0
    score = projected_gaussian_kl(aq, indefinite, ak, Gk, projection_q=(sigma, inverse, S),
                                  projection_k=(sigma, inverse, identity))
    (grad,) = torch.autograd.grad(score.sum(), S)
    assert torch.isfinite(score).all() and torch.isfinite(grad).all()
