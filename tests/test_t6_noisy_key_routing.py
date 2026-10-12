r"""Noisy-key kernel routing at temperature 1 (audit 2026-10-12, T6).

``relational_routing='noisy_key'`` scores :math:`D_{ij} = \mathrm{KL}(N(a_i, V_{Q,i})\,\|\,N(b_j, V_{K,j} + n_hI))` with a
learned per-head noise :math:`n_h = e^{\ell_h}` (``routing_key_log_noise``) and routes at temperature exactly 1, so the
routing objective is the KL of the declared joint :math:`\pi_{ij}N(y; b_j, V_{K,j} + n_hI)`. Pinned on tiny float64
CPU cases: the cost against the closed-form KL, the :math:`n_h\to0` limit (the Gaussian-KL router at temperature 1),
the unit temperature, the gradient to :math:`\ell_h`, causality, the common reference change, the config contract and
checkpoint compatibility.
"""
import math
import warnings

import pytest
import torch

from tests.profiles import make_relational_operators
from tests.test_audit_20261012_floor import (
    FLOOR, D, H, N, _change_reference, _inputs, _reference_change, _relative,
)
from vfe3.contracts import CanonicalFrameContext
from vfe3.gradients.relational import relational_predictive_readout
from vfe3.relational_scoring import projected_gaussian_kl, projected_noisy_key_cost

F64 = torch.float64


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _moments(seed: int = 3, n: int = 4, heads: int = 2, r: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """Random projected means (1, N, H, r) and exactly symmetric SPD covariances (1, N, H, r, r)."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(1, n, heads, r, generator=g, dtype=F64)
    root = torch.randn(1, n, heads, r, r, generator=g, dtype=F64)
    G = root @ root.transpose(-1, -2) + .3 * torch.eye(r, dtype=F64)
    return a, .5 * (G + G.transpose(-1, -2))


def _operators(routing: str, seed: int = 0, **extra) -> torch.nn.Module:
    ops = make_relational_operators([D] * H, 11, readout_mode="predictive_content", attention_residual=False,
                                    predictive_covariance="moment_matched", predictive_covariance_floor=FLOOR,
                                    routing=routing, **extra).double()
    g = torch.Generator().manual_seed(100 + seed)
    with torch.no_grad():
        for name, p in ops.named_parameters():
            if name != "routing_key_log_noise":
                p.add_(0.3 * torch.randn(p.shape, generator=g, dtype=F64))
    return ops


def _readout(ops, U, mu, sigma, prior, log_prior, tau: float = 0.9, capture: dict | None = None):
    frame = CanonicalFrameContext(U, torch.linalg.inv(U), block_diagonal=True)
    return relational_predictive_readout(mu, sigma, frame, ops, tau=tau, log_prior=log_prior,
                                         return_covariance=True, floor_anchor=prior, capture=capture)


def test_cost_is_the_kl_against_the_broadened_key_law() -> None:
    a, G = _moments()
    noise = torch.tensor([.4, 2.5], dtype=F64)
    D_ij = projected_noisy_key_cost(a, G, a, G, key_noise=noise)
    assert D_ij.shape == (1, 2, 4, 4)
    for h in range(2):
        for i in range(4):
            for j in range(4):
                p = torch.distributions.MultivariateNormal(a[0, i, h], G[0, i, h])
                q = torch.distributions.MultivariateNormal(a[0, j, h], G[0, j, h] + noise[h] * torch.eye(3, dtype=F64))
                torch.testing.assert_close(D_ij[0, h, i, j], torch.distributions.kl_divergence(p, q),
                                           rtol=1e-10, atol=1e-12)
    with pytest.raises(TypeError, match="key_noise"):
        projected_noisy_key_cost(a, G, a, G)                     # never the noise-free KL by omission
    with pytest.raises(ValueError, match="one entry per head"):
        projected_noisy_key_cost(a, G, a, G, key_noise=noise[:1])


def test_temperature_is_exactly_one() -> None:
    ops = _operators("noisy_key")
    assert ops.routing_temperature(0.9, F64) == 1.0
    assert ops.routing_temperature(torch.tensor([3.0, 5.0], dtype=F64), F64) == 1.0
    U, mu, sigma, prior, log_prior = _inputs("full")
    one, _ = _readout(ops, U, mu, sigma, prior, log_prior, tau=0.9)
    other, _ = _readout(ops, U, mu, sigma, prior, log_prior, tau=7.0)
    assert torch.equal(one, other)
    with pytest.raises(ValueError, match="temperature 1"):
        make_relational_operators([D] * H, 11, readout_mode="predictive_content", routing="noisy_key",
                                  learn_attention_temperature=True)


@pytest.mark.parametrize("family", ["diagonal", "full"])
def test_vanishing_noise_is_the_gaussian_kl_router_at_unit_temperature(family: str) -> None:
    U, mu, sigma, prior, log_prior = _inputs(family)
    kl_ops = _operators("gaussian_kl")
    noisy_ops = _operators("noisy_key")
    missing = noisy_ops.load_state_dict(kl_ops.state_dict(), strict=False)
    assert missing.missing_keys == ["routing_key_log_noise"] and not missing.unexpected_keys
    with torch.no_grad():
        noisy_ops.routing_key_log_noise.fill_(math.log(1e-13))
    cap_kl, cap_noisy = {}, {}
    mean_kl, cov_kl = _readout(kl_ops, U, mu, sigma, prior, log_prior, tau=1.0, capture=cap_kl)
    mean_n, cov_n = _readout(noisy_ops, U, mu, sigma, prior, log_prior, tau=0.37, capture=cap_noisy)
    torch.testing.assert_close(cap_noisy["beta_pre"], cap_kl["beta_pre"], rtol=0, atol=1e-9)
    torch.testing.assert_close(mean_n, mean_kl, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(cov_n, cov_kl, rtol=1e-9, atol=1e-10)
    # The opposite limit: every pair cost tends to the same r log(n) / 2 up to O(1/n), so the routing tends to the
    # prior (uniform on the causal support).
    with torch.no_grad():
        noisy_ops.routing_key_log_noise.fill_(math.log(1e12))
    _readout(noisy_ops, U, mu, sigma, prior, log_prior, capture=cap_noisy)
    prior_rows = log_prior.softmax(-1).expand_as(cap_noisy["beta_pre"])
    torch.testing.assert_close(cap_noisy["beta_pre"], prior_rows, rtol=0, atol=1e-6)


def test_gradient_reaches_the_log_noise_and_matches_finite_differences() -> None:
    U, mu, sigma, prior, log_prior = _inputs("full")
    ops = _operators("noisy_key")
    mean, cov = _readout(ops, U, mu, sigma, prior, log_prior)
    (mean.square().sum() + cov.sum()).backward()
    grad = ops.routing_key_log_noise.grad
    assert grad is not None and torch.isfinite(grad).all() and float(grad.abs().min()) > 1e-8

    a, G = _moments(seed=5)

    def cost(log_noise: torch.Tensor) -> torch.Tensor:
        return projected_noisy_key_cost(a, G, a, G, key_noise=log_noise.exp())

    assert torch.autograd.gradcheck(cost, (torch.tensor([-.3, .8], dtype=F64, requires_grad=True),),
                                    eps=1e-6, atol=1e-6, rtol=1e-5)


def test_routing_is_causal() -> None:
    U, mu, sigma, prior, log_prior = _inputs("full")
    ops = _operators("noisy_key")
    mean, cov = _readout(ops, U, mu, sigma, prior, log_prior)
    mu2, sigma2, U2 = mu.clone(), sigma.clone(), U.clone()
    mu2[:, -1] += 5.0
    sigma2[:, -1] = 3.0 * sigma2[:, -1]
    U2[:, -1] = U2[:, -1] @ torch.tensor([[1.5, .2], [-.1, .8]], dtype=F64)
    mean2, cov2 = _readout(ops, U2, mu2, sigma2, prior, log_prior)
    assert torch.equal(mean2[:, :-1], mean[:, :-1]) and torch.equal(cov2[:, :-1], cov[:, :-1])
    assert not torch.allclose(mean2[:, -1], mean[:, -1])


@pytest.mark.parametrize("family", ["diagonal", "full"])
def test_common_reference_change_is_exact(family: str) -> None:
    U, mu, sigma, prior, log_prior = _inputs(family)
    ops = _operators("noisy_key")
    with torch.no_grad():
        ops.routing_key_log_noise.copy_(torch.tensor([-.5, .7], dtype=F64))
    cap, cap_a = {}, {}
    mean, cov = _readout(ops, U, mu, sigma, prior, log_prior, capture=cap)
    a = _reference_change()
    _change_reference(ops, a)
    mean_a, cov_a = _readout(ops, U @ a, mu, sigma, prior, log_prior, capture=cap_a)
    assert _relative(cap_a["beta_pre"], cap["beta_pre"]) < 1e-12
    assert _relative(mean_a, mean) < 1e-12 and _relative(cov_a, cov) < 1e-12


def test_config_contract_and_inactive_fields() -> None:
    from dataclasses import asdict

    from tests.profiles import tiny_config
    from tests.test_relational_integration import _cfg
    from vfe3.config import config_from_serialized, serialize_config

    predictive = dict(attention_mode="projected_predictive")
    default = tiny_config()
    assert "relational_key_noise" in default.inactive_fields()
    assert "kappa_beta" not in _cfg(**predictive).inactive_fields()
    cfg = _cfg(**predictive, relational_routing="noisy_key", relational_key_noise=0.25)
    inactive = cfg.inactive_fields()
    assert "relational_key_noise" not in inactive and "kappa_beta" in inactive
    assert "pair_kl_gemm_tol" not in inactive           # the factor-route selector is merged into projected_factorization (C4)
    assert asdict(config_from_serialized(serialize_config(cfg), source="noisy key")) == asdict(cfg)
    for bad in (0.0, -1.0, float("nan"), True):
        with pytest.raises(ValueError, match="relational_key_noise"):
            _cfg(**predictive, relational_routing="noisy_key", relational_key_noise=bad)
    with pytest.raises(ValueError, match="noisy_key"):
        _cfg(**predictive, relational_routing="noisy_key", learn_attention_temperature=True)
    with pytest.raises(ValueError, match="requires attention_mode='projected_predictive'"):
        _cfg(relational_routing="noisy_key")                      # _cfg defaults to projected_gradient


def test_model_trains_groups_the_noise_and_leaves_existing_checkpoints_unchanged() -> None:
    from tests.test_relational_integration import _cfg, _tokens
    from vfe3.model.model import VFEModel
    from vfe3.run_artifacts import _relational_artifact_fields
    from vfe3.train import build_optimizer

    predictive = dict(attention_mode="projected_predictive", attention_residual=True)
    torch.manual_seed(23)
    kl_model = VFEModel(_cfg(**predictive))
    assert not any("routing_key_log_noise" in name for name in kl_model.state_dict())
    cfg = _cfg(**predictive, relational_routing="noisy_key", relational_key_noise=0.5)
    noisy_model = VFEModel(cfg)
    result = noisy_model.load_state_dict(kl_model.state_dict(), strict=False)
    assert len(result.missing_keys) == cfg.n_layers and all(
        key.endswith("routing_key_log_noise") for key in result.missing_keys)
    assert not result.unexpected_keys
    noise = [p for n, p in noisy_model.named_parameters() if n.endswith("routing_key_log_noise")]
    assert noise and all(torch.allclose(p, torch.full_like(p, math.log(0.5))) for p in noise)
    optimizer = build_optimizer(noisy_model, cfg)
    score_group = [group for group in optimizer.param_groups if group.get("lr_aux_role") == "relational_score"]
    assert score_group and all(any(p is q for q in score_group[0]["params"]) for p in noise)
    ids, targets = _tokens()
    _, logits = noisy_model.forward_beliefs(ids, return_logits=True)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 7), targets.reshape(-1))
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in noise)
    record = _relational_artifact_fields(cfg)["relational_operators"]
    baseline = _relational_artifact_fields(_cfg(**predictive))["relational_operators"]
    assert record["routing"] == "noisy_key" and record["relational_key_noise_init"] == 0.5
    assert record["temperature_convention"] == "unit_temperature_learned_key_noise"
    H_heads = len(noisy_model.group.irrep_dims)
    assert record["total_parameters"] == baseline["total_parameters"] + cfg.n_layers * H_heads
    assert sum(p.numel() for p in noisy_model.relational_operators.parameters()) == record["total_parameters"]
