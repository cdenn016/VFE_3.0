r"""Opt-in learnable routing temperature and comparison-map tie (audit 2026-10-09, info-geometer).

A. ``learn_attention_temperature``: per layer and head, the Gibbs routing temperature is
:math:`\tau_{\ell,h} = \kappa_h\sqrt{d_h}\exp(\mathrm{clamp}(\delta_{\ell,h}, -3, 3))`, i.e. the
log-temperature :math:`s_{\ell,h} = \log\kappa_h + \delta_{\ell,h}` with :math:`\delta = 0` at
initialization, so the configured temperature is reproduced exactly at step 0.

B. ``tie_comparison_maps``: :math:`S_K := S_Q` on the projected_predictive kernel route, so the
self-pair cost :math:`D_{ii} = \mathrm{KL}(q_i \| q_i) = 0` and, with identity frames,
:math:`D_{ij} = \mathrm{KL}(S_\# q_i \| S_\# q_j)`.
"""
import math

import pytest
import torch

from tests.profiles import make_relational_operators, tiny_config
from vfe3.contracts import CanonicalFrameContext
from vfe3.free_energy import attention_tau
from vfe3.gradients.relational import relational_predictive_readout, relational_score
from vfe3.model.model import VFEModel


def _fixture(n: int = 4, seed: int = 3) -> tuple:
    """Tiny fp64 beliefs (B=1, N=n, K=4) with a non-identity head-block frame and a causal prior."""
    g = torch.Generator().manual_seed(seed)
    mu = torch.randn(1, n, 4, generator=g, dtype=torch.float64)
    sigma = 0.5 + torch.rand(1, n, 4, generator=g, dtype=torch.float64)
    U = torch.eye(4, dtype=torch.float64).expand(1, n, 4, 4).clone()
    U[..., 0, 1] = 0.3
    U[..., 2, 3] = -0.2
    frame = CanonicalFrameContext(U, torch.linalg.inv(U))
    prior = torch.zeros(n, n, dtype=torch.float64).masked_fill(
        torch.ones(n, n, dtype=torch.bool).triu(1), -math.inf)
    return mu, sigma, frame, prior


def _entropy(beta: torch.Tensor) -> torch.Tensor:
    return -torch.special.xlogy(beta, beta).sum(-1)


# ---------------------------------------------------------------------------------------------
# A. learnable routing temperature
# ---------------------------------------------------------------------------------------------

def test_temperature_parameter_exists_only_when_enabled() -> None:
    torch.manual_seed(0)
    off = VFEModel(tiny_config(n_layers=2))
    torch.manual_seed(0)
    on = VFEModel(tiny_config(n_layers=2, learn_attention_temperature=True))
    assert not any("routing_log_temperature" in k for k in off.state_dict())
    keys = sorted(k for k in on.state_dict() if "routing_log_temperature" in k)
    assert keys == ["relational_operators.routing_log_temperature",
                    "relational_operators_extra.0.routing_log_temperature"]
    for ops in on.relational_operators_by_layer:
        assert ops.routing_log_temperature.shape == (2,)
        assert torch.equal(ops.routing_log_temperature, torch.zeros(2))
    # Every pre-existing parameter keeps its name and initial value.
    on_state = on.state_dict()
    for name, value in off.state_dict().items():
        assert torch.equal(on_state[name], value), name


@pytest.mark.parametrize("kappa", [1.0, 0.7, [0.6, 1.3]])
def test_temperature_at_init_equals_configured_tau(kappa: object) -> None:
    ops = make_relational_operators([2, 2], 5, learn_attention_temperature=True).double()
    base = attention_tau(torch.tensor(kappa) if isinstance(kappa, list) else kappa, [2, 2])
    expected = torch.as_tensor(base, dtype=torch.float64).expand(2)
    torch.testing.assert_close(ops.routing_temperature(base, torch.float64), expected, rtol=0, atol=0)


def test_model_forward_at_init_matches_fixed_temperature() -> None:
    torch.manual_seed(0)
    off = VFEModel(tiny_config(n_layers=2, kappa_beta=0.7))
    torch.manual_seed(0)
    on = VFEModel(tiny_config(n_layers=2, kappa_beta=0.7, learn_attention_temperature=True))
    ids = torch.tensor([[0, 1, 2]])
    _, logits_off = off.forward_beliefs(ids, return_logits=True)
    _, logits_on = on.forward_beliefs(ids, return_logits=True)
    torch.testing.assert_close(logits_on, logits_off, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["projected_predictive", "projected_gradient"])
def test_gradient_reaches_every_layer_temperature(mode: str) -> None:
    extra = ({} if mode == "projected_predictive"
             else dict(predictive_covariance="passthrough", attention_residual=False))
    torch.manual_seed(1)
    model = VFEModel(tiny_config(n_layers=2, attention_mode=mode, learn_attention_temperature=True, **extra))
    ids = torch.tensor([[0, 1, 2]])
    _, logits = model.forward_beliefs(ids, return_logits=True)
    logits.square().sum().backward()
    for ops in model.relational_operators_by_layer:
        grad = ops.routing_log_temperature.grad
        assert grad is not None and torch.isfinite(grad).all() and bool((grad != 0).any())


def test_temperature_finite_difference() -> None:
    mu, sigma, frame, prior = _fixture()
    ops = make_relational_operators([2, 2], 11, learn_attention_temperature=True).double()
    with torch.no_grad():
        ops.routing_log_temperature.copy_(torch.tensor([0.2, -0.4]))
    weight = torch.randn(1, 4, 4, generator=torch.Generator().manual_seed(9), dtype=torch.float64)

    def objective() -> torch.Tensor:
        return (relational_predictive_readout(mu, sigma, frame, ops, tau=0.9, log_prior=prior) * weight).sum()

    derivative = torch.autograd.grad(objective(), ops.routing_log_temperature)[0]
    for h in range(2):
        with torch.no_grad():
            ops.routing_log_temperature[h] += 1e-6
        plus = objective().detach()
        with torch.no_grad():
            ops.routing_log_temperature[h] -= 2e-6
        minus = objective().detach()
        with torch.no_grad():
            ops.routing_log_temperature[h] += 1e-6
        torch.testing.assert_close(derivative[h], (plus - minus) / 2e-6, rtol=1e-5, atol=1e-9)


def test_row_min_learned_temperature_sets_routing_entropy() -> None:
    r"""Under row_min, beta = softmax_j(-(D_ij - min_j D_ij)/tau_h): hotter is flatter, colder is sharper."""
    mu, sigma, frame, prior = _fixture()
    ops = make_relational_operators([2, 2], 11, learn_attention_temperature=True,
                                    kl_max_reference="row_min").double()
    entropies = []
    for offset in (-1.0, 0.0, 1.0):
        with torch.no_grad():
            ops.routing_log_temperature.fill_(offset)
        capture = {}
        relational_predictive_readout(mu, sigma, frame, ops, tau=0.9, kl_max=1e6, log_prior=prior,
                                      capture=capture)
        beta = capture["beta_pre"]
        D = relational_score(mu, sigma, mu, sigma, frame, ops)
        D = D - D.masked_fill(~torch.isfinite(prior), math.inf).amin(-1, keepdim=True)
        reference = torch.softmax(-D / (0.9 * math.exp(offset)) + prior, dim=-1)
        torch.testing.assert_close(beta, reference, rtol=1e-12, atol=1e-12)
        entropies.append(_entropy(beta)[..., 1:].sum())        # rows 1.. have two or more sources
    assert entropies[0] < entropies[1] < entropies[2]


def test_fully_capped_absolute_row_has_zero_temperature_gradient() -> None:
    r"""The temperature cools only below the cap: an all-capped absolute row is uniform for every tau."""
    mu, sigma, frame, prior = _fixture()
    ops = make_relational_operators([2, 2], 11, learn_attention_temperature=True,
                                    kl_max_reference="absolute").double()
    D = relational_score(mu, sigma, mu, sigma, frame, ops).detach()
    kl_max = 0.5 * float(D.min())                                  # every admissible cost is capped
    assert kl_max > 0
    capture = {}
    out = relational_predictive_readout(mu, sigma, frame, ops, tau=0.9, kl_max=kl_max, log_prior=prior,
                                        capture=capture)
    beta = capture["beta_pre"]
    support = torch.isfinite(prior).to(beta.dtype)
    torch.testing.assert_close(beta, (support / support.sum(-1, keepdim=True)).expand_as(beta))
    grad = torch.autograd.grad(out.square().sum(), ops.routing_log_temperature)[0]
    # Zero up to the softmax-backward roundoff of equal logits (about 1e-17 here).
    assert grad.abs().max() < 1e-14
    live = relational_predictive_readout(mu, sigma, frame, ops, tau=0.9, kl_max=1e6, log_prior=prior)
    assert torch.autograd.grad(live.square().sum(), ops.routing_log_temperature)[0].abs().max() > 1e-6


def test_temperature_offset_is_clamped_like_decode_log_scale() -> None:
    ops = make_relational_operators([2, 2], 5, learn_attention_temperature=True).double()
    with torch.no_grad():
        ops.routing_log_temperature.copy_(torch.tensor([5.0, -5.0]))
    tau = ops.routing_temperature(2.0, torch.float64)
    torch.testing.assert_close(tau, torch.tensor([2.0 * math.exp(3.0), 2.0 * math.exp(-3.0)],
                                                 dtype=torch.float64))


def test_temperature_is_in_a_no_decay_score_group() -> None:
    from vfe3.train import _parameter_group, build_optimizer

    cfg = tiny_config(n_layers=2, learn_attention_temperature=True, m_relational_score_lr=3e-3)
    model = VFEModel(cfg)
    optimizer = build_optimizer(model, cfg)
    targets = {id(ops.routing_log_temperature) for ops in model.relational_operators_by_layer}
    groups = [g for g in optimizer.param_groups if targets & {id(p) for p in g["params"]}]
    assert len(groups) == 1
    assert targets <= {id(p) for p in groups[0]["params"]}
    assert groups[0]["weight_decay"] == 0.0 and groups[0]["lr"] == 3e-3
    for name, parameter in model.named_parameters():
        if "routing_log_temperature" in name:
            assert _parameter_group(name) == "relational_score"


def test_temperature_rejected_with_minimal_agent() -> None:
    with pytest.raises(ValueError, match="learn_attention_temperature"):
        tiny_config(learn_attention_temperature=True, predictive_readout="minimal_agent",
                    attention_residual=False)


# ---------------------------------------------------------------------------------------------
# B. comparison-map tie S_K := S_Q
# ---------------------------------------------------------------------------------------------

def test_tied_self_pair_cost_is_zero() -> None:
    mu, sigma, frame, _ = _fixture(n=5)
    ops = make_relational_operators([2, 2], 11, tie_comparison_maps=True).double()
    with torch.no_grad():
        ops.raw_score_q.add_(0.3 * torch.randn(ops.raw_score_q.shape, dtype=torch.float64,
                                               generator=torch.Generator().manual_seed(4)))
    D = relational_score(mu, sigma, mu, sigma, frame, ops)                     # (1, H, N, N)
    diagonal = D.diagonal(dim1=-2, dim2=-1)
    assert diagonal.abs().max() < 1e-12
    assert D.abs().max() > 1e-2                                                # off-diagonal costs are live
    untied = make_relational_operators([2, 2], 11).double()
    assert relational_score(mu, sigma, mu, sigma, frame, untied).diagonal(dim1=-2, dim2=-1).abs().min() > 1e-3


def test_tied_cost_with_identity_frames_is_the_pushed_forward_kl() -> None:
    mu, sigma, _, _ = _fixture(n=4)
    eye = torch.eye(4, dtype=torch.float64).expand(1, 4, 4, 4).clone()
    frame = CanonicalFrameContext(eye, eye.clone())
    ops = make_relational_operators([2, 2], 11, tie_comparison_maps=True).double()
    D = relational_score(mu, sigma, mu, sigma, frame, ops)
    S = ops.raw_score_q.detach()                                               # (H, d, K)
    for h in range(2):
        for i in range(4):
            for j in range(4):
                a, b = S[h] @ mu[0, i], S[h] @ mu[0, j]
                A = S[h] @ torch.diag(sigma[0, i]) @ S[h].T
                B = S[h] @ torch.diag(sigma[0, j]) @ S[h].T
                p = torch.distributions.MultivariateNormal(a, A)
                q = torch.distributions.MultivariateNormal(b, B)
                torch.testing.assert_close(D[0, h, i, j], torch.distributions.kl_divergence(p, q),
                                           rtol=1e-10, atol=1e-12)


def test_tie_removes_the_key_map_parameters() -> None:
    from vfe3.run_artifacts import _relational_artifact_fields

    cfg_off, cfg_on = tiny_config(n_layers=2), tiny_config(n_layers=2, tie_comparison_maps=True)
    off, on = VFEModel(cfg_off), VFEModel(cfg_on)
    K = cfg_on.embed_dim
    count = lambda m: sum(p.numel() for p in m.parameters())                     # noqa: E731
    assert count(off) - count(on) == cfg_on.n_layers * K * K
    assert not any("raw_score_k" in k for k in on.state_dict())
    for ops in on.relational_operators_by_layer:
        assert all(q is k for q, k in ops.comparison_runs())
        assert ops.comparison_pair()[0] is ops.comparison_pair()[1]
    fields_off = _relational_artifact_fields(cfg_off)["relational_operators"]
    fields_on = _relational_artifact_fields(cfg_on)["relational_operators"]
    assert fields_off["per_layer_parameter_counts"]["score"] - fields_on["per_layer_parameter_counts"]["score"] == K * K
    assert fields_on["score_role_parameter_counts"]["k"] == 0


def test_tied_model_trains_and_posthoc_maps_read_the_tie() -> None:
    from vfe3.posthoc.maps import relational_maps

    torch.manual_seed(2)
    model = VFEModel(tiny_config(n_layers=2, tie_comparison_maps=True))
    _, logits = model.forward_beliefs(torch.tensor([[0, 1, 2]]), return_logits=True)
    logits.square().sum().backward()
    for ops in model.relational_operators_by_layer:
        assert ops.raw_score_q.grad is not None and bool((ops.raw_score_q.grad != 0).any())
    maps = relational_maps(model.state_dict())
    for layer in maps.values():
        assert torch.equal(layer["S_K"], layer["S_Q"])


@pytest.mark.parametrize("overrides", [
    dict(predictive_readout="minimal_agent", attention_residual=False),
    dict(attention_mode="projected_gradient", predictive_covariance="passthrough", attention_residual=False),
])
def test_tie_is_rejected_off_the_kernel_predictive_route(overrides: dict) -> None:
    with pytest.raises(ValueError, match="tie_comparison_maps"):
        tiny_config(tie_comparison_maps=True, **overrides)
