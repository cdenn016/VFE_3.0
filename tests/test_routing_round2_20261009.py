r"""Router remediation of the October 9 round-2 audit: tie warning, smooth temperature bound, row-minimum gradient,
single tied projection, and the keyword-required clamp reference."""
import math
import warnings

import pytest
import torch

from tests.profiles import make_relational_operators, tiny_config
from vfe3.contracts import CanonicalFrameContext
from vfe3.free_energy import attention_weights
from vfe3.gradients.relational import clamped_routing_cost, relational_score


def _fixture(n: int = 4, seed: int = 3) -> tuple:
    g = torch.Generator().manual_seed(seed)
    mu = torch.randn(1, n, 4, generator=g, dtype=torch.float64)
    sigma = 0.5 + torch.rand(1, n, 4, generator=g, dtype=torch.float64)
    U = torch.eye(4, dtype=torch.float64).expand(1, n, 4, 4).clone()
    U[..., 0, 1] = 0.3
    frame = CanonicalFrameContext(U, torch.linalg.inv(U))
    return mu, sigma, frame


# ------------------------------------------------------------------------------------- item 2: tie with a self prior

@pytest.mark.parametrize("prior", ["causal", "causal_alibi"])
def test_tie_with_a_self_admitting_prior_warns(prior: str) -> None:
    with pytest.warns(UserWarning, match="causal_noself"):
        tiny_config(tie_comparison_maps=True, beta_attention_prior=prior)


@pytest.mark.parametrize("prior", ["causal_noself", "causal_alibi_noself"])
def test_tie_with_a_self_masking_prior_is_silent(prior: str) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tiny_config(tie_comparison_maps=True, beta_attention_prior=prior)
    assert not any("tie_comparison_maps" in str(w.message) for w in caught)


# ------------------------------------------------------------------------------------- item 3: smooth temperature bound

def test_temperature_bound_is_smooth_and_exact_at_zero() -> None:
    ops = make_relational_operators([2, 2], 5, learn_attention_temperature=True).double()
    torch.testing.assert_close(ops.routing_temperature(2.0, torch.float64),
                               torch.full((2,), 2.0, dtype=torch.float64), rtol=0, atol=0)
    with torch.no_grad():
        ops.routing_log_temperature.copy_(torch.tensor([5.0, -5.0]))
    tau = ops.routing_temperature(2.0, torch.float64)
    expected = 2.0 * torch.exp(3.0 * torch.tanh(torch.tensor([5.0, -5.0], dtype=torch.float64) / 3.0))
    torch.testing.assert_close(tau, expected, rtol=1e-15, atol=0)
    assert bool((tau < 2.0 * math.exp(3.0)).all()) and bool((tau > 2.0 * math.exp(-3.0)).all())
    grad = torch.autograd.grad(tau.sum(), ops.routing_log_temperature)[0]
    assert bool((grad.abs() > 1e-3).all())                     # nonzero beyond the former clamp window


# ------------------------------------------------------------------------------------- item 4: row minimum carries gradient

def _clamped_gibbs(D: torch.Tensor, ops: torch.nn.Module, kl_max: float, tau: float,
                   prior: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    beta = attention_weights(clamped_routing_cost(D, ops, kl_max=kl_max, log_prior=prior), tau=tau, log_prior=prior)
    return (beta * weight).sum()


def test_row_min_clamped_gibbs_gradient_matches_finite_differences_when_capped() -> None:
    r"""Hot temperature with capped entries: autograd through the clamp and the row minimum equals central
    differences of the cost grid (unique argmin per row, no entry at the cap boundary)."""
    ops = make_relational_operators([2, 2], 3, kl_max_reference="row_min").double()
    g = torch.Generator().manual_seed(1)
    n = 5
    D = (6.0 * torch.rand(1, 2, n, n, generator=g, dtype=torch.float64)).requires_grad_()
    prior = torch.zeros(n, n, dtype=torch.float64).masked_fill(torch.ones(n, n, dtype=torch.bool).triu(1), -math.inf)
    weight = torch.randn(1, 2, n, n, generator=g, dtype=torch.float64)
    kl_max, tau = 2.5, 4.0
    gaps = D.detach() - D.detach().masked_fill(~torch.isfinite(prior), math.inf).amin(-1, keepdim=True)
    assert bool(((gaps > kl_max) & torch.isfinite(prior)).any())                      # the cap binds
    assert float((gaps - kl_max).abs().min()) > 1e-3
    analytic = torch.autograd.grad(_clamped_gibbs(D, ops, kl_max, tau, prior, weight), D)[0]
    numeric = torch.zeros_like(D)
    h = 1e-6
    with torch.no_grad():
        for index in torch.cartesian_prod(*(torch.arange(s) for s in D.shape)):
            index = tuple(index.tolist())
            base = D.detach().clone()
            base[index] += h
            plus = _clamped_gibbs(base, ops, kl_max, tau, prior, weight)
            base[index] -= 2 * h
            minus = _clamped_gibbs(base, ops, kl_max, tau, prior, weight)
            numeric[index] = (plus - minus) / (2 * h)
    torch.testing.assert_close(analytic, numeric, rtol=1e-6, atol=1e-9)
    # The detached reference gave a different gradient on the argmin entries.
    shifted = D.detach() - D.detach().masked_fill(~torch.isfinite(prior), math.inf).amin(-1, keepdim=True)
    shifted.requires_grad_()
    from vfe3.families.base import safe_kl_clamp
    beta = attention_weights(safe_kl_clamp(shifted, kl_max=kl_max), tau=tau, log_prior=prior)
    detached = torch.autograd.grad((beta * weight).sum(), shifted)[0]
    assert float((detached - numeric).abs().max()) > 1e-4


def test_row_min_gradient_rows_sum_to_zero() -> None:
    r"""The shifted cost is row-shift invariant, so with the row minimum differentiated the cost gradient of every
    row sums to zero (the exactness condition of the row-constant query log-determinant skip)."""
    ops = make_relational_operators([2, 2], 3, kl_max_reference="row_min").double()
    g = torch.Generator().manual_seed(2)
    D = (6.0 * torch.rand(1, 2, 4, 4, generator=g, dtype=torch.float64)).requires_grad_()
    weight = torch.randn(1, 2, 4, 4, generator=g, dtype=torch.float64)
    beta = attention_weights(clamped_routing_cost(D, ops, kl_max=2.0), tau=3.0)
    grad = torch.autograd.grad((beta * weight).sum(), D)[0]
    assert float(grad.sum(-1).abs().max()) < 1e-15


# ------------------------------------------------------------------------------------- item 5: one tied projection

@pytest.mark.parametrize("rope", [None, 100.0])
def test_tied_score_is_bitwise_the_two_role_projection(rope: object) -> None:
    mu, sigma, frame = _fixture(n=5)
    tied = make_relational_operators([2, 2], 11, tie_comparison_maps=True, comparison_rope_base=rope).double()
    untied = make_relational_operators([2, 2], 11, comparison_rope_base=rope).double()
    with torch.no_grad():
        untied.raw_score_q.copy_(tied.raw_score_q)
        untied.raw_score_k.copy_(tied.raw_score_q)
    for ops in (tied, untied):
        ops.float()
    a = relational_score(mu.float(), sigma.float(), mu.float(), sigma.float(),
                         CanonicalFrameContext(frame.forward.float(), frame.inverse.float()), tied)
    b = relational_score(mu.float(), sigma.float(), mu.float(), sigma.float(),
                         CanonicalFrameContext(frame.forward.float(), frame.inverse.float()), untied)
    assert torch.equal(a, b)


# ------------------------------------------------------------------------------------- item 7: required clamp reference

def test_kl_max_reference_is_keyword_required() -> None:
    from vfe3.model.relational_operators import RelationalOperators
    from tests.profiles import config_values

    values = config_values()
    required = dict(attention_residual=False, predictive_covariance="passthrough",
                    receiver_law=values["relational_receiver_law"],
                    receiver_update=values["relational_receiver_update"],
                    content_link_chart=values["content_link_chart"],
                    minimal_agent_inference=values["minimal_agent_inference"],
                    rope_comparison_covariance="full")
    with pytest.raises(TypeError, match="kl_max_reference"):
        RelationalOperators([2, 2], 1, **required)
    assert RelationalOperators([2, 2], 1, kl_max_reference="row_min", **required).kl_max_reference == "row_min"
    assert make_relational_operators([2, 2], 1).kl_max_reference == values["routing_kl_max_reference"]
