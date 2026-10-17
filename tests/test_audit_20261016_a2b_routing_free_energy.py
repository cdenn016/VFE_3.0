r"""The kernel router's own label free energy (audit 2026-10-16 F114).

``relational_routing_free_energy`` is, per token and summed over heads, the reduced free energy
:math:`-\tau_h\log\sum_j\pi_{ij}e^{-D_{ij}/\tau_h}` of the clamped cost :math:`D` the router read, at the
temperature it ran at. Its minimizer over the label rows is the model's router :math:`\beta`, so it equals
:math:`\sum_j\beta_{ij}D_{ij} + \tau_h\sum_j\beta_{ij}\log(\beta_{ij}/\pi_{ij})` at the captured :math:`\beta`.
NaN under the minimal agent, whose label rows are not a softmax of one captured cost under every inference
mode. Tiny CPU models only (K = 8, two heads of 4, N = 5, vocab 9).
"""
import math
import warnings
from typing import Any, Dict

import pytest
import torch

import vfe3.free_energy as fe
from tests.profiles import tiny_config
from vfe3.model.block import _as_coeff
from vfe3.model.model import VFEModel
from vfe3.posthoc.klcap import causal_support

TOKENS = torch.tensor([[0, 1, 2, 3, 4], [4, 5, 6, 7, 8]])


def _model(**overrides: Any) -> VFEModel:
    values: Dict[str, Any] = dict(vocab_size=9, embed_dim=8, n_heads=2, max_seq_len=5, n_layers=2, phi_scale=0.4)
    values.update(overrides)
    torch.manual_seed(3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VFEModel(tiny_config(**values)).eval()


def _tau(model: VFEModel, layer: int) -> torch.Tensor:
    r"""(H,) float64 temperature the layer's router runs at."""
    cpu = torch.device("cpu")
    tau = model.relational_operators_by_layer[layer].routing_temperature(
        fe.attention_tau(_as_coeff(model.cfg.kappa_beta, cpu), model.group.irrep_dims, device=cpu), torch.float64)
    return torch.as_tensor(tau, dtype=torch.float64).detach().reshape(-1).expand(len(model.group.irrep_dims))


def _envelope_at_router(model: VFEModel, snapshot: Any, layer: int) -> float:
    r"""Per-token :math:`\sum_h[\sum_j\beta D + \tau_h\sum_j\beta\log(\beta/\pi)]` at the captured router."""
    cap = model._layer_capture(snapshot, layer)
    beta, score = cap["beta_pre"].double(), cap["score_energy_pre"].double()   # (B, H, N, N)
    support = causal_support(beta, snapshot.log_prior)
    prior = snapshot.log_prior.double()
    while prior.dim() < 4:
        prior = prior.unsqueeze(0)
    log_pi = torch.where(support, torch.log_softmax(prior, dim=-1).expand_as(beta), 0.0)
    tau = _tau(model, layer)[None, :, None]                                     # (1, H, 1)
    coupling = (beta * torch.where(support, score, 0.0)).sum(dim=-1)            # (B, H, N)
    entropy = (torch.special.xlogy(beta, beta) - beta * log_pi).sum(dim=-1)     # (B, H, N) KL(beta || pi)
    return float((coupling + tau * entropy).sum(dim=1).mean())


def _log_partition_form(model: VFEModel, snapshot: Any, layer: int) -> float:
    cap = model._layer_capture(snapshot, layer)
    score = cap["score_energy_pre"].double()
    support = causal_support(cap["beta_pre"], snapshot.log_prior)
    prior = snapshot.log_prior.double()
    while prior.dim() < 4:
        prior = prior.unsqueeze(0)
    tau = _tau(model, layer)[None, :, None, None]                               # (1, H, 1, 1)
    logits = torch.where(support, torch.log_softmax(prior, dim=-1) - score / tau, float("-inf"))
    rows = -tau[..., 0] * torch.logsumexp(logits, dim=-1)                       # (B, H, N)
    return float(torch.where(support.any(dim=-1), rows, 0.0).sum(dim=1).mean())


@pytest.mark.parametrize("overrides", [
    dict(),
    dict(beta_attention_prior="causal_alibi"),
    dict(beta_attention_prior="causal_alibi", routing_kl_max_reference="row_min", kl_max=0.5),
    dict(beta_attention_prior="causal_alibi", kl_max=0.5),
    dict(learn_attention_temperature=True),
    dict(relational_routing="noisy_key"),
    dict(attention_mode="projected_gradient", attention_residual=False, predictive_covariance="passthrough",
         lambda_alpha_mode="state_dependent"),
], ids=["causal", "alibi", "alibi_row_min_capped", "alibi_absolute_capped", "learned_tau", "noisy_key",
        "full_link"])
def test_column_is_the_label_free_energy_the_router_minimizes(overrides: Dict[str, Any]) -> None:
    model = _model(**overrides)
    if model.cfg.learn_attention_temperature:
        with torch.no_grad():
            for layer, ops in enumerate(model.relational_operators_by_layer):
                ops.routing_log_temperature.copy_(torch.tensor([0.6, -0.8]) * (layer + 1))
    snapshot = model.build_diagnostic_snapshot(TOKENS)
    d = model.diagnostics(TOKENS, snapshot=snapshot)
    per_layer = model.diagnostics_per_layer(TOKENS, snapshot=snapshot)["relational_routing_free_energy"]
    assert len(per_layer) == model.cfg.n_layers and d["relational_routing_free_energy"] == per_layer[-1]
    for layer, value in enumerate(per_layer):
        assert math.isfinite(value)
        assert value == pytest.approx(_log_partition_form(model, snapshot, layer), rel=1e-12, abs=1e-12)
        # the captured router is the minimizer: the envelope at beta equals the reduced free energy
        assert value == pytest.approx(_envelope_at_router(model, snapshot, layer), rel=2e-5, abs=2e-5)


def test_minimal_agent_reports_nan() -> None:
    model = _model(predictive_readout="minimal_agent", attention_residual=False)
    per_layer = model.diagnostics_per_layer(TOKENS)["relational_routing_free_energy"]
    assert len(per_layer) == 2 and all(math.isnan(value) for value in per_layer)
