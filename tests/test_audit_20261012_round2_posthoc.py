r"""Round-2 audit of 2026-10-12, item 11: the routing-covariance intervention rows under ``noisy_key`` routing.

Both counterfactual scorers called the routing cost without ``key_noise``, so under
``relational_routing='noisy_key'`` the rows raised a TypeError and were recorded as failed. Tiny CPU models only.
"""
from __future__ import annotations

import warnings
from typing import Any, Callable

import pytest
import torch

from tests.profiles import test_config
from vfe3.model.model import VFEModel
from vfe3.posthoc import interventions as iv


@pytest.fixture(autouse=True)
def _one_thread() -> None:
    torch.set_num_threads(1)


def _model() -> VFEModel:
    cfg = test_config(vocab_size=11, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1, batch_size=2,
                      attention_mode="projected_predictive", predictive_readout="kernel",
                      relational_routing="noisy_key", relational_key_noise=0.7, block_mlp_expansion=2,
                      norm_type_block="none", layernorm_affine=False, skip_belief_sigma_update=False,
                      beta_attention_prior="causal")
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(cfg).eval()
    with torch.no_grad():
        model.prior_bank.sigma_log_embed.normal_(0.0, 0.5)
        model.prior_bank.phi_embed.normal_(0.0, 0.3)
    return model


def _logits(model: VFEModel) -> torch.Tensor:
    ids = torch.randint(0, 11, (2, 5), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        return model.forward_beliefs(ids, return_logits=True)[1]


@pytest.mark.parametrize("arm", [iv.routing_covariances_identity, iv.frames_removed_routing_covariance])
def test_routing_covariance_rows_compute_under_noisy_key(arm: Callable[..., Any]) -> None:
    r"""Each row scores with the head key noise :math:`n_h`: it runs, moves the logits, and depends on :math:`n_h`."""
    model = _model()
    base = _logits(model)
    with arm(model) as record:
        logits = _logits(model)
    assert record["intercepted_calls"] > 0
    assert torch.isfinite(logits).all() and not torch.equal(logits, base)
    operators = model.relational_operators_by_layer[0]
    with torch.no_grad():
        operators.routing_key_log_noise.add_(1.0)
    with arm(model):
        moved = _logits(model)
    assert not torch.equal(moved, logits)
