"""Projected-filtering envelope admits prior-bank decode and does not pin the sigma flag.

The train_vfe3 projected_predictive profile with family='gaussian_full' must be able
to select decode_mode='full_chunked' (the principled prior-bank KL readout). The
previous envelope pinned decode_mode='linear' and skip_belief_sigma_update=True as
support defaults with no stated mathematical obstruction.

These tests pin the admission and a tiny CPU forward/backward. They do not claim the
projected E-step updates Sigma_q -- it does not (predictive keeps covariance,
filtering returns no sigma gradient), which is reported separately by inactive_fields.
"""
from __future__ import annotations

from tests.profiles import test_config

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pytest
import torch

from tests.profiles import tiny_config
from vfe3.model.model import VFEModel

torch.set_num_threads(1)


def _projected_surface(**overrides: object) -> dict:
    """Minimal projected_predictive surface, shrunk to CPU-tiny widths."""
    values: dict = dict(
        vocab_size=7, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1,
        n_e_steps=1, family="gaussian_full",
        attention_mode="projected_predictive",
        relational_input_scope="full", relational_output_scope="full",
        attention_residual=True, relational_identity_content=False,
        divergence_family="renyi", renyi_order=1.0, include_attention_entropy=True,
        gradient_mode="filtering", e_step_update="gradient",
        e_step_gradient="unroll", oracle_unroll_grad=True,
        e_phi_lr=0.0, transport_mode="flat", gauge_group="block_glk",
        encode_mode="per_token", decode_mode="full_chunked",
        skip_belief_sigma_update=False,
        decode_bias=False, decode_unigram_prior=False, untie_decode_bank=False,
        use_block_mlp=False, use_head_mixer=False,
        norm_type_block="none", norm_type_final="none",
        beta_attention_prior="causal",
        fp32_matmul_precision=None, decode_av_precision="fp32",
        decode_ranking_fp64_escalation=False,
        e_mu_q_trust=None, exp_fp64_norm_threshold=15.0,
        full_cov_congruence_certification=False,
        transport_clamp_monitor=False, generate_embedding_figures=False,
        warmup_steps=0, max_steps=2, log_interval=0, eval_interval=0,
    )
    values.update(overrides)
    return values


def test_projected_predictive_admits_full_chunked_and_unpinned_sigma_flag() -> None:
    """decode_mode='full_chunked' and skip_belief_sigma_update=False construct under projected_predictive."""
    cfg = test_config(**_projected_surface())
    assert cfg.decode_mode == "full_chunked"
    assert cfg.skip_belief_sigma_update is False
    assert cfg.family == "gaussian_full"
    assert cfg.attention_mode == "projected_predictive"


def test_projected_predictive_admits_family_chunked() -> None:
    """family_chunked is admitted alongside full_chunked for a full-covariance family."""
    cfg = test_config(**_projected_surface(decode_mode="family_chunked"))
    assert cfg.decode_mode == "family_chunked"


def test_projected_predictive_admits_diagonal_chunked_for_diagonal_family() -> None:
    """diagonal_chunked is admitted when paired with a diagonal-covariance family."""
    cfg = test_config(**_projected_surface(
        decode_mode="diagonal_chunked", family="gaussian_diagonal",
    ))
    assert cfg.decode_mode == "diagonal_chunked"
    assert cfg.family == "gaussian_diagonal"


def test_projected_predictive_admits_skip_belief_sigma_update_true() -> None:
    """The frozen-sigma ablation remains reachable alongside the prior-bank decoders."""
    cfg = test_config(**_projected_surface(skip_belief_sigma_update=True))
    assert cfg.skip_belief_sigma_update is True


def test_projected_predictive_full_chunked_tiny_train_step() -> None:
    """A tiny CPU forward + CE + backward completes under the train_vfe3 prior-bank surface.

    Fused chunked CE returns logits=None by design (model.py:1321) -- the (B, N, V)
    tensor is never materialized -- so the assertion is on the loss, not the logits.
    """
    cfg = test_config(**_projected_surface())
    torch.manual_seed(0)
    model = VFEModel(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 4))
    targets = torch.randint(0, cfg.vocab_size, (2, 4))
    logits, loss, _ = model(tokens, targets)
    assert logits is None
    assert torch.isfinite(loss)
    loss.backward()


def test_projected_filtering_skip_false_does_not_typeerror() -> None:
    """skip_belief_sigma_update=False on the conditional route must not hit float * None.

    relational filtering returns grad_sigma=None by signature; the E-step now freezes
    sigma when nat_sigma is None instead of retracting with a None step.
    """
    cfg = test_config(**_projected_surface(
        attention_mode="projected_conditional",
        relational_output_scope="head",
        attention_residual=False,
        skip_belief_sigma_update=False,
    ))
    torch.manual_seed(0)
    model = VFEModel(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 4))
    belief, logits = model.forward_beliefs(tokens, return_logits=True)
    assert torch.isfinite(logits).all()
    assert belief.sigma.shape == belief.mu.shape or (
        belief.sigma.dim() == belief.mu.dim() + 1
    )


def test_transported_linear_still_constructs() -> None:
    """The pure-path (transported + linear) profile is unchanged by the envelope edit."""
    cfg = tiny_config(
        attention_mode="transported",
        predictive_covariance="passthrough",
        attention_residual=False,
        relational_output_scope="head",
        decode_mode="linear",
        skip_belief_sigma_update=True,
    )
    assert cfg.decode_mode == "linear"
    assert cfg.attention_mode == "transported"


def test_projected_still_rejects_non_gaussian_family() -> None:
    """The Gaussian-family admission on the projected route is retained."""
    with pytest.raises(ValueError, match="unsupported projected_filtering configuration"):
        test_config(**_projected_surface(family="laplace_diagonal"))


def test_projected_full_chunked_still_rejects_cov_kind_mismatch() -> None:
    """full_chunked requires a full-covariance query family; the rank check is not bypassed."""
    with pytest.raises(ValueError, match="rank-incompatible"):
        test_config(**_projected_surface(
            decode_mode="full_chunked", family="gaussian_diagonal",
        ))
