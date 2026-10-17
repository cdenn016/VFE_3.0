r"""The LayerNorm pushforward trace-cap toggle and the per-layer block-variance diagnostic (2026-10-17).

``block_norm_trace_cap=False`` returns the uncapped delta-method pushforward; True (default) keeps the sphere-bound
scale :math:`\min(1, K\max_k\gamma_k^2/\operatorname{tr}P)`. ``diagnostics/block_variance.csv`` records, per layer
and stage (pre-norm, post-norm, block output), quantiles of :math:`\mathrm{diag}(\Sigma_i)` and the median mean rms,
from the eval-cadence capture pass. Tiny float64 CPU cases only.
"""
from __future__ import annotations

import warnings

import pytest
import torch

import vfe3.train as T
from tests.profiles import tiny_config
from vfe3.geometry.norms import AffineLayerNorm
from vfe3.model import block_mlp as bm
from vfe3.model.model import VFEModel

K = 4


@pytest.fixture(autouse=True)
def _one_thread() -> None:
    torch.set_num_threads(1)


def _norm() -> AffineLayerNorm:
    norm = AffineLayerNorm(K, eps=1e-5).double()
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([0.5, 1.0, 1.5, 0.8], dtype=torch.float64))
    return norm


def test_cap_off_returns_the_uncapped_pushforward_and_counts_nothing() -> None:
    norm = _norm()
    g = torch.Generator().manual_seed(0)
    mu = 0.1 * torch.randn(3, K, generator=g, dtype=torch.float64)          # small s: Sigma >> s^2, the cap binds
    sigma = 50.0 + torch.rand(3, K, generator=g, dtype=torch.float64)
    uncapped = bm._uncapped_block_norm_pushforward(norm, mu, sigma, 1e-3)
    bm.reset_layernorm_trace_cap_elements()
    capped = bm.block_norm_covariance_pushforward(norm, mu, sigma, 1e-3)
    assert bm.layernorm_trace_cap_elements() == 3                            # every token capped
    bound = K * float(norm.weight.square().max())
    torch.testing.assert_close(capped.sum(-1), torch.full((3,), bound, dtype=torch.float64))
    bm.reset_layernorm_trace_cap_elements()
    off = bm.block_norm_covariance_pushforward(norm, mu, sigma, 1e-3, trace_cap=False)
    assert bm.layernorm_trace_cap_elements() == 0
    torch.testing.assert_close(off, uncapped)
    assert bool((off.sum(-1) > bound).all())


def _cfg(**over: object):
    values = dict(embed_dim=K, n_heads=2, vocab_size=9, max_seq_len=5, n_layers=2,
                  attention_mode="projected_predictive", predictive_covariance="moment_matched",
                  norm_type_block="layernorm", layernorm_affine=True, use_block_mlp=True,
                  block_mlp_mode="coordinate", block_mlp_expansion=2, block_norm_covariance="pushforward",
                  generate_embedding_figures=False)
    values.update(over)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tiny_config(**values)


def test_config_validation_inactive_marking_and_model_forward() -> None:
    with pytest.raises(ValueError, match="block_norm_trace_cap"):
        _cfg(block_norm_trace_cap=1)
    assert "block_norm_trace_cap" not in _cfg().inactive_fields()
    assert "block_norm_trace_cap" in _cfg(block_norm_covariance="passthrough").inactive_fields()
    tokens = torch.randint(0, 9, (2, 5), generator=torch.Generator().manual_seed(1))
    for cap in (True, False):
        torch.manual_seed(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = VFEModel(_cfg(block_norm_trace_cap=cap)).double().eval()
        belief, _ = model.forward_beliefs(tokens)
        assert bool(torch.isfinite(belief.sigma).all())


def test_block_variance_rows_cover_every_layer_and_stage() -> None:
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(_cfg())
    windows = torch.randint(0, 9, (3, 5), generator=torch.Generator().manual_seed(2))
    variances: dict = {}
    T._residual_and_attention_rows(model, windows, 7, torch.device("cpu"), variance_out=variances)
    rows = T._block_variance_rows(variances, 7)
    assert [(r["layer"], r["stage"]) for r in rows] == [
        (layer, stage) for layer in range(2) for stage in ("pre_norm", "post_norm", "block_out")]
    for row in rows:
        assert set(row) == set(T.BLOCK_VARIANCE_FIELDS) and row["step"] == 7
        assert 0.0 < row["var_min"] <= row["var_q001"] <= row["var_q01"] <= row["var_median"] <= row["var_max"]
        assert row["mean_rms_median"] > 0.0


def test_block_variance_rows_match_a_direct_computation() -> None:
    # [variance chunks, mean-rms chunks]: variances 1, 2, 3, 4, 10 and rms 3, 5, 30
    variances = {(0, "pre_norm"): [[torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([10.0])],
                                   [torch.tensor([3.0, 5.0]), torch.tensor([30.0])]]}
    (row,) = T._block_variance_rows(variances, 3)
    assert row["var_min"] == 1.0 and row["var_max"] == 10.0 and row["var_median"] == pytest.approx(3.0)
    assert row["mean_rms_median"] == pytest.approx(5.0)
