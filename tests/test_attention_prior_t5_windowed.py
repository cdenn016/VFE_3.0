r"""T5 relative-position bias and windowed (local-attention) attention priors (PL13-priors).

These extend the attention-prior registry beyond uniform/causal/alibi. Each is a pure builder
returning a log-prior bias (Nq, Nk); they are config-selectable through the live _PRIORS
registry (config.py validates beta_attention_prior against tuple(sorted(_PRIORS))),
so no config-validator edit is needed. Variant params (window, T5 bucketing) use defaults when
invoked through the model's call site, mirroring the existing alibi prior (whose slope is also
default-only from the model); tuning them from config is the separate call-site-threading item.
"""

from tests.profiles import test_config
import math

import torch

from vfe3.attention_prior import (
    _PRIORS,
    _t5_relative_position_bucket,
    attention_log_prior,
)
from vfe3.model.model import VFEModel


def test_new_priors_are_registered_and_config_selectable():
    for name in ("windowed", "causal_windowed", "t5_relative_bias"):
        assert name in _PRIORS
        # config validation accepts the new prior names against the live registry
        test_config(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, beta_attention_prior=name)


def test_windowed_is_a_symmetric_band():
    B = attention_log_prior("windowed", 6, 6, window=2)
    inf = float("-inf")
    for i in range(6):
        for j in range(6):
            if abs(i - j) <= 2:
                assert B[i, j] == 0.0
            else:
                assert B[i, j] == inf


def test_causal_windowed_is_causal_and_banded():
    B = attention_log_prior("causal_windowed", 6, 6, window=2)
    inf = float("-inf")
    for i in range(6):
        for j in range(6):
            if 0 <= i - j <= 2:                     # past, within band
                assert B[i, j] == 0.0
            else:                                   # future OR too far back
                assert B[i, j] == inf


def test_t5_bucket_matches_reference_small_and_large():
    # Non-bidirectional (causal): future -> bucket 0; past distance d<max_exact is exact (bucket=d).
    rel = torch.tensor([[-3, -1, 0, 1, 5]])         # key - query
    buckets = _t5_relative_position_bucket(
        rel, bidirectional=False, num_buckets=32, max_distance=128)
    # rel>0 (future) clamps to past-distance 0 -> bucket 0; rel<=0 -> past distance |rel| (exact, <16)
    assert buckets.tolist() == [[3, 1, 0, 0, 0]]


def test_t5_bucket_is_monotone_nondecreasing_in_past_distance():
    rel = -torch.arange(0, 200).view(1, -1)         # past distances 0..199
    buckets = _t5_relative_position_bucket(
        rel, bidirectional=False, num_buckets=32, max_distance=128).flatten()
    diffs = buckets[1:] - buckets[:-1]
    assert torch.all(diffs >= 0)                    # never decreases as the key recedes into the past
    assert int(buckets.max()) == 31                 # clamped to num_buckets-1


def test_t5_relative_bias_is_causal_by_default():
    B = attention_log_prior("t5_relative_bias", 5, 5)
    inf = float("-inf")
    for i in range(5):
        for j in range(5):
            if j > i:
                assert B[i, j] == inf               # decoder form masks the future
            else:
                assert math.isfinite(B[i, j].item())


def test_t5_relative_bias_uses_supplied_learnable_table():
    # Passing a (num_buckets,) bias_values (a model's learnable handle) gathers per-bucket bias.
    table = torch.linspace(-1.0, -5.0, steps=8)
    B = attention_log_prior("t5_relative_bias", 4, 4, bias_values=table,
                            num_buckets=8, max_distance=16)
    # diagonal is relative distance 0 -> bucket 0 -> table[0]
    for i in range(4):
        assert torch.isclose(B[i, i], table[0])


def _model_forward_finite(prior_name: str):
    cfg = test_config(vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                     n_e_steps=1, m_phi_lr=0.0, beta_attention_prior=prior_name)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    tokens = torch.randint(0, 12, (2, 8))
    targets = torch.randint(0, 12, (2, 8))
    logits = model(tokens)
    _, _, ce = model(tokens, targets)
    assert torch.isfinite(logits).all() and torch.isfinite(ce)


def test_model_forward_under_windowed_priors():
    for name in ("causal_windowed", "windowed", "t5_relative_bias"):
        _model_forward_finite(name)


# --- learnable T5 relative-position bias (t5_learnable_bias) -----------------
# GL(K)_attention.tex:826-838 derives the relative-position bias as a LEARNABLE function of offset
# (pi_j ∝ exp(b_{i-j})); the model owns a (num_buckets,) nn.Parameter table read by the prior.
# Sanctioned no-NN exception (default OFF): the bias is a scalar function of position offset, touches
# no gauge transport, so it does NOT break equivariance.

def _t5_learnable_cfg(**kw):
    base = dict(vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                n_e_steps=1, m_phi_lr=0.0,
                beta_attention_prior="t5_relative_bias", t5_learnable_bias=True)
    base.update(kw)
    return test_config(**base)


def test_t5_learnable_bias_defaults_false():
    assert test_config().t5_learnable_bias is False


def test_t5_relative_bias_gradient_flows_to_table():
    # Function-level contract: a requires_grad bias_values table receives gradient through the gather
    # (the future-masked -inf entries overwrite table values, so grad reaches only the lower-triangle
    # buckets -- which is exactly what trains).
    table = torch.zeros(8, requires_grad=True)
    B = attention_log_prior("t5_relative_bias", 4, 4, bias_values=table,
                            num_buckets=8, max_distance=16)
    B[torch.isfinite(B)].sum().backward()
    assert table.grad is not None
    assert table.grad.abs().sum() > 0


def test_t5_learnable_bias_creates_trainable_parameter():
    m = VFEModel(_t5_learnable_cfg())
    assert isinstance(m.t5_bias, torch.nn.Parameter)
    assert m.t5_bias.shape == (m.cfg.t5_num_buckets,)
    assert m.t5_bias.requires_grad
    assert any(p is m.t5_bias for p in m.parameters())   # registered, so the optimizer sees it


def test_t5_learnable_bias_absent_when_off():
    # Pure path: t5_learnable_bias=False creates no parameter (no t5_bias attribute).
    m = VFEModel(_t5_learnable_cfg(t5_learnable_bias=False))
    assert not hasattr(m, "t5_bias")


def test_t5_learnable_bias_inits_to_fixed_default():
    # Byte-identical at step 0: the parameter inits to the fixed-table default -log1p(bucket), so the
    # learnable model's attention prior equals the non-learnable t5_relative_bias prior before training.
    m = VFEModel(_t5_learnable_cfg())
    dev = torch.device("cpu")
    a = m._attention_log_prior(8, dev)
    b = attention_log_prior("t5_relative_bias", 8, 8, device=dev, dtype=a.dtype,
                            num_buckets=m.cfg.t5_num_buckets, max_distance=m.cfg.t5_max_distance)
    mask = torch.isfinite(a)
    assert torch.equal(mask, torch.isfinite(b))          # identical -inf (future-mask) structure
    assert torch.allclose(a[mask], b[mask], atol=1e-6)


def test_t5_learnable_bias_cache_bypassed_when_table_changes():
    # The learnable table is a live parameter that changes every step, so the (name,N,...) prior cache
    # MUST be bypassed -- a cached tensor would serve a stale table. Mutating the parameter must change
    # the produced prior.
    m = VFEModel(_t5_learnable_cfg())
    dev = torch.device("cpu")
    a1 = m._attention_log_prior(8, dev)
    with torch.no_grad():
        m.t5_bias.sub_(1.0)                              # shift every bucket bias
    a2 = m._attention_log_prior(8, dev)
    mask = torch.isfinite(a1)
    assert not torch.allclose(a1[mask], a2[mask])


def test_t5_learnable_bias_trains_end_to_end():
    torch.manual_seed(0)
    m = VFEModel(_t5_learnable_cfg())
    x = torch.randint(0, 12, (2, 8)); y = torch.randint(0, 12, (2, 8))
    _, loss, _ = m(x, y); loss.backward()
    assert torch.isfinite(loss)
    assert m.t5_bias.grad is not None
    assert torch.isfinite(m.t5_bias.grad).all() and m.t5_bias.grad.abs().sum() > 0


def test_t5_learnable_bias_is_grouped_by_build_optimizer():
    # build_optimizer asserts every trainable parameter lands in exactly one group; the learnable
    # T5 bias must be grouped or it silently never trains (and the coverage guard raises).
    from vfe3.train import build_optimizer
    m = VFEModel(_t5_learnable_cfg())
    opt = build_optimizer(m, m.cfg)                       # raised AssertionError before the fix
    grouped = {p for g in opt.param_groups for p in g["params"]}
    assert any(p is m.t5_bias for p in grouped)
