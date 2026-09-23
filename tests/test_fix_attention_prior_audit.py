r"""Audit fix: a Press-et-al.-faithful `causal_alibi` attention prior.

The `alibi` prior returns a SYMMETRIC bias B_ij = -slope*|i-j| with no causal
mask, so for an autoregressive LM it leaks to future keys (j > i). This pins the
new `causal_alibi` prior that adds the linear distance bias ON TOP OF the causal
mask, and guards that `causal`/`alibi`/`uniform` are unchanged.
"""

import torch

from vfe3.attention_prior import (
    _PRIORS,
    attention_log_prior,
    get_prior,
    prior_causal,
)


def test_causal_alibi_is_registered_and_retrievable():
    # (1) present in the registry and reachable via the getter / dispatcher.
    assert "causal_alibi" in _PRIORS
    assert get_prior("causal_alibi") is _PRIORS["causal_alibi"]
    H, N = 1, 4
    B = attention_log_prior("causal_alibi", N, N, n_heads=H, alibi_slope=0.5)
    assert B.shape == (H, N, N)


def test_causal_alibi_masks_strictly_above_diagonal():
    # (2) -inf strictly above the diagonal (j > i); finite on and below it. (H, N, N) shape.
    H, n = 1, 5
    B = attention_log_prior("causal_alibi", n, n, n_heads=H, alibi_slope=0.5)
    assert B.shape == (H, n, n)
    for i in range(n):
        for j in range(n):
            if j > i:
                assert torch.isneginf(B[0, i, j]), (i, j, B[0, i, j])
            else:
                assert torch.isfinite(B[0, i, j]), (i, j, B[0, i, j])


def test_causal_alibi_equals_causal_mask_plus_alibi_on_triangle():
    # (3) on the causal triangle (j <= i) value == -slope_h*(i - j).  H=1 for simplicity.
    import math
    H, n = 1, 6
    alibi_slope = 0.25
    B = attention_log_prior("causal_alibi", n, n, n_heads=H, alibi_slope=alibi_slope)
    assert B.shape == (H, n, n)
    mask = prior_causal(n, n)  # 0 on j<=i, -inf on j>i
    slope_h = alibi_slope * (2.0 ** (-8.0 * 1 / 1))   # _press_slopes(1, alibi_slope)[0]
    for i in range(n):
        for j in range(n):
            if j <= i:
                assert mask[i, j] == 0.0  # masking construction reused verbatim
                expected = -slope_h * (i - j)
                assert torch.isclose(B[0, i, j], torch.tensor(expected), atol=1e-6)


def test_causal_alibi_is_not_symmetric():
    # (4) directional bias under the mask -> NOT symmetric per-head (alibi is symmetric, causal_alibi is not).
    H, n = 1, 4
    B = attention_log_prior("causal_alibi", n, n, n_heads=H, alibi_slope=0.5)
    assert B.shape == (H, n, n)
    assert not torch.equal(B[0], B[0].T)
    # Concretely: below-diagonal entry is finite while its transpose is -inf.
    assert torch.isfinite(B[0, 2, 0]) and torch.isneginf(B[0, 0, 2])


def test_default_slope_matches_alibi():
    # Default n_heads=1, alibi_slope=0.5: Press slope for h=1,H=1 is 2^(-8), scaled by base 0.5.
    # On the diagonal dist=0 so bias is 0; one step below: -0.5 * 2^(-8) * 1.
    import math
    H, n = 1, 3
    B = attention_log_prior("causal_alibi", n, n)   # default n_heads=1, alibi_slope=0.5
    assert B.shape == (H, n, n)
    assert B[0, 1, 1].item() == 0.0
    expected = -(0.5 * 2.0 ** -8.0)                 # _press_slopes(1, 0.5)[0] * 1 step
    assert torch.isclose(B[0, 1, 0], torch.tensor(expected), atol=1e-6)


def test_existing_priors_unchanged_regression_guard():
    # (5) regression guard: recompute uniform / causal / alibi and pin behavior.
    # uniform and causal are still (N,N); alibi/causal_alibi are now (H,N,N).
    assert torch.equal(attention_log_prior("uniform", 4, 4), torch.zeros(4, 4))

    Bc = attention_log_prior("causal", 3, 3)
    assert torch.isneginf(Bc[0, 1]) and torch.isneginf(Bc[0, 2]) and torch.isneginf(Bc[1, 2])
    assert Bc[2, 0] == 0.0 and Bc[1, 1] == 0.0 and Bc[2, 2] == 0.0

    import math
    H, N = 1, 4
    alibi_slope = 0.5
    Ba = attention_log_prior("alibi", N, N, n_heads=H, alibi_slope=alibi_slope)
    assert Ba.shape == (H, N, N)
    slope_h = alibi_slope * (2.0 ** (-8.0 * 1 / 1))
    for i in range(N):
        for j in range(N):
            # alibi is SYMMETRIC with no causal mask (per head).
            assert torch.isclose(Ba[0, i, j], torch.tensor(-slope_h * abs(i - j)), atol=1e-6)
    assert torch.equal(Ba[0], Ba[0].T)  # symmetric per head: distinguishes it from causal_alibi
