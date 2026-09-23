from collections.abc import MutableMapping

import pytest
import torch

from vfe3.attention_prior import attention_log_prior, register_prior


_MISSING = object()


def _restore_registry_entry(
    registry: MutableMapping[str, object],
    name:     str,
    previous: object,
) -> None:
    if previous is _MISSING:
        registry.pop(name, None)
    else:
        registry[name] = previous


def test_uniform_is_zero_bias():
    B = attention_log_prior("uniform", 4, 4)
    assert torch.allclose(B, torch.zeros(4, 4))


def test_causal_masks_future_keys():
    B = attention_log_prior("causal", 3, 3)
    # j > i masked (-inf), j <= i allowed (0)
    assert torch.isneginf(B[0, 1]) and torch.isneginf(B[0, 2]) and torch.isneginf(B[1, 2])
    assert B[2, 0] == 0.0 and B[1, 1] == 0.0 and B[2, 2] == 0.0


def test_alibi_is_linear_in_distance():
    # n_heads=1 -> (1, N, N); Press slope for h=1, H=1: 2^(-8) * alibi_slope
    H, N = 1, 4
    B = attention_log_prior("alibi", N, N, n_heads=H, alibi_slope=1.0)
    assert B.shape == (H, N, N)
    import math
    slope = 1.0 * (2.0 ** (-8.0 * 1 / 1))          # _press_slopes(1, 1.0)[0]
    for i in range(N):
        for j in range(N):
            assert torch.isclose(B[0, i, j], torch.tensor(-slope * abs(i - j)), atol=1e-6)


@pytest.mark.registry_mutation
def test_new_prior_with_novel_kwarg_reachable_without_editing_dispatcher():
    # Modularity: a new prior's OWN param must flow through the dispatcher's **kwargs
    # (not a hard-coded slope union), so it selects-with-config without editing the call site.
    from vfe3.attention_prior import _PRIORS
    name = "_test_windowed"
    previous = _PRIORS.get(name, _MISSING)
    try:
        @register_prior(name, override=previous is not _MISSING)
        def _windowed(n_query, n_key, *, width=1, **kwargs):
            i = torch.arange(n_query).unsqueeze(-1)
            j = torch.arange(n_key).unsqueeze(0)
            return torch.where((i - j).abs() <= width, 0.0, float("-inf"))

        B = attention_log_prior(name, 4, 4, width=1)
        assert B[0, 0] == 0.0 and B[0, 1] == 0.0
        assert torch.isneginf(B[0, 2])
    finally:
        _restore_registry_entry(_PRIORS, name, previous)
