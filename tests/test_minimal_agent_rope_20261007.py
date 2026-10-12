r"""Comparison-space RoPE on the minimal agent (October 7, 2026).

The record of a source at offset j - i arrives through the known channel R_{j-i}: labels depend on position only
through j - i, the result is invariant under a common shift of every position (so the receiver statistics are
rotated back correctly), and R = I is the unrotated model.
"""

import warnings

import pytest
import torch

from tests.test_audit_20261007_opus_fixes import _cfg, _minimal_agent
from vfe3.geometry.rope import build_rope_rotation
from vfe3.model.minimal_agent import minimal_agent_readout

H, D, NP = 2, 4, 6
K = H * D


@pytest.fixture(autouse=True)
def _one_thread() -> None:
    torch.set_num_threads(1)


def _inputs(seed: int, constant: bool = False):
    g = torch.Generator().manual_seed(seed)
    xi = torch.randn(1, NP, K, generator=g, dtype=torch.float64)
    X = torch.randn(1, NP, K, K, generator=g, dtype=torch.float64)
    C = X @ X.transpose(-1, -2) / K + 0.5 * torch.eye(K, dtype=torch.float64)
    if constant:
        xi, C = xi[:, :1].expand(1, NP, K), C[:, :1].expand(1, NP, K, K)
    S = torch.randn(H, D, K, generator=g, dtype=torch.float64)
    T = torch.randn(H, D, K, generator=g, dtype=torch.float64)
    log_noise = torch.full((H,), -0.5, dtype=torch.float64)
    causal = torch.zeros(NP, NP, dtype=torch.float64).masked_fill(
        torch.triu(torch.ones(NP, NP, dtype=torch.bool), 1), float("-inf"))
    return xi, C, S, T, log_noise, causal


def _rope(offset: int = 0, base: float = 10.0, fraction: float = 1.0) -> torch.Tensor:
    return build_rope_rotation(torch.arange(NP) + offset, [D], base=base, fraction=fraction, dtype=torch.float64)


@pytest.mark.parametrize("inference", ["exact", "mean_field"])
def test_labels_depend_only_on_relative_position(inference: str) -> None:
    xi, C, S, T, log_noise, causal = _inputs(0, constant=True)
    out = minimal_agent_readout(xi, C, S, T, log_noise, causal, inference, rope=_rope())
    # Exact ADF: head h > 0 reads each receiver's moments after the earlier heads, which differ across receivers
    # (receiver i had i + 1 sources) even without RoPE; head 0, and every mean-field head, reads the input moments.
    heads = slice(0, 1) if inference == "exact" else slice(None)
    log_w = out.weights[0, heads].log()                                          # (H', N, N)
    for m in range(1, NP):
        gaps = torch.stack([log_w[:, i, i - m] - log_w[:, i, i] for i in range(m, NP)], dim=-1)
        assert torch.allclose(gaps, gaps[..., :1].expand_as(gaps), atol=1e-10), m
    plain = minimal_agent_readout(xi, C, S, T, log_noise, causal, inference)
    assert not torch.allclose(out.weights, plain.weights)                        # the rotation acts


@pytest.mark.parametrize("inference", ["exact", "mean_field"])
@pytest.mark.parametrize("fraction", [1.0, 0.5])
def test_common_position_shift_leaves_the_posterior_unchanged(inference: str, fraction: float) -> None:
    r"""R_i -> R_{i+c}: the record channel R_{j-i} and hence every output are unchanged."""
    xi, C, S, T, log_noise, causal = _inputs(1)
    a = minimal_agent_readout(xi, C, S, T, log_noise, causal, inference, diagnostics=True,
                              rope=_rope(fraction=fraction))
    b = minimal_agent_readout(xi, C, S, T, log_noise, causal, inference, diagnostics=True,
                              rope=_rope(offset=17, fraction=fraction))
    for name in ("xi", "gains", "M", "weights", "gain_fraction", "covariance_contraction"):
        assert torch.allclose(getattr(a, name), getattr(b, name), rtol=1e-9, atol=1e-10), name
    if inference == "mean_field":
        assert torch.allclose(a.label_staleness, b.label_staleness, rtol=1e-8, atol=1e-10)
    else:
        assert torch.allclose(a.spread_share, b.spread_share, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("inference", ["exact", "mean_field"])
def test_identity_rotation_is_the_unrotated_model(inference: str) -> None:
    xi, C, S, T, log_noise, causal = _inputs(2)
    eye = torch.eye(D, dtype=torch.float64).expand(NP, D, D)
    a = minimal_agent_readout(xi, C, S, T, log_noise, causal, inference, rope=eye)
    b = minimal_agent_readout(xi, C, S, T, log_noise, causal, inference)
    for name in ("xi", "gains", "M", "weights", "cost"):
        assert torch.allclose(getattr(a, name), getattr(b, name), rtol=1e-12, atol=1e-12), name


def test_one_source_posterior_matches_conditioning_on_the_rotated_record() -> None:
    r"""One head, receiver i with a single admissible source j: xi^+ is the Kalman update on
    o = S X_i - R_{j-i} T X_j + eps, computed directly."""
    xi, C, S, T, log_noise, _ = _inputs(3)
    S, T, log_noise = S[:1], T[:1], log_noise[:1]
    i, j = 4, 1
    prior = torch.full((NP, NP), float("-inf"), dtype=torch.float64)
    prior[torch.arange(NP), torch.arange(NP)] = 0.0
    prior[i, i], prior[i, j] = float("-inf"), 0.0
    R = _rope()
    out = minimal_agent_readout(xi, C, S, T, log_noise, prior, "exact", rope=R)
    Rji = R[i].T @ R[j]
    a, A = S[0] @ xi[0, i], S[0] @ C[0, i] @ S[0].T
    b, B = Rji @ T[0] @ xi[0, j], Rji @ T[0] @ C[0, j] @ T[0].T @ Rji.T + log_noise.exp() * torch.eye(D, dtype=torch.float64)
    G = C[0, i] @ S[0].T @ torch.linalg.inv(A + B)
    assert torch.allclose(out.xi[0, i], xi[0, i] + G @ (b - a), rtol=1e-10, atol=1e-12)


def test_config_accepts_comparison_rope_on_the_minimal_agent() -> None:
    from vfe3.model.model import VFEModel

    for inference in ("exact", "mean_field"):
        cfg = _cfg(_minimal_agent(rope_mode="comparison",
                                  minimal_agent_inference=inference))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.manual_seed(0)
            model = VFEModel(cfg)
        tokens = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
        _, loss, _ = model(tokens, tokens)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    with pytest.raises(ValueError, match="rope_mode='frame'"):
        _cfg(_minimal_agent(rope_mode="frame"))
