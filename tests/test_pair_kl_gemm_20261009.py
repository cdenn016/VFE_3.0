"""Healthy projected pair KL as one centered expanded GEMM (audit 2026-10-09, general sweep: core math).

2 KL_ij = <G_i + alpha_i alpha_i^T, P_j>_F - alpha_i^T (P_j + P_j^T) b_j + b_j^T P_j b_j + log det G_j - r - log det G_i,
with means centered on the first key mean. Checked against a float64 direct full-Gaussian KL on tiny inputs,
including ill-conditioned covariances and mean offsets far larger than the spread, by float64 gradcheck on
both the plain and the checkpointed route, and by the softmax it feeds. Tiny CPU shapes only.
"""
import math
from typing import Tuple

import pytest
import torch

import vfe3.relational_scoring as scoring
from vfe3.relational_scoring import projected_gaussian_kl


def _spd(shape: Tuple[int, ...], r: int, cond: float, g: torch.Generator) -> torch.Tensor:
    Q, _ = torch.linalg.qr(torch.randn(*shape, r, r, generator=g, dtype=torch.float64))
    spectrum = torch.logspace(0, math.log10(cond), r, dtype=torch.float64).expand(*shape, r)
    G = (Q * spectrum.unsqueeze(-2)) @ Q.transpose(-1, -2)
    return .5 * (G + G.transpose(-1, -2))


def _direct_kl64(a: torch.Tensor, G: torch.Tensor, b: torch.Tensor, Gk: torch.Tensor) -> torch.Tensor:
    r"""Direct float64 KL(N(a_i, G_i) || N(b_j, Gk_j)) on (B, N, H, r) inputs, returned (B, H, Nq, Nk)."""
    a, G, b, Gk = (x.double() for x in (a, G, b, Gk))
    r = a.shape[-1]
    P = torch.linalg.inv(Gk)
    delta = a.unsqueeze(-3) - b.unsqueeze(-4)                                   # (B, Nq, Nk, H, r)
    trace = torch.einsum("bihrs,bjhsr->bhij", G, P)
    quadratic = torch.einsum("bijhr,bjhrs,bijhs->bhij", delta, P, delta)
    logdet_q = torch.linalg.slogdet(G)[1].transpose(-1, -2)                     # (B, H, Nq)
    logdet_k = torch.linalg.slogdet(Gk)[1].transpose(-1, -2)
    return .5 * (trace + quadratic - r + logdet_k.unsqueeze(-2) - logdet_q.unsqueeze(-1))


def _inputs(cond: float, offset: float, seed: int = 0, B: int = 2, N: int = 6, H: int = 2, r: int = 4):
    g = torch.Generator().manual_seed(seed)
    shift = offset * torch.randn(1, 1, H, r, generator=g, dtype=torch.float64)
    a = (shift + torch.randn(B, N, H, r, generator=g, dtype=torch.float64)).float()
    b = (a.double() + .5 * torch.randn(B, N, H, r, generator=g, dtype=torch.float64)).float()
    G = _spd((B, N, H), r, cond, g).float()
    Gk = _spd((B, N, H), r, cond, g).float()
    return a, .5 * (G + G.transpose(-1, -2)), b, .5 * (Gk + Gk.transpose(-1, -2))


def _uncentered_gemm(aq, ak, q_cov, k_prec, logdet_q, logdet_k):
    r = aq.shape[-1]
    Pb = (k_prec @ ak.unsqueeze(-1)).squeeze(-1)
    PTb = (k_prec.transpose(-1, -2) @ ak.unsqueeze(-1)).squeeze(-1)
    query = torch.cat(((q_cov + aq.unsqueeze(-1) * aq.unsqueeze(-2)).flatten(-2), aq), dim=-1)
    key = torch.cat((k_prec.flatten(-2), -(Pb + PTb)), dim=-1)
    return .5 * (query @ key.transpose(-1, -2) + ((ak * Pb).sum(-1) + logdet_k).unsqueeze(-2)
                 - r - logdet_q.unsqueeze(-1))


@pytest.mark.parametrize("cond, offset", [(10., 0.), (10., 1e3), (1e3, 0.), (1e3, 1e3)])
def test_centered_gemm_matches_float64_direct_kl(cond: float, offset: float) -> None:
    a, G, b, Gk = _inputs(cond, offset)
    reference = _direct_kl64(a, G, b, Gk)
    error = float((projected_gaussian_kl(a, G, b, Gk).double() - reference).abs().max())
    # fp32 rounding of the key precision dominates: eps_32 kappa times the KL scale, both routes.
    scale = float(reference.abs().max()) * cond * torch.finfo(torch.float32).eps
    assert error <= 16 * scale + 1e-5, (error, scale)


def test_centering_bounds_cancellation_at_large_mean_offsets(monkeypatch: pytest.MonkeyPatch) -> None:
    a, G, b, Gk = _inputs(10., 1e3, seed=3)
    reference = _direct_kl64(a, G, b, Gk)
    centered = float((projected_gaussian_kl(a, G, b, Gk).double() - reference).abs().max())
    monkeypatch.setattr(scoring, "_projected_kl_gemm", _uncentered_gemm)
    uncentered = float((projected_gaussian_kl(a, G, b, Gk).double() - reference).abs().max())
    assert centered < 1e-4
    assert uncentered > 100 * centered, (centered, uncentered)


def test_softmax_rows_and_common_translation_are_unaffected() -> None:
    a, G, b, Gk = _inputs(1e2, 0., seed=5)
    reference = _direct_kl64(a, G, b, Gk)
    kl = projected_gaussian_kl(a, G, b, Gk)
    torch.testing.assert_close((-kl.double()).softmax(-1), (-reference).softmax(-1), rtol=0, atol=1e-5)
    # A common translation of every mean leaves each KL unchanged up to the rounding of the shifted inputs.
    t = torch.full((a.shape[-1],), 1e3)
    shifted = projected_gaussian_kl(a + t, G, b + t, Gk)
    reference_shifted = _direct_kl64(a + t, G, b + t, Gk)
    assert float((shifted.double() - reference_shifted).abs().max()) < 1e-4


@pytest.mark.parametrize("checkpointed", [False, True])
def test_pair_gemm_gradcheck_float64(monkeypatch: pytest.MonkeyPatch, checkpointed: bool) -> None:
    calls = []
    real = scoring.checkpoint

    def spy(function, *args, **kwargs):
        calls.append(getattr(function, "__name__", ""))
        return real(function, *args, **kwargs)

    monkeypatch.setattr(scoring, "checkpoint", spy)
    if checkpointed:
        monkeypatch.setattr(scoring, "_PROJECTION_TILE_BYTES", 1)
    g = torch.Generator().manual_seed(7)
    B, N, H, r = 1, 3, 2, 2
    a = torch.randn(B, N, H, r, generator=g, dtype=torch.float64, requires_grad=True)
    b = torch.randn(B, N, H, r, generator=g, dtype=torch.float64, requires_grad=True)
    X = torch.randn(B, N, H, r, r, generator=g, dtype=torch.float64, requires_grad=True)
    Y = torch.randn(B, N, H, r, r, generator=g, dtype=torch.float64, requires_grad=True)
    eye = torch.eye(r, dtype=torch.float64)

    def cost(a, b, X, Y):
        Gq = X @ X.transpose(-1, -2) + eye
        Gk = Y @ Y.transpose(-1, -2) + eye
        return projected_gaussian_kl(a, .5 * (Gq + Gq.transpose(-1, -2)), b, .5 * (Gk + Gk.transpose(-1, -2)))

    assert torch.autograd.gradcheck(cost, (a, b, X, Y), eps=1e-6, atol=1e-7, rtol=1e-6)
    # The GEMM saves only its token-linear inputs at every size (_IEEEFP32Recompute, audit 2026-10-10 round 2,
    # item 2), so it no longer needs the checkpoint above the projection budget.
    assert "_projected_kl_gemm" not in calls


def test_symmetric_inputs_certificate_skips_only_the_symmetry_check() -> None:
    a, G, b, Gk = _inputs(10., 0., seed=9)
    torch.testing.assert_close(projected_gaussian_kl(a, G, b, Gk, symmetric_inputs=True),
                               projected_gaussian_kl(a, G, b, Gk), rtol=0, atol=0)
    skew = G.clone()
    skew[..., 0, 1] += 1e-2
    with pytest.raises(ValueError, match="symmetric"):
        projected_gaussian_kl(a, skew, b, Gk)
