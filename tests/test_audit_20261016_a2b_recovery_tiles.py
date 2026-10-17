r"""Recovery pair tiles of the 2026-10-16 audit (F28), tiny CPU shapes.

A pair touching a recovered entry is scored by the factor-route triangular-solve tile. Those tiles now read only the
(batch, head) slices that hold a recovered entry, gathered as the GEMM-fallback branch gathers its slices, and their
size follows the gathered slice count (``_RECOVERY_PAIR_TILE_BYTES``). Healthy pairs stay the GEMM values bit for bit,
and a recovered slice is scored as in a batch that holds it alone.
"""
import math
from typing import Any, List, Tuple

import pytest
import torch

import vfe3.relational_scoring as scoring
from vfe3.relational_scoring import projected_gaussian_kl, projected_moments

B, N, H, D_HEAD, R = 3, 6, 2, 4, 3
FLAGGED = (1, 2, 0)                                                              # (b, i, h), ill-conditioned


def _sources(dtype: torch.dtype = torch.float32, seed: int = 0, tiny: float = 1e-5) -> List[torch.Tensor]:
    r"""Well-conditioned sources (orthonormal comparison rows) with one ill-conditioned entry, kappa(G) ~ 1 / tiny."""
    g = torch.Generator().manual_seed(seed)
    mu = torch.randn(B, N, H, D_HEAD, generator=g, dtype=torch.float64)
    sigma = torch.rand(B, N, H, D_HEAD, generator=g, dtype=torch.float64) + .5
    sigma[FLAGGED] = torch.tensor([1.] + [tiny] * (D_HEAD - 1), dtype=torch.float64)
    inverse = torch.eye(D_HEAD, dtype=torch.float64).expand(B, N, H, D_HEAD, D_HEAD).clone()
    Q, _ = torch.linalg.qr(torch.randn(H, D_HEAD, D_HEAD, generator=g, dtype=torch.float64))
    S = Q[:, :R, :]                                                              # (H, r, d), singular values 1
    return [x.to(dtype).clone().requires_grad_() for x in (mu, sigma, inverse, S)]


def _score(sources: List[torch.Tensor], recovery: bool) -> torch.Tensor:
    mu, sigma, inverse, S = sources
    a, G = projected_moments(mu, sigma, inverse, S)
    projection = (sigma, inverse, S, mu)
    previous = scoring.set_projected_condition_recovery(recovery)
    try:
        return projected_gaussian_kl(a, G, a, G, projection_q=projection, projection_k=projection,
                                     symmetric_inputs=True)                     # (B, H, N, N)
    finally:
        scoring.set_projected_condition_recovery(previous)


def _factor_tile_spy(monkeypatch: pytest.MonkeyPatch) -> List[Tuple[int, ...]]:
    shapes: List[Tuple[int, ...]] = []
    real = scoring._projected_kl_tile

    def spy(*args: Any) -> torch.Tensor:
        if args[-1]:
            shapes.append(tuple(args[0].shape))
        return real(*args)
    monkeypatch.setattr(scoring, "_projected_kl_tile", spy)
    return shapes


def _touched() -> torch.Tensor:                                                  # (B, H, N, N)
    b, i, h = FLAGGED
    mask = torch.zeros(B, H, N, N, dtype=torch.bool)
    mask[b, h, i, :] = True
    mask[b, h, :, i] = True
    return mask


def test_recovery_tiles_read_only_the_recovered_slices(monkeypatch: pytest.MonkeyPatch) -> None:
    shapes = _factor_tile_spy(monkeypatch)
    before = scoring.source_recovery_elements()
    D = _score(_sources(), recovery=True)
    assert scoring.source_recovery_elements() - before == 2                     # precondition: the entry, per role
    assert shapes and all(math.prod(shape[:-2]) == 1 for shape in shapes), shapes
    assert torch.isfinite(D).all()


def test_recovery_tile_size_follows_the_gathered_slice_count(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""A budget of one slice's full pair grid is one tile; one slice's row of pairs is N tiles."""
    shapes = _factor_tile_spy(monkeypatch)
    element = torch.finfo(torch.float32).bits // 8
    monkeypatch.setattr(scoring, "_RECOVERY_PAIR_TILE_BYTES", element * R * R * N * N, raising=False)
    _score(_sources(), recovery=True)
    assert len(shapes) == 1
    shapes.clear()
    monkeypatch.setattr(scoring, "_RECOVERY_PAIR_TILE_BYTES", element * R * R * N, raising=False)
    _score(_sources(), recovery=True)
    assert len(shapes) == N and all(shape[-2] == N for shape in shapes), shapes


def test_healthy_pairs_keep_the_gemm_and_recovered_pairs_match_float64() -> None:
    healthy = _score(_sources(), recovery=False)                                # nothing flagged: the GEMM grid
    recovered = _score(_sources(), recovery=True)
    touched = _touched()
    assert torch.equal(recovered[~touched], healthy[~touched])
    reference = _score(_sources(torch.float64), recovery=False)                  # float64: no entry fails
    error = (recovered.double() - reference)[touched].abs().max() / reference[touched].abs().max()
    assert float(error) < 1e-4


def test_recovered_slice_is_scored_as_in_a_batch_that_holds_it_alone() -> None:
    sources = _sources()
    full = _score(sources, recovery=True)
    b = FLAGGED[0]
    mu, sigma, inverse, S = sources                                              # S: (H, r, d) has no batch axis
    alone = _score([*(x[b:b + 1].detach().clone().requires_grad_() for x in (mu, sigma, inverse)), S],
                   recovery=True)
    assert torch.equal(full[b:b + 1], alone)


def test_recovery_tiles_gradcheck_float64() -> None:
    r"""Float64 inputs whose flagged Gram lost SPD: the gathered float64 factor tiles carry exact gradients."""
    g = torch.Generator().manual_seed(11)
    mu = torch.randn(2, 3, 2, 3, generator=g, dtype=torch.float64).requires_grad_()
    sigma = (torch.rand(2, 3, 2, 3, generator=g, dtype=torch.float64) + .5).requires_grad_()
    inverse = torch.eye(3, dtype=torch.float64).expand(2, 3, 2, 3, 3).clone()
    S = torch.randn(2, 2, 3, generator=g, dtype=torch.float64).requires_grad_()

    def score(mu: torch.Tensor, sigma: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        a, G = projected_moments(mu, sigma, inverse, S)
        lost = torch.zeros(G.shape[:-2], dtype=torch.bool)
        lost[1, 2, 0] = True                                                     # one slice, (b, h) = (1, 0)
        G = torch.where(lost[..., None, None], -G, G)                            # negative definite: Cholesky fails
        projection = (sigma, inverse, S, mu)
        return projected_gaussian_kl(a, G, a, G, projection_q=projection, projection_k=projection,
                                     symmetric_inputs=True)

    torch.autograd.gradcheck(score, (mu, sigma, S), eps=1e-6, atol=1e-6, rtol=1e-5)
