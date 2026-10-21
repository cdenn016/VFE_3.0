r"""The per-step cost of the frame condition bound ``phi_frame_condition_max`` (2026-10-21).

A live K=128 run (four heads of d = 32, V = 50257) slowed from 252 to over 600 ms per step within 1000 steps once the
bound was on. Three causes are pinned here. Rows inside the bound that fail the Frobenius screen were eigendecomposed on
every call; a Cholesky window certificate around the midpoint cached from a row's last eigendecomposition now passes them,
so such a row reaches the eigensolver at most once while it stays put. The spread projection ran a 64-step bisection;
it is now the exact root of a piecewise-linear function, in a fixed number of operations. The flagged rows were
processed in 341-row chunks with host reads in every chunk, and the float32 eigendecomposition certificate ran its
products under the process TF32 policy, which fails it almost always and sends every flagged row to float64; the host
reads are now a fixed number per table and the bound runs under IEEE fp32 matmuls.

Tiny CPU models only (K = 6, two heads of d = 3), plus one CUDA-lane test at the live head shape, registered in
``tests/pytest_policy.py::CUDA_TESTS``; it skips without CUDA.
"""
import math
import os
from typing import Any, Callable, Dict, List, Optional

import pytest
import torch
from torch.overrides import TorchFunctionMode

from tests.profiles import test_config
from vfe3.gauge_optim import (
    _bounded_traceless_spectrum,
    _frame_bound_rows,
    _window_certified,
    bound_phi_frame_condition_,
)
from vfe3.model.model import VFEModel

KAPPA = 20.0
C = 0.5 * math.log(KAPPA)          # pos_phi="none": the token factor's half-width log(kappa)/2

# Tensor operations that read device memory on the host (a CUDA synchronization).
_HOST_READS = frozenset({"nonzero", "argwhere", "masked_select", "unique", "__bool__", "item", "tolist", "cpu"})


class _TorchCalls(TorchFunctionMode):
    r"""Count torch calls, and separately those that read device memory on the host."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.host_reads = 0

    def __torch_function__(self, func: Callable, types: Any, args: tuple = (), kwargs: Optional[Dict] = None) -> Any:
        name = getattr(func, "__name__", "")
        self.calls += 1
        index = args[1] if name in ("__getitem__", "__setitem__") and len(args) > 1 else None
        indices = index if isinstance(index, tuple) else (index,)
        if name in _HOST_READS or any(isinstance(i, torch.Tensor) and i.dtype == torch.bool for i in indices):
            self.host_reads += 1
        return func(*args, **(kwargs or {}))


def _model(vocab_size: int = 6) -> VFEModel:
    return VFEModel(test_config(vocab_size=vocab_size, embed_dim=6, n_heads=2, max_seq_len=3, gauge_group="block_glk",
                                pos_phi="none", phi_frame_condition_max=KAPPA))


def _rows_with_spectrum(
    spectrum:  List[float],          # traceless symmetric spectrum of every head block
    n_rows:    int,
    H:         int,

    generator: torch.Generator,
) -> torch.Tensor:                   # (n_rows, H d d) float64 row-major head blocks
    d = len(spectrum)
    Q, _ = torch.linalg.qr(torch.randn(n_rows, H, d, d, generator=generator, dtype=torch.float64))
    lam = torch.tensor(spectrum, dtype=torch.float64).expand(n_rows, H, d)
    S0 = Q @ torch.diag_embed(lam) @ Q.mT
    skew = torch.randn(n_rows, H, d, d, generator=generator, dtype=torch.float64)
    X = 0.5 * (S0 + S0.mT) + 0.5 * (skew - skew.mT) + 0.3 * torch.eye(d, dtype=torch.float64)
    return X.reshape(n_rows, -1)


def _stretched_model() -> VFEModel:
    model = _model(vocab_size=12)
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        phi = model.prior_bank.phi_embed
        phi.copy_(1e-3 * torch.randn(phi.shape, generator=generator))           # inside the bound
        phi[::3].copy_(2.0 * torch.randn(phi[::3].shape, generator=generator))   # every third row far outside
    return model


def _float64_reference(model: VFEModel) -> torch.Tensor:
    phi = model.prior_bank.phi_embed.detach()
    new, over, _ = _frame_bound_rows(phi, None, None, 2, 3, C, torch.float64, store_dtype=phi.dtype)
    return torch.where(over[:, None], new.to(phi.dtype), phi)


def _uncertified_float32_eigh(monkeypatch: pytest.MonkeyPatch) -> List[torch.dtype]:
    r"""Make every float32 eigendecomposition fail its orthogonality certificate; return the eigh dtypes called."""
    eigh, dtypes = torch.linalg.eigh, []

    def corrupted(A: torch.Tensor) -> tuple:
        lam, Q = eigh(A)
        dtypes.append(A.dtype)
        return (lam, 1.01 * Q) if A.dtype == torch.float32 else (lam, Q)

    monkeypatch.setattr(torch.linalg, "eigh", corrupted)
    return dtypes


def test_rows_inside_the_bound_reach_the_eigensolver_at_most_once(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model()
    a = 0.9 * 2.0 * C / 3.0                                     # spectrum (a, a, -2a): spread 0.9 * 2c
    assert math.sqrt(2.0) * math.sqrt(6.0) * a > 2.0 * C        # yet sqrt(2) ||S_0||_F > 2c: the screen flags it
    with torch.no_grad():
        phi = model.prior_bank.phi_embed
        phi.copy_(_rows_with_spectrum([a, a, -2.0 * a], phi.shape[0], 2, torch.Generator().manual_seed(0)))
    before = phi.detach().clone()
    eigh, decomposed = torch.linalg.eigh, []

    def counting(A: torch.Tensor) -> tuple:
        decomposed.append(A.numel() // 9)
        return eigh(A)

    monkeypatch.setattr(torch.linalg, "eigh", counting)
    assert int(bound_phi_frame_condition_(model, KAPPA)) == 0
    first = sum(decomposed)
    decomposed.clear()
    for _ in range(3):
        assert int(bound_phi_frame_condition_(model, KAPPA)) == 0
    assert first > 0                                            # the first window, centered at 0, misses -2a = -1.2c
    assert sum(decomposed) == 0                                 # later windows, at the cached midpoints, certify
    assert torch.equal(phi.detach(), before)


def test_float32_route_runs_under_ieee_fp32_matmul(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _stretched_model()
    eigh, policies = torch.linalg.eigh, []

    def recording(A: torch.Tensor) -> tuple:
        policies.append(torch.get_float32_matmul_precision())
        return eigh(A)

    monkeypatch.setattr(torch.linalg, "eigh", recording)
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        bound_phi_frame_condition_(model, KAPPA)
        assert torch.get_float32_matmul_precision() == "high"   # restored on exit
    finally:
        torch.set_float32_matmul_precision(previous)
    assert policies and set(policies) == {"highest"}


@pytest.mark.parametrize("temporary_bytes", [1, 1 << 30])
def test_host_reads_per_call_do_not_grow_with_the_chunk_count(monkeypatch: pytest.MonkeyPatch,
                                                              temporary_bytes: int) -> None:
    r"""One-row chunks and one chunk read the device the same number of times (at most three per table), with every
    float32 block uncertified, and the result is the float64 projection."""
    model = _stretched_model()
    reference = _float64_reference(model)
    dtypes = _uncertified_float32_eigh(monkeypatch)
    with _TorchCalls() as calls:
        bound_phi_frame_condition_(model, KAPPA, temporary_bytes=temporary_bytes)
    assert calls.host_reads <= 3
    assert torch.float32 in dtypes and torch.float64 in dtypes
    torch.testing.assert_close(model.prior_bank.phi_embed.detach(), reference, rtol=0.0, atol=1e-6)


def _bisection(lam: torch.Tensor, c: torch.Tensor, steps: int = 200) -> torch.Tensor:
    r"""The earlier bisection for the window center, run to convergence in float64."""
    lo, hi = lam[..., :1] + c, lam[..., -1:] - c
    for _ in range(steps):
        t = 0.5 * (lo + hi)
        below = torch.clamp(lam, t - c, t + c).sum(-1, keepdim=True) < 0.0
        lo, hi = torch.where(below, t, lo), torch.where(below, hi, t)
    x = torch.clamp(lam, 0.5 * (lo + hi) - c, 0.5 * (lo + hi) + c)
    return x - x.mean(-1, keepdim=True)


def test_spread_projection_is_the_exact_root_in_a_fixed_number_of_operations() -> None:
    generator = torch.Generator().manual_seed(4)
    scale = torch.logspace(-2, 1, 500, dtype=torch.float64)[:, None, None]
    lam = torch.randn(500, 2, 7, generator=generator, dtype=torch.float64) * scale
    lam = (lam - lam.mean(-1, keepdim=True)).sort(-1).values                    # (500, 2, 7) zero-sum, ascending
    spread = lam[..., -1] - lam[..., 0]
    c = (0.5 * spread * torch.rand(500, 2, generator=generator, dtype=torch.float64))[..., None]   # below half-spread
    with _TorchCalls() as calls:
        x = _bounded_traceless_spectrum(lam, c)
    assert calls.calls <= 60                                    # fixed (37 now); the bisection made 587
    torch.testing.assert_close(x, _bisection(lam, c), rtol=0.0, atol=1e-12)
    assert bool(((x[..., -1] - x[..., 0]) <= 2.0 * c[..., 0] * (1 + 1e-12)).all())
    torch.testing.assert_close(x.sum(-1), torch.zeros_like(spread), rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("d", [3, 32])
def test_window_certificate_is_sound_and_sharp(d: int) -> None:
    r"""A certified window contains the exact spectrum of the float32 block; a window with 1% slack is certified."""
    generator = torch.Generator().manual_seed(5)
    n = 400
    lam = torch.randn(n, 1, d, generator=generator, dtype=torch.float64) * torch.logspace(-2, 1, n)[:, None, None]
    lam = lam - lam.mean(-1, keepdim=True)
    Q, _ = torch.linalg.qr(torch.randn(n, 1, d, d, generator=generator, dtype=torch.float64))
    S = Q @ torch.diag_embed(lam) @ Q.mT
    S0 = (0.5 * (S + S.mT)).to(torch.float32)                                   # (n, 1, d, d)
    exact = torch.linalg.eigvalsh(S0.to(torch.float64))                         # spectrum of the stored block
    mid, half = 0.5 * (exact[..., 0] + exact[..., -1]), 0.5 * (exact[..., -1] - exact[..., 0])   # (n, 1)

    t = (mid + 0.3 * half * torch.randn(n, 1, generator=generator, dtype=torch.float64)).to(torch.float32)
    c = (half * (0.5 + torch.rand(n, 1, generator=generator, dtype=torch.float64))).to(torch.float32)
    certified = _window_certified(S0, t, c)
    t64, c64 = t.to(torch.float64), c.to(torch.float64)
    inside = (exact[..., 0] >= t64 - c64) & (exact[..., -1] <= t64 + c64)
    assert bool(certified.any()) and not bool((certified & ~inside).any())

    assert bool(_window_certified(S0, mid.to(torch.float32), (1.01 * half).to(torch.float32)).all())
    assert not bool(_window_certified(S0, mid.to(torch.float32), (half * (1 - 1e-5)).to(torch.float32)).any())


class _TokenTable(torch.nn.Module):
    r"""The two attributes the bound reads: a block group and a trainable token frame table, no positional factor."""

    def __init__(self, group: Any, phi: torch.Tensor) -> None:
        super().__init__()
        self.group = group
        self.prior_bank = torch.nn.Module()
        self.prior_bank.phi_embed = torch.nn.Parameter(phi)


def test_cuda_frame_bound_at_the_live_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Four heads of d = 32 under the live ``fp32_matmul_precision='high'``: no float64 recomputation, rows inside
    the bound are not decomposed again, the bound holds, and the host reads are three plus one per eigensolver call."""
    if os.environ.get("VFE3_TEST_DEVICE", "cpu") != "cuda" or not torch.cuda.is_available():
        pytest.skip("CUDA lane only")
    import warnings

    from vfe3.geometry.groups import get_group
    H, d, n = 4, 32, 3072
    generator = torch.Generator().manual_seed(6)
    tiny = 0.01 * torch.randn(n, H * d * d, generator=generator, dtype=torch.float64)
    u = torch.rand(n, H, d, generator=generator, dtype=torch.float64).sort(-1).values
    u = (u - u[..., :1]) / (u[..., -1:] - u[..., :1])                          # spectra on [0, 1], asymmetric
    Q, _ = torch.linalg.qr(torch.randn(n, H, d, d, generator=generator, dtype=torch.float64))

    def blocks(spread: float) -> torch.Tensor:
        lam = spread * (u - u.mean(-1, keepdim=True))
        S0 = Q @ torch.diag_embed(lam) @ Q.mT
        return (0.5 * (S0 + S0.mT)).reshape(n, -1)

    phi = torch.cat((tiny, blocks(0.8 * 2.0 * C), blocks(1.5 * 2.0 * C))).to(torch.float32)
    n_over = n
    model = _TokenTable(get_group("block_glk")(H * d, H, device="cuda"), phi.cuda())
    table = model.prior_bank.phi_embed
    before = table.detach().clone()

    eigh, decomposed, dtypes = torch.linalg.eigh, [], []

    def counting(A: torch.Tensor) -> tuple:
        decomposed.append(A.numel() // (d * d))
        dtypes.append(A.dtype)
        return eigh(A)

    monkeypatch.setattr(torch.linalg, "eigh", counting)
    torch.set_float32_matmul_precision("high")

    def host_reads(function: Callable[[], Any]) -> int:
        torch.cuda.synchronize()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torch.cuda.set_sync_debug_mode("warn")
            try:
                function()
            finally:
                torch.cuda.set_sync_debug_mode(0)
        return sum("called a synchronizing CUDA operation" in str(w.message) for w in caught)

    for call in range(2):
        decomposed.clear()
        reads = host_reads(lambda: bound_phi_frame_condition_(model, KAPPA))
        assert reads <= 3 + len(decomposed), (call, reads, len(decomposed))
        if call == 1:
            assert sum(decomposed) <= H * n_over
    assert torch.float64 not in dtypes
    assert torch.equal(table.detach()[:2 * n], before[:2 * n])                   # inside the bound: bitwise unchanged

    X = table.detach().to(torch.float64).reshape(-1, H, d, d)
    S = 0.5 * (X + X.mT)
    trace = S.diagonal(dim1=-2, dim2=-1).sum(-1)
    lam = torch.linalg.eigvalsh(S - (trace / d)[..., None, None] * torch.eye(d, dtype=torch.float64, device="cuda"))
    assert float((lam[..., -1] - lam[..., 0]).max()) <= 2.0 * C * (1 + 1e-5)
    X0 = before.to(torch.float64).reshape(-1, H, d, d)
    torch.testing.assert_close(X - X.mT, X0 - X0.mT, rtol=0.0, atol=1e-5)
    torch.testing.assert_close(trace, (0.5 * (X0 + X0.mT)).diagonal(dim1=-2, dim2=-1).sum(-1), rtol=0.0, atol=1e-4)
