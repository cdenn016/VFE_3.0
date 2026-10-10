"""Round-2 remediation (October 9, 2026): projected-scorer numerics and frame-contract cost.

Covers the centered pair-KL GEMM error estimate and direct-tile fallback (#7), the IEEE pin of the
checkpointed projection tiles on recompute (#15), the compact head-block covariance contraction (#21),
and unique-row validation of gathered canonical frame pairs (#41).
"""
import math
import os
from collections.abc import Iterator

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import vfe3.relational_scoring as rs
from vfe3.contracts import CanonicalFrameContext, _gathered_source, mark_gathered_frame_pair
from vfe3.geometry import transport


@pytest.fixture
def device() -> torch.device:
    return torch.device(os.environ.get("VFE3_TEST_DEVICE", "cpu"))


@pytest.fixture(autouse=True)
def one_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def tf32_policy() -> Iterator[None]:
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    yield
    torch.set_float32_matmul_precision(previous)


def _spd(n: int, r: int, cond: float, generator: torch.Generator, dtype: torch.dtype) -> torch.Tensor:
    Q, _ = torch.linalg.qr(torch.randn(n, r, r, generator=generator, dtype=torch.float64))
    eig = 0.1 * torch.logspace(0, math.log10(cond), r, dtype=torch.float64).expand(n, r)
    G = Q @ torch.diag_embed(eig) @ Q.transpose(-1, -2)
    return (0.5 * (G + G.transpose(-1, -2))).to(dtype)


def _kl64(aq: torch.Tensor, Gq: torch.Tensor, ak: torch.Tensor, Gk: torch.Tensor) -> torch.Tensor:
    r"""Direct float64 KL(q_i || k_j) for (N, r) means and (N, r, r) covariances."""
    aq, Gq, ak, Gk = (t.double() for t in (aq, Gq, ak, Gk))
    P = torch.cholesky_inverse(torch.linalg.cholesky(Gk))
    delta = aq[:, None] - ak[None]
    return 0.5 * (torch.einsum("iab,jab->ij", Gq, P) + torch.einsum("ijr,jrs,ijs->ij", delta, P, delta)
                  - aq.shape[-1] + torch.logdet(Gk)[None] - torch.logdet(Gq)[:, None])


def _two_slice_case(device: torch.device, r: int = 4, n: int = 12) -> tuple:
    r"""Head 0 has unit-spread means (GEMM keeps it); head 1 spread 1000 (estimate exceeds 1e-3 nats)."""
    g = torch.Generator().manual_seed(7)
    heads = []
    for spread in (1.0, 1000.0):
        G = _spd(n, r, 10.0, g, torch.float64)
        ak = spread * torch.randn(n, r, generator=g, dtype=torch.float64)
        aq = ak + 3e-3 * torch.randn(n, r, generator=g, dtype=torch.float64)    # near-minimum pairs
        heads.append((aq, G, ak, G.clone()))
    stack = [torch.stack([h[k] for h in heads], dim=1) for k in range(4)]       # (N, H, ...)
    return tuple(t.to(device) for t in stack)


def test_gemm_estimate_routes_the_spread_slice_to_direct_tiles(device: torch.device) -> None:
    aq, Gq, ak, Gk = _two_slice_case(device)
    reference = torch.stack([_kl64(aq[:, h], Gq[:, h], ak[:, h], Gk[:, h]) for h in range(2)])
    rs.reset_gemm_fallback_slices()
    fixed = rs.projected_gaussian_kl(aq.float(), Gq.float(), ak.float(), Gk.float(), symmetric_inputs=True,
                                     gemm_expansion_tol=rs._GEMM_EXPANSION_TOL)
    assert rs.gemm_fallback_slices() == 1
    gemm_only = rs.projected_gaussian_kl(aq.float(), Gq.float(), ak.float(), Gk.float(), symmetric_inputs=True)
    assert rs.gemm_fallback_slices() == 1
    # The healthy slice keeps the GEMM bit for bit; the spread slice is repaired.
    assert torch.equal(fixed[0], gemm_only[0])
    near = reference < 10.0
    error_fixed = (fixed.double().cpu() - reference.cpu()).abs()[near.cpu()].max()
    error_gemm = (gemm_only.double().cpu() - reference.cpu()).abs()[near.cpu()].max()
    assert error_fixed < 1e-4
    assert error_gemm > 1e-2


@pytest.mark.parametrize("r", [4, 8])
@pytest.mark.parametrize("cond", [1.0, 1e3])
@pytest.mark.parametrize("spread,outlier", [(1.0, 0.0), (100.0, 0.0), (1000.0, 0.0), (4000.0, 0.0), (1.0, 4000.0)])
def test_estimate_majorizes_the_gemm_error_on_the_audit_sweep(
        r: int, cond: float, spread: float, outlier: float, device: torch.device) -> None:
    g = torch.Generator().manual_seed(3)
    n = 16
    G = _spd(n, r, cond, g, torch.float64)
    ak = 3.0 + spread * torch.randn(n, r, generator=g, dtype=torch.float64)
    ak[0] += outlier
    aq = ak + 3e-3 * torch.randn(n, r, generator=g, dtype=torch.float64)
    reference = _kl64(aq, G, ak, G)
    a_q, a_k, G32 = aq.float().to(device), ak.float().to(device), G.float().to(device)
    chol = torch.linalg.cholesky(G32)
    P = torch.cholesky_inverse(chol)
    logdet = 2 * chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    gemm = rs._projected_kl_gemm(a_q[None], a_k[None], G32[None], P[None], logdet[None], logdet[None])[0]
    estimate = rs._gemm_expansion_error(a_q[:, None], a_k[:, None], P[:, None],
                                        torch.zeros(n, 1, dtype=torch.bool, device=device))
    near = reference < 10.0
    error = (gemm.double().cpu() - reference)[near].abs().max()
    assert error <= float(estimate.max())
    # After the fix every near-minimum pair sits at the direct tile's accuracy.
    fixed = rs.projected_gaussian_kl(a_q[:, None], G32[:, None], a_k[:, None], G32[:, None],
                                     symmetric_inputs=True, gemm_expansion_tol=rs._GEMM_EXPANSION_TOL)[0]
    bound = float(estimate.max()) if float(estimate.max()) <= rs._GEMM_EXPANSION_TOL else 0.
    assert (fixed.double().cpu() - reference)[near].abs().max() < 1e-4 + bound


def test_fallback_adds_no_host_synchronization(device: torch.device) -> None:
    class CountScalarReads(TorchDispatchMode):
        count = 0

        def __torch_dispatch__(self, func: object, types: tuple, args: tuple = (),
                               kwargs: dict | None = None) -> object:
            if func == torch.ops.aten._local_scalar_dense.default:
                self.count += 1
            return func(*args, **(kwargs or {}))

    aq, Gq, ak, Gk = (t.float() for t in _two_slice_case(device))
    counts = []
    for heads in (slice(0, 1), slice(0, 2)):          # healthy only, then one fallback slice
        counter = CountScalarReads()
        with counter:
            rs.projected_gaussian_kl(aq[:, heads], Gq[:, heads], ak[:, heads], Gk[:, heads],
                                     symmetric_inputs=True, gemm_expansion_tol=rs._GEMM_EXPANSION_TOL)
        counts.append(counter.count)
    assert counts[0] == counts[1]


def test_fallback_gradient_matches_float64(device: torch.device) -> None:
    aq, Gq, ak, Gk = _two_slice_case(device, r=3, n=6)
    weights = torch.randn(2, 6, 6, generator=torch.Generator().manual_seed(5), dtype=torch.float64).to(device)
    grads = []
    for dtype in (torch.float32, torch.float64):
        a = aq.to(dtype).requires_grad_()
        out = rs.projected_gaussian_kl(a, Gq.to(dtype), ak.to(dtype), Gk.to(dtype), symmetric_inputs=True,
                                       gemm_expansion_tol=rs._GEMM_EXPANSION_TOL)
        (grad,) = torch.autograd.grad((out.double() * weights).sum(), a)
        grads.append(grad.double())
    torch.testing.assert_close(grads[0], grads[1], rtol=2e-3, atol=2e-3 * float(grads[1].abs().max()))


def _policy_spy(monkeypatch: pytest.MonkeyPatch, name: str) -> list:
    seen = []
    original = getattr(rs, name)

    def spy(*args: object, **kwargs: object) -> object:
        seen.append(torch.get_float32_matmul_precision())
        return original(*args, **kwargs)
    monkeypatch.setattr(rs, name, spy)
    return seen


def test_projected_moments_tiles_recompute_at_ieee(
        tf32_policy: None, device: torch.device, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rs, "_PROJECTION_TILE_BYTES", 64)
    seen = _policy_spy(monkeypatch, "_projected_linear_map")
    g = torch.Generator().manual_seed(1)
    N, H, d, r = 5, 2, 3, 2
    mu = torch.randn(N, H, d, generator=g).to(device).requires_grad_()
    Sigma = (torch.rand(N, H, d, generator=g) + 0.5).to(device).requires_grad_()
    inverse = (torch.eye(d) + 0.1 * torch.randn(N, H, d, d, generator=g)).to(device)
    S = torch.randn(H, r, d, generator=g).to(device).requires_grad_()
    a, G = rs.projected_moments(mu, Sigma, inverse, S)
    forward_calls = len(seen)
    (a.sum() + G.sum()).backward()
    assert len(seen) > forward_calls > 1                 # tiled, and the tiles were recomputed
    assert set(seen) == {"highest"}
    assert torch.get_float32_matmul_precision() == "high"


def test_projected_root_factor_tiles_recompute_at_ieee(
        tf32_policy: None, device: torch.device, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rs, "_PROJECTION_TILE_BYTES", 64)
    seen = _policy_spy(monkeypatch, "cholesky_qr2")
    g = torch.Generator().manual_seed(2)
    N, H, d, r = 5, 2, 3, 2
    Sigma = (torch.rand(N, H, d, generator=g) + 0.5).to(device).requires_grad_()
    inverse = (torch.eye(d) + 0.1 * torch.randn(N, H, d, d, generator=g)).to(device)
    S = torch.randn(H, r, d, generator=g).to(device)
    C, failed = rs._projected_root_factor((Sigma, inverse, S), torch.Size((N, H)), torch.float32)
    assert not failed.any()
    forward_calls = len(seen)
    C.sum().backward()
    assert len(seen) > forward_calls > 1
    assert set(seen) == {"highest"}


def _block_case(dtype: torch.dtype, device: torch.device) -> tuple:
    g = torch.Generator().manual_seed(4)
    B, N, H, d, r = 2, 3, 3, 2, 2
    K = H * d
    root = torch.randn(B, N, H, d, d, generator=g, dtype=torch.float64)
    blocks = (root @ root.transpose(-1, -2) + 0.5 * torch.eye(d, dtype=torch.float64)).to(dtype).to(device)
    blocks.requires_grad_()
    mu = torch.randn(B, N, K, generator=g, dtype=torch.float64).to(dtype).to(device).requires_grad_()
    inverse = (torch.eye(d, dtype=torch.float64) + 0.2 * torch.randn(B, N, H, d, d, generator=g,
                                                                        dtype=torch.float64)).to(dtype).to(device)
    S = torch.randn(2, H, r, K, generator=g, dtype=torch.float64).to(dtype).to(device).requires_grad_()
    return blocks, mu, inverse, S, H, K


def _dense_sigma(blocks: torch.Tensor, H: int, K: int, per_head: bool) -> torch.Tensor:
    d = blocks.shape[-1]
    Sigma = torch.zeros(*blocks.shape[:-3], K, K, dtype=blocks.dtype, device=blocks.device)
    for m in range(H):
        Sigma = Sigma.index_put((..., slice(m * d, (m + 1) * d), slice(m * d, (m + 1) * d)), blocks[..., m, :, :]) \
            if False else Sigma + torch.nn.functional.pad(
                blocks[..., m, :, :], (m * d, K - (m + 1) * d, m * d, K - (m + 1) * d))
    Sigma_h = Sigma.unsqueeze(-3).expand(*Sigma.shape[:-2], H, K, K)
    return Sigma_h.contiguous() if per_head else Sigma_h


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("per_head", [False, True])
@pytest.mark.parametrize("stacked", [False, True])
@pytest.mark.parametrize("tiled", [False, True])
def test_head_block_projection_matches_dense(
        dtype: torch.dtype, per_head: bool, stacked: bool, tiled: bool,
        device: torch.device, monkeypatch: pytest.MonkeyPatch) -> None:
    if tiled:
        monkeypatch.setattr(rs, "_PROJECTION_TILE_BYTES", 64)
    blocks, mu, inverse, S, H, K = _block_case(dtype, device)
    maps = S if stacked else S[0]
    mu_h = mu.unsqueeze(-2).expand(*mu.shape[:-1], H, K)
    results = []
    for head_dims in (None, (K // H,) * H):
        Sigma_h = _dense_sigma(blocks, H, K, per_head)
        a, G = rs.projected_moments(mu_h, Sigma_h, inverse, maps, head_dims=head_dims)
        loss = (G * torch.linspace(-1, 1, G.numel(), dtype=dtype, device=device).reshape(G.shape)).sum() + a.square().sum()
        grads = torch.autograd.grad(loss, (blocks, mu, S))
        results.append((a, G, *grads))
    tol = dict(rtol=1e-5, atol=1e-5) if dtype == torch.float32 else dict(rtol=1e-12, atol=1e-12)
    for dense, compact in zip(*results):
        torch.testing.assert_close(compact, dense, **tol)


def test_head_block_projection_rejects_bad_declarations(device: torch.device) -> None:
    blocks, mu, inverse, S, H, K = _block_case(torch.float64, device)
    mu_h = mu.unsqueeze(-2).expand(*mu.shape[:-1], H, K)
    Sigma_h = _dense_sigma(blocks, H, K, False)
    for bad in ((K,) * 2, (1, K - 1), (0,) * H):
        with pytest.raises(ValueError, match="head_dims"):
            rs.projected_moments(mu_h, Sigma_h, inverse, S[0], head_dims=bad)


def test_head_block_projection_never_materializes_the_broadcast_covariance(device: torch.device) -> None:
    class CountAllocations(TorchDispatchMode):
        def __init__(self) -> None:
            super().__init__()
            self.sizes: list = []

        def __torch_dispatch__(self, func: object, types: tuple, args: tuple = (),
                               kwargs: dict | None = None) -> object:
            out = func(*args, **(kwargs or {}))
            inputs = {t.untyped_storage().data_ptr() for t in torch.utils._pytree.tree_leaves((args, kwargs))
                      if isinstance(t, torch.Tensor)}
            for t in (out if isinstance(out, (tuple, list)) else (out,)):
                if isinstance(t, torch.Tensor) and t.untyped_storage().data_ptr() not in inputs:
                    self.sizes.append(t.numel())             # a fresh allocation, not a view
            return out

    g = torch.Generator().manual_seed(9)
    N, H, d, r, R = 4, 4, 8, 8, 2
    K = H * d
    blocks = torch.randn(N, H, d, d, generator=g)
    blocks = blocks @ blocks.transpose(-1, -2) + torch.eye(d)
    Sigma = torch.zeros(N, K, K)
    for m in range(H):
        Sigma[:, m * d:(m + 1) * d, m * d:(m + 1) * d] = blocks[:, m]
    mu_h = torch.randn(N, K, generator=g).unsqueeze(-2).expand(N, H, K)
    Sigma_h = Sigma.unsqueeze(-3).expand(N, H, K, K)                     # head broadcast, as relational.py builds it
    inverse = torch.eye(d).expand(N, H, d, d)
    S = torch.randn(R, H, r, K, generator=g)
    largest, totals = [], []
    for head_dims in (None, (d,) * H):
        counter = CountAllocations()
        with counter:
            rs.projected_moments(mu_h.to(device), Sigma_h.to(device), inverse.to(device), S.to(device),
                                 head_dims=head_dims)
        largest.append(max(counter.sizes))
        totals.append(sum(counter.sizes))
    broadcast = R * N * H * K * K
    assert largest[0] >= broadcast                     # the dense product copies Sigma per head and role
    assert largest[1] <= R * N * H * r * K             # the compact path allocates at most an L-sized tensor
    assert totals[1] < totals[0]


def _gathered_pair() -> tuple:
    ids = torch.tensor([[2, 0, 2, 1], [0, 2, 2, 0]])
    table = torch.randn(3, 2, 2, 2, generator=torch.Generator().manual_seed(6), dtype=torch.float64) * 0.1
    forward, inverse = transport._stable_compact_glk_exp_pair(table[ids].float(), _content_token_ids=ids)
    return ids, forward, inverse


def test_gathered_pair_is_validated_on_unique_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    _, forward, inverse = _gathered_pair()
    source = _gathered_source(forward, inverse)
    assert source is not None and source[0].shape[0] == 3 and forward.shape[:2] == (2, 4)
    products = []
    original = torch.Tensor.__matmul__

    def spy(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        if self.dtype == torch.float64:
            products.append(tuple(self.shape))
        return original(self, other)
    monkeypatch.setattr(torch.Tensor, "__matmul__", spy)
    CanonicalFrameContext(forward, inverse, True)
    assert products and all(shape[0] == 3 and len(shape) == 4 for shape in products)
    products.clear()
    CanonicalFrameContext(forward[:, :3], inverse[:, :3], True)        # a slice validates in full
    assert products and all(shape[:2] == (2, 3) for shape in products)


def test_gathered_pair_failure_matches_full_validation() -> None:
    table = torch.randn(3, 2, 2, 2, generator=torch.Generator().manual_seed(8), dtype=torch.float64) * 0.1
    unique_forward = torch.linalg.matrix_exp(table).float()
    unique_inverse = torch.linalg.inv(unique_forward.double()).float()
    unique_inverse[1] *= 1.01                                         # one unique row is not an inverse
    index = torch.tensor([2, 0, 2, 1, 0, 1, 2, 0])
    messages = []
    for marked in (False, True):
        forward = unique_forward.index_select(0, index).reshape(2, 4, 2, 2, 2)
        inverse = unique_inverse.index_select(0, index).reshape(2, 4, 2, 2, 2)
        if marked:
            mark_gathered_frame_pair(forward, inverse, unique_forward, unique_inverse)
            assert _gathered_source(forward, inverse) is not None
        with pytest.raises(ValueError) as error:
            CanonicalFrameContext(forward, inverse, True)
        messages.append(str(error.value))
    assert messages[0] == messages[1]
    assert "mutual inverses" in messages[0]


def test_in_place_write_voids_the_gather_record() -> None:
    _, forward, inverse = _gathered_pair()
    forward.mul_(1.0)
    assert _gathered_source(forward, inverse) is None
    CanonicalFrameContext(forward, inverse, True)


def test_gemm_fallback_is_off_by_default_and_wired_from_the_config() -> None:
    from tests.profiles import test_config
    from vfe3.model.model import VFEModel
    aq, Gq, ak, Gk = (t.float() for t in _two_slice_case(torch.device("cpu")))
    rs.reset_gemm_fallback_slices()
    rs.projected_gaussian_kl(aq, Gq, ak, Gk, symmetric_inputs=True)
    assert rs.gemm_fallback_slices() == 0
    with pytest.raises(ValueError, match="pair_kl_gemm_tol"):
        test_config(pair_kl_gemm_tol=0.0)
    for tol in (None, 1e-3):
        model = VFEModel(test_config(pair_kl_gemm_tol=tol))
        assert {op.pair_kl_gemm_tol for op in model.modules() if hasattr(op, "pair_kl_gemm_tol")} == {tol}
