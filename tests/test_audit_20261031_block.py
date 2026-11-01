r"""Remediation of the October 31, 2026 ultradeep audit, package ``block``.

The block MLP, the block norms, the block wiring and the CG coupling. Every case is a tiny CPU model or module
(K <= 12, at most two layers, two heads, vocabulary <= 11); configuration comes from ``tests.profiles``.
"""
from __future__ import annotations

import copy
import math
import warnings
from typing import Any, Callable, Dict, Iterator, List, Tuple

import pytest
import torch

from tests.profiles import config_values, reference_k8_values
from vfe3.belief import BeliefState
from vfe3.config import VFE3Config
from vfe3.contracts import CanonicalFrameContext
from vfe3.geometry.norms import AffineLayerNorm
from vfe3.model.block_mlp import GaugeGateBlockMLP, PrototypeAgentBlockMLP
from vfe3.model.cg_coupling import CGCoupling, CGMomentResult
from vfe3.model.model import VFEModel
from vfe3.numerics import ValidatedCholeskySolve

F64 = torch.float64


@pytest.fixture(autouse=True)
def _two_threads() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield
    torch.set_num_threads(previous)


def _quiet_config(**values: Any) -> VFE3Config:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VFE3Config(**values)


class _HostReads:
    r"""Count ``bool(tensor)`` calls (host reads of a device verdict) while active."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.count = 0
        original = torch.Tensor.__bool__

        def counting(tensor: torch.Tensor) -> bool:
            self.count += 1
            return original(tensor)

        monkeypatch.setattr(torch.Tensor, "__bool__", counting)


def _spd(n: int, K: int, kappa: float, generator: torch.Generator) -> torch.Tensor:   # (n, K, K) float64
    Q, _ = torch.linalg.qr(torch.randn(n, K, K, generator=generator, dtype=F64))
    lam = torch.logspace(0.0, -math.log10(kappa), K, dtype=F64)
    S = (Q * lam) @ Q.mT
    return 0.5 * (S + S.mT)


# A016 (gap-mixed-precision-routes-1): the CG delta_full Jacobian and covariance pushforward run in an fp32 island.

def _cg_coupling(mode: str, weight: float) -> CGCoupling:
    from vfe3.geometry.groups import get_group
    group = get_group("so_n")(4, group_n=3, irrep_spec=[("l1", 1), ("l0", 1)], dtype=F64)
    coupling = CGCoupling(3, "so", group.irrep_dims, group.irrep_labels, cg_covariance_mode=mode)
    with torch.no_grad():
        coupling.path_weights.fill_(weight)
    return coupling


def test_cg_delta_full_covariance_is_the_fp32_result_under_bf16_autocast() -> None:
    coupling = _cg_coupling("delta_full", 0.5)
    g = torch.Generator().manual_seed(3)
    sigma = torch.cat([_spd(200, 4, kappa, g) for kappa in (1e1, 1e2, 1e3)]).float() * 15.0   # (600, 4, 4)
    mu = torch.randn(600, 4, generator=g)
    with torch.no_grad():
        reference = coupling.forward_moments(mu, sigma)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            amp = coupling.forward_moments(mu, sigma)
            mu_call, sigma_call = coupling(mu, sigma)                 # the call vfe3.model.block makes
    assert amp.sigma.dtype == torch.float32 and amp.jacobian.dtype == torch.float32 and amp.mu.dtype == torch.float32
    assert torch.equal(amp.jacobian, reference.jacobian)
    assert torch.equal(amp.sigma, reference.sigma)
    assert torch.equal(sigma_call, reference.sigma)
    assert bool((torch.linalg.cholesky_ex(reference.sigma).info == 0).all())
    assert bool((torch.linalg.cholesky_ex(amp.sigma).info == 0).all())
    # The mean delta stays the documented amp consumer: bf16 contractions, float32 result near the fp32 one.
    assert not torch.equal(amp.mu, reference.mu)
    torch.testing.assert_close(amp.mu, reference.mu, rtol=2e-2, atol=2e-2)
    assert torch.equal(mu_call, amp.mu)


def _cg_model(amp_dtype: Any, weight: float, n_layers: int) -> VFEModel:
    cfg = _quiet_config(**config_values(
        vocab_size=11, embed_dim=4, n_heads=2, max_seq_len=6, batch_size=8, eval_stride=None, attention_window=6,
        n_layers=n_layers, attention_mode="projected_predictive", predictive_readout="kernel", family="gaussian_full",
        decode_mode="full_chunked", use_block_mlp=False, norm_type_block="none", layernorm_affine=False,
        relational_routing="noisy_key", predictive_covariance="moment_matched", rope_mode="none",
        content_link_chart="exponential", kl_max=32.0, fp32_matmul_precision="highest", decode_av_precision="fp32",
        gauge_group="so_n", group_n=3, irrep_spec=[("l1", 1), ("l0", 1)], gauge_tying=None, use_cg_coupling=True,
        cg_covariance_mode="delta_full", amp_dtype=amp_dtype, mu_init_std=1.0, sigma_init=1.0))
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(cfg)
    with torch.no_grad():
        model.cg_coupling.path_weights.fill_(weight)
    return model


@pytest.mark.parametrize("weight", [2.0, 4.0])
def test_cg_delta_full_two_layer_bf16_forward_and_backward_complete(weight: float) -> None:
    r"""The probe route of the audit: in bf16 the second layer's fail-closed factorization raised on an indefinite
    post-CG covariance; with the fp32 island every post-CG covariance is positive definite and the step runs."""
    model = _cg_model("bf16", weight, n_layers=2)
    post_cg: List[bool] = []
    real = model.cg_coupling.forward_moments

    def spy(mu: torch.Tensor, sigma: torch.Tensor) -> CGMomentResult:
        result = real(mu, sigma)
        post_cg.append(bool((torch.linalg.cholesky_ex(result.sigma.detach().double()).info == 0).all()))
        return result

    model.cg_coupling.forward_moments = spy
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 11, (8, 6), generator=g)
    targets = torch.randint(0, 11, (8, 6), generator=g)
    _, loss, _ = model(ids, targets)
    loss.backward()
    assert post_cg == [True, True]
    assert bool(torch.isfinite(loss))
    assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
    assert model.cg_coupling.path_weights.grad is not None


# A191 (gauge-structure-7): cached per-device group indices and one stacked Jacobian contraction per group.

def _reference_cg_moments(coupling: CGCoupling, mu: torch.Tensor,
                          sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""``CGCoupling.forward_moments`` as of 5bb0a57 (list-indexed weights, one einsum pair per path)."""
    cg = coupling._cast_buffers(mu.dtype, mu.device)
    w = coupling.path_weights.to(dtype=mu.dtype)
    K = mu.shape[-1]
    delta = torch.zeros_like(mu)
    jac = torch.eye(K, dtype=mu.dtype).expand(*mu.shape[:-1], K, K).clone()
    for (t, m, sc, dc, idxs, srcs) in coupling._groups:
        _, da0, _, db0 = srcs[0]
        Cm = cg[t][m].reshape(dc, da0, db0)
        x = torch.stack([mu[..., sa:sa + da] for (sa, da, _sb, _db) in srcs], dim=0)
        y = torch.stack([mu[..., sb:sb + db] for (_sa, _da, sb, db) in srcs], dim=0)
        xy = (x.unsqueeze(-1) * y.unsqueeze(-2)).reshape(len(srcs), *x.shape[1:-1], -1)
        wsum = torch.einsum("p,p...d->...d", w[idxs], xy)
        delta[..., sc:sc + dc] = delta[..., sc:sc + dc] + torch.einsum("cd,...d->...c", Cm.reshape(dc, -1), wsum)
        for local_p, (sa, da, sb, db) in enumerate(srcs):
            wp = w[idxs[local_p]]
            jac[..., sc:sc + dc, sa:sa + da] = (jac[..., sc:sc + dc, sa:sa + da]
                                                + wp * torch.einsum("cij,...j->...ci", Cm, mu[..., sb:sb + db]))
            jac[..., sc:sc + dc, sb:sb + db] = (jac[..., sc:sc + dc, sb:sb + db]
                                                + wp * torch.einsum("cij,...i->...cj", Cm, mu[..., sa:sa + da]))
    jsj = torch.einsum("...ik,...lk->...il", torch.einsum("...ij,...jk->...ik", jac, sigma), jac)
    return mu + delta, 0.5 * (jsj + jsj.mT) + coupling.cg_covariance_floor * sigma, jac


def _tower_coupling(seed: int) -> CGCoupling:
    from vfe3.geometry.groups import get_group
    group = get_group("so_n")(12, group_n=3, irrep_spec=[("l0", 1), ("l1", 2), ("l2", 1)], dtype=F64)
    coupling = CGCoupling(3, "so", group.irrep_dims, group.irrep_labels, cg_covariance_mode="delta_full").double()
    with torch.no_grad():
        coupling.path_weights.copy_(0.4 * torch.randn(coupling.path_weights.shape,
                                                      generator=torch.Generator().manual_seed(seed), dtype=F64))
    return coupling


def test_cg_stacked_jacobian_equals_the_per_path_accumulation() -> None:
    coupling = _tower_coupling(seed=0)
    assert any(len(srcs) > 1 for (*_, srcs) in coupling._groups)          # groups with several source pairs
    g = torch.Generator().manual_seed(1)
    mu = torch.randn(2, 3, 12, generator=g, dtype=F64)
    sigma = _spd(6, 12, 50.0, g).reshape(2, 3, 12, 12)
    result = coupling.forward_moments(mu, sigma)
    mu_ref, sigma_ref, jac_ref = _reference_cg_moments(coupling, mu, sigma)
    torch.testing.assert_close(result.jacobian, jac_ref, rtol=1e-14, atol=1e-15)
    torch.testing.assert_close(result.mu, mu_ref, rtol=0.0, atol=0.0)
    torch.testing.assert_close(result.sigma, sigma_ref, rtol=1e-13, atol=1e-14)
    J_auto = torch.autograd.functional.jacobian(lambda m: coupling.forward_moments(m, sigma[0, 0]).mu, mu[0, 0])
    torch.testing.assert_close(coupling.forward_moments(mu[0, 0], sigma[0, 0]).jacobian, J_auto,
                               rtol=0.0, atol=1e-12)


def test_cg_moments_gradients_match_finite_differences() -> None:
    coupling = _cg_coupling("delta_full", 0.0).double()
    g = torch.Generator().manual_seed(2)
    weights = (0.5 * torch.randn(coupling.path_weights.shape, generator=g, dtype=F64)).requires_grad_()
    mu = torch.randn(2, 4, generator=g, dtype=F64).requires_grad_()
    sigma = _spd(2, 4, 10.0, g).requires_grad_()

    def moments(w: torch.Tensor, m: torch.Tensor, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.func.functional_call(coupling, {"path_weights": w}, (m, s))

    assert torch.autograd.gradcheck(moments, (weights, mu, sigma), eps=1e-6, atol=1e-8, rtol=1e-6)


def test_cg_group_indices_are_built_once_per_device() -> None:
    coupling = _tower_coupling(seed=3)
    mu = torch.randn(2, 12, dtype=F64)
    sigma = _spd(2, 12, 10.0, torch.Generator().manual_seed(4))
    coupling.forward_moments(mu, sigma)
    cached = coupling._index_cache[mu.device]
    assert [index.tolist() for index in cached] == [idxs for (_t, _m, _sc, _dc, idxs, _srcs) in coupling._groups]
    coupling(mu, sigma)
    coupling.forward_moments(mu, sigma)
    assert coupling._index_cache[mu.device] is cached
    assert all(a is b for a, b in zip(coupling._index_cache[mu.device], cached))
    coupling.float()                                                     # a conversion drops the caches
    assert coupling._index_cache == {}
    with torch.inference_mode():                                         # never cached under inference mode
        coupling(mu.float(), sigma.float())
    assert coupling._index_cache == {}


# A019 (gap-paper-arm-offlive-routes-1, gap-route-admissibility-fuzz-2) and A186 (gap-paper-arm-offlive-routes-5):
# the full-layout gauge gate keeps the working dtype after a float64 escalation, certifies all heads in one call and
# recomputes only the failed rows.

def _gate(irrep_dims: Tuple[int, ...], contract: str, seed: int = 0) -> GaugeGateBlockMLP:
    torch.manual_seed(seed)
    mlp = GaugeGateBlockMLP(irrep_dims, expansion=2, activation="gelu", dropout=0.0, covariance_contract=contract)
    gen = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():                                             # fc2 starts at zero: give the gate a response
        mlp.fc1.weight.div_(1e3)                                      # an ill-conditioned head reaches s_h ~ 1e5
        mlp.fc2.weight.copy_(0.3 * torch.randn(mlp.fc2.weight.shape, generator=gen))
        mlp.fc2.bias.copy_(0.1 * torch.randn(mlp.fc2.bias.shape, generator=gen))
    return mlp


def _head_block_covariance(n: int, irrep_dims: Tuple[int, ...], ill: Dict[Tuple[int, int], float],
                           generator: torch.Generator) -> torch.Tensor:   # (n, K, K) float32, block diagonal
    K = sum(irrep_dims)
    sigma = torch.zeros(n, K, K, dtype=F64)
    start = 0
    for h, d in enumerate(irrep_dims):
        blocks = _spd(n, d, 10.0, generator) * 2.0
        for (token, head), kappa in ill.items():
            if head == h:
                blocks[token] = _spd(1, d, kappa, generator)[0] * 2.0
        sigma[:, start:start + d, start:start + d] = blocks
        start += d
    return sigma.float()


@pytest.mark.parametrize("contract", ["passthrough", "delta_full"])
def test_gauge_gate_escalation_keeps_the_working_dtype(contract: str) -> None:
    irrep_dims = (4, 4)
    g = torch.Generator().manual_seed(5)
    sigma = _head_block_covariance(6, irrep_dims, {(2, 0): 1e6}, g)      # one head block past the fp32 limit 32768
    mu = torch.randn(6, 8, generator=g)
    gate = _gate(irrep_dims, contract)
    gate64 = copy.deepcopy(gate).double()
    mu = mu.requires_grad_()
    mu64 = mu.detach().double().requires_grad_()
    out = gate.forward_moments(mu, sigma)
    reference = gate64.forward_moments(mu64, sigma.double())
    assert out.mu.dtype == torch.float32
    assert gate.mean_update(mu, sigma).dtype == torch.float32
    torch.testing.assert_close(out.mu.double(), reference.mu, rtol=1e-4, atol=1e-5)
    if contract == "delta_full":
        assert out.sigma.dtype == torch.float32 and out.jacobian.dtype == torch.float32
        torch.testing.assert_close(out.jacobian.double(), reference.jacobian, rtol=1e-3, atol=1e-3)
    else:
        assert out.sigma is sigma
    # The escalated row differentiates through its float64 solve: the gradient matches the float64 module's.
    upstream = torch.randn(6, 8, generator=g)
    (grad,) = torch.autograd.grad((out.mu * upstream).sum(), mu)
    (grad64,) = torch.autograd.grad((reference.mu * upstream.double()).sum(), mu64)
    assert grad.dtype == torch.float32
    torch.testing.assert_close(grad.double(), grad64, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("contract", ["passthrough", "delta_full"])
def test_gauge_gate_full_layout_gradients_match_finite_differences(contract: str) -> None:
    r"""float64 gradcheck of the stacked full-layout invariants (heads of two widths), in the mean and in a symmetric
    covariance; the certificate refuses an asymmetric perturbation, so the covariance is symmetrized inside."""
    irrep_dims = (2, 3, 3)
    gate = _gate(irrep_dims, contract).double()
    g = torch.Generator().manual_seed(17)
    mu = torch.randn(3, 8, generator=g, dtype=F64).requires_grad_()
    S = (_spd(3, 8, 10.0, g) * 2.0).requires_grad_()

    def moments(m: torch.Tensor, s: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        result = gate.forward_moments(m, 0.5 * (s + s.mT))
        return (result.mu,) if contract == "passthrough" else (result.mu, result.sigma)

    assert torch.autograd.gradcheck(moments, (mu, S), eps=1e-6, atol=1e-7, rtol=1e-5)


def test_gauge_gate_certifies_every_head_in_one_call_with_one_host_read(monkeypatch: pytest.MonkeyPatch) -> None:
    import vfe3.model.block_mlp as block_mlp
    from vfe3.numerics import validated_cholesky_solve

    irrep_dims = (4, 4)
    g = torch.Generator().manual_seed(6)
    sigma = _head_block_covariance(5, irrep_dims, {}, g)
    mu = torch.randn(5, 8, generator=g)
    gate = _gate(irrep_dims, "passthrough")
    calls: List[Tuple[Tuple[int, ...], torch.dtype]] = []

    def counting(matrix: torch.Tensor, rhs: Any = None, **kwargs: Any) -> ValidatedCholeskySolve:
        calls.append((tuple(matrix.shape), matrix.dtype))
        return validated_cholesky_solve(matrix, rhs, **kwargs)

    monkeypatch.setattr(block_mlp, "validated_cholesky_solve", counting)
    reads = _HostReads(monkeypatch)
    invariants, _, solutions = gate._invariants(mu, sigma)
    assert calls == [((5, 2, 4, 4), torch.float32)] and reads.count == 1
    # Value-identical to the per-head certified solves.
    for h in range(2):
        block = slice(4 * h, 4 * h + 4)
        checked = validated_cholesky_solve(sigma[:, block, block], mu[:, block].unsqueeze(-1))
        assert bool(checked.certified.all())
        assert torch.equal(solutions[h], checked.solution.squeeze(-1))
        assert torch.equal(invariants[:, h], (mu[:, block] * checked.solution.squeeze(-1)).sum(-1))


def test_gauge_gate_escalates_only_the_failed_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    import vfe3.model.block_mlp as block_mlp
    from vfe3.numerics import validated_cholesky_solve

    irrep_dims = (4, 4)
    g = torch.Generator().manual_seed(7)
    sigma = _head_block_covariance(5, irrep_dims, {(3, 1): 1e6}, g)
    mu = torch.randn(5, 8, generator=g)
    gate = _gate(irrep_dims, "passthrough")
    calls: List[Tuple[Tuple[int, ...], torch.dtype]] = []

    def counting(matrix: torch.Tensor, rhs: Any = None, **kwargs: Any) -> ValidatedCholeskySolve:
        calls.append((tuple(matrix.shape), matrix.dtype))
        return validated_cholesky_solve(matrix, rhs, **kwargs)

    monkeypatch.setattr(block_mlp, "validated_cholesky_solve", counting)
    invariants, _, solutions = gate._invariants(mu, sigma)
    assert calls == [((5, 2, 4, 4), torch.float32), ((1, 4, 4), F64)]
    assert invariants.dtype == torch.float32 and all(s.dtype == torch.float32 for s in solutions)
    exact = torch.linalg.solve(sigma[3, 4:, 4:].double(), mu[3, 4:].double())
    torch.testing.assert_close(solutions[1][3].double(), exact, rtol=1e-6, atol=0.0)


def test_gauge_gate_flop_metadata_books_the_certificate() -> None:
    from vfe3.model.block_mlp import block_mlp_build_metadata
    irrep_dims = (3, 1, 2)
    full = block_mlp_build_metadata(enabled=True, mode="gauge_gate", covariance="passthrough", expansion=3,
                                    activation="gelu", dropout=0.0, covariance_floor=1e-4, embed_dim=6,
                                    irrep_dims=irrep_dims, n_layers=2, covariance_layout="full")
    gate_mlp = 4.0 * 3 * 3 ** 2
    solve = sum(2.0 * d ** 3 + 2.0 * d ** 2 for d in irrep_dims)
    certificate = sum((10.0 / 3.0) * d ** 3 + 12.0 * d ** 2 for d in irrep_dims)
    assert full.flops_per_token == pytest.approx(2 * (gate_mlp + solve + certificate), rel=1e-14)


# A145 (block-mlp-norm-2): the frame-diagonal reference block reads the stored reference variances in the diagonal
# layout; A175 (block-mlp-norm-3): the trace cap of a full layout in closed form; A176 (block-mlp-norm-4): only the
# failed rows of a certificate are recomputed in float64.

class _Calls:
    r"""Record the calls of the dense helpers of ``vfe3.model.block_mlp`` (shape and dtype of the first operand)."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
        import vfe3.model.block_mlp as block_mlp
        self.seen: List[Tuple[str, Tuple[int, ...], torch.dtype]] = []
        for name in names:
            original = getattr(block_mlp, name)

            def wrapped(*args: Any, _name: str = name, _original: Callable = original, **kwargs: Any):
                first = args[0]
                self.seen.append((_name, tuple(first.shape), first.dtype))
                return _original(*args, **kwargs)

            monkeypatch.setattr(block_mlp, name, wrapped)


def _frame_diagonal_case(seed: int, block_diagonal: bool
                         ) -> Tuple[BeliefState, CanonicalFrameContext, AffineLayerNorm]:
    B, N, H, d = 2, 3, 2, 3
    K = H * d
    g = torch.Generator().manual_seed(seed)
    if block_diagonal:
        U = torch.linalg.matrix_exp(0.3 * torch.randn(B, N, H, d, d, generator=g, dtype=F64))   # (B, N, H, d, d)
    else:
        U = torch.linalg.matrix_exp(0.3 * torch.randn(B, N, K, K, generator=g, dtype=F64))      # (B, N, K, K)
    frame = CanonicalFrameContext(U, torch.linalg.inv(U), block_diagonal=block_diagonal)
    mu = torch.randn(B, N, K, generator=g, dtype=F64)
    c = torch.empty(B, N, K, dtype=F64).uniform_(-1.0, 4.0, generator=g).exp()   # Sigma/s^2 up to ~50: the cap binds
    norm = AffineLayerNorm(K).double()
    with torch.no_grad():
        norm.weight.copy_(1.0 + 0.2 * torch.randn(K, generator=g, dtype=F64))
        norm.bias.copy_(0.1 * torch.randn(K, generator=g, dtype=F64))
    return BeliefState(mu=mu, sigma=c, phi=torch.zeros(B, N, 1, dtype=F64)), frame, norm


@pytest.mark.parametrize("block_diagonal", [True, False])
@pytest.mark.parametrize("mode, fallback", [("pushforward", False), ("pushforward", True),
                                            ("pushforward_moment", False)])
def test_frame_diagonal_reference_norm_pushforward_reads_the_diagonal_layout(
        mode: str, fallback: bool, block_diagonal: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    from vfe3.model.block import _reference_coordinate_block
    from vfe3.model.block_mlp import block_norm_covariance_pushforward, reset_layernorm_trace_cap_elements
    from vfe3.model.frame_actions import frame_matvec
    out, frame, norm = _frame_diagonal_case(0, block_diagonal)
    xi = frame_matvec(frame.inverse, out.mu, block_diagonal)
    # The route of 5bb0a57: the dense C = diag(c) through the full-layout pushforward, read back as its diagonal.
    dense = block_norm_covariance_pushforward(norm, xi, torch.diag_embed(out.sigma), 1e-3, mode=mode, trace_cap=True,
                                              cap_moment_fallback=fallback)
    expected = torch.diagonal(dense, dim1=-2, dim2=-1)
    calls = _Calls(monkeypatch, "layernorm_full_pushforward", "_gram_pushforward")
    result = _reference_coordinate_block(out, block_norm=norm, block_mlp=None, frame=frame, capture=None,
                                         hidden_bias=None, push_norm=True, covariance_floor=1e-3, norm_covariance=mode,
                                         trace_cap=True, cap_moment_fallback=fallback, frame_diagonal=True)
    reset_layernorm_trace_cap_elements()
    assert calls.seen == []
    assert result.sigma.shape == out.sigma.shape
    torch.testing.assert_close(result.sigma, expected, rtol=1e-12, atol=1e-14)
    assert torch.equal(result.mu, frame_matvec(frame.forward, norm(xi, out.sigma), block_diagonal))


@pytest.mark.parametrize("block_diagonal", [True, False])
@pytest.mark.parametrize("trace_cap", [True, False])
def test_frame_diagonal_reference_delta_full_reads_the_diagonal_layout(
        trace_cap: bool, block_diagonal: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    from vfe3.model.block import _reference_coordinate_block
    from vfe3.model.block_mlp import (BlockMLP, block_norm_linearization, block_norm_radial_term,
                                      block_norm_trace_cap, reset_layernorm_trace_cap_elements)
    from vfe3.model.frame_actions import frame_matvec
    out, frame, norm = _frame_diagonal_case(1, block_diagonal)
    torch.manual_seed(2)
    mlp = BlockMLP(6, 2, "gelu", 0.0, covariance_contract="delta_full", covariance_floor=1e-3).double()
    xi = frame_matvec(frame.inverse, out.mu, block_diagonal)
    C = torch.diag_embed(out.sigma)                                      # the dense C of 5bb0a57
    linearization = block_norm_linearization(norm, xi)
    reference = mlp.forward_moments(norm(xi, C), C, input_jacobian=linearization.jacobian,
                                    input_scale=linearization.scale, input_radial=block_norm_radial_term(norm, xi, C),
                                    input_cap=block_norm_trace_cap(norm, xi, C, 1e-3) if trace_cap else None)
    calls = _Calls(monkeypatch, "layernorm_full_pushforward", "_gram_pushforward")
    result = _reference_coordinate_block(out, block_norm=norm, block_mlp=mlp, frame=frame, capture=None,
                                         hidden_bias=None, push_norm=False, covariance_floor=1e-3,
                                         trace_cap=trace_cap, frame_diagonal=True)
    reset_layernorm_trace_cap_elements()
    assert calls.seen == []
    torch.testing.assert_close(result.sigma, torch.diagonal(reference.sigma, dim1=-2, dim2=-1),
                               rtol=1e-12, atol=1e-14)
    assert torch.equal(result.mu, frame_matvec(frame.forward, reference.mu, block_diagonal))


@pytest.mark.parametrize("route", ["pushforward", "pushforward_moment", "delta_full"])
def test_frame_diagonal_reference_block_gradients_match_the_dense_route_and_finite_differences(route: str) -> None:
    r"""The diagonal route's gradients in the mean, the reference variances and the parameters equal those of the
    dense route of 5bb0a57 to float64 rounding, and its mean and variance gradients pass a float64 gradcheck."""
    from vfe3.model.block import _reference_coordinate_block
    from vfe3.model.block_mlp import (BlockMLP, block_norm_covariance_pushforward, block_norm_linearization,
                                      block_norm_radial_term, block_norm_trace_cap,
                                      reset_layernorm_trace_cap_elements)
    from vfe3.model.frame_actions import frame_matvec
    out, frame, norm = _frame_diagonal_case(3, block_diagonal=True)
    delta_full = route == "delta_full"
    torch.manual_seed(4)
    mlp = BlockMLP(6, 2, "gelu", 0.0, covariance_contract="delta_full", covariance_floor=1e-3).double() \
        if delta_full else None
    mode = "pushforward" if delta_full else route
    params = list(norm.parameters()) + ([] if mlp is None else list(mlp.parameters()))
    upstream = torch.randn(out.sigma.shape, generator=torch.Generator().manual_seed(5), dtype=F64)

    def new_route(mu: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        result = _reference_coordinate_block(out._replace(mu=mu, sigma=c), block_norm=norm, block_mlp=mlp,
                                             frame=frame, capture=None, hidden_bias=None, push_norm=not delta_full,
                                             covariance_floor=1e-3, norm_covariance=mode, trace_cap=True,
                                             frame_diagonal=True)
        return result.mu, result.sigma

    def dense_route(mu: torch.Tensor, c: torch.Tensor) -> torch.Tensor:          # the dense C = diag(c) of 5bb0a57
        xi, C = frame_matvec(frame.inverse, mu, True), torch.diag_embed(c)
        if not delta_full:
            pushed = block_norm_covariance_pushforward(norm, xi, C, 1e-3, mode=mode, trace_cap=True)
        else:
            lin = block_norm_linearization(norm, xi)
            pushed = mlp.forward_moments(norm(xi, C), C, input_jacobian=lin.jacobian, input_scale=lin.scale,
                                         input_radial=block_norm_radial_term(norm, xi, C),
                                         input_cap=block_norm_trace_cap(norm, xi, C, 1e-3)).sigma
        return torch.diagonal(pushed, dim1=-2, dim2=-1)

    mu, c = out.mu.clone().requires_grad_(), out.sigma.clone().requires_grad_()
    grads = []
    for sigma_of in (lambda m, s: new_route(m, s)[1], dense_route):
        grads.append(torch.autograd.grad((sigma_of(mu, c) * upstream).sum(), [mu, c, *params], allow_unused=True))
    for new, dense in zip(*grads):                                    # beta (and fc2's bias) never reach the variance
        assert (new is None) == (dense is None)
        if new is not None:
            torch.testing.assert_close(new, dense, rtol=1e-10, atol=1e-13)
    assert grads[0][0] is not None and grads[0][1] is not None and grads[0][2] is not None
    assert torch.autograd.gradcheck(new_route, (mu, c), eps=1e-6, atol=1e-7, rtol=1e-5)
    reset_layernorm_trace_cap_elements()


@pytest.mark.parametrize("affine", [False, True])
def test_trace_cap_of_a_full_layout_is_the_closed_form_trace(affine: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    from vfe3.geometry.norms import AffineLayerNorm, LayerNorm
    from vfe3.model.block_mlp import (block_norm_trace_cap, layernorm_full_pushforward, layernorm_trace_cap,
                                      reset_layernorm_trace_cap_elements)
    K = 6
    g = torch.Generator().manual_seed(8)
    norm = AffineLayerNorm(K).double() if affine else LayerNorm(K)
    if affine:
        with torch.no_grad():
            norm.weight.copy_(1.0 + 0.3 * torch.randn(K, generator=g, dtype=F64))
    weight = norm.weight if affine else None
    mu = torch.randn(40, K, generator=g, dtype=F64)
    scale = torch.logspace(-2.0, 2.0, 40, dtype=F64).reshape(40, 1, 1)
    sigma = _spd(40, K, 20.0, g) * scale                                 # (40, K, K): the cap binds on the large ones
    reference = layernorm_trace_cap(layernorm_full_pushforward(mu, sigma, eps=norm.eps, covariance_floor=1e-3,
                                                               weight=weight), weight, False)
    calls = _Calls(monkeypatch, "layernorm_full_pushforward", "_cholesky_failures")
    lam = block_norm_trace_cap(norm, mu, sigma, 1e-3)
    assert calls.seen == []
    assert bool((lam < 1.0).any()) and bool((lam == 1.0).any())
    torch.testing.assert_close(lam, reference, rtol=1e-12, atol=0.0)
    # float32 inputs read the same scale to float32 rounding
    lam32 = block_norm_trace_cap(norm.float() if affine else norm, mu.float(), sigma.float(), 1e-3)
    torch.testing.assert_close(lam32, reference, rtol=1e-5, atol=0.0)
    reset_layernorm_trace_cap_elements()


def test_trace_cap_closed_form_gradients_match_finite_differences() -> None:
    r"""``block_norm_trace_cap`` on a full layout is ``layernorm_trace_cap`` of the closed-form diagonal; its gradient
    in the mean, the covariance and gamma, where the cap binds (lambda = B / tr), against float64 differences."""
    from vfe3.model.block_mlp import (layernorm_full_pushforward_diagonal, layernorm_trace_cap,
                                      reset_layernorm_trace_cap_elements)
    K = 4
    g = torch.Generator().manual_seed(9)
    mu = torch.randn(3, K, generator=g, dtype=F64).requires_grad_()
    sigma = (_spd(3, K, 5.0, g) * 30.0).requires_grad_()
    gamma = (1.0 + 0.3 * torch.randn(K, generator=g, dtype=F64)).requires_grad_()

    def cap(m: torch.Tensor, s: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        entries = layernorm_full_pushforward_diagonal(m, s, eps=1e-6, covariance_floor=1e-3, weight=w)
        return layernorm_trace_cap(entries, w, True)

    with torch.no_grad():
        assert bool((cap(mu, sigma, gamma) < 1.0).all())
    assert torch.autograd.gradcheck(cap, (mu, sigma, gamma), eps=1e-6, atol=1e-8, rtol=1e-6)
    reset_layernorm_trace_cap_elements()


def test_layernorm_full_pushforward_recomputes_only_the_failed_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    from vfe3.model.block_mlp import _cholesky_failures, _layernorm_full_pushforward_at, layernorm_full_pushforward
    K = 5
    g = torch.Generator().manual_seed(10)
    mu = torch.randn(6, K, generator=g)
    sigma = _spd(6, K, 10.0, g).float()
    sigma[2] = -sigma[2]                                                 # one deliberately indefinite token
    weight = 1.0 + 0.2 * torch.randn(K, generator=g)
    low = _layernorm_full_pushforward_at(mu, sigma, 1e-6, 1e-3, weight, None)
    failed = _cholesky_failures(low, None)
    assert failed.tolist() == [False, False, True, False, False, False]
    inputs = [t.clone().requires_grad_() for t in (mu, sigma, weight)]
    upstream = torch.randn(6, K, K, generator=g)

    def whole_batch(m: torch.Tensor, s: torch.Tensor, w: torch.Tensor) -> torch.Tensor:   # the recompute of 5bb0a57
        high = _layernorm_full_pushforward_at(m.double(), s.double(), 1e-6, 1e-3, w.double(), None)
        return torch.where(failed[:, None, None], high.float(), _layernorm_full_pushforward_at(m, s, 1e-6, 1e-3, w,
                                                                                               None))

    expected = whole_batch(*inputs)
    expected_grads = torch.autograd.grad((expected * upstream).sum(), inputs)
    calls = _Calls(monkeypatch, "_layernorm_full_pushforward_at")
    pushed = layernorm_full_pushforward(inputs[0], inputs[1], eps=1e-6, covariance_floor=1e-3, weight=inputs[2])
    assert [shape for (_, shape, dtype) in calls.seen if dtype == F64] == [(1, K)]
    assert torch.equal(pushed, expected)
    # The gradient routing is unchanged: the failed row through its float64 recompute, the others at float32.
    for new, old in zip(torch.autograd.grad((pushed * upstream).sum(), inputs), expected_grads):
        torch.testing.assert_close(new, old, rtol=1e-6, atol=1e-6)


def test_delta_covariance_recomputes_only_the_failed_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    from vfe3.model.block_mlp import LayerNormRadialTerm, _delta_covariance, _gram_pushforward
    K = 5
    g = torch.Generator().manual_seed(11)
    J = torch.eye(K).expand(6, K, K) + 0.05 * torch.randn(6, K, K, generator=g)
    J[4] = 0.1 * torch.eye(K)                                            # J Sigma J^T - 0.5 Sigma fails here only
    sigma = _spd(6, K, 10.0, g).float()
    scale = 1.0 + 0.1 * torch.rand(6, K, generator=g)
    rho, direction = 0.01 * torch.rand(6, 1, generator=g), torch.randn(6, K, generator=g)
    low = _gram_pushforward(J, sigma, -0.5, scale, LayerNormRadialTerm(rho, direction))
    failed = torch.linalg.cholesky_ex(low).info != 0
    assert failed.tolist() == [False, False, False, False, True, False]
    inputs = [t.clone().requires_grad_() for t in (J, sigma, scale, rho, direction)]
    upstream = torch.randn(6, K, K, generator=g)

    def whole_batch(j: torch.Tensor, s: torch.Tensor, d: torch.Tensor, r: torch.Tensor,
                    h: torch.Tensor) -> torch.Tensor:                    # the whole-batch recompute of 5bb0a57
        radial = LayerNormRadialTerm(r, h)
        high = _gram_pushforward(j.double(), s.double(), -0.5, d.double(), radial)
        return torch.where(failed[:, None, None], high.float(), _gram_pushforward(j, s, -0.5, d, radial))

    expected = whole_batch(*inputs)
    expected_grads = torch.autograd.grad((expected * upstream).sum(), inputs)
    calls = _Calls(monkeypatch, "_gram_pushforward")
    pushed = _delta_covariance(inputs[0], inputs[1], -0.5, inputs[2], LayerNormRadialTerm(inputs[3], inputs[4]), None)
    assert [shape for (_, shape, dtype) in calls.seen] == [(6, K, K), (1, K, K)]
    assert calls.seen[1][2] == F64
    assert torch.equal(pushed, expected)
    for new, old in zip(torch.autograd.grad((pushed * upstream).sum(), inputs), expected_grads):
        torch.testing.assert_close(new, old, rtol=1e-6, atol=1e-6)


def test_covariance_cholesky_refactors_only_the_failed_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    from vfe3.model.block_mlp import _covariance_cholesky
    K = 4
    sigma = _spd(5, K, 10.0, torch.Generator().manual_seed(12)).float()
    real = torch.linalg.cholesky_ex
    float64_batches: List[int] = []

    def cholesky_ex(A: torch.Tensor, *args: Any, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        L, info = real(A, *args, **kwargs)
        if A.dtype == torch.float32:
            info = info.clone()
            info[3] = 1                                                  # report row 3 as failed in float32
        else:
            float64_batches.append(A.shape[0])
        return L, info

    monkeypatch.setattr(torch.linalg, "cholesky_ex", cholesky_ex)
    factor = _covariance_cholesky(sigma)
    monkeypatch.setattr(torch.linalg, "cholesky_ex", real)
    assert float64_batches == [1]
    expected = real(sigma)[0].clone()
    expected[3] = real(sigma[3].double())[0].float()
    assert torch.equal(factor, expected)


# A185 (gap-paper-arm-offlive-routes-4): the Mahalanobis norm solves per head on gaussian_block, one host read per call.

def _block_diagonal_spd(n: int, H: int, d: int, kappa: float, generator: torch.Generator) -> torch.Tensor:
    sigma = torch.zeros(n, H * d, H * d, dtype=F64)
    for h in range(H):
        sigma[:, h * d:(h + 1) * d, h * d:(h + 1) * d] = _spd(n, d, kappa, generator) * (1.0 + h)
    return sigma


def test_mahalanobis_norm_per_head_equals_the_dense_solve_on_head_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    from vfe3.geometry.norms import MahalanobisNorm
    H, d = 3, 2
    K = H * d
    g = torch.Generator().manual_seed(13)
    sigma = _block_diagonal_spd(5, H, d, 50.0, g)
    mu = torch.randn(5, K, generator=g, dtype=F64)
    dense = MahalanobisNorm(K, family="gaussian_full")
    per_head = MahalanobisNorm(K, family="gaussian_block", head_dims=(d,) * H)
    torch.testing.assert_close(per_head(mu, sigma), dense(mu, sigma), rtol=1e-13, atol=1e-14)
    torch.testing.assert_close(per_head(mu.float(), sigma.float()), dense(mu, sigma).float(), rtol=1e-5, atol=1e-6)
    solves: List[Tuple[int, ...]] = []
    real = torch.cholesky_solve

    def counting(rhs: torch.Tensor, factor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        solves.append(tuple(factor.shape))
        return real(rhs, factor, *args, **kwargs)

    monkeypatch.setattr(torch, "cholesky_solve", counting)
    for norm, factor_shape in ((dense, (5, K, K)), (per_head, (5, H, d, d))):
        reads = _HostReads(monkeypatch)
        solves.clear()
        norm(mu, sigma)
        assert reads.count == 1 and solves == [factor_shape]
    monkeypatch.setattr(torch, "cholesky_solve", real)
    # Gradients: equal in the mean and on the head blocks; the per-head norm does not read the off-head blocks.
    mu_g, sigma_g = mu.clone().requires_grad_(), sigma.clone().requires_grad_()
    upstream = torch.randn(5, K, generator=g, dtype=F64)
    dense_grads = torch.autograd.grad((dense(mu_g, sigma_g) * upstream).sum(), (mu_g, sigma_g))
    head_grads = torch.autograd.grad((per_head(mu_g, sigma_g) * upstream).sum(), (mu_g, sigma_g))
    torch.testing.assert_close(head_grads[0], dense_grads[0], rtol=1e-12, atol=1e-14)
    on_blocks = torch.block_diag(*[torch.ones(d, d, dtype=torch.bool)] * H)            # (K, K)
    torch.testing.assert_close(head_grads[1][:, on_blocks], dense_grads[1][:, on_blocks], rtol=1e-12, atol=1e-14)
    assert bool((head_grads[1][:, ~on_blocks] == 0).all()) and bool((dense_grads[1][:, ~on_blocks] != 0).any())


def test_mahalanobis_norm_per_head_gradients_match_finite_differences() -> None:
    from vfe3.geometry.norms import MahalanobisNorm
    H, d, n = 2, 3, 2
    K = H * d
    g = torch.Generator().manual_seed(18)
    norm = MahalanobisNorm(K, family="gaussian_block", head_dims=(d,) * H)
    mu = torch.randn(n, K, generator=g, dtype=F64).requires_grad_()
    blocks = torch.stack([_spd(n, d, 10.0, g) for _ in range(H)], dim=1).requires_grad_()   # (n, H, d, d)

    def normed(m: torch.Tensor, b: torch.Tensor) -> torch.Tensor:          # sigma = blockdiag(sym(b_h)), (n, K, K)
        sym = 0.5 * (b + b.mT)
        zero = torch.zeros(n, d, d, dtype=F64)
        sigma = torch.cat([torch.cat([sym[:, h] if c == h else zero for c in range(H)], dim=-1) for h in range(H)],
                          dim=-2)
        return norm(m, sigma)

    assert torch.autograd.gradcheck(normed, (mu, blocks), eps=1e-6, atol=1e-7, rtol=1e-5)


def test_dense_mahalanobis_norm_is_bitwise_the_two_read_form() -> None:
    r"""Without ``head_dims`` the one-read exact path gives the value and the gradients of 5bb0a57 bit for bit."""
    from vfe3.geometry.norms import MahalanobisNorm
    from vfe3.numerics import safe_cholesky
    K = 6
    g = torch.Generator().manual_seed(16)
    sigma = (_spd(4, K, 30.0, g) * 2.0).requires_grad_()
    mu = torch.randn(4, K, generator=g, dtype=F64).requires_grad_()
    upstream = torch.randn(4, K, generator=g, dtype=F64)

    def reference(m: torch.Tensor, s: torch.Tensor) -> torch.Tensor:          # the exact path of 5bb0a57
        sigma64 = 0.5 * (s.double() + s.double().transpose(-1, -2))
        factor, ok = safe_cholesky(sigma64, rounds=0)
        assert bool(ok.all())
        solved = torch.cholesky_solve(m.double().unsqueeze(-1), factor).squeeze(-1)
        s2 = (m.double() * solved).sum(dim=-1, keepdim=True).to(m.dtype)
        return m * torch.sqrt(K / s2.clamp(min=1e-6))

    results = []
    for fn in (reference, MahalanobisNorm(K, family="gaussian_full")):
        out = fn(mu, sigma)
        results.append((out, *torch.autograd.grad((out * upstream).sum(), (mu, sigma))))
    for new, old in zip(results[1], results[0]):
        assert torch.equal(new, old)


def test_mahalanobis_norm_keeps_its_dense_fallback_under_head_dims() -> None:
    r"""A singular head block sends the call to the dense solve and its explicitly approximate fallback, as before."""
    from vfe3.geometry.norms import MahalanobisNorm
    H, d = 2, 3
    g = torch.Generator().manual_seed(14)
    sigma = _block_diagonal_spd(2, H, d, 10.0, g)
    sigma[1, :d, :d] = torch.zeros(d, d, dtype=F64)                      # token 1, head 0: singular
    mu = torch.randn(2, H * d, generator=g, dtype=F64)
    out = MahalanobisNorm(H * d, family="gaussian_block", head_dims=(d,) * H)(mu, sigma)
    dense = MahalanobisNorm(H * d, family="gaussian_full")(mu, sigma)
    assert bool(torch.isfinite(out).all())
    assert torch.equal(out, dense)


def test_model_builds_the_mahalanobis_norm_with_the_head_partition() -> None:
    values = _gate_model_values("gaussian_block", n_layers=1)
    model = VFEModel(_quiet_config(**values))
    assert model.block_norm.head_dims == tuple(model.group.irrep_dims)
    model = VFEModel(_quiet_config(**_gate_model_values("gaussian_full", n_layers=1)))
    assert model.block_norm.head_dims is None


# A150 (block-mlp-norm-5): the opt-in fused affine LayerNorm.

def test_fused_affine_layernorm_is_opt_in_and_matches_to_rounding() -> None:
    from vfe3.geometry.norms import AffineLayerNorm, get_norm
    K = 8
    g = torch.Generator().manual_seed(15)
    unfused = get_norm("layernorm")(K, eps=1e-6, affine=True)
    fused = get_norm("layernorm")(K, eps=1e-6, affine=True, fused=True)
    assert isinstance(fused, AffineLayerNorm) and fused.fused and not unfused.fused
    with torch.no_grad():
        unfused.weight.copy_(1.0 + 0.2 * torch.randn(K, generator=g))
        unfused.bias.copy_(0.1 * torch.randn(K, generator=g))
    fused.load_state_dict(unfused.state_dict())                          # same parameters and state-dict keys
    mu = (0.3 * torch.randn(2, 5, K, generator=g) + 0.1).requires_grad_()
    upstream = torch.randn(2, 5, K, generator=g)
    outputs, grads = [], []
    for norm in (unfused, fused):
        out = norm(mu, torch.ones(2, 5, K))
        outputs.append(out)
        grads.append(torch.autograd.grad((out * upstream).sum(), (mu, norm.weight, norm.bias)))
    torch.testing.assert_close(outputs[1], outputs[0], rtol=1e-6, atol=1e-6)
    for fused_grad, unfused_grad in zip(grads[1], grads[0]):
        torch.testing.assert_close(fused_grad, unfused_grad, rtol=1e-5, atol=1e-5)
    # bf16 input: both promote to float32 and return float32
    low = mu.detach().bfloat16()
    assert fused(low, None).dtype == torch.float32 and unfused(low, None).dtype == torch.float32
    torch.testing.assert_close(fused(low, None), unfused(low, None), rtol=1e-6, atol=1e-6)


def test_live_profile_fused_layernorm_loss_and_gradients_agree_to_rounding() -> None:
    r"""On the tiny live profile (affine block LayerNorm forced on) the opt-in fused norm moves the training loss and
    every parameter gradient by float32 rounding only; the default is the unfused form, so the live route is
    untouched."""
    from tests.profiles import live_values
    results = []
    for fused in (False, True):
        cfg = _quiet_config(**live_values(norm_type_block="layernorm", layernorm_affine=True, layernorm_fused=fused))
        torch.manual_seed(0)
        model = VFEModel(cfg)
        assert model.block_norm.fused is fused
        g = torch.Generator().manual_seed(1)
        ids = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.max_seq_len), generator=g)
        targets = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.max_seq_len), generator=g)
        _, loss, _ = model(ids, targets)
        loss.backward()
        results.append((loss.detach(), {n: p.grad for n, p in model.named_parameters() if p.grad is not None}))
    (loss_unfused, grads_unfused), (loss_fused, grads_fused) = results
    torch.testing.assert_close(loss_fused, loss_unfused, rtol=1e-6, atol=0.0)
    assert grads_fused.keys() == grads_unfused.keys()
    for name, grad in grads_unfused.items():
        torch.testing.assert_close(grads_fused[name], grad, rtol=1e-4, atol=1e-6, msg=name)


def test_layernorm_fused_is_a_validated_inert_aware_field() -> None:
    from tests.profiles import test_config
    assert test_config().layernorm_fused is False
    model = VFEModel(test_config(layernorm_fused=True))
    assert model.block_norm.fused is True
    assert VFEModel(test_config()).block_norm.fused is False
    for bad in (1, "yes", None):
        with pytest.raises(ValueError, match="layernorm_fused"):
            test_config(layernorm_fused=bad)
    assert "layernorm_fused" not in test_config(layernorm_fused=True).inactive_fields()
    assert "layernorm_fused" in test_config(layernorm_affine=False).inactive_fields()
    assert "layernorm_fused" in test_config(norm_type_block="none").inactive_fields()


# A086 (memory-local-learner-2): working-variance calibration leaves no two templates equal.

def _prototype(K: int = 4, expansion: int = 2) -> PrototypeAgentBlockMLP:
    torch.manual_seed(0)
    return PrototypeAgentBlockMLP(K, expansion, 0.0, inference="exact")


def test_calibration_with_fewer_rows_than_templates_breaks_every_tie() -> None:
    mlp = _prototype()                                                   # M = 8 templates, K = 4
    g = torch.Generator().manual_seed(3)
    mu = torch.randn(1, 3, 4, generator=g)                               # three calibration rows
    sigma = torch.rand(1, 3, 4, generator=g) + 0.5
    stats = mlp.calibrate_templates(mu, sigma, pull=0.09, generator=torch.Generator().manual_seed(1))
    assert torch.unique(mlp.nu.detach(), dim=0).shape[0] == 8                # 5bb0a57: 3 distinct of 8
    assert stats["distinct_inputs"] == 3 and stats["jittered_templates"] == 5
    for row in mu.reshape(-1, 4):                                        # every distinct input is a template once
        assert sum(bool(torch.equal(template, row)) for template in mlp.nu.detach()) == 1
    # The ties of 5bb0a57 never separated: AdamW on a random loss now moves every template differently.
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=1e-2)
    xi, c = torch.randn(6, 4, generator=g), torch.rand(6, 4, generator=g) + 0.5
    for _ in range(3):
        optimizer.zero_grad()
        out = mlp.forward_moments(xi, c)
        (out.mu * torch.randn(6, 4, generator=g)).sum().backward()
        optimizer.step()
    assert torch.unique(mlp.nu.detach(), dim=0).shape[0] == 8


def test_calibration_draws_over_distinct_rows_and_keeps_the_untied_draw() -> None:
    from vfe3.model.block_mlp import CueResultMemoryBlockMLP
    g = torch.Generator().manual_seed(4)
    rows = torch.randn(12, 4, generator=g)
    sigma = torch.rand(12, 4, generator=g) + 0.5
    # Twelve distinct rows for eight templates: the draw without replacement, as before.
    mlp = _prototype()
    mlp.calibrate_templates(rows, sigma, pull=0.2, generator=torch.Generator().manual_seed(5))
    expected = rows[torch.randperm(12, generator=torch.Generator().manual_seed(5))[:8]]
    assert torch.equal(mlp.nu.detach(), expected)
    # Repeated rows (the same block input in two windows): drawn once each, the rest jittered.
    repeated = torch.cat([rows[:5], rows[:5], rows[5:7]])                # 12 rows, 7 distinct
    stats = mlp.calibrate_templates(repeated, sigma, pull=0.2, generator=torch.Generator().manual_seed(6))
    assert torch.unique(mlp.nu.detach(), dim=0).shape[0] == 8
    assert stats["distinct_inputs"] == 7 and stats["jittered_templates"] == 1
    # The cue/result memory calibrates its cue bank by the same draw.
    torch.manual_seed(0)
    memory = CueResultMemoryBlockMLP(4, 2, 0.0, regression_rank=1)
    stats = memory.calibrate_templates(repeated, sigma, pull=0.2, generator=torch.Generator().manual_seed(6))
    assert torch.equal(memory.nu.detach(), mlp.nu.detach()) and torch.equal(memory.omega, memory.nu)
    assert stats["distinct_inputs"] == 7


# A088 (memory-local-learner-5): the diagonal cue/result memory books its elementwise cost.

def test_diagonal_cue_result_memory_books_its_elementwise_flops() -> None:
    from vfe3.model.block_mlp import block_mlp_build_metadata

    def metadata(K: int, e: int, layout: str, rank: Any) -> float:
        return block_mlp_build_metadata(enabled=True, mode="cue_result_memory", covariance="posterior", expansion=e,
                                        activation="gelu", dropout=0.0, covariance_floor=1e-4, embed_dim=K,
                                        irrep_dims=(K // 2, K // 2), n_layers=2, covariance_layout=layout,
                                        regression_rank=rank).flops_per_token

    K, e = 8, 2
    for rank in (1, 2):
        assert metadata(K, e, "diagonal", rank) == 2 * (e * K) * K * (24 + 4 * rank + 6 * rank ** 2)
    assert metadata(K, e, "diagonal", None) == 2 * (e * K) * (4 * K * K + 19 * K)
    assert metadata(K, e, "full", 1) == metadata(K, e, "full", None)      # the dense-record bound, unchanged
    assert metadata(K, e, "full", 1) > 2 * (25.0 / 3.0) * e * K ** 4 - 1e-6
    # At the launcher width the dense bound overstated the elementwise route by about four orders of magnitude.
    assert metadata(192, 8, "full", 1) / metadata(192, 8, "diagonal", 1) > 5e3


def test_frame_diagonal_memory_model_records_the_diagonal_flops() -> None:
    cfg = _quiet_config(**config_values(
        attention_mode="projected_predictive", predictive_readout="kernel", family="gaussian_frame_diagonal",
        decode_mode="linear", norm_type_block="none", block_coordinates="reference", use_block_mlp=True,
        block_mlp_mode="cue_result_memory", block_mlp_inference="exact", block_mlp_covariance="posterior",
        block_mlp_dropout=0.0, block_mlp_expansion=2, n_layers=2, max_seq_len=4, block_memory_regression_rank=2))
    model = VFEModel(cfg)
    K, M = cfg.embed_dim, cfg.embed_dim * cfg.block_mlp_expansion
    assert model.executable_build.block_mlp.flops_per_token == 2 * M * K * (24 + 4 * 2 + 6 * 4)


def _gate_model_values(family: str, n_layers: int) -> Dict[str, Any]:
    r"""The covariant reading of the intertwiner arm at K = 8: gauge gate, Mahalanobis norm, reference decoder, and a
    passthrough predictive covariance, so the encoded head blocks reach the gate unchanged."""
    return reference_k8_values(family=family, decode_mode="full_chunked", block_mlp_mode="gauge_gate",
                               norm_type_block="mahalanobis", block_norm_covariance="passthrough",
                               decode_coordinates="reference", block_coordinates="local", n_layers=n_layers,
                               layernorm_affine=False, block_mlp_template_init="absolute",
                               predictive_covariance="passthrough", attention_residual=True)


@pytest.mark.parametrize("family", ["gaussian_full", "gaussian_block"])
def test_gauge_gate_model_with_an_ill_conditioned_head_block_trains_in_float32(family: str) -> None:
    import vfe3.train as T
    cfg = _quiet_config(**_gate_model_values(family, n_layers=2))
    torch.manual_seed(0)
    model = VFEModel(cfg)
    with torch.no_grad():
        model.prior_bank.sigma_log_embed[0, 0] = math.log(1e-5)     # token 0, head 0: condition 1.5e6 against 15
    tokens = torch.tensor([[0, 3, 1, 2], [4, 0, 5, 6]])
    targets = torch.tensor([[3, 1, 2, 4], [0, 5, 6, 1]])
    capture: Dict[str, Any] = {}
    model.eval()
    with torch.no_grad():
        belief, logits = model.forward_beliefs(tokens, capture=capture, return_logits=True)
    assert belief.mu.dtype == torch.float32 and belief.sigma.dtype == torch.float32
    assert capture["block_mlp_output"].mu.dtype == torch.float32 and logits.dtype == torch.float32
    assert bool(torch.isfinite(logits).all())
    model.train()
    optimizer = T.build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    loss = T.train_step(model, optimizer, scheduler, tokens, targets, grad_clip=cfg.grad_clip,
                        grad_clip_scope=cfg.grad_clip_scope)
    assert math.isfinite(float(loss))
