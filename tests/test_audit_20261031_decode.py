r"""Audit 2026-10-31 remediation, package ``decode``: the KL decode boundary of ``vfe3/model/prior_bank.py`` (CPU).

numerics-1 (A007)                 ``decode_backward_ieee_fp32`` runs the backward GEMMs of the decode cross term
                                  (dlhs = G rhs, drhs = G^T lhs) at IEEE fp32 under a TF32 policy; default off.
prior-bank-decode-1 (A028)        ``decode_av_precision='fp64'`` upcasts the tables and the query before it forms
                                  c, mc, 1/sigma_v, the log-sums and per_pos, so its logits are a float64 bank's.
numerics-3 (A093)                 the second-order linear-decode variance is a centered moment, clamped at zero.
prior-bank-decode-2 (A096)        one eps-clamped query variance serves the trace and the log-det terms.
perf-live-path-3 (A154)           ``decode_bank_triton`` (opt-in; CPU tensors ignore it, see the Triton test file).
gap-checkpoint-...-4 (A181)       the decode CE checkpoints preserve no RNG state.
prior-bank-decode-4 (A199)        the expected-likelihood logits tile their rows under a byte budget.
gap-mixed-precision-routes-4      the bf16 linear-logit error bound counts the operand casts.
  (A262)
type-contracts-4 (A274)           the linear fused-CE Protocol declares sigma_q and canonical_frame.
dead-code-5 (A313)                PriorBank takes no gauge_group_name.
prior-bank-decode-6 (A320)        the registrations carry no base-table omission capabilities.
dead-code-6 (A329)                one chunk loop serves the diagonal and full KL decode CE (bitwise pins against the
                                  5bb0a57 bodies in ``tests/test_audit_20261031_decode_bitwise.py``).
prior-bank-decode-3 (A370)        reference_decode reads the decode frame.

TF32 does not exist on the CPU, so the TF32 checks emulate it: a dispatch mode rounds both operands of every float32
GEMM to TF32 (round to nearest even, 10 explicit mantissa bits) whenever the process policy is not ``"highest"``, the
behavior of cuBLAS under ``fp32_matmul_precision='high'``. Tiny models only.
"""
from __future__ import annotations

import copy
import inspect
import math
import warnings
from dataclasses import fields
from typing import Any, Iterator, List, Tuple

import pytest
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

import vfe3.model.prior_bank as bank_module
from tests.profiles import live_config, make_prior_bank, tiny_config
from vfe3.contracts import CanonicalFrameContext
from vfe3.model.model import VFEModel
from vfe3.model.prior_bank import (
    DecodeRegistration,
    EncodeRegistration,
    LinearFusedCECallable,
    PriorBank,
    _DECODERS,
    _decode_family,
    _linear_logits,
    get_decode_registration,
    register_decode,
)

aten = torch.ops.aten


@pytest.fixture(autouse=True)
def _quiet_and_restore_policy() -> Iterator[None]:
    previous, threads = torch.get_float32_matmul_precision(), torch.get_num_threads()
    torch.set_num_threads(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield
    torch.set_float32_matmul_precision(previous)
    torch.set_num_threads(threads)


# ------------------------------------------------------------------------------------- TF32 emulation

def _tf32(x: torch.Tensor) -> torch.Tensor:
    r"""Round float32 to TF32 (10 explicit mantissa bits), round to nearest even; other dtypes pass through."""
    if x.dtype is not torch.float32:
        return x
    i = x.contiguous().view(torch.int32)
    rounded = (i + 0x0FFF + ((i >> 13) & 1)) & -0x2000
    return rounded.view(torch.float32).reshape(x.shape)


_GEMM_OPERANDS = {aten.mm.default: (0, 1), aten.mm.out: (0, 1), aten.bmm.default: (0, 1), aten.bmm.out: (0, 1),
                  aten.addmm.default: (1, 2), aten.addmm.out: (1, 2), aten.addmm_.default: (1, 2),
                  aten.baddbmm.default: (1, 2)}


class _EmulatedTF32(TorchDispatchMode):
    r"""Round the operands of every float32 GEMM to TF32 while the process policy is not "highest", and record the
    policy each GEMM ran under."""

    def __init__(self) -> None:
        super().__init__()
        self.policies: List[str] = []

    def __torch_dispatch__(self, func: Any, types: Any, args: Tuple[Any, ...] = (), kwargs: Any = None) -> Any:
        kwargs = kwargs or {}
        operands = _GEMM_OPERANDS.get(func)
        if operands is not None:
            policy = torch.get_float32_matmul_precision()
            self.policies.append(policy)
            if policy != "highest":
                args = tuple(_tf32(a) if i in operands and isinstance(a, torch.Tensor) else a
                             for i, a in enumerate(args))
        return func(*args, **kwargs)


# ------------------------------------------------------------------------------------- shared builders

def _kl_bank(mode: str, *, V: int, K: int, tau: float, chunk: int, precision: str = "fp32",
             escalation: bool = False, unigram: bool = False, ieee_backward: bool = False,
             heads: int = 1, seed: int = 0) -> PriorBank:
    full = mode == "full_chunked"
    torch.manual_seed(seed)
    toggle = {"decode_backward_ieee_fp32": True} if ieee_backward else {}
    pb = make_prior_bank(V, K, 3, decode_mode=mode, family="gaussian_full" if full else "gaussian_diagonal",
                         diagonal_covariance=not full, decode_tau=tau, mu_init_std=1.0, decode_chunk_size=chunk,
                         decode_av_precision=precision, decode_ranking_fp64_escalation=escalation,
                         decode_unigram_prior=unigram, untie_decode_bank=True,
                         irrep_dims=[K // heads] * heads, **toggle)
    pb.decode_log_scale.requires_grad_(False)
    return pb


def _exact_diag_logits(mu_q: torch.Tensor, sq: torch.Tensor, mu_v: torch.Tensor, s_v: torch.Tensor,
                       tau: float) -> torch.Tensor:
    r"""-KL(N(mu_q, diag sq) || N(mu_v, diag s_v)) / tau, float64, in the unexpanded form."""
    mu_q, sq, mu_v, s_v = mu_q.double(), sq.double(), mu_v.double(), s_v.double()
    d = mu_q.unsqueeze(-2) - mu_v
    kl = 0.5 * ((sq.unsqueeze(-2) + d ** 2) / s_v - 1.0 + torch.log(s_v) - torch.log(sq.unsqueeze(-2))).sum(-1)
    return -kl / tau


# ------------------------------------------------------------------------------------- A028: the float64 island

@pytest.mark.parametrize("mode", ["diagonal_chunked", "full_chunked"])
@pytest.mark.parametrize("placement", ["near", "mid"])
def test_fp64_decode_logits_are_a_float64_banks(mode: str, placement: str) -> None:
    r"""At the module note's test point (K = 20, V = 64, sigma_v = 1e-4) the "fp64" logits are those of the same bank
    in float64 (``copy.deepcopy(bank).double()``), to 1e-8 on each row's five best tokens. Before the fix the island
    started from float32-rounded c, mc, 1/sigma_v and log-sums and missed by about 1e-1."""
    V, K, tau = 64, 20, 0.125                                        # tau exact in float32
    full = mode == "full_chunked"
    pb = _kl_bank(mode, V=V, K=K, tau=tau, chunk=24, precision="fp64", heads=4)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        pb.decode_mu_embed.copy_(torch.randn(V, K, generator=g))
        pb.decode_sigma_log_embed.fill_(math.log(1e-4))
    mu_v, s_v = pb.decode_mu_embed.detach(), torch.exp(pb.decode_sigma_log_embed.detach())
    t = torch.randint(0, V, (2, 16), generator=g)
    if placement == "near":
        mu_q = mu_v[t] + 0.5 * s_v[t].sqrt() * torch.randn(2, 16, K, generator=g)
    else:
        t2 = (t + 1 + torch.randint(0, V - 1, (2, 16), generator=g)) % V
        mu_q = 0.5 * (mu_v[t] + mu_v[t2])
    sq = s_v[t] * torch.exp(0.3 * torch.randn(2, 16, K, generator=g))
    sigma_q = torch.diag_embed(sq) if full else sq
    reference = copy.deepcopy(pb).double()
    with torch.no_grad():
        got = pb.decode(mu_q, sigma_q)
        want = reference.decode(mu_q.double(), sigma_q.double())
    assert got.dtype is torch.float64
    top = want.topk(5, dim=-1).indices
    # The full family's per-position log-determinant comes from the float32 Cholesky factor (the exclusion mask must
    # match the dense path's), a constant per row: compare the logits relative to each row's best token there.
    shift = (lambda x: x - x.gather(-1, top[..., :1])) if full else (lambda x: x)
    err = (shift(got) - shift(want)).gather(-1, top).abs().max().item()
    assert err <= 1e-8, err


@pytest.mark.parametrize("mode", ["diagonal_chunked", "full_chunked"])
def test_fp64_decode_ce_and_gradients_are_a_float64_banks(mode: str) -> None:
    r"""The fused CE (torch chunk loop) under "fp64" equals the float64 bank's CE, and its query and table gradients
    are the float64 bank's rounded to float32."""
    V, K, tau = 37, 8, 0.125
    full = mode == "full_chunked"
    pb = _kl_bank(mode, V=V, K=K, tau=tau, chunk=10, precision="fp64", unigram=True, heads=2)
    g = torch.Generator().manual_seed(2)
    with torch.no_grad():
        pb.decode_mu_embed.copy_(torch.randn(V, K, generator=g))
        pb.decode_sigma_log_embed.copy_(math.log(1e-3) + 0.5 * torch.randn(V, K, generator=g))
    pb.set_unigram_log_prior(torch.randint(1, 40, (V,), generator=g).float())
    t = torch.randint(0, V, (2, 6), generator=g)
    mu0 = pb.decode_mu_embed.detach()[t] + 0.02 * torch.randn(2, 6, K, generator=g)
    s0 = 1e-3 * torch.exp(0.3 * torch.randn(2, 6, K, generator=g))
    sigma0 = torch.diag_embed(s0) if full else s0
    targets = torch.randint(0, V, (2, 6), generator=g)
    targets[0, 2] = -100
    reference = copy.deepcopy(pb).double()

    def run(bank: PriorBank, dtype: torch.dtype) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        mu = mu0.to(dtype).requires_grad_()
        sigma = sigma0.to(dtype).requires_grad_()
        fn = bank.decode_ce_full_chunked if full else bank.decode_ce_diagonal_chunked
        ce = fn(mu, sigma, targets)
        grads = torch.autograd.grad(ce, [mu, sigma, bank.decode_mu_embed, bank.decode_sigma_log_embed])
        return ce, list(grads)

    ce, grads = run(pb, torch.float32)
    ce_ref, grads_ref = run(reference, torch.float64)
    assert ce.dtype is torch.float64
    torch.testing.assert_close(ce, ce_ref, rtol=1e-12, atol=0.0)
    for got, want in zip(grads, grads_ref):
        torch.testing.assert_close(got, want.to(got.dtype), rtol=2e-6, atol=1e-30)


def test_fp64_escalated_argmax_matches_the_exact_float64_argmax() -> None:
    r"""Midpoint queries (probe 6): with "fp64" and the ranking escalation the decode argmax equals the exact float64
    argmax in every row, as float32 with escalation does; before the fix about a quarter of the rows disagreed."""
    V, K, N, tau = 64, 20, 512, 0.125
    g = torch.Generator().manual_seed(7)
    mu_v = torch.randn(V, K, generator=g)
    t = torch.randint(0, V, (1, N), generator=g)
    t2 = (t + 1 + torch.randint(0, V - 1, (1, N), generator=g)) % V
    mu_q = 0.5 * (mu_v[t] + mu_v[t2])                                # float32 midpoints: near ties
    for sv in (1e-2, 1e-4):
        sq = torch.full((1, N, K), sv)
        exact = _exact_diag_logits(mu_q, sq, mu_v, torch.full((V, K), sv), tau).argmax(-1)
        for precision in ("fp32", "fp64"):
            pb = _kl_bank("diagonal_chunked", V=V, K=K, tau=tau, chunk=16, precision=precision, escalation=True)
            with torch.no_grad():
                pb.decode_mu_embed.copy_(mu_v)
                pb.decode_sigma_log_embed.fill_(math.log(sv))
                wrong = int((pb.decode(mu_q, sq).argmax(-1) != exact).sum())
            assert wrong == 0, (sv, precision, wrong)


@pytest.mark.parametrize("mode", ["diagonal_chunked", "full_chunked"])
def test_fp64_variances_keep_the_cached_overflow_check(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    r"""The float64 island upcasts the log table after the overflow check of the stored table, so repeated decodes of
    an unchanged table reduce it once (the identity/version cache of ``_check_max_log``) instead of once per call."""
    import vfe3.numerics as numerics

    full = mode == "full_chunked"
    pb = _kl_bank(mode, V=11, K=4, tau=0.3, chunk=4, precision="fp64", heads=2)
    reductions: List[int] = []
    original = numerics._max_log_exceeded
    monkeypatch.setattr(numerics, "_max_log_exceeded", lambda *a, **k: (reductions.append(1), original(*a, **k))[1])
    mu = torch.randn(1, 3, 4, generator=torch.Generator().manual_seed(8))
    sigma = torch.eye(4).expand(1, 3, 4, 4).clone() if full else torch.ones(1, 3, 4)
    fn = pb.decode_ce_full_chunked if full else pb.decode_ce_diagonal_chunked
    for _ in range(3):
        fn(mu, sigma, torch.tensor([[0, 5, 10]]))
        pb.decode(mu, sigma)
    assert len(reductions) == 1, len(reductions)


# ------------------------------------------------------------------------------------- A096: one eps clamp

def _eps_edge_bank(mode: str = "diagonal_chunked") -> Tuple[PriorBank, torch.Tensor, torch.Tensor]:
    r"""Probe 5: a query variance 1e-9 < eps in coordinate 0, where every decode variance sits at the eps floor."""
    V, K = 5, 4
    pb = _kl_bank(mode, V=V, K=K, tau=0.1, chunk=8).double()
    mu_v = torch.zeros(V, K, dtype=torch.float64)
    mu_v[1, 1] = 0.3
    mu_v[2:, 1] = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)
    log_s = torch.zeros(V, K, dtype=torch.float64)
    log_s[:, 0] = math.log(pb.eps)
    with torch.no_grad():
        pb.decode_mu_embed.copy_(mu_v)
        pb.decode_sigma_log_embed.copy_(log_s)
    mu_q = torch.zeros(1, 1, K, dtype=torch.float64)
    sq = torch.tensor([[[1e-9, 1.0, 1.0, 1.0]]], dtype=torch.float64)
    return pb, mu_q, sq


def test_a_query_variance_below_eps_keeps_the_nearest_tokens_apart() -> None:
    r"""Decode logit gaps equal the family oracle's (which clamps the variance at eps in both terms), the CE equals the
    oracle's cross-entropy, and the nearest token keeps its gradient. Before the fix tokens 0 and 1 tied at logit 0
    (exact gap 0.45) and d log p(token 1) / d mu_q was about 3e-6 against 1.8."""
    pb, mu_q, sq = _eps_edge_bank()
    oracle = _decode_family(pb, mu_q, sq, pb._tau_eff(None))
    for escalation in (False, True):
        pb.decode_ranking_fp64_escalation = escalation
        with torch.no_grad():
            got = pb.decode(mu_q, sq)
        torch.testing.assert_close(got - got[..., :1], oracle - oracle[..., :1], rtol=0.0, atol=1e-9)
    pb.decode_ranking_fp64_escalation = False
    mu, mu_oracle = mu_q.clone().requires_grad_(), mu_q.clone().requires_grad_()
    torch.log_softmax(pb.decode(mu, sq), -1)[0, 0, 1].backward()
    torch.log_softmax(_decode_family(pb, mu_oracle, sq, pb._tau_eff(None)), -1)[0, 0, 1].backward()
    assert abs(mu_oracle.grad[0, 0, 1].item()) > 1.0                 # about 1.8
    torch.testing.assert_close(mu.grad, mu_oracle.grad, rtol=1e-9, atol=1e-12)
    targets = torch.tensor([[1]])
    ce = pb.decode_ce_diagonal_chunked(mu_q, sq, targets)
    torch.testing.assert_close(ce, F.cross_entropy(oracle.reshape(1, -1), targets.reshape(-1)), rtol=1e-12, atol=0.0)


def test_one_clamp_is_bitwise_where_the_variances_exceed_eps() -> None:
    r"""With every query variance at or above eps the clamped trace term is the raw one, so the decode logits and the
    CE value and gradients equal the previous expression (raw lhs, clamped per_pos) bitwise."""
    V, K = 11, 4
    pb = _kl_bank("diagonal_chunked", V=V, K=K, tau=0.3, chunk=4)
    g = torch.Generator().manual_seed(3)
    mu_q = torch.randn(2, 3, K, generator=g)
    sq = 0.5 + torch.rand(2, 3, K, generator=g)
    sq[0, 0, 0] = pb.eps                                             # exactly at the floor
    mu_v = pb._decode_mu_table()
    s_v = torch.exp(pb._decode_sigma_log_table()).clamp(min=pb.eps)
    c = mu_v.mean(0, keepdim=True)
    mc_q, mc_v, inv_v = mu_q - c, mu_v - c, 1.0 / s_v
    lhs = torch.cat([sq + mc_q ** 2, -2.0 * mc_q], -1)                # the raw variance in the trace term
    rhs = torch.cat([inv_v, mc_v * inv_v], -1)
    bias = (mc_v.square() * inv_v).sum(-1) + torch.log(s_v).sum(-1)
    with bank_module.ieee_fp32_matmul():
        a_v = lhs @ rhs.T + bias
    per_pos = K + torch.log(sq.clamp(min=pb.eps)).sum(-1, keepdim=True)
    previous = -(0.5 * (a_v - per_pos)).clamp(min=0.0) / pb._tau_eff(None)
    with torch.no_grad():
        assert torch.equal(pb.decode(mu_q, sq), previous.detach())
    targets = torch.tensor([[0, 4, 10], [7, -100, 1]])
    mu, s = mu_q.clone().requires_grad_(), sq.clone().requires_grad_()
    ce = pb.decode_ce_diagonal_chunked(mu, s, targets)
    lp = torch.log_softmax(previous, -1)
    valid = targets != -100
    want = -(lp.gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1) * valid).sum() / valid.sum()
    torch.testing.assert_close(ce, want, rtol=2e-6, atol=0.0)


# ------------------------------------------------------------------------------------- A093: second-order variance

def _second_order_bank(V: int = 256, K: int = 16) -> PriorBank:
    torch.manual_seed(0)
    pb = make_prior_bank(V, K, 3, decode_mode="linear", decode_unigram_prior=False, decode_chunk_size=100)
    with torch.no_grad():
        pb.output_proj_weight.normal_(std=1.0)
        pb.output_proj_bias.zero_()
    return pb


def _concentrated_query(pb: PriorBank, gap: float) -> Tuple[torch.Tensor, int]:
    r"""A (1, 1, K) query whose softmax puts 1 - p_top = ``gap`` on the other classes (bisection in float64), and the
    top class."""
    W = pb.output_proj_weight.detach().double()
    top = int(W.norm(dim=-1).argmax())
    direction = W[top] / W[top].norm()
    lo, hi = 0.0, 400.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        p = torch.softmax(W @ (mid * direction), -1)
        lo, hi = (mid, hi) if 1.0 - p.max().item() > gap else (lo, mid)
    mu = (hi * direction).float().reshape(1, 1, -1)
    assert int((mu.double().reshape(-1) @ W.T).argmax()) == top
    return mu, top


def _dense_delta_term(pb: PriorBank, mu: torch.Tensor, sigma: torch.Tensor) -> float:
    r"""Float64 :math:`\tfrac12\sum_k\sigma_k\mathrm{Var}_p(w_k)` at the softmax of the emulated TF32 logits GEMM (the
    logits keep the caller's policy; the finding is the cancellation of the moment GEMM)."""
    W = pb.output_proj_weight.detach().double()
    z = _tf32(mu).double() @ _tf32(pb.output_proj_weight.detach()).double().T + pb.output_proj_bias.detach().double()
    p = torch.softmax(z, -1)                                                                  # (1, 1, V)
    var = p @ (W * W) - (p @ W) ** 2                                                          # (1, 1, K)
    return 0.5 * float((sigma.double() * var).sum())


@pytest.mark.parametrize("gap", [3.6e-5, 1e-5, 1.8e-8])
def test_second_order_variance_is_nonnegative_and_accurate_at_a_concentrated_softmax(gap: float) -> None:
    r"""Under emulated TF32 the delta-method term of a confident token is nonnegative and within 1e-2 relative of
    float64 (absolute 1e-9 once the term itself is below float32 resolution). Before the fix the moment GEMM ran at
    TF32 and E_p[w^2] - E_p[w]^2 cancelled to about -7e-3 of either sign."""
    pb = _second_order_bank()
    mu, top = _concentrated_query(pb, gap)
    sigma = torch.full_like(mu, 0.5)
    targets = torch.tensor([[top]])
    torch.set_float32_matmul_precision("high")
    with torch.no_grad(), _EmulatedTF32():
        result = pb.decode_ce_linear_chunked(mu, targets, sigma_q=sigma, return_stats=True)
    delta = float(result.objective - result.ce)
    want = _dense_delta_term(pb, mu, sigma)
    assert delta >= 0.0, delta
    assert abs(delta - want) <= 1e-2 * want + 1e-9, (delta, want)


def test_second_order_moment_gemm_runs_at_ieee_fp32() -> None:
    pb = _second_order_bank(V=37, K=4)
    mu = torch.randn(1, 3, 4)
    torch.set_float32_matmul_precision("high")
    with torch.no_grad(), _EmulatedTF32() as mode:
        pb.decode_ce_linear_chunked(mu, torch.tensor([[1, 2, 3]]), chunk_size=10, sigma_q=torch.ones_like(mu))
    # Per chunk: the logits GEMM at the caller's policy, the moment GEMM pinned.
    assert mode.policies == ["high", "highest"] * 4, mode.policies


# ------------------------------------------------------------------------------------- A007: the backward IEEE pin

def _required_prior_bank_keyword(name: str) -> bool:
    parameter = inspect.signature(PriorBank.__init__).parameters[name]
    return parameter.kind is inspect.Parameter.KEYWORD_ONLY and parameter.default is inspect.Parameter.empty


def test_decode_backward_ieee_fp32_is_an_opt_in_bool_field() -> None:
    r"""A validated bool, introduced at False (a record without it reloads at the unpinned backward), marked inert
    where the cross-term GEMMs already run at IEEE precision, and a required PriorBank keyword. The launcher
    dictionaries are not read: the user retired the rule that every field appears in train_vfe3.py (2026-11-01)."""
    from vfe3.config_history import INTRODUCED_AFTER_CAA0114

    assert INTRODUCED_AFTER_CAA0114["decode_backward_ieee_fp32"] is False
    assert _required_prior_bank_keyword("decode_backward_ieee_fp32")
    with pytest.raises(ValueError, match="decode_backward_ieee_fp32"):
        tiny_config(decode_backward_ieee_fp32=1)
    on = dict(decode_mode="diagonal_chunked", fp32_matmul_precision="high")
    assert "decode_backward_ieee_fp32" not in tiny_config(**on).inactive_fields()
    assert "decode_backward_ieee_fp32" not in tiny_config(decode_mode="full_chunked", family="gaussian_full",
                                                          fp32_matmul_precision="high").inactive_fields()
    for off in (dict(decode_mode="linear"), dict(decode_mode="expected_likelihood_chunked"),
                dict(on, decode_av_precision="fp64"), dict(on, fp32_matmul_precision="highest")):
        assert "decode_backward_ieee_fp32" in tiny_config(**off).inactive_fields(), off
    model = VFEModel(tiny_config(decode_mode="diagonal_chunked", decode_backward_ieee_fp32=True))
    assert model.prior_bank.decode_backward_ieee_fp32 is True
    with pytest.raises(ValueError, match="decode_backward_ieee_fp32 must be bool"):
        make_prior_bank(7, 4, 3, decode_backward_ieee_fp32=1)                 # type: ignore[arg-type]


@pytest.mark.parametrize("mode", ["diagonal_chunked", "full_chunked"])
@pytest.mark.parametrize("ieee", [False, True])
def test_the_torch_route_backward_gemms_follow_the_toggle(mode: str, ieee: bool) -> None:
    r"""Under a process policy of "high" the forward cross term is pinned in both settings; the two backward GEMMs per
    chunk run at "high" by default and at "highest" with the toggle."""
    full = mode == "full_chunked"
    pb = _kl_bank(mode, V=13, K=4, tau=0.3, chunk=5, ieee_backward=ieee)
    mu = torch.randn(1, 3, 4, requires_grad=True)
    # sigma takes no gradient, so the full family's Cholesky backward adds no GEMM of its own
    sigma = (torch.eye(4).expand(1, 3, 4, 4) if full else torch.ones(1, 3, 4)).clone()
    fn = pb.decode_ce_full_chunked if full else pb.decode_ce_diagonal_chunked
    torch.set_float32_matmul_precision("high")
    with _EmulatedTF32() as forward_mode:
        ce = fn(mu, sigma, torch.tensor([[0, 4, 12]]))
    with _EmulatedTF32() as backward_mode:
        ce.backward()
    assert set(forward_mode.policies) == {"highest"}
    backward = [p for p in backward_mode.policies]
    assert len(backward) == 2 * 3 and set(backward) == {"highest" if ieee else "high"}, backward


def _variance_gradient_error(ieee: bool, s: float) -> float:
    r"""Relative error of the decode log-variance gradient against float64 under emulated TF32 (the numerics-1 probe:
    V = 96, K = 32, 128 rows, tau 0.1, three vocabulary chunks, s_q = t_v = s, unit-spread means offset by 3, odd
    entries near duplicates of the even ones so the target's softmax is not one-hot, queries near their targets)."""
    V, K, tau = 96, 32, 0.1
    pb = _kl_bank("diagonal_chunked", V=V, K=K, tau=tau, chunk=32, ieee_backward=ieee)
    g = torch.Generator().manual_seed(2)
    centers = 3.0 + torch.randn(V, K, generator=g, dtype=torch.float64)
    centers[1::2] = centers[0::2] + 0.5 * math.sqrt(s) * torch.randn(V // 2, K, generator=g, dtype=torch.float64)
    with torch.no_grad():
        pb.decode_mu_embed.copy_(centers.float())
        pb.decode_sigma_log_embed.copy_(math.log(s) + 0.1 * torch.randn(V, K, generator=g))
    targets = torch.randint(0, V, (1, 128), generator=g)
    mu0 = (centers[targets] + 0.3 * math.sqrt(s) * torch.randn(1, 128, K, generator=g, dtype=torch.float64)).float()
    sigma0 = s * torch.exp(0.1 * torch.randn(1, 128, K, generator=g))
    reference = copy.deepcopy(pb).double()

    def grad(bank: PriorBank, dtype: torch.dtype) -> torch.Tensor:
        ce = bank.decode_ce_diagonal_chunked(mu0.to(dtype), sigma0.to(dtype), targets)
        return torch.autograd.grad(ce, bank.decode_sigma_log_embed)[0]

    want = grad(reference, torch.float64)
    torch.set_float32_matmul_precision("high")
    with _EmulatedTF32():
        got = grad(pb, torch.float32)
    return float((got.double() - want).norm() / want.norm())


def test_the_toggle_restores_ieee_accuracy_of_the_decode_variance_gradient() -> None:
    r"""At s = 0.01 (mean(mc_q^2)/s near 100) the emulated TF32 backward carries a relative error of order 1e-1 in the
    decode log-variance gradient; the toggle brings it to the IEEE fp32 level."""
    off = _variance_gradient_error(False, 1e-2)
    on = _variance_gradient_error(True, 1e-2)
    assert off > 5e-2, off
    assert on < 1e-2 and on < 0.2 * off, (on, off)


def test_the_toggle_is_bitwise_on_the_cpu_live_route() -> None:
    r"""The CPU runs IEEE fp32 whatever the policy, so the toggle leaves the tiny live route's loss and every parameter
    gradient bitwise unchanged there."""
    results = []
    for ieee in (False, True):
        torch.manual_seed(0)
        cfg = live_config(decode_backward_ieee_fp32=ieee, fp32_matmul_precision=None)
        model = VFEModel(cfg)
        g = torch.Generator().manual_seed(1)
        tokens = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.max_seq_len), generator=g)
        targets = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.max_seq_len), generator=g)
        _, loss, _ = model(tokens, targets)
        loss.backward()
        results.append((loss.detach(), [p.grad for p in model.parameters() if p.grad is not None]))
    (loss_a, grads_a), (loss_b, grads_b) = results
    assert torch.equal(loss_a, loss_b)
    assert len(grads_a) == len(grads_b) and all(torch.equal(a, b) for a, b in zip(grads_a, grads_b))


# ------------------------------------------------------------------------------------- A154: CPU side of the toggle

def test_decode_bank_triton_is_an_opt_in_bool_field_ignored_by_cpu_tensors() -> None:
    r"""A validated bool, introduced at False, inert off the fused Triton KL decode CE, a required PriorBank keyword,
    and without effect on CPU tensors (the torch preparation runs, bitwise)."""
    from vfe3.config_history import INTRODUCED_AFTER_CAA0114

    assert INTRODUCED_AFTER_CAA0114["decode_bank_triton"] is False
    assert _required_prior_bank_keyword("decode_bank_triton")
    with pytest.raises(ValueError, match="decode_bank_triton"):
        tiny_config(decode_mode="diagonal_chunked", decode_bank_triton="yes")
    assert "decode_bank_triton" not in tiny_config(decode_mode="diagonal_chunked",
                                                   decode_ce_triton=True).inactive_fields()
    for off in (dict(decode_mode="linear"), dict(decode_mode="diagonal_chunked", decode_ce_triton=False)):
        assert "decode_bank_triton" in tiny_config(**off).inactive_fields(), off
    results = []
    for fused in (False, True):
        pb = _kl_bank("diagonal_chunked", V=13, K=4, tau=0.3, chunk=5)
        pb.decode_ce_triton, pb.decode_bank_triton = True, fused
        mu = torch.randn(2, 3, 4, generator=torch.Generator().manual_seed(4)).requires_grad_()
        ce = pb.decode_ce_diagonal_chunked(mu, torch.ones(2, 3, 4), torch.tensor([[0, 4, 12], [1, 2, 3]]))
        assert pb.realized_decode_ce_kernel == "torch"
        results.append((ce, torch.autograd.grad(ce, [mu, pb.decode_mu_embed, pb.decode_sigma_log_embed])))
    assert torch.equal(results[0][0], results[1][0])
    assert all(torch.equal(a, b) for a, b in zip(results[0][1], results[1][1]))


# ------------------------------------------------------------------------------------- A181: checkpoint RNG state

def _decode_ce_calls(mode: str) -> Tuple[PriorBank, Any]:
    torch.manual_seed(0)
    if mode == "linear":
        pb = make_prior_bank(13, 4, 3, decode_mode="linear", decode_chunk_size=5)
    else:
        full = mode == "full_chunked"
        pb = make_prior_bank(13, 4, 3, decode_mode=mode, family="gaussian_full" if full else "gaussian_diagonal",
                             diagonal_covariance=not full, decode_chunk_size=5)
    g = torch.Generator().manual_seed(1)
    mu = torch.randn(1, 3, 4, generator=g)
    sigma = torch.eye(4).expand(1, 3, 4, 4).clone() if mode == "full_chunked" else 0.5 + torch.rand(1, 3, 4, generator=g)
    targets = torch.tensor([[0, 6, 12]])

    def call(second_order: bool = False) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        m, s = mu.clone().requires_grad_(), sigma.clone().requires_grad_()
        if mode == "linear":
            ce = pb.decode_ce_linear_chunked(m, targets, sigma_q=s if second_order else None,
                                             return_stats=second_order)
            ce = ce.objective if second_order else ce
        else:
            ce = getattr(pb, "decode_ce_" + mode)(m, s, targets)
        leaves = [m, s] + [p for p in pb.parameters() if p.requires_grad]
        return ce, list(torch.autograd.grad(ce, leaves, allow_unused=True))
    return pb, call


@pytest.mark.parametrize("mode, second_order", [("diagonal_chunked", False), ("full_chunked", False),
                                                ("expected_likelihood_chunked", False), ("linear", False),
                                                ("linear", True)])
def test_decode_ce_checkpoints_preserve_no_rng_state(mode: str, second_order: bool,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    pb, call = _decode_ce_calls(mode)
    pb.decode_ce_checkpoint = "never"
    ce_ref, grads_ref = call(second_order)
    reads: List[str] = []
    for name in ("get_rng_state", "set_rng_state"):
        original = getattr(torch, name)
        monkeypatch.setattr(torch, name, lambda *a, _o=original, _n=name, **k: (reads.append(_n), _o(*a, **k))[1])
    pb.decode_ce_checkpoint = "always"
    ce, grads = call(second_order)
    assert reads == []
    assert torch.equal(ce, ce_ref)
    assert all((a is None and b is None) or torch.equal(a, b) for a, b in zip(grads, grads_ref))


# ------------------------------------------------------------------------------------- A199: EL logits transient

class _LargestOutput(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.largest = 0

    def __torch_dispatch__(self, func: Any, types: Any, args: Tuple[Any, ...] = (), kwargs: Any = None) -> Any:
        out = func(*args, **(kwargs or {}))
        for t in (out if isinstance(out, (tuple, list)) else (out,)):
            if isinstance(t, torch.Tensor) and t.dim() >= 3 and t.shape[-1] == 4:
                self.largest = max(self.largest, t.numel() * t.element_size())
        return out


def test_expected_likelihood_logits_bound_their_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""The logits path tiles its rows so no (rows, Vc, K) transient exceeds the byte budget, and the logits equal the
    untiled broadcast bitwise."""
    V, K = 23, 4
    pb = make_prior_bank(V, K, 3, decode_mode="expected_likelihood_chunked", decode_chunk_size=10)
    g = torch.Generator().manual_seed(5)
    mu = torch.randn(3, 7, K, generator=g)
    sigma = 0.5 + torch.rand(3, 7, K, generator=g)
    mu_v = pb._decode_mu_table().detach()
    s_v = torch.exp(pb._decode_sigma_log_table().detach()).clamp(min=pb.eps)
    tau = pb._tau_eff(None).detach()
    untiled = torch.cat([-0.5 * ((mu.unsqueeze(-2) - mu_v[v0:v0 + 10]) ** 2 / (sigma.unsqueeze(-2) + s_v[v0:v0 + 10])
                                 + torch.log(sigma.unsqueeze(-2) + s_v[v0:v0 + 10])).sum(-1) / tau
                         for v0 in range(0, V, 10)], dim=-1)
    budget = 5 * 10 * K * 4                                         # five rows of a full chunk
    monkeypatch.setattr(bank_module, "EXPECTED_LIKELIHOOD_TILE_BYTES", budget)
    with torch.no_grad(), _LargestOutput() as spy:
        logits = pb.decode(mu, sigma)
    assert 0 < spy.largest <= budget, spy.largest
    assert torch.equal(logits, untiled)


# ------------------------------------------------------------------------------------- A262: bf16 logit error bound

def test_bf16_linear_logit_error_counts_the_operand_casts() -> None:
    r"""Under bf16 autocast the logits obey :math:`|\delta z| \le u(2\sum_k|x_kw_k| + |z|)`, :math:`u = 2^{-8}` (to
    first order), and a cancelling input exceeds the output-rounding term :math:`u|z|` alone."""
    pb = make_prior_bank(64, 32, 3, decode_mode="linear")
    pb.decode_logits_amp_dtype = torch.bfloat16
    g = torch.Generator().manual_seed(6)
    W = torch.randn(64, 32, generator=g)
    x = 4.0 * torch.randn(16, 32, generator=g)
    u = 2.0 ** -8
    z = (x.double() @ W.double().T)
    bound = u * (2.0 * (x.double().abs() @ W.double().abs().T) + z.abs()) * (1.0 + 4 * u)
    err = (_linear_logits(pb, x, W).double() - z).abs()
    assert bool((err <= bound).all()), float((err / bound).max())
    # A cancelling input: x orthogonal to w_0, so z_0 vanishes while sum_k |x_k w_0k| does not.
    xc = (W[1] - (W[1] @ W[0]) / (W[0] @ W[0]) * W[0]).reshape(1, -1)
    zc = xc.double() @ W.double().T
    errc = (_linear_logits(pb, xc, W).double() - zc).abs()
    assert errc[0, 0] > 10.0 * u * zc[0, 0].abs(), (float(errc[0, 0]), float(zc[0, 0]))
    assert r"2\sum_k |x_k w_k|" in _linear_logits.__doc__


# ------------------------------------------------------------------------------------- A274, A313, A320

def test_fused_ce_protocols_accept_every_keyword_the_model_passes() -> None:
    r"""The model passes ``return_stats`` to every fused CE, ``canonical_frame`` to the KL and linear ones, and
    ``sigma_q`` to the linear one (``vfe3/model/model.py`` fused branch)."""
    passed = {"linear": {"return_stats", "canonical_frame", "sigma_q"},
              "diagonal_chunked": {"return_stats", "canonical_frame"},
              "full_chunked": {"return_stats", "canonical_frame"},
              "expected_likelihood_chunked": {"return_stats"}}
    linear = set(inspect.signature(LinearFusedCECallable.__call__).parameters)
    assert passed["linear"] <= linear, sorted(passed["linear"] - linear)
    for mode, names in passed.items():
        accepted = set(inspect.signature(get_decode_registration(mode).fused_ce).parameters)
        assert names <= accepted, (mode, sorted(names - accepted))


def test_prior_bank_takes_no_gauge_group_name() -> None:
    with pytest.raises(TypeError, match="gauge_group_name"):
        make_prior_bank(7, 4, 3, gauge_group_name="block_glk")
    assert not hasattr(VFEModel(tiny_config()).prior_bank, "gauge_group_name")


def test_registrations_carry_no_base_table_omission_fields() -> None:
    for record in (DecodeRegistration, EncodeRegistration):
        names = {f.name for f in fields(record)}
        assert not {"can_omit_base_mean", "can_omit_base_variance"} & names, (record, names)
    with pytest.raises(TypeError):
        register_decode("_audit_20261031_omit", can_omit_base_mean=True)              # type: ignore[call-arg]
    assert "_audit_20261031_omit" not in _DECODERS


# ------------------------------------------------------------------------------------- A370: reference_decode frame

def _frame_fixture() -> Tuple[CanonicalFrameContext, torch.Tensor, torch.Tensor]:
    B, N, H, d = 2, 3, 2, 2
    g = torch.Generator().manual_seed(5)
    blocks = torch.eye(d, dtype=torch.float64) + 0.5 * torch.randn(B, N, H, d, d, generator=g, dtype=torch.float64)
    frame = CanonicalFrameContext(blocks, torch.linalg.inv(blocks), True)
    mu = torch.randn(B, N, H * d, generator=g, dtype=torch.float64)
    sig = torch.exp(0.4 * torch.randn(B, N, H * d, generator=g, dtype=torch.float64))
    return frame, mu, sig


def _frame_bank(mode: str, family: str, coords: str) -> PriorBank:
    torch.manual_seed(0)
    pb = make_prior_bank(13, 4, 3, decode_mode=mode, family=family, diagonal_covariance=family != "gaussian_full",
                         decode_tau=0.1, mu_init_std=0.5, decode_chunk_size=5, untie_decode_bank=True,
                         decode_coordinates=coords, irrep_dims=[2, 2]).double()
    with torch.no_grad():
        pb.decode_sigma_log_embed.copy_(0.5 * torch.randn(13, 4, dtype=torch.float64))
    return pb


@pytest.mark.parametrize("mode, family, coords", [
    ("diagonal_chunked", "gaussian_diagonal", "reference"),
    ("full_chunked", "gaussian_full", "reference"),
    ("diagonal_chunked", "gaussian_frame_diagonal", "local"),
    ("diagonal_chunked", "gaussian_frame_diagonal", "reference"),
])
def test_reference_decode_reads_the_decode_frame(mode: str, family: str, coords: str) -> None:
    r"""Given the frame, ``reference_decode`` scores the moments ``decode()`` scores: the two agree under log-softmax
    (probe 3 found differences of 245 and 56 without the frame); without it, or with a frame in local coordinates
    of a frame-free family, it refuses."""
    frame, mu, sig = _frame_fixture()
    pb = _frame_bank(mode, family, coords)
    if family == "gaussian_full":
        a = 0.5 * torch.randn(2, 3, 4, 4, generator=torch.Generator().manual_seed(9), dtype=torch.float64)
        sig = a @ a.transpose(-1, -2) + 0.3 * torch.eye(4, dtype=torch.float64)
    with torch.no_grad():
        got = pb.reference_decode(mu, sig, canonical_frame=frame)
        want = pb.decode(mu, sig, canonical_frame=frame)
    torch.testing.assert_close(torch.log_softmax(got, -1), torch.log_softmax(want, -1), rtol=0.0, atol=1e-10)
    with pytest.raises(ValueError, match="canonical_frame"):
        pb.reference_decode(mu, sig)
    local = _frame_bank("diagonal_chunked", "gaussian_diagonal", "local")
    with pytest.raises(ValueError, match="canonical_frame"):
        local.reference_decode(mu, torch.exp(mu), canonical_frame=frame)
