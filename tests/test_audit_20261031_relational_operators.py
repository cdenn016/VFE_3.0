r"""Audit 2026-10-31 remediation, package ``relational-operators`` (findings A020, A025, A141, A146, A091, A180, A184,
A187, A257, A261, A272, A273).

A020: the covariance floor runs with autocast off, so a bf16 run of the full-layout moment-matched minimal agent trains
and is bitwise equal to fp32, as ``inactive_fields`` claims. A025: the opt-in ``content_link_latent_guard`` keeps a
receiver's previous latent factor where the specified fixed-point step raises its MAP energy. A141: the opt-in fp32
exponential of the content chart agrees with the float64 chart, and the default chart is bitwise the committed one on
the live route. A146: the causal head-pair pooling equals the full contraction and reads only the causal sources.
A091: the full-link receiving-law retry rewrites only the failed entries. A184: the head-block floor equals the dense
one. A187: the causal lowrank edge coefficients equal the all-pairs ones at half the matrix functions. A257: the
noisy-key parameters are noted for the matched baseline. A180 (the knob comment), A261, A272 and A273: corrected texts.
Tiny CPU shapes, float64 where values are compared to rounding.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import pytest
import torch

from tests.profiles import live_config, make_relational_operators, test_config
from vfe3.attention_prior import attention_log_prior
from vfe3.contracts import CanonicalFrameContext
from vfe3.model import edge_links
from vfe3.model import relational_operators as ro

F64 = torch.float64
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _two_threads() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(min(2, previous))
    yield
    torch.set_num_threads(previous)


def _relative(got: torch.Tensor, want: torch.Tensor) -> float:
    return float((got.double() - want.double()).abs().max()) / max(float(want.double().abs().max()), 1e-300)


def _assert_pairs(got: List[Optional[torch.Tensor]], want: List[Optional[torch.Tensor]], tol: Optional[float]) -> None:
    r"""Equal ``None`` pattern, and each tensor pair bitwise equal (``tol`` None) or within ``tol`` relative."""
    assert len(got) == len(want)
    for k, (a, b) in enumerate(zip(got, want)):
        assert (a is None) == (b is None), k
        if a is not None:
            assert (torch.equal(a, b) if tol is None else _relative(a, b) <= tol), (k, _relative(a, b))


def _comment_before(text: str, anchor: str) -> str:
    r"""The comment block that ends at the line holding ``anchor`` in a source text."""
    lines = text[:text.index(anchor)].splitlines()
    block: List[str] = []
    for line in reversed(lines[:-1] if lines and not lines[-1].strip().startswith("#") else lines):
        if not line.strip().startswith("#"):
            break
        block.append(line)
    return "\n".join(reversed(block))


# ---------------------------------------------------------------------------------------------------------------------
# A020: bf16 autocast and the whitened covariance floor
# ---------------------------------------------------------------------------------------------------------------------

def _dense_floor_case(seed: int = 0) -> Tuple[Any, torch.Tensor, torch.Tensor, CanonicalFrameContext]:
    r"""A full-layout minimal-agent operator, an SPD covariance, an SPD anchor and a dense frame (float32)."""
    H, d, N = 2, 2, 3
    K = H * d
    g = torch.Generator().manual_seed(seed)
    ops = make_relational_operators([d] * H, 3, readout_mode="predictive_content", predictive_readout="minimal_agent",
                                    predictive_covariance="moment_matched", predictive_covariance_floor=0.3,
                                    predictive_covariance_floor_mode="max")
    factor = torch.randn(2, N, K, K, generator=g)
    sigma = factor @ factor.mT + 0.5 * torch.eye(K)
    anchor_factor = torch.randn(2, N, K, K, generator=g)
    anchor = anchor_factor @ anchor_factor.mT / K + 0.4 * torch.eye(K)
    U = torch.eye(K) + 0.3 * torch.randn(2, N, K, K, generator=g)
    return ops, sigma, anchor, CanonicalFrameContext(U, torch.linalg.inv(U))


@pytest.mark.parametrize("mode", ["max", "relative", "prior"])
def test_a020_floor_is_float32_and_bitwise_under_cpu_bf16_autocast(mode: str) -> None:
    ops, sigma, anchor, frame = _dense_floor_case()
    ops.predictive_covariance_floor_mode = mode
    leaves = [x.clone().requires_grad_(True) for x in (sigma, anchor)]
    reference = ops.floor_term(leaves[1], leaves[0], frame, diagonal=False)
    reference.sum().backward()
    want = [reference.detach()] + [x.grad for x in leaves]          # "relative" reads no anchor, "prior" no sigma
    leaves = [x.clone().requires_grad_(True) for x in (sigma, anchor)]
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        floor = ops.floor_term(leaves[1], leaves[0], frame, diagonal=False)
    assert floor.dtype == torch.float32                    # "max" returned bfloat16 before the fix
    floor.sum().backward()                                 # and its spectral clip's backward raised
    _assert_pairs([floor.detach()] + [x.grad for x in leaves], want, None)


_AMP_ROUTES = {f"{inference}_{covariance}_{mode}": (inference, covariance, mode)
               for inference, covariance in (("exact", "moment_matched"), ("mean_field", "moment_matched"),
                                             ("mean_field", "spread_corrected"))
               for mode in ("max", "relative", "prior")}


@pytest.mark.parametrize("route", sorted(_AMP_ROUTES))
def test_a020_full_layout_minimal_agent_bf16_route_is_inert_and_bitwise(route: str, device: torch.device) -> None:
    r"""The routes of the finding (``inactive_fields`` marks ``amp_dtype`` inert there): no reduced-precision op, and
    the bf16 step equals the fp32 step bit for bit in loss, every gradient and the eval logits."""
    from tests.test_amp_routes_20261012 import _MINIMAL, _check_inert_route
    inference, covariance, mode = _AMP_ROUTES[route]
    values = dict(_MINIMAL, family="gaussian_full", decode_mode="full_chunked", predictive_covariance=covariance,
                  predictive_covariance_floor_mode=mode, minimal_agent_inference=inference)
    _check_inert_route(values, device)


# ---------------------------------------------------------------------------------------------------------------------
# A025: the guarded latent receiver step
# ---------------------------------------------------------------------------------------------------------------------

_SIGMA_L = 0.1                                             # the default content_link_latent_sigma


def _latent_problem(sender: float, seed: int, batch: Tuple[int, ...] = (2,)) -> Tuple[Any, ...]:
    from tests.test_latent_links_20261005 import _inputs, _latent_ops, _problem
    ops = _latent_ops(True, seed=seed, sender=sender, receiver=0.5)
    data = _inputs(seed=seed + 1, batch=batch)
    ctx, sources = _problem(ops, data["mu"], data["sigma_diag"], data["beta"], data["compact"])
    return ops, data, ctx, sources


def _overshooting_problem() -> Tuple[Any, ...]:
    r"""The first seeded problem on which the specified step at the default sigma_L raises some, not all, F_i."""
    for sender in (1.0, 2.0, 4.0):
        for seed in range(6):
            ops, data, ctx, sources = _latent_problem(sender, seed)
            with torch.no_grad():
                A1 = edge_links.infer_latent_receivers(ctx, data["beta"], sources, sigma=_SIGMA_L, steps=1,
                                                       damping=1.0)
                F0 = edge_links.latent_receiver_energy(ctx.A, ctx.A, ctx, data["beta"], sources, _SIGMA_L)
                F1 = edge_links.latent_receiver_energy(A1, ctx.A, ctx, data["beta"], sources, _SIGMA_L)
            rose = ~torch.isfinite(F1) | (F1 > F0)
            if bool(rose.any()) and not bool(rose.all()):
                return ops, data, ctx, sources, A1, F0, F1
    raise AssertionError("no seeded problem overshoots at the default sigma_L; widen the search")


def test_a025_specified_step_raises_the_map_energy_and_the_guard_keeps_every_receiver_descending() -> None:
    ops, data, ctx, sources, A_free, F0, F_free = _overshooting_problem()
    beta = data["beta"]
    rose = ~torch.isfinite(F_free) | (F_free > F0)                                      # the defect: (B, N)
    with torch.no_grad():
        A_safe = edge_links.infer_latent_receivers(ctx, beta, sources, sigma=_SIGMA_L, steps=1, damping=1.0,
                                                   guard=True)
        F_safe = edge_links.latent_receiver_energy(A_safe, ctx.A, ctx, beta, sources, _SIGMA_L)
    assert bool(torch.isfinite(A_safe).all()) and bool(torch.isfinite(F_safe).all())
    assert bool((F_safe <= F0).all())
    keep = rose[..., None, None].expand_as(A_safe)
    assert torch.equal(A_safe[~keep], A_free[~keep])            # a descending receiver takes the step bit for bit
    assert torch.equal(A_safe[keep], ctx.A.detach()[keep])      # an overshooting one keeps its amortized factor
    with torch.no_grad():                                       # the default stays the specified iteration
        assert torch.equal(edge_links.infer_latent_receivers(ctx, beta, sources, sigma=_SIGMA_L, steps=1,
                                                             damping=1.0, guard=False), A_free)


@pytest.mark.parametrize("damping", [1.0, 0.5])
def test_a025_guarded_steps_never_raise_the_energy(damping: float) -> None:
    r"""The T-step result is the T-th iterate, so F_i(A^(T)) must be finite and nonincreasing in T for every
    receiver."""
    _, data, ctx, sources, _, F0, _ = _overshooting_problem()
    beta = data["beta"]
    previous = F0
    with torch.no_grad():
        for steps in (1, 2, 3):
            A = edge_links.infer_latent_receivers(ctx, beta, sources, sigma=_SIGMA_L, steps=steps, damping=damping,
                                                  guard=True)
            current = edge_links.latent_receiver_energy(A, ctx.A, ctx, beta, sources, _SIGMA_L)   # (B, N) F_i
            assert bool(torch.isfinite(A).all()) and bool(torch.isfinite(current).all()), steps
            assert bool((current <= previous).all()), steps
            previous = current


def test_a025_guarded_step_gradient_matches_finite_differences() -> None:
    r"""The guard is a per-receiver ``torch.where``: away from the acceptance boundary the guarded step is smooth, and
    its unrolled graph (the kept receivers' identity, the accepted ones' Newton-like step) passes gradcheck."""
    _, data, ctx, sources, _, F0, F1 = _overshooting_problem()
    finite = torch.isfinite(F1)
    assert float((F1 - F0)[finite].abs().min()) > 1e-3          # no receiver near the boundary for eps = 1e-6
    beta = data["beta"]
    A0 = ctx.A.detach().clone().requires_grad_(True)

    def step(A: torch.Tensor) -> torch.Tensor:
        return edge_links.infer_latent_receivers(ctx._replace(A=A), beta, sources, sigma=_SIGMA_L, steps=1,
                                                 damping=1.0, guard=True)

    assert torch.autograd.gradcheck(step, (A0,), eps=1e-6, atol=1e-6, rtol=1e-4)


def test_a025_readout_reads_the_operator_guard_and_trains_finitely(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_latent_links_20261005 import _readout
    ops, data, _, _, _, _, _ = _overshooting_problem()
    seen: List[bool] = []
    real = edge_links.infer_latent_receivers

    def spy(*args: Any, guard: bool = False, **kwargs: Any) -> torch.Tensor:
        seen.append(guard)
        return real(*args, guard=guard, **kwargs)

    monkeypatch.setattr(edge_links, "infer_latent_receivers", spy)
    ops.content_link_latent_guard = True
    ops.content_link_latent_steps = 2
    mean, cov = _readout(ops, data["mu"], data["sigma_diag"], data["compact"], log_prior=None)
    (mean.square().sum() + cov.sum()).backward()
    assert seen == [True]
    grads = [p.grad for p in ops.parameters() if p.grad is not None]
    assert grads and all(bool(torch.isfinite(g).all()) for g in grads)


def test_a025_config_field_reaches_the_operators_and_is_marked_inert_without_latent_links() -> None:
    from vfe3.model.model import VFEModel
    plain = test_config(attention_mode="projected_predictive", content_link_chart="exponential")
    assert "content_link_latent_guard" in plain.inactive_fields()
    with pytest.raises(ValueError, match="content_link_latent_guard"):
        test_config(content_link_latent_guard=1)
    cfg = test_config(attention_mode="projected_predictive", content_link_chart="exponential",
                      content_link_edges="lowrank", content_link_edge_rank=2, content_link_latent=True,
                      content_link_latent_guard=True)
    assert "content_link_latent_guard" not in cfg.inactive_fields()
    torch.manual_seed(0)
    model = VFEModel(cfg)
    assert all(ops.content_link_latent_guard for ops in model.relational_operators_by_layer)


# ---------------------------------------------------------------------------------------------------------------------
# A141: the opt-in fp32 exponential of the content chart
# ---------------------------------------------------------------------------------------------------------------------

def _chart_maps(K: int, H: int, cond: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Float64 stored maps T (H, d, K) with cond(T) = ``cond`` and O (H, K, d) with a generator O T of moderate norm."""
    g = torch.Generator().manual_seed(seed)
    Q1, _ = torch.linalg.qr(torch.randn(K, K, generator=g, dtype=F64))
    Q2, _ = torch.linalg.qr(torch.randn(K, K, generator=g, dtype=F64))
    s = torch.logspace(0.0, -float(torch.log10(torch.tensor(cond, dtype=F64))), K, dtype=F64)
    T = Q1 @ torch.diag(s) @ Q2                                                       # (K, K) rows by head
    O = 0.6 * torch.randn(K, K, generator=g, dtype=F64) @ torch.linalg.inv(T) / K ** 0.5   # O T of moderate norm
    d = K // H
    return T.reshape(H, d, K), O.reshape(K, H, d).permute(1, 0, 2).contiguous()


@pytest.mark.parametrize("cond", [1.0, 30.0, 1e3])
def test_a141_fp32_exponential_matches_the_float64_chart_in_values_and_gradients(cond: float) -> None:
    r"""O' to float32 rounding at every cond(T); the gradients of T and O to about eps_32 cond(T) (the documented
    limit of the option: their float32 Frechet adjoint reads the cotangent G T^{-T})."""
    K, H = 8, 2
    T0, O0 = _chart_maps(K, H, cond, seed=int(cond))
    W = torch.randn(H, K, K // H, generator=torch.Generator().manual_seed(5), dtype=F64)
    results = {}
    for precision in ("fp64", "fp32"):
        T, O = T0.clone().requires_grad_(True), O0.clone().requires_grad_(True)
        receive = ro.chart_receive_map(T, O, "exponential", precision)
        assert receive.dtype == F64
        (receive * W).sum().backward()
        results[precision] = (receive.detach(), T.grad, O.grad)
    (value, *grads), (value64, *grads64) = results["fp32"], results["fp64"]
    assert 0.0 < _relative(value, value64) <= 2e-6, _relative(value, value64)   # ran in fp32, at its rounding
    for got, want in zip(grads, grads64):
        assert _relative(got, want) <= max(2e-5, 5e-6 * cond), (cond, _relative(got, want))
    # The default is the float64 chart bit for bit; an unknown precision is refused.
    assert torch.equal(ro.chart_receive_map(T0, O0, "exponential"), results["fp64"][0])
    with pytest.raises(ValueError, match="content_link_exp_precision"):
        ro.chart_receive_map(T0, O0, "exponential", "fp16")


def _chart_cfg(**over: Any) -> Any:
    values: Dict[str, Any] = dict(attention_mode="projected_predictive", content_link_chart="exponential", n_layers=2)
    values.update(over)
    return test_config(**values)


def _perturb_content(layers: Any, seed: int = 2) -> None:
    r"""Move every stored O off its initialization (O = 0 is the identity exponential, exact in any precision)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for ops in layers:
            if getattr(ops, "O", None) is not None:
                for O in ops._stored_runs("O"):
                    O.add_(0.3 * torch.randn(O.shape, generator=g, dtype=O.dtype))


def _loss_and_grads(cfg: Any) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
    from vfe3.model.model import VFEModel
    torch.manual_seed(0)
    model = VFEModel(cfg)
    _perturb_content(model.relational_operators_by_layer)
    g = torch.Generator().manual_seed(1)
    shape = (cfg.batch_size, cfg.max_seq_len)
    ids = torch.randint(0, cfg.vocab_size, shape, generator=g)
    targets = torch.randint(0, cfg.vocab_size, shape, generator=g)
    _, loss, _ = model(ids, targets)
    loss.backward()
    return loss.detach(), {name: p.grad for name, p in model.named_parameters()}


def test_a141_operators_shared_charts_and_inert_marks() -> None:
    from vfe3.model.model import VFEModel
    cfg = _chart_cfg(content_link_exp_precision="fp32")
    assert "content_link_exp_precision" not in cfg.inactive_fields()
    torch.manual_seed(0)
    model = VFEModel(cfg)
    layers = list(model.relational_operators_by_layer)
    _perturb_content(layers)
    assert layers and all(ops.chart_exp_precision == "fp32" for ops in layers)
    per_layer = [ops.content_factor_runs()[1] for ops in layers]
    with ro.shared_content_charts(layers):                       # one batched (L, K, K) fp32-exponential call
        shared = [ops.content_factor_runs()[1] for ops in layers]
    for a, b in zip(per_layer, shared):                          # a batched matrix_exp may round differently
        for x, y in zip(a, b):
            assert _relative(x, y) <= 1e-6
    fp64 = [ro.chart_receive_runs(ops._stored_runs("T"), ops._stored_runs("O"), "exponential") for ops in layers]
    assert 0.0 < max(_relative(x, y) for a, b in zip(per_layer, fp64) for x, y in zip(a, b)) <= 1e-5
    # The gradient receiver and the edge links keep the float64 chart, and the configuration marks the field there.
    receiver = make_relational_operators([2, 2], 1, readout_mode="full_link_update", content_link_chart="exponential",
                                         content_link_exp_precision="fp32")
    assert receiver.chart_exp_precision == "fp64"
    edged = make_relational_operators([2, 2], 1, predictive_covariance="moment_matched",
                                      content_link_chart="exponential", content_link_edges="lowrank",
                                      content_link_edge_rank=2, content_link_exp_precision="fp32")
    assert edged.chart_exp_precision == "fp64"
    assert "content_link_exp_precision" in test_config().inactive_fields()          # projected_gradient, affine
    assert "content_link_exp_precision" in _chart_cfg(content_link_edges="lowrank",
                                                      content_link_edge_rank=2).inactive_fields()
    assert "content_link_exp_precision" in _chart_cfg(content_link_chart="first_order").inactive_fields()
    with pytest.raises(ValueError, match="content_link_exp_precision"):
        test_config(content_link_exp_precision="fp16")
    with pytest.raises(ValueError, match="content_link_exp_precision"):
        make_relational_operators([2, 2], 1, content_link_exp_precision="bf16")


def test_a141_fp32_chart_trains_close_to_float64() -> None:
    loss64, grads64 = _loss_and_grads(_chart_cfg())
    loss32, grads32 = _loss_and_grads(_chart_cfg(content_link_exp_precision="fp32"))
    assert _relative(loss32, loss64) <= 1e-5
    for name, grad in grads64.items():
        other = grads32[name]
        assert (grad is None) == (other is None), name
        if grad is not None:
            assert bool(torch.isfinite(other).all()), name
            assert _relative(other, grad) <= 1e-3, (name, _relative(other, grad))


def _committed_chart_receive(T_stack: torch.Tensor, O_stack: torch.Tensor, chart: str,
                             exp_precision: str = "fp64") -> Tuple[torch.Tensor, torch.Tensor]:
    r""":func:`vfe3.model.relational_operators._chart_receive` as committed at 5bb0a57 (float64 exponential only)."""
    from vfe3.cpu_linalg import single_thread_for_batched_cpu
    assert exp_precision == "fp64"
    with single_thread_for_batched_cpu(T_stack):
        inverse = torch.linalg.inv_ex(T_stack).inverse
    if chart == "first_order":
        return inverse, inverse + O_stack
    return inverse, torch.linalg.matrix_exp(O_stack @ T_stack) @ inverse


def _counted(calls: Dict[str, int], name: str, real: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        calls[name] += 1
        return real(*args, **kwargs)
    return wrapper


def test_live_route_loss_and_gradients_are_bitwise_the_committed_ones(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Rule 2 of the remediation brief on the tiny live profile: the one changed function the live route runs (the
    chart of the shared content link) gives the committed loss and every gradient bit for bit, and the other changes
    sit behind switches the route has off (moment-matched covariance, edge links, the gradient receiver)."""
    from vfe3.gradients import full_link
    cfg = live_config()
    calls = {name: 0 for name in ("floor_term", "_causal_row_tiles", "edge_coefficients", "_spd_precision",
                                  "_content_link_exp_fp32")}
    monkeypatch.setattr(ro.RelationalOperators, "floor_term",
                        _counted(calls, "floor_term", ro.RelationalOperators.floor_term))
    monkeypatch.setattr(ro, "_causal_row_tiles", _counted(calls, "_causal_row_tiles", ro._causal_row_tiles))
    monkeypatch.setattr(ro, "_content_link_exp_fp32",
                        _counted(calls, "_content_link_exp_fp32", ro._content_link_exp_fp32))
    monkeypatch.setattr(edge_links, "edge_coefficients",
                        _counted(calls, "edge_coefficients", edge_links.edge_coefficients))
    monkeypatch.setattr(full_link, "_spd_precision", _counted(calls, "_spd_precision", full_link._spd_precision))
    loss, grads = _loss_and_grads(cfg)
    assert calls["_content_link_exp_fp32"] == 0
    if cfg.predictive_covariance == "passthrough":
        assert calls["floor_term"] == 0 and calls["_causal_row_tiles"] == 0
    if cfg.content_link_edges == "off":
        assert calls["edge_coefficients"] == 0
    if cfg.attention_mode == "projected_predictive":
        assert calls["_spd_precision"] == 0
    monkeypatch.setattr(ro, "_chart_receive", _committed_chart_receive)
    committed_loss, committed_grads = _loss_and_grads(cfg)
    assert torch.equal(loss, committed_loss)
    for name, grad in grads.items():
        other = committed_grads[name]
        assert (grad is None) == (other is None), name
        if grad is not None:
            assert torch.equal(grad, other), name


# ---------------------------------------------------------------------------------------------------------------------
# A146: causal head-pair pooling
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("symmetric, block", [(False, False), (True, False), (False, True)],
                         ids=["blockwise", "symmetric_pairs", "head_block_family"])   # symmetric pairs: diagonal only
@pytest.mark.parametrize("prior", ["causal", "causal_alibi"])
def test_a146_causal_pooling_equals_the_full_contraction(symmetric: bool, block: bool, prior: str) -> None:
    from tests.test_audit_20261020_covariance_layout import _inputs, _operators, _relative_error, _update_and_grads
    H, d, N = 3, 2, 9
    x = _inputs(H, d, N, (2,), prior, block, seed=13)
    full = _update_and_grads(_operators(H, d, False, symmetric, False, block, F64, x), x, F64, False, False)
    causal = _update_and_grads(_operators(H, d, False, symmetric, False, block, F64, x), x, F64, True, False)
    assert _relative_error(causal, full) <= 1e-12


def test_a146_causal_pooling_reads_only_the_causal_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_audit_20261020_covariance_layout import _inputs, _operators, _update_and_grads
    H, d, N = 2, 2, 9
    assert ro._causal_row_tiles(N) == [(0, 3), (3, 6), (6, 9)]
    assert ro._causal_row_tiles(1) == [(0, 1)] and ro._causal_row_tiles(5) == [(0, 2), (2, 4), (4, 5)]
    x = _inputs(H, d, N, (2,), "causal", False, seed=17)
    seen: List[Tuple[int, int]] = []
    real = torch.Tensor.__matmul__

    def spy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # The pooling GEMM c @ source: (..., H, H, I, S) by (..., H, H, S, d d).
        if a.dim() >= 4 and a.shape[-4:-2] == (H, H) and b.shape[-1] == d * d:
            seen.append((a.shape[-2], a.shape[-1]))
        return real(a, b)

    monkeypatch.setattr(torch.Tensor, "__matmul__", spy)
    _update_and_grads(_operators(H, d, False, False, False, False, F64, x), x, F64, True, False)
    assert seen == [(3, 3), (3, 6), (3, 9)]                     # each query tile against its causal sources
    seen.clear()
    _update_and_grads(_operators(H, d, False, False, False, False, F64, x), x, F64, False, False)
    assert seen == [(N, N)]


def test_a146_dense_route_causal_pooling_equals_the_full_contraction() -> None:
    from tests.test_audit_20261020_covariance_layout import _inputs, _operators
    H, d, N = 2, 2, 7
    x = _inputs(H, d, N, (2,), "causal_alibi", True, seed=19)
    outs = []
    for causal in (False, True):
        ops = _operators(H, d, False, False, False, False, F64, x)
        leaves = {k: x[k].clone().requires_grad_(True) for k in ("mu", "sigma", "logits", "blocks", "inverse")}
        beta = torch.softmax(leaves["logits"] + x["log_prior"], dim=-1)
        frame = CanonicalFrameContext(leaves["blocks"], leaves["inverse"], block_diagonal=True)
        kwargs = {"causal_sources": True} if causal else {}
        out = ops._predictive_covariance_update_dense(leaves["mu"], leaves["sigma"], beta, frame, None, **kwargs)
        (out * x["weights"]).sum().backward()
        outs.append([out.detach()] + [leaves[k].grad for k in leaves] + [ops.T.grad, ops.O.grad])
    _assert_pairs(outs[1], outs[0], 1e-12)


# ---------------------------------------------------------------------------------------------------------------------
# A091: the full-link receiving-law retry writes back only the failed entries
# ---------------------------------------------------------------------------------------------------------------------

def _float32_failure(d: int = 3) -> torch.Tensor:
    r"""A float32 SPD matrix whose float32 Cholesky fails while the float64 Cholesky of the same values succeeds."""
    lam = torch.tensor([1.0, 1e-4, 1e-9], dtype=F64)
    for trial in range(4000):
        g = torch.Generator().manual_seed(trial)
        Q, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=F64))
        A = (Q @ torch.diag(lam) @ Q.T).float()
        A = 0.5 * (A + A.T)
        if int(torch.linalg.cholesky_ex(A)[1]) != 0 and int(torch.linalg.cholesky_ex(A.double())[1]) == 0:
            return A
    raise AssertionError("no float32-only Cholesky failure found")


def test_a091_receiving_law_retry_keeps_the_float32_inverse_of_healthy_entries() -> None:
    from vfe3.gradients import full_link
    g = torch.Generator().manual_seed(3)
    factor = torch.randn(2, 3, 3, 3, generator=g)
    healthy = factor @ factor.mT + 0.5 * torch.eye(3)                      # (2, 3, d, d) float32 SPD
    target = healthy.clone()
    target[1, 2] = _float32_failure()                                      # one later source fails in float32
    keep = torch.ones(2, 3, dtype=torch.bool)
    keep[1, 2] = False
    before = full_link.receiver_law_retry_counter(target.device)
    leaf = target.clone().requires_grad_(True)
    out = full_link._spd_precision(leaf)
    assert int(full_link.receiver_law_retry_counter(target.device) - before) == 1
    out.sum().backward()
    # Healthy entries: the float32 inverse and its gradient bit for bit.
    ref_leaf = healthy.clone().requires_grad_(True)
    reference = torch.cholesky_inverse(torch.linalg.cholesky(ref_leaf))
    reference.sum().backward()
    assert torch.equal(out.detach()[keep], reference.detach()[keep])
    assert torch.equal(leaf.grad[keep], ref_leaf.grad[keep])
    # The failed entry: the float64 inverse cast back, and the gradient of the float64 computation.
    wide_leaf = target[1, 2].double().requires_grad_(True)
    wide = torch.cholesky_inverse(torch.linalg.cholesky(wide_leaf))
    wide.sum().backward()
    assert torch.equal(out.detach()[1, 2], wide.detach().float())
    assert bool(torch.isfinite(leaf.grad).all())
    assert _relative(leaf.grad[1, 2], wide_leaf.grad) <= 1e-6


def test_a091_full_link_prefix_receivers_ignore_a_failing_later_source() -> None:
    r"""The context build of the gradient receiver: an ill-conditioned last token leaves earlier receivers bitwise."""
    from vfe3.gradients import full_link
    H, d, N = 2, 2, 3
    K = H * d
    ops = make_relational_operators([d] * H, 3, readout_mode="full_link_update")
    g = torch.Generator().manual_seed(4)
    mu = torch.randn(1, N, K, generator=g)
    sigma = torch.rand(1, N, K, generator=g) + 0.5
    U = torch.eye(d) + 0.4 * torch.randn(1, N, H, d, d, generator=g)        # (1, N, H, d, d) block frames
    frame = CanonicalFrameContext(U, torch.linalg.inv(U), block_diagonal=True)
    kwargs = dict(tau=1.0, kl_max=100.0, log_prior=attention_log_prior("causal", N, N), receiver_noise=1e-6)
    base = full_link.build_full_link_context(mu, sigma, mu, sigma, frame, ops, **kwargs)
    changed = None
    # A last token whose channel covariance R_h2 = T_h C_2 T_h^T + nu I fails to factor in float32 but not in float64.
    for scale in (1e4, 1e5, 1e6, 1e7):
        for pattern in ((scale, 1 / scale, 1 / scale, 1 / scale), (scale, 1 / scale, scale, 1 / scale),
                        (1 / scale, scale, 1 / scale, 1 / scale)):
            bad_sigma = sigma.clone()
            bad_sigma[0, 2] = torch.tensor(pattern)
            before = full_link.receiver_law_retry_counter(mu.device)
            try:
                trial = full_link.build_full_link_context(mu, bad_sigma, mu, bad_sigma, frame, ops, **kwargs)
            except (torch.linalg.LinAlgError, ValueError):   # float64 failed as well: too ill-conditioned
                continue
            if int(full_link.receiver_law_retry_counter(mu.device) - before) >= 1:
                changed = trial
                break
        if changed is not None:
            break
    assert changed is not None, "no last token found whose channel covariance fails only in float32"
    assert torch.equal(changed.Lambda[..., :2, :, :], base.Lambda[..., :2, :, :])
    assert torch.equal(changed.eta[..., :2, :], base.eta[..., :2, :])


# ---------------------------------------------------------------------------------------------------------------------
# A184: the head-block family's floor on its head blocks
# ---------------------------------------------------------------------------------------------------------------------

def _block_floor_case(seed: int) -> Tuple[torch.Tensor, ...]:
    H, d, N = 3, 2, 4
    K = H * d
    g = torch.Generator().manual_seed(seed)
    eye_h = torch.eye(H, dtype=F64)
    embed = lambda X: torch.einsum("...hab,hg->...hagb", X, eye_h).reshape(*X.shape[:-3], K, K)
    A = torch.randn(2, N, H, d, d, generator=g, dtype=F64)
    sigma = embed(A @ A.mT + 0.3 * torch.eye(d, dtype=F64))
    P = torch.randn(2, N, H, d, d, generator=g, dtype=F64)
    anchor = embed(P @ P.mT / d + 0.2 * torch.eye(d, dtype=F64))
    U = torch.eye(d, dtype=F64) + 0.4 * torch.randn(2, N, H, d, d, generator=g, dtype=F64)
    return sigma, anchor, U, torch.linalg.inv(U)


@pytest.mark.parametrize("mode", ["max", "relative"])
def test_a184_head_block_floor_equals_the_dense_floor(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    sigma, anchor, U, U_inv = _block_floor_case(seed=21)
    H, d = U.shape[-3], U.shape[-1]
    shapes: List[Tuple[int, ...]] = []
    dense: Dict[bool, int] = {False: 0, True: 0}
    real = ro._whitened_max
    monkeypatch.setattr(ro, "_whitened_max", lambda P, R: shapes.append(tuple(P.shape)) or real(P, R))
    results = {}
    for block in (False, True):
        ops = make_relational_operators([d] * H, 2, readout_mode="predictive_content",
                                        predictive_covariance="moment_matched", predictive_covariance_floor=0.4,
                                        predictive_covariance_floor_mode=mode, block_covariance=block).double()
        reference = ops._reference_covariance
        monkeypatch.setattr(ops, "_reference_covariance",
                            lambda *a, block=block, **k: dense.__setitem__(block, dense[block] + 1) or reference(*a, **k))
        leaves = [x.clone().requires_grad_(True) for x in (sigma, anchor, U, U_inv)]
        frame = CanonicalFrameContext(leaves[2], leaves[3], block_diagonal=True)
        floor = ops.floor_term(leaves[1], leaves[0], frame, diagonal=False)
        (floor * torch.linspace(0.5, 1.5, floor.numel(), dtype=F64).reshape(floor.shape)).sum().backward()
        results[block] = [floor.detach()] + [x.grad for x in leaves]        # "relative" reads no anchor
    off_head = torch.block_diag(*[torch.ones(d, d, dtype=torch.bool)] * H).logical_not()
    # The family's covariances are head-block diagonal (structural zeros off the blocks), so the gradients with respect
    # to the dense (K, K) inputs are compared on the head blocks: the dense whitened max also answers perturbations of
    # the anchor's off-head zeros, which no gaussian_block producer can move. The head-block floor gives them 0.
    on_blocks = lambda grads: [g if g is None or k not in (1, 2) else g.masked_fill(off_head, 0.0)
                               for k, g in enumerate(grads)]
    _assert_pairs(results[True], on_blocks(results[False]), 1e-12)
    for grad in results[True][1:3]:
        assert grad is None or bool((grad[..., off_head] == 0).all())
    assert dense == {False: 1, True: 0}                      # the head-block floor forms no dense K x K C_i
    assert bool((results[True][0][..., off_head] == 0).all())               # exact zeros off the head blocks
    if mode == "max":
        assert shapes == [(2, 4, H * d, H * d), (2, 4, H, d, d)]             # dense K x K, then per head block


@pytest.mark.parametrize("mode", ["max", "relative"])
def test_a184_head_block_floor_passes_gradcheck(mode: str) -> None:
    sigma, anchor, U, _ = _block_floor_case(seed=23)
    H, d = U.shape[-3], U.shape[-1]
    ops = make_relational_operators([d] * H, 2, readout_mode="predictive_content",
                                    predictive_covariance="moment_matched", predictive_covariance_floor=0.4,
                                    predictive_covariance_floor_mode=mode, block_covariance=True).double()
    sym = lambda X: 0.5 * (X + X.mT)

    def floor(sigma_blocks: torch.Tensor, anchor_blocks: torch.Tensor, U_blocks: torch.Tensor) -> torch.Tensor:
        frame = CanonicalFrameContext(U_blocks, torch.linalg.inv(U_blocks), block_diagonal=True)
        return ops.floor_term(ro._embed_head_blocks(sym(anchor_blocks)), ro._embed_head_blocks(sym(sigma_blocks)),
                              frame, diagonal=False)                                     # (1, 2, K, K)

    inputs = tuple(x[:1, :2].clone().requires_grad_(True)
                   for x in (ro._head_blocks(sigma, d), ro._head_blocks(anchor, d), U))   # (1, 2, H, d, d) each
    assert torch.autograd.gradcheck(floor, inputs, eps=1e-6, atol=1e-7, rtol=1e-5)


# ---------------------------------------------------------------------------------------------------------------------
# A187: causal lowrank edge coefficients
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [F64, torch.float32])
def test_a187_causal_edge_coefficients_equal_the_all_pairs_ones(dtype: torch.dtype,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    B_, N, K, r, H = 2, 5, 4, 2, 2
    g = torch.Generator().manual_seed(23)
    A0 = torch.randn(B_, N, K, r, generator=g, dtype=F64).to(dtype)
    B0 = torch.randn(B_, N, K, r, generator=g, dtype=F64).to(dtype)
    log_prior = attention_log_prior("causal_alibi", N, N, n_heads=H, alibi_slope=0.5, dtype=dtype)
    beta = torch.softmax(torch.randn(B_, H, N, N, generator=g, dtype=F64).to(dtype) + log_prior, dim=-1)
    W = torch.randn(B_, N, N, r, r, generator=g, dtype=F64).to(dtype)
    counts: List[int] = []
    real = edge_links.phi1
    monkeypatch.setattr(edge_links, "phi1", lambda X: counts.append(X[..., 0, 0].numel()) or real(X))
    results = {}
    for causal in (False, True):
        A, B = A0.clone().requires_grad_(True), B0.clone().requires_grad_(True)
        Phi, _, mask = edge_links.edge_coefficients(A, B, beta, "lowrank", causal)
        (Phi * W).sum().backward()
        results[causal] = (Phi.detach(), mask, A.grad, B.grad)
    assert counts == [B_ * N * N, B_ * N * (N + 1) // 2]                     # about half the matrix functions
    assert torch.equal(results[True][0], results[False][0])
    assert torch.equal(results[True][1], results[False][1])
    for got, want in zip(results[True][2:], results[False][2:]):
        assert _relative(got, want) <= (1e-14 if dtype == F64 else 1e-6)


def test_a187_readout_passes_the_causal_flag_and_keeps_its_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_edge_links_20261005 import _inputs, _ops, _prior
    import vfe3.gradients.relational as relational_gradients
    from vfe3.gradients.relational import relational_predictive_readout
    data = _inputs(seed=3, batch=(2,))
    N = data["mu"].shape[-2]
    seen: List[bool] = []
    real = edge_links.build_edge_context

    def spy(*args: Any, causal: bool = False, **kwargs: Any) -> Any:
        seen.append(causal)
        return real(*args, causal=causal, **kwargs)

    monkeypatch.setattr(edge_links, "build_edge_context", spy)

    def readout(log_prior: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        ops = _ops("lowrank", True, seed=2)
        mean, cov = relational_predictive_readout(
            data["mu"], data["sigma_diag"], data["compact"], ops, log_prior=log_prior, return_covariance=True,
            floor_anchor=_prior(data["sigma_diag"], False))
        return mean.detach(), cov.detach()

    causal = attention_log_prior("causal", N, N, dtype=F64)
    mean, cov = readout(causal)
    readout(None)
    assert seen == [True, False]
    # The causal prior's readout equals the all-pairs evaluation of the same context.
    monkeypatch.setattr(relational_gradients, "strictly_causal_routing_prior", lambda prior: False)
    mean_all, cov_all = readout(causal)
    assert seen[-1] is False
    assert _relative(mean, mean_all) <= 1e-13
    assert _relative(cov, cov_all) <= 1e-13


# ---------------------------------------------------------------------------------------------------------------------
# A257: the matched baseline notes the noisy-key parameters
# ---------------------------------------------------------------------------------------------------------------------

def test_a257_matched_arm_notes_the_noisy_key_noise(tmp_path: Path) -> None:
    from tests.test_audit_20261009_round2_paper import _base, _plan
    plan = _plan(_base(tmp_path, relational_routing="noisy_key"), "Matched context-processing parameters",
                 counts=False)
    assert plan.status == "ready", plan.reason
    assert any("routing_key_log_noise" in note and "H x L = 4" in note for note in plan.notes), plan.notes
    plain = _plan(_base(tmp_path, "plain"), "Matched context-processing parameters", counts=False)
    assert not any("routing_key_log_noise" in note for note in plain.notes)
    from vfe3.baseline import BaselineConfig
    assert "routing_key_log_noise" in BaselineConfig.matched_to.__doc__


# ---------------------------------------------------------------------------------------------------------------------
# A180, A261, A272, A273: corrected texts
# ---------------------------------------------------------------------------------------------------------------------

def test_a272_a273_receiver_citations_and_floor_mode_texts() -> None:
    from vfe3.gradients import full_link
    for rel in ("vfe3/gradients/full_link.py", "vfe3/model/relational_operators.py", "vfe3/config.py"):
        assert "projected_gradient.tex" not in (ROOT / rel).read_text(encoding="utf-8"), rel
    assert "sec:receiver_inference" in full_link.__doc__ and "sec:rec_updates" in full_link.__doc__
    assert "prop:rec_flow" in full_link.full_link_minimizer.__doc__
    config_text = (ROOT / "vfe3/config.py").read_text(encoding="utf-8")
    assert '# "prior" | "relative" | "max"' in _comment_before(config_text, "predictive_covariance_floor_mode: str")
    assert ro.PREDICTIVE_COVARIANCE_FLOOR_MODES == ("prior", "relative", "max")
    doc = ro.RelationalOperators.predictive_covariance_update.__doc__
    assert "float32 values are discarded" not in doc and "only the failing receivers" in doc


def test_a180_a261_checkpoint_texts_state_what_the_recompute_runs() -> None:
    config_text = (ROOT / "vfe3/config.py").read_text(encoding="utf-8")
    comment = _comment_before(config_text, "predictive_covariance_checkpoint: bool")
    assert "moment_matched only" not in comment
    assert "minimal_agent" in comment and "passthrough included" in comment
    assert "three times" in comment and "decision host reads" in comment
    body = " ".join(ro.RelationalOperators._predictive_covariance_update.__doc__.split())
    assert "also when the checkpoint recomputes it" not in body
    assert "IEEE only where the tile function itself is pinned" in body
