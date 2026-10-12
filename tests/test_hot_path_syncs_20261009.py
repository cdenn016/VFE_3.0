"""Hot-path scheduling changes of October 9, 2026 (tiny CPU models).

1. Under ``routing_kl_max_reference='row_min'`` the query log-determinant of the Gaussian KL is a row
   constant the router cannot read, so it is computed without autograd; values and routing are unchanged.
2. Raise-only frame certificates are deferred to the next mandatory host read inside
   :func:`vfe3.contracts.deferred_host_checks` and still fail closed with their own messages.
3. The per-role clip norms ride the fused finiteness transfer of ``train_step``.
"""
from typing import Any, Callable, Dict, Iterator, Tuple

import pytest
import torch

import vfe3.relational_scoring as scoring
from tests.test_audit_20261007_opus_fixes import B, K, N, V, _cfg, _minimal_agent
from tests.test_projected_condition_recovery_20260923 import _ill_conditioned
from vfe3.model.model import VFEModel


@pytest.fixture(autouse=True)
def _restore_scorer_policies() -> Iterator[None]:
    saved = (scoring._CONDITION_RECOVERY, scoring._FACTOR_ROUTE)
    yield
    scoring._CONDITION_RECOVERY, scoring._FACTOR_ROUTE = saved


def _kernel_values(reference: str, **overrides: Any) -> Dict[str, Any]:
    return _minimal_agent(predictive_readout="kernel", block_mlp_mode="coordinate", decode_mode="diagonal_chunked",
                          routing_kl_max_reference=reference, **overrides)


def _run(values: Dict[str, Any], cost: Callable) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    r"""fp64 loss, per-layer routing weights and parameter gradients with ``cost`` as the gaussian_kl router."""
    original = scoring.ROUTING_COSTS["gaussian_kl"]
    scoring.ROUTING_COSTS["gaussian_kl"] = cost
    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(0)
        model = VFEModel(_cfg(values)).double()
        g = torch.Generator().manual_seed(7)
        ids = torch.randint(0, V, (B, N), generator=g)
        tgt = torch.randint(0, V, (B, N), generator=g)
        _, loss, _ = model(ids, tgt)
        loss.backward()
        with torch.no_grad():
            beta = model.attention_maps(ids)
    finally:
        scoring.ROUTING_COSTS["gaussian_kl"] = original
        torch.set_default_dtype(default)
    assert loss.dtype == torch.float64
    return loss.detach(), beta, {n: p.grad for n, p in model.named_parameters() if p.grad is not None}


def _spy(seen: list) -> Callable:
    original = scoring.projected_gaussian_kl

    def cost(*args: Any, **kwargs: Any) -> torch.Tensor:
        seen.append(kwargs.get("query_row_constant_grad", True))
        return original(*args, **kwargs)
    return cost


def _forced_query_grad(*args: Any, **kwargs: Any) -> torch.Tensor:
    kwargs.pop("query_row_constant_grad", None)
    return scoring.projected_gaussian_kl(*args, **kwargs)


@pytest.mark.parametrize("kl_max", [1e9, 8.0 * K])
def test_row_min_drops_query_logdet_grad_without_changing_routing(kl_max: float) -> None:
    """Item 1: loss and routing are bitwise unchanged; parameter gradients agree to fp64 roundoff."""
    values = _kernel_values("row_min", kl_max=kl_max)
    seen: list = []
    loss_new, beta_new, grads_new = _run(values, _spy(seen))
    assert seen and not any(seen)                       # every router call declared the row constant
    loss_old, beta_old, grads_old = _run(values, _forced_query_grad)
    assert torch.equal(loss_new, loss_old)
    assert torch.equal(beta_new, beta_old)
    assert grads_new.keys() == grads_old.keys() and grads_new
    for name in grads_old:
        assert torch.allclose(grads_new[name], grads_old[name], rtol=1e-10, atol=1e-13), name


def test_absolute_reference_keeps_the_query_logdet_grad() -> None:
    """Item 1: under 'absolute' the clamp reads the row constant, so the default call is made."""
    values = _kernel_values("absolute")
    seen: list = []
    loss_new, _, grads_new = _run(values, _spy(seen))
    assert seen and all(seen)
    loss_old, _, grads_old = _run(values, _forced_query_grad)
    assert torch.equal(loss_new, loss_old)
    for name in grads_old:
        assert torch.equal(grads_new[name], grads_old[name]), name


def _pair(dtype: torch.dtype = torch.float64) -> Tuple[torch.Tensor, ...]:
    g = torch.Generator().manual_seed(11)
    a_q, a_k = (torch.randn(2, 5, 3, 2, generator=g, dtype=dtype) for _ in range(2))
    X = torch.randn(2, 5, 3, 2, 4, generator=g, dtype=dtype)
    Y = torch.randn(2, 5, 3, 2, 4, generator=g, dtype=dtype)
    G_q, G_k = X @ X.transpose(-1, -2), Y @ Y.transpose(-1, -2)
    return tuple(x.requires_grad_(True) for x in (a_q, G_q, a_k, G_k))


def test_query_logdet_carries_no_grad_but_the_value_is_bitwise() -> None:
    """Item 1: the cost values are bitwise; d cost.sum()/dG_q loses exactly -N_k G_q^{-1}/2."""
    a_q, G_q, a_k, G_k = _pair()
    full = scoring.projected_gaussian_kl(a_q, G_q, a_k, G_k)
    rowfree = scoring.projected_gaussian_kl(a_q, G_q, a_k, G_k, query_row_constant_grad=False)
    assert torch.equal(full, rowfree)
    (g_full,) = torch.autograd.grad(full.sum(), G_q)
    (g_free,) = torch.autograd.grad(rowfree.sum(), G_q)
    n_k = a_k.shape[-3]
    expected = g_full + 0.5 * n_k * torch.linalg.inv(G_q.detach())
    assert torch.allclose(g_free, expected, atol=1e-10)


@pytest.mark.parametrize("recovered_role", ["query", "key"])
def test_recovery_path_refactors_the_query_with_autograd(recovered_role: str) -> None:
    """Item 1: a recovered pair reads the query factor in its trace, whose gradient must survive."""
    scoring.set_projected_condition_recovery(True)
    *_, sources = _ill_conditioned(torch.device("cpu"))
    if recovered_role == "key":
        mu_q, sigma_q, mu_k, sigma_k, inverse, S = sources
        sources = (mu_k, sigma_k, mu_q, sigma_q, inverse, S)

    def row_free_loss(flag: bool) -> Tuple[torch.Tensor, ...]:
        mu_q, sigma_q, mu_k, sigma_k, inverse, S = (x.clone().requires_grad_(x.is_floating_point())
                                                    for x in sources)
        aq, Gq = scoring.projected_moments(mu_q, sigma_q, inverse, S)
        ak, Gk = scoring.projected_moments(mu_k, sigma_k, inverse, S)
        scoring.reset_source_recovery_elements()
        cost = scoring.projected_gaussian_kl(
            aq, Gq, ak, Gk, projection_q=(sigma_q, inverse, S, mu_q),
            projection_k=(sigma_k, inverse, S, mu_k), query_row_constant_grad=flag)
        assert scoring.source_recovery_elements() > 0
        # A row-invariant loss that weights the recovered pairs (whose softmax weight underflows).
        weight = torch.linspace(-1.0, 2.0, cost.numel(), dtype=cost.dtype).reshape(cost.shape)
        loss = ((cost - cost.mean(dim=-1, keepdim=True)) * weight).sum()
        return (cost.detach(), *torch.autograd.grad(loss, (mu_q, sigma_q, mu_k, sigma_k)))

    with_grad, without = row_free_loss(True), row_free_loss(False)
    assert torch.equal(with_grad[0], without[0])
    for a, b in zip(with_grad[1:], without[1:]):
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-6)


def _step_grads(grad_clip: float, reference: bool, monkeypatch: pytest.MonkeyPatch,
                metrics: Dict[str, Any]) -> Dict[int, torch.Tensor]:
    r"""Gradients that reach optimizer.step, from train_step or from the former per-role clip_grad_norm_."""
    from tests.test_train import _clip_case
    from vfe3.train import train_step
    torch.manual_seed(0)
    model, optimizer, scheduler, tokens, targets = _clip_case()
    seen: Dict[int, torch.Tensor] = {}
    optimizer.register_step_pre_hook(lambda *_: seen.update(
        {i: p.grad.clone() for i, p in enumerate(model.parameters()) if p.grad is not None}))
    if reference:
        # The former clip: one clip_grad_norm_ per nonempty role after the host barrier.
        def clip_by_norm(parameters: list, max_norm: float, total_norm: torch.Tensor, **_: Any) -> None:
            torch.nn.utils.clip_grad_norm_(parameters, max_norm)
        monkeypatch.setattr(torch.nn.utils, "clip_grads_with_norm_", clip_by_norm)
    train_step(model, optimizer, scheduler, tokens, targets, grad_clip=grad_clip, metrics_out=metrics)
    monkeypatch.undo()
    return seen


@pytest.mark.parametrize("grad_clip", [1e-3, 1e6])
def test_fused_clip_norms_scale_exactly_as_clip_grad_norm(monkeypatch: pytest.MonkeyPatch,
                                                          grad_clip: float) -> None:
    """Item 3: norms taken before the barrier clip bitwise as the former post-barrier clip_grad_norm_."""
    new_metrics: Dict[str, Any] = {}
    ref_metrics: Dict[str, Any] = {}
    new = _step_grads(grad_clip, False, monkeypatch, new_metrics)
    ref = _step_grads(grad_clip, True, monkeypatch, ref_metrics)
    assert new and new.keys() == ref.keys()
    for i in ref:
        assert torch.equal(new[i], ref[i]), i
    assert new_metrics["grad_clip_finite"] == ref_metrics["grad_clip_finite"] == 1.0
    assert new_metrics["grad_norm"] == ref_metrics["grad_norm"]


def test_nonfinite_role_norm_skips_the_step(monkeypatch: pytest.MonkeyPatch) -> None:
    """Item 3: an overflowing role norm with finite gradients still skips the update and logs it."""
    from tests.test_train import _clip_case
    from vfe3.train import train_step
    model, optimizer, scheduler, tokens, targets = _clip_case()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    real = torch.nn.utils.get_total_norm
    calls: list = []

    def overflow(tensors: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        calls.append(1)
        return real(tensors, *args, **kwargs) * (float("inf") if len(calls) == 1 else 1.0)

    monkeypatch.setattr(torch.nn.utils, "get_total_norm", overflow)
    metrics: Dict[str, Any] = {}
    status: Dict[str, Any] = {}
    train_step(model, optimizer, scheduler, tokens, targets, grad_clip=1.0,
               metrics_out=metrics, status_out=status)
    assert calls and metrics["grad_finite"] == 1.0 and metrics["grad_clip_finite"] == 0.0
    assert status["did_step"] is False
    for n, p in model.named_parameters():
        assert torch.equal(p.detach(), before[n]), n


def _bad_frame() -> Any:
    from vfe3.contracts import CanonicalFrameContext
    eye = torch.eye(2).expand(1, 3, 2, 2, 2).clone()
    return CanonicalFrameContext(forward=eye, inverse=2.0 * eye, block_diagonal=True)


def test_frame_contract_raises_at_once_outside_the_scope() -> None:
    with pytest.raises(ValueError, match="mutual inverses"):
        _bad_frame()


def test_deferred_frame_contract_raises_at_scope_exit() -> None:
    """Item 2: inside the scope the raise-only verdict waits for a host read, then raises unchanged."""
    from vfe3.contracts import deferred_host_checks
    reached = False
    with pytest.raises(ValueError, match="mutual inverses"):
        with deferred_host_checks():
            _bad_frame()
            reached = True
    assert reached
    _ok = torch.eye(2).expand(1, 3, 2, 2, 2).clone()
    with deferred_host_checks():                         # the failed queue does not leak into a new scope
        from vfe3.contracts import CanonicalFrameContext
        CanonicalFrameContext(forward=_ok, inverse=_ok.clone(), block_diagonal=True)


def test_deferred_check_raises_before_the_scorer_reads_its_own_verdict() -> None:
    """Item 2: the scorer's host read resolves queued checks first, in registration order."""
    from vfe3.contracts import deferred_host_checks
    from vfe3.geometry.transport import _checked_group_inverse_certified
    a_q, G_q, a_k, G_k = (x.detach() for x in _pair())
    G_q = G_q.clone()
    G_q[0, 0, 0] = float("nan")                          # the scorer alone would raise its own error
    with pytest.raises(ValueError, match="singular"):
        with deferred_host_checks():
            inverse, escalate = _checked_group_inverse_certified(torch.zeros(2, 3, 3))
            assert escalate is False
            _bad_frame()
            scoring.projected_gaussian_kl(a_q, G_q, a_k, G_k)
    with pytest.raises(ValueError, match="projected covariance must be finite"):
        with deferred_host_checks():
            scoring.projected_gaussian_kl(a_q, G_q, a_k, G_k)


def test_train_step_resolves_deferred_checks_before_backward(monkeypatch: pytest.MonkeyPatch) -> None:
    """Item 2: a deferred transport verdict that fails stops train_step before any update."""
    import vfe3.geometry.transport as transport
    from tests.test_train import _clip_case
    from vfe3.train import train_step
    model, optimizer, scheduler, tokens, targets = _clip_case()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    calls: list = []

    def failing(host: list, *args: Any) -> None:
        calls.append(host)
        raise FloatingPointError("deferred verdict reached")

    monkeypatch.setattr(transport, "_group_inverse_verdict", failing)
    backward = torch.Tensor.backward
    monkeypatch.setattr(torch.Tensor, "backward", lambda *a, **k: pytest.fail("backward ran"))
    with pytest.raises(FloatingPointError, match="deferred verdict reached"):
        train_step(model, optimizer, scheduler, tokens, targets, grad_clip=1.0)
    monkeypatch.setattr(torch.Tensor, "backward", backward)
    assert calls
    for n, p in model.named_parameters():
        assert torch.equal(p.detach(), before[n]), n
