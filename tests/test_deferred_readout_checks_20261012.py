"""Host reads of a training forward (audit 2026-10-12, item 22; tiny CPU models).

The prototype-agent factorization checks are raise-only, so inside
:func:`vfe3.contracts.deferred_host_checks` they join the next mandatory host read; the minimal-agent
readout's per-layer status read is that read (:func:`vfe3.contracts.host_read`), as the projected
scorer's already was. A healthy step is bitwise unchanged, a failed prototype check still stops the
step before backward with its own message, and a flagged minimal-agent layer still takes its float64
recomputation.
"""
from typing import Any, Dict, Tuple

import pytest
import torch

import vfe3.model.prototype_agents as prototype_agents
from tests.test_audit_20261007_opus_fixes import B, N, V, _cfg, _minimal_agent
from vfe3.contracts import deferred_host_checks
from vfe3.model.model import VFEModel
from vfe3.model.relational_operators import RelationalOperators
from vfe3.relational_scoring import reset_source_recovery_elements, source_recovery_elements
from vfe3.train import _transfer_targets, build_optimizer, lr_lambda, train_step

L = 3
READOUTS = ("minimal_agent", "kernel")


def _values(readout: str) -> Dict[str, Any]:
    return _minimal_agent(predictive_readout=readout, n_layers=L, block_mlp_inference="mean_field")


def _batch() -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(7)
    return torch.randint(0, V, (B, N), generator=g), torch.randint(0, V, (B, N), generator=g)


def _loss_and_grads(readout: str, deferred: bool) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    torch.manual_seed(0)
    model = VFEModel(_cfg(_values(readout)))
    ids, tgt = _batch()
    if deferred:
        with deferred_host_checks():
            _, loss, _ = model(ids, tgt)
    else:
        _, loss, _ = model(ids, tgt)
    loss.backward()
    return loss.detach(), {n: p.grad for n, p in model.named_parameters() if p.grad is not None}


@pytest.mark.parametrize("readout", READOUTS)
def test_deferred_checks_leave_a_healthy_step_bitwise_unchanged(readout: str) -> None:
    loss_now, grads_now = _loss_and_grads(readout, deferred=False)
    loss_def, grads_def = _loss_and_grads(readout, deferred=True)
    assert torch.isfinite(loss_now)
    assert torch.equal(loss_now, loss_def)
    assert grads_now.keys() == grads_def.keys() and grads_now
    for name, grad in grads_now.items():
        assert torch.equal(grad, grads_def[name]), name


def _step_case(readout: str) -> Tuple[VFEModel, Any, Any, torch.Tensor, torch.Tensor]:
    cfg = _cfg(_values(readout))
    torch.manual_seed(0)
    model = VFEModel(cfg)
    model.prior_bank.set_unigram_log_prior(torch.ones(V))   # a set table: no unset-table probe
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_lambda(step, cfg))
    ids, tgt = _batch()
    # The training loop validates targets on the host before transfer; mirror it.
    return model, optimizer, scheduler, ids, _transfer_targets(tgt, model, torch.device("cpu"))


@pytest.mark.parametrize("readout", READOUTS)
def test_training_forward_makes_one_read_per_layer_and_one_at_scope_exit(
        readout: str, monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Every ``tolist`` / ``item`` / ``__bool__`` of ``train_step`` before its ``backward``.

    Formerly 2L + 1 (minimal agent) and 2L (kernel): each prototype-agent check synchronized on its own.
    """
    model, optimizer, scheduler, ids, tgt = _step_case(readout)
    seen: list = []
    state = {"forward": True}
    for name in ("tolist", "item", "__bool__"):
        original = getattr(torch.Tensor, name)

        def spy(self: torch.Tensor, *args: Any, _original: Any = original, _name: str = name, **kwargs: Any) -> Any:
            if state["forward"]:
                seen.append(_name)
            return _original(self, *args, **kwargs)
        monkeypatch.setattr(torch.Tensor, name, spy)
    backward = torch.Tensor.backward

    def mark(self: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        state["forward"] = False
        return backward(self, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "backward", mark)
    train_step(model, optimizer, scheduler, ids, tgt, grad_clip=1.0)
    assert seen == ["tolist"] * (L + 1)


@pytest.mark.parametrize("readout", READOUTS)
def test_failed_prototype_check_stops_the_step_before_backward(readout: str,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    model, optimizer, scheduler, ids, tgt = _step_case(readout)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    logits = prototype_agents.template_kl_logits
    monkeypatch.setattr(prototype_agents, "template_kl_logits",
                        lambda *args: torch.full_like(logits(*args), float("nan")))
    monkeypatch.setattr(torch.Tensor, "backward", lambda *a, **k: pytest.fail("backward ran"))
    with pytest.raises(ValueError, match="power-posterior matrix"):
        train_step(model, optimizer, scheduler, ids, tgt, grad_clip=1.0)
    for n, p in model.named_parameters():
        assert torch.equal(p.detach(), before[n]), n


def test_failed_prototype_check_raises_at_once_outside_the_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(0)
    model = VFEModel(_cfg(_values("minimal_agent")))
    ids, tgt = _batch()
    logits = prototype_agents.template_kl_logits
    monkeypatch.setattr(prototype_agents, "template_kl_logits",
                        lambda *args: torch.full_like(logits(*args), float("nan")))
    with pytest.raises(ValueError, match="power-posterior matrix"):
        model(ids, tgt)


def test_flagged_minimal_agent_layer_takes_the_float64_recomputation(monkeypatch: pytest.MonkeyPatch) -> None:
    """A float32 status flag inside the scope recomputes that layer in float64, as outside it."""
    model, optimizer, scheduler, ids, tgt = _step_case("minimal_agent")
    torch.manual_seed(0)
    reference = VFEModel(_cfg(_values("minimal_agent")))
    reference.prior_bank.set_unigram_log_prior(torch.ones(V))
    with torch.no_grad():
        _, healthy, _ = reference(ids, tgt.clone())
    call = RelationalOperators._minimal_agent_call
    dtypes: list = []

    def flag_first(self: RelationalOperators, *args: Any) -> Any:
        mu_plus, sigma_plus, record, status = call(self, *args)
        dtypes.append(args[0].dtype)
        if len(dtypes) == 1:
            status = status | torch.tensor([False, True])          # the first layer: ill-conditioned
        return mu_plus, sigma_plus, record, status

    monkeypatch.setattr(RelationalOperators, "_minimal_agent_call", flag_first)
    reset_source_recovery_elements()
    status: Dict[str, Any] = {}
    loss_value = train_step(model, optimizer, scheduler, ids, tgt, grad_clip=1.0, status_out=status)
    assert dtypes == [torch.float32, torch.float64] + [torch.float32] * (L - 1)
    assert source_recovery_elements() == 2 * B * N * 2          # 2 B N H entries of the recomputed layer
    assert status["did_step"] is True and status["nonfinite_skip"] is False
    assert torch.isfinite(torch.tensor(loss_value))
    assert abs(loss_value - float(healthy)) < 1e-4 * max(1.0, abs(float(healthy)))
