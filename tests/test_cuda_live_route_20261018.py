r"""CUDA lane: the tiny live launcher route on the device against a float64 CPU copy (audit 2026-10-17 item 26).

The CPU lanes run ``live_config`` only on the CPU, where the fixed-schedule frame exponential, the fused Triton
KL-decode CE and the device precision policy never execute. This test builds the tiny live profile on CUDA in
float32 with ``decode_ce_triton=True`` on a chunked KL decode (the launcher's ``decode_mode`` is overridden when it
is ``"linear"``, which has no fused kernel), copies its weights into a float64 CPU model and compares the loss and
every parameter gradient at 1e-4 relative. A parameter whose float32 gradient is cancellation-limited even on the
CPU torch route (a CPU dry run measured 1.4e-4 on a ``channel_log_noise`` gradient of norm 4e-6) is held to four
times that CPU float32 error instead. Registered in ``tests/pytest_policy.py::CUDA_TESTS``; it skips without CUDA.
"""

import os
from typing import Any, Dict

import pytest
import torch

from tests.profiles import live_config
from vfe3.model.model import VFEModel

requires_cuda = pytest.mark.skipif(
    os.environ.get("VFE3_TEST_DEVICE") == "cpu" or not torch.cuda.is_available(),
    reason="requires the CUDA lane (VFE3_TEST_DEVICE=cuda) and a CUDA device")

_REL_TOL = 1e-4


def _grads(model: VFEModel) -> Dict[str, torch.Tensor]:
    return {name: p.grad.detach().double().cpu() for name, p in model.named_parameters()
            if p.requires_grad and p.grad is not None}


@requires_cuda
def test_cuda_live_route_matches_a_float64_cpu_copy(record_property: Any) -> None:
    pytest.importorskip("triton")
    overrides: Dict[str, Any] = {"decode_ce_triton": True}
    if live_config().decode_mode == "linear":
        # The fused chunk kernels need a KL decoder: the one whose covariance rank matches the launcher's family
        # (the coupling the validator enforces), so the case is defined whatever family the launcher holds.
        overrides["decode_mode"] = ("full_chunked" if live_config().family in ("gaussian_full", "gaussian_block")
                                    else "diagonal_chunked")
    cfg = live_config(**overrides)
    device = torch.device("cuda")
    torch.manual_seed(0)
    cuda_model = VFEModel(cfg).to(device)
    bank = cuda_model.prior_bank
    if bank.decode_unigram_prior:
        counts = torch.randint(0, 50, (bank.vocab_size,), generator=torch.Generator().manual_seed(6))
        bank.set_unigram_log_prior(counts.to(device))
    state = {key: value.detach().cpu() for key, value in cuda_model.state_dict().items()}
    # The nonzero unigram buffer travels with the state dict; the CPU banks read it as set.
    cpu_model = VFEModel(cfg).double()
    cpu_model.load_state_dict(state)
    cpu32_model = VFEModel(cfg)
    cpu32_model.load_state_dict(state)

    g = torch.Generator().manual_seed(1)
    shape = (cfg.batch_size, cfg.max_seq_len)
    tokens = torch.randint(0, cfg.vocab_size, shape, generator=g)
    targets = torch.randint(0, cfg.vocab_size, shape, generator=g)

    _, loss_cuda, _ = cuda_model(tokens.to(device), targets.to(device))
    loss_cuda.backward()
    assert cuda_model.prior_bank.realized_decode_ce_kernel == "triton"
    _, loss_cpu, _ = cpu_model(tokens, targets)
    loss_cpu.backward()
    _, loss_cpu32, _ = cpu32_model(tokens, targets)
    loss_cpu32.backward()

    loss_rel = abs(float(loss_cuda) - float(loss_cpu)) / abs(float(loss_cpu))
    g32, g64, g32_cpu = _grads(cuda_model), _grads(cpu_model), _grads(cpu32_model)
    assert set(g32) == set(g64) == set(g32_cpu) and g64
    # the router is live at the tiny cap, so the comparison maps are compared, not filtered out as zero-norm
    # (audit 2026-10-20 item 5)
    router = [name for name in g64 if "raw_score" in name]
    assert router and all(float(g64[name].norm()) > 0.0 for name in router)

    def rel(grads: Dict[str, torch.Tensor], name: str) -> float:
        return float((grads[name] - g64[name]).norm() / g64[name].norm().clamp_min(1e-300))

    grad_rel = {name: rel(g32, name) for name in g64}
    floor = {name: max(_REL_TOL, 4.0 * rel(g32_cpu, name)) for name in g64}
    record_property("cuda_live_loss_rel_vs_fp64", loss_rel)
    record_property("cuda_live_grad_rel_vs_fp64", str(grad_rel))
    assert loss_rel <= _REL_TOL, loss_rel
    # A parameter whose gradient is tiny against the whole model's (channel_log_noise, norm about 4e-6) has an
    # ill-conditioned relative error, so it also passes when its absolute error is below 1e-5 of the total norm.
    total = float(torch.sqrt(sum(g.double().norm() ** 2 for g in g64.values())))
    abs_err = {name: float((g32[name].double() - g64[name].double()).norm()) for name in g64}
    assert {name: (value, floor[name], abs_err[name]) for name, value in grad_rel.items()
            if value > floor[name] and float(g64[name].norm()) > 0.0 and abs_err[name] > 1e-5 * total} == {}
