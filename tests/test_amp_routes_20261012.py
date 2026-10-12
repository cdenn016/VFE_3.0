r"""Which operations run in reduced precision under ``amp_dtype`` (audit 2026-10-12 round 2, item 18).

A ``TorchDispatchMode`` probe records every operation whose floating output is bf16/fp16 over a training forward and
backward and a ``no_grad`` evaluation forward. On every route below no operation runs in reduced precision, and the
loss, every parameter gradient and the evaluation logits equal the ``amp_dtype=None`` run bit for bit, so
``inactive_fields()`` marks ``amp_dtype``. The two consumers, the linear decode's logits matmul under
``decode_logits_amp`` and the CG coupling's moment contraction, do run bf16 GEMMs and stay unmarked; fp16 is never
marked because it also enables the GradScaler. Tiny models (V=11, K=4, H=2, N=6); the CUDA test repeats the probe on
the device, whose autocast operation lists differ from the CPU's.
"""
import warnings
from typing import Any, Dict, Set, Tuple

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from tests.profiles import config_values
from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel

V, K, H, N, B = 11, 4, 2, 6, 2
_REDUCED = (torch.bfloat16, torch.float16)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

# The user's live route at tiny size: full family, full_chunked decode, kernel readout, noisy-key routing,
# moment-matched covariance, comparison-invariant RoPE, block_glk frames, coordinate block MLP with layernorm.
_KERNEL: Dict[str, Any] = dict(
    vocab_size=V, embed_dim=K, n_heads=H, max_seq_len=N, batch_size=B, eval_stride=None, attention_window=N,
    n_layers=2, attention_mode="projected_predictive", predictive_readout="kernel", family="gaussian_full",
    decode_mode="full_chunked", use_block_mlp=True, block_mlp_mode="coordinate", norm_type_block="layernorm",
    relational_routing="noisy_key", predictive_covariance="moment_matched", rope_mode="comparison_invariant",
    content_link_chart="exponential", gauge_group="block_glk", kl_max=8 * K, projected_factorization="gram",
    fp32_matmul_precision=None, decode_av_precision="fp32")
# The committed launcher's route: diagonal family, linear decode, minimal agent, prototype agents.
_MINIMAL: Dict[str, Any] = dict(
    _KERNEL, predictive_readout="minimal_agent", family="gaussian_diagonal", decode_mode="linear",
    block_mlp_mode="prototype_agents", relational_routing="gaussian_kl", rope_mode="none")

_INERT_ROUTES = {
    "live_kernel": _KERNEL,
    "committed_minimal_agent": _MINIMAL,
    "kernel_linear_decode": dict(_KERNEL, decode_mode="linear"),
    "kernel_delta_full": dict(_KERNEL, family="gaussian_diagonal", decode_mode="diagonal_chunked",
                              block_mlp_covariance="delta_full"),
    "kernel_mahalanobis_norm": dict(_KERNEL, norm_type_block="mahalanobis"),
    "kernel_expected_likelihood": dict(_KERNEL, family="gaussian_diagonal",
                                       decode_mode="expected_likelihood_chunked"),
    "kernel_reference_coordinates": dict(_KERNEL, block_coordinates="reference", decode_coordinates="reference"),
    "kernel_content_link_edges": dict(_KERNEL, content_link_edges="lowrank"),
    "projected_gradient": dict(_KERNEL, attention_mode="projected_gradient", relational_routing="gaussian_kl",
                               rope_mode="none", relational_receiver_update="euler"),
    "minimal_agent_full_decode": dict(_MINIMAL, family="gaussian_full", decode_mode="full_chunked"),
}
_CG = dict(_KERNEL, gauge_group="so_n", group_n=3, irrep_spec=[("l1", 1), ("l0", 1)], gauge_tying=None,
           use_cg_coupling=True)


class _ReducedPrecisionProbe(TorchDispatchMode):
    r"""Record the name of every operation that returns a bf16/fp16 floating tensor."""

    def __init__(self) -> None:
        super().__init__()
        self.ops: Set[str] = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for value in (out if isinstance(out, (tuple, list)) else (out,)):
            if isinstance(value, torch.Tensor) and value.dtype in _REDUCED:
                self.ops.add(func.__name__.split(".")[0])
        return out


def _config(values: Dict[str, Any], amp_dtype: Any) -> VFE3Config:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VFE3Config(**config_values(**{**values, "amp_dtype": amp_dtype}))


def _run(values: Dict[str, Any], amp_dtype: Any, device: torch.device
         ) -> Tuple[torch.Tensor, Dict[str, Any], torch.Tensor, Set[str]]:
    r"""Loss, parameter gradients and eval logits of one seeded model, with the reduced-precision op names."""
    torch.manual_seed(0)
    model = VFEModel(_config(values, amp_dtype)).to(device)
    g = torch.Generator().manual_seed(7)
    ids = torch.randint(0, V, (B, N), generator=g).to(device)
    targets = torch.randint(0, V, (B, N), generator=g).to(device)
    probe = _ReducedPrecisionProbe()
    with probe:
        _, loss, _ = model(ids, targets)
        loss.backward()
        model.eval()
        with torch.no_grad():
            logits = model(ids)
    grads = {name: p.grad for name, p in model.named_parameters()}
    return loss.detach(), grads, logits, probe.ops


def _assert_bitwise_equal(a, b) -> None:
    loss_a, grads_a, logits_a, _ = a
    loss_b, grads_b, logits_b, _ = b
    assert torch.equal(loss_a, loss_b)
    assert torch.equal(logits_a, logits_b)
    for name, grad in grads_a.items():
        other = grads_b[name]
        assert (grad is None) == (other is None), name
        if grad is not None:
            assert torch.equal(grad, other), name


def _check_inert_route(values: Dict[str, Any], device: torch.device) -> None:
    assert "amp_dtype" in _config(values, "bf16").inactive_fields()
    reference = _run(values, None, device)
    amp = _run(values, "bf16", device)
    assert reference[3] == set() and amp[3] == set(), amp[3]
    _assert_bitwise_equal(reference, amp)


@pytest.mark.parametrize("route", sorted(_INERT_ROUTES))
def test_bf16_runs_no_reduced_precision_op_and_is_marked_inert(route: str, device: torch.device) -> None:
    _check_inert_route(_INERT_ROUTES[route], device)


@pytest.mark.parametrize("values", [dict(_KERNEL, decode_mode="linear", decode_logits_amp=True),
                                    dict(_MINIMAL, decode_logits_amp=True)], ids=["kernel", "minimal_agent"])
def test_decode_logits_amp_is_a_bf16_consumer_and_stays_live(values: Dict[str, Any], device: torch.device) -> None:
    assert "amp_dtype" not in _config(values, "bf16").inactive_fields()
    loss_ref = _run(values, None, device)[0]
    loss, _, _, ops = _run(values, "bf16", device)
    assert "mm" in ops                                   # the logits matmul x W^T, bf16 operands
    assert loss.dtype == torch.float32 and not torch.equal(loss, loss_ref)


def test_cg_coupling_is_a_bf16_consumer_and_stays_live(device: torch.device) -> None:
    assert "amp_dtype" not in _config(_CG, "bf16").inactive_fields()
    assert "bmm" in _run(_CG, "bf16", device)[3]


@pytest.mark.parametrize("values", [_KERNEL, _MINIMAL], ids=["kernel", "minimal_agent"])
def test_fp16_and_none_are_never_marked(values: Dict[str, Any]) -> None:
    for amp_dtype in (None, "fp16"):
        assert "amp_dtype" not in _config(values, amp_dtype).inactive_fields()


@requires_cuda
def test_cuda_autocast_probe_matches_the_rule() -> None:
    r"""CUDA autocast lowers a different operation list from CPU autocast; repeat the probe on the device."""
    device = torch.device("cuda")
    for route in ("live_kernel", "committed_minimal_agent", "projected_gradient"):
        _check_inert_route(_INERT_ROUTES[route], device)
    values = dict(_MINIMAL, decode_logits_amp=True)
    assert "mm" in _run(values, "bf16", device)[3]
    assert "bmm" in _run(_CG, "bf16", device)[3]
