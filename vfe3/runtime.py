"""Shared runtime-state setup and reporting for click-to-run entry points."""

import os
from typing import Dict, Optional

import torch


_INITIAL_CUBLAS_WORKSPACE_CONFIG = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
_FP32_MATMUL_PRECISIONS = ("highest", "high", "medium")


def set_fp32_matmul_precision(policy: Optional[str]) -> None:
    r"""Pin the fp32 matmul / TF32 policy, or leave the card as it is (audit 2026-08-06 C7/F25).

    ``None`` is the historical behavior: production set nothing, so the effective policy was
    whatever the card defaulted to -- while ``tests/conftest.py`` pinned one, meaning every tolerance
    in the suite was asserted under a policy the training run never stated. Passing a value pins it
    AND makes it recordable, since :func:`deterministic_state` reads it back from the backend rather
    than echoing the request.

    ``torch.use_deterministic_algorithms(True)`` does NOT constrain TF32, so this is a genuinely
    independent axis and not implied by ``deterministic``. Setting it CAN change training numerics
    on a TF32-capable card, which is why the default leaves it alone.
    """
    if policy is None:
        return
    if policy not in _FP32_MATMUL_PRECISIONS:
        raise ValueError(
            f"fp32_matmul_precision must be None or one of {_FP32_MATMUL_PRECISIONS}, got {policy!r}")
    # ONLY the new API. Do NOT also set torch.backends.cuda.matmul.allow_tf32 / cudnn.allow_tf32 to
    # "keep them consistent": from torch 2.9 the two are different generations of the same knob, and
    # touching both puts the runtime in a mixed-API state where torch.get_float32_matmul_precision()
    # RAISES ("you have used mix of the legacy and new APIs"). That would break the unconditional
    # recording in deterministic_state below -- i.e. the consistency gesture destroys the very
    # provenance this finding is about. Measured on torch 2.9 / py3.14. The legacy booleans remain
    # readable and stay in agreement on their own, because they are views of this same setting.
    torch.set_float32_matmul_precision(policy)


def seed_everything(
    seed: int,

    *,
    deterministic: bool,
    fp32_matmul_precision: Optional[str] = None,
) -> None:
    """Seed PyTorch and configure the requested deterministic execution state."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_fp32_matmul_precision(fp32_matmul_precision)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    elif _INITIAL_CUBLAS_WORKSPACE_CONFIG is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = _INITIAL_CUBLAS_WORKSPACE_CONFIG


def _fp32_matmul_precision() -> Dict[str, object]:
    r"""Effective fp32 matmul / TF32 policy, best-effort across torch versions.

    ``torch.use_deterministic_algorithms(True)`` does NOT constrain TF32 (audit 2026-08-06 F25), so
    this is a real degree of freedom that production neither pinned nor recorded while
    ``tests/conftest.py:61-99`` pins and restores it. That left every numeric tolerance in the suite
    asserted under a policy the training run never stated, and ``provenance.json`` unable to show
    the drift.

    RECORDING is unconditional and cannot change numerics -- it reads the backend BACK rather than
    echoing whatever was requested, so a request that a torch version silently ignored still shows up
    here as the effective policy. SETTING is opt-in through ``cfg.fp32_matmul_precision`` /
    :func:`set_fp32_matmul_precision`, defaulting to None ("leave the card as it is"), because on a
    TF32-capable card pinning it would change results mid-project.
    """
    state: Dict[str, object] = {}
    try:                                   # torch >= 2.9 typed accessor
        state["matmul_fp32_precision"] = str(torch.backends.cuda.matmul.fp32_precision)
    except AttributeError:
        state["matmul_fp32_precision"] = None
    for name, get in (
        ("matmul_allow_tf32", lambda: torch.backends.cuda.matmul.allow_tf32),
        ("cudnn_allow_tf32",  lambda: torch.backends.cudnn.allow_tf32),
        ("float32_matmul_precision", torch.get_float32_matmul_precision),
    ):
        try:
            state[name] = get()
        except Exception:                  # noqa: BLE001 - a missing backend knob is not an error
            state[name] = None
    return state


def deterministic_state() -> Dict[str, object]:
    """Return the effective PyTorch, cuDNN, cuBLAS, and fp32-matmul determinism state."""
    return {
        "algorithms":              torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic":     bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark":         bool(torch.backends.cudnn.benchmark),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        **_fp32_matmul_precision(),
    }
