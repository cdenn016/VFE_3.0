r"""Single-threaded batched LU-family linear algebra on CPU tensors (audit 2026-10-20 item 28).

With more than one thread, batched ``torch.linalg.inv_ex`` of :math:`n \times n` matrices with :math:`n \gtrsim 160` on
CPU (torch 2.10 and 2.11 with oneMKL 2025.3) returns an inverse with :math:`\lVert AA^{-1} - I\rVert` of order
:math:`10` to :math:`10^3` and no error, or raises a pivot error from ``lu_solve``; batched ``slogdet`` is off by tens
of nats. One matrix at a time, or the batched call under one thread, is exact; ``eigh`` and ``cholesky`` are not
affected. A multi-threaded CPU replay of a K >= 160 run reported CE 22.6 for 3.0 through the content-chart inverse.

:func:`single_thread_for_batched_cpu` runs one call site under one thread when an operand is a batched CPU tensor;
:func:`batched_cpu_lu_guard` installs a ``TorchFunctionMode`` that does the same for every LU-family call
(``inv``, ``inv_ex``, ``solve``, ``solve_ex``, ``det``, ``slogdet``, ``logdet``, ``lu``, ``lu_factor``,
``lu_factor_ex``, ``lu_solve`` and the tensor methods) inside the CPU post-hoc and semantics drivers. CUDA tensors and
unbatched calls are untouched.
"""
import functools
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Sequence, TypeVar

import torch
from torch.overrides import TorchFunctionMode

F = TypeVar("F", bound=Callable[..., Any])


def _is_batched_cpu(value: Any) -> bool:
    r"""Whether ``value`` is a CPU tensor with more than one matrix in its leading axes."""
    if not isinstance(value, torch.Tensor) or value.is_cuda or value.ndim <= 2:
        return False
    batch = 1
    for size in value.shape[:-2]:
        batch *= int(size)
    return batch > 1


@contextmanager
def single_thread_for_batched_cpu(*tensors: Any) -> Iterator[None]:
    r"""Run the enclosed code under one torch thread when any operand is a batched CPU tensor; restore after."""
    previous = torch.get_num_threads()
    if previous <= 1 or not any(_is_batched_cpu(tensor) for tensor in tensors):
        yield
        return
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _lu_functions() -> frozenset:
    names = ("inv", "inv_ex", "solve", "solve_ex", "det", "slogdet", "lu", "lu_factor", "lu_factor_ex", "lu_solve")
    functions = {getattr(torch.linalg, name) for name in names if hasattr(torch.linalg, name)}
    functions |= {getattr(torch, name) for name in ("inverse", "det", "logdet", "slogdet", "lu_solve")
                  if hasattr(torch, name)}
    functions |= {getattr(torch.Tensor, name) for name in ("inverse", "det", "logdet", "slogdet", "lu_solve")
                  if hasattr(torch.Tensor, name)}
    return frozenset(functions)


_LU_FUNCTIONS = _lu_functions()


def _tensor_operands(args: Sequence[Any], kwargs: Mapping[str, Any]) -> Iterator[Any]:
    for value in (*args, *kwargs.values()):
        if isinstance(value, (tuple, list)):
            yield from value
        else:
            yield value


class _BatchedCPULUGuard(TorchFunctionMode):
    r"""Runs each LU-family call on a batched CPU operand under one thread."""

    def __torch_function__(
        self,
        func:   Callable[..., Any],
        types:  Any,
        args:   Sequence[Any] = (),
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> Any:
        kwargs = kwargs or {}
        if func in _LU_FUNCTIONS:
            with single_thread_for_batched_cpu(*_tensor_operands(args, kwargs)):
                return func(*args, **kwargs)
        return func(*args, **kwargs)


@contextmanager
def batched_cpu_lu_guard() -> Iterator[None]:
    r"""Inside this scope every LU-family call on a batched CPU tensor runs under one thread (module docstring)."""
    with _BatchedCPULUGuard():
        yield


def with_batched_cpu_lu_guard(function: F) -> F:
    r"""Decorate a driver entry point so its whole body runs inside :func:`batched_cpu_lu_guard`."""
    @functools.wraps(function)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        with batched_cpu_lu_guard():
            return function(*args, **kwargs)
    return guarded  # type: ignore[return-value]
