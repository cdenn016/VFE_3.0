r"""Batched LU-family linear algebra on CPU tensors runs single-threaded (audit 2026-10-20 item 28).

With more than one MKL thread, batched ``torch.linalg.inv_ex`` of 160 x 160 matrices returns a wrong inverse with no
error (or raises a pivot error) and batched ``slogdet`` is off by tens of nats on this machine's oneMKL 2025.3; a
single thread, or one matrix at a time, is exact. These tests run at two threads and compare the guarded results with
a per-matrix float64 reference. On a build without the defect they pass as well.
"""
from typing import Iterator

import pytest
import torch

from vfe3.cpu_linalg import batched_cpu_lu_guard, single_thread_for_batched_cpu
from vfe3.model.relational_operators import _chart_receive

F64 = torch.float64
N = 160


@pytest.fixture
def two_threads() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _batch(seed: int = 0, batch: int = 3) -> torch.Tensor:                 # (batch, N, N) well-conditioned
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, N, N, generator=g, dtype=F64) + N ** 0.5 * torch.eye(N, dtype=F64)


def _loop_inverse(A: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.linalg.inv(a) for a in A])


def test_single_thread_context_restores_the_thread_count(two_threads: None) -> None:
    A = _batch()
    with single_thread_for_batched_cpu(A):
        assert torch.get_num_threads() == 1
    assert torch.get_num_threads() == 2
    with single_thread_for_batched_cpu(A[0]):                                  # one matrix: left alone
        assert torch.get_num_threads() == 2


def test_guard_mode_matches_a_per_matrix_float64_reference(two_threads: None) -> None:
    A = _batch()
    eye = torch.eye(N, dtype=F64)
    with batched_cpu_lu_guard():
        inverse = torch.linalg.inv_ex(A).inverse
        logdet = torch.linalg.slogdet(A).logabsdet
        solved = torch.linalg.solve(A, eye.expand(3, N, N))
        method = A.inverse()
        assert torch.get_num_threads() == 2                                    # restored after each call
    reference = _loop_inverse(A)
    torch.testing.assert_close(inverse, reference, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(solved, reference, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(method, reference, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(logdet, torch.stack([torch.linalg.slogdet(a).logabsdet for a in A]),
                               rtol=1e-12, atol=1e-10)
    assert float((A @ inverse - eye).norm()) < 1e-10


def test_posthoc_and_semantics_entry_points_run_inside_the_guard(two_threads: None) -> None:
    from vfe3.cpu_linalg import with_batched_cpu_lu_guard
    from vfe3.posthoc.report import run_posthoc
    from vfe3.semantics.report import analyze_run

    for entry in (run_posthoc, analyze_run):
        assert getattr(entry, "__wrapped__", None) is not None, entry

    @with_batched_cpu_lu_guard
    def driver(A: torch.Tensor) -> torch.Tensor:
        return torch.linalg.inv_ex(A).inverse

    A = _batch(seed=3)
    torch.testing.assert_close(driver(A), _loop_inverse(A), rtol=1e-10, atol=1e-12)


def test_content_chart_inverse_is_exact_at_two_threads(two_threads: None) -> None:
    r"""The (layers, K, K) content-chart stack of the replay path at K = 160."""
    T = _batch(seed=1)
    g = torch.Generator().manual_seed(2)
    O = 0.01 * torch.randn(3, N, N, generator=g, dtype=F64)
    inverse, receive = _chart_receive(T, O, "first_order")
    reference = _loop_inverse(T)
    torch.testing.assert_close(inverse, reference, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(receive, reference + O, rtol=1e-10, atol=1e-12)
