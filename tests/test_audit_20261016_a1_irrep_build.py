r"""so(N) tensor-power irreps: single-threaded build and the documented skewness (audit 2026-10-16, F38).

The su(N) tensor irreps already built under a single-thread guard because a threaded matmul changed the last
bit of the (2, 2) images and the tower fingerprint hashes the bytes. The so(N) tensor-power builder ('l<p>',
N != 3, p >= 2), whose pinv, Gram-Schmidt and matmuls feed the same fingerprint, ran with the caller's thread
count. It now shares the guard, which restores the caller's count on return and on error. Its images are
skew to rounding (about 1e-16), not exactly: they are not antisymmetrized, since that would change their bytes
and refuse saved checkpoints through the fingerprint. The docstrings now say so.

Tiny float64 builds only.
"""
from __future__ import annotations

import pytest
import torch

F64 = torch.float64


@pytest.fixture
def two_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_tensor_power_irrep_builds_single_threaded_and_restores_the_count(monkeypatch, two_threads) -> None:
    from vfe3.geometry import irreps
    from vfe3.geometry.generators import generate_son

    seen: list[int] = []
    original = irreps._leibniz_action

    def recording(*args, **kwargs):
        seen.append(torch.get_num_threads())
        return original(*args, **kwargs)

    monkeypatch.setattr(irreps, "_leibniz_action", recording)
    rho = irreps._tensor_power_irrep(generate_son(4, dtype=F64), 2, traceless=True)
    assert rho.shape == (6, 9, 9)
    assert seen and set(seen) == {1}
    assert torch.get_num_threads() == 2


def test_tensor_power_irrep_guard_restores_the_count_on_error(monkeypatch, two_threads) -> None:
    from vfe3.geometry import irreps
    from vfe3.geometry.generators import generate_son

    def failing(*args, **kwargs):
        raise RuntimeError("probe")

    monkeypatch.setattr(irreps, "_leibniz_action", failing)
    with pytest.raises(RuntimeError, match="probe"):
        irreps._tensor_power_irrep(generate_son(4, dtype=F64), 2, traceless=True)
    assert torch.get_num_threads() == 2


@pytest.mark.parametrize("N,spec,K", [(4, [("l1", 1), ("l2", 1)], 13), (4, [("l3", 1)], 16)])
def test_so_n_tensor_power_fingerprint_is_thread_independent(N: int, spec, K: int, two_threads) -> None:
    from vfe3.geometry.groups import get_group

    digests = []
    for threads in (1, 2):
        torch.set_num_threads(threads)
        digests.append(get_group("so_n")(K, group_n=N, irrep_spec=spec, dtype=F64).basis_fingerprint)
    assert digests[0] == digests[1]


@pytest.mark.parametrize("N,label,K,exact", [
    (3, "l2", 5, True),                      # SO(3) closed-form ladder
    (4, "l1", 4, True),                      # defining representation
    (5, "w2", 10, True),                     # integer wedge recipe
    (4, "l2", 9, False),                     # Gram-Schmidt tensor power: skew to rounding
    (4, "l3", 16, False),
])
def test_so_n_block_skewness_matches_the_documented_claim(N: int, label: str, K: int, exact: bool) -> None:
    from vfe3.geometry.groups import get_group

    G = get_group("so_n")(K, group_n=N, irrep_spec=[(label, 1)], dtype=F64).generators
    skew = float((G + G.transpose(-1, -2)).abs().max())
    if exact:
        assert skew == 0.0
    else:
        assert skew <= 1e-15 * float(G.abs().max())
