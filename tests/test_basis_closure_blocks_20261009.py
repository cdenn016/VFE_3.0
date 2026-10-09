"""Bracket-closure guard on direct-sum bases scans the distinct local blocks (audit 2026-10-09, gauge-theorist).

``require_basis_closed`` used to skip any basis whose full ``(n_gen, n_gen, K, K)`` scan exceeded
``max_elements``, so block_glk was never checked. A block-diagonal basis is now scanned on its distinct
per-block sub-bases, and a gl(d) block is closed by its full span. Tiny CPU bases only.
"""
import pytest
import torch

from vfe3.geometry import lie_ops
from vfe3.geometry.generators import generate_block_local
from vfe3.geometry.lie_ops import _direct_sum_blocks, require_basis_closed


def _gl_local(d: int) -> torch.Tensor:
    return torch.eye(d * d, dtype=torch.float64).reshape(d * d, d, d)          # E_ij basis of gl(d)


def _not_closed_local() -> torch.Tensor:
    # span{E_01, E_10} is not closed: [E_01, E_10] = E_00 - E_11 lies outside it.
    E01 = torch.zeros(2, 2, dtype=torch.float64)
    E01[0, 1] = 1.
    return torch.stack((E01, E01.T.clone()))


def _clear_cache() -> None:
    lie_ops._BRACKET_CLOSURE_RES.clear()


def test_identical_head_blocks_are_scanned_once() -> None:
    G = generate_block_local(6, 3, _gl_local(2), dtype=torch.float64)          # (12, 6, 6)
    blocks = _direct_sum_blocks(G)
    assert blocks is not None and len(blocks) == 1 and blocks[0].shape == (4, 2, 2)


def test_block_glk_is_verified_beyond_the_full_scan_budget() -> None:
    _clear_cache()
    G = generate_block_local(8, 2, _gl_local(4), dtype=torch.float64)          # (32, 8, 8)
    # Full scan 32^2 * 8^2 = 65536 elements and block scan 16^2 * 4^2 = 4096 both exceed the budget;
    # the gl(4) block is closed by its full span.
    require_basis_closed(G, where="test", max_elements=1000)
    assert lie_ops._BRACKET_CLOSURE_RES[lie_ops._cached_basis_value_signature(G)] == 0.0


def test_non_closed_local_block_is_caught_where_the_full_scan_was_skipped() -> None:
    _clear_cache()
    G = generate_block_local(6, 3, _not_closed_local(), dtype=torch.float64)   # (6, 6, 6)
    # Full scan 6^2 * 6^2 = 1296 > 100 (the former silent skip); block scan 2^2 * 2^2 = 16 fits.
    with pytest.raises(ValueError, match="not closed under the Lie bracket"):
        require_basis_closed(G, where="test", max_elements=100)


def test_a_generator_shared_across_blocks_keeps_the_whole_basis_scan() -> None:
    local = _not_closed_local()
    G = generate_block_local(4, 2, local, dtype=torch.float64)                 # (4, 4, 4)
    tied = G[:2] + G[2:]                                                       # each acts on both blocks
    assert _direct_sum_blocks(tied) is None
