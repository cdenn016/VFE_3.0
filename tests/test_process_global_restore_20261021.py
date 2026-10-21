r"""The autouse restore of the process-global state VFEModel installs (audit 2026-10-20 item 24).

The two tests run in file order: the first installs non-default values through the same setters
``VFEModel.__init__`` calls, and the second must still see the process defaults.
"""
import torch

import vfe3.geometry.transport as transport
import vfe3.model.relational_operators as relational_operators
import vfe3.relational_scoring as scoring


def test_a_installs_non_default_process_globals() -> None:
    torch.set_float32_matmul_precision("high")
    scoring.set_projection_tile_bytes(512 * 1024 ** 2)
    relational_operators.set_spread_tile_bytes(512 * 1024 ** 2)
    scoring.set_diagonal_moments_kernel("triton")
    transport.set_frame_pair_fp64_escalation(True)
    transport.set_frame_pair_condition_tol(0.5)
    assert torch.get_float32_matmul_precision() == "high"
    assert scoring.projection_tile_bytes() == 512 * 1024 ** 2


def test_b_sees_the_process_defaults_again() -> None:
    assert torch.get_float32_matmul_precision() == "highest"
    assert scoring.projection_tile_bytes() == 32 * 1024 ** 2
    assert scoring._tile_token_cap() == 32
    assert relational_operators.spread_tile_bytes() == 32 * 1024 ** 2
    assert scoring.diagonal_moments_kernel() == "torch"
    assert transport.frame_pair_fp64_escalation() is False
    assert transport._FRAME_PAIR_CONDITION_TOL == transport.DEFAULT_FRAME_PAIR_CONDITION_TOL
