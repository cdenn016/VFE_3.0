r"""The selectable moment-matched covariance floor (``predictive_covariance_floor_mode``, 2026-10-16).

``"prior"`` (default) is the prior-anchored floor :math:`f\Sigma_{p,i}` adopted on 2026-10-12; ``"relative"`` restores
the earlier floor :math:`f \tfrac{\operatorname{tr}C_i}{K}U_iU_i^\top`, :math:`C_i = U_i^{-1}\Sigma_iU_i^{-\top}`, which
scales with the input covariance. Pinned against a dense float64 construction on tiny CPU cases, for the diagonal and
the full family, through the update and the config.
"""
import pytest
import torch

from tests.profiles import test_config, tiny_config
from tests.test_blockwise_moment_matched_20261004 import _case
from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel


def _dense_frame(frame) -> torch.Tensor:   # (..., N, H, d, d) blocks -> (..., N, K, K)
    blocks = frame.forward.double()
    H, d = blocks.shape[-3], blocks.shape[-1]
    return torch.einsum("...hab,hg->...hagb", blocks, torch.eye(H, dtype=torch.float64)).reshape(
        *blocks.shape[:-3], H * d, H * d)


@pytest.mark.parametrize("batch", [(), (2,)])
@pytest.mark.parametrize("diagonal", [True, False])
def test_relative_floor_matches_the_dense_construction(batch, diagonal: bool) -> None:
    ops, mu, sigma, beta, frame, receive = _case(2, 3, False, batch, torch.float64, seed=2)
    if not diagonal:
        A = torch.randn(*sigma.shape, sigma.shape[-1], dtype=torch.float64, generator=torch.Generator().manual_seed(4))
        sigma = A @ A.mT + torch.diag_embed(sigma)
    ops.predictive_covariance_floor_mode = "relative"
    U = _dense_frame(frame)
    U_inv = torch.linalg.inv(U)
    local = torch.diag_embed(sigma) if diagonal else sigma
    C = U_inv @ local @ U_inv.mT
    K = sigma.shape[-1]
    expected = ops.predictive_covariance_floor * (torch.diagonal(C, dim1=-2, dim2=-1).sum(-1) / K)[..., None, None] \
        * (U @ U.mT)
    if diagonal:
        expected = torch.diagonal(expected, dim1=-2, dim2=-1)
    got = ops.floor_term(None, sigma, frame, diagonal=diagonal)
    torch.testing.assert_close(got, expected.expand_as(got), rtol=1e-12, atol=1e-14)


def test_update_swaps_only_the_floor() -> None:
    ops, mu, sigma, beta, frame, receive = _case(2, 3, False, (2,), torch.float64, seed=3)
    anchor = torch.full_like(sigma, 2.0)
    prior = ops.predictive_covariance_update(mu, sigma, beta, frame, receive, floor_anchor=anchor)
    ops.predictive_covariance_floor_mode = "relative"
    relative = ops.predictive_covariance_update(mu, sigma, beta, frame, receive, floor_anchor=anchor)
    swap = ops.relative_floor_term(sigma, frame, True) - ops.predictive_covariance_floor * anchor
    torch.testing.assert_close(relative - prior, swap, rtol=1e-10, atol=1e-14)


def test_relative_floor_scales_with_the_input_covariance() -> None:
    r"""The point of the relative floor: it grows with the covariance it floors (the prior floor does not)."""
    ops, mu, sigma, beta, frame, receive = _case(2, 3, False, (), torch.float64, seed=5)
    ops.predictive_covariance_floor_mode = "relative"
    torch.testing.assert_close(ops.floor_term(None, 100.0 * sigma, frame, diagonal=True),
                               100.0 * ops.floor_term(None, sigma, frame, diagonal=True))


def test_relative_mode_needs_the_frame() -> None:
    ops, mu, sigma, *_ = _case(2, 3, False, (), torch.float64)
    ops.predictive_covariance_floor_mode = "relative"
    with pytest.raises(ValueError, match="frame"):
        ops.floor_term(None, sigma)


def test_config_default_validation_and_wiring() -> None:
    for mode in ("prior", "relative"):
        assert test_config(attention_mode="projected_predictive", predictive_covariance="moment_matched",
                           predictive_covariance_floor_mode=mode).predictive_covariance_floor_mode == mode
    with pytest.raises(ValueError, match="predictive_covariance_floor_mode"):
        test_config(attention_mode="projected_predictive", predictive_covariance="moment_matched", predictive_covariance_floor_mode="absolute")
    cfg = tiny_config(attention_mode="projected_predictive", predictive_covariance="moment_matched",
                      predictive_covariance_floor_mode="relative", generate_embedding_figures=False)
    assert "predictive_covariance_floor_mode" not in cfg.inactive_fields()
    model = VFEModel(cfg)
    modes = {m.predictive_covariance_floor_mode for m in model.modules()
             if hasattr(m, "predictive_covariance_floor_mode")}
    assert modes == {"relative"}
    off = tiny_config(attention_mode="projected_predictive", predictive_covariance="passthrough")
    assert "predictive_covariance_floor_mode" in off.inactive_fields()
