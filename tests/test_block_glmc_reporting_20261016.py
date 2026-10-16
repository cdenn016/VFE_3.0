r"""``block_glmc`` M3 (gauge-group-buildouts-2026-10-05.md, section 1): reporting and diagnostics without the dense basis.

The projected gauge-equivariance diagnostic draws block-local group elements from the (m, d, d) local basis, so it
neither reads the dense generators nor loses exactness; the geometry-health figure gains a "Complex structure"
panel from the ``block_glmc`` columns and drops it for every other group. Tiny CPU models only.
"""
import warnings

import numpy as np
import pytest
import torch

import vfe3.geometry.groups as groups_module
from tests.profiles import tiny_config
from vfe3.model.model import VFEModel
from vfe3.viz import extract
from vfe3.viz import figures as figs


@pytest.mark.parametrize("name", ("block_so", "block_co", "block_glmc"))
def test_equivariance_draw_uses_the_local_basis(name: str) -> None:
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(tiny_config(gauge_group=name, embed_dim=8, n_heads=2)).double().eval()
    with torch.no_grad():
        model.prior_bank.phi_embed.normal_(mean=0.0, std=0.3)
    residuals = extract.projected_gauge_equivariance(model, torch.tensor([[1, 4, 2]]), n_draws=2)["residuals"]
    assert residuals and all(max(v) < 1e-10 for v in residuals.values()), residuals
    assert isinstance(model.group._generator_source, groups_module._BlockLocalBasis)   # dense basis never built


def _titles(fig) -> set:
    return {ax.get_title() for ax in fig.axes if ax.get_visible()}


def test_geometry_health_shows_the_complex_structure_panel_only_for_block_glmc() -> None:
    steps = np.arange(1, 5, dtype=float)
    base = {"step": steps, "phi_norm_mean": np.linspace(0.1, 0.4, 4), "phi_norm_std": np.full(4, 0.05)}
    glmc = {**base, "gauge_head_phase_abs_mean": np.linspace(0.0, 0.3, 4),
            "belief_circularity_mean": np.linspace(0.2, 0.1, 4)}
    with_panel = figs.plot_geometry_health(glmc)
    without = figs.plot_geometry_health(base)
    assert "Complex structure (block_glmc)" in _titles(with_panel)
    assert "Complex structure (block_glmc)" not in _titles(without)
    assert {"gauge_head_phase_abs_mean", "belief_circularity_mean"} <= set(figs.GEOMETRY_HEALTH_COLUMNS)
    figs.plt.close("all")
