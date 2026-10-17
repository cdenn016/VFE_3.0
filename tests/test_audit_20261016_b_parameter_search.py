r"""Audit 2026-10-16, lane B batch 1, F56: the joint search resolves ``kl_max = 8 K`` per candidate width.

``parameter_optimization`` hands the ablation runner, and builds each candidate from, the declared baseline
(every field at the value the config dictionary gives it, the default where it gives none), not the baseline
``VFE3Config.__post_init__`` resolved, so a candidate or cell that changes ``embed_dim`` resolves its own
``kl_max = 8 K`` as an ablation cell does. The manifest still records the resolved baseline. Nothing is trained.
"""

from __future__ import annotations

import warnings

import pytest

from vfe3.config import VFE3Config


# ---- F56: the joint search resolves kl_max per candidate width --------------------------------------

def test_joint_search_candidates_resolve_kl_max_at_their_own_width(monkeypatch: pytest.MonkeyPatch) -> None:
    import ablation
    import parameter_optimization as po

    assert po.config["kl_max"] is None
    width = 2 * po.config["embed_dim"]
    baseline = po._candidate_config({}, po.config, seed=0, max_steps=None)
    assert baseline["kl_max"] == 8.0 * po.config["embed_dim"]
    wide = po._candidate_config({"embed_dim": width}, po.config, seed=0, max_steps=None)
    assert wide["kl_max"] == 8.0 * width
    # The ablation runner receives the declared baseline, so its cells resolve kl_max per cell as well.
    monkeypatch.setattr(ablation, "fully_gauge_covariant", False)
    with po._ablation_baseline_scope(po.config):
        assert ablation.BASELINE_CONFIG["kl_max"] is None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cell = VFE3Config(**ablation._cell_cfg_dict({"embed_dim": width}, seed=0))
    assert cell.kl_max == 8.0 * width
    # The manifest still records the effective (resolved) baseline.
    assert po._resolved_model_config(po.config)["kl_max"] == 8.0 * po.config["embed_dim"]
