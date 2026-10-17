r"""Audit 2026-10-16, lane C2 (semantic analysis), batch 4: F74, F76.

Fake runs, planted vector views and a tiny model (vocabulary of about 90 words, K=4, two heads, at most
two layers); nothing is trained, everything runs on the CPU.

* F74: only the flat frame transport :math:`\Omega_{ij} = U_iU_j^{-1}` has identity loop holonomy. The
  REPORT.md gauge paragraph and the phi-section note say so, and name the relational content links as a
  separate connection whose triangle holonomy is not the identity in general, whatever
  ``content_link_edges`` is.
* F76: without training counts the frame reference set is the first candidates in token-id order, and
  the phi note, the ``phi_log`` view note and the phi summary say so instead of "most frequent".
"""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vfe3.semantics import contextual as C
from vfe3.semantics import frames as F
from vfe3.semantics import report as R
from vfe3.semantics import vectors
from vfe3.semantics.vectors import VectorView


# ----------------------------------------------------------------------------------------------
# F74: identity holonomy is a statement about the frame transport only
# ----------------------------------------------------------------------------------------------

def _gauge_run(frames_active: bool = True, **cfg) -> SimpleNamespace:
    base = dict(gauge_group="block_glk", attention_mode="projected_predictive", norm_type_block="none",
                content_link_edges="off")
    base.update(cfg)
    return SimpleNamespace(cfg=SimpleNamespace(**base), frames_active=frames_active, sigma_frozen=False,
                           phi_frozen=False)


def test_gauge_paragraph_restricts_identity_holonomy_to_the_frame_transport() -> None:
    text = R._gauge_paragraph(_gauge_run())
    assert "flat frame transport Ω_ij = U_i U_j⁻¹ telescopes, so its loop holonomy is identically" in text
    assert "With the flat transport" not in text and "so holonomy is identically" not in text
    # the content links are named as a separate connection with edges off, which is the default
    assert "content links" in text and "not the identity in general" in text
    assert "routing_holonomy.csv" in text
    edges = R._gauge_paragraph(_gauge_run(content_link_edges="lowrank"))
    assert "content_link_edges='lowrank'" in edges
    assert "holonomy" not in R._gauge_paragraph(_gauge_run(frames_active=False))


def test_phi_note_names_the_content_links_with_edges_off(tmp_path: Path) -> None:
    from tests.test_semantics_frames import _section_setup

    run, ctx = _section_setup(tmp_path)
    run.cfg.content_link_edges = "off"
    res = F.run_phi_section(run, ctx)
    assert res.status == "ok", res.error
    note = next(n for n in res.notes if n.startswith("flat frame transport"))
    assert "its loop holonomy is identically the identity" in note
    assert "content links" in note and "not the identity in general" in note
    assert "content_link_edges" not in note                             # the edges clause only when on


def test_phi_holonomy_note_adds_the_edge_factors_when_edges_are_on() -> None:
    off = F._frame_holonomy_note(SimpleNamespace(content_link_edges="off"))
    on = F._frame_holonomy_note(SimpleNamespace(content_link_edges="pure_gauge"))
    assert "not the identity in general" in off and "content_link_edges" not in off
    assert "content_link_edges='pure_gauge'" in on and on.startswith(off.split(";")[0])


# ----------------------------------------------------------------------------------------------
# F76: the frame reference set is labeled by how it was chosen
# ----------------------------------------------------------------------------------------------

def test_phi_reference_set_without_counts_is_labeled_id_order(tmp_path: Path) -> None:
    from tests.test_semantics_frames import _section_setup

    run, ctx = _section_setup(tmp_path)
    run.counts = None
    ctx = dataclasses.replace(ctx, counts=None, candidate_ids=np.sort(ctx.candidate_ids))
    res = F.run_phi_section(run, ctx)
    assert res.status == "ok", res.error
    assert res.metrics["ref_order"] == "token_id"
    assert not any("most frequent" in n for n in res.notes)
    assert any("first 20 candidate frames in token-id order (training counts unavailable)" in n
               for n in res.notes)
    summary = R._phi_summary(res)
    assert "most frequent" not in summary and "first 20 candidate frames in token-id order" in summary


def test_phi_reference_phrase_and_legacy_summary() -> None:
    assert F.reference_set_phrase(7, by_count=True) == "the 7 most frequent candidate frames"
    assert F.reference_set_phrase(7, by_count=False) == (
        "the first 7 candidate frames in token-id order (training counts unavailable)")
    legacy = SimpleNamespace(metrics={"chart": {"n": 9, "ok_fraction": 1.0, "n_ref": 5,
                                                "barycenter_converged_all": True}})
    assert "the barycenter over the 5 reference frames converged" in R._phi_summary(legacy)
    counted = SimpleNamespace(metrics={**legacy.metrics, "ref_order": "frequency"})
    assert "the barycenter over the 5 most frequent candidate frames converged" in R._phi_summary(counted)


def test_phi_log_view_note_without_counts(tmp_path: Path) -> None:
    from tests.test_semantics_report import _context, _load, _tiny_run

    run_dir, _ = _tiny_run(tmp_path)
    run = _load(run_dir)
    ctx = _context(run, tmp_path / "out", views=("phi_log",))
    ctx = dataclasses.replace(ctx, counts=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        views = R.build_views(run, ctx)
    note = views["phi_log"].note
    assert "most frequent" not in note and "candidate frames in token-id order" in note
