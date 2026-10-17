r"""Audit 2026-10-16, lane C2 (post-hoc analysis): F65, F66, F68, F69, F120.

Synthetic strings and tiny models only (single-digit dimensions, CPU, one thread); nothing is trained.

* F65: :func:`vfe3.posthoc.calibration.text_audit` reports ``same_text`` only when every literal
  segment matched in its wildcard window and every ``<unk>`` wildcard matched characters.
"""

from __future__ import annotations

import pytest
import torch

from vfe3.posthoc import calibration as cal


@pytest.fixture(autouse=True)
def _one_thread() -> None:
    torch.set_num_threads(1)


# ----------------------------------------------------------------------------------------------
# F65: text audit verdict
# ----------------------------------------------------------------------------------------------

def test_text_audit_rejects_a_segment_that_failed_its_wildcard_window() -> None:
    audit = cal.text_audit("The cat", " The <unk> cat\n")         # the <unk> word is absent from the BPE text
    assert audit["mismatched_segments"] == 1
    assert audit["matched_chars"] == audit["word_chars"] and audit["unmatched_bpe_chars"] == 0
    assert not audit["same_text"]
    assert audit["unmatched_unk_wildcards"] == 1


def test_text_audit_rejects_a_trailing_wildcard_with_no_bpe_tail() -> None:
    audit = cal.text_audit("The cat", " The cat <unk>\n")
    assert audit["mismatched_segments"] == 0 and audit["unmatched_bpe_chars"] == 0
    assert audit["unmatched_unk_wildcards"] == 1
    assert not audit["same_text"]


def test_text_audit_still_accepts_matched_wildcards() -> None:
    inner = cal.text_audit("The big cat", " The <unk> cat\n")
    trailing = cal.text_audit("The cat sat", " The cat <unk>\n")
    for audit in (inner, trailing):
        assert audit["same_text"] and audit["unmatched_unk_wildcards"] == 0
        assert audit["mismatched_segments"] == 0
