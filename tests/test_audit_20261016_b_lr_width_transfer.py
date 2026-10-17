r"""Audit 2026-10-16 F15, F53: the LR width-transfer fit reads seeded rows and the final validation PPL.

F15: a sweep that declares ``seeds`` labels its cells ``<label>__s<seed>`` (``run_sweep``), and the fit's anchored
label pattern skipped every such row, so the promised seed average never ran. F53: the fit selected by
``primary_val_ppl`` (the minimum of any periodic best and the final validation PPL) while its comment and docstring
select by the final validation PPL, so a cell that was healthy at a periodic evaluation and blew up later could
still win. Both are checked on synthetic sweep rows; no model is trained.
"""
import math
from typing import Any, Dict, List, Optional

import ablation


def _ppl(embed_dim: int, multiplier: float, base: float = 0.004, exponent: float = -1.0) -> float:
    r"""A parabola in :math:`\log \eta` with its minimum at :math:`\eta^* = \text{base} (K / 20)^b`."""
    optimum = math.log(base * (embed_dim / 20) ** exponent)
    return 50.0 + (math.log(base * multiplier) - optimum) ** 2


def _row(label: str, final: Optional[float], primary: Optional[float] = None) -> Dict[str, Any]:
    r"""One sweep_results.csv row; ``primary`` defaults to ``final`` (no periodic evaluation ran)."""
    primary = final if primary is None else primary
    return {"label": label, "error": "",
            "final_val_ppl": "" if final is None else final,
            "primary_val_ppl": "" if primary is None else primary}


WIDTHS = [(20, 2), (40, 4), (80, 8)]
MULTIPLIERS = [2.0 ** k for k in range(-4, 3)]                    # 1/16 ... 4


def test_seeded_rows_are_averaged_and_fitted() -> None:
    offsets = {0: -1.0, 1: +1.0}                                   # seed noise that averages out exactly
    rows: List[Dict[str, Any]] = []
    for embed_dim, n_heads in WIDTHS:
        for m in MULTIPLIERS:
            label = ablation._lr_width_transfer_label(embed_dim, n_heads, "m_phi_lr", m)
            rows.extend(_row(f"{label}__s{seed}", _ppl(embed_dim, m) + offset) for seed, offset in offsets.items())
    fit = ablation.fit_lr_width_transfer(rows, {"m_phi_lr": 0.004})["fields"]["m_phi_lr"]
    k20 = fit["widths"][0]
    assert abs(k20["ppl_by_multiplier"]["1"] - _ppl(20, 1.0)) < 1e-12   # mean of the two seeds
    assert k20["diverged_multipliers"] == []
    assert [w["best_multiplier"] for w in fit["widths"]] == [1.0, 0.5, 0.25]
    assert fit["interior_widths"] == 3
    assert abs(fit["exponent"] + 1.0) < 1e-12


def test_one_diverged_seed_makes_the_cell_diverged() -> None:
    rows: List[Dict[str, Any]] = []
    for embed_dim, n_heads in WIDTHS:
        for m in MULTIPLIERS:
            label = ablation._lr_width_transfer_label(embed_dim, n_heads, "m_phi_lr", m)
            rows.append(_row(f"{label}__s0", _ppl(embed_dim, m)))
            rows.append(_row(f"{label}__s1", None if (embed_dim, m) == (20, 2.0) else _ppl(embed_dim, m)))
    k20 = ablation.fit_lr_width_transfer(rows, {"m_phi_lr": 0.004})["fields"]["m_phi_lr"]["widths"][0]
    assert k20["diverged_multipliers"] == [2.0] and k20["ppl_by_multiplier"]["2"] is None


def test_selection_reads_the_final_validation_ppl() -> None:
    r"""At K=20 the x2 cell had the lowest periodic PPL and then blew up: its final PPL loses to x1."""
    rows: List[Dict[str, Any]] = []
    for embed_dim, n_heads in WIDTHS:
        for m in MULTIPLIERS:
            label = ablation._lr_width_transfer_label(embed_dim, n_heads, "m_phi_lr", m)
            if (embed_dim, m) == (20, 2.0):
                rows.append(_row(label, final=900.0, primary=40.0))
            else:
                rows.append(_row(label, final=_ppl(embed_dim, m)))
    k20 = ablation.fit_lr_width_transfer(rows, {"m_phi_lr": 0.004})["fields"]["m_phi_lr"]["widths"][0]
    assert k20["best_multiplier"] == 1.0
    assert k20["ppl_by_multiplier"]["2"] == 900.0


def test_a_missing_final_ppl_counts_as_diverged() -> None:
    rows = [_row(ablation._lr_width_transfer_label(20, 2, "m_phi_lr", m), final=_ppl(20, m)) for m in MULTIPLIERS]
    rows[-1] = {**rows[-1], "final_val_ppl": "", "primary_val_ppl": 45.0}   # finite periodic best, no final
    k20 = ablation.fit_lr_width_transfer(rows, {"m_phi_lr": 0.004})["fields"]["m_phi_lr"]["widths"][0]
    assert k20["diverged_multipliers"] == [MULTIPLIERS[-1]]
