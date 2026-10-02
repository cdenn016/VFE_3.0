r"""LaTeX tabular environments and markdown for the manuscript tables of ``GL(K)_attention_v2``.

Each builder returns ``(latex, markdown, rows)``: ``latex`` is the complete ``tabular``/``tabularx``
environment with the column specification and header row copied from the manuscript, ready to
replace the placeholder's tabular (the float, caption and label stay in the manuscript); ``markdown``
carries the same cells plus the source of every number; ``rows`` is the machine-readable record.

* :func:`main_results_table`: ``tab:placeholder_main_results`` (M2). Per arm: the number of seeds
  with a selected test value, context-processing and total parameters, test CE and PPL as mean
  :math:`\pm` SD over seeds (ddof = 1), and device-hours per run.
* :func:`per_seed_table`: ``tab:placeholder_per_seed``. Per run: model and data-order seed, selected
  step, validation CE, test CE and PPL under the declared selection rule.
* :func:`relational_run_table`: ``tab:relational_run`` for the evaluated run.
* :func:`resources_table`: ``tab:placeholder_resources`` (evaluated run | final run).
* :func:`methods_contract_table`: ``tab:placeholder_methods_contract`` from the final run's saved
  artifacts; a field the run does not record reads ``not recorded``.

A cell whose value does not exist yet reads ``pending`` (the manuscript convention). Arm labels are
written verbatim (they are manuscript row text and may contain LaTeX).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from vfe3.paper.runs import (
    Arm, Manifest, RunRecord, device_hours, finite_float, optimizer_groups, parameter_counts,
    posthoc_pass, select_checkpoint,
)

PENDING = "pending"
NOT_RECORDED = "not recorded"
CE_DIGITS = 4            # test / validation CE decimals
PPL_DIGITS_MAIN = 2      # PPL decimals in M2 and the per-seed table
PPL_DIGITS_RUN = 4       # PPL decimals in tab:relational_run (as in the manuscript)
HASH_CHARS_LATEX = 12    # hashes are shortened to this many hex characters in LaTeX (full in markdown)

# Column specifications and header rows copied from GL(K)_attention_v2.tex.
MAIN_RESULTS_SPEC = r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}Xccccc@{}}"
MAIN_RESULTS_HEADER = r"Model arm & Seeds & Parameters & Test CE & Test PPL & GPU-hours \\"
PER_SEED_SPEC = r"\begin{tabular}{lccccc}"
PER_SEED_HEADER = r"Model arm & Seeds & Selected step & Validation CE & Test CE & Test PPL \\"
RELATIONAL_RUN_SPEC = r"\begin{tabular}{lrr}"
RELATIONAL_RUN_HEADER = r"Evaluation & Step & Perplexity \\"
RESOURCES_SPEC = r"\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}Xcc@{}}"
RESOURCES_HEADER = r"Parameter role or resource & Evaluated run & Final run \\"
METHODS_SPEC = r"\begin{tabular}{p{0.30\linewidth}p{0.62\linewidth}}"
METHODS_HEADER = r"Reporting field & Final-run value or artifact identifier \\"

RESOURCE_ROWS: Tuple[str, ...] = (
    "Per-type tables (token frames, prior means and covariances, decoder)",
    "Positional frame generators",
    "Context-processing parameters (interface maps, MLP, normalization)",
    "Parameters read per token",
    "Optimizer state and peak device memory",
    "Training throughput and device-hours",
    "Inference latency at declared batch and context",
)
METHODS_ROWS: Tuple[str, ...] = (
    "Source and configuration", "Data and tokenizer", "Architecture", "Initialization",
    "Optimization", "Numerical realization", "Training budget", "Evaluation",
)

# Resolved-operator selectors reported from config.json["relational_operators"], when recorded.
SELECTOR_KEYS: Tuple[str, ...] = (
    "attention_mode", "routing", "mode", "attention_score", "readout_mode", "inference_semantics",
    "input_scope", "output_scope", "mean_update", "content_mode", "score_parameterization",
    "covariance_update", "temperature_convention",
)
PRECISION_KEYS: Tuple[str, ...] = (
    "amp_dtype", "fp32_matmul_precision", "full_cov_kl_precision",
    "decode_av_precision", "deterministic",
)
FACTORIZATION_KEYS: Tuple[str, ...] = (
    "projected_factor_route", "projected_cov_condition_recovery",
)
EXPONENTIAL_KEYS: Tuple[str, ...] = ("exp_fp64_mode", "transport_chart_max_norm")


# ----------------------------------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------------------------------

_TEX_ESCAPES = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def tex_escape(text: Any) -> str:
    r"""Escape LaTeX special characters in plain text."""
    return "".join(_TEX_ESCAPES.get(ch, ch) for ch in str(text))


def tex_tt(text: Any) -> str:
    r"""``\texttt{...}`` of an identifier, breakable after ``_``, ``/``, ``=`` and a ``.`` that is not a
    decimal point."""
    text = str(text)
    out = []
    for i, ch in enumerate(text):
        out.append(_TEX_ESCAPES.get(ch, ch))
        decimal_point = ch == "." and i + 1 < len(text) and text[i + 1].isdigit()
        if ch in "_/.=" and not decimal_point:
            out.append(r"\allowbreak ")
    return r"\texttt{" + "".join(out) + "}"


def tex_int(value: Any) -> str:
    r"""``$456{,}654{,}162$`` (the manuscript's thousands separator)."""
    return "$" + f"{int(value):,}".replace(",", "{,}") + "$"


def tex_float(value: Optional[float], digits: int) -> str:
    return PENDING if value is None else f"${value:.{digits}f}$"


def mean_sd(values: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    r"""Mean and unbiased SD (ddof = 1; None for fewer than two values)."""
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, None
    return mean, math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def tex_mean_sd(values: Sequence[float], digits: int) -> str:
    mean, sd = mean_sd(values)
    if mean is None:
        return PENDING
    return f"${mean:.{digits}f} \\pm {sd:.{digits}f}$" if sd is not None else f"${mean:.{digits}f}$"


def md_mean_sd(values: Sequence[float], digits: int) -> str:
    mean, sd = mean_sd(values)
    if mean is None:
        return PENDING
    return f"{mean:.{digits}f} ± {sd:.{digits}f}" if sd is not None else f"{mean:.{digits}f}"


def count_short(value: Any) -> str:
    r"""Compact count: ``5252608 -> '5.25M'``, ``456654162 -> '456.7M'``, ``1.2e9 -> '1.20B'``."""
    value = float(value)
    for scale, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(value) >= scale:
            scaled = value / scale
            return f"{scaled:.1f}{suffix}" if abs(scaled) >= 100 else f"{scaled:.2f}{suffix}"
    return f"{value:.0f}"


class Code(str):
    r"""An identifier: ``\texttt`` in LaTeX, backticks in markdown."""


class Hash(str):
    r"""A hex digest: shortened in LaTeX, full in markdown."""


Item = Tuple[str, Union[str, Code, Hash]]


def _render_value(value: Any, latex: bool) -> str:
    if isinstance(value, Hash):
        return tex_tt(value[:HASH_CHARS_LATEX]) if latex else f"`{value}`"
    if isinstance(value, Code):
        return tex_tt(value) if latex else f"`{value}`"
    return tex_escape(value) if latex else str(value)


def render_items(items: Sequence[Item], latex: bool) -> str:
    r"""``label value; label value; ...`` of one methods-contract field."""
    if not items:
        return NOT_RECORDED
    parts = []
    for label, value in items:
        text = _render_value(value, latex)
        parts.append(f"{tex_escape(label) if latex else label} {text}".strip())
    return "; ".join(parts)


def _tabular(spec: str, header: str, body: Sequence[str]) -> str:
    end = r"\end{tabularx}" if spec.startswith(r"\begin{tabularx}") else r"\end{tabular}"
    return "\n".join([spec, r"\toprule", header, r"\midrule", *body, r"\bottomrule", end]) + "\n"


def _md_row(cells: Sequence[Any]) -> str:
    return "| " + " | ".join(str(c).replace("|", "\\|") for c in cells) + " |"


# ----------------------------------------------------------------------------------------------
# M2 and the per-seed table
# ----------------------------------------------------------------------------------------------

def _arm_record(arm: Arm, rule: str, *, read_checkpoints: bool) -> Dict[str, Any]:
    r"""Selection, parameters and device-hours of every run of one arm."""
    runs = []
    for run in arm.runs:
        selection = select_checkpoint(run, rule)
        hours, hours_source = device_hours(run)
        runs.append({"run": run, "selection": selection, "hours": hours, "hours_source": hours_source,
                     "params": parameter_counts(run, read_checkpoints=read_checkpoints)})
    return {"arm": arm, "runs": runs}


def _param_cell(params: Mapping[str, Any], latex: bool) -> str:
    total = params.get("total")
    context = (params.get("context_processing") or {}).get("total")
    if total is None:
        return PENDING
    if context is None:
        return f"{count_short(total)} total"
    return f"{count_short(context)} / {count_short(total)}"


def main_results_table(
    manifest: Manifest,

    *,
    read_checkpoints: bool = True,
) -> Tuple[str, str, List[Dict[str, Any]]]:
    r"""``tab:placeholder_main_results``: one row per declared arm, with section headings.

    Test CE and PPL are the mean :math:`\pm` SD (ddof = 1) over the runs whose selected test value
    exists, with :math:`\mathrm{PPL}_s = e^{\mathrm{CE}_s}` per seed; one run shows the value alone.
    Parameters are context-processing / total (``count_short``); GPU-hours are the mean device-hours
    per run. A config difference within an arm is written as a LaTeX comment above its row and listed
    in the markdown.
    """
    body: List[str] = []
    md = [_md_row(["Model arm", "Seeds", "Parameters (context / total)", "Test CE", "Test PPL",
                   "GPU-hours per run", "Notes"]), _md_row(["---"] * 7)]
    rows: List[Dict[str, Any]] = []
    current_group: Optional[str] = None
    for index, arm in enumerate(manifest.arms):
        if arm.group and arm.group != current_group:
            if body:
                body.append(r"\midrule")
            body.append(r"\multicolumn{6}{@{}l}{\emph{" + arm.group + r"}}\\")
            md.append(_md_row([f"*{arm.group}*", "", "", "", "", "", ""]))
            current_group = arm.group
        record = _arm_record(arm, manifest.selection, read_checkpoints=read_checkpoints)
        ok = [r for r in record["runs"] if r["selection"]["status"] == "ok"]
        ces = [r["selection"]["test_ce"] for r in ok]
        ppls = [r["selection"]["test_ppl"] for r in ok]
        hours = [r["hours"] for r in record["runs"] if r["hours"] is not None]
        params = next((r["params"] for r in record["runs"] if r["params"].get("total") is not None), {})
        totals = {r["params"].get("total") for r in record["runs"] if r["params"].get("total") is not None}
        notes: List[str] = []
        if not arm.runs:
            notes.append("no runs declared")
        elif len(ok) < len(arm.runs):
            missing = [r["run"].name for r in record["runs"] if r["selection"]["status"] != "ok"]
            notes.append(f"{len(ok)} of {len(arm.runs)} runs have a test value under rule "
                         f"'{manifest.selection}' (missing: {', '.join(missing)})")
        if len(totals) > 1:
            notes.append(f"parameter totals differ across runs: {sorted(totals)}")
        violations = arm.homogeneity.get("violations") or []
        if violations:
            notes.append("config differs across runs in: " + ", ".join(v["field"] for v in violations))
        if arm.homogeneity.get("duplicate_seed_pairs"):
            notes.append(f"duplicate (seed, data seed): {arm.homogeneity['duplicate_seed_pairs']}")
        seeds = (str(len(ok)) if len(ok) == len(arm.runs) else f"{len(ok)} of {len(arm.runs)}") \
            if arm.runs else PENDING
        hours_mean = sum(hours) / len(hours) if hours else None
        if violations:
            body.append("% " + arm.label.replace("\n", " ") + ": config differs across runs in "
                        + ", ".join(v["field"] for v in violations))
        body.append(" & ".join([
            arm.label, seeds, _param_cell(params, True) if params else PENDING,
            tex_mean_sd(ces, CE_DIGITS), tex_mean_sd(ppls, PPL_DIGITS_MAIN),
            tex_float(hours_mean, 1),
        ]) + r" \\")
        md.append(_md_row([arm.label, seeds, _param_cell(params, False) if params else PENDING,
                           md_mean_sd(ces, CE_DIGITS), md_mean_sd(ppls, PPL_DIGITS_MAIN),
                           f"{hours_mean:.2f}" if hours_mean is not None else PENDING,
                           "; ".join(notes)]))
        rows.append({
            "arm": arm.label, "group": arm.group, "n_runs": len(arm.runs), "n_selected": len(ok),
            "test_ce": ces, "test_ppl": ppls, "test_ce_mean_sd": mean_sd(ces),
            "test_ppl_mean_sd": mean_sd(ppls), "device_hours": hours, "parameters": params,
            "notes": notes, "homogeneity": arm.homogeneity,
            "runs": [{"run_dir": str(r["run"].run_dir), "seed": r["run"].seed,
                      "data_seed": r["run"].data_seed, "selection": r["selection"],
                      "device_hours": r["hours"], "device_hours_source": r["hours_source"],
                      "parameters_source": r["params"].get("source")} for r in record["runs"]],
        })
    caption = (f"\nSelection rule: '{manifest.selection}'. Test CE/PPL: mean ± SD (ddof = 1) over seeds; "
               "PPL is exp(CE) per seed. Parameters: context-processing / total. GPU-hours: mean "
               "device-hours per run (one device).\n")
    return _tabular(MAIN_RESULTS_SPEC, MAIN_RESULTS_HEADER, body), "\n".join(md) + "\n" + caption, rows


def _seed_cell(run: RunRecord) -> str:
    model = str(run.seed) if run.seed is not None else NOT_RECORDED
    if run.data_seed is not None:
        data = str(run.data_seed)
    else:
        data = "unspecified" if run.data_seed_recorded else NOT_RECORDED
    return f"{model} / {data}"


def per_seed_table(manifest: Manifest) -> Tuple[str, str, List[Dict[str, Any]]]:
    r"""``tab:placeholder_per_seed``: one row per run (model seed / data-order seed)."""
    body: List[str] = []
    md = [_md_row(["Model arm", "Seeds (model / data)", "Selected step", "Validation CE", "Test CE",
                   "Test PPL", "Run", "Sources"]), _md_row(["---"] * 8)]
    rows: List[Dict[str, Any]] = []
    for arm in manifest.arms:
        if not arm.runs:
            body.append(" & ".join([arm.label] + [PENDING] * 5) + r" \\")
            md.append(_md_row([arm.label] + [PENDING] * 5 + ["", ""]))
            continue
        for position, run in enumerate(arm.runs):
            sel = select_checkpoint(run, manifest.selection)
            label = arm.label if position == 0 else ""
            step = tex_int(sel["step"]) if sel["step"] is not None else PENDING
            body.append(" & ".join([
                label, _seed_cell(run), step, tex_float(sel["val_ce"], CE_DIGITS),
                tex_float(sel["test_ce"], CE_DIGITS), tex_float(sel["test_ppl"], PPL_DIGITS_MAIN),
            ]) + r" \\")
            sources = f"val: {sel['val_source'] or '–'}; test: {sel['test_source'] or sel['reason']}"
            md.append(_md_row([
                label, _seed_cell(run), sel["step"] if sel["step"] is not None else PENDING,
                f"{sel['val_ce']:.{CE_DIGITS}f}" if sel["val_ce"] is not None else PENDING,
                f"{sel['test_ce']:.{CE_DIGITS}f}" if sel["test_ce"] is not None else PENDING,
                f"{sel['test_ppl']:.{PPL_DIGITS_MAIN}f}" if sel["test_ppl"] is not None else PENDING,
                run.name, sources]))
            rows.append({"arm": arm.label, "run_dir": str(run.run_dir), "seed": run.seed,
                         "data_seed": run.data_seed, **sel})
    note = f"\nSelection rule: '{manifest.selection}', applied identically to every run.\n"
    return _tabular(PER_SEED_SPEC, PER_SEED_HEADER, body), "\n".join(md) + "\n" + note, rows


# ----------------------------------------------------------------------------------------------
# tab:relational_run
# ----------------------------------------------------------------------------------------------

def relational_run_rows(run: RunRecord) -> List[Dict[str, Any]]:
    r"""The five manuscript rows of ``tab:relational_run`` with CE, PPL, step and source.

    Recorded values (summary.json) are preferred; the post-hoc full-split passes fill the rows that
    runs trained before 2026-09-28 did not record (test at the best weights, the training sample).
    """
    s = run.summary
    final_step = run.final_step()
    best_step = s.get("best_step") if isinstance(s.get("best_step"), int) else None
    final_headline = s.get("evaluation_weights", "final") == "final" and not s.get("reloaded_best")

    def from_pass(checkpoint: str, split: str) -> Tuple[Optional[float], Optional[str]]:
        entry = posthoc_pass(run, checkpoint, split)
        return ((float(entry["ce"]), f"posthoc.json {checkpoint} {split}") if entry else (None, None))

    rows: List[Dict[str, Any]] = []

    def add(label: str, step: Optional[int], ce: Optional[float], source: Optional[str]) -> None:
        rows.append({"label": label, "step": step, "ce": ce,
                     "ppl": math.exp(ce) if ce is not None else None, "source": source})

    best_ppl = finite_float(s.get("best_val_ppl"))
    add("Best recorded validation", best_step,
        math.log(best_ppl) if best_ppl else None, "summary.json best_val_ppl" if best_ppl else None)
    ce, source = finite_float(s.get("final_val_ce")), "summary.json final_val_ce"
    if ce is None:
        ce, source = from_pass("final_model.pt", "validation")
    add("Final validation", final_step, ce, source if ce is not None else None)
    ce, source = (finite_float(s.get("test_ce")), "summary.json test_ce") if final_headline else (None, None)
    if ce is None:
        ce, source = from_pass("final_model.pt", "test")
    add("Test, final weights", final_step, ce, source)
    best = select_checkpoint(run, "best")
    add("Test, best-validation weights", best["step"], best["test_ce"], best["test_source"])
    ce, source = finite_float(s.get("train_sample_ce_final")), "summary.json train_sample_ce_final"
    if ce is None:
        ce, source = from_pass("final_model.pt", "train_sample")
    add("Training-split sample, evaluation protocol", final_step, ce, source)
    return rows


def relational_run_table(run: Optional[RunRecord]) -> Tuple[str, str, List[Dict[str, Any]]]:
    r"""``tab:relational_run`` for the evaluated run (all cells pending when none is declared)."""
    rows = relational_run_rows(run) if run is not None else [
        {"label": label, "step": None, "ce": None, "ppl": None, "source": None}
        for label in ("Best recorded validation", "Final validation", "Test, final weights",
                      "Test, best-validation weights", "Training-split sample, evaluation protocol")]
    body = [" & ".join([r["label"], tex_int(r["step"]) if r["step"] is not None else PENDING,
                        tex_float(r["ppl"], PPL_DIGITS_RUN)]) + r" \\" for r in rows]
    md = [_md_row(["Evaluation", "Step", "Perplexity", "CE (nats)", "Source"]), _md_row(["---"] * 5)]
    for r in rows:
        md.append(_md_row([r["label"], r["step"] if r["step"] is not None else PENDING,
                           f"{r['ppl']:.{PPL_DIGITS_RUN}f}" if r["ppl"] is not None else PENDING,
                           f"{r['ce']:.6f}" if r["ce"] is not None else PENDING, r["source"] or "–"]))
    head = f"Run: `{run.run_dir}`\n\n" if run is not None else "No evaluated run declared.\n\n"
    return _tabular(RELATIONAL_RUN_SPEC, RELATIONAL_RUN_HEADER, body), head + "\n".join(md) + "\n", rows


# ----------------------------------------------------------------------------------------------
# tab:placeholder_resources
# ----------------------------------------------------------------------------------------------

def resource_record(run: RunRecord, *, read_checkpoints: bool = True) -> Dict[str, Any]:
    r"""Storage and cost facts of one run, each with its source.

    Sources: ``posthoc.json`` resources (post-hoc record, which alone measures inference latency),
    then ``resources.json`` (runs trained on or after 2026-09-28), then
    :func:`vfe3.run_resources.throughput_summary` on the run directory and, with
    ``read_checkpoints``, :func:`vfe3.run_resources.optimizer_state_bytes` on the latest training
    checkpoint (tensor metadata only). Memory is reported in GB (:math:`10^9` bytes); the logged
    ``peak_mem_mb`` is in :math:`10^6` bytes.
    """
    ph = (run.posthoc or {}).get("resources") or {}
    rj = run.resources or {}
    out: Dict[str, Any] = {"run_dir": str(run.run_dir),
                           "parameters": parameter_counts(run, read_checkpoints=read_checkpoints)}
    opt = (ph.get("optimizer_state") or {}).get("optimizer_state_bytes")
    opt_source = "posthoc.json" if isinstance(opt, int) else None
    if opt_source is None and isinstance((rj.get("optimizer_state_bytes") or {}).get("total"), int):
        opt, opt_source = rj["optimizer_state_bytes"]["total"], "resources.json"
    if opt_source is None and read_checkpoints:
        from vfe3.paper.runs import latest_training_checkpoint
        path = latest_training_checkpoint(run)
        if path is not None:
            try:
                from vfe3.run_resources import optimizer_state_bytes
                opt, opt_source = optimizer_state_bytes(path)["optimizer_state_bytes"], path.name
            except Exception as exc:                          # noqa: BLE001 - recorded, not raised
                run.notes.append(f"optimizer state bytes from {path.name} failed: {exc!r}")
    out["optimizer_state_bytes"] = opt if opt_source else None
    out["optimizer_state_source"] = opt_source
    throughput = ph.get("throughput")
    if not isinstance(throughput, Mapping):
        try:
            from vfe3.run_resources import throughput_summary
            throughput = throughput_summary(run.run_dir)
        except Exception as exc:                              # noqa: BLE001
            throughput = {}
            run.notes.append(f"throughput summary failed: {exc!r}")
    peak = finite_float(rj.get("peak_gpu_memory_mb"))
    out["peak_memory_mb"], out["peak_memory_source"] = (
        (peak, "resources.json") if peak is not None else
        (finite_float(throughput.get("peak_mem_mb")), "metrics.csv peak_mem_mb"))
    tps = ((throughput.get("throughput_tokens_per_s") or {}).get("tokens_per_s") or {}).get("median")
    if tps is None:
        tps = (((rj.get("throughput_tokens_per_s") or {}).get("tokens_per_s")) or {}).get("q50")
    out["tokens_per_s_median"] = finite_float(tps)
    out["device_hours"], out["device_hours_source"] = device_hours(run)
    out["device_name"] = rj.get("device_name") or run.provenance.get("device_name")
    latency = ph.get("inference_latency") or {}
    out["latency"] = latency if latency.get("status") == "ok" else None
    return out


def _resource_cells(record: Optional[Mapping[str, Any]], latex: bool) -> List[str]:
    if record is None:
        return [PENDING] * len(RESOURCE_ROWS)
    params = record["parameters"]
    num = tex_int if latex else (lambda v: f"{int(v):,}")
    pct = (lambda s: f"${100 * s:.2f}\\%$") if latex else (lambda s: f"{100 * s:.2f}%")

    def section(name: str) -> str:
        entry = params.get(name) or {}
        if entry.get("total") is None:
            return NOT_RECORDED
        share = entry.get("share")
        return f"{num(entry['total'])} ({pct(share)})" if share is not None else num(entry["total"])

    read = params.get("parameters_read_per_token")
    opt, peak = record["optimizer_state_bytes"], record["peak_memory_mb"]
    memory = "; ".join([f"{opt / 1e9:.2f} GB optimizer state" if opt else f"optimizer state {NOT_RECORDED}",
                        f"peak {peak / 1e3:.2f} GB" if peak else f"peak {NOT_RECORDED}"])
    tps, hours = record["tokens_per_s_median"], record["device_hours"]
    throughput = "; ".join([f"median {tps:,.0f} tokens/s" if tps else f"throughput {NOT_RECORDED}",
                            f"{hours:.2f} device-hours" if hours is not None else f"device-hours {NOT_RECORDED}"])
    lat = record["latency"]
    latency = (f"{1000 * lat['seconds_per_batch_median']:.1f} ms per batch of {lat['batch_size']} x "
               f"{lat['context']} ({lat.get('precision')})" if lat else NOT_RECORDED)
    cells = [section("per_type_tables"), section("positional_frames"), section("context_processing"),
             num(read) if read is not None else NOT_RECORDED, memory, throughput, latency]
    return [c if not latex or c.startswith("$") or "$" in c else tex_escape(c) for c in cells]


def resources_table(
    evaluated: Optional[RunRecord],
    final:     Optional[RunRecord],

    *,
    read_checkpoints: bool = True,
) -> Tuple[str, str, Dict[str, Any]]:
    r"""``tab:placeholder_resources`` with the evaluated and final runs as columns."""
    records = {key: (resource_record(run, read_checkpoints=read_checkpoints) if run is not None else None)
               for key, run in (("evaluated", evaluated), ("final", final))}
    cols_tex = {key: _resource_cells(rec, True) for key, rec in records.items()}
    cols_md = {key: _resource_cells(rec, False) for key, rec in records.items()}
    body = [" & ".join([label, cols_tex["evaluated"][i], cols_tex["final"][i]]) + r" \\"
            for i, label in enumerate(RESOURCE_ROWS)]
    md = [_md_row(["Parameter role or resource", "Evaluated run", "Final run"]), _md_row(["---"] * 3)]
    md += [_md_row([label, cols_md["evaluated"][i], cols_md["final"][i]]) for i, label in enumerate(RESOURCE_ROWS)]
    lines = ["\n".join(md), ""]
    for key, rec in records.items():
        if rec is not None:
            total = rec["parameters"].get("total")
            lines.append(f"{key}: `{rec['run_dir']}`; total parameters {total:,} "
                         f"({rec['parameters'].get('source')}); device {rec['device_name']}; "
                         f"device-hours from {rec['device_hours_source']}; optimizer state from "
                         f"{rec['optimizer_state_source']}." if total is not None else
                         f"{key}: `{rec['run_dir']}`; parameters {NOT_RECORDED}.")
    return _tabular(RESOURCES_SPEC, RESOURCES_HEADER, body), "\n".join(lines) + "\n", records


# ----------------------------------------------------------------------------------------------
# tab:placeholder_methods_contract
# ----------------------------------------------------------------------------------------------

def _raw_items(run: RunRecord, keys: Sequence[str]) -> List[Item]:
    r"""``key=value`` items for the keys the saved config records (absent keys are omitted)."""
    return [("", Code(f"{key}={_fmt_value(run.raw(key))}")) for key in keys if key in run.config_raw]


def _fmt_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _source_items(run: RunRecord) -> List[Item]:
    p, rec = run.provenance, run.config_record
    items: List[Item] = []
    archive = p.get("source_archive")
    if isinstance(archive, Mapping) and archive.get("available"):
        items.append(("source archive", Code(str(archive.get("path")))))
        items.append(("sha256", Hash(str(archive.get("sha256")))))
        items.append(("tree sha256", Hash(str(archive.get("tree_sha256")))))
        items.append(("files", str(archive.get("n_files"))))
        if p.get("source_archive_drift"):
            end = p.get("source_archive_end") or {}
            items.append(("drifted during the run; end archive", Code(str(end.get("path")))))
            items.append(("end sha256", Hash(str(end.get("sha256")))))
    else:
        items.append(("source archive", NOT_RECORDED))
    if isinstance(p.get("git_sha"), str):
        dirty = p.get("git_dirty")
        items.append(("Git revision at start", Hash(p["git_sha"])))
        items.append(("dirty tree", "yes" if dirty else "no" if dirty is False else NOT_RECORDED))
        final = p.get("final_git_identity") or {}
        if p.get("source_identity_drift") and isinstance(final.get("git_sha"), str):
            items.append(("source drifted; final revision", Hash(final["git_sha"])))
    schema = run.raw("__vfe3_config_schema__", run.dropped_config_fields.get("__vfe3_config_schema__"))
    items.append(("config schema", str(schema) if schema is not None else NOT_RECORDED))
    retired = sorted(k for k in run.dropped_config_fields if k != "__vfe3_config_schema__")
    if retired:
        items.append(("retired fields ignored on reload", ", ".join(retired)))
    ops = rec.get("relational_operators")
    if isinstance(ops, Mapping):
        items.extend(("", Code(f"{key}={ops[key]}")) for key in SELECTOR_KEYS if key in ops)
    else:
        items.append(("resolved operator selectors", NOT_RECORDED))
    mlp = (rec.get("executable_build") or {}).get("block_mlp")
    if isinstance(mlp, Mapping) and mlp.get("report_label"):
        items.append(("MLP build", Code(str(mlp["report_label"]))))
    inactive = rec.get("inactive_fields")
    items.append(("accepted but inactive selectors",
                  ", ".join(sorted(inactive)) if isinstance(inactive, Mapping) and inactive
                  else "none" if isinstance(inactive, Mapping) else NOT_RECORDED))
    return items


def _tokens_seen(run: RunRecord) -> Tuple[Optional[int], Optional[str]]:
    value = (run.resources or {}).get("tokens_seen")
    if isinstance(value, int):
        return value, "resources.json tokens_seen"
    metrics = run.metrics()
    if metrics is not None and "tokens_seen" in metrics:
        finite = [v for v in metrics["tokens_seen"] if math.isfinite(v)]
        if finite:
            return int(max(finite)), "metrics.csv tokens_seen"
    point = (run.summary.get("scaling_point") or {}).get("tokens_seen")
    if isinstance(point, int):
        return point, "summary.json scaling_point (max_steps x batch x context)"
    return None, None


def _data_items(run: RunRecord) -> List[Item]:
    p = run.provenance
    items: List[Item] = [("dataset", Code(run.dataset) if run.dataset else NOT_RECORDED)]
    items.append(("split revision", str(p["dataset_revision"]) if p.get("dataset_revision") else NOT_RECORDED))
    manifest = p.get("tokenizer_manifest")
    if isinstance(manifest, Mapping) and manifest.get("status") == "recorded":
        items.append(("tokenizer", Code(f"{manifest.get('kind')}/{manifest.get('encoding')}")))
        items.append(("library version", str(manifest.get("library_version"))))
        digest = manifest.get("ranks_sha256") or manifest.get("vocab_sha256")
        if digest:
            items.append(("table sha256", Hash(str(digest))))
    else:
        status = ((p.get("selection_data_identity") or {}).get("source") or {}).get("tokenizer_provenance_status")
        tag = p.get("tokenizer_tag")
        items.append(("tokenizer", Code(str(tag)) if tag else NOT_RECORDED))
        items.append(("tokenizer manifest", NOT_RECORDED + (f" ({status})" if status else "")))
    items.append(("vocabulary", str(run.raw("vocab_size")) if "vocab_size" in run.config_raw else NOT_RECORDED))
    policy = p.get("concatenation_policy")
    items.append(("concatenation policy", str(policy) if policy else NOT_RECORDED))
    for split in ("train", "val", "test"):
        digest, n = p.get(f"{split}_data_sha256"), p.get(f"{split}_data_n_tokens")
        if isinstance(digest, str):
            items.append((f"{split} tokens sha256", Hash(digest)))
            if isinstance(n, int):
                items.append((f"{split} tokens", f"{n:,}"))
        else:
            items.append((f"{split} tokens sha256", NOT_RECORDED))
    unique = p.get("train_data_n_tokens")
    seen, _ = _tokens_seen(run)
    if isinstance(unique, int) and unique > 0 and seen is not None:
        items.append(("passes over the unique training tokens", f"{seen / unique:.2f}"))
    else:
        items.append(("passes", NOT_RECORDED))
    return items


def _architecture_items(run: RunRecord) -> List[Item]:
    rec = run.config_record
    ops = rec.get("relational_operators") if isinstance(rec.get("relational_operators"), Mapping) else {}
    mlp = (rec.get("executable_build") or {}).get("block_mlp") or {}
    items = _raw_items(run, ("embed_dim", "gauge_group", "n_heads", "n_layers"))
    blocks = (run.summary.get("scaling_point") or {}).get("n_blocks")
    if not isinstance(blocks, int) and isinstance(mlp.get("irrep_dims"), list):
        blocks = len(mlp["irrep_dims"])
    items.append(("frame blocks", str(blocks) if isinstance(blocks, int) else NOT_RECORDED))
    for key in ("score_rank_per_head", "content_rank_per_head"):              # r_h and s_h
        items.append(("", Code(f"{key}={ops[key]}")) if key in ops else (key, NOT_RECORDED))
    items += _raw_items(run, ("norm_type_block", "norm_type_final", "layernorm_affine",
                              "attention_residual", "use_block_mlp", "block_mlp_mode",
                              "block_mlp_expansion"))
    if (run.raw("use_block_mlp") and run.raw("block_mlp_mode") == "coordinate"
            and isinstance(run.raw("block_mlp_expansion"), int) and isinstance(run.raw("embed_dim"), int)):
        items.append(("MLP hidden width", str(run.raw("block_mlp_expansion") * run.raw("embed_dim"))))
    items += _raw_items(run, ("decode_mode", "decode_bias", "decode_unigram_prior", "unigram_kappa",
                              "untie_decode_bank", "pos_phi"))
    return items


def _initialization_items(run: RunRecord) -> List[Item]:
    ops = run.config_record.get("relational_operators")
    items = _raw_items(run, ("mu_init_std", "sigma_init", "phi_scale", "pos_phi_scale",
                             "relational_output_init"))
    if isinstance(ops, Mapping) and "score_initialization" in ops:
        items.append(("score maps", Code(str(ops["score_initialization"]))))
    return items


def _group_name(group: Mapping[str, Any]) -> str:
    names = group.get("names") or []
    if 1 <= len(names) <= 2:
        return ", ".join(names)
    role = group.get("aux_role") or group.get("role") or "group"
    return f"{role} ({len(names)} tensors)" if names else str(role)


def _optimization_items(run: RunRecord) -> List[Item]:
    items: List[Item] = []
    groups = optimizer_groups(run)
    if groups is not None and groups["groups"]:
        decoupled = {g.get("decoupled_weight_decay") for g in groups["groups"]}
        items.append(("optimizer", "AdamW, decoupled weight decay" if decoupled == {True}
                      else f"AdamW (decoupled_weight_decay={sorted(map(str, decoupled))})"))
        for g in groups["groups"]:
            items.append(("", Code(f"{_group_name(g)}: lr={_fmt_value(g['initial_lr'])}, "
                                   f"wd={_fmt_value(g['weight_decay'])}")))
        items.append(("groups read from", Code(Path(groups["checkpoint"]).name)))
    else:
        sched = (run.summary.get("scheduler") or {}).get("groups") or []
        if sched:
            for g in sched:
                items.append(("", Code(f"{g.get('role')}: lr={_fmt_value(g.get('base_lr'))}")))
            items.append(("per-table weight decay", NOT_RECORDED + " (no training checkpoint)"))
        else:
            items.append(("per-table learning rates", NOT_RECORDED))
    kind = (run.summary.get("scheduler") or {}).get("kind")
    items.append(("schedule", Code(kind) if kind else NOT_RECORDED))
    items += _raw_items(run, ("warmup_steps", "max_steps", "min_lr", "min_lr_frac", "grad_clip",
                              "grad_clip_per_role", "m_phi_update_mode", "phi_mstep_max_matrix_norm"))
    ops = run.config_record.get("relational_operators")
    if isinstance(ops, Mapping) and ops.get("gradient_clipping"):
        items.append(("relational clipping", Code(str(ops["gradient_clipping"]))))
    return items


def _numerics_items(run: RunRecord) -> List[Item]:
    items = _raw_items(run, PRECISION_KEYS)
    state = run.provenance.get("deterministic_state") or {}
    if "matmul_allow_tf32" in state:
        items.append(("TF32 matmul allowed", str(state["matmul_allow_tf32"])))
    items += _raw_items(run, ("kl_max",))
    ops = run.config_record.get("relational_operators")
    if isinstance(ops, Mapping):
        for key in ("factorization_policy", "factorization_fallback"):
            items.append(("", Code(f"{key}={ops[key]}")) if key in ops else (key, NOT_RECORDED))
    items += _raw_items(run, FACTORIZATION_KEYS + EXPONENTIAL_KEYS)
    for key in ("device_name", "torch_version", "cuda_version"):
        if run.provenance.get(key):
            items.append((key.replace("_", " "), str(run.provenance[key])))
    return items


def _budget_items(run: RunRecord) -> List[Item]:
    s = run.summary
    seen, source = _tokens_seen(run)
    items: List[Item] = [("token presentations", f"{seen:,} ({source})" if seen is not None else NOT_RECORDED)]
    accepted, attempted = s.get("accepted_updates"), s.get("attempted_steps", s.get("n_steps"))
    items.append(("updates accepted / attempted",
                  f"{accepted:,} / {attempted:,}" if isinstance(accepted, int) and isinstance(attempted, int)
                  else NOT_RECORDED))
    items += _raw_items(run, ("batch_size", "max_seq_len", "grad_accum_steps"))
    hours, hours_source = device_hours(run)
    items.append(("device-hours", f"{hours:.2f} ({hours_source})" if hours is not None else NOT_RECORDED))
    return items


def _evaluation_items(run: RunRecord, rule: str) -> List[Item]:
    s = run.summary
    items: List[Item] = [("checkpoint-selection rule", Code(rule))]
    if s.get("evaluation_weights"):
        items.append(("headline weights", Code(str(s["evaluation_weights"]))))
    if isinstance(s.get("best_step"), int):
        items.append(("best-validation step", f"{s['best_step']:,}"))
    items += _raw_items(run, ("max_seq_len", "eval_stride", "beta_attention_prior"))
    for prefix, label in (("test_", "test"), ("final_val_", "validation")):
        scored, expected = s.get(f"{prefix}scored_targets"), s.get(f"{prefix}expected_targets")
        excluded = s.get(f"{prefix}excluded_targets")
        if isinstance(scored, int):
            items.append((f"{label} scored targets", f"{scored:,}"
                          + (f" of {expected:,} window positions ({excluded:,} overlapping positions "
                             f"excluded)" if isinstance(expected, int) and isinstance(excluded, int) else "")))
        else:
            items.append((f"{label} scored targets", NOT_RECORDED))
    return items


def methods_contract_items(run: RunRecord, rule: str) -> Dict[str, List[Item]]:
    r"""The eight reporting fields of ``tab:placeholder_methods_contract`` as ``(label, value)`` items."""
    return {
        "Source and configuration": _source_items(run),
        "Data and tokenizer": _data_items(run),
        "Architecture": _architecture_items(run),
        "Initialization": _initialization_items(run),
        "Optimization": _optimization_items(run),
        "Numerical realization": _numerics_items(run),
        "Training budget": _budget_items(run),
        "Evaluation": _evaluation_items(run, rule),
    }


def methods_contract_table(run: Optional[RunRecord], rule: str) -> Tuple[str, str, Dict[str, Any]]:
    r"""``tab:placeholder_methods_contract`` populated from ``run``'s saved artifacts."""
    fields = methods_contract_items(run, rule) if run is not None else {name: [] for name in METHODS_ROWS}
    body = [f"{name} & {render_items(fields[name], True) if run is not None else PENDING} \\\\"
            for name in METHODS_ROWS]
    md = [_md_row(["Reporting field", "Final-run value or artifact identifier"]), _md_row(["---"] * 2)]
    md += [_md_row([name, render_items(fields[name], False) if run is not None else PENDING])
           for name in METHODS_ROWS]
    head = f"Run: `{run.run_dir}`\n\n" if run is not None else "No final run declared.\n\n"
    record = {name: [(label, str(value)) for label, value in items] for name, items in fields.items()}
    return _tabular(METHODS_SPEC, METHODS_HEADER, body), head + "\n".join(md) + "\n", record
