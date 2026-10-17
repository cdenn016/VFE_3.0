r"""Assemble every manuscript table and figure of a declared experiment manifest into one directory.

:func:`assemble` resolves the manifest (:func:`vfe3.paper.runs.resolve_manifest`), writes

* ``tables/<name>.tex`` (the tabular environment with the manuscript's column layout) and
  ``tables/<name>.md`` (the same cells with sources) for ``main_results``, ``per_seed``,
  ``decision_rule`` (the registered seed-paired decision rule of the frame claim), ``relational_run``,
  ``resources`` and ``methods_contract``;
* ``figures/m3_learning_curves``, ``figures/s1_numerics`` and ``figures/s3_scaling`` (PNG + PDF);
* ``paper_artifacts.json`` (the resolved manifest, homogeneity reports, the selection record of every
  run, sources, skip reasons, and the notes of every run) and ``PAPER_ARTIFACTS.md`` (an index).

A table or figure that fails is recorded with its error and the others are still written; a figure
without data is recorded as skipped with its reason. Nothing is written into the run directories.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np

from vfe3.paper import figures as pf
from vfe3.paper import tables as pt
from vfe3.paper.runs import Manifest, resolve_manifest


def json_clean(value: Any) -> Any:
    r"""JSON-valid copy: numpy scalars and arrays to Python, non-finite floats to None, paths to str."""
    if isinstance(value, Mapping):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_clean(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _write(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def _index_markdown(report: Mapping[str, Any]) -> str:
    lines = [f"# Manuscript artifacts ({report['created']})", "",
             f"Selection rule: `{report['selection']}`. Tables are LaTeX tabular environments with the "
             "column layout of GL(K)_attention_v2.tex; the `.md` files carry the same cells with sources.",
             ""]
    if report["problems"]:
        lines += ["## Manifest problems", ""] + [f"- {p}" for p in report["problems"]] + [""]
    lines += ["## Arms", "", "| Arm | Runs | Seeds (model / data) | Homogeneous | Differing fields |",
              "|---|---|---|---|---|"]
    for arm in report["arms"]:
        h = arm["homogeneity"]
        seeds = ", ".join(f"{s} / {d}" for s, d in zip(h.get("seeds", []), h.get("data_seeds", [])))
        fields = ", ".join(v["field"] for v in h.get("violations", [])) or "–"
        lines.append(f"| {arm['label']} | {h.get('n_runs', 0)} | {seeds or '–'} | "
                     f"{'yes' if h.get('homogeneous') else 'no'} | {fields} |")
    if report["scaling"]:
        lines += ["", "## Scaling grids", ""]
        for axis, grid in report["scaling"].items():
            h = grid["homogeneity"]
            fields = ", ".join(v["field"] for v in h.get("violations", [])) or "none"
            lines.append(f"- `{axis}`: {h.get('n_runs', 0)} runs; varied on purpose: "
                         f"{', '.join(grid['vary']) or 'none'}; other differing fields: {fields}")
    lines += ["", "## Tables", ""]
    for name, entry in report["tables"].items():
        lines.append(f"- `{name}`: " + (f"`{entry['tex']}`, `{entry['md']}`" if "tex" in entry
                                         else f"FAILED: {entry.get('error')}"))
    lines += ["", "## Figures", ""]
    for name, entry in report["figures"].items():
        if entry.get("paths"):
            skipped = entry.get("skipped") or {}
            extra = ("; panels left out: " + "; ".join(f"{k}: {v}" for k, v in skipped.items())) if skipped else ""
            lines.append(f"- `{name}`: " + ", ".join(f"`{p}`" for p in entry["paths"]) + extra)
        else:
            lines.append(f"- `{name}`: {entry.get('status')}: {entry.get('reason')}")
        panels = entry.get("panels")
        if isinstance(panels, Mapping):
            for key, panel in panels.items():
                if isinstance(panel, Mapping) and panel.get("status") == "skipped":
                    lines.append(f"  - panel {key} skipped: {panel.get('reason')}")
    lines += ["", "## Runs", ""]
    for run in report["runs"]:
        dropped = ", ".join(run["dropped_config_fields"]) or "none"
        notes = "; ".join(run["notes"]) or "–"
        lines.append(f"- `{run['run_dir']}`: config {run['config_load']}; retired fields dropped: {dropped}; "
                     f"notes: {notes}")
    return "\n".join(lines) + "\n"


def assemble(
    config: Mapping[str, Any],

    *,
    out_dir: Optional[Any] = None,
    log:     Callable[[str], None] = print,
) -> Dict[str, Any]:
    r"""Resolve ``config`` (a manifest, see :mod:`vfe3.paper.runs`) and write every table and figure.

    ``config`` also takes ``out_dir`` (used when the argument is None), ``drop_unknown_config_fields``
    (default True) and ``read_checkpoints`` (default True: parameter counts, optimizer groups, map
    invariants and log-variance tables may be read memory-mapped from the saved weights when no
    record holds them). Returns the report written to ``paper_artifacts.json``.
    """
    started = time.perf_counter()
    target = Path(out_dir if out_dir is not None else config.get("out_dir") or "paper_artifacts")
    read_checkpoints = bool(config.get("read_checkpoints", True))
    manifest: Manifest = resolve_manifest(
        config, drop_unknown_config_fields=bool(config.get("drop_unknown_config_fields", True)))
    report: Dict[str, Any] = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "out_dir": str(target),
        "selection": manifest.selection,
        "problems": list(manifest.problems),
        "arms": [{"label": arm.label, "group": arm.group, "runs": [str(r.run_dir) for r in arm.runs],
                  "unresolved": arm.unresolved, "homogeneity": arm.homogeneity} for arm in manifest.arms],
        "evaluated_run": str(manifest.evaluated.run_dir) if manifest.evaluated else None,
        "final_run": str(manifest.final.run_dir) if manifest.final else None,
        "numerics_run": str(manifest.numerics.run_dir) if manifest.numerics else None,
        "scaling": {axis: {"runs": [str(r.run_dir) for r in grid.runs], "vary": list(grid.vary),
                           "homogeneity": grid.homogeneity} for axis, grid in manifest.scaling.items()},
        "tables": {}, "figures": {},
    }
    log(f"manifest: {len(manifest.arms)} arms, "
        f"{sum(len(a.runs) for a in manifest.arms)} runs, selection '{manifest.selection}'")
    contract_run = manifest.final or manifest.evaluated
    builders = {
        "main_results": lambda: pt.main_results_table(manifest, read_checkpoints=read_checkpoints),
        "per_seed": lambda: pt.per_seed_table(manifest),
        "decision_rule": lambda: pt.decision_rule_table(manifest),
        "relational_run": lambda: pt.relational_run_table(manifest.evaluated),
        "resources": lambda: pt.resources_table(manifest.evaluated, manifest.final,
                                                read_checkpoints=read_checkpoints),
        "methods_contract": lambda: pt.methods_contract_table(contract_run, manifest.selection),
    }
    for name, build in builders.items():
        try:
            latex, markdown, rows = build()
            report["tables"][name] = {
                "tex": _write(target / "tables" / f"{name}.tex", latex),
                "md": _write(target / "tables" / f"{name}.md", markdown),
                "rows": rows,
            }
            log(f"  table {name}: written")
        except Exception as exc:                               # noqa: BLE001 - one table never aborts
            report["tables"][name] = {"error": f"{type(exc).__name__}: {exc}"}
            log(f"  table {name}: FAILED ({type(exc).__name__}: {exc})")
    figure_dir = target / "figures"
    figure_builders = {
        "m3_learning_curves": lambda: pf.plot_learning_curves(manifest, figure_dir),
        "s1_numerics": (lambda: pf.plot_numerics(manifest.numerics, figure_dir)) if manifest.numerics else None,
        "s3_scaling": lambda: pf.plot_scaling(manifest, figure_dir),
    }
    for name, build in figure_builders.items():
        if build is None:
            report["figures"][name] = {"status": "skipped", "reason": "no numerics or evaluated run declared"}
            log(f"  figure {name}: skipped (no run declared)")
            continue
        try:
            entry = build()
            entry["status"] = "ok"
            report["figures"][name] = entry
            log(f"  figure {name}: written")
        except pf.FigureSkipped as exc:
            report["figures"][name] = {"status": "skipped", "reason": str(exc)}
            log(f"  figure {name}: skipped ({exc})")
        except Exception as exc:                               # noqa: BLE001
            report["figures"][name] = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
            log(f"  figure {name}: FAILED ({type(exc).__name__}: {exc})")
    runs = {}
    for arm in manifest.arms:
        for run in arm.runs:
            runs[str(run.run_dir)] = run
    for run in (manifest.evaluated, manifest.final, manifest.numerics,
                *[r for grid in manifest.scaling.values() for r in grid.runs]):
        if run is not None:
            runs[str(run.run_dir)] = run
    report["runs"] = [{"run_dir": key, "config_load": run.config_load,
                       "dropped_config_fields": sorted(run.dropped_config_fields),
                       "seed": run.seed, "data_seed": run.data_seed, "notes": list(run.notes)}
                      for key, run in runs.items()]
    report["seconds"] = time.perf_counter() - started
    target.mkdir(parents=True, exist_ok=True)
    (target / "paper_artifacts.json").write_text(json.dumps(json_clean(report), indent=1), encoding="utf-8")
    (target / "PAPER_ARTIFACTS.md").write_text(_index_markdown(report), encoding="utf-8")
    log(f"report: {target / 'paper_artifacts.json'}")
    return report
