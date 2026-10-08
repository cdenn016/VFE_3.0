r"""Click-to-run: assemble the manuscript tables and figures of GL(K)_attention_v2 from run directories.

Edit ``CONFIG`` below and run ``python paper_artifacts.py``, or pass a JSON file with the same keys as
the only command-line argument. No model is built and nothing is written into the run directories;
everything goes to ``CONFIG["out_dir"]`` (see :mod:`vfe3.paper.assemble` for the layout):

* tables (LaTeX tabular with the manuscript's column layout, plus markdown with sources):
  ``tab:placeholder_main_results``, ``tab:placeholder_per_seed``, ``tab:relational_run``,
  ``tab:placeholder_resources``, ``tab:placeholder_methods_contract``;
* figures (PNG + PDF, 7.2 in wide at most, text >= 6 pt): ``fig:placeholder_learning_curves`` (M3),
  ``fig:placeholder_numerics`` (S1), ``fig:placeholder_scaling`` (S3).

Manifest keys (:mod:`vfe3.paper.runs`):

* ``arms``: ordered ``{row label: runs}``; ``runs`` is a list of run directories, globs or multi-seed
  roots, or ``{"runs": [...], "group": "<section heading>"}``. An empty list keeps the row as
  ``pending``. Labels are written verbatim into LaTeX.
* ``evaluated_run`` / ``final_run``: the two columns of the resources table; the final run (else the
  evaluated run) fills the methods contract, and the evaluated run fills ``tab:relational_run``.
* ``numerics_run``: the run of Figure S1 (None: the evaluated run).
* ``scaling``: ``{"K": runs, "depth": runs}`` (each optionally ``{"runs": [...], "vary": [...]}``).
* ``selection``: ``"best"`` (best recorded validation, default) or ``"final"``.

The run paths below are placeholders for the evaluated single-seed run. The existing runs are not
necessarily the final data of the paper: replace them with the final multi-seed run directories.
Configs written by older code lines are read with ``drop_unknown_config_fields`` (retired toggles are
dropped and listed in the report).
"""

import json
import os
import sys
from pathlib import Path

_RETRAINING = "Retraining arms at the evaluated configuration and token budget"
_BASELINES = "Conventional transformers: same tokenizer, context, stride, and budget"
_WORD_LEVEL = "Word-level protocol, every target scored"
_CONFOUNDS = "Confound controls for the per-type read-table comparison"

CONFIG = {
    "arms": {
        # Row labels of tab:placeholder_main_results; fill each list with the run dirs of every seed.
        "Evaluated: learned frames, decayed stored covariance": {
            "group": _RETRAINING, "runs": ["D:/DATA_KEEP/31.04_wikitext-103_K256_4-layer_noHM"]},
        "Learned frames, stored covariance without decay": {"group": _RETRAINING, "runs": []},
        "Identity frames, decayed stored covariance": {"group": _RETRAINING, "runs": []},
        "Identity frames, stored covariance without decay": {"group": _RETRAINING, "runs": []},
        "Identity frames, per-type additive table of equal size": {"group": _RETRAINING, "runs": []},
        "Attention residual removed": {"group": _RETRAINING, "runs": []},
        "Affine content chart": {"group": _RETRAINING, "runs": []},
        "First-order content chart": {"group": _RETRAINING, "runs": []},
        "Covariance passthrough instead of moment matching": {"group": _RETRAINING, "runs": []},
        "Frames frozen at their random initialization": {"group": _RETRAINING, "runs": []},
        "LayerNorm and MLP in reference coordinates": {"group": _RETRAINING, "runs": []},
        "Intertwiner-constrained nonlinear block": {"group": _RETRAINING, "runs": []},
        "MLP removed": {"group": _RETRAINING, "runs": []},
        "MLP replaced by Gaussian template routing": {"group": _RETRAINING, "runs": []},
        "Matched context-processing parameters": {"group": _BASELINES, "runs": []},
        "Matched measured compute": {"group": _BASELINES, "runs": []},
        "Per-type table of the frame table's size": {"group": _BASELINES, "runs": []},
        "Learned frames, ALiBi prior": {"group": _CONFOUNDS, "runs": []},
        "Covariance-blind routing (mean distance)": {"group": _CONFOUNDS, "runs": []},
        "Per-type read table, ALiBi (as run)": {"group": _CONFOUNDS, "runs": []},
        "Per-type read table, ALiBi, realization optimizer": {"group": _CONFOUNDS, "runs": []},
        "Per-type read-and-write table, ALiBi": {"group": _CONFOUNDS, "runs": []},
        "Per-type read table, ALiBi, shared post-norm": {"group": _CONFOUNDS, "runs": []},
        "Gaussian attention": {"group": _WORD_LEVEL, "runs": []},
        "Conventional transformer, matched": {"group": _WORD_LEVEL, "runs": []},
    },
    "evaluated_run":              "D:/DATA_KEEP/31.04_wikitext-103_K256_4-layer_noHM",
    "final_run":                  None,          # None -> the resources table's second column is pending
    "numerics_run":               None,          # None -> the evaluated run
    "scaling":                    {"K": [], "depth": []},
    "selection":                  "best",        # "best" (best recorded validation) or "final"
    "out_dir":                    None,          # None -> <this folder>/paper_artifacts
    "drop_unknown_config_fields": True,          # read configs of older code lines (2026-09-28 policy)
    "read_checkpoints":           True,          # read saved weights (mmap) when no record holds a value
    "threads":                    None,          # CPU threads; None -> cpu_count - 2
}

_THREADS = int(CONFIG["threads"] or max(1, (os.cpu_count() or 2) - 2))
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = str(_THREADS)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def main(argv: "list[str]") -> int:
    from vfe3.paper import assemble

    if len(argv) > 2:
        raise SystemExit(f"usage: {argv[0]} [manifest.json]")
    config = dict(CONFIG)
    if len(argv) == 2:
        config.update(json.loads(Path(argv[1]).read_text(encoding="utf-8")))
    out_dir = config.get("out_dir") or Path(__file__).resolve().parent / "paper_artifacts"
    report = assemble(config, out_dir=out_dir, log=lambda message: print(message, flush=True))
    failed = [name for name, entry in report["tables"].items() if "error" in entry]
    failed += [name for name, entry in report["figures"].items() if entry.get("status") == "failed"]
    print(f"\nindex: {Path(report['out_dir']) / 'PAPER_ARTIFACTS.md'}")
    if report["problems"]:
        print("manifest problems:\n  " + "\n  ".join(report["problems"]))
    if failed:
        print(f"FAILED: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
