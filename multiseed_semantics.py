r"""Click-to-run: across-seed aggregation of the semantic analysis (:mod:`vfe3.semantics.multiseed`).

Analyze every seed first with ``analyze_semantics.py`` (``CONFIG["preset"] = "quick"`` is the intended
per-seed battery), so that each run holds ``<run>/semantics/metrics.json`` (or the training report's
``<run>/figures/semantics/metrics.json``, or a pre-preset ``<run>/figures/metrics.json``) and, for
analyses written by this code line, ``candidate_ids.npy``.
Then edit ``CONFIG`` below and run ``python multiseed_semantics.py``, or pass the run directories
(two or more) or one parent folder on the command line to override ``CONFIG["runs"]``.

Outputs in ``CONFIG["out_dir"]`` (default ``<parent>/multiseed_semantics``): ``multiseed_semantics.json``,
``MULTISEED_SEMANTICS.md`` and ``figures/`` (headline metrics per seed with the across-seed mean and SD;
seed x seed stability heatmaps). The report states whether the seeds share one training data order
(``provenance.json["data_seed"]``): when they do, the across-seed SD measures initialization and
optimization only and is a lower bound on the variance over data orders.

The cross-seed stability rebuilds each seed's causal decode rows, frame chart and prior means from its
checkpoint (the frame chart costs about a minute per seed at K = 160 on 16 CPU threads);
``CONFIG["stability"] = False`` skips it. ``CONFIG["device"] = "cuda"`` runs the kNN and CKA on the
GPU; the frame chart stays on the CPU.
"""

import os
import sys

CONFIG = {
    # One parent folder whose analyzed children are the seeds (the per-seed run dirs train_vfe3.py
    # writes with NUM_RUNS/SEEDS), or a list of two or more run directories. Edit before running.
    "runs":                "D:/DATA_KEEP/<multi-seed run folder>",
    "out_dir":             None,          # None -> <parent>/multiseed_semantics
    "checkpoint":          None,          # None -> the checkpoint each analysis recorded
    "stability":           True,          # kNN Jaccard@10 and linear CKA between every pair of seeds
    "device":              "cpu",         # "cpu" | "cuda": stability kNN and CKA
    "threads":             None,          # None -> cpu_count - 2
    # Reopen runs trained by an older code line: config fields this build no longer declares
    # (retired toggles) are dropped at load and listed in the notes.
    "drop_unknown_config_fields": True,
}

_THREADS = int(CONFIG["threads"] or max(1, (os.cpu_count() or 2) - 2))
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = str(_THREADS)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def main(argv: "list[str]") -> None:
    from vfe3.semantics.multiseed import aggregate_seeds

    if len(argv) == 1:
        runs = CONFIG["runs"]
    elif len(argv) == 2:
        runs = argv[1]                                         # one parent folder
    else:
        runs = argv[1:]                                        # two or more run directories
    shown = runs if isinstance(runs, str) else "\n    " + "\n    ".join(str(r) for r in runs)
    print(f"\nvfe3 multi-seed semantic aggregation\n  runs: {shown}\n  stability: {CONFIG['stability']}\n"
          f"  device: {CONFIG['device']}\n  threads: {_THREADS}", flush=True)
    report = aggregate_seeds(runs, out_dir=CONFIG["out_dir"], checkpoint=CONFIG["checkpoint"],
                             stability=bool(CONFIG["stability"]), device=CONFIG["device"], threads=_THREADS,
                             drop_unknown_config_fields=bool(CONFIG["drop_unknown_config_fields"]))
    print(f"\nreport: {report}")


if __name__ == "__main__":
    main(sys.argv)
