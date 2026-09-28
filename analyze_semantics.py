r"""Click-to-run: semantic geometry of one trained run (:mod:`vfe3.semantics`).

Edit ``CONFIG`` below and run ``python analyze_semantics.py``, or pass a run directory as the only
command-line argument to override ``CONFIG["run_dir"]``. The run directory needs ``config.json`` and
the checkpoint (``best_model.pt``); ``provenance.json`` / ``summary.json`` are not required.
Figures (PNG 300 dpi + PDF), markdown tables, ``metrics.json`` and ``REPORT.md`` are written to
``<run_dir>/semantics`` unless ``CONFIG["out_dir"]`` names another directory.

The thread-count environment variables are set here, before numpy and torch are imported, so the
BLAS/OpenMP pools honor them; ``analyze_run`` additionally pins torch and the pools to the same count.
"""

import os
import sys

CONFIG = {
    "run_dir":             "D:/DATA_KEEP/30.14_wikitext-103_K160_n=256_6-layer",
    "out_dir":             None,          # None -> <run_dir>/semantics
    "checkpoint":          "best_model.pt",
    "device":              "cpu",         # contextual forward passes; the tables are read on the CPU
    "threads":             None,          # None -> cpu_count - 2
    "sections":            ("embeddings", "sigma", "phi", "contextual"),
    "views":               None,          # None -> every applicable view (see report.VIEW_NAMES)
    "umap":                True,
    "seed":                0,
    "min_count":           100,           # training-count floor for the candidate vocabulary
    "max_candidates":      20000,
    "geometry_max_points": 8000,
    "frame_ref_count":     4096,          # reference frames for the barycenter / D0 of the phi chart
    "reference_position":  None,          # None -> min(max_seq_len, 256) // 2
    "contextual_tokens":   32768,
    "occurrence_tokens":   131072,
    "occurrence_split":    "train",
    "occurrence_per_word": 4,
    "context_len":         48,
}

_THREADS = int(CONFIG["threads"] or max(1, (os.cpu_count() or 2) - 2))
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = str(_THREADS)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def main(argv: "list[str]") -> None:
    from vfe3.semantics.report import analyze_run

    if len(argv) > 2:
        raise SystemExit(f"usage: {argv[0]} [run_dir]")
    run_dir = argv[1] if len(argv) == 2 else CONFIG["run_dir"]
    options = {key: value for key, value in CONFIG.items() if key not in ("run_dir", "out_dir", "checkpoint")}
    options["threads"] = _THREADS
    print(f"\nvfe3 semantic analysis\n  run_dir: {run_dir}\n  device:  {options['device']}\n"
          f"  threads: {_THREADS}\n  sections: {', '.join(options['sections'])}", flush=True)
    report = analyze_run(run_dir, out_dir=CONFIG["out_dir"], checkpoint=CONFIG["checkpoint"], options=options)
    print(f"\nreport: {report}")


if __name__ == "__main__":
    main(sys.argv)
