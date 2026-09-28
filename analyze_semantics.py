r"""Click-to-run: semantic geometry of one trained run (:mod:`vfe3.semantics`).

Edit ``CONFIG`` below and run ``python analyze_semantics.py``, or pass a run directory as the only
command-line argument to override ``CONFIG["run_dir"]``. The run directory needs ``config.json`` and
the checkpoint (``best_model.pt``); ``provenance.json`` / ``summary.json`` are not required.
Figures (PNG 300 dpi + PDF), markdown tables, ``metrics.json``, ``candidate_ids.npy`` and
``REPORT.md`` are written to ``<run_dir>/semantics`` unless ``CONFIG["out_dir"]`` names another
directory.

``CONFIG["preset"]`` selects the budget (``vfe3.semantics.report.PRESETS``): ``"full"`` runs every
applicable view with UMAP and the full budgets (measured 1,644 s on the K160 run with 16 CPU
threads); ``"quick"`` is the lean per-seed battery of a multi-seed study (the headline decode view,
phi_log, mu_raw and xi_ref; no UMAP; 100 reranked group analogies; 4,000 geometry rows; a 16,384-token
contextual bank and a 65,536-token occurrence budget; token clouds kept), estimated at 12-14 minutes
on the same 16 CPU threads (see the ``report`` module docstring for the per-section numbers).
``OVERRIDES`` sets any preset-controlled option explicitly (for example ``{"umap": True}`` or
``{"views": ("decode_W_causal", "mu_raw")}``) on top of the preset.

``CONFIG["drop_unknown_config_fields"]`` (default ``True``) reopens a checkpoint whose config carries
fields this code line no longer declares (retired toggles, e.g. the K256 run's ``qk_frame_mode`` or
``emission_mode``): they are dropped at load and listed in the report's run notes. ``False`` refuses
such a checkpoint, as ``vfe3.config.config_from_serialized`` does.

``CONFIG["device"]``: ``"cuda"`` runs the contextual forward passes (and the vector battery, where
it has a GPU path) on the GPU; the frame chart stays on the CPU in float64. Schedule GPU analyses
between training runs.

The thread-count environment variables are set here, before numpy and torch are imported, so the
BLAS/OpenMP pools honor them; ``analyze_run`` additionally pins torch and the pools to the same count.
"""

import os
import sys

CONFIG = {
    "run_dir":             "D:/DATA_KEEP/30.14_wikitext-103_K160_n=256_6-layer",
    "out_dir":             None,          # None -> <run_dir>/semantics
    "checkpoint":          "best_model.pt",
    "preset":              "full",        # "full" | "quick" (vfe3.semantics.report.PRESETS)
    "device":              "cpu",         # "cpu" | "cuda": contextual forwards and the vector battery
    "threads":             None,          # None -> cpu_count - 2
    "sections":            ("embeddings", "sigma", "phi", "contextual"),
    "seed":                0,
    "min_count":           100,           # training-count floor for the candidate vocabulary
    "max_candidates":      20000,
    "frame_ref_count":     4096,          # reference frames for the barycenter / D0 of the phi chart
    "reference_position":  None,          # None -> min(max_seq_len, 256) // 2
    "occurrence_split":    "train",
    "occurrence_per_word": 4,
    "context_len":         48,
    # Reopen runs trained by an older code line: config fields this build no longer declares
    # (retired toggles) are dropped at load and listed in the report's run notes.
    "drop_unknown_config_fields": True,
}

# Preset-controlled options set explicitly (applied on top of the preset); empty keeps the preset's
# values. Keys: views, umap, geometry_max_points, frame_rerank_questions, contextual_tokens,
# occurrence_tokens (and any other analysis option).
OVERRIDES = {}

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
    options.update(OVERRIDES)
    options["threads"] = _THREADS
    print(f"\nvfe3 semantic analysis\n  run_dir: {run_dir}\n  preset:  {options['preset']}\n"
          f"  device:  {options['device']}\n  threads: {_THREADS}\n  sections: {', '.join(options['sections'])}"
          + (f"\n  overrides: {OVERRIDES}" if OVERRIDES else ""), flush=True)
    report = analyze_run(run_dir, out_dir=CONFIG["out_dir"], checkpoint=CONFIG["checkpoint"], options=options)
    print(f"\nreport: {report}")


if __name__ == "__main__":
    main(sys.argv)
