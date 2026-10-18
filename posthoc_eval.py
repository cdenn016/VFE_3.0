r"""Click-to-run: post-hoc analyses of one trained run (:mod:`vfe3.posthoc`), no training.

Edit ``CONFIG`` below and run ``python posthoc_eval.py``; there is no command-line configuration.
The run directory needs ``final_model.pt``,
``best_model.pt``, ``summary.json`` and ``provenance.json``; ``checkpoints/step_*.pt`` (optimizer
state), ``metrics.csv`` (throughput) and ``map_snapshots/`` (map invariants over training) are used
when present.

Computed under the run's own evaluation protocol (``max_seq_len``, ``eval_stride``, ``batch_size`` from
the checkpoint config) on the run's identity-checked token streams:

* validation, test and a fixed training-split sample at the final and best-validation weights, each
  cross-checked against ``vfe3.train.evaluate`` and, on a full split, against ``summary.json``
  (``tab:relational_run``);
* the M5 frozen interventions, rows 1-12 (rows 9-12 remove the frames one role at a time), on the
  validation split with paired article-bootstrap intervals (4096-token blocks when the stream has no
  article headings);
* the beta-weighted KL-cap activation per layer and head (validation and test);
* per-head attention statistics on the validation split (S2 panel B);
* the relational-map invariants and head-by-block norms (S1 panel B, S2 panel A);
* the parameter, optimizer-state, throughput and latency record (``tab:placeholder_resources``);
* per layer the residual ratio ||m||/||mu|| and relative LayerNorm change quantiles with the mean
  self-weight, at every checkpoint on ``residual_split`` (v2:899, S1 panel A endpoints);
* the realized-frame routing covariance gap C_i vs c (U^T U)^{-1} (Frobenius and log det) and the
  hidden-temperature factor of the stored variances, at every checkpoint (v2:872);
* the word-normalized test perplexity after the BPE/word text audit (``tab:placeholder_calibration``;
  needs the full test split, and the word-level text from the local Hugging Face cache or
  ``word_stream_path``);
* the M4 routing-score decomposition from ``vfe3.posthoc.routing`` when that module is present
  (``fig:placeholder_routing_decomposition``), and CE by context length from the per-target NLL.

A section that is not defined for the run's route (no projected router, no block norm, no additive
residual, a non-WikiText or word-level dataset, ...) is recorded in ``posthoc.json`` with its reason.

Everything is written to ``<run_dir>/posthoc`` (``posthoc.json``, ``tables/*.md``, ``csv/``,
``figures/``, ``nll/``) unless ``CONFIG["out_dir"]`` names another directory. A full run takes a few
GPU-minutes per pass; on the CPU use ``max_windows`` for a quick subset (hours otherwise).

``drop_unknown_config_fields`` (default True here, False in the library) ignores config fields that
this code line no longer declares (retired toggles of older runs, e.g. the K256 run). The dropped
fields are recorded in ``posthoc.json`` next to the final-weight reproduction check, so a dropped
toggle that mattered shows as a reproduction mismatch.
"""

import os

CONFIG = {
    "run_dir":                    "D:/DATA_KEEP/31.04_wikitext-103_K256_4-layer_noHM",
    "out_dir":                    None,          # None -> <run_dir>/posthoc
    "device":                     "auto",        # "auto" -> cuda when available, else cpu
    "precision":                  "run",         # "run": the run's amp dtype on CUDA (fp32 on CPU); "fp32"
    "threads":                    None,          # CPU threads; None -> cpu_count - 2
    "checkpoints":                ("final_model.pt", "best_model.pt"),
    "splits":                     ("validation", "test", "train_sample"),
    "batch_size":                 None,          # None -> the run's cfg.batch_size
    "max_windows":                None,          # None -> full splits; an int -> evenly spaced subset
    "window_indices":             None,          # explicit validation/test windows (overrides max_windows)
    "cross_check":                True,          # also run vfe3.train.evaluate on the same windows
    "interventions":              None,          # None -> M5 rows 1-12; () -> skip; or a tuple of keys
    "intervention_checkpoint":    "final_model.pt",
    "intervention_split":         "validation",
    "klcap_splits":               ("validation", "test"),
    "attention_split":            "validation",
    "maps":                       True,
    "resources":                  True,
    "latency_batches":            20,
    "figures":                    True,
    "save_nll":                   True,
    "bootstrap_resamples":        10_000,
    "allow_stream_mismatch":      False,         # True only to evaluate on a stream the run did not record
    "drop_unknown_config_fields": True,          # the user's instruction for legacy runs (2026-09-28)
    "cache_dir":                  None,          # None -> ~/.cache/tokenized_cache (the run's loaders)
    "residual_split":             "validation",  # v2:899 residual ratios per checkpoint; None disables
    "covariance":                 True,          # v2:872 C_i gap and hidden temperature per checkpoint
    "covariance_split":           "validation",  # source of the occurrence-weighted (type, position) sample
    "calibration":                True,          # word-normalized test PPL (needs the full test split)
    "word_stream_path":           None,          # None -> pinned Salesforce/wikitext parquet in the HF cache
    "routing":                    True,          # M4 via vfe3.posthoc.routing (skipped with a reason if absent)
    "routing_split":              "validation",  # at the final weights
}

_THREADS = int(CONFIG["threads"] or max(1, (os.cpu_count() or 2) - 2))
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = str(_THREADS)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def main() -> None:
    import torch

    from vfe3.posthoc.report import PosthocOptions, run_posthoc

    from pathlib import Path

    from vfe3.paper.runs import find_run_folder

    # click-to-run: no command-line configuration. The entry resolves against this folder, else by its
    # folder name under vfe3_runs and D:/DATA_KEEP (audit 2026-10-17 item 52).
    run_dir = find_run_folder(CONFIG["run_dir"], base=Path(__file__).resolve().parent)
    torch.set_num_threads(_THREADS)
    fields = {key: value for key, value in CONFIG.items() if key not in ("run_dir", "out_dir", "threads")}
    options = PosthocOptions(**fields)
    print(f"\nvfe3 post-hoc analyses\n  run_dir: {run_dir}\n  device:  {options.device}\n"
          f"  threads: {_THREADS}\n  splits:  {', '.join(options.splits)}", flush=True)
    report = run_posthoc(run_dir, out_dir=CONFIG["out_dir"], options=options,
                         log=lambda message: print(message, flush=True))
    print(f"\nposthoc.json: {report['out_dir']}/posthoc.json")
    reproduction = report.get("reproduction") or {}
    print(f"final weights reproduced: {reproduction.get('final_weights_reproduced')}; "
          f"dropped config fields: {sorted(reproduction.get('dropped_config_fields') or {})}")


if __name__ == "__main__":
    main()
