r"""Click-to-run: score the uniform average of two or more checkpoints of one run (audit 2026-10-20 item 8), no training.

Edit ``CONFIG`` below and run ``python average_checkpoints.py``; there is no command-line configuration.

The listed bundles (paths relative to the run directory: ``best_model.pt``, ``final_model.pt`` or
``checkpoints/step_*.pt``) must share the run's config and every tensor's shape and dtype. Every floating tensor of
the state dict is averaged uniformly (float64 accumulation); non-floating buffers come from the last listed
checkpoint. The average is scored with ``vfe3.train.evaluate`` on the run's identity-checked validation and test
streams at the run's ``max_seq_len`` and ``eval_stride`` (:mod:`vfe3.checkpoint_average`), and the report is written
to ``<run_dir>/averaged_eval.json``; no existing artifact of the run is modified. ``evaluate_inputs`` also scores
each listed checkpoint on the same windows, which a ``max_windows`` subset needs for a paired comparison.

On the CPU each batched LU-family call runs under one thread (:mod:`vfe3.cpu_linalg`); a full split of a K = 160 run
takes hours there, so use ``max_windows`` for a quick subset or run on the GPU.
"""

import os

CONFIG = {
    "run_dir":                    "30.37_wikitext-103_K160_5-epochs_6-layer",
    "checkpoints":                ("best_model.pt", "final_model.pt"),   # or ("checkpoints/step_135000.pt", ...)
    "splits":                     ("validation", "test"),
    "device":                     "auto",        # "auto" -> cuda when available, else cpu
    "precision":                  "run",         # "run": the run's amp dtype on CUDA (fp32 on CPU); "fp32"
    "threads":                    None,          # CPU threads; None -> cpu_count - 2
    "batch_size":                 None,          # None -> the run's cfg.batch_size
    "max_windows":                None,          # None -> full splits; an int -> evenly spaced subset
    "evaluate_inputs":            False,         # True -> also score each listed checkpoint on the same windows
    "allow_stream_mismatch":      False,         # True only to evaluate on a stream the run did not record
    "drop_unknown_config_fields": True,          # ignore retired config fields of older runs
    "cache_dir":                  None,          # None -> ~/.cache/tokenized_cache (the run's loaders)
    "out_name":                   "averaged_eval.json",
}

_THREADS = int(CONFIG["threads"] or max(1, (os.cpu_count() or 2) - 2))
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = str(_THREADS)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def main() -> None:
    from pathlib import Path

    import torch

    from vfe3.checkpoint_average import CheckpointAverageOptions, run_checkpoint_average
    from vfe3.paper.runs import find_run_folder

    run_dir = find_run_folder(CONFIG["run_dir"], base=Path(__file__).resolve().parent)
    torch.set_num_threads(_THREADS)
    fields = {key: value for key, value in CONFIG.items() if key not in ("run_dir", "threads")}
    options = CheckpointAverageOptions(**fields)
    print(f"\nvfe3 checkpoint average\n  run_dir:     {run_dir}\n  checkpoints: {', '.join(options.checkpoints)}\n"
          f"  device:      {options.device}\n  threads:     {_THREADS}", flush=True)
    report = run_checkpoint_average(run_dir, options=options, log=lambda message: print(message, flush=True))
    for split, result in report["averaged"].items():
        print(f"averaged {split}: CE {result['ce']:.4f} | PPL {result['ppl']:.3f}")
    print(f"\n{report['out_path']}")


if __name__ == "__main__":
    main()
