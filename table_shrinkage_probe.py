r"""Click-to-run: which per-type table carries the memorized component of a run? Evaluation only, no training.

Edit ``CONFIG`` below and run ``python table_shrinkage_probe.py``; there is no command-line configuration.

For each table in ``tables`` and each count threshold ``c``, the rows of types seen fewer than ``c`` times in the
training stream are reset to a neutral target (``mode="reset"``; frames to the identity, every other table to its
occurrence-weighted mean row), or shrunk toward it by :math:`n_v/(n_v + c)` (``mode="shrink"``). The run's fixed
training sample and the held-out splits are scored with ``vfe3.train.evaluate`` at the run's ``max_seq_len`` and
``eval_stride`` on the same windows for every arm (:mod:`vfe3.table_shrinkage`). The memorization score of an arm is
``delta CE(train sample) - delta CE(validation)``: large and positive where the moved rows held training-specific fit.
The report goes to ``<run_dir>/table_shrinkage_probe.json``; no existing artifact of the run is modified.

Cost: one scoring pass per split per arm, about (1 + n_tables + 1) x n_thresholds + 2 passes. With ``max_windows``
1000 per split this is minutes on the GPU; on the CPU a K = 256 run is slow, so prefer the GPU.
"""

import os

CONFIG = {
    "run_dir":                    "29.83_wikitext-103_K256_block_glk_mlp_s6",
    "checkpoint":                 "final_model.pt",
    "tables":                     ("phi_embed", "mu_embed", "sigma_log_embed", "decode_mu_embed",
                                   "decode_sigma_log_embed"),
    "thresholds":                 (10, 100, 1000),    # count threshold c (reset) or shrinkage constant c (shrink)
    "mode":                       "reset",            # "reset" | "shrink"
    "all_tables_arm":             True,               # also move every listed table at once, per threshold
    "splits":                     ("train_sample", "validation"),   # add "test" if wanted
    "max_windows":                1000,               # per split, evenly spaced and shared by every arm; None = all
    "device":                     "auto",             # "auto" -> cuda when available, else cpu
    "precision":                  "run",              # "run": the run's amp dtype on CUDA (fp32 on CPU); "fp32"
    "threads":                    None,               # CPU threads; None -> cpu_count - 2
    "batch_size":                 None,               # None -> the run's cfg.batch_size
    "allow_stream_mismatch":      False,
    "drop_unknown_config_fields": True,
    "cache_dir":                  None,               # None -> ~/.cache/tokenized_cache (the run's loaders)
    "out_name":                   "table_shrinkage_probe.json",
}

_THREADS = int(CONFIG["threads"] or max(1, (os.cpu_count() or 2) - 2))
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = str(_THREADS)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def main() -> None:
    from pathlib import Path

    import torch

    from vfe3.paper.runs import find_run_folder
    from vfe3.table_shrinkage import TableShrinkageOptions, run_table_shrinkage, summary_table

    run_dir = find_run_folder(CONFIG["run_dir"], base=Path(__file__).resolve().parent)
    torch.set_num_threads(_THREADS)
    options = TableShrinkageOptions(**{k: v for k, v in CONFIG.items() if k not in ("run_dir", "threads")})
    print(f"\nvfe3 table shrinkage probe\n  run_dir:    {run_dir}\n  checkpoint: {options.checkpoint}\n"
          f"  mode:       {options.mode}, thresholds {tuple(options.thresholds)}\n  device:     {options.device}",
          flush=True)
    report = run_table_shrinkage(run_dir, options=options, log=lambda message: print(message, flush=True))
    print("\n" + summary_table(report))
    print(f"\nbaseline restored exactly: {report['baseline_restored_matches']}\n{report['out_path']}")


if __name__ == "__main__":
    main()
