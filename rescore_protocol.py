r"""Click-to-run: score checkpoints of several runs under ONE shared evaluation protocol, no training.

Edit ``CONFIG`` below and run ``python rescore_protocol.py``; there is no command-line configuration.

Each run's recorded perplexity uses its own window length N and stride s, so a scored target sees at least N - s
tokens of context (96 for the 30.37 run at N=128, s=32; 192 for the 29.83 run at N=256, s=64). This script scores
every listed checkpoint on the same validation and test windows at the chosen (N, s) with ``vfe3.train.evaluate``
(:mod:`vfe3.protocol_rescore`), so the remaining differences are the models'. Every protocol scores each target of a
split once, so the target set is identical; only the context differs.

Default: the 30.37 protocol (N=128, s=32), scoring 29.83's final weights and 30.37's best and final weights. Scoring
30.37 under its own protocol should reproduce its recorded numbers, which checks the setup. The report goes to
``vfe3_runs/protocol_rescores/<out_name>``; no run artifact is modified.
"""

import os

CONFIG = {
    "entries": (                                     # (run directory under vfe3_runs, checkpoint)
        ("29.83_wikitext-103_K256_block_glk_mlp_s6", "final_model.pt"),
        ("30.37_wikitext-103_K160_5-epochs_6-layer", "best_model.pt"),
        ("30.37_wikitext-103_K160_5-epochs_6-layer", "final_model.pt"),
    ),
    "seq_len":                    128,               # N, the window length of the shared protocol
    "eval_stride":                32,                # s, targets scored per window (min context N - s)
    "splits":                     ("validation", "test"),
    "device":                     "auto",            # "auto" -> cuda when available, else cpu
    "precision":                  "run",             # "run": each run's amp dtype on CUDA (fp32 on CPU); "fp32"
    "threads":                    None,              # CPU threads; None -> cpu_count - 2
    "batch_size":                 None,              # None -> each run's cfg.batch_size
    "max_windows":                None,              # None -> full splits; an int -> evenly spaced subset
    "allow_stream_mismatch":      False,
    "drop_unknown_config_fields": True,
    "cache_dir":                  None,              # None -> ~/.cache/tokenized_cache (the runs' loaders)
    "out_name":                   "rescore_N128_s32.json",
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
    from vfe3.protocol_rescore import ProtocolRescoreOptions, comparison_table, run_protocol_rescore

    base = Path(__file__).resolve().parent
    torch.set_num_threads(_THREADS)
    entries = tuple((str(find_run_folder(run, base=base)), checkpoint) for run, checkpoint in CONFIG["entries"])
    options = ProtocolRescoreOptions(entries=entries, **{k: v for k, v in CONFIG.items()
                                                         if k not in ("entries", "threads", "out_name")})
    out_path = base / "vfe3_runs" / "protocol_rescores" / CONFIG["out_name"]
    print(f"\nvfe3 protocol re-score at N={options.seq_len}, stride={options.eval_stride} "
          f"(min context {options.seq_len - options.eval_stride})", flush=True)
    report = run_protocol_rescore(out_path, options=options, log=lambda message: print(message, flush=True))
    print("\n" + comparison_table(report))
    for entry in report["results"]:
        if entry["trained_protocol"] == {"seq_len": options.seq_len, "eval_stride": options.eval_stride}:
            print(f"  {entry['run']} / {entry['checkpoint']} was scored under its own protocol; recorded: "
                  f"{entry['recorded']}")
    print(f"\n{report['out_path']}")


if __name__ == "__main__":
    main()
