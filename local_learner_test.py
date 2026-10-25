r"""Click-to-run local-learner test (design: docs/research/2026-10-21-local-learner-test/design.md).

Edit CONFIG and run. Stage "timing" measures one chunk per arm; "A" is the synthetic identifiability check;
"B" is WikiText-103 bytes. Results append to ``<out_dir>/results_<stage>.jsonl`` and completed runs are
skipped on relaunch. The seeds of one (arm, d) train as one stacked batch, so each kernel launch serves all
of them; stacked runs never exchange information. With ``device = "cuda"`` run it under the CUDA interpreter (``$CUDA_PYTHON``,
``C:/anaconda/python.exe``); the n-gram references and synthetic sampling stay on the CPU.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

CONFIG = {
    
    "stage":            "S",                    # "timing" | "A" | "B" | "S" (scaling follow-up to B)
    "device":           "cuda",                 # "cuda" | "cpu"
    "threads":          12,                      # CPU intra-op threads (machine stays usable)
    "out_dir":          "docs/research/2026-10-21-local-learner-test/results",
    "chunk":            16,                     # M-step / truncated-BPTT length (smoothing windows must divide it)
    "streams":          512,                    # parallel training streams (512 * 16 = 8192 tokens per update)
    "arms":             ["local_w1", "local_w4", "local_w16", "blind", "backprop"],   # stage B: local_w<window>[@schedule] | blind | backprop
    "A_arms":           ["local_w1", "local_w4", "local_w16", "blind", "backprop",      # stage A arms (the schedule comparison)
                         "local_w16@a03", "local_w16@floor05", "local_w16@floor20", "local_w16@reset",
                         "local_w16_cav@floor20"],                                    # cavity node statistics
    "local_schedule":   "floor20",                 # schedule of local arms without "@": a key of local_schedules
    # Stepwise-EM averaging gamma_k = max((k+1)^-alpha, floor), k restarted every reset updates ("pass" = one
    # pass over the stage A training set; None = never). rm06 is the original Cappe-Moulines schedule.
    "local_schedules":  {"rm06":    {"alpha": 0.6, "floor": 0.0,  "reset": None},
                         "a03":     {"alpha": 0.3, "floor": 0.0,  "reset": None},     # forgets faster
                         "floor05": {"alpha": 0.6, "floor": 0.05, "reset": None},     # memory of about 20 updates
                         "floor20": {"alpha": 0.6, "floor": 0.2,  "reset": None},     # memory of about 5 updates
                         "reset":   {"alpha": 0.6, "floor": 0.0,  "reset": "pass"}},  # averages restart each pass
    "backprop_lr":      1e-2,                   # chosen on stage A only
    "ridge":            1e-6,
    "q_floor":          1e-4,
    
    # stage A (synthetic, known theta*)
    "A_d":              8,
    "A_train_len":      2000,                   # per stream: 512 * 2000 = 1.02M tokens
    "A_epochs":         8,                      # passes over the synthetic training set
    "A_val_streams":    64,
    "A_val_len":        1000,
    "A_seeds":          [0],
    
    # stage S (scaling follow-up on the stage B data; results_S.jsonl). Each run sets its own chunk and stream
    # count with chunk * streams = 8192 tokens per update, as in stage B, so every run makes 610 updates; "stack"
    # caps how many seeds train in one batch (backprop at d = 64 keeps one seed per batch for GPU memory).
    # Part 1 adds d = 64 to the stage B grid; part 2 lengthens the smoothing window and the truncated-backprop
    # horizon to 64 steps at d = 32 and 64.
    "S_runs":           [{"arm": "local_w16", "d": 64, "chunk": 16, "streams": 512, "stack": 3},
                         {"arm": "blind",     "d": 64, "chunk": 16, "streams": 512, "stack": 3},
                         {"arm": "backprop",  "d": 64, "chunk": 16, "streams": 512, "stack": 1},
                         {"arm": "local_w64", "d": 32, "chunk": 64, "streams": 128, "stack": 3},
                         {"arm": "backprop",  "d": 32, "chunk": 64, "streams": 128, "stack": 3},
                         {"arm": "local_w64", "d": 64, "chunk": 64, "streams": 128, "stack": 3},
                         {"arm": "backprop",  "d": 64, "chunk": 64, "streams": 128, "stack": 1},
                         # Node statistics at the EP cavity of the token's posterior (local_w<window>_cav).
                         {"arm": "local_w16_cav", "d": 32, "chunk": 16, "streams": 512, "stack": 3},
                         {"arm": "local_w16_cav", "d": 64, "chunk": 16, "streams": 512, "stack": 3}],

    # stage B (WikiText-103 bytes)
    "B_dims":           [8, 16, 32],
    "B_seeds":          [0, 1, 2],
    "B_train_bytes":    5_000_000,
    "B_val_bytes":      200_000,
    "B_val_streams":    64,
    "ngram_k_grid":     [1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0],
    
    # particle-filter approximation gap
    "pf_streams":       8,
    "pf_len":           2500,                   # 8 * 2500 = 20k scored-region bytes
    "pf_particles":     4096,
}

os.environ.setdefault("OMP_NUM_THREADS", str(CONFIG["threads"]))
os.environ.setdefault("MKL_NUM_THREADS", str(CONFIG["threads"]))

import numpy as np   # noqa: E402
import torch         # noqa: E402

from vfe3.local_learner import (TreeSensor, evaluate_filter, init_runs, ngram_nll,   # noqa: E402
                                particle_filter_nll, sample_sequences, train_backprop, train_local)
from vfe3.local_learner.core import Params   # noqa: E402

LN2 = math.log(2.0)
DEVICE = torch.device("cpu")                    # set from CONFIG["device"] in main()
CACHE = Path.home() / ".cache" / "tokenized_cache"


def load_bytes(split: str, n: int) -> np.ndarray:
    r"""First n bytes of the cached WikiText-103 byte stream (cache-only; fails if absent)."""
    path = CACHE / f"wikitext-103_{split}_bytes_tokens.bin"
    meta = json.loads(path.with_suffix(".bin.meta.json").read_text())
    arr = np.memmap(path, dtype=np.dtype(meta["dtype"]), mode="r")[:n]
    if len(arr) < n or arr.min() < 0 or arr.max() > 255:
        raise ValueError(f"{path}: expected {n} byte tokens in [0, 255]")
    return np.asarray(arr, dtype=np.int64)


def as_streams(x: np.ndarray, S: int) -> torch.Tensor:
    L = len(x) // S
    return torch.from_numpy(x[:S * L].reshape(S, L).copy())


def train_arm(arm: str, prm0: Params, train: torch.Tensor, tree: TreeSensor, log, chunk: int | None = None) -> Params:
    c = CONFIG
    chunk = c["chunk"] if chunk is None else chunk
    if arm == "backprop":
        return train_backprop(prm0, train, tree, chunk=chunk, lr=c["backprop_lr"], log=log)
    base, _, name = arm.partition("@")
    sensor_stats = "sequential"
    if base.endswith("_cav"):                   # node statistics at the EP cavity of the token's posterior
        base, sensor_stats = base[:-4], "cavity"
    sched = c["local_schedules"][name or c["local_schedule"]]
    reset = sched["reset"]
    if reset == "pass":
        if c["stage"] != "A":
            raise ValueError("the 'pass' reset needs repeated passes (stage A)")
        reset = max(1, c["A_train_len"] // c["chunk"])
    kw = dict(chunk=chunk, ridge=c["ridge"], q_floor=c["q_floor"], gamma_exponent=sched["alpha"],
              gamma_floor=sched["floor"], reset_every=reset, sensor_stats=sensor_stats, log=log)
    if base == "blind":
        return train_local(prm0, train, tree, window=1, blind=True, **kw)
    if base.startswith("local_w"):
        return train_local(prm0, train, tree, window=int(base[7:]), **kw)
    raise ValueError(arm)


def scores(prm: Params, val: torch.Tensor, pf: torch.Tensor, tree: TreeSensor, seed: int) -> list[dict]:
    r"""Validation bits per byte and the particle-filter gap, one dict per stacked run."""
    nll, n = evaluate_filter(prm, val, tree)
    a_nll, a_n = evaluate_filter(prm, pf, tree)
    p_nll, p_n = particle_filter_nll(prm, pf, tree, CONFIG["pf_particles"], seed)
    return [{"val_bits": float(nll[r]) / n / LN2, "pf_subset_filter_bits": float(a_nll[r]) / a_n / LN2,
             "pf_subset_exact_bits": float(p_nll[r]) / p_n / LN2, "n_val": n, "n_pf": p_n} for r in range(prm.R)]


def done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    return {(r["arm"], r["d"], r["seed"]) for r in map(json.loads, path.read_text().splitlines()) if "arm" in r}


def append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def run_timing(tree: TreeSensor) -> None:
    chunks = 20
    seeds = CONFIG["B_seeds"]
    x = load_bytes("train", CONFIG["streams"] * (chunks * CONFIG["chunk"] + 1))
    train = as_streams(x, CONFIG["streams"]).to(DEVICE)
    for d in CONFIG["B_dims"]:
        for arm in CONFIG["arms"]:
            train_arm(arm, init_runs(d, tree.V, seeds, DEVICE), train[:, :CONFIG["chunk"] + 1], tree, None)   # warm-up
            sync()
            t0 = time.perf_counter()
            train_arm(arm, init_runs(d, tree.V, seeds, DEVICE), train, tree, None)
            sync()
            sec = (time.perf_counter() - t0) / chunks
            print(f"d={d:3d} {arm:11s} {sec:7.3f} s per chunk of {CONFIG['streams'] * CONFIG['chunk']} tokens "
                  f"for {len(seeds)} stacked seeds", flush=True)


def run_scaling(tree: TreeSensor) -> None:
    r"""Stage S: the runs of ``S_runs`` on the stage B data, each with its own chunk, stream count and stack size."""
    c = CONFIG
    out = Path(c["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    path = out / "results_S.jsonl"
    done = done_keys(path)
    xtrain = load_bytes("train", c["B_train_bytes"])
    xval = load_bytes("validation", c["B_val_bytes"])
    val = as_streams(xval, c["B_val_streams"]).to(DEVICE)
    pf = as_streams(xval[:c["pf_streams"] * c["pf_len"]], c["pf_streams"]).to(DEVICE)
    for spec in c["S_runs"]:
        arm, d, chunk, streams = spec["arm"], spec["d"], spec["chunk"], spec["streams"]
        label = f"{arm}_c{chunk}"
        train = as_streams(xtrain, streams).to(DEVICE)
        todo = [s for s in c["B_seeds"] if (label, d, s) not in done]
        for i in range(0, len(todo), spec["stack"]):
            group = todo[i:i + spec["stack"]]
            t0 = time.perf_counter()
            history: list[dict] = []
            prm = train_arm(arm, init_runs(d, tree.V, group, DEVICE), train, tree, history.append, chunk=chunk)
            sync()
            sec = time.perf_counter() - t0
            spectral = torch.linalg.matrix_norm(prm.G, 2).tolist()
            for r, (seed, sc) in enumerate(zip(group, scores(prm, val, pf, tree, 1000 * d + group[0]))):
                append(path, {"stage": "S", "arm": label, "base_arm": arm, "d": d, "seed": seed, "chunk": chunk,
                              "streams": streams, "train_sec": sec, "stacked_runs": len(group),
                              "train_tokens": int(train.shape[0] * (train.shape[1] - 1)),
                              "updates": history[-1]["chunk"] if history else None,
                              "final_train_bits": history[-1]["train_bits"][r] if history else None,
                              "G_spectral": spectral[r], **sc})
            del prm
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()


def run_stage(stage: str, tree: TreeSensor) -> None:
    c = CONFIG
    out = Path(c["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"results_{stage}.jsonl"
    done = done_keys(path)
    if stage == "A":
        g = torch.Generator().manual_seed(1234)
        d = c["A_d"]
        q, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
        truth = Params(G=0.9 * q[None], E=0.5 * torch.randn(1, tree.V, d, generator=g, dtype=torch.float64),
                       Q=0.2 * torch.eye(d, dtype=torch.float64)[None],
                       A=torch.randn(1, tree.V - 1, d, generator=g, dtype=torch.float64),
                       b=0.5 * torch.randn(1, tree.V - 1, generator=g, dtype=torch.float64))
        train_cpu = sample_sequences(truth, tree, c["streams"], c["A_train_len"], seed=1)
        val_cpu = sample_sequences(truth, tree, c["A_val_streams"], c["A_val_len"], seed=2)
        train = train_cpu.repeat(1, c["A_epochs"]).to(DEVICE)
        val = val_cpu.to(DEVICE)
        pf = sample_sequences(truth, tree, c["pf_streams"], c["pf_len"], seed=3).to(DEVICE)
        truth = truth.to(DEVICE)
        dims, seeds = [d], c["A_seeds"]
        if ("truth", d, -1) not in done:
            append(path, {"stage": "A", "arm": "truth", "d": d, "seed": -1, **scores(truth, val, pf, tree, 0)[0]})
        uni, n = ngram_nll(train_cpu.reshape(-1).numpy(), val_cpu.numpy(), tree.V, 1, 1.0)
        if ("unigram", 0, 0) not in done:
            append(path, {"stage": "A", "arm": "unigram", "d": 0, "seed": 0, "val_bits": uni / n / LN2})
    else:
        xtrain = load_bytes("train", c["B_train_bytes"])
        xval = load_bytes("validation", c["B_val_bytes"])
        val_cpu = as_streams(xval, c["B_val_streams"])
        train = as_streams(xtrain, c["streams"]).to(DEVICE)
        val = val_cpu.to(DEVICE)
        pf = as_streams(xval[:c["pf_streams"] * c["pf_len"]], c["pf_streams"]).to(DEVICE)
        for order in (1, 2, 3):
            key = (f"{order}-gram", 0, 0)
            if key in done:
                continue
            best = min((ngram_nll(xtrain, val_cpu.numpy(), tree.V, order, k), k) for k in c["ngram_k_grid"])
            (nll, n), k = best
            append(path, {"stage": "B", "arm": key[0], "d": 0, "seed": 0, "val_bits": nll / n / LN2, "k": k})
        dims, seeds = c["B_dims"], c["B_seeds"]
    for d in dims:
        for arm in (c["A_arms"] if stage == "A" else c["arms"]):
            todo = [s for s in seeds if (arm, d, s) not in done]
            if not todo:
                continue
            t0 = time.perf_counter()
            history: list[dict] = []
            prm = train_arm(arm, init_runs(d, tree.V, todo, DEVICE), train, tree, history.append)   # seeds stacked
            sync()
            sec = time.perf_counter() - t0
            spectral = torch.linalg.matrix_norm(prm.G, 2).tolist()
            for r, (seed, sc) in enumerate(zip(todo, scores(prm, val, pf, tree, 1000 * d + todo[0]))):
                append(path, {"stage": stage, "arm": arm, "d": d, "seed": seed, "train_sec": sec,
                              "stacked_runs": len(todo), "train_tokens": int(train.shape[0] * (train.shape[1] - 1)),
                              "final_train_bits": history[-1]["train_bits"][r] if history else None,
                              "G_spectral": spectral[r], **sc})


def sync() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    global DEVICE
    torch.set_num_threads(CONFIG["threads"])
    torch.set_default_dtype(torch.float64)
    DEVICE = torch.device(CONFIG["device"])
    if DEVICE.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' but CUDA is unavailable: run under $CUDA_PYTHON (C:/anaconda/python.exe)")
    tree = TreeSensor(8, DEVICE)
    if CONFIG["stage"] == "timing":
        run_timing(tree)
    elif CONFIG["stage"] == "S":
        run_scaling(tree)
    else:
        run_stage(CONFIG["stage"], tree)


if __name__ == "__main__":
    main()
