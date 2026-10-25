r"""Click-to-run local-learner test (design: docs/research/2026-10-21-local-learner-test/design.md).

Edit CONFIG and run. Stage "timing" measures one chunk per arm; "A" is the synthetic identifiability check;
"B" is WikiText-103 bytes. Results append to ``<out_dir>/results_<stage>.jsonl``; each record stores its effective
arm settings, data protocol and source hash, and a completed run is skipped on relaunch only when these match the
current ones (a mismatch stops with the differing fields; records without them are reused with a warning). Stage W
records carry the same identity (both arms, its pass and point counts, its data protocol, and a source hash that
includes its runner) and the stage is skipped once its final row exists for every seed. The
seeds of one (arm, d) train as one stacked batch, so each kernel launch serves all of them; stacked runs never
exchange information. With ``device = "cuda"`` run it under the CUDA interpreter (``$CUDA_PYTHON``,
``C:/anaconda/python.exe``); the n-gram references and synthetic sampling stay on the CPU.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

CONFIG = {
    
    "stage":            "C",                    # "timing" | "A" | "B" | "S" (scaling) | "W" (warm start) | "X" (acceleration) | "C" (exp links)
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

    # stage W (diagnostic on the stage B data; results_W.jsonl). warm: backpropagation trains one pass, then the
    # local arm continues from its parameters for one pass; if validation NLL drifts up, approximate EM's fixed
    # point differs from the predictive optimum, and if it holds, the gap is optimization. passes: the local arm
    # and backpropagation from the stage B initialization for W_passes passes. Validation is scored at W_points
    # evenly spaced points per pass; local_arm uses local_schedule. Training restarts the filter and the
    # stepwise-EM averages at each scoring point (a few updates of averaging memory under floor20).
    "W_d":              32,
    "W_local_arm":      "local_w16",
    "W_passes":         3,
    "W_points":         5,                      # scoring points per pass

    # stage X (accelerated local learners on the stage B data; results_X.jsonl). Each arm trains X_passes passes
    # from the stage B initialization at d = X_d, scored at W_points evenly spaced points per pass, as in stage W.
    # Arm suffixes: _px = parameter-expanded probit M-step (each node also fits its utility variance, then
    # rescales), _orNN = overrelaxed M-step with omega = NN/10 (or15 = 1.5). Backpropagation's one-pass score at
    # d = 32 is in results_W.jsonl (3.270 bits per byte).
    "X_d":              32,
    "X_passes":         1,
    "X_arms":           ["local_w16", "local_w16_or15", "local_w16_or18", "local_w16_px", "local_w16_px_or15"],

    # stage C (group-valued links on the stage B data; results_C.jsonl). The transition is G = exp(L): the local
    # arm updates L by the natural-gradient step D exp_L[dL] = link_step (G_EM - G) (equations (4)-(5) of the
    # October 20 local-learning note) and backpropagation learns L through matrix_exp. The matrix-chart arms rerun
    # under the same code as the reference. One pass from the stage B initialization (G = 0.5 I, L = log(0.5) I),
    # three stacked seeds, scored at the end with the particle-filter gap, as in stage B.
    "C_dims":           [32],
    "C_arms":           ["local_w16", "local_w16_exp", "backprop", "backprop_exp"],
    "link_step":        1.0,                    # natural-gradient step on L, in (0, 1]

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


def arm_settings(arm: str, chunk: int | None = None) -> dict:
    r"""The effective training settings of an arm under CONFIG: the learning rule and every keyword its trainer
    receives (schedules resolved to their values). ``train_arm`` trains from exactly this dict, and each result
    record stores it, so a completed run is reused only under the same settings."""
    c = CONFIG
    chunk = c["chunk"] if chunk is None else chunk
    if arm in ("backprop", "backprop_exp"):
        out = {"rule": "backprop", "chunk": chunk, "lr": c["backprop_lr"]}
        return {**out, "link_chart": "exp"} if arm == "backprop_exp" else out
    base, _, name = arm.partition("@")
    sensor_stats, expand_sensor, overrelax, link_chart = "sequential", False, 1.0, "matrix"
    while True:                                 # suffixes, in any order: _cav, _px, _orNN, _exp
        head, _, tail = base.rpartition("_")
        if tail == "exp":                       # group-valued link G = exp(L), natural-gradient step on L
            link_chart = "exp"
        elif tail == "cav":                     # node statistics at the EP cavity of the token's posterior
            sensor_stats = "cavity"
        elif tail == "px":                      # parameter-expanded probit M-step
            expand_sensor = True
        elif tail.startswith("or") and tail[2:].isdigit():
            overrelax = int(tail[2:]) / 10      # overrelaxed M-step, omega = NN/10
        else:
            break
        base = head
    sched = c["local_schedules"][name or c["local_schedule"]]
    reset = sched["reset"]
    if reset == "pass":
        if c["stage"] != "A":
            raise ValueError("the 'pass' reset needs repeated passes (stage A)")
        reset = max(1, c["A_train_len"] // c["chunk"])
    if base == "blind":
        window, blind = 1, True
    elif base.startswith("local_w"):
        window, blind = int(base[7:]), False
    else:
        raise ValueError(arm)
    out = {"rule": "local", "chunk": chunk, "window": window, "blind": blind, "ridge": c["ridge"],
           "q_floor": c["q_floor"], "gamma_exponent": sched["alpha"], "gamma_floor": sched["floor"],
           "reset_every": reset, "sensor_stats": sensor_stats, "expand_sensor": expand_sensor,
           "overrelax": overrelax}
    # The chart keys appear only off the default, so earlier records keep their stored settings.
    return {**out, "link_chart": "exp", "link_step": c["link_step"]} if link_chart == "exp" else out


def train_arm(arm: str, prm0: Params, train: torch.Tensor, tree: TreeSensor, log, chunk: int | None = None) -> Params:
    kw = arm_settings(arm, chunk)
    if kw.pop("rule") == "backprop":
        return train_backprop(prm0, train, tree, log=log, **kw)
    return train_local(prm0, train, tree, log=log, **kw)


def protocol(stage: str, streams: int | None = None) -> dict:
    r"""Data and evaluation identity of a stage: corpus, budgets, stream layout, validation and particle-filter
    slices. ``streams`` overrides the training stream count (stage S sets it per run)."""
    c = CONFIG
    common = {"stage": stage, "streams": c["streams"] if streams is None else streams, "pf_streams": c["pf_streams"],
              "pf_len": c["pf_len"], "pf_particles": c["pf_particles"]}
    if stage == "A":
        return {**common, "data": "synthetic", "A_d": c["A_d"], "A_train_len": c["A_train_len"],
                "A_epochs": c["A_epochs"], "A_val_streams": c["A_val_streams"], "A_val_len": c["A_val_len"]}
    return {**common, "data": "wikitext-103 bytes", "B_train_bytes": c["B_train_bytes"],
            "B_val_bytes": c["B_val_bytes"], "B_val_streams": c["B_val_streams"]}


def source_identity(extra: tuple = ()) -> str:
    r"""SHA-256 of the local-learner package source and of this script's training, scoring and data-loading
    functions (line endings normalized), so a result is not reused across a change of the code that made it.
    ``extra`` adds stage-specific functions (stage W adds its runner, which holds its training and scoring loop)."""
    import hashlib
    import inspect
    import vfe3.local_learner as pkg
    h = hashlib.sha256()
    for f in sorted(Path(pkg.__file__).parent.glob("*.py")):
        h.update(f.name.encode() + b"\0" + f.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    for fn in (arm_settings, train_arm, protocol, scores, load_bytes, as_streams, *extra):
        h.update(inspect.getsource(fn).replace("\r\n", "\n").encode() + b"\0")
    return h.hexdigest()


def identity(arm_cfg: dict | None, proto: dict) -> dict:
    r"""The identity stored in a result record and compared before reuse (JSON round-tripped so it compares
    equal to the stored copy)."""
    return json.loads(json.dumps({"arm_settings": arm_cfg, "protocol": proto, "source": source_identity()}))


def scores(prm: Params, val: torch.Tensor, pf: torch.Tensor, tree: TreeSensor, seed: int) -> list[dict]:
    r"""Validation bits per byte and the particle-filter gap, one dict per stacked run."""
    nll, n = evaluate_filter(prm, val, tree)
    a_nll, a_n = evaluate_filter(prm, pf, tree)
    p_nll, p_n = particle_filter_nll(prm, pf, tree, CONFIG["pf_particles"], seed)
    return [{"val_bits": float(nll[r]) / n / LN2, "pf_subset_filter_bits": float(a_nll[r]) / a_n / LN2,
             "pf_subset_exact_bits": float(p_nll[r]) / p_n / LN2, "n_val": n, "n_pf": p_n} for r in range(prm.R)]


def done_records(path: Path) -> dict:
    r"""Completed result records keyed by (arm, d, seed); a later record of the same key replaces an earlier one."""
    if not path.exists():
        return {}
    return {(r["arm"], r["d"], r["seed"]): r for r in map(json.loads, path.read_text().splitlines()) if "arm" in r}


def _differences(old: object, new: object, prefix: str = "") -> list[str]:
    r"""Dotted paths at which two nested identity dicts differ."""
    if isinstance(old, dict) and isinstance(new, dict):
        return [p for k in sorted(set(old) | set(new)) for p in _differences(old.get(k), new.get(k), f"{prefix}{k}.")]
    return [] if old == new else [prefix.rstrip(".") or "identity"]


def pending(done: dict, arm: str, d: int, seeds: list[int], ident: dict) -> list[int]:
    r"""Seeds of (arm, d) still to run. A completed record is reused only when its stored identity equals
    ``ident``; a record with a different identity raises, since appending a second result under the same key
    would leave two incompatible results in one file. Records written before identities were stored cannot be
    checked: they are reused and reported."""
    todo, legacy, clash = [], [], []
    for s in seeds:
        r = done.get((arm, d, s))
        if r is None:
            todo.append(s)
        elif "identity" not in r:
            legacy.append(s)
        elif r["identity"] != ident:
            clash.append(f"seed {s}: {', '.join(_differences(r['identity'], ident))}")
    if clash:
        raise RuntimeError(f"completed results for arm {arm!r}, d={d} were made under different settings "
                           f"({'; '.join(clash)}). Set a new out_dir or remove those records to rerun.")
    if legacy:
        print(f"warning: reusing results for arm {arm!r}, d={d}, seeds {legacy} that predate stored settings; "
              f"their settings cannot be checked against CONFIG", flush=True)
    return todo


def warm_identity() -> dict:
    r"""Identity of a stage W record: the local and backpropagation arm settings, the pass and scoring-point counts,
    the data protocol stage W reads (training streams and the stage B byte budgets), and the source including
    :func:`run_warm`. The dimension and seed are the record's key, as in the other stages."""
    c = CONFIG
    arms = {"local_arm": c["W_local_arm"], "local": arm_settings(c["W_local_arm"]), "backprop": arm_settings("backprop"),
            "W_passes": c["W_passes"], "W_points": c["W_points"]}
    proto = {"stage": "W", "data": "wikitext-103 bytes", "streams": c["streams"], "B_train_bytes": c["B_train_bytes"],
             "B_val_bytes": c["B_val_bytes"], "B_val_streams": c["B_val_streams"]}
    return json.loads(json.dumps({"arm_settings": arms, "protocol": proto,
                                  "source": source_identity(extra=(run_warm,))}))


def warm_done(path: Path) -> dict:
    r"""Stage W records keyed by (run, d, seed, updates_done); a later record of the same key replaces an earlier one."""
    if not path.exists():
        return {}
    return {(r["run"], r["d"], r["seed"], r["updates_done"]): r
            for r in map(json.loads, path.read_text().splitlines()) if "run" in r}


def warm_pending(done: dict, d: int, seeds: list[int], ident: dict) -> bool:
    r"""Whether stage W must run at dimension d for these seeds. Records of (d, seed) made under a different
    identity raise, as :func:`pending` does; records without an identity cannot be checked and are reported. The
    stage is complete, and skipped, when its final row (the multi-pass backpropagation score) exists for every seed.
    Stage W trains all of its runs in one sequence, so an incomplete stage reruns whole; rows already present are
    not written twice."""
    rows = [r for (_, rd, rs, _), r in done.items() if rd == d and rs in seeds]
    clash = sorted({f"run {r['run']!r} seed {r['seed']}: {', '.join(_differences(r['identity'], ident))}"
                    for r in rows if "identity" in r and r["identity"] != ident})
    if clash:
        raise RuntimeError(f"completed stage W results for d={d} were made under different settings "
                           f"({'; '.join(clash[:5])}{'; ...' if len(clash) > 5 else ''}). Set a new out_dir or remove "
                           "those records to rerun.")
    legacy = sorted({r["seed"] for r in rows if "identity" not in r})
    if legacy:
        print(f"warning: reusing stage W results for d={d}, seeds {legacy} that predate stored settings; their "
              "settings cannot be checked against CONFIG", flush=True)
    final = f"backprop_{CONFIG['W_passes']}pass"
    finished = {rs for (run, rd, rs, _) in done if run == final and rd == d}
    return not set(seeds) <= finished


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
    done = done_records(path)
    xtrain = load_bytes("train", c["B_train_bytes"])
    xval = load_bytes("validation", c["B_val_bytes"])
    val = as_streams(xval, c["B_val_streams"]).to(DEVICE)
    pf = as_streams(xval[:c["pf_streams"] * c["pf_len"]], c["pf_streams"]).to(DEVICE)
    for spec in c["S_runs"]:
        arm, d, chunk, streams = spec["arm"], spec["d"], spec["chunk"], spec["streams"]
        label = f"{arm}_c{chunk}"
        train = as_streams(xtrain, streams).to(DEVICE)
        ident = identity(arm_settings(arm, chunk), protocol("S", streams))
        todo = pending(done, label, d, c["B_seeds"], ident)
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
                              "G_spectral": spectral[r], **sc, "identity": ident})
            del prm
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()


def run_warm(tree: TreeSensor) -> None:
    r"""Stage W: warm-start the local arm from backpropagation's solution, and train both arms for several passes,
    scoring validation along the way."""
    c = CONFIG
    out = Path(c["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    path = out / "results_W.jsonl"
    done = warm_done(path)
    d, seeds, local_arm, P = c["W_d"], c["B_seeds"], c["W_local_arm"], c["W_points"]
    ident = warm_identity()
    if not warm_pending(done, d, seeds, ident):
        print(f"stage W at d={d} is complete in {path}; nothing to run", flush=True)
        return
    xtrain = load_bytes("train", c["B_train_bytes"])
    xval = load_bytes("validation", c["B_val_bytes"])
    train = as_streams(xtrain, c["streams"]).to(DEVICE)
    val = as_streams(xval, c["B_val_streams"]).to(DEVICE)
    L = train.shape[1]
    cuts = [round(i * (L - 1) / P) for i in range(P + 1)]             # scoring points within one pass
    per_pass = (L - 1) // c["chunk"]

    def score(run: str, prm: Params, updates: int, sec: float) -> None:
        nll, n = evaluate_filter(prm, val, tree)
        for r, seed in enumerate(seeds):
            if (run, d, seed, updates) not in done:
                append(path, {"stage": "W", "run": run, "d": d, "seed": seed, "updates_done": updates,
                              "val_bits": float(nll[r]) / n / LN2, "train_sec": sec, "stacked_runs": len(seeds),
                              "local_arm": local_arm, "local_schedule": c["local_schedule"],
                              "backprop_lr": c["backprop_lr"], "identity": ident})

    def train_passes(run: str, arm: str, prm: Params, passes: int, start_updates: int = 0) -> Params:
        t0, updates = time.perf_counter(), start_updates
        for _ in range(passes):
            for i in range(P):
                segment = train[:, cuts[i]:cuts[i + 1] + 1]                # overlapping first token is context only
                prm = train_arm(arm, prm, segment, tree, None)
                sync()
                updates += -(-(segment.shape[1] - 1) // c["chunk"])
                score(run, prm, updates, time.perf_counter() - t0)
        return prm

    t0 = time.perf_counter()
    init = init_runs(d, tree.V, seeds, DEVICE)
    score("init", init, 0, 0.0)
    bp = train_arm("backprop", init, train, tree, None)                    # one uninterrupted pass, as in stage B
    sync()
    score("backprop_1pass", bp, per_pass, time.perf_counter() - t0)
    torch.save({f: getattr(bp, f).cpu() for f in ("G", "E", "Q", "A", "b")}, out / f"W_backprop_d{d}.pt")
    train_passes("warm_local_from_backprop", local_arm, bp, 1, per_pass)
    train_passes(f"local_{c['W_passes']}pass", local_arm, init, c["W_passes"])
    # Backpropagation trains its passes as one call (segmenting would restart Adam's moments) and is scored once.
    t0 = time.perf_counter()
    bp_multi = train_arm("backprop", init, train.repeat(1, c["W_passes"]), tree, None)
    sync()
    score(f"backprop_{c['W_passes']}pass", bp_multi, per_pass * c["W_passes"], time.perf_counter() - t0)


def run_accel(tree: TreeSensor) -> None:
    r"""Stage X: each arm of ``X_arms`` trains ``X_passes`` passes from the stage B initialization at d = ``X_d``,
    three stacked seeds, scored at ``W_points`` points per pass. An arm whose final point is already recorded
    under the same identity is skipped; a recorded final point under a different identity raises."""
    c = CONFIG
    out = Path(c["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    path = out / "results_X.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    recorded = {(r["run"], r["d"], r["seed"], r["updates_done"]): r for r in rows}
    d, seeds, P = c["X_d"], c["B_seeds"], c["W_points"]
    xtrain = load_bytes("train", c["B_train_bytes"])
    xval = load_bytes("validation", c["B_val_bytes"])
    train = as_streams(xtrain, c["streams"]).to(DEVICE)
    val = as_streams(xval, c["B_val_streams"]).to(DEVICE)
    L = train.shape[1]
    cuts = [round(i * (L - 1) / P) for i in range(P + 1)]
    final = sum(-(-(cuts[i + 1] - cuts[i]) // c["chunk"]) for i in range(P)) * c["X_passes"]
    proto = {**protocol("B"), "stage": "X", "X_passes": c["X_passes"], "W_points": P}
    for arm in c["X_arms"]:
        ident = identity(arm_settings(arm), proto)
        last = [recorded.get((arm, d, s, final)) for s in seeds]
        if all(r is not None for r in last):
            if any(r.get("identity") != ident for r in last):
                raise RuntimeError(f"stage X results for {arm!r} were made under different settings; set a new "
                                   f"out_dir or remove those records to rerun")
            continue
        prm, updates, t0 = init_runs(d, tree.V, seeds, DEVICE), 0, time.perf_counter()
        for _ in range(c["X_passes"]):
            for i in range(P):
                prm = train_arm(arm, prm, train[:, cuts[i]:cuts[i + 1] + 1], tree, None)
                sync()
                updates += -(-(cuts[i + 1] - cuts[i]) // c["chunk"])
                nll, n = evaluate_filter(prm, val, tree)
                spectral = torch.linalg.matrix_norm(prm.G, 2).tolist()
                for r, seed in enumerate(seeds):
                    append(path, {"stage": "X", "run": arm, "d": d, "seed": seed, "updates_done": updates,
                                  "val_bits": float(nll[r]) / n / LN2, "train_sec": time.perf_counter() - t0,
                                  "stacked_runs": len(seeds), "G_spectral": spectral[r], "identity": ident})


def run_stage(stage: str, tree: TreeSensor) -> None:
    c = CONFIG
    out = Path(c["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"results_{stage}.jsonl"
    done = done_records(path)
    proto = protocol(stage)
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
        ident = identity({"rule": "truth"}, proto)
        if pending(done, "truth", d, [-1], ident):
            append(path, {"stage": "A", "arm": "truth", "d": d, "seed": -1, **scores(truth, val, pf, tree, 0)[0],
                          "identity": ident})
        uni, n = ngram_nll(train_cpu.reshape(-1).numpy(), val_cpu.numpy(), tree.V, 1, 1.0)
        ident = identity({"rule": "ngram", "order": 1, "k": 1.0}, proto)
        if pending(done, "unigram", 0, [0], ident):
            append(path, {"stage": "A", "arm": "unigram", "d": 0, "seed": 0, "val_bits": uni / n / LN2,
                          "identity": ident})
    else:
        xtrain = load_bytes("train", c["B_train_bytes"])
        xval = load_bytes("validation", c["B_val_bytes"])
        val_cpu = as_streams(xval, c["B_val_streams"])
        train = as_streams(xtrain, c["streams"]).to(DEVICE)
        val = val_cpu.to(DEVICE)
        pf = as_streams(xval[:c["pf_streams"] * c["pf_len"]], c["pf_streams"]).to(DEVICE)
        for order in ((1, 2, 3) if stage == "B" else ()):    # stage C reuses stage B's n-gram references
            key = (f"{order}-gram", 0, 0)
            ident = identity({"rule": "ngram", "order": order, "k_grid": c["ngram_k_grid"]}, proto)
            if not pending(done, key[0], 0, [0], ident):
                continue
            best = min((ngram_nll(xtrain, val_cpu.numpy(), tree.V, order, k), k) for k in c["ngram_k_grid"])
            (nll, n), k = best
            append(path, {"stage": "B", "arm": key[0], "d": 0, "seed": 0, "val_bits": nll / n / LN2, "k": k,
                          "identity": ident})
        dims, seeds = (c["C_dims"] if stage == "C" else c["B_dims"]), c["B_seeds"]
    for d in dims:
        for arm in {"A": c["A_arms"], "C": c.get("C_arms", [])}.get(stage, c["arms"]):
            ident = identity(arm_settings(arm), proto)
            todo = pending(done, arm, d, seeds, ident)
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
                              "G_spectral": spectral[r], **sc, "identity": ident})


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
    elif CONFIG["stage"] == "W":
        run_warm(tree)
    elif CONFIG["stage"] == "X":
        run_accel(tree)
    else:
        run_stage(CONFIG["stage"], tree)


if __name__ == "__main__":
    main()
