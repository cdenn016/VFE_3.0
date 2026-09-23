r"""Click-to-run for the tests that need the GPU.

Two checks live here, both requiring CUDA (your RTX 5090):

  * t3  -- the learnability gate ``test_training_decreases_loss_on_structured_stream``. It is kept as
           a NON-strict xfail because on CPU the per-head-tau model clears the ln(3) floor by only a
           thin margin and the LRs were calibrated for the old sqrt(embed_dim) tau. This script RUNS
           the same 3-seed period-3 training on the GPU and REPORTS the median end-CE, the ln(3)-margin
           threshold, and the headroom -- so you can decide whether to promote it to a hard gate (and
           re-tune the LRs via LR_OVERRIDES below).
  * CUDA lane -- every test selected by the canonical ``cuda`` marker, including CUDA-only tests and
                 the ordinary numerical contracts mirrored onto CUDA by the collection policy.

No CLI args -- edit the config, then::

    python check_gpu_tests.py
"""

import os

# ---- config (edit me) ------------------------------------------------------
N_SEEDS      = 3          # t3 averages the median end-CE over this many fixed seeds
MAX_STEPS    = 200        # t3 training steps per seed (the gate's default)
LR_OVERRIDES = {}         # re-tune for the per-head tau, e.g. {"e_phi_lr": 0.4, "m_phi_lr": 0.06}
RUN_T6       = True       # also run the canonical CUDA marker lane
# ---------------------------------------------------------------------------

os.environ["VFE3_TEST_DEVICE"] = "cuda"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import sys
from dataclasses import replace

import torch

from check_junit import junit_is_exact_all_pass, run_pytest_junit
from tests.pytest_policy import CUDA_MIRROR_TESTS, CUDA_TESTS


REQUIRED_CUDA_TEST_KEYS = CUDA_TESTS | CUDA_MIRROR_TESTS


def _run_t3(dev: torch.device) -> bool:
    from tests.test_train import (_MARGINAL_ENTROPY_P3, _CUTOVER_MARGIN,
                                  _median, _periodic_loader, _structured_cfg)
    from vfe3.model.model import VFEModel
    from vfe3.train import train

    threshold = _MARGINAL_ENTROPY_P3 - _CUTOVER_MARGIN
    print("t3 -- learnability gate (period-3 next-token structure, gauge transport ON):")
    ends = []
    for seed in range(N_SEEDS):
        torch.manual_seed(seed)
        cfg = _structured_cfg()
        if LR_OVERRIDES:
            cfg = replace(cfg, **LR_OVERRIDES)
        cfg = replace(cfg, max_steps=MAX_STEPS)
        model = VFEModel(cfg).to(dev)
        losses = train(model, _periodic_loader(V=6, period=3, seed=seed),
                       cfg, n_steps=MAX_STEPS, grad_clip=cfg.grad_clip, device=dev)
        ends.append(float(losses[-1]))
        print(f"    seed {seed}: end CE = {ends[-1]:.4f}")
    med = _median(ends)
    headroom = threshold - med
    clears = headroom > 0.0
    print(f"    median end CE = {med:.4f}   threshold = ln(3)-{_CUTOVER_MARGIN} = {threshold:.4f}")
    print(f"    headroom = {headroom:+.4f}  ->  {'CLEARS the floor (learns the period)' if clears else 'FAILS (pins at ln 3)'}")
    if clears and headroom < 0.05:
        print("    NOTE: clears, but headroom < 0.05 -- still thin; consider LR_OVERRIDES before promoting to strict.")
    print()
    return clears


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA is not available in this environment -- these tests NEED the GPU.")
        print("Run this on the RTX 5090 (and make sure torch was installed with CUDA support).")
        return 2
    dev = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}  |  torch {torch.__version__}\n")

    t3_ok = _run_t3(dev)

    cuda_code = 0
    cuda_counts = None
    cuda_ok = True
    if RUN_T6:
        print("CUDA marker lane -- canonical CUDA-only and mirrored tests:")
        cuda_code, cuda_counts = run_pytest_junit(
            ["-m", "cuda", "-v", "-p", "no:cacheprovider"],
            prefix="vfe3-gpu-cuda-",
            required_node_keys=REQUIRED_CUDA_TEST_KEYS,
        )
        cuda_ok = cuda_code == 0 and junit_is_exact_all_pass(
            cuda_counts,
            expected_tests=cuda_counts["collected"],
        )

    bar = "=" * 64
    print("\n" + bar)
    if RUN_T6:
        cuda_status = "GREEN" if cuda_ok else f"FAIL (exit {cuda_code})"
        cuda_summary = (
            f"  |  CUDA {cuda_status} "
            f"({cuda_counts['passes']} passed, {cuda_counts['skipped']} skipped, "
            f"{cuda_counts['failures']} failed, {cuda_counts['errors']} errors)"
        )
    else:
        cuda_summary = ""
    print(f"GPU CHECK: t3 gate {'CLEARS' if t3_ok else 'FAILS'} the ln(3) margin" + cuda_summary)
    print(bar)
    # exit 0 only if the t3 gate clears AND (the CUDA lane ran green or was skipped)
    return 0 if (t3_ok and cuda_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
