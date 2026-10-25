"""Tiny CPU fixture for a normal best-model publication blocked by a Windows lock.

Coordinator execution only. Produces an ordinary earlier best-model bundle and
simulates PermissionError on its next publication. No existing files are edited.
"""
from __future__ import annotations

import importlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["VFE3_TEST_DEVICE"] = "cpu"
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from torch.utils.data import DataLoader
from tests.profiles import test_config
from vfe3.data.datasets import TokenWindows
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import RunArtifacts

T = importlib.import_module("vfe3.train")
R = importlib.import_module("vfe3.run_artifacts")


def main() -> None:
    torch.set_num_threads(1)
    cfg = test_config(vocab_size=5, embed_dim=4, n_heads=2, n_layers=1,
                      batch_size=1, max_seq_len=2, eval_stride=None, max_steps=2, warmup_steps=0,
                      use_block_mlp=False, decode_mode="linear",
                      decode_unigram_prior=False, generate_embedding_figures=False)
    model = VFEModel(cfg)
    loader = DataLoader(TokenWindows(torch.arange(9) % 5, 2, pad_final=True),
                        batch_size=1, shuffle=False)
    directory = Path(tempfile.mkdtemp(prefix="locked-selection-probe-", dir=Path(__file__).parent))
    artifacts = RunArtifacts(directory, cfg, model)
    artifacts.bind_selection_data_identity(T._loader_data_identity(loader, cfg.vocab_size))
    assert artifacts.maybe_save_best(1, model, 20.0)
    with torch.no_grad():
        next(model.parameters()).add_(0.01)
    original_replace = R._atomic_replace

    def locked_replace(final: Path, temporary: Path, **kwargs: Any) -> None:
        if final.name == "best_model.pt":
            raise PermissionError("simulated ordinary Windows reader lock")
        original_replace(final, temporary, **kwargs)

    def terminal_metrics(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"ce": math.log(18.0), "ppl": 18.0,
                "bits_per_token": math.log2(18.0), "bpc": None,
                "expected_targets": 8, "scored_targets": 8, "excluded_targets": 0}

    R._atomic_replace = locked_replace
    T.evaluate = terminal_metrics
    R._write_provenance = lambda *args, **kwargs: None
    R._pure_path_report = lambda *args, **kwargs: {}
    R._run_figures_isolated = lambda *args, **kwargs: True
    result = R.finalize_validation_run(model, artifacts, cfg, loader, device=torch.device("cpu"))
    observed = {"primary_val_ppl": result["primary_val_ppl"],
                "best_val_ppl": result["best_val_ppl"], "best_step": result["best_step"],
                "summary_written": (directory / "summary.json").is_file(),
                "artifacts": str(directory)}
    print(json.dumps(observed, indent=2))
    assert observed["primary_val_ppl"] == 18.0
    assert observed["best_val_ppl"] == 20.0
    assert observed["best_step"] == 1
    assert observed["summary_written"]


if __name__ == "__main__":
    main()
