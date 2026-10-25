"""Tiny CPU reproduction: artifact-free resume drops the loss-spike rejection state.

Run only by the audit coordinator, serially. This creates a normal checkpoint and
never edits its bytes. The synthetic forward adds a constant 10-nat loss spike
after resume; its derivative remains the original model derivative.
"""
from __future__ import annotations

import importlib
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VFE3_TEST_DEVICE"] = "cpu"
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from tests.profiles import test_config
from vfe3.data.datasets import TokenWindows
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import RunArtifacts

T = importlib.import_module("vfe3.train")


def loader() -> DataLoader:
    return DataLoader(TokenWindows(torch.arange(15) % 5, 2), batch_size=1,
                      shuffle=False, drop_last=True)


def main() -> None:
    torch.set_num_threads(1)
    T.DIVERGENCE_EMA_WARMUP = 1
    cfg = test_config(vocab_size=5, embed_dim=4, n_heads=2, n_layers=1,
                      n_e_steps=1, batch_size=1, max_seq_len=2, eval_stride=None,
                      max_steps=3, warmup_steps=0, min_lr_frac=0.0,
                      grad_clip=None, log_interval=0, eval_interval=0,
                      checkpoint_interval=1, checkpoint_at_end=False,
                      decode_mode="linear", decode_unigram_prior=False,
                      attention_mode="projected_predictive", family="gaussian_diagonal",
                      use_block_mlp=False, skip_belief_sigma_update=True,
                      m_table_update_mode="adamw", m_phi_update_mode="adamw",
                      amp_dtype=None, fp32_matmul_precision="highest",
                      loss_spike_skip_nats=1.0, generate_embedding_figures=False)
    logger = logging.getLogger("resume-monitor-probe")
    with tempfile.TemporaryDirectory(prefix="resume-monitor-", dir=Path(__file__).parent, delete=False) as temporary:
        directory = Path(temporary)
        torch.manual_seed(17)
        model = VFEModel(cfg)
        artifacts = RunArtifacts(directory / "source", cfg, model)
        T.train(model, loader(), cfg, n_steps=1, grad_clip=cfg.grad_clip,
                log_interval=0, eval_interval=0, artifacts=artifacts,
                device=torch.device("cpu"), logger=logger, generate_samples=False)
        checkpoint = directory / "source" / "checkpoints" / "step_1.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        results: dict[str, Any] = {"saved_monitor": {
            "loss_seen": saved["divergence_monitor"]["monitor"]["loss_seen"],
            "loss_ema": saved["divergence_monitor"]["monitor"]["loss_ema"],
        }}
        original_step = T.train_step
        for use_artifacts in (True, False):
            torch.manual_seed(29)
            resumed = VFEModel(cfg)
            original_forward = resumed.forward

            def spiked_forward(*args: Any, **kwargs: Any) -> Any:
                logits, loss, statistics = original_forward(*args, **kwargs)
                return logits, loss + 10.0, statistics

            resumed.forward = spiked_forward
            observed: dict[str, Any] = {}

            def observe_step(*args: Any, **kwargs: Any) -> float:
                observed["loss_skip_threshold"] = kwargs.get("loss_skip_threshold")
                loss = original_step(*args, **kwargs)
                status = kwargs["status_out"]
                observed.update(loss=float(loss), did_step=status["did_step"],
                                loss_spike_skip=status["loss_spike_skip"])
                return loss

            T.train_step = observe_step
            destination = (RunArtifacts(directory / "resumed", cfg, resumed)
                           if use_artifacts else None)
            try:
                T.train(resumed, loader(), cfg, n_steps=2, grad_clip=cfg.grad_clip,
                        log_interval=0, eval_interval=0, artifacts=destination,
                        device=torch.device("cpu"), logger=logger,
                        generate_samples=False, resume_from=checkpoint)
            finally:
                T.train_step = original_step
            results["with_artifacts" if use_artifacts else "without_artifacts"] = observed
        print(json.dumps(results, indent=2, allow_nan=False))
        assert results["with_artifacts"]["loss_spike_skip"] is True
        assert results["without_artifacts"]["loss_skip_threshold"] is None
        assert results["without_artifacts"]["did_step"] is True


if __name__ == "__main__":
    main()
