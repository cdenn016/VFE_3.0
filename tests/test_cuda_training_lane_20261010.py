r"""CUDA lane: a tiny training run, checkpoint and resume, and evaluation on the device (audit 2026-10-10, item 50).

The CPU lane cannot reach the device-only paths of the loop: fused AdamW, autocast, pinned non-blocking
transfers and the CUDA RNG restore. Both tests are registered in ``tests/pytest_policy.py::CUDA_TESTS``
and skip without CUDA.
"""

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from tests.profiles import test_config
from vfe3.data.datasets import TokenWindows
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import RunArtifacts
from vfe3.train import evaluate, train

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _loader(*, train_split: bool) -> DataLoader:
    g = torch.Generator().manual_seed(0)
    stream = torch.randint(0, 9, (8 * 12 + 1,), generator=g)
    if train_split:
        return DataLoader(TokenWindows(stream, 8), batch_size=4, shuffle=False, drop_last=True)
    return DataLoader(TokenWindows(stream, 8, pad_final=True, stride=4), batch_size=4,
                      shuffle=False, drop_last=False)


def _cfg(**overrides: object):
    return test_config(**{**dict(vocab_size=9, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=2,
                                 warmup_steps=1, max_steps=4, amp_dtype="bf16", checkpoint_interval=2),
                          **overrides})


def _params(model: torch.nn.Module) -> list:
    return [p.detach().float().cpu().clone() for p in model.parameters()]


@requires_cuda
def test_cuda_resume_matches_the_uninterrupted_run(tmp_path: Path) -> None:
    device = torch.device("cuda")
    cfg = _cfg()
    torch.manual_seed(0)
    straight = VFEModel(cfg).to(device)
    initial = _params(straight)
    train(straight, _loader(train_split=True), cfg, n_steps=4, device=device)
    final = _params(straight)
    assert any(not torch.equal(a, b) for a, b in zip(initial, final))

    torch.manual_seed(0)
    first = VFEModel(cfg).to(device)
    artifacts = RunArtifacts(tmp_path / "run", cfg, first)
    train(first, _loader(train_split=True), cfg, n_steps=2, device=device, artifacts=artifacts)
    checkpoint = tmp_path / "run" / "checkpoints" / "step_2.pt"
    assert checkpoint.exists()
    resumed = VFEModel(cfg).to(device)
    losses = train(resumed, _loader(train_split=True), cfg, n_steps=4, device=device, resume_from=checkpoint)
    assert len(losses) == 2
    for a, b in zip(final, _params(resumed)):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)


@requires_cuda
def test_cuda_evaluate_matches_cpu_evaluate() -> None:
    cfg = _cfg(amp_dtype=None, fp32_matmul_precision="highest")
    torch.manual_seed(1)
    model = VFEModel(cfg)
    on_cpu = evaluate(model, _loader(train_split=False), device=torch.device("cpu"))
    model = model.to("cuda")
    on_cuda = evaluate(model, _loader(train_split=False), device=torch.device("cuda"))
    assert on_cpu.get("scored_targets") == on_cuda.get("scored_targets")
    assert abs(on_cpu["ce"] - on_cuda["ce"]) < 1e-4
