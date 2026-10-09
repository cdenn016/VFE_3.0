r"""Per-epoch training-window offsets (``train_window_offset``, audit 2026-10-09 training sweep).

Under ``"fixed"`` (the default) training window ``i`` starts at ``i * stride`` in every epoch, bitwise
as before. Under ``"per_epoch_random"`` every epoch draws an origin :math:`o \in [0, s)` from the train
loader's seeded generator, window ``i`` starts at :math:`o + i s`, and the window count is the one that
fits every origin. Tiny streams and models only (K = 4, two heads, one layer).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, List, Tuple

import pytest
import torch
from torch.utils.data import DataLoader

import vfe3.train as vt
from tests.profiles import config_values, test_config
from vfe3.data.datasets import TokenWindows
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import RunArtifacts

SEQ = 8
T_TOKENS = SEQ * 10 + 1          # fixed: 10 windows; per-epoch offsets: 9 windows fit every origin
BATCH = 3                        # 3 steps per epoch under the offsets


def _stream(n: int = T_TOKENS, vocab: int = 6) -> torch.Tensor:
    return torch.arange(n, dtype=torch.long) % vocab


def _loader(seed: int = 5, tokens: torch.Tensor | None = None) -> DataLoader:
    ds = TokenWindows(_stream() if tokens is None else tokens, SEQ)
    return DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=True,
                      generator=torch.Generator().manual_seed(seed))


def _cfg(**kw: Any) -> Any:
    base = dict(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=SEQ, n_layers=1, n_e_steps=1,
                e_q_mu_lr=0.1, warmup_steps=1, max_steps=8, batch_size=BATCH)
    base.update(kw)
    return test_config(**base)


# ---- config ------------------------------------------------------------------------------------

def test_field_defaults_to_fixed_is_validated_and_is_a_shared_baseline_protocol_field() -> None:
    import paper_arms as pa
    from vfe3.baseline import BaselineConfig
    from vfe3.config import VFE3Config

    assert VFE3Config.__dataclass_fields__["train_window_offset"].default == "fixed"
    assert BaselineConfig().train_window_offset == "fixed"
    assert config_values()["train_window_offset"] == "fixed"
    with pytest.raises(ValueError, match="train_window_offset"):
        _cfg(train_window_offset="random")
    with pytest.raises(ValueError, match="train_window_offset"):
        BaselineConfig(train_window_offset="random")
    assert pa.BASELINE_PROTOCOL_FIELDS["train_window_offset"] == "train_window_offset"
    matched = BaselineConfig.matched_to(_cfg(train_window_offset="per_epoch_random", block_mlp_expansion=2,
                                             use_block_mlp=True))
    assert matched.train_window_offset == "per_epoch_random"


# ---- TokenWindows ------------------------------------------------------------------------------

def test_fixed_windows_are_bitwise_unchanged() -> None:
    tokens = _stream()
    ds = TokenWindows(tokens, SEQ)
    assert ds.window_offset_mode == "fixed" and ds.window_offset == 0 and len(ds) == 10
    expected = [(tokens[i * SEQ:i * SEQ + SEQ], tokens[i * SEQ + 1:i * SEQ + SEQ + 1]) for i in range(10)]
    ds.set_window_offset_mode("fixed")                               # a reset on a fresh dataset is a no-op
    for (x, y), (ex, ey) in zip((ds[i] for i in range(len(ds))), expected):
        assert torch.equal(x, ex) and torch.equal(y, ey)
    with pytest.raises(RuntimeError):
        ds.draw_window_offset(torch.Generator().manual_seed(0))      # fixed windows draw nothing


def test_per_epoch_offsets_tile_each_epoch_without_duplication() -> None:
    tokens = torch.arange(T_TOKENS, dtype=torch.long)                # position-valued stream
    ds = TokenWindows(tokens, SEQ)
    ds.set_window_offset_mode("per_epoch_random")
    assert len(ds) == 9                                              # floor((T - 1 - 2 SEQ + 1) / SEQ) + 1
    generator = torch.Generator().manual_seed(0)
    origins = set()
    for _ in range(12):
        offset = ds.draw_window_offset(generator)
        assert 0 <= offset < SEQ and ds.window_offset == offset
        origins.add(offset)
        starts = sorted(int(ds[i][0][0]) for i in range(len(ds)))
        assert starts == [offset + i * SEQ for i in range(len(ds))]
        scored = torch.cat([ds[i][1] for i in range(len(ds))])     # target positions = transitions scored
        assert scored.unique().numel() == scored.numel() == len(ds) * SEQ       # no transition twice
        assert int(scored.min()) == offset + 1 and int(scored.max()) <= T_TOKENS - 1
        assert offset + len(ds) * SEQ <= T_TOKENS - 1                # every window lies in the stream
    assert len(origins) > 1                                          # origins differ across epochs
    assert torch.equal(ds.at_origin()[3][0], tokens[3 * SEQ:4 * SEQ])  # the fixed-origin view
    ds.set_window_offset_mode("fixed")
    assert len(ds) == 10 and ds.window_offset == 0
    with pytest.raises(ValueError):
        TokenWindows(tokens, SEQ, pad_final=True).set_window_offset_mode("per_epoch_random")


# ---- the training loop -------------------------------------------------------------------------

def _record_batches(monkeypatch: pytest.MonkeyPatch) -> List[Tuple[torch.Tensor, int]]:
    seen: List[Tuple[torch.Tensor, int]] = []
    original = vt.train_step

    def recording(model: Any, optimizer: Any, scheduler: Any, tokens: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        seen.append(tokens.detach().cpu().clone())
        return original(model, optimizer, scheduler, tokens, *args, **kwargs)

    monkeypatch.setattr(vt, "train_step", recording)
    return seen


def test_fixed_training_reads_the_historical_epoch_order(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _record_batches(monkeypatch)
    cfg = _cfg()
    torch.manual_seed(0)
    vt.train(VFEModel(cfg), _loader(), cfg, n_steps=8, log_interval=0, generate_samples=False)
    replay = _loader()
    expected: List[torch.Tensor] = []
    while len(expected) < 8:
        expected.extend(x for x, _ in replay)
    assert len(seen) == 8 and all(torch.equal(a, b) for a, b in zip(seen, expected[:8]))
    identity = vt._loader_data_identity(_loader(), cfg.vocab_size)
    assert "window_offset" not in identity["iterator"]


def test_per_epoch_offsets_change_the_origin_and_resume_exactly_mid_epoch(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _record_batches(monkeypatch)
    drawn: List[int] = []
    original_draw = TokenWindows.draw_window_offset

    def recording_draw(self: TokenWindows, generator: torch.Generator) -> int:
        drawn.append(original_draw(self, generator))
        return drawn[-1]

    monkeypatch.setattr(TokenWindows, "draw_window_offset", recording_draw)
    # Position-valued tokens (vocabulary above the stream length) identify every window's start.
    cfg = _cfg(train_window_offset="per_epoch_random", checkpoint_interval=4, vocab_size=96)
    tokens = torch.arange(T_TOKENS, dtype=torch.long)

    torch.manual_seed(0)
    straight = _loader(tokens=tokens)
    vt.train(VFEModel(cfg), straight, cfg, n_steps=8, log_interval=0, generate_samples=False)
    assert len(seen) == 8 and len(straight) == 3 and len(straight.dataset) == 9
    assert len(drawn) == 3 and len(set(drawn)) > 1                   # epochs start at steps 0, 3, 6
    for step, batch in enumerate(seen):
        assert all(int(start) % SEQ == drawn[step // 3] for start in batch[:, 0])
    epoch_windows = torch.cat(seen[:3])[:, 0]
    assert epoch_windows.unique().numel() == 9                         # one epoch: every window once
    straight_steps, straight_drawn = [s.clone() for s in seen], list(drawn)
    seen.clear()
    drawn.clear()

    torch.manual_seed(0)
    model = VFEModel(cfg)
    art = RunArtifacts(tmp_path / "run", cfg, model)
    first = _loader(tokens=tokens)
    vt.train(model, first, cfg, n_steps=4, artifacts=art, log_interval=0, generate_samples=False)
    ckpt = tmp_path / "run" / "checkpoints" / "step_4.pt"
    saved = torch.load(ckpt, weights_only=False)["data_state"]
    assert saved["epoch"] == 1 and saved["batches_consumed"] == 1      # mid-epoch
    assert saved["window_offset"] == straight_drawn[1] == first.dataset.window_offset
    assert saved["data_identity"]["iterator"]["window_offset"] == "per_epoch_random"
    assert all(torch.equal(a, b) for a, b in zip(seen, straight_steps[:4]))
    seen.clear()

    resumed = _loader(tokens=tokens)
    vt.train(VFEModel(cfg), resumed, cfg, n_steps=8, resume_from=ckpt, log_interval=0, generate_samples=False)
    assert len(seen) == 4 and all(torch.equal(a, b) for a, b in zip(seen, straight_steps[4:]))
    # A fixed-window run cannot resume this checkpoint: its iterator identity differs.
    with pytest.raises(RuntimeError, match="data identity"):
        fixed = dataclasses.replace(cfg, train_window_offset="fixed")
        vt.train(VFEModel(fixed), _loader(tokens=tokens), fixed, n_steps=8, resume_from=ckpt,
                 log_interval=0, generate_samples=False)


def test_per_epoch_offsets_refuse_a_loader_without_a_generator() -> None:
    cfg = _cfg(train_window_offset="per_epoch_random")
    ds = TokenWindows(_stream(), SEQ)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False, drop_last=True)
    with pytest.raises(ValueError, match="torch.Generator"):
        vt.train(VFEModel(cfg), loader, cfg, n_steps=1, log_interval=0, generate_samples=False)


def test_a_shared_loader_returns_to_fixed_windows_under_a_fixed_config() -> None:
    loader = _loader()
    vt.train(VFEModel(_cfg(train_window_offset="per_epoch_random")), loader,
             _cfg(train_window_offset="per_epoch_random"), n_steps=4, log_interval=0, generate_samples=False)
    assert loader.dataset.window_offset_mode == "per_epoch_random"
    vt.train(VFEModel(_cfg()), loader, _cfg(), n_steps=1, log_interval=0, generate_samples=False)
    assert loader.dataset.window_offset_mode == "fixed" and loader.dataset.window_offset == 0
    assert len(loader.dataset) == 10
