r"""Tests for the EXP-1 fixed data-order generator (S6), added 2026-06-21.

make_dataloader now threads a torch.Generator so the TRAIN shuffle order can be fixed independent of
the global RNG (so a multi-seed run shares one batch order while model-init RNG varies). Uses a
monkeypatched token cache so no real corpus is required."""
import torch

import vfe3.data.datasets as ds


def _patch_tokens(monkeypatch, n=2000):
    monkeypatch.setattr(ds, "load_cached_tokens",
                        lambda *a, **k: torch.arange(n, dtype=torch.long))
    monkeypatch.setattr(ds, "cache_source_identity", lambda *a, **k: {
        "format": "pt", "tokenizer_tag": "fixture", "size_bytes": n * 8,
        "sha256": "0" * 64, "meta": None, "meta_sha256": None,
    })


def _full_order(loader):
    """Concatenate every batch into one flat list -- the full data order the shuffle produced."""
    flat = []
    for b in loader:
        x = b[0] if isinstance(b, (tuple, list)) else b
        flat.extend(x.reshape(-1).tolist())
    return flat


def _loader(generator):
    return ds.make_dataloader("x", "train", 8, 4, shuffle=True, generator=generator)


def test_same_generator_seed_gives_identical_order(monkeypatch):
    _patch_tokens(monkeypatch)
    a = _full_order(_loader(torch.Generator().manual_seed(123)))
    b = _full_order(_loader(torch.Generator().manual_seed(123)))
    assert a == b


def test_different_generator_seed_changes_order(monkeypatch):
    _patch_tokens(monkeypatch)
    a = _full_order(_loader(torch.Generator().manual_seed(123)))
    b = _full_order(_loader(torch.Generator().manual_seed(999)))
    assert a != b


def test_generator_none_preserves_global_rng_behavior(monkeypatch):
    _patch_tokens(monkeypatch)
    torch.manual_seed(7)
    a = _full_order(_loader(None))
    torch.manual_seed(7)
    b = _full_order(_loader(None))
    assert a == b                          # default path: global-RNG shuffle, reseed reproduces it
