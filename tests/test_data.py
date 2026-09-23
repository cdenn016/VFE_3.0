import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vfe3.data.datasets import (
    TokenWindows,
    _tokenizer_tag,
    cache_source_identity,
    cache_path,
    default_cache_dir,
    get_tiktoken_decoder,
    load_cached_tokens,
    tiktoken_encoding_name,
    tokenizer_vocab_size,
)


def test_token_windows_shift_and_length():
    tokens = torch.arange(20)
    ds = TokenWindows(tokens, seq_len=5, stride=5)
    assert len(ds) == (20 - 5 - 1) // 5 + 1
    x, y = ds[0]
    assert x.shape == (5,) and y.shape == (5,)
    assert torch.equal(x, torch.arange(0, 5))
    assert torch.equal(y, torch.arange(1, 6))                 # target is input shifted by 1


def test_token_windows_casts_only_the_active_l_plus_one_slice():
    tokens = torch.arange(20, dtype=torch.int32)
    ds = TokenWindows(tokens, seq_len=4, stride=4)
    x, y = ds[1]
    assert x.dtype == y.dtype == torch.long
    assert x.tolist() == [4, 5, 6, 7]
    assert y.tolist() == [5, 6, 7, 8]
    assert ds.tokens.dtype == torch.int32


def test_token_windows_never_casts_the_backing_corpus(monkeypatch):
    tokens = torch.arange(100, dtype=torch.int32)
    ds = TokenWindows(tokens, seq_len=8, stride=8)
    original = torch.Tensor.to
    converted_numel = []

    def tracked_to(self, *args, **kwargs):
        converted_numel.append(self.numel())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", tracked_to)
    x, y = ds[3]
    assert x.shape == y.shape == (8,)
    assert converted_numel == [9]


def test_get_tiktoken_decoder_synthetic_is_none():
    # The synthetic anchor has no real tokenizer, so no decoder (the caller emits no sample text).
    assert get_tiktoken_decoder("synthetic-period3") is None


def test_get_tiktoken_decoder_roundtrips_when_tiktoken_present():
    # When tiktoken is installed, the gpt2 decoder round-trips a known id sequence to text;
    # skip cleanly on a box without tiktoken (the decoder is None there, never a crash).
    tiktoken = pytest.importorskip("tiktoken")
    dec = get_tiktoken_decoder("wikitext-103")
    assert dec is not None
    assert dec([15496, 995]) == "Hello world"                  # gpt2 ids for 'Hello world'


def test_wiki_ar_uses_cl100k_cache_and_vocab(tmp_path):
    assert _tokenizer_tag("wiki-ar") == "tiktoken_cl100k"
    assert tiktoken_encoding_name("wiki-ar") == "cl100k_base"
    assert tokenizer_vocab_size("wiki-ar") == 100277
    assert cache_path("wiki-ar", "validation", cache_dir=tmp_path).name == (
        "wiki-ar_validation_tiktoken_cl100k_tokens.pt"
    )


def test_multilingual_decoders_select_cl100k(monkeypatch):
    calls = []
    fake = SimpleNamespace(
        get_encoding=lambda name: calls.append(name) or SimpleNamespace(
            decode=lambda ids: ",".join(str(i) for i in ids)
        )
    )
    monkeypatch.setitem(sys.modules, "tiktoken", fake)

    assert get_tiktoken_decoder("wiki-ja")([1, 2]) == "1,2"
    assert get_tiktoken_decoder("wiki-ar")([3]) == "3"
    assert calls == ["cl100k_base", "cl100k_base"]


def test_token_windows_rejects_short_stream():
    with pytest.raises(ValueError):
        TokenWindows(torch.arange(3), seq_len=10)


def test_load_synthetic_pt_cache(tmp_path):
    # write a synthetic .pt cache under the loader's naming convention and load it
    p = cache_path("wikitext-103", "train", suffix="pt", cache_dir=tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    toks = torch.randint(0, 100, (500,), dtype=torch.int64)
    torch.save(toks, p)
    out = load_cached_tokens("wikitext-103", "train", cache_dir=tmp_path)
    assert out.dtype == torch.long and out.shape == (500,)
    assert torch.equal(out, toks)


def test_legacy_cache_tokenizer_provenance_is_explicitly_unverified(tmp_path):
    path = cache_path("wikitext-103", "train", suffix="pt", cache_dir=tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.arange(8, dtype=torch.int64), path)

    identity = cache_source_identity("wikitext-103", "train", cache_dir=tmp_path)

    assert identity["tokenizer_provenance_status"] == "filename_inferred_unverified"
    assert identity["tokenizer_provenance"] is None
    assert identity["tokenizer_provenance_sha256"] is None


def test_cache_tokenizer_provenance_manifest_binds_exact_payload(tmp_path):
    dataset, split = "wikitext-103", "train"
    path = cache_path(dataset, split, suffix="pt", cache_dir=tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.arange(8, dtype=torch.int64), path)
    legacy = cache_source_identity(dataset, split, cache_dir=tmp_path)
    manifest = {
        "schema_version": 1,
        "dataset": dataset,
        "split": split,
        "tokenizer_tag": _tokenizer_tag(dataset),
        "tokenizer_encoding": tiktoken_encoding_name(dataset),
        "tokenizer_vocab_size": tokenizer_vocab_size(dataset),
        "payload_sha256": legacy["sha256"],
    }
    Path(str(path) + ".provenance.json").write_text(
        json.dumps(manifest), encoding="utf-8",
    )

    verified = cache_source_identity(dataset, split, cache_dir=tmp_path)

    assert verified["tokenizer_provenance_status"] == "manifest_verified"
    assert verified["tokenizer_provenance"] == manifest
    assert isinstance(verified["tokenizer_provenance_sha256"], str)

    manifest["payload_sha256"] = "0" * 64
    Path(str(path) + ".provenance.json").write_text(
        json.dumps(manifest), encoding="utf-8",
    )
    with pytest.raises(ValueError, match="payload_sha256"):
        cache_source_identity(dataset, split, cache_dir=tmp_path)


def test_load_synthetic_bin_cache(tmp_path):
    # wiki-en-style int32 memmap + meta.json sidecar. An uncapped load stays mapped in the native
    # int32 dtype (no corpus-sized int64 copy); the exact values still round-trip.
    p = cache_path("wiki-en", "test", suffix="bin", cache_dir=tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.randint(0, 100277, size=(300,), dtype=np.int32)
    arr.tofile(p)
    (p.parent / (p.name + ".meta.json")).write_text(json.dumps({"n_tokens": 300, "dtype": "int32"}))
    out = load_cached_tokens("wiki-en", "test", cache_dir=tmp_path)
    assert out.dtype == torch.int32 and out.shape == (300,)
    assert torch.equal(out, torch.from_numpy(arr))
    assert out.max() < 100277


def test_load_real_wikitext2_if_present():
    # uses the user's actual cache; skip cleanly if absent
    try:
        toks = load_cached_tokens("wikitext-2", "validation")
    except FileNotFoundError:
        pytest.skip("wikitext-2 validation cache not present")
    assert toks.dim() == 1 and toks.numel() > 1000 and toks.dtype == torch.long
    ds = TokenWindows(toks, seq_len=16)
    x, y = ds[0]
    assert x.shape == (16,) and y.shape == (16,)


def test_make_dataloader_eval_keeps_tail_and_is_sequential(monkeypatch):
    r"""Audit F1: an eval loader (shuffle=False, drop_last=False) must read the WHOLE split in a
    deterministic order, while the train loader (defaults shuffle=True, drop_last=True) drops the
    partial last batch. RED against the old make_dataloader, which had no drop_last param and
    hardcoded drop_last=True for every split."""
    from torch.utils.data import RandomSampler, SequentialSampler

    import vfe3.data.datasets as dsmod

    monkeypatch.setattr(dsmod, "load_cached_tokens", lambda *a, **k: torch.arange(30))
    monkeypatch.setattr(dsmod, "cache_source_identity", lambda *a, **k: {
        "format": "pt", "tokenizer_tag": "tiktoken", "size_bytes": 30,
        "sha256": "0" * 64, "meta": None, "meta_sha256": None,
    })
    # Evaluation covers 29 transitions in 8 windows; training keeps its legacy 7 full windows.
    seq_len, batch_size = 4, 3

    val = dsmod.make_dataloader("ds", "validation", seq_len, batch_size, shuffle=False, drop_last=False)
    n_full = len(val.dataset)
    assert n_full % batch_size != 0                  # precondition: there IS a partial tail
    assert val.drop_last is False
    assert isinstance(val.sampler, SequentialSampler)
    assert sum(b[0].shape[0] for b in val) == n_full                       # tail kept

    train = dsmod.make_dataloader("ds", "train", seq_len, batch_size)       # defaults: shuffle, drop_last
    assert train.drop_last is True
    assert isinstance(train.sampler, RandomSampler)
    assert sum(b[0].shape[0] for b in train) == n_full - (n_full % batch_size)   # tail dropped
