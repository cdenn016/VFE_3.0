r"""Strided sliding-window evaluation (``cfg.eval_stride``).

With the default disjoint windows (``stride == seq_len``) the first target of every held-out window
is predicted from one token of context, the second from two, and so on, so the reported cross-entropy
is inflated by that starvation -- the more so the shorter ``seq_len`` is. Setting
``eval_stride < max_seq_len`` selects the standard sliding-window evaluation: each window still runs
on its full context, but only its final ``stride`` targets are scored and the overlapping prefix is
masked to ``-100``, so every transition is counted EXACTLY ONCE at ``seq_len - stride`` tokens of
context or better.

The invariant these tests defend is the exactly-once contract: whatever the stride, the number of
scored targets equals the number of transitions in the stream, each exactly once. The default path
(``eval_stride=None``) must stay byte-identical to the disjoint windows it replaces.
"""

import pytest
import torch
from torch.utils.data import DataLoader

import ablation
import train_vfe3
from vfe3.config import VFE3Config
from vfe3.data.datasets import TokenWindows


def _scored_transitions(windows: TokenWindows) -> list:
    r"""Every (input token, target token) transition the dataset actually scores, in order."""
    scored = []
    for i in range(len(windows)):
        inputs, targets = windows[i]
        for position in range(windows.seq_len):
            if int(targets[position]) != -100:
                scored.append((int(inputs[position]), int(targets[position])))
    return scored


# --- the exactly-once contract ----------------------------------------------------------

@pytest.mark.parametrize(
    "n_tokens,seq_len,stride",
    [
        (100, 8, 8), (100, 8, 4), (100, 8, 2), (100, 8, 1),      # divides and does not divide
        (101, 8, 3), (33, 8, 5), (1000, 128, 64),                # ragged tail, production-ish shape
        (9, 8, 8), (9, 8, 2), (17, 16, 4),                       # degenerate: one window only
    ],
)
def test_every_transition_is_scored_exactly_once(n_tokens, seq_len, stride):
    tokens = torch.arange(n_tokens, dtype=torch.long)
    windows = TokenWindows(tokens, seq_len, stride=stride, pad_final=True)
    scored = _scored_transitions(windows)
    # On this stream token t is followed by t+1, so transition t is the pair (t, t+1).
    assert scored == [(t, t + 1) for t in range(n_tokens - 1)]


def test_scored_count_is_stride_independent_end_to_end():
    # Through a real DataLoader, which is how evaluate() consumes it.
    tokens = torch.arange(400, dtype=torch.long)
    counts = set()
    for stride in (8, 4, 2, 1):
        loader = DataLoader(TokenWindows(tokens, 8, stride=stride, pad_final=True),
                            batch_size=4, shuffle=False, drop_last=False)
        counts.add(sum(int((targets != -100).sum()) for _, targets in loader))
    assert counts == {399}                      # 400 tokens -> 399 transitions, at every stride


def test_overlap_prefix_is_masked_on_every_window_but_the_first():
    # The prefix is CONTEXT, not padding: the inputs stay real tokens, only the targets are ignored,
    # so evaluate() runs the model on the whole window and the masked positions inform the
    # prediction without entering the average.
    tokens = torch.arange(100, dtype=torch.long)
    windows = TokenWindows(tokens, 8, stride=3, pad_final=True)
    assert windows.overlap_prefix == 5

    inputs, targets = windows[0]
    assert int((targets == -100).sum()) == 0                     # window 0 scores its whole row

    for i in range(1, len(windows) - 1):                         # skip the padded tail window
        inputs, targets = windows[i]
        assert bool((targets[:5] == -100).all())
        assert bool((targets[5:] != -100).all())
        assert bool((inputs == torch.arange(i * 3, i * 3 + 8)).all())   # real context, not zeros


def test_iterating_does_not_mutate_the_token_stream():
    # ``.to(torch.long)`` is a no-op for an int64 stream, so the returned targets are a VIEW into
    # the corpus; masking in place would corrupt every later window.
    tokens = torch.arange(200, dtype=torch.long)
    reference = tokens.clone()
    windows = TokenWindows(tokens, 8, stride=4, pad_final=True)
    for i in range(len(windows)):
        windows[i]
    assert torch.equal(tokens, reference)


# --- the default path is untouched ------------------------------------------------------

def test_default_stride_is_byte_identical_to_disjoint_windows():
    tokens = torch.arange(200, dtype=torch.long)
    implicit = TokenWindows(tokens, 8, pad_final=True)                      # stride=None
    explicit = TokenWindows(tokens, 8, stride=8, pad_final=True)
    assert len(implicit) == len(explicit) == (199 + 7) // 8                 # ceil(transitions/L)
    assert implicit.overlap_prefix == 0
    for i in range(len(implicit)):
        assert torch.equal(implicit[i][0], explicit[i][0])
        assert torch.equal(implicit[i][1], explicit[i][1])


def test_disjoint_windows_mask_nothing_but_the_padded_tail():
    tokens = torch.arange(100, dtype=torch.long)
    windows = TokenWindows(tokens, 8, stride=8, pad_final=True)
    for i in range(len(windows) - 1):
        assert int((windows[i][1] == -100).sum()) == 0


# --- fail-closed --------------------------------------------------------------------------

def test_stride_above_seq_len_is_rejected_under_the_eval_contract():
    # A gap between consecutive windows drops transitions outright, and no target mask recovers them.
    tokens = torch.arange(100, dtype=torch.long)
    with pytest.raises(ValueError, match="stride <= seq_len"):
        TokenWindows(tokens, 8, stride=12, pad_final=True)


def test_train_contract_still_allows_a_stride_above_seq_len():
    # Training shuffles and drops its tail, so exactly-once never applied to it; a sparse stride is a
    # legitimate subsample there.
    tokens = torch.arange(100, dtype=torch.long)
    assert len(TokenWindows(tokens, 8, stride=12, pad_final=False)) == (100 - 8 - 1) // 12 + 1


def _cfg(**kw) -> VFE3Config:
    base = dict(vocab_size=16, embed_dim=4, max_seq_len=8, n_layers=1, n_heads=1,
                gauge_group="glk", n_e_steps=1)
    base.update(kw)
    return VFE3Config(**base)


def test_eval_stride_defaults_to_none():
    assert _cfg().eval_stride is None


@pytest.mark.parametrize("bad", [0, -1, 9, 2.5, "4"])
def test_eval_stride_rejects_out_of_range_and_non_int(bad):
    with pytest.raises(ValueError, match="eval_stride"):
        _cfg(eval_stride=bad)


def test_eval_stride_accepts_the_whole_valid_range():
    for stride in range(1, 9):
        assert _cfg(eval_stride=stride).eval_stride == stride


# --- driver wiring ------------------------------------------------------------------------

def test_select_loader_strides_eval_splits_only(monkeypatch):
    captured = {}

    def fake_make_dataloader(dataset, split, seq_len, batch_size, **kw):
        captured[split] = kw
        return ("loader", split)

    monkeypatch.setattr(train_vfe3, "make_dataloader", fake_make_dataloader)
    cfg = VFE3Config(eval_stride=64)
    for split in ("train", "validation", "test"):
        train_vfe3._select_loader("wikitext-103", cfg, split=split)

    assert captured["train"]["stride"] is None                   # train never strides
    assert captured["validation"]["stride"] == 64
    assert captured["test"]["stride"] == 64


def test_select_loader_default_passes_no_stride(monkeypatch):
    captured = {}

    def fake_make_dataloader(dataset, split, seq_len, batch_size, **kw):
        captured[split] = kw
        return ("loader", split)

    monkeypatch.setattr(train_vfe3, "make_dataloader", fake_make_dataloader)
    for split in ("train", "validation", "test"):
        train_vfe3._select_loader("wikitext-103", VFE3Config(), split=split)
    assert all(captured[split]["stride"] is None for split in captured)


def test_ablation_get_loader_strides_eval_only_and_keys_its_cache(monkeypatch):
    # The memo key MUST carry the stride: a sweep over eval_stride would otherwise be served the
    # first arm's loader and every arm would silently report the same windows.
    built = []

    def fake_make_dataloader(dataset, split, seq_len, batch_size, **kw):
        built.append((split, kw.get("stride")))
        return object()                                          # non-None sentinel get_loader caches

    monkeypatch.setattr(ablation, "make_dataloader", fake_make_dataloader)
    monkeypatch.setattr(ablation, "cache_source_identity", lambda dataset, split: {
        "format": "pt", "tokenizer_tag": "fixture", "size_bytes": len(split),
        "sha256": "0" * 64, "meta": None, "meta_sha256": None,
    })
    ablation._LOADER_CACHE.clear()
    try:
        first = ablation.get_loader("wikitext-103", 16, 4, "validation", eval_stride=8)
        again = ablation.get_loader("wikitext-103", 16, 4, "validation", eval_stride=8)
        other = ablation.get_loader("wikitext-103", 16, 4, "validation", eval_stride=4)
        ablation.get_loader("wikitext-103", 16, 4, "train", eval_stride=8)
    finally:
        ablation._LOADER_CACHE.clear()

    assert first is again                                        # same stride -> cache hit
    assert other is not first                                    # different stride -> distinct loader
    assert built == [("validation", 8), ("validation", 4), ("train", None)]


# --- what the metric actually changes -----------------------------------------------------

def test_strided_eval_removes_the_head_of_window_context_starvation():
    r"""On a stream whose structure needs context, the disjoint windows score strictly worse.

    A period-5 stream is predictable from one token of context and unpredictable from none, so the
    only positions a trained model gets wrong are the starved heads of each disjoint window. Halving
    the stride halves how many such heads exist per scored token, so the CE must FALL while the
    scored-token count stays fixed -- the signature of removing starvation rather than of a better
    model or of double counting.
    """
    from vfe3.model.model import VFEModel
    from vfe3.train import evaluate

    seq_len = 8
    cfg = VFE3Config(vocab_size=12, embed_dim=4, max_seq_len=seq_len, n_layers=1, n_heads=1,
                     gauge_group="glk", n_e_steps=2, batch_size=8)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    tokens = torch.tensor([3, 7, 1, 9, 4]).repeat(160)[:800].clone()

    train_loader = DataLoader(TokenWindows(tokens, seq_len), batch_size=8,
                              shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    model.train()
    for _ in range(4):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            _, loss, _ = model(inputs, targets)
            loss.backward()
            optimizer.step()

    results = {}
    for stride in (seq_len, seq_len // 2):
        loader = DataLoader(TokenWindows(tokens, seq_len, stride=stride, pad_final=True),
                            batch_size=8, shuffle=False, drop_last=False)
        assert sum(int((t != -100).sum()) for _, t in loader) == tokens.numel() - 1
        results[stride] = evaluate(model, loader, device=torch.device("cpu"))["ce"]

    assert results[seq_len // 2] < results[seq_len]


def test_scaling_get_loader_strides_eval_only_and_keys_its_cache(monkeypatch):
    # The third driver reporting held-out numbers. If it ignored cfg.eval_stride, a scaling run and a
    # train_vfe3 run under one config would publish two different metrics under one name.
    import scaling

    built = []

    def fake_make_dataloader(dataset, split, seq_len, batch_size, **kw):
        built.append((split, kw.get("stride")))
        return object()

    monkeypatch.setattr(scaling, "make_dataloader", fake_make_dataloader)
    scaling._LOADER_CACHE.clear()
    try:
        first = scaling.get_loader("wikitext-103", 16, 4, "validation", eval_stride=8)
        again = scaling.get_loader("wikitext-103", 16, 4, "validation", eval_stride=8)
        other = scaling.get_loader("wikitext-103", 16, 4, "test", eval_stride=4)
        scaling.get_loader("wikitext-103", 16, 4, "train", eval_stride=8)
    finally:
        scaling._LOADER_CACHE.clear()

    assert first is again
    assert other is not first
    assert built == [("validation", 8), ("test", 4), ("train", None)]


def test_figure_stream_is_deliberately_not_strided(monkeypatch):
    # The report loader feeds diagnostic figures, not a metric; overlapping windows would make
    # consecutive figure batches near-duplicates. Pinned so a later consistency sweep does not
    # "fix" it into the metric contract by accident.
    import vfe3.data.datasets as datasets_module
    from vfe3.viz import report

    captured = {}

    def fake_make_dataloader(dataset, split, seq_len, batch_size, **kw):
        captured.update(kw)
        return object()

    monkeypatch.setattr(datasets_module, "make_dataloader", fake_make_dataloader)
    report._build_loader("wikitext-103", _cfg(eval_stride=4), "validation")
    assert "stride" not in captured
