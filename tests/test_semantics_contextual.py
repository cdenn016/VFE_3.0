r"""Contextual (per-layer) section of :mod:`vfe3.semantics` on tiny models.

Every model here is tiny (vocabulary 24, K=4, two heads, two layers, context 8), saved to a
temporary run directory and reopened with ``load_semantic_run``; token streams are synthetic and
injected by replacing ``SemanticRun.token_stream`` on the instance.
"""
from tests.profiles import test_config

import json
import re
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from vfe3.config import serialize_config
from vfe3.model.model import VFEModel
from vfe3.semantics import contextual as C
from vfe3.semantics.common import AnalogyQuestion, AnalysisContext, SectionResult, diagonal_redundancy_status, to_jsonable
from vfe3.semantics.gaussians import kl_diag
from vfe3.semantics.run import load_semantic_run
from vfe3.semantics.vocab import custom_vocab


STRINGS = ["<unk>", "king", "queen", "man", "woman", "France", "Paris", "Germany", "Berlin", "dog",
           "cat", "red", "blue", "one", "two", "the", ",", "bank", "river", "money", "Italy", "Rome",
           "green", "three"]
V = len(STRINGS)
ID = {s: i for i, s in enumerate(STRINGS)}
COUNTS = np.array([0, 50, 40, 60, 55, 30, 25, 28, 22, 20, 18, 45, 12, 100, 90, 500, 400, 35, 15, 14,
                   16, 13, 11, 70], dtype=np.int64)
N_CTX = 8


@pytest.fixture(autouse=True)
def _fast_figures(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Keep the section's style but save at 40 dpi with one font (glyph measurement dominates)."""
    original = C.apply_style

    def low_dpi_style() -> None:
        original()
        import matplotlib.pyplot as plt
        plt.rcParams.update({"savefig.dpi": 40, "font.family": ["DejaVu Sans"],
                             "font.sans-serif": ["DejaVu Sans"]})

    monkeypatch.setattr(C, "apply_style", low_dpi_style)


def _tiny_model(**over) -> "tuple[VFE3Config, VFEModel]":
    options = dict(vocab_size=V, embed_dim=4, n_heads=2, max_seq_len=N_CTX, n_layers=2, e_phi_lr=0.0,
                   norm_type_block="none", layernorm_affine=False,           # pinned: independent of
                   skip_belief_sigma_update=False)                           # the config defaults
    options.update(over)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = test_config(**options)
        torch.manual_seed(0)
        model = VFEModel(cfg).eval()
    with torch.no_grad():                                              # move tables off their init
        model.prior_bank.sigma_log_embed.normal_(0.0, 0.5)
        if hasattr(model.prior_bank, "decode_sigma_log_embed"):
            model.prior_bank.decode_sigma_log_embed.normal_(0.0, 0.5)
            model.prior_bank.decode_mu_embed.normal_()
        model.prior_bank.decode_log_scale.fill_(0.3)
        if cfg.decode_unigram_prior:
            model.prior_bank.set_unigram_log_prior(torch.from_numpy(COUNTS).float())
    return cfg, model


def _stream(seed: int, n: int = 3000, exclude: tuple = ()) -> torch.Tensor:
    allowed = np.array([i for i in range(1, V) if i not in exclude], dtype=np.int64)
    return torch.from_numpy(np.random.default_rng(seed).choice(allowed, size=n))


def _run(tmp_path: Path, name: str, streams: Dict[str, Optional[torch.Tensor]], **over):
    cfg, model = _tiny_model(**over)
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True)
    torch.save({"model_state": model.state_dict(), "config": serialize_config(cfg),
                "selection_data_identity": None}, run_dir / "best_model.pt")
    (run_dir / "config.json").write_text(
        json.dumps({"config": serialize_config(cfg), "dataset": "wikitext-103"}), encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = load_semantic_run(run_dir, vocab=custom_vocab(STRINGS))

    def token_stream(split: str) -> torch.Tensor:
        stream = streams.get(split)
        if stream is None:
            raise FileNotFoundError(f"no synthetic {split!r} stream")
        return stream

    run.token_stream = token_stream                                     # instance override
    return run


def _build(run):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run.build_model()


def _ctx(tmp_path: Path, run, **options) -> AnalysisContext:
    out = tmp_path / "out"
    base = {"n_boot": 100, "contextual_points": 256}
    base.update(options)
    lexicon = {"people": [ID[w] for w in ("king", "queen", "man", "woman")],
               "places": [ID[w] for w in ("France", "Paris", "Germany", "Berlin", "Italy", "Rome")],
               "animals": [ID["dog"], ID["cat"]],
               "colors": [ID[w] for w in ("red", "blue", "green")],
               "numbers": [ID[w] for w in ("one", "two", "three")]}
    q = lambda c, a, b, x, y: AnalogyQuestion(c, ID[a], ID[b], ID[x], ID[y])   # noqa: E731
    analogies = [q("capital-common-countries", "France", "Paris", "Germany", "Berlin"),
                 q("capital-common-countries", "Germany", "Berlin", "Italy", "Rome"),
                 q("capital-common-countries", "Italy", "Rome", "France", "Paris"),
                 q("family", "man", "woman", "king", "queen")]
    return AnalysisContext(
        out_dir=out, figures_dir=out / "figures", tables_dir=out / "tables", vocab=run.vocab,
        candidate_ids=np.arange(1, V, dtype=np.int64), counts=run.counts, lexicon=lexicon, relations={},
        analogies=analogies, curated_analogies=[q("family", "man", "woman", "king", "queen")],
        similarity={}, hypernyms=[], axes={}, probe_ids=[], ambiguous_ids=[ID["bank"]], coverage={},
        seed=0, device="cpu", threads=1, options=base)


# ----------------------------------------------------------------------------------------------
# The readout the lens uses is the model's own
# ----------------------------------------------------------------------------------------------

READOUT_CONFIGS = [
    dict(decode_mode="linear"),                                                      # Sigma moves
    dict(attention_mode="projected_predictive", decode_mode="diagonal_chunked", untie_decode_bank=True),
    dict(norm_type_final="layernorm"),
    dict(family="gaussian_full", decode_mode="full_chunked"),
    dict(encode_mode="canonical_content_projected", decode_mode="full_chunked", lambda_alpha_mode="constant"),
]


@pytest.mark.parametrize("over", READOUT_CONFIGS, ids=["linear", "projected_kl", "final_layernorm",
                                                       "full_family", "canonical_projected"])
def test_lens_at_last_layer_reproduces_forward_logits(over):
    cfg, model = _tiny_model(**over)
    x = torch.from_numpy(np.random.default_rng(3).integers(1, V, size=(3, N_CTX)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = C._forward_layers(model, x, return_logits=True)
        with torch.no_grad():
            belief, want = model.forward_beliefs(x, return_logits=True)          # no capture
        layers, frame = out["layers"], out["canonical_frame"]
        assert len(layers) == cfg.n_layers + 1
        got = C._readout(model, layers[-1].mu, layers[-1].sigma, frame)
        torch.testing.assert_close(got, want, rtol=0.0, atol=1e-6)
        final_mu = layers[-1].mu if model.final_norm is None else model.final_norm(layers[-1].mu, layers[-1].sigma)
        torch.testing.assert_close(final_mu, belief.mu, rtol=0.0, atol=0.0)
        first = C._readout(model, layers[0].mu, layers[0].sigma, frame)
        assert (first - want).abs().max() > 1e-3                        # the layers really differ
        if cfg.norm_type_final != "none":                                 # the final norm is load-bearing
            raw = model._decode_belief_with_context(layers[-1].mu, layers[-1].sigma, canonical_frame=frame)
            assert (raw - want).abs().max() > 1e-3
    if cfg.encode_mode == "canonical_content_projected":
        assert frame is not None                                          # decode needs the realized frame


# ----------------------------------------------------------------------------------------------
# Bank
# ----------------------------------------------------------------------------------------------

def test_collect_bank_windows_targets_layers_and_frozen_detection(tmp_path):
    stream = _stream(1)
    run = _run(tmp_path, "moving", {"validation": stream}, decode_mode="linear")   # transported: Sigma moves
    model = _build(run)
    ctx = _ctx(tmp_path, run)
    bank = C.collect_bank(run, model, ctx, n_tokens=40, batch_size=3)
    N, L = N_CTX, run.n_layers
    assert bank.n == 40 and bank.windows.shape == (5, N) and bank.mu.shape == (L + 1, 40, 4)
    s = stream.numpy()
    assert np.all(bank.offsets % (N + 1) == 0) and np.unique(bank.offsets).size == 5
    for w, o in enumerate(bank.offsets):
        np.testing.assert_array_equal(bank.windows[w], s[o:o + N])
        np.testing.assert_array_equal(bank.targets[w * N:(w + 1) * N], s[o + 1:o + N + 1])
    np.testing.assert_array_equal(bank.token_ids, bank.windows.reshape(-1))
    np.testing.assert_array_equal(bank.mu[0], run.mu(bank.token_ids).astype(np.float32))
    np.testing.assert_allclose(bank.sigma0, np.exp(run.log_var(bank.token_ids)), rtol=1e-6)
    with torch.no_grad():
        belief, _ = model.forward_beliefs(torch.from_numpy(bank.windows))
    np.testing.assert_allclose(bank.mu[L], belief.mu.reshape(-1, 4).numpy(), rtol=0, atol=1e-6)
    assert bank.sigma is not None and not bank.sigma_frozen and not bank.phi_moves
    np.testing.assert_allclose(bank.sigma[L], belief.sigma.reshape(-1, 4).numpy(), rtol=0, atol=1e-6)
    again = C.collect_bank(run, model, ctx, n_tokens=40, batch_size=5)
    np.testing.assert_array_equal(again.offsets, bank.offsets)          # seeded
    np.testing.assert_allclose(again.mu, bank.mu, rtol=0, atol=1e-6)    # batch size does not matter

    frozen_run = _run(tmp_path, "frozen", {"validation": stream}, attention_mode="projected_predictive")
    frozen = C.collect_bank(frozen_run, _build(frozen_run), _ctx(tmp_path, frozen_run), n_tokens=16)
    assert frozen.sigma is None and frozen.sigma_frozen and frozen.diagnostics["sigma_frozen_config"] is True
    assert any("Σ is frozen" in note for note in frozen.notes)
    short = C.collect_bank(frozen_run, _build(frozen_run), _ctx(tmp_path, frozen_run), n_tokens=10 ** 6)
    assert short.windows.shape[0] == 3000 // (N + 1)                     # capped, and said so
    assert any("disjoint windows" in note for note in short.notes)
    with pytest.raises(ValueError):
        C.collect_bank(run, model, ctx, seq_len=N + 1)


# ----------------------------------------------------------------------------------------------
# Re-split on the model's exact-symmetry failure
# ----------------------------------------------------------------------------------------------

SYMMETRY_ERROR = "projected covariance must be symmetric for zero-jitter Cholesky"


class _FakeModel:
    r"""Raises the model's exact-symmetry ValueError for batches larger than 2 and for any batch
    holding a poisoned window (first token in ``poison``); otherwise returns ``mu = x``."""

    def __init__(self, poison: tuple = (), message: str = SYMMETRY_ERROR) -> None:
        self.poison, self.message, self.calls = set(poison), message, []

    def forward_beliefs(self, x, return_logits=False, capture=None):
        self.calls.append(int(x.shape[0]))
        if x.shape[0] > 2 or any(int(row[0]) in self.poison for row in x):
            raise ValueError(self.message)
        belief = SimpleNamespace(mu=x.float()[..., None], sigma=torch.ones(*x.shape, 1), phi=None)
        capture["diagnostic"] = {"initial_belief": belief, "layer_outputs": [belief], "encoded_belief": belief}
        return belief, None


class _FlakyModel:
    r"""A real model whose forward raises the exact-symmetry ValueError for batches larger than
    ``max_batch`` and for any batch holding the window ``poison``; everything else is delegated."""

    def __init__(self, model, max_batch: int = 2, poison: Optional[np.ndarray] = None) -> None:
        self._model, self.max_batch, self.calls = model, max_batch, []
        self.poison = None if poison is None else torch.as_tensor(np.asarray(poison, dtype=np.int64))

    def __getattr__(self, name):
        return getattr(self._model, name)

    def forward_beliefs(self, x, **kwargs):
        self.calls.append(int(x.shape[0]))
        if x.shape[0] > self.max_batch or (self.poison is not None and any(torch.equal(r, self.poison) for r in x)):
            raise ValueError(SYMMETRY_ERROR)
        return self._model.forward_beliefs(x, **kwargs)


def test_forward_split_halves_failed_batches_and_skips_single_failures():
    x = torch.arange(7)[:, None].repeat(1, 3)                           # window i is (i, i, i)
    model = _FakeModel()
    log = C._SplitLog()
    pieces = C._forward_split(model, x, log, offset=10)
    assert model.calls == [7, 3, 1, 2, 4, 2, 2] and log.resplit_batches == 3 and log.skipped == []
    assert [(a, b) for a, b, _ in pieces] == [(0, 1), (1, 3), (3, 5), (5, 7)]
    mu = torch.cat([out["layers"][-1].mu[..., 0] for _, _, out in pieces])
    torch.testing.assert_close(mu, x.float(), rtol=0.0, atol=0.0)     # every window, in order

    log = C._SplitLog()
    pieces = C._forward_split(_FakeModel(poison=(4,)), x, log, offset=10)
    assert log.skipped == [14] and log.resplit_batches == 4             # 7, 3, 4 and (3, 4) were halved
    assert [(a, b) for a, b, _ in pieces] == [(0, 1), (1, 3), (3, 4), (5, 7)]
    assert log.record() == {"resplit_batches": 4, "oom_resplits": 0, "skipped_windows": 1,
                            "skipped_window_index": [14]}
    with pytest.raises(ValueError, match="other failure"):              # any other error propagates
        C._forward_split(_FakeModel(message="other failure"), x, C._SplitLog())


class _OutOfMemoryModel(_FakeModel):
    r"""Raises ``torch.cuda.OutOfMemoryError`` for batches larger than ``max_batch``."""

    def __init__(self, max_batch: int = 2) -> None:
        super().__init__()
        self.max_batch = max_batch

    def forward_beliefs(self, x, return_logits=False, capture=None):
        if x.shape[0] > self.max_batch:
            self.calls.append(int(x.shape[0]))
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (planted)")
        return super().forward_beliefs(x, return_logits=return_logits, capture=capture)


def test_forward_split_halves_out_of_memory_batches():
    x = torch.arange(7)[:, None].repeat(1, 3)
    model = _OutOfMemoryModel(max_batch=2)
    log = C._SplitLog()
    pieces = C._forward_split(model, x, log, offset=0)
    assert log.oom_resplits == 3 and log.resplit_batches == 0 and log.skipped == []
    assert [(a, b) for a, b, _ in pieces] == [(0, 1), (1, 3), (3, 5), (5, 7)]
    mu = torch.cat([out["layers"][-1].mu[..., 0] for _, _, out in pieces])
    torch.testing.assert_close(mu, x.float(), rtol=0.0, atol=0.0)
    assert log.record()["oom_resplits"] == 3
    with pytest.raises(torch.cuda.OutOfMemoryError):                    # a single window cannot be split
        C._forward_split(_OutOfMemoryModel(max_batch=0), x[:1], C._SplitLog())


def test_device_budgets_and_host_sync_free_helpers(tmp_path):
    cpu, cuda = torch.device("cpu"), torch.device("cuda")
    assert C._auto_batch_windows(cpu, 256) == 8 and C._auto_batch_windows(cuda, 256) == 16
    assert C._auto_batch_windows(cuda, 10 ** 5) == 1                    # at least one window
    assert C._auto_lens_chunk_tokens(cpu, 50257, 256) == 1024
    gib = 1 << 30
    # 0.25 * 8 GiB / (4 bytes * 50257 * 8 live (B, N, V) tensors) = 1335 tokens
    assert C._auto_lens_chunk_tokens(cuda, 50257, 256, free_bytes=8 * gib) == (2 * gib) // (4 * 50257 * 8)
    assert C._auto_lens_chunk_tokens(cuda, 50257, 256, free_bytes=1 << 20) == 256    # at least one window
    assert C._auto_lens_chunk_tokens(cuda, 100, 256, free_bytes=64 * gib) == 16384   # capped
    run = _run(tmp_path, "budgets", {})
    opts = C._resolved_options(_ctx(tmp_path, run), run)
    assert opts["contextual_batch_size"] == 8 and opts["occurrence_batch_size"] == 8
    assert opts["lens_chunk_tokens"] == 1024
    explicit = C._resolved_options(_ctx(tmp_path, run, contextual_batch_size=3, lens_chunk_tokens=16), run)
    assert explicit["contextual_batch_size"] == 3 and explicit["lens_chunk_tokens"] == 16

    a = torch.randn(2, 3, 5)
    assert bool(C._equal_flag(a, a.clone(), cpu)) and not bool(C._equal_flag(a, a + 1.0, cpu))
    assert not bool(C._equal_flag(a, a[:1], cpu)) and bool(C._equal_flag(None, None, cpu))
    assert not bool(C._equal_flag(a, None, cpu))
    b = torch.randn(2, 3, 5)
    torch.testing.assert_close(C._rowdot(a, b), (a * b).sum(-1), rtol=1e-6, atol=1e-6)


def test_bank_lens_and_occurrences_survive_the_symmetry_failure(tmp_path):
    train, bank_pos, _ = _planted_train()
    run = _run(tmp_path, "flaky", {"validation": _stream(1), "train": train}, decode_mode="linear")   # Σ moves
    model = _build(run)
    ctx = _ctx(tmp_path, run)
    ref = C.collect_bank(run, model, ctx, n_tokens=40, batch_size=5)
    flaky = C.collect_bank(run, _FlakyModel(model), ctx, n_tokens=40, batch_size=5)
    np.testing.assert_allclose(flaky.mu, ref.mu, rtol=0, atol=1e-6)    # batch-independent per window
    assert flaky.diagnostics["forward_resplit"]["resplit_batches"] > 0

    N = N_CTX
    poisoned = C.collect_bank(run, _FlakyModel(model, poison=ref.windows[1]), ctx, n_tokens=40, batch_size=5)
    keep = np.array([0, 2, 3, 4])
    rows = (keep[:, None] * N + np.arange(N)).reshape(-1)
    assert poisoned.windows.shape == (4, N) and poisoned.n == 32
    np.testing.assert_array_equal(poisoned.windows, ref.windows[keep])
    np.testing.assert_array_equal(poisoned.offsets, ref.offsets[keep])
    np.testing.assert_array_equal(poisoned.targets, ref.targets[rows])
    np.testing.assert_array_equal(poisoned.seq_index, np.repeat(np.arange(4), N))
    np.testing.assert_allclose(poisoned.mu, ref.mu[:, rows], rtol=0, atol=1e-6)
    np.testing.assert_allclose(poisoned.sigma, ref.sigma[:, rows], rtol=0, atol=1e-6)
    rec = poisoned.diagnostics["forward_resplit"]
    assert rec["skipped_windows"] == 1 and rec["skipped_stream_offsets"] == [int(ref.offsets[1])]

    lens_ref = C.logit_lens(run, model, poisoned, ctx, max_tokens=40, chunk_tokens=40)
    lens = C.logit_lens(run, _FlakyModel(model, poison=ref.windows[1]), ref, ctx, max_tokens=40, chunk_tokens=40)
    np.testing.assert_array_equal(lens["window_index"], keep)
    assert lens["n_windows"] == 4 and lens["n_windows_requested"] == 5 and lens["forward_resplit"]["skipped_windows"] == 1
    for key in ("ce", "acc", "kl"):
        for got, want in zip(lens[key], lens_ref[key]):
            assert abs(got["mean"] - want["mean"]) < 1e-5
    sig = C._sigma_dynamics(ref, lens, ctx)                              # lens tokens align with bank rows
    assert "spearman_entropy" in sig

    words = [ID["bank"], ID["blue"]]
    kw = dict(per_word=4, context_len=N, budget_tokens=N * 20, per_word_overrides={ID["bank"]: 9})
    occ_ref = C.collect_occurrences(run, model, ctx, words, batch_size=64, **kw)
    occ = C.collect_occurrences(run, _FlakyModel(model), ctx, words, **kw)   # default batch 8, re-split
    assert occ["forward_resplit"]["resplit_batches"] > 0 and occ["forward_resplit"]["skipped_windows"] == 0
    for w in words:
        np.testing.assert_allclose(occ["vectors"][w], occ_ref["vectors"][w], rtol=0, atol=1e-6)
    s = train.numpy()
    p = int(bank_pos[1])
    occ = C.collect_occurrences(run, _FlakyModel(model, poison=s[p - N + 1:p + 1]), ctx, words, **kw)
    assert occ["forward_resplit"]["skipped_stream_positions"] == [p]
    assert occ["n_occurrences"][ID["bank"]] == 8 and p not in set(occ["positions"][ID["bank"]])
    keep_b = occ_ref["positions"][ID["bank"]] != p
    np.testing.assert_allclose(occ["vectors"][ID["bank"]], occ_ref["vectors"][ID["bank"]][:, keep_b], rtol=0, atol=1e-6)
    assert occ["n_forward_tokens"] == occ_ref["n_forward_tokens"]


# ----------------------------------------------------------------------------------------------
# Logit lens
# ----------------------------------------------------------------------------------------------

def test_logit_lens_statistics_equal_direct_computation(tmp_path):
    stream = _stream(2)
    run = _run(tmp_path, "lens", {"validation": stream}, attention_mode="projected_predictive",
               decode_mode="diagonal_chunked", untie_decode_bank=True)
    model = _build(run)
    ctx = _ctx(tmp_path, run)
    bank = C.collect_bank(run, model, ctx, n_tokens=48)
    lens = C.logit_lens(run, model, bank, ctx, max_tokens=32, chunk_tokens=16)
    N, L = N_CTX, run.n_layers
    assert lens["n_windows"] == 4 and lens["n_tokens"] == 32
    assert lens["consistency"]["max_abs_logit_diff_final_vs_forward"] == 0.0
    assert lens["consistency"]["max_abs_mu_diff_rerun_vs_bank"] < 1e-6
    x = torch.from_numpy(bank.windows[:4])
    y = torch.from_numpy(bank.targets[:4 * N]).reshape(4, N)
    with torch.no_grad():
        _, final = model.forward_beliefs(x, return_logits=True)
        enc = model.prior_bank.encode(x)                                 # per_token: layer 0 = the prior
        first = model.prior_bank.decode(enc.mu, enc.sigma)
    for l, logits in ((L, final), (0, first)):
        ce = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1)).item()
        assert abs(lens["ce"][l]["mean"] - ce) < 1e-5
        acc = (logits.argmax(-1) == y).double().mean().item()
        assert abs(lens["acc"][l]["mean"] - acc) < 1e-12
        logp = torch.log_softmax(logits, -1)
        ent = -(logp.exp() * logp).sum(-1).mean().item()
        assert abs(lens["entropy"][l]["mean"] - ent) < 1e-5
    logp_L, logp_0 = torch.log_softmax(final, -1), torch.log_softmax(first, -1)
    kl0 = (logp_L.exp() * (logp_L - logp_0)).sum(-1).mean().item()
    assert abs(lens["kl"][0]["mean"] - kl0) < 1e-5
    assert lens["agree"][L]["mean"] == 1.0 and abs(lens["kl"][L]["mean"]) < 1e-7
    u = run.unigram_log_prior()
    assert abs(lens["unigram_ce"]["mean"] + u[y.numpy()].mean()) < 1e-6     # kappa = 1, u normalized (float32)
    for rec in lens["ce"]:
        assert rec["ci_lo"] <= rec["mean"] <= rec["ci_hi"] and rec["n_clusters"] == 4
    assert lens["example"]["top1"].shape == (L + 1, N)
    np.testing.assert_allclose(np.nanmean(lens["ce_by_position"][L]), lens["ce"][L]["mean"], rtol=1e-9)


def test_cluster_bootstrap_resamples_windows():
    rng = np.random.default_rng(0)
    values = np.repeat([0.0, 1.0, 2.0, 3.0], 5)                          # constant within a window
    clusters = np.repeat(np.arange(4), 5)
    ci = C._cluster_mean_ci(values, clusters, rng, n_boot=400)
    assert ci["mean"] == 1.5 and ci["n_clusters"] == 4 and ci["n"] == 20
    assert 0.0 <= ci["ci_lo"] < 1.5 < ci["ci_hi"] <= 3.0
    token_level = C._cluster_mean_ci(values, np.arange(20), rng, n_boot=400)
    assert (token_level["ci_hi"] - token_level["ci_lo"]) < (ci["ci_hi"] - ci["ci_lo"])   # clustering widens


# ----------------------------------------------------------------------------------------------
# Contextuality on a planted bank
# ----------------------------------------------------------------------------------------------

def _planted_bank(sigma_moves: bool) -> C.ContextualBank:
    rng = np.random.default_rng(5)
    n_seq, N, K = 6, 5, 3
    token_ids = np.tile(np.array([ID["king"], ID["queen"], ID["dog"], ID["king"], ID["queen"]]), n_seq)
    n = token_ids.size
    per_type = {t: rng.normal(size=K) for t in np.unique(token_ids)}
    mu0 = np.stack([per_type[t] for t in token_ids])                     # identical occurrences
    mu1 = rng.normal(size=(n, K))
    var0 = np.exp(rng.normal(size=(n, K)) * 0.3)
    sigma = np.stack([var0, var0 * np.exp(rng.normal(size=(n, K)) * 0.2)]).astype(np.float32) if sigma_moves else None
    return C.ContextualBank(
        token_ids=token_ids.astype(np.int64), targets=np.roll(token_ids, -1).astype(np.int64),
        positions=np.tile(np.arange(N), n_seq), seq_index=np.repeat(np.arange(n_seq), N),
        mu=np.stack([mu0, mu1]).astype(np.float32), sigma=sigma, phi_moves=False,
        sigma0=var0.astype(np.float32), windows=token_ids.reshape(n_seq, N).astype(np.int64),
        offsets=np.arange(n_seq, dtype=np.int64) * (N + 1), split="validation",
        sigma_frozen=not sigma_moves, sigma_kind="diagonal")


@pytest.mark.parametrize("sigma_moves", [False, True], ids=["frozen", "moving"])
def test_contextuality_planted_values(tmp_path, sigma_moves):
    run = _run(tmp_path, "cx", {})
    ctx = _ctx(tmp_path, run, min_type_occurrences=5, max_type_occurrences=50)
    bank = _planted_bank(sigma_moves)
    cx = C.contextuality(bank, run, ctx)
    assert cx["n_types"] == 3
    l0, l1 = cx["layers"]
    assert abs(l0["self_similarity"]["mean"] - 1.0) < 1e-12              # identical occurrences
    assert abs(l0["mev"]["mean"] - 1.0) < 1e-12
    assert l0["kl_to_stack_input"]["mean"] == 0.0 and l0["variance_share"]["mean"] == 0.0
    M = bank.mu[1].astype(np.float64)
    rows = np.flatnonzero(bank.token_ids == ID["dog"])                   # the 6-occurrence type
    Z = M[rows] / np.linalg.norm(M[rows], axis=1, keepdims=True)
    G = Z @ Z.T
    selfsim_dog = (G.sum() - np.trace(G)) / (rows.size * (rows.size - 1))
    per_type = []
    for t in np.unique(bank.token_ids):
        r = np.flatnonzero(bank.token_ids == t)
        Zt = M[r] / np.linalg.norm(M[r], axis=1, keepdims=True)
        Gt = Zt @ Zt.T
        per_type.append((Gt.sum() - np.trace(Gt)) / (r.size * (r.size - 1)))
    assert abs(l1["self_similarity"]["mean"] - np.mean(per_type)) < 1e-12
    assert selfsim_dog in per_type
    Zall = M / np.linalg.norm(M, axis=1, keepdims=True)
    Gall = Zall @ Zall.T
    baseline = (Gall.sum() - np.trace(Gall)) / (M.shape[0] * (M.shape[0] - 1))
    assert abs(l1["baseline_cosine"] - baseline) < 1e-12
    assert abs(l1["self_similarity_adjusted"]["mean"] - (np.mean(per_type) - baseline)) < 1e-12
    w0 = M[:5]
    mean_dir = w0.mean(0) / np.linalg.norm(w0.mean(0))
    intra0 = float((Zall[:5] @ mean_dir).mean())
    Wm = M.reshape(6, 5, 3).mean(1)
    intra = ((Zall.reshape(6, 5, 3) * (Wm / np.linalg.norm(Wm, axis=1, keepdims=True))[:, None]).sum(-1)).mean(1)
    assert abs(intra[0] - intra0) < 1e-12 and abs(l1["intra_similarity"]["mean"] - intra.mean()) < 1e-12
    var0 = bank.sigma0.astype(np.float64)
    mu0 = bank.mu[0].astype(np.float64)
    if sigma_moves:
        kl = kl_diag(M, np.log(bank.sigma[1].astype(np.float64)), mu0, np.log(var0))
        var1 = bank.sigma[1].astype(np.float64)
    else:
        kl = 0.5 * ((M - mu0) ** 2 / var0).sum(1)
        var1 = var0
    assert abs(l1["kl_to_stack_input"]["mean"] - kl.mean()) < 1e-10
    assert abs(l1["kl_to_stack_input"]["median"] - np.median(kl)) < 1e-10
    cov = M[rows].var(0)
    sbar = var1[rows].mean(0)
    share_dog = cov.sum() / (cov.sum() + sbar.sum())
    shares = []
    for t in np.unique(bank.token_ids):
        r = np.flatnonzero(bank.token_ids == t)
        c, sb = M[r].var(0), var1[r].mean(0)
        shares.append(c.sum() / (c.sum() + sb.sum()))
    assert share_dog in shares and abs(l1["variance_share"]["mean"] - np.mean(shares)) < 1e-12
    s = np.linalg.svd(M[rows], compute_uv=False)
    assert abs(C._mev(M[rows]) - s[0] ** 2 / (s ** 2).sum()) < 1e-15
    assert "content word" in l1["by_class"]
    json.dumps(to_jsonable(cx), allow_nan=False)


# ----------------------------------------------------------------------------------------------
# Occurrences, distillation and clouds
# ----------------------------------------------------------------------------------------------

def _planted_train(n: int = 1200) -> "tuple[torch.Tensor, np.ndarray, np.ndarray]":
    s = _stream(9, n=n, exclude=(ID["bank"], ID["dog"], ID["cat"])).numpy().copy()
    bank_pos = np.array([3, 50, 120, 200, 300, 400, 500, 600, 700, 800])
    dog_pos = np.array([60, 160])
    s[bank_pos] = ID["bank"]
    s[dog_pos] = ID["dog"]
    return torch.from_numpy(s), bank_pos, dog_pos


def test_collect_occurrences_windows_budget_and_chunking(tmp_path):
    train, bank_pos, dog_pos = _planted_train()
    run = _run(tmp_path, "occ", {"train": train}, attention_mode="projected_predictive")
    model = _build(run)
    ctx = _ctx(tmp_path, run)
    words = [ID["bank"], ID["dog"], ID["cat"], ID["blue"]]
    kw = dict(per_word=4, context_len=N_CTX, budget_tokens=N_CTX * 7, per_word_overrides={ID["bank"]: 6})
    occ = C.collect_occurrences(run, model, ctx, words, **kw)
    assert occ["n_occurrences"] == {ID["bank"]: 6, ID["dog"]: 1}
    assert occ["missing"] == [ID["cat"]] and occ["dropped_budget"] == [ID["blue"]]
    assert occ["partial"] == {ID["dog"]: 1} and occ["n_forward_tokens"] == 7 * N_CTX
    b = occ["positions"][ID["bank"]]
    assert set(b) <= set(bank_pos[1:]) and np.unique(b).size == 6        # position 3 lacks left context
    assert set(occ["positions"][ID["dog"]]) <= set(dog_pos)
    s = train.numpy()
    for w in (ID["bank"], ID["dog"]):
        ctxs = occ["contexts"][w]
        assert np.all(ctxs[:, -1] == w)
        for row, p in zip(ctxs, occ["positions"][w]):
            np.testing.assert_array_equal(row, s[p - N_CTX + 1:p + 1])
    assert occ["vectors"][ID["bank"]].shape == (run.n_layers + 1, 6, 4)
    with torch.no_grad():
        belief, _ = model.forward_beliefs(torch.from_numpy(occ["contexts"][ID["bank"]]))
    np.testing.assert_allclose(occ["vectors"][ID["bank"]][-1], belief.mu[:, -1].numpy(), rtol=0, atol=1e-6)
    np.testing.assert_array_equal(occ["vectors"][ID["bank"]][0],
                                  np.repeat(run.mu([ID["bank"]]).astype(np.float32), 6, axis=0))
    chunked = C.collect_occurrences(run, model, ctx, words, chunk=37, **kw)   # rank bookkeeping across chunks
    for w in (ID["bank"], ID["dog"]):
        np.testing.assert_array_equal(chunked["positions"][w], occ["positions"][w])
    assert any("TRAIN split" in note for note in occ["notes"])


def test_layerwise_semantics_and_token_clouds(tmp_path):
    train, _, _ = _planted_train(4000)
    run = _run(tmp_path, "sem", {"train": train}, attention_mode="projected_predictive")
    model = _build(run)
    ctx = _ctx(tmp_path, run)
    words, overrides = C.occurrence_words(ctx)
    overrides[ID["bank"]] = 9
    occ = C.collect_occurrences(run, model, ctx, words, per_word=3, context_len=N_CTX,
                                budget_tokens=N_CTX * 200, per_word_overrides=overrides)
    sem = C.layerwise_semantics(occ, run, ctx)
    assert len(sem["layers"]) == run.n_layers + 1
    assert sem["layer0_minus_static_max_abs"] < 1e-7                   # per_token: layer 0 = mu_v
    assert sem["layers"][0]["analogies"]["n_covered"] == 4
    assert sem["static_mu"]["analogies"]["n_covered"] == 4
    assert sem["layers"][0]["analogies"]["overall"]["3cosadd"]["mean"] == \
        sem["static_mu"]["analogies"]["overall"]["3cosadd"]["mean"]
    assert sem["decode"]["label"] == "linear decode rows" and "fields" in sem["decode"]
    assert ID["cat"] in occ["missing"]                                 # never occurs in the planted stream
    assert sem["layers"][-1]["fields"]["n_fields"] == 4                 # so "animals" keeps one word and drops out
    json.dumps(to_jsonable(sem), allow_nan=False)
    clouds = C.token_clouds(occ, ctx)
    cloud = clouds["words"][str(ID["bank"])]
    assert cloud["n"] == 9 and len(cloud["layers"]) == run.n_layers + 1
    assert cloud["layers"][-1]["pca"].shape == (9, 2)
    assert sum(c["size"] for c in cloud["clusters_last"]) == 9
    assert all("[bank]" in e for c in cloud["clusters_last"] for e in c["examples"])
    assert abs(cloud["layers"][0]["self_similarity"] - 1.0) < 1e-6       # prior: identical occurrences


def test_occurrence_words_priority(tmp_path):
    run = _run(tmp_path, "prio", {})
    ctx = _ctx(tmp_path, run)
    words, overrides = C.occurrence_words(ctx)
    assert words[0] == ID["bank"] and overrides == {ID["bank"]: 32}
    assert words[1:5] == [ID["man"], ID["woman"], ID["king"], ID["queen"]]
    assert len(words) == len(set(words))
    rest = words[5:]
    assert rest[0] == ID["dog"]                                         # lexicon round-robin: animals first
    assert rest[1] in (ID["France"], ID["Paris"], ID["Germany"], ID["Berlin"], ID["Italy"], ID["Rome"])


# ----------------------------------------------------------------------------------------------
# Section
# ----------------------------------------------------------------------------------------------

def _section_options() -> dict:
    return {"contextual_tokens": 64, "lens_tokens": 32, "lens_chunk_tokens": 16, "occurrence_tokens": N_CTX * 150,
            "occurrence_per_word": 3, "cloud_per_word": 9, "context_len": N_CTX, "n_boot": 100,
            "contextual_points": 64}


def test_run_contextual_section_end_to_end_moving_sigma(tmp_path):
    train, _, _ = _planted_train(4000)
    run = _run(tmp_path, "section", {"validation": _stream(4), "train": train}, decode_mode="linear")
    ctx = _ctx(tmp_path, run, **_section_options())
    threads = torch.get_num_threads()
    res = C.run_contextual_section(run, ctx)
    assert res.status == "ok", res.error
    assert torch.get_num_threads() == threads
    names = {p.stem for p in res.figures}
    assert names == {"logit_lens", "geometry", "cka", "contextuality", "layerwise_semantics", "token_clouds", "sigma"}
    assert all(p.is_file() for p in res.figures) and len(res.figures) == 14
    assert set(res.tables) == {"logit_lens", "layerwise_semantics", "token_clouds"}
    for name in res.tables:
        assert (ctx.tables_dir / "contextual" / f"{name}.md").is_file()
    assert "| man:woman::king:queen |" in res.tables["layerwise_semantics"]      # curated ranks, one row
    assert res.tables["logit_lens"].count("top-1 @ layer") == run.n_layers + 1
    m = res.metrics
    assert m["bank"]["sigma_frozen_measured"] is False
    assert m["logit_lens"]["consistency"]["max_abs_logit_diff_final_vs_forward"] == 0.0
    assert "arrays" not in m["logit_lens"]
    for key in ("geometry", "cka", "contextuality", "sigma", "layerwise_semantics", "token_clouds"):
        assert m[key].get("status") not in ("failed", "skipped"), (key, m[key])
    json.dumps(to_jsonable(m), allow_nan=False)
    json.dumps(to_jsonable(res.headline), allow_nan=False)
    assert any(k.startswith("contextual: lens CE") for k in res.headline)
    assert m["failed_steps"] == [] and m["forward_resplits"]["resplit_batches"] == 0
    assert m["forward_resplits"]["oom_resplits"] == 0 and m["device"] == "cpu"
    assert m["options"]["contextual_batch_size"] == 8 and m["options"]["occurrence_batch_size"] == 8
    assert m["options"]["lens_chunk_tokens"] == 16                     # explicit value kept
    sem = m["layerwise_semantics"]
    head = res.headline["contextual: distilled 3CosAdd (ONLY-B) final layer / static μ, retrieval among the "
                        "collected words"]
    assert f"layer {run.n_layers}, retrieval among {sem['n_words']} words" in head
    assert f"{sem['layers'][-1]['analogies']['n_covered']} of 4 questions covered" in head
    assert head.startswith(f"{sem['layers'][-1]['analogies']['overall']['3cosadd']['mean']:.3f} "
                           f"({sem['layers'][-1]['analogies']['overall']['only_b']['mean']:.3f}) / ")

    summary = C.contextual_summary(json.loads(json.dumps(to_jsonable(m), allow_nan=False)))
    sentences = re.findall(r"[^.]*?\.(?=\s|$)", summary)
    assert 3 <= len(sentences) <= 6, summary
    lens = m["logit_lens"]
    assert f"{lens['ce'][0]['mean']:.3f} nats at the stack input" in summary
    assert f"{lens['unigram_ce']['mean']:.3f} nats for the context-free unigram readout" in summary
    assert f"{m['cka']['prior_mu']['debiased'][-1]:.3f} with the static prior mean" in summary
    assert f"among the {sem['n_words']} collected words" in summary and "ONLY-B" in summary
    assert "Occurrence clouds of 1 ambiguous word (bank)" in summary
    assert C.contextual_summary({}) == ""


def test_headline_reports_the_final_layer_with_pool_and_only_b():
    def rec(layer, acc, only_b, purity, null):
        return {"layer": layer, "analogies": {"n_total": 4, "n_covered": 3,
                                              "overall": {"n": 3, "3cosadd": {"mean": acc}, "only_b": {"mean": only_b}}},
                "fields": {"purity": purity, "null_mean": null, "n_words": 12}}

    sem = {"n_words": 30, "layers": [rec(0, 0.9, 0.1, 0.8, 0.2), rec(1, 0.2, 0.05, 0.3, 0.25)],
           "static_mu": rec(None, 0.5, 0.4, 0.6, 0.2)}
    res = SectionResult(name="contextual")
    C._headline(res, None, None, None, sem, SimpleNamespace(sigma_frozen=True, phi_moves=False))
    h = res.headline
    assert h["contextual: distilled 3CosAdd (ONLY-B) final layer / static μ, retrieval among the collected words"] == (
        "0.200 (0.050) / 0.500 (0.400); layer 1, retrieval among 30 words, 3 of 4 questions covered")
    assert h["contextual: distilled field purity final layer / static μ"] == (
        "0.300 / 0.600; layer 1, permutation null 0.250, 12 lexicon words")
    assert not any("best layer" in k for k in h)


def test_section_records_resplits_skips_and_every_failed_step(tmp_path, monkeypatch):
    train, bank_pos, _ = _planted_train(4000)
    run = _run(tmp_path, "flaky_section", {"validation": _stream(4), "train": train}, decode_mode="linear")
    p = int(bank_pos[1])
    flaky = _FlakyModel(_build(run), poison=train.numpy()[p - N_CTX + 1:p + 1])
    monkeypatch.setattr(run, "build_model", lambda device="cpu": flaky)

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(C, "token_clouds", boom)
    monkeypatch.setattr(C, "_plot_geometry", boom)
    monkeypatch.setattr(C, "_lens_table", boom)
    monkeypatch.setattr(C.vectors, "field_clustering", boom)
    ctx = _ctx(tmp_path, run, **{**_section_options(), "lens_chunk_tokens": 32})   # 4-window lens forwards
    res = C.run_contextual_section(run, ctx)
    assert res.status == "ok", res.error
    m = res.metrics
    fr = m["forward_resplits"]
    assert fr["resplit_batches"] > 0 and fr["skipped_windows"] == 1
    assert fr["by_step"]["occurrences"]["skipped_stream_positions"] == [p]
    assert fr["by_step"]["bank"]["resplit_batches"] > 0 and fr["by_step"]["logit_lens"]["resplit_batches"] > 0
    assert any(f"{fr['resplit_batches']} batches were re-run in smaller sub-batches because the model's "
               "exact-symmetry check failed under CPU BLAS summation order; 1 windows were skipped" in n
               for n in res.notes)
    steps = {f["step"]: f["error"] for f in m["failed_steps"]}
    assert steps["token_clouds"] == "RuntimeError: boom"
    assert steps["figure geometry"] == "RuntimeError: boom" and steps["table logit_lens"] == "RuntimeError: boom"
    assert steps["occurrences forward (1 single windows skipped)"] == f"ValueError: {SYMMETRY_ERROR}"
    for l in range(run.n_layers + 1):
        assert steps[f"layerwise_semantics.layers[{l}].fields"] == "RuntimeError: boom"
    assert steps["layerwise_semantics.static_mu.fields"] == "RuntimeError: boom"
    assert any("contextual layerwise_semantics.layers[0].fields failed" in n for n in res.notes)
    assert "logit_lens" not in res.tables and "geometry" not in {p.stem for p in res.figures}
    json.dumps(to_jsonable(m), allow_nan=False)


def test_section_records_a_failed_occurrence_step(tmp_path, monkeypatch):
    train, _, _ = _planted_train()
    run = _run(tmp_path, "occ_fail", {"validation": _stream(6), "train": train}, attention_mode="projected_predictive")

    def boom(*args, **kwargs):
        raise RuntimeError("occurrence boom")

    monkeypatch.setattr(C, "collect_occurrences", boom)
    res = C.run_contextual_section(run, _ctx(tmp_path, run, **_section_options()))
    assert res.status == "ok", res.error
    assert res.metrics["occurrences"]["status"] == "failed"
    assert {"step": "occurrences", "error": "RuntimeError: occurrence boom"} in res.metrics["failed_steps"]
    assert any("contextual occurrences failed: RuntimeError: occurrence boom" in n for n in res.notes)


@pytest.mark.parametrize("over", [dict(), dict(norm_type_block="layernorm"),
                                  dict(attention_mode="projected_predictive")], ids=["default", "layernorm", "projected"])
def test_contextuality_gauge_note_follows_the_shared_predicate(tmp_path, over):
    run = _run(tmp_path, "gauge", {}, **over)
    cx = C.contextuality(_planted_bank(False), run, _ctx(tmp_path, run))
    status, reason = diagonal_redundancy_status(run.cfg)
    assert cx["diagonal_redundancy"] == {"status": status, "reason": reason}
    note = next(n for n in cx["notes"] if n.startswith("cosine-based terms"))
    expected = {"broken": "is not a redundancy of this run", "exact": "an exact redundancy of this run",
                "assumed": "assumed but not verified"}[status]
    assert expected in note and (status == "exact" or reason in note)
    if over.get("norm_type_block") == "layernorm":
        assert status == "broken"


def test_run_contextual_section_frozen_sigma_without_train_stream(tmp_path):
    run = _run(tmp_path, "frozen_section", {"validation": _stream(6)}, attention_mode="projected_predictive")
    ctx = _ctx(tmp_path, run, **_section_options())
    res = C.run_contextual_section(run, ctx)
    assert res.status == "ok", res.error
    assert {p.stem for p in res.figures} == {"logit_lens", "geometry", "cka", "contextuality"}
    assert res.metrics["occurrences"]["status"] == "skipped"
    assert res.metrics["sigma"]["status"] == "skipped"
    assert any("Σ is frozen" in n for n in res.notes) and any("occurrence analyses skipped" in n for n in res.notes)
    json.dumps(to_jsonable(res.metrics), allow_nan=False)

    no_data = _run(tmp_path, "no_validation", {}, attention_mode="projected_predictive")
    skipped = C.run_contextual_section(no_data, _ctx(tmp_path, no_data, **_section_options()))
    assert skipped.status == "skipped" and any("no cached" in n for n in skipped.notes)
