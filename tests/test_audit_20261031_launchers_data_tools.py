r"""Remediation of the 2026-10-31 audit, package ``launchers-data``: the local learner, the evaluation tools and the
token caches.

A117 (memory-local-learner-8): the backpropagation arm under the exponential chart logs the spectral norm of the
transition :math:`G = e^L`, not of :math:`L`.

A211 (memory-local-learner-10): the n-gram reference builds each order's tables once and scores every smoothing
constant from the same lookups, with the per-constant sums of the former per-constant loop, bit for bit.

A362 (memory-local-learner-9): the reuse identity of stages X, A/B/C and S hashes the stage's runner.

A382 (train-loop-lifecycle-7): the averaging report names the steps of ``best_model.pt`` and ``final_model.pt`` from
``summary.json`` and flags inputs whose states are bitwise equal.

A282 (evaluation-metrics-4): the protocol report records the minimum context :math:`N - s + 1` of every window after
the first and the :math:`N - s` first-window targets below it.

A358 (evaluation-metrics-5): every re-scored entry records its streams, scored targets and realized precision, and the
difference table flags entries that did not score the same targets.

A178 (data-pipeline-2): an uncapped ``.bin`` cache stays mapped (copy-on-write) through ``make_dataloader``.

A355 (data-pipeline-4): the word builder writes int32 ids for a new build, read back as the int64 build's windows.

A312 (data-pipeline-3): the GPT-2 builder applies the reconstructed recipe, its segmented encoding equals the
whole-text encoding, and an existing cache is compared rather than overwritten.

A357 (dead-code-8): the zero-reference export is gone and the retained test oracles are labeled.

Tiny CPU fixtures only.
"""
from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterator, List

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _one_thread() -> None:
    torch.set_num_threads(1)


@pytest.fixture
def float64_default() -> Iterator[None]:
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


# ---------------------------------------------------------------------------------------------------------------- A117

@pytest.mark.parametrize("link_chart", ["exp", "matrix"])
def test_backprop_logs_the_spectral_norm_of_the_transition(link_chart: str, float64_default: None) -> None:
    from vfe3.local_learner import TreeSensor, init_runs, train_backprop

    tree = TreeSensor(3)
    streams = torch.randint(0, tree.V, (4, 25), generator=torch.Generator().manual_seed(0))
    history: List[Dict[str, Any]] = []
    prm = train_backprop(init_runs(2, tree.V, [0, 1], "cpu"), streams, tree, chunk=4, lr=5e-2, link_chart=link_chart,
                         log=history.append, log_every=1)
    returned = torch.linalg.matrix_norm(prm.G, 2)
    torch.testing.assert_close(torch.tensor(history[-1]["G_spectral"], dtype=torch.float64), returned,
                               rtol=1e-12, atol=0.0)
    # The first row follows one update from G = 0.5 I, so it sits near 0.5, not near |log 0.5| = 0.693.
    assert all(abs(value - 0.5) < 0.15 for value in history[0]["G_spectral"])


# ---------------------------------------------------------------------------------------------------------------- A211

def _ngram_nll_per_constant(train: np.ndarray, streams: np.ndarray, V: int, order: int,
                            k: float) -> "tuple[float, int]":
    r"""The pre-2026-10-31 reference: every table rebuilt and every context total recounted for one constant."""
    train = train.astype(np.int64)
    tables = {}
    for o in range(1, order + 1):
        idx = np.zeros(len(train) - o + 1, dtype=np.int64)
        for j in range(o):
            idx = idx * V + train[j:len(train) - o + 1 + j]
        tables[o] = np.bincount(idx, minlength=V ** o).reshape((V,) * o).astype(np.float64)
    total, count = 0.0, 0
    for row in streams.astype(np.int64):
        for t in range(1, len(row)):
            o = min(order, t + 1)
            ctx = tuple(row[t - o + 1:t])
            counts = tables[o][ctx] if o > 1 else tables[1]
            total -= math.log((counts[row[t]] + k) / (counts.sum() + k * V))
            count += 1
    return total, count


@pytest.mark.parametrize("order", [1, 2, 3])
def test_ngram_grid_reproduces_the_per_constant_sums_exactly(order: int, monkeypatch: pytest.MonkeyPatch) -> None:
    import vfe3.local_learner.baselines as baselines

    rng = np.random.default_rng(order)
    V = 6
    train = rng.integers(0, V, 400)
    streams = rng.integers(0, V, (3, 17))
    ks = [1e-3, 0.1, 1.0]
    builds: List[int] = []
    original = baselines._ngram_tables
    monkeypatch.setattr(baselines, "_ngram_tables", lambda *args: builds.append(1) or original(*args))
    grid = baselines.ngram_nll_grid(train, streams, V, order, ks)
    assert len(builds) == 1                                         # the tables of every order, built once
    assert grid == [_ngram_nll_per_constant(train, streams, V, order, k) for k in ks]
    assert [baselines.ngram_nll(train, streams, V, order, k) for k in ks] == grid


# ---------------------------------------------------------------------------------------------------------------- A362

@pytest.fixture
def launcher(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    r"""``local_learner_test`` with a tiny CONFIG, synthetic bytes and stubbed training and scoring."""
    import local_learner_test as llt
    from vfe3.local_learner import TreeSensor

    monkeypatch.setattr(llt, "CONFIG", json.loads(json.dumps(llt.CONFIG)))
    llt.CONFIG.update(out_dir=str(tmp_path / "results"), chunk=4, streams=4, log_every=0, B_dims=[2], C_dims=[2],
                      B_seeds=[0], arms=["blind"], C_arms=["blind"], B_train_bytes=200, B_val_bytes=60,
                      B_val_streams=2, pf_streams=1, pf_len=5, ngram_k_grid=[0.1, 1.0], X_d=2, X_arms=["blind"],
                      X_passes=1, W_points=2, local_schedule="floor20",
                      S_runs=[{"arm": "blind", "d": 2, "chunk": 4, "streams": 4, "stack": 1}])
    monkeypatch.setattr(llt, "DEVICE", torch.device("cpu"))
    monkeypatch.setattr(llt, "load_bytes", lambda split, n: (np.arange(n) * 7 % 8).astype(np.int64))
    monkeypatch.setattr(llt, "train_arm", lambda arm, prm, train, tree, log, chunk=None: prm)
    monkeypatch.setattr(llt, "scores", lambda prm, val, pf, tree, seed: [{"val_bits": 1.0}] * prm.R)
    monkeypatch.setattr(llt, "evaluate_filter", lambda prm, val, tree: (torch.zeros(prm.R, dtype=torch.float64), 1))
    llt._test_tree = TreeSensor(3)
    return llt


def _rows(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_stage_x_identity_hashes_its_runner(launcher: Any, float64_default: None) -> None:
    llt = launcher
    llt.CONFIG["stage"] = "X"
    llt.run_accel(llt._test_tree)
    rows = _rows(Path(llt.CONFIG["out_dir"]) / "results_X.jsonl")
    assert rows and {r["identity"]["source"] for r in rows} == {llt.source_identity(extra=(llt.run_accel,))}
    assert llt.source_identity(extra=(llt.run_accel,)) != llt.source_identity()


@pytest.mark.parametrize("stage", ["B", "C"])
def test_stage_b_and_c_identities_hash_their_runner(stage: str, launcher: Any, float64_default: None,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    llt = launcher
    llt.CONFIG["stage"] = stage
    grid_calls: List[int] = []
    original = llt.ngram_nll_grid
    monkeypatch.setattr(llt, "ngram_nll_grid", lambda *args: grid_calls.append(args[3]) or original(*args))
    llt.run_stage(stage, llt._test_tree)
    rows = _rows(Path(llt.CONFIG["out_dir"]) / f"results_{stage}.jsonl")
    assert rows and {r["identity"]["source"] for r in rows} == {llt.source_identity(extra=(llt.run_stage,))}
    assert grid_calls == ([1, 2, 3] if stage == "B" else [])       # one grid per order, not one per constant


def test_stage_s_identity_hashes_its_runner(launcher: Any, float64_default: None) -> None:
    llt = launcher
    llt.CONFIG["stage"] = "S"
    llt.run_scaling(llt._test_tree)
    rows = _rows(Path(llt.CONFIG["out_dir"]) / "results_S.jsonl")
    assert rows and {r["identity"]["source"] for r in rows} == {llt.source_identity(extra=(llt.run_scaling,))}


# ---------------------------------------------------------------------------------------------------------------- A382

def test_average_report_names_the_default_steps_and_flags_identical_inputs(tmp_path: Path) -> None:
    import shutil

    from tests.test_posthoc_20260928 import _fake_run
    from vfe3.checkpoint_average import CheckpointAverageOptions, run_checkpoint_average

    run_dir, cache_dir = _fake_run(tmp_path)
    options = dict(device="cpu", precision="fp32", cache_dir=str(cache_dir), splits=("validation",))
    messages: List[str] = []
    report = run_checkpoint_average(run_dir, options=CheckpointAverageOptions(**options), log=messages.append)
    assert report["checkpoint_steps"] == {"best_model.pt": 10, "final_model.pt": 20}     # from summary.json
    assert report["checkpoint_step_sources"] == {"best_model.pt": "summary.json best_step",
                                                 "final_model.pt": "summary.json evaluated_step"}
    assert report["identical_inputs"] == [] and not any("bitwise" in m for m in messages)

    shutil.copy(run_dir / "final_model.pt", run_dir / "final_copy.pt")
    messages.clear()
    report = run_checkpoint_average(run_dir, options=CheckpointAverageOptions(
        **options, checkpoints=("final_model.pt", "final_copy.pt"), out_name="copy_eval.json"), log=messages.append)
    assert report["identical_inputs"] == [["final_model.pt", "final_copy.pt"]]
    assert report["checkpoint_steps"] == {"final_model.pt": 20, "final_copy.pt": None}
    assert any("bitwise" in m and "final_copy.pt" in m for m in messages)


def test_bitwise_equal_groups_compare_bytes() -> None:
    from vfe3.checkpoint_average import bitwise_equal_groups

    a = {"w": torch.tensor([1.0, float("nan")]), "n": torch.tensor([3])}
    b = {"w": torch.tensor([1.0, float("nan")]), "n": torch.tensor([3])}
    c = {"w": torch.tensor([1.0, 2.0]), "n": torch.tensor([3])}
    d = {"w": torch.tensor([-0.0, float("nan")]), "n": torch.tensor([3])}
    assert bitwise_equal_groups([a, b, c, d], ["a", "b", "c", "d"]) == [["a", "b"]]


# ---------------------------------------------------------------------------------------------------------- A282, A358

def _contexts(N: int, s: int, T: int) -> List[List[int]]:
    r"""Context length of every scored target, per window, of the padded protocol windows over T tokens."""
    from vfe3.data.datasets import TokenWindows

    windows = TokenWindows(torch.arange(T), N, stride=s, pad_final=True)
    out = []
    for index in range(len(windows)):
        _, targets = windows[index]
        out.append([position + 1 for position in range(N) if int(targets[position]) != -100])
    return out


@pytest.mark.parametrize("N, s", [(8, 3), (8, 4), (4, 2), (6, 6)])
def test_protocol_context_fields_match_the_windows(N: int, s: int) -> None:
    from vfe3.protocol_rescore import protocol_context

    contexts = _contexts(N, s, 61)
    fields = protocol_context(N, s)
    assert fields["min_context_after_first_window"] == min(min(c) for c in contexts[1:]) == N - s + 1
    assert fields["first_window_targets_below_min_context"] == sum(
        1 for c in contexts[0] if c < N - s + 1) == N - s


def test_rescore_entries_record_their_streams_and_precision(tmp_path: Path) -> None:
    from tests.test_posthoc_20260928 import _fake_run
    from vfe3.protocol_rescore import ProtocolRescoreOptions, run_protocol_rescore

    run_dir, cache_dir = _fake_run(tmp_path)
    report = run_protocol_rescore(tmp_path / "out.json", options=ProtocolRescoreOptions(
        entries=((str(run_dir), "final_model.pt"),), seq_len=8, eval_stride=4, device="cpu", precision="fp32",
        cache_dir=str(cache_dir)), log=lambda _: None)
    entry = report["results"][0]
    assert entry["realized_amp_dtype"] is None and entry["fp32_matmul_precision"] == "highest"
    for split in ("validation", "test"):
        record = entry["splits"][split]
        assert record["dataset"] == "synthetic-period3"
        assert len(record["content_sha256"]) == 64 and record["n_tokens"] > 0
        assert record["scored_targets"] == record["n_tokens"] - 1
    assert report["protocol"]["min_context_after_first_window"] == 5
    assert report["protocol"]["first_window_targets_below_min_context"] == 4


def _entry(name: str, ce: float, sha: str, scored: int) -> Dict[str, Any]:
    split = {"ce": ce, "ppl": math.exp(ce), "dataset": "d", "content_sha256": sha, "n_tokens": 10,
             "scored_targets": scored}
    return {"run": name, "checkpoint": "final_model.pt", "splits": {"validation": dict(split)}}


def test_comparison_table_flags_entries_that_scored_other_targets() -> None:
    from vfe3.protocol_rescore import comparison_table

    same = comparison_table({"results": [_entry("a", 3.0, "0" * 64, 9), _entry("b", 3.5, "0" * 64, 9)]})
    assert len(same.splitlines()) == 3 and "0.5000" in same
    stream = comparison_table({"results": [_entry("a", 3.0, "0" * 64, 9), _entry("b", 3.5, "1" * 64, 9)]})
    assert "0.5000" not in stream and "content_sha256" in stream
    scored = comparison_table({"results": [_entry("a", 3.0, "0" * 64, 9), _entry("b", 3.5, "0" * 64, 8)]})
    assert "0.5000" not in scored and "scored_targets" in scored


# ---------------------------------------------------------------------------------------------------------------- A178

def test_bin_cache_stays_mapped_through_make_dataloader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vfe3.data.datasets as D

    n = 4096
    values = np.random.default_rng(0).integers(0, 100277, size=n, dtype=np.int32)
    path = D.cache_path("wiki-en", "validation", suffix="bin", cache_dir=tmp_path)
    values.tofile(path)
    Path(str(path) + ".meta.json").write_text(json.dumps({"n_tokens": n, "dtype": "int32"}), encoding="utf-8")
    clones: List[int] = []
    original_clone = torch.Tensor.clone
    modes: List[str] = []
    original_memmap = np.memmap

    def recording_clone(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.numel() == n:
            clones.append(n)
        return original_clone(self, *args, **kwargs)

    def recording_memmap(*args: Any, **kwargs: Any) -> Any:
        modes.append(kwargs.get("mode", "r+"))
        return original_memmap(*args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", recording_clone)
    monkeypatch.setattr(np, "memmap", recording_memmap)
    loader = D.make_dataloader("wiki-en", "validation", 16, 4, shuffle=False, drop_last=False, cache_dir=tmp_path,
                               vocab_size=100277)
    monkeypatch.undo()
    assert clones == [] and modes == ["c"]                          # mapped copy-on-write, no corpus-sized copy
    assert loader.dataset.tokens.dtype == torch.int32
    assert torch.equal(loader.dataset.tokens, torch.from_numpy(values))
    inputs, targets = loader.dataset[1]
    assert inputs.tolist() == values[16:32].tolist() and targets.tolist() == values[17:33].tolist()
    del loader, inputs, targets


# ---------------------------------------------------------------------------------------------------------------- A355

_FAKE_ROWS = {"train|0": [" the cat sat ", " the cat , the dog . "], "validation|0": [" the cat "],
              "test|0": [" dog . the "]}


@pytest.fixture
def word_builder(monkeypatch: pytest.MonkeyPatch) -> Any:
    import build_wikitext103_word_cache as builder

    monkeypatch.setattr(builder, "SPLIT_FILES",
                        {"train": ("train|0",), "validation": ("validation|0",), "test": ("test|0",)})
    monkeypatch.setattr(builder, "_read_rows", lambda name: list(_FAKE_ROWS[name]))
    return builder


def test_word_builder_writes_int32_ids_that_load_as_the_int64_build(word_builder: Any, tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    import vfe3.data.datasets as D

    monkeypatch.setattr(word_builder, "TOKEN_DTYPE", "int32")      # the click-to-run setting a new build reads
    word_builder.build(min_count=1, cache_dir=tmp_path / "new")
    word_builder.build(min_count=1, cache_dir=tmp_path / "old", token_dtype="int64")
    for split in ("train", "validation", "test"):
        new = torch.load(D.cache_path(word_builder.DATASET, split, cache_dir=tmp_path / "new"), weights_only=True)
        old = torch.load(D.cache_path(word_builder.DATASET, split, cache_dir=tmp_path / "old"), weights_only=True)
        assert new.dtype == torch.int32 and old.dtype == torch.int64 and new.tolist() == old.tolist()
    loaders = [D.make_dataloader(word_builder.DATASET, "train", 2, 1, shuffle=False, drop_last=False,
                                 cache_dir=tmp_path / name, vocab_size=3) for name in ("new", "old")]
    for index in range(len(loaders[0].dataset)):
        assert all(torch.equal(a, b) for a, b in zip(loaders[0].dataset[index], loaders[1].dataset[index]))
    with pytest.raises(ValueError, match="token_dtype"):
        word_builder.build(min_count=1, cache_dir=tmp_path / "bad", token_dtype="int16")


# ---------------------------------------------------------------------------------------------------------------- A312

_RAW_ROWS = {
    "train|0": ["", " = Valkyria = \n", "", " The game 's sequel @-@ series , with 1 @,@ 000 units @.@ \n",
                "  two leading spaces here \n", " 'quoted' start \n"],
    "train|1": ["", " Second shard row \n", " = = Section = = \n", " Year 2011 été \n"],
    "validation|0": ["", " The history of the city \n", " In 1923 , the company \n"],
    "test|0": ["", " Last row @-@ here \n"],
}


@pytest.fixture
def gpt2_builder(monkeypatch: pytest.MonkeyPatch) -> Any:
    pytest.importorskip("tiktoken")
    import build_wikitext103_gpt2_cache as builder

    monkeypatch.setattr(builder, "SPLIT_FILES",
                        {"train": ("train|0", "train|1"), "validation": ("validation|0",), "test": ("test|0",)})
    monkeypatch.setattr(builder, "_read_rows", lambda name: list(_RAW_ROWS[name]))
    return builder


def test_recipe_text_drops_empty_rows_joins_and_merges_markers(gpt2_builder: Any) -> None:
    text = gpt2_builder.recipe_text(["", " a @-@ b , 1 @,@ 0 @.@ 5 \n", "", " c \n"])
    assert text == " a-b , 1,0.5 \n\n\n c \n"


def test_segmented_encoding_equals_the_whole_text_encoding(gpt2_builder: Any) -> None:
    import tiktoken

    rows = [row for name in ("train|0", "train|1") for row in _RAW_ROWS[name]] * 3
    encoding = tiktoken.get_encoding("gpt2")
    whole = encoding.encode_ordinary(gpt2_builder.recipe_text(rows))
    for target in (1, 7, 40, 10_000):
        ids, n_chars = gpt2_builder.encode_rows(rows, segment_chars=target)
        assert ids.tolist() == whole and n_chars == len(gpt2_builder.recipe_text(rows))


def test_gpt2_builder_writes_a_verified_cache_and_compares_an_existing_one(gpt2_builder: Any, tmp_path: Path) -> None:
    import vfe3.data.datasets as D

    summary = gpt2_builder.build(cache_dir=tmp_path, dataset="wikitext-103", token_dtype="int32")
    assert summary["mode"] == "build"
    for split in ("train", "validation", "test"):
        assert summary["splits"][split]["status"] == "written"
        tokens = torch.load(D.cache_path("wikitext-103", split, cache_dir=tmp_path), weights_only=True)
        assert tokens.dtype == torch.int32
        identity = D.cache_source_identity("wikitext-103", split, cache_dir=tmp_path)
        assert identity["tokenizer_provenance_status"] == "manifest_verified"
        meta = json.loads(Path(str(D.cache_path("wikitext-103", split, cache_dir=tmp_path)) + ".meta.json")
                          .read_text(encoding="utf-8"))
        assert D.tokens_per_char("wikitext-103", split, cache_dir=tmp_path) == pytest.approx(
            meta["n_tokens"] / meta["n_chars_source"], rel=0.0, abs=1e-15)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    compared = gpt2_builder.build(cache_dir=tmp_path, dataset="wikitext-103", token_dtype="int32")
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before          # nothing written
    assert compared["mode"] == "compare"
    assert all(compared["splits"][split]["status"] == "matches_existing" for split in ("train", "validation", "test"))
    with pytest.raises(SystemExit, match="GPT-2"):
        gpt2_builder.build(cache_dir=tmp_path, dataset="wiki-en")


def test_gpt2_builder_reports_an_existing_cache_built_by_another_recipe(gpt2_builder: Any, tmp_path: Path,
                                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    import vfe3.data.datasets as D

    monkeypatch.setattr(gpt2_builder, "MARKERS", ())                # an existing cache built without merged markers
    gpt2_builder.build(cache_dir=tmp_path, dataset="wikitext-103", token_dtype="int64")
    monkeypatch.undo()
    monkeypatch.setattr(gpt2_builder, "SPLIT_FILES",
                        {"train": ("train|0", "train|1"), "validation": ("validation|0",), "test": ("test|0",)})
    monkeypatch.setattr(gpt2_builder, "_read_rows", lambda name: list(_RAW_ROWS[name]))
    compared = gpt2_builder.build(cache_dir=tmp_path, dataset="wikitext-103", publish_manifest=True)
    assert compared["splits"]["validation"]["status"] == "matches_existing"     # the split has no marker
    assert compared["splits"]["test"]["status"] == "differs_from_existing"
    assert compared["splits"]["test"]["first_difference"] is not None
    assert compared["manifests_published"] is False                 # published only when every split matches


# ---------------------------------------------------------------------------------------------------------------- A242

def test_tied_maps_warning_states_the_self_cost_of_the_routing_law() -> None:
    import warnings

    from tests.profiles import tiny_config

    with pytest.warns(UserWarning, match=r"D_ii = KL\(q_i \|\| q_i\) = 0 every row's minimum"):
        tiny_config(tie_comparison_maps=True, beta_attention_prior="causal", relational_routing="gaussian_kl")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tiny_config(tie_comparison_maps=True, beta_attention_prior="causal", relational_routing="noisy_key")
    messages = [str(w.message) for w in caught if "tie_comparison_maps=True" in str(w.message)]
    assert messages and all("= 0 every row" not in m and "broadening" in m and "causal_noself" in m
                            for m in messages)


def test_template_init_knobs_name_both_readers_when_inert() -> None:
    from tests.profiles import test_config

    inactive = test_config(use_block_mlp=True, block_mlp_mode="coordinate").inactive_fields()
    six = ["block_mlp_inference", "block_mlp_template_init_variance", "block_mlp_template_init",
           "block_mlp_template_init_pull", "block_mlp_cavi_steps", "block_mlp_em_weight"]
    assert [key for key in inactive if key in six] == six                   # report order unchanged
    for key in six[1:4]:
        assert "'prototype_agents' or 'cue_result_memory'" in inactive[key]
    for key in (six[0], *six[4:]):
        assert inactive[key] == "only read by block_mlp_mode='prototype_agents' with use_block_mlp=True"


# ---------------------------------------------------------------------------------------------------------------- A357

def test_zero_reference_export_is_removed_and_test_oracles_are_labeled() -> None:
    import vfe3.checkpoint_average as checkpoint_average
    import vfe3.divergence as divergence
    import vfe3.families.base as base

    assert not hasattr(base, "has_per_coord_functional") and "has_per_coord_functional" not in divergence.__all__
    assert "List" not in vars(checkpoint_average)
    import paper_arms
    from vfe3.data.word_vocab import vocabulary_size_for_min_count
    from vfe3.geometry.groups import check_admissible
    from vfe3.geometry.lie_ops import lie_bracket_coords
    from vfe3.geometry.phi_preconditioner import set_phi_group_differential_mode, uspace_spatial_covector
    from vfe3.geometry.transport import get_transport
    from vfe3.metrics import estep_residuals, group_gauge_invariant, per_head_complex_phase, per_head_gauge_invariants
    from vfe3.metrics_schema import unregistered_columns
    from vfe3.model.block_mlp import block_norm_jacobian
    from vfe3.model.edge_links import edge_link_matrices, latent_receiver_energy
    from vfe3.model.relational_operators import spread_tile_bytes
    from vfe3.posthoc.interventions import frame_role_composite_check
    from vfe3.relational_scoring import diagonal_moments_kernel, projection_tile_bytes
    from vfe3.semantics.gaussians import hellinger_sq_diag, log_expected_likelihood
    from vfe3.semantics.vocab import custom_vocab

    for function in (set_phi_group_differential_mode, uspace_spatial_covector, estep_residuals, group_gauge_invariant,
                     per_head_complex_phase, per_head_gauge_invariants, edge_link_matrices, latent_receiver_energy,
                     frame_role_composite_check, diagonal_moments_kernel, projection_tile_bytes, spread_tile_bytes,
                     base.divergence_functionals, base.divergence_functionals_per_coord, vocabulary_size_for_min_count,
                     check_admissible, lie_bracket_coords, get_transport, unregistered_columns, block_norm_jacobian,
                     hellinger_sq_diag, log_expected_likelihood, custom_vocab, paper_arms.find_complete_run):
        assert "referenced only by the tests" in " ".join((inspect.getdoc(function) or "").split()), \
            function.__qualname__
