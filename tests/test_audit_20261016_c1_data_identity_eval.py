r"""Code identity, short training streams, word-cache provenance and deferred evaluation checks, 2026-10-16 deep
audit, lane C1 (findings 78, 79, 80 and 125).

F78: the package code identity strips only ``config`` and UPPER_CASE top-level assignments from the root drivers,
so the lowercase launcher toggle ``fully_gauge_covariant`` counted as executable code and flipping it discarded a
resumed checkpoint's best-model selection. It is declarative now, and every name ``ablation.py``'s own runner hash
treats as declarative is declarative for the package identity too.

F79: a train stream whose windows do not fill one ``drop_last`` batch gave ``len(loader) == 0`` and ``train()``
died with a bare ``StopIteration``. It now refuses the loader before step 0 and names the minimum stream length;
an unsized loader that yields nothing after re-iteration raises a ``RuntimeError``.

F80: the word-cache builder wrote neither the vocabulary digest nor the ``<cache>.provenance.json`` manifest the
loader checks, so neither check ran on a cache the repository builds.

F125: evaluation forwards ran outside ``deferred_host_checks`` and paid one host synchronization per raise-only
certificate. Each attempt of the evaluation forward on the replay ladder now runs inside the deferral scope; the
metrics are unchanged and a deferred verdict still reaches the ladder.

Tiny CPU models only (K=4, two heads, N=4, vocab 7; the deferral test runs ``tests.profiles.reference_config``).
"""
import ast
import contextlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
import torch
from torch.utils.data import DataLoader

import build_wikitext103_word_cache as builder
import vfe3.contracts as contracts
import vfe3.data.datasets as datasets_mod
import vfe3.run_artifacts as ra
import vfe3.train as T
from tests.test_audit2_20261011_training import _cfg
from vfe3.data.datasets import TokenWindows, cache_path, cache_source_identity, word_vocabulary_sha256
from vfe3.data.word_vocab import WordVocabulary, vocabulary_path
from vfe3.model.model import VFEModel

CPU = torch.device("cpu")
REPOSITORY = Path(ra.__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _one_thread() -> None:
    torch.set_num_threads(1)


# ------------------------------------------------------------------------------------------------ F78

def _flipped(source: bytes, name: str) -> bytes:
    line = f"{name} = False".encode("utf-8")
    assert source.count(line) == 1, name
    return source.replace(line, f"{name} = True".encode("utf-8"))


@pytest.mark.parametrize("driver", ["train_vfe3.py", "ablation.py"])
def test_flipping_fully_gauge_covariant_keeps_the_driver_logic_identity(driver: str) -> None:
    source = (REPOSITORY / driver).read_bytes()
    assert ra._driver_logic_bytes(source) == ra._driver_logic_bytes(_flipped(source, "fully_gauge_covariant"))


def test_a_lowercase_code_assignment_still_changes_the_driver_identity() -> None:
    base = b"import logging\nlogger = logging.getLogger('a')\nfully_gauge_covariant = False\n"
    edited = b"import logging\nlogger = logging.getLogger('b')\nfully_gauge_covariant = True\n"
    assert ra._driver_logic_bytes(base) != ra._driver_logic_bytes(edited)


def test_every_ablation_declarative_name_is_declarative_for_the_package_identity() -> None:
    tree = ast.parse((REPOSITORY / "ablation.py").read_text(encoding="utf-8"))
    names = next(
        ast.literal_eval(statement.value) for statement in tree.body
        if isinstance(statement, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_DECLARATIVE_ABLATION_NAMES"
                for target in statement.targets))
    assert "fully_gauge_covariant" in names
    for name in sorted(names):
        statement = ast.parse(f"{name} = 1").body[0]
        assert ra._is_declarative_driver_statement(statement), name


# ------------------------------------------------------------------------------------------------ F79

def _train(cfg: Any, loader: Any, n_steps: int) -> List[float]:
    torch.manual_seed(5)
    model = VFEModel(cfg)
    return T.train(model, loader, cfg, n_steps=n_steps, grad_clip=cfg.grad_clip, log_interval=0, eval_interval=0,
                   device=CPU, logger=logging.getLogger("audit-c1-short-stream"), generate_samples=False)


def test_train_refuses_a_stream_shorter_than_one_batch_per_epoch() -> None:
    cfg = _cfg(batch_size=4)
    seq_len = cfg.max_seq_len
    # Per-epoch origins fit floor((T - L) / s) windows once T >= L + s, so one batch of B needs T >= L + B s.
    minimum = seq_len + cfg.batch_size * seq_len
    short = DataLoader(TokenWindows(torch.arange(minimum - 1) % cfg.vocab_size, seq_len),
                       batch_size=cfg.batch_size, shuffle=False, drop_last=True)
    with pytest.raises(ValueError, match=rf"at least {minimum} tokens"):
        _train(cfg, short, n_steps=2)
    enough = DataLoader(TokenWindows(torch.arange(minimum) % cfg.vocab_size, seq_len),
                        batch_size=cfg.batch_size, shuffle=False, drop_last=True)
    assert len(_train(cfg, enough, n_steps=2)) == 2


def test_train_names_an_unsized_loader_that_cannot_be_cycled() -> None:
    cfg = _cfg()
    tokens = torch.arange(cfg.batch_size * cfg.max_seq_len).reshape(cfg.batch_size, cfg.max_seq_len) % cfg.vocab_size
    one_shot = iter([(tokens, tokens.roll(-1, dims=1))])                   # no __len__, one batch, no restart
    with pytest.raises(RuntimeError, match="yielded no batch"):
        _train(cfg, one_shot, n_steps=3)


# ------------------------------------------------------------------------------------------------ F80

_ROWS: Dict[Tuple[str, int], List[str]] = {
    ("train", 0): [" the cat sat ", " the cat , the dog . "],
    ("validation", 0): [" the cat "],
    ("test", 0): [" dog . "],
}


@pytest.fixture()
def fake_corpus(monkeypatch: pytest.MonkeyPatch) -> Dict[Tuple[str, int], List[str]]:
    rows = {key: list(value) for key, value in _ROWS.items()}
    monkeypatch.setattr(builder, "SPLIT_FILES",
                        {"train": ("train|0",), "validation": ("validation|0",), "test": ("test|0",)})

    def read_rows(file_name: str) -> List[str]:
        split, index = file_name.split("|")
        return list(rows[(split, int(index))])

    monkeypatch.setattr(builder, "_read_rows", read_rows)
    datasets_mod._WORD_VOCABULARY_CACHE.clear()
    return rows


def _meta(dataset: str, split: str, root: Path) -> Dict[str, Any]:
    return json.loads(Path(str(cache_path(dataset, split, suffix="pt", cache_dir=root)) + ".meta.json")
                      .read_text(encoding="utf-8"))


def test_builder_records_the_vocabulary_digest_and_a_verified_provenance_manifest(
        tmp_path: Path, fake_corpus: Dict[Tuple[str, int], List[str]]) -> None:
    builder.build(min_count=1, cache_dir=tmp_path)
    vocabulary = WordVocabulary.load(vocabulary_path(builder.DATASET, tag="word", cache_dir=tmp_path))
    for split in builder.SPLIT_FILES:
        assert _meta(builder.DATASET, split, tmp_path)["vocabulary_sha256"] == word_vocabulary_sha256(vocabulary.words)
        identity = cache_source_identity(builder.DATASET, split, cache_dir=tmp_path)
        assert identity["tokenizer_provenance_status"] == "manifest_verified", split
        assert identity["tokenizer_provenance"]["tokenizer_vocab_size"] == vocabulary.vocab_size


def test_loader_rejects_a_same_size_reordered_vocabulary_of_a_built_cache(
        tmp_path: Path, fake_corpus: Dict[Tuple[str, int], List[str]]) -> None:
    builder.build(min_count=1, cache_dir=tmp_path)
    sidecar = vocabulary_path(builder.DATASET, tag="word", cache_dir=tmp_path)
    vocabulary = WordVocabulary.load(sidecar)
    words = list(vocabulary.words)
    words[1], words[2] = words[2], words[1]                                 # same size, different id order
    sidecar.unlink()
    WordVocabulary(words, min_count=vocabulary.min_count, eos=vocabulary.eos).save(sidecar)
    datasets_mod._WORD_VOCABULARY_CACHE.clear()
    with pytest.raises(ValueError, match="vocabulary_sha256"):
        datasets_mod.make_dataloader(builder.DATASET, "test", 2, 1, shuffle=False, drop_last=False,
                                     cache_dir=tmp_path)


def test_forced_rebuild_publishes_a_manifest_for_the_new_payload(
        tmp_path: Path, fake_corpus: Dict[Tuple[str, int], List[str]]) -> None:
    builder.build(min_count=1, cache_dir=tmp_path)
    before = cache_source_identity(builder.DATASET, "validation", cache_dir=tmp_path)
    fake_corpus[("validation", 0)] = [" the the cat cat "]
    builder.build(min_count=1, cache_dir=tmp_path, force=True)
    after = cache_source_identity(builder.DATASET, "validation", cache_dir=tmp_path)
    assert after["sha256"] != before["sha256"]
    assert after["tokenizer_provenance_status"] == "manifest_verified"
    assert after["tokenizer_provenance"]["payload_sha256"] == after["sha256"]




# ------------------------------------------------------------------------------------------------ F125

def _counted_host_checks(monkeypatch: pytest.MonkeyPatch) -> Dict[str, int]:
    r"""Count every ``host_check`` call as immediate (outside a deferral scope) or queued (inside one)."""
    original = contracts.host_check
    counts = {"immediate": 0, "queued": 0}

    def counting(flags: torch.Tensor, verdict: Any, *args: Any, **kwargs: Any) -> None:
        counts["immediate" if contracts._DEFERRED_CHECKS is None else "queued"] += 1
        original(flags, verdict, *args, **kwargs)

    for name, module in list(sys.modules.items()):
        if name.startswith("vfe3") and getattr(module, "host_check", None) is original:
            monkeypatch.setattr(module, "host_check", counting)
    return counts


def _live_evaluation() -> Tuple[VFEModel, List[Tuple[torch.Tensor, torch.Tensor]]]:
    r"""The frozen reference route (``tests.profiles.reference_config``), whose ``comparison_invariant`` RoPE takes the
    diagonal comparison route and its ``host_check(..., detail=...)`` certificates, on three evaluation batches."""
    from tests.profiles import reference_config
    cfg = reference_config(rope_mode="comparison_invariant", relational_diag_moments_kernel="torch")
    torch.manual_seed(11)
    model = VFEModel(cfg)
    generator = torch.Generator().manual_seed(12)
    stream = torch.randint(0, cfg.vocab_size, (3, cfg.batch_size, cfg.max_seq_len + 1), generator=generator)
    return model, [(window[:, :-1], window[:, 1:]) for window in stream]   # (B, N) inputs and next-token targets


def test_evaluation_forward_defers_its_raise_only_certificates(monkeypatch: pytest.MonkeyPatch) -> None:
    model, batches = _live_evaluation()
    counts = _counted_host_checks(monkeypatch)
    T.evaluate(model, batches, device=CPU)
    assert counts["queued"] > 0
    assert counts["immediate"] == 0, counts


def test_deferred_evaluation_keeps_the_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    model, batches = _live_evaluation()
    deferred = T.evaluate(model, batches, device=CPU)
    monkeypatch.setattr(T, "deferred_host_checks", contextlib.nullcontext)
    immediate = T.evaluate(model, batches, device=CPU)
    assert deferred == immediate


def test_a_deferred_evaluation_verdict_still_reaches_the_precision_ladder() -> None:
    seen: List[Tuple[str, bool]] = []

    def failing(host: List[bool]) -> None:
        if host[0]:
            raise ValueError("projected covariance is not positive definite")

    class Fake:
        def __call__(self, *args: Any, **kwargs: Any) -> str:
            policy = torch.get_float32_matmul_precision()
            seen.append((policy, contracts._DEFERRED_CHECKS is not None))
            contracts.host_check(torch.tensor([policy != "highest"]), failing)
            return "ok"

    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        assert T._eval_forward_with_precision_retry(Fake()) == "ok"
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        torch.set_float32_matmul_precision(previous)
    assert seen == [("high", True), ("highest", True)]
    assert contracts._DEFERRED_CHECKS is None
