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

F125: evaluation and diagnostic forwards ran outside ``deferred_host_checks`` and paid one host synchronization per
raise-only certificate. The no-update replay ladder now runs its call inside the deferral scope; the outputs are
unchanged and a deferred verdict still reaches the ladder.

Tiny CPU models only (K=4, two heads, N=4, vocab 7; the deferral test runs ``tests.profiles.live_config``).
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


