r"""Word resources for :mod:`vfe3.semantics`: curated lexicon/relations and external benchmarks.

The curated resources ship as package data next to this module (``resources/*.json``):

* ``semantic_fields.json`` -- disjoint English semantic fields (field-purity and clustering tests);
* ``relations.json`` -- relation pairs (gender, capital, nationality, number, degree, tense,
  adverb, antonym), famous analogies, hypernym sets, semantic-projection axes (Grand et al. 2022),
  neighbor-table probe words and ambiguous words.

The external benchmarks are read from the copies bundled with gensim's test data when gensim is
installed: the Google analogy set ``questions-words.txt`` (Mikolov et al. 2013), SimLex-999
(Hill, Reichart & Korhonen 2015) and WordSim-353 (Finkelstein et al. 2002). They are located with
:func:`importlib.util.find_spec`, so gensim itself is never imported.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

RESOURCE_DIR = Path(__file__).parent / "resources"

SIMILARITY_FILES: Dict[str, str] = {"simlex999": "simlex999.txt", "wordsim353": "wordsim353.tsv"}
GOOGLE_ANALOGY_FILE = "questions-words.txt"

_RELATION_KEYS = ("relations", "curated_analogies", "hypernyms", "axes", "probe_words", "ambiguous_words")
_AXIS_KEYS = ("pairs", "words", "neg_label", "pos_label")


def _read_resource(name: str) -> Dict[str, Any]:
    path = RESOURCE_DIR / name
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"resource {path} must be a JSON object")
    return payload


def _require_words(words: Any, where: str) -> List[str]:
    if not isinstance(words, list) or not all(isinstance(w, str) and w for w in words):
        raise ValueError(f"{where} must be a list of non-empty strings")
    return list(words)


def _require_pairs(pairs: Any, where: str) -> List[List[str]]:
    if not isinstance(pairs, list) or not all(
            isinstance(p, list) and len(p) == 2 and all(isinstance(w, str) and w for w in p)
            for p in pairs):
        raise ValueError(f"{where} must be a list of [word, word] pairs")
    return [list(p) for p in pairs]


def load_fields() -> Dict[str, List[str]]:
    r"""The curated semantic fields ``{field: [words]}``; raises if a word appears in two fields."""
    fields = _read_resource("semantic_fields.json").get("fields")
    if not isinstance(fields, dict) or not fields:
        raise ValueError("semantic_fields.json must hold a non-empty 'fields' object")
    owner: Dict[str, str] = {}
    duplicates: List[str] = []
    out: Dict[str, List[str]] = {}
    for name, words in fields.items():
        out[name] = _require_words(words, f"field {name!r}")
        for word in out[name]:
            if word in owner:
                duplicates.append(f"{word!r} in {owner[word]!r} and {name!r}")
            owner[word] = name
    if duplicates:
        raise ValueError("semantic fields must be disjoint: " + "; ".join(duplicates))
    return out


def load_relations() -> Dict[str, Any]:
    r"""The curated relation resources (validated), with the keys of ``relations.json``:

    ``relations`` ``{name: [[a, a_star], ...]}``; ``curated_analogies``
    ``[[a, a_star, b, b_star, category], ...]``; ``hypernyms`` ``{hypernym: [hyponyms]}``; ``axes``
    ``{name: {"pairs": [[neg, pos], ...], "words": [...], "neg_label": str, "pos_label": str}}``;
    ``probe_words`` and ``ambiguous_words`` (lists of words).
    """
    payload = _read_resource("relations.json")
    missing = [key for key in _RELATION_KEYS if key not in payload]
    if missing:
        raise ValueError(f"relations.json lacks {missing}")
    relations = {name: _require_pairs(pairs, f"relation {name!r}")
                 for name, pairs in payload["relations"].items()}
    curated = payload["curated_analogies"]
    if not isinstance(curated, list) or not all(
            isinstance(q, list) and len(q) == 5 and all(isinstance(w, str) and w for w in q)
            for q in curated):
        raise ValueError("curated_analogies must be [a, a_star, b, b_star, category] rows")
    hypernyms = {name: _require_words(words, f"hypernym {name!r}")
                 for name, words in payload["hypernyms"].items()}
    axes: Dict[str, Dict[str, Any]] = {}
    for name, axis in payload["axes"].items():
        if not isinstance(axis, dict) or any(key not in axis for key in _AXIS_KEYS):
            raise ValueError(f"axis {name!r} must carry {list(_AXIS_KEYS)}")
        axes[name] = {
            "pairs":     _require_pairs(axis["pairs"], f"axis {name!r} pairs"),
            "words":     _require_words(axis["words"], f"axis {name!r} words"),
            "neg_label": str(axis["neg_label"]),
            "pos_label": str(axis["pos_label"]),
        }
    return {
        "relations":         relations,
        "curated_analogies": [list(q) for q in curated],
        "hypernyms":         hypernyms,
        "axes":              axes,
        "probe_words":       _require_words(payload["probe_words"], "probe_words"),
        "ambiguous_words":   _require_words(payload["ambiguous_words"], "ambiguous_words"),
    }


def gensim_test_data_dir() -> Optional[Path]:
    r"""``<gensim package>/test/test_data`` when gensim is installed (located, not imported)."""
    spec = importlib.util.find_spec("gensim")
    if spec is None or not spec.origin:
        return None
    directory = Path(spec.origin).parent / "test" / "test_data"
    return directory if directory.is_dir() else None


def _benchmark_path(file_name: str, path: Optional[Path]) -> Optional[Path]:
    if path is not None:
        return Path(path)
    directory = gensim_test_data_dir()
    if directory is None or not (directory / file_name).is_file():
        return None
    return directory / file_name


def load_google_analogies(path: Optional[Path] = None) -> Optional[List[Tuple[str, str, str, str, str]]]:
    r"""Google analogy questions as ``(category, a, a_star, b, b_star)``, or ``None`` when absent.

    Format of ``questions-words.txt``: a line ``: category`` opens a category; each following line
    ``a a_star b b_star`` is one question ``a : a_star :: b : b_star`` whose answer is ``b_star``.
    ``path`` overrides the gensim test-data location.
    """
    source = _benchmark_path(GOOGLE_ANALOGY_FILE, path)
    if source is None:
        return None
    questions: List[Tuple[str, str, str, str, str]] = []
    category: Optional[str] = None
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        if line.startswith(":"):
            category = line[1:].strip()
            continue
        words = line.split()
        if len(words) != 4 or category is None:
            raise ValueError(f"{source}:{number}: expected 'a a_star b b_star' under a ': category' line")
        questions.append((category, words[0], words[1], words[2], words[3]))
    return questions


def load_similarity(name: str, path: Optional[Path] = None) -> Optional[List[Tuple[str, str, float]]]:
    r"""Word-similarity benchmark ``name`` (``"simlex999"`` | ``"wordsim353"``) as
    ``(w1, w2, human score)`` rows, or ``None`` when the file is absent.

    Reads the three-column gensim copies (``w1 <tab> w2 <tab> score``), skipping ``#`` comment
    lines and a leading non-numeric header row. ``path`` overrides the gensim test-data location.
    """
    if name not in SIMILARITY_FILES:
        raise ValueError(f"unknown similarity benchmark {name!r}; known: {sorted(SIMILARITY_FILES)}")
    source = _benchmark_path(SIMILARITY_FILES[name], path)
    if source is None:
        return None
    rows: List[Tuple[str, str, float]] = []
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t") if "\t" in line else line.split()
        if len(cols) < 3:
            raise ValueError(f"{source}:{number}: expected 'w1 w2 score'")
        try:
            score = float(cols[2])
        except ValueError:
            if not rows:                                   # a header row before the data
                continue
            raise ValueError(f"{source}:{number}: non-numeric score {cols[2]!r}") from None
        rows.append((cols[0].strip(), cols[1].strip(), score))
    return rows


def resolve_words(vocab: Any, words: Iterable[str], allowed: Set[int]) -> Tuple[Dict[str, int], List[str]]:
    r"""Resolve ``words`` through ``vocab.word_id`` and keep those whose id is in ``allowed``.

    Returns ``(resolved, missing)``: ``resolved`` maps each resolvable word to its token id (input
    order, duplicates collapsed); ``missing`` lists the others (not a single token, or filtered out
    of ``allowed``, e.g. below the frequency floor).
    """
    resolved: Dict[str, int] = {}
    missing: List[str] = []
    for word in dict.fromkeys(words):
        token_id = vocab.word_id(word)
        if token_id is not None and int(token_id) in allowed:
            resolved[word] = int(token_id)
        else:
            missing.append(word)
    return resolved, missing
