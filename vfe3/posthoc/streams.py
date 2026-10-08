r"""The run's evaluation streams, pinned by identity, and their article segmentation.

A post-hoc number is only comparable with the run's own numbers when it is computed on the same token
streams. :func:`resolve_stream` loads the cache that :func:`vfe3.data.datasets.make_dataloader` would
load for ``(dataset, split)`` and checks it against what the run recorded:

* validation: the cache FILE's SHA-256 against the checkpoint's ``selection_data_identity`` (the
  identity the best-checkpoint selection was made on), and the canonical int64 CONTENT SHA-256 and
  token count against ``provenance.json`` (``val_data_sha256``, ``val_data_n_tokens``);
* test and train: the content SHA-256 and token count against ``provenance.json``
  (``test_data_*``, ``train_data_*``). The train stream is loaded with the run's recorded
  ``max_tokens`` cap (the cap applies to the train split only), as the training loader loaded it.

A mismatch raises :class:`StreamIdentityError` unless ``allow_mismatch`` is set, in which case the
record says so. A stream with no recorded identity is used and marked ``unverified``.

The fixed training-split sample (:func:`fixed_train_sample`) follows the evaluation protocol:
windows of ``seq_len`` tokens starting at multiples of ``stride`` whose last ``stride`` targets are
scored (every target has at least ``seq_len - stride`` tokens of context), drawn without replacement
by a generator with the fixed seed :data:`TRAIN_SAMPLE_SEED` (independent of the model and data
seeds), with as many scored targets as the validation split, capped at
:data:`TRAIN_SAMPLE_MAX_TARGETS`. The training loop scores a smaller sample under its own cap
(:data:`vfe3.train.TRAIN_SAMPLE_MAX_TARGETS`); both are prefixes of the same seeded permutation, so the
training sample is a subset of the post-hoc one (:func:`training_loop_sample_windows`), and the
reproduction of the logged ``train_sample_ce`` is checked on that subset only.

WikiText articles begin at level-1 headings, lines of the form ``" = Title = "``;
:func:`article_ids` finds them in the decoded byte stream and assigns every token the index of the
article it belongs to, the resampling unit of the article-level bootstrap.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

TRAIN_SAMPLE_SEED: int = 0                 # generator seed of the fixed training sample (shared by all seeds)
TRAIN_SAMPLE_MAX_TARGETS: int = 262_144    # cap on the training sample's scored targets
FALLBACK_BLOCK_TOKENS: int = 4096          # resampling block when a stream has no article segmentation

# provenance.json key prefix of each split
_PROVENANCE_PREFIX = {"validation": "val", "test": "test", "train": "train"}

# A level-1 WikiText heading line: " = Title = " (level 2+ headings start with " = = ").
_HEADING = re.compile(rb"(?m)^ = [^=\n][^\n]* = $")


class StreamIdentityError(ValueError):
    r"""A token stream differs from the identity the run recorded."""


# ----------------------------------------------------------------------------------------------
# The run
# ----------------------------------------------------------------------------------------------

def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


@dataclass
class RunInfo:
    r"""The saved facts of a run directory needed to reproduce its evaluation protocol."""

    run_dir:            Path
    cfg:                Any = field(repr=False)                 # VFE3Config from the checkpoint bundle
    dataset:            str
    summary:            Dict[str, Any] = field(default_factory=dict, repr=False)
    provenance:         Dict[str, Any] = field(default_factory=dict, repr=False)
    selection_identity: Optional[Dict[str, Any]] = field(default=None, repr=False)
    checkpoint:         str = "final_model.pt"
    dropped_config_fields: Dict[str, Any] = field(default_factory=dict)   # retired toggles ignored

    @property
    def seq_len(self) -> int:
        return int(self.cfg.max_seq_len)

    @property
    def eval_stride(self) -> int:
        stride = self.cfg.eval_stride
        return int(stride) if stride is not None else int(self.cfg.max_seq_len)


def config_from_bundle(
    payload: Mapping[str, Any],

    *,
    source:                     str,
    drop_unknown_config_fields: bool = False,
) -> Tuple[Any, Dict[str, Any]]:
    r"""``(config, dropped)`` from a checkpoint's serialized config (``VFE3Config`` unless tagged).

    Strict by default (:func:`vfe3.config.config_from_serialized` refuses undeclared fields). With
    ``drop_unknown_config_fields`` the fields this build no longer declares are dropped by
    :func:`vfe3.legacy_config.lenient_config_from_serialized` and returned, so the caller records
    them next to its reproduction check: a dropped toggle that changed the forward pass shows up as a
    failure to reproduce the run's recorded loss. A payload tagged ``model_family =
    'conventional_transformer'`` builds :class:`vfe3.baseline.BaselineConfig` under the same rule.
    """
    from vfe3.contracts import is_vfe_config

    if not is_vfe_config(payload):
        from vfe3.baseline import baseline_config_from_serialized

        return baseline_config_from_serialized(
            payload, source=source, drop_unknown_config_fields=drop_unknown_config_fields)
    if drop_unknown_config_fields:
        from vfe3.legacy_config import lenient_config_from_serialized

        cfg, dropped = lenient_config_from_serialized(payload, source=source)
        return cfg, dict(dropped)
    from vfe3.config import config_from_serialized

    return config_from_serialized(payload, source=source), {}


def load_run_info(
    run_dir:    "str | Path",
    checkpoint: str = "final_model.pt",

    *,
    drop_unknown_config_fields: bool = False,
) -> RunInfo:
    r"""Read the config (from the checkpoint bundle, memory-mapped), dataset, summary and provenance."""
    run_dir = Path(run_dir)
    path = run_dir / checkpoint
    if not path.is_file():
        raise FileNotFoundError(f"no checkpoint at {path}")
    bundle = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(bundle, Mapping) or not isinstance(bundle.get("config"), Mapping):
        raise ValueError(f"{path} is not a model bundle with a 'config'")
    cfg, dropped = config_from_bundle(bundle["config"], source=str(path),
                                      drop_unknown_config_fields=drop_unknown_config_fields)
    provenance = _read_json(run_dir / "provenance.json")
    selection = bundle.get("selection_data_identity") or provenance.get("selection_data_identity")
    selection = dict(selection) if isinstance(selection, Mapping) else None
    dataset = (selection or {}).get("dataset") or _read_json(run_dir / "config.json").get("dataset")
    if not isinstance(dataset, str) or not dataset:
        raise ValueError(f"the dataset of {run_dir} is recorded neither in the checkpoint nor config.json")
    return RunInfo(run_dir=run_dir, cfg=cfg, dataset=dataset,
                   summary=_read_json(run_dir / "summary.json"), provenance=provenance,
                   selection_identity=selection, checkpoint=checkpoint, dropped_config_fields=dropped)


# ----------------------------------------------------------------------------------------------
# Stream identity
# ----------------------------------------------------------------------------------------------

def token_content_sha256(tokens: torch.Tensor, *, chunk_tokens: int = 1 << 20) -> str:
    r"""SHA-256 of the token values as little-endian int64, in bounded chunks (the provenance digest)."""
    flat = tokens.reshape(-1)
    digest = hashlib.sha256()
    for start in range(0, flat.numel(), chunk_tokens):
        chunk = flat[start:start + chunk_tokens].to(torch.long).contiguous()
        digest.update(chunk.numpy().tobytes())
    return digest.hexdigest()


def resolve_stream(
    info:  RunInfo,
    split: str,

    *,
    cache_dir:      Optional[Path] = None,
    allow_mismatch: bool           = False,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    r"""Load ``split`` as the run's loaders would and check it against the recorded identity.

    Returns ``(tokens, record)``; ``record`` holds the resolved cache path and format, the file and
    content SHA-256, the token count, every expected value found, the per-field checks and a
    ``status`` of ``"verified"``, ``"unverified"`` (nothing recorded) or ``"mismatch_allowed"``.
    """
    from vfe3.data.datasets import cache_path, cache_source_identity, load_cached_tokens

    if split not in _PROVENANCE_PREFIX:
        raise ValueError(f"split must be one of {sorted(_PROVENANCE_PREFIX)}, got {split!r}")
    source = cache_source_identity(info.dataset, split, cache_dir=cache_dir)
    max_tokens = info.provenance.get("max_tokens") if split == "train" else None
    limit = int(max_tokens) if max_tokens is not None else None   # the run capped its train stream only
    tokens = load_cached_tokens(info.dataset, split, cache_dir=cache_dir, limit=limit)
    suffix = "pt" if source["format"] == "pt" else "bin"
    record: Dict[str, Any] = {
        "split":          split,
        "dataset":        info.dataset,
        "path":           str(cache_path(info.dataset, split, suffix=suffix, cache_dir=cache_dir)),
        "format":         source["format"],
        "file_sha256":    source["sha256"],
        "content_sha256": token_content_sha256(tokens),
        "n_tokens":       int(tokens.numel()),
        "max_tokens":     limit,
        "tokenizer_provenance_status": source.get("tokenizer_provenance_status"),
    }
    expected: Dict[str, Any] = {}
    selection = info.selection_identity or {}
    if selection.get("split") == split:
        recorded = (selection.get("source") or {}).get("sha256")
        if recorded:
            expected["file_sha256"] = recorded
    prefix = _PROVENANCE_PREFIX[split]
    if info.provenance.get(f"{prefix}_data_sha256"):
        expected["content_sha256"] = info.provenance[f"{prefix}_data_sha256"]
    if info.provenance.get(f"{prefix}_data_n_tokens") is not None:
        expected["n_tokens"] = int(info.provenance[f"{prefix}_data_n_tokens"])
    checks = [{"field": key, "expected": value, "actual": record[key], "ok": record[key] == value}
              for key, value in expected.items()]
    record["expected"] = expected
    record["checks"] = checks
    failed = [c for c in checks if not c["ok"]]
    if failed:
        message = (f"{info.dataset}/{split} stream at {record['path']} differs from the run's recorded "
                   "identity: " + "; ".join(f"{c['field']} {c['actual']} != recorded {c['expected']}"
                                             for c in failed))
        if not allow_mismatch:
            raise StreamIdentityError(message)
        record["status"] = "mismatch_allowed"
        record["mismatch"] = message
    else:
        record["status"] = "verified" if checks else "unverified"
    return tokens, record


# ----------------------------------------------------------------------------------------------
# Windows
# ----------------------------------------------------------------------------------------------

def eval_windows(tokens: torch.Tensor, seq_len: int, stride: int) -> Dataset:
    r"""The evaluation-protocol windows (``TokenWindows(..., pad_final=True)``), as the run's loaders."""
    from vfe3.data.datasets import TokenWindows

    return TokenWindows(tokens, seq_len, stride=stride, pad_final=True)


def evenly_spaced(n_total: int, n_select: Optional[int]) -> np.ndarray:
    r"""``n_select`` distinct indices spread evenly over ``range(n_total)``; all indices when None."""
    if n_select is None or n_select >= n_total:
        return np.arange(n_total, dtype=np.int64)
    if n_select <= 0:
        raise ValueError("n_select must be positive")
    return np.unique(np.round(np.linspace(0, n_total - 1, n_select)).astype(np.int64))


class ScoredTailWindows(Dataset):
    r"""Windows of a stream at given starts, scoring only the last ``stride`` targets of each.

    Item ``k`` is ``(inputs, targets)`` with ``inputs = tokens[s_k : s_k + L]`` and
    ``targets = tokens[s_k + 1 : s_k + L + 1]``, the first ``L - stride`` targets set to ``-100``.
    """

    def __init__(self, tokens: torch.Tensor, starts: Sequence[int], seq_len: int, stride: int) -> None:
        if stride <= 0 or stride > seq_len:
            raise ValueError("stride must lie in [1, seq_len]")
        self.tokens = tokens
        self.starts = np.asarray(starts, dtype=np.int64)
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        if self.starts.size and int(self.starts.max()) + self.seq_len + 1 > int(tokens.numel()):
            raise ValueError("a window runs past the end of the stream")

    def __len__(self) -> int:
        return int(self.starts.size)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = int(self.starts[idx])
        window = self.tokens[start:start + self.seq_len + 1].to(torch.long)
        inputs, targets = window[:-1], window[1:].clone()
        targets[:self.seq_len - self.stride] = -100
        return inputs, targets

    def window_start(self, idx: int) -> int:
        return int(self.starts[idx])


def fixed_train_sample(
    train_tokens: torch.Tensor,
    seq_len:      int,
    stride:       int,
    n_targets:    int,

    *,
    max_targets: int = TRAIN_SAMPLE_MAX_TARGETS,
    seed:        int = TRAIN_SAMPLE_SEED,
) -> ScoredTailWindows:
    r"""The fixed training-split sample under the evaluation protocol.

    Candidate windows are the training loader's windows, starting at :math:`s = k L` for
    :math:`k = 0, \dots, \lfloor (T - L - 1)/L \rfloor`; the sample is the first
    :math:`\lceil \min(n, n_{\max}) / \mathrm{stride} \rceil` entries of a ``randperm`` drawn by a
    generator seeded with ``seed``, sorted, and each window scores its last ``stride`` targets.
    With ``max_targets = vfe3.train.TRAIN_SAMPLE_MAX_TARGETS`` this is the sample ``vfe3.train``
    scores during training (``_build_fixed_train_sample``); the default post-hoc cap is larger, and
    :func:`training_loop_sample_windows` locates the training sample inside it.
    """
    n_candidates = (int(train_tokens.numel()) - seq_len - 1) // seq_len + 1
    if n_candidates <= 0:
        raise ValueError("training stream is shorter than one window")
    n_windows = min(n_candidates, max(1, math.ceil(min(int(n_targets), int(max_targets)) / stride)))
    generator = torch.Generator().manual_seed(int(seed))
    chosen = torch.randperm(n_candidates, generator=generator)[:n_windows].sort().values
    return ScoredTailWindows(train_tokens, (chosen * seq_len).numpy(), seq_len, stride)


def training_loop_sample_windows(
    sample:       ScoredTailWindows,
    train_tokens: torch.Tensor,
    n_targets:    int,

    *,
    seed: int = TRAIN_SAMPLE_SEED,
) -> Tuple[np.ndarray, int]:
    r"""``(indices into sample, max_targets)`` of the windows the training loop scored as its fixed sample.

    The training loop draws the first :math:`\lceil \min(n, n^{\rm train}_{\max}) / \mathrm{stride} \rceil`
    entries of the same seeded permutation (:data:`vfe3.train.TRAIN_SAMPLE_MAX_TARGETS`), a prefix of
    the post-hoc draw whenever the post-hoc cap is at least as large. Raises ``ValueError`` when the
    training windows are not all in ``sample`` (a post-hoc cap below the training cap).
    """
    from vfe3.train import TRAIN_SAMPLE_MAX_TARGETS as training_max_targets

    training = fixed_train_sample(train_tokens, sample.seq_len, sample.stride, n_targets,
                                  max_targets=training_max_targets, seed=seed)
    inside = np.isin(sample.starts, training.starts)
    if int(inside.sum()) != len(training):
        raise ValueError("the training loop's fixed sample is not contained in the post-hoc sample")
    return np.flatnonzero(inside).astype(np.int64), int(training_max_targets)


# ----------------------------------------------------------------------------------------------
# Articles
# ----------------------------------------------------------------------------------------------

def _byte_decoder(dataset: str) -> Optional[Callable[[Sequence[int]], bytes]]:
    from vfe3.data.datasets import get_tiktoken_byte_decoder

    return get_tiktoken_byte_decoder(dataset)


def article_ids(
    tokens:  torch.Tensor,
    dataset: Optional[str] = None,

    *,
    decode_bytes: Optional[Callable[[Sequence[int]], bytes]] = None,
    vocab_size:   Optional[int] = None,
) -> Optional[Tuple[np.ndarray, List[str], Dict[str, Any]]]:
    r"""Per-token article index from the level-1 headings of the decoded stream, or ``None``.

    The stream is decoded to bytes; each match of ``^ = Title = $`` (a line whose first character
    after ``" = "`` is not ``"="``) starts an article at the token containing the heading's first
    byte. Tokens before the first heading join article 0. Returns ``(ids (T,) int64, titles, info)``,
    or ``None`` when no byte decoder exists for the dataset or no heading is found. ``vocab_size``,
    when given, bounds the ids (``ValueError`` otherwise); only the ids present are decoded.
    """
    if decode_bytes is None:
        if dataset is None:
            return None
        decode_bytes = _byte_decoder(dataset)
        if decode_bytes is None:
            return None
    ids = tokens.reshape(-1).to(torch.long).numpy()
    if ids.size == 0:
        return None
    if vocab_size is not None and (int(ids.min()) < 0 or int(ids.max()) >= int(vocab_size)):
        raise ValueError(f"token ids span [{int(ids.min())}, {int(ids.max())}], outside the vocabulary "
                         f"[0, {int(vocab_size)})")
    # Byte lengths of the ids that occur only: a tokenizer may leave ids below its vocabulary size
    # unassigned (cl100k_base), and decoding those raises.
    present, inverse = np.unique(ids, return_inverse=True)                            # (U,), (T,)
    lengths = np.fromiter((len(decode_bytes([int(v)])) for v in present), dtype=np.int64,
                          count=present.size)                                         # (U,)
    ends = np.cumsum(lengths[inverse.reshape(-1)])                                    # (T,) exclusive end byte
    text = decode_bytes(ids.tolist())
    if len(text) != int(ends[-1]):
        raise ValueError("per-token byte lengths do not add up to the decoded stream")
    matches = list(_HEADING.finditer(text))
    if not matches:
        return None
    starts = np.searchsorted(ends, np.array([m.start() for m in matches], dtype=np.int64), side="right")
    first = np.zeros(ids.size, dtype=np.int64)
    first[starts] = 1
    article = np.cumsum(first) - 1                                                    # (T,)
    n_preamble = int((article < 0).sum())
    article = np.maximum(article, 0)
    titles = [m.group(0).decode("utf-8", errors="replace").strip(" =") for m in matches]
    info = {"n_articles": len(matches), "n_preamble_tokens": n_preamble,
            "rule": "level-1 heading lines ' = Title = ' in the decoded byte stream"}
    return article, titles, info


def resampling_units(
    tokens:  torch.Tensor,
    dataset: Optional[str],

    *,
    decode_bytes: Optional[Callable[[Sequence[int]], bytes]] = None,
    vocab_size:   Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    r"""Per-token resampling unit: the article when headings are found, else a fixed token block."""
    found = article_ids(tokens, dataset, decode_bytes=decode_bytes, vocab_size=vocab_size)
    if found is not None:
        ids, titles, info = found
        return ids, {"unit": "article", "n_units": int(info["n_articles"]), "titles": titles, **info}
    n = int(tokens.numel())
    ids = np.arange(n, dtype=np.int64) // FALLBACK_BLOCK_TOKENS
    return ids, {"unit": f"block_{FALLBACK_BLOCK_TOKENS}", "n_units": int(ids[-1]) + 1 if n else 0,
                 "rule": "no article headings or byte decoder: contiguous token blocks"}
