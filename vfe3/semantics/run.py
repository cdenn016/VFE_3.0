r"""Load a trained vfe3 run for semantic analysis and prepare the shared analysis context.

:func:`load_semantic_run` reads ``<run_dir>/best_model.pt`` memory-mapped on the CPU and exposes
the token tables through float64 accessors without building the model: the prior means ``mu_v``,
the prior log-variances (family-aware), the gauge-frame algebra coordinates ``phi_v`` and their
per-block matrices ``X_v = sum_a phi_{v,a} G_a``, the positional right factors ``Y_p``, and the
decode (unembedding) tables. It works without ``provenance.json`` / ``summary.json``: the config and
the data identity come from the checkpoint bundle, the dataset name from the bundle's
``selection_data_identity`` or, failing that, from ``config.json``.

:func:`prepare_context` builds the :class:`~vfe3.semantics.common.AnalysisContext` every section
shares: the frequency-filtered candidate vocabulary, the curated lexicon and relations, the
external benchmarks resolved to token ids, and their coverage.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from vfe3.config import VFE3Config, config_from_serialized
from vfe3.semantics.common import AnalogyQuestion, AnalysisContext
from vfe3.semantics.resources import (
    SIMILARITY_FILES,
    load_fields,
    load_google_analogies,
    load_relations,
    load_similarity,
    resolve_words,
)
from vfe3.semantics.vocab import Vocab, vocab_for_run

if TYPE_CHECKING:
    from vfe3.geometry.groups import GaugeGroup
    from vfe3.model.model import VFEModel


# model_state keys of the tables the accessors read
MU_KEY = "prior_bank.mu_embed"
SIGMA_LOG_KEY = "prior_bank.sigma_log_embed"
SIGMA_CHOL_KEY = "prior_bank.sigma_chol_embed"          # prior_covariance_table='block_cholesky' (V, H, d(d+1)/2)
PHI_KEY = "prior_bank.phi_embed"
DECODE_MU_KEY = "prior_bank.decode_mu_embed"
DECODE_SIGMA_LOG_KEY = "prior_bank.decode_sigma_log_embed"
DECODE_LOG_SCALE_KEY = "prior_bank.decode_log_scale"
OUTPUT_W_KEY = "prior_bank.output_proj_weight"
OUTPUT_B_KEY = "prior_bank.output_proj_bias"
UNIGRAM_KEY = "prior_bank.unigram_log_prior"
POS_PHI_KEY = "pos_phi_free"

# Decoders whose logits are -KL(q || N(m_v, diag t_v)) / tau_eff (+ kappa log pi_v).
_KL_DECODE_MODES = frozenset(("diagonal", "diagonal_chunked", "full", "full_chunked"))
_FAMILY_DECODE_MODES = frozenset(("family", "family_chunked"))
_GAUSSIAN_KL_FAMILIES = frozenset(("gaussian_diagonal", "gaussian_full"))

CORE_DEFAULT_OPTIONS: Dict[str, Any] = {
    "min_count":      100,
    "max_candidates": 20000,
    "seed":           0,
    "device":         "cpu",
    "threads":        None,
    # BPE only: keep out of the frequency head the whole-word-looking tokens that are mostly word
    # prefixes (" Squ", " env", " un"), i.e. followed by a continuation piece in more than this
    # fraction of their training occurrences. Curated/benchmark words are exempt (" cat" is followed
    # by a continuation 56% of the time but is a word). None disables the filter.
    "max_continuation_fraction": 0.9,
}

_CONTINUATION_CHUNK = 1 << 24          # stream tokens per chunk of the continuation-fraction scan

_BLOCK_EMBED_BUDGET = 2 ** 22          # float64 entries per chunk of dense (n, K, K) work (32 MiB)


# ----------------------------------------------------------------------------------------------
# Frequency counts
# ----------------------------------------------------------------------------------------------

def counts_from_unigram_log_prior(
    log_prior: np.ndarray,                               # (V,) u_v = log pi_v

    *,
    integrality_tol: float = 0.05,
    max_log_error:   float = 1e-4,
) -> Tuple[np.ndarray, Dict[str, float]]:
    r"""Invert the add-one-smoothed unigram table back to training counts.

    The table is $u_v = \log\pi_v = \log\frac{c_v + 1}{N + V}$ with $N = \sum_v c_v$. When some
    token has count zero, $\min_v u_v = -\log(N + V)$, hence

    .. math:: c_v = \exp(u_v - \min_w u_w) - 1 .

    The zero-count premise is checked, not assumed: if the rarest token had count $m > 0$ the
    recovered values $(c_v + 1)/(m + 1) - 1$ would be non-integral, so the fractional residual of the
    recovered counts up to 1000 must stay below ``integrality_tol``. The rounded counts must then
    reproduce the table, $\max_v |\log\frac{c_v+1}{\sum c + V} - u_v| \le$ ``max_log_error``
    (float32 storage gives about $10^{-6}$). Raises ``ValueError`` when the table is unset
    (all-zero) or either check fails.
    """
    u = np.asarray(log_prior, dtype=np.float64).reshape(-1)             # (V,)
    if u.size == 0 or not np.isfinite(u).all():
        raise ValueError("unigram_log_prior must be a finite, non-empty (V,) table")
    if not np.any(u != 0.0):
        raise ValueError("unigram_log_prior is all-zero (never set): it carries no counts")
    u_min = float(u.min())
    recovered = np.exp(u - u_min) - 1.0                                 # (V,) real-valued c_v
    counts = np.rint(recovered).astype(np.int64)
    small = recovered <= 1000.0
    fractional = float(np.abs(recovered - counts)[small].max())
    if fractional > integrality_tol:
        raise ValueError(
            f"inverted unigram counts are not integral (max fractional residual {fractional:.3g} "
            f"over counts <= 1000): the rarest token does not have count 0, so N + V != exp(-min u)")
    V = u.size
    total = int(counts.sum())
    roundtrip = np.log((counts + 1.0) / (total + V))
    log_error = float(np.abs(roundtrip - u).max())
    if log_error > max_log_error:
        raise ValueError(
            f"inverted unigram counts do not reproduce the table (max |delta log pi| = {log_error:.3g})")
    diagnostics = {
        "N":                           total,
        "N_plus_V_from_min":           float(np.exp(-u_min)),
        "n_zero":                      int((counts == 0).sum()),
        "max_fractional_residual":     fractional,
        "max_abs_log_roundtrip_error": log_error,
    }
    return counts, diagnostics


def counts_from_token_stream(tokens: torch.Tensor, vocab_size: int) -> np.ndarray:
    r"""(V,) int64 occurrence counts of a 1-D token-id stream (``bincount`` with ``minlength=V``)."""
    ids = tokens.numpy() if isinstance(tokens, torch.Tensor) else np.asarray(tokens)
    if ids.ndim != 1:
        raise ValueError(f"token stream must be 1-D, got shape {ids.shape}")
    if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= vocab_size):
        raise ValueError(f"token stream has ids outside [0, {vocab_size})")
    return np.bincount(ids, minlength=vocab_size).astype(np.int64)


# ----------------------------------------------------------------------------------------------
# The run
# ----------------------------------------------------------------------------------------------

@dataclass
class SemanticRun:
    r"""A trained run opened for analysis: config, memory-mapped state, vocabulary, counts, facts.

    Table accessors take token ids (any int sequence) and return float64 numpy rows in the given
    order. ``notes`` collects loader decisions the report must state (count provenance, the
    fp32 replay of a bf16-trained model, block-structure fallbacks).
    """

    run_dir:            Path
    checkpoint_path:    Path
    cfg:                VFE3Config = field(repr=False)
    state:              Mapping[str, torch.Tensor] = field(repr=False)   # memory-mapped model_state
    selection_identity: Optional[dict]
    dataset:            Optional[str]
    vocab:              Vocab = field(repr=False)
    counts:             Optional[np.ndarray] = field(repr=False)          # (V,) int64 training counts
    counts_source:      str                             # "unigram_log_prior" | "train_cache_bincount" | "unavailable"
    group:              "GaugeGroup" = field(repr=False)
    K:                  int
    n_layers:           int
    n_heads:            int
    n_gen:              int
    block_slices:       List[slice]                     # coordinate range each frame block acts on
    frames_active:      bool
    decode_kind:        str                             # "linear" | "gaussian" | "unsupported"
    sigma_frozen:       Optional[bool]
    phi_frozen:         bool
    counts_diagnostics: Dict[str, Any] = field(default_factory=dict)
    notes:              List[str] = field(default_factory=list)
    cache_dir:          Optional[Path] = None           # token-cache directory (None -> default)

    # ---- helpers ------------------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return int(self.state[MU_KEY].shape[0])

    def _table(self, key: str) -> torch.Tensor:
        if key not in self.state:
            raise KeyError(f"checkpoint {self.checkpoint_path} has no {key!r} table")
        return self.state[key]

    def _index(self, ids: Sequence[int]) -> torch.Tensor:
        index = np.asarray(ids, dtype=np.int64).reshape(-1)
        if index.size and (int(index.min()) < 0 or int(index.max()) >= self.vocab_size):
            raise IndexError(f"token ids must lie in [0, {self.vocab_size})")
        return torch.from_numpy(index)

    def _rows(self, key: str, ids: Sequence[int]) -> torch.Tensor:
        return self._table(key)[self._index(ids)].to(torch.float64)      # (n, ...) float64

    def _require_decode(self, kind: str) -> None:
        if self.decode_kind != kind:
            raise ValueError(
                f"this accessor needs decode_kind={kind!r}; the run decodes with "
                f"decode_mode={self.cfg.decode_mode!r} (decode_kind={self.decode_kind!r})")

    def _coords_to_blocks(self, coords: torch.Tensor) -> List[np.ndarray]:
        r"""Per-block matrices of algebra coordinates: ``[(n, d_b, d_b)]`` with
        ``X = sum_a coords_a G_a`` restricted to each ``block_slices`` block."""
        n = int(coords.shape[0])
        coords = coords.to(torch.float64)
        layout = self.group.elementary_block_shape
        if layout is not None and len(self.block_slices) == layout[0]:
            H, d = layout                                                   # block_head_row_major
            X = coords.reshape(n, H, d, d)                                  # (n, H, d, d)
            return [X[:, h].contiguous().numpy() for h in range(H)]
        blocks = [np.empty((n, s.stop - s.start, s.stop - s.start)) for s in self.block_slices]
        chunk = max(1, _BLOCK_EMBED_BUDGET // (self.K * self.K))
        for i0 in range(0, n, chunk):
            i1 = min(n, i0 + chunk)
            X = self.group.embed_algebra(coords[i0:i1], dtype=torch.float64)   # (c, K, K)
            for b, s in enumerate(self.block_slices):
                blocks[b][i0:i1] = X[:, s, s].numpy()
        return blocks

    # ---- prior tables -------------------------------------------------------------------------

    def mu(self, ids: Sequence[int]) -> np.ndarray:
        r"""(n, K) prior means ``mu_v`` (the token table ``mu_embed``)."""
        return self._rows(MU_KEY, ids).numpy()

    def log_var(self, ids: Sequence[int]) -> np.ndarray:
        r"""(n, K) log of the prior covariance diagonal, family-aware.

        The slot stores $s_{vk}$; the family dispersion is
        $\max(\exp(\min(s_{vk}, 80)), \epsilon)$ (``bounded_variance_from_log``), which is the
        variance for Gaussian families and the scale $b$ for Laplace (variance $2b^2$). The
        family hook ``covariance_diagonal`` maps it to $\operatorname{diag}\Sigma_v$; a full family
        reads the diagonal of the (exactly diagonal) promoted prior ``diag_embed``.

        Under ``prior_covariance_table='block_cholesky'`` (no ``sigma_log_embed``) it is
        $\log\operatorname{diag}C_v$ of the per-head blocks that
        :func:`vfe3.model.prior_bank.packed_cholesky_covariance` builds from ``sigma_chol_embed``
        (audit 2026-10-09 round 2, item 33).
        """
        from vfe3.families.base import get_family
        from vfe3.numerics import bounded_variance_from_log

        if SIGMA_LOG_KEY not in self.state:
            return self._block_cholesky_log_var(ids)
        family = get_family(self.cfg.family)
        s = self._rows(SIGMA_LOG_KEY, ids)                                  # (n, K)
        dispersion = bounded_variance_from_log(s, eps=self.cfg.eps)         # (n, K)
        if family.cov_kind == "full":
            chunk = max(1, _BLOCK_EMBED_BUDGET // (self.K * self.K))       # rows per (c, K, K) chunk
            parts = [family.covariance_diagonal(torch.diag_embed(dispersion[i:i + chunk]), eps=self.cfg.eps)
                     for i in range(0, dispersion.shape[0], chunk)]
            variance = torch.cat(parts, dim=0) if parts else dispersion
        else:
            variance = family.covariance_diagonal(dispersion, eps=self.cfg.eps)   # (n, K)
        return torch.log(variance).numpy()

    def _block_cholesky_log_var(self, ids: Sequence[int]) -> np.ndarray:
        r"""(n, K) $\log\operatorname{diag}C_v$ of the ``block_cholesky`` prior blocks, in row chunks."""
        parts = [C.diagonal(dim1=-2, dim2=-1).reshape(-1, self.K)          # (c, K) diag C_v
                 for C in self._block_cholesky_chunks(ids)]
        variance = torch.cat(parts, dim=0) if parts else torch.zeros(0, self.K, dtype=torch.float64)
        return torch.log(variance).numpy()

    def _block_cholesky_chunks(self, ids: Sequence[int]) -> Iterator[torch.Tensor]:
        r"""The ``block_cholesky`` prior blocks of ``ids`` as float64 ``(c, H, d, d)`` row chunks."""
        from vfe3.model.prior_bank import packed_cholesky_covariance

        if SIGMA_CHOL_KEY not in self.state:
            raise KeyError(f"checkpoint {self.checkpoint_path} has neither {SIGMA_LOG_KEY!r} nor {SIGMA_CHOL_KEY!r}: "
                           "the prior log-variance is not applicable to this run")
        packed = self._rows(SIGMA_CHOL_KEY, ids)                            # (n, H, d(d+1)/2)
        n, H = int(packed.shape[0]), int(packed.shape[1])
        d = self.K // H
        chunk = max(1, _BLOCK_EMBED_BUDGET // (self.K * d))                  # rows per (c, H, d, d) chunk
        for i in range(0, n, chunk):
            yield packed_cholesky_covariance(packed[i:i + chunk], d, eps=self.cfg.eps)   # (c, H, d, d)

    @property
    def prior_covariance_blocks(self) -> bool:
        r"""Whether the prior covariance is the dense per-head ``block_cholesky`` table (not a diagonal)."""
        return SIGMA_LOG_KEY not in self.state and SIGMA_CHOL_KEY in self.state

    def log_det(self, ids: Sequence[int]) -> np.ndarray:
        r"""(n,) $\log\det\Sigma_v$ of the prior covariance.

        For a diagonal table (and a full family's exactly diagonal promoted prior) this is the row sum
        of :meth:`log_var`. Under ``block_cholesky`` it is $\sum_h\log\det C_v^{(h)}$ over the per-head
        blocks, from their Cholesky factors; the diagonal sum $\sum_k\log C_{v,kk}$ is only an upper
        bound on it (Hadamard's inequality; audit 2026-10-10 #20).
        """
        if not self.prior_covariance_blocks:
            return self.log_var(ids).sum(axis=1)
        parts = [2.0 * torch.linalg.cholesky(C).diagonal(dim1=-2, dim2=-1).log().sum(dim=(-2, -1))   # (c,)
                 for C in self._block_cholesky_chunks(ids)]
        return (torch.cat(parts) if parts else torch.zeros(0, dtype=torch.float64)).numpy()

    def phi(self, ids: Sequence[int]) -> np.ndarray:
        r"""(n, n_gen) raw gauge-frame algebra coordinates ``phi_v`` (trained trivialization)."""
        return self._rows(PHI_KEY, ids).numpy()

    def phi_blocks(self, ids: Sequence[int]) -> List[np.ndarray]:
        r"""Per frame block ``b``: (n, d_b, d_b) matrices $X_v|_b$, $X_v = \sum_a \phi_{v,a} G_a$."""
        return self._coords_to_blocks(self._rows(PHI_KEY, ids))

    def pos_phi_blocks(self, positions: Sequence[int]) -> Optional[List[np.ndarray]]:
        r"""Per block: (P, d_b, d_b) positional right factors $Y_p|_b$, or ``None`` if ``pos_phi="none"``.

        Reproduces ``VFEModel._pos_phi_right``: ``positional_phi_coords`` (float32, as the model
        computes it; learned tables clamp positions past their length to the last row), then the
        per-block trace projection when ``pos_phi_project_slk``. The realized frame at position
        $p$ is $U_{v,p} = e^{X_v} e^{Y_p}$.
        """
        if self.cfg.pos_phi == "none":
            return None
        from vfe3.geometry.lie_ops import project_phi_to_slk
        from vfe3.model.positional_phi import positional_phi_coords

        index = np.asarray(positions, dtype=np.int64).reshape(-1)
        if index.size == 0 or int(index.min()) < 0:
            raise ValueError("positions must be a non-empty sequence of non-negative integers")
        table = self.state.get(POS_PHI_KEY)
        coords = positional_phi_coords(
            self.cfg.pos_phi, int(index.max()) + 1, self.n_gen,
            scale=self.cfg.pos_phi_scale,
            pos_phi_free=None if table is None else table.to(torch.float32),
            device=torch.device("cpu"), dtype=torch.float32,
        )                                                                   # (N, n_gen)
        if coords is None:
            return None
        if self.cfg.pos_phi_project_slk:
            coords = project_phi_to_slk(coords, self.group.generators, self.group.irrep_dims)
        return self._coords_to_blocks(coords[torch.from_numpy(index)])

    # ---- decode tables ------------------------------------------------------------------------

    def decode_W(self, ids: Sequence[int]) -> np.ndarray:
        r"""(n, K) linear-decode weight rows ``output_proj_weight`` (logits $= \mu W^\top + b + \kappa\log\pi$)."""
        self._require_decode("linear")
        return self._rows(OUTPUT_W_KEY, ids).numpy()

    def decode_b(self, ids: Sequence[int]) -> np.ndarray:
        r"""(n,) linear-decode bias ``output_proj_bias`` (exactly zero when ``decode_bias=False``).

        The unigram prior $\kappa\log\pi_v$ is separate (:meth:`unigram_log_prior`)."""
        self._require_decode("linear")
        if OUTPUT_B_KEY in self.state:
            return self._rows(OUTPUT_B_KEY, ids).numpy()
        if self.cfg.decode_bias:
            raise KeyError(f"decode_bias=True but {OUTPUT_B_KEY!r} is absent from the checkpoint")
        return np.zeros(self._index(ids).shape[0], dtype=np.float64)

    def _decode_untied(self) -> bool:
        untied = bool(self.cfg.untie_decode_bank) and self.cfg.decode_mode != "linear"
        if untied != (DECODE_MU_KEY in self.state):
            raise ValueError(
                f"untie_decode_bank={self.cfg.untie_decode_bank} disagrees with the checkpoint's "
                f"{DECODE_MU_KEY!r} presence")
        return untied

    def decode_mu(self, ids: Sequence[int]) -> np.ndarray:
        r"""(n, K) KL-decode means $m_v$ (untied ``decode_mu_embed`` when present, else ``mu_embed``)."""
        self._require_decode("gaussian")
        return self._rows(DECODE_MU_KEY if self._decode_untied() else MU_KEY, ids).numpy()

    def decode_log_var(self, ids: Sequence[int]) -> np.ndarray:
        r"""(n, K) KL-decode log-variances $\log t_v$, $t_v = \max(\exp(\min(s, 80)), \epsilon)$."""
        from vfe3.numerics import bounded_variance_from_log

        self._require_decode("gaussian")
        s = self._rows(DECODE_SIGMA_LOG_KEY if self._decode_untied() else SIGMA_LOG_KEY, ids)
        return torch.log(bounded_variance_from_log(s, eps=self.cfg.eps)).numpy()

    def tau_eff(self) -> float:
        r"""KL-decode temperature $\tau_\mathrm{eff} = \tau\,\exp(-\operatorname{clamp}(\ell, -3, 3))$,
        $\ell$ = ``decode_log_scale``."""
        self._require_decode("gaussian")
        log_scale = self._table(DECODE_LOG_SCALE_KEY).to(torch.float64).reshape(-1)[0]
        return float(self.cfg.decode_tau * torch.exp(-log_scale.clamp(-3.0, 3.0)))

    def unigram_log_prior(self) -> Optional[np.ndarray]:
        r"""(V,) decode unigram table $\log\pi_v$ (added as $\kappa\log\pi_v$), or ``None``."""
        if UNIGRAM_KEY not in self.state:
            return None
        return self.state[UNIGRAM_KEY].to(torch.float64).numpy()

    # ---- model and data -----------------------------------------------------------------------

    def build_model(self, device: str = "cpu") -> "VFEModel":
        r"""The trained model, strictly loaded, in eval mode, replayed in fp32.

        ``amp_dtype`` is replaced by ``None``: CPU bf16 autocast is disabled deliberately so the
        analysis replays the checkpoint in float32 (recorded in ``notes``).
        """
        from vfe3.model.model import VFEModel

        cfg = dataclasses.replace(self.cfg, amp_dtype=None)
        model = VFEModel(cfg)
        model.load_state_dict(self.state, strict=True)
        model.to(device).eval()
        if self.cfg.amp_dtype is not None:
            note = (f"model replayed in float32 with autocast disabled (trained with "
                    f"amp_dtype={self.cfg.amp_dtype!r})")
            if note not in self.notes:
                self.notes.append(note)
        return model

    def token_stream(self, split: str) -> torch.Tensor:
        r"""The 1-D cached token stream of ``split`` (``load_cached_tokens``).

        When ``split`` is the checkpoint's selection split, the cache file's SHA-256 must equal the
        recorded ``selection_data_identity["source"]["sha256"]`` (else ``ValueError``): a different
        stream would silently change every downstream statistic. Other splits carry no recorded
        identity and are loaded as found.
        """
        from vfe3.data.datasets import cache_source_identity, load_cached_tokens

        if self.dataset is None:
            raise ValueError("the run's dataset is unknown; no token stream can be located")
        identity = self.selection_identity or {}
        if identity.get("split") == split:
            source = identity.get("source") or {}
            expected = source.get("sha256")
            if not expected:
                raise ValueError(f"selection identity for split {split!r} records no source sha256")
            actual = cache_source_identity(self.dataset, split, cache_dir=self.cache_dir)
            if actual["sha256"] != expected or (
                    source.get("format") is not None and actual["format"] != source["format"]):
                raise ValueError(
                    f"cached {self.dataset}/{split} stream (sha256 {actual['sha256']}, format "
                    f"{actual['format']}) differs from the checkpoint's selection identity "
                    f"(sha256 {expected}, format {source.get('format')})")
        return load_cached_tokens(self.dataset, split, cache_dir=self.cache_dir)


# ----------------------------------------------------------------------------------------------
# Loader
# ----------------------------------------------------------------------------------------------

def _block_slices(group: "GaugeGroup", K: int) -> Tuple[List[slice], Optional[str]]:
    r"""Coordinate slices of the frame blocks (``irrep_dims``), verified block diagonal.

    A lazily represented elementary block basis is block diagonal by construction; any other
    multi-block basis is checked for off-block generator support, and a basis with such support is
    treated as one ``K x K`` block (with the reason returned as a note).
    """
    dims = [int(d) for d in group.irrep_dims]
    if sum(dims) != K:
        raise ValueError(f"group irrep_dims {dims} do not sum to K={K}")
    edges = np.cumsum([0, *dims])
    slices = [slice(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]
    if len(slices) == 1 or group.elementary_block_shape is not None:
        return slices, None
    inside = torch.zeros(K, K, dtype=torch.bool)
    for s in slices:
        inside[s, s] = True
    generators = group.generators                                          # (n_gen, K, K)
    if bool((generators[:, ~inside] != 0).any()):
        return [slice(0, K)], (f"{group.name} generators have support outside the irrep_dims "
                               f"blocks {dims}: frames are analyzed as one {K}x{K} block")
    return slices, None


def _decode_kind(cfg: VFE3Config) -> str:
    if cfg.decode_mode == "linear":
        return "linear"
    if cfg.decode_mode in _KL_DECODE_MODES:
        return "gaussian"
    if cfg.decode_mode in _FAMILY_DECODE_MODES and cfg.family in _GAUSSIAN_KL_FAMILIES:
        # The family decode scores the KL divergence (Renyi order 1), the only retained divergence.
        return "gaussian"
    return "unsupported"


def _dataset_from_config_json(run_dir: Path) -> Optional[str]:
    path = run_dir / "config.json"
    if not path.is_file():
        return None
    meta = json.loads(path.read_text(encoding="utf-8"))
    dataset = meta.get("dataset") if isinstance(meta, dict) else None
    return dataset if isinstance(dataset, str) and dataset else None


def load_semantic_run(
    run_dir:    "str | Path",

    *,
    checkpoint: str = "best_model.pt",
    vocab:      Optional[Vocab] = None,
    cache_dir:  Optional[Path] = None,
    drop_unknown_config_fields: bool = False,
) -> SemanticRun:
    r"""Open ``<run_dir>/<checkpoint>`` for analysis without building the model.

    The bundle is loaded with ``torch.load(map_location="cpu", weights_only=True, mmap=True)``;
    the config is ``config_from_serialized(bundle["config"])``; the gauge group comes from
    ``build_group(cfg)`` (generators and ``irrep_dims`` only). Counts come from inverting the
    checkpoint's ``unigram_log_prior`` (:func:`counts_from_unigram_log_prior`) when present and
    set, else from a bincount of the cached training stream, else they are unavailable; the choice
    and any reason for it are recorded in ``notes``. ``cache_dir`` overrides the token-cache
    directory (default ``~/.cache/tokenized_cache``). ``drop_unknown_config_fields`` reopens a run
    trained by an older code line: config fields this build no longer declares (retired toggles)
    are dropped by :func:`vfe3.legacy_config.lenient_config_from_serialized` and listed in ``notes``.
    """
    from vfe3.model.model import build_group

    run_dir = Path(run_dir)
    checkpoint_path = run_dir / checkpoint
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"no checkpoint at {checkpoint_path}")
    bundle = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(bundle, Mapping) or not isinstance(bundle.get("model_state"), Mapping) \
            or not isinstance(bundle.get("config"), Mapping):
        raise ValueError(f"{checkpoint_path} is not a model bundle with 'model_state' and 'config'")
    notes: List[str] = []
    if drop_unknown_config_fields:
        from vfe3.legacy_config import lenient_config_from_serialized

        cfg, dropped = lenient_config_from_serialized(bundle["config"], source=str(checkpoint_path))
        if dropped:
            notes.append("config fields this code line no longer declares were dropped at load "
                         "(retired toggles): " + ", ".join(f"{k}={v!r}" for k, v in sorted(dropped.items())))
    else:
        cfg = config_from_serialized(bundle["config"], source=str(checkpoint_path))
    state: Mapping[str, torch.Tensor] = bundle["model_state"]

    selection_identity = bundle.get("selection_data_identity")
    if selection_identity is not None and not isinstance(selection_identity, Mapping):
        raise ValueError(f"{checkpoint_path} selection_data_identity must be a mapping")
    selection_identity = None if selection_identity is None else dict(selection_identity)
    dataset = (selection_identity or {}).get("dataset") or _dataset_from_config_json(run_dir)
    if selection_identity is None:
        notes.append("checkpoint carries no selection_data_identity (legacy bundle): the dataset "
                     "name comes from config.json and no token stream can be identity-checked")

    # The prior covariance table is read only where it is needed (log_var, decode_log_var): a
    # block_cholesky run stores sigma_chol_embed instead of sigma_log_embed (audit 2026-10-09 round 2, item 33).
    for key in (MU_KEY, PHI_KEY):
        if key not in state:
            raise KeyError(f"{checkpoint_path} model_state lacks {key!r}")
    V, K = (int(n) for n in state[MU_KEY].shape)
    if K != cfg.embed_dim or V != cfg.vocab_size:
        raise ValueError(f"mu_embed shape {(V, K)} disagrees with the config "
                         f"(vocab_size={cfg.vocab_size}, embed_dim={cfg.embed_dim})")
    if vocab is None:
        vocab = vocab_for_run(dataset, selection_identity, V)
    elif vocab.size != V:
        raise ValueError(f"supplied vocabulary has {vocab.size} ids; the model has {V}")

    group = build_group(cfg)
    n_gen = int(group.generator_shape[0])
    if tuple(state[PHI_KEY].shape) != (V, n_gen):
        raise ValueError(f"phi_embed shape {tuple(state[PHI_KEY].shape)} != (V, n_gen) = {(V, n_gen)}")
    block_slices, block_note = _block_slices(group, K)
    if block_note is not None:
        notes.append(block_note)

    decode_kind = _decode_kind(cfg)
    if decode_kind == "unsupported":
        notes.append(f"decode_mode={cfg.decode_mode!r} (family={cfg.family!r}) is neither linear nor "
                     f"a Gaussian KL decode: decode-table analyses do not apply")
    if cfg.encode_mode != "per_token":
        notes.append(f"encode_mode={cfg.encode_mode!r}: the tables are read as stored; the encoder "
                     f"transforms them before the first layer")
    try:
        from vfe3.metrics import covariance_frozen_at_encode
    except ImportError:
        sigma_frozen: Optional[bool] = None
        notes.append("vfe3.metrics.covariance_frozen_at_encode is not importable: Sigma-frozen status unknown")
    else:
        sigma_frozen = bool(covariance_frozen_at_encode(cfg))
    phi_frozen = True  # both supported attention modes fix frames during inference

    counts: Optional[np.ndarray] = None
    counts_source = "unavailable"
    counts_diagnostics: Dict[str, Any] = {}
    if UNIGRAM_KEY in state:
        try:
            counts, counts_diagnostics = counts_from_unigram_log_prior(state[UNIGRAM_KEY].numpy())
        except ValueError as exc:
            notes.append(f"unigram_log_prior not inverted: {exc}")
        else:
            counts_source = "unigram_log_prior"
    if counts is None and dataset is not None:
        from vfe3.data.datasets import load_cached_tokens

        try:
            train = load_cached_tokens(dataset, "train", cache_dir=cache_dir)
        except FileNotFoundError as exc:
            notes.append(f"no training-cache counts: {exc}")
        else:
            counts = counts_from_token_stream(train, V)
            counts_source = "train_cache_bincount"
            counts_diagnostics = {"N": int(counts.sum()), "n_zero": int((counts == 0).sum())}
            notes.append("counts are a bincount of the full cached training stream; a run trained "
                         "with a token cap saw a prefix of it")
    if counts is None:
        notes.append("training counts unavailable: no frequency floor or frequency analyses")
    # The token-frame analyses need learned token frames: not the identity (the rule the interventions and the
    # routing-covariance gap share, audit 2026-10-10 round 2, #46) and not the random fixed frames of 'frozen'.
    from vfe3.model.frame_actions import frames_identity_reason
    identity_reason = frames_identity_reason(cfg, n_gen, positional=False)
    if identity_reason is not None:
        notes.append(f"token frames inactive: {identity_reason}")
    elif cfg.gauge_transport == "frozen":
        notes.append("token frames inactive: gauge_transport='frozen' fixes random frames that carry no "
                     "learned geometry")

    return SemanticRun(
        run_dir=run_dir, checkpoint_path=checkpoint_path, cfg=cfg, state=state,
        selection_identity=selection_identity, dataset=dataset, vocab=vocab,
        counts=counts, counts_source=counts_source, group=group,
        K=K, n_layers=int(cfg.n_layers), n_heads=int(cfg.n_heads), n_gen=n_gen,
        block_slices=block_slices,
        frames_active=identity_reason is None and cfg.gauge_transport != "frozen",
        decode_kind=decode_kind, sigma_frozen=sigma_frozen, phi_frozen=phi_frozen,
        counts_diagnostics=counts_diagnostics, notes=notes, cache_dir=cache_dir,
    )


# ----------------------------------------------------------------------------------------------
# Analysis context
# ----------------------------------------------------------------------------------------------

def _coverage_entry(n_total: int, n_resolved: int, missing: Optional[List[str]] = None) -> Dict[str, Any]:
    entry: Dict[str, Any] = {"n_total": int(n_total), "n_resolved": int(n_resolved)}
    if missing is not None:
        entry["missing"] = list(missing)
    return entry


def continuation_fraction(stream: torch.Tensor, vocab: Vocab) -> np.ndarray:
    r"""Per-token fraction of occurrences followed by a BPE continuation piece.

    A continuation piece is a token whose text starts with a letter or digit (no leading space), so
    it glues onto the preceding token to form one word. For a whole-word-looking token $v$,
    $f_v = |\{t : x_t = v,\ x_{t+1} \text{ continues}\}| / |\{t : x_t = v\}|$; a large $f_v$ marks a
    word prefix (" Squ" + "irrel") rather than a word. Tokens that never occur get $f_v = 0$.
    Scans ``stream`` in chunks of :data:`_CONTINUATION_CHUNK` tokens. Returns (V,) float64.
    """
    V = vocab.size
    continues = np.array([len(s) > 0 and s[0].isalnum() for s in vocab.strings], dtype=bool)  # (V,)
    followed = np.zeros(V, dtype=np.float64)
    seen = np.zeros(V, dtype=np.float64)
    n = int(stream.shape[0])
    for start in range(0, n - 1, _CONTINUATION_CHUNK):
        stop = min(start + _CONTINUATION_CHUNK + 1, n)
        chunk = stream[start:stop].numpy().astype(np.int64, copy=False)    # (c+1,) ids
        cur, nxt = chunk[:-1], chunk[1:]
        seen += np.bincount(cur, minlength=V)[:V]
        followed += np.bincount(cur, weights=continues[nxt], minlength=V)[:V]
    return np.divide(followed, seen, out=np.zeros(V), where=seen > 0)


def _candidate_order(ids: np.ndarray, counts: Optional[np.ndarray]) -> np.ndarray:
    r"""``ids`` sorted by descending count (ties by ascending id); by id when counts are absent."""
    ids = np.asarray(ids, dtype=np.int64)
    if counts is None:
        return np.sort(ids)
    return ids[np.lexsort((ids, -counts[ids]))]


def prepare_context(run: SemanticRun, out_dir: Path, *, options: Mapping[str, Any]) -> AnalysisContext:
    r"""The :class:`AnalysisContext` shared by every section.

    Candidates: whole-word tokens (``vocab.word_mask()``) with count $\ge$ ``min_count`` and
    count $> 0$, sorted by descending count and cut to ``max_candidates``; every token resolved for
    a resource (fields, relations, analogies, similarity, hypernyms, axes, probes) that passes the
    same filters is appended after the cut, so resources never lose words to the truncation. With
    counts unavailable every whole-word token is a candidate (no floor, id order), which is noted.
    For an English BPE vocabulary the word prefixes (``max_continuation_fraction``) are removed
    from the non-resource candidates in both cases; ``coverage["candidates"]["prefix_filter"]``
    records what was applied. For a non-English vocabulary the whole-word mask does not apply:
    candidates are the valid-text, non-whitespace tokens under the same count filters, and the
    English resources and the prefix filter are skipped, with notes. Figures go to
    ``out_dir/<section>/`` and tables to ``out_dir/tables/<section>/``.
    """
    opts: Dict[str, Any] = {**CORE_DEFAULT_OPTIONS, **dict(options)}
    min_count = int(opts["min_count"])
    max_candidates = int(opts["max_candidates"])
    threads = opts["threads"]
    threads = int(threads) if threads is not None else max(1, (os.cpu_count() or 2) - 2)
    vocab = run.vocab
    counts = run.counts
    notes: List[str] = []

    if vocab.english:
        token_mask = vocab.word_mask()                                     # (V,) bool whole words
        kind = "whole-word"
    else:
        token_mask = vocab.valid_text & np.array([bool(s.strip()) for s in vocab.strings], dtype=bool)   # (V,)
        kind = "valid-text, non-whitespace"
        notes.append("non-English vocabulary: candidates are valid-text, non-whitespace tokens (no whole-word "
                     "mask)")
    max_cont = opts.get("max_continuation_fraction")
    prefix: Optional[np.ndarray] = None                                    # (V,) bool prefix pieces
    prefix_filter: Dict[str, Any] = {"applied": False, "max_continuation_fraction": max_cont}
    if max_cont is not None and vocab.is_bpe and not vocab.english:
        prefix_filter["reason"] = "non-English vocabulary: the whole-word prefix test does not apply"
    elif max_cont is not None and vocab.is_bpe:
        try:
            train = run.token_stream("train")
        except (FileNotFoundError, ValueError) as exc:
            prefix_filter["reason"] = f"no training stream ({exc})"
            notes.append(f"BPE prefix filter skipped: no training stream ({exc})")
        else:
            frac = continuation_fraction(train, vocab)                     # (V,)
            prefix = token_mask & (frac > float(max_cont))
            prefix_filter["applied"] = True
    if counts is not None:
        eligible = token_mask & (counts >= min_count) & (counts > 0)
    else:
        eligible = token_mask.copy()
        notes.append(f"training counts unavailable: candidates are every {kind} token "
                     "(no frequency floor, id order, no truncation)")
    ordered = _candidate_order(np.flatnonzero(eligible), counts)
    if prefix is not None:
        dropped = ordered[prefix[ordered]]
        ordered = ordered[~prefix[ordered]]                               # non-resource only; resources exempt
        prefix_filter.update(n_removed=int(dropped.size),
                             examples=[vocab.display(int(t)) for t in dropped[:20]])
    top = ordered[:max_candidates] if counts is not None else ordered
    allowed: Set[int] = set(np.flatnonzero(eligible).tolist())

    lexicon: Dict[str, List[int]] = {}
    relations: Dict[str, List[Tuple[int, int]]] = {}
    analogies: List[AnalogyQuestion] = []
    curated: List[AnalogyQuestion] = []
    similarity: Dict[str, List[Tuple[int, int, float]]] = {}
    hypernyms: List[Tuple[int, int]] = []
    axes: Dict[str, Dict[str, Any]] = {}
    probe_ids: List[int] = []
    ambiguous_ids: List[int] = []
    coverage: Dict[str, Any] = {}

    if not vocab.english:
        notes.append("non-English vocabulary: English word resources and benchmarks are skipped")
    else:
        resolved_cache: Dict[str, Optional[int]] = {}

        def rid(word: str) -> Optional[int]:
            if word not in resolved_cache:
                found, _ = resolve_words(vocab, [word], allowed)
                resolved_cache[word] = found.get(word)
            return resolved_cache[word]

        def pair_ids(pairs: Sequence[Sequence[str]]) -> Tuple[List[Tuple[int, int]], List[str]]:
            kept, missing = [], []
            for a, b in pairs:
                ia, ib = rid(a), rid(b)
                if ia is None or ib is None:
                    missing.append(f"{a}:{b}")
                else:
                    kept.append((ia, ib))
            return kept, missing

        fields = load_fields()
        per_field: Dict[str, Any] = {}
        for name, words in fields.items():
            found, missing = resolve_words(vocab, words, allowed)
            lexicon[name] = list(found.values())
            per_field[name] = _coverage_entry(len(words), len(found), missing)
        coverage["fields"] = {
            "n_total": sum(len(w) for w in fields.values()),
            "n_resolved": sum(len(v) for v in lexicon.values()),
            "per_field": per_field,
        }

        resource = load_relations()
        per_relation: Dict[str, Any] = {}
        for name, pairs in resource["relations"].items():
            kept, missing = pair_ids(pairs)
            relations[name] = kept
            per_relation[name] = _coverage_entry(len(pairs), len(kept), missing)
        coverage["relations"] = {
            "n_total": sum(len(p) for p in resource["relations"].values()),
            "n_resolved": sum(len(p) for p in relations.values()),
            "per_relation": per_relation,
        }

        missing_curated: List[str] = []
        for a, a_star, b, b_star, category in resource["curated_analogies"]:
            words = (a, a_star, b, b_star)
            ids = [rid(w) for w in words]
            if any(i is None for i in ids):
                missing_curated.append(f"{a}:{a_star}::{b}:{b_star}")
            else:
                curated.append(AnalogyQuestion(category, *ids))
        coverage["curated_analogies"] = _coverage_entry(
            len(resource["curated_analogies"]), len(curated), missing_curated)

        google = load_google_analogies()
        if google is None:
            coverage["google_analogies"] = {"available": False, "n_total": 0, "n_resolved": 0}
            notes.append("Google analogy set unavailable (gensim test data not found)")
        else:
            per_category: Dict[str, Dict[str, int]] = {}
            for category, a, a_star, b, b_star in google:
                ids = [rid(w) for w in (a, a_star, b, b_star)]
                entry = per_category.setdefault(category, {"n_total": 0, "n_resolved": 0})
                entry["n_total"] += 1
                if all(i is not None for i in ids):
                    analogies.append(AnalogyQuestion(category, *ids))
                    entry["n_resolved"] += 1
            coverage["google_analogies"] = {
                "available": True, "n_total": len(google), "n_resolved": len(analogies),
                "per_category": per_category,
            }

        coverage["similarity"] = {}
        for bench in SIMILARITY_FILES:
            rows = load_similarity(bench)
            if rows is None:
                coverage["similarity"][bench] = {"available": False, "n_total": 0, "n_resolved": 0}
                notes.append(f"similarity benchmark {bench} unavailable (gensim test data not found)")
                continue
            kept_rows = []
            for w1, w2, score in rows:
                i1, i2 = rid(w1), rid(w2)
                if i1 is not None and i2 is not None:
                    kept_rows.append((i1, i2, float(score)))
            similarity[bench] = kept_rows
            coverage["similarity"][bench] = {"available": True, "n_total": len(rows),
                                             "n_resolved": len(kept_rows)}

        per_hypernym: Dict[str, Any] = {}
        n_hyponyms = 0
        for hypernym, hyponyms in resource["hypernyms"].items():
            n_hyponyms += len(hyponyms)
            h_id = rid(hypernym)
            kept, missing = [], []
            for word in hyponyms:
                w_id = rid(word)
                if h_id is None or w_id is None:
                    missing.append(word)
                else:
                    kept.append((w_id, h_id))
            hypernyms.extend(kept)
            per_hypernym[hypernym] = {**_coverage_entry(len(hyponyms), len(kept), missing),
                                      "hypernym_resolved": h_id is not None}
        coverage["hypernyms"] = {"n_total": n_hyponyms, "n_resolved": len(hypernyms),
                                 "per_hypernym": per_hypernym}

        per_axis: Dict[str, Any] = {}
        for name, axis in resource["axes"].items():
            kept_pairs, missing_pairs = pair_ids(axis["pairs"])
            found, missing_words = resolve_words(vocab, axis["words"], allowed)
            axes[name] = {"pairs": kept_pairs, "words": list(found.values()),
                          "neg_label": axis["neg_label"], "pos_label": axis["pos_label"]}
            per_axis[name] = {
                "pairs": _coverage_entry(len(axis["pairs"]), len(kept_pairs), missing_pairs),
                "words": _coverage_entry(len(axis["words"]), len(found), missing_words),
            }
        coverage["axes"] = per_axis

        for key, target in (("probe_words", probe_ids), ("ambiguous_words", ambiguous_ids)):
            found, missing = resolve_words(vocab, resource[key], allowed)
            target.extend(found.values())
            coverage[key] = _coverage_entry(len(resource[key]), len(found), missing)

    # every resolved resource token is a candidate, appended after the frequency cut
    resource_ids: Set[int] = set()
    for ids in lexicon.values():
        resource_ids.update(ids)
    for pairs in relations.values():
        resource_ids.update(i for pair in pairs for i in pair)
    for question in (*analogies, *curated):
        resource_ids.update((question.a, question.a_star, question.b, question.b_star))
    for rows in similarity.values():
        resource_ids.update(i for w1, w2, _ in rows for i in (w1, w2))
    resource_ids.update(i for pair in hypernyms for i in pair)
    for axis in axes.values():
        resource_ids.update(i for pair in axis["pairs"] for i in pair)
        resource_ids.update(axis["words"])
    resource_ids.update(probe_ids)
    resource_ids.update(ambiguous_ids)
    in_top = set(top.tolist())
    forced = _candidate_order(np.asarray(sorted(resource_ids - in_top), dtype=np.int64), counts)
    candidate_ids = np.concatenate([top, forced]).astype(np.int64)

    coverage["candidates"] = {
        "prefix_filter":  prefix_filter,
        "min_count":      min_count if counts is not None else None,
        "max_candidates": max_candidates if counts is not None else None,
        "n_eligible":     int(eligible.sum()),
        "n_top":          int(top.size),
        "n_forced":       int(forced.size),
        "n_candidates":   int(candidate_ids.size),
        "counts_source":  run.counts_source,
    }
    coverage["notes"] = notes

    out_dir = Path(out_dir)
    figures_dir = out_dir                                                  # <out_dir>/<section>/*.png|pdf
    tables_dir = out_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    return AnalysisContext(
        out_dir=out_dir, figures_dir=figures_dir, tables_dir=tables_dir, vocab=vocab,
        candidate_ids=candidate_ids, counts=counts, lexicon=lexicon, relations=relations,
        analogies=analogies, curated_analogies=curated, similarity=similarity,
        hypernyms=hypernyms, axes=axes, probe_ids=probe_ids, ambiguous_ids=ambiguous_ids,
        coverage=coverage, seed=int(opts["seed"]), device=str(opts["device"]), threads=threads,
        options=opts,
    )
