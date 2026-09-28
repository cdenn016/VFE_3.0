r"""Semantic geometry of a trained vfe3 checkpoint.

Given a run directory (``config.json`` + ``best_model.pt``) this package measures and plots the
semantic structure of the belief mean ``mu``, the covariance ``Sigma``, the gauge frame ``phi``,
the decode (unembedding) bank, and the per-layer contextual beliefs, using the protocols of the
word-embedding and transformer-interpretability literature (nearest neighbors, 3CosAdd/3CosMul
analogies, relation offsets, concept directions under the causal inner product, similarity
benchmarks, field clustering, anisotropy/effective rank/intrinsic dimension, logit lens, CKA,
contextuality) adapted to Gaussian beliefs and GL(K) frames.

Entry point: :func:`vfe3.semantics.report.analyze_run`; click-to-run driver ``analyze_semantics.py``.
"""

from typing import Any

__all__ = ["analyze_run", "load_semantic_run"]


def __getattr__(name: str) -> Any:
    # Lazy re-exports: importing the package must not import torch/matplotlib-heavy modules.
    if name == "analyze_run":
        from vfe3.semantics.report import analyze_run
        return analyze_run
    if name == "load_semantic_run":
        from vfe3.semantics.run import load_semantic_run
        return load_semantic_run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
