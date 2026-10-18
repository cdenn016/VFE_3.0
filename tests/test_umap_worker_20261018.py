r"""The single reusable UMAP worker behind ``umap_embed`` (audit 2026-10-17 item 76). Tiny CPU arrays."""

import importlib.util
import os

import numpy as np
import pytest

from vfe3.viz import figures
from vfe3.viz.umap_worker import UMAPWorker


def test_umap_embed_degenerate_input_needs_no_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("a degenerate bank must not start a worker")

    monkeypatch.setattr(UMAPWorker, "_ensure", refuse)
    out = figures.umap_embed(np.ones((6, 3)), n_components=2)
    assert out.shape == (6, 2) and not out.any()


@pytest.mark.skipif(importlib.util.find_spec("umap") is None, reason="umap-learn is not installed")
def test_shared_worker_reuses_one_interpreter_and_matches_isolated_call() -> None:
    # find_spec, not importorskip: importing umap's native layer in the test process is the hazard
    # the worker isolates.
    X = np.random.default_rng(0).standard_normal((20, 4))
    with UMAPWorker(timeout=600.0) as worker:
        first = figures.umap_embed(X, n_neighbors=5, seed=3, worker=worker)
        pid = worker._workers[0].process.pid
        second = figures.umap_embed(X, n_neighbors=5, seed=3, worker=worker)
        assert len(worker._workers) == 1 and worker._workers[0].process.pid == pid
        workdir = worker._workdir
    assert first.shape == (20, 2) and np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)                 # seeded fit: same interpreter, same layout
    assert workdir is not None and not os.path.exists(workdir)   # close removes the request scratch
    isolated = figures.umap_embed(X, n_neighbors=5, seed=3)
    np.testing.assert_array_equal(first, isolated)
