r"""Tests for :mod:`vfe3.semantics.frames` on tiny synthetic frame tables (float64, CPU).

The priority is the mathematics: the principal real log (round trips, branch-cut rejection, the
inverse-scaling-and-squaring fallback), equivariance of the bi-invariant barycenter, invariance of
the gauge-fixed chart and of the reference content under the model's global redundancies, the exact
routing-covariance identity, exact retrieval on a planted group structure, and the relative-position
test on a one-parameter subgroup.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Sequence

import numpy as np
import pytest
import scipy.linalg as sla
import torch

from vfe3.semantics import frames as F
from vfe3.semantics.common import AnalogyQuestion, AnalysisContext, to_jsonable

_D = 3                    # block size
_B = 2                    # blocks
_K = _D * _B


def _logm_np(A: np.ndarray) -> np.ndarray:
    r"""Independent principal real log (SciPy) of a single matrix."""
    L = sla.logm(A)
    assert np.abs(np.imag(L)).max() < 1e-10
    return np.real(L)


def _batch_logm(A: np.ndarray) -> np.ndarray:
    return np.stack([_logm_np(a) for a in A])


def _expm_np(X: np.ndarray) -> np.ndarray:
    return torch.matrix_exp(torch.from_numpy(X)).numpy()


class FakeRun:
    r"""Minimal object exposing the SemanticRun accessors the frames module reads."""

    def __init__(self, X: List[np.ndarray], log_var: np.ndarray, mu: np.ndarray,
                 Y: Optional[List[np.ndarray]] = None, counts: Optional[np.ndarray] = None,
                 frames_active: bool = True) -> None:
        self._X, self._lv, self._mu, self._Y = X, log_var, mu, Y
        self.block_slices = [slice(b * X[0].shape[-1], (b + 1) * X[0].shape[-1]) for b in range(len(X))]
        self.K = log_var.shape[1]
        self.counts = counts
        self.frames_active = frames_active
        self.phi_frozen = True
        self.cfg = SimpleNamespace(pos_phi="none" if Y is None else "learned",
                                   max_seq_len=0 if Y is None else Y[0].shape[0], transport_mode="flat",
                                   norm_type_block="none", norm_type_final="none", layernorm_affine=False)

    def phi_blocks(self, ids: Sequence[int]) -> List[np.ndarray]:
        idx = np.asarray(ids, dtype=np.int64)
        return [x[idx] for x in self._X]

    def log_var(self, ids: Sequence[int]) -> np.ndarray:
        return self._lv[np.asarray(ids, dtype=np.int64)]

    def mu(self, ids: Sequence[int]) -> np.ndarray:
        return self._mu[np.asarray(ids, dtype=np.int64)]

    def pos_phi_blocks(self, positions: Sequence[int]) -> Optional[List[np.ndarray]]:
        if self._Y is None:
            return None
        idx = np.asarray(positions, dtype=np.int64)
        return [y[idx] for y in self._Y]


def _random_run(V: int = 40, P: int = 6, seed: int = 0, scale: float = 0.3) -> FakeRun:
    rng = np.random.default_rng(seed)
    X = [scale * rng.standard_normal((V, _D, _D)) for _ in range(_B)]
    Y = [0.2 * rng.standard_normal((P, _D, _D)) for _ in range(_B)]
    lv = 0.3 * rng.standard_normal((V, _K))
    mu = rng.standard_normal((V, _K))
    counts = rng.integers(100, 10000, size=V)
    return FakeRun(X, lv, mu, Y, counts)


def _transformed(run: FakeRun, *, k: Optional[List[np.ndarray]] = None, D: Optional[np.ndarray] = None,
                 h: Optional[List[np.ndarray]] = None, split: bool = False) -> FakeRun:
    r"""Apply the global redundancies to the tables.

    ``k`` (per block, right): e^X -> e^X k (and e^Y -> k^{-1} e^Y when ``split``);
    ``D`` (positive diagonal, left): mu -> D mu, log var -> log var + 2 log D, e^X -> D e^X;
    ``h`` (per block, right on Y): e^Y -> e^Y h.
    """
    X, Y = [x.copy() for x in run._X], [y.copy() for y in run._Y] if run._Y is not None else None
    lv, mu = run._lv.copy(), run._mu.copy()
    for b, sl in enumerate(run.block_slices):
        U = _expm_np(X[b])
        if D is not None:
            U = D[sl][None, :, None] * U
        if k is not None:
            U = U @ k[b]
        X[b] = _batch_logm(U)
        if Y is not None and (split or h is not None):
            E = _expm_np(Y[b])
            if split:
                E = np.linalg.inv(k[b])[None] @ E
            if h is not None:
                E = E @ h[b]
            Y[b] = _batch_logm(E)
    if D is not None:
        mu = mu * D[None, :]
        lv = lv + 2.0 * np.log(D)[None, :]
    return FakeRun(X, lv, mu, Y, run.counts)


def _group_elements(rng: np.random.Generator, scale: float = 0.3) -> List[np.ndarray]:
    return [_expm_np(scale * rng.standard_normal((_D, _D))) for _ in range(_B)]


# ----------------------------------------------------------------------------------------------
# logm_real
# ----------------------------------------------------------------------------------------------

def test_logm_real_inverts_expm_for_small_X() -> None:
    X = 0.5 * torch.randn(64, 5, 5, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    L, res, ok = F.logm_real(torch.matrix_exp(X))
    assert bool(ok.all())
    assert float((L - X).abs().max()) < 1e-12
    assert float(res.max()) < 1e-13


def test_expm_of_logm_real_reconstructs_well_conditioned_A_and_flags_the_rest() -> None:
    g = torch.Generator().manual_seed(2)
    A = torch.randn(300, 5, 5, generator=g, dtype=torch.float64) + 3.0 * torch.eye(5, dtype=torch.float64)
    L, res, ok = F.logm_real(A)
    ev = np.linalg.eigvals(A.numpy())
    expected_ok = ~((np.abs(ev.imag) == 0) & (ev.real <= 0)).any(axis=1)
    assert expected_ok.sum() > 250 and (~expected_ok).sum() > 0      # both regimes exercised
    assert np.array_equal(ok.numpy(), expected_ok)
    rel = torch.linalg.matrix_norm(torch.matrix_exp(L[ok]) - A[ok]) / torch.linalg.matrix_norm(A[ok])
    assert float(rel.max()) < 1e-11
    assert bool(torch.isnan(L[~ok]).all())


def test_logm_real_matches_scipy_principal_log() -> None:
    rng = np.random.default_rng(3)
    A = rng.standard_normal((20, 4, 4)) + 2.5 * np.eye(4)
    L, _, ok = F.logm_real(A)
    for a, l, o in zip(A, L.numpy(), ok.numpy()):
        if o:
            assert np.abs(l - _logm_np(a)).max() < 1e-10


def test_logm_real_rejects_negative_real_eigenvalues_instead_of_faking_a_log() -> None:
    rng = np.random.default_rng(4)
    Q = rng.standard_normal((3, 3))
    neg = Q @ np.diag([-1.0, 2.0, 3.0]) @ np.linalg.inv(Q)           # real eigenvalue -1
    rot_pi = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])   # rotation by pi: on the cut
    singular = np.diag([0.0, 1.0, 2.0])
    good = _expm_np(0.3 * rng.standard_normal((3, 3)))
    L, res, ok = F.logm_real(np.stack([neg, rot_pi, singular, good]))
    assert ok.tolist() == [False, False, False, True]
    assert bool(torch.isnan(L[:3]).all()) and bool(torch.isnan(res[:3]).all())
    assert bool(torch.isfinite(L[3]).all())


def test_logm_real_defective_matrix_uses_inverse_scaling_and_squaring() -> None:
    J = np.array([[2.0, 1.0, 0.0], [0.0, 2.0, 1.0], [0.0, 0.0, 2.0]])      # Jordan block: eig path fails
    L, res, ok = F.logm_real(J[None])
    assert bool(ok[0]) and float(res[0]) < 1e-13
    assert np.abs(L[0].numpy() - _logm_np(J)).max() < 1e-12


def test_inverse_scaling_and_squaring_matches_scipy() -> None:
    rng = np.random.default_rng(5)
    A = np.stack([_expm_np(1.2 * rng.standard_normal((4, 4))) for _ in range(12)])   # far from I: several roots
    L, conv = F._logm_iss(torch.from_numpy(A))
    assert bool(conv.all())
    assert np.abs(L.numpy() - _batch_logm(A)).max() < 1e-10


def test_logm_real_preserves_batch_shape() -> None:
    X = 0.2 * torch.randn(2, 3, 4, 4, generator=torch.Generator().manual_seed(6), dtype=torch.float64)
    L, res, ok = F.logm_real(torch.matrix_exp(X))
    assert L.shape == X.shape and res.shape == (2, 3) and ok.shape == (2, 3)
    assert float((L - X).abs().max()) < 1e-12


# ----------------------------------------------------------------------------------------------
# Bi-invariant barycenter
# ----------------------------------------------------------------------------------------------

def test_bi_invariant_mean_is_left_and_right_equivariant() -> None:
    g_ = torch.Generator().manual_seed(7)
    U = torch.matrix_exp(0.4 * torch.randn(25, 4, 4, generator=g_, dtype=torch.float64))
    g = torch.matrix_exp(0.6 * torch.randn(4, 4, generator=g_, dtype=torch.float64))
    h = torch.matrix_exp(0.6 * torch.randn(4, 4, generator=g_, dtype=torch.float64))
    m, info = F.bi_invariant_mean(U)
    m2, info2 = F.bi_invariant_mean(g @ U @ h)
    assert info["converged"] and info2["converged"]
    assert float(torch.linalg.matrix_norm(m2 - g @ m @ h) / torch.linalg.matrix_norm(m2)) < 1e-9
    Lr, _, _ = F.logm_real(torch.linalg.solve(m, U))                   # barycenter equation
    assert float(torch.linalg.matrix_norm(Lr.mean(0))) < 1e-9


def test_bi_invariant_mean_extends_while_contracting_and_otherwise_returns_best_iterate() -> None:
    g_ = torch.Generator().manual_seed(22)
    U = torch.matrix_exp(0.6 * torch.randn(25, 4, 4, generator=g_, dtype=torch.float64))
    m_ref, info_ref = F.bi_invariant_mean(U)
    assert info_ref["converged"] and info_ref["iterations"] >= 2
    m, info = F.bi_invariant_mean(U, max_iter=1, max_iter_cap=40)      # contracting: extended past max_iter
    assert info["converged"] and info["extended"] and info["returned"] == "fixed_point"
    assert float((m - m_ref).abs().max()) < 1e-9
    _, info4 = F.bi_invariant_mean(U, max_iter=1)                      # default cap 4 max_iter: stops unconverged
    assert info4["extended"] and not info4["converged"] and info4["iterations"] == 4
    assert info4["returned"] == "best_iterate" and info4["residual"] == min(info4["history"])
    m0, info0 = F.bi_invariant_mean(U, max_iter=0, max_iter_cap=0)     # no extension allowed
    assert not info0["converged"] and not info0["extended"] and info0["returned"] == "best_iterate"
    assert info0["iterations"] == 0 and info0["residual"] == info0["history"][0] > 1e-10


def test_bi_invariant_mean_of_commuting_elements_is_exp_of_mean_log() -> None:
    rng = np.random.default_rng(8)
    X = np.stack([np.diag(rng.standard_normal(3)) for _ in range(10)])
    m, info = F.bi_invariant_mean(_expm_np(X), log_U=X)
    assert info["converged"]
    assert np.abs(m.numpy() - _expm_np(X.mean(0))).max() < 1e-12


# ----------------------------------------------------------------------------------------------
# Gauge-fixed chart
# ----------------------------------------------------------------------------------------------

def _cf(run: FakeRun) -> F.CanonicalFrames:
    ids = np.arange(run._lv.shape[0])
    ref = ids[:30]
    return F.canonical_frames(run, ids, ref, run.log_var(ref))


def _assert_same_chart(a: F.CanonicalFrames, b: F.CanonicalFrames, tol: float = 1e-8) -> None:
    assert a.ok.all() and b.ok.all()
    for Ua, Ub, La, Lb in zip(a.U_hat, b.U_hat, a.L, b.L):
        assert float((Ua - Ub).abs().max()) < tol
        assert float((La - Lb).abs().max()) < tol


def test_canonical_frames_invariant_under_global_right_k() -> None:
    run = _random_run()
    k = _group_elements(np.random.default_rng(9), 0.4)
    _assert_same_chart(_cf(run), _cf(_transformed(run, k=k)))


def test_canonical_frames_invariant_under_global_positive_diagonal() -> None:
    run = _random_run()
    D = np.exp(0.4 * np.random.default_rng(10).standard_normal(_K))
    _assert_same_chart(_cf(run), _cf(_transformed(run, D=D)))


def test_canonical_frames_invariant_under_combined_redundancies_and_centered_on_reference() -> None:
    run = _random_run()
    rng = np.random.default_rng(11)
    cf = _cf(run)
    cf2 = _cf(_transformed(run, k=_group_elements(rng, 0.3), D=np.exp(0.3 * rng.standard_normal(_K))))
    _assert_same_chart(cf, cf2)
    for Uh in cf.U_hat:                                                 # barycenter of the chart on ref = I
        m_hat, _ = F.bi_invariant_mean(Uh[:30])
        assert float((m_hat - torch.eye(_D, dtype=torch.float64)).abs().max()) < 1e-9
    assert cf.diagnostics["barycenter_converged_all"]


def test_invariant_scalars_trace_identities_and_invariance() -> None:
    run = _random_run()
    cf = _cf(run)
    sc = F.invariant_scalars(cf)
    tr_L = sum(torch.diagonal(Lb, dim1=-2, dim2=-1).sum(-1).numpy() for Lb in cf.L)
    assert np.abs(sc["trace"] - tr_L).max() < 1e-11                    # log det U_hat = tr L
    tr_X = sum(np.trace(x, axis1=1, axis2=2) for x in run._X)
    logdet_m = sum(float(torch.linalg.slogdet(m).logabsdet) for m in cf.m)
    assert np.abs(tr_L - (tr_X - logdet_m)).max() < 1e-11              # tr L = tr X - log det m
    assert max(b["trace_identity_max_abs_error"] for b in cf.diagnostics["blocks"]) < 1e-11
    sc2 = F.invariant_scalars(_cf(_transformed(run, k=_group_elements(np.random.default_rng(12)))))
    for key in ("trace", "stretch", "rms_log_sv", "rotation", "sym_norm", "skew_norm"):
        assert np.abs(sc[key] - sc2[key]).max() < 1e-8
    assert np.all(sc["stretch"] >= 0) and np.all(np.isfinite(sc["rotation"]))


# ----------------------------------------------------------------------------------------------
# Reference content and routing covariance
# ----------------------------------------------------------------------------------------------

def _gram(Z: np.ndarray) -> np.ndarray:
    return Z @ Z.T


@pytest.mark.parametrize("eps_rel, tol", [(0.0, 1e-8), (1e-6, 1e-4)])
def test_reference_content_invariant_up_to_per_block_orthogonal_map(eps_rel: float, tol: float) -> None:
    run = _random_run(V=40)
    rng = np.random.default_rng(13)
    ids = np.arange(40)
    xi = F.reference_content(run, ids, position=3, eps_rel=eps_rel)
    k, h = _group_elements(rng, 0.3), _group_elements(rng, 0.3)
    D = np.exp(0.3 * rng.standard_normal(_K))
    variants = [
        _transformed(run, k=k, split=True),                            # (1) split redundancy
        _transformed(run, h=h),                                        # (2) global right h on Y
        _transformed(run, D=D),                                        # (3) global positive diagonal
    ]
    for other in variants:
        xo = F.reference_content(other, ids, position=3, eps_rel=eps_rel)
        for sl in run.block_slices:
            G, Go = _gram(xi[:, sl]), _gram(xo[:, sl])
            assert np.abs(G - Go).max() < tol * np.abs(G).max()
    # (1) and (3) leave the unwhitened content exactly unchanged: same rows, not only same Gram
    xo = F.reference_content(variants[0], ids, position=3, eps_rel=eps_rel)
    assert np.abs(xo - xi).max() < 1e-8


def test_reference_content_whitening() -> None:
    run = _random_run(V=60)
    xi = F.reference_content(run, np.arange(60), position=0, eps_rel=0.0)
    for sl in run.block_slices:
        C = xi[:, sl].T @ xi[:, sl] / 60
        assert np.abs(C - np.eye(_D)).max() < 1e-10
        assert np.abs(xi[:, sl].mean(0)).max() < 1e-12


def _ctx(tmp_path: Path, run: FakeRun, ids: np.ndarray, **kw) -> AnalysisContext:
    names = kw.pop("names", None)
    vocab = SimpleNamespace(display=(lambda i: names[i] if names is not None else f"w{i}"))
    base = dict(out_dir=tmp_path, figures_dir=tmp_path / "figures", tables_dir=tmp_path / "tables",
                vocab=vocab, candidate_ids=ids, counts=run.counts, lexicon={}, relations={}, analogies=[],
                curated_analogies=[], similarity={}, hypernyms=[], axes={}, probe_ids=[], ambiguous_ids=[],
                coverage={}, seed=0, options={})
    base.update(kw)
    return AnalysisContext(**base)


def test_routing_covariance_log_det_identity(tmp_path: Path) -> None:
    run = _random_run()
    ids = np.arange(40)
    out = F.routing_covariance_summary(run, _ctx(tmp_path, run, ids), ids)
    lv = run.log_var(ids)
    expected = sum(lv[:, sl].sum(1) - 2.0 * np.trace(x, axis1=1, axis2=2)
                   for sl, x in zip(run.block_slices, run._X))
    assert np.abs(out["log_det_C"] - expected).max() < 1e-12
    assert out["identity_max_abs_error"] < 1e-11                      # slogdet(C) vs sum lv - 2 tr X
    shares = out["variance_shares"]
    assert abs(shares["sigma"] + shares["frame"] + shares["cross"] - 1.0) < 1e-10
    assert out["spearman_log_count"]["log_det_C"]["n"] == 40
    assert out["log_cond_block"].shape == (40, _B) and np.all(out["log_cond_block"] >= 0)


def test_routing_covariance_gap_vanishes_for_isotropic_blocks(tmp_path: Path) -> None:
    run = _random_run()
    lv = np.repeat(np.repeat(np.random.default_rng(14).standard_normal((40, _B)), _D, axis=1), 1, axis=0)
    iso = FakeRun(run._X, lv, run._mu, run._Y, run.counts)             # var constant within each block
    ids = np.arange(40)
    out = F.routing_covariance_summary(iso, _ctx(tmp_path, iso, ids), ids)
    assert out["rel_gap"].max() < 1e-12                                # C = cbar (U^T U)^{-1} exactly


# ----------------------------------------------------------------------------------------------
# Commutators
# ----------------------------------------------------------------------------------------------

def test_commutator_diagnostics_zero_for_commuting_frames() -> None:
    rng = np.random.default_rng(15)
    X = [np.stack([np.diag(rng.standard_normal(_D)) for _ in range(20)]) for _ in range(_B)]
    run = FakeRun(X, np.zeros((20, _K)), np.zeros((20, _K)))
    out = F.commutator_diagnostics(run, np.arange(20), np.random.default_rng(0), n_pairs=30)
    assert np.nanmax(out["ratio"]) < 1e-14 and np.nanmax(out["bch_gap"]) < 1e-12
    run2 = _random_run(V=20)
    out2 = F.commutator_diagnostics(run2, np.arange(20), np.random.default_rng(0), n_pairs=30)
    assert out2["ratio"].min() > 0 and out2["bch_gap"].min() > 0
    assert out2["ratio_block"].shape == (30, _B) and out2["norm_block"].shape == (20, _B)
    Xa = [torch.from_numpy(x[:5]) for x in run2._X]
    Xb = [torch.from_numpy(x[5:10]) for x in run2._X]
    st = F._commutator_stats(Xa, Xb)
    C = Xa[0][0] @ Xb[0][0] - Xb[0][0] @ Xa[0][0]
    ref = float(torch.linalg.matrix_norm(C) / (torch.linalg.matrix_norm(Xa[0][0]) * torch.linalg.matrix_norm(Xb[0][0])))
    assert abs(st["ratio_block"][0, 0] - ref) < 1e-14


# ----------------------------------------------------------------------------------------------
# Group analogies and relation transports on a planted group structure
# ----------------------------------------------------------------------------------------------

def _planted_run(n_base: int = 12, n_noise: int = 10, seed: int = 16) -> FakeRun:
    r"""Tokens 0..n-1: base frames e^{X_i}; tokens n..2n-1: T e^{X_i} (one relation, left action);
    then ``n_noise`` random distractors."""
    rng = np.random.default_rng(seed)
    T = _group_elements(rng, 0.25)
    X = []
    for b in range(_B):
        base = 0.3 * rng.standard_normal((n_base, _D, _D))
        rel = _batch_logm(T[b][None] @ _expm_np(base))
        noise = 0.3 * rng.standard_normal((n_noise, _D, _D))
        X.append(np.concatenate([base, rel, noise]))
    V = 2 * n_base + n_noise
    lv = 0.2 * rng.standard_normal((V, _K))
    return FakeRun(X, lv, rng.standard_normal((V, _K)), None, rng.integers(100, 1000, size=V))


def test_group_analogies_exact_on_planted_structure() -> None:
    run = _planted_run()
    cf = _cf(run)
    n = 12
    qs = [AnalogyQuestion("planted", i, i + n, j, j + n) for i in range(4) for j in range(n) if j != i]
    out = F.group_analogies(cf, qs, topk=3, rerank=8)
    assert out["n_covered"] == len(qs)
    assert out["overall"]["chart"]["acc"] == 1.0
    assert out["overall"]["rerank"]["acc"] == 1.0 and out["overall"]["rerank"]["in_pool"] == 1.0
    assert all(q["top_chart"][0] == q["b_star"] for q in out["questions"])
    # chart score equals the brute-force block distance ordering
    q = qs[0]
    P = [Uh[q.a_star] @ torch.linalg.inv(Uh[q.a]) @ Uh[q.b] for Uh in cf.U_hat]
    dist = sum(((Uh - Pb[None]) ** 2).sum((1, 2)) for Uh, Pb in zip(cf.U_hat, P)).numpy()
    dist[[q.a, q.a_star, q.b]] = np.inf
    assert list(np.argsort(dist)[:3]) == out["questions"][0]["top_chart"]
    assert np.isfinite(out["bch_gap"]["median"]) and out["bch_gap"]["median"] > 0   # group != additive


def test_group_analogies_rerank_sampling_and_uncovered_questions() -> None:
    run = _planted_run()
    cf = _cf(run)
    qs = [AnalogyQuestion("planted", 0, 12, j, j + 12) for j in range(1, 12)] + [AnalogyQuestion("x", 0, 1, 2, 999)]
    out = F.group_analogies(cf, qs, rerank=5, max_rerank_questions=4, rng=np.random.default_rng(0))
    assert out["n_covered"] == 11 and out["n_reranked"] == 4
    assert sum(q["reranked"] for q in out["questions"]) == 4
    assert out["overall"]["rerank"]["n"] == 4


def test_group_analogies_chart_on_reranked_subset_and_pool_flag() -> None:
    run = _planted_run()
    cf = _cf(run)
    qs = [AnalogyQuestion("planted", 0, 12, j, j + 12) for j in range(1, 12)]
    out = F.group_analogies(cf, qs, rerank=5, max_rerank_questions=4, rng=np.random.default_rng(1))
    ov = out["overall"]
    sel = [q for q in out["questions"] if q["reranked"]]
    assert ov["chart_reranked"]["n"] == ov["rerank"]["n"] == 4 and ov["chart"]["n"] == 11
    assert ov["chart_reranked"]["acc"] == np.mean([q["rank_chart"] == 1 for q in sel])
    assert all(q["in_rerank_pool"] is None for q in out["questions"] if not q["reranked"])
    # an answer the chart ranks outside a pool of 2 has no rerank rank, and says why
    miss = F.group_analogies(cf, [AnalogyQuestion("x", 0, 12, 1, 30)], rerank=2)["questions"][0]
    assert miss["rank_chart"] > 2 and miss["rank_rerank"] is None and miss["in_rerank_pool"] is False
    assert F._rerank_rank_text(miss, 2) == "outside top-2 pool"
    assert F._rerank_rank_text({**miss, "reranked": False}, 2) == "not reranked"
    assert F._rerank_rank_text({**miss, "rank_rerank": 3}, 2) == "3"


def test_relation_transports_consistency_on_planted_relation() -> None:
    run = _planted_run()
    cf = _cf(run)
    planted = [(i, i + 12) for i in range(12)]
    rng = np.random.default_rng(17)
    random_pairs = [(int(a), int(b)) for a, b in rng.choice(np.arange(24, 34), size=(5, 2), replace=False)]
    out = F.relation_transports(cf, {"planted": planted, "random": random_pairs, "tiny": planted[:2]}, rng, n_null=200)
    p = out["planted"]
    assert p["status"] == "ok" and abs(p["consistency"] - 1.0) < 1e-10
    assert p["null_mean"] < 0.9 and p["z"] > 5 and p["p"] <= 1.0 / 201 + 1e-12
    assert out["random"]["status"] == "ok" and out["random"]["consistency"] < 0.9
    assert out["tiny"]["status"] == "skipped"
    assert p["phase_true"].size == 12 * _K and p["phase_random"].size == 12 * 11 * _K


# ----------------------------------------------------------------------------------------------
# Positional frames
# ----------------------------------------------------------------------------------------------

def test_positional_frames_one_parameter_subgroup_is_purely_relative() -> None:
    rng = np.random.default_rng(18)
    P = 16
    A = [0.15 * rng.standard_normal((_D, _D)) for _ in range(_B)]
    Y = [np.arange(P)[:, None, None] * a[None] for a in A]
    run = FakeRun([np.zeros((2, _D, _D))] * _B, np.zeros((2, _K)), np.zeros((2, _K)), Y)
    out = F.positional_frames(run)
    assert out["r2_offset"] > 1.0 - 1e-9
    assert out["r2_absolute"] < 0.9
    S = out["spectral_norm"]
    assert np.allclose(S, S.T) and np.all(np.diag(S) == 0)
    assert out["n_pairs"] == P * (P - 1) // 2 and out["s_mean_by_offset"].shape == (P - 1,)


def test_positional_frames_outlier_position_is_reported_and_excluded_in_interior_r2() -> None:
    rng = np.random.default_rng(21)
    P = 16
    A = [0.03 * rng.standard_normal((_D, _D)) for _ in range(_B)]
    Y = [np.arange(P)[:, None, None] * a[None] for a in A]
    Y[0][0] = 6.0 * rng.standard_normal((_D, _D))                     # position 0: outlying, breaks the subgroup
    run = FakeRun([np.zeros((2, _D, _D))] * _B, np.zeros((2, _K)), np.zeros((2, _K)), Y)
    out = F.positional_frames(run, n_perm=50, rng=np.random.default_rng(0))
    assert out["outlier_positions"].tolist() == [0]
    assert out["n_pairs_interior"] == (P - 1) * (P - 2) // 2
    assert out["r2_offset_interior"] > 1.0 - 1e-9
    assert out["r2_offset"] < out["r2_offset_interior"] - 0.05
    ni = out["null_interior"]["offset"]                                # interior-only permutation null
    assert out["null_interior"] is not out["null"] and ni["n_perm"] == 50
    assert abs(ni["p"] - 1.0 / 51) < 1e-15 and ni["q95"] < 0.9


def _pair_features(Y: List[np.ndarray], inverse: bool = False) -> np.ndarray:
    r"""Direct (N, 2 d B) sorted log|lambda| and sorted |arg lambda| of R_mn = e^{Y_m} e^{-Y_n}, m > n,
    per block (of R_mn^{-1} when ``inverse``), from NumPy eigenvalues."""
    mm, nn = np.tril_indices(Y[0].shape[0], k=-1)
    out = []
    for Yb in Y:
        E = _expm_np(Yb)
        R = E[mm] @ np.linalg.inv(E[nn])
        lam = np.linalg.eigvals(np.linalg.inv(R) if inverse else R)
        out.append(np.concatenate([np.sort(np.log(np.abs(lam)), axis=1), np.sort(np.abs(np.angle(lam)), axis=1)],
                                  axis=1))
    return np.concatenate(out, axis=1)


def test_positional_permutation_null_matches_direct_recomputation() -> None:
    rng = np.random.default_rng(23)
    P = 7
    Y = [0.4 * rng.standard_normal((P, _D, _D)) for _ in range(_B)]
    mm, nn = np.tril_indices(P, k=-1)
    F_fwd, F_inv = _pair_features(Y), _pair_features(Y, inverse=True)
    cols = [slice(2 * _D * b, 2 * _D * (b + 1)) for b in range(_B)]
    for c in cols:                                                     # R^{-1}: 1/lambda
        lm, ph = slice(c.start, c.start + _D), slice(c.start + _D, c.stop)
        assert np.abs(F_inv[:, lm] + F_fwd[:, lm][:, ::-1]).max() < 1e-10
        assert np.abs(F_inv[:, ph] - F_fwd[:, ph]).max() < 1e-10
    F2 = np.concatenate([F_fwd, F_inv])
    perm = rng.permutation(P)
    rows = F._permuted_pair_rows(perm, mm, nn, mm.size)
    assert np.abs(F2[rows] - _pair_features([y[perm] for y in Y])).max() < 1e-10
    draws = F._positional_r2_null(torch.from_numpy(F2 - F_fwd.mean(0)), mm, nn, np.arange(P), P, cols, 3,
                                  np.random.default_rng(5))
    gen = np.random.default_rng(5)                                     # same relabelings, recomputed from scratch
    for r in range(3):
        perm = gen.permutation(P)
        Fp = _pair_features([y[perm] for y in Y])
        for key, lab in (("offset", mm - nn), ("absolute", mm)):
            ref = [F._grouped_r2(Fp, lab)] + [F._grouped_r2(Fp[:, c], lab) for c in cols]
            assert np.abs(draws[key][r] - np.array(ref)).max() < 1e-10


def test_positional_null_flags_relative_structure_and_replaces_the_iid_chance() -> None:
    rng = np.random.default_rng(18)
    P = 16
    A = [0.15 * rng.standard_normal((_D, _D)) for _ in range(_B)]
    sub = FakeRun([np.zeros((2, _D, _D))] * _B, np.zeros((2, _K)), np.zeros((2, _K)),
                  [np.arange(P)[:, None, None] * a[None] for a in A])
    out = F.positional_frames(sub, n_perm=100, rng=np.random.default_rng(0))
    nd = out["null"]["offset"]
    assert nd["n_perm"] == 100 and abs(nd["p"] - 1.0 / 101) < 1e-15 and nd["z"] > 5 and nd["q95"] < 0.5
    assert all(abs(b["null_offset"]["p"] - 1.0 / 101) < 1e-15 for b in out["per_block"])
    assert out["null_interior"] is out["null"]                         # no outliers: interior = all pairs
    # i.i.d. positions: the pairs grouped by m share V_m, so their null sits far above (G-1)/(N-1)
    rng = np.random.default_rng(19)
    rnd = FakeRun([np.zeros((2, _D, _D))] * _B, np.zeros((2, _K)), np.zeros((2, _K)),
                  [0.3 * rng.standard_normal((P, _D, _D)) for _ in range(_B)])
    out = F.positional_frames(rnd, n_perm=100, rng=np.random.default_rng(0))
    assert out["null"]["absolute"]["mean"] > 2.0 * out["r2_chance"]
    assert out["null"]["offset"]["p"] > 0.01 and out["null"]["absolute"]["p"] > 0.01
    assert F.positional_frames(rnd, n_perm=0)["null"] is None


def test_positional_frames_random_is_not_relative_and_none_without_positions() -> None:
    rng = np.random.default_rng(19)
    P = 16
    Y = [0.3 * rng.standard_normal((P, _D, _D)) for _ in range(_B)]
    run = FakeRun([np.zeros((2, _D, _D))] * _B, np.zeros((2, _K)), np.zeros((2, _K)), Y)
    out = F.positional_frames(run)
    assert out["r2_offset"] < 0.35
    assert abs(out["r2_chance"] - (P - 2) / (P * (P - 1) // 2 - 1)) < 1e-15
    assert F.positional_frames(_planted_run()) is None                 # pos_phi == "none"


# ----------------------------------------------------------------------------------------------
# Section driver
# ----------------------------------------------------------------------------------------------

def _section_setup(tmp_path: Path, **options) -> tuple:
    r"""Planted run with positional frames and a context for :func:`frames.run_phi_section`."""
    run = _planted_run()
    rng = np.random.default_rng(20)
    run._Y = [0.2 * rng.standard_normal((6, _D, _D)) for _ in range(_B)]
    run.cfg.pos_phi, run.cfg.max_seq_len = "learned", 6
    V = run._lv.shape[0]
    names = [f"w{i}" for i in range(V)]
    names[0], names[12], names[1], names[13] = "man", "woman", "king", "queen"
    ids = np.argsort(-run.counts, kind="stable")
    ctx = _ctx(tmp_path, run, ids, names=names,
               relations={"planted": [(i, i + 12) for i in range(12)], "short": [(0, 1)]},
               analogies=[AnalogyQuestion("planted", 2, 14, j, j + 12) for j in range(3, 12)],
               curated_analogies=[AnalogyQuestion("gender", 0, 12, 1, 13)],
               options={"frame_ref_count": 20, "frame_null": 50, "frame_rerank": 6,
                        "frame_rerank_questions": 5, "frame_commutator_pairs": 40, **options})
    return run, ctx


def test_run_phi_section_on_fake_run(tmp_path: Path) -> None:
    run, ctx = _section_setup(tmp_path)
    res = F.run_phi_section(run, ctx)
    assert res.status == "ok", res.error
    assert res.metrics["failed_steps"] == []
    assert isinstance(ctx.cache["canonical_frames"], F.CanonicalFrames)
    written = {p.name for p in res.figures}
    for stem in ("overview", "invariants_vs_frequency", "group_analogies", "relation_transports", "positional"):
        assert f"{stem}.png" in written and f"{stem}.pdf" in written
    assert all(p.exists() for p in res.figures)
    assert (tmp_path / "tables" / "phi" / "group_analogies_curated.md").exists()
    assert res.headline["phi: man:woman::king:? answer rank (chart / log rerank)"] == "1 / 1"
    ov = res.metrics["group_analogies"]["benchmark"]["overall"]
    assert ov["chart"]["acc"] == 1.0 and ov["chart"]["n"] == 9 and ov["chart_reranked"]["n"] == 5
    assert res.headline["phi: group analogy acc, all covered questions: chart / ONLY-B"].endswith("(n = 9)")
    assert res.headline["phi: group analogy acc, reranked subset: chart / log rerank"] == (
        "1.000 / 1.000 (n = 5, same questions)")
    pos_key = "phi: positional R2 by offset / by absolute m (position-permutation null mean, q95, p)"
    assert res.headline[pos_key].count("null") == 2
    assert not any("chance" in k for k in res.headline)
    pos = res.metrics["positional"]
    assert pos["null"]["offset"]["n_perm"] == 200 and "null_offset" in pos["per_block"][0]
    # FakeRun's cfg lacks attention_mode, so the diagonal redundancy is assumed, not measured
    assert res.metrics["diagonal_redundancy"]["status"] == "assumed"
    assert any("assumed, not verified" in n for n in res.notes)
    assert not any("only up to O(r)" in n for n in res.notes)
    json.dumps(to_jsonable(res.metrics), allow_nan=False)             # strict-JSON serializable
    assert any("holonomy" in n for n in res.notes)


def test_diagonal_note_follows_the_shared_redundancy_status() -> None:
    from vfe3.semantics.common import diagonal_redundancy_status

    base = dict(gauge_group="block_glk", attention_mode="projected_predictive", norm_type_block="none",
                norm_type_final="none")
    exact = diagonal_redundancy_status(SimpleNamespace(**base))
    ln = diagonal_redundancy_status(SimpleNamespace(**{**base, "norm_type_block": "layernorm",
                                                       "layernorm_affine": False}))
    mixer = diagonal_redundancy_status(SimpleNamespace(**{**base, "use_head_mixer": True}))
    assert (exact[0], ln[0], mixer[0]) == ("exact", "broken", "assumed")
    assert "so D0 is a gauge fixing" in F._diagonal_note(*exact)
    assert "not a gauge fixing" in F._diagonal_note(*ln) and "LayerNorm" in F._diagonal_note(*ln)
    assert "assumed, not verified" in F._diagonal_note(*mixer) and "use_head_mixer" in F._diagonal_note(*mixer)


def test_run_phi_section_qualifies_invariance_when_barycenter_is_unconverged(tmp_path: Path,
                                                                              monkeypatch) -> None:
    from functools import partial

    run, ctx = _section_setup(tmp_path, frame_positional_null=10)
    monkeypatch.setattr(F, "canonical_frames", partial(F.canonical_frames, max_iter=0, max_iter_cap=0))
    res = F.run_phi_section(run, ctx)
    chart = res.metrics["chart"]
    assert not chart["barycenter_converged_all"] and chart["barycenter_residual_max"] > 1e-10
    r = f"{chart['barycenter_residual_max']:.1e}"
    qualified = [n for n in res.notes if "only up to O(r)" in n]
    assert len(qualified) == 2 and all(r in n for n in qualified)    # chart note and non-convergence note


def test_run_phi_section_records_a_failed_step_and_continues(tmp_path: Path, monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    run, ctx = _section_setup(tmp_path, frame_positional_null=10)
    monkeypatch.setattr(F, "relation_transports", boom)
    res = F.run_phi_section(run, ctx)
    assert res.status == "ok"
    assert res.metrics["failed_steps"] == [{"step": "relation_transports", "error": "RuntimeError: boom"}]
    assert any("relation_transports failed (RuntimeError: boom)" in n for n in res.notes)
    assert res.metrics["relation_transports"] == {} and res.metrics["positional"] is not None
    written = {p.name for p in res.figures}
    assert "positional.png" in written and "relation_transports.png" not in written
    json.dumps(to_jsonable(res.metrics), allow_nan=False)


def test_run_phi_section_skips_without_frames(tmp_path: Path) -> None:
    run = _random_run()
    run.frames_active = False
    res = F.run_phi_section(run, _ctx(tmp_path, run, np.arange(40)))
    assert res.status == "skipped" and res.notes and not res.figures


@pytest.mark.parametrize("n_ids", [0, 1])
def test_run_phi_section_skips_with_too_few_candidates(tmp_path: Path, n_ids: int) -> None:
    run = _random_run()
    res = F.run_phi_section(run, _ctx(tmp_path, run, np.arange(n_ids, dtype=np.int64)))
    assert res.status == "skipped" and res.error is None and not res.figures
    assert "candidate token" in res.notes[0]
