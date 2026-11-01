r"""Regression tests of the semantics-b remediation package (ultradeep audit of 2026-10-31).

* A029 (semantics-contextual-1): the contextual :math:`\mathrm{KL}(q^{(l)}_i \| q^{(0)}_i)` and the within-type
  variance shares read the family's own geometry: :math:`(\xi = U^{-1}\mu, c)` of ``gaussian_frame_diagonal``, and
  the dense frozen covariance of a ``block_cholesky`` prior table.
* A030 (semantics-contextual-2): the phi section labels its per-token frame statistics gauge-dependent where the
  token frame is pure gauge, and :math:`D_0` of the frame-diagonal family transports under the global diagonal.
* A031 (semantics-contextual-3): under a reference-coordinate decoder the contextual geometry, CKA, cosines and the
  occurrence vectors read :math:`\xi = U^{-1}\mu` and do not move under a per-type reparametrization of the tables.
* A126, A215, A216, A303, A304, A374, A375, A376: one test each (see the test names).

Tiny models only (vocabulary 24, K = 4, two heads, two layers, context 8); CPU, float32 forwards, float64 checks.
"""
from tests.profiles import config_values, test_config

import json
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
import torch

from tests import test_semantics_frames as TF
from tests.test_semantics_contextual import COUNTS, ID, N_CTX, STRINGS, V, _ctx, _stream
from vfe3.config import VFE3Config, serialize_config
from vfe3.gauge_covariant import fully_gauge_covariant_overrides, gauge_covariance_failures
from vfe3.model.model import VFEModel
from vfe3.semantics import contextual as C
from vfe3.semantics import frames as F
from vfe3.semantics import multiseed as M
from vfe3.semantics import stats
from vfe3.semantics.run import load_semantic_run
from vfe3.semantics.vocab import custom_vocab


_BASE: Dict[str, Any] = dict(vocab_size=V, embed_dim=4, n_heads=2, max_seq_len=N_CTX, n_layers=2,
                             norm_type_block="none", layernorm_affine=False, skip_belief_sigma_update=False)


@pytest.fixture(autouse=True)
def _fast_figures(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""The package style at 40 dpi with one font in the modules that draw here."""
    from vfe3.semantics import common

    original = common.apply_style

    def low_dpi_style() -> None:
        original()
        import matplotlib.pyplot as plt
        plt.rcParams.update({"savefig.dpi": 40, "font.family": ["DejaVu Sans"], "font.sans-serif": ["DejaVu Sans"]})

    for module in (C, F, M):
        monkeypatch.setattr(module, "apply_style", low_dpi_style)


# ----------------------------------------------------------------------------------------------
# Tiny runs
# ----------------------------------------------------------------------------------------------

def _model(cfg: VFE3Config, *, phi: float = 0.4, seed: int = 0) -> VFEModel:
    r"""A tiny model whose frames (``phi``), means, log-dispersions and Cholesky tables are moved off their init."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.manual_seed(seed)
        model = VFEModel(cfg).eval()
    gen = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        bank = model.prior_bank
        for name, scale in (("phi_embed", phi), ("mu_embed", 0.5), ("sigma_log_embed", 0.5)):
            p = getattr(bank, name, None)
            if p is not None and scale:
                p.add_(scale * torch.randn(p.shape, generator=gen, dtype=p.dtype))
        for name, p in bank.named_parameters():
            if "chol" in name:
                p.add_(0.4 * torch.randn(p.shape, generator=gen, dtype=p.dtype))
        if cfg.decode_unigram_prior:                                        # the run's counts come from it
            bank.set_unigram_log_prior(torch.from_numpy(COUNTS).float())
    return model


def _save(tmp_path: Path, name: str, cfg: VFE3Config, state: Dict[str, torch.Tensor],
          streams: Dict[str, Optional[torch.Tensor]]) -> Any:
    r"""Save ``state`` as a run directory, reopen it with ``load_semantic_run`` and inject synthetic streams."""
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True)
    torch.save({"model_state": state, "config": serialize_config(cfg), "selection_data_identity": None},
               run_dir / "best_model.pt")
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

    run.token_stream = token_stream
    return run


def _build(run: Any) -> VFEModel:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run.build_model()


def _capture(model: VFEModel, windows: np.ndarray) -> Dict[str, Any]:
    r"""One captured forward of ``windows``: the layer beliefs and the realized relational frame."""
    capture: Dict[str, Any] = {"diagnostic": {}}
    with torch.no_grad():
        model.forward_beliefs(torch.from_numpy(windows), capture=capture)
    diag = capture["diagnostic"]
    frame = capture.get("relational_frame")
    return {"layers": [diag["initial_belief"], *diag["layer_outputs"]],
            "frame": frame if frame is not None else capture.get("canonical_frame")}


def _dense_frame(frame: Any, K: int) -> torch.Tensor:
    r"""(n, K, K) float64 dense realized frames :math:`U_i` of a compact or dense frame context."""
    fwd = frame.forward.double()
    if frame.block_diagonal:
        G, d = fwd.shape[-3], fwd.shape[-1]
        U = torch.zeros(*fwd.shape[:-3], K, K, dtype=torch.float64)
        for g in range(G):
            U[..., g * d:(g + 1) * d, g * d:(g + 1) * d] = fwd[..., g, :, :]
        return U.reshape(-1, K, K)
    return fwd.reshape(-1, K, K)


def _dense_kl(mu_q: torch.Tensor, S_q: torch.Tensor, mu_p: torch.Tensor, S_p: torch.Tensor) -> np.ndarray:
    r""":math:`\mathrm{KL}(\mathcal N(\mu_q, S_q) \| \mathcal N(\mu_p, S_p))` per row, dense float64."""
    K = mu_q.shape[-1]
    Lp = torch.linalg.cholesky(S_p)
    d = (mu_q - mu_p).unsqueeze(-1)                                              # (n, K, 1)
    maha = (torch.cholesky_solve(d, Lp) * d).sum((-2, -1))
    tr = torch.cholesky_solve(S_q, Lp).diagonal(dim1=-2, dim2=-1).sum(-1)
    logdet_p = 2.0 * Lp.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    logdet_q = torch.linalg.slogdet(S_q).logabsdet
    return (0.5 * (tr + maha - K + logdet_p - logdet_q)).numpy()


def _type_groups(token_ids: np.ndarray, min_occ: int) -> List[np.ndarray]:
    types, counts = np.unique(token_ids, return_counts=True)
    return [np.flatnonzero(token_ids == t) for t, c in zip(types, counts) if c >= min_occ]


def _share(means: np.ndarray, var: np.ndarray, groups: List[np.ndarray]) -> float:
    r"""Mean over types of the within-type variance share
    :math:`\mathrm{tr}\,\mathrm{Cov}_c(m_c) / (\mathrm{tr}\,\mathrm{Cov}_c(m_c) + \mathrm{tr}\,\bar v)`."""
    out = []
    for g in groups:
        cov, vbar = means[g].var(axis=0), var[g].mean(axis=0)
        out.append(cov.sum() / (cov.sum() + vbar.sum()))
    return float(np.mean(out))


# ----------------------------------------------------------------------------------------------
# A029: KL and variance shares in the family's geometry
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("decode_coordinates", ["local", "reference"])
@pytest.mark.parametrize("predictive_covariance", ["passthrough", "moment_matched"], ids=["frozen", "moving"])
def test_contextual_kl_and_shares_read_the_frame_diagonal_geometry(tmp_path: Path, predictive_covariance: str,
                                                                   decode_coordinates: str) -> None:
    cfg = test_config(**_BASE, attention_mode="projected_predictive", family="gaussian_frame_diagonal",
                      decode_mode="diagonal_chunked", untie_decode_bank=True,
                      predictive_covariance=predictive_covariance, decode_coordinates=decode_coordinates)
    run = _save(tmp_path, "framediag", cfg, _model(cfg).state_dict(), {"validation": _stream(1)})
    model = _build(run)
    ctx = _ctx(tmp_path, run, min_type_occurrences=3, max_type_occurrences=50)
    bank = C.collect_bank(run, model, ctx, n_tokens=96, batch_size=4)
    assert bank.sigma_frozen is (predictive_covariance == "passthrough")
    cx = C.contextuality(bank, run, ctx)
    K, L = run.K, run.n_layers
    cap = _capture(model, bank.windows)
    U = _dense_frame(cap["frame"], K)                                          # (n, K, K) realized U_i
    assert float((U - torch.eye(K, dtype=torch.float64)).abs().max()) > 0.1    # frames well off the identity
    mus = [b.mu.double().reshape(-1, K) for b in cap["layers"]]
    cs = [b.sigma.double().reshape(-1, K) for b in cap["layers"]]             # reference variances c
    Sigma = [U @ torch.diag_embed(c) @ U.transpose(-1, -2) for c in cs]       # Sigma_i = U diag(c) U^T
    xi = [torch.linalg.solve(U, m.unsqueeze(-1)).squeeze(-1).numpy() for m in mus]
    groups = _type_groups(bank.token_ids, 3)
    assert len(groups) == cx["n_types"] >= 5
    for l in range(L + 1):
        kl = _dense_kl(mus[l], Sigma[l], mus[0], Sigma[0])
        rec = cx["layers"][l]["kl_to_stack_input"]
        assert rec["mean"] == pytest.approx(kl.mean(), rel=1e-4, abs=1e-7)
        assert rec["median"] == pytest.approx(np.median(kl), rel=1e-4, abs=1e-7)
        assert rec["q75"] == pytest.approx(np.percentile(kl, 75.0), rel=1e-4, abs=1e-7)
        if decode_coordinates == "reference":                                 # one chart: (xi, c)
            share = _share(xi[l], cs[l].numpy(), groups)
        else:                                                                 # one chart: (mu, diag Sigma)
            share = _share(mus[l].numpy(), Sigma[l].diagonal(dim1=-2, dim2=-1).numpy(), groups)
        assert cx["layers"][l]["variance_share"]["mean"] == pytest.approx(share, rel=1e-4, abs=1e-7)
    assert bank.chart == cx["chart"] == decode_coordinates
    if bank.sigma_frozen:
        assert cx["kl_form"].startswith("Mahalanobis displacement")


def test_contextual_kl_uses_the_dense_frozen_covariance_of_a_block_cholesky_table(tmp_path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = test_config(**_BASE, attention_mode="projected_predictive", family="gaussian_full",
                          decode_mode="full_chunked", prior_covariance_table="block_cholesky", gauge_transport="off",
                          pos_phi="none", untie_decode_bank=True, predictive_covariance="passthrough")
    run = _save(tmp_path, "cholesky", cfg, _model(cfg, phi=0.0).state_dict(), {"validation": _stream(2)})
    model = _build(run)
    ctx = _ctx(tmp_path, run, min_type_occurrences=3, max_type_occurrences=50)
    bank = C.collect_bank(run, model, ctx, n_tokens=96, batch_size=4)
    assert bank.sigma_frozen and bank.sigma_kind == "full"
    cx = C.contextuality(bank, run, ctx)
    K = run.K
    cap = _capture(model, bank.windows)
    S0 = cap["layers"][0].sigma.double().reshape(-1, K, K)                     # dense frozen Sigma_0
    off = S0 - torch.diag_embed(S0.diagonal(dim1=-2, dim2=-1))
    assert float(off.abs().max()) > 0.05                                       # really dense
    mu0 = cap["layers"][0].mu.double().reshape(-1, K)
    for l in range(run.n_layers + 1):
        d = (cap["layers"][l].mu.double().reshape(-1, K) - mu0).unsqueeze(-1)
        maha = 0.5 * (d * torch.linalg.solve(S0, d)).sum((-2, -1)).numpy()   # 1/2 dmu^T Sigma_0^-1 dmu
        rec = cx["layers"][l]["kl_to_stack_input"]
        assert rec["mean"] == pytest.approx(maha.mean(), rel=1e-4, abs=1e-7)
        assert rec["median"] == pytest.approx(np.median(maha), rel=1e-4, abs=1e-7)
    assert cx["kl_form"].startswith("Mahalanobis displacement")
    assert not any("diagonal only" in n for n in cx["notes"])


def test_contextual_kl_is_the_dense_gaussian_kl_when_a_full_covariance_moves(tmp_path: Path) -> None:
    cfg = test_config(**_BASE, family="gaussian_full", decode_mode="full_chunked")   # gradient E-step: Sigma moves
    run = _save(tmp_path, "full_moving", cfg, _model(cfg).state_dict(), {"validation": _stream(6)})
    model = _build(run)
    ctx = _ctx(tmp_path, run, min_type_occurrences=3, max_type_occurrences=50)
    bank = C.collect_bank(run, model, ctx, n_tokens=96, batch_size=4)
    assert not bank.sigma_frozen and bank.sigma_kind == "full"
    cx = C.contextuality(bank, run, ctx)
    K = run.K
    cap = _capture(model, bank.windows)
    mus = [b.mu.double().reshape(-1, K) for b in cap["layers"]]
    Sigma = [b.sigma.double().reshape(-1, K, K) for b in cap["layers"]]
    off = Sigma[-1] - torch.diag_embed(Sigma[-1].diagonal(dim1=-2, dim2=-1))
    assert float(off.abs().max()) > 1e-3                                       # the moved covariance is dense
    for l in range(run.n_layers + 1):
        kl = _dense_kl(mus[l], Sigma[l], mus[0], Sigma[0])
        rec = cx["layers"][l]["kl_to_stack_input"]
        assert rec["mean"] == pytest.approx(kl.mean(), rel=1e-4, abs=1e-7)
        assert rec["median"] == pytest.approx(np.median(kl), rel=1e-4, abs=1e-7)
    assert cx["kl_form"] == "dense Gaussian KL(q_l ‖ q_0)"


def test_reference_decoder_statistics_read_xi_for_a_local_diagonal_family(tmp_path: Path) -> None:
    cfg = test_config(**_BASE, attention_mode="projected_predictive", decode_mode="diagonal_chunked",
                      decode_coordinates="reference", predictive_covariance="moment_matched")
    run = _save(tmp_path, "diag_reference", cfg, _model(cfg).state_dict(), {"validation": _stream(7)})
    model = _build(run)
    ctx = _ctx(tmp_path, run, min_type_occurrences=3, max_type_occurrences=50)
    bank = C.collect_bank(run, model, ctx, n_tokens=96, batch_size=4)
    geo, cx = C.layer_geometry(bank, ctx), C.contextuality(bank, run, ctx)
    K, L = run.K, run.n_layers
    cap = _capture(model, bank.windows)
    U = _dense_frame(cap["frame"], K)
    xi = [torch.linalg.solve(U, b.mu.double().reshape(-1, K, 1)).squeeze(-1).numpy() for b in cap["layers"]]
    Uinv = torch.linalg.inv(U)
    ref_var = [(Uinv @ torch.diag_embed(b.sigma.double().reshape(-1, K)) @ Uinv.transpose(-1, -2))
               .diagonal(dim1=-2, dim2=-1).numpy() for b in cap["layers"]]     # diag C = diag U^-1 Sigma U^-T
    rows = C._token_sample(bank, ctx, int(C._opt(ctx, "contextual_points")))
    groups = _type_groups(bank.token_ids, 3)
    for l in range(L + 1):
        assert geo["layers"][l]["anisotropy_raw"] == pytest.approx(stats.anisotropy(xi[l][rows]), abs=1e-5)
        share = _share(xi[l], ref_var[l], groups)
        assert cx["layers"][l]["variance_share"]["mean"] == pytest.approx(share, rel=1e-4, abs=1e-7)
    assert any("reference means ξ" in n for n in cx["notes"]) and geo["chart"] == cx["chart"] == "reference"
    assert bank.chart == "reference" and not bank.reference_variances


# ----------------------------------------------------------------------------------------------
# A031: reference-coordinate decoders read xi in the cross-row statistics
# ----------------------------------------------------------------------------------------------

def _switched_values(**over: Any) -> Dict[str, Any]:
    r"""The fully gauge-covariant frame-diagonal route at tiny scale (the audit probe's recipe)."""
    raw = config_values(**{**_BASE, "attention_mode": "projected_predictive", "predictive_readout": "kernel",
                           "predictive_covariance": "moment_matched",
                           "lambda_alpha_mode": "state_dependent_per_coord", "family": "gaussian_frame_diagonal",
                           "decode_mode": "diagonal_chunked", "block_mlp_expansion": 2, **over})
    return {**raw, **fully_gauge_covariant_overrides(raw)}


def _per_type_state(state: Dict[str, torch.Tensor], generators: torch.Tensor,
                    g: torch.Tensor) -> Dict[str, torch.Tensor]:
    r"""The tables of the per-type action: :math:`\phi_v \to \mathrm{Log}(g_v e^{X_v})`, :math:`\mu_v \to g_v\mu_v`,
    reference variances :math:`c_v` fixed (the family stores them in the frame)."""
    from vfe3.geometry.lie_ops import embed_phi, extract_phi
    from vfe3.geometry.matrix_log import logm_principal

    G = generators.to(torch.float64)
    out = {k: v.clone() for k, v in state.items()}
    phi = state["prior_bank.phi_embed"]
    log_gU, failed, _ = logm_principal(g @ torch.linalg.matrix_exp(embed_phi(phi.double(), G)))
    assert not bool(failed.any())
    new_phi = extract_phi(log_gU, G)
    assert float((embed_phi(new_phi, G) - log_gU).abs().max()) < 1e-9
    out["prior_bank.phi_embed"] = new_phi.to(phi.dtype)
    mu = state["prior_bank.mu_embed"]
    out["prior_bank.mu_embed"] = (g @ mu.double().unsqueeze(-1)).squeeze(-1).to(mu.dtype)
    return out


def test_reference_chart_statistics_do_not_move_under_a_per_type_reparametrization(tmp_path: Path) -> None:
    cfg = VFE3Config(**_switched_values())
    assert gauge_covariance_failures(cfg) == [] and cfg.decode_coordinates == "reference"
    model = _model(cfg, phi=0.3)
    gen = torch.Generator().manual_seed(4)
    a = torch.randn(V, model.group.generators.shape[0], generator=gen, dtype=torch.float64)
    from vfe3.geometry.lie_ops import embed_phi
    g = torch.linalg.matrix_exp(0.3 * embed_phi(a, model.group.generators.to(torch.float64)))   # (V, K, K)
    streams = {"validation": _stream(3), "train": _stream(9, n=1200)}
    runs = [_save(tmp_path, "original", cfg, model.state_dict(), streams),
            _save(tmp_path, "per_type", cfg, _per_type_state(model.state_dict(), model.group.generators, g), streams)]
    models = [_build(r) for r in runs]
    x = torch.from_numpy(np.random.default_rng(5).integers(1, V, size=(3, N_CTX)))
    with torch.no_grad():
        logits = [m.forward_beliefs(x, return_logits=True)[1] for m in models]
    torch.testing.assert_close(logits[1], logits[0], rtol=1e-4, atol=1e-4)    # a redundancy of the forward map
    ctxs = [_ctx(tmp_path, r, min_type_occurrences=3, max_type_occurrences=50) for r in runs]
    banks = [C.collect_bank(r, m, c, n_tokens=96, batch_size=4) for r, m, c in zip(runs, models, ctxs)]
    assert float(np.abs(banks[0].mu - banks[1].mu).max()) > 1e-2               # the local means do move
    geo = [C.layer_geometry(b, c) for b, c in zip(banks, ctxs)]
    cka = [C.layer_cka(b, r, c) for b, r, c in zip(banks, runs, ctxs)]
    cx = [C.contextuality(b, r, c) for b, r, c in zip(banks, runs, ctxs)]
    tol = 1e-4
    for l in range(runs[0].n_layers + 1):
        for key in ("anisotropy_raw", "anisotropy_centered", "isoscore", "effective_rank", "participation_ratio"):
            assert geo[1]["layers"][l][key] == pytest.approx(geo[0]["layers"][l][key], abs=tol), (l, key)
        for key in ("self_similarity", "intra_similarity", "mev", "variance_share", "variance_share_per_coordinate"):
            assert cx[1]["layers"][l][key]["mean"] == pytest.approx(cx[0]["layers"][l][key]["mean"], abs=tol), (l, key)
        assert cx[1]["layers"][l]["kl_to_stack_input"]["median"] == pytest.approx(
            cx[0]["layers"][l]["kl_to_stack_input"]["median"], abs=tol)
    np.testing.assert_allclose(cka[1]["debiased"], cka[0]["debiased"], atol=tol)
    np.testing.assert_allclose(cka[1]["prior_mu"]["debiased"], cka[0]["prior_mu"]["debiased"], atol=tol)
    np.testing.assert_allclose(cka[1]["next_token_decode"]["debiased"], cka[0]["next_token_decode"]["debiased"],
                               atol=tol)
    words = [ID["king"], ID["queen"], ID["man"], ID["woman"], ID["dog"], ID["blue"]]
    occ = [C.collect_occurrences(r, m, c, words, per_word=3, context_len=N_CTX, budget_tokens=N_CTX * 30)
           for r, m, c in zip(runs, models, ctxs)]
    assert set(occ[0]["vectors"]) == set(occ[1]["vectors"]) and occ[0]["vectors"]
    for w in occ[0]["vectors"]:
        np.testing.assert_allclose(occ[1]["vectors"][w], occ[0]["vectors"][w], atol=tol)
    sem = [C.layerwise_semantics(o, r, c) for o, r, c in zip(occ, runs, ctxs)]
    for s in sem:
        assert s["layer0_minus_static_max_abs"] < 1e-5                          # static table in the same chart


# ----------------------------------------------------------------------------------------------
# A030: the phi section on a configuration whose token frame is pure gauge
# ----------------------------------------------------------------------------------------------

def test_token_frame_gauge_follows_the_forward_map() -> None:
    switched = VFE3Config(**_switched_values())
    assert F.token_frame_gauge(switched)["status"] == "gauge"
    bounded = VFE3Config(**{**_switched_values(), "phi_frame_condition_max": 10.0})   # an M-step bound only
    assert "no_phi_mstep_chart_cap" in gauge_covariance_failures(bounded)
    assert F.token_frame_gauge(bounded)["status"] == "gauge"
    local = test_config(**_BASE, attention_mode="projected_predictive")      # local diagonal family, local decoder
    status = F.token_frame_gauge(local)
    assert status["status"] == "content" and status["forward_gauge_failures"]
    assert F.token_frame_gauge(SimpleNamespace(norm_type_block="none"))["status"] == "unknown"


def test_phi_section_labels_per_token_statistics_where_the_token_frame_is_pure_gauge(tmp_path: Path) -> None:
    options = dict(frame_ref_count=20, frame_null=20, frame_rerank=6, frame_rerank_questions=4,
                   frame_commutator_pairs=20, frame_positional_null=10)
    cfg = VFE3Config(**_switched_values())
    run = _save(tmp_path, "switched", cfg, _model(cfg, phi=0.3).state_dict(), {})
    res = F.run_phi_section(run, _ctx(tmp_path, run, **options))
    assert res.status == "ok", res.error
    assert res.metrics["token_frame_gauge"]["status"] == "gauge"
    assert any("pure gauge" in n and "phi_log" in n for n in res.notes)
    assert res.headline["phi: per-token frame statistics"].startswith("gauge-dependent")
    from vfe3.semantics import report as R
    assert "pure gauge" in R._phi_summary(res)
    view = R.build_views(run, _ctx(tmp_path, run, **options, views=("phi_log",)))["phi_log"]
    assert "pure gauge" in view.note
    local_cfg = test_config(**_BASE, attention_mode="projected_predictive")
    local = _save(tmp_path, "local", local_cfg, _model(local_cfg, phi=0.3).state_dict(), {})
    res_local = F.run_phi_section(local, _ctx(tmp_path, local, **options))
    assert res_local.status == "ok", res_local.error
    assert res_local.metrics["token_frame_gauge"]["status"] == "content"
    assert not any("pure gauge" in n for n in res_local.notes)
    assert "phi: per-token frame statistics" not in res_local.headline
    assert "pure gauge" not in R._phi_summary(res_local)
    local_view = R.build_views(local, _ctx(tmp_path, local, **options, views=("phi_log",)))["phi_log"]
    assert "pure gauge" not in local_view.note


def _frame_diagonal_fake(X: List[np.ndarray], log_c: np.ndarray, mu: np.ndarray, Y: Optional[List[np.ndarray]],
                         counts: np.ndarray) -> TF.FakeRun:
    run = TF.FakeRun(X, log_c, mu, Y, counts)
    run.reference_variances = True                                         # log_var returns log c_v
    return run


def test_canonical_frames_of_the_frame_diagonal_family_are_invariant_under_the_global_diagonal() -> None:
    base = TF._random_run()
    run = _frame_diagonal_fake(base._X, base._lv, base._mu, base._Y, base.counts)
    D = np.exp(0.4 * np.random.default_rng(10).standard_normal(TF._K))
    # U -> D U, mu -> D mu, c fixed: the global positive diagonal of gaussian_frame_diagonal
    X2 = [TF._batch_logm(D[sl][None, :, None] * TF._expm_np(x)) for x, sl in zip(base._X, base.block_slices)]
    rescaled = _frame_diagonal_fake(X2, base._lv.copy(), base._mu * D[None, :], base._Y, base.counts)
    cf, cf2 = TF._cf(run), TF._cf(rescaled)
    TF._assert_same_chart(cf, cf2)
    sc, sc2 = F.invariant_scalars(cf), F.invariant_scalars(cf2)
    for key in ("trace", "stretch", "rms_log_sv", "rotation", "sym_norm", "skew_norm"):
        assert np.abs(sc[key] - sc2[key]).max() < 1e-8, key
    k = TF._group_elements(np.random.default_rng(12), 0.3)                  # the split redundancy still holds
    split = TF._transformed(base, k=k, split=True)
    TF._assert_same_chart(cf, TF._cf(_frame_diagonal_fake(split._X, split._lv, split._mu, split._Y, split.counts)))


# ----------------------------------------------------------------------------------------------
# A374: layer 0 against the token-table prior
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("encode_mode, equal", [("per_token", True), ("per_token_additive", False)])
def test_bank_compares_the_stack_input_with_the_token_table(tmp_path: Path, encode_mode: str, equal: bool) -> None:
    cfg = test_config(**_BASE, encode_mode=encode_mode, pos_phi="none", decode_mode="diagonal_chunked",
                      lambda_alpha_mode="state_dependent")
    run = _save(tmp_path, encode_mode, cfg, _model(cfg).state_dict(), {"validation": _stream(4)})
    bank = C.collect_bank(run, _build(run), _ctx(tmp_path, run), n_tokens=40, batch_size=4)
    diag = bank.diagnostics
    assert diag["stack_input_equals_token_prior"] is equal
    assert "encoded_equals_stack_input" not in diag
    assert (diag["stack_input_minus_token_prior_max_abs_mu"] == 0.0) is equal
    assert any("differs from the token-table prior" in n for n in bank.notes) is (not equal)


# ----------------------------------------------------------------------------------------------
# A215: the logit lens decodes the final layer once per chunk
# ----------------------------------------------------------------------------------------------

def test_logit_lens_requests_forward_logits_only_for_the_contract_check(tmp_path: Path) -> None:
    cfg = test_config(**_BASE, attention_mode="projected_predictive", decode_mode="diagonal_chunked",
                      untie_decode_bank=True)
    run = _save(tmp_path, "lens", cfg, _model(cfg).state_dict(), {"validation": _stream(5)})
    model = _build(run)
    ctx = _ctx(tmp_path, run)
    bank = C.collect_bank(run, model, ctx, n_tokens=64)
    calls = {"decode": 0, "return_logits": []}
    decode, forward = model._decode_belief_with_context, model.forward_beliefs

    def counting_decode(*args: Any, **kwargs: Any) -> torch.Tensor:
        calls["decode"] += 1
        return decode(*args, **kwargs)

    def recording_forward(x: torch.Tensor, **kwargs: Any) -> Any:
        calls["return_logits"].append(bool(kwargs.get("return_logits")))
        return forward(x, **kwargs)

    model._decode_belief_with_context = counting_decode
    model.forward_beliefs = recording_forward
    lens = C.logit_lens(run, model, bank, ctx, max_tokens=64, chunk_tokens=16)   # 8 windows in 4 chunks of 2
    L = run.n_layers
    assert calls["return_logits"] == [True, False, False, False]
    assert calls["decode"] == 4 * (L + 1) + 1                   # L + 1 readouts per chunk, one forward decode
    assert lens["consistency"]["max_abs_logit_diff_final_vs_forward"] == 0.0
    assert lens["consistency"]["windows_checked"] == 2


# ----------------------------------------------------------------------------------------------
# A216: barycenter iterations check the reconstruction only where the eigenvectors are ill-conditioned
# ----------------------------------------------------------------------------------------------

def test_barycenter_iterations_skip_the_reconstruction_check_of_well_conditioned_rows(monkeypatch) -> None:
    g_ = torch.Generator().manual_seed(7)
    X = 0.4 * torch.randn(25, 4, 4, generator=g_, dtype=torch.float64)
    U = torch.matrix_exp(X)
    checked: List[int] = []
    original = F._relative_residual

    def counting(L: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        checked.append(int(A.shape[0]))
        return original(L, A)

    monkeypatch.setattr(F, "_relative_residual", counting)
    m_full, info_full = F.bi_invariant_mean(U, log_U=X, screen_cond=None)     # every iterate fully checked
    n_full = sum(checked)
    checked.clear()
    m, info = F.bi_invariant_mean(U, log_U=X)
    assert info_full["converged"] and info["converged"] and info["iterations"] == info_full["iterations"]
    assert n_full == U.shape[0] * (info_full["iterations"] + 1)
    assert sum(checked) == U.shape[0]                                          # only the returned iterate
    assert float((m - m_full).abs().max()) < 1e-12
    assert info["residual_checked"] == pytest.approx(info["residual"], abs=1e-13)
    assert info["n_log_failed"] == 0 and info["screen"]["rows_checked"] == 0


def test_screened_log_routes_ill_conditioned_rows_to_the_checked_log() -> None:
    g_ = torch.Generator().manual_seed(3)
    A = torch.matrix_exp(0.3 * torch.randn(6, 3, 3, generator=g_, dtype=torch.float64))
    near = torch.tensor([[1.0, 100.0, 0.0], [0.0, 1.0 + 1e-7, 0.0], [0.0, 0.0, 2.0]], dtype=torch.float64)
    A[2] = near                                                                # nearly defective: kappa(V) ~ 1e9
    neg = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64))
    A[4] = neg                                                                 # no real principal log
    L, ok = F._logm_screened(A, cond_max=1e4)
    L_ref, _, ok_ref = F.logm_real(A)
    assert ok.tolist() == ok_ref.tolist() and not bool(ok[4]) and bool(ok[2])
    assert bool(torch.isnan(L[4]).all())
    torch.testing.assert_close(L[ok], L_ref[ok], rtol=0.0, atol=1e-10)
    torch.testing.assert_close(L[2], L_ref[2], rtol=0.0, atol=1e-12)          # the checked path itself


# ----------------------------------------------------------------------------------------------
# A375: the Holm-adjusted relation-transport test can reject at the default budget
# ----------------------------------------------------------------------------------------------

def test_default_null_budget_lets_the_holm_adjusted_transport_test_reject() -> None:
    import inspect

    n_null = int(F.PHI_DEFAULTS["frame_null"])
    assert n_null >= 2000
    assert 10 / (1.0 + n_null) < F._TRANSPORT_ALPHA                           # ten curated relations
    assert inspect.signature(F.relation_transports).parameters["n_null"].default == n_null


# ----------------------------------------------------------------------------------------------
# A303 and A304: docstrings and output text
# ----------------------------------------------------------------------------------------------

def test_docstrings_name_the_live_frame_contract() -> None:
    assert "transport_mode" not in F.__doc__ and "gauge_transport='on'" in F.__doc__
    for doc in (C.__doc__, C._readout.__doc__):
        assert "_decoder_reads_frame" in doc and "decode_coordinates='reference' only" not in doc


def test_multiseed_text_names_the_sample_sd_and_the_candidate_battery(tmp_path: Path) -> None:
    payload = {"created": "2026-11-01T00:00:00", "data_order": {"caveat": "caveat"},
               "config_homogeneity": {"homogeneous": True, "differences": {}}, "analysis_option_differences": {},
               "runs": [], "metrics": {}, "stability": {"status": "skipped", "reason": "test"}, "figures": [],
               "notes": []}
    text = M._render_markdown(payload, [], tmp_path)
    for doc in (text, M.__doc__, M.summarize.__doc__):
        flat = " ".join(doc.split())
        assert "unbiased SD" not in flat and "``sd`` is the unbiased" not in flat and "full-vocabulary" not in flat
    assert "sample SD (n − 1 denominator)" in text and "candidate-vocabulary battery" in text
    assert "full-vocabulary" not in C.contextual_summary({"layerwise_semantics": {
        "n_words": 9, "layers": [{"analogies": {"n_covered": 3, "overall": {"3cosadd": {"mean": 0.5}}}}]}})


# ----------------------------------------------------------------------------------------------
# A126: configuration homogeneity across seeds
# ----------------------------------------------------------------------------------------------

def test_config_homogeneity_ignores_number_spelling_and_cadence_fields() -> None:
    a = test_config(rope_fraction=1, seed=1)
    b = test_config(rope_fraction=1.0, seed=2, resume_from="elsewhere/checkpoints/step_10.pt", log_interval=7,
                    checkpoint_interval=3)
    assert type(a.rope_fraction) is int and type(b.rope_fraction) is float
    same = M.config_homogeneity([a, b], ["seed 1", "seed 2"])
    assert same["homogeneous"], same["differences"]
    assert {"resume_from", "log_interval", "checkpoint_interval", "seed"} <= set(same["exempt"])
    differ = M.config_homogeneity([a, test_config(rope_fraction=1.0, embed_dim=6, n_heads=2)], ["a", "b"])
    assert set(differ["differences"]) == {"embed_dim"} and differ["differences"]["embed_dim"] == {"a": 4, "b": 6}
    plain = M.config_homogeneity([{"lr": 1, "seed": 0}, {"lr": 1.0, "seed": 3}], ["a", "b"])
    assert plain["homogeneous"]


# ----------------------------------------------------------------------------------------------
# A376: one term-reusing CKA in stats, called by multiseed
# ----------------------------------------------------------------------------------------------

def _linear_cka_reference(X: np.ndarray, Y: np.ndarray, debiased: bool) -> float:
    r"""The pre-remediation :func:`stats.linear_cka` arithmetic, verbatim (numpy float64)."""
    X, Y = X - X.mean(axis=0), Y - Y.mean(axis=0)
    n = X.shape[0]
    xty = float(np.linalg.norm(X.T @ Y) ** 2)
    xtx, yty = float(np.linalg.norm(X.T @ X) ** 2), float(np.linalg.norm(Y.T @ Y) ** 2)
    if not debiased:
        return xty / np.sqrt(xtx * yty)
    ssr_x, ssr_y = (X * X).sum(axis=1), (Y * Y).sum(axis=1)
    sq_x, sq_y = float(ssr_x.sum()), float(ssr_y.sum())

    def dot(t: float, a: np.ndarray, b: np.ndarray, sa: float, sb: float) -> float:
        return float(t - n / (n - 2.0) * float(a @ b) + sa * sb / ((n - 1.0) * (n - 2.0)))

    return dot(xty, ssr_x, ssr_y, sq_x, sq_y) / np.sqrt(dot(xtx, ssr_x, ssr_x, sq_x, sq_x)
                                                         * dot(yty, ssr_y, ssr_y, sq_y, sq_y))


def test_cka_terms_are_shared_by_stats_and_multiseed(monkeypatch) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 5)) @ rng.normal(size=(5, 5))
    Y = X @ rng.normal(size=(5, 3)) + 0.5 * rng.normal(size=(60, 3))
    for debiased in (False, True):
        assert stats.linear_cka(X, Y, debiased=debiased) == _linear_cka_reference(X, Y, debiased)   # bitwise
    calls = {"terms": 0, "cka": 0}
    terms, from_terms = stats.cka_self_terms, stats.cka_from_terms

    def counting_terms(Xc: Any) -> Dict[str, Any]:
        calls["terms"] += 1
        return terms(Xc)

    def counting_cka(*args: Any, **kwargs: Any) -> Any:
        calls["cka"] += 1
        return from_terms(*args, **kwargs)

    monkeypatch.setattr(stats, "cka_self_terms", counting_terms)
    monkeypatch.setattr(stats, "cka_from_terms", counting_cka)
    ta, tb = (torch.as_tensor(Z - Z.mean(axis=0), dtype=torch.float32) for Z in (X, Y))
    biased, debiased = M.cka_pair(ta, tb, M._cka_self_terms(ta), M._cka_self_terms(tb))
    assert calls == {"terms": 2, "cka": 1}
    assert biased == pytest.approx(stats.linear_cka(X, Y), abs=1e-5)
    assert debiased == pytest.approx(stats.linear_cka(X, Y, debiased=True), abs=1e-5)
    assert np.isnan(stats.cka_from_terms(0.0, stats.cka_self_terms(np.zeros((5, 2))),
                                         stats.cka_self_terms(np.ones((5, 2)) - 1.0), 5)[1])
