r"""Diagnostics, probes and figures of the minimal-agent readout and the prototype-agent block MLP (2026-10-06).

References are computed independently: the label staleness from the dense posterior covariance and
``torch.distributions`` KLs, the covariance contraction from the dense :math:`C^+`, the spread share from
:func:`vfe3.model.minimal_agent.exact_record_posterior` directly. Tiny CPU models only.
"""

import json
import math
import warnings
from pathlib import Path

import pytest
import torch
from torch.distributions import MultivariateNormal, kl_divergence

import vfe3.train as T
from tests.profiles import test_config
from vfe3.model import minimal_agent as ma
from vfe3.model.model import VFEModel
from vfe3.posthoc import feature_probes
from vfe3.viz import report

torch.set_num_threads(1)
F64 = torch.float64


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _inputs(seed: int, N: int = 4, K: int = 4, H: int = 2, d: int = 2):
    g = _gen(seed)
    xi = torch.randn(N, K, generator=g, dtype=F64)
    X = torch.randn(N, K, K, generator=g, dtype=F64)
    C = X @ X.transpose(-1, -2) / K + 0.5 * torch.eye(K, dtype=F64)
    S = torch.randn(H, d, K, generator=g, dtype=F64)
    Tm = torch.randn(H, d, K, generator=g, dtype=F64)
    log_noise = torch.log(torch.tensor([0.6, 1.3], dtype=F64)[:H])
    lp = (0.3 * torch.randn(H, N, N, generator=g, dtype=F64)).masked_fill(
        ~torch.ones(N, N, dtype=torch.bool).tril(), -math.inf)
    return xi, C, S, Tm, log_noise, lp


def _dense_posterior_covariance(C: torch.Tensor, result) -> torch.Tensor:
    return C - torch.einsum("nhks,nhst,nhlt->nkl", result.gains, result.M, result.gains)


def test_label_staleness_equals_one_relabeling_at_the_posterior() -> None:
    xi, C, S, Tm, log_noise, lp = _inputs(1)
    res = ma.minimal_agent_readout(xi, C, S, Tm, log_noise, lp, "mean_field", diagnostics=True)
    C_plus = _dense_posterior_covariance(C, res)
    H, d, _ = S.shape
    N = xi.shape[0]
    want = torch.empty(N, H, dtype=F64)
    for h in range(H):
        noise = log_noise[h].exp()
        for i in range(N):
            q = MultivariateNormal(S[h] @ res.xi[i], S[h] @ C_plus[i] @ S[h].T)
            kl = torch.stack([kl_divergence(q, MultivariateNormal(
                Tm[h] @ xi[j], Tm[h] @ C[j] @ Tm[h].T + noise * torch.eye(d, dtype=F64))) for j in range(N)])
            relabeled = torch.softmax(lp[h, i] - kl, dim=0)
            want[i, h] = 0.5 * (relabeled - res.weights[h, i]).abs().sum()
    torch.testing.assert_close(res.label_staleness, want, rtol=1e-9, atol=1e-12)
    assert float(want.max()) > 1e-6                                        # a nontrivial relabeling
    exact = ma.minimal_agent_readout(xi, C, S, Tm, log_noise, lp, "exact", diagnostics=True)
    assert bool(torch.isnan(exact.label_staleness).all())


@pytest.mark.parametrize("inference", ["exact", "mean_field"])
def test_covariance_contraction_is_the_trace_ratio_of_the_dense_posterior(inference: str) -> None:
    xi, C, S, Tm, log_noise, lp = _inputs(2)
    res = ma.minimal_agent_readout(xi, C, S, Tm, log_noise, lp, inference, diagnostics=True)
    C_plus = _dense_posterior_covariance(C, res)
    want = torch.diagonal(C_plus, dim1=-2, dim2=-1).sum(-1) / torch.diagonal(C, dim1=-2, dim2=-1).sum(-1)
    torch.testing.assert_close(res.covariance_contraction, want, rtol=1e-12, atol=1e-12)
    assert bool(((want > 0) & (want < 1)).all())
    blocks = torch.stack([C[:, :2, :2], C[:, 2:, 2:]], dim=1)               # (N, 2, 2, 2) diagonal blocks
    C_block = torch.zeros_like(C)                                           # the same block-diagonal C, dense
    C_block[:, :2, :2], C_block[:, 2:, 2:] = blocks[:, 0], blocks[:, 1]
    dense = ma.minimal_agent_readout(xi, C_block, S, Tm, log_noise, lp, inference, diagnostics=True)
    blocked = ma.minimal_agent_readout(xi, blocks, S, Tm, log_noise, lp, inference, diagnostics=True)
    torch.testing.assert_close(blocked.covariance_contraction, dense.covariance_contraction, rtol=1e-10, atol=1e-12)


def test_spread_share_of_one_head_matches_the_record_posterior() -> None:
    xi, C, S, Tm, log_noise, lp = _inputs(3, H=1)
    res = ma.minimal_agent_readout(xi, C, S, Tm, log_noise, lp, "exact", diagnostics=True)
    A = S[0] @ C @ S[0].T
    B = Tm[0] @ C @ Tm[0].T + log_noise[0].exp() * torch.eye(2, dtype=F64)
    post = ma.exact_record_posterior(xi @ S[0].T, A, xi @ Tm[0].T, B, lp[0])
    want = (torch.diagonal(A @ post.R, dim1=-2, dim2=-1).sum(-1)
            / torch.diagonal(A @ post.Lambda_bar, dim1=-2, dim2=-1).sum(-1))
    torch.testing.assert_close(res.spread_share[:, 0], want, rtol=1e-10, atol=1e-12)
    assert bool((want >= 0).all()) and float(want.max()) > 0
    mf = ma.minimal_agent_readout(xi, C, S, Tm, log_noise, lp, "mean_field", diagnostics=True)
    assert bool(torch.isnan(mf.spread_share).all())


def _model(inference: str, covariance: str = "posterior") -> VFEModel:
    cfg = test_config(attention_mode="projected_predictive", predictive_readout="minimal_agent",
                      minimal_agent_inference=inference, block_mlp_mode="prototype_agents",
                      block_mlp_inference=inference, block_mlp_expansion=2, n_layers=2, max_seq_len=6,
                      attention_residual=False, predictive_covariance="moment_matched",
                      block_mlp_covariance=covariance)
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VFEModel(cfg)


@pytest.mark.parametrize("inference", ["exact", "mean_field"])
def test_eval_feature_rows_and_wrapper_removal(inference: str) -> None:
    model = _model(inference)
    windows = torch.randint(0, model.cfg.vocab_size, (3, 6), generator=_gen(4))
    features: dict = {}
    T._residual_and_attention_rows(model, windows, 5, torch.device("cpu"), features_out=features)
    assert all("forward_moments" not in mlp.__dict__ for mlp in model.block_mlps)
    heads, templates = T._feature_rows(model, features, 5)
    assert len(heads) == model.cfg.n_layers * model.cfg.n_heads
    assert all(set(row) == set(T.MINIMAL_AGENT_HEAD_FIELDS) for row in heads)
    for row in heads:
        assert 0.0 < row["gain_fraction"] < 1.0 and 0.0 < row["covariance_contraction"] < 1.0
        assert math.isnan(row["label_staleness"]) == (inference == "exact")
        assert math.isnan(row["spread_share"]) == (inference == "mean_field")
    M = model.block_mlps[0].log_pi.numel()
    assert len(templates) == model.cfg.n_layers
    for row in templates:
        assert set(row) == set(T.PROTOTYPE_TEMPLATE_FIELDS)
        assert 1.0 <= row["effective_templates"] <= M + 1e-9 and row["n_tokens"] == windows.numel()
        assert row["log_lambda_min"] == pytest.approx(math.log(10.0)) == pytest.approx(row["log_lambda_max"])
        assert row["prior_entropy"] == pytest.approx(math.log(M))
        assert 0.0 <= row["dead_fraction"] <= 1.0 and 0.0 < row["gain_fraction"] < 1.0


def test_models_without_the_features_produce_no_rows() -> None:
    cfg = test_config(attention_mode="projected_predictive", n_layers=1, max_seq_len=6)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    features: dict = {}
    T._residual_and_attention_rows(model, torch.randint(0, cfg.vocab_size, (2, 6)), 1, torch.device("cpu"),
                                   features_out=features)
    assert T._feature_rows(model, features, 1) == ([], [])
    assert feature_probes.minimal_agent_gap(model, [torch.zeros(1, 6, dtype=torch.long)], torch.device("cpu")) is None
    assert feature_probes.prototype_probe(model, [torch.zeros(1, 6, dtype=torch.long)], torch.device("cpu")) is None


def test_gap_probe_restores_the_model_and_reports_both_modes() -> None:
    model = _model("mean_field")
    ids = torch.randint(0, model.cfg.vocab_size, (2, 6), generator=_gen(5))
    before = model(ids, ids)[1].detach()
    gap = feature_probes.minimal_agent_gap(model, [ids], torch.device("cpu"))
    probe = feature_probes.prototype_probe(model, [ids], torch.device("cpu"), gap_tokens=5)
    assert torch.equal(model(ids, ids)[1].detach(), before)               # inference modes and methods restored
    assert all(ops.minimal_agent_inference == "mean_field" for ops in model.relational_operators_by_layer)
    assert all(mlp.inference == "mean_field" for mlp in model.block_mlps)
    for row in gap["layers"]:
        assert len(row["label_tv"]) == model.cfg.n_heads and all(v >= 0 for v in row["label_tv"])
        assert all(math.isfinite(v) for v in row["spread_share"])
        assert math.isfinite(row["whitened_mean_shift"]) and math.isfinite(row["logdet_ratio_per_dim"])
    for row in probe["layers"]:
        assert sum(row["usage"]) == pytest.approx(1.0) and row["gap_tokens"] == 5
        assert sum(sum(c for _, c in item["top_tokens"]) for item in row["codebook"]) == ids.numel()
        assert math.isfinite(row["logdet_ratio_per_dim"])                 # posterior covariance is returned


def test_probe_and_history_figures_are_written_and_umap_is_gated(tmp_path: Path) -> None:
    model = _model("exact")
    ids = torch.randint(0, model.cfg.vocab_size, (2, 6), generator=_gen(6))
    figdir = tmp_path / "figures"
    figdir.mkdir()
    names = {p.name for p in report.generate_feature_probe_figures(
        model, [ids], tmp_path, figdir, torch.device("cpu"), generate_embedding_figures=False)}
    assert names == {"feature_probes.json", "minimal_agent_gap.png", "template_usage.png", "template_codebook.png"}
    summary = json.loads((tmp_path / "feature_probes.json").read_text())
    assert "nu" not in summary["prototype"]["layers"][0]
    (tmp_path / "diagnostics").mkdir()
    for step in (1, 2):
        features: dict = {}
        T._residual_and_attention_rows(model, ids, step, torch.device("cpu"), features_out=features)
        heads, templates = T._feature_rows(model, features, step)
        for name, fields, rows in (("minimal_agent_heads.csv", T.MINIMAL_AGENT_HEAD_FIELDS, heads),
                                   ("prototype_templates.csv", T.PROTOTYPE_TEMPLATE_FIELDS, templates)):
            path = tmp_path / "diagnostics" / name
            lines = ([",".join(fields)] if not path.exists() else []) + [
                ",".join("" if isinstance(r[f], float) and math.isnan(r[f]) else str(r[f]) for f in fields)
                for r in rows]
            with path.open("a", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + "\n")
    history = {p.name for p in report.generate_history_figures(tmp_path, cfg=model.cfg)}
    assert {"minimal_agent_heads.png", "prototype_templates.png"} <= history


def test_template_codebook_decodes_tokens_with_the_sequence_decoder(tmp_path) -> None:
    r"""The dataset decoder takes a list of ids; the codebook labels must be the decoded words, one panel per layer."""
    from vfe3.viz import feature_figures
    calls = []

    def decode(ids):
        calls.append(ids)
        return {7: " cat", 9: ","}[ids[0]]

    probe = {"layers": [{"layer": layer, "codebook": [
        {"template": 3, "usage": 0.5, "top_tokens": [[7, 4], [9, 1]]},
        {"template": 1, "usage": 0.1, "top_tokens": []}]} for layer in range(3)]}
    import matplotlib.pyplot as plt
    fig = feature_figures.plot_template_codebook(probe, decode=decode, path=str(tmp_path / "codebook.png"))
    labels = {t.get_text() for ax in fig.axes for t in ax.get_yticklabels()}
    titled_visible = sum(ax.get_visible() for ax in fig.axes if ax.get_title())
    plt.close(fig)                                  # the plotting call leaves the figure open
    assert calls and all(isinstance(ids, list) for ids in calls)
    assert "\u2018cat\u2019  \u2018,\u2019" in labels and "(none)" in labels and "#3" in labels
    assert titled_visible == 3
