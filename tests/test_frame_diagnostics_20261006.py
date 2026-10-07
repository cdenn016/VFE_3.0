r"""Frame class-function diagnostics and the restored pullback_group config checks (2026-10-06).

(1) The projected diagnostics row carries the frame class functions again:
``gauge_invariant_mean``/``_spread``, ``gauge_head_aniso_mean``, ``gauge_head_logdet_spread`` and, for
``block_glmc``, ``gauge_head_phase_abs_mean`` and ``belief_circularity_mean``. Checked against an
independent float64 rebuild of the frames U_i = exp(phi_i . G) exp(Y_i . G) from the snapshot, and through a
train-loop metrics row. (2) The ``d_head <= 12``, ``pos_phi_project_slk`` and chart-ordering refusals of
``m_phi_update_mode='pullback_group'``, which sat after a ``raise`` and never ran, now fire. Tiny CPU models.
"""
import math
import warnings

import pytest
import torch

from tests.profiles import test_config, tiny_config
from vfe3 import metrics
from vfe3.model.model import VFEModel

FRAME_KEYS = ("gauge_invariant_mean", "gauge_invariant_spread",
              "gauge_head_aniso_mean", "gauge_head_logdet_spread")
GLMC_KEYS = ("gauge_head_phase_abs_mean", "belief_circularity_mean")


def _model(**overrides: object) -> VFEModel:
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(tiny_config(**overrides)).double().eval()
    with torch.no_grad():
        model.prior_bank.phi_embed.normal_(mean=0.0, std=0.3)
    return model


@pytest.mark.parametrize("overrides", [
    dict(gauge_group="block_glk", embed_dim=6, n_heads=2),
    dict(gauge_group="block_glk", embed_dim=6, n_heads=2, pos_phi="learned"),
    dict(gauge_group="block_glmc", embed_dim=8, n_heads=2),
    dict(gauge_group="so_n", group_n=3, irrep_spec=[("l1", 2)], embed_dim=6, n_heads=2),
])
def test_frame_keys_match_an_independent_rebuild(overrides: dict) -> None:
    model = _model(**overrides)
    tokens = torch.tensor([[1, 4, 2, 3], [2, 2, 5, 1]])
    snapshot = model.build_diagnostic_snapshot(tokens, return_logits=False)
    d = model.diagnostics(tokens, snapshot=snapshot)
    belief = snapshot.layer_converged[-1]
    G = model.group.generators.double()
    U = torch.linalg.matrix_exp(torch.einsum("...a,aij->...ij", belief.phi.double(), G))
    if belief.right_phi is not None:
        U = U @ torch.linalg.matrix_exp(torch.einsum("...a,aij->...ij", belief.right_phi.double(), G))
    U = U.reshape(-1, *U.shape[-2:])
    assert math.isnan(d["gauge_trace_spread"])                 # legacy column, NaN by contract
    inv = metrics.group_gauge_invariant(U, model.group)
    assert math.isclose(d["gauge_invariant_spread"], float(inv.std(unbiased=False)), rel_tol=1e-9, abs_tol=1e-12)
    heads = metrics.per_head_gauge_invariants(U, model.group.irrep_dims)
    assert math.isclose(d["gauge_head_logdet_spread"], float(heads["logdet"].std(unbiased=False)),
                        rel_tol=1e-9, abs_tol=1e-12)
    assert all(math.isfinite(d[k]) for k in FRAME_KEYS)
    if overrides["gauge_group"] == "block_glmc":
        assert all(math.isfinite(d[k]) for k in GLMC_KEYS)
        phase = metrics.per_head_complex_phase(U, model.group.irrep_dims).abs().mean()
        assert math.isclose(d["gauge_head_phase_abs_mean"], float(phase), rel_tol=1e-9)
        assert 0.0 <= d["belief_circularity_mean"] < 1.0
    else:
        assert not any(k in d for k in GLMC_KEYS)
    per_layer = model.diagnostics_per_layer(tokens, snapshot=snapshot)
    assert all(len(per_layer[k]) == model.cfg.n_layers for k in FRAME_KEYS)


def test_frame_keys_reach_the_training_metrics_row(tmp_path, monkeypatch) -> None:
    import csv
    import logging

    from torch.utils.data import DataLoader

    import vfe3.run_artifacts as ra
    import vfe3.train as T
    from vfe3.data.datasets import TokenWindows
    monkeypatch.setattr(T, "_save_eval_attention_maps", lambda *args, **kwargs: None)
    monkeypatch.setattr(ra, "_run_figures_isolated", lambda *args, **kwargs: True)
    cfg = test_config(vocab_size=7, embed_dim=4, n_heads=2, batch_size=2, max_seq_len=4, n_layers=1,
                      n_e_steps=1, warmup_steps=0, max_steps=4, log_interval=0, eval_interval=0,
                      checkpoint_interval=0, generate_embedding_figures=False, gauge_group="block_glmc",
                      family="gaussian_diagonal", attention_mode="projected_predictive", decode_mode="linear")
    torch.manual_seed(0)
    model = VFEModel(cfg)
    run_dir = tmp_path / "run"
    artifacts = ra.RunArtifacts(run_dir, cfg, model, save_model_artifacts=False)
    T.train(model, DataLoader(TokenWindows(torch.arange(41) % 7, 4), batch_size=2), cfg, n_steps=4,
            log_interval=2, eval_interval=2,
            val_loader=DataLoader(TokenWindows((torch.arange(23) * 3) % 7, 4, pad_final=True), batch_size=2),
            device=torch.device("cpu"), logger=logging.getLogger("frame-diag-test"), artifacts=artifacts,
            generate_samples=False)
    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows and artifacts.diagnostics_failures == {}
    for key in FRAME_KEYS + GLMC_KEYS + ("val_gauge_invariant_spread",):
        assert rows[-1].get(key, "") != "" and math.isfinite(float(rows[-1][key])), key


@pytest.mark.parametrize("bad, match", [
    (dict(embed_dim=26, n_heads=2), "d_head <= 12"),
    (dict(pos_phi="learned", pos_phi_project_slk=True), "pos_phi_project_slk"),
    (dict(transport_chart_max_norm=4.0), "transport_chart_max_norm"),
])
def test_pullback_group_checks_now_fire(bad: dict, match: str) -> None:
    base = dict(embed_dim=4, n_heads=2, gauge_group="block_glk", pos_phi="none",
                m_phi_update_mode="pullback_group", phi_precond_mode="pullback_per_block",
                transport_chart_max_norm=6.0)
    test_config(**base)                                      # the reference config is accepted
    with pytest.raises(ValueError, match=match):
        test_config(**{**base, **bad})
