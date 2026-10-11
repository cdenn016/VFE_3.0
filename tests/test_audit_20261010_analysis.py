r"""Audit 2026-10-10, analysis sweep and transformer-ML items owned by the analysis remediation.

#18 frame interventions and the covariance gap refuse identity frames; #30 an identical best
checkpoint is aliased to the final one; #20 the ``block_cholesky`` log det comes from the blocks and the
diagonal-marginal divergences are disclosed; #52 single-seed scaling points take the pooled pure-error
variance; #53 formula fragments are not WikiText article headings; #34 the residual-ratio panels are
titled by the run's update form; #35 ``generate_text.stream_ce`` scores overlapping windows. Tiny CPU
models and synthetic arrays only.
"""

from __future__ import annotations

from tests.profiles import test_config, tiny_config

import math
import shutil
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.posthoc import interventions as iv

FRAME_ROWS = ("token_frames_identity", "frames_removed_input_moments", "frames_removed_routing_covariance",
              "frames_removed_norm_mlp", "frames_removed_decoder")


@pytest.fixture(autouse=True)
def _two_threads() -> None:
    torch.set_num_threads(2)


def _cfg(**over: Any) -> VFE3Config:
    values = dict(vocab_size=24, embed_dim=4, n_heads=2, max_seq_len=8, eval_stride=4, n_layers=2,
                  batch_size=2, attention_mode="projected_predictive", block_mlp_expansion=2,
                  attention_residual=True, norm_type_block="none", layernorm_affine=False,
                  skip_belief_sigma_update=False)
    values.update(over)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return test_config(**values)


def _model(cfg: VFE3Config) -> VFEModel:
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VFEModel(cfg).eval()


# ----------------------------------------------------------------------------------------------
# #18: frame rows and the covariance gap on identity frames
# ----------------------------------------------------------------------------------------------

def test_frame_rows_and_covariance_gap_refuse_gauge_transport_off() -> None:
    from vfe3.posthoc.covariance import gap_applicability

    model = _model(_cfg(gauge_transport="off"))
    assert "gauge_transport='off'" in iv.frames_inactive_reason(model)
    for key in FRAME_ROWS:
        with pytest.raises(iv.NotApplicable, match="gauge_transport='off'.*measured zero"):
            with iv.by_key([key])[1].apply(model):
                pass
    reason = gap_applicability(model)
    assert reason is not None and "not a frame quantity" in reason


def test_frames_inactive_reason_cases() -> None:
    from vfe3.posthoc.covariance import gap_applicability

    on = _model(_cfg())
    assert iv.frames_inactive_reason(on) is None and gap_applicability(on) is None
    assert iv.frames_inactive_reason(_model(_cfg(gauge_transport="frozen"))) is None   # random frames act
    still = _model(_cfg(phi_scale=0.0, m_phi_lr=0.0, pos_phi="none"))
    assert "phi_scale=0" in iv.frames_inactive_reason(still)
    positional = _model(_cfg(phi_scale=0.0, m_phi_lr=0.0, pos_phi="learned"))
    assert iv.frames_inactive_reason(positional) is None                     # U_i = exp(Y_i) != I
    assert "phi_scale=0" in iv.frames_inactive_reason(positional, positional=False)   # row 7's token factor
    with pytest.raises(iv.NotApplicable, match="phi_scale=0"):
        with iv.token_frames_identity(positional):
            pass


# ----------------------------------------------------------------------------------------------
# #30: best_model.pt identical to final_model.pt is aliased
# ----------------------------------------------------------------------------------------------

def test_identical_best_checkpoint_is_aliased_not_reevaluated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tests.test_posthoc_20260928 as tp
    from vfe3.posthoc import evaluate as ev
    from vfe3.posthoc import report as rp

    run_dir, cache_dir = tp._fake_run(tmp_path)
    assert not rp._identical_model_states(run_dir / "final_model.pt", run_dir / "best_model.pt")
    shutil.copyfile(run_dir / "final_model.pt", run_dir / "best_model.pt")
    assert rp._identical_model_states(run_dir / "final_model.pt", run_dir / "best_model.pt")
    loaded = []
    original = ev.load_model
    monkeypatch.setattr(ev, "load_model", lambda run, checkpoint, **kw: loaded.append(checkpoint)
                        or original(run, checkpoint, **kw))
    options = rp.PosthocOptions(
        device="cpu", cache_dir=str(cache_dir), splits=("validation", "test"), interventions=(), maps=False,
        resources=False, figures=False, save_nll=False, latency_batches=0, bootstrap_resamples=0,
        klcap_splits=(), attention_split=None, routing=False, calibration=False)
    report = rp.run_posthoc(run_dir, options=options, log=lambda message: None)
    assert loaded == ["final_model.pt"]
    assert report["checkpoint_aliases"]["best_model.pt"]["aliased_to"] == "final_model.pt"
    passes = {(p["checkpoint"], p["split"]): p for p in report["passes"]}
    for split in ("validation", "test"):
        best, final = passes[("best_model.pt", split)], passes[("final_model.pt", split)]
        assert best["aliased_to"] == "final_model.pt" and best["ce"] == final["ce"] and best["step"] == 10
    # The reproduction check is the best checkpoint's own: log(best_val_ppl) of the (different) seed-1 model.
    reproduction = passes[("best_model.pt", "validation")]["reproduction"]
    assert reproduction["reference"]["source"] == "summary.json log(best_val_ppl)"
    assert reproduction["status"] == "mismatch"
    assert report["residual"]["best_model.pt"]["aliased_to"] == "final_model.pt"
    assert report["covariance"]["best_model.pt"]["stem"] == "best"


# ----------------------------------------------------------------------------------------------
# #20: block_cholesky log det from the blocks; divergences disclosed
# ----------------------------------------------------------------------------------------------

def test_block_cholesky_log_det_comes_from_the_blocks(tmp_path: Path) -> None:
    import tests.test_semantics_core as sc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = tiny_config(embed_dim=6, n_heads=2, vocab_size=sc.V, max_seq_len=5, n_layers=1,
                          family="gaussian_block", decode_mode="full_chunked", untie_decode_bank=True,
                          block_coordinates="reference", decode_coordinates="reference",
                          norm_type_block="layernorm", layernorm_affine=True, use_block_mlp=True,
                          block_mlp_mode="coordinate", content_link_chart="exponential", pos_phi="none",
                          gauge_transport="off", prior_covariance_table="block_cholesky")
        torch.manual_seed(0)
        model = VFEModel(cfg).eval()
    with torch.no_grad():
        model.prior_bank.sigma_chol_embed.add_(0.5 * torch.randn_like(model.prior_bank.sigma_chol_embed))
    run = sc._load(sc._save_run(tmp_path, "chol", cfg, model), cache_dir=tmp_path / "empty")
    ids = [1, 4, 7]
    with torch.no_grad():
        blocks = model.prior_bank._prior_covariance_blocks(torch.tensor(ids)).double()   # (3, H, d, d)
    expected = torch.logdet(blocks).sum(-1).numpy()
    assert run.prior_covariance_blocks is True
    np.testing.assert_allclose(run.log_det(ids), expected, rtol=1e-6, atol=1e-6)
    assert (run.log_var(ids).sum(axis=1) > run.log_det(ids) + 1e-3).all()   # Hadamard: the diagonal sum overstates


def test_sigma_section_uses_the_block_log_det_and_discloses_marginal_divergences(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tests.test_semantics_gaussians as tg
    from vfe3.semantics import gaussians as G

    monkeypatch.setattr(G, "_bank_outputs", lambda *args, **kwargs: None)       # no figures needed here
    run = tg._FakeRun(frames_active=False)
    ctx = tg._context(run, tmp_path)
    run.prior_covariance_blocks = True
    run.log_det = lambda ids: run.log_var(ids).sum(axis=1) - 1.5              # a stand-in block log det
    result = G.run_sigma_section(run, ctx)
    prior = result.metrics["banks"]["prior"]
    ids = np.asarray(ctx.candidate_ids)
    assert prior["dispersion"]["logdet_mean"] == pytest.approx(float(run.log_det(ids).mean()))
    assert result.metrics["prior_covariance"]["table"] == "block_cholesky"
    assert any("diagonal marginals" in note for note in result.notes)
    plain = tg._FakeRun(frames_active=False)
    flat = G.run_sigma_section(plain, tg._context(plain, tmp_path / "plain"))
    assert flat.metrics["prior_covariance"]["table"] == "diagonal"
    assert not any("diagonal marginals" in note for note in flat.notes)


# ----------------------------------------------------------------------------------------------
# #52: single-seed scaling points take the pooled pure-error variance
# ----------------------------------------------------------------------------------------------

def test_single_seed_points_take_the_pooled_pure_error_weight() -> None:
    from vfe3.viz.figures import _scaling_point_stats, _scaling_pooled_log_variance, _scaling_sem_weights

    points = [{"n_params": 1e6, "ce_seeds": [4.00, 4.04, 3.97]},
              {"n_params": 2e6, "ce_seeds": [3.80, 3.83, 3.78]},
              {"n_params": 4e6, "ce_seeds": [3.60, 3.61, 3.66]},
              {"n_params": 8e6, "ce_seeds": [3.45]}]
    logs = [np.log(p["ce_seeds"]) for p in points]
    s2 = sum(float(((y - y.mean()) ** 2).sum()) for y in logs) / (10 - 4)
    st = _scaling_point_stats(points)
    assert st["pooled_log_var"] == pytest.approx(s2, rel=1e-12)
    w = _scaling_sem_weights(st["mean"], st["sem"], n_seeds=st["n"], pooled_log_variance=st["pooled_log_var"])
    np.testing.assert_allclose(w[:3], (st["mean"][:3] / st["sem"][:3]) ** 2, rtol=1e-12)
    assert w[3] == pytest.approx(1.0 / s2, rel=1e-12)
    assert w[3] / w[:3].max() > 0.05                                           # no longer a vanishing weight
    assert _scaling_sem_weights(st["mean"], st["sem"])[3] == 1.0              # legacy call unchanged
    assert math.isnan(_scaling_pooled_log_variance([[3.0], [2.5]]))           # no replicate: no pooled estimate


# ----------------------------------------------------------------------------------------------
# #53: WikiText headings
# ----------------------------------------------------------------------------------------------

def test_formula_fragments_are_not_article_headings() -> None:
    from vfe3.posthoc import streams as st

    text = (b" = Alpha = \n\n\n Alpha is a thing . \n\n\n = = Section = = \n\n\n The cut-off is wc \n\n\n"
            b" = 1 rad / s and impedance k = \n\n\n 1 Ohm . This is produced by inductance L \n\n\n"
            b" = 1 henry and C = \n\n\n 1 farad . More text . \n\n\n = Beta = \n\n\n Beta starts here . \n\n\n"
            b" Key \n\n\n = Gamma = \n\n\n Gamma follows a table row . \n")
    tokens = torch.tensor(list(text), dtype=torch.long)
    ids, titles, info = st.article_ids(tokens, decode_bytes=lambda values: bytes(values))
    assert titles == ["Alpha", "Beta", "Gamma"] and info["n_articles"] == 3
    assert int(ids[text.index(b"1 Ohm")]) == 0 and int(ids[text.index(b" = Beta")]) == 1


# ----------------------------------------------------------------------------------------------
# #34: residual-ratio titles follow the update form
# ----------------------------------------------------------------------------------------------

def test_residual_ratio_panels_are_titled_by_the_update_form(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    from vfe3.viz import figures as figs
    from vfe3.viz import report

    table = {"step": np.array([10.0, 10.0]), "layer": np.array([0.0, 1.0]), "q10": np.array([0.5, 0.6]),
             "q50": np.array([1.0, 1.2]), "q90": np.array([2.0, 2.4]), "ln_rel_change_q50": np.full(2, np.nan)}
    assert figs._residual_ratio_training_panel(table)["title"].startswith("Residual ratio")
    assert figs._residual_ratio_training_panel(table, additive=False)["title"].startswith("Displacement ratio")
    fig = figs.plot_residual_ratio(table, additive=False, path=str(tmp_path / "rr.png"))
    assert "Displacement ratio" in fig._suptitle.get_text() and r"\mu^{\rm out}" in fig.axes[0].get_title()
    plt.close(fig)
    assert report._residual_additive(tmp_path, _cfg(attention_residual=True)) is True
    assert report._residual_additive(tmp_path, _cfg(attention_residual=False)) is False
    assert report._residual_additive(tmp_path, None) is True                  # no config.json: legacy title


# ----------------------------------------------------------------------------------------------
# #35: generate_text.stream_ce overlapping windows
# ----------------------------------------------------------------------------------------------

class _ConstantLogits(torch.nn.Module):
    r"""Context-free logits, so the score is the mean of a per-target-value cost."""

    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([3.0, 1.0, 0.0, -1.0, -2.0]))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.logits.expand(*inputs.shape, 5)


class _NeedsContext(torch.nn.Module):
    r"""Predicts token 0 sharply only at window positions with at least ``need`` tokens of context."""

    def __init__(self, need: int) -> None:
        super().__init__()
        self.need = need
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(*inputs.shape, 5)
        context = torch.arange(1, inputs.shape[-1] + 1)                     # tokens seen at each position
        logits[..., 0] = torch.where(context >= self.need, 30.0, 0.0)
        return logits


@pytest.mark.parametrize("n_prefix,n_tail,seq_len,stride",
                         [(0, 17, 8, 2), (3, 17, 8, 3), (9, 30, 8, 5), (1, 1, 4, 1), (20, 9, 8, 8)])
def test_stream_ce_scores_every_tail_token_once_at_any_stride(
        n_prefix: int, n_tail: int, seq_len: int, stride: int) -> None:
    import generate_text

    g = torch.Generator().manual_seed(n_prefix + 7 * n_tail)
    prefix = torch.randint(0, 5, (n_prefix,), generator=g)
    tail = torch.randint(0, 5, (n_tail,), generator=g)
    model = _ConstantLogits()
    cost = -torch.log_softmax(model.logits.detach(), dim=-1)
    scored = tail if n_prefix else tail[1:]
    expected = float(cost[scored].mean())
    actual = generate_text.stream_ce(model, prefix, tail, seq_len, stride=stride)
    assert actual == pytest.approx(expected, rel=1e-6)


def test_stream_ce_overlap_gives_every_target_the_window_context() -> None:
    import generate_text

    seq_len, stride = 8, 2
    model = _NeedsContext(seq_len - stride + 1)
    prefix, tail = torch.zeros(seq_len, dtype=torch.long), torch.zeros(23, dtype=torch.long)
    background = math.log(1.0 + 4.0 * math.exp(-30.0))
    overlapping = generate_text.stream_ce(model, prefix, tail, seq_len, stride=stride)
    disjoint = generate_text.stream_ce(model, prefix, tail, seq_len)
    assert overlapping == pytest.approx(background, rel=1e-6)
    assert disjoint > overlapping + 0.5                                        # starved first targets
    assert generate_text.scoring_stride(128) == 32 and generate_text.scoring_stride(128, 16) == 16
    with pytest.raises(ValueError, match="stride"):
        generate_text.stream_ce(model, prefix, tail, seq_len, stride=seq_len + 1)


# ----------------------------------------------------------------------------------------------
# #64: per-head cap and holonomy history figures
# ----------------------------------------------------------------------------------------------

def _write_rows(path: Path, header: str, rows: "list[tuple]") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n" + "".join(",".join(str(v) for v in row) + "\n" for row in rows),
                    encoding="utf-8")


def test_routing_head_figures_are_drawn_from_the_cap_and_holonomy_tables(tmp_path: Path) -> None:
    from vfe3.viz import figures as figs
    from vfe3.viz import report

    diag = tmp_path / "diagnostics"
    keys = [(step, layer, head) for step in (10, 20) for layer in (0, 1) for head in (0, 1)]
    _write_rows(diag / "cap_activation.csv", "step,layer,head,capped,entries",
                [(s, l, h, s + l + h, 100) for s, l, h in keys])
    _write_rows(diag / "cap_clamped_rows.csv", "step,layer,head,clamped_rows,rows",
                [(s, l, h, l, 0 if (l, h) == (1, 1) else 8) for s, l, h in keys])
    _write_rows(diag / "routing_holonomy.csv",
                "step,layer,head,path_mass,log_defect_mean,log_defect_rms,prior_log_defect_mean,"
                "prior_log_defect_rms,n_rows", [(s, l, h, 0.5, 0.1, 1.0 + h, 0.0, 2.0, 7) for s, l, h in keys])
    table = figs.routing_head_table(*(figs.read_long_table(diag / name) for name in report._ROUTING_HISTORY_TABLES))
    row = int(np.flatnonzero((table["step"] == 20) & (table["layer"] == 1) & (table["head"] == 0))[0])
    assert table["cap_fraction"][row] == pytest.approx(0.21)
    assert table["clamped_row_fraction"][row] == pytest.approx(1 / 8)
    assert table["log_defect_rms"][row] == 1.0 and table["prior_log_defect_rms"][row] == 2.0
    zero = int(np.flatnonzero((table["step"] == 20) & (table["layer"] == 1) & (table["head"] == 1))[0])
    assert math.isnan(table["clamped_row_fraction"][zero])                    # zero query rows
    assert figs.routing_head_table(None, None, None) is None
    written = {p.name for p in report.generate_history_figures(tmp_path)}
    assert written == {"routing_heads.png", "routing_heads_training.png"}
