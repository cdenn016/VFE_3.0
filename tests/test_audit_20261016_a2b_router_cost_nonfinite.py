r"""Non-finite router costs are recorded before the ``kl_max`` clamp (audit 2026-10-16 F10).

``safe_kl_clamp`` maps NaN and :math:`+\infty` to ``kl_max``, so the captured cost ``score_energy_pre`` cannot tell
a failed cost from a finite one at or above the cap. The kernel readouts (``relational_predictive_readout`` and the
full-link context) now capture the boolean map ``score_nonfinite_pre`` of the raw cost, and the diagnostics report
its share of the causal support at the last layer (``relational_score_nonfinite_frac``) and its layer maximum
(``relational_score_nonfinite_frac_layer_max``). Tiny CPU models only (K = 8, two heads of 4, N = 5, vocab 9).
"""
import math
import warnings
from typing import Any, Callable, Dict, List

import pytest
import torch

import vfe3.gradients.full_link as full_link
import vfe3.gradients.relational as relational
from tests.profiles import tiny_config
from vfe3.model.model import VFEModel
from vfe3.posthoc.klcap import causal_support

TOKENS = torch.tensor([[0, 1, 2, 3, 4], [4, 5, 6, 7, 8]])
# (batch, head, query, source) causal entries of the injected layer: NaN, +inf and a finite cost above the cap.
NAN_ENTRY, INF_ENTRY, LARGE_ENTRY = (0, 0, 3, 1), (0, 1, 4, 2), (1, 0, 2, 0)


def _model(**overrides: Any) -> VFEModel:
    values: Dict[str, Any] = dict(vocab_size=9, embed_dim=8, n_heads=2, max_seq_len=5, n_layers=2, phi_scale=0.4)
    values.update(overrides)
    torch.manual_seed(3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VFEModel(tiny_config(**values)).eval()


def _injecting(original: Callable[..., torch.Tensor], kl_max: float, calls: List[int]) -> Callable[..., torch.Tensor]:
    r"""``relational_score`` that writes NaN, +inf and ``10 kl_max`` into the cost of every second call (layer 1)."""
    def score(*args: Any, **kwargs: Any) -> torch.Tensor:
        out = original(*args, **kwargs)                                     # (B, H, N, N)
        calls.append(1)
        if len(calls) % 2 == 0:
            out = out.clone()
            out[NAN_ENTRY], out[INF_ENTRY], out[LARGE_ENTRY] = float("nan"), float("inf"), 10.0 * kl_max
        return out
    return score


@pytest.mark.parametrize("reference", ["absolute", "row_min"])
def test_predictive_readout_records_nonfinite_costs_the_clamp_hides(
        reference: str, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model(routing_kl_max_reference=reference)
    kl_max = float(model.cfg.kl_max)
    calls: List[int] = []
    monkeypatch.setattr(relational, "relational_score", _injecting(relational.relational_score, kl_max, calls))
    snapshot = model.build_diagnostic_snapshot(TOKENS)
    assert len(calls) == 2
    clean, injected = model._layer_capture(snapshot, 0), model._layer_capture(snapshot, 1)
    for cap in (clean, injected):
        assert cap["score_nonfinite_pre"].dtype == torch.bool
        assert cap["score_nonfinite_pre"].shape == cap["score_energy_pre"].shape
        assert bool(torch.isfinite(cap["score_energy_pre"]).all())             # the clamp leaves no trace
    assert not bool(clean["score_nonfinite_pre"].any())
    expected = torch.zeros_like(injected["score_nonfinite_pre"])
    expected[NAN_ENTRY] = expected[INF_ENTRY] = True
    assert torch.equal(injected["score_nonfinite_pre"], expected)              # the finite large cost is not flagged
    score = injected["score_energy_pre"]
    assert float(score[NAN_ENTRY]) == float(score[INF_ENTRY]) == float(score[LARGE_ENTRY]) == kl_max

    d = model.diagnostics(TOKENS, snapshot=snapshot)
    per_layer = model.diagnostics_per_layer(TOKENS, snapshot=snapshot)
    support = causal_support(injected["beta_pre"], snapshot.log_prior)
    share = 2.0 / float(support.sum())
    assert d["relational_score_nonfinite_frac"] == pytest.approx(share, rel=1e-12)
    assert per_layer["relational_score_nonfinite_frac"] == pytest.approx([0.0, share], rel=1e-12)
    assert d["relational_score_nonfinite_frac_layer_max"] == pytest.approx(share, rel=1e-12)
    # the capped share still counts all three entries: only the new column separates the failed costs
    assert d["relational_score_klmax_frac"] >= 3.0 / float(support.sum()) - 1e-12


def test_clean_forward_reports_zero_on_every_layer() -> None:
    model = _model(n_layers=3)
    snapshot = model.build_diagnostic_snapshot(TOKENS)
    d = model.diagnostics(TOKENS, snapshot=snapshot)
    per_layer = model.diagnostics_per_layer(TOKENS, snapshot=snapshot)
    assert per_layer["relational_score_nonfinite_frac"] == [0.0, 0.0, 0.0]
    assert d["relational_score_nonfinite_frac"] == 0.0
    assert d["relational_score_nonfinite_frac_layer_max"] == 0.0


def test_full_link_context_records_nonfinite_costs(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model(attention_mode="projected_gradient", attention_residual=False,
                   predictive_covariance="passthrough", lambda_alpha_mode="state_dependent")
    assert model.relational_operators_by_layer[0].readout_mode == "full_link_update"
    kl_max = float(model.cfg.kl_max)
    calls: List[int] = []
    monkeypatch.setattr(full_link, "relational_score", _injecting(full_link.relational_score, kl_max, calls))
    snapshot = model.build_diagnostic_snapshot(TOKENS)
    assert len(calls) == 2
    injected = model._layer_capture(snapshot, 1)
    assert not bool(model._layer_capture(snapshot, 0)["score_nonfinite_pre"].any())
    assert int(injected["score_nonfinite_pre"].sum()) == 2
    assert bool(injected["score_nonfinite_pre"][NAN_ENTRY]) and bool(injected["score_nonfinite_pre"][INF_ENTRY])
    assert not bool(injected["score_nonfinite_pre"][LARGE_ENTRY])
    d = model.diagnostics(TOKENS, snapshot=snapshot)
    support = causal_support(injected["beta_pre"], snapshot.log_prior)
    assert d["relational_score_nonfinite_frac"] == pytest.approx(2.0 / float(support.sum()), rel=1e-12)


def test_minimal_agent_reads_its_unclamped_cost() -> None:
    model = _model(predictive_readout="minimal_agent", attention_residual=False)
    snapshot = model.build_diagnostic_snapshot(TOKENS)
    d = model.diagnostics(TOKENS, snapshot=snapshot)
    assert "score_nonfinite_pre" not in model._layer_capture(snapshot, 1)
    assert d["relational_score_nonfinite_frac"] == 0.0
    assert d["relational_score_nonfinite_frac_layer_max"] == 0.0


def test_capture_without_the_map_reports_nan() -> None:
    model = _model()
    snapshot = model.build_diagnostic_snapshot(TOKENS)
    model._layer_capture(snapshot, 1).pop("score_nonfinite_pre")
    d = model.diagnostics(TOKENS, snapshot=snapshot)
    assert math.isnan(d["relational_score_nonfinite_frac"])
    assert d["relational_score_nonfinite_frac_layer_max"] == 0.0               # the measured layer 0
