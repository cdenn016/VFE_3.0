"""Layer maxima of the last-layer guard columns in VFEModel.diagnostics() (audit 2026-10-09).

The headline ``relational_score_klmax_frac``, ``relational_projected_cov_max_condition`` and
``belief_cond_median`` read the last layer only; the ``_layer_max`` keys (and the klmax argmax layer) must
equal the maxima of the per-layer values, while the last-layer keys stay unchanged. Tiny CPU model.
"""
import torch

from tests.profiles import test_config
from vfe3.model.model import VFEModel


def test_diagnostics_report_layer_maxima_of_the_guard_columns() -> None:
    torch.manual_seed(0)
    cfg = test_config(vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=6, n_layers=3,
                      n_e_steps=1, e_q_mu_lr=0.05, prior_handoff_rho=0.5, kl_max=0.5)
    model = VFEModel(cfg)
    tokens = torch.randint(0, 12, (2, 6), generator=torch.Generator().manual_seed(1))
    snapshot = model.build_diagnostic_snapshot(tokens, return_logits=False)
    d = model.diagnostics(tokens, snapshot=snapshot)
    per_layer = model.diagnostics_per_layer(tokens, snapshot=snapshot)
    for key in ("relational_score_klmax_frac", "relational_projected_cov_max_condition", "belief_cond_median"):
        assert d[key] == per_layer[key][-1]                               # last-layer key unchanged
        assert d[f"{key}_layer_max"] == max(per_layer[key])
    fractions = per_layer["relational_score_klmax_frac"]
    assert d["relational_score_klmax_frac_argmax_layer"] == float(fractions.index(max(fractions)))
