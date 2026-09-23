import torch

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel
from vfe3.viz.extract import (
    across_layer_belief_trace,
    belief_bank,
    e_step_belief_trace,
    numerical_health,
    per_unit_eval_nats,
)


def _model(n_layers=2, **over):
    cfg = VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=n_layers,
                     n_e_steps=2, e_q_mu_lr=0.1, e_phi_lr=0.0, **over)
    torch.manual_seed(0)
    return VFEModel(cfg)


def test_per_unit_eval_nats_shapes_and_token_count():
    model = _model()
    tokens = torch.randint(0, 20, (3, 5))
    targets = tokens.clone()
    targets[0, 0] = -100; targets[1, 4] = -100; targets[2, 2] = -100   # 3 ignored
    out = per_unit_eval_nats(model, [(tokens, targets)])
    assert out["per_seq_nats"].shape == (3,)
    assert out["per_seq_tokens"].shape == (3,)
    assert float(out["per_seq_tokens"].sum()) == 12.0                  # 15 - 3 ignored
    assert out["per_token_nats"].shape == (12,)
    assert torch.isfinite(out["per_token_nats"]).all()


def test_belief_bank_collects_all_components():
    model = _model()
    tokens = torch.randint(0, 20, (3, 5))
    bank = belief_bank(model, [tokens])
    assert bank["mu"].shape == (15, 4)
    assert bank["phi"].shape[0] == 15
    assert bank["token_ids"].shape == (15,)
    assert bank["seq_idx"].shape == (15,)
    assert int(bank["seq_idx"].max()) == 2                             # three sequences
    assert torch.equal(bank["token_ids"], tokens.reshape(-1))


def test_numerical_health_under_rope_does_not_raise():
    # m5: numerical_health used RopeTransport without importing it -> NameError under pos_rotation='rope',
    # which report.py's _safe silently swallowed (blank health panel). Assert it returns the dict instead.
    model = _model(n_layers=1, pos_rotation="rope")
    tok = torch.randint(0, 20, (1, 5))
    health = numerical_health(model, tok)
    for k in ("nan_mu", "nan_sigma", "nan_phi", "nan_energy", "nan_beta", "max_condition"):
        assert k in health


def test_e_step_belief_trace_captures_trajectory():
    model = _model()
    tokens = torch.randint(0, 20, (1, 5))
    tr = e_step_belief_trace(model, tokens, n_iter=4)
    assert tr["mu"].shape == (5, 5, 4)                                 # T+1, N, K
    assert tr["phi"].shape[0] == 5
    assert tr["free_energy"].shape == (5,)
    assert torch.isfinite(tr["free_energy"]).all()


def test_across_layer_belief_trace_depth_and_base_zero():
    model = _model(n_layers=3)
    tokens = torch.randint(0, 20, (1, 5))
    al = across_layer_belief_trace(model, tokens)
    assert al["mu"].shape == (3, 5, 4)                                 # L, N, K
    assert al["d_ai"].shape == (3,)
    assert float(al["d_ai"][0].abs()) < 1e-6                           # layer 0 distance to itself
    assert al["effective_rank"].shape == (3,)


def test_numerical_health_all_finite():
    model = _model()
    tokens = torch.randint(0, 20, (1, 5))
    h = numerical_health(model, tokens)
    assert h["nan_mu"] == 0.0 and h["nan_sigma"] == 0.0 and h["nan_phi"] == 0.0
    assert h["nan_energy"] == 0.0 and h["nan_beta"] == 0.0
    import math
    assert math.isfinite(h["max_condition"])
