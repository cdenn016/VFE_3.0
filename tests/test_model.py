from tests.profiles import test_config
import torch
import pytest
from types import SimpleNamespace


from vfe3.model.model import VFEModel


def test_model_forward_shapes_and_loss():
    cfg = test_config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=2,
                     n_e_steps=1, e_q_mu_lr=0.05)
    model = VFEModel(cfg)
    tokens = torch.randint(0, 20, (3, 5))
    logits = model(tokens)
    assert logits.shape == (3, 5, 20)
    targets = torch.randint(0, 20, (3, 5))
    _, loss, _ = model(tokens, targets)
    assert loss.shape == () and torch.isfinite(loss)


def test_loss_backward_reaches_prior_tables():
    # THE crown jewel: the unrolled E-step keeps the training graph connected, so the
    # M-step gradient reaches the encode/phi prior parameters (not just decode).
    cfg = test_config(vocab_size=15, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=2,
                     n_e_steps=2, e_q_mu_lr=0.05)
    model = VFEModel(cfg)
    tokens = torch.randint(0, 15, (2, 4)); targets = torch.randint(0, 15, (2, 4))
    _, loss, _ = model(tokens, targets)
    loss.backward()
    assert model.prior_bank.mu_embed.grad is not None
    assert model.prior_bank.phi_embed.grad is not None
    assert model.prior_bank.mu_embed.grad.abs().sum() > 0          # gradient actually flows
    assert model.prior_bank.phi_embed.grad.abs().sum() > 0


def test_prior_handoff_sigma_must_be_convex():
    import pytest
    # rho_s outside [0,1] is a non-convex sigma blend that can drive the prior variance
    # negative off the SPD cone; the config must reject it (matching prior_handoff_rho).
    for bad in (2.0, -1.0):
        with pytest.raises(ValueError):
            test_config(prior_handoff_sigma=bad)
    test_config(prior_handoff_sigma=0.5)  # in-range is accepted


def test_all_ignore_batch_yields_finite_zero_loss():
    # A microbatch where every target == -100 must not poison the loss with NaN.
    cfg = test_config(vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1,
                     n_e_steps=1, e_q_mu_lr=0.05)
    model = VFEModel(cfg)
    tokens = torch.randint(0, 12, (2, 4)); targets = torch.full((2, 4), -100)
    _, loss, _ = model(tokens, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(model.prior_bank.mu_embed.grad).all()


def test_model_has_no_nn_layers():
    import torch.nn as nn
    nn_layers = (nn.Linear, nn.MultiheadAttention, nn.RNNBase, nn.Conv1d)
    # The pure path: with the opt-in block MLP off, no conventional nn layer exists anywhere.
    cfg = test_config(vocab_size=10, embed_dim=4, n_heads=2, max_seq_len=3, use_block_mlp=False)
    model = VFEModel(cfg)
    for m in model.modules():
        assert not isinstance(m, nn_layers)
    # With the block MLP on (the current default), its fc1/fc2 are the ONLY nn layers.
    mlp_model = VFEModel(test_config(vocab_size=10, embed_dim=4, n_heads=2, max_seq_len=3,
                                    use_block_mlp=True))
    layer_names = [name for name, m in mlp_model.named_modules() if isinstance(m, nn_layers)]
    assert layer_names and all(name.startswith("block_mlps.") for name in layer_names)


def test_build_group_dispatches_on_arity_for_every_group() -> None:
    from vfe3.model.model import build_group
    # This is a standalone group-builder contract; model inference admits block_glk only.
    for name, group_n, irrep_spec in (("block_glk", None, None), ("so_n", 4, [("l1", 1)])):
        cfg = SimpleNamespace(gauge_group=name, embed_dim=4, n_heads=2,
                              group_n=group_n, irrep_spec=irrep_spec)
        group = build_group(cfg)
        assert group.generators.shape[-1] == 4


def test_group_generators_follow_dtype_move():
    # GaugeGroup is not an nn.Module; its generators must still follow model.to(dtype)
    # so the E-step transport (belief.phi, which moves) is matmul'd against same-dtype
    # generators. _apply re-maps them.
    cfg = test_config(vocab_size=10, embed_dim=4, n_heads=2, max_seq_len=3)
    model = VFEModel(cfg)
    model.to(torch.float64)
    assert model.prior_bank.mu_embed.dtype == torch.float64
    assert model.group.generators.dtype == torch.float64


def test_norm_type_final_is_wired():
    # norm_type_final must actually be applied before decode (not a dead seam): a
    # configured final norm changes the logits relative to 'none'.
    base = dict(vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1,
                n_e_steps=1, e_q_mu_lr=0.05)
    tok = torch.randint(0, 12, (2, 4))
    torch.manual_seed(0); m_norm = VFEModel(test_config(**base, norm_type_final="mahalanobis"))
    torch.manual_seed(0); m_none = VFEModel(test_config(**base, norm_type_final="none"))
    assert not torch.allclose(m_norm(tok), m_none(tok))


def test_norm_type_layernorm_is_wired_both_seams():
    # "layernorm" flows end-to-end through BOTH norm seams (block + final) with no model.py change:
    # the registry key builds via get_norm(cfg.norm_type_*) and the standardized means change the
    # logits vs 'none'. Parameter-free: it adds no learnable parameters over the 'none' model, so
    # there is no optimizer wiring to do (the whole point of the parameter-free variant).
    base = dict(vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=2,
                n_e_steps=1, e_q_mu_lr=0.05, layernorm_affine=False)     # parameter-free variant
    tok = torch.randint(0, 12, (2, 4))
    torch.manual_seed(0); m_ln   = VFEModel(test_config(**base, norm_type_block="layernorm",
                                                       norm_type_final="layernorm"))
    torch.manual_seed(0); m_none = VFEModel(test_config(**base, norm_type_block="none",
                                                       norm_type_final="none"))
    assert not torch.allclose(m_ln(tok), m_none(tok))
    assert sum(p.numel() for p in m_ln.parameters()) == sum(p.numel() for p in m_none.parameters())


# --- Audit 2026-05-31 --------------------------------------------------------
def test_manual_seed_makes_model_init_reproducible():
    """run_training pins reproducibility via torch.manual_seed(cfg.seed) before building the model;
    verify the mechanism it relies on -- seeding then constructing yields identical prior tables,
    and a different seed yields different ones. (Seeding lives at the entry point, not __init__, so
    it does not clobber a caller-set RNG state.)"""
    cfg = test_config(vocab_size=40, embed_dim=4, n_heads=2, max_seq_len=4)
    torch.manual_seed(123); m1 = VFEModel(cfg)
    torch.manual_seed(123); m2 = VFEModel(cfg)
    assert torch.equal(m1.prior_bank.mu_embed, m2.prior_bank.mu_embed)
    assert torch.equal(m1.prior_bank.phi_embed, m2.prior_bank.phi_embed)
    torch.manual_seed(999); m3 = VFEModel(cfg)
    assert not torch.equal(m1.prior_bank.mu_embed, m3.prior_bank.mu_embed)


def test_killing_per_block_precond_runs_through_forward():
    # phi_precond_mode='killing_per_block' is a validated config seam; with e_phi_lr>0 it
    # must not crash the forward. The per-block Killing metric needs the group's irrep_dims,
    # which the E-step must thread into precondition_phi_gradient.
    cfg = test_config(vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1,
                     n_e_steps=2, e_q_mu_lr=0.05,
                     phi_precond_mode="killing_per_block")
    model = VFEModel(cfg)
    tokens = torch.randint(0, 12, (2, 4)); targets = torch.randint(0, 12, (2, 4))
    _, loss, _ = model(tokens, targets)
    assert torch.isfinite(loss)


def test_model_rejects_tied_block_glk() -> None:
    with pytest.raises(ValueError, match="gauge_group"):
        test_config(gauge_group="tied_block_glk")


def test_diagnostics_reports_projected_geometry_semantics() -> None:
    import math
    cfg = test_config(vocab_size=7, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1,
                      n_e_steps=2, e_q_mu_lr=0.05)
    d = VFEModel(cfg).diagnostics(torch.randint(0, 7, (2, 4)))
    assert d["relational_semantics"] == "relational_full_link_frozen_filtering"
    assert d["relational_legacy_global_f_available"] == 0.0
    assert math.isnan(d["holonomy_deviation"]) and math.isnan(d["gauge_trace_spread"])
    assert math.isfinite(d["relational_receiver_cost_before"])
    assert math.isfinite(d["relational_receiver_cost_after"])


def test_attention_maps_shape_and_row_stochastic():
    # Per-layer, per-head attention: (L, H, N, N) with H = len(irrep_dims) (block_glk K=4 -> 2 heads),
    # and every query row of every head is row-stochastic (softmax over keys -> sums to 1).
    cfg = test_config(vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=6, n_layers=3,
                     n_e_steps=1, e_q_mu_lr=0.05, prior_handoff_rho=0.5)
    model = VFEModel(cfg)
    tokens = torch.randint(0, 12, (2, 6))
    maps = model.attention_maps(tokens)
    assert maps.shape == (3, len(model.group.irrep_dims), 6, 6)     # (L, H, N, N), H=2
    assert torch.allclose(maps.sum(dim=-1), torch.ones_like(maps.sum(dim=-1)), atol=1e-5)


def test_attention_maps_last_layer_matches_diagnostics():
    # The canonical resolution of "what is per-layer attention": the last layer's map is the SAME
    # attention diagnostics() reads at the converged belief. diagnostics() scores the pre-transform
    # converged belief (audit 2026-09-23 #20) while attention_maps scores the block output, so the
    # post-E-step transforms are switched off here; then the mean row entropy of maps[-1] equals
    # diagnostics' attn_entropy, which guards the block-by-block replay against trajectory drift.
    from vfe3 import metrics
    cfg = test_config(vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1,
                     n_e_steps=2, e_q_mu_lr=0.05,
                     norm_type_block="none", use_block_mlp=False)
    model = VFEModel(cfg)
    # attention_maps reports sequence 0; projected diagnostics average the batch (audit 2026-10-01
    # F5), so the contract is checked on a single-sequence batch with seeded tokens.
    tokens = torch.randint(0, 12, (1, 5), generator=torch.Generator().manual_seed(0))
    maps = model.attention_maps(tokens)
    ent = float(metrics.attention_entropy(maps[-1]))
    assert abs(ent - model.diagnostics(tokens)["attn_entropy"]) < 1e-6
    # A multi-sequence batch still reports sequence 0's maps.
    batch = torch.cat((tokens, torch.randint(0, 12, (1, 5), generator=torch.Generator().manual_seed(1))))
    torch.testing.assert_close(model.attention_maps(batch), maps)


def test_diagnostics_per_layer_shape_and_finite():
    # Each layer reports available projected diagnostics and explicit unavailable global terms.
    import math
    cfg = test_config(vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=6, n_layers=3,
                     n_e_steps=1, e_q_mu_lr=0.05, prior_handoff_rho=0.5)
    model = VFEModel(cfg)
    tokens = torch.randint(0, 12, (2, 6))
    pl = model.diagnostics_per_layer(tokens)
    for k in ("effective_rank", "belief_cond_median", "phi_norm_mean", "attn_entropy"):
        assert k in pl and len(pl[k]) == 3
        assert all(math.isfinite(v) for v in pl[k])
    for k in ("self_coupling", "belief_coupling", "attention_entropy", "total",
              "holonomy_deviation", "gauge_trace_spread"):
        assert k in pl and len(pl[k]) == 3
        assert all(math.isnan(v) for v in pl[k])


def test_diagnostics_per_layer_attn_entropy_matches_attention_maps():
    # The per-layer replay must reproduce the SAME attention as attention_maps at EVERY layer (both
    # replay vfe_stack's block loop identically); guards the per-layer diagnostics against drift.
    from vfe3 import metrics
    cfg = test_config(vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=3,
                     n_e_steps=2, e_q_mu_lr=0.05, prior_handoff_rho=0.5)
    model = VFEModel(cfg)
    tokens = torch.randint(0, 12, (1, 5), generator=torch.Generator().manual_seed(0))   # see F5 note above
    pl = model.diagnostics_per_layer(tokens)
    maps = model.attention_maps(tokens)
    for li in range(3):
        assert abs(pl["attn_entropy"][li] - float(metrics.attention_entropy(maps[li]))) < 1e-5


def test_model_rejects_single_block_attention() -> None:
    with pytest.raises(ValueError, match="gauge_group"):
        test_config(gauge_group="so_k", n_heads=1)


def test_model_rejects_sp_gauge_group() -> None:
    with pytest.raises(ValueError, match="gauge_group"):
        test_config(gauge_group="sp")


