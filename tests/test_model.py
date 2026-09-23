import torch
from vfe3.belief import BeliefState
from vfe3.config import VFE3Config
from vfe3.geometry.groups import get_group
from vfe3.model.block import vfe_block
from vfe3.model.stack import vfe_stack


def _belief(N=4, K=4, n_gen=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return BeliefState(
        mu=torch.randn(N, K, generator=g),
        sigma=torch.rand(N, K, generator=g) + 0.5,
        phi=0.1 * torch.randn(N, n_gen, generator=g),
    )


def test_block_runs_e_step_and_preserves_shapes():
    cfg = VFE3Config(embed_dim=4, n_heads=2, n_e_steps=2, e_q_mu_lr=0.05, e_phi_lr=0.0)
    grp = get_group("block_glk")(4, 2)
    b = _belief(K=4, n_gen=grp.generators.shape[0])
    out = vfe_block(b, b.mu, b.sigma, grp, cfg)
    assert out.mu.shape == b.mu.shape and (out.sigma > 0).all()


def test_stack_handoff_updates_prior_across_blocks():
    cfg = VFE3Config(embed_dim=4, n_heads=2, n_layers=3, n_e_steps=1,
                     e_q_mu_lr=0.05, e_phi_lr=0.0, prior_handoff_rho=1.0)
    grp = get_group("block_glk")(4, 2)
    b = _belief(K=4, n_gen=grp.generators.shape[0])
    out = vfe_stack(b, b.mu, b.sigma, grp, cfg)
    assert out.mu.shape == b.mu.shape and (out.sigma > 0).all()
    # with rho=1 and a nonzero E-step, the stack moves the belief off the input
    assert not torch.allclose(out.mu, b.mu, atol=1e-4)


from vfe3.model.model import VFEModel


def test_model_forward_shapes_and_loss():
    cfg = VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=2,
                     n_e_steps=1, e_q_mu_lr=0.05, e_phi_lr=0.0)
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
    cfg = VFE3Config(vocab_size=15, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=2,
                     n_e_steps=2, e_q_mu_lr=0.05, e_phi_lr=0.02, gradient_mode="filtering")
    model = VFEModel(cfg)
    tokens = torch.randint(0, 15, (2, 4)); targets = torch.randint(0, 15, (2, 4))
    _, loss, _ = model(tokens, targets)
    loss.backward()
    assert model.prior_bank.mu_embed.grad is not None
    assert model.prior_bank.phi_embed.grad is not None
    assert model.prior_bank.mu_embed.grad.abs().sum() > 0          # gradient actually flows
    assert model.prior_bank.phi_embed.grad.abs().sum() > 0


def test_phi_e_step_actually_moves_phi():
    # Isolate the phi E-step: the crown jewel's phi_embed.grad>0 check passes even with
    # e_phi_lr=0 (phi flows via the in-graph transport omega in the kernel), so it does
    # not by itself exercise the phi update. Here we assert the OUTPUT belief.phi changes
    # between e_phi_lr=0 (phi step skipped) and e_phi_lr>0 (phi natgrad applied), which
    # exercises the phi E-step itself, not only the transport coupling.
    grp = get_group("block_glk")(4, 2)
    n_gen = grp.generators.shape[0]

    def run(e_phi_lr):
        b = _belief(K=4, n_gen=n_gen)
        cfg = VFE3Config(embed_dim=4, n_heads=2, n_layers=1, n_e_steps=2,
                         e_q_mu_lr=0.05, e_phi_lr=e_phi_lr)
        return vfe_stack(b, b.mu, b.sigma, grp, cfg).phi

    phi_off = run(0.0)
    phi_on = run(0.05)
    assert not torch.allclose(phi_off, phi_on, atol=1e-6)  # the phi step moves phi


def test_detached_e_step_with_phi_runs_and_reaches_decode_priors():
    # The documented fixed-point/truncated mode must complete a
    # forward+backward even with the phi E-step enabled (e_phi_lr>0): the phi natgrad runs
    # under its own enable_grad island, so the blanket no_grad does not strangle it.
    # Under detach the E-step is severed, so priors are reached only via decode (mu/sigma);
    # phi_embed is frozen (no gradient path) -- pin both as the as-built semantics.
    # The prior-bank decode reaches mu_embed because its KL scores
    # against it; the linear-decode default would leave mu_embed grad None under a severed E-step.
    cfg = VFE3Config(vocab_size=15, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=2,
                     n_e_steps=2, e_q_mu_lr=0.05, e_phi_lr=0.02,
                     e_step_gradient="detach", decode_mode="diagonal_chunked")
    model = VFEModel(cfg)
    tokens = torch.randint(0, 15, (2, 4)); targets = torch.randint(0, 15, (2, 4))
    _, loss, _ = model(tokens, targets)
    assert torch.isfinite(loss)                                    # no crash, finite loss
    loss.backward()
    assert model.prior_bank.mu_embed.grad is not None              # decode reaches mu prior
    assert model.prior_bank.mu_embed.grad.abs().sum() > 0
    assert model.prior_bank.phi_embed.grad is None                 # phi frozen under detach


def test_prior_handoff_sigma_must_be_convex():
    import pytest
    # rho_s outside [0,1] is a non-convex sigma blend that can drive the prior variance
    # negative off the SPD cone; the config must reject it (matching prior_handoff_rho).
    for bad in (2.0, -1.0):
        with pytest.raises(ValueError):
            VFE3Config(prior_handoff_sigma=bad)
    VFE3Config(prior_handoff_sigma=0.5)  # in-range is accepted


def test_all_ignore_batch_yields_finite_zero_loss():
    # A microbatch where every target == -100 must not poison the loss with NaN.
    cfg = VFE3Config(vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1,
                     n_e_steps=1, e_q_mu_lr=0.05)
    model = VFEModel(cfg)
    tokens = torch.randint(0, 12, (2, 4)); targets = torch.full((2, 4), -100)
    _, loss, _ = model(tokens, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(model.prior_bank.mu_embed.grad).all()


def test_model_has_no_nn_layers():
    import torch.nn as nn
    cfg = VFE3Config(vocab_size=10, embed_dim=4, n_heads=2, max_seq_len=3)
    model = VFEModel(cfg)
    for m in model.modules():
        assert not isinstance(m, (nn.Linear, nn.MultiheadAttention, nn.RNNBase, nn.Conv1d))


def test_build_group_dispatches_on_arity_for_every_group():
    from vfe3.model.model import build_group
    # build_group must construct every registered group by positional arity alone,
    # so a new group slots in via register_group without editing the dispatcher.
    for g, n_gen_expected in (("glk", None), ("block_glk", None), ("so_k", None)):
        cfg = VFE3Config(vocab_size=10, embed_dim=4, n_heads=2, max_seq_len=3, gauge_group=g)
        grp = build_group(cfg)
        assert grp.generators.shape[-1] == 4  # K


def test_group_generators_follow_dtype_move():
    # GaugeGroup is not an nn.Module; its generators must still follow model.to(dtype)
    # so the E-step transport (belief.phi, which moves) is matmul'd against same-dtype
    # generators. _apply re-maps them.
    cfg = VFE3Config(vocab_size=10, embed_dim=4, n_heads=2, max_seq_len=3)
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
    torch.manual_seed(0); m_norm = VFEModel(VFE3Config(**base, norm_type_final="mahalanobis"))
    torch.manual_seed(0); m_none = VFEModel(VFE3Config(**base, norm_type_final="none"))
    assert not torch.allclose(m_norm(tok), m_none(tok))


def test_norm_type_layernorm_is_wired_both_seams():
    # "layernorm" flows end-to-end through BOTH norm seams (block + final) with no model.py change:
    # the registry key builds via get_norm(cfg.norm_type_*) and the standardized means change the
    # logits vs 'none'. Parameter-free: it adds no learnable parameters over the 'none' model, so
    # there is no optimizer wiring to do (the whole point of the parameter-free variant).
    base = dict(vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=2,
                n_e_steps=1, e_q_mu_lr=0.05)
    tok = torch.randint(0, 12, (2, 4))
    torch.manual_seed(0); m_ln   = VFEModel(VFE3Config(**base, norm_type_block="layernorm",
                                                       norm_type_final="layernorm"))
    torch.manual_seed(0); m_none = VFEModel(VFE3Config(**base, norm_type_block="none",
                                                       norm_type_final="none"))
    assert not torch.allclose(m_ln(tok), m_none(tok))
    assert sum(p.numel() for p in m_ln.parameters()) == sum(p.numel() for p in m_none.parameters())


# --- Audit 2026-05-31 --------------------------------------------------------
def test_manual_seed_makes_model_init_reproducible():
    """run_training pins reproducibility via torch.manual_seed(cfg.seed) before building the model;
    verify the mechanism it relies on -- seeding then constructing yields identical prior tables,
    and a different seed yields different ones. (Seeding lives at the entry point, not __init__, so
    it does not clobber a caller-set RNG state.)"""
    cfg = VFE3Config(vocab_size=40, embed_dim=4, n_heads=2, max_seq_len=4)
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
    cfg = VFE3Config(vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1,
                     n_e_steps=2, e_q_mu_lr=0.05, e_phi_lr=0.05,
                     phi_precond_mode="killing_per_block")
    model = VFEModel(cfg)
    tokens = torch.randint(0, 12, (2, 4)); targets = torch.randint(0, 12, (2, 4))
    _, loss, _ = model(tokens, targets)
    assert torch.isfinite(loss)


def test_b0_c0_threaded_into_state_dependent_alpha():
    """cfg.b0/c0 reach the state-dependent self-coupling alpha; changing b0 changes the belief."""
    grp = get_group("block_glk")(4, 2)
    n_gen = grp.generators.shape[0]

    def run(b0):
        b = _belief(K=4, n_gen=n_gen)
        cfg = VFE3Config(embed_dim=4, n_heads=2, n_layers=1, n_e_steps=2, e_q_mu_lr=0.1,
                         e_phi_lr=0.0, lambda_alpha_mode="state_dependent", b0=b0)
        return vfe_block(b, b.mu, b.sigma, grp, cfg).mu

    assert not torch.allclose(run(1.0), run(8.0), atol=1e-5)


def test_mass_phi_penalizes_phi_inside_the_e_step():
    """mass_phi enters the phi E-step objective (manuscript Algorithm 1: +alpha_phi/2 ||phi||^2
    inside the phi gradient), pulling the inferred phi toward 0 during inference -- not only the
    outer M-step training loss. A larger mass_phi must shrink the E-step's output phi."""
    grp = get_group("block_glk")(4, 2)
    n_gen = grp.generators.shape[0]

    def run(mass):
        b = _belief(K=4, n_gen=n_gen)
        cfg = VFE3Config(embed_dim=4, n_heads=2, n_layers=1, n_e_steps=3, e_q_mu_lr=0.05,
                         e_phi_lr=0.2, mass_phi=mass)
        return vfe_stack(b, b.mu, b.sigma, grp, cfg).phi

    assert run(5.0).norm() < run(0.0).norm()


def test_model_runs_under_tied_block_glk():
    # The tied_block_glk group (shared gauge across heads) must run end-to-end through the same
    # model/E-step machinery as block_glk (it only changes the generator layout, not irrep_dims).
    cfg = VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1,
                     n_e_steps=2, e_q_mu_lr=0.05, e_phi_lr=0.05, m_phi_lr=0.01,
                     gauge_group="tied_block_glk", phi_precond_mode="none")
    model = VFEModel(cfg)
    tok = torch.randint(0, 20, (3, 5)); tgt = torch.randint(0, 20, (3, 5))
    _, loss, _ = model(tok, tgt)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.prior_bank.phi_embed.grad is not None and model.prior_bank.mu_embed.grad.abs().sum() > 0


def test_diagnostics_includes_gauge_geometry_probes():
    # Part 1 (diagnostics tier): diagnostics must surface the gauge-geometry probes that
    # vfe3/metrics.py already provides -- holonomy_deviation (curvature proxy on the converged
    # transport) and gauge_trace_spread (std of log|det Omega| across tokens) -- as finite floats.
    import math
    cfg = VFE3Config(vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=1,
                     n_e_steps=2, e_q_mu_lr=0.05, e_phi_lr=0.05)
    model = VFEModel(cfg)
    tokens = torch.randint(0, 12, (2, 4))
    d = model.diagnostics(tokens)
    assert "holonomy_deviation" in d and "gauge_trace_spread" in d
    assert math.isfinite(d["holonomy_deviation"]) and math.isfinite(d["gauge_trace_spread"])


def test_attention_maps_shape_and_row_stochastic():
    # Per-layer, per-head attention: (L, H, N, N) with H = len(irrep_dims) (block_glk K=4 -> 2 heads),
    # and every query row of every head is row-stochastic (softmax over keys -> sums to 1).
    cfg = VFE3Config(vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=6, n_layers=3,
                     n_e_steps=1, e_q_mu_lr=0.05, e_phi_lr=0.0, prior_handoff_rho=0.5)
    model = VFEModel(cfg)
    tokens = torch.randint(0, 12, (2, 6))
    maps = model.attention_maps(tokens)
    assert maps.shape == (3, len(model.group.irrep_dims), 6, 6)     # (L, H, N, N), H=2
    assert torch.allclose(maps.sum(dim=-1), torch.ones_like(maps.sum(dim=-1)), atol=1e-5)


def test_attention_maps_last_layer_matches_diagnostics():
    # The canonical resolution of "what is per-layer attention": the last layer's map is the SAME
    # attention diagnostics() reads at the converged belief. At n_layers=1 (stack is one block, the
    # handoff loop is empty) this is exact -- the mean row entropy of maps[-1] equals diagnostics'
    # attn_entropy byte for byte, which guards the block-by-block replay against trajectory drift.
    from vfe3 import metrics
    cfg = VFE3Config(vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1,
                     n_e_steps=2, e_q_mu_lr=0.05, e_phi_lr=0.05)
    model = VFEModel(cfg)
    tokens = torch.randint(0, 12, (2, 5))
    maps = model.attention_maps(tokens)
    ent = float(metrics.attention_entropy(maps[-1]))
    assert abs(ent - model.diagnostics(tokens)["attn_entropy"]) < 1e-6


def test_diagnostics_per_layer_shape_and_finite():
    # Per-layer (inference-depth) belief-channel diagnostics: one finite value per layer for every key,
    # exposing the depth axis the scalar diagnostics() collapses to the final block.
    import math
    cfg = VFE3Config(vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=6, n_layers=3,
                     n_e_steps=1, e_q_mu_lr=0.05, e_phi_lr=0.0, prior_handoff_rho=0.5)
    model = VFEModel(cfg)
    tokens = torch.randint(0, 12, (2, 6))
    pl = model.diagnostics_per_layer(tokens)
    for k in ("self_coupling", "belief_coupling", "attention_entropy", "total", "holonomy_deviation",
              "gauge_trace_spread", "effective_rank", "belief_cond_median", "phi_norm_mean"):
        assert k in pl and len(pl[k]) == 3                          # one value per layer
        assert all(math.isfinite(v) for v in pl[k])


def test_diagnostics_per_layer_attn_entropy_matches_attention_maps():
    # The per-layer replay must reproduce the SAME attention as attention_maps at EVERY layer (both
    # replay vfe_stack's block loop identically); guards the per-layer diagnostics against drift.
    from vfe3 import metrics
    cfg = VFE3Config(vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=3,
                     n_e_steps=2, e_q_mu_lr=0.05, e_phi_lr=0.0, prior_handoff_rho=0.5)
    model = VFEModel(cfg)
    tokens = torch.randint(0, 12, (2, 5))
    pl = model.diagnostics_per_layer(tokens)
    maps = model.attention_maps(tokens)
    for li in range(3):
        assert abs(pl["attn_entropy"][li] - float(metrics.attention_entropy(maps[li]))) < 1e-5


def test_attention_maps_single_block_group_has_unit_head_axis():
    # A single-irrep-block group (so_k) yields H=1: the energy is full-K (N,N) and the method
    # inserts a unit head axis, so the result is still (L, 1, N, N) (uniform with the multi-head case).
    cfg = VFE3Config(vocab_size=12, embed_dim=4, n_heads=1, max_seq_len=5, n_layers=1,
                     gauge_group="so_k", n_e_steps=1, e_q_mu_lr=0.05, e_phi_lr=0.0)
    model = VFEModel(cfg)
    maps = model.attention_maps(torch.randint(0, 12, (2, 5)))
    assert maps.shape == (1, 1, 5, 5)


def test_model_runs_under_sp_gauge_group():
    # The symplectic Sp(2m,R) gauge (single-block, non-skew) must run end-to-end through the
    # same model/E-step machinery: a forward + loss.backward() with targets yields a finite loss
    # and finite, grad-connected gradients on the prior tables. embed_dim=4 (even, K=2m with m=2);
    # n_heads=2 divides it for the multihead attention/decode (the gauge group itself is the
    # single super-block sp(4)). Mirrors test_model_runs_under_cross_coupled_block_glk.
    cfg = VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1,
                     n_e_steps=2, e_q_mu_lr=0.05, e_phi_lr=0.05, m_phi_lr=0.01,
                     gauge_group="sp", phi_precond_mode="none")
    model = VFEModel(cfg)
    assert model.group.generators.shape[0] == 2 * (2 * 2 + 1)   # m(2m+1) = 10 for m=2
    assert model.group.irrep_dims == [4]
    tok = torch.randint(0, 20, (3, 5)); tgt = torch.randint(0, 20, (3, 5))
    _, loss, _ = model(tok, tgt)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(model.prior_bank.mu_embed.grad).all()
    assert torch.isfinite(model.prior_bank.phi_embed.grad).all()


def test_model_runs_under_cross_coupled_block_glk():
    # The cross-coupled block_glk gauge (config cross_couplings=[(0,1)]) must run end-to-end
    # through the same model/E-step machinery: a forward + loss.backward() with targets yields a
    # finite loss and finite, grad-connected gradients on the prior tables. This mirrors the
    # verify-first check; it is the oracle that the wired cross-coupled path works end-to-end.
    cfg = VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1,
                     n_e_steps=2, e_q_mu_lr=0.05, e_phi_lr=0.05, m_phi_lr=0.01,
                     gauge_group="block_glk", cross_couplings=[(0, 1)],
                     phi_precond_mode="none")
    model = VFEModel(cfg)
    # the cross-coupled group is the extended (larger-than-direct-sum) basis
    d_head = cfg.embed_dim // cfg.n_heads
    assert model.group.generators.shape[0] == 3 * d_head * d_head
    tok = torch.randint(0, 20, (3, 5)); tgt = torch.randint(0, 20, (3, 5))
    _, loss, _ = model(tok, tgt)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(model.prior_bank.mu_embed.grad).all()
    assert torch.isfinite(model.prior_bank.phi_embed.grad).all()
    assert model.prior_bank.mu_embed.grad.abs().sum() > 0
    assert model.prior_bank.phi_embed.grad.abs().sum() > 0


def test_phi_retraction_bch_is_reached_by_the_stack(monkeypatch) -> None:
    import importlib
    module = importlib.import_module("vfe3.inference.e_step")
    original = module.e_step_iteration
    seen: list[str] = []

    def record(*args: object, **kwargs: object) -> object:
        seen.append(kwargs["phi_retract_mode"])
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "e_step_iteration", record)
    grp = get_group("block_glk")(4, 2)
    b = _belief(K=4, n_gen=grp.generators.shape[0])
    cfg = VFE3Config(embed_dim=4, n_heads=2, n_layers=1, n_e_steps=3,
                    e_q_mu_lr=0.05, e_phi_lr=0.2)
    out = vfe_stack(b, b.mu, b.sigma, grp, cfg)
    assert seen == ["bch"] * 3
    assert torch.isfinite(out.phi).all()
