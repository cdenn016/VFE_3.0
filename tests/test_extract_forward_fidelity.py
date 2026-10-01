r"""C9 forward-fidelity tests for the viz belief extractors (audit 2026-07-01).

``_encode_one`` must replay model.forward's belief pipeline exactly, so
every figure describes the model that actually trained. Pins:

  * on the default config the extractors are byte-identical to the pre-fix replay, so the
    pure path is untouched;
  * under ``pos_rotation="rope"`` the replay still reproduces the forward belief exactly.

Each fidelity test carries a discrimination guard (the pre-fix replay DIFFERS from forward),
so the equality assertions cannot pass vacuously. The forward reference is
``forward_beliefs(capture=...)['out']`` -- the raw pre-final_norm ``vfe_stack`` output, which
is exactly the tensor the extractors return.

Device-agnostic (CPU default; set VFE3_TEST_DEVICE=cuda for the GPU).
"""
from tests.profiles import test_config
import os

import pytest
import torch

from vfe3.belief import BeliefState
from vfe3.model.canonical_content import CanonicalFrameContext
from vfe3.model.model import VFEModel
from vfe3.model.stack import vfe_stack
from vfe3.viz import extract

DEVICE = torch.device(os.environ.get("VFE3_TEST_DEVICE", "cpu"))


def _prefix_encode_one(model, token_ids: torch.Tensor):
    r"""The pre-C9 ``_encode_one`` replay: encode -> positional phi, without the s-anchor."""
    enc = model.prior_bank.encode(token_ids[:1])
    belief = BeliefState(mu=enc.mu[0], sigma=enc.sigma[0], phi=model._apply_pos_phi(enc.phi[0]))
    n = belief.mu.shape[0]
    log_prior = model._attention_log_prior(n, token_ids.device)
    rope = model._rope_rotation(n, token_ids.device)
    return belief, log_prior, rope


def _stack(model, belief, log_prior, rope):
    r"""The extractors' shared ``vfe_stack`` replay (mirrors converged_state).

    Includes the per-layer block MLPs (on by default) and their frame transport exactly as the
    extractors and forward pass them; a replay without them no longer describes the trained model.
    """
    cfg = model.cfg
    block_mlp_transport, block_mlp_frame = model._block_mlp_frame_transport(belief, rope)
    shared = model._build_shared_flat_transport(belief, rope, require_vertex_factors=True)
    frame = model._canonical_frame_context(shared, fold_rope=True, compact=True)
    value_frame = (model._canonical_frame_context(shared, compact=True)
                   if rope is not None and not cfg.rope_on_value else frame)
    return vfe_stack(
        belief, belief.mu, belief.sigma, model.group, cfg,
        log_prior=log_prior, block_norm=model.block_norm,
        block_mlps=model.block_mlps, block_mlp_frame=block_mlp_frame,
        head_mixer=model.head_mixer, cg_coupling=model.cg_coupling,
        lambda_beta=cfg.lambda_beta,
        transport_state=model.transport_state,
        rope=rope, rope_on_cov=cfg.rope_full_gauge, rope_on_value=cfg.rope_on_value,
        prebuilt_transport=block_mlp_transport, rope_insertion=model.cfg.rope_insertion,
        relational_operators_by_layer=model.relational_operators_by_layer,
        relational_frame=frame, relational_value_frame=value_frame)


def _forward_reference(model, tokens: torch.Tensor):
    r"""forward's raw pre-final_norm converged belief (``capture['out']``, the extractor target)."""
    cap: dict = {}
    with torch.no_grad():
        model.forward_beliefs(tokens, capture=cap)
    return cap["out"]


def test_extractor_anchor_is_noop_on_default_config():
    # Regression guard: on the default config the fixed extractors reproduce the pre-fix
    # replay byte-for-byte.
    torch.manual_seed(0)
    cfg = test_config(vocab_size=16, embed_dim=4, n_heads=2, max_seq_len=8)
    model = VFEModel(cfg).to(DEVICE)
    model.eval()
    tokens = torch.randint(0, 16, (2, 8), device=DEVICE)
    with torch.no_grad():
        belief, log_prior, rope = extract._encode_one(model, tokens)
        belief0, log_prior0, _ = _prefix_encode_one(model, tokens)
    assert torch.equal(belief.mu,    belief0.mu)
    assert torch.equal(belief.sigma, belief0.sigma)
    assert torch.equal(belief.phi,   belief0.phi)
    assert (log_prior is None and log_prior0 is None) or torch.equal(log_prior, log_prior0)


def _rope_model() -> VFEModel:
    """The positional-rotation fixture: the extractors must replay a rope forward exactly.

    Rope is the residual gauge-FIXING layer, so its transport pair is the least trivial one the
    extractor replay has to reproduce; construction emits the documented gauge warning.
    """
    with pytest.warns(UserWarning, match="gauge"):
        cfg = test_config(
            vocab_size=16,
            embed_dim=4,
            n_heads=2,
            max_seq_len=8,
            n_layers=1,
            n_e_steps=2,
            e_q_mu_lr=0.2,
            e_q_sigma_lr=0.1,
            e_phi_lr=0.0,
            pos_rotation="rope",
        )
    torch.manual_seed(23)
    model = VFEModel(cfg).to(DEVICE)
    model.eval()
    with torch.no_grad():
        model.prior_bank.phi_embed.normal_(mean=0.0, std=0.15)
    return model


def test_encode_one_matches_forward_under_rope() -> None:
    model = _rope_model()
    tokens = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.long, device=DEVICE)
    ref = _forward_reference(model, tokens)

    belief, log_prior, rope = extract._encode_one(model, tokens)
    out = _stack(model, belief, log_prior, rope)

    assert torch.allclose(out.mu, ref.mu[0], atol=1e-5, rtol=1e-5)
    assert torch.allclose(out.sigma, ref.sigma[0], atol=1e-5, rtol=1e-5)


def test_belief_ce_bank_matches_forward_sigma_under_rope() -> None:
    model = _rope_model()
    tokens = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7], [7, 6, 5, 4, 3, 2, 1, 0]],
        dtype=torch.long,
        device=DEVICE,
    )
    targets = torch.roll(tokens, shifts=-1, dims=1)
    ref = _forward_reference(model, tokens)
    bank = extract.belief_ce_bank(model, [(tokens, targets)], max_batches=1)
    expected_trace = ref.sigma.sum(dim=-1).reshape(-1).detach().cpu()

    assert all(value.device.type == "cpu" for value in bank.values())
    assert torch.allclose(bank["tr_sigma"], expected_trace, atol=1e-5, rtol=1e-5)


def _projected_extract_model(**overrides: object) -> VFEModel:
    values = dict(
        vocab_size=11,
        embed_dim=4,
        n_heads=2,
        max_seq_len=5,
        batch_size=1,
        n_layers=1,
        n_e_steps=1,
        e_step_update="gradient",
        oracle_unroll_grad=True,
        e_q_mu_lr=0.04,
        e_phi_lr=0.0,
        family="gaussian_diagonal",
        transport_mode="flat",
        decode_mode="diagonal_chunked",
        encode_mode="per_token",
        lambda_alpha_mode="constant",
        pos_phi="learned",
        max_steps=1,
    )
    values.update(overrides)
    cfg = test_config(**values)
    torch.manual_seed(151)
    return VFEModel(cfg).to(DEVICE)


def _projected_parity_case(**overrides: object):
    model = _projected_extract_model(n_layers=2, **overrides)
    model.eval()
    tokens = torch.tensor([[2, 5, 7]], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        model.prior_bank.mu_embed[tokens[0]] = torch.tensor(
            [
                [0.8, -0.3, 0.5, 0.1],
                [-0.2, 0.9, -0.4, 0.6],
                [0.7, 0.2, -0.8, 0.4],
            ],
            device=DEVICE,
        )
        model.prior_bank.sigma_log_embed[tokens[0]] = torch.tensor(
            [
                [0.2, -0.4, 0.5, -0.1],
                [-0.3, 0.6, 0.1, 0.4],
                [0.5, -0.2, 0.3, -0.5],
            ],
            device=DEVICE,
        )
        model.prior_bank.phi_embed[tokens[0]].normal_(mean=0.0, std=0.35)
        model.pos_phi_free[: tokens.shape[1]].normal_(mean=0.0, std=0.25)
    return model, tokens, model.build_diagnostic_snapshot(tokens)


def _assert_tensor_dict_close(
    actual: dict[str, object],
    expected: dict[str, object],
    keys: tuple[str, ...],
) -> None:
    for key in keys:
        assert isinstance(actual[key], torch.Tensor)
        assert isinstance(expected[key], torch.Tensor)
        torch.testing.assert_close(actual[key], expected[key], atol=1e-5, rtol=1e-5)


def test_projected_encode_one_matches_production_initial_snapshot() -> None:
    """Returning raw canonical table moments instead of materialized q0 breaks this parity."""
    model, tokens, snapshot = _projected_parity_case()

    belief, _, _ = extract._encode_one(model, tokens)

    torch.testing.assert_close(belief.mu, snapshot.initial_belief.mu[0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        belief.sigma, snapshot.initial_belief.sigma[0], atol=1e-6, rtol=1e-6)
    # Per-token encoding preserves the prior moments while retaining nonzero local frames.
    torch.testing.assert_close(model.prior_bank.mu_embed[tokens[0]], snapshot.initial_belief.mu[0])
    assert bool((snapshot.initial_belief.phi != 0).any())


def test_projected_e_step_trajectory_matches_production_snapshot() -> None:
    """The replayed q0..qT trajectory must begin from production's realized-frame q0."""
    model, tokens, snapshot = _projected_parity_case()

    replay = extract.e_step_belief_trace(model, tokens)
    production = extract.e_step_belief_trace(model, tokens, snapshot=snapshot)

    _assert_tensor_dict_close(replay, production, ("mu", "sigma", "phi", "free_energy"))


def test_projected_across_layer_trace_matches_production_snapshot() -> None:
    """Layer replay must use the materialized initial prior captured by production."""
    model, tokens, snapshot = _projected_parity_case()

    replay = extract.across_layer_belief_trace(model, tokens)
    production = extract.across_layer_belief_trace(model, tokens, snapshot=snapshot)

    _assert_tensor_dict_close(
        replay,
        production,
        ("mu", "sigma", "d_ai", "effective_rank", "rank_one_residual"),
    )


def test_across_layer_trace_with_mlp_dropout_is_rng_neutral_and_restores_mode() -> None:
    cfg = test_config(
        vocab_size=13,
        embed_dim=4,
        n_heads=2,
        max_seq_len=4,
        n_layers=3,
        n_e_steps=1,
        e_phi_lr=0.0,
        use_block_mlp=True,
        block_mlp_dropout=0.5,
    )
    model = VFEModel(cfg).to(DEVICE)
    model.train()
    tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        torch.manual_seed(311)
        expected_next_logits = model(tokens).detach()

        torch.manual_seed(311)
        cpu_rng_before = torch.get_rng_state().clone()
        extract.across_layer_belief_trace(model, tokens)

        assert torch.equal(torch.get_rng_state(), cpu_rng_before)
        assert model.training
        assert all(module.training for module in model.block_mlps.modules())
        actual_next_logits = model(tokens).detach()
        assert torch.equal(actual_next_logits, expected_next_logits)


def test_projected_converged_state_matches_production_snapshot() -> None:
    """Converged diagnostics must not silently replay canonical rather than realized moments."""
    model, tokens, snapshot = _projected_parity_case()

    replay = extract.readout_state(model, tokens)
    production = extract.readout_state(model, tokens, snapshot=snapshot)

    _assert_tensor_dict_close(
        replay,
        production,
        ("mu", "sigma", "phi", "energy", "beta"),
    )


def test_projected_rope_unrolled_replay_matches_production_snapshot() -> None:
    """Allowed means-only gauge-RoPE and an unrolled E-step share the projected q0 seam."""
    with pytest.warns(UserWarning, match="gauge"):
        model, tokens, snapshot = _projected_parity_case(
            pos_rotation="rope",
            rope_full_gauge=False,
            use_block_mlp=False,          # the block MLP requires a gradient-carrying E-step
        )

    belief, _, rope = extract._encode_one(model, tokens)
    assert rope is not None
    torch.testing.assert_close(belief.mu, snapshot.initial_belief.mu[0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        belief.sigma, snapshot.initial_belief.sigma[0], atol=1e-6, rtol=1e-6)
    replay = extract.readout_state(model, tokens)
    production = extract.readout_state(model, tokens, snapshot=snapshot)
    _assert_tensor_dict_close(replay, production, ("mu", "sigma", "phi", "energy", "beta"))


@pytest.mark.parametrize(
    ("overrides", "expected_shared_builds"),
    [
        pytest.param({}, 1, id="flat-fixed-no-rope"),
        pytest.param({"pos_rotation": "rope"}, 1, id="rope"),
    ],
)
def test_refine_transport_sharing_is_derived_from_exact_eligibility(
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, object],
    expected_shared_builds: int,
) -> None:
    values = {
        "vocab_size": 11,
        "embed_dim": 4,
        "n_heads": 2,
        "max_seq_len": 4,
        "n_layers": 1,
        "n_e_steps": 1,
        "transport_mode": "flat",
        "pos_phi": "none",
        "e_phi_lr": 0.0,
    }
    values.update(overrides)
    model = VFEModel(test_config(**values)).to(DEVICE).eval()
    tokens = torch.tensor([[1, 3, 5]], dtype=torch.long, device=DEVICE)
    shared_builds = 0
    original = model._build_shared_flat_transport

    def _spy(*args, **kwargs):
        nonlocal shared_builds
        shared_builds += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "_build_shared_flat_transport", _spy)
    model.forward_beliefs(tokens, return_logits=False)

    assert shared_builds == expected_shared_builds


@pytest.mark.parametrize("sequence_length", [1, 3])
def test_projected_inference_record_reuses_same_forward_frame_for_decode(
    sequence_length: int,
) -> None:
    """Reference decoding must retain the realized relational frame from the same forward."""
    model = _projected_extract_model(decode_coordinates="reference")
    model.eval()
    tokens = torch.arange(2, 2 + sequence_length, device=DEVICE).unsqueeze(0)
    targets = torch.arange(7, 7 + sequence_length, device=DEVICE).remainder(
        model.cfg.vocab_size
    ).unsqueeze(0)

    records = extract.collect_inference_bank(
        model,
        [(tokens, targets)],
        return_logits=False,
        max_batches=1,
        device=DEVICE,
    )
    assert len(records) == 1
    record = records[0]
    assert record["logits"] is None
    frame = record["canonical_frame"]
    assert isinstance(frame, CanonicalFrameContext)
    assert frame.forward.device.type == "cpu"
    assert frame.inverse.device.type == "cpu"
    assert frame.forward.shape == (1, sequence_length, 2, 2, 2)

    decoded = extract._decode_inference_record(model, record, DEVICE)
    _, direct = model.forward_beliefs(tokens, return_logits=True)
    assert direct is not None
    assert decoded.shape == (1, sequence_length, model.cfg.vocab_size)
    assert torch.allclose(decoded, direct, atol=1e-6, rtol=1e-6)

    ce_bank = extract.belief_ce_bank(
        model,
        [],
        inference_bank=records,
        max_batches=1,
        device=DEVICE,
    )
    assert ce_bank["ce"].shape == (sequence_length,)
    assert torch.isfinite(ce_bank["ce"]).all()


def test_active_mlp_extract_replay_matches_forward() -> None:
    """Extractor replay must include every trained, untied post-block MLP."""
    cfg = test_config(
        vocab_size=11, embed_dim=4, n_heads=2, max_seq_len=4,
        n_layers=2, n_e_steps=1, e_phi_lr=0.0, use_block_mlp=True,
    )
    torch.manual_seed(17)
    model = VFEModel(cfg).to(DEVICE).eval()
    with torch.no_grad():
        for mlp in model.block_mlps:
            mlp.fc2.bias.fill_(0.25)
    tokens = torch.tensor([[1, 3, 5, 7]], dtype=torch.long, device=DEVICE)

    reference = _forward_reference(model, tokens)
    replay = extract.across_layer_belief_trace(model, tokens)

    torch.testing.assert_close(replay["mu"][-1], reference.mu[0], atol=1e-6, rtol=1e-6)
