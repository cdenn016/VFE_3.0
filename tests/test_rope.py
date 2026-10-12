from tests.profiles import test_config

import torch

from vfe3.geometry.rope import build_rope_rotation, get_pos_rotation


def test_rope_rotation_is_orthogonal_and_block_diagonal():
    irrep_dims = [4, 4]                                     # two head-blocks of size 4
    R = build_rope_rotation(torch.arange(6), irrep_dims, base=100.0,
                            device=torch.device("cpu"), dtype=torch.float32)
    assert R.shape == (6, 8, 8)
    eye = torch.eye(8).expand(6, 8, 8)
    assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-5)   # orthogonal
    # off-block entries are exactly zero (block-diagonal on irrep_dims)
    assert torch.count_nonzero(R[:, 0:4, 4:8]) == 0
    assert torch.count_nonzero(R[:, 4:8, 0:4]) == 0


def test_rope_position_zero_is_identity():
    R = build_rope_rotation(torch.arange(3), [4], base=100.0,
                            device=torch.device("cpu"), dtype=torch.float32)
    assert torch.allclose(R[0], torch.eye(4), atol=1e-6)   # position 0 -> angle 0 -> I


def test_pos_rotation_none_registered():
    assert get_pos_rotation("none")(torch.arange(3), [4], base=100.0,
                                    device=torch.device("cpu"), dtype=torch.float32) is None


from vfe3.geometry.transport import RopeTransport


from vfe3.geometry.groups import get_group
from vfe3.inference.e_step import build_belief_transport


def test_build_belief_transport_wraps_in_ropetransport_when_rope_set():
    g = get_group("block_glk")(8, 2)
    phi = torch.randn(1, 6, g.generators.shape[0])
    R = build_rope_rotation(torch.arange(6), g.irrep_dims, base=100.0,
                            device=phi.device, dtype=phi.dtype)
    out = build_belief_transport(phi, g, rope=R, rope_insertion="left")
    assert isinstance(out, RopeTransport)
    # rope=None reproduces the plain build (no wrapper).
    plain = build_belief_transport(phi, g)
    assert not isinstance(plain, RopeTransport)


from vfe3.model.model import VFEModel


def _rope_cfg(**kw):
    base = dict(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                n_e_steps=1, e_q_mu_lr=0.1, m_phi_lr=0.0, gauge_group="block_glk",
                warmup_steps=1, max_steps=4)
    base.update(kw)
    return test_config(**base)


def test_rope_cache_key_tracks_mutated_rope_base():
    model = VFEModel(_rope_cfg(
        embed_dim=8, n_heads=2, rope_mode="frame_value",  rope_base=100.0,
    ))
    device = model.prior_bank.mu_embed.device
    first = model._rope_rotation(8, device)

    model.cfg.rope_base = 2.0
    second = model._rope_rotation(8, device)
    fresh = VFEModel(_rope_cfg(
        embed_dim=8, n_heads=2, rope_mode="frame_value",  rope_base=2.0,
    ))._rope_rotation(8, device)

    assert not torch.equal(first, second)
    assert torch.equal(second, fresh)
    assert len(model._rope_cache) == 1


def test_runtime_length_caches_retain_only_the_latest_shape():
    model = VFEModel(_rope_cfg(rope_mode="frame_value", ))
    device = model.prior_bank.mu_embed.device

    for length in range(1, model.cfg.max_seq_len + 1):
        model._attention_log_prior(length, device)
        model._rope_rotation(length, device)

    assert len(model._log_prior_cache) == 1
    assert len(model._rope_cache) == 1


def test_rope_changes_logits_vs_no_rope():
    torch.manual_seed(0)
    x = torch.randint(0, 6, (2, 8))
    base = VFEModel(_rope_cfg(rope_mode="none"))
    roped = VFEModel(_rope_cfg(rope_mode="frame_value", ))
    roped.load_state_dict(base.state_dict())
    assert not torch.allclose(base(x), roped(x), atol=1e-5)   # RoPE perturbs attention -> logits


def test_attention_maps_reflect_rope():
    torch.manual_seed(0)
    x = torch.randint(0, 6, (1, 8))
    base = VFEModel(_rope_cfg(rope_mode="none"))
    roped = VFEModel(_rope_cfg(rope_mode="frame_value", ))
    roped.load_state_dict(base.state_dict())
    a = base.attention_maps(x)
    b = roped.attention_maps(x)
    assert a.shape == b.shape
    assert not torch.allclose(a, b, atol=1e-5)             # RoPE changes the per-head attention


def test_diagnostics_runs_under_rope() -> None:
    """RoPE diagnostics capture the supported full-link receiver episode in float32."""
    torch.manual_seed(0)
    x = torch.randint(0, 6, (1, 8))
    roped = VFEModel(_rope_cfg(rope_mode="frame_value"))
    snapshot = roped.build_diagnostic_snapshot(x)
    diag = roped.diagnostics(x, snapshot=snapshot)
    assert snapshot.rope is not None
    assert diag["relational_legacy_global_f_available"] == 0.0
    for key in ("relational_receiver_cost_before", "relational_receiver_cost_after",
                "relational_receiver_cost_delta", "effective_rank", "belief_cond_median"):
        assert torch.isfinite(torch.as_tensor(diag[key])).all(), key


def _full_cov_cfg(**kw):
    """_rope_cfg defaults plus the full-covariance pair (family/decode_mode); diagonal_covariance derived."""
    base = dict(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                n_e_steps=1, e_q_mu_lr=0.1, m_phi_lr=0.0, gauge_group="block_glk",
                warmup_steps=1, max_steps=4,
                family="gaussian_full", decode_mode="full_chunked")
    base.update(kw)
    return test_config(**base)


def test_full_gauge_model_runs_forward_backward():
    # Reachability: a full-covariance rope_full_gauge model trains (finite gradients) end to end.
    cfg = _full_cov_cfg(rope_mode="frame_value")
    torch.manual_seed(0)
    m = VFEModel(cfg)
    x = torch.randint(0, 6, (1, 6)); y = torch.randint(0, 6, (1, 6))
    _, loss, _ = m(x, y); loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(m.prior_bank.mu_embed.grad).all()


def test_2x2_positional_ablation_runs():
    torch.manual_seed(0)
    x = torch.randint(0, 6, (2, 8))
    for pr in ("none", "frame_value"):
        for pp in ("none", "learned"):
            m = VFEModel(_rope_cfg(rope_mode=pr, pos_phi=pp))
            out = m(x)
            assert out.shape[0] == 2 and torch.isfinite(out).all()


# --- value-gauge decoupling (rope_mode "frame" versus "frame_value") ------------------

def test_frame_value_rope_can_be_enabled_explicitly():
    assert test_config(rope_mode="frame_value").rope_mode == "frame_value"


def test_rope_on_value_false_decouples_and_changes_logits():
    # With RoPE on, factoring the value gauge from the attention gauge (rope_mode="frame") gives a
    # different forward than the coherent single-gauge default -- the value aggregation drops RoPE.
    torch.manual_seed(0)
    x = torch.randint(0, 6, (2, 8))
    coupled = VFEModel(_rope_cfg(rope_mode="frame_value"))
    decoupled = VFEModel(_rope_cfg(rope_mode="frame"))
    decoupled.load_state_dict(coupled.state_dict())
    assert not torch.allclose(coupled(x), decoupled(x), atol=1e-5)


def test_rope_on_value_false_model_trains_end_to_end():
    # Decoupled value gauge routes the belief gradient to the autograd oracle; the model must still
    # train (finite loss and prior-table gradients) end to end.
    torch.manual_seed(0)
    m = VFEModel(_rope_cfg(rope_mode="frame"))
    x = torch.randint(0, 6, (1, 8)); y = torch.randint(0, 6, (1, 8))
    _, loss, _ = m(x, y); loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(m.prior_bank.mu_embed.grad).all()
