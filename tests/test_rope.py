from tests.profiles import test_config
import importlib

import pytest
import torch

from vfe3.belief import BeliefState
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


from vfe3.geometry.transport import RopeTransport, transport_mean, transport_covariance


def test_rope_mean_at_identity_omega_is_relative():
    # Omega = I -> Omega^RoPE = R(theta_i) R(theta_j)^T = R(theta_i - theta_j): relative position.
    N, K = 5, 4
    R = build_rope_rotation(torch.arange(N), [K], base=100.0,
                            device=torch.device("cpu"), dtype=torch.float32)
    omega_I = torch.eye(K).expand(N, N, K, K).contiguous()
    mu = torch.randn(N, K)
    rt = RopeTransport(base=omega_I, rope=R, on_cov=False, insertion="left")
    mu_t = transport_mean(rt, mu)                            # (N, N, K)
    mu_const = torch.ones(N, K)
    rt_c = RopeTransport(base=omega_I, rope=R, on_cov=False, insertion="left")
    t = transport_mean(rt_c, mu_const)                      # (N, N, K)
    # rows of equal (i-j) give equal transported vectors (relative-position property)
    assert torch.allclose(t[2, 1], t[3, 2], atol=1e-5)      # both are (i-j)=1
    assert torch.allclose(t[3, 1], t[4, 2], atol=1e-5)      # both are (i-j)=2


def test_rope_mean_only_leaves_covariance_unrotated():
    N, K = 4, 4
    R = build_rope_rotation(torch.arange(N), [K], base=10.0,
                            device=torch.device("cpu"), dtype=torch.float32)
    omega_I = torch.eye(K).expand(N, N, K, K).contiguous()
    sigma = torch.rand(N, K) + 0.5
    rt = RopeTransport(base=omega_I, rope=R, on_cov=False, insertion="left")
    plain = transport_covariance(omega_I, sigma)            # un-rotated diagonal sandwich
    roped = transport_covariance(rt, sigma)                 # mu-only -> ignores rope
    assert torch.allclose(plain, roped, atol=1e-6)


from vfe3.geometry.groups import get_group
from vfe3.inference.e_step import build_belief_transport


def test_build_belief_transport_wraps_in_ropetransport_when_rope_set():
    g = get_group("block_glk")(8, 2)
    phi = torch.randn(1, 6, g.generators.shape[0])
    R = build_rope_rotation(torch.arange(6), g.irrep_dims, base=100.0,
                            device=phi.device, dtype=phi.dtype)
    out = build_belief_transport(phi, g, transport_mode="flat", rope=R, rope_on_cov=False, rope_insertion="left")
    assert isinstance(out, RopeTransport)
    # rope=None reproduces the plain build (no wrapper).
    plain = build_belief_transport(phi, g, transport_mode="flat")
    assert not isinstance(plain, RopeTransport)


def _free_energy_rope_fixture() -> tuple[
    BeliefState,
    torch.Tensor,
    torch.Tensor,
    object,
    torch.Tensor,
]:
    torch.manual_seed(5)
    N, K, H = 5, 4, 2
    group = get_group("block_glk")(K, H)
    belief = BeliefState(
        mu=0.1 * torch.randn(N, K),
        sigma=torch.exp(torch.randn(N, K)),
        phi=0.05 * torch.randn(N, group.generators.shape[0]),
    )
    mu_p = torch.zeros(N, K)
    sigma_p = torch.ones(N, K)
    rope = build_rope_rotation(
        torch.arange(N),
        group.irrep_dims,
        base=10.0,
        device=belief.mu.device,
        dtype=belief.mu.dtype,
    )
    return belief, mu_p, sigma_p, group, rope


def test_global_free_energy_rope_matches_certified_compact_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    e_step_module = importlib.import_module("vfe3.inference.e_step")
    belief, mu_p, sigma_p, group, rope = _free_energy_rope_fixture()
    kwargs = {
        "tau": 1.3,
        "rope": rope,
        "rope_insertion": "left",   # legacy composition under test
        "rope_on_cov": True,
        "compact_phi_block_transport": True,
        "transport_mean_per_head": True,
    }
    actual = e_step_module.free_energy_value(belief, mu_p, sigma_p, group, **kwargs)
    original_transport = e_step_module.RopeTransport

    class _CertifiedRopeTransport(original_transport):
        def __init__(self, *args: object, **inner_kwargs: object) -> None:
            inner_kwargs["same_frame_flat_cocycle"] = True
            super().__init__(*args, **inner_kwargs)

    monkeypatch.setattr(e_step_module, "RopeTransport", _CertifiedRopeTransport)
    expected = e_step_module.free_energy_value(belief, mu_p, sigma_p, group, **kwargs)

    assert torch.equal(actual, expected)


def test_free_energy_rope_certificate_is_global_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    e_step_module = importlib.import_module("vfe3.inference.e_step")
    belief, mu_p, sigma_p, group, rope = _free_energy_rope_fixture()
    original_transport = e_step_module.RopeTransport
    certificates: list[object] = []

    class _RecordingRopeTransport(original_transport):
        def __init__(self, *args: object, **kwargs: object) -> None:
            certificates.append(kwargs.get("same_frame_flat_cocycle"))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(e_step_module, "RopeTransport", _RecordingRopeTransport)
    kwargs = {
        "tau": 1.3,
        "rope": rope,
        "rope_insertion": "left",   # legacy composition under test
        "rope_on_cov": True,
        "compact_phi_block_transport": True,
        "transport_mean_per_head": True,
    }
    e_step_module.free_energy_value(belief, mu_p, sigma_p, group, **kwargs)
    frozen_keys = belief._replace(phi=belief.phi + 0.01)
    e_step_module.free_energy_value(
        belief,
        mu_p,
        sigma_p,
        group,
        keys=frozen_keys,
        **kwargs,
    )

    assert certificates == [True, False]


from vfe3.model.model import VFEModel


def _rope_cfg(**kw):
    base = dict(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                n_e_steps=1, e_q_mu_lr=0.1, e_phi_lr=0.0, m_phi_lr=0.0, gauge_group="block_glk",
                warmup_steps=1, max_steps=4)
    base.update(kw)
    return test_config(**base)


def test_rope_cache_key_tracks_mutated_rope_base():
    model = VFEModel(_rope_cfg(
        embed_dim=8, n_heads=2, pos_rotation="rope",  rope_base=100.0,
    ))
    device = model.prior_bank.mu_embed.device
    first = model._rope_rotation(8, device)

    model.cfg.rope_base = 2.0
    second = model._rope_rotation(8, device)
    fresh = VFEModel(_rope_cfg(
        embed_dim=8, n_heads=2, pos_rotation="rope",  rope_base=2.0,
    ))._rope_rotation(8, device)

    assert not torch.equal(first, second)
    assert torch.equal(second, fresh)
    assert len(model._rope_cache) == 1


def test_rope_cache_key_tracks_mutated_pos_rotation():
    from vfe3.geometry import rope as rope_module

    name = "audit_negated_rope"

    @rope_module.register_pos_rotation(name)
    def _negated_rope(positions, irrep_dims, *, base=100.0, device=None, dtype=torch.float32):
        return -build_rope_rotation(
            positions,
            irrep_dims,
            base=base,
            device=device,
            dtype=dtype,
        )

    try:
        model = VFEModel(_rope_cfg(pos_rotation="rope", ))
        device = model.prior_bank.mu_embed.device
        first = model._rope_rotation(8, device)

        model.cfg.pos_rotation = name
        second = model._rope_rotation(8, device)

        assert torch.equal(second, -first)
        assert len(model._rope_cache) == 1
    finally:
        rope_module._POS_ROTATIONS.pop(name, None)


def test_runtime_length_caches_retain_only_the_latest_shape():
    model = VFEModel(_rope_cfg(pos_rotation="rope", ))
    device = model.prior_bank.mu_embed.device

    for length in range(1, model.cfg.max_seq_len + 1):
        model._attention_log_prior(length, device)
        model._rope_rotation(length, device)

    assert len(model._log_prior_cache) == 1
    assert len(model._rope_cache) == 1


def test_rope_changes_logits_vs_no_rope():
    torch.manual_seed(0)
    x = torch.randint(0, 6, (2, 8))
    base = VFEModel(_rope_cfg(pos_rotation="none"))
    roped = VFEModel(_rope_cfg(pos_rotation="rope", ))
    roped.load_state_dict(base.state_dict())
    assert not torch.allclose(base(x), roped(x), atol=1e-5)   # RoPE perturbs attention -> logits


def test_attention_maps_reflect_rope():
    torch.manual_seed(0)
    x = torch.randint(0, 6, (1, 8))
    base = VFEModel(_rope_cfg(pos_rotation="none"))
    roped = VFEModel(_rope_cfg(pos_rotation="rope", ))
    roped.load_state_dict(base.state_dict())
    a = base.attention_maps(x)
    b = roped.attention_maps(x)
    assert a.shape == b.shape
    assert not torch.allclose(a, b, atol=1e-5)             # RoPE changes the per-head attention


def test_diagnostics_runs_under_rope() -> None:
    """RoPE diagnostics capture the supported full-link receiver episode in float32."""
    torch.manual_seed(0)
    x = torch.randint(0, 6, (1, 8))
    roped = VFEModel(_rope_cfg(pos_rotation="rope"))
    snapshot = roped.build_diagnostic_snapshot(x)
    diag = roped.diagnostics(x, snapshot=snapshot)
    assert snapshot.rope is not None
    assert diag["relational_legacy_global_f_available"] == 0.0
    for key in ("relational_receiver_cost_before", "relational_receiver_cost_after",
                "relational_receiver_cost_delta", "effective_rank", "belief_cond_median"):
        assert torch.isfinite(torch.as_tensor(diag[key])).all(), key


from vfe3.gradients.kernels import belief_gradients
from vfe3.gradients.oracle import belief_gradients_autograd


def _full_cov_cfg(**kw):
    """_rope_cfg defaults plus the full-covariance pair (family/decode_mode); diagonal_covariance derived."""
    base = dict(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1,
                n_e_steps=1, e_q_mu_lr=0.1, e_phi_lr=0.0, m_phi_lr=0.0, gauge_group="block_glk",
                warmup_steps=1, max_steps=4,
                family="gaussian_full", decode_mode="full_chunked")
    base.update(kw)
    return test_config(**base)


def test_rope_means_only_kernel_matches_oracle():
    # The analytic belief-gradient kernel must still agree with autograd-of-F when the transport is
    # rope-rotated (means-only). Both consume the RopeTransport opaquely; agreement isolates RoPE.
    torch.manual_seed(0)
    g = get_group("block_glk")(8, 2)
    N, K, n_gen = 5, 8, g.generators.shape[0]
    phi = torch.randn(1, N, n_gen) * 0.1
    R = build_rope_rotation(torch.arange(N), g.irrep_dims, base=100.0,
                            device=phi.device, dtype=phi.dtype)
    omega = build_belief_transport(phi, g, transport_mode="flat", rope=R, rope_on_cov=False, rope_insertion="left")
    mu   = torch.randn(1, N, K); sigma   = torch.rand(1, N, K) + 0.5
    mu_p = torch.randn(1, N, K); sigma_p = torch.rand(1, N, K) + 0.5
    kw = dict(tau=1.0, renyi_order=1.0, kl_max=100.0, eps=1e-6, b0=1.0, c0=1.0, value=1.0,
              include_attention_entropy=True, gradient_mode="filtering", family="gaussian_diagonal",
              divergence_family="renyi", lambda_alpha_mode="constant", irrep_dims=g.irrep_dims, log_prior=None)
    gk = belief_gradients(mu, sigma, mu_p, sigma_p, omega, **kw)          # hand kernel
    go = belief_gradients_autograd(mu, sigma, mu_p, sigma_p, omega, **kw)  # autograd-of-F oracle
    assert torch.allclose(gk[0], go[0], atol=1e-5)
    assert torch.allclose(gk[1], go[1], atol=1e-5)


def test_rope_full_gauge_covariance_equals_manual_sandwich():
    # Full-gauge covariance: transport_covariance(RopeTransport(on_cov=True)) must equal the manual
    # sandwich with the rotated operator Omega'_ij = R_i Omega_ij R_j^T. Pure property; no model.
    torch.manual_seed(0)
    N, K = 4, 4
    R = build_rope_rotation(torch.arange(N), [K], base=10.0,
                            device=torch.device("cpu"), dtype=torch.float64)
    omega = torch.randn(N, N, K, K, dtype=torch.float64)
    A = torch.randn(N, K, K, dtype=torch.float64)
    sigma = A @ A.transpose(-1, -2) + K * torch.eye(K, dtype=torch.float64)   # SPD full cov
    got = transport_covariance(RopeTransport(base=omega, rope=R, on_cov=True, insertion="left"), sigma)
    Op = torch.einsum("ikl,ijlm,jnm->ijkn", R, omega, R)                      # R_i Omega_ij R_j^T
    manual = torch.einsum("ijkl,jlm,ijnm->ijkn", Op, sigma, Op)
    assert torch.allclose(got, manual, atol=1e-9)


def test_full_gauge_model_runs_forward_backward():
    # Reachability: a full-covariance rope_full_gauge model trains (finite gradients) end to end.
    cfg = _full_cov_cfg(pos_rotation="rope",  rope_full_gauge=True)
    torch.manual_seed(0)
    m = VFEModel(cfg)
    x = torch.randint(0, 6, (1, 6)); y = torch.randint(0, 6, (1, 6))
    _, loss, _ = m(x, y); loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(m.prior_bank.mu_embed.grad).all()


def test_2x2_positional_ablation_runs():
    torch.manual_seed(0)
    x = torch.randint(0, 6, (2, 8))
    for pr in ("none", "rope"):
        for pp in ("none", "learned"):
            m = VFEModel(_rope_cfg(pos_rotation=pr, pos_phi=pp))
            out = m(x)
            assert out.shape[0] == 2 and torch.isfinite(out).all()


# --- value-gauge decoupling toggle (rope_on_value) -------------------------

def test_rope_on_value_can_be_enabled_explicitly():
    assert test_config(rope_on_value=True).rope_on_value is True


def test_rope_on_value_false_decouples_and_changes_logits():
    # With RoPE on, factoring the value gauge from the attention gauge (rope_on_value=False) gives a
    # different forward than the coherent single-gauge default -- the value aggregation drops RoPE.
    torch.manual_seed(0)
    x = torch.randint(0, 6, (2, 8))
    coupled = VFEModel(_rope_cfg(pos_rotation="rope",  rope_on_value=True))
    decoupled = VFEModel(_rope_cfg(pos_rotation="rope",  rope_on_value=False))
    decoupled.load_state_dict(coupled.state_dict())
    assert not torch.allclose(coupled(x), decoupled(x), atol=1e-5)


def test_rope_on_value_inert_without_rope():
    # rope_on_value only factors the gauge-RoPE rotation; with pos_rotation='none' there is no
    # rotation, so the flag is a no-op and the forward is byte-identical.
    torch.manual_seed(0)
    x = torch.randint(0, 6, (2, 8))
    a = VFEModel(_rope_cfg(pos_rotation="none", rope_on_value=True))
    b = VFEModel(_rope_cfg(pos_rotation="none", rope_on_value=False))
    b.load_state_dict(a.state_dict())
    assert torch.allclose(a(x), b(x), atol=1e-6)


def test_rope_on_value_false_model_trains_end_to_end():
    # Decoupled value gauge routes the belief gradient to the autograd oracle; the model must still
    # train (finite loss and prior-table gradients) end to end.
    torch.manual_seed(0)
    m = VFEModel(_rope_cfg(pos_rotation="rope",  rope_on_value=False))
    x = torch.randint(0, 6, (1, 8)); y = torch.randint(0, 6, (1, 8))
    _, loss, _ = m(x, y); loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(m.prior_bank.mu_embed.grad).all()
