import pytest
import torch

from vfe3.geometry.groups import get_group
from vfe3.geometry.transport import (
    _TRANSPORTS,
    compute_transport_operators,
    get_transport,
    get_transport_registration,
    register_transport,
)


def test_so2_transport_is_exact_rotation():
    # exp(theta * L_01) with L_01 = [[0,1],[-1,0]] is the rotation
    # [[cos, sin], [-sin, cos]]. Independent closed-form check.
    import math
    grp = get_group("so_n")(2, group_n=2, irrep_spec=[("l1", 1)])
    theta = 0.7
    phi = torch.zeros(1, 1, 1)
    phi[0, 0, 0] = theta
    out = compute_transport_operators(phi, grp, gauge_mode="learned")
    c, s = math.cos(theta), math.sin(theta)
    expected = torch.tensor([[c, s], [-s, c]])
    assert torch.allclose(out["exp_phi"][0, 0], expected, atol=1e-5)


def test_phi_path_cocycle_identity():
    # Flat (Regime I) transport is a cocycle: Omega_ij @ Omega_jk = Omega_ik.
    grp = get_group("so_n")(4, group_n=4, irrep_spec=[("l1", 1)])
    g = torch.Generator().manual_seed(31)
    phi = 0.3 * torch.randn(1, 4, grp.generators.shape[0], generator=g)
    omega = compute_transport_operators(phi, grp, gauge_mode="learned")["Omega"]
    lhs = omega[0, 0, 1] @ omega[0, 1, 2]
    rhs = omega[0, 0, 2]
    assert torch.allclose(lhs, rhs, atol=1e-4)


# --- register_transport / get_transport seam (roadmap; connection-regime axis) ---
def test_register_transport_requires_covariance_class():
    with pytest.raises(TypeError, match="covariance_class"):
        register_transport("_test_missing_covariance_class")


def test_transport_registry_round_trip():
    """register_transport/get_transport round-trip and the unknown-name KeyError."""
    sentinel = object()

    @register_transport("_test_dummy_transport", covariance_class="test-only")
    def _dummy(*args, **kwargs):
        return sentinel

    try:
        assert get_transport("_test_dummy_transport")() is sentinel
        registration = get_transport_registration("_test_dummy_transport")
        assert registration.callable is _dummy
        assert registration.needs_mu is False
        assert registration.needs_sigma is False
        assert registration.batch_independent is False
        assert registration.covariance_class == "test-only"
        with pytest.raises(KeyError):
            get_transport("nope")
        with pytest.raises(KeyError):
            get_transport_registration("nope")
    finally:
        _TRANSPORTS.pop("_test_dummy_transport", None)


def test_every_transport_registers_covariance_class():
    expected = {
        "flat": "covariant (flat)",
    }
    assert {name: _TRANSPORTS[name].covariance_class for name in expected} == expected
    assert all(
        isinstance(registration.covariance_class, str) and registration.covariance_class
        for registration in _TRANSPORTS.values()
    )


def test_flat_is_registered():
    """The default flat (Regime I) phi-cocycle builder is registered under 'flat'."""
    assert callable(get_transport("flat"))


def test_flat_builder_bit_identical_to_direct_call():
    """The 'flat' builder's TransportDict is bit-identical (torch.equal) to a direct
    ``compute_transport_operators(phi, group)`` call on a fixed-seed phi."""
    grp = get_group("so_n")(4, group_n=4, irrep_spec=[("l1", 1)])
    g = torch.Generator().manual_seed(77)
    phi = 0.3 * torch.randn(2, 3, grp.generators.shape[0], generator=g)
    direct = compute_transport_operators(phi, grp)
    seam = get_transport("flat")(phi, grp, gauge_mode="learned")
    assert set(seam) == set(direct)
    for key in ("Omega", "exp_phi", "exp_neg_phi"):
        assert torch.equal(seam[key], direct[key])


def test_flat_builder_tolerates_extra_kwargs():
    """The 'flat' adapter swallows unknown kwargs (so a future stateful non-flat builder
    can share the call shape) without changing its output."""
    grp = get_group("so_n")(4, group_n=4, irrep_spec=[("l1", 1)])
    g = torch.Generator().manual_seed(78)
    phi = 0.3 * torch.randn(1, 2, grp.generators.shape[0], generator=g)
    direct = compute_transport_operators(phi, grp)
    seam = get_transport("flat")(phi, grp, gauge_mode="learned", connection_state=object())
    assert torch.equal(seam["Omega"], direct["Omega"])


# --- factored transport (P0 #2): fuse the per-token exps into the mean/cov contractions,
#     skipping the dense (B,N,N,K,K) Omega on the flat + block-diagonal path ---
def _block_inputs(seed=0):
    from vfe3.geometry.groups import get_group as _gg
    torch.manual_seed(seed)
    grp = _gg("block_glk")(8, 2)                      # irrep_dims [4, 4]
    n_gen = grp.generators.shape[0]
    phi = 0.2 * torch.randn(2, 5, n_gen)
    mu = torch.randn(2, 5, 8)
    sig = torch.rand(2, 5, 8) + 0.5
    return grp, phi, mu, sig


def test_build_factored_transport_does_not_form_dense_omega():
    """The factored builder exposes only the per-token (B,N,K,K) factors, never the dense Omega."""
    from vfe3.geometry.transport import FactoredTransport, build_factored_transport
    grp, phi, mu, sig = _block_inputs(3)
    factored = build_factored_transport(phi, grp)
    assert isinstance(factored, FactoredTransport)
    assert factored.exp_phi.shape == (2, 5, 8, 8)
    assert factored.exp_neg_phi.shape == (2, 5, 8, 8)
    assert factored.irrep_dims == [4, 4]


# --- stable_matrix_exp_pair clamp monitor (audit 2026-07-01 C14 safe variant) ---------------------

def test_clamp_monitor_warns_when_active():
    """clamp_monitor=True surfaces the Frobenius-clamp surrogate: a matrix with ||M||_F >> max_norm
    warns (the returned factor is exp(max_norm*M/||M||_F), not exp(M)); a small-norm matrix with
    the flag on does NOT warn (the clamp is inactive, the operator exact)."""
    import warnings
    from vfe3.geometry.transport import stable_matrix_exp_pair

    big = torch.zeros(2, 3, 3)
    big[0, 0, 1] = 100.0                                          # ||M||_F = 100 >> max_norm = 15
    with pytest.warns(RuntimeWarning, match="Frobenius clamp active"):
        stable_matrix_exp_pair(big, clamp_monitor=True)

    small = 0.1 * torch.eye(3).expand(2, 3, 3).contiguous()       # ||M||_F well below max_norm
    with warnings.catch_warnings():
        warnings.simplefilter("error")                            # any warning -> test failure
        stable_matrix_exp_pair(small, clamp_monitor=True)


def test_clamp_monitor_default_off_is_bit_identical():
    """The default clamp_monitor=False path is bit-identical to an explicit False call and emits
    no warning even when the clamp is active -- the hot path stays reduction/host-sync free."""
    import warnings
    from vfe3.geometry.transport import stable_matrix_exp_pair

    g = torch.Generator().manual_seed(0)
    m = 30.0 * torch.randn(2, 4, 4, generator=g)                  # clamp active on this input
    with warnings.catch_warnings():
        warnings.simplefilter("error")                            # any warning -> test failure
        exp_default, neg_default = stable_matrix_exp_pair(m)      # default flag: silent
        exp_off, neg_off = stable_matrix_exp_pair(m, clamp_monitor=False)
    assert torch.equal(exp_default, exp_off)
    assert torch.equal(neg_default, neg_off)


def test_skew_transport_exp_not_clamped_for_large_phi():
    # m16: matrix_exp of a SKEW matrix is orthogonal at ANY norm, so the flat transport builder must
    # NOT apply the Frobenius clamp for skew (so_n/so_k) groups -- else a large ||embed(phi)|| silently
    # returns exp of the RESCALED matrix (a shorter rotation), not exp(embed(phi)), on the pure path.
    from vfe3.geometry.transport import compute_transport_operators
    grp = get_group("so_n")(5, group_n=5, irrep_spec=[("l1", 1)])
    assert grp.skew_symmetric
    phi = torch.zeros(1, 1, grp.generators.shape[0])
    phi[0, 0, 0] = 30.0                                          # ||embed(phi)||_F >> 15 (the clamp threshold)
    out = compute_transport_operators(phi, grp)
    phi_mat = torch.einsum("bna,aij->bnij", phi, grp.generators)
    true_exp = torch.linalg.matrix_exp(phi_mat)
    assert torch.allclose(out["exp_phi"], true_exp, atol=1e-4)   # RED pre-fix: clamped to a shorter rotation
    eye = torch.eye(5).expand(1, 1, 5, 5)
    assert torch.allclose(torch.matmul(out["exp_phi"], out["exp_phi"].transpose(-1, -2)), eye, atol=1e-4)
