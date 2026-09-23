r"""Isotypic (label-grouped) HeadMixer: the full linear commutant of a mixed irrep tower."""

import math

import pytest
import torch

from vfe3.geometry.groups import get_group
from vfe3.model.head_mixer import HeadMixer


def test_label_runs_become_components():
    m = HeadMixer([1, 3, 3, 7], irrep_labels=["l0", "l1", "l1", "l3"])
    assert [tuple(d.shape) for d in m.mixer_deltas] == [(1, 1), (2, 2), (1, 1)]
    assert m.is_identity()


def test_unlabeled_unequal_dims_still_raise():
    with pytest.raises(ValueError, match="equal-size blocks"):
        HeadMixer([1, 3, 7, 9])


def test_legacy_single_component_keeps_mixer_delta_attribute():
    m = HeadMixer([4, 4, 4])
    assert m.mixer_delta.shape == (3, 3)        # back-compat accessor (single component)


def test_identity_init_is_exact_passthrough():
    m = HeadMixer([1, 3, 3, 7], irrep_labels=["l0", "l1", "l1", "l3"])
    mu = torch.randn(2, 5, 14)
    sig = torch.rand(2, 5, 14) + 0.5
    mu2, sig2 = m(mu, sig)
    assert torch.equal(mu2, mu) and torch.equal(sig2, sig)


def test_isotypic_mixer_exactly_equivariant_under_tower_gauge_full_cov():
    # mix(g mu, g S g^T) == (g mix_mu, g mix_S g^T) for a trained (non-identity) mixer,
    # because blockdiag_t(A_t kron I_d) is the commutant of the tower (real-type irreps).
    torch.manual_seed(0)
    grp = get_group("so_n")(14, group_n=3,
                            irrep_spec=[("l0", 1), ("l1", 2), ("l3", 1)],
                            dtype=torch.float64)
    m = HeadMixer(grp.irrep_dims, irrep_labels=grp.irrep_labels).double()
    with torch.no_grad():
        for d in m.mixer_deltas:
            d.copy_(0.3 * torch.randn(*d.shape, dtype=torch.float64))
    g = torch.linalg.matrix_exp(
        torch.einsum("a,aij->ij", 0.4 * torch.randn(3, dtype=torch.float64), grp.generators))
    mu = torch.randn(5, 14, dtype=torch.float64)
    A = torch.randn(5, 14, 14, dtype=torch.float64)
    S = A @ A.transpose(-1, -2) + torch.eye(14, dtype=torch.float64)
    mu_m, S_m = m(mu, S)
    mu_mg = torch.einsum("kl,nl->nk", g, mu_m)
    S_mg = g @ S_m @ g.T
    mu_gm, S_gm = m(torch.einsum("kl,nl->nk", g, mu), g @ S @ g.T)
    assert (mu_gm - mu_mg).abs().max() < 1e-12
    assert (S_gm - S_mg).abs().max() < 1e-11


def test_mults_one_tower_gives_scalar_gains():
    m = HeadMixer([1, 3, 5, 7], irrep_labels=["l0", "l1", "l2", "l3"])
    assert all(tuple(d.shape) == (1, 1) for d in m.mixer_deltas)
    with torch.no_grad():
        m.mixer_deltas[2].fill_(0.5)            # gain exp(0.5) on the l2 head
    mu = torch.randn(3, 16)
    mu2, _ = m(mu, torch.ones(3, 16))
    assert torch.allclose(mu2[:, 4:9], math.exp(0.5) * mu[:, 4:9])
    assert torch.equal(mu2[:, :4], mu[:, :4])   # other heads untouched


def test_same_label_run_with_unequal_dims_raises():
    with pytest.raises(ValueError, match="share label"):
        HeadMixer([1, 3, 3, 7], irrep_labels=["x", "x", "x", "x"])
    with pytest.raises(ValueError, match="irrep_labels has"):
        HeadMixer([3, 3], irrep_labels=["a"])


def test_old_single_component_state_dict_still_loads():
    m = HeadMixer([4, 4, 4])
    old_style = {"mixer_delta": 0.25 * torch.ones(3, 3)}
    m.load_state_dict(old_style)
    assert torch.allclose(m.mixer_deltas[0], 0.25 * torch.ones(3, 3))


def test_so_n_mixed_tower_model_constructs_with_mixer_and_trains():
    from vfe3.config import VFE3Config
    from vfe3.model.model import VFEModel
    cfg = VFE3Config(vocab_size=20, embed_dim=8, n_heads=2, max_seq_len=5, n_layers=1,
                     n_e_steps=1, e_q_mu_lr=0.05, e_phi_lr=0.0,
                     gauge_group="so_n", group_n=3,
                     irrep_spec=[("l0", 1), ("l1", 1), ("l0", 1), ("l1", 1)],
                     use_head_mixer=True, phi_precond_mode="none")
    model = VFEModel(cfg)                       # pre-fix: raises (unequal dims [1,3,1,3])
    assert [tuple(d.shape) for d in model.head_mixer.mixer_deltas] == [(1, 1), (1, 1), (1, 1), (1, 1)]
    with torch.no_grad():
        model.head_mixer.mixer_deltas[1].fill_(0.1)
    tok = torch.randint(0, 20, (2, 5)); tgt = torch.randint(0, 20, (2, 5))
    _, loss, _ = model(tok, tgt)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(model.head_mixer.mixer_deltas[1].grad).all()
