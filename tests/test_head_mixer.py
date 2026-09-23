r"""Schur-commutant head mixer (use_head_mixer, opt-in).

Mixes the equal-size gauge-irrep blocks (under block_glk: the n_heads heads) with a learned
per-type matrix A = exp(delta) embedded as kron(A, I_d), applied symmetrically to mu (M mu) and
Sigma (M Sigma M^T; diagonal closed form sigma'[m] = sum_n A[m,n]^2 sigma[n]). Identity init
(delta=0) makes a mixer-on model bitwise-identical to mixer-off at step 0. Under block_glk's
UNTIED per-block gauge the mixer breaks strict gauge equivariance (exact at init, deviates as A
drifts) -- user-accepted; this suite pins the algebra and the wiring, not equivariance.
"""

import pytest
import torch

from vfe3.config import VFE3Config
from vfe3.model.head_mixer import HeadMixer
from vfe3.model.model import VFEModel
from vfe3.train import build_optimizer


def test_identity_init_is_a_noop():
    mix = HeadMixer([2, 2])                                   # K=4, 2 blocks of dim 2
    mu = torch.randn(3, 5, 4)
    sigma = torch.rand(3, 5, 4) + 0.5
    mu2, sigma2 = mix(mu, sigma)
    assert torch.allclose(mu2, mu)
    assert torch.allclose(sigma2, sigma)
    assert mix.is_identity()


def test_mixes_means_across_blocks():
    mix = HeadMixer([2, 2])
    with torch.no_grad():
        mix.mixer_delta.copy_(torch.tensor([[0.0, 1.0], [0.0, 0.0]]))   # A = [[1,1],[0,1]]
    mu = torch.zeros(1, 1, 4)
    mu[..., 0:2] = torch.tensor([1.0, 2.0])                  # block 0
    mu[..., 2:4] = torch.tensor([3.0, 4.0])                  # block 1
    mu2, _ = mix(mu, torch.ones(1, 1, 4))
    # A=[[1,1],[0,1]]: block0' = block0 + block1 ; block1' = block1
    assert torch.allclose(mu2[..., 0:2], torch.tensor([4.0, 6.0]))
    assert torch.allclose(mu2[..., 2:4], torch.tensor([3.0, 4.0]))


def test_diagonal_sigma_closed_form_is_A_squared():
    mix = HeadMixer([1, 1])                                  # K=2, 2 scalar blocks
    with torch.no_grad():
        mix.mixer_delta.copy_(torch.tensor([[0.0, 1.0], [0.0, 0.0]]))   # A=[[1,1],[0,1]]
    sigma = torch.tensor([[[2.0, 3.0]]])                     # (1,1,2): sigma0=2, sigma1=3
    _, sigma2 = mix(torch.zeros(1, 1, 2), sigma)
    # sigma'[0] = A[0,0]^2*2 + A[0,1]^2*3 = 2 + 3 = 5 ; sigma'[1] = A[1,1]^2*3 = 3
    assert torch.allclose(sigma2, torch.tensor([[[5.0, 3.0]]]))


def test_exponential_parameterization_prevents_singular_covariance():
    mix = HeadMixer([1, 1])
    with torch.no_grad():
        mix.mixer_delta.copy_(-torch.eye(2))

    mu = torch.tensor([[[1.0, 2.0]]])
    sigma = torch.tensor([[[2.0, 3.0]]])
    _, sigma_diag = mix(mu, sigma)
    _, sigma_full = mix(mu, torch.diag_embed(sigma))

    assert (sigma_diag > 0.0).all()
    assert (torch.linalg.eigvalsh(sigma_full) > 0.0).all()


def test_laplace_exponential_mixer_has_positive_scale_and_finite_gradients():
    mix = HeadMixer([1, 1], family="laplace_diagonal")
    with torch.no_grad():
        mix.mixer_delta.copy_(-torch.eye(2))
    dispersion = torch.tensor([[[2.0, 3.0]]], requires_grad=True)

    mu_out, dispersion_out = mix(torch.tensor([[[1.0, 2.0]]]), dispersion)
    (mu_out.sum() + dispersion_out.sum()).backward()

    assert (dispersion_out > 0.0).all()
    assert torch.isfinite(mix.mixer_delta.grad).all()
    assert torch.isfinite(dispersion.grad).all()


def test_full_cov_sandwich_matches_explicit_kron():
    torch.manual_seed(0)
    n, d = 2, 2
    mix = HeadMixer([d, d])
    with torch.no_grad():
        mix.mixer_delta.normal_(0.0, 0.3)
    A = torch.linalg.matrix_exp(mix.mixer_delta)
    M = torch.kron(A, torch.eye(d))
    base = torch.randn(n * d, n * d)
    S = (base.t() @ base).reshape(1, 1, n * d, n * d)        # SPD full covariance
    _, S2 = mix(torch.zeros(1, 1, n * d), S)
    assert torch.allclose(S2[0, 0], M @ S[0, 0] @ M.t(), atol=1e-5)


def test_requires_at_least_two_equal_blocks():
    with pytest.raises(ValueError):
        HeadMixer([4])                                       # single block: nothing to mix
    with pytest.raises(ValueError):
        HeadMixer([2, 3])                                    # unequal blocks: kron(A, I_d) ill-defined


def test_config_use_head_mixer_default_off_and_toggles():
    assert VFE3Config().use_head_mixer is False
    assert VFE3Config(use_head_mixer=True).use_head_mixer is True


def test_model_head_mixer_is_noop_at_init_and_trains():
    base = dict(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1,
                n_e_steps=1, e_q_mu_lr=0.05, e_phi_lr=0.0)
    tok = torch.randint(0, 20, (3, 5))
    torch.manual_seed(0)
    m_off = VFEModel(VFE3Config(**base, use_head_mixer=False))
    torch.manual_seed(0)
    m_on = VFEModel(VFE3Config(**base, use_head_mixer=True))
    assert m_off.head_mixer is None and m_on.head_mixer is not None
    assert torch.allclose(m_off(tok), m_on(tok), atol=1e-6)  # identity init -> same logits at step 0
    targets = torch.randint(0, 20, (3, 5))
    _, loss, _ = m_on(tok, targets)
    loss.backward()
    assert m_on.head_mixer.mixer_delta.grad is not None      # the mixer is in the training graph


def test_optimizer_covers_head_mixer_params():
    cfg = VFE3Config(
        vocab_size=20,
        embed_dim=4,
        n_heads=2,
        use_head_mixer=True,
        m_p_mu_lr=0.0123,
        m_head_mixer_lr=0.0022,
    )
    model = VFEModel(cfg)
    opt = build_optimizer(model, cfg)
    grouped = {p for g in opt.param_groups for p in g["params"]}
    mixer_group = next(
        group for group in opt.param_groups
        if any(parameter is model.head_mixer.mixer_delta for parameter in group["params"])
    )
    assert grouped == set(model.parameters())
    assert model.head_mixer.mixer_delta in grouped
    assert mixer_group["lr"] == pytest.approx(cfg.m_head_mixer_lr)
    assert mixer_group["weight_decay"] == pytest.approx(cfg.weight_decay)
    assert mixer_group["lr_aux_role"] == "head_mixer"


def test_head_mixer_equivariant_under_tied_gauge_full_cov():
    # THEORY PAYOFF: under a TIED gauge Omega = kron(I_n, h) (h in GL(d)), the Schur-commutant
    # mixer M = kron(A, I_d) commutes with Omega, so the FULL-COVARIANCE mixer is EXACTLY gauge-
    # equivariant at the mixer-operation level:
    #     mix(Omega mu, Omega Sigma Omega^T) == (Omega (M mu), Omega (M Sigma M^T) Omega^T).
    # This is the equivariance that block_glk's UNTIED per-head gauge breaks and tied_block_glk
    # (kron(I_n, gl(d))) restores. NOTE: this is a statement about the MIXER operation under the
    # tied gauge, NOT a claim that the whole model is gauge-equivariant. It is tested on the
    # full-covariance path; the diagonal closed form is only equivariant under DIAGONAL gauges
    # (the diagonal-of-sandwich approximation used throughout V3), so it is deliberately not
    # asserted under this general tied gauge.
    torch.manual_seed(0)
    n, d = 2, 3
    K = n * d
    mix = HeadMixer([d, d])
    with torch.no_grad():
        mix.mixer_delta.normal_(0.0, 0.4)                    # A = exp(Delta), nontrivial
    h = torch.eye(d) + 0.3 * torch.randn(d, d)               # h in GL(d) (near I, invertible)
    Omega = torch.kron(torch.eye(n), h)                      # (K, K) tied gauge: same h in every head
    mu = torch.randn(1, 1, K)
    base = torch.randn(K, K)
    Sigma = (base @ base.t() + K * torch.eye(K)).reshape(1, 1, K, K)   # SPD full covariance

    mu_g = (Omega @ mu.unsqueeze(-1)).squeeze(-1)            # gauge then mix
    Sigma_g = Omega @ Sigma @ Omega.t()
    mu_L, Sigma_L = mix(mu_g, Sigma_g)

    mu_m, Sigma_m = mix(mu, Sigma)                           # mix then gauge
    mu_R = (Omega @ mu_m.unsqueeze(-1)).squeeze(-1)
    Sigma_R = Omega @ Sigma_m @ Omega.t()

    assert torch.allclose(mu_L, mu_R, atol=1e-5)
    assert torch.allclose(Sigma_L, Sigma_R, atol=1e-4)


def test_head_mixer_rejected_for_single_block_group_at_model_build():
    # glk / so_k resolve to a single irrep block (nothing to mix); requesting the mixer there
    # must fail at construction, not silently no-op or crash at the first forward.
    with pytest.raises(ValueError):
        VFEModel(VFE3Config(vocab_size=10, embed_dim=4, n_heads=2, max_seq_len=3,
                            gauge_group="glk", use_head_mixer=True))
