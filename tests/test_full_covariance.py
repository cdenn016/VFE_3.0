r"""Full-covariance (gaussian_full) pure path: end-to-end runnability + golden equivalence.

GL(K) audit finding #2: the GL(K)-invariant covariance sandwich Omega Sigma Omega^T must be
runnable end-to-end through VFEModel / the E-step under appropriate toggles, not only as
isolated kernels. The toggles are family='gaussian_full' + decode_mode='full_chunked'
(diagonal_covariance is a derived read-only property of family).
"""

from tests.profiles import test_config
import pytest
import torch

from vfe3.model.model import VFEModel


def test_full_covariance_config_derives_diagonal_covariance_flag():
    """diagonal_covariance is a derived read-only property of family (single source of truth)."""
    assert test_config(family="gaussian_diagonal").diagonal_covariance is True
    assert test_config(family="gaussian_full", decode_mode="full_chunked").diagonal_covariance is False
    with pytest.raises(TypeError):
        test_config(family="gaussian_full", diagonal_covariance=True)    # no longer a settable field


def test_full_gaussian_fisher_precision_has_no_ridge_on_valid_spd_input():
    from vfe3.families.gaussian import FullGaussian

    sigma = torch.tensor([[2.0, 0.25], [0.25, 1.0]], dtype=torch.float64)
    small_eps = FullGaussian.mean_fisher_precision(sigma, eps=1e-12)
    large_eps = FullGaussian.mean_fisher_precision(sigma, eps=1e-2)

    assert torch.equal(small_eps, large_eps)
    torch.testing.assert_close(small_eps @ sigma, torch.eye(2, dtype=sigma.dtype))


def test_full_covariance_model_runs_end_to_end():
    """The full-covariance pure path runs encode -> E-step -> full decode -> CE, forward+backward."""
    cfg = test_config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1,
                     n_e_steps=2, e_q_mu_lr=0.05, e_q_sigma_lr=0.01,
                     family="gaussian_full", decode_mode="full_chunked")
    model = VFEModel(cfg)
    tokens = torch.randint(0, 20, (2, 5)); targets = torch.randint(0, 20, (2, 5))
    beliefs = model.prior_bank.encode(tokens)
    assert beliefs.sigma.shape == (2, 5, 4, 4)              # full SPD covariance encode
    logits, loss, _ = model(tokens, targets)
    assert logits is None and torch.isfinite(loss)
    loss.backward()
    assert model.prior_bank.mu_embed.grad is not None and model.prior_bank.mu_embed.grad.abs().sum() > 0


def _spd(N, K, gen):
    r"""A batch of genuinely non-diagonal SPD matrices A Aᵀ + K I."""
    A = torch.randn(N, K, K, generator=gen)
    return A @ A.transpose(-1, -2) + K * torch.eye(K)


def test_full_kl_survives_non_pd_covariance():
    # The alpha=1 full-covariance KL must CLAMP (not raise) on a numerically non-PD covariance.
    # Such a prior covariance can arise after training shifts it, and full-cov configs route the
    # belief KL through this closed form via the E-step oracle (e.g. decode_mode='full_chunked'); a raw
    # torch.linalg.cholesky there raises and kills the run. The alpha != 1 branch was already
    # hardened with safe_cholesky; this pins the same robustness for alpha = 1.
    from vfe3.families.gaussian import FullGaussian
    K = 4
    mu = torch.zeros(2, K)
    sigma_q = torch.eye(K).expand(2, K, K).contiguous()
    bad = torch.eye(K).clone(); bad[0, 0] = -1.0                 # a negative eigenvalue -> not PD
    sigma_p = bad.expand(2, K, K).contiguous()
    kl = FullGaussian(mu, sigma_q).renyi_closed_form(            # must NOT raise
        FullGaussian(mu, sigma_p), alpha=1.0, kl_max=100.0, eps=1e-6)
    assert torch.isfinite(kl).all()
    assert (kl <= 100.0 + 1e-3).all()


def test_full_entropy_marks_non_pd_covariance_invalid():
    # FullGaussian.entropy must use safe_cholesky without reporting a finite-but-wrong entropy when
    # every jitter round fails.  Its sibling log_partition_at path propagates NaN under the same
    # condition, while the valid-SPD path remains finite.
    from vfe3.families.gaussian import FullGaussian
    K = 4
    mu = torch.zeros(2, K)
    bad = torch.eye(K).clone(); bad[0, 0] = -1.0                 # negative eigenvalue -> not PD
    h = FullGaussian(mu, bad.expand(2, K, K).contiguous()).entropy()   # must NOT raise
    assert torch.isnan(h).all()


# ===========================================================================
# PB-14 (Task 5, 2026-07-12): family/divergence-consistent FULL decode.
#
# The generic `family` decode scores a full q against the intentionally DIAGONAL vocabulary
# prior table (promoted to full via diag_embed only when the family is full) through the
# configured divergence, with NO kl_max ranking clamp. It must equal a direct full-family
# functional call, and the full config must construct via covariance-kind membership WITHOUT
# adding a vocabulary/decode lower-triangle state key.
# ===========================================================================

from vfe3.divergence import get_family, get_functional                # noqa: E402
from vfe3.model.prior_bank import PriorBank                # noqa: E402
from vfe3.model.prior_bank import _decode_family, _decode_full
from vfe3.numerics import bounded_variance_from_log                    # noqa: E402


def _full_reference_logits(pb, mu_q, sigma_q, tau_eff):
    family_cls = get_family(pb.family)
    q = family_cls(mu_q.unsqueeze(-2), sigma_q.unsqueeze(-3))
    p_sigma = torch.diag_embed(bounded_variance_from_log(pb._decode_sigma_log_table(), eps=pb.eps))
    p = family_cls(pb._decode_mu_table(), p_sigma)
    functional = get_functional("renyi")
    energy = functional(q, p, alpha=1.0, kl_max=float("inf"), eps=pb.eps)
    return -energy / tau_eff


def _spd_batch(B, N, K, gen):
    A = torch.randn(B, N, K, K, generator=gen)
    return A @ A.transpose(-1, -2) + K * torch.eye(K)


@pytest.mark.parametrize("alpha", [1.0])
def test_family_decode_matches_direct_functional_full_gaussian(alpha):
    g = torch.Generator().manual_seed(0)
    V, K, n_gen = 7, 3, 4
    pb = PriorBank(V, K, n_gen, decode_tau=1.2, family="gaussian_full",
                   decode_mode="full_chunked",
                   diagonal_covariance=False)
    with torch.no_grad():
        pb.mu_embed.normal_(0.0, 0.6)
        pb.sigma_log_embed.normal_(0.0, 0.3)
    mu_q = torch.randn(2, 4, K, generator=g)
    sigma_q = _spd_batch(2, 4, K, g)                            # full SPD query covariance
    tau_eff = pb._tau_eff()
    got = _decode_family(pb, mu_q, sigma_q, tau_eff)
    exp = _full_reference_logits(pb, mu_q, sigma_q, tau_eff)
    assert got.shape == (2, 4, V)
    assert torch.allclose(got, exp, atol=1e-4, rtol=0.0)
    if alpha == 1.0:                                            # canonical: matches the fast full kernel
        full = _decode_full(pb, mu_q, sigma_q, tau_eff)
        assert torch.allclose(got, full, atol=1e-3)


def test_full_family_decode_config_has_no_decode_lower_triangle_keys():
    cfg = test_config(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1,
                     family="gaussian_full", decode_mode="full_chunked",
                     untie_decode_bank=True)
    model = VFEModel(cfg)
    names = {n for n, _ in model.named_parameters()}
    assert not any("sigma_lower" in n for n in names)           # decode/vocab tables stay diagonal
    assert model.prior_bank.decode_sigma_log_embed.shape == (6, 4)
    # end-to-end: full family + full-covariance decode runs forward + backward.
    tokens = torch.randint(0, 6, (2, 5)); targets = torch.randint(0, 6, (2, 5))
    _, loss, _ = model(tokens, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.prior_bank.mu_embed.grad is not None
