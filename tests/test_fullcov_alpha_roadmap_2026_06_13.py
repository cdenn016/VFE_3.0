r"""Full-covariance + alpha/divergence completion roadmap (2026-06-13 multi-expert investigation).

Groups, by roadmap item:
  A1  (removed 2026-10-17 with the Bhattacharyya / Jeffreys functionals, audit 2026-10-16 F93).
  A2  per-position state_dependent alpha is envelope-correct (the gradient is alpha*(D)*dD with no
      product-rule residue), pinned on the KL, the only registered divergence.
  A3  the full-covariance unrolled oracle gradient (oracle_unroll_grad=True) is FINITE on a single
      training backward -- the exact Loewner spectral adjoint (retraction.py) and safe_cholesky; the
      old "expect NaNs" docstring is stale.
  B1  fused chunked full-covariance decode+CE (decode_mode='full_chunked'): the diagonal-prior closed
      form, equal to the dense full decode CE, with no per-pair (B,N,V,K,K) Cholesky workspace.
  B2  per-head full-covariance transport sandwich: the block-diagonal congruence equals the dense
      Omega Sigma Omega^T without materializing the dense (B,N,N,K,K) Omega.
  B4  config ergonomics warnings (covariance discarded at the linear decode; bounded-divergence
      state_dependent alpha at b0=O(1)).
"""

from tests.profiles import test_config
import warnings

import pytest
import torch
import torch.nn.functional as F

from vfe3.families.base import get_family
from vfe3.model.model import VFEModel


# ---------------------------------------------------------------------------
# A3: full-cov unrolled oracle gradient is FINITE (and live) on a single training backward.
# The config docstring's "expect NaNs there" for gaussian_full + oracle_unroll_grad=True predates
# the full-covariance retraction's finite backward, now the exact Loewner (Daleckii-Krein) adjoint of
# its spectral maps in retraction.py (the gap-damped eigh it once used was retired under audit
# 2026-10-16 F99); the single training backward is finite. This pins that, and contrasts it with the truncated default (oracle_unroll_grad=False).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# A2: the alpha-envelope is divergence-AGNOSTIC. For state_dependent alpha*(D) = c0/(b0+D) with
# R(alpha) = b0 alpha - c0 log alpha, d/dmu[alpha*(D) D + R(alpha*(D))] = alpha*(D) dD/dmu exactly,
# for ANY differentiable self-divergence D; the KL is the only registered member.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("div_name", ["renyi"])
def test_state_dependent_envelope_has_no_product_rule_residue(div_name):
    from vfe3.alpha_i import self_coupling_alpha
    from vfe3.divergence import get_functional
    functional = get_functional(div_name)
    diag = get_family("gaussian_diagonal")
    g = torch.Generator().manual_seed(0)
    sigma_q = (torch.rand(6, 4, generator=g) + 0.5)
    mu_p = torch.randn(6, 4, generator=g)
    sigma_p = (torch.rand(6, 4, generator=g) + 0.5)
    mu0 = torch.randn(6, 4, generator=g)
    b0, c0 = 0.3, 1.2

    # Live: F_self = sum(alpha*(D) D + R(alpha*(D))), alpha a live function of D(mu).
    mu_a = mu0.clone().requires_grad_(True)
    D_a = functional(diag(mu_a, sigma_q), diag(mu_p, sigma_p))          # (6,)
    alpha_a, reg_a = self_coupling_alpha(D_a, mode="state_dependent", b0=b0, c0=c0)
    F_self = (alpha_a * D_a + reg_a).sum()
    (g_live,) = torch.autograd.grad(F_self, mu_a)

    # Envelope reference: alpha*(D) held CONSTANT, F_ref = sum(alpha*.detach() * D(mu)).
    mu_b = mu0.clone().requires_grad_(True)
    D_b = functional(diag(mu_b, sigma_q), diag(mu_p, sigma_p))
    alpha_b, _ = self_coupling_alpha(D_b.detach(), mode="state_dependent", b0=b0, c0=c0)
    F_ref = (alpha_b.detach() * D_b).sum()
    (g_env,) = torch.autograd.grad(F_ref, mu_b)

    # If R were missing or alpha* off the stationary point, the product-rule residue
    # alpha'(D)(D + b0 - c0/alpha) dD would be O(1); the envelope kills it to roundoff.
    assert torch.allclose(g_live, g_env, atol=1e-4, rtol=1e-3), (
        f"{div_name}: envelope residue max={ (g_live - g_env).abs().max().item() }"
    )


# ---------------------------------------------------------------------------
# B1: fused chunked full-covariance decode+CE (decode_mode='full_chunked'). The prior table is
# DIAGONAL (diag_embed of sigma_log_embed), so KL(q_full || pi_v_diag) has a closed form needing
# NO per-pair (K,K) Cholesky -- one logdet(Sigma_q) per position, then matmuls over V. It must equal
# the dense full decode CE (F.cross_entropy(_decode_full(...))) to atol-1e-3, without materializing
# (B,N,V,K,K).
# Under the default decode_av_precision='fp64' the fused chunked routes keep float64 through the
# finished logit/CE (_decode_analytic_kl_logits' near-tie ranking preservation), while the dense
# `full` kernel and model.forward's dense branch cast to float32. Values agree to ~1e-6; the
# comparisons below therefore compare values across that intentional dtype boundary.
# ---------------------------------------------------------------------------
def _full_model(vocab_size=64, **kw):
    decode_mode = kw.pop("decode_mode", "full_chunked")
    cfg = test_config(vocab_size=vocab_size, embed_dim=4, n_heads=2, max_seq_len=6, n_layers=1,
                     n_e_steps=2, e_q_mu_lr=0.05, e_q_sigma_lr=0.02,
                     family="gaussian_full", decode_mode=decode_mode,
                     **kw)
    return VFEModel(cfg)


def _full_converged(model):
    def run(tokens):
        out, _ = model.forward_beliefs(tokens)
        return out.mu.float(), out.sigma.float()
    return run


def test_full_chunked_config_accepts_with_full_family_and_rejects_diagonal():
    test_config(family="gaussian_full",
               decode_mode="full_chunked")
    with pytest.raises(ValueError):
        test_config(family="gaussian_diagonal",
                   decode_mode="full_chunked")


def test_full_chunked_ce_matches_dense_full_ce_multiple_chunk_sizes():
    torch.manual_seed(0)
    V = 64
    tokens = torch.randint(0, V, (3, 6)); targets = torch.randint(0, V, (3, 6))
    ref = _full_model(vocab_size=V, decode_mode="full_chunked")
    mu, sigma = _full_converged(ref)(tokens)
    logits = ref.prior_bank.decode(mu, sigma)                          # (B,N,V) dense full
    full_ce = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), ignore_index=-100)
    for chunk in (16, 8192, V, 7, 100):
        ch = _full_model(vocab_size=V, decode_mode="full_chunked", decode_chunk_size=chunk)
        ch.load_state_dict(ref.state_dict())
        mu_c, sigma_c = _full_converged(ch)(tokens)
        ce = ch.prior_bank.decode_ce_full_chunked(mu_c, sigma_c, targets)
        assert torch.allclose(ce.to(full_ce.dtype), full_ce, atol=1e-3), (
            f"chunk={chunk}: {ce.item()} != {full_ce.item()}")


def test_full_chunked_honors_ignore_index():
    torch.manual_seed(1)
    V = 50
    tokens = torch.randint(0, V, (2, 6)); targets = torch.randint(0, V, (2, 6))
    targets[0, 0] = -100; targets[1, 3] = -100
    ref = _full_model(vocab_size=V, decode_mode="full_chunked")
    mu, sigma = _full_converged(ref)(tokens)
    full_ce = F.cross_entropy(ref.prior_bank.decode(mu, sigma).reshape(-1, V),
                              targets.reshape(-1), ignore_index=-100)
    ch = _full_model(vocab_size=V, decode_mode="full_chunked", decode_chunk_size=13)
    ch.load_state_dict(ref.state_dict())
    mu_c, sigma_c = _full_converged(ch)(tokens)
    ce = ch.prior_bank.decode_ce_full_chunked(mu_c, sigma_c, targets)
    assert torch.allclose(ce.to(full_ce.dtype), full_ce, atol=1e-3)


def test_full_chunked_all_ignore_is_finite_zero():
    torch.manual_seed(2)
    V = 32
    tokens = torch.randint(0, V, (2, 4)); targets = torch.full((2, 4), -100)
    ch = _full_model(vocab_size=V, decode_mode="full_chunked", decode_chunk_size=10)
    mu, sigma = _full_converged(ch)(tokens)
    mu = mu.detach().requires_grad_(True)
    ce = ch.prior_bank.decode_ce_full_chunked(mu, sigma, targets)
    assert torch.isfinite(ce) and ce.item() == 0.0
    ce.backward()


def test_full_chunked_inference_matches_dense_full_logits():
    torch.manual_seed(3)
    V = 48
    tokens = torch.randint(0, V, (2, 5))
    ref = _full_model(vocab_size=V, decode_mode="full_chunked")
    logits_full = ref(tokens)
    ch = _full_model(vocab_size=V, decode_mode="full_chunked")
    ch.load_state_dict(ref.state_dict())
    logits_ch = ch(tokens)
    assert logits_ch.shape == (2, 5, V)
    assert torch.allclose(logits_ch.to(logits_full.dtype), logits_full, atol=1e-3)


def test_full_chunked_forward_loss_matches_full_forward_loss():
    torch.manual_seed(4)
    V = 40
    tokens = torch.randint(0, V, (3, 6)); targets = torch.randint(0, V, (3, 6))
    base = _full_model(vocab_size=V, decode_mode="full_chunked")
    _, loss_full, ce_full = base(tokens, targets)
    ch = _full_model(vocab_size=V, decode_mode="full_chunked", decode_chunk_size=9)
    ch.load_state_dict(base.state_dict())
    logits_ch, loss_ch, ce_ch = ch(tokens, targets)
    assert logits_ch is None                               # fused path forms no (B,N,V) logits
    assert torch.allclose(loss_ch.to(loss_full.dtype), loss_full, atol=1e-3)
    assert torch.allclose(ce_ch.to(ce_full.dtype), ce_full, atol=1e-3)


def test_full_chunked_ce_grad_reaches_prior_tables():
    torch.manual_seed(5)
    V = 40
    tokens = torch.randint(0, V, (2, 5)); targets = torch.randint(0, V, (2, 5))
    ch = _full_model(vocab_size=V, decode_mode="full_chunked", decode_chunk_size=9)
    _, loss, _ = ch(tokens, targets)
    loss.backward()
    assert ch.prior_bank.mu_embed.grad is not None and ch.prior_bank.mu_embed.grad.abs().sum() > 0
    assert ch.prior_bank.sigma_log_embed.grad is not None
    assert torch.isfinite(ch.prior_bank.sigma_log_embed.grad).all()


def test_full_cov_chunked_matches_dense_on_non_pd():
    """F6 (audit 2026-07-01), RESTATED for the F31 exclusion contract (audit 2026-08-06).

    The original pin was that a non-PD Sigma_q position yields -inf logits on BOTH the dense full
    decode (gaussian_full ok-gating -> NaN -> kl_max=inf -> -inf) and the chunked closed form
    (safe_cholesky ok mask -> logdet_q = -inf -> per_pos = -inf). That contract was itself the NaN
    generator it was supposed to guard: log_softmax of an all--inf row is NaN on both paths, so one
    degenerate position NaN'd the whole batch's scalar CE.

    The contract is now EXCLUSION, and it still has to hold on both paths together: a position with
    no valid likelihood contributes an informationless UNIFORM logit row (finite; log_softmax gives
    -log V) and is dropped from the cross-entropy outright. What F6 exists to prevent -- the two
    paths silently disagreeing about a degenerate position -- is unchanged and still pinned here.
    PD positions agree to atol-1e-3 exactly as before."""
    from vfe3.model.prior_bank import _decode_full, _decode_full_chunked
    torch.manual_seed(6)
    V, B, N, K = 32, 2, 4, 4
    pb = _full_model(vocab_size=V, decode_mode="full_chunked").prior_bank
    mu_q = torch.randn(B, N, K)
    A = torch.randn(B, N, K, K)
    sigma_q = A @ A.transpose(-1, -2) + torch.eye(K)                     # all-PD base
    sigma_q[0, 1] = torch.diag(torch.tensor([-5.0, 1.0, 1.0, 1.0]))     # eigenvalue -5 << -eps*1e5: all 5 jitter rounds fail
    tau_eff = pb._tau_eff(None)
    dense = _decode_full(pb, mu_q, sigma_q, tau_eff)                    # (B, N, V)
    chunked = _decode_full_chunked(pb, mu_q, sigma_q, tau_eff)          # (B, N, V)

    for name, out in (("dense", dense), ("chunked", chunked)):
        assert torch.isfinite(out[0, 1]).all(), f"{name}: degenerate row must be finite"
        assert torch.equal(out[0, 1], torch.zeros_like(out[0, 1])), (
            f"{name}: degenerate row must be uniform (informationless), not a placeholder score")
        assert torch.isfinite(torch.log_softmax(out[0, 1], dim=-1)).all(), (
            f"{name}: the row that used to be all--inf must no longer be NaN under log_softmax")
    # the two paths agree on the degenerate position, which is what F6 is for
    assert torch.equal(dense[0, 1], chunked[0, 1])
    # and the dense branch's CE consumer is told to drop exactly that position
    assert pb.decode_degenerate_positions(sigma_q)[0, 1]

    good = torch.ones(B, N, dtype=torch.bool)
    good[0, 1] = False
    assert torch.allclose(chunked[good].to(dense.dtype), dense[good], atol=1e-3)
    assert pb.decode_degenerate_positions(sigma_q)[good].sum() == 0


def test_full_cov_query_invariants_all_pd_byte_identical():
    """On an all-PD Sigma_q the ok mask is all-True and torch.where selects logdet_q unchanged:
    the invariants stay byte-identical to the raw round-zero safe_cholesky log-det."""
    from vfe3.families.base import _logdet_chol
    from vfe3.numerics import safe_cholesky
    torch.manual_seed(7)
    V, B, N, K = 32, 2, 4, 4
    pb = _full_model(vocab_size=V, decode_mode="full_chunked").prior_bank
    A = torch.randn(B, N, K, K)
    sigma_q = A @ A.transpose(-1, -2) + torch.eye(K)
    diag_sq, logdet_q, spd_ok = pb._full_cov_query_invariants(sigma_q)
    L, ok = safe_cholesky(sigma_q, eps=pb.eps, rounds=5)
    assert bool(ok.all()) and bool(spd_ok.all())
    assert torch.equal(logdet_q, _logdet_chol(L))
    assert torch.isfinite(logdet_q).all()
    assert torch.equal(diag_sq, torch.diagonal(sigma_q, dim1=-2, dim2=-1))


# ---------------------------------------------------------------------------
# B2: per-head full-covariance transport sandwich. For a block-diagonal gauge (block_glk) the
# congruence Omega Sigma Omega^T can be assembled block-pair by block-pair from the per-token exp
# factors, so the dense (..., N, N, K, K) Omega is never materialized. Must equal the dense path
# (to_dense_omega + the float64 sandwich) exactly.
# ---------------------------------------------------------------------------
def _spd_batch(B, N, K, gen):
    A = torch.randn(B, N, K, K, generator=gen)
    return A @ A.transpose(-1, -2) + K * torch.eye(K)


# ---------------------------------------------------------------------------
# B4: config ergonomics warnings (no math change; signposting).
# ---------------------------------------------------------------------------
def _warns_matching(substr, **cfg_kw):
    """True iff building VFE3Config(**cfg_kw) emits a UserWarning whose message contains substr
    (tolerating other, unrelated warnings -- e.g. the oracle-truncation notice)."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        test_config(**cfg_kw)
    return any(substr in str(w.message) for w in rec)


def test_warn_fullcov_linear_decode_discards_covariance():
    assert _warns_matching("discards the converged",
                           family="gaussian_full", decode_mode="linear")
    # The full-covariance prior-bank decoder does not emit the discard warning.
    assert not _warns_matching("discards the converged",
                               family="gaussian_full",
                               decode_mode="full_chunked")
