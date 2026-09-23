r"""Full-covariance + alpha/divergence completion roadmap (2026-06-13 multi-expert investigation).

Groups, by roadmap item:
  A1  per-coordinate Bhattacharyya / Jeffreys (the two non-Renyi divergences that DO decompose
      coordinate-wise for a diagonal Gaussian); squared_hellinger stays rejected (1-exp(-D/2) is
      non-additive).
  A2  per-position state_dependent alpha is envelope-correct for EVERY divergence (the gradient is
      alpha*(D)*dD with no product-rule residue), the divergence-agnostic alpha-envelope.
  A3  the full-covariance unrolled oracle gradient (oracle_unroll_grad=True) is FINITE on a single
      training backward -- the _eigh_damped/safe_cholesky fix; the old "expect NaNs" docstring is stale.
  B1  fused chunked full-covariance decode+CE (decode_mode='full_chunked'): the diagonal-prior closed
      form, equal to the dense full decode CE, with no per-pair (B,N,V,K,K) Cholesky workspace.
  B2  per-head full-covariance transport sandwich: the block-diagonal congruence equals the dense
      Omega Sigma Omega^T without materializing the dense (B,N,N,K,K) Omega.
  B4  config ergonomics warnings (covariance discarded at the linear decode; bounded-divergence
      state_dependent alpha at b0=O(1)).
"""

import warnings

import pytest
import torch
import torch.nn.functional as F

from vfe3.config import VFE3Config
from vfe3.families.base import get_family
from vfe3.model.model import VFEModel


# ---------------------------------------------------------------------------
# A1: per-coordinate Bhattacharyya / Jeffreys
# ---------------------------------------------------------------------------
def test_per_coord_bhattacharyya_and_jeffreys_construct_squared_hellinger_still_rejected():
    """state_dependent_per_coord now accepts the divergences that DECOMPOSE coordinate-wise
    (bhattacharyya = 0.5 D_1/2, jeffreys = KL + KL_rev), and still rejects squared_hellinger
    (H^2 = 1 - exp(-D_1/2/2) is a nonlinear transform of the SUMMED divergence)."""
    VFE3Config(lambda_alpha_mode="state_dependent_per_coord", divergence_family="bhattacharyya")
    VFE3Config(lambda_alpha_mode="state_dependent_per_coord", divergence_family="jeffreys")
    VFE3Config(lambda_alpha_mode="state_dependent_per_coord", divergence_family="renyi")
    with pytest.raises(ValueError):
        VFE3Config(lambda_alpha_mode="state_dependent_per_coord", divergence_family="squared_hellinger")


def test_per_coord_bhattacharyya_jeffreys_sum_to_scalar():
    """The per-coordinate form summed over k recovers the scalar divergence (kl_max=inf so no
    per-coordinate clamp diverges from the single summed clamp)."""
    from vfe3.divergence import (
        bhattacharyya, jeffreys,
        bhattacharyya_per_coord, jeffreys_per_coord,
    )
    g = torch.Generator().manual_seed(0)
    diag = get_family("gaussian_diagonal")
    mu_q = torch.randn(4, 3, generator=g)
    sigma_q = torch.rand(4, 3, generator=g) + 0.5
    mu_p = torch.randn(4, 3, generator=g)
    sigma_p = torch.rand(4, 3, generator=g) + 0.5
    q = diag(mu_q, sigma_q)
    p = diag(mu_p, sigma_p)

    bc_pc = bhattacharyya_per_coord(q, p, kl_max=float("inf"))    # (4, 3)
    bc = bhattacharyya(q, p, kl_max=float("inf"))                 # (4,)
    assert bc_pc.shape == (4, 3)
    assert torch.allclose(bc_pc.sum(-1), bc, atol=1e-5)

    jf_pc = jeffreys_per_coord(q, p, kl_max=float("inf"))
    jf = jeffreys(q, p, kl_max=float("inf"))
    assert torch.allclose(jf_pc.sum(-1), jf, atol=1e-5)


def test_per_coord_bhattacharyya_forward_backward_runs():
    """A model with state_dependent_per_coord + bhattacharyya runs forward+backward (routes to the
    autograd oracle; the per-coordinate self-divergence shapes a per-coordinate alpha)."""
    cfg = VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1,
                     n_e_steps=2, e_q_mu_lr=0.05, e_phi_lr=0.0,
                     lambda_alpha_mode="state_dependent_per_coord", divergence_family="bhattacharyya")
    model = VFEModel(cfg)
    tokens = torch.randint(0, 20, (2, 5)); targets = torch.randint(0, 20, (2, 5))
    _, loss, _ = model(tokens, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.prior_bank.mu_embed.grad is not None
    assert model.prior_bank.mu_embed.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# A3: full-cov unrolled oracle gradient is FINITE (and live) on a single training backward.
# The config docstring's "expect NaNs there" for gaussian_full + oracle_unroll_grad=True predates
# the _eigh_damped gap-regularized eigh backward (retraction.py); the single training backward is
# now finite. This pins that, and contrasts it with the truncated default (oracle_unroll_grad=False).
# ---------------------------------------------------------------------------
def _fullcov_model(renyi_order=1.0, oracle_unroll_grad=True, **kw):
    # A noncanonical Renyi order uses the family-consistent chunked decoder; canonical order 1
    # keeps the fast full-covariance chunked kernel.
    decode_mode = kw.pop("decode_mode", "full_chunked" if renyi_order == 1.0 else "family_chunked")
    cfg = VFE3Config(vocab_size=24, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1,
                     n_e_steps=2, e_q_mu_lr=0.05, e_q_sigma_lr=0.02, e_phi_lr=0.01,
                     family="gaussian_full", decode_mode=decode_mode,
                     pos_phi="learned",
                     renyi_order=renyi_order, e_step_gradient="unroll",
                     oracle_unroll_grad=oracle_unroll_grad, **kw)
    return VFEModel(cfg)


@pytest.mark.parametrize("renyi_order", [0.5, 1.0])
def test_fullcov_unroll_oracle_grad_is_finite(renyi_order):
    """gaussian_full + e_step_gradient='unroll' + oracle_unroll_grad=True: the single training
    backward is finite across Renyi orders (the _eigh_damped/safe_cholesky fix; the old
    'expect NaNs' docstring is stale for the single-backward training path)."""
    torch.manual_seed(0)
    model = _fullcov_model(renyi_order=renyi_order)
    tokens = torch.randint(0, 24, (2, 5)); targets = torch.randint(0, 24, (2, 5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                  # alpha>1 convex-regime + oracle notices
        _, loss, _ = model(tokens, targets)
        loss.backward()
    assert torch.isfinite(loss)
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {name} at renyi_order={renyi_order}"


def test_fullcov_unroll_oracle_grad_reaches_gauge_frame_only_when_enabled():
    """The through-inference signal to the gauge-frame table phi_embed is LIVE under
    oracle_unroll_grad=True and TRUNCATED (None/zero) under the default False -- the documented
    remedy actually works for full covariance."""
    torch.manual_seed(1)
    tokens = torch.randint(0, 24, (2, 5)); targets = torch.randint(0, 24, (2, 5))

    on = _fullcov_model(oracle_unroll_grad=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, loss_on, _ = on(tokens, targets)
        loss_on.backward()
    g_on = on.prior_bank.phi_embed.grad
    assert g_on is not None and torch.isfinite(g_on).all() and g_on.abs().sum() > 0

    off = _fullcov_model(oracle_unroll_grad=False)
    off.load_state_dict(on.state_dict())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, loss_off, _ = off(tokens, targets)
        loss_off.backward()
    g_off = off.prior_bank.phi_embed.grad
    # truncated: the E-step's through-inference signal to phi_embed never flows on the detached oracle
    assert g_off is None or g_off.abs().sum() == 0


# ---------------------------------------------------------------------------
# A2: the alpha-envelope is divergence-AGNOSTIC. For state_dependent alpha*(D) = c0/(b0+D) with
# R(alpha) = b0 alpha - c0 log alpha, d/dmu[alpha*(D) D + R(alpha*(D))] = alpha*(D) dD/dmu exactly,
# for ANY differentiable self-divergence D -- so state_dependent + a non-Renyi divergence is
# mathematically correct (the user's "doesn't work with non-renyi" is the per-COORD config lock,
# A1, not a wrong per-position gradient).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("div_name", ["renyi", "squared_hellinger", "bhattacharyya", "jeffreys"])
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


@pytest.mark.parametrize("div_name", ["squared_hellinger", "bhattacharyya", "jeffreys"])
def test_state_dependent_per_position_nonrenyi_forward_backward(div_name):
    """Per-position state_dependent + a non-Renyi divergence constructs and trains (routes to the
    oracle); the prior tables get a finite gradient."""
    cfg = VFE3Config(vocab_size=20, embed_dim=4, n_heads=2, max_seq_len=5, n_layers=1,
                     n_e_steps=2, e_q_mu_lr=0.05, e_phi_lr=0.0,
                     lambda_alpha_mode="state_dependent", divergence_family=div_name, b0=0.3)
    model = VFEModel(cfg)
    tokens = torch.randint(0, 20, (2, 5)); targets = torch.randint(0, 20, (2, 5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, loss, _ = model(tokens, targets)
        loss.backward()
    assert torch.isfinite(loss)
    assert model.prior_bank.mu_embed.grad is not None
    assert torch.isfinite(model.prior_bank.mu_embed.grad).all()


# ---------------------------------------------------------------------------
# B1: fused chunked full-covariance decode+CE (decode_mode='full_chunked'). The prior table is
# DIAGONAL (diag_embed of sigma_log_embed), so KL(q_full || pi_v_diag) has a closed form needing
# NO per-pair (K,K) Cholesky -- one logdet(Sigma_q) per position, then matmuls over V. It must equal
# the dense full decode CE (F.cross_entropy(_decode_full(...))) to atol-1e-3, without materializing
# (B,N,V,K,K).
# ---------------------------------------------------------------------------
def _full_model(vocab_size=64, **kw):
    decode_mode = kw.pop("decode_mode", "full_chunked")
    cfg = VFE3Config(vocab_size=vocab_size, embed_dim=4, n_heads=2, max_seq_len=6, n_layers=1,
                     n_e_steps=2, e_q_mu_lr=0.05, e_q_sigma_lr=0.02, e_phi_lr=0.0,
                     family="gaussian_full", decode_mode=decode_mode,
                     _allow_legacy_dense_decode=decode_mode == "full", **kw)
    return VFEModel(cfg)


def _full_converged(model):
    from vfe3.model.stack import vfe_stack

    def run(tokens):
        N = tokens.shape[1]
        beliefs = model.prior_bank.encode(tokens)
        log_prior = model._attention_log_prior(N, tokens.device)
        out = vfe_stack(beliefs, beliefs.mu, beliefs.sigma, model.group, model.cfg,
                        log_prior=log_prior, block_norm=model.block_norm)
        return out.mu.float(), out.sigma.float()
    return run


def test_full_chunked_config_accepts_with_full_family_and_rejects_diagonal():
    VFE3Config(family="gaussian_full",
               decode_mode="full_chunked")
    with pytest.raises(ValueError):
        VFE3Config(family="gaussian_diagonal",
                   decode_mode="full_chunked")


def test_full_chunked_ce_matches_dense_full_ce_multiple_chunk_sizes():
    torch.manual_seed(0)
    V = 64
    tokens = torch.randint(0, V, (3, 6)); targets = torch.randint(0, V, (3, 6))
    ref = _full_model(vocab_size=V, decode_mode="full")
    mu, sigma = _full_converged(ref)(tokens)
    logits = ref.prior_bank.decode(mu, sigma)                          # (B,N,V) dense full
    full_ce = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), ignore_index=-100)
    for chunk in (16, 8192, V, 7, 100):
        ch = _full_model(vocab_size=V, decode_mode="full_chunked", decode_chunk_size=chunk)
        ch.load_state_dict(ref.state_dict())
        mu_c, sigma_c = _full_converged(ch)(tokens)
        ce = ch.prior_bank.decode_ce_full_chunked(mu_c, sigma_c, targets)
        assert torch.allclose(ce, full_ce, atol=1e-3), f"chunk={chunk}: {ce.item()} != {full_ce.item()}"


def test_full_chunked_honors_ignore_index():
    torch.manual_seed(1)
    V = 50
    tokens = torch.randint(0, V, (2, 6)); targets = torch.randint(0, V, (2, 6))
    targets[0, 0] = -100; targets[1, 3] = -100
    ref = _full_model(vocab_size=V, decode_mode="full")
    mu, sigma = _full_converged(ref)(tokens)
    full_ce = F.cross_entropy(ref.prior_bank.decode(mu, sigma).reshape(-1, V),
                              targets.reshape(-1), ignore_index=-100)
    ch = _full_model(vocab_size=V, decode_mode="full_chunked", decode_chunk_size=13)
    ch.load_state_dict(ref.state_dict())
    mu_c, sigma_c = _full_converged(ch)(tokens)
    ce = ch.prior_bank.decode_ce_full_chunked(mu_c, sigma_c, targets)
    assert torch.allclose(ce, full_ce, atol=1e-3)


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
    ref = _full_model(vocab_size=V, decode_mode="full")
    logits_full = ref(tokens)
    ch = _full_model(vocab_size=V, decode_mode="full_chunked")
    ch.load_state_dict(ref.state_dict())
    logits_ch = ch(tokens)
    assert logits_ch.shape == (2, 5, V)
    assert torch.allclose(logits_ch, logits_full, atol=1e-3)


def test_full_chunked_forward_loss_matches_full_forward_loss():
    torch.manual_seed(4)
    V = 40
    tokens = torch.randint(0, V, (3, 6)); targets = torch.randint(0, V, (3, 6))
    base = _full_model(vocab_size=V, decode_mode="full")
    _, loss_full, ce_full = base(tokens, targets)
    ch = _full_model(vocab_size=V, decode_mode="full_chunked", decode_chunk_size=9)
    ch.load_state_dict(base.state_dict())
    logits_ch, loss_ch, ce_ch = ch(tokens, targets)
    assert logits_ch is None                               # fused path forms no (B,N,V) logits
    assert torch.allclose(loss_ch, loss_full, atol=1e-3)
    assert torch.allclose(ce_ch, ce_full, atol=1e-3)


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
    pb = _full_model(vocab_size=V, decode_mode="full").prior_bank
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
    assert torch.allclose(chunked[good], dense[good], atol=1e-3)
    assert pb.decode_degenerate_positions(sigma_q)[good].sum() == 0


def test_full_cov_query_invariants_all_pd_byte_identical():
    """On an all-PD Sigma_q the ok mask is all-True and torch.where selects logdet_q unchanged:
    the invariants stay byte-identical to the raw round-zero safe_cholesky log-det."""
    from vfe3.families.base import _logdet_chol
    from vfe3.numerics import safe_cholesky
    torch.manual_seed(7)
    V, B, N, K = 32, 2, 4, 4
    pb = _full_model(vocab_size=V, decode_mode="full").prior_bank
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


def test_factored_full_cov_sandwich_equals_dense():
    from vfe3.geometry.groups import get_group
    from vfe3.geometry.transport import (
        build_factored_transport, transport_covariance, _factored_full_covariance,
    )
    g = torch.Generator().manual_seed(0)
    K, H, N, B = 6, 3, 4, 2                                 # block_glk: 3 heads of d=2
    grp = get_group("block_glk")(K, H)
    n_gen = grp.generators.shape[0]
    phi = 0.2 * torch.randn(B, N, n_gen, generator=g)
    factored = build_factored_transport(phi, grp)
    sigma = _spd_batch(B, N, K, g)                          # (B, N, K, K) genuinely non-diagonal SPD

    out_fac = _factored_full_covariance(factored, sigma)    # (B, N, N, K, K)
    dense = transport_covariance(factored.to_dense_omega(), sigma)
    assert out_fac.shape == (B, N, N, K, K)
    assert torch.allclose(out_fac, dense, atol=1e-5), (out_fac - dense).abs().max().item()


def test_transport_covariance_routes_full_factored_to_per_head_and_matches_dense():
    from vfe3.geometry.groups import get_group
    from vfe3.geometry.transport import build_factored_transport, transport_covariance
    g = torch.Generator().manual_seed(1)
    K, H, N, B = 4, 2, 5, 2
    grp = get_group("block_glk")(K, H)
    n_gen = grp.generators.shape[0]
    phi = 0.15 * torch.randn(B, N, n_gen, generator=g)
    factored = build_factored_transport(phi, grp)
    sigma = _spd_batch(B, N, K, g)
    via_factored = transport_covariance(factored, sigma)            # routed (per-head)
    via_dense = transport_covariance(factored.to_dense_omega(), sigma)
    assert torch.allclose(via_factored, via_dense, atol=1e-5)


# ---------------------------------------------------------------------------
# B4: config ergonomics warnings (no math change; signposting).
# ---------------------------------------------------------------------------
def _warns_matching(substr, **cfg_kw):
    """True iff building VFE3Config(**cfg_kw) emits a UserWarning whose message contains substr
    (tolerating other, unrelated warnings -- e.g. the oracle-truncation notice)."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        VFE3Config(**cfg_kw)
    return any(substr in str(w.message) for w in rec)


def test_warn_fullcov_linear_decode_discards_covariance():
    assert _warns_matching("discards the converged",
                           family="gaussian_full", decode_mode="linear")
    # The full-covariance prior-bank decoder does not emit the discard warning.
    assert not _warns_matching("discards the converged",
                               family="gaussian_full",
                               decode_mode="full_chunked")


def test_warn_state_dependent_bounded_divergence_b0():
    assert _warns_matching("nearly constant",
                           lambda_alpha_mode="state_dependent", divergence_family="squared_hellinger", b0=1.0)
    # A small b0 (restoring a wide alpha* range) does NOT warn about the range.
    assert not _warns_matching("nearly constant",
                               lambda_alpha_mode="state_dependent", divergence_family="squared_hellinger", b0=0.05)
    # An unbounded divergence (renyi/KL) does NOT warn at b0=1.
    assert not _warns_matching("nearly constant",
                               lambda_alpha_mode="state_dependent", divergence_family="renyi", b0=1.0)
