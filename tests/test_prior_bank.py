from tests.profiles import test_config
import pytest
import torch
import torch.nn.functional as F
from vfe3.belief import BeliefState
from vfe3.divergence import get_family, get_functional
from vfe3.model.prior_bank import (
    DecodeRegistration,
    PriorBank,
    _DECODERS,
    get_decode,
    get_decode_registration,
    register_decode,
)
from vfe3.numerics import bounded_variance_from_log


def test_encode_shapes_and_positive_sigma():
    V, K, n_gen = 20, 4, 16
    pb = PriorBank(V, K, n_gen)
    tokens = torch.randint(0, V, (2, 5))
    b = pb.encode(tokens)
    assert isinstance(b, BeliefState)
    assert b.mu.shape == (2, 5, K) and b.sigma.shape == (2, 5, K) and b.phi.shape == (2, 5, n_gen)
    assert (b.sigma > 0).all()


def test_encode_is_a_lookup():
    V, K, n_gen = 6, 3, 9
    pb = PriorBank(V, K, n_gen)
    b = pb.encode(torch.tensor([[0, 0]]))
    assert torch.allclose(b.mu[0, 0], b.mu[0, 1])             # same token -> same prior


def test_decode_matches_divergence_seam_exactly():
    rng = torch.Generator().manual_seed(0)
    V, K, n_gen = 12, 4, 16
    pb = PriorBank(V, K, n_gen)
    mu_q = torch.randn(2, 3, K, generator=rng); sigma_q = torch.rand(2, 3, K, generator=rng) + 0.5
    logits = pb.decode(mu_q, sigma_q)
    # decode_log_scale=0 -> tau_eff=decode_tau
    ref = pb.reference_decode(mu_q, sigma_q, tau=pb.decode_tau)
    assert torch.allclose(logits, ref, atol=1e-3)             # EXACT -KL/tau (per-position term kept)
    # shift-invariant pin (robust to a dropped-constant variant):
    assert torch.allclose(F.log_softmax(logits, dim=-1), F.log_softmax(ref, dim=-1), atol=1e-4)


@pytest.mark.parametrize(
    ("untie_decode_bank",),
    [
        (False,),
        (True,),
    ],
    ids=["tied-token", "untied-token"],
)
def test_decode_matches_reference_across_table_routes(
    untie_decode_bank: bool,
) -> None:
    V, K, n_gen = 5, 3, 9
    pb = PriorBank(
        V,
        K,
        n_gen,
        decode_tau=1.7,
        untie_decode_bank=untie_decode_bank,
    )
    with torch.no_grad():
        pb._decode_mu_table().copy_(torch.linspace(-0.75, 0.75, V * K).reshape(V, K))
        pb._decode_sigma_log_table().copy_(
            torch.tensor([-20.0, -0.25, 81.0, -20.0, 0.5]).unsqueeze(-1).expand(V, K)
        )

    mu_q = torch.tensor([[[0.2, -0.3, 0.4]]])
    sigma_q = torch.tensor([[[0.8, 1.1, 0.6]]])
    with pytest.warns(RuntimeWarning, match="max_log=80"):
        logits = pb.decode(mu_q, sigma_q, tau=1.7)
    with pytest.warns(RuntimeWarning, match="max_log=80"):
        ref = pb.reference_decode(mu_q, sigma_q, tau=1.7)
    # The sigma_log=-20 rows put |logit| near 1.3e6, where one float32 ulp is 0.125, so an absolute
    # 1e-3 alone would demand bitwise-identical rounding from two different evaluation orders
    # (decode's closed form vs the family seam; each lands within 1-2 ulp of the fp64 value, at
    # different entries). Allow a few ulps relative on top of the 1e-3 used for O(1) entries: a wrong
    # table route moves these logits by O(1)..O(1e6), far outside this bound.
    few_ulps = 4 * torch.finfo(torch.float32).eps
    torch.testing.assert_close(logits, ref, atol=1e-3, rtol=few_ulps)


def test_decode_tau_scaling():
    rng = torch.Generator().manual_seed(1)
    V, K = 10, 3
    pb = PriorBank(V, K, 9)
    mu_q = torch.randn(1, 2, K, generator=rng); sigma_q = torch.rand(1, 2, K, generator=rng) + 0.5
    l1 = pb.decode(mu_q, sigma_q, tau=1.0)
    l2 = pb.decode(mu_q, sigma_q, tau=2.0)
    assert torch.allclose(l1, 2.0 * l2, atol=1e-3)            # logits ~ 1/tau


def test_decode_matches_seam_in_large_kl_regime():
    # Regression for the clamp-asymmetry defect: tight priors + separated means drive
    # KL >> 100, where the seam's default kl_max=100 would saturate to a flat -100 and
    # destroy the ranking. Both decode paths use kl_max=inf, so they must still agree
    # EXACTLY and predict the SAME token (argmax preserved, no flattening).
    torch.manual_seed(1)
    V, K, n_gen = 6, 4, 16
    pb = PriorBank(V, K, n_gen)
    with torch.no_grad():
        pb.sigma_log_embed.fill_(-4.0)                       # sigma_v = exp(-4) ~ 0.018 (tight)
        pb.mu_embed.normal_(0.0, 1.0)
    mu_q = 5.0 * torch.ones(1, 1, K); sigma_q = torch.ones(1, 1, K)
    logits = pb.decode(mu_q, sigma_q)
    ref = pb.reference_decode(mu_q, sigma_q, tau=pb.decode_tau)
    implied_kl = (-logits * pb.decode_tau)
    assert implied_kl.max().item() > 100.0                   # genuinely past the old clamp
    assert torch.allclose(logits, ref, atol=1e-3)            # EXACT pin holds in the clamped regime
    assert torch.allclose(                                   # shift-invariant pin holds
        F.log_softmax(logits, dim=-1), F.log_softmax(ref, dim=-1), atol=1e-4
    )
    assert logits.argmax(-1).item() == ref.argmax(-1).item()  # same predicted token (no flattening)


def test_decode_exact_at_large_mean_offset():
    # Regression for the catastrophic-cancellation defect: a large common offset on the
    # means makes the expanded-square matmul subtract large near-equal quantities. The
    # mean-centered fused kernel must still match the seam to atol 1e-3 (the un-centered
    # version exceeded it near offset ~100 and grew ~mu^2 thereafter).
    torch.manual_seed(0)
    V, K, n_gen = 8, 4, 16
    pb = PriorBank(V, K, n_gen)
    with torch.no_grad():
        pb.mu_embed.normal_(0.0, 0.1).add_(1000.0)           # means clustered far from zero
    mu_q = (1000.0 + 0.1) * torch.ones(1, 1, K); sigma_q = torch.ones(1, 1, K)
    logits = pb.decode(mu_q, sigma_q)
    ref = pb.reference_decode(mu_q, sigma_q, tau=pb.decode_tau)
    assert torch.allclose(logits, ref, atol=1e-3)
    assert torch.allclose(
        F.log_softmax(logits, dim=-1), F.log_softmax(ref, dim=-1), atol=1e-4
    )


# ===========================================================================
# PB-14 (Task 5, 2026-07-12): family/divergence-consistent prior-bank decode.
#
# The generic `family`/`family_chunked` decode modes score logits = -D_configured(q||p_v)/
# tau_eff through the CONFIGURED belief family and divergence functional, with NO kl_max ranking
# clamp, so the readout matches the E-step geometry instead of a hardcoded gaussian alpha=1 KL.
# The DecodeRegistration carries a resolved covariance_kinds set (backward-compatible with the
# legacy supports_full bit) and a family_consistent flag the config validates against.
# ===========================================================================


def _family_reference_logits(pb, mu_q, sigma_q, tau_eff):
    r"""Direct family-functional decode logits (the golden the registered kernel must reproduce)."""
    family_cls = get_family(pb.family)
    q_sigma = sigma_q.unsqueeze(-3 if family_cls.cov_kind == "full" else -2)
    q = family_cls(mu_q.unsqueeze(-2), q_sigma)
    p_sigma_diag = bounded_variance_from_log(pb._decode_sigma_log_table(), eps=pb.eps)
    p_sigma = (torch.diag_embed(p_sigma_diag) if family_cls.cov_kind == "full"
               else p_sigma_diag)
    p = family_cls(pb._decode_mu_table(), p_sigma)
    functional = get_functional(pb.divergence_family)
    energy = functional(q, p, alpha=pb.renyi_order, kl_max=float("inf"), eps=pb.eps)
    return -energy / tau_eff


# --- dense reference: diagonal Gaussian at alpha 0.5 / 1.0 / 1.5 ----------------------------

@pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5])
def test_family_decode_matches_direct_functional_diagonal_gaussian(alpha):
    torch.manual_seed(0)
    V, K, n_gen = 7, 3, 4
    pb = PriorBank(V, K, n_gen, decode_tau=1.3, family="gaussian_diagonal",
                   divergence_family="renyi", renyi_order=alpha, decode_mode="family")
    with torch.no_grad():
        pb.mu_embed.normal_(0.0, 0.7)
        pb.sigma_log_embed.normal_(0.0, 0.4)
    mu_q = torch.randn(2, 4, K); sigma_q = torch.rand(2, 4, K) + 0.3
    tau_eff = pb._tau_eff()
    got = get_decode("family")(pb, mu_q, sigma_q, tau_eff)
    exp = _family_reference_logits(pb, mu_q, sigma_q, tau_eff)
    assert got.shape == (2, 4, V)
    assert torch.allclose(got, exp, atol=1e-5, rtol=0.0)
    if alpha == 1.0:                                             # canonical: matches the fast diagonal kernel
        diag = get_decode("diagonal")(pb, mu_q, sigma_q, tau_eff)
        assert torch.allclose(got, diag, atol=1e-4)


def test_noncanonical_family_decode_ranking_differs_from_gaussian_kl():
    # A Renyi-0.5 decode over a table with a WIDE spread of prior variances ranks the vocabulary
    # differently from the alpha=1 KL readout (the alpha blend reweights the tight-vs-loose priors).
    torch.manual_seed(1)
    V, K, n_gen = 7, 3, 4
    pb_kl = PriorBank(V, K, n_gen, family="gaussian_diagonal", renyi_order=1.0, decode_mode="family")
    pb_renyi = PriorBank(V, K, n_gen, family="gaussian_diagonal", renyi_order=0.5, decode_mode="family")
    with torch.no_grad():
        mu = torch.randn(V, K)
        sig_log = torch.linspace(-3.0, 3.0, V).unsqueeze(-1).expand(V, K).contiguous()
        for pb in (pb_kl, pb_renyi):
            pb.mu_embed.copy_(mu)
            pb.sigma_log_embed.copy_(sig_log)
    mu_q = torch.randn(1, 6, K); sigma_q = torch.rand(1, 6, K) + 0.3
    tau_eff = pb_kl._tau_eff()
    kl_logits = get_decode("family")(pb_kl, mu_q, sigma_q, tau_eff)
    renyi_logits = get_decode("family")(pb_renyi, mu_q, sigma_q, tau_eff)
    assert not torch.equal(kl_logits.argmax(-1), renyi_logits.argmax(-1))   # ranking genuinely flips


# --- reference_decode dispatches through the configured family/divergence -------------------

@pytest.mark.parametrize("alpha", [0.5, 1.0])
def test_reference_decode_dispatches_configured_divergence(alpha):
    torch.manual_seed(2)
    V, K, n_gen = 7, 3, 4
    pb = PriorBank(V, K, n_gen, decode_tau=1.1, family="gaussian_diagonal",
                   renyi_order=alpha, decode_mode="family")
    with torch.no_grad():
        pb.mu_embed.normal_(0.0, 0.5)
    mu_q = torch.randn(2, 3, K); sigma_q = torch.rand(2, 3, K) + 0.3
    ref = pb.reference_decode(mu_q, sigma_q, tau=pb.decode_tau)
    exp = _family_reference_logits(pb, mu_q, sigma_q, pb._tau_eff(pb.decode_tau))
    assert torch.allclose(ref, exp, atol=1e-5, rtol=0.0)


# --- covariance-kind membership: both diagonal and full configs construct --------------------

def test_family_decode_config_accepts_diagonal_and_full_by_membership():
    dcfg = test_config(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=5,
                      family="gaussian_diagonal", decode_mode="family_chunked")
    assert dcfg.decode_mode == "family_chunked"
    fcfg = test_config(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=5,
                      family="gaussian_full", decode_mode="family_chunked")
    assert fcfg.decode_mode == "family_chunked"
    # the family_chunked rank also accepts both kinds.
    test_config(vocab_size=6, embed_dim=4, n_heads=2, max_seq_len=5,
               family="gaussian_full", decode_mode="family_chunked")


# --- PriorBank constructor stays source-compatible (no new args required) --------------------

def test_prior_bank_family_metadata_defaults_and_state_dict_unchanged():
    torch.manual_seed(0)
    pb = PriorBank(6, 3, 9)                                      # NO family/divergence/order args
    assert pb.family == "gaussian_diagonal"
    assert pb.divergence_family == "renyi"
    assert pb.renyi_order == 1.0
    # the three new attributes are plain fields (not params/buffers): the state dict is unchanged.
    assert set(pb.state_dict().keys()) == {
        "mu_embed", "sigma_log_embed", "phi_embed", "decode_log_scale",
    }
    # every existing test-style constructor keeps working unchanged.
    assert PriorBank(6, 3, 9, decode_mode="linear").divergence_family == "renyi"
    assert PriorBank(6, 3, 9, decode_tau=1.7).renyi_order == 1.0


# --- DecodeRegistration metadata resolution --------------------------------------------------

def test_register_decode_resolves_covariance_kinds_from_supports_full():
    def _fn(pb, mu_q, sigma_q, tau_eff):
        return mu_q.new_zeros(mu_q.shape[:-1] + (pb.vocab_size,))
    names = []
    try:
        register_decode("_pb14_omitted")(_fn); names.append("_pb14_omitted")
        r = get_decode_registration("_pb14_omitted")
        assert r.supports_full is False
        assert r.covariance_kinds == frozenset({"diagonal"})
        assert r.family_consistent is False

        register_decode("_pb14_false", supports_full=False)(_fn); names.append("_pb14_false")
        r = get_decode_registration("_pb14_false")
        assert r.supports_full is False and r.covariance_kinds == frozenset({"diagonal"})

        register_decode("_pb14_true", supports_full=True)(_fn); names.append("_pb14_true")
        r = get_decode_registration("_pb14_true")
        assert r.supports_full is True and r.covariance_kinds == frozenset({"full"})

        register_decode("_pb14_dual", covariance_kinds=frozenset({"diagonal", "full"}),
                        family_consistent=True)(_fn); names.append("_pb14_dual")
        r = get_decode_registration("_pb14_dual")
        assert r.covariance_kinds == frozenset({"diagonal", "full"})
        assert r.supports_full is True                          # derived from membership
        assert r.family_consistent is True

        with pytest.raises(ValueError):                         # contradictory dual metadata
            register_decode("_pb14_bad", supports_full=False,
                            covariance_kinds=frozenset({"diagonal", "full"}))(_fn)
    finally:
        for n in names:
            _DECODERS.pop(n, None)
        _DECODERS.pop("_pb14_bad", None)


def test_direct_decode_registration_is_backward_compatible():
    def _fn(pb, mu_q, sigma_q, tau_eff):
        return mu_q
    reg_full = DecodeRegistration(_fn, True, False, None)        # legacy positional construction
    assert reg_full.supports_full is True
    assert reg_full.covariance_kinds == frozenset({"full"})
    assert reg_full.family_consistent is False
    reg_diag = DecodeRegistration(_fn, False, False, None)
    assert reg_diag.covariance_kinds == frozenset({"diagonal"})


def test_decode_registration_rejects_noncallable_fused_ce_direct_and_decorator():
    def _fn(pb, mu_q, sigma_q, tau_eff):
        return mu_q

    with pytest.raises(TypeError, match="fused_ce.*callable"):
        DecodeRegistration(
            _fn,
            supports_full=False,
            supports_chunked=True,
            fused_ce=object(),
        )

    name = "_pb_task2_noncallable_fused"
    try:
        with pytest.raises(TypeError, match="fused_ce.*callable"):
            register_decode(
                name,
                supports_full=False,
                supports_chunked=True,
                fused_ce=object(),
            )(_fn)
    finally:
        _DECODERS.pop(name, None)


def test_registered_family_modes_are_family_consistent_dual_rank():
    for name in ("family", "family_chunked"):
        r = get_decode_registration(name)
        assert r.family_consistent is True
        assert r.covariance_kinds == frozenset({"diagonal", "full"})
    assert get_decode_registration("family_chunked").supports_chunked is True
    # the fast canonical kernels stay single-rank and NOT family-consistent.
    assert get_decode_registration("diagonal").covariance_kinds == frozenset({"diagonal"})
    assert get_decode_registration("full").covariance_kinds == frozenset({"full"})
    assert get_decode_registration("diagonal").family_consistent is False
    assert get_decode_registration("full").family_consistent is False
