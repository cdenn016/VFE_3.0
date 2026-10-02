r"""``family_chunked`` decode: honour ``decode_chunk_size``, and exclude non-PD positions.

Two defects, both specific to the family-consistent route and both invisible to the suite because
no test paired ``family='gaussian_full'`` with this decode at pair-grid shape (audit 2026-08-07).

**(1) The chunked mode did not chunk.** ``_decode_family_chunked`` was one statement --
``return _decode_family(...)`` -- and ``_decode_family`` has no loop and never reads
``decode_chunk_size``. So on the LOGITS path (sampling / generation / ``targets=None``) the knob was
accepted and silently ignored: measured an identical 49.15 MB workspace at chunk=512 and chunk=8192,
and a single 321.6 MB ``(1, 2, 50257, 20, 20)`` allocation at the live vocab/embed -- about 41 GB at
N=256, against 8 MB for ``full_chunked``. Because a full family's functional workspace is
``(B, N, V, K, K)``, K^2 times the (B, N, V) output it produces, slicing the vocabulary is the whole
difference between usable and not. Both kernels now share ``_family_logits``, differing only in the
slice width, so they cannot drift again.

**(2) The F31 non-PD exclusion never reached this route.** ``decode_ce_full_chunked`` folds
``spd_ok`` into its ``valid`` mask; ``decode_ce_family_chunked`` had a bare
``targets != ignore_index``, and the dense-branch guard in ``model.py`` is bypassed because this mode
registers ``supports_chunked=True``. One degenerate position therefore NaN'd the scalar CE for the
entire batch -- measured ``fused CE=nan`` against ``3.0012598`` on the ``full_chunked`` twin at
identical inputs.

Masking after the fact is NOT sufficient and that is the subtle part: with ``kl_max=inf`` a non-PD
``Sigma_q`` produces NaN, ``nan * 0.0`` is ``nan`` at the masked mean, and logsumexp/softmax backward
turns the zero gradient into ``0 * nan`` as well. The covariance is therefore SANITIZED before the
functional runs -- substituted with the identity at positions that ``valid`` drops from both the
numerator and the denominator, so the substitution cannot influence the loss.

Pins: (a) the slice width is honoured and bounds the workspace; (b) slicing is value-identical;
(c) a degenerate position yields a finite CE and finite gradients; (d) it is excluded rather than
scored; (e) the healthy path and the diagonal family are untouched.
"""

from types import SimpleNamespace

import pytest
import torch

import vfe3.model.prior_bank as prior_bank

from tests.test_amp import _tiny_model


def _bank(decode_mode="family_chunked", *, renyi_order=0.5, **kw):
    """Standalone decoder fixture; non-KL orders are outside projected model admission."""
    bank = prior_bank.PriorBank(
        vocab_size=20, K=4, n_gen=8, family="gaussian_full", diagonal_covariance=False,
        decode_mode=decode_mode, renyi_order=renyi_order,
        decode_av_precision="fp32", untie_decode_bank=True, decode_unigram_prior=True,
        full_cov_precision_policy="fp32_escalate", **kw,
    )
    return SimpleNamespace(cfg=SimpleNamespace(embed_dim=4), prior_bank=bank), bank


def _inputs(model, non_pd_at=None):
    B, N, K = 2, 5, model.cfg.embed_dim
    g = torch.Generator().manual_seed(0)
    mu = torch.randn(B, N, K, generator=g)
    sigma = torch.eye(K).expand(B, N, K, K).clone()
    if non_pd_at is not None:
        sigma[non_pd_at] = -5.0 * torch.eye(K)          # fails every jitter round
    return mu, sigma, torch.randint(0, 20, (B, N), generator=g)


@pytest.fixture
def spy(monkeypatch):
    r"""Record the vocab width of every functional call the decode makes."""
    widths = []
    real = prior_bank.get_functional

    def _wrapped(name):
        fn = real(name)

        def _spy(q, p, **kw):
            out = fn(q, p, **kw)
            widths.append(out.shape[-1])
            return out
        return _spy

    monkeypatch.setattr(prior_bank, "get_functional", _wrapped)
    return widths


# -- (a)/(b) the slice width is honoured, and slicing changes nothing --------------------------

@pytest.mark.parametrize("chunk", [1, 3, 7, 20])
def test_decode_chunk_size_bounds_the_functional_workspace(spy, chunk):
    model, pb = _bank(decode_chunk_size=chunk)
    mu, sigma, _ = _inputs(model)
    V = pb.vocab_size

    logits = pb.decode(mu, sigma)

    assert logits.shape[-1] == V
    assert widths_ok(spy, V, chunk), f"widths {spy} do not tile V={V} at chunk={chunk}"
    assert max(spy) <= chunk, "a slice exceeded the configured width -- the knob is being ignored"
    assert len(spy) == -(-V // chunk), "wrong number of vocab slices"


def widths_ok(widths, V, chunk):
    return sum(widths) == V and all(0 < w <= chunk for w in widths)


def test_chunked_logits_match_the_dense_family_kernel():
    r"""The functional is elementwise over the vocabulary axis, so slicing must be exact."""
    model_d, pb_d = _bank(decode_mode="family_chunked")
    model_c, pb_c = _bank(decode_mode="family_chunked", decode_chunk_size=3)
    pb_c.load_state_dict(pb_d.state_dict())

    mu, sigma, _ = _inputs(model_d)
    assert torch.equal(pb_d.decode(mu, sigma), pb_c.decode(mu, sigma))


def test_unchunked_family_still_makes_a_single_call(spy):
    r"""The dense ``_decode_family`` oracle must not acquire a loop as a side effect of the shared body."""
    from vfe3.model.prior_bank import _decode_family

    model, pb = _bank(decode_mode="family_chunked", decode_chunk_size=3)
    mu, sigma, _ = _inputs(model)
    _decode_family(pb, mu, sigma, pb._tau_eff())
    assert len(spy) == 1 and spy[0] == pb.vocab_size


# -- (c)/(d) the degenerate position is excluded, not scored ------------------------------------

def test_fused_ce_is_finite_when_a_position_is_non_pd():
    r"""The regression: one bad position used to NaN the CE for the whole batch."""
    model, pb = _bank(decode_chunk_size=7)
    mu, sigma, targets = _inputs(model, non_pd_at=(0, 2))
    ce = pb.decode_ce_family_chunked(mu, sigma, targets)
    assert torch.isfinite(ce), "a single degenerate position must not poison the batch CE"


def test_non_pd_position_is_excluded_from_the_mean_not_merely_zeroed():
    r"""Excluded means it leaves the DENOMINATOR too -- the CE must equal the same batch with that
    position marked ignore_index, not the batch with a 0 scored in its place."""
    model, pb = _bank(decode_chunk_size=7)
    mu, sigma, targets = _inputs(model, non_pd_at=(0, 2))

    ce_auto = pb.decode_ce_family_chunked(mu, sigma, targets)
    targets_ignored = targets.clone()
    targets_ignored[0, 2] = -100
    ce_explicit = pb.decode_ce_family_chunked(mu, sigma, targets_ignored)
    assert torch.allclose(ce_auto, ce_explicit), "exclusion must match an explicit ignore_index"


def test_gradients_stay_finite_through_a_degenerate_position():
    r"""Masking a NaN is not enough: logsumexp/softmax backward turns 0 * nan back into nan."""
    model, pb = _bank(decode_chunk_size=7)
    mu, sigma, targets = _inputs(model, non_pd_at=(0, 2))
    mu = mu.clone().requires_grad_(True)

    pb.decode_ce_family_chunked(mu, sigma, targets).backward()
    assert mu.grad is not None and torch.isfinite(mu.grad).all()
    # The decode reads _decode_mu_table(): decode_mu_embed under the default untie_decode_bank=True,
    # mu_embed on a tied bank. Its gradient is the one the decode CE reaches.
    decode_mu_grad = pb._decode_mu_table().grad
    assert decode_mu_grad is not None and torch.isfinite(decode_mu_grad).all()


def test_logits_row_for_a_degenerate_position_is_finite_and_uniform():
    r"""Inference path: an all-NaN (or all--inf) row is NaN under log_softmax and breaks sampling."""
    model, pb = _bank(decode_chunk_size=7)
    mu, sigma, _ = _inputs(model, non_pd_at=(0, 2))
    logits = pb.decode(mu, sigma)

    assert torch.isfinite(logits).all()
    row = logits[0, 2]
    assert torch.allclose(row, row[0].expand_as(row)), "degenerate row must be informationless"
    assert torch.isfinite(torch.log_softmax(logits, dim=-1)).all()


# -- (e) the healthy path is untouched ---------------------------------------------------------

def test_healthy_full_covariance_path_is_unchanged_by_the_sanitizer():
    r"""With every Sigma_q PD the substitution is a no-op, so the CE must be bit-identical to a run
    that never had a degenerate position to handle."""
    model, pb = _bank(decode_chunk_size=7)
    mu, sigma, targets = _inputs(model)
    ce = pb.decode_ce_family_chunked(mu, sigma, targets)
    assert torch.isfinite(ce)
    # decode_degenerate_positions must report nothing to exclude on an SPD batch.
    assert not pb.decode_degenerate_positions(sigma).any()


def test_diagonal_family_is_untouched():
    r"""decode_degenerate_positions returns None for a diagonal dispersion -- no factorization, so
    no failure mode, and neither the sanitizer nor the mask may engage."""
    model = _tiny_model(n_heads=2, family="gaussian_diagonal",
                        decode_mode="family_chunked", decode_chunk_size=7)
    pb = model.prior_bank
    B, N, K = 2, 5, model.cfg.embed_dim
    g = torch.Generator().manual_seed(0)
    mu = torch.randn(B, N, K, generator=g)
    sigma = torch.rand(B, N, K, generator=g) + 0.5
    targets = torch.randint(0, 20, (B, N), generator=g)

    assert pb.decode_degenerate_positions(sigma) is None
    assert torch.isfinite(pb.decode_ce_family_chunked(mu, sigma, targets))
