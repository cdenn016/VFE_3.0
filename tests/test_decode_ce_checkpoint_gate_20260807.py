r"""decode_ce_checkpoint (config.py): gate the per-chunk gradient checkpoint every
decode_ce_*_chunked fused-CE kernel wraps its (B, N, chunk_width) reduction in
(vfe3/model/prior_bank.py's ``_chunk_summaries`` closures).

Checkpointing is a pure memory/compute reassociation: it changes WHEN the chunk's forward runs
(inside backward, via recompute) but must never change the value it computes or any gradient. The
correctness bar throughout is therefore bit-identical (``torch.equal``), not ``allclose`` -- the
recomputed closure is a deterministic (RNG-free), pure function of the SAME tensors the original
forward saw, so a difference here would be a genuine checkpoint-wiring bug, not floating-point
tolerance noise.

Primary target: ``decode_ce_full_chunked`` (``decode_mode='full_chunked'``, the live
``train_vfe3.py`` config, measured at 8-13%% of the training step at reference scale entirely from
an unconditional recompute). A handful of tests also exercise
``decode_ce_diagonal_chunked``/``decode_ce_linear_chunked``/``decode_ce_expected_likelihood_chunked``/
``decode_ce_family_chunked``: a grep for ``_checkpoint.checkpoint`` across ``vfe3/`` turned up the
SAME unconditional ``if torch.is_grad_enabled() and ...requires_grad:`` pattern at all five call
sites (prior_bank.py's only five checkpoint uses), so the fix -- a shared
``_decode_ce_should_checkpoint`` dispatch reading ``pb.decode_ce_checkpoint`` -- was applied to
all five for consistency, not just the dispatched target.
"""

from tests.profiles import test_config
import warnings

import pytest
import torch

import vfe3.model.prior_bank as prior_bank_module
from vfe3.config import ConfigNotice
from vfe3.model.prior_bank import (
    DECODE_CE_CHECKPOINT_AUTO_BYTES,
    PriorBank,
    _decode_ce_chunk_activation_bytes,
    _decode_ce_should_checkpoint,
)

V, K, N_GEN = 50, 6, 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _full_bank(seed=0, **kw):
    """Seeded full-covariance PriorBank routed to decode_mode='full_chunked' (the dispatched
    target). Constructed directly (bypassing VFE3Config.__post_init__'s family/decode_mode
    rank cross-check), matching the existing test_tier12_decode.py / test_fullcov_alpha_roadmap
    convention for exercising a PriorBank kernel in isolation.
    """
    torch.manual_seed(seed)
    return PriorBank(V, K, N_GEN, mu_init_std=0.4, diagonal_covariance=False,
                     decode_mode="full_chunked", **kw)


def _spd_leaf(B, N, k, seed, dtype=torch.float32):
    """A genuinely full (off-diagonal) SPD (B, N, K, K) leaf: A @ A^T + 0.5*I is PD for any A, so
    safe_cholesky succeeds without a jitter fallback and the leaf is an independent requires_grad
    tensor (not a view through diag_embed)."""
    torch.manual_seed(seed)
    a = torch.randn(B, N, k, k, dtype=dtype)
    sigma = a @ a.transpose(-1, -2) + 0.5 * torch.eye(k, dtype=dtype)
    return sigma.detach().requires_grad_(True)


def _leaves(B=2, N=3, seed=1, dtype=torch.float32):
    torch.manual_seed(seed)
    mu = (0.3 * torch.randn(B, N, K, dtype=dtype)).detach().requires_grad_(True)
    sigma = _spd_leaf(B, N, K, seed + 1, dtype=dtype)
    targets = torch.randint(0, V, (B, N))
    return mu, sigma, targets


def _spy_checkpoint(monkeypatch):
    """Wrap torch.utils.checkpoint.checkpoint with a call counter that still delegates to the real
    implementation, so results stay correct while the number of checkpoint invocations is observable.
    """
    calls = {"n": 0}
    real_checkpoint = torch.utils.checkpoint.checkpoint

    def _spy(fn, *args, **kwargs):
        calls["n"] += 1
        return real_checkpoint(fn, *args, **kwargs)

    monkeypatch.setattr(prior_bank_module._checkpoint, "checkpoint", _spy)
    return calls


# ---------------------------------------------------------------------------
# 1. Value identity: 'always' vs 'never', fp32 and fp64, several seeds and chunk widths.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_full_chunked_value_identical_always_vs_never(dtype):
    for seed in (0, 1, 2):
        pb = _full_bank(seed=seed)
        if dtype is torch.float64:
            pb = pb.double()
        mu, sigma, targets = _leaves(seed=seed + 10, dtype=dtype)
        for chunk in (7, 100):                      # 7 < V=50 (multiple chunks); 100 > V (one chunk)
            pb.decode_ce_checkpoint = "always"
            ce_always = pb.decode_ce_full_chunked(mu, sigma, targets, chunk_size=chunk)
            pb.decode_ce_checkpoint = "never"
            ce_never = pb.decode_ce_full_chunked(mu, sigma, targets, chunk_size=chunk)
            assert torch.equal(ce_always, ce_never), (
                f"seed={seed} dtype={dtype} chunk={chunk}: "
                f"always={ce_always.item()!r} != never={ce_never.item()!r}"
            )


# ---------------------------------------------------------------------------
# 2. Gradient identity: mu_q, sigma_q, and both decode tables.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_full_chunked_gradient_identical_always_vs_never(dtype):
    """torch.autograd.grad of the CE w.r.t. mu_q, sigma_q, and the decode tables (mu_embed,
    sigma_log_embed) must match between 'always' and 'never'. Asserted bit-identical
    (torch.equal): use_reentrant=False checkpointing recomputes the SAME _chunk_summaries closure
    on the SAME (deterministic, RNG-free) saved inputs, so the recomputed backward is not an
    approximation of the saved-activation backward -- it is the identical floating-point program
    executed a second time, with no accumulation-order or reduction-order difference for autograd
    to introduce rounding into. Two separately-constructed PriorBanks with the SAME seed give
    byte-identical initial tables (deterministic CPU RNG), so this compares two full, independent
    graphs rather than reusing one bank's accumulated .grad across branches.
    """
    for seed in (0, 1):
        for chunk in (7, 100):
            pb_always = _full_bank(seed=seed)
            pb_never = _full_bank(seed=seed)
            if dtype is torch.float64:
                pb_always, pb_never = pb_always.double(), pb_never.double()
            pb_always.decode_ce_checkpoint = "always"
            pb_never.decode_ce_checkpoint = "never"

            mu0, sigma0, targets = _leaves(seed=seed + 20, dtype=dtype)
            mu_a = mu0.detach().clone().requires_grad_(True)
            sigma_a = sigma0.detach().clone().requires_grad_(True)
            mu_n = mu0.detach().clone().requires_grad_(True)
            sigma_n = sigma0.detach().clone().requires_grad_(True)

            ce_a = pb_always.decode_ce_full_chunked(mu_a, sigma_a, targets, chunk_size=chunk)
            g_mu_a, g_sigma_a, g_mue_a, g_sige_a = torch.autograd.grad(
                ce_a, (mu_a, sigma_a, pb_always.mu_embed, pb_always.sigma_log_embed))

            ce_n = pb_never.decode_ce_full_chunked(mu_n, sigma_n, targets, chunk_size=chunk)
            g_mu_n, g_sigma_n, g_mue_n, g_sige_n = torch.autograd.grad(
                ce_n, (mu_n, sigma_n, pb_never.mu_embed, pb_never.sigma_log_embed))

            assert torch.equal(ce_a, ce_n), f"seed={seed} chunk={chunk} dtype={dtype}: CE mismatch"
            assert torch.equal(g_mu_a, g_mu_n), f"seed={seed} chunk={chunk}: mu_q grad mismatch"
            assert torch.equal(g_sigma_a, g_sigma_n), f"seed={seed} chunk={chunk}: sigma_q grad mismatch"
            assert torch.equal(g_mue_a, g_mue_n), f"seed={seed} chunk={chunk}: mu_embed grad mismatch"
            assert torch.equal(g_sige_a, g_sige_n), f"seed={seed} chunk={chunk}: sigma_log_embed grad mismatch"


# ---------------------------------------------------------------------------
# 3. 'auto' dispatches on the real activation-byte estimate, not on a config guess or a timing.
# ---------------------------------------------------------------------------
def test_auto_activation_bytes_formula_matches_documented_formula():
    """_decode_ce_chunk_activation_bytes must equal batch*positions*chunk_width*itemsize (times the
    `inner` multiplier the K/K*K-workspace kernels pass), exactly the formula documented on the
    decode_ce_checkpoint config field -- not a fudge factor.
    """
    ref = torch.zeros(3, 5, 1, dtype=torch.float32)
    assert _decode_ce_chunk_activation_bytes(ref, 8192) == 3 * 5 * 8192 * 4
    assert _decode_ce_chunk_activation_bytes(ref, 8192, inner=6) == 3 * 5 * 8192 * 6 * 4
    ref64 = ref.double()
    assert _decode_ce_chunk_activation_bytes(ref64, 8192) == 3 * 5 * 8192 * 8


def test_auto_selects_no_checkpoint_at_reference_scale():
    """Reference scale from the dispatched task: B=32, N=64, K=20, V=50257, decode_chunk_size=8192.
    (B, N, chunk_width) at fp32 is ~64 MiB, far under DECODE_CE_CHECKPOINT_AUTO_BYTES (2 GiB), so
    'auto' must decide NOT to checkpoint -- asserted on the actual boolean decision, not a timing.
    """
    B, N, chunk = 32, 64, 8192
    ref = torch.zeros(B, N, 1, dtype=torch.float32)
    activation_bytes = _decode_ce_chunk_activation_bytes(ref, chunk)
    assert activation_bytes < DECODE_CE_CHECKPOINT_AUTO_BYTES, activation_bytes
    assert _decode_ce_should_checkpoint("auto", True, activation_bytes) is False


def test_auto_selects_checkpoint_at_large_batch_seq_chunk():
    """A deliberately large B*N*chunk_width (B=64, N=1024, chunk=50000, fp32 -> ~12.2 GiB) must
    exceed the byte budget, so 'auto' must decide TO checkpoint -- this is exactly the regime the
    checkpoint exists to bound, so 'auto' must not silently disable it at scale.
    """
    B, N, chunk = 64, 1024, 50000
    ref = torch.zeros(B, N, 1, dtype=torch.float32)
    activation_bytes = _decode_ce_chunk_activation_bytes(ref, chunk)
    assert activation_bytes > DECODE_CE_CHECKPOINT_AUTO_BYTES, activation_bytes
    assert _decode_ce_should_checkpoint("auto", True, activation_bytes) is True


def test_auto_wired_into_full_chunked_call_site(monkeypatch):
    """Prove decode_ce_full_chunked's checkpoint branch actually consults decode_ce_checkpoint AND
    the real per-chunk activation-byte estimate at the call site -- not merely that the standalone
    dispatch helpers are correct in isolation. Spies on torch.utils.checkpoint.checkpoint (still
    delegating to the real implementation) and drives the SAME real, small (V=50) call through both
    sides of the 'auto' decision by monkeypatching DECODE_CE_CHECKPOINT_AUTO_BYTES down to 1 byte,
    rather than materializing a gigabyte-scale tensor to force the real threshold (the whole point
    of 'auto' is to avoid paying that cost at reference scale).
    """
    pb = _full_bank(seed=0)
    pb.decode_ce_checkpoint = "auto"
    mu, sigma, targets = _leaves(B=2, N=3, seed=1)
    calls = _spy_checkpoint(monkeypatch)

    # Real (2 GiB) budget: the tiny (B=2, N=3, chunk<=V=50) tensor is far under it.
    calls["n"] = 0
    pb.decode_ce_full_chunked(mu, sigma, targets, chunk_size=7)
    assert calls["n"] == 0, "auto checkpointed a tiny chunk well under the byte budget"

    # Same real call, budget monkeypatched to 1 byte: every one of the 8 chunks (ceil(50/7)) now
    # exceeds it, so auto must checkpoint every one of them.
    monkeypatch.setattr(prior_bank_module, "DECODE_CE_CHECKPOINT_AUTO_BYTES", 1)
    calls["n"] = 0
    pb.decode_ce_full_chunked(mu, sigma, targets, chunk_size=7)
    assert calls["n"] == 8, f"expected 8/8 chunks checkpointed under a 1-byte budget, got {calls['n']}"


# ---------------------------------------------------------------------------
# 4. Gradients still flow under 'never': finite, none None.
# ---------------------------------------------------------------------------
def test_never_gradients_flow_and_are_finite():
    pb = _full_bank(seed=0)
    pb.decode_ce_checkpoint = "never"
    mu, sigma, targets = _leaves(B=2, N=3, seed=1)
    ce = pb.decode_ce_full_chunked(mu, sigma, targets, chunk_size=7)
    ce.backward()
    for name, t in (("mu_q", mu), ("sigma_q", sigma),
                    ("mu_embed", pb.mu_embed), ("sigma_log_embed", pb.sigma_log_embed)):
        assert t.grad is not None, f"{name}.grad is None under decode_ce_checkpoint='never'"
        assert torch.isfinite(t.grad).all(), f"{name}.grad has non-finite entries"


# ---------------------------------------------------------------------------
# 5. Eval path unchanged: under torch.no_grad() the checkpoint branch is never taken, in any mode.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["auto", "always", "never"])
def test_no_grad_never_checkpoints_in_any_mode(monkeypatch, mode):
    pb = _full_bank(seed=0)
    pb.decode_ce_checkpoint = mode
    mu, sigma, targets = _leaves(B=2, N=3, seed=1)
    mu, sigma = mu.detach(), sigma.detach()
    calls = _spy_checkpoint(monkeypatch)

    with torch.no_grad():
        pb.decode_ce_full_chunked(mu, sigma, targets, chunk_size=7)
    assert calls["n"] == 0, f"mode={mode!r} checkpointed under torch.no_grad()"


# ---------------------------------------------------------------------------
# 6. Config validation: invalid value raises; default is 'auto'.
# ---------------------------------------------------------------------------
def test_config_decode_ce_checkpoint_default_is_auto():
    assert test_config().decode_ce_checkpoint == "auto"


def test_config_decode_ce_checkpoint_rejects_invalid_value():
    with pytest.raises(ValueError, match="decode_ce_checkpoint"):
        test_config(decode_ce_checkpoint="sometimes")


@pytest.mark.parametrize("value", ["auto", "always", "never"])
def test_config_decode_ce_checkpoint_accepts_valid_values(value):
    test_config(decode_ce_checkpoint=value)


def test_config_decode_ce_checkpoint_inert_when_decode_route_has_no_fused_ce():
    # decode_mode='full' (not 'full_chunked') has no fused_ce registration: the dense
    # decode() -> F.cross_entropy path has no chunk loop, so decode_ce_checkpoint does nothing.
    # oracle_unroll_grad=True silences the unrelated (pre-existing, orthogonal) "gaussian_full
    # routes the belief gradient to the detached oracle" advisory this family/decode_mode pair
    # also triggers, so pytest.warns below asserts on decode_ce_checkpoint specifically.
    with pytest.warns(ConfigNotice, match="decode_ce_checkpoint"):
        test_config(decode_mode="full", family="gaussian_full",
                   _allow_legacy_dense_decode=True,
                   oracle_unroll_grad=True, decode_ce_checkpoint="always")


def test_config_decode_ce_checkpoint_silent_at_default_even_when_inert():
    # The silence contract (audit 2026-07-25 F4 family): leaving a field at its dataclass default
    # must not warn even where it would be inert under this config, matching every sibling
    # `_changed(...)` guard in the inertness block. oracle_unroll_grad=True silences the unrelated
    # detached-oracle advisory this family/decode_mode pair also triggers (see test above), and
    # skip_belief_sigma_update=False silences the frozen-sigma correctness advisory that the
    # current default skip_belief_sigma_update=True legitimately raises under a prior-bank decode.
    from vfe3.config import VFE3Config
    selected = VFE3Config.__dataclass_fields__["decode_ce_checkpoint"].default
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConfigNotice)
        cfg = test_config(decode_mode="full", family="gaussian_full",
                          _allow_legacy_dense_decode=True,
                          oracle_unroll_grad=True, skip_belief_sigma_update=False,
                          decode_ce_checkpoint=selected)
    assert "decode_ce_checkpoint" in cfg.inactive_fields()
    assert not any("decode_ce_checkpoint=" in str(item.message) for item in caught)


def test_config_decode_ce_checkpoint_live_under_full_chunked():
    # The dispatched target and the live train_vfe3.py decode_mode: no inert warning here.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConfigNotice)
        cfg = test_config(decode_mode="full_chunked", family="gaussian_full",
                          oracle_unroll_grad=True, decode_ce_checkpoint="always")
    assert "decode_ce_checkpoint" not in cfg.inactive_fields()
    assert not any("decode_ce_checkpoint=" in str(item.message) for item in caught)


# ---------------------------------------------------------------------------
# Consistency check: the other four decode_ce_*_chunked kernels got the SAME fix (see module
# docstring) -- pin the same value-identity contract on each, lightly.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("decode_mode,kwargs", [
    ("diagonal_chunked", {}),
    ("expected_likelihood_chunked", {}),
    ("family_chunked", {"family": "gaussian_diagonal", "divergence_family": "renyi", "renyi_order": 1.0}),
])
def test_other_diagonal_chunked_kernels_value_identical_always_vs_never(decode_mode, kwargs):
    torch.manual_seed(0)
    pb = PriorBank(V, K, N_GEN, mu_init_std=0.4, decode_mode=decode_mode, **kwargs)
    mu = (0.3 * torch.randn(2, 3, K)).requires_grad_(True)
    sigma = (0.5 + 0.5 * torch.rand(2, 3, K)).requires_grad_(True)
    targets = torch.randint(0, V, (2, 3))
    fn = getattr(pb, f"decode_ce_{decode_mode}")
    for chunk in (7, 100):
        pb.decode_ce_checkpoint = "always"
        ce_always = fn(mu, sigma, targets, chunk_size=chunk)
        pb.decode_ce_checkpoint = "never"
        ce_never = fn(mu, sigma, targets, chunk_size=chunk)
        assert torch.equal(ce_always, ce_never), f"{decode_mode} chunk={chunk}"


def test_linear_chunked_value_identical_always_vs_never():
    torch.manual_seed(0)
    pb = PriorBank(V, K, N_GEN, mu_init_std=0.4, decode_mode="linear")
    mu = (0.3 * torch.randn(2, 3, K)).requires_grad_(True)
    targets = torch.randint(0, V, (2, 3))
    for chunk in (7, 100):
        pb.decode_ce_checkpoint = "always"
        ce_always = pb.decode_ce_linear_chunked(mu, targets, chunk_size=chunk)
        pb.decode_ce_checkpoint = "never"
        ce_never = pb.decode_ce_linear_chunked(mu, targets, chunk_size=chunk)
        assert torch.equal(ce_always, ce_never), f"linear_chunked chunk={chunk}"
