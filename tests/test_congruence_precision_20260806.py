r"""Full-covariance congruence precision policy (audit 2026-08-06 C1/F3).

``Omega Sigma Omega^T`` hard-coded ``.double()`` at every site -- no dtype parameter, no module
global, no autocast guard -- unlike the KL (``full_cov_kl_precision``) and unlike the diagonal
sibling's ``_escalate_if_negative``. At the live shape it is the largest stage in the step.

Pins: (a) the DEFAULT ``"fp64"`` is byte-identical to the pre-policy literal-``.double()``
contraction, on the compact, blocks-only and dense routes, so every run on disk stays reproducible;
(b) a float64 SOURCE contracts in float64 under either policy; (c) ``"fp32_escalate"`` agrees with
float64 to float32 rounding on well-conditioned input; (d) it recovers finiteness by escalating when
the float32 contraction overflows; (e) the config field validates and reaches the geometry through
``VFEModel``.
"""

import pytest
import torch

import vfe3.geometry.transport as T
from vfe3.config import VFE3Config
from vfe3.geometry.transport import (
    CompactFactoredTransport,
    full_cov_congruence_precision,
    set_full_cov_congruence_precision,
    transport_covariance,
)

H, d = 2, 5
K = H * d
B, N = 2, 6


@pytest.fixture(autouse=True)
def _restore_policy():
    previous = full_cov_congruence_precision()
    yield
    set_full_cov_congruence_precision(previous)


def _omega(seed=0, scale=0.2):
    g = torch.Generator().manual_seed(seed)
    a = scale * torch.randn(B, N, H, d, d, generator=g, dtype=torch.float64)
    a = a - a.transpose(-1, -2)
    return CompactFactoredTransport(
        torch.matrix_exp(a).float(), torch.matrix_exp(-a).float(), K)


def _sigma(seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, N, K, K, generator=g, dtype=torch.float64)
    s = x @ x.transpose(-1, -2) + K * torch.eye(K, dtype=torch.float64)
    return s.to(dtype)


def _reference_blocks_only(om, sigma):
    r"""The pre-policy contraction, written out with literal .double() as it used to be."""
    blocks = T._compact_pair_blocks(om).double()
    sb = sigma.double().reshape(*sigma.shape[:-3], sigma.shape[-3], H, d, H, d)
    sd = torch.diagonal(sb, dim1=-4, dim2=-2).movedim(-1, -3)
    diag = torch.einsum("...ijhkl,...jhlm,...ijhnm->...ijhkn", blocks, sd, blocks)
    out = diag.new_zeros((*diag.shape[:-3], H, d, H, d))
    for h in range(H):
        out[..., h, :, h, :] = diag[..., h, :, :]
    return out.reshape(*out.shape[:-4], K, K)


# -- (a) the default is byte-identical to the pre-policy build --------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_default_fp64_is_byte_identical_to_the_literal_double_contraction(seed):
    set_full_cov_congruence_precision("fp64")
    om, sigma = _omega(seed), _sigma(seed)
    got = transport_covariance(om, sigma, retain_full_precision=True,
                               diagonal_out=False, marginal_blocks=[d] * H)
    assert torch.equal(got, _reference_blocks_only(om, sigma))


def test_default_fp64_byte_identical_on_the_dense_route():
    set_full_cov_congruence_precision("fp64")
    g = torch.Generator().manual_seed(5)
    omega = 0.1 * torch.randn(B, N, N, K, K, generator=g) + torch.eye(K)
    sigma = _sigma(5)
    got = transport_covariance(omega, sigma, retain_full_precision=True, diagonal_out=False)
    ref = torch.einsum("...ijkl,...jlm,...ijnm->...ijkn",
                       omega.double(), sigma.double(), omega.double())
    assert torch.equal(got, ref)


# -- (b) a float64 source is never demoted ----------------------------------------------------


def test_float64_source_contracts_in_float64_under_either_policy():
    om = _omega(3)
    sigma64 = _sigma(3, dtype=torch.float64)
    outs = {}
    for policy in ("fp64", "fp32_escalate"):
        set_full_cov_congruence_precision(policy)
        outs[policy] = transport_covariance(
            om, sigma64, retain_full_precision=True, diagonal_out=False,
            marginal_blocks=[d] * H)
    assert outs["fp64"].dtype is torch.float64
    assert torch.equal(outs["fp64"], outs["fp32_escalate"])


# -- (c)/(d) the opt-in: accurate on good input, escalates on overflow ------------------------


def test_fp32_escalate_matches_fp64_to_float32_rounding():
    om, sigma = _omega(4), _sigma(4)
    set_full_cov_congruence_precision("fp64")
    ref = transport_covariance(om, sigma, diagonal_out=False, marginal_blocks=[d] * H)
    set_full_cov_congruence_precision("fp32_escalate")
    fast = transport_covariance(om, sigma, diagonal_out=False, marginal_blocks=[d] * H)
    scale = ref.abs().max()
    assert (fast - ref).abs().max() <= 8.0 * torch.finfo(torch.float32).eps * scale


def test_fp32_escalate_recovers_finiteness_by_escalating():
    r"""A float32 contraction that overflows must come back finite, not as inf."""
    om = _omega(6)
    sigma = _sigma(6) * 1e30                       # float32-representable, but its square is not
    set_full_cov_congruence_precision("fp32_escalate")
    out = transport_covariance(om, sigma, retain_full_precision=True,
                               diagonal_out=False, marginal_blocks=[d] * H)
    assert torch.isfinite(out).all(), "the escalation did not recover the overflowed contraction"


# -- (e) the config field is validated and reaches the geometry -------------------------------


def test_config_default_and_validation():
    cfg = VFE3Config(vocab_size=32, embed_dim=K, n_heads=H, max_seq_len=8)
    # The shipped default was deliberately moved to "fp32_escalate": the congruence is the
    # largest stage in the step and its float64 cost is not repaid at the trained operating
    # point. Runs recorded BEFORE that flip were produced under "fp64" -- pass it explicitly
    # to reproduce them. The byte-identity pins above set the policy themselves, so they are
    # unaffected by which value is the default.
    assert cfg.full_cov_congruence_precision == "fp32_escalate"
    with pytest.raises(ValueError, match="full_cov_congruence_precision"):
        VFE3Config(vocab_size=32, embed_dim=K, n_heads=H, max_seq_len=8,
                   full_cov_congruence_precision="bogus")


def test_model_construction_sets_the_process_policy():
    from vfe3.model.model import VFEModel

    set_full_cov_congruence_precision("fp64")
    VFEModel(VFE3Config(vocab_size=32, embed_dim=K, n_heads=H, max_seq_len=8,
                        full_cov_congruence_precision="fp32_escalate"))
    assert full_cov_congruence_precision() == "fp32_escalate"
