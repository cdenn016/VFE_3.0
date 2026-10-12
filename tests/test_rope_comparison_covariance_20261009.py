r"""Rotation-invariant comparison covariance under comparison-space RoPE (audit 2026-10-09 round 2, addendum).

Under ``rope_space='comparison'`` token i's projected moments are rotated, a -> R_i a and G -> R_i G R_i^T,
so the pair cost is KL(N(a_q, G_q) || N(R_{j-i} a_k, R_{j-i} G_k R_{j-i}^T)). The trace and log-determinant
terms then penalize every non-self pair in proportion to the anisotropy of the comparison covariance, and the
self pair, compared without a relative rotation, wins (the layer-1 self-lock). With
``rope_comparison_covariance='rotation_invariant'`` each comparison covariance is projected onto the commutant
of the RoPE torus,

.. math::
    P(G) = \bigoplus_p \tfrac12 (G_{aa} + G_{bb}) I_2 \oplus G_{00},

one isotropic variance per rotating plane p = (a, b), the full block G_{00} of the non-rotating coordinates, and
zero cross-covariance. Then R P(G) R^T = P(G) for every torus element, so only the mean term carries relative
position. Also: the tie under comparison-space RoPE is not the flat-alignment identity (item 6).
"""
import math
import warnings

import pytest
import torch

from tests.profiles import make_relational_operators, tiny_config
from vfe3.contracts import CanonicalFrameContext
from vfe3.geometry.rope import build_rope_rotation, rope_commutant_projection, rope_rotated_pairs
from vfe3.gradients.relational import relational_score
from vfe3.model.model import VFEModel


def _spd(r: int, seed: int, batch: tuple = ()) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(*batch, r, r, generator=g, dtype=torch.float64)
    return A @ A.transpose(-1, -2) + 0.1 * torch.eye(r, dtype=torch.float64)


def _identity_frame(n: int, heads: int, d: int) -> CanonicalFrameContext:
    eye = torch.eye(d, dtype=torch.float64).expand(1, n, heads, d, d).clone()
    return CanonicalFrameContext(forward=eye, inverse=eye.clone(), block_diagonal=True)


def _ops(invariant: bool, *, fraction: float = 1.0, tie: bool = True, heads: int = 2, d: int = 4,
         seed: int = 11) -> torch.nn.Module:
    return make_relational_operators(
        [d] * heads, seed, comparison_rope_base=100.0, comparison_rope_fraction=fraction,
        tie_comparison_maps=tie,
        rope_comparison_covariance="rotation_invariant" if invariant else "full").double()


# ------------------------------------------------------------------------------------- (a) the commutant projection

@pytest.mark.parametrize("r, fraction", [(4, 1.0), (5, 1.0), (4, 0.5), (6, 0.5), (2, 1.0)])
def test_projected_covariance_commutes_with_every_rotation(r: int, fraction: float) -> None:
    G = _spd(r, 3, (2,))
    n_pairs = rope_rotated_pairs(r, fraction)
    P = rope_commutant_projection(G, n_pairs)
    R = build_rope_rotation(torch.arange(9), [r], base=100.0, fraction=fraction, dtype=torch.float64)  # (9, r, r)
    torch.testing.assert_close(R.unsqueeze(1) @ P @ R.transpose(-1, -2).unsqueeze(1),
                               P.expand(9, *P.shape), rtol=0, atol=1e-14)
    assert torch.equal(P, P.transpose(-1, -2))
    torch.testing.assert_close(rope_commutant_projection(P, n_pairs), P, rtol=0, atol=0)    # idempotent
    assert bool((torch.linalg.eigvalsh(P) > 0).all())
    m = 2 * n_pairs
    # One isotropic variance per rotating plane, the non-rotating block kept, no cross-covariance.
    for p in range(n_pairs):
        a, b = 2 * p, 2 * p + 1
        torch.testing.assert_close(P[..., a, a], 0.5 * (G[..., a, a] + G[..., b, b]), rtol=0, atol=0)
        torch.testing.assert_close(P[..., b, b], P[..., a, a], rtol=0, atol=0)
    torch.testing.assert_close(P[..., m:, m:], G[..., m:, m:], rtol=0, atol=0)
    off = P - torch.diag_embed(torch.diagonal(P, dim1=-2, dim2=-1))
    off[..., m:, m:] = 0
    assert off.abs().max() == 0
    # The projection is the Haar (torus) average: the Frobenius-orthogonal projection onto the commutant.
    theta = torch.rand(4000, n_pairs, generator=torch.Generator().manual_seed(5), dtype=torch.float64) * 2 * math.pi
    T = torch.eye(r, dtype=torch.float64).expand(4000, r, r).clone()
    for p in range(n_pairs):
        c, s = theta[:, p].cos(), theta[:, p].sin()
        T[:, 2 * p, 2 * p], T[:, 2 * p, 2 * p + 1], T[:, 2 * p + 1, 2 * p], T[:, 2 * p + 1, 2 * p + 1] = c, -s, s, c
    average = (T.unsqueeze(1) @ G @ T.transpose(-1, -2).unsqueeze(1)).mean(0)
    torch.testing.assert_close(average, P, rtol=0, atol=0.15 * float(G.abs().max()))


def test_rope_frequencies_are_distinct_for_every_base_other_than_one() -> None:
    r"""freq_k = base^{-2k/r} is strictly monotone in k for base != 1, so the planes rotate independently.

    At base = 1 every plane has the same frequency; P(G) still commutes with every R_i (it lies in the commutant
    of the whole torus, which is contained in the commutant of any subgroup), only the realized commutant is
    larger than P's range."""
    for base in (100.0, 10000.0, 0.5):
        for r in (4, 6, 20, 32):
            freqs = torch.tensor([base ** (-2.0 * k / r) for k in range(r // 2)], dtype=torch.float64)
            assert bool((freqs[1:] != freqs[:-1]).all())
    R = build_rope_rotation(torch.arange(5), [4], base=1.0, dtype=torch.float64)
    P = rope_commutant_projection(_spd(4, 1), 2)
    torch.testing.assert_close(R @ P @ R.transpose(-1, -2), P.expand(5, 4, 4), rtol=0, atol=1e-14)


# ------------------------------------------------------------------------------------- (b) position-free covariance terms

def _identical_beliefs(n: int, K: int, zero_mean: bool) -> tuple:
    g = torch.Generator().manual_seed(2)
    mu = (torch.zeros(1, 1, K, dtype=torch.float64) if zero_mean
          else torch.randn(1, 1, K, generator=g, dtype=torch.float64)).expand(1, n, K)
    A = torch.randn(K, K, generator=g, dtype=torch.float64)
    sigma = (A @ A.T + 0.05 * torch.eye(K, dtype=torch.float64)).expand(1, n, K, K)       # anisotropic, full
    return mu, sigma


def test_rotation_invariant_cost_has_no_self_advantage() -> None:
    r"""Identical zero-mean anisotropic beliefs, S_Q = S_K: every rotated mean is 0, so every pair coincides.

    As run, D_ii = 0 while D_ij > 0 off the diagonal (the trace and log-det penalty of the relative rotation);
    under the toggle every pair costs exactly what self costs."""
    n, heads, d = 6, 2, 4
    mu, sigma = _identical_beliefs(n, heads * d, zero_mean=True)
    frame = _identity_frame(n, heads, d)
    full = relational_score(mu, sigma, mu, sigma, frame, _ops(False))               # (1, H, n, n)
    invariant = relational_score(mu, sigma, mu, sigma, frame, _ops(True))
    off = ~torch.eye(n, dtype=torch.bool)
    assert full.diagonal(dim1=-2, dim2=-1).abs().max() < 1e-12
    assert float(full[..., off].min()) > 1e-2
    assert invariant.abs().max() < 1e-12


def test_rotation_invariant_trace_term_is_constant_and_cost_is_the_mean_term() -> None:
    r"""Nonzero mean: under the toggle D_ij = (1/2) (R_{j-i} a - a)^T P(G)^{-1} (R_{j-i} a - a), the trace and
    log-det terms are constant (zero for equal covariances), while as run the trace term varies with j - i."""
    n, heads, d = 6, 2, 4
    mu, sigma = _identical_beliefs(n, heads * d, zero_mean=False)
    frame = _identity_frame(n, heads, d)
    ops = _ops(True)
    D = relational_score(mu, sigma, mu, sigma, frame, ops)                          # (1, H, n, n)
    S = ops.raw_score_q.detach()                                                    # (H, d, K)
    R = build_rope_rotation(torch.arange(n), [d], base=100.0, dtype=torch.float64)   # (n, d, d)
    full_trace = []
    for h in range(heads):
        a = S[h] @ mu[0, 0]
        G = S[h] @ sigma[0, 0] @ S[h].T
        P = rope_commutant_projection(G, 2)
        for i in range(n):
            for j in range(n):
                delta = R[j] @ a - R[i] @ a
                expected = 0.5 * delta @ torch.linalg.solve(P, delta)
                torch.testing.assert_close(D[0, h, i, j], expected, rtol=1e-10, atol=1e-12)
        # As run the trace term tr(G^{-1} R_{j-i} G R_{j-i}^T) - d varies with the distance.
        full_trace.append(torch.stack([torch.trace(torch.linalg.solve(G, R[k] @ G @ R[k].T)) - d
                                       for k in range(n)]))
    trace = torch.stack(full_trace)                                                 # (H, n)
    assert float(trace[:, 1:].min()) > 1e-3 and float((trace[:, 1:].max(-1).values - trace[:, 1:].min(-1).values).min()) > 1e-3


def test_rotation_invariant_cost_is_relative_for_distinct_beliefs() -> None:
    r"""Distinct beliefs, untied maps, p-RoPE: shifting the sequence by s positions shifts the cost grid."""
    n, s, heads, d = 5, 2, 2, 4
    g = torch.Generator().manual_seed(8)
    mu = torch.randn(1, n, heads * d, generator=g, dtype=torch.float64)
    sigma = 0.3 + torch.rand(1, n, heads * d, generator=g, dtype=torch.float64)
    pad_mu = torch.cat((torch.randn(1, s, heads * d, generator=g, dtype=torch.float64), mu), dim=1)
    pad_sigma = torch.cat((0.3 + torch.rand(1, s, heads * d, generator=g, dtype=torch.float64), sigma), dim=1)
    ops = _ops(True, fraction=0.5, tie=False)
    D = relational_score(mu, sigma, mu, sigma, _identity_frame(n, heads, d), ops)
    D_shift = relational_score(pad_mu, pad_sigma, pad_mu, pad_sigma, _identity_frame(n + s, heads, d), ops)
    torch.testing.assert_close(D_shift[..., s:, s:], D, rtol=1e-10, atol=1e-12)


# ------------------------------------------------------------------------------------- (c) default off, (d) gradients

def test_full_option_is_the_default_and_matches_the_as_run_rotation() -> None:
    cfg = tiny_config(embed_dim=8, pos_rotation="rope", rope_space="comparison", rope_on_value=False)
    assert cfg.rope_comparison_covariance == "full"
    model = VFEModel(cfg)
    for ops in model.relational_operators_by_layer:
        assert ops.rope_comparison_covariance == "full"
    on = VFEModel(tiny_config(embed_dim=8, pos_rotation="rope", rope_space="comparison", rope_on_value=False,
                              rope_comparison_covariance="rotation_invariant"))
    assert sorted(on.state_dict()) == sorted(model.state_dict())                     # no new parameter


def test_rotation_invariant_gradcheck() -> None:
    n, heads, d = 3, 1, 4
    g = torch.Generator().manual_seed(4)
    mu = torch.randn(1, n, d, generator=g, dtype=torch.float64, requires_grad=True)
    sigma = (0.4 + torch.rand(1, n, d, generator=g, dtype=torch.float64)).requires_grad_()
    ops = _ops(True, fraction=0.5, tie=False, heads=heads, d=d)
    frame = _identity_frame(n, heads, d)
    S_q, S_k = ops.raw_score_q, ops.raw_score_k
    assert torch.autograd.gradcheck(lambda m, s: relational_score(m, s, m, s, frame, ops), (mu, sigma),
                                    eps=1e-6, atol=1e-6, rtol=1e-5)
    grads = torch.autograd.grad(relational_score(mu, sigma, mu, sigma, frame, ops).sum(), (S_q, S_k))
    assert all(bool(torch.isfinite(x).all()) and bool((x != 0).any()) for x in grads)


def test_rotation_invariant_model_trains() -> None:
    torch.manual_seed(0)
    cfg = tiny_config(embed_dim=8, n_layers=2, pos_rotation="rope", rope_space="comparison", rope_on_value=False,
                      rope_comparison_covariance="rotation_invariant")
    model = VFEModel(cfg)
    _, logits = model.forward_beliefs(torch.tensor([[0, 1, 2]]), return_logits=True)
    logits.square().sum().backward()
    for ops in model.relational_operators_by_layer:
        assert ops.raw_score_q.grad is not None and bool((ops.raw_score_q.grad != 0).any())


# ------------------------------------------------------------------------------------- configuration

def test_option_is_inactive_without_comparison_rope_and_validated() -> None:
    assert "rope_comparison_covariance" in tiny_config().inactive_fields()
    assert "rope_comparison_covariance" in tiny_config(
        pos_rotation="rope", rope_space="frame", rope_on_value=True).inactive_fields()
    on = tiny_config(embed_dim=8, pos_rotation="rope", rope_space="comparison", rope_on_value=False,
                     rope_comparison_covariance="rotation_invariant")
    inactive = on.inactive_fields()
    assert "rope_comparison_covariance" not in inactive
    with pytest.raises(ValueError, match="rope_comparison_covariance"):
        tiny_config(rope_comparison_covariance="isotropic")
    with pytest.raises(ValueError, match="rope_comparison_covariance"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tiny_config(embed_dim=8, pos_rotation="rope", rope_space="comparison", rope_on_value=False,
                        predictive_readout="minimal_agent", attention_residual=False,
                        rope_comparison_covariance="rotation_invariant")


# ------------------------------------------------------------------------------------- item 6: the tie under comparison RoPE

def test_tie_under_comparison_rope_is_not_flat_alignment() -> None:
    r"""With S_K = S_Q and identity frames the flat-alignment identity would give D_ij = KL(q_i || q_j) = 0 for
    identical beliefs. Under comparison-space RoPE (covariance 'full', as run) the relative rotation makes
    D_ij > 0 for i != j while D_ii = 0."""
    n, heads, d = 5, 2, 4
    mu, sigma = _identical_beliefs(n, heads * d, zero_mean=False)
    D = relational_score(mu, sigma, mu, sigma, _identity_frame(n, heads, d), _ops(False, tie=True))
    assert D.diagonal(dim1=-2, dim2=-1).abs().max() < 1e-12
    assert float(D[..., ~torch.eye(n, dtype=torch.bool)].min()) > 1e-3
