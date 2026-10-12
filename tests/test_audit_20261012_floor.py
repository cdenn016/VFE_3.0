r"""The prior-anchored covariance floor of the moment-matched readouts (audit 2026-10-12, item 4 and T5).

Both projected-predictive readouts (the kernel ``moment_matched`` covariance and the minimal-agent ``mean_field``
posterior covariance) now floor their local covariance with :math:`f\,\Sigma_{p,i}`, the token's prior covariance
in local coordinates, in place of :math:`f\,\tfrac{\operatorname{tr}C_i}{K}U_iU_i^\top`. Pinned here on tiny float64
CPU cases, for the diagonal and the full family:

* exact invariance under a common reference change :math:`U_i\to U_ia` with a non-orthogonal :math:`a`
  (:math:`\operatorname{cond}a = 10`, head-block diagonal as the compact frames require), applied to every frame and
  to every map that reads or writes reference coordinates (query :math:`S\to Sa`, key :math:`K\to Ka`, sending
  :math:`T\to Ta`, receiving :math:`O\to a^{-1}O`), so that :math:`\xi_i\to a^{-1}\xi_i`,
  :math:`C_i\to a^{-1}C_ia^{-\top}` and every link :math:`U_iM_hU_j^{-1}` is unchanged (Theorem 4.2(3)); the former
  floor, evaluated by hand on the same outputs, moves by far more than rounding because
  :math:`\operatorname{tr}(a^{-1}C_ia^{-\top})\ne\operatorname{tr}C_i`;
* the floor bound :math:`\Sigma_i^+\succeq f\,\Sigma_{p,i}` (Loewner order, full family; elementwise, diagonal);
* no compounding along a residual-free stack: every layer's smallest eigenvalue stays at least
  :math:`f\,\lambda_{\min}(\Sigma_{p,i})`.
"""
import warnings
from typing import Tuple

import pytest
import torch

from tests.profiles import make_relational_operators
from vfe3.contracts import CanonicalFrameContext
from vfe3.gradients.relational import relational_predictive_readout
from vfe3.model.relational_operators import RelationalOperators

F64 = torch.float64
H, D, N = 2, 2, 4
K = H * D
FLOOR = 1e-2
READOUTS = ["kernel", "minimal_agent"]
FAMILIES = ["diagonal", "full"]


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _operators(readout: str, seed: int = 0) -> RelationalOperators:
    r"""Float64 operators of one readout (``kernel``: moment-matched kernel, ``minimal_agent``: mean field)."""
    extra = (dict(predictive_readout="minimal_agent", minimal_agent_inference="mean_field")
             if readout == "minimal_agent" else {})
    ops = make_relational_operators([D] * H, 11, readout_mode="predictive_content", attention_residual=False,
                                    predictive_covariance="moment_matched", predictive_covariance_floor=FLOOR,
                                    **extra).double()
    g = torch.Generator().manual_seed(100 + seed)
    with torch.no_grad():
        for p in ops.parameters():
            p.add_(0.3 * torch.randn(p.shape, generator=g, dtype=F64))
    return ops


def _block_dense(blocks: torch.Tensor) -> torch.Tensor:            # (..., H, d, d) -> (..., K, K)
    out = blocks.new_zeros(*blocks.shape[:-3], K, K)
    for h in range(H):
        out[..., h * D:(h + 1) * D, h * D:(h + 1) * D] = blocks[..., h, :, :]
    return out


def _spd(g: torch.Generator, *batch: int) -> torch.Tensor:          # (*batch, K, K)
    X = torch.randn(*batch, K, K, generator=g, dtype=F64)
    return X @ X.transpose(-1, -2) / K + 0.4 * torch.eye(K, dtype=F64)


def _inputs(family: str, seed: int = 1) -> Tuple[torch.Tensor, ...]:
    r"""Head-block frames ``(1, N, H, d, d)``, means, the covariance and the prior in the family's layout, and a
    causal log prior."""
    g = torch.Generator().manual_seed(seed)
    U = torch.eye(D, dtype=F64) + 0.3 * torch.randn(1, N, H, D, D, generator=g, dtype=F64)
    mu = torch.randn(1, N, K, generator=g, dtype=F64)
    if family == "diagonal":
        sigma = torch.rand(1, N, K, generator=g, dtype=F64) + 0.3
        prior = torch.rand(1, N, K, generator=g, dtype=F64) + 0.4
    else:
        sigma, prior = _spd(g, 1, N), _spd(g, 1, N)
    log_prior = torch.zeros(N, N, dtype=F64).masked_fill(~torch.ones(N, N, dtype=torch.bool).tril(), -torch.inf)
    return U, mu, sigma, prior, log_prior


def _reference_change(seed: int = 7) -> torch.Tensor:              # (H, d, d)
    r"""Head blocks :math:`a_h = Q_1\operatorname{diag}(10^{-1/2}, 10^{1/2})Q_2^\top`: every block, and so
    :math:`a`, has condition number 10."""
    g = torch.Generator().manual_seed(seed)
    Q1 = torch.linalg.qr(torch.randn(H, D, D, generator=g, dtype=F64))[0]
    Q2 = torch.linalg.qr(torch.randn(H, D, D, generator=g, dtype=F64))[0]
    a = Q1 @ torch.diag_embed(torch.logspace(-0.5, 0.5, D, dtype=F64).expand(H, D)) @ Q2.transpose(-1, -2)
    dense = _block_dense(a)
    assert abs(float(torch.linalg.cond(dense)) - 10.0) < 1e-9
    assert float((dense @ dense.T - torch.eye(K, dtype=F64)).abs().max()) > 0.5  # far from orthogonal
    return a


def _change_reference(ops: RelationalOperators, a: torch.Tensor) -> None:
    r"""Maps reading reference coordinates take :math:`X\to Xa`; the receiving map takes :math:`O\to a^{-1}O`."""
    a = _block_dense(a)
    with torch.no_grad():
        for name in ("raw_score_q", "raw_score_k", "T"):
            p = getattr(ops, name, None)
            if p is not None:
                p.copy_(p @ a)                                       # (H, d, K) @ (K, K)
        if getattr(ops, "O", None) is not None:
            ops.O.copy_(torch.linalg.inv(a) @ ops.O)                 # (K, K) @ (H, K, d)


def _readout(ops: RelationalOperators, U: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor,
             prior: torch.Tensor, log_prior: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    frame = CanonicalFrameContext(U, torch.linalg.inv(U), block_diagonal=True)
    return relational_predictive_readout(mu, sigma, frame, ops, tau=0.9, log_prior=log_prior,
                                         return_covariance=True, floor_anchor=prior)


def _old_floor(U: torch.Tensor, sigma: torch.Tensor, diagonal: bool) -> torch.Tensor:
    r"""The former floor :math:`f\,\tfrac{\operatorname{tr}C_i}{K}U_iU_i^\top` (its diagonal on the diagonal family)."""
    U = _block_dense(U)                                                                          # (1, N, K, K)
    U_inv = torch.linalg.inv(U)
    Sigma = torch.diag_embed(sigma) if diagonal else sigma
    trace = torch.diagonal(U_inv @ Sigma @ U_inv.transpose(-1, -2), dim1=-2, dim2=-1).sum(-1)    # (1, N) tr C_i
    old = (FLOOR * trace / K)[..., None, None] * (U @ U.transpose(-1, -2))                        # (1, N, K, K)
    return torch.diagonal(old, dim1=-2, dim2=-1) if diagonal else old


def _relative(x: torch.Tensor, y: torch.Tensor) -> float:
    return float((x - y).abs().max() / y.abs().max())


@pytest.mark.parametrize("readout", READOUTS)
@pytest.mark.parametrize("family", FAMILIES)
def test_common_reference_change_with_non_orthogonal_a_is_exact(readout: str, family: str) -> None:
    U, mu, sigma, prior, log_prior = _inputs(family)
    ops = _operators(readout)
    mean, cov = _readout(ops, U, mu, sigma, prior, log_prior)
    a = _reference_change()
    _change_reference(ops, a)
    mean_a, cov_a = _readout(ops, U @ a, mu, sigma, prior, log_prior)          # U_i -> U_i a, blockwise
    mean_dev, cov_dev = _relative(mean_a, mean), _relative(cov_a, cov)
    # The former floor on the same kernel or posterior part (out - f Sigma_p is frame-free and invariant).
    diagonal = family == "diagonal"
    old = cov - FLOOR * prior + _old_floor(U, sigma, diagonal)
    old_a = cov_a - FLOOR * prior + _old_floor(U @ a, sigma, diagonal)
    old_dev = _relative(old_a, old)
    print(f"\n[floor] {readout}/{family}: reference change cond(a)=10, mean {mean_dev:.2e}, prior-anchored "
          f"covariance {cov_dev:.2e}, former trace floor {old_dev:.2e}")
    assert mean_dev < 1e-12                       # the transformation of frames and maps is complete
    assert cov_dev < 1e-12                        # the prior-anchored floor is exactly reference invariant
    assert old_dev > 1e-8                         # the former floor is not: the test discriminates


@pytest.mark.parametrize("readout", READOUTS)
@pytest.mark.parametrize("family", FAMILIES)
def test_output_is_at_least_the_floor(readout: str, family: str) -> None:
    r""":math:`\Sigma_i^+ - f\,\Sigma_{p,i}` is the kernel or posterior covariance: positive semidefinite (full
    family) or positive (diagonal family), and not itself of the floor's order."""
    U, mu, sigma, prior, log_prior = _inputs(family, seed=3)
    _, cov = _readout(_operators(readout, seed=2), U, mu, sigma, prior, log_prior)
    excess = cov - FLOOR * prior
    if family == "diagonal":
        assert bool((excess > 0).all())
    else:
        eigenvalues = torch.linalg.eigvalsh(excess)
        assert float(eigenvalues.min()) >= -1e-12 * float(eigenvalues.max())
    assert float(excess.abs().max()) > 10 * FLOOR * float(prior.abs().max())   # the content is not negligible


@pytest.mark.parametrize("readout", READOUTS)
def test_floor_does_not_compound_along_a_residual_free_stack(readout: str) -> None:
    r"""Six layers feeding ``(mu, Sigma)`` forward with the prior fixed: every output is
    :math:`\succeq f\,\Sigma_p`, so its smallest eigenvalue is at least :math:`f\,\lambda_{\min}(\Sigma_p)` at every
    depth, whatever the incoming covariance."""
    U, mu, sigma, prior, log_prior = _inputs("full", seed=5)
    layers = [_operators(readout, seed=10 + layer) for layer in range(6)]
    floor_min = FLOOR * torch.linalg.eigvalsh(prior)[..., 0]                    # (1, N)
    state = (mu, sigma)
    for ops in layers:
        state = _readout(ops, U, *state, prior, log_prior)
        excess = torch.linalg.eigvalsh(state[1] - FLOOR * prior)
        assert float(excess.min()) >= -1e-12 * float(excess.max())
        assert bool((torch.linalg.eigvalsh(state[1])[..., 0] >= floor_min * (1 - 1e-9)).all())
