r"""Independent tests for the cue/result generative memory (``block_mlp_mode='cue_result_memory'``).

Every expected value is derived here from the declared joint model, never from the kernel's formulas. For each
component the stacked Gaussian :math:`(X, Y, O)` has mean :math:`(\nu_a, \omega_a, \nu_a)` and covariance

.. math::
    \begin{pmatrix} L & LF^\top & L\\ FL & FLF^\top + Q & FL\\ L & LF^\top & L + C\end{pmatrix},

and :math:`(X, Y)` is conditioned on :math:`O = \xi` by the generic Schur complement; the component weights come from
the record density of ``torch.distributions.MultivariateNormal`` and the mixture moments from the law of total
covariance. The tests also cover chunking, the dense/low-rank equivalence, the same-state reduction against the
independent ``prototype_exact_posterior``, the input-independent component means at :math:`F = 0`, float64
``gradcheck``, passive frame covariance of the reference block, the configured model path, optimizer groups,
parameter counts and the configuration restrictions. CPU only, one thread, :math:`K \le 4`, :math:`M \le 8`.
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, Optional, Tuple

import pytest
import torch

from tests.profiles import test_config
from vfe3.belief import BeliefState
from vfe3.contracts import CanonicalFrameContext
from vfe3.model.block import _reference_coordinate_block
from vfe3.model.block_mlp import (
    CueResultMemoryBlockMLP,
    block_mlp_build_metadata,
    build_block_mlp,
    get_block_mlp_registration,
)
from vfe3.model.cue_result_memory import cue_result_exact_posterior
from vfe3.model.model import VFEModel
from vfe3.model.prototype_agents import prototype_exact_posterior

torch.set_num_threads(1)
DT = torch.float64
TOL = 1e-10


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _spd(batch: Tuple[int, ...], K: int, seed: int, scale: float = 0.6) -> torch.Tensor:
    A = scale * torch.randn(*batch, K, K, generator=_gen(seed), dtype=DT)
    return A @ A.transpose(-1, -2) + 0.5 * torch.eye(K, dtype=DT)


def _close(a: torch.Tensor, b: torch.Tensor, tol: float, what: str) -> None:
    a, b = a.detach(), b.detach()
    err = float((a - b).abs().max())
    scale = max(1.0, float(b.abs().max()))
    assert err <= tol * scale, f"{what}: max error {err:.3e} (scale {scale:.3e})"


def _bank(K: int, M: int, seed: int) -> Dict[str, torch.Tensor]:
    g = _gen(seed)
    return {
        "nu": 1.5 * torch.randn(M, K, generator=g, dtype=DT),
        "omega": torch.randn(M, K, generator=g, dtype=DT),
        "log_lambda": 0.5 * torch.randn(M, K, generator=g, dtype=DT),
        "log_result_noise": 0.5 * torch.randn(M, K, generator=g, dtype=DT) - 1.0,
        "log_pi": torch.randn(M, generator=g, dtype=DT),
    }


def _rotation_and_reversal(K: int, M: int, seed: int) -> torch.Tensor:
    r"""Regression matrices including a rotation and a negative response direction (not reachable by same-state)."""
    F = torch.randn(M, K, K, generator=_gen(seed), dtype=DT)
    theta = 0.9
    R = torch.eye(K, dtype=DT)
    R[0, 0], R[0, 1], R[1, 0], R[1, 1] = math.cos(theta), -math.sin(theta), math.sin(theta), math.cos(theta)
    F[0] = R
    F[1] = -torch.eye(K, dtype=DT)
    return F


def _reference(xi: torch.Tensor, C: torch.Tensor, bank: Dict[str, torch.Tensor],
               F: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Moments of p(Y | O = xi), weights and log evidence for one record, from the stacked joint Gaussian."""
    K, M = xi.shape[-1], bank["nu"].shape[0]
    log_pi_hat = torch.log_softmax(bank["log_pi"], dim=-1)
    means, covs, logs = [], [], []
    for a in range(M):
        L = torch.diag(bank["log_lambda"][a].exp())
        Q = torch.diag(bank["log_result_noise"][a].exp())
        Fa = F[a]
        J = torch.zeros(3 * K, 3 * K, dtype=DT)
        blocks = [[L, L @ Fa.T, L], [Fa @ L, Fa @ L @ Fa.T + Q, Fa @ L], [L, L @ Fa.T, L + C]]
        for r in range(3):
            for c in range(3):
                J[r * K:(r + 1) * K, c * K:(c + 1) * K] = blocks[r][c]
        mean = torch.cat([bank["nu"][a], bank["omega"][a], bank["nu"][a]])
        S_ab, S_bb = J[:2 * K, 2 * K:], J[2 * K:, 2 * K:]
        cond_mean = mean[:2 * K] + S_ab @ torch.linalg.solve(S_bb, xi - mean[2 * K:])
        cond_cov = J[:2 * K, :2 * K] - S_ab @ torch.linalg.solve(S_bb, S_ab.T)
        means.append(cond_mean[K:])
        covs.append(cond_cov[K:, K:])
        record = torch.distributions.MultivariateNormal(bank["nu"][a], covariance_matrix=L + C)
        logs.append(log_pi_hat[a] + record.log_prob(xi))
    log_unnorm = torch.stack(logs)
    log_evidence = torch.logsumexp(log_unnorm, 0)
    w = torch.exp(log_unnorm - log_evidence)
    m = torch.stack(means)
    mbar = w @ m
    cov = sum(w[a] * (covs[a] + torch.outer(m[a] - mbar, m[a] - mbar)) for a in range(M))
    return mbar, cov, log_unnorm - log_evidence, log_evidence


def _low_rank(F: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Exact U V^T = F - I with r = K (SVD), for the dense/low-rank equivalence."""
    U, S, Vh = torch.linalg.svd(F - torch.eye(F.shape[-1], dtype=DT))
    return U * S.unsqueeze(-2), Vh.transpose(-1, -2)


@pytest.mark.parametrize("representation", ("dense", "low_rank"))
def test_kernel_matches_independent_joint_conditioning(representation: str) -> None:
    K, M, T = 3, 5, 4
    bank = _bank(K, M, seed=1)
    F = _rotation_and_reversal(K, M, seed=2)
    xi = torch.randn(T, K, generator=_gen(3), dtype=DT)
    C = _spd((T,), K, seed=4)                                           # correlated record covariances
    reg = ({"regression_offset": F - torch.eye(K, dtype=DT)} if representation == "dense"
           else dict(zip(("regression_left", "regression_right"), _low_rank(F))))
    out = cue_result_exact_posterior(xi, C, **bank, **reg)
    assert out.mean.shape == (T, K) and out.covariance.shape == (T, K, K) and out.log_weights.shape == (T, M)
    for t in range(T):
        mean, cov, log_w, log_z = _reference(xi[t], C[t], bank, F)
        _close(out.mean[t], mean, TOL, "result mean")
        _close(out.covariance[t], cov, TOL, "result covariance")
        _close(out.log_weights[t], log_w, TOL, "log weights")
        _close(out.log_evidence[t], log_z, TOL, "log evidence")
    assert bool((torch.linalg.eigvalsh(out.covariance) > 0).all())


def test_chunking_does_not_change_the_result() -> None:
    K, M, T = 3, 7, 5
    bank = _bank(K, M, seed=11)
    reg = {"regression_offset": 0.3 * torch.randn(M, K, K, generator=_gen(12), dtype=DT)}
    xi = torch.randn(T, K, generator=_gen(13), dtype=DT)
    C = _spd((T,), K, seed=14)
    whole = cue_result_exact_posterior(xi, C, **bank, **reg)
    for tile in (1, 6 * T * K * K * 8 * 2, 6 * T * K * K * 8 * 3):       # 1, 2 and 3 components per chunk
        tiled = cue_result_exact_posterior(xi, C, **bank, **reg, tile_bytes=tile)
        for field in ("mean", "covariance", "log_weights", "log_evidence"):
            _close(getattr(tiled, field), getattr(whole, field), 1e-12, f"tile {tile}: {field}")


def test_same_state_reduction_matches_prototype_kernel() -> None:
    r"""With omega = nu and F = I the result mean is the same-state posterior mean, and the covariance is the
    same-state posterior covariance plus the weighted result noise sum_a w_a Q_a."""
    K, M, T = 3, 4, 3
    bank = _bank(K, M, seed=21)
    bank["omega"] = bank["nu"].clone()
    xi = torch.randn(T, K, generator=_gen(22), dtype=DT)
    C = _spd((T,), K, seed=23)
    out = cue_result_exact_posterior(xi, C, **bank, regression_offset=torch.zeros(M, K, K, dtype=DT))
    proto = prototype_exact_posterior(xi, C, bank["nu"], bank["log_lambda"], bank["log_pi"], need_covariance=True)
    _close(out.mean, proto.xi_plus, TOL, "same-state mean")
    _close(out.log_weights.exp(), proto.weights, TOL, "same-state weights")
    noise = out.log_weights.exp() @ bank["log_result_noise"].exp()     # (T, K) sum_a w_a diag Q_a
    _close(out.covariance, proto.C_plus + torch.diag_embed(noise), TOL, "same-state covariance plus noise")


def test_zero_regression_gives_input_independent_component_means() -> None:
    r"""F = 0 (offset -I): every component mean is omega_a whatever xi is; the weights still depend on xi."""
    K, M = 2, 3
    bank = _bank(K, M, seed=31)
    reg = {"regression_offset": -torch.eye(K, dtype=DT).expand(M, K, K).clone()}
    C = _spd((2,), K, seed=32)
    xi = torch.stack([torch.zeros(K, dtype=DT), torch.full((K,), 2.0, dtype=DT)])
    out = cue_result_exact_posterior(xi, C, **bank, **reg)
    w = out.log_weights.exp()
    _close(out.mean, w @ bank["omega"], TOL, "means are weighted omegas")
    assert float((w[0] - w[1]).abs().max()) > 1e-3


@pytest.mark.parametrize("representation", ("dense", "low_rank"))
def test_gradcheck_float64(representation: str) -> None:
    K, M, T, r = 2, 3, 2, 1
    bank = _bank(K, M, seed=41)
    xi = torch.randn(T, K, generator=_gen(42), dtype=DT)
    A = 0.5 * torch.randn(T, K, K, generator=_gen(43), dtype=DT)
    reg = ([0.3 * torch.randn(M, K, K, generator=_gen(44), dtype=DT)] if representation == "dense"
           else [0.3 * torch.randn(M, K, r, generator=_gen(45), dtype=DT),
                 0.3 * torch.randn(M, K, r, generator=_gen(46), dtype=DT)])
    names = ("nu", "omega", "log_lambda", "log_result_noise", "log_pi")
    inputs = [xi, A, *(bank[n] for n in names), *reg]
    inputs = [t.clone().requires_grad_(True) for t in inputs]

    def f(xi_, A_, nu, omega, log_lambda, log_q, log_pi, *reg_):
        C_ = A_ @ A_.transpose(-1, -2) + 0.5 * torch.eye(K, dtype=DT)
        kw = ({"regression_offset": reg_[0]} if representation == "dense"
              else {"regression_left": reg_[0], "regression_right": reg_[1]})
        out = cue_result_exact_posterior(xi_, C_, nu, omega, log_lambda, log_q, log_pi, **kw)
        return out.mean, out.covariance, out.log_evidence

    assert torch.autograd.gradcheck(f, tuple(inputs), eps=1e-6, atol=1e-7, rtol=1e-5)


def test_dense_and_low_rank_agree_and_validate_shapes() -> None:
    K, M = 3, 2
    bank = _bank(K, M, seed=51)
    F = _rotation_and_reversal(K, M, seed=52)
    xi, C = torch.randn(2, K, generator=_gen(53), dtype=DT), _spd((2,), K, seed=54)
    dense = cue_result_exact_posterior(xi, C, **bank, regression_offset=F - torch.eye(K, dtype=DT))
    left, right = _low_rank(F)
    low = cue_result_exact_posterior(xi, C, **bank, regression_left=left, regression_right=right)
    _close(low.mean, dense.mean, TOL, "low-rank mean")
    _close(low.covariance, dense.covariance, TOL, "low-rank covariance")
    with pytest.raises(ValueError, match="exactly one"):
        cue_result_exact_posterior(xi, C, **bank)
    with pytest.raises(ValueError, match="exactly one"):
        cue_result_exact_posterior(xi, C, **bank, regression_offset=F, regression_left=left, regression_right=right)
    with pytest.raises(ValueError, match="dense record covariance"):
        cue_result_exact_posterior(xi, torch.ones(2, K, dtype=DT), **bank, regression_offset=F)


# ----------------------------------------------------------------------------------------------
# Block module: initialization, reference-coordinate covariance, model path and configuration
# ----------------------------------------------------------------------------------------------

def _module(K: int, rank: Optional[int], seed: int) -> CueResultMemoryBlockMLP:
    torch.manual_seed(seed)
    mlp = CueResultMemoryBlockMLP(K, 2, 0.0, template_init_variance=2.0, regression_rank=rank).double()
    with torch.no_grad():                                               # move off the identity initialization
        mlp.omega.add_(0.3 * torch.randn(mlp.omega.shape, generator=_gen(seed + 1), dtype=DT))
        if rank is None:
            mlp.regression_offset.add_(0.2 * torch.randn(mlp.regression_offset.shape, generator=_gen(seed + 2),
                                                         dtype=DT))
        else:
            mlp.regression_left.add_(0.2 * torch.randn(mlp.regression_left.shape, generator=_gen(seed + 2), dtype=DT))
    return mlp


@pytest.mark.parametrize("rank", (None, 2))
def test_initialization_is_identity_regression_with_live_gradients(rank: Optional[int]) -> None:
    K = 3
    torch.manual_seed(0)
    mlp = CueResultMemoryBlockMLP(K, 2, 0.0, template_init_variance=2.0, regression_rank=rank,
                                  result_noise_ratio=0.05).double()
    M = 2 * K
    assert torch.equal(mlp.omega, mlp.nu)
    _close(mlp.log_result_noise, mlp.log_lambda + math.log(0.05), 1e-6, "Q = rho L (built in float32)")
    names = [n for n, _ in mlp.named_parameters()]
    tail = ["regression_offset"] if rank is None else ["regression_left", "regression_right"]
    assert names == ["nu", "log_lambda", "log_pi", "omega", "log_result_noise", *tail]
    xi, C = torch.randn(4, K, generator=_gen(61), dtype=DT), _spd((4,), K, seed=62)
    out = mlp.forward_moments(xi, C)
    proto = prototype_exact_posterior(xi, C, mlp.nu, mlp.log_lambda, mlp.log_pi, need_covariance=True)
    _close(out.mu, proto.xi_plus, TOL, "identity initialization is the same-state mean")
    (out.mu.square().sum() + out.sigma.diagonal(dim1=-2, dim2=-1).sum()).backward()
    for name, p in mlp.named_parameters():
        assert p.grad is not None and bool(torch.isfinite(p.grad).all()), name
    if rank is None:
        assert float(mlp.regression_offset.grad.abs().max()) > 0.0
    else:                                    # U = 0, V != 0: U moves at once, V once U is nonzero
        assert float(mlp.regression_left.grad.abs().max()) > 0.0
        assert float(mlp.regression_right.grad.abs().max()) == 0.0
    assert sum(p.numel() for p in mlp.parameters()) == get_block_mlp_registration(
        "cue_result_memory").parameter_count(embed_dim=K, irrep_dims=(K,), expansion=2, regression_rank=rank)
    assert M == mlp.nu.shape[0]


def _frame(K: int, seed: int, scale: float = 0.4) -> Tuple[CanonicalFrameContext, torch.Tensor]:
    U = torch.linalg.matrix_exp(scale * torch.randn(2, 3, K, K, generator=_gen(seed), dtype=DT))
    return CanonicalFrameContext(U, torch.linalg.inv(U), block_diagonal=False), U


def _run(mlp: CueResultMemoryBlockMLP, frame: CanonicalFrameContext, mu: torch.Tensor,
         sigma: torch.Tensor) -> BeliefState:
    belief = BeliefState(mu=mu, sigma=sigma, phi=torch.zeros(2, 3, 1, dtype=DT))
    return _reference_coordinate_block(belief, block_norm=None, block_mlp=mlp, frame=frame, capture=None,
                                       hidden_bias=None)


@pytest.mark.parametrize("rank", (None, 2))
def test_reference_block_conjugation_and_passive_covariance(rank: Optional[int]) -> None:
    K = 4
    mlp = _module(K, rank, seed=71)
    frame, U = _frame(K, seed=72)
    g_frame, g = _frame(K, seed=73, scale=0.7)                            # nonorthogonal GL(K) elements
    mu = torch.randn(2, 3, K, generator=_gen(74), dtype=DT)
    Sigma = _spd((2, 3), K, seed=75)
    U_inv = torch.linalg.inv(U)
    xi = (U_inv @ mu.unsqueeze(-1)).squeeze(-1)
    C = U_inv @ Sigma @ U_inv.transpose(-1, -2)
    direct = mlp.posterior(xi, C)
    base = _run(mlp, frame, mu, Sigma)
    _close(base.mu, (U @ direct.mean.unsqueeze(-1)).squeeze(-1), TOL, "mu_plus = U m_Y")
    _close(base.sigma, U @ direct.covariance @ U.transpose(-1, -2), TOL, "Sigma_plus = U P_Y U^T")
    moved_frame = CanonicalFrameContext(g_frame.forward @ frame.forward, frame.inverse @ g_frame.inverse,
                                        block_diagonal=False)
    moved = _run(mlp, moved_frame, (g @ mu.unsqueeze(-1)).squeeze(-1), g @ Sigma @ g.transpose(-1, -2))
    _close(moved.mu, (g @ base.mu.unsqueeze(-1)).squeeze(-1), TOL, "mu_plus covariant")
    _close(moved.sigma, g @ base.sigma @ g.transpose(-1, -2), TOL, "Sigma_plus covariant")
    # Negative control: the same module applied to local moments is not covariant, so the test has power.
    local = mlp.forward_moments((g @ mu.unsqueeze(-1)).squeeze(-1), g @ Sigma @ g.transpose(-1, -2)).mu
    local_base = mlp.forward_moments(mu, Sigma).mu
    assert float((local - (g @ local_base.unsqueeze(-1)).squeeze(-1)).abs().max()) > 1e-3


_ROUTE = dict(attention_mode="projected_predictive", predictive_readout="minimal_agent",
              minimal_agent_inference="exact", family="gaussian_full", decode_mode="full_chunked",
              norm_type_block="none", block_coordinates="reference", use_block_mlp=True,
              block_mlp_mode="cue_result_memory", block_mlp_inference="exact", block_mlp_covariance="posterior",
              block_mlp_dropout=0.0, block_mlp_expansion=2, n_layers=2, max_seq_len=4)


def _cfg(**overrides):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return test_config(**{**_ROUTE, **overrides})


@pytest.mark.parametrize("rank", (None, 2))
def test_configured_model_path_groups_and_metadata(rank: Optional[int]) -> None:
    from vfe3.train import build_optimizer

    cfg = _cfg(block_memory_regression_rank=rank)
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(cfg)
    assert all(type(m) is CueResultMemoryBlockMLP for m in model.block_mlps)
    realized = sum(p.numel() for p in model.block_mlps.parameters())
    assert model.executable_build.block_mlp.parameter_count == realized
    meta = block_mlp_build_metadata(enabled=True, mode="cue_result_memory", covariance="posterior",
                                    expansion=2, activation="gelu", dropout=0.0, covariance_floor=1e-4,
                                    embed_dim=cfg.embed_dim, irrep_dims=(cfg.embed_dim,), n_layers=cfg.n_layers,
                                    regression_rank=rank)
    assert meta.parameter_count == realized
    optimizer = build_optimizer(model, cfg)
    undecayed = {id(p) for m in model.block_mlps for p in m.undecayed_parameters()}
    holders = [g for g in optimizer.param_groups if {id(p) for p in g["params"]} & undecayed]
    assert len(holders) == 1 and {id(p) for p in holders[0]["params"]} == undecayed
    assert holders[0]["weight_decay"] == 0.0
    decayed = [g for g in optimizer.param_groups if g.get("lr_aux_role") == "block_mlp"]
    assert len(decayed) == 1 and decayed[0]["weight_decay"] == cfg.weight_decay
    assert {id(p) for p in decayed[0]["params"]} == {id(p) for p in model.block_mlps.parameters()} - undecayed
    tokens = torch.randint(0, cfg.vocab_size, (2, 4), generator=_gen(81))
    model.train()
    _, loss, _ = model(tokens, tokens)
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in model.named_parameters():
        if name.startswith("block_mlps.") and not name.endswith("regression_right"):
            assert p.grad is not None and bool(torch.isfinite(p.grad).all()), name
            assert float(p.grad.abs().max()) > 0.0, name


def test_working_variance_calibration_runs_on_the_new_block() -> None:
    cfg = _cfg(block_mlp_template_init="working_variance")
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (3, 4), generator=_gen(91))
    stats = model.calibrate_prototype_templates(tokens, pull=0.2, seed=5)
    assert len(stats) == cfg.n_layers
    for mlp, s in zip(model.block_mlps, stats):
        _close(mlp.log_lambda, torch.full_like(mlp.log_lambda, math.log(s["lambda0"])), 1e-6, "lambda_0")
        assert torch.equal(mlp.omega, mlp.nu)


def test_configuration_restrictions_and_inactive_fields() -> None:
    cfg = _cfg()
    inactive = set(cfg.inactive_fields())
    assert {"block_mlp_inference", "block_mlp_cavi_steps", "block_mlp_em_weight", "block_mlp_activation"} <= inactive
    assert not ({"block_memory_regression_rank", "block_memory_result_noise_ratio",
                 "block_mlp_template_init"} & inactive)
    assert {"block_memory_regression_rank", "block_memory_result_noise_ratio"} <= set(
        test_config().inactive_fields())
    bad = [dict(block_coordinates="local"), dict(family="gaussian_diagonal", decode_mode="linear"),
           dict(block_mlp_inference="mean_field"), dict(block_mlp_dropout=0.1), dict(block_mlp_em_weight=0.1)]
    for over in bad:
        with pytest.raises(ValueError):
            _cfg(**over)
    with pytest.raises(ValueError):
        _cfg(block_mlp_covariance="passthrough")
    for rank in (0, cfg.embed_dim + 1, True, 1.5):
        with pytest.raises(ValueError, match="block_memory_regression_rank"):
            _cfg(block_memory_regression_rank=rank)
    for ratio in (0.0, -1.0, float("nan"), True):
        with pytest.raises(ValueError, match="block_memory_result_noise_ratio"):
            _cfg(block_memory_result_noise_ratio=ratio)
    with pytest.raises(ValueError):
        build_block_mlp("cue_result_memory", embed_dim=3, irrep_dims=(3,), expansion=2, activation="gelu",
                        dropout=0.0, covariance_contract="passthrough", covariance_floor=1e-4)
