r"""Tests for the cue/result regime belief carried through depth (``block_memory_regime_transition``).

The block is driven layer by layer through a ``regime_state`` dict exactly as the stack does, and compared with an
explicit hidden-Markov recursion written here from the definitions: the transition matrix :math:`T` is formed
explicitly (sticky :math:`sI + (1 - s)\mathbf 1\hat\pi^\top`, or the row softmax of the learned logits), the record densities
come from ``torch.distributions.MultivariateNormal``, and layer :math:`\ell+1`'s posterior is
:math:`\propto\hat\pi_b(w^\ell T)_b^\beta\,\mathcal N(\xi^{\ell+1};\nu_b, L_b + C^{\ell+1})`. The tests also cover the per-token prior of
both kernels, the :math:`\beta = 0` reduction to the shared prior, the transition initialization, the configured model
path (dense and frame-diagonal), parameter counts, optimizer groups and the configuration. CPU only, one thread.
"""

from __future__ import annotations

import math
import warnings
from typing import List, Optional

import pytest
import torch

from tests.profiles import test_config
from vfe3.model.block_mlp import CueResultMemoryBlockMLP
from vfe3.model.cue_result_memory import cue_result_diagonal_posterior, cue_result_exact_posterior
from vfe3.model.model import VFEModel
from vfe3.train import build_optimizer

torch.set_num_threads(1)
DT = torch.float64
TOL = 1e-11


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _close(a: torch.Tensor, b: torch.Tensor, tol: float, what: str) -> None:
    a, b = a.detach(), b.detach()
    err = float((a - b).abs().max())
    scale = max(1.0, float(b.abs().max()))
    assert err <= tol * scale, f"{what}: max error {err:.3e} (scale {scale:.3e})"


def _spd(batch, K: int, seed: int) -> torch.Tensor:
    A = 0.5 * torch.randn(*batch, K, K, generator=_gen(seed), dtype=DT)
    return A @ A.transpose(-1, -2) + 0.5 * torch.eye(K, dtype=DT)


@pytest.mark.parametrize("diagonal", (False, True))
def test_per_token_prior_equals_rowwise_shared_prior(diagonal: bool) -> None:
    K, M, T = 3, 5, 4
    g = _gen(1)
    nu, omega = 1.5 * torch.randn(M, K, generator=g, dtype=DT), torch.randn(M, K, generator=g, dtype=DT)
    log_lambda, log_q = 0.4 * torch.randn(M, K, generator=g, dtype=DT), 0.4 * torch.randn(M, K, generator=g, dtype=DT)
    left, right = 0.5 * torch.randn(M, K, 2, generator=g, dtype=DT), 0.5 * torch.randn(M, K, 2, generator=g, dtype=DT)
    xi = torch.randn(T, K, generator=g, dtype=DT)
    C = 0.3 + torch.rand(T, K, generator=g, dtype=DT) if diagonal else _spd((T,), K, seed=2)
    log_pi = torch.randn(T, M, generator=g, dtype=DT)                         # one prior per token
    kernel = cue_result_diagonal_posterior if diagonal else cue_result_exact_posterior
    reg = dict(regression_left=left, regression_right=right)
    for tile in (10 ** 9, 200):
        joint = kernel(xi, C, nu, omega, log_lambda, log_q, log_pi, **reg, tile_bytes=tile)
        for t in range(T):
            row = kernel(xi[t:t + 1], C[t:t + 1], nu, omega, log_lambda, log_q, log_pi[t], **reg)
            for field in ("mean", "covariance", "log_weights", "log_evidence"):
                _close(getattr(joint, field)[t], getattr(row, field)[0], TOL, f"tile {tile}, token {t}: {field}")
    with pytest.raises(ValueError, match="log_pi"):
        kernel(xi, C, nu, omega, log_lambda, log_q, log_pi[:, :-1], **reg)


def _module(K: int, transition: str, beta: float, seed: int) -> CueResultMemoryBlockMLP:
    torch.manual_seed(seed)
    mlp = CueResultMemoryBlockMLP(K, 2, 0.0, template_init_variance=1.5, regression_rank=2,
                                  regime_transition=transition, regime_beta=beta, regime_stay=0.7).double()
    with torch.no_grad():
        g = _gen(seed + 1)
        mlp.nu.mul_(4.0)
        mlp.log_pi.copy_(torch.randn(mlp.log_pi.shape, generator=g, dtype=DT))
        mlp.regression_left.normal_(0.0, 0.3, generator=g)
        if transition == "learned":
            mlp.regime_transition_logits.add_(0.5 * torch.randn(mlp.regime_transition_logits.shape, generator=g,
                                                                dtype=DT))
    return mlp


def _transition(mlp: CueResultMemoryBlockMLP) -> torch.Tensor:
    r"""The explicit (M, M) row-stochastic T of the module's declared transition."""
    M = mlp.nu.shape[0]
    pi = torch.softmax(mlp.log_pi.detach(), -1)
    if mlp.regime_transition == "sticky":
        s = float(torch.sigmoid(mlp.regime_stay_logit.detach()))
        return s * torch.eye(M, dtype=DT) + (1.0 - s) * pi.expand(M, M)
    return torch.softmax(mlp.regime_transition_logits.detach(), -1)


@pytest.mark.parametrize("transition", ("sticky", "learned"))
@pytest.mark.parametrize("beta", (1.0, 0.4))
def test_block_through_depth_matches_explicit_hmm_recursion(transition: str, beta: float) -> None:
    K, L, T = 2, 3, 3
    mlp = _module(K, transition, beta, seed=10)
    M = mlp.nu.shape[0]
    xis = [torch.randn(T, K, generator=_gen(20 + l), dtype=DT) for l in range(L)]
    Cs = [_spd((T,), K, seed=30 + l) for l in range(L)]
    state: dict = {}
    got: List[torch.Tensor] = []
    for l in range(L):                                                       # as vfe_stack drives the blocks
        mlp.forward_moments(xis[l], Cs[l], regime_state=state)
        got.append(state["current"])
        state["previous"] = state.pop("current", None)
    pi = torch.softmax(mlp.log_pi.detach(), -1)
    Tm = _transition(mlp)
    lam = mlp.log_lambda.detach().exp()
    prev: Optional[torch.Tensor] = None
    for l in range(L):
        log_prior = pi.log().expand(T, M) if prev is None else pi.log() + beta * (prev.exp() @ Tm).log()
        log_like = torch.stack([torch.distributions.MultivariateNormal(mlp.nu[b].detach(), Cs[l] + torch.diag(lam[b]))
                                .log_prob(xis[l]) for b in range(M)], dim=-1)                  # (T, M)
        ref = torch.log_softmax(log_prior + log_like, dim=-1)
        _close(got[l], ref, 1e-10, f"layer {l} log weights")
        prev = ref
    # The carried prior changes the result: the test has power.
    plain = mlp.posterior(xis[1], Cs[1]).log_weights
    assert float((plain - got[1]).abs().max()) > 1e-3


def test_beta_zero_recovers_the_shared_prior() -> None:
    K = 2
    mlp = _module(K, "sticky", 0.0, seed=40)
    xi, C = torch.randn(3, K, generator=_gen(41), dtype=DT), _spd((3,), K, seed=42)
    state = {"previous": torch.log_softmax(torch.randn(3, mlp.nu.shape[0], generator=_gen(43), dtype=DT), -1)}
    carried = mlp.forward_moments(xi, C, regime_state=state)
    shared = mlp.forward_moments(xi, C)
    _close(carried.mu, shared.mu, TOL, "mean")
    _close(carried.sigma, shared.sigma, TOL, "covariance")


def test_transition_initialization_and_counts() -> None:
    K, s = 3, 0.8
    torch.manual_seed(0)
    sticky = CueResultMemoryBlockMLP(K, 2, 0.0, regression_rank=1, regime_transition="sticky", regime_stay=s)
    learned = CueResultMemoryBlockMLP(K, 2, 0.0, regression_rank=1, regime_transition="learned", regime_stay=s)
    off = CueResultMemoryBlockMLP(K, 2, 0.0, regression_rank=1)
    M = 2 * K
    _close(torch.sigmoid(sticky.regime_stay_logit), torch.tensor(s), 1e-6, "stay probability")
    T = torch.softmax(learned.regime_transition_logits, -1)
    _close(T, s * torch.eye(M) + (1 - s) / M, 1e-6, "learned T at sI + (1-s)/M")
    n = lambda m: sum(p.numel() for p in m.parameters())                     # noqa: E731
    assert n(sticky) == n(off) + 1 and n(learned) == n(off) + M * M
    assert any(p is sticky.regime_stay_logit for p in sticky.undecayed_parameters())
    assert any(p is learned.regime_transition_logits for p in learned.undecayed_parameters())
    with pytest.raises(ValueError):
        CueResultMemoryBlockMLP(K, 2, 0.0, regime_transition="hmm")
    for bad in (-0.1, float("nan"), True):
        with pytest.raises(ValueError):
            CueResultMemoryBlockMLP(K, 2, 0.0, regime_beta=bad)
    for bad in (0.0, 1.0, True):
        with pytest.raises(ValueError):
            CueResultMemoryBlockMLP(K, 2, 0.0, regime_stay=bad)


_DENSE = dict(attention_mode="projected_predictive", predictive_readout="minimal_agent",
              minimal_agent_inference="exact", family="gaussian_full", decode_mode="full_chunked",
              norm_type_block="none", block_coordinates="reference", use_block_mlp=True,
              block_mlp_mode="cue_result_memory", block_mlp_inference="exact", block_mlp_covariance="posterior",
              block_mlp_dropout=0.0, block_mlp_expansion=2, n_layers=3, max_seq_len=4,
              block_memory_regression_rank=2, block_mlp_tie_layers=True)
_DIAG = dict(_DENSE, predictive_readout="kernel", family="gaussian_frame_diagonal", decode_mode="linear")


def _cfg(base: dict, **overrides):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return test_config(**{**base, **overrides})


@pytest.mark.parametrize("route", ("dense", "diagonal"))
@pytest.mark.parametrize("transition", ("sticky", "learned"))
def test_configured_model_path(route: str, transition: str) -> None:
    cfg = _cfg(_DENSE if route == "dense" else _DIAG, block_memory_regime_transition=transition,
               block_memory_regime_beta=0.5)
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(cfg)
    mlp = model.block_mlps[0]
    assert all(m is mlp for m in model.block_mlps) and mlp.regime_transition == transition
    assert model.executable_build.block_mlp.parameter_count == sum(p.numel() for p in mlp.parameters())
    optimizer = build_optimizer(model, cfg)
    seen = [id(p) for g in optimizer.param_groups for p in g["params"]]
    assert len(seen) == len(set(seen))
    regime_param = mlp.regime_stay_logit if transition == "sticky" else mlp.regime_transition_logits
    holders = [g for g in optimizer.param_groups if any(p is regime_param for p in g["params"])]
    assert len(holders) == 1 and holders[0]["weight_decay"] == 0.0
    tokens = torch.randint(0, cfg.vocab_size, (2, 4), generator=_gen(50))
    model.train()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, loss, _ = model(tokens, tokens)
    assert torch.isfinite(loss)
    loss.backward()
    assert regime_param.grad is not None and float(regime_param.grad.abs().max()) > 0.0
    for name, p in model.named_parameters():
        if name.startswith("block_mlps.") and p.grad is not None:
            assert bool(torch.isfinite(p.grad).all()), name


def test_model_with_beta_zero_equals_regime_off() -> None:
    tokens = torch.randint(0, 7, (2, 4), generator=_gen(60))
    losses = []
    for over in (dict(), dict(block_memory_regime_transition="sticky", block_memory_regime_beta=0.0)):
        cfg = _cfg(_DIAG, **over)
        torch.manual_seed(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = VFEModel(cfg)
            model.eval()
            _, loss, _ = model(tokens, tokens)
        losses.append(loss.detach())
    _close(losses[1], losses[0], 1e-6, "beta = 0 equals off")


def test_configuration() -> None:
    cfg = _cfg(_DIAG, block_memory_regime_transition="sticky")
    inactive = set(cfg.inactive_fields())
    assert not ({"block_memory_regime_transition", "block_memory_regime_beta", "block_memory_regime_stay"} & inactive)
    assert {"block_memory_regime_beta", "block_memory_regime_stay"} <= set(_cfg(_DIAG).inactive_fields())
    assert "block_memory_regime_transition" in set(test_config().inactive_fields())
    with pytest.raises(ValueError, match="tied banks"):
        _cfg(_DIAG, block_memory_regime_transition="sticky", block_mlp_tie_layers=False)
    with pytest.raises(ValueError, match="block_memory_regime_transition"):
        _cfg(_DIAG, block_memory_regime_transition="hmm")
    for bad in (-1.0, float("inf"), True):
        with pytest.raises(ValueError, match="block_memory_regime_beta"):
            _cfg(_DIAG, block_memory_regime_beta=bad)
    for bad in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError, match="block_memory_regime_stay"):
            _cfg(_DIAG, block_memory_regime_stay=bad)
