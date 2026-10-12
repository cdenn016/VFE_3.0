r"""Numeric-policy toggles exposed on 2026-10-04.

The frame-pair float64 escalation is opt-in again (``tests/test_frame_pair_escalation_20260923.py``).
This file covers the projected source-recovery precision and the certificate tolerances (module
policies that the merged ``projected_factorization`` fixes at fp32, 1e-4 and 3e-4 since 2026-10-12),
and the bit-identical ``predictive_covariance_checkpoint``. The recovery fixture is the rank-ill source of
``tests/test_relational_source_factorization.py``: S has singular values (1, 0.5, 1e-4), so
kappa(X) = 1e4 and the rounded float32 Gram G = X X^T (kappa 1e8) loses SPD.
"""
from typing import Dict, Iterator, Tuple

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import vfe3.model.relational_operators as relational_operators
import vfe3.relational_scoring as scoring
from tests.profiles import test_config, tiny_config
from tests.test_relational_source_factorization import _inputs
from vfe3.model.model import VFEModel
from vfe3.numerics import triangular_condition_estimate
from vfe3.relational_scoring import (
    _cholesky_status, _factor_status, projected_gaussian_kl, projected_moments,
    reset_source_recovery_elements, set_projected_condition_recovery, set_projected_factor_route,
    set_projected_recovery_precision, source_recovery_elements,
)


@pytest.fixture(autouse=True)
def _restore_scorer_policies() -> Iterator[None]:
    saved = (scoring._CONDITION_RECOVERY, scoring._FACTOR_ROUTE, scoring._RECOVERY_PRECISION)
    yield
    scoring._CONDITION_RECOVERY, scoring._FACTOR_ROUTE, scoring._RECOVERY_PRECISION = saved


class _Ops(TorchDispatchMode):
    r"""Record dispatched op names and the ops that produced a float64 tensor."""

    def __init__(self) -> None:
        super().__init__()
        self.names: set = set()
        self.float64: set = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = func.overloadpacket.__name__
        self.names.add(name)
        if any(isinstance(o, torch.Tensor) and o.dtype == torch.float64
               for o in (out if isinstance(out, (tuple, list)) else (out,))):
            self.float64.add(name)
        return out


def _case(
    device: torch.device,
    role:   str,                 # which comparison map carries the ill-conditioned S
    full:   bool,                # full (d, d) source covariance instead of diagonal
    S:      torch.Tensor,        # (1, 3, 3) the rank-ill map, possibly a grad leaf
) -> Tuple[torch.Tensor, ...]:
    mu, sigma, inverse, _ = _inputs(device)
    if full:
        sigma = torch.diag_embed(sigma)
    identity = torch.eye(3, device=device)[None]
    S_q, S_k = (S, identity) if role == "q" else (identity, S)
    aq, Gq = projected_moments(mu, sigma, inverse, S_q)
    ak, Gk = projected_moments(mu, sigma, inverse, S_k)
    return aq, Gq, ak, Gk, (sigma, inverse, S_q, mu), (sigma, inverse, S_k, mu)


def _reference(device: torch.device, role: str, full: bool) -> torch.Tensor:
    _, _, _, S = _inputs(device)
    aq, Gq, ak, Gk, source_q, source_k = _case(device, role, full, S)
    as64 = [tuple(x.double() for x in source) for source in (source_q, source_k)]
    (sigma_q, inverse_q, S_q, mu_q), (sigma_k, inverse_k, S_k, mu_k) = as64
    return projected_gaussian_kl(*projected_moments(mu_q, sigma_q, inverse_q, S_q),
                                 *projected_moments(mu_k, sigma_k, inverse_k, S_k))


# -- config surface -------------------------------------------------------------------------------

def test_fields_validate_and_report_inactivity() -> None:
    cfg = test_config()
    assert cfg.projected_factorization == "gram"
    assert cfg.predictive_covariance_checkpoint is False
    with pytest.raises(ValueError, match="projected_factorization"):
        test_config(projected_factorization="cholesky_qr2")
    with pytest.raises(ValueError, match="predictive_covariance_checkpoint"):
        test_config(predictive_covariance_checkpoint=1)
    assert "predictive_covariance_checkpoint" in test_config(
        predictive_covariance="passthrough").inactive_fields()
    assert "predictive_covariance_checkpoint" not in tiny_config(
        attention_mode="projected_predictive", predictive_covariance="moment_matched").inactive_fields()


@pytest.mark.parametrize(("factorization", "recovery", "route"), [
    ("gram", False, "gram"), ("gram_certified", True, "gram"), ("cholesky_qr2_certified", True, "cholesky_qr2")])
def test_model_installs_the_scorer_policies(factorization: str, recovery: bool, route: str) -> None:
    set_projected_recovery_precision("fp64")
    VFEModel(tiny_config(projected_factorization=factorization))
    assert scoring.projected_condition_recovery() is recovery
    assert scoring.projected_factor_route() == route
    assert scoring.projected_recovery_precision() == "fp32"
    assert scoring.GRAM_CONDITION_TOL == 1e-4 and scoring.FACTOR_CONDITION_TOL == 3e-4
    with pytest.raises(ValueError, match="recovery precision"):
        set_projected_recovery_precision("bf16")


# -- the certificate tolerances reach the certificates ----------------------------------------------

def test_gram_tolerance_moves_the_certificate(device: torch.device, monkeypatch: pytest.MonkeyPatch) -> None:
    set_projected_condition_recovery(True)
    G = torch.diag(torch.tensor([1.0, 1e-4], device=device))[None]    # eps_32 kappa_F(G) ~ 1.2e-3
    assert bool(_cholesky_status(G)[2].all())
    monkeypatch.setattr(scoring, "GRAM_CONDITION_TOL", 1e-2)
    assert not bool(_cholesky_status(G)[2].any())


def test_factor_tolerance_moves_the_certificate(device: torch.device, monkeypatch: pytest.MonkeyPatch) -> None:
    set_projected_condition_recovery(True)
    generator = torch.Generator().manual_seed(4)
    Q = torch.linalg.qr(torch.randn(3, 3, generator=generator))[0]
    R = torch.linalg.qr(torch.randn(3, 3, generator=generator))[0]
    S = (Q @ torch.diag(torch.tensor([1., .5, 1e-3])) @ R.T)[None].to(device)   # kappa(X) = 1e3
    mu, sigma, inverse, _ = _inputs(device)
    _, G = projected_moments(mu, sigma, inverse, S)
    projection = (sigma, inverse, S, mu)
    chol, failed, _, _ = _factor_status(G, projection)
    assert not bool(failed.any())                                    # CholeskyQR2 succeeds
    measured = torch.finfo(torch.float32).eps * triangular_condition_estimate(chol)
    monkeypatch.setattr(scoring, "FACTOR_CONDITION_TOL", 0.5 * float(measured.min()))
    assert bool(_factor_status(G, projection)[2].all())
    monkeypatch.setattr(scoring, "FACTOR_CONDITION_TOL", 2.0 * float(measured.max()))
    assert not bool(_factor_status(G, projection)[2].any())


# -- recovery precision -----------------------------------------------------------------------------

@pytest.mark.parametrize("role", ["q", "k"])
@pytest.mark.parametrize("full", [False, True])
def test_fp32_recovery_stays_in_the_working_dtype(device: torch.device, role: str, full: bool) -> None:
    reference = _reference(device, role, full)
    for precision in ("fp64", "fp32"):
        set_projected_recovery_precision(precision)
        S = _inputs(device)[3].clone().requires_grad_(True)
        aq, Gq, ak, Gk, source_q, source_k = _case(device, role, full, S)
        assert torch.linalg.cholesky_ex(Gq if role == "q" else Gk)[1].ne(0).any()   # lost SPD
        reset_source_recovery_elements()
        ops = _Ops()
        with ops:
            score = projected_gaussian_kl(aq, Gq, ak, Gk, projection_q=source_q, projection_k=source_k)
        assert source_recovery_elements() > 0
        assert score.dtype == torch.float32
        assert ("linalg_qr" in ops.names) is (precision == "fp64")
        if precision == "fp32":
            # Only the per-entry source root is formed in float64; no float64 QR or pair tiles run.
            assert "linalg_solve_triangular" not in ops.float64
        # Measured: 3e-8 (fp64) and <= 6.2e-6 (fp32) relative to |reference| + 1.
        torch.testing.assert_close(score.double(), reference, rtol=5e-5, atol=5e-5)
        score.sum().backward()
        assert torch.isfinite(S.grad).all()


def test_fp32_recovery_falls_back_to_fp64_when_the_retry_fails(device: torch.device,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    set_projected_recovery_precision("fp32")
    real = scoring.cholesky_qr2

    def failing(root: torch.Tensor, shift: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        factor, failed = real(root, shift=shift)
        return factor, (torch.ones_like(failed) if shift else failed)

    monkeypatch.setattr(scoring, "cholesky_qr2", failing)
    aq, Gq, ak, Gk, source_q, source_k = _case(device, "q", False, _inputs(device)[3])
    ops = _Ops()
    with ops:
        score = projected_gaussian_kl(aq, Gq, ak, Gk, projection_q=source_q, projection_k=source_k)
    assert "linalg_qr" in ops.names
    torch.testing.assert_close(score.double(), _reference(device, "q", False), rtol=2e-5, atol=2e-5)


def test_fp32_recovery_keeps_the_factor_route_factor(device: torch.device) -> None:
    set_projected_factor_route("cholesky_qr2")
    set_projected_recovery_precision("fp32")
    aq, Gq, ak, Gk, source_q, source_k = _case(device, "k", False, _inputs(device)[3])
    ops = _Ops()
    with ops:
        score = projected_gaussian_kl(aq, Gq, ak, Gk, projection_q=source_q, projection_k=source_k)
    assert "linalg_qr" not in ops.names and "linalg_solve_triangular" not in ops.float64
    torch.testing.assert_close(score.double(), _reference(device, "k", False), rtol=5e-5, atol=5e-5)


# -- moment-matched covariance checkpoint ---------------------------------------------------------

def test_covariance_checkpoint_is_bit_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    real = relational_operators.checkpoint

    def counting(*args: object, **kwargs: object) -> torch.Tensor:
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(relational_operators, "checkpoint", counting)

    def run(flag: bool) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cfg = tiny_config(attention_mode="projected_predictive", predictive_covariance="moment_matched",
                          attention_residual=True, predictive_covariance_checkpoint=flag)
        torch.manual_seed(0)
        model = VFEModel(cfg)
        tokens = torch.randint(0, cfg.vocab_size, (2, min(cfg.max_seq_len, 6)),
                               generator=torch.Generator().manual_seed(1))
        _, loss, _ = model(tokens, tokens)
        loss.backward()
        return loss.detach(), {name: p.grad.detach().clone()
                               for name, p in model.named_parameters() if p.grad is not None}

    loss_saved, grads_saved = run(False)
    assert not calls
    loss_recomputed, grads_recomputed = run(True)
    assert calls                                          # the update ran under the checkpoint
    assert torch.equal(loss_saved, loss_recomputed)
    assert grads_saved.keys() == grads_recomputed.keys()
    assert all(torch.equal(grads_saved[name], grads_recomputed[name]) for name in grads_saved)


# -- deterministic fill of uninitialized memory -----------------------------------------------------

def test_model_construction_keeps_the_uninitialized_fill_and_training_turns_it_off() -> None:
    from vfe3.runtime import deterministic_state
    from vfe3.train import train
    previous = torch.utils.deterministic.fill_uninitialized_memory
    try:
        torch.utils.deterministic.fill_uninitialized_memory = True
        cfg = tiny_config(max_steps=1, warmup_steps=0)
        model = VFEModel(cfg)
        assert torch.utils.deterministic.fill_uninitialized_memory is True    # tests that build models keep it
        tokens = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.max_seq_len))
        train(model, [(tokens, tokens)], cfg, n_steps=1, log_interval=0, eval_interval=0, generate_samples=False)
        assert torch.utils.deterministic.fill_uninitialized_memory is False
        assert deterministic_state()["fill_uninitialized_memory"] is False
    finally:
        torch.utils.deterministic.fill_uninitialized_memory = previous
