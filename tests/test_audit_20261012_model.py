r"""Model and configuration remediation of the 2026-10-12 deep audit (docs/audits/audit-2026-10-12.md).

Items: 2 (the prototype mean field flags a non-finite or non-positive diagonal power-posterior matrix),
21 (the minimal-agent label step skips the KL symmetry check and the query-factor autograd, bitwise),
C12 (``min_lr_frac > 1`` and a negative ``pos_phi_scale`` are refused). Tiny CPU cases only.
"""
import math
import warnings
from typing import Any, Dict

import pytest
import torch

from tests.profiles import test_config, tiny_config
from vfe3.model import minimal_agent as ma
from vfe3.model import prototype_agents as pa
from vfe3.model.model import VFEModel

torch.set_num_threads(1)
F64 = torch.float64


def _templates(M: int = 3, K: int = 4) -> tuple:
    g = torch.Generator().manual_seed(0)
    return (torch.randn(M, K, generator=g, dtype=F64), torch.zeros(M, K, dtype=F64), torch.zeros(M, dtype=F64))


# ----------------------------------------------------------------------------------------------
# Item 2
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [math.nan, -50.0])
def test_prototype_diagonal_mean_field_flags_a_failed_power_posterior(bad: float) -> None:
    r"""A NaN or negative working variance makes :math:`1 + c_kD_k` non-positive or NaN; the diagonal layout raises
    like the block and dense layouts' Cholesky check instead of returning a NaN mean."""
    nu, log_lambda, log_pi = _templates()
    xi = torch.randn(2, 4, dtype=F64)
    C = torch.ones(2, 4, dtype=F64)
    C[1, 2] = bad
    with pytest.raises(ValueError, match="not positive semidefinite or not finite"):
        pa.prototype_mean_field_posterior(xi, C, nu, log_lambda, log_pi)


def test_prototype_diagonal_mean_field_healthy_values_unchanged() -> None:
    nu, log_lambda, log_pi = _templates()
    xi = torch.randn(2, 4, dtype=F64)
    C = torch.rand(2, 4, dtype=F64) + 0.5
    post = pa.prototype_mean_field_posterior(xi, C, nu, log_lambda, log_pi, cavi_steps=2)
    D = post.D
    P = C / (1.0 + C * D)
    assert torch.equal(post.C_plus, P)
    assert torch.isfinite(post.xi_plus).all()


# ----------------------------------------------------------------------------------------------
# Item 21
# ----------------------------------------------------------------------------------------------

def _spd(*shape: int, r: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(*shape, r, r, generator=g, dtype=F64)
    return ma._symmetrize(X @ X.transpose(-1, -2) + r * torch.eye(r, dtype=F64))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_labels_only_cost_is_bitwise_the_default_label_step(dtype: torch.dtype) -> None:
    r"""Values of the labels, cost and information statistics are bitwise those of the default call; the gradients
    lose only the analytically zero log-determinant term, so they agree to rounding."""
    H, I, r = 2, 5, 3
    g = torch.Generator().manual_seed(3)
    base = dict(a=torch.randn(H, I, r, generator=g, dtype=F64), A=_spd(H, I, r=r, seed=1),
                b=torch.randn(H, I, r, generator=g, dtype=F64), B=_spd(H, I, r=r, seed=2))
    lp = torch.where(                                                       # causal prior
torch.ones(I, I, dtype=torch.bool).tril(), torch.zeros(I, I, dtype=F64), -math.inf)
    outs: Dict[bool, Any] = {}
    for flag in (False, True):
        leaves = {k: v.to(dtype).clone().requires_grad_(True) for k, v in base.items()}
        post = ma.mean_field_record_posterior(leaves["a"], leaves["A"], leaves["b"], leaves["B"], lp.to(dtype),
                                              labels_only_cost=flag)
        w = torch.randn(post.weights.shape, generator=torch.Generator().manual_seed(5), dtype=F64).to(dtype)
        objective = ((post.weights * w).sum() + post.Lambda.square().sum() + post.eta.sin().sum())
        grads = torch.autograd.grad(objective, list(leaves.values()))
        outs[flag] = (post.weights.detach(), post.cost.detach(), post.Lambda.detach(), post.eta.detach(), grads)
    for x, y in zip(outs[False][:4], outs[True][:4]):
        assert torch.equal(x, y)
    rtol = 1e-5 if dtype == torch.float32 else 1e-12
    for x, y in zip(outs[False][4], outs[True][4]):
        assert torch.allclose(x, y, rtol=rtol, atol=rtol * x.abs().max().item())


def test_minimal_agent_model_loss_and_gradients_unchanged_by_item_21(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""The live readout passes ``labels_only_cost=True``; forcing the default gives bitwise the same loss and every
    parameter gradient to float32 rounding (mean field, two sweeps, so the posterior-moment label step is covered
    too)."""
    cfg = tiny_config(predictive_readout="minimal_agent", minimal_agent_inference="mean_field",
                      attention_residual=False, max_seq_len=4, n_layers=2, minimal_agent_cavi_steps=2)
    tokens = torch.randint(0, cfg.vocab_size, (2, 4), generator=torch.Generator().manual_seed(7))
    flags = []
    original = ma.mean_field_record_posterior

    def run(force_default: bool) -> tuple:
        def spy(*args: Any, **kwargs: Any) -> Any:
            flags.append(kwargs.get("labels_only_cost", False))
            if force_default:
                kwargs["labels_only_cost"] = False
            return original(*args, **kwargs)

        monkeypatch.setattr(ma, "mean_field_record_posterior", spy)
        torch.manual_seed(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = VFEModel(cfg)
        _, loss, _ = model(tokens, tokens)
        loss.backward()
        monkeypatch.setattr(ma, "mean_field_record_posterior", original)
        return loss.detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    loss_new, grads_new = run(False)
    assert flags and all(flags)
    loss_old, grads_old = run(True)
    assert torch.equal(loss_new, loss_old)
    assert grads_new.keys() == grads_old.keys()
    for name in grads_new:
        scale = grads_old[name].abs().max().item()
        assert torch.allclose(grads_new[name], grads_old[name], rtol=0.0, atol=1e-6 * scale), name


# ----------------------------------------------------------------------------------------------
# C12
# ----------------------------------------------------------------------------------------------

def test_min_lr_frac_above_one_is_refused() -> None:
    test_config(min_lr_frac=1.0)
    with pytest.raises(ValueError, match="min_lr_frac must be <= 1"):
        test_config(min_lr_frac=1.5)


def test_negative_pos_phi_scale_is_refused_when_pos_phi_is_active() -> None:
    test_config(pos_phi="learned", pos_phi_scale=0.0)
    with pytest.raises(ValueError, match="pos_phi_scale must be finite and >= 0"):
        test_config(pos_phi="learned", pos_phi_scale=-0.02)
    test_config(pos_phi="none", pos_phi_scale=-0.02)            # inert without positional frames
