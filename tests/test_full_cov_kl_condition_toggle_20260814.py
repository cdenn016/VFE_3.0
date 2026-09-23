r"""Performance gate for the optional full-covariance KL condition certificate."""

import importlib
from pathlib import Path

import pytest
import torch

import vfe3.families.gaussian as gaussian_mod
import scaling_analysis
from vfe3.config import VFE3Config
from vfe3.families.gaussian import FullGaussian
from vfe3.model.model import VFEModel


ROOT = Path(__file__).resolve().parents[1]
BASE = dict(vocab_size=16, embed_dim=4, n_heads=2, max_seq_len=4)


def _pair(policy: str, *, ill_conditioned: bool = False):
    if ill_conditioned:
        diagonal = torch.tensor([1.0, 1.0e-3, 1.0e-5, 1.0e-6])
        sigma = torch.diag(diagonal).unsqueeze(0)
    else:
        sigma = torch.eye(4).unsqueeze(0)
    q = FullGaussian(
        torch.zeros(1, 4), sigma, _precision_policy=policy)
    t = FullGaussian(
        torch.full((1, 4), 0.1), sigma, _precision_policy=policy)
    return q, t


def test_default_precision_policy_is_fp32_escalate():
    cfg = VFE3Config(**BASE)
    assert cfg.full_cov_kl_precision == "fp32_escalate"


def test_model_uses_selected_precision_policy():
    common = dict(
        **BASE,
        family="gaussian_full",
        full_cov_kl_precision="fp32_escalate",
        full_cov_congruence_precision="fp32_escalate",
        pos_phi="none",
        e_phi_lr=0.0,
    )
    fast = VFEModel(VFE3Config(**common))
    strict_cfg = dict(common)
    strict_cfg["full_cov_kl_precision"] = "fp32_escalate_cond"
    strict = VFEModel(VFE3Config(**strict_cfg))
    assert fast.cfg.full_cov_kl_precision == "fp32_escalate"
    assert strict.cfg.full_cov_kl_precision == "fp32_escalate_cond"
    assert fast.full_cov_kl_precision == "fp32_escalate"
    assert strict.full_cov_kl_precision == "fp32_escalate_cond"


def test_fast_policy_does_not_run_the_spectral_certificate(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("fast KL path called validated_cholesky_solve")

    monkeypatch.setattr(gaussian_mod, "validated_cholesky_solve", forbidden)
    q, t = _pair("fp32_escalate")
    result = q.renyi_closed_form(t, alpha=1.0, kl_max=1.0e9)
    assert torch.isfinite(result).all()


def test_strict_policy_runs_both_spectral_certificates(monkeypatch):
    calls = []
    original = gaussian_mod.validated_cholesky_solve

    def recording(matrix, *args, **kwargs):
        calls.append(tuple(matrix.shape))
        return original(matrix, *args, **kwargs)

    monkeypatch.setattr(gaussian_mod, "validated_cholesky_solve", recording)
    q, t = _pair("fp32_escalate_cond")
    result = q.renyi_closed_form(t, alpha=1.0, kl_max=1.0e9)
    assert torch.isfinite(result).all()
    assert calls == [(1, 4, 4), (1, 4, 4)]


def test_fast_policy_does_not_promote_an_ill_conditioned_valid_pair():
    q, t = _pair("fp32_escalate", ill_conditioned=True)
    result = q.renyi_closed_form(t, alpha=1.0, kl_max=1.0e9)
    assert result.dtype is torch.float32


def test_strict_policy_promotes_the_same_ill_conditioned_pair():
    q, t = _pair("fp32_escalate_cond", ill_conditioned=True)
    result = q.renyi_closed_form(t, alpha=1.0, kl_max=1.0e9)
    assert result.dtype is torch.float64


def test_fast_and_strict_policies_agree_on_well_conditioned_input():
    fast_q, fast_t = _pair("fp32_escalate")
    strict_q, strict_t = _pair("fp32_escalate_cond")
    fast = fast_q.renyi_closed_form(fast_t, alpha=1.0, kl_max=1.0e9)
    strict = strict_q.renyi_closed_form(strict_t, alpha=1.0, kl_max=1.0e9)
    assert torch.equal(fast, strict)


@pytest.mark.parametrize(
    ("entrypoint", "attribute"),
    [("train_vfe3", "config"), ("ablation", "BASELINE_CONFIG")],
)
def test_click_to_run_entrypoints_express_the_default_policy(entrypoint, attribute):
    # f54c756 dropped the restated literal from both launchers, which now inherit VFE3Config's
    # default. Assert the policy each entrypoint actually RUNS with rather than a source-text
    # literal -- the same invariant-not-literal rule the audit 2026-07-25 F19 fixes applied.
    module = importlib.import_module(entrypoint)
    effective = VFE3Config(**dict(getattr(module, attribute)))
    assert effective.full_cov_kl_precision == "fp32_escalate"


@pytest.mark.parametrize(
    "field,fast_value,strict_value",
    [
        ("full_cov_kl_precision", "fp32_escalate", "fp32_escalate_cond"),
    ],
)
def test_scaling_signature_separates_kl_precision_routes(field, fast_value, strict_value):
    fast = {name: None for name in scaling_analysis._SCALING_STRUCTURAL_FIELDS}
    strict = dict(fast)
    fast[field] = fast_value
    strict[field] = strict_value
    assert scaling_analysis._structural_signature(fast) != scaling_analysis._structural_signature(strict)
