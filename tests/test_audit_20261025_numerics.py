r"""Regressions for the numerics findings of the 2026-10-25 ultradeep audit (F05, F11-F14, F18, F20-F22).

Each case is built from the audit's own probes (``docs/audits/ultradeep-2026-10-25``) at tiny CPU shapes:

* F05: a target-scored inference-mode call with repeated token ids read the version counter of an inference
  tensor in the gathered frame certificate (``vfe3/contracts.py``).
* F11: the small Fisher chart eigenvalue :math:`\tfrac12(2 + a - \sqrt{a^2 + 4})` rounded to zero at small
  whitened mean norms, and the chart optimizer raised on the infinite condition certificate.
* F12: a single-layer model skipped the block-mode inactivity marks (``inactive_fields``).
* F13: CG constants cast under inference mode were cached and broke a later gradient call.
* F14: the zero-weight no-grad CG shortcut skipped the ``delta_full`` covariance floor.
* F18: the fixed-context receiver cost reduced per-coordinate log variances before weighting them.
* F20: an ill-conditioned suffix escalated the whole minimal-agent call and changed healthy prefix outputs.
* F21: the float32 covariance trust norm overflowed and scaled a finite large tangent to zero.
* F22: the expected-likelihood Triton forward formed ``-inf - -inf`` after an all ``-inf`` first tile
  (checked under the Triton CPU interpreter in a subprocess, so this process never imports triton).
"""
import json
import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest
import torch

from tests.profiles import make_relational_operators, test_config
from vfe3.contracts import CanonicalFrameContext
from vfe3.relational_scoring import set_projected_condition_recovery

REPO = Path(__file__).resolve().parents[1]
_NO_TRITON = 77                                       # subprocess exit code: triton unavailable


# ----------------------------------------------------------------------------------------------
# F05: target-scored inference with repeated token ids
# ----------------------------------------------------------------------------------------------

def _target_loss(tokens: list, mode: str) -> float:
    from vfe3.model.model import VFEModel
    torch.manual_seed(7)
    model = VFEModel(test_config(vocab_size=7, embed_dim=4, n_heads=2, n_layers=1, max_seq_len=3,
                                 use_block_mlp=False)).eval()
    ids = torch.tensor([tokens])
    scope = torch.inference_mode() if mode == "inference" else torch.no_grad()
    with scope:
        _, loss, _ = model(ids, targets=ids)
        return float(loss)


@pytest.mark.parametrize("tokens", [[1, 1, 2], [1, 2, 3]])
def test_target_scored_inference_mode_accepts_repeated_tokens(tokens: list) -> None:
    r"""Repeated ids gather the frame pair from unique rows; under inference mode no version-counter record is
    kept and the context validates in full. The loss equals the no-grad loss."""
    reference = _target_loss(tokens, "no_grad")
    inferred = _target_loss(tokens, "inference")
    assert math.isfinite(inferred)
    assert abs(inferred - reference) <= 1e-6 * max(1.0, abs(reference))


# ----------------------------------------------------------------------------------------------
# F11: Fisher chart spectrum at small whitened mean norms
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("a", [1e-16, 1e-18, 1e-12, 0.3])
def test_stabilizer_small_eigenvalue_is_free_of_cancellation(a: float) -> None:
    r"""The smaller root of :math:`\lambda^2 - (2 + a)\lambda + a` is :math:`a/\lambda_+`, about :math:`a/2`."""
    from vfe3.geometry.phi_fisher import fisher_spectrum_bounds
    lo, hi = fisher_spectrum_bounds(torch.tensor([a], dtype=torch.float64), 2, rotation_weight=2.0,
                                    completion="stabilizer")
    plus = 0.5 * (2.0 + a + math.sqrt(a * a + 4.0))
    assert float(lo) > 0.0 and math.isfinite(float(hi) / float(lo))
    assert abs(float(lo) - a / plus) <= 1e-14 * (a / plus)


@pytest.mark.parametrize("lam", [0.5, 2.0, 3.0])
def test_rotation_completion_small_eigenvalue_matches_the_product_of_roots(lam: float) -> None:
    from vfe3.geometry.phi_fisher import completion_coefficients, fisher_spectrum_bounds
    c1, c2 = completion_coefficients(lam)
    a = torch.tensor([1e-9, 0.7], dtype=torch.float64)
    lo, _ = fisher_spectrum_bounds(a, 3, rotation_weight=lam)
    for k, value in enumerate(a.tolist()):
        root = math.sqrt(value * value + 4.0 * c2 * c2)
        plus, minus = 0.5 * (2 * c1 + value + root), 0.5 * (2 * c1 + value - root)
        members = [2.0 + value, minus, plus, 2.0, lam]
        assert abs(float(lo[k]) - min(members)) <= 1e-12 * min(members)


@pytest.mark.parametrize("mean_scale", [1e-8, 1e-9])
def test_chart_optimizer_takes_a_finite_step_near_zero_mean(mean_scale: float) -> None:
    r"""The audit's chart probe raised ``FloatingPointError`` at means 1e-8 and 1e-9; the step now matches the
    neighboring means 0 and 1e-6 (a diagonal step of about -0.005)."""
    from vfe3.gauge_optim import GaugeManifoldAdamW
    from vfe3.geometry.groups import get_group
    from vfe3.geometry.phi_fisher import FisherRowStatistics
    p = torch.nn.Parameter(torch.zeros(1, 4, dtype=torch.float64))
    group = get_group("block_glk")(K=2, n_heads=1)
    stats = FisherRowStatistics(kind="type", count=torch.ones(1, dtype=torch.float64),
                                total=torch.tensor(1., dtype=torch.float64),
                                local_mean=torch.tensor([[[mean_scale, 0.]]], dtype=torch.float64),
                                local_var=torch.ones(1, 1, 2, dtype=torch.float64))
    opt = GaugeManifoldAdamW([dict(params=[p], pullback_group=True, lr=.01, weight_decay=0)], group,
                             lr=.01, phi_group_trust_radius=.1, phi_chart_max_norm=5.,
                             phi_group_retraction="chart", phi_statistics_provider=lambda param, rows: stats,
                             phi_fisher_completion="stabilizer", weight_decay=0)
    p.grad = torch.eye(2, dtype=torch.float64).reshape(1, 4)
    opt.step()
    step = p.detach().reshape(2, 2)
    assert torch.isfinite(step).all()
    torch.testing.assert_close(step.diagonal(), torch.full((2,), -0.005, dtype=torch.float64), rtol=1e-6, atol=0)


# ----------------------------------------------------------------------------------------------
# F12: single-layer inactivity marks
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["prototype_agents", "gaussian_template"])
def test_single_layer_keeps_the_block_mode_inactivity_marks(mode: str) -> None:
    for depth in (1, 2):
        inactive = test_config(n_layers=depth, use_block_mlp=True, block_mlp_mode=mode).inactive_fields()
        assert "block_mlp_activation" in inactive, (mode, depth)
        assert ("block_mlp_tie_layers" in inactive) == (depth == 1), (mode, depth)


# ----------------------------------------------------------------------------------------------
# F13, F14: CG coupling caches and the zero-weight floor
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("covariance_mode", ["passthrough", "delta_full"])
def test_cg_inference_call_does_not_poison_a_later_gradient_call(covariance_mode: str) -> None:
    r"""The float64 intertwiners are cast to float32 inside inference mode first; the later gradient call must
    still save its constants for backward."""
    from vfe3.model.cg_coupling import CGCoupling
    c = CGCoupling(3, "so", [1, 3], ["l0", "l1"], cg_covariance_mode=covariance_mode)
    with torch.no_grad():
        c.path_weights.fill_(0.1)
    mu = torch.arange(4, dtype=torch.float32)
    sigma = torch.eye(4) if covariance_mode == "delta_full" else torch.ones(4)
    with torch.inference_mode():
        c(mu, sigma)
    leaf = mu.clone().requires_grad_()
    out_mu, out_sigma = c(leaf, sigma)
    (out_mu.sum() + out_sigma.sum()).backward()
    assert leaf.grad is not None and torch.isfinite(leaf.grad).all()
    assert c.path_weights.grad is not None and torch.isfinite(c.path_weights.grad).all()


def test_zero_weight_cg_applies_the_delta_full_floor_in_every_grad_mode() -> None:
    from vfe3.model.cg_coupling import CGCoupling
    c = CGCoupling(3, "so", [1, 3], ["l0", "l1"], cg_covariance_mode="delta_full",
                   cg_covariance_floor=0.5).double()
    mu, sigma = torch.tensor([1., 2., 3., 4.], dtype=torch.float64), torch.eye(4, dtype=torch.float64)
    training = c(mu, sigma)
    with torch.no_grad():
        evaluation = c(mu, sigma)
        moments = c.forward_moments(mu, sigma)
    torch.testing.assert_close(evaluation[1], training[1].detach(), rtol=0, atol=0)
    torch.testing.assert_close(evaluation[1], moments.sigma, rtol=0, atol=0)
    torch.testing.assert_close(evaluation[1].diagonal(), torch.full((4,), 1.5, dtype=torch.float64))
    torch.testing.assert_close(evaluation[0], mu, rtol=0, atol=0)


# ----------------------------------------------------------------------------------------------
# F18: fixed-context receiver cost with per-coordinate entropy weights
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("length", [1, 3])
def test_fixed_context_cost_weights_each_log_variance(length: int) -> None:
    r""":math:`F = \tfrac12\sum_{ik}(\Lambda_{kk}s_{ik} - c_{ik}\log s_{ik})` at zero means and identity precision."""
    from vfe3.gradients.full_link import FullLinkContext, full_link_cost, full_link_cost_parts
    mu = torch.zeros(1, length, 2, dtype=torch.float64)
    sigma = torch.tensor([2.0, 3.0], dtype=mu.dtype).expand_as(mu).clone().requires_grad_()
    Lambda = torch.eye(2, dtype=mu.dtype).expand(1, length, 2, 2)
    c = torch.tensor([1.0, 2.0], dtype=mu.dtype).expand_as(mu)
    ctx = FullLinkContext(Lambda, mu, c, torch.ones(1, 1, length, length) / length,
                          torch.zeros(1, 1, length, length), None)
    expected = .5 * (sigma - c * sigma.log()).sum()
    cost = full_link_cost(mu, sigma, ctx)
    torch.testing.assert_close(cost, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(cost, sum(full_link_cost_parts(mu, sigma, ctx)), rtol=1e-12, atol=1e-12)


# ----------------------------------------------------------------------------------------------
# F20: minimal-agent precision escalation preserves the healthy prefix
# ----------------------------------------------------------------------------------------------

def _suffix_case(inference: str) -> Tuple[Any, CanonicalFrameContext, torch.Tensor, torch.Tensor, torch.Tensor,
                                          torch.Tensor, torch.Tensor]:
    r"""The audit's strictly causal N = 3, H = 1, d = 2 case; the suffix variant makes token 2 ill-conditioned."""
    ops = make_relational_operators(
        [2], 0, attention_residual=False,
        predictive_readout="minimal_agent", minimal_agent_inference=inference,
        predictive_covariance="moment_matched", predictive_covariance_floor=1e-4,
    )
    with torch.no_grad():
        ops.raw_score_q.copy_(torch.tensor([[[1.0, 0.2], [-0.1, 1.0]]]))
        ops.T.copy_(torch.tensor([[[0.8, -0.3], [0.1, 1.1]]]))
        ops.channel_log_noise.fill_(-0.4)
    eye = torch.eye(2).expand(1, 3, 2, 2)
    frame = CanonicalFrameContext(eye, eye, block_diagonal=False)
    mu = torch.tensor([[[0.3, -0.7], [1.1, 0.2], [-0.5, 0.4]]])
    sigma = torch.tensor([[[0.7, 1.3], [1.2, 0.8], [0.9, 1.1]]])
    suffix = sigma.clone()
    suffix[:, 2, :] = torch.tensor([1e4, 1.0])
    prior = torch.ones_like(sigma)
    lp = torch.zeros(3, 3).masked_fill(torch.ones(3, 3, dtype=torch.bool).triu(1), -torch.inf)
    return ops, frame, mu, sigma, suffix, prior, lp


def test_ill_conditioned_suffix_leaves_the_prefix_bitwise() -> None:
    ops, frame, mu, sigma, suffix, prior, lp = _suffix_case("exact")
    previous = set_projected_condition_recovery(True)
    try:
        with torch.no_grad():
            statuses, outputs = [], []
            for covariance in (sigma, suffix):
                floor = ops.floor_term(prior, covariance, frame, diagonal=True)
                statuses.append(ops._minimal_agent_body(mu, covariance, frame, lp, True, False, floor)[3].tolist())
                outputs.append(ops.minimal_agent_update(mu, covariance, frame, lp, return_covariance=True,
                                                        floor_anchor=prior))
            wide_mean, wide_cov = ops.minimal_agent_update(mu.double(), suffix.double(), frame, lp.double(),
                                                           return_covariance=True, floor_anchor=prior.double())
    finally:
        set_projected_condition_recovery(previous)
    assert statuses == [[False, False, False], [False, True, False]]    # the suffix alone trips the certificate
    (mean_a, cov_a), (mean_b, cov_b) = outputs
    assert torch.equal(mean_a[:, :2], mean_b[:, :2]) and torch.equal(cov_a[:, :2], cov_b[:, :2])
    # The flagged receiver takes the float64 recompute, cast back.
    assert torch.equal(mean_b[:, 2], wide_mean[:, 2].float()) and torch.equal(cov_b[:, 2], wide_cov[:, 2].float())


def test_escalated_call_backpropagates_through_the_float64_recompute() -> None:
    r"""Gradients of an escalated call are the recompute's at every receiver (finite, and equal to a float64
    call's gradient cast at float32 rounding)."""
    ops, frame, mu, _, suffix, prior, lp = _suffix_case("exact")
    previous = set_projected_condition_recovery(True)
    try:
        leaf = mu.clone().requires_grad_()
        mean, cov = ops.minimal_agent_update(leaf, suffix, frame, lp, return_covariance=True, floor_anchor=prior)
        (mean.square().sum() + cov.sum()).backward()
        wide_leaf = mu.double().requires_grad_()
        mean64, cov64 = ops.minimal_agent_update(wide_leaf, suffix.double(), frame, lp.double(),
                                                 return_covariance=True, floor_anchor=prior.double())
        (mean64.square().sum() + cov64.sum()).backward()
    finally:
        set_projected_condition_recovery(previous)
    assert torch.isfinite(leaf.grad).all()
    torch.testing.assert_close(leaf.grad.double(), wide_leaf.grad, rtol=1e-5, atol=1e-6)


# ----------------------------------------------------------------------------------------------
# F21: overflow-safe covariance trust norm
# ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("magnitude", [2e19, 1e20])
def test_float32_trust_scaling_keeps_large_finite_tangents(magnitude: float) -> None:
    r"""Both retractions bound the whitened tangent to the trust ball: :math:`\exp(1/\sqrt2)` on the diagonal."""
    from vfe3.geometry.retraction import retract_spd_diagonal, retract_spd_full
    expected = math.exp(1.0 / math.sqrt(2.0))
    diag = retract_spd_diagonal(torch.ones(2), torch.full((2,), magnitude), trust_region=1, sigma_max=None)
    full = retract_spd_full(torch.eye(2), magnitude * torch.eye(2), trust_region=1, sigma_max=None)
    torch.testing.assert_close(diag, torch.full((2,), expected), rtol=1e-6, atol=0)
    torch.testing.assert_close(full, expected * torch.eye(2), rtol=1e-6, atol=1e-6)


def test_overflow_safe_norm_is_the_plain_norm_in_the_normal_range() -> None:
    from vfe3.geometry.retraction import _overflow_safe_norm
    torch.manual_seed(0)
    x = torch.randn(5, 3, 3)
    assert torch.equal(_overflow_safe_norm(x, event_ndim=2), torch.linalg.norm(x, ord="fro", dim=(-2, -1),
                                                                                keepdim=True))
    assert torch.equal(_overflow_safe_norm(x[..., 0], event_ndim=1),
                       torch.linalg.vector_norm(x[..., 0], dim=-1, keepdim=True))
    assert torch.equal(_overflow_safe_norm(torch.zeros(2, 3), event_ndim=1), torch.zeros(2, 1))


# ----------------------------------------------------------------------------------------------
# F22: expected-likelihood Triton forward after an all -inf first tile (CPU interpreter, subprocess)
# ----------------------------------------------------------------------------------------------

_EMPTY_PREFIX_SCRIPT = """
import json, sys
try:
    import triton  # noqa: F401
except Exception as exc:
    print("triton import failed:", exc)
    sys.exit(77)
import torch
torch.set_num_threads(1)
from vfe3.kernels.expected_likelihood import expected_likelihood_ce
from vfe3.kernels import _expected_likelihood_triton as tk
assert tk.INTERPRETED
mq, sq = torch.zeros(1, 1), torch.full((1, 1), 0.5)
mv, sv = torch.full((65, 1), 3e19), torch.full((65, 1), 0.5)
mv[-1] = 0                                             # the first 64-class tile is all -inf; class 64 is finite
logits = -0.5 * (((mq - mv).square() / (sq + sv)) + (sq + sv).log()).sum(-1)
reference = torch.logsumexp(logits, 0) - logits[-1]
fused = expected_likelihood_ce(mq, sq, mv, sv, None, torch.tensor(1.), torch.tensor([64]))
print(json.dumps({"reference": float(reference), "fused": float(fused)}))
"""


def _subprocess_env(**extra: str) -> Dict[str, str]:
    env = dict(os.environ)
    env.pop("TRITON_INTERPRET", None)
    env.update(CUDA_VISIBLE_DEVICES="-1", VFE3_TEST_DEVICE="cpu", PYTHONPATH=str(REPO), OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", **extra)
    return env


def test_expected_likelihood_kernel_recovers_after_an_all_negative_infinite_tile() -> None:
    proc = subprocess.run([sys.executable, "-c", textwrap.dedent(_EMPTY_PREFIX_SCRIPT)], cwd=REPO,
                          env=_subprocess_env(TRITON_INTERPRET="1"), capture_output=True, text=True, timeout=900)
    if proc.returncode == _NO_TRITON:
        pytest.skip(f"triton unavailable in {sys.executable}: {proc.stdout.strip()[-300:]}")
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["reference"] == 0.0
    assert math.isfinite(result["fused"]) and abs(result["fused"] - result["reference"]) <= 1e-6, result
