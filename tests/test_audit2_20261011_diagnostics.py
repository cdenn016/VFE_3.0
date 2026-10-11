r"""Model diagnostics and the canonical free energy (audit 2026-10-10 round 2, agent 2 remediation).

Items 3 and 13 (per-layer covariance scale, the stack-entry ratio and the Gram-normalized, orbit-invariant
:math:`\lambda_{\min}`), 4 (rank-one residual, self-map identity distance), 7 (dead-router warning under every
reference), 18 (comparison RoPE in :math:`\beta^*`, ``canonical_self_weight``), 27 (canonical F at the readout
output), 31 (``inactive_fields``, minimal-agent temperature), 38 (NaN rows of ``canonical_routing_rows``) and 53
(per-head float64 exponentials). Tiny CPU models only (K = 8, two heads of 4, N = 5, vocab 9).
"""
from pathlib import Path
import csv
import dataclasses
import logging
import math
import warnings
from typing import Any, Dict, List, Optional

import pytest
import torch
from torch.utils.data import DataLoader

import vfe3.free_energy as fe
import vfe3.run_artifacts as ra
import vfe3.train as T
from tests.profiles import test_config, tiny_config
from vfe3 import metrics
from vfe3.data.datasets import TokenWindows
from vfe3.geometry.rope import build_rope_rotation
from vfe3.model.model import CANONICAL_DIAGNOSTIC_KEYS, DiagnosticSnapshot, VFEModel
from vfe3.posthoc.klcap import causal_support

TOKENS = torch.tensor([[0, 1, 2, 3, 4], [4, 5, 6, 7, 8]])
COMPARISON_ROPE = dict(pos_rotation="rope", rope_space="comparison", rope_on_value=False,
                       beta_attention_prior="causal_alibi", rope_comparison_covariance="rotation_invariant",
                       routing_kl_max_reference="row_min")
MINIMAL_AGENT = dict(predictive_readout="minimal_agent", attention_residual=False)
PER_LAYER_KEYS = ("relational_projected_cov_scale", "relational_projected_cov_gram_min_eigenvalue",
                  "relational_comparison_map_norm_spread", "relational_input_cov_scale_ratio",
                  "relational_rank_one_residual", "relational_link_unit_distance_min")
AGGREGATE_KEYS = (
    "relational_projected_cov_min_eigenvalue_layer_min", "relational_projected_cov_min_eigenvalue_argmin_layer",
    "relational_projected_cov_scale_layer_min", "relational_projected_cov_scale_argmin_layer",
    "relational_projected_cov_gram_min_eigenvalue_layer_min",
    "relational_projected_cov_gram_min_eigenvalue_argmin_layer",
    "relational_input_cov_scale_ratio_layer_min", "relational_input_cov_scale_ratio_argmin_layer",
    "relational_comparison_map_norm_spread_layer_max", "relational_self_lock_unit_distance",
    "relational_score_klmax_frac_head_argmax_layer", "relational_score_klmax_frac_head_argmax_head")


def _model(**overrides: Any) -> VFEModel:
    values: Dict[str, Any] = dict(vocab_size=9, embed_dim=8, n_heads=2, max_seq_len=5, n_layers=3,
                                  phi_scale=0.4, diagnose_canonical_free_energy=True)
    values.update(overrides)
    torch.manual_seed(3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VFEModel(tiny_config(**values)).eval()


def _dense_kl(mu_q: torch.Tensor, C_q: torch.Tensor, mu_t: torch.Tensor, C_t: torch.Tensor) -> float:
    r"""KL(N(mu_q, C_q) || N(mu_t, C_t)) with an explicit inverse and log-determinants."""
    precision = torch.linalg.inv(C_t)
    diff = mu_t - mu_q
    return 0.5 * float(torch.trace(precision @ C_q) + diff @ precision @ diff - mu_q.numel()
                       + torch.linalg.slogdet(C_t)[1] - torch.linalg.slogdet(C_q)[1])


def _dense_frame_inverse(model: VFEModel, snapshot: DiagnosticSnapshot, layer: int) -> torch.Tensor:
    r"""The dense (B, N, K, K) float64 inverse frames, built as the diagnostic built them before item 53."""
    group, start = model.group, snapshot.initial_belief
    inverse = torch.linalg.matrix_exp(-group.embed_algebra(start.phi.double(), dtype=torch.float64))
    if start.right_phi is not None:
        inverse = torch.linalg.matrix_exp(
            -group.embed_algebra(start.right_phi.double(), dtype=torch.float64)) @ inverse
    if snapshot.rope is not None:
        inverse = snapshot.rope.double() @ inverse
    ops = model.relational_operators_by_layer[layer]
    if ops.comparison_rope_base is not None:
        N = start.mu.shape[-2]
        rotation = torch.block_diag(*(torch.eye(d, dtype=torch.float64) for d in group.irrep_dims)
                                    ).expand(N, -1, -1).clone()
        offset = 0
        for d in group.irrep_dims:
            rotation[:, offset:offset + d, offset:offset + d] = build_rope_rotation(
                torch.arange(N), [d], base=ops.comparison_rope_base, fraction=ops.comparison_rope_fraction,
                dtype=torch.float64)
            offset += d
        inverse = rotation @ inverse
    return inverse


def _tau(model: VFEModel, layer: int) -> 'float | torch.Tensor':
    from vfe3.model.block import _as_coeff
    ops = model.relational_operators_by_layer[layer]
    if ops.predictive_readout == "minimal_agent":
        return 1.0
    cpu = torch.device("cpu")
    return ops.routing_temperature(
        fe.attention_tau(_as_coeff(model.cfg.kappa_beta, cpu), model.group.irrep_dims, device=cpu), torch.float64)


def _input(snapshot: DiagnosticSnapshot, layer: int):
    return snapshot.initial_belief if layer == 0 else snapshot.layer_outputs[layer - 1]


# --------------------------------------------------------------------------- keys, finiteness, columns

@pytest.mark.parametrize("overrides", [COMPARISON_ROPE, MINIMAL_AGENT, dict()],
                         ids=["comparison_rope_alibi", "minimal_agent", "plain"])
def test_new_keys_exist_are_finite_and_have_val_columns(overrides: Dict[str, Any]) -> None:
    model = _model(**overrides)
    snapshot = model.build_diagnostic_snapshot(TOKENS)
    last = model.diagnostics(TOKENS, snapshot=snapshot)
    per_layer = model.diagnostics_per_layer(TOKENS, snapshot=snapshot)
    minimal = model.cfg.predictive_readout == "minimal_agent"
    nan_by_definition = {"relational_link_unit_distance_min", "relational_self_lock_unit_distance"} if minimal else set()
    for key in PER_LAYER_KEYS:
        assert len(per_layer[key]) == model.cfg.n_layers
        assert last[key] == per_layer[key][-1] or (math.isnan(last[key]) and math.isnan(per_layer[key][-1])), key
        values = per_layer[key]
        assert all(math.isnan(v) for v in values) if key in nan_by_definition else all(
            math.isfinite(v) for v in values), (key, values)
    for key in AGGREGATE_KEYS:
        assert math.isnan(last[key]) if key in nan_by_definition else math.isfinite(last[key]), key
    for key in CANONICAL_DIAGNOSTIC_KEYS:
        assert all(math.isfinite(v) for v in per_layer[key]), key
    val_keys = set(T._VAL_RELATIONAL_DIAG_KEYS)
    assert {f"val_{key}" for key in (*PER_LAYER_KEYS, *AGGREGATE_KEYS)} <= val_keys
    # every relational_* scalar the training row and the validation row carry has a fixed val_* column
    assert {f"val_{key}" for key, value in last.items()
            if key.startswith("relational_") and isinstance(value, float)} <= val_keys | set(
        T._VAL_MINIMAL_AGENT_DIAG_KEYS)
    # the layer extremes reduce the per-layer columns, with the first layer attaining them
    for key in ("relational_projected_cov_min_eigenvalue", "relational_projected_cov_scale",
                "relational_projected_cov_gram_min_eigenvalue", "relational_input_cov_scale_ratio"):
        values = per_layer[key]
        assert last[f"{key}_layer_min"] == min(values)
        assert last[f"{key}_argmin_layer"] == float(values.index(min(values)))
    assert last["relational_comparison_map_norm_spread_layer_max"] == max(
        per_layer["relational_comparison_map_norm_spread"])
    # the stack entry against itself
    assert per_layer["relational_input_cov_scale_ratio"][0] == 1.0


def test_rank_one_residual_reuses_the_dong_definition() -> None:
    from vfe3.viz.extract import across_layer_belief_trace
    model = _model(**COMPARISON_ROPE)
    snapshot = model.build_diagnostic_snapshot(TOKENS[:1])
    per_layer = model.diagnostics_per_layer(TOKENS[:1], snapshot=snapshot)
    trace = across_layer_belief_trace(model, TOKENS[:1], snapshot=snapshot)
    assert per_layer["relational_rank_one_residual"] == pytest.approx(
        trace["rank_one_residual"].double().tolist(), rel=1e-6)
    for layer in range(model.cfg.n_layers):
        expected = float(metrics.rank_one_residual(snapshot.layer_outputs[layer].mu.double()).mean())
        assert per_layer["relational_rank_one_residual"][layer] == expected


def test_self_map_identity_distance_is_the_spectral_radius_of_w_minus_identity() -> None:
    model = _model()
    with torch.no_grad():
        for ops in model.relational_operators_by_layer:
            ops.O.mul_(0.5)                                       # affine chart, identity init: W_hh = 0.5 I
    d = model.diagnostics(TOKENS)
    assert d["relational_link_unit_distance_min"] == pytest.approx(0.5, abs=1e-6)
    assert d["relational_link_unit_distance_max"] == pytest.approx(0.5, abs=1e-6)
    assert d["relational_self_lock_unit_distance"] == pytest.approx(0.5, abs=1e-6)
    with torch.no_grad():
        model.relational_operators_by_layer[0].O[:, :, :] *= 2.0                    # layer 0 back to W = I
    distances = VFEModel._self_map_unit_distance(model.relational_operators_by_layer[0])
    assert torch.allclose(distances, torch.zeros_like(distances), atol=1e-6)


# --------------------------------------------------------------------------- items 3 and 13: scale and orbit

def _commutant(r: int, scales: List[float], angle: float) -> torch.Tensor:
    r"""A block-diagonal commutant element c_p R(angle) on each rotating plane of an r-dim comparison block."""
    blocks = []
    for c in scales:
        rotation = torch.tensor([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        blocks.append(c * rotation)
    return torch.block_diag(*blocks)


@pytest.mark.parametrize("rope", [True, False], ids=["comparison_rope", "no_rope"])
def test_gram_normalized_lambda_min_is_invariant_along_the_free_comparison_orbit(rope: bool) -> None:
    model = _model(**(COMPARISON_ROPE if rope else {}))
    before = model.diagnostics(TOKENS)
    generator = torch.Generator().manual_seed(4)
    with torch.no_grad():
        for ops in model.relational_operators_by_layer:
            r = ops.raw_score_q.shape[-2]
            A = (_commutant(r, [0.5, 3.0], 0.3) if rope
                 else torch.eye(r) + 0.4 * torch.randn(r, r, generator=generator))     # (r, r), in GL(r)
            ops.raw_score_q.copy_(A @ ops.raw_score_q)
            ops.raw_score_k.copy_(A @ ops.raw_score_k)
    after = model.diagnostics(TOKENS)
    # the routing (and so the whole forward) is unchanged up to float32 rounding ...
    assert after["canonical_beta_tv"] == pytest.approx(before["canonical_beta_tv"], rel=1e-4)
    assert after["relational_input_cov_scale_ratio_layer_min"] == pytest.approx(
        before["relational_input_cov_scale_ratio_layer_min"], rel=1e-4)
    # ... the scale-free statistic does not move, while the absolute spectrum and the map-norm spread do
    for key in ("relational_projected_cov_gram_min_eigenvalue", "relational_projected_cov_gram_min_eigenvalue_layer_min"):
        assert after[key] == pytest.approx(before[key], rel=1e-4), key
    assert not after["relational_projected_cov_min_eigenvalue"] == pytest.approx(
        before["relational_projected_cov_min_eigenvalue"], rel=1e-2)
    assert not after["relational_projected_cov_scale"] == pytest.approx(before["relational_projected_cov_scale"],
                                                                        rel=1e-2)
    assert after["relational_comparison_map_norm_spread_layer_max"] > 2.0 * before[
        "relational_comparison_map_norm_spread_layer_max"]
    if rope and before["relational_comparison_map_norm_spread_layer_max"] == pytest.approx(1.0):
        # equal plane norms, one plane scaled by 0.5 and the other by 3: spread (3 / 0.5)^2
        assert after["relational_comparison_map_norm_spread_layer_max"] == pytest.approx(36.0, rel=1e-4)


@pytest.mark.parametrize("rope", [True, False], ids=["comparison_rope", "no_rope"])
def test_gram_normalized_lambda_min_reads_an_isotropic_reference_variance(rope: bool) -> None:
    # G = sigma^2 S C S^T with C = I read by S_Q = S_K = S: every head, plane and block reads sigma^2
    model = _model(**(COMPARISON_ROPE if rope else {}), tie_comparison_maps=True)
    snapshot = model.build_diagnostic_snapshot(TOKENS)
    ops = model.relational_operators_by_layer[0]
    with torch.no_grad():
        ops.raw_score_q.add_(0.3 * torch.randn(ops.raw_score_q.shape, generator=torch.Generator().manual_seed(1)))
    S = ops.raw_score_q.detach().double()                                            # (H, r, K)
    sigma2 = 0.37
    G = (sigma2 * S @ S.transpose(-1, -2)).expand(*TOKENS.shape, *S.shape[:1], S.shape[1], S.shape[1])
    stats = model._covariance_scale_statistics(
        snapshot, 0, {"projected_covariance_pre": G, "projected_key_covariance_pre": G})
    assert stats["relational_projected_cov_gram_min_eigenvalue"] == pytest.approx(sigma2, rel=1e-10)
    scale = sigma2 * float(torch.diagonal(S @ S.transpose(-1, -2), dim1=-2, dim2=-1).sum(-1).median()) / S.shape[1]
    assert stats["relational_projected_cov_scale"] == pytest.approx(scale, rel=1e-10)


def test_covariance_collapse_warns_on_a_contracting_stack_and_not_on_a_healthy_one(
        caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("covariance-collapse")
    healthy = _model()
    collapsed = _model(attention_residual=False)
    with torch.no_grad():
        for ops in collapsed.relational_operators_by_layer:
            ops.O.mul_(1e-2)                     # affine chart: every layer's content map M scaled by 1e-2
    d_healthy, d_collapsed = healthy.diagnostics(TOKENS), collapsed.diagnostics(TOKENS)
    assert d_healthy["relational_input_cov_scale_ratio_layer_min"] >= T.COVARIANCE_COLLAPSE_RATIO
    # floor 1e-4 of tr C plus the 1e-4 scale of M Sigma M^T per layer
    assert d_collapsed["relational_input_cov_scale_ratio_layer_min"] < T.COVARIANCE_COLLAPSE_RATIO
    assert d_collapsed["relational_input_cov_scale_ratio_argmin_layer"] == 2.0
    # collapse of the covariance lowers the scale-free statistic as well, not only the absolute one
    assert (d_collapsed["relational_projected_cov_gram_min_eigenvalue_layer_min"]
            < 1e-2 * d_healthy["relational_projected_cov_gram_min_eigenvalue_layer_min"])
    warned: set = set()
    with caplog.at_level(logging.WARNING, logger="covariance-collapse"):
        T._warn_covariance_collapse(d_healthy, warned, logger)
        assert not caplog.records
        T._warn_covariance_collapse(d_collapsed, warned, logger)
        T._warn_covariance_collapse(d_collapsed, warned, logger)            # once per layer
    assert len(caplog.records) == 1 and caplog.records[0].getMessage().startswith(" covariance layer 2:")
    assert warned == {("covariance_scale", 2, -1)}
    T._warn_covariance_collapse({}, set(), logger)                          # keys absent: skipped


# --------------------------------------------------------------------------- item 7: dead-router warning

def test_dead_router_warning_fires_on_the_live_capped_row_and_not_on_a_healthy_one(
        caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("dead-router-capped")
    live = {"relational_score_klmax_frac_head_max": 0.939, "relational_score_klmax_frac_head_argmax_layer": 4,
            "relational_score_klmax_frac_head_argmax_head": 2, "relational_self_weight_head_max": 0.379,
            "relational_self_lock_argmax_layer": 0, "relational_self_lock_argmax_head": 1,
            "relational_clamped_row_frac_head_max": 0.0}
    healthy = dict(live, relational_score_klmax_frac_head_max=0.0)
    with caplog.at_level(logging.WARNING, logger="dead-router-capped"):
        T._warn_dead_router(healthy, set(), logger, reference="row_min")
        assert not caplog.records
        T._warn_dead_router(live, set(), logger, reference="row_min")
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert message.startswith(" router layer 4 head 2:") and "near one-hot" in message
    assert "uniform" not in message


def test_self_lock_warning_names_softmax_saturation_under_the_absolute_reference(
        caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("dead-router-self")
    locked = {"relational_self_weight_head_max": 0.998, "relational_self_lock_argmax_layer": 3,
              "relational_self_lock_argmax_head": 1, "relational_self_lock_unit_distance": 0.012,
              "relational_clamped_row_frac_head_max": 0.0, "relational_score_klmax_frac_head_max": 0.0}
    with caplog.at_level(logging.WARNING, logger="dead-router-self"):
        T._warn_dead_router(locked, set(), logger, reference="absolute")
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert message.startswith(" router layer 3 head 1:") and "softmax saturates" in message
    assert "0.012" in message and "capped" not in message


def test_dead_router_warning_reads_a_real_capped_model(caplog: pytest.LogCaptureFixture) -> None:
    model = _model(**COMPARISON_ROPE, kl_max=1e-3)       # every non-best source sits above the tiny row gap cap
    d = model.diagnostics(TOKENS)
    assert d["relational_score_klmax_frac_head_max"] > T.DEAD_ROUTER_CAPPED_FRAC - 0.5
    logger = logging.getLogger("dead-router-real")
    with caplog.at_level(logging.WARNING, logger="dead-router-real"):
        T._warn_dead_router(dict(d, relational_score_klmax_frac_head_max=0.95), set(), logger, reference="row_min")
    capped = [r.getMessage() for r in caplog.records if "admissible costs are capped" in r.getMessage()]
    layer, head = (int(d[f"relational_score_klmax_frac_head_argmax_{axis}"]) for axis in ("layer", "head"))
    assert len(capped) == 1 and capped[0].startswith(f" router layer {layer} head {head}:")


# --------------------------------------------------------------------------- item 18: comparison RoPE in beta*

def test_beta_star_matches_a_brute_force_rotated_reference() -> None:
    model = _model(**COMPARISON_ROPE)
    cfg, group = model.cfg, model.group
    snapshot = model.build_diagnostic_snapshot(TOKENS)
    per_layer = model.diagnostics_per_layer(TOKENS, snapshot=snapshot)
    dims = list(group.irrep_dims)
    log_prior = snapshot.log_prior.double()
    while log_prior.dim() < 4:
        log_prior = log_prior.unsqueeze(0)
    for layer in range(cfg.n_layers):
        q = _input(snapshot, layer)
        rows = fe.canonical_routing_rows(q.mu, q.sigma, model._canonical_frame_inverse(snapshot, layer),
                                         snapshot.log_prior, tau=_tau(model, layer), irrep_dims=dims)
        start = snapshot.initial_belief
        U = torch.linalg.matrix_exp(group.embed_algebra(start.phi.double(), dtype=torch.float64))
        if start.right_phi is not None:
            U = U @ torch.linalg.matrix_exp(group.embed_algebra(start.right_phi.double(), dtype=torch.float64))
        V = torch.linalg.inv(U)                                                          # (B, N, K, K)
        B, N, K = q.mu.shape
        Sigma = torch.diag_embed(q.sigma.double())
        ops = model.relational_operators_by_layer[layer]
        tau = [cfg.kappa_beta * math.sqrt(d) for d in dims]
        coupling = torch.zeros(B, N, dtype=torch.float64)
        unrotated_differs = False
        for b in range(B):
            xi = [V[b, i] @ q.mu[b, i].double() for i in range(N)]
            C = [V[b, i] @ Sigma[b, i] @ V[b, i].T for i in range(N)]
            offset = 0
            for h, d in enumerate(dims):
                block = slice(offset, offset + d)
                offset += d
                R = build_rope_rotation(torch.arange(N), [d], base=ops.comparison_rope_base,
                                        fraction=ops.comparison_rope_fraction, dtype=torch.float64)   # (N, d, d)
                moments = [(R[i] @ xi[i][block], R[i] @ C[i][block, block] @ R[i].T) for i in range(N)]
                plain = [(xi[i][block], C[i][block, block]) for i in range(N)]
                for i in range(N):
                    bias = log_prior[0 if log_prior.shape[0] == 1 else b, h if log_prior.shape[1] > 1 else 0, i]
                    allowed = [j for j in range(N) if math.isfinite(float(bias[j]))]
                    pi = torch.softmax(bias[allowed], dim=0)
                    D = torch.tensor([_dense_kl(*moments[i], *moments[j]) for j in allowed], dtype=torch.float64)
                    beta = pi * torch.exp(-D / tau[h])
                    beta = beta / beta.sum()
                    expected = torch.zeros(N, dtype=torch.float64)
                    expected[allowed] = beta
                    assert torch.allclose(rows.beta[b, h, i], expected, rtol=1e-9, atol=1e-12), (layer, b, h, i)
                    coupling[b, i] += float((beta * D).sum())
                    D_plain = torch.tensor([_dense_kl(*plain[i], *plain[j]) for j in allowed], dtype=torch.float64)
                    unrotated_differs |= not torch.allclose(D_plain, D, rtol=1e-6, atol=1e-9)
        assert unrotated_differs                     # the rotation reaches the energies, so beta* reads j - i
        assert per_layer["belief_coupling"][layer] == pytest.approx(float(coupling.mean()), rel=1e-9)


def test_canonical_self_weight_is_the_mean_diagonal_of_beta_star() -> None:
    for overrides in (COMPARISON_ROPE, dict(beta_attention_prior="causal_noself")):
        model = _model(**overrides)
        snapshot = model.build_diagnostic_snapshot(TOKENS)
        per_layer = model.diagnostics_per_layer(TOKENS, snapshot=snapshot)
        for layer in range(model.cfg.n_layers):
            q = _input(snapshot, layer)
            rows = fe.canonical_routing_rows(q.mu, q.sigma, model._canonical_frame_inverse(snapshot, layer),
                                             snapshot.log_prior, tau=_tau(model, layer),
                                             irrep_dims=list(model.group.irrep_dims))
            support = causal_support(rows.beta, snapshot.log_prior)                 # (B, H, N, N)
            self_rows = (support.sum(-1) >= 2) & support.diagonal(dim1=-2, dim2=-1)
            diagonal = rows.beta.diagonal(dim1=-2, dim2=-1)
            value = per_layer["canonical_self_weight"][layer]
            if bool(self_rows.any()):
                assert value == pytest.approx(float(diagonal[self_rows].mean()), rel=1e-12)
                assert 0.0 < value <= 1.0
            else:
                assert math.isnan(value)                                            # causal_noself


# --------------------------------------------------------------------------- item 27: F at the readout output

def test_delta_f_is_the_difference_of_the_output_and_input_evaluations() -> None:
    model = _model(**COMPARISON_ROPE)
    snapshot = model.build_diagnostic_snapshot(TOKENS)
    per_layer = model.diagnostics_per_layer(TOKENS, snapshot=snapshot)
    for layer in range(model.cfg.n_layers):
        assert per_layer["canonical_total_delta"][layer] == (
            per_layer["canonical_total_output"][layer] - per_layer["total"][layer])
    for layer in range(1, model.cfg.n_layers):
        # The input evaluation of a snapshot whose previous output is this layer's converged output is the
        # output evaluation of the original: same prior, frames and temperature.
        outputs = list(snapshot.layer_outputs)
        outputs[layer - 1] = snapshot.layer_converged[layer]
        moved = dataclasses.replace(snapshot, layer_outputs=tuple(outputs))
        cap = model._layer_capture(snapshot, layer)
        beta = cap["beta_pre"]
        out = model._canonical_free_energy_diagnostics(moved, layer, beta, causal_support(beta, snapshot.log_prior))
        assert out["total"] == per_layer["canonical_total_output"][layer]


# --------------------------------------------------------------------------- item 38: NaN rows

def test_failed_factor_propagates_nan_and_an_empty_row_reports_zero() -> None:
    B, N, K, dims = 1, 4, 4, [2, 2]
    generator = torch.Generator().manual_seed(2)
    mu = torch.randn(B, N, K, generator=generator, dtype=torch.float64)
    sigma = 0.5 + torch.rand(B, N, K, generator=generator, dtype=torch.float64)
    sigma[0, 2, 0] = -1.0                                  # token 2: head 0 block not positive definite
    frame_inverse = torch.eye(K, dtype=torch.float64).expand(B, N, K, K)
    causal = torch.full((N, N), float("-inf"), dtype=torch.float64).triu(1)
    noself = causal.clone().fill_diagonal_(float("-inf"))
    rows = fe.canonical_routing_rows(mu, sigma, frame_inverse, torch.stack((causal, noself)),
                                     tau=torch.tensor([1.0, 2.0], dtype=torch.float64), irrep_dims=dims)
    for name in ("belief_coupling", "attention_entropy", "log_partition"):
        values = getattr(rows, name)[0, 0]                                           # head 0, (N,)
        assert bool(torch.isnan(values[2:]).all()), name                             # rows that read token 2
        assert bool(torch.isfinite(values[:2]).all()), name
    assert bool(torch.isnan(rows.beta[0, 0, 2:]).any(dim=-1).all())
    assert torch.isfinite(rows.beta[0, 0, :2]).all()
    head1 = rows.belief_coupling[0, 1], rows.attention_entropy[0, 1], rows.log_partition[0, 1]
    assert bool(torch.isfinite(head1[0]).all()) and bool(torch.isfinite(head1[1]).all())
    assert float(head1[0][0]) == 0.0 and float(head1[1][0]) == 0.0                  # empty noself row 0
    assert math.isinf(float(head1[2][0])) and float(head1[2][0]) < 0.0
    assert torch.equal(rows.beta[0, 1, 0], torch.zeros(N, dtype=torch.float64))


# --------------------------------------------------------------------------- item 53: per-head exponentials

@pytest.mark.parametrize("overrides", [
    dict(),                                                          # packed block_glk coordinates
    dict(gauge_group="block_so"),                                    # head_block_generators basis
    dict(pos_rotation="rope", beta_attention_prior="causal_alibi"),  # frame-space rotation blocks
    COMPARISON_ROPE,                                                 # comparison rotation on head blocks
    dict(family="gaussian_full"),
], ids=["block_glk", "block_so", "frame_rope", "comparison_rope", "full"])
def test_per_head_exponentials_equal_the_dense_path_in_float64(overrides: Dict[str, Any]) -> None:
    model = _model(**overrides)
    snapshot = model.build_diagnostic_snapshot(TOKENS)
    dims = list(model.group.irrep_dims)
    H, d = len(dims), dims[0]
    for layer in range(model.cfg.n_layers):
        compact = model._canonical_frame_inverse(snapshot, layer)
        dense = _dense_frame_inverse(model, snapshot, layer)
        assert compact.dtype == torch.float64 and tuple(compact.shape[-3:]) == (H, d, d)
        for h in range(H):
            assert torch.allclose(compact[:, :, h], dense[..., h * d:(h + 1) * d, h * d:(h + 1) * d],
                                  rtol=1e-12, atol=1e-14)
        q = _input(snapshot, layer)
        a, b = (fe.canonical_routing_rows(q.mu, q.sigma, inverse, snapshot.log_prior, tau=_tau(model, layer),
                                          irrep_dims=dims) for inverse in (compact, dense))
        for name in ("energy", "beta", "belief_coupling", "attention_entropy", "log_partition"):
            assert torch.allclose(getattr(a, name), getattr(b, name), rtol=1e-10, atol=1e-12), name


def test_unequal_head_towers_keep_the_dense_path() -> None:
    model = _model(gauge_group="so_n", group_n=3, irrep_spec=[("l1", 1), ("l2", 1)], embed_dim=8, n_heads=2,
                   phi_scale=0.3)
    snapshot = model.build_diagnostic_snapshot(TOKENS)
    inverse = model._canonical_frame_inverse(snapshot, 0)
    assert inverse.dim() == 4 and torch.allclose(inverse, _dense_frame_inverse(model, snapshot, 0), rtol=1e-13,
                                                 atol=1e-15)
    assert all(math.isfinite(v) for v in model.diagnostics_per_layer(TOKENS, snapshot=snapshot)["total"])


# --------------------------------------------------------------------------- item 31: inactive_fields, temperature

def test_inactive_fields_keep_the_diagnostic_inputs_live() -> None:
    def inactive(**overrides: Any) -> Dict[str, str]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return tiny_config(**overrides).inactive_fields()

    for n_layers in (1, 2):
        for mode, live in (("state_dependent", {"lambda_alpha_mode", "b0", "c0", "lambda_beta"}),
                           ("constant", {"lambda_alpha_mode", "lambda_alpha", "lambda_beta"})):
            off = inactive(lambda_alpha_mode=mode, n_layers=n_layers)
            on = inactive(lambda_alpha_mode=mode, n_layers=n_layers, diagnose_canonical_free_energy=True)
            assert live <= set(off), (mode, n_layers)
            assert not live & set(on), (mode, n_layers, live & set(on))
            assert "would feed that log-step diagnostic only" in off["lambda_beta"]
        # the mode rules still hold: the diagnostic reads only the coefficients of its alpha form
        assert "lambda_alpha" in inactive(lambda_alpha_mode="state_dependent", diagnose_canonical_free_energy=True)
        assert {"b0", "c0"} <= set(inactive(lambda_alpha_mode="constant", diagnose_canonical_free_energy=True))
    assert {"prior_handoff_rho", "prior_handoff_sigma"} <= set(inactive(n_layers=2))
    assert not {"prior_handoff_rho", "prior_handoff_sigma"} & set(
        inactive(n_layers=2, diagnose_canonical_free_energy=True))
    minimal_off = inactive(**MINIMAL_AGENT)
    minimal_on = inactive(**MINIMAL_AGENT, diagnose_canonical_free_energy=True)
    assert "kl_max" in minimal_off and "kl_max" not in minimal_on
    assert "kappa_beta" in minimal_on                       # the diagnostic uses the router's temperature 1


def test_minimal_agent_diagnostic_uses_the_routers_temperature() -> None:
    model = _model(**MINIMAL_AGENT, kappa_beta=1.7)
    snapshot = model.build_diagnostic_snapshot(TOKENS)
    per_layer = model.diagnostics_per_layer(TOKENS, snapshot=snapshot)
    dims = list(model.group.irrep_dims)
    for layer in range(model.cfg.n_layers):
        q = _input(snapshot, layer)
        inverse = model._canonical_frame_inverse(snapshot, layer)
        unit = fe.canonical_routing_rows(q.mu, q.sigma, inverse, snapshot.log_prior, tau=1.0, irrep_dims=dims)
        kappa = fe.canonical_routing_rows(q.mu, q.sigma, inverse, snapshot.log_prior,
                                          tau=1.7 * math.sqrt(dims[0]), irrep_dims=dims)
        assert per_layer["belief_coupling"][layer] == pytest.approx(
            float(unit.belief_coupling.sum(dim=1).mean()), rel=1e-12)
        assert not per_layer["belief_coupling"][layer] == pytest.approx(
            float(kappa.belief_coupling.sum(dim=1).mean()), rel=1e-6)


# --------------------------------------------------------------------------- default path bitwise unchanged

def _loss_and_grads(model: VFEModel, tokens: torch.Tensor, targets: torch.Tensor):
    model.zero_grad(set_to_none=True)
    _, loss, _ = model(tokens, targets)
    loss.backward()
    return loss.detach().clone(), {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}


@pytest.mark.parametrize("overrides", [COMPARISON_ROPE, MINIMAL_AGENT], ids=["comparison_rope", "minimal_agent"])
def test_forward_loss_and_gradients_are_bitwise_independent_of_the_diagnostics(overrides: Dict[str, Any]) -> None:
    off = _model(**overrides, diagnose_canonical_free_energy=False).train()
    on = _model(**overrides).train()
    targets = torch.roll(TOKENS, -1, dims=-1)
    loss_off, grads_off = _loss_and_grads(off, TOKENS, targets)
    loss_on, grads_on = _loss_and_grads(on, TOKENS, targets)
    assert torch.equal(loss_off, loss_on) and grads_off.keys() == grads_on.keys()
    assert all(torch.equal(grads_off[name], grads_on[name]) for name in grads_off)
    on.eval()
    on.diagnostics(TOKENS)                                  # fills the float64 rotation and frame caches
    off.eval()
    off.diagnostics(TOKENS)
    on.train()
    off.train()
    loss_after, grads_after = _loss_and_grads(on, TOKENS, targets)
    assert torch.equal(loss_after, loss_off)
    assert all(torch.equal(grads_after[name], grads_off[name]) for name in grads_off)


# --------------------------------------------------------------------------- training rows

def _train_run(tmp_path: Path, flag: bool) -> List[Dict[str, str]]:
    r"""metrics.csv rows of a 4-step tiny run (log every step, evaluate every 2) under comparison RoPE."""
    B, N, V = 2, 4, 7
    cfg = test_config(vocab_size=V, embed_dim=8, n_heads=2, batch_size=B, max_seq_len=N, n_layers=2,
                      warmup_steps=0, max_steps=4, log_interval=0, eval_interval=0, checkpoint_interval=0,
                      generate_embedding_figures=False, attention_mode="projected_predictive",
                      attention_residual=True, phi_scale=0.3, diagnose_canonical_free_energy=flag,
                      **{key: value for key, value in COMPARISON_ROPE.items() if key != "beta_attention_prior"})
    torch.manual_seed(5)
    model = VFEModel(cfg)
    loaders = [DataLoader(TokenWindows(torch.arange(41) % V, N), batch_size=B, shuffle=False),
               DataLoader(TokenWindows((torch.arange(23) * 3) % V, N, pad_final=True), batch_size=2,
                          shuffle=False)]
    run_dir = tmp_path / ("on" if flag else "off")
    artifacts = ra.RunArtifacts(run_dir, cfg, model, save_model_artifacts=False)
    torch.manual_seed(11)
    T.train(model, loaders[0], cfg, n_steps=4, grad_clip=1.0, log_interval=1, eval_interval=2,
            val_loader=loaders[1], device=torch.device("cpu"), logger=logging.getLogger("audit2-diagnostics"),
            artifacts=artifacts, generate_samples=False)
    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_training_rows_carry_the_new_columns_rectangularly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(T, "_save_eval_attention_maps", lambda *args, **kwargs: None)
    monkeypatch.setattr(ra, "_run_figures_isolated", lambda *args, **kwargs: True)
    rows = _train_run(tmp_path, True)
    assert len(rows) == 4
    new = (*PER_LAYER_KEYS, *AGGREGATE_KEYS)
    for row in rows:
        for key in new:
            assert math.isfinite(float(row[key])), key
        for key in CANONICAL_DIAGNOSTIC_KEYS:
            assert math.isfinite(float(row[key])), key
        evaluated = int(row["step"]) % 2 == 0
        for key in (*new, *CANONICAL_DIAGNOSTIC_KEYS):
            assert (row[f"val_{key}"] != "") == evaluated, (row["step"], key)
            if evaluated:
                assert math.isfinite(float(row[f"val_{key}"])), key
    # Without the flag only the canonical columns go (test_canonical_free_energy_20261010.py pins that set).
