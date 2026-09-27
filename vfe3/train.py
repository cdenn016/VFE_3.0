r"""Training (M-step) for VFE_3.0: AdamW per-group learning rates + warmup/cosine.

The model has no neural layers (no nn.Linear/MLP/activation). The trainable parameters are the
PriorBank prior tables plus the model-owned tables their toggles create -- the default
``pos_phi='learned'`` positional table, and the default-OFF exceptions (head mixer,
learnable T5 bias, linear decode).
``loss.backward()`` flows through the unrolled E-step to those tables; AdamW updates
them. The M-step minimizes the cross-entropy of the decode boundary over the prior
tables, with the E-step (the differentiable filtering kernel) unrolled into the graph,
so a gradient step on the priors improves inference end to end. Click-to-run: edit a
``VFE3Config`` and call ``run_training`` (no CLI).
"""

import contextlib
import json
import logging
import math
import time
from numbers import Real
from pathlib import Path
from typing import (Callable, Collection, Dict, Iterable, List, Mapping, NamedTuple, Optional,
                    Sequence, Tuple)

import torch

try:                                                # live per-step it/s via a tqdm progress bar
    from tqdm import tqdm as _tqdm                  # (plain tqdm, not tqdm.auto: the notebook
    from tqdm.contrib.logging import (              # widget is swallowed by some Run-button
        logging_redirect_tqdm as _redirect_logging,  # consumers / non-TTY stdout)
    )
except ImportError:                                 # tqdm optional: absent -> no bar, the periodic
    _tqdm = None                                    # log lines still emit at log_interval as before
    _redirect_logging = contextlib.nullcontext

from vfe3.contracts import is_conditional_filtering
from vfe3.config import VFE3Config
from vfe3.contracts import DataState, DataStateBuffer
from vfe3.data.datasets import make_dataloader
from vfe3.free_energy import attention_tau
from vfe3.gauge_optim import (
    GaugeManifoldAdamW,
    embedded_phi_frobenius_norm,
    get_phi_update_policy,
    phi_projection_chunk_rows,
    project_phi_parameter_rows_,
)
from vfe3.metric_contracts import perplexity_from_ce
from vfe3.model.block import _as_coeff
from vfe3.model.model import DiagnosticSnapshot, VFEModel
from vfe3.model.prior_bank import PriorBank, get_decode_registration
from vfe3.run_artifacts import RunArtifacts          # top-level safe: run_artifacts imports evaluate
#                                                      lazily (function-local), so there is no cycle
from vfe3.runtime import seed_everything
from vfe3.timing import TrainingTimer
from vfe3.viz.extract import (
    e_step_belief_trace as _snapshot_trace,
    e_step_fixed_point_diagnostics as _snapshot_fixed_point,
    converged_state as _snapshot_converged,
)
from vfe3.geometry.transport import TRANSPORT_CLAMP_MAX_NORM   # single source for the phi-clamp threshold (M2)
from vfe3.geometry.retraction import (                          # SPD-retraction non-finite tangent guard
    nonfinite_tangent_elements,
    reset_nonfinite_tangent_elements,
)
from vfe3.numerics import (                    # non-equivariant / non-PD fallback accounting
    decode_logdet_fallback_elements,
    mm_cholesky_fallback_elements,
    mu_trust_fallback_elements,
    reset_decode_logdet_fallback_elements,
    reset_mm_cholesky_fallback_elements,
    reset_mu_trust_fallback_elements,
)
from vfe3.relational_scoring import reset_source_recovery_elements   # projected float64 source-factor recovery count
from vfe3.geometry.transport import (                           # float64 frame-pair escalation count
    frame_pair_fp64_escalations,
    reset_frame_pair_fp64_escalations,
)


_PHI_CLAMP_WARNED:   bool = False
_SUCCESSFUL_UPDATES_KEY = "successful_updates"
def _successful_update_count(
    optimizer: torch.optim.Optimizer,

    *,
    legacy_default: int = 0,
) -> int:
    r"""Read the accepted-update clock persisted in optimizer parameter-group metadata.

    The checkpoint loader preserves this one runtime cursor while continuing to replace configured
    group metadata from the current run. A checkpoint written before this field existed falls back
    to its completed outer-step cursor.
    """
    if type(legacy_default) is not int or legacy_default < 0:
        raise ValueError("legacy successful-update default must be a non-negative integer")
    if not optimizer.param_groups:
        return legacy_default
    present = [_SUCCESSFUL_UPDATES_KEY in group for group in optimizer.param_groups]
    if any(present) and not all(present):
        raise RuntimeError("successful_updates must be present on every optimizer group or none")
    if not any(present):
        return legacy_default
    values = [group[_SUCCESSFUL_UPDATES_KEY] for group in optimizer.param_groups]
    if any(type(value) is not int or value < 0 for value in values):
        raise RuntimeError("successful_updates must be exact non-negative integers")
    if len(set(values)) != 1:
        raise RuntimeError("successful_updates must agree across all optimizer groups")
    return values[0]


def _set_successful_update_count(
    optimizer: torch.optim.Optimizer,
    count:     int,
) -> None:
    if type(count) is not int or count < 0:
        raise ValueError("successful-update count must be a non-negative integer")
    for group in optimizer.param_groups:
        group[_SUCCESSFUL_UPDATES_KEY] = count


def _warn_phi_transport_clamp(
    model:    VFEModel,

    max_norm: float = TRANSPORT_CLAMP_MAX_NORM,   # single source: stable_matrix_exp_pair's Frobenius clamp
) -> None:
    r"""Warn once per channel when a gauge-frame table exceeds the transport clamp or chart bound.

    ``stable_matrix_exp_pair`` rescales any ``||M||_F > max_norm`` (the shared transport-clamp
    default) and returns the surrogate ``exp(max_norm * M/||M||_F)``, NOT ``exp(M)``; its per-call
    monitor is opt-in because
    it costs a host sync on the E-step hot path. The M-step, however, steps ``phi_embed`` /
    ``pos_phi_free`` under plain AdamW with no configured projection, so a drifting
    row silently enters the surrogate regime (audit 2026-07-05 m8). This check runs only on
    log/eval-cadence steps (off the hot path) and shares the exact norm kernel used by the projected
    M-step. Certified orthogonal bases use their diagonal Gram; uncertified custom bases use the
    dense exact fallback in bounded chunks.

    When ``cfg.transport_chart_max_norm`` is set, the transport build RAISES (fail-closed) for any
    frame above it, which aborts training; the warning therefore trips at
    :math:`\min(\text{max\_norm}, \text{transport\_chart\_max\_norm})`, the first bound a drifting
    row actually meets.
    """
    global _PHI_CLAMP_WARNED
    chart_max_norm = getattr(getattr(model, "cfg", None), "transport_chart_max_norm", None)
    threshold = max_norm if chart_max_norm is None else min(max_norm, float(chart_max_norm))
    chart_binds = threshold < max_norm
    tables = [("belief.phi_embed", getattr(model.prior_bank, "phi_embed", None)),
              ("belief.pos_phi_free", getattr(model, "pos_phi_free", None))]
    with torch.no_grad():
        for name, tab in tables:
            if tab is None:
                continue
            if _PHI_CLAMP_WARNED:
                continue
            phi = tab.reshape(-1, tab.shape[-1])
            route = getattr(model.group, "phi_norm_route", lambda: "dense_fallback")()
            chunk_rows = phi_projection_chunk_rows(
                phi.shape[-1],
                model.group.generator_shape[-1],
                phi.element_size(),
                dense_fallback=route == "dense_fallback",
            )
            norm_max = torch.zeros((), device=phi.device, dtype=phi.dtype)
            for start in range(0, phi.shape[0], chunk_rows):
                norm = embedded_phi_frobenius_norm(
                    phi[start:start + chunk_rows],
                    model.group,
                    warn_fallback=False,
                )
                norm_max.copy_(torch.maximum(norm_max, norm.max()))
            if bool(norm_max > threshold):             # host sync: log-cadence only
                import warnings
                if chart_binds:
                    consequence = (
                        f"exceeds transport_chart_max_norm={threshold}; the fail-closed chart "
                        "validity check raises ValueError on the next transport build that reads this "
                        "row, which aborts training. Set phi_mstep_max_matrix_norm below the chart "
                        "bound (or lower m_phi_lr)")
                else:
                    consequence = (
                        f"exceeds the transport clamp max_norm={threshold}; stable_matrix_exp_pair "
                        "now returns the clamped surrogate exp(max_norm*M/||M||), not exp(M). Bound "
                        "the belief frame (mass_phi, lower m_phi_lr) or accept the surrogate transport")
                warnings.warn(
                    f"{name}: embedded gauge-frame Frobenius norm {float(norm_max):.2f} "
                    f"{consequence}. Warned once for this channel; further drift is not re-reported.",
                    RuntimeWarning, stacklevel=2,
                )
                _PHI_CLAMP_WARNED = True


def build_optimizer(
    model: VFEModel,
    cfg:   VFE3Config,
) -> torch.optim.Optimizer:
    r"""AdamW with per-group M-step learning rates over the PriorBank prior tables.

    The prior-table roles carry distinct natural scales, so each is given its own
    M-step learning rate: every live mean table at ``m_p_mu_lr``; every live (log) scale
    table and the decode temperature ``decode_log_scale`` together
    at ``m_p_sigma_lr``; the belief gauge-frame coordinates ``phi_embed`` at ``m_phi_lr``.
    The weight decay ``cfg.weight_decay`` is shared.

    Optional parameters are grouped only when their toggle is on: the linear decode weight
    ``output_proj_weight`` (decode_mode='linear') at ``m_p_mu_lr`` (a mean-readout scale); the
    PriorBank ``head_evidence_logits`` at ``m_head_evidence_lr`` and the post-belief head-mixer
    ``mixer_delta`` (use_head_mixer=True) at ``m_head_mixer_lr`` (each inherits ``m_p_mu_lr`` when
    its dedicated setting is None).
    A final assertion pins that the groups cover ``model.parameters()`` EXACTLY -- a new
    parameter that is forgotten here would otherwise silently never receive a gradient.
    Registry-owned trainable transport state is grouped at ``m_phi_lr`` (a gauge-connection scale),
    so a newly registered stateful transport trains without a parameter-name branch while the exact
    model-parameter coverage guard remains fail-closed.
    """
    pb = model.prior_bank
    # The selected policy contributes metadata only to stored phi-factor groups. An empty AdamW
    # policy leaves every established group dictionary unchanged. The pullback-group policy marks
    # those factors for stateless local-group staging and supplies weight_decay=0.
    phi_update_policy = get_phi_update_policy(cfg.m_phi_update_mode)
    phi_group_metadata = dict(phi_update_policy.optimizer_group_metadata)
    # Each group carries an explicit "role" in {mu, sigma, phi} -- the belief-component family it
    # steps (mean-LR / scale-LR / gauge-LR). The grad-norm decomposition (train_step) aggregates the
    # pre-clip grad by role, so the figure attributes the signal correctly REGARDLESS of group order
    # or which tables are live. Role is used in
    # preference to the group INDEX (the old 0/1/2 assumption broke whenever a config rerouted the
    # active mean/scale capacity off mu_embed/sigma_log_embed) and to the LR VALUE (m_p_mu_lr and
    # m_phi_lr may coincide). Extra dict keys ride alongside the optimizer metadata and are ignored
    # by base AdamW.
    phi_group = None
    if pb.phi_embed is not None:
        phi_group = {
            "params": [pb.phi_embed],
            "lr": cfg.m_phi_lr,
            "weight_decay": cfg.phi_weight_decay,
            "role": "phi",
        }
        phi_group.update(phi_group_metadata)
    # sigma_weight_decay (default None = inherit the global weight_decay, the long-standing
    # behavior): a dedicated AdamW decay for the log-variance tables. The global decay pulls
    # log sigma toward 0 (sigma toward 1) -- an unintended lognormal prior fighting the configured
    # sigma_init on the KL-decode path; sigma_weight_decay=0.0 exempts the sigma sector. Applied to
    # every sigma-role CAPACITY table (belief, untied decode).
    sigma_wd = {} if cfg.sigma_weight_decay is None else {"weight_decay": cfg.sigma_weight_decay}
    # mu_weight_decay (default None = inherit the global weight_decay, the long-standing behavior):
    # the exact mirror of sigma_wd for the MEAN-role capacity tables. Decoupled AdamW decay reaches
    # every row of a live embedding table on every step, so the rarely-visited rows are decayed far
    # faster than their gather-driven gradient can restore them; mu_weight_decay=0.0 exempts the
    # mean sector. Applied to the same class sigma_wd covers (belief, untied decode) plus
    # the linear-decode projection, which is the mean-role twin of decode_log_scale.
    mu_wd = {} if cfg.mu_weight_decay is None else {"weight_decay": cfg.mu_weight_decay}
    groups = []
    if pb.mu_embed is not None:
        groups.append({"params": [pb.mu_embed], "lr": cfg.m_p_mu_lr, "role": "mu", **mu_wd})
    # decode_temperature_learned=False freezes decode_log_scale (requires_grad=False in VFEModel), so
    # this filter keeps it out of every optimizer group and the readout temperature stays fixed.
    sigma_parameters = [parameter for parameter in (pb.sigma_log_embed, pb.decode_log_scale)
                        if parameter is not None and parameter.requires_grad]
    if sigma_parameters:
        groups.append({"params": sigma_parameters, "lr": cfg.m_p_sigma_lr,
                       "role": "sigma", **sigma_wd})
    if phi_group is not None:
        groups.append(phi_group)
    if getattr(pb, "decode_mu_embed", None) is not None:        # untie_decode_bank=True decode tables
        # Cloned from the encode tables at init (step-0 byte-identical decode), trained separately so
        # the decode direction can decouple from the E-step prior/self-coupling target; grouped like
        # the tables they were cloned from (mean@m_p_mu_lr, log-scale@m_p_sigma_lr).
        groups.append({"params": [pb.decode_mu_embed],        "lr": cfg.m_p_mu_lr,    "role": "mu", **mu_wd})
        groups.append({"params": [pb.decode_sigma_log_embed], "lr": cfg.m_p_sigma_lr, "role": "sigma", **sigma_wd})
    if pb.output_proj_weight is not None:                       # decode_mode='linear'
        groups.append({"params": [pb.output_proj_weight], "lr": cfg.m_p_mu_lr, "role": "mu", **mu_wd})
    if pb.output_proj_bias is not None:                         # decode_bias: learned log-unigram prior
        # weight_decay=0 -- decaying a unigram prior toward zero biases it to a flat distribution
        # (the same protection phi/Omega carry).
        groups.append({"params": [pb.output_proj_bias], "lr": cfg.m_p_mu_lr, "weight_decay": 0.0, "role": "mu"})
    if getattr(pb, "head_evidence_logits", None) is not None:
        head_evidence_lr = (cfg.m_p_mu_lr if cfg.m_head_evidence_lr is None
                            else cfg.m_head_evidence_lr)
        groups.append({"params": [pb.head_evidence_logits], "lr": head_evidence_lr,
                       "weight_decay": 0.0, "role": "mu",
                       "lr_aux_role": "head_evidence"})
    if getattr(model, "head_mixer", None) is not None:          # use_head_mixer=True Schur mixer
        head_mixer_lr = (cfg.m_p_mu_lr if cfg.m_head_mixer_lr is None
                         else cfg.m_head_mixer_lr)
        groups.append({"params": list(model.head_mixer.parameters()), "lr": head_mixer_lr,
                       "role": "mu", "lr_aux_role": "head_mixer"})
    if getattr(model, "block_mlps", None) is not None:         # use_block_mlp=True coordinate MLPs
        block_mlp_lr = (cfg.m_p_mu_lr if cfg.m_block_mlp_lr is None
                        else cfg.m_block_mlp_lr)
        groups.append({"params": list(model.block_mlps.parameters()), "lr": block_mlp_lr,
                       "weight_decay": cfg.weight_decay,
                       "role": "mu", "lr_aux_role": "block_mlp"})
    if getattr(model, "relational_operators", None) is not None:
        layers = model.relational_operators_by_layer
        for role, names in (("score", ("raw_score_q", "raw_score_k")),
                            ("content", ("T", "O"))):
            parameters = [getattr(ops, name) for ops in layers for name in names]
            parameters = [p for p in parameters if p is not None and p.requires_grad]
            if parameters:
                rate = getattr(cfg, f"m_relational_{role}_lr")
                groups.append({"params": parameters, "lr": cfg.m_p_mu_lr if rate is None else rate,
                               "weight_decay": 0.0, "role": "mu", "lr_aux_role": f"relational_{role}"})

    if getattr(model, "cg_coupling", None) is not None:         # use_cg_coupling=True CG path weights
        groups.append({"params": [model.cg_coupling.path_weights], "lr": cfg.m_p_mu_lr, "role": "mu"})
    if getattr(model, "pos_phi_free", None) is not None:        # pos_phi='learned' positional table
        # A gauge-frame scale, decayed like phi_embed (both compose into the ONE clamped frame, so
        # they share phi_weight_decay), but with its OWN optional step size: m_pos_phi_lr=None
        # inherits m_phi_lr, which is the historical yoked behavior.
        pos_lr = cfg.m_phi_lr if cfg.m_pos_phi_lr is None else cfg.m_pos_phi_lr
        pos_group = {"params": [model.pos_phi_free], "lr": pos_lr,
                     "weight_decay": cfg.phi_weight_decay, "role": "phi"}
        pos_group.update(phi_group_metadata)
        groups.append(pos_group)
    if getattr(model, "t5_bias", None) is not None:             # t5_learnable_bias=True relative-position bias
        # weight_decay=0: the per-bucket T5 bias b_{i-j} is a relative-position PRIOR shaping the
        # attention pi, not capacity; L2-decaying it toward zero biases the prior toward a flat/uniform
        # relative-position distribution (the same exemption output_proj_bias / r / the gauge frame
        # carry). role='mu' is the catch-all for learned non-variance/non-gauge tables (head_mixer,
        # ...); the bias is not a gauge frame, so it steps under the mean LR, not m_phi_lr.
        groups.append({"params": [model.t5_bias], "lr": cfg.m_p_mu_lr, "weight_decay": 0.0, "role": "mu"})
    # LayerNorm affine (layernorm_affine=True on a "layernorm" seam): learned per-feature gamma/beta
    # on the belief mean, carried by the block/final AffineLayerNorm nn.Modules. weight_decay=0 --
    # gamma/beta are normalization calibration, not capacity (decaying gamma toward 0 shrinks the
    # normalized signal; the same exemption t5_bias / output_proj_bias carry). role='mu'
    # is the catch-all for learned non-variance/non-gauge tables; NO gauge flag (the affine touches
    # no gauge transport), so it rides as plain AdamW even under GaugeManifoldAdamW.
    ln_affine = []
    for _nm in (getattr(model, "block_norm", None), getattr(model, "final_norm", None)):
        if isinstance(_nm, torch.nn.Module):
            ln_affine += [p for p in _nm.parameters() if p.requires_grad]
    if ln_affine:
        # Normalization calibration, not prior-table capacity, so it carries its own optional rate
        # (m_layernorm_lr=None inherits m_p_mu_lr, the historical behavior) and its own aux role, the
        # way head_mixer/block_mlp do. Both seams' gamma/beta share this ONE group.
        layernorm_lr = cfg.m_p_mu_lr if cfg.m_layernorm_lr is None else cfg.m_layernorm_lr
        groups.append({"params": ln_affine, "lr": layernorm_lr, "weight_decay": 0.0,
                       "role": "mu", "lr_aux_role": "layernorm"})

    # Stable CSV learning-rate fields are attached to semantic roles rather than optimizer indices.
    # When the base mean table is absent, the first realized mean-role group supplies lr_mu.
    reported_roles = set()
    for group in groups:
        role = group.get("role")
        if role in {"mu", "sigma", "phi"} and role not in reported_roles:
            group["lr_report_role"] = role
            reported_roles.add(role)

    # Exact-coverage guard: every TRAINABLE model parameter (requires_grad=True) must land in exactly
    # one group. A missing group would leave that weight frozen (no AdamW update) with no error -- the
    # bug class the optimizer is most prone to as new learnable seams (output_proj, head mixer, ...) are
    # added. Non-trainable params (requires_grad=False, e.g. a parameter frozen by design) are
    # intentionally exempt: they are fixed by design and need no optimizer group.
    # NOTE: this guards GROUPING/coverage, not gradient FLOW. A grouped parameter can still receive
    # a null gradient under specific opt-in toggles, by design: phi_embed under
    # e_step_gradient='detach'
    # (the E-step is detached; test-pinned in test_model.py), decode_log_scale under
    # decode_mode='linear' (the readout discards tau_eff), ALL encode tables under
    # decode_mode='linear' AND e_step_gradient='detach' (only output_proj_weight reaches the loss; the
    # model emits a warning for that combination). These are intentional route properties, not
    # coverage bugs.
    grouped = {p for g in groups for p in g["params"]}
    missing = {p for p in model.parameters() if p.requires_grad} - grouped   # frozen params are exempt
    if missing:
        raise AssertionError(
            f"build_optimizer left {len(missing)} model parameter(s) ungrouped; they would never "
            f"train. Add them to a param group."
        )

    if phi_update_policy.requires_manifold_optimizer:
        # Stored phi factors use one mixed optimizer; every other group remains standard AdamW.
        #
        # FUSED IS FORWARDED HERE (audit 2026-08-07). It used to be left unset on this route, which
        # is NOT the same as off: torch.optim.AdamW defaults `fused=None`, and with both `fused` and
        # `foreach` unresolved `adam()` selects the FOREACH multi-tensor path -- 9 kernel launches
        # per parameter group per step against 1 for the fused kernel. So choosing the manifold
        # route silently deoptimized every OTHER group (mu_embed, sigma/decode_log_scale,
        # transport_state, t5_bias, LayerNorm affine), none of which the manifold logic
        # touches. `step()` sets `p.grad = None` on every pullback_group parameter BEFORE
        # the trailing `super().step()`, so base AdamW skips exactly those and the fused kernel only
        # ever sees the standard groups.
        #
        # NOT VERIFIED ON DEVICE: `fused` is a CUDA-only path, so this change cannot be exercised on
        # a CPU box. If a manifold run regresses, this line is the first thing to revert.
        return GaugeManifoldAdamW(
            groups,
            model.group,
            phi_group_trust_radius=cfg.m_phi_group_trust_radius,
            phi_chart_max_norm=(
                cfg.phi_mstep_max_matrix_norm
                if cfg.phi_mstep_max_matrix_norm is not None
                else 5.0
            ),
            phi_bch_residual_max=(
                min(1e-6, cfg.bch_residual_max)
                if cfg.bch_residual_max is not None
                else 1e-6
            ),
            phi_precond_mode=cfg.phi_precond_mode,
            weight_decay=cfg.weight_decay,
            fused=next(model.parameters()).is_cuda,
        )
    # fused AdamW (one CUDA kernel for the whole M-step) when the priors live on CUDA; it is
    # CUDA-only, so on a CPU box this is the standard AdamW. Per-group LRs are honored either way.
    use_fused = next(model.parameters()).is_cuda
    return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay, fused=use_fused)


def _learning_rates_by_role(
    param_groups: Sequence[Dict[str, object]],
    lrs:          Sequence[float],
) -> Dict[str, float]:
    """Resolve the stable CSV learning-rate fields from optimizer role metadata."""
    if len(param_groups) != len(lrs):
        raise RuntimeError("optimizer parameter groups and scheduler learning rates differ in length")
    resolved = {
        str(group["lr_report_role"]): float(lr)
        for group, lr in zip(param_groups, lrs)
        if "lr_report_role" in group
    }
    required = {"mu", "sigma", "phi"}
    if set(resolved) != required:
        raise RuntimeError(
            f"optimizer learning-rate report roles must be exactly {sorted(required)}, "
            f"got {sorted(resolved)}"
        )
    return resolved


def _learning_rates_by_aux_role(
    param_groups: Sequence[Dict[str, object]],
    lrs:          Sequence[float],
) -> Dict[str, float]:
    """Resolve optional component-specific CSV learning rates from optimizer metadata.

    Unlike the required mean/sigma/phi report roles, an auxiliary role is absent when its
    component is disabled. Missing roles therefore resolve to NaN so every metrics row keeps the
    same auxiliary learning-rate columns and :class:`RunArtifacts` renders inactive values as blank cells.
    """
    if len(param_groups) != len(lrs):
        raise RuntimeError("optimizer parameter groups and scheduler learning rates differ in length")
    allowed = {"head_evidence", "head_mixer", "block_mlp", "relational_score", "relational_content",
               "layernorm"}
    resolved = {role: float("nan") for role in allowed}
    seen = set()
    for group, lr in zip(param_groups, lrs):
        role = group.get("lr_aux_role")
        if role is None:
            continue
        role = str(role)
        if role not in allowed:
            raise RuntimeError(f"unknown optimizer learning-rate auxiliary role {role!r}")
        if role in seen:
            raise RuntimeError(f"duplicate optimizer learning-rate auxiliary role {role!r}")
        seen.add(role)
        resolved[role] = float(lr)
    return resolved


def lr_lambda(
    step: int,
    cfg:  VFE3Config,
) -> float:
    r"""Learning-rate multiplier: linear warmup then cosine decay.

    Linear warmup to 1.0 over ``warmup_steps``, then a half-cosine to 0.0 at
    ``max_steps``::

        lr_mult(t) = t / warmup_steps                                 t <  warmup_steps
                   = 0.5 (1 + cos(pi * (t - warmup) / (max - warmup))) t >= warmup_steps

    The cosine argument is clamped to [0, pi] so steps beyond ``max_steps`` stay at 0.
    """
    if step < cfg.warmup_steps:
        return step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _floor_lr_lambdas(
    base_lrs: Sequence[float],
    cfg:      VFE3Config,
) -> List[Callable[[int], float]]:
    r"""Per-group ``LambdaLR`` multipliers that floor each group's ABSOLUTE LR at
    ``max(cfg.min_lr, cfg.min_lr_frac * base)``.

    A ``LambdaLR`` scales each group's base LR by its multiplier, so flooring the
    *multiplier* at ``max(min_lr/base, min_lr_frac)`` floors the *product*::

        base * max(min_lr/base, min_lr_frac, cosine) = max(min_lr, min_lr_frac*base, base*cosine).

    The absolute floor ``min_lr`` is shared across groups; the fractional floor
    ``min_lr_frac`` scales with each group's own base, preserving the m_mu:m_sigma:m_phi
    ratios into the cosine tail. A group whose base LR is 0 (a deliberately frozen
    channel, e.g. ``m_phi_lr=0``) drops the ``min_lr/base`` term (no division by zero)
    and stays frozen: ``min_lr`` does not resurrect it. With ``min_lr=min_lr_frac=0``
    this is the pure half-cosine-to-zero.
    """
    def make(base: float) -> Callable[[int], float]:
        abs_mult = cfg.min_lr / base if base > 0.0 else 0.0     # shared absolute floor (skip if frozen)
        floor    = max(abs_mult, cfg.min_lr_frac)               # combined per-group multiplier floor
        return lambda s: max(floor, lr_lambda(s, cfg))
    return [make(b) for b in base_lrs]


def _default_sample_decoder(
    cfg:     VFE3Config,

    *,
    dataset: Optional[str] = None,        # when known, keys the tokenizer instead of guessing
    cache_dir: Optional[Path] = None,
) -> 'Optional[Callable[[Sequence[int]], str]]':
    r"""A best-guess tiktoken ``decode(ids) -> str`` from ``cfg.vocab_size``, or None.

    Activates ONLY for a recognized real-corpus tokenizer vocab -- gpt2 (~50257) or cl100k
    (~100277) -- so a click-to-run on wikitext-*/wiki-* prints sample text with no wiring, while
    a tiny synthetic/test vocab (e.g. 6) gets no decoder and stays silent (the pure path is
    preserved without an extra toggle). The ranges tolerate vocab padding. Lazy-imports tiktoken;
    returns None if it is absent. An explicit ``sample_decode`` argument always takes precedence."""
    if dataset is not None:
        # Dataset-keyed lookup wins (audit 2026-07-27). The vocab-range heuristic below is a guess:
        # cfg.vocab_size is an independent user-set field, so padding a GPT-2 corpus to a
        # GPU-friendly multiple above 90,000 silently selected cl100k and printed wrong sample text
        # with no warning. datasets.get_tiktoken_decoder resolves the encoding from the SAME mapping
        # that built the cache filename and validated its provenance, so it cannot disagree.
        from vfe3.data.datasets import get_tiktoken_decoder
        return get_tiktoken_decoder(dataset, cache_dir=cache_dir)
    try:
        import tiktoken
    except ImportError:
        return None
    if 40_000 <= cfg.vocab_size <= 60_000:
        enc = tiktoken.get_encoding("gpt2")
    elif 90_000 <= cfg.vocab_size <= 110_000:
        enc = tiktoken.get_encoding("cl100k_base")
    else:
        return None
    return lambda ids: enc.decode([int(t) for t in ids])


class PeriodicGenerationRestorationError(RuntimeError):
    """Periodic sampling changed state that could not be restored exactly."""


def _periodic_generation(
    model:      torch.nn.Module,
    prompt:     torch.Tensor,
    new_tokens: int,
    decode:     Callable[[Sequence[int]], str],
) -> Tuple[str, str]:
    """Generate in eval mode without changing module modes or global RNG streams."""
    modules = tuple(model.modules())
    training_flags = tuple(module.training for module in modules)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = (
        tuple(state.clone() for state in torch.cuda.get_rng_state_all())
        if torch.cuda.is_available()
        else None
    )
    try:
        model.eval()
        generated = model.generate(prompt, new_tokens, greedy=True)[0]
        prompt_text = decode(prompt[0].tolist())
        continuation_text = decode(generated[prompt.shape[1]:].tolist())
        return prompt_text, continuation_text
    finally:
        restoration_errors = []
        for module, was_training in zip(modules, training_flags):
            try:
                module.training = was_training
            except BaseException as exc:
                restoration_errors.append(("module mode", exc))
        try:
            torch.set_rng_state(cpu_rng)
        except BaseException as exc:
            restoration_errors.append(("CPU RNG", exc))
        if cuda_rng is not None:
            try:
                torch.cuda.set_rng_state_all(list(cuda_rng))
            except BaseException as exc:
                restoration_errors.append(("CUDA RNG", exc))
        if restoration_errors:
            detail = "; ".join(
                f"{label}: {type(exc).__name__}: {exc}"
                for label, exc in restoration_errors
            )
            raise PeriodicGenerationRestorationError(
                f"periodic generation restoration failed ({detail})"
            ) from restoration_errors[0][1]


def _excluded_token_count(decode_stats: object, expected: int) -> torch.Tensor:
    """Resolve the optional legacy exclusion channel without a host synchronization."""
    scored = decode_stats.scored_tokens
    excluded = decode_stats.excluded_tokens
    if excluded is not None:
        return excluded
    return scored.new_tensor(expected, dtype=torch.int64) - scored


def _gradient_isfinite(gradient: torch.Tensor) -> torch.Tensor:
    r"""Test real gradient finiteness via ``isfinite(max_i |g_i|)`` without a full Boolean table.

    The infinity norm has no sum-of-squares overflow. Complex/empty/non-strided inputs retain
    the elementwise predicate because complex magnitude can overflow for finite components.
    """
    gradient = gradient.detach()
    if (gradient.layout == torch.strided and gradient.is_floating_point()
            and gradient.numel() > 0):
        return torch.isfinite(torch.linalg.vector_norm(gradient, ord=float("inf")))
    return torch.isfinite(gradient).all()


def _transfer_targets(
    targets: torch.Tensor,             # (B, N) CPU-loader labels or public/custom input
    model:   VFEModel,
    device:  torch.device,
) -> torch.Tensor:
    """Validate an owned CPU batch before transfer for the built-in linear fused decoder.

    Public/custom decoders and device-origin loaders retain their normal validation. Views made
    later (accumulation or grouped padding) do not inherit the one-use record and revalidate.
    """
    if (type(model) is VFEModel and type(model.prior_bank) is PriorBank
            and model.cfg.decode_mode == "linear"
            and get_decode_registration("linear").fused_ce is PriorBank.decode_ce_linear_chunked
            and type(targets) is torch.Tensor and targets.device.type == "cpu"
            and targets.dtype == torch.long and targets.layout == torch.strided
            and not torch.is_inference_mode_enabled()):
        # Isolate both mutable loader aliases and the source of an asynchronous device copy.
        owned = torch.empty_like(targets, pin_memory=targets.is_pinned()).copy_(targets)
        has_ignored = model.prior_bank._validate_fused_ce_targets(owned)
        transferred = owned.to(device, non_blocking=True)
        transferred._vfe3_host_target_validation = (
            id(transferred), transferred._version, model.prior_bank.vocab_size, -100, has_ignored)
        return transferred
    return targets.to(device, non_blocking=True)


def train_step(
    model:     VFEModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    tokens:    torch.Tensor,             # (B, N) input token ids
    targets:   torch.Tensor,             # (B, N) next-token ids (-100 = ignore)

    *,
    grad_clip:        Optional[float]                    = 1.0,
    grad_accum_steps: int                                = 1,
    scaler:           Optional['torch.amp.GradScaler']  = None,
    metrics_out:      Optional[dict]                     = None,
    status_out:       Optional[dict]                     = None,
) -> float:
    r"""One M-step (one optimizer step) on the cross-entropy of a batch; returns the loss.

    Zeroes the prior-table gradients, runs the forward (encode -> unrolled E-step ->
    decode -> CE), backpropagates the loss through inference to the prior tables, clips
    each optimizer role's gradient norm to ``grad_clip``, then takes one AdamW + scheduler step.
    ``grad_clip`` of ``None`` or ``0.0`` disables clipping entirely (the optimizer and
    scheduler still step on the raw gradient). After accumulation and unscale, a positive
    threshold is applied once independently to each optimizer role (mu/sigma/phi/other).

    With ``grad_accum_steps == K > 1`` the batch is split into ``K`` equal-size chunks along
    the batch axis. Each microbatch mean loss is multiplied by the decoder's exact scored-token
    count before ``backward()``, producing token-SUM gradients. After the ONE GradScaler unscale,
    all accumulated gradients are divided by the total scored-token count, then the single clip +
    ``optimizer.step()`` + ``scheduler.step()`` fires. This is the exact token-level objective even
    when decoder exclusions make the microbatch counts uneven; the existing non-CE regularizers
    remain token-weighted in that case and emit a warning because their native reductions differ.
    A "step" stays an OPTIMIZER step (scheduler/warmup/max_steps accounting is unchanged), and the
    clip threshold is not rescaled by ``K``. The returned loss is the same scored-token-weighted
    accumulation-boundary objective. A zero-scored accumulation skips every update-side effect.
    ``K == 1`` keeps the single-backward path (no chunking or gradient renormalization). Exact
    target counts stay on-device on silent steps; partition-validity and zero-scored flags ride
    the existing host transfer. Requires ``B % K == 0``.

    ``scaler`` is an optional :class:`torch.amp.GradScaler` for fp16 training (prevents
    gradient underflow through the unrolled E-step). A disabled scaler (``enabled=False``)
    is a documented no-op: ``scale`` is identity, ``unscale_`` is a no-op, and ``step``
    calls ``optimizer.step()`` directly — so passing ``scaler=None`` (or an
    ``enabled=False`` instance) keeps this function byte-identical to the unscaled path.

    When ``status_out`` is provided, ``status_out["did_step"]`` records whether the optimizer
    accepted the update. It is false for the explicit nonfinite gate and when an enabled
    GradScaler decreases its scale after ``update()``, which signals an overflow-skipped step.
    """
    # A disabled scaler is a documented no-op (scale -> identity, unscale_ -> nothing,
    # step -> optimizer.step()), so scaler=None keeps this path byte-identical to the unscaled loop.
    _scaler = scaler if scaler is not None else torch.amp.GradScaler(
        device=tokens.device.type, enabled=False)

    successful_updates = _successful_update_count(optimizer)
    optimizer.zero_grad(set_to_none=True)
    _mb_tok: List[int] = []                                     # per-microbatch counted-token spread (accum only)
    # E-step belief-gradient capture: a dict the forward fills with the raw ||grad_mu/sigma/phi|| of F
    # (the inference analogue of the M-step per-role grad norms). Created ONLY when metrics are being
    # logged this step; None -> the forward skips the capture entirely (zero overhead, byte-identical).
    _egrad = {} if metrics_out is not None and grad_accum_steps == 1 else None
    _egrad_sums: Dict[str, float] = {}
    _egrad_counts: Dict[str, int] = {}
    if grad_accum_steps == 1:                                   # default path: byte-identical to the single-step loop
        _, loss, decode_stats = model(
            tokens, targets, estep_grad_out=_egrad, return_decode_stats=True)
        ce = decode_stats.ce
        _scored_det = decode_stats.scored_tokens.detach()
        _excluded_det = _excluded_token_count(decode_stats, targets.numel()).detach()
        _expected_det = torch.tensor(targets.numel(), dtype=torch.int64, device=targets.device)
        _scaler.scale(loss).backward()
        _loss_det = loss.detach()                               # host read DEFERRED: fused with the grad-finite
        step_loss = None                                        # flag at the gate below (audit 2026-07-01 round-3)
        # CE is synced to a Python float only when a metrics dict is being filled this step (it feeds
        # metrics_out['train_ce'] and nothing else); on a silent step the extra D2H copy is skipped.
        step_ce = (float(ce.detach()) if (ce is not None and metrics_out is not None) else float("nan"))
    else:
        if tokens.shape[0] % grad_accum_steps != 0:            # equal-size microbatches require an even split
            raise ValueError(
                f"grad_accum_steps={grad_accum_steps} must divide the batch size "
                f"{tokens.shape[0]} for equal microbatches; got remainder "
                f"{tokens.shape[0] % grad_accum_steps}."
            )
        tok_chunks = torch.chunk(tokens, grad_accum_steps, dim=0)
        tgt_chunks = torch.chunk(targets, grad_accum_steps, dim=0)
        count_tensors = []
        excluded_count_tensors = []
        expected_count_tensors = []
        loss_numerator = torch.zeros((), dtype=torch.float64, device=tokens.device)
        ce_numerator = torch.zeros((), dtype=torch.float64, device=tokens.device)
        for tok_mb, tgt_mb in zip(tok_chunks, tgt_chunks):
            _egrad_mb = {} if metrics_out is not None else None
            _, loss_mb, decode_stats_mb = model(
                tok_mb, tgt_mb, estep_grad_out=_egrad_mb, return_decode_stats=True)
            ce_mb = decode_stats_mb.ce
            count_mb = decode_stats_mb.scored_tokens.detach()
            count_tensors.append(count_mb)
            excluded_count_tensors.append(
                _excluded_token_count(decode_stats_mb, tgt_mb.numel()).detach())
            expected_count_tensors.append(torch.tensor(
                tgt_mb.numel(), dtype=torch.int64, device=tgt_mb.device))
            if _egrad_mb is not None:
                for _name, _value in _egrad_mb.items():
                    if isinstance(_value, Real):
                        _egrad_sums[_name] = _egrad_sums.get(_name, 0.0) + float(_value)
                        _egrad_counts[_name] = _egrad_counts.get(_name, 0) + 1
            count_weight = count_mb.to(dtype=loss_mb.dtype)
            _scaler.scale(loss_mb * count_weight).backward()
            loss_numerator = loss_numerator + (
                loss_mb.detach().to(dtype=torch.float64) * count_mb)
            ce_numerator = ce_numerator + (
                ce_mb.detach().to(dtype=torch.float64) * count_mb)
        _counts = torch.stack(count_tensors)
        _scored_det = _counts.sum(dtype=torch.int64)
        _excluded_det = torch.stack(excluded_count_tensors).sum(dtype=torch.int64)
        _expected_det = torch.stack(expected_count_tensors).sum(dtype=torch.int64)
        _denominator = _scored_det.clamp_min(1).to(dtype=torch.float64)
        _loss_det = loss_numerator / _denominator
        _ce_det = ce_numerator / _denominator
        step_loss = None
        step_ce = float(_ce_det) if metrics_out is not None else float("nan")
        _mb_tok[:] = [int(value) for value in _counts.cpu().tolist()]
        # Decoder exclusions, not merely ignore_index, define the CE denominator. Weighting the
        # non-CE regularizers by that same count remains approximate when microbatch counts differ.
        if _mb_tok and max(_mb_tok) != min(_mb_tok):
            import warnings
            warnings.warn(
                "grad_accum_steps>1 with uneven counted-token microbatches: non-CE regularizers "
                "(mass_phi, mstep_self_coupling) are token-weighted by n_mb/n_tot "
                "rather than by their own reduction, so their accumulated gradient is an "
                "approximation. Use an unpadded/equal-token loader or grad_accum_steps=1 for the "
                "exact objective.",
                RuntimeWarning, stacklevel=2,
            )
    _target_partition_valid_det = _target_partition_valid_from_device(
        _expected_det, _scored_det, _excluded_det)
    _zero_scored_det = _scored_det == 0
    _scaler_enabled = scaler is not None and scaler.is_enabled()
    target_partition_valid = None
    zero_scored_targets = None
    # The enabled scaler's ordinary finite-loss path delegates overflow detection to GradScaler.
    # Resolve the scalar loss first so the rare nonfinite-loss branch can explicitly inspect gradients
    # and distinguish scale backoff (nonfinite gradients) from scale hold (finite gradients).
    if _scaler_enabled:
        if step_loss is None:
            _loss_flags = torch.stack((
                _loss_det.to(dtype=torch.float64),
                _target_partition_valid_det.to(dtype=torch.float64),
                _zero_scored_det.to(dtype=torch.float64),
            )).tolist()
            step_loss = _loss_flags[0]
            target_partition_valid = bool(_loss_flags[1])
            zero_scored_targets = bool(_loss_flags[2])
        loss_finite = math.isfinite(step_loss)
    else:
        loss_finite = True                                  # resolved with the fused default-path scan below
    # Unscale once when clipping/metrics needs true-unit gradients or the enabled scaler must classify
    # a nonfinite scalar loss. GradScaler remembers that unscale_ ran; its later step does not repeat it.
    need_unscale = (
        grad_accum_steps > 1
        or
        (grad_clip is not None and grad_clip > 0)
        or (metrics_out is not None)
        or (_scaler_enabled and not loss_finite)
    )
    if need_unscale:
        _scaler.unscale_(optimizer)
    if grad_accum_steps > 1:
        # Accumulate token-SUM gradients above, then normalize exactly once in true gradient units.
        # This ordering is required for GradScaler: dividing scaled gradients before unscale would
        # change overflow detection and the realized objective.
        grad_denominator = _scored_det.clamp_min(1)
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.grad.div_(grad_denominator.to(dtype=parameter.grad.dtype))
    # Finite-GRADIENT gate (audit 2026-07-01 F1): a FINITE scalar loss can still carry a NaN/Inf
    # parameter gradient through the unrolled E-step on a degenerate batch; stepping AdamW on it
    # would permanently poison the exp_avg/exp_avg_sq moment buffers. Checked on EVERY step on the
    # disabled-scaler default path. The deferred loss, grad-finite flag, exact-partition validity,
    # and zero-scored status ride ONE fused D2H transfer. Exact int64 counts are materialized only
    # for requested metrics or an invalid-partition error, so the silent default path retains one
    # unconditional synchronization per step (audit 2026-07-01 round-3). The enabled fp16 scaler
    # path checks gradients internally via found_inf, but the scalar loss is still checked
    # explicitly because it can be nonfinite while every parameter gradient is finite.
    grad_finite = True
    explicit_grad_check = (
        not _scaler_enabled
        or metrics_out is not None
        or not loss_finite
    )
    _flags = (
        [_gradient_isfinite(p.grad)
         for g in optimizer.param_groups
         for p in g["params"] if p.grad is not None]
        if explicit_grad_check else []
    )
    if not _scaler_enabled and _flags and step_loss is None:    # fuse default-path loss + grad flag
        _values = torch.stack((
            _loss_det.to(dtype=torch.float64),
            torch.stack(_flags).all().to(dtype=torch.float64),
            _target_partition_valid_det.to(dtype=torch.float64),
            _zero_scored_det.to(dtype=torch.float64),
        )).tolist()
        step_loss = _values[0]
        grad_finite = bool(_values[1])
        target_partition_valid = bool(_values[2])
        zero_scored_targets = bool(_values[3])
    elif _flags:
        grad_finite = bool(torch.stack(_flags).all())
    if step_loss is None:                                       # fp16 / no-grads fallback: one plain loss sync
        _loss_flags = torch.stack((
            _loss_det.to(dtype=torch.float64),
            _target_partition_valid_det.to(dtype=torch.float64),
            _zero_scored_det.to(dtype=torch.float64),
        )).tolist()
        step_loss = _loss_flags[0]
        target_partition_valid = bool(_loss_flags[1])
        zero_scored_targets = bool(_loss_flags[2])
    if target_partition_valid is None or zero_scored_targets is None:
        raise RuntimeError("target-accounting gate was not resolved at the host-transfer barrier")
    if not target_partition_valid:
        # Materialize exact int64 values only on this fail-visible path so the exception retains
        # its actionable partition detail without charging the ordinary silent training cadence.
        _target_accounting_from_device(_expected_det, _scored_det, _excluded_det)
        raise RuntimeError("target accounting device partition is invalid")
    _target_partition = (
        _target_accounting_from_device(_expected_det, _scored_det, _excluded_det)
        if metrics_out is not None else None
    )
    loss_finite = math.isfinite(step_loss)
    if metrics_out is not None:
        # Pre-clip gradient health: the combined L2 norm plus
        # per-ROLE norms (mu/sigma/phi from each group's "role" tag in build_optimizer, aggregated in
        # quadrature across ALL groups carrying that role). Role -- not the old groups[0/1/2] index --
        # so the LIVE tables are attributed correctly under any config (previously mu/sigma logged a
        # flat 0 whenever a config rerouted the active capacity). Captured AFTER unscale_ but BEFORE
        # clip so the value is the true pre-clip gradient magnitude. The three roles partition every
        # group, so role grads sum in quadrature to grad_norm.
        total_sq: float = 0.0
        role_g_sq = {"mu": 0.0, "sigma": 0.0, "phi": 0.0}
        role_w_sq = {"mu": 0.0, "sigma": 0.0, "phi": 0.0}
        for g in optimizer.param_groups:
            gsq = sum(float(torch.linalg.vector_norm(p.grad.detach()).square())
                      for p in g["params"] if p.grad is not None)
            total_sq += gsq
            _aux_role = g.get("lr_aux_role")
            if _aux_role in ("relational_score", "relational_content"):
                metrics_out[f"grad_norm_{_aux_role}"] = gsq ** 0.5
            _role = g.get("role")
            if _role in role_g_sq:
                role_g_sq[_role] += gsq
                role_w_sq[_role] += sum(float(torch.linalg.vector_norm(p.detach()).square()) for p in g["params"])
        metrics_out["grad_norm"] = total_sq ** 0.5
        for _name in ("mu", "sigma", "phi"):
            metrics_out[f"grad_norm_{_name}"] = role_g_sq[_name] ** 0.5
            # weight norm of the same role: with the logged grad_norm + per-group LR, the
            # update-to-weight ratio (the ~1e-3 LR-scale sanity check) is derivable offline.
            metrics_out[f"weight_norm_{_name}"] = role_w_sq[_name] ** 0.5
        # E-step belief-gradient norms: the raw ||grad_mu/sigma/phi|| of F captured inside the last
        # E-step iteration (model.forward estep_grad_out) -- the INFERENCE analogue of the M-step
        # per-role grads above (parameter learning). Under accumulation, each contributing
        # microbatch is captured independently and the arithmetic mean is named explicitly.
        if _egrad is not None:
            for _name in ("mu", "sigma", "phi"):
                metrics_out[f"estep_grad_norm_{_name}"] = float(_egrad.get(_name, 0.0))
        for _name, _total in _egrad_sums.items():
            metrics_out[f"estep_grad_norm_{_name}_microbatch_mean"] = (
                _total / _egrad_counts[_name])
        metrics_out["loss_finite"] = float(loss_finite)
        metrics_out["train_ce"] = step_ce            # pre-step CE (matches step_loss; not a post-update re-forward)
        if _target_partition is None:
            raise RuntimeError("target accounting metrics were not materialized")
        metrics_out.update(_target_partition)
        if _mb_tok:                                             # grad_accum_steps>1: token-spread bias check
            metrics_out["grad_accum_tok_spread"] = float(max(_mb_tok) - min(_mb_tok))
        if scaler is not None and scaler.is_enabled():          # fp16: surface the loss-scale (Tier-2 health)
            metrics_out["grad_scale"] = float(scaler.get_scale())
    # A nonfinite scalar loss independently rejects every route. A finite loss does not imply finite
    # gradients, so the disabled-scaler path additionally applies the explicit gradient gate. The
    # enabled scaler performs its own gradient found_inf check inside step().
    skip_step = (
        zero_scored_targets
        or (not loss_finite)
        or (explicit_grad_check and not grad_finite)
    )
    if metrics_out is not None:
        metrics_out["grad_finite"] = float(grad_finite)
    if grad_clip is not None and grad_clip > 0 and not skip_step:
        clip_norms = []
        # Optimizer groups partition parameters into roles; each role gets one L2 bound.
        role_params: dict = {}
        for group in optimizer.param_groups:
            role_params.setdefault(group.get("role", "other"), []).extend(group["params"])
        for parameters in role_params.values():
            if parameters:
                clip_norms.append(torch.nn.utils.clip_grad_norm_(parameters, grad_clip))
        clip_finite = bool(torch.isfinite(torch.stack([norm.to(tokens.device) for norm in clip_norms])).all())
        skip_step = not clip_finite
        if metrics_out is not None:
            metrics_out["grad_clip_finite"] = float(clip_finite)
    if skip_step and _scaler_enabled and not explicit_grad_check:
        # Clipping may be the first overflow detector on a metrics-free step.
        # Resolve it before zero_grad so true overflow triggers scaler backoff.
        grad_flags = [_gradient_isfinite(p.grad) for group in optimizer.param_groups
                      for p in group["params"] if p.grad is not None]
        grad_finite = bool(torch.stack(grad_flags).all()) if grad_flags else True
    scale_before = float(_scaler.get_scale()) if _scaler_enabled else None
    if skip_step:
        optimizer.zero_grad(set_to_none=True)                # drop poisoned grads; do NOT step AdamW
        if _scaler_enabled and not grad_finite:
            _scaler.update()                                 # found_inf drives the configured scale backoff
        elif _scaler_enabled:
            _scaler.update(new_scale=scale_before)           # finite grads + nonfinite loss: do not grow scale
        else:
            _scaler.update()
    else:
        _scaler.step(optimizer)
        _scaler.update()
    did_step = not skip_step
    if scale_before is not None and float(_scaler.get_scale()) < scale_before:
        did_step = False
    if status_out is not None:
        status_out["did_step"] = did_step
    if metrics_out is not None:
        metrics_out["step_skipped"] = float(not did_step)
    _cfg = model.cfg
    if (
        did_step
        and getattr(_cfg, "m_phi_update_mode", "adamw") == "adamw"
        and _cfg.phi_mstep_max_matrix_norm is not None
    ):
        collect_projection_stats = metrics_out is not None
        if collect_projection_stats:
            projection_device = model.group.generator_device
            if projection_device.type == "cuda":
                with torch.cuda.device(projection_device):
                    projection_start = torch.cuda.Event(enable_timing=True)
                    projection_end = torch.cuda.Event(enable_timing=True)
                    projection_start.record()
                    projection_stats = project_phi_parameter_rows_(
                        model,
                        _cfg.phi_mstep_max_matrix_norm,
                        collect_stats=True,
                    )
                    projection_end.record()
                    projection_end.synchronize()
                    projection_ms = float(projection_start.elapsed_time(projection_end))
            else:
                projection_start_cpu = time.perf_counter()
                projection_stats = project_phi_parameter_rows_(
                    model,
                    _cfg.phi_mstep_max_matrix_norm,
                    collect_stats=True,
                )
                projection_ms = (time.perf_counter() - projection_start_cpu) * 1000.0
            metrics_out.update(projection_stats)
            metrics_out["phi_chart_projection_ms"] = projection_ms
            metrics_out["phi_chart_projection_stats_collected"] = 1.0
        else:
            project_phi_parameter_rows_(
                model,
                _cfg.phi_mstep_max_matrix_norm,
                collect_stats=False,
            )
    if did_step:
        successful_updates += 1
        scheduler.step()                   # accepted optimizer updates are the scheduler's clock
    _set_successful_update_count(optimizer, successful_updates)
    return step_loss


def _perplexity_from_ce(ce: float) -> float:
    """Compatibility wrapper around the shared exact CE/PPL contract."""
    return perplexity_from_ce(ce)


def _target_accounting(expected: int, scored: int, excluded: int) -> Dict[str, int]:
    """Validate and serialize one exact target partition."""
    expected, scored, excluded = int(expected), int(scored), int(excluded)
    if min(expected, scored, excluded) < 0 or expected != scored + excluded:
        raise RuntimeError(
            "target accounting requires expected_targets == scored_targets + excluded_targets; "
            f"got {expected} != {scored} + {excluded}")
    return {
        "expected_targets": expected,
        "scored_targets": scored,
        "excluded_targets": excluded,
    }


def _validated_target_count_tensors(
    expected: torch.Tensor,
    scored: torch.Tensor,
    excluded: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate the structural contract for exact device-side target counts."""
    tensors = (expected, scored, excluded)
    if any(not isinstance(value, torch.Tensor) or value.dtype != torch.int64
           for value in tensors):
        raise TypeError("target accounting device counts must all be torch.int64 tensors")
    if any(value.numel() != 1 for value in tensors):
        raise ValueError("target accounting device counts must all be scalar tensors")
    if len({value.device for value in tensors}) != 1:
        raise ValueError("target accounting device counts must share one device")
    return tensors


def _target_partition_valid_from_device(
    expected: torch.Tensor,
    scored: torch.Tensor,
    excluded: torch.Tensor,
) -> torch.Tensor:
    """Return an exact scalar device predicate without materializing target counts on the host."""
    expected, scored, excluded = _validated_target_count_tensors(
        expected, scored, excluded)
    return (
        (expected >= 0)
        & (scored >= 0)
        & (excluded >= 0)
        & (scored <= expected)
        & (excluded == expected - scored)
    )


def _target_accounting_from_device(
    expected: torch.Tensor,
    scored: torch.Tensor,
    excluded: torch.Tensor,
) -> Dict[str, int]:
    """Transfer exact device-side int64 counts together, then validate their partition."""
    tensors = _validated_target_count_tensors(expected, scored, excluded)
    values = torch.stack(tensors).cpu().tolist()
    return _target_accounting(*values)




def _evaluation_totals_from_device(
    total_nats: torch.Tensor,
    expected: torch.Tensor,
    scored: torch.Tensor,
    excluded: torch.Tensor,
) -> Tuple[float, Dict[str, int]]:
    """Transfer floating nats separately from the exact int64 target partition."""
    if not isinstance(total_nats, torch.Tensor) or total_nats.numel() != 1:
        raise TypeError("evaluation total_nats must be a scalar tensor")
    total_nats_value = total_nats.to(dtype=torch.float64).cpu().item()
    accounting = _target_accounting_from_device(expected, scored, excluded)
    return total_nats_value, accounting


@torch.no_grad()
def evaluate(
    model:  VFEModel,
    loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],   # yields (tokens, targets) batches

    *,
    max_batches:     Optional[int]          = None,
    tokens_per_char: Optional[float]        = None,
    device:          Optional[torch.device] = None,
) -> Dict[str, Optional[float]]:
    r"""Token-weighted corpus evaluation with distinct token and character bit metrics.

    .. math::
        \mathrm{CE} = \frac{\sum_b n_b\, \mathrm{ce}_b}{\sum_b n_b},\quad
        \mathrm{PPL} = e^{\mathrm{CE}},\quad
        \mathrm{BPT} = \frac{\mathrm{CE}}{\ln 2},\quad
        \mathrm{BPC} = \mathrm{BPT}\,\cdot\,\mathrm{tokens\_per\_char},

    with ``n_b`` the exact decoder-reported number of scored target tokens in batch ``b``.
    Aggregating by token count (not per-batch mean) reproduces one cross-entropy over
    the concatenated corpus, including a partial last batch. ``tokens_per_char`` is the
    bits-per-CHARACTER correction (``n_tokens / n_codepoints`` from
    :func:`vfe3.data.datasets.tokens_per_char`). ``bits_per_token`` is always published under its
    own name. When character normalization is unavailable, ``tokens_per_char=None`` leaves ``bpc``
    null rather than silently relabeling bits per token as bits per character.
    """
    if device is None:
        device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        total_nats = torch.zeros((), dtype=torch.float64, device=device)
        total_tok = torch.zeros((), dtype=torch.int64, device=device)
        total_expected = torch.zeros((), dtype=torch.int64, device=device)
        total_excluded = torch.zeros((), dtype=torch.int64, device=device)
        for i, (tokens, targets) in enumerate(loader):
            # Inspect CPU-loader padding before transfer; device-origin loaders retain checks.
            total_expected.add_(targets.numel())
            valid = targets != -100
            has_ignored = bool((~valid).any())
            grouped_trailing_padding = False
            if (has_ignored and tokens.ndim == 2 and targets.ndim == 2
                    and tokens.shape == targets.shape):
                valid_lengths = valid.sum(dim=1)
                positions = torch.arange(targets.shape[1], device=targets.device).unsqueeze(0)
                grouped_trailing_padding = torch.equal(
                    valid,
                    positions < valid_lengths.unsqueeze(1),
                )
            tokens = tokens.to(device, non_blocking=True)
            targets = _transfer_targets(targets, model, device)
            if grouped_trailing_padding:
                lengths = torch.unique(valid_lengths).cpu().tolist()
                valid_lengths = valid_lengths.to(device, non_blocking=True)
                for length in lengths:
                    if length == 0:
                        total_excluded.add_((valid_lengths == length).sum() * targets.shape[1])
                        continue
                    rows = valid_lengths == length
                    group_tokens = tokens[rows, :length]
                    group_targets = targets[rows, :length]
                    _, _, decode_stats = model(
                        group_tokens, group_targets, return_decode_stats=True)
                    n_b = decode_stats.scored_tokens
                    total_nats.add_(decode_stats.ce.to(dtype=torch.float64) * n_b)
                    total_tok.add_(n_b)
                    total_excluded.add_(_excluded_token_count(
                        decode_stats, group_targets.numel()))
                    total_excluded.add_(rows.sum() * (targets.shape[1] - length))
            else:
                _, _, decode_stats = model(tokens, targets, return_decode_stats=True)
                n_b = decode_stats.scored_tokens
                total_nats.add_(decode_stats.ce.to(dtype=torch.float64) * n_b)
                total_tok.add_(n_b)
                total_excluded.add_(_excluded_token_count(decode_stats, targets.numel()))
            if max_batches is not None and i + 1 >= max_batches:
                break               # draw exactly max_batches (process-then-break; no extra pull)
        total_nats_value, accounting = _evaluation_totals_from_device(
            total_nats, total_expected, total_tok, total_excluded)
        total_tok_value = accounting["scored_targets"]
        if total_tok_value == 0:
            raise ValueError(
                "evaluation produced no scored target tokens; no non-ignored target tokens "
                "remain scored, so metrics are undefined"
            )
        ce = total_nats_value / total_tok_value
    finally:
        if was_training:
            model.train()
    bits_per_token = ce / math.log(2.0)
    return {
        "ce":             ce,
        "ppl":            _perplexity_from_ce(ce),
        "bits_per_token": bits_per_token,
        "bpc":            (bits_per_token * tokens_per_char
                           if tokens_per_char is not None else None),
        **accounting,
    }


@torch.no_grad()
def _eval_protocol_train_ce(
    model:       VFEModel,
    tokens:      torch.Tensor,           # (B, N) disjoint training windows
    targets:     torch.Tensor,           # (B, N) next-token ids (-100 = ignore)
    eval_stride: int,                    # validation scores the last eval_stride targets per window
) -> float:
    r"""Training-batch CE scored under the validation sliding-window protocol.

    With ``eval_stride = s < N`` every held-out window after the first scores only its last ``s``
    targets, so each scored position sees at least :math:`N - s + 1` tokens of context, while the
    disjoint training windows score every position :math:`t = 0, \dots, N-1` uniformly. Comparing
    the two biases ``generalization_gap`` negative, because CE falls with context. This scores the
    same position set on the training batch,

    .. math::
        \mathrm{CE}_{\text{train}} = \frac{1}{|S|} \sum_{(b, t) \in S} -\log p(y_{b,t} \mid y_{b, \le t}),
        \qquad S = \{(b, t) : t \ge N - s,\ y_{b,t} \ne -100\},

    in evaluation mode like :func:`evaluate`, so both sides of the gap share one protocol. The CPU
    and device RNG streams are restored, so the call leaves the training stream untouched.
    """
    masked = targets.clone()
    masked[:, :max(0, targets.shape[-1] - eval_stride)] = -100             # (B, N) score the tail only
    device = tokens.device
    was_training = model.training
    with torch.random.fork_rng(devices=([device] if device.type == "cuda" else [])):
        model.eval()
        try:
            _, _, decode_stats = model(tokens, masked, return_decode_stats=True)
        finally:
            if was_training:
                model.train()
    return float(decode_stats.ce)


# Fixed column set for the held-out per-eval probes (initialized to NaN before training and carried
# forward like last_val, so every metrics.csv row carries the SAME columns -- rectangular -- whether
# or not an eval has run yet). Config-conditional keys (val_head_redundancy_js needs >= 2 heads;
# pos_loss_* needs targets and seq_len >= 4) stay NaN on runs where they do not apply.
_VAL_DIAG_KEYS = (
    "val_self_coupling", "val_self_divergence", "val_belief_coupling", "val_attention_entropy",
    "val_inner_alignment_energy_total",
    "val_attn_entropy", "val_effective_rank", "val_belief_cond_median", "val_attn_entropy_min",
    "val_attn_entropy_min_all", "val_attn_entropy_min_norm",
    "val_attn_collapsed_heads", "val_future_leakage", "val_row_sum_error",
    "val_pos_content_r2", "val_prev_token_mass", "val_period_match_mass", "val_head_redundancy_js",
    "estep_f_drop", "estep_f_nondecreasing_frac", "estep_f_comparisons",
    "estep_r_mu_last", "estep_r_sigma_last",
    "estep_r_phi_last", "estep_fp_kl", "estep_fp_mu_rms", "estep_fp_sigma_rms",
    "estep_fp_phi_rms", "estep_target_gap", "estep_beta_js", "estep_alpha_rms_delta",
    "pos_loss_first_q", "pos_loss_last_q", "pos_loss_ratio",
    "val_builder_resid",
    # held-out gauge / SPD / Fisher geometry (surfaced from the already-computed val diagnostics dict)
    "val_holonomy_wilson", "val_cocycle_residual", "val_gauge_invariant_spread",
    "val_fisher_trace_mean", "val_belief_cond_p95", "val_belief_lam_min", "val_belief_lam_max",
    "val_belief_cond_valid_frac",
    "val_phi_norm_mean", "val_phi_norm_std",
    "val_phi_matrix_norm_p95", "val_phi_matrix_norm_p99", "val_phi_matrix_norm_max",
    "val_phi_exp_clamp_frac", "val_phi_exp_scale_min", "val_vertex_cond_p99",
    "val_pos_phi_matrix_norm_p95", "val_pos_phi_matrix_norm_p99", "val_pos_phi_matrix_norm_max",
    "val_pos_phi_exp_clamp_frac", "val_pos_phi_exp_scale_min",
    "val_guard_sigma_floor_frac", "val_guard_sigma_ceil_frac", "val_guard_energy_klmax_frac",
    "val_guard_energy_klmax_frac_live", "val_live_pair_frac",
    "val_nonfinite_frac",
)


# Relational probes exist only for relational runs, but their columns must exist before
# the first validation and stay present (blank) between evaluations or after a failed replay.
_VAL_RELATIONAL_DIAG_KEYS = (
    "val_relational_legacy_global_f_available",
    "val_relational_score_klmax_frac", "val_relational_value_klmax_frac",
    "val_relational_conditional_cost_before", "val_relational_conditional_cost_after",
    "val_relational_conditional_cost_delta", "val_relational_conditional_cost_tokens",
    "val_relational_mean_displacement_norm", "val_relational_snapshot_ce",
    "val_relational_identity_content",
) + tuple("val_relational_" + key for key in (
    "score_raw_min_singular_value", "score_q_raw_min_singular_value",
    "score_k_raw_min_singular_value", "score_rank_per_head", "content_rank_per_head",
    "parameter_count", "content_max_singular_value", "content_effective_rank",
    "projected_cov_min_eigenvalue", "projected_cov_max_condition", "covariance_repair_count"))


class ValidationDiagnostics(NamedTuple):
    r"""Metrics and bounded held-out state shared by one periodic evaluation event."""

    metrics:   Dict[str, float]
    token_ids: torch.Tensor
    snapshot:  DiagnosticSnapshot


@torch.no_grad()
def _val_diagnostics(
    model:      VFEModel,
    val_loader: Iterable,
    device:     torch.device,

    *,
    batches:    int = 1,
) -> ValidationDiagnostics:
    r"""Held-out per-eval probes, averaged over the first ``batches`` validation batches.

    EVERY ``val_*`` column used to be a statistic of ONE fixed 64-token sequence (audit 2026-08-06
    F6): the worker below takes ``next(iter(val_loader))`` and then ``[:1]``, the loader is
    ``shuffle=False``, and the sampler is sequential -- so the same sequence was scored at every eval
    of every run. It was visible in the artifacts: the guard fractions are exact multiples of
    1/1280 (K*N) and 1/8192 (H*N^2), which only holds at B=1. That is a sample size of one, not a
    held-out estimate, and it is the denominator behind every held-out number the project reports.

    ``batches=1`` (the default) is the historical behavior EXACTLY -- one batch, one worker call, no
    averaging -- so runs on disk stay reproducible. Above 1, each batch contributes its own first
    sequence and the numeric columns are averaged.

    The token_ids and snapshot come from the FIRST batch regardless, because they feed the attention
    figures, which are a picture of one sequence and would be meaningless averaged. Later batches'
    snapshots are dropped as soon as their metrics are read: each carries dense held-out logits, and
    holding N of them is exactly the regression tests/test_audit_diagnostic_memory_20260720.py
    exists to prevent.

    Non-finite probes are skipped per column rather than poisoning the mean, so one failed replay
    costs that column one sample instead of the whole eval.
    """
    if batches <= 1:
        return _val_diagnostics_one(model, next(iter(val_loader)), device)

    first: Optional[ValidationDiagnostics] = None
    sums:   Dict[str, float] = {}
    counts: Dict[str, int]   = {}
    for index, batch in enumerate(val_loader):
        if index >= batches:
            break
        one = _val_diagnostics_one(model, batch, device)
        for key, value in one.metrics.items():
            if math.isfinite(value):
                sums[key]   = sums.get(key, 0.0) + value
                counts[key] = counts.get(key, 0) + 1
        if first is None:
            first = one
        else:
            del one                      # release this batch's snapshot (dense logits) immediately
    if first is None:                    # empty loader: preserve the old failure mode
        return _val_diagnostics_one(model, next(iter(val_loader)), device)
    averaged = {
        key: (sums[key] / counts[key]) if counts.get(key) else float("nan")
        for key in first.metrics
    }
    return ValidationDiagnostics(averaged, first.token_ids, first.snapshot)


_SNAPSHOT_FRAME_METHODS = {
    name: getattr(VFEModel, name) for name in (
        "build_diagnostic_snapshot", "forward_beliefs", "diagnostics",
        "_attention_map_for_belief", "_diagnostic_transport",
        "_validate_diagnostic_snapshot",
    )
}


def _can_share_snapshot_frames(model: VFEModel) -> bool:
    """Admit the built-in validation path whose frozen frame fields are read-only."""
    from vfe3.viz import extract as ex
    from torch.nn.modules import module as hooks
    from vfe3.families.base import get_family
    from vfe3.families.gaussian import DiagonalGaussian
    from vfe3.geometry.transport import _build_flat, get_transport_registration
    from vfe3.model.block_mlp import BlockMLP

    if (type(model) is not VFEModel or model.cfg.transport_mode != "flat"
            or model.cfg.e_phi_lr != 0.0
            or get_family(model.cfg.family) is not DiagonalGaussian
            or model.group.elementary_block_shape is None
            or get_transport_registration("flat").callable is not _build_flat
            or (model.block_mlps is not None and any(
                type(mlp) is not BlockMLP or mlp.covariance_contract != "passthrough"
                or getattr(mlp.forward_moments, "__func__", None) is not BlockMLP.forward_moments
                for mlp in model.block_mlps))
            or any(getattr(getattr(model, name), "__func__", None) is not method
                   for name, method in _SNAPSHOT_FRAME_METHODS.items())
            or (ex.e_step_belief_trace, ex.e_step_fixed_point_diagnostics, ex.converged_state)
               != (_snapshot_trace, _snapshot_fixed_point, _snapshot_converged)
            or hooks._global_forward_hooks or hooks._global_forward_pre_hooks):
        return False
    return not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())


def _terminal_validation_signature(
    model:  VFEModel,
    loader: Iterable,
) -> Optional[tuple]:
    """Bind the private final-metric handoff to a deterministic built-in evaluation.

    Tensor identity, storage, and version checks avoid copying model or corpus payloads.
    This deliberately excludes extension hooks, custom inputs, and numerical fallbacks.
    The admitted zero-worker sequential loader draws RNG only when its iterator is built;
    the launcher repeats that construction when it consumes the metrics.
    """
    from torch.utils.data import BatchSampler, DataLoader, SequentialSampler, default_collate
    from vfe3.data.datasets import TokenWindows
    from vfe3.model.prior_bank import _encode_per_token, get_encode

    if (torch.is_inference_mode_enabled() or type(loader) is not DataLoader
            or type(loader.dataset) is not TokenWindows
            or type(loader.sampler) is not SequentialSampler
            or type(loader.batch_sampler) is not BatchSampler
            or loader.sampler.data_source is not loader.dataset
            or loader.batch_sampler.sampler is not loader.sampler
            or loader.collate_fn is not default_collate or loader.num_workers != 0
            or loader.generator is not None or loader.worker_init_fn is not None
            or loader.persistent_workers or loader.timeout != 0
            or not _can_share_snapshot_frames(model)
            or type(model.prior_bank) is not PriorBank
            or getattr(model.forward, "__func__", None) is not VFEModel.forward
            or getattr(model.prior_bank.encode, "__func__", None) is not PriorBank.encode
            or get_encode(model.prior_bank.encode_mode) is not _encode_per_token
            or model.cfg.decode_mode != "linear"
            or get_decode_registration("linear").fused_ce is not PriorBank.decode_ce_linear_chunked
            or model.cfg.norm_type_block != "none" or model.cfg.norm_type_final != "none"
            or model.cg_coupling is not None
            or any("forward" in vars(module) for module in model.modules())):
        return None
    dataset = loader.dataset
    tensors = (*model.parameters(), *model.buffers(), dataset.tokens)
    if (type(dataset.tokens) is not torch.Tensor or dataset.tokens.device.type != "cpu"
            or any(type(tensor) not in (torch.Tensor, torch.nn.Parameter)
                   or tensor.is_inference() for tensor in tensors)):
        return None
    return (
        repr(vars(model.cfg)),
        tuple((id(module), module.training) for module in model.modules()),
        tuple((id(tensor), tensor.data_ptr(), tensor._version, tensor.shape, tensor.dtype,
               tensor.device) for tensor in tensors),
        (id(loader), id(dataset), dataset.seq_len, dataset.stride, dataset.pad_final,
         dataset.overlap_prefix, dataset.n, loader.batch_size, loader.drop_last,
         loader.batch_sampler.batch_size, loader.batch_sampler.drop_last),
        (nonfinite_tangent_elements(), mu_trust_fallback_elements(),
         mm_cholesky_fallback_elements(), decode_logdet_fallback_elements(),
         frame_pair_fp64_escalations()),
    )


@torch.no_grad()
def _val_diagnostics_one(
    model:      VFEModel,
    batch:      object,
    device:     torch.device,
) -> ValidationDiagnostics:
    r"""Held-out per-eval probes (the train-loop ``diagnostics`` runs on the live TRAIN batch).

    Computes the validation-side F decomposition, attention-map structure across ALL layers/heads
    (entropy collapse, causal-mask sanity, positional-vs-content, induction/copy, head redundancy),
    the E-step F-descent + belief-residual convergence certificate, and the per-position
    within-sequence loss. Off the graph (no_grad). Best-effort: the CALLER wraps this so a replay
    error simply leaves the previous values carried forward; the returned subset is ``.update``-d
    into a NaN-initialized dict, so the CSV stays rectangular.
    """
    import torch.nn.functional as F
    from vfe3 import metrics as M
    from vfe3.viz import extract as ex

    out: Dict[str, float] = {}
    val_tok, val_tgt = (batch if isinstance(batch, (tuple, list)) else (batch, None))
    diagnostic_tokens = val_tok[:1].to(device)
    diagnostic_targets = val_tgt[:1].to(device) if val_tgt is not None else None
    vn = max(int(diagnostic_tokens.shape[1]), 1)
    # This private validation lifetime only reads frames; public builders retain independent copies.
    snapshot = model.build_diagnostic_snapshot(
        diagnostic_tokens,
        **({"_share_frame_copies": True} if _can_share_snapshot_frames(model) else {}))

    vd = model.diagnostics(                                     # held-out F decomposition (per token)
        diagnostic_tokens,
        snapshot=snapshot,
    )
    if is_conditional_filtering(model):
        out.update({f"val_{key}": value for key, value in vd.items()
                    if key.startswith("relational_") and isinstance(value, (int, float))})
        out["val_attn_entropy"] = vd["attn_entropy"]
        out["val_attn_entropy_min"] = vd["attn_entropy_min"]
        out["estep_f_comparisons"] = 0.0
        out["estep_f_drop"] = float("nan")
        if diagnostic_targets is not None:
            out["val_relational_snapshot_ce"] = float(F.cross_entropy(
                snapshot.logits.reshape(-1, snapshot.logits.shape[-1]).float(),
                diagnostic_targets.reshape(-1)))
        return ValidationDiagnostics(out, diagnostic_tokens, snapshot)
    out["val_self_coupling"]      = vd["self_coupling"]     / vn
    out["val_self_divergence"]    = vd["self_divergence"]   / vn
    out["val_belief_coupling"]    = vd["belief_coupling"]   / vn
    out["val_attention_entropy"]  = vd["attention_entropy"] / vn
    out["val_inner_alignment_energy_total"] = vd["total"]   / vn
    # The "val_free_energy_total" alias is retired (audit 2026-07-25 F7): vd["total"] is assembled
    # with log_likelihood=None, so it omits -E_q[log p(o|x)] entirely and the name claimed a
    # quantity it never carried. multiseed_analysis.py still reads the old name when the correct
    # column is absent, so CSVs written before this change keep working.
    out["val_attn_entropy"]       = vd["attn_entropy"]
    out["val_effective_rank"]     = vd["effective_rank"]
    out["val_belief_cond_median"] = vd["belief_cond_median"]
    out["val_attn_entropy_min"]   = vd["attn_entropy_min"]
    # Held-out gauge / SPD / Fisher geometry: vd = model.diagnostics(val_tok) ALREADY computed the full
    # geometry dict above, so surfacing these is near-free -- the held-out counterpart to the train-batch
    # geometry-health columns, the more credible evidence for LEARNED geometry vs a train-batch artifact.
    out["val_holonomy_wilson"]         = vd["holonomy_wilson"]
    out["val_cocycle_residual"]        = vd["cocycle_residual"]
    out["val_gauge_invariant_spread"]  = vd["gauge_invariant_spread"]
    out["val_fisher_trace_mean"]       = vd["fisher_trace_mean"]
    out["val_belief_cond_p95"]         = vd["belief_cond_p95"]
    # Raw extremes so the ratio above is interpretable (audit 2026-08-06 F16): once
    # val_guard_sigma_floor_frac > 0 the condition number is lam_max/eps, a guard readout.
    out["val_belief_lam_min"]          = vd["belief_lam_min"]
    out["val_belief_lam_max"]          = vd["belief_lam_max"]
    # ... and the validity flag on the same row, so val_belief_cond_p95 can be read without
    # knowing that val_guard_sigma_floor_frac exists and is what invalidates it. 1.0 = the ratio
    # is a belief property everywhere; anything less and that fraction of tokens reports lam_max/eps.
    out["val_belief_cond_valid_frac"]  = vd["belief_cond_valid_frac"]
    out["val_phi_norm_mean"]           = vd["phi_norm_mean"]
    out["val_phi_norm_std"]            = vd["phi_norm_std"]
    for _source, _target in (
        ("phi_matrix_norm_p95", "val_phi_matrix_norm_p95"),
        ("phi_matrix_norm_p99", "val_phi_matrix_norm_p99"),
        ("phi_matrix_norm_max", "val_phi_matrix_norm_max"),
        ("phi_exp_clamp_frac", "val_phi_exp_clamp_frac"),
        ("phi_exp_scale_min", "val_phi_exp_scale_min"),
        ("vertex_cond_p99", "val_vertex_cond_p99"),
        ("pos_phi_matrix_norm_p95", "val_pos_phi_matrix_norm_p95"),
        ("pos_phi_matrix_norm_p99", "val_pos_phi_matrix_norm_p99"),
        ("pos_phi_matrix_norm_max", "val_pos_phi_matrix_norm_max"),
        ("pos_phi_exp_clamp_frac", "val_pos_phi_exp_clamp_frac"),
        ("pos_phi_exp_scale_min", "val_pos_phi_exp_scale_min"),
    ):
        if _source in vd:
            out[_target] = vd[_source]
    out["val_guard_sigma_floor_frac"]  = vd["guard_sigma_floor_frac"]
    out["val_guard_sigma_ceil_frac"]   = vd["guard_sigma_ceil_frac"]
    out["val_guard_energy_klmax_frac"] = vd["guard_energy_klmax_frac"]
    # Companion restricted to causally-admissible pairs (audit 2026-08-06 F17). The whole-grid
    # column above is KEPT as-is: masked pairs are not numerically inert, so their kl_max pin is a
    # real firewall reading. The pair tells you which triangle is actually saturating.
    out["val_guard_energy_klmax_frac_live"] = vd.get("guard_energy_klmax_frac_live", float("nan"))
    out["val_live_pair_frac"]               = vd.get("guard_energy_live_pair_frac", float("nan"))
    out["val_nonfinite_frac"]          = vd["nonfinite_frac"]

    # score_at="entry" is the beta the FORWARD actually used (audit 2026-08-05: the "converged"
    # default re-scores the belief AFTER the E-step moved it -- measured TV(forward, converged) up
    # to 0.96 per row, vs TV(forward, entry) == 0.000000, so every val_attn_* below was a statistic
    # of a post-hoc re-score rather than of the model). The converged/prior triple is still saved
    # separately under attention_estep/ for anyone who wants the comparison.
    # NO snapshot here: a snapshot stores only the CONVERGED maps, so model.py:3281 rejects the
    # pair outright (audit 2026-08-06 -- passing both raised ValueError at every single validation,
    # and because _val_diagnostics is called under one try/except at :1710 that NaNs the whole
    # _VAL_DIAG_KEYS set, it silently cost the free-energy decomposition, guard fractions, phi norms
    # and E-step certificate too, not just the attention block). Entry scoring must be replayed
    # from the model, exactly as the attention_estep/ figures at :1140 already do. The replay
    # re-scores token_ids[:1], which IS diagnostic_tokens (val_tok[:1] at :982), so the sample is
    # unchanged -- only the scoring point moves.
    amaps = model.attention_maps(                               # (L, H, N, N) all layers/heads
        diagnostic_tokens,
        score_at="entry",
    )
    # Row-support filter (audit 2026-08-05): attention_entropy_rows floors beta at eps, so a
    # hard-masked row of an N-key grid returns (#masked)*eps*|ln eps| -- a pure function of N and
    # eps. Unfiltered, val_attn_entropy_min_all was CONSTANT across every logged step of every run
    # (1.740754e-09 = 63*1e-12*ln(1e-12)) and val_attn_collapsed_heads pinned at n_heads, so the
    # collapsed-head alarm could never fire. model.py already applies this filter for the
    # val_attn_entropy_min twin; these two were missed.
    # The collapse test is scored on entropy NORMALIZED by each row's own ceiling ln(support_i)
    # (audit 2026-08-06 F22). A fixed ln 2 threshold is exactly the ceiling of the first two-key row,
    # so the strict `<` was decided by float noise -- measured 2 of 2 heads flagged on an untrained
    # model whose attention is as uniform as the causal mask permits. See metrics.attention_head_collapse.
    ent_rows = M.attention_entropy_rows(amaps)                  # (L, H, N)
    scorable = (amaps > 0.0).sum(dim=-1) >= 2                   # rows with >=2 admissible keys
    hmin = ent_rows.masked_fill(~scorable, float("inf")).min(dim=-1).values   # (L, H)
    finite_hmin = hmin[torch.isfinite(hmin)]
    out["val_attn_entropy_min_all"] = (float(finite_hmin.min()) if finite_hmin.numel()
                                       else float("nan"))
    worst_norm, collapsed = M.attention_head_collapse(amaps)    # (L, H) each
    finite_norm = worst_norm[torch.isfinite(worst_norm)]
    out["val_attn_entropy_min_norm"] = (float(finite_norm.min()) if finite_norm.numel()
                                        else float("nan"))
    out["val_attn_collapsed_heads"] = float(collapsed.float().sum())
    cs = M.causal_sanity(amaps)
    out["val_future_leakage"] = float(cs["future_leakage"].max())   # soft causal prior can leak silently
    out["val_row_sum_error"]  = float(cs["row_sum_error"].max())
    out["val_pos_content_r2"] = float(M.positional_content_score(amaps).mean())
    sh = M.structured_head_scores(amaps)
    out["val_prev_token_mass"]   = float(sh["prev_token"].mean())
    out["val_period_match_mass"] = float(sh["period_match"].mean())
    if amaps.shape[1] > 1:                                       # head redundancy needs >= 2 heads
        h = amaps.shape[1]
        off = ~torch.eye(h, dtype=torch.bool, device=amaps.device)
        out["val_head_redundancy_js"] = float(torch.stack(
            [M.head_redundancy_js(amaps[li])[off].mean() for li in range(amaps.shape[0])]).mean())

    tr = ex.e_step_belief_trace(                                # captured E-step trajectory
        model,
        diagnostic_tokens,
        snapshot=snapshot,
    )
    f = tr["free_energy"] / vn                                  # PER-TOKEN (free_energy_value is a per-seq SUM)
    out["estep_f_drop"] = float(f[-1] - f[0])                   # < 0 = F descended over the inner loop
    # fraction of inner iterations that did NOT decrease F. EXPECTED to be nonzero for parallel
    # (Jacobi) mean-field with a finite step (e_step.py: 'not guaranteed monotone per iteration') --
    # a descent-quality readout, not a convergence-FAILURE flag.
    out["estep_f_nondecreasing_frac"] = (
        float((f[1:] > f[:-1] + 1e-9).float().mean()) if f.numel() > 1 else 0.0)
    # The DENOMINATOR of that fraction, because without it the column is unreadable (audit
    # 2026-08-06 F6). The trace has n_e_steps+1 entries, so at the live n_e_steps=1 the "fraction"
    # is a mean over exactly ONE comparison -- a single Bernoulli draw that can only be 0.0 or 1.0,
    # which is precisely what every run on disk shows (one run all-0.0, five all-1.0). Reporting
    # "1.0" from one coin flip and "1.0" from twenty is the same number with wildly different
    # evidential weight; this column is what tells them apart. It also rises with
    # val_diagnostic_batches, since each batch contributes its own draws to the average.
    out["estep_f_comparisons"] = float(max(f.numel() - 1, 0))
    res = M.estep_residuals(                                      # last-iter belief change (SPD metric for sigma)
        tr["mu"], tr["sigma"], tr["phi"], diagonal=model.cfg.diagonal_covariance)
    for _nm, _key in (("r_mu", "estep_r_mu_last"), ("r_sigma", "estep_r_sigma_last"),
                      ("r_phi", "estep_r_phi_last")):
        out[_key] = float(res[_nm][-1].mean()) if res[_nm].numel() else 0.0
    out.update(ex.e_step_fixed_point_diagnostics(
        model,
        diagnostic_tokens,
        snapshot=snapshot,
    ))

    if diagnostic_targets is not None:                          # per-position within-sequence loss
        vlog = snapshot.logits                                  # (B, N, V) same captured inference path
        b, n = diagnostic_tokens.shape
        per = F.cross_entropy(vlog.reshape(-1, vlog.shape[-1]).float(), diagnostic_targets.reshape(-1),
                              ignore_index=-100, reduction="none").reshape(b, n)
        valid = (diagnostic_targets != -100).float()
        pos_ce = (per * valid).sum(0) / valid.sum(0).clamp(min=1.0)   # (N,) mean CE at each position
        q = n // 4
        if q > 0:
            out["pos_loss_first_q"] = float(pos_ce[:q].mean())
            out["pos_loss_last_q"]  = float(pos_ce[-q:].mean())
            out["pos_loss_ratio"]   = float(pos_ce[-q:].mean() / pos_ce[:q].mean().clamp(min=1e-9))

    # Builder-break gauge-equivariance residual per eval (A2/EXP-9): the head-mixer congruence defect
    # at the converged belief (~eps under the tied gauge; climbs as the untied block_glk mixer drifts
    # from identity -> the residual-drift-vs-step series). Only with a head mixer; isolated so a replay
    # fault drops just this scalar (NaN-rectangular), not the whole val-diag row.
    if model.head_mixer is not None:
        try:
            cst = ex.converged_state(model, diagnostic_tokens, snapshot=snapshot)
            br = M.head_mixer_gauge_residual(cst["mu"], cst["sigma"], model.head_mixer, model.group,
                                             diagonal=model.cfg.diagonal_covariance)
            out["val_builder_resid"] = float(torch.cat([br["mu_residual"], br["sigma_residual"]]).median())
        except Exception as exc:                                   # leave NaN; never fail the eval
            logging.getLogger(__name__).warning(
                "head-mixer builder-residual diagnostic failed (%s); val_builder_resid remains "
                "unavailable for this evaluation",
                exc,
            )
    return ValidationDiagnostics(out, diagnostic_tokens, snapshot)


def _save_eval_attention_maps(
    token_ids: torch.Tensor,                # (B, N) post-step evaluation batch

    model:     VFEModel,
    artifacts: RunArtifacts,
    logger:    logging.Logger,

    *,
    step:      int,
) -> None:
    r"""Save beta maps from one post-step evaluation batch.

    The shipped ``attention/`` set is scored at ``"entry"`` to match the CSV (audit 2026-08-06 F19).
    It previously came off the snapshot, which stores only the CONVERGED maps -- so after the metric
    path moved to ``"entry"`` on 2026-08-05 the same run directory published a figure and a column
    that measured DIFFERENT objects under the same name. Entry is the beta the forward actually used
    (measured TV(forward, converged) up to 0.96 per row, TV(forward, entry) == 0.0), so the figure
    follows the metric rather than the other way round. Scoring at entry cannot come from a snapshot,
    hence the replay; the converged and prior arms remain available under ``attention_estep/``.

    The ``attention_estep/`` set is a REPLAY (audit 2026-07-26 R-3) for the same reason. A replay
    failure is swallowed like every other figure -- the shipped maps and the run itself do not
    depend on it.
    """
    entry_maps = model.attention_maps(token_ids, score_at="entry")
    artifacts.save_attention_maps(step, entry_maps, logger=logger)
    try:
        estep_maps = torch.stack([
            entry_maps,
            model.attention_maps(token_ids, score_at="converged"),
            model.attention_maps(token_ids, score_at="prior"),
        ])                                                   # (3, L, H, N, N)
    except Exception as exc:                                 # noqa: BLE001 - figures are best-effort
        logger.warning("E-step attention replay failed (%s); attention_estep skipped", exc)
        return
    artifacts.save_estep_attention_maps(step, estep_maps, logger=logger)


class TrainingTerminalState(NamedTuple):
    r"""Snapshot of the resumable training state handed to a ``train`` terminal callback.

    Captured immediately after the final optimizer step: the completed-step count, live optimizer,
    scaler, and data cursor, plus cloned copies of the last-iterate
    ``state_dict`` and CPU/CUDA global RNG states. A validation-only finalizer scores validation,
    restores the captured model/RNG state, and writes a resumable checkpoint whose weights and
    optimizer moments describe the same iterate.
    """
    step:                 int
    optimizer:            torch.optim.Optimizer
    scaler:               Optional["torch.amp.GradScaler"]
    data_state:           Optional[DataState]
    raw_model_state:      Dict[str, torch.Tensor]
    rng_state:            Dict[str, object]


def _loader_data_identity(
    loader:     object,
    vocab_size: int,
) -> Dict[str, object]:
    r"""Return the exact data and iterator contract bound to a resumable cursor.

    Production loaders receive a cache-backed contract from ``make_dataloader``. A direct
    ``TokenWindows`` loader used by tests or library callers receives an equivalent in-memory
    contract whose bounded canonical digest still detects a different tensor at resume.
    """
    dataset = getattr(loader, "dataset", None)
    tokens = getattr(dataset, "tokens", None)
    if not isinstance(tokens, torch.Tensor):
        raise RuntimeError(
            "exact data resume requires loader.dataset.tokens or a cache-backed data identity")
    for field in ("seq_len", "stride", "pad_final"):
        if not hasattr(dataset, field):
            raise RuntimeError(
                f"exact data resume requires loader.dataset.{field} iterator metadata")

    sampler = getattr(loader, "sampler", None)
    batch_sampler = getattr(loader, "batch_sampler", None)
    if type(batch_sampler) is not torch.utils.data.BatchSampler:
        raise RuntimeError("exact data resume requires the standard torch BatchSampler")
    if type(sampler) is torch.utils.data.RandomSampler:
        sampler_kind = "random"
        sampler_replacement: Optional[bool] = bool(sampler.replacement)
    elif type(sampler) is torch.utils.data.SequentialSampler:
        sampler_kind = "sequential"
        sampler_replacement = None
    else:
        raise RuntimeError(
            "exact data resume supports only standard RandomSampler or SequentialSampler")
    batch_size = getattr(loader, "batch_size", None)
    drop_last = getattr(loader, "drop_last", None)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise RuntimeError("exact data resume requires a positive integer loader.batch_size")
    if not isinstance(drop_last, bool):
        raise RuntimeError("exact data resume requires a boolean loader.drop_last")
    if getattr(loader, "collate_fn", None) is not torch.utils.data.default_collate:
        raise RuntimeError("exact data resume requires PyTorch's default collate function")

    identity = getattr(dataset, "data_identity", None)
    if identity is not None:
        normalized = json.loads(json.dumps(identity, sort_keys=True, separators=(",", ":")))
        if normalized.get("model_vocab_size") != int(vocab_size):
            raise RuntimeError(
                "loader data identity model vocabulary does not match the active config")
    else:
        from vfe3.run_artifacts import _loader_token_content_summary
        digest, n_tokens, _counts = _loader_token_content_summary(
            loader,
            vocab_size=int(vocab_size),
        )
        assert digest is not None and n_tokens is not None
        normalized = {
            "schema_version":       2,
            "dataset":              f"in-memory:{type(dataset).__module__}.{type(dataset).__qualname__}",
            "split":                "train",
            "tokenizer_tag":        None,
            "tokenizer_encoding":   None,
            "tokenizer_vocab_size": None,
            "model_vocab_size":     int(vocab_size),
            "max_tokens":           None,
            "source": {
                "format":        "tensor",
                "tokenizer_tag": None,
                "size_bytes":    int(tokens.numel()) * int(tokens.element_size()),
                "sha256":        digest,
                "meta":          {"n_tokens": n_tokens, "dtype": str(tokens.dtype)},
                "meta_sha256":   None,
            },
        }
    normalized["schema_version"] = 2
    normalized["iterator"] = {
        "dataset_type":       f"{type(dataset).__module__}.{type(dataset).__qualname__}",
        "seq_len":            int(dataset.seq_len),
        "stride":             int(dataset.stride),
        "pad_final":          bool(dataset.pad_final),
        "n_windows":          int(len(dataset)),
        "batch_size":         batch_size,
        "drop_last":          drop_last,
        "sampler":            sampler_kind,
        "sampler_replacement": sampler_replacement,
        "sampler_num_samples": int(len(sampler)),
    }
    return normalized


def _training_cursor_fields(
    completed_step:  int,
    steps_per_epoch: int,
) -> Dict[str, 'float | int']:
    r"""Return a one-based data cursor for an absolute completed optimizer step."""
    if completed_step < 1:
        raise ValueError(f"completed_step must be >= 1, got {completed_step}")
    if steps_per_epoch < 1:
        raise ValueError(f"steps_per_epoch must be >= 1, got {steps_per_epoch}")
    epoch_index, batch_index = divmod(completed_step - 1, steps_per_epoch)
    return {
        "epoch":           epoch_index + 1,
        "batch_in_epoch":  batch_index + 1,
        "steps_per_epoch": steps_per_epoch,
        "corpus_pass":     completed_step / steps_per_epoch,
    }


def train(
    model:  VFEModel,
    loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],   # yields (tokens, targets) batches
    cfg:    VFE3Config,

    *,
    n_steps:   int             = 100,
    grad_clip: Optional[float] = 1.0,

    log_interval:    Optional[int]            = None,
    eval_interval:   Optional[int]            = None,
    val_loader:      Optional[Iterable]       = None,
    tokens_per_char: Optional[float]           = None,   # None -> BPC unavailable; BPT remains defined
    device:          Optional[torch.device]   = None,
    logger:          Optional[logging.Logger] = None,
    artifacts:       Optional["RunArtifacts"] = None,
    resume_from:     'Optional[str | Path]'   = None,   # checkpoints/step_<N>.pt to resume from (None -> cfg.resume_from -> from scratch)
    terminal_callback: Optional[Callable[[TrainingTerminalState, List[float]], None]] = None,   # invoked ONCE after the final step (PB-02)
    _final_validation_out: Optional[Dict[str, Optional[float]]] = None,   # private launcher handoff

    generate_samples:  bool                                     = True,   # False -> pure silent path (no sample text)
    sample_decode:     Optional[Callable[[Sequence[int]], str]] = None,   # token-ids -> text; None -> auto
    sample_dataset:    Optional[str]                            = None,   # keys the sample tokenizer; None -> vocab-size guess
    sample_new_tokens: int                                      = 40,     # greedy continuation length
    sample_prompt_len: int                                      = 6,     # seq-0 prompt length to continue
) -> List[float]:
    r"""Train ``n_steps`` M-step iterations (cycling the loader); return the loss history.

    Builds the per-group AdamW optimizer and the warmup/cosine ``LambdaLR``, then takes
    ``n_steps`` gradient steps, re-iterating the loader when it is exhausted. The loss
    history is the per-step cross-entropy; the cutover criterion is that it decreases.

    With ``log_interval`` falsy (``None`` or ``0``) and ``eval_interval`` falsy the loop
    is bitwise-identical to the silent path: the two truthiness-guarded blocks
    short-circuit, drawing no RNG, running no extra forward, and printing nothing. When
    ``log_interval`` is positive a per-step line is emitted every
    ``log_interval`` steps (CE and diagnostics recomputed under ``no_grad`` only at those
    steps, off the training graph), AND -- when ``tqdm`` is installed -- the step loop runs
    under a ``tqdm`` progress bar whose built-in rate readout shows live ``it/s`` every step
    (the formatted lines render above it via ``logging_redirect_tqdm``); when ``eval_interval``
    is positive and ``val_loader`` is given a validation block is emitted every
    ``eval_interval`` steps.
    """
    # Never expose a stale result after a partial/resumed run or a failed call.
    if _final_validation_out is not None:
        _final_validation_out.clear()
    terminal_validation_signature = None
    terminal_validation_metrics = None
    optimizer = build_optimizer(model, cfg)
    # Warmup/cosine multiplier, floored per group so each group's ABSOLUTE LR never decays below
    # max(cfg.min_lr, cfg.min_lr_frac * base) -- see _floor_lr_lambdas. With cfg.min_lr=cfg.min_lr_frac=0
    # this is exactly the pure half-cosine-to-zero (the theoretically pure path). base_lrs are the
    # CONFIGURED per-group LRs, captured before any scheduler multiplier or resume-load mutates group['lr'].
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    if artifacts is not None:
        artifacts.scheduler_base_lrs = [float(value) for value in base_lrs]
        artifacts.scheduler_group_roles = [
            str(group.get("lr_report_role") or group.get("lr_aux_role")
                or group.get("role") or f"group_{index}")
            for index, group in enumerate(optimizer.param_groups)
        ]
    # Opt-in RESUME (PL8): an explicit resume_from arg, else cfg.resume_from, else from scratch. When set,
    # restore model weights + AdamW momentum + RNG from the checkpoint and rebuild the cosine LambdaLR at
    # the persisted successful-update count so a rejected update does not consume schedule progress.
    # start_step remains the outer-loop/data cursor; the loop runs range(start_step, n_steps). Legacy
    # checkpoints without the new optimizer-group clock fall back to start_step.
    resume_path = resume_from if resume_from is not None else cfg.resume_from
    start_step = 0
    if device is None:
        device = next(model.parameters()).device
    loader_sampler = getattr(loader, "sampler", None)
    shuffled_loader = isinstance(loader_sampler, torch.utils.data.RandomSampler)
    loader_generator = getattr(loader, "generator", None)
    try:
        steps_per_epoch = int(len(loader))  # type: ignore[arg-type]
    except (TypeError, AttributeError):
        steps_per_epoch = 0
    if steps_per_epoch < 0:
        raise ValueError(f"loader length must be nonnegative, got {steps_per_epoch}")
    periodic_checkpoint_requested = bool(
        artifacts is not None
        and cfg.checkpoint_interval > 0
        and n_steps >= cfg.checkpoint_interval
    )
    cursor_requested = bool(
        resume_path is not None
        or terminal_callback is not None
        or periodic_checkpoint_requested
    )
    cursor_supported = isinstance(
        getattr(getattr(loader, "dataset", None), "tokens", None), torch.Tensor)
    if cursor_requested and not cursor_supported:
        raise RuntimeError(
            "exact data persistence requires a supported DataLoader over a token-window dataset")
    loader_data_identity = (
        _loader_data_identity(loader, cfg.vocab_size)
        if cursor_requested and cursor_supported else None
    )
    # Best-model SELECTION binds the validation contract that chose the published weights, and that
    # is a different question from whether a resumable data CURSOR is required. Periodic eval calls
    # maybe_save_best on runs with no resume, no terminal callback, and checkpoint_interval above
    # n_steps -- all three cursor triggers off. Binding under `cursor_requested` alone left exactly
    # that cell publishing best_model.pt with selection_data_identity=None and a finite
    # best_val_ppl, which finalize_run then refuses to reload ("best-model finalization requires a
    # bound validation-data identity") only AFTER the whole training budget was spent. So bind
    # whenever selection can run, mirroring `cursor_supported`: a val_loader that cannot state an
    # identity (a plain iterable of batches, from a library caller) still selects unbound rather
    # than raising here, which keeps that documented diagnostic path working.
    selection_requested = bool(
        artifacts is not None and val_loader is not None and bool(eval_interval))
    selection_supported = isinstance(
        getattr(getattr(val_loader, "dataset", None), "tokens", None), torch.Tensor)
    selection_data_identity = None
    if (artifacts is not None and val_loader is not None
            and (cursor_requested or (selection_requested and selection_supported))):
        selection_data_identity = _loader_data_identity(val_loader, cfg.vocab_size)
        artifacts.bind_selection_data_identity(selection_data_identity)
    resume_data_state: DataStateBuffer = {}
    if (cursor_requested and shuffled_loader
            and not isinstance(loader_generator, torch.Generator)):
        raise RuntimeError(
            "exact shuffled resume requires loader.generator to expose a torch.Generator")
    if (cursor_requested and shuffled_loader
            and getattr(loader_sampler, "generator", None) is not loader_generator):
        raise RuntimeError(
            "exact shuffled resume requires sampler.generator to be loader.generator")
    # fp16 training needs loss scaling (gradients underflow through the unrolled E-step); bf16/fp32
    # do not. enabled=False is a no-op, so non-fp16 amp_dtype keeps this loop byte-identical.
    # Created BEFORE the resume block so load_checkpoint can restore its scale/growth state
    # (audit 2026-06-09 IE3).
    scaler = torch.amp.GradScaler(device=device.type, enabled=(cfg.amp_dtype == "fp16"))
    if resume_path is not None:
        from vfe3.run_artifacts import load_checkpoint           # local import avoids any import cycle
        start_step = load_checkpoint(resume_path, model, optimizer, map_location=device,
                                     max_step=n_steps,
                                     scaler=scaler, cfg=cfg, artifacts=artifacts,
                                     data_state=resume_data_state,
                                     expected_data_identity=loader_data_identity,
                                     expected_selection_data_identity=selection_data_identity,
                                     expected_steps_per_epoch=steps_per_epoch)
        if not resume_data_state:
            raise RuntimeError(
                "exact data resume requires checkpoint data_state; this checkpoint predates "
                "iterator persistence")
        # LambdaLR with last_epoch != -1 requires 'initial_lr' on every group; set it from the configured
        # base (not the post-load group['lr'], which the restored optimizer state overwrote with base*cos).
        for group, base in zip(optimizer.param_groups, base_lrs):
            group["initial_lr"] = base
        successful_updates = _successful_update_count(
            optimizer, legacy_default=start_step)
        _set_successful_update_count(optimizer, successful_updates)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, _floor_lr_lambdas(base_lrs, cfg), last_epoch=successful_updates - 1)
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _floor_lr_lambdas(base_lrs, cfg))
        _set_successful_update_count(optimizer, 0)
    # Unigram log-prior decode (cfg.decode_unigram_prior, default OFF): fill the PriorBank's
    # unigram table from the TRAINING stream once, before the loop -- a fixed data statistic
    # (add-one-smoothed log frequencies), not a learned parameter. The counts come from the
    # loader's TokenWindows dataset (its flat token stream); a loader without one warns and
    # leaves the table unset (the decode then warns once and is a value no-op).
    if cfg.decode_unigram_prior:
        _tok = getattr(getattr(loader, "dataset", None), "tokens", None)
        if _tok is not None:
            from vfe3.run_artifacts import _loader_token_content_summary
            _digest, _n_tokens, _counts = _loader_token_content_summary(
                loader,
                vocab_size=cfg.vocab_size,
            )
            if _counts is None:
                raise RuntimeError("training token counts are unavailable for unigram decode")
            _counts = _counts.to(torch.float32)
            model.prior_bank.set_unigram_log_prior(_counts.to(device))
        else:
            import warnings
            warnings.warn(
                "decode_unigram_prior=True but the training loader exposes no flat token stream "
                "(loader.dataset.tokens); the unigram table stays unset (decode warns, value "
                "no-op). Call model.prior_bank.set_unigram_log_prior(counts) manually.",
                UserWarning, stacklevel=2)
    losses: List[float] = []
    model.train()
    logger = logger or logging.getLogger(__name__)
    # Live per-step it/s: iterate the step loop through a tqdm bar whose built-in rate readout
    # refreshes every step. Gated on log_interval so the documented silent path (log_interval
    # falsy) stays bitwise-identical -- no bar, no redirect, nothing printed. The generator holds
    # logging_redirect_tqdm open across the whole loop (it suspends at `yield` INSIDE the `with`),
    # so the periodic logger.info lines below render above the bar instead of interleaving with it
    # on stderr; it closes the bar on normal exit or exception.
    show_bar = bool(log_interval) and _tqdm is not None

    # Outcome half of the freeze guard (see parameter_motion_report): clone the parameters BEFORE the
    # first step so the end-of-run comparison measures motion over the steps this call actually took.
    # Skipped when the loop is empty, where "nothing moved" is the correct outcome, not a finding.
    motion_snapshot = (snapshot_parameters(model)
                       if n_steps > start_step
                       else None)
    # Which parameters never even receive a gradient is read off the FIRST REAL STEP below rather
    # than from a synthetic parameter_report probe here: train_step zeroes grads on entry and not on
    # exit, so the real .grad is still live when it returns. That costs no extra forward/backward,
    # and it reports the graph the run actually trains through instead of a probe's approximation of
    # it. The end-of-run warning subtracts this set, so it fires only on a parameter that DID get a
    # gradient and still never moved -- the case no existing check covers.
    motion_known_dead: Tuple[str, ...] = ()
    motion_dead_captured = False
    skipped_steps = 0                 # cumulative REJECTED optimizer updates (audit 2026-08-06 F5)
    consecutive_skips = 0
    reset_nonfinite_tangent_elements()   # per-run accounting for the SPD-retraction guard
    reset_mu_trust_fallback_elements()   # ... the mu-trust-region equivariance fallback
    reset_decode_logdet_fallback_elements()   # ... and the decode log-det non-PD fallback
    reset_mm_cholesky_fallback_elements()   # ... and failed full-Gaussian MM Cholesky rows
    reset_source_recovery_elements()        # ... and projected float64 source-factor recoveries
    reset_frame_pair_fp64_escalations()     # ... and float64 frame-pair escalations

    def _step_indices() -> Iterable[int]:
        if not show_bar:
            yield from range(start_step, n_steps)               # range start_step..n_steps (== 0..n_steps from scratch)
            return
        bar = _tqdm(range(start_step, n_steps), desc="Training", total=n_steps,
                    initial=start_step, ascii=True)             # ascii=True: the default block glyph
        #                          U+2588 is not cp1252-encodable on a Windows console (raises
        #                          UnicodeEncodeError mid-run); " #" renders anywhere. initial=start_step
        #                          keeps the bar's absolute step readout correct on a resumed run.
        with _redirect_logging():
            try:
                yield from bar
            finally:
                bar.close()

    epoch = 0
    batches_consumed = 0
    if resume_data_state:
        required_data_state = {
            "epoch_start_generator_state", "batches_consumed", "epoch", "data_identity",
        }
        missing_data_state = required_data_state - resume_data_state.keys()
        if missing_data_state:
            raise RuntimeError(
                f"checkpoint data_state is missing required field(s) {sorted(missing_data_state)}")
        saved_generator_state = resume_data_state["epoch_start_generator_state"]
        epoch = resume_data_state["epoch"]
        saved_batches_consumed = resume_data_state["batches_consumed"]
        if shuffled_loader:
            if not isinstance(loader_generator, torch.Generator):
                raise RuntimeError(
                    "exact shuffled resume requires loader.generator to expose a torch.Generator")
            if not isinstance(saved_generator_state, torch.Tensor):
                raise RuntimeError(
                    "shuffled checkpoint data_state requires an epoch generator tensor")
            epoch_start_generator_state = saved_generator_state.cpu().clone()
            loader_generator.set_state(epoch_start_generator_state)
        else:
            if saved_generator_state is not None:
                raise RuntimeError(
                    "sequential checkpoint data_state requires a null epoch generator state")
            epoch_start_generator_state = None
    else:
        saved_batches_consumed = 0
        epoch_start_generator_state = (
            loader_generator.get_state().clone()
            if shuffled_loader and isinstance(loader_generator, torch.Generator) else None)
    # load_checkpoint restored the global RNG to the instant of publication. Reconstructing a
    # DataLoader iterator consumes a CPU base-seed draw even for a deterministic sequential loader
    # with num_workers=0; that draw already happened at the saved epoch start. Preserve the restored
    # stream around resume-only reconstruction/skip so stochastic model behavior continues exactly.
    replay_cpu_rng = torch.get_rng_state().clone() if resume_data_state else None
    try:
        it = iter(loader)
        for _ in range(saved_batches_consumed):
            try:
                next(it)
            except StopIteration as exc:
                raise RuntimeError(
                    "checkpoint data_state cannot be replayed by the current loader: "
                    "batches_consumed exceeds the saved epoch") from exc
            batches_consumed += 1
    finally:
        if replay_cpu_rng is not None:
            torch.set_rng_state(replay_cpu_rng)
    # A timer is useful only if its absolute execution interval reaches a reporting step.
    timing_enabled = (
        bool(log_interval) and n_steps // log_interval > start_step // log_interval
    ) or (
        artifacts is not None and val_loader is not None and bool(eval_interval)
        and n_steps // eval_interval > start_step // eval_interval
    )
    timer = TrainingTimer(device) if timing_enabled else None
    last_val: Dict[str, Optional[float]] = {}        # most recent validation, carried into each CSV row
    val_diag_keys = _VAL_DIAG_KEYS + (
        _VAL_RELATIONAL_DIAG_KEYS if is_conditional_filtering(model) else ())
    last_val_diag: Dict[str, float] = {k: float("nan") for k in val_diag_keys}
    for step in _step_indices():
        try:
            tokens, targets = next(it)
        except StopIteration:
            epoch += 1
            batches_consumed = 0
            epoch_start_generator_state = (
                loader_generator.get_state().clone()
                if shuffled_loader and isinstance(loader_generator, torch.Generator) else None)
            it = iter(loader)
            tokens, targets = next(it)
        batches_consumed += 1
        tokens = tokens.to(device, non_blocking=True)
        targets = _transfer_targets(targets, model, device)
        do_log  = bool(log_interval) and (step + 1) % log_interval == 0
        do_eval = bool(eval_interval) and val_loader is not None and (step + 1) % eval_interval == 0
        do_csv  = artifacts is not None and (do_log or do_eval)
        # Capture pre-clip gradient health only on a step that will log or persist. A silent step
        # performs no exact-count host materialization, extra unscale_, or grad-norm pass; its
        # device-side partition/zero-count predicates ride the existing scalar barrier.
        step_metrics: Optional[Dict[str, float]] = {} if (do_log or do_csv) else None
        if hasattr(optimizer, "_collect_gauge_diag"):        # D1/EXP-8: sparse log/eval diagnostics
            optimizer._collect_gauge_diag = bool(do_log or do_csv or do_eval)
            if optimizer._collect_gauge_diag:
                # A nonfinite/otherwise skipped optimizer step must not leave the prior log step's
                # health values visible as if they were collected for this attempt.
                optimizer._gauge_diag = {}
        if do_log or do_csv:                                 # off the hot path (audit 2026-07-05 m8)
            _warn_phi_transport_clamp(model)
            # m7: diagnostics on the PRE-step weights so the logged F-decomposition shares the pre-step
            # provenance of train_loss/train_ce (train_ce is captured pre-step inside train_step). Under
            # no_grad, so no weight change,
            # leaving the train_step RNG stream and weights byte-identical.
            d = model.diagnostics(tokens)
        # generalization_gap's train side under the validation protocol when that protocol is the
        # sliding window (eval_stride < max_seq_len); pre-step, like train_ce. None keeps train_ce.
        gap_train_ce: Optional[float] = None
        if (do_eval and do_csv and cfg.eval_stride is not None
                and cfg.eval_stride < cfg.max_seq_len):
            gap_train_ce = _eval_protocol_train_ce(model, tokens, targets, cfg.eval_stride)
        step_status: Dict[str, bool] = {}
        if timer is not None:
            timer.start_step()
        losses.append(train_step(model, optimizer, scheduler, tokens, targets,
                                  grad_clip=grad_clip, grad_accum_steps=cfg.grad_accum_steps,
                                  scaler=scaler, metrics_out=step_metrics, status_out=step_status))
        if timer is not None:
            timer.finish_step(n_tokens=tokens.numel())
        # Gradient reachability, read off the first step that actually updated: grads are still live
        # here (train_step zeroes on entry). Gated on did_step because the nonfinite-grad path drops
        # them, which would read as "every parameter is dead" and mute the whole end-of-run check.
        # Cumulative accepted/rejected accounting (audit 2026-08-06 F5). step_skipped is written
        # per-row but only on LOGGING steps, and summary.json reports cfg.max_steps -- the CONFIGURED
        # value, never the realized one. A real run
        # (832.33_wikitext-103_K30_block_glk_s6) accepted 453 updates of 60000 over 11.4 hours and
        # published a normal-looking summary; the live arms lose 0.2%-4.5% ARM-DEPENDENTLY, so arms
        # of one sweep are compared at different effective budgets and different points on the LR
        # schedule (the scheduler clocks on accepted updates). Free: status_out["did_step"] is
        # already returned every step.
        if step_status["did_step"]:
            consecutive_skips = 0
        else:
            skipped_steps += 1
            consecutive_skips += 1
            if skipped_steps == 1:
                logger.warning(
                    "step %d: optimizer update REJECTED (non-finite gradient or AMP scale "
                    "reduction). The LR scheduler clocks on accepted updates, so skips shorten the "
                    "realized schedule; accepted_updates is reported in summary.json.", step + 1)
            elif consecutive_skips > 10 and consecutive_skips % 10 == 0:
                logger.warning(
                    "step %d: %d CONSECUTIVE rejected updates (%d of %d total so far)",
                    step + 1, consecutive_skips, skipped_steps, step + 1)
        if motion_snapshot is not None and not motion_dead_captured and step_status["did_step"]:
            motion_known_dead = tuple(name for name, p in model.named_parameters()
                                      if p.requires_grad and p.grad is None)
            motion_dead_captured = True
        if do_log or do_csv:                                 # diagnostics (off graph), ONCE
            # train_ce is the PRE-step CE captured inside train_step (matches train_loss); the old
            # post-step re-forward made train_ce one optimizer step ahead of train_loss (audit r2 id5).
            ce = step_metrics.get("train_ce", float("nan")) if step_metrics is not None else float("nan")
            # d (the F-decomposition) is computed PRE-step above so it shares train_ce's provenance (m7).
            assert timer is not None
            train_timing = timer.sample_train_window()
            if device.type == "cuda":
                peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1e6
                torch.cuda.reset_peak_memory_stats(device)
            else:
                peak_mem_mb = float("nan")

        if do_log:
            logger.info(
                "Step %d/%d | Loss: %.4f | CE: %.4f | H(b): %.3f | train it/s: %.2f | \n\n         Train PPL: %.1f \n",
                step + 1, n_steps, losses[-1], ce, d["attn_entropy"],
                train_timing.train_steps_per_s, _perplexity_from_ce(ce),
            )
            bits_per_token = ce / math.log(2.0)
            bpc_text = (
                f" | BPC {bits_per_token * tokens_per_char:.4f}"
                if tokens_per_char is not None
                else " | BPC unavailable"
            )
            logger.info(
                "    Inner alignment energy: self %.4f | belief %.4f | entropy %.4f | total %.4f | eff_rank %.2f | BPT %.4f%s",
                d["self_coupling"], d["belief_coupling"], d["attention_entropy"],
                d["total"], d["effective_rank"], bits_per_token, bpc_text,
            )

        if do_eval:
            # RNG ISOLATION (audit 2026-08-06 F30). Everything in this block that builds a DataLoader
            # iterator -- ``evaluate`` below and ``_val_diagnostics`` further down -- draws a
            # base_seed from the GLOBAL CPU RNG, and the TRAIN loader's RandomSampler draws every
            # epoch permutation from that same stream (datasets.py:686-687). So without this, two
            # runs with the same seed but different eval cadences see different data orders from the
            # first epoch boundary onward, and eval_interval silently becomes a training
            # hyperparameter. Sampling and figure generation are inside the guard for the same
            # reason. The resume path already applies exactly this snapshot/restore (:1612).
            #
            # CPU only, matching that precedent: the permutation is a CPU draw, and reading the CUDA
            # state costs a host sync per eval. INERT on every run currently on disk -- at
            # max_steps=10000 on wikitext-103 the run covers 0.26 epochs, so no boundary is reached
            # and no permutation is ever redrawn.
            eval_cpu_rng = torch.get_rng_state().clone()
            if (_final_validation_out is not None and terminal_callback is None
                    and step + 1 == n_steps):
                terminal_validation_signature = _terminal_validation_signature(model, val_loader)
            m = evaluate(model, val_loader, tokens_per_char=tokens_per_char, device=device)
            if terminal_validation_signature is not None:
                terminal_validation_metrics = dict(m)
            logger.info(" \n Validation @ step %d:", step + 1)
            logger.info(                                         # val has no separate loss; CE is the loss
                "\n       CE: %.4f \n      Val PPL: %.1f \n       BPT: %.4f%s \n\n",
                m["ce"], m["ppl"], m["bits_per_token"],
                (f"\n       BPC: {float(m['bpc']):.4f}" if m["bpc"] is not None
                 else "\n       BPC: unavailable"),
            )
            # Sample text directly below the BPC value. ``generate_samples=False`` forces the pure
            # silent path (no generation, no Sample line). Otherwise the decoder is an explicit
            # ``sample_decode`` if given, else an AUTO-DEFAULT picked from cfg.vocab_size (gpt2 /
            # cl100k) -- so a real click-to-run prints samples with no wiring, while tiny
            # synthetic/test vocabs get no decoder. When a decoder exists, greedily continue seq 0 of
            # the live batch by sample_new_tokens and decode prompt + continuation. Best-effort: a
            # generation/decode error is logged, never fatal (model.generate is @torch.no_grad).
            decode = None if not generate_samples else (
                sample_decode if sample_decode is not None
                else _default_sample_decoder(cfg, dataset=sample_dataset,
                     cache_dir=getattr(getattr(loader, "dataset", None), "cache_dir", None)))
            if decode is not None:
                try:
                    prompt = tokens[:1, :sample_prompt_len]                       # (1, P) seq-0 prompt
                    p_txt, c_txt = _periodic_generation(
                        model, prompt, sample_new_tokens, decode,
                    )
                    logger.info("       Sample: %r  ->  %r\n", p_txt, c_txt)
                except PeriodicGenerationRestorationError:
                    raise
                except Exception as exc:                                          # never let sampling kill training
                    logger.warning("       (sample generation failed: %s)", exc)
            last_val = {
                "ce":             m["ce"],
                "ppl":            m["ppl"],
                "bits_per_token": m["bits_per_token"],
                "bpc":            m["bpc"],
                "expected_targets": m["expected_targets"],
                "scored_targets": m["scored_targets"],
                "excluded_targets": m["excluded_targets"],
            }
            if artifacts is not None:
                # Held-out per-eval probes (validation F decomposition, attention-map structure,
                # E-step convergence certificate, per-position loss). Best-effort: a replay error
                # RESETS the probes to NaN (blank CSV cells) so a previous eval's values are never
                # carried forward as if fresh (audit 2026-07-01 F11), and a viz/replay fault never
                # kills training.
                validation_diagnostics: Optional[ValidationDiagnostics] = None
                try:
                    validation_diagnostics = _val_diagnostics(
                        model, val_loader, device,
                        batches=getattr(cfg, "val_diagnostic_batches", 1))
                    last_val_diag.update(validation_diagnostics.metrics)
                except Exception as exc:
                    terminal_validation_signature = None
                    logger.warning("       (validation diagnostics failed: %s); continuing", exc)
                    last_val_diag.update({k: float("nan") for k in val_diag_keys})
                try:
                    artifacts.maybe_save_best(step + 1, model, m["ppl"])
                    # Per-layer/per-head attention heatmap grid for this eval (off the graph, seq 0
                    # of the held-out batch). Kept at EVAL cadence (one grid per eval, not per log),
                    # and guarded so replay errors remain nonfatal (audit 2026-07-01 F11).
                    if validation_diagnostics is not None:
                        try:
                            _save_eval_attention_maps(
                                validation_diagnostics.token_ids,
                                model,
                                artifacts,
                                logger,
                                step=step + 1,
                            )
                        except Exception as exc:
                            terminal_validation_signature = None
                            logger.warning("       (attention-map replay failed: %s); continuing", exc)
                finally:
                    # This loop-local otherwise retains the dense held-out logits and frozen clone
                    # until the next evaluation overwrites it. Release after the final consumer on
                    # enabled, disabled, and failed-map branches before training resumes.
                    validation_diagnostics = None
            torch.set_rng_state(eval_cpu_rng)            # F30: hand the training stream back untouched

        # Periodic resumable checkpoint (opt-in; needs the artifacts dir and the optimizer state).
        if (artifacts is not None and cfg.checkpoint_interval
                and (step + 1) % cfg.checkpoint_interval == 0):
            checkpoint_data_state: Optional[DataState] = ({
                "epoch_start_generator_state": epoch_start_generator_state,
                "batches_consumed":            batches_consumed,
                "epoch":                       epoch,
                "data_identity":               loader_data_identity,
            } if loader_data_identity is not None else None)
            # Best-effort, like every other periodic side effect above (audit 2026-07-27). A
            # checkpoint is a convenience for RESUMING, not a correctness requirement of the step
            # loop, so a failed write must cost one checkpoint and not the run: a transient
            # torch.save stream failure ("ios_base::badbit set") at step 120000 of 180000
            # previously aborted 10 hours of training with a perfectly healthy model in memory.
            # Loud warning, then continue -- the next interval attempts a fresh save.
            try:
                artifacts.save_checkpoint(step + 1, model, optimizer, cfg, scaler=scaler,
                                          data_state=checkpoint_data_state)
            except Exception as exc:
                logger.warning("       (checkpoint save at step %d failed: %s); training "
                               "continues, next checkpoint retries", step + 1, exc)

        # Persistence is opt-in: with no artifacts object do_csv is False, so the silent/in-memory
        # path is unchanged. A metrics.csv row is written every LOG_INTERVAL (and every eval) -- the
        # dense per-step diagnostics off the graph. The EVAL-CADENCE columns (val_ce/ppl/bpc,
        # generalization_gap, and the held-out val_*/estep_*/pos_loss_* probes) carry a value ONLY on a
        # step where the eval above just ran (do_eval); on the denser log-interval rows in between they
        # are NaN, which log_metrics renders as a BLANK cell -- so each validation appears exactly once
        # per eval_interval, not carried forward to every log line. The
        # in-memory history keeps the NaN (figures already drop non-finite rows). The four F-stack
        # diagnostics are per-sequence SUMS over seq 0, normalized to PER TOKEN so they are
        # commensurate with val_ce, a token-weighted mean (nats/token; see audit-2026-06-05 Finding 2).
        if do_csv:
            n_tok = max(int(tokens.shape[1]), 1)
            raw_lrs = scheduler.get_last_lr()
            lrs = _learning_rates_by_role(optimizer.param_groups, raw_lrs)
            aux_lrs = _learning_rates_by_aux_role(optimizer.param_groups, raw_lrs)
            row = {
                "step":              step + 1,
                "train_loss":        losses[-1],
                "train_ce":          ce,                      # true CE (nats), off the graph
                "train_ppl":         _perplexity_from_ce(ce),  # exact exp(CE), infinity on overflow
                "lr_block_mlp":       aux_lrs["block_mlp"],
                "lr_layernorm":      aux_lrs["layernorm"],
                "lr_mu":             lrs["mu"],
                "lr_sigma":          lrs["sigma"],
                "lr_phi":            lrs["phi"],
                "lr_head_evidence":  aux_lrs["head_evidence"],
                "lr_head_mixer":     aux_lrs["head_mixer"],
                "lr_relational_score":   aux_lrs["relational_score"],
                "lr_relational_content": aux_lrs["relational_content"],
                "val_ce":            last_val["ce"]  if do_eval else float("nan"),  # eval-cadence: fresh on
                "val_ppl":           last_val["ppl"] if do_eval else float("nan"),  # an eval step (last_val just
                "val_bits_per_token": (last_val["bits_per_token"] if do_eval
                                       else float("nan")),
                "val_bpc":           (last_val["bpc"] if do_eval and last_val["bpc"] is not None
                                      else float("nan")),
                "val_expected_targets": (last_val["expected_targets"] if do_eval else float("nan")),
                "val_scored_targets": (last_val["scored_targets"] if do_eval else float("nan")),
                "val_excluded_targets": (last_val["excluded_targets"] if do_eval else float("nan")),
                "attn_entropy":       d["attn_entropy"],
                "self_coupling":      d["self_coupling"]     / n_tok,   # alpha-regularized F self-term sum_i[alpha_i D + R(alpha_i)]
                "self_divergence":    d["self_divergence"]   / n_tok,   # raw sum_i D(q_i||p_i) drift; == self_coupling only at lambda_alpha_mode='constant'
                "belief_coupling":    d["belief_coupling"]   / n_tok,
                "attention_entropy":  d["attention_entropy"] / n_tok,
                "inner_alignment_energy_total": d["total"]  / n_tok,
                # Weights emitted alongside the RAW per-term fields (audit 2026-07-25 F7): those
                # fields are deliberately unweighted while "total" routes through the weighted
                # evaluator, so a naive column sum missed the total by up to 98% relative at
                # lambda_beta=0.5 with no column to reconcile it. Now each row is self-describing.
                "lambda_beta":        float(cfg.lambda_beta),
                "effective_rank":     d["effective_rank"],
                "holonomy_deviation": d["holonomy_deviation"],
                "gauge_trace_spread": d["gauge_trace_spread"],
            }
            row.update({key: value for key, value in d.items() if key.startswith(("qkv_", "relational_"))})
            if steps_per_epoch:
                row.update(_training_cursor_fields(step + 1, steps_per_epoch))
            else:
                row.update({
                    "epoch":           epoch + 1,
                    "batch_in_epoch":  batches_consumed,
                    "steps_per_epoch": float("nan"),
                    "corpus_pass":     float("nan"),
                })
            # Peak memory (Tier-1): CUDA peak MB at the clean train-window boundary.
            row["peak_mem_mb"] = peak_mem_mb
            # Generalization gap (Tier-1): val-set CE minus the per-step train CE (positive = overfit,
            # the standard convention). The train side is this step's batch while val is the
            # token-weighted val-set mean, so read it as a TREND, not an absolute. Under a sliding
            # validation window (eval_stride < max_seq_len) the train side is scored on the SAME
            # positions (the last eval_stride of each window, eval mode; _eval_protocol_train_ce),
            # since the disjoint-window train_ce averages shorter contexts and biases the gap negative.
            # Eval-cadence like val_ce above: written only on an eval row, blank (NaN) in between.
            # (The complexity-vs-fit comparison is inner_alignment_energy_total vs
            # val_ce, both already columns; a separate "elbo_ce_gap" would subtract a CE from a
            # complexity-only F -- d["total"] carries no -E_q[log p] data term -- so it is NOT emitted.)
            gap_train_side = ce if gap_train_ce is None else gap_train_ce
            row["generalization_gap"] = (last_val["ce"] - gap_train_side) if do_eval else float("nan")
            if cfg.decode_mode != "linear":
                # Realized readout temperature tau_eff = decode_tau * exp(-clamp(decode_log_scale,
                # -3, 3)). While decode_temperature_learned=True, decode_tau is only its init.
                row["decode_tau_eff"] = float(model.prior_bank._tau_eff().detach())
            # Extended per-eval diagnostics (Tier-1/2): already-reduced gauge / geometry / numerical-
            # health scalars from diagnostics() -- NOT per-token sums, so logged RAW (no /n_tok).
            # Conditional mixer/connection keys appear only with their toggle; the
            # config is fixed per run so the CSV stays rectangular.
            for _dk in ("holonomy_ci_lo", "holonomy_ci_hi", "holonomy_wilson",
                        "gauge_invariant_mean", "gauge_invariant_spread",
                        "phi_norm_mean", "phi_norm_std",
                        "belief_cond_median", "belief_cond_p95", "belief_cond_max", "belief_pd_margin",
                        "eff_rank_p5", "eff_rank_median", "eff_rank_p95",
                        "fisher_trace_mean", "fisher_trace_median",
                        "guard_sigma_floor_frac", "guard_sigma_ceil_frac",
                        "guard_energy_klmax_frac", "guard_selfdiv_klmax_frac",
                        "nonfinite_frac", "renyi_band_frac",
                        "attn_entropy_min", "attn_entropy_collapsed_heads",
                        "cocycle_residual", "vertex_cond_max", "sandwich_absmax", "transport_asymmetry",
                        "energy_abs_asymmetry", "energy_rel_asymmetry",
                        "gauge_head_aniso_mean", "gauge_head_logdet_spread",
                        "phi_matrix_norm_median", "phi_matrix_norm_p95", "phi_matrix_norm_p99",
                        "phi_matrix_norm_max", "phi_exp_clamp_frac", "phi_exp_scale_min",
                        "vertex_cond_median", "vertex_cond_p95", "vertex_cond_p99",
                        "pos_phi_matrix_norm_p95", "pos_phi_matrix_norm_p99", "pos_phi_matrix_norm_max",
                        "pos_phi_exp_clamp_frac", "pos_phi_exp_scale_min",
                        "head_mixer_drift",
                        "head_evidence_weights", "head_evidence_entropy",
                        "head_evidence_max_abs_drift"):
                if _dk in d:
                    row[_dk] = d[_dk]
            # Gradient health (Tier-1/2): global + per-group (mu/sigma/phi) grad AND weight norms (so
            # the update-to-weight ratio is derivable with the logged LR), loss finiteness, the fp16
            # loss-scale, and the grad-accum token spread -- captured by train_step into step_metrics.
            if step_metrics:
                for _gk in ("grad_norm", "grad_norm_mu", "grad_norm_sigma", "grad_norm_phi",
                            "grad_norm_relational_score", "grad_norm_relational_content",
                            "weight_norm_mu", "weight_norm_sigma", "weight_norm_phi",
                            "estep_grad_norm_mu", "estep_grad_norm_sigma", "estep_grad_norm_phi",
                            "estep_grad_norm_mu_microbatch_mean",
                            "estep_grad_norm_sigma_microbatch_mean",
                            "estep_grad_norm_phi_microbatch_mean",
                            "loss_finite", "grad_finite", "step_skipped",
                            "grad_scale", "grad_accum_tok_spread",
                            "expected_targets", "scored_targets", "excluded_targets"):
                    if _gk in step_metrics:
                        row[_gk] = step_metrics[_gk]
            if cfg.phi_mstep_max_matrix_norm is not None:
                for _pk in (
                    "phi_chart_projected_rows",
                    "phi_chart_total_rows",
                    "phi_chart_projected_fraction",
                    "phi_chart_preproject_max",
                    "phi_chart_projection_scale_min",
                    "phi_chart_projection_ms",
                    "phi_chart_projection_stats_collected",
                ):
                    row[_pk] = (
                        step_metrics.get(_pk, float("nan"))
                        if step_metrics is not None
                        else float("nan")
                    )
            # Pullback-group M-step diagnostics use a fixed key set from the first logged row. The
            # optimizer computes and host-reduces them only on this gated log/eval attempt; missing
            # active rows emit NaN placeholders so metrics.csv remains rectangular. Plain AdamW has
            # no _gauge_diag attribute.
            if hasattr(optimizer, "_gauge_diag"):
                _gd = optimizer._gauge_diag or {}
                if any(group.get("pullback_group", False) for group in optimizer.param_groups):
                    for _pk in (
                        "phi_ridge_direction_cosine_mean",
                        "phi_pullback_damped_gen_cond_median",
                        "phi_pullback_damped_gen_cond_max",
                        "phi_group_trust_scale_mean",
                        "phi_group_trust_scale_min",
                        "phi_group_active_rows",
                        "phi_group_chart_norm_max",
                        "phi_group_rejected_rows",
                        "phi_chart_displacement_mean",
                        "phi_chart_displacement_max",
                        "phi_frame_displacement_mean",
                        "phi_frame_displacement_max",
                        "phi_covector_norm_mean",
                        "phi_conditioned_direction_norm_mean",
                        "phi_effective_step_scale_mean",
                        "phi_effective_step_scale_min",
                    ):
                        row[_pk] = _gd.get(_pk, float("nan"))
            # Held-out per-eval probes (Tier-2): the full fixed val-diag column set. Eval-cadence like
            # val_ce above -- the fresh probe values on an eval row, NaN (rendered blank) on the
            # log-interval rows in between, NOT carried forward. The key set is identical in both
            # branches (val_diag_keys), so the CSV stays rectangular.
            row.update(last_val_diag if do_eval else {k: float("nan") for k in val_diag_keys})
            pipeline_timing = timer.sample_pipeline_window()
            row["train_step_ms_mean"]      = train_timing.train_step_ms_mean
            row["train_step_tokens_per_s"] = train_timing.train_step_tokens_per_s
            row["pipeline_tokens_per_s"]   = pipeline_timing.pipeline_tokens_per_s
            row["tokens_per_s"]            = pipeline_timing.pipeline_tokens_per_s
            row["wall_clock_s"]            = pipeline_timing.wall_clock_s
            artifacts.log_metrics(row)
            timer.reset_pipeline_window()
    # Terminal callback seam (PB-02): immediately after the final optimizer step, hand the resumable
    # training state to an optional callback so a
    # default cell (log/eval interval above max_steps, checkpoint_interval=0) can finalize a complete
    # artifact set -- validation eval, best weights, summary, and a resumable terminal checkpoint -- in
    # one opt-in operation. The model-state and CPU/CUDA RNG clones let the finalizer restore the
    # final training iterate after validation. terminal_callback=None keeps the pure path free of
    # those snapshots.
    if terminal_callback is not None:
        raw_model_state = {name: tensor.detach().clone()
                           for name, tensor in model.state_dict().items()}
        rng_state: Dict[str, object] = {
            "cpu":  torch.get_rng_state().clone(),
            "cuda": ([s.clone() for s in torch.cuda.get_rng_state_all()]
                     if torch.cuda.is_available() else None),
        }
        terminal_data_state: Optional[DataState] = ({
            "epoch_start_generator_state": epoch_start_generator_state,
            "batches_consumed":            batches_consumed,
            "epoch":                       epoch,
            "data_identity":               loader_data_identity,
        } if loader_data_identity is not None else None)
        terminal_callback(
            TrainingTerminalState(
                step=n_steps,
                optimizer=optimizer,
                scaler=scaler,
                data_state=terminal_data_state,
                raw_model_state=raw_model_state,
                rng_state=rng_state,
            ),
            losses,
        )
    # Did every trainable parameter actually move on the final optimizer iterate?
    if motion_snapshot is not None:
        motion = parameter_motion_report(
            model, motion_snapshot,
            rel_tol=1e-6,
            known_dead=motion_known_dead,
        )
        motion["steps"] = int(n_steps - start_step)
        if artifacts is not None:
            artifacts.save_json("parameter_motion.json", motion)
        frozen = motion["frozen"]
        if frozen:
            import warnings
            warnings.warn(
                f"parameter motion: {len(frozen)} trainable parameter(s) did not move over "
                f"{motion['steps']} step(s) and are frozen for this run: "
                f"{', '.join(str(n) for n in frozen)}. A parameter can be reported live by the "
                f"init-time grad probe and still never move -- check its group learning rate and "
                f"whether it reaches the loss only through a detached E-step tangent.",
                UserWarning,
                stacklevel=2,
            )
        elif logger is not None:
            logger.info(" parameter motion: all %d trainable parameters moved",
                        sum(1 for m in motion["motion"].values() if m["requires_grad"]))
    # Realized-vs-configured update accounting (audit 2026-08-06 F5), handed to finalize_run through
    # the artifacts object because train() returns only `losses`. summary.json's "n_steps" is
    # cfg.max_steps -- what was ASKED for. These say what actually happened, so an arm that quietly
    # trained on 95% of its budget is visible instead of being compared as if it had the full one.
    # CUMULATIVE over a resume: attempted counts every outer step from 0, and accepted_updates is the
    # optimizer's persisted accepted-update clock (restored by load_checkpoint and advanced by
    # train_step), so a continuation reports the whole run's budget, not only its own segment.
    # The fallback-element counters below are per-process and still cover this call only.
    attempted = int(n_steps)
    accepted = _successful_update_count(optimizer)
    skipped_total = attempted - accepted
    if artifacts is not None:
        neutralized = nonfinite_tangent_elements()
        mu_fallbacks = mu_trust_fallback_elements()
        mm_fallbacks = mm_cholesky_fallback_elements()
        artifacts.final_step = int(n_steps)
        artifacts.realized_updates = {
            "attempted_steps":      attempted,
            "accepted_updates":     int(accepted),
            "skipped_steps":        int(skipped_total),
            "accepted_update_frac": (float(accepted) / attempted
                                     if attempted else float("nan")),
            "nonfinite_tangent_elements": int(neutralized),
            "mu_trust_fallback_elements": int(mu_fallbacks),
            "mm_cholesky_fallback_elements": int(mm_fallbacks),
            "decode_logdet_fallback_elements": int(decode_logdet_fallback_elements()),
            "frame_pair_fp64_escalations": int(frame_pair_fp64_escalations()),
        }
        if skipped_total or neutralized or mu_fallbacks or mm_fallbacks:
            logger.warning(
                " realized budget: %d/%d optimizer updates accepted (%.3f%%); "
                "%d SPD tangent element(s) neutralized as non-finite; "
                "%d belief(s) whitened by the non-equivariant mu-trust fallback; "
                "%d belief row(s) retained after MM Cholesky fallback",
                accepted, attempted,
                100.0 * accepted / max(attempted, 1),
                neutralized, mu_fallbacks, mm_fallbacks)
    if (terminal_validation_signature is not None
            and terminal_validation_metrics is not None
            and all(value is None or math.isfinite(value)
                    for value in terminal_validation_metrics.values())
            and terminal_validation_signature == _terminal_validation_signature(model, val_loader)):
        _final_validation_out.update(terminal_validation_metrics)
    return losses


def coverage_lines(
    loader:     object,                  # DataLoader over a TokenWindows stream
    n_steps:    int,                     # optimizer steps == batches consumed (grad_accum subdivides one batch)
    dataset:    str,

    *,
    full_corpus_tokens: Optional[int] = None,   # uncapped corpus size; emits a "stream is X% of full" line when it exceeds the loaded stream
) -> List[str]:
    r"""Banner lines reporting corpus coverage and epoch count for a training run.

    One optimizer step pulls exactly one ``batch_size`` window-batch from ``loader``
    (``grad_accum_steps`` only ``torch.chunk``-subdivides that single batch -- it draws no extra
    batches), so ``n_steps`` is the number of batches consumed. With ``steps_per_epoch = len(loader)``
    (already ``drop_last``-aware), ``epochs = n_steps / steps_per_epoch``. The default
    ``stride == seq_len`` tiles the corpus exactly once per epoch, so unique corpus coverage saturates
    at one epoch (``min(1, epochs)``); past that the corpus is re-seen ``epochs`` times.
    ``tokens_seen = n_steps * batch_size * seq_len`` is the raw token throughput (counts re-seen tokens).
    """
    ds = loader.dataset
    T = int(ds.tokens.numel())                       # tokens in the (possibly capped) stream
    seq_len = int(ds.seq_len)
    stride = int(ds.stride)
    batch = int(loader.batch_size)
    windows = len(ds)
    spe = len(loader)                                # batches per epoch (honors drop_last)
    epochs = (n_steps / spe) if spe > 0 else float("nan")
    tokens_seen = n_steps * batch * seq_len

    lines = [
        f" data: {dataset}  tokens={T:,}  windows={windows:,}  seq_len={seq_len} stride={stride}",
    ]
    if dataset == "synthetic-period3":               # "% of wiki" is meaningless on the synthetic anchor
        lines.append(f" coverage: epochs={epochs:.2f}  steps/epoch={spe:,}  tokens_seen={tokens_seen:,}")
    else:
        cov_pct = min(1.0, epochs) * 100.0           # unique fraction of corpus, saturating at one epoch
        passes = f"  ({epochs:.2f}x passes)" if epochs >= 1.0 else ""
        lines.append(
            f" coverage: epochs={epochs:.2f}  corpus={cov_pct:.1f}%{passes}  "
            f"steps/epoch={spe:,}  tokens_seen={tokens_seen:,}"
        )
        if full_corpus_tokens is not None and full_corpus_tokens > T:
            frac = T / full_corpus_tokens * 100.0
            lines.append(f" stream: {T:,}/{full_corpus_tokens:,} tokens = {frac:.1f}% of full {dataset} (capped)")
    return lines


def _fmt_mixer_lr(configured: Optional[float], mean_lr: float) -> str:
    """Format a dedicated mixer LR while making ``None`` inheritance explicit."""
    return f"{mean_lr} (inherit mu)" if configured is None else str(configured)


def _fmt_tau(cfg: VFE3Config, model: VFEModel) -> str:
    r"""Banner format for the softmax temperature: a scalar kappa -> '1.5000', a per-head
    (list) kappa -> '[t0, t1, ...]' (T1). Converts a list kappa to a tensor first so attention_tau
    does not choke on a raw list."""
    dev = next(model.parameters()).device
    tau = attention_tau(_as_coeff(cfg.kappa_beta, dev), model.group.irrep_dims)
    if isinstance(tau, torch.Tensor) and tau.dim() >= 1:
        return "[" + ", ".join(f"{x:.4f}" for x in tau.tolist()) + "]"
    return f"{float(tau):.4f}"


def parameter_report(
    model:  VFEModel,

    *,
    device: Optional[torch.device] = None,
    batch:  int = 2,
    seqlen: int = 16,
) -> Dict[str, object]:
    r"""Total params plus the ACTUAL-trained-under-this-config count.

    ``total`` is ``sum(p.numel())``; ``trainable`` filters ``requires_grad`` (catches the parameters
    frozen by design). ``live`` is MEASURED, not inferred from
    toggles: one synthetic forward+backward runs and a parameter counts as live only if it receives
    a non-None gradient -- so config-dead tables that remain allocated and grouped
    (``decode_log_scale`` under ``decode_mode='linear'``, ``phi_embed`` under
    ``e_step_gradient='detach'``, ...) are counted as
    dead REGARDLESS of which toggle silenced them, so the report never drifts from the config as the
    toggles change. The synthetic ids come from a LOCAL generator (the global RNG is untouched) and
    ``.grad`` is cleared afterward, so the probe is side-effect-free. Best-effort: any failure (e.g. a
    config whose loss does not require grad) falls back to ``live = trainable`` with ``probed=False``.
    """
    named     = list(model.named_parameters())
    total     = sum(p.numel() for _, p in named)
    trainable = sum(p.numel() for _, p in named if p.requires_grad)
    rep: Dict[str, object] = {
        "total": total,
        "allocated": total,
        "trainable": trainable,
        "optimizer_updated": trainable,
        "live": trainable,
        "dead": 0,
        "dead_names": [],
        "probed": False,
    }
    try:
        device = device or next(model.parameters()).device
        n = max(2, min(int(seqlen), int(model.cfg.max_seq_len)))
        g = torch.Generator().manual_seed(0)                       # local: global RNG untouched
        ids = torch.randint(0, int(model.cfg.vocab_size), (int(batch), n), generator=g).to(device)
        tgt = torch.randint(0, int(model.cfg.vocab_size), (int(batch), n), generator=g).to(device)
        model.zero_grad(set_to_none=True)
        rng_state = torch.get_rng_state()                          # m31: the probe forward may draw the
        import vfe3.model.prior_bank as _prior_bank                # the probe runs before train() fills the
        unigram_warned = _prior_bank._WARNED_UNIGRAM_UNSET          # unigram table: hold PriorBank's once-per-
        _prior_bank._WARNED_UNIGRAM_UNSET = True                   # process "unset" warning, which would be false
        try:                                                       # restore global RNG
            out  = model(ids, tgt)                                 # (logits|None, loss, ce) with targets
            loss = out[1] if isinstance(out, tuple) else out
            loss.backward()
        finally:
            torch.set_rng_state(rng_state)                         # global stream untouched, per the docstring
            _prior_bank._WARNED_UNIGRAM_UNSET = unigram_warned
        dead_names = [name for name, p in named if p.requires_grad and p.grad is None]
        live = sum(p.numel() for _, p in named if p.requires_grad and p.grad is not None)
        model.zero_grad(set_to_none=True)                          # leave no grads for training
        rep.update(live=live, dead=trainable - live, dead_names=dead_names, probed=True)
    except Exception as exc:
        rep["probe_error"] = repr(exc)                             # surfaced, not swallowed; live/dead stay unknown
    return rep


def snapshot_parameters(
    model: torch.nn.Module,
) -> Dict[str, torch.Tensor]:
    r"""Detached CPU clones of every named parameter, for a later ``parameter_motion_report``.

    Cloned to the HOST so a large model costs no extra device memory; the comparison at the end of
    the run is the only transfer. Taken at the first step of THIS run, so under a resume the motion
    reported is motion since the resume point, not since the original init.
    """
    return {name: p.detach().to("cpu", copy=True) for name, p in model.named_parameters()}


def parameter_motion_report(
    model:      torch.nn.Module,
    snapshot:   Mapping[str, torch.Tensor],         # from snapshot_parameters, taken before training

    *,
    rel_tol:    float                = 1e-6,        # below this relative motion a parameter is frozen
    known_dead: Collection[str]      = (),          # names that never receive a gradient at all
) -> Dict[str, object]:
    r"""How far each parameter ACTUALLY moved over the run, and which ones did not move at all.

    Motion is ``||p - p_0||_2``, reported relative to ``||p_0||_2`` where that is nonzero. For a
    zero-init table (``mixer_deltas``, ``path_weights``) the relative form is
    undefined, so the absolute norm is the criterion instead and ANY nonzero motion counts as
    trained -- which makes those the sharpest members of the set, not the murkiest.

    Only ``requires_grad`` parameters can be reported frozen: a parameter deliberately held fixed
    (``requires_grad=False``) must not be flagged.

    This is the OUTCOME half of the pair whose CAPABILITY half is ``parameter_report``. That one runs
    a synthetic backward at init and flags ``p.grad is None``, so it sees a table severed from the
    loss; it cannot see a table that is handed a gradient and still never moves. Measured on a tiny
    model: with ``m_phi_lr=0.0``, ``parameter_report`` reports ``phi_embed`` live while it moves
    exactly ``0.0`` across real optimizer steps.

    ``known_dead`` -- the names that never receive a gradient at all, which ``train`` reads off the
    first real step and ``parameter_report`` supplies as ``dead_names`` -- splits the immobile set in
    two, because only one half is news. A table already known dead is config-dead by construction
    -- ``decode_log_scale`` under the default ``decode_mode='linear'`` is the standing example -- and
    lands in ``frozen_known_dead``; it is reported but never warned about, so the warning does not
    fire on every run of an intentional ablation and get tuned out. ``frozen`` keeps only the
    parameters that were live at init and still did not move, which no existing check can see.
    """
    rep: Dict[str, object] = {"rel_tol": float(rel_tol), "frozen": [], "frozen_known_dead": [],
                              "missing": [], "motion": {}}
    motion: Dict[str, Dict[str, object]] = rep["motion"]                          # type: ignore[assignment]
    dead = set(known_dead)
    for name, parameter in model.named_parameters():
        initial = snapshot.get(name)
        if initial is None or initial.shape != parameter.shape:
            rep["missing"].append(name)                                           # type: ignore[union-attr]
            continue                                                              # added/reshaped since the snapshot
        current   = parameter.detach().to("cpu")
        delta     = float((current - initial).norm())
        init_norm = float(initial.norm())
        relative  = (delta / init_norm) if init_norm > 0.0 else None
        frozen    = (delta <= rel_tol * init_norm) if init_norm > 0.0 else (delta == 0.0)
        motion[name] = {"abs": delta, "rel": relative, "init_norm": init_norm,
                        "requires_grad": bool(parameter.requires_grad), "frozen": bool(frozen),
                        "known_dead": name in dead}
        if frozen and parameter.requires_grad:
            key = "frozen_known_dead" if name in dead else "frozen"
            rep[key].append(name)                                                 # type: ignore[union-attr]
    rep["n_parameters"] = len(motion)
    return rep


def _banner(
    model:       VFEModel,
    cfg:         VFE3Config,
    dataset:     str,
    device:      torch.device,
    n_steps:     int,
    train_loader: object = None,
) -> str:
    r"""Compact init banner (no FLOPs counter)."""
    rep = parameter_report(model, device=device)
    bar = "=" * 64
    cov = coverage_lines(train_loader, n_steps, dataset) if train_loader is not None else []
    live_note = (f" ({rep['live']:,} live, {rep['dead']:,} dead)"
                 if rep["probed"] and rep["dead"] else "")
    probe_note = "" if rep["probed"] else "  [live/dead probe failed]"   # distinguish failure from all-live
    dead_line = ([" dead under config (no grad): "
                  + ", ".join(n.replace("prior_bank.", "") for n in rep["dead_names"])]
                 if rep["probed"] and rep["dead_names"] else [])
    head_evidence_lr = (
        _fmt_mixer_lr(cfg.m_head_evidence_lr, cfg.m_p_mu_lr)
        if cfg.use_priorbank_head_evidence_mixer else "inactive"
    )
    return "\n".join([
        bar,
        f" Gauge VFE Transformer | {rep['total']:,} params{live_note}{probe_note} | {device}",
        bar,
        f" K={cfg.embed_dim}  N={cfg.max_seq_len}  L={cfg.n_layers}  heads={len(model.group.irrep_dims)}  "
        f"group={cfg.gauge_group}  family={cfg.family}",
        f" steps={n_steps}  batch={cfg.batch_size}  dataset={dataset}",
        *cov,
        *dead_line,
        f" M-LRs: mu={cfg.m_p_mu_lr}  sigma={cfg.m_p_sigma_lr}  phi={cfg.m_phi_lr}",
        f" mixer-LRs: head_evidence={head_evidence_lr}  "
        f"head_mixer={_fmt_mixer_lr(cfg.m_head_mixer_lr, cfg.m_p_mu_lr)}",
        f" VFE: lambda_alpha={cfg.lambda_alpha}  kappa_beta={cfg.kappa_beta}  "
        f"tau={_fmt_tau(cfg, model)}  mass_phi={cfg.mass_phi}",
        f" seed={cfg.seed}",
        bar,
    ])


def run_training(
    cfg:     VFE3Config,
    dataset: str = "wikitext-2",
    split:   str = "train",

    *,
    n_steps:    int           = 1000,
    max_tokens: Optional[int] = None,
) -> Tuple[VFEModel, List[float]]:
    r"""Click-to-run entry: build a model + a cached dataloader by name and train (no CLI).

    Constructs a ``VFEModel`` from ``cfg``, a causal-LM dataloader from the tokenized
    cache for ``dataset``/``split`` (capped at ``max_tokens`` for fast runs), prints the
    init banner, and trains for ``n_steps`` M-steps with the config-selected console
    logging (``cfg.log_interval``, ``cfg.eval_interval``). Returns the trained model and
    its loss history.

    DEPRECATED / minimal: superseded by ``train_vfe3.main()``, which is the canonical entry
    point. This helper passes no ``artifacts`` (so ``checkpoint_interval``/best-model/CSV are
    never written), reuses ``loader`` as the validation loader (train == val), and never runs
    the end-of-run test eval. Kept only for the lightweight in-process smoke use it already had.
    """
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VFEModel(cfg).to(device)         # move to CUDA where available (mirrors train_vfe3.main)
    loader = make_dataloader(dataset, split, cfg.max_seq_len, cfg.batch_size,
                             max_tokens=max_tokens, vocab_size=cfg.vocab_size)
    logger = logging.getLogger(__name__)
    logger.info(_banner(model, cfg, dataset, device, n_steps, train_loader=loader))
    losses = train(
        model, loader, cfg,
        n_steps=n_steps,
        grad_clip=cfg.grad_clip,
        log_interval=cfg.log_interval,
        eval_interval=cfg.eval_interval,
        val_loader=loader,
        device=device,
    )
    return model, losses
