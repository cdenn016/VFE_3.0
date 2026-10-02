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
from typing import (Any, Callable, Collection, Dict, Iterable, List, Mapping, NamedTuple, Optional,
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

from vfe3.contracts import is_vfe_model, model_family_of
from vfe3.config import VFE3Config
from vfe3.contracts import DataState, DataStateBuffer
from vfe3.data.datasets import make_dataloader
from vfe3.free_energy import attention_tau
from vfe3.gauge_optim import (
    FISHER_DIAGNOSTIC_KEYS,
    GROUP_LOG_DIAGNOSTIC_KEYS,
    GaugeManifoldAdamW,
    embedded_phi_frobenius_norm,
    get_phi_update_policy,
    group_log_block_shape,
    phi_projection_chunk_rows,
    project_phi_parameter_rows_,
)
from vfe3.geometry.phi_fisher import FisherRowStatistics, middle_fiber_moments, pooled_pos_operator
from vfe3.metric_contracts import perplexity_from_ce
from vfe3.metrics import LINK_SPECTRUM_KEYS
from vfe3.model.block import _as_coeff
from vfe3.model.model import DiagnosticSnapshot, VFEModel
from vfe3.model.prior_bank import PriorBank, get_decode_registration
from vfe3.run_artifacts import RunArtifacts          # top-level safe: run_artifacts imports evaluate
#                                                      lazily (function-local), so there is no cycle
from vfe3.runtime import seed_everything
from vfe3.timing import TrainingTimer
from vfe3.viz.extract import e_step_belief_trace as _snapshot_trace
from vfe3.geometry.transport import TRANSPORT_CLAMP_MAX_NORM   # single source for the phi-clamp threshold (M2)
from vfe3.geometry.retraction import (                          # SPD-retraction non-finite tangent guard
    nonfinite_tangent_elements,
    reset_nonfinite_tangent_elements,
)
from vfe3.numerics import (                    # non-equivariant / non-PD fallback accounting
    decode_logdet_fallback_elements,
    mu_trust_fallback_elements,
    reset_decode_logdet_fallback_elements,
    reset_mu_trust_fallback_elements,
    bounded_variance_from_log,
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


class PhiFisherStatisticsProvider:
    r"""Batch belief statistics for the ``fisher_group`` phi direction (derivation 2026-09-28, Track 2).

    The metric is evaluated at the layer-0 prior-bank rows -- the generative-model object the
    M-step updates -- never at deeper-layer posteriors. :meth:`set_batch` is called by
    :func:`train_step` immediately before the optimizer step with the full step's input tokens (the
    microbatches of ``grad_accum_steps > 1`` are chunks of that tensor along dim 0, so it is their
    concatenation) and the scored-token count :math:`N` that divides the CE gradient.

    For ``phi_embed`` rows :math:`v` it returns the table row :math:`(\mu_v, s_v)` and
    :math:`n_v`, the occurrences of :math:`v` among the input tokens. For ``pos_phi_free`` rows
    :math:`p` it returns the pooled operator of the batch column ``tokens[:, p]`` over the
    middle-fiber beliefs :math:`E_{v_b}^{-1}\cdot q_{v_b}`, and :math:`B_{\rm tot}`. Everything is
    float64, detached, on the table's device.
    """

    def __init__(self, model: VFEModel, rotation_weight: float) -> None:
        self._model = model
        self._rotation_weight = float(rotation_weight)
        self._blocks = group_log_block_shape(model.group)                     # (H, d)
        self._tokens: Optional[torch.Tensor] = None
        self._scored: Optional[torch.Tensor] = None

    def set_batch(
        self,
        tokens: Optional[torch.Tensor],                                       # (B_tot, N) input ids
        scored: Optional[torch.Tensor],                                       # () scored-token count
    ) -> None:
        r"""Record the step's tokens and scored count; ``None`` clears them after the step."""
        self._tokens = tokens
        self._scored = scored

    def _type_moments(
        self,
        rows: torch.Tensor,                                                   # (T,) type ids
    ) -> Tuple[torch.Tensor, torch.Tensor]:                                   # (T, H, d) mu, s
        pb = self._model.prior_bank
        H, d = self._blocks
        mu = pb._prior_mu_table()[rows].detach().to(dtype=torch.float64)
        s = bounded_variance_from_log(
            pb._prior_sigma_log_table()[rows].detach().to(dtype=torch.float64), eps=pb.eps)
        return mu.view(-1, H, d), s.view(-1, H, d)

    @torch.no_grad()
    def __call__(
        self,
        param: torch.nn.Parameter,
        rows:  torch.Tensor,                                                  # (R,) active row ids
    ) -> FisherRowStatistics:
        if self._tokens is None or self._scored is None:
            raise RuntimeError(
                "fisher_group statistics requested without batch tokens; train_step sets them "
                "immediately before the optimizer step")
        tokens = self._tokens.to(device=rows.device)
        total = self._scored.to(device=rows.device, dtype=torch.float64).clamp_min(1.0)
        pb = self._model.prior_bank
        if param is pb.phi_embed:
            mu, s = self._type_moments(rows)
            count = torch.bincount(tokens.reshape(-1), minlength=param.shape[0])[rows]
            return FisherRowStatistics(kind="type", count=count.to(dtype=torch.float64),
                                       total=total, local_mean=mu, local_var=s)
        if param is getattr(self._model, "pos_phi_free", None):
            H, d = self._blocks
            columns = tokens[:, rows]                                         # (B_tot, R)
            types, inverse = columns.unique(return_inverse=True)
            X = pb.phi_embed[types].detach().to(dtype=torch.float64).view(-1, H, d, d)
            xi0, C0, C0_inv = middle_fiber_moments(X, *self._type_moments(types))   # (T, H, ...)
            gather = inverse.transpose(0, 1)                                  # (R, B_tot)
            operator = pooled_pos_operator(
                xi0[gather], C0[gather], C0_inv[gather],
                rotation_weight=self._rotation_weight)                        # (R, H, d*d, d*d)
            count = torch.full((rows.shape[0],), float(tokens.shape[0]),
                               dtype=torch.float64, device=rows.device)
            return FisherRowStatistics(kind="pos", count=count, total=total,
                                       pooled_operator=operator)
        raise ValueError("fisher_group statistics exist only for phi_embed and pos_phi_free")


def _decode_learning_rates(cfg: VFE3Config) -> tuple[float, float]:
    """Resolve split decoder rates, falling back to the encoder rates."""
    mu_lr = cfg.m_p_mu_lr if cfg.m_decode_mu_lr is None else cfg.m_decode_mu_lr
    sigma_lr = cfg.m_p_sigma_lr if cfg.m_decode_sigma_lr is None else cfg.m_decode_sigma_lr
    return mu_lr, sigma_lr


def build_optimizer(
    model: VFEModel,
    cfg:   VFE3Config,
) -> torch.optim.Optimizer:
    r"""AdamW with per-group M-step learning rates over the PriorBank prior tables.

    The prior-table roles carry distinct natural scales, so each is given its own
    M-step learning rate: every live mean table at ``m_p_mu_lr``; every live (log) scale
    table and the decode temperature ``decode_log_scale`` together
    at ``m_p_sigma_lr``; the belief gauge-frame coordinates ``phi_embed`` at ``m_phi_lr``.
    ``m_decode_mu_lr`` controls separate decode means or linear weight/bias;
    ``m_decode_sigma_lr`` controls separate decode log variances. Each falls back to the
    corresponding encoder rate. Learned temperature trains at ``m_p_sigma_lr``; shared
    encode/decode tables keep encoder rates.
    Omitting overrides preserves historical group topology and semantic mu/sigma decays.
    Head evidence keeps its dedicated ``m_head_evidence_lr``.

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

    The per-type block-MLP hidden bias (``per_type_hidden_bias``, the capacity-bearing per-type table
    of the identity-frame arm) takes the frame table's hyperparameters, the step size
    ``m_per_type_hidden_bias_lr`` (the declared ``m_phi_lr`` unless set) and ``phi_weight_decay``,
    and its role ``"phi"`` (so its own per-role clip budget, as the frame table it stands in for),
    without the pullback-group metadata: it is not a Lie-algebra coordinate.

    A model of another family (``model_family``, e.g. :class:`vfe3.baseline.BaselineTransformer`)
    builds its own optimizer through ``model.build_optimizer(cfg)``.
    """
    if not is_vfe_model(model):
        return model.build_optimizer(cfg)
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
    decode_mu_lr, decode_sigma_lr = _decode_learning_rates(cfg)
    # The historical shared sigma/temperature group.
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
        # the tables they were cloned from unless their decoder rate overrides the fallback.
        groups.append({"params": [pb.decode_mu_embed],        "lr": decode_mu_lr,    "role": "mu", **mu_wd})
        groups.append({"params": [pb.decode_sigma_log_embed], "lr": decode_sigma_lr, "role": "sigma", **sigma_wd})
    if pb.output_proj_weight is not None:                       # decode_mode='linear'
        groups.append({"params": [pb.output_proj_weight], "lr": decode_mu_lr, "role": "mu", **mu_wd})
    if pb.output_proj_bias is not None:                         # decode_bias: learned log-unigram prior
        # weight_decay=0 -- decaying a unigram prior toward zero biases it to a flat distribution
        # (the same protection phi/Omega carry).
        groups.append({"params": [pb.output_proj_bias], "lr": decode_mu_lr, "weight_decay": 0.0, "role": "mu"})
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
    hidden_bias = getattr(model, "per_type_hidden_bias", None)
    if hidden_bias is not None:                                 # per_type_hidden_bias=True table
        hidden_bias_parameters = ([hidden_bias] if isinstance(hidden_bias, torch.nn.Parameter)
                                  else list(hidden_bias.parameters()))
        hidden_bias_parameters = [p for p in hidden_bias_parameters if p.requires_grad]
        # m_per_type_hidden_bias_lr resolves (in VFE3Config) to the DECLARED m_phi_lr, which
        # gauge_transport 'off'/'frozen' would otherwise have coerced to 0 for this table too.
        hidden_bias_lr = getattr(cfg, "m_per_type_hidden_bias_lr", None)
        if hidden_bias_parameters:
            groups.append({"params": hidden_bias_parameters,
                           "lr": cfg.m_phi_lr if hidden_bias_lr is None else hidden_bias_lr,
                           "weight_decay": cfg.phi_weight_decay, "role": "phi"})
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
    # a null gradient under specific opt-in toggles, by design: decode_log_scale under
    # decode_mode='linear' (the readout discards tau_eff). That is an intentional route property,
    # not a coverage bug.
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
            phi_group_retraction=cfg.m_phi_group_retraction,
            phi_statistics_provider=(
                PhiFisherStatisticsProvider(model, cfg.m_phi_fisher_rotation_weight)
                if phi_update_policy.requires_fisher_statistics else None
            ),
            phi_fisher_rotation_weight=cfg.m_phi_fisher_rotation_weight,
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

    *,
    required:     Collection[str] = ("mu", "sigma", "phi"),
) -> Dict[str, float]:
    """Resolve the stable CSV learning-rate fields from optimizer role metadata.

    The report roles must include every ``required`` role and lie within ``{mu, sigma, phi}``; a
    role that is allowed but absent (only possible when ``required`` is narrowed, as for a
    conventional-transformer baseline) resolves to NaN. With the default ``required`` this is the
    exact-set check the VFE optimizer has always satisfied.
    """
    if len(param_groups) != len(lrs):
        raise RuntimeError("optimizer parameter groups and scheduler learning rates differ in length")
    resolved = {
        str(group["lr_report_role"]): float(lr)
        for group, lr in zip(param_groups, lrs)
        if "lr_report_role" in group
    }
    allowed = {"mu", "sigma", "phi"}
    required_set = set(required)
    if not required_set <= set(resolved) or not set(resolved) <= allowed:
        raise RuntimeError(
            f"optimizer learning-rate report roles must include {sorted(required_set)} within "
            f"{sorted(allowed)}, got {sorted(resolved)}"
        )
    return {**{role: float("nan") for role in ("mu", "sigma", "phi")}, **resolved}


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
        if torch.cuda.device_count() > 0
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
    """Validate an owned CPU batch before transfer for each built-in fused decoder.

    Public/custom decoders and device-origin loaders retain their normal validation. Views made
    later (accumulation or grouped padding) do not inherit the one-use record and revalidate.
    """
    decode_mode = getattr(getattr(model, "cfg", None), "decode_mode", None)
    if (type(model) is VFEModel and type(model.prior_bank) is PriorBank
            and decode_mode in ("linear", "diagonal_chunked", "full_chunked",
                                "expected_likelihood_chunked", "family_chunked")
            and get_decode_registration(decode_mode).fused_ce is getattr(
                PriorBank, "decode_ce_linear_chunked" if decode_mode == "linear"
                else f"decode_ce_{decode_mode}")
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
    frame_projection_out: Optional[torch.Tensor]         = None,   # (2,) int64 device counter
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

    ``frame_projection_out`` (S1 panel D counters): a ``(2,)`` int64 tensor on the generator
    device. When given and the post-update frame projection runs, the rows it rescales and the rows
    it checks are added to it on the device (no host synchronization); ``None`` makes exactly the
    call the step made before the counter existed.
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
        # Pre-clip gradient norm per NAMED parameter group (audit 2026-09-28): the role tags above
        # lump W_out, the MLPs, the relational maps and the LayerNorm into "mu". Read-only; a
        # failure is handed to train() under a private key and never reaches the update.
        try:
            metrics_out.update(_group_gradient_norms(model))
        except Exception as exc:
            metrics_out[_GROUP_GRAD_ERROR_KEY] = f"{type(exc).__name__}: {exc}"
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
        # fisher_group: the belief-Fisher direction reads this step's tokens and CE denominator.
        _fisher_provider = getattr(optimizer, "phi_statistics_provider", None)
        if _fisher_provider is not None:
            _fisher_provider.set_batch(tokens, _scored_det)
        _scaler.step(optimizer)
        _scaler.update()
        if _fisher_provider is not None:
            _fisher_provider.set_batch(None, None)
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
        and getattr(_cfg, "phi_mstep_max_matrix_norm", None) is not None
    ):
        collect_projection_stats = metrics_out is not None
        # The device counter rides along only when requested; None keeps the original call.
        projection_count = ({} if frame_projection_out is None
                            else {"count_out": frame_projection_out})
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
                        **projection_count,
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
                    **projection_count,
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
                **projection_count,
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
    "estep_fp_phi_rms", "estep_beta_js", "estep_alpha_rms_delta",
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
    "val_relational_score_klmax_frac",
    "val_relational_mean_displacement_norm", "val_relational_snapshot_ce",
    "val_relational_receiver_cost_before", "val_relational_receiver_cost_after",
    "val_relational_receiver_cost_delta", "val_relational_receiver_noise",
    "val_relational_receiver_mean_cost_delta", "val_relational_receiver_covariance_cost_delta",
    "val_relational_receiver_target_distance_ratio",
) + tuple("val_relational_" + key for key in (
    "score_raw_min_singular_value", "score_q_raw_min_singular_value",
    "score_k_raw_min_singular_value", "score_rank_per_head", "content_rank_per_head",
    "parameter_count", "content_max_singular_value", "content_effective_rank",
    "projected_cov_min_eigenvalue", "projected_cov_max_condition", "covariance_repair_count")
) + tuple(f"val_{key}" for key in LINK_SPECTRUM_KEYS)


# =============================================================================
# Training-time diagnostics for the multi-seed runs (audit 2026-09-28, Section 2).
#
# Every quantity below is computed only on a log/eval-cadence step (or at step 0 of an evaluated
# run), off the training graph, with the CPU and CUDA RNG streams restored. None of it runs on the
# silent path (log_interval and eval_interval falsy), and none of it feeds back into the optimizer,
# so the training trajectory is unchanged. A failure is logged once per diagnostic name, counted in
# ``RunArtifacts.diagnostics_failures`` (summary.json) and in the cumulative ``diagnostics_failures``
# metrics column, and training continues.
# =============================================================================

# Log-spaced evaluations below the first regular evaluation, in completed optimizer steps. Step 0
# is evaluated before the first update. Only steps below ``eval_interval`` are used, and only when
# evaluation is enabled with a run directory (``artifacts``) and ``eval_interval >=
# EARLY_EVAL_MIN_INTERVAL``: a shorter cadence already samples the start of training, and the gate
# keeps the metrics.csv schedule of short debug/test runs unchanged. The step-0 evaluation does not
# take part in best-model selection (the untrained initialization is never a selection candidate);
# the positive early steps are ordinary evaluations and do.
EARLY_EVAL_STEPS:        Tuple[int, ...] = (0, 1000, 3000, 8000)
EARLY_EVAL_MIN_INTERVAL: int             = 1000

# Fixed training sample for the generalization gap: windows of the TRAIN split drawn by a generator
# with this fixed seed (independent of cfg.seed and the data seed, so every seed of a multi-seed
# study scores the same windows), scored under the validation protocol, and sized to the validation
# split's scored-target count, capped here.
TRAIN_SAMPLE_SEED:        int = 0
TRAIN_SAMPLE_MAX_TARGETS: int = 262_144

# Fixed validation windows for the per-layer residual ratio and per-head attention statistics.
DIAGNOSTIC_WINDOWS:       int = 8
DIAGNOSTIC_CHUNK_WINDOWS: int = 4          # windows per diagnostic forward (bounds capture memory)
UPDATE_NORM_CHUNK_ELEMENTS: int = 1 << 22  # device chunk for ||theta_{t+1} - theta_t|| (~32 MB fp64 per temporary)

# Named parameter groups for gradient, parameter and update norms (by parameter NAME, not by the
# optimizer's role tag, whose "mu" role also holds W_out, the MLPs, the relational maps and the
# LayerNorm). "per_type_table" is the equal-size capacity-control table: the VFE per-type block-MLP
# hidden bias, or the conventional baseline's per-type read map (its columns are blank otherwise).
PARAMETER_GROUPS: Tuple[str, ...] = (
    "mu_embed", "sigma_log_embed", "phi_embed", "pos_phi", "decode",
    "relational_score", "relational_content", "block_mlp", "norm", "per_type_table", "other",
)
_GROUP_COLUMNS: Tuple[str, ...] = tuple(
    f"{kind}_group_{group}"
    for kind in ("grad_norm", "param_norm", "update_ratio") for group in PARAMETER_GROUPS)
_GRAD_GROUP_COLUMNS: Tuple[str, ...] = tuple(f"grad_norm_group_{group}" for group in PARAMETER_GROUPS)
_TRAIN_SAMPLE_COLUMNS: Tuple[str, ...] = (
    "train_sample_ce", "train_sample_ppl", "generalization_gap_fixed")
_GROUP_GRAD_ERROR_KEY = "_group_gradient_norm_error"     # private train_step -> train handoff

# Held-out columns the conditional route now fills from its own capture (all members of _VAL_DIAG_KEYS).
_CONDITIONAL_STRUCTURE_KEYS: Tuple[str, ...] = (
    "val_effective_rank", "val_belief_cond_median",
    "val_attn_entropy_min_all", "val_attn_entropy_min_norm", "val_attn_collapsed_heads",
    "val_future_leakage", "val_row_sum_error", "val_pos_content_r2", "val_prev_token_mass",
    "val_period_match_mass", "val_head_redundancy_js",
    "pos_loss_first_q", "pos_loss_last_q", "pos_loss_ratio",
)

RESIDUAL_RATIO_FIELDS: Tuple[str, ...] = (
    "step", "layer", "q10", "q50", "q90", "ln_rel_change_q50")
ATTENTION_STATS_FIELDS: Tuple[str, ...] = (
    "step", "layer", "head", "entropy_norm", "distance_mean", "self_weight", "prior_overlap", "n_rows")

# S1 panel D numerical-event counters (GL(K)_attention_v2 fig:placeholder_numerics, 2026-09-28).
# Every count is an integer with its denominator beside it. NUMERICS_EVAL_COLUMNS are measured at
# every evaluation (and at step 0) on the DIAGNOSTIC_WINDOWS fixed validation windows inside the
# residual-ratio capture pass (no extra forward), with one diagnostics/cap_activation.csv row per
# (layer, head); they are NaN on rows without an evaluation, when the run
# has no diagnostic windows, and when their capture pass fails. NUMERICS_CUMULATIVE_COLUMNS are on
# every metrics row and cover every attempted optimizer step since step 0 (frame_projection_* NaN
# without a post-update frame projection). The exact definitions and skip rules are in
# _capture_numerics_counts, _numerics_counter_skip_reasons and _cumulative_counter_columns; the
# final values go to summary.json "numerics_counters" (_numerics_counters_summary).
NUMERICS_EVAL_COLUMNS: Tuple[str, ...] = (
    "cap_active_count", "cap_entry_count", "factorization_fallback_count", "factorization_count")
NUMERICS_CUMULATIVE_COLUMNS: Tuple[str, ...] = (
    "frame_projection_count", "frame_projection_total", "skipped_updates", "attempted_updates")
CAP_ACTIVATION_FIELDS: Tuple[str, ...] = ("step", "layer", "head", "capped", "entries")
_EVAL_CADENCE_COLUMNS: Tuple[str, ...] = _TRAIN_SAMPLE_COLUMNS + NUMERICS_EVAL_COLUMNS


def _numerics_counter_skip_reasons(
    model: VFEModel,
) -> Dict[str, Optional[str]]:
    r"""Why each S1 panel D counter family is undefined for ``model``; None where it is measured.

    ``cap_activation`` (``cap_active_count``, ``cap_entry_count``, ``diagnostics/cap_activation.csv``)
    and ``factorization`` (``factorization_fallback_count``, ``factorization_count``) are measured on
    both supported projected routes, whose pre-update router records its capped costs and reduced
    covariances in the diagnostic capture. The full-link gradient episode keeps that router fixed
    across its receiver iterations.
    ``frame_projection`` (``frame_projection_count``, ``frame_projection_total``) is measured when
    :func:`train_step` runs the post-update frame projection, i.e. ``m_phi_update_mode='adamw'`` with
    ``phi_mstep_max_matrix_norm`` set; otherwise both stay NaN. ``skipped_updates`` and
    ``attempted_updates`` are defined on every route.
    """
    cfg = model.cfg
    if not is_vfe_model(model):
        family = model_family_of(model)
        return {
            "cap_activation": f"model_family={family!r}: no capped routing cost exists",
            "factorization": f"model_family={family!r}: no reduced covariance is factorized",
            "frame_projection": f"model_family={family!r}: no frame table is projected",
        }
    reasons: Dict[str, Optional[str]] = {
        "cap_activation": None, "factorization": None, "frame_projection": None}
    mode = getattr(cfg, "m_phi_update_mode", "adamw")
    if mode != "adamw":
        reasons["frame_projection"] = (
            f"m_phi_update_mode={mode!r}: no post-update frame projection (the pullback-group step "
            "rejects rows outside the chart bound instead of projecting them)")
    elif cfg.phi_mstep_max_matrix_norm is None:
        reasons["frame_projection"] = (
            "phi_mstep_max_matrix_norm is None: the run has no post-update frame projection")
    return reasons


@torch.no_grad()
def _capture_numerics_counts(
    diagnostic: Mapping[str, Any],
    kl_max:     float,
) -> Tuple[List[List[Tuple[int, int]]], List[Tuple[int, int]]]:
    r"""Cost-cap activations per (layer, head) and reduced-covariance factorizations per layer.

    Reads the projected-route records ``layer_conditional_captures`` and ``log_prior`` of one
    ``forward_beliefs(capture={"diagnostic": {}})`` call over ``B`` windows.

    Cap activation. The router bounds every routing cost before the softmax,
    :math:`\bar D_{ij} = \min(D_{ij}, \kappa)` with :math:`\kappa` = ``cfg.kl_max`` (a non-finite cost is
    set to :math:`\kappa`). With :math:`S_i` the causal sources of query :math:`i` (the finite entries of
    the attention log-prior; :math:`\beta_{ij} > 0` when the model has none), the entries evaluated are
    the pairs :math:`(i, j)`, :math:`j \in S_i`, of every query row with :math:`|S_i| \ge 2` (a
    single-source row puts all its weight on itself whatever its cost, so the cap cannot act on it),
    in every window, layer and head. An entry is capped when :math:`\bar D_{ij} \ge \kappa`. This is
    the ``computed`` population of :class:`vfe3.posthoc.klcap.KLCapObserver` on unpadded windows.
    With ``kl_max = inf`` only non-finite costs count as capped.

    Factorizations. Each layer factors its reduced query and key covariances
    :math:`G = L \Sigma L^\top` (:math:`L = S U^{-1}`, comparison rank :math:`r`) once per (window,
    token, head) and role, so the layer's denominator is
    :math:`\mathrm{numel}(G_q)/r^2 + \mathrm{numel}(G_k)/r^2`. The factorization uses zero jitter and
    no spectral repair; its only fallback is the source-factor recovery of
    :func:`vfe3.relational_scoring._strict_cholesky` (float64 QR of the source root
    :math:`L \Sigma^{1/2}`, or a shifted CholeskyQR3 retry on the ``cholesky_qr2`` factor route) for an
    entry whose working-precision factor failed, or failed the conditioning certificate under
    ``projected_cov_condition_recovery``. The numerator is the layer's recovered-entry count
    ``relational_covariance_repair_count``, a subset of the same entries. The pair-level factorizations
    of :math:`G_{q,i} + G_{k,j}` on the noisy-query route have no fallback (a failure raises) and are
    not counted.

    Returns ``(cap, factorization)`` with ``cap[l][h] = (capped, entries)`` and
    ``factorization[l] = (recovered, attempted)``, all Python ints, after one host transfer.
    """
    from vfe3.posthoc.klcap import causal_support       # the post-hoc cap statistic's support rule

    captures = diagnostic.get("layer_conditional_captures") or []
    if not captures:
        raise RuntimeError("the projected-route capture holds no per-layer router records")
    log_prior = diagnostic.get("log_prior")
    device_counts: List[torch.Tensor] = []
    heads: List[int] = []
    attempted: List[int] = []
    for capture in captures:
        score = capture["score_energy_pre"].detach()                        # (B, H, N, N) capped cost
        beta = capture["beta_pre"].detach()                                 # (B, H, N, N) router
        if beta.dim() == 3:
            score, beta = score.unsqueeze(1), beta.unsqueeze(1)             # (B, 1, N, N)
        support = causal_support(beta, log_prior)                           # (B, H, N, N) bool
        rows = support.sum(dim=-1, keepdim=True) >= 2                       # (B, H, N, 1)
        entries = support & rows                                            # (B, H, N, N)
        capped = entries & (score >= kl_max)                                # (B, H, N, N)
        G_q = capture["projected_covariance_pre"]                           # (B, N, H, r, r)
        G_k = capture["projected_key_covariance_pre"]                       # (B, N, H, r, r)
        attempted.append(int(G_q.numel() // (G_q.shape[-1] ** 2)
                             + G_k.numel() // (G_k.shape[-1] ** 2)))
        recovered = torch.as_tensor(capture["relational_covariance_repair_count"]).reshape(1)
        heads.append(int(beta.shape[1]))
        device_counts.append(torch.cat((
            capped.sum(dim=(0, 2, 3)), entries.sum(dim=(0, 2, 3)),          # (H,), (H,)
            recovered.to(device=beta.device, dtype=torch.long))))          # (1,)
    values = torch.cat(device_counts).cpu().tolist()                         # one host transfer
    cap: List[List[Tuple[int, int]]] = []
    factorization: List[Tuple[int, int]] = []
    offset = 0
    for n_heads, n_attempted in zip(heads, attempted):
        capped_h = values[offset:offset + n_heads]
        entries_h = values[offset + n_heads:offset + 2 * n_heads]
        cap.append([(int(c), int(e)) for c, e in zip(capped_h, entries_h)])
        factorization.append((int(values[offset + 2 * n_heads]), n_attempted))
        offset += 2 * n_heads + 1
    return cap, factorization


def _merge_numerics_counts(
    accumulator:   Dict[str, Any],
    cap:           List[List[Tuple[int, int]]],
    factorization: List[Tuple[int, int]],
) -> None:
    r"""Add one chunk's counts to the per-evaluation accumulator (keys ``cap`` and ``factorization``)."""
    per_head = accumulator.setdefault("cap", {})
    for layer, layer_counts in enumerate(cap):
        for head, (capped, entries) in enumerate(layer_counts):
            total = per_head.setdefault((layer, head), [0, 0])
            total[0] += capped
            total[1] += entries
    fact = accumulator.setdefault("factorization", [0, 0])
    for recovered, attempted in factorization:
        fact[0] += recovered
        fact[1] += attempted


def _numerics_eval_columns(
    accumulator: Mapping[str, Any],
    step:        int,
) -> Tuple[Dict[str, int], List[Dict[str, int]]]:
    r"""Metrics columns and ``cap_activation.csv`` rows from one evaluation's accumulator.

    Raises the error a chunk stored (so the caller counts one ``numerics_counters`` failure and
    leaves the columns NaN) and refuses an accumulator that saw no capture.
    """
    error = accumulator.get("error")
    if error is not None:
        raise error
    cap = accumulator.get("cap") or {}
    fact = accumulator.get("factorization")
    if not cap or fact is None:
        raise RuntimeError("no projected-route capture was counted on the diagnostic windows")
    rows = [{"step": int(step), "layer": int(layer), "head": int(head),
             "capped": int(capped), "entries": int(entries)}
            for (layer, head), (capped, entries) in sorted(cap.items())]
    columns = {
        "cap_active_count": sum(row["capped"] for row in rows),
        "cap_entry_count": sum(row["entries"] for row in rows),
        "factorization_fallback_count": int(fact[0]),
        "factorization_count": int(fact[1]),
    }
    return columns, rows


def _note_numerics_eval(
    last:   Dict[str, Any],
    step:   int,
    values: Mapping[str, object],
) -> None:
    r"""Keep the latest evaluation whose per-evaluation counters were all measured (summary.json)."""
    counts = {column: values.get(column) for column in NUMERICS_EVAL_COLUMNS}
    if all(isinstance(value, int) and not isinstance(value, bool) for value in counts.values()):
        last.clear()
        last.update({"eval_step": int(step), **counts})


def _frame_projection_state(
    frame_counter: Optional[torch.Tensor],
    frame_base:    Tuple[int, int],
    since_step:    Optional[int],
) -> Optional[Dict[str, int]]:
    r"""Host copy of the cumulative frame-projection counts for a checkpoint (one host transfer)."""
    if frame_counter is None:
        return None
    changed, checked = (int(value) for value in frame_counter.tolist())
    return {"frame_projection_count": int(frame_base[0]) + changed,
            "frame_projection_total": int(frame_base[1]) + checked,
            "frame_projection_since_step": int(since_step or 0)}


def _numerics_counters_summary(
    completed_steps: int,
    optimizer:       torch.optim.Optimizer,
    frame_state:     Optional[Mapping[str, int]],
    last_eval:       Mapping[str, Any],
    reasons:         Mapping[str, Optional[str]],
    windows:         Optional[torch.Tensor],   # (W, N) diagnostic windows
) -> Dict[str, object]:
    r"""The ``numerics_counters`` record of summary.json (JSON null wherever a count is undefined).

    ``cap_active_count``, ``cap_entry_count``, ``factorization_fallback_count`` and
    ``factorization_count`` are those of the last evaluation that measured them (``eval_step``) on
    ``diagnostic_windows`` fixed validation windows; ``frame_projection_count`` /
    ``frame_projection_total`` (counted from ``frame_projection_since_step``, 0 unless a resumed
    checkpoint carried no counts), ``skipped_updates`` and ``attempted_updates`` are the final
    cumulative values. ``skip_reasons`` names why a family is undefined (None where measured).
    """
    accepted = _successful_update_count(optimizer)
    frame = dict(frame_state) if frame_state is not None else {}
    return {
        "eval_step": last_eval.get("eval_step"),
        **{column: last_eval.get(column) for column in NUMERICS_EVAL_COLUMNS},
        "diagnostic_windows": (int(windows.shape[0]) if isinstance(windows, torch.Tensor) else None),
        "frame_projection_count": frame.get("frame_projection_count"),
        "frame_projection_total": frame.get("frame_projection_total"),
        "frame_projection_since_step": frame.get("frame_projection_since_step"),
        "skipped_updates": int(completed_steps) - int(accepted),
        "attempted_updates": int(completed_steps),
        "skip_reasons": dict(reasons),
    }


def _cumulative_counter_columns(
    completed_steps: int,
    optimizer:       torch.optim.Optimizer,
    frame_counter:   Optional[torch.Tensor],   # (2,) int64 device [rows rescaled, rows checked]
    frame_base:      Tuple[int, int],
) -> Dict[str, float]:
    r"""Cumulative update and frame-projection counters since step 0, for one metrics row.

    ``attempted_updates`` is the number of completed outer training steps (every attempted
    optimizer update) and ``skipped_updates`` the ones the optimizer rejected (nonfinite loss or
    gradient, nonfinite clip norm, zero scored targets, or an AMP scale backoff): attempted minus the
    accepted-update clock that :func:`train_step` keeps in the optimizer groups (restored on resume).
    Both are host integers already, so they cost nothing.

    ``frame_projection_count`` / ``frame_projection_total``: frame-generator rows (rows of
    ``prior_bank.phi_embed`` and of the learned positional table ``pos_phi_free``) that the
    post-update projection onto the embedded Frobenius ball of radius ``phi_mstep_max_matrix_norm``
    rescaled (scale :math:`< 1`), and the rows it checked, summed over every projection call, i.e.
    over the accepted updates. They accumulate on the device inside
    :func:`vfe3.gauge_optim.project_phi_parameter_rows_` and are read here, on a metrics row only.
    ``frame_base`` carries the counts restored from a checkpoint on resume. NaN when the run has
    no post-update projection (see :func:`_numerics_counter_skip_reasons`) or no counter.
    """
    accepted = _successful_update_count(optimizer)
    out: Dict[str, float] = {
        "frame_projection_count": float("nan"), "frame_projection_total": float("nan"),
        "skipped_updates": int(completed_steps) - int(accepted),
        "attempted_updates": int(completed_steps),
    }
    if frame_counter is not None:
        changed, checked = (int(value) for value in frame_counter.tolist())   # log row only
        out["frame_projection_count"] = int(frame_base[0]) + changed
        out["frame_projection_total"] = int(frame_base[1]) + checked
    return out


def _parameter_group(
    name: str,
) -> str:
    r"""Named group of one model parameter (see ``PARAMETER_GROUPS``).

    ``decode`` holds the linear readout ``output_proj_*``, the untied decode tables ``decode_*``
    and ``decode_log_scale``; ``norm`` holds the block and final normalization affines;
    ``per_type_table`` holds the per-type block-MLP hidden bias ``per_type_hidden_bias``; anything
    not matched lands in ``other`` (listed once in the training log). A leading wrapper prefix
    (``_orig_mod.`` from ``torch.compile``, ``module.`` from data parallelism) is ignored.
    Names no VFE rule matches are looked up in the conventional-transformer role map
    (:func:`vfe3.baseline.baseline_parameter_group`), whose names no VFE parameter uses.
    """
    for prefix in ("_orig_mod.", "module."):
        if name.startswith(prefix):
            name = name[len(prefix):]
    leaf = name.rsplit(".", 1)[-1]
    if name == "per_type_hidden_bias" or name.startswith("per_type_hidden_bias."):
        return "per_type_table"
    if name == "prior_bank.mu_embed":
        return "mu_embed"
    if name == "prior_bank.sigma_log_embed":
        return "sigma_log_embed"
    if name == "prior_bank.phi_embed":
        return "phi_embed"
    if name == "pos_phi_free":
        return "pos_phi"
    if name.startswith("prior_bank.") and leaf.startswith(("output_proj", "decode_")):
        return "decode"
    if name.startswith("relational_operators"):
        if leaf in ("raw_score_q", "raw_score_k"):
            return "relational_score"
        if leaf in ("T", "O"):
            return "relational_content"
        return "other"
    if name.startswith("block_mlps."):
        return "block_mlp"
    if name.startswith(("block_norm.", "final_norm.")):
        return "norm"
    from vfe3.baseline import baseline_parameter_group
    return baseline_parameter_group(name) or "other"


def _group_gradient_norms(
    model: torch.nn.Module,
) -> Dict[str, float]:
    r"""Pre-clip gradient norm of each named group, ``||g_G|| = (sum_{p in G} ||grad_p||_2^2)^{1/2}``.

    A group with trainable parameters but no gradient reports 0.0 (the role columns' convention); a
    group with no trainable parameter reports NaN. One host transfer for all groups.
    """
    present = set()
    squares: List[torch.Tensor] = []
    owners: List[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = _parameter_group(name)
        present.add(group)
        if parameter.grad is not None:
            squares.append(torch.linalg.vector_norm(parameter.grad.detach()).to(torch.float64).square())
            owners.append(group)
    totals = {group: 0.0 for group in present}
    if squares:
        for group, value in zip(owners, torch.stack(squares).cpu().tolist()):
            totals[group] += value
    return {f"grad_norm_group_{group}": (math.sqrt(totals[group]) if group in totals else float("nan"))
            for group in PARAMETER_GROUPS}


def _parameter_snapshot(
    model: torch.nn.Module,
) -> Dict[str, torch.Tensor]:
    r"""Host copies of every trainable parameter, taken before an optimizer step on a log step.

    Kept on the CPU so the update-ratio probe never adds device memory to a training step.
    """
    return {name: parameter.detach().to("cpu", copy=True)
            for name, parameter in model.named_parameters() if parameter.requires_grad}


@torch.no_grad()
def _group_update_statistics(
    model:  torch.nn.Module,
    before: Mapping[str, torch.Tensor],       # _parameter_snapshot taken before the step
) -> Dict[str, float]:
    r"""Parameter norm and realized update ratio of each named group across one training step.

    .. math::
        \|\theta_G\| = \Big(\sum_{p \in G} \|p_t\|_2^2\Big)^{1/2},\qquad
        r_G = \frac{\big(\sum_{p \in G} \|p_{t+1} - p_t\|_2^2\big)^{1/2}}{\|\theta_G\|},

    with :math:`p_t` the pre-step value and :math:`p_{t+1}` the value after the optimizer step and any
    post-step projection (the realized update, including weight decay). A rejected step gives
    :math:`r_G = 0`. The difference is formed in float64 on the parameter's device in bounded chunks.
    """
    delta_sq: Dict[str, float] = {}
    norm_sq: Dict[str, float] = {}
    for name, parameter in model.named_parameters():
        previous = before.get(name)
        if previous is None:
            continue
        group = _parameter_group(name)
        current = parameter.detach().reshape(-1)
        prior = previous.reshape(-1)
        delta = torch.zeros((), dtype=torch.float64, device=current.device)
        for start in range(0, current.numel(), UPDATE_NORM_CHUNK_ELEMENTS):
            stop = start + UPDATE_NORM_CHUNK_ELEMENTS
            chunk = prior[start:stop].to(device=current.device, dtype=torch.float64)
            delta.add_((current[start:stop].to(torch.float64) - chunk).square().sum())
        delta_sq[group] = delta_sq.get(group, 0.0) + float(delta)
        norm_sq[group] = norm_sq.get(group, 0.0) + float(
            torch.linalg.vector_norm(prior, dtype=torch.float64).square())   # host tensor
    out: Dict[str, float] = {}
    for group in PARAMETER_GROUPS:
        if group not in norm_sq:
            out[f"param_norm_group_{group}"] = float("nan")
            out[f"update_ratio_group_{group}"] = float("nan")
            continue
        weight = math.sqrt(norm_sq[group])
        out[f"param_norm_group_{group}"] = weight
        out[f"update_ratio_group_{group}"] = (
            math.sqrt(delta_sq[group]) / weight if weight > 0.0 else float("nan"))
    return out


@torch.no_grad()
def _group_parameter_norms(
    model: torch.nn.Module,
) -> Dict[str, float]:
    r"""``param_norm_group_<g>`` alone (no step): the step-0 reference norms."""
    norm_sq: Dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            group = _parameter_group(name)
            norm_sq[group] = norm_sq.get(group, 0.0) + float(
                torch.linalg.vector_norm(parameter.detach()).to(torch.float64).square())
    return {f"param_norm_group_{group}": (math.sqrt(norm_sq[group]) if group in norm_sq
                                          else float("nan"))
            for group in PARAMETER_GROUPS}


@torch.no_grad()
def _sigma_log_statistics(
    model: torch.nn.Module,
) -> Dict[str, float]:
    r"""Mean and standard deviation of the whole prior log-variance table ``sigma_log_embed``.

    Reported instead of its norm: the norm decays toward 0 under weight decay, so a
    gradient-to-weight ratio built on it rises for a trivial reason. Population statistics
    (``correction=0``) in the table's own precision, with no full-size float64 copy on the device.
    """
    table = getattr(getattr(model, "prior_bank", None), "sigma_log_embed", None)
    if not isinstance(table, torch.Tensor) or table.numel() == 0:
        return {"sigma_log_mean": float("nan"), "sigma_log_sd": float("nan")}
    values = table.detach()
    if not values.is_floating_point() or values.element_size() < 4:
        values = values.float()
    sd, mean = torch.std_mean(values, correction=0)
    return {"sigma_log_mean": float(mean), "sigma_log_sd": float(sd)}


def _record_diagnostic_failure(
    failures: Dict[str, int],
    name:     str,
    exc:      BaseException,
    logger:   logging.Logger,
) -> None:
    r"""Count one failed diagnostic; warn only on its first failure."""
    failures[name] = failures.get(name, 0) + 1
    if failures[name] == 1:
        logger.warning(
            "       (diagnostic %r failed: %s: %s); training continues and further failures of "
            "this diagnostic are counted in diagnostics_failures", name, type(exc).__name__, exc)


class FixedTrainSample:
    r"""Fixed training-split windows scored under the validation protocol (an ``evaluate`` loader).

    Window ``k`` is training window ``indices[k]`` of the train loader's dataset (the windows the
    model is trained on). Under a sliding validation window (``stride < seq_len``) only its last
    ``stride`` targets are scored, so every scored target has at least ``seq_len - stride`` tokens of
    context, as in validation; with disjoint validation windows every target is scored. Batches are
    assembled on the CPU without a DataLoader, so iteration draws no RNG.
    """

    def __init__(
        self,
        dataset:    object,
        indices:    Sequence[int],
        batch_size: int,
        stride:     int,
    ) -> None:
        self.dataset = dataset
        self.indices = [int(index) for index in indices]
        self.batch_size = max(1, int(batch_size))
        self.seq_len = int(getattr(dataset, "seq_len"))
        self.stride = int(stride)
        self.masked_prefix = max(0, self.seq_len - self.stride)

    def __len__(self) -> int:
        return -(-len(self.indices) // self.batch_size)

    @property
    def scored_targets_per_window(self) -> int:
        return self.seq_len - self.masked_prefix

    def _window(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = self.dataset[index]
        targets = targets.clone()                     # never mask a view into the corpus
        if self.masked_prefix:
            targets[:self.masked_prefix] = -100
        return inputs, targets

    def __iter__(self) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        for start in range(0, len(self.indices), self.batch_size):
            pairs = [self._window(index) for index in self.indices[start:start + self.batch_size]]
            yield (torch.stack([pair[0] for pair in pairs]),        # (B, N)
                   torch.stack([pair[1] for pair in pairs]))        # (B, N)


def _build_fixed_train_sample(
    train_loader: object,
    val_loader:   Optional[object],
    cfg:          VFE3Config,
) -> Tuple[Optional[FixedTrainSample], Optional[str]]:
    r"""Choose the fixed training sample; returns ``(sample, None)`` or ``(None, reason)``.

    Target count :math:`T = \min(T_{\rm val}, \texttt{TRAIN\_SAMPLE\_MAX\_TARGETS})`, with
    :math:`T_{\rm val}` the validation split's transition count (every transition is scored once
    under the padded evaluation contract), or the cap when the validation loader exposes no token
    stream. :math:`\lceil T / s \rceil` windows are drawn without replacement by
    ``torch.randperm`` under a generator seeded with ``TRAIN_SAMPLE_SEED``, and sorted.
    """
    dataset = getattr(train_loader, "dataset", None)
    if dataset is None or not isinstance(getattr(dataset, "tokens", None), torch.Tensor):
        return None, "train loader exposes no token-window dataset"
    try:
        n_windows_total = int(len(dataset))
        seq_len = int(dataset.seq_len)
    except (TypeError, AttributeError) as exc:
        return None, f"train dataset has no window length ({exc!r})"
    if n_windows_total <= 0:
        return None, "train dataset has no windows"
    stride = seq_len if cfg.eval_stride is None else min(int(cfg.eval_stride), seq_len)
    val_tokens = getattr(getattr(val_loader, "dataset", None), "tokens", None)
    val_targets = (int(val_tokens.numel()) - 1 if isinstance(val_tokens, torch.Tensor)
                   else TRAIN_SAMPLE_MAX_TARGETS)
    n_targets = max(1, min(val_targets, TRAIN_SAMPLE_MAX_TARGETS))
    n_windows = min(n_windows_total, -(-n_targets // stride))
    generator = torch.Generator().manual_seed(TRAIN_SAMPLE_SEED)
    chosen = torch.randperm(n_windows_total, generator=generator)[:n_windows]
    indices = sorted(int(index) for index in chosen.tolist())
    batch_size = int(getattr(val_loader, "batch_size", None) or cfg.batch_size)
    return FixedTrainSample(dataset, indices, batch_size, stride), None


def _diagnostic_windows(
    val_loader: object,

    *,
    n_windows: int = DIAGNOSTIC_WINDOWS,
) -> Optional[torch.Tensor]:
    r"""``(W, N)`` fixed validation input windows for the residual and attention probes, or None.

    From a window dataset: ``W = min(n_windows, usable)`` windows at evenly spaced indices over the
    split (the padded final window of an evaluation stream is excluded). Otherwise the first rows of
    the loader's batches. Callers hold the RNG streams fixed around this (a DataLoader iterator draws
    a base seed).
    """
    dataset = getattr(val_loader, "dataset", None)
    if dataset is not None and hasattr(dataset, "__getitem__") and hasattr(dataset, "__len__"):
        total = int(len(dataset))
        usable = total - 1 if (getattr(dataset, "pad_final", False) and total > 1) else total
        if usable > 0:
            count = min(int(n_windows), usable)
            positions = torch.linspace(0, usable - 1, count, dtype=torch.float64).round().long()
            indices = sorted(set(int(index) for index in positions.tolist()))
            return torch.stack([dataset[index][0].to(torch.long) for index in indices])
    rows: List[torch.Tensor] = []
    for batch in val_loader:
        tokens = batch[0] if isinstance(batch, (tuple, list)) else batch
        for row in tokens:
            rows.append(row.to(torch.long))
            if len(rows) >= n_windows:
                return torch.stack(rows)
    return torch.stack(rows) if rows else None


def _map_snapshot_state(
    model: torch.nn.Module,
) -> Dict[str, torch.Tensor]:
    r"""Every ``relational_operators*`` and ``pos_phi_free`` tensor of the state dict, CPU fp32."""
    return {name: tensor.detach().to(device="cpu", dtype=torch.float32, copy=True)
            for name, tensor in model.state_dict().items()
            if name.startswith("relational_operators") or name.split(".")[-1] == "pos_phi_free"}


def _residual_update_is_additive(
    model: VFEModel,
) -> bool:
    r"""Whether the layer's mean update has the residual form ``mu + m`` (so removing ``mu`` is defined).

    False only for the predictive-content readout with ``attention_residual=False``, which replaces
    the mean by the predictive mean.
    """
    cfg = model.cfg
    return not (getattr(cfg, "relational_readout_mode", None) == "predictive_content"
                and not getattr(cfg, "attention_residual", False))


@torch.no_grad()
def _residual_and_attention_rows(
    model:   VFEModel,
    windows: torch.Tensor,                    # (W, N) fixed validation input windows
    step:    int,
    device:  torch.device,

    *,
    counters_out: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    r"""Per-layer residual-ratio quantiles and per-head attention statistics on fixed windows.

    ``counters_out`` (projected routes): when a dict is given, the S1 panel D per-evaluation counts
    of the same captures (:func:`_capture_numerics_counts`) are accumulated into it, so the counters
    share this forward. A counting error is stored under ``"error"`` (later chunks are then not
    counted) and never disturbs the residual and attention rows.

    One ``forward_beliefs(capture={"diagnostic": {}})`` per chunk of windows, in eval mode (module
    modes restored). For layer :math:`l`, input mean :math:`\mu_i` (the stack-entry belief at
    :math:`l = 0`, else layer :math:`l-1`'s output) and the layer's attention output
    :math:`\mu_i + m_i` (the pre-transform converged mean), per token

    .. math::
        \rho_i = \frac{\|m_i\|}{\|\mu_i\|},\qquad
        \delta_i = \frac{\|\mathrm{LN}(\mu_i + m_i) - \mathrm{LN}(m_i)\|}{\|\mathrm{LN}(\mu_i + m_i)\|},

    reporting the 10/50/90% quantiles of :math:`\rho` and the median of :math:`\delta`.
    :math:`\delta` is NaN when the block has no normalization, when a head mixer or CG coupling sits
    between the E-step and the norm (the norm then does not see :math:`\mu + m`), and for the
    predictive readout without ``attention_residual`` (no residual exists to remove).

    Attention: :math:`\beta` is the router the forward used (the captured ``beta_pre``). Over query rows with at least two admissible sources
    (support of the attention log-prior, or :math:`\beta > 0` without one), per layer and head, the
    means of :math:`H(\beta_{i\cdot})/\log|S_i|`, :math:`\sum_j \beta_{ij}|i-j|`, :math:`\beta_{ii}` and
    :math:`\sum_j \min(\beta_{ij}, \pi_{ij})` with :math:`\pi = \mathrm{softmax}_j(B)` the prior alone.
    """
    block_norm = model.block_norm
    ln_defined = (block_norm is not None and model.head_mixer is None
                  and model.cg_coupling is None and _residual_update_is_additive(model)
                  and getattr(model.cfg, "block_coordinates", "local") == "local")
    ratios: Dict[int, List[torch.Tensor]] = {}
    ln_changes: Dict[int, List[torch.Tensor]] = {}
    sums: Dict[Tuple[int, int], List[float]] = {}          # (layer, head) -> [H_norm, dist, self, overlap, n]
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        for chunk in windows.split(max(1, DIAGNOSTIC_CHUNK_WINDOWS)):
            ids = chunk.to(device)                                              # (B, N)
            capture: dict = {"diagnostic": {}}
            model.forward_beliefs(ids, capture=capture)
            diagnostic = capture["diagnostic"]
            if counters_out is not None and "error" not in counters_out:
                try:
                    cap_counts, factorization_counts = _capture_numerics_counts(
                        diagnostic, float(model.cfg.kl_max))
                    _merge_numerics_counts(counters_out, cap_counts, factorization_counts)
                except Exception as exc:                  # noqa: BLE001 - reported by the caller
                    counters_out["error"] = exc
            n_layers = len(diagnostic["layer_converged"])
            for layer in range(n_layers):
                source = (diagnostic["initial_belief"] if layer == 0
                          else diagnostic["layer_outputs"][layer - 1])
                mu_in = source.mu.float()                                       # (B, N, K)
                converged = diagnostic["layer_converged"][layer]
                mu_out = converged.mu.float()                                   # (B, N, K) = mu + m
                message = mu_out - mu_in                                        # (B, N, K) m_i
                in_norm = torch.linalg.vector_norm(mu_in, dim=-1)               # (B, N)
                ratio = torch.linalg.vector_norm(message, dim=-1) / in_norm
                ratios.setdefault(layer, []).append(ratio[torch.isfinite(ratio)].reshape(-1).cpu())
                if ln_defined:
                    sigma = converged.sigma.float()
                    full = block_norm(mu_out, sigma).float()                    # (B, N, K) LN(mu + m)
                    removed = block_norm(message, sigma).float()                # (B, N, K) LN(m)
                    change = (torch.linalg.vector_norm(full - removed, dim=-1)
                              / torch.linalg.vector_norm(full, dim=-1))
                    ln_changes.setdefault(layer, []).append(
                        change[torch.isfinite(change)].reshape(-1).cpu())
            log_prior = diagnostic.get("log_prior")
            betas = [capture_l["beta_pre"].float()
                     for capture_l in diagnostic["layer_conditional_captures"]]
            for layer, beta in enumerate(betas):
                if beta.dim() == 3:
                    beta = beta.unsqueeze(1)                                    # (B, 1, N, N)
                n = beta.shape[-1]
                if isinstance(log_prior, torch.Tensor):
                    prior_logits = log_prior.float()
                    while prior_logits.dim() < 4:
                        prior_logits = prior_logits.unsqueeze(0)                # (1|B, 1|H, N, N)
                    support = torch.isfinite(prior_logits).expand_as(beta)
                    prior = torch.softmax(prior_logits, dim=-1).expand_as(beta)
                else:
                    support = beta > 0.0
                    prior = torch.full_like(beta, 1.0 / n)
                n_sources = support.sum(dim=-1)                                 # (B, H, N)
                rows = n_sources >= 2                                           # (B, H, N)
                entropy = -torch.special.xlogy(beta, beta).sum(dim=-1)          # (B, H, N)
                entropy_norm = entropy / torch.log(n_sources.clamp_min(2).to(beta.dtype))
                offsets = torch.arange(n, device=beta.device)
                distance = (beta * (offsets.view(-1, 1) - offsets.view(1, -1)).abs().to(beta.dtype)
                            ).sum(dim=-1)                                       # (B, H, N)
                self_weight = torch.diagonal(beta, dim1=-2, dim2=-1)            # (B, H, N)
                overlap = torch.minimum(beta, prior).sum(dim=-1)                # (B, H, N)
                weights = rows.to(torch.float64)
                per_head = torch.stack([
                    (entropy_norm.double() * weights).sum(dim=(0, 2)),
                    (distance.double() * weights).sum(dim=(0, 2)),
                    (self_weight.double() * weights).sum(dim=(0, 2)),
                    (overlap.double() * weights).sum(dim=(0, 2)),
                    weights.sum(dim=(0, 2)),
                ], dim=-1).cpu().tolist()                                       # (H, 5)
                for head, values in enumerate(per_head):
                    total = sums.setdefault((layer, head), [0.0] * 5)
                    for slot, value in enumerate(values):
                        total[slot] += value
    finally:
        for module, was_training in modes:
            module.training = was_training
    nan = float("nan")
    residual_rows: List[Dict[str, float]] = []
    for layer in sorted(ratios):
        values = torch.cat(ratios[layer]).double()
        q10, q50, q90 = (torch.quantile(values, torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)).tolist()
                         if values.numel() else (nan, nan, nan))
        changes = torch.cat(ln_changes[layer]).double() if layer in ln_changes else torch.empty(0)
        residual_rows.append({
            "step": int(step), "layer": int(layer), "q10": q10, "q50": q50, "q90": q90,
            "ln_rel_change_q50": (float(changes.median()) if changes.numel() else nan),
        })
    attention_rows: List[Dict[str, float]] = []
    for (layer, head), (h_sum, d_sum, s_sum, o_sum, count) in sorted(sums.items()):
        attention_rows.append({
            "step": int(step), "layer": int(layer), "head": int(head),
            "entropy_norm": h_sum / count if count else nan,
            "distance_mean": d_sum / count if count else nan,
            "self_weight": s_sum / count if count else nan,
            "prior_overlap": o_sum / count if count else nan,
            "n_rows": int(count),
        })
    return residual_rows, attention_rows


def _eval_cadence_diagnostics(
    model:        VFEModel,
    artifacts:    "RunArtifacts",
    logger:       logging.Logger,

    *,
    step:         int,
    val_ce:       float,
    train_sample: Optional[FixedTrainSample],
    windows:      Optional[torch.Tensor],
    device:       torch.device,
    failures:     Dict[str, int],
    count_numerics: bool = False,
) -> Dict[str, float]:
    r"""Run the eval-cadence additions for one evaluation; return its metrics columns.

    (1) CE of the fixed training sample under the validation protocol, and
    ``generalization_gap_fixed = val_ce - train_sample_ce`` (positive = the model fits its training
    windows better than held-out text); (2) ``diagnostics/residual_ratio.csv`` and
    ``diagnostics/attention_stats.csv`` rows on the fixed validation windows; (3) with
    ``count_numerics`` (projected routes), the S1 panel D per-evaluation counters of the same
    capture pass (``NUMERICS_EVAL_COLUMNS``, see :func:`_capture_numerics_counts`) and their
    per-(layer, head) rows in ``diagnostics/cap_activation.csv``; (4) the map snapshot
    ``map_snapshots/step_<N:07d>.pt``. Each part is isolated: a failure is counted and the others
    still run (the counters are NaN when their shared capture pass fails; that failure is counted
    once, as ``residual_attention``). CPU and CUDA RNG states are restored on exit.
    """
    out = {column: float("nan") for column in _EVAL_CADENCE_COLUMNS}
    rng_devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=rng_devices):
        if train_sample is not None:
            try:
                sample = evaluate(model, train_sample, device=device)
                out["train_sample_ce"] = float(sample["ce"])
                out["train_sample_ppl"] = float(sample["ppl"])
                out["generalization_gap_fixed"] = float(val_ce) - float(sample["ce"])
                artifacts.train_sample_by_step[int(step)] = float(sample["ce"])
            except Exception as exc:
                _record_diagnostic_failure(failures, "train_sample", exc, logger)
        if windows is not None:
            counters: Optional[Dict[str, Any]] = {} if count_numerics else None
            try:
                residual_rows, attention_rows = _residual_and_attention_rows(
                    model, windows, step, device, counters_out=counters)
            except Exception as exc:
                _record_diagnostic_failure(failures, "residual_attention", exc, logger)
                residual_rows = attention_rows = None
                counters = None                   # its capture pass failed; counted once above
            if residual_rows is not None:
                try:
                    artifacts.append_diagnostic_rows(
                        "residual_ratio.csv", RESIDUAL_RATIO_FIELDS, residual_rows)
                    artifacts.append_diagnostic_rows(
                        "attention_stats.csv", ATTENTION_STATS_FIELDS, attention_rows)
                except Exception as exc:
                    _record_diagnostic_failure(failures, "residual_attention", exc, logger)
            if counters is not None:
                try:
                    columns, cap_rows = _numerics_eval_columns(counters, step)
                    out.update(columns)
                    artifacts.append_diagnostic_rows(
                        "cap_activation.csv", CAP_ACTIVATION_FIELDS, cap_rows)
                except Exception as exc:
                    _record_diagnostic_failure(failures, "numerics_counters", exc, logger)
        try:
            state = _map_snapshot_state(model)
            if state:
                artifacts.save_map_snapshot(step, state)
        except Exception as exc:
            _record_diagnostic_failure(failures, "map_snapshot", exc, logger)
    return out


def _optimizer_state_bytes(
    optimizer: torch.optim.Optimizer,
) -> Dict[str, int]:
    r"""Bytes held by optimizer state tensors (AdamW moments, step counters), total and by dtype."""
    total = 0
    by_dtype: Dict[str, int] = {}
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                size = int(value.numel()) * int(value.element_size())
                total += size
                by_dtype[str(value.dtype)] = by_dtype.get(str(value.dtype), 0) + size
    return {"total": total, **{f"dtype:{key}": value for key, value in sorted(by_dtype.items())}}


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
    failures:   Optional[Dict[str, int]] = None,
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
        return _val_diagnostics_one(model, next(iter(val_loader)), device, failures=failures)

    first: Optional[ValidationDiagnostics] = None
    sums:   Dict[str, float] = {}
    counts: Dict[str, int]   = {}
    for index, batch in enumerate(val_loader):
        if index >= batches:
            break
        one = _val_diagnostics_one(model, batch, device, failures=failures)
        for key, value in one.metrics.items():
            if math.isfinite(value):
                sums[key]   = sums.get(key, 0.0) + value
                counts[key] = counts.get(key, 0) + 1
        if first is None:
            first = one
        else:
            del one                      # release this batch's snapshot (dense logits) immediately
    if first is None:                    # empty loader: preserve the old failure mode
        return _val_diagnostics_one(model, next(iter(val_loader)), device, failures=failures)
    averaged = {
        key: (sums[key] / counts[key]) if counts.get(key) else float("nan")
        for key in first.metrics
    }
    return ValidationDiagnostics(averaged, first.token_ids, first.snapshot)


_SNAPSHOT_FRAME_METHODS = {
    name: getattr(VFEModel, name) for name in (
        "build_diagnostic_snapshot", "forward_beliefs", "diagnostics",
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

    if (type(model) is not VFEModel
            or get_family(model.cfg.family) is not DiagonalGaussian
            or model.group.elementary_block_shape is None
            or get_transport_registration("flat").callable is not _build_flat
            or (model.block_mlps is not None and any(
                type(mlp) is not BlockMLP or mlp.covariance_contract != "passthrough"
                or getattr(mlp.forward_moments, "__func__", None) is not BlockMLP.forward_moments
                for mlp in model.block_mlps))
            or any(getattr(getattr(model, name), "__func__", None) is not method
                   for name, method in _SNAPSHOT_FRAME_METHODS.items())
            or ex.e_step_belief_trace is not _snapshot_trace
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
         decode_logdet_fallback_elements(), frame_pair_fp64_escalations()),
    )


def _attention_structure_metrics(
    amaps: torch.Tensor,                      # (L, H, N, N) all layers/heads, one sequence
) -> Dict[str, float]:
    r"""Held-out attention-map structure: entropy collapse, causal sanity, positional/content, redundancy.

    Shared by both routes of :func:`_val_diagnostics_one`; the legacy route passes the entry-scored
    replay, the conditional route the actual pre-update router ``snapshot.beta_maps``.
    """
    from vfe3 import metrics as M

    out: Dict[str, float] = {}
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
    return out


def _position_loss_metrics(
    logits:  torch.Tensor,                    # (B, N, V) captured held-out logits
    targets: torch.Tensor,                    # (B, N) next-token ids (-100 = ignore)
) -> Dict[str, float]:
    r"""Per-position within-sequence loss: mean CE over the first and last quarter of positions."""
    import torch.nn.functional as F

    out: Dict[str, float] = {}
    b, n = targets.shape
    per = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1),
                          ignore_index=-100, reduction="none").reshape(b, n)
    valid = (targets != -100).float()
    pos_ce = (per * valid).sum(0) / valid.sum(0).clamp(min=1.0)   # (N,) mean CE at each position
    q = n // 4
    if q > 0:
        out["pos_loss_first_q"] = float(pos_ce[:q].mean())
        out["pos_loss_last_q"]  = float(pos_ce[-q:].mean())
        out["pos_loss_ratio"]   = float(pos_ce[-q:].mean() / pos_ce[:q].mean().clamp(min=1e-9))
    return out


@torch.no_grad()
def _val_diagnostics_one(
    model:      VFEModel,
    batch:      object,
    device:     torch.device,

    *,
    failures:   Optional[Dict[str, int]] = None,
) -> ValidationDiagnostics:
    r"""Held-out per-eval probes (the train-loop ``diagnostics`` runs on the live TRAIN batch).

    Computes the relational validation diagnostics, attention-map structure across ALL layers/heads
    (entropy collapse, causal-mask sanity, positional-vs-content, induction/copy, head redundancy),
    and the per-position within-sequence loss. Off the graph (no_grad). Best-effort: the CALLER wraps this so a replay
    error simply leaves the previous values carried forward; the returned subset is ``.update``-d
    into a NaN-initialized dict, so the CSV stays rectangular.

    Conditional (relational) route: the attention-structure columns (``val_attn_entropy_min_all``,
    ``val_attn_entropy_min_norm``, ``val_attn_collapsed_heads``, ``val_future_leakage``,
    ``val_row_sum_error``, ``val_pos_content_r2``, ``val_prev_token_mass``,
    ``val_period_match_mass``, ``val_head_redundancy_js``) are computed on the actual pre-update
    router ``snapshot.beta_maps`` of the same capture, the ``pos_loss_*`` columns on its logits, and
    ``val_effective_rank`` / ``val_belief_cond_median`` from the last layer's converged covariance
    (as the relational diagnostics report them). The following stay NaN on this route, by definition
    rather than by omission: the legacy free-energy decomposition (``val_self_coupling``,
    ``val_self_divergence``, ``val_belief_coupling``, ``val_attention_entropy``,
    ``val_inner_alignment_energy_total``), because routing plus readout does not define the shared
    stationary objective; the E-step certificate (``estep_*`` except ``estep_f_comparisons = 0``),
    because the route performs one pre-update readout with no iterate sequence; and the held-out
    geometry/guard columns (``val_holonomy_wilson``, ``val_cocycle_residual``,
    ``val_gauge_invariant_spread``, ``val_fisher_trace_mean``, ``val_belief_cond_p95``,
    ``val_belief_lam_*``, ``val_belief_cond_valid_frac``, ``val_phi_*``, ``val_vertex_cond_p99``,
    ``val_pos_phi_*``, ``val_guard_*``, ``val_live_pair_frac``, ``val_nonfinite_frac``), which come from
    the legacy ``diagnostics`` geometry pass that the relational diagnostics do not run. The
    head-mixer ``val_builder_resid`` stays legacy-route only.
    """
    import torch.nn.functional as F

    out: Dict[str, float] = {}
    val_tok, val_tgt = (batch if isinstance(batch, (tuple, list)) else (batch, None))
    diagnostic_tokens = val_tok[:1].to(device)
    diagnostic_targets = val_tgt[:1].to(device) if val_tgt is not None else None
    # This private validation lifetime only reads frames; public builders retain independent copies.
    snapshot = model.build_diagnostic_snapshot(
        diagnostic_tokens,
        **({"_share_frame_copies": True} if _can_share_snapshot_frames(model) else {}))

    vd = model.diagnostics(                                     # held-out F decomposition (per token)
        diagnostic_tokens,
        snapshot=snapshot,
    )
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
    # Definitions that carry over from the legacy route, on the same capture (audit
    # 2026-09-28: the early return left these NaN). Isolated so a failure here keeps the
    # relational columns above; the whole set is NaN-reset first so a failed evaluation can
    # never carry a previous evaluation's value forward.
    out.update({key: float("nan") for key in _CONDITIONAL_STRUCTURE_KEYS})
    try:
        structure: Dict[str, float] = {
            "val_effective_rank": float(vd["effective_rank"]),
            "val_belief_cond_median": float(vd["belief_cond_median"]),
        }
        structure.update(_attention_structure_metrics(snapshot.beta_maps.float()))
        if diagnostic_targets is not None:
            structure.update(_position_loss_metrics(snapshot.logits, diagnostic_targets))
        out.update(structure)
    except Exception as exc:
        if failures is not None:
            _record_diagnostic_failure(failures, "conditional_val_structure", exc,
                                       logging.getLogger(__name__))
        else:
            logging.getLogger(__name__).warning(
                "conditional-route attention-structure diagnostics failed (%s)", exc)
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
        from vfe3.viz.figures import route_figure_skip_reasons

        skip = route_figure_skip_reasons(model.cfg).get("attention_estep")
    except Exception:                                        # noqa: BLE001 - no gate: replay as before
        skip = None
    if skip is not None:
        logger.info("attention_estep skipped (%s)", skip)
        return
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
    scaler, and data cursor, plus host (CPU) copies of the last-iterate ``state_dict`` and clones of
    the CPU/CUDA global RNG states. A validation-only finalizer scores validation,
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
            getattr(model, "prior_bank", model).set_unigram_log_prior(_counts.to(device))
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
    # Early evaluations (audit 2026-09-28): log-spaced completed-step counts below the first regular
    # evaluation, gated by EARLY_EVAL_MIN_INTERVAL and a run directory. Empty on the silent path.
    early_eval_enabled = bool(
        eval_interval and val_loader is not None and artifacts is not None
        and eval_interval >= EARLY_EVAL_MIN_INTERVAL)
    early_eval_steps = frozenset(
        int(s) for s in EARLY_EVAL_STEPS
        if early_eval_enabled and 0 < int(s) < eval_interval and start_step < int(s) <= n_steps)
    run_step_zero = bool(early_eval_enabled and 0 in EARLY_EVAL_STEPS
                         and start_step == 0 and n_steps > 0)
    # A timer is useful only if its absolute execution interval reaches a reporting step.
    timing_enabled = (
        bool(log_interval) and n_steps // log_interval > start_step // log_interval
    ) or (
        artifacts is not None and val_loader is not None and bool(eval_interval)
        and n_steps // eval_interval > start_step // eval_interval
    ) or bool(early_eval_steps)
    resumed_wall_clock = getattr(artifacts, "resumed_wall_clock_s", None) if start_step > 0 else None
    timer = (TrainingTimer(device, wall_clock_offset=resumed_wall_clock or 0.0)
             if timing_enabled else None)
    last_val: Dict[str, Optional[float]] = {}        # most recent validation, carried into each CSV row
    # VFE-only diagnostics (held-out probes, residual/attention capture, frame-clamp warning) are
    # skipped for another model family; their columns keep the VFE schema and stay blank.
    vfe_model = is_vfe_model(model)
    val_diag_keys = _VAL_DIAG_KEYS + (
        _VAL_RELATIONAL_DIAG_KEYS if vfe_model else ())
    last_val_diag: Dict[str, float] = {k: float("nan") for k in val_diag_keys}
    # Training-time diagnostic state (audit 2026-09-28). Built only for a persisted run; the silent
    # path and the console-only path allocate nothing here.
    diagnostics_failures: Dict[str, int] = {}
    fixed_train_sample: Optional[FixedTrainSample] = None
    diagnostic_windows: Optional[torch.Tensor] = None
    last_train_sample: Dict[str, float] = {column: float("nan") for column in _EVAL_CADENCE_COLUMNS}
    pending_step_zero_row: Optional[Dict[str, float]] = None
    tokens_seen = int(start_step) * int(cfg.batch_size) * int(cfg.max_seq_len)   # resume: nominal B*N per step
    # S1 panel D counters (2026-09-28). The frame-projection counter is a (2,) int64 device tensor
    # that project_phi_parameter_rows_ adds to on every accepted step (no host sync); it exists only
    # for a persisted run whose config has the post-update projection. frame_projection_base holds
    # the counts a resumed checkpoint carried; frame_projection_since is the step they start from.
    numerics_reasons: Dict[str, Optional[str]] = {}
    frame_projection_counter: Optional[torch.Tensor] = None
    frame_projection_base: Tuple[int, int] = (0, 0)
    frame_projection_since: Optional[int] = None
    numerics_last_eval: Dict[str, Any] = {}
    if artifacts is not None:
        other_parameters = [name for name, parameter in model.named_parameters()
                            if parameter.requires_grad and _parameter_group(name) == "other"]
        if other_parameters and (log_interval or eval_interval):
            logger.info(" parameter group 'other' (grad/param/update_ratio_group_other) holds: %s",
                        ", ".join(other_parameters))
        existing_failures = getattr(artifacts, "diagnostics_failures", None)
        if isinstance(existing_failures, dict):
            diagnostics_failures = existing_failures
        else:
            artifacts.diagnostics_failures = diagnostics_failures
        if not isinstance(getattr(artifacts, "train_sample_by_step", None), dict):
            artifacts.train_sample_by_step = {}
        try:
            numerics_reasons = _numerics_counter_skip_reasons(model)
            for family, reason in sorted(numerics_reasons.items()):
                if reason is not None:
                    logger.info(" S1 numerics counter %r not measured: %s", family, reason)
            if numerics_reasons["frame_projection"] is None:
                frame_projection_counter = torch.zeros(
                    2, dtype=torch.long, device=model.group.generator_device)
                resumed = getattr(artifacts, "resumed_numerics_counters", None)
                if start_step == 0:
                    frame_projection_since = 0
                elif (isinstance(resumed, Mapping)
                      and isinstance(resumed.get("frame_projection_count"), int)
                      and isinstance(resumed.get("frame_projection_total"), int)):
                    frame_projection_base = (int(resumed["frame_projection_count"]),
                                             int(resumed["frame_projection_total"]))
                    since = resumed.get("frame_projection_since_step")
                    frame_projection_since = int(since) if isinstance(since, int) else 0
                else:
                    frame_projection_since = int(start_step)
                    logger.info(" resumed checkpoint carries no frame-projection counts; "
                                "frame_projection_* count from step %d", start_step)
        except Exception as exc:
            _record_diagnostic_failure(diagnostics_failures, "numerics_counter_setup", exc, logger)
            frame_projection_counter = None
        if eval_interval and val_loader is not None:
            with torch.random.fork_rng(devices=([device] if device.type == "cuda" else [])):
                try:
                    fixed_train_sample, sample_reason = _build_fixed_train_sample(
                        loader, val_loader, cfg)
                    if fixed_train_sample is None:
                        logger.info(" fixed training sample unavailable (%s); train_sample_* "
                                    "columns stay blank", sample_reason)
                    else:
                        logger.info(
                            " fixed training sample: %d windows x %d scored targets "
                            "(seed %d, eval protocol seq_len=%d stride=%d)",
                            len(fixed_train_sample.indices),
                            fixed_train_sample.scored_targets_per_window, TRAIN_SAMPLE_SEED,
                            fixed_train_sample.seq_len, fixed_train_sample.stride)
                except Exception as exc:
                    _record_diagnostic_failure(diagnostics_failures, "train_sample_build", exc, logger)
                if vfe_model:
                    try:
                        diagnostic_windows = _diagnostic_windows(val_loader)
                    except Exception as exc:
                        _record_diagnostic_failure(diagnostics_failures, "diagnostic_windows", exc,
                                                   logger)
                else:
                    logger.info(" residual-ratio and attention-statistics captures skipped "
                                "(model_family=%r: they read the VFE belief capture)",
                                model_family_of(model))
    if run_step_zero:
        # Step-0 evaluation before the first update (M3 curves need a beginning). Everything here
        # runs under a forked CPU/CUDA RNG, so the training stream is untouched. Its row is held
        # until the first regular metrics row fixes the CSV schema, then written first.
        with torch.random.fork_rng(devices=([device] if device.type == "cuda" else [])):
            try:
                zero = evaluate(model, val_loader, tokens_per_char=tokens_per_char, device=device)
                logger.info(" \n Validation @ step 0 (before the first update): CE %.4f | PPL %.1f",
                            zero["ce"], zero["ppl"])
                zero_diag: Dict[str, float] = {k: float("nan") for k in val_diag_keys}
                try:
                    if vfe_model:
                        zero_diag.update(_val_diagnostics(
                            model, val_loader, device,
                            batches=getattr(cfg, "val_diagnostic_batches", 1),
                            failures=diagnostics_failures).metrics)
                except Exception as exc:
                    _record_diagnostic_failure(diagnostics_failures, "step0_val_diagnostics",
                                               exc, logger)
                zero_extra = _eval_cadence_diagnostics(
                    model, artifacts, logger, step=0, val_ce=float(zero["ce"]),
                    train_sample=fixed_train_sample, windows=diagnostic_windows,
                    device=device, failures=diagnostics_failures,
                    count_numerics=numerics_reasons.get("cap_activation", "unset") is None)
                _note_numerics_eval(numerics_last_eval, 0, zero_extra)
                zero_cumulative = {column: float("nan") for column in NUMERICS_CUMULATIVE_COLUMNS}
                try:
                    zero_cumulative = _cumulative_counter_columns(
                        0, optimizer, frame_projection_counter, frame_projection_base)
                except Exception as exc:
                    _record_diagnostic_failure(diagnostics_failures, "numerics_cumulative",
                                               exc, logger)
                pending_step_zero_row = {
                    **zero_cumulative,
                    "step": 0,
                    "val_ce": zero["ce"], "val_ppl": zero["ppl"],
                    "val_bits_per_token": zero["bits_per_token"],
                    "val_bpc": (zero["bpc"] if zero["bpc"] is not None else float("nan")),
                    "val_expected_targets": zero["expected_targets"],
                    "val_scored_targets": zero["scored_targets"],
                    "val_excluded_targets": zero["excluded_targets"],
                    **zero_diag, **zero_extra,
                    **_group_parameter_norms(model), **_sigma_log_statistics(model),
                    "tokens_seen": 0,
                    "diagnostics_failures": int(sum(diagnostics_failures.values())),
                }
            except Exception as exc:
                _record_diagnostic_failure(diagnostics_failures, "step0_evaluation", exc, logger)
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
        tokens_seen += int(tokens.numel())                   # input-token presentations (metadata only)
        tokens = tokens.to(device, non_blocking=True)
        targets = _transfer_targets(targets, model, device)
        do_log  = bool(log_interval) and (step + 1) % log_interval == 0
        do_eval = bool(eval_interval) and val_loader is not None and (
            (step + 1) % eval_interval == 0 or (step + 1) in early_eval_steps)
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
            if vfe_model:
                _warn_phi_transport_clamp(model)
            # m7: diagnostics on the PRE-step weights so the logged F-decomposition shares the pre-step
            # provenance of train_loss/train_ce (train_ce is captured pre-step inside train_step). Under
            # no_grad, so no weight change,
            # leaving the train_step RNG stream and weights byte-identical.
            try:
                d = model.diagnostics(tokens)
            except Exception as exc:                         # optional probes cannot stop training
                _record_diagnostic_failure(diagnostics_failures, "training_diagnostics", exc, logger)
                d = {key: float("nan") for key in (
                    "attn_entropy", "self_coupling", "self_divergence", "belief_coupling",
                    "attention_entropy", "total", "effective_rank", "holonomy_deviation",
                    "gauge_trace_spread",
                )}
        # generalization_gap's train side under the validation protocol when that protocol is the
        # sliding window (eval_stride < max_seq_len); pre-step, like train_ce. None keeps train_ce.
        gap_train_ce: Optional[float] = None
        if (do_eval and do_csv and cfg.eval_stride is not None
                and cfg.eval_stride < cfg.max_seq_len):
            gap_train_ce = _eval_protocol_train_ce(model, tokens, targets, cfg.eval_stride)
        # Named-group update ratios (audit 2026-09-28): host copies of the pre-step parameters on a
        # persisted log/eval row only, outside the timed train step. Reading parameters does not
        # change the step.
        update_before: Optional[Dict[str, torch.Tensor]] = None
        group_stats: Dict[str, float] = {}
        if do_csv:
            try:
                update_before = _parameter_snapshot(model)
            except Exception as exc:
                _record_diagnostic_failure(diagnostics_failures, "parameter_snapshot", exc, logger)
        step_status: Dict[str, bool] = {}
        if timer is not None:
            timer.start_step()
        losses.append(train_step(model, optimizer, scheduler, tokens, targets,
                                  grad_clip=grad_clip, grad_accum_steps=cfg.grad_accum_steps,
                                  scaler=scaler, metrics_out=step_metrics, status_out=step_status,
                                  **({"frame_projection_out": frame_projection_counter}
                                     if frame_projection_counter is not None else {})))
        if timer is not None:
            timer.finish_step(n_tokens=tokens.numel())
        if update_before is not None:
            try:
                group_stats = _group_update_statistics(model, update_before)
            except Exception as exc:
                _record_diagnostic_failure(diagnostics_failures, "group_update_statistics", exc, logger)
            update_before = None                              # release the host copy immediately
        if step_metrics is not None and _GROUP_GRAD_ERROR_KEY in step_metrics:
            _record_diagnostic_failure(diagnostics_failures, "group_gradient_norms",
                                       RuntimeError(step_metrics.pop(_GROUP_GRAD_ERROR_KEY)), logger)
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
                    if vfe_model:
                        validation_diagnostics = _val_diagnostics(
                            model, val_loader, device,
                            batches=getattr(cfg, "val_diagnostic_batches", 1),
                            failures=diagnostics_failures)
                        last_val_diag.update(validation_diagnostics.metrics)
                except Exception as exc:
                    terminal_validation_signature = None
                    logger.warning("       (validation diagnostics failed: %s); continuing", exc)
                    _record_diagnostic_failure(diagnostics_failures, "val_diagnostics", exc, logger)
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
                # Fixed training-sample CE, residual/attention CSVs and the map snapshot (audit
                # 2026-09-28). Each part is exception-isolated and counted; RNG is forked inside.
                try:
                    last_train_sample = _eval_cadence_diagnostics(
                        model, artifacts, logger, step=step + 1, val_ce=float(m["ce"]),
                        train_sample=fixed_train_sample, windows=diagnostic_windows,
                        device=device, failures=diagnostics_failures,
                        count_numerics=numerics_reasons.get("cap_activation", "unset") is None)
                    _note_numerics_eval(numerics_last_eval, step + 1, last_train_sample)
                except Exception as exc:
                    _record_diagnostic_failure(diagnostics_failures, "eval_cadence", exc, logger)
                    last_train_sample = {column: float("nan") for column in _EVAL_CADENCE_COLUMNS}
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
            # The cumulative frame-projection counts travel with the checkpoint (one host read at
            # checkpoint cadence), so a resumed run keeps counting from step 0.
            try:
                artifacts.numerics_counters_state = _frame_projection_state(
                    frame_projection_counter, frame_projection_base, frame_projection_since)
            except Exception as exc:
                _record_diagnostic_failure(diagnostics_failures, "numerics_checkpoint_state",
                                           exc, logger)
                artifacts.numerics_counters_state = None
            artifacts.wall_clock_state = timer.elapsed_wall_clock_s() if timer is not None else None
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
            lrs = (_learning_rates_by_role(optimizer.param_groups, raw_lrs) if vfe_model
                   else _learning_rates_by_role(optimizer.param_groups, raw_lrs, required=("mu",)))
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
                "lambda_beta":        float(getattr(cfg, "lambda_beta", float("nan"))),
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
            # generalization_gap is kept for continuity with earlier runs, but it is a ONE-BATCH
            # estimate (the current training batch), so its eval-to-eval noise is of the order of
            # the gap itself. generalization_gap_fixed (below) replaces it for analysis: validation
            # CE minus the CE of a FIXED training sample of validation size, scored under the same
            # protocol at every eval (FixedTrainSample; audit 2026-09-28 M5).
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
            if getattr(cfg, "phi_mstep_max_matrix_norm", None) is not None:
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
                    # group_log adds its own fixed key set; the chart route's columns are unchanged.
                    if getattr(optimizer, "_phi_group_retraction", "chart") == "group_log":
                        for _pk in GROUP_LOG_DIAGNOSTIC_KEYS:
                            row[_pk] = _gd.get(_pk, float("nan"))
                    # fisher_group adds the belief-Fisher key set.
                    if getattr(optimizer, "phi_statistics_provider", None) is not None:
                        for _pk in FISHER_DIAGNOSTIC_KEYS:
                            row[_pk] = _gd.get(_pk, float("nan"))
            # Held-out per-eval probes (Tier-2): the full fixed val-diag column set. Eval-cadence like
            # val_ce above -- the fresh probe values on an eval row, NaN (rendered blank) on the
            # log-interval rows in between, NOT carried forward. The key set is identical in both
            # branches (val_diag_keys), so the CSV stays rectangular.
            row.update(last_val_diag if do_eval else {k: float("nan") for k in val_diag_keys})
            # Training-time diagnostics (audit 2026-09-28). Fixed key set on every row: the
            # train-sample columns are eval-cadence (NaN between evals); the named-group grad norms
            # come from train_step, the parameter norms and realized update ratios from the host
            # snapshot around this step; NaN wherever a probe failed or a group is empty.
            # The S1 panel D per-evaluation counters ride in the same eval-cadence set.
            row.update(last_train_sample if do_eval
                       else {column: float("nan") for column in _EVAL_CADENCE_COLUMNS})
            group_values = {**(step_metrics or {}), **group_stats}
            for column in _GROUP_COLUMNS:
                row[column] = float(group_values.get(column, float("nan")))
            try:
                row.update(_sigma_log_statistics(model))
            except Exception as exc:
                _record_diagnostic_failure(diagnostics_failures, "sigma_log_statistics", exc, logger)
                row.update({"sigma_log_mean": float("nan"), "sigma_log_sd": float("nan")})
            # S1 panel D cumulative counters on every row (one host read of the device counter).
            try:
                row.update(_cumulative_counter_columns(
                    step + 1, optimizer, frame_projection_counter, frame_projection_base))
            except Exception as exc:
                _record_diagnostic_failure(diagnostics_failures, "numerics_cumulative", exc, logger)
                row.update({column: float("nan") for column in NUMERICS_CUMULATIVE_COLUMNS})
            row["tokens_seen"] = tokens_seen
            row["diagnostics_failures"] = int(sum(diagnostics_failures.values()))
            pipeline_timing = timer.sample_pipeline_window()
            row["train_step_ms_mean"]      = train_timing.train_step_ms_mean
            row["train_step_tokens_per_s"] = train_timing.train_step_tokens_per_s
            row["pipeline_tokens_per_s"]   = pipeline_timing.pipeline_tokens_per_s
            row["tokens_per_s"]            = pipeline_timing.pipeline_tokens_per_s
            row["wall_clock_s"]            = pipeline_timing.wall_clock_s
            if pending_step_zero_row is not None:
                # The step-0 evaluation row goes first, on this row's key set (NaN where step 0 has
                # no value: no training step, no timing window).
                zero_row = {key: pending_step_zero_row.get(key, float("nan")) for key in row}
                pending_step_zero_row = None
                try:
                    artifacts.log_metrics(zero_row)
                except Exception as exc:
                    _record_diagnostic_failure(diagnostics_failures, "step0_row", exc, logger)
            artifacts.log_metrics(row)
            timer.reset_pipeline_window()
    # S1 panel D handoff (2026-09-28), before the terminal callback so its checkpoint and summary see
    # it: the cumulative frame-projection counts for any later checkpoint, and summary.json's
    # "numerics_counters" (final cumulative counts, last measured per-evaluation counts, skip reasons).
    if artifacts is not None:
        try:
            artifacts.numerics_counters_state = _frame_projection_state(
                frame_projection_counter, frame_projection_base, frame_projection_since)
            artifacts.numerics_counters = _numerics_counters_summary(
                n_steps, optimizer, artifacts.numerics_counters_state, numerics_last_eval,
                numerics_reasons, diagnostic_windows)
        except Exception as exc:
            _record_diagnostic_failure(diagnostics_failures, "numerics_summary", exc, logger)
    # Terminal callback seam (PB-02): immediately after the final optimizer step, hand the resumable
    # training state to an optional callback so a
    # default cell (log/eval interval above max_steps, checkpoint_interval=0) can finalize a complete
    # artifact set -- validation eval, best weights, summary, and a resumable terminal checkpoint -- in
    # one opt-in operation. The model-state and CPU/CUDA RNG clones let the finalizer restore the
    # final training iterate after validation. terminal_callback=None keeps the pure path free of
    # those snapshots. The model-state copy goes to HOST memory: a device clone of every table while
    # the parameters, gradients and AdamW moments are still resident doubles the parameter footprint
    # at the end of the run (a likely OOM for vocabulary-sized tables); load_state_dict copies it back.
    if terminal_callback is not None:
        raw_model_state = {name: tensor.detach().to("cpu", copy=True)
                           for name, tensor in model.state_dict().items()}
        rng_state: Dict[str, object] = {
            "cpu":  torch.get_rng_state().clone(),
            "cuda": ([s.clone() for s in torch.cuda.get_rng_state_all()]
                     if torch.cuda.device_count() > 0 else None),
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
        # Training-time diagnostic handoff to finalize_run (audit 2026-09-28): a step-0 row that no
        # later metrics row flushed (no log/eval row in the run), the optimizer-state footprint for
        # resources.json, and the per-name diagnostic failure counts for summary.json.
        if pending_step_zero_row is not None:
            try:
                artifacts.log_metrics(pending_step_zero_row)
            except Exception as exc:
                _record_diagnostic_failure(diagnostics_failures, "step0_row", exc, logger)
            pending_step_zero_row = None
        try:
            artifacts.optimizer_state_bytes = _optimizer_state_bytes(optimizer)
        except Exception as exc:
            _record_diagnostic_failure(diagnostics_failures, "optimizer_state_bytes", exc, logger)
        artifacts.diagnostics_failures = diagnostics_failures
        if diagnostics_failures:
            logger.warning(" diagnostic failures during training (counted, training unaffected): %s",
                           dict(sorted(diagnostics_failures.items())))
        neutralized = nonfinite_tangent_elements()
        mu_fallbacks = mu_trust_fallback_elements()
        artifacts.final_step = int(n_steps)
        artifacts.realized_updates = {
            "attempted_steps":      attempted,
            "accepted_updates":     int(accepted),
            "skipped_steps":        int(skipped_total),
            "accepted_update_frac": (float(accepted) / attempted
                                     if attempted else float("nan")),
            "nonfinite_tangent_elements": int(neutralized),
            "mu_trust_fallback_elements": int(mu_fallbacks),
            "decode_logdet_fallback_elements": int(decode_logdet_fallback_elements()),
            "frame_pair_fp64_escalations": int(frame_pair_fp64_escalations()),
        }
        if skipped_total or neutralized or mu_fallbacks:
            logger.warning(
                " realized budget: %d/%d optimizer updates accepted (%.3f%%); "
                "%d SPD tangent element(s) neutralized as non-finite; "
                "%d belief(s) whitened by the non-equivariant mu-trust fallback",
                accepted, attempted,
                100.0 * accepted / max(attempted, 1),
                neutralized, mu_fallbacks)
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


def _fmt_decode_lrs(cfg: VFE3Config) -> str:
    """Show realized decoder rates, including inactive variance and shared-table cases."""
    mu_lr, sigma_lr = _decode_learning_rates(cfg)
    if cfg.decode_mode == "linear":
        return f"mu={mu_lr}  sigma=inactive"
    if not cfg.untie_decode_bank:
        return f"mu={cfg.m_p_mu_lr} (shared)  sigma={cfg.m_p_sigma_lr} (shared)"
    return f"mu={mu_lr}  sigma={sigma_lr}"


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
    (``decode_log_scale`` under ``decode_mode='linear'``, ...) are counted as
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
        f" decoder M-LRs: {_fmt_decode_lrs(cfg)}",
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
    device = torch.device("cuda" if torch.cuda.device_count() > 0 else "cpu")
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
