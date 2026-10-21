r"""Training (M-step) for VFE_3.0: AdamW per-group learning rates + warmup/cosine.

The trainable parameters are the PriorBank prior tables (means, variances, frames), the learned
comparison and content maps of every attention layer (4K^2 parameters per layer on the kernel
readout, the count of dense W_Q, W_K, W_V, W_O; 2K^2 + H on the minimal agent), the default linear decoder (``decode_mode='linear'``), the default
coordinate block MLP (``block_mlp_mode='coordinate'``: nn.Linear fc1/fc2 with a pointwise
activation), the default ``pos_phi='learned'`` positional table, and the tables of opt-in toggles
(CG coupling, LayerNorm affine). These are learned linear and neural maps trained by
backpropagation. No nn.Linear module is built only with ``use_block_mlp=False`` or a
``prototype_agents`` / ``gaussian_template`` block MLP together with a prior-bank decoder, and the
attention maps remain learned linear maps there. The M-step minimizes the decode cross-entropy;
``projected_predictive`` takes one differentiable readout step per layer, and ``projected_gradient``
unrolls its ``n_e_steps`` receiver updates into the graph. Click-to-run: edit a ``VFE3Config`` and call ``run_training`` (no CLI).
"""

import collections
import contextlib
import itertools
import json
import logging
import math
import time
import traceback
from numbers import Real
from pathlib import Path
from typing import (Any, Callable, Collection, Deque, Dict, Iterable, Iterator, List, Mapping, NamedTuple, Optional,
                    Sequence, Set, Tuple)

import torch

try:                                                # live per-step it/s via a tqdm progress bar
    from tqdm import tqdm as _tqdm                  # (plain tqdm, not tqdm.auto: the notebook
    from tqdm.contrib.logging import (              # widget is swallowed by some Run-button
        logging_redirect_tqdm as _redirect_logging,  # consumers / non-TTY stdout)
    )
except ImportError:                                 # tqdm optional: absent -> no bar, the periodic
    _tqdm = None                                    # log lines still emit at log_interval as before
    _redirect_logging = contextlib.nullcontext

from vfe3.attention_prior import get_prior_registration
from vfe3.contracts import deferred_host_checks, is_vfe_model, model_family_of
from vfe3.config import VFE3Config
from vfe3.contracts import DataState, DataStateBuffer
from vfe3.data.datasets import make_dataloader
from vfe3.free_energy import attention_tau
from vfe3.gauge_optim import (
    FISHER_DIAGNOSTIC_KEYS,
    FRAME_CONDITION_GROUPS,
    GROUP_LOG_DIAGNOSTIC_KEYS,
    GaugeManifoldAdamW,
    bound_phi_frame_condition_,
    get_phi_update_policy,
    group_log_block_shape,
    project_phi_parameter_rows_,
)
from vfe3.geometry.phi_fisher import FisherRowStatistics, middle_fiber_moments
from vfe3.table_optim import (TABLE_FISHER_DIAGNOSTIC_KEYS, TableFisherAdamW, TableFisherGaugeManifoldAdamW,
                              TableFisherStatistics)
from vfe3.metric_contracts import perplexity_from_ce
from vfe3.metrics import COLLAPSE_ENTROPY_FRACTION, LINK_SPECTRUM_KEYS, routing_triangle_defect_sums
from vfe3.model.block import _as_coeff
from vfe3.model.block_mlp import PrototypeAgentBlockMLP
from vfe3.model.minimal_agent import MINIMAL_AGENT_ELBO_KEYS, MINIMAL_AGENT_IS_SAMPLES
from vfe3.model.model import CANONICAL_DIAGNOSTIC_KEYS, DiagnosticSnapshot, VFEModel
from vfe3.model.prior_bank import PriorBank, get_decode_registration
from vfe3.run_artifacts import RunArtifacts          # top-level safe: run_artifacts imports evaluate
#                                                      lazily (function-local), so there is no cycle
from vfe3.runtime import disable_fill_uninitialized_memory, seed_everything
from vfe3.timing import TrainingTimer
from vfe3.viz.extract import e_step_belief_trace as _snapshot_trace
from vfe3.geometry.retraction import (                          # SPD-retraction non-finite tangent guard
    nonfinite_tangent_elements,
    public_spd_ceiling_contact_elements,
    public_spd_repair_elements,
    reset_nonfinite_tangent_elements,
    reset_public_spd_repair_elements,
)
from vfe3.numerics import (                    # non-equivariant / non-PD fallback accounting
    decode_logdet_fallback_elements,
    mu_trust_fallback_elements,
    reset_decode_logdet_fallback_elements,
    reset_mu_trust_fallback_elements,
    bounded_variance_from_log,
)
from vfe3.relational_scoring import (                                # projected float64 source-factor recovery count
    gemm_fallback_slices, reset_gemm_fallback_slices,                   # ... and pair-KL GEMM fallback slices
    reset_source_recovery_elements, source_recovery_elements)
from vfe3.geometry.transport import (                           # float64 frame-pair escalation count
    frame_pair_fp64_escalations,
    reset_frame_pair_fp64_escalations,
)
from vfe3.model.block_mlp import (                              # LayerNorm pushforward sphere-bound caps
    layernorm_trace_cap_elements, reset_layernorm_trace_cap_elements)
from vfe3.model.covariance_ceiling import (                     # moment-matched covariance ceiling clamps
    covariance_ceiling_elements, reset_covariance_ceiling_elements)
from vfe3.model.relational_operators import (                   # moment-matched covariance fp64 escalations
    covariance_escalation_elements, reset_covariance_escalation_elements)


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


class PhiFisherStatisticsProvider:
    r"""Batch belief statistics for the ``fisher_group`` phi direction (derivation 2026-09-28, Track 2).

    The metric is evaluated at the layer-0 prior-bank rows -- the generative-model object the
    M-step updates -- never at deeper-layer posteriors. :meth:`set_batch` is called by
    :func:`train_step` immediately before the optimizer step with the full step's input tokens (the
    microbatches of ``grad_accum_steps > 1`` are chunks of that tensor along dim 0, so it is their
    concatenation) and the scored-token count :math:`N` that divides the CE gradient.

    For ``phi_embed`` rows :math:`v` it returns the table row :math:`(\mu_v, s_v)` and
    :math:`n_v`, the occurrences of :math:`v` among the input tokens. For ``pos_phi_free`` rows
    :math:`p` it returns the middle-fiber moments :math:`E_{v}^{-1}\cdot q_{v}` of the distinct column
    types and the gather index of each batch column ``tokens[:, p]``, from which the consumer forms the
    pooled operators a row chunk at a time (:func:`~vfe3.geometry.phi_fisher.pooled_operator_chunks`),
    and :math:`B_{\rm tot}`. Everything is float64, detached, on the table's device.
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
            count = torch.full((rows.shape[0],), float(tokens.shape[0]),
                               dtype=torch.float64, device=rows.device)
            return FisherRowStatistics(kind="pos", count=count, total=total,
                                       pooled_moments=(xi0, C0, C0_inv, gather))
        raise ValueError("fisher_group statistics exist only for phi_embed and pos_phi_free")


def build_optimizer(
    model: VFEModel,
    cfg:   VFE3Config,
) -> torch.optim.Optimizer:
    r"""AdamW with per-group M-step learning rates over the PriorBank prior tables.

    The prior-table roles carry distinct natural scales, so each is given its own
    M-step learning rate: every live mean table at ``m_p_mu_lr``; every live (log) scale
    table at ``m_p_sigma_lr`` (the decode temperature ``decode_log_scale`` is frozen and
    ungrouped); the belief gauge-frame coordinates ``phi_embed`` at ``m_phi_lr``.
    ``m_decode_mu_lr`` controls separate decode means or linear weight/bias;
    ``m_decode_sigma_lr`` controls separate decode log variances. Each falls back to the
    corresponding encoder rate; shared encode/decode tables keep encoder rates.
    Omitting overrides preserves historical group topology and semantic mu/sigma decays.

    Optional parameters are grouped only when their toggle is on: the linear decode weight
    ``output_proj_weight`` and bias (decode_mode='linear') at ``m_decode_mu_lr`` (``m_p_mu_lr`` when None).
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
    # Every None-inherits rate and decay below resolves through VFE3Config.resolved_hyperparameter, the one place
    # the inheritance rule lives (audit 2026-10-12 C7).
    sigma_wd = {"weight_decay": cfg.resolved_hyperparameter("sigma_weight_decay")}
    # mu_weight_decay (default None = inherit the global weight_decay, the long-standing behavior):
    # the exact mirror of sigma_wd for the MEAN-role capacity tables. Decoupled AdamW decay reaches
    # every row of a live embedding table on every step, so the rarely-visited rows are decayed far
    # faster than their gather-driven gradient can restore them; mu_weight_decay=0.0 exempts the
    # mean sector. Applied to the same class sigma_wd covers (belief, untied decode) plus
    # the linear-decode projection, which is the mean-role twin of decode_log_scale.
    mu_wd = {"weight_decay": cfg.resolved_hyperparameter("mu_weight_decay")}
    # decode_mu_weight_decay (default None = mu_wd): a separate decoupled decay for the decode-side mean tables
    # only (untied decode means, linear readout weight). Decoupled AdamW decay is not an L2 penalty or a
    # Gaussian-prior MAP (Loshchilov and Hutter 2019); against a mean Adam-normalized update u_bar it settles at
    # |theta| = |u_bar| / lambda, which bounds the rows of rarely-targeted types that otherwise only receive the
    # softmax push away from the queries (audit 2026-10-10, item 2; round 2, item 43).
    decode_mu_wd = {"weight_decay": cfg.resolved_hyperparameter("decode_mu_weight_decay")}
    # m_table_update_mode='fisher' marks the type-prior tables for the natural-gradient step of
    # vfe3.table_optim; their AdamW rates then only schedule that step and set its decoupled decay.
    table_fisher = cfg.m_table_update_mode == "fisher"
    groups = []
    if pb.mu_embed is not None:
        groups.append({"params": [pb.mu_embed], "lr": cfg.m_p_mu_lr, "role": "mu", **mu_wd,
                       **({"table_fisher": "mu"} if table_fisher else {})})
    # decode_log_scale is always frozen (requires_grad=False in VFEModel; the learned temperature was
    # retired 2026-10-02), so it joins no optimizer group and the readout temperature stays fixed.
    decode_mu_lr = cfg.resolved_hyperparameter("m_decode_mu_lr")
    decode_sigma_lr = cfg.resolved_hyperparameter("m_decode_sigma_lr")
    # prior_covariance_table='block_cholesky' replaces sigma_log_embed by the Cholesky table sigma_chol_embed.
    sigma_parameters = [parameter for parameter in (pb.sigma_log_embed, getattr(pb, "sigma_chol_embed", None))
                        if parameter is not None and parameter.requires_grad]
    if sigma_parameters:
        groups.append({"params": sigma_parameters, "lr": cfg.m_p_sigma_lr,
                       "role": "sigma", **sigma_wd,
                       **({"table_fisher": "sigma"} if table_fisher and sigma_parameters == [pb.sigma_log_embed]
                          else {})})
    if phi_group is not None:
        groups.append(phi_group)
    if getattr(pb, "decode_mu_embed", None) is not None:        # untie_decode_bank=True decode tables
        # Cloned from the encode tables at init (step-0 byte-identical decode), trained separately so
        # the decode direction can decouple from the E-step prior/self-coupling target; grouped like
        # the tables they were cloned from unless their decoder rate overrides the fallback.
        groups.append({"params": [pb.decode_mu_embed],        "lr": decode_mu_lr,    "role": "mu", **decode_mu_wd})
        groups.append({"params": [pb.decode_sigma_log_embed], "lr": decode_sigma_lr, "role": "sigma", **sigma_wd})
    if pb.output_proj_weight is not None:                       # decode_mode='linear'
        groups.append({"params": [pb.output_proj_weight], "lr": decode_mu_lr, "role": "mu", **decode_mu_wd})
    if pb.output_proj_bias is not None:                         # decode_mode='linear': learned log-unigram prior
        # weight_decay=0 -- decaying a unigram prior toward zero biases it to a flat distribution
        # (the same protection phi/Omega carry).
        groups.append({"params": [pb.output_proj_bias], "lr": decode_mu_lr, "weight_decay": 0.0, "role": "mu"})
    if getattr(model, "block_mlps", None) is not None:         # use_block_mlp=True coordinate MLPs
        block_mlp_lr = cfg.resolved_hyperparameter("m_block_mlp_lr")
        # weight_decay=0 on the prototype_agents template log-variances log(lambda) and log-weights log(pi):
        # decaying them pulls lambda toward 1 and pi toward uniform, a prior on the templates rather than a
        # capacity penalty. The template means nu keep the global decay.
        template_priors = [p for mlp in model.block_mlps if isinstance(mlp, PrototypeAgentBlockMLP)
                           for p in (mlp.log_lambda, mlp.log_pi)]
        exempt = {id(p) for p in template_priors}
        groups.append({"params": [p for p in model.block_mlps.parameters() if id(p) not in exempt],
                       "lr": block_mlp_lr, "weight_decay": cfg.weight_decay,
                       "role": "mu", "lr_aux_role": "block_mlp"})
        if template_priors:
            groups.append({"params": template_priors, "lr": block_mlp_lr, "weight_decay": 0.0, "role": "mu"})
    if getattr(model, "relational_operators", None) is not None:
        layers = model.relational_operators_by_layer
        # channel_log_noise (predictive_readout='minimal_agent') is a score-side parameter: it enters the routing
        # through the record covariance. raw_score_k and O are None there (tied key, derived receiving map).
        # routing_key_log_noise (relational_routing='noisy_key') is score-side for the same reason.
        for role, names in (("score", ("raw_score_q", "raw_score_k", "channel_log_noise", "routing_log_temperature",
                                       "routing_key_log_noise")),
                            ("content", ("T", "O"))):
            # Mixed head sizes store a ParameterList per name (one tensor per run of equal heads).
            parameters = [p for ops in layers for name in names
                          for p in (ops._stored_runs(name) if getattr(ops, name, None) is not None else [])]
            if role == "content":
                # content_link_edges: the edge factor maps are content-link parameters (absent when off).
                parameters += [p for ops in layers for p in getattr(ops, "edge_parameters", list)()]
            parameters = [p for p in parameters if p.requires_grad]
            if parameters:
                groups.append({"params": parameters, "lr": cfg.resolved_hyperparameter(f"m_relational_{role}_lr"),
                               "weight_decay": 0.0, "role": "mu", "lr_aux_role": f"relational_{role}"})

    if getattr(model, "cg_coupling", None) is not None:         # use_cg_coupling=True CG path weights
        groups.append({"params": [model.cg_coupling.path_weights], "lr": cfg.m_p_mu_lr, "role": "mu"})
    if getattr(model, "pos_phi_free", None) is not None:        # pos_phi='learned' positional table
        # A gauge-frame scale, decayed like phi_embed (both compose into the ONE clamped frame, so
        # they share phi_weight_decay), but with its OWN optional step size: m_pos_phi_lr=None
        # inherits m_phi_lr, which is the historical yoked behavior.
        pos_group = {"params": [model.pos_phi_free], "lr": cfg.resolved_hyperparameter("m_pos_phi_lr"),
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
        if hidden_bias_parameters:
            groups.append({"params": hidden_bias_parameters,
                           "lr": cfg.resolved_hyperparameter("m_per_type_hidden_bias_lr"),
                           "weight_decay": cfg.phi_weight_decay, "role": "phi"})
    # LayerNorm affine (layernorm_affine=True on a "layernorm" block norm): learned per-feature gamma/beta
    # on the belief mean, carried by the block AffineLayerNorm nn.Module. weight_decay=0 --
    # gamma/beta are normalization calibration, not capacity (decaying gamma toward 0 shrinks the
    # normalized signal; the same exemption output_proj_bias carries). role='mu'
    # is the catch-all for learned non-variance/non-gauge tables; NO gauge flag (the affine touches
    # no gauge transport), so it rides as plain AdamW even under GaugeManifoldAdamW.
    ln_affine = []
    _nm = getattr(model, "block_norm", None)
    if isinstance(_nm, torch.nn.Module):
        ln_affine += [p for p in _nm.parameters() if p.requires_grad]
    if ln_affine:
        # Normalization calibration, not prior-table capacity, so it carries its own optional rate
        # (m_layernorm_lr=None inherits m_p_mu_lr, the historical behavior) and its own aux role, the
        # way block_mlp does.
        groups.append({"params": ln_affine, "lr": cfg.resolved_hyperparameter("m_layernorm_lr"), "weight_decay": 0.0,
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
    # bug class the optimizer is most prone to as new learnable seams (output_proj, block MLP, ...) are
    # added. Non-trainable params (requires_grad=False, e.g. a parameter frozen by design) are
    # intentionally exempt: they are fixed by design and need no optimizer group.
    # NOTE: this guards GROUPING/coverage, not gradient FLOW. A grouped parameter can still receive
    # a null gradient under specific opt-in toggles, by design; that is an intentional route
    # property, not a coverage bug.
    grouped = {p for g in groups for p in g["params"]}
    missing = {p for p in model.parameters() if p.requires_grad} - grouped   # frozen params are exempt
    if missing:
        raise AssertionError(
            f"build_optimizer left {len(missing)} model parameter(s) ungrouped; they would never "
            f"train. Add them to a param group."
        )

    table_kwargs = {}
    if table_fisher:
        # gaussian_frame_diagonal: the metric of p_v in mu_v is U_v^-T diag(1/s_v) U_v^-1, so the step reads the layer-0
        # frames of the step's types; on identity frames it is the diagonal metric and the diagonal step is exact.
        from vfe3.model.frame_actions import frames_identity_reason
        frame_builder = (model.token_type_frames
                         if pb.frame_diagonal and frames_identity_reason(cfg, pb.phi_embed.shape[-1]) is None
                         else None)
        table_kwargs = dict(statistics=TableFisherStatistics(pb, frame_builder=frame_builder),
                            table_fisher_lr=cfg.m_table_fisher_lr,
                            table_fisher_trust_radius=cfg.m_table_fisher_trust_radius)
    if phi_update_policy.requires_manifold_optimizer:
        # Stored phi factors use one mixed optimizer; every other group remains standard AdamW.
        #
        # FUSED IS FORWARDED HERE (audit 2026-08-07). It used to be left unset on this route, which
        # is NOT the same as off: torch.optim.AdamW defaults `fused=None`, and with both `fused` and
        # `foreach` unresolved `adam()` selects the FOREACH multi-tensor path -- 9 kernel launches
        # per parameter group per step against 1 for the fused kernel. So choosing the manifold
        # route silently deoptimized every OTHER group (mu_embed, sigma_log_embed,
        # transport_state, LayerNorm affine), none of which the manifold logic
        # touches. `step()` sets `p.grad = None` on every pullback_group parameter BEFORE
        # the trailing `super().step()`, so base AdamW skips exactly those and the fused kernel only
        # ever sees the standard groups.
        #
        # NOT VERIFIED ON DEVICE: `fused` is a CUDA-only path, so this change cannot be exercised on
        # a CPU box. If a manifold run regresses, this line is the first thing to revert.
        return (TableFisherGaugeManifoldAdamW if table_fisher else GaugeManifoldAdamW)(
            groups,
            model.group,
            **table_kwargs,
            phi_group_trust_radius=cfg.m_phi_group_trust_radius,
            phi_chart_max_norm=(
                cfg.phi_mstep_max_matrix_norm
                if cfg.phi_mstep_max_matrix_norm is not None
                else 5.0
            ),
            phi_group_retraction=cfg.m_phi_group_retraction,
            phi_statistics_provider=(
                PhiFisherStatisticsProvider(model, cfg.m_phi_fisher_rotation_weight)
                if phi_update_policy.requires_fisher_statistics else None
            ),
            phi_fisher_rotation_weight=cfg.m_phi_fisher_rotation_weight,
            # The exact Daleckii-Krein route, the eigenbasis group_log fallback and the stabilizer completion are
            # config fields, read per optimizer (audit 2026-10-16 F37, F40, F50).
            phi_group_differential_mode=cfg.m_phi_group_differential_mode,
            phi_group_log_fallback=cfg.m_phi_group_log_fallback,
            phi_fisher_completion=cfg.m_phi_fisher_completion,
            weight_decay=cfg.weight_decay,
            fused=next(model.parameters()).is_cuda,
        )
    # fused AdamW (one CUDA kernel for the whole M-step) when the priors live on CUDA; it is
    # CUDA-only, so on a CPU box this is the standard AdamW. Per-group LRs are honored either way.
    use_fused = next(model.parameters()).is_cuda
    if table_fisher:
        return TableFisherAdamW(groups, weight_decay=cfg.weight_decay, fused=use_fused, **table_kwargs)
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
    allowed = {"block_mlp", "relational_score", "relational_content", "layernorm"}
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
    ``cfg.min_lr_frac * base``.

    A ``LambdaLR`` scales each group's base LR by its multiplier, so flooring the
    *multiplier* at ``min_lr_frac`` floors the *product*::

        base * max(min_lr_frac, cosine) = max(min_lr_frac*base, base*cosine).

    The fractional floor scales with each group's own base, preserving the m_mu:m_sigma:m_phi
    ratios into the cosine tail; a group whose base LR is 0 (a deliberately frozen channel,
    e.g. ``m_phi_lr=0``) stays frozen. With ``min_lr_frac=0`` this is the pure
    half-cosine-to-zero.
    """
    def make(base: float) -> Callable[[int], float]:
        floor = cfg.min_lr_frac                                 # per-group multiplier floor
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


def _gradients_finite(gradients: Sequence[torch.Tensor]) -> torch.Tensor:
    r"""One ``()`` bool device flag: every gradient is finite, by :math:`\mathrm{isfinite}(\max_i|g_i|)` per tensor.

    The real strided floating gradients share one ``torch._foreach_norm(ord=inf)`` call instead of one infinity-norm
    launch per parameter (audit 2026-10-17 item 70); the verdict equals :func:`_gradient_isfinite` on each tensor,
    which the remaining (complex, empty or non-strided) gradients still use.
    """
    fast = [g.detach() for g in gradients
            if g.layout == torch.strided and g.is_floating_point() and g.numel() > 0]
    flags: List[torch.Tensor] = []
    if fast:
        norms = torch._foreach_norm(fast, float("inf"))
        flags.append(torch.isfinite(torch.stack([norm.to(torch.float32) for norm in norms])).all())
    flags.extend(_gradient_isfinite(g) for g in gradients
                 if not (g.layout == torch.strided and g.is_floating_point() and g.numel() > 0))
    return torch.stack(flags).all() if len(flags) > 1 else flags[0]


CLIP_COEFFICIENT_EPS: float = 1e-6        # clip_grads_with_norm_ scales by c / (||g|| + 1e-6), at most 1


def _role_norms_and_clip_coefficients(
    norm_sets: Sequence[Tuple[str, list, bool]],      # (role, parameters, clipped) of train_step
    values:    Optional[Sequence[float]],             # host pre-clip norm of each set; None if not transferred
    grad_clip: Optional[float],                       # the applied threshold c; None when no clip ran
) -> Tuple[Dict[str, float], Dict[str, float]]:
    r"""Per-role pre-clip norms and realized clip coefficients from the host copies of the set norms.

    A role combines its sets in quadrature, :math:`\|g_r\| = (\sum_{s\in r}\|g_s\|^2)^{1/2}` (a role with one
    set reports that norm unchanged). A clipped set is scaled by :math:`\min(1, c/(\|g_s\| + 10^{-6}))`, as
    ``clip_grads_with_norm_`` scales it (here in float64 on the host, a diagnostic copy of the device value);
    a role reports the smallest coefficient of its clipped sets, so a value below 1 means its clip was binding.
    """
    if values is None:
        return {}, {}
    parts: Dict[str, List[float]] = {}
    coefficients: Dict[str, float] = {}
    for (role, _, clipped), value in zip(norm_sets, values):
        parts.setdefault(role, []).append(float(value))
        if clipped and grad_clip is not None:
            coefficient = min(1.0, float(grad_clip) / (float(value) + CLIP_COEFFICIENT_EPS))
            coefficients[role] = min(coefficient, coefficients.get(role, coefficient))
    norms = {role: (values_[0] if len(values_) == 1 else math.sqrt(sum(v * v for v in values_)))
             for role, values_ in parts.items()}
    return norms, coefficients


def _parameter_clip_scales(
    norm_sets: Sequence[Tuple[str, list, bool]],      # (role, parameters, clipped) of train_step
    values:    Optional[Sequence[float]],             # host pre-clip norm of each set; None if not transferred
    grad_clip: Optional[float],                       # the applied threshold c; None when no clip ran
) -> Dict[int, float]:
    r"""The coefficient :math:`\min(1, c/(\|g_s\| + 10^{-6}))` that scaled each clipped parameter, keyed by ``id``."""
    if values is None or grad_clip is None:
        return {}
    scales: Dict[int, float] = {}
    for (_, parameters, clipped), value in zip(norm_sets, values):
        if clipped:
            coefficient = min(1.0, float(grad_clip) / (float(value) + CLIP_COEFFICIENT_EPS))
            for parameter in parameters:
                scales[id(parameter)] = coefficient
    return scales


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
                                "expected_likelihood_chunked")
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
    grad_clip_scope:  str                                = "role",
    grad_accum_steps: int                                = 1,
    scaler:           Optional['torch.amp.GradScaler']  = None,
    metrics_out:      Optional[dict]                     = None,
    status_out:       Optional[dict]                     = None,
    frame_projection_out: Optional[torch.Tensor]         = None,   # (2,) int64 device counter
    loss_skip_threshold:  Optional[float]                = None,   # reject the update when the loss exceeds it
) -> float:
    r"""One M-step (one optimizer step) on the cross-entropy of a batch; returns the loss.

    Zeroes the prior-table gradients, runs the forward (encode -> unrolled E-step ->
    decode -> CE), backpropagates the loss through inference to the prior tables, clips
    each optimizer role's gradient norm to ``grad_clip``, then takes one AdamW + scheduler step.
    ``grad_clip`` of ``None`` or ``0.0`` disables clipping entirely (the optimizer and
    scheduler still step on the raw gradient). After accumulation and unscale, a positive
    threshold is applied once independently to each clip set: each optimizer role
    (mu/sigma/phi/other) under ``grad_clip_scope='role'``, each optimizer group under ``'group'``.
    A set whose pre-clip norm is :math:`\|g\|` is scaled by :math:`\min(1, c/(\|g\| + 10^{-6}))`
    (``clip_grads_with_norm_``). Groups marked ``table_fisher`` are never clipped: their per-row
    Fisher trust radius bounds the step (audit 2026-10-10 round 2, items 8 and 9).

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
    ``status_out`` also receives, from host values the step already transfers (no additional
    synchronization): ``"nonfinite_skip"`` (the update was rejected for a non-finite loss, gradient
    or clip norm), ``"ce"`` (the pre-step CE), ``"role_grad_norms"`` (pre-clip L2 norm per role,
    the clip sets of a role combined in quadrature; empty when no norm was transferred) and
    ``"clip_coefs"`` (per role, the smallest realized clip coefficient of its clipped sets; empty
    when no clip was applied). Asking for ``status_out`` also computes the set norms when clipping
    is off, so they join the same transfer (audit 2026-10-10 round 2, item 1). Under an enabled GradScaler,
    asking for ``status_out`` unscales the gradients once before the norms (``GradScaler.step`` does not
    repeat it), and with clipping off the set norms take their own transfer, as on the clipped scaler
    route (audit 2026-10-16, F51). On a step that reaches
    ``optimizer.step``, ``status_out["grads"]`` holds references, keyed by ``id``, to the gradients the
    step consumed (after unscale and clip; no copy and no synchronization), because the table Fisher step
    and the manifold phi step clear ``.grad`` inside ``step()`` (audit 2026-10-16, F4).

    ``frame_projection_out`` (S1 panel D counters): a ``(2,)`` int64 tensor on the generator
    device. When given and the post-update frame projection runs, the rows it rescales and the rows
    it checks are added to it on the device (no host synchronization); ``None`` makes exactly the
    call the step made before the counter existed.

    ``loss_skip_threshold`` (``loss_spike_skip_nats``; audit 2026-10-17 item 17): when given, a finite loss above it
    rejects the update as a non-finite one is rejected (gradients dropped, no optimizer or scheduler step), read
    from the host copy of the loss the step already transfers, so no synchronization is added. ``status_out``
    then carries ``"loss_spike_skip": True`` and ``"nonfinite_skip": False`` (the precision ladder does not replay
    it). ``None`` keeps the step unchanged.

    A clipped set whose host pre-clip norm :math:`v` satisfies :math:`v + 10^{-6} \le c(1 - 10^{-6})` has the clip
    coefficient exactly 1, so its multiply is skipped (bit-identical; audit 2026-10-17 item 70).
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
        # Raise-only frame certificates join the forward's next mandatory host read; the scope closes
        # (resolving anything still queued) before backward, so no uncertified value reaches a gradient.
        with deferred_host_checks():
            _, loss, decode_stats = model(
                tokens, targets, estep_grad_out=_egrad, return_decode_stats=True)
        ce = decode_stats.ce
        _step_ce_det = ce.detach() if ce is not None else None   # () pre-step CE, read at the existing barrier
        _scored_det = decode_stats.scored_tokens.detach()
        _excluded_det = _excluded_token_count(decode_stats, targets.numel()).detach()
        # A device fill, not a pageable host-to-device copy (which synchronizes the stream).
        _expected_det = torch.full((), targets.numel(), dtype=torch.int64, device=targets.device)
        _scaler.scale(loss).backward()
        _loss_det = loss.detach()                               # host read DEFERRED: fused with the grad-finite
        step_loss = None                                        # flag at the gate below (audit 2026-07-01 round-3)
        # CE is synced to a Python float only when a metrics dict is being filled this step (it feeds
        # metrics_out['train_ce'] and nothing else); on a silent step the extra D2H copy is skipped.
        step_ce = (float(ce.detach()) if (ce is not None and metrics_out is not None) else float("nan"))
        # decode_second_order: the second-order objective the step trained on, beside the plug-in CE.
        step_objective = (float(decode_stats.objective)
                          if (decode_stats.objective is not None and metrics_out is not None) else None)
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
        objective_numerator: Optional[torch.Tensor] = None     # decode_second_order only
        for tok_mb, tgt_mb in zip(tok_chunks, tgt_chunks):
            _egrad_mb = {} if metrics_out is not None else None
            with deferred_host_checks():
                _, loss_mb, decode_stats_mb = model(
                    tok_mb, tgt_mb, estep_grad_out=_egrad_mb, return_decode_stats=True)
            ce_mb = decode_stats_mb.ce
            count_mb = decode_stats_mb.scored_tokens.detach()
            count_tensors.append(count_mb)
            excluded_count_tensors.append(
                _excluded_token_count(decode_stats_mb, tgt_mb.numel()).detach())
            expected_count_tensors.append(torch.full(
                (), tgt_mb.numel(), dtype=torch.int64, device=tgt_mb.device))
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
            if decode_stats_mb.objective is not None:
                objective_numerator = (0.0 if objective_numerator is None else objective_numerator) + (
                    decode_stats_mb.objective.to(dtype=torch.float64) * count_mb)
        _counts = torch.stack(count_tensors)
        _scored_det = _counts.sum(dtype=torch.int64)
        _excluded_det = torch.stack(excluded_count_tensors).sum(dtype=torch.int64)
        _expected_det = torch.stack(expected_count_tensors).sum(dtype=torch.int64)
        _denominator = _scored_det.clamp_min(1).to(dtype=torch.float64)
        _loss_det = loss_numerator / _denominator
        _ce_det = ce_numerator / _denominator
        _step_ce_det = _ce_det
        step_loss = None
        step_ce = float(_ce_det) if metrics_out is not None else float("nan")
        step_objective = (float(objective_numerator / _denominator)
                          if (objective_numerator is not None and metrics_out is not None) else None)
        _mb_tok[:] = [int(value) for value in _counts.cpu().tolist()]
        # Decoder exclusions, not merely ignore_index, define the CE denominator. Weighting the
        # non-CE regularizers by that same count remains approximate when microbatch counts differ.
        if _mb_tok and max(_mb_tok) != min(_mb_tok):
            import warnings
            warnings.warn(
                "grad_accum_steps>1 with uneven counted-token microbatches: non-CE regularizers "
                "(mstep_self_coupling) are token-weighted by n_mb/n_tot "
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
    # The pre-step CE rides whichever transfer resolves the loss below (audit 2026-10-10 round 2, item 1).
    _ce_tail = (_step_ce_det.to(dtype=torch.float64),) if _step_ce_det is not None else ()
    step_ce_value = float("nan")
    # The enabled scaler's ordinary finite-loss path delegates overflow detection to GradScaler.
    # Resolve the scalar loss first so the rare nonfinite-loss branch can explicitly inspect gradients
    # and distinguish scale backoff (nonfinite gradients) from scale hold (finite gradients).
    if _scaler_enabled:
        if step_loss is None:
            _loss_flags = torch.stack((
                _loss_det.to(dtype=torch.float64),
                _target_partition_valid_det.to(dtype=torch.float64),
                _zero_scored_det.to(dtype=torch.float64),
                *_ce_tail,
            )).tolist()
            step_loss = _loss_flags[0]
            target_partition_valid = bool(_loss_flags[1])
            zero_scored_targets = bool(_loss_flags[2])
            if _ce_tail:
                step_ce_value = _loss_flags[3]
        loss_finite = math.isfinite(step_loss)
    else:
        loss_finite = True                                  # resolved with the fused default-path scan below
    # Unscale once when clipping/metrics/status needs true-unit gradients or the enabled scaler must classify
    # a nonfinite scalar loss. GradScaler remembers that unscale_ ran; its later step does not repeat it.
    # status_out joins the list so its role norms and the spike attribution read true units with clipping
    # off (audit 2026-10-16, F51); without a scaler unscale_ is a no-op.
    need_unscale = (
        grad_accum_steps > 1
        or
        (grad_clip is not None and grad_clip > 0)
        or (metrics_out is not None)
        or (_scaler_enabled and not loss_finite)
        or (_scaler_enabled and status_out is not None)
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
    _live_grads = ([p.grad for g in optimizer.param_groups for p in g["params"] if p.grad is not None]
                   if explicit_grad_check else [])
    _flags = [_gradients_finite(_live_grads)] if _live_grads else []
    # Clip sets (audit 2026-10-10 round 2, items 8 and 9). Under grad_clip_scope='role' the optimizer
    # groups partition parameters into roles and each role gets one L2 bound (the historical sets, in
    # the same order); under 'group' each optimizer group gets its own. Groups marked table_fisher are
    # never clipped; their norm is still taken, after the clipped sets, for the per-role report.
    if grad_clip_scope not in ("role", "group"):
        raise ValueError(f"grad_clip_scope must be 'role' or 'group', got {grad_clip_scope!r}")
    clip_on = grad_clip is not None and grad_clip > 0
    norm_sets: List[Tuple[str, list, bool]] = []                 # (role, parameters, clipped)
    if clip_on or status_out is not None:
        clipped_groups = [group for group in optimizer.param_groups if not group.get("table_fisher")]
        if grad_clip_scope == "role":
            role_params: dict = {}
            for group in clipped_groups:
                role_params.setdefault(group.get("role", "other"), []).extend(group["params"])
            norm_sets = [(role, parameters, clip_on) for role, parameters in role_params.items() if parameters]
        else:
            norm_sets = [(group.get("role", "other"), list(group["params"]), clip_on)
                         for group in clipped_groups if group["params"]]
        norm_sets += [(group.get("role", "other"), list(group["params"]), False)
                      for group in optimizer.param_groups if group.get("table_fisher") and group["params"]]
    clip_roles: List[list] = [parameters for _, parameters, clipped in norm_sets if clipped]

    def _set_norms() -> List[torch.Tensor]:
        # L2 norm of each set's unscaled, accumulated gradients, exactly what clip_grad_norm_ computes.
        return [torch.nn.utils.get_total_norm([p.grad for p in parameters if p.grad is not None])
                for _, parameters, _ in norm_sets]

    set_norms: Optional[List[torch.Tensor]] = None
    set_norm_values: Optional[List[float]] = None              # host copies, one per norm set
    clip_norms: Optional[List[torch.Tensor]] = None
    clip_finite: Optional[bool] = None
    if not _scaler_enabled and _flags and step_loss is None:    # fuse default-path loss + grad flag
        if norm_sets:
            # Taken before this barrier (nothing below changes a gradient before the clip), so the norms
            # themselves, not only their finiteness, join this one transfer instead of a second sync.
            set_norms = _set_norms()
        _values = torch.stack((
            _loss_det.to(dtype=torch.float64),
            torch.stack(_flags).all().to(dtype=torch.float64),
            _target_partition_valid_det.to(dtype=torch.float64),
            _zero_scored_det.to(dtype=torch.float64),
            *_ce_tail,
            *(norm.to(device=tokens.device, dtype=torch.float64) for norm in (set_norms or ())),
        )).tolist()
        step_loss = _values[0]
        grad_finite = bool(_values[1])
        target_partition_valid = bool(_values[2])
        zero_scored_targets = bool(_values[3])
        if _ce_tail:
            step_ce_value = _values[4]
        if set_norms is not None:
            set_norm_values = _values[4 + len(_ce_tail):]
            if clip_roles:
                clip_norms = [norm for norm, (_, _, clipped) in zip(set_norms, norm_sets) if clipped]
                clip_finite = all(math.isfinite(value) for value, (_, _, clipped)
                                  in zip(set_norm_values, norm_sets) if clipped)
    elif _flags:
        grad_finite = bool(torch.stack(_flags).all())
    if step_loss is None:                                       # fp16 / no-grads fallback: one plain loss sync
        _loss_flags = torch.stack((
            _loss_det.to(dtype=torch.float64),
            _target_partition_valid_det.to(dtype=torch.float64),
            _zero_scored_det.to(dtype=torch.float64),
            *_ce_tail,
        )).tolist()
        step_loss = _loss_flags[0]
        target_partition_valid = bool(_loss_flags[1])
        zero_scored_targets = bool(_loss_flags[2])
        if _ce_tail:
            step_ce_value = _loss_flags[3]
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
        if step_objective is not None:
            # decode_second_order: train_ce stays the plug-in CE validation scores; the second-order
            # objective the step trained on is logged under its own name (audit 2026-10-12 round 2, item 12).
            metrics_out["train_second_order_ce"] = step_objective
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
    # Opt-in loss-spike rejection on the host loss already in hand (no added synchronization; audit 2026-10-17 item 17).
    loss_spike_skip = bool(loss_skip_threshold is not None and not skip_step and step_loss > loss_skip_threshold)
    skip_step = skip_step or loss_spike_skip
    if metrics_out is not None:
        metrics_out["grad_finite"] = float(grad_finite)
    clip_applied = False
    if (status_out is not None and _scaler_enabled and not clip_on and not skip_step
            and set_norm_values is None and norm_sets):
        # Enabled scaler with clipping off: no transfer above carried the set norms, so they take their own,
        # as on the clipped scaler route below; the gradients were unscaled above (audit 2026-10-16, F51).
        set_norm_values = torch.stack([norm.to(device=tokens.device, dtype=torch.float64)
                                       for norm in _set_norms()]).tolist()
    if grad_clip is not None and grad_clip > 0 and not skip_step:
        if clip_norms is None:                                  # scaler / fallback routes: own transfer
            set_norms = _set_norms()
            set_norm_values = torch.stack([norm.to(device=tokens.device, dtype=torch.float64)
                                           for norm in set_norms]).tolist()
            clip_norms = [norm for norm, (_, _, clipped) in zip(set_norms, norm_sets) if clipped]
            clip_finite = all(math.isfinite(value) for value, (_, _, clipped)
                              in zip(set_norm_values, norm_sets) if clipped)
        # clip_grad_norm_ is get_total_norm followed by clip_grads_with_norm_; the split keeps its values. A set whose
        # host norm gives a coefficient of exactly 1 (clamp(c/(v + 1e-6), max=1) = 1 with margin over fp32 rounding)
        # skips the multiply by 1, which is bit-identical (audit 2026-10-17 item 70).
        clip_values = [value for value, (_, _, clipped) in zip(set_norm_values or (), norm_sets) if clipped]
        for index, (parameters, norm) in enumerate(zip(clip_roles, clip_norms)):
            value = clip_values[index] if index < len(clip_values) else None
            if (value is not None and math.isfinite(value)
                    and value + CLIP_COEFFICIENT_EPS <= float(grad_clip) * (1.0 - 1e-6)):
                continue
            torch.nn.utils.clip_grads_with_norm_(parameters, grad_clip, norm)
        clip_applied = True
        skip_step = not clip_finite
        if metrics_out is not None:
            metrics_out["grad_clip_finite"] = float(clip_finite)
    if skip_step and _scaler_enabled and not explicit_grad_check:
        # Clipping may be the first overflow detector on a metrics-free step.
        # Resolve it before zero_grad so true overflow triggers scaler backoff.
        live = [p.grad for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
        grad_finite = bool(_gradients_finite(live)) if live else True
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
        # m_table_update_mode='fisher': the table step's counts n_v and denominator N.
        _table_provider = getattr(optimizer, "table_statistics_provider", None)
        if _table_provider is not None:
            _table_provider.set_batch(tokens, _scored_det)
        if status_out is not None:
            # References (no copy, no host sync) to the gradients that reach the step: the table Fisher step
            # and the manifold phi step set .grad = None inside step(), so a post-step reader of .grad (the spike
            # attribution, the known-dead set) would miss those tables (audit 2026-10-16, F4).
            status_out["grads"] = {id(p): p.grad for group in optimizer.param_groups
                                   for p in group["params"] if p.grad is not None}
        _scaler.step(optimizer)
        _scaler.update()
        if _fisher_provider is not None:
            _fisher_provider.set_batch(None, None)
        if _table_provider is not None:
            _table_provider.set_batch(None, None)
    did_step = not skip_step
    if scale_before is not None and float(_scaler.get_scale()) < scale_before:
        did_step = False
    if status_out is not None:
        status_out["did_step"] = did_step
        status_out["nonfinite_skip"] = bool(skip_step and not zero_scored_targets and not loss_spike_skip)
        status_out["loss_spike_skip"] = loss_spike_skip
        status_out["ce"] = step_ce_value
        role_norms, clip_coefs = _role_norms_and_clip_coefficients(
            norm_sets, set_norm_values, grad_clip if clip_applied and not skip_step else None)
        status_out["role_grad_norms"] = role_norms
        status_out["clip_coefs"] = clip_coefs
        # Per-parameter clip coefficient (keyed by id), so a spike can undo the clip on the live gradients.
        status_out["clip_scales"] = _parameter_clip_scales(
            norm_sets, set_norm_values, grad_clip if clip_applied and not skip_step else None)
    if metrics_out is not None:
        metrics_out["step_skipped"] = float(not did_step)
    _cfg = model.cfg
    if (
        did_step
        and not get_phi_update_policy(getattr(_cfg, "m_phi_update_mode", "adamw")).requires_manifold_optimizer
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
    # Opt-in frame condition bound (audit 2026-10-12 T4); validation admits it only under adamw and
    # per_token, and the orthogonal/conformal groups have cond_2 = 1, so nothing runs for them. Its changed
    # and checked rows join the S1 panel D frame-projection counter on the device (round 2, item 14).
    if (did_step and getattr(_cfg, "phi_frame_condition_max", None) is not None
            and model.group.name in FRAME_CONDITION_GROUPS):
        bound_phi_frame_condition_(model, _cfg.phi_frame_condition_max,
                                   **({} if frame_projection_out is None else {"count_out": frame_projection_out}))
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


def _require_causal_next_token_prior(
    model: torch.nn.Module,
) -> None:
    r"""Refuse next-token training or scoring of a VFE model whose attention prior is not causal.

    Under a non-causal prior position :math:`i` attends to later tokens, so the cross-entropy of target
    :math:`y_{i+1}` is scored by a model that reads it (audit 2026-10-10 #4). Every built-in prior is causal
    since the non-causal ``uniform``, ``alibi`` and ``windowed`` were removed on 2026-10-12; this check reads the
    registry's causality metadata and guards a future non-causal registration. The baseline admits only
    causal priors at construction (``vfe3/baseline.py``), and a baseline model is not checked here, nor a model
    with no configured prior.
    """
    cfg = getattr(model, "cfg", None)
    if not is_vfe_model(model) or getattr(cfg, "beta_attention_prior", None) is None:
        return
    if not get_prior_registration(cfg.beta_attention_prior).is_causal_for_next_token_scoring(cfg):
        raise ValueError(
            f"beta_attention_prior={cfg.beta_attention_prior!r} is not causal for next-token scoring: "
            "each position would attend to the targets it is scored on. Train and evaluate with a prior "
            "registered as causal")


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

    The ignored targets are also split by position within each window row: ``overlap_context_targets``
    counts the leading run (the sliding-window context prefix) of rows that score at least one target,
    and ``padding_targets`` counts the trailing run (the padded final window) plus every target of a
    row that scores none. Ignored targets between scored ones fall in neither count.
    """
    _require_causal_next_token_prior(model)
    if device is None:
        device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        total_nats = torch.zeros((), dtype=torch.float64, device=device)
        total_tok = torch.zeros((), dtype=torch.int64, device=device)
        total_expected = torch.zeros((), dtype=torch.int64, device=device)
        total_excluded = torch.zeros((), dtype=torch.int64, device=device)
        total_overlap = torch.zeros((), dtype=torch.int64, device=device)
        total_padding = torch.zeros((), dtype=torch.int64, device=device)
        for i, (tokens, targets) in enumerate(loader):
            # Inspect CPU-loader padding before transfer; device-origin loaders retain checks.
            total_expected.add_(targets.numel())
            valid = targets != -100
            row_scores = valid.any(dim=-1, keepdim=True)
            leading = valid.cumsum(dim=-1) == 0                                 # (..., N) before the first scored target
            trailing = valid.flip(-1).cumsum(dim=-1) == 0                       # (..., N) after the last, reversed
            total_overlap.add_((leading & row_scores).sum().to(device, non_blocking=True))
            total_padding.add_(trailing.sum().to(device, non_blocking=True))
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
                    _, _, decode_stats = _eval_forward_with_precision_retry(
                        model, group_tokens, group_targets, return_decode_stats=True)
                    n_b = decode_stats.scored_tokens
                    total_nats.add_(decode_stats.ce.to(dtype=torch.float64) * n_b)
                    total_tok.add_(n_b)
                    total_excluded.add_(_excluded_token_count(
                        decode_stats, group_targets.numel()))
                    total_excluded.add_(rows.sum() * (targets.shape[1] - length))
            else:
                _, _, decode_stats = _eval_forward_with_precision_retry(
                    model, tokens, targets, return_decode_stats=True)
                n_b = decode_stats.scored_tokens
                total_nats.add_(decode_stats.ce.to(dtype=torch.float64) * n_b)
                total_tok.add_(n_b)
                total_excluded.add_(_excluded_token_count(decode_stats, targets.numel()))
            if max_batches is not None and i + 1 >= max_batches:
                break               # draw exactly max_batches (process-then-break; no extra pull)
        total_nats_value, accounting = _evaluation_totals_from_device(
            total_nats, total_expected, total_tok, total_excluded)
        overlap_value, padding_value = torch.stack((total_overlap, total_padding)).cpu().tolist()
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
        "overlap_context_targets": int(overlap_value),
        "padding_targets":         int(padding_value),
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
            _, _, decode_stats = _eval_forward_with_precision_retry(
                model, tokens, masked, return_decode_stats=True)
        finally:
            if was_training:
                model.train()
    return float(decode_stats.ce)


# The model.diagnostics keys the training loop reads unconditionally (console line and metrics row); a
# failed log-step diagnostic falls back to these as NaN, widened to the first metrics row's key set.
_TRAIN_DIAGNOSTIC_FALLBACK_KEYS: Tuple[str, ...] = (
    "attn_entropy", "self_coupling", "self_divergence", "belief_coupling", "attention_entropy",
    "total", "effective_rank",
)


# Held-out geometry columns copied, under a "val_" prefix, from the relational diagnostics of the validation
# capture (audit 2026-10-12 round 2, item 13).
_VAL_GEOMETRY_KEYS: Tuple[str, ...] = (
    "fisher_trace_mean", "belief_cond_p95", "belief_lam_min", "belief_lam_max", "belief_cond_valid_frac",
    "phi_norm_mean", "phi_norm_std",
    "phi_matrix_norm_p95", "phi_matrix_norm_p99", "phi_matrix_norm_max",
    "vertex_cond_p99",
    "pos_phi_matrix_norm_p95", "pos_phi_matrix_norm_p99", "pos_phi_matrix_norm_max",
    "guard_sigma_floor_frac", "guard_sigma_ceil_frac",
    "nonfinite_frac",
    # the argmin token position of belief_lam_min and the minimum over positions >= 1 (audit 2026-10-17 item 13)
    "belief_lam_min_argmin_position", "belief_lam_min_from_position_1",
)


# Fixed column set for the held-out per-eval probes (initialized to NaN before training and carried
# forward like last_val, so every metrics.csv row carries the SAME columns -- rectangular -- whether
# or not an eval has run yet). Config-conditional keys (val_head_redundancy_js needs >= 2 heads;
# pos_loss_* needs targets and seq_len >= 4) stay NaN on runs where they do not apply.
_VAL_DIAG_KEYS = (
    "val_self_coupling", "val_self_divergence", "val_belief_coupling", "val_attention_entropy",
    "val_inner_alignment_energy_total",
    "val_attn_entropy", "val_effective_rank", "val_belief_cond_median", "val_attn_entropy_min",
    "val_attn_entropy_min_all", "val_attn_entropy_min_norm",
    "val_attn_collapsed_heads",        # per-head mean over the diagnostic windows (_collapsed_head_count)
    "val_future_leakage", "val_row_sum_error",
    "val_pos_content_r2", "val_prev_token_mass", "val_period_match_mass", "val_head_redundancy_js",
    "pos_loss_first_q", "pos_loss_last_q", "pos_loss_ratio",
    # held-out gauge / SPD / Fisher geometry (surfaced from the already-computed val diagnostics dict)
    "val_gauge_invariant_spread",
    *(f"val_{key}" for key in _VAL_GEOMETRY_KEYS),
)


# Relational probes exist only for relational runs, but their columns must exist before
# the first validation and stay present (blank) between evaluations or after a failed replay.
_VAL_RELATIONAL_DIAG_KEYS = (
    "val_relational_legacy_global_f_available",
    "val_relational_score_klmax_frac",
    # the per-layer maximum and its layer (the headline column above reduces only the last layer)
    "val_relational_score_klmax_frac_layer_max", "val_relational_score_klmax_frac_argmax_layer",
    # raw router costs that were non-finite before the kl_max clamp, and the router's label free energy
    # (VFEModel._nonfinite_frac, VFEModel._routing_free_energy; audit 2026-10-16 F10, F114)
    "val_relational_score_nonfinite_frac", "val_relational_score_nonfinite_frac_layer_max",
    "val_relational_routing_free_energy",
    # pre-clamp router cost, cost-term medians and capped routing mass (VFEModel._raw_router_cost_statistics and
    # _layer_guard_maxima; audit 2026-10-17 item 9)
    *(f"val_relational_raw_row_{name}_{level}" for name in ("min", "gap") for level in ("q10", "q50", "q90")),
    "val_relational_capped_mass_mean", "val_relational_capped_mass_max",
    *(f"val_relational_score_term_{term}_median"
      for term in ("query_constant", "key_logdet", "trace", "mahalanobis")),
    "val_relational_raw_row_min_median_head_max", "val_relational_raw_row_min_median_argmax_layer",
    "val_relational_raw_row_min_median_argmax_head", "val_relational_capped_mass_mean_head_max",
    # per-(layer, head) router saturation, maxima over layers and heads (VFEModel._layer_guard_maxima)
    "val_relational_score_klmax_frac_head_max",
    "val_relational_clamped_row_frac_head_max", "val_relational_clamped_row_frac_argmax_layer",
    "val_relational_clamped_row_frac_argmax_head",
    "val_relational_self_weight_head_max", "val_relational_self_argmax_frac_head_max",
    "val_relational_self_lock_argmax_layer", "val_relational_self_lock_argmax_head",
    "val_belief_cond_median_layer_max",
    # every layer beside the last-layer headline columns (VFEModel._layer_guard_maxima; audit 2026-10-20 item 14)
    "val_relational_capped_mass_mean_layer_max", "val_relational_capped_mass_mean_argmax_layer",
    "val_relational_score_term_query_constant_median_layer_max",
    "val_belief_lam_min_layer_min", "val_belief_lam_min_argmin_layer", "val_belief_lam_max_layer_max",
    "val_belief_cond_max_layer_max", "val_belief_eig_ratio_layer_min", "val_belief_eig_ratio_argmin_layer",
    "val_relational_mean_displacement_norm", "val_relational_snapshot_ce",
    "val_relational_receiver_cost_before", "val_relational_receiver_cost_after",
    "val_relational_receiver_cost_delta", "val_relational_receiver_noise",
    "val_relational_receiver_mean_cost_delta", "val_relational_receiver_covariance_cost_delta",
    "val_relational_receiver_target_distance_ratio",
    # closed_form_profiled: the root solve's largest residual and floored-slope count (audit 2026-10-12 round 2, M6)
    "val_relational_receiver_profiled_residual", "val_relational_receiver_profiled_floored_slopes",
) + tuple("val_relational_" + key for key in (
    "score_raw_min_singular_value", "score_q_raw_min_singular_value",
    "score_k_raw_min_singular_value", "score_rank_per_head", "content_rank_per_head",
    "parameter_count", "content_max_singular_value", "content_effective_rank",
    "content_T_min_singular_value",
    "projected_cov_min_eigenvalue", "projected_cov_max_condition", "covariance_repair_count",
    "projected_cov_max_condition_layer_max",
    # covariance scale, the scale-free Gram-normalized lambda_min and the rank-one residual of the last layer,
    # and their layer extremes (VFEModel._covariance_scale_statistics; audit 2026-10-10 round 2, items 3, 4, 13),
    # and the rank-one residual of the reference means (audit 2026-10-16 F45)
    "projected_cov_scale", "projected_cov_gram_min_eigenvalue", "comparison_map_norm_spread",
    "input_cov_scale_ratio", "rank_one_residual", "rank_one_residual_reference",
    "projected_cov_min_eigenvalue_layer_min", "projected_cov_min_eigenvalue_argmin_layer",
    "projected_cov_scale_layer_min", "projected_cov_scale_argmin_layer",
    "projected_cov_gram_min_eigenvalue_layer_min", "projected_cov_gram_min_eigenvalue_argmin_layer",
    "input_cov_scale_ratio_layer_min", "input_cov_scale_ratio_argmin_layer",
    "comparison_map_norm_spread_layer_max",
    # self-lock and capped-entry locations (round 2, items 4 and 7)
    "link_unit_distance_min", "self_lock_unit_distance",
    "score_klmax_frac_head_argmax_layer", "score_klmax_frac_head_argmax_head",
    # audit 2026-10-17: the projected-covariance minimum's argmin position and minimum over positions >= 1
    # (item 13), and the diagonal projection loss and reference-trace ratio with their layer maxima (items 3, 18)
    "projected_cov_min_eigenvalue_argmin_position", "projected_cov_min_eigenvalue_from_position_1",
    "diagonal_projection_kl", "diagonal_projection_kl_max", "diagonal_projection_trace_ratio",
    "diagonal_projection_trace_ratio_max", "diagonal_projection_kl_layer_max",
    "diagonal_projection_kl_argmax_layer", "diagonal_projection_trace_ratio_layer_max",
    "diagonal_projection_trace_ratio_argmax_layer")
) + tuple(f"val_{key}" for key in LINK_SPECTRUM_KEYS)

# Edge-link probes (content_link_edges != 'off' only; edge_links.edge_diagnostic_capture), appended to the
# relational columns of an edge-enabled run so the metrics schema of every other run is unchanged.
_VAL_EDGE_DIAG_KEYS = tuple("val_relational_edge_" + key for key in (
    "part_norm_mean", "part_norm_max", "sender_norm_mean", "sender_norm_max", "receiver_norm_mean",
    # 1.0: the val_relational_content_* and val_relational_link_* columns of this run (except the edge-aware
    # self-loop radius) describe the shared generator e^{L}, not the context-dependent links.
    "link_spectrum_shared_generator"))
# Latent receiver-factor inference (content_link_latent only; edge_links.LATENT_DIAGNOSTIC_KEYS): the MAP energy
# at A^0 and at A^(T), the loop energy (NaN with the loop prior off) and ||A^(T) - A^0|| / ||A^0||, receiver means.
_VAL_LATENT_DIAG_KEYS = tuple("val_relational_latent_" + key for key in (
    "energy_initial", "energy_final", "loop_energy", "receiver_shift"))
# Minimal-agent readout (predictive_readout='minimal_agent' only; vfe3.model.minimal_agent): the head-mean gain
# fraction tr(A Lambda_bar)/d (mean field tr(A M)/d at the input moments), the head-mean channel noise n_h and the
# head-max coupling ||S_h C S_g^T||_F / (||S_h C S_h^T||_F ||S_g C S_g^T||_F)^{1/2} at the input moments (0: ADF is
# exact), the head-mean mean-field label staleness (TV of one relabeling at the posterior; NaN under 'exact') and the
# covariance contraction tr C^+ / tr C and the head-mean exact spread share tr(A R) / tr(A Lambda_bar) (NaN under
# 'mean_field'), the head-mean mean-field M-step optimum n_h^* over the heads with an interior optimum
# (minimal_agent.channel_noise_mstep; NaN under 'exact') and the fraction of heads at its n -> 0 bracket.
_VAL_MINIMAL_AGENT_DIAG_KEYS = tuple("val_relational_minimal_agent_" + key for key in (
    "gain_fraction", "channel_noise", "head_coupling", "label_staleness", "covariance_contraction",
    "spread_share", "channel_noise_mstep", "channel_noise_mstep_boundary"))
# Under diagnose_agent_elbo (minimal agent only; audit 2026-10-12 T1 and round 2, item 12 and M1): the last layer's
# receiver-mean evidence lower bound of the posterior the readout returns, the ADF evidence of the record model
# (assumed density; exact only for uncoupled heads), the exact evidence (NaN unless every receiver's heads are
# uncoupled), the importance-sampled evidence under q, and the gaps to the bound, in nats.
_VAL_MINIMAL_AGENT_ELBO_KEYS = tuple("val_" + key for key in MINIMAL_AGENT_ELBO_KEYS)


# =============================================================================
# Training-time diagnostics for the multi-seed runs (audit 2026-09-28, Section 2).
#
# Every quantity below is computed only on a log/eval-cadence step, off the training graph, with the
# CPU and CUDA RNG streams restored. None of it runs on the silent path (log_interval and
# eval_interval falsy), and none of it feeds back into the optimizer, so the training trajectory is
# unchanged. A failure is logged once per diagnostic name, counted in
# ``RunArtifacts.diagnostics_failures`` (summary.json) and in the cumulative ``diagnostics_failures``
# metrics column, and training continues. Evaluations run only every ``eval_interval`` completed
# steps: there is no step-0 evaluation and no fixed early schedule.
# =============================================================================

# Fixed training sample for the generalization gap: windows of the TRAIN split drawn by a generator
# with this fixed seed (independent of cfg.seed and the data seed, so every seed of a multi-seed
# study scores the same windows), scored under the validation protocol, and sized to the validation
# split's scored-target count, capped here. The cap (32,768 targets, 1,024 windows at stride 32)
# binds on WikiText-103, so the sample costs about an eighth of a validation pass per evaluation
# instead of a second full pass (2026-10-07). Estimated, not measured: with a per-token CE spread of
# about 2.5 nats and tokens correlated within a window, its mean CE has a standard error of a few
# hundredths of a nat. The post-hoc evaluator keeps its own larger cap
# (vfe3.posthoc.streams.TRAIN_SAMPLE_MAX_TARGETS).
TRAIN_SAMPLE_SEED:        int = 0
TRAIN_SAMPLE_MAX_TARGETS: int = 32_768

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
# Per-element RMS realized step in units of the step size for the prior tables (audit 2026-10-09 round 2,
# item 26): update_ratio_group_* of a log-scale table divides by a norm dominated by the log(sigma_init)
# offset, so it is not a trainability measure; this column is O(1) for an AdamW table whose elements move.
# Under m_table_update_mode='fisher' the unit is the realized Fisher step size eta*lr_t/lr_0 (the row's
# lr_mu/lr_sigma), and the value scales with the gradient, the trust clip and the fraction of rows with
# n_v > 0, so the O(1) reading does not carry over (audit 2026-10-16, F13).
_STEP_LR_GROUPS: Tuple[str, ...] = ("mu_embed", "sigma_log_embed")
_STEP_LR_COLUMNS: Tuple[str, ...] = tuple(f"update_rms_per_lr_group_{group}" for group in _STEP_LR_GROUPS)
_TRAIN_SAMPLE_COLUMNS: Tuple[str, ...] = (
    "train_sample_ce", "train_sample_ppl", "generalization_gap_fixed", "train_sample_seen_fraction")
_GROUP_GRAD_ERROR_KEY = "_group_gradient_norm_error"     # private train_step -> train handoff

# Held-out columns the conditional route now fills from its own capture (all members of _VAL_DIAG_KEYS).
_CONDITIONAL_STRUCTURE_KEYS: Tuple[str, ...] = (
    "val_effective_rank", "val_belief_cond_median", "val_gauge_invariant_spread",
    "val_attn_entropy_min_all", "val_attn_entropy_min_norm",
    "val_future_leakage", "val_row_sum_error", "val_pos_content_r2", "val_prev_token_mass",
    "val_period_match_mass", "val_head_redundancy_js",
    "pos_loss_first_q", "pos_loss_last_q", "pos_loss_ratio",
)

RESIDUAL_RATIO_FIELDS: Tuple[str, ...] = (
    "step", "layer", "q10", "q50", "q90", "ln_rel_change_q50")
ATTENTION_STATS_FIELDS: Tuple[str, ...] = (
    "step", "layer", "head", "entropy_norm", "distance_mean", "self_weight", "prior_overlap", "n_rows")
# Head-sum link curvature of the routing (vfe3.metrics.routing_triangle_defect_sums): path-weighted
# mean and RMS of log rho_ijk = log(beta_ij beta_jk / beta_ik) over strict triangles, for the learned
# router and for its prior alone (zero for flat routing beta_ij = c_i / c_j); path_mass is the mean
# two-step path weight per query row.
ROUTING_HOLONOMY_FIELDS: Tuple[str, ...] = (
    "step", "layer", "head", "path_mass", "log_defect_mean", "log_defect_rms",
    "prior_log_defect_mean", "prior_log_defect_rms", "n_rows")
# Minimal-agent readout per (layer, head) (predictive_readout='minimal_agent'; vfe3.model.minimal_agent), means over
# the fixed validation windows: gain fraction, channel noise n_h, head coupling, mean-field label staleness (TV of one
# relabeling at the posterior; NaN under 'exact'), the layer's covariance contraction tr C^+ / tr C and the exact
# spread share tr(A R) / tr(A Lambda_bar) (the part of the removed variance the mean field drops; NaN under the mean field)
# and the mean-field M-step optimum n_h^* of the channel noise beside the learned channel_noise (NaN under 'exact'
# and at the n -> 0 bracket) with its bracket flag (1 at the bracket; audit 2026-10-12 round 2, M2).
MINIMAL_AGENT_HEAD_FIELDS: Tuple[str, ...] = (
    "step", "layer", "head", "gain_fraction", "channel_noise", "head_coupling", "label_staleness",
    "covariance_contraction", "spread_share", "channel_noise_mstep", "channel_noise_mstep_boundary")
_MINIMAL_AGENT_HEAD_CAPTURE = tuple("relational_minimal_agent_" + key for key in MINIMAL_AGENT_HEAD_FIELDS[3:])
# Prototype-agent block MLP per layer (block_mlp_mode='prototype_agents'), over the tokens of the fixed validation
# windows: effective templates in use exp(H(mean_t w_t)), the token-mean weight entropy, the fraction of templates
# whose mean weight is below 1/(10 M), log lambda quantiles over all template coordinates, H(pi), the gain
# fraction (share of the working variance the block removes) and the token-mean mean-field label staleness (TV of one
# relabeling at the last sweep's posterior; NaN under 'exact'; audit 2026-10-10 round 2, item 54).
PROTOTYPE_TEMPLATE_FIELDS: Tuple[str, ...] = (
    "step", "layer", "effective_templates", "weight_entropy", "dead_fraction", "log_lambda_min",
    "log_lambda_median", "log_lambda_max", "prior_entropy", "gain_fraction", "label_staleness", "n_tokens")
# Per-layer evidence diagnostics (diagnose_agent_elbo; audit 2026-10-12 T1), means over the receivers (tokens) of the
# fixed validation windows, in nats: the evidence lower bound of the posterior the layer forms, the evidence of its
# declared record model and their gap. Minimal agent (round 2, item 12 and M1): the ADF evidence over the heads
# (assumed density, exact for uncoupled heads) and its gap, the exact evidence (NaN unless every receiver's heads are
# uncoupled), and the importance-sampled evidence under q with its gap, the effective sample size fraction of its
# weights (is_ess, unitless; audit 2026-10-16 F47) and sample count. Prototypes: exact.
# Written to diagnostics/minimal_agent_elbo.csv and diagnostics/prototype_elbo.csv only when the toggle is on.
MINIMAL_AGENT_ELBO_FIELDS: Tuple[str, ...] = (
    "step", "layer", *(key[len("relational_minimal_agent_"):] for key in MINIMAL_AGENT_ELBO_KEYS), "is_samples")
PROTOTYPE_ELBO_FIELDS: Tuple[str, ...] = ("step", "layer", "elbo", "evidence", "elbo_gap", "n_tokens")

# S1 panel D numerical-event counters (GL(K)_attention_v2 fig:placeholder_numerics, 2026-09-28).
# Every count is an integer with its denominator beside it. NUMERICS_EVAL_COLUMNS are measured at
# every evaluation on the DIAGNOSTIC_WINDOWS fixed validation windows inside the
# residual-ratio capture pass (no extra forward), with one diagnostics/cap_activation.csv row per
# (layer, head); they are NaN on rows without an evaluation, when the run
# has no diagnostic windows, and when their capture pass fails. NUMERICS_CUMULATIVE_COLUMNS are on
# every metrics row and cover every attempted optimizer step since step 0 (frame_projection_* NaN
# without a post-update frame projection, the chart-norm ball or the condition bound). The exact
# definitions and skip rules are in _capture_numerics_counts, _numerics_counter_skip_reasons and
# _cumulative_counter_columns; the final values go to summary.json "numerics_counters" (_numerics_counters_summary).
NUMERICS_EVAL_COLUMNS: Tuple[str, ...] = (
    "cap_active_count", "cap_entry_count", "factorization_fallback_count", "factorization_count")
NUMERICS_CUMULATIVE_COLUMNS: Tuple[str, ...] = (
    "frame_projection_count", "frame_projection_total", "skipped_updates", "attempted_updates")
CAP_ACTIVATION_FIELDS: Tuple[str, ...] = ("step", "layer", "head", "capped", "entries")
# diagnostics/cap_clamped_rows.csv: per (layer, head), query rows (|S_i| >= 2) whose every source cost is
# capped, the only case in which the clamp changes beta (audit 2026-10-08 critic-gapfill-2#3).
CAP_CLAMPED_ROW_FIELDS: Tuple[str, ...] = ("step", "layer", "head", "clamped_rows", "rows")
# diagnostics/block_variance.csv: per layer and stage, quantiles over the diagnostic windows of the local variances
# diag(Sigma_i) and the median per-token mean rms ||mu_i|| / sqrt(K). Stages: "pre_norm" (the converged belief
# entering the block norm), "post_norm" (after the norm, the block MLP's input) and "block_out" (the layer output).
# Audit 2026-10-17 item 35: n_entries and n_nonfinite count the variance entries and those dropped as non-finite;
# var_min / var_max are taken before any subsampling; scale_* are quantiles of the centered LayerNorm scale
# s_i = ||Pi mu_i|| / sqrt(K) (Pi = I - 1 1^T / K, no epsilon) and trace_ratio_* of r_i = tr(Pi Sigma_i Pi) / (K s_i^2),
# the ratio that decides whether the delta-method pushforward of the norm is in its regime (r << 1).
# Audit 2026-10-20 (spike report section 7): var_q99 and var_q999 give the upper tail between the median and the
# maximum, so the share of entries with an exploded variance can be read off; they are appended last, and an older
# file is migrated to the wider header on the next append (RunArtifacts.append_diagnostic_rows).
BLOCK_VARIANCE_FIELDS: Tuple[str, ...] = (
    "step", "layer", "stage", "var_min", "var_q001", "var_q01", "var_median", "var_max", "mean_rms_median",
    "n_entries", "n_nonfinite", "scale_q01", "scale_median", "scale_q99",
    "trace_ratio_median", "trace_ratio_q99", "trace_ratio_max", "var_q99", "var_q999")
_BLOCK_VARIANCE_STAGES: Tuple[Tuple[str, str], ...] = (
    ("pre_norm", "layer_converged"), ("post_norm", "layer_block_mlp_inputs"), ("block_out", "layer_outputs"))
_EVAL_CADENCE_COLUMNS: Tuple[str, ...] = _TRAIN_SAMPLE_COLUMNS + NUMERICS_EVAL_COLUMNS


# Dead-router thresholds (audit 2026-10-09, round 2, item 9; audit 2026-10-10 round 2, item 7), read per
# (layer, head) under every clamp reference. Under the absolute reference a head with more than
# DEAD_ROUTER_CLAMPED_ROW_FRAC of its scored rows fully clamped at kl_max routes those rows by the attention prior pi
# alone (uniformly under a plain causal mask) with zero router gradient (layer 1 of the 32.65 run); under 'row_min' and under tie_comparison_maps no row can be fully
# clamped. A head whose mean self weight beta_ii exceeds DEAD_ROUTER_SELF_WEIGHT saturates its softmax on the
# self pair (layer 1 head 0 of the 99.68 run), whatever the reference. A head with more than
# DEAD_ROUTER_CAPPED_FRAC of its admissible entries capped gives those entries no router gradient; under
# 'row_min' each sits at least kl_max above its row's best source, so the rows route near one-hot (the live
# K = 128 row of 2026-10-10: 0.939 capped at self weight 0.379). A fully one-hot causal head at sequence length
# N caps (N - 1)/(N + 1) of its entries (0.984 at N = 128).
DEAD_ROUTER_CLAMPED_ROW_FRAC: float = 0.5
DEAD_ROUTER_SELF_WEIGHT:      float = 0.9
DEAD_ROUTER_CAPPED_FRAC:      float = 0.9
# Covariance-scale collapse (audit 2026-10-10 round 2, item 3): a layer whose router reads a covariance below this
# fraction of the stack-entry covariance (relational_input_cov_scale_ratio, a median over tokens of tr ratios).
COVARIANCE_COLLAPSE_RATIO:    float = 1e-3


def _finite_or_none(value: Any) -> Optional[float]:
    r"""``float(value)`` when it is present and finite, else None."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


WarningSink = Callable[[Dict[str, Any]], None]          # receives one warnings.jsonl record per emitted warning
# (kind, head-max key, argmax prefix, per-head table key of VFEModel.last_router_head_tables)
_ROUTER_HEAD_KINDS: Tuple[Tuple[str, str, str, str], ...] = (
    ("self_lock", "relational_self_weight_head_max", "relational_self_lock", "self_weight"),
    ("clamped", "relational_clamped_row_frac_head_max", "relational_clamped_row_frac", "clamped_row_frac"),
    ("capped", "relational_score_klmax_frac_head_max", "relational_score_klmax_frac_head", "klmax_frac"),
)


def _emit_warning(
    logger:  logging.Logger,
    sink:    Optional[WarningSink],
    kind:    str,
    message: str,
    **fields: Any,
) -> None:
    r"""Log ``message`` at WARNING and hand ``{kind, message, **fields}`` to ``sink`` (``warnings.jsonl``) when given."""
    logger.warning(message)
    if sink is not None:
        sink({"kind": kind, "message": message.strip(), **fields})


def _router_heads_above(
    table:     Any,                        # (L, H) per-head values, NaN-padded, or None
    threshold: float,
) -> List[Tuple[int, int, float]]:
    r"""Every (layer, head, value) of a per-head table above ``threshold``, in (layer, head) order."""
    if table is None:
        return []
    values = torch.as_tensor(table, dtype=torch.float64).cpu()
    if values.dim() != 2:
        return []
    out: List[Tuple[int, int, float]] = []
    for layer in range(values.shape[0]):
        for head in range(values.shape[1]):
            value = float(values[layer, head])
            if math.isfinite(value) and value > threshold:
                out.append((layer, head, value))
    return out


def _warn_dead_router(
    diagnostics: Mapping[str, Any],
    warned:      Set[Tuple[str, int, int]],   # (kind, layer, head) already reported; updated in place
    logger:      logging.Logger,

    *,
    reference:   str  = "absolute",           # cfg.routing_kl_max_reference
    tied:        bool = False,                # cfg.tie_comparison_maps
    head_tables: Optional[Mapping[str, Any]] = None,   # VFEModel.last_router_head_tables of the same call
    sink:        Optional[WarningSink] = None,
) -> None:
    r"""Warn once per (kind, layer, head) when a router head is saturated or dead, reading every signal.

    Each signal is read under every clamp reference (``VFEModel._layer_guard_maxima``; audit 2026-10-10 round 2,
    item 7). With ``head_tables`` (the per-(layer, head) tables of the same diagnostics call) every head above the
    threshold is named (audit 2026-10-17 item 34); without them only the (layer, head) attaining the maximum is,
    as before:

    ``self_lock``: ``relational_self_weight_head_max`` above :data:`DEAD_ROUTER_SELF_WEIGHT`, at
    ``relational_self_lock_argmax_{layer,head}``. The softmax saturates on the self pair (a gap of a few tens of
    nats, far below ``kl_max``, suffices), so the head routes near one-hot to self and the gradient
    :math:`\partial\beta_{ii}/\partial D_{ij} = \beta_{ii}\beta_{ij}/\tau` that could unlock it vanishes with
    :math:`\beta_{ij}`. The cause is saturation, not the cap. The message carries the self-map identity distance
    :math:`\rho(\hat T_h\hat O_h - I)` of that head (``relational_self_lock_unit_distance`` at the attaining head),
    which is near zero for a self-locked identity head and separates it from a locked head that does not pass
    content through.

    ``clamped``: ``relational_clamped_row_frac_head_max`` above :data:`DEAD_ROUTER_CLAMPED_ROW_FRAC`, at
    ``relational_clamped_row_frac_argmax_{layer,head}``: fully clamped rows route by the attention prior pi alone,
    with zero router gradient. Only the absolute reference without the tie can clamp a whole row; the value is zero otherwise.

    ``capped``: ``relational_score_klmax_frac_head_max`` above :data:`DEAD_ROUTER_CAPPED_FRAC`, at
    ``relational_score_klmax_frac_head_argmax_{layer,head}``: the capped entries carry a constant cost and get no
    router gradient; under ``'row_min'`` each sits at least ``kl_max`` above its row's best source, so the rows
    route near one-hot to their best sources, whether or not that source is self.

    A key the model does not report is skipped. Each warning is also handed to ``sink`` (``warnings.jsonl``).
    """
    clamp = ("tie_comparison_maps" if tied else f"routing_kl_max_reference={reference!r}")
    thresholds = {"self_lock": DEAD_ROUTER_SELF_WEIGHT, "clamped": DEAD_ROUTER_CLAMPED_ROW_FRAC,
                  "capped": DEAD_ROUTER_CAPPED_FRAC}
    distances = head_tables.get("unit_distance") if head_tables else None
    for kind, value_key, prefix, table_key in _ROUTER_HEAD_KINDS:
        threshold = thresholds[kind]
        value = _finite_or_none(diagnostics.get(value_key))
        if value is None or value <= threshold:
            continue
        argmax = tuple(-1 if v is None else int(v) for v in (
            _finite_or_none(diagnostics.get(f"{prefix}_argmax_{axis}")) for axis in ("layer", "head")))
        heads = _router_heads_above(head_tables.get(table_key) if head_tables else None, threshold)
        if not heads:                                         # no per-head table: the attaining head only
            heads = [(argmax[0], argmax[1], value)]
        for layer, head, head_value in heads:
            if (kind, layer, head) in warned:
                continue
            warned.add((kind, layer, head))
            where = (layer if layer >= 0 else "unknown", head if head >= 0 else "unknown")
            if kind == "self_lock":
                distance = None
                if (layer, head) == argmax:
                    distance = _finite_or_none(diagnostics.get("relational_self_lock_unit_distance"))
                elif distances is not None and layer >= 0 and head >= 0:
                    try:
                        distance = _finite_or_none(torch.as_tensor(distances, dtype=torch.float64)[layer, head])
                    except (IndexError, RuntimeError):
                        distance = None
                message = (
                    " router layer %s head %s: mean self weight beta_ii %.3f (> %.2f) under %s; the softmax saturates "
                    "on the self pair, so the head routes near one-hot to self and its off-self keys get a vanishing "
                    "router gradient (self-map identity distance rho(T O' - I) = %s)"
                    % (*where, head_value, threshold, clamp, "n/a" if distance is None else f"{distance:.3g}"))
            elif kind == "clamped":
                message = (
                    " router layer %s head %s: %.3f of its scored rows are fully clamped at kl_max (> %.2f); those rows "
                    "route by the attention prior alone with zero router gradient "
                    "(consider routing_kl_max_reference='row_min')" % (*where, head_value, threshold))
            else:
                message = (
                    " router layer %s head %s: %.3f of its admissible costs are capped at kl_max%s (> %.2f) under %s; "
                    "the capped entries get no router gradient%s" % (
                        *where, head_value, " above the row minimum" if reference == "row_min" else "", threshold,
                        clamp, ", so its rows route near one-hot to their best sources" if reference == "row_min" else ""))
            _emit_warning(logger, sink, f"dead_router_{kind}", message, layer=layer, head=head, value=head_value,
                          threshold=threshold)


def _warn_covariance_collapse(
    diagnostics: Mapping[str, Any],
    warned:      Set[Tuple[str, int, int]],   # (kind, layer, head) already reported; updated in place
    logger:      logging.Logger,

    *,
    sink:        Optional[WarningSink] = None,
) -> None:
    r"""Warn once per layer when a layer's router reads a covariance orders of magnitude below the stack entry.

    Reads ``relational_input_cov_scale_ratio_layer_min`` and its ``_argmin_layer``
    (``VFEModel._covariance_scale_statistics``; audit 2026-10-10 round 2, item 3): the median over tokens of
    :math:`\operatorname{tr}\Sigma_i/\operatorname{tr}\Sigma^{(0)}_i` of the belief the router read against the
    stack-entry belief, warned below :data:`COVARIANCE_COLLAPSE_RATIO`. In a residual-free stack whose block
    norm passes covariance through, nothing restores an absolute scale, and the trace-relative predictive floor
    :math:`f\operatorname{tr}C_i/K` shrinks with the covariance it floors. The ratio does not read the comparison
    maps. The message adds the layer minima of the projected scale :math:`\operatorname{tr}G/r` and of the
    Gram-normalized :math:`\lambda_{\min}` (item 13): a collapse of the covariance lowers all three, while a
    drift of the comparison maps along their free orbit lowers only the projected scale. A model that does not
    report the keys is skipped. A trace median is dominated by the large variances, so an eigenvalue collapse is
    warned separately by :func:`_warn_eigenvalue_collapse` (audit 2026-10-17 item 14).
    """
    value = _finite_or_none(diagnostics.get("relational_input_cov_scale_ratio_layer_min"))
    if value is None or value >= COVARIANCE_COLLAPSE_RATIO:
        return
    layer = _finite_or_none(diagnostics.get("relational_input_cov_scale_ratio_argmin_layer"))
    layer = -1 if layer is None else int(layer)
    if ("covariance_scale", layer, -1) in warned:
        return
    warned.add(("covariance_scale", layer, -1))
    message = (
        " covariance layer %s: the router reads tr(Sigma) at %.3g of the stack entry (median over tokens, "
        "< %.0e); the trace-relative floor cannot bound this contraction (layer minima: projected scale "
        "tr(G)/r %.3g, Gram-normalized lambda_min %.3g)" % (
            layer if layer >= 0 else "unknown", value, COVARIANCE_COLLAPSE_RATIO,
            float(diagnostics.get("relational_projected_cov_scale_layer_min", float("nan"))),
            float(diagnostics.get("relational_projected_cov_gram_min_eigenvalue_layer_min", float("nan")))))
    _emit_warning(logger, sink, "covariance_scale", message, layer=layer, value=value,
                  threshold=COVARIANCE_COLLAPSE_RATIO)


# Eigenvalue-collapse alarms (audit 2026-10-17 item 14). 41.53's belief_lam_min fell from 4.3e-6 to 2.0e-14 between
# steps 500 and 1000 while the trace-median ratio of _warn_covariance_collapse never crossed its threshold.
EIGENVALUE_COLLAPSE_RATIO:    float = 1e3 * float(torch.finfo(torch.float32).eps)   # lam_min / lam_max below this
TABLE_STEP_COLLAPSE_FACTOR:   float = 100.0      # table_fisher_step_norm_mean falling by more within one log interval


def _covariance_floor_reference(
    model: torch.nn.Module,
) -> Optional[float]:
    r"""The absolute moment-matched floor :math:`f\min_{v,k}s_{vk}` of the prior table, or None.

    Under ``predictive_covariance='moment_matched'`` with ``predictive_covariance_floor_mode='prior'`` the readout
    adds :math:`f\Sigma_p` to the predictive covariance, so its smallest eigenvalue cannot fall below
    :math:`f\lambda_{\min}(\Sigma_p) \ge f\min_{v,k}s_{vk}` for the diagonal prior table. Under ``'max'`` the floor
    is the larger of the relative floor and :math:`f\Sigma_p` (``RelationalOperators.covariance_floor``), so the same
    lower bound holds and the alarm is armed there too (audit 2026-10-20 item 14). The projected covariance the
    router reads is that covariance carried by the comparison maps, so the comparison against this value is an alarm
    in the reference coordinates, not an exact bound. None under the relative floor (no absolute value exists),
    another readout, another table, or another model family. One host read, at log cadence.
    """
    cfg = getattr(model, "cfg", None)
    if (cfg is None or getattr(cfg, "predictive_covariance", None) != "moment_matched"
            or getattr(cfg, "predictive_covariance_floor_mode", None) not in ("prior", "max")
            or getattr(cfg, "prior_covariance_table", "diagonal") != "diagonal"):
        return None
    try:
        from vfe3.numerics import bounded_variance_from_log
        bank = model.prior_bank
        with torch.no_grad():
            variances = bounded_variance_from_log(bank._prior_sigma_log_table().double(), eps=bank.eps)
        smallest = float(variances.min())
    except Exception:                                         # noqa: BLE001 - an alarm, never fatal
        return None
    floor = float(cfg.predictive_covariance_floor) * smallest
    return floor if math.isfinite(floor) and floor > 0.0 else None


def _warn_eigenvalue_collapse(
    diagnostics:     Mapping[str, Any],
    warned:          Set[Tuple[str, int, int]],   # (kind, layer, head) already reported; updated in place
    logger:          logging.Logger,

    *,
    belief_layer:    int,                         # the layer belief_lam_min / belief_lam_max are read at
    floor_reference: Optional[float] = None,      # _covariance_floor_reference(model)
    sink:            Optional[WarningSink] = None,
    layer_extremes:  Optional[Mapping[str, Sequence[float]]] = None,   # VFEModel.last_belief_layer_extremes
) -> None:
    r"""Warn once per layer on an eigenvalue collapse that a trace ratio cannot see (audit 2026-10-17 item 14).

    ``belief_eigenvalue_ratio``: :math:`\lambda_{\min}/\lambda_{\max}` of the converged belief covariances below
    :data:`EIGENVALUE_COLLAPSE_RATIO` :math:`= 10^3\epsilon_{32}`, where float32 arithmetic on that covariance has lost
    all but about three digits of its smallest direction. With ``layer_extremes`` (``{"lam_min": [...], "lam_max":
    [...]}``, one entry per layer) every layer is tested (audit 2026-10-20 item 14: an early layer at 3.5e-5 was never
    warned because only the last layer was read); without it the unsuffixed ``belief_lam_min`` / ``belief_lam_max``
    are read as layer ``belief_layer``.
    ``projected_floor``: ``relational_projected_cov_min_eigenvalue_layer_min`` below ``floor_reference``
    (:func:`_covariance_floor_reference`), at its ``_argmin_layer``.
    """
    if layer_extremes is not None:
        pairs = list(enumerate(zip(layer_extremes.get("lam_min", ()), layer_extremes.get("lam_max", ()))))
    else:
        pairs = [(int(belief_layer), (diagnostics.get("belief_lam_min"), diagnostics.get("belief_lam_max")))]
    for layer, (low, high) in pairs:
        lam_min, lam_max = _finite_or_none(low), _finite_or_none(high)
        if lam_min is None or lam_max is None or lam_max <= 0.0:
            continue
        ratio = lam_min / lam_max
        key = ("belief_eigenvalue_ratio", int(layer), -1)
        if ratio < EIGENVALUE_COLLAPSE_RATIO and key not in warned:
            warned.add(key)
            message = (" covariance layer %d: converged belief lambda_min / lambda_max = %.3g (lambda_min %.3g, "
                       "lambda_max %.3g) is below 1e3 eps32 = %.3g; the smallest covariance direction has collapsed"
                       % (layer, ratio, lam_min, lam_max, EIGENVALUE_COLLAPSE_RATIO))
            _emit_warning(logger, sink, "belief_eigenvalue_ratio", message, layer=int(layer), value=ratio,
                          threshold=EIGENVALUE_COLLAPSE_RATIO)
    projected = _finite_or_none(diagnostics.get("relational_projected_cov_min_eigenvalue_layer_min"))
    if projected is not None and floor_reference is not None and projected < floor_reference:
        layer = _finite_or_none(diagnostics.get("relational_projected_cov_min_eigenvalue_argmin_layer"))
        layer = -1 if layer is None else int(layer)
        key = ("projected_floor", layer, -1)
        if key not in warned:
            warned.add(key)
            message = (" covariance layer %s: projected-covariance lambda_min %.3g is below the moment-matched floor "
                       "f min s_v = %.3g of the prior table" % (layer if layer >= 0 else "unknown", projected,
                                                                 floor_reference))
            _emit_warning(logger, sink, "projected_floor", message, layer=layer, value=projected,
                          threshold=floor_reference)


def _warn_table_step_collapse(
    previous: Optional[float],                    # table_fisher_step_norm_mean of the previous log row
    current:  Optional[float],                    # ... of this row
    logger:   logging.Logger,

    *,
    step:     int,
    sink:     Optional[WarningSink] = None,
) -> bool:
    r"""Warn when the realized table Fisher step norm fell by more than :data:`TABLE_STEP_COLLAPSE_FACTOR` within one
    log interval (41.53: 0.136 to 7.2e-6 while the covariance collapsed; audit 2026-10-17 item 14). Returns
    whether it warned."""
    previous, current = _finite_or_none(previous), _finite_or_none(current)
    if previous is None or current is None or previous <= 0.0 or current < 0.0:
        return False
    if current * TABLE_STEP_COLLAPSE_FACTOR >= previous:
        return False
    message = (" step %d: table_fisher_step_norm_mean fell from %.3g to %.3g (more than %.0fx) within one log "
               "interval; the type-prior tables have nearly stopped moving" % (
                   step, previous, current, TABLE_STEP_COLLAPSE_FACTOR))
    _emit_warning(logger, sink, "table_step_collapse", message, previous=previous, value=current,
                  factor=TABLE_STEP_COLLAPSE_FACTOR)
    return True


RUN_WARNINGS_LOG = "warnings.jsonl"                    # run_dir: one JSON line per emitted training warning


def _append_run_warning(
    run_dir: Path,
    record:  Mapping[str, Any],
) -> Path:
    r"""Append one JSON line to ``run_dir/warnings.jsonl`` (non-finite floats written as null)."""
    path = Path(run_dir) / RUN_WARNINGS_LOG
    clean = {key: (None if isinstance(value, float) and not math.isfinite(value) else value)
             for key, value in record.items()}
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(clean, sort_keys=True, default=str) + "\n")
    return path


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
    :func:`train_step` runs a post-update frame projection, i.e. ``m_phi_update_mode='adamw'`` with
    ``phi_mstep_max_matrix_norm`` set, or ``phi_frame_condition_max`` set on a group the condition bound acts
    on; otherwise both stay NaN. ``skipped_updates`` and ``attempted_updates`` are defined on every route.
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
    if get_phi_update_policy(mode).requires_manifold_optimizer:
        reasons["frame_projection"] = (
            f"m_phi_update_mode={mode!r}: no post-update frame projection (the pullback-group step "
            "rejects rows outside the chart bound instead of projecting them)")
    elif cfg.phi_mstep_max_matrix_norm is None and not (
            cfg.phi_frame_condition_max is not None and model.group.name in FRAME_CONDITION_GROUPS):
        reasons["frame_projection"] = (
            "phi_mstep_max_matrix_norm and phi_frame_condition_max are None (or the condition bound is inert "
            "for this gauge group): the run has no post-update frame projection")
    return reasons


@torch.no_grad()
def _capture_numerics_counts(
    diagnostic: Mapping[str, Any],
    kl_max:     float,

    *,
    rows_out:   Optional[List[List[Tuple[int, int]]]] = None,
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
    With ``kl_max = inf`` only non-finite costs count as capped; a minimal-agent layer (``predictive_readout``,
    whose posterior applies no clamp) is counted under that rule.

    Fully clamped rows. A capped entry changes :math:`\beta` only when its row's smallest cost reaches
    :math:`\kappa`; a row with :math:`|S_i| \ge 2` whose every entry is capped routes by the attention prior
    :math:`\pi` alone with zero routing gradient. When ``rows_out`` is given, one list per layer of
    ``(clamped_rows, rows)`` per head is appended to it, with ``rows`` the number of query rows with
    :math:`|S_i| \ge 2` (audit 2026-10-08 critic-gapfill-2#3). The entry counts above are unchanged.

    Factorizations. Each layer factors its reduced query and key covariances
    :math:`G = L \Sigma L^\top` (:math:`L = S U^{-1}`, comparison rank :math:`r`) once per (window,
    token, head) and role, so the layer's denominator is
    :math:`\mathrm{numel}(G_q)/r^2 + \mathrm{numel}(G_k)/r^2`. The factorization uses zero jitter and
    no spectral repair; its only fallback is the source-factor recovery of
    :func:`vfe3.relational_scoring._strict_cholesky` (float32 shifted CholeskyQR3 of the source root
    :math:`L \Sigma^{1/2}`, with float64 QR when that fails; float64 QR for every rebuilt entry under
    ``cholesky_qr2_certified``) for an entry whose working-precision factor
    failed, or failed the conditioning certificate under a certified ``projected_factorization`` (on
    either factor route). The numerator is the layer's recovered-entry count
    ``relational_covariance_repair_count``, a subset of the same entries. The pair-level factorizations
    of :math:`G_{q,i} + G_{k,j}` on the noisy-query route have no fallback (a failure raises) and are
    not counted. The minimal agent's fallback is the float64 recomputation of a whole float32 layer call
    whose factorizations failed (or whose ADF query covariance failed the certificate under
    a certified ``projected_factorization``); it counts every recorded query and key entry of that call, so
    there the ratio is the share of recomputed layer calls (plus, for the mean field under the certificate,
    the KL factors it refactors in calls that were not recomputed).

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
        # The minimal-agent posterior applies no clamp: the kl_max = inf rule (only a non-finite cost counts).
        cap_level = math.inf if "relational_minimal_agent_channel_noise" in capture else kl_max
        capped = entries & (score >= cap_level)                             # (B, H, N, N)
        rows_h = rows.squeeze(-1)                                           # (B, H, N)
        clamped_rows = rows_h & (capped.sum(dim=-1) == entries.sum(dim=-1))  # (B, H, N) every entry capped
        G_q = capture["projected_covariance_pre"]                           # (B, N, H, r, r); per run if mixed
        G_k = capture["projected_key_covariance_pre"]                       # (B, N, H, r, r); per run if mixed
        attempted.append(sum(int(G.numel() // (G.shape[-1] ** 2))
                             for G in ([G_q, G_k] if torch.is_tensor(G_q) else [*G_q, *G_k])))
        recovered = torch.as_tensor(capture["relational_covariance_repair_count"]).reshape(1)
        heads.append(int(beta.shape[1]))
        device_counts.append(torch.cat((
            capped.sum(dim=(0, 2, 3)), entries.sum(dim=(0, 2, 3)),          # (H,), (H,)
            clamped_rows.sum(dim=(0, 2)), rows_h.sum(dim=(0, 2)),           # (H,), (H,)
            recovered.to(device=beta.device, dtype=torch.long))))          # (1,)
    values = torch.cat(device_counts).cpu().tolist()                         # one host transfer
    cap: List[List[Tuple[int, int]]] = []
    factorization: List[Tuple[int, int]] = []
    offset = 0
    for n_heads, n_attempted in zip(heads, attempted):
        capped_h = values[offset:offset + n_heads]
        entries_h = values[offset + n_heads:offset + 2 * n_heads]
        clamped_h = values[offset + 2 * n_heads:offset + 3 * n_heads]
        rows_total_h = values[offset + 3 * n_heads:offset + 4 * n_heads]
        cap.append([(int(c), int(e)) for c, e in zip(capped_h, entries_h)])
        if rows_out is not None:
            rows_out.append([(int(c), int(r)) for c, r in zip(clamped_h, rows_total_h)])
        factorization.append((int(values[offset + 4 * n_heads]), n_attempted))
        offset += 4 * n_heads + 1
    return cap, factorization


def _merge_numerics_counts(
    accumulator:   Dict[str, Any],
    cap:           List[List[Tuple[int, int]]],
    factorization: List[Tuple[int, int]],
    clamped_rows:  Optional[List[List[Tuple[int, int]]]] = None,
) -> None:
    r"""Add one chunk's counts to the per-evaluation accumulator (keys ``cap``, ``factorization``, ``clamped_rows``)."""
    if clamped_rows is not None:
        per_row = accumulator.setdefault("clamped_rows", {})
        for layer, layer_counts in enumerate(clamped_rows):
            for head, (clamped, rows) in enumerate(layer_counts):
                total = per_row.setdefault((layer, head), [0, 0])
                total[0] += clamped
                total[1] += rows
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


def _cap_clamped_row_rows(
    accumulator: Mapping[str, Any],
    step:        int,
) -> List[Dict[str, int]]:
    r"""``cap_clamped_rows.csv`` rows (``CAP_CLAMPED_ROW_FIELDS``) from one evaluation's accumulator."""
    return [{"step": int(step), "layer": int(layer), "head": int(head),
             "clamped_rows": int(clamped), "rows": int(rows)}
            for (layer, head), (clamped, rows) in sorted((accumulator.get("clamped_rows") or {}).items())]


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


# Run-total numerical-event counters (audit 2026-10-09 round 2, item 35). The process-wide element
# counters reset at the start of train(); a resumed run adds the totals its checkpoint carried in
# numerics_counters_state, so metrics.csv and summary.json realized_updates stay whole-run totals.
RUN_TOTAL_COUNTER_KEYS: Tuple[str, ...] = (
    "nonfinite_tangent_elements", "mu_trust_fallback_elements", "decode_logdet_fallback_elements",
    "frame_pair_fp64_escalations", "public_spd_repair_elements", "public_spd_ceiling_contact_elements",
    "source_recovery_elements", "layernorm_trace_cap_elements", "covariance_ceiling_elements",
    "covariance_escalation_elements", "gemm_fallback_slices", "phi_rejected_rows", "matmul_precision_replays")


def _process_counter_values() -> Dict[str, int]:
    r"""Current values of the process-wide element counters since their last reset (no precision replays)."""
    return {"nonfinite_tangent_elements": int(nonfinite_tangent_elements()),
            "mu_trust_fallback_elements": int(mu_trust_fallback_elements()),
            "decode_logdet_fallback_elements": int(decode_logdet_fallback_elements()),
            "frame_pair_fp64_escalations": int(frame_pair_fp64_escalations()),
            "public_spd_repair_elements": int(public_spd_repair_elements()),
            "public_spd_ceiling_contact_elements": int(public_spd_ceiling_contact_elements()),
            "source_recovery_elements": int(source_recovery_elements()),
            "layernorm_trace_cap_elements": int(layernorm_trace_cap_elements()),
            "covariance_ceiling_elements": int(covariance_ceiling_elements()),
            "covariance_escalation_elements": int(covariance_escalation_elements()),
            "gemm_fallback_slices": int(gemm_fallback_slices())}


def _resumed_run_totals(
    resumed:    Optional[Mapping[str, Any]],   # artifacts.resumed_numerics_counters
    start_step: int,
    logger:     logging.Logger,
) -> Dict[str, int]:
    r"""Run totals a resumed checkpoint carried (0 for a fresh run, and for a key the checkpoint lacks)."""
    base = {key: 0 for key in RUN_TOTAL_COUNTER_KEYS}
    if start_step <= 0:
        return base
    carried = resumed if isinstance(resumed, Mapping) else {}
    missing = [key for key in RUN_TOTAL_COUNTER_KEYS
               if not (type(carried.get(key)) is int and carried[key] >= 0)]
    for key in RUN_TOTAL_COUNTER_KEYS:
        if key not in missing:
            base[key] = int(carried[key])
    if missing:
        logger.info(" resumed checkpoint carries no run total for %s; those counters count from step %d",
                    ", ".join(missing), start_step)
    return base


def _run_total_counters(
    base:               Mapping[str, int],     # _resumed_run_totals
    precision_replays:  int,                   # whole-run count (already includes its base)

    *,
    phi_rejected_rows:  int = 0,               # GaugeManifoldAdamW.phi_rejected_rows of this process
) -> Dict[str, int]:
    r"""Whole-run values of ``RUN_TOTAL_COUNTER_KEYS``: the resumed base plus this process's counts.

    The optimizer's rejected pullback-group rows restart at 0 after a resume (the count is not optimizer state),
    so the whole-run total adds the count the checkpoint carried (audit 2026-10-16 F126).
    """
    current = _process_counter_values()
    out = {key: int(base[key]) + current[key] for key in current}
    out["phi_rejected_rows"] = int(base["phi_rejected_rows"]) + int(phi_rejected_rows)
    out["matmul_precision_replays"] = int(precision_replays)
    return out


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
    ``prior_bank.phi_embed`` and of the learned positional table ``pos_phi_free``) that a post-update
    projection changed, and the rows it checked, summed over every projection call, i.e. over the accepted
    updates. The projections are the one onto the embedded Frobenius ball of radius
    ``phi_mstep_max_matrix_norm`` (a row rescaled with scale :math:`< 1`) and the condition bound
    ``phi_frame_condition_max`` (a row whose traceless symmetric spread was clamped); when both run, each
    counts its own rows. They accumulate on the device inside
    :func:`vfe3.gauge_optim.project_phi_parameter_rows_` and
    :func:`vfe3.gauge_optim.bound_phi_frame_condition_` and are read here, on a metrics row only.
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

    ``sigma_log_embed`` holds the token-prior covariance table, ``sigma_log_embed`` or (under
    ``prior_covariance_table='block_cholesky'``) ``sigma_chol_embed``;
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
    if name in ("prior_bank.sigma_log_embed", "prior_bank.sigma_chol_embed"):
        return "sigma_log_embed"                  # sigma_chol_embed: the block_cholesky prior covariance table
    if name == "prior_bank.phi_embed":
        return "phi_embed"
    if name == "pos_phi_free":
        return "pos_phi"
    if name.startswith("prior_bank.") and leaf.startswith(("output_proj", "decode_")):
        return "decode"
    if name.startswith("relational_operators"):
        # Mixed head sizes store per-run lists: the attribute is the segment before the run index.
        attribute = name.rsplit(".", 2)[-2] if leaf.isdigit() else leaf
        if attribute in ("raw_score_q", "raw_score_k", "channel_log_noise", "routing_log_temperature",
                         "routing_key_log_noise"):
            return "relational_score"                         # channel_log_noise: minimal agent
        if attribute in ("T", "O"):
            return "relational_content"
        if attribute.startswith("edge_"):                     # content_link_edges factor maps
            return "relational_content"
        return "other"
    if name.startswith("block_mlps."):
        return "block_mlp"
    if name.startswith("block_norm."):
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


def _parameter_learning_rates(
    model:     torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Dict[str, float]:
    r"""Current step size of every optimized parameter, by parameter name (read before the step).

    The group's AdamW rate, except for a group marked ``table_fisher``: under
    ``m_table_update_mode='fisher'`` that rate only schedules the table step and sets its decay, and the
    table moves at the realized Fisher step size :math:`\eta\,\mathrm{lr}_t/\mathrm{lr}_0`, the value
    ``lr_mu``/``lr_sigma`` report (audit 2026-10-16, F13).
    """
    eta = getattr(optimizer, "_table_fisher_lr", None)
    if eta is not None:
        from vfe3.table_optim import _schedule_factor

    def rate(group: Mapping[str, Any]) -> float:
        if eta is not None and group.get("table_fisher"):
            return float(eta) * _schedule_factor(group)                  # type: ignore[arg-type]
        return float(group["lr"])

    by_id = {id(parameter): rate(group) for group in optimizer.param_groups for parameter in group["params"]}
    return {name: by_id[id(parameter)] for name, parameter in model.named_parameters() if id(parameter) in by_id}


@torch.no_grad()
def _group_update_statistics(
    model:  torch.nn.Module,
    before: Mapping[str, torch.Tensor],       # _parameter_snapshot taken before the step

    *,
    learning_rates: Optional[Mapping[str, float]] = None,   # _parameter_learning_rates before the step
) -> Dict[str, float]:
    r"""Parameter norm and realized update ratio of each named group across one training step.

    .. math::
        \|\theta_G\| = \Big(\sum_{p \in G} \|p_t\|_2^2\Big)^{1/2},\qquad
        r_G = \frac{\big(\sum_{p \in G} \|p_{t+1} - p_t\|_2^2\big)^{1/2}}{\|\theta_G\|},

    with :math:`p_t` the pre-step value and :math:`p_{t+1}` the value after the optimizer step and any
    post-step projection (the realized update, including weight decay). A rejected step gives
    :math:`r_G = 0`. The difference is formed in float64 on the parameter's device in bounded chunks.

    With ``learning_rates``, the groups of ``_STEP_LR_GROUPS`` also report the per-element RMS step in
    learning-rate units,

    .. math::
        s_G = \Big(\frac{1}{n_G}\sum_{p \in G} \frac{\|p_{t+1} - p_t\|_2^2}{\eta_p^2}\Big)^{1/2},

    with :math:`n_G` the group's element count and :math:`\eta_p` the step's realized step size from
    :func:`_parameter_learning_rates` (NaN if any is 0): the AdamW learning rate, or under
    ``m_table_update_mode='fisher'`` the Fisher step size :math:`\eta\,\mathrm{lr}_t/\mathrm{lr}_0`.
    """
    delta_sq: Dict[str, float] = {}
    norm_sq: Dict[str, float] = {}
    step_sq: Dict[str, float] = {}
    step_numel: Dict[str, int] = {}
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
        if learning_rates is not None and group in _STEP_LR_GROUPS:
            rate = float(learning_rates.get(name, 0.0))
            step_sq[group] = step_sq.get(group, 0.0) + (float(delta) / rate ** 2 if rate > 0.0 else math.nan)
            step_numel[group] = step_numel.get(group, 0) + current.numel()
    out: Dict[str, float] = {
        f"update_rms_per_lr_group_{group}": (math.sqrt(step_sq[group] / step_numel[group])
                                            if step_numel.get(group) else float("nan"))
        for group in _STEP_LR_GROUPS}
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
def _sigma_log_statistics(
    model: torch.nn.Module,
) -> Dict[str, float]:
    r"""Mean and standard deviation of the whole prior log-variance table ``sigma_log_embed``.

    Reported instead of its norm: the norm decays toward 0 under weight decay, so a
    gradient-to-weight ratio built on it rises for a trivial reason. Population statistics
    (``correction=0``) in the table's own precision, with no full-size float64 copy on the device.

    Under ``prior_covariance_table='block_cholesky'`` (no ``sigma_log_embed``) the same statistics are
    taken over the prior log-variances :math:`\log (C_v)_{kk}`, read through the
    ``PriorBank._prior_covariance_blocks`` accessor in vocabulary chunks, so they do not depend on how
    the Cholesky table is parameterized (audit 2026-10-09 round 2, item 25).
    """
    bank = getattr(model, "prior_bank", None)
    table = getattr(bank, "sigma_log_embed", None)
    if table is None and isinstance(getattr(bank, "sigma_chol_embed", None), torch.Tensor):
        return _block_cholesky_log_variance_statistics(bank)
    if not isinstance(table, torch.Tensor) or table.numel() == 0:
        return {"sigma_log_mean": float("nan"), "sigma_log_sd": float("nan")}
    values = table.detach()
    if not values.is_floating_point() or values.element_size() < 4:
        values = values.float()
    sd, mean = torch.std_mean(values, correction=0)
    return {"sigma_log_mean": float(mean), "sigma_log_sd": float(sd)}


# Decode-table variance columns of every metrics row (audit 2026-10-16 F12); NaN under the linear readout.
DECODE_VARIANCE_COLUMNS: Tuple[str, ...] = (
    "decode_sigma_v_min", "decode_sigma_log_mean", "decode_sigma_log_sd")


@torch.no_grad()
def _decode_variance_statistics(
    model: torch.nn.Module,
) -> Dict[str, float]:
    r"""The variance margin of the table the prior-bank decoder scores against.

    ``decode_sigma_v_min`` is :meth:`PriorBank.decode_sigma_v_min`,
    :math:`\min_{v,k}\max(e^{\ell_{vk}}, \varepsilon)` over the decode log-variance table :math:`\ell`
    (``decode_sigma_log_embed`` when the bank is untied, else ``sigma_log_embed``), the number that sets
    how many digits the fp32 ``a_v`` reduction loses (``decode_av_precision``). The mean and standard
    deviation of :math:`\ell` (``correction=0``) sit beside it: on an untied bank ``sigma_log_mean`` and
    ``sigma_log_sd`` read the encoder table only, which does not see the decode table's lower tail.
    NaN under ``decode_mode='linear'``, whose readout reads no variance table.
    """
    nan = {column: float("nan") for column in DECODE_VARIANCE_COLUMNS}
    bank = getattr(model, "prior_bank", None)
    if bank is None or getattr(bank, "decode_mode", "linear") == "linear":
        return nan
    table = bank._decode_sigma_log_table()                                # (V, K) decode log-variances
    if not isinstance(table, torch.Tensor) or table.numel() == 0:
        return nan
    values = table.detach()
    if not values.is_floating_point() or values.element_size() < 4:
        values = values.float()
    sd, mean = torch.std_mean(values, correction=0)
    return {"decode_sigma_v_min": float(bank.decode_sigma_v_min()),
            "decode_sigma_log_mean": float(mean), "decode_sigma_log_sd": float(sd)}


SIGMA_LOG_CHUNK_TYPES: int = 4096          # vocabulary rows per block_cholesky covariance chunk


@torch.no_grad()
def _block_cholesky_log_variance_statistics(
    bank: torch.nn.Module,                 # PriorBank with prior_covariance_table='block_cholesky'
) -> Dict[str, float]:
    r"""Population mean and standard deviation of :math:`\log (C_v)_{kk}` over every type :math:`v` and
    coordinate :math:`k` of the per-head prior blocks, accumulated in float64 over vocabulary chunks."""
    n_types = int(bank.sigma_chol_embed.shape[0])
    total = torch.zeros((), dtype=torch.float64, device=bank.sigma_chol_embed.device)
    square = torch.zeros_like(total)
    count = 0
    with torch.autocast(device_type=bank.sigma_chol_embed.device.type, enabled=False):
        for start in range(0, n_types, SIGMA_LOG_CHUNK_TYPES):
            index = torch.arange(start, min(start + SIGMA_LOG_CHUNK_TYPES, n_types),
                                 device=bank.sigma_chol_embed.device)
            blocks = bank._prior_covariance_blocks(index)                     # (V_c, H, d, d) C_v per head
            log_variance = blocks.diagonal(dim1=-2, dim2=-1).double().log()   # (V_c, H, d)
            total += log_variance.sum()
            square += log_variance.square().sum()
            count += log_variance.numel()
    if count == 0:
        return {"sigma_log_mean": float("nan"), "sigma_log_sd": float("nan")}
    mean = total / count
    return {"sigma_log_mean": float(mean),
            "sigma_log_sd": float((square / count - mean.square()).clamp(min=0.0).sqrt())}


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
    if getattr(dataset, "window_offset_mode", "fixed") != "fixed":
        # Fixed windows and the fixed mode's window count, whatever this epoch's origin, so the sample
        # draws the same indices as under 'fixed' (audit 2026-10-09 round 2, item 34).
        dataset = dataset.at_origin()
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

    False only for the predictive-content kernel readout with ``attention_residual=False``, which replaces
    the mean by the predictive mean. The minimal-agent readout is additive whatever ``attention_residual`` (inert
    there) says: its posterior mean is the prior mean plus the gain correction, :math:`\xi_i + \sum_h\mathsf K_i^hu_i^h`.
    """
    cfg = model.cfg
    return not (getattr(cfg, "relational_readout_mode", None) == "predictive_content"
                and getattr(cfg, "predictive_readout", "kernel") == "kernel"
                and not getattr(cfg, "attention_residual", False))


@torch.no_grad()
def _residual_and_attention_rows(
    model:   VFEModel,
    windows: torch.Tensor,                    # (W, N) fixed validation input windows
    step:    int,
    device:  torch.device,

    *,
    counters_out: Optional[Dict[str, Any]] = None,
    holonomy_out: Optional[Dict[Any, Any]] = None,
    features_out: Optional[Dict[str, Any]] = None,
    variance_out: Optional[Dict[Tuple[int, str], List[Any]]] = None,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    r"""Per-layer residual-ratio quantiles and per-head attention statistics on fixed windows.

    ``variance_out``: when a dict is given, per (layer, stage) of :data:`_BLOCK_VARIANCE_STAGES` the local variances
    :math:`\mathrm{diag}(\Sigma_i)` and per-token mean rms values of the same captures are collected into it, for
    :func:`_block_variance_rows`.

    ``features_out``: when a dict is given, the minimal-agent captures are summed per (layer, head) under
    ``"minimal_agent"`` and, for ``prototype_agents`` block MLPs, the template weights and removed variance of the
    block inputs are summed per layer under ``"prototype"`` (the block's ``forward_moments`` is wrapped on the
    instance for this pass only), for :func:`_feature_rows`.

    ``holonomy_out``: when a dict is given, per (layer, head) the path-weighted triangle-defect sums
    of :func:`vfe3.metrics.routing_triangle_defect_sums` for the router and for its prior are
    accumulated into it (``[mass, sum x, sum x^2, prior mass, prior sum x, prior sum x^2, rows]``),
    for :func:`_routing_holonomy_rows`. An error is stored under ``"error"`` and stops the
    accumulation without disturbing the other rows.

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
    :math:`\delta` is NaN when the block has no normalization, when a CG coupling sits
    between the E-step and the norm (the norm then does not see :math:`\mu + m`), and for the
    predictive readout without ``attention_residual`` (no residual exists to remove).

    Attention: :math:`\beta` is the router the forward used (the captured ``beta_pre``). Over query rows with at least two admissible sources
    (support of the attention log-prior, or :math:`\beta > 0` without one), per layer and head, the
    means of :math:`H(\beta_{i\cdot})/\log|S_i|`, :math:`\sum_j \beta_{ij}|i-j|`, :math:`\beta_{ii}` and
    :math:`\sum_j \min(\beta_{ij}, \pi_{ij})` with :math:`\pi = \mathrm{softmax}_j(B)` the prior alone.
    """
    block_norm = model.block_norm
    ln_defined = (block_norm is not None
                  and model.cg_coupling is None and _residual_update_is_additive(model)
                  and getattr(model.cfg, "block_coordinates", "local") == "local")
    ratios: Dict[int, List[torch.Tensor]] = {}
    ln_changes: Dict[int, List[torch.Tensor]] = {}
    sums: Dict[Tuple[int, int], List[float]] = {}          # (layer, head) -> [H_norm, dist, self, overlap, n]
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    wrapped = (_wrap_prototype_blocks(model, features_out,
                                      elbo=bool(getattr(model.cfg, "diagnose_agent_elbo", False)))
               if features_out is not None else [])
    try:
        for chunk in windows.split(max(1, DIAGNOSTIC_CHUNK_WINDOWS)):
            ids = chunk.to(device)                                              # (B, N)
            capture: dict = {"diagnostic": {}}
            model.forward_beliefs(ids, capture=capture)
            diagnostic = capture["diagnostic"]
            if features_out is not None:
                _accumulate_minimal_agent(features_out, diagnostic)
            if counters_out is not None and "error" not in counters_out:
                try:
                    clamped_counts: List[List[Tuple[int, int]]] = []
                    cap_counts, factorization_counts = _capture_numerics_counts(
                        diagnostic, float(model.cfg.kl_max), rows_out=clamped_counts)
                    _merge_numerics_counts(counters_out, cap_counts, factorization_counts, clamped_counts)
                except Exception as exc:                  # noqa: BLE001 - reported by the caller
                    counters_out["error"] = exc
            n_layers = len(diagnostic["layer_converged"])
            if variance_out is not None:
                _collect_block_variances(diagnostic, variance_out)
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
                if holonomy_out is not None and "error" not in holonomy_out:
                    try:
                        baseline = torch.where(support, prior, torch.zeros_like(prior))
                        baseline = baseline / baseline.sum(dim=-1, keepdim=True).clamp_min(1e-30)
                        learned = routing_triangle_defect_sums(beta, support)           # (H, 3)
                        reference = routing_triangle_defect_sums(baseline, support)     # (H, 3)
                        n_rows = float(beta.shape[0] * n)
                        for head in range(learned.shape[0]):
                            values = learned[head].tolist() + reference[head].tolist() + [n_rows]
                            total = holonomy_out.setdefault((layer, head), [0.0] * 7)
                            for slot, value in enumerate(values):
                                total[slot] += value
                    except Exception as exc:              # noqa: BLE001 - reported by the caller
                        holonomy_out["error"] = exc
    finally:
        for module, was_training in modes:
            module.training = was_training
        for mlp in wrapped:
            del mlp.forward_moments                                            # back to the class method
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


def _collect_block_variances(
    diagnostic:   Mapping[str, Any],                     # one forward_beliefs diagnostic capture
    variance_out: Dict[Tuple[int, str], List[Any]],      # (layer, stage) -> [variances, rms, scale, ratio] chunks
) -> None:
    r"""Append one capture's per-token block statistics to ``variance_out`` for :func:`_block_variance_rows`.

    Per (layer, stage) of :data:`_BLOCK_VARIANCE_STAGES`: the local variances :math:`\mathrm{diag}(\Sigma_i)`, the
    uncentered mean rms :math:`\|\mu_i\|/\sqrt K`, the centered LayerNorm scale :math:`s_i = \|\Pi\mu_i\|/\sqrt K`
    and :math:`r_i = \operatorname{tr}(\Pi\Sigma_i\Pi)/(K s_i^2)` with
    :math:`\operatorname{tr}(\Pi\Sigma\Pi) = \operatorname{tr}\Sigma - \mathbf 1^\top\Sigma\mathbf 1/K`, all in float64.
    """
    for stage, key in _BLOCK_VARIANCE_STAGES:
        for layer, belief in enumerate(diagnostic.get(key) or ()):
            sigma = belief.sigma.detach().double()
            mu = belief.mu.detach().double()                                   # (B, N, K)
            K = mu.shape[-1]
            diagonal = sigma.dim() == mu.dim()
            variances = sigma if diagonal else torch.diagonal(sigma, dim1=-2, dim2=-1)   # (B, N, K) diag Sigma_i
            rms = mu.square().mean(dim=-1).sqrt()                              # (B, N) ||mu_i|| / sqrt(K)
            centered = mu - mu.mean(dim=-1, keepdim=True)
            scale_sq = centered.square().mean(dim=-1)                          # (B, N) s_i^2
            total = variances.sum(dim=-1) if diagonal else sigma.sum(dim=(-2, -1))   # (B, N) 1^T Sigma_i 1
            projected_trace = variances.sum(dim=-1) - total / K                # (B, N) tr(Pi Sigma_i Pi)
            ratio = projected_trace / (K * scale_sq)                           # (B, N) r_i
            slot = variance_out.setdefault((layer, stage), [[], [], [], []])
            for index, values in enumerate((variances, rms, scale_sq.sqrt(), ratio)):
                slot[index].append(values.reshape(-1).cpu())


def _quantiles_of_finite(
    values:    torch.Tensor,                 # (n,) float64
    levels:    Sequence[float],
) -> List[float]:
    r"""Quantiles of the finite entries (a seeded 2^24 subsample above ``torch.quantile``'s input limit); NaN if none."""
    values = values[torch.isfinite(values)]
    if not values.numel():
        return [float("nan")] * len(levels)
    if values.numel() > 2 ** 24:                                              # torch.quantile's input limit
        generator = torch.Generator().manual_seed(0)
        values = values[torch.randperm(values.numel(), generator=generator)[:2 ** 24]]
    return torch.quantile(values, torch.tensor(list(levels), dtype=torch.float64)).tolist()


def _block_variance_rows(
    variance_out: Mapping[Tuple[int, str], List[Any]],   # filled by _residual_and_attention_rows
    step:         int,
) -> List[Dict[str, Any]]:
    r"""``diagnostics/block_variance.csv`` rows (:data:`BLOCK_VARIANCE_FIELDS`), one per (layer, stage).

    The entry counts and the variance extremes are over every captured entry, before the subsample that
    ``torch.quantile`` needs above :math:`2^{24}` entries (audit 2026-10-17 item 35). A slot with only the variance
    and rms lists (an older caller) reports NaN for the scale and trace-ratio columns.
    """
    order = {stage: index for index, (stage, _) in enumerate(_BLOCK_VARIANCE_STAGES)}
    rows: List[Dict[str, Any]] = []
    nan = float("nan")
    for (layer, stage) in sorted(variance_out, key=lambda key: (key[0], order[key[1]])):
        slot = variance_out[(layer, stage)]
        variances = torch.cat(slot[0]).double()
        rms = torch.cat(slot[1]).double()
        finite = variances[torch.isfinite(variances)]
        quantiles = _quantiles_of_finite(finite, (0.001, 0.01, 0.5, 0.99, 0.999))
        scale = torch.cat(slot[2]).double() if len(slot) > 2 and slot[2] else None
        ratio = torch.cat(slot[3]).double() if len(slot) > 3 and slot[3] else None
        scale_q = _quantiles_of_finite(scale, (0.01, 0.5, 0.99)) if scale is not None else [nan] * 3
        ratio_q = _quantiles_of_finite(ratio, (0.5, 0.99)) if ratio is not None else [nan] * 2
        finite_ratio = ratio[torch.isfinite(ratio)] if ratio is not None else None
        rows.append({
            "step": int(step), "layer": int(layer), "stage": stage,
            "var_min": float(finite.min()) if finite.numel() else nan,
            "var_q001": quantiles[0], "var_q01": quantiles[1], "var_median": quantiles[2],
            "var_max": float(finite.max()) if finite.numel() else nan,
            "mean_rms_median": float(rms[torch.isfinite(rms)].median()) if rms.numel() else nan,
            "n_entries": int(variances.numel()),
            "n_nonfinite": int(variances.numel() - finite.numel()),
            "scale_q01": scale_q[0], "scale_median": scale_q[1], "scale_q99": scale_q[2],
            "trace_ratio_median": ratio_q[0], "trace_ratio_q99": ratio_q[1],
            "trace_ratio_max": (float(finite_ratio.max()) if finite_ratio is not None and finite_ratio.numel()
                                else nan),
            "var_q99": quantiles[3], "var_q999": quantiles[4],
        })
    return rows


@torch.no_grad()
def _append_block_variance(
    model:     VFEModel,
    artifacts: "RunArtifacts",
    windows:   torch.Tensor,                 # (W, N) fixed validation input windows
    step:      int,
    device:    torch.device,
) -> None:
    r"""Write ``block_variance.csv`` rows of the current weights from a variance-only capture of ``windows``
    (``block_variance_at_log``; audit 2026-10-17 item 35). Eval mode with module modes restored, RNG streams forked."""
    variances: Dict[Tuple[int, str], List[Any]] = {}
    modes = [(module, module.training) for module in model.modules()]
    rng_devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=rng_devices):
        model.eval()
        try:
            for chunk in windows.split(max(1, DIAGNOSTIC_CHUNK_WINDOWS)):
                capture: dict = {"diagnostic": {}}
                model.forward_beliefs(chunk.to(device), capture=capture)
                _collect_block_variances(capture["diagnostic"], variances)
        finally:
            for module, was_training in modes:
                module.training = was_training
    if variances:
        artifacts.append_diagnostic_rows("block_variance.csv", BLOCK_VARIANCE_FIELDS,
                                         _block_variance_rows(variances, step))


def _collapsed_head_count(
    attention_rows: Sequence[Mapping[str, float]],   # one evaluation's _residual_and_attention_rows rows
) -> float:
    r"""``val_attn_collapsed_heads``: the number of (layer, head) pairs whose mean normalized entropy is below
    :data:`vfe3.metrics.COLLAPSE_ENTROPY_FRACTION`.

    The mean is the ``entropy_norm`` of ``diagnostics/attention_stats.csv``, over every query row with at least
    two admissible keys of the ``DIAGNOSTIC_WINDOWS`` fixed validation windows,

    .. math::
        \bar h_{lh} = \frac{1}{|R|}\sum_{(w,i)\in R}\frac{H(\beta^{lh}_{w,i\cdot})}{\log|S_{w,i}|}, \qquad
        \#\{(l,h) : \bar h_{lh} < 0.5\}.

    It replaces the count of heads with any single row below 0.5 on the first validation window, which
    saturated (audit 2026-10-10 #15). NaN when no head has a scorable row.
    """
    means = [float(row["entropy_norm"]) for row in attention_rows if math.isfinite(float(row["entropy_norm"]))]
    if not means:
        return float("nan")
    return float(sum(value < COLLAPSE_ENTROPY_FRACTION for value in means))


def _wrap_prototype_blocks(
    model:        VFEModel,
    features_out: Dict[str, Any],

    *,
    elbo:         bool = False,
) -> List[torch.nn.Module]:
    r"""Wrap ``forward_moments`` of every ``prototype_agents`` block on the instance so the eval pass sums, per
    layer, the template weights ``(M,)``, the weight entropy, the removed variance, the label staleness and the token
    count of its inputs into ``features_out["prototype"][layer]``; returns the wrapped blocks (the caller deletes the
    wrappers). With ``elbo`` (``diagnose_agent_elbo``) it also sums the block's evidence lower bound and log evidence
    (:meth:`PrototypeAgentBlockMLP.elbo_and_evidence`) into ``features_out["prototype_elbo"][layer]``."""
    blocks = getattr(model, "block_mlps", None)
    wrapped: List[torch.nn.Module] = []
    if blocks is None:
        return wrapped
    sums = features_out.setdefault("prototype", {})
    sums_elbo = features_out.setdefault("prototype_elbo", {}) if elbo else {}
    for layer, mlp in enumerate(blocks):
        if not isinstance(mlp, PrototypeAgentBlockMLP):
            continue
        original = mlp.forward_moments

        def recording(mu, sigma, *args, _layer=layer, _mlp=mlp, _original=original, **kwargs):
            weights, gain, staleness = _mlp.weights_gain_and_staleness(mu, sigma)   # (..., M), (...), (...)
            weights = weights.double().reshape(-1, weights.shape[-1])            # (T, M)
            entry = sums.setdefault(_layer, {"weights": torch.zeros_like(weights[0]), "entropy": 0.0,
                                             "gain": 0.0, "staleness": 0.0, "tokens": 0, "K": int(mu.shape[-1])})
            entry["weights"] += weights.sum(dim=0)
            entry["entropy"] += float(-torch.special.xlogy(weights, weights).sum())
            entry["gain"] += float(gain.double().sum())
            entry["staleness"] += float(staleness.double().sum())                # NaN under exact inference
            entry["tokens"] += int(weights.shape[0])
            if elbo:
                bound, evidence = _mlp.elbo_and_evidence(mu, sigma)              # (...) float64 each
                totals = sums_elbo.setdefault(_layer, [0.0, 0.0, 0.0, 0])        # [elbo, evidence, gap, tokens]
                totals[0] += float(bound.sum())
                totals[1] += float(evidence.sum())
                totals[2] += float((evidence - bound).sum())
                totals[3] += int(bound.numel())
            return _original(mu, sigma, *args, **kwargs)

        mlp.forward_moments = recording
        wrapped.append(mlp)
    return wrapped


def _accumulate_minimal_agent(
    features_out: Dict[str, Any],
    diagnostic:   Mapping[str, Any],
) -> None:
    r"""Sum the per-head minimal-agent captures of one chunk into ``features_out["minimal_agent"][layer]``."""
    for layer, capture in enumerate(diagnostic.get("layer_conditional_captures") or []):
        if _MINIMAL_AGENT_HEAD_CAPTURE[0] not in capture:
            continue
        values = []
        for key in _MINIMAL_AGENT_HEAD_CAPTURE:
            value = capture[key].detach().double().cpu().reshape(-1)             # (H,) or (1,) for the layer scalar
            values.append(value)
        H = values[0].numel()
        stacked = torch.stack([v.expand(H) if v.numel() == 1 else v for v in values])   # (F, H)
        entry = features_out.setdefault("minimal_agent", {}).setdefault(layer, [torch.zeros_like(stacked), 0])
        entry[0] += stacked
        entry[1] += 1
        if MINIMAL_AGENT_ELBO_KEYS[0] in capture:                                # diagnose_agent_elbo
            bound = torch.stack([capture[key].detach().double().cpu().reshape(()) for key in MINIMAL_AGENT_ELBO_KEYS])
            total = features_out.setdefault("minimal_agent_elbo", {}).setdefault(layer, [torch.zeros_like(bound), 0])
            total[0] += bound                                                    # (F,) MINIMAL_AGENT_ELBO_KEYS
            total[1] += 1


def _elbo_rows(
    features: Mapping[str, Any],
    step:     int,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    r"""``diagnostics/minimal_agent_elbo.csv`` and ``diagnostics/prototype_elbo.csv`` rows (:data:`MINIMAL_AGENT_ELBO_FIELDS`,
    :data:`PROTOTYPE_ELBO_FIELDS`) from the sums of :func:`_residual_and_attention_rows`; empty lists without
    ``diagnose_agent_elbo``. The minimal-agent values are chunk means of the receiver means; the prototype values are
    token means."""
    minimal_rows: List[Dict[str, float]] = []
    for layer, (total, count) in sorted(features.get("minimal_agent_elbo", {}).items()):
        minimal_rows.append({"step": int(step), "layer": int(layer),
                             **dict(zip(MINIMAL_AGENT_ELBO_FIELDS[2:-1], (total / count).tolist())),
                             "is_samples": MINIMAL_AGENT_IS_SAMPLES})
    prototype_rows: List[Dict[str, float]] = []
    for layer, (elbo, evidence, gap, tokens) in sorted(features.get("prototype_elbo", {}).items()):
        n = max(1, tokens)
        prototype_rows.append({"step": int(step), "layer": int(layer), "elbo": elbo / n, "evidence": evidence / n,
                               "elbo_gap": gap / n, "n_tokens": int(tokens)})
    return minimal_rows, prototype_rows


def _feature_rows(
    model:    VFEModel,
    features: Mapping[str, Any],
    step:     int,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    r"""``diagnostics/minimal_agent_heads.csv`` and ``diagnostics/prototype_templates.csv`` rows from the sums of
    :func:`_residual_and_attention_rows` (``features_out``); empty lists for a feature the model does not use."""
    head_rows: List[Dict[str, float]] = []
    for layer, (total, count) in sorted(features.get("minimal_agent", {}).items()):
        mean = (total / count).tolist()                                         # (F, H)
        for head in range(len(mean[0])):
            row = {"step": int(step), "layer": int(layer), "head": int(head)}
            row.update({name: float(mean[f][head]) for f, name in enumerate(MINIMAL_AGENT_HEAD_FIELDS[3:])})
            head_rows.append(row)
    template_rows: List[Dict[str, float]] = []
    for layer, entry in sorted(features.get("prototype", {}).items()):
        mlp = model.block_mlps[layer]
        tokens = max(1, entry["tokens"])
        usage = entry["weights"] / tokens                                       # (M,) mean template weight
        M = usage.numel()
        log_lambda = mlp.log_lambda.detach().double().reshape(-1)
        prior = torch.softmax(mlp.log_pi.detach().double(), dim=-1)
        template_rows.append({
            "step": int(step), "layer": int(layer),
            "effective_templates": float(torch.exp(-torch.special.xlogy(usage, usage).sum())),
            "weight_entropy": entry["entropy"] / tokens,
            "dead_fraction": float((usage < 1.0 / (10 * M)).double().mean()),
            "log_lambda_min": float(log_lambda.min()),
            "log_lambda_median": float(log_lambda.median()),
            "log_lambda_max": float(log_lambda.max()),
            "prior_entropy": float(-torch.special.xlogy(prior, prior).sum()),
            "gain_fraction": entry["gain"] / tokens / entry["K"],
            "label_staleness": entry["staleness"] / tokens,
            "n_tokens": int(entry["tokens"]),
        })
    return head_rows, template_rows


def _routing_holonomy_rows(
    sums: Mapping[Any, Any],
    step: int,
) -> List[Dict[str, float]]:
    r"""``diagnostics/routing_holonomy.csv`` rows from the sums of ``_residual_and_attention_rows``.

    Mean :math:`\sum wx/\sum w` and RMS :math:`(\sum wx^2/\sum w)^{1/2}` of
    :math:`x=\log\rho_{ijk}` (NaN without strict triangles), and the mean path mass per query row.
    """
    nan = float("nan")
    rows: List[Dict[str, float]] = []
    for key in sorted(k for k in sums if k != "error"):
        mass, s1, s2, p_mass, p1, p2, n_rows = sums[key]
        rows.append({
            "step": int(step), "layer": int(key[0]), "head": int(key[1]),
            "path_mass": mass / n_rows if n_rows else nan,
            "log_defect_mean": s1 / mass if mass > 0 else nan,
            "log_defect_rms": math.sqrt(s2 / mass) if mass > 0 else nan,
            "prior_log_defect_mean": p1 / p_mass if p_mass > 0 else nan,
            "prior_log_defect_rms": math.sqrt(p2 / p_mass) if p_mass > 0 else nan,
            "n_rows": int(n_rows),
        })
    return rows


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
    windows better than held-out text), with ``train_sample_seen_fraction``
    :math:`= \min(1, sB/W)` after :math:`s` steps of batch :math:`B` over :math:`W` training windows: the
    expected fraction of the fixed sample the shuffled training stream has already served, since the
    sample is drawn independently of the epoch permutation. Before one pass the gap therefore compares
    validation with mostly unseen training text (audit 2026-10-07, F46); (2) ``diagnostics/residual_ratio.csv`` and
    ``diagnostics/attention_stats.csv`` rows on the fixed validation windows, and from the latter
    ``val_attn_collapsed_heads`` (:func:`_collapsed_head_count`; returned beside the columns, NaN when the
    capture fails, and filed by the caller with the held-out probes); (3) with
    ``count_numerics`` (projected routes), the S1 panel D per-evaluation counters of the same
    capture pass (``NUMERICS_EVAL_COLUMNS``, see :func:`_capture_numerics_counts`) and their
    per-(layer, head) rows in ``diagnostics/cap_activation.csv``; (4) the map snapshot
    ``map_snapshots/step_<N:07d>.pt``. Each part is isolated: a failure is counted and the others
    still run (the counters are NaN when their shared capture pass fails; that failure is counted
    once, as ``residual_attention``). CPU and CUDA RNG states are restored on exit.
    """
    out = {column: float("nan") for column in (*_EVAL_CADENCE_COLUMNS, "val_attn_collapsed_heads")}
    rng_devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=rng_devices):
        if train_sample is not None:
            try:
                sample = evaluate(model, train_sample, device=device)
                out["train_sample_ce"] = float(sample["ce"])
                out["train_sample_ppl"] = float(sample["ppl"])
                out["generalization_gap_fixed"] = float(val_ce) - float(sample["ce"])
                n_windows_total = int(len(train_sample.dataset))
                out["train_sample_seen_fraction"] = min(
                    1.0, int(step) * int(model.cfg.batch_size) / max(n_windows_total, 1))
                artifacts.train_sample_by_step[int(step)] = float(sample["ce"])
            except Exception as exc:
                _record_diagnostic_failure(failures, "train_sample", exc, logger)
        if windows is not None:
            counters: Optional[Dict[str, Any]] = {} if count_numerics else None
            holonomy: Optional[Dict[Any, Any]] = {}
            features: Optional[Dict[str, Any]] = {}
            variances: Optional[Dict[Tuple[int, str], List[Any]]] = {}
            try:
                residual_rows, attention_rows = _residual_and_attention_rows(
                    model, windows, step, device, counters_out=counters, holonomy_out=holonomy,
                    features_out=features, variance_out=variances)
            except Exception as exc:
                _record_diagnostic_failure(failures, "residual_attention", exc, logger)
                residual_rows = attention_rows = None
                counters = holonomy = features = variances = None   # their capture pass failed; counted once above
            if variances:
                try:
                    artifacts.append_diagnostic_rows(
                        "block_variance.csv", BLOCK_VARIANCE_FIELDS, _block_variance_rows(variances, step))
                except Exception as exc:
                    _record_diagnostic_failure(failures, "block_variance", exc, logger)
            if features:
                try:
                    head_rows, template_rows = _feature_rows(model, features, step)
                    if head_rows:
                        artifacts.append_diagnostic_rows(
                            "minimal_agent_heads.csv", MINIMAL_AGENT_HEAD_FIELDS, head_rows)
                    if template_rows:
                        artifacts.append_diagnostic_rows(
                            "prototype_templates.csv", PROTOTYPE_TEMPLATE_FIELDS, template_rows)
                    minimal_elbo_rows, prototype_elbo_rows = _elbo_rows(features, step)
                    if minimal_elbo_rows:
                        artifacts.append_diagnostic_rows(
                            "minimal_agent_elbo.csv", MINIMAL_AGENT_ELBO_FIELDS, minimal_elbo_rows)
                    if prototype_elbo_rows:
                        artifacts.append_diagnostic_rows(
                            "prototype_elbo.csv", PROTOTYPE_ELBO_FIELDS, prototype_elbo_rows)
                except Exception as exc:
                    _record_diagnostic_failure(failures, "feature_diagnostics", exc, logger)
            if holonomy is not None:
                try:
                    if "error" in holonomy:
                        raise holonomy["error"]
                    artifacts.append_diagnostic_rows(
                        "routing_holonomy.csv", ROUTING_HOLONOMY_FIELDS,
                        _routing_holonomy_rows(holonomy, step))
                except Exception as exc:
                    _record_diagnostic_failure(failures, "routing_holonomy", exc, logger)
            if residual_rows is not None:
                out["val_attn_collapsed_heads"] = _collapsed_head_count(attention_rows)
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
                    artifacts.append_diagnostic_rows(
                        "cap_clamped_rows.csv", CAP_CLAMPED_ROW_FIELDS, _cap_clamped_row_rows(counters, step))
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
    if "pos_loss_ratio" in averaged:
        # The ratio of the averaged quarters, not the average of per-batch ratios, so the three columns agree
        # (a sliding-window batch whose first quarter is masked contributes to last_q alone; audit 2026-10-12
        # round 2, item 16).
        averaged["pos_loss_ratio"] = averaged["pos_loss_last_q"] / max(averaged["pos_loss_first_q"], 1e-9)
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
            or model.cfg.norm_type_block != "none"
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
         decode_logdet_fallback_elements(), frame_pair_fp64_escalations(), public_spd_repair_elements(),
         public_spd_ceiling_contact_elements()),
    )


def _tensor_versions(
    model: torch.nn.Module,
) -> tuple:
    r"""Identity, storage and in-place version of every parameter and buffer: equal tuples mean unchanged weights."""
    return tuple((id(tensor), tensor.data_ptr(), tensor._version)
                 for tensor in (*model.parameters(), *model.buffers()))


def _snapshot_key_support(
    log_prior: Optional[torch.Tensor],        # (N, N), (H, N, N) or (1, H, N, N) for one sequence
) -> Optional[torch.Tensor]:                  # bool, broadcastable to (L, H, N, N); None if absent
    r"""Admissible-key mask ``isfinite(log_prior)`` of one diagnostic sequence (audit 2026-10-07 F14)."""
    if not isinstance(log_prior, torch.Tensor):
        return None
    if log_prior.dim() == 4:
        log_prior = log_prior[0]
    return torch.isfinite(log_prior)


def _attention_structure_metrics(
    amaps: torch.Tensor,                      # (L, H, N, N) all layers/heads, one sequence
    support: Optional[torch.Tensor] = None,   # bool, broadcastable to amaps: admissible keys
) -> Dict[str, float]:
    r"""Held-out attention-map structure: entropy collapse, causal sanity, positional/content, redundancy.

    Shared by both routes of :func:`_val_diagnostics_one`; the legacy route passes the entry-scored
    replay, the conditional route the actual pre-update router ``snapshot.beta_maps``. ``support``
    is the prior's admissible-key mask; without it row support is read from ``amaps > 0``.
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
    admitted = (amaps > 0.0) if support is None else support.expand_as(amaps)
    scorable = admitted.sum(dim=-1) >= 2                        # rows with >=2 admissible keys
    hmin = ent_rows.masked_fill(~scorable, float("inf")).min(dim=-1).values   # (L, H)
    finite_hmin = hmin[torch.isfinite(hmin)]
    out["val_attn_entropy_min_all"] = (float(finite_hmin.min()) if finite_hmin.numel()
                                       else float("nan"))
    # val_attn_collapsed_heads is not scored here: one row of this one window flagged a head, so the
    # count saturated (audit 2026-10-10 #15). It comes from the diagnostic windows (_collapsed_head_count).
    worst_norm, _ = M.attention_head_collapse(amaps, support=support)           # (L, H)
    finite_norm = worst_norm[torch.isfinite(worst_norm)]
    out["val_attn_entropy_min_norm"] = (float(finite_norm.min()) if finite_norm.numel()
                                        else float("nan"))
    cs = M.causal_sanity(amaps)
    out["val_future_leakage"] = float(cs["future_leakage"].max())   # soft causal prior can leak silently
    out["val_row_sum_error"]  = float(cs["row_sum_error"].max())
    out["val_pos_content_r2"] = float(M.positional_content_score(amaps, support=support).mean())
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
    r"""Per-position within-sequence loss: mean CE over the scored positions of the first and last quarter.

    A position with no scored target (the masked overlap prefix of a sliding-window eval window, or the padded
    tail of a final window) has no CE, so it is left out of its quarter's mean rather than counted as 0; a quarter
    with no scored position is NaN, which the multi-batch average skips (audit 2026-10-12 #13).
    """
    import torch.nn.functional as F

    out: Dict[str, float] = {}
    b, n = targets.shape
    per = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1),
                          ignore_index=-100, reduction="none").reshape(b, n)
    valid = (targets != -100).float()
    n_valid = valid.sum(0)                                        # (N,) scored targets at each position
    pos_ce = (per * valid).sum(0) / n_valid.clamp(min=1.0)        # (N,) mean CE at each position
    pos_ce = pos_ce.masked_fill(n_valid == 0, float("nan"))       # unscored position: no CE
    q = n // 4
    if q > 0:
        first_q, last_q = pos_ce[:q].nanmean(), pos_ce[-q:].nanmean()
        out["pos_loss_first_q"] = float(first_q)
        out["pos_loss_last_q"]  = float(last_q)
        out["pos_loss_ratio"]   = float(last_q / first_q.clamp(min=1e-9))
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
    ``val_attn_entropy_min_norm``, ``val_future_leakage``,
    ``val_row_sum_error``, ``val_pos_content_r2``, ``val_prev_token_mass``,
    ``val_period_match_mass``, ``val_head_redundancy_js``) are computed on the actual pre-update
    router ``snapshot.beta_maps`` of the same capture (``val_attn_collapsed_heads`` is not: the
    eval-cadence pass scores it on the diagnostic windows, :func:`_collapsed_head_count`), the
    ``pos_loss_*`` columns on its logits, and
    ``val_effective_rank`` / ``val_belief_cond_median`` from the last layer's converged covariance
    (as the relational diagnostics report them), and the held-out geometry and guard columns
    (``val_fisher_trace_mean``, ``val_belief_cond_p95``, ``val_belief_lam_*``, ``val_belief_cond_valid_frac``,
    ``val_phi_*``, ``val_vertex_cond_p99``, ``val_pos_phi_*`` when a positional factor exists,
    ``val_guard_sigma_*``, ``val_nonfinite_frac``) from the same diagnostics. The following stay NaN on this
    route, by definition rather than by omission: the legacy free-energy decomposition (``val_self_coupling``,
    ``val_self_divergence``, ``val_belief_coupling``, ``val_attention_entropy``,
    ``val_inner_alignment_energy_total``), because routing plus readout does not define the shared
    stationary objective (``diagnose_canonical_free_energy`` fills them, per token, with the canonical
    free energy at the beliefs the last router read, and adds ``val_canonical_beta_tv``,
    ``val_canonical_self_weight``, ``val_canonical_total_output`` and ``val_canonical_total_delta``).
    No E-step certificate column is written, because the route performs one pre-update readout with
    no iterate sequence (audit 2026-10-17 item 71).
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
    out["val_belief_cond_median_layer_max"] = float(vd.get("belief_cond_median_layer_max", float("nan")))
    for key in ("belief_lam_min_layer_min", "belief_lam_min_argmin_layer", "belief_lam_max_layer_max",
                "belief_cond_max_layer_max", "belief_eig_ratio_layer_min", "belief_eig_ratio_argmin_layer"):
        out[f"val_{key}"] = float(vd.get(key, float("nan")))
    out.update({f"val_{key}": float(vd[key]) for key in _VAL_GEOMETRY_KEYS if key in vd})
    if "canonical_beta_tv" in vd:                               # diagnose_canonical_free_energy
        out.update(val_self_coupling=vd["self_coupling"], val_self_divergence=vd["self_divergence"],
                   val_belief_coupling=vd["belief_coupling"], val_attention_entropy=vd["attention_entropy"],
                   val_inner_alignment_energy_total=vd["total"])
        out.update({f"val_{key}": vd[key] for key in CANONICAL_DIAGNOSTIC_KEYS})
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
            "val_gauge_invariant_spread": float(vd["gauge_invariant_spread"]),
        }
        structure.update(_attention_structure_metrics(
            snapshot.beta_maps.float(), _snapshot_key_support(snapshot.log_prior)))
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
    hence the replay.
    """
    # The maps are computed here; their isolated figure child runs on a background thread when the artifacts object
    # offers one, so training does not wait for it (audit 2026-10-17 item 68); the run reaps it before finalizing.
    submit = getattr(artifacts, "submit_figure_work", None)
    entry_maps = model.attention_maps(token_ids, score_at="entry")
    if callable(submit):
        entry_cpu = entry_maps.detach().cpu()
        submit("attention", lambda: artifacts.save_attention_maps(step, entry_cpu, logger=logger))
    else:
        artifacts.save_attention_maps(step, entry_maps, logger=logger)


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
    # The loop's validation of these same final weights when its last eval fell on ``step``, so the finalizer
    # neither scores validation again nor writes a second metrics row for the step (audit 2026-10-12 #12):
    # ``{"val_loader", "tokens_per_char", "metrics"}``. None when the last step was not an eval step.
    final_validation:     Optional[Dict[str, object]] = None


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
    if getattr(dataset, "window_offset_mode", "fixed") != "fixed":
        normalized["iterator"]["window_offset"] = str(dataset.window_offset_mode)
    return normalized


def _configure_train_window_offset(
    loader: object,
) -> Optional[object]:
    r"""Give the train loader's windows a per-epoch random origin; return the dataset to draw on.

    A training :class:`~vfe3.data.datasets.TokenWindows` dataset draws a new origin each epoch, just
    before the epoch's iterator: from the loader's ``torch.Generator`` when the loader shuffles with one
    (the saved epoch-start generator state then replays origin and permutation on resume), else from the
    global CPU RNG. It needs in-process loading (``num_workers=0``, so the dataset the workers read is
    the one whose origin was set). Any other train dataset (in-memory batches, an evaluation stream
    with ``pad_final``) has no window origin to draw and returns None.
    """
    dataset = getattr(loader, "dataset", None)
    if not hasattr(dataset, "use_per_epoch_offsets") or getattr(dataset, "pad_final", False):
        return None
    if getattr(loader, "num_workers", 0) != 0:
        raise ValueError("per-epoch training-window origins need num_workers=0")
    dataset.use_per_epoch_offsets()
    return dataset


def _empty_training_epoch_message(
    loader: object,
) -> str:
    r"""Describe a sized training loader that yields no batch per epoch, with the shortest stream that fills one.

    With ``drop_last`` a loader over :math:`n` windows yields :math:`\lfloor n / B \rfloor` batches of :math:`B`.
    For a :class:`~vfe3.data.datasets.TokenWindows` stream of :math:`T` tokens, window length :math:`L` and stride
    :math:`s`, the per-epoch origins of training fit :math:`n = \lfloor (T - L) / s \rfloor` windows once
    :math:`T \ge L + s` (one window below that), so a batch of :math:`B \ge 2` needs :math:`T \ge L + B s`; fixed
    origins fit :math:`n = \lfloor (T - L - 1) / s \rfloor + 1` and need :math:`T \ge L + 1 + (B - 1) s`.
    """
    dataset = getattr(loader, "dataset", None)
    batch_size = getattr(loader, "batch_size", None)
    tokens = getattr(dataset, "tokens", None)
    seq_len, stride = getattr(dataset, "seq_len", None), getattr(dataset, "stride", None)
    message = "the training loader yields no batch per epoch (len(loader) == 0)"
    if (not isinstance(tokens, torch.Tensor) or not isinstance(batch_size, int) or not isinstance(seq_len, int)
            or not isinstance(stride, int)):
        return message + "; give it at least one full batch"
    if getattr(dataset, "window_offset_mode", "fixed") == "per_epoch_random":
        minimum = seq_len + 1 if batch_size == 1 else seq_len + batch_size * stride
    else:
        minimum = seq_len + 1 + (batch_size - 1) * stride
    return (f"{message}: {len(dataset)} training windows (seq_len={seq_len}, stride={stride}) of a "
            f"{tokens.numel()}-token stream do not fill one batch of batch_size={batch_size} with drop_last; the "
            f"stream needs at least {minimum} tokens. Raise the token cap (MAX_TOKENS / max_tokens) or lower "
            "batch_size")


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


_MATMUL_PRECISION_LADDER: Tuple[str, ...] = ("medium", "high", "highest")


class ConsecutiveSkipAbort(RuntimeError):
    r"""Raised by the training loop when ``max_consecutive_skips`` rejected updates occurred in a row (audit 2026-10-17
    item 16); it ends the run through the crash path (``diagnostics/crash.json`` and a resumable checkpoint)."""


def _reset_scaler_step_state(
    scaler:    Optional["torch.amp.GradScaler"],
    optimizer: torch.optim.Optimizer,
) -> None:
    r"""Forget a failed step's per-optimizer GradScaler stage (an ``unscale_`` without ``update``), so the next step's
    ``unscale_`` does not refuse; a no-op for a disabled or absent scaler."""
    if scaler is None or not scaler.is_enabled():
        return
    states = getattr(scaler, "_per_optimizer_states", None)
    if isinstance(states, dict):
        states.pop(id(optimizer), None)


def _parameter_nonfinite_count(
    model: torch.nn.Module,
) -> int:
    r"""Number of non-finite entries over every floating parameter (one host read; audit 2026-10-17 item 47)."""
    counts = [(~torch.isfinite(p.detach())).sum() for p in model.parameters()
              if p.is_floating_point() or p.is_complex()]
    if not counts:
        return 0
    return int(torch.stack([count.to(counts[0].device) for count in counts]).sum())


class NonFiniteStepSkipped(RuntimeError):
    r"""A train step rejected its update for a non-finite value under a reduced matmul precision; raised only
    inside :func:`_train_step_with_precision_retry` to replay the step one rung stricter, never to a caller."""

# Wall-clock cost of the work a logging or evaluation step adds outside the timed train step, one
# metrics.csv column each (NaN when that phase did not run on the row). Each phase is bracketed by a
# device synchronization, so queued GPU work is charged to the phase that issued it.
LOG_TIMING_COLUMNS: Tuple[str, ...] = (
    "log_diagnostics_ms", "log_group_stats_ms", "log_row_ms",
    "eval_validation_ms", "eval_sample_ms", "eval_val_diagnostics_ms", "eval_best_and_maps_ms",
    "eval_cadence_ms", "eval_gap_train_ms",
)


def _synced_ms(device: torch.device) -> float:
    r"""Milliseconds on the host clock after the device has drained its queued work."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() * 1000.0


@contextlib.contextmanager
def _float32_matmul_precision(policy: str) -> Iterator[None]:
    r"""Set the process fp32 matmul policy for the enclosed forward and backward, then restore it."""
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(policy)
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def _no_update_call_with_precision_retry(
    model: VFEModel,
    call:  Callable[[], Any],
    label: str,

    *,
    restore_rng: bool = False,
) -> Any:
    r"""Run a no-update call (an evaluation forward or a diagnostic) on the replay ladder of training.

    A call that raises at a reduced ``fp32_matmul_precision`` (a fail-closed certificate tripped by TF32 or
    bf16 GEMM rounding) is run again one rung stricter,
    ``"medium" -> "high" -> "highest"``, as :func:`_train_step_with_precision_retry` does for the train
    step (audit 2026-10-08 P0-1). A no-update call changes no parameter, so a replay needs no gradient
    or optimizer rollback; ``restore_rng`` also rewinds the CPU and CUDA RNG streams before each replay,
    so a replay draws what the first attempt drew. A failure that persists at ``"highest"`` is raised
    with the earlier failures attached as notes; out-of-memory errors are never replayed.
    """
    current = torch.get_float32_matmul_precision()
    if current == "highest":
        return call()
    cpu_rng = torch.get_rng_state().clone() if restore_rng else None
    cuda_rng = ([state.clone() for state in torch.cuda.get_rng_state_all()]
                if restore_rng and torch.cuda.is_available() and torch.cuda.is_initialized() else None)
    failures: List[str] = []
    for policy in (current, *_MATMUL_PRECISION_LADDER[_MATMUL_PRECISION_LADDER.index(current) + 1:]):
        if failures and cpu_rng is not None:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
        try:
            with _float32_matmul_precision(policy):
                out = call()
            if failures:
                logging.getLogger(__name__).warning(
                    "%s raised under fp32_matmul_precision=%r (%s); it completed at %r",
                    label, current, failures[0], policy)
            return out
        except torch.OutOfMemoryError:
            raise
        except Exception as exc:
            if policy == "highest":
                for note in failures:
                    exc.add_note(f"earlier attempt of this {label} failed: {note}")
                raise
            failures.append(f"{type(exc).__name__}: {exc} [at {policy!r}]")
    raise AssertionError("unreachable: the ladder ends at 'highest'")


def _eval_forward_with_precision_retry(model: VFEModel, *args: Any, **kwargs: Any) -> Any:
    r"""Run a no-update evaluation forward ``model(*args, **kwargs)`` on the replay ladder of training
    (:func:`_no_update_call_with_precision_retry`).

    Each attempt runs inside :func:`~vfe3.contracts.deferred_host_checks`, as the forward of :func:`train_step`
    does: its raise-only certificates join the forward's next mandatory host read instead of synchronizing one at
    a time, and the scope closes (resolving anything still queued) when the forward returns, before the caller
    reads its decode statistics, so a deferred verdict still raises inside the ladder and is replayed (audit
    2026-10-16 F125). Training diagnostics stay outside the scope: ``model.diagnostics`` reads its values on the
    host before it returns, which the scope's contract excludes.
    """
    def forward() -> Any:
        with deferred_host_checks():
            return model(*args, **kwargs)
    return _no_update_call_with_precision_retry(model, forward, "evaluation forward")


def _diagnostics_with_precision_retry(
    model:  VFEModel,
    tokens: torch.Tensor,                # (B, N) input token ids of the training batch

    *,
    before_attempt: Optional[Callable[[], None]] = None,   # clears per-attempt captures (prototype sums)
) -> Dict[str, Any]:
    r"""``model.diagnostics(tokens)`` on the replay ladder that :func:`evaluate` uses (audit 2026-10-10 round 2,
    item 10), with the RNG streams rewound before a replay so the training stream after it is unchanged."""
    def attempt() -> Dict[str, Any]:
        if before_attempt is not None:
            before_attempt()
        return model.diagnostics(tokens)
    return _no_update_call_with_precision_retry(model, attempt, "training diagnostics", restore_rng=True)


def _train_step_with_precision_retry(
    model:     VFEModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    tokens:    torch.Tensor,             # (B, N) input token ids
    targets:   torch.Tensor,             # (B, N) next-token ids

    *,
    logger:    logging.Logger,
    step:      int,
    quiet:     bool = False,             # log the replay at DEBUG (the caller already reported one)
    replay_log: Optional[List[Dict[str, object]]] = None,   # one record appended per failed attempt
    **step_kwargs: Any,
) -> Tuple[float, int]:                  # (loss, number of escalated replays the step needed)
    r"""Run :func:`train_step`; under a reduced fp32 matmul policy replay a failed step at a stricter one.

    Under a reduced ``fp32_matmul_precision`` the GEMM rounding is about :math:`2^{-10}` (TF32,
    ``"high"``) or :math:`2^{-7}` (bf16, ``"medium"``) instead of :math:`2^{-23}`, enough to trip a
    fail-closed certificate (frame-pair inverse residual, positive source variance, factorization)
    on an ill-conditioned frame. When the step raises before the optimizer has updated (a step
    pre-hook records any update), the gradients are dropped, the CPU and CUDA RNG states are
    restored, and the same step runs again, forward and backward, one rung stricter on the ladder
    ``"medium" -> "high" -> "highest"``. A failure that persists at ``"highest"`` is raised with the
    earlier failures attached as notes, so a genuine defect still stops the run. Out-of-memory errors
    and failures after an update are never replayed. Not used with an enabled GradScaler (fp16),
    whose per-step unscale state a replay would violate. Module-level numerics counters may count
    the abandoned attempts. Each failed attempt that is replayed appends ``{step, policy, replay_policy,
    exception, message, amp_dtype, autocast_active}`` to ``replay_log`` when given, so the reasons survive
    the DEBUG-level logging of later replays (audit 2026-10-08 critic-gapfill-1#4). ``autocast_active``
    is true when ``cfg.amp_dtype`` is set and not marked inert; such a record also carries
    ``autocast_note``: the ladder changes only the fp32 matmul policy, so a failure caused by the
    autocast dtype's rounding is not repaired by the replay (audit 2026-10-09 round 2, item 40).

    A step that returns without an update because its loss, gradient or clip norm was non-finite
    (``status_out["nonfinite_skip"]``) is replayed the same way: no optimizer update ran, so the
    replay is as safe as for a raised step (audit 2026-10-10 round 2, item 58). Its record names the
    exception ``NonFiniteStepSkipped``. A skip that persists at ``"highest"`` is returned as a skipped
    step, as before.
    """
    scaler = step_kwargs.get("scaler")
    current = torch.get_float32_matmul_precision()
    if current == "highest" or (scaler is not None and scaler.is_enabled()):
        return train_step(model, optimizer, scheduler, tokens, targets, **step_kwargs), 0
    status = step_kwargs.get("status_out")
    if status is None:                                  # the skip verdict is read from the step's status
        status = {}
        step_kwargs = {**step_kwargs, "status_out": status}
    rungs = _MATMUL_PRECISION_LADDER[_MATMUL_PRECISION_LADDER.index(current) + 1:]
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = ([state.clone() for state in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available() and torch.cuda.is_initialized() else None)
    failures: List[str] = []
    verbs: List[str] = []
    amp_dtype = getattr(model.cfg, "amp_dtype", None)
    for replays, policy in enumerate((current, *rungs)):
        if replays:
            optimizer.zero_grad(set_to_none=True)
            for out in (step_kwargs.get("metrics_out"), step_kwargs.get("status_out")):
                if out is not None:
                    out.clear()
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            logger.log(logging.DEBUG if quiet else logging.WARNING,
                       "step %d %s under fp32_matmul_precision=%r (%s); replaying it at %r%s",
                       step, verbs[-1], (current, *rungs)[replays - 1], failures[-1], policy,
                       "" if quiet else ("; further replays are logged at DEBUG, counted in metrics.csv "
                                         "matmul_precision_replays and summarized at each log step"))
        updated: List[bool] = []
        handle = optimizer.register_step_pre_hook(lambda *_args: updated.append(True))
        try:
            with _float32_matmul_precision(policy):
                loss = train_step(model, optimizer, scheduler, tokens, targets, **step_kwargs)
            if policy != "highest" and not updated and status.get("nonfinite_skip"):
                raise NonFiniteStepSkipped(
                    f"update rejected for a non-finite loss, gradient or clip norm (loss {loss!r})")
            return loss, replays
        except torch.OutOfMemoryError:
            raise
        except Exception as exc:
            verbs.append("skipped a non-finite update" if isinstance(exc, NonFiniteStepSkipped) else "raised")
            if updated or policy == "highest":
                for note in failures:
                    exc.add_note(f"earlier attempt of this step failed: {note}")
                raise
            failures.append(f"{type(exc).__name__}: {exc} [at {policy!r}]")
            if replay_log is not None:
                autocast_active = amp_dtype is not None and "amp_dtype" not in model.cfg.inactive_fields()
                record: Dict[str, object] = {
                    "step": int(step), "policy": policy, "replay_policy": (current, *rungs)[replays + 1],
                    "exception": type(exc).__name__, "message": str(exc),
                    "amp_dtype": amp_dtype, "autocast_active": autocast_active}
                if autocast_active:
                    record["autocast_note"] = (
                        f"{amp_dtype} autocast was active; the replay ladder changes only "
                        "fp32_matmul_precision, so a failure caused by autocast rounding is not repaired")
                replay_log.append(record)
        finally:
            handle.remove()
    raise AssertionError("unreachable: the ladder ends at 'highest'")


MATMUL_PRECISION_REPLAY_LOG = "matmul_precision_replays.jsonl"


def _append_replay_records(
    run_dir: Path,
    records: Sequence[Mapping[str, object]],
) -> Path:
    r"""Append one JSON line per replayed attempt to ``run_dir/diagnostics/matmul_precision_replays.jsonl``."""
    directory = Path(run_dir) / "diagnostics"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MATMUL_PRECISION_REPLAY_LOG
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), sort_keys=True) + "\n")
    return path


# Divergence monitor (audit 2026-10-10 round 2, item 1). Host-side only: it reads the loss, CE and pre-clip
# role norms that train_step already transfers every step, so it adds no device synchronization. The
# constants are conservative: an ordinary batch-to-batch fluctuation of a healthy run stays well inside them.
DIVERGENCE_EMA_DECAY:         float = 0.98     # EMAs of the loss and of each role norm (time constant 50 steps)
DIVERGENCE_EMA_WARMUP:        int   = 50       # finite observations of a quantity before it can spike
DIVERGENCE_NORM_SPIKE_RATIO:  float = 10.0     # a role's pre-clip norm above 10x its EMA is a spike
DIVERGENCE_LOSS_SPIKE_MARGIN: float = 1.0      # nats: a training loss above its EMA by more is a spike
DIVERGENCE_SPIKE_COOLDOWN:    int   = 100      # steps between reported spikes (each forces a metrics row)
DIVERGENCE_SPIKE_ESCALATION:  float = 10.0     # an event this many times the last reported severity is reported anyway
DIVERGENCE_ROLES: Tuple[str, ...] = ("mu", "sigma", "phi")
# Columns on every metrics row: statistics over the steps since the previous log/eval row, the realized clip
# coefficients of the row's own step, and whether the row was forced by a spike.
DIVERGENCE_COLUMNS: Tuple[str, ...] = (
    "interval_steps", "train_loss_interval_mean", "train_loss_interval_max", "train_ce_interval_mean",
    "train_loss_interval_median", "train_ce_interval_median",
    *(f"grad_norm_{role}_interval_{stat}" for role in DIVERGENCE_ROLES for stat in ("mean", "max")),
    "grad_clip_binding_frac",
    *(f"grad_clip_coef_{role}" for role in DIVERGENCE_ROLES),
    "divergence_spike",
)


class _RunningStat:
    r"""Count, sum and maximum of the finite values offered to it; with ``keep_values`` also the values, for the
    interval median (audit 2026-10-20, spike report section 7: the interval mean includes blow-ups)."""

    def __init__(self, keep_values: bool = False) -> None:
        self.count, self.total, self.peak = 0, 0.0, -math.inf
        self.values: Optional[List[float]] = [] if keep_values else None

    def add(self, value: float) -> None:
        if math.isfinite(value):
            self.count += 1
            self.total += value
            self.peak = max(self.peak, value)
            if self.values is not None:
                self.values.append(float(value))

    def median(self) -> float:
        r"""The median of the kept finite values (the mean of the two middle ones for an even count); NaN when
        none were kept."""
        if not self.values:
            return float("nan")
        ordered = sorted(self.values)
        middle = len(ordered) // 2
        return ordered[middle] if len(ordered) % 2 else 0.5 * (ordered[middle - 1] + ordered[middle])

    def mean(self) -> float:
        return self.total / self.count if self.count else float("nan")

    def max(self) -> float:
        return self.peak if self.count else float("nan")


DIVERGENCE_SPIKE_LOG = "divergence_spikes.jsonl"          # run_dir/diagnostics: one record per reported spike
_SPIKE_TOP_PARAMETERS = 5
_SPIKE_TOP_ROWS = 3
_SPIKE_TABLE_INDEX = {"mu_embed": "token", "sigma_log_embed": "token", "phi_embed": "token", "pos_phi": "position"}
# Decode-side tables (untied decode bank) are read at the step's targets, so their rows are counted over targets.
_SPIKE_DECODE_TABLES = ("decode_mu_embed", "decode_sigma_log_embed")
_SPIKE_ROW_OCCURRENCES = 8          # (batch, position) pairs listed per top row
# A reported spike re-runs the spike batch without gradients, row by row, to record the largest final belief
# variance and the per-position CE maximum (audit 2026-10-20 item 11). It costs about one forward per reported spike.
DIVERGENCE_SPIKE_FORWARD_PROBE: bool = True


def _phi_row_head_conditions(
    model: torch.nn.Module,
    phi:   torch.Tensor,                   # (R, n_gen) frame-table rows
) -> List[List[float]]:
    r"""Per row and head the condition number :math:`\kappa_h = s_{\max}(U_h)/s_{\min}(U_h)` of the head block
    :math:`U_h = e^{M_h}` of the row's frame :math:`U = e^{\sum_a\phi_aG_a}`, in float64.

    Uses :func:`vfe3.metrics.head_block_algebra` when the basis is block-diagonal on the heads, else the diagonal
    blocks of the dense exponential. The positional factor a token's frame also carries is not included: this is the
    condition of the table row alone.
    """
    from vfe3 import metrics
    group = getattr(model, "group", None)
    if group is None or phi.numel() == 0:
        return []
    with torch.no_grad(), torch.autocast(device_type=phi.device.type, enabled=False):
        phi = phi.detach().to(torch.float64)
        algebra = metrics.head_block_algebra(phi, group)
        if algebra is not None:
            blocks = [torch.linalg.matrix_exp(M) for M in algebra]                       # per head (R, d_h, d_h)
        else:
            U = torch.linalg.matrix_exp(group.embed_algebra(phi, dtype=torch.float64))  # (R, K, K)
            offsets = [0, *itertools.accumulate(group.irrep_dims)]
            blocks = [U[:, a:b, a:b] for a, b in zip(offsets[:-1], offsets[1:])]
        conditions = []
        for block in blocks:
            singular = torch.linalg.svdvals(block)                                        # (R, d_h) descending
            conditions.append(singular[..., 0] / singular[..., -1])
        return torch.stack(conditions, dim=-1).cpu().tolist()                             # (R, H)


@torch.no_grad()
def _spike_forward_state(
    model:   torch.nn.Module,
    tokens:  torch.Tensor,                 # (B, N) the spike step's input tokens
    targets: torch.Tensor,                 # (B, N) its targets (-100 = ignore)
) -> Dict[str, Any]:
    r"""The largest final belief variance and the per-position CE maximum of a spike batch, from a no-grad forward.

    The forward runs one sequence at a time (so the logits are (1, N, V)), in eval mode, with the global RNG streams
    restored afterwards, at the parameters after the spike step's update: the step's own forward is not kept, so the
    values describe the batch under the updated model. The variance is the largest diagonal entry of the final
    ``belief.sigma`` (the diagonal of a matrix covariance), with its (batch, position) and the CE at that position; the
    CE is the plug-in cross-entropy :math:`-\log p(y_{bn}\mid q_{bn})` at each scored position, and its maximum is
    reported with its (batch, position) and the variance there.
    """
    import torch.nn.functional as F

    was_training = model.training
    rng_devices = [tokens.device] if tokens.device.type == "cuda" else []
    variance_rows: List[torch.Tensor] = []                           # per sequence (N,)
    ce_rows: List[torch.Tensor] = []                                 # per sequence (N,)
    with torch.random.fork_rng(devices=rng_devices):
        model.eval()
        try:
            for b in range(tokens.shape[0]):
                belief, logits = model.forward_beliefs(tokens[b:b + 1], return_logits=True)
                sigma = belief.sigma.detach()
                if sigma.dim() == 4:                                 # (1, N, K, K) matrix covariance
                    sigma = sigma.diagonal(dim1=-2, dim2=-1)
                variance_rows.append(sigma.float().reshape(sigma.shape[1], -1).amax(dim=-1))       # (N,)
                ce = F.cross_entropy(logits[0].float(), targets[b], ignore_index=-100, reduction="none")  # (N,)
                ce_rows.append(torch.where(targets[b] == -100, torch.full_like(ce, float("nan")), ce))
        finally:
            model.train(was_training)
    variance = torch.stack(variance_rows).cpu()                      # (B, N)
    ce = torch.stack(ce_rows).cpu()                                  # (B, N)
    n = variance.shape[1]
    out: Dict[str, Any] = {"parameters": "after the spike step's update", "mode": "eval, one sequence at a time"}
    finite_variance = torch.where(torch.isfinite(variance), variance, torch.full_like(variance, -math.inf))
    flat = int(finite_variance.reshape(-1).argmax())
    out["sigma_q_max"] = float(variance.reshape(-1)[flat])
    out["sigma_q_max_at"] = [flat // n, flat % n]
    out["sigma_q_nonfinite"] = int((~torch.isfinite(variance)).sum())
    out["ce_at_sigma_q_max"] = float(ce.reshape(-1)[flat])
    finite_ce = torch.where(torch.isnan(ce), torch.full_like(ce, -math.inf), ce)
    if torch.isfinite(finite_ce).any() or torch.isposinf(finite_ce).any():
        flat_ce = int(finite_ce.reshape(-1).argmax())
        out["ce_max"] = float(ce.reshape(-1)[flat_ce])
        out["ce_max_at"] = [flat_ce // n, flat_ce % n]
        out["sigma_q_at_ce_max"] = float(variance.reshape(-1)[flat_ce])
        out["ce_mean"] = float(ce[~torch.isnan(ce)].mean())
    else:
        out["ce_max"], out["ce_max_at"], out["sigma_q_at_ce_max"], out["ce_mean"] = None, None, None, None
    return out


def _spike_attribution(
    model:            torch.nn.Module,
    optimizer:        torch.optim.Optimizer,
    status:           Mapping[str, Any],          # train_step status_out of the spiking step
    tokens:           torch.Tensor,               # (B, N) the step's input tokens
    *,
    roles:            Sequence[str],              # the roles to attribute (the spiking roles, or all of them)
    loss_spiked:      bool,
    replayed:         bool,                       # the step was replayed at a stricter matmul precision
    last_group_norms: Mapping[str, float],        # pre-clip grad_norm_group_* of the last logged row
    targets:          Optional[torch.Tensor] = None,   # (B, N) the step's targets; counts decode-table rows
    step:             Optional[int] = None,            # completed-step number of the spiking step
    last_seen:        Optional[Mapping[str, torch.Tensor]] = None,   # "token"/"target": (V [+1],) last step seen
) -> Dict[str, Any]:
    r"""Which parameters, named groups and table rows carry a spiking role's pre-clip gradient, and what coincides.

    Reads the gradients that reached ``optimizer.step``: the references ``train_step`` keeps in ``status["grads"]``,
    since the table Fisher step and the manifold phi step clear ``.grad`` inside ``step()``, else the live
    gradients (``train_step`` zeroes them on entry, not on exit). It undoes each parameter's clip
    coefficient (``status["clip_scales"]``), so :math:`\|g_p\|` is the pre-clip norm and a role's norm is
    :math:`(\sum_{p\in r}\|g_p\|^2)^{1/2}`. Shares are of the squared role norm. For the token and positional tables
    among the top parameters, the rows with the largest gradient are reported with their share of the table's
    squared norm and, for token tables, their count in the batch (decode tables: their count among the targets),
    up to ``_SPIKE_ROW_OCCURRENCES`` of their (batch, position) occurrences and, given ``step`` and ``last_seen``
    (the last completed step at which each id occurred, -1 when not since the start or resume), the steps since the
    row's previous occurrence. For ``phi_embed`` rows the per-head condition numbers of the row's frame
    (:func:`_phi_row_head_conditions`, at the parameters after the update) are added (audit 2026-10-20 item 11).
    ``causes`` lists what coincides with the spike, without asserting a cause: the loss spike on the same step, a
    binding clip, a precision replay, one parameter or a few rows carrying most of the norm, and groups far above
    their last logged norm. One host transfer for the norms.
    """
    scales: Mapping[int, float] = status.get("clip_scales") or {}
    stashed: Mapping[int, torch.Tensor] = status.get("grads") or {}
    role_of = {id(p): group.get("role", "other") for group in optimizer.param_groups for p in group["params"]}
    wanted = set(roles)
    entries: List[Tuple[str, torch.nn.Parameter, str, str]] = []
    grad_of: Dict[str, torch.Tensor] = {}                          # the gradient each entry is read from
    norms: List[torch.Tensor] = []
    for name, parameter in model.named_parameters():
        role = role_of.get(id(parameter))
        grad = stashed.get(id(parameter), parameter.grad)
        if grad is None or role is None:
            continue
        entries.append((name, parameter, role, _parameter_group(name)))
        grad_of[name] = grad
        norms.append(torch.linalg.vector_norm(grad.detach()).to(torch.float64))
    values = torch.stack(norms).cpu().tolist() if norms else []
    pre = [value / scales.get(id(parameter), 1.0) for value, (_, parameter, _, _) in zip(values, entries)]
    report: Dict[str, Any] = {"roles": {}, "top_parameters": [], "table_rows": {}, "group_norms": {}, "causes": []}
    group_sq: Dict[str, float] = {}
    for norm, (_, _, _, group) in zip(pre, entries):
        group_sq[group] = group_sq.get(group, 0.0) + norm * norm
    report["group_norms"] = {group: math.sqrt(value) for group, value in group_sq.items()}
    role_sq: Dict[str, float] = {}
    for role in roles:
        members = [(norm, group) for norm, (_, _, r, group) in zip(pre, entries) if r == role]
        total = sum(norm * norm for norm, _ in members)
        if not members:
            continue
        role_sq[role] = total
        shares: Dict[str, float] = {}
        for norm, group in members:
            shares[group] = shares.get(group, 0.0) + (norm * norm / total if total > 0 else 0.0)
        report["roles"][role] = {"pre_clip_norm": math.sqrt(total), "group_shares": shares,
                                 "clip_coef": float((status.get("clip_coefs") or {}).get(role, 1.0))}
    ranked = sorted((item for item in zip(pre, entries) if item[1][2] in wanted), key=lambda item: -item[0])
    for norm, (name, parameter, role, group) in ranked[:_SPIKE_TOP_PARAMETERS]:
        share = norm * norm / role_sq[role] if role_sq.get(role, 0.0) > 0 else 0.0
        report["top_parameters"].append({"name": name, "role": role, "group": group, "pre_clip_norm": norm,
                                         "role_share": share})
        index = _SPIKE_TABLE_INDEX.get(group)
        if index is None and targets is not None and name.rsplit(".", 1)[-1] in _SPIKE_DECODE_TABLES:
            index = "target"
        grad = grad_of[name]
        if index is None or grad.dim() < 2:
            continue
        with torch.no_grad():
            rows = grad.detach().flatten(1).to(torch.float64).square().sum(dim=1)              # (rows,)
            k = min(_SPIKE_TOP_ROWS, rows.numel())
            top_sq, top_rows = rows.topk(k)
            entry: Dict[str, Any] = {"indexed_by": index, "rows": top_rows.cpu().tolist(),
                                     "top_share": float(top_sq.sum() / rows.sum().clamp_min(1e-300)),
                                     "rows_with_gradient": int((rows > 0).sum())}
            if index in ("token", "target"):
                source = tokens if index == "token" else targets
                ids = source.reshape(-1).to(rows.device)
                ids = ids[(ids >= 0) & (ids < rows.numel())]                  # drop ignore-index targets
                counts = torch.bincount(ids, minlength=rows.numel())
                entry["batch_counts"] = counts[top_rows].cpu().tolist()
                host_ids = source.detach().cpu()                             # (B, N)
                entry["occurrences"] = [
                    [[int(b), int(n)] for b, n in (host_ids == row).nonzero().tolist()[:_SPIKE_ROW_OCCURRENCES]]
                    for row in entry["rows"]]
                seen = None if last_seen is None else last_seen.get(index)
                if seen is not None and step is not None:
                    previous = seen[top_rows.to(seen.device)].cpu().tolist()
                    entry["steps_since_last_occurrence"] = [
                        None if last < 0 else int(step) - int(last) for last in previous]
            elif index == "position":
                entry["occurrences"] = [[[b, int(row)] for b in range(int(tokens.shape[0]))][:_SPIKE_ROW_OCCURRENCES]
                                        for row in entry["rows"]]
            if group == "phi_embed" and grad.shape[0] == parameter.shape[0]:
                try:
                    entry["head_condition"] = _phi_row_head_conditions(
                        model, parameter.detach()[top_rows.to(parameter.device)])
                except Exception as exc:                                    # noqa: BLE001 - a diagnostic
                    entry["head_condition_error"] = f"{type(exc).__name__}: {exc}"
        report["table_rows"][name] = entry
    causes: List[str] = report["causes"]
    if loss_spiked:
        causes.append("the training loss spiked on the same step")
    if replayed:
        causes.append("the step was replayed at a stricter matmul precision")
    for role, info in report["roles"].items():
        if info["clip_coef"] < 1.0:
            causes.append(f"clip binding on {role} (coefficient {info['clip_coef']:.3g})")
    for entry in report["top_parameters"]:
        if entry["role_share"] >= 0.5:
            causes.append(f"{entry['name']} carries {entry['role_share']:.0%} of ||g_{entry['role']}||^2")
    for name, entry in report["table_rows"].items():
        if entry["top_share"] >= 0.5:
            counts = f", batch counts {entry['batch_counts']}" if "batch_counts" in entry else ""
            if "occurrences" in entry:
                counts += ", first (batch, position) " + str(
                    [tuple(found[0]) if found else None for found in entry["occurrences"]])
            if "steps_since_last_occurrence" in entry:
                counts += f", steps since last occurrence {entry['steps_since_last_occurrence']}"
            if "head_condition" in entry:
                counts += ", max head cond(U) " + str(
                    [float(f"{max(conds):.3g}") if conds else None for conds in entry["head_condition"]])
            causes.append(f"{name} {entry['indexed_by']} rows {entry['rows']} carry {entry['top_share']:.0%} "
                          f"of its gradient{counts}")
    for group, norm in sorted(report["group_norms"].items()):
        last = last_group_norms.get(group)
        if last is not None and math.isfinite(last) and last > 0 and norm > 10.0 * last:
            causes.append(f"group {group} is {norm / last:.0f}x its last logged norm {last:.3g}")
    return report


def _spike_summary(report: Mapping[str, Any]) -> str:
    r"""One log line of :func:`_spike_attribution`: role norms by group, the top parameters and the causes."""
    parts = []
    for role, info in report["roles"].items():
        groups = ", ".join(f"{group} {share:.0%}" for group, share in
                           sorted(info["group_shares"].items(), key=lambda item: -item[1]) if share >= 0.01)
        parts.append(f"{role} {info['pre_clip_norm']:.4g} [{groups}]")
    top = ", ".join(f"{entry['name']} {entry['pre_clip_norm']:.3g} ({entry['role_share']:.0%} of {entry['role']})"
                    for entry in report["top_parameters"][:3])
    causes = "; ".join(report["causes"]) or "none identified"
    forward = report.get("forward_state")
    state = ""
    if isinstance(forward, Mapping) and "sigma_q_max" in forward:
        state = (f"; post-update forward: max sigma_q {forward['sigma_q_max']:.4g} at (batch, position) "
                 f"{tuple(forward['sigma_q_max_at'])}")
        if forward.get("ce_max") is not None:
            state += (f", max per-position CE {forward['ce_max']:.4g} at {tuple(forward['ce_max_at'])} "
                      f"(batch mean {forward['ce_mean']:.4g})")
    return f"{' | '.join(parts)}; top: {top}; coincident: {causes}{state}"


def _append_spike_record(
    run_dir: Path,
    record:  Mapping[str, object],
) -> Path:
    r"""Append one JSON line to ``run_dir/diagnostics/divergence_spikes.jsonl``."""
    directory = Path(run_dir) / "diagnostics"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / DIVERGENCE_SPIKE_LOG
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(record), sort_keys=True) + "\n")
    return path


class _DivergenceMonitor:
    r"""Interval statistics and spike detection over the host values of every training step.

    The interval is the steps since the previous log or eval row (a forced row does not end it). A step
    spikes when a role's pre-clip norm exceeds ``DIVERGENCE_NORM_SPIKE_RATIO`` times its EMA, or the
    training loss exceeds its EMA by more than ``DIVERGENCE_LOSS_SPIKE_MARGIN`` nats, each tested against
    the EMA before the step's own value enters it and only after ``DIVERGENCE_EMA_WARMUP`` finite values,

    .. math::
        \bar x_t = \rho\,\bar x_{t-1} + (1 - \rho)\,x_t,\qquad \rho = 0.98.

    Non-finite values enter neither the EMAs nor the interval statistics (the rejected-update accounting
    covers them). A spiking channel's EMA is frozen on that step, :math:`\bar x_t = \bar x_{t-1}`, so a burst of
    spikes cannot ratchet it. The earlier bounded update :math:`\min(x, 10\,\bar x)` grew a norm EMA by a factor
    of up to :math:`0.98 + 0.02\cdot 10 = 1.18` per spiking step, which carried one run's mu-norm EMA from 0.86 to
    1.19e4 and stopped flagging late steps (audit 2026-10-20, spike report section 7; before that, audit
    2026-10-17 item 15). The warmup count still advances on a frozen step. A sustained shift of a level above the
    threshold is therefore an event on every step and is reported once per cooldown until the level falls back.

    The interval statistics keep the loss and CE values, so a row carries their interval median beside the mean,
    which a single blow-up dominates.

    Every spike is an *event* (:attr:`event`, recorded by the caller in ``divergence_spikes.jsonl``). Its severity
    is the largest of :math:`x/(10\,\bar x)` over the spiking norms and :math:`(\ell - \bar\ell)/1` for the loss, so
    a spike has severity above 1. An event is *reported* (console line and forced metrics row; the return value of
    :meth:`observe`) at most once per ``DIVERGENCE_SPIKE_COOLDOWN`` steps, except that an event inside the cooldown
    whose severity exceeds ``DIVERGENCE_SPIKE_ESCALATION`` times that of the last reported one is reported too, so
    the worst spike of a burst is never the one dropped.
    """

    def __init__(self) -> None:
        self._loss_ema: Optional[float] = None
        self._loss_seen = 0
        self._norm_ema: Dict[str, float] = {}
        self._norm_seen: Dict[str, int] = {}
        self._last_spike: Optional[int] = None
        self._last_severity: Optional[float] = None
        self.spike_roles: Tuple[str, ...] = ()
        self.spike_loss = False
        self.event: Optional[str] = None               # this step's spike description, reported or not
        self.event_severity: float = float("nan")
        self.reset_interval()

    def loss_skip_threshold(
        self,
        margin: float,                     # nats above the loss EMA (loss_spike_skip_nats)
    ) -> Optional[float]:
        r"""The loss above which ``loss_spike_skip_nats`` rejects the next update, :math:`\bar\ell + m`, once the loss
        EMA has seen ``DIVERGENCE_EMA_WARMUP`` finite values; None before (audit 2026-10-17 item 17)."""
        if self._loss_ema is None or self._loss_seen < DIVERGENCE_EMA_WARMUP:
            return None
        return float(self._loss_ema) + float(margin)

    def reset_interval(self) -> None:
        self.steps = 0
        self._loss = _RunningStat(keep_values=True)
        self._ce = _RunningStat(keep_values=True)
        self._norms = {role: _RunningStat() for role in DIVERGENCE_ROLES}
        self._clipped = 0
        self._binding = 0

    def observe(
        self,
        step:   int,                       # completed-step number of this step
        loss:   float,
        status: Mapping[str, Any],         # train_step status_out of this step
    ) -> Optional[str]:
        r"""Add one step to the interval and the EMAs; return a spike description, or ``None``."""
        self.steps += 1
        self._loss.add(loss)
        self._ce.add(float(status.get("ce", float("nan"))))
        norms: Mapping[str, float] = status.get("role_grad_norms") or {}
        for role in DIVERGENCE_ROLES:
            if role in norms:
                self._norms[role].add(float(norms[role]))
        coefficients: Mapping[str, float] = status.get("clip_coefs") or {}
        if coefficients:
            self._clipped += 1
            self._binding += int(any(value < 1.0 for value in coefficients.values()))
        reasons: List[str] = []
        roles: List[str] = []
        loss_spiked = False
        severity = 0.0
        for role, value in sorted(norms.items()):
            if not math.isfinite(value):
                continue
            ema, seen = self._norm_ema.get(role), self._norm_seen.get(role, 0)
            update = value
            if (ema is not None and seen >= DIVERGENCE_EMA_WARMUP and ema > 0.0
                    and value > DIVERGENCE_NORM_SPIKE_RATIO * ema):
                reasons.append(f"pre-clip grad_norm_{role} {value:.4g} is {value / ema:.1f}x its EMA {ema:.4g}")
                roles.append(role)
                severity = max(severity, value / (DIVERGENCE_NORM_SPIKE_RATIO * ema))
                update = ema                                             # frozen on a spike (no ratchet)
            self._norm_ema[role] = value if ema is None else (
                DIVERGENCE_EMA_DECAY * ema + (1.0 - DIVERGENCE_EMA_DECAY) * update)
            self._norm_seen[role] = seen + 1
        if math.isfinite(loss):
            ema = self._loss_ema
            update = loss
            if (ema is not None and self._loss_seen >= DIVERGENCE_EMA_WARMUP
                    and loss > ema + DIVERGENCE_LOSS_SPIKE_MARGIN):
                reasons.append(f"train loss {loss:.4f} exceeds its EMA {ema:.4f} by {loss - ema:.3f} nats")
                loss_spiked = True
                severity = max(severity, ((loss - ema) / DIVERGENCE_LOSS_SPIKE_MARGIN
                                          if DIVERGENCE_LOSS_SPIKE_MARGIN > 0 else math.inf))
                update = ema                                             # frozen on a spike (no ratchet)
            self._loss_ema = loss if ema is None else (
                DIVERGENCE_EMA_DECAY * ema + (1.0 - DIVERGENCE_EMA_DECAY) * update)
            self._loss_seen += 1
        self.event = "; ".join(reasons) if reasons else None
        self.event_severity = severity if reasons else float("nan")
        if not reasons:
            return None
        # Read by the spike attribution: the roles over their EMA, and whether the loss spiked too.
        self.spike_roles = tuple(roles)
        self.spike_loss = loss_spiked
        outside_cooldown = self._last_spike is None or step - self._last_spike >= DIVERGENCE_SPIKE_COOLDOWN
        escalated = (self._last_severity is not None
                     and severity > DIVERGENCE_SPIKE_ESCALATION * self._last_severity)
        if outside_cooldown or escalated:
            self._last_spike = step
            self._last_severity = severity
            return self.event
        return None

    def columns(
        self,
        status:  Mapping[str, Any],        # train_step status_out of the row's own step
        flagged: bool,                     # the row's step spiked
    ) -> Dict[str, float]:
        r"""The ``DIVERGENCE_COLUMNS`` of a metrics row (NaN where nothing was measured)."""
        out: Dict[str, float] = {
            "interval_steps": float(self.steps),
            "train_loss_interval_mean": self._loss.mean(),
            "train_loss_interval_max": self._loss.max(),
            "train_ce_interval_mean": self._ce.mean(),
            "train_loss_interval_median": self._loss.median(),
            "train_ce_interval_median": self._ce.median(),
        }
        for role in DIVERGENCE_ROLES:
            out[f"grad_norm_{role}_interval_mean"] = self._norms[role].mean()
            out[f"grad_norm_{role}_interval_max"] = self._norms[role].max()
        out["grad_clip_binding_frac"] = (self._binding / self._clipped if self._clipped else float("nan"))
        coefficients: Mapping[str, float] = status.get("clip_coefs") or {}
        for role in DIVERGENCE_ROLES:
            out[f"grad_clip_coef_{role}"] = float(coefficients.get(role, float("nan")))
        out["divergence_spike"] = float(flagged)
        return out

    def state_dict(
        self,
        *,
        interval_ends: bool,               # this step writes a log/eval row, which ends the open interval
    ) -> Dict[str, Any]:
        r"""The host state a resumed run needs to continue the EMAs, warmups, cooldown and interval exactly.

        A periodic checkpoint is written before its step's metrics row, so on a log/eval step
        (``interval_ends``) the saved interval is the empty one :meth:`reset_interval` leaves after the row.
        Plain floats, ints and dicts, so the record loads under ``torch.load(weights_only=True)``.
        """
        def stat(value: _RunningStat) -> List[Any]:
            return [value.count, value.total, value.peak]

        def kept(value: _RunningStat) -> List[float]:
            return list(value.values or [])

        source = _DivergenceMonitor() if interval_ends else self
        return {
            "loss_ema":   self._loss_ema,
            "loss_seen":  self._loss_seen,
            "norm_ema":   dict(self._norm_ema),
            "norm_seen":  dict(self._norm_seen),
            "last_spike": self._last_spike,
            "last_severity": self._last_severity,
            "interval": {
                "steps":   source.steps,
                "loss":    stat(source._loss),
                "ce":      stat(source._ce),
                "loss_values": kept(source._loss),       # for the interval medians (absent in older checkpoints)
                "ce_values":   kept(source._ce),
                "norms":   {role: stat(source._norms[role]) for role in DIVERGENCE_ROLES},
                "clipped": source._clipped,
                "binding": source._binding,
            },
        }

    def load_state_dict(
        self,
        state: object,                     # a state_dict() record, possibly from an older or malformed checkpoint
    ) -> bool:
        r"""Restore :meth:`state_dict`; a malformed record leaves the monitor fresh and returns False."""
        monitor = _divergence_monitor_state(state)
        if monitor is None:
            return False

        def stat(values: List[Any], kept: Optional[List[float]] = None, keep: bool = False) -> _RunningStat:
            out = _RunningStat(keep_values=keep)
            out.count, out.total, out.peak = values
            if keep:
                out.values = list(kept or [])      # an older checkpoint: the median covers the resumed part only
            return out

        interval = monitor["interval"]
        self._loss_ema, self._loss_seen = monitor["loss_ema"], monitor["loss_seen"]
        self._norm_ema, self._norm_seen = dict(monitor["norm_ema"]), dict(monitor["norm_seen"])
        self._last_spike = monitor["last_spike"]
        self._last_severity = monitor.get("last_severity")
        self.steps = interval["steps"]
        self._loss = stat(interval["loss"], interval["loss_values"], keep=True)
        self._ce = stat(interval["ce"], interval["ce_values"], keep=True)
        self._norms = {role: stat(interval["norms"][role]) for role in DIVERGENCE_ROLES}
        self._clipped, self._binding = interval["clipped"], interval["binding"]
        return True


DIVERGENCE_STATE_SCHEMA: int = 1          # checkpoint "divergence_monitor" record (audit 2026-10-16 F77)


def _count(value: object) -> Optional[int]:
    r"""``value`` when it is an int >= 0 (not a bool), else None."""
    return value if type(value) is int and value >= 0 else None


def _real(value: object) -> Optional[float]:
    r"""``value`` as a float when it is a non-bool int or float, else None (NaN and infinities pass)."""
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _divergence_monitor_state(
    value: object,
) -> Optional[Dict[str, Any]]:
    r"""A validated copy of :meth:`_DivergenceMonitor.state_dict`, or None when anything is malformed."""
    if not isinstance(value, Mapping):
        return None

    def stat(item: object) -> Optional[List[Any]]:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            return None
        count, total, peak = _count(item[0]), _real(item[1]), _real(item[2])
        return None if count is None or total is None or peak is None else [count, total, peak]

    def table(item: object, kind: Callable[[object], Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(item, Mapping):
            return None
        out = {key: kind(entry) for key, entry in item.items()}
        return out if all(isinstance(key, str) and entry is not None for key, entry in out.items()) else None

    loss_ema = value.get("loss_ema")
    last_spike = value.get("last_spike")
    last_severity = value.get("last_severity")              # absent before audit 2026-10-17 item 15
    interval = value.get("interval")
    out: Dict[str, Any] = {
        "loss_ema":   None if loss_ema is None else _real(loss_ema),
        "loss_seen":  _count(value.get("loss_seen")),
        "norm_ema":   table(value.get("norm_ema"), _real),
        "norm_seen":  table(value.get("norm_seen"), _count),
        "last_spike": None if last_spike is None else _count(last_spike),
        "last_severity": None if last_severity is None else _real(last_severity),
    }
    if (out["loss_seen"] is None or out["norm_ema"] is None or out["norm_seen"] is None
            or (loss_ema is not None and out["loss_ema"] is None)
            or (last_spike is not None and out["last_spike"] is None)
            or (last_severity is not None and out["last_severity"] is None) or not isinstance(interval, Mapping)):
        return None
    norms = interval.get("norms")
    if not isinstance(norms, Mapping):
        return None
    def kept(item: object) -> Optional[List[float]]:
        if item is None:                                     # absent before the interval medians
            return []
        if not isinstance(item, (list, tuple)):
            return None
        values = [_real(entry) for entry in item]
        return None if any(entry is None for entry in values) else values

    restored = {
        "steps":   _count(interval.get("steps")),
        "loss":    stat(interval.get("loss")),
        "ce":      stat(interval.get("ce")),
        "loss_values": kept(interval.get("loss_values")),
        "ce_values":   kept(interval.get("ce_values")),
        "norms":   {role: stat(norms.get(role)) for role in DIVERGENCE_ROLES},
        "clipped": _count(interval.get("clipped")),
        "binding": _count(interval.get("binding")),
    }
    if (any(restored[key] is None for key in ("steps", "loss", "ce", "loss_values", "ce_values", "clipped",
                                              "binding"))
            or any(entry is None for entry in restored["norms"].values())):
        return None
    out["interval"] = restored
    return out


def _divergence_checkpoint_record(
    divergence:        _DivergenceMonitor,
    *,
    interval_ends:     bool,                     # the checkpoint step writes a log/eval row after the save
    skipped_steps:     int,                      # rejected updates so far (whole run)
    consecutive_skips: int,
    last_group_norms:  Mapping[str, float],      # pre-clip group norms of the last logged step
    failed_step_skips: int = 0,                  # steps skipped under max_failed_step_skips (whole run)
    loss_spike_rejections: int = 0,              # updates rejected under loss_spike_skip_nats (whole run)
) -> Dict[str, Any]:
    r"""The checkpoint's ``divergence_monitor`` entry: the monitor state plus the host counters beside it."""
    return {"schema": DIVERGENCE_STATE_SCHEMA,
            "monitor": divergence.state_dict(interval_ends=interval_ends),
            "skipped_steps": int(skipped_steps), "consecutive_skips": int(consecutive_skips),
            "last_group_norms": {str(key): float(norm) for key, norm in last_group_norms.items()},
            "failed_step_skips": int(failed_step_skips), "loss_spike_rejections": int(loss_spike_rejections)}


def _checkpoint_divergence_state(
    value: object,
) -> Optional[Dict[str, Any]]:
    r"""A validated copy of a :func:`_divergence_checkpoint_record`, or None.

    Tolerant like the checkpoint's numerics counters: the record is a diagnostic, so a missing or malformed
    one (an older checkpoint) makes the resumed run restart its monitor instead of failing the resume.
    """
    if not isinstance(value, Mapping) or value.get("schema") != DIVERGENCE_STATE_SCHEMA:
        return None
    monitor = _divergence_monitor_state(value.get("monitor"))
    skipped, consecutive = _count(value.get("skipped_steps")), _count(value.get("consecutive_skips"))
    norms = value.get("last_group_norms")
    if monitor is None or skipped is None or consecutive is None or not isinstance(norms, Mapping):
        return None
    group_norms = {key: _real(norm) for key, norm in norms.items()}
    if not all(isinstance(key, str) and norm is not None for key, norm in group_norms.items()):
        return None
    # Counters added by audit 2026-10-17 items 16 and 17; an older record carries none (0).
    extras = {key: _count(value.get(key, 0)) for key in ("failed_step_skips", "loss_spike_rejections")}
    return {"schema": DIVERGENCE_STATE_SCHEMA, "monitor": monitor, "skipped_steps": skipped,
            "consecutive_skips": consecutive, "last_group_norms": group_norms,
            **{key: (count if count is not None else 0) for key, count in extras.items()}}


def _conform_row(
    row:    Mapping[str, Any],
    fields: Sequence[str],
) -> Dict[str, Any]:
    r"""``row`` on the column order ``fields``, NaN (a blank cell) where it has no value. Columns the row has
    beyond ``fields`` are kept at the end, so ``RunArtifacts.log_metrics`` still refuses a real schema change."""
    return {**{key: float("nan") for key in fields}, **row}


def _table_fisher_learning_rates(
    optimizer: torch.optim.Optimizer,
) -> Dict[str, float]:
    r"""Realized table step size :math:`\eta\,\mathrm{lr}_t/\mathrm{lr}_0` of each ``table_fisher`` group that carries
    the ``lr_mu``/``lr_sigma`` report role; empty for any other optimizer (audit 2026-10-10 round 2, items 20, 37).
    Under ``m_table_update_mode='fisher'`` the AdamW rate of these groups only schedules the step and sets its decay."""
    eta = getattr(optimizer, "_table_fisher_lr", None)
    if eta is None:
        return {}
    from vfe3.table_optim import _schedule_factor
    return {str(group["lr_report_role"]): float(eta) * _schedule_factor(group)
            for group in optimizer.param_groups if group.get("table_fisher") and "lr_report_role" in group}


CRASH_RECORD_NAME = "crash.json"                      # run_dir/diagnostics: the record of a failed run (item 4)


def _json_safe(value: Any) -> Any:
    r"""``value`` with non-finite floats as None and tensors as lists, recursively, for a strict JSON record."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    return str(value)


def _write_crash_record(
    directory: Path,
    record:    Mapping[str, Any],
) -> Path:
    r"""Atomically write ``directory/crash.json`` (:data:`CRASH_RECORD_NAME`)."""
    from vfe3.run_artifacts import _write_json_atomic
    return _write_json_atomic(Path(directory) / CRASH_RECORD_NAME, _json_safe(record))


class _PeakMemory:
    r"""CUDA peak allocation split into training and non-training phases (audit 2026-10-17 item 49).

    ``torch.cuda.max_memory_allocated`` is a host counter (no synchronization). The diagnostics forward, the
    evaluation block and the other off-step probes are bracketed by :meth:`begin_other` / :meth:`end_other`, which
    read and reset the peak at each boundary, so :meth:`sample` returns the overall peak since the previous row (the
    historical ``peak_mem_mb``) and the peak of the training phases alone (``peak_mem_train_mb``). NaN off CUDA.
    """

    def __init__(self, device: torch.device) -> None:
        self._cuda = device.type == "cuda"
        self._device = device
        self._train = 0.0
        self._other = 0.0

    def _read_and_reset(self) -> float:
        value = torch.cuda.max_memory_allocated(self._device) / 1e6
        torch.cuda.reset_peak_memory_stats(self._device)
        return float(value)

    def begin_other(self) -> None:
        if self._cuda:
            self._train = max(self._train, self._read_and_reset())

    def end_other(self) -> None:
        if self._cuda:
            self._other = max(self._other, self._read_and_reset())

    def sample(self) -> Tuple[float, float]:
        r"""``(overall, training-only)`` peak MB since the previous sample, then start a new window."""
        if not self._cuda:
            return float("nan"), float("nan")
        train = max(self._train, self._read_and_reset())
        overall = max(train, self._other)
        self._train = self._other = 0.0
        return overall, train


# Allocator and clock telemetry on every metrics row (audit 2026-10-20 item 20). NaN off CUDA and where a reading is
# unavailable (no NVML and no nvidia-smi, an older torch without a counter).
DEVICE_TELEMETRY_COLUMNS: Tuple[str, ...] = (
    "cuda_mem_allocated_mb", "cuda_mem_reserved_mb", "cuda_mem_reserved_peak_mb", "cuda_mem_inactive_split_mb",
    "cuda_alloc_retries", "cuda_device_allocs", "cuda_device_frees", "cuda_ooms",
    "gpu_sm_clock_mhz", "gpu_temperature_c", "gpu_power_w",
)
_NVIDIA_SMI_TIMEOUT_S: float = 5.0


class _DeviceTelemetry:
    r"""Allocator counters and SM clock, temperature and power of the training device, read once per metrics row.

    The allocator columns come from ``torch.cuda.memory_stats`` (host counters, no synchronization): current allocated,
    reserved and inactive-split bytes, the reserved peak since the last peak reset (the peak-memory split resets the
    peaks at each phase boundary, so this is the peak of the last phase window), and the cumulative counts of allocator
    retries, ``cudaMalloc`` and ``cudaFree`` calls and OOMs. A rising retry or device-allocation count with flat peak
    memory points at allocator churn; a falling SM clock at the same utilization points at clock drift.

    The clock columns read NVML through ``pynvml`` when it imports, else one ``nvidia-smi --query-gpu`` call per row.
    The device is matched by its UUID, so ``CUDA_VISIBLE_DEVICES`` remapping does not misattribute it. A failed or
    slow (more than ``_NVIDIA_SMI_TIMEOUT_S``) query disables the clock source for the rest of the run; the clock
    columns are then NaN. Nothing here raises.
    """

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._cuda = device.type == "cuda"
        self._source: Optional[str] = None if self._cuda else "off"     # "nvml" | "smi" | "off", chosen lazily
        self._nvml_handle: Any = None
        self._uuid: Optional[str] = None

    def _init_clock_source(self) -> None:
        try:
            raw = str(torch.cuda.get_device_properties(self._device).uuid)
            self._uuid = raw if raw.startswith("GPU-") else f"GPU-{raw}"
        except Exception:                                     # noqa: BLE001 - telemetry, never fatal
            self._uuid = None
        try:
            import pynvml                                     # type: ignore[import-not-found]
            pynvml.nvmlInit()
            index = self._device.index if self._device.index is not None else torch.cuda.current_device()
            self._nvml_handle = (pynvml.nvmlDeviceGetHandleByUUID(self._uuid) if self._uuid is not None
                                 else pynvml.nvmlDeviceGetHandleByIndex(index))
            self._source = "nvml"
            return
        except Exception:                                     # noqa: BLE001 - pynvml absent or NVML failed
            self._nvml_handle = None
        import shutil
        self._source = "smi" if shutil.which("nvidia-smi") is not None else "off"

    def _clock(self) -> Dict[str, float]:
        nan = float("nan")
        out = {"gpu_sm_clock_mhz": nan, "gpu_temperature_c": nan, "gpu_power_w": nan}
        if self._source is None:
            self._init_clock_source()
        if self._source == "nvml":
            try:
                import pynvml                                 # type: ignore[import-not-found]
                handle = self._nvml_handle
                out["gpu_sm_clock_mhz"] = float(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))
                out["gpu_temperature_c"] = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
                out["gpu_power_w"] = float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1e3
            except Exception:                                 # noqa: BLE001
                self._source = "off"
        elif self._source == "smi":
            import subprocess
            command = ["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw",
                       "--format=csv,noheader,nounits"]
            if self._uuid is not None:
                command += ["-i", self._uuid]
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=_NVIDIA_SMI_TIMEOUT_S,
                                        check=True)
                fields = [field.strip() for field in result.stdout.strip().splitlines()[0].split(",")]
                for key, field in zip(("gpu_sm_clock_mhz", "gpu_temperature_c", "gpu_power_w"), fields):
                    try:
                        out[key] = float(field)
                    except ValueError:
                        out[key] = nan                        # "[N/A]" on boards that do not report it
            except Exception:                                 # noqa: BLE001 - missing tool, timeout, parse error
                self._source = "off"
        return out

    def sample(self) -> Dict[str, float]:
        r"""The :data:`DEVICE_TELEMETRY_COLUMNS` of one metrics row (NaN off CUDA)."""
        nan = float("nan")
        out = {column: nan for column in DEVICE_TELEMETRY_COLUMNS}
        if not self._cuda:
            return out
        try:
            stats = torch.cuda.memory_stats(self._device)

            def read(key: str, scale: float = 1.0) -> float:
                value = stats.get(key)
                return float(value) / scale if value is not None else nan

            out.update({
                "cuda_mem_allocated_mb": read("allocated_bytes.all.current", 1e6),
                "cuda_mem_reserved_mb": read("reserved_bytes.all.current", 1e6),
                "cuda_mem_reserved_peak_mb": read("reserved_bytes.all.peak", 1e6),
                "cuda_mem_inactive_split_mb": read("inactive_split_bytes.all.current", 1e6),
                "cuda_alloc_retries": read("num_alloc_retries"),
                "cuda_device_allocs": read("num_device_alloc"),
                "cuda_device_frees": read("num_device_free"),
                "cuda_ooms": read("num_ooms"),
            })
        except Exception:                                     # noqa: BLE001 - telemetry, never fatal
            pass
        out.update(self._clock())
        return out


RUN_LOG_NAME = "train.log"                            # run_dir: the run's log records (audit 2026-10-17 item 4)
_RUN_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def _attach_run_log(
    artifacts: Optional["RunArtifacts"],
    logger:    Optional[logging.Logger],
) -> Optional[Tuple[logging.Handler, List[logging.Logger]]]:
    r"""Attach a ``run_dir/train.log`` file handler for the duration of :func:`train`; None without a run directory.

    The handler sits on the root logger, so every propagating logger of the run (the training logger, the artifact
    and numerics modules) reaches it, and also on ``logger`` when that one does not propagate. Records are written at
    INFO and above, as far as each logger's own level lets them through.
    """
    run_dir = getattr(artifacts, "run_dir", None)
    if run_dir is None:
        return None
    try:
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(Path(run_dir) / RUN_LOG_NAME, mode="a", encoding="utf-8", delay=False)
    except OSError:
        return None
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(_RUN_LOG_FORMAT))
    targets = [logging.getLogger()]
    if logger is not None and not logger.propagate:
        targets.append(logger)
    for target in targets:
        target.addHandler(handler)
    return handler, targets


def _detach_run_log(
    attached: Optional[Tuple[logging.Handler, List[logging.Logger]]],
) -> None:
    r"""Remove and close the handler of :func:`_attach_run_log`."""
    if attached is None:
        return
    handler, targets = attached
    for target in targets:
        target.removeHandler(handler)
    handler.close()


def train(
    model:  VFEModel,
    loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],   # yields (tokens, targets) batches
    cfg:    VFE3Config,

    *,
    n_steps:   Optional[int]   = None,   # None -> cfg.max_steps
    grad_clip: Optional[float] = None,   # None -> cfg.grad_clip (pass 0.0 to disable clipping)

    log_interval:    Optional[int]            = None,   # None -> cfg.log_interval (0 = silent)
    eval_interval:   Optional[int]            = None,   # None -> cfg.eval_interval (0 = off)
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

    ``n_steps``, ``grad_clip``, ``log_interval`` and ``eval_interval`` left ``None`` resolve from ``cfg``
    (``max_steps``, ``grad_clip``, ``log_interval``, ``eval_interval``), so a caller passing only ``cfg`` trains
    what ``config.json`` records (audit 2026-10-17, item 54). With ``log_interval`` and ``eval_interval`` resolved to
    ``0`` the loop is bitwise-identical to the silent path: the two truthiness-guarded blocks
    short-circuit, drawing no RNG, running no extra forward, and printing nothing. When
    ``log_interval`` is positive a per-step line is emitted every
    ``log_interval`` steps (CE and diagnostics recomputed under ``no_grad`` only at those
    steps, off the training graph), AND -- when ``tqdm`` is installed -- the step loop runs
    under a ``tqdm`` progress bar whose built-in rate readout shows live ``it/s`` every step
    (the formatted lines render above it via ``logging_redirect_tqdm``); when ``eval_interval``
    is positive and ``val_loader`` is given a validation block is emitted every
    ``eval_interval`` steps.

    Every step feeds a host-side divergence monitor (:class:`_DivergenceMonitor`) from values
    ``train_step`` already transfers, so the trajectory is unchanged: each metrics row carries the
    ``DIVERGENCE_COLUMNS`` (loss, CE and per-role pre-clip norm statistics since the previous row,
    the clip-binding fraction and the realized clip coefficients), and a reported spike logs a warning and
    forces a metrics row at its step (audit 2026-10-10 round 2, item 1); every spike is recorded in
    ``diagnostics/divergence_spikes.jsonl`` (audit 2026-10-17 item 15).

    With ``artifacts`` the run's log records also go to ``run_dir/train.log`` while this call runs, and an exception
    of any kind in the step loop (a training step, an evaluation, a keyboard interrupt) first writes
    ``diagnostics/crash.json``, the failing batch and, where the in-memory state is a step boundary, a resumable
    ``checkpoints/step_<S>.pt`` (:func:`_train_loop`; audit 2026-10-17 item 4), then propagates. Background
    ``best_model.pt`` saves and figure workers are reaped before this call returns or raises.
    """
    attached = _attach_run_log(artifacts, logger)
    try:
        return _train_loop(
            model, loader, cfg, n_steps=n_steps, grad_clip=grad_clip, log_interval=log_interval,
            eval_interval=eval_interval, val_loader=val_loader, tokens_per_char=tokens_per_char, device=device,
            logger=logger, artifacts=artifacts, resume_from=resume_from, terminal_callback=terminal_callback,
            _final_validation_out=_final_validation_out, generate_samples=generate_samples,
            sample_decode=sample_decode, sample_dataset=sample_dataset, sample_new_tokens=sample_new_tokens,
            sample_prompt_len=sample_prompt_len)
    finally:
        try:
            wait = getattr(artifacts, "wait_background", None)
            if callable(wait):
                wait()
        finally:
            _detach_run_log(attached)


def _train_loop(
    model:  VFEModel,
    loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],   # yields (tokens, targets) batches
    cfg:    VFE3Config,

    *,
    n_steps:   int             = 100,
    grad_clip: Optional[float] = 1.0,

    log_interval:    Optional[int]            = None,
    eval_interval:   Optional[int]            = None,
    val_loader:      Optional[Iterable]       = None,
    tokens_per_char: Optional[float]           = None,
    device:          Optional[torch.device]   = None,
    logger:          Optional[logging.Logger] = None,
    artifacts:       Optional["RunArtifacts"] = None,
    resume_from:     'Optional[str | Path]'   = None,
    terminal_callback: Optional[Callable[[TrainingTerminalState, List[float]], None]] = None,
    _final_validation_out: Optional[Dict[str, Optional[float]]] = None,

    generate_samples:  bool                                     = True,
    sample_decode:     Optional[Callable[[Sequence[int]], str]] = None,
    sample_dataset:    Optional[str]                            = None,
    sample_new_tokens: int                                      = 40,
    sample_prompt_len: int                                      = 6,
) -> List[float]:
    r"""The body of :func:`train` (same arguments), run while ``train.log`` is attached.

    Crash handling (audit 2026-10-17 item 4): the step loop runs under ``try/except BaseException``. On an exception it
    writes ``diagnostics/crash.json`` (step, epoch, batches consumed, the loop phase, the exception with its notes and
    traceback, the precision-replay records of the failing step, the last ``log_interval`` per-step losses), saves
    the failing ``(tokens, targets)`` batch to ``diagnostics/crash_batch_step_<S>.pt``, flushes pending metrics rows
    and replay records, waits for background saves, and writes a resumable checkpoint when the state is a step
    boundary: before the optimizer's step pre-hook fired for the failing step the state is exactly that after
    :math:`S` = (failing index) completed steps, saved as ``checkpoints/step_<S>.pt`` with ``batches_consumed - 1``
    and the RNG streams of the step's start; after the step returned (an evaluation or a row failure) it is that
    after :math:`S + 1` steps. An update that started without the step returning leaves no step boundary and no
    checkpoint (the record says why). Then the exception propagates unchanged.
    """
    n_steps = cfg.max_steps if n_steps is None else n_steps
    grad_clip = cfg.grad_clip if grad_clip is None else grad_clip
    log_interval = cfg.log_interval if log_interval is None else log_interval
    eval_interval = cfg.eval_interval if eval_interval is None else eval_interval
    # Never expose a stale result after a partial/resumed run or a failed call.
    if _final_validation_out is not None:
        _final_validation_out.clear()
    _require_causal_next_token_prior(model)
    # Process-global; torch reads it only under deterministic algorithms. Set by the training run, not
    # by model construction, so code that only builds models keeps torch's NaN fill of uninitialized reads.
    disable_fill_uninitialized_memory()
    terminal_validation_signature = None
    terminal_validation_metrics = None
    # (step, metrics, parameter/buffer versions) of the latest eval, handed to terminal_callback (audit 2026-10-12 #12).
    last_eval_record: Optional[Tuple[int, Dict[str, Any], tuple]] = None
    optimizer = build_optimizer(model, cfg)
    # Warmup/cosine multiplier, floored per group so each group's ABSOLUTE LR never decays below
    # cfg.min_lr_frac * base -- see _floor_lr_lambdas. With cfg.min_lr_frac=0
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
    offset_windows = _configure_train_window_offset(loader)
    offset_generator = (loader_generator if shuffled_loader and isinstance(loader_generator, torch.Generator)
                        else None)                           # None: the global CPU RNG
    try:
        steps_per_epoch = int(len(loader))  # type: ignore[arg-type]
        sized_loader = True
    except (TypeError, AttributeError):
        steps_per_epoch, sized_loader = 0, False
    if steps_per_epoch < 0:
        raise ValueError(f"loader length must be nonnegative, got {steps_per_epoch}")
    if sized_loader and steps_per_epoch == 0 and n_steps > 0:
        # Every epoch would be empty: refuse before step 0 rather than die on a bare StopIteration.
        raise ValueError(_empty_training_epoch_message(loader))
    periodic_checkpoint_requested = bool(
        artifacts is not None
        and ((cfg.checkpoint_interval > 0 and n_steps >= cfg.checkpoint_interval)
             or bool(getattr(cfg, "checkpoint_at_end", False)))
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
    # Train steps can replay on the precision ladder (_train_step_with_precision_retry): a reduced fp32 matmul
    # policy without the fp16 GradScaler.
    precision_ladder = torch.get_float32_matmul_precision() != "highest" and not scaler.is_enabled()
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
    # Data-dependent prototype template initialization (block_mlp_template_init='working_variance'): a fresh
    # start only, on cfg.batch_size training windows drawn by a private generator, so the data order and the
    # global RNG streams are unchanged (audit 2026-10-07, F77/F78).
    if (resume_path is None and getattr(cfg, "block_mlp_template_init", "absolute") == "working_variance"
            and getattr(cfg, "use_block_mlp", False) and getattr(cfg, "block_mlp_mode", None) == "prototype_agents"):
        dataset = getattr(loader, "dataset", None)
        if dataset is None or not hasattr(dataset, "__getitem__"):
            raise RuntimeError("block_mlp_template_init='working_variance' needs a window dataset on the loader")
        count = min(int(cfg.batch_size), len(dataset))
        picks = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(int(cfg.seed)))[:count]
        windows = torch.stack([dataset[int(i)][0] for i in picks]).to(device)
        calibration = model.calibrate_prototype_templates(
            windows, pull=float(cfg.block_mlp_template_init_pull), seed=int(cfg.seed))
        for layer, stats in enumerate(calibration):
            logger.info(" prototype templates, layer %d: working variance %.4g, lambda_0 %.4g",
                        layer, stats["working_variance"], stats["lambda0"])
        if artifacts is not None:
            artifacts.save_json("prototype_calibration.json", {
                "pull": float(cfg.block_mlp_template_init_pull), "n_windows": int(count),
                "layers": [dict(stats, layer=layer) for layer, stats in enumerate(calibration)]})
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
    # exit, so the real .grad is still live when it returns, and it keeps references to the gradients
    # an optimizer clears inside step() (status_out["grads"]). That costs no extra forward/backward,
    # and it reports the graph the run actually trains through instead of a probe's approximation of
    # it. The end-of-run warning subtracts this set, so it fires only on a parameter that DID get a
    # gradient and still never moved -- the case no existing check covers.
    motion_known_dead: Tuple[str, ...] = ()
    motion_dead_captured = False
    skipped_steps = 0                 # cumulative REJECTED optimizer updates (audit 2026-08-06 F5)
    consecutive_skips = 0
    # Finite-divergence detection (audit 2026-10-10 round 2, item 1): interval statistics and spike warnings
    # on host values train_step already transfers.
    divergence = _DivergenceMonitor()
    last_group_norms: Dict[str, float] = {}               # pre-clip group norms of the last logged step
    # The last completed step at which each token id occurred as an input and as a target (-1: not since the start
    # or resume), for the spike attribution's steps-since-last-occurrence (audit 2026-10-20 item 11). One device
    # index_fill per step and no host read; the extra target slot absorbs ignore-index targets.
    occurrence_steps: Dict[str, torch.Tensor] = {
        "token":  torch.full((int(cfg.vocab_size),), -1, dtype=torch.int64, device=device),
        "target": torch.full((int(cfg.vocab_size) + 1,), -1, dtype=torch.int64, device=device),
    }
    failed_step_skips = 0            # strictest-rung failures skipped under max_failed_step_skips (item 16)
    loss_spike_rejections = 0         # updates rejected under loss_spike_skip_nats (item 17)
    max_failed_skips: Optional[int] = getattr(cfg, "max_failed_step_skips", None)
    max_consecutive: Optional[int] = getattr(cfg, "max_consecutive_skips", None)
    loss_spike_margin: Optional[float] = getattr(cfg, "loss_spike_skip_nats", None)
    # A resumed run continues the monitor, the skip counters and the last logged group norms from the
    # checkpoint, so its spikes and divergence columns match the uninterrupted run (audit 2026-10-16 F77).
    # An older checkpoint carries none: the monitor restarts and spikes wait for DIVERGENCE_EMA_WARMUP steps.
    resumed_divergence = getattr(artifacts, "resumed_divergence_state", None) if start_step > 0 else None
    if resumed_divergence is not None and divergence.load_state_dict(resumed_divergence["monitor"]):
        skipped_steps = resumed_divergence["skipped_steps"]
        consecutive_skips = resumed_divergence["consecutive_skips"]
        last_group_norms.update(resumed_divergence["last_group_norms"])
        failed_step_skips = int(resumed_divergence.get("failed_step_skips", 0))
        loss_spike_rejections = int(resumed_divergence.get("loss_spike_rejections", 0))
    elif start_step > 0 and artifacts is not None:
        logger.info(" resume: the checkpoint carries no divergence-monitor state; spike detection restarts "
                    "and waits %d finite steps", DIVERGENCE_EMA_WARMUP)
    completed_steps = int(n_steps)
    reset_nonfinite_tangent_elements()   # per-run accounting for the SPD-retraction guard
    reset_public_spd_repair_elements()   # ... and its certificate repairs
    reset_mu_trust_fallback_elements()   # ... the mu-trust-region equivariance fallback
    reset_decode_logdet_fallback_elements()   # ... and the decode log-det non-PD fallback
    reset_source_recovery_elements()        # ... and projected float64 source-factor recoveries
    reset_gemm_fallback_slices()            # ... and pair-KL GEMM slices rescored by the direct tiles
    reset_frame_pair_fp64_escalations()     # ... and float64 frame-pair escalations
    reset_layernorm_trace_cap_elements()    # ... and LayerNorm pushforward sphere-bound caps
    reset_covariance_ceiling_elements()     # ... and moment-matched covariance ceiling clamps
    reset_covariance_escalation_elements()  # ... and moment-matched covariance fp64 escalations
    run_total_base = _resumed_run_totals(getattr(artifacts, "resumed_numerics_counters", None),
                                         start_step, logger)
    # steps replayed at a stricter matmul policy (whole run), and its value at the last log step
    precision_replays = precision_replays_logged = run_total_base["matmul_precision_replays"]
    dead_router_warned: Set[Tuple[str, int, int]] = set()    # (kind, layer, head) reported by _warn_dead_router

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
    if offset_windows is not None:
        saved_offset = resume_data_state.get("window_offset") if resume_data_state else None
        if saved_offset is not None and offset_generator is None:
            # The global RNG was restored after this epoch's draw: reinstate the saved origin.
            offset_windows.window_offset = int(saved_offset)
        else:
            drawn_offset = offset_windows.draw_window_offset(offset_generator)
            if saved_offset is not None and saved_offset != drawn_offset:
                raise RuntimeError(f"checkpoint data_state window_offset {saved_offset} is not the replayed "
                                   f"origin {drawn_offset}")
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
    resumed_wall_clock = getattr(artifacts, "resumed_wall_clock_s", None) if start_step > 0 else None
    timer = (TrainingTimer(device, wall_clock_offset=resumed_wall_clock or 0.0)
             if timing_enabled else None)
    wall_clock_origin = time.perf_counter()

    def _elapsed_wall_clock() -> float:
        r"""Cumulative training wall clock (the resumed offset included) for a checkpoint: the timer's when it
        runs, so metrics.csv ``wall_clock_s`` and the checkpoint agree, else a clock of this call
        (audit 2026-10-16 F81: a run without log/eval rows has no timer, and its checkpoints carried None)."""
        if timer is not None:
            return timer.elapsed_wall_clock_s()
        return (resumed_wall_clock or 0.0) + time.perf_counter() - wall_clock_origin

    last_val: Dict[str, Optional[float]] = {}        # most recent validation, carried into each CSV row
    # VFE-only diagnostics (held-out probes, residual/attention capture, frame-clamp warning) are
    # skipped for another model family; their columns keep the VFE schema and stay blank.
    vfe_model = is_vfe_model(model)
    val_diag_keys = _VAL_DIAG_KEYS + (
        _VAL_RELATIONAL_DIAG_KEYS if vfe_model else ())
    if vfe_model and getattr(cfg, "content_link_edges", "off") != "off":
        val_diag_keys = val_diag_keys + _VAL_EDGE_DIAG_KEYS
        if getattr(cfg, "content_link_latent", False):
            val_diag_keys = val_diag_keys + _VAL_LATENT_DIAG_KEYS
    if vfe_model and getattr(cfg, "predictive_readout", "kernel") == "minimal_agent":
        val_diag_keys = val_diag_keys + _VAL_MINIMAL_AGENT_DIAG_KEYS
        if getattr(cfg, "diagnose_agent_elbo", False):
            val_diag_keys = val_diag_keys + _VAL_MINIMAL_AGENT_ELBO_KEYS
    if vfe_model and getattr(cfg, "diagnose_canonical_free_energy", False):
        val_diag_keys = val_diag_keys + tuple(f"val_{key}" for key in CANONICAL_DIAGNOSTIC_KEYS)
    last_val_diag: Dict[str, float] = {k: float("nan") for k in val_diag_keys}
    diag_keys: Optional[Tuple[str, ...]] = None      # model.diagnostics keys of the first SUCCESSFUL call
    # metrics.csv columns are fixed by the first row written, so a row that cannot carry the full column set
    # (failed diagnostics before diag_keys is known, or a forced spike row) waits here until the first
    # complete row fixes the schema, and is then written NaN-filled to it (audit 2026-10-10 round 2, item 10).
    metrics_fields: Optional[List[str]] = (list(artifacts._fieldnames)
                                           if getattr(artifacts, "_fieldnames", None) else None)
    pending_rows: List[Dict[str, Any]] = []
    # Training-time diagnostic state (audit 2026-09-28). Built only for a persisted run; the silent
    # path and the console-only path allocate nothing here.
    diagnostics_failures: Dict[str, int] = {}
    fixed_train_sample: Optional[FixedTrainSample] = None
    diagnostic_windows: Optional[torch.Tensor] = None
    last_train_sample: Dict[str, float] = {column: float("nan") for column in _EVAL_CADENCE_COLUMNS}
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
    # Crash and recovery state (audit 2026-10-17 items 4 and 16). The pre-hook marks that the optimizer update of the
    # current step has started; until it fires the in-memory state is exactly the state before that step.
    update_flag: Dict[str, bool] = {"fired": False}
    update_hook = optimizer.register_step_pre_hook(lambda *_args: update_flag.__setitem__("fired", True))
    crash_state: Dict[str, Any] = {"phase": "setup", "step": None, "rng": None, "eval_rng": None,
                                   "tokens": None, "targets": None, "replays": [], "replays_flushed": True}
    recent_losses: Deque[float] = collections.deque(maxlen=max(1, int(log_interval or 100)))
    warning_sink: Optional[WarningSink] = None
    if artifacts is not None and getattr(artifacts, "run_dir", None) is not None:
        def warning_sink(record: Dict[str, Any]) -> None:
            try:
                _append_run_warning(artifacts.run_dir, {"step": crash_state["step"] + 1
                                                        if crash_state["step"] is not None else None, **record})
            except Exception as exc:                          # noqa: BLE001 - a log line, never fatal
                _record_diagnostic_failure(diagnostics_failures, "warnings_log", exc, logger)
    floor_reference: Optional[float] = None                   # _covariance_floor_reference, read at log steps
    last_table_step_norm: Optional[float] = None              # table_fisher_step_norm_mean of the previous row
    checkpoint_ms_pending: List[float] = []                   # checkpoint saves since the previous row
    peak_memory = _PeakMemory(device)
    device_telemetry = _DeviceTelemetry(device)               # allocator and clock columns (audit 2026-10-20 item 20)

    def _data_state_for(batches: int, identity: Optional[Mapping[str, object]]) -> Optional[DataState]:
        r"""The checkpoint data cursor at ``batches`` consumed batches of the current epoch, or None."""
        if identity is None:
            return None
        return {
            "epoch_start_generator_state": epoch_start_generator_state,
            "batches_consumed":            int(batches),
            "epoch":                       epoch,
            "data_identity":               identity,
            **({"window_offset": offset_windows.window_offset} if offset_windows is not None else {}),
        }

    def _publish_checkpoint_state(interval_ends: bool) -> None:
        r"""The host-side state a checkpoint carries beside the weights (numerics counts, wall clock, monitor)."""
        try:
            artifacts.numerics_counters_state = {
                **(_frame_projection_state(frame_projection_counter, frame_projection_base,
                                           frame_projection_since) or {}),
                **_run_total_counters(run_total_base, precision_replays,
                                      phi_rejected_rows=int(getattr(optimizer, "phi_rejected_rows", 0)))}
        except Exception as exc:
            _record_diagnostic_failure(diagnostics_failures, "numerics_checkpoint_state", exc, logger)
            artifacts.numerics_counters_state = None
        artifacts.wall_clock_state = _elapsed_wall_clock()
        artifacts.divergence_state = _divergence_checkpoint_record(
            divergence, interval_ends=interval_ends, skipped_steps=skipped_steps,
            consecutive_skips=consecutive_skips, last_group_norms=last_group_norms,
            failed_step_skips=failed_step_skips, loss_spike_rejections=loss_spike_rejections)

    def _emergency_checkpoint(record: Dict[str, Any]) -> None:
        r"""Write a resumable checkpoint of the crashed loop when its state is a step boundary (item 4)."""
        phase, index = crash_state["phase"], crash_state["step"]
        if not getattr(artifacts, "save_model_artifacts", False):
            record["checkpoint_reason"] = "model-artifact persistence is disabled for this run"
            return
        if index is None:
            record["checkpoint_reason"] = "the failure preceded the first step"
            return
        if phase in ("pre_step", "step") and not update_flag["fired"]:
            saved_step, batches, rng = int(index), batches_consumed - 1, crash_state["rng"]
        elif phase == "post_step":
            saved_step, batches, rng = int(index) + 1, batches_consumed, None
            if crash_state["eval_rng"] is not None:
                rng = {"cpu": crash_state["eval_rng"], "cuda": None}   # the stream the eval block hands back
        else:
            record["checkpoint_reason"] = ("the optimizer update of the failing step had started and the step did "
                                           "not return, so the in-memory state is not a step boundary")
            return
        if batches < 0:
            record["checkpoint_reason"] = "no batch of the current epoch was consumed before the failure"
            return
        identity = loader_data_identity
        if identity is None:
            if not cursor_supported:
                record["checkpoint_reason"] = ("the training loader has no exact data cursor (a DataLoader over a "
                                               "token-window dataset is required)")
                return
            if shuffled_loader and not (isinstance(loader_generator, torch.Generator)
                                        and getattr(loader_sampler, "generator", None) is loader_generator):
                record["checkpoint_reason"] = "the shuffled loader exposes no shared torch.Generator"
                return
            identity = _loader_data_identity(loader, cfg.vocab_size)
        if rng is not None:
            torch.set_rng_state(rng["cpu"])
            if rng.get("cuda") is not None:
                torch.cuda.set_rng_state_all(rng["cuda"])
        if phase in ("pre_step", "step"):
            optimizer.zero_grad(set_to_none=True)
        _publish_checkpoint_state(interval_ends=False)
        path = artifacts.save_checkpoint(saved_step, model, optimizer, cfg, scaler=scaler,
                                         data_state=_data_state_for(batches, identity))
        record["checkpoint"] = str(path)
        record["checkpoint_step"] = saved_step

    def _on_crash(exc: BaseException) -> None:
        r"""Write the crash record, the failing batch, pending rows and replay records and the emergency checkpoint.

        The record and the stop line come first (audit 2026-10-20 item 12): a preliminary ``crash.json`` with the
        exception, its traceback, the step and the recent losses is written and the error is logged before any
        slow stage, so an interrupt during the multi-gigabyte emergency checkpoint still leaves both. Each later
        stage is guarded by ``except BaseException``; an interrupt (``KeyboardInterrupt``, ``SystemExit``) inside a
        stage is listed in the record and skips the remaining stages. The record is rewritten atomically after the
        stages with ``record_complete`` True. A failure inside the handler never replaces the original exception,
        which the caller re-raises.
        """
        if artifacts is None or getattr(artifacts, "run_dir", None) is None:
            return
        index = crash_state["step"]
        record: Dict[str, Any] = {
            "step": None if index is None else int(index) + 1,       # 1-based number of the failing step
            "steps_completed_before": None if index is None else int(index),
            "phase": crash_state["phase"],
            "optimizer_update_started": bool(update_flag["fired"]),
            "epoch": int(epoch),
            "batches_consumed": int(batches_consumed),
            "exception": {
                "type": type(exc).__name__,
                "message": str(exc),
                "notes": [str(note) for note in getattr(exc, "__notes__", ())],
                "traceback": traceback.format_exception(type(exc), exc, exc.__traceback__),
            },
            "precision_replays": [dict(item) for item in crash_state["replays"]],
            "recent_losses": [float(value) for value in recent_losses],
            "failed_batch": None, "checkpoint": None, "checkpoint_step": None, "checkpoint_reason": None,
            "handler_errors": [],
            "record_complete": False,                             # True once every stage below has returned
            "interrupted_stage": None,                            # the stage an interrupt stopped, if any
        }
        diagnostics_dir = Path(artifacts.run_dir) / "diagnostics"

        def _write_record() -> None:
            try:
                diagnostics_dir.mkdir(parents=True, exist_ok=True)
                _write_crash_record(diagnostics_dir, record)
            except Exception as handler_exc:                      # noqa: BLE001 - the original exception wins
                logger.error("crash record could not be written (%s)", handler_exc)

        _write_record()                                           # preliminary record before any slow stage
        logger.error("training stopped at step %s (%s: %s); crash record %s; writing the failing batch and the "
                     "emergency checkpoint", record["step"], type(exc).__name__, exc,
                     diagnostics_dir / CRASH_RECORD_NAME)

        def _save_batch() -> None:
            if crash_state["tokens"] is not None:
                batch_path = diagnostics_dir / f"crash_batch_step_{record['step']}.pt"
                torch.save({"step": record["step"], "tokens": crash_state["tokens"],
                            "targets": crash_state["targets"]}, batch_path)
                record["failed_batch"] = str(batch_path)

        def _flush_replays() -> None:
            if crash_state["replays"] and not crash_state["replays_flushed"]:
                _append_replay_records(artifacts.run_dir, crash_state["replays"])
                crash_state["replays_flushed"] = True

        def _flush_pending() -> None:
            if pending_rows:
                union = (metrics_fields if metrics_fields is not None
                         else list(dict.fromkeys(key for pending in pending_rows for key in pending)))
                for pending in pending_rows:
                    artifacts.log_metrics(_conform_row(pending, union))
                pending_rows.clear()

        stages = (("failed batch", _save_batch), ("replay records", _flush_replays), ("pending rows", _flush_pending),
                  ("background jobs", artifacts.wait_background), ("checkpoint", lambda: _emergency_checkpoint(record)))
        for name, stage in stages:
            try:
                stage()
            except BaseException as handler_exc:                  # noqa: BLE001 - the original exception wins
                if name == "checkpoint":
                    record["checkpoint"] = None
                record["handler_errors"].append(f"{name}: {type(handler_exc).__name__}: {handler_exc}")
                if not isinstance(handler_exc, Exception):        # an interrupt: stop here, keep the record
                    record["interrupted_stage"] = name
                    break
            _write_record()                                       # each stage's result reaches the record
        else:
            record["record_complete"] = True
        _write_record()
        logger.error("crash handling at step %s finished; checkpoint %s%s", record["step"],
                     record["checkpoint"] or f"none ({record['checkpoint_reason']})",
                     "" if record["interrupted_stage"] is None
                     else f"; interrupted during the {record['interrupted_stage']} stage")

    def _save_failed_batch(step_number: int, exc: BaseException) -> Optional[str]:
        r"""Save the batch of a skipped failed step to ``diagnostics/failed_steps/step_<S>.pt`` (item 16)."""
        if artifacts is None or getattr(artifacts, "run_dir", None) is None:
            return None
        directory = Path(artifacts.run_dir) / "diagnostics" / "failed_steps"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"step_{int(step_number)}.pt"
        torch.save({"step": int(step_number), "tokens": crash_state["tokens"], "targets": crash_state["targets"],
                    "exception": f"{type(exc).__name__}: {exc}",
                    "notes": [str(note) for note in getattr(exc, "__notes__", ())]}, path)
        return str(path)

    try:
        for step in _step_indices():
            try:
                tokens, targets = next(it)
            except StopIteration:
                epoch += 1
                batches_consumed = 0
                epoch_start_generator_state = (
                    loader_generator.get_state().clone()
                    if shuffled_loader and isinstance(loader_generator, torch.Generator) else None)
                if offset_windows is not None:
                    offset_windows.draw_window_offset(offset_generator)
                it = iter(loader)
                try:
                    tokens, targets = next(it)
                except StopIteration:
                    raise RuntimeError(
                        f"the training loader yielded no batch after it was re-iterated for step {step}: "
                        "train() cycles its loader, so it needs a re-iterable loader (a DataLoader or a list "
                        "of batches), not a one-shot iterator, and at least one batch per epoch") from None
            batches_consumed += 1
            # Crash state of this step (audit 2026-10-17 item 4): the loader's host batch, and the RNG streams after the
            # batch was drawn, which an emergency checkpoint of the pre-step state restores.
            crash_state.update(step=step, phase="pre_step", tokens=tokens, targets=targets, eval_rng=None,
                               replays=[], replays_flushed=True,
                               rng={"cpu": torch.get_rng_state(),
                                    "cuda": ([state.clone() for state in torch.cuda.get_rng_state_all()]
                                             if torch.cuda.is_available() and torch.cuda.is_initialized()
                                             else None)})
            update_flag["fired"] = False
            router_head_tables: Optional[Mapping[str, Any]] = None
            belief_layer_extremes: Optional[Mapping[str, Sequence[float]]] = None
            tokens_seen += int(tokens.numel())                   # input-token presentations (metadata only)
            tokens = tokens.to(device, non_blocking=True)
            targets = _transfer_targets(targets, model, device)
            do_log  = bool(log_interval) and (step + 1) % log_interval == 0
            do_eval = bool(eval_interval) and val_loader is not None and (step + 1) % eval_interval == 0
            do_csv  = artifacts is not None and (do_log or do_eval)
            # Capture pre-clip gradient health only on a step that will log or persist. A silent step
            # performs no exact-count host materialization, extra unscale_, or grad-norm pass; its
            # device-side partition/zero-count predicates ride the existing scalar barrier.
            step_metrics: Optional[Dict[str, float]] = {} if (do_log or do_csv) else None
            phase_ms: Dict[str, float] = {}                       # LOG_TIMING_COLUMNS of this step
            if hasattr(optimizer, "_collect_table_diag"):        # m_table_update_mode='fisher' step health
                optimizer._collect_table_diag = bool(do_log or do_csv or do_eval)
                if optimizer._collect_table_diag:
                    # As for _gauge_diag below: a skipped step never reaches the table step, so the previous
                    # row's values must not be reported for it (audit 2026-10-10 round 2, item 37).
                    optimizer._table_diag = {}
            if hasattr(optimizer, "_collect_gauge_diag"):        # D1/EXP-8: sparse log/eval diagnostics
                optimizer._collect_gauge_diag = bool(do_log or do_csv or do_eval)
                if optimizer._collect_gauge_diag:
                    # A nonfinite/otherwise skipped optimizer step must not leave the prior log step's
                    # health values visible as if they were collected for this attempt.
                    optimizer._gauge_diag = {}
            if do_log or do_csv:                                 # off the hot path (audit 2026-07-05 m8)
                # m7: diagnostics on the PRE-step weights so the logged F-decomposition shares the pre-step
                # provenance of train_loss/train_ce (train_ce is captured pre-step inside train_step). Under
                # no_grad, so no weight change,
                # leaving the train_step RNG stream and weights byte-identical.
                peak_memory.begin_other()
                _t0 = _synced_ms(device)
                # Prototype-template statistics of the same diagnostic forward on the training batch
                # (diagnostics/prototype_templates_train.csv); the eval-cadence file alone missed a collapse
                # that finished inside the first eval interval (docs/experiments/2026-10-09).
                train_features: Optional[Dict[str, Any]] = {} if do_csv else None
                wrapped_templates = (_wrap_prototype_blocks(model, train_features)
                                     if train_features is not None else [])
                diag_failed = False
                prototype_sums = (train_features or {}).get("prototype")
                try:
                    # The precision ladder of evaluate (audit 2026-10-10 round 2, item 10); a replay restarts the
                    # prototype sums so a failed attempt is not counted.
                    if hasattr(model, "last_router_head_tables"):
                        model.last_router_head_tables = None
                    if hasattr(model, "last_belief_layer_extremes"):
                        model.last_belief_layer_extremes = None
                    d = _diagnostics_with_precision_retry(
                        model, tokens, before_attempt=(prototype_sums.clear if prototype_sums is not None else None))
                    router_head_tables = getattr(model, "last_router_head_tables", None)
                    belief_layer_extremes = getattr(model, "last_belief_layer_extremes", None)
                except Exception as exc:                         # optional probes cannot stop training
                    _record_diagnostic_failure(diagnostics_failures, "training_diagnostics", exc, logger)
                    d = {key: float("nan") for key in (diag_keys or _TRAIN_DIAGNOSTIC_FALLBACK_KEYS)}
                    diag_failed = True
                    train_features = None
                    router_head_tables = None
                    belief_layer_extremes = None
                finally:
                    for mlp in wrapped_templates:
                        del mlp.forward_moments                  # back to the class method
                if do_csv and not diag_failed and diag_keys is None:
                    # log_metrics refuses a changed field set, so the short fallback above aborted the run at
                    # the next metrics.csv write (audit 2026-10-10 #16). Only a successful call fixes the
                    # diagnostics key set; a failed row carries every key of it as NaN (blank cells), and a row
                    # that fails before any success waits in pending_rows for the full column set.
                    diag_keys = tuple(d)
                if train_features:
                    try:
                        _, template_rows = _feature_rows(model, train_features, step + 1)
                        if template_rows:
                            artifacts.append_diagnostic_rows(
                                "prototype_templates_train.csv", PROTOTYPE_TEMPLATE_FIELDS, template_rows)
                    except Exception as exc:
                        _record_diagnostic_failure(diagnostics_failures, "prototype_templates_train", exc, logger)
                phase_ms["log_diagnostics_ms"] = _synced_ms(device) - _t0
                peak_memory.end_other()
            # generalization_gap's train side under the validation protocol when that protocol is the
            # sliding window (eval_stride < max_seq_len); pre-step, like train_ce. None keeps train_ce.
            gap_train_ce: Optional[float] = None
            if (do_eval and do_csv and cfg.eval_stride is not None
                    and cfg.eval_stride < cfg.max_seq_len):
                peak_memory.begin_other()
                _t0 = _synced_ms(device)
                gap_train_ce = _eval_protocol_train_ce(model, tokens, targets, cfg.eval_stride)
                phase_ms["eval_gap_train_ms"] = _synced_ms(device) - _t0
                peak_memory.end_other()
            # Named-group update ratios (audit 2026-09-28): host copies of the pre-step parameters on a
            # persisted log/eval row only, outside the timed train step. Reading parameters does not
            # change the step.
            update_before: Optional[Dict[str, torch.Tensor]] = None
            update_rates: Optional[Dict[str, float]] = None
            group_stats: Dict[str, float] = {}
            if do_csv:
                _t0 = _synced_ms(device)
                try:
                    update_before = _parameter_snapshot(model)
                    update_rates = _parameter_learning_rates(model, optimizer)
                except Exception as exc:
                    _record_diagnostic_failure(diagnostics_failures, "parameter_snapshot", exc, logger)
                phase_ms["log_group_stats_ms"] = _synced_ms(device) - _t0
            step_status: Dict[str, Any] = {}
            if timer is not None:
                timer.start_step()
            replay_records: List[Dict[str, object]] = []
            crash_state.update(phase="step", replays=replay_records, replays_flushed=False)
            # The rates this step uses, read before train_step advances the scheduler, so a logged lr_* is the
            # step's own rate on an accepted and on a skipped step alike (audit 2026-10-12 #11).
            step_raw_lrs = scheduler.get_last_lr()
            step_table_fisher_lrs = _table_fisher_learning_rates(optimizer)
            # loss_spike_skip_nats (audit 2026-10-17 item 17): the threshold is the monitor's loss EMA before this step
            # plus the margin, a host float, so the step reads it against the loss it already transfers.
            loss_threshold = (divergence.loss_skip_threshold(loss_spike_margin)
                              if loss_spike_margin is not None else None)
            failed_skip = False
            try:
                step_loss, replays = _train_step_with_precision_retry(
                    model, optimizer, scheduler, tokens, targets, logger=logger, step=step,
                    quiet=precision_replays > 0, replay_log=replay_records, grad_clip=grad_clip,
                    grad_clip_scope=getattr(cfg, "grad_clip_scope", "role"),
                    grad_accum_steps=cfg.grad_accum_steps,
                    scaler=scaler, metrics_out=step_metrics, status_out=step_status,
                    **({"frame_projection_out": frame_projection_counter}
                       if frame_projection_counter is not None else {}),
                    **({"loss_skip_threshold": loss_threshold} if loss_threshold is not None else {}))
            except Exception as exc:
                # max_failed_step_skips (audit 2026-10-17 item 16): a failure that persisted to the strictest rung
                # before any optimizer update is skipped like a rejected update, up to the configured count.
                if (max_failed_skips is None or isinstance(exc, torch.OutOfMemoryError) or update_flag["fired"]
                        or failed_step_skips >= max_failed_skips):
                    raise
                failed_step_skips += 1
                failed_skip = True
                optimizer.zero_grad(set_to_none=True)
                _reset_scaler_step_state(scaler, optimizer)
                for out in (step_metrics, step_status):
                    if out is not None:
                        out.clear()
                for name in ("_table_diag", "_gauge_diag"):
                    if hasattr(optimizer, name):
                        setattr(optimizer, name, {})
                step_status.update(did_step=False, nonfinite_skip=False, failed_step_skip=True, ce=float("nan"),
                                   role_grad_norms={}, clip_coefs={}, clip_scales={})
                step_loss, replays = float("nan"), len(replay_records)
                if step_metrics is not None:
                    step_metrics.update(step_skipped=1.0, loss_finite=0.0)
                try:
                    batch_path = _save_failed_batch(step + 1, exc)
                except Exception as save_exc:                    # noqa: BLE001 - the skip itself proceeds
                    batch_path = None
                    _record_diagnostic_failure(diagnostics_failures, "failed_step_batch", save_exc, logger)
                message = (" step %d: the step failed at the strictest precision before any update (%s: %s); skipped "
                           "(%d of max_failed_step_skips=%d), batch saved to %s" % (
                               step + 1, type(exc).__name__, exc, failed_step_skips, max_failed_skips, batch_path))
                _emit_warning(logger, warning_sink, "failed_step_skip", message, exception=type(exc).__name__,
                              detail=str(exc), notes=[str(note) for note in getattr(exc, "__notes__", ())],
                              batch=batch_path, count=failed_step_skips)
            crash_state["phase"] = "post_step"
            losses.append(step_loss)
            recent_losses.append(float(step_loss))
            precision_replays += int(replays > 0)
            if step_status.get("loss_spike_skip"):
                loss_spike_rejections += 1
                if loss_spike_rejections == 1:
                    logger.warning("step %d: update rejected by loss_spike_skip_nats=%g (loss %.4f above the "
                                   "threshold %.4f); further rejections are counted in loss_spike_rejected_updates",
                                   step + 1, loss_spike_margin, step_loss, loss_threshold)
            if replay_records and artifacts is not None:
                try:
                    _append_replay_records(artifacts.run_dir, replay_records)
                except Exception as exc:
                    _record_diagnostic_failure(diagnostics_failures, "matmul_precision_replay_log", exc, logger)
            crash_state["replays_flushed"] = True
            if timer is not None:
                timer.finish_step(n_tokens=tokens.numel())
            if update_before is not None:
                _t0 = _synced_ms(device)
                try:
                    group_stats = _group_update_statistics(model, update_before, learning_rates=update_rates)
                except Exception as exc:
                    _record_diagnostic_failure(diagnostics_failures, "group_update_statistics", exc, logger)
                update_before = None                              # release the host copy immediately
                phase_ms["log_group_stats_ms"] = (phase_ms.get("log_group_stats_ms", 0.0)
                                                  + _synced_ms(device) - _t0)
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
                if max_consecutive is not None and consecutive_skips >= max_consecutive:
                    # Audit 2026-10-17 item 16: abort through the crash path, which records the failure and writes a
                    # resumable checkpoint of this (post-step) state.
                    raise ConsecutiveSkipAbort(
                        f"{consecutive_skips} consecutive rejected updates at step {step + 1} reached "
                        f"max_consecutive_skips={max_consecutive} ({skipped_steps} rejected of {step + 1} so far)")
            # Divergence monitor (audit 2026-10-10 round 2, item 1), on host values only. A spike is logged and
            # forces a metrics row at this step.
            spike = divergence.observe(step + 1, step_loss, step_status)
            spike_groups: Dict[str, float] = {}
            if divergence.event is not None:
                # Every event is attributed and recorded in divergence_spikes.jsonl; only a reported one (outside the
                # cooldown, or 10x the last reported severity) logs to the console and forces a row (item 15).
                reported = spike is not None
                event_record = {"step": step + 1, "spike": divergence.event, "reported": reported,
                                "severity": divergence.event_severity,
                                "spiking_roles": list(divergence.spike_roles), "loss": step_loss}
                if reported:
                    logger.warning("step %d: divergence spike: %s", step + 1, spike)
                if step_status["did_step"]:                      # a rejected update dropped its gradients
                    try:
                        report = _spike_attribution(
                            model, optimizer, step_status, tokens,
                            roles=divergence.spike_roles or DIVERGENCE_ROLES, loss_spiked=divergence.spike_loss,
                            replayed=replays > 0, last_group_norms=last_group_norms, targets=targets,
                            step=step + 1, last_seen=occurrence_steps)
                        report["loss_over_3_log_v"] = bool(math.isfinite(step_loss)
                                                           and step_loss > 3.0 * math.log(cfg.vocab_size))
                        if reported and DIVERGENCE_SPIKE_FORWARD_PROBE:
                            try:
                                report["forward_state"] = _spike_forward_state(model, tokens, targets)
                            except Exception as exc:                 # noqa: BLE001 - a diagnostic
                                report["forward_state"] = {"error": f"{type(exc).__name__}: {exc}"}
                        if reported:
                            spike_groups = report["group_norms"]
                            logger.warning("step %d: spike attribution: %s", step + 1, _spike_summary(report))
                        if artifacts is not None:
                            _append_spike_record(artifacts.run_dir, {**event_record, **report})
                    except Exception as exc:
                        _record_diagnostic_failure(diagnostics_failures, "divergence_spike_attribution", exc, logger)
                else:
                    if reported:
                        logger.warning("step %d: spike attribution unavailable: the update was rejected and its "
                                       "gradients dropped", step + 1)
                    if artifacts is not None:
                        try:
                            _append_spike_record(artifacts.run_dir, {
                                **event_record, "attribution": "unavailable: the update was rejected and its "
                                                               "gradients dropped"})
                        except Exception as exc:
                            _record_diagnostic_failure(diagnostics_failures, "divergence_spike_attribution", exc,
                                                       logger)
            # This step's ids become the previous occurrence for later spikes (after the attribution read them).
            occurrence_steps["token"].index_fill_(0, tokens.reshape(-1), step + 1)
            occurrence_steps["target"].index_fill_(0, torch.where(
                targets.reshape(-1) >= 0, targets.reshape(-1), int(cfg.vocab_size)), step + 1)
            if step_metrics is not None:                         # the reference for "x its last logged norm"
                last_group_norms.update({column[len("grad_norm_group_"):]: float(step_metrics[column])
                                         for column in _GRAD_GROUP_COLUMNS if column in step_metrics})
            forced_row = (not do_csv and artifacts is not None and timer is not None and spike is not None)
            if forced_row:
                # Nothing was collected before the step: the pre-step diagnostics are blank on this row.
                d = {key: float("nan") for key in (diag_keys or _TRAIN_DIAGNOSTIC_FALLBACK_KEYS)}
            if motion_snapshot is not None and not motion_dead_captured and step_status["did_step"]:
                motion_known_dead = _known_dead_parameters(model, step_status)
                motion_dead_captured = True
            if do_log or do_csv or forced_row:                   # diagnostics (off graph), ONCE
                # train_ce is the PRE-step CE captured inside train_step (matches train_loss); the old
                # post-step re-forward made train_ce one optimizer step ahead of train_loss (audit r2 id5).
                ce = (step_metrics.get("train_ce", float("nan")) if step_metrics is not None
                      else float(step_status.get("ce", float("nan"))))
                # decode_second_order: the second-order objective the step trained on (train_ce stays plug-in).
                second_order_ce = (step_metrics.get("train_second_order_ce", float("nan"))
                                   if step_metrics is not None else float("nan"))
                # d (the F-decomposition) is computed PRE-step above so it shares train_ce's provenance (m7).
                assert timer is not None
                train_timing = timer.sample_train_window()
                # Allocator and clock telemetry before the peak reset below (audit 2026-10-20 item 20).
                telemetry = device_telemetry.sample()
                # Overall peak since the previous row (as before) and the training-only peak (audit 2026-10-17 item 49).
                peak_mem_mb, peak_mem_train_mb = peak_memory.sample()

            if do_log:
                logger.info(
                    "Step %d/%d | Loss: %.4f | CE: %.4f | H(b): %.3f | train it/s: %.2f | \n\n         Train PPL: %.1f \n",
                    step + 1, n_steps, losses[-1], ce, d["attn_entropy"],
                    train_timing.train_steps_per_s, _perplexity_from_ce(ce),
                )
                if getattr(cfg, "decode_second_order", False):
                    logger.info("    Second-order CE (trained on): %.4f | plug-in CE (as validation): %.4f",
                                second_order_ce, ce)
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
                crash_state["eval_rng"] = eval_cpu_rng
                peak_memory.begin_other()
                if (_final_validation_out is not None and terminal_callback is None
                        and step + 1 == n_steps):
                    terminal_validation_signature = _terminal_validation_signature(model, val_loader)
                _t0 = _synced_ms(device)
                try:
                    m = evaluate(model, val_loader, tokens_per_char=tokens_per_char, device=device)
                except Exception as exc:
                    # Under max_failed_step_skips a failed evaluation reports NaN validation and updates no best
                    # model (audit 2026-10-17 item 16); otherwise it propagates to the crash path as before.
                    if max_failed_skips is None or isinstance(exc, torch.OutOfMemoryError):
                        raise
                    terminal_validation_signature = None
                    _record_diagnostic_failure(diagnostics_failures, "evaluation", exc, logger)
                    _emit_warning(logger, warning_sink, "failed_evaluation",
                                  " step %d: evaluation failed (%s: %s); validation is NaN for this eval and the best "
                                  "model is not updated" % (step + 1, type(exc).__name__, exc),
                                  exception=type(exc).__name__, detail=str(exc))
                    m = {"ce": float("nan"), "ppl": float("nan"), "bits_per_token": float("nan"), "bpc": None,
                         "expected_targets": float("nan"), "scored_targets": float("nan"),
                         "excluded_targets": float("nan")}
                phase_ms["eval_validation_ms"] = _synced_ms(device) - _t0
                if terminal_validation_signature is not None:
                    terminal_validation_metrics = dict(m)
                if terminal_callback is not None and artifacts is not None:      # do_csv: this step writes its row
                    last_eval_record = (step + 1, dict(m), _tensor_versions(model))
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
                    _t0 = _synced_ms(device)
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
                    phase_ms["eval_sample_ms"] = _synced_ms(device) - _t0
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
                    _t0 = _synced_ms(device)
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
                    phase_ms["eval_val_diagnostics_ms"] = _synced_ms(device) - _t0
                    _t0 = _synced_ms(device)
                    try:
                        # Best-effort like the periodic checkpoint below (audit 2026-10-10 #17): a transient
                        # torch.save failure ("ios_base::badbit set") aborted the run. maybe_save_best
                        # advances best_val_ppl only after the file is replaced, so the record keeps
                        # matching best_model.pt and the next eval that beats it retries the save.
                        try:
                            # A NaN validation (failed evaluation) never selects; the save runs on a worker thread
                            # from a host copy of the weights (audit 2026-10-17 items 16 and 68).
                            if math.isfinite(float(m["ppl"])):
                                artifacts.maybe_save_best(step + 1, model, m["ppl"], background=True)
                        except Exception as exc:
                            logger.warning("       (best_model.pt save at step %d failed: %s); training "
                                           "continues, the next improving eval retries", step + 1, exc)
                            diagnostics_failures["best_model_save"] = (
                                diagnostics_failures.get("best_model_save", 0) + 1)
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
                        phase_ms["eval_best_and_maps_ms"] = _synced_ms(device) - _t0
                    # Fixed training-sample CE, residual/attention CSVs and the map snapshot (audit
                    # 2026-09-28). Each part is exception-isolated and counted; RNG is forked inside.
                    _t0 = _synced_ms(device)
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
                    # A held-out probe scored on the diagnostic windows of the pass above (audit 2026-10-10 #15).
                    last_val_diag["val_attn_collapsed_heads"] = last_train_sample.pop(
                        "val_attn_collapsed_heads", float("nan"))
                    phase_ms["eval_cadence_ms"] = _synced_ms(device) - _t0
                torch.set_rng_state(eval_cpu_rng)            # F30: hand the training stream back untouched
                crash_state["eval_rng"] = None
                peak_memory.end_other()

            # Periodic resumable checkpoint (opt-in; needs the artifacts dir and the optimizer state).
            if (artifacts is not None and cfg.checkpoint_interval
                    and (step + 1) % cfg.checkpoint_interval == 0):
                checkpoint_data_state: Optional[DataState] = ({
                    "epoch_start_generator_state": epoch_start_generator_state,
                    "batches_consumed":            batches_consumed,
                    "epoch":                       epoch,
                    "data_identity":               loader_data_identity,
                    **({"window_offset": offset_windows.window_offset} if offset_windows is not None else {}),
                } if loader_data_identity is not None else None)
                # Best-effort, like every other periodic side effect above (audit 2026-07-27). A
                # checkpoint is a convenience for RESUMING, not a correctness requirement of the step
                # loop, so a failed write must cost one checkpoint and not the run: a transient
                # torch.save stream failure ("ios_base::badbit set") at step 120000 of 180000
                # previously aborted 10 hours of training with a perfectly healthy model in memory.
                # Loud warning, then continue -- the next interval attempts a fresh save.
                # The cumulative frame-projection counts travel with the checkpoint (one host read at
                # checkpoint cadence), so a resumed run keeps counting from step 0.
                # The row of this step is written after the save and then ends the interval (do_log or do_csv).
                _publish_checkpoint_state(interval_ends=bool(do_log or do_csv))
                _t0 = time.perf_counter()
                try:
                    artifacts.save_checkpoint(step + 1, model, optimizer, cfg, scaler=scaler,
                                              data_state=checkpoint_data_state)
                    # checkpoint_keep_last (audit 2026-10-17 item 48): only after a successful save.
                    artifacts.prune_checkpoints(getattr(cfg, "checkpoint_keep_last", None))
                except Exception as exc:
                    logger.warning("       (checkpoint save at step %d failed: %s); training "
                                   "continues, next checkpoint retries", step + 1, exc)
                checkpoint_ms_pending.append((time.perf_counter() - _t0) * 1000.0)

            # Persistence is opt-in: with no artifacts object do_csv is False, so the silent/in-memory
            # path is unchanged. A metrics.csv row is written every LOG_INTERVAL (and every eval) -- the
            # dense per-step diagnostics off the graph. The EVAL-CADENCE columns (val_ce/ppl/bpc,
            # generalization_gap, and the held-out val_*/estep_*/pos_loss_* probes) carry a value ONLY on a
            # step where the eval above just ran (do_eval); on the denser log-interval rows in between they
            # are NaN, which log_metrics renders as a BLANK cell -- so each validation appears exactly once
            # per eval_interval, not carried forward to every log line. The
            # in-memory history keeps the NaN (figures already drop non-finite rows). The F-stack
            # diagnostics (NaN unless diagnose_canonical_free_energy) arrive as PER-TOKEN means over the
            # batch, commensurate with val_ce, a token-weighted mean (nats/token; audit-2026-06-05 Finding 2).
            if do_csv or forced_row:
                _row_t0 = _synced_ms(device)
                lrs = (_learning_rates_by_role(optimizer.param_groups, step_raw_lrs) if vfe_model
                       else _learning_rates_by_role(optimizer.param_groups, step_raw_lrs, required=("mu",)))
                # m_table_update_mode='fisher': lr_mu / lr_sigma report the realized table step size eta lr_t/lr_0,
                # not the AdamW rate that only schedules it (audit 2026-10-10 round 2, items 20 and 37).
                lrs.update(step_table_fisher_lrs)
                aux_lrs = _learning_rates_by_aux_role(optimizer.param_groups, step_raw_lrs)
                row = {
                    "step":              step + 1,
                    # Audit 2026-10-17 item 5: an eval row (which is also a log row when both fall on the step), a log
                    # row, or a row forced by a divergence spike; readers drop or weight the spike rows.
                    "row_kind":          "eval" if do_eval else ("log" if do_csv else "spike"),
                    "train_loss":        losses[-1],
                    "train_ce":          ce,                      # true CE (nats), off the graph
                    "train_ppl":         _perplexity_from_ce(ce),  # exact exp(CE), infinity on overflow
                    # decode_second_order: the delta-method objective trained on; train_ce/train_ppl stay the
                    # plug-in CE that validation scores (audit 2026-10-12 round 2, item 12).
                    **({"train_second_order_ce": second_order_ce} if getattr(cfg, "decode_second_order", False) else {}),
                    "lr_block_mlp":       aux_lrs["block_mlp"],
                    "lr_layernorm":      aux_lrs["layernorm"],
                    "lr_mu":             lrs["mu"],
                    "lr_sigma":          lrs["sigma"],
                    "lr_phi":            lrs["phi"],
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
                    "self_coupling":      d["self_coupling"],     # alpha-regularized F self-term alpha_i D + R(alpha_i)
                    "self_divergence":    d["self_divergence"],   # raw D(q_i||p_i) drift, unweighted by alpha_i and without R(alpha_i)
                    "belief_coupling":    d["belief_coupling"],
                    "attention_entropy":  d["attention_entropy"],
                    "inner_alignment_energy_total": d["total"],
                    # Weights emitted alongside the RAW per-term fields (audit 2026-07-25 F7): those
                    # fields are deliberately unweighted while "total" routes through the weighted
                    # evaluator, so a naive column sum missed the total by up to 98% relative at
                    # lambda_beta=0.5 with no column to reconcile it. Now each row is self-describing.
                    "lambda_beta":        float(getattr(cfg, "lambda_beta", float("nan"))),
                    "effective_rank":     d["effective_rank"],
                }
                row.update({key: value for key, value in d.items() if key.startswith(("qkv_", "relational_"))})
                _warn_dead_router(d, dead_router_warned, logger,
                                  reference=getattr(cfg, "routing_kl_max_reference", "absolute"),
                                  tied=bool(getattr(cfg, "tie_comparison_maps", False)),
                                  head_tables=router_head_tables, sink=warning_sink)
                _warn_covariance_collapse(d, dead_router_warned, logger, sink=warning_sink)
                if do_csv and vfe_model:
                    if floor_reference is None:
                        floor_reference = _covariance_floor_reference(model)
                    _warn_eigenvalue_collapse(d, dead_router_warned, logger,
                                              belief_layer=int(getattr(cfg, "n_layers", 1)) - 1,
                                              floor_reference=floor_reference, sink=warning_sink,
                                              layer_extremes=belief_layer_extremes)
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
                row["peak_mem_train_mb"] = peak_mem_train_mb   # the training phases alone (audit 2026-10-17 item 49)
                row.update(telemetry)                          # DEVICE_TELEMETRY_COLUMNS (audit 2026-10-20 item 20)
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
                    # -3, 3)); decode_log_scale is frozen, so this is constant over a run.
                    row["decode_tau_eff"] = float(model.prior_bank._tau_eff().detach())
                # Extended per-eval diagnostics (Tier-1/2): already-reduced gauge / geometry / numerical-
                # health scalars from diagnostics() -- NOT per-token sums, so logged RAW (no /n_tok).
                # Conditional mixer/connection keys appear only with their toggle; the
                # config is fixed per run so the CSV stays rectangular.
                # Every key is produced by the relational diagnostics (VFEModel._relational_diagnostics); the
                # legacy closure, pair-energy and transport-asymmetry keys that no route produces were removed
                # (audit 2026-10-12 round 2, item 13).
                for _dk in ("gauge_invariant_mean", "gauge_invariant_spread",
                            "phi_norm_mean", "phi_norm_std",
                            "belief_cond_median", "belief_cond_p95", "belief_cond_max",
                            "belief_lam_min", "belief_lam_max", "belief_cond_valid_frac",
                            "belief_lam_min_argmin_position", "belief_lam_min_from_position_1",
                            "belief_cond_median_layer_max",
                            # every layer beside the last-layer columns (audit 2026-10-20 item 14)
                            "belief_lam_min_layer_min", "belief_lam_min_argmin_layer", "belief_lam_max_layer_max",
                            "belief_cond_max_layer_max", "belief_eig_ratio_layer_min",
                            "belief_eig_ratio_argmin_layer",
                            "eff_rank_p5", "eff_rank_median", "eff_rank_p95",
                            "fisher_trace_mean", "fisher_trace_median",
                            "guard_sigma_floor_frac", "guard_sigma_ceil_frac",
                            "nonfinite_frac",
                            "attn_entropy_min", "attn_entropy_collapsed_heads",
                            "vertex_cond_max",
                            "gauge_head_aniso_mean", "gauge_head_logdet_spread",
                            "gauge_head_phase_abs_mean", "belief_circularity_mean",
                            "phi_matrix_norm_median", "phi_matrix_norm_p95", "phi_matrix_norm_p99",
                            "phi_matrix_norm_max",
                            "vertex_cond_median", "vertex_cond_p95", "vertex_cond_p99",
                            "pos_phi_matrix_norm_p95", "pos_phi_matrix_norm_p99", "pos_phi_matrix_norm_max",
                            *CANONICAL_DIAGNOSTIC_KEYS):        # diagnose_canonical_free_energy only
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
                elif forced_row:
                    # A forced row collected no step metrics; it reports the pre-clip role norms the step transferred.
                    _role_norms = step_status.get("role_grad_norms") or {}
                    for _name in DIVERGENCE_ROLES:
                        row[f"grad_norm_{_name}"] = float(_role_norms.get(_name, float("nan")))
                    row["grad_norm"] = (math.sqrt(sum(value * value for value in _role_norms.values()))
                                        if _role_norms else float("nan"))
                    row["loss_finite"] = float(math.isfinite(losses[-1]))
                    row["step_skipped"] = float(not step_status["did_step"])
                if precision_ladder:
                    row["matmul_precision_replays"] = float(precision_replays)
                    if precision_replays > precision_replays_logged:
                        logger.info("matmul precision: %d step(s) replayed at a stricter policy since the last log "
                                    "step (%d in total)", precision_replays - precision_replays_logged,
                                    precision_replays)
                        precision_replays_logged = precision_replays
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
                if hasattr(optimizer, "_table_diag"):             # fixed key set under m_table_update_mode='fisher'
                    _td = {} if forced_row else (optimizer._table_diag or {})   # a forced row collected none
                    for _pk in TABLE_FISHER_DIAGNOSTIC_KEYS:
                        row[_pk] = _td.get(_pk, float("nan"))
                    if not forced_row:
                        # Second collapse alarm: the realized table step norm falling 100x in one interval (item 14).
                        current = _finite_or_none(row.get("table_fisher_step_norm_mean"))
                        _warn_table_step_collapse(last_table_step_norm, current, logger, step=step + 1,
                                                  sink=warning_sink)
                        if current is not None:
                            last_table_step_norm = current
                if hasattr(optimizer, "_gauge_diag"):
                    _gd = {} if forced_row else (optimizer._gauge_diag or {})
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
                            # factor-scale halvings of the accepted step, on both retractions (audit 2026-10-16 F14)
                            "phi_group_backtracks_mean",
                            "phi_group_backtracks_max",
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
                # A forced spike row collected no step metrics: its group norms are the spike attribution's (pre-clip).
                group_values = {**{f"grad_norm_group_{group}": norm for group, norm in spike_groups.items()},
                                **(step_metrics or {}), **group_stats}
                for column in _GROUP_COLUMNS + _STEP_LR_COLUMNS:
                    row[column] = float(group_values.get(column, float("nan")))
                try:
                    row.update(_sigma_log_statistics(model))
                except Exception as exc:
                    _record_diagnostic_failure(diagnostics_failures, "sigma_log_statistics", exc, logger)
                    row.update({"sigma_log_mean": float("nan"), "sigma_log_sd": float("nan")})
                # The decode table's variance margin (decode_av_precision's gauge; NaN under the linear readout).
                try:
                    row.update(_decode_variance_statistics(model))
                except Exception as exc:
                    _record_diagnostic_failure(diagnostics_failures, "decode_variance_statistics", exc, logger)
                    row.update({column: float("nan") for column in DECODE_VARIANCE_COLUMNS})
                # S1 panel D cumulative counters on every row (one host read of the device counter).
                try:
                    row.update(_cumulative_counter_columns(
                        step + 1, optimizer, frame_projection_counter, frame_projection_base))
                except Exception as exc:
                    _record_diagnostic_failure(diagnostics_failures, "numerics_cumulative", exc, logger)
                    row.update({column: float("nan") for column in NUMERICS_CUMULATIVE_COLUMNS})
                # Cumulative projected source-factor recoveries since step 0 (diagnostic and evaluation
                # forwards included); a rise localizes fp64 promotions in the step time (audit 2026-10-09).
                row["source_recovery_elements"] = (run_total_base["source_recovery_elements"]
                                                   + source_recovery_elements())
                # Cumulative moment-matched covariance entries clamped by predictive_covariance_ceiling.
                row["covariance_ceiling_elements"] = (run_total_base["covariance_ceiling_elements"]
                                                      + covariance_ceiling_elements())
                # Cumulative tokens of moment-matched covariance updates recomputed in float64.
                row["covariance_escalation_elements"] = (run_total_base["covariance_escalation_elements"]
                                                         + covariance_escalation_elements())
                # Cumulative LayerNorm-pushforward tokens capped at the sphere bound (the delta method outside its
                # regime; zero under pushforward_moment, whose cap is read net of its floor; audit 2026-10-16 F3).
                row["layernorm_trace_cap_elements"] = (run_total_base["layernorm_trace_cap_elements"]
                                                       + layernorm_trace_cap_elements())
                # Cumulative (batch, head) slices whose pair KL the pair_kl_gemm_tol fallback moved to the direct
                # tiles; 0 while pair_kl_gemm_tol is None (audit 2026-10-16 F95).
                row["gemm_fallback_slices"] = run_total_base["gemm_fallback_slices"] + gemm_fallback_slices()
                row["tokens_seen"] = tokens_seen
                row["diagnostics_failures"] = int(sum(diagnostics_failures.values()))
                # Audit 2026-10-17 items 16, 17, 47 and 48: cumulative skipped failed steps and loss-spike rejections,
                # non-finite parameter entries after this step (one host read at row cadence), and the time spent in
                # checkpoint saves since the previous row (NaN when none ran).
                row["failed_step_skips"] = int(failed_step_skips)
                row["loss_spike_rejected_updates"] = int(loss_spike_rejections)
                try:
                    row["param_nonfinite_count"] = _parameter_nonfinite_count(model)
                except Exception as exc:
                    _record_diagnostic_failure(diagnostics_failures, "param_nonfinite_count", exc, logger)
                    row["param_nonfinite_count"] = float("nan")
                row["checkpoint_ms"] = (float(sum(checkpoint_ms_pending)) if checkpoint_ms_pending
                                        else float("nan"))
                checkpoint_ms_pending.clear()
                if (do_log and not do_eval and getattr(cfg, "block_variance_at_log", False)
                        and diagnostic_windows is not None and vfe_model):
                    # block_variance.csv at log cadence (audit 2026-10-17 item 35; the eval cadence writes its own).
                    peak_memory.begin_other()
                    try:
                        _append_block_variance(model, artifacts, diagnostic_windows, step + 1, device)
                    except Exception as exc:
                        _record_diagnostic_failure(diagnostics_failures, "block_variance", exc, logger)
                    peak_memory.end_other()
                pipeline_timing = timer.sample_pipeline_window()
                row["train_step_ms_mean"]      = train_timing.train_step_ms_mean
                row["train_step_tokens_per_s"] = train_timing.train_step_tokens_per_s
                row["pipeline_tokens_per_s"]   = pipeline_timing.pipeline_tokens_per_s
                row["tokens_per_s"]            = pipeline_timing.pipeline_tokens_per_s
                row["wall_clock_s"]            = pipeline_timing.wall_clock_s
                phase_ms["log_row_ms"] = _synced_ms(device) - _row_t0
                row.update({column: phase_ms.get(column, float("nan")) for column in LOG_TIMING_COLUMNS})
                row.update(divergence.columns(step_status, flagged=spike is not None))
                # The first complete row (regular, with the full diagnostics key set) fixes the metrics.csv schema;
                # earlier incomplete rows are then written NaN-filled to it, in step order, and later incomplete
                # rows are written NaN-filled at once (audit 2026-10-10 round 2, items 1 and 10).
                complete = do_csv and diag_keys is not None and not failed_skip   # a failed step has no metrics
                if complete and metrics_fields is None:
                    metrics_fields = list(row)
                    for pending in pending_rows:
                        artifacts.log_metrics(_conform_row(pending, metrics_fields))
                    pending_rows.clear()
                if metrics_fields is None:
                    pending_rows.append(row)
                else:
                    artifacts.log_metrics(row if complete else _conform_row(row, metrics_fields))
                timer.reset_pipeline_window()
            if do_log or do_csv:                                 # a log/eval row ends the divergence interval
                divergence.reset_interval()
    except BaseException as exc:
        _on_crash(exc)
        raise
    finally:
        update_hook.remove()
    if pending_rows:
        # No complete row was ever written (every diagnostic call failed): the incomplete rows share the union
        # of their columns.
        union = list(dict.fromkeys(key for pending in pending_rows for key in pending))
        for pending in pending_rows:
            artifacts.log_metrics(_conform_row(pending, union))
        pending_rows.clear()
    # S1 panel D handoff (2026-09-28), before the terminal callback so its checkpoint and summary see
    # it: the cumulative frame-projection counts for any later checkpoint, and summary.json's
    # "numerics_counters" (final cumulative counts, last measured per-evaluation counts, skip reasons).
    # The terminal checkpoint also carries the run's wall clock and the divergence-monitor state, which only the
    # periodic branch set before, so a resume from it keeps both (audit 2026-10-16 F77, F81). The last row has
    # already ended its interval, so the monitor is saved as it stands.
    if artifacts is not None:
        artifacts.wall_clock_state = _elapsed_wall_clock()
        artifacts.divergence_state = _divergence_checkpoint_record(
            divergence, interval_ends=False, skipped_steps=skipped_steps,
            consecutive_skips=consecutive_skips, last_group_norms=last_group_norms,
            failed_step_skips=failed_step_skips, loss_spike_rejections=loss_spike_rejections)
        try:
            artifacts.numerics_counters_state = {
                **(_frame_projection_state(frame_projection_counter, frame_projection_base,
                                           frame_projection_since) or {}),
                **_run_total_counters(run_total_base, precision_replays,
                                      phi_rejected_rows=int(getattr(optimizer, "phi_rejected_rows", 0)))}
            artifacts.numerics_counters = _numerics_counters_summary(
                completed_steps, optimizer, artifacts.numerics_counters_state, numerics_last_eval,
                numerics_reasons, diagnostic_windows)
        except Exception as exc:
            _record_diagnostic_failure(diagnostics_failures, "numerics_summary", exc, logger)
        # checkpoint_at_end (audit 2026-10-17 item 48): a resumable checkpoint of the final iterate, unless the periodic
        # cadence just wrote this step. Off by default, since each checkpoint holds the optimizer moments (several GB
        # at the live scale).
        if (getattr(cfg, "checkpoint_at_end", False) and getattr(artifacts, "save_model_artifacts", False)
                and n_steps > start_step
                and not (cfg.checkpoint_interval and n_steps % cfg.checkpoint_interval == 0)):
            _t0 = time.perf_counter()
            try:
                artifacts.save_checkpoint(n_steps, model, optimizer, cfg, scaler=scaler,
                                          data_state=_data_state_for(batches_consumed, loader_data_identity))
                artifacts.prune_checkpoints(getattr(cfg, "checkpoint_keep_last", None))
                logger.info(" end-of-run checkpoint step_%d.pt written in %.0f ms", n_steps,
                            (time.perf_counter() - _t0) * 1000.0)
            except Exception as exc:
                logger.warning(" end-of-run checkpoint at step %d failed: %s", n_steps, exc)
    # Realized-vs-configured update accounting (audit 2026-08-06 F5), handed to finalize_run through
    # the artifacts object because train() returns only `losses`. Built before the terminal callback,
    # so the ablation finalizer (finalize_validation_run) publishes it too (audit 2026-10-12 round 2,
    # item 7). The launcher summary.json's "n_steps" is
    # cfg.max_steps -- what was ASKED for. These say what actually happened, so an arm that quietly
    # trained on 95% of its budget is visible instead of being compared as if it had the full one.
    # CUMULATIVE over a resume: attempted counts every outer step from 0, and accepted_updates is the
    # optimizer's persisted accepted-update clock (restored by load_checkpoint and advanced by
    # train_step), so a continuation reports the whole run's budget, not only its own segment.
    # The fallback-element counters below are per-process and still cover this call only.
    attempted = int(completed_steps)
    accepted = _successful_update_count(optimizer)
    skipped_total = attempted - accepted
    if artifacts is not None:
        # Training-time diagnostic handoff to finalize_run (audit 2026-09-28): the optimizer-state
        # footprint for resources.json and the per-name diagnostic failure counts for summary.json.
        try:
            artifacts.optimizer_state_bytes = _optimizer_state_bytes(optimizer)
        except Exception as exc:
            _record_diagnostic_failure(diagnostics_failures, "optimizer_state_bytes", exc, logger)
        artifacts.diagnostics_failures = diagnostics_failures
        if diagnostics_failures:
            logger.warning(" diagnostic failures during training (counted, training unaffected): %s",
                           dict(sorted(diagnostics_failures.items())))
        run_totals = _run_total_counters(                                 # whole run across resumes
            run_total_base, precision_replays, phi_rejected_rows=int(getattr(optimizer, "phi_rejected_rows", 0)))
        neutralized = run_totals["nonfinite_tangent_elements"]
        mu_fallbacks = run_totals["mu_trust_fallback_elements"]
        artifacts.final_step = int(completed_steps)
        artifacts.realized_updates = {
            "attempted_steps":      attempted,
            "accepted_updates":     int(accepted),
            "skipped_steps":        int(skipped_total),
            "accepted_update_frac": (float(accepted) / attempted
                                     if attempted else float("nan")),
            **{key: run_totals[key] for key in RUN_TOTAL_COUNTER_KEYS if key != "matmul_precision_replays"},
            **({"matmul_precision_replays": run_totals["matmul_precision_replays"]}
               if precision_ladder else {}),
        }
        if skipped_total or neutralized or mu_fallbacks:
            logger.warning(
                " realized budget: %d/%d optimizer updates accepted (%.3f%%); "
                "%d SPD tangent element(s) neutralized as non-finite; "
                "%d belief(s) whitened by the non-equivariant mu-trust fallback",
                accepted, attempted,
                100.0 * accepted / max(attempted, 1),
                neutralized, mu_fallbacks)
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
            **({"window_offset": offset_windows.window_offset} if offset_windows is not None else {}),
        } if loader_data_identity is not None else None)
        terminal_callback(
            TrainingTerminalState(
                step=completed_steps,
                optimizer=optimizer,
                scaler=scaler,
                data_state=terminal_data_state,
                raw_model_state=raw_model_state,
                rng_state=rng_state,
                final_validation=(
                    {"val_loader": val_loader, "tokens_per_char": tokens_per_char,
                     "metrics": last_eval_record[1]}
                    if (last_eval_record is not None and last_eval_record[0] == completed_steps
                        and last_eval_record[2] == _tensor_versions(model))
                    else None),
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
        motion["steps"] = int(completed_steps - start_step)
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
    """Format a dedicated auxiliary LR while making ``None`` inheritance explicit."""
    return f"{mean_lr} (inherit mu)" if configured is None else str(configured)


def _fmt_decode_lrs(cfg: VFE3Config) -> str:
    """Show realized decoder rates, including inactive variance and shared-table cases."""
    mu_lr, sigma_lr = (cfg.resolved_hyperparameter(name) for name in ("m_decode_mu_lr", "m_decode_sigma_lr"))
    if cfg.decode_mode == "linear":
        return f"mu={mu_lr}  sigma=inactive"
    if not cfg.untie_decode_bank:
        return f"mu={cfg.m_p_mu_lr} (shared)  sigma={cfg.m_p_sigma_lr} (shared)"
    return f"mu={mu_lr}  sigma={sigma_lr}"


def _fmt_table_lrs(cfg: VFE3Config) -> str:
    r"""Banner rates of the type-prior tables. Under ``m_table_update_mode='fisher'`` the live step size is
    ``m_table_fisher_lr``; ``m_p_mu_lr`` / ``m_p_sigma_lr`` only schedule it and set the decoupled decay (audit
    2026-10-10 round 2, item 20)."""
    if getattr(cfg, "m_table_update_mode", "adamw") != "fisher":
        return f"mu={cfg.m_p_mu_lr}  sigma={cfg.m_p_sigma_lr}"
    return (f"tables=fisher eta={cfg.m_table_fisher_lr} r={cfg.m_table_fisher_trust_radius} "
            f"(schedule/decay rates mu={cfg.m_p_mu_lr} sigma={cfg.m_p_sigma_lr})")


def _fmt_tau(cfg: VFE3Config, model: VFEModel) -> str:
    r"""Banner format for the softmax temperature: a scalar kappa -> '1.5000', a per-head
    (list) kappa -> '[t0, t1, ...]' (T1). Converts a list kappa to a tensor first so attention_tau
    does not choke on a raw list. The minimal-agent readout routes at temperature 1 and never reads
    kappa_beta, so its banner prints that temperature instead (audit 2026-10-07 F30)."""
    if (getattr(cfg, "attention_mode", None) == "projected_predictive"
            and getattr(cfg, "predictive_readout", "kernel") == "minimal_agent"):
        return "1 (minimal_agent posterior; kappa_beta unused)"
    if (getattr(cfg, "attention_mode", None) == "projected_predictive"
            and getattr(cfg, "relational_routing", "gaussian_kl") == "noisy_key"):
        return "1 (noisy_key routing; kappa_beta unused)"
    dev = next(model.parameters()).device
    tau = attention_tau(_as_coeff(cfg.kappa_beta, dev), model.group.irrep_dims, device=dev)
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

    ``zero_grad_names`` lists the live parameters whose probe gradient is identically zero: autograd
    reaches them but their contribution vanishes at this point, as for the first layer of a block whose
    second layer starts at zero, or a path whose terms cancel. They stay counted as live, since a zero
    gradient at init need not persist; the list only keeps them from passing as ordinary live parameters
    (audit 2026-10-16, F110).
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
        "zero_grad_names": [],
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
        reached = [(name, p.grad) for name, p in named if p.requires_grad and p.grad is not None]
        nonzero = (torch.stack([grad.detach().ne(0).any().to(device) for _, grad in reached]).cpu().tolist()
                   if reached else [])                             # one host transfer for every flag
        zero_grad_names = [name for (name, _), flag in zip(reached, nonzero) if not flag]
        model.zero_grad(set_to_none=True)                          # leave no grads for training
        rep.update(live=live, dead=trainable - live, dead_names=dead_names,
                   zero_grad_names=zero_grad_names, probed=True)
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


def _known_dead_parameters(
    model:  torch.nn.Module,
    status: Mapping[str, Any],                  # train_step status_out of an accepted step
) -> Tuple[str, ...]:
    r"""Names of the trainable parameters that received no gradient on an accepted step.

    A parameter is live when it holds a gradient after the step or one reached ``optimizer.step``
    (``status["grads"]``): the table Fisher step and the manifold phi step clear ``.grad`` inside
    ``step()``, so ``p.grad is None`` alone would class those tables as dead and move a frozen one to
    the unwarned ``frozen_known_dead`` list (audit 2026-10-16, F4).
    """
    stashed: Mapping[int, torch.Tensor] = status.get("grads") or {}
    return tuple(name for name, p in model.named_parameters()
                 if p.requires_grad and p.grad is None and id(p) not in stashed)


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
    zero_line = ([" zero gradient at the init probe (counted live): "
                  + ", ".join(n.replace("prior_bank.", "") for n in rep["zero_grad_names"])]
                 if rep["probed"] and rep["zero_grad_names"] else [])
    return "\n".join([
        bar,
        f" Gauge VFE Transformer | {rep['total']:,} params{live_note}{probe_note} | {device}",
        bar,
        f" K={cfg.embed_dim}  N={cfg.max_seq_len}  L={cfg.n_layers}  heads={len(model.group.irrep_dims)}  "
        f"group={cfg.gauge_group}  family={cfg.family}",
        f" steps={n_steps}  batch={cfg.batch_size}  dataset={dataset}",
        *cov,
        *dead_line,
        *zero_line,
        f" M-LRs: {_fmt_table_lrs(cfg)}  phi={cfg.m_phi_lr}",
        f" decoder M-LRs: {_fmt_decode_lrs(cfg)}",
        f" VFE: lambda_alpha_mode={cfg.lambda_alpha_mode}  kappa_beta={cfg.kappa_beta}  "
        f"tau={_fmt_tau(cfg, model)}",
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
