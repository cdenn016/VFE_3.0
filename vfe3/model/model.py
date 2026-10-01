r"""The full VFE_3.0 model: encode -> E-step inference -> decode -> cross-entropy.

On the pure default path there are no neural layers (no nn.Linear/MLP/activation): the parameters are the
PriorBank's prior tables, plus the model-owned learned tables their toggles create -- the default
pos_phi='learned' positional table, and the default-OFF head mixer, CG coupling,
and learnable T5-bias scalar. The E-step
is unrolled into the training graph (the differentiable filtering kernel), so the CE
loss backpropagates through inference to the encode/phi priors. Batching loops over
the batch around the (unbatched) E-step; decode and CE are batched.
"""

import hashlib
import inspect
import math
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Dict

import torch
import torch.nn.functional as F
from torch import nn

from vfe3.attention_prior import attention_log_prior
from vfe3.belief import BeliefState
from vfe3.config import VFE3Config
from vfe3.contracts import (
    CanonicalFrameContext,
    ExecutableBuildMetadata,
    EStepGradientOutput,
    EStepGradientRecord,
    MStepCapture,
)
from vfe3.geometry.groups import GaugeGroup, get_group
from vfe3.geometry.lie_ops import CompactBlockElement, _equal_diag_blocks, project_phi_to_slk
from vfe3.geometry.norms import get_norm
from vfe3.geometry.rope import get_pos_rotation
from vfe3.geometry.transport import (
    CompactFactoredTransport,
    FactoredTransport,
    RopeTransport,
    TransportState,
    get_transport_registration,
    _build_flat,
)
from vfe3.model.canonical_content import project_canonical_diagonal
from vfe3.model.head_mixer import HeadMixer
from vfe3.model.block_mlp import (
    block_mlp_build_metadata,
    build_block_mlp,
    get_block_mlp_registration,
)
from vfe3.model.block import _as_coeff, vfe_block
from vfe3.model.frame_actions import frame_matvec
from vfe3.model.positional_phi import positional_phi_coords
from vfe3.model.prior_bank import (
    DecodeCEResult,
    PriorBank,
    get_decode_registration,
    get_encode,
    _encode_per_token,
)
from vfe3.model.stack import vfe_stack
from vfe3.families.gaussian import (
    FullGaussian,
    set_full_cov_kl_precision,
    validate_full_cov_kl_precision,
)
from vfe3.families.base import BeliefParams
from vfe3.families.policy import coupling_energy, family_instance
from vfe3.geometry.transport import (
    _rope_at_frame_precision,
    set_frame_pair_fp64_escalation,
    set_full_cov_congruence_certification,
    set_full_cov_congruence_precision,
)
from vfe3.numerics import set_mu_trust_cholesky_rounds, set_safe_cholesky_jitter_mode
from vfe3.relational_scoring import set_projected_condition_recovery, set_projected_factor_route
from vfe3.runtime import set_fp32_matmul_precision
from vfe3.model.relational_operators import RelationalOperators


def _block_mlp_diagnostic_eval(method: Callable) -> Callable:
    r"""Run an MLP-bearing diagnostic without stochastic dropout or mode leakage.

    Diagnostics may be called while the enclosing model remains in training mode. Only the
    coordinate MLPs need a mode override: putting them in evaluation mode makes their dropout
    deterministic without changing ordinary training forwards. Every descendant's prior mode is
    restored exactly, including on exceptions and for callers with mixed per-module modes.
    """
    @wraps(method)
    def guarded(self: "VFEModel", *args: Any, **kwargs: Any) -> Any:
        block_mlps = self.block_mlps
        if block_mlps is None:
            return method(self, *args, **kwargs)
        prior_modes = tuple((module, module.training) for module in block_mlps.modules())
        block_mlps.eval()
        try:
            return method(self, *args, **kwargs)
        finally:
            for module, was_training in prior_modes:
                module.training = was_training

    return guarded


def _derived_seed(seed: int, *stream: object) -> int:
    r"""Deterministic 63-bit generator seed for one named init stream of ``seed``.

    Each module that owns a dedicated ``torch.Generator`` draws from its own stream, keyed by
    ``(seed, *stream)`` through BLAKE2b, so two modules never replay the same normal sequence and
    ``seed + layer`` offsets cannot alias layer ``l`` of seed ``s`` onto layer 0 of seed ``s + l``.
    Python's ``hash`` is avoided because string hashing is salted per process.
    """
    key = repr((int(seed), *stream)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") & (2 ** 63 - 1)


def _positional_arity(builder: Callable) -> int:
    r"""Count the builder's required positional parameters (the K, n_heads, ... axes)."""
    n = 0
    for p in inspect.signature(builder).parameters.values():
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty:
            n += 1
    return n


def per_type_hidden_widths(n_gen: int, n_layers: int) -> Tuple[int, ...]:
    r"""Per-layer widths of the per-type hidden-bias table, :math:`\sum_\ell w_\ell = n_{\mathrm{gen}}`.

    .. math:: w_\ell = \lfloor n_{\mathrm{gen}}/L \rfloor + [\ell < n_{\mathrm{gen}} \bmod L],

    so the remainder units go to the first layers.
    """
    if n_layers < 1 or n_gen < 1:
        raise ValueError(f"per_type_hidden_widths needs n_gen >= 1 and n_layers >= 1, got {n_gen}, {n_layers}")
    base, extra = divmod(n_gen, n_layers)
    return tuple(base + (1 if layer < extra else 0) for layer in range(n_layers))


def build_group(cfg: VFE3Config) -> GaugeGroup:
    r"""Construct the gauge group from config, dispatching on the builder's positional
    arity so a newly registered group slots in by ``register_group`` alone (no call-site
    edit). Arity 1 -> ``builder(K)`` (glk, so_k); arity 2 -> ``builder(K, n_heads)``
    (block_glk). Higher arities are an unsupported registration error.

    ``cfg.cross_couplings`` (off-block GL(K) head coupling) is forwarded as a keyword only
    when set AND the selected builder accepts it (block_glk); otherwise the call is the bare
    positional dispatch, so the default (``cross_couplings=None``) path produces the SAME group
    object as before (byte-identical). Config validation already rejects cross_couplings against
    a builder that does not accept the kwarg, so this is the forwarding seam, not a second guard.
    ``cfg.group_n`` / ``cfg.irrep_spec`` are forwarded the same way (the so_n/sp_n irrep-tower
    builders; config validation rejects them for any other group)."""
    builder = get_group(cfg.gauge_group)
    arity = _positional_arity(builder)
    params = inspect.signature(builder).parameters
    kwargs: dict = {}
    if cfg.cross_couplings is not None and "cross_couplings" in params:
        kwargs["cross_couplings"] = cfg.cross_couplings
    if cfg.group_n is not None and "group_n" in params:
        kwargs["group_n"] = cfg.group_n
    if cfg.irrep_spec is not None and "irrep_spec" in params:
        kwargs["irrep_spec"] = cfg.irrep_spec
    if "close_basis" in params:
        kwargs["close_basis"] = cfg.cross_couplings is not None
    if arity == 1:
        return builder(cfg.embed_dim, **kwargs)
    if arity == 2:
        return builder(cfg.embed_dim, cfg.n_heads, **kwargs)
    raise ValueError(
        f"gauge group {cfg.gauge_group!r} builder has unsupported positional arity {arity}; "
        f"build_group dispatches K (arity 1) or (K, n_heads) (arity 2)"
    )


@dataclass(frozen=True)
class DiagnosticSnapshot:
    r"""Frozen tensors captured from one no-grad belief forward for evaluation consumers."""

    owner:             object = field(repr=False, compare=False)
    token_ids:         torch.Tensor
    encoded_belief:    BeliefState
    initial_belief:    BeliefState
    layer_priors:      'Tuple[Tuple[torch.Tensor, torch.Tensor], ...]'
    layer_converged:   'Tuple[BeliefState, ...]'
    layer_outputs:     'Tuple[BeliefState, ...]'
    layer_block_mlp_inputs:  'Tuple[BeliefState, ...]'
    layer_block_mlp_outputs: 'Tuple[BeliefState, ...]'
    stack_output:      BeliefState
    final_belief:      BeliefState
    logits:            torch.Tensor
    beta_maps:         torch.Tensor
    trace_states:      'Tuple[BeliefState, ...]'
    trace_free_energy: torch.Tensor
    rope:              Optional[torch.Tensor]
    log_prior:         Optional[torch.Tensor]
    relational_capture: Optional[dict] = None
    layer_conditional_captures: Tuple[dict, ...] = ()

    @property
    def conditional_capture(self) -> Optional[dict]:
        return self.relational_capture


def _freeze_tensor(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if value is None:
        return None
    frozen = value.detach()
    if frozen.dtype in (torch.float16, torch.bfloat16):
        frozen = frozen.float()
    return frozen.clone()


_FrameFreezeCache = Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]]


def _freeze_frame_copy(
    value: Optional[torch.Tensor],
    cache: Optional[_FrameFreezeCache],
) -> Optional[torch.Tensor]:
    """Own one version-bound frame copy for a private read-only snapshot lifetime."""
    if (cache is None or type(value) is not torch.Tensor
            or torch.is_inference_mode_enabled() or value.is_inference()):
        return _freeze_tensor(value)
    key = (id(value), value._version)
    if key not in cache:
        frozen = _freeze_tensor(value)
        cache[key] = (value, frozen)  # retain the source so Python cannot reuse its identity
        cache[(id(frozen), frozen._version)] = (frozen, frozen)
    return cache[key][1]


def _freeze_belief(
    belief: BeliefState,
    *,
    _frame_cache: Optional[_FrameFreezeCache] = None,
) -> BeliefState:
    return belief._replace(
        mu=_freeze_tensor(belief.mu),
        sigma=_freeze_tensor(belief.sigma),
        phi=_freeze_frame_copy(belief.phi, _frame_cache),
        right_phi=_freeze_frame_copy(belief.right_phi, _frame_cache),
    )


def _sequence_belief(belief: BeliefState, index: int = 0) -> BeliefState:
    return belief._replace(
        mu=belief.mu[index],
        sigma=belief.sigma[index],
        phi=belief.phi[index],
        right_phi=(belief.right_phi[index]
                   if belief.right_phi is not None and belief.right_phi.dim() == belief.phi.dim()
                   else belief.right_phi),
    )


def _belief_states_mechanically_equal(left: BeliefState, right: BeliefState) -> bool:
    """Prove exact state equality from the actual belief seam, not derived mappings."""
    if left is right:
        return True
    for name in left._fields:
        lhs, rhs = getattr(left, name), getattr(right, name)
        if lhs is rhs:
            continue
        if lhs is None or rhs is None:
            return False
        if isinstance(lhs, CompactBlockElement) and isinstance(rhs, CompactBlockElement):
            if lhs.K != rhs.K or lhs.tied != rhs.tied or not torch.equal(lhs.blocks, rhs.blocks):
                return False
        elif isinstance(lhs, torch.Tensor) and isinstance(rhs, torch.Tensor):
            if not torch.equal(lhs, rhs):
                return False
        elif lhs != rhs:
            return False
    return True


def _state_specific_free_energy_diagnostics(
    pre_block_mlp: Mapping[str, float],
    post_block_mlp: Mapping[str, float],
    *,
    states_equal_proven: bool = False,
) -> Dict[str, float]:
    """Serialize separately evaluated states; reject an accidental same-object alias.

    Only ``pre_block_mlp_*`` / ``post_block_mlp_*`` keys are emitted. The unqualified ``total`` is
    the caller's headline F at the converged E-step state and is never overwritten here.
    """
    if pre_block_mlp is post_block_mlp and not states_equal_proven:
        raise RuntimeError(
            "pre/post BlockMLP diagnostics require separate component evaluations unless "
            "state equality is mechanically established")
    components = (
        "self_coupling", "belief_coupling", "attention_entropy",
        "observation_nll",
    )
    raw_components = (
        "self_divergence", "observation_likelihood",
    )

    def normalized(source: Mapping[str, float]) -> Dict[str, float]:
        values = {name: float(source.get(name, 0.0)) for name in components}
        values.update({name: float(source.get(name, 0.0)) for name in raw_components})
        values["total"] = float(sum(values[name] for name in components))
        return values

    pre = normalized(pre_block_mlp)
    post = normalized(post_block_mlp)
    out = {f"pre_block_mlp_{name}": value for name, value in pre.items()}
    out.update({f"post_block_mlp_{name}": value for name, value in post.items()})
    return out


def _numerical_policy_diagnostics(
    *,
    m_phi_group_trust_radius: Optional[float],
    phi_mstep_max_matrix_norm: Optional[float],
    transport_chart_max_norm: Optional[float],
    exp_fp64_norm_threshold: Optional[float],
) -> Dict[str, Optional[float]]:
    """Return the exact active production numerical policy, including disabled bounds."""
    return {
        "m_phi_group_trust_radius": m_phi_group_trust_radius,
        "phi_mstep_max_matrix_norm": phi_mstep_max_matrix_norm,
        "transport_chart_max_norm": transport_chart_max_norm,
        "exp_fp64_norm_threshold": exp_fp64_norm_threshold,
    }


class VFEModel(nn.Module):
    """encode -> E-step stack -> decode -> CE; default-off MLP preserves the existing topology."""

    def __init__(self, cfg: VFE3Config) -> None:
        super().__init__()
        # Reproducibility is pinned at the entry point run_training (torch.manual_seed(cfg.seed)
        # before model + loader are built), NOT here: seeding inside __init__ would clobber a
        # caller-set RNG state (e.g. a test that seeds then constructs several models).
        self.cfg = cfg
        # The full-covariance KL precision policy is model state and the process-wide default for
        # un-threaded family callers. It is installed under the same one-place-set rule as the
        # congruence, Cholesky, and fp32-matmul policies below, so direct FullGaussian calls and the
        # PriorBank fallback resolve the config-selected policy rather than the module default.
        self.full_cov_kl_precision = validate_full_cov_kl_precision(
            cfg.full_cov_kl_precision)
        self.full_cov_kl_backend = cfg.full_cov_kl_backend
        # Same one-place-set rule for the KL policy itself and the congruence that feeds it
        # (audit 2026-08-06 C1).
        set_full_cov_kl_precision(self.full_cov_kl_precision)
        set_full_cov_congruence_precision(cfg.full_cov_congruence_precision)
        set_full_cov_congruence_certification(cfg.full_cov_congruence_certification)
        set_frame_pair_fp64_escalation(cfg.frame_pair_fp64_escalation)
        set_projected_condition_recovery(cfg.projected_cov_condition_recovery)
        set_projected_factor_route(cfg.projected_factor_route)
        set_safe_cholesky_jitter_mode(cfg.safe_cholesky_jitter_mode)
        set_mu_trust_cholesky_rounds(cfg.mu_trust_cholesky_rounds)
        # ... and the fp32 matmul / TF32 policy (audit 2026-08-06 C7/F25), which is process-global
        # for the same reason and belongs to the same one-place-set rule. It lives here rather than
        # in `seed_everything` because that is called by four separate driver entry points, only one
        # of which is canonical -- so wiring it there would leave the policy applied on some runs and
        # not others, which is the exact failure C7 exists to close. None (the default) leaves the
        # card untouched, so this line changes nothing unless a config asks for it.
        set_fp32_matmul_precision(cfg.fp32_matmul_precision)
        self._transport_status: Dict[str, bool] = {}
        self.group = build_group(cfg)
        # Keep layer zero's state-dict prefix for existing single-layer checkpoints.
        self.relational_operators = None
        self.relational_operators_extra = nn.ModuleList()
        if cfg.relational_mode == "projected_filtering":
            # 'depth_scaled' starts the residual-branch writer O at 1/sqrt(2 L) so the L-layer
            # stack of near-uniform routing at init does not add a unit-gain token mean per layer.
            output_init_scale = (1.0 / math.sqrt(2.0 * cfg.n_layers)
                                 if cfg.relational_output_init == "depth_scaled" else 1.0)
            layers = [
                RelationalOperators(list(self.group.irrep_dims),
                                    _derived_seed(cfg.seed, "relational_operators", layer_index),
                                    readout_mode=cfg.relational_readout_mode,
                                    input_scope=("full" if cfg.attention_mode == "projected_predictive"
                                                 else cfg.relational_input_scope),
                                    output_scope=("full" if cfg.attention_mode == "projected_predictive"
                                                  else cfg.relational_output_scope),
                                    attention_residual=cfg.attention_residual,
                                    identity_content=cfg.relational_identity_content,
                                    output_init_scale=output_init_scale,
                                    routing=cfg.relational_routing,
                                    predictive_covariance=cfg.predictive_covariance,
                                    predictive_covariance_centering=cfg.predictive_covariance_centering,
                                    predictive_covariance_floor=cfg.predictive_covariance_floor)
                for layer_index in range(cfg.n_layers)
            ]
            self.relational_operators = layers[0]
            self.relational_operators_extra.extend(layers[1:])
        # ALiBi-family priors carry a per-head (n_heads, N, N) axis, while the energy's head axis
        # is len(irrep_dims); a mismatch right-aligns the prior's head axis against the BATCH axis
        # of a single-block (B, N, N) energy -- silent corruption at B=1, RuntimeError otherwise
        # (audit 2026-06-09 P1). Reject at construction. Single-block groups may run alibi with
        # n_heads=1 (the (1, N, N) prior is squeezed to the (N, N) convention in
        # _attention_log_prior below).
        for _pname in ("beta_attention_prior",):
            if (getattr(cfg, _pname) in ("alibi", "causal_alibi", "causal_alibi_noself")
                    and cfg.n_heads != len(self.group.irrep_dims)):
                raise ValueError(
                    f"{_pname}={getattr(cfg, _pname)!r} builds an (n_heads, N, N) prior but the "
                    f"energy head axis is {len(self.group.irrep_dims)} "
                    f"(irrep_dims={self.group.irrep_dims}); set "
                    f"n_heads={len(self.group.irrep_dims)} or use a headless prior "
                    f"(uniform/causal/windowed/...)."
                )
        n_gen = self.group.generator_shape[0]
        # canonical_content_gauge intentionally reaches the existing per-token table path below:
        # gaussian_frame_diagonal reads those values as intrinsic (a_v, s_v), so VFEModel must not
        # materialize a dense frame pushforward before the flat-cocycle family cancels phi. An
        # optional unigram base-rate is added only after this exact Gaussian-divergence decode.
        self.prior_bank = PriorBank(
            cfg.vocab_size, cfg.embed_dim, n_gen,
            mu_init_std=cfg.mu_init_std, sigma_init=cfg.sigma_init, phi_scale=cfg.phi_scale,
            decode_tau=cfg.decode_tau, eps=cfg.eps,
            decode_av_precision=cfg.decode_av_precision,
            diagonal_covariance=cfg.diagonal_covariance,
            family=cfg.family,
            # divergence_family/renyi_order feed the family-consistent decode kernels
            # (decode_mode='family'/'family_chunked'), so the readout scores the SAME divergence the
            # E-step minimized (PB-14). The fast gaussian kernels ignore them.
            divergence_family=cfg.divergence_family, renyi_order=cfg.renyi_order,
            full_cov_precision_policy=self.full_cov_kl_precision,
            decode_bias=cfg.decode_bias,
            encode_mode=cfg.encode_mode, decode_mode=cfg.decode_mode,
            decode_chunk_size=cfg.decode_chunk_size,
            decode_ce_checkpoint=cfg.decode_ce_checkpoint,
            decode_ce_stable_reduction=cfg.decode_ce_stable_reduction,
            decode_ranking_fp64_escalation=cfg.decode_ranking_fp64_escalation,
            decode_ranking_refinement=cfg.decode_ranking_refinement,
            # Tier-1/Tier-2 decode toggles (2026-07-05; all default OFF, byte-identical):
            unigram_kappa=cfg.unigram_kappa,
            decode_unigram_prior=cfg.decode_unigram_prior,
            untie_decode_bank=cfg.untie_decode_bank and cfg.decode_mode != "linear",
            irrep_dims=list(self.group.irrep_dims),
            use_priorbank_head_evidence_mixer=cfg.use_priorbank_head_evidence_mixer,
            gauge_group_is_tied=(cfg.gauge_group == "tied_block_glk"),
            gauge_group_name=self.group.name,
            decode_coordinates=cfg.decode_coordinates,
        )
        if not cfg.decode_temperature_learned:
            # Fixed Gibbs readout temperature: the parameter keeps its name (checkpoint layout is
            # unchanged) but leaves autograd and, through build_optimizer's requires_grad filter,
            # the optimizer, so tau_eff = decode_tau * exp(-clamp(decode_log_scale, -3, 3)) stays at
            # its initialized (or loaded) value.
            self.prior_bank.decode_log_scale.requires_grad_(False)
        # Norm instances built ONCE (audit 2d/4f). The gauge-pure norms (none/mahalanobis) and the
        # parameter-free layernorm are stateless pure maps (K, eps); layernorm_affine=True makes a
        # "layernorm" seam an AffineLayerNorm nn.Module whose gamma/beta REGISTER as submodule params
        # (grouped by build_optimizer). Each seam gets its own affine pair; the block seam's affine
        # is shared across the L blocks (the block norm is a single shared instance). affine=... is
        # ignored by the none/mahalanobis builders (they accept **kwargs).
        self.block_norm = get_norm(cfg.norm_type_block)(cfg.embed_dim, eps=cfg.eps,
                                                         family=cfg.family,
                                                         affine=cfg.layernorm_affine) \
            if cfg.norm_type_block != "none" else None
        # One untied selected residual transform per block. The disabled path adds no module,
        # parameter, or state-dict key; coordinate + passthrough retains the historical topology.
        self.block_mlps = (
            nn.ModuleList([
                build_block_mlp(
                    cfg.block_mlp_mode,
                    embed_dim=cfg.embed_dim,
                    irrep_dims=self.group.irrep_dims,
                    expansion=cfg.block_mlp_expansion,
                    covariance_floor=cfg.block_mlp_covariance_floor,
                    activation=cfg.block_mlp_activation,
                    dropout=cfg.block_mlp_dropout,
                    covariance_contract=cfg.block_mlp_covariance,
                )
                for _ in range(cfg.n_layers)
            ]) if cfg.use_block_mlp else None
        )
        # Per-type hidden bias (default off: no parameter, no state-dict key, no RNG draw). A (V, n_gen)
        # table E, zero-initialized so step 0 equals the model without it, split over the layers into
        # widths w_l; layer l adds E[x_i, slice_l] to the first w_l hidden pre-activation units of its
        # block MLP. Every w_l must fit that layer's hidden width.
        self._per_type_hidden_widths: Tuple[int, ...] = ()
        if not cfg.per_type_hidden_bias:
            self.per_type_hidden_bias = None
        else:
            widths = per_type_hidden_widths(n_gen, cfg.n_layers)
            for layer_index, (width, mlp) in enumerate(zip(widths, self.block_mlps)):
                fc1 = getattr(mlp, "fc1", None)
                if fc1 is None or width > fc1.out_features:
                    raise ValueError(
                        f"per_type_hidden_bias: layer {layer_index} needs {width} hidden units of the "
                        f"n_gen={n_gen} table split over n_layers={cfg.n_layers}, but its block MLP "
                        f"{type(mlp).__name__} has "
                        f"{'no fc1' if fc1 is None else f'{fc1.out_features} hidden units'}; raise "
                        "block_mlp_expansion")
            self.per_type_hidden_bias = nn.Parameter(torch.zeros(cfg.vocab_size, n_gen))   # (V, n_gen)
            self._per_type_hidden_widths = widths
        self._block_mlp_registration = get_block_mlp_registration(cfg.block_mlp_mode)
        self.executable_build = ExecutableBuildMetadata(
            block_mlp=block_mlp_build_metadata(
                enabled=cfg.use_block_mlp,
                mode=cfg.block_mlp_mode,
                covariance=cfg.block_mlp_covariance,
                expansion=cfg.block_mlp_expansion,
                activation=cfg.block_mlp_activation,
                dropout=cfg.block_mlp_dropout,
                covariance_floor=cfg.block_mlp_covariance_floor,
                embed_dim=cfg.embed_dim,
                irrep_dims=self.group.irrep_dims,
                n_layers=cfg.n_layers,
                covariance_layout=("full" if cfg.family == "gaussian_full" else "diagonal"),
            )
        )
        self.final_norm = get_norm(cfg.norm_type_final)(cfg.embed_dim, eps=cfg.eps,
                                                         family=cfg.family,
                                                         affine=cfg.layernorm_affine) \
            if cfg.norm_type_final != "none" else None
        # layernorm_affine footgun / inert warnings. The affine gamma/beta exist only on a
        # "layernorm" seam; on the BLOCK seam they are applied inside the E-step, so a severing
        # estimator freezes them.
        if cfg.layernorm_affine:
            _ln_block = cfg.norm_type_block == "layernorm"
            _ln_final = cfg.norm_type_final == "layernorm"
            if not (_ln_block or _ln_final):
                import warnings
                warnings.warn(
                    "layernorm_affine=True but neither norm_type_block nor norm_type_final is "
                    "'layernorm': no affine gamma/beta are created and the toggle is inert. Set a "
                    "norm seam to 'layernorm' to use the learned affine.",
                    UserWarning, stacklevel=2,
                )
            elif _ln_block and cfg.e_step_gradient == "detach":
                # The block affine is applied to the belief VALUE, so it trains under 'unroll' AND
                # 'straight_through' (both keep the value differentiable) and is frozen ONLY by the
                # fully-detached E-step, which no_grads the whole stack.
                import warnings
                warnings.warn(
                    "layernorm_affine=True with norm_type_block='layernorm' and the effective "
                    "E-step estimator 'detach' freezes the BLOCK norm's gamma/beta: 'detach' runs "
                    "the whole belief stack (including the block norm) under no_grad, so the affine "
                    "receives no gradient. Use 'unroll' or 'straight_through' (both keep the belief "
                    "value differentiable), or place the affine layernorm on norm_type_final "
                    "(post-stack, trains under any estimator).",
                    UserWarning, stacklevel=2,
                )
        # Opt-in Schur-commutant head mixer (default off). Built ONCE from the gauge group's
        # irrep blocks. Label-less groups need >= 2 EQUAL blocks (block_glk/tied_block_glk);
        # labeled irrep towers (so_n/sp_n) mix per isotypic component (mults-one towers get
        # per-head scalar gains -- the entire linear commutant there). Bad pairings fail here,
        # not at forward.
        self.head_mixer = HeadMixer(self.group.irrep_dims, family=cfg.family,
                                     irrep_labels=self.group.irrep_labels) \
            if cfg.use_head_mixer else None
        # Opt-in CG cross-type coupling (default off; so_n/sp_n only). Built ONCE from the
        # group's labels; CGCoupling raises at construction when no admissible paths exist.
        # The algebra key comes from the GROUP OBJECT (set by the so_n/sp_n builders), not a
        # re-derivation from the config string, so a newly registered labeled-tower group
        # cannot mis-dispatch here (audit 2026-06-09 overnight RF1).
        if cfg.use_cg_coupling:
            from vfe3.model.cg_coupling import CGCoupling
            self.cg_coupling = CGCoupling(
                cfg.group_n, self.group.algebra,
                self.group.irrep_dims, self.group.irrep_labels,
                cg_covariance_mode=cfg.cg_covariance_mode,
                cg_covariance_floor=cfg.cg_covariance_floor)
        else:
            self.cg_coupling = None
        if (cfg.use_head_mixer
                or (cfg.use_cg_coupling and cfg.cg_energy_weight == 0.0)) \
                and cfg.e_step_gradient == "detach":
            # Footgun (mirrors the pos_phi_free / t5_bias footguns below): the
            # mixer and the CG coupling are applied INSIDE the vfe_stack call, which the
            # 'detach' estimator wraps wholesale in no_grad (block.py:73-78 under
            # model.forward's `run`), so mixer_deltas / path_weights build no graph, receive
            # no gradient, and silently stay frozen at their identity/zero init -- the model
            # trains its other parameters and LOOKS healthy while these two opt-in components
            # never adapt (audit 2026-06-09 overnight F31, challenge-upheld). Gate on the
            # Canonical e_step_gradient selector determines whether the stack is grad-enabled.
            # EXCEPTION (PB-13): with
            # cg_energy_weight>0 the post-stack torch.enable_grad re-evaluation trains
            # path_weights through the CG moment energy even under detach (the mean->CE path
            # stays detached), so the 'frozen at zero init' claim is false there and the CG
            # clause is gated off; the head mixer has no such side channel and always warns.
            import warnings
            warnings.warn(
                "use_head_mixer/use_cg_coupling with the effective E-step estimator 'detach' "
                "freezes mixer_deltas/path_weights: both modules are applied inside the "
                "no_grad-wrapped vfe_stack, so they receive NO gradient and stay at their "
                "identity/zero init. Use e_step_gradient='unroll' or 'straight_through' to "
                "train them (or, for the "
                "CG coupling alone, cg_energy_weight>0, whose detached-mode re-evaluation "
                "trains path_weights).",
                stacklevel=2,
            )
        # Optional trainable transport state is owned by the selected registry record. Parameters are
        # registered directly on this module under the record's declared serialization keys: this
        # preserves the established top-level checkpoint names while the model remains independent of
        # every concrete transport and state name. The flat registration has no builder, so the pure
        # path constructs no trainable transport state.
        transport_registration = get_transport_registration(cfg.transport_mode)
        self._transport_state_keys = transport_registration.serialization_keys
        self._transport_offdiag_state_keys = transport_registration.offdiag_serialization_keys
        if transport_registration.state_builder is not None:
            built_state = transport_registration.state_builder(cfg, self.group)
            if set(built_state) != set(self._transport_state_keys):
                raise ValueError(
                    f"transport {cfg.transport_mode!r} state builder returned keys "
                    f"{tuple(built_state)!r}; declared serialization keys are "
                    f"{self._transport_state_keys!r}"
                )

            if len({id(parameter) for parameter in built_state.values()}) != len(built_state):
                raise ValueError(f"transport {cfg.transport_mode!r} state parameters must be unique")
            for state_key in self._transport_state_keys:
                parameter = built_state[state_key]
                if not isinstance(parameter, nn.Parameter):
                    raise TypeError(
                        f"transport {cfg.transport_mode!r} state {state_key!r} must be an nn.Parameter, "
                        f"got {type(parameter).__name__}"
                    )
                self.register_parameter(state_key, parameter)
            if cfg.e_step_gradient == "detach":
                import warnings
                warnings.warn(
                    f"transport_mode={cfg.transport_mode!r} with e_step_gradient='detach' freezes trainable "
                    f"transport state {self._transport_state_keys!r}: that state enters the loss only "
                    "through the E-step, which the detached (no_grad) path severs. Set "
                    "e_step_gradient='unroll' to train the transport state.",
                    stacklevel=2,
                )
        if cfg.decode_mode == "linear" and cfg.e_step_gradient == "detach":
            # Joint-toggle footgun (audit 2026-05-31): the detached E-step severs the encode prior
            # tables (mu/sigma/phi_embed) from the loss, and the linear decode reads only mu_final,
            # so ONLY output_proj_weight would train -- the prior bank is effectively frozen.
            import warnings
            warnings.warn(
                "decode_mode='linear' with e_step_gradient='detach' freezes the encode prior tables "
                "(mu_embed/sigma_log_embed/phi_embed): the detached E-step cuts them off and the "
                "linear decode reads only mu_final, so only output_proj_weight trains. Set "
                "e_step_gradient='unroll' to learn the prior tables.",
                stacklevel=2,
            )
        # Causal/attention log-prior is loop-invariant for fixed (N, device, dtype); cache it
        # (audit 4e) keyed on those so it is built once, not every forward. Not an nn.buffer
        # because it depends on the runtime N (sequence length), which varies across calls.
        self._log_prior_cache: dict = {}
        # Gauge-RoPE rotation R(theta) is similarly loop-invariant for fixed (N, device, dtype);
        # cache keyed on (N, device, dtype) so it is built once. None on the 'none' path.
        self._rope_cache: dict = {}
        # BCH positional encoding (default-off): a learned per-position Lie-algebra element table
        # composed into the gauge frame before transport. Created ONLY for pos_phi='learned' (a raw
        # nn.Parameter, not a network); the "none"/"frozen" paths add no
        # parameter, so the pure path stays param-free. Init scaled by pos_phi_scale.
        if cfg.pos_phi == "learned":
            # Seed pos_phi_free from a DEDICATED generator, independent of the global RNG stream
            # position, so its init does not depend on how much RNG the prior-bank tables consumed
            # before it. Its stream is derived from (cfg.seed, 'pos_phi_free'), distinct from every
            # relational operator stream. A separate generator does NOT clobber the caller's global
            # RNG, so the reproducibility seam stays at run_training; mu/sigma/phi inits are unchanged.
            _g = torch.Generator().manual_seed(_derived_seed(cfg.seed, "pos_phi_free"))
            self.pos_phi_free = nn.Parameter(
                torch.randn(cfg.max_seq_len, n_gen, generator=_g) * cfg.pos_phi_scale)
            if cfg.e_step_gradient in ("detach", "straight_through"):
                # Footgun (mirrors the t5_bias footgun below): pos_phi_free enters the loss ONLY
                # through the E-step belief transport. The 'detach' and 'straight_through' estimators
                # both sever that path (no_grad / a detached tangent), so the positional table
                # receives no gradient and stays frozen at init. Set the canonical E-step estimator
                # to 'unroll' to learn it.
                import warnings
                warnings.warn(
                    "pos_phi='learned' with the effective E-step estimator "
                    f"{cfg.e_step_gradient!r} freezes pos_phi_free: the positional gauge "
                    "element enters the loss only through the E-step transport, which the "
                    "detached / straight-through E-step severs. Use an 'unroll' E-step "
                    "(e_step_gradient='unroll') to train it.",
                    stacklevel=2,
                )
        # gauge_transport='frozen'/'off' DEFINE the frame as non-learning: config.__post_init__
        # coerces e_phi_lr=m_phi_lr=0.0, so neither the E-step phi substep (gated on e_phi_lr>0) nor
        # the M-step optimizer can move these tables. Sever them from autograd as well. Under
        # requires_grad=False the transport build records no graph edge from the frame, so the
        # backward never instantiates or traverses the Omega exp/einsum adjoint -- it previously did
        # so to produce a phi gradient that the zero LR could only multiply by zero. Numerics are
        # unchanged, because a zero-LR step is a no-op. build_optimizer's exact-coverage guard
        # exempts requires_grad=False parameters as frozen-by-design, so the frame STAYS in its
        # optimizer group: AdamW skips it (p.grad is None) while the phi-role weight norm remains
        # reportable. 'off' already coerces pos_phi='none', so only the token frame exists there.
        if cfg.gauge_transport in ("frozen", "off"):
            for _frame in (getattr(self.prior_bank, "phi_embed", None),
                           getattr(self, "pos_phi_free", None)):
                if _frame is not None:
                    _frame.requires_grad_(False)
        # NEURAL-NETWORK EXCEPTION (sanctioned, default-off): a LEARNED T5 relative-position
        # attention-bias table. When t5_learnable_bias=True, create t5_bias as a trainable
        # nn.Parameter of shape (t5_num_buckets,); the attention log-prior reads it as the per-bucket
        # bias b_{i-j} (manuscript GL(K)_attention.tex:826-838: pi_j ∝ exp(b_{i-j}), beta_ij =
        # softmax_j(-E_ij/tau + b_{i-j}), the first-principles T5 derivation). Init to the fixed-table
        # default -log1p(bucket) so a learnable model is byte-identical to the fixed t5_relative_bias
        # prior at step 0, then trains. Unlike the gauge/value exceptions this bias is a scalar
        # function of position OFFSET only and touches NO gauge transport, so it does NOT break gauge
        # equivariance. Created ONLY when t5_relative_bias is an active channel (so the parameter is
        # never orphaned); else no t5_bias attribute and the
        # pure path stays param-free (the fixed-table default still runs).
        if cfg.t5_learnable_bias and cfg.beta_attention_prior == "t5_relative_bias":
            self.t5_bias = nn.Parameter(-torch.log1p(torch.arange(cfg.t5_num_buckets, dtype=torch.float32)))
            if cfg.e_step_gradient in ("detach", "straight_through"):
                # Footgun (mirrors pos_phi_free above):
                # the attention log-prior is consumed INSIDE the E-step, and BOTH severing estimators
                # cut t5_bias's only gradient path -- 'detach' wraps the whole E-step in no_grad, and
                # 'straight_through' detaches the per-iteration belief tangent (e_step.py), the sole
                # carrier of the attention-prior signal into the belief. Either way t5_bias.grad is
                # None and the bias stays frozen at its fixed-table init; only the 'unroll' (default)
                # E-step trains it.
                import warnings
                warnings.warn(
                    f"t5_learnable_bias=True with the effective E-step estimator "
                    f"{cfg.e_step_gradient!r} freezes t5_bias: the T5 relative-position bias "
                    f"enters the loss only through the attention log-prior consumed inside the E-step, "
                    f"whose gradient both 'detach' (no_grad) and 'straight_through' (detached tangent) "
                    f"sever, so t5_bias.grad is None and the bias stays at its fixed-table init. Use an "
                    f"'unroll' E-step (e_step_gradient='unroll') to train it.",
                    stacklevel=2,
                )
        elif cfg.t5_learnable_bias:
            # t5_learnable_bias=True but no 't5_relative_bias' channel is active (the beta attention
            # prior is not 't5_relative_bias'): no t5_bias parameter is created and the toggle is
            # silently inert. Warn so the dead toggle is not mistaken for a trained bias.
            import warnings
            warnings.warn(
                "t5_learnable_bias=True but no 't5_relative_bias' attention channel is active "
                "(beta_attention_prior): no learnable t5_bias is created and "
                "the toggle is inert. Set beta_attention_prior to "
                "'t5_relative_bias' to use the learnable bias.",
                UserWarning, stacklevel=2,
            )
        # DEFAULT-OFF learned partner-prior residual. Each irrep block learns only two scalar
        # coefficients applied to detached scalar pair-energy features captured once at
        # the fixed pre-stack seam:
        #   B_ij^h = tanh(w_h^T [log(1 + E_ij^h/d_h), log(1 + E_ji^h/d_h)]).
        # Zero initialization makes the enabled model exactly equal to the base prior at step 0 and
        # draws no RNG. The feature computation follows the active transport and gauge-RoPE but is
        # detached, so only this bounded scalar prior learns; the analytic E-step still receives one
        # fixed log-prior tensor rather than a candidate-dependent score network.
    def _coupling_energy(
        self, family: type[BeliefParams], *args: Any,
        log_prior: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Evaluate a model-owned coupling grid under this model's immutable precision policy."""
        return coupling_energy(
            family, *args, log_prior=log_prior, precision_policy=self.full_cov_kl_precision,
            score_backend=self.full_cov_kl_backend, **kwargs)

    @property
    def relational_operators_by_layer(self) -> Tuple[RelationalOperators, ...]:
        """Return independent layer projections without registering duplicate modules."""
        if self.relational_operators is None:
            return ()
        return (self.relational_operators, *self.relational_operators_extra)

    @property
    def conditional_filtering_enabled(self) -> bool:
        return self.relational_operators is not None

    @property
    def conditional_capture_key(self) -> str:
        return "relational_capture"

    @property
    def conditional_semantics(self) -> str:
        return (self.relational_operators.inference_semantics if self.relational_operators is not None
                else "legacy_mean_field")

    def _belief_score_irrep_dims(self) -> Sequence[int]:
        r"""Return the ambient block sizes the beta score is computed in."""
        return self.group.irrep_dims

    def _family_instance(
        self, family: type[BeliefParams], mu: torch.Tensor, sigma: torch.Tensor,
    ) -> BeliefParams:
        """Construct a model-owned Gaussian with this model's immutable policy."""
        return family_instance(mu, sigma, family, self.full_cov_kl_precision)

    def _transport_dispersion(self, family, dispersion, omega, **kwargs) -> torch.Tensor:
        """Transport model-owned full covariance under this model's immutable policy."""
        if family is FullGaussian:
            kwargs["precision_policy"] = self.full_cov_kl_precision
        return family.transport_dispersion(dispersion, omega, **kwargs)

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict:     bool = True,
        assign:     bool = False,
    ):
        """Strictly load current state."""
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True) -> "VFEModel":
        r"""Carry the gauge group's generators through ``.to(...)`` / ``.cuda()`` etc.

        ``self.group`` is a plain ``GaugeGroup`` dataclass, not an ``nn.Module``, so its
        ``generators`` tensor is outside the parameter/buffer system and would NOT follow a
        dtype/device move -- leaving the E-step transport (belief.phi, which DOES move)
        matmul'd against stale-device/dtype generators. Re-map them here so the module's
        device/dtype contract holds (CLAUDE.md: device-agnostic, float32-with-CUDA)."""
        super()._apply(fn, recurse)
        self.group.apply_generators(fn)
        self._log_prior_cache.clear()        # device/dtype moved: cached masks are now stale
        self._rope_cache.clear()             # device/dtype moved: cached rotations are now stale
        return self

    def _attention_log_prior(
        self,
        n:      int,                          # sequence length N (varies across calls)
        device: torch.device,

        *,
        prior:  Optional[str] = None,         # prior-registry name; None -> cfg.beta_attention_prior (belief block)
    ) -> torch.Tensor:
        r"""Loop-invariant attention log-prior, cached on (name, N, device, dtype) (audit 4e).

        The dtype is taken from the prior-bank mean table so the mask matches the belief
        dtype after a ``.to(torch.float64)`` move (audit 2f: the old call omitted dtype). ``prior``
        lets the gamma model-coupling block reuse the same cache under its own attention prior."""
        name = prior if prior is not None else self.cfg.beta_attention_prior
        dtype = self.group.generator_dtype
        # Learnable T5 bias: the per-bucket table is a live nn.Parameter that changes every step, so
        # the (name, N, ...) cache MUST be bypassed -- a cached tensor would serve a stale table and
        # sever the gradient. Build fresh each call, passing the parameter as bias_values so the loss
        # backpropagates to t5_bias (through the E-step). getattr keeps this None-safe when the param
        # was not created (t5_learnable_bias off, or no active t5 channel); .to preserves the graph.
        if name == "t5_relative_bias" and getattr(self, "t5_bias", None) is not None:
            out = attention_log_prior(
                name, n, n,
                device=device, dtype=dtype,
                n_heads=self.cfg.n_heads, alibi_slope=self.cfg.alibi_slope,
                window=self.cfg.attention_window,
                num_buckets=self.cfg.t5_num_buckets, max_distance=self.cfg.t5_max_distance,
                bias_values=self.t5_bias.to(device=device, dtype=dtype))
            if out.dim() == 3 and len(self.group.irrep_dims) == 1:
                out = out.squeeze(0)
            return out
        # The key carries every cfg field a builder consumes (n_heads, alibi_slope, window, T5
        # bucketing) so a post-construction cfg mutation cannot serve a stale prior (audit PP4/P9).
        key = (name, n, device, dtype, self.cfg.n_heads, self.cfg.alibi_slope,
               self.cfg.attention_window, self.cfg.t5_num_buckets, self.cfg.t5_max_distance)
        cached = self._log_prior_cache.get(key)
        if cached is None:
            cached = attention_log_prior(
                name, n, n,
                device=device, dtype=dtype,
                n_heads=self.cfg.n_heads, alibi_slope=self.cfg.alibi_slope,
                window=self.cfg.attention_window,
                num_buckets=self.cfg.t5_num_buckets, max_distance=self.cfg.t5_max_distance,
            )
            # A (1, N, N) per-head prior on a single-block group collapses to the (N, N)
            # single-block convention (the energy carries no head axis there); the construction
            # guard above pins n_heads == 1 for that case, so shape[0] == 1 here.
            if cached.dim() == 3 and len(self.group.irrep_dims) == 1:
                cached = cached.squeeze(0)
            self._log_prior_cache.clear()
            self._log_prior_cache[key] = cached
        return cached

    def _rope_rotation(
        self,
        n:      int,                          # sequence length N (varies across calls)
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        r"""Cached gauge-RoPE rotation R(theta) for length n (None when pos_rotation='none')."""
        if self.cfg.pos_rotation == "none":
            return None
        dtype = self.group.generator_dtype
        key = (n, device, dtype, self.cfg.pos_rotation, self.cfg.rope_base)
        cached = self._rope_cache.get(key)
        if cached is None:
            cached = get_pos_rotation(self.cfg.pos_rotation)(
                torch.arange(n, device=device), self.group.irrep_dims,
                base=self.cfg.rope_base, device=device, dtype=dtype)
            self._rope_cache.clear()
            self._rope_cache[key] = cached
        return cached

    def _amp_context(
        self,
        device: torch.device,            # resolves autocast device_type ('cuda' | 'cpu')
    ):
        r"""Opt-in autocast context for the E-step (cfg.amp_dtype), else a nullcontext.

        amp_dtype=None (default) returns ``nullcontext()`` so the default path NEVER instantiates
        a ``torch.autocast`` object -- the forward stays byte-identical to the no-AMP build. 'bf16'
        / 'fp16' return ``torch.autocast(device_type=device.type, dtype=...)``; device_type is taken
        from the runtime tensors so a CPU box still exercises the path (a hardcoded 'cuda' autocast
        is inert on CPU tensors)."""
        if self.cfg.amp_dtype is None:
            return nullcontext()
        # Explicit, defensive mapping (no bare-else silent-fp16 fallthrough). config.py
        # (_require) accepts amp_dtype in (None, 'bf16', 'fp16'), so both 'bf16' and 'fp16' are
        # reachable non-None values; fp16 training is loss-scaled by the GradScaler in train.py
        # (enabled when amp_dtype=='fp16'). Map both and raise on anything else.
        if self.cfg.amp_dtype == "bf16":
            dtype = torch.bfloat16
        elif self.cfg.amp_dtype == "fp16":
            dtype = torch.float16
        else:
            raise ValueError(f"unsupported amp_dtype {self.cfg.amp_dtype!r}")
        return torch.autocast(device_type=device.type, dtype=dtype)

    def _amp_off_context(
        self,
        device: torch.device,            # resolves autocast device_type for the disable wrapper
    ):
        r"""fp32 island for the decode + cross-entropy, else a nullcontext.

        amp_dtype=None (default) returns ``nullcontext()`` -- no autocast object is entered on the
        default path (byte-identity). When AMP is on, returns ``torch.autocast(..., enabled=False)``
        so no FURTHER downcasting happens inside the decode/CE; the actual fp32 guarantee comes from
        ``.float()``-ing the decode/CE inputs at the call site (autocast-disable alone cannot upcast
        a tensor that already arrived bf16)."""
        if self.cfg.amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=device.type, enabled=False)

    def _compact_phi_blocks_enabled(self) -> bool:
        r"""Whether this model is on the canonical route supported by packed phi transport."""
        cfg = self.cfg
        return (
            cfg.transport_mode == "flat"
            and self.group.phi_coordinate_layout == "block_head_row_major"
        )

    def _apply_pos_phi(self, phi: torch.Tensor) -> torch.Tensor:
        r"""Return stored left coordinates for the group-product positional frame."""
        return phi

    def _pos_phi_right(self, phi: torch.Tensor) -> Optional[torch.Tensor]:
        r"""Return positional coordinates Y for the exact frame exp(X) exp(Y), else ``None``."""
        if self.cfg.pos_phi == "none":
            return None
        coords = positional_phi_coords(
            self.cfg.pos_phi, phi.shape[-2], phi.shape[-1],
            scale=self.cfg.pos_phi_scale,
            pos_phi_free=getattr(self, "pos_phi_free", None),
            device=phi.device, dtype=phi.dtype,
        )
        if coords is not None and self.cfg.pos_phi_project_slk:
            coords = project_phi_to_slk(coords, self.group.generators, self.group.irrep_dims)
        return coords

    @staticmethod
    def _canonical_frame_context(
        transport: 'CompactFactoredTransport | FactoredTransport | RopeTransport',
        *,
        fold_rope: bool = False,
        compact:   bool = False,
    ) -> CanonicalFrameContext:
        r"""Read realized vertex factors, optionally retaining packed head blocks for the readout."""
        if isinstance(transport, RopeTransport):
            # Fold inverse pairs at their stored precision, even inside training AMP.
            with torch.autocast(device_type=transport.rope.device.type, enabled=False):
                base = transport.score_operator() if fold_rope else transport.base
        else:
            base = transport
        if isinstance(base, FactoredTransport):
            if not base.same_frame_flat_cocycle:
                raise RuntimeError(
                    "projected canonical content requires a certified same-frame flat cocycle")
            return CanonicalFrameContext(forward=base.exp_phi, inverse=base.exp_neg_phi)
        if isinstance(base, CompactFactoredTransport):
            if not base.same_frame_flat_cocycle:
                raise RuntimeError(
                    "projected canonical content requires a certified same-frame flat cocycle")
            if compact:
                return CanonicalFrameContext(
                    forward=base.exp_blocks, inverse=base.inv_blocks, block_diagonal=True)
            # Expand only the O(B N K^2) vertex factors. Never call to_dense_omega(), whose
            # O(B N^2 K^2) pairwise operator would discard the compact transport's purpose.
            return CanonicalFrameContext(
                forward=CompactBlockElement(base.exp_blocks, base.K).to_dense(),
                inverse=CompactBlockElement(base.inv_blocks, base.K).to_dense(),
            )
        raise RuntimeError(
            "projected canonical content requires FactoredTransport or "
            f"CompactFactoredTransport vertex factors, got {type(base).__name__}")

    def _block_mlp_frame_transport(
        self,
        beliefs: BeliefState,
        rope: Optional[torch.Tensor],
    ) -> 'tuple[Optional[object], Optional[CanonicalFrameContext]]':
        block_mlp_build = self.executable_build.block_mlp
        if not (
            block_mlp_build.enabled
            and self._block_mlp_registration.requires_frame_context
        ):
            return None, None
        transport = self._build_shared_flat_transport(
            beliefs, rope, require_vertex_factors=True)
        if not isinstance(transport, (FactoredTransport, CompactFactoredTransport, RopeTransport)):
            raise RuntimeError(
                "canonical_frame block MLP did not receive factored vertex frames")
        return transport, self._canonical_frame_context(
            transport, fold_rope=True)

    def _build_shared_flat_transport(
        self,
        beliefs: BeliefState,
        rope: Optional[torch.Tensor],
        *,
        require_vertex_factors: bool,
        _content_token_ids: Optional[torch.Tensor] = None,
    ) -> 'torch.Tensor | CompactFactoredTransport | FactoredTransport | RopeTransport':
        r"""Build the one flat transport shared by projected materialization and inference."""
        from vfe3.inference.e_step import build_belief_transport

        transport = build_belief_transport(
            beliefs.phi,
            self.group,
            transport_mode="flat",
            right_phi=beliefs.right_phi,
            clamp_monitor=self.cfg.transport_clamp_monitor,
            transport_mean_per_head=True,
            congruence_cond_escalation=self.cfg.congruence_cond_escalation,
            compact_phi_block_transport=self._compact_phi_blocks_enabled(),
            require_vertex_factors=require_vertex_factors,
            exp_fp64_norm_threshold=self.cfg.exp_fp64_norm_threshold,
            validity_max_norm=self.cfg.transport_chart_max_norm,
            exactness_out=self._transport_status,
            _content_token_ids=_content_token_ids,
            rope=rope,
            rope_on_cov=self.cfg.rope_full_gauge,
            rope_on_value=self.cfg.rope_on_value,
            rope_insertion=self.cfg.rope_insertion,
        )
        return transport

    def _prepare_projected_canonical_content(
        self,
        beliefs: BeliefState,
        rope: Optional[torch.Tensor],
    ) -> 'Tuple[BeliefState, CompactFactoredTransport | FactoredTransport | RopeTransport, CanonicalFrameContext]':
        r"""Materialize canonical table moments through production's authoritative frame."""
        if self.cfg.encode_mode != "canonical_content_projected":
            raise ValueError(
                "projected canonical-content preparation requires "
                "encode_mode='canonical_content_projected'")
        transport = self._build_shared_flat_transport(
            beliefs,
            rope,
            require_vertex_factors=True,
        )
        if not isinstance(transport, (FactoredTransport, CompactFactoredTransport, RopeTransport)):
            raise RuntimeError(
                "projected canonical content did not receive a factored vertex representation")
        canonical_frame = self._canonical_frame_context(transport)
        if (canonical_frame.forward.dtype == torch.float64
                and beliefs.mu.dtype != torch.float64):
            # Escalated float64 frame (audit 2026-09-23 #12): project at float64, keep belief dtypes.
            materialized_mu, materialized_sigma = project_canonical_diagonal(
                beliefs.mu.double(),
                beliefs.sigma.double(),
                canonical_frame.forward,
            )
            materialized_mu = materialized_mu.to(beliefs.mu.dtype)
            materialized_sigma = materialized_sigma.to(beliefs.sigma.dtype)
        else:
            materialized_mu, materialized_sigma = project_canonical_diagonal(
                beliefs.mu,
                beliefs.sigma,
                canonical_frame.forward,
            )
        return (
            beliefs._replace(mu=materialized_mu, sigma=materialized_sigma),
            transport,
            canonical_frame,
        )

    @staticmethod
    def _slice_canonical_frame(
        canonical_frame: Optional[CanonicalFrameContext],
        position_slice: slice,
    ) -> Optional[CanonicalFrameContext]:
        """Slice only token positions while preserving the captured frame-factor graph."""
        if canonical_frame is None:
            return None
        return CanonicalFrameContext(
            forward=canonical_frame.forward[:, position_slice],
            inverse=canonical_frame.inverse[:, position_slice],
            block_diagonal=canonical_frame.block_diagonal,
        )

    def _decode_belief_with_context(
        self,
        mu_q:            torch.Tensor,                           # (B, N, K)
        sigma_q:         torch.Tensor,                           # (B, N, K) or (B, N, K, K)

        *,
        canonical_frame: Optional[CanonicalFrameContext] = None,  # (B, N, ...) realized frame
        decode_last:     bool = False,
    ) -> torch.Tensor:                                            # (B, N, V), or (B, 1, V) when decode_last
        r"""Decode one realized belief through the model-owned frame/context boundary.

        Decoding is positionwise, so ``decode_last`` slices the query, its covariance, and its
        realized frame to the final position BEFORE the vocabulary readout rather than decoding
        all ``N`` positions and discarding ``N - 1`` of them.
        """
        if decode_last:
            last = slice(-1, None)
            mu_q, sigma_q = mu_q[:, last], sigma_q[:, last]
            canonical_frame = self._slice_canonical_frame(canonical_frame, last)
        with self._amp_off_context(mu_q.device):
            logits = self.prior_bank.decode(
                mu_q.float(),
                sigma_q.float(),
                canonical_frame=canonical_frame,
            )
        return logits

    @property
    def transport_state(self) -> TransportState:
        r"""Live trainable state declared by the selected transport registration.

        The mapping is rebuilt from top-level registered parameters on every access. This preserves
        stable checkpoint names and follows parameter replacement performed by ``Module._apply`` when
        the model moves across devices or dtypes.
        """
        return {key: getattr(self, key) for key in self._transport_state_keys}

    def forward_beliefs(
        self,
        token_ids:      torch.Tensor,                    # (B, N) integer token ids

        *,
        return_logits:  bool           = False,          # also decode logits; else logits is None
        decode_last:    bool           = False,          # decode only the last position as (B, 1, V)
        _reuse_content_factors: bool   = False,          # private loss path; public beliefs retain occurrence gradients
        _mean_only_loss: bool          = False,          # private linear-loss path; public beliefs retain full covariance
        capture:        Optional[MStepCapture]       = None,   # out-param: M-step intermediates (q*, final-block prior, raw out)
        estep_grad_out: Optional[EStepGradientOutput] = None,   # diag out-param: E-step belief-grad norms (forwarded)
        _decode_frame_out: Optional[dict]            = None,   # private out-param: the reference decode frame
    ) -> 'Tuple[BeliefState, Optional[torch.Tensor]]':
        r"""Run the belief pipeline and return the converged belief q* (post final_norm), optionally
        with the decoded logits. This is the single belief-production seam used by ``forward`` and,
        through it, the ordinary autoregressive sampler in ``generate``.

        Factors the (previously inline) sequence q_i(0) = p_i = encode(token) -> phi <- pos_phi ->
        (optional s-refine) -> precision-bias fold -> vfe_stack (L blocks of E-step belief descent) ->
        final_norm, i.e. the map from token ids to the converged Gaussian tuple q* = (mu*, Sigma*, phi*).
        The returned ``BeliefState`` carries mu = final_norm(mu_final, sigma_final), sigma = sigma_final,
        and phi = out.phi UNCHANGED (final_norm transforms only the mean), so a caller reads q*.phi for
        the M-step gauge penalty exactly as forward does. ``return_logits`` decodes p(o | q*_i) via
        ``prior_bank.decode`` inside the SAME fp32 island forward's inference branch uses. By default
        the decode covers every position and remains byte-identical to the pre-refactor
        ``forward(targets=None)`` return; ``decode_last=True`` decodes only the final belief position
        and returns logits shaped ``(B, 1, V)``.

        ``capture`` (an out-param dict, non-None only when ``mstep_self_coupling_weight>0``) is filled
        with the pre-transform intermediates the M-step self-coupling term reads and cannot recover
        from the post-final_norm belief: ``capture['converged']`` (the last block's converged
        pre-transform q*) and ``capture['final_block_prior']`` (the live ``(mu_p, sigma_p)`` entering
        that block), both written by ``vfe_stack``. ``capture['prior']`` retains the encode-time prior
        (post s-refine), and ``capture['out']`` retains the raw stack output (pre final_norm).

        Reference coordinates. Under ``block_coordinates='reference'`` the final norm, like the block
        norm and MLP, acts on :math:`\xi_i = U_i^{-1}\mu_i` and is mapped back,
        :math:`\mu \mapsto U_i\,\mathrm{FN}(U_i^{-1}\mu)`; the returned belief stays in local coordinates.
        Under ``decode_coordinates='reference'`` the realized relational frame :math:`U_i` is the decode
        frame: it is stored as ``capture['canonical_frame']`` (the key every decode caller already
        threads into ``_decode_belief_with_context``) and in the private ``_decode_frame_out['frame']``
        for the fused training CE, so the logits read :math:`(\xi_i, C_i)` on every decode path.
        ``per_type_hidden_bias`` gathers its table once here and hands layer ``l`` its width-``w_l``
        slice.

        This seam carries the same internal
        ``run = no_grad() if e_step_gradient=='detach'
        else nullcontext()`` and ``amp`` contexts forward establishes, so a grad-enabled training caller
        and a no-grad inference caller both get the identical forward value. The no-grad property used by
        generation comes from the caller's ``@torch.no_grad`` scope, not from this method.
        """
        if self.conditional_filtering_enabled and torch.is_inference_mode_enabled():
            raise RuntimeError("conditional filtering requires local autograd; use torch.no_grad() instead of inference_mode()")
        B, N = token_ids.shape
        if decode_last and N <= 0:
            raise ValueError("decode_last=True requires a nonempty token context")
        beliefs = self.prior_bank.encode(token_ids)              # (B, N, K) ...
        encoded_phi = beliefs.phi
        beliefs = beliefs._replace(
            phi=self._apply_pos_phi(beliefs.phi),
            right_phi=self._pos_phi_right(beliefs.phi),
        )
        diagnostic_capture = capture.get("diagnostic") if capture is not None else None
        if diagnostic_capture is not None:
            diagnostic_capture["encoded_belief"] = beliefs
        log_prior = self._attention_log_prior(N, token_ids.device)
        rope = self._rope_rotation(N, token_ids.device)

        # The E-step stack is vectorized over the batch (audit 4c): the belief tuple carries a
        # leading B axis through transport / gradients / retraction in one set of kernels, instead
        # of a serial per-sequence Python loop. Sequences are independent (each reads only its own
        # belief and the shared, sequence-independent log_prior), so the batched result equals the
        # per-sample result (pinned by tests/test_perf_equivalence.py::test_batched_forward_equals_per_sample).
        # lambda_beta: the belief-coupling weight (the constant cfg.lambda_beta).
        lambda_beta = self.cfg.lambda_beta
        # The registry-owned mapping is the complete trainable transport state. It is empty for the
        # flat pure path and reaches every registered builder without a mode or state-name branch.
        transport_state = self.transport_state
        # E-step backward estimator: 'detach' wraps the whole E-step in no_grad;
        # 'unroll' (default) and 'straight_through'
        # both run grad-enabled and thread the mode down to e_step_iteration, which only changes the
        # mu/sigma backward (straight_through detaches the per-iteration tangent; unroll keeps the
        # second-order term). Forward VALUE is identical across unroll/straight_through.
        e_step_gradient = self.cfg.e_step_gradient
        run = torch.no_grad() if e_step_gradient == "detach" else nullcontext()
        # Opt-in mixed precision (cfg.amp_dtype): wrap the E-step / belief pipeline in autocast for
        # CUDA throughput. amp_dtype=None (default) -> nullcontext -> NO autocast object is ever
        # instantiated on the default path, so logits/loss are byte-identical to the no-AMP build.
        # device_type is resolved from the runtime tensors (not hardcoded 'cuda') so the path is
        # exercised on whatever device the tokens live on. The outer autocast remains active through
        # transport construction, model-channel refinement, and vfe_stack; only the existing
        # matrix-exp/SPD kernels and the autograd oracle's inner objective/derivative builder enter
        # their own narrow fp32 islands. Decode + CE are protected separately below (their inputs are
        # .float()-ed; see _amp_off_context).
        amp = self._amp_context(token_ids.device)
        with run, amp:
            relational_frame = None
            relational_value_frame = None
            relational_capture = {} if self.conditional_filtering_enabled and capture is not None else None
            if relational_capture is not None:
                capture[self.conditional_capture_key] = relational_capture
            shared_omega = None
            canonical_frame: Optional[CanonicalFrameContext] = None
            block_mlp_frame: Optional[CanonicalFrameContext] = None
            projected_content = self.cfg.encode_mode == "canonical_content_projected"
            share_flat_transport = (
                self.cfg.transport_mode == "flat"
                and self.cfg.e_phi_lr == 0.0
                and rope is None
            )
            if projected_content:
                # q(0) and p are the SAME materialized diagonal-family tensors, produced with the
                # exact transport object vfe_stack receives below.
                beliefs, shared_omega, canonical_frame = (
                    self._prepare_projected_canonical_content(beliefs, rope))
                if capture is not None:
                    capture["canonical_frame"] = canonical_frame
            elif share_flat_transport:
                # Share one fixed flat transport when phi cannot move inside the E-step and no RoPE
                # wrapper is active.
                shared_omega = self._build_shared_flat_transport(
                    beliefs,
                    None,
                    require_vertex_factors=(self.conditional_filtering_enabled),
                    _content_token_ids=(
                        token_ids if (
                            type(self) is VFEModel
                            and type(self.prior_bank) is PriorBank
                            and getattr(self.prior_bank.encode, "__func__", None) is PriorBank.encode
                            and get_encode(self.prior_bank.encode_mode) is _encode_per_token
                            and get_transport_registration("flat").callable is _build_flat
                            and type(token_ids) is torch.Tensor
                            and token_ids.dtype == torch.long
                            and token_ids.device == beliefs.phi.device
                            and self.group.elementary_block_shape is not None
                            and beliefs.phi is encoded_phi
                            and getattr(self._apply_pos_phi, "__func__", None) is VFEModel._apply_pos_phi
                            and getattr(self._pos_phi_right, "__func__", None) is VFEModel._pos_phi_right
                            and _reuse_content_factors
                            and capture is None
                            and not beliefs.phi.retains_grad
                            and not beliefs.phi._backward_hooks
                        ) else None),
                )
            if self.conditional_filtering_enabled:
                if shared_omega is None:
                    shared_omega = self._build_shared_flat_transport(
                        beliefs, rope, require_vertex_factors=True)
                relational_frame = self._canonical_frame_context(shared_omega, fold_rope=True, compact=True)
                relational_value_frame = (
                    self._canonical_frame_context(shared_omega, fold_rope=False, compact=True)
                    if rope is not None and not self.cfg.rope_on_value else relational_frame
                )
                if self.cfg.family == "gaussian_frame_diagonal":
                    base = self._canonical_frame_context(shared_omega, fold_rope=False, compact=True)
                    identity = torch.eye(base.forward.shape[-1], device=base.forward.device,
                                         dtype=base.forward.dtype).expand_as(base.forward)
                    intrinsic = CanonicalFrameContext(identity, identity, base.block_diagonal)
                    with torch.autocast(device_type=base.forward.device.type, enabled=False):
                        if rope is not None and base.forward.dtype == torch.float64:
                            # Escalated float64 frames (audit 2026-09-23 #12): U^-1 (U R^T) = R^T
                            # exactly, so the intrinsic frame is built from the re-orthogonalized
                            # rotation; forming it through U leaves cond(U) eps64 of rounding that
                            # the float64 frame contract rejects. Float32 frames keep the product.
                            rot = _rope_at_frame_precision(rope, torch.float64)   # (N, K, K)
                            if base.block_diagonal:
                                rot = _equal_diag_blocks(rot, base.forward.shape[-3],
                                                         base.forward.shape[-1])  # (N, H, d, d)
                            relational_frame = CanonicalFrameContext(
                                rot.transpose(-1, -2).expand_as(base.forward),
                                rot.expand_as(base.forward), base.block_diagonal)
                        else:
                            relational_frame = (CanonicalFrameContext(
                                base.inverse @ relational_frame.forward,
                                relational_frame.inverse @ base.forward, base.block_diagonal)
                                if rope is not None else intrinsic)
                    relational_value_frame = (relational_frame
                                              if self.cfg.rope_on_value else intrinsic)
            if self.cfg.decode_coordinates == "reference":
                # The covariant decoder reads (xi, C) through the frame the attention reads.
                if relational_frame is None:
                    raise RuntimeError(
                        "decode_coordinates='reference' requires the realized relational frame; this "
                        "forward built none (projected attention route required)")
                canonical_frame = relational_frame
                if capture is not None:
                    capture["canonical_frame"] = canonical_frame
                if _decode_frame_out is not None:
                    _decode_frame_out["frame"] = canonical_frame
            if (
                self.executable_build.block_mlp.enabled
                and self._block_mlp_registration.requires_frame_context
            ):
                if shared_omega is None:
                    shared_omega, block_mlp_frame = self._block_mlp_frame_transport(
                        beliefs, rope)
                else:
                    block_mlp_frame = self._canonical_frame_context(
                        shared_omega, fold_rope=True)
            if diagnostic_capture is not None:
                diagnostic_capture["initial_belief"] = beliefs
                diagnostic_capture["rope"] = rope
                diagnostic_capture["log_prior"] = log_prior
            # capture: the last block's CONVERGED (pre-transform) belief q*, consumed by the
            # M-step self-coupling term in forward (manuscript: the self-term reads q*, not the
            # transformed handoff; audit 2026-06-09 overnight F19). None when the term is off.
            grad_rec: Optional[EStepGradientRecord] = (
                {} if estep_grad_out is not None else None
            )                                                       # E-step belief-grad capture (gated, off by default)
            stack_extra: Dict[str, Any] = {}
            if self.per_type_hidden_bias is not None:
                # One gather per forward, split over the layers: layer l reads E^(l)[x_i], (B, N, w_l).
                table = self.per_type_hidden_bias[token_ids]                    # (B, N, n_gen)
                stack_extra["block_mlp_hidden_biases"] = [
                    part if part.shape[-1] > 0 else None
                    for part in torch.split(table, list(self._per_type_hidden_widths), dim=-1)]
            out = vfe_stack(
                beliefs, beliefs.mu, beliefs.sigma, self.group, self.cfg,
                log_prior=log_prior, block_norm=self.block_norm,
                block_mlps=self.block_mlps,
                block_mlp_frame=block_mlp_frame,
                head_mixer=self.head_mixer, cg_coupling=self.cg_coupling,
                lambda_beta=lambda_beta,
                transport_state=transport_state,
                e_step_gradient=e_step_gradient,
                rope=rope, rope_on_cov=self.cfg.rope_full_gauge,
                rope_on_value=self.cfg.rope_on_value, rope_insertion=self.cfg.rope_insertion,
                relational_operators=self.relational_operators,
                relational_operators_by_layer=self.relational_operators_by_layer or None,
                relational_frame=relational_frame,
                relational_value_frame=relational_value_frame,
                relational_capture=relational_capture,
                _need_final_covariance=not (
                    _mean_only_loss and capture is None and not return_logits
                    and type(self) is VFEModel and type(self.prior_bank) is PriorBank
                    and self.final_norm is None and self.cfg.decode_mode == "linear"
                    and get_decode_registration("linear").supports_chunked
                    and get_decode_registration("linear").fused_ce is PriorBank.decode_ce_linear_chunked
                    and getattr(self.forward_beliefs, "__func__", None) is VFEModel.forward_beliefs
                ),
                capture=capture, grad_record=grad_rec,
                transport_status=self._transport_status,
                prebuilt_transport=shared_omega,
                **stack_extra)
        if estep_grad_out is not None:                           # real tensor norms read only when requested
            for _gk in ("mu", "sigma", "phi"):
                _gv = grad_rec.get(_gk, 0.0) if grad_rec is not None else 0.0
                estep_grad_out[_gk] = float(_gv) if _gv is not None else float("nan")
        mu_final = out.mu                                        # (B, N, K); head mixer (if any) applied PER BLOCK
        sigma_final = out.sigma                                  # inside vfe_stack now, not post-stack

        if self.final_norm is not None:                          # config-selected final norm (cached)
            if self.cfg.block_coordinates == "reference":
                # Reference coordinates: mu -> U_i FN(U_i^{-1} mu), the same conjugation as the block.
                if relational_frame is None:
                    raise RuntimeError(
                        "block_coordinates='reference' requires the realized relational frame for the "
                        "final norm; this forward built none")
                bd = relational_frame.block_diagonal
                xi_final = frame_matvec(relational_frame.inverse, mu_final, bd)    # (B, N, K)
                mu_final = frame_matvec(relational_frame.forward,
                                        self.final_norm(xi_final, sigma_final), bd)
            else:
                mu_final = self.final_norm(mu_final, sigma_final)

        belief = BeliefState(mu=mu_final, sigma=sigma_final, phi=out.phi,
                             right_phi=out.right_phi)   # carry the exact positional right factor
        # TODO(frame-conjugated-head-mixer): thread the realized query frame through this boundary
        # before implementing M_i = U_i (A kron I) U_i^{-1}; do not co-transform q and every
        # vocabulary prior by the same M, because that is a KL-invariant no-op.
        if capture is not None:
            # M-step out-param enrichment: vfe_stack already wrote capture['converged'] (q*); add the
            # encode-time prior p (post s-refine) and the raw pre-final_norm stack output for callers
            # that need the complete pre-normalization pipeline state.
            capture["prior"] = beliefs
            capture["out"]   = out
        logits = None
        if return_logits:
            logits = self._decode_belief_with_context(
                mu_final,
                sigma_final,
                canonical_frame=canonical_frame,
                decode_last=decode_last,
            )
        return belief, logits


    def forward(
        self,
        token_ids: torch.Tensor,         # (B, N) integer token ids
        targets:   Optional[torch.Tensor] = None,   # (B, N) next-token ids (-100 = ignore)

        *,
        estep_grad_out: Optional[EStepGradientOutput] = None,   # diag out-param: filled with the E-step belief-grad norms
        return_decode_stats: bool = False,
    ) -> 'torch.Tensor | Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor | DecodeCEResult]':
        r"""Forward pass; returns logits, or (logits, loss, ce) when targets are given.

        On the fused-chunked training path logits is None (callers discard it there), hence the
        Optional first element of the training tuple. When ``estep_grad_out`` (a dict) is passed, it
        is filled with the LAST-block / LAST-iteration raw E-step belief-gradient norms
        ``{'mu','sigma','phi'}`` (||grad_mu/sigma/phi|| of F over the belief tuple) -- the E-step
        analogue of the M-step per-group grad norms; default None is zero-overhead and byte-identical.
        ``return_decode_stats=True`` replaces only the third training-tuple item with an immutable
        :class:`DecodeCEResult`; the default three-tuple remains unchanged.

        Belief production is factored into :meth:`forward_beliefs` (the shared seam); this method is the
        decode + cross-entropy + M-step assembly on top of it."""
        if targets is None:
            # Inference: logits via the shared belief seam. estep_grad_out is forwarded so the
            # diagnostic still fills on this path (byte-identical to the pre-refactor return).
            return self.forward_beliefs(
                token_ids,
                return_logits=True,
                estep_grad_out=estep_grad_out,
            )[1]
        # Training path: produce the converged belief q* (no (B,N,V) logits) via the shared seam, then
        # run the existing decode + cross-entropy + M-step assembly reading belief.mu / sigma / phi.
        # Capture regularizer state and, in projected mode, the exact decode frame. Regularizer
        # captures also retain the converged q*, live final-block prior, encode-time prior, and raw
        # pre-final_norm stack output.
        cap: Optional[MStepCapture] = (
            {} if (self.cfg.mstep_self_coupling_weight > 0.0
                   or self.cfg.cg_energy_weight > 0.0
                   or self.cfg.encode_mode == "canonical_content_projected") else None
        )
        if cap is not None and self.cfg.cg_energy_weight > 0.0:
            # Initialize the CG moment-energy lists ONLY when the regularizer is on. A capture
            # allocated solely for M-step self-coupling leaves these keys absent, so vfe_block never
            # appends to (or stacks) a CG list (PB-13).
            cap["cg_moment_energy_rows"] = []
            cap["cg_pre_moments"] = []
        # decode_coordinates='reference' receives the decode frame through a private out-param, so the
        # captureless fast paths (content-factor reuse, mean-only final layer) stay available.
        decode_frame_out: Optional[dict] = (
            {} if self.cfg.decode_coordinates == "reference" else None)
        belief, _ = self.forward_beliefs(
            token_ids,
            return_logits=False,
            capture=cap,
            estep_grad_out=estep_grad_out,
            _reuse_content_factors=True,
            _mean_only_loss=True,
            **({} if decode_frame_out is None else {"_decode_frame_out": decode_frame_out}),
        )
        mu_final, sigma_final = belief.mu, belief.sigma          # (B, N, K) post final_norm; sigma = out.sigma
        canonical_frame = (
            cap.get("canonical_frame") if cap is not None else None
        )
        if decode_frame_out is not None:
            canonical_frame = decode_frame_out["frame"]          # realized U_i: the decoder reads (xi, C)

        # Decode + cross-entropy fp32 island. The decode matmul (_decode_diagonal) reconstructs the
        # Mahalanobis term via a catastrophically-cancelling subtraction pinned at atol-1e-3, and CE
        # is a log-sum-exp over V=50257; both MUST stay fp32 even when amp_dtype is on. The
        # load-bearing guard is the explicit .float() on the inputs (autocast(enabled=False) only
        # blocks FURTHER downcasting -- it does NOT upcast a tensor that already arrived bf16 from
        # the autocast E-step), mirroring retraction.py's in-island sigma.float(). On the default
        # fp32 path .float() is a value-identical no-op AND the island is a nullcontext (see
        # _amp_off_context), so this block is byte-identical to the no-AMP build.
        # Select the same decoder record PriorBank.decode uses. Capability routing and fused
        # dispatch come from this one registration. logits is None on the fused branch by design.
        active_decode_mode = self.cfg.decode_mode
        decode_registration = get_decode_registration(active_decode_mode)
        fused_chunked = (
            targets is not None
            and decode_registration.supports_chunked
        )
        scored_tokens = None
        excluded_tokens = None
        if fused_chunked:
            with self._amp_off_context(token_ids.device):
                fused_kwargs = {}
                if return_decode_stats and decode_registration.fused_ce_supports_stats:
                    fused_kwargs["return_stats"] = True
                if active_decode_mode != "linear":
                    if canonical_frame is None:
                        fused_result = decode_registration.fused_ce(
                            self.prior_bank, mu_final.float(), sigma_final.float(), targets,
                            **fused_kwargs,
                        )
                    else:
                        fused_result = decode_registration.fused_ce(
                            self.prior_bank, mu_final.float(), sigma_final.float(), targets,
                            canonical_frame=canonical_frame,
                            **fused_kwargs,
                        )
                elif canonical_frame is None:
                    fused_result = decode_registration.fused_ce(
                        self.prior_bank, mu_final.float(), targets,
                        **fused_kwargs,
                    )
                else:                                            # decode_coordinates='reference'
                    fused_result = decode_registration.fused_ce(
                        self.prior_bank, mu_final.float(), targets,
                        canonical_frame=canonical_frame,
                        **fused_kwargs,
                    )
                if return_decode_stats and decode_registration.fused_ce_supports_stats:
                    if not isinstance(fused_result, DecodeCEResult):
                        raise TypeError(
                            f"decode mode {active_decode_mode!r} declares fused CE stats support "
                            f"but returned {type(fused_result).__name__}"
                        )
                    ce = fused_result.ce
                    excluded_tokens = fused_result.excluded_tokens
                    scored_tokens = fused_result.scored_tokens
                else:
                    ce = fused_result
                    if return_decode_stats:
                        # Legacy custom fused decoders have no exclusion channel beyond ignore_index.
                        scored_tokens = (targets != -100).sum(dtype=torch.int64)
                        excluded_tokens = scored_tokens.new_tensor(targets.numel()) - scored_tokens
            logits = None                                        # no (B, N, V) tensor on the fused path
        else:
            logits = self._decode_belief_with_context(
                mu_final,
                sigma_final,
                canonical_frame=canonical_frame,
            )                                                        # (B, N, V) fp32
            # targets is guaranteed not None here (the inference path returned via forward_beliefs above).
            with self._amp_off_context(token_ids.device):
                # A position whose Sigma_q is not PD has no valid likelihood and is EXCLUDED, exactly
                # as the fused chunked kernels exclude it (audit 2026-08-06 F31). The dense branch
                # cannot see that from the logits alone -- the decode emits an informationless
                # uniform row there -- so it asks the bank and marks the position ignore_index, which
                # drops it from both F.cross_entropy's sum and the n_valid denominator below. None on
                # every diagonal decode, where there is no factorization to fail.
                ce_targets = targets
                if active_decode_mode != "linear":
                    degenerate = self.prior_bank.decode_degenerate_positions(
                        sigma_final.float(),
                        canonical_frame=canonical_frame,
                    )
                    if degenerate is not None:
                        ce_targets = torch.where(
                            degenerate, targets.new_full((), -100), targets)
                flat_logits = logits.reshape(-1, self.cfg.vocab_size).float()
                flat_targets = ce_targets.reshape(-1)
                # Branchless masked mean (no host sync to test .any()): sum-reduced CE over the
                # non-ignored tokens divided by a device-side clamped count. An all-ignore microbatch
                # gives 0/1 = a finite grad-connected 0; F.cross_entropy's default mean would be
                # 0/0 = NaN there, poisoning logging / NaN-guards / grad-accum means.
                scored_tokens = (flat_targets != -100).sum(dtype=torch.int64)
                excluded_tokens = scored_tokens.new_tensor(flat_targets.numel()) - scored_tokens
                n_valid = scored_tokens.clamp_min(1)
                ce = F.cross_entropy(flat_logits, flat_targets, ignore_index=-100,
                                     reduction="sum") / n_valid
        loss = ce
        if self.cfg.mass_phi > 0.0:
            # M-step gauge-frame penalty (manuscript Algorithm 1 M-step loss): regularizes the
            # CONVERGED output phi -> backprops to the learned prior table phi_embed. This is the
            # outer-loss role; mass_phi ALSO enters the inner phi E-step objective (e_step:
            # phi_alignment_loss), shaping the inference trajectory. Both roles are in the
            # manuscript algorithm (E-step phi gradient and M-step loss both carry alpha_phi/2||phi||^2).
            # D-08: this used (phi ** 2).mean(), which differs from the E-step's (phi ** 2).sum()
            # at e_step.phi_alignment_loss by phi.numel() (10^3-10^5 here), so ONE mass_phi named
            # two different penalty strengths in the two places the manuscript gives one alpha_phi.
            # Both now carry the per-position squared norm ||phi_i||^2 = (phi ** 2).sum(dim=-1);
            # the E-step sums it over the sequence to match its per-sequence F, and the M-step
            # averages it over positions to match the per-token mean that `ce` already is. Latent
            # at the 0.0 default (no shipped config changes), material on a mass_phi sweep.
            loss = loss + 0.5 * self.cfg.mass_phi * (belief.phi ** 2).sum(dim=-1).mean()
        if self.cfg.mstep_self_coupling_weight > 0.0:
            # M-step self-coupling regularizer (manuscript Algorithm 1, GL(K)_attention.tex:2111):
            # L += alpha_hat * sum_i alpha_i D(q_i*||p_i), the alpha-weighted self-coupling of the
            # CONVERGED variational belief q* (captured by vfe_block BEFORE head_mixer /
            # cg_coupling / block_norm -- the belief the E-step's F was actually minimized over,
            # which the manuscript pins the self-term to) against the live prior captured immediately
            # before the final block. With the three transform toggles
            # at their defaults q* IS the returned `out` (same object), so the pure path is
            # unchanged; under the toggles the term now reads q* rather than the transformed
            # handoff T(q*) it accidentally read before (audit 2026-06-09 overnight F19,
            # challenge-upheld; restores the documented intent and E-step/M-step consistency).
            # alpha_i is the SAME registered self-coupling form as the E-step / diagnostics
            # (self_coupling_alpha keyed off cfg.lambda_alpha_mode), so under state_dependent_per_coord the
            # term carries the per-token, per-coordinate alpha_i^(k)* = c0/(b0+D^(k)) rather than a
            # flat scalar; cfg.mstep_self_coupling_weight (= alpha_hat) is the overall scale. alpha_i
            # is DETACHED: by the alpha-envelope (alpha* is the stationary point of alpha*D + R(alpha),
            # so d/dalpha[alpha*D + R] = 0 there), the M-step gradient of the F self-term w.r.t. the
            # prior is alpha_i* dD/dtheta with alpha_i* held fixed -- detaching it (and dropping R)
            # is exact for the closed-form forms (constant/state_dependent/state_dependent_per_coord).
            # At constant alpha=1.0 (the default) alpha_i==1, byte-identical to the prior mean-D form.
            # Grad-connected through D (no detach on D), so it backprops to the learned prior tables,
            # like mass_phi. Capturing the final-block input preserves the exact stack recurrence at
            # every depth; at n_layers=1 it is the encode-time prior, preserving the existing path.
            from vfe3.families import get_family
            from vfe3.free_energy import self_divergence_for_alpha
            from vfe3.alpha_i import self_coupling_alpha, alpha_is_per_coord
            cfg = self.cfg
            mu_p, sigma_p = cap["final_block_prior"]            # live prior entering the final block
            fam = get_family(cfg.family)
            q_conv = cap["converged"]                           # q*: pre-transform converged belief
            self_div = self_divergence_for_alpha(               # (B, N) summed, or (B, N, K) per-coord
                self._family_instance(fam, q_conv.mu, q_conv.sigma),
                self._family_instance(fam, mu_p, sigma_p),
                alpha=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
                divergence_family=cfg.divergence_family, lambda_alpha_mode=cfg.lambda_alpha_mode,
            )
            alpha_sc, _ = self_coupling_alpha(                  # SAME form as the E-step / diagnostics
                self_div, mode=cfg.lambda_alpha_mode, value=cfg.lambda_alpha, b0=_as_coeff(cfg.b0, cap["out"].mu.device), c0=_as_coeff(cfg.c0, cap["out"].mu.device),
            )
            coupling = alpha_sc.detach() * self_div            # alpha_i^(k)* D^(k) (envelope: alpha* fixed)
            if alpha_is_per_coord(cfg.lambda_alpha_mode):
                coupling = coupling.sum(dim=-1)                # sum_k alpha^(k) D^(k) -> per-token
            sc = coupling.mean()                               # mean over batch and tokens (B, N)
            loss = loss + cfg.mstep_self_coupling_weight * sc
        if self.cfg.cg_energy_weight > 0.0:
            # CG MOMENT-ENERGY REGULARIZER (PB-13): a q-only term added to the outer objective ONCE,
            # NEVER routed through hierarchical_free_energy_terms and never reweighting the canonical
            # q/p/s/h total. Under an attached E-step estimator each block already appended its
            # grad-connected D(q_post||q_pre) rows; under the 'detach' estimator the blocks stashed the
            # detached pre-CG moments, and the shared CGCoupling.forward_moments is re-evaluated here
            # under torch.enable_grad from each fixed pair -- detached from belief inference but
            # attached to path_weights. The token-then-layer mean (each row.mean() over tokens, then
            # mean over layers) is weighted and added; the per-layer values and the assembled
            # objective_total_with_cg are reported on the diagnostic side channel only.
            from vfe3.model.cg_coupling import cg_moment_energy_rows
            cfg = self.cfg
            if cfg.e_step_gradient == "detach":
                per_layer_rows = []
                with torch.enable_grad():
                    for pre_mu, pre_sigma in cap["cg_pre_moments"]:
                        res = self.cg_coupling.forward_moments(pre_mu, pre_sigma)
                        per_layer_rows.append(cg_moment_energy_rows(
                            pre_mu, pre_sigma, res.mu, res.sigma,
                            renyi_order=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
                            family=cfg.family, divergence_family=cfg.divergence_family,
                            full_cov_kl_precision=self.full_cov_kl_precision))
            else:
                per_layer_rows = cap["cg_moment_energy_rows"]
            if (len(per_layer_rows) != cfg.n_layers
                    or any(r.numel() == 0 for r in per_layer_rows)):
                raise RuntimeError(
                    f"cg_energy_weight>0 requires exactly n_layers={cfg.n_layers} nonempty CG "
                    f"moment-energy row tensors; got {len(per_layer_rows)} "
                    f"({[tuple(r.shape) for r in per_layer_rows]}). An empty capture must never "
                    f"reach torch.stack."
                )
            layer_means = torch.stack([rows.mean() for rows in per_layer_rows])   # (n_layers,)
            cg_moment_energy = layer_means.mean()
            loss = loss + cfg.cg_energy_weight * cg_moment_energy
            self._cg_energy_diagnostics = {
                "cg_moment_energy":        cg_moment_energy.detach(),
                "cg_moment_energy_layers": layer_means.detach(),
                "objective_total_with_cg": loss.detach(),
            }
        # TODO(canonical-vfe-phi-mstep): `loss` is the outer supervised objective (CE plus
        # enabled outer regularizers), so the pullback/group phi step consumes its covector.
        # Add a separately selected fixed-returned-state VFE frame objective, declare whether
        # beta/gamma are frozen or envelope-eliminated, and keep the optimizer objective-agnostic.
        ce_detached = ce.detach()
        if return_decode_stats:
            if scored_tokens is None or excluded_tokens is None:
                raise RuntimeError("decoder stats requested without scored/excluded-token counts")
            return logits, loss, DecodeCEResult(
                ce_detached, scored_tokens, excluded_tokens)
        return logits, loss, ce_detached

    @torch.no_grad()
    def generate(
        self,
        token_ids:      torch.Tensor,        # (B, N0) prompt token ids

        max_new_tokens: int,

        *,
        temperature:    float           = 1.0,   # >0; applied to logits before sampling; ignored if greedy
        greedy:         bool            = False, # True -> argmax; ignores temperature/top_k/top_p
        top_k:          Optional[int]   = None,  # keep the k largest-logit tokens, -inf the rest
        top_p:          Optional[float] = None,  # nucleus: smallest set with softmax cumsum >= p
    ) -> torch.Tensor:                       # (B, N0 + max_new_tokens) prompt followed by generated ids
        r"""Autoregressively extend each prompt by ``max_new_tokens`` tokens.

        Each step feeds the running sequence through :meth:`forward_beliefs` and decodes only the
        last position as logits shaped ``(B, 1, V)``. The sampler truncates its context to the last
        ``cfg.max_seq_len`` tokens. The returned sequence keeps the full prompt followed by the
        generated ids. Because generation never enters the
        training/loss branch, it cannot corrupt training (runs under ``torch.no_grad``).

        Greedy (``greedy=True``) takes the argmax and ignores ``temperature``/``top_k``/
        ``top_p``. Otherwise the logits are divided by ``temperature``, then ``top_k``
        (keep the k largest, ``-inf`` the rest), then ``top_p`` (nucleus: smallest set
        whose softmax cumsum reaches ``p``, ``-inf`` the rest, always keeping the top
        token), then softmaxed and sampled with :func:`torch.multinomial`.

        This is the correct-but-slow first version: it re-runs the FULL forward (encode
        -> E-step -> decode) for every generated token. Incremental belief reuse across
        steps is a future optimization.
        """
        # audit C13 (2026-07-01): validate the sampler arguments up front. A negative max_new_tokens
        # would silently no-op (empty loop, prompt returned unchanged); temperature<=0, out-of-range
        # top_k, and top_p outside (0, 1] fail late or produce invalid probabilities. Greedy ignores
        # temperature/top_k/top_p, so those three are checked only on the sampled path.
        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")
        if token_ids.shape[1] <= 0:
            raise ValueError("generate requires a nonempty token context")
        if not greedy:
            if not (temperature > 0.0):
                raise ValueError(f"temperature must be > 0, got {temperature}")
            if top_k is not None and not (1 <= top_k <= self.cfg.vocab_size):
                raise ValueError(f"top_k must be in [1, vocab_size={self.cfg.vocab_size}], got {top_k}")
            if top_p is not None and not (0.0 < top_p <= 1.0):
                raise ValueError(f"top_p must be in (0, 1], got {top_p}")
        # audit F10 (2026-07-01), warn-only (mirrors the D3 link-mode memory estimator, f3387b9):
        # generate() has NO incremental belief/KV cache -- every generated token re-runs the FULL
        # forward (encode -> E-step -> decode) over the whole <=max_seq_len window. Estimate the
        # dominant fp32 per-forward transients at the max_seq_len bound -- the (B, H, N, N)
        # attention/KL maps plus the (B, N, V) logits and (B, N, K) beliefs -- and warn ONCE past
        # the documented 2 GiB budget; never raise (the pure path stays runnable). Incremental
        # belief reuse across steps is the deferred optimization (see the docstring above).
        _B, _N, _K = token_ids.shape[0], self.cfg.max_seq_len, self.cfg.embed_dim
        _est_bytes = 4 * _B * (self.cfg.n_heads * _N * _N + _N * self.cfg.vocab_size + _N * _K)
        if _est_bytes > 2 * 1024 ** 3:                           # documented budget: 2 GiB per forward
            import warnings
            warnings.warn(
                f"generate(): estimated per-forward peak ~{_est_bytes / 2 ** 30:.1f} GiB (B={_B}, "
                f"N={_N}, heads={self.cfg.n_heads}, V={self.cfg.vocab_size}) exceeds the 2 GiB "
                "budget, and generate() re-runs the FULL forward (encode -> E-step -> decode) for "
                "EVERY generated token -- there is no incremental belief/KV cache yet (the "
                "incremental-cache optimization is deferred; see the generate docstring). Expect "
                "O(max_new_tokens * forward(max_seq_len)) time and this peak per step; reduce the "
                "batch/context or generate fewer tokens.",
                UserWarning,
                stacklevel=2,
            )
        seq = token_ids
        with self.prior_bank.fixed_decode_bank():
            for _ in range(max_new_tokens):
                context = seq[:, -self.cfg.max_seq_len:]                 # (B, <=max_seq_len)
                _belief, decoded = self.forward_beliefs(
                    context, return_logits=True, decode_last=True)
                logits = decoded[:, 0, :]                                # (B, V) last position
                invalid_row = torch.isnan(logits).any(dim=-1) | torch.isposinf(logits).any(dim=-1)
                finite_row = torch.isfinite(logits).any(dim=-1)
                if greedy:
                    row_errors = torch.stack((invalid_row, ~finite_row))
                else:
                    logits = logits / temperature
                    retained = torch.ones_like(logits, dtype=torch.bool)
                    if top_k is not None:
                        kth = logits.topk(top_k, dim=-1).values[:, -1:]  # (B, 1) k-th largest
                        remove_topk = logits < kth
                        retained = ~remove_topk
                        logits = logits.masked_fill(remove_topk, float("-inf"))
                    retained_invalid = (retained & ~torch.isfinite(logits)).any(dim=-1)
                    row_errors = torch.stack((invalid_row, ~finite_row, retained_invalid))
                # One device-to-host decision keeps the original diagnostic priority. Validate
                # retained sampled logits before nucleus filtering can remove a non-finite entry.
                rejected = row_errors.any(dim=-1).tolist()
                if rejected[0]:
                    rows = invalid_row.nonzero(as_tuple=False).flatten().tolist()
                    raise ValueError(
                        f"generation logits contain NaN or +inf values in rows {rows}")
                if rejected[1]:
                    rows = (~finite_row).nonzero(as_tuple=False).flatten().tolist()
                    raise ValueError(f"generation logits have no finite value in rows {rows}")
                if not greedy and rejected[2]:
                    rows = retained_invalid.nonzero(as_tuple=False).flatten().tolist()
                    raise ValueError(
                        f"generation retained logits contain non-finite values in rows {rows}")
                if greedy:
                    # A row with a finite value and no NaN/+inf has a finite argmax.
                    next_token = logits.argmax(dim=-1, keepdim=True)    # (B, 1)
                else:
                    if top_p is not None:
                        sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
                        sorted_probs = sorted_logits.softmax(dim=-1)       # compute the softmax once
                        cumprobs = sorted_probs.cumsum(dim=-1)
                        # Keep the smallest nucleus whose cumprob reaches top_p; the strict
                        # shift always keeps the top token (its cumprob>=p never removes it).
                        remove = cumprobs - sorted_probs >= top_p
                        remove_unsorted = remove.scatter(-1, sorted_idx, remove)
                        logits = logits.masked_fill(remove_unsorted, float("-inf"))
                    # Nucleus filtering only removes entries from the already validated set.
                    probs = logits.softmax(dim=-1)                      # (B, V)
                    next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
                seq = torch.cat([seq, next_token], dim=-1)
        return seq


    def _first_sequence_log_prior(
        self,
        log_prior: Optional[torch.Tensor],
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        r"""Drop a captured batch axis while preserving the optional attention-head axis."""
        if log_prior is None:
            return None
        unbatched_rank = 2 if len(self.group.irrep_dims) == 1 else 3
        if log_prior.dim() == unbatched_rank + 1 and log_prior.shape[0] == batch_size:
            return log_prior[0]
        return log_prior

    def _validate_diagnostic_snapshot(
        self,
        token_ids: torch.Tensor,
        snapshot: DiagnosticSnapshot,
    ) -> DiagnosticSnapshot:
        if snapshot.owner is not self:
            raise ValueError("diagnostic snapshot belongs to a different model instance")
        if not torch.equal(snapshot.token_ids, token_ids):
            raise ValueError("diagnostic snapshot token_ids do not match this request")
        return snapshot

    @torch.no_grad()
    def _diagnostic_transport(
        self,
        belief: BeliefState,
    ) -> 'torch.Tensor | CompactFactoredTransport | FactoredTransport':
        r"""Build the active diagnostic transport without materializing compact block pairs."""
        from vfe3.inference.e_step import _transport, build_belief_transport

        cfg = self.cfg
        registration = get_transport_registration(cfg.transport_mode)
        kwargs = {
            "transport_mode": cfg.transport_mode,
            "mu": (belief.mu if registration.needs_mu else None),
            "sigma": (belief.sigma if registration.needs_sigma else None),
            "transport_state": self.transport_state,
            "clamp_monitor": cfg.transport_clamp_monitor,
            "validity_max_norm": cfg.transport_chart_max_norm,
            "exactness_out": self._transport_status,
            "right_phi": belief.right_phi,
        }
        if self._compact_phi_blocks_enabled():
            return build_belief_transport(
                belief.phi,
                self.group,
                compact_phi_block_transport=True,
                transport_mean_per_head=True,
                congruence_cond_escalation=cfg.congruence_cond_escalation,
                exp_fp64_norm_threshold=cfg.exp_fp64_norm_threshold,
                **kwargs,
            )
        return _transport(belief.phi, self.group, **kwargs)

    @torch.no_grad()
    def _attention_map_for_belief(
        self,
        belief:   BeliefState,
        log_prior: Optional[torch.Tensor],
        rope:      Optional[torch.Tensor],
    ) -> torch.Tensor:                         # (H, N, N)
        r"""Compute beta from an already captured block output, without replaying inference."""
        if self.conditional_filtering_enabled:
            raise ValueError("conditional attention must be read from the actual forward snapshot")
        from vfe3.families.base import get_family
        from vfe3.free_energy import attention_weights, attention_tau

        cfg = self.cfg
        fam = get_family(cfg.family)
        omega = self._diagnostic_transport(belief)
        if rope is not None:
            effective = RopeTransport(
                base=omega, rope=rope, on_cov=cfg.rope_full_gauge,
                on_value=cfg.rope_on_value, insertion=cfg.rope_insertion,
                same_frame_flat_cocycle=getattr(
                    omega, "same_frame_flat_cocycle", False))
            mu, sigma = belief.mu, belief.sigma
            batched = False
        else:
            effective = omega.unsqueeze(0)
            mu, sigma = belief.mu.unsqueeze(0), belief.sigma.unsqueeze(0)
            batched = True
        energy = self._coupling_energy(fam,
            mu, sigma, mu, sigma, effective,
            log_prior=log_prior,
            alpha=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
            divergence_family=cfg.divergence_family, irrep_dims=self.group.irrep_dims,
        )
        if batched:
            energy = energy[0]
        base_tau = attention_tau(
            _as_coeff(cfg.kappa_beta, belief.mu.device), self._belief_score_irrep_dims())
        beta = attention_weights(energy, tau=base_tau, log_prior=log_prior)
        return beta.unsqueeze(0) if beta.dim() == 2 else beta

    @torch.no_grad()
    @_block_mlp_diagnostic_eval
    def build_diagnostic_snapshot(
        self,
        token_ids: torch.Tensor,                 # (B, N) integer token ids
        *,
        _share_frame_copies: bool = False,        # private read-only validation consumers only
    ) -> DiagnosticSnapshot:
        r"""Capture one detached evaluation snapshot from exactly one belief forward.

        The capture path records the encoded and stack-entry beliefs, every block's live prior,
        pre-transform converged belief and post-transform output, plus the first block's actual
        E-step states and free-energy values. Beta/gamma maps are then reduced from those captured
        tensors; no encode, E-step, block, or stack replay is performed.
        Public calls own independent Tensor fields. The private ``_share_frame_copies``
        contract permits phi/right_phi aliases only for internal read-only consumers.
        """
        frame_cache: Optional[_FrameFreezeCache] = {} if _share_frame_copies else None
        diagnostic: dict = {}
        capture = {"diagnostic": diagnostic}
        final_belief, logits = self.forward_beliefs(
            token_ids, return_logits=True, capture=capture)
        if logits is None:
            raise RuntimeError("diagnostic snapshot requested logits but the decoder returned None")

        log_prior = diagnostic["log_prior"]
        log_prior0 = self._first_sequence_log_prior(log_prior, token_ids.shape[0])
        # Autocast can leave captured head-mixer outputs in bf16/fp16. Diagnostic transport is
        # intentionally rebuilt outside autocast so matrix factorizations and condition-number
        # probes remain fp32. Freeze and upcast the captured boundary before any replay contraction;
        # otherwise compact transport combines fp32 frame factors with a low-precision belief.
        layer_outputs = tuple(_freeze_belief(layer, _frame_cache=frame_cache) for layer in diagnostic["layer_outputs"])
        stack_output = _freeze_belief(capture["out"], _frame_cache=frame_cache)
        relational_capture = capture.get(self.conditional_capture_key)
        if relational_capture is not None:
            layer_betas = [cap["beta_pre"][0] for cap in diagnostic["layer_conditional_captures"]]
            beta_maps = torch.stack([beta if beta.dim() == 3 else beta.unsqueeze(0)
                                     for beta in layer_betas])
        else:
            beta_maps = torch.stack([
                self._attention_map_for_belief(_sequence_belief(layer), log_prior0, diagnostic["rope"])
                for layer in layer_outputs
            ])
        trace = diagnostic["e_step_trace"]
        trace_free_energy = (torch.stack([
            value.reshape(()) for value in trace["free_energy"]])
            if trace["free_energy"] else final_belief.mu.new_empty(0))

        return DiagnosticSnapshot(
            owner=self,
            token_ids=_freeze_tensor(token_ids),
            encoded_belief=_freeze_belief(diagnostic["encoded_belief"], _frame_cache=frame_cache),
            initial_belief=_freeze_belief(diagnostic["initial_belief"], _frame_cache=frame_cache),
            layer_priors=tuple(
                (_freeze_tensor(mu_p), _freeze_tensor(sigma_p))
                for mu_p, sigma_p in diagnostic["layer_priors"]),
            layer_converged=tuple(
                _freeze_belief(belief, _frame_cache=frame_cache) for belief in diagnostic["layer_converged"]),
            layer_outputs=layer_outputs,
            layer_block_mlp_inputs=tuple(
                _freeze_belief(layer, _frame_cache=frame_cache) for layer in diagnostic["layer_block_mlp_inputs"]),
            layer_block_mlp_outputs=tuple(
                _freeze_belief(layer, _frame_cache=frame_cache) for layer in diagnostic["layer_block_mlp_outputs"]),
            stack_output=stack_output,
            final_belief=_freeze_belief(final_belief, _frame_cache=frame_cache),
            logits=_freeze_tensor(logits),
            beta_maps=_freeze_tensor(beta_maps),
            trace_states=tuple(_freeze_belief(belief, _frame_cache=frame_cache) for belief in trace["beliefs"]),
            trace_free_energy=_freeze_tensor(trace_free_energy),
            **{self.conditional_capture_key: (None if relational_capture is None else {
                key: _freeze_tensor(value) if isinstance(value, torch.Tensor) else value
                for key, value in relational_capture.items()})},
            layer_conditional_captures=tuple({
                key: _freeze_tensor(value) if isinstance(value, torch.Tensor) else value
                for key, value in cap.items()
            } for cap in diagnostic.get("layer_conditional_captures", ())),
            rope=_freeze_tensor(diagnostic["rope"]),
            log_prior=_freeze_tensor(log_prior),
        )

    def _relational_diagnostics(self, snapshot: DiagnosticSnapshot, layer_index: int = -1) -> dict:
        """Report the actual relational router and fixed-context cost; legacy global-F fields are unavailable."""
        cap = (snapshot.layer_conditional_captures[layer_index]
               if snapshot.layer_conditional_captures else snapshot.conditional_capture)
        if cap is None:
            raise ValueError("relational diagnostic snapshot is missing its forward capture")
        beta, score = cap["beta_pre"], cap["score_energy_pre"]
        value_energy = cap.get("value_energy_pre")
        entropy = -torch.special.xlogy(beta, beta).sum(-1)
        scorable = (beta > 0).sum(-1) >= 2
        d = {key: float("nan") for key in (
            "self_coupling", "self_divergence", "belief_coupling", "attention_entropy",
            "total", "effective_rank", "holonomy_deviation", "gauge_trace_spread")}
        d.update(
            relational_semantics=self.conditional_semantics,
            relational_attention_capture="pre_update",
            relational_legacy_global_f_available=0.0,
            attn_entropy=float(entropy.mean()),
            attn_entropy_min=(float(entropy[scorable].min()) if scorable.any() else float("nan")),
            relational_score_klmax_frac=float((score >= self.cfg.kl_max).float().mean()),
            relational_value_klmax_frac=(float((value_energy >= self.cfg.kl_max).float().mean())
                                          if value_energy is not None else float("nan")),
        )
        from vfe3 import metrics
        d["covariance_available"] = 1.0
        d["belief_cond_median"] = float("nan")
        spectrum = metrics.belief_spectrum(
            snapshot.layer_converged[layer_index].sigma, diagonal=self.cfg.diagonal_covariance,
            eps=self.cfg.eps, family=self.cfg.family)
        d["effective_rank"] = float(spectrum["effective_rank"].mean())
        d["belief_cond_median"] = float(spectrum["condition"].median())
        for name in ("relational_conditional_cost_before", "relational_conditional_cost_after",
                     "relational_mean_displacement_norm"):
            d[name] = float(cap.get(name, float("nan")))
        d["relational_conditional_cost_delta"] = (
            d["relational_conditional_cost_after"] - d["relational_conditional_cost_before"])
        d["relational_conditional_cost_tokens"] = float(snapshot.token_ids.numel())
        ops = self.relational_operators_by_layer[layer_index]
        raw_maps = (ops.raw_score_q, ops.raw_score_k)
        minima = []
        for role, raw in zip(("q", "k"), raw_maps):
            minimum = 1.0 if raw is None else float(torch.linalg.svdvals(raw.float()).min())
            d[f"relational_score_{role}_raw_min_singular_value"] = minimum
            minima.append(minimum)
        d["relational_score_raw_min_singular_value"] = min(minima)
        d["relational_score_rank_per_head"] = float(self.cfg.d_head)
        d["relational_content_rank_per_head"] = float(self.cfg.d_head)
        d["relational_identity_content"] = float(self.cfg.relational_identity_content)
        d["relational_parameter_count"] = float(sum(
            p.numel() for layer in self.relational_operators_by_layer for p in layer.parameters()))
        factors = ops.content_factors()
        if factors is not None:
            T, O = factors
            # sigma_max(M_h) is basis-dependent (changes under (T, O) -> (T a, a^{-1} O)); rho = inf of sigma_max over similarity.
            singular = torch.linalg.svdvals(O.float() @ T.float())
            d["relational_content_max_singular_value"] = float(singular.max())
            d["relational_content_effective_rank"] = float(
                torch.linalg.matrix_rank(O.float() @ T.float()).float().mean())
        # Similarity invariants of the content links L_ij^h = U_i M_h U_j^{-1}, M_h = O_hat_h T_hat_h:
        # unchanged by the frames, by (T_h, O_h) -> (B T_h, O_h B^{-1}) and by (T, O) -> (T a, a^{-1} O).
        # relational_content_max_singular_value above is not invariant under the reference change.
        if factors is None:
            d.update({key: float("nan") for key in metrics.LINK_SPECTRUM_KEYS})
        else:
            with torch.autocast(device_type=ops.T.device.type, enabled=False):
                T_hat, O_hat = ops._ambient_content_maps(torch.float64)          # (H, d, K), (H, K, d)
            d.update(metrics.content_link_spectrum(
                T_hat, O_hat,
                beta_self=(beta.diagonal(dim1=-2, dim2=-1)                      # (B, H, N) beta_ii^h
                           if ops.readout_mode == "predictive_content" else None),
                residual=ops.attention_residual))
        covariance = torch.stack((cap["projected_covariance_pre"],
                                  cap["projected_key_covariance_pre"])).float()
        eig = torch.linalg.eigvalsh(covariance)
        d["relational_projected_cov_min_eigenvalue"] = float(eig.min())
        d["relational_projected_cov_max_condition"] = float((eig[..., -1]/eig[..., 0]).max())
        d["relational_covariance_repair_count"] = float(cap.get("relational_covariance_repair_count", 0.0))
        return d

    @torch.no_grad()
    @_block_mlp_diagnostic_eval
    def diagnostics(
        self,
        token_ids: torch.Tensor,           # (B, N) token ids; only sequence 0 is used

        *,
        log_likelihood: Optional[torch.Tensor] = None,  # (N,) optional E_q[log p(o|k)] diagnostic seam
        snapshot:       Optional[DiagnosticSnapshot] = None,  # one-forward captured evaluation state
    ) -> dict:
        r"""Faithful per-step VFE diagnostics at the converged belief (no_grad).

        Recomputes the SAME quantities the E-step uses (transport, pairwise energy
        E_ij, attention weights beta_ij, self-divergence D(q_i||p_i), self-coupling
        alpha_i) at the converged belief returned by :func:`vfe_stack`, then feeds
        them to :mod:`vfe3.metrics`. Every knob matches what the forward pass passed
        to the E-step (``cfg.tau``, ``cfg.family``, ``cfg.divergence_family``,
        ``cfg.renyi_order``, ``cfg.kl_max``, ``cfg.eps``,
        ``cfg.lambda_alpha_mode``/``value``/``b0``/``c0``, ``group.irrep_dims``, the cached
        attention log-prior), so the diagnostic beta is the attention pattern at the
        fixed point, not a re-derivation.

        With built-in full Gaussians on compact equal-head transport, no score
        projection, a prior mask, and no two-hop term, energy statistics use only
        admitted pairs. Compact transport substitutes constant
        SPD placeholders for excluded congruences before certification.
        ``energy_evaluated_pair_frac`` records that scope; energy asymmetry is
        unavailable because it requires reverse pairs. ``sandwich_absmax`` then
        measures admitted head-marginal blocks, or is NaN when that route is
        unavailable. Other families and nonzero two-hop retain all-pair probes.

        This runs under ``torch.no_grad`` at the training log cadence or on demand;
        it adds no argument or branch to :meth:`forward`. The headline terms score the
        last block's converged, pre-transform belief :math:`q^*_L` against the exact live
        prior :math:`p_L` that entered that block (captured by :func:`vfe_stack`, or read
        from ``snapshot.layer_priors[-1]``), so

        .. math::
            F_L = \sum_i \Big[\alpha_i D(q^*_{L,i} \| p_{L,i}) + R(\alpha_i)
                  + \lambda_\beta \sum_j \big(\beta_{ij} E_{ij}(q^*_L)
                  + \tau \beta_{ij} \log(\beta_{ij}/\pi_{ij})\big)\Big]

        is the objective the E-step actually minimized, with no handoff reconstruction.
        The post-E-step transforms (head mixer, CG, block norm, block MLP) are reported
        separately under the ``pre_block_mlp_*`` / ``post_block_mlp_*`` keys.

        Returns ``{attn_entropy, self_coupling, belief_coupling, attention_entropy,
        total, effective_rank}`` plus a conditional ``observation_likelihood`` field (nats; ``effective_rank`` is the per-token
        belief-variance spectrum effective rank, not an attention rank).
        """
        if self.conditional_filtering_enabled:
            snapshot = (self.build_diagnostic_snapshot(token_ids[:1]) if snapshot is None
                        else self._validate_diagnostic_snapshot(token_ids, snapshot))
            return self._relational_diagnostics(snapshot)
        from vfe3.geometry.transport import build_factored_transport
        from vfe3.families.base import get_family
        from vfe3.free_energy import self_divergence_for_alpha, attention_weights, attention_tau
        from vfe3.alpha_i import self_coupling_alpha
        from vfe3 import metrics
        from vfe3 import numerics

        cfg = self.cfg
        fam = get_family(cfg.family)
        if snapshot is None:
            enc = self.prior_bank.encode(token_ids[:1])                    # (1, N, ...)
            belief = BeliefState(mu=enc.mu[0], sigma=enc.sigma[0], phi=self._apply_pos_phi(enc.phi[0]),
                                 right_phi=self._pos_phi_right(enc.phi[0]))
            n = belief.mu.shape[0]
            log_prior = self._attention_log_prior(n, token_ids.device)
            rope = self._rope_rotation(n, token_ids.device)
            cap: dict = {}
            block_mlp_transport, block_mlp_frame = self._block_mlp_frame_transport(
                belief, rope)
            out = vfe_stack(
                belief, belief.mu, belief.sigma, self.group, cfg,
                log_prior=log_prior, block_norm=self.block_norm,
                block_mlps=self.block_mlps,
                block_mlp_frame=block_mlp_frame,
                head_mixer=self.head_mixer, cg_coupling=self.cg_coupling,
                lambda_beta=cfg.lambda_beta,
                transport_state=self.transport_state,
                rope=rope, rope_on_cov=cfg.rope_full_gauge,
                rope_on_value=cfg.rope_on_value,
                capture=cap,
                transport_status=self._transport_status,
                prebuilt_transport=block_mlp_transport,
                rope_insertion=cfg.rope_insertion,
            )
        else:
            snapshot = self._validate_diagnostic_snapshot(token_ids, snapshot)
            belief = _sequence_belief(snapshot.initial_belief)
            n = belief.mu.shape[0]
            log_prior = self._first_sequence_log_prior(snapshot.log_prior, token_ids.shape[0])
            rope = snapshot.rope
            out = _sequence_belief(snapshot.stack_output)
            cap = {
                "converged": _sequence_belief(snapshot.layer_converged[-1]),
                "block_mlp_input": _sequence_belief(snapshot.layer_block_mlp_inputs[-1]),
                "block_mlp_output": _sequence_belief(snapshot.layer_block_mlp_outputs[-1]),
            }

        # The headline F is the E-step's own objective at the last block: the converged,
        # pre-transform belief q*_L scored against the EXACT live prior p_L that entered that block
        # (vfe_stack's capture, or the snapshot's layer_priors), as diagnostics_per_layer reads them.
        # Energy, beta, and the self term are all evaluated at q*_L; the post-norm/MLP states get
        # their own separately named pre_/post_block_mlp_* columns below.
        q_star = cap["converged"]                                   # (N, K) mean, (N, K[, K]) cov
        if snapshot is None:
            mu_p, sigma_p = cap["final_block_prior"]
        else:
            mu_p, sigma_p = (moment[0] for moment in snapshot.layer_priors[-1])

        # Match the forward's transport regime so holonomy_deviation reads the ACTUAL connection.
        # (rope was computed above and now also shaped the converged belief.)
        omega = self._diagnostic_transport(q_star)
        base_omega = None
        if rope is not None:
            effective = RopeTransport(base=omega, rope=rope, on_cov=cfg.rope_full_gauge,
                                      on_value=cfg.rope_on_value, insertion=cfg.rope_insertion,
                                      same_frame_flat_cocycle=getattr(
                                          omega, "same_frame_flat_cocycle", False))
            mu, sigma = q_star.mu, q_star.sigma
            batched = False
            if not cfg.rope_on_value:
                base_omega = omega
        else:
            effective = omega.unsqueeze(0)
            mu, sigma = q_star.mu.unsqueeze(0), q_star.sigma.unsqueeze(0)
            batched = True
        energy = self._coupling_energy(fam,                          # (N, N) or (H, N, N)
            mu, sigma, mu, sigma, effective,
            log_prior=log_prior,
            alpha=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
            divergence_family=cfg.divergence_family,
            irrep_dims=self.group.irrep_dims,
        )
        coupling_energy = None
        if base_omega is not None:
            coupling_energy = self._coupling_energy(fam,
                mu, sigma, mu, sigma, base_omega,
                log_prior=log_prior,
                alpha=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
                divergence_family=cfg.divergence_family,
                irrep_dims=self.group.irrep_dims,
            )
        if batched:
            energy = energy[0]
        energy_support = None
        if (fam is FullGaussian and log_prior is not None
                and sigma.dim() == mu.dim() + 1):
            from vfe3.geometry.transport import (
                _reads_only_head_marginals, _resolve_compact_for_head_blocks,
            )

            compact = _resolve_compact_for_head_blocks(effective)
            if compact is not None and _reads_only_head_marginals(compact, self.group.irrep_dims):
                candidate_support = (~torch.isneginf(log_prior)).expand_as(energy)
                if not bool(candidate_support.all()):
                    energy_support = candidate_support
        measured_energy = energy if energy_support is None else energy[energy_support]
        _tau_b = attention_tau(
            _as_coeff(cfg.kappa_beta, out.mu.device), self._belief_score_irrep_dims())
        # beta at q*_L itself. The snapshot's beta_maps score each block OUTPUT (post-transform),
        # which is not the state this headline evaluates.
        beta = attention_weights(energy, tau=_tau_b, log_prior=log_prior)   # (N, N) or (H, N, N)
        _q_pre_block_mlp = cap["block_mlp_input"]                 # immediate input after mixer/CG/norm
        _q_conv = cap["block_mlp_output"]                         # immediate BlockMLP output
        _block_mlp_states_equal = _belief_states_mechanically_equal(
            _q_pre_block_mlp, _q_conv)
        self_div = self_divergence_for_alpha(                     # D(q*_L || p_L): the E-step and M-step term (F19)
            self._family_instance(fam, q_star.mu, q_star.sigma),
            self._family_instance(fam, mu_p, sigma_p),
            alpha=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
            divergence_family=cfg.divergence_family, lambda_alpha_mode=cfg.lambda_alpha_mode,
        )
        alpha, alpha_reg = self_coupling_alpha(
            self_div, mode=cfg.lambda_alpha_mode, value=cfg.lambda_alpha, b0=_as_coeff(cfg.b0, out.mu.device), c0=_as_coeff(cfg.c0, out.mu.device),
        )

        # PB-07: route the migrated numeric diagnostics (row entropy, free-energy decomposition,
        # belief effective rank) through the metric registry instead of three direct calls, so a
        # config-selected override dispatches here too. The context carries exactly the inputs the
        # three wrappers read. ``_diag`` is the dimension-based diagonal(N,K)-vs-full(N,K,K) flag
        # passed explicitly to effective_rank because shape-squareness auto-inference mis-reads a
        # diagonal (N, K) table as a full covariance when N == K (e.g. max_seq_len == embed_dim).
        # ``attention_entropy`` (row entropy) is aliased to ``attn_entropy`` so it does not overwrite
        # the free-energy component of the same name that ``free_energy_terms`` flattens in; the
        # flatten carries an overwrite-collision guard. Holonomy/gauge metrics keep their bespoke
        # paths below (they consume sibling confidence bounds / active-frame branches).
        _lb = cfg.lambda_beta   # scaled-F total reflects lambda_beta
        _diag = out.sigma.dim() == out.mu.dim()                     # diagonal (N,K) vs full (N,K,K)
        metric_context = {
            "sigma":                     out.sigma,
            "diagonal":                  _diag,
            "eps":                       cfg.eps,
            "family":                    cfg.family,
            "self_div":                  self_div,
            "energy":                    energy,
            "beta":                      beta,
            "alpha":                     alpha,
            "tau":                       _tau_b,
            "lambda_beta":               _lb,
            "include_attention_entropy": cfg.include_attention_entropy,
            "log_prior":                 log_prior,
            "alpha_reg":                 (alpha_reg if cfg.lambda_alpha_mode != "constant" else None),
            "coupling_energy":           coupling_energy,
            "log_likelihood":            log_likelihood,
        }
        registered = metrics.compute_metrics(list(metrics.DIAGNOSTIC_METRIC_NAMES), **metric_context)

        def state_terms(state: BeliefState) -> 'Tuple[torch.Tensor, Mapping[str, Any]]':
            r"""Return (D(q||p_L), F terms) for one post-E-step state against the exact prior p_L.

            A state mechanically equal to q*_L reuses the headline evaluation; any other state
            (block norm, mixer, CG, or MLP output) gets its own complete pair-energy evaluation.
            """
            if _belief_states_mechanically_equal(state, q_star):
                return self_div, registered["free_energy_terms"]
            state_omega = self._diagnostic_transport(state)
            state_base_omega = None
            if rope is not None:
                state_effective = RopeTransport(
                    base=state_omega, rope=rope, on_cov=cfg.rope_full_gauge,
                    on_value=cfg.rope_on_value, insertion=cfg.rope_insertion,
                    same_frame_flat_cocycle=getattr(state_omega, "same_frame_flat_cocycle", False),
                )
                state_mu, state_sigma = state.mu, state.sigma
                state_batched = False
                if not cfg.rope_on_value:
                    state_base_omega = state_omega
            else:
                state_effective = state_omega.unsqueeze(0)
                state_mu = state.mu.unsqueeze(0)
                state_sigma = state.sigma.unsqueeze(0)
                state_batched = True
            state_energy = self._coupling_energy(
                fam, state_mu, state_sigma, state_mu, state_sigma, state_effective,
                log_prior=log_prior,
                alpha=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
                divergence_family=cfg.divergence_family, irrep_dims=self.group.irrep_dims,
            )
            state_coupling_energy = None
            if state_base_omega is not None:
                state_coupling_energy = self._coupling_energy(
                    fam, state_mu, state_sigma, state_mu, state_sigma, state_base_omega,
                    log_prior=log_prior,
                    alpha=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
                    divergence_family=cfg.divergence_family, irrep_dims=self.group.irrep_dims,
                )
            if state_batched:
                state_energy = state_energy[0]
            state_beta = attention_weights(state_energy, tau=_tau_b, log_prior=log_prior)
            state_self_div = self_divergence_for_alpha(
                self._family_instance(fam, state.mu, state.sigma),
                self._family_instance(fam, mu_p, sigma_p),
                alpha=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
                divergence_family=cfg.divergence_family, lambda_alpha_mode=cfg.lambda_alpha_mode,
            )
            state_alpha, state_alpha_reg = self_coupling_alpha(
                state_self_div, mode=cfg.lambda_alpha_mode, value=cfg.lambda_alpha,
                b0=_as_coeff(cfg.b0, out.mu.device), c0=_as_coeff(cfg.c0, out.mu.device),
            )
            return state_self_div, metrics.free_energy_terms(
                state_self_div, state_energy, state_beta, state_alpha, tau=_tau_b,
                lambda_beta=_lb,
                include_attention_entropy=cfg.include_attention_entropy, log_prior=log_prior,
                alpha_reg=(state_alpha_reg if cfg.lambda_alpha_mode != "constant" else None),
                coupling_energy=state_coupling_energy, log_likelihood=log_likelihood,
            )

        # The immediate MLP input and output are evaluated separately unless actual belief
        # tensors prove them equal, in which case one evaluation serves both.
        pre_self_div, pre_terms = state_terms(_q_pre_block_mlp)
        if _block_mlp_states_equal:
            post_self_div, post_terms = pre_self_div, pre_terms
        else:
            post_self_div, post_terms = state_terms(_q_conv)
        d: Dict[str, float] = {
            "energy_evaluated_pair_frac": (
                1.0 if energy_support is None else float(energy_support.float().mean())),
        }
        for metric_name, output_name, flatten in metrics.DIAGNOSTIC_METRIC_OUTPUTS:
            value = registered[metric_name]
            if flatten:
                overlap = set(value) & set(d)
                if overlap:
                    raise KeyError(f"diagnostic metric {metric_name!r} would overwrite {sorted(overlap)}")
                d.update({key: float(item) for key, item in value.items()})
            else:
                d[output_name] = float(value)
        # Raw (un-regularized) belief->prior drift sum_i D(q_i||p_i): the divergence WITHOUT the
        # alpha_i coefficient OR the R(alpha_i) regularizer that free_energy_terms folds into
        # self_coupling. Under lambda_alpha_mode='constant' (alpha=1, R=0) the two coincide; under the
        # state-dependent envelope self_coupling is pinned near the K*c0 regularizer floor (alpha_i*
        # D + R = c0[1 + log((b0+D)/c0)] per coord), so this raw term is the informative drift signal.
        d["self_divergence"] = float(self_div.sum())            # D(q*_L||p_L), summed over tokens (and coords)
        d["post_block_mlp_self_divergence"] = float(post_self_div.sum())
        d["pre_block_mlp_self_divergence"] = float(pre_self_div.sum())
        def state_components(terms: Mapping[str, float], raw_self: torch.Tensor) -> Dict[str, float]:
            return {
                "self_coupling": float(terms["self_coupling"]),
                "belief_coupling": float(_lb) * float(terms["belief_coupling"]),
                "attention_entropy": (
                    float(_lb) * float(terms["attention_entropy"])
                    if cfg.include_attention_entropy else 0.0
                ),
                "observation_nll": -float(terms.get("observation_likelihood", 0.0)),
                "self_divergence": float(raw_self.sum()),
                "observation_likelihood": float(terms.get("observation_likelihood", 0.0)),
            }

        post_components = state_components(post_terms, post_self_div)
        pre_components = state_components(pre_terms, pre_self_div)
        d.update(_state_specific_free_energy_diagnostics(
            pre_components, post_components,
            states_equal_proven=_block_mlp_states_equal,
        ))
        d["block_mlp_state_equality_proven"] = _block_mlp_states_equal
        d.update(_numerical_policy_diagnostics(
            m_phi_group_trust_radius=cfg.m_phi_group_trust_radius,
            phi_mstep_max_matrix_norm=cfg.phi_mstep_max_matrix_norm,
            transport_chart_max_norm=cfg.transport_chart_max_norm,
            exp_fp64_norm_threshold=cfg.exp_fp64_norm_threshold,
        ))
        # Frame-numerics probes (diagnostics tier): the coboundary closure residual -- mean Frobenius
        # departure of Omega_ij Omega_jk Omega_ki from I, identically 0 in exact arithmetic for
        # Omega_ij = U_i U_j^{-1}, so it measures roundoff of the stored (U, U^{-1}) pair, not curvature --
        # and the spread of log|det Omega| = tr(embed(phi)) across tokens (0 at phi=0). Pure
        # measurements at the converged transport; off the training graph (no_grad).
        # The residual uses the SAMPLED estimator (seeded random distinct triples) rather than the
        # deterministic row-major prefix, which at N=128/max_triangles=512 covers only anchor i=0's
        # local neighborhood -- a systematically biased sample. The sampled mean is representative and
        # still ~0 on the flat phi-cocycle (a roundoff monitor, not a flatness certificate; see the
        # metrics.py holonomy section note); the dict key is unchanged.
        # ---- extended per-eval observability (2026-06-13 run-diagnostics rollout) ----
        # Every reduction below reads tensors already materialized above (out.mu/sigma/active frame,
        # omega, energy, beta, self_div); no extra forward, no_grad. NEW keys only -- d["total"] and
        # the existing block values are untouched (the run-diagnostics contract pins
        # d["holonomy_deviation"], whose semantics is preserved as the mean below).
        _LOG2 = 0.6931471805599453                                   # row-entropy floor for a 2-way split
        # ``_diag`` (diagonal (N,K) vs full (N,K,K)) is computed once in the registry-dispatch block
        # above and reused by the spectrum/Fisher/guard metrics below; it is passed explicitly because
        # shape-squareness auto-inference mis-reads a diagonal (N, K) table as full when N == K.

        hol = metrics.holonomy_deviation_sampled(omega)
        d["holonomy_deviation"] = float(hol["mean"])                 # unchanged key/semantics
        d["holonomy_ci_lo"]     = float(hol["ci_lo"])                # bootstrap band over sampled triples
        d["holonomy_ci_hi"]     = float(hol["ci_hi"])
        # Conjugation-invariant trace analog 1 - Re Tr(H)/K (PIFB:862-869) of the residual above; on the
        # coboundary it is also a roundoff monitor (~0), not a curvature measurement.
        d["holonomy_wilson"]    = float(metrics.holonomy_wilson_sampled(omega)["deviation_mean"])
        # Active-frame health, evaluated on the active vertex frame.
        active_vertex = build_factored_transport(
            out.phi, self.group,
            exp_fp64_norm_threshold=cfg.exp_fp64_norm_threshold,
            clamp_monitor=cfg.transport_clamp_monitor,
            validity_max_norm=cfg.transport_chart_max_norm,
            right_phi=out.right_phi,
        ).exp_phi
        active_logdet = torch.linalg.slogdet(active_vertex).logabsdet
        d["gauge_trace_spread"] = float(active_logdet.std(unbiased=False))
        ginv = metrics.group_gauge_invariant(active_vertex, self.group).float()
        active_svd = torch.linalg.svdvals(active_vertex)
        vertex_cond = active_svd[..., 0] / active_svd[..., -1].clamp(min=cfg.eps)
        _ghi = metrics.per_head_gauge_invariants(active_vertex, self.group.irrep_dims)
        d["gauge_invariant_mean"]   = float(ginv.mean())
        d["gauge_invariant_spread"] = float(ginv.std(unbiased=False))

        # Transport DIRECTEDNESS + conditioning + sandwich overflow. NOTE: directedness
        # (transport_asymmetry, energy_*_asymmetry) is intrinsic to ANY nonzero gauge -- it is nonzero
        # on the FLAT cocycle (Omega_ji = Omega_ij^{-1} != Omega_ij), so it is NOT a curvature /
        # non-flatness signal; cocycle_residual and holonomy_deviation are the flatness diagnostics.
        d["cocycle_residual"] = float(metrics.cocycle_residual_sampled(omega))   # composition-law flatness
        d["vertex_cond_max"]  = float(vertex_cond.max())
        #   FLAT path: pairwise cond(Omega_ij) = cond(exp_phi_i exp(-phi_j)) <= vertex_cond_max^2.
        #   sandwich_absmax
        #   below is the direct (Omega Sigma Omega^T) overflow signal.
        # The forward sandwich is an overflow probe on the TRANSPORT itself, so it is built here
        # rather than reused from the coupling energy: a family that evaluates the transported
        # divergence some other way (the backward pull of gaussian_diagonal_exact, the frame-cancelling
        # identity of gaussian_frame_diagonal) never forms Omega Sigma Omega^T, and this diagnostic
        # must still report it.
        if energy_support is None:
            sandwich = self._transport_dispersion(fam, sigma, effective,
                                                diagonal_out=(sigma.dim() == mu.dim()))
            d["sandwich_absmax"] = float(sandwich.abs().max())
            _ed = metrics.energy_directedness(energy)
            d["energy_abs_asymmetry"] = float(_ed["abs_asymmetry"])
            d["energy_rel_asymmetry"] = float(_ed["rel_asymmetry"])
        else:
            from vfe3.geometry.transport import transport_covariance_head_blocks

            # Only admitted head marginals were evaluated; never certify excluded
            # pairs again for an optional overflow or reverse-energy probe.
            sandwich_support = (energy_support.movedim(-3, -1)
                                if len(self.group.irrep_dims) > 1
                                else energy_support.unsqueeze(-1))
            sandwich = transport_covariance_head_blocks(
                effective, sigma,
                pair_support=sandwich_support,
                retain_full_precision=(self.full_cov_kl_precision == "fp64"),
                diagonal_out=(sigma.dim() == mu.dim()),
                marginal_blocks=self.group.irrep_dims,
            )
            if sandwich is None:
                d["sandwich_absmax"] = float("nan")
            else:
                supported_sandwich = sandwich[sandwich_support.expand(sandwich.shape[:-2])]
                d["sandwich_absmax"] = (float(supported_sandwich.abs().max())
                                        if supported_sandwich.numel() else float("nan"))
            d["energy_abs_asymmetry"] = float("nan")
            d["energy_rel_asymmetry"] = float("nan")
        d["transport_asymmetry"] = float(metrics.transport_asymmetry(omega).mean())
        # Per-head gauge specialization: do the per-head GL(d_head) frames specialize, or collapse to a
        # shared frame? Mean block anisotropy + spread of log|det| across heads (single block -> 0
        # spread). Informative for block_glk / tied_block_glk (independent per-block GL frames); for the
        # orthogonal/symplectic tied towers (so_k/so_n/sp_n) it is structurally vacuous (det=1, unit
        # singular values, one shared group element), so read it only on the GL-block groups.
        d["gauge_head_aniso_mean"]    = float(_ghi["anisotropy"].float().mean())
        d["gauge_head_logdet_spread"] = float(_ghi["logdet"].float().std(unbiased=False))

        # phi frame magnitude: a collapse to phi=0 silently degenerates to an UNGAUGED transformer
        # (trivially equivariant, so no equivariance metric flags it).
        phi_norm = torch.linalg.norm(out.phi, dim=-1)           # (N,)
        d["phi_norm_mean"] = float(phi_norm.mean())
        d["phi_norm_std"]  = float(phi_norm.std(unbiased=False))
        from vfe3.geometry.transport import TRANSPORT_CLAMP_MAX_NORM
        matrix_norm = metrics.diagnostic_frame_norm(
            out.phi.float(), self.group, max_norm=TRANSPORT_CLAMP_MAX_NORM, eps=cfg.eps)
        exp_scale = (
            TRANSPORT_CLAMP_MAX_NORM / matrix_norm.clamp(min=cfg.eps)
        ).clamp(max=1.0)
        d["phi_matrix_norm_median"] = float(matrix_norm.median())
        d["phi_matrix_norm_p95"] = float(torch.quantile(matrix_norm, 0.95))
        d["phi_matrix_norm_p99"] = float(torch.quantile(matrix_norm, 0.99))
        d["phi_matrix_norm_max"] = float(matrix_norm.max())
        d["phi_exp_clamp_frac"] = float((exp_scale < 1.0).float().mean())
        d["phi_exp_scale_min"] = float(exp_scale.min())
        d["vertex_cond_median"] = float(vertex_cond.float().median())
        d["vertex_cond_p95"] = float(torch.quantile(vertex_cond.float(), 0.95))
        d["vertex_cond_p99"] = float(torch.quantile(vertex_cond.float(), 0.99))
        if out.right_phi is not None:
            right_norm = metrics.diagnostic_frame_norm(
                out.right_phi.float(), self.group, max_norm=TRANSPORT_CLAMP_MAX_NORM, eps=cfg.eps)
            right_scale = (
                TRANSPORT_CLAMP_MAX_NORM / right_norm.clamp(min=cfg.eps)
            ).clamp(max=1.0)
            d["pos_phi_matrix_norm_p95"] = float(torch.quantile(right_norm, 0.95))
            d["pos_phi_matrix_norm_p99"] = float(torch.quantile(right_norm, 0.99))
            d["pos_phi_matrix_norm_max"] = float(right_norm.max())
            d["pos_phi_exp_clamp_frac"] = float((right_scale < 1.0).float().mean())
            d["pos_phi_exp_scale_min"] = float(right_scale.min())

        # Belief covariance conditioning + PD margin (effective_rank is blind to one collapsing mode).
        bs = metrics.belief_spectrum(
            out.sigma,
            diagonal=_diag,
            eps=cfg.eps,
            family=cfg.family,
        )
        cond = bs["condition"].float()
        d["belief_cond_median"] = float(cond.median())
        d["belief_cond_p95"]    = float(torch.quantile(cond, 0.95))
        d["belief_cond_max"]    = float(cond.max())
        # The RAW extremes beside the ratio (audit 2026-08-06 F16). Once an eigenvalue reaches the
        # eps floor, belief_cond_* is lam_max/eps -- a guard readout, not a belief property -- and
        # the three reductions above cannot distinguish "the belief became ill-conditioned" from
        # "the floor engaged". lam_min does, and it is the quantity the SPD trust region moves.
        d["belief_lam_min"]     = float(bs["lam_min"].float().min())
        d["belief_lam_max"]     = float(bs["lam_max"].float().max())
        # What fraction of the belief_cond_* reductions above are actually about the belief
        # (audit 2026-08-06 F16). 1.0 = every token's lam_min clears the floor and the ratios mean
        # what they say; below 1.0 the floored tokens report lam_max/eps, and once the floored set
        # exceeds 5% the p95 itself lands inside it -- which is exactly how "val_belief_cond_p95
        # 7.76 -> 1.236e+06" was read as the belief degrading when it was the guard engaging.
        d["belief_cond_valid_frac"] = float(bs["condition_is_belief"].float().mean())
        # Headroom behind the decode's a_v accuracy (audit 2026-08-06 F32): a_v sums terms of size
        # 1/sigma_v and is divided by 2*tau, so this is the number that says how many digits the
        # working precision still has. Free -- one reduction over a parameter table, off the graph.
        if self.cfg.decode_mode != "linear":
            d["decode_sigma_v_min"] = self.prior_bank.decode_sigma_v_min()
        # clamp lam_min at eps before dividing -- matches the floor belief_spectrum's condition number
        # uses, so a floored / sub-floor belief reads ~1.0 (not 0.0) consistently across the reductions.
        d["belief_pd_margin"]   = float((bs["eigenvalues"][..., -1].clamp(min=cfg.eps).float() / cfg.eps).min())

        # Per-token effective-rank distribution (the logged mean hides a bimodal rank-1/rank-K collapse).
        er = metrics.effective_rank_per_token(
            out.sigma,
            diagonal=_diag,
            eps=cfg.eps,
            family=cfg.family,
        ).float()
        d["eff_rank_p5"]     = float(torch.quantile(er, 0.05))
        d["eff_rank_median"] = float(er.median())
        d["eff_rank_p95"]    = float(torch.quantile(er, 0.95))

        # One-half mean-block Fisher trace (the KL quadratic coefficient and belief precision).
        fish = metrics.half_fisher_trace(
            out.sigma,
            diagonal=_diag,
            eps=cfg.eps,
            family=cfg.family,
        ).float()
        d["fisher_trace_mean"]   = float(fish.mean())
        d["fisher_trace_median"] = float(fish.median())

        # Numerical safety rails inert (pure path) vs load-bearing (fixed point is a clamp artifact)?
        # A pruned grid contains constant placeholders, so guard fractions must
        # denominate over evaluated pairs. Full-grid callers retain the old split.
        gs = metrics.guard_saturation(out.sigma, measured_energy, self_div, diagonal=_diag,
                                      eps=cfg.eps, sigma_max=cfg.sigma_max, kl_max=cfg.kl_max,
                                      log_prior=(log_prior if energy_support is None else None))
        if energy_support is not None:
            gs["energy_klmax_frac_live"] = gs["energy_klmax_frac"]
            gs["energy_live_pair_frac"] = d["energy_evaluated_pair_frac"]
        for _k, _v in gs.items():
            d[f"guard_{_k}"] = float(_v)
        # Energy-near-ceiling fraction: the share of evaluated pair energies in (0.9, 1.0)*kl_max, an
        # early warning that the KL clamp is about to saturate (softer than guard_energy_klmax_frac's
        # exact pin). It does NOT measure Renyi alpha->1 cancellation: that band depends on
        # |renyi_order - 1| (gaussian._RENYI_KL_BAND), not on E/kl_max, and it is empty at
        # renyi_order=1. The historical key name is kept so metrics.csv columns stay comparable.
        d["renyi_band_frac"] = float(((measured_energy > 0.9 * cfg.kl_max)
                                      & (measured_energy < cfg.kl_max)).float().mean())

        # Non-finite fraction over the converged operator tensors (one NaN silently poisons AdamW).
        d["nonfinite_frac"] = float(max(
            numerics.nan_inf_fraction(out.mu),  numerics.nan_inf_fraction(out.sigma),
            numerics.nan_inf_fraction(out.phi), numerics.nan_inf_fraction(measured_energy),
            numerics.nan_inf_fraction(beta),
        ))

        # Attention-entropy COLLAPSE: per-head min row entropy + count of near-deterministic heads at
        # the converged (last-block) belief; the single logged attn_entropy averages collapse away.
        ent_rows = metrics.attention_entropy_rows(beta)             # (N,) single head or (H, N) multi-head
        # Skip query rows whose causal support cannot carry entropy (audit 2026-07-27). Under
        # causal / causal_noself / causal_alibi* / causal_windowed, row 0 has exactly ONE admissible
        # key, so beta[..., 0, :] is one-hot and H_0 = 0 for every head, layer, seed and parameter
        # value -- the min over rows could never exceed the eps artifact. Measured: attn_entropy_min
        # was 8.28931e-11 and collapsed_heads 2.0, byte-identical across two seeds AND three distinct
        # causal priors, while the `uniform` control gave 1.386293 / 0.0. Both columns were therefore
        # structurally constant, not a head-collapse signal, and were logged and plotted as one.
        # Row i admits i+1 keys under a causal mask (i under no-self), so the first row that can
        # carry entropy is the first with at least two admissible keys.
        # A row admitting s_i keys cannot exceed ln(s_i) nats, so the collapse test is scored on the
        # NORMALIZED entropy H_i / ln(s_i) (audit 2026-08-06 F22): against a fixed ln 2 the first
        # two-key row sits ON the threshold and float noise decided the alarm, flagging every head
        # on essentially every step. ``attn_entropy_min`` keeps its raw-nats meaning so runs already
        # on disk stay comparable; ``attn_entropy_min_norm`` is the commensurate twin.
        row_support = (beta > 0.0).sum(dim=-1)                       # (..., N) admissible keys per row
        scorable = row_support >= 2                                  # rows a min-entropy probe can see
        if bool(scorable.any()):
            masked = ent_rows.masked_fill(~scorable, float("inf"))
            head_min = masked.min(dim=-1).values if masked.dim() >= 2 else masked.min().reshape(1)
            worst_norm, collapsed = metrics.attention_head_collapse(beta)
            d["attn_entropy_min"]             = float(head_min.min())
            d["attn_entropy_min_norm"]        = float(torch.nan_to_num(
                worst_norm, nan=float("inf")).min())
            d["attn_entropy_collapsed_heads"] = float(collapsed.float().sum())
        else:                                                        # N == 1, or a fully one-hot map
            d["attn_entropy_min"]             = float("nan")
            d["attn_entropy_min_norm"]        = float("nan")
            d["attn_entropy_collapsed_heads"] = float("nan")

        # Registry-owned transport order parameters (conditional columns, mirroring lambda_beta).
        # Lowercasing the stable serialization key preserves the established W/M/L metric names and
        # gives newly registered state a model-level norm diagnostic without a central name branch.
        # Only registrations that declare edge-table topology emit an off-diagonal norm.
        for state_key, parameter in self.transport_state.items():
            metric_key = state_key.lower()
            detached = parameter.detach()
            d[f"{metric_key}_norm"] = float(torch.linalg.norm(detached))
            if state_key in self._transport_offdiag_state_keys:
                off_diagonal = ~torch.eye(
                    detached.shape[0], dtype=torch.bool, device=detached.device)
                d[f"{metric_key}_offdiag_norm"] = float(torch.linalg.norm(detached[off_diagonal]))
        _hm = getattr(self, "head_mixer", None)
        if _hm is not None and hasattr(_hm, "mixer_deltas"):        # use_head_mixer=True
            d["head_mixer_drift"] = max(
                (float(torch.linalg.norm(p.detach())) for p in _hm.mixer_deltas), default=0.0)
        if hasattr(self.prior_bank, "head_evidence_logits"):
            weights, _ = self.prior_bank.head_evidence_weights(
                dtype=self.prior_bank.head_evidence_logits.dtype,
                device=self.prior_bank.head_evidence_logits.device,
            )
            probabilities = weights / weights.numel()
            d["head_evidence_weights"] = weights.detach().cpu().tolist()
            d["head_evidence_entropy"] = float(torch.special.entr(probabilities).sum())
            d["head_evidence_max_abs_drift"] = float((weights - 1.0).abs().max())
        return d

    @torch.no_grad()
    @_block_mlp_diagnostic_eval
    def attention_maps(
        self,
        token_ids: torch.Tensor,           # (B, N) token ids; only sequence 0 is used

        *,
        score_at:  str                          = "converged",   # "converged" | "entry" | "prior"
        snapshot:  Optional[DiagnosticSnapshot] = None,
    ) -> torch.Tensor:                     # (L, H, N, N) per-layer, per-head attention beta_ij
        r"""Per-layer, per-head attention weights ``beta_ij`` for sequence 0 (no_grad).

        ``score_at`` selects WHERE in the block the pattern is read (audit 2026-07-26 R-3):

        - ``"converged"`` (default, unchanged) scores at each block's OUTPUT belief, after the
          E-step has run and after ``head_mixer``/``block_norm``/block MLP. This is what the
          shipped ``attention/`` heatmaps have always plotted. ``diagnostics`` scores the
          pre-transform converged belief instead, so the two coincide only when those
          post-E-step transforms are off.
        - ``"entry"`` scores at the belief ENTERING the block -- the pattern the E-step's first
          iteration actually used to aggregate. At ``n_e_steps=1`` this IS the model's attention;
          the converged map is a post-hoc re-score of a belief the E-step already moved, and the
          two differ by up to the full row mass.
        - ``"prior"`` scores with the coupling BELIEF energy zeroed, leaving
          ``softmax_j(log_prior)`` alone -- the reference for how much of a map is the attention
          prior rather than content. The prior is purely positional.

        ``snapshot`` short-circuits to the stored maps and is only valid for ``"converged"``.

        Replays the :func:`vfe_stack` block loop one block at a time -- mirroring its
        ``mu_p``/``sigma_p`` handoff (``prior_handoff_rho``/``prior_handoff_sigma``) line for
        line -- and, at the CONVERGED output belief of each block, recomputes the attention
        pattern the SAME way :meth:`diagnostics` does at the final belief: transport
        Omega_ij(phi) -> pairwise energy E_ij = D(q_i || Omega_ij q_j) -> beta = softmax_j
        (log_prior - E/tau). The per-irrep-block (per-head) energy gives a leading head axis
        ``H = len(group.irrep_dims)`` (1 for a single-block group: glk / so_k), so the result is
        ``(L, H, N, N)`` (rows = query i, cols = key j).

        This is OFF the training hot path (no_grad, no graph) and is intended for periodic
        figure generation, not every step. With the post-E-step transforms off, the LAST layer's
        map equals the attention :meth:`diagnostics` reads, since both then score the same
        converged belief of the final block.
        """
        if self.conditional_filtering_enabled:
            if score_at not in ("converged", "entry", "prior"):
                raise ValueError(f"unsupported conditional attention score_at={score_at!r}")
            snapshot = (self.build_diagnostic_snapshot(token_ids[:1]) if snapshot is None
                        else self._validate_diagnostic_snapshot(token_ids, snapshot))
            # Both ordinary and entry displays expose the actual pre-update router;
            # a converged re-score is not part of the conditional forward model.
            if score_at != "prior":
                return snapshot.beta_maps
            from vfe3.free_energy import attention_weights
            return attention_weights(torch.zeros_like(snapshot.beta_maps),
                                     log_prior=snapshot.log_prior)
        if score_at not in ("converged", "entry", "prior"):
            raise ValueError(f"score_at must be 'converged', 'entry' or 'prior', got {score_at!r}")
        if snapshot is not None:
            if score_at != "converged":
                raise ValueError(
                    f"a diagnostic snapshot stores only the converged maps; score_at={score_at!r} "
                    "must be replayed from the model")
            return self._validate_diagnostic_snapshot(token_ids, snapshot).beta_maps
        from vfe3.families.base import get_family
        from vfe3.free_energy import attention_weights, attention_tau

        cfg = self.cfg
        enc = self.prior_bank.encode(token_ids[:1])                   # (1, N, ...)
        belief = BeliefState(mu=enc.mu[0], sigma=enc.sigma[0], phi=self._apply_pos_phi(enc.phi[0]),
                             right_phi=self._pos_phi_right(enc.phi[0]))
        n = belief.mu.shape[0]
        rope = self._rope_rotation(n, token_ids.device)
        log_prior = self._attention_log_prior(n, token_ids.device)   # (N, N)
        fam = get_family(cfg.family)
        rho, rho_s = cfg.prior_handoff_rho, cfg.prior_handoff_sigma
        mu_p, sigma_p = belief.mu, belief.sigma

        _base_tau = attention_tau(
            _as_coeff(cfg.kappa_beta, belief.mu.device), self._belief_score_irrep_dims())

        def _score(state) -> torch.Tensor:                            # (H, N, N) beta at `state`
            # Attention at `state`, recomputed exactly as diagnostics does: the transport regime is
            # matched, and
            # the energy is per-irrep-block (per-head). Shared by the entry and converged scoring
            # points so the two can only differ by WHERE they are called (audit 2026-07-26 R-3).
            tau = _base_tau
            omega = self._diagnostic_transport(state)
            if rope is not None:
                effective = RopeTransport(base=omega, rope=rope, on_cov=cfg.rope_full_gauge,
                                          on_value=cfg.rope_on_value, insertion=cfg.rope_insertion,
                                          same_frame_flat_cocycle=getattr(
                                              omega, "same_frame_flat_cocycle", False))
                mu, sigma = state.mu, state.sigma
                batched = False
            else:
                effective = omega.unsqueeze(0)
                mu, sigma = state.mu.unsqueeze(0), state.sigma.unsqueeze(0)
                batched = True
            energy = self._coupling_energy(fam,                       # (N, N) or (H, N, N)
                mu, sigma, mu, sigma, effective,
                log_prior=log_prior,
                alpha=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
                divergence_family=cfg.divergence_family, irrep_dims=self.group.irrep_dims,
            )
            if batched:
                energy = energy[0]
            if score_at == "prior":
                energy = torch.zeros_like(energy)                     # positional channel alone
            beta = attention_weights(energy, tau=tau, log_prior=log_prior)
            return beta.unsqueeze(0) if beta.dim() == 2 else beta     # single-block group -> H=1

        block_mlp_transport, block_mlp_frame = self._block_mlp_frame_transport(
            belief, rope)
        maps = []
        for layer_index in range(cfg.n_layers):
            if score_at != "converged":
                maps.append(_score(belief))                           # BEFORE the E-step runs
            belief = vfe_block(                                       # converged belief at this block
                belief, mu_p, sigma_p, self.group, cfg, log_prior=log_prior,
                block_norm=self.block_norm,
                block_mlp=(self.block_mlps[layer_index] if self.block_mlps is not None else None),
                block_mlp_frame=block_mlp_frame,
                head_mixer=self.head_mixer,                            # replay the mixer too (audit 2026-06-09 overnight F32)
                lambda_beta=cfg.lambda_beta,
                transport_state=self.transport_state,
                cg_coupling=self.cg_coupling,
                rope=rope, rope_on_cov=cfg.rope_full_gauge,            # match forward: converge WITH rope
                rope_on_value=cfg.rope_on_value,
                transport_status=self._transport_status,
                prebuilt_transport=block_mlp_transport,
                tau=_base_tau,
                rope_insertion=cfg.rope_insertion,
            )
            if score_at == "converged":
                maps.append(_score(belief))                          # (H, N, N) at the block OUTPUT

            mu_p = (1.0 - rho) * mu_p + rho * belief.mu              # handoff (mirrors vfe_stack)
            sigma_p = (1.0 - rho_s) * sigma_p + rho_s * belief.sigma
        return torch.stack(maps, dim=0)                              # (L, H, N, N)

    @torch.no_grad()
    @_block_mlp_diagnostic_eval
    def diagnostics_per_layer(
        self,
        token_ids: torch.Tensor,           # (B, N) token ids; only sequence 0 is used

        *,
        log_likelihood: Optional[torch.Tensor] = None,  # (N,) optional E_q[log p(o|k)] diagnostic seam
        snapshot:       Optional[DiagnosticSnapshot] = None,
    ) -> dict:                             # each value a list of length L = cfg.n_layers
        r"""Per-LAYER (inference-depth) belief-channel diagnostics for sequence 0 (no_grad).

        :meth:`diagnostics` collapses the block stack to the FINAL belief and reports one scalar per
        metric, so the metrics.csv and the converged-state figures never expose the depth axis. This
        replays the :func:`vfe_stack` block loop one block at a time -- mirroring its
        ``mu_p``/``sigma_p`` handoff EXACTLY as :meth:`attention_maps` does -- and at each block's
        CONVERGED belief recomputes the SAME belief-channel quantities :meth:`diagnostics` uses
        (transport Omega_ij(phi), pairwise energy E_ij, attention beta_ij, self-divergence
        D(q_i||p_i) against THAT block's prior, self-coupling alpha_i), then the same
        :mod:`vfe3.metrics` reductions. Unlike diagnostics' last-block prior reconstruction, the
        self-term here reads each block's OWN handoff prior, so the per-layer self-coupling is exact.

        The model-channel blocks (hyper-prior, gamma) are a single hierarchical coupling evaluated
        once at the converged frame, NOT iterated per block, so they are deliberately absent: the
        per-layer ``total`` is the BELIEF-channel free energy at that depth. OFF the training hot path
        (no_grad, no graph); intended for periodic figure / per-layer-CSV generation, not every step.

        Returns a dict of L-length lists: ``self_coupling``, ``belief_coupling``,
        ``attention_entropy``, ``total`` (belief-channel F), ``self_divergence``,
        ``holonomy_deviation``, ``holonomy_wilson``, ``gauge_trace_spread``,
        ``gauge_invariant_spread``, ``effective_rank``, ``attn_entropy``, ``belief_cond_median``,
        ``phi_norm_mean``.
        """
        if self.conditional_filtering_enabled:
            snapshot = (self.build_diagnostic_snapshot(token_ids[:1]) if snapshot is None
                        else self._validate_diagnostic_snapshot(token_ids, snapshot))
            rows = [self._relational_diagnostics(snapshot, layer_index)
                    for layer_index in range(self.cfg.n_layers)]
            return {key: [row[key] for row in rows] for key in rows[0]}
        from vfe3.geometry.transport import build_factored_transport
        from vfe3.families.base import get_family
        from vfe3.free_energy import (self_divergence_for_alpha,
                                      attention_weights, attention_tau)
        from vfe3.alpha_i import self_coupling_alpha
        from vfe3 import metrics

        cfg = self.cfg
        if snapshot is None:
            enc = self.prior_bank.encode(token_ids[:1])                   # (1, N, ...)
            belief = BeliefState(mu=enc.mu[0], sigma=enc.sigma[0], phi=self._apply_pos_phi(enc.phi[0]),
                                 right_phi=self._pos_phi_right(enc.phi[0]))
            n = belief.mu.shape[0]
            rope = self._rope_rotation(n, token_ids.device)
            log_prior = self._attention_log_prior(n, token_ids.device)
        else:
            snapshot = self._validate_diagnostic_snapshot(token_ids, snapshot)
            belief = _sequence_belief(snapshot.initial_belief)
            n = belief.mu.shape[0]
            rope = snapshot.rope
            log_prior = self._first_sequence_log_prior(snapshot.log_prior, token_ids.shape[0])
        fam = get_family(cfg.family)
        _lb = cfg.lambda_beta
        _tau = attention_tau(
            _as_coeff(cfg.kappa_beta, belief.mu.device), self._belief_score_irrep_dims())
        rho, rho_s = cfg.prior_handoff_rho, cfg.prior_handoff_sigma
        mu_p, sigma_p = belief.mu, belief.sigma
        keys = ["self_coupling", "belief_coupling", "attention_entropy", "total", "self_divergence",
                "holonomy_deviation", "holonomy_wilson", "gauge_trace_spread", "gauge_invariant_spread",
                "effective_rank", "attn_entropy", "belief_cond_median", "phi_norm_mean"]
        if log_likelihood is not None:
            keys.append("observation_likelihood")
        rec: dict = {k: [] for k in keys}
        block_mlp_transport, block_mlp_frame = (
            self._block_mlp_frame_transport(belief, rope)
            if snapshot is None else (None, None)
        )
        for layer_index in range(cfg.n_layers):
            if snapshot is None:
                cap: dict = {}
                belief = vfe_block(
                    belief, mu_p, sigma_p, self.group, cfg, log_prior=log_prior,
                    block_norm=self.block_norm, head_mixer=self.head_mixer, cg_coupling=self.cg_coupling,
                    block_mlp=(self.block_mlps[layer_index] if self.block_mlps is not None else None),
                    block_mlp_frame=block_mlp_frame,
                    lambda_beta=cfg.lambda_beta,
                    transport_state=self.transport_state,
                    rope=rope, rope_on_cov=cfg.rope_full_gauge, rope_on_value=cfg.rope_on_value,
                    rope_insertion=cfg.rope_insertion,
                    transport_status=self._transport_status,
                    prebuilt_transport=block_mlp_transport,
                    tau=_tau,
                    capture=cap,
                )
            else:
                prior = snapshot.layer_priors[layer_index]
                mu_p, sigma_p = prior[0][0], prior[1][0]
                belief = _sequence_belief(snapshot.layer_outputs[layer_index])
                cap = {"converged": _sequence_belief(snapshot.layer_converged[layer_index])}
            _tau_c = _tau
            omega = self._diagnostic_transport(belief)
            base_omega = None
            if rope is not None:
                effective = RopeTransport(base=omega, rope=rope, on_cov=cfg.rope_full_gauge,
                                          on_value=cfg.rope_on_value, insertion=cfg.rope_insertion,
                                          same_frame_flat_cocycle=getattr(
                                              omega, "same_frame_flat_cocycle", False))
                mu, sigma = belief.mu, belief.sigma
                batched = False
                if not cfg.rope_on_value:
                    base_omega = omega
            else:
                effective = omega.unsqueeze(0)
                mu, sigma = belief.mu.unsqueeze(0), belief.sigma.unsqueeze(0)
                batched = True
            energy = self._coupling_energy(fam,                       # (N, N) or (H, N, N)
                mu, sigma, mu, sigma, effective,
                log_prior=log_prior,
                alpha=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
                divergence_family=cfg.divergence_family, irrep_dims=self.group.irrep_dims,
            )
            coupling_energy = None
            if base_omega is not None:
                coupling_energy = self._coupling_energy(fam,
                    mu, sigma, mu, sigma, base_omega,
                    log_prior=log_prior,
                    alpha=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
                    divergence_family=cfg.divergence_family,
                    irrep_dims=self.group.irrep_dims,
                )
            if batched:
                energy = energy[0]
            if snapshot is None:
                beta = attention_weights(energy, tau=_tau_c, log_prior=log_prior)
            else:
                beta = snapshot.beta_maps[layer_index]
                if energy.dim() == 2:
                    beta = beta[0]
            _q = cap["converged"]                                    # self-term reads THIS block's prior (per-layer exact)
            self_div = self_divergence_for_alpha(
                self._family_instance(fam, _q.mu, _q.sigma),
                self._family_instance(fam, mu_p, sigma_p),
                alpha=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
                divergence_family=cfg.divergence_family, lambda_alpha_mode=cfg.lambda_alpha_mode,
            )
            alpha, alpha_reg = self_coupling_alpha(
                self_div, mode=cfg.lambda_alpha_mode, value=cfg.lambda_alpha,
                b0=_as_coeff(cfg.b0, belief.mu.device), c0=_as_coeff(cfg.c0, belief.mu.device),
            )
            terms = metrics.free_energy_terms(
                self_div, energy, beta, alpha, tau=_tau_c, lambda_beta=_lb, log_prior=log_prior,
                include_attention_entropy=cfg.include_attention_entropy,
                alpha_reg=(alpha_reg if cfg.lambda_alpha_mode != "constant" else None),
                coupling_energy=coupling_energy,
                log_likelihood=log_likelihood,
            )
            rec["self_coupling"].append(float(terms["self_coupling"]))
            rec["belief_coupling"].append(float(terms["belief_coupling"]))
            rec["attention_entropy"].append(float(terms["attention_entropy"]))
            rec["total"].append(float(terms["total"]))
            if "observation_likelihood" in terms:
                rec["observation_likelihood"].append(float(terms["observation_likelihood"]))
            rec["self_divergence"].append(float(self_div.sum()))
            rec["holonomy_deviation"].append(float(metrics.holonomy_deviation_sampled(omega)["mean"]))
            rec["holonomy_wilson"].append(float(metrics.holonomy_wilson_sampled(omega)["deviation_mean"]))
            rec["gauge_trace_spread"].append(float(
                metrics.gauge_trace_spread(belief.phi, self.group)))
            active_vertex = build_factored_transport(
                belief.phi, self.group,
                exp_fp64_norm_threshold=cfg.exp_fp64_norm_threshold,
                clamp_monitor=cfg.transport_clamp_monitor,
                validity_max_norm=cfg.transport_chart_max_norm,
                right_phi=belief.right_phi,
            ).exp_phi
            active_invariant = metrics.group_gauge_invariant(
                active_vertex, self.group).float()
            rec["gauge_invariant_spread"].append(
                float(active_invariant.std(unbiased=False)))
            _diag = belief.sigma.dim() == belief.mu.dim()
            rec["attn_entropy"].append(float(metrics.attention_entropy(beta)))
            bs = metrics.belief_spectrum(
                belief.sigma,
                diagonal=_diag,
                eps=cfg.eps,
                family=cfg.family,
            )
            rec["effective_rank"].append(float(bs["effective_rank"].mean()))
            rec["belief_cond_median"].append(float(bs["condition"].float().median()))
            rec["phi_norm_mean"].append(
                float(torch.linalg.norm(belief.phi, dim=-1).mean())
            )

            if snapshot is None:
                mu_p = (1.0 - rho) * mu_p + rho * belief.mu          # handoff (mirrors vfe_stack)
                sigma_p = (1.0 - rho_s) * sigma_p + rho_s * belief.sigma
        return rec
