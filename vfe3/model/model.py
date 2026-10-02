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
from vfe3.geometry.lie_ops import CompactBlockElement, project_phi_to_slk
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
from vfe3.model.head_mixer import HeadMixer
from vfe3.model.block_mlp import (
    block_mlp_build_metadata,
    build_block_mlp,
    get_block_mlp_registration,
)
from vfe3.model.block import _as_coeff
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
    set_full_cov_kl_precision,
    validate_full_cov_kl_precision,
)
from vfe3.families.base import BeliefParams
from vfe3.families.policy import family_instance
from vfe3.numerics import safe_eigvalsh, set_mu_trust_cholesky_rounds
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
    edit). Arity 1 -> ``builder(K)``; arity 2 -> ``builder(K, n_heads)`` (block_glk).
    Higher arities are an unsupported registration error.

    ``cfg.group_n`` / ``cfg.irrep_spec`` are forwarded as keywords only when set AND the selected
    builder accepts them (the so_n irrep-tower builder; config validation rejects them for any
    other group)."""
    builder = get_group(cfg.gauge_group)
    arity = _positional_arity(builder)
    params = inspect.signature(builder).parameters
    kwargs: dict = {}
    if cfg.group_n is not None and "group_n" in params:
        kwargs["group_n"] = cfg.group_n
    if cfg.irrep_spec is not None and "irrep_spec" in params:
        kwargs["irrep_spec"] = cfg.irrep_spec
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
    logits:            Optional[torch.Tensor]   # None when built with return_logits=False
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
    transport_chart_max_norm: float,
) -> Dict[str, Optional[float]]:
    """Return the exact active production numerical policy, including disabled bounds."""
    return {
        "m_phi_group_trust_radius": m_phi_group_trust_radius,
        "phi_mstep_max_matrix_norm": phi_mstep_max_matrix_norm,
        "transport_chart_max_norm": transport_chart_max_norm,
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
        # Cholesky and fp32-matmul policies below, so direct full-Gaussian calls and the
        # PriorBank fallback resolve the config-selected policy rather than the module default.
        self.full_cov_kl_precision = validate_full_cov_kl_precision(
            cfg.full_cov_kl_precision)
        # Same one-place-set rule for the KL policy itself (audit 2026-08-06 C1).
        set_full_cov_kl_precision(self.full_cov_kl_precision)
        set_projected_condition_recovery(cfg.projected_cov_condition_recovery)
        set_projected_factor_route(cfg.projected_factor_route)
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
        self.relational_operators_extra = nn.ModuleList()
        # 'depth_scaled' starts the residual-branch writer O at 1/sqrt(2 L) so the L-layer
        # stack of near-uniform routing at init does not add a unit-gain token mean per layer.
        output_init_scale = (1.0 / math.sqrt(2.0 * cfg.n_layers)
                             if cfg.relational_output_init == "depth_scaled" else 1.0)
        layers = [
            RelationalOperators(list(self.group.irrep_dims),
                                _derived_seed(cfg.seed, "relational_operators", layer_index),
                                readout_mode=cfg.relational_readout_mode,
                                attention_residual=cfg.attention_residual,
                                output_init_scale=output_init_scale,
                                receiver_noise=cfg.relational_receiver_noise,
                                receiver_law=cfg.relational_receiver_law,
                                receiver_update=cfg.relational_receiver_update,
                                receiver_damping=cfg.relational_receiver_damping,
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
        self.prior_bank = PriorBank(
            cfg.vocab_size, cfg.embed_dim, n_gen,
            mu_init_std=cfg.mu_init_std, sigma_init=cfg.sigma_init, phi_scale=cfg.phi_scale,
            decode_tau=cfg.decode_tau, eps=cfg.eps,
            decode_av_precision=cfg.decode_av_precision,
            diagonal_covariance=cfg.diagonal_covariance,
            family=cfg.family,
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
            gauge_group_name=self.group.name,
            decode_coordinates=cfg.decode_coordinates,
        )
        # Fixed Gibbs readout temperature (the learned temperature was retired 2026-10-02): the
        # parameter keeps its name (checkpoint layout is unchanged) but leaves autograd and, through
        # build_optimizer's requires_grad filter, the optimizer, so
        # tau_eff = decode_tau * exp(-clamp(decode_log_scale, -3, 3)) stays at its initialized (or
        # loaded) value.
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
        if (cfg.use_block_mlp and self._block_mlp_registration.requires_frame_context
                and cfg.pos_rotation == "rope"):
            rope_caveats = []
            if not cfg.rope_on_value:
                rope_caveats.append(
                    "Values use U_i; the MLP uses U_i R_i^T (position-dependent MLP).")
            if rope_caveats:
                # A build banner remains visible even when Python warnings are filtered.
                print("\n" + "-" * 72 + "\n"
                      "WARNING: canonical_frame + RoPE uses a split positional path.\n"
                      + "\n".join(rope_caveats) + "\n" + "-" * 72 + "\n")
        self.final_norm = get_norm(cfg.norm_type_final)(cfg.embed_dim, eps=cfg.eps,
                                                         family=cfg.family,
                                                         affine=cfg.layernorm_affine) \
            if cfg.norm_type_final != "none" else None
        # layernorm_affine inert warning: the affine gamma/beta exist only on a "layernorm" seam.
        if cfg.layernorm_affine:
            if not (cfg.norm_type_block == "layernorm" or cfg.norm_type_final == "layernorm"):
                import warnings
                warnings.warn(
                    "layernorm_affine=True but neither norm_type_block nor norm_type_final is "
                    "'layernorm': no affine gamma/beta are created and the toggle is inert. Set a "
                    "norm seam to 'layernorm' to use the learned affine.",
                    UserWarning, stacklevel=2,
                )
        # Opt-in Schur-commutant head mixer (default off). Built ONCE from the gauge group's
        # irrep blocks. Label-less groups need >= 2 EQUAL blocks (block_glk);
        # labeled irrep towers (so_n) mix per isotypic component (mults-one towers get
        # per-head scalar gains -- the entire linear commutant there). Bad pairings fail here,
        # not at forward.
        self.head_mixer = HeadMixer(self.group.irrep_dims, family=cfg.family,
                                     irrep_labels=self.group.irrep_labels) \
            if cfg.use_head_mixer else None
        # Opt-in CG cross-type coupling (default off; so_n only). Built ONCE from the
        # group's labels; CGCoupling raises at construction when no admissible paths exist.
        # The algebra key comes from the GROUP OBJECT (set by the so_n builder), not a
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
        # The projected routes run on the flat connection, whose registration has no trainable
        # transport state; the registry keys are kept so ``transport_state`` stays the one seam.
        transport_registration = get_transport_registration("flat")
        self._transport_state_keys = transport_registration.serialization_keys
        self._transport_offdiag_state_keys = transport_registration.offdiag_serialization_keys
        if transport_registration.state_builder is not None:
            raise ValueError("the flat transport registration must not build trainable state")
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
        # gauge_transport='frozen'/'off' DEFINE the frame as non-learning: config.__post_init__
        # coerces m_phi_lr=0.0 and the projected E-step never moves phi, so neither inference nor
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
    @property
    def relational_operators_by_layer(self) -> Tuple[RelationalOperators, ...]:
        """Return independent layer projections without registering duplicate modules."""
        return (self.relational_operators, *self.relational_operators_extra)

    @property
    def conditional_capture_key(self) -> str:
        return "relational_capture"

    @property
    def conditional_semantics(self) -> str:
        return self.relational_operators.inference_semantics

    def _belief_score_irrep_dims(self) -> Sequence[int]:
        r"""Return the ambient block sizes the beta score is computed in."""
        return self.group.irrep_dims

    def _family_instance(
        self, family: type[BeliefParams], mu: torch.Tensor, sigma: torch.Tensor,
    ) -> BeliefParams:
        """Construct a model-owned Gaussian with this model's immutable policy."""
        return family_instance(mu, sigma, family, self.full_cov_kl_precision)

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
        return self.group.phi_coordinate_layout == "block_head_row_major"

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

    def _relational_frames(
        self,
        beliefs: BeliefState,
        rope: Optional[torch.Tensor],
        *,
        _content_token_ids: Optional[torch.Tensor] = None,
    ) -> 'tuple[CompactFactoredTransport | FactoredTransport | RopeTransport, CanonicalFrameContext, CanonicalFrameContext]':
        r"""Build the shared flat transport and the realized routing and value frames.

        The routing frame folds RoPE into the vertex factors, :math:`U_i R_i^\top`; the value frame
        is the same unless ``rope_on_value=False``, which keeps the un-rotated :math:`U_i`. Both are
        compact head blocks: :math:`\xi_i = U_i^{-1}\mu_i` and
        :math:`C_i = U_i^{-1}\Sigma_i U_i^{-\top}` are the reference moments the projected scorers read.
        """
        shared_omega = self._build_shared_flat_transport(
            beliefs, rope, require_vertex_factors=True, _content_token_ids=_content_token_ids)
        frame = self._canonical_frame_context(shared_omega, fold_rope=True, compact=True)
        value_frame = (
            self._canonical_frame_context(shared_omega, fold_rope=False, compact=True)
            if rope is not None and not self.cfg.rope_on_value else frame)
        return shared_omega, frame, value_frame

    def _build_shared_flat_transport(
        self,
        beliefs: BeliefState,
        rope: Optional[torch.Tensor],
        *,
        require_vertex_factors: bool,
        _content_token_ids: Optional[torch.Tensor] = None,
    ) -> 'torch.Tensor | CompactFactoredTransport | FactoredTransport | RopeTransport':
        r"""Build the one flat transport shared by the realized frames and inference."""
        from vfe3.inference.e_step import build_belief_transport

        transport = build_belief_transport(
            beliefs.phi,
            self.group,
            right_phi=beliefs.right_phi,
            compact_phi_block_transport=self._compact_phi_blocks_enabled(),
            require_vertex_factors=require_vertex_factors,
            validity_max_norm=self.cfg.transport_chart_max_norm,
            exactness_out=self._transport_status,
            _content_token_ids=_content_token_ids,
            rope=rope,
            rope_on_value=self.cfg.rope_on_value,
            rope_insertion=self.cfg.rope_insertion,
        )
        return transport

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

        This seam carries the same ``amp`` context forward establishes, so a grad-enabled training
        caller and a no-grad inference caller both get the identical forward value. The no-grad property used by
        generation comes from the caller's ``@torch.no_grad`` scope, not from this method.
        """
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
        # Opt-in mixed precision (cfg.amp_dtype): wrap the E-step / belief pipeline in autocast for
        # CUDA throughput. amp_dtype=None (default) -> nullcontext -> NO autocast object is ever
        # instantiated on the default path, so logits/loss are byte-identical to the no-AMP build.
        # device_type is resolved from the runtime tensors (not hardcoded 'cuda') so the path is
        # exercised on whatever device the tokens live on. The outer autocast remains active through
        # transport construction, model-channel refinement, and vfe_stack; only the existing
        # matrix-exp/SPD kernels and the projected scorers enter their own narrow fp32 islands. Decode + CE are protected separately below (their inputs are
        # .float()-ed; see _amp_off_context).
        amp = self._amp_context(token_ids.device)
        with amp:
            relational_capture = {} if capture is not None else None
            if relational_capture is not None:
                capture[self.conditional_capture_key] = relational_capture
            canonical_frame: Optional[CanonicalFrameContext] = None
            block_mlp_frame: Optional[CanonicalFrameContext] = None
            # Without a RoPE wrapper the private loss path may reuse per-content vertex factors.
            content_token_ids = (
                token_ids if (
                    rope is None
                    and type(self) is VFEModel
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
                ) else None)
            shared_omega, relational_frame, relational_value_frame = self._relational_frames(
                beliefs, rope, _content_token_ids=content_token_ids)
            if self.cfg.decode_coordinates == "reference":
                # The covariant decoder reads (xi, C) through the frame the attention reads.
                canonical_frame = relational_frame
                if capture is not None:
                    capture["canonical_frame"] = canonical_frame
                if _decode_frame_out is not None:
                    _decode_frame_out["frame"] = canonical_frame
            if (
                self.executable_build.block_mlp.enabled
                and self._block_mlp_registration.requires_frame_context
            ):
                block_mlp_frame = self._canonical_frame_context(shared_omega, fold_rope=True)
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
                   or self.cfg.cg_energy_weight > 0.0) else None
        )
        if cap is not None and self.cfg.cg_energy_weight > 0.0:
            # Initialize the CG moment-energy lists ONLY when the regularizer is on. A capture
            # allocated solely for M-step self-coupling leaves these keys absent, so vfe_block never
            # appends to (or stacks) a CG list (PB-13).
            cap["cg_moment_energy_rows"] = []
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
            # CONVERGED output phi -> backprops to the learned prior table phi_embed. The projected
            # routes run no inner phi E-step, so this is the only place mass_phi enters.
            # D-08: the penalty is the per-position squared norm ||phi_i||^2 = (phi ** 2).sum(dim=-1),
            # averaged over positions to match the per-token mean that `ce` already is. Latent at
            # the 0.0 default (no shipped config changes), material on a mass_phi sweep.
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
                kl_max=cfg.kl_max, eps=cfg.eps, lambda_alpha_mode=cfg.lambda_alpha_mode,
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
            # q/p/s/h total. Each block already appended its grad-connected D(q_post||q_pre) rows.
            # The token-then-layer mean (each row.mean() over tokens, then
            # mean over layers) is weighted and added; the per-layer values and the assembled
            # objective_total_with_cg are reported on the diagnostic side channel only.
            cfg = self.cfg
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
        r"""Build the flat diagnostic transport without materializing compact block pairs."""
        from vfe3.inference.e_step import _transport, build_belief_transport

        cfg = self.cfg
        kwargs = {
            "transport_state": self.transport_state,
            "validity_max_norm": cfg.transport_chart_max_norm,
            "exactness_out": self._transport_status,
            "right_phi": belief.right_phi,
        }
        if self._compact_phi_blocks_enabled():
            return build_belief_transport(
                belief.phi,
                self.group,
                compact_phi_block_transport=True,
                **kwargs,
            )
        return _transport(belief.phi, self.group, **kwargs)

    @torch.no_grad()
    @_block_mlp_diagnostic_eval
    def build_diagnostic_snapshot(
        self,
        token_ids: torch.Tensor,                 # (B, N) integer token ids
        *,
        _share_frame_copies: bool = False,        # private read-only validation consumers only
        return_logits:       bool = True,         # False skips the (B, N, V) decode (logits is None)
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
            token_ids, return_logits=return_logits, capture=capture)
        if return_logits and logits is None:
            raise RuntimeError("diagnostic snapshot requested logits but the decoder returned None")

        log_prior = diagnostic["log_prior"]
        # Autocast can leave captured head-mixer outputs in bf16/fp16. Diagnostic transport is
        # intentionally rebuilt outside autocast so matrix factorizations and condition-number
        # probes remain fp32. Freeze and upcast the captured boundary before any replay contraction;
        # otherwise compact transport combines fp32 frame factors with a low-precision belief.
        layer_outputs = tuple(_freeze_belief(layer, _frame_cache=frame_cache) for layer in diagnostic["layer_outputs"])
        stack_output = _freeze_belief(capture["out"], _frame_cache=frame_cache)
        relational_capture = capture.get(self.conditional_capture_key)
        if relational_capture is None:
            raise RuntimeError("diagnostic snapshot requires the relational forward capture")
        layer_betas = [cap["beta_pre"][0] for cap in diagnostic["layer_conditional_captures"]]
        beta_maps = torch.stack([beta if beta.dim() == 3 else beta.unsqueeze(0)
                                 for beta in layer_betas])
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
            logits=_freeze_tensor(logits) if logits is not None else None,
            beta_maps=_freeze_tensor(beta_maps),
            trace_states=tuple(_freeze_belief(belief, _frame_cache=frame_cache) for belief in trace["beliefs"]),
            trace_free_energy=_freeze_tensor(trace_free_energy),
            **{self.conditional_capture_key: {
                key: _freeze_tensor(value) if isinstance(value, torch.Tensor) else value
                for key, value in relational_capture.items()}},
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
            # Capped fraction over the routable support (beta > 0), not over masked pairs.
            relational_score_klmax_frac=float(
                ((score >= self.cfg.kl_max) & (beta > 0)).sum() / (beta > 0).sum().clamp(min=1)),
        )
        from vfe3 import metrics
        d["covariance_available"] = 1.0
        d["belief_cond_median"] = float("nan")
        spectrum = metrics.belief_spectrum(
            snapshot.layer_converged[layer_index].sigma, diagonal=self.cfg.diagonal_covariance,
            eps=self.cfg.eps, family=self.cfg.family)
        d["effective_rank"] = float(spectrum["effective_rank"].mean())
        d["belief_cond_median"] = float(spectrum["condition"].median())
        d["relational_mean_displacement_norm"] = float(
            cap.get("relational_mean_displacement_norm", float("nan")))
        if self.cfg.attention_mode == "projected_gradient":
            # These are the variable terms of the frozen receiver energy; no global-F
            # or comparison-router stationarity is implied by their difference.
            for name in ("relational_receiver_cost_before", "relational_receiver_cost_after"):
                d[name] = float(cap.get(name, float("nan")))
            d["relational_receiver_cost_delta"] = (
                d["relational_receiver_cost_after"] - d["relational_receiver_cost_before"])
            # The total delta can fall while the mean moves away from its minimizer; report the
            # mean and covariance parts and the distance to mu* = Lambda^{-1} eta separately.
            for part in ("mean", "covariance"):
                before = float(cap.get(f"relational_receiver_{part}_cost_before", float("nan")))
                after = float(cap.get(f"relational_receiver_{part}_cost_after", float("nan")))
                d[f"relational_receiver_{part}_cost_delta"] = after - before
            distance_before = float(cap.get("relational_receiver_target_distance_before", float("nan")))
            distance_after = float(cap.get("relational_receiver_target_distance_after", float("nan")))
            d["relational_receiver_target_distance_ratio"] = (
                distance_after / distance_before if distance_before > 0 else float("nan"))
            d["relational_receiver_noise"] = float(self.cfg.relational_receiver_noise)
            d["relational_attention_capture"] = "episode_initial"
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
        d["relational_parameter_count"] = float(sum(
            p.numel() for layer in self.relational_operators_by_layer for p in layer.parameters()))
        T, O = ops.content_factors()
        # sigma_max(M_h) is basis-dependent (changes under (T, O) -> (T a, a^{-1} O)); rho = inf of sigma_max over similarity.
        singular = torch.linalg.svdvals(O.float() @ T.float())
        d["relational_content_max_singular_value"] = float(singular.max())
        d["relational_content_effective_rank"] = float(
            torch.linalg.matrix_rank(O.float() @ T.float()).float().mean())
        # Similarity invariants of the content links L_ij^h = U_i M_h U_j^{-1}, M_h = O_hat_h T_hat_h:
        # unchanged by the frames, by (T_h, O_h) -> (B T_h, O_h B^{-1}) and by (T, O) -> (T a, a^{-1} O).
        # relational_content_max_singular_value above is not invariant under the reference change.
        with torch.autocast(device_type=ops.T.device.type, enabled=False):
            T_hat, O_hat = ops._ambient_content_maps(torch.float64)          # (H, d, K), (H, K, d)
        d.update(metrics.content_link_spectrum(
            T_hat, O_hat,
            beta_self=(beta.diagonal(dim1=-2, dim2=-1)                      # (B, H, N) beta_ii^h
                       if ops.readout_mode == "predictive_content" else None),
            residual=ops.attention_residual))
        covariance = torch.stack((cap["projected_covariance_pre"],
                                  cap["projected_key_covariance_pre"])).float()
        eig = safe_eigvalsh(covariance)
        d["relational_projected_cov_min_eigenvalue"] = float(eig.min())
        d["relational_projected_cov_max_condition"] = float((eig[..., -1]/eig[..., 0]).max())
        d["relational_covariance_repair_count"] = float(cap.get("relational_covariance_repair_count", 0.0))
        # Frame magnitude and decoder/head-module state remain meaningful on both readouts.
        belief = snapshot.layer_converged[layer_index]
        phi_norm = torch.linalg.norm(belief.phi, dim=-1)
        d["phi_norm_mean"] = float(phi_norm.mean())
        d["phi_norm_std"] = float(phi_norm.std(unbiased=False))
        matrix_norm = metrics.diagnostic_frame_norm(
            belief.phi.float(), self.group, max_norm=self.cfg.transport_chart_max_norm,
            eps=self.cfg.eps)
        d["phi_matrix_norm_median"] = float(matrix_norm.median())
        d["phi_matrix_norm_p95"] = float(torch.quantile(matrix_norm, 0.95))
        d["phi_matrix_norm_p99"] = float(torch.quantile(matrix_norm, 0.99))
        d["phi_matrix_norm_max"] = float(matrix_norm.max())
        if self.head_mixer is not None and hasattr(self.head_mixer, "mixer_deltas"):
            d["head_mixer_drift"] = max(
                (float(torch.linalg.norm(p.detach())) for p in self.head_mixer.mixer_deltas), default=0.0)
        if hasattr(self.prior_bank, "head_evidence_logits"):
            weights, _ = self.prior_bank.head_evidence_weights(
                dtype=self.prior_bank.head_evidence_logits.dtype,
                device=self.prior_bank.head_evidence_logits.device)
            probabilities = weights / weights.numel()
            d["head_evidence_weights"] = weights.detach().cpu().tolist()
            d["head_evidence_entropy"] = float(torch.special.entr(probabilities).sum())
            d["head_evidence_max_abs_drift"] = float((weights - 1.0).abs().max())
        return d

    @torch.no_grad()
    @_block_mlp_diagnostic_eval
    def diagnostics(
        self,
        token_ids: torch.Tensor,           # (B, N) token ids; projected diagnostics reduce the captured batch

        *,
        log_likelihood: Optional[torch.Tensor] = None,  # (N,) optional E_q[log p(o|k)] diagnostic seam
        snapshot:       Optional[DiagnosticSnapshot] = None,  # one-forward captured evaluation state
    ) -> dict:
        r"""Report the captured projected routing, receiver costs, and state diagnostics (no_grad).

        Reads one diagnostic snapshot (built here unless supplied) and reduces the captured batch
        through :meth:`_relational_diagnostics` at the last layer. Routing and readout do not
        define one shared stationary objective, so the legacy global free-energy fields
        (``self_coupling``, ``belief_coupling``, ``attention_entropy``, ``total``, ...) are NaN.
        ``log_likelihood`` is accepted for call-site compatibility and unused.
        """
        snapshot = (self.build_diagnostic_snapshot(token_ids, return_logits=False) if snapshot is None
                    else self._validate_diagnostic_snapshot(token_ids, snapshot))
        return self._relational_diagnostics(snapshot)

    @torch.no_grad()
    @_block_mlp_diagnostic_eval
    def attention_maps(
        self,
        token_ids: torch.Tensor,           # (B, N) token ids; only sequence 0 is used

        *,
        score_at:  str                          = "converged",   # "converged" | "entry" | "prior"
        snapshot:  Optional[DiagnosticSnapshot] = None,
    ) -> torch.Tensor:                     # (L, H, N, N) per-layer, per-head attention beta_ij
        r"""Per-layer, per-head routing weights ``beta_ij`` for sequence 0, shape ``(L, H, N, N)`` (no_grad).

        Projected attention computes beta once, from the belief entering each block, before the
        receiver update; the snapshot stores exactly that pre-update router. ``score_at`` in
        ``("converged", "entry")`` therefore returns the same captured maps (a converged re-score
        is not part of the projected forward model), and ``"prior"`` returns
        ``softmax_j(log_prior)``, the positional reference without content.
        """
        # Projected routes return the captured router of sequence 0. diagnostics() and
        # diagnostics_per_layer() average over the whole batch, so the two agree for a
        # single-sequence batch (audit 2026-10-01 F5).
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

    @torch.no_grad()
    @_block_mlp_diagnostic_eval
    def diagnostics_per_layer(
        self,
        token_ids: torch.Tensor,           # (B, N) token ids; projected diagnostics reduce the captured batch

        *,
        log_likelihood: Optional[torch.Tensor] = None,  # (N,) optional E_q[log p(o|k)] diagnostic seam
        snapshot:       Optional[DiagnosticSnapshot] = None,
    ) -> dict:                             # each value a list of length L = cfg.n_layers
        r"""Per-layer projected diagnostics: :meth:`_relational_diagnostics` at every layer (no_grad).

        Returns a dict of ``L``-length lists keyed like :meth:`diagnostics`, from one snapshot.
        ``log_likelihood`` is accepted for call-site compatibility and unused.
        """
        snapshot = (self.build_diagnostic_snapshot(token_ids, return_logits=False) if snapshot is None
                    else self._validate_diagnostic_snapshot(token_ids, snapshot))
        rows = [self._relational_diagnostics(snapshot, layer_index)
                for layer_index in range(self.cfg.n_layers)]
        return {key: [row[key] for row in rows] for key in rows[0]}
