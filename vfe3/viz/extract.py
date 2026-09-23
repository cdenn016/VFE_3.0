r"""Belief / evaluation EXTRACTORS for VFE_3.0 publication figures.

These RUN the model -- reload-and-recompute converged beliefs, loop the E-step capturing the
belief trajectory, replay the block stack per layer, score a loader per unit, or tally numerical
health. They have side effects / drive the model, so they live here rather than in
``vfe3.metrics`` (whose contract is pure, side-effect-free measurement). Each returns plain
tensors / dicts that the pure metrics and the figure functions consume. Everything runs under
``torch.no_grad`` and OFF the training hot path.
"""

from typing import TYPE_CHECKING, Dict, Iterable, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F

from vfe3 import metrics
from vfe3.contracts import is_conditional_filtering
from vfe3.alpha_i import self_coupling_alpha
from vfe3.belief import BeliefState
from vfe3.contracts import CanonicalFrameContext
from vfe3.families.base import get_family, kl
from vfe3.model.block import _as_coeff, e_step_shared_kwargs   # shared cfg->kwargs bag (audit 2026-07-12 N5)
from vfe3.model.model import _block_mlp_diagnostic_eval
from vfe3.free_energy import (
    attention_tau,
    attention_weights,
    self_divergence_for_alpha,
)
from vfe3.inference.e_step import (
    canonical_e_step_update,
    e_step_iteration,
    free_energy_value,
)
from vfe3.geometry.lie_ops import CompactBlockElement
from vfe3.model.block import vfe_block

if TYPE_CHECKING:
    from vfe3.model.model import DiagnosticSnapshot


def _model_device(model) -> torch.device:
    return next(model.parameters()).device


def _require_legacy_inference_diagnostic(model: torch.nn.Module, diagnostic: str) -> None:
    """Reject reconstruction probes that assume one stationary shared-Q/K/V objective."""
    if is_conditional_filtering(model):
        raise NotImplementedError(
            f"{diagnostic} is unavailable for relational conditional filtering; "
            "use the captured pre-update routing and fixed-context conditional diagnostics"
        )


def _cpu_bank_tensor(value: torch.Tensor) -> torch.Tensor:
    r"""Detach and CPU-host one tensor, widening a reduced-precision float to float32.

    The viz sibling of ``model._freeze_tensor``: ``collect_inference_bank`` captures the belief
    straight out of the autocast E-step, so under ``cfg.amp_dtype`` the retained ``mu``/``sigma``
    arrive in bf16/fp16 (``phi`` keeps its own fp32 gauge island). The analysis layer downstream is
    fp32 -- NumPy has NO bfloat16 dtype, and ``torch.linalg.eigh`` has no bfloat16 kernel -- so the
    bank is the boundary that must widen. bf16/fp16 -> fp32 is exact (both are truncations of
    float32), so this preserves the captured values. Integer and float32 fields pass through
    untouched, keeping the amp_dtype=None default path byte-identical.
    """
    hosted = value.detach().cpu()
    if hosted.dtype in (torch.float16, torch.bfloat16):
        hosted = hosted.float()
    return hosted


def _cpu_bank_value(value: object) -> object:
    r"""Detach and CPU-host every tensor retained by a bank record or field."""
    if isinstance(value, torch.Tensor):
        return _cpu_bank_tensor(value)
    if isinstance(value, CanonicalFrameContext):
        return CanonicalFrameContext(
            forward=_cpu_bank_tensor(value.forward),
            inverse=_cpu_bank_tensor(value.inverse),
        )
    if isinstance(value, CompactBlockElement):
        return CompactBlockElement(
            _cpu_bank_tensor(value.blocks),
            value.K,
            value.tied,
        )
    if isinstance(value, BeliefState):
        return BeliefState(
            mu=_cpu_bank_value(value.mu),
            sigma=_cpu_bank_value(value.sigma),
            phi=_cpu_bank_value(value.phi),
            right_phi=(_cpu_bank_value(value.right_phi)
                       if value.right_phi is not None else None),
        )
    return value


def _inference_bank_tensors(value: object) -> Iterator[torch.Tensor]:
    """Yield tensors nested in the stable list-of-records inference-bank API."""
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, CanonicalFrameContext):
        yield value.forward
        yield value.inverse
    elif isinstance(value, CompactBlockElement):
        yield value.blocks
    elif isinstance(value, dict):
        for item in value.values():
            yield from _inference_bank_tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _inference_bank_tensors(item)


def inference_bank_nbytes(records: List[Dict[str, object]]) -> int:
    r"""Return aggregate bytes of unique tensor storages retained by an inference bank."""
    total = 0
    seen: set[tuple[str, Optional[int], int]] = set()
    for tensor in _inference_bank_tensors(records):
        storage = tensor.untyped_storage()
        key = (tensor.device.type, tensor.device.index, storage.data_ptr())
        if key in seen:
            continue
        seen.add(key)
        total += storage.nbytes()
    return total


def _validate_bank_caps(
    *,
    max_tokens:    Optional[int],
    max_sequences: Optional[int],
) -> None:
    """Reject ambiguous or empty belief-bank population requests."""
    if max_tokens is not None and max_sequences is not None:
        raise ValueError("max_tokens and max_sequences are mutually exclusive")
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if max_sequences is not None and max_sequences <= 0:
        raise ValueError("max_sequences must be positive")


def _slice_bank_to_cap(
    bank: Dict[str, torch.Tensor],

    *,
    max_tokens:    Optional[int],
    max_sequences: Optional[int],
) -> Dict[str, torch.Tensor]:
    """Slice every token-aligned bank field to the requested exact population."""
    if max_tokens is not None:
        limit = min(max_tokens, bank["token_ids"].shape[0])
    elif max_sequences is not None:
        limit = int((bank["seq_idx"] < max_sequences).sum().item())
    else:
        return bank
    return {key: value[:limit] for key, value in bank.items()}


def _inference_record_valid_positions(
    tokens:  torch.Tensor,
    targets: Optional[torch.Tensor],
) -> torch.Tensor:
    r"""Return the token-position mask carried by an inference record's targets."""
    if targets is None:
        return torch.ones_like(tokens, dtype=torch.bool)
    if targets.shape != tokens.shape:
        raise ValueError(
            "inference record targets must have the same shape as tokens, got "
            f"{tuple(targets.shape)} and {tuple(tokens.shape)}"
        )
    return targets.to(tokens.device) != -100


def _snapshot_sequence(belief: BeliefState, index: int = 0) -> BeliefState:
    r"""Select one batch row while preserving every optional belief/frame channel."""
    return belief._replace(
        mu=belief.mu[index], sigma=belief.sigma[index], phi=belief.phi[index],
        right_phi=(belief.right_phi[index]
                   if belief.right_phi is not None and belief.right_phi.dim() == belief.phi.dim()
                   else belief.right_phi),
    )


def _encode_one(model, token_ids: torch.Tensor) -> Tuple[BeliefState, torch.Tensor, Optional[torch.Tensor]]:
    r"""Encode sequence 0 to the initial belief (pos_phi applied), with its log-prior and RoPE.

    Mirrors ``forward``'s encode seam for the extracted trajectory/figure consumers: the encoded
    belief is the prior the belief E-step starts from, so every figure describes the model that
    actually trained."""
    enc = model.prior_bank.encode(token_ids[:1])                  # (1, N, ...)
    beliefs = enc._replace(
        phi=model._apply_pos_phi(enc.phi),
        right_phi=model._pos_phi_right(enc.phi),
    )
    n = beliefs.mu.shape[1]
    rope = model._rope_rotation(n, token_ids.device)
    if model.cfg.encode_mode == "canonical_content_projected":
        beliefs, _, _ = model._prepare_projected_canonical_content(beliefs, rope)
    belief = _snapshot_sequence(beliefs)
    log_prior = model._attention_log_prior(n, token_ids.device)
    return belief, log_prior, rope                                    # forward's beliefs.sigma (model.py:762)


@torch.no_grad()
def collect_inference_bank(
    model,
    loader: Iterable,

    *,
    return_logits: bool                   = False,
    max_batches:  Optional[int]           = None,
    device:       Optional[torch.device]  = None,
) -> List[Dict[str, object]]:
    r"""Capture one authoritative belief inference per population batch for report reuse.

    Each record retains CPU-hosted input/target tensors, the post-final-norm converged belief, raw
    stack output, and optional decode logits. Complete CPU offload bounds accelerator residency to
    the current inference or consumer batch. Report callers omit logits and stream decode consumers
    from the retained converged belief, avoiding both E-step replay and population-wide
    full-vocabulary retention.
    """
    device = device or _model_device(model)
    was_training = model.training
    model.eval()
    records: List[Dict[str, object]] = []
    try:
        for i, batch in enumerate(loader):
            if isinstance(batch, (tuple, list)):
                tokens = batch[0]
                targets = batch[1] if len(batch) > 1 else None
            else:
                tokens = batch
                targets = None
            tokens = tokens.to(device)
            targets = targets.to(device) if targets is not None else None
            capture: dict = {}
            beliefs, logits = model.forward_beliefs(
                tokens,
                return_logits=return_logits,
                capture=capture,
            )
            out = capture["out"]
            canonical_frame = capture.get("canonical_frame")
            out_cpu = _cpu_bank_value(out)
            belief_mu_cpu = (
                out_cpu.mu
                if beliefs.mu is out.mu
                else _cpu_bank_value(beliefs.mu)
            )
            belief_cpu = out_cpu._replace(mu=belief_mu_cpu)
            records.append({
                "tokens": _cpu_bank_value(tokens),
                "targets": _cpu_bank_value(targets) if targets is not None else None,
                "belief": belief_cpu,
                "out": out_cpu,
                "logits": _cpu_bank_value(logits) if logits is not None else None,
                "canonical_frame": (
                    _cpu_bank_value(canonical_frame)
                    if canonical_frame is not None else None
                ),
            })
            stop = max_batches is not None and i + 1 >= max_batches
            # The record now owns detached CPU copies. Release every accelerator-side inference
            # workset before the next model forward instead of relying on the loop variable overwrite.
            del batch, tokens, targets, capture, beliefs, logits, out
            del canonical_frame, out_cpu, belief_mu_cpu, belief_cpu
            if stop:
                break
    finally:
        if was_training:
            model.train()
    return records


def _decode_inference_record(
    model,
    record: Dict[str, object],
    device: torch.device,
) -> torch.Tensor:
    r"""Move or reconstruct one record's logits without retaining a population-wide logit bank."""
    stored_logits = record.get("logits")
    if isinstance(stored_logits, torch.Tensor):
        return stored_logits.to(device)
    belief = record.get("belief")
    if not isinstance(belief, BeliefState):
        raise ValueError(
            "inference records require decoded logits or a converged belief"
        )
    mu = belief.mu.to(device)
    sigma = belief.sigma.to(device)
    canonical_frame = record.get("canonical_frame")
    if canonical_frame is not None:
        if not isinstance(canonical_frame, CanonicalFrameContext):
            raise ValueError(
                "inference record canonical_frame must be a CanonicalFrameContext"
            )
        canonical_frame = CanonicalFrameContext(
            forward=canonical_frame.forward.to(device),
            inverse=canonical_frame.inverse.to(device),
        )
    logits = model._decode_belief_with_context(
        mu,
        sigma,
        canonical_frame=canonical_frame,
    )
    del mu, sigma, canonical_frame
    return logits


def _iter_kwargs(model, log_prior: torch.Tensor, rope: Optional[torch.Tensor]) -> dict:
    r"""The full ``e_step_iteration`` knob bag: the production shared cfg bag
    (``e_step_shared_kwargs``, audit 2026-07-12 N5 -- previously a hand-rolled copy that silently
    dropped ``e_step_update``/``mm_damping``/``skip_belief_sigma_update``) plus
    the runtime extras ``vfe_block``/``e_step`` bind per call (tau, step sizes, connections,
    active per-head transport contraction, log_prior, rope)."""
    _require_legacy_inference_diagnostic(model, "_iter_kwargs")
    cfg = model.cfg
    kw = e_step_shared_kwargs(cfg, _model_device(model))
    kw.update(
        tau=attention_tau(
            _as_coeff(cfg.kappa_beta, _model_device(model)), model._belief_score_irrep_dims()),
        e_q_mu_lr=cfg.e_q_mu_lr, e_q_sigma_lr=cfg.e_q_sigma_lr, e_phi_lr=cfg.e_phi_lr,
        lambda_beta=cfg.lambda_beta,
        transport_state=model.transport_state,
        compact_phi_block_transport=model._compact_phi_blocks_enabled(),
        transport_mean_per_head=True,
        congruence_cond_escalation=cfg.congruence_cond_escalation,
        exp_fp64_norm_threshold=cfg.exp_fp64_norm_threshold,
        transport_chart_max_norm=cfg.transport_chart_max_norm,
        transport_status=model._transport_status,
        log_prior=log_prior,
        rope=rope, rope_on_cov=cfg.rope_full_gauge, rope_on_value=cfg.rope_on_value,
        rope_insertion=cfg.rope_insertion,
    )
    return kw


def _fe_kwargs(model, log_prior: torch.Tensor, rope: Optional[torch.Tensor] = None) -> dict:
    r"""The ``free_energy_value`` knob bag: the SAME shared cfg bag production's ``e_step``
    forwards to its diagnostic F (``free_energy_value`` declares the iteration-only knobs as
    explicit accept-and-ignore parameters, audit 2026-07-12 N5), plus
    the runtime extras. ``rope`` is honored (audit PP6): the logged F carries the RoPE-wrapped
    transport. The step-size knobs (e_q_mu_lr/e_q_sigma_lr/e_phi_lr) stay off this bag --
    ``free_energy_value`` rejects them, exactly as production binds them on ``e_step`` only."""
    _require_legacy_inference_diagnostic(model, "_fe_kwargs")
    cfg = model.cfg
    kw = e_step_shared_kwargs(cfg, _model_device(model))
    kw.update(
        tau=attention_tau(
            _as_coeff(cfg.kappa_beta, _model_device(model)), model._belief_score_irrep_dims()),
        lambda_beta=cfg.lambda_beta,
        transport_state=model.transport_state,
        compact_phi_block_transport=model._compact_phi_blocks_enabled(),
        transport_mean_per_head=True,
        congruence_cond_escalation=cfg.congruence_cond_escalation,
        transport_chart_max_norm=cfg.transport_chart_max_norm,
        transport_status=model._transport_status,
        log_prior=log_prior,
        rope=rope, rope_on_cov=cfg.rope_full_gauge, rope_on_value=cfg.rope_on_value,
        rope_insertion=cfg.rope_insertion,
    )
    return kw


@torch.no_grad()
def per_unit_eval_nats(
    model,
    loader:      Iterable[Tuple[torch.Tensor, torch.Tensor]],   # yields (tokens, targets) batches

    *,
    device:      Optional[torch.device] = None,
    max_batches: Optional[int]          = None,
) -> Dict[str, torch.Tensor]:
    r"""Per-SEQUENCE and per-TOKEN cross-entropy (nats) over a loader.

    The training ``evaluate`` retains only the aggregate token-weighted
    {ce, ppl, bits_per_token, bpc}; the
    single-seed bootstrap bands (``metrics.bootstrap_ce_band`` over sequences,
    ``metrics.bootstrap_token_ce_band`` over tokens) need the per-unit nats this produces. Runs
    ``model(tokens)`` for full logits (the inference path; the fused chunked decode is bypassed by
    passing no targets) and reduces the cross-entropy with ``reduction='none'``.

    Returns ``per_seq_nats`` (S,) summed nats per sequence, ``per_seq_tokens`` (S,) the
    non-ignored token count per sequence, and ``per_token_nats`` (M,) the valid per-position nats.
    """
    device = device or _model_device(model)
    was_training = model.training
    model.eval()
    seq_nats: List[torch.Tensor] = []
    seq_tok: List[torch.Tensor] = []
    tok_nats: List[torch.Tensor] = []
    try:
        for i, (tokens, targets) in enumerate(loader):
            tokens = tokens.to(device)
            targets = targets.to(device)
            b, n = tokens.shape
            logits = model(tokens)                                # (B, N, V) inference path
            v = logits.shape[-1]
            per = F.cross_entropy(logits.reshape(-1, v).float(), targets.reshape(-1),
                                  ignore_index=-100, reduction="none").reshape(b, n)
            valid = (targets != -100)
            seq_nats.append((per * valid).sum(dim=1))
            seq_tok.append(valid.sum(dim=1).to(per.dtype))
            tok_nats.append(per[valid])
            if max_batches is not None and i + 1 >= max_batches:
                break
    finally:
        if was_training:
            model.train()
    return {
        "per_seq_nats":   torch.cat(seq_nats),
        "per_seq_tokens": torch.cat(seq_tok),
        "per_token_nats": torch.cat(tok_nats),
    }


@torch.no_grad()
def belief_ce_bank(
    model,
    loader:      Iterable[Tuple[torch.Tensor, torch.Tensor]],   # yields (tokens, targets) batches

    *,
    device:         Optional[torch.device]              = None,
    max_batches:    Optional[int]                       = None,
    inference_bank: 'Optional[List[Dict[str, object]]]' = None,
) -> Dict[str, torch.Tensor]:
    r"""The Sigma_q <-> CE JOIN for the calibration probe (B1/EXP-3): per-token belief-covariance
    trace tr(Sigma_q) aligned with the decode cross-entropy that same belief produced.

    For each ``(tokens, targets)`` batch this REPLAYS model.forward's belief path EXACTLY
    (``prior_bank.encode`` -> pos_phi -> the precision-bias fold -> ``vfe_stack`` with the trained
    head_mixer / cg_coupling / block_norm) for the per-token covariance ``out.sigma``, AND the
    inference forward ``model(tokens)`` for the decode logits, then reduces the cross-entropy with
    ``reduction='none'``. The replay is applied here (and in ``belief_bank`` too) so the traced
    Sigma_q IS the belief whose mean produced the logits -- otherwise tr(Sigma_q) and CE would come
    from different beliefs and the calibration headline reads off the wrong covariance. The belief at
    position n is the one whose mean produced the logits predicting target n, so tr(Sigma_q)[n] and
    CE[n] are aligned position-by-position; only valid targets (``targets != -100``) are kept.
    Returns the flattened per-token ``tr_sigma`` (M,), ``ce`` (M,) nats, the predicted-token
    ``token_ids`` (M,), and the per-token ``conf`` (M,) max-softmax probability and ``correct`` (M,)
    argmax-equals-gold indicator -- the aligned pairs the Spearman rho / CV gate
    (``vfe3.metrics.spearman_rho`` / ``cv``) and the Sigma-stratified-error / Sigma-CE-scatter
    figures consume. ``max_batches`` caps the join.

    When ``metrics.covariance_frozen_at_encode(model.cfg)`` holds (projected routes, or
    ``skip_belief_sigma_update``), ``out.sigma`` is the encode lookup, so ``tr_sigma`` is the input
    token's table variance tr Sigma(x_i) and the consumers label it that way.
    """
    device = device or _model_device(model)
    if inference_bank is not None:
        tr_sig, ces, tids, confs, corrects = [], [], [], [], []
        for i, record in enumerate(inference_bank):
            record_targets = record["targets"]
            out = record["out"]
            if record_targets is None:
                raise ValueError("belief_ce_bank inference records require targets")
            targets = record_targets.to(device)
            logits = _decode_inference_record(model, record, device)
            flat_logits = logits.reshape(-1, logits.shape[-1]).float()
            tgt = targets.reshape(-1)
            per = F.cross_entropy(
                flat_logits,
                tgt,
                ignore_index=-100,
                reduction="none",
            )
            conf_flat, pred_flat = flat_logits.softmax(dim=-1).max(dim=-1)
            trs = metrics.sigma_trace(
                out.sigma,
                diagonal=model.cfg.diagonal_covariance,
                family=model.cfg.family,
            ).reshape(-1)
            valid = tgt != -100
            valid_cpu = record_targets.reshape(-1) != -100
            tr_sig.append(_cpu_bank_value(trs[valid_cpu]))
            ces.append(_cpu_bank_value(per[valid]))
            tids.append(_cpu_bank_value(tgt[valid]))
            confs.append(_cpu_bank_value(conf_flat[valid]))
            corrects.append(_cpu_bank_value((pred_flat[valid] == tgt[valid]).float()))
            stop = max_batches is not None and i + 1 >= max_batches
            del targets, logits, flat_logits, tgt, per, conf_flat, pred_flat
            del trs, valid, valid_cpu
            if stop:
                break
        return {
            "tr_sigma": torch.cat(tr_sig),
            "ce": torch.cat(ces),
            "token_ids": torch.cat(tids),
            "conf": torch.cat(confs),
            "correct": torch.cat(corrects),
        }

    from vfe3.model.stack import vfe_stack
    cfg = model.cfg
    was_training = model.training
    model.eval()
    tr_sig, ces, tids, confs, corrects = [], [], [], [], []
    try:
        for i, (tokens, targets) in enumerate(loader):
            tokens = tokens.to(device)
            targets = targets.to(device)
            n = tokens.shape[1]
            logits = model(tokens)                                # (B, N, V) inference path
            flat_logits = logits.reshape(-1, logits.shape[-1]).float()           # (B*N, V)
            per = F.cross_entropy(flat_logits, targets.reshape(-1),
                                  ignore_index=-100, reduction="none")            # (B*N,)
            conf_flat, pred_flat = flat_logits.softmax(dim=-1).max(dim=-1)        # (B*N,) confidence, argmax
            beliefs = model.prior_bank.encode(tokens)             # mirror forward so the traced sigma
            beliefs = beliefs._replace(
                phi=model._apply_pos_phi(beliefs.phi),
                right_phi=model._pos_phi_right(beliefs.phi),
            )                                                       # IS the decode's belief
            rope = model._rope_rotation(n, device)
            log_prior = model._attention_log_prior(n, device)
            block_mlp_transport, block_mlp_frame = model._block_mlp_frame_transport(
                beliefs, rope)
            if is_conditional_filtering(model):
                conditional_capture: dict = {}
                model.forward_beliefs(tokens, capture=conditional_capture)
                out = conditional_capture["out"]
                del conditional_capture
            else:
                out = vfe_stack(
                    beliefs, beliefs.mu, beliefs.sigma, model.group, cfg,
                    log_prior=log_prior, block_norm=model.block_norm,
                    block_mlps=model.block_mlps,
                    block_mlp_frame=block_mlp_frame,
                    head_mixer=model.head_mixer, cg_coupling=model.cg_coupling,
                    lambda_beta=cfg.lambda_beta,
                    transport_state=model.transport_state,
                    rope=rope, rope_on_cov=cfg.rope_full_gauge, rope_on_value=cfg.rope_on_value,
                    prebuilt_transport=block_mlp_transport,
                    rope_insertion=cfg.rope_insertion,
                )
            trs = metrics.sigma_trace(
                out.sigma,
                diagonal=cfg.diagonal_covariance,
                family=cfg.family,
            ).reshape(-1)                                                    # (B*N,)
            tgt = targets.reshape(-1)
            valid = tgt != -100
            tr_sig.append(_cpu_bank_value(trs[valid]))
            ces.append(_cpu_bank_value(per[valid]))
            tids.append(_cpu_bank_value(tgt[valid]))
            confs.append(_cpu_bank_value(conf_flat[valid]))
            corrects.append(_cpu_bank_value((pred_flat[valid] == tgt[valid]).float()))
            stop = max_batches is not None and i + 1 >= max_batches
            del tokens, targets, logits, flat_logits, per, conf_flat, pred_flat
            del beliefs, rope, log_prior
            del out, trs, tgt, valid
            if stop:
                break
    finally:
        if was_training:
            model.train()
    return {
        "tr_sigma":  torch.cat(tr_sig),
        "ce":        torch.cat(ces),
        "token_ids": torch.cat(tids),
        "conf":      torch.cat(confs),               # per-token max softmax prob (sigma-gate ECE)
        "correct":   torch.cat(corrects),            # per-token 1.0 if argmax == gold (sigma-gate ECE)
    }


@torch.no_grad()
def belief_bank(
    model,
    token_batches:  Iterable[torch.Tensor],   # iterable of (B, N) token-id batches

    *,
    device:         Optional[torch.device]              = None,
    max_tokens:     Optional[int]                       = None,
    max_sequences:  Optional[int]                       = None,
    inference_bank: 'Optional[List[Dict[str, object]]]' = None,
) -> Dict[str, torch.Tensor]:
    r"""Collect converged beliefs (mu, Sigma, phi) over many sequences into one bank.

    For each batch runs the model's belief pipeline (prior_bank.encode -> pos_phi -> vfe_stack
    with the SAME per-block head_mixer / cg_coupling / block_norm the training forward applies,
    mirroring forward up to the stack's handoff belief; only final-norm decode prep is omitted) and
    stacks the per-token converged ``mu`` (M, K), ``sigma`` (M, K) or (M, K, K), ``phi``
    (M, n_gen), with ``token_ids`` (M,), ``seq_idx`` (M,), and within-sequence ``pos_idx``
    (M,). ``max_tokens`` and ``max_sequences`` are mutually exclusive; either cap is applied
    exactly to every aligned field. Shared inference records retain only positions whose aligned
    target is not ``-100``; ``token_ids`` remains the input token encoded by that belief row. Feeds
    the mu / Sigma / phi UMAP triptych and the at-scale clustering scores.
    """
    _validate_bank_caps(max_tokens=max_tokens, max_sequences=max_sequences)
    if inference_bank is not None:
        mus, sigmas, phis, tids, sidx, pidx = [], [], [], [], [], []
        seq_counter = 0
        for record in inference_bank:
            tokens = record["tokens"]
            out = record["out"]
            b, n = tokens.shape
            valid = _inference_record_valid_positions(
                tokens,
                record["targets"],
            ).reshape(-1)
            mus.append(out.mu.reshape(b * n, -1)[valid])
            sigmas.append(out.sigma.reshape(b * n, *out.sigma.shape[2:])[valid])
            phis.append(out.phi.reshape(b * n, -1)[valid])
            tids.append(tokens.reshape(b * n)[valid])
            sidx.append(
                torch.arange(
                    seq_counter,
                    seq_counter + b,
                    device=tokens.device,
                ).repeat_interleave(n)[valid]
            )
            pidx.append(torch.arange(n, device=tokens.device).repeat(b)[valid])
            seq_counter += b
            token_count = sum(batch.shape[0] for batch in tids)
            if ((max_tokens is not None and token_count >= max_tokens)
                    or (max_sequences is not None and seq_counter >= max_sequences)):
                break
        bank = {
            "mu": torch.cat(mus),
            "sigma": torch.cat(sigmas),
            "phi": torch.cat(phis),
            "token_ids": torch.cat(tids),
            "seq_idx": torch.cat(sidx),
            "pos_idx": torch.cat(pidx),
        }
        return _slice_bank_to_cap(
            bank,
            max_tokens=max_tokens,
            max_sequences=max_sequences,
        )

    from vfe3.model.stack import vfe_stack
    device = device or _model_device(model)
    cfg = model.cfg
    was_training = model.training
    model.eval()
    mus, sigmas, phis, tids, sidx, pidx = [], [], [], [], [], []
    seq_counter = 0
    try:
        for tokens in token_batches:
            tokens = tokens.to(device)
            beliefs = model.prior_bank.encode(tokens)
            beliefs = beliefs._replace(
                phi=model._apply_pos_phi(beliefs.phi),
                right_phi=model._pos_phi_right(beliefs.phi),
            )
            n = tokens.shape[1]
            rope = model._rope_rotation(n, device)
            log_prior = model._attention_log_prior(n, device)
            block_mlp_transport, block_mlp_frame = model._block_mlp_frame_transport(
                beliefs, rope)
            if is_conditional_filtering(model):
                conditional_capture: dict = {}
                model.forward_beliefs(tokens, capture=conditional_capture)
                out = conditional_capture["out"]
                del conditional_capture
            else:
                out = vfe_stack(
                    beliefs, beliefs.mu, beliefs.sigma, model.group, cfg,
                    log_prior=log_prior, block_norm=model.block_norm,
                    block_mlps=model.block_mlps,
                    block_mlp_frame=block_mlp_frame,
                    head_mixer=model.head_mixer, cg_coupling=model.cg_coupling,   # replay the trained
                    lambda_beta=cfg.lambda_beta,  # model
                    transport_state=model.transport_state,
                    rope=rope, rope_on_cov=cfg.rope_full_gauge, rope_on_value=cfg.rope_on_value,
                    prebuilt_transport=block_mlp_transport,
                    rope_insertion=cfg.rope_insertion,
                )
            b = tokens.shape[0]
            mus.append(_cpu_bank_value(out.mu.reshape(b * n, -1)))
            sigmas.append(_cpu_bank_value(out.sigma.reshape(b * n, *out.sigma.shape[2:])))
            phis.append(_cpu_bank_value(out.phi.reshape(b * n, -1)))
            tids.append(_cpu_bank_value(tokens.reshape(b * n)))
            sidx.append(_cpu_bank_value(
                torch.arange(seq_counter, seq_counter + b, device=device).repeat_interleave(n)
            ))
            pidx.append(_cpu_bank_value(torch.arange(n, device=device).repeat(b)))
            seq_counter += b
            token_count = sum(batch.shape[0] for batch in tids)
            stop = ((max_tokens is not None and token_count >= max_tokens)
                    or (max_sequences is not None and seq_counter >= max_sequences))
            del tokens, beliefs, rope, log_prior, out
            if stop:
                break
    finally:
        if was_training:
            model.train()
    bank = {
        "mu":        torch.cat(mus),
        "sigma":     torch.cat(sigmas),
        "phi":       torch.cat(phis),
        "token_ids": torch.cat(tids),
        "seq_idx":   torch.cat(sidx),
        "pos_idx":   torch.cat(pidx),
    }
    return _slice_bank_to_cap(
        bank,
        max_tokens=max_tokens,
        max_sequences=max_sequences,
    )


@torch.no_grad()
def e_step_belief_trace(
    model,
    token_ids: torch.Tensor,           # (B, N) token ids; only sequence 0 is used

    *,
    n_iter:    Optional[int] = None,
    snapshot: 'Optional[DiagnosticSnapshot]' = None,
) -> Dict[str, object]:
    r"""Loop the inner E-step capturing the FULL belief (and F) at every iteration for one sequence.

    The E-step ``return_trajectory`` yields the free-energy floats only; this captures the belief
    tuple (mu, sigma, phi) per inner iteration so the belief PATH and the SPD-metric residuals can
    be drawn. ``n_iter`` defaults to the trained ``cfg.n_e_steps`` (crank it up to show convergence
    past the trained budget). Returns ``mu`` (T+1, N, K), ``sigma`` (T+1, N, [K]), ``phi``
    (T+1, N, n_gen), and ``free_energy`` (T+1,) the global F at each iterate.
    Mean-only Q/K/V returns its captured single-step states and an empty ``free_energy`` tensor,
    with explicit availability metadata and the before/after costs at fixed conditional context.
    """
    conditional_enabled = is_conditional_filtering(model)
    if conditional_enabled:
        if n_iter is not None and n_iter != 1:
            raise NotImplementedError("Q/K/V diagnostics support the configured single E-step only")
        if snapshot is None:
            snapshot = model.build_diagnostic_snapshot(token_ids)
        if snapshot.conditional_capture is None:
            raise RuntimeError("Q/K/V belief trace requires the executed inference capture")
    if snapshot is not None:
        snapshot = model._validate_diagnostic_snapshot(token_ids, snapshot)
        actual_n_iter = len(snapshot.trace_states) - 1
        if n_iter is not None and int(n_iter) != actual_n_iter:
            raise ValueError(
                f"snapshot contains {actual_n_iter} E-step iterations, requested n_iter={n_iter}")
        return {
            "mu": torch.stack([belief.mu[0] for belief in snapshot.trace_states]),
            "sigma": torch.stack([belief.sigma[0] for belief in snapshot.trace_states]),
            "phi": torch.stack([belief.phi[0] for belief in snapshot.trace_states]),
            "free_energy": snapshot.trace_free_energy,
            "diagonal_covariance": model.cfg.diagonal_covariance,
            **({
                "inference_semantics": model.conditional_semantics,
                "free_energy_available": False,
                "free_energy_unavailable_reason": "routing and content use distinct conditional stages",
                "conditional_cost_available": model.cfg.relational_readout_mode != "predictive_content",
                **{f"relational_{key}": value
                   for key in ("conditional_cost_before", "conditional_cost_after")
                   if (value := snapshot.conditional_capture.get(
                       f"relational_{key}")) is not None},
            } if conditional_enabled else {}),
        }
    n_iter = int(n_iter if n_iter is not None else model.cfg.n_e_steps)
    belief, log_prior, rope = _encode_one(model, token_ids)
    mu_p, sigma_p = belief.mu, belief.sigma
    ikw = _iter_kwargs(model, log_prior, rope)
    fkw = _fe_kwargs(model, log_prior, rope)

    mus = [belief.mu]
    sigmas = [belief.sigma]
    phis = [belief.phi]
    fs = [free_energy_value(belief, mu_p, sigma_p, model.group, **fkw)]
    for _ in range(n_iter):
        belief = e_step_iteration(belief, mu_p, sigma_p, model.group, **ikw)
        mus.append(belief.mu)
        sigmas.append(belief.sigma)
        phis.append(belief.phi)
        fs.append(free_energy_value(belief, mu_p, sigma_p, model.group, **fkw))
    return {
        "mu":           torch.stack(mus),
        "sigma":        torch.stack(sigmas),
        "phi":          torch.stack(phis),
        "free_energy":  torch.stack([f.reshape(()) for f in fs]),
    }


@torch.no_grad()
def e_step_fixed_point_diagnostics(
    model,
    token_ids: torch.Tensor,           # (B, N) token ids; only sequence 0 is used

    *,
    snapshot: 'Optional[DiagnosticSnapshot]' = None,
) -> Dict[str, float]:
    r"""Measure configured movement and the distinct one-step-ahead residual.

    The executable map is replayed for the configured ``T`` iterations and once more when no
    snapshot is supplied. A diagnostic snapshot reuses its captured ``q_0,...,q_T`` states, where
    ``T`` is the configured terminal iteration, and executes only the
    new ``q_T -> q_{T+1}`` step. The
    ``estep_r_*_last`` fields retain their historical step-length definitions on
    ``q_{T-1} -> q_T``. The ``estep_fp_*`` fields measure ``q_T -> q_{T+1}``, which is the
    actual residual of the live state-dependent update map. For the frozen-surrogate updater,
    ``estep_target_gap`` measures the distance from the damped next iterate to the corresponding
    undamped frozen-surrogate minimizer.
    """
    _require_legacy_inference_diagnostic(model, "e_step_fixed_point_diagnostics")
    cfg = model.cfg
    if snapshot is None:
        belief0, log_prior, rope = _encode_one(model, token_ids)
        states = [belief0]
        belief = belief0
        mu_p, sigma_p = belief0.mu, belief0.sigma
        ikw = _iter_kwargs(model, log_prior, rope)
        for _ in range(int(cfg.n_e_steps)):
            belief = e_step_iteration(belief, mu_p, sigma_p, model.group, **ikw)
            states.append(belief)
    else:
        snapshot = model._validate_diagnostic_snapshot(token_ids, snapshot)
        states = [_snapshot_sequence(state) for state in snapshot.trace_states]
        if not states:
            raise ValueError("diagnostic snapshot contains no E-step states")
        belief0 = states[0]
        log_prior = model._first_sequence_log_prior(
            snapshot.log_prior, token_ids.shape[0])
        rope = snapshot.rope
    mu_p, sigma_p = belief0.mu, belief0.sigma
    ikw = _iter_kwargs(model, log_prior, rope)
    q_t = states[-1]
    q_next = e_step_iteration(q_t, mu_p, sigma_p, model.group, **ikw)
    out: Dict[str, float] = {}
    if len(states) > 1:
        configured = metrics.estep_residuals(
            torch.stack([state.mu for state in states]),
            torch.stack([state.sigma for state in states]),
            torch.stack([state.phi for state in states]),
            diagonal=cfg.diagonal_covariance,
        )
        for name, key in (
            ("r_mu", "estep_r_mu_last"),
            ("r_sigma", "estep_r_sigma_last"),
            ("r_phi", "estep_r_phi_last"),
        ):
            out[key] = float(configured[name][-1].mean())
    else:
        out.update({
            "estep_r_mu_last":    0.0,
            "estep_r_sigma_last": 0.0,
            "estep_r_phi_last":   0.0,
        })

    out["estep_fp_mu_rms"] = float((q_next.mu - q_t.mu).square().mean().sqrt())
    out["estep_fp_sigma_rms"] = float((q_next.sigma - q_t.sigma).square().mean().sqrt())
    out["estep_fp_phi_rms"] = float((q_next.phi - q_t.phi).square().mean().sqrt())

    family = get_family(cfg.family)
    out["estep_fp_kl"] = float(kl(
        model._family_instance(family, q_next.mu, q_next.sigma),
        model._family_instance(family, q_t.mu, q_t.sigma),
        kl_max=cfg.kl_max,
        eps=cfg.eps,
    ).mean())

    beta_t = model._attention_map_for_belief(q_t, log_prior, rope).float().clamp(min=cfg.eps)
    beta_next = model._attention_map_for_belief(q_next, log_prior, rope).float().clamp(min=cfg.eps)
    beta_mid = 0.5 * (beta_t + beta_next)
    beta_js = 0.5 * (
        (beta_t * (beta_t.log() - beta_mid.log())).sum(dim=-1)
        + (beta_next * (beta_next.log() - beta_mid.log())).sum(dim=-1)
    )
    out["estep_beta_js"] = float(beta_js.mean())

    prior = family(mu_p, sigma_p)

    def _alpha(state: BeliefState) -> torch.Tensor:
        divergence = self_divergence_for_alpha(
            family(state.mu, state.sigma),
            prior,
            alpha=cfg.renyi_order,
            kl_max=cfg.kl_max,
            eps=cfg.eps,
            divergence_family=cfg.divergence_family,
            lambda_alpha_mode=cfg.lambda_alpha_mode,
        )
        return self_coupling_alpha(
            divergence,
            mode=cfg.lambda_alpha_mode,
            value=cfg.lambda_alpha,
            b0=_as_coeff(cfg.b0, state.mu.device),
            c0=_as_coeff(cfg.c0, state.mu.device),
        )[0]

    out["estep_alpha_rms_delta"] = float((_alpha(q_next) - _alpha(q_t)).square().mean().sqrt())

    out["estep_target_gap"] = float("nan")
    if canonical_e_step_update(cfg.e_step_update) == "mm_exact":
        target_kwargs = dict(ikw)
        target_kwargs["mm_damping"] = 1.0
        q_target = e_step_iteration(q_t, mu_p, sigma_p, model.group, **target_kwargs)
        target_mse = torch.stack([
            (q_next.mu - q_target.mu).square().mean(),
            (q_next.sigma - q_target.sigma).square().mean(),
            (q_next.phi - q_target.phi).square().mean(),
        ]).mean()
        out["estep_target_gap"] = float(target_mse.sqrt())
    return out


@torch.no_grad()
@_block_mlp_diagnostic_eval
def across_layer_belief_trace(
    model,
    token_ids: torch.Tensor,           # (B, N) token ids; only sequence 0 is used

    *,
    snapshot: 'Optional[DiagnosticSnapshot]' = None,
) -> Dict[str, torch.Tensor]:
    r"""Per-LAYER converged beliefs by replaying the block stack (mirrors vfe_stack's handoff).

    Returns ``mu`` (L, N, K), ``sigma`` (L, N, [K]), the cumulative affine-invariant SPD geodesic
    distance ``d_ai`` (L,) of each layer's covariance from layer 0, and per-layer mean effective
    rank ``effective_rank`` (L,). Shows how the belief geometry transforms with inference depth.
    """
    if is_conditional_filtering(model) and snapshot is None:
        snapshot = model.build_diagnostic_snapshot(token_ids)
    cfg = model.cfg
    if snapshot is None:
        belief, log_prior, rope = _encode_one(model, token_ids)
        mu_p, sigma_p = belief.mu, belief.sigma
        rho, rho_s = cfg.prior_handoff_rho, cfg.prior_handoff_sigma
        mus, sigmas = [], []
        block_mlp_transport, block_mlp_frame = model._block_mlp_frame_transport(
            belief, rope)
        for layer_index in range(cfg.n_layers):
            belief = vfe_block(
                belief, mu_p, sigma_p, model.group, cfg, log_prior=log_prior,
                block_norm=model.block_norm, head_mixer=model.head_mixer,
                block_mlp=(model.block_mlps[layer_index] if model.block_mlps is not None else None),
                block_mlp_frame=block_mlp_frame,
                cg_coupling=model.cg_coupling,                       # replay the trained model
                lambda_beta=cfg.lambda_beta, transport_state=model.transport_state,
                rope=rope, rope_on_cov=cfg.rope_full_gauge, rope_on_value=cfg.rope_on_value,
                prebuilt_transport=block_mlp_transport,
                rope_insertion=cfg.rope_insertion,
            )
            mus.append(belief.mu)
            sigmas.append(belief.sigma)
            mu_p = (1.0 - rho) * mu_p + rho * belief.mu
            sigma_p = (1.0 - rho_s) * sigma_p + rho_s * belief.sigma
    else:
        snapshot = model._validate_diagnostic_snapshot(token_ids, snapshot)
        mus = [belief.mu[0] for belief in snapshot.layer_outputs]
        sigmas = [belief.sigma[0] for belief in snapshot.layer_outputs]
    mu_stack = torch.stack(mus)                                   # (L, N, K)
    sig_stack = torch.stack(sigmas)                              # (L, N, [K])
    base = sig_stack[0].unsqueeze(0).expand_as(sig_stack)
    d_ai = metrics.spd_geodesic_distance(                       # (L,) mean over tokens
        base,
        sig_stack,
        diagonal=cfg.diagonal_covariance,
    ).mean(dim=-1)
    eff = metrics.effective_rank_per_token(
        sig_stack,
        diagonal=cfg.diagonal_covariance,
        eps=cfg.eps,
        family=cfg.family,
    ).mean(dim=-1)                                                        # (L,)
    r_one = metrics.rank_one_residual(mu_stack)                          # (L,) Dong r(X) per layer (F2/EXP-7)
    return {"mu": mu_stack, "sigma": sig_stack, "d_ai": d_ai,
            "effective_rank": eff, "rank_one_residual": r_one}


@torch.no_grad()
def numerical_health(
    model,
    token_ids: torch.Tensor,           # (B, N) token ids; only sequence 0 is used

    *,
    snapshot: 'Optional[DiagnosticSnapshot]' = None,
) -> Dict[str, float]:
    r"""Non-finite fractions of the converged intermediates + the worst covariance conditioning.

    Replays the converged belief and its energy / attention (as model.diagnostics does) and
    reports ``nan_inf_fraction`` for mu, sigma, phi, energy, and beta, plus the maximum spectral
    condition number of the belief covariances. A near-zero finiteness map certifies stability is
    genuine convergence, not masked blow-ups. (Numerical-FALLBACK activation counters --
    safe_cholesky jitter rounds, pinv fallbacks -- would require instrumenting numerics and are
    left to a future pass.)
    """
    if is_conditional_filtering(model):
        from vfe3.numerics import nan_inf_fraction

        if snapshot is None:
            snapshot = model.build_diagnostic_snapshot(token_ids)
        snapshot = model._validate_diagnostic_snapshot(token_ids, snapshot)
        out = _snapshot_sequence(snapshot.layer_converged[0])
        capture = snapshot.conditional_capture
        if capture is None:
            raise RuntimeError("Q/K/V numerical health requires the executed inference capture")
        condition = metrics.belief_spectrum(
            out.sigma, diagonal=True, eps=model.cfg.eps, family=model.cfg.family)["condition"]
        return {
            "nan_mu": nan_inf_fraction(out.mu),
            "nan_sigma": nan_inf_fraction(out.sigma),
            "nan_phi": nan_inf_fraction(out.phi),
            "nan_energy": nan_inf_fraction(capture["score_energy_pre"]),
            "nan_beta": nan_inf_fraction(capture["beta_pre"]),
            "max_condition": float(condition.max()),
        }
    from vfe3.numerics import nan_inf_fraction
    from vfe3.geometry.transport import RopeTransport   # m5: used below under pos_rotation='rope' (was NameError)
    cfg = model.cfg
    if snapshot is None:
        belief, log_prior, rope = _encode_one(model, token_ids)
        ikw = _iter_kwargs(model, log_prior, rope)
        out = belief
        for _ in range(cfg.n_e_steps):
            out = e_step_iteration(out, belief.mu, belief.sigma, model.group, **ikw)
    else:
        snapshot = model._validate_diagnostic_snapshot(token_ids, snapshot)
        out = _snapshot_sequence(snapshot.layer_converged[0])
        log_prior = model._first_sequence_log_prior(snapshot.log_prior, token_ids.shape[0])
        rope = snapshot.rope
    # Build Omega under the ACTIVE connection regime (audit 2026-06-10 F8e): this previously
    # defaulted to flat transport, so a reported nan/beta/energy fraction could describe a
    # different belief than the model that trained. Mirrors converged_state.
    omega = model._diagnostic_transport(out)
    # Wrap in RopeTransport under pos_rotation='rope' so the reported nan/energy/beta fractions
    # describe the RoPE-rotated belief the model runs, mirroring converged_state/diagnostics (r2 id11).
    fam = get_family(cfg.family)
    if rope is not None:
        effective = RopeTransport(base=omega, rope=rope, on_cov=cfg.rope_full_gauge,
                                  on_value=cfg.rope_on_value, insertion=cfg.rope_insertion,
                                  same_frame_flat_cocycle=getattr(
                                      omega, "same_frame_flat_cocycle", False))
        mu, sigma = out.mu, out.sigma
        batched = False
    else:
        effective = omega.unsqueeze(0)
        mu, sigma = out.mu.unsqueeze(0), out.sigma.unsqueeze(0)
        batched = True
    energy = model._coupling_energy(fam, mu, sigma, mu, sigma, effective,
                                 alpha=cfg.renyi_order,
                                 kl_max=cfg.kl_max, eps=cfg.eps,
                                 divergence_family=cfg.divergence_family,
                                 irrep_dims=model.group.irrep_dims)
    if batched:
        energy = energy[0]
    beta = attention_weights(
        energy,
        tau=attention_tau(
            _as_coeff(cfg.kappa_beta, out.mu.device), model._belief_score_irrep_dims()),
        log_prior=log_prior)
    condition = metrics.belief_spectrum(
        out.sigma,
        diagonal=(out.sigma.dim() == out.mu.dim()),
        eps=cfg.eps,
        family=cfg.family,
    )["condition"]
    cond = float(condition.max())
    health = {
        "nan_mu":     nan_inf_fraction(out.mu),
        "nan_sigma":  nan_inf_fraction(out.sigma),
        "nan_phi":    nan_inf_fraction(out.phi),
        "nan_energy": nan_inf_fraction(energy),
        "nan_beta":   nan_inf_fraction(beta),
        "max_condition": cond,
    }
    return health


@torch.no_grad()
def readout_state(
    model:     torch.nn.Module,
    token_ids: torch.Tensor,

    *,
    snapshot: 'Optional[DiagnosticSnapshot]' = None,
) -> Dict[str, object]:
    """Captured conditional output geometry and pre-update scores, without a shared objective."""
    if snapshot is None:
        snapshot = model.build_diagnostic_snapshot(token_ids)
    snapshot = model._validate_diagnostic_snapshot(token_ids, snapshot)
    out = _snapshot_sequence(snapshot.layer_converged[-1])
    cap = snapshot.conditional_capture
    if cap is None:
        raise ValueError("readout_state requires a conditional forward capture")
    guard = metrics.guard_saturation(
        out.sigma, cap["score_energy_pre"], out.mu.new_empty(0), diagonal=True,
        eps=model.cfg.eps, sigma_max=model.cfg.sigma_max, kl_max=model.cfg.kl_max,
    )
    guard.pop("selfdiv_klmax_frac", None)  # no self-energy was evaluated by this diagnostic
    state = {
        "mu": out.mu, "sigma": out.sigma, "phi": out.phi,
        "energy": cap["score_energy_pre"], "beta": cap["beta_pre"], "guard": guard,
        "inference_semantics": model.conditional_semantics,
        "covariance_semantics": "receiver_bookkeeping",
    }
    # This is the same-frame receiver geometry, not an equivariance certificate
    # for the learned comparison maps. Keep pair transport compact.
    omega = model._diagnostic_transport(out)
    if hasattr(omega, "exp_blocks"):
        blocks = omega.exp_blocks
        n, h, d, _ = blocks.shape
        exp_phi = blocks.new_zeros(n, h * d, h * d)
        for head in range(h):
            exp_phi[:, head*d:(head+1)*d, head*d:(head+1)*d] = blocks[:, head]
    else:
        exp_phi = omega.exp_phi
    state.update(omega=omega, exp_phi=exp_phi, frame_scope="receiver_same_frame")
    return state


@torch.no_grad()
def converged_state(
    model,
    token_ids: torch.Tensor,           # (B, N) token ids; only sequence 0 is used

    *,
    snapshot: 'Optional[DiagnosticSnapshot]' = None,
) -> Dict[str, object]:
    r"""The converged-belief diagnostic state of one sequence, as tensors for the figures.

    Mirrors :meth:`VFEModel.diagnostics` EXACTLY (same active config: transport mode,
    connection_W, rope, family, divergence, alpha) but returns the underlying tensors the scalar
    diagnostics discard. The gauge-equivariance certificate, per-head gauge invariants, belief
    spectrum / SPD ellipses, and the guard-saturation / causal panels all read from these. Returns
    the converged ``mu`` (N, K), ``sigma`` (N, K) or (N, K, K), ``phi`` (N, n_gen), the active
    per-token vertex factor under the compatibility key ``exp_phi`` (N, K, K), and the pre-rope
    pairwise transport ``omega`` that the equivariance/holonomy metrics consume (compact-factored
    when the compact phi-block path is active, otherwise dense), the pairwise ``energy`` and
    attention ``beta`` ((N, N) or (H, N, N)), and the per-token self-divergence ``self_div``
    (N,) or (N, K).
    """
    _require_legacy_inference_diagnostic(model, "converged_state")
    from vfe3.model.stack import vfe_stack
    from vfe3.geometry.transport import RopeTransport, build_factored_transport

    cfg = model.cfg
    was_training = model.training
    model.eval()
    try:
        if snapshot is None:
            belief, log_prior, rope = _encode_one(model, token_ids)
            cap: dict = {}                                       # q* capture (F self-term, as diagnostics)
            block_mlp_transport, block_mlp_frame = model._block_mlp_frame_transport(
                belief, rope)
            out = vfe_stack(
                belief, belief.mu, belief.sigma, model.group, cfg,
                log_prior=log_prior, block_norm=model.block_norm,
                block_mlps=model.block_mlps,
                block_mlp_frame=block_mlp_frame,
                head_mixer=model.head_mixer, cg_coupling=model.cg_coupling,
                lambda_beta=cfg.lambda_beta,
                transport_state=model.transport_state,
                rope=rope, rope_on_cov=cfg.rope_full_gauge, rope_on_value=cfg.rope_on_value,
                capture=cap,
                prebuilt_transport=block_mlp_transport,
                rope_insertion=cfg.rope_insertion,
            )
        else:
            snapshot = model._validate_diagnostic_snapshot(token_ids, snapshot)
            belief = _snapshot_sequence(snapshot.initial_belief)
            out = _snapshot_sequence(snapshot.stack_output)
            log_prior = model._first_sequence_log_prior(snapshot.log_prior, token_ids.shape[0])
            rope = snapshot.rope
            cap = {"converged": _snapshot_sequence(snapshot.layer_converged[-1])}
        rho, rho_s = cfg.prior_handoff_rho, cfg.prior_handoff_sigma     # rebuild last-block prior
        mu_p, sigma_p = belief.mu, belief.sigma                         # exact iff L==1 or rho==0
        for _ in range(cfg.n_layers - 1):
            mu_p = (1.0 - rho) * mu_p + rho * out.mu
            sigma_p = (1.0 - rho_s) * sigma_p + rho_s * out.sigma
        omega = model._diagnostic_transport(out)                       # active pre-rope transport
        fam = get_family(cfg.family)
        if rope is not None:
            effective = RopeTransport(base=omega, rope=rope, on_cov=cfg.rope_full_gauge,
                                      on_value=cfg.rope_on_value, insertion=cfg.rope_insertion,
                                      same_frame_flat_cocycle=getattr(
                                          omega, "same_frame_flat_cocycle", False))
            mu, sigma = out.mu, out.sigma
            batched = False
        else:
            effective = omega.unsqueeze(0)
            mu, sigma = out.mu.unsqueeze(0), out.sigma.unsqueeze(0)
            batched = True
        energy = model._coupling_energy(fam,
            mu, sigma, mu, sigma, effective,
            alpha=cfg.renyi_order,
            kl_max=cfg.kl_max, eps=cfg.eps, divergence_family=cfg.divergence_family,
            irrep_dims=model.group.irrep_dims,
        )
        if batched:
            energy = energy[0]
        if snapshot is None:
            beta = attention_weights(
                energy,
                tau=attention_tau(
                    _as_coeff(cfg.kappa_beta, out.mu.device), model._belief_score_irrep_dims()),
                log_prior=log_prior)
        else:
            beta = snapshot.beta_maps[-1]
            if energy.dim() == 2:
                beta = beta[0]
        _q_conv = cap["converged"]                           # q*: the F self-term reads the pre-
        self_div = self_divergence_for_alpha(                # transform converged belief (F19,
            model._family_instance(fam, _q_conv.mu, _q_conv.sigma),
            model._family_instance(fam, mu_p, sigma_p), alpha=cfg.renyi_order,   # as diagnostics)
            kl_max=cfg.kl_max, eps=cfg.eps, divergence_family=cfg.divergence_family,
            lambda_alpha_mode=cfg.lambda_alpha_mode,
        )
        exp_phi = build_factored_transport(
            out.phi, model.group,
            exp_fp64_norm_threshold=cfg.exp_fp64_norm_threshold,
            clamp_monitor=cfg.transport_clamp_monitor,
            validity_max_norm=cfg.transport_chart_max_norm,
            right_phi=out.right_phi,
        ).exp_phi
    finally:
        if was_training:
            model.train()
    return {
        "mu":       out.mu,
        "sigma":    out.sigma,
        "phi":      out.phi,
        "exp_phi":  exp_phi,
        "omega":    omega,
        "energy":   energy,
        "beta":     beta,
        "self_div": self_div,
    }


def attention_entropy_cov_gap(
    model,
    token_ids: torch.Tensor,           # (B, N) token ids; only sequence 0 is used
) -> Dict[str, torch.Tensor]:
    r"""C1/EXP-4: the attention-entropy gradient gap -tau^{-1} Cov_beta(E, dE) on the converged belief.

    The production closed-form kernel CANNOT compute the attention-entropy term, so it descends the
    SURROGATE belief gradient; only the autograd ORACLE with ``include_attention_entropy=True``
    realizes the canonical gradient. Differencing the oracle gradient with the entropy term ON vs OFF
    on the SAME converged-belief snapshot isolates exactly the entropy term's contribution, which by
    the envelope identity equals ``lambda_beta * (-tau^{-1} Cov_beta(E, dE))`` (= the bare identity on
    the pure ``lambda_beta=1`` path; the scalar identity is proven in
    ``tests/test_free_energy.py::test_gradient_gap_canonical_minus_surrogate_is_neg_cov_over_tau``).
    Returns the per-token gap magnitude ``cov_gap_per_token`` (N,) = ||g_canon - g_surrogate|| over
    the (mu, sigma) blocks, and the scalar mean ``cov_gap``. Built for the
    single-block (L=1) operating point and the flat-transport path (a mu-dependent
    Omega would need an omega_builder; that is not the EXP-4 operating point).
    """
    _require_legacy_inference_diagnostic(model, "attention_entropy_cov_gap")
    from vfe3.gradients.oracle import belief_gradients_autograd
    from vfe3.geometry.transport import RopeTransport
    cfg = model.cfg
    dev = _model_device(model)
    with torch.no_grad():                                          # snapshot build is grad-free
        belief, log_prior, rope = _encode_one(model, token_ids)   # _encode_one folds the precision bias
        ikw = _iter_kwargs(model, log_prior, rope)
        mu_p, sigma_p = belief.mu, belief.sigma
        out = belief
        for _ in range(cfg.n_e_steps):                            # converge to q* (the operating point)
            out = e_step_iteration(out, mu_p, sigma_p, model.group, **ikw)
        omega = model._diagnostic_transport(out)
        if rope is not None:
            omega = RopeTransport(base=omega, rope=rope, on_cov=cfg.rope_full_gauge,
                                  on_value=cfg.rope_on_value, insertion=cfg.rope_insertion,
                                  same_frame_flat_cocycle=getattr(
                                      omega, "same_frame_flat_cocycle", False))
    kw = dict(                                                    # the oracle's free-energy knob bag
        tau=attention_tau(
            _as_coeff(cfg.kappa_beta, dev), model._belief_score_irrep_dims()),
        renyi_order=cfg.renyi_order, kl_max=cfg.kl_max, eps=cfg.eps,
        b0=_as_coeff(cfg.b0, dev), c0=_as_coeff(cfg.c0, dev), value=cfg.lambda_alpha,
        lambda_beta=cfg.lambda_beta, gradient_mode=cfg.gradient_mode, family=cfg.family,
        full_cov_kl_backend=model.full_cov_kl_backend,
        divergence_family=cfg.divergence_family, lambda_alpha_mode=cfg.lambda_alpha_mode,
        irrep_dims=model.group.irrep_dims, log_prior=log_prior,
    )                                                             # grad ON here (outside the no_grad block)
    gc_mu, gc_sig = belief_gradients_autograd(out.mu, out.sigma, mu_p, sigma_p, omega,
                                              include_attention_entropy=True, **kw)
    gs_mu, gs_sig = belief_gradients_autograd(out.mu, out.sigma, mu_p, sigma_p, omega,
                                              include_attention_entropy=False, **kw)
    n = out.mu.shape[0]
    per_tok = torch.sqrt((gc_mu - gs_mu).pow(2).reshape(n, -1).sum(dim=-1)
                         + (gc_sig - gs_sig).pow(2).reshape(n, -1).sum(dim=-1))   # (N,)
    return {"cov_gap_per_token": per_tok, "cov_gap": per_tok.mean()}


def _vocab_display_panel(
    probs:    torch.Tensor,               # (P0, V) per-position softmax for ONE sequence
    tokens:   torch.Tensor,               # (N,)    that sequence's input token ids
    targets:  Optional[torch.Tensor] = None,  # (P0,) aligned gold ids; None -> tokens[1:]

    *,
    max_rows:      int = 40,
    per_pos_k:     int = 3,
    max_positions: int = 64,
) -> Dict[str, object]:
    r"""Select a compact ``(R, P)`` probability sub-grid for the Seq x top-k heatmap.

    The full ``p(o_{n+1} | o_{<=n})`` over ``V = 50257`` tokens is illegible, so the y-axis is the
    union of each shown position's top-``per_pos_k`` predicted token ids together with its aligned
    target, ranked by summed probability over the shown positions and capped at ``max_rows``; the
    x-axis is the first ``max_positions`` positions. When ``targets`` is absent, the token-only
    caller supplies the historical in-window shift ``tokens[1:]``. ``disp_truth_row`` gives, per
    position, the row index of the true next token (-1 when it falls outside the shown rows), for
    the ground-truth overlay.
    """
    P = min(int(probs.shape[0]), max_positions)
    p = probs[:P]                                                # (P, V)
    context_ids = tokens[:P]                                     # (P,) token AT each shown position
    target_ids = tokens[1:P + 1] if targets is None else targets[:P]
    topk = p.topk(min(per_pos_k, p.shape[-1]), dim=-1).indices.reshape(-1)   # (P*k,)
    cand = torch.cat([topk, target_ids]).unique()               # (C,) candidate row ids
    mass = p[:, cand].sum(dim=0)                                # (C,) total mass per candidate
    order = mass.argsort(descending=True)[:max_rows]
    row_ids = cand[order]                                        # (R,)
    disp_probs = p[:, row_ids].transpose(0, 1).contiguous()     # (R, P)
    pos_in = {int(t): i for i, t in enumerate(row_ids.tolist())}
    truth_row = torch.tensor([pos_in.get(int(t), -1) for t in target_ids.tolist()], dtype=torch.long)
    return {
        "disp_context_ids": context_ids.cpu(),
        "disp_target_ids":  target_ids.cpu(),
        "row_ids":          row_ids.cpu(),
        "disp_probs":       disp_probs.cpu(),
        "disp_truth_row":   truth_row,
    }


@torch.no_grad()
def vocab_prediction_stats(
    model,
    token_batches:  Iterable[torch.Tensor],   # iterable of (B, N) token-id batches

    *,
    max_rows:       int                                    = 40,
    per_pos_k:      int                                    = 3,
    max_positions:  int                                    = 64,
    max_pairs:      int                                    = 200_000,
    device:         Optional[torch.device]                 = None,
    inference_bank: 'Optional[List[Dict[str, object]]]'    = None,
) -> Dict[str, object]:
    r"""Next-token predictive distributions over the vocabulary, for the vocab-probability figures.

    Runs the inference forward (``model(tokens)`` -> ``(B, N, V)`` logits) and softmaxes to
    ``p(o_{n+1} | o_{<=n})``. The token-only path preserves its historical in-window contract:
    positions ``0..N-2`` use ``tokens[:, 1:]`` and the final position is dropped. A shared
    inference record instead supplies the evaluation target aligned with every logit row; positions
    carrying ``-100`` are excluded and a final real target outside the input window is retained.

    Returns one dict feeding three figures:
      display (first sequence) -- ``disp_context_ids`` (P,), ``disp_target_ids`` (P,),
        ``row_ids`` (R,), ``disp_probs`` (R, P), ``disp_truth_row`` (P,);
      calibration (all positions) -- ``mean_pred_prob`` (V,), ``unigram`` (V,),
        ``mean_pred_entropy`` (), ``unigram_entropy`` (), ``n_positions`` ();
      confusion (capped sample of <= ``max_pairs``) -- ``true_ids`` (M,), ``pred_ids`` (M,) argmax.
    Entropies are in nats: ``mean_pred_entropy`` is the mean conditional ``H(p(.|context))`` and
    ``unigram_entropy`` is ``H`` of the empirical marginal; their gap is the context information
    the model uses, which collapses toward zero when predictions degenerate to the marginal.
    """
    device = device or _model_device(model)
    was_training = model.training
    model.eval()
    V = int(model.cfg.vocab_size)
    prob_sum = torch.zeros(V, dtype=torch.float64, device=device)
    tgt_count = torch.zeros(V, dtype=torch.float64, device=device)
    ent_sum = torch.zeros((), dtype=torch.float64, device=device)
    n_pos = 0
    true_ids: List[torch.Tensor] = []
    pred_ids: List[torch.Tensor] = []
    n_pairs = 0
    disp: Optional[Dict[str, object]] = None
    try:
        sources = inference_bank if inference_bank is not None else token_batches
        for batch in sources:
            stored_targets = None
            valid = None
            if inference_bank is not None:
                tokens = batch["tokens"].to(device)
                decoded = _decode_inference_record(model, batch, device)
                if batch["targets"] is not None:
                    stored_targets = batch["targets"].to(device)
            else:
                tokens = batch[0] if isinstance(batch, (tuple, list)) else batch
                tokens = tokens.to(device)
                decoded = model(tokens)
            if tokens.dim() == 1:
                tokens = tokens.unsqueeze(0)
            if stored_targets is not None and stored_targets.dim() == 1:
                stored_targets = stored_targets.unsqueeze(0)
            b, n = tokens.shape
            if stored_targets is not None:
                valid = _inference_record_valid_positions(tokens, stored_targets)
                valid_flat = valid.reshape(-1)
                if not bool(valid_flat.any()):
                    del batch, tokens, decoded, stored_targets, valid, valid_flat
                    continue
                logits = decoded.reshape(-1, V)[valid_flat].float()
                targets = stored_targets.reshape(-1)[valid_flat]
                decoded = None
                probs = torch.softmax(logits, dim=-1)
                flatp = probs
            else:
                if n < 2:
                    del batch, tokens, decoded, stored_targets
                    continue
                logits = decoded[:, :-1].float()                    # (B, N-1, V) drop last (no in-window target)
                targets = tokens[:, 1:]                             # (B, N-1) true next token
                probs = torch.softmax(logits, dim=-1)               # (B, N-1, V)
                flatp = probs.reshape(-1, V)                        # (B*(N-1), V)
            decoded = None
            logits = None
            prob_sum += flatp.sum(dim=0).double()
            entropy_terms = flatp.clamp_min(1e-12)
            entropy_terms.log_()
            entropy_terms.mul_(flatp)
            ent_sum -= entropy_terms.sum(dim=-1).sum().double()
            del entropy_terms
            tgt_flat = targets.reshape(-1)
            tgt_count += torch.bincount(tgt_flat, minlength=V).double()
            n_pos += int(tgt_flat.numel())
            if n_pairs < max_pairs:
                take = min(max_pairs - n_pairs, int(tgt_flat.numel()))
                true_ids.append(tgt_flat[:take].cpu())
                pred_ids.append(flatp[:take].argmax(dim=-1).cpu())
                n_pairs += take
            if disp is None:
                if stored_targets is None:
                    disp = _vocab_display_panel(
                        probs[0],
                        tokens[0],
                        max_rows=max_rows,
                        per_pos_k=per_pos_k,
                        max_positions=max_positions,
                    )
                else:
                    row = int(valid.any(dim=1).nonzero()[0, 0])
                    offset = int(valid[:row].sum())
                    length = int(valid[row].sum())
                    disp = _vocab_display_panel(
                        flatp[offset:offset + length],
                        tokens[row][valid[row]],
                        stored_targets[row][valid[row]],
                        max_rows=max_rows,
                        per_pos_k=per_pos_k,
                        max_positions=max_positions,
                    )
            # Aggregates and display samples are already detached onto their owning devices.
            # Drop the full-vocabulary current-batch workset before the next decode/softmax.
            del batch, tokens, decoded, stored_targets, logits, targets, probs, flatp, tgt_flat
            if valid is not None:
                del valid, valid_flat
    finally:
        if was_training:
            model.train()
    if disp is None:
        raise ValueError("vocab_prediction_stats: no usable (N>=2) batch in token_batches")
    denom = max(n_pos, 1)
    mean_pred_prob = prob_sum / denom
    unigram = tgt_count / denom
    uni_ent = float(-(unigram * unigram.clamp_min(1e-12).log()).sum())
    return {
        **disp,
        "mean_pred_prob":    mean_pred_prob.float().cpu(),
        "unigram":           unigram.float().cpu(),
        "mean_pred_entropy": float(ent_sum / denom),
        "unigram_entropy":   uni_ent,
        "n_positions":       int(n_pos),
        "true_ids":          torch.cat(true_ids) if true_ids else torch.empty(0, dtype=torch.long),
        "pred_ids":          torch.cat(pred_ids) if pred_ids else torch.empty(0, dtype=torch.long),
    }


@torch.no_grad()
def belief_ce_vocab_stats(
    model,
    inference_bank: List[Dict[str, object]],

    *,
    max_rows:      int                     = 40,
    per_pos_k:     int                     = 3,
    max_positions: int                     = 64,
    max_pairs:     int                     = 200_000,
    max_batches:   Optional[int]           = None,
    device:        Optional[torch.device]  = None,
) -> 'Tuple[Dict[str, torch.Tensor], Optional[Dict[str, object]]]':
    r"""Stream one decode per inference record into both full-vocabulary report products.

    The converged belief is decoded once for each bounded record. Cross-entropy, confidence,
    vocabulary aggregates, display rows, and confusion pairs are reduced before that record's
    full-vocabulary tensors are released. The returned dictionaries retain the public schemas of
    ``belief_ce_bank`` and ``vocab_prediction_stats``; the vocabulary result is ``None`` only when
    every aligned target is masked.
    """
    device = device or _model_device(model)
    was_training = model.training
    model.eval()
    vocab_size = int(model.cfg.vocab_size)
    tr_sig: List[torch.Tensor] = []
    ces: List[torch.Tensor] = []
    tids: List[torch.Tensor] = []
    confs: List[torch.Tensor] = []
    corrects: List[torch.Tensor] = []
    prob_sum = torch.zeros(vocab_size, dtype=torch.float64, device=device)
    tgt_count = torch.zeros(vocab_size, dtype=torch.float64, device=device)
    ent_sum = torch.zeros((), dtype=torch.float64, device=device)
    true_ids: List[torch.Tensor] = []
    pred_ids: List[torch.Tensor] = []
    n_pos = 0
    n_pairs = 0
    disp: Optional[Dict[str, object]] = None
    try:
        for i, record in enumerate(inference_bank):
            record_targets = record["targets"]
            if record_targets is None:
                raise ValueError("belief_ce_vocab_stats inference records require targets")
            record_tokens = record["tokens"]
            out = record["out"]
            tokens = record_tokens.to(device)
            targets = record_targets.to(device)
            if tokens.dim() == 1:
                tokens = tokens.unsqueeze(0)
            if targets.dim() == 1:
                targets = targets.unsqueeze(0)
            decoded = _decode_inference_record(model, record, device)
            flat_logits = decoded.reshape(-1, vocab_size).float()
            tgt = targets.reshape(-1)
            valid = tgt != -100
            valid_cpu = record_targets.reshape(-1) != -100
            valid_logits = flat_logits[valid]
            tgt_valid = tgt[valid]
            decoded = None
            flat_logits = None
            valid_count = int(tgt_valid.numel())
            if valid_count:
                per = F.cross_entropy(
                    valid_logits,
                    tgt_valid,
                    reduction="none",
                )
                probabilities = valid_logits.softmax(dim=-1)
                conf_flat, pred_flat = probabilities.max(dim=-1)
            else:
                per = valid_logits.new_empty((0,))
                probabilities = valid_logits.new_empty((0, vocab_size))
                conf_flat = valid_logits.new_empty((0,))
                pred_flat = tgt_valid.new_empty((0,))
            valid_logits = None
            trs = metrics.sigma_trace(
                out.sigma,
                diagonal=model.cfg.diagonal_covariance,
                family=model.cfg.family,
            ).reshape(-1)
            tr_sig.append(_cpu_bank_value(trs[valid_cpu]))
            ces.append(_cpu_bank_value(per))
            tids.append(_cpu_bank_value(tgt_valid))
            confs.append(_cpu_bank_value(conf_flat))
            corrects.append(_cpu_bank_value((pred_flat == tgt_valid).float()))

            if valid_count:
                flatp = probabilities
                pred_valid = pred_flat
                probabilities = None
                prob_sum += flatp.sum(dim=0).double()
                entropy_terms = flatp.clamp_min(1e-12)
                entropy_terms.log_()
                entropy_terms.mul_(flatp)
                ent_sum -= entropy_terms.sum(dim=-1).sum().double()
                del entropy_terms
                tgt_count += torch.bincount(tgt_valid, minlength=vocab_size).double()
                n_pos += valid_count
                if n_pairs < max_pairs:
                    take = min(max_pairs - n_pairs, int(tgt_valid.numel()))
                    true_ids.append(tgt_valid[:take].cpu())
                    pred_ids.append(pred_valid[:take].cpu())
                    n_pairs += take
                if disp is None:
                    valid_rows = valid.reshape(tokens.shape)
                    row = int(valid_rows.any(dim=1).nonzero()[0, 0])
                    offset = int(valid_rows[:row].sum())
                    length = int(valid_rows[row].sum())
                    disp = _vocab_display_panel(
                        flatp[offset:offset + length],
                        tokens[row][valid_rows[row]],
                        targets[row][valid_rows[row]],
                        max_rows=max_rows,
                        per_pos_k=per_pos_k,
                        max_positions=max_positions,
                    )
                del flatp, pred_valid
            stop = max_batches is not None and i + 1 >= max_batches
            del record, record_targets, record_tokens, out, tokens, targets
            del decoded, flat_logits, valid_logits, tgt, tgt_valid, per, probabilities
            del conf_flat, pred_flat, trs, valid, valid_cpu
            if stop:
                break
    finally:
        if was_training:
            model.train()
    ce_stats = {
        "tr_sigma": torch.cat(tr_sig),
        "ce": torch.cat(ces),
        "token_ids": torch.cat(tids),
        "conf": torch.cat(confs),
        "correct": torch.cat(corrects),
    }
    if disp is None:
        return ce_stats, None
    denom = max(n_pos, 1)
    mean_pred_prob = prob_sum / denom
    unigram = tgt_count / denom
    vocab_stats: Dict[str, object] = {
        **disp,
        "mean_pred_prob": mean_pred_prob.float().cpu(),
        "unigram": unigram.float().cpu(),
        "mean_pred_entropy": float(ent_sum / denom),
        "unigram_entropy": float(-(unigram * unigram.clamp_min(1e-12).log()).sum()),
        "n_positions": int(n_pos),
        "true_ids": torch.cat(true_ids) if true_ids else torch.empty(0, dtype=torch.long),
        "pred_ids": torch.cat(pred_ids) if pred_ids else torch.empty(0, dtype=torch.long),
    }
    return ce_stats, vocab_stats


@torch.no_grad()
def decode_readout(
    model,

    *,
    max_rows: int = 96,
) -> Optional[Dict[str, object]]:
    r"""The linear-decode readout matrix ``W`` (``logits = mu_q @ W^T``) for the Decode-W heatmap.

    Returns ``None`` on prior-bank decode modes (no ``W`` exists there).
    Otherwise selects the ``max_rows`` highest-L2-norm vocabulary rows -- the output directions the
    readout most strongly distinguishes -- and returns ``weight`` (R, K), their ``row_ids`` (R,),
    the per-row ``row_norm`` (R,), and the optional log-unigram ``bias`` (R,).
    """
    pb = model.prior_bank
    W = getattr(pb, "output_proj_weight", None)
    if W is None:
        return None
    W = W.detach().float()
    norms = W.norm(dim=1)                                        # (V,)
    row_ids = norms.argsort(descending=True)[:max_rows]
    bias = getattr(pb, "output_proj_bias", None)
    return {
        "weight":   W[row_ids].cpu(),
        "row_ids":  row_ids.cpu(),
        "row_norm": norms[row_ids].cpu(),
        "bias":     (bias.detach().float()[row_ids].cpu() if bias is not None else None),
    }
