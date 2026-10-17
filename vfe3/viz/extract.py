r"""Belief / evaluation EXTRACTORS for VFE_3.0 publication figures.

These RUN the model -- read one diagnostic snapshot of the projected forward (captured receiver
states, per-layer outputs, routing), score a loader per unit, measure the gauge equivariance of the
executed projected score and readout, or tally numerical health. They have side effects / drive the model, so they live here rather than in
``vfe3.metrics`` (whose contract is pure, side-effect-free measurement). Each returns plain
tensors / dicts that the pure metrics and the figure functions consume. Everything runs under
``torch.no_grad`` and OFF the training hot path.
"""

import math
from typing import TYPE_CHECKING, Dict, Iterable, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F

from vfe3 import metrics
from vfe3.belief import BeliefState
from vfe3.contracts import CanonicalFrameContext
from vfe3.model.block import _as_coeff
from vfe3.model.model import _block_mlp_diagnostic_eval
from vfe3.free_energy import attention_tau
from vfe3.geometry.lie_ops import CompactBlockElement

if TYPE_CHECKING:
    from vfe3.model.model import DiagnosticSnapshot


def _model_device(model) -> torch.device:
    return next(model.parameters()).device


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
            block_diagonal=value.block_diagonal,              # compact (..., H, d, d) head blocks stay compact
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
    """Reject ambiguous or empty report-bank population requests."""
    if max_tokens is not None and max_sequences is not None:
        raise ValueError("max_tokens and max_sequences are mutually exclusive")
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if max_sequences is not None and max_sequences <= 0:
        raise ValueError("max_sequences must be positive")


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
            block_diagonal=canonical_frame.block_diagonal,
        )
    logits = model._decode_belief_with_context(
        mu,
        sigma,
        canonical_frame=canonical_frame,
    )
    del mu, sigma, canonical_frame
    return logits


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
    single-seed bootstrap band (``metrics.bootstrap_token_ce_band`` over tokens or
    sequence blocks) needs the per-unit nats this produces. Runs
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

    For each ``(tokens, targets)`` batch this captures the stack output ``out.sigma`` of
    ``model.forward_beliefs`` and runs the inference forward ``model(tokens)`` for the decode logits,
    then reduces the cross-entropy with ``reduction='none'``, so the traced Sigma_q IS the belief
    whose mean produced the logits. The belief at
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

    cfg = model.cfg
    was_training = model.training
    model.eval()
    tr_sig, ces, tids, confs, corrects = [], [], [], [], []
    try:
        for i, (tokens, targets) in enumerate(loader):
            tokens = tokens.to(device)
            targets = targets.to(device)
            logits = model(tokens)                                # (B, N, V) inference path
            flat_logits = logits.reshape(-1, logits.shape[-1]).float()           # (B*N, V)
            per = F.cross_entropy(flat_logits, targets.reshape(-1),
                                  ignore_index=-100, reduction="none")            # (B*N,)
            conf_flat, pred_flat = flat_logits.softmax(dim=-1).max(dim=-1)        # (B*N,) confidence, argmax
            # The traced covariance is the stack output of the same inference forward.
            conditional_capture: dict = {}
            model.forward_beliefs(tokens, capture=conditional_capture)
            out = conditional_capture["out"]
            del conditional_capture
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
def e_step_belief_trace(
    model,
    token_ids: torch.Tensor,           # (B, N) token ids; only sequence 0 is used

    *,
    n_iter:    Optional[int] = None,
    snapshot: 'Optional[DiagnosticSnapshot]' = None,
) -> Dict[str, object]:
    r"""The captured receiver states ``q_0, ..., q_T`` of sequence 0 and their fixed-context costs.

    Projected routes compute routing and content in distinct conditional stages, so no global free
    energy is defined: ``free_energy`` is the snapshot's (empty) trace and ``free_energy_available``
    is False. Returns ``mu`` (T+1, N, K), ``sigma`` (T+1, N, [K]), ``phi`` (T+1, N, n_gen), plus the
    receiver costs before/after the update on the full-link readout. ``n_iter`` must equal the
    configured ``cfg.n_e_steps``.
    """
    if n_iter is not None and int(n_iter) != model.cfg.n_e_steps:
        raise NotImplementedError("projected diagnostics support the configured E-step count only")
    if snapshot is None:
        snapshot = model.build_diagnostic_snapshot(token_ids)
    snapshot = model._validate_diagnostic_snapshot(token_ids, snapshot)
    if snapshot.conditional_capture is None:
        raise RuntimeError("projected belief trace requires the executed inference capture")
    return {
        "mu": torch.stack([belief.mu[0] for belief in snapshot.trace_states]),
        "sigma": torch.stack([belief.sigma[0] for belief in snapshot.trace_states]),
        "phi": torch.stack([belief.phi[0] for belief in snapshot.trace_states]),
        "free_energy": snapshot.trace_free_energy,
        "diagonal_covariance": model.cfg.diagonal_covariance,
        "inference_semantics": model.conditional_semantics,
        "free_energy_available": False,
        "free_energy_unavailable_reason": "routing and content use distinct conditional stages",
        "conditional_cost_available": model.cfg.relational_readout_mode != "predictive_content",
        **{f"relational_{key}": value
           for key in ("receiver_cost_before", "receiver_cost_after")
           if (value := snapshot.conditional_capture.get(f"relational_{key}")) is not None},
    }


@torch.no_grad()
@_block_mlp_diagnostic_eval
def across_layer_belief_trace(
    model,
    token_ids: torch.Tensor,           # (B, N) token ids; only sequence 0 is used

    *,
    snapshot: 'Optional[DiagnosticSnapshot]' = None,
) -> Dict[str, torch.Tensor]:
    r"""Per-LAYER output beliefs of sequence 0, read from one diagnostic snapshot.

    Returns ``mu`` (L, N, K), ``sigma`` (L, N, [K]), the cumulative affine-invariant SPD geodesic
    distance ``d_ai`` (L,) of each layer's covariance from layer 0, and per-layer mean effective
    rank ``effective_rank`` (L,). Shows how the belief geometry transforms with inference depth.
    """
    cfg = model.cfg
    if snapshot is None:
        snapshot = model.build_diagnostic_snapshot(token_ids)
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
    r"""Non-finite fractions of every layer's captured intermediates + the worst covariance conditioning.

    Reads sequence 0 of each layer's converged belief :math:`(\mu_\ell, \Sigma_\ell, \phi_\ell)` and of each
    layer's own routing capture (the pre-update cost ``score_energy_pre`` and weights ``beta_pre`` of
    ``snapshot.layer_conditional_captures``; the shared ``conditional_capture`` holds only the LAST block's)
    for :math:`\ell = 1, \dots, L`. Each field :math:`x` is pooled over the layers,

    .. math::

        \mathrm{nan}_x = \frac{\sum_\ell \#\{\text{non-finite entries of } x_\ell\}}{\sum_\ell |x_\ell|},

    reported as ``nan_<x>`` with the pooled entry count ``numel_<x>`` :math:`= \sum_\ell |x_\ell|` (so a
    fraction converts exactly to "k of N"), together with ``max_condition``
    :math:`= \max_\ell \max_i \kappa(\Sigma_{\ell, i})`, the spectral condition number over every layer's
    covariances in the family's layout, and ``n_layers`` :math:`= L`. A near-zero finiteness map certifies
    stability is genuine convergence, not masked blow-ups. (Numerical-FALLBACK activation counters --
    safe_cholesky jitter rounds, pinv fallbacks -- would require instrumenting numerics and are
    left to a future pass.)
    """
    if snapshot is None:
        snapshot = model.build_diagnostic_snapshot(token_ids)
    snapshot = model._validate_diagnostic_snapshot(token_ids, snapshot)
    beliefs = [_snapshot_sequence(belief) for belief in snapshot.layer_converged]
    captures = tuple(snapshot.layer_conditional_captures)
    if not captures:
        # A one-capture snapshot carries only the shared capture, which is the last block's: it pairs with
        # the converged belief only when there is one layer.
        if snapshot.conditional_capture is None:
            raise RuntimeError("projected numerical health requires the executed inference capture")
        if len(beliefs) != 1:
            raise RuntimeError(
                f"projected numerical health needs one routing capture per layer; this {len(beliefs)}-layer "
                "snapshot carries only the last layer's")
        captures = (snapshot.conditional_capture,)
    if len(captures) != len(beliefs):
        raise ValueError(
            f"snapshot has {len(beliefs)} converged layers but {len(captures)} routing captures")
    fields: Dict[str, List[torch.Tensor]] = {
        "mu":     [belief.mu for belief in beliefs],                          # each (N, K)
        "sigma":  [belief.sigma for belief in beliefs],                       # each (N, K) or (N, K, K)
        "phi":    [belief.phi for belief in beliefs],
        "energy": [capture["score_energy_pre"][0] for capture in captures],   # each (H, N, N), sequence 0
        "beta":   [capture["beta_pre"][0] for capture in captures],           # each (H, N, N), sequence 0
    }
    health: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for name, tensors in fields.items():
        numel = sum(int(tensor.numel()) for tensor in tensors)
        nonfinite = sum(int((~torch.isfinite(tensor)).sum()) for tensor in tensors)
        health[f"nan_{name}"] = float(nonfinite) / numel if numel else 0.0
        counts[f"numel_{name}"] = numel
    conditions = torch.stack([                                                # (L,) per-layer maxima
        metrics.belief_spectrum(
            belief.sigma, diagonal=model.cfg.diagonal_covariance, eps=model.cfg.eps,
            family=model.cfg.family)["condition"].max()
        for belief in beliefs])
    return {
        **health,
        "max_condition": float(conditions.max()),
        **counts,
        "n_layers": len(beliefs),
    }


def guard_denominators(
    sigma:    torch.Tensor,                    # (..., K) diagonal OR (..., K, K) full covariance
    energy:   torch.Tensor,                    # (..., N, N) pairwise energies
    self_div: Optional[torch.Tensor] = None,   # (..., N) or (..., N, K) self-divergences

    *,
    diagonal: Optional[bool] = None,
) -> Dict[str, int]:
    r"""Entry counts behind each :func:`vfe3.metrics.guard_saturation` fraction.

    The variance guards examine the covariance spectrum: one value per coordinate of a diagonal
    covariance (``sigma.numel()``) and ``K`` eigenvalues per ``K x K`` full covariance
    (``sigma.numel() / K``). The energy guard examines every pair of the ``(..., N, N)`` grid,
    masked pairs included, and the self-divergence guard every self-divergence entry. A count
    multiplied by its fraction gives the number of pinned entries. ``diagonal`` is resolved exactly
    as ``guard_saturation`` resolves it (explicit value, else a trailing square pair means full).
    """
    full = (not diagonal) if diagonal is not None else (
        sigma.dim() >= 2 and sigma.shape[-1] == sigma.shape[-2])
    spectrum = int(sigma.numel() // sigma.shape[-1]) if full else int(sigma.numel())
    counts = {
        "sigma_floor_frac":  spectrum,
        "sigma_ceil_frac":   spectrum,
        "energy_klmax_frac": int(energy.numel()),
    }
    if self_div is not None:
        counts["selfdiv_klmax_frac"] = int(self_div.numel())
    return counts


@torch.no_grad()
def vocabulary_sigma_trace(
    model:  torch.nn.Module,

    *,
    chunk:  int = 1024,
) -> torch.Tensor:
    r"""Per-type encode covariance trace :math:`\mathrm{tr}\,\Sigma(v)` for every vocabulary id ``v``.

    Encodes the ids ``0..V-1`` through ``model.prior_bank.encode`` in chunks and reduces with
    :func:`vfe3.metrics.sigma_trace` (the same reduction as :func:`belief_ce_bank`). On a route whose
    covariance is frozen at encode this is the covariance every occurrence of ``v`` carries, so its
    coefficient of variation over the vocabulary is the type-level spread, unweighted by frequency.
    Returns a CPU float64 tensor of shape (V,).
    """
    cfg = model.cfg
    device = _model_device(model)
    traces = []
    for start in range(0, int(cfg.vocab_size), int(chunk)):
        ids = torch.arange(start, min(start + int(chunk), int(cfg.vocab_size)),
                           device=device).unsqueeze(0)                       # (1, n)
        sigma = model.prior_bank.encode(ids).sigma                           # (1, n, K) or (1, n, K, K)
        traces.append(metrics.sigma_trace(
            sigma, diagonal=cfg.diagonal_covariance, family=cfg.family).reshape(-1).double().cpu())
    return torch.cat(traces)


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
    # The minimal-agent posterior applies no cost clamp, so only a non-finite cost reads as saturated.
    kl_max = (math.inf if getattr(model.cfg, "predictive_readout", "kernel") == "minimal_agent"
              else model.cfg.kl_max)
    # The covariance layout follows the family (audit 2026-10-09 round 2, item 23): a gaussian_block or
    # gaussian_full belief is a dense K x K matrix whose spectrum, not its entries, meets the variance guards.
    diagonal = bool(model.cfg.diagonal_covariance)
    guard = metrics.guard_saturation(
        out.sigma, cap["score_energy_pre"], out.mu.new_empty(0), diagonal=diagonal,
        eps=model.cfg.eps, sigma_max=model.cfg.sigma_max, kl_max=kl_max,
    )
    guard.pop("selfdiv_klmax_frac", None)  # no self-energy was evaluated by this diagnostic
    state = {
        "mu": out.mu, "sigma": out.sigma, "phi": out.phi,
        "energy": cap["score_energy_pre"], "beta": cap["beta_pre"], "guard": guard,
        "guard_denominators": guard_denominators(out.sigma, cap["score_energy_pre"], diagonal=diagonal),
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


def _dense_frame(frame_tensor: torch.Tensor, block_diagonal: bool) -> torch.Tensor:
    r"""Expand compact head blocks ``(..., N, H, d, d)`` to the dense ``(..., N, K, K)`` frame."""
    if not block_diagonal:
        return frame_tensor
    H, d = frame_tensor.shape[-3], frame_tensor.shape[-1]
    eye_h = torch.eye(H, dtype=frame_tensor.dtype, device=frame_tensor.device)
    return torch.einsum("...hab,hg->...hagb", frame_tensor, eye_h).reshape(
        *frame_tensor.shape[:-3], H * d, H * d)


def _act_on_frame(
    g:     torch.Tensor,                   # (..., N, H, d, d) head-block or (..., N, K, K) dense group elements
    frame: CanonicalFrameContext,
) -> CanonicalFrameContext:
    r"""Passive frame action :math:`U_i \mapsto g_i U_i`, :math:`U_i^{-1} \mapsto U_i^{-1} g_i^{-1}`.

    A dense ``g`` on a compact frame acts through its head blocks: a group whose frames are compact has every
    element block-diagonal over the heads (:meth:`vfe3.geometry.groups.GaugeGroup.head_block_generators`).
    A dense ``g`` with any nonzero entry outside those blocks raises ``ValueError`` instead of being truncated.
    """
    if frame.block_diagonal and g.dim() == frame.forward.dim() - 1:            # dense g on a compact frame
        H, d = frame.forward.shape[-3], frame.forward.shape[-1]
        on_block = torch.block_diag(*[torch.ones(d, d, dtype=torch.bool, device=g.device)] * H)   # (K, K)
        off_block = g[..., ~on_block]                                          # (..., N, K^2 - H d^2)
        if bool((off_block != 0).any()):
            raise ValueError(
                f"a dense group element acting on a compact {H}-head frame must be block-diagonal over the "
                f"heads; it has nonzero off-block entries (largest {float(off_block.abs().max()):.3e})")
        g = torch.stack([g[..., h * d:(h + 1) * d, h * d:(h + 1) * d] for h in range(H)], dim=-3)
    g_inv = torch.linalg.inv(g)
    if frame.block_diagonal:
        return CanonicalFrameContext(g @ frame.forward, frame.inverse @ g_inv, True)
    if g.dim() == frame.forward.dim():                                         # dense g on a dense frame
        return CanonicalFrameContext(g @ frame.forward, frame.inverse @ g_inv, False)
    G, G_inv = _dense_frame(g, True), _dense_frame(g_inv, True)
    return CanonicalFrameContext(G @ frame.forward, frame.inverse @ G_inv, False)


def _relative_deviation(reference: torch.Tensor, transformed: torch.Tensor) -> float:
    r"""``max |x' - x| / max |x|`` over entries finite in both (masked routing entries are skipped)."""
    finite = torch.isfinite(reference) & torch.isfinite(transformed)
    if not bool(finite.any()):
        return float("nan")
    scale = reference[finite].abs().max()
    delta = (transformed - reference)[finite].abs().max()
    return float(delta / scale.clamp(min=torch.finfo(reference.dtype).tiny))


@torch.no_grad()
def projected_gauge_equivariance(
    model:     torch.nn.Module,
    token_ids: torch.Tensor,               # (B, N) token ids; only sequence 0 is used

    *,
    n_draws:   int   = 3,
    scale:     float = 0.5,
    seed:      int   = 0,
) -> Dict[str, object]:
    r"""Gauge-equivariance residual of the executed projected routing and readout.

    The projected scorers read only the reference moments
    :math:`\xi_i = U_i^{-1}\mu_i` and :math:`C_i = U_i^{-1}\Sigma_i U_i^{-\top}`. A passive frame
    action by per-token head-block elements :math:`g_i = \exp(s A_i)`, with :math:`A_i` standard
    normal in each head block (``block_glk``), or :math:`A_i = \sum_a a_i^a T_a` with standard normal
    coordinates over the group generators otherwise (``so_n``: :math:`g_i \in \rho(SO(N))`),

    .. math::
        U_i \mapsto g_i U_i,\qquad \mu_i \mapsto g_i\mu_i,\qquad
        \Sigma_i \mapsto g_i\Sigma_i g_i^\top,

    leaves :math:`\xi_i` and :math:`C_i` fixed, so the routing score :math:`D_{ij}^h`, the weights
    :math:`\beta_{ij}^h`, and the readout pulled back to reference coordinates must not change.
    The layer-0 input of sequence 0 (encode, positional frame, and the realized routing and value
    frames of :meth:`VFEModel._relational_frames`) is scored through every layer's operators with
    the live functions: :func:`relational_predictive_readout` for ``predictive_content`` (mean
    :math:`U_i^{-1}m_i` and covariance :math:`U_i^{-1}\Sigma_i^+U_i^{-\top}`), and
    :func:`build_full_link_context` for ``full_link_update`` (the receiver force
    :math:`U_i^\top\nabla_{\mu_i}F`, the closed-form update in reference coordinates, and the
    receiver energy after adding back its exact shift :math:`\sum_i c_i\log|\det g_i|`, which the
    term :math:`-\tfrac{c_i}{2}\log\det\Sigma_i` produces). The receiver's prior moves with the belief.

    When the M-step self-coupling (``mstep_self_coupling_weight > 0``) or the canonical diagnostic
    (``diagnose_canonical_free_energy``) reads the self term, ``self_coupling`` is its per-token value
    :math:`\sum_k\alpha_i^{(k)}D_i^{(k)} + R(\alpha_i)` (one term for the per-position forms) against an
    independent fixed prior :math:`p_i`, a random draw moved with the belief,
    :math:`(\mu_{p,i},\Sigma_{p,i})\mapsto(g_i\mu_{p,i},g_i\Sigma_{p,i}g_i^\top)`. The summed
    :math:`D(q_i\|p_i)` is invariant, so a constant or per-position :math:`\alpha` sits at roundoff, while
    the ordered conditional KLs :math:`D_i^{(k)}` of ``state_dependent_per_coord`` move with the frame.

    Everything runs in float64 with the covariance lifted to a full matrix, because the diagonal
    family is closed only under diagonal :math:`g_i`; the residual certifies the
    reference-coordinate construction, not the diagonal family's local projection (on the lifted
    covariances the per-coordinate self term is the full family's ordered form). Each entry of
    ``residuals`` is a per-layer list of the maximum over ``n_draws`` of
    :math:`\max|x'-x|/\max|x|`, which should sit at float64 roundoff (``unit_roundoff``) times the
    conditioning of the frames and covariances.
    """
    from dataclasses import replace

    from vfe3.alpha_i import alpha_is_per_coord, self_coupling_alpha
    from vfe3.families.gaussian import FullGaussian
    from vfe3.free_energy import self_divergence_for_alpha
    from vfe3.gradients.full_link import (
        build_full_link_context, full_link_cost, full_link_gradients, full_link_minimizer)
    from vfe3.gradients.relational import relational_predictive_readout

    cfg = model.cfg
    tokens = token_ids[:1]
    n = tokens.shape[1]
    encoded = model.prior_bank.encode(tokens)                                  # (1, N, ...)
    beliefs = encoded._replace(phi=model._apply_pos_phi(encoded.phi),
                               right_phi=model._pos_phi_right(encoded.phi))
    rope = model._rope_rotation(n, tokens.device)
    _, frame32, value_frame32 = model._relational_frames(beliefs, rope)
    log_prior = model._attention_log_prior(n, tokens.device)
    log_prior = None if log_prior is None else log_prior.double()
    tau = attention_tau(_as_coeff(cfg.kappa_beta, tokens.device), model._belief_score_irrep_dims(),
                        device=tokens.device)
    # The receiver's prior moves with the belief (mu_p = mu, Sigma_p = Sigma), so D(q||p) = 0 and
    # alpha* = c0/(b0 + D) is the constant c0/b0 at this snapshot.
    receiver = any(ops.readout_mode != "predictive_content" for ops in model.relational_operators_by_layer)
    if receiver:
        ratio = torch.as_tensor(cfg.c0, dtype=torch.float64) / torch.as_tensor(cfg.b0, dtype=torch.float64)
        if bool((ratio != ratio.flatten()[0]).any()):
            raise ValueError("the gauge-equivariance witness needs a coordinate-uniform c0/b0: a "
                             "non-uniform per-coordinate alpha is not frame covariant")
        alpha = float(ratio.flatten()[0])

    def as64(frame: CanonicalFrameContext) -> CanonicalFrameContext:
        # The float64 frame contract needs a float64-exact inverse, not an upcast float32 one.
        forward = frame.forward.double()
        return CanonicalFrameContext(forward, torch.linalg.inv(forward), frame.block_diagonal)

    frame, value_frame = as64(frame32), as64(value_frame32)
    mu = beliefs.mu.double()                                                   # (1, N, K)
    sigma = beliefs.sigma.double()
    sigma = torch.diag_embed(sigma) if sigma.dim() == mu.dim() else sigma      # (1, N, K, K)
    H, d = len(model.group.irrep_dims), model.group.irrep_dims[0]

    # The self term against an independent prior: mu_p = mu + N(0, I) and Sigma_p = B Sigma B^T with B = I + A/4
    # block-diagonal over the irrep blocks (so a head-block family keeps its layout), from its own generator so
    # the frame draws below are unchanged.
    self_coupling_live = cfg.mstep_self_coupling_weight > 0.0 or bool(cfg.diagnose_canonical_free_energy)
    if self_coupling_live:
        prior_generator = torch.Generator().manual_seed(int(seed) + 1)
        mixing = torch.zeros_like(sigma)                                       # (1, N, K, K)
        offset = 0
        for k in model.group.irrep_dims:
            block = torch.randn(1, n, k, k, generator=prior_generator, dtype=torch.float64).to(mu.device)
            mixing[..., offset:offset + k, offset:offset + k] = (
                torch.eye(k, dtype=torch.float64, device=mu.device) + 0.25 * block)
            offset += k
        mu_p = mu + torch.randn(mu.shape, generator=prior_generator, dtype=torch.float64).to(mu.device)
        sigma_p = mixing @ sigma @ mixing.transpose(-1, -2)                    # (1, N, K, K)
        b0, c0 = (torch.as_tensor(v, dtype=torch.float64, device=mu.device) if isinstance(v, (list, tuple))
                  else v for v in (cfg.b0, cfg.c0))

    def self_coupling(mu: torch.Tensor, sigma: torch.Tensor, mu_p: torch.Tensor,
                      sigma_p: torch.Tensor) -> torch.Tensor:                 # (1, N)
        div = self_divergence_for_alpha(                                       # (1, N) or (1, N, K)
            FullGaussian(mu, sigma), FullGaussian(mu_p, sigma_p),
            kl_max=cfg.kl_max, eps=cfg.eps, lambda_alpha_mode=cfg.lambda_alpha_mode)
        alpha_i, regularizer = self_coupling_alpha(
            div, mode=cfg.lambda_alpha_mode, b0=b0, c0=c0)
        term = alpha_i * div + regularizer
        return term.sum(dim=-1) if alpha_is_per_coord(cfg.lambda_alpha_mode) else term

    def evaluate(ops, mu, sigma, frame, value_frame, link_state=None) -> Dict[str, torch.Tensor]:
        capture: Dict[str, torch.Tensor] = {}
        U_v = _dense_frame(value_frame.forward, value_frame.block_diagonal)    # (1, N, K, K)
        U_v_inv = _dense_frame(value_frame.inverse, value_frame.block_diagonal)
        if ops.readout_mode == "predictive_content":
            mean, cov = relational_predictive_readout(
                mu, sigma, frame, ops, tau=tau, kl_max=cfg.kl_max, log_prior=log_prior,
                value_frame=value_frame, capture=capture, return_covariance=True,
                floor_anchor=sigma,                     # the snapshot prior moves with the belief: Sigma_p = Sigma
                **({"link_state": link_state} if link_state is not None else {}))
            out = {"readout_mean": (U_v_inv @ mean.unsqueeze(-1)).squeeze(-1),
                   "readout_covariance": U_v_inv @ cov @ U_v_inv.transpose(-1, -2)}
        else:
            # The scalar form with (b0, c0) = (1, c0/b0) gives alpha* = c0/b0 at D = 0 on every family; the
            # profile is dropped, so the witness reads the receiver energy frozen at that alpha.
            context = replace(build_full_link_context(
                mu, sigma, mu, sigma, frame, ops, tau=tau, lambda_beta=cfg.lambda_beta,
                receiver_noise=ops.receiver_noise,
                kl_max=cfg.kl_max, log_prior=log_prior, value_frame=value_frame, capture=capture,
                lambda_alpha_mode="state_dependent", b0=1.0, c0=alpha), profile=None)
            force, _ = full_link_gradients(mu, sigma, context)                  # (1, N, K)
            new_mu, new_sigma = full_link_minimizer(
                mu, sigma, context, damping=ops.receiver_damping, sigma_max=None)
            out = {"receiver_force": (U_v.transpose(-1, -2) @ force.unsqueeze(-1)).squeeze(-1),
                   "receiver_update_mean": (U_v_inv @ new_mu.unsqueeze(-1)).squeeze(-1),
                   "receiver_update_covariance": U_v_inv @ new_sigma @ U_v_inv.transpose(-1, -2),
                   "receiver_energy": full_link_cost(mu, sigma, context),
                   "_c": context.c}
        out["routing_score"] = capture["score_energy_pre"]
        out["routing_weights"] = capture["beta_pre"]
        return out

    generator = torch.Generator().manual_seed(int(seed))
    layers = model.relational_operators_by_layer
    residuals: Dict[str, List[float]] = {}
    log_det_shift = 0.0                         # max |sum_i c_i log|det g_i|| added back to the receiver energy
    # Latent loop prior (content_link_loop_weight > 0): layer l reads layer l-1's inferred links, evaluated on the
    # same input. Those links are reference data (fixed by the passive action), so the moved evaluations reuse them.
    loops = any(getattr(ops, "content_link_loop_weight", 0.0) > 0.0 for ops in layers)
    previous_links = None
    links = lambda: (None if not loops else {} if previous_links is None else {"previous": previous_links})
    base_self_coupling = self_coupling(mu, sigma, mu_p, sigma_p) if self_coupling_live else None
    for layer_index, ops in enumerate(layers):
        base_links = links()
        base = evaluate(ops, mu, sigma, frame, value_frame, base_links)
        if self_coupling_live:
            base["self_coupling"] = base_self_coupling
        worst: Dict[str, float] = {}
        for _ in range(int(n_draws)):
            local = model.group.certified_block_local_basis()                  # (m, d, d) or None
            if model.group.name == "block_glk" and model.group.tying == "per_head":
                A = torch.randn(1, n, H, d, d, generator=generator, dtype=torch.float64)
                g = torch.linalg.matrix_exp(float(scale) * A).to(mu.device)    # (1, N, H, d, d)
            elif local is not None and H > 1:
                # Block-local group (block_so / co / sl / borel / unipotent / glmc): per-head g_h = exp(s a_h . L) from its
                # (m, d, d) local basis, without building the dense (H m, K, K) generators.
                a = torch.randn(1, n, H, local.shape[0], generator=generator, dtype=torch.float64)
                g = torch.linalg.matrix_exp(
                    float(scale) * torch.einsum("...hm,mij->...hij", a, local.to(torch.float64))).to(mu.device)
            else:
                # Draw from the selected group, g_i = exp(s a_i . T) with a_i standard normal over
                # its generators T (shared tying: one element acting in every block; untied so_n:
                # one element per block).
                gens = model.group.generators.to(dtype=torch.float64)          # (n_gen, K, K)
                a = torch.randn(1, n, gens.shape[0], generator=generator, dtype=torch.float64)
                # Kept dense: its irrep blocks may differ in size, and the dense frame takes it whole.
                g = torch.linalg.matrix_exp(
                    float(scale) * torch.einsum("...a,akl->...kl", a, gens)).to(mu.device)   # (1, N, K, K)
            G = _dense_frame(g, True) if g.dim() == mu.dim() + 2 else g        # (1, N, K, K)
            moved = evaluate(
                ops, (G @ mu.unsqueeze(-1)).squeeze(-1), G @ sigma @ G.transpose(-1, -2),
                _act_on_frame(g, frame), _act_on_frame(g, value_frame), links())
            if "receiver_energy" in base:
                # -c_i/2 log det(g Sigma g^T) = -c_i/2 log det Sigma - c_i log|det g_i|.
                shift = (base["_c"] * torch.linalg.slogdet(G)[1]).sum()
                moved["receiver_energy"] = moved["receiver_energy"] + shift
                log_det_shift = max(log_det_shift, abs(float(shift)))
            if self_coupling_live:
                moved["self_coupling"] = self_coupling(
                    (G @ mu.unsqueeze(-1)).squeeze(-1), G @ sigma @ G.transpose(-1, -2),
                    (G @ mu_p.unsqueeze(-1)).squeeze(-1), G @ sigma_p @ G.transpose(-1, -2))
            for key, value in base.items():
                if not key.startswith("_"):
                    worst[key] = max(worst.get(key, 0.0), _relative_deviation(value, moved[key]))
        for key, value in worst.items():
            residuals.setdefault(key, [float("nan")] * len(layers))[layer_index] = value
        if loops:
            previous_links = base_links.get("current")
    return {
        "residuals": residuals,
        "readout_mode": cfg.relational_readout_mode,
        "n_layers": len(layers),
        "n_draws": int(n_draws),
        "scale": float(scale),
        "dtype": "float64",
        "unit_roundoff": float(torch.finfo(torch.float64).eps),
        "log_det_shift_max": log_det_shift,     # 0 up to roundoff for det-1 groups (block_sl, block_unipotent, so_n)
    }


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
