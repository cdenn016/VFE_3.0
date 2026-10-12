r"""Attention-prior registry for VFE_3.0 (the prior pi_ij over keys).

Each prior returns a LOG-PRIOR BIAS B_ij added to the attention logits:
    beta*_ij = softmax_j(B_ij - E_ij / tau),
and the normalized prior used in the attention-entropy term is pi = softmax_j(B).
  causal   B = 0 (j<=i), -inf (j>i)   -> uniform over the causal active set.
  causal_alibi  B_ij = -slope*(i-j) (j<=i), -inf (j>i) -> causal + ALiBi (Press et al.).
  causal_noself / causal_alibi_noself -> the same with the self key j=i also masked
    (except (0,0); see prior_causal_noself for the attention-sink rationale).
  causal_windowed -> hard causal local band 0 <= i-j <= window.
Every registered prior is causal for next-token scoring; the non-causal uniform, alibi and windowed
priors were removed on 2026-10-12, and the T5 relative-position bias in the round-2 audit.
Config-selected so a new prior slots in by register_prior without editing the free-energy call
site; a non-causal registration is refused by training (vfe3/train.py).
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch


def _press_slopes(
    n_heads: int,
    base:    float,
    device:  'torch.device | str | None',
    dtype:   torch.dtype,
) -> torch.Tensor:                          # (H,)
    r"""Reference Press et al. ALiBi slopes, including non-power-of-two head counts.

    For a power-of-two ``H``, ``slope_h = 2^(-8h/H)``. Otherwise the reference construction
    concatenates the nearest lower-power-of-two schedule with the even-indexed entries of the
    next power-of-two schedule. ``base`` scales the completed reference schedule.
    """
    if n_heads < 1:
        raise ValueError(f"n_heads must be positive, got {n_heads}")

    def _power_of_two_slopes(count: int) -> torch.Tensor:
        h = torch.arange(1, count + 1, device=device, dtype=dtype)
        return torch.pow(2.0, -8.0 * h / count)

    closest_power = 1 << (n_heads.bit_length() - 1)
    slopes = _power_of_two_slopes(closest_power)
    if closest_power != n_heads:
        extra = _power_of_two_slopes(2 * closest_power)[::2][:n_heads - closest_power]
        slopes = torch.cat((slopes, extra), dim=0)
    return base * slopes


_PRIORS: Dict[str, Callable] = {}
_PRIOR_CAUSALITY: Dict[str, tuple[Callable, bool, Optional[str]]] = {}


@dataclass(frozen=True)
class AttentionPriorRegistration:
    """One active prior builder and its next-token causality declaration."""

    callable: Callable
    causal_for_next_token_scoring: bool
    noncausal_when_config_true: Optional[str] = None

    def is_causal_for_next_token_scoring(self, cfg: object) -> bool:
        if not self.causal_for_next_token_scoring:
            return False
        if self.noncausal_when_config_true is None:
            return True
        return not bool(getattr(cfg, self.noncausal_when_config_true, False))


def register_prior(
    name: str, *, causal_for_next_token_scoring: Optional[bool] = None,
    noncausal_when_config_true: Optional[str] = None, override: bool = False,
) -> Callable:
    """Decorator registering an attention-prior builder -> log-prior bias (Nq, Nk).

    Duplicate keys fail closed (audit 2026-07-01 F12): a second registration under an
    existing name silently shadowed the first. Pass ``override=True`` to replace deliberately.
    """
    def _wrap(fn: Callable) -> Callable:
        if name in _PRIORS and not override:
            raise KeyError(f"attention prior {name!r} already registered; pass override=True to replace")
        causal = (
            bool(getattr(fn, "_vfe3_causal_for_next_token_scoring", False))
            if causal_for_next_token_scoring is None else bool(causal_for_next_token_scoring)
        )
        config_key = (
            getattr(fn, "_vfe3_noncausal_when_config_true", None)
            if noncausal_when_config_true is None else noncausal_when_config_true
        )
        if config_key is not None and not causal:
            raise ValueError(
                "noncausal_when_config_true requires causal_for_next_token_scoring=True"
            )
        setattr(fn, "_vfe3_causal_for_next_token_scoring", causal)
        setattr(fn, "_vfe3_noncausal_when_config_true", config_key)
        _PRIORS[name] = fn
        _PRIOR_CAUSALITY[name] = (fn, causal, config_key)
        return fn
    return _wrap


def get_prior(name: str) -> Callable:
    """Return the registered attention-prior builder (KeyError if absent)."""
    if name not in _PRIORS:
        raise KeyError(f"no attention prior {name!r}; available: {sorted(_PRIORS)}")
    return _PRIORS[name]


def get_prior_registration(name: str) -> AttentionPriorRegistration:
    """Return the callable plus fail-closed causal metadata for ``name``."""
    fn = get_prior(name)
    record = _PRIOR_CAUSALITY.get(name)
    if record is not None and record[0] is fn:
        _, causal, config_key = record
    else:
        causal = bool(getattr(fn, "_vfe3_causal_for_next_token_scoring", False))
        config_key = getattr(fn, "_vfe3_noncausal_when_config_true", None)
    return AttentionPriorRegistration(fn, causal, config_key)


@register_prior("causal", causal_for_next_token_scoring=True)
def prior_causal(
    n_query: int,
    n_key:   int,

    *,
    device:  'torch.device | str | None' = None,
    dtype:   torch.dtype                  = torch.float32,
    **kwargs,
) -> torch.Tensor:
    r"""Causal prior: 0 for key j <= query i, -inf for j > i."""
    i = torch.arange(n_query, device=device).unsqueeze(-1)
    j = torch.arange(n_key, device=device).unsqueeze(0)
    allowed = j <= i
    B = torch.zeros(n_query, n_key, device=device, dtype=dtype)
    return B.masked_fill(~allowed, float("-inf"))


@register_prior("causal_noself", causal_for_next_token_scoring=True)
def prior_causal_noself(
    n_query: int,
    n_key:   int,

    *,
    device:  'torch.device | str | None' = None,
    dtype:   torch.dtype                  = torch.float32,
    **kwargs,
) -> torch.Tensor:
    r"""Causal prior with the SELF key masked: 0 for j < i, -inf for j > i AND for j == i > 0.

    Rationale: the flat cocycle gives Omega_ii = I, so the self energy E_ii = D(q_i || q_i) ~ 0
    permanently -- the diagonal is a STRUCTURAL attention sink the softmax can never unlearn.
    Masking j == i removes it. Entry (0, 0) stays allowed by construction: row 0 has no other
    causal key, and an all--inf logits row makes the softmax NaN. Equals `causal` off-diagonal.
    """
    i = torch.arange(n_query, device=device).unsqueeze(-1)
    j = torch.arange(n_key, device=device).unsqueeze(0)
    allowed = (j < i) | ((i == 0) & (j == 0))
    B = torch.zeros(n_query, n_key, device=device, dtype=dtype)
    return B.masked_fill(~allowed, float("-inf"))


@register_prior("causal_alibi", causal_for_next_token_scoring=True)
def prior_causal_alibi(
    n_query:     int,
    n_key:       int,

    *,
    n_heads:     int                          = 1,
    alibi_slope: float                        = 0.5,
    device:      'torch.device | str | None'  = None,
    dtype:       torch.dtype                  = torch.float32,
    **kwargs,
) -> torch.Tensor:
    r"""Causal + ALiBi prior (Press et al. 2022 faithful, autoregressive form), per head.

    Returns shape ``(H, N_q, N_k)`` where ``H = n_heads``. Each head uses Press et al.'s
    reference slope schedule; the causal mask is applied uniformly across heads:

        B_hij = -slope_h * |i - j|   for j <= i   (allowed lower triangle),
        B_hij = -inf                  for j >  i   (causal mask).

    On the causal triangle ``i - j >= 0`` so ``|i - j| == i - j``, matching the
    directed ``-slope*(i-j)`` convention of the original ``causal_alibi`` prior.
    """
    i       = torch.arange(n_query, device=device).unsqueeze(-1)
    j       = torch.arange(n_key,   device=device).unsqueeze(0)
    dist    = (i - j).abs().to(dtype)                               # (N_q, N_k)
    slopes  = _press_slopes(n_heads, alibi_slope, device, dtype)    # (H,)
    B       = (-slopes.view(n_heads, 1, 1) * dist)                  # (H, N_q, N_k)
    allowed = (j <= i)                                              # (N_q, N_k)
    return B.masked_fill(~allowed.unsqueeze(0), float("-inf"))      # (H, N_q, N_k)


@register_prior("causal_alibi_noself", causal_for_next_token_scoring=True)
def prior_causal_alibi_noself(
    n_query:     int,
    n_key:       int,

    *,
    n_heads:     int                          = 1,
    alibi_slope: float                        = 0.5,
    device:      'torch.device | str | None'  = None,
    dtype:       torch.dtype                  = torch.float32,
    **kwargs,
) -> torch.Tensor:
    r"""Causal + ALiBi prior (per-head Press et al. slopes) with the SELF key masked.

    Identical to ``causal_alibi`` -- ``B_hij = -slope_h * (i - j)`` on the allowed set, -inf on the
    causal complement -- except the diagonal j == i is ALSO masked to -inf (see ``causal_noself``:
    the flat cocycle's Omega_ii = I makes E_ii ~ 0, a structural attention sink). Entry (0, 0)
    stays allowed since row 0 has no other causal key (an all--inf logits row NaNs the softmax);
    its ALiBi bias there is -slope_h * 0 = 0. Returns ``(H, N_q, N_k)``, ``H = n_heads``.
    """
    i       = torch.arange(n_query, device=device).unsqueeze(-1)
    j       = torch.arange(n_key,   device=device).unsqueeze(0)
    dist    = (i - j).abs().to(dtype)                               # (N_q, N_k)
    slopes  = _press_slopes(n_heads, alibi_slope, device, dtype)    # (H,)
    B       = (-slopes.view(n_heads, 1, 1) * dist)                  # (H, N_q, N_k)
    allowed = (j < i) | ((i == 0) & (j == 0))                       # diagonal dropped (except (0,0))
    return B.masked_fill(~allowed.unsqueeze(0), float("-inf"))      # (H, N_q, N_k)


@register_prior("causal_windowed", causal_for_next_token_scoring=True)
def prior_causal_windowed(
    n_query: int,
    n_key:   int,

    *,
    window:  int                          = 128,
    device:  'torch.device | str | None'  = None,
    dtype:   torch.dtype                   = torch.float32,
    **kwargs,
) -> torch.Tensor:
    r"""Causal windowed (sliding-window local-attention) prior.

        B_ij = 0      for 0 <= i - j <= window   (past keys within the band),
             = -inf   otherwise                  (future keys, or keys further back than window).
    It forbids the future like `causal` and additionally drops keys more than ``window`` steps in
    the past, the sliding-window local attention of Longformer/Mistral. ``window`` is threaded from
    the model (``window=cfg.attention_window``, validated in ``config.py``). It is inert at
    ``attention_window >= max_seq_len - 1``, where it equals plain ``causal``."""
    i = torch.arange(n_query, device=device).unsqueeze(-1)
    j = torch.arange(n_key, device=device).unsqueeze(0)
    d = i - j
    allowed = (d >= 0) & (d <= window)
    B = torch.zeros(n_query, n_key, device=device, dtype=dtype)
    return B.masked_fill(~allowed, float("-inf"))


def attention_log_prior(
    name:    str,
    n_query: int,
    n_key:   int,

    *,
    device:  'torch.device | str | None'  = None,
    dtype:   torch.dtype                   = torch.float32,
    **kwargs,                              # variant params (e.g. alibi's `slope`)
) -> torch.Tensor:
    r"""Dispatch to the registered attention-prior `name`; returns log-prior bias (Nq, Nk).

    `device`/`dtype` are universal (every builder consumes them); variant-specific
    params (e.g. ALiBi's `slope`) flow through **kwargs, so a new prior with a novel
    param (windowed width, learned-bias handle, ...) slots in by `register_prior` +
    config without editing this dispatcher.
    """
    return get_prior(name)(n_query, n_key, device=device, dtype=dtype, **kwargs)
