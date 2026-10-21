r"""Opt-in exponential moving average of the trainable weights (``weight_ema_decay``; audit 2026-10-20 item 8).

With decay :math:`d \in (0, 1)` and update interval :math:`k`, the shadow :math:`\bar\theta` of every trainable
floating parameter is initialized to :math:`\theta_{t_0}` at the first accepted optimizer update :math:`t_0` at or
after the start step :math:`s = \max(1, \lceil f\,T\rceil)` (``weight_ema_start_frac`` :math:`f`, ``max_steps``
:math:`T`, steps counted from 1), and after every :math:`k`-th later accepted update

.. math::
    \bar\theta \leftarrow d^k\,\bar\theta + (1 - d^k)\,\theta,

computed as ``lerp(shadow, theta, 1 - d^k)``. For :math:`k = 1` this is the per-step EMA
:math:`\bar\theta_t = d\,\bar\theta_{t-1} + (1 - d)\theta_t`, whose weight on an iterate :math:`j` steps old is
:math:`(1-d)d^j` (time constant :math:`1/(1-d)` updates); for :math:`k > 1` it samples the trajectory every
:math:`k` updates with the decay of :math:`k` steps. A rejected update (non-finite, a loss-spike skip or a skipped
failed step) changes no weight, so it neither updates the shadow nor counts toward :math:`k`.

Only parameters with ``requires_grad`` and a floating dtype are averaged. Buffers and frozen parameters are not:
:meth:`WeightEMA.applied` swaps the shadow into the averaged parameters and leaves every other tensor of the model
at its raw value, then restores the raw parameters bit for bit from a copy. ``weight_ema_device='cpu'`` keeps the
shadow (and the raw copy taken during a swap) in host memory.

Checkpoints carry :meth:`WeightEMA.state_dict` under the key ``"weight_ema"``. The retired key ``"ema_state"`` stays
refused by :func:`vfe3.run_artifacts.load_checkpoint`: its runtime and semantics are gone, so a checkpoint written by
it cannot be continued exactly, and the new key does not reinterpret it.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Mapping, Optional

import torch

WEIGHT_EMA_CHECKPOINT_KEY: str = "weight_ema"
WEIGHT_EMA_SCHEMA_VERSION: int = 1


def weight_ema_start_step(start_frac: float, max_steps: int) -> int:
    r"""The first step (counted from 1) at which an accepted update may start the EMA: :math:`\max(1, \lceil fT\rceil)`."""
    return max(1, int(math.ceil(float(start_frac) * int(max_steps))))


class WeightEMA:
    r"""The shadow weights of the trainable floating parameters of ``model`` (module docstring)."""

    def __init__(
        self,
        model:           torch.nn.Module,

        *,
        decay:           float,
        start_step:      int,
        update_interval: int = 1,
        device:          str = "same",         # "same" | "cpu"
    ) -> None:
        if not 0.0 < float(decay) < 1.0:
            raise ValueError(f"decay must lie in (0, 1), got {decay!r}")
        if int(update_interval) < 1 or int(start_step) < 1:
            raise ValueError("update_interval and start_step must be >= 1")
        if device not in ("same", "cpu"):
            raise ValueError(f"device must be 'same' or 'cpu', got {device!r}")
        named = [(name, p) for name, p in model.named_parameters() if p.requires_grad and p.is_floating_point()]
        self.names: List[str] = [name for name, _ in named]
        self.params: List[torch.nn.Parameter] = [p for _, p in named]
        self.decay = float(decay)
        self.start_step = int(start_step)
        self.update_interval = int(update_interval)
        self.device = device
        self.shadow: Optional[List[torch.Tensor]] = None
        self.updates = 0               # shadow writes, the initialization included
        self.accepted = 0              # accepted updates since the initialization (the interval counter)
        self.initialized_step: Optional[int] = None

    @classmethod
    def from_config(cls, model: torch.nn.Module, cfg: Any) -> Optional["WeightEMA"]:
        r"""The EMA ``cfg`` asks for, or ``None`` when ``weight_ema_decay`` is unset."""
        decay = getattr(cfg, "weight_ema_decay", None)
        if decay is None:
            return None
        return cls(model, decay=float(decay),
                   start_step=weight_ema_start_step(cfg.weight_ema_start_frac, cfg.max_steps),
                   update_interval=int(cfg.weight_ema_update_interval), device=str(cfg.weight_ema_device))

    @property
    def active(self) -> bool:
        r"""Whether the shadow has been initialized."""
        return self.shadow is not None

    def _storage(self, p: torch.Tensor) -> torch.device:
        return torch.device("cpu") if self.device == "cpu" else p.device

    @torch.no_grad()
    def observe(self, step: int, accepted: bool) -> bool:
        r"""Account for completed step ``step`` (from 1); return whether the shadow was written."""
        if not accepted or int(step) < self.start_step:
            return False
        if self.shadow is None:
            self.shadow = [p.detach().to(self._storage(p), copy=True) for p in self.params]
            self.updates, self.accepted, self.initialized_step = 1, 0, int(step)
            return True
        self.accepted += 1
        if self.accepted % self.update_interval != 0:
            return False
        weight = 1.0 - self.decay ** self.update_interval
        sources = [p.detach() if self.device == "same" else p.detach().to("cpu") for p in self.params]
        torch._foreach_lerp_(self.shadow, sources, weight)
        self.updates += 1
        return True

    @contextmanager
    def applied(self) -> Iterator[None]:
        r"""Swap the shadow into the averaged parameters; restore the raw values bit for bit on exit."""
        if self.shadow is None:
            raise RuntimeError("the weight EMA has not started; there are no EMA weights to apply")
        with torch.no_grad():
            raw = [p.detach().to(self._storage(p), copy=True) for p in self.params]
            for p, s in zip(self.params, self.shadow):
                p.copy_(s)
        try:
            yield
        finally:
            with torch.no_grad():
                for p, r in zip(self.params, raw):
                    p.copy_(r)
            del raw

    def shadow_state(self) -> Dict[str, torch.Tensor]:
        r"""``{parameter name: shadow tensor}`` (the live tensors; empty before the start)."""
        return {} if self.shadow is None else dict(zip(self.names, self.shadow))

    def state_dict(self) -> Dict[str, Any]:
        r"""The checkpoint record (``"weight_ema"``): settings, counters and the shadow (``None`` before the start)."""
        return {
            "schema_version":   WEIGHT_EMA_SCHEMA_VERSION,
            "decay":            self.decay,
            "start_step":       self.start_step,
            "update_interval":  self.update_interval,
            "updates":          self.updates,
            "accepted":         self.accepted,
            "initialized_step": self.initialized_step,
            "names":            list(self.names),
            "shadow":           None if self.shadow is None else [s.detach() for s in self.shadow],
        }

    def validate_state(self, state: object) -> None:
        r"""Raise ``RuntimeError`` unless ``state`` is a record this EMA can continue (names, shapes, dtypes)."""
        if not isinstance(state, Mapping) or state.get("schema_version") != WEIGHT_EMA_SCHEMA_VERSION:
            raise RuntimeError("checkpoint weight_ema is not a schema-1 weight-EMA record")
        names = state.get("names")
        if list(names or ()) != self.names:
            raise RuntimeError("checkpoint weight_ema averages a different parameter set than this model")
        for key in ("updates", "accepted"):
            value = state.get(key)
            if type(value) is not int or value < 0:
                raise RuntimeError(f"checkpoint weight_ema {key} must be a non-negative int")
        shadow = state.get("shadow")
        if shadow is None:
            return
        if not isinstance(shadow, (list, tuple)) or len(shadow) != len(self.params):
            raise RuntimeError("checkpoint weight_ema shadow does not match the parameter list")
        for name, s, p in zip(self.names, shadow, self.params):
            if not isinstance(s, torch.Tensor) or s.shape != p.shape or s.dtype != p.dtype:
                raise RuntimeError(f"checkpoint weight_ema shadow of {name} does not match the parameter")

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        r"""Continue from a checkpoint record (validated first); the live settings (decay, start, interval) are kept."""
        self.validate_state(state)
        shadow = state.get("shadow")
        self.shadow = (None if shadow is None else
                       [s.to(self._storage(p), copy=True) for s, p in zip(shadow, self.params)])
        self.updates = int(state["updates"])
        self.accepted = int(state["accepted"])
        initialized = state.get("initialized_step")
        self.initialized_step = int(initialized) if isinstance(initialized, int) else None

    def summary(self) -> Dict[str, Any]:
        r"""Settings and counters for ``summary.json`` (no tensors)."""
        return {"weight_ema_decay": self.decay, "weight_ema_start_step": self.start_step,
                "weight_ema_update_interval": self.update_interval, "weight_ema_device": self.device,
                "weight_ema_updates": self.updates, "weight_ema_initialized_step": self.initialized_step}


# metrics.csv columns the weight EMA writes (vfe3.train) and :func:`best_ema_validation` reads.
WEIGHT_EMA_METRIC_COLUMNS = ("val_ema_ce", "val_ema_ppl", "weight_ema_updates")


def best_ema_validation(history: Any) -> Dict[str, Any]:
    r"""The lowest finite ``val_ema_ppl`` of the metrics rows, its step, CE and EMA update count (``None`` when absent).

    The EMA is not used for model selection (``best_model.pt`` stays on the raw weights); this records where the
    EMA weights' validation was best, from which a start fraction or decay choice can be read.
    """
    best: Dict[str, Any] = {"best_val_ema_ppl": None, "best_val_ema_ce": None, "best_val_ema_step": None,
                            "best_val_ema_updates": None}
    for row in history or ():
        ppl = row.get("val_ema_ppl") if isinstance(row, Mapping) else None
        if not isinstance(ppl, (int, float)) or isinstance(ppl, bool) or not math.isfinite(float(ppl)):
            continue
        if best["best_val_ema_ppl"] is None or float(ppl) < best["best_val_ema_ppl"]:
            best.update(best_val_ema_ppl=float(ppl), best_val_ema_ce=row.get("val_ema_ce"),
                        best_val_ema_step=row.get("step"), best_val_ema_updates=row.get("weight_ema_updates"))
    return best

