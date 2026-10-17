r"""Natural-gradient M-step for the type-prior tables (``m_table_update_mode='fisher'``; audit 2026-10-10, item 22).

The prior of type :math:`v` is :math:`p_v = \mathcal N(\mu_v, \operatorname{diag}s_v)` with :math:`s_v = e^{l_v}`
(``mu_embed`` and ``sigma_log_embed``). Its Fisher metric in the coordinates :math:`(\mu_v, l_v)` is
:math:`G_v = \operatorname{diag}(1/s_v, 1/2)`. A step's loss is the mean over :math:`N` scored targets and row
:math:`v` is gathered at the :math:`n_v` input positions that hold :math:`v`, so the step preconditions that row
with :math:`(n_v/N)G_v`, the Fisher metric of the prior family :math:`p_v` counted once per occurrence. That
matrix is the exact Hessian of the self-coupling term :math:`\frac1N\sum_i \mathrm{KL}(q_i\|p_{x_i})` in
:math:`\mu_v` (and its expected Hessian in :math:`l_v`) and of nothing else; for the cross-entropy the step
trains it is a chosen preconditioner, not the metric of that objective (audit 2026-10-10 round 2, item 35).
The natural direction is

.. math::
    d_v = \frac{N}{n_v}\bigl(s_v\odot\partial_{\mu_v}\mathcal L,\ 2\,\partial_{l_v}\mathcal L\bigr),

the mean per-occurrence gradient carried to the tangent space of :math:`p_v`. The step :math:`-\eta d_v` is scaled
back to Fisher norm at most :math:`r`,

.. math::
    \|\Delta_v\|_{G_v}^2 = \sum_k\Bigl(\frac{\Delta\mu_{vk}^2}{s_{vk}} + \frac{\Delta l_{vk}^2}{2}\Bigr) \le r^2,

and is then multiplied by the schedule factor :math:`\mathrm{lr}_t/\mathrm{lr}_0` of the table's AdamW group. The
trust radius bounds the divergence only to second order, :math:`\mathrm{KL}(p_v^{+}\|p_v) = \tfrac12\|\Delta_v\|^2_{G_v}
+ O(r^3) \le r^2/2 + O(r^3)`: along :math:`+\Delta l` (one coordinate, :math:`\Delta l = x = \sqrt2 r`) the exact
divergence is :math:`\mathrm{KL} = \tfrac12(e^{x} - 1 - x)` and :math:`r^2/2 = x^2/4`, so the ratio
:math:`\mathrm{KL}/(r^2/2) = 2(e^{x} - 1 - x)/x^2` is 1.049 at :math:`r = 0.1`, 1.158 at :math:`r = 0.3` and 1.284
at :math:`r = 0.5` (audit 2026-10-10 round 2, item 36; the factor 2, audit 2026-10-16 F105). Rows absent from
the step (:math:`n_v = 0`) take no gradient step. Each group's decoupled decay is applied as AdamW applies it,
:math:`p \leftarrow p(1 - \mathrm{lr}_t\lambda)`, so ``mu_weight_decay`` and ``sigma_weight_decay`` keep their
meaning; the decay sits outside the trust-radius bound.
The groups marked ``table_fisher`` are excluded from gradient clipping (``vfe3.train.train_step``), so the realized
step is the one above. AdamW cannot express this update: its per-coordinate normalization cancels any slowly
varying diagonal preconditioner such as :math:`s_v`. The step keeps no optimizer state.

The base optimizer steps first, with the table gradients hidden, and the table step follows, so a frame step whose
metric reads the prior rows (``fisher_group``, :class:`vfe3.train.PhiFisherStatisticsProvider`) evaluates it at
the pre-step rows (audit 2026-10-10 round 2, item 34).
"""

from typing import Dict, Optional, Tuple

import torch

from vfe3.gauge_optim import GaugeManifoldAdamW
from vfe3.numerics import bounded_variance_from_log

# Log-row diagnostics of the table step (fixed key set; NaN when no row was active or the step was skipped).
# table_fisher_step_norm_mean is the realized Fisher norm of the scheduled gradient step,
# lr_t/lr_0 * min(||d||, r) per active row (decay excluded).
TABLE_FISHER_DIAGNOSTIC_KEYS: Tuple[str, ...] = (
    "table_fisher_active_rows",
    "table_fisher_trust_scale_mean",
    "table_fisher_trust_scale_min",
    "table_fisher_step_norm_mean",
)


class TableFisherStatistics:
    r"""The step's input tokens and scored-target count :math:`N`, set by ``train_step`` around the optimizer step.

    The token tensor is the whole step's input (the microbatches of ``grad_accum_steps > 1`` are its chunks along
    dim 0), so :math:`n_v` counts every position whose prior row is :math:`v`. The prior bank supplies the metric
    point :math:`s_v` and its variance floor.
    """

    def __init__(self, prior_bank: torch.nn.Module) -> None:
        self._prior_bank = prior_bank
        self._tokens: Optional[torch.Tensor] = None
        self._scored: Optional[torch.Tensor] = None

    def set_batch(
        self,
        tokens: Optional[torch.Tensor],                 # (B_tot, N) input ids, or None to clear
        scored: Optional[torch.Tensor],                 # () scored-target count, or None
    ) -> None:
        self._tokens = tokens
        self._scored = scored

    def batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._tokens is None or self._scored is None:
            raise RuntimeError("m_table_update_mode='fisher' needs the step's tokens; train_step sets them "
                               "immediately before the optimizer step")
        return self._tokens, self._scored

    def variances(self, dtype: torch.dtype) -> torch.Tensor:   # (V, K) s_v
        # The table itself, not a fresh detached copy: the forward already checked this (id, version) of the
        # table for overflow, so the cached verdict is reused instead of a new host read (audit 2026-10-10
        # round 2, item 57). The values are those of the detached copy.
        pb = self._prior_bank
        return bounded_variance_from_log(pb._prior_sigma_log_table().to(dtype), eps=pb.eps).detach()


def type_counts(
    tokens: torch.Tensor,                     # (...) input ids in [0, V)
    V:      int,
    device: torch.device,
) -> torch.Tensor:                            # (V,) int64 n_v
    r"""Occurrences :math:`n_v` of each type, ``torch.bincount(tokens, minlength=V)`` without its host read.

    ``bincount`` sizes its output from the data, which costs a device-to-host read on CUDA; accumulating into a
    length-``V`` buffer with ``index_add_`` gives the same int64 counts with no synchronization (audit 2026-10-10
    round 2, item 57).
    """
    ids = tokens.reshape(-1).to(device)
    return torch.zeros(V, dtype=torch.int64, device=device).index_add_(0, ids, torch.ones_like(ids))


def _schedule_factor(group: Dict[str, object]) -> float:
    r""":math:`\mathrm{lr}_t/\mathrm{lr}_0` of a group (``initial_lr`` is set by the LR scheduler); 0 for a frozen rate."""
    lr = float(group["lr"])
    initial = float(group.get("initial_lr", lr))
    return 0.0 if initial == 0.0 or lr == 0.0 else lr / initial


class _TableFisherMixin:
    r"""Steps the groups marked ``table_fisher`` (``"mu"``, ``"sigma"``) after the base optimizer's step."""

    def _init_table_fisher(
        self,
        statistics:   TableFisherStatistics,
        *,
        learning_rate: float,
        trust_radius:  float,
    ) -> None:
        self.table_statistics_provider = statistics
        self._table_fisher_lr = float(learning_rate)
        self._table_fisher_trust_radius = float(trust_radius)
        self._collect_table_diag = False
        self._table_diag: Dict[str, float] = {}

    def _table_fisher_groups(self) -> Tuple[Optional[Dict[str, object]], Optional[Dict[str, object]]]:
        marked = {group["table_fisher"]: group for group in self.param_groups if group.get("table_fisher")}
        return marked.get("mu"), marked.get("sigma")

    @torch.no_grad()
    def _table_fisher_step(self) -> None:
        mu_group, sigma_group = self._table_fisher_groups()
        mu = mu_group["params"][0] if mu_group is not None else None              # (V, K) mu_embed
        log_s = sigma_group["params"][0] if sigma_group is not None else None     # (V, K) sigma_log_embed
        live = [p for p in (mu, log_s) if p is not None and p.grad is not None]
        if self._collect_table_diag:
            self._table_diag = {}
        if not live:
            return
        # GradScaler contract of the fused kernel (as GaugeManifoldAdamW): skip an overflow step, else unscale.
        # The skip is decided on the device (no host read of found_inf; audit 2026-10-10 round 2, item 57).
        grad_scale, found_inf = getattr(self, "grad_scale", None), getattr(self, "found_inf", None)
        tokens, scored = self.table_statistics_provider.batch()
        reference = live[0]
        dtype = torch.promote_types(reference.dtype, torch.float32)
        V = reference.shape[0]
        skip = None if found_inf is None else found_inf.to(device=reference.device) != 0             # () bool
        count = type_counts(tokens, V, reference.device).to(dtype)                                   # (V,) n_v
        total = scored.to(device=reference.device, dtype=dtype).clamp_min(1.0)                       # () N
        weight = (total / count.clamp_min(1.0)).unsqueeze(-1)                                       # (V, 1) N/n_v
        s = self.table_statistics_provider.variances(dtype)                                         # (V, K)

        def gradient(p: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if p is None or p.grad is None:
                return None
            g = p.grad.to(dtype)
            return g if grad_scale is None else g / grad_scale.to(device=g.device, dtype=dtype)

        g_mu, g_l = gradient(mu), gradient(log_s)
        eta = self._table_fisher_lr
        step_mu = None if g_mu is None else -eta * weight * s * g_mu                                 # (V, K)
        step_l = None if g_l is None else -eta * weight * 2.0 * g_l                                  # (V, K)
        norm_sq_mu = None if step_mu is None else (step_mu.square() / s).sum(-1)                     # (V,)
        norm_sq_l = None if step_l is None else 0.5 * step_l.square().sum(-1)                        # (V,)
        norm_sq = torch.zeros(V, dtype=dtype, device=reference.device)
        if norm_sq_mu is not None:
            norm_sq = norm_sq + norm_sq_mu
        if norm_sq_l is not None:
            norm_sq = norm_sq + norm_sq_l
        norm = norm_sq.sqrt()                                                                        # (V,)
        active = count > 0
        trust = (self._table_fisher_trust_radius / norm.clamp_min(torch.finfo(dtype).tiny)).clamp(max=1.0)
        trust = torch.where(active, trust, torch.zeros_like(trust)).unsqueeze(-1)                    # (V, 1)
        realized_sq = torch.zeros(V, dtype=dtype, device=reference.device)    # (V,) ||scheduled step||^2 / trust^2
        for p, step, group, part in ((mu, step_mu, mu_group, norm_sq_mu), (log_s, step_l, sigma_group, norm_sq_l)):
            if step is None:
                continue
            decay = float(group["lr"]) * float(group.get("weight_decay", 0.0))
            if decay != 0.0:
                if skip is None:
                    p.mul_(1.0 - decay)
                else:
                    keep = torch.full((), 1.0 - decay, dtype=p.dtype, device=p.device)
                    p.mul_(torch.where(skip, torch.ones_like(keep), keep))
            factor = _schedule_factor(group)
            if factor != 0.0:
                update = (factor * trust * step).to(p.dtype)
                p.add_(update if skip is None else torch.where(skip, torch.zeros_like(update), update))
                realized_sq = realized_sq + factor ** 2 * part
            p.grad = None
        if self._collect_table_diag:
            accepted = trust.squeeze(-1)                                                             # (V,)
            n_active = active.sum().to(dtype)
            realized = accepted * realized_sq.sqrt()                                                 # (V,) realized norm
            values = torch.stack([
                n_active,
                torch.where(active, accepted, torch.zeros_like(accepted)).sum() / n_active,
                torch.where(active, accepted, torch.full_like(accepted, float("inf"))).amin(),
                torch.where(active, realized, torch.zeros_like(realized)).sum() / n_active,
                (skip if skip is not None else torch.zeros((), dtype=torch.bool, device=reference.device)).to(dtype),
            ]).cpu().tolist()
            if values[4]:                                                                            # overflow skip
                return
            nan = float("nan")
            self._table_diag = dict(zip(TABLE_FISHER_DIAGNOSTIC_KEYS, (
                values[0], *((nan, nan, nan) if values[0] == 0.0 else values[1:4]))))

    @torch.no_grad()
    def step(self, closure=None):                                       # type: ignore[override]
        if closure is not None:
            raise NotImplementedError("the table Fisher step consumes the table gradients after the base step, "
                                      "so a closure that re-backpropagates would step them twice")
        # The base step runs with the table gradients hidden (it skips a parameter whose grad is None, decay
        # included), so the frame step's Fisher metric reads the pre-step prior rows; the table step follows.
        tables = [group["params"][0] for group in self._table_fisher_groups() if group is not None]
        stashed = [(p, p.grad) for p in tables]
        for p, _ in stashed:
            p.grad = None
        try:
            result = super().step()                                     # type: ignore[misc]
        finally:
            for p, grad in stashed:
                p.grad = grad
        self._table_fisher_step()
        return result


class TableFisherAdamW(_TableFisherMixin, torch.optim.AdamW):
    r"""AdamW for every group except the type-prior tables, which take the natural-gradient step of this module."""

    def __init__(self, params, statistics: TableFisherStatistics, *, table_fisher_lr: float,
                 table_fisher_trust_radius: float, **kwargs) -> None:
        super().__init__(params, **kwargs)
        self._init_table_fisher(statistics, learning_rate=table_fisher_lr, trust_radius=table_fisher_trust_radius)


class TableFisherGaugeManifoldAdamW(_TableFisherMixin, GaugeManifoldAdamW):
    r""":class:`GaugeManifoldAdamW` (manifold phi groups) with the natural-gradient step on the type-prior tables."""

    def __init__(self, params, group, statistics: TableFisherStatistics, *, table_fisher_lr: float,
                 table_fisher_trust_radius: float, **kwargs) -> None:
        super().__init__(params, group, **kwargs)
        self._init_table_fisher(statistics, learning_rate=table_fisher_lr, trust_radius=table_fisher_trust_radius)
