r"""Natural-gradient M-step for the type-prior tables (``m_table_update_mode='fisher'``; audit 2026-10-10, item 22).

The prior of type :math:`v` is :math:`p_v = \mathcal N(\mu_v, \operatorname{diag}s_v)` with :math:`s_v = e^{l_v}`
(``mu_embed`` and ``sigma_log_embed``). Its Fisher metric in the coordinates :math:`(\mu_v, l_v)` is
:math:`G_v = \operatorname{diag}(1/s_v, 1/2)`. A step's loss is the mean over :math:`N` scored targets and row
:math:`v` is gathered at the :math:`n_v` input positions that hold :math:`v`, so the metric of the step's
objective on that row is :math:`(n_v/N)G_v` and the natural direction is

.. math::
    d_v = \frac{N}{n_v}\bigl(s_v\odot\partial_{\mu_v}\mathcal L,\ 2\,\partial_{l_v}\mathcal L\bigr),

the mean per-occurrence gradient carried to the tangent space of :math:`p_v`. The step :math:`-\eta d_v` is scaled
back to Fisher norm at most :math:`r`,

.. math::
    \|\Delta_v\|_{G_v}^2 = \sum_k\Bigl(\frac{\Delta\mu_{vk}^2}{s_{vk}} + \frac{\Delta l_{vk}^2}{2}\Bigr) \le r^2,

so that :math:`\mathrm{KL}(p_v^{+}\|p_v) \approx \tfrac12\|\Delta_v\|^2_{G_v} \le r^2/2`, and is then multiplied by the
schedule factor :math:`\mathrm{lr}_t/\mathrm{lr}_0` of the table's AdamW group. Rows absent from the step
(:math:`n_v = 0`) take no gradient step. Each group's decoupled decay is applied as AdamW applies it,
:math:`p \leftarrow p(1 - \mathrm{lr}_t\lambda)`, so ``mu_weight_decay`` and ``sigma_weight_decay`` keep their meaning.
AdamW cannot express this update: its per-coordinate normalization cancels any slowly varying diagonal
preconditioner such as :math:`s_v`. The step keeps no optimizer state.
"""

from typing import Dict, Optional, Tuple

import torch

from vfe3.gauge_optim import GaugeManifoldAdamW
from vfe3.numerics import bounded_variance_from_log

# Log-row diagnostics of the table step (fixed key set; NaN when no row was active).
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
        pb = self._prior_bank
        return bounded_variance_from_log(pb._prior_sigma_log_table().detach().to(dtype), eps=pb.eps)


def _schedule_factor(group: Dict[str, object]) -> float:
    r""":math:`\mathrm{lr}_t/\mathrm{lr}_0` of a group (``initial_lr`` is set by the LR scheduler); 0 for a frozen rate."""
    lr = float(group["lr"])
    initial = float(group.get("initial_lr", lr))
    return 0.0 if initial == 0.0 or lr == 0.0 else lr / initial


class _TableFisherMixin:
    r"""Steps the groups marked ``table_fisher`` (``"mu"``, ``"sigma"``) before the base optimizer's step."""

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

    @torch.no_grad()
    def _table_fisher_step(self) -> None:
        marked = {group["table_fisher"]: group for group in self.param_groups if group.get("table_fisher")}
        mu_group, sigma_group = marked.get("mu"), marked.get("sigma")
        mu = mu_group["params"][0] if mu_group is not None else None              # (V, K) mu_embed
        log_s = sigma_group["params"][0] if sigma_group is not None else None     # (V, K) sigma_log_embed
        live = [p for p in (mu, log_s) if p is not None and p.grad is not None]
        if self._collect_table_diag:
            self._table_diag = {}
        if not live:
            return
        # GradScaler contract of the fused kernel (as GaugeManifoldAdamW): skip an overflow step, else unscale.
        grad_scale, found_inf = getattr(self, "grad_scale", None), getattr(self, "found_inf", None)
        if found_inf is not None and bool(found_inf):
            for p in live:
                p.grad = None
            return
        tokens, scored = self.table_statistics_provider.batch()
        reference = live[0]
        dtype = torch.promote_types(reference.dtype, torch.float32)
        V = reference.shape[0]
        count = torch.bincount(tokens.reshape(-1).to(reference.device), minlength=V)[:V].to(dtype)   # (V,) n_v
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
        norm_sq = torch.zeros(V, dtype=dtype, device=reference.device)
        if step_mu is not None:
            norm_sq = norm_sq + (step_mu.square() / s).sum(-1)
        if step_l is not None:
            norm_sq = norm_sq + 0.5 * step_l.square().sum(-1)
        norm = norm_sq.sqrt()                                                                        # (V,)
        active = count > 0
        trust = (self._table_fisher_trust_radius / norm.clamp_min(torch.finfo(dtype).tiny)).clamp(max=1.0)
        trust = torch.where(active, trust, torch.zeros_like(trust)).unsqueeze(-1)                    # (V, 1)
        for p, step, group in ((mu, step_mu, mu_group), (log_s, step_l, sigma_group)):
            if step is None:
                continue
            decay = float(group["lr"]) * float(group.get("weight_decay", 0.0))
            if decay != 0.0:
                p.mul_(1.0 - decay)
            factor = _schedule_factor(group)
            if factor != 0.0:
                p.add_((factor * trust * step).to(p.dtype))
            p.grad = None
        if self._collect_table_diag:
            n_active = active.sum()
            accepted = trust.squeeze(-1)[active]
            if accepted.numel() == 0:
                nan = float("nan")
                self._table_diag = {"table_fisher_active_rows": 0.0, "table_fisher_trust_scale_mean": nan,
                                    "table_fisher_trust_scale_min": nan, "table_fisher_step_norm_mean": nan}
            else:
                values = torch.stack([n_active.to(dtype), accepted.mean(), accepted.amin(),
                                      (accepted * norm[active]).mean()]).cpu().tolist()
                self._table_diag = dict(zip(TABLE_FISHER_DIAGNOSTIC_KEYS, values))

    @torch.no_grad()
    def step(self, closure=None):                                       # type: ignore[override]
        if closure is not None:
            raise NotImplementedError("the table Fisher step consumes the table gradients before the base step, "
                                      "so a closure that re-backpropagates would step them twice")
        self._table_fisher_step()
        return super().step()                                           # type: ignore[misc]


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
