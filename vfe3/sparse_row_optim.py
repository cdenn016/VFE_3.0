r"""Row-lazy AdamW for a per-type table (``m_phi_sparse_rows=True``; audit 2026-10-20 items 2, 10, 49, mechanism M2).

Dense AdamW updates every row of ``phi_embed`` on every step. A row :math:`v` whose gradient is zero this step still
takes the decayed momentum step :math:`\mathrm{lr}\,\hat m_v/(\sqrt{\hat v_v}+\epsilon)` and the decoupled decay, and a
row revisited after a long gap moves about :math:`\min(\sqrt T, 31.6)\,\mathrm{lr}` per coordinate per appearance,
whatever the size of its gradient, because :math:`m` decays as :math:`\beta_1^k` while :math:`\sqrt v` decays as
:math:`\beta_2^{k/2}` and the bias correction reads the global step count :math:`T` (audit 2026-10-20 item 2).

The row-lazy step keeps the dense moment tables and a per-row step count :math:`k_v`. Let :math:`A_t` be the rows
touched at step :math:`t`: those among the step's input tokens, together with every row whose gradient is not
identically zero (a channel that reaches a row outside the batch is never dropped). For :math:`v\in A_t`,

.. math::
    k_v &\leftarrow k_v + 1, \\
    \theta_v &\leftarrow \theta_v(1 - \mathrm{lr}_t\lambda), \\
    m_v &\leftarrow \beta_1 m_v + (1-\beta_1)g_v, \qquad
    s_v \leftarrow \beta_2 s_v + (1-\beta_2)g_v\odot g_v, \\
    \theta_v &\leftarrow \theta_v - \frac{\mathrm{lr}_t}{1-\beta_1^{k_v}}\,
        \frac{m_v}{\sqrt{s_v}/\sqrt{1-\beta_2^{k_v}} + \epsilon},

and every :math:`v\notin A_t` keeps :math:`(\theta_v, m_v, s_v, k_v)` exactly. When every row is touched at every step
:math:`k_v = T` and this is ``torch.optim.AdamW`` (the per-row arithmetic is the single-tensor kernel's, so the two agree
to rounding). The row then moves only on the steps where it is seen, and its bias correction is that of a fresh
optimizer after :math:`k_v` steps, so the first touch of a row moves it by :math:`\mathrm{lr}` per coordinate instead of
the dense first-touch step that grows with :math:`T`. This is the lazy Adam of ``tf.contrib.opt.LazyAdamOptimizer``
with a per-row (not global) bias correction; it changes the dynamics of rarely visited rows and is not a dense-AdamW
equivalent (audit 2026-10-20 item 49).

State per table: ``exp_avg`` and ``exp_avg_sq`` (dense, as AdamW), ``step`` (the number of optimizer steps, a scalar
tensor laid out as AdamW lays it out, so a checkpoint resumed by dense AdamW keeps a valid clock) and ``row_step``
(:math:`k_v`, shape ``(V,)``, in the parameter's floating dtype, exact up to :math:`2^{24}` steps in float32). All
four are checkpointed with the optimizer state; a dense AdamW checkpoint resumed here starts every row at
:math:`k_v = T`.

The moment tables stay dense, so the memory of item 49 is not reduced; the step gathers the touched rows (one host
read of their count per step) and writes them back, so it reads and writes :math:`O(|A_t|\,P)` entries of the table
and its moments.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch

from vfe3.table_optim import _TableFisherMixin


class _RowLazyAdamWMixin:
    r"""Steps the groups marked ``sparse_rows`` with row-lazy AdamW after the base optimizer's step."""

    def _init_row_lazy(
        self,
        statistics: Optional[Any],                 # provider with step_tokens() -> (B_tot, N) ids or None
    ) -> None:
        if statistics is not None:
            self.table_statistics_provider = statistics
        self._row_lazy_statistics = statistics

    def _row_lazy_parameters(self) -> List[Tuple[torch.Tensor, Dict[str, Any]]]:
        return [(p, group) for group in self.param_groups if group.get("sparse_rows") for p in group["params"]]

    def _row_lazy_state(
        self,
        p:     torch.Tensor,                       # (V, ...) table
        group: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        state = self.state[p]
        if len(state) == 0:
            on_device = bool(group.get("fused")) or bool(group.get("capturable"))
            # The scalar clock is laid out as torch.optim.AdamW lays it out, so a dense optimizer can resume it.
            state["step"] = (torch.zeros((), dtype=torch.float32, device=p.device) if on_device
                             else torch.tensor(0.0, dtype=torch.float32))
            state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
        if "row_step" not in state:
            # A fresh table, or a dense AdamW state resumed here: every row has taken the global count.
            state["row_step"] = torch.full((p.shape[0],), float(state["step"]), dtype=p.dtype, device=p.device)
        return state

    def _touched_rows(
        self,
        gradient: torch.Tensor,                    # (V, P) flattened table gradient
    ) -> torch.Tensor:                             # (R,) int64 touched row ids, sorted
        touched = gradient.count_nonzero(dim=1) > 0                                        # (V,)
        statistics = self._row_lazy_statistics
        tokens = statistics.step_tokens() if statistics is not None else None
        if tokens is not None:
            ids = tokens.reshape(-1).to(device=gradient.device, dtype=torch.int64)
            touched = touched.index_fill(0, ids, True)
        return touched.nonzero().squeeze(1)        # host read of |A_t|

    @torch.no_grad()
    def _row_lazy_step(
        self,
        parameters: List[Tuple[torch.Tensor, Dict[str, Any]]],
    ) -> None:
        grad_scale, found_inf = getattr(self, "grad_scale", None), getattr(self, "found_inf", None)
        if found_inf is not None and bool(found_inf.ne(0).any()):
            return                                 # GradScaler overflow step: nothing moves (as the fused kernel)
        for p, group in parameters:
            if p.grad is None:
                continue
            if group.get("amsgrad") or group.get("maximize") or not group.get("decoupled_weight_decay", True):
                raise ValueError("m_phi_sparse_rows implements decoupled-decay AdamW without amsgrad or maximize")
            if p.grad.is_sparse:
                raise ValueError("m_phi_sparse_rows expects a dense (strided) table gradient")
            state = self._row_lazy_state(p, group)
            state["step"] += 1
            V = p.shape[0]
            gradient = p.grad.reshape(V, -1)                                               # (V, P)
            if grad_scale is not None:
                gradient = gradient / grad_scale.to(device=gradient.device, dtype=gradient.dtype)
            rows = self._touched_rows(gradient)                                            # (R,)
            if rows.numel() == 0:
                continue
            lr = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            beta1, beta2 = (float(beta) for beta in group["betas"])
            eps = float(group["eps"])
            table = p.view(V, -1)                                                          # (V, P)
            exp_avg = state["exp_avg"].view(V, -1)                                         # (V, P)
            exp_avg_sq = state["exp_avg_sq"].view(V, -1)                                   # (V, P)
            row_step = state["row_step"]                                                   # (V,)
            k = row_step.index_select(0, rows) + 1                                         # (R,) k_v
            row_step.index_copy_(0, rows, k)
            theta = table.index_select(0, rows)                                            # (R, P)
            g = gradient.index_select(0, rows).to(theta.dtype)                             # (R, P)
            m = exp_avg.index_select(0, rows)                                              # (R, P)
            s = exp_avg_sq.index_select(0, rows)                                           # (R, P)
            if weight_decay != 0.0:
                theta.mul_(1.0 - lr * weight_decay)
            m.lerp_(g, 1.0 - beta1)
            s.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
            k64 = k.to(torch.float64)
            bias1 = 1.0 - torch.pow(beta1, k64)                                            # (R,) 1 - beta1^k_v
            bias2_sqrt = (1.0 - torch.pow(beta2, k64)).sqrt()                              # (R,) sqrt(1 - beta2^k_v)
            step_size = (lr / bias1).to(theta.dtype).unsqueeze(-1)                         # (R, 1)
            denom = (s.sqrt() / bias2_sqrt.to(theta.dtype).unsqueeze(-1)).add_(eps)        # (R, P)
            theta.sub_(step_size * (m / denom))
            table.index_copy_(0, rows, theta)
            exp_avg.index_copy_(0, rows, m)
            exp_avg_sq.index_copy_(0, rows, s)

    @torch.no_grad()
    def step(self, closure=None):                                       # type: ignore[override]
        if closure is not None:
            raise NotImplementedError("the row-lazy step consumes the table gradient after the base step, so a "
                                      "closure that re-backpropagates would step it twice")
        # The base step (and a table Fisher step, under RowLazyTableFisherAdamW) runs with the lazy tables' gradients
        # hidden, so it neither steps nor decays them, and reads the pre-step frames; the lazy step follows.
        parameters = self._row_lazy_parameters()
        stashed = [(p, p.grad) for p, _ in parameters]
        for p, _ in stashed:
            p.grad = None
        try:
            result = super().step()                                     # type: ignore[misc]
        finally:
            for p, grad in stashed:
                p.grad = grad
        self._row_lazy_step(parameters)
        return result


class RowLazyAdamW(_RowLazyAdamWMixin, torch.optim.AdamW):
    r"""AdamW for every group except those marked ``sparse_rows``, which take the row-lazy step of this module."""

    def __init__(self, params, statistics: Optional[Any] = None, **kwargs) -> None:
        super().__init__(params, **kwargs)
        self._init_row_lazy(statistics)


class RowLazyTableFisherAdamW(_RowLazyAdamWMixin, _TableFisherMixin, torch.optim.AdamW):
    r""":class:`vfe3.table_optim.TableFisherAdamW` with the row-lazy step on the groups marked ``sparse_rows``."""

    def __init__(self, params, statistics: Any, *, table_fisher_lr: float, table_fisher_trust_radius: float,
                 **kwargs) -> None:
        super().__init__(params, **kwargs)
        self._init_table_fisher(statistics, learning_rate=table_fisher_lr, trust_radius=table_fisher_trust_radius)
        self._init_row_lazy(statistics)
