r"""Natural-gradient M-step for the type-prior tables (``m_table_update_mode='fisher'``; audit 2026-10-10, item 22).

The prior of type :math:`v` is :math:`p_v = \mathcal N(\mu_v, \operatorname{diag}s_v)` with :math:`s_v = e^{l_v}`
(``mu_embed`` and ``sigma_log_embed``). Its Fisher metric in the coordinates :math:`(\mu_v, l_v)` is
:math:`G_v = \operatorname{diag}(1/s_v, 1/2)`. A step's loss is the mean over :math:`N` scored targets and row
:math:`v` is gathered at the :math:`n_v` input positions that hold :math:`v`, so the step preconditions that row
with :math:`(n_v/N)G_v`, the Fisher metric of the prior family :math:`p_v` counted once per occurrence. That
matrix is the exact Hessian of the self-coupling term :math:`\frac1N\sum_i \mathrm{KL}(q_i\|p_{x_i})` in
:math:`\mu_v` (and its expected Hessian in :math:`l_v`) and of nothing else; for the cross-entropy the step
trains it is a chosen preconditioner, not the metric of that objective (audit 2026-10-10 round 2, item 35).
The implemented M-step self term (``mstep_self_coupling_weight > 0``) is that type-prior KL only at ``n_layers = 1``
or ``prior_handoff_rho = 0`` (and ``prior_handoff_sigma = 0``): otherwise it reads the prior handed into the final
block, whose mean :math:`(1-\rho)\mu_p + \rho\mu_{\mathrm{belief}}` carries the previous block's output, so the
metric above is not its Hessian either (audit 2026-10-20 item 70).
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

Frame-diagonal family (``gaussian_frame_diagonal``, 2026-10-18). There the prior of type :math:`v` at a position whose
layer-0 frame is :math:`U_v` is :math:`p_v = \mathcal N(\mu_v, U_v\operatorname{diag}(s_v)U_v^\top)`, so its Fisher
metric in the local mean is :math:`U_v^{-\top}\operatorname{diag}(1/s_v)U_v^{-1}` and in :math:`l_v` it is still
:math:`\operatorname{diag}(1/2)`, since :math:`s_v` is a reference-coordinate diagonal. Without positional frames
:math:`U_v` depends on the type alone (:meth:`vfe3.model.model.VFEModel.token_type_frames`), and the natural step is
the diagonal step in the reference mean :math:`\xi_v = U_v^{-1}\mu_v`:

.. math::
    g_{\xi_v} = U_v^\top\partial_{\mu_v}\mathcal L,\qquad
    \Delta\xi_v = -\eta\frac{N}{n_v}\,s_v\odot g_{\xi_v},\qquad
    \Delta\mu_v = U_v\Delta\xi_v = -\eta\frac{N}{n_v}U_v\operatorname{diag}(s_v)U_v^\top\partial_{\mu_v}\mathcal L,

with :math:`\|\Delta\|_G^2 = \Delta\mu^\top U^{-\top}\operatorname{diag}(1/s)U^{-1}\Delta\mu + \tfrac12\|\Delta l\|^2
= \sum_k\Delta\xi_k^2/s_k + \tfrac12\|\Delta l\|^2`. The trust radius, the weight :math:`N/n_v`, the schedule factor,
the decay order and the diagnostics are those of the diagonal step, evaluated on :math:`(\Delta\xi, \Delta l)`. The
frames are built detached, before the base step moves ``phi_embed``, for the step's distinct types only (one
``unique`` over the step's tokens, a host read the diagonal step does not make). Every other family, and this
family on identity frames, takes the diagonal step unchanged.

Untied decode bank (``m_decode_update_mode='fisher'``; audit 2026-10-20 M6). The decode row of type :math:`v` is
:math:`p_v = \mathcal N(m_v, \operatorname{diag}t_v)`, :math:`t_v = e^{l_v}` (``decode_mu_embed``,
``decode_sigma_log_embed``), and it enters the cross-entropy through the logit :math:`-\mathrm{KL}(q_i\|p_v)/\tau`.
The Fisher metric of :math:`p_v` in :math:`(m_v, l_v)` is :math:`G_v = \operatorname{diag}(1/t_v, 1/2)`: from
:math:`\log p_v(x) = -\tfrac12\sum_k[(x_k - m_{vk})^2 e^{-l_{vk}} + l_{vk} + \log 2\pi]` the scores are
:math:`\partial_{m_{vk}}\log p_v = (x_k - m_{vk})/t_{vk}` and :math:`\partial_{l_{vk}}\log p_v =
\tfrac12[(x_k - m_{vk})^2/t_{vk} - 1]`, whose second moments under :math:`p_v` are :math:`1/t_{vk}` and
:math:`\tfrac14\operatorname{Var}(\chi^2_1) = \tfrac12`, and whose cross moments (odd moments) vanish. The same matrix
is the Hessian of :math:`\mathrm{KL}(q\|p_v)` in :math:`m_v` (exactly) and in :math:`l_v` at :math:`q = p_v`
(:math:`\partial^2_{l_{vk}}\mathrm{KL} = \tfrac12(s_k + \Delta_k^2)/t_{vk}`), so :math:`(c_v/(N\tau))\,G_v` is the
Hessian of the target (pull) term :math:`\frac{1}{N\tau}\sum_{i:y_i=v}\mathrm{KL}(q_i\|p_v)` of the cross-entropy at
:math:`q_i = p_v`, with :math:`c_v` the number of scored targets equal to :math:`v`. The softmax (push) term is not
described by it, so for the cross-entropy it is a chosen preconditioner, as on the encoder side; the factor
:math:`1/\tau` is absorbed into :math:`\eta`. Every decode row takes a softmax gradient whether or not it is a target,
where the metric :math:`(c_v/N)G_v` is singular, so the weight is :math:`N/\max(c_v, 1)`: a row that is no target of
the step is preconditioned as one occurrence. The step is

.. math::
    \Delta_v = -\eta\,\frac{N}{\max(c_v,1)}\bigl(t_v\odot\partial_{m_v}\mathcal L,\ 2\,\partial_{l_v}\mathcal L\bigr),
    \qquad \|\Delta_v\|_{G_v}^2 = \sum_k\Bigl(\frac{\Delta m_{vk}^2}{t_{vk}} + \frac{\Delta l_{vk}^2}{2}\Bigr) \le r^2,

at the encoder step's :math:`\eta` (``m_table_fisher_lr``) and :math:`r` (``m_table_fisher_trust_radius``): the
direction is scaled back to Fisher norm at most :math:`r` and multiplied by the schedule factor of the decode AdamW
group, and the decoupled decay :math:`p \leftarrow p(1 - \mathrm{lr}_t\lambda)` (``decode_mu_weight_decay``,
``sigma_weight_decay``) sits outside the bound. The two banks share :math:`(\eta, r)` because both are Gaussian rows in
the same metric, the radius bounds the same quantity (a row's per-step KL to second order), and the audit measured
nothing that sets them apart. The bound also caps the logit shift one step of row :math:`v` causes for every query,
:math:`|\delta\mathrm{KL}(q\|p_v)| \le \|\partial\mathrm{KL}\|_{G_v^{-1}}\,r` to first order, which is the M6 channel.
The step is computed in float64 (:func:`_robust_table_steps`), keeps no state and reads the step's targets
(:meth:`TableFisherStatistics.set_batch`), ``-100`` ignored as in the cross-entropy.
"""

import math
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from vfe3.gauge_optim import GaugeManifoldAdamW
from vfe3.model.frame_actions import frame_matvec
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

    def __init__(
        self,
        prior_bank:    torch.nn.Module,
        frame_builder: Optional[Callable[[torch.Tensor], Any]] = None,   # (R,) type ids -> CanonicalFrameContext
    ) -> None:
        self._prior_bank = prior_bank
        self._frame_builder = frame_builder
        self._tokens: Optional[torch.Tensor] = None
        self._scored: Optional[torch.Tensor] = None
        self._targets: Optional[torch.Tensor] = None
        self._frames: Optional[Tuple[torch.Tensor, Any]] = None

    def set_batch(
        self,
        tokens:  Optional[torch.Tensor],                # (B_tot, N) input ids, or None to clear
        scored:  Optional[torch.Tensor],                # () scored-target count, or None
        targets: Optional[torch.Tensor] = None,         # (B_tot, N) next-token ids (-100 = ignore), or None
    ) -> None:
        self._tokens = tokens
        self._scored = scored
        self._targets = targets
        self._frames = None

    def step_tokens(self) -> Optional[torch.Tensor]:
        r"""The step's input ids, or ``None`` outside ``train_step`` (the row-lazy frame step's occurrence mask)."""
        return self._tokens

    def targets(self) -> torch.Tensor:
        r"""The step's target ids (``-100`` = ignored), read by the decode-bank Fisher step."""
        if self._targets is None:
            raise RuntimeError("m_decode_update_mode='fisher' needs the step's targets; train_step passes them to "
                               "set_batch immediately before the optimizer step")
        return self._targets

    def decode_variances(self, dtype: torch.dtype) -> torch.Tensor:   # (V, K) t_v
        r"""The decode-bank variances :math:`t_v` as the decoder reads them (``bounded_variance_from_log``, floor eps)."""
        pb = self._prior_bank
        return bounded_variance_from_log(pb._decode_sigma_log_table().detach().to(dtype), eps=pb.eps)

    @property
    def reads_frames(self) -> bool:
        r"""Whether the metric of :math:`p_v` in :math:`\mu_v` reads the frame :math:`U_v` (``gaussian_frame_diagonal``)."""
        return self._frame_builder is not None

    def capture_frames(self) -> None:
        r"""Build the detached frames :math:`U_v` of the step's distinct types from the current (pre-step) ``phi``.

        Called by the optimizer before its base step, which may move ``phi_embed``; a no-op without a frame builder.
        """
        self._frames = None
        if self._frame_builder is None:
            return
        tokens, _ = self.batch()
        rows = torch.unique(tokens.reshape(-1))                                     # (R,) distinct types, sorted
        self._frames = (rows, self._frame_builder(rows))

    def frames(self) -> Optional[Tuple[torch.Tensor, Any]]:
        r"""``(rows, frame)`` captured by :meth:`capture_frames`, or ``None`` (diagonal metric)."""
        if self._frame_builder is not None and self._frames is None:
            raise RuntimeError("the frame-diagonal table Fisher step needs the frames captured before the base step")
        return self._frames

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


def target_counts(
    targets: torch.Tensor,                    # (...) target ids in [0, V), -100 = ignored
    V:       int,
    device:  torch.device,
) -> torch.Tensor:                            # (V,) int64 c_v
    r"""Scored occurrences :math:`c_v` of each type among the targets (``-100`` ignored), without a host read."""
    ids = targets.reshape(-1).to(device)
    valid = ids >= 0
    return torch.zeros(V, dtype=torch.int64, device=device).index_add_(
        0, torch.where(valid, ids, torch.zeros_like(ids)), valid.to(torch.int64))


def _schedule_factor(group: Dict[str, object]) -> float:
    r""":math:`\mathrm{lr}_t/\mathrm{lr}_0` of a group (``initial_lr`` is set by the LR scheduler); 0 for a frozen rate."""
    lr = float(group["lr"])
    initial = float(group.get("initial_lr", lr))
    return 0.0 if initial == 0.0 or lr == 0.0 else lr / initial


def _overflow_candidate_rows(
    tokens: torch.Tensor,                     # (...) the step's input ids in [0, V)
    V:      int,
    device: torch.device,
) -> torch.Tensor:                            # (R,) int64 row ids, R = min(tokens.numel(), V)
    r"""Rows on which the float64 overflow twin runs: a superset of the active rows :math:`\{v : n_v > 0\}`.

    The step's input ids (with repeats) when there are fewer of them than rows, else every row. Both sizes are known
    on the host, so the gather needs no device-to-host read (no ``nonzero``); a repeated id recomputes the same row.
    """
    ids = tokens.reshape(-1)
    if ids.numel() >= V:
        return torch.arange(V, device=device)
    return ids.to(device=device, dtype=torch.int64)


def _robust_table_steps(
    g_mu:         Optional[torch.Tensor],     # (V, K) unscaled mean gradient, or None
    g_l:          Optional[torch.Tensor],     # (V, K) unscaled log-variance gradient, or None
    weight:       torch.Tensor,               # (V, 1) N / n_v
    s:            torch.Tensor,               # (V, K) prior variances s_v
    eta:          float,
    trust_radius: float,
    active:       torch.Tensor,               # (V,) bool n_v > 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""The trust-scaled table step :math:`\min(1, r/\|d\|)\,d` computed without overflow (audit 2026-10-17 item 47).

    In float64 the step :math:`d = -\eta(N/n_v)(s\odot g_\mu, 2g_l)` of a finite float32 gradient cannot overflow, and
    its Fisher norm is taken max-abs normalized, :math:`\|d\| = a\,\|d/a\|` with :math:`a = \max_k|c_k|` over the
    metric-scaled components :math:`c = (d_\mu/\sqrt s, d_l/\sqrt2)`, so the sum of squares cannot overflow either.
    Returns ``(scaled_mu, scaled_l, sq_mu, sq_l, trust)``: the float64 scaled steps (zeros where a part is absent),
    their squared Fisher norms per row and the trust factor :math:`\min(1, r/\|d\|)` (0 on inactive rows).
    """
    s64 = s.to(torch.float64)
    weight64 = weight.to(torch.float64)
    reference = g_mu if g_mu is not None else g_l
    zeros = torch.zeros(reference.shape, dtype=torch.float64, device=reference.device)
    step_mu = -eta * weight64 * s64 * g_mu.to(torch.float64) if g_mu is not None else zeros         # (V, K)
    step_l = -eta * weight64 * 2.0 * g_l.to(torch.float64) if g_l is not None else zeros            # (V, K)
    c_mu = step_mu / s64.sqrt()
    c_l = step_l / math.sqrt(2.0)
    scale = torch.maximum(c_mu.abs().amax(-1), c_l.abs().amax(-1))                                  # (V,) a
    safe = torch.where(scale > 0, scale, torch.ones_like(scale)).unsqueeze(-1)
    norm = scale * ((c_mu / safe).square().sum(-1) + (c_l / safe).square().sum(-1)).sqrt()          # (V,) ||d||
    trust = torch.where(norm > 0, (trust_radius / torch.where(norm > 0, norm, torch.ones_like(norm))).clamp(max=1.0),
                        torch.ones_like(norm))
    trust = torch.where(active, trust, torch.zeros_like(trust))                                     # (V,)
    scaled_mu = trust.unsqueeze(-1) * step_mu
    scaled_l = trust.unsqueeze(-1) * step_l
    sq_mu = (scaled_mu.square() / s64).sum(-1)
    sq_l = 0.5 * scaled_l.square().sum(-1)
    return scaled_mu, scaled_l, sq_mu, sq_l, trust


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

    def _decode_fisher_groups(self) -> Tuple[Optional[Dict[str, object]], Optional[Dict[str, object]]]:
        marked = {group["table_fisher"]: group for group in self.param_groups if group.get("table_fisher")}
        return marked.get("decode_mu"), marked.get("decode_sigma")

    @torch.no_grad()
    def _decode_fisher_step(self) -> None:
        r"""The natural-gradient step of the untied decode bank (module docstring, ``m_decode_update_mode``)."""
        mu_group, sigma_group = self._decode_fisher_groups()
        m = mu_group["params"][0] if mu_group is not None else None               # (V, K) decode_mu_embed
        log_t = sigma_group["params"][0] if sigma_group is not None else None     # (V, K) decode_sigma_log_embed
        live = [p for p in (m, log_t) if p is not None and p.grad is not None]
        if not live:
            return
        grad_scale, found_inf = getattr(self, "grad_scale", None), getattr(self, "found_inf", None)
        statistics = self.table_statistics_provider
        targets = statistics.targets()
        _, scored = statistics.batch()
        reference = live[0]
        dtype = torch.promote_types(reference.dtype, torch.float32)
        V = reference.shape[0]
        skip = None if found_inf is None else found_inf.to(device=reference.device) != 0             # () bool
        count = target_counts(targets, V, reference.device).to(dtype)                               # (V,) c_v
        total = scored.to(device=reference.device, dtype=dtype).clamp_min(1.0)                       # () N
        weight = (total / count.clamp_min(1.0)).unsqueeze(-1)                                       # (V, 1)
        t = statistics.decode_variances(dtype)                                                      # (V, K) t_v

        def gradient(p: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if p is None or p.grad is None:
                return None
            g = p.grad.to(dtype)
            return g if grad_scale is None else g / grad_scale.to(device=g.device, dtype=dtype)

        every_row = torch.ones(V, dtype=torch.bool, device=reference.device)
        step_m, step_l, _, _, _ = _robust_table_steps(gradient(m), gradient(log_t), weight, t,
                                                      self._table_fisher_lr, self._table_fisher_trust_radius,
                                                      every_row)                                    # (V, K) float64
        for p, step, group in ((m, step_m, mu_group), (log_t, step_l, sigma_group)):
            if p is None or p.grad is None:
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
                update = (factor * step).to(p.dtype)                                                # (V, K)
                p.add_(update if skip is None else torch.where(skip, torch.zeros_like(update), update))
            p.grad = None

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
        # gaussian_frame_diagonal: the step is the diagonal step in xi_v = U_v^{-1} mu_v, whose gradient is U_v^T g_mu.
        frames = self.table_statistics_provider.frames() if g_mu is not None else None
        if frames is not None:
            rows, frame = frames
            rows = rows.to(device=g_mu.device)
            forward = frame.forward.to(device=g_mu.device)                         # (R, G, d, d) or (R, K, K)
            g_mu = g_mu.index_copy(0, rows, frame_matvec(forward.transpose(-1, -2), g_mu[rows],
                                                         frame.block_diagonal))      # (V, K) g_xi on the rows
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
        # Overflow-robust rows (audit 2026-10-17 item 47). A large finite gradient overflows norm_sq (|g| ~ 1e17 at
        # N/n_v = 8192, s = 5: trust 0, the row frozen) or the step itself (|g| ~ 1e35: trust * inf = NaN reaches
        # the table). Those rows, and only those (a finite norm_sq keeps every other row bit-identical), take the
        # same trust-region step from the float64 direction, where neither can overflow for float32 gradients.
        overflow = active & ~torch.isfinite(norm_sq)                                                 # (V,)
        # The float64 twin runs on the candidate rows only (audit 2026-10-20 item 44): an overflowed row is active,
        # so it is among the step's input ids, a set of host-known size with no host read; each twin output is
        # row-wise, so a gathered row equals its full-table value bit for bit and duplicates scatter equal values.
        cand = _overflow_candidate_rows(tokens, V, reference.device)                                 # (R,) row ids
        cand_overflow = overflow[cand]                                                               # (R,)
        wide_mu, wide_l, wide_sq_mu, wide_sq_l, wide_trust = _robust_table_steps(
            None if g_mu is None else g_mu[cand], None if g_l is None else g_l[cand], weight[cand], s[cand], eta,
            self._table_fisher_trust_radius, active[cand])                                           # (R, K), (R,)
        realized_sq = torch.zeros(V, dtype=dtype, device=reference.device)    # (V,) ||scheduled step||^2 / trust^2
        overflow_sq = torch.zeros(V, dtype=torch.float64, device=reference.device)   # (V,) ||scheduled step||^2
        for p, step, group, part, wide, wide_sq in ((mu, step_mu, mu_group, norm_sq_mu, wide_mu, wide_sq_mu),
                                                    (log_s, step_l, sigma_group, norm_sq_l, wide_l, wide_sq_l)):
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
                scheduled = torch.where(trust > 0, factor * trust * step, torch.zeros_like(step))   # (V, K)
                scheduled = scheduled.index_put(
                    (cand,), torch.where(cand_overflow.unsqueeze(-1), (factor * wide).to(dtype), scheduled[cand]))
                if frames is not None and p is mu:
                    # Delta mu_v = U_v Delta xi_v on the step's rows (every other row's step is zero).
                    scheduled = scheduled.index_copy(0, rows, frame_matvec(forward, scheduled[rows],
                                                                           frame.block_diagonal))   # (V, K)
                update = scheduled.to(p.dtype)
                p.add_(update if skip is None else torch.where(skip, torch.zeros_like(update), update))
                realized_sq = realized_sq + factor ** 2 * torch.where(overflow, torch.zeros_like(part), part)
                overflow_sq = overflow_sq.index_put((cand,), overflow_sq[cand] + factor ** 2 * wide_sq)
            p.grad = None
        if self._collect_table_diag:
            accepted = trust.squeeze(-1).index_put(
                (cand,), torch.where(cand_overflow, wide_trust.to(dtype), trust.squeeze(-1)[cand]))   # (V,)
            n_active = active.sum().to(dtype)
            realized = torch.where(overflow, overflow_sq.sqrt().to(dtype),
                                   accepted * realized_sq.sqrt())                                    # (V,) realized norm
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
        if self.table_statistics_provider.reads_frames and any(p.grad is not None for p in tables):
            self.table_statistics_provider.capture_frames()                 # U_v at the pre-step phi
        decode_tables = [group["params"][0] for group in self._decode_fisher_groups() if group is not None]
        stashed = [(p, p.grad) for p in tables + decode_tables]
        for p, _ in stashed:
            p.grad = None
        try:
            result = super().step()                                     # type: ignore[misc]
        finally:
            for p, grad in stashed:
                p.grad = grad
        self._table_fisher_step()
        self._decode_fisher_step()
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
