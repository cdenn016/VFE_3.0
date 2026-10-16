r"""Role-specific comparison and content operators in realized canonical frames."""

import math
from contextlib import contextmanager
from numbers import Real
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from vfe3.contracts import CanonicalFrameContext, host_read
from vfe3.model import edge_links, minimal_agent
from vfe3.model.covariance_ceiling import covariance_ceiling
from vfe3.model.edge_links import EDGE_MODES, EdgeContext
from vfe3.model.frame_actions import frame_matvec
from vfe3.relational_scoring import (
    ROUTING_COSTS, _compute_dtype, _count_source_recovery, projected_condition_recovery, source_recovery_counter,
)
from vfe3.runtime import ieee_fp32_matmul

# Byte budget of one query-row tile of the centered predictive spread (the (rows, N, H, d) differences).
_SPREAD_TILE_BYTES = 32 * 1024 ** 2

# Receiving-map charts of the content link on the stored factors (O, T) (attention_unified.tex,
# "The tiled message order by order"). A later truncation such as "second_order" adds one branch
# to ``chart_receive_map`` and one initial value to ``_chart_initial_output``.
CONTENT_LINK_CHARTS: Tuple[str, ...] = ("affine", "first_order", "exponential")


def _phi1(A: torch.Tensor) -> torch.Tensor:          # (..., n, n) -> (..., n, n)
    r"""Return :math:`\varphi_1(A)=\sum_{n\ge1}A^{n-1}/n!` without inverting :math:`A`.

    .. math::
        \exp\begin{pmatrix}A & I\\ 0 & 0\end{pmatrix}
        =\begin{pmatrix}e^A & \varphi_1(A)\\ 0 & I\end{pmatrix},

    because the block matrix has :math:`n`-th power :math:`\begin{pmatrix}A^n & A^{n-1}\\ 0 & 0\end{pmatrix}`
    for :math:`n\ge1`. Leading batch axes are evaluated matrix by matrix (one batched ``matrix_exp``).
    """
    n = A.shape[-1]
    eye = torch.eye(n, dtype=A.dtype, device=A.device).expand(*A.shape[:-2], n, n)
    augmented = torch.cat([torch.cat([A, eye], dim=-1),
                           torch.zeros(*A.shape[:-2], n, 2 * n, dtype=A.dtype, device=A.device)],
                          dim=-2)                                                       # (..., 2n, 2n)
    return torch.linalg.matrix_exp(augmented)[..., :n, n:]


def _content_link_exp(X: torch.Tensor) -> torch.Tensor:   # (..., K, K) -> (..., K, K)
    r"""The shared content link :math:`e^X`, :math:`X = OT`, one batched ``matrix_exp`` over the leading axes."""
    return torch.linalg.matrix_exp(X)


def _centered_spread_tile(
    b:      torch.Tensor,  # (..., H, I, N) routing rows of the tile
    values: torch.Tensor,  # (..., N, H, d) source values
    mean:   torch.Tensor,  # (..., I, H, d) routed means of the tile's rows
) -> torch.Tensor:         # (..., I, H, d, d)
    r"""One query-row tile of :func:`_centered_spread`."""
    diff = values.unsqueeze(-4) - mean.unsqueeze(-3)                          # (..., I, N, H, d) v_j - m_i
    weighted = b.movedim(-3, -1).unsqueeze(-1) * diff                          # (..., I, N, H, d)
    spread = torch.einsum("...ijhs,...ijht->...ihst", weighted, diff)         # (..., I, H, d, d)
    residual = weighted.sum(dim=-3)                                            # (..., I, H, d) sum_j b (v_j - m_i)
    return spread - residual.unsqueeze(-1) * residual.unsqueeze(-2)


def _centered_spread(
    b:      torch.Tensor,  # (..., H, N, N) routing weights, rows sum to one
    values: torch.Tensor,  # (..., N, H, d) source values v_j^h
    mean:   torch.Tensor,  # (..., N, H, d) routed means m_i^h = sum_j beta_ij^h v_j^h
) -> torch.Tensor:         # (..., N, H, d, d)
    r"""Routed spread accumulated about each row's own routed mean (audit 2026-10-08 P2-20).

    .. math::
        S_i=\sum_j\beta_{ij}(v_j-\hat m_i)(v_j-\hat m_i)^\top-\delta_i\delta_i^\top,
        \qquad \delta_i=\sum_j\beta_{ij}(v_j-\hat m_i),

    which equals :math:`\sum_j\beta_{ij}v_jv_j^\top-m_im_i^\top` in exact arithmetic for any center
    :math:`\hat m_i` when the row sums to one; :math:`\delta_i` is the rounding error of the computed
    mean, a second-order correction. The one-pass uncentered form cancels :math:`m_im_i^\top` against the
    second moment, so its error grows with the routed sources' offset from the common shift and returned
    nonpositive variances in fp32 under sharp routing. Query rows are tiled to a fixed byte budget and
    each tile is checkpointed under autograd, so the saved differences stay bounded.
    """
    N = b.shape[-2]
    per_row = values[..., 0, :, :].numel() * values.element_size() * max(1, values.shape[-3])
    step = max(1, min(N, _SPREAD_TILE_BYTES // max(1, per_row)))
    tiles = []
    for i in range(0, N, step):
        args = (b[..., i:i + step, :], values, mean[..., i:i + step, :, :])
        if torch.is_grad_enabled() and step < N and any(x.requires_grad for x in args):
            tiles.append(checkpoint(_centered_spread_tile, *args, use_reentrant=False, preserve_rng_state=False))
        else:
            tiles.append(_centered_spread_tile(*args))
    return tiles[0] if len(tiles) == 1 else torch.cat(tiles, dim=-4)


# "prior": f Sigma_p (frame-free, reference-invariant); "relative": f (tr C_i / K) U_i U_i^T, the floor before
# October 12, 2026, relative to the input covariance C_i = U_i^{-1} Sigma_i U_i^{-T} (not reference-invariant).
PREDICTIVE_COVARIANCE_FLOOR_MODES = ("prior", "relative")

_FP64_ESCALATION_FAILURE = ("the moment-matched covariance update is not positive definite in float64 either (the "
                            "float32 update failed first and was recomputed in float64); no repair applied")

# Tokens of moment-matched covariance updates recomputed in float64, per device, counted on the device
# (its own counter, separate from the projected source-factor recoveries; audit 2026-10-15, finding 8).
_COVARIANCE_ESCALATION_COUNTS: Dict[str, torch.Tensor] = {}


def _count_covariance_escalation(tokens: int, device: torch.device) -> None:
    key = str(device)
    total = torch.full((), tokens, dtype=torch.long, device=device)
    _COVARIANCE_ESCALATION_COUNTS[key] = (total if key not in _COVARIANCE_ESCALATION_COUNTS
                                          else _COVARIANCE_ESCALATION_COUNTS[key] + total)


def covariance_escalation_elements() -> int:
    r"""Tokens of moment-matched covariance updates recomputed in float64 since the last reset. Host-syncs."""
    return int(sum(int(count) for count in _COVARIANCE_ESCALATION_COUNTS.values()))


def reset_covariance_escalation_elements() -> None:
    r"""Zero the per-device covariance-escalation counters."""
    _COVARIANCE_ESCALATION_COUNTS.clear()


def _covariance_positive(
    sigma:    torch.Tensor,   # (..., N, K) diagonal or (..., N, K, K) covariance
    diagonal: bool,
) -> bool:
    r"""Whether every variance is finite and positive (diagonal) or every matrix factors by Cholesky. Host-syncs."""
    with torch.no_grad():
        if diagonal:
            healthy = (torch.isfinite(sigma) & (sigma > 0)).all()
        else:
            healthy = (torch.linalg.cholesky_ex(sigma.detach())[1] == 0).all()
    return bool(host_read(healthy.reshape(1))[0])


def _per_block_bmm(
    left:  torch.Tensor,  # (H, a, k)
    right: torch.Tensor,  # (H, k, c)
) -> torch.Tensor:        # (H, a, c)
    r"""``torch.bmm`` over the H head blocks of the blockwise covariance update; one GEMM per block on CUDA.

    In backward the weight gradient of a block is an :math:`a\times k` or :math:`k\times c` matrix reduced over
    every token. Batched over H = 4 heads that is a handful of output tiles on CUDA (3.6 ms per GEMM at the live
    width and B = 32), while one GEMM per block gets a split reduction (0.6 ms). Each block keeps the reduction
    order of the batched call. On CPU the batched kernel has no such cost, so it runs unchanged.
    """
    if left.device.type != "cuda":
        return torch.bmm(left, right)
    return torch.stack([torch.bmm(x[None], y[None])[0] for x, y in zip(left.unbind(0), right.unbind(0))])


def head_runs(head_dims: Sequence[int]) -> List[Tuple[int, int, int, int]]:
    r"""Maximal runs of contiguous equal-size heads as ``(first_head, n_heads, d, first_coordinate)``.

    Heads keep their order: concatenating per-run outputs along the head axis restores it, so
    head-indexed quantities (ALiBi slopes, per-head kappa and lambda) need no scatter.
    """
    runs: List[Tuple[int, int, int, int]] = []
    head = coordinate = 0
    for d in head_dims:
        if runs and runs[-1][2] == d:
            first, n, _, start = runs[-1]
            runs[-1] = (first, n + 1, d, start)
        else:
            runs.append((head, 1, d, coordinate))
        head += 1
        coordinate += d
    return runs


def chart_stacks(
    T_runs: Sequence[torch.Tensor],     # per run (n_r, d_r, K) stored sending maps, sum_r n_r d_r = K
    O_runs: Sequence[torch.Tensor],     # per run (n_r, K, d_r) stored receiving factors
    chart:  str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:   # float64 (K, K) stacks T, T^{-1}, O'
    r"""Float64 stacks :math:`(T, T^{-1}, O')` of a non-affine chart, from one inversion of :math:`T`.

    :math:`T` stacks the sending maps by rows and :math:`O'` the effective receiving maps by columns
    (:func:`chart_receive_runs` splits :math:`O'` into heads). The edge links
    (:mod:`vfe3.model.edge_links`) read all three, with :math:`F = O'T = e^{OT}` in the exponential chart.
    """
    if chart not in ("first_order", "exponential"):
        raise ValueError(f"chart_stacks evaluates the first_order and exponential charts, got {chart!r}")
    with torch.autocast(device_type=T_runs[0].device.type, enabled=False):
        T_stack, O_stack = _chart_operands(T_runs, O_runs)
        inverse, receive = _chart_receive(T_stack, O_stack, chart)
    return T_stack, inverse, receive


def _chart_operands(
    T_runs: Sequence[torch.Tensor],     # per run (n_r, d_r, K)
    O_runs: Sequence[torch.Tensor],     # per run (n_r, K, d_r)
) -> Tuple[torch.Tensor, torch.Tensor]:  # float64 (K, K) row stack T and column stack O
    r"""Validated float64 stacks :math:`T` (rows by head) and :math:`O` (columns by head)."""
    K = T_runs[0].shape[-1]
    if (sum(T.shape[0] * T.shape[1] for T in T_runs) != K
            or any(tuple(O.shape) != (T.shape[0], K, T.shape[1]) for T, O in zip(T_runs, O_runs))):
        raise ValueError(f"content link charts need tiled square stacked maps, got T "
                         f"{[tuple(T.shape) for T in T_runs]}, O {[tuple(O.shape) for O in O_runs]}")
    T_stack = torch.cat([T.to(torch.float64).reshape(-1, K) for T in T_runs], dim=0)       # (K, K) rows by head
    O_stack = torch.cat([O.to(torch.float64).permute(1, 0, 2).reshape(K, -1)
                         for O in O_runs], dim=1)                                        # (K, K) columns by head
    return T_stack, O_stack


def _chart_receive(
    T_stack: torch.Tensor,              # (..., K, K) float64 row stacks
    O_stack: torch.Tensor,              # (..., K, K) float64 column stacks
    chart:   str,
) -> Tuple[torch.Tensor, torch.Tensor]:  # (..., K, K) T^{-1} and O'
    r""":math:`(T^{-1}, O')` of a non-affine chart; leading axes (a layer stack) are one batched call each.

    :math:`O'_{\mathrm{exponential}} = e^{OT}T^{-1}`, which equals :math:`T^{-1} + O\varphi_1(TO)` because
    :math:`e^{OT} = I + O\varphi_1(TO)T`. The product form takes one :math:`K\times K` ``matrix_exp`` where the
    :math:`\varphi_1` form took the :math:`2K\times 2K` augmented one, and its backward a :math:`2K\times 2K`
    exponential instead of :math:`4K\times 4K` (3.6 to 5 times cheaper on the K = 128 tables; audit 2026-10-10,
    item 31). Against a 60-digit reference (K = 6, cond(T) up to 1e9) the two forms have the same relative error;
    they differ by up to 3.5e-12 in absolute value at :math:`\|O'\|\approx 7\times10^2` (cond(T) = 1e3).
    """
    # inv_ex: no host sync (linalg.inv checks info on the host); a singular T yields non-finite maps.
    inverse = torch.linalg.inv_ex(T_stack).inverse                                      # (..., K, K) order zero
    if chart == "first_order":
        return inverse, inverse + O_stack
    return inverse, _content_link_exp(O_stack @ T_stack) @ inverse                      # all orders


def _split_receive(
    receive: torch.Tensor,              # (K, K) float64 O' with columns by head
    T_runs:  Sequence[torch.Tensor],    # per run (n_r, d_r, K)
) -> List[torch.Tensor]:                # per run (n_r, K, d_r)
    r"""Split the column stack :math:`O'` into its per-run head maps."""
    K = receive.shape[-1]
    out, start = [], 0
    for T in T_runs:
        n, d = T.shape[0], T.shape[1]
        out.append(receive[:, start:start + n * d].reshape(K, n, d).permute(1, 0, 2))   # (n_r, K, d_r)
        start += n * d
    return out


def chart_receive_runs(
    T_runs: Sequence[torch.Tensor],     # per run (n_r, d_r, K) stored sending maps, sum_r n_r d_r = K
    O_runs: Sequence[torch.Tensor],     # per run (n_r, K, d_r) stored receiving factors
    chart:  str,
) -> List[torch.Tensor]:                # per run (n_r, K, d_r) effective receiving maps O'_h
    r"""Effective receiving maps :math:`O'` of the tiled content link in the chosen chart.

    With the row stack :math:`T=[T_1;\dots;T_H]` and the column stack :math:`O=[O_1,\dots,O_H]`
    square (:math:`\sum_hd_h=K`), the predictive message is :math:`U_i\sum_jO'B_{ij}T\xi_j` with
    :math:`B_{ij}=\bigoplus_h\beta^h_{ij}I_{d_h}` and

    .. math::
        O'_{\mathrm{affine}}=O,\qquad
        O'_{\mathrm{first\ order}}=T^{-1}+O,\qquad
        O'_{\mathrm{exponential}}=T^{-1}+O\,\varphi_1(TO)=e^{OT}T^{-1}.

    The exponential chart is the shared-generator exponential link :math:`X=OT` with routing outside
    the exponent (:math:`O'T=e^{OT}`, so the effective transfer :math:`TO'=e^{TO}` has positive
    determinant); its order-:math:`n` term is :math:`O(TO)^{n-1}/n!` and order zero is
    :math:`T^{-1}B_{ij}T=\sum_h\beta^h_{ij}P_h`, the head-routed flat transport. The first-order chart
    keeps orders zero and one. Non-affine charts invert :math:`T` and are evaluated in float64.
    Heads of different sizes enter only through the stacks, which are K x K whatever the partition.
    """
    if chart == "affine":
        return list(O_runs)
    if chart not in CONTENT_LINK_CHARTS:
        raise ValueError(f"content_link_chart must be one of {CONTENT_LINK_CHARTS}, got {chart!r}")
    _, _, receive = chart_stacks(T_runs, O_runs, chart)                                 # (K, K) float64
    return _split_receive(receive, T_runs)


@contextmanager
def shared_content_charts(operators: Sequence["RelationalOperators"]) -> Iterator[None]:
    r"""Evaluate every layer's :math:`O'` once, in one batched float64 call per chart, for the enclosed forward.

    Each distinct operator that stores O, uses a non-affine chart and has the edge links off (the edge
    links evaluate their own chart stacks) receives its slice of one ``(L, K, K)`` evaluation of
    :func:`_chart_receive`, which :meth:`RelationalOperators.content_factor_runs` returns until the
    context exits. This replaces one ``matrix_exp`` per layer per forward, whose host
    synchronizations dominated the exponential chart's cost (audit 2026-10-09). On the CPU the batched
    values equal the per-layer ones bit for bit; on CUDA the batched GEMMs may round differently.
    """
    groups: Dict[Tuple[str, int, torch.device], List["RelationalOperators"]] = {}
    seen = set()
    for ops in operators:
        if (id(ops) in seen or getattr(ops, "O", None) is None or ops.content_link_chart == "affine"
                or ops.content_link_edges != "off"):
            continue
        seen.add(id(ops))
        T0 = ops._stored_runs("T")[0]
        groups.setdefault((ops.content_link_chart, T0.shape[-1], T0.device), []).append(ops)
    installed = []
    try:
        for (chart, _, device), members in groups.items():
            runs = [(ops._stored_runs("T"), ops._stored_runs("O")) for ops in members]
            with torch.autocast(device_type=device.type, enabled=False):
                stacks = [_chart_operands(T_runs, O_runs) for T_runs, O_runs in runs]
                _, receive = _chart_receive(torch.stack([T for T, _ in stacks]),
                                            torch.stack([O for _, O in stacks]), chart)    # (L, K, K)
            for index, (ops, (T_runs, _)) in enumerate(zip(members, runs)):
                installed.append((ops, ops.__dict__.get("_shared_receive_runs")))
                ops.__dict__["_shared_receive_runs"] = _split_receive(receive[index], T_runs)
        yield
    finally:
        for ops, previous in reversed(installed):
            ops.__dict__["_shared_receive_runs"] = previous


def chart_receive_map(
    T:     torch.Tensor,                # (H, d, K) stored sending maps, H*d = K
    O:     torch.Tensor,                # (H, K, d) stored receiving factors
    chart: str,
) -> torch.Tensor:                      # (H, K, d) effective receiving maps O'_h
    r"""Equal-head form of :func:`chart_receive_runs` (one run)."""
    return chart_receive_runs([T], [O], chart)[0]


def _chart_initial_output(chart: str, scale: float) -> float:
    r"""Initial :math:`O=cI` (with :math:`T=I`) giving the content :math:`sB_{ij}` in every chart."""
    return {"affine": scale, "first_order": scale - 1.0, "exponential": math.log(scale)}[chart]


def _validate_latent_links(
    edges:   str,
    latent:  bool,
    steps:   int,
    sigma:   float,
    damping: float,
    weight:  float,
    window:  Optional[int],
) -> None:
    r"""Refuse an invalid latent receiver-factor or loop-prior setting (plan Section 3.4)."""
    finite = lambda x: (not isinstance(x, bool) and isinstance(x, Real) and math.isfinite(float(x)))
    if not isinstance(latent, bool):
        raise ValueError(f"content_link_latent must be bool, got {latent!r}")
    if latent and edges != "lowrank":
        raise ValueError("content_link_latent requires content_link_edges='lowrank' (the latent variable is the "
                         f"receiver factor A_i of the low-rank edge links), got {edges!r}")
    if type(steps) is not int or steps < 1:
        raise ValueError(f"content_link_latent_steps must be an int >= 1, got {steps!r}")
    if not finite(sigma) or sigma <= 0.0:
        raise ValueError(f"content_link_latent_sigma must be finite and positive, got {sigma!r}")
    if not finite(damping) or not 0.0 < damping <= 1.0:
        raise ValueError(f"content_link_latent_damping must lie in (0, 1], got {damping!r}")
    if not finite(weight) or weight < 0.0:
        raise ValueError(f"content_link_loop_weight must be finite and >= 0, got {weight!r}")
    if weight > 0.0 and not latent:
        raise ValueError("content_link_loop_weight > 0 requires content_link_latent=True (the loop prior acts "
                         "on the latent receiver factors)")
    if window is not None and (type(window) is not int or window < 1):
        raise ValueError(f"content_link_loop_window must be None or an int >= 1, got {window!r}")


class RelationalOperators(nn.Module):
    r"""Full-width comparison maps S_Q, S_K and content B=OT.

    Every rank is the head dimension: S_Q/S_K are ``(H, d, input_dim)`` and T/O are
    ``(H, d, input_dim)`` / ``(H, output_dim, d)``, the untied standard-attention shapes.
    Every head reads and writes the full state while retaining its reduced channel rank.
    Heads may differ in size (an ``so_n`` irrep tower): the maps are then stored per run of
    contiguous equal-size heads (:func:`head_runs`), run r as ``(n_r, d_r, K)`` / ``(n_r, K, d_r)``
    in a ``ParameterList`` under the same four names (state-dict keys ``T.0``, ``T.1``, ...). One run,
    which includes every equal-head partition, stores today's single tensors.
    Independent maps are free matrices initialized with orthonormal rows.
    Their projected covariances must remain strictly positive definite.
    ``output_init_scale`` sets the initial content :math:`sB_{ij}` (placed on the head blocks under full
    output scope); ``s = 1`` is the identity start. ``content_link_chart`` selects the effective
    receiving maps built from the stored (O, T) (:func:`chart_receive_map`); the stored O starts at
    :math:`sI`, :math:`(s-1)I` or :math:`(\log s)I` per chart so every chart starts at :math:`sB_{ij}`.
    Every consumer reads the content maps through :meth:`content_factors`.
    ``receiver_measurement`` (full_link_update) selects the
    receiving measurement :math:`P_h` of :math:`\xi_i`: the adjoint :math:`O_h'^\top` or the head-h rows of
    :math:`O'^{-1}` (:func:`vfe3.gradients.full_link.build_full_link_context`); it is an attribute, not a parameter.
    ``routing`` selects the projected routing cost: Gaussian KL, or the noisy-query cost
    N(a_i; b_j, V_Q + V_K) of the expected likelihood kernel (predictive_content only).
    ``routing="noisy_key"`` (predictive_content, kernel readout) scores
    :math:`\mathrm{KL}(N(a_i, V_{Q,i})\,\|\,N(b_j, V_{K,j} + n_hI))` at temperature 1
    (:func:`vfe3.relational_scoring.projected_noisy_key_cost`, :meth:`routing_temperature`) and adds
    ``routing_key_log_noise`` ``(H,)``, :math:`\log n_h`, initialized to ``log(routing_key_noise)``; with any other
    routing no such parameter exists.
    ``predictive_covariance_ceiling`` (None = off) bounds the moment-matched covariance above by
    :math:`c\Sigma_{p,i}` in the Loewner order (:mod:`vfe3.model.covariance_ceiling`), the counterpart of the
    floor :math:`f\Sigma_{p,i}`; it is a clamp and is counted where it binds.
    ``predictive_covariance_checkpoint`` recomputes the moment-matched covariance update in backward
    instead of saving its (..., N, K, K) intermediates: the same values and gradients bit for bit,
    for about half the activation memory of a large-K step.
    ``predictive_covariance_fp64_escalation`` recomputes a float32 kernel-readout moment-matched update in float64
    when its output is not positive definite (:meth:`predictive_covariance_update`).
    ``block_covariance`` (family ``gaussian_block``; equal heads, the kernel predictive readout, edges off) makes
    the moment-matched update return the head-block M-projection of :math:`\Sigma_i^+` in the dense layout,
    one SPD :math:`d\times d` block per head (:meth:`_predictive_covariance_update_blockwise`).
    ``content_link_edges`` ("lowrank" or "pure_gauge", exponential chart, equal heads, predictive
    content only) gives every edge the link :math:`G_{ij} = g_ie^{L}E_{ij}` of :mod:`vfe3.model.edge_links`,
    with rank-``content_link_edge_rank`` factor maps stored as ``edge_receiver_map`` (lowrank) or
    ``edge_site_basis`` (pure gauge) and ``edge_sender_map`` (zero at initialization, so the edge links
    start at the exponential chart). With the edges off no edge parameter exists.
    ``content_link_latent`` (lowrank only) replaces the amortized receiver factors by their per-context MAP
    estimate (:func:`vfe3.model.edge_links.infer_latent_receivers`: ``content_link_latent_steps`` damped
    fixed-point steps with prior scale ``content_link_latent_sigma`` and damping ``content_link_latent_damping``),
    optionally with the loop prior of weight ``content_link_loop_weight`` over ``content_link_loop_window``
    against the previous layer's links. It adds no parameter.
    ``predictive_readout="minimal_agent"`` (predictive content, equal heads, edges off, no residual flag) replaces the
    content readout by exact Gaussian conditioning on a noisy comparison record (:mod:`vfe3.model.minimal_agent`,
    :meth:`minimal_agent_update`): ``raw_score_q`` is the query map :math:`S_h`, ``T`` is both the sending map and the
    key (tied), and ``channel_log_noise`` ``(H,)`` holds :math:`\log n_h` of the channel noise :math:`n_hI_d`
    (initialized to ``log(minimal_agent_channel_noise)``). ``raw_score_k`` and ``O`` are not created (registered as
    None: the key is tied and the receiving map is derived), so the chart, the initial output scale and
    ``routing`` are unused, and every O-reading method raises. ``minimal_agent_inference`` selects the exact
    ADF posterior or the mean-field step, and ``minimal_agent_cavi_steps`` the mean field's number of
    coordinate-ascent sweeps (:func:`vfe3.model.minimal_agent.minimal_agent_readout`).
    ``predictive_covariance='spread_corrected'`` (minimal agent, mean field only) adds the linearized label spread
    :math:`\sum_hP_iS_h^\top R'^h_iS_hP_i` to the mean-field covariance (audit 2026-10-12 T2); the kernel readout and
    the exact minimal agent refuse it. ``elbo_diagnostic`` (minimal agent; a plain attribute, not state) adds the
    per-layer evidence lower bound, the ADF evidence and their gap to a diagnostic capture
    (:data:`vfe3.model.minimal_agent.MINIMAL_AGENT_ELBO_KEYS`); it never enters the loss.
    ``tie_comparison_maps`` (kernel predictive readout) stores no ``raw_score_k``: the key map is the query map,
    :math:`S_K := S_Q`. ``learn_attention_temperature`` adds ``routing_log_temperature`` ``(H,)``, the offset of
    the log routing temperature (:meth:`routing_temperature`); with it off no such parameter exists.
    ``attention_residual``, ``predictive_covariance``, ``receiver_update``,
    ``content_link_chart``, ``minimal_agent_inference``, ``kl_max_reference`` and
    ``rope_comparison_covariance`` are keyword-only without defaults, so no constructor default can drift
    from :class:`vfe3.config.VFE3Config` (audit 2026-10-09).
    """

    def __init__(
        self,
        irrep_dims:            List[int],
        seed:                  int,
        readout_mode:          str   = "predictive_content",
        *,
        attention_residual:    bool,
        output_init_scale:     float = 1.0,
        routing:               str   = "gaussian_kl",
        predictive_covariance: str,
        predictive_covariance_floor: float = 1e-4,
        predictive_covariance_floor_mode: str = "prior",
        predictive_covariance_ceiling: Optional[float] = None,
        receiver_noise:        float = 1.0,
        receiver_update:       str,
        receiver_damping:      float = 1.0,
        receiver_measurement:  str   = "inverse",
        content_link_chart:    str,
        predictive_covariance_checkpoint: bool = False,
        predictive_covariance_fp64_escalation: bool = False,
        content_link_edges:    str   = "off",
        content_link_edge_rank: int  = 4,
        content_link_latent:   bool  = False,
        content_link_latent_steps: int = 1,
        content_link_latent_sigma: float = 0.1,
        content_link_latent_damping: float = 1.0,
        content_link_loop_weight: float = 0.0,
        content_link_loop_window: Optional[int] = 1,
        predictive_readout:    str   = "kernel",
        minimal_agent_inference: str,
        minimal_agent_channel_noise: float = 1.0,
        minimal_agent_cavi_steps: int = 1,
        comparison_rope_base: Optional[float] = None,
        comparison_rope_fraction: float = 1.0,
        kl_max_reference:      str,
        rope_comparison_covariance: str,
        learn_attention_temperature: bool = False,
        tie_comparison_maps:   bool  = False,
        block_covariance:      bool  = False,
        pair_kl_gemm_tol:      Optional[float] = None,
        routing_key_noise:     float = 1.0,
        elbo_diagnostic:       bool  = False,
    ) -> None:
        super().__init__()
        self.kl_max_reference = kl_max_reference      # "absolute" | "row_min" kernel-router clamp reference
        self.pair_kl_gemm_tol = pair_kl_gemm_tol      # None, or the opt-in pair-KL GEMM fallback tolerance (nats)
        # comparison rope_mode: the RoPE frequency base of the rotation applied to each head's projected
        # routing moments after the learned maps (relational_score); None leaves routing un-rotated here.
        self.comparison_rope_base = None if comparison_rope_base is None else float(comparison_rope_base)
        self.comparison_rope_fraction = float(comparison_rope_fraction)              # p-RoPE fraction
        if rope_comparison_covariance not in ("full", "rotation_invariant"):
            raise ValueError("rope_comparison_covariance must be 'full' or 'rotation_invariant', "
                             f"got {rope_comparison_covariance!r}")
        # "full" | "rotation_invariant": the comparison covariance rotated as R_i G R_i^T, or projected onto the
        # commutant of the RoPE torus (read only when comparison_rope_base is set; relational_score).
        self.rope_comparison_covariance = rope_comparison_covariance
        if not irrep_dims or any(type(d) is not int or d < 1 for d in irrep_dims):
            raise ValueError("relational operators require positive integer head sizes")
        if type(seed) is not int:
            raise ValueError("relational seed must be an integer")
        if (isinstance(output_init_scale, bool) or not isinstance(output_init_scale, (int, float))
                or not math.isfinite(output_init_scale) or output_init_scale <= 0.0):
            raise ValueError(
                f"output_init_scale must be finite and positive, got {output_init_scale!r}")
        if attention_residual and readout_mode != "predictive_content":
            raise ValueError("attention residual requires predictive_content")
        self.attention_residual = attention_residual
        if readout_mode not in ("predictive_content", "full_link_update"):
            raise ValueError("relational readout_mode must be predictive_content or full_link_update")
        if (isinstance(receiver_noise, bool) or not isinstance(receiver_noise, Real)
                or not math.isfinite(receiver_noise) or receiver_noise <= 0.0):
            raise ValueError("receiver_noise must be a finite positive reference variance")
        self.receiver_noise = float(receiver_noise)
        if receiver_update not in ("euler", "closed_form", "closed_form_profiled"):
            raise ValueError("receiver_update must be 'euler', 'closed_form' or 'closed_form_profiled'")
        if (isinstance(receiver_damping, bool) or not isinstance(receiver_damping, Real)
                or not math.isfinite(receiver_damping) or not 0.0 < receiver_damping <= 1.0):
            raise ValueError("receiver_damping must lie in (0, 1]")
        if receiver_measurement not in ("adjoint", "inverse"):
            raise ValueError("receiver_measurement must be 'adjoint' or 'inverse'")
        # Full-link receiver measurement and update (projected_gradient.tex); inert under predictive_content.
        # P_h = O'_h^T ("adjoint") or the head-h rows of O'^{-1} ("inverse").
        self.receiver_measurement = receiver_measurement
        self.receiver_update = receiver_update
        self.receiver_damping = float(receiver_damping)
        self.readout_mode = readout_mode
        if routing not in ROUTING_COSTS:
            raise ValueError(f"relational routing must be one of {tuple(ROUTING_COSTS)}, got {routing!r}")
        if routing != "gaussian_kl" and readout_mode != "predictive_content":
            raise ValueError(f"{routing} routing requires predictive_content")
        self.routing = routing
        if predictive_covariance not in minimal_agent.PREDICTIVE_COVARIANCES:
            raise ValueError(f"predictive_covariance must be one of {minimal_agent.PREDICTIVE_COVARIANCES}")
        if predictive_covariance in minimal_agent.READOUT_COVARIANCES and readout_mode != "predictive_content":
            raise ValueError(f"{predictive_covariance} predictive covariance requires predictive_content")
        if predictive_covariance == "spread_corrected" and (predictive_readout != "minimal_agent"
                                                            or minimal_agent_inference != "mean_field"):
            raise ValueError("predictive_covariance='spread_corrected' corrects the minimal agent's mean-field "
                             "covariance for the label spread; it requires predictive_readout='minimal_agent' with "
                             "minimal_agent_inference='mean_field' (the kernel readout and the exact posterior "
                             "already carry the between-source spread)")
        if (isinstance(predictive_covariance_floor, bool)
                or not isinstance(predictive_covariance_floor, (int, float))
                or not math.isfinite(predictive_covariance_floor) or predictive_covariance_floor <= 0.0):
            raise ValueError(
                f"predictive_covariance_floor must be finite and positive, got {predictive_covariance_floor!r}")
        self.predictive_covariance = predictive_covariance
        self.predictive_covariance_floor = float(predictive_covariance_floor)
        if predictive_covariance_floor_mode not in PREDICTIVE_COVARIANCE_FLOOR_MODES:
            raise ValueError(f"predictive_covariance_floor_mode must be one of {PREDICTIVE_COVARIANCE_FLOOR_MODES}, "
                             f"got {predictive_covariance_floor_mode!r}")
        self.predictive_covariance_floor_mode = predictive_covariance_floor_mode
        if predictive_covariance_ceiling is not None and (
                isinstance(predictive_covariance_ceiling, bool)
                or not isinstance(predictive_covariance_ceiling, (int, float))
                or not math.isfinite(predictive_covariance_ceiling)
                or predictive_covariance_ceiling <= self.predictive_covariance_floor):
            raise ValueError("predictive_covariance_ceiling must be None or finite and above "
                             f"predictive_covariance_floor ({self.predictive_covariance_floor}), "
                             f"got {predictive_covariance_ceiling!r}")
        self.predictive_covariance_ceiling = (None if predictive_covariance_ceiling is None
                                              else float(predictive_covariance_ceiling))
        if not isinstance(predictive_covariance_checkpoint, bool):
            raise ValueError("predictive_covariance_checkpoint must be bool, got "
                             f"{predictive_covariance_checkpoint!r}")
        self.predictive_covariance_checkpoint = predictive_covariance_checkpoint
        if not isinstance(predictive_covariance_fp64_escalation, bool):
            raise ValueError("predictive_covariance_fp64_escalation must be bool, got "
                             f"{predictive_covariance_fp64_escalation!r}")
        self.predictive_covariance_fp64_escalation = predictive_covariance_fp64_escalation
        if content_link_chart not in CONTENT_LINK_CHARTS:
            raise ValueError(
                f"content_link_chart must be one of {CONTENT_LINK_CHARTS}, got {content_link_chart!r}")
        self.content_link_chart = content_link_chart
        K = sum(irrep_dims)
        self.input_dim = K
        self.irrep_dims = list(irrep_dims)
        self.runs = head_runs(self.irrep_dims)
        equal = len(self.runs) == 1
        if content_link_edges not in EDGE_MODES:
            raise ValueError(f"content_link_edges must be one of {EDGE_MODES}, got {content_link_edges!r}")
        if content_link_edges != "off":
            if type(content_link_edge_rank) is not int or not 1 <= content_link_edge_rank <= K:
                raise ValueError(f"content_link_edge_rank must be an int in [1, {K}], got {content_link_edge_rank!r}")
            if content_link_chart != "exponential":
                raise ValueError("content_link_edges requires content_link_chart='exponential' (the edge links "
                                 f"multiply the shared exponential link), got {content_link_chart!r}")
            if readout_mode != "predictive_content":
                raise ValueError("content_link_edges requires the predictive_content readout "
                                 "(attention_mode='projected_predictive'); the projected_gradient receiver "
                                 "reads the shared content maps only")
            if not equal:
                raise ValueError("content_link_edges requires equal heads (one head run); this partition has "
                                 f"head sizes {self.irrep_dims} (a mixed so_n irrep tower)")
        self.content_link_edges = content_link_edges
        self.content_link_edge_rank = content_link_edge_rank
        _validate_latent_links(content_link_edges, content_link_latent, content_link_latent_steps,
                               content_link_latent_sigma, content_link_latent_damping,
                               content_link_loop_weight, content_link_loop_window)
        self.content_link_latent = content_link_latent
        self.content_link_latent_steps = content_link_latent_steps
        self.content_link_latent_sigma = float(content_link_latent_sigma)
        self.content_link_latent_damping = float(content_link_latent_damping)
        self.content_link_loop_weight = float(content_link_loop_weight)
        self.content_link_loop_window = content_link_loop_window
        if predictive_readout not in minimal_agent.PREDICTIVE_READOUTS:
            raise ValueError(f"predictive_readout must be one of {minimal_agent.PREDICTIVE_READOUTS}, "
                             f"got {predictive_readout!r}")
        if minimal_agent_inference not in minimal_agent.MINIMAL_AGENT_INFERENCE:
            raise ValueError(f"minimal_agent_inference must be one of {minimal_agent.MINIMAL_AGENT_INFERENCE}, "
                             f"got {minimal_agent_inference!r}")
        if (isinstance(minimal_agent_channel_noise, bool) or not isinstance(minimal_agent_channel_noise, Real)
                or not math.isfinite(minimal_agent_channel_noise) or minimal_agent_channel_noise <= 0.0):
            raise ValueError("minimal_agent_channel_noise must be finite and positive, "
                             f"got {minimal_agent_channel_noise!r}")
        if (isinstance(minimal_agent_cavi_steps, bool) or not isinstance(minimal_agent_cavi_steps, int)
                or minimal_agent_cavi_steps < 1):
            raise ValueError(f"minimal_agent_cavi_steps must be an int >= 1, got {minimal_agent_cavi_steps!r}")
        minimal = predictive_readout == "minimal_agent"
        if minimal:
            if readout_mode != "predictive_content":
                raise ValueError("predictive_readout='minimal_agent' requires the predictive_content readout "
                                 "(attention_mode='projected_predictive')")
            if not equal:
                raise ValueError("predictive_readout='minimal_agent' requires equal heads (one head run); this "
                                 f"partition has head sizes {self.irrep_dims} (a mixed so_n irrep tower)")
            if content_link_edges != "off":
                raise ValueError("predictive_readout='minimal_agent' requires content_link_edges='off' (the edge "
                                 "links multiply the content factors O, which the minimal agent derives)")
            if attention_residual:
                raise ValueError("predictive_readout='minimal_agent' derives a precision-weighted residual; "
                                 "attention_residual must be False")
        self.predictive_readout = predictive_readout
        self.minimal_agent_inference = minimal_agent_inference
        self.minimal_agent_channel_noise = float(minimal_agent_channel_noise)
        self.minimal_agent_cavi_steps = minimal_agent_cavi_steps   # mean-field CAVI sweeps (plain attribute, not state)
        if not isinstance(elbo_diagnostic, bool):
            raise ValueError(f"elbo_diagnostic must be bool, got {elbo_diagnostic!r}")
        self.elbo_diagnostic = elbo_diagnostic                       # T1 diagnostic capture (plain attribute, not state)
        if not isinstance(block_covariance, bool):
            raise ValueError(f"block_covariance must be bool, got {block_covariance!r}")
        if block_covariance:
            block_problems = []
            if not equal:
                block_problems.append(f"equal heads (head sizes {self.irrep_dims})")
            if readout_mode != "predictive_content" or minimal:
                block_problems.append("the kernel predictive readout (predictive_content, predictive_readout='kernel')")
            if content_link_edges != "off":
                block_problems.append("content_link_edges='off'")
            if block_problems:
                raise ValueError("block_covariance (the gaussian_block head-block covariance update) requires "
                                 + ", ".join(block_problems))
        # gaussian_block: the moment-matched update returns the head-block M-projection (one SPD block per head).
        self.block_covariance = block_covariance
        if not isinstance(learn_attention_temperature, bool) or not isinstance(tie_comparison_maps, bool):
            raise ValueError("learn_attention_temperature and tie_comparison_maps must be bool")
        if minimal and (learn_attention_temperature or tie_comparison_maps):
            raise ValueError("predictive_readout='minimal_agent' routes at temperature 1 with the key tied to T; "
                             "learn_attention_temperature and tie_comparison_maps need the kernel readout")
        noisy_key = routing == "noisy_key" and not minimal   # the minimal agent does not read ``routing``
        if noisy_key and learn_attention_temperature:
            raise ValueError("routing='noisy_key' routes at temperature 1 (the learned key noise sets the sharpness); "
                             "learn_attention_temperature must be False")
        if (isinstance(routing_key_noise, bool) or not isinstance(routing_key_noise, Real)
                or not math.isfinite(routing_key_noise) or routing_key_noise <= 0.0):
            raise ValueError(f"routing_key_noise must be finite and positive, got {routing_key_noise!r}")
        if tie_comparison_maps and readout_mode != "predictive_content":
            raise ValueError("tie_comparison_maps requires the predictive_content readout")
        self.tie_comparison_maps = tie_comparison_maps
        # delta_{l,h} of tau_{l,h} = tau_h exp(3 tanh(delta_{l,h} / 3)); absent (no state-dict key) when off.
        if learn_attention_temperature:
            self.routing_log_temperature = nn.Parameter(torch.zeros(len(self.irrep_dims)))   # (H,)
        else:
            self.register_parameter("routing_log_temperature", None)
        # log n_h of the noisy-key law N(b_j, V_K,j + n_h I); absent (no state-dict key) under any other routing.
        if noisy_key:
            self.routing_key_log_noise = nn.Parameter(
                torch.full((len(self.irrep_dims),), math.log(float(routing_key_noise))))     # (H,)
        else:
            self.register_parameter("routing_key_log_noise", None)
        self.score_rank_per_head = irrep_dims[0] if equal else list(irrep_dims)
        self.content_rank_per_head = irrep_dims[0] if equal else list(irrep_dims)
        if minimal:
            _, n, d, start = self.runs[0]
            T_full = torch.zeros(n, d, K)
            for h in range(n):
                T_full[h, :, start+h*d:start+(h+1)*d] = torch.eye(d)
            # S_Q starts EQUAL to T's block projector (S_Q = T at step 0): the record S_Q X_i - T X_j then compares
            # like coordinates, so the initial routing is the overlap of the same observable of receiver and source
            # and the derived gain C_i S^T (...)^{-1} writes the heard utterance back into the coordinates it was
            # read from. Independent random query rows would compare unrelated coordinates of receiver and source.
            self.raw_score_q = nn.Parameter(T_full.clone())
            self.register_parameter("raw_score_k", None)                   # tied key: S_K = T
            self.T = nn.Parameter(T_full)                                  # block-projector initialization
            self.register_parameter("O", None)                             # derived receiving map
            self.channel_log_noise = nn.Parameter(torch.full((n,), math.log(float(minimal_agent_channel_noise))))
            return
        generator = torch.Generator().manual_seed(seed)
        stored = {}
        for role in ("q", "k"):
            stored[f"raw_score_{role}"] = [
                self._orthonormal_columns(torch.randn(n, K, d, generator=generator)).transpose(-1, -2).contiguous()
                for _, n, d, _ in self.runs]
        if tie_comparison_maps:
            # S_K := S_Q. The key draw above still runs, so T, O and the edge maps match the untied model.
            del stored["raw_score_k"]
            self.register_parameter("raw_score_k", None)
        # Preserve the released block-projector initialization and parameter names.
        initial_output = _chart_initial_output(content_link_chart, float(output_init_scale))
        stored["T"], stored["O"] = [], []
        for _, n, d, start in self.runs:
            eye = torch.eye(d).expand(n, d, d).clone()
            T_full = eye.new_zeros(n, d, K)
            O_full = eye.new_zeros(n, K, d)
            for h in range(n):
                T_full[h, :, start+h*d:start+(h+1)*d] = eye[h]
                O_full[h, start+h*d:start+(h+1)*d] = eye[h] * initial_output
            stored["T"].append(T_full)
            stored["O"].append(O_full)
        for name, tensors in stored.items():
            setattr(self, name, nn.Parameter(tensors[0]) if equal
                    else nn.ParameterList(nn.Parameter(t) for t in tensors))
        if content_link_edges != "off":
            # Drawn after every existing initialization, so the shared maps are unchanged by the edges.
            r = content_link_edge_rank
            if content_link_edges == "lowrank":
                # W_A ~ N(0, K^{-2}): A_i = reshape(W_A xi^_i) has columns of expected unit norm.
                self.edge_receiver_map = nn.Parameter(torch.randn(K * r, K, generator=generator) / K)
            else:
                self.edge_site_basis = nn.Parameter(
                    self._orthonormal_columns(torch.randn(K, r, generator=generator)).contiguous())
            self.edge_sender_map = nn.Parameter(torch.zeros(K * r, K))    # B = 0: the exponential chart

    def _stored_runs(self, name: str) -> List[torch.Tensor]:
        r"""The per-run tensors stored under ``name`` (one entry for an equal-head partition)."""
        value = getattr(self, name)
        return list(value) if isinstance(value, nn.ParameterList) else [value]

    @property
    def mixed_heads(self) -> bool:
        r"""Whether the heads differ in size (more than one run)."""
        return len(self.runs) > 1

    def _head_slice(self, x: torch.Tensor, run: int, dim: int) -> torch.Tensor:
        r"""Run ``run``'s heads of a head-indexed tensor (the tensor itself for one run)."""
        if not self.mixed_heads:
            return x
        first, n = self.runs[run][0], self.runs[run][1]
        return x.narrow(dim, first, n)

    @property
    def inference_semantics(self) -> str:
        """Name the actual mean update used by this module."""
        if self.predictive_readout == "minimal_agent":
            return f"relational_minimal_agent_{self.minimal_agent_inference}"
        return ("relational_full_link_frozen_filtering" if self.readout_mode == "full_link_update"
                else "relational_predictive_content")

    def _require_kernel_readout(self, what: str) -> None:
        r"""Refuse an O-reading method under ``predictive_readout='minimal_agent'``, which stores no content factor O."""
        if self.predictive_readout != "kernel":
            raise ValueError(f"{what} reads the content factor O, which predictive_readout='minimal_agent' does not "
                             "store (its receiving map is derived from the receiver covariance); use "
                             "minimal_agent_update")

    @property
    def input_scope(self) -> str:
        """Report the fixed full-state input architecture."""
        return "full"

    @property
    def output_scope(self) -> str:
        """Report the fixed full-state output architecture."""
        return "full"

    def predictive_content(
        self,
        mu:    torch.Tensor,  # (..., N, K), source means with live gradients
        beta:  torch.Tensor,  # (..., H, N, N)
        frame: CanonicalFrameContext,
        receive: 'Optional[torch.Tensor | Sequence[torch.Tensor]]' = None,   # O'_h, (H, K, d) or per run
        *,
        edges: Optional[EdgeContext] = None,   # content_link_edges: the layer's edge context (built when None)
    ) -> torch.Tensor:
        r"""Compute m_i = A_i sum_h O'_h sum_j beta_ij^h T_h B_j^{-1} mu_j (O' of content_link_chart).

        Under ``content_link_edges`` the message is :math:`m_i = U_i\sum_jG_{ij}\Pi(\beta_{ij})\xi_j`
        (:func:`vfe3.model.edge_links.edge_reference_message`), read from ``edges`` when given. Under
        ``content_link_latent`` the context must be given (its inference reads the source covariances).
        """
        self._require_kernel_readout("predictive_content")
        canonical = self.canonical_mean(mu, frame)                          # (..., N, K)
        T_runs = self._stored_runs("T")
        dtype = _compute_dtype(mu, *self._content_parameters(), frame.forward)
        if edges is None and self.content_link_edges != "off":
            edges = self.edge_context(mu, beta, frame, receive)
        receive = (self._receive_runs(receive) if edges is None
                   else [edge_links.receive_heads(edges, len(self.irrep_dims))])
        with torch.autocast(device_type=mu.device.type, enabled=False):
            coordinates = canonical.unsqueeze(-2).to(dtype)                 # (..., N, 1, K)
            body = None
            for run, (T, O) in enumerate(zip(T_runs, receive)):
                values = (T.to(dtype) @ coordinates.unsqueeze(-1)).squeeze(-1)     # (..., N, n_r, d_r)
                pooled = torch.einsum("...hij,...jhs->...ihs",
                                      self._head_slice(beta, run, -3).to(dtype), values)
                if edges is not None:                                       # one run (equal heads)
                    body = edge_links.edge_reference_message(
                        edge_links.cast_context(edges, dtype), beta.to(dtype), pooled.flatten(-2))
                    continue
                # Sum heads inside the contraction, avoiding a (..., N, H, K) result.
                term = torch.einsum("hks,...nhs->...nk", O.to(dtype), pooled)     # (..., N, K)
                body = term if body is None else body + term
            if frame.block_diagonal:
                body = body.reshape(*mu.shape[:-1], len(self.irrep_dims), self.irrep_dims[0])
            with ieee_fp32_matmul():                                       # U_i body_i, kappa(U)-sensitive
                result = (frame.forward.to(dtype) @ body.unsqueeze(-1)).squeeze(-1)
        return result.reshape_as(mu).to(mu.dtype)

    def _content_parameters(self) -> List[torch.Tensor]:
        r"""Every stored content tensor (T and O of every run), for dtype resolution."""
        self._require_kernel_readout("the content-parameter list")
        return self._stored_runs("T") + self._stored_runs("O")

    def edge_parameters(self) -> List[torch.Tensor]:
        r"""The edge-link parameters (``edge_receiver_map`` or ``edge_site_basis``, then ``edge_sender_map``); empty when off."""
        return [getattr(self, name) for name in edge_links.EDGE_PARAMETER_NAMES if hasattr(self, name)]

    def edge_context(
        self,
        mu:      torch.Tensor,        # (..., N, K) source means (local coordinates)
        beta:    torch.Tensor,        # (..., H, N, N) routing weights
        frame:   CanonicalFrameContext,   # the value frame the content reads
        receive: 'Optional[torch.Tensor | Sequence[torch.Tensor]]' = None,   # (H, K, d) O'_h; None evaluates the chart
        *,
        sigma:      Optional[torch.Tensor] = None,   # (..., N, K) or (..., N, K, K): content_link_latent reads it
        log_prior:  Optional[torch.Tensor] = None,   # (N, N) or (H, N, N) routing log-prior (loop weights)
        link_state: Optional[dict] = None,           # loop prior: {"previous": LinkState}; "current" is written
        latent_out: Optional[dict] = None,           # out-param: edge_links.LATENT_DIAGNOSTIC_KEYS
    ) -> EdgeContext:
        r"""The layer's :class:`vfe3.model.edge_links.EdgeContext`, computed once and shared by mean, covariance and diagnostics.

        :math:`(T, R = T^{-1}, O', F = O'T)` come from one float64 chart evaluation
        (:func:`chart_stacks`; a supplied ``receive`` replaces :math:`O'` and costs a separate inversion)
        and are cast with the factors and :math:`\Phi_{ij}` to the readout compute dtype (float64 under
        an escalated frame), with autocast off and IEEE fp32 GEMMs.

        Under ``content_link_latent`` the receiver factors are the MAP estimates
        :func:`vfe3.model.edge_links.infer_latent_receivers` returns, from the reference covariances
        :math:`C_j = U_j^{-1}\Sigma_jU_j^{-\top}` of ``sigma`` in ``frame``. With a positive loop weight
        ``link_state`` is required (an empty dict at layer 0, where there are no loops); its ``"previous"``
        entry (the previous layer's :class:`vfe3.model.edge_links.LinkState`) feeds the loop prior with weights
        from ``log_prior``, and this layer's state is written to ``"current"``.
        """
        self._require_kernel_readout("edge_context")
        if self.content_link_edges == "off":
            raise ValueError("edge_context needs content_link_edges 'lowrank' or 'pure_gauge'")
        if self.content_link_latent and sigma is None:
            raise ValueError("content_link_latent infers the receiver factors from the source covariances; pass "
                             "sigma, or the edge context relational_predictive_readout builds")
        loops = self.content_link_latent and self.content_link_loop_weight > 0.0
        if loops and link_state is None:
            raise ValueError("content_link_loop_weight > 0 reads the previous layer's links; pass link_state "
                             "(an empty dict at layer 0), as vfe_stack does")
        K = self.input_dim
        dtype = _compute_dtype(mu, *self._content_parameters(), *self.edge_parameters(), frame.forward)
        receiver = (self.edge_receiver_map if self.content_link_edges == "lowrank" else self.edge_site_basis)
        with torch.autocast(device_type=mu.device.type, enabled=False), ieee_fp32_matmul():
            if receive is None:
                T64, R64, O64 = chart_stacks([self.T], [self.O], self.content_link_chart)
            else:
                O_heads = receive if torch.is_tensor(receive) else list(receive)[0]       # (H, K, d)
                T64 = self.T.to(torch.float64).reshape(K, K)
                R64 = torch.linalg.inv_ex(T64).inverse
                O64 = O_heads.to(torch.float64).permute(1, 0, 2).reshape(K, K)
            F64 = O64 @ T64                                                                # (K, K) e^{L}
            xi = self.canonical_mean(mu, frame).to(dtype)                                  # (..., N, K)
            b = beta.to(dtype)
            ctx = edge_links.build_edge_context(
                xi, b, T64.to(dtype), R64.to(dtype), O64.to(dtype), F64.to(dtype),
                self.content_link_edges, self.edge_sender_map.to(dtype), receiver.to(dtype))
            if not self.content_link_latent:
                return ctx
            previous = link_state.get("previous") if loops else None
            loop = (None if previous is None else edge_links.LoopInputs(
                previous, edge_links.prior_routing_weights(log_prior, mu.shape[-2], dtype=torch.float64,
                                                           device=mu.device),
                self.content_link_loop_window))
            C = self._reference_covariance(sigma, frame, dtype, diagonal=sigma.dim() == mu.dim())
            sources = edge_links.latent_sources(ctx, b, xi, C, self.attention_residual)
            A = edge_links.infer_latent_receivers(
                ctx, b, sources, sigma=self.content_link_latent_sigma, steps=self.content_link_latent_steps,
                damping=self.content_link_latent_damping, loop_weight=self.content_link_loop_weight,
                loop=loop, stats=latent_out)
            ctx = edge_links.with_receiver_factors(ctx, A, b)
            if loops:
                L64 = self.O.to(torch.float64).permute(1, 0, 2).reshape(K, K) @ self.T.to(torch.float64).reshape(K, K)
                link_state["current"] = edge_links.LinkState(F64, torch.linalg.matrix_exp(-L64), A, ctx.B)
            return ctx

    def _reference_covariance(
        self,
        sigma:    torch.Tensor,        # (..., N, K) diagonal or (..., N, K, K) full local covariance
        frame:    CanonicalFrameContext,
        dtype:    torch.dtype,
        *,
        diagonal: bool,
    ) -> torch.Tensor:                 # (..., N, H, d, d) blocks, or (..., N, K, K)
        r""":math:`C_i = U_i^{-1}\Sigma_iU_i^{-\top}`: diagonal blocks for diagonal beliefs on block frames, else dense.

        The same construction as the covariance bodies (:meth:`_predictive_covariance_update_blockwise`,
        :meth:`_predictive_covariance_update_dense`), symmetrized.
        """
        H, d, K = len(self.irrep_dims), self.irrep_dims[0], self.input_dim
        inverse = frame.inverse.to(dtype)
        if frame.block_diagonal and diagonal:
            sigma_h = sigma.to(dtype).reshape(*sigma.shape[:-1], H, d)                     # (..., N, H, d)
            C = (inverse * sigma_h.unsqueeze(-2)) @ inverse.transpose(-1, -2)              # (..., N, H, d, d)
        else:
            if frame.block_diagonal:                                                       # (..., N, H, d, d) -> (..., N, K, K)
                eye_h = torch.eye(H, dtype=dtype, device=sigma.device)
                inverse = torch.einsum("...hab,hg->...hagb", inverse, eye_h).reshape(*inverse.shape[:-3], K, K)
            local = torch.diag_embed(sigma.to(dtype)) if diagonal else sigma.to(dtype)
            C = inverse @ local @ inverse.transpose(-1, -2)                                # (..., N, K, K)
        return 0.5 * (C + C.transpose(-1, -2))

    def floor_term(
        self,
        floor_anchor: Optional[torch.Tensor],      # the local prior covariance Sigma_p, sigma's layout (broadcastable)
        sigma:        torch.Tensor,                # (..., N, K) or (..., N, K, K) the covariance being floored
        frame:        Optional[CanonicalFrameContext] = None,   # required under floor_mode="relative"
        diagonal:     bool = True,                 # sigma is (..., N, K) diagonal (read under floor_mode="relative")
    ) -> torch.Tensor:                             # sigma's shape and dtype: f Sigma_p (or the relative floor)
        r"""The prior-anchored covariance floor :math:`f\,\Sigma_{p,i}` of the moment-matched readouts.

        Under ``predictive_covariance_floor_mode="relative"`` it is instead :meth:`relative_floor_term`, the floor used
        before October 12, 2026 (kept for testing).

        :math:`\Sigma_{p,i}` is the prior covariance of token :math:`i` in local coordinates (the layer's prior, which is
        the embedding prior under ``prior_handoff_sigma=0``). It involves no frame, so the floored covariance is exactly
        invariant under a common reference change :math:`U_i \to U_ia` for every :math:`a`, covariant under a site action
        that also acts on the prior, and the same in every covariance family. It replaces the floor
        :math:`f\,\tfrac{\operatorname{tr}C_i}{K}U_iU_i^\top`, which changed under a non-orthogonal :math:`a` because
        :math:`\operatorname{tr}(a^{-1}Ca^{-\top}) \ne \operatorname{tr}C` (audit 2026-10-12, item 4 and T5). It does not
        compound with depth: every layer's output is at least :math:`f\,\Sigma_{p,i}` whatever the input was.
        """
        if self.predictive_covariance_floor_mode == "relative":
            return self.relative_floor_term(sigma, frame, diagonal)
        if floor_anchor is None:
            raise ValueError("the moment-matched covariance needs floor_anchor, the local prior covariance Sigma_p of "
                             "each token (the covariance floor is f * Sigma_p)")
        return torch.broadcast_to(self.predictive_covariance_floor * floor_anchor.to(sigma.dtype), sigma.shape)

    def relative_floor_term(
        self,
        sigma: torch.Tensor,                       # (..., N, K) or (..., N, K, K) local input covariance Sigma_i
        frame: Optional[CanonicalFrameContext],
        diagonal: bool,                            # sigma is (..., N, K) diagonal
    ) -> torch.Tensor:                             # sigma's shape and dtype
        r"""The relative floor :math:`f \tfrac{\operatorname{tr}C_i}{K} U_iU_i^\top`, :math:`C_i = U_i^{-1}\Sigma_iU_i^{-\top}`.

        The floor before October 12, 2026: a ridge :math:`f \tfrac{\operatorname{tr}C_i}{K} I` on the attended covariance
        in reference coordinates, here added after the sandwich (equal in exact arithmetic). It scales with the input
        covariance, so the output stays above :math:`f` times the input's mean reference variance however the
        covariance grows through depth, but it changes under a non-orthogonal common reference change
        :math:`U_i \to U_ia` because :math:`\operatorname{tr}(a^{-1}Ca^{-\top}) \ne \operatorname{tr}C` (audit
        2026-10-12, item 4). A diagonal family reads :math:`\mathrm{diag}(U_iU_i^\top)`.
        """
        if frame is None:
            raise ValueError("predictive_covariance_floor_mode='relative' needs the frame of the floored covariance")
        C = self._reference_covariance(sigma, frame, sigma.dtype, diagonal=diagonal)   # blocks or (..., N, K, K)
        blocks = frame.block_diagonal and diagonal                                     # C is (..., N, H, d, d)
        trace = torch.diagonal(C, dim1=-2, dim2=-1).sum(dim=(-2, -1) if blocks else -1)
        scale = trace / sigma.shape[-1]                                                # (..., N) tr(C_i)/K
        forward = frame.forward.to(sigma.dtype)
        if frame.block_diagonal:
            gram = forward @ forward.transpose(-1, -2)                                     # (..., N, H, d, d)
            if diagonal:
                base = torch.diagonal(gram, dim1=-2, dim2=-1).flatten(-2)                  # (..., N, K)
            else:
                H = gram.shape[-3]
                eye_h = torch.eye(H, dtype=gram.dtype, device=gram.device)
                base = torch.einsum("...hab,hg->...hagb", gram, eye_h).reshape(*gram.shape[:-3], H * gram.shape[-1],
                                                                                 H * gram.shape[-1])   # (..., N, K, K)
        else:
            gram = forward @ forward.transpose(-1, -2)                                     # (..., N, K, K)
            base = torch.diagonal(gram, dim1=-2, dim2=-1) if diagonal else gram
        scale = scale[..., None] if diagonal else scale[..., None, None]
        return torch.broadcast_to(self.predictive_covariance_floor * scale * base, sigma.shape).to(sigma.dtype)

    def ceiling_term(
        self,
        sigma_plus:   torch.Tensor,                # (..., N, K) or (..., N, K, K) floored moment-matched covariance
        floor_anchor: torch.Tensor,                # the local prior covariance Sigma_p, sigma_plus's layout
        diagonal:     bool,
    ) -> torch.Tensor:                             # sigma_plus, clamped to Sigma^+ <= c Sigma_p when c is set
        r"""The prior-anchored Loewner ceiling :math:`\Sigma_i^+ \preceq c\Sigma_{p,i}`
        (:func:`vfe3.model.covariance_ceiling.covariance_ceiling`); the identity when ``predictive_covariance_ceiling``
        is None. Under ``block_covariance`` each head block is clamped on its own."""
        if self.predictive_covariance_ceiling is None:
            return sigma_plus
        return covariance_ceiling(sigma_plus, floor_anchor, self.predictive_covariance_ceiling, diagonal=diagonal,
                                  head_dims=self.irrep_dims if (self.block_covariance and not diagonal) else None)

    def minimal_agent_update(
        self,
        mu:        torch.Tensor,                    # (..., N, K) local means
        sigma:     torch.Tensor,                    # (..., N, K) diagonal or (..., N, K, K) full local covariance
        frame:     CanonicalFrameContext,           # the routing frame: query, key, content and write-back
        log_prior: Optional[torch.Tensor] = None,   # (N, N) or (H, N, N) routing log-prior, the label prior pi
        *,
        return_covariance: bool = False,
        capture:   Optional[dict] = None,           # out-param: router, costs and minimal-agent diagnostics
        floor_anchor: Optional[torch.Tensor] = None,   # return_covariance: local prior covariance Sigma_p
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:  # local (mu_plus, sigma_plus or None), input layout
        r"""Minimal-agent posterior of every receiver in local coordinates (:mod:`vfe3.model.minimal_agent`).

        With :math:`\xi_i = U_i^{-1}\mu_i` and :math:`C_i = U_i^{-1}\Sigma_iU_i^{-\top}` (diagonal blocks for diagonal
        beliefs on block frames, else dense), :func:`vfe3.model.minimal_agent.minimal_agent_readout` gives
        :math:`(\xi_i^+, \mathsf K_i^h, M_i^h)` and

        .. math::
            \mu_i^+ = U_i\xi_i^+,\qquad
            \Sigma_i^+ = \Sigma_i - \sum_hZ_i^hM_i^hZ_i^{h\top} + f\,\Sigma_{p,i},
            \qquad Z_i^h = U_i\mathsf K_i^h,

        the posterior mean (no residual is added: the residual is the precision-weighted part of the gain update)
        and, with ``return_covariance``, the posterior covariance plus the prior-anchored floor
        :math:`f\,\Sigma_{p,i}` (:meth:`floor_term`; :math:`f = ` ``predictive_covariance_floor``, :math:`\Sigma_{p,i} = `
        ``floor_anchor``), in the input layout (a diagonal input returns the diagonal). ``predictive_covariance='passthrough'`` is applied by the caller, which keeps
        the incoming covariance and does not request it; the mean uses :math:`C_i` in its gain either way. Under
        ``predictive_covariance_checkpoint`` the whole readout is a non-reentrant checkpoint (it draws no random
        numbers), so values and gradients equal the plain call. ``capture`` receives ``beta_pre`` (the labels each
        head used, ``(..., H, N, N)``), ``score_energy_pre`` (their per-pair costs),
        ``projected_covariance_pre`` (:math:`S_hC_iS_h^\top` at the input), ``projected_key_covariance_pre``
        (:math:`T_hC_jT_h^\top`, the noise-free key law), ``relational_mean_displacement_norm`` and the per-head
        :data:`vfe3.model.minimal_agent.MINIMAL_AGENT_DIAGNOSTIC_KEYS`: per head ``(H,)`` the gain fraction, head
        coupling, mean-field label staleness and exact spread share (means over receivers; each NaN in the other
        mode), the mean-field M-step optimum :math:`n_h^*` of the call
        (:func:`vfe3.model.minimal_agent.channel_noise_mstep`; NaN under exact inference and at the :math:`n\to 0`
        bracket, which the boundary flag marks) and
        :math:`n_h`, and the layer's covariance contraction :math:`\operatorname{tr}C^+/\operatorname{tr}C` (repeated
        per head). Under ``elbo_diagnostic`` the capture also receives the receiver means of the evidence lower bound
        of the returned posterior, the ADF evidence, the exact evidence (NaN unless every receiver's heads are
        uncoupled), the importance-sampled evidence and the gaps to the bound
        (:data:`vfe3.model.minimal_agent.MINIMAL_AGENT_ELBO_KEYS`, float64 scalars, computed without autograd). Under ``predictive_covariance='spread_corrected'`` the returned covariance also
        carries :math:`\sum_hZ'^h_iR'^h_iZ'^{h\top}_i`, :math:`Z'^h_i = U_iP_iS_h^\top`
        (:func:`vfe3.model.minimal_agent.local_posterior_covariance`), with the same floor.

        One host check covers the call. A float32 call that reports a failed factorization (an ADF query covariance
        :math:`A^h` that lost positive definiteness by cancellation, a record covariance, ...) or, under
        a certified ``projected_factorization``, a query certificate (module docstring of
        :mod:`vfe3.model.minimal_agent`) is recomputed whole in float64 from the local beliefs, frame and prior, and
        its outputs and capture are cast back to the input dtypes. The float32 attempt is discarded, so backward
        never runs through its failed factorizations. Every query and key entry of the call (:math:`2BNH`) is then
        counted as recovered in the source-recovery counter that the training loop reads
        (``relational_covariance_repair_count``). A float64 failure raises ``ValueError``; nothing is repaired.
        """
        if self.predictive_readout != "minimal_agent":
            raise ValueError("minimal_agent_update needs predictive_readout='minimal_agent'")
        dtype = _compute_dtype(mu, sigma, self.raw_score_q, self.T, self.channel_log_noise, frame.forward)
        # A float32 mean-field KL may refactor its factors in float64 under the certificate and count them; an
        # escalated call then adds only the rest of its entries below, so no entry is counted twice.
        before = (source_recovery_counter(mu.device) if dtype != torch.float64
                  and self.minimal_agent_inference == "mean_field" and projected_condition_recovery() else None)
        floor = (self.floor_term(floor_anchor, sigma, frame, diagonal=sigma.dim() == mu.dim())
                 if return_covariance else None)
        mu_plus, sigma_plus, record, status = self._minimal_agent_call(
            mu, sigma, frame, log_prior, return_covariance, capture is not None, floor)
        # The one host read of a healthy call; it also resolves raise-only checks queued by
        # vfe3.contracts.deferred_host_checks (earlier frame certificates and prototype-agent checks) first.
        failed, ill_conditioned = host_read(status)
        if failed and dtype == torch.float64:
            raise ValueError(minimal_agent.readout_failure_message(dtype))
        if failed or ill_conditioned:
            # float64 beliefs make the body compute in float64; it casts the frame factors itself, so the float32
            # frame is passed unchanged (a float64 CanonicalFrameContext would re-validate a float32 inverse pair
            # against the float64 residual tolerance).
            mu_plus, sigma_plus, record, status = self._minimal_agent_call(
                mu.double(), sigma.double(), frame, None if log_prior is None else log_prior.double(),
                return_covariance, capture is not None, None if floor is None else floor.double())
            if bool(status[0]):
                raise ValueError(minimal_agent.readout_failure_message(torch.float64)
                                 + "; the float32 call failed first and was recomputed in float64")
            mu_plus = mu_plus.to(mu.dtype)
            sigma_plus = None if sigma_plus is None else sigma_plus.to(sigma.dtype)
            record = {key: (value.to(mu.dtype) if torch.is_tensor(value) and value.is_floating_point() else value)
                      for key, value in record.items()}
            entries = torch.full((), 2 * math.prod(mu.shape[:-1]) * len(self.irrep_dims), dtype=torch.long,
                                 device=mu.device)                     # every query and key entry, 2 B N H
            if before is not None:
                entries = entries - (source_recovery_counter(mu.device) - before)
            _count_source_recovery(entries)
        if capture is not None:
            capture.update(record)
            capture["relational_mean_displacement_norm"] = (mu_plus - mu).norm().detach()
        if sigma_plus is not None:
            sigma_plus = self.ceiling_term(sigma_plus, floor_anchor, diagonal=sigma.dim() == mu.dim())
        return mu_plus, sigma_plus

    def _minimal_agent_call(
        self,
        *args: object,                              # the arguments of _minimal_agent_body
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict, torch.Tensor]:
        r""":meth:`_minimal_agent_body`, as one non-reentrant checkpoint under ``predictive_covariance_checkpoint``."""
        if self.predictive_covariance_checkpoint and torch.is_grad_enabled():
            return checkpoint(self._minimal_agent_body, *args, use_reentrant=False, preserve_rng_state=False)
        return self._minimal_agent_body(*args)

    def _minimal_agent_body(
        self,
        mu:                torch.Tensor,            # (..., N, K)
        sigma:             torch.Tensor,            # (..., N, K) or (..., N, K, K)
        frame:             CanonicalFrameContext,
        log_prior:         Optional[torch.Tensor],
        return_covariance: bool,
        diagnostics:       bool,
        floor:             Optional[torch.Tensor],  # sigma's layout: f Sigma_p (return_covariance only)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict, torch.Tensor]:
        r"""Body of :meth:`minimal_agent_update`: autocast off, IEEE fp32 GEMMs, float64 under an escalated frame.

        Returns the local outputs, the capture record and the readout's status ``(failed, ill_conditioned)``
        ``(2,)`` bool, unchecked (the caller makes the host check)."""
        H, K = len(self.irrep_dims), self.input_dim
        S, T, log_noise = self.raw_score_q, self.T, self.channel_log_noise
        dtype = _compute_dtype(mu, sigma, S, T, log_noise, frame.forward)
        with torch.autocast(device_type=mu.device.type, enabled=False), ieee_fp32_matmul():
            xi = self.canonical_mean(mu.to(dtype), frame).to(dtype)                       # (..., N, K) U_i^{-1} mu_i
            C = self._reference_covariance(sigma, frame, dtype, diagonal=sigma.dim() == mu.dim())
            rope = None
            if self.comparison_rope_base is not None:
                # comparison rope_mode: the record of a source at offset j - i arrives through R_{j-i}.
                rope = self.comparison_rope_rotation(xi.shape[-2], S.shape[1], xi.device, dtype)  # (N, d, d)
            # The constructor refuses spread_corrected with exact inference; a probe that switches the inference to
            # "exact" on a spread-corrected layer (vfe3.posthoc.feature_probes.minimal_agent_gap) gets the exact
            # posterior's covariance, which already carries the spread.
            spread_corrected = (return_covariance and self.predictive_covariance == "spread_corrected"
                                and self.minimal_agent_inference == "mean_field")
            result = minimal_agent.minimal_agent_readout(
                xi, C, S, T, log_noise, log_prior, self.minimal_agent_inference, diagnostics=diagnostics,
                check=False, rope=rope, cavi_steps=self.minimal_agent_cavi_steps,
                spread_corrected=spread_corrected, elbo=diagnostics and self.elbo_diagnostic)
            forward = frame.forward.to(dtype)
            mu_plus = frame_matvec(forward, result.xi, frame.block_diagonal)              # (..., N, K) U_i xi_i^+
            sigma_plus = None
            if return_covariance:
                sigma_plus = minimal_agent.local_posterior_covariance(
                    sigma.to(dtype), forward, frame.block_diagonal, result.gains, result.M, floor.to(dtype),
                    result.spread_gains, result.label_spread)
        record: dict = {}
        if diagnostics:
            with torch.no_grad():
                record = {
                    "beta_pre": result.weights.detach(),                                 # (..., H, N, N)
                    "score_energy_pre": result.cost.detach(),                            # (..., H, N, N)
                    "projected_covariance_pre": result.query_covariance.detach(),        # (..., N, H, d, d)
                    "projected_key_covariance_pre": result.key_covariance.detach(),      # (..., N, H, d, d)
                    "relational_minimal_agent_gain_fraction": result.gain_fraction.detach().reshape(-1, H).mean(0),
                    "relational_minimal_agent_channel_noise": log_noise.detach().to(dtype).exp(),
                    "relational_minimal_agent_head_coupling": result.head_coupling.reshape(-1, H).mean(0),
                    "relational_minimal_agent_label_staleness": result.label_staleness.reshape(-1, H).nanmean(0),
                    "relational_minimal_agent_covariance_contraction":                     # layer value per head
                        result.covariance_contraction.mean().expand(H),
                    "relational_minimal_agent_spread_share": result.spread_share.reshape(-1, H).nanmean(0),
                    # mean field: the M-step optimum n_h^* of the call beside the learned n_h (exact: NaN)
                    "relational_minimal_agent_channel_noise_mstep": result.channel_noise_mstep.to(dtype),
                    # mean field: 1 where n_h^* sits at the n -> 0 bracket (n_h^* is then NaN); exact: NaN
                    "relational_minimal_agent_channel_noise_mstep_boundary": (
                        result.channel_noise_mstep_boundary.to(dtype)
                        if result.channel_noise_mstep_boundary is not None
                        else torch.full((H,), math.nan, dtype=dtype, device=mu.device)),
                }
                if result.elbo is not None:                                              # elbo_diagnostic: nats
                    # The exact evidence is reported only when every receiver's heads are uncoupled (else NaN).
                    record.update(zip(minimal_agent.MINIMAL_AGENT_ELBO_KEYS, (
                        result.elbo.nanmean(), result.adf_evidence.nanmean(),
                        (result.adf_evidence - result.elbo).nanmean(), result.evidence.mean(),
                        result.is_evidence.nanmean(), (result.is_evidence - result.elbo).nanmean())))   # () float64
        status = torch.stack((result.failed, result.ill_conditioned))                         # (2,) bool
        return (mu_plus.to(mu.dtype), None if sigma_plus is None else sigma_plus.to(sigma.dtype), record, status)

    def _receive_runs(
        self,
        receive: 'Optional[torch.Tensor | Sequence[torch.Tensor]]',
    ) -> List[torch.Tensor]:
        r"""Per-run O'_h: the chart evaluated when ``receive`` is None; a lone tensor is one run."""
        self._require_kernel_readout("the effective receiving maps O'")
        if receive is None:
            return self.content_factor_runs()[1]
        return [receive] if torch.is_tensor(receive) else list(receive)

    def _ambient_content_runs(
        self,
        dtype:   torch.dtype,
        receive: 'Optional[torch.Tensor | Sequence[torch.Tensor]]' = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        r"""Per-run full-state T_h (n_r, d_r, K) and effective O'_h (n_r, K, d_r)."""
        self._require_kernel_readout("the ambient content maps")
        return ([T.to(dtype) for T in self._stored_runs("T")],
                [O.to(dtype) for O in self._receive_runs(receive)])

    def _ambient_content_maps(
        self,
        dtype:   torch.dtype,
        receive: Optional[torch.Tensor] = None,   # (H, K, d) precomputed O'_h; None evaluates the chart
    ) -> 'Tuple[torch.Tensor, torch.Tensor] | Tuple[List[torch.Tensor], List[torch.Tensor]]':
        r"""Return full-state T_h and effective O'_h of shapes (H,d,K) and (H,K,d); per-run lists for mixed heads."""
        T_runs, O_runs = self._ambient_content_runs(dtype, receive)
        return (T_runs, O_runs) if self.mixed_heads else (T_runs[0], O_runs[0])

    def predictive_covariance_update(
        self,
        mu:    torch.Tensor,  # (..., N, K) source means, the values predictive_content reads
        sigma: torch.Tensor,  # (..., N, K) diagonal or (..., N, K, K) full local covariance
        beta:  torch.Tensor,  # (..., H, N, N) routing weights, rows sum to one
        frame: CanonicalFrameContext,
        receive: Optional[torch.Tensor] = None,   # (H, K, d) precomputed O'_h; None evaluates the chart
        *,
        edges: Optional[EdgeContext] = None,      # content_link_edges: the layer's edge context (built when None)
        floor_anchor: Optional[torch.Tensor] = None,   # the local prior covariance Sigma_p (required)
    ) -> torch.Tensor:        # same layout as sigma: the moment-matched covariance Sigma_i^+
        r"""Moment-matched covariance of the predictive kernel (GL(K) v2, Theorem thm:probabilistic_status (2)).

        With independent head labels J^h ~ beta_i^h, independent sources k_j ~ N(mu_j, Sigma_j), and
        the kernel k_i' | k, J ~ N(r k_i + sum_h L_{iJ^h}^h k_{J^h}, Q_i), where r = 1 under
        ``attention_residual`` and 0 otherwise, the covariance in reference coordinates is

        .. math::
            C_i^{\mathrm{att}} = r\Bigl[C_i + \sum_h \beta_{ii}^h (M_h C_i + C_i M_h^\top)\Bigr]
              + \sum_j \sum_{h,h'} c_{ij}^{hh'} M_h C_j M_{h'}^\top
              + \sum_h \Bigl[\sum_j \beta_{ij}^h n_{ij}^h n_{ij}^{h\top} - \bar n_i^h \bar n_i^{h\top}\Bigr],

        with M_h = O_h T_h, C_j = U_j^{-1} Sigma_j U_j^{-T}, c^{hh} = beta^h, c^{hh'} = beta^h beta^{h'}
        (h != h'), n_ij^h = M_h xi_j. The result is Sigma_i^+ = U_i C_i^att U_i^T + f Sigma_{p,i} in the frame the
        mean content uses, with the kernel noise Q_i = f Sigma_{p,i} the prior-anchored floor in local coordinates
        (:meth:`floor_term`, ``floor_anchor``; audit 2026-10-12 T5): frame-free, so it is exactly reference-invariant and
        holds as Sigma_i^+ >= f Sigma_{p,i} in every family. A diagonal input returns diag(U_i C_i^att U_i^T) +
        f diag(Sigma_{p,i}), the KL(p || q) moment projection onto the diagonal family.
        Head-space pooling keeps the cost at an attention with d x d values per head pair.
        Under ``predictive_covariance_checkpoint`` the update is recomputed in backward (non-reentrant
        checkpoint; the update draws no random numbers), so values and gradients are unchanged.

        Under ``predictive_covariance_fp64_escalation`` a float32 update whose floored output has a nonpositive or
        non-finite variance (diagonal family) or fails a Cholesky factorization (full and head-block families) has
        its failing sequences recomputed in float64 and cast back (:meth:`_escalate_failed_sequences`); their
        float32 values are discarded, so backward runs through the float64 recompute there. The diagonal :math:`\mathrm{diag}(U_iC_i^{\mathrm{att}}U_i^\top)` leaves reference
        coordinates through :math:`U_i`, so float32 rounding of :math:`C_i^{\mathrm{att}}` is amplified by about
        :math:`\kappa(U_i)^2`; on ill-conditioned block_glk frames it returned negative variances. The check is one
        host read per call; every token of a recomputed sequence is counted in its own counter
        (``covariance_escalation_elements``). A float64 failure raises ``ValueError``; nothing is repaired. The ceiling
        is applied after the check.

        Under ``content_link_edges`` the kernel is the same with :math:`M_h \to M_h^{ij} = G_{ij}\Pi_h`:

        .. math::
            C_i^{\mathrm{att}} = g_iF\Bigl(\hat P_i + \sum_hR_h\tilde\Sigma_i^hR_h^\top
              + A_iX_i + X_i^\top A_i^\top + A_iY_iA_i^\top\Bigr)F^\top g_i^\top
              + r\bigl[C_i + \mathrm{cross}_i + \mathrm{cross}_i^\top\bigr],
            \qquad \mathrm{cross}_i = \sum_h\beta_{ii}^h\,g_iF(I + A_i\Phi_{ii}B_i^\top)\Pi_hC_i,

        with :math:`X_i, Y_i` of :func:`vfe3.model.edge_links.edge_covariance_factors`; the edge context is
        computed outside the checkpointed region and enters it as an input.
        """
        self._require_kernel_readout("predictive_covariance_update")
        floor = self.floor_term(floor_anchor, sigma, frame, diagonal=sigma.dim() == mu.dim())
        if edges is None and self.content_link_edges != "off":
            edges = self.edge_context(mu, beta, frame, receive, sigma=sigma)
        if edges is not None:
            receive = edge_links.receive_heads(edges, len(self.irrep_dims))
        extra = () if edges is None else (edges,)
        diagonal = sigma.dim() == mu.dim()
        sigma_plus = self._kernel_covariance(mu, sigma, beta, frame, receive, extra) + floor   # f Sigma_p, local
        if self.predictive_covariance_fp64_escalation and sigma_plus.dtype != torch.float64:
            sigma_plus = self._escalate_failed_sequences(sigma_plus, floor, mu, sigma, beta, frame, receive, extra,
                                                         diagonal)
        return self.ceiling_term(sigma_plus, floor_anchor, diagonal=diagonal)   # at most c Sigma_p when set

    def _escalate_failed_sequences(
        self,
        sigma_plus: torch.Tensor,                  # (B, N, K) or (B, N, K, K) floored float32 update
        floor:      torch.Tensor,                  # sigma_plus's shape: f Sigma_p
        mu:         torch.Tensor,                  # (B, N, K)
        sigma:      torch.Tensor,                  # (B, N, K) or (B, N, K, K)
        beta:       torch.Tensor,                  # (B, H, N, N)
        frame:      CanonicalFrameContext,         # (B or 1, N, H, d, d) or unbatched
        receive:    Optional[torch.Tensor],
        extra:      Tuple[EdgeContext, ...],
        diagonal:   bool,
    ) -> torch.Tensor:                             # sigma_plus with every failing sequence recomputed in float64
        r"""The float64 escalation of :meth:`predictive_covariance_update`, restricted to the failing sequences.

        A token fails when a variance is nonpositive or non-finite (diagonal) or its matrix does not factor by
        Cholesky (full and head-block families); one host read finds the sequences with a failing token. The
        update of a receiver reads only its own sequence (beta is (B, H, N, N)), so those sequences are recomputed
        whole in float64 and written back, and the others keep their float32 values and graph bit for bit. Every
        token of a recomputed sequence is counted. Without a single leading batch axis, or with edge links (whose
        context is not split here), the whole call is recomputed. A float64 failure raises; nothing is repaired.
        """
        with torch.no_grad():
            if diagonal:
                bad = ~(torch.isfinite(sigma_plus) & (sigma_plus > 0)).all(dim=-1)              # (..., N)
            else:
                bad = torch.linalg.cholesky_ex(sigma_plus.detach())[1] != 0                    # (..., N)
        batched = mu.dim() == 3 and beta.dim() == 4 and not extra
        if batched:
            rows = bad.any(dim=-1).nonzero().flatten()                     # failing sequences (one host sync)
            n_rows = rows.numel()
            if n_rows == 0:
                return sigma_plus
        elif not bool(host_read(bad.any().reshape(1))[0]):
            return sigma_plus
        wide_receive = None if receive is None else [O.double() for O in self._receive_runs(receive)]
        if not batched:
            # float64 beliefs make the body compute in float64; it casts the frame and the content maps itself.
            sigma_wide = self._kernel_covariance(
                mu.double(), sigma.double(), beta.double(), frame, wide_receive,
                tuple(edge_links.cast_context(e, torch.float64) for e in extra)) + floor.double()
            if not _covariance_positive(sigma_wide, diagonal):
                raise ValueError(_FP64_ESCALATION_FAILURE)
            _count_covariance_escalation(math.prod(mu.shape[:-1]), mu.device)
            return sigma_wide.to(sigma_plus.dtype)
        # A frame with its own batch axis is split with the beliefs; a shared (unbatched or size-1) frame broadcasts.
        per_sequence = (frame.forward.dim() >= (5 if frame.block_diagonal else 4)
                        and frame.forward.shape[0] == mu.shape[0] > 1)
        sub_frame = (CanonicalFrameContext(frame.forward[rows], frame.inverse[rows],
                                           block_diagonal=frame.block_diagonal) if per_sequence else frame)
        sigma_wide = self._kernel_covariance(
            mu[rows].double(), sigma[rows].double(), beta[rows].double(), sub_frame, wide_receive, ())             + floor[rows].double()
        if not _covariance_positive(sigma_wide, diagonal):
            raise ValueError(_FP64_ESCALATION_FAILURE)
        _count_covariance_escalation(n_rows * mu.shape[-2], mu.device)
        return sigma_plus.index_copy(0, rows, sigma_wide.to(sigma_plus.dtype))

    def _kernel_covariance(
        self,
        mu:      torch.Tensor,                     # (..., N, K)
        sigma:   torch.Tensor,                     # (..., N, K) or (..., N, K, K)
        beta:    torch.Tensor,                     # (..., H, N, N)
        frame:   CanonicalFrameContext,
        receive: Optional[torch.Tensor],           # (H, K, d) or None
        extra:   Tuple[EdgeContext, ...],          # () or (edges,)
    ) -> torch.Tensor:                             # sigma's layout: U C^att U^T, without the floor
        r""":meth:`_predictive_covariance_update`, one non-reentrant checkpoint under ``predictive_covariance_checkpoint``."""
        if self.predictive_covariance_checkpoint and torch.is_grad_enabled():
            return checkpoint(self._predictive_covariance_update, mu, sigma, beta, frame, receive, *extra,
                              use_reentrant=False, preserve_rng_state=False)
        return self._predictive_covariance_update(mu, sigma, beta, frame, receive, *extra)

    @ieee_fp32_matmul()
    def _predictive_covariance_update(
        self,
        mu:    torch.Tensor,  # (..., N, K)
        sigma: torch.Tensor,  # (..., N, K) or (..., N, K, K)
        beta:  torch.Tensor,  # (..., H, N, N)
        frame: CanonicalFrameContext,
        receive: Optional[torch.Tensor] = None,   # (H, K, d)
        edges: Optional[EdgeContext] = None,
    ) -> torch.Tensor:
        r"""Body of :meth:`predictive_covariance_update`: blockwise for diagonal beliefs on block frames.

        Runs at IEEE fp32 whatever ``fp32_matmul_precision`` is (:func:`vfe3.runtime.ieee_fp32_matmul`),
        also when the checkpoint recomputes it. :math:`\mathrm{diag}(U C^{\mathrm{att}} U^\top)` with
        :math:`C = U^{-1}\Sigma U^{-\top}` passes through :math:`U^{-1}` and back through :math:`U`, so
        its rounding is amplified by about :math:`\kappa(U)^2`; at TF32's :math:`2^{-10}` small output
        variances came out nonpositive on ill-conditioned frames.

        Under ``block_covariance`` (``gaussian_block``) the head-block covariance always takes the blockwise
        route, which returns the head-block M-projection; it needs the compact block frame.
        """
        if self.block_covariance and not frame.block_diagonal:
            raise ValueError("block_covariance (gaussian_block) reads the compact (..., N, H, d, d) block frame; "
                             "a dense frame was supplied")
        if frame.block_diagonal and (sigma.dim() == mu.dim() or self.block_covariance):
            return self._predictive_covariance_update_blockwise(mu, sigma, beta, frame, receive, edges)
        return self._predictive_covariance_update_dense(mu, sigma, beta, frame, receive, edges)

    def _predictive_covariance_update_blockwise(
        self,
        mu:    torch.Tensor,  # (..., N, K) source means
        sigma: torch.Tensor,  # (..., N, K) diagonal, or (..., N, K, K) head-block diagonal (block_covariance)
        beta:  torch.Tensor,  # (..., H, N, N) routing weights, rows sum to one
        frame: CanonicalFrameContext,             # block_diagonal: (..., N, H, d, d) blocks U_{i,m}
        receive: Optional[torch.Tensor] = None,   # (H, K, d) precomputed O'_h; None evaluates the chart
        edges: Optional[EdgeContext] = None,      # content_link_edges: the layer's edge context
    ) -> torch.Tensor:        # (..., N, K) diag(Sigma_i^+), or (..., N, K, K) its head blocks
        r"""Diagonal (or head blocks) of the moment-matched covariance on block-diagonal frames.

        With :math:`U_i=\bigoplus_m U_{i,m}` and :math:`\Sigma_i=\mathrm{diag}(\sigma_i)` the reference
        covariance is block diagonal, :math:`C_i=\bigoplus_m C_{i,m}` with
        :math:`C_{i,m}=U_{i,m}^{-1}\mathrm{diag}(\sigma_{i,m})U_{i,m}^{-\top}`. Writing
        :math:`T_h^{(m)}` and :math:`O_h^{(l)}` for the :math:`d\times d` column and row blocks of
        :math:`T_h` and :math:`O'_h`,

        .. math::
            T_h C_i T_g^\top = \sum_m T_h^{(m)} C_{i,m} T_g^{(m)\top},

        and because :math:`U_i` is block diagonal,
        :math:`\mathrm{diag}(\Sigma_i^+)=\mathrm{diag}(U_i C_i^{\mathrm{att}} U_i^\top)` reads only the
        diagonal blocks

        .. math::
            [C_i^{\mathrm{att}}]_{ll} = \sum_{h,g} O_h^{(l)} P_{i,hg} O_g^{(l)\top}
              + r\bigl[C_{i,l} + X_{i,l} + X_{i,l}^\top\bigr],
            \qquad X_{i,l} = \sum_h \beta_{ii}^h O_h^{(l)} T_h^{(l)} C_{i,l},

        with :math:`P_{i,hg}` the head-pair pooled covariance of
        :meth:`_predictive_covariance_update_dense` (the floor :math:`f\,\Sigma_{p,i}` is added in local coordinates by
        :meth:`predictive_covariance_update`). It is the
        dense update reassociated, equal up to rounding, at about :math:`2K^3 + NK^2` multiply-adds per
        token instead of about :math:`9.5K^3 + NK^2`, with no :math:`K \times K` intermediate.

        Head-block family (``block_covariance``, ``gaussian_block``). The input
        :math:`\Sigma_i=\bigoplus_m\Sigma_{i,m}` (dense storage, zero off-head blocks) gives
        :math:`C_{i,m}=U_{i,m}^{-1}\Sigma_{i,m}U_{i,m}^{-\top}`, the same block-diagonal :math:`C_i` the formulas
        above take, and the result is the head-block M-projection of :math:`\Sigma_i^+=U_iC_i^{\mathrm{att}}U_i^\top`,

        .. math::
            \Sigma_i^{+} \mapsto \bigoplus_l U_{i,l}\,[C_i^{\mathrm{att}}]_{ll}\,U_{i,l}^\top,

        returned in the dense layout. :math:`\arg\min_q\mathrm{KL}(p\|q)` over block-diagonal Gaussians keeps the
        marginal blocks, and under the passive action :math:`U_i\to g_iU_i` (:math:`g_i\in\prod_hGL(d)`) the
        reference blocks :math:`[C_i^{\mathrm{att}}]_{ll}` are unchanged, so the projection is frame-independent;
        the diagonal projection :math:`\mathrm{diag}(U_iC_i^{\mathrm{att}}U_i^\top)` is not.
        """
        H, d = len(self.irrep_dims), self.irrep_dims[0]
        block = sigma.dim() == mu.dim() + 1                                            # head-block family
        dtype = _compute_dtype(mu, sigma, self.T, self.O, frame.forward)
        if edges is None and self.content_link_edges != "off":
            edges = self.edge_context(mu, beta, frame, receive, sigma=sigma)
        if edges is not None:
            edges = edge_links.cast_context(edges, dtype)
            receive = edge_links.receive_heads(edges, H)
        with torch.autocast(device_type=mu.device.type, enabled=False):
            forward, inverse = frame.forward.to(dtype), frame.inverse.to(dtype)        # (..., N, H, d, d)
            if block:
                S = sigma.to(dtype).reshape(*sigma.shape[:-2], H, d, H, d)            # (..., N, H, d, H, d)
                sigma_blocks = torch.diagonal(S, dim1=-4, dim2=-2).movedim(-1, -3)    # (..., N, H, d, d) Sigma_{i,m}
                C = inverse @ sigma_blocks @ inverse.transpose(-1, -2)                 # (..., N, H, d, d) C_{i,m}
            else:
                sigma_h = sigma.to(dtype).reshape(*sigma.shape[:-1], H, d)             # (..., N, H, d)
                C = (inverse * sigma_h.unsqueeze(-2)) @ inverse.transpose(-1, -2)      # (..., N, H, d, d) C_{i,m}
            C = 0.5 * (C + C.transpose(-1, -2))
            mu_h = mu.to(dtype).reshape(*mu.shape[:-1], H, d)
            xi = (inverse @ mu_h.unsqueeze(-1)).squeeze(-1).flatten(-2)                # (..., N, K)
            T_hat, O_hat = self._ambient_content_maps(dtype, receive)                  # (H,d,K), (H,K,d)
            T_blocks = T_hat.reshape(H, d, H, d)                                       # [h, s, m, k] = T_h^{(m)}[s, k]
            O_blocks = O_hat.reshape(H, H, d, d)                                       # [h, l, k, s] = O_h^{(l)}[k, s]
            b = beta.to(dtype)                                                         # (..., H, N, N)
            eye_h = torch.eye(H, dtype=dtype, device=mu.device)

            # Common shift by token 0, exactly as on the dense route.
            xi = xi - xi[..., :1, :]                                                   # (..., N, K)
            values = torch.einsum("hsk,...nk->...nhs", T_hat, xi)                      # (..., N, H, d)
            nl = C.dim() - 4                                                           # leading batch dims
            C_cols = C.movedim(-3, 0).movedim(-2, 1).reshape(H, d, -1)                # (m, k, [..., n, t])
            T_rows = T_blocks.permute(2, 0, 1, 3).reshape(H, H * d, d)                # (m, [h, s], k)
            TC = _per_block_bmm(T_rows, C_cols).view(H, H, d, *C.shape[:-3], d)       # (m, h, s, ..., n, t)
            TC = TC.permute(*range(3, nl + 4), 1, 0, 2, nl + 4)                        # (..., N, H, H, d, d) T_h^{(m)} C_{i,m}
            source = torch.einsum("...nhmsk,gtmk->...nhgst", TC, T_blocks)            # (..., N, H, H, d, d) T_h C_i T_g^T
            c = torch.einsum("...hij,...gij->...hgij", b, b)
            c = c * (1.0 - eye_h)[..., None, None] + torch.einsum("...hij,hg->...hgij", b, eye_h)
            pooled = torch.einsum("...hgij,...jhgst->...ihgst", c, source)            # (..., N, H, H, d, d)
            mean = torch.einsum("...hij,...jhs->...ihs", b, values)                    # (..., N, H, d)
            spread = _centered_spread(b, values, mean)                                 # (..., N, H, d, d)
            pooled.diagonal(dim1=-4, dim2=-3).add_(spread.movedim(-3, -1))            # P_hh += S^h, in place
            lifted = torch.einsum("hks,...nhgst->...ngkt", O_hat, pooled)             # (..., N, H, K, d) sum_h O'_h P_hg
            lifted = lifted.reshape(*lifted.shape[:-2], H, d, d)                       # (..., N, g, l, k, t)
            rows = lifted.permute(nl + 2, *range(nl + 1), nl + 3, nl + 1, nl + 4)      # (l, ..., n, k, g, t)
            O_cols = O_blocks.permute(1, 0, 3, 2).reshape(H, H * d, d)                # (l, [g, t], m)
            C_att = _per_block_bmm(rows.reshape(H, -1, H * d), O_cols)                 # (l, [..., n, k], m)
            C_att = C_att.view(H, *C.shape[:-3], d, d).movedim(0, -3)                  # (..., N, H, d, d) blocks [ll]
            if edges is not None:
                # [g_i F (P^ + R Sigma~ R^T + A X + X^T A^T + A Y A^T) F^T g_i^T]_ll (edge_links).
                A_hat, X_hat, Y = edge_links.edge_covariance_factors(edges, b, c, source, values, mean)
                inner = C_att + edge_links.edge_correction_blocks(A_hat, X_hat, Y, H)
                C_att = (edge_links.outer_gauge_blocks(
                    edges, inner, edge_links.edge_inner_rows(edges, pooled, A_hat, X_hat, Y))
                    if edges.mode == "pure_gauge" else inner)
            if self.attention_residual:
                self_weight = b.diagonal(dim1=-2, dim2=-1).transpose(-1, -2)          # (..., N, H) beta_ii^h
                weighted = TC * self_weight[..., None, None, None]                    # (..., N, H, H, d, d)
                weighted = weighted.permute(nl + 2, nl + 1, nl + 3, *range(nl + 1), nl + 4)   # (l, h, s, ..., n, m)
                O_rows = O_blocks.permute(1, 2, 0, 3).reshape(H, d, H * d)            # (l, k, [h, s])
                cross = _per_block_bmm(O_rows, weighted.reshape(H, H * d, -1))         # (l, k, [..., n, m])
                cross = cross.view(H, d, *C.shape[:-3], d).movedim((0, 1), (-3, -2))   # (..., N, H, d, d) X_{i,l}
                if edges is not None:                                                  # blocks of sum_h beta_ii M_h^{ii} C_i
                    cross = edge_links.edge_residual_cross_blocks(edges, A_hat, self_weight, TC, cross)
                C_att = C_att + C + cross + cross.transpose(-1, -2)
            if block:
                blocks = forward @ C_att @ forward.transpose(-1, -2)                   # (..., N, H, d, d) U_l [C_att]_ll U_l^T
                blocks = 0.5 * (blocks + blocks.transpose(-1, -2))
                result = torch.einsum("...hab,hg->...hagb", blocks, eye_h).reshape(*sigma.shape)   # (..., N, K, K)
                return result.to(sigma.dtype)
            result = ((forward @ C_att) * forward).sum(dim=-1)                         # (..., N, H, d) diag(U C U^T)
        return result.flatten(-2).to(sigma.dtype)

    def _predictive_covariance_update_dense(
        self,
        mu:    torch.Tensor,  # (..., N, K)
        sigma: torch.Tensor,  # (..., N, K) or (..., N, K, K)
        beta:  torch.Tensor,  # (..., H, N, N)
        frame: CanonicalFrameContext,
        receive: Optional[torch.Tensor] = None,   # (H, K, d)
        edges: Optional[EdgeContext] = None,      # content_link_edges: the layer's edge context
    ) -> torch.Tensor:
        r"""Dense K x K evaluation of :meth:`predictive_covariance_update` (full family or dense frames).

        With heads of different sizes the head-pair terms run over pairs of runs (r, g): the
        coefficients :math:`c^{hh'}` act on :math:`d_r\times d_g` blocks
        :math:`T_hC_jT_{h'}^\top`, one batched contraction per run pair, and the spread and the
        self-coupling :math:`c^{hh}=\beta^h` live on the diagonal pairs r = g.
        """
        H, K = len(self.irrep_dims), self.input_dim
        diagonal = sigma.dim() == mu.dim()
        dtype = _compute_dtype(mu, sigma, *self._content_parameters(), frame.forward)
        if edges is None and self.content_link_edges != "off":
            edges = self.edge_context(mu, beta, frame, receive, sigma=sigma)
        if edges is not None:
            edges = edge_links.cast_context(edges, dtype)
            receive = edge_links.receive_heads(edges, H)
        with torch.autocast(device_type=mu.device.type, enabled=False):
            forward, inverse = frame.forward.to(dtype), frame.inverse.to(dtype)
            if frame.block_diagonal:                            # (..., N, H, d, d) -> (..., N, K, K)
                eye_h = torch.eye(H, dtype=dtype, device=mu.device)
                forward = torch.einsum("...hab,hg->...hagb", forward, eye_h).reshape(*forward.shape[:-3], K, K)
                inverse = torch.einsum("...hab,hg->...hagb", inverse, eye_h).reshape(*inverse.shape[:-3], K, K)
            local = torch.diag_embed(sigma.to(dtype)) if diagonal else sigma.to(dtype)
            C = inverse @ local @ inverse.transpose(-1, -2)                        # (..., N, K, K)
            C = 0.5 * (C + C.transpose(-1, -2))
            xi = (inverse @ mu.to(dtype).unsqueeze(-1)).squeeze(-1)                # (..., N, K)
            T_runs, O_runs = self._ambient_content_runs(dtype, receive)            # (n_r,d_r,K), (n_r,K,d_r)
            b = beta.to(dtype)                                                     # (..., H, N, N)

            # Shift every source mean by token 0's. Each beta row sums to 1, so the spread
            # sum_j beta_ij v_j v_j^T - m_i m_i^T is invariant under a common shift of the v_j, and
            # the shift is causal because token 0 lies in every row's support. The spread itself is
            # accumulated about each row's routed mean (_centered_spread), so no offset cancels there.
            xi = xi - xi[..., :1, :]                                              # (..., N, K)
            b_runs = [self._head_slice(b, run, -3) for run in range(len(self.runs))]   # (..., n_r, N, N)
            TC_runs = [torch.einsum("hsk,...nkm->...nhsm", T_hat, C)               # (..., N, n_r, d_r, K)
                       for T_hat in T_runs]
            C_att = None
            for r, (T_r, O_r, b_r, TC) in enumerate(zip(T_runs, O_runs, b_runs, TC_runs)):
                for g, (T_g, O_g, b_g) in enumerate(zip(T_runs, O_runs, b_runs)):
                    source = torch.einsum("...nhsm,gtm->...nhgst", TC, T_g)       # (..., N, n_r, n_g, d_r, d_g)
                    c = torch.einsum("...hij,...gij->...hgij", b_r, b_g)
                    if r == g:
                        # c^{hh'}_ij: beta^h on the head diagonal, beta^h beta^h' off it.
                        eye_h = torch.eye(b_r.shape[-3], dtype=dtype, device=mu.device)
                        c = c * (1.0 - eye_h)[..., None, None] + torch.einsum("...hij,hg->...hgij", b_r, eye_h)
                    pooled = torch.einsum("...hgij,...jhgst->...ihgst", c, source)
                    if r == g:
                        values = torch.einsum("hsk,...nk->...nhs", T_r, xi)        # (..., N, n_r, d_r)
                        mean = torch.einsum("...hij,...jhs->...ihs", b_r, values)  # (..., N, n_r, d_r)
                        spread = _centered_spread(b_r, values, mean)               # (..., N, n_r, d_r, d_r)
                        pooled = pooled + torch.einsum("...ihst,hg->...ihgst", spread, eye_h)
                    lifted = torch.einsum("hks,...nhgst->...ngkt", O_r, pooled)   # (..., N, n_g, K, d_g)
                    term = torch.einsum("...ngkt,gmt->...nkm", lifted, O_g)        # (..., N, K, K)
                    C_att = term if C_att is None else C_att + term
            if edges is not None:
                # One run (equal heads): source, c, values and mean are the head-pair pieces above.
                # g_i F (P^ + R Sigma~ R^T + A X + X^T A^T + A Y A^T) F^T g_i^T (edge_links).
                A_hat, X_hat, Y = edge_links.edge_covariance_factors(edges, b, c, source, values, mean)
                C_att = edge_links.outer_gauge_sandwich(
                    edges, C_att + edge_links.edge_correction_dense(A_hat, X_hat, Y))
            if self.attention_residual:
                cross = None
                for O_r, b_r, TC in zip(O_runs, b_runs, TC_runs):
                    self_weight = b_r.diagonal(dim1=-2, dim2=-1).transpose(-1, -2)   # (..., N, n_r) beta_ii^h
                    term = torch.einsum("hks,...nhsm->...nkm", O_r,
                                        TC * self_weight[..., None, None])        # sum_h beta_ii M_h C_i
                    cross = term if cross is None else cross + term
                if edges is not None:                                              # sum_h beta_ii M_h^{ii} C_i
                    cross = edge_links.outer_gauge_left(edges, cross + edge_links.edge_residual_cross_dense(
                        edges, A_hat, self_weight, TC_runs[0]))
                C_att = C_att + C + cross + cross.transpose(-1, -2)
            result = forward @ C_att @ forward.transpose(-1, -2)
            result = 0.5 * (result + result.transpose(-1, -2))
            if diagonal:
                result = torch.diagonal(result, dim1=-2, dim2=-1)
        return result.to(sigma.dtype)

    @staticmethod
    def _orthonormal_columns(raw: torch.Tensor) -> torch.Tensor:
        r"""Return signed thin-QR Q, rejecting deficient raw column rank before use."""
        with torch.autocast(device_type=raw.device.type, enabled=False):
            work = raw.to(_compute_dtype(raw))
            if not bool(torch.isfinite(work).all()):
                raise ValueError("relational raw comparison/content basis must be finite and full rank")
            Q, R = torch.linalg.qr(work, mode="reduced")
            pivots = R.diagonal(dim1=-2, dim2=-1)
            scale = work.norm(dim=(-2, -1), keepdim=False)
            threshold = torch.finfo(work.dtype).eps * max(work.shape[-2:]) * scale
            if bool((pivots.abs() <= threshold.unsqueeze(-1)).any()):
                raise ValueError("relational raw comparison/content basis lost column rank (QR pivot tolerance)")
            signs = torch.where(pivots < 0., -torch.ones_like(pivots), torch.ones_like(pivots))
            return Q * signs.unsqueeze(-2)

    def _require_equal_heads(self, what: str) -> None:
        if self.mixed_heads:
            raise ValueError(f"{what} stacks equal heads; this partition has head sizes {self.irrep_dims} "
                             f"(use the per-run accessor)")

    def comparison_pair(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Return (S_Q,S_K), each (H,r,input_dim), with live role graphs (equal heads); S_K = T under the minimal agent."""
        self._require_equal_heads("comparison_pair")
        if self.tie_comparison_maps:
            return self.raw_score_q, self.raw_score_q                                     # S_K := S_Q
        return self.raw_score_q, (self.T if self.predictive_readout == "minimal_agent" else self.raw_score_k)

    def comparison_runs(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        r"""Per-run (S_Q, S_K), each (n_r, d_r, input_dim), with live role graphs; the tied S_K = T under the minimal agent."""
        key = ("T" if self.predictive_readout == "minimal_agent"
               else "raw_score_q" if self.tie_comparison_maps else "raw_score_k")
        return list(zip(self._stored_runs("raw_score_q"), self._stored_runs(key)))

    def routing_temperature(
        self,
        tau:   'float | torch.Tensor',          # configured tau_h = kappa_h sqrt(d_h): scalar or (H,)
        dtype: torch.dtype,                     # working dtype of the routing cost
    ) -> 'float | torch.Tensor':                # tau unchanged, or the learned (H,) tau_{l,h}
        r"""Gibbs routing temperature of this layer.

        Under ``routing="noisy_key"`` it is exactly 1 (the learned key noise :math:`n_h` sets the sharpness and the
        routing objective is the KL of the declared joint). Otherwise, without ``learn_attention_temperature`` the
        configured temperature is returned unchanged. With it,

        .. math::
            \tau_{\ell,h} = \kappa_h\sqrt{d_h}\exp(3\tanh(\delta_{\ell,h}/3)),

        so the log-temperature offset :math:`3\tanh(\delta/3)` equals :math:`\delta` to first order, is
        :math:`0` at :math:`\delta = 0` (the configured temperature exactly, as at initialization) and stays
        inside the :math:`\pm 3` window of the decoder's ``decode_log_scale``. This smooth bound replaced a hard
        ``clamp(delta, -3, 3)`` (audit 2026-10-09 round 2), which was absorbing: past the window the offset had
        zero gradient. The bound has gradient :math:`1 - \tanh^2(\delta/3) > 0` everywhere. A saved nonzero
        offset reloads at a different temperature than under the clamp. The product is formed in float64 and
        returned at ``dtype``. The temperature acts after the ``kl_max`` clamp, so it sharpens or flattens
        only costs below the cap.
        """
        if self.routing_key_log_noise is not None:
            return 1.0                                  # noisy_key: the KL of the declared joint, no kappa sqrt(d)
        delta = self.routing_log_temperature
        if delta is None:
            return tau
        scale = torch.exp(3.0 * torch.tanh(delta.to(torch.float64) / 3.0))               # (H,)
        base = torch.as_tensor(tau, dtype=torch.float64, device=scale.device)            # () or (H,)
        return (base * scale).to(dtype)                                                  # (H,)

    def comparison_rope_rotation(
        self,
        n:      int,                   # sequence length N
        r:      int,                   # comparison rank of the rotated run
        device: torch.device,
        dtype:  torch.dtype,
    ) -> torch.Tensor:                 # (N, r, r) R_i, positions 0..N-1
        r"""Cached comparison-space RoPE rotation :math:`R_i` under a comparison ``rope_mode``.

        The rotation is a constant of (N, r, base, fraction, device, dtype) with no graph, so one build
        serves every layer and forward. A build under ``torch.inference_mode`` is not cached, since an
        inference tensor cannot later be saved for backward.
        """
        from vfe3.geometry.rope import build_rope_rotation
        key = (n, r, self.comparison_rope_base, self.comparison_rope_fraction, str(device), dtype)
        cache = self.__dict__.setdefault("_comparison_rope_cache", {})
        rotation = cache.get(key)
        if rotation is None:
            rotation = build_rope_rotation(torch.arange(n, device=device), [r], base=self.comparison_rope_base,
                                           fraction=self.comparison_rope_fraction, device=device, dtype=dtype)
            if not torch.is_inference_mode_enabled():
                if len(cache) >= 8:
                    cache.clear()
                cache[key] = rotation
        return rotation

    def content_factors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Return full-state (T, O') for M_h=O'_h T_h; O' = O in the affine chart (the stored parameter)."""
        self._require_kernel_readout("content_factors")
        self._require_equal_heads("content_factors")
        return self.T, chart_receive_map(self.T, self.O, self.content_link_chart)

    def content_factor_runs(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        r"""Per-run (T, O'): T (n_r, d_r, K) and O' (n_r, K, d_r) of the content link's chart."""
        self._require_kernel_readout("content_factor_runs")
        T_runs = self._stored_runs("T")
        shared = self.__dict__.get("_shared_receive_runs")          # installed by shared_content_charts
        if shared is not None:
            return T_runs, shared
        return T_runs, chart_receive_runs(T_runs, self._stored_runs("O"), self.content_link_chart)

    @ieee_fp32_matmul()
    def canonical_mean(
        self,
        mu:    torch.Tensor,  # (..., N, K)
        frame: CanonicalFrameContext,
    ) -> torch.Tensor:        # (..., N, K)
        r"""Compute xi_i=U_i^{-1}mu_i using the supplied realized ordered frame."""
        K = self.input_dim
        if mu.shape[-1] != K:
            raise ValueError("relational mean dimension does not match its head partition")
        if frame.block_diagonal and (self.mixed_heads or frame.inverse.shape[-3:] != (
                len(self.irrep_dims), self.irrep_dims[0], self.irrep_dims[0])):
            raise ValueError("relational compact frame blocks do not match its head partition")
        if not frame.block_diagonal and frame.inverse.shape[-2:] != (K, K):
            raise ValueError("relational dense frame dimension does not match the mean")
        dtype = _compute_dtype(mu, frame.inverse)
        with torch.autocast(device_type=mu.device.type, enabled=False):
            coordinates = (mu.to(dtype).reshape(*mu.shape[:-1], len(self.irrep_dims), self.irrep_dims[0])
                           if frame.block_diagonal else mu.to(dtype))
            canonical = (frame.inverse.to(dtype) @ coordinates.unsqueeze(-1)).squeeze(-1)
        return canonical.reshape(*mu.shape[:-1], K)
