r"""Minimal-agent attention readout: the receiver conditions exactly on a noisy comparison record.

Source theory: ``Research/manuscripts/Attention/minimal_agent_model.tex`` (Definition ``ma:def:model``,
Proposition ``ma:prop:minimal`` (a)-(e), Proposition ``ma:prop:mean_field``); implementation plan
``docs/implementation/minimal-agent-plan-2026-10-06.md``, Sections 1 and 2.

Token :math:`i` carries the reference moments :math:`\xi_i = U_i^{-1}\mu_i` and
:math:`C_i = U_i^{-1}\Sigma_iU_i^{-\top}`. Head :math:`h` (width :math:`d`) has a query map :math:`S_h`
(``raw_score_q[h]``, :math:`d\times K`), a sending map :math:`T_h` (``T[h]``) that is also its key (tied key and
content), and the isotropic channel noise :math:`N_h = n_hI_d`, :math:`n_h = e^{\ell_h}` (``channel_log_noise``).
The receiver's state :math:`X_i\sim N(\xi_i, C_i)` is conditioned on the records
:math:`o_h = S_hX_i - T_hX_{J^h} + \varepsilon_h = 0`, :math:`\varepsilon_h\sim N(0, N_h)`. The label :math:`J^h` has
the routing prior :math:`\pi^h_{ij}` (the attention log-prior, causal mask and ALiBi included), every source state
is an independent snapshot of its input belief (the self edge included), and each head hears its own snapshot
(the per-head snapshot convention, which coincides with the note's model when :math:`T_hC_jT_g^\top = 0`).

One head (index suppressed), with receiver query moments :math:`a_i = S\xi_i`, :math:`A_i = SC_iS^\top`, source
utterance moments :math:`b_j = T\xi_j`, :math:`B_j = TC_jT^\top + nI` and record covariance
:math:`W_{ij} = A_i + B_j\succ 0` (:func:`exact_record_posterior`, Proposition ``ma:prop:minimal`` (a)):

.. math::
    w_{ij} = \operatorname{softmax}_j\bigl(\log\pi_{ij} + \log N(b_j; a_i, W_{ij})\bigr),\qquad
    r_{ij} = W_{ij}^{-1}(b_j - a_i),\qquad u_i = \sum_jw_{ij}r_{ij},

.. math::
    \bar\Lambda_i = \sum_jw_{ij}W_{ij}^{-1},\qquad R_i = \sum_jw_{ij}r_{ij}r_{ij}^\top - u_iu_i^\top,\qquad
    \mathsf K_i = C_iS^\top,

.. math::
    \xi_i^+ = \xi_i + \mathsf K_iu_i,\qquad C_i^+ = C_i - \mathsf K_i(\bar\Lambda_i - R_i)\mathsf K_i^\top.

These are the moments of the exact posterior mixture :math:`\sum_jw_{ij}N(m_{ij}, P_{ij})` with
:math:`m_{ij} = \xi_i + G_{ij}(b_j - a_i)`, :math:`G_{ij} = C_iS^\top W_{ij}^{-1}`, :math:`P_{ij} = C_i - G_{ij}SC_i`:
:math:`m_{ij} - \xi_i = \mathsf K_ir_{ij}` and :math:`P_{ij} = C_i - \mathsf K_iW_{ij}^{-1}\mathsf K_i^\top`, so the law of
total covariance gives :math:`C_i^+`, a mixture covariance (positive definite, since every
:math:`P_{ij}^{-1} = C_i^{-1} + S^\top B_j^{-1}S` is). The receiving map :math:`G_{ij}`, the residual and the routing
are derived; the residual is precision weighted, :math:`I - G_{ij}S = P_{ij}C_i^{-1}`, and no residual is added.

Heads by assumed-density filtering (:func:`sequential_head_assembly`). The exact joint posterior over the label
tuple has :math:`N^H` components. Head :math:`h` instead applies the one-head posterior to the Gaussian
:math:`(\xi_i^{(h-1)}, C_i^{(h-1)})` left by heads :math:`1, \dots, h-1` (the sources stay their input snapshots) and
collapses the mixture to its first two moments,

.. math::
    \xi^{(h)} = \xi^{(h-1)} + \mathsf K^hu^h,\qquad C^{(h)} = C^{(h-1)} - \mathsf K^hM^h\mathsf K^{h\top},\qquad
    \mathsf K^h = C^{(h-1)}S_h^\top,\quad M^h = \bar\Lambda^h - R^h,

with :math:`(\xi^{(0)}, C^{(0)}) = (\xi_i, C_i)`. :math:`C^{(h-1)}` is never formed: with the :math:`d\times d` cross
blocks :math:`Y_{hg} = S_h\mathsf K^g`,

.. math::
    \mathsf K^h = CS_h^\top - \sum_{g<h}\mathsf K^gM^gY_{hg}^\top,\qquad
    A^h = S_hCS_h^\top - \sum_{g<h}Y_{hg}M^gY_{hg}^\top.

Exactness. If :math:`S_hCS_g^\top = 0` for :math:`h\ne g`, then :math:`S_hC^{(g)} = S_hC` for every :math:`g < h`, by
induction on :math:`g` (:math:`S_h\mathsf K^g = S_hC^{(g-1)}S_g^\top = S_hCS_g^\top = 0`). Head :math:`h` therefore sees
the input moments :math:`a^h = S_h\xi`, :math:`A^h = S_hCS_h^\top`, its label posterior is the item (a) posterior at
the input, :math:`\mathsf K^h = CS_h^\top`, and
:math:`\xi^{(H)} = \xi + \sum_h\sum_jw^h_{ij}G^h_{ij}(T_h\xi_j - S_h\xi)`, :math:`C^{(H)} = C - \sum_hCS_h^\top M^hS_hC`.
These are the mean and the moment-matched covariance of Proposition ``ma:prop:minimal`` (b) in the per-head
snapshot model: the record covariance of a label tuple is block diagonal, so the labels factor, and the head terms
of the conditional means are functions of independent labels, so their covariances add. Without the condition ADF
is a declared, order-dependent approximation; every step is an exact mixture moment match, so it always returns a
valid Gaussian.

Mean field (Proposition ``ma:prop:mean_field``, the first structured-mean-field step started at the receiver's
prior; :func:`mean_field_record_posterior`). Every head's label update is evaluated at the INPUT receiver
moments, :math:`\beta^h_{ij} = \operatorname{softmax}_j(\log\pi^h_{ij} - \mathrm{KL}(N(a_i^h, A_i^h)\,\|\,N(b_j^h, B_j^h)))`
at temperature 1, and the power posterior at these labels fuses the heads in information form,

.. math::
    P_i^{-1} = C_i^{-1} + \sum_hS_h^\top\Lambda_i^hS_h,\qquad
    \xi_i^+ = P_i\Bigl(C_i^{-1}\xi_i + \sum_hS_h^\top\eta_i^h\Bigr),\qquad C_i^+ = P_i,

with :math:`\Lambda_i^h = \sum_j\beta^h_{ij}B_j^{h\,-1}` and :math:`\eta_i^h = \sum_j\beta^h_{ij}B_j^{h\,-1}b_j^h`. The
fusion runs through the same assembly (:func:`information_step`): head :math:`h` is the Gaussian pseudo-observation
:math:`y^h = \Lambda^{h\,-1}\eta^h` of :math:`S_hX_i` with covariance :math:`\Lambda^{h\,-1}`, whose Kalman step adds exactly
:math:`S_h^\top\Lambda^hS_h` to the precision and :math:`S_h^\top\eta^h` to the information vector, so the sequential
result is :math:`(\xi_i^+, P_i)` exactly, in any head order. The mean-field covariance omits the between-source
spread :math:`G\operatorname{Cov}_w(T\xi)G^\top` that the exact posterior keeps.

Float32. The query covariance of a later head is a downdate,
:math:`A^h = S_hCS_h^\top - \sum_{g<h}Y_{hg}M^gY_{hg}^\top`. Its float32 error grows with
:math:`\|S_hCS_h^\top\|` and with the float32 error of the earlier heads' inverses :math:`W_{ij}^{-1}` (about
:math:`\kappa(W_{ij})\varepsilon`). With coupled heads and an ill-conditioned :math:`C_i` the error can cancel the
smallest eigenvalues of :math:`A^h` and leave it indefinite, although the exact :math:`A^h` is positive definite.
:func:`minimal_agent_readout` therefore factors every :math:`A^h` (zero jitter) and reports a failure together with
its other factorizations. Under ``projected_cov_condition_recovery`` it also flags any float32 receiver whose
:math:`A^h` fails the certificate :math:`\varepsilon\|S_hC_iS_h^\top\|_F\|(A_i^h)^{-1}\|_F > \mathrm{tol}`, with
:math:`\mathrm{tol}` = ``projected_gram_condition_tol``. For the first head this is the kernel's Gram-route
certificate of the projected query covariance. It is a screen: it does not bound the error contributed by the
earlier inverses. :meth:`RelationalOperators.minimal_agent_update` recomputes a flagged
float32 layer call in float64 (the whole call, from the local beliefs) and raises only if the float64 call fails
too.

Comparison-space RoPE (``rope_space='comparison'``). With the per-position rotation :math:`R_i` of
:func:`vfe3.geometry.rope.build_rope_rotation` (orthogonal, :math:`d\times d`), head :math:`h`'s record is
:math:`o_h = R_iS_hX_i - R_jT_hX_{J^h} + \varepsilon_h`; the noise :math:`n_hI_d` is rotation invariant, so this is the
record :math:`S_hX_i - R_{j-i}T_hX_{J^h} + \varepsilon'_h` of a source whose utterance reaches receiver :math:`i` through the
known channel :math:`R_{j-i} = R_i^\top R_j`. Every cost (exact and mean field) is computed on the rotated moments
:math:`(R_ia_i, R_iA_iR_i^\top)` and :math:`(R_jb_j, R_jB_jR_j^\top)`, so the labels depend on position only through
:math:`j - i`, and the receiver statistics are rotated back into the receiver's record frame,
:math:`u_i \to R_i^\top u_i`, :math:`\bar\Lambda_i, R_i^{\rm spread}, \Lambda_i \to R_i^\top(\cdot)R_i`,
:math:`\eta_i \to R_i^\top\eta_i`. This is the exact posterior of the rotated record: the gain
:math:`\mathsf K_i = C_iS^\top` is unchanged and the received content arrives through :math:`R_{j-i}` and is decoded by it,
so routing and content stay in one frame (no ``rope_on_value`` split). :math:`R = I` is the unrotated model.

Pure functions with shape comments; :class:`vfe3.model.relational_operators.RelationalOperators` owns the parameters
(``raw_score_q``, ``T``, ``channel_log_noise``) and calls these. Factorizations use zero jitter and no covariance
repair, and a failed factorization is never repaired: it raises ``ValueError`` or, with ``check=False``, is
reported to the caller.
"""

import math
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import torch
from torch.utils.checkpoint import checkpoint

from vfe3.relational_scoring import (
    _compute_dtype, projected_condition_recovery, projected_gaussian_kl, projected_gram_condition_tol,
)
from vfe3.runtime import ieee_fp32_matmul

PREDICTIVE_READOUTS: Tuple[str, ...] = ("kernel", "minimal_agent")
MINIMAL_AGENT_INFERENCE: Tuple[str, ...] = ("exact", "mean_field")
# Bytes of pair matrices one exact-routing tile may hold: W_ij, its Cholesky factor and W_ij^{-1}, r x r each.
# Read at call time, so a test can force tiny tiles by setting it.
PAIR_TILE_BYTES: int = 32 * 1024 ** 2
# Per-layer, per-head diagnostics written to the relational capture (means over receivers).
MINIMAL_AGENT_DIAGNOSTIC_KEYS: Tuple[str, ...] = (
    "relational_minimal_agent_gain_fraction", "relational_minimal_agent_channel_noise",
    "relational_minimal_agent_head_coupling", "relational_minimal_agent_label_staleness",
    "relational_minimal_agent_covariance_contraction", "relational_minimal_agent_spread_share")
# Diagnostics defined in one inference mode only (NaN in the other).
MINIMAL_AGENT_MODE_ONLY_KEYS: Dict[str, str] = {
    "relational_minimal_agent_label_staleness": "mean_field",
    "relational_minimal_agent_spread_share": "exact",
}


class RecordPosterior(NamedTuple):
    r"""Exact one-head source posterior and its gain statistics (:func:`exact_record_posterior`)."""

    weights:    torch.Tensor   # (..., I, J) w_ij, rows sum to one, exactly 0 on masked pairs
    cost:       torch.Tensor   # (..., I, J) -log N(b_j; a_i, W_ij) - (r/2) log 2 pi; 0 in a skipped masked tile
    u:          torch.Tensor   # (..., I, r) u_i = sum_j w_ij W_ij^{-1}(b_j - a_i)
    Lambda_bar: torch.Tensor   # (..., I, r, r) sum_j w_ij W_ij^{-1}
    R:          torch.Tensor   # (..., I, r, r) between-source spread of r_ij
    failed:     torch.Tensor   # () bool: a zero-jitter factorization failed or a cost is not finite


class MeanFieldPosterior(NamedTuple):
    r"""Mean-field label step and its information statistics (:func:`mean_field_record_posterior`)."""

    weights: torch.Tensor      # (..., I, J) beta_ij = softmax_j(log pi_ij - KL_ij)
    cost:    torch.Tensor      # (..., I, J) KL(N(a_i, A_i) || N(b_j, B_j))
    Lambda:  torch.Tensor      # (..., I, r, r) sum_j beta_ij B_j^{-1}
    eta:     torch.Tensor      # (..., I, r) sum_j beta_ij B_j^{-1} b_j
    failed:  torch.Tensor      # () bool


class MinimalAgentResult(NamedTuple):
    r"""Posterior reference moments of every receiver, in the factored form of the head assembly.

    :math:`C_i^+ = C_i - \sum_h\mathsf K_i^hM_i^h\mathsf K_i^{h\top}` (``gains``, ``M``). The two covariances at the
    input moments are always returned; ``gain_fraction``, ``head_coupling``, ``covariance_contraction``,
    ``label_staleness`` (mean field) and ``spread_share`` (exact: the share :math:`\operatorname{tr}(AR)/\operatorname{tr}(A\bar\Lambda)`
    of the removed probed variance that the between-source spread gives back, the part the mean field drops) only
    with ``diagnostics=True``.
    ``failed`` reports a failed
    zero-jitter factorization or a non-finite cost anywhere in the call; ``ill_conditioned`` reports a float32 ADF
    query covariance that failed the certificate (only under ``projected_cov_condition_recovery``, else False).
    """

    xi:               torch.Tensor             # (..., N, K) xi_i^+
    gains:            torch.Tensor             # (..., N, H, K, d) Kmat_i^h = C_i^{(h-1)} S_h^T
    M:                torch.Tensor             # (..., N, H, d, d) M_i^h
    weights:          torch.Tensor             # (..., H, N, N) labels actually used (w^h or beta^h)
    cost:             torch.Tensor             # (..., H, N, N) the per-pair costs those labels read (exact: 0 in a skipped masked tile)
    gain_fraction:    Optional[torch.Tensor]   # (..., N, H) in [0, 1): exact at the ADF moments, mean field at the input
    query_covariance: torch.Tensor             # (..., N, H, d, d) S_h C_i S_h^T at the input moments
    key_covariance:   torch.Tensor             # (..., N, H, d, d) T_h C_j T_h^T (noise-free key law)
    head_coupling:    Optional[torch.Tensor]   # (..., N, H) max_{g != h} normalized ||S_h C_i S_g^T||_F
    failed:           torch.Tensor             # () bool: a factorization failed or a cost is not finite
    ill_conditioned:  torch.Tensor             # () bool: a float32 A_i^h failed the certificate (no failure)
    covariance_contraction: Optional[torch.Tensor] = None   # (..., N) tr C_i^+ / tr C_i (diagnostics only)
    label_staleness:  Optional[torch.Tensor] = None         # (..., N, H) mean field: TV of one relabeling (diagnostics)
    spread_share:     Optional[torch.Tensor] = None         # (..., N, H) exact: tr(A R) / tr(A Lambda_bar) (diagnostics)


def _symmetrize(X: torch.Tensor) -> torch.Tensor:          # (..., r, r) -> (..., r, r)
    return 0.5 * (X + X.transpose(-1, -2))


def _raise_on_failure(flags: List[torch.Tensor], what: str) -> None:
    r"""One host check for every factorization of a call; a failure raises, nothing is repaired."""
    if flags and bool(torch.stack([flag.reshape(()) for flag in flags]).any()):
        raise ValueError(f"{what} is not positive definite or not finite: zero-jitter Cholesky failed; "
                         "no repair applied")


def readout_failure_message(dtype: torch.dtype) -> str:
    r"""The error of a failed :func:`minimal_agent_readout` call in the compute dtype ``dtype``."""
    message = ("a minimal-agent covariance is not positive definite or not finite: the ADF query covariance "
               "A^h = S_h C^(h-1) S_h^T, a record covariance W_ij = A_i + B_j, an utterance covariance B_j, a label "
               "precision Lambda_i or I + L^T A L (zero-jitter Cholesky failed; no repair applied)")
    if dtype != torch.float64:
        message += ("; in float32 the downdate A^h = S_h C S_h^T - sum_{g<h} Y_hg M^g Y_hg^T can lose positive "
                    "definiteness by cancellation when heads are coupled and C_i is ill conditioned "
                    "(RelationalOperators.minimal_agent_update recomputes such a layer call in float64)")
    return message


# ----------------------------------------------------------------------------------------------
# One head: exact source posterior (Proposition ma:prop:minimal (a))
# ----------------------------------------------------------------------------------------------

def _exact_tile(
    a:         torch.Tensor,   # (..., Iq, r) receiver query means
    A:         torch.Tensor,   # (..., Iq, r, r) receiver query covariances
    b:         torch.Tensor,   # (..., Jk, r) source utterance means
    B:         torch.Tensor,   # (..., Jk, r, r) noisy utterance covariances
    log_prior: torch.Tensor,   # (..., Iq, Jk) log pi_ij (-inf masks a pair)
) -> Tuple[torch.Tensor, ...]:
    r"""One pair tile of :func:`exact_record_posterior`: costs and the tile's statistics at its own row maximum.

    With :math:`L_{ij}L_{ij}^\top = W_{ij}` and :math:`z_{ij} = L_{ij}^{-1}(b_j - a_i)`,

    .. math::
        c_{ij} = \tfrac12\|z_{ij}\|^2 + \sum_k\log(L_{ij})_{kk} = -\log N(b_j; a_i, W_{ij}) - \tfrac r2\log 2\pi,
        \qquad \ell_{ij} = \log\pi_{ij} - c_{ij},

    and with the tile's row maximum :math:`t_i = \max_j\ell_{ij}` (detached) and :math:`e_{ij} = e^{\ell_{ij} - t_i}`
    the tile's weight :math:`s_i = \sum_je_{ij}`, weighted mean :math:`\bar r_i = \sum_je_{ij}r_{ij}/s_i`, information
    sum :math:`\sum_je_{ij}W_{ij}^{-1}` and centered spread
    :math:`M_{2,i} = \sum_je_{ij}(r_{ij} - \bar r_i)(r_{ij} - \bar r_i)^\top` (two-pass, so a common offset of the
    :math:`r_{ij}` does not cancel in finite precision). The caller combines tiles by an online log-sum-exp with Chan's
    pairwise update outside any checkpoint, so a checkpointed tile saves only its input slices. A masked pair
    (:math:`\log\pi_{ij} = -\infty`) contributes exactly 0; its :math:`W_{ij}` is still positive definite. A row with no
    live pair has :math:`s_i = 0` and mean 0. Returns ``(cost, t, s, mean, Lam, M2, failed)``.
    """
    delta = b.unsqueeze(-3) - a.unsqueeze(-2)                                   # (..., Iq, Jk, r) b_j - a_i
    W = A.unsqueeze(-3) + B.unsqueeze(-4)                                       # (..., Iq, Jk, r, r), exactly symmetric
    L, info = torch.linalg.cholesky_ex(W)
    z = torch.linalg.solve_triangular(L, delta.unsqueeze(-1), upper=False)       # (..., Iq, Jk, r, 1)
    r_ij = torch.linalg.solve_triangular(L.transpose(-1, -2), z, upper=True).squeeze(-1)   # W_ij^{-1}(b_j - a_i)
    W_inv = torch.cholesky_inverse(L)                                           # (..., Iq, Jk, r, r)
    cost = 0.5 * z.squeeze(-1).square().sum(dim=-1) + L.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
    logits = log_prior - cost                                                   # (..., Iq, Jk)
    t = logits.amax(dim=-1).detach()                                            # (..., Iq) tile row maximum
    shift = torch.where(torch.isfinite(t), t, torch.zeros_like(t))              # a fully masked tile row: 0
    e = torch.exp(logits - shift.unsqueeze(-1))                                 # (..., Iq, Jk), 0 on masked pairs
    s = e.sum(dim=-1)
    s_safe = torch.where(s > 0, s, torch.ones_like(s))                          # a fully masked row: mean 0
    mean = torch.einsum("...ij,...ijr->...ir", e, r_ij) / s_safe.unsqueeze(-1)  # (..., Iq, r)
    Lam = torch.einsum("...ij,...ijrs->...irs", e, W_inv)
    centered = r_ij - mean.unsqueeze(-2)                                        # (..., Iq, Jk, r)
    M2 = torch.einsum("...ij,...ijr,...ijs->...irs", e, centered, centered)
    failed = (info != 0).any() | ~torch.isfinite(cost).all()
    return cost, t, s, mean, Lam, M2, failed


def exact_record_posterior(
    a:         torch.Tensor,                    # (..., I, r) receiver query means a_i = S xi_i
    A:         torch.Tensor,                    # (..., I, r, r) receiver query covariances A_i = S C_i S^T
    b:         torch.Tensor,                    # (..., J, r) source utterance means b_j = T xi_j
    B:         torch.Tensor,                    # (..., J, r, r) noisy utterance covariances T C_j T^T + N
    log_prior: Optional[torch.Tensor] = None,   # broadcastable to (..., I, J); -inf masks a pair; None: uniform
    *,
    check:     bool = True,
) -> RecordPosterior:
    r"""Exact one-head source posterior :math:`w_{ij}` and the gain statistics :math:`(u_i, \bar\Lambda_i, R_i)`.

    .. math::
        w_{ij} = \operatorname{softmax}_j\bigl(\log\pi_{ij} - c_{ij}\bigr),\quad
        c_{ij} = \tfrac12(b_j - a_i)^\top W_{ij}^{-1}(b_j - a_i) + \tfrac12\log\det W_{ij},\quad W_{ij} = A_i + B_j,

    at temperature 1 with no cost clamp; :math:`c_{ij}` is the negative log expected likelihood up to the constant
    :math:`\tfrac r2\log 2\pi` (``projected_noisy_query_cost`` subtracts the further row constant
    :math:`\tfrac12\log\det A_i`, which the softmax ignores and which is not formed here, so this function needs no
    factor of :math:`A_i`; :func:`minimal_agent_readout` checks the ADF query covariances separately). Then
    :math:`r_{ij} = W_{ij}^{-1}(b_j - a_i)`,
    :math:`u_i = \sum_jw_{ij}r_{ij}`, :math:`\bar\Lambda_i = \sum_jw_{ij}W_{ij}^{-1}` and
    :math:`R_i = \sum_jw_{ij}r_{ij}r_{ij}^\top - u_iu_i^\top`.

    Every pair needs its own :math:`r\times r` Cholesky factor of :math:`W_{ij}`. One pass over bounded pair tiles
    (:data:`PAIR_TILE_BYTES` of :math:`W_{ij}`, its factor and :math:`W_{ij}^{-1}`), with an online log-sum-exp over the
    source tiles (:func:`_exact_tile`), factors each pair once. A tile whose every pair the prior masks in every
    batch element (above the causal diagonal) contributes exactly 0 to every sum, so it is skipped: it is not
    factored and its entries of ``cost`` read 0 (their weights are exactly 0 either way). Each tile is a
    non-reentrant checkpoint when gradients are on and the grid has more than one tile, and a grid of more than
    one tile makes one host read of the tile mask. Autocast is disabled, the compute dtype is
    ``_compute_dtype`` of the inputs, factorizations use zero jitter and no repair. ``check`` makes the one host
    check for factorization failure and raises ``ValueError``; ``check=False`` returns the flag in ``failed`` for the
    caller to fold into its own single check. Every row needs at least one finite prior entry.
    """
    if a.dim() < 2 or b.dim() < 2 or a.shape[-1] != b.shape[-1]:
        raise ValueError("exact_record_posterior needs means (..., I, r) and (..., J, r) of one comparison rank")
    r = a.shape[-1]
    if A.shape != (*a.shape, r) or B.shape != (*b.shape, r):
        raise ValueError("exact_record_posterior needs covariances (..., I, r, r) and (..., J, r, r)")
    dtype = _compute_dtype(a, A, b, B)
    batch = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    I, J = a.shape[-2], b.shape[-2]
    with torch.autocast(device_type=a.device.type, enabled=False):
        a, A, b, B = a.to(dtype), A.to(dtype), b.to(dtype), B.to(dtype)
        lp = (torch.zeros(I, J, dtype=dtype, device=a.device) if log_prior is None else log_prior.to(dtype))
        lp = torch.broadcast_to(lp, (*batch, I, J))
        element = torch.finfo(dtype).bits // 8
        budget = max(1, PAIR_TILE_BYTES // (element * max(1, math.prod(batch)) * 3 * r * r))
        q_step = max(1, min(I, 32, budget))
        k_step = max(1, min(J, budget // q_step))
        tiled = (I > q_step or J > k_step) and torch.is_grad_enabled() and any(
            x.requires_grad for x in (a, A, b, B, lp))
        n_q, n_k = -(-I // q_step), -(-J // k_step)
        if n_q * n_k > 1:
            # A tile the prior masks in every batch element (the causal upper triangle) adds exactly 0 to every sum
            # and is not factored; its costs read 0. One host read of the tile mask.
            grid = torch.zeros(n_q * q_step, n_k * k_step, dtype=torch.bool, device=a.device)
            grid[:I, :J] = torch.isfinite(lp).reshape(-1, I, J).any(dim=0)
            live = grid.view(n_q, q_step, n_k, k_step).any(dim=3).any(dim=1).tolist()   # (n_q, n_k) bool
        else:
            live = [[True]]
        rows, U_rows, Lam_rows, Q_rows, s_rows, flags = [], [], [], [], [], []
        for i in range(0, I, q_step):
            n_i = min(q_step, I - i)
            # Online log-sum-exp over the source tiles with Chan's pairwise update. At the running row maximum m_i
            # the accumulators hold the weight s = sum_j exp(l_ij - m_i), the weighted mean U of r_ij, the weighted
            # sum Lam of W^-1, and the CENTERED spread Q = sum_j exp(l_ij - m_i)(r_ij - U)(r_ij - U)^T, so R never
            # forms as a difference of large second moments (audit 2026-10-07, F01/F58). The scales are detached
            # (the moments do not depend on m).
            m = torch.full((*batch, n_i), -math.inf, dtype=dtype, device=a.device)
            s = torch.zeros((*batch, n_i), dtype=dtype, device=a.device)
            U = torch.zeros((*batch, n_i, r), dtype=dtype, device=a.device)
            Lam = torch.zeros((*batch, n_i, r, r), dtype=dtype, device=a.device)
            Q = torch.zeros((*batch, n_i, r, r), dtype=dtype, device=a.device)
            columns = []
            for j in range(0, J, k_step):
                if not live[i // q_step][j // k_step]:
                    columns.append(torch.zeros((*batch, n_i, min(k_step, J - j)), dtype=dtype, device=a.device))
                    continue
                args = (a[..., i:i + q_step, :], A[..., i:i + q_step, :, :], b[..., j:j + k_step, :],
                        B[..., j:j + k_step, :, :], lp[..., i:i + q_step, j:j + k_step])
                if tiled:
                    out = checkpoint(_exact_tile, *args, use_reentrant=False, preserve_rng_state=False)
                else:
                    out = _exact_tile(*args)
                cost, t, s_t, U_t, Lam_t, Q_t, failed = out                   # U_t tile mean, Q_t centered spread
                m_new = torch.maximum(m, t)
                shift = torch.where(torch.isfinite(m_new), m_new, torch.zeros_like(m_new))
                old, new = torch.exp(m - shift), torch.exp(t - shift)          # (..., n_i), 0 where -inf
                S_a, S_b = old * s, new * s_t
                S = S_a + S_b
                S_safe = torch.where(S > 0, S, torch.ones_like(S))
                delta = U_t - U                                                # (..., n_i, r)
                U = U + delta * (S_b / S_safe).unsqueeze(-1)
                Lam = old[..., None, None] * Lam + new[..., None, None] * Lam_t
                Q = (old[..., None, None] * Q + new[..., None, None] * Q_t
                     + (S_a * S_b / S_safe)[..., None, None] * (delta.unsqueeze(-1) * delta.unsqueeze(-2)))
                s = S
                m = m_new
                columns.append(cost)
                flags.append(failed)
            rows.append(torch.cat(columns, dim=-1))
            s_rows.append(s)
            U_rows.append(U)
            Lam_rows.append(Lam)
            Q_rows.append(Q)
        cost = torch.cat(rows, dim=-2)                                          # (..., I, J)
        s = torch.cat(s_rows, dim=-1)                                           # (..., I)
        u = torch.cat(U_rows, dim=-2)                                           # (..., I, r) sum_j w_ij r_ij
        Lambda_bar = _symmetrize(torch.cat(Lam_rows, dim=-3) / s[..., None, None])
        R = _symmetrize(torch.cat(Q_rows, dim=-3) / s[..., None, None])         # sum_j w_ij (r_ij - u_i)(r_ij - u_i)^T
        weights = torch.softmax(lp - cost, dim=-1)                              # (..., I, J)
        failed = torch.stack(flags).any()
    if check:
        _raise_on_failure([failed], "minimal-agent record covariance W_ij = A_i + B_j")
    return RecordPosterior(weights, cost, u, Lambda_bar, R, failed)


# ----------------------------------------------------------------------------------------------
# One head: mean-field label step and information fusion (Proposition ma:prop:mean_field)
# ----------------------------------------------------------------------------------------------

def label_statistics(
    weights: torch.Tensor,                      # (..., I, J) labels (rows sum to one)
    b:       torch.Tensor,                      # (..., J, r) utterance means
    B:       torch.Tensor,                      # (..., J, r, r) noisy utterance covariances
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:   # Lambda (..., I, r, r), eta (..., I, r), failed ()
    r"""Information statistics of the power posterior at fixed labels.

    .. math::
        \Lambda_i = \sum_j\beta_{ij}B_j^{-1},\qquad \eta_i = \sum_j\beta_{ij}B_j^{-1}b_j,

    from one zero-jitter Cholesky factor per source (no per-pair inverse).
    """
    L, info = torch.linalg.cholesky_ex(B)                                       # (..., J, r, r)
    B_inv = torch.cholesky_inverse(L)                                           # (..., J, r, r)
    B_inv_b = torch.cholesky_solve(b.unsqueeze(-1), L).squeeze(-1)              # (..., J, r)
    Lambda = _symmetrize(torch.einsum("...ij,...jrs->...irs", weights, B_inv))
    eta = torch.einsum("...ij,...jr->...ir", weights, B_inv_b)
    return Lambda, eta, (info != 0).any()


def mean_field_record_posterior(
    a:         torch.Tensor,                    # (..., I, r) receiver query means at the input moments
    A:         torch.Tensor,                    # (..., I, r, r) receiver query covariances at the input moments
    b:         torch.Tensor,                    # (..., J, r) utterance means
    B:         torch.Tensor,                    # (..., J, r, r) noisy utterance covariances
    log_prior: Optional[torch.Tensor] = None,   # broadcastable to (..., I, J); None: uniform
    *,
    check:     bool = True,
) -> MeanFieldPosterior:
    r"""Mean-field label step at the receiver's prior and the label-averaged channel information.

    .. math::
        \beta_{ij} = \operatorname{softmax}_j\bigl(\log\pi_{ij} - \mathrm{KL}(N(a_i, A_i)\,\|\,N(b_j, B_j))\bigr),\qquad
        \mathrm{KL} = \tfrac12\bigl[\operatorname{tr}(B_j^{-1}A_i) + \delta_{ij}^\top B_j^{-1}\delta_{ij} - r
            + \log\det B_j - \log\det A_i\bigr],

    :math:`\delta_{ij} = a_i - b_j`, at temperature 1 with no clamp (Proposition ``ma:prop:mean_field`` (i); for
    :math:`n = 0` it is Gibbs routing with :math:`S_K = T`). The KL is
    :func:`vfe3.relational_scoring.projected_gaussian_kl` with the noisy key covariance :math:`B_j` (separable: one
    factor per receiver and per sender, no per-pair inverse; it raises if :math:`A_i` or :math:`B_j` is not
    positive definite). :func:`label_statistics` gives :math:`(\Lambda_i, \eta_i)`.
    """
    if A.shape != (*a.shape, a.shape[-1]) or B.shape != (*b.shape, b.shape[-1]):
        raise ValueError("mean_field_record_posterior needs covariances (..., I, r, r) and (..., J, r, r)")
    dtype = _compute_dtype(a, A, b, B)
    I, J = a.shape[-2], b.shape[-2]
    with torch.autocast(device_type=a.device.type, enabled=False):
        a, A, b, B = a.to(dtype), A.to(dtype), b.to(dtype), B.to(dtype)
        cost = projected_gaussian_kl(a.unsqueeze(-2), A.unsqueeze(-3),
                                     b.unsqueeze(-2), B.unsqueeze(-3)).squeeze(-3)   # (..., I, J)
        lp = (torch.zeros(I, J, dtype=dtype, device=a.device) if log_prior is None else log_prior.to(dtype))
        weights = torch.softmax(lp - cost, dim=-1)                              # (..., I, J)
        Lambda, eta, failed = label_statistics(weights, b, B)
    if check:
        _raise_on_failure([failed], "minimal-agent utterance covariance B_j")
    return MeanFieldPosterior(weights, cost, Lambda, eta, failed)


def information_step(
    a:      torch.Tensor,                       # (..., I, r) query mean of the current Gaussian, S xi
    A:      torch.Tensor,                       # (..., I, r, r) query covariance of the current Gaussian, S C S^T
    Lambda: torch.Tensor,                       # (..., I, r, r) label-averaged channel precision
    eta:    torch.Tensor,                       # (..., I, r) label-averaged information vector
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:   # u (..., I, r), M (..., I, r, r), failed ()
    r"""Kalman step of the pseudo-observation :math:`y = \Lambda^{-1}\eta` of :math:`SX` with covariance :math:`\Lambda^{-1}`.

    .. math::
        M = (A + \Lambda^{-1})^{-1} = LE^{-1}L^\top,\qquad u = M(\Lambda^{-1}\eta - a) = LE^{-1}(L^{-1}\eta - L^\top a),
        \qquad \Lambda = LL^\top,\quad E = I + L^\top AL,

    so that with :math:`\mathsf K = CS^\top` the update :math:`\xi + \mathsf Ku`, :math:`C - \mathsf KM\mathsf K^\top` multiplies the
    current Gaussian by :math:`\exp(-\tfrac12x^\top S^\top\Lambda Sx + x^\top S^\top\eta)`: the precision gains exactly
    :math:`S^\top\Lambda S` and the information vector :math:`S^\top\eta`. :math:`E` has eigenvalues at least 1, so no
    inverse of :math:`\Lambda` is formed.
    """
    r = a.shape[-1]
    L, info_L = torch.linalg.cholesky_ex(Lambda)                                # (..., I, r, r)
    E = _symmetrize(torch.eye(r, dtype=A.dtype, device=A.device) + L.transpose(-1, -2) @ A @ L)
    L_E, info_E = torch.linalg.cholesky_ex(E)
    v = (torch.linalg.solve_triangular(L, eta.unsqueeze(-1), upper=False)
         - L.transpose(-1, -2) @ a.unsqueeze(-1))                               # (..., I, r, 1)
    u = (L @ torch.cholesky_solve(v, L_E)).squeeze(-1)                          # (..., I, r)
    M = _symmetrize(L @ torch.cholesky_inverse(L_E) @ L.transpose(-1, -2))
    return u, M, (info_L != 0).any() | (info_E != 0).any()


# ----------------------------------------------------------------------------------------------
# Heads: the sequential assembly (ADF for the exact path, exact information fusion for the mean field)
# ----------------------------------------------------------------------------------------------

def gain_basis(
    C:      torch.Tensor,                       # (..., N, K, K) dense or (..., N, G, g, g) block-diagonal C_i
    S:      torch.Tensor,                       # (H, d, K) query maps
    blocks: bool,                               # C holds the diagonal blocks of a block-diagonal C_i
) -> torch.Tensor:                              # (..., N, H, K, d) C_i S_h^T
    r"""Input gain bases :math:`C_iS_h^\top` of every head; block-diagonal :math:`C_i = \bigoplus_mC_{i,m}` gives row block
    :math:`m` as :math:`C_{i,m}S_h^{(m)\top}` (:math:`S_h^{(m)}` the :math:`d\times g` column block), no :math:`K\times K` matrix."""
    H, d, K = S.shape
    if not blocks:
        if C.shape[-2:] != (K, K):
            raise ValueError(f"dense covariances {tuple(C.shape[-2:])} do not match the map width {K}")
        return torch.einsum("...nkm,hsm->...nhks", C, S)                        # C symmetric: C S_h^T
    G, g = C.shape[-3], C.shape[-1]
    if G * g != K or C.shape[-2] != g:
        raise ValueError(f"block covariances {tuple(C.shape[-3:])} do not tile the map width {K}")
    X = torch.einsum("...nmab,hsmb->...nhmas", C, S.reshape(H, d, G, g))       # (..., N, H, G, g, d)
    return X.reshape(*X.shape[:-3], K, d)


def sequential_head_assembly(
    xi:          torch.Tensor,                  # (..., N, K) receiver reference means
    X0:          torch.Tensor,                  # (..., N, H, K, d) input gain bases C_i S_h^T (gain_basis)
    S:           torch.Tensor,                  # (H, d, K) query maps
    head_update: Callable[[int, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Condition the receivers on the heads in index order, collapsing to two moments after each head.

    ``head_update(h, a, A)`` receives the query moments :math:`a^h = S_h\xi^{(h-1)}` ``(..., N, d)`` and
    :math:`A^h = S_hC^{(h-1)}S_h^\top` ``(..., N, d, d)`` of the Gaussian left by heads :math:`1..h-1` and returns
    :math:`(u^h, M^h)`; then :math:`\xi^{(h)} = \xi^{(h-1)} + \mathsf K^hu^h` and :math:`C^{(h)} = C^{(h-1)} - \mathsf K^hM^h\mathsf K^{h\top}`
    with :math:`\mathsf K^h = C^{(h-1)}S_h^\top = CS_h^\top - \sum_{g<h}\mathsf K^gM^gY_{hg}^\top` and
    :math:`A^h = S_hCS_h^\top - \sum_{g<h}Y_{hg}M^gY_{hg}^\top`, :math:`Y_{hg} = S_h\mathsf K^g` (module docstring). Only
    :math:`K\times d` and :math:`d\times d` matrices are formed. Returns
    ``(xi_plus (..., N, K), gains (..., N, H, K, d), M (..., N, H, d, d), A (..., N, H, d, d))``.
    """
    H = S.shape[0]
    gains: List[torch.Tensor] = []
    Ms: List[torch.Tensor] = []
    As: List[torch.Tensor] = []
    state = xi
    for h in range(H):
        S_h = S[h]                                                              # (d, K)
        Kmat = X0[..., h, :, :]                                                 # (..., N, K, d) C S_h^T
        A = S_h @ Kmat                                                          # (..., N, d, d) S_h C S_h^T
        for g in range(h):
            Y = S_h @ gains[g]                                                  # (..., N, d, d) S_h Kmat^g
            YM = Y @ Ms[g]                                                      # (..., N, d, d)
            Kmat = Kmat - gains[g] @ YM.transpose(-1, -2)                       # - Kmat^g M^g Y_hg^T
            A = A - YM @ Y.transpose(-1, -2)
        A = _symmetrize(A)
        a = (S_h @ state.unsqueeze(-1)).squeeze(-1)                             # (..., N, d)
        u, M = head_update(h, a, A)
        state = state + (Kmat @ u.unsqueeze(-1)).squeeze(-1)                    # (..., N, K)
        gains.append(Kmat)
        Ms.append(M)
        As.append(A)
    return state, torch.stack(gains, dim=-3), torch.stack(Ms, dim=-3), torch.stack(As, dim=-3)


def _head_coupling(
    S:  torch.Tensor,                           # (H, d, K)
    X0: torch.Tensor,                           # (..., N, H, K, d) C_i S_h^T
) -> torch.Tensor:                              # (..., N, H)
    r""":math:`c_i^h = \max_{g\ne h}\|S_hC_iS_g^\top\|_F/(\|S_hC_iS_h^\top\|_F\|S_gC_iS_g^\top\|_F)^{1/2}` at the input
    moments; 0 means ADF is exact (module docstring), and a single head reports 0."""
    H = S.shape[0]
    if H == 1:
        return X0.new_zeros(X0.shape[:-2])
    cross = torch.einsum("hsk,...ngkt->...nhgst", S, X0)                        # (..., N, H, H, d, d) S_h C S_g^T
    norms = cross.flatten(-2).norm(dim=-1)                                      # (..., N, H, H)
    own = torch.diagonal(norms, dim1=-2, dim2=-1)                               # (..., N, H)
    ratio = norms / torch.sqrt(own.unsqueeze(-1) * own.unsqueeze(-2))
    diagonal = torch.eye(H, dtype=torch.bool, device=S.device)
    return ratio.masked_fill(diagonal, -math.inf).amax(dim=-1)


def _covariance_contraction(
    C:      torch.Tensor,                       # (..., N, K, K) dense or (..., N, G, g, g) blocks
    gains:  torch.Tensor,                       # (..., N, H, K, d)
    M:      torch.Tensor,                       # (..., N, H, d, d)
    blocks: bool,
) -> torch.Tensor:                              # (..., N)
    r""":math:`\operatorname{tr}C_i^+/\operatorname{tr}C_i = 1 - \sum_h\operatorname{tr}(\mathsf K_i^hM_i^h\mathsf K_i^{h\top})/\operatorname{tr}C_i`."""
    trace = torch.diagonal(C, dim1=-2, dim2=-1).sum(dim=(-2, -1) if blocks else -1)
    removed = torch.einsum("...nhks,...nhst,...nhkt->...n", gains, M, gains)
    return 1.0 - removed / trace


def _label_staleness(
    xi_plus: torch.Tensor,                      # (..., N, K) posterior reference means
    V_Q:     torch.Tensor,                      # (..., N, H, d, d) S_h C S_h^T at the input moments
    S:       torch.Tensor,                      # (H, d, K)
    gains:   torch.Tensor,                      # (..., N, H, K, d)
    M:       torch.Tensor,                      # (..., N, H, d, d)
    b:       torch.Tensor,                      # (..., H, N, d)
    B:       torch.Tensor,                      # (..., H, N, d, d)
    lp:      torch.Tensor,                      # (..., H, N, N)
    weights: torch.Tensor,                      # (..., H, N, N) labels used (input moments)
    rope:    Optional[torch.Tensor] = None,     # (N, d, d) comparison rotation R_i; b, B already rotated by R_j
) -> torch.Tensor:                              # (..., N, H)
    r"""Total variation :math:`\tfrac12\sum_j|\beta^+_{ij} - \beta_{ij}|` between the mean-field labels at the input
    moments and the same KL labels recomputed at the posterior query moments
    :math:`(S_h\xi_i^+, S_hC_i^+S_h^\top)`, with
    :math:`S_hC^+S_h^\top = S_hCS_h^\top - \sum_gY_{hg}M^gY_{hg}^\top`, :math:`Y_{hg} = S_h\mathsf K^g`: one further
    coordinate-ascent label step. Near 0 means a second mean-field step would not move the labels. Computed in
    float64; a receiver whose posterior query covariance does not factor or whose posterior mean is not finite reads
    NaN, and every receiver of a head with an utterance covariance that does not factor reads NaN. These inputs are
    replaced by placeholders before the KL, so a float32 attempt that failed (and that the caller recomputes in
    float64) never raises here.
    """
    f64 = torch.float64
    Y = torch.einsum("hsk,...ngkt->...nhgst", S.to(f64), gains.to(f64))        # (..., N, H, H, d, d) S_h K^g
    A_plus = _symmetrize(V_Q.to(f64) - torch.einsum("...nhgst,...ngtu,...nhgvu->...nhsv", Y, M.to(f64), Y))
    a_plus = torch.einsum("hsk,...nk->...nhs", S.to(f64), xi_plus.to(f64))     # (..., N, H, d)
    if rope is not None:
        R64 = rope.to(f64)                                                      # (N, d, d)
        a_plus = torch.einsum("nst,...nht->...nhs", R64, a_plus)
        A_plus = _symmetrize(torch.einsum("nst,...nhtu,nvu->...nhsv", R64, A_plus, R64))
    eye = torch.eye(A_plus.shape[-1], dtype=f64, device=A_plus.device)
    bad = ((torch.linalg.cholesky_ex(torch.nan_to_num(A_plus))[1] != 0)
           | ~torch.isfinite(A_plus).all(dim=(-2, -1)) | ~torch.isfinite(a_plus).all(dim=-1))   # (..., N, H)
    A_plus = torch.where(bad[..., None, None], eye, A_plus)                     # (..., N, H, d, d)
    a_plus = torch.where(bad[..., None], torch.zeros((), dtype=f64, device=a_plus.device), a_plus)
    B64 = B.to(f64)
    bad_B = torch.linalg.cholesky_ex(B64)[1] != 0                               # (..., H, N)
    B64 = torch.where(bad_B[..., None, None], eye, B64)
    cost = projected_gaussian_kl(a_plus, A_plus, b.to(f64).movedim(-3, -2),
                                 B64.movedim(-4, -3))                           # (..., H, N, N)
    relabeled = torch.softmax(lp.to(f64) - cost, dim=-1)
    tv = 0.5 * (relabeled - weights.to(f64)).abs().sum(dim=-1)                  # (..., H, N)
    tv = tv.masked_fill(bad_B.any(dim=-1, keepdim=True), math.nan)              # a failed utterance spoils its head
    return tv.movedim(-1, -2).masked_fill(bad, math.nan)                        # (..., N, H)


def minimal_agent_readout(
    xi:          torch.Tensor,                  # (..., N, K) reference means xi_i
    C:           torch.Tensor,                  # (..., N, K, K) dense or (..., N, G, g, g) block-diagonal C_i
    S:           torch.Tensor,                  # (H, d, K) query maps S_h
    T:           torch.Tensor,                  # (H, d, K) sending (and key) maps T_h
    log_noise:   torch.Tensor,                  # (H,) log n_h
    log_prior:   Optional[torch.Tensor] = None,  # (N, N), (H, N, N) or broadcastable to (..., H, N, N)
    inference:   str = "exact",
    *,
    diagnostics: bool = False,
    check:       bool = True,
    rope:        Optional[torch.Tensor] = None,  # (N, d, d) comparison-space rotation R_i, or None
) -> MinimalAgentResult:
    r"""Posterior reference moments of every receiver under the minimal agent model.

    Sources are the input snapshots, :math:`b_j^h = T_h\xi_j` and :math:`B_j^h = T_hC_jT_h^\top + n_hI_d` (input
    :math:`C_j`). ``inference="exact"`` runs ADF over the heads with :func:`exact_record_posterior` at each head's
    current moments, :math:`M^h = \bar\Lambda^h - R^h`; ``"mean_field"`` takes every head's labels from
    :func:`mean_field_record_posterior` at the input moments and fuses them with :func:`information_step`
    (module docstring).

    The gain fraction of the exact path is :math:`\operatorname{tr}(A_i^h\bar\Lambda_i^h)/d` at head :math:`h`'s ADF
    moments: the label-averaged share of the probed variance that the component updates remove, in
    :math:`A^h`-whitened coordinates, :math:`\in [0, 1)`; like ADF it depends on the head order. The mean field's is
    evaluated at the INPUT moments :math:`A_i^h = S_hC_iS_h^\top`, like its labels, so it does not depend on the
    head order, as the fused posterior does not:

    .. math::
        \tfrac1d\operatorname{tr}(A^hM_0^h) = 1 - \tfrac1d\operatorname{tr}(E^{-1}),\qquad
        M_0^h = (A^h + \Lambda^{h\,-1})^{-1} = LE^{-1}L^\top,\quad \Lambda^h = LL^\top,\quad E = I + L^\top A^hL,

    the share of the probed variance that head :math:`h`'s channel information alone would remove from the input
    Gaussian. It equals the exact fraction for one head with equal :math:`B_j` and :math:`\beta = w`. The gain
    fraction and the head coupling are diagnostics, computed only with ``diagnostics=True`` (else None).

    ``C`` with two more axes than ``xi`` is read as the diagonal blocks of a block-diagonal :math:`C_i` (diagonal
    beliefs on block frames); no :math:`K\times K` matrix is then formed. Every factorization of the call has zero
    jitter, the ADF query covariances :math:`A^h` included (module docstring, Float32), and their failure flags are
    folded into ``failed``. Under ``projected_cov_condition_recovery`` a float32 call also screens every :math:`A^h`
    with the certificate into ``ill_conditioned``. ``check=True`` makes one host check and raises ``ValueError`` on
    a failure (a certificate alone does not raise); ``check=False`` returns both flags for the caller's single
    check. The mean field's input covariances are factored here first, so ``projected_gaussian_kl`` (which keeps its
    own host transfer) never receives a failed one. Requires :math:`n_h > 0`, every :math:`S_h` of full row rank and
    :math:`C_i\succ 0`. ``rope`` applies the comparison-space rotation of the module docstring (labels depend on
    position only through :math:`j - i`; ``query_covariance`` and ``key_covariance`` stay unrotated).
    """
    if inference not in MINIMAL_AGENT_INFERENCE:
        raise ValueError(f"minimal_agent_inference must be one of {MINIMAL_AGENT_INFERENCE}, got {inference!r}")
    if S.shape != T.shape or S.dim() != 3 or log_noise.shape != (S.shape[0],):
        raise ValueError("minimal_agent_readout needs S and T of shape (H, d, K) and log_noise of shape (H,)")
    H, d, K = S.shape
    N = xi.shape[-2]
    if xi.shape[-1] != K:
        raise ValueError(f"reference means of width {xi.shape[-1]} do not match the map width {K}")
    if rope is not None and rope.shape != (N, d, d):
        raise ValueError(f"rope must be (N, d, d) = {(N, d, d)}, got {tuple(rope.shape)}")
    blocks = C.dim() == xi.dim() + 2                                            # (..., N, G, g, g) diagonal blocks
    dtype = _compute_dtype(xi, C, S, T, log_noise)
    certify = dtype == torch.float32 and projected_condition_recovery()          # the A^h certificate is opt-in
    with torch.autocast(device_type=xi.device.type, enabled=False), ieee_fp32_matmul():
        xi, C, S, T = xi.to(dtype), C.to(dtype), S.to(dtype), T.to(dtype)
        noise = torch.exp(log_noise.to(dtype))                                  # (H,) n_h
        eye = torch.eye(d, dtype=dtype, device=xi.device)
        X0 = gain_basis(C, S, blocks)                                           # (..., N, H, K, d) C S_h^T
        V_Q = _symmetrize(torch.einsum("hsk,...nhkt->...nhst", S, X0))           # (..., N, H, d, d) S_h C S_h^T
        V_K = _symmetrize(torch.einsum("hsk,...nhkt->...nhst", T, gain_basis(C, T, blocks)))   # T_h C T_h^T
        b = torch.einsum("hsk,...nk->...hns", T, xi)                            # (..., H, N, d) b_j^h
        B = V_K.movedim(-3, -4) + noise[:, None, None, None] * eye              # (..., H, N, d, d) B_j^h
        Rot = None if rope is None else rope.to(dtype)                          # (N, d, d) R_i
        if Rot is not None:
            # Sources in the absolute comparison frame: b_j -> R_j b_j, B_j -> R_j B_j R_j^T (n I is invariant).
            b = torch.einsum("nst,...hnt->...hns", Rot, b)
            B = _symmetrize(torch.einsum("nst,...hntu,nvu->...hnsv", Rot, B, Rot))

        def rotate_in(a: torch.Tensor, A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            r"""Receiver query moments in the absolute frame, :math:`(R_ia_i, R_iA_iR_i^\top)`; identity without RoPE."""
            if Rot is None:
                return a, A
            return (torch.einsum("nst,...nt->...ns", Rot, a),
                    _symmetrize(torch.einsum("nst,...ntu,nvu->...nsv", Rot, A, Rot)))

        def rotate_back(v: Optional[torch.Tensor], X: Optional[torch.Tensor]):
            r"""Receiver statistics back to the record frame, :math:`R_i^\top v_i` and :math:`R_i^\top X_iR_i`."""
            if Rot is None:
                return v, X
            v = None if v is None else torch.einsum("nts,...nt->...ns", Rot, v)
            X = None if X is None else _symmetrize(torch.einsum("nts,...ntu,nuv->...nsv", Rot, X, Rot))
            return v, X
        lp = (torch.zeros(N, N, dtype=dtype, device=xi.device) if log_prior is None
              else log_prior.to(dtype))
        if lp.dim() == 2:
            lp = lp.unsqueeze(-3)                                               # (1, N, N): one prior for every head
        lp = torch.broadcast_to(lp, (*lp.shape[:-3], H, N, N))
        # A self-masking prior (causal_noself, causal_alibi_noself) keeps entry (0, 0) only so that row 0 is not
        # all -inf. Under the minimal agent that entry would be a record of the receiver's own snapshot, so a
        # receiver whose only admissible source is itself, under a prior that masks self elsewhere, receives no
        # record: its head updates are zero (audit 2026-10-07, F29). Plain causal priors are unaffected.
        with torch.no_grad():
            admissible = torch.isfinite(lp)                                     # (..., H, N, N)
            self_admissible = admissible.diagonal(dim1=-2, dim2=-1)             # (..., H, N)
            no_record = (self_admissible & (admissible.sum(dim=-1) == 1)
                         & (~self_admissible).any(dim=-1, keepdim=True))        # (..., H, N)
        weights: List[torch.Tensor] = []
        costs: List[torch.Tensor] = []
        fractions: List[torch.Tensor] = []
        spreads: List[torch.Tensor] = []                                        # exact: tr(A R) / tr(A Lambda_bar)
        flags: List[torch.Tensor] = []                                          # failed factorizations
        screens: List[torch.Tensor] = []                                        # certificate flags (certify only)

        def check_query(h: int, A: torch.Tensor) -> None:
            r"""Factor the ADF query covariance :math:`A^h` with zero jitter (a failure is a flag) and, under
            ``certify``, screen :math:`\varepsilon\|S_hC_iS_h^\top\|_F\|(A_i^h)^{-1}\|_F > \mathrm{tol}`."""
            with torch.no_grad():
                L_A, info = torch.linalg.cholesky_ex(A)                         # (..., N, d, d)
                bad = info != 0                                                 # (..., N)
                flags.append(bad.any())
                if certify:
                    A_inv = torch.cholesky_inverse(torch.where(bad[..., None, None], eye, L_A))
                    screens.append((torch.finfo(dtype).eps * V_Q[..., h, :, :].norm(dim=(-2, -1))
                                    * A_inv.norm(dim=(-2, -1)) > projected_gram_condition_tol()).any())

        if inference == "exact":
            def head_update(h: int, a: torch.Tensor, A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                check_query(h, A)
                a_r, A_r = rotate_in(a, A)
                post = exact_record_posterior(a_r, A_r, b[..., h, :, :], B[..., h, :, :, :], lp[..., h, :, :],
                                              check=False)
                weights.append(post.weights)
                costs.append(post.cost)
                u, Lambda_bar = rotate_back(post.u, post.Lambda_bar)
                _, spread = rotate_back(None, post.R)
                if diagnostics:
                    with torch.no_grad():
                        removed = torch.diagonal(A @ Lambda_bar, dim1=-2, dim2=-1).sum(-1)
                        fractions.append(removed / d)
                        spreads.append(torch.diagonal(A @ spread, dim1=-2, dim2=-1).sum(-1) / removed)
                flags.append(post.failed)
                return u, _symmetrize(Lambda_bar - spread)
        else:
            a0 = torch.einsum("hsk,...nk->...hns", S, xi)                       # (..., H, N, d) input query means
            # projected_gaussian_kl raises on a failed factorization, so the input query and utterance covariances
            # are factored here first: a failure becomes a flag and its entry enters the KL as an identity
            # placeholder (the caller then recomputes or refuses the call); a healthy call is unchanged.
            with torch.no_grad():
                bad_Q = torch.linalg.cholesky_ex(V_Q)[1] != 0                    # (..., N, H)
                bad_B = torch.linalg.cholesky_ex(B)[1] != 0                      # (..., H, N)
            flags += [bad_Q.any(), bad_B.any()]
            V_Q_kl = torch.where(bad_Q[..., None, None], eye, V_Q)
            B_kl = torch.where(bad_B[..., None, None], eye, B)
            a0_r, V_Q_r = a0, V_Q_kl.movedim(-3, -4)                            # (..., H, N, d), (..., H, N, d, d)
            if Rot is not None:
                a0_r = torch.einsum("nst,...hnt->...hns", Rot, a0_r)
                V_Q_r = _symmetrize(torch.einsum("nst,...hntu,nvu->...hnsv", Rot, V_Q_r, Rot))
            labels = mean_field_record_posterior(a0_r, V_Q_r, b, B_kl, lp, check=False)
            if Rot is not None:
                # Label information back to receiver i's record frame: Lambda -> R_i^T Lambda R_i, eta -> R_i^T eta.
                labels = labels._replace(
                    Lambda=_symmetrize(torch.einsum("nts,...hntu,nuv->...hnsv", Rot, labels.Lambda, Rot)),
                    eta=torch.einsum("nts,...hnt->...hns", Rot, labels.eta))
            weights.append(labels.weights)
            costs.append(labels.cost)
            flags.append(labels.failed)
            if diagnostics:
                with torch.no_grad():                                           # gain fraction at the input moments
                    L_lab = torch.linalg.cholesky_ex(labels.Lambda)[0]           # (..., H, N, d, d) Lambda = L L^T
                    E0 = _symmetrize(eye + L_lab.transpose(-1, -2) @ V_Q.movedim(-3, -4) @ L_lab)
                    E0_inv = torch.cholesky_inverse(torch.linalg.cholesky_ex(E0)[0])
                    fractions.append(1.0 - torch.diagonal(E0_inv, dim1=-2, dim2=-1).sum(-1).movedim(-2, -1) / d)

            def head_update(h: int, a: torch.Tensor, A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                check_query(h, A)
                u, M, failed = information_step(a, A, labels.Lambda[..., h, :, :, :], labels.eta[..., h, :, :])
                flags.append(failed)
                return u, M
        route_update = head_update

        def head_update(h: int, a: torch.Tensor, A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            u, M_h = route_update(h, a, A)
            skip = no_record[..., h, :]                                         # (..., N)
            zero = torch.zeros((), dtype=u.dtype, device=u.device)
            return torch.where(skip[..., None], zero, u), torch.where(skip[..., None, None], zero, M_h)

        xi_plus, gains, M, _ = sequential_head_assembly(xi, X0, S, head_update)
        weights_out = torch.stack(weights, dim=-3) if inference == "exact" else weights[0]   # (..., H, N, N)
        cost_out = torch.stack(costs, dim=-3) if inference == "exact" else costs[0]
        gain_fraction = coupling = contraction = staleness = spread = None
        if diagnostics:
            gain_fraction = torch.stack(fractions, dim=-1) if inference == "exact" else fractions[0]   # (..., N, H)
            with torch.no_grad():
                coupling = _head_coupling(S, X0)
                contraction = _covariance_contraction(C, gains, M, blocks)       # (..., N)
                staleness = (_label_staleness(xi_plus, V_Q, S, gains, M, b, B, lp, weights_out, rope=Rot)
                             if inference == "mean_field"
                             else torch.full_like(coupling, math.nan))           # (..., N, H)
                spread = (torch.stack(spreads, dim=-1) if inference == "exact"
                          else torch.full_like(coupling, math.nan))              # (..., N, H)
        failed = torch.stack([flag.reshape(()) for flag in flags]).any()
        ill_conditioned = (torch.stack([screen.reshape(()) for screen in screens]).any() if screens
                           else torch.zeros((), dtype=torch.bool, device=xi.device))
    if check and bool(failed):                                                  # the one host check
        raise ValueError(readout_failure_message(dtype))
    return MinimalAgentResult(xi_plus, gains, M, weights_out, cost_out, gain_fraction, V_Q, V_K, coupling,
                              failed, ill_conditioned, contraction, staleness, spread)


# ----------------------------------------------------------------------------------------------
# Local coordinates
# ----------------------------------------------------------------------------------------------

def local_posterior_covariance(
    sigma:          torch.Tensor,               # (..., N, K) diagonal or (..., N, K, K) full input Sigma_i
    forward:        torch.Tensor,               # (..., N, G, g, g) compact or (..., N, K, K) dense U_i
    block_diagonal: bool,
    gains:          torch.Tensor,               # (..., N, H, K, d) Kmat_i^h
    M:              torch.Tensor,               # (..., N, H, d, d) M_i^h
    floor_scale:    torch.Tensor,               # (..., N) f tr(C_i) / K, the covariant floor coefficient
) -> torch.Tensor:                              # same layout as sigma
    r"""Local posterior covariance :math:`\Sigma_i^+ = U_iC_i^+U_i^\top + f\,\tfrac{\operatorname{tr}C_i}{K}U_iU_i^\top`.

    .. math::
        \Sigma_i^+ = \Sigma_i - \sum_hZ_i^hM_i^hZ_i^{h\top} + f\,\tfrac{\operatorname{tr}C_i}{K}U_iU_i^\top,\qquad
        Z_i^h = U_i\mathsf K_i^h\ (K\times d),

    using :math:`\Sigma_i = U_iC_iU_i^\top` exactly, so the input covariance is not recomputed. A diagonal input returns
    the diagonal (the moment projection onto the diagonal family),
    :math:`\operatorname{diag}\Sigma_i - \sum_h\operatorname{rowquad}(Z_i^h, M_i^h) + f\,\tfrac{\operatorname{tr}C_i}{K}
    \operatorname{diag}(U_iU_i^\top)` with :math:`\operatorname{rowquad}(Z, M)_k = Z_kMZ_k^\top`, which never forms a
    :math:`K\times K` matrix.
    """
    H, K, d = gains.shape[-3:]
    if block_diagonal:
        G, g = forward.shape[-3], forward.shape[-1]
        Z = torch.einsum("...nmab,...nhmbs->...nhmas", forward,
                         gains.reshape(*gains.shape[:-2], G, g, d)).reshape(gains.shape)    # (..., N, H, K, d)
    else:
        Z = forward.unsqueeze(-3) @ gains                                       # (..., N, H, K, d)
    if sigma.dim() == gains.dim() - 2:                                          # diagonal (..., N, K)
        if block_diagonal:
            frame_diag = forward.square().sum(dim=-1).flatten(-2)               # (..., N, K) diag(U U^T)
        else:
            frame_diag = forward.square().sum(dim=-1)
        downdate = torch.einsum("...nhks,...nhst,...nhkt->...nk", Z, M, Z)       # sum_h rowquad(Z^h, M^h)
        return sigma - downdate + floor_scale.unsqueeze(-1) * frame_diag
    if block_diagonal:
        blocks = forward @ forward.transpose(-1, -2)                            # (..., N, G, g, g) U_m U_m^T
        G, g = blocks.shape[-3], blocks.shape[-1]
        frame_outer = torch.einsum("...mab,mn->...manb", blocks,
                                   torch.eye(G, dtype=blocks.dtype, device=blocks.device)).reshape(
                                       *blocks.shape[:-3], K, K)               # (..., N, K, K)
    else:
        frame_outer = forward @ forward.transpose(-1, -2)
    downdate = torch.einsum("...nhks,...nhst,...nhlt->...nkl", Z, M, Z)          # (..., N, K, K)
    return _symmetrize(sigma - downdate + floor_scale[..., None, None] * frame_outer)
