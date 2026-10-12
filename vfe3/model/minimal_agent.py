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

The self-edge snapshot is a trade-off, not an identity (audit 2026-10-12, item 10). The record of source :math:`i` at
receiver :math:`i` treats an independent copy of the receiver's own prior as evidence, so it counts that prior twice:
at :math:`S = T` a receiver whose only admissible source is itself (token 0 under the causal prior) keeps its mean
but contracts its covariance with no new information. Conditioning on the self edge exactly (the same :math:`X_i` on
both sides of the record) removes the double count but makes the self record free, which at :math:`S = T` locks the
routing onto the self edge. The share of the self term is measurable against the ``causal_noself`` prior.

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

Spread-corrected mean-field covariance (``predictive_covariance='spread_corrected'``, audit 2026-10-12 T2;
:func:`label_spread`). With the labels :math:`\beta^h` and the Gaussian :math:`(\xi_i^+, P_i)` of the last sweep,
:math:`a_h^+ = S_h\xi_i^+`, the gradient of :math:`\log p(X, J, o)` at :math:`X = \xi_i^+` for a label tuple
:math:`J` is :math:`\sum_hS_h^\top d^h_{J^h}` with

.. math::
    d^h_j = B_j^{h\,-1}(b_j^h - a_h^+) - \sum_k\beta^h_{ik}B_k^{h\,-1}(b_k^h - a_h^+),

because its label average :math:`-C_i^{-1}(\xi_i^+ - \xi_i) + \sum_hS_h^\top(\eta^h - \Lambda^ha_h^+)` vanishes at the
power posterior. One Newton step with the mean-field Hessian :math:`P_i^{-1}` gives the label-conditional means
:math:`E[X\mid J] \approx \xi_i^+ + P_i\sum_hS_h^\top d^h_{J^h}`, and the law of total variance over the independent
labels :math:`J^h\sim\beta^h` gives

.. math::
    C_i^+ = P_i + \sum_hP_iS_h^\top R'^h_iS_hP_i,\qquad R'^h_i = \sum_j\beta^h_{ij}d^h_jd_j^{h\top}.

It is positive definite by construction: :math:`R'^h_i` is a nonnegative combination of rank-one positive semidefinite
terms, so :math:`P_iS_h^\top R'^h_iS_hP_i = (S_hP_i)^\top R'^h_i(S_hP_i)\succeq 0`, and :math:`P_i\succ 0`. It is a
linearization (a hybrid of coordinate ascent and assumed-density filtering), not an exact moment: the precision of
each label-conditional Gaussian is held at the label-averaged :math:`P_i^{-1}`. Under comparison-space RoPE
:math:`d^h_j` is formed on the rotated moments and :math:`R'^h_i` is rotated back, :math:`R_i^\top(\cdot)R_i`, like
:math:`\Lambda^h`. It is defined for the mean field only.

Per-layer evidence bound (``diagnose_agent_elbo``, audit 2026-10-12 T1; :func:`minimal_agent_elbo`,
:func:`minimal_agent_log_evidence`). In the per-head snapshot model the sources marginalize to
:math:`p(o_h = 0\mid X, J^h = j) = N(S_hX; b_j, B_j)`, and given :math:`X` the heads are independent, so the declared
joint is :math:`N(X; \xi, C)\prod_h\hat\pi^h_{J^h}N(S_hX; b^h_{J^h}, B^h_{J^h})` with
:math:`\hat\pi^h = \operatorname{softmax}_j\log\pi^h` (the normalized routing prior). Its evidence lower bound on
:math:`q(X)\prod_hq(J^h)`, :math:`q(X) = N(\xi^+, C^+)`, :math:`q(J^h) = \beta^h`, is

.. math::
    \mathcal L_i = -\mathrm{KL}(N(\xi_i^+, C_i^+)\,\|\,N(\xi_i, C_i)) + \sum_h\Bigl[-\sum_j\beta^h_{ij}
        \mathrm{KL}(N(a_h^+, A_h^+)\,\|\,N(b_j^h, B_j^h)) - \mathrm{KL}(\beta^h_i\,\|\,\hat\pi^h_i)
        - \tfrac d2(1 + \log 2\pi) - \tfrac12\log\det A_h^+\Bigr],

with :math:`A_h^+ = S_hC_i^+S_h^\top`, from
:math:`E_q\log N(S_hX; b_j, B_j) = -\mathrm{KL}(N(a^+, A^+)\|N(b_j, B_j)) - \tfrac d2(1 + \log 2\pi) - \tfrac12\log\det A^+`.
Coordinate ascent maximizes :math:`\mathcal L` over :math:`q(J)` and :math:`q(X)` in turn (the label and fusion steps
above are its exact coordinate maxima), so :math:`\mathcal L` does not decrease over sweeps. The evidence of one head is
:math:`\log p(o_h) = \operatorname{logsumexp}_j(\log\hat\pi^h_{ij} - c^h_{ij}) - \tfrac d2\log 2\pi` with the exact cost
:math:`c^h_{ij}` of :func:`exact_record_posterior`; summing it over the heads at the ADF moments is the
assumed-density-filtering evidence, which is the exact :math:`\log p(o)` for one head and for heads that satisfy the
exactness condition above, and an approximation otherwise (it is then not guaranteed to exceed :math:`\mathcal L`).
The record is declared rather than observed, so the evidence is unbounded above as :math:`S, T\to 0` and
:math:`n\to 0`: it is a diagnostic and never enters the loss.

Coordinate ascent (``cavi_steps`` :math:`= n`). The step above is one sweep of coordinate-ascent variational
inference on the structured family :math:`q(X_i)\prod_hq(J^h)`. Sweep :math:`s > 1` recomputes every head's labels
at the posterior query moments of sweep :math:`s - 1`,
:math:`\beta^{h,(s)}_{ij} = \operatorname{softmax}_j(\log\pi^h_{ij} - \mathrm{KL}(N(S_h\xi_i^{(s-1)}, S_hC_i^{(s-1)}S_h^\top)\|N(b_j^h, B_j^h)))`
(rotated as below under comparison-space RoPE), and fuses the heads again with them, starting from the INPUT
moments :math:`(\xi_i, C_i)`. This is the exact CAVI label update:
:math:`E_{q(X)}[\log N(S_hX; b_j, B_j)] = -\mathrm{KL}(N(a, A)\|N(b_j, B_j)) + c(a, A)` with a :math:`j`-independent
:math:`c`, and the label factors of different heads decouple given :math:`q(X)`, so updating them together is a
block coordinate step. :math:`n = 1` is the step above; the sweeps are unrolled under autograd.

Float32. The query covariance of a later head is a downdate,
:math:`A^h = S_hCS_h^\top - \sum_{g<h}Y_{hg}M^gY_{hg}^\top`. Its float32 error grows with
:math:`\|S_hCS_h^\top\|` and with the float32 error of the earlier heads' inverses :math:`W_{ij}^{-1}` (about
:math:`\kappa(W_{ij})\varepsilon`). With coupled heads and an ill-conditioned :math:`C_i` the error can cancel the
smallest eigenvalues of :math:`A^h` and leave it indefinite, although the exact :math:`A^h` is positive definite.
:func:`minimal_agent_readout` therefore factors every :math:`A^h` (zero jitter) and reports a failure together with
its other factorizations. Under a certified ``projected_factorization`` it also flags any float32 receiver whose
:math:`A^h` fails the certificate :math:`\varepsilon\|S_hC_iS_h^\top\|_F\|(A_i^h)^{-1}\|_F > \mathrm{tol}`, with
:math:`\mathrm{tol}` = ``projected_gram_condition_tol``. For the first head this is the kernel's Gram-route
certificate of the projected query covariance. It is a screen: it does not bound the error contributed by the
earlier inverses. :meth:`RelationalOperators.minimal_agent_update` recomputes a flagged
float32 layer call in float64 (the whole call, from the local beliefs) and raises only if the float64 call fails
too.

Comparison-space RoPE (``rope_mode='comparison'``). With the per-position rotation :math:`R_i` of
:func:`vfe3.geometry.rope.build_rope_rotation` (orthogonal, :math:`d\times d`), head :math:`h`'s record is
:math:`o_h = R_iS_hX_i - R_jT_hX_{J^h} + \varepsilon_h`; the noise :math:`n_hI_d` is rotation invariant, so this is the
record :math:`S_hX_i - R_{j-i}T_hX_{J^h} + \varepsilon'_h` of a source whose utterance reaches receiver :math:`i` through the
known channel :math:`R_{j-i} = R_i^\top R_j`. Every cost (exact and mean field) is computed on the rotated moments
:math:`(R_ia_i, R_iA_iR_i^\top)` and :math:`(R_jb_j, R_jB_jR_j^\top)`, so the labels depend on position only through
:math:`j - i`, and the receiver statistics are rotated back into the receiver's record frame,
:math:`u_i \to R_i^\top u_i`, :math:`\bar\Lambda_i, R_i^{\rm spread}, \Lambda_i \to R_i^\top(\cdot)R_i`,
:math:`\eta_i \to R_i^\top\eta_i`. This is the exact posterior of the rotated record: the gain
:math:`\mathsf K_i = C_iS^\top` is unchanged and the received content arrives through :math:`R_{j-i}` and is decoded by it,
so routing and content stay in one frame (no ``rope_mode='frame'`` split). :math:`R = I` is the unrotated model.

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
    _compute_dtype, _GramPrecision, _projected_kl_gemm, projected_condition_recovery, projected_gaussian_kl,
    projected_gram_condition_tol,
)
from vfe3.runtime import ieee_fp32_matmul

PREDICTIVE_READOUTS: Tuple[str, ...] = ("kernel", "minimal_agent")
MINIMAL_AGENT_INFERENCE: Tuple[str, ...] = ("exact", "mean_field")
# predictive_covariance values, and those under which the readout returns a covariance (passthrough keeps the
# incoming one). spread_corrected is the minimal agent's mean field with the linearized label spread (T2).
PREDICTIVE_COVARIANCES: Tuple[str, ...] = ("passthrough", "moment_matched", "spread_corrected")
READOUT_COVARIANCES: Tuple[str, ...] = ("moment_matched", "spread_corrected")
# Bytes of pair matrices one exact-routing tile may hold: W_ij, its Cholesky factor and W_ij^{-1}, r x r each.
# Read at call time, so a test can force tiny tiles by setting it.
PAIR_TILE_BYTES: int = 32 * 1024 ** 2
# Bytes of the (rows x sources x d) label-deviation tensor one label-spread tile may hold (label_spread).
SPREAD_TILE_BYTES: int = 32 * 1024 ** 2
# Per-layer evidence diagnostics (diagnose_agent_elbo only; means over receivers, nats): the evidence lower bound of
# the posterior the readout returns, the ADF (assumed-density) evidence and its difference to the bound (not
# guaranteed nonnegative with coupled heads), the exact evidence (the ADF value when every receiver's heads are
# uncoupled, S_h C_i S_g^T = 0, where ADF is exact; NaN otherwise), and the importance-sampled log evidence under q with
# its difference to the bound (audit 2026-10-12 round 2, item 12 and M1).
MINIMAL_AGENT_ELBO_KEYS: Tuple[str, ...] = (
    "relational_minimal_agent_elbo", "relational_minimal_agent_adf_evidence", "relational_minimal_agent_adf_elbo_gap",
    "relational_minimal_agent_evidence", "relational_minimal_agent_is_evidence", "relational_minimal_agent_is_elbo_gap")
# Importance-sampled log evidence (minimal_agent_importance_evidence): draws per receiver, the seed of its local
# generator (the training RNG is never touched) and the bytes of one tile of its (samples x receivers x sources x d)
# record terms.
MINIMAL_AGENT_IS_SAMPLES: int = 64
MINIMAL_AGENT_IS_SEED: int = 0
MINIMAL_AGENT_IS_TILE_BYTES: int = 32 * 1024 ** 2
# Per-layer, per-head diagnostics written to the relational capture (means over receivers).
MINIMAL_AGENT_DIAGNOSTIC_KEYS: Tuple[str, ...] = (
    "relational_minimal_agent_gain_fraction", "relational_minimal_agent_channel_noise",
    "relational_minimal_agent_head_coupling", "relational_minimal_agent_label_staleness",
    "relational_minimal_agent_covariance_contraction", "relational_minimal_agent_spread_share",
    "relational_minimal_agent_channel_noise_mstep", "relational_minimal_agent_channel_noise_mstep_boundary")
# Diagnostics defined in one inference mode only (NaN in the other).
MINIMAL_AGENT_MODE_ONLY_KEYS: Dict[str, str] = {
    "relational_minimal_agent_label_staleness": "mean_field",
    "relational_minimal_agent_spread_share": "exact",
    "relational_minimal_agent_channel_noise_mstep": "mean_field",
    "relational_minimal_agent_channel_noise_mstep_boundary": "mean_field",
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


class KeyFactors(NamedTuple):
    r"""One zero-jitter factorization of the noisy utterance covariances :math:`B_j`, shared by the label KL, the
    label statistics and the label spread of a mean-field call (:func:`key_factors`; audit 2026-10-12 round 2, item 15)."""

    chol:   torch.Tensor       # (..., J, r, r) L_j with B_j = L_j L_j^T (identity where failed)
    failed: torch.Tensor       # (..., J) bool
    P:      torch.Tensor       # (..., J, r, r) B_j^{-1} (identity where failed)
    logdet: torch.Tensor       # (..., J) log det B_j (0 where failed)
    Pb:     torch.Tensor       # (..., J, r) B_j^{-1} b_j


def key_factors(
    b: torch.Tensor,                            # (..., J, r) utterance means
    B: torch.Tensor,                            # (..., J, r, r) noisy utterance covariances, bitwise symmetric
) -> KeyFactors:
    r""":class:`KeyFactors` of :math:`B_j` from one factorization (:class:`vfe3.relational_scoring._GramPrecision`: the
    precision and log-determinant carry its closed-form backward, the factor the Cholesky backward). A failed entry
    reads the identity, as the identity placeholder of the readout does."""
    chol, failed, P, logdet = _GramPrecision.apply(B)
    eye = torch.eye(B.shape[-1], dtype=B.dtype, device=B.device)
    chol = torch.where(failed[..., None, None], eye, chol)
    logdet = logdet.masked_fill(failed, 0.0)
    Pb = torch.cholesky_solve(b.unsqueeze(-1), chol).squeeze(-1)               # (..., J, r)
    return KeyFactors(chol, failed, P, logdet, Pb)


def _query_logdet(
    A: torch.Tensor,                            # (..., I, r, r) query covariances
) -> Tuple[torch.Tensor, torch.Tensor]:         # log det A (..., I) without autograd (0 where failed), failed (..., I)
    r"""Zero-jitter factorization of the label step's query covariances, read for their failure flag and their
    log-determinant, the row constant of the label KL (no autograd: the softmax is invariant to it)."""
    with torch.no_grad():
        L, info = torch.linalg.cholesky_ex(torch.nan_to_num(A))
        failed = (info != 0) | ~torch.isfinite(A).all(dim=(-2, -1))
        return (2.0 * L.diagonal(dim1=-2, dim2=-1).log().sum(-1)).masked_fill(failed, 0.0), failed


class MinimalAgentResult(NamedTuple):
    r"""Posterior reference moments of every receiver, in the factored form of the head assembly.

    :math:`C_i^+ = C_i - \sum_h\mathsf K_i^hM_i^h\mathsf K_i^{h\top}` (``gains``, ``M``). The two covariances at the
    input moments are always returned; ``gain_fraction``, ``head_coupling``, ``covariance_contraction``,
    ``label_staleness`` (mean field) and ``spread_share`` (exact: the share :math:`\operatorname{tr}(AR)/\operatorname{tr}(A\bar\Lambda)`
    of the removed probed variance that the between-source spread gives back, the part the mean field drops) only
    with ``diagnostics=True``.
    ``failed`` reports a failed
    zero-jitter factorization or a non-finite cost anywhere in the call; ``ill_conditioned`` reports a float32 ADF
    query covariance that failed the certificate (only under a certified ``projected_factorization``, else False).
    """

    xi:               torch.Tensor             # (..., N, K) xi_i^+
    gains:            torch.Tensor             # (..., N, H, K, d) Kmat_i^h = C_i^{(h-1)} S_h^T
    M:                torch.Tensor             # (..., N, H, d, d) M_i^h
    weights:          torch.Tensor             # (..., H, N, N) labels actually used (w^h, or beta^h of the last sweep)
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
    channel_noise_mstep: Optional[torch.Tensor] = None      # (H,) mean field: M-step n_h^* (channel_noise_mstep; diagnostics)
    spread_gains:     Optional[torch.Tensor] = None         # (..., N, H, K, d) P_i S_h^T (spread_corrected only)
    label_spread:     Optional[torch.Tensor] = None         # (..., N, H, d, d) R'_i^h (spread_corrected only)
    elbo:             Optional[torch.Tensor] = None         # (..., N) float64 evidence lower bound (elbo only)
    adf_evidence:     Optional[torch.Tensor] = None         # (..., N) float64 ADF log evidence (elbo only)
    evidence:         Optional[torch.Tensor] = None         # (..., N) float64 exact log evidence: ADF where uncoupled, else NaN
    is_evidence:      Optional[torch.Tensor] = None         # (..., N) float64 importance-sampled log evidence (elbo only)
    channel_noise_mstep_boundary: Optional[torch.Tensor] = None   # (H,) bool: n_h^* at the n -> 0 bracket (diagnostics)


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
    factors: Optional[KeyFactors] = None,       # B's factorization (key_factors), or None to factor here
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:   # Lambda (..., I, r, r), eta (..., I, r), failed ()
    r"""Information statistics of the power posterior at fixed labels.

    .. math::
        \Lambda_i = \sum_j\beta_{ij}B_j^{-1},\qquad \eta_i = \sum_j\beta_{ij}B_j^{-1}b_j,

    from one zero-jitter Cholesky factor per source (no per-pair inverse), the caller's ``factors`` when given.
    """
    if factors is None:
        factors = key_factors(b, B)
    Lambda = _symmetrize(torch.einsum("...ij,...jrs->...irs", weights, factors.P))
    eta = torch.einsum("...ij,...jr->...ir", weights, factors.Pb)
    return Lambda, eta, factors.failed.any()


def mean_field_record_posterior(
    a:         torch.Tensor,                    # (..., I, r) receiver query means at the input moments
    A:         torch.Tensor,                    # (..., I, r, r) receiver query covariances at the input moments
    b:         torch.Tensor,                    # (..., J, r) utterance means
    B:         torch.Tensor,                    # (..., J, r, r) noisy utterance covariances
    log_prior: Optional[torch.Tensor] = None,   # broadcastable to (..., I, J); None: uniform
    *,
    check:     bool = True,
    labels_only_cost: bool = False,
    factors:   Optional[KeyFactors] = None,     # B's factorization (key_factors), shared with the caller
    query_logdet: Optional[torch.Tensor] = None,   # (..., I) log det A_i from the caller's factorization
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

    ``labels_only_cost=True`` is the caller's declaration that ``A`` and ``B`` are bitwise symmetric and that it
    uses the returned cost only detached, so the cost reaches autograd only through the label softmax, which is
    invariant to the row constant :math:`-\tfrac12\log\det A_i`. The KL then skips its symmetry check and runs the
    query factorization without autograd (``symmetric_inputs``, ``query_row_constant_grad=False``). The values are
    bitwise those of the default; the gradients lose only the log-determinant term, which the softmax backward
    multiplies by a row sum that is zero analytically, so they agree to rounding (audit 2026-10-12, item 21).

    ``factors`` (:func:`key_factors` of ``B``) and ``query_logdet`` (the log-determinants of ``A`` from the caller's
    own factorization, read without autograd, so only with ``labels_only_cost``) let the readout factor each
    covariance once per layer: the KL is then the healthy centered
    GEMM of :func:`vfe3.relational_scoring.projected_gaussian_kl` on those factors, the label statistics read the
    same precision, and a non-finite cost is a failure flag. Values equal the unshared call to rounding (audit
    2026-10-12 round 2, item 15). Under :func:`vfe3.relational_scoring.projected_condition_recovery`, which may
    rebuild ill-conditioned factors in float64, the KL keeps its own factorization.
    """
    if A.shape != (*a.shape, a.shape[-1]) or B.shape != (*b.shape, b.shape[-1]):
        raise ValueError("mean_field_record_posterior needs covariances (..., I, r, r) and (..., J, r, r)")
    dtype = _compute_dtype(a, A, b, B)
    I, J = a.shape[-2], b.shape[-2]
    with torch.autocast(device_type=a.device.type, enabled=False):
        a, A, b, B = a.to(dtype), A.to(dtype), b.to(dtype), B.to(dtype)
        shared = (factors is not None and query_logdet is not None and labels_only_cost
                  and not projected_condition_recovery())
        if shared:
            cost = _projected_kl_gemm(a, b, A, factors.P, query_logdet.to(dtype), factors.logdet)   # (..., I, J)
        else:
            cost = projected_gaussian_kl(a.unsqueeze(-2), A.unsqueeze(-3),
                                         b.unsqueeze(-2), B.unsqueeze(-3),
                                         symmetric_inputs=labels_only_cost,
                                         query_row_constant_grad=not labels_only_cost).squeeze(-3)   # (..., I, J)
        lp = (torch.zeros(I, J, dtype=dtype, device=a.device) if log_prior is None else log_prior.to(dtype))
        weights = torch.softmax(lp - cost, dim=-1)                              # (..., I, J)
        Lambda, eta, failed = label_statistics(weights, b, B, factors)
        if shared:
            failed = failed | ~torch.isfinite(cost).all()
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


def _posterior_query_moments(
    xi_plus: torch.Tensor,                      # (..., N, K) posterior reference means
    V_Q:     torch.Tensor,                      # (..., N, H, d, d) S_h C S_h^T at the input moments
    S:       torch.Tensor,                      # (H, d, K)
    gains:   torch.Tensor,                      # (..., N, H, K, d)
    M:       torch.Tensor,                      # (..., N, H, d, d)
    rope:    Optional[torch.Tensor] = None,     # (N, d, d) comparison rotation R_i
) -> Tuple[torch.Tensor, torch.Tensor]:         # a^+ (..., N, H, d), A^+ (..., N, H, d, d)
    r"""Posterior query moments :math:`(S_h\xi_i^+, S_hC_i^+S_h^\top)` of every head, in the comparison frame.

    .. math::
        S_hC^+S_h^\top = S_hCS_h^\top - \sum_gY_{hg}M^gY_{hg}^\top,\qquad Y_{hg} = S_h\mathsf K^g,

    then :math:`(R_ia^+, R_iA^+R_i^\top)` under comparison-space RoPE. Computed in the dtype of the inputs; nothing is
    factored here (the label step of a further sweep and :func:`_label_staleness` read these moments).
    """
    Y = torch.einsum("hsk,...ngkt->...nhgst", S, gains)                        # (..., N, H, H, d, d) S_h K^g
    A_plus = _symmetrize(V_Q - torch.einsum("...nhgst,...ngtu,...nhgvu->...nhsv", Y, M, Y))
    a_plus = torch.einsum("hsk,...nk->...nhs", S, xi_plus)                     # (..., N, H, d)
    if rope is not None:
        a_plus = torch.einsum("nst,...nht->...nhs", rope, a_plus)
        A_plus = _symmetrize(torch.einsum("nst,...nhtu,nvu->...nhsv", rope, A_plus, rope))
    return a_plus, A_plus


def _unfactorable(
    a: torch.Tensor,                            # (..., r) query means
    A: torch.Tensor,                            # (..., r, r) query covariances
) -> torch.Tensor:                              # (...) bool
    r"""Entries whose covariance fails a zero-jitter Cholesky factorization or whose moments are not finite."""
    return ((torch.linalg.cholesky_ex(torch.nan_to_num(A))[1] != 0)
            | ~torch.isfinite(A).all(dim=(-2, -1)) | ~torch.isfinite(a).all(dim=-1))


def channel_noise_mstep(
    a_plus:  torch.Tensor,                      # (..., N, H, d) posterior query means S_h xi_i^+ (record frame)
    A_plus:  torch.Tensor,                      # (..., N, H, d, d) posterior query covariances S_h C_i^+ S_h^T
    b:       torch.Tensor,                      # (..., H, N, d) utterance means T_h xi_j (same frame)
    V_K:     torch.Tensor,                      # (..., H, N, d, d) noise-free utterance covariances T_h C_j T_h^T
    weights: torch.Tensor,                      # (..., H, N, N) labels beta_ij of the last sweep
    heard:   torch.Tensor,                      # (..., H, N) bool: receivers that hear a record
    *,
    grid:    int = 64,
    iterations: int = 40,
) -> Tuple[torch.Tensor, torch.Tensor]:         # (H,) float64 n_h^* (NaN at the bracket), (H,) bool bracket hit
    r"""Variational M-step for the channel noise :math:`n_h` at the mean-field posterior (a diagnostic).

    With the sources collapsed (:math:`X_j\sim N(\xi_j, C_j)` integrated out, so the record of head :math:`h` given
    :math:`J = j` is :math:`N(S_hX_i; b_j, V_j + nI)`, :math:`V_j = T_hC_jT_h^\top`), the terms of the mean-field
    free energy on :math:`q(X_i)\prod_hq(J^h)` that depend on :math:`n = n_h` are, pooled over the receivers,

    .. math::
        -\sum_i\sum_j\beta_{ij}E_{q(X_i)}\bigl[\log N(S_hX_i; b_j, V_j + nI)\bigr]
        = \tfrac12F(n) + \mathrm{const},\qquad
        F(n) = \sum_j\bigl[m_j\log\det(V_j + nI) + \operatorname{tr}\bigl((V_j + nI)^{-1}E_j\bigr)\bigr],

    with :math:`m_j = \sum_i\beta_{ij}` and :math:`E_j = \sum_i\beta_{ij}\bigl(A_i^+ + (a_i^+ - b_j)(a_i^+ - b_j)^\top\bigr)`,
    :math:`(a_i^+, A_i^+) = (S_h\xi_i^+, S_hC_i^+S_h^\top)`. One eigendecomposition :math:`V_j = Q_j\operatorname{diag}(v_j)Q_j^\top`
    per source and :math:`e_{jk} = (Q_j^\top E_jQ_j)_{kk}\ge0` make it scalar,

    .. math::
        F(n) = \sum_{j,k}\Bigl[m_j\log(v_{jk} + n) + \frac{e_{jk}}{v_{jk} + n}\Bigr],\qquad
        F'(n) = \sum_{j,k}\Bigl[\frac{m_j}{v_{jk} + n} - \frac{e_{jk}}{(v_{jk} + n)^2}\Bigr].

    Each term decreases for :math:`n < e_{jk}/m_j - v_{jk}` and increases beyond, so every minimizer over
    :math:`n\ge0` lies in :math:`[0, U]`, :math:`U = \max_{j,k}(e_{jk}/m_j - v_{jk})^+`, and :math:`U = 0` gives
    :math:`n^* = 0`. :math:`F` is not convex in general, so the global minimizer is located on a log grid of ``grid``
    points over :math:`[10^{-12}U, U]` and refined by safeguarded Newton steps on :math:`\partial F/\partial\log n`
    inside the neighboring grid cells (bisection when a step leaves the bracket). This is the stationarity condition
    of the collapsed record.

    Boundary. The minimizer sits at the :math:`n\to 0` end of the bracket in two cases. With :math:`U = 0` every term
    increases in :math:`n`: the source covariance :math:`V_j` alone already exceeds the expected mismatch
    :math:`E_j/m_j` in every direction, so the constrained minimizer is the boundary :math:`n = 0`. With :math:`U > 0`
    and a minimizer at the lowest grid point, :math:`F` still decreases toward :math:`0`, and a direction with
    :math:`v_{jk} = e_{jk} = 0` makes it unbounded below (:math:`m_j\log n\to -\infty`): the declared record's likelihood
    is unbounded as :math:`n\to 0`. In neither case is the value an interior optimum. At the launcher's
    ``sigma_init = 15`` every head sat at the bracket and the column read 0.0 as if it had been learned (audit
    2026-10-12 round 2, M2), so such a head now returns NaN and sets its flag in the second output; :math:`n_h^*` is
    reported only when the optimum is interior. Treating :math:`X_j` as a
    latent at its snapshot law instead gives the closed form :math:`n = \sum\beta E\|S_hX_i - T_hX_j\|^2/(d\sum\beta)`,
    which minimizes a looser bound and is not this M-step (audit 2026-10-12, T6).

    The labels ``weights`` and the posterior moments must be the last CAVI sweep's pair, which is the state at which
    the M-step maximizes the free energy. ``heard`` drops receivers that hear no record (a self-masking prior's row
    0). Means are shifted by the per-head mean utterance, which leaves every difference unchanged. Computed in
    float64 without autograd; a head with a non-finite statistic reads NaN.
    """
    f64 = torch.float64
    with torch.no_grad():
        a = a_plus.movedim(-2, -3).to(f64)                                      # (..., H, N, d)
        A = A_plus.movedim(-3, -4).to(f64)                                      # (..., H, N, d, d)
        b, V = b.to(f64), V_K.to(f64)
        H, d = b.shape[-3], b.shape[-1]
        eye = torch.eye(d, dtype=f64, device=b.device)
        V = torch.where(torch.isfinite(V).all(dim=(-2, -1), keepdim=True), V, eye)   # eigh never sees a NaN
        beta = weights.to(f64) * heard.to(f64).unsqueeze(-1)                    # (..., H, N, N), rows i
        center = b.mean(dim=-2, keepdim=True)                                   # (..., H, 1, d)
        a, b = a - center, b - center
        m = beta.sum(dim=-2)                                                    # (..., H, N) m_j
        s = torch.einsum("...hij,...hir->...hjr", beta, a)                      # (..., H, N, d) sum_i beta_ij a_i
        P = A + a.unsqueeze(-1) * a.unsqueeze(-2)                               # (..., H, N, d, d) A_i + a_i a_i^T
        E = (torch.einsum("...hij,...hirs->...hjrs", beta, P)
             - s.unsqueeze(-1) * b.unsqueeze(-2) - b.unsqueeze(-1) * s.unsqueeze(-2)
             + m[..., None, None] * b.unsqueeze(-1) * b.unsqueeze(-2))          # (..., H, N, d, d) E_j
        v, Q = torch.linalg.eigh(_symmetrize(V))                                # (..., H, N, d), (..., H, N, d, d)
        e = torch.einsum("...rk,...rs,...sk->...k", Q, E, Q)                    # (..., H, N, d) diag(Q^T E Q)
        v = v.clamp_min(0.0).movedim(-3, 0).reshape(H, -1)                     # (H, T) per-direction terms
        e = e.clamp_min(0.0).movedim(-3, 0).reshape(H, -1)
        m = m.unsqueeze(-1).expand(*m.shape, d).movedim(-3, 0).reshape(H, -1)
        healthy = torch.isfinite(e).all(dim=-1) & torch.isfinite(m).all(dim=-1)   # (H,)
        e, m = torch.nan_to_num(e), torch.nan_to_num(m)
        ratio = torch.where(m > 0, e / torch.where(m > 0, m, torch.ones_like(m)) - v,
                            torch.full_like(v, -math.inf))
        U = ratio.amax(dim=-1).clamp_min(0.0)                                   # (H,) upper bracket
        positive = U > 0
        log_U = torch.where(positive, U, torch.ones_like(U)).log()              # (H,)
        t_grid = log_U.unsqueeze(-1) + torch.linspace(math.log(1e-12), 0.0, grid, dtype=f64, device=U.device)

        def objective(t: torch.Tensor) -> torch.Tensor:                         # (H, G) -> (H, G) F(e^t)
            shifted = v.unsqueeze(-2) + t.exp().unsqueeze(-1)                   # (H, G, T) v + n
            return (m.unsqueeze(-2) * shifted.log() + e.unsqueeze(-2) / shifted).sum(dim=-1)

        F_grid = torch.cat([objective(t_grid[:, k:k + 8]) for k in range(0, grid, 8)], dim=-1)   # (H, G)
        best = F_grid.argmin(dim=-1)                                            # (H,)
        rows = torch.arange(H, device=U.device)
        lo = t_grid[rows, (best - 1).clamp_min(0)]
        hi = t_grid[rows, (best + 1).clamp_max(grid - 1)]
        t = t_grid[rows, best]
        for _ in range(iterations):
            n = t.exp().unsqueeze(-1)                                           # (H, 1)
            shifted = v + n                                                     # (H, T)
            F_n = (m / shifted - e / shifted.square()).sum(dim=-1)              # (H,)
            F_nn = (-m / shifted.square() + 2.0 * e / shifted.pow(3)).sum(dim=-1)
            n = n.squeeze(-1)
            F_t, F_tt = n * F_n, n * F_n + n.square() * F_nn                    # d/dlog n, d^2/dlog n^2
            # Bracket update: F_t < 0 means the minimizer lies above t.
            lo = torch.where(F_t < 0, t, lo)
            hi = torch.where(F_t > 0, t, hi)
            newton = t - F_t / torch.where(F_tt > 0, F_tt, torch.ones_like(F_tt))
            inside = (F_tt > 0) & (newton > lo) & (newton < hi)
            t = torch.where(inside, newton, 0.5 * (lo + hi))
        refined = objective(t.unsqueeze(-1)).squeeze(-1)                       # (H,)
        t = torch.where(refined <= F_grid[rows, best], t, t_grid[rows, best])
        boundary = healthy & ~(positive & (best > 0))                           # (H,) the n -> 0 bracket
        nan = torch.full_like(t, math.nan)
        return torch.where(healthy & ~boundary, t.exp(), nan), boundary


def _label_staleness(
    xi_plus: torch.Tensor,                      # (..., N, K) posterior reference means
    V_Q:     torch.Tensor,                      # (..., N, H, d, d) S_h C S_h^T at the input moments
    S:       torch.Tensor,                      # (H, d, K)
    gains:   torch.Tensor,                      # (..., N, H, K, d)
    M:       torch.Tensor,                      # (..., N, H, d, d)
    b:       torch.Tensor,                      # (..., H, N, d)
    B:       torch.Tensor,                      # (..., H, N, d, d)
    lp:      torch.Tensor,                      # (..., H, N, N)
    weights: torch.Tensor,                      # (..., H, N, N) labels the last sweep used
    rope:    Optional[torch.Tensor] = None,     # (N, d, d) comparison rotation R_i; b, B already rotated by R_j
) -> torch.Tensor:                              # (..., N, H)
    r"""Total variation :math:`\tfrac12\sum_j|\beta^+_{ij} - \beta_{ij}|` between the mean-field labels the last sweep
    used and the same KL labels recomputed at its posterior query moments
    :math:`(S_h\xi_i^+, S_hC_i^+S_h^\top)` (:func:`_posterior_query_moments`): one further coordinate-ascent label
    step. Near 0 means a further sweep would not move the labels. Computed in
    float64; a receiver whose posterior query covariance does not factor or whose posterior mean is not finite reads
    NaN, and every receiver of a head with an utterance covariance that does not factor reads NaN. These inputs are
    replaced by placeholders before the KL, so a float32 attempt that failed (and that the caller recomputes in
    float64) never raises here.
    """
    f64 = torch.float64
    a_plus, A_plus = _posterior_query_moments(xi_plus.to(f64), V_Q.to(f64), S.to(f64), gains.to(f64), M.to(f64),
                                              None if rope is None else rope.to(f64))
    eye = torch.eye(A_plus.shape[-1], dtype=f64, device=A_plus.device)
    bad = _unfactorable(a_plus, A_plus)                                         # (..., N, H)
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


def _label_spread_tile(
    weights: torch.Tensor,                      # (..., I, J) labels of the tile's receivers
    a:       torch.Tensor,                      # (..., I, r) posterior query means of the tile's receivers
    B_inv:   torch.Tensor,                      # (..., J, r, r) B_j^{-1}
    v:       torch.Tensor,                      # (..., J, r) B_j^{-1} b_j
) -> torch.Tensor:                              # (..., I, r, r)
    r"""One receiver-row tile of :func:`label_spread`."""
    e = v.unsqueeze(-3) - torch.einsum("...jst,...it->...ijs", B_inv, a)       # (..., I, J, r) B_j^{-1}(b_j - a_i)
    mean = torch.einsum("...ij,...ijs->...is", weights, e)                      # (..., I, r) eta_i - Lambda_i a_i
    diff = e - mean.unsqueeze(-2)                                               # (..., I, J, r) d_ij
    weighted = weights.unsqueeze(-1) * diff                                     # (..., I, J, r)
    spread = torch.einsum("...ijs,...ijt->...ist", weighted, diff)              # (..., I, r, r), contracted over j
    residual = weighted.sum(dim=-2)                                             # (..., I, r) rounding of the mean
    return spread - residual.unsqueeze(-1) * residual.unsqueeze(-2)


def label_spread(
    weights: torch.Tensor,                      # (..., I, J) labels beta_ij (rows sum to one, 0 on masked pairs)
    a:       torch.Tensor,                      # (..., I, r) posterior query means a_i^+
    b:       torch.Tensor,                      # (..., J, r) utterance means
    B:       torch.Tensor,                      # (..., J, r, r) noisy utterance covariances
    factors: Optional[KeyFactors] = None,       # B's factorization (key_factors), or None to factor here
) -> Tuple[torch.Tensor, torch.Tensor]:         # R'_i (..., I, r, r), failed ()
    r"""Linearized label spread of the spread-corrected mean-field covariance (module docstring, T2).

    .. math::
        R'_i = \sum_j\beta_{ij}d_{ij}d_{ij}^\top,\qquad
        d_{ij} = B_j^{-1}(b_j - a_i) - \sum_k\beta_{ik}B_k^{-1}(b_k - a_i),

    accumulated about each row's own mean (the two-pass form of ``_centered_spread``, so the label average is not
    cancelled against a second moment) and positive semidefinite up to the rounding of that mean. One Cholesky factor
    per source; receiver rows are tiled so that one tile holds at most :data:`SPREAD_TILE_BYTES` of the
    :math:`(I, J, r)` deviations, and each tile is a non-reentrant checkpoint under autograd when there is more than
    one. No :math:`(I, J, r, r)` tensor is formed: the spread is a batched product over :math:`j`. Cost
    :math:`O(IJr^2)` per batch element. ``factors`` is the label step's factorization of ``B`` (:func:`key_factors`),
    so the readout factors it once (audit 2026-10-12 round 2, item 15).
    """
    if factors is None:
        factors = key_factors(b, B)
    B_inv = _symmetrize(factors.P)                                              # (..., J, r, r)
    v = factors.Pb                                                              # (..., J, r) B_j^{-1} b_j
    I, J, r = a.shape[-2], b.shape[-2], a.shape[-1]
    batch = math.prod(torch.broadcast_shapes(weights.shape[:-2], a.shape[:-2], b.shape[:-2]))
    per_row = max(1, batch * J * r * a.element_size())
    step = max(1, min(I, SPREAD_TILE_BYTES // per_row))
    tiles = []
    for i in range(0, I, step):
        args = (weights[..., i:i + step, :], a[..., i:i + step, :], B_inv, v)
        if torch.is_grad_enabled() and step < I and any(x.requires_grad for x in args):
            tiles.append(checkpoint(_label_spread_tile, *args, use_reentrant=False, preserve_rng_state=False))
        else:
            tiles.append(_label_spread_tile(*args))
    spread = tiles[0] if len(tiles) == 1 else torch.cat(tiles, dim=-3)
    return _symmetrize(spread), factors.failed.any()


def _spread_gains(
    X0:    torch.Tensor,                        # (..., N, H, K, d) C_i S_h^T
    S:     torch.Tensor,                        # (H, d, K)
    gains: torch.Tensor,                        # (..., N, H, K, d) Kmat^g
    M:     torch.Tensor,                        # (..., N, H, d, d) M^g
) -> torch.Tensor:                              # (..., N, H, K, d) C_i^+ S_h^T
    r""":math:`C_i^+S_h^\top = C_iS_h^\top - \sum_g\mathsf K^gM^gY_{hg}^\top`, :math:`Y_{hg} = S_h\mathsf K^g`; the mean field's
    :math:`P_iS_h^\top` (only :math:`K\times d` and :math:`d\times d` products)."""
    Y = torch.einsum("hsk,...ngkt->...nhgst", S, gains)                        # (..., N, H, H, d, d) S_h K^g
    return X0 - torch.einsum("...ngkt,...ngtu,...nhgsu->...nhks", gains, M, Y)


def _dense_covariance(
    C:      torch.Tensor,                       # (..., N, K, K) dense or (..., N, G, g, g) blocks
    blocks: bool,
) -> torch.Tensor:                              # (..., N, K, K)
    r"""The dense :math:`C_i` of a dense or block-diagonal input."""
    if not blocks:
        return C
    G, g = C.shape[-3], C.shape[-1]
    eye = torch.eye(G, dtype=C.dtype, device=C.device)
    return torch.einsum("...nmab,mq->...nmaqb", C, eye).reshape(*C.shape[:-3], G * g, G * g)


def _posterior_covariance(
    C:            torch.Tensor,                 # (..., N, K, K) dense input C_i
    gains:        torch.Tensor,                 # (..., N, H, K, d) Kmat_i^h
    M:            torch.Tensor,                 # (..., N, H, d, d) M_i^h
    spread_gains: Optional[torch.Tensor],       # (..., N, H, K, d) P_i S_h^T, or None
    label_spread: Optional[torch.Tensor],       # (..., N, H, d, d) R'_i^h, or None
) -> torch.Tensor:                              # (..., N, K, K)
    r"""Dense posterior covariance of the readout in reference coordinates,
    :math:`C_i - \sum_h\mathsf K_i^hM_i^h\mathsf K_i^{h\top}`, plus :math:`\sum_hP_iS_h^\top R'^h_iS_hP_i` under
    ``spread_corrected`` (``label_spread`` given), in the dtype of ``C``."""
    C_plus = C - torch.einsum("...nhks,...nhst,...nhlt->...nkl", gains, M, gains)
    if label_spread is not None:
        G = spread_gains.to(C.dtype)
        C_plus = C_plus + torch.einsum("...nhks,...nhst,...nhlt->...nkl", G, label_spread.to(C.dtype), G)
    return _symmetrize(C_plus)


def _cholesky_logdet(L: torch.Tensor) -> torch.Tensor:   # (..., r, r) -> (...)
    return 2.0 * L.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)


def minimal_agent_elbo(
    xi:        torch.Tensor,                    # (..., N, K) input reference means xi_i
    C:         torch.Tensor,                    # (..., N, K, K) dense or (..., N, G, g, g) blocks, input C_i
    xi_plus:   torch.Tensor,                    # (..., N, K) posterior means
    gains:     torch.Tensor,                    # (..., N, H, K, d) Kmat_i^h
    M:         torch.Tensor,                    # (..., N, H, d, d) M_i^h
    V_Q:       torch.Tensor,                    # (..., N, H, d, d) S_h C_i S_h^T at the input moments
    S:         torch.Tensor,                    # (H, d, K)
    b:         torch.Tensor,                    # (..., H, N, d) utterance means (comparison frame)
    B:         torch.Tensor,                    # (..., H, N, d, d) noisy utterance covariances (comparison frame)
    lp:        torch.Tensor,                    # (..., H, N, N) routing log-prior log pi (unnormalized; -inf masks)
    weights:   torch.Tensor,                    # (..., H, N, N) label factors q(J^h) = beta^h
    no_record: torch.Tensor,                    # (..., H, N) bool: the receiver hears no record from head h
    *,
    blocks:    bool,
    rope:      Optional[torch.Tensor] = None,   # (N, d, d) comparison rotation R_i
    spread_gains: Optional[torch.Tensor] = None,   # (..., N, H, K, d) P_i S_h^T (spread_corrected only)
    label_spread: Optional[torch.Tensor] = None,   # (..., N, H, d, d) R'_i^h (spread_corrected only)
) -> torch.Tensor:                              # (..., N) float64
    r"""Evidence lower bound :math:`\mathcal L_i` of the declared record model at
    :math:`q = N(\xi_i^+, C_i^+)\prod_h\beta^h_i` (module docstring, T1), per receiver, in nats.

    .. math::
        \mathcal L_i = -\mathrm{KL}(N(\xi_i^+, C_i^+)\,\|\,N(\xi_i, C_i)) + \sum_{h:\,\text{record}}\Bigl[
            -\sum_j\beta^h_{ij}\mathrm{KL}_{ij}^{h,+} - \mathrm{KL}(\beta^h_i\,\|\,\hat\pi^h_i)
            - \tfrac d2(1 + \log 2\pi) - \tfrac12\log\det A_h^+\Bigr],

    :math:`C_i^+ = C_i - \sum_h\mathsf K_i^hM_i^h\mathsf K_i^{h\top}` (formed dense, :math:`K\times K`),
    :math:`\mathrm{KL}^{h,+}_{ij} = \mathrm{KL}(N(a_h^+, A_h^+)\|N(b_j^h, B_j^h))` on the comparison-frame posterior query
    moments of :func:`_posterior_query_moments` (rotated under ``rope``; :math:`\log\det A^+` is rotation invariant),
    :math:`\hat\pi^h_i = \operatorname{softmax}_j\log\pi^h_{ij}` and :math:`0\log 0 = 0`. A head that gives a receiver no
    record (``no_record``) contributes nothing, as in the readout. Every quantity is detached and the computation
    runs in float64 without autograd: it is a diagnostic of the posterior the readout returned, valid for any
    :math:`q` (mean field: the last sweep's labels and fused Gaussian; exact: the ADF posterior and its per-head
    labels), and never enters the loss. A receiver whose :math:`C_i^+` or :math:`A_h^+` does not factor, or a head
    with an utterance covariance that does not, reads NaN.

    With ``spread_gains`` and ``label_spread`` (``predictive_covariance='spread_corrected'``) :math:`q(X)` is the
    Gaussian the readout returns, :math:`C_i^+ = P_i + \sum_hP_iS_h^\top R'^h_iS_hP_i` (module docstring, T2), and
    :math:`A_h^+ = S_hC_i^+S_h^\top` carries the spread too; the bound holds for any :math:`q`, so it is still a lower
    bound, now of the posterior the layer passes on (audit 2026-10-12 round 2, item 12). It is not the coordinate-ascent
    objective at its maximum, so it need not increase over sweeps. The prior-anchored floor is added in local
    coordinates after this posterior and is not part of :math:`q`.
    """
    f64 = torch.float64
    with torch.no_grad():
        xi, xi_plus, gains, M = (t.detach().to(f64) for t in (xi, xi_plus, gains, M))
        V_Q, S, b, B, lp, weights = (t.detach().to(f64) for t in (V_Q, S, b, B, lp, weights))
        rope = None if rope is None else rope.detach().to(f64)
        H, d, K = S.shape
        C = _dense_covariance(C.detach().to(f64), blocks)                       # (..., N, K, K)
        C_plus = _posterior_covariance(C, gains, M, spread_gains, label_spread)   # (..., N, K, K) C_i^+ of q
        L_C, info_C = torch.linalg.cholesky_ex(C)
        L_P, info_P = torch.linalg.cholesky_ex(C_plus)
        eye_K = torch.eye(K, dtype=f64, device=C.device)
        L_P = torch.where((info_P != 0)[..., None, None], eye_K, L_P)
        whitened = torch.linalg.solve_triangular(L_C, L_P, upper=False)         # L_C^{-1} L_P
        shift = torch.linalg.solve_triangular(L_C, (xi_plus - xi).unsqueeze(-1), upper=False).squeeze(-1)
        kl_X = 0.5 * (whitened.square().sum(dim=(-2, -1)) + shift.square().sum(dim=-1) - K
                      + _cholesky_logdet(L_C) - _cholesky_logdet(L_P))           # (..., N)
        kl_X = kl_X.masked_fill((info_C != 0) | (info_P != 0), math.nan)

        a_plus, A_plus = _posterior_query_moments(xi_plus, V_Q, S, gains, M, None)   # (..., N, H, d), (..., N, H, d, d)
        if label_spread is not None:                                            # + S_h (sum_g P S_g^T R'_g S_g P) S_h^T
            Y = torch.einsum("hsk,...ngkt->...nhgst", S, spread_gains.detach().to(f64))   # (..., N, H, H, d, d)
            A_plus = _symmetrize(A_plus + torch.einsum("...nhgst,...ngtu,...nhgvu->...nhsv", Y,
                                                       label_spread.detach().to(f64), Y))
        if rope is not None:
            a_plus = torch.einsum("nst,...nht->...nhs", rope, a_plus)
            A_plus = _symmetrize(torch.einsum("nst,...nhtu,nvu->...nhsv", rope, A_plus, rope))
        eye = torch.eye(d, dtype=f64, device=C.device)
        bad = _unfactorable(a_plus, A_plus)                                     # (..., N, H)
        A_plus = torch.where(bad[..., None, None], eye, A_plus)
        a_plus = torch.where(bad[..., None], torch.zeros((), dtype=f64, device=C.device), a_plus)
        bad_B = torch.linalg.cholesky_ex(B)[1] != 0                              # (..., H, N)
        B = torch.where(bad_B[..., None, None], eye, B)
        cost = projected_gaussian_kl(a_plus, A_plus, b.movedim(-3, -2), B.movedim(-4, -3))   # (..., H, N, N)
        log_pi = torch.log_softmax(lp, dim=-1)                                  # (..., H, N, N) normalized
        live = weights > 0
        zero = torch.zeros((), dtype=f64, device=C.device)
        expected = torch.where(live, weights * cost, zero).sum(dim=-1)          # (..., H, N) sum_j beta KL^+
        kl_labels = torch.where(live, weights * (weights.log() - log_pi), zero).sum(dim=-1)   # KL(beta || pi_hat)
        logdet_A = _cholesky_logdet(torch.linalg.cholesky_ex(A_plus)[0])        # (..., N, H) A_plus factors (bad -> I)
        head = (-(expected + kl_labels).movedim(-1, -2) - 0.5 * d * (1.0 + math.log(2.0 * math.pi))
                - 0.5 * logdet_A)                                               # (..., N, H)
        head = head.masked_fill(bad | bad_B.any(dim=-1, keepdim=True).movedim(-1, -2), math.nan)
        head = head.masked_fill(no_record.movedim(-1, -2), 0.0)
        return head.sum(dim=-1) - kl_X                                          # (..., N)


def minimal_agent_log_evidence(
    cost:      torch.Tensor,                    # (..., H, N, N) exact costs c_ij at each head's ADF moments
    lp:        torch.Tensor,                    # (..., H, N, N) routing log-prior (unnormalized; -inf masks)
    no_record: torch.Tensor,                    # (..., H, N) bool
    d:         int,                             # comparison rank
) -> torch.Tensor:                              # (..., N) float64
    r"""ADF log evidence :math:`\sum_h\bigl[\operatorname{logsumexp}_j(\log\hat\pi^h_{ij} - c^h_{ij}) - \tfrac d2\log 2\pi\bigr]`.

    :math:`c^h_{ij} = -\log N(b_j; a_i^h, W_{ij}^h) - \tfrac d2\log 2\pi` is the cost of :func:`exact_record_posterior`
    at head :math:`h`'s ADF moments, so each bracket is :math:`\log\sum_j\hat\pi^h_{ij}N(0; a_i^h - b_j^h, W^h_{ij})`, the
    log probability of the record :math:`o_h = 0` given the records of heads :math:`1..h-1` under the collapsed
    Gaussian. For one head this is the exact :math:`\log p(o)`; for several it is exact when
    :math:`S_hC_iS_g^\top = 0` for :math:`h\ne g` (module docstring, Exactness). A head without a record contributes 0.
    Detached, float64, no autograd; costs of a skipped masked tile (0) meet :math:`\log\hat\pi = -\infty` and add nothing.
    """
    with torch.no_grad():
        cost, lp = cost.detach().to(torch.float64), lp.detach().to(torch.float64)
        per_head = (torch.logsumexp(torch.log_softmax(lp, dim=-1) - cost, dim=-1)
                    - 0.5 * d * math.log(2.0 * math.pi))                        # (..., H, N)
        return per_head.masked_fill(no_record, 0.0).sum(dim=-2)                 # (..., N)


def minimal_agent_importance_evidence(
    xi:        torch.Tensor,                    # (..., N, K) input reference means xi_i
    C:         torch.Tensor,                    # (..., N, K, K) dense or (..., N, G, g, g) blocks, input C_i
    xi_plus:   torch.Tensor,                    # (..., N, K) posterior means (the proposal mean)
    C_plus:    torch.Tensor,                    # (..., N, K, K) dense posterior covariance (the proposal covariance)
    S:         torch.Tensor,                    # (H, d, K)
    b:         torch.Tensor,                    # (..., H, N, d) utterance means (comparison frame)
    B:         torch.Tensor,                    # (..., H, N, d, d) noisy utterance covariances (comparison frame)
    lp:        torch.Tensor,                    # (..., H, N, N) routing log-prior (unnormalized; -inf masks)
    no_record: torch.Tensor,                    # (..., H, N) bool
    *,
    blocks:    bool,
    rope:      Optional[torch.Tensor] = None,   # (N, d, d) comparison rotation R_i
    samples:   int = MINIMAL_AGENT_IS_SAMPLES,
) -> torch.Tensor:                              # (..., N) float64
    r"""Importance-sampled log evidence of the declared record model with the readout's :math:`q(X_i)` as proposal.

    With the labels summed out exactly, :math:`p(X, o) = N(X; \xi_i, C_i)\prod_h\sum_j\hat\pi^h_{ij}N(R_iS_hX; b_j^h, B_j^h)`
    (module docstring, T1; a head without a record contributes 1), and for :math:`X^{(s)}\sim q = N(\xi_i^+, C_i^+)`,
    :math:`s = 1..n`,

    .. math::
        \widehat{\log p}(o) = \operatorname{logsumexp}_s\log w_s - \log n,\qquad
        w_s = p(X^{(s)}, o)/q(X^{(s)}).

    :math:`\frac1n\sum_sw_s` is unbiased for :math:`p(o)`; its logarithm is a consistent estimator of
    :math:`\log p(o)` and, by Jensen's inequality, biased downward in expectation, with
    :math:`E\widehat{\log p} \ge E_q\log w \ge \mathcal L_i` (summing the labels out removes the label KL of
    :func:`minimal_agent_elbo`). It needs no exactness condition, so it is the evidence estimate for coupled heads,
    where the ADF value of :func:`minimal_agent_log_evidence` is an approximation (audit 2026-10-12 round 2, M1).
    The draws come from a local generator seeded with :data:`MINIMAL_AGENT_IS_SEED`, so the training RNG is untouched
    and the estimate is reproducible; ``samples`` = :data:`MINIMAL_AGENT_IS_SAMPLES` by default. Detached, float64,
    no autograd, tiled over the samples by :data:`MINIMAL_AGENT_IS_TILE_BYTES`. A receiver whose :math:`C_i` or
    :math:`C_i^+` does not factor, or a head with an utterance covariance that does not, reads NaN.
    """
    f64 = torch.float64
    with torch.no_grad():
        xi, xi_plus, C_plus = (t.detach().to(f64) for t in (xi, xi_plus, C_plus))
        S, b, B, lp = (t.detach().to(f64) for t in (S, b, B, lp))
        rope = None if rope is None else rope.detach().to(f64)
        H, d, K = S.shape
        C = _dense_covariance(C.detach().to(f64), blocks)                       # (..., N, K, K)
        eye_K, eye = torch.eye(K, dtype=f64, device=C.device), torch.eye(d, dtype=f64, device=C.device)
        L_C, info_C = torch.linalg.cholesky_ex(C)
        L_Q, info_Q = torch.linalg.cholesky_ex(C_plus)
        bad_X = (info_C != 0) | (info_Q != 0)                                   # (..., N)
        L_C = torch.where(bad_X[..., None, None], eye_K, L_C)
        L_Q = torch.where(bad_X[..., None, None], eye_K, L_Q)
        bad_B = torch.linalg.cholesky_ex(B)[1] != 0                              # (..., H, N)
        L_B = torch.linalg.cholesky_ex(torch.where(bad_B[..., None, None], eye, B))[0]   # (..., H, N, d, d)
        P_B = torch.cholesky_inverse(L_B)                                       # (..., H, N, d, d) B_j^{-1}
        P_b = (P_B @ b.unsqueeze(-1)).squeeze(-1)                               # (..., H, N, d) B_j^{-1} b_j
        source = (-0.5 * (b * P_b).sum(dim=-1) - 0.5 * _cholesky_logdet(L_B)
                  - 0.5 * d * math.log(2.0 * math.pi))                          # (..., H, N) j-only terms
        log_pi = torch.log_softmax(lp, dim=-1)                                  # (..., H, N, N) normalized
        const = _cholesky_logdet(L_C) - _cholesky_logdet(L_Q)                  # (..., N) log det C_i - log det C_i^+
        per_sample = math.prod(xi.shape[:-2]) * H * xi.shape[-2] ** 2 * d * 8   # bytes of one sample's pair tensor
        chunk = max(1, min(samples, MINIMAL_AGENT_IS_TILE_BYTES // max(1, per_sample)))
        generator = torch.Generator(device=xi.device).manual_seed(MINIMAL_AGENT_IS_SEED)
        log_w: List[torch.Tensor] = []
        for start in range(0, samples, chunk):
            n = min(chunk, samples - start)
            z = torch.randn((n, *xi_plus.shape), generator=generator, dtype=f64, device=xi.device)   # (n, ..., N, K)
            X = xi_plus + (L_Q @ z.unsqueeze(-1)).squeeze(-1)                   # (n, ..., N, K)
            r = torch.linalg.solve_triangular(L_C, (X - xi).unsqueeze(-1), upper=False).squeeze(-1)
            log_ratio = 0.5 * (z.square().sum(dim=-1) - r.square().sum(dim=-1)) - 0.5 * const   # log N(X; xi, C) - log q
            y = torch.einsum("hsk,...nk->...hns", S, X)                         # (n, ..., H, N, d) S_h X_i
            if rope is not None:
                y = torch.einsum("nst,...hnt->...hns", rope, y)                 # R_i S_h X_i
            # log N(y_i; b_j, B_j) = -1/2 y_i^T B_j^{-1} y_i + y_i^T B_j^{-1} b_j + source_j       (n, ..., H, N_i, N_j)
            quadratic = torch.einsum("...his,...hjst,...hit->...hij", y, P_B, y)
            cross = torch.einsum("...his,...hjs->...hij", y, P_b)
            record = torch.logsumexp(log_pi - 0.5 * quadratic + cross + source.unsqueeze(-2), dim=-1)   # (n, ..., H, N)
            log_w.append(log_ratio + record.masked_fill(no_record, 0.0).sum(dim=-2))   # (n, ..., N)
        estimate = torch.logsumexp(torch.cat(log_w), dim=0) - math.log(samples)          # (..., N)
        estimate = estimate.masked_fill(bad_X, math.nan)
        return estimate.masked_fill(bad_B.any(dim=-1).any(dim=-1, keepdim=True), math.nan)


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
    cavi_steps:  int = 1,                       # mean field: coordinate-ascent sweeps (1: labels at the input)
    spread_corrected: bool = False,             # mean field: also return P S_h^T and the label spread R'^h (T2)
    elbo:        bool = False,                  # also return the evidence lower bound and the ADF evidence (T1)
) -> MinimalAgentResult:
    r"""Posterior reference moments of every receiver under the minimal agent model.

    Sources are the input snapshots, :math:`b_j^h = T_h\xi_j` and :math:`B_j^h = T_hC_jT_h^\top + n_hI_d` (input
    :math:`C_j`). ``inference="exact"`` runs ADF over the heads with :func:`exact_record_posterior` at each head's
    current moments, :math:`M^h = \bar\Lambda^h - R^h`; ``"mean_field"`` takes every head's labels from
    :func:`mean_field_record_posterior` at the input moments and fuses them with :func:`information_step`
    (module docstring). With ``cavi_steps`` :math:`= n > 1` the mean field runs :math:`n` coordinate-ascent sweeps:
    each later sweep recomputes the labels at the previous sweep's posterior query moments
    (:func:`_posterior_query_moments`, rotated under ``rope``) and fuses them from the input moments again. Those
    moments are factored with zero jitter first; a failure is a flag and enters the KL as an identity placeholder.
    ``weights``, ``cost``, the gain fraction and the label staleness then belong to the last sweep. The exact path
    does not read ``cavi_steps``.

    The gain fraction of the exact path is :math:`\operatorname{tr}(A_i^h\bar\Lambda_i^h)/d` at head :math:`h`'s ADF
    moments: the label-averaged share of the probed variance that the component updates remove, in
    :math:`A^h`-whitened coordinates, :math:`\in [0, 1)`; like ADF it depends on the head order. The mean field's is
    evaluated at the INPUT moments :math:`A_i^h = S_hC_iS_h^\top`, from which every sweep fuses (with the last
    sweep's :math:`\Lambda^h`), so it does not depend on the head order, as the fused posterior does not:

    .. math::
        \tfrac1d\operatorname{tr}(A^hM_0^h) = 1 - \tfrac1d\operatorname{tr}(E^{-1}),\qquad
        M_0^h = (A^h + \Lambda^{h\,-1})^{-1} = LE^{-1}L^\top,\quad \Lambda^h = LL^\top,\quad E = I + L^\top A^hL,

    the share of the probed variance that head :math:`h`'s channel information alone would remove from the input
    Gaussian. It equals the exact fraction for one head with equal :math:`B_j` and :math:`\beta = w`. The gain
    fraction and the head coupling are diagnostics, computed only with ``diagnostics=True`` (else None).

    ``C`` with two more axes than ``xi`` is read as the diagonal blocks of a block-diagonal :math:`C_i` (diagonal
    beliefs on block frames); no :math:`K\times K` matrix is then formed. Every factorization of the call has zero
    jitter, the ADF query covariances :math:`A^h` included (module docstring, Float32), and their failure flags are
    folded into ``failed``. Under a certified ``projected_factorization`` a float32 call also screens every :math:`A^h`
    with the certificate into ``ill_conditioned``. ``check=True`` makes one host check and raises ``ValueError`` on
    a failure (a certificate alone does not raise); ``check=False`` returns both flags for the caller's single
    check. The mean field's input covariances are factored here first, so ``projected_gaussian_kl`` (which keeps its
    own host transfer) never receives a failed one. Requires :math:`n_h > 0`, every :math:`S_h` of full row rank and
    :math:`C_i\succ 0`. ``rope`` applies the comparison-space rotation of the module docstring (labels depend on
    position only through :math:`j - i`; ``query_covariance`` and ``key_covariance`` stay unrotated).

    ``spread_corrected`` (mean field only) also returns ``spread_gains`` :math:`P_iS_h^\top` (:func:`_spread_gains`) and
    ``label_spread`` :math:`R'^h_i` (:func:`label_spread`, in the record frame, 0 for a head without a record), from
    which :func:`local_posterior_covariance` adds :math:`\sum_hP_iS_h^\top R'^h_iS_hP_i` (module docstring, T2); its
    factorization flags join ``failed``. ``elbo`` also returns, per receiver in float64 and without autograd,
    :func:`minimal_agent_elbo` at the returned posterior (with the label spread under ``spread_corrected``),
    ``adf_evidence`` from :func:`minimal_agent_log_evidence` at the ADF moments of the exact path (computed in
    float64: a mean-field or float32 call reruns that path once in float64 on detached inputs for it; NaN if it
    fails), ``evidence``, the same value where the receiver's heads are uncoupled and ADF is exact (NaN elsewhere),
    and ``is_evidence`` from :func:`minimal_agent_importance_evidence` with the returned posterior as proposal
    (audit 2026-10-12 round 2, item 12 and M1). Under ``diagnostics`` the mean field's
    ``channel_noise_mstep_boundary`` flags the heads whose M-step optimum sits at the :math:`n\to 0` bracket.
    """
    if inference not in MINIMAL_AGENT_INFERENCE:
        raise ValueError(f"minimal_agent_inference must be one of {MINIMAL_AGENT_INFERENCE}, got {inference!r}")
    if spread_corrected and inference != "mean_field":
        raise ValueError("the spread-corrected covariance is a mean-field correction; the exact path already returns "
                         "the moments of the exact posterior mixture (spread included)")
    if isinstance(cavi_steps, bool) or not isinstance(cavi_steps, int) or cavi_steps < 1:
        raise ValueError(f"cavi_steps must be an int >= 1, got {cavi_steps!r}")
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
            # Each covariance is factored once per layer (audit 2026-10-12 round 2, item 15): the query factorization
            # gives the failure flag and the KL's row constant log det A_i, and the one factorization of B_j serves
            # the KL, the label statistics of every sweep and the label spread.
            logdet_Q, bad_Q = _query_logdet(V_Q)                                # (..., N, H) each
            factors = key_factors(b, B)                                         # B_j: (..., H, N, d, d)
            bad_B = factors.failed                                              # (..., H, N)
            flags += [bad_Q.any(), bad_B.any()]
            V_Q_kl = torch.where(bad_Q[..., None, None], eye, V_Q)
            B_kl = torch.where(bad_B[..., None, None], eye, B)
            a0_r, V_Q_r = a0, V_Q_kl.movedim(-3, -4)                            # (..., H, N, d), (..., H, N, d, d)
            logdet_r = logdet_Q.movedim(-1, -2)                                 # (..., H, N)
            if Rot is not None:
                a0_r = torch.einsum("nst,...hnt->...hns", Rot, a0_r)
                V_Q_r = _symmetrize(torch.einsum("nst,...hntu,nvu->...hnsv", Rot, V_Q_r, Rot))
                # The rotation rounds, so the rotated covariance is factored again: a failure is a flag
                # with the identity placeholder, as for the unrotated one (audit 2026-10-08).
                logdet_r, bad_Q_r = _query_logdet(V_Q_r)                        # (..., H, N) each
                flags.append(bad_Q_r.any())
                V_Q_r = torch.where(bad_Q_r[..., None, None], eye, V_Q_r)

            def label_step(a_r: torch.Tensor, A_r: torch.Tensor, logdet_A: torch.Tensor) -> MeanFieldPosterior:
                r"""Labels at the comparison-frame query moments ``(a_r, A_r)`` ``(..., H, N, d)``, ``(..., H, N, d, d)``
                (``logdet_A`` ``(..., H, N)`` from their factorization), their information back in the record frame;
                the labels, costs and flag are recorded."""
                # A_r and B_kl are bitwise symmetric (_symmetrize, a diagonal shift, identity placeholders) and the
                # cost leaves this readout only detached (score_energy_pre).
                step = mean_field_record_posterior(a_r, A_r, b, B_kl, lp, check=False, labels_only_cost=True,
                                                   factors=factors, query_logdet=logdet_A)
                if Rot is not None:
                    # Label information back to receiver i's record frame: Lambda -> R_i^T Lambda R_i, eta -> R_i^T eta.
                    step = step._replace(
                        Lambda=_symmetrize(torch.einsum("nts,...hntu,nuv->...hnsv", Rot, step.Lambda, Rot)),
                        eta=torch.einsum("nts,...hnt->...hns", Rot, step.eta))
                weights.append(step.weights)
                costs.append(step.cost)
                flags.append(step.failed)
                return step

            labels = label_step(a0_r, V_Q_r, logdet_r)                          # sweep 1: the input moments

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
        for _ in range(1, cavi_steps if inference == "mean_field" else 1):
            # A further CAVI sweep: labels at the current posterior query moments (head_update reads the rebound
            # labels), then the Gaussian fused again from the input moments.
            a_post, A_post = _posterior_query_moments(xi_plus, V_Q, S, gains, M, Rot)   # (..., N, H, d), (..., N, H, d, d)
            logdet_post, bad_post = _query_logdet(A_post)                       # (..., N, H) each
            bad_post = bad_post | ~torch.isfinite(a_post).all(dim=-1)
            flags.append(bad_post.any())
            A_post = torch.where(bad_post[..., None, None], eye, A_post)
            a_post = torch.where(bad_post[..., None], torch.zeros((), dtype=dtype, device=xi.device), a_post)
            labels = label_step(a_post.movedim(-2, -3), A_post.movedim(-3, -4),
                                logdet_post.masked_fill(bad_post, 0.0).movedim(-1, -2))
            xi_plus, gains, M, _ = sequential_head_assembly(xi, X0, S, head_update)
        if diagnostics and inference == "mean_field":
            with torch.no_grad():                                               # gain fraction at the input moments
                L_lab = torch.linalg.cholesky_ex(labels.Lambda)[0]               # (..., H, N, d, d) Lambda = L L^T
                E0 = _symmetrize(eye + L_lab.transpose(-1, -2) @ V_Q.movedim(-3, -4) @ L_lab)
                E0_inv = torch.cholesky_inverse(torch.linalg.cholesky_ex(E0)[0])
                fractions.append(1.0 - torch.diagonal(E0_inv, dim1=-2, dim2=-1).sum(-1).movedim(-2, -1) / d)
        weights_out = torch.stack(weights, dim=-3) if inference == "exact" else weights[-1]   # (..., H, N, N)
        cost_out = torch.stack(costs, dim=-3) if inference == "exact" else costs[-1]
        gain_fraction = coupling = contraction = staleness = spread = mstep = mstep_boundary = None
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
                if inference == "mean_field":
                    # The M-step reads the last sweep's labels with the posterior they produced, in the record
                    # frame of the labels (sources and receivers rotated under comparison-space RoPE).
                    f64 = torch.float64
                    a_post, A_post = _posterior_query_moments(
                        xi_plus.to(f64), V_Q.to(f64), S.to(f64), gains.to(f64), M.to(f64),
                        None if Rot is None else Rot.to(f64))                   # (..., N, H, d), (..., N, H, d, d)
                    V_src = V_K.movedim(-3, -4).to(f64)                         # (..., H, N, d, d) T_h C_j T_h^T
                    if Rot is not None:
                        V_src = _symmetrize(torch.einsum("nst,...hntu,nvu->...hnsv", Rot.to(f64), V_src, Rot.to(f64)))
                    heard = torch.broadcast_to(~no_record, weights_out.shape[:-1])   # (..., H, N)
                    mstep, mstep_boundary = channel_noise_mstep(a_post, A_post, b.to(f64), V_src, weights_out,
                                                                heard)          # (H,), (H,)
                else:
                    mstep = torch.full((H,), math.nan, dtype=torch.float64, device=xi.device)
        spread_gains = label_spread_out = None
        if spread_corrected:
            # T2: the label spread at the last sweep's labels and posterior query means, in the comparison frame,
            # rotated back to the record frame; a head without a record adds nothing.
            a_post = torch.einsum("hsk,...nk->...hns", S, xi_plus)              # (..., H, N, d) a^+
            if Rot is not None:
                a_post = torch.einsum("nst,...hnt->...hns", Rot, a_post)
            R_prime, bad_spread = label_spread(weights_out, a_post, b, B_kl, factors)   # (..., H, N, d, d)
            flags.append(bad_spread)
            if Rot is not None:
                R_prime = _symmetrize(torch.einsum("nts,...hntu,nuv->...hnsv", Rot, R_prime, Rot))
            R_prime = torch.where(no_record[..., None, None], torch.zeros((), dtype=dtype, device=xi.device), R_prime)
            label_spread_out = R_prime.movedim(-4, -3)                          # (..., N, H, d, d)
            spread_gains = _spread_gains(X0, S, gains, M)                       # (..., N, H, K, d) P S_h^T
        elbo_out = adf_evidence_out = evidence_out = is_evidence_out = None
        if elbo:
            with torch.no_grad():
                # q(X) is the Gaussian the readout returns: P_i plus the label spread under spread_corrected.
                elbo_out = minimal_agent_elbo(xi, C, xi_plus, gains, M, V_Q, S, b, B, lp, weights_out, no_record,
                                              blocks=blocks, rope=Rot, spread_gains=spread_gains,
                                              label_spread=label_spread_out)    # (..., N)
                C_q = _posterior_covariance(_dense_covariance(C.detach().to(torch.float64), blocks),
                                            gains.detach().to(torch.float64), M.detach().to(torch.float64),
                                            None if spread_gains is None else spread_gains.detach(),
                                            None if label_spread_out is None else label_spread_out.detach())
                is_evidence_out = minimal_agent_importance_evidence(
                    xi, C, xi_plus, C_q, S, b, B, lp, no_record, blocks=blocks, rope=Rot)   # (..., N)
                exact_cost = cost_out
                if inference != "exact" or dtype != torch.float64:
                    # The ADF costs in float64 from the detached inputs (the exact path of this call when it already
                    # ran in float64).
                    f64 = torch.float64
                    exact = minimal_agent_readout(xi.detach().to(f64), C.detach().to(f64), S.detach().to(f64),
                                                  T.detach().to(f64), log_noise.detach().to(f64), lp.detach().to(f64),
                                                  "exact", check=False,
                                                  rope=None if Rot is None else Rot.detach().to(f64))
                    exact_cost = exact.cost.masked_fill(exact.failed, math.nan)
                adf_evidence_out = minimal_agent_log_evidence(exact_cost, lp, no_record, d)   # (..., N)
                # ADF is exact where S_h C_i S_g^T = 0 for every h != g (module docstring, Exactness).
                uncoupled = (coupling if coupling is not None else _head_coupling(S, X0)).amax(dim=-1) == 0   # (..., N)
                evidence_out = adf_evidence_out.masked_fill(~uncoupled, math.nan)
        failed = torch.stack([flag.reshape(()) for flag in flags]).any()
        ill_conditioned = (torch.stack([screen.reshape(()) for screen in screens]).any() if screens
                           else torch.zeros((), dtype=torch.bool, device=xi.device))
    if check and bool(failed):                                                  # the one host check
        raise ValueError(readout_failure_message(dtype))
    return MinimalAgentResult(xi_plus, gains, M, weights_out, cost_out, gain_fraction, V_Q, V_K, coupling,
                              failed, ill_conditioned, contraction, staleness, spread, mstep,
                              spread_gains, label_spread_out, elbo_out, adf_evidence_out, evidence_out,
                              is_evidence_out, mstep_boundary)


# ----------------------------------------------------------------------------------------------
# Local coordinates
# ----------------------------------------------------------------------------------------------

def local_posterior_covariance(
    sigma:          torch.Tensor,               # (..., N, K) diagonal or (..., N, K, K) full input Sigma_i
    forward:        torch.Tensor,               # (..., N, G, g, g) compact or (..., N, K, K) dense U_i
    block_diagonal: bool,
    gains:          torch.Tensor,               # (..., N, H, K, d) Kmat_i^h
    M:              torch.Tensor,               # (..., N, H, d, d) M_i^h
    floor:          torch.Tensor,               # sigma's layout: f Sigma_{p,i}, the prior-anchored floor
    spread_gains:   Optional[torch.Tensor] = None,   # (..., N, H, K, d) P_i S_h^T (spread_corrected only)
    spread:         Optional[torch.Tensor] = None,   # (..., N, H, d, d) R'_i^h (spread_corrected only)
) -> torch.Tensor:                              # same layout as sigma
    r"""Local posterior covariance :math:`\Sigma_i^+ = U_iC_i^+U_i^\top + f\,\Sigma_{p,i}`.

    .. math::
        \Sigma_i^+ = \Sigma_i - \sum_hZ_i^hM_i^hZ_i^{h\top} + f\,\Sigma_{p,i},\qquad
        Z_i^h = U_i\mathsf K_i^h\ (K\times d),

    using :math:`\Sigma_i = U_iC_iU_i^\top` exactly, so the input covariance is not recomputed. The floor
    :math:`f\,\Sigma_{p,i}` (``floor``, formed by the caller) is the token's prior covariance in local coordinates: it
    involves no frame, so the result is exactly invariant under a common reference change and covariant under a site
    action, and it does not compound with depth (audit 2026-10-12, item 4 and T5). A diagonal input returns the
    diagonal (the moment projection onto the diagonal family),
    :math:`\operatorname{diag}\Sigma_i - \sum_h\operatorname{rowquad}(Z_i^h, M_i^h) + f\,\operatorname{diag}\Sigma_{p,i}`
    with :math:`\operatorname{rowquad}(Z, M)_k = Z_kMZ_k^\top`, which never forms a :math:`K\times K` matrix.

    With ``spread_gains`` :math:`P_iS_h^\top` and ``spread`` :math:`R'^h_i` (``spread_corrected``, module docstring T2)
    the term :math:`\sum_hZ'^h_iR'^h_iZ'^{h\top}_i`, :math:`Z'^h_i = U_iP_iS_h^\top`, is added (its row quadratic forms on
    the diagonal family), which is :math:`U_i(\sum_hP_iS_h^\top R'^h_iS_hP_i)U_i^\top`. Without them the arithmetic is
    the moment-matched one, operation for operation.
    """
    if floor.shape != sigma.shape:
        raise ValueError(f"the floor must have the covariance's shape {tuple(sigma.shape)}, got {tuple(floor.shape)}")
    if (spread_gains is None) != (spread is None):
        raise ValueError("spread_gains and spread are given together (spread_corrected) or not at all")
    H, K, d = gains.shape[-3:]

    def local(X: torch.Tensor) -> torch.Tensor:                                 # (..., N, H, K, d) -> U_i X
        if block_diagonal:
            G, g = forward.shape[-3], forward.shape[-1]
            return torch.einsum("...nmab,...nhmbs->...nhmas", forward,
                                X.reshape(*X.shape[:-2], G, g, d)).reshape(X.shape)
        return forward.unsqueeze(-3) @ X

    Z = local(gains)                                                            # (..., N, H, K, d)
    if sigma.dim() == gains.dim() - 2:                                          # diagonal (..., N, K)
        downdate = torch.einsum("...nhks,...nhst,...nhkt->...nk", Z, M, Z)       # sum_h rowquad(Z^h, M^h)
        if spread is None:
            return sigma - downdate + floor
        Zs = local(spread_gains)                                                # (..., N, H, K, d) U_i P_i S_h^T
        return sigma - downdate + torch.einsum("...nhks,...nhst,...nhkt->...nk", Zs, spread, Zs) + floor
    downdate = torch.einsum("...nhks,...nhst,...nhlt->...nkl", Z, M, Z)          # (..., N, K, K)
    if spread is None:
        return _symmetrize(sigma - downdate + floor)
    Zs = local(spread_gains)
    return _symmetrize(sigma - downdate + torch.einsum("...nhks,...nhst,...nhlt->...nkl", Zs, spread, Zs) + floor)
