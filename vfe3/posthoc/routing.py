r"""M4 routing-score decomposition (``fig:placeholder_routing_decomposition`` of GL(K)_attention_v2).

The object is the evaluated router of ``attention_mode='projected_predictive'`` with
``relational_routing='gaussian_kl'``. Fix a layer and a head (indices suppressed). With the layer-input
belief :math:`(\mu_i, \Sigma_i)`, the head block of the realized routing frame inverse :math:`U_i^{-1}`
(token and positional frames, and any folded RoPE rotation) and the comparison maps :math:`S_Q, S_K`,
the projected moments of Eq. (eq:projected_qk_moments) are

.. math::
    a_i = S_Q U_i^{-1}\mu_i, \quad b_j = S_K U_j^{-1}\mu_j, \quad
    V_{Q,i} = S_Q C_i S_Q^\top, \quad V_{K,j} = S_K C_j S_K^\top, \quad
    C_i = U_i^{-1}\Sigma_i U_i^{-\top},

and the model's routing logit is :math:`\ell_{ij} = B_{ij} - \bar D_{ij}/\tau_h` with the attention
log-prior :math:`B_{ij}` (``-inf`` off the causal support :math:`\mathcal S_i`), the temperature
:math:`\tau_h = \kappa_h\sqrt{d_h}`, the Gaussian KL :math:`D_{ij}` of Eq. (eq:projected_full_kl) and
the capped cost :math:`\bar D_{ij} = \min(\max(D_{ij}, 0), \kappa)` (``safe_kl_clamp``, non-finite
costs set to :math:`\kappa` = ``kl_max``). Everything is recomputed in float64 from the exact inputs the
model hands to :func:`vfe3.relational_scoring.projected_moments`, intercepted during an ordinary
forward (no model code is changed).

**Whitening.** :math:`V_*` is the pooled projected key covariance, the arithmetic mean of
:math:`V_{K,j}` over every source position with a real input token in every evaluated window (a first
pass). With the symmetric inverse square root, :math:`\mathsf q_i = V_*^{-1/2}a_i`,
:math:`\mathsf k_j = V_*^{-1/2}b_j`, and :math:`P_j = V_{K,j}^{-1}`, :math:`\Delta_j = P_j - V_*^{-1}`,
:math:`\delta_{ij} = a_i - b_j`.

**Panel A (logit components).** Expanding Eq. (eq:projected_full_kl) about :math:`V_*` gives, exactly,

.. math::
    \ell_{ij} = \underbrace{\frac{\mathsf q_i^\top\mathsf k_j}{\tau}}_{\text{cross}}
    \underbrace{-\frac{\|\mathsf k_j\|^2}{2\tau}}_{\text{key norm}}
    \underbrace{-\frac{\operatorname{Tr}(P_jV_{Q,i})}{2\tau}}_{\text{trace}}
    \underbrace{-\frac{\log\det V_{K,j}}{2\tau}}_{\text{log-det}}
    \underbrace{-\frac{\delta_{ij}^\top\Delta_j\delta_{ij}}{2\tau}}_{\text{precision}}
    + \underbrace{B_{ij}}_{\text{prior}}
    \underbrace{-\frac{\bar D_{ij} - D_{ij}}{\tau}}_{\text{cap}}
    + \underbrace{\frac{r + \log\det V_{Q,i} - \|\mathsf q_i\|^2}{2\tau}}_{\text{row constant}}.

The first four and the prior are the terms the manuscript names. The *precision* term is the
source-dependent part of the quadratic :math:`\delta_{ij}^\top P_j\delta_{ij}` beyond its pooled
whitened form (it vanishes, like the source dependence of the trace and log-det terms, under the
shared-covariance hypothesis :math:`V_{K,j} = V_*` of Proposition prop:projected_dot_product); *cap*
is zero wherever :math:`0 \le D_{ij} \le \kappa`; the *row constant* cancels in the softmax. The
decomposition is verified per batch against :func:`~vfe3.relational_scoring.projected_gaussian_kl`
in float64 and against the model's captured costs and routing weights (``checks``).

Statistics run over the causal entries :math:`(i, j)`, :math:`j \in \mathcal S_i`, of every query row
with a real input token and :math:`|\mathcal S_i| \ge 2`, uniformly weighted. With the total
:math:`\ell = \sum_c x_c` and :math:`\tilde x_c` a component centered within its row,

.. math::
    \mathrm{share}_c = \frac{\operatorname{Cov}(\tilde x_c, \tilde\ell)}{\operatorname{Var}(\tilde\ell)},
    \qquad \sum_c \mathrm{share}_c = 1,

since :math:`\tilde\ell = \sum_c\tilde x_c`. Within-row centering is the primary statistic because
the softmax sees only within-row differences (the row constant has zero within-row share). The same
decomposition over entries without centering (``*_raw``) is also reported.

**Panel B (bias error and the softmax bound).** Per row,
:math:`\varepsilon_i = (\max_{j\in\mathcal S_i}\|\mathsf k_j\|^2 - \min_{j\in\mathcal S_i}\|\mathsf k_j\|^2)/(4\tau)`
(Eq. eq:projected_bias_error at the midrange row constant), the dot-product routing
:math:`p_{ij} = \operatorname{softmax}_j(\mathsf q_i^\top\mathsf k_j/\tau + B_{ij})`, the shared-covariance
Gaussian routing :math:`\beta^{\mathrm{sh}}_{ij} = \operatorname{softmax}_j(\mathsf q_i^\top\mathsf k_j/\tau
- \|\mathsf k_j\|^2/(2\tau) + B_{ij})` (Eq. eq:main_dot_product_scope with :math:`V_*`) and the model's
routing :math:`\beta_{ij} = \operatorname{softmax}_j(\ell_{ij})`. The bound of Eq.
(eq:projected_softmax_bound), :math:`2\tanh(\varepsilon_i/2)`, is a theorem for
:math:`\ell_1(\beta^{\mathrm{sh}}_i, p_i)` (``max_shared_excess`` checks it); for the trained router
:math:`\ell_1(\beta_i, p_i)` it holds only if the model met the proposition's hypotheses, so the
fraction of rows within it measures the distance from them. :math:`\ell_1(\beta_i, \beta^{\mathrm{sh}}_i)`
isolates the source-dependent covariance terms and the cap. For :math:`\varepsilon_i \gtrsim 4` the bound
approaches 2, the largest possible :math:`\ell_1`, and is vacuous, so the within-bound fraction is
informative only together with the distribution of :math:`\varepsilon_i`. Distributions use a uniform sample of at
most :data:`ROW_SAMPLE_SIZE` rows per layer and head: each (window, layer, head, row) gets a key from
``numpy.random.default_rng([ROW_SAMPLE_SEED, window])`` and the smallest keys are kept, so the sample
depends on the seed and the windows but not on the batch size. Means, maxima and within-bound
fractions are exact over all rows.

**Panel C (key-covariance dispersion).** The dispersion is the log-det spread about the arithmetic
mean,

.. math::
    \delta = \log\det\bar V - \frac1n\sum_j\log\det V_{K,j}
           = \frac2n\sum_j D_{\mathrm{KL}}\bigl(\mathcal N(0,V_{K,j})\,\|\,\mathcal N(0,\bar V)\bigr) \ge 0,
    \qquad \bar V = \frac1n\sum_j V_{K,j},

over the pooled sources, zero exactly when all :math:`V_{K,j}` are equal. It is invariant under the
comparison-basis redundancy :math:`S_K \mapsto AS_K` (congruence :math:`V \mapsto AVA^\top`), the only
freedom left in the gauge-invariant reference moments. It is evaluated for four substitutions in the
key moments: ``model`` (as trained); ``sigma_mean`` (:math:`\Sigma_j \to \bar\Sigma`, the pooled
token average of the layer-input covariance head block: only the frames vary); ``sigma_identity``
(:math:`\Sigma_j \to I`: frames only, isotropic covariance); ``frames_identity`` (:math:`U_j^{-1} \to I`:
only the covariances vary). The frame share is

.. math::
    s_U = \frac{\delta_{\bar\Sigma}}{\delta_{\bar\Sigma} + \delta_{U=I}},

the dispersion left when only the frames vary relative to the sum of that and the dispersion left when
only the stored (layer-input) covariances vary; :math:`\delta_{\mathrm{model}} - \delta_{\bar\Sigma} -
\delta_{U=I}` is reported as the interaction, since :math:`\delta` is not additive over the two
factors. Also reported for the trained keys: the mean squared affine-invariant distance to the pooled
covariance, :math:`\frac1n\sum_j\|\log(V_*^{-1/2}V_{K,j}V_*^{-1/2})\|_F^2` (congruence invariant), and
the condition number of :math:`V_*`.

Skips (``status='skipped'`` with ``reason``): any route whose router is not this Gaussian KL
(``attention_mode`` other than ``projected_predictive``, ``relational_routing='noisy_query'``, no
E-step iteration, or no intercepted routing call).
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

COMPONENTS: Tuple[str, ...] = ("cross", "key_norm", "trace", "logdet", "precision", "prior", "cap", "row")
COMPONENT_LABELS: Dict[str, str] = {
    "cross":     r"$\mathsf{q}_i^\top\mathsf{k}_j/\tau$",
    "key_norm":  r"$-\|\mathsf{k}_j\|^2/2\tau$",
    "trace":     r"$-\mathrm{Tr}(P_jV_{Q,i})/2\tau$",
    "logdet":    r"$-\log\det V_{K,j}/2\tau$",
    "precision": r"$-\delta_{ij}^\top\Delta_j\delta_{ij}/2\tau$",
    "prior":     r"$B_{ij}$",
    "cap":       "cap",
    "row":       "row const.",
}
VARIANTS: Tuple[str, ...] = ("model", "sigma_mean", "sigma_identity", "frames_identity")
VARIANT_LABELS: Dict[str, str] = {
    "model": "trained", "sigma_mean": r"$\Sigma\to\bar\Sigma$", "sigma_identity": r"$\Sigma\to I$",
    "frames_identity": r"$U\to I$"}
SAMPLE_FIELDS: Tuple[str, ...] = ("epsilon", "bound", "l1_model", "l1_shared", "l1_model_shared", "n_sources")
ROW_SAMPLE_SEED: int = 0
ROW_SAMPLE_SIZE: int = 4096          # sampled query rows kept per (layer, head) for the distributions
BOUND_SLACK: float = 1e-9            # absolute slack on l1 <= 2 tanh(eps/2) (float64 softmax roundoff)
FIGURE_POINTS_PER_LAYER: int = 3000  # sampled rows drawn per layer in the panel-B scatter


class _Unsupported(RuntimeError):
    r"""The forward did not expose the projected Gaussian router in the expected form (recorded skip)."""


# ----------------------------------------------------------------------------------------------
# Applicability
# ----------------------------------------------------------------------------------------------

def not_applicable_reason(model: torch.nn.Module) -> Optional[str]:
    r"""Why the projected Gaussian cost is not this model's router, or ``None`` when it is."""
    cfg = model.cfg
    mode = getattr(cfg, "attention_mode", None)
    if mode != "projected_predictive":
        return (f"attention_mode={mode!r}: the decomposition is defined for the projected predictive "
                "router (attention_mode='projected_predictive'), whose routing costs are the Gaussian KL "
                "between projected query and key moments")
    operators = [op for op in (getattr(model, "relational_operators_by_layer", None) or ()) if op is not None]
    if not operators:
        return "the model has no relational operators, so no projected Gaussian routing is run"
    if any(getattr(op, "predictive_readout", "kernel") == "minimal_agent" for op in operators):
        return ("predictive_readout='minimal_agent': the router is the exact source posterior "
                "softmax_j(log pi_ij + log N(T xi_j; S xi_i, W_ij)) at temperature 1 with no clamp (or its KL label "
                "step), not the clamped Gaussian KL of Eq. (projected_full_kl)")
    routes = sorted({str(getattr(op, "routing", "gaussian_kl")) for op in operators})
    if routes != ["gaussian_kl"]:
        return (f"relational_routing={routes}: the noisy-query cost -log N(a_i; b_j, V_Q + V_K) and the "
                "mean-distance cost ||a_i - b_j||^2/2 are not the Gaussian KL of Eq. (projected_full_kl)")
    readouts = sorted({str(op.readout_mode) for op in operators})
    if readouts != ["predictive_content"]:
        return f"readout_mode={readouts}: routing is not the predictive-content readout"
    if int(getattr(cfg, "n_e_steps", 1)) < 1:
        return "n_e_steps=0: no E-step iteration, so the router is never evaluated"
    if any(getattr(op, "mixed_heads", False) for op in operators):
        return "heads of different sizes (a mixed so_n irrep tower) are not supported by this analysis yet"
    return None


def _skipped(reason: str) -> Dict[str, Any]:
    return {"status": "skipped", "reason": reason}


# ----------------------------------------------------------------------------------------------
# Exact routing inputs from an ordinary forward
# ----------------------------------------------------------------------------------------------

@contextmanager
def _record_routing_inputs() -> Iterator[List[Dict[str, Any]]]:
    r"""Record every ``relational_score`` call and the ``projected_moments`` inputs it makes.

    Both names are rebound in :mod:`vfe3.gradients.relational` (the predictive readout's lookup) and
    restored on exit; values are unchanged. Each record holds ``moments`` (the
    ``(mu_h, Sigma_h, U^{-1} blocks, S)`` arguments, detached) and ``cost`` (the returned raw costs).
    """
    from vfe3.gradients import relational

    original_score, original_moments = relational.relational_score, relational.projected_moments
    calls: List[Dict[str, Any]] = []
    active: List[Dict[str, Any]] = []

    def moments(mu_h: torch.Tensor, Sigma_h: torch.Tensor, inverse_blocks: torch.Tensor,
                S: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if active:
            active[-1]["moments"].append(tuple(x.detach() for x in (mu_h, Sigma_h, inverse_blocks, S)))
        return original_moments(mu_h, Sigma_h, inverse_blocks, S)

    def score(*args: Any, **kwargs: Any) -> torch.Tensor:
        record: Dict[str, Any] = {"moments": []}
        calls.append(record)
        active.append(record)
        try:
            out = original_score(*args, **kwargs)
        finally:
            active.pop()
        record["cost"] = out.detach()
        return out

    relational.relational_score, relational.projected_moments = score, moments
    try:
        yield calls
    finally:
        relational.relational_score, relational.projected_moments = original_score, original_moments


def _layer_inputs(calls: Sequence[Dict[str, Any]], n_layers: int) -> List[Dict[str, Any]]:
    r"""Per layer, the query and key ``(mu_h, Sigma_h, U^{-1}, S)`` inputs and the model's raw costs.

    With ``n_e_steps`` readouts per block the last one is kept, as in the model's per-layer capture.
    """
    if not calls or len(calls) % n_layers:
        raise _Unsupported(f"{len(calls)} routing calls intercepted for {n_layers} layers")
    per_layer = len(calls) // n_layers
    out: List[Dict[str, Any]] = []
    for layer in range(n_layers):
        record = calls[(layer + 1) * per_layer - 1]
        moments = record["moments"]
        if len(moments) == 1 and moments[0][3].ndim == 4 and moments[0][3].shape[0] == 2:
            mu_h, sigma_h, inverse, S = moments[0]
            out.append({"query": (mu_h, sigma_h, inverse, S[0]), "key": (mu_h, sigma_h, inverse, S[1]),
                        "stacked": True, "cost": record["cost"]})
        elif len(moments) == 2 and all(m[3].ndim == 3 for m in moments):
            out.append({"query": moments[0], "key": moments[1], "stacked": False, "cost": record["cost"]})
        else:
            raise _Unsupported(f"unexpected projected-moment call structure ({len(moments)} calls)")
    return out


def _forward(model: torch.nn.Module, tokens: torch.Tensor) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    r"""One no-grad forward with the diagnostic capture; returns (per-layer inputs, diagnostic)."""
    capture: Dict[str, Any] = {"diagnostic": {}}
    with torch.no_grad(), _record_routing_inputs() as calls:
        model.forward_beliefs(tokens, return_logits=False, capture=capture)
    diagnostic = capture["diagnostic"]
    layers = _layer_inputs(calls, int(model.cfg.n_layers))
    captures = diagnostic.get("layer_conditional_captures") or []
    if len(captures) != len(layers):
        raise _Unsupported(f"{len(captures)} per-layer routing captures for {len(layers)} layers")
    return layers, diagnostic


def _moments64(
    inputs:  Tuple[torch.Tensor, ...],           # (mu_h, Sigma_h, U^{-1} blocks, S)
    *,
    sigma:   Optional[torch.Tensor] = None,       # substitute for Sigma_h (same shape)
    inverse: Optional[torch.Tensor] = None,       # substitute for the U^{-1} blocks (same shape)
) -> Tuple[torch.Tensor, torch.Tensor]:           # (B, N, H, r), (B, N, H, r, r)
    r"""Projected moments :math:`a = S U^{-1}\mu`, :math:`V = S U^{-1}\Sigma U^{-\top}S^\top` in float64."""
    from vfe3.relational_scoring import projected_moments

    mu_h, sigma_h, inv, S = inputs
    return projected_moments(mu_h.double(), (sigma_h if sigma is None else sigma).double(),
                             (inv if inverse is None else inverse).double(), S.double())


# ----------------------------------------------------------------------------------------------
# Pure computations (tested directly)
# ----------------------------------------------------------------------------------------------

def pooled_whitener(V_star: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""``(V_*^{-1/2}, V_*^{-1}, cond(V_*))`` per head from the symmetric eigendecomposition; V_* (H, r, r)."""
    evals, evecs = torch.linalg.eigh(V_star.double())                       # (H, r), (H, r, r)
    if not bool((evals > 0).all()):
        raise _Unsupported("the pooled projected key covariance is not positive definite")
    W = evecs @ torch.diag_embed(evals.rsqrt()) @ evecs.transpose(-1, -2)    # (H, r, r)
    V_inv = evecs @ torch.diag_embed(evals.reciprocal()) @ evecs.transpose(-1, -2)
    return W, V_inv, evals[..., -1] / evals[..., 0]


def logit_components(
    a_q:       torch.Tensor,                 # (B, N, H, r) projected query means
    V_q:       torch.Tensor,                 # (B, N, H, r, r) projected query covariances
    a_k:       torch.Tensor,                 # (B, N, H, r) projected key means
    V_k:       torch.Tensor,                 # (B, N, H, r, r) projected key covariances
    W:         torch.Tensor,                 # (H, r, r) V_*^{-1/2}
    V_star_inv: torch.Tensor,                # (H, r, r) V_*^{-1}
    log_prior: Optional[torch.Tensor],       # broadcastable to (B, H, N, N); -inf off the support
    tau:       torch.Tensor,                 # (H,) temperatures
    kl_max:    float,
) -> Dict[str, torch.Tensor]:                # each (B, H, N, N) float64, plus "D", "whitened_key_sq"
    r"""The eight components of :math:`\ell_{ij}` (module docstring), the uncapped :math:`D_{ij}`
    they imply, and :math:`\|\mathsf k_j\|^2` (B, H, N).

    Every pair term is a batched matrix product: :math:`\operatorname{Tr}(P_jV_{Q,i}) =
    \operatorname{vec}(V_{Q,i})\cdot\operatorname{vec}(P_j)` and :math:`\delta_{ij}^\top\Delta_j\delta_{ij}
    = \operatorname{vec}(a_ia_i^\top)\cdot\operatorname{vec}(\Delta_j) - 2a_i^\top\Delta_jb_j + b_j^\top\Delta_jb_j`.
    """
    from vfe3.families.base import safe_kl_clamp

    aq, ak = a_q.double().movedim(-2, -3), a_k.double().movedim(-2, -3)     # (B, H, N, r)
    Vq, Vk = V_q.double().movedim(-3, -4), V_k.double().movedim(-3, -4)     # (B, H, N, r, r)
    B, H, N, r = aq.shape
    t = tau.double().to(aq.device).reshape(1, H, 1, 1)
    W, V_star_inv = W.double().to(aq.device), V_star_inv.double().to(aq.device)
    chol_k, info_k = torch.linalg.cholesky_ex(Vk)
    chol_q, info_q = torch.linalg.cholesky_ex(Vq)
    if bool((info_k != 0).any()) or bool((info_q != 0).any()):
        raise _Unsupported("a projected covariance failed its float64 Cholesky factorization")
    P = torch.cholesky_inverse(chol_k)                                       # (B, H, N, r, r)
    logdet_k = 2.0 * chol_k.diagonal(dim1=-2, dim2=-1).log().sum(-1)         # (B, H, N)
    logdet_q = 2.0 * chol_q.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    q = torch.einsum("hrs,bhns->bhnr", W, aq)                                # (B, H, N, r) whitened
    k = torch.einsum("hrs,bhns->bhnr", W, ak)
    k_sq = k.square().sum(-1)                                                # (B, H, N)
    Delta = P - V_star_inv[None, :, None]                                    # (B, H, N, r, r)
    Db = (Delta @ ak.unsqueeze(-1)).squeeze(-1)                              # (B, H, N, r) Delta_j b_j
    quad_aa = (aq.unsqueeze(-1) * aq.unsqueeze(-2)).flatten(-2) @ Delta.flatten(-2).transpose(-1, -2)
    quad = quad_aa - 2.0 * (aq @ Db.transpose(-1, -2)) + (ak * Db).sum(-1).unsqueeze(-2)   # (B, H, N, N)
    comps: Dict[str, torch.Tensor] = {
        "cross":     (q @ k.transpose(-1, -2)) / t,
        "key_norm":  (-0.5 * k_sq / t[..., 0]).unsqueeze(-2).expand(B, H, N, N),
        "trace":     -0.5 * (Vq.flatten(-2) @ P.flatten(-2).transpose(-1, -2)) / t,
        "logdet":    (-0.5 * logdet_k / t[..., 0]).unsqueeze(-2).expand(B, H, N, N),
        "precision": -0.5 * quad / t,
        "row":       (0.5 * (r + logdet_q - q.square().sum(-1)) / t[..., 0]).unsqueeze(-1).expand(B, H, N, N),
    }
    D = -t * sum(comps[c] for c in ("cross", "key_norm", "trace", "logdet", "precision", "row"))
    D_bar = safe_kl_clamp(D, kl_max=kl_max)
    comps["cap"] = -(D_bar - D) / t
    prior = (torch.zeros((), dtype=torch.float64, device=aq.device) if log_prior is None
             else log_prior.double().to(aq.device))
    comps["prior"] = prior.expand(B, H, N, N) if prior.dim() else torch.zeros_like(D)
    comps["D"] = D
    comps["whitened_key_sq"] = k_sq
    return comps


def routing_bound_rows(
    dot_logits:   torch.Tensor,              # (..., N) q_i^T k_j / tau + B_ij
    key_bias:     torch.Tensor,              # (..., N) -||k_j||^2 / (2 tau)
    support:      torch.Tensor,              # (..., N) bool, the row's allowed sources
    model_logits: Optional[torch.Tensor] = None,   # (..., N) the router's logits
) -> Dict[str, torch.Tensor]:                # each (...,)
    r"""Row-wise :math:`\varepsilon_i`, the bound :math:`2\tanh(\varepsilon_i/2)` and :math:`\ell_1` distances.

    :math:`\varepsilon_i` is half the range of the key bias over the support (the midrange row
    constant), :math:`p = \operatorname{softmax}(\text{dot})`, :math:`\beta^{\mathrm{sh}} =
    \operatorname{softmax}(\text{dot} + \text{key bias})`, :math:`\beta = \operatorname{softmax}(\text{model})`.
    """
    neg_inf = torch.full_like(dot_logits, -math.inf)
    p = torch.softmax(torch.where(support, dot_logits, neg_inf), dim=-1)
    shared = torch.softmax(torch.where(support, dot_logits + key_bias, neg_inf), dim=-1)
    top = torch.where(support, -key_bias, neg_inf).amax(dim=-1)
    bottom = torch.where(support, -key_bias, -neg_inf).amin(dim=-1)
    epsilon = 0.5 * (top - bottom)
    out = {"epsilon": epsilon, "bound": 2.0 * torch.tanh(0.5 * epsilon),
           "l1_shared": (shared - p).abs().sum(-1), "n_sources": support.sum(-1).to(dot_logits.dtype)}
    if model_logits is not None:
        beta = torch.softmax(torch.where(support, model_logits, neg_inf), dim=-1)
        out["l1_model"] = (beta - p).abs().sum(-1)
        out["l1_model_shared"] = (beta - shared).abs().sum(-1)
    return out


def logdet_spread(
    sum_V:      torch.Tensor,                # (..., r, r) sum of covariances over n sources
    sum_logdet: torch.Tensor,                # (...,) sum of their log-determinants
    n:          "int | torch.Tensor",
) -> torch.Tensor:                           # (...,)
    r""":math:`\delta = \log\det(\frac1n\sum_jV_j) - \frac1n\sum_j\log\det V_j \ge 0` (zero iff all equal)."""
    n = torch.as_tensor(n, dtype=torch.float64, device=sum_V.device)
    return torch.linalg.slogdet(sum_V.double() / n[..., None, None])[1] - sum_logdet.double() / n


def frame_share(delta_frames: float, delta_covariances: float) -> float:
    r""":math:`s_U = \delta_{\bar\Sigma} / (\delta_{\bar\Sigma} + \delta_{U=I})` (NaN when both vanish)."""
    total = delta_frames + delta_covariances
    return float(delta_frames / total) if total > 0 else float("nan")


# ----------------------------------------------------------------------------------------------
# Streaming accumulation
# ----------------------------------------------------------------------------------------------

def _token_sum(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    r"""Sum of ``x`` (B, N, ...) over the masked tokens (B, N), in float64."""
    weights = mask.to(torch.float64).reshape(*mask.shape, *([1] * (x.dim() - 2)))
    return (x.double() * weights).sum(dim=(0, 1))


def _masked_logdet(V: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    r"""``sum_V``, ``sum_logdet``, ``n`` and ``failed`` per head (H,) over the masked tokens.

    V (B, N, H, r, r) float64, mask (B, N). Tokens whose float64 Cholesky fails are excluded from the
    sums and counted in ``failed`` (no repair).
    """
    chol, info = torch.linalg.cholesky_ex(V)
    ok = mask[..., None] & (info == 0)                                       # (B, N, H)
    logdet = 2.0 * torch.where(ok[..., None], chol.diagonal(dim1=-2, dim2=-1),
                               torch.ones((), dtype=V.dtype, device=V.device)).log().sum(-1)
    weights = ok.to(V.dtype)
    return {"sum_V": (V * weights[..., None, None]).sum(dim=(0, 1)),
            "sum_logdet": (logdet * weights).sum(dim=(0, 1)),
            "n": ok.sum(dim=(0, 1)).double(),
            "failed": (mask[..., None] & (info != 0)).sum(dim=(0, 1)).double()}


class _Reservoir:
    r"""Uniform sample of at most ``size`` rows by smallest random key (priority sampling)."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.keys = np.zeros(0)
        self.values = np.zeros((0, len(SAMPLE_FIELDS)), dtype=np.float32)

    def add(self, keys: np.ndarray, values: np.ndarray) -> None:
        keys = np.concatenate([self.keys, keys])
        values = np.concatenate([self.values, values.astype(np.float32)])
        if keys.size > self.size:
            keep = np.argpartition(keys, self.size - 1)[:self.size]
            keys, values = keys[keep], values[keep]
        self.keys, self.values = keys, values


class _LayerAccumulator:
    r"""Panel A/B/C sums of one layer over all batches (float64, on the CPU)."""

    def __init__(self, H: int, r: int) -> None:
        C = len(COMPONENTS)
        z = lambda *shape: torch.zeros(*shape, dtype=torch.float64)          # noqa: E731
        self.a = {"w_cl": z(C, H), "w_cc": z(C, H), "w_ll": z(H), "r_c": z(C, H), "r_cc": z(C, H),
                  "r_cl": z(C, H), "r_l": z(H), "r_ll": z(H), "entries": z(H), "rows": z(H)}
        self.b = {k: z(H) for k in ("epsilon", "bound", "l1_model", "l1_shared", "l1_model_shared",
                                    "within_bound", "rows")}
        self.b_max = {"epsilon": torch.full((H,), -math.inf, dtype=torch.float64),
                      "shared_excess": torch.full((H,), -math.inf, dtype=torch.float64)}
        self.c = {v: {"sum_V": z(H, r, r), "sum_logdet": z(H), "n": z(H), "failed": z(H)} for v in VARIANTS}
        self.ai = z(H)
        self.checks = {k: torch.zeros(H, dtype=torch.float64) for k in (
            "identity_max_rel_error", "model_cost_max_abs_error", "model_logit_max_abs_error",
            "model_beta_max_abs_error")}
        self.reservoirs = [_Reservoir(ROW_SAMPLE_SIZE) for _ in range(H)]

    def covariances(self, comps: Mapping[str, torch.Tensor], entries: torch.Tensor) -> None:
        r"""Within-row and raw second moments of every component against the total (Panel A)."""
        zero = torch.zeros((), dtype=torch.float64, device=entries.device)
        x = torch.stack([torch.where(entries, comps[c], zero) for c in COMPONENTS])   # (C, B, H, N, N)
        total = x.sum(0)                                                              # (B, H, N, N)
        count = entries.sum(-1).clamp_min(1).to(torch.float64)                        # (B, H, N)
        centered = torch.where(entries, x - (x.sum(-1) / count).unsqueeze(-1), zero)  # (C, B, H, N, N)
        centered_total = centered.sum(0)
        cpu = lambda v: v.cpu()                                                       # noqa: E731
        self.a["w_cl"] += cpu((centered * centered_total).sum(dim=(1, 3, 4)))
        self.a["w_cc"] += cpu(centered.square().sum(dim=(1, 3, 4)))
        self.a["w_ll"] += cpu(centered_total.square().sum(dim=(0, 2, 3)))
        self.a["r_c"] += cpu(x.sum(dim=(1, 3, 4)))
        self.a["r_cc"] += cpu(x.square().sum(dim=(1, 3, 4)))
        self.a["r_cl"] += cpu((x * total).sum(dim=(1, 3, 4)))
        self.a["r_l"] += cpu(total.sum(dim=(0, 2, 3)))
        self.a["r_ll"] += cpu(total.square().sum(dim=(0, 2, 3)))
        self.a["entries"] += cpu(entries.sum(dim=(0, 2, 3)).double())
        self.a["rows"] += cpu(entries.any(-1).sum(dim=(0, 2)).double())

    def bounds(self, rows: Mapping[str, torch.Tensor], row_mask: torch.Tensor, keys: np.ndarray) -> None:
        r"""Exact row sums and maxima (Panel B) and the row sample; ``keys`` (B, H, N) sampling keys."""
        zero = torch.zeros((), dtype=torch.float64, device=row_mask.device)
        for name in ("epsilon", "bound", "l1_model", "l1_shared", "l1_model_shared"):
            self.b[name] += torch.where(row_mask, rows[name], zero).sum(dim=(0, 2)).cpu()
        within = row_mask & (rows["l1_model"] <= rows["bound"] + BOUND_SLACK)
        self.b["within_bound"] += within.sum(dim=(0, 2)).double().cpu()
        self.b["rows"] += row_mask.sum(dim=(0, 2)).double().cpu()
        low = torch.full_like(rows["epsilon"], -math.inf)
        self.b_max["epsilon"] = torch.maximum(
            self.b_max["epsilon"], torch.where(row_mask, rows["epsilon"], low).amax(dim=(0, 2)).cpu())
        self.b_max["shared_excess"] = torch.maximum(
            self.b_max["shared_excess"],
            torch.where(row_mask, rows["l1_shared"] - rows["bound"], low).amax(dim=(0, 2)).cpu())
        fields = torch.stack([rows[f] for f in SAMPLE_FIELDS], dim=-1).cpu().numpy()   # (B, H, N, F)
        mask = row_mask.cpu().numpy()
        for head, reservoir in enumerate(self.reservoirs):
            selected = mask[:, head]
            if selected.any():
                reservoir.add(keys[:, head][selected], fields[:, head][selected])


def _update_max(store: Dict[str, torch.Tensor], name: str, value: torch.Tensor) -> None:
    store[name] = torch.maximum(store[name], value.cpu().double())


# ----------------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------------

def _head_temperatures(model: torch.nn.Module, H: int, device: torch.device) -> torch.Tensor:   # (H,)
    r""":math:`\tau_h = \kappa_h\sqrt{d_h}` exactly as ``vfe_stack`` computes it."""
    from vfe3.free_energy import attention_tau
    from vfe3.model.block import _as_coeff

    tau = attention_tau(_as_coeff(model.cfg.kappa_beta, device), model.group.irrep_dims, device=device)
    tau = torch.as_tensor(tau, dtype=torch.float64, device=device).reshape(-1)
    return tau.expand(H).clone() if tau.numel() == 1 else tau


def _batch_masks(
    dataset: Any, windows: np.ndarray, N: int, device: torch.device,
) -> torch.Tensor:                                   # (B, N) bool, positions holding a real input token
    from vfe3.posthoc.evaluate import window_start

    starts = np.array([window_start(dataset, int(w)) for w in windows], dtype=np.int64)
    stream = getattr(dataset, "tokens", None)
    if stream is None:
        return torch.ones(len(windows), N, dtype=torch.bool, device=device)
    return torch.as_tensor(starts[:, None] + np.arange(N)[None, :] < int(stream.numel()), device=device)


def _support(log_prior: Optional[torch.Tensor], beta: torch.Tensor) -> torch.Tensor:   # (B, H, N, N) bool
    from vfe3.posthoc.klcap import causal_support

    return causal_support(beta, log_prior)


def routing_decomposition(
    model:      torch.nn.Module,
    dataset:    Any,
    indices:    Sequence[int],

    *,
    batch_size: int,
    device:     torch.device,
    progress:   Optional[Callable[[str, int, int], None]] = None,
) -> Dict[str, Any]:
    r"""Panels A-C of M4 over the windows ``indices`` of ``dataset`` (module docstring for definitions).

    Two streaming passes over the windows: the first pools :math:`V_*` and :math:`\bar\Sigma` per
    layer and head, the second accumulates the component covariances, the row-wise bound statistics
    with a seeded row sample, and the key-covariance dispersions. Memory is one batch of pair tensors
    plus fixed-size sums and samples. Returns ``status='skipped'`` with ``reason`` when the projected
    Gaussian cost is not the router. ``samples`` holds numpy arrays; the other keys are JSON-ready.
    """
    from vfe3.posthoc.evaluate import iter_batches
    from vfe3.relational_scoring import projected_gaussian_kl

    reason = not_applicable_reason(model)
    if reason is not None:
        return _skipped(reason)
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        return _skipped("no evaluation windows were given")
    cfg = model.cfg
    L = int(cfg.n_layers)
    kl_max = float(cfg.kl_max)
    n_batches = math.ceil(indices.size / batch_size)

    # Pass 1: pooled projected key covariance V_* and layer-input covariance Sigma-bar.
    pooled: List[Dict[str, Any]] = [{} for _ in range(L)]
    try:
        for batch_index, (windows, inputs, _) in enumerate(iter_batches(dataset, indices, batch_size)):
            layers, _ = _forward(model, inputs.to(device))
            valid = _batch_masks(dataset, windows, inputs.shape[1], device)
            for layer, record in enumerate(layers):
                _, V_k = _moments64(record["key"])                                  # (B, N, H, r, r)
                sigma_h = record["key"][1]
                acc = pooled[layer]
                if not acc:
                    acc.update(sum_V=torch.zeros(V_k.shape[-3:], dtype=torch.float64, device=device),
                               sum_sigma=torch.zeros(sigma_h.shape[2:], dtype=torch.float64, device=device),
                               n=0, sigma_shape=tuple(sigma_h.shape[2:]))
                acc["sum_V"] += _token_sum(V_k, valid)
                acc["sum_sigma"] += _token_sum(sigma_h, valid)
                acc["n"] += int(valid.sum())
            if progress is not None:
                progress("pooled covariance", batch_index + 1, n_batches)
    except _Unsupported as error:
        return _skipped(str(error))

    H, r = pooled[0]["sum_V"].shape[0], pooled[0]["sum_V"].shape[-1]
    tau = _head_temperatures(model, H, device)
    whiteners = []
    try:
        for acc in pooled:
            W, V_inv, cond = pooled_whitener(acc["sum_V"] / acc["n"])
            whiteners.append((W, V_inv, cond, acc["sum_sigma"] / acc["n"]))
    except _Unsupported as error:
        return _skipped(str(error))

    # Pass 2: components, bound statistics, dispersions and checks.
    accs = [_LayerAccumulator(H, r) for _ in range(L)]
    coverage = {"windows": int(indices.size), "batches": 0, "sources": 0}
    try:
        for batch_index, (windows, inputs, _) in enumerate(iter_batches(dataset, indices, batch_size)):
            layers, diagnostic = _forward(model, inputs.to(device))
            B, N = inputs.shape
            valid = _batch_masks(dataset, windows, N, device)
            log_prior = diagnostic.get("log_prior")
            keys = np.stack([np.random.default_rng([ROW_SAMPLE_SEED, int(w)]).random((L, H, N))
                             for w in windows], axis=0)                              # (B, L, H, N)
            coverage["batches"] += 1
            coverage["sources"] += int(valid.sum())
            for layer, (record, capture) in enumerate(zip(layers, diagnostic["layer_conditional_captures"])):
                W, V_inv, _, sigma_bar = whiteners[layer]
                acc = accs[layer]
                if record["stacked"]:
                    a, V = _moments64(record["query"][:3] + (torch.stack((record["query"][3],
                                                                          record["key"][3])),))
                    (a_q, a_k), (V_q, V_k) = a.unbind(0), V.unbind(0)
                else:
                    a_q, V_q = _moments64(record["query"])
                    a_k, V_k = _moments64(record["key"])
                beta_model = capture["beta_pre"].double()                            # (B, H, N, N)
                neg_inf = torch.full_like(beta_model, -math.inf)
                support = _support(log_prior, beta_model)                            # (B, H, N, N)
                row_mask = valid[:, None, :] & (support.sum(-1) >= 2)                # (B, H, N)
                entries = support & row_mask.unsqueeze(-1)                           # (B, H, N, N)
                comps = logit_components(a_q, V_q, a_k, V_k, W, V_inv, log_prior, tau, kl_max)
                logits = sum(comps[c] for c in COMPONENTS)                           # (B, H, N, N)
                acc.covariances(comps, entries)
                rows = routing_bound_rows(comps["cross"] + comps["prior"], comps["key_norm"], support,
                                          model_logits=logits)
                acc.bounds(rows, row_mask, keys[:, layer])
                # Checks: the decomposition against the scoring function (float64) and the model.
                D_ref = projected_gaussian_kl(a_q, V_q, a_k, V_k)                    # (B, H, N, N)
                t = tau.reshape(1, H, 1, 1)
                zero = torch.zeros((), dtype=torch.float64, device=entries.device)
                model_cost = record["cost"].double()
                model_logit = (-capture["score_energy_pre"].double() / t
                               + (zero if log_prior is None else log_prior.double()))
                uncapped = entries & (model_cost <= kl_max) & (model_cost >= 0)
                for name, error in (
                        ("identity_max_rel_error", torch.where(entries, (comps["D"] - D_ref).abs()
                                                                / D_ref.abs().clamp_min(1.0), zero)),
                        ("model_cost_max_abs_error", torch.where(uncapped, (comps["D"] - model_cost).abs(), zero)),
                        ("model_logit_max_abs_error", torch.where(entries, (logits - model_logit).abs(), zero)),
                        ("model_beta_max_abs_error", torch.where(
                            row_mask.unsqueeze(-1),
                            (torch.softmax(torch.where(support, logits, neg_inf), dim=-1) - beta_model).abs(),
                            zero))):
                    _update_max(acc.checks, name, error.amax(dim=(0, 2, 3)))
                # Panel C: key-covariance dispersion under the substitutions.
                key = record["key"]
                eye_inverse = torch.eye(key[2].shape[-1], dtype=torch.float64,
                                        device=key[2].device).expand(key[2].shape)
                sigma_mean = sigma_bar.to(key[1].device).expand(key[1].shape)
                sigma_identity = (torch.ones((), dtype=torch.float64, device=key[1].device).expand(key[1].shape)
                                  if key[1].shape == key[0].shape
                                  else torch.eye(key[1].shape[-1], dtype=torch.float64,
                                                 device=key[1].device).expand(key[1].shape))
                variants = {"model": V_k,
                            "sigma_mean": _moments64(key, sigma=sigma_mean)[1],
                            "sigma_identity": _moments64(key, sigma=sigma_identity)[1],
                            "frames_identity": _moments64(key, inverse=eye_inverse)[1]}
                for name, V_var in variants.items():
                    for field, value in _masked_logdet(V_var, valid).items():
                        acc.c[name][field] += value.cpu()
                whitened = torch.einsum("hrs,bnhst,hut->bnhru", W, V_k, W)          # (B, N, H, r, r)
                log_eigs = torch.linalg.eigvalsh(whitened).log()                     # (B, N, H, r)
                acc.ai += _token_sum(log_eigs.square().sum(-1), valid).cpu()
                del comps, logits, rows, D_ref, variants
            if progress is not None:
                progress("decomposition", batch_index + 1, n_batches)
    except _Unsupported as error:
        return _skipped(str(error))

    return _assemble(model, accs, whiteners, pooled, tau, coverage, indices, batch_size, device)


def _assemble(
    model:       torch.nn.Module,
    accs:        Sequence[_LayerAccumulator],
    whiteners:   Sequence[Tuple[torch.Tensor, ...]],
    pooled:      Sequence[Mapping[str, Any]],
    tau:         torch.Tensor,
    coverage:    Dict[str, Any],
    indices:     np.ndarray,
    batch_size:  int,
    device:      torch.device,
) -> Dict[str, Any]:
    r"""Turn the per-layer sums into the result rows."""
    cfg = model.cfg
    panel_a: List[Dict[str, Any]] = []
    panel_b: List[Dict[str, Any]] = []
    panel_c: List[Dict[str, Any]] = []
    checks: List[Dict[str, Any]] = []
    samples: Dict[str, Dict[str, np.ndarray]] = {}
    for layer, acc in enumerate(accs):
        a = {k: v.numpy() for k, v in acc.a.items()}
        H = a["w_ll"].shape[0]
        cond = whiteners[layer][2].cpu().numpy()
        for head in range(H):
            n = a["entries"][head]
            row: Dict[str, Any] = {"layer": layer, "head": head, "n_rows": int(a["rows"][head]),
                                   "n_entries": int(n)}
            var_w = a["w_ll"][head] / n if n else math.nan
            mean_l = a["r_l"][head] / n if n else math.nan
            var_r = a["r_ll"][head] / n - mean_l ** 2 if n else math.nan
            row["logit_var_within"], row["logit_var_raw"] = float(var_w), float(var_r)
            for index, c in enumerate(COMPONENTS):
                cov_w = a["w_cl"][index, head] / n if n else math.nan
                mean_c = a["r_c"][index, head] / n if n else math.nan
                cov_r = a["r_cl"][index, head] / n - mean_c * mean_l if n else math.nan
                row[f"var_within_{c}"] = float(a["w_cc"][index, head] / n) if n else math.nan
                row[f"cov_within_{c}"] = float(cov_w)
                row[f"share_within_{c}"] = float(cov_w / var_w) if var_w > 0 else math.nan
                row[f"var_raw_{c}"] = float(a["r_cc"][index, head] / n - mean_c ** 2) if n else math.nan
                row[f"cov_raw_{c}"] = float(cov_r)
                row[f"share_raw_{c}"] = float(cov_r / var_r) if var_r > 0 else math.nan
            panel_a.append(row)

            n_rows = float(acc.b["rows"][head])
            sample = acc.reservoirs[head].values
            fields = {f: sample[:, i].astype(np.float64) for i, f in enumerate(SAMPLE_FIELDS)}
            samples[f"L{layer}H{head}"] = {f: sample[:, i] for i, f in enumerate(SAMPLE_FIELDS)}

            def quantile(values: np.ndarray, q: float) -> float:
                return float(np.quantile(values, q)) if values.size else math.nan

            b_row: Dict[str, Any] = {"layer": layer, "head": head, "n_rows": int(n_rows),
                                     "rows_sampled": int(sample.shape[0])}
            for name in ("epsilon", "bound", "l1_model", "l1_shared", "l1_model_shared"):
                b_row[f"{name}_mean"] = float(acc.b[name][head] / n_rows) if n_rows else math.nan
            for q in (0.1, 0.5, 0.9):
                b_row[f"epsilon_q{int(q * 100):02d}"] = quantile(fields["epsilon"], q)
                b_row[f"l1_model_q{int(q * 100):02d}"] = quantile(fields["l1_model"], q)
            b_row["epsilon_max"] = float(acc.b_max["epsilon"][head]) if n_rows else math.nan
            b_row["frac_model_within_bound"] = float(acc.b["within_bound"][head] / n_rows) if n_rows else math.nan
            b_row["max_shared_excess"] = float(acc.b_max["shared_excess"][head]) if n_rows else math.nan
            panel_b.append(b_row)

            c_row: Dict[str, Any] = {"layer": layer, "head": head,
                                     "n_sources": int(acc.c["model"]["n"][head]),
                                     "pooled_cond": float(cond[head])}
            for name in VARIANTS:
                c = acc.c[name]
                c_row[f"logdet_spread_{name}"] = float(logdet_spread(
                    c["sum_V"][head], c["sum_logdet"][head], c["n"][head])) if c["n"][head] > 0 else math.nan
            c_row["factorization_failures"] = int(sum(float(acc.c[v]["failed"][head]) for v in VARIANTS))
            c_row["frame_share"] = frame_share(c_row["logdet_spread_sigma_mean"],
                                               c_row["logdet_spread_frames_identity"])
            c_row["interaction"] = (c_row["logdet_spread_model"] - c_row["logdet_spread_sigma_mean"]
                                    - c_row["logdet_spread_frames_identity"])
            n_src = float(acc.c["model"]["n"][head])
            c_row["ai_dispersion_model"] = float(acc.ai[head] / n_src) if n_src else math.nan
            panel_c.append(c_row)

            checks.append({"layer": layer, "head": head,
                           **{k: float(v[head]) for k, v in acc.checks.items()}})
    n_rows_total = int(sum(r["n_rows"] for r in panel_b))
    coverage.update({
        "window_indices": [int(i) for i in indices],
        "rows_per_layer_head": int(panel_b[0]["n_rows"]) if panel_b else 0,
        "entries_per_layer_head": int(panel_a[0]["n_entries"]) if panel_a else 0,
        "rows_total": n_rows_total,
        "entries_total": int(sum(r["n_entries"] for r in panel_a)),
        "population": ("every query row with a real input token and at least two causal sources, in "
                       "every evaluated window (overlapping windows recompute positions at their own context)"),
    })
    worst = {k: max((c[k] for c in checks), default=math.nan) for k in (
        "identity_max_rel_error", "model_cost_max_abs_error", "model_logit_max_abs_error",
        "model_beta_max_abs_error")}
    return {
        "status": "ok",
        "setup": {
            "n_layers": int(cfg.n_layers), "n_heads": len(accs[0].reservoirs), "rank": int(pooled[0]["sum_V"].shape[-1]),
            "input_width": int(pooled[0]["sigma_shape"][1]),
            "covariance": "diagonal" if len(pooled[0]["sigma_shape"]) == 2 else "full",
            "tau": [float(x) for x in tau.cpu()], "kl_max": float(cfg.kl_max),
            "attention_prior": getattr(cfg, "beta_attention_prior", None),
            "input_scope": "full",
            "predictive_covariance": getattr(cfg, "predictive_covariance", None),
            "family": getattr(cfg, "family", None),
            "model_dtype": str(next(model.parameters()).dtype).replace("torch.", ""),
            "analysis_dtype": "float64", "device": str(device), "batch_size": int(batch_size),
        },
        "coverage": coverage,
        "sampling": {"seed": ROW_SAMPLE_SEED, "rows_per_layer_head_max": ROW_SAMPLE_SIZE,
                     "rule": "smallest keys of numpy.random.default_rng([seed, window]).random((L, H, N))"},
        "definitions": {
            "whitening": "V_* = arithmetic mean of V_K,j over all pooled sources, per layer and head; "
                         "q_i = V_*^{-1/2} a_i, k_j = V_*^{-1/2} b_j (symmetric inverse square root)",
            "panel_a": "share_c = Cov(x_c, l) / Var(l) over causal entries of rows with >= 2 sources; "
                       "'within' centers each component within its row, 'raw' does not; shares sum to one",
            "panel_b": "epsilon_i = range_j ||k_j||^2 / (4 tau); bound = 2 tanh(epsilon_i / 2); l1_shared "
                       "compares the shared-V_* Gaussian routing with dot-product routing (bound is a theorem); "
                       "l1_model compares the model's routing with dot-product routing",
            "panel_c": "logdet_spread = logdet(mean V) - mean logdet V over pooled sources; frame_share = "
                       "spread(Sigma->token mean) / (spread(Sigma->token mean) + spread(U->I)); interaction = "
                       "spread(model) - spread(Sigma->mean) - spread(U->I); ai_dispersion = mean ||log(V_*^-1/2 V "
                       "V_*^-1/2)||_F^2",
        },
        "components": [{"key": c, "label": COMPONENT_LABELS[c]} for c in COMPONENTS],
        "panel_a": panel_a,
        "panel_b": panel_b,
        "panel_c": panel_c,
        "checks": checks,
        "checks_worst": worst,
        "samples": samples,
    }


# ----------------------------------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------------------------------

def _fmt(value: Any, digits: int = 3) -> str:
    from vfe3.semantics.common import format_value

    if isinstance(value, float) and math.isfinite(value) and value != 0.0 and (
            abs(value) < 10 ** -digits or abs(value) >= 1e5):
        return f"{value:.2e}"
    return format_value(value, digits)


def routing_table(result: Mapping[str, Any]) -> str:
    r"""The M4 markdown: coverage, definitions and the Panel A/B/C and check tables."""
    if result.get("status") != "ok":
        return f"# M4 routing-score decomposition\n\nSkipped: {result.get('reason', 'unknown reason')}\n"
    cov, setup = result["coverage"], result["setup"]
    lines = [
        "# M4 routing-score decomposition",
        "",
        f"Coverage: {cov['windows']} windows, {cov['rows_per_layer_head']:,} query rows and "
        f"{cov['entries_per_layer_head']:,} causal entries per layer and head "
        f"({cov['rows_total']:,} rows, {cov['entries_total']:,} entries over all {setup['n_layers']} layers x "
        f"{setup['n_heads']} heads), {cov['sources']:,} pooled sources per layer and head. Rank r = {setup['rank']}, "
        f"tau = {', '.join(f'{t:.4g}' for t in sorted(set(setup['tau'])))}, kl_max = {setup['kl_max']:g}, "
        f"{setup['covariance']} layer-input covariance, analysis in float64 (model {setup['model_dtype']}).",
        f"Row sample for distributions: at most {result['sampling']['rows_per_layer_head_max']} rows per layer and "
        f"head, seed {result['sampling']['seed']}.",
        f"Worst checks: decomposition vs projected_gaussian_kl (relative) "
        f"{_fmt(result['checks_worst']['identity_max_rel_error'])}, logits vs model "
        f"{_fmt(result['checks_worst']['model_logit_max_abs_error'])}, routing weights vs model "
        f"{_fmt(result['checks_worst']['model_beta_max_abs_error'])}.",
        "",
        "## Panel A: within-row covariance shares of the routing logit",
        "",
        "Share of the within-row variance of the logit carried by each component (Cov(x_c, l) / Var(l), "
        "rows centered); shares sum to one per row of the table.",
        "",
        "| layer | head | " + " | ".join(COMPONENTS) + " | SD(logit) | entries |",
        "|" + "---|" * (len(COMPONENTS) + 4),
    ]
    def share(value: float) -> str:
        text = f"{value:.3f}" if math.isfinite(value) else "–"
        return "0.000" if text == "-0.000" else text

    for row in result["panel_a"]:
        lines.append(f"| {row['layer']} | {row['head']} | "
                     + " | ".join(share(row[f"share_within_{c}"]) for c in COMPONENTS)
                     + f" | {_fmt(math.sqrt(row['logit_var_within']) if row['logit_var_within'] >= 0 else math.nan)}"
                     + f" | {row['n_entries']:,} |")
    lines += [
        "",
        "## Panel B: whitened key-norm bias error and the softmax bound",
        "",
        "epsilon_i = range of ||k_j||^2 over the row / (4 tau); bound 2 tanh(epsilon/2). l1 model: the "
        "trained router against dot-product routing; l1 shared: Gaussian routing with the pooled V_* "
        "against dot-product routing (must satisfy the bound; max excess <= 0).",
        "",
        "| layer | head | rows | eps q10 | eps median | eps q90 | eps max | bound mean | l1 model mean | "
        "l1 shared mean | l1 model-shared mean | within bound | max shared excess |",
        "|" + "---|" * 13,
    ]
    for row in result["panel_b"]:
        lines.append(" | ".join([f"| {row['layer']}", str(row["head"]), f"{row['n_rows']:,}",
                                 _fmt(row["epsilon_q10"]), _fmt(row["epsilon_q50"]), _fmt(row["epsilon_q90"]),
                                 _fmt(row["epsilon_max"]), _fmt(row["bound_mean"]), _fmt(row["l1_model_mean"]),
                                 _fmt(row["l1_shared_mean"]), _fmt(row["l1_model_shared_mean"]),
                                 _fmt(row["frac_model_within_bound"]), _fmt(row["max_shared_excess"])]) + " |")
    lines += [
        "",
        "## Panel C: dispersion of the projected key covariances",
        "",
        "Log-det spread about the arithmetic mean (2 x mean KL to the pooled Gaussian) under the "
        "substitutions; frame share = spread(Sigma -> token mean) / (spread(Sigma -> token mean) + "
        "spread(U -> I)).",
        "",
        "| layer | head | trained | Sigma -> mean | Sigma -> I | U -> I | frame share | interaction | "
        "AI dispersion | cond(V_*) |",
        "|" + "---|" * 10,
    ]
    for row in result["panel_c"]:
        lines.append(" | ".join([f"| {row['layer']}", str(row["head"])]
                                + [_fmt(row[f"logdet_spread_{v}"]) for v in VARIANTS]
                                + [_fmt(row["frame_share"]), _fmt(row["interaction"]),
                                   _fmt(row["ai_dispersion_model"]), _fmt(row["pooled_cond"], 1)]) + " |")
    lines += [
        "",
        "## Checks",
        "",
        "| layer | head | decomposition vs scoring (rel) | cost vs model (abs, uncapped) | logit vs model | "
        "beta vs model |",
        "|---|---|---|---|---|---|",
    ]
    for row in result["checks"]:
        lines.append(f"| {row['layer']} | {row['head']} | {_fmt(row['identity_max_rel_error'])} | "
                     f"{_fmt(row['model_cost_max_abs_error'])} | {_fmt(row['model_logit_max_abs_error'])} | "
                     f"{_fmt(row['model_beta_max_abs_error'])} |")
    return "\n".join(lines) + "\n"


def plot_routing_decomposition(
    result:  Mapping[str, Any],
    out_dir: Path,
    name:    str = "m4_routing_decomposition",
) -> List[Path]:
    r"""The M4 figure: A (component shares and logit SD per layer and head), B (epsilon distribution
    and routing l1 against the bound), C (log-det spread, frame share and substitutions)."""
    import matplotlib.pyplot as plt
    from vfe3.posthoc.figures import MAX_WIDTH_IN, MIN_FONT_PT, _grid, _heatmap, _save, _style, _title

    _style()
    rows_a, rows_b, rows_c = result["panel_a"], result["panel_b"], result["panel_c"]
    pairs = [(int(r["layer"]), int(r["head"])) for r in rows_a]
    layers = sorted({layer for layer, _ in pairs})
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, max(1, len(layers))))
    height_a = max(1.9, 0.105 * len(pairs) + 0.8)
    fig = plt.figure(figsize=(MAX_WIDTH_IN, height_a + 4.3))
    outer = fig.add_gridspec(3, 1, height_ratios=[height_a, 2.15, 2.15], hspace=0.55)

    # Panel A: within-row covariance shares (layer.head x component) and the logit SD.
    grid_a = outer[0].subgridspec(1, 4, width_ratios=[3.6, 0.08, 0.3, 1.0], wspace=0.08)
    ax = fig.add_subplot(grid_a[0])
    shares = np.array([[r[f"share_within_{c}"] for c in COMPONENTS] for r in rows_a], dtype=float)
    limit = float(np.nanmax(np.abs(shares))) if np.isfinite(shares).any() else 1.0
    image = ax.imshow(shares, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto", interpolation="nearest")
    if shares.size <= 400:
        for (i, j), value in np.ndenumerate(shares):
            if np.isfinite(value):
                text = f"{value:.2f}"
                ax.text(j, i, "0.00" if text == "-0.00" else text, ha="center", va="center",
                        fontsize=MIN_FONT_PT, color="white" if abs(value) > 0.6 * limit else "black")
    ax.set_xticks(range(len(COMPONENTS)))
    ax.set_xticklabels([COMPONENT_LABELS[c] for c in COMPONENTS], rotation=25, ha="right")
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([f"{layer}.{head}" for layer, head in pairs])
    ax.set_ylabel("layer.head")
    _title(ax, "A", "within-row covariance share of the routing logit")
    colorbar = fig.colorbar(image, cax=fig.add_subplot(grid_a[1]))
    colorbar.ax.tick_params(labelsize=MIN_FONT_PT)
    ax_sd = fig.add_subplot(grid_a[3])
    sd = np.sqrt(np.clip([r["logit_var_within"] for r in rows_a], 0.0, None))
    ax_sd.barh(range(len(pairs)), sd, color=[colors[layers.index(layer)] for layer, _ in pairs], height=0.75)
    ax_sd.set_ylim(len(pairs) - 0.5, -0.5)
    ax_sd.set_yticks([])
    ax_sd.set_xlabel("within-row SD of logit")

    # Panel B: epsilon distribution and routing l1 against 2 tanh(eps / 2).
    grid_b = outer[1].subgridspec(1, 3, wspace=0.5, width_ratios=[1.0, 1.25, 0.9])
    samples = result.get("samples", {})
    pooled = {layer: {f: np.concatenate([np.asarray(samples.get(f"L{layer}H{head}", {}).get(f, np.zeros(0)))
                                         for pair_layer, head in pairs if pair_layer == layer] or [np.zeros(0)])
                      for f in SAMPLE_FIELDS} for layer in layers}
    ax = fig.add_subplot(grid_b[0])
    positive = [v[v > 0] for v in (pooled[layer]["epsilon"] for layer in layers)]
    for layer, values in zip(layers, positive):
        if values.size:
            x = np.sort(values)
            ax.step(x, np.arange(1, x.size + 1) / x.size, where="post", color=colors[layers.index(layer)],
                    lw=1.0, label=f"layer {layer}")
    if any(v.size for v in positive):
        ax.set_xscale("log")
    ax.set_xlabel(r"$\varepsilon_i$ (whitened key-norm range $/4\tau$)")
    ax.set_ylabel("fraction of rows")
    _title(ax, "B", r"bias error $\varepsilon_i$")
    ax.legend(loc="lower right", handlelength=1.2)
    ax = fig.add_subplot(grid_b[1])
    eps_all = np.concatenate([pooled[layer]["epsilon"] for layer in layers])
    for layer in layers:
        eps = pooled[layer]["epsilon"][:FIGURE_POINTS_PER_LAYER]
        ax.scatter(np.maximum(eps, 1e-6), pooled[layer]["l1_shared"][:FIGURE_POINTS_PER_LAYER], s=1.5,
                   color="0.6", alpha=0.5, lw=0, rasterized=True)
        ax.scatter(np.maximum(eps, 1e-6), pooled[layer]["l1_model"][:FIGURE_POINTS_PER_LAYER], s=1.5,
                   color=colors[layers.index(layer)], alpha=0.5, lw=0, rasterized=True)
    positive_eps = eps_all[eps_all > 0]
    lo, hi = (float(positive_eps.min()), float(positive_eps.max())) if positive_eps.size else (1e-3, 1.0)
    grid = np.logspace(math.log10(lo) - 0.2, math.log10(max(hi, lo * 10)) + 0.2, 200)
    ax.plot(grid, 2.0 * np.tanh(grid / 2.0), color="black", lw=1.0, label=r"$2\tanh(\varepsilon_i/2)$")
    ax.plot(grid, np.minimum(grid, 2.0), color="black", lw=0.8, ls="--", label=r"$\min(\varepsilon_i,2)$")
    ax.scatter([], [], s=6, color="0.6", label="shared $V_*$")
    ax.scatter([], [], s=6, color=colors[0], label="trained")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\varepsilon_i$")
    ax.set_ylabel(r"$\sum_j|\beta_{ij}-p_{ij}|$")
    ax.set_title(r"row $\ell_1$ to dot-product routing", loc="left")
    ax.legend(loc="upper left", handlelength=1.2, borderaxespad=0.2, labelspacing=0.3, fontsize=MIN_FONT_PT)
    ax = fig.add_subplot(grid_b[2])
    _heatmap(ax, fig, _grid(rows_b, "frac_model_within_bound"), "B", "", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title("trained rows within bound", loc="left")

    # Panel C: key-covariance dispersion and the frame share.
    grid_c = outer[2].subgridspec(1, 3, wspace=0.62, width_ratios=[1.0, 1.0, 1.25])
    ax = fig.add_subplot(grid_c[0])
    _heatmap(ax, fig, _grid(rows_c, "logdet_spread_model"), "C", r"log-det spread of $V_{K,j}$",
             cmap="magma", fmt="{:.2g}")
    ax = fig.add_subplot(grid_c[1])
    _heatmap(ax, fig, _grid(rows_c, "frame_share"), "C", "", cmap="cividis", vmin=0.0, vmax=1.0)
    ax.set_title(r"frame share $s_U$", loc="left")
    ax = fig.add_subplot(grid_c[2])
    width = 0.8 / len(VARIANTS)
    variant_colors = ["black", "#0072B2", "#56B4E9", "#E69F00"]
    heights = []
    for index, variant in enumerate(VARIANTS):
        means = np.maximum([np.nanmean([r[f"logdet_spread_{variant}"] for r in rows_c if r["layer"] == layer])
                            for layer in layers], 1e-12)
        heights.extend(means)
        ax.bar(np.arange(len(layers)) + (index - (len(VARIANTS) - 1) / 2) * width, means,
               width=width, color=variant_colors[index], label=VARIANT_LABELS[variant])
    ax.set_yscale("log")
    finite = [h for h in heights if np.isfinite(h)]
    if finite:
        ax.set_ylim(min(finite) / 2.0, max(finite) * 30.0)                   # headroom for the legend
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([str(layer) for layer in layers])
    ax.set_xlabel("layer")
    ax.set_ylabel("spread (head mean)")
    ax.set_title("substitutions", loc="left")
    ax.legend(loc="upper center", ncol=2, handlelength=1.0, columnspacing=0.8, fontsize=MIN_FONT_PT)
    return _save(fig, out_dir, name)


def write_routing_outputs(result: Mapping[str, Any], out_dir: Path) -> List[Path]:
    r"""Write ``m4_routing_decomposition.{md,json}``, the row sample ``.npz`` and the figure (PNG, PDF).

    A skipped result writes only the markdown and JSON stating the reason.
    """
    from vfe3.semantics.common import write_json

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = out_dir / "m4_routing_decomposition.md"
    table.write_text(routing_table(result), encoding="utf-8")
    summary = write_json(out_dir / "m4_routing_decomposition.json",
                         {k: v for k, v in result.items() if k != "samples"})
    written = [table, summary]
    if result.get("status") != "ok":
        return written
    arrays = {f"{key}_{field}": np.asarray(values) for key, fields in result.get("samples", {}).items()
              for field, values in fields.items()}
    npz = out_dir / "m4_routing_samples.npz"
    np.savez_compressed(npz, **arrays)
    written.append(npz)
    written.extend(plot_routing_decomposition(result, out_dir))
    return written
