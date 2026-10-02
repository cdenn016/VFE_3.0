r"""Frozen interventions on a trained model (Table M5, ``tab:placeholder_interventions``, rows 1-12).

Each intervention is a context manager that modifies one input of the trained predictor and restores
it exactly on exit; nothing is retrained. With routing weights
:math:`\beta_{ij}^{(h)} = \operatorname{softmax}_j(\log\pi_{ij}^{(h)} - \bar D_{ij}^{(h)}/\tau_h)`,
reference moments :math:`\xi_i = U_i^{-1}\mu_i`, :math:`C_i = U_i^{-1}\Sigma_i U_i^{-\top}` and frames
:math:`U_i = \exp(X_{x_i})\exp(Y_i)`, the rows are

====  =====================================  ==============================================================
row   key                                    modification
====  =====================================  ==============================================================
1     ``baseline``                           none (final weights)
2     ``routing_costs_removed``              :math:`\beta = \operatorname{softmax}_j(\log\pi_{ij})`: costs zeroed, prior kept
3     ``prior_flattened``                    :math:`\log\pi_{ij} \to 0` on the causal support (the ``-inf`` mask is kept)
4     ``residual_removed``                   :math:`\mu_i + m_i \mapsto m_i` at every layer
5     ``stored_covariances_identity``        :math:`\Sigma_{p,v} \to I` (``sigma_log_embed`` :math:`\to 0`)
6     ``routing_covariances_identity``       :math:`C_i \to I` for every query and key: projected covariances
                                             :math:`S C_i S^\top \to S S^\top`, projected means unchanged
7     ``token_frames_identity``              :math:`\exp(X_v) \to I` (``phi_embed`` :math:`\to 0`), in every role
8     ``positional_frames_identity``         :math:`\exp(Y_i) \to I` (``pos_phi_free`` :math:`\to 0`), in every role
9     ``frames_removed_input_moments``       :math:`\xi_i^{(0)} = U_i^{-1}\mu_{p,x_i} \to \mu_{p,x_i}` only
10    ``frames_removed_routing_covariance``  :math:`C_i = U_i^{-1}\Sigma_{p,x_i}U_i^{-\top} \to \Sigma_{p,x_i}` only
11    ``frames_removed_norm_mlp``            :math:`U_i^{-1}\Phi^{(\ell)}(U_i\,\cdot) \to \Phi^{(\ell)}` only
12    ``frames_removed_decoder``             :math:`W_{\mathrm{out}}U_i\xi_i^{(L)} \to W_{\mathrm{out}}\xi_i^{(L)}` only
====  =====================================  ==============================================================

Rows 9-12 remove the composed frame :math:`U_i` from one role of the reference normal form
(Corollary ``cor:normal_form``, Figure ``fig:placeholder_reference_gauge``) while every other role keeps
the trained frames; :func:`frames_removed` derives the local-coordinate form the model computes.

Table substitutions swap a parameter's ``.data`` for a zero-stride zero tensor and swap the original
storage back on exit, so no table is copied (``phi_embed`` is 1 GB on K160) and the restored model is
bit-identical. Function patches replace every module binding of the patched function and count the
intercepted calls; an arm whose patch intercepts nothing raises :class:`NotApplicable` instead of
reporting a number that describes no intervention.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from vfe3.model.frame_actions import frame_matvec as _frame_matvec
from vfe3.model.frame_actions import reference_variances as _reference_variances


class NotApplicable(RuntimeError):
    r"""The intervention has nothing to act on in this model (recorded, not a failure)."""


@dataclass(frozen=True)
class Intervention:
    r"""One row of Table M5."""

    row:   int
    key:   str
    label: str                                                 # manuscript row text
    apply: Callable[[torch.nn.Module], Any]                    # model -> context manager yielding a record


# ----------------------------------------------------------------------------------------------
# Patching helpers
# ----------------------------------------------------------------------------------------------

@contextmanager
def swap_parameter_data(parameter: torch.Tensor, replacement: torch.Tensor) -> Iterator[None]:
    r"""Temporarily point ``parameter.data`` at ``replacement``; restore the original storage on exit."""
    if tuple(replacement.shape) != tuple(parameter.shape):
        raise ValueError("replacement must have the parameter's shape")
    original = parameter.data
    parameter.data = replacement
    try:
        yield
    finally:
        parameter.data = original


def zeros_view(parameter: torch.Tensor) -> torch.Tensor:
    r"""A zero tensor of ``parameter``'s shape, dtype and device with zero strides (no allocation)."""
    return torch.zeros((), dtype=parameter.dtype, device=parameter.device).expand(parameter.shape)


@contextmanager
def patch_bindings(
    modules:     Sequence[Any],
    name:        str,
    replacement: Callable[..., Any],
) -> Iterator[List[Any]]:
    r"""Rebind ``name`` in every module of ``modules`` that binds the same original object."""
    original = getattr(modules[0], name)
    binders = [m for m in modules if getattr(m, name, None) is original]
    for module in binders:
        setattr(module, name, replacement)
    try:
        yield binders
    finally:
        for module in binders:
            setattr(module, name, original)


def _attention_weight_modules() -> List[Any]:
    from vfe3 import free_energy
    from vfe3.gradients import full_link
    from vfe3.inference import e_step

    return [free_energy, e_step, full_link]


def _relational_score_modules() -> List[Any]:
    from vfe3.gradients import full_link, relational

    return [relational, full_link]


# ----------------------------------------------------------------------------------------------
# Rows 2 and 3: the routing softmax
# ----------------------------------------------------------------------------------------------

@contextmanager
def _routing_arm(model: torch.nn.Module, arm: str) -> Iterator[Dict[str, Any]]:
    from vfe3 import free_energy

    original = free_energy.attention_weights
    record: Dict[str, Any] = {"intercepted_calls": 0}

    def patched(energy: torch.Tensor, *, tau: Any = 1.0, log_prior: Optional[torch.Tensor] = None):
        record["intercepted_calls"] += 1
        if arm == "no_energy":
            return original(torch.zeros_like(energy), tau=tau, log_prior=log_prior)
        keep = None if log_prior is None else torch.where(
            torch.isneginf(log_prior), log_prior, torch.zeros_like(log_prior))
        return original(energy, tau=tau, log_prior=keep)

    if arm == "no_prior" and getattr(model.cfg, "beta_attention_prior", None) in (None, "uniform"):
        raise NotApplicable("the model has no attention log-prior to flatten")
    with patch_bindings(_attention_weight_modules(), "attention_weights", patched):
        yield record
    if record["intercepted_calls"] == 0:
        raise NotApplicable("no routing softmax was intercepted on this route")


def routing_costs_removed(model: torch.nn.Module) -> Any:
    r"""Row 2: :math:`\beta_{ij} = \operatorname{softmax}_j(\log\pi_{ij})`."""
    return _routing_arm(model, "no_energy")


def prior_flattened(model: torch.nn.Module) -> Any:
    r"""Row 3: :math:`\beta_{ij} = \operatorname{softmax}_j(-\bar D_{ij}/\tau)` on the causal support."""
    return _routing_arm(model, "no_prior")


# ----------------------------------------------------------------------------------------------
# Row 4: the attention residual
# ----------------------------------------------------------------------------------------------

@contextmanager
def residual_removed(model: torch.nn.Module) -> Iterator[Dict[str, Any]]:
    r"""Row 4: ``attention_residual = False`` on every layer's operators (:math:`\mu_i + m_i \mapsto m_i`)."""
    layers = list(model.relational_operators_by_layer)
    saved = [bool(op.attention_residual) for op in layers]
    if not any(saved):
        raise NotApplicable("no layer carries an attention residual")
    try:
        for op in layers:
            op.attention_residual = False
        yield {"layers_changed": int(sum(saved))}
    finally:
        for op, value in zip(layers, saved):
            op.attention_residual = value


# ----------------------------------------------------------------------------------------------
# Rows 5, 7, 8: table substitutions
# ----------------------------------------------------------------------------------------------

@contextmanager
def stored_covariances_identity(model: torch.nn.Module) -> Iterator[Dict[str, Any]]:
    r"""Row 5: ``sigma_log_embed`` :math:`\to 0`, so every stored :math:`\Sigma_{p,v} = I`."""
    table = getattr(model.prior_bank, "sigma_log_embed", None)
    if table is None:
        raise NotApplicable("the prior bank has no sigma_log_embed table")
    with swap_parameter_data(table, zeros_view(table)):
        yield {"table": "prior_bank.sigma_log_embed"}


@contextmanager
def token_frames_identity(model: torch.nn.Module) -> Iterator[Dict[str, Any]]:
    r"""Row 7: ``phi_embed`` :math:`\to 0`, so :math:`\exp(X_v) = I` in every role."""
    table = getattr(model.prior_bank, "phi_embed", None)
    if table is None:
        raise NotApplicable("the prior bank has no phi_embed table")
    with swap_parameter_data(table, zeros_view(table)):
        yield {"table": "prior_bank.phi_embed"}


@contextmanager
def positional_frames_identity(model: torch.nn.Module) -> Iterator[Dict[str, Any]]:
    r"""Row 8: positional generators :math:`\to 0`, so :math:`\exp(Y_i) = I` in every role.

    A learned table (``pos_phi_free``) is zeroed; a fixed positional chart is replaced by zero
    coordinates through an instance override of ``_pos_phi_right``.
    """
    table = getattr(model, "pos_phi_free", None)
    if table is not None:
        with swap_parameter_data(table, zeros_view(table)):
            yield {"table": "pos_phi_free"}
        return
    if getattr(model.cfg, "pos_phi", "none") == "none":
        raise NotApplicable("the model has no positional frames")
    bound = model._pos_phi_right

    def zero_right(phi: torch.Tensor) -> Optional[torch.Tensor]:
        coords = bound(phi)
        return None if coords is None else torch.zeros_like(coords)

    model._pos_phi_right = zero_right
    try:
        yield {"table": f"pos_phi={model.cfg.pos_phi!r} coordinates"}
    finally:
        del model._pos_phi_right


# ----------------------------------------------------------------------------------------------
# Row 6: routing covariances
# ----------------------------------------------------------------------------------------------

def _identity_covariance_score(
    mu:           torch.Tensor,
    sigma:        torch.Tensor,
    source_mu:    torch.Tensor,
    source_sigma: torch.Tensor,
    frame:        Any,
    operators:    Any,
    capture:      Optional[Dict[str, Any]] = None,
) -> torch.Tensor:                                   # (..., H, N, N)
    r"""``relational_score`` with every reference covariance :math:`C_i` replaced by :math:`I`.

    The projected means :math:`a = S U_i^{-1}\mu_i` are the trained ones; the projected covariances
    become :math:`G_Q = S_Q S_Q^\top`, :math:`G_K = S_K S_K^\top` for every token. The source tuples
    handed to the KL are ``(1, I, S)``, so a factor or recovery route rebuilds exactly these
    covariances (:math:`X = S \cdot I \cdot 1^{1/2}`) and never re-projects the means.
    """
    from vfe3.relational_scoring import (
        _compute_dtype, projected_gaussian_kl, projected_moments, projected_noisy_query_cost)

    H, d = len(operators.irrep_dims), operators.irrep_dims[0]
    inverse = frame.inverse
    if not frame.block_diagonal:
        inverse = torch.stack([inverse[..., h*d:(h+1)*d, h*d:(h+1)*d] for h in range(H)], dim=-3)
    mu_h, source_mu_h = (x.unsqueeze(-2).expand(*x.shape[:-1], H, H * d) for x in (mu, source_mu))
    sigma_h = (sigma.unsqueeze(-3).expand(*sigma.shape[:-2], H, H * d, H * d)
               if sigma.ndim == mu.ndim + 1 else sigma.unsqueeze(-2).expand(*sigma.shape[:-1], H, H * d))
    S_q, S_k = operators.comparison_pair()                                    # (H, r, w)
    a, _ = projected_moments(mu_h, sigma_h, inverse, torch.stack((S_q, S_k)))  # (2, ..., N, H, r)
    if mu is not source_mu:
        a_k, _ = projected_moments(source_mu_h, sigma_h, inverse, S_k)
        a = torch.stack((a[0], a_k))
    public = _compute_dtype(mu, sigma, source_mu, source_sigma, S_q, S_k)
    dtype = a.dtype
    batch, r = a.shape[1:-1], a.shape[-1]                                     # (..., N, H), rank
    G_q = (S_q.to(dtype) @ S_q.to(dtype).transpose(-1, -2)).expand(*batch, r, r).contiguous()   # (..., N, H, r, r)
    G_k = (S_k.to(dtype) @ S_k.to(dtype).transpose(-1, -2)).expand(*batch, r, r).contiguous()
    ones = torch.ones(mu_h.shape, dtype=dtype, device=mu.device)              # unit diagonal source covariance
    eye = torch.eye(inverse.shape[-1], dtype=dtype, device=mu.device).expand(inverse.shape)
    if capture is not None:
        capture["projected_covariance_pre"] = G_q.detach().to(public)
        capture["projected_key_covariance_pre"] = G_k.detach().to(public)
    score = projected_noisy_query_cost if operators.routing == "noisy_query" else projected_gaussian_kl
    return score(a[0], G_q, a[1], G_k,
                 projection_q=(ones, eye, S_q), projection_k=(ones, eye, S_k)).to(public)


@contextmanager
def routing_covariances_identity(model: torch.nn.Module) -> Iterator[Dict[str, Any]]:
    r"""Row 6: every routing covariance :math:`C_i \to I` (projected covariances :math:`S S^\top`)."""
    record: Dict[str, Any] = {"intercepted_calls": 0}

    def patched(*args: Any, **kwargs: Any) -> torch.Tensor:
        record["intercepted_calls"] += 1
        return _identity_covariance_score(*args, **kwargs)

    with patch_bindings(_relational_score_modules(), "relational_score", patched):
        yield record
    if record["intercepted_calls"] == 0:
        raise NotApplicable("no relational routing score was intercepted on this route")


@contextmanager
def baseline(model: torch.nn.Module) -> Iterator[Dict[str, Any]]:
    r"""Row 1: no modification."""
    yield {}


# ----------------------------------------------------------------------------------------------
# Rows 9-12: frames removed by role
# ----------------------------------------------------------------------------------------------

ROLE_INPUT_MOMENTS:      str = "input_moments"         # row 9
ROLE_ROUTING_COVARIANCE: str = "routing_covariance"    # row 10
ROLE_NORM_MLP:           str = "norm_mlp"              # row 11
ROLE_DECODER:            str = "decoder"               # row 12
FRAME_ROLES: Tuple[str, ...] = (ROLE_INPUT_MOMENTS, ROLE_ROUTING_COVARIANCE, ROLE_NORM_MLP, ROLE_DECODER)
FRAME_DESCRIPTION: str = (
    "composed realized frame U_i = exp(X_{x_i}) exp(Y_i) (token and positional factors together): the "
    "vertex factor the relational operators read")


# _frame_matvec and _reference_variances (imported above from vfe3.model.frame_actions) are shared with
# the training-time reference-coordinate block and decoder (``block_coordinates`` /
# ``decode_coordinates='reference'``), so the M5 patches and the trained arms run one numerics path:
# autocast off, float32 (float64 for an escalated frame), operand dtype out.


class _ConjugatedNorm:
    r"""A mean-only normalization with the frame removed from its conjugation: :math:`x \mapsto U_i N(U_i^{-1}x)`."""

    def __init__(self, inner: Callable[..., torch.Tensor], frame: Any) -> None:
        self.inner, self.frame = inner, frame

    def __call__(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:   # (..., N, K)
        bd = self.frame.block_diagonal
        return _frame_matvec(self.frame.forward, self.inner(_frame_matvec(self.frame.inverse, mu, bd), sigma), bd)


class _ConjugatedFinalNormModule(torch.nn.Module):
    r"""Module form of :class:`_ConjugatedNorm` for a final norm registered as a submodule.

    The wrapped norm is held outside the module tree, so the swap adds no parameter or state key.
    """

    def __init__(self, inner: Callable[..., torch.Tensor], state: Dict[str, Any]) -> None:
        super().__init__()
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_state", state)

    def forward(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return _ConjugatedNorm(self._inner, _current_frame(self._state))(mu, sigma)


def _final_norm_replacement(inner: Callable[..., torch.Tensor], state: Dict[str, Any]) -> Callable[..., torch.Tensor]:
    r"""A replacement of the same kind (module or plain callable), so ``setattr`` swaps it and restores it."""
    if isinstance(inner, torch.nn.Module):
        return _ConjugatedFinalNormModule(inner, state)
    return lambda mu, sigma: _ConjugatedNorm(inner, _current_frame(state))(mu, sigma)


class _ConjugatedBlockMLP:
    r"""A local-coordinate block MLP with the frame removed from its conjugation.

    Mean map :math:`\mu \mapsto U_i F(U_i^{-1}\mu)` with the covariance passed through. The Gaussian-template
    block reads the reference variances :math:`\operatorname{diag}C_i`, the covariance its reference
    conjugate sees.
    """

    def __init__(self, inner: torch.nn.Module, frame: Any, *, template: bool) -> None:
        self.inner, self.frame, self.template = inner, frame, template

    def forward_moments(
        self,
        mu:    torch.Tensor,                               # (..., N, K)
        sigma: torch.Tensor,                               # (..., N, K) or (..., N, K, K), passed through
        *,
        frame_context: Optional[Any] = None,
        hidden_bias:   Optional[torch.Tensor] = None,      # (..., N, w) per-type hidden bias, reference side
    ) -> Any:
        from vfe3.model.block_mlp import BlockMLPMomentResult

        del frame_context
        if hidden_bias is not None and self.template:
            raise ValueError("hidden_bias does not apply to the Gaussian-template block")
        bd = self.frame.block_diagonal
        xi = _frame_matvec(self.frame.inverse, mu, bd)                                     # (..., N, K)
        extra = {} if hidden_bias is None else {"hidden_bias": hidden_bias}
        out = (self.inner.mean_update(xi, _reference_variances(mu, sigma, self.frame)) if self.template
               else self.inner.forward(xi, **extra))
        return BlockMLPMomentResult(_frame_matvec(self.frame.forward, out, bd), sigma, None)


def _current_frame(state: Dict[str, Any]) -> Any:
    frame = state.get("frame")
    if frame is None:
        raise RuntimeError("a frame-role patch ran before the belief stack supplied the realized frame")
    return frame


def _require_normal_form(model: torch.nn.Module) -> None:
    r"""Raise :class:`NotApplicable` unless the model is on the route the four-role normal form describes."""
    cfg = model.cfg
    layers = tuple(getattr(model, "relational_operators_by_layer", ()) or ())
    if not layers or any(getattr(op, "readout_mode", None) != "predictive_content" for op in layers):
        raise NotApplicable(
            "the frame roles are the channels of the reference normal form (Corollary cor:normal_form) of the "
            "projected predictive route; this model has attention_mode="
            f"{getattr(cfg, 'attention_mode', None)!r} without relational predictive content on every layer")
    if getattr(cfg, "pos_rotation", "none") != "none":
        raise NotApplicable(
            f"pos_rotation={cfg.pos_rotation!r} folds a positional rotation into the routing and content "
            "frames, a frame channel outside the four-role normal form")
    if getattr(cfg, "encode_mode", None) != "per_token":
        raise NotApplicable(
            f"encode_mode={getattr(cfg, 'encode_mode', None)!r}: the input-moment role "
            "xi^(0) = U^{-1} mu_p is defined for local-coordinate token tables (encode_mode='per_token')")
    extra = [name for name in ("head_mixer", "cg_coupling") if getattr(model, name, None) is not None]
    if extra:
        raise NotApplicable(
            f"{' and '.join(extra)} act(s) in local coordinates between attention and normalization, a frame "
            "channel outside the four-role normal form")


def _require_static_routing_covariance(model: torch.nn.Module) -> None:
    r"""Row 10 needs :math:`C_i^{(\ell)} = C_i^{(0)}` at every layer (covariance passthrough)."""
    if any(getattr(op, "predictive_covariance", "passthrough") != "passthrough"
           for op in model.relational_operators_by_layer):
        raise NotApplicable(
            "predictive_covariance='moment_matched': the routing covariance of layer l > 0 is the moment-matched "
            "C_i^att, not the static C_i = U_i^{-1} Sigma_p U_i^{-T} of the normal form; removing the frame from "
            "the input covariance alone would need the full local covariance U_i Sigma_p U_i^T as stack input "
            "(and on the diagonal family the frame also enters the diagonal moment projection at every layer)")
    mlps = getattr(model, "block_mlps", None)
    if mlps is not None and any(getattr(m, "covariance_contract", "passthrough") != "passthrough" for m in mlps):
        raise NotApplicable(
            "the block MLP propagates the covariance (covariance_contract != 'passthrough'), so the routing "
            "covariance is not the static C_i of the normal form")


def _norm_plan(model: torch.nn.Module, norm: Any, norm_type: str) -> str:
    r"""``"absent"``, ``"conjugated"`` (mean-only coordinate LayerNorm) or ``"covariant"`` (no frame role)."""
    from vfe3.geometry.norms import AffineLayerNorm, LayerNorm, get_norm_registration

    if norm is None:
        return "absent"
    if isinstance(norm, (LayerNorm, AffineLayerNorm)):
        return "conjugated"
    if getattr(model.cfg, "family", None) == "gaussian_full" and get_norm_registration(norm_type).gauge_equivariant:
        return "covariant"                  # U^{-1} N(U xi, U C U^T) = N(xi, C): frame-free in reference coordinates
    raise NotApplicable(
        f"normalization {type(norm).__name__} ({norm_type!r}) reads the covariance in local coordinates and is "
        "not GL(K)-covariant on this family; its reference conjugate is not the normal form's mean-only "
        "normalization")


def _mlp_plan(mlp: torch.nn.Module) -> str:
    r"""``"conjugated"``, ``"conjugated_template"`` or ``"covariant"`` (already :math:`U_i G(U_i^{-1}\cdot)`)."""
    from vfe3.model.block_mlp import BlockMLP, CanonicalFrameBlockMLP, GaussianTemplateBlockMLP

    if isinstance(mlp, CanonicalFrameBlockMLP):
        return "covariant"
    if type(mlp) is BlockMLP and mlp.covariance_contract == "passthrough":
        return "conjugated"
    if type(mlp) is GaussianTemplateBlockMLP:
        return "conjugated_template"
    raise NotApplicable(
        f"block MLP {type(mlp).__name__} (covariance_contract={getattr(mlp, 'covariance_contract', None)!r}) "
        "has no mean-only reference conjugate here")


def _role_trained_frame_free(model: torch.nn.Module, role: str) -> Optional[str]:
    r"""Why ``role`` already carries no frame on a model trained in reference coordinates, else ``None``.

    ``block_coordinates='reference'`` trains :math:`\mu \mapsto U_i\Phi(U_i^{-1}\mu)` (normalization, MLP
    and final norm in reference coordinates), so row 11 has nothing to remove and a second conjugation
    would be wrong; ``decode_coordinates='reference'`` trains logits :math:`W_{\mathrm{out}}U_i^{-1}\mu_i`
    (or a Gaussian decode of :math:`(U_i^{-1}\mu_i, C_i)`), so row 12 has nothing to remove.
    """
    cfg = model.cfg
    if role == ROLE_NORM_MLP and getattr(cfg, "block_coordinates", "local") == "reference":
        return ("block_coordinates='reference': the normalization and MLP already act in reference "
                "coordinates (mu -> U_i Phi(U_i^{-1} mu)), so the norm/MLP role carries no frame: "
                "nothing to remove")
    if role == ROLE_DECODER and getattr(cfg, "decode_coordinates", "local") == "reference":
        return ("decode_coordinates='reference': the decoder already reads the reference moments "
                "xi_i = U_i^{-1} mu_i, so the decoder role carries no frame: nothing to remove")
    return None


def _require_linear_decoder(model: torch.nn.Module) -> None:
    mode = getattr(model.cfg, "decode_mode", None)
    if mode != "linear":
        raise NotApplicable(
            f"decode_mode={mode!r}: the decoder role W_out U_i is defined for the linear decoder; a Gaussian "
            "decode reads (mu_i, Sigma_i) against the token bank in local coordinates and has no W_out, and "
            "removing U_i from it would need the full reference covariance C_i")


def _frame_free_covariance_score(
    mu:           torch.Tensor,
    sigma:        torch.Tensor,
    source_mu:    torch.Tensor,
    source_sigma: torch.Tensor,
    frame:        Any,
    operators:    Any,
    capture:      Optional[Dict[str, Any]] = None,
) -> torch.Tensor:                                   # (..., H, N, N)
    r"""``relational_score`` with the frame removed from the routing covariance only (row 10).

    The projected means keep the trained frames, :math:`a_i = S_Q U_i^{-1}\mu_i`,
    :math:`b_j = S_K U_j^{-1}\mu_j` (computed exactly as ``relational_score`` computes them); the projected
    covariances become :math:`V_{Q,i} = S_Q\Sigma_i S_Q^\top` and :math:`V_{K,j} = S_K\Sigma_j S_K^\top`,
    i.e. :math:`C_i \to \Sigma_i` in :math:`S C_i S^\top`. The source tuples handed to the KL are
    :math:`(\Sigma, I, S)`, so a factor or recovery route rebuilds exactly these covariances and never
    re-projects the means (a recovered pair reads the working-precision means, as in row 6).
    """
    from vfe3.relational_scoring import (
        _compute_dtype, projected_gaussian_kl, projected_moments, projected_noisy_query_cost)

    H, d = len(operators.irrep_dims), operators.irrep_dims[0]
    inverse = frame.inverse
    if not frame.block_diagonal:
        if H > 1:
            with torch.no_grad():
                mask = torch.block_diag(*(torch.ones(d, d, dtype=torch.bool, device=inverse.device)
                                          for _ in range(H)))
                if bool((inverse.detach()[..., ~mask] != 0).any()):
                    raise ValueError("relational_score requires a head-block-diagonal frame")
        inverse = torch.stack([inverse[..., h*d:(h+1)*d, h*d:(h+1)*d] for h in range(H)], dim=-3)  # (..., N, H, d, d)
    mu_h, source_mu_h = (x.unsqueeze(-2).expand(*x.shape[:-1], H, H * d) for x in (mu, source_mu))
    sigma_h, source_sigma_h = (
        x.unsqueeze(-3).expand(*x.shape[:-2], H, H * d, H * d)
        if x.ndim == mu.ndim + 1 else x.unsqueeze(-2).expand(*x.shape[:-1], H, H * d)
        for x in (sigma, source_sigma))
    eye = torch.eye(inverse.shape[-1], dtype=inverse.dtype, device=inverse.device).expand(inverse.shape)
    S_q, S_k = operators.comparison_pair()                                              # (H, r, w)
    if mu is source_mu and sigma is source_sigma:
        S = torch.stack((S_q, S_k))
        a, _ = projected_moments(mu_h, sigma_h, inverse, S)                             # (2, ..., N, H, r)
        _, G = projected_moments(mu_h, sigma_h, eye, S)                                 # (2, ..., N, H, r, r)
        a_q, a_k, G_q, G_k = a[0], a[1], G[0], G[1]
    else:
        a_q, _ = projected_moments(mu_h, sigma_h, inverse, S_q)
        _, G_q = projected_moments(mu_h, sigma_h, eye, S_q)
        a_k, _ = projected_moments(source_mu_h, source_sigma_h, inverse, S_k)
        _, G_k = projected_moments(source_mu_h, source_sigma_h, eye, S_k)
    public = _compute_dtype(mu, sigma, source_mu, source_sigma, S_q, S_k)
    if capture is not None:
        capture["projected_covariance_pre"] = G_q.detach().to(public)
        capture["projected_key_covariance_pre"] = G_k.detach().to(public)
    score = projected_noisy_query_cost if operators.routing == "noisy_query" else projected_gaussian_kl
    return score(a_q, G_q, a_k, G_k,
                 projection_q=(sigma_h, eye, S_q), projection_k=(source_sigma_h, eye, S_k)).to(public)


@contextmanager
def _instance_override(obj: Any, name: str, value: Any) -> Iterator[None]:
    r"""Shadow a method by an instance attribute and remove the shadow on exit (the class method returns)."""
    had, previous = name in vars(obj), vars(obj).get(name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if had:
            setattr(obj, name, previous)
        else:
            delattr(obj, name)


@contextmanager
def _swap_attribute(obj: Any, name: str, value: Any) -> Iterator[None]:
    original = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, original)


@contextmanager
def frames_removed(model: torch.nn.Module, roles: Sequence[str]) -> Iterator[Dict[str, Any]]:
    r"""Remove the frames from the given roles of the reference normal form, keeping them in every other role.

    **Route.** On the projected predictive route with covariance passthrough the model computes, in local
    coordinates and with the realized frames :math:`U_i` (the composed vertex factor
    :math:`\exp(X_{x_i})\exp(Y_i)` the relational operators read),

    .. math::
        \mu_i^{(0)} = \mu_{p,x_i},\quad \Sigma_i = \Sigma_{p,x_i},\quad
        a_i = S_Q U_i^{-1}\mu_i,\quad V_{Q,i} = S_Q U_i^{-1}\Sigma_i U_i^{-\top}S_Q^\top
        \ (\text{keys alike}),

    .. math::
        \mu_i^{\mathrm{att}} = \mu_i + U_i\sum_h O_h\sum_j\beta_{ij}^{(h)}T_h U_j^{-1}\mu_j,\qquad
        \mu_i^{(\ell+1)} = \Phi^{(\ell)}(\mu_i^{\mathrm{att}}),\qquad
        \ell_i = W_{\mathrm{out}}\mu_i^{(L)} + b,

    with :math:`\Phi^{(\ell)} = (\mathrm{id}+\mathrm{MLP}^{(\ell)})\circ\mathrm{LN}` (and a final norm, if
    any, before the decoder). Writing :math:`\mu_i = U_i\xi_i` gives the normal form of Eq.
    ``eq:reference_normal_form``: :math:`\xi_i^{(0)} = U_i^{-1}\mu_{p,x_i}`,
    :math:`C_i = U_i^{-1}\Sigma_{p,x_i}U_i^{-\top}`, the frame-free attention step
    :math:`\xi_i^{\mathrm{att}} = \xi_i + \sum_h O_h\sum_j\beta_{ij}^{(h)}T_h\xi_j` with :math:`\beta` a
    function of :math:`(S\xi, SCS^\top)` only (Theorem ``thm:reference_moments`` (1)),
    :math:`\xi_i^{(\ell+1)} = U_i^{-1}\Phi^{(\ell)}(U_i\xi_i^{\mathrm{att}})`, and logits
    :math:`W_{\mathrm{out}}U_i\xi_i^{(L)} + b`. Covariance passthrough makes :math:`C_i^{(\ell)} = C_i`
    at every layer, so the covariance's only frame channel is the routing covariance.

    **Local-coordinate equivalents.** Setting :math:`U_i = I` in one role while the others keep
    :math:`U_i`, and mapping back with the model's own state :math:`\mu_i = U_i\xi_i`, the model must compute

    * row 9 (``input_moments``): :math:`\xi_i^{(0)} = \mu_{p,x_i}`, i.e. the stack starts from
      :math:`\mu_i^{(0)} = U_i\mu_{p,x_i}` (the prior mean handed to the stack likewise);
    * row 10 (``routing_covariance``): :math:`C_i = \Sigma_{p,x_i}`, i.e.
      :math:`V_{Q,i} = S_Q\Sigma_i S_Q^\top`, :math:`V_{K,j} = S_K\Sigma_j S_K^\top` with the means
      :math:`S U^{-1}\mu` unchanged (:func:`_frame_free_covariance_score`);
    * row 11 (``norm_mlp``): :math:`\xi_i^{(\ell+1)} = \Phi^{(\ell)}(\xi_i^{\mathrm{att}})`, i.e.
      :math:`\mu_i^{(\ell+1)} = U_i\Phi^{(\ell)}(U_i^{-1}\mu_i^{\mathrm{att}})`, realized factor by factor as
      :math:`U_i\,\mathrm{LN}(U_i^{-1}\cdot)` followed by :math:`U_i(\mathrm{id}+\mathrm{MLP})(U_i^{-1}\cdot)`,
      whose composition is :math:`U_i\Phi U_i^{-1}`; a final norm :math:`\mathrm{FN}` becomes
      :math:`U_i\mathrm{FN}(U_i^{-1}\cdot)`; a block already of the form :math:`U_iG(U_i^{-1}\cdot)`
      (``canonical_frame`` MLP, or a GL(K)-covariant norm on the full family) has no frame role and is kept;
    * row 12 (``decoder``): logits :math:`W_{\mathrm{out}}\xi_i^{(L)} + b`, i.e.
      :math:`W_{\mathrm{out}}U_i^{-1}\mu_i^{(L)} + b` (unigram prior unchanged).

    Substituting :math:`\mu = U\xi` into each patched local computation reproduces the reference computation
    with :math:`U = I` in that role alone, because the attention step and the other roles' local formulas
    are untouched. Each patch is the identity map when :math:`U_i = I`. With all four roles removed the
    predictor is the :math:`U_i = I` model, i.e. rows 7 and 8 applied together; it equals row 7 alone only
    when the positional factor :math:`\exp(Y_i)` is already :math:`I`. In floating point the two differ by
    the round trips :math:`U_i(U_i^{-1}x)` (relative error :math:`O(\epsilon\,\kappa(U_i))` each).

    **Positional frames.** The model realizes one vertex factor per token, the composed
    :math:`U_i = \exp(X_{x_i})\exp(Y_i)`, and the normal form is written in it, so rows 9-12 remove the
    composed frame (token and positional factors together) from their role; rows 7 and 8 separate the two
    factors across all roles.

    **Scope.** Outside the normal form each row raises :class:`NotApplicable` with the reason: no relational
    predictive content on every layer, RoPE, frame-intrinsic beliefs, a non-``per_token`` encoder, a head
    mixer or CG coupling; row 10 also needs covariance passthrough in attention and MLP, row 11 a mean-only
    normalization or MLP acting in local coordinates, row 12 the linear decoder. A model trained with
    ``block_coordinates='reference'`` (row 11) or ``decode_coordinates='reference'`` (row 12) carries no
    frame in that role, so the row raises :class:`NotApplicable` rather than conjugating twice. The decoder patch acts on
    the logits readout (``forward_beliefs(..., return_logits=True)``, the post-hoc evaluation path), not on
    the fused training cross-entropy.
    """
    wanted = set(roles)
    unknown = sorted(wanted - set(FRAME_ROLES))
    if unknown or not wanted:
        raise ValueError(f"roles must be a nonempty subset of {FRAME_ROLES}, got {sorted(wanted)}")
    roles = tuple(role for role in FRAME_ROLES if role in wanted)
    _require_normal_form(model)
    cfg = model.cfg
    for role in roles:
        reason = _role_trained_frame_free(model, role)
        if reason is not None:
            raise NotApplicable(reason)
    record: Dict[str, Any] = {"roles": list(roles), "frame": FRAME_DESCRIPTION, "stack_calls": 0}
    norm_plan, final_plan, mlp_plans = "absent", "absent", []
    if ROLE_ROUTING_COVARIANCE in roles:
        _require_static_routing_covariance(model)
        record["intercepted_calls"] = 0
    if ROLE_NORM_MLP in roles:
        norm_plan = _norm_plan(model, getattr(model, "block_norm", None), cfg.norm_type_block)
        final_plan = _norm_plan(model, getattr(model, "final_norm", None), cfg.norm_type_final)
        mlp_plans = [_mlp_plan(m) for m in (getattr(model, "block_mlps", None) or ())]
        if "conjugated" not in (norm_plan, final_plan) and not any(p.startswith("conjugated") for p in mlp_plans):
            raise NotApplicable(
                "no normalization or MLP acts in local coordinates (block norm "
                f"{norm_plan}, final norm {final_plan}, MLP {mlp_plans or 'absent'}): nothing to remove")
        record.update(block_norm=norm_plan, final_norm=final_plan, block_mlp=mlp_plans)
    if ROLE_DECODER in roles:
        _require_linear_decoder(model)
        record["decode_calls"] = 0

    from vfe3.model import model as model_module

    state: Dict[str, Any] = {"frame": None}
    original_stack = model_module.vfe_stack

    def stack(belief: Any, mu_p: torch.Tensor, sigma_p: torch.Tensor, group: Any, stack_cfg: Any,
              **kwargs: Any) -> Any:
        frame = kwargs.get("relational_frame")
        if frame is None:
            raise NotApplicable("the belief stack received no realized relational frame U_i")
        if kwargs.get("rope") is not None:
            raise NotApplicable("a positional rotation is folded into the frames (RoPE)")
        value_frame = kwargs.get("relational_value_frame")
        if value_frame is not None and value_frame is not frame:
            raise NotApplicable("the content frame differs from the routing frame")
        state["frame"] = frame
        record["stack_calls"] += 1
        bd = frame.block_diagonal
        if ROLE_INPUT_MOMENTS in roles:
            mu_0 = _frame_matvec(frame.forward, belief.mu, bd)                              # U_i mu_p
            mu_p = mu_0 if mu_p is belief.mu else _frame_matvec(frame.forward, mu_p, bd)
            belief = belief._replace(mu=mu_0)
        if ROLE_NORM_MLP in roles:
            if norm_plan == "conjugated":
                kwargs["block_norm"] = _ConjugatedNorm(kwargs["block_norm"], frame)
            mlps = kwargs.get("block_mlps")
            if mlps is not None:
                kwargs["block_mlps"] = [
                    _ConjugatedBlockMLP(m, frame, template=(plan == "conjugated_template"))
                    if plan.startswith("conjugated") else m
                    for m, plan in zip(mlps, mlp_plans)]
        return original_stack(belief, mu_p, sigma_p, group, stack_cfg, **kwargs)

    def scored(*args: Any, **kwargs: Any) -> torch.Tensor:
        record["intercepted_calls"] += 1
        return _frame_free_covariance_score(*args, **kwargs)

    bound_decode = model._decode_belief_with_context

    def decode(mu_q: torch.Tensor, sigma_q: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        frame = _current_frame(state)
        record["decode_calls"] += 1
        return bound_decode(_frame_matvec(frame.inverse, mu_q, frame.block_diagonal), sigma_q, **kwargs)

    try:
        with ExitStack() as patches:
            patches.enter_context(patch_bindings([model_module], "vfe_stack", stack))
            if ROLE_ROUTING_COVARIANCE in roles:
                patches.enter_context(patch_bindings(_relational_score_modules(), "relational_score", scored))
            if ROLE_DECODER in roles:
                patches.enter_context(_instance_override(model, "_decode_belief_with_context", decode))
            if final_plan == "conjugated":
                patches.enter_context(_swap_attribute(
                    model, "final_norm", _final_norm_replacement(model.final_norm, state)))
            yield record
    finally:
        state["frame"] = None
    if record["stack_calls"] == 0:
        raise NotApplicable("no belief stack with a realized relational frame was intercepted")
    if ROLE_ROUTING_COVARIANCE in roles and record["intercepted_calls"] == 0:
        raise NotApplicable("no relational routing score was intercepted on this route")
    if ROLE_DECODER in roles and record["decode_calls"] == 0:
        raise NotApplicable("no logits readout was intercepted (only forward_beliefs(return_logits=True) is patched)")


def frames_removed_input_moments(model: torch.nn.Module) -> Any:
    r"""Row 9: :math:`\xi_i^{(0)} = U_i^{-1}\mu_{p,x_i} \to \mu_{p,x_i}`; locally the stack starts from :math:`U_i\mu_{p,x_i}`."""
    return frames_removed(model, (ROLE_INPUT_MOMENTS,))


def frames_removed_routing_covariance(model: torch.nn.Module) -> Any:
    r"""Row 10: :math:`C_i \to \Sigma_{p,x_i}`; locally :math:`V = S\Sigma S^\top`, means :math:`S U^{-1}\mu` kept."""
    return frames_removed(model, (ROLE_ROUTING_COVARIANCE,))


def frames_removed_norm_mlp(model: torch.nn.Module) -> Any:
    r"""Row 11: :math:`U_i^{-1}\Phi(U_i\,\cdot) \to \Phi`; locally :math:`\Phi \to U_i\Phi(U_i^{-1}\cdot)`."""
    return frames_removed(model, (ROLE_NORM_MLP,))


def frames_removed_decoder(model: torch.nn.Module) -> Any:
    r"""Row 12: :math:`W_{\mathrm{out}}U_i\xi_i \to W_{\mathrm{out}}\xi_i`; locally logits from :math:`U_i^{-1}\mu_i^{(L)}`."""
    return frames_removed(model, (ROLE_DECODER,))


def frame_role_composite_check(
    model:      torch.nn.Module,
    dataset:    Any,
    indices:    Sequence[int],

    *,
    batch_size: int,
    device:     torch.device,
) -> Dict[str, Any]:
    r"""Rows 9-12 applied together against rows 7 and 8 applied together, on the same windows.

    Both are the predictor with :math:`U_i = I` in every role (:func:`frames_removed`), so the per-target
    NLLs agree up to the floating-point round trips :math:`U_i(U_i^{-1}x)`. Without positional frames
    the reference is row 7 alone (``positional_frames`` False). A role the model was trained without a
    frame in (``block_coordinates`` or ``decode_coordinates`` set to ``'reference'``) is already
    frame-free, so it is left out of the patch and listed under ``frame_free_by_construction``.
    """
    from vfe3.posthoc import evaluate as ev

    frame_free = [role for role in FRAME_ROLES if _role_trained_frame_free(model, role) is not None]
    patched = tuple(role for role in FRAME_ROLES if role not in frame_free)
    if not patched:
        raise NotApplicable("every frame role is frame-free by construction on this model")
    with frames_removed(model, patched) as record:
        composite = ev.nll_pass(model, dataset, indices, batch_size=batch_size, device=device)
    with ExitStack() as patches:
        patches.enter_context(token_frames_identity(model))
        try:
            patches.enter_context(positional_frames_identity(model))
            positional = True
        except NotApplicable:
            positional = False
        reference = ev.nll_pass(model, dataset, indices, batch_size=batch_size, device=device)
    difference = np.abs(composite.nll - reference.nll)
    return {"ce_all_roles_removed": composite.ce, "ce_frames_identity": reference.ce,
            "positional_frames": positional, "n_targets": composite.n,
            "max_abs_nll_difference": float(difference.max()) if difference.size else float("nan"),
            "mean_abs_nll_difference": float(difference.mean()) if difference.size else float("nan"),
            "frame_free_by_construction": frame_free,
            "record": dict(record)}


# ----------------------------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------------------------

INTERVENTIONS: Tuple[Intervention, ...] = (
    Intervention(1, "baseline", "None (final weights)", baseline),
    Intervention(2, "routing_costs_removed", "Routing costs removed; positional prior retained",
                 routing_costs_removed),
    Intervention(3, "prior_flattened", "Positional prior flattened on the causal support", prior_flattened),
    Intervention(4, "residual_removed", r"Residual term removed at every layer, $\mu_i+m_i\mapsto m_i$",
                 residual_removed),
    Intervention(5, "stored_covariances_identity",
                 r"Stored covariances $\Sigma_{p,v}$ replaced by the identity", stored_covariances_identity),
    Intervention(6, "routing_covariances_identity",
                 r"All routing covariances $C_i$ replaced by the identity", routing_covariances_identity),
    Intervention(7, "token_frames_identity", "Token frames replaced by the identity in every role",
                 token_frames_identity),
    Intervention(8, "positional_frames_identity", "Positional frames replaced by the identity in every role",
                 positional_frames_identity),
    Intervention(9, "frames_removed_input_moments", "Frames removed from the input moments only",
                 frames_removed_input_moments),
    Intervention(10, "frames_removed_routing_covariance", "Frames removed from the routing covariance only",
                 frames_removed_routing_covariance),
    Intervention(11, "frames_removed_norm_mlp", "Frames removed from the normalization and MLP conjugation only",
                 frames_removed_norm_mlp),
    Intervention(12, "frames_removed_decoder", "Frames removed from the decoder only", frames_removed_decoder),
)

# Every row of Table M5 is implemented; kept (empty) for the report's not-implemented bookkeeping.
NOT_IMPLEMENTED: Tuple[Tuple[int, str, str], ...] = ()
NOT_IMPLEMENTED_REASON = "not implemented in vfe3.posthoc.interventions"


def by_key(keys: Optional[Sequence[str]] = None) -> List[Intervention]:
    r"""The registered interventions in row order, restricted to ``keys`` (always including the baseline)."""
    if keys is None:
        return list(INTERVENTIONS)
    known = {i.key for i in INTERVENTIONS}
    unknown = sorted(set(keys) - known)
    if unknown:
        raise ValueError(f"unknown interventions {unknown}; known: {sorted(known)}")
    wanted = set(keys) | {"baseline"}
    return [i for i in INTERVENTIONS if i.key in wanted]
