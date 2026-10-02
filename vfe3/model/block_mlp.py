from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping, NamedTuple, Optional, Sequence

import torch
from torch import nn

from vfe3.contracts import BlockMLPBuildMetadata, CanonicalFrameContext
from vfe3.numerics import validated_cholesky_solve


_ACTIVATIONS = {
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "relu": nn.ReLU,
}


_BLOCK_MLP_COVARIANCE_CONTRACTS = frozenset({"passthrough", "delta_full"})
# Tensor layout of the belief covariance the block reads: (..., K, K) or its diagonal (..., K).
_COVARIANCE_LAYOUTS = frozenset({"full", "diagonal"})


def _validate_block_mlp_covariance_contract(covariance_contract: str) -> str:
    if covariance_contract not in _BLOCK_MLP_COVARIANCE_CONTRACTS:
        raise ValueError(f"unknown BlockMLP covariance contract {covariance_contract!r}")
    return covariance_contract


def _validate_covariance_layout(covariance_layout: str) -> str:
    if covariance_layout not in _COVARIANCE_LAYOUTS:
        raise ValueError(
            f"covariance_layout must be one of {sorted(_COVARIANCE_LAYOUTS)}, "
            f"got {covariance_layout!r}")
    return covariance_layout


def _require_canonical_frame_context(
    frame_context: Optional[CanonicalFrameContext],
) -> CanonicalFrameContext:
    if frame_context is None:
        raise ValueError("canonical_frame block MLP requires a realized canonical frame")
    if not isinstance(frame_context, CanonicalFrameContext):
        raise TypeError(
            "canonical_frame block MLP frame_context must be a CanonicalFrameContext")
    return frame_context


def _escalated_frame(frame: CanonicalFrameContext, mu: torch.Tensor) -> bool:
    r"""Whether a float64 frame (escalated pair, audit 2026-09-23 #12) acts on lower-precision beliefs."""
    return frame.forward.dtype == torch.float64 and mu.dtype != torch.float64


def _frame_action(
    frame: CanonicalFrameContext,
    mu:    torch.Tensor,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    r"""The map ``(M, x) -> M x`` used for this frame's vertex factors.

    An escalated frame applies each factor at float64 and returns ``x.dtype``, so the MLP and every
    downstream tensor keep their dtypes. Any other frame contracts with autocast disabled at
    ``promote_types(x.dtype, float32)``: an ambient bf16 autocast would otherwise store the belief
    mean in bf16 (the MLP's own residual then adds its bf16 increment to a float32 mean).
    """
    if _escalated_frame(frame, mu):
        return lambda M, x: torch.einsum("...ij,...j->...i", M, x.double()).to(x.dtype)

    def apply(M: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        dtype = torch.promote_types(torch.promote_types(x.dtype, M.dtype), torch.float32)
        with torch.autocast(device_type=x.device.type, enabled=False):
            return torch.einsum("...ij,...j->...i", M.to(dtype), x.to(dtype))
    return apply



def block_mlp_activation(name: str) -> nn.Module:
    try:
        return _ACTIVATIONS[name]()
    except KeyError as exc:
        raise ValueError(f"unknown block MLP activation {name!r}") from exc


def _activation_derivative(name: str, value: torch.Tensor) -> torch.Tensor:
    if name == "relu":
        return (value > 0).to(dtype=value.dtype)
    if name == "silu":
        sigmoid = torch.sigmoid(value)
        return sigmoid * (1.0 + value * (1.0 - sigmoid))
    if name == "gelu":
        inv_sqrt_two = value.new_tensor(2.0).rsqrt()
        inv_sqrt_two_pi = value.new_tensor(2.0 * torch.pi).rsqrt()
        return 0.5 * (1.0 + torch.erf(value * inv_sqrt_two)) \
            + value * torch.exp(-0.5 * value.square()) * inv_sqrt_two_pi
    raise RuntimeError(f"unsupported block MLP activation {name!r}")


def _dropout_scale(dropout: nn.Dropout, reference: torch.Tensor) -> torch.Tensor:
    if not dropout.training or dropout.p == 0.0:
        return torch.ones_like(reference)
    return torch.nn.functional.dropout(
        torch.ones_like(reference), p=dropout.p, training=True)


def _add_hidden_bias(hidden_pre: torch.Tensor, hidden_bias: torch.Tensor) -> torch.Tensor:
    r"""Add a per-token bias to the first ``w`` units of the fc1 pre-activation.

    .. math::
        h_{ij} = (W_1 u_i + b_1)_j + E_{x_i j}\,[j < w],

    with ``hidden_pre`` (..., H) and ``hidden_bias`` (..., w), identical leading axes and w <= H. The
    bias is cast to the pre-activation dtype, as fc1's own bias is under autocast, so the hidden path
    keeps the dtype of the unbiased path; its gradient still reaches the stored table at full precision.
    """
    hidden_width = hidden_pre.shape[-1]
    if hidden_bias.dim() != hidden_pre.dim() or hidden_bias.shape[:-1] != hidden_pre.shape[:-1]:
        raise ValueError(
            "hidden_bias must be (..., w) with the leading axes of the hidden pre-activation "
            f"{tuple(hidden_pre.shape[:-1])}, got {tuple(hidden_bias.shape)}")
    width = hidden_bias.shape[-1]
    if width > hidden_width:
        raise ValueError(
            f"hidden_bias width w={width} exceeds the fc1 hidden width {hidden_width}")
    bias = hidden_bias.to(hidden_pre.dtype)                               # (..., w)
    if width == hidden_width:
        return hidden_pre + bias
    # Units w..H-1 are passed through untouched (not added to zero), so they stay bit-identical.
    return torch.cat([hidden_pre[..., :width] + bias, hidden_pre[..., width:]], dim=-1)


def _reject_hidden_bias(hidden_bias: Optional[torch.Tensor], reason: str) -> None:
    if hidden_bias is not None:
        raise ValueError(f"hidden_bias does not apply: {reason}")


def _is_diagonal_layout(mu: torch.Tensor, sigma: torch.Tensor) -> bool:
    r"""Whether ``sigma`` is the diagonal layout (..., K) of Sigma rather than the full (..., K, K)."""
    return sigma.shape == mu.shape


class BlockMLPMomentResult(NamedTuple):
    """Post-MLP belief moments and, when requested, the mean-map Jacobian."""

    mu: torch.Tensor
    sigma: torch.Tensor
    jacobian: Optional[torch.Tensor]


def _delta_covariance(
    jacobian: torch.Tensor,
    sigma: torch.Tensor,
    covariance_floor: float,
) -> torch.Tensor:
    js = torch.einsum("...ij,...jk->...ik", jacobian, sigma)
    jsj = torch.einsum("...ik,...lk->...il", js, jacobian)
    pushed = jsj + covariance_floor * sigma
    return 0.5 * (pushed + pushed.transpose(-1, -2))


class BlockMLP(nn.Module):
    """Ordinary coordinate residual MLP, retained as the backward-compatible default."""

    def __init__(
        self,
        embed_dim: int,
        expansion: int,
        activation: str,
        dropout: float,
        *,
        covariance_contract: str = "passthrough",
        covariance_floor: float = 1e-4,
    ) -> None:
        super().__init__()
        hidden_dim = embed_dim * expansion
        self.activation_name = activation
        self.covariance_contract = _validate_block_mlp_covariance_contract(covariance_contract)
        self.covariance_floor = float(covariance_floor)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.activation = block_mlp_activation(activation)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def _hidden_pre(
        self,
        mu:          torch.Tensor,                    # (..., K)
        hidden_bias: Optional[torch.Tensor],          # (..., w) or None
    ) -> torch.Tensor:                                # (..., H)
        hidden_pre = self.fc1(mu)
        if hidden_bias is None:
            return hidden_pre
        return _add_hidden_bias(hidden_pre, hidden_bias)

    def forward(
        self,
        mu: torch.Tensor,
        *,
        hidden_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Residual update mu + W_2 act(W_1 mu + b_1 + E) + b_2, E = [hidden_bias, 0] (absent when None)."""
        return mu + self.dropout(self.fc2(self.activation(self._hidden_pre(mu, hidden_bias))))

    def _coordinate_update_with_jacobian(
        self,
        mu: torch.Tensor,
        hidden_bias: Optional[torch.Tensor] = None,
    ) -> 'tuple[torch.Tensor, torch.Tensor]':
        hidden_pre = self._hidden_pre(mu, hidden_bias)
        residual = self.fc2(self.activation(hidden_pre))
        dropout_scale = _dropout_scale(self.dropout, residual)
        mu_out = mu + dropout_scale * residual
        residual_jacobian = torch.einsum(
            "oh,...h,hi->...oi",
            self.fc2.weight,
            _activation_derivative(self.activation_name, hidden_pre),
            self.fc1.weight,
        )
        residual_jacobian = dropout_scale.unsqueeze(-1) * residual_jacobian
        identity = torch.eye(mu.shape[-1], dtype=mu.dtype, device=mu.device)
        return mu_out, identity + residual_jacobian

    def forward_moments(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        *,
        frame_context: Optional[CanonicalFrameContext] = None,
        hidden_bias: Optional[torch.Tensor] = None,    # (..., w), w <= H: per-token fc1 bias
    ) -> BlockMLPMomentResult:
        del frame_context
        if self.covariance_contract == "passthrough":
            return BlockMLPMomentResult(self.forward(mu, hidden_bias=hidden_bias), sigma, None)
        mu_out, jacobian = self._coordinate_update_with_jacobian(mu, hidden_bias)
        return BlockMLPMomentResult(
            mu_out, _delta_covariance(jacobian, sigma, self.covariance_floor), jacobian)


class GaugeGateBlockMLP(nn.Module):
    """Gauge-equivariant residual gate built from Mahalanobis invariants.

    The deterministic/evaluation path and every fixed realization of the scalar per-block
    dropout mask are exactly equivariant. Independent dropout draws are equivariant in
    distribution, not identical sample by sample.

    For every declared irrep block h, s_h = mu_h^T Sigma_hh^-1 mu_h is invariant under the
    corresponding invertible frame action. A scalar MLP maps the invariant vector s to one
    gate per block, and that scalar multiplies every coordinate in the block.

    Sigma may be given in the full layout (..., K, K) or the diagonal layout (..., K). On the
    diagonal layout the invariant is s_h = sum_{k in h} mu_k^2 / sigma_k, and the gate equals the
    full-layout gate on diag_embed(sigma). The operator is covariant under the block frame group
    prod_h GL(d_h) (not under a dense GL(K)): the diagonal family is not closed under that action,
    but the image g diag(sigma) g^T is block-diagonal and the full-layout gate evaluates it.
    delta_full (J Sigma J^T is dense) requires the full layout.
    """

    def __init__(
        self,
        irrep_dims: Sequence[int],
        expansion: int,
        activation: str,
        dropout: float,
        *,
        covariance_contract: str = "passthrough",
        covariance_floor: float = 1e-4,
    ) -> None:
        super().__init__()
        self.irrep_dims = tuple(int(dim) for dim in irrep_dims)
        if not self.irrep_dims or any(dim < 1 for dim in self.irrep_dims):
            raise ValueError(f"irrep_dims must contain positive dimensions, got {irrep_dims!r}")
        n_blocks = len(self.irrep_dims)
        hidden_dim = n_blocks * expansion
        self.activation_name = activation
        self.covariance_contract = _validate_block_mlp_covariance_contract(covariance_contract)
        self.fc1 = nn.Linear(n_blocks, hidden_dim)
        self.covariance_floor = float(covariance_floor)
        self.activation = block_mlp_activation(activation)
        self.fc2 = nn.Linear(hidden_dim, n_blocks)
        self.dropout = nn.Dropout(dropout)
        # Identity at step zero keeps this opt-in transform from perturbing a baseline before its
        # invariant gate has learned evidence for a nonzero update.
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def _invariants(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
    ) -> 'tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]':
        mu_blocks = []
        expected_dim = sum(self.irrep_dims)
        if mu.shape[-1] != expected_dim:
            raise ValueError(
                "irrep_dims must sum to the mean dimension, got "
                f"sum(irrep_dims)={expected_dim} and mean dimension={mu.shape[-1]}"
            )
        if _is_diagonal_layout(mu, sigma):
            return self._diagonal_invariants(mu, sigma)
        if sigma.shape[-2:] != (expected_dim, expected_dim):
            raise ValueError(
                "gauge gate covariance axes must match sum(irrep_dims), got "
                f"{tuple(sigma.shape[-2:])} and expected {(expected_dim, expected_dim)}"
            )
        sigma_solutions = []
        invariants = []
        start = 0
        for dim in self.irrep_dims:
            stop = start + dim
            mu_h = mu[..., start:stop]
            sigma_hh = sigma[..., start:stop, start:stop]
            rhs = mu_h.unsqueeze(-1)
            checked = validated_cholesky_solve(sigma_hh, rhs)
            if bool(checked.certified.all()):
                assert checked.solution is not None
                solved = checked.solution.squeeze(-1)
            elif sigma_hh.dtype is torch.float64:
                raise FloatingPointError(
                    "GaugeGate Mahalanobis system could not be certified in float64")
            else:
                high = validated_cholesky_solve(sigma_hh.double(), rhs.double())
                if not bool(high.certified.all()):
                    raise FloatingPointError(
                        "GaugeGate Mahalanobis system could not be certified after float64 "
                        "escalation")
                assert checked.solution is not None and high.solution is not None
                solved = torch.where(
                    checked.certified.unsqueeze(-1).unsqueeze(-1),
                    checked.solution.double(), high.solution,
                ).squeeze(-1)
            mu_blocks.append(mu_h)
            sigma_solutions.append(solved)
            invariants.append((mu_h.to(solved.dtype) * solved).sum(dim=-1))
            start = stop
        return torch.stack(invariants, dim=-1), mu_blocks, sigma_solutions

    def _diagonal_invariants(
        self,
        mu:    torch.Tensor,                          # (..., K)
        sigma: torch.Tensor,                          # (..., K) = diag Sigma, sigma_k > 0
    ) -> 'tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]':
        r"""Block invariants of a diagonal covariance layout.

        .. math::
            s_b = \mu_b^\top \Sigma_{bb}^{-1} \mu_b = \sum_{k \in b} \frac{\mu_k^2}{\sigma_k},

        evaluated at ``promote_types(mu, sigma, float32)`` (at least fp32; float64 stays float64). No
        clamp: sigma is positive through its log parameterization, and a clamp would break the exact
        agreement with the full-layout path on ``diag_embed(sigma)``.
        """
        compute_dtype = torch.promote_types(
            torch.promote_types(mu.dtype, sigma.dtype), torch.float32)
        mu_work = mu.to(compute_dtype)                # (..., K)
        sigma_work = sigma.to(compute_dtype)          # (..., K)
        mu_blocks, solutions, invariants = [], [], []
        start = 0
        for dim in self.irrep_dims:
            stop = start + dim
            mu_h = mu_work[..., start:stop]                               # (..., d_b)
            solved = mu_h / sigma_work[..., start:stop]                   # Sigma_bb^{-1} mu_b
            mu_blocks.append(mu[..., start:stop])
            solutions.append(solved)
            invariants.append((mu_h * solved).sum(dim=-1))                # (...,)
            start = stop
        return torch.stack(invariants, dim=-1), mu_blocks, solutions      # (..., B)

    def _gate_values(
        self,
        invariants: torch.Tensor,
        *,
        need_jacobian: bool,
    ) -> 'tuple[torch.Tensor, Optional[torch.Tensor]]':
        hidden_pre = torch.nn.functional.linear(
            invariants,
            self.fc1.weight.to(invariants.dtype),
            self.fc1.bias.to(invariants.dtype),
        )
        raw_gates = torch.nn.functional.linear(
            self.activation(hidden_pre),
            self.fc2.weight.to(hidden_pre.dtype),
            self.fc2.bias.to(hidden_pre.dtype),
        )
        if not need_jacobian:
            return self.dropout(raw_gates), None
        dropout_scale = _dropout_scale(self.dropout, raw_gates)
        dgate_ds = torch.einsum(
            "oh,...h,hi->...oi",
            self.fc2.weight.to(hidden_pre.dtype),
            _activation_derivative(self.activation_name, hidden_pre),
            self.fc1.weight.to(hidden_pre.dtype),
        )
        return dropout_scale * raw_gates, dropout_scale.unsqueeze(-1) * dgate_ds

    def mean_update(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        invariants, mu_blocks, _ = self._invariants(mu, sigma)
        gates, _ = self._gate_values(invariants, need_jacobian=False)
        return torch.cat([
            (1.0 + gates[..., index:index + 1]) * mu_h
            for index, mu_h in enumerate(mu_blocks)
        ], dim=-1)

    def forward_moments(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        *,
        frame_context: Optional[CanonicalFrameContext] = None,
        hidden_bias: Optional[torch.Tensor] = None,
    ) -> BlockMLPMomentResult:
        del frame_context
        _reject_hidden_bias(
            hidden_bias,
            "the gauge_gate fc1 acts on the per-block Mahalanobis invariants, not on a coordinate "
            "hidden layer")
        need_jacobian = self.covariance_contract == "delta_full"
        if need_jacobian and _is_diagonal_layout(mu, sigma):
            raise ValueError(
                "gauge_gate covariance_contract='delta_full' needs the full (..., K, K) covariance "
                "layout: the pushed covariance J Sigma J^T is dense, so a diagonal layout cannot "
                "carry it")
        invariants, mu_blocks, sigma_solutions = self._invariants(mu, sigma)
        gates, dgate_ds = self._gate_values(invariants, need_jacobian=need_jacobian)
        mu_out = torch.cat([
            (1.0 + gates[..., index:index + 1]) * mu_h
            for index, mu_h in enumerate(mu_blocks)
        ], dim=-1)
        if not need_jacobian:
            return BlockMLPMomentResult(mu_out, sigma, None)

        K = mu.shape[-1]
        jacobian = torch.zeros(
            *mu.shape[:-1], K, K, dtype=gates.dtype, device=mu.device)
        row_start = 0
        for out_index, (out_dim, mu_h) in enumerate(zip(self.irrep_dims, mu_blocks)):
            row_stop = row_start + out_dim
            jacobian[..., row_start:row_stop, row_start:row_stop] = \
                (1.0 + gates[..., out_index]).unsqueeze(-1).unsqueeze(-1) \
                * torch.eye(out_dim, dtype=gates.dtype, device=mu.device)
            col_start = 0
            for in_index, (in_dim, solved) in enumerate(
                    zip(self.irrep_dims, sigma_solutions)):
                col_stop = col_start + in_dim
                cross = torch.einsum(
                    "...i,...,...j->...ij",
                    mu_h.to(jacobian.dtype),
                    dgate_ds[..., out_index, in_index],
                    2.0 * solved,
                )
                jacobian[..., row_start:row_stop, col_start:col_stop] = \
                    jacobian[..., row_start:row_stop, col_start:col_stop] + cross
                col_start = col_stop
            row_start = row_stop
        return BlockMLPMomentResult(
            mu_out,
            _delta_covariance(jacobian, sigma.to(jacobian.dtype), self.covariance_floor),
            jacobian,
        )


class CanonicalFrameBlockMLP(BlockMLP):
    """Ordinary residual MLP in a realized, frame-fixed vertex coordinate system."""

    def forward(
        self,
        mu: torch.Tensor,
        *,
        frame_context: Optional[CanonicalFrameContext] = None,
        hidden_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.mean_update(mu, frame_context, hidden_bias=hidden_bias)

    def mean_update(
        self,
        mu: torch.Tensor,
        frame: Optional[CanonicalFrameContext],
        *,
        hidden_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""mu -> U F(U^{-1} mu); a hidden_bias enters F's fc1 pre-activation in canonical coordinates."""
        frame = _require_canonical_frame_context(frame)
        apply = _frame_action(frame, mu)
        canonical_mu = apply(frame.inverse, mu)
        canonical_out = super().forward(canonical_mu, hidden_bias=hidden_bias)
        return apply(frame.forward, canonical_out)

    def forward_moments(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        *,
        frame_context: Optional[CanonicalFrameContext] = None,
        hidden_bias: Optional[torch.Tensor] = None,
    ) -> BlockMLPMomentResult:
        frame_context = _require_canonical_frame_context(frame_context)
        apply = _frame_action(frame_context, mu)
        canonical_mu = apply(frame_context.inverse, mu)
        if self.covariance_contract == "passthrough":
            canonical_out = super().forward(canonical_mu, hidden_bias=hidden_bias)
            mu_out = apply(frame_context.forward, canonical_out)
            return BlockMLPMomentResult(mu_out, sigma, None)
        canonical_out, canonical_jacobian = self._coordinate_update_with_jacobian(
            canonical_mu, hidden_bias)
        mu_out = apply(frame_context.forward, canonical_out)
        if not _escalated_frame(frame_context, mu):
            realized_jacobian = (
                frame_context.forward @ canonical_jacobian @ frame_context.inverse
            )
        else:
            realized_jacobian = (
                frame_context.forward @ canonical_jacobian.double() @ frame_context.inverse
            ).to(canonical_jacobian.dtype)
        return BlockMLPMomentResult(
            mu_out,
            _delta_covariance(realized_jacobian, sigma, self.covariance_floor),
            realized_jacobian,
        )

class GaussianTemplateBlockMLP(nn.Module):
    r"""Residual block computed as Gibbs routing to M = expansion * K learned Gaussian templates.

    Each template a carries a diagonal Gaussian p_a = N(nu_a, diag(lambda_a)), a log prior weight
    log pi_a, and a value w_a. Under a mixture prior p(k) = sum_a pi_a p_a, the optimal variational
    responsibilities of the belief q_i = N(mu_i, Sigma_i) are

    .. math::
        r_{ia} \propto \pi_a \exp\bigl(-D_{\mathrm{KL}}(q_i \,\|\, p_a)\bigr),

    because the entropy H(q_i) is template-independent. Dropping every a-independent term,

    .. math::
        \log r_{ia} = \log\pi_a - \tfrac12 \sum_k \Bigl[
            \frac{(\Sigma_i)_{kk} + (\mu_{ik} - \nu_{ak})^2}{\lambda_{ak}} + \log\lambda_{ak}
        \Bigr] + \mathrm{const}_i,

    and the mean update is mu_i + sum_a r_ia w_a (Gaussian mixture regression with zero regression
    blocks). Only diag(Sigma_i) enters, so full and diagonal covariance layouts give the same
    logits. With shared isotropic covariances the logits are affine in mu_i and the block is a
    width-M MLP whose GELU gate is replaced by a softmax over templates. The covariance passes
    through unchanged; the block acts in the same local coordinates as the coordinate MLP.
    """

    def __init__(
        self,
        embed_dim: int,
        expansion: int,
        dropout: float,
        *,
        covariance_contract: str = "passthrough",
    ) -> None:
        super().__init__()
        if covariance_contract != "passthrough":
            raise ValueError(
                "gaussian_template block MLP supports only covariance_contract='passthrough', "
                f"got {covariance_contract!r}")
        n_templates = embed_dim * expansion
        self.covariance_contract = covariance_contract
        # Initialization mirrors BlockMLP: template means match fc1 rows, values match fc2 columns.
        bound_in = embed_dim ** -0.5
        bound_out = n_templates ** -0.5
        self.nu = nn.Parameter(torch.empty(n_templates, embed_dim).uniform_(-bound_in, bound_in))
        self.log_lambda = nn.Parameter(torch.zeros(n_templates, embed_dim))    # lambda_a = 1 at init
        self.log_pi = nn.Parameter(torch.zeros(n_templates))                   # uniform prior at init
        self.values = nn.Parameter(
            torch.empty(n_templates, embed_dim).uniform_(-bound_out, bound_out))
        self.dropout = nn.Dropout(dropout)

    def responsibilities(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        r"""Return r_ia, shape (..., M), from mu (..., K) and sigma (..., K) or (..., K, K).

        The expanded square below reaches magnitudes of order K (Sigma_kk + mu_k^2) / lambda_ak and
        cancels terms of that size, so it is evaluated with autocast disabled at
        ``promote_types(mu, float32)`` (float64 stays float64); r is returned in ``mu.dtype``. Under
        bf16 autocast the rounded logits otherwise moved r by a mean total variation of 0.05 at
        variance ~1 and 0.26 at variance ~15 (CPU autocast, K = 256, M = 2048, 2026-09-28).
        """
        sigma_diag = sigma if sigma.dim() == mu.dim() else torch.diagonal(sigma, dim1=-2, dim2=-1)
        dtype = torch.promote_types(mu.dtype, torch.float32)
        with torch.autocast(device_type=mu.device.type, enabled=False):
            mu_work = mu.to(dtype)                                             # (..., K)
            nu = self.nu.to(dtype)                                             # (M, K)
            precision = torch.exp(-self.log_lambda.to(dtype))                  # (M, K)
            # sum_k (Sigma_kk + mu_k^2 - 2 mu_k nu_ak + nu_ak^2) / lambda_ak, contracted as matmuls
            quadratic = (
                (sigma_diag.to(dtype) + mu_work.square()) @ precision.T        # (..., M)
                - 2.0 * mu_work @ (nu * precision).T
                + (nu.square() * precision).sum(dim=-1)
            )
            logits = self.log_pi.to(dtype) - 0.5 * (
                quadratic + self.log_lambda.to(dtype).sum(dim=-1))
            responsibilities = torch.softmax(logits, dim=-1)                   # (..., M)
        return responsibilities.to(mu.dtype)

    def mean_update(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        message = self.responsibilities(mu, sigma) @ self.values.to(mu.dtype)  # (..., K)
        return mu + self.dropout(message)

    def forward(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return self.mean_update(mu, sigma)

    def forward_moments(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        *,
        frame_context: Optional[CanonicalFrameContext] = None,
        hidden_bias: Optional[torch.Tensor] = None,
    ) -> BlockMLPMomentResult:
        del frame_context
        _reject_hidden_bias(
            hidden_bias,
            "the gaussian_template block routes to templates and has no fc1 hidden layer")
        return BlockMLPMomentResult(self.mean_update(mu, sigma), sigma, None)


BlockMLPFactory = Callable[..., nn.Module]
BlockMLPReportLabel = Callable[[str], str]
BlockMLPWidth = Callable[[int, Sequence[int]], int]
BlockMLPExtraFlops = Callable[[int, Sequence[int]], float]


@dataclass(frozen=True)
class BlockMLPRegistration:
    """Factory and every structural capability owned by one BlockMLP mode."""

    factory: BlockMLPFactory
    requires_frame_context: bool
    covariance_kinds: frozenset[str]
    gauge_class: str
    intertwiner_compatible: bool
    width: BlockMLPWidth
    extra_flops_per_layer: BlockMLPExtraFlops
    report_label: BlockMLPReportLabel
    # (width, expansion) -> per-layer count, for modes whose parameters are not a two-layer MLP.
    parameter_count_per_layer: Optional[Callable[[int, int], int]] = None
    # (width, expansion) -> per-layer matmul FLOPs replacing the two-layer 4 * expansion * width^2.
    matmul_flops_per_layer: Optional[Callable[[int, int], float]] = None
    # extra_flops_per_layer for the diagonal covariance layout (..., K), for modes whose extra cost
    # depends on the layout; None means the layout does not change the cost.
    diagonal_extra_flops_per_layer: Optional[BlockMLPExtraFlops] = None

    def parameter_count(
        self,
        *,
        embed_dim: int,
        irrep_dims: Sequence[int],
        expansion: int,
        n_layers: int = 1,
    ) -> int:
        width = self.width(embed_dim, irrep_dims)
        if self.parameter_count_per_layer is not None:
            return int(n_layers) * self.parameter_count_per_layer(width, expansion)
        per_layer = 2 * expansion * width * width + (expansion + 1) * width
        return int(n_layers) * per_layer

    def flops_per_token(
        self,
        *,
        embed_dim: int,
        irrep_dims: Sequence[int],
        expansion: int,
        covariance_contract: str,
        n_layers: int,
        covariance_layout: str = "full",
    ) -> float:
        width = self.width(embed_dim, irrep_dims)
        per_layer = (
            self.matmul_flops_per_layer(width, expansion)
            if self.matmul_flops_per_layer is not None
            else 4.0 * expansion * width * width
        )
        extra_flops = self.extra_flops_per_layer
        if _validate_covariance_layout(covariance_layout) == "diagonal" \
                and self.diagonal_extra_flops_per_layer is not None:
            extra_flops = self.diagonal_extra_flops_per_layer
        per_layer +=extra_flops(embed_dim, irrep_dims)
        if covariance_contract == "delta_full":
            per_layer += 4.0 * embed_dim ** 3
        return float(n_layers) * per_layer


def _coordinate_factory(**kwargs) -> nn.Module:
    kwargs.pop("irrep_dims")
    return BlockMLP(**kwargs)


def _gauge_gate_factory(**kwargs) -> nn.Module:
    kwargs.pop("embed_dim")
    return GaugeGateBlockMLP(**kwargs)


def _canonical_frame_factory(**kwargs) -> nn.Module:
    kwargs.pop("irrep_dims")
    return CanonicalFrameBlockMLP(**kwargs)


def _gaussian_template_factory(**kwargs) -> nn.Module:
    for unused in ("irrep_dims", "activation", "covariance_floor"):
        kwargs.pop(unused)
    return GaussianTemplateBlockMLP(**kwargs)


def _coordinate_report_label(covariance: str) -> str:
    return (
        "coordinate_mean_only_nonintertwiner"
        if covariance == "passthrough"
        else "coordinate_delta_full_covariant_floor_nonintertwiner"
    )


BLOCK_MLP_REGISTRATIONS: Mapping[str, BlockMLPRegistration] = MappingProxyType({
    "coordinate": BlockMLPRegistration(
        factory=_coordinate_factory,
        requires_frame_context=False,
        covariance_kinds=_BLOCK_MLP_COVARIANCE_CONTRACTS,
        gauge_class="coordinate_nonintertwiner",
        intertwiner_compatible=False,
        width=lambda embed_dim, _irrep_dims: embed_dim,
        extra_flops_per_layer=lambda _embed_dim, _irrep_dims: 0.0,
        report_label=_coordinate_report_label,
    ),
    "gauge_gate": BlockMLPRegistration(
        factory=_gauge_gate_factory,
        requires_frame_context=False,
        covariance_kinds=_BLOCK_MLP_COVARIANCE_CONTRACTS,
        gauge_class="invariant_scalar_intertwiner",
        intertwiner_compatible=True,
        width=lambda _embed_dim, irrep_dims: len(irrep_dims),
        extra_flops_per_layer=lambda _embed_dim, irrep_dims: float(sum(
            2.0 * dim ** 3 + 2.0 * dim ** 2 for dim in irrep_dims
        )),
        report_label=lambda _covariance: "invariant_scalar_gate",
        # Diagonal layout: s_b = sum_k mu_k * (mu_k / sigma_k), a divide, a multiply and an add per
        # coordinate, in place of the per-block Cholesky solve.
        diagonal_extra_flops_per_layer=lambda _embed_dim, irrep_dims: 3.0 * float(sum(irrep_dims)),
    ),
    "canonical_frame": BlockMLPRegistration(
        factory=_canonical_frame_factory,
        requires_frame_context=True,
        covariance_kinds=_BLOCK_MLP_COVARIANCE_CONTRACTS,
        gauge_class="left_equivariant_right_fixed",
        intertwiner_compatible=False,
        width=lambda embed_dim, _irrep_dims: embed_dim,
        extra_flops_per_layer=lambda embed_dim, _irrep_dims: float(4 * embed_dim ** 2),
        report_label=lambda _covariance: "canonical_frame_left_equivariant_right_fixed",
    ),
    # nu, log_lambda, values: (M, K) each; log_pi: (M,); M = expansion * K.
    "gaussian_template": BlockMLPRegistration(
        factory=_gaussian_template_factory,
        requires_frame_context=False,
        covariance_kinds=frozenset({"passthrough"}),
        gauge_class="coordinate_nonintertwiner",
        intertwiner_compatible=False,
        width=lambda embed_dim, _irrep_dims: embed_dim,
        extra_flops_per_layer=lambda _embed_dim, _irrep_dims: 0.0,
        report_label=lambda _covariance: "gaussian_template_mean_only_nonintertwiner",
        parameter_count_per_layer=lambda width, expansion: (
            3 * expansion * width * width + expansion * width),
        matmul_flops_per_layer=lambda width, expansion: 6.0 * expansion * width * width,
    ),
})


def get_block_mlp_registration(mode: str) -> BlockMLPRegistration:
    try:
        return BLOCK_MLP_REGISTRATIONS[mode]
    except KeyError as exc:
        raise ValueError(
            f"unknown block_mlp_mode {mode!r}; available: {sorted(BLOCK_MLP_REGISTRATIONS)}"
        ) from exc


def block_mlp_build_metadata(
    *,
    enabled: bool,
    mode: str,
    covariance: str,
    expansion: int,
    activation: str,
    dropout: float,
    covariance_floor: float,
    embed_dim: int,
    irrep_dims: Sequence[int],
    n_layers: int,
    covariance_layout: str = "full",
) -> BlockMLPBuildMetadata:
    r"""Executable build record; ``covariance_layout`` ('full' | 'diagonal') selects layout-dependent FLOPs."""
    registration = get_block_mlp_registration(mode)
    if covariance not in registration.covariance_kinds:
        raise ValueError(f"unknown BlockMLP covariance contract {covariance!r}")
    realized_irrep_dims = tuple(int(dim) for dim in irrep_dims)
    realized_parameter_count = registration.parameter_count(
        embed_dim=embed_dim,
        irrep_dims=realized_irrep_dims,
        expansion=expansion,
        n_layers=n_layers,
    )
    realized_flops_per_token = registration.flops_per_token(
        embed_dim=embed_dim,
        irrep_dims=realized_irrep_dims,
        expansion=expansion,
        covariance_contract=covariance,
        n_layers=n_layers,
        covariance_layout=covariance_layout,
    )
    covariance_report = (
        "not_applicable" if not enabled else
        "delta_full_plus_covariant_floor" if covariance == "delta_full" else covariance
    )
    return BlockMLPBuildMetadata(
        enabled=bool(enabled),
        mode=mode,
        covariance=covariance,
        expansion=int(expansion),
        activation=activation,
        dropout=float(dropout),
        covariance_floor=float(covariance_floor),
        embed_dim=int(embed_dim),
        irrep_dims=realized_irrep_dims,
        n_layers=int(n_layers),
        parameter_count=(realized_parameter_count if enabled else 0),
        flops_per_token=(realized_flops_per_token if enabled else 0.0),
        report_label=(registration.report_label(covariance) if enabled else "disabled"),
        covariance_report_label=covariance_report,
        gauge_class=(registration.gauge_class if enabled else "disabled"),
        intertwiner_compatible=(registration.intertwiner_compatible if enabled else True),
    )


def build_block_mlp(
    mode: str,
    *,
    embed_dim: int,
    irrep_dims: Sequence[int],
    expansion: int,
    activation: str,
    dropout: float,
    covariance_contract: str,
    covariance_floor: float,
) -> nn.Module:
    registration = get_block_mlp_registration(mode)
    if covariance_contract not in registration.covariance_kinds:
        raise ValueError(f"unknown BlockMLP covariance contract {covariance_contract!r}")
    return registration.factory(
        embed_dim=embed_dim,
        irrep_dims=irrep_dims,
        expansion=expansion,
        activation=activation,
        dropout=dropout,
        covariance_contract=covariance_contract,
        covariance_floor=covariance_floor,
    )
