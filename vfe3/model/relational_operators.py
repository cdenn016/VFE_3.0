r"""Role-specific comparison and content operators in realized canonical frames."""

from typing import List, Optional, Tuple

import torch
from torch import nn

from vfe3.contracts import CanonicalFrameContext
from vfe3.relational_scoring import _compute_dtype


class RelationalOperators(nn.Module):
    r"""Headwise comparison maps S_Q, S_K and identity or full-width content B=OT.

    Every rank is the head dimension: S_Q/S_K are ``(H, d, input_dim)`` and T/O are
    ``(H, d, input_dim)`` / ``(H, output_dim, d)``, the untied standard-attention shapes.
    Independent maps are free matrices initialized with orthonormal rows.
    Their projected covariances must remain strictly positive definite.
    ``identity_content`` drops T/O and carries the transported belief unchanged.
    """

    def __init__(
        self,
        irrep_dims:            List[int],
        seed:                  int,
        input_scope:           str = "head",
        readout_mode:          str = "conditional_update",
        output_scope:          str = "head",
        attention_residual:    bool = False,
        identity_content:      bool = False,
    ) -> None:
        super().__init__()
        if not irrep_dims or any(type(d) is not int or d < 1 for d in irrep_dims) or len(set(irrep_dims)) != 1:
            raise ValueError("relational operators require positive equal-sized integer heads")
        d, H = irrep_dims[0], len(irrep_dims)
        if type(seed) is not int:
            raise ValueError("relational seed must be an integer")
        if type(identity_content) is not bool:
            raise ValueError("identity_content must be a bool")
        if attention_residual and readout_mode != "predictive_content":
            raise ValueError("attention residual requires predictive_content")
        self.attention_residual = attention_residual
        if input_scope not in ("head", "full"):
            raise ValueError("relational input_scope must be 'head' or 'full'")
        if readout_mode not in ("conditional_update", "predictive_content"):
            raise ValueError("invalid relational readout_mode")
        if output_scope not in ("head", "full"):
            raise ValueError("invalid relational output_scope")
        if output_scope == "full" and readout_mode != "predictive_content":
            raise ValueError("full output requires predictive_content")
        if readout_mode == "predictive_content" and identity_content:
            raise ValueError("predictive_content requires learned content maps")
        self.readout_mode = readout_mode
        self.output_scope = output_scope
        self.input_scope = input_scope
        self.input_dim = H * d if input_scope == "full" else d
        self.irrep_dims = list(irrep_dims)
        self.score_rank_per_head = d
        self.content_rank_per_head = None if identity_content else d
        self.identity_content = identity_content
        generator = torch.Generator().manual_seed(seed)
        eye = torch.eye(d).expand(H, d, d).clone()
        for role in ("q", "k"):
            raw = torch.randn(H, self.input_dim, d, generator=generator)
            rows = self._orthonormal_columns(raw).transpose(-1, -2).contiguous()
            setattr(self, f"raw_score_{role}", nn.Parameter(rows))
        if identity_content:
            self.register_parameter("T", None)
            self.register_parameter("O", None)
        else:
            self.T = nn.Parameter(eye.clone())
            self.O = nn.Parameter(eye.clone())
        if input_scope == "full" and self.T is not None:
            # Preserve the local identity/projector start; all input columns train.
            T_full = self.T.new_zeros(H, d, H * d)
            for h in range(H):
                T_full[h, :, h*d:(h+1)*d] = self.T.detach()[h]
            self.T = nn.Parameter(T_full)
        if output_scope == "full":
            # Start at the old block placement; off-block output rows are trainable.
            O_full = self.O.new_zeros(H, H * d, d)
            for h in range(H):
                O_full[h, h*d:(h+1)*d] = self.O.detach()[h]
            self.O = nn.Parameter(O_full)

    @property
    def inference_semantics(self) -> str:
        """Name the actual mean update used by this module."""
        return ("relational_predictive_content" if self.readout_mode == "predictive_content"
                else "relational_projected_conditional_filtering")

    def predictive_content(
        self,
        mu:    torch.Tensor,  # (..., N, K), source means with live gradients
        beta:  torch.Tensor,  # (..., H, N, N)
        frame: CanonicalFrameContext,
    ) -> torch.Tensor:
        r"""Compute m_i = A_i sum_h O_h sum_j beta_ij^h T_h B_j^{-1} mu_j."""
        canonical = self.canonical_mean(mu, frame)
        H, d = len(self.irrep_dims), self.irrep_dims[0]
        dtype = _compute_dtype(mu, self.T, self.O, frame.forward)
        with torch.autocast(device_type=mu.device.type, enabled=False):
            coordinates = canonical.flatten(-2).unsqueeze(-2) if self.input_scope == "full" else canonical
            values = (self.T.to(dtype) @ coordinates.to(dtype).unsqueeze(-1)).squeeze(-1)
            pooled = torch.einsum("...hij,...jhs->...ihs", beta.to(dtype), values)
            mixed = (self.O.to(dtype) @ pooled.unsqueeze(-1)).squeeze(-1)
            body = mixed.sum(-2) if self.output_scope == "full" else mixed.flatten(-2)
            if frame.block_diagonal:
                body = body.reshape(*mu.shape[:-1], H, d)
            result = (frame.forward.to(dtype) @ body.unsqueeze(-1)).squeeze(-1)
        return result.reshape_as(mu).to(mu.dtype)

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

    def comparison_pair(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Return (S_Q,S_K), each (H,r,input_dim), with live role graphs."""
        return self.raw_score_q, self.raw_score_k

    def content_factors(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        r"""Return (T,O) for B=OT, or None for parameter-free identity content."""
        return None if self.T is None else (self.T, self.O)

    def canonical_mean(
        self,
        mu:    torch.Tensor,  # (..., N, K)
        frame: CanonicalFrameContext,
    ) -> torch.Tensor:        # (..., N, H, d)
        r"""Compute xi_i=U_i^{-1}mu_i using the supplied realized ordered frame."""
        H, d = len(self.irrep_dims), self.irrep_dims[0]
        if mu.shape[-1] != H * d:
            raise ValueError("relational mean dimension does not match its head partition")
        if frame.block_diagonal and frame.inverse.shape[-3:] != (H, d, d):
            raise ValueError("relational compact frame blocks do not match its head partition")
        if not frame.block_diagonal and frame.inverse.shape[-2:] != (H * d, H * d):
            raise ValueError("relational dense frame dimension does not match the mean")
        dtype = _compute_dtype(mu, frame.inverse)
        with torch.autocast(device_type=mu.device.type, enabled=False):
            coordinates = mu.to(dtype).reshape(*mu.shape[:-1], H, d) if frame.block_diagonal else mu.to(dtype)
            canonical = (frame.inverse.to(dtype) @ coordinates.unsqueeze(-1)).squeeze(-1)
        return canonical.reshape(*mu.shape[:-1], H, d)

    def apply_content(
        self,
        mu:    torch.Tensor,  # (..., N, K)
        frame: CanonicalFrameContext,
    ) -> torch.Tensor:        # (..., N, K)
        r"""Apply the local content template U_i O T U_i^{-1}mu_i to means only."""
        if self.output_scope == "full":
            raise ValueError("full output mixes after routing; use predictive_content")
        canonical = self.canonical_mean(mu, frame)
        if self.T is None:
            return mu
        dtype = _compute_dtype(canonical, frame.forward, self.T, self.O)
        with torch.autocast(device_type=mu.device.type, enabled=False):
            coordinates = canonical.flatten(-2).unsqueeze(-2) if self.input_scope == "full" else canonical
            content = self.T.to(dtype) @ coordinates.to(dtype).unsqueeze(-1)
            body = (self.O.to(dtype) @ content).squeeze(-1)
            if not frame.block_diagonal:
                body = body.reshape(*mu.shape[:-1], mu.shape[-1])
            result = (frame.forward.to(dtype) @ body.unsqueeze(-1)).squeeze(-1)
        return result.reshape(*mu.shape[:-1], mu.shape[-1])
