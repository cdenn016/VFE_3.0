r"""Role-specific comparison and content operators in realized canonical frames."""

import math
from typing import List, Optional, Tuple

import torch
from torch import nn

from vfe3.contracts import CanonicalFrameContext
from vfe3.relational_scoring import _compute_dtype


class RelationalOperators(nn.Module):
    r"""Headwise comparison maps S_Q, S_K and identity or full-width content B=OT.

    Every rank is the head dimension: S_Q/S_K are ``(H, d, input_dim)`` and T/O are
    ``(H, d, input_dim)`` / ``(H, output_dim, d)``, the untied standard-attention shapes.
    Predictive content always reads and writes the full state; conditional updates retain their scopes.
    Independent maps are free matrices initialized with orthonormal rows.
    Their projected covariances must remain strictly positive definite.
    ``identity_content`` drops T/O and carries the transported belief unchanged.
    ``output_init_scale`` multiplies the initial output map, :math:`O_h(0) = s\,I_d` (placed on its
    head block under full output scope); ``s = 1`` is the identity start.
    ``routing`` selects the projected routing cost: Gaussian KL, or the noisy-query cost
    N(a_i; b_j, V_Q + V_K) of the expected likelihood kernel (predictive_content only).
    """

    def __init__(
        self,
        irrep_dims:            List[int],
        seed:                  int,
        readout_mode:          str   = "conditional_update",
        attention_residual:    bool  = False,
        identity_content:      bool  = False,
        output_init_scale:     float = 1.0,
        routing:               str   = "gaussian_kl",
        predictive_covariance: str   = "passthrough",
        predictive_covariance_floor: float = 1e-4,
        predictive_covariance_centering: bool = False,
        input_scope:           Optional[str] = None,
        output_scope:          Optional[str] = None,
    ) -> None:
        super().__init__()
        if not irrep_dims or any(type(d) is not int or d < 1 for d in irrep_dims) or len(set(irrep_dims)) != 1:
            raise ValueError("relational operators require positive equal-sized integer heads")
        d, H = irrep_dims[0], len(irrep_dims)
        if type(seed) is not int:
            raise ValueError("relational seed must be an integer")
        if type(identity_content) is not bool:
            raise ValueError("identity_content must be a bool")
        if (isinstance(output_init_scale, bool) or not isinstance(output_init_scale, (int, float))
                or not math.isfinite(output_init_scale) or output_init_scale <= 0.0):
            raise ValueError(
                f"output_init_scale must be finite and positive, got {output_init_scale!r}")
        if attention_residual and readout_mode != "predictive_content":
            raise ValueError("attention residual requires predictive_content")
        self.attention_residual = attention_residual
        default_scope = "full" if readout_mode == "predictive_content" else "head"
        input_scope = default_scope if input_scope is None else input_scope
        output_scope = default_scope if output_scope is None else output_scope
        if readout_mode == "predictive_content" and (input_scope != "full" or output_scope != "full"):
            raise ValueError("predictive_content fixes input_scope and output_scope to 'full'")
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
        if routing not in ("gaussian_kl", "noisy_query"):
            raise ValueError("relational routing must be 'gaussian_kl' or 'noisy_query'")
        if routing == "noisy_query" and readout_mode != "predictive_content":
            raise ValueError("noisy_query routing requires predictive_content")
        self.routing = routing
        if predictive_covariance not in ("passthrough", "moment_matched"):
            raise ValueError("predictive_covariance must be 'passthrough' or 'moment_matched'")
        if predictive_covariance == "moment_matched" and readout_mode != "predictive_content":
            raise ValueError("moment_matched predictive covariance requires predictive_content")
        if (isinstance(predictive_covariance_floor, bool)
                or not isinstance(predictive_covariance_floor, (int, float))
                or not math.isfinite(predictive_covariance_floor) or predictive_covariance_floor <= 0.0):
            raise ValueError(
                f"predictive_covariance_floor must be finite and positive, got {predictive_covariance_floor!r}")
        if type(predictive_covariance_centering) is not bool:
            raise ValueError("predictive_covariance_centering must be a bool")
        self.predictive_covariance_centering = predictive_covariance_centering
        self.predictive_covariance = predictive_covariance
        self.predictive_covariance_floor = float(predictive_covariance_floor)
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
            self.O = nn.Parameter(eye.clone() * float(output_init_scale))
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
            coordinates = canonical.flatten(-2).unsqueeze(-2)              # (..., N, 1, K)
            values = (self.T.to(dtype) @ coordinates.to(dtype).unsqueeze(-1)).squeeze(-1)
            pooled = torch.einsum("...hij,...jhs->...ihs", beta.to(dtype), values)
            # Sum heads inside the contraction, avoiding a (..., N, H, K) result.
            body = torch.einsum("hks,...nhs->...nk", self.O.to(dtype), pooled)
            if frame.block_diagonal:
                body = body.reshape(*mu.shape[:-1], H, d)
            result = (frame.forward.to(dtype) @ body.unsqueeze(-1)).squeeze(-1)
        return result.reshape_as(mu).to(mu.dtype)

    def _ambient_content_maps(self, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Return (T_hat, O_hat), each head's content maps embedded in the full state space.

        T_hat is ``(H, d, K)`` and O_hat is ``(H, K, d)``, so M_h = O_hat_h T_hat_h is the head's
        K x K content map in every input/output scope. Head scope places T_h / O_h on head block h.
        """
        H, d = len(self.irrep_dims), self.irrep_dims[0]
        eye = torch.eye(H, dtype=dtype, device=self.T.device)
        T, O = self.T.to(dtype), self.O.to(dtype)
        if self.input_scope == "head":
            T = torch.einsum("hab,hg->hagb", T, eye).reshape(H, d, H * d)
        if self.output_scope == "head":
            O = torch.einsum("hab,hg->hgab", O, eye).reshape(H, H * d, d)
        return T, O

    def predictive_covariance_update(
        self,
        mu:    torch.Tensor,  # (..., N, K) source means, the values predictive_content reads
        sigma: torch.Tensor,  # (..., N, K) diagonal or (..., N, K, K) full local covariance
        beta:  torch.Tensor,  # (..., H, N, N) routing weights, rows sum to one
        frame: CanonicalFrameContext,
    ) -> torch.Tensor:        # same layout as sigma: the moment-matched covariance Sigma_i^+
        r"""Moment-matched covariance of the predictive kernel (GL(K) v2, Theorem thm:probabilistic_status (2)).

        With independent head labels J^h ~ beta_i^h, independent sources k_j ~ N(mu_j, Sigma_j), and
        the kernel k_i' | k, J ~ N(r k_i + sum_h L_{iJ^h}^h k_{J^h}, Q_i), where r = 1 under
        ``attention_residual`` and 0 otherwise, the covariance in reference coordinates is

        .. math::
            C_i^{\mathrm{att}} = \widetilde Q_i
              + r\Bigl[C_i + \sum_h \beta_{ii}^h (M_h C_i + C_i M_h^\top)\Bigr]
              + \sum_j \sum_{h,h'} c_{ij}^{hh'} M_h C_j M_{h'}^\top
              + \sum_h \Bigl[\sum_j \beta_{ij}^h n_{ij}^h n_{ij}^{h\top} - \bar n_i^h \bar n_i^{h\top}\Bigr],

        with M_h = O_h T_h, C_j = U_j^{-1} Sigma_j U_j^{-T}, c^{hh} = beta^h, c^{hh'} = beta^h beta^{h'}
        (h != h'), n_ij^h = M_h xi_j, and the reference floor Q~_i = floor * (tr C_i / K) I, which is
        covariant (local Q_i = floor * (tr C_i / K) U_i U_i^T, with tr C_i frame-invariant).
        The result is Sigma_i^+ = U_i C_i^att U_i^T in the frame the mean content uses. A diagonal
        input returns diag(Sigma_i^+), the KL(p || q) moment projection onto the diagonal family.
        Head-space pooling keeps the cost at an attention with d x d values per head pair.
        """
        H, d = len(self.irrep_dims), self.irrep_dims[0]
        K = H * d
        diagonal = sigma.dim() == mu.dim()
        dtype = _compute_dtype(mu, sigma, self.T, self.O, frame.forward)
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
            T_hat, O_hat = self._ambient_content_maps(dtype)                       # (H,d,K), (H,K,d)
            b = beta.to(dtype)                                                     # (..., H, N, N)
            eye_h = torch.eye(H, dtype=dtype, device=mu.device)

            if self.predictive_covariance_centering:
                # A shared shift preserves covariance for normalized beta and reduces cancellation.
                xi = xi - xi[..., :1, :]                                          # (..., N, K)
            values = torch.einsum("hsk,...nk->...nhs", T_hat, xi)                  # (..., N, H, d)
            TC = torch.einsum("hsk,...nkm->...nhsm", T_hat, C)                     # (..., N, H, d, K)
            source = torch.einsum("...nhsm,gtm->...nhgst", TC, T_hat)             # (..., N, H, H, d, d)
            # c^{hh'}_ij: beta^h on the head diagonal, beta^h beta^h' off it.
            c = torch.einsum("...hij,...gij->...hgij", b, b)
            c = c * (1.0 - eye_h)[..., None, None] + torch.einsum("...hij,hg->...hgij", b, eye_h)
            pooled = torch.einsum("...hgij,...jhgst->...ihgst", c, source)        # (..., N, H, H, d, d)
            second = torch.einsum("...hij,...jhst->...ihst",
                                  b, values.unsqueeze(-1) * values.unsqueeze(-2))  # (..., N, H, d, d)
            mean = torch.einsum("...hij,...jhs->...ihs", b, values)                # (..., N, H, d)
            spread = second - mean.unsqueeze(-1) * mean.unsqueeze(-2)
            pooled = pooled + torch.einsum("...ihst,hg->...ihgst", spread, eye_h)
            lifted = torch.einsum("hks,...nhgst->...ngkt", O_hat, pooled)         # (..., N, H, K, d)
            C_att = torch.einsum("...ngkt,gmt->...nkm", lifted, O_hat)             # (..., N, K, K)
            if self.attention_residual:
                self_weight = b.diagonal(dim1=-2, dim2=-1).transpose(-1, -2)       # (..., N, H) beta_ii^h
                cross = torch.einsum("hks,...nhsm->...nkm", O_hat,
                                     TC * self_weight[..., None, None])            # sum_h beta_ii M_h C_i
                C_att = C_att + C + cross + cross.transpose(-1, -2)
            # Isotropic reference floor scaled by the invariant tr(C_i)/K: a relative floor
            # floor * C_i compounds as floor**depth along a residual-free stack and loses float32
            # definiteness; this one keeps cond(C_att) bounded by the content spread over floor.
            scale = torch.diagonal(C, dim1=-2, dim2=-1).mean(dim=-1)              # (..., N) tr(C_i)/K
            C_att = C_att + (self.predictive_covariance_floor * scale)[..., None, None] * torch.eye(
                K, dtype=dtype, device=mu.device)
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
        # An escalated float64 frame (audit 2026-09-23 #12) promotes the computation only; the
        # result keeps the dtype the lower-precision frame would have produced.
        return result.reshape(*mu.shape[:-1], mu.shape[-1]).to(_compute_dtype(mu, self.T, self.O))
