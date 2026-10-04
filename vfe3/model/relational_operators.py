r"""Role-specific comparison and content operators in realized canonical frames."""

import math
from numbers import Real
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from vfe3.contracts import CanonicalFrameContext
from vfe3.relational_scoring import _compute_dtype

# Receiving-map charts of the content link on the stored factors (O, T) (attention_unified.tex,
# "The tiled message order by order"). A later truncation such as "second_order" adds one branch
# to ``chart_receive_map`` and one initial value to ``_chart_initial_output``.
CONTENT_LINK_CHARTS: Tuple[str, ...] = ("affine", "first_order", "exponential")


def _phi1(A: torch.Tensor) -> torch.Tensor:          # (n, n) -> (n, n)
    r"""Return :math:`\varphi_1(A)=\sum_{n\ge1}A^{n-1}/n!` without inverting :math:`A`.

    .. math::
        \exp\begin{pmatrix}A & I\\ 0 & 0\end{pmatrix}
        =\begin{pmatrix}e^A & \varphi_1(A)\\ 0 & I\end{pmatrix},

    because the block matrix has :math:`n`-th power :math:`\begin{pmatrix}A^n & A^{n-1}\\ 0 & 0\end{pmatrix}`
    for :math:`n\ge1`.
    """
    n = A.shape[-1]
    eye = torch.eye(n, dtype=A.dtype, device=A.device)
    augmented = torch.cat([torch.cat([A, eye], dim=-1),
                           torch.zeros(n, 2 * n, dtype=A.dtype, device=A.device)], dim=-2)   # (2n, 2n)
    return torch.linalg.matrix_exp(augmented)[:n, n:]


def chart_receive_map(
    T:     torch.Tensor,                # (H, d, K) stored sending maps, H*d = K
    O:     torch.Tensor,                # (H, K, d) stored receiving factors
    chart: str,
) -> torch.Tensor:                      # (H, K, d) effective receiving maps O'_h
    r"""Effective receiving maps :math:`O'` of the tiled content link in the chosen chart.

    With the stacked :math:`T` (:math:`Hd\times K`) and :math:`O` (:math:`K\times Hd`) square, the
    predictive message is :math:`U_i\sum_jO'B_{ij}T\xi_j` with :math:`B_{ij}=\bigoplus_h\beta^h_{ij}I_d` and

    .. math::
        O'_{\mathrm{affine}}=O,\qquad
        O'_{\mathrm{first\ order}}=T^{-1}+O,\qquad
        O'_{\mathrm{exponential}}=T^{-1}+O\,\varphi_1(TO)=e^{OT}T^{-1}.

    The exponential chart is the shared-generator exponential link :math:`X=OT` with routing outside
    the exponent (:math:`O'T=e^{OT}`, so the effective transfer :math:`TO'=e^{TO}` has positive
    determinant); its order-:math:`n` term is :math:`O(TO)^{n-1}/n!` and order zero is
    :math:`T^{-1}B_{ij}T=\sum_h\beta^h_{ij}P_h`, the head-routed flat transport. The first-order chart
    keeps orders zero and one. Non-affine charts invert :math:`T` and are evaluated in float64.
    """
    if chart == "affine":
        return O
    if chart not in CONTENT_LINK_CHARTS:
        raise ValueError(f"content_link_chart must be one of {CONTENT_LINK_CHARTS}, got {chart!r}")
    H, d, K = T.shape
    if H * d != K or tuple(O.shape) != (H, K, d):
        raise ValueError(f"content link charts need tiled square stacked maps, got T {tuple(T.shape)}, "
                         f"O {tuple(O.shape)}")
    with torch.autocast(device_type=T.device.type, enabled=False):
        T_stack = T.to(torch.float64).reshape(H * d, K)                           # (Hd, K) rows by head
        O_stack = O.to(torch.float64).permute(1, 0, 2).reshape(K, H * d)          # (K, Hd) columns by head
        # inv_ex: no host sync (linalg.inv checks info on the host); a singular T yields non-finite maps.
        receive = torch.linalg.inv_ex(T_stack).inverse                            # (K, Hd) order zero
        if chart == "first_order":
            receive = receive + O_stack
        else:
            receive = receive + O_stack @ _phi1(T_stack @ O_stack)                # all orders
    return receive.reshape(K, H, d).permute(1, 0, 2)                              # (H, K, d)


def _chart_initial_output(chart: str, scale: float) -> float:
    r"""Initial :math:`O=cI` (with :math:`T=I`) giving the content :math:`sB_{ij}` in every chart."""
    return {"affine": scale, "first_order": scale - 1.0, "exponential": math.log(scale)}[chart]


class RelationalOperators(nn.Module):
    r"""Full-width comparison maps S_Q, S_K and content B=OT.

    Every rank is the head dimension: S_Q/S_K are ``(H, d, input_dim)`` and T/O are
    ``(H, d, input_dim)`` / ``(H, output_dim, d)``, the untied standard-attention shapes.
    Every head reads and writes the full state while retaining its reduced channel rank.
    Independent maps are free matrices initialized with orthonormal rows.
    Their projected covariances must remain strictly positive definite.
    ``output_init_scale`` sets the initial content :math:`sB_{ij}` (placed on the head blocks under full
    output scope); ``s = 1`` is the identity start. ``content_link_chart`` selects the effective
    receiving maps built from the stored (O, T) (:func:`chart_receive_map`); the stored O starts at
    :math:`sI`, :math:`(s-1)I` or :math:`(\log s)I` per chart so every chart starts at :math:`sB_{ij}`.
    Every consumer reads the content maps through :meth:`content_factors`.
    ``routing`` selects the projected routing cost: Gaussian KL, or the noisy-query cost
    N(a_i; b_j, V_Q + V_K) of the expected likelihood kernel (predictive_content only).
    ``predictive_covariance_checkpoint`` recomputes the moment-matched covariance update in backward
    instead of saving its (..., N, K, K) intermediates: the same values and gradients bit for bit,
    for about half the activation memory of a large-K step.
    """

    def __init__(
        self,
        irrep_dims:            List[int],
        seed:                  int,
        readout_mode:          str   = "predictive_content",
        attention_residual:    bool  = False,
        output_init_scale:     float = 1.0,
        routing:               str   = "gaussian_kl",
        predictive_covariance: str   = "passthrough",
        predictive_covariance_floor: float = 1e-4,
        *,
        receiver_noise:        float = 1.0,
        receiver_law:          str   = "ambient_alignment",
        receiver_update:       str   = "euler",
        receiver_damping:      float = 1.0,
        content_link_chart:    str   = "affine",
        predictive_covariance_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        if not irrep_dims or any(type(d) is not int or d < 1 for d in irrep_dims) or len(set(irrep_dims)) != 1:
            raise ValueError("relational operators require positive equal-sized integer heads")
        d, H = irrep_dims[0], len(irrep_dims)
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
        if receiver_law not in ("ambient_alignment", "channel_likelihood"):
            raise ValueError("receiver_law must be 'ambient_alignment' or 'channel_likelihood'")
        if receiver_update not in ("euler", "closed_form"):
            raise ValueError("receiver_update must be 'euler' or 'closed_form'")
        if (isinstance(receiver_damping, bool) or not isinstance(receiver_damping, Real)
                or not math.isfinite(receiver_damping) or not 0.0 < receiver_damping <= 1.0):
            raise ValueError("receiver_damping must lie in (0, 1]")
        # Full-link receiver law and update (projected_gradient.tex); inert under predictive_content.
        self.receiver_law = receiver_law
        self.receiver_update = receiver_update
        self.receiver_damping = float(receiver_damping)
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
        self.predictive_covariance = predictive_covariance
        self.predictive_covariance_floor = float(predictive_covariance_floor)
        if not isinstance(predictive_covariance_checkpoint, bool):
            raise ValueError("predictive_covariance_checkpoint must be bool, got "
                             f"{predictive_covariance_checkpoint!r}")
        self.predictive_covariance_checkpoint = predictive_covariance_checkpoint
        if content_link_chart not in CONTENT_LINK_CHARTS:
            raise ValueError(
                f"content_link_chart must be one of {CONTENT_LINK_CHARTS}, got {content_link_chart!r}")
        self.content_link_chart = content_link_chart
        self.input_dim = H * d
        self.irrep_dims = list(irrep_dims)
        self.score_rank_per_head = d
        self.content_rank_per_head = d
        generator = torch.Generator().manual_seed(seed)
        eye = torch.eye(d).expand(H, d, d).clone()
        for role in ("q", "k"):
            raw = torch.randn(H, self.input_dim, d, generator=generator)
            rows = self._orthonormal_columns(raw).transpose(-1, -2).contiguous()
            setattr(self, f"raw_score_{role}", nn.Parameter(rows))
        # Preserve the released block-projector initialization and parameter names.
        T_full = eye.new_zeros(H, d, H * d)
        O_full = eye.new_zeros(H, H * d, d)
        initial_output = _chart_initial_output(content_link_chart, float(output_init_scale))
        for h in range(H):
            T_full[h, :, h*d:(h+1)*d] = eye[h]
            O_full[h, h*d:(h+1)*d] = eye[h] * initial_output
        self.T = nn.Parameter(T_full)
        self.O = nn.Parameter(O_full)

    @property
    def inference_semantics(self) -> str:
        """Name the actual mean update used by this module."""
        return ("relational_full_link_frozen_filtering" if self.readout_mode == "full_link_update"
                else "relational_predictive_content")

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
        receive: Optional[torch.Tensor] = None,   # (H, K, d) precomputed O'_h; None evaluates the chart
    ) -> torch.Tensor:
        r"""Compute m_i = A_i sum_h O'_h sum_j beta_ij^h T_h B_j^{-1} mu_j (O' of content_link_chart)."""
        canonical = self.canonical_mean(mu, frame)
        H, d = len(self.irrep_dims), self.irrep_dims[0]
        dtype = _compute_dtype(mu, self.T, self.O, frame.forward)
        if receive is None:
            _, receive = self.content_factors()                            # (H, K, d) O'_h
        with torch.autocast(device_type=mu.device.type, enabled=False):
            coordinates = canonical.flatten(-2).unsqueeze(-2)              # (..., N, 1, K)
            values = (self.T.to(dtype) @ coordinates.to(dtype).unsqueeze(-1)).squeeze(-1)
            pooled = torch.einsum("...hij,...jhs->...ihs", beta.to(dtype), values)
            # Sum heads inside the contraction, avoiding a (..., N, H, K) result.
            body = torch.einsum("hks,...nhs->...nk", receive.to(dtype), pooled)
            if frame.block_diagonal:
                body = body.reshape(*mu.shape[:-1], H, d)
            result = (frame.forward.to(dtype) @ body.unsqueeze(-1)).squeeze(-1)
        return result.reshape_as(mu).to(mu.dtype)

    def _ambient_content_maps(
        self,
        dtype:   torch.dtype,
        receive: Optional[torch.Tensor] = None,   # (H, K, d) precomputed O'_h; None evaluates the chart
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Return full-state T_h and effective O'_h of shapes (H,d,K) and (H,K,d)."""
        if receive is None:
            _, receive = self.content_factors()
        return self.T.to(dtype), receive.to(dtype)

    def predictive_covariance_update(
        self,
        mu:    torch.Tensor,  # (..., N, K) source means, the values predictive_content reads
        sigma: torch.Tensor,  # (..., N, K) diagonal or (..., N, K, K) full local covariance
        beta:  torch.Tensor,  # (..., H, N, N) routing weights, rows sum to one
        frame: CanonicalFrameContext,
        receive: Optional[torch.Tensor] = None,   # (H, K, d) precomputed O'_h; None evaluates the chart
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
        Under ``predictive_covariance_checkpoint`` the update is recomputed in backward (non-reentrant
        checkpoint; the update draws no random numbers), so values and gradients are unchanged.
        """
        if self.predictive_covariance_checkpoint and torch.is_grad_enabled():
            return checkpoint(self._predictive_covariance_update, mu, sigma, beta, frame, receive,
                              use_reentrant=False, preserve_rng_state=False)
        return self._predictive_covariance_update(mu, sigma, beta, frame, receive)

    def _predictive_covariance_update(
        self,
        mu:    torch.Tensor,  # (..., N, K)
        sigma: torch.Tensor,  # (..., N, K) or (..., N, K, K)
        beta:  torch.Tensor,  # (..., H, N, N)
        frame: CanonicalFrameContext,
        receive: Optional[torch.Tensor] = None,   # (H, K, d)
    ) -> torch.Tensor:
        r"""Body of :meth:`predictive_covariance_update`: blockwise for diagonal beliefs on block frames."""
        if frame.block_diagonal and sigma.dim() == mu.dim():
            return self._predictive_covariance_update_blockwise(mu, sigma, beta, frame, receive)
        return self._predictive_covariance_update_dense(mu, sigma, beta, frame, receive)

    def _predictive_covariance_update_blockwise(
        self,
        mu:    torch.Tensor,  # (..., N, K) source means
        sigma: torch.Tensor,  # (..., N, K) diagonal local covariance
        beta:  torch.Tensor,  # (..., H, N, N) routing weights, rows sum to one
        frame: CanonicalFrameContext,             # block_diagonal: (..., N, H, d, d) blocks U_{i,m}
        receive: Optional[torch.Tensor] = None,   # (H, K, d) precomputed O'_h; None evaluates the chart
    ) -> torch.Tensor:        # (..., N, K) diag(Sigma_i^+)
        r"""Diagonal of the moment-matched covariance for diagonal beliefs on block-diagonal frames.

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
              + r\bigl[C_{i,l} + X_{i,l} + X_{i,l}^\top\bigr] + \tilde q_i I_d,
            \qquad X_{i,l} = \sum_h \beta_{ii}^h O_h^{(l)} T_h^{(l)} C_{i,l},

        with :math:`P_{i,hg}` the head-pair pooled covariance of
        :meth:`_predictive_covariance_update_dense` and :math:`\tilde q_i = f\,\mathrm{tr}(C_i)/K`. It is the
        dense update reassociated, equal up to rounding, at about :math:`2K^3 + NK^2` multiply-adds per
        token instead of about :math:`9.5K^3 + NK^2`, with no :math:`K \times K` intermediate.
        """
        H, d = len(self.irrep_dims), self.irrep_dims[0]
        dtype = _compute_dtype(mu, sigma, self.T, self.O, frame.forward)
        with torch.autocast(device_type=mu.device.type, enabled=False):
            forward, inverse = frame.forward.to(dtype), frame.inverse.to(dtype)        # (..., N, H, d, d)
            sigma_h = sigma.to(dtype).reshape(*sigma.shape[:-1], H, d)                 # (..., N, H, d)
            C = (inverse * sigma_h.unsqueeze(-2)) @ inverse.transpose(-1, -2)          # (..., N, H, d, d) C_{i,m}
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
            TC = torch.einsum("hsmk,...nmkt->...nhmst", T_blocks, C)                  # (..., N, H, H, d, d) T_h^{(m)} C_{i,m}
            source = torch.einsum("...nhmsk,gtmk->...nhgst", TC, T_blocks)            # (..., N, H, H, d, d) T_h C_i T_g^T
            c = torch.einsum("...hij,...gij->...hgij", b, b)
            c = c * (1.0 - eye_h)[..., None, None] + torch.einsum("...hij,hg->...hgij", b, eye_h)
            pooled = torch.einsum("...hgij,...jhgst->...ihgst", c, source)            # (..., N, H, H, d, d)
            second = torch.einsum("...hij,...jhst->...ihst",
                                  b, values.unsqueeze(-1) * values.unsqueeze(-2))      # (..., N, H, d, d)
            mean = torch.einsum("...hij,...jhs->...ihs", b, values)                    # (..., N, H, d)
            spread = second - mean.unsqueeze(-1) * mean.unsqueeze(-2)
            pooled = pooled + torch.einsum("...ihst,hg->...ihgst", spread, eye_h)
            lifted = torch.einsum("hks,...nhgst->...ngkt", O_hat, pooled)             # (..., N, H, K, d) sum_h O'_h P_hg
            lifted = lifted.reshape(*lifted.shape[:-2], H, d, d)                       # (..., N, g, l, k, t)
            C_att = torch.einsum("...nglkt,glmt->...nlkm", lifted, O_blocks)          # (..., N, H, d, d) blocks [ll]
            if self.attention_residual:
                self_weight = b.diagonal(dim1=-2, dim2=-1).transpose(-1, -2)          # (..., N, H) beta_ii^h
                cross = torch.einsum("hlks,...nhlsm->...nlkm", O_blocks,
                                     TC * self_weight[..., None, None, None])          # (..., N, H, d, d) X_{i,l}
                C_att = C_att + C + cross + cross.transpose(-1, -2)
            scale = torch.diagonal(C, dim1=-2, dim2=-1).flatten(-2).mean(dim=-1)      # (..., N) tr(C_i)/K
            C_att = C_att + (self.predictive_covariance_floor * scale)[..., None, None, None] * torch.eye(
                d, dtype=dtype, device=mu.device)
            result = ((forward @ C_att) * forward).sum(dim=-1)                         # (..., N, H, d) diag(U C U^T)
        return result.flatten(-2).to(sigma.dtype)

    def _predictive_covariance_update_dense(
        self,
        mu:    torch.Tensor,  # (..., N, K)
        sigma: torch.Tensor,  # (..., N, K) or (..., N, K, K)
        beta:  torch.Tensor,  # (..., H, N, N)
        frame: CanonicalFrameContext,
        receive: Optional[torch.Tensor] = None,   # (H, K, d)
    ) -> torch.Tensor:
        r"""Dense K x K evaluation of :meth:`predictive_covariance_update` (full family or dense frames)."""
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
            T_hat, O_hat = self._ambient_content_maps(dtype, receive)              # (H,d,K), (H,K,d)
            b = beta.to(dtype)                                                     # (..., H, N, N)
            eye_h = torch.eye(H, dtype=dtype, device=mu.device)

            # Shift every source mean by token 0's. Each beta row sums to 1, so the spread
            # sum_j beta_ij v_j v_j^T - m_i m_i^T is invariant under a common shift of the v_j, and
            # the shift is causal because token 0 lies in every row's support. It removes the
            # common offset that would otherwise cancel between the two terms.
            xi = xi - xi[..., :1, :]                                              # (..., N, K)
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

    def content_factors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Return full-state (T, O') for M_h=O'_h T_h; O' = O in the affine chart (the stored parameter)."""
        return self.T, chart_receive_map(self.T, self.O, self.content_link_chart)

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
