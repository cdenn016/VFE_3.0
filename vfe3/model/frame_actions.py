r"""Frame actions of the realized vertex factors :math:`U_i` on belief moments.

One numerics path for every consumer that moves a belief between local and reference coordinates:
the reference-coordinate block (``block_coordinates='reference'``), the reference-coordinate decoder
(``decode_coordinates='reference'``), and the frozen M5 frame-role interventions
(``vfe3.posthoc.interventions``). With :math:`\mu_i = U_i\xi_i` and :math:`\Sigma_i = U_iC_iU_i^\top`,

.. math::
    \xi_i = U_i^{-1}\mu_i,\qquad C_i = U_i^{-1}\Sigma_iU_i^{-\top}.

Every action runs with autocast disabled, at float32, or at float64 when the frame (or the operand)
is float64 (an escalated frame pair, audit 2026-09-23 #12), and returns the operand's dtype. The
frame is either dense, ``(..., N, K, K)``, or compact block-diagonal, ``(..., N, G, d, d)`` with
:math:`K = Gd` (``CanonicalFrameContext.block_diagonal``); a compact frame never materializes the
dense :math:`K \times K` factor.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from vfe3.runtime import ieee_fp32_matmul


def frames_identity_reason(
    cfg:        Any,                               # VFE3Config (or a namespace with its fields)
    n_gen:      int,                               # generator count of the gauge group (phi_embed width)

    *,
    positional: bool = True,
) -> Optional[str]:
    r"""Why every realized frame :math:`U_i = \exp(X_{x_i})\exp(Y_i)` of ``cfg`` is the identity, else ``None``.

    The one rule shared by the semantics run (``SemanticRun.frames_active``), the frozen interventions and the
    post-hoc routing-covariance gap (audit 2026-10-10 round 2, #46). The token frames are the identity under
    ``gauge_transport='off'`` (which coerces ``phi_scale=0``, ``m_phi_lr=0`` and ``pos_phi='none'`` and refuses a
    frame-space RoPE), for a group without generators, and for a token table that starts at zero and never moves
    (``phi_scale=0`` with ``m_phi_lr=0``). ``gauge_transport='frozen'`` is not the identity: its random fixed
    frames act on every forward pass. With ``positional`` a positional factor (``pos_phi`` over a group with
    generators, or a frame-space RoPE fold) keeps the composed frame active; without it only the token factor
    :math:`\exp(X_v)` is judged.
    """
    if getattr(cfg, "gauge_transport", "on") == "off":
        return ("gauge_transport='off': every realized frame is the identity (phi_scale=0, m_phi_lr=0 and "
                "pos_phi='none' are coerced)")
    if int(n_gen) == 0:
        reason = "the gauge group has no generators, so every token frame is the identity"
    elif float(getattr(cfg, "phi_scale", 1.0)) == 0.0 and float(getattr(cfg, "m_phi_lr", 1.0)) == 0.0:
        reason = "phi_scale=0 with m_phi_lr=0: the token frames start at the identity and never move"
    else:
        return None
    if positional and ((int(n_gen) > 0 and getattr(cfg, "pos_phi", "none") != "none")
                       or getattr(cfg, "rope_mode", "none") in ("frame", "frame_value")):
        return None
    return reason


def _work_dtype(*tensors: torch.Tensor) -> torch.dtype:
    r"""float64 when any operand is float64 (an escalated frame), else float32."""
    return torch.float64 if any(t.dtype == torch.float64 for t in tensors) else torch.float32


@ieee_fp32_matmul()
def frame_matvec(
    matrix:         torch.Tensor,                  # (..., N, G, d, d) compact blocks or (..., N, K, K) dense
    x:              torch.Tensor,                  # (..., N, K)
    block_diagonal: bool,
) -> torch.Tensor:                                 # (..., N, K) in x's dtype
    r"""Per-token :math:`M_i x_i` for a realized frame factor :math:`M_i \in \{U_i, U_i^{-1}\}`.

    Evaluated with autocast off at the frame's precision (float64 for an escalated frame) and returned
    in ``x``'s dtype, as the model's own frame actions (``RelationalOperators.canonical_mean``) are.
    """
    dtype = _work_dtype(matrix, x)
    with torch.autocast(device_type=x.device.type, enabled=False):
        work, factor = x.to(dtype), matrix.to(dtype)
        if block_diagonal:
            G, d = factor.shape[-3], factor.shape[-1]
            out = (factor @ work.reshape(*x.shape[:-1], G, d).unsqueeze(-1)).flatten(-3)   # (..., N, K)
        else:
            out = (factor @ work.unsqueeze(-1)).squeeze(-1)                               # (..., N, K)
    return out.to(x.dtype)


@ieee_fp32_matmul()
def reference_variances(
    mu:    torch.Tensor,                           # (..., N, K) local means (layout reference only)
    sigma: torch.Tensor,                           # (..., N, K) diagonal or (..., N, K, K) full local covariance
    frame: Any,                                    # CanonicalFrameContext of U_i
) -> torch.Tensor:                                 # (..., N, K)
    r"""Reference variances :math:`\operatorname{diag}C_i` with :math:`C_i = U_i^{-1}\Sigma_i U_i^{-\top}`.

    For diagonal :math:`\Sigma_i = \operatorname{diag}\sigma_i`,
    :math:`(C_i)_{kk} = \sum_m (U_i^{-1})_{km}^2\,\sigma_{im}`; for full :math:`\Sigma_i`,
    :math:`(C_i)_{kk} = \sum_m (U_i^{-1}\Sigma_i)_{km}(U_i^{-1})_{km}`, blockwise for a block-diagonal
    frame, where only the diagonal blocks of :math:`\Sigma_i` enter.
    """
    inverse = frame.inverse
    diagonal = sigma.dim() == mu.dim()
    dtype = _work_dtype(inverse, sigma)
    with torch.autocast(device_type=sigma.device.type, enabled=False):
        inv, s = inverse.to(dtype), sigma.to(dtype)
        if frame.block_diagonal:
            G, d = inv.shape[-3], inv.shape[-1]
            if diagonal:
                out = (inv.square() @ s.reshape(*s.shape[:-1], G, d).unsqueeze(-1)).flatten(-3)
            else:
                blocks = torch.stack([s[..., g*d:(g+1)*d, g*d:(g+1)*d] for g in range(G)], dim=-3)  # (..., N, G, d, d)
                out = ((inv @ blocks) * inv).sum(dim=-1).flatten(-2)
        elif diagonal:
            out = (inv.square() @ s.unsqueeze(-1)).squeeze(-1)
        else:
            out = ((inv @ s) * inv).sum(dim=-1)
    return out.to(sigma.dtype)


@ieee_fp32_matmul()
def reference_covariance(
    mu:    torch.Tensor,                           # (..., N, K) local means (layout reference only)
    sigma: torch.Tensor,                           # (..., N, K) diagonal or (..., N, K, K) full local covariance
    frame: Any,                                    # CanonicalFrameContext of U_i
) -> torch.Tensor:                                 # (..., N, K, K), symmetric
    r"""Full reference covariance :math:`C_i = U_i^{-1}\Sigma_i U_i^{-\top}`.

    For a compact block-diagonal frame :math:`U_i^{-1} = \bigoplus_g A_g` the blocks are
    :math:`(C_i)_{gh} = A_g\Sigma_{gh}A_h^\top`; a diagonal :math:`\Sigma_i` gives a block-diagonal
    :math:`C_i` with :math:`(C_i)_{gg} = A_g\operatorname{diag}(\sigma_g)A_g^\top`. The result is
    symmetrized, :math:`\tfrac12(C + C^\top)`, so a downstream Cholesky sees an exactly symmetric matrix.
    """
    inverse = frame.inverse
    diagonal = sigma.dim() == mu.dim()
    K = mu.shape[-1]
    batch = sigma.shape[:-1] if diagonal else sigma.shape[:-2]                          # (..., N)
    dtype = _work_dtype(inverse, sigma)
    with torch.autocast(device_type=sigma.device.type, enabled=False):
        inv, s = inverse.to(dtype), sigma.to(dtype)
        if frame.block_diagonal:
            G, d = inv.shape[-3], inv.shape[-1]
            if diagonal:
                s_b = s.reshape(*s.shape[:-1], G, d)                                   # (..., N, G, d)
                blocks = (inv * s_b.unsqueeze(-2)) @ inv.transpose(-1, -2)             # (..., N, G, d, d)
                eye_g = torch.eye(G, dtype=dtype, device=s.device)
                out = torch.einsum("...gij,gh->...gihj", blocks, eye_g)                # (..., N, G, d, G, d)
            else:
                S = s.reshape(*s.shape[:-2], G, d, G, d)                               # (..., N, G, d, G, d)
                out = torch.einsum("...gij,...gjhk,...hlk->...gihl", inv, S, inv)     # (..., N, G, d, G, d)
            out = out.reshape(*batch, K, K)                                            # (..., N, K, K)
        elif diagonal:
            out = (inv * s.unsqueeze(-2)) @ inv.transpose(-1, -2)                      # (..., N, K, K)
        else:
            out = inv @ s @ inv.transpose(-1, -2)                                      # (..., N, K, K)
        out = 0.5 * (out + out.transpose(-1, -2))
    return out.to(sigma.dtype)


@ieee_fp32_matmul()
def reference_covariance_compact(
    mu:    torch.Tensor,                           # (..., N, K) local means (layout reference only)
    sigma: torch.Tensor,                           # (..., N, K) diagonal or (..., N, K, K) full local covariance
    frame: Any,                                    # CanonicalFrameContext of U_i
) -> torch.Tensor:                                 # (..., N, G, d, d) blocks, or (..., N, K, K)
    r"""Reference covariance :math:`C_i = U_i^{-1}\Sigma_i U_i^{-\top}` in its most compact exact layout.

    For a diagonal :math:`\Sigma_i` on a compact block-diagonal frame :math:`U_i^{-1} = \bigoplus_g A_g`,
    :math:`C_i` is block diagonal and is returned as its blocks
    :math:`(C_i)_{gg} = A_g\operatorname{diag}(\sigma_g)A_g^\top`, ``(..., N, G, d, d)``; otherwise as the dense
    :func:`reference_covariance`. Symmetrized, in ``sigma``'s dtype. This is the layout the prototype-agent
    block MLP reads under ``block_coordinates='reference'``, the construction of
    ``RelationalOperators._reference_covariance`` with the block shape taken from the frame.
    """
    if not (frame.block_diagonal and sigma.dim() == mu.dim()):
        return reference_covariance(mu, sigma, frame)
    dtype = _work_dtype(frame.inverse, sigma)
    with torch.autocast(device_type=sigma.device.type, enabled=False):
        inv, s = frame.inverse.to(dtype), sigma.to(dtype)
        G, d = inv.shape[-3], inv.shape[-1]
        s_b = s.reshape(*s.shape[:-1], G, d)                                           # (..., N, G, d)
        blocks = (inv * s_b.unsqueeze(-2)) @ inv.transpose(-1, -2)                     # (..., N, G, d, d)
        blocks = 0.5 * (blocks + blocks.transpose(-1, -2))
    return blocks.to(sigma.dtype)


@ieee_fp32_matmul()
def local_covariance(
    mu:       torch.Tensor,                        # (..., N, K) local means (layout reference only)
    C:        torch.Tensor,                        # (..., N, G, d, d) blocks or (..., N, K, K) reference covariance
    frame:    Any,                                 # CanonicalFrameContext of U_i
    *,
    diagonal: bool,                                # True: return diag Sigma_i, (..., N, K)
) -> torch.Tensor:                                 # (..., N, K) or (..., N, K, K)
    r"""Local covariance :math:`\Sigma_i = U_iC_iU_i^\top` of a reference covariance in either compact layout.

    Blocks ``(..., N, G, d, d)`` need a compact block-diagonal frame of the same block shape and give
    :math:`(\Sigma_i)_{gg} = U_g(C_i)_{gg}U_g^\top`; a dense :math:`C_i` gives the dense product (blockwise
    :math:`U_g(C_i)_{gh}U_h^\top` for a compact frame). With ``diagonal`` only
    :math:`\operatorname{diag}\Sigma_i` is formed, :math:`(\Sigma_i)_{kk} = \sum_{mn}U_{km}(C_i)_{mn}U_{kn}`,
    the moment-matched diagonal; it reads only the diagonal blocks of :math:`C_i` on a block frame. Inverse
    of :func:`reference_covariance_compact` for a diagonal :math:`\Sigma_i` and of :func:`reference_covariance`
    for a full one. Returned in ``C``'s dtype.
    """
    K = mu.shape[-1]
    blocks = C.dim() == mu.dim() + 2
    if blocks and not frame.block_diagonal:
        raise ValueError("a block reference covariance (..., N, G, d, d) needs a compact block-diagonal frame")
    dtype = _work_dtype(frame.forward, C)
    with torch.autocast(device_type=C.device.type, enabled=False):
        U, c = frame.forward.to(dtype), C.to(dtype)
        if blocks:
            if U.shape[-3:] != c.shape[-3:]:
                raise ValueError(f"reference covariance blocks {tuple(c.shape[-3:])} do not match the frame "
                                 f"blocks {tuple(U.shape[-3:])}")
            if diagonal:
                out = ((U @ c) * U).sum(dim=-1).flatten(-2)                            # (..., N, K)
            else:
                S = U @ c @ U.transpose(-1, -2)                                        # (..., N, G, d, d)
                eye_g = torch.eye(S.shape[-3], dtype=dtype, device=S.device)
                out = torch.einsum("...gij,gh->...gihj", S, eye_g).reshape(*S.shape[:-3], K, K)
        elif frame.block_diagonal:
            G, d = U.shape[-3], U.shape[-1]
            c_b = c.reshape(*c.shape[:-2], G, d, G, d)                                 # (..., N, G, d, G, d)
            if diagonal:
                diag_blocks = torch.diagonal(c_b, dim1=-4, dim2=-2).movedim(-1, -3)    # (..., N, G, d, d)
                out = ((U @ diag_blocks) * U).sum(dim=-1).flatten(-2)                  # (..., N, K)
            else:
                out = torch.einsum("...gij,...gjhk,...hlk->...gihl", U, c_b, U).reshape(*c.shape[:-2], K, K)
        elif diagonal:
            out = ((U @ c) * U).sum(dim=-1)                                            # (..., N, K)
        else:
            out = U @ c @ U.transpose(-1, -2)                                          # (..., N, K, K)
        if not diagonal:
            out = 0.5 * (out + out.transpose(-1, -2))
    return out.to(C.dtype)


def identity_frame(frame: Any) -> Any:             # CanonicalFrameContext -> CanonicalFrameContext
    r"""The identity frame in ``frame``'s layout, dtype and device (an expanded view, no copy).

    The frame-diagonal family (``gaussian_frame_diagonal``) stores its covariance in reference coordinates, so a
    consumer written for local moments and a frame reads its reference moments :math:`(\xi_i, c_i)` exactly through
    this frame: :math:`I^{-1}\xi_i = \xi_i` and :math:`I\operatorname{diag}(c_i)I^\top = \operatorname{diag}(c_i)`.
    """
    from vfe3.contracts import trusted_identity_context

    eye = torch.eye(frame.forward.shape[-1], dtype=frame.forward.dtype, device=frame.forward.device)
    eye = eye.expand(frame.forward.shape)                                              # (..., N, [G,] d|K, d|K)
    # An exact identity: the float64 value validation of CanonicalFrameContext cannot fail on it.
    return trusted_identity_context(eye, frame.block_diagonal)


def frame_diagonal_reference(
    mu:    torch.Tensor,                           # (..., N, K) local means
    frame: Any,                                    # CanonicalFrameContext of U_i
) -> 'tuple[torch.Tensor, Any]':                   # (..., N, K) xi = U^{-1} mu, and the identity frame
    r"""The reference means :math:`\xi_i = U_i^{-1}\mu_i` and the identity frame (:func:`identity_frame`).

    With the stored reference variances :math:`c_i` of ``gaussian_frame_diagonal`` the pair
    :math:`(\xi_i, \operatorname{diag}c_i)` and the identity frame are a belief in the layout every local-coordinate
    consumer reads, with the same reference moments; idempotent on an identity frame.
    """
    return frame_matvec(frame.inverse, mu, frame.block_diagonal), identity_frame(frame)


@ieee_fp32_matmul()
def frame_diagonal_local_covariance(
    c:        torch.Tensor,                        # (..., N, K) reference variances c_i
    frame:    Any,                                 # CanonicalFrameContext of U_i
    *,
    diagonal: bool,                                # True: return diag Sigma_i, (..., N, K)
) -> torch.Tensor:                                 # (..., N, K) or (..., N, K, K) in c's dtype
    r"""The local covariance :math:`\Sigma_i = U_i\operatorname{diag}(c_i)U_i^\top` of a frame-diagonal belief.

    With ``diagonal`` only :math:`(\Sigma_i)_{kk} = \sum_m(U_i)_{km}^2c_{i,m}` is formed (blockwise on a compact frame),
    else the dense, symmetrized product. Autocast off, float64 for an escalated frame, returned in ``c``'s dtype.
    """
    dtype = _work_dtype(frame.forward, c)
    K = c.shape[-1]
    with torch.autocast(device_type=c.device.type, enabled=False):
        U, s = frame.forward.to(dtype), c.to(dtype)
        if frame.block_diagonal:
            G, d = U.shape[-3], U.shape[-1]
            s_b = s.reshape(*s.shape[:-1], G, d)                                       # (..., N, G, d)
            if diagonal:
                out = (U.square() @ s_b.unsqueeze(-1)).flatten(-3)                     # (..., N, K)
            else:
                blocks = (U * s_b.unsqueeze(-2)) @ U.transpose(-1, -2)                 # (..., N, G, d, d)
                eye_g = torch.eye(G, dtype=dtype, device=c.device)
                out = torch.einsum("...gij,gh->...gihj", blocks, eye_g).reshape(*s.shape[:-1], K, K)
        elif diagonal:
            out = (U.square() @ s.unsqueeze(-1)).squeeze(-1)                           # (..., N, K)
        else:
            out = (U * s.unsqueeze(-2)) @ U.transpose(-1, -2)                          # (..., N, K, K)
        if not diagonal:
            out = 0.5 * (out + out.transpose(-1, -2))
    return out.to(c.dtype)
