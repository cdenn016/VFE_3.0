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

from typing import Any

import torch


def _work_dtype(*tensors: torch.Tensor) -> torch.dtype:
    r"""float64 when any operand is float64 (an escalated frame), else float32."""
    return torch.float64 if any(t.dtype == torch.float64 for t in tensors) else torch.float32


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
