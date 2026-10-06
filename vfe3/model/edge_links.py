r"""Edge-dependent content links (GL(K)_attention_v2 Section 3: ``thm:routed_transport``, ``lem:lie_lowrank``).

Every edge :math:`j \to i` of the projected predictive readout carries the group-valued reference link

.. math::
    G_{ij} = g_i\,F\,E_{ij},\qquad E_{ij} = I_K + A_i\,\Phi_{ij}\,B_j^\top,\qquad F = e^{L},

with per-token factors :math:`A_i, B_j \in \mathbb{R}^{K\times r}` and an :math:`r\times r` coefficient
:math:`\Phi_{ij}`. With :math:`R = T^{-1}`, :math:`R_h = R_{:,h}` (the :math:`K\times d` column block of head
:math:`h`) and the head projector :math:`\Pi_h = R_h T_h`, the head edge map is :math:`M_h^{ij} = G_{ij}\Pi_h`
and the message is :math:`\tilde m_i = \sum_j G_{ij}\Pi(\beta_{ij})\xi_j`, :math:`\Pi(\beta) = \sum_h\beta^h\Pi_h`.

* ``"lowrank"`` (curved): :math:`g_i = I`, :math:`A_i = a(\hat\xi_i)`, :math:`B_j = b(\hat\xi_j)` and
  :math:`\Phi_{ij} = \varphi_1(B_j^\top A_i)`, so :math:`E_{ij} = \exp(A_iB_j^\top)` by
  :math:`\exp(AB^\top) = I + A\varphi_1(B^\top A)B^\top`.
* ``"pure_gauge"`` (flat control): one shared basis :math:`A`, :math:`g_i = \exp(AB_i^\top)` and
  :math:`\Phi_{ij} = -\varphi_1(-B_j^\top A)`, so :math:`E_{ij} = g_j^{-1}` and
  :math:`G_{ij} = g_i F g_j^{-1}`: every triangle holonomy :math:`G_{ik}^{-1}G_{ij}G_{jk} = g_kFg_k^{-1}`.

The factor maps read the RMS-normalized reference mean :math:`\hat\xi_i = \xi_i/(\overline{\xi_i^2} + 10^{-12})^{1/2}`,
which the passive action :math:`\mu\to g\mu`, :math:`U\to gU` leaves fixed, so :math:`G_{ij}` is reference data.
With :math:`B = 0` both modes are the exponential chart exactly.

Pure functions with shape comments; :class:`vfe3.model.relational_operators.RelationalOperators` owns the
parameters (``edge_receiver_map`` / ``edge_site_basis`` and ``edge_sender_map``) and calls these.

Level 2 (``content_link_latent``, ``"lowrank"`` only; GL(K)_supplementary_v2 Section S6) replaces the
amortized receiver factor :math:`A_i^0 = a(\hat\xi_i)` by the MAP estimate of a latent :math:`A_i` with the
prior :math:`N(A_i^0, \sigma_L^2 I)`: :func:`latent_receiver_energy` is the per-receiver energy,
:func:`infer_latent_receivers` the damped fixed-point inference and :func:`loop_energy` the optional
Wilson-type loop prior against the previous layer's links.
"""

import math
from typing import Dict, NamedTuple, Optional, Tuple

import torch

EDGE_MODES: Tuple[str, ...] = ("off", "lowrank", "pure_gauge")
EDGE_PARAMETER_NAMES: Tuple[str, ...] = ("edge_receiver_map", "edge_site_basis", "edge_sender_map")
RMS_EPS: float = 1e-12
# Per-layer latent-inference statistics (means over receivers), written to the relational capture.
LATENT_DIAGNOSTIC_KEYS: Tuple[str, ...] = (
    "relational_latent_energy_initial", "relational_latent_energy_final",
    "relational_latent_loop_energy", "relational_latent_receiver_shift")


def phi1(X: torch.Tensor) -> torch.Tensor:          # (..., n, n) -> (..., n, n)
    r""":math:`\varphi_1(X)` through the shared augmented ``matrix_exp`` of ``relational_operators._phi1``."""
    from vfe3.model.relational_operators import _phi1
    return _phi1(X)


class EdgeContext(NamedTuple):
    r"""Per-layer edge-link state, computed once and shared by the mean, the covariance and the diagnostics.

    Every tensor is at the readout's compute dtype; ``T``, ``R``, ``receive`` and ``F`` come from one
    float64 chart evaluation. ``A`` is ``(..., N, K, r)`` for ``"lowrank"`` and the shared ``(K, r)`` basis
    for ``"pure_gauge"``. ``Phi[..., i, j]`` is :math:`\Phi_{ij}` for receiver i and sender j; on pairs with
    no routing weight in any head it is :math:`\varphi_1(0) = I` (``"lowrank"``) or 0 (``"pure_gauge"``),
    values that every consumer multiplies by :math:`\beta_{ij} = 0`.
    """

    mode:    str
    T:       torch.Tensor              # (K, K) stacked sending maps, rows of head h = T_h
    R:       torch.Tensor              # (K, K) T^{-1}, columns of head h = R_h
    receive: torch.Tensor              # (K, K) O' = F R, columns of head h = O'_h
    F:       torch.Tensor              # (K, K) e^{L} = O' T
    A:       torch.Tensor              # (..., N, K, r) receiver factors A_i, or (K, r) shared basis
    B:       torch.Tensor              # (..., N, K, r) sender factors B_j
    B_ref:   torch.Tensor              # (..., N, K, r) B~_j = T^{-T} B_j; row block h is (B~_j)_h
    p:       torch.Tensor              # (..., N, H, r) p_j^h = (B~_j)_h^T T_h xi_j (unshifted values)
    Phi:     torch.Tensor              # (..., N, N, r, r) Phi_ij
    g_phi:   Optional[torch.Tensor]    # (..., N, r, r) phi_1(B_i^T A) of g_i (pure_gauge), else None
    mask:    torch.Tensor              # (..., N, N) bool, routed in some head


def cast_context(ctx: EdgeContext, dtype: torch.dtype) -> EdgeContext:
    r"""The context with every floating tensor at ``dtype`` (a no-op at the context's own dtype)."""
    return EdgeContext(*(x.to(dtype) if torch.is_tensor(x) and x.is_floating_point() else x for x in ctx))


def receive_heads(ctx: EdgeContext, n_heads: int) -> torch.Tensor:     # (H, K, d)
    r"""The effective receiving maps :math:`O'_h` of the context as the ``(H, K, d)`` head stack."""
    K = ctx.receive.shape[-1]
    return ctx.receive.reshape(K, n_heads, K // n_heads).permute(1, 0, 2)


def _head_rows(X: torch.Tensor, n_heads: int) -> torch.Tensor:          # (..., K, r) -> (..., H, d, r)
    return X.reshape(*X.shape[:-2], n_heads, X.shape[-2] // n_heads, X.shape[-1])


def rms_normalize(xi: torch.Tensor) -> torch.Tensor:                     # (..., N, K) -> (..., N, K)
    r""":math:`\hat\xi_i = \xi_i/\sqrt{K^{-1}\sum_k\xi_{ik}^2 + 10^{-12}}`, invariant under the passive gauge action."""
    return xi / torch.sqrt(xi.square().mean(dim=-1, keepdim=True) + RMS_EPS)


def edge_factors(
    xi_hat:         torch.Tensor,      # (..., N, K) RMS-normalized reference means
    mode:           str,
    sender_map:     torch.Tensor,      # (K r, K) W_B
    receiver_param: torch.Tensor,      # (K r, K) W_A (lowrank) or (K, r) shared basis A (pure_gauge)
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Factors :math:`A_i` (``(..., N, K, r)`` or the shared ``(K, r)``) and :math:`B_j` (``(..., N, K, r)``).

    .. math::
        B_j = \operatorname{reshape}(W_B\hat\xi_j, (K, r)),\qquad
        A_i = \operatorname{reshape}(W_A\hat\xi_i, (K, r))\ \text{(lowrank)},\quad A_i = A\ \text{(pure gauge)}.
    """
    if mode not in ("lowrank", "pure_gauge"):
        raise ValueError(f"edge factors need content_link_edges 'lowrank' or 'pure_gauge', got {mode!r}")
    K = xi_hat.shape[-1]
    r = sender_map.shape[0] // K
    B = (xi_hat @ sender_map.transpose(-1, -2)).reshape(*xi_hat.shape[:-1], K, r)        # (..., N, K, r)
    if mode == "lowrank":
        A = (xi_hat @ receiver_param.transpose(-1, -2)).reshape(*xi_hat.shape[:-1], K, r)
    else:
        A = receiver_param                                                                 # (K, r)
    return A, B


def edge_coefficients(
    A:    torch.Tensor,                # (..., N, K, r) or (K, r)
    B:    torch.Tensor,                # (..., N, K, r)
    beta: torch.Tensor,                # (..., H, N, N) routing weights
    mode: str,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    r"""Per-edge :math:`\Phi_{ij}` ``(..., N, N, r, r)``, the outer :math:`\varphi_1(B_i^\top A)` (pure gauge) and the routed mask.

    ``"lowrank"``: :math:`\Phi_{ij} = \varphi_1(B_j^\top A_i)`, the argument zeroed on pairs whose routing
    weight vanishes in every head (the future under a causal prior) BEFORE :math:`\varphi_1`, so a large
    unused argument cannot overflow into :math:`\infty\cdot 0`. ``"pure_gauge"``:
    :math:`\Phi_{ij} = -\varphi_1(-B_j^\top A)` depends on the sender only (N matrix functions, no per-edge
    one), broadcast to the edges and zeroed on unrouted pairs; :math:`g_i` uses :math:`\varphi_1(B_i^\top A)`.
    """
    mask = (beta != 0).any(dim=-3)                                                       # (..., N, N)
    if mode == "lowrank":
        arg = torch.einsum("...jka,...ikb->...ijab", B, A)                              # (..., N, N, r, r) B_j^T A_i
        arg = torch.where(mask[..., None, None], arg, torch.zeros_like(arg))
        return phi1(arg), None, mask
    BA = torch.einsum("...jka,kb->...jab", B, A)                                         # (..., N, r, r) B_j^T A
    sender = -phi1(-BA)                                                                  # (..., N, r, r)
    Phi = torch.where(mask[..., None, None], sender.unsqueeze(-4), sender.new_zeros(()))   # (..., N, N, r, r)
    return Phi, phi1(BA), mask


def build_edge_context(
    xi:             torch.Tensor,      # (..., N, K) reference means U_j^{-1} mu_j (value frame)
    beta:           torch.Tensor,      # (..., H, N, N)
    T:              torch.Tensor,      # (K, K) stacked T
    R:              torch.Tensor,      # (K, K) T^{-1}
    receive:        torch.Tensor,      # (K, K) O'
    F:              torch.Tensor,      # (K, K) e^{L} = O' T
    mode:           str,
    sender_map:     torch.Tensor,      # (K r, K) W_B
    receiver_param: torch.Tensor,      # (K r, K) W_A or (K, r) A
) -> EdgeContext:
    r"""Factors, :math:`\tilde B_j = T^{-\top}B_j`, :math:`p_j^h = (\tilde B_j)_h^\top T_h\xi_j` and :math:`\Phi_{ij}`.

    :math:`p_j^h` is formed from the UNshifted values :math:`v_j^h = T_h\xi_j`: the covariance's common
    shift by token 0 leaves the spread invariant, but not the edge coefficients :math:`\Phi_{ij}p_j^h`.
    """
    H, K = beta.shape[-3], xi.shape[-1]
    A, B = edge_factors(rms_normalize(xi), mode, sender_map, receiver_param)
    B_ref = R.transpose(-1, -2) @ B                                                      # (..., N, K, r)
    values = torch.einsum("hsk,...nk->...nhs", T.reshape(H, K // H, K), xi)              # (..., N, H, d) v_j^h
    p = torch.einsum("...nhsa,...nhs->...nha", _head_rows(B_ref, H), values)             # (..., N, H, r)
    Phi, g_phi, mask = edge_coefficients(A, B, beta, mode)
    return EdgeContext(mode, T, R, receive, F, A, B, B_ref, p, Phi, g_phi, mask)


def outer_gauge_vector(ctx: EdgeContext, x: torch.Tensor) -> torch.Tensor:   # (..., N, K) -> (..., N, K)
    r""":math:`g_i x_i = x_i + A\varphi_1(B_i^\top A)B_i^\top x_i` (pure gauge); the identity otherwise."""
    if ctx.mode != "pure_gauge":
        return x
    Bx = torch.einsum("...nka,...nk->...na", ctx.B, x)                                   # (..., N, r)
    return x + (ctx.A @ (ctx.g_phi @ Bx.unsqueeze(-1))).squeeze(-1)


def outer_gauge_left(ctx: EdgeContext, X: torch.Tensor) -> torch.Tensor:     # (..., N, K, M) -> same
    r""":math:`g_iX_i` for a matrix per token (pure gauge); the identity otherwise."""
    if ctx.mode != "pure_gauge":
        return X
    return X + ctx.A @ (ctx.g_phi @ (ctx.B.transpose(-1, -2) @ X))


def outer_gauge_sandwich(ctx: EdgeContext, X: torch.Tensor) -> torch.Tensor:  # (..., N, K, K) -> same
    r""":math:`g_iX_ig_i^\top` (pure gauge); the identity otherwise."""
    if ctx.mode != "pure_gauge":
        return X
    return outer_gauge_left(ctx, outer_gauge_left(ctx, X).transpose(-1, -2)).transpose(-1, -2)


def edge_reference_message(
    ctx:  EdgeContext,
    beta: torch.Tensor,                # (..., H, N, N)
    z:    torch.Tensor,                # (..., N, K) stacked pooled values z_i^h = sum_j beta_ij^h v_j^h
) -> torch.Tensor:                     # (..., N, K) reference message m~_i
    r"""The reference message :math:`\tilde m_i = \sum_jG_{ij}\Pi(\beta_{ij})\xi_j`.

    .. math::
        \tilde m_i = g_i\,O'\bigl(z_i + TA_i\,y_i\bigr),\qquad y_i = \sum_j\Phi_{ij}c_{ij},\qquad
        c_{ij} = \sum_h\beta_{ij}^h p_j^h,

    since :math:`\Pi(\beta_{ij})\xi_j = Ry_{ij}` with :math:`y_{ij} = (\beta_{ij}^hv_j^h)_h`, :math:`FR = O'`,
    :math:`FA_i = O'TA_i` and :math:`B_j^\top Ry_{ij} = c_{ij}`.
    """
    c = torch.einsum("...hij,...jhb->...ijb", beta, ctx.p)                               # (..., N, N, r) c_ij
    y = torch.einsum("...ijab,...ijb->...ia", ctx.Phi, c)                                # (..., N, r) y_i
    TA = ctx.T @ ctx.A                                                                   # (..., N, K, r) or (K, r)
    inner = z + (TA @ y.unsqueeze(-1)).squeeze(-1)                                       # (..., N, K) z_i + T A_i y_i
    return outer_gauge_vector(ctx, inner @ ctx.receive.transpose(-1, -2))


def edge_covariance_factors(
    ctx:    EdgeContext,
    beta:   torch.Tensor,              # (..., H, N, N)
    c:      torch.Tensor,              # (..., H, H, N, N) c_ij^{hh'}: beta^h on the diagonal, beta^h beta^h' off it
    source: torch.Tensor,              # (..., N, H, H, d, d) s_j^{hh'} = T_h C_j T_h'^T
    values: torch.Tensor,              # (..., N, H, d) values v_j^h, shifted by token 0
    mean:   torch.Tensor,              # (..., N, H, d) pooled values z_i^h of the shifted values
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""``(A_hat, X_hat, Y)`` of the edge correction of the lifted covariance (plan Section 2.3).

    .. math::
        F\bigl(A_iX_i + X_i^\top A_i^\top + A_iY_iA_i^\top\bigr)F^\top
          = \hat A_i\hat X_i^\top + \hat X_i\hat A_i^\top + \hat A_iY_i\hat A_i^\top,
        \qquad \hat A_i = FA_i,\quad \hat X_i = FX_i^\top = \sum_g O'_g\,\Xi_i^{g\top},

    with :math:`X_i = \sum_g\Xi_i^gR_g^\top` and

    .. math::
        \Xi_i^g = \sum_j\Phi_{ij}\sum_hc_{ij}^{hg}(\tilde B_j)_h^\top s_j^{hg}
          + \sum_j\beta_{ij}^g\Phi_{ij}p_j^g v_j^{g\top} - q_i^g z_i^{g\top},\qquad
        q_i^g = \sum_j\beta_{ij}^g\Phi_{ij}p_j^g,

    .. math::
        Y_i = \sum_j\Phi_{ij}\Bigl[\sum_{h,g}c_{ij}^{hg}(\tilde B_j)_h^\top s_j^{hg}(\tilde B_j)_g\Bigr]\Phi_{ij}^\top
          + \sum_h\Bigl[\sum_j\beta_{ij}^h(\Phi_{ij}p_j^h)(\Phi_{ij}p_j^h)^\top - q_i^hq_i^{h\top}\Bigr].

    The spread terms are covariances over the sender label, so the shifted values may enter the
    :math:`v` slot; :math:`p_j^h` stays unshifted (:func:`build_edge_context`).
    """
    H = beta.shape[-3]
    Bt = _head_rows(ctx.B_ref, H)                                                        # (..., N, H, d, r) (B~_j)_h
    O_heads = receive_heads(ctx, H)                                                      # (H, K, d) O'_g
    P = torch.einsum("...nhsa,...nhgst->...nhgat", Bt, source)                           # (..., N, H, H, r, d) (B~_j)_h^T s_j^{hg}
    Xi = None
    for h in range(H):
        # sum_j c_ij^{hg} Phi_ij P_j^{hg} for one source head: a (..., G, N, N, r, r) weight at a time.
        weighted = c[..., h, :, :, :].unsqueeze(-1).unsqueeze(-1) * ctx.Phi.unsqueeze(-5)
        term = torch.einsum("...gijab,...jgbt->...igat", weighted, P[..., h, :, :, :])  # (..., N, H, r, d)
        Xi = term if Xi is None else Xi + term
    Q = torch.einsum("...nhgat,...ngtb->...nhgab", P, Bt)                                # (..., N, H, H, r, r)
    M = torch.einsum("...hgij,...jhgab->...ijab", c, Q)                                  # (..., N, N, r, r)
    Y = torch.einsum("...ijab,...ijcb->...iac", ctx.Phi @ M, ctx.Phi)                    # (..., N, r, r)
    u = torch.einsum("...ijab,...jhb->...ijha", ctx.Phi, ctx.p)                          # (..., N, N, H, r) Phi_ij p_j^h
    wu = beta.movedim(-3, -1).unsqueeze(-1) * u                                          # (..., N, N, H, r) beta_ij^h Phi_ij p_j^h
    q = wu.sum(dim=-3)                                                                   # (..., N, H, r) q_i^h
    Xi = Xi + torch.einsum("...ijha,...jhs->...ihas", wu, values) - q.unsqueeze(-1) * mean.unsqueeze(-2)
    Y = (Y + torch.einsum("...ijha,...ijhb->...iab", wu, u)
         - torch.einsum("...iha,...ihb->...iab", q, q))
    X_hat = torch.einsum("gks,...ngas->...nka", O_heads, Xi)                             # (..., N, K, r)
    A_hat = ctx.F @ ctx.A                                                                # (..., N, K, r) or (K, r)
    return A_hat, X_hat, Y


def edge_correction_dense(
    A_hat: torch.Tensor,               # (..., N, K, r) or (K, r)
    X_hat: torch.Tensor,               # (..., N, K, r)
    Y:     torch.Tensor,               # (..., N, r, r)
) -> torch.Tensor:                     # (..., N, K, K)
    r""":math:`\hat A\hat X^\top + \hat X\hat A^\top + \hat AY\hat A^\top` as a dense K x K term per token."""
    AX = A_hat @ X_hat.transpose(-1, -2)
    return AX + AX.transpose(-1, -2) + A_hat @ Y @ A_hat.transpose(-1, -2)


def edge_correction_blocks(
    A_hat:   torch.Tensor,             # (..., N, K, r) or (K, r)
    X_hat:   torch.Tensor,             # (..., N, K, r)
    Y:       torch.Tensor,             # (..., N, r, r)
    n_heads: int,
) -> torch.Tensor:                     # (..., N, H, d, d) diagonal blocks [.]_ll
    r"""Diagonal blocks :math:`\hat A^{(l)}\hat X^{(l)\top} + \hat X^{(l)}\hat A^{(l)\top} + \hat A^{(l)}Y\hat A^{(l)\top}`."""
    A_l, X_l = _head_rows(A_hat, n_heads), _head_rows(X_hat, n_heads)                    # (..., [N,] H, d, r)
    AX = A_l @ X_l.transpose(-1, -2)
    return AX + AX.transpose(-1, -2) + A_l @ Y.unsqueeze(-3) @ A_l.transpose(-1, -2)


def edge_self_coefficients(ctx: EdgeContext) -> torch.Tensor:          # (..., N, r, r)
    r"""The self-edge coefficients :math:`\Phi_{ii}`."""
    return torch.diagonal(ctx.Phi, dim1=-4, dim2=-3).movedim(-1, -3)


def edge_residual_cross_dense(
    ctx:         EdgeContext,
    A_hat:       torch.Tensor,         # (..., N, K, r) or (K, r)
    self_weight: torch.Tensor,         # (..., N, H) beta_ii^h
    TC:          torch.Tensor,         # (..., N, H, d, K) T_h C_i
) -> torch.Tensor:                     # (..., N, K, K)
    r"""Edge part of the residual cross term, :math:`\hat A_i\Phi_{ii}\kappa_i` with
    :math:`\kappa_i = \sum_h\beta_{ii}^h(\tilde B_i)_h^\top T_hC_i`, so that
    :math:`\sum_h\beta_{ii}^hM_h^{ii}C_i = g_i\bigl[\sum_h\beta_{ii}^hO'_hT_hC_i + \hat A_i\Phi_{ii}\kappa_i\bigr]`.
    """
    kappa = torch.einsum("...nh,...nhsa,...nhsm->...nam", self_weight,
                         _head_rows(ctx.B_ref, self_weight.shape[-1]), TC)                # (..., N, r, K)
    return A_hat @ (edge_self_coefficients(ctx) @ kappa)


def edge_residual_cross_blocks(
    ctx:         EdgeContext,
    A_hat:       torch.Tensor,         # (..., N, K, r) or (K, r)
    self_weight: torch.Tensor,         # (..., N, H) beta_ii^h
    TC:          torch.Tensor,         # (..., N, H, H_m, d, d) T_h^{(m)} C_{i,m}
    cross:       torch.Tensor,         # (..., N, H, d, d) blocks of sum_h beta_ii^h O'_h T_h C_i
) -> torch.Tensor:                     # (..., N, H, d, d) blocks of sum_h beta_ii^h M_h^{ii} C_i
    r"""Diagonal blocks of the edge residual cross term for block-diagonal :math:`C_i`.

    :math:`\kappa_i^{(l)} = \sum_h\beta_{ii}^h(\tilde B_i)_h^\top T_h^{(l)}C_{i,l}` gives the edge part
    :math:`\hat A_i^{(l)}\Phi_{ii}\kappa_i^{(l)}`; under ``"pure_gauge"`` the outer factor adds
    :math:`A^{(l)}\varphi_i(B_i^\top\mathrm{Cr}_i)_{:,l}` with
    :math:`B_i^\top\mathrm{Cr}_i = \sum_h\beta_{ii}^h(B_i^\top O'_h)T_hC_i + (B_i^\top\hat A_i)\Phi_{ii}\kappa_i`.
    """
    H = self_weight.shape[-1]
    Bt = _head_rows(ctx.B_ref, H)                                                        # (..., N, H, d, r)
    kappa = torch.einsum("...nh,...nhsa,...nhlst->...nlat", self_weight, Bt, TC)        # (..., N, H, r, d)
    Phi_ii = edge_self_coefficients(ctx).unsqueeze(-3)                                   # (..., N, 1, r, r)
    A_l = _head_rows(A_hat, H)                                                           # (..., [N,] H, d, r)
    out = cross + A_l @ (Phi_ii @ kappa)
    if ctx.mode != "pure_gauge":
        return out
    BO = torch.einsum("...nka,hks->...nhas", ctx.B, receive_heads(ctx, H))               # (..., N, H, r, d) B_i^T O'_h
    BA = ctx.B.transpose(-1, -2) @ A_hat                                                 # (..., N, r, r) B_i^T A_hat_i
    rows = (torch.einsum("...nh,...nhas,...nhlst->...nlat", self_weight, BO, TC)
            + BA.unsqueeze(-3) @ (Phi_ii @ kappa))                                      # (..., N, H, r, d) (B^T Cr)_{:,l}
    return out + _head_rows(ctx.A, H) @ (ctx.g_phi.unsqueeze(-3) @ rows)


def outer_gauge_blocks(
    ctx:    EdgeContext,
    inner:  torch.Tensor,              # (..., N, H, d, d) diagonal blocks of a symmetric Inner_i
    rows:   torch.Tensor,              # (..., N, r, K) B_i^T Inner_i
) -> torch.Tensor:                     # (..., N, H, d, d) diagonal blocks of g_i Inner_i g_i^T
    r"""Diagonal blocks of :math:`g_i\,\mathrm{Inner}_i\,g_i^\top` for :math:`g_i = I + A\varphi_iB_i^\top`.

    .. math::
        (g\,\mathrm{In}\,g^\top)_{ll} = \mathrm{In}_{ll} + A^{(l)}\varphi_i(B_i^\top\mathrm{In})_{:,l}
          + \bigl[A^{(l)}\varphi_i(B_i^\top\mathrm{In})_{:,l}\bigr]^\top
          + A^{(l)}\varphi_i(B_i^\top\mathrm{In}B_i)\varphi_i^\top A^{(l)\top},

    from the r x K rows :math:`B_i^\top\mathrm{In}` without a dense K x K Inner.
    """
    if ctx.mode != "pure_gauge":
        return inner
    H = inner.shape[-3]
    phi = ctx.g_phi.unsqueeze(-3)                                                        # (..., N, 1, r, r)
    A_l = _head_rows(ctx.A, H)                                                           # (H, d, r)
    rows_l = rows.reshape(*rows.shape[:-1], H, rows.shape[-1] // H).movedim(-2, -3)     # (..., N, H, r, d)
    BIB = (rows @ ctx.B).unsqueeze(-3)                                                   # (..., N, 1, r, r)
    first = A_l @ (phi @ rows_l)
    return inner + first + first.transpose(-1, -2) + A_l @ phi @ BIB @ phi.transpose(-1, -2) @ A_l.transpose(-1, -2)


def edge_inner_rows(
    ctx:    EdgeContext,
    pooled: torch.Tensor,              # (..., N, H, H, d, d) head-pair pooled covariance P_i^{hg} (with spread)
    A_hat:  torch.Tensor,              # (K, r) for pure_gauge
    X_hat:  torch.Tensor,              # (..., N, K, r)
    Y:      torch.Tensor,              # (..., N, r, r)
) -> torch.Tensor:                     # (..., N, r, K)
    r"""Rows :math:`B_i^\top\mathrm{Inner}_i` of :math:`\mathrm{Inner} = \sum_{h,g}O'_hP^{hg}O'^\top_g + \hat A\hat X^\top + \hat X\hat A^\top + \hat AY\hat A^\top`."""
    H = pooled.shape[-3]
    O_heads = receive_heads(ctx, H)                                                      # (H, K, d)
    BO = torch.einsum("...nka,hks->...nhas", ctx.B, O_heads)                             # (..., N, H, r, d)
    lifted = torch.einsum("...ngat,gmt->...nam",
                          torch.einsum("...nhas,...nhgst->...ngat", BO, pooled), O_heads)   # (..., N, r, K)
    Bt = ctx.B.transpose(-1, -2)                                                         # (..., N, r, K)
    BA, BX = Bt @ A_hat, Bt @ X_hat                                                      # (..., N, r, r)
    return (lifted + BA @ X_hat.transpose(-1, -2) + BX @ A_hat.transpose(-1, -2)
            + BA @ Y @ A_hat.transpose(-1, -2))


def edge_link_matrices(ctx: EdgeContext) -> torch.Tensor:             # (..., N, N, K, K)
    r"""Every reference link :math:`G_{ij} = g_iFE_{ij}` explicitly (all pairs, unmasked; small N only)."""
    K = ctx.F.shape[-1]
    eye = torch.eye(K, dtype=ctx.F.dtype, device=ctx.F.device)
    if ctx.mode == "lowrank":
        Phi = phi1(torch.einsum("...jka,...ikb->...ijab", ctx.B, ctx.A))                # (..., N, N, r, r)
        E = eye + ctx.A.unsqueeze(-3) @ Phi @ ctx.B.unsqueeze(-4).transpose(-1, -2)
        return ctx.F @ E
    sender = -phi1(-torch.einsum("...jka,kb->...jab", ctx.B, ctx.A))                    # (..., N, r, r)
    E = eye + ctx.A @ sender @ ctx.B.transpose(-1, -2)                                   # (..., N, K, K) = g_j^{-1}
    g = eye + ctx.A @ ctx.g_phi @ ctx.B.transpose(-1, -2)                                # (..., N, K, K) g_i
    return g.unsqueeze(-3) @ ctx.F @ E.unsqueeze(-4)


def edge_part_norms(ctx: EdgeContext) -> torch.Tensor:                 # (..., N, N)
    r""":math:`\|A_i\Phi_{ij}B_j^\top\|_F = \operatorname{tr}(A_i^\top A_i\Phi_{ij}B_j^\top B_j\Phi_{ij}^\top)^{1/2}` per edge (r x r Grams)."""
    GA = ctx.A.transpose(-1, -2) @ ctx.A                                                 # (..., N, r, r) or (r, r)
    GB = ctx.B.transpose(-1, -2) @ ctx.B                                                 # (..., N, r, r)
    left = (GA.unsqueeze(-3) if GA.dim() > 2 else GA) @ ctx.Phi                          # (..., N, N, r, r) GA_i Phi_ij
    right = ctx.Phi @ GB.unsqueeze(-4)                                                   # Phi_ij GB_j
    return (left * right).sum(dim=(-1, -2)).clamp_min(0.0).sqrt()


@torch.no_grad()
def edge_diagnostic_capture(
    ctx:  EdgeContext,
    beta: torch.Tensor,                # (..., H, N, N)
) -> Dict[str, torch.Tensor]:
    r"""Cheap per-layer edge statistics and the detached factors the self-loop diagnostic reads.

    ``relational_edge_part_norm_mean`` is the head-mean routing-weighted mean over receivers of
    :math:`\sum_j\bar\beta_{ij}\|A_i\Phi_{ij}B_j^\top\|_F` (the edge part :math:`E_{ij} - I`),
    ``relational_edge_part_norm_max`` its maximum over routed edges, and the ``sender`` / ``receiver``
    entries the mean and maximum of :math:`\|B_j\|_F` / :math:`\|A_i\|_F` over tokens.
    """
    norms = edge_part_norms(ctx)                                                         # (..., N, N)
    beta_bar = beta.mean(dim=-3).to(norms.dtype)                                         # (..., N, N)
    routed = ctx.mask
    weighted = (beta_bar * norms).sum(dim=-1)                                            # (..., N)
    sender = ctx.B.flatten(-2).norm(dim=-1)                                              # (..., N)
    receiver = ctx.A.flatten(-2).norm(dim=-1) if ctx.mode == "lowrank" else ctx.A.norm().reshape(1)
    out: Dict[str, torch.Tensor] = {
        "relational_edge_part_norm_mean": weighted.mean(),
        "relational_edge_part_norm_max": (norms[routed].max() if bool(routed.any())
                                          else norms.new_zeros(())),
        "relational_edge_sender_norm_mean": sender.mean(),
        "relational_edge_sender_norm_max": sender.max(),
        "relational_edge_receiver_norm_mean": receiver.mean(),
        "edge_receiver_factors": ctx.A.detach().clone(),   # a copy: pure_gauge's A may alias the parameter
        "edge_sender_factors": ctx.B.detach(),
        "edge_self_coefficients": edge_self_coefficients(ctx).detach(),
    }
    if ctx.g_phi is not None:                                                            # pure_gauge outer g_i
        out["edge_outer_coefficients"] = ctx.g_phi.detach()
    return out


def edge_self_transfer(
    T:       torch.Tensor,             # (K, K) stacked T
    R:       torch.Tensor,             # (K, K) T^{-1}
    F:       torch.Tensor,             # (K, K) e^{L}
    A:       torch.Tensor,             # (..., N, K, r) or (K, r)
    B:       torch.Tensor,             # (..., N, K, r)
    Phi_ii:  torch.Tensor,             # (..., N, r, r)
    g_phi:   Optional[torch.Tensor],   # (..., N, r, r) (pure_gauge) or None
) -> torch.Tensor:                     # (..., N, K, K)
    r""":math:`W_i = TG_{ii}R`, whose product with :math:`D_i = \bigoplus_h\beta_{ii}^hI_d` is similar to the
    self-loop map :math:`\sum_h\beta_{ii}^hM_h^{ii} = G_{ii}RD_iT`."""
    K = F.shape[-1]
    eye = torch.eye(K, dtype=F.dtype, device=F.device)
    G = F @ (eye + A @ Phi_ii @ B.transpose(-1, -2))                                     # (..., N, K, K) F E_ii
    if g_phi is not None:
        G = G + A @ (g_phi @ (B.transpose(-1, -2) @ G))                                  # g_i F E_ii
    return T @ G @ R


# ==============================================================================================
# Level 2: latent receiver factors (plan Section 3; GL(K)_supplementary_v2 Section S6)
# ==============================================================================================

class LatentSources(NamedTuple):
    r"""The receiver-factor-independent pieces of the per-receiver link energy, computed once per layer.

    ``c``, ``source``, ``values`` and ``mean`` are the head-pair pieces of the moment-matched covariance
    (Section 2.3): :math:`c_{ij}^{hg}`, :math:`s_j^{hg} = T_hC_jT_g^\top`, the token-0-shifted values and their
    pooled means. ``const`` is the part of :math:`\operatorname{tr}C_i^{\mathrm{att}}` (no floor) that no
    receiver factor reaches, ``scale`` the receiver's mean reference variance :math:`c_i = \operatorname{tr}(C_i)/K`
    and ``kappa`` the residual cross-term rows :math:`\kappa_i` (None without the residual).
    """

    c:      torch.Tensor              # (..., H, H, N, N) c_ij^{hg}
    source: torch.Tensor              # (..., N, H, H, d, d) s_j^{hg} = T_h C_j T_g^T
    values: torch.Tensor              # (..., N, H, d) values of the token-0-shifted reference means
    mean:   torch.Tensor              # (..., N, H, d) pooled shifted values z_i^h
    kappa:  Optional[torch.Tensor]    # (..., N, r, K) sum_h beta_ii^h (B~_i)_h^T T_h C_i, or None
    scale:  torch.Tensor              # (..., N) c_i = tr(C_i) / K
    const:  torch.Tensor              # (..., N) receiver-factor-independent part of tr C_i^att


class LinkState(NamedTuple):
    r"""One layer's inferred link field :math:`G_{ab}^l = e^{L_l}E_{ab}^l`, threaded to the next layer's loop prior.

    ``F`` and ``F_inv`` are :math:`e^{\pm L_l}` in float64; ``A`` are the inferred receiver factors and ``B`` the
    sender factors of layer l.
    """

    F:     torch.Tensor               # (K, K) e^{L_l}, float64
    F_inv: torch.Tensor               # (K, K) e^{-L_l}, float64
    A:     torch.Tensor               # (..., N, K, r) inferred A_a^l
    B:     torch.Tensor               # (..., N, K, r) B_b^l


class LoopInputs(NamedTuple):
    r"""What a receiver's loop prior reads: the previous layer's links, the prior weights and the window."""

    previous: LinkState
    weights:  torch.Tensor            # (N, N) head-averaged routing prior pibar_ij
    window:   Optional[int]


def with_receiver_factors(
    ctx:  EdgeContext,                 # "lowrank" context
    A:    torch.Tensor,                # (..., N, K, r) receiver factors
    beta: torch.Tensor,                # (..., H, N, N)
) -> EdgeContext:
    r"""The context with receiver factors ``A`` and :math:`\Phi_{ij} = \varphi_1(B_j^\top A_i)` recomputed (masked)."""
    Phi, _, _ = edge_coefficients(A, ctx.B, beta, "lowrank")
    return ctx._replace(A=A, Phi=Phi)


def latent_sources(
    ctx:      EdgeContext,
    beta:     torch.Tensor,            # (..., H, N, N)
    xi:       torch.Tensor,            # (..., N, K) unshifted reference means (value frame)
    C:        torch.Tensor,            # (..., N, K, K) reference covariances, or (..., N, H, d, d) diagonal blocks
    residual: bool,                    # attention_residual (r = 1) or not (r = 0)
) -> LatentSources:
    r"""Receiver-factor-independent pieces of :func:`latent_transport_energy`.

    With :math:`\mathcal G_{hg} = O_h'^\top O'_g` and the pooled head-pair covariance
    :math:`P_i^{hg} = \sum_jc_{ij}^{hg}s_j^{hg} + \delta_{hg}\tilde\Sigma_i^h`, the part of the trace that no
    receiver factor reaches is

    .. math::
        \mathrm{const}_i = \sum_{h,g}\langle P_i^{hg}, \mathcal G_{hg}\rangle
          + r\Bigl[\operatorname{tr}C_i + 2\sum_h\beta_{ii}^h\operatorname{tr}(O'_hT_hC_i)\Bigr],

    computed from :math:`\tau_j^{hg} = \langle s_j^{hg}, \mathcal G_{hg}\rangle` and
    :math:`\langle\tilde\Sigma_i^h, \mathcal G_{hh}\rangle = \sum_j\beta_{ij}^hv_j^{h\top}\mathcal G_{hh}v_j^h
    - z_i^{h\top}\mathcal G_{hh}z_i^h`, so no pooled :math:`d\times d` head-pair covariance is formed. Block
    inputs ``C`` (diagonal beliefs on block frames) give the same values as their dense embedding.
    """
    H, K = beta.shape[-3], xi.shape[-1]
    d = K // H
    T_heads = ctx.T.reshape(H, d, K)                                                     # (H, d, K) T_h
    if C.dim() == xi.dim() + 2:                                                          # block-diagonal C_i
        TC = torch.einsum("hsmk,...nmkt->...nhmst", T_heads.reshape(H, d, H, d), C)     # (..., N, H, H_m, d, d)
        TC = TC.transpose(-3, -2).reshape(*C.shape[:-3], H, d, K)                       # (..., N, H, d, K) T_h C_i
        trace_C = torch.diagonal(C, dim1=-2, dim2=-1).sum(dim=(-2, -1))                 # (..., N)
    else:
        TC = torch.einsum("hsk,...nkm->...nhsm", T_heads, C)                             # (..., N, H, d, K) T_h C_i
        trace_C = torch.diagonal(C, dim1=-2, dim2=-1).sum(dim=-1)
    source = torch.einsum("...nhsm,gtm->...nhgst", TC, T_heads)                          # (..., N, H, H, d, d)
    eye_h = torch.eye(H, dtype=beta.dtype, device=beta.device)
    c = torch.einsum("...hij,...gij->...hgij", beta, beta)
    c = c * (1.0 - eye_h)[..., None, None] + torch.einsum("...hij,hg->...hgij", beta, eye_h)
    values = torch.einsum("hsk,...nk->...nhs", T_heads, xi - xi[..., :1, :])            # (..., N, H, d)
    mean = torch.einsum("...hij,...jhs->...ihs", beta, values)                           # (..., N, H, d)
    O_heads = receive_heads(ctx, H)                                                      # (H, K, d) O'_h
    gram = torch.einsum("hks,gkt->hgst", O_heads, O_heads)                               # (H, H, d, d) O'_h^T O'_g
    heads = torch.arange(H, device=gram.device)
    gram_diag = gram[heads, heads]                                                       # (H, d, d) O'_h^T O'_h
    tau = torch.einsum("...jhgst,hgst->...jhg", source, gram)                            # (..., N, H, H)
    const = (torch.einsum("...hgij,...jhg->...i", c, tau)
             + torch.einsum("...hij,...jh->...i", beta,
                            torch.einsum("...jhs,hst,...jht->...jh", values, gram_diag, values))
             - torch.einsum("...ihs,hst,...iht->...i", mean, gram_diag, mean))           # (..., N)
    kappa = None
    if residual:
        self_weight = beta.diagonal(dim1=-2, dim2=-1).transpose(-1, -2)                 # (..., N, H) beta_ii^h
        const = const + trace_C + 2.0 * torch.einsum("...nh,hks,...nhsk->...n", self_weight, O_heads, TC)
        kappa = torch.einsum("...nh,...nhsa,...nhsm->...nam",
                             self_weight, _head_rows(ctx.B_ref, H), TC)                  # (..., N, r, K)
    return LatentSources(c, source, values, mean, kappa, trace_C / K, const)


def latent_transport_energy(
    A:       torch.Tensor,             # (..., N, K, r) receiver factors
    ctx:     EdgeContext,              # "lowrank" context (its A and Phi are replaced)
    beta:    torch.Tensor,             # (..., H, N, N)
    sources: LatentSources,
) -> torch.Tensor:                     # (..., N) E_i(A_i)
    r"""Expected transport energy :math:`E_i(A_i) = \operatorname{tr}C_i^{\mathrm{att}}(A_i)\big|_{\text{no floor}}/(2c_i)`.

    Profiling the receiver's next-layer law :math:`N(r\xi_i + \tilde m_i, c_iI)` out of the S6 MAP free energy
    leaves half the :math:`Q^{-1}`-weighted trace of the moment-matched predictive covariance (plan Section 3.1).
    In reference coordinates (``"lowrank"``, :math:`g_i = I`), with :math:`\hat A_i = FA_i` and
    :math:`\hat X_i, Y_i` of :func:`edge_covariance_factors` at :math:`\Phi_{ij} = \varphi_1(B_j^\top A_i)`,

    .. math::
        \operatorname{tr}C_i^{\mathrm{att}} = \mathrm{const}_i + 2\langle\hat A_i, \hat X_i\rangle
          + \langle Y_i, \hat A_i^\top\hat A_i\rangle + 2r\operatorname{tr}(\Phi_{ii}\kappa_i\hat A_i).

    :math:`E_i` reads :math:`A_i` only (receivers decouple) and the sources :math:`j \le i` that the routing
    reaches, so it is causal; every input is reference data, so it is gauge invariant.
    """
    ctx_A = with_receiver_factors(ctx, A, beta)
    A_hat, X_hat, Y = edge_covariance_factors(ctx_A, beta, sources.c, sources.source, sources.values, sources.mean)
    trace = (sources.const + 2.0 * (A_hat * X_hat).sum(dim=(-2, -1))
             + (Y * (A_hat.transpose(-1, -2) @ A_hat)).sum(dim=(-2, -1)))                # (..., N)
    if sources.kappa is not None:                                                        # r = 1
        trace = trace + 2.0 * ((edge_self_coefficients(ctx_A) @ sources.kappa)
                               * A_hat.transpose(-1, -2)).sum(dim=(-2, -1))
    return trace / (2.0 * sources.scale)


def prior_routing_weights(
    log_prior: Optional[torch.Tensor],  # (N, N) or (H, N, N) routing log-prior, or None
    n:         int,
    *,
    dtype:     torch.dtype,
    device:    torch.device,
) -> torch.Tensor:                      # (N, N) pibar_ij
    r"""Head-averaged routing prior :math:`\bar\pi_{ij}`: the head mean of ``softmax(log_prior)`` rows, causal uniform without a prior."""
    if log_prior is None:
        causal = torch.ones(n, n, dtype=dtype, device=device).tril()
        return causal / causal.sum(dim=-1, keepdim=True)
    return torch.softmax(log_prior.to(dtype), dim=-1).reshape(-1, n, n).mean(dim=0)


def loop_indices(
    n:      int,
    window: Optional[int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:   # (L,) each: receivers i, middles j, origins k
    r"""Every loop :math:`(i, j, k)` with :math:`i - w \le k < j \le i` (:math:`w` = ``window``; None: :math:`0 \le k`)."""
    k, j = torch.triu_indices(n, n, offset=1, device=device)                            # all pairs k < j
    if window is not None:
        keep = j - k <= window
        k, j = k[keep], j[keep]
    upper = torch.full_like(k, n - 1) if window is None else torch.clamp(k + window, max=n - 1)
    counts = upper - j + 1                                                               # receivers j <= i <= upper
    k, j = k.repeat_interleave(counts), j.repeat_interleave(counts)
    start = torch.cumsum(counts, dim=0) - counts
    i = j + torch.arange(int(counts.sum()), device=device) - start.repeat_interleave(counts)
    return i, j, k


def loop_energy(
    A:        torch.Tensor,            # (..., N, K, r) receiver factors A_i^{l+1}
    B:        torch.Tensor,            # (..., N, K, r) sender factors B_b^{l+1}
    previous: LinkState,               # layer l's links (fixed during layer l+1's inference)
    weights:  torch.Tensor,            # (N, N) pibar_ij
    window:   Optional[int],
) -> torch.Tensor:                     # (..., N) Loop_i
    r"""Wilson-type loop energy :math:`\mathrm{Loop}_i = \sum_{(j,k)}\bar\pi_{ij}\bar\pi_{jk}\Psi(W_{ijk})`, :math:`\Psi(W) = \tfrac12\|W - I\|_F^2`.

    The loop :math:`W_{ijk} = (G_{ik}^{l+1}G_{kk}^l)^{-1}G_{ij}^{l+1}G_{jk}^l` (:math:`i - w \le k < j \le i`,
    time-like for :math:`j = i`) loses the layer-:math:`(l+1)` shared generator,

    .. math::
        W_{ijk} = (E_{kk}^l)^{-1}F_l^{-1}(E_{ik}^{l+1})^{-1}E_{ij}^{l+1}F_lE_{jk}^l,\qquad F_l = e^{L_l},

    and is the identity for the static shared-generator model. With :math:`\Phi_{ij} = \varphi_1(B_j^\top A_i)`,
    :math:`\Psi_{ik} = \varphi_1(-B_k^\top A_i)` at layer l+1 and :math:`\phi_{jk} = \varphi_1(b_k^\top a_j)`,
    :math:`\psi_{kk} = \varphi_1(-b_k^\top a_k)` for layer l's factors :math:`a, b`, every factor is a rank-r
    update of the identity, so

    .. math::
        W_{ijk} - I = V_{jk}b_k^\top + P_{ik}Q_{ijk}^\top,\qquad
        V_{jk} = a_j\phi_{jk} - a_k\psi_{kk}(I + b_k^\top a_j\phi_{jk}),\qquad
        P_{ik} = (E_{kk}^l)^{-1}F_l^{-1}A_i,

    .. math::
        Q_{ijk}^\top = Z^\top F_l + Z^\top F_la_j\phi_{jk}b_k^\top,\qquad
        Z^\top F_l = (I - \Psi_{ik}B_k^\top A_i)\Phi_{ij}B_j^\top F_l - \Psi_{ik}B_k^\top F_l,

    and :math:`\|W - I\|_F^2 = \operatorname{tr}(V^\top Vb_k^\top b_k) + 2\operatorname{tr}(V^\top PQ^\top b_k)
    + \operatorname{tr}(P^\top PQ^\top Q)` costs :math:`O(Kr^2)` per loop after :math:`O(NK^2r)` per layer.
    Everything runs in float64: :math:`e^{-L}` can be very large for trained generators, and the loop is a
    near-cancellation of its two paths. ``window=None`` keeps every loop, :math:`O(N^3)` loops per sequence
    with :math:`(L, K, r)` factors (small N only). Receivers with no loop (:math:`i < 1`) get 0. Loops of zero
    prior weight (a hard-masked routing prior, e.g. ``causal_windowed`` narrower than the loop window) are
    dropped before any factor is formed: they contribute nothing, and evaluating them would let a far
    sender's :math:`\varphi_1` overflow turn into :math:`0\cdot\infty = \mathrm{NaN}`.
    """
    f64 = torch.float64
    i, j, k = loop_indices(A.shape[-3], window, A.device)
    w = weights.to(f64)[i, j] * weights.to(f64)[j, k]                                    # (L,) pibar_ij pibar_jk
    keep = w > 0.0                                                                       # zero-weight loops are skipped
    if not bool(keep.all()):
        i, j, k, w = i[keep], j[keep], k[keep], w[keep]
    if i.numel() == 0:
        return A.new_zeros(A.shape[:-2])
    A64, B64 = A.to(f64), B.to(f64)
    a, b = previous.A.to(f64), previous.B.to(f64)
    F, F_inv = previous.F.to(f64), previous.F_inv.to(f64)
    tr = lambda X: X.transpose(-1, -2)
    a_j, a_k, b_k = a[..., j, :, :], a[..., k, :, :], b[..., k, :, :]                     # (..., L, K, r) layer l
    ba = tr(b_k) @ a_j                                                                   # (..., L, r, r) b_k^T a_j
    phi_jk = phi1(ba)
    psi_kk = phi1(-(tr(b_k) @ a_k))
    V = a_j @ phi_jk - a_k @ (psi_kk + psi_kk @ ba @ phi_jk)                            # (..., L, K, r)
    A_i = A64[..., i, :, :]                                                              # (..., L, K, r)
    B_j, B_k = B64[..., j, :, :], B64[..., k, :, :]
    BkA = tr(B_k) @ A_i                                                                  # (..., L, r, r)
    Phi_ij, Psi_ik = phi1(tr(B_j) @ A_i), phi1(-BkA)
    FiA = (F_inv @ A64)[..., i, :, :]                                                    # (..., L, K, r) F^{-1} A_i
    P = FiA - a_k @ (psi_kk @ (tr(b_k) @ FiA))                                           # (..., L, K, r)
    BF = tr(B64) @ F                                                                     # (..., N, r, K) B_b^T F_l
    eye_r = torch.eye(A.shape[-1], dtype=f64, device=A.device)
    ZF = (eye_r - Psi_ik @ BkA) @ Phi_ij @ BF[..., j, :, :] - Psi_ik @ BF[..., k, :, :]   # (..., L, r, K)
    Qt = ZF + (ZF @ a_j) @ phi_jk @ tr(b_k)                                              # (..., L, r, K)
    squared = (((tr(V) @ V) * (tr(b_k) @ b_k)).sum(dim=(-2, -1))
               + 2.0 * ((tr(V) @ P) * tr(Qt @ b_k)).sum(dim=(-2, -1))
               + ((tr(P) @ P) * (Qt @ tr(Qt))).sum(dim=(-2, -1)))                        # (..., L) ||W - I||_F^2
    out = torch.zeros(A.shape[:-2], dtype=f64, device=A.device).index_add(-1, i, 0.5 * w * squared)
    return out.to(A.dtype)


def latent_receiver_energy(
    A:           torch.Tensor,          # (..., N, K, r) receiver factors
    A0:          torch.Tensor,          # (..., N, K, r) amortized anchors a(xi^_i)
    ctx:         EdgeContext,
    beta:        torch.Tensor,          # (..., H, N, N)
    sources:     LatentSources,
    sigma:       float,                 # prior scale sigma_L
    loop_weight: float = 0.0,           # lambda_W
    loop:        Optional[LoopInputs] = None,
) -> torch.Tensor:                      # (..., N) F_i(A_i)
    r"""Per-receiver MAP energy (plan Section 3.1)

    .. math::
        \mathcal F_i(A_i) = E_i(A_i) + \frac{1}{2\sigma_L^2}\|A_i - A_i^0\|_F^2 + \lambda_W\mathrm{Loop}_i(A_i),

    with :math:`E_i` of :func:`latent_transport_energy` and :math:`\mathrm{Loop}_i` of :func:`loop_energy`
    (absent when ``loop`` is None or :math:`\lambda_W = 0`).
    """
    energy = (latent_transport_energy(A, ctx, beta, sources)
              + (A - A0).square().sum(dim=(-2, -1)) / (2.0 * sigma ** 2))
    if loop is not None and loop_weight > 0.0:
        energy = energy + loop_weight * loop_energy(A, ctx.B, *loop)
    return energy


def infer_latent_receivers(
    ctx:         EdgeContext,           # "lowrank" context; ctx.A are the amortized anchors A^0
    beta:        torch.Tensor,          # (..., H, N, N)
    sources:     LatentSources,
    *,
    sigma:       float,                 # sigma_L
    steps:       int,                   # T >= 1
    damping:     float,                 # rho in (0, 1]
    loop_weight: float = 0.0,           # lambda_W
    loop:        Optional[LoopInputs] = None,
    stats:       Optional[Dict[str, torch.Tensor]] = None,   # out-param: LATENT_DIAGNOSTIC_KEYS
) -> torch.Tensor:                      # (..., N, K, r) A^{(T)}
    r"""Damped fixed-point MAP inference of the latent receiver factors.

    .. math::
        A_i^{(t)} = (1-\rho)A_i^{(t-1)} + \rho\Bigl(A_i^0 - \sigma_L^2\nabla_{A_i}\bigl[E_i
          + \lambda_W\mathrm{Loop}_i\bigr](A_i^{(t-1)})\Bigr),\qquad A_i^{(0)} = A_i^0,

    the unit natural-gradient step of :math:`\mathcal F_i` in the Fisher metric of the prior; a fixed point is a
    stationary point of :func:`latent_receiver_energy`. The gradient of the summed (decoupled) energies comes
    from ``torch.autograd.grad``. With gradients enabled it uses ``create_graph=True``, so every outer parameter
    trains through the unrolled steps; under ``torch.no_grad`` the steps run under ``torch.enable_grad`` from a
    detached :math:`A^{(0)}` with ``create_graph=False`` and return a detached :math:`A^{(T)}`. ``stats``
    receives the means over receivers of :math:`\mathcal F_i(A^{(0)})`, :math:`\mathcal F_i(A^{(T)})`,
    :math:`\mathrm{Loop}_i(A^{(T)})` (0 without a previous layer, NaN with the loop prior off) and
    :math:`\|A_i^{(T)} - A_i^0\|_F/\|A_i^0\|_F`.
    """
    if torch.is_inference_mode_enabled():
        raise RuntimeError("content_link_latent differentiates the link energy and cannot run under "
                           "torch.inference_mode; use torch.no_grad")
    use_loop = loop is not None and loop_weight > 0.0

    def link_energy(A: torch.Tensor) -> torch.Tensor:              # (..., N) E_i + lambda_W Loop_i
        energy = latent_transport_energy(A, ctx, beta, sources)
        return energy + loop_weight * loop_energy(A, ctx.B, *loop) if use_loop else energy

    create_graph = torch.is_grad_enabled()
    A0 = ctx.A if create_graph else ctx.A.detach()
    A, initial = A0, None
    for _ in range(steps):
        with torch.enable_grad():
            point = A if A.requires_grad else A.detach().requires_grad_(True)
            energy = link_energy(point)
            grad, = torch.autograd.grad(energy.sum(), point, create_graph=create_graph)
        if initial is None:
            initial = energy.detach()                                  # F_i(A^0): the prior vanishes there
        target = A0 - sigma ** 2 * grad
        A = target if damping == 1.0 else (1.0 - damping) * A + damping * target
    if stats is not None:
        with torch.no_grad():
            final_A, anchor = A.detach(), A0.detach()
            shift = (final_A - anchor).square().sum(dim=(-2, -1))                            # (..., N)
            final = latent_transport_energy(final_A, ctx, beta, sources) + shift / (2.0 * sigma ** 2)
            loop_value = loop_energy(final_A, ctx.B, *loop) if use_loop else None
            if loop_value is not None:
                final = final + loop_weight * loop_value
            norm0 = anchor.square().sum(dim=(-2, -1)).sqrt().clamp_min(torch.finfo(anchor.dtype).tiny)
            stats.update({
                "relational_latent_energy_initial": initial.mean(),
                "relational_latent_energy_final": final.mean(),
                "relational_latent_loop_energy": (loop_value.mean() if loop_value is not None else
                                                  initial.new_tensor(0.0 if loop_weight > 0.0 else math.nan)),
                "relational_latent_receiver_shift": (shift.sqrt() / norm0).mean(),
            })
    return A
