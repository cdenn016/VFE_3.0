"""Full-input projected Gaussian routing and predictive content readout."""

from typing import Any, Dict, Optional, Tuple

import torch

from vfe3.contracts import CanonicalFrameContext
from vfe3.geometry.rope import (
    rope_commutant_diagonal, rope_commutant_is_diagonal, rope_commutant_projection, rope_rotated_pairs,
)
from vfe3.model.edge_links import edge_diagnostic_capture, receive_heads as edge_receive_heads
from vfe3.model.relational_operators import RelationalOperators, strictly_causal_routing_prior
from vfe3.relational_scoring import (
    _compute_dtype, diagonal_kl_term_medians, projected_condition_recovery, projected_moments, routing_cost,
)
from vfe3.runtime import ieee_fp32_matmul


# The clamp references whose scorer drops the query-only row constant (routing_kl_max_reference).
_QUERY_SHIFTED_REFERENCES: Tuple[str, ...] = ("query_shifted", "query_shifted_v1")


def _key_log_volume(
    G_k:       torch.Tensor,                   # (..., N, H, r) diagonal or (..., N, H, r, r) key covariances
    routing:   str,                            # operators.routing

    *,
    diagonal:  bool,
    key_noise: Optional[torch.Tensor] = None,  # (H,) n_h under noisy_key
) -> torch.Tensor:                             # (..., H, N) ell_j, NaN where the key covariance is not PD
    r"""The log-volume :math:`\ell_j = \tfrac12\log\det G_j` of each key covariance the routing cost reads.

    It is the key log-determinant term of the Gaussian KL, :math:`D_{ij} = c_i + \ell_j + t_{ij} + m_{ij}`
    (:func:`vfe3.relational_scoring.diagonal_kl_term_medians`), with :math:`G_j + n_hI` under ``noisy_key`` (the
    broadened key law the cost scores) and :math:`G_j` under ``noisy_query``. ``mean_distance`` reads no covariance
    and returns zeros. Under a joint rescaling of a head's comparison maps :math:`(S_Q, S_K) \to c(S_Q, S_K)` every
    :math:`G_j \to c^2G_j`, so :math:`\ell_j \to \ell_j + r\ln c` for every key at once, which is the shift the
    query-shifted cost :math:`D_{ij} - c_i` takes (audit 2026-10-20 item 1). It carries gradient. The dense route
    factors :math:`G_j` once more by Cholesky in the working dtype; a key whose factorization fails gives NaN, which
    :func:`query_shifted_row_reference` leaves out of the row mean.
    """
    if routing == "mean_distance":
        shape = G_k.shape[:-1] if diagonal else G_k.shape[:-2]                  # (..., N, H)
        return torch.zeros(shape, dtype=G_k.dtype, device=G_k.device).movedim(-1, -2)
    if diagonal:
        return .5 * G_k.log().sum(-1).movedim(-1, -2)                           # (..., H, N)
    if key_noise is not None:
        r = G_k.shape[-1]
        G_k = G_k + key_noise.to(G_k.dtype)[:, None, None] * torch.eye(r, dtype=G_k.dtype, device=G_k.device)
    L, info = torch.linalg.cholesky_ex(G_k)                                     # (..., N, H, r, r)
    volume = L.diagonal(dim1=-2, dim2=-1).log().sum(-1)                        # (..., N, H) = log det / 2
    volume = torch.where(info == 0, volume, torch.full_like(volume, float("nan")))
    return volume.movedim(-1, -2)                                               # (..., H, N)


@torch.no_grad()
def _query_row_constant(
    G_q:      torch.Tensor,                    # (..., N, H, r) diagonal or (..., N, H, r, r) query covariances
    routing:  str,                             # operators.routing

    *,
    diagonal: bool,
) -> torch.Tensor:                             # (..., H, N) float64 c_i, NaN where G_q is not PD
    r"""The query-only row constant :math:`c_i` the shifted references drop, for diagnostics (audit 2026-10-20 item 39).

    :math:`c_i = -\tfrac12(r + \log\det G_i)` for the Gaussian KL and ``noisy_key`` (the key noise broadens only the
    key law), :math:`-\tfrac12\log\det G_{q,i}` for ``noisy_query`` and 0 for ``mean_distance``, the constants of
    ``drop_query_constant`` in :mod:`vfe3.relational_scoring`. Float64, no gradient.
    """
    G = G_q.double()
    r = G.shape[-1]
    if routing == "mean_distance":
        shape = G.shape[:-1] if diagonal else G.shape[:-2]
        return torch.zeros(shape, dtype=torch.float64, device=G.device).movedim(-1, -2)
    if diagonal:
        logdet = G.log().sum(-1)                                                # (..., N, H)
    else:
        L, info = torch.linalg.cholesky_ex(G)
        logdet = 2 * L.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        logdet = torch.where(info == 0, logdet, torch.full_like(logdet, float("nan")))
    offset = 0.0 if routing == "noisy_query" else float(r)
    return (-.5 * (offset + logdet)).movedim(-1, -2)                            # (..., H, N)


def query_shifted_row_reference(
    score:          torch.Tensor,                  # (..., H, N, N) cost without its query constant, D_ij - c_i
    key_log_volume: torch.Tensor,                  # (..., H, N) ell_j of :func:`_key_log_volume`
    log_prior:      Optional[torch.Tensor] = None,  # (..., N, N) or (..., H, N, N), -inf = inadmissible
) -> torch.Tensor:                                 # (..., H, N, 1) m_i
    r"""Row mean of the key log-volumes over each query's admissible keys, the ``query_shifted`` clamp reference.

    .. math::
        m_i = \frac{1}{|\mathcal A_i|}\sum_{j\in\mathcal A_i}\ell_j,

    with :math:`\mathcal A_i` the keys of row :math:`i` whose cost and attention log-prior are finite and whose
    log-volume is finite; :math:`m_i = 0` on a row with none. It carries gradient (the exact derivative of the
    clamped cost; where nothing is capped every row of the softmax gradient sums to zero and the term vanishes up to
    roundoff, as for the ``row_min`` reference).
    """
    finite_volume = torch.isfinite(key_log_volume).unsqueeze(-2)                # (..., H, 1, N)
    admissible = torch.isfinite(score) & finite_volume                          # (..., H, N, N)
    if log_prior is not None:
        admissible = admissible & torch.isfinite(log_prior)
    volume = torch.where(finite_volume, key_log_volume.unsqueeze(-2),
                         torch.zeros_like(key_log_volume.unsqueeze(-2)))         # (..., H, 1, N)
    weights = admissible.to(score.dtype)                                        # (..., H, N, N)
    count = weights.sum(-1, keepdim=True)                                       # (..., H, N, 1)
    return (weights * volume.to(score.dtype)).sum(-1, keepdim=True) / count.clamp(min=1)


def _diagonal_comparison_route(operators: RelationalOperators, r: int, n_pairs: int) -> bool:
    r"""Whether a rotation-invariant comparison block of rank r scores its Gaussian KL on the diagonal route.

    :math:`P(G)` is structurally diagonal when at most one coordinate of the block stays unrotated
    (:func:`vfe3.geometry.rope.rope_commutant_is_diagonal`). The diagonal route
    (``diagonal_covariance`` of :func:`vfe3.relational_scoring.projected_gaussian_kl`) also requires the Gaussian-KL
    cost and conditioning-triggered recovery off (``projected_factorization='gram'``); every other case keeps the
    dense route with its certificate and recovery (audit 2026-10-10 round 2, item 30). ``pair_kl_gemm_tol`` no longer
    moves a diagonal :math:`P(G)` onto the dense route: the diagonal route has its own error estimate and
    direct-difference fallback (audit 2026-10-17 item 11).

    The recovery clause keeps what the certified toggles add on the dense route, not a float64 rescoring: a commutant
    projection has no source-root factor route, so ``cholesky_qr2_certified`` runs here as ``gram_certified``. An
    entry failing :math:`\varepsilon_{32}\lVert G\rVert_F\lVert G^{-1}\rVert_F \le 10^{-4}` has :math:`P(XX^\top)`
    rebuilt in the working dtype from its source root (:func:`vfe3.relational_scoring._commutant_root_factor`; float64 QR
    only if that rebuild fails), its pairs leave the GEMM for working-dtype triangular-solve tiles, and it is counted in
    :func:`vfe3.relational_scoring.source_recovery_elements`. On a diagonal :math:`P(G)` this moved a flagged pair by
    at most about 2e-5 nats against the diagonal route on the audit probe (audit 2026-10-16 F9).
    """
    return (rope_commutant_is_diagonal(r, n_pairs) and operators.routing == "gaussian_kl"
            and not projected_condition_recovery())


def relational_score(
    mu:           torch.Tensor,                        # (..., N, K) receiver (query) means
    sigma:        torch.Tensor,                        # (..., N, K) diagonal or (..., N, K, K) full covariance
    source_mu:    torch.Tensor,                        # (..., N, K) source (key) means
    source_sigma: torch.Tensor,                        # (..., N, K) or (..., N, K, K), same layout as sigma
    frame:        CanonicalFrameContext,               # (..., N, H, d, d) compact or (..., N, K, K) dense
    operators:    RelationalOperators,
    capture:      Optional[Dict[str, Any]] = None,     # out-param: pre-update projected covariances

    *,
    row_invariant: bool = False,
    term_support:  Optional[torch.Tensor] = None,      # (..., N, N) or (..., H, N, N) bool pairs of the term medians
    routing_aux:   Optional[Dict[str, torch.Tensor]] = None,   # out-param: key_log_volume (..., H, N)
) -> torch.Tensor:                                     # (..., H, N, N) projected routing costs
    r"""Compute KL(S_Q U_i^{-1} q_i || S_K U_j^{-1} q_j) or the noisy-query cost, retaining full reduced covariance.

    ``operators.routing`` selects the cost: ``"gaussian_kl"`` (projected_gaussian_kl) or
    ``"noisy_query"`` (projected_noisy_query_cost, the expected-likelihood source posterior), or
    ``"noisy_key"`` (projected_noisy_key_cost, the Gaussian KL against the key law broadened by the head's learned
    noise :math:`n_hI`, ``operators.routing_key_log_noise``), or
    ``"mean_distance"`` (projected_mean_distance_cost, the covariance-blind control).

    Every head projects the full canonical state through its own Q/K maps. The
    inverse frame retains its compact gauge blocks. A dense-layout frame must
    represent the same block-diagonal structure; cross-head frame coupling is
    outside the supported projected architecture.

    ``row_invariant=True`` declares that the caller reads the cost only through quantities invariant
    to a per-query-row constant (the kernel router under ``routing_kl_max_reference='row_min'``). The
    Gaussian KL then computes its query log-determinant without autograd (``query_row_constant_grad``
    of :func:`vfe3.relational_scoring.projected_gaussian_kl`); the returned values are unchanged.

    Under ``operators.kl_max_reference`` ``'query_shifted'`` or ``'query_shifted_v1'`` every cost is returned without
    its query-only row constant (``drop_query_constant`` of the scorers; :math:`-\tfrac12(r + \log\det G_i)` for the
    Gaussian KL), so the clamp of :func:`clamped_routing_cost` tests only the part of :math:`D_{ij}` the softmax reads
    (audit 2026-10-17 item 2). The softmax over :math:`j` is unchanged; the values can be negative. Under
    ``'query_shifted'`` the clamp also reads each key's log-volume :math:`\ell_j` (:func:`_key_log_volume`, with
    gradient), written to ``routing_aux["key_log_volume"]`` as a (..., H, N) tensor when ``routing_aux`` is given
    (audit 2026-10-20 item 1). With ``capture`` under either shifted reference, ``score_query_constant_pre`` records
    the dropped constant :math:`c_i` (:func:`_query_row_constant`) as a (..., H, N) float64 tensor (audit 2026-10-20
    item 39).

    With ``capture`` on the diagonal route, ``score_term_medians_pre`` records the per-head medians of the four cost
    terms (:func:`vfe3.relational_scoring.diagonal_kl_term_medians`, a (4, H + 1) float64 tensor, rows query constant,
    key log-determinant, trace, Mahalanobis, columns the heads and then all heads pooled) over the pairs of
    ``term_support`` (every pair when None); the dense
    route does not form the terms and records nothing (audit 2026-10-17 item 9).

    Under ``rope_mode='comparison_invariant'`` (``operators.rope_comparison_covariance == 'rotation_invariant'``) each
    comparison covariance is replaced by its projection :math:`P(G)` onto the commutant of the RoPE torus
    (:func:`vfe3.geometry.rope.rope_commutant_projection`) before the rotation, so the trace and log-determinant
    terms are position-free and only the mean term reads :math:`j - i`. The capture then records
    :math:`P(G)`, the covariance the router scores. The scorer takes its Gram route; the source tuples carry the
    rotated maps :math:`R_iS` with ``commutant_pairs``, so a failed (or, under condition recovery, flagged)
    factorization is rebuilt as :math:`P(XX^\top)` from the source (audit 2026-10-10 round 2, item 12). When
    :math:`P(G)` is structurally diagonal and :func:`_diagonal_comparison_route` admits it (Gaussian-KL routing,
    ``projected_factorization='gram'``, ``pair_kl_gemm_tol`` None) only diag(G) is formed and the
    Gaussian KL takes its diagonal route, with no Gram, factorization or source tuple.
    Under ``tie_comparison_maps`` a belief scored against itself is projected once and both roles share it.

    Under ``operators.frame_diagonal`` (``gaussian_frame_diagonal``) ``sigma`` and ``source_sigma`` hold the reference
    variances :math:`c_i`, so :math:`S U_i^{-1}\Sigma_iU_i^{-\top}S^\top = S\operatorname{diag}(c_i)S^\top`: the means are
    moved to :math:`\xi_i = U_i^{-1}\mu_i` and every role reads them through the identity frame, the diagonal source
    root :math:`S\operatorname{diag}(c_i)^{1/2}` included, with no frame product in the covariance.
    """
    if getattr(operators, "frame_diagonal", False):
        from vfe3.model.frame_actions import frame_diagonal_reference, frame_matvec
        same = source_mu is mu
        mu, reference_frame = frame_diagonal_reference(mu, frame)                 # (..., N, K) xi, identity frame
        source_mu = mu if same else frame_matvec(frame.inverse, source_mu, frame.block_diagonal)
        frame = reference_frame
    dims = operators.irrep_dims
    inverse = frame.inverse
    if not frame.block_diagonal:
        if len(dims) > 1:
            with torch.no_grad():
                block_mask = torch.block_diag(
                    *(torch.ones(d, d, dtype=torch.bool, device=inverse.device) for d in dims))
                if bool((inverse.detach()[..., ~block_mask] != 0).any()):       # one host sync
                    raise ValueError(
                        "relational_score requires a head-block-diagonal frame: the dense frame "
                        "inverse couples gauge heads, and scoring its diagonal blocks would disagree "
                        "with the full-inverse content readout")
        if operators.mixed_heads:
            # Every head reads the whole belief through the whole frame, L_h = S_h U^{-1}. Blocks of
            # different sizes have no compact stack, so each run reads the dense inverse as one
            # full-width block, (..., N, 1, K, K).
            inverse_runs = [inverse.unsqueeze(-3)] * len(operators.runs)
        else:
            d = dims[0]
            inverse_runs = [torch.stack(                                       # (..., N, H, d, d)
                [inverse[..., h*d:(h+1)*d, h*d:(h+1)*d] for h in range(len(dims))], dim=-3)]
    elif operators.mixed_heads:
        raise ValueError("a compact (..., N, H, d, d) frame cannot carry heads of different sizes")
    else:
        inverse_runs = [inverse]
    comparison = operators.comparison_runs()                                   # per run (n_r, d_r, K) pairs
    # An escalated float64 frame (audit 2026-09-23 #12) runs the whole projection and score in
    # float64; the results are returned at the dtype a lower-precision frame would have produced.
    public = _compute_dtype(mu, sigma, source_mu, source_sigma, *(S for pair in comparison for S in pair))
    score = routing_cost(operators.routing)                                  # ROUTING_COSTS registry
    # noisy_key is the Gaussian KL of a broadened key law, so it takes the Gaussian-KL options.
    gaussian_kl = operators.routing in ("gaussian_kl", "noisy_key")
    # query_shifted (and its v1 form): the scorers drop the query-only row constant, so its gradient is exactly zero
    # as under row_min.
    reference = getattr(operators, "kl_max_reference", "absolute")
    query_shifted = reference in _QUERY_SHIFTED_REFERENCES
    # query_shifted (not v1) clamps relative to the row mean of the key log-volumes, which the clamp reads from here.
    want_key_volume = routing_aux is not None and reference == "query_shifted"
    cost_options: Dict[str, Any] = ({"query_row_constant_grad": False}
                                    if (row_invariant or query_shifted) and gaussian_kl else {})
    if query_shifted:
        cost_options["drop_query_constant"] = True
    if gaussian_kl and operators.pair_kl_gemm_tol is not None:
        cost_options["gemm_expansion_tol"] = operators.pair_kl_gemm_tol
    key_log_noise = getattr(operators, "routing_key_log_noise", None)             # (H,) noisy_key only
    scores, captured_q, captured_k, term_medians = [], [], [], []
    key_volumes, query_constants = [], []                                       # per run (..., H_run, N)
    # One scoring call per run: the heads of a run share their rank d_r. Each call ends in its own
    # host transfer, so a tower of R runs pays R of them per call (one for equal heads), unless its
    # verdicts can only raise (the diagonal route), which deferred_host_checks then queues.
    first_head = 0                                                             # index of the run's first head
    for (S_q, S_k), inverse in zip(comparison, inverse_runs):
        H, K = S_q.shape[0], S_q.shape[-1]
        # Both roles read the one shared frame: there is no separate key-side inverse.
        key_inverse = inverse
        # Each head reads the whole belief; gauge blocks stay compact.
        mu_h, source_mu_h = (                                                   # (..., N, H, K)
            x.unsqueeze(-2).expand(*x.shape[:-1], H, K) for x in (mu, source_mu))
        sigma_h, source_sigma_h = (                                             # (..., N, H, K[, K])
            x.unsqueeze(-3).expand(*x.shape[:-2], H, K, K)
            if x.ndim == mu.ndim + 1 else x.unsqueeze(-2).expand(*x.shape[:-1], H, K)
            for x in (sigma, source_sigma))
        self_scored = mu is source_mu and sigma is source_sigma and inverse is key_inverse
        # tie_comparison_maps scoring a belief against itself: S_K is S_Q, so both roles are one projection.
        tied = self_scored and S_q is S_k
        # gaussian_block: Sigma is head-block diagonal, so G = L Sigma L^T reads only the diagonal blocks.
        head_dims = tuple(operators.irrep_dims) if operators.block_covariance else None
        rope_base = getattr(operators, "comparison_rope_base", None)
        invariant = rope_base is not None and operators.rope_comparison_covariance == "rotation_invariant"
        n_pairs = rope_rotated_pairs(S_q.shape[-2], operators.comparison_rope_fraction) if invariant else 0
        # A structurally diagonal P(G) needs only diag(G): the moments return it (..., N, H, r) in place of G.
        diagonal = invariant and _diagonal_comparison_route(operators, S_q.shape[-2], n_pairs)
        moment_options = {"head_dims": head_dims, **({"diagonal_only": True} if diagonal else {})}
        if tied:
            a_q, G_q = projected_moments(mu_h, sigma_h, inverse, S_q, **moment_options)  # (..., N, H, d), (..., N, H, d, d)
            a_k, G_k = a_q, G_q
        elif self_scored:
            # Predictive readout scores a belief against itself: one batched projection over both
            # comparison maps. S_q and S_k still differ, so two L = S U^-1 maps are formed, but the
            # covariance contraction G = L Sigma L^T is batched across the role axis.
            a, G = projected_moments(mu_h, sigma_h, inverse, torch.stack((S_q, S_k)), **moment_options)
            a_q, G_q = a[0], G[0]
            a_k, G_k = a[1], G[1]
        else:
            a_q, G_q = projected_moments(mu_h, sigma_h, inverse, S_q, **moment_options)  # (..., N, H, d), (..., N, H, d, d)
            a_k, G_k = projected_moments(source_mu_h, source_sigma_h, key_inverse, S_k, **moment_options)
        if invariant:
            # rope_comparison_covariance='rotation_invariant' (audit 2026-10-09 round 2, comparison-space RoPE
            # self-lock): project G onto the commutant of the RoPE torus, one isotropic variance per rotating
            # plane and the full non-rotating block, so R_i P(G) R_i^T = P(G) and only the mean term of the
            # pair cost carries the relative position j - i. On the diagonal route P(G) is its diagonal.
            project = rope_commutant_diagonal if diagonal else rope_commutant_projection
            G_q = project(G_q, n_pairs)                                         # (..., N, H, r[, r])
            G_k = G_q if tied else project(G_k, n_pairs)
        if capture is not None:
            # The capture records the matrix P(G) on both routes.
            captured_q.append((torch.diag_embed(G_q.detach()) if diagonal else G_q.detach()).to(public))
            captured_k.append((torch.diag_embed(G_k.detach()) if diagonal else G_k.detach()).to(public))
        projection_q = (sigma_h, inverse, S_q, mu_h)
        projection_k = (source_sigma_h, key_inverse, S_k, source_mu_h)
        if rope_base is not None:
            # comparison rope_mode (audit 2026-10-07, F28): rotate token i's projected moments by R_i in the
            # r-dimensional comparison space, a -> R_i a, G -> R_i G R_i^T, for query and key alike. A Gaussian KL
            # (and the noisy-query and mean-distance costs) is invariant under a common orthogonal map, so the
            # pair (i, j) cost equals the cost of (a_i, G_i) against (R_{j-i} b_j, R_{j-i} G_j R_{j-i}^T): a function
            # of positions only through j - i. The rotated moments are those of the per-token comparison map
            # R_i S, so the source tuples carry R_i S and the source-root recovery and factor route factor
            # X_i = R_i S U_i^{-1} Sigma_i^{1/2} exactly as on the unrotated route (audit 2026-10-08 P1-2).
            # The rotation runs under the same IEEE-fp32 pin as the moments it rotates (P1-1).
            # Under rope_comparison_covariance='rotation_invariant' only the means rotate: R_i P(G) R_i^T = P(G)
            # exactly, so G is left as projected (no rounding from a rotation). P is invariant under every torus
            # element, P(R_i X X^T R_i^T) = P(X X^T), so the source tuples still carry R_i S and, with
            # commutant_pairs, a recovery rebuilds P(X X^T) from that root (audit 2026-10-10 round 2, item 12).
            # The diagonal route reads no source tuple and forms no rotated map.
            r = S_q.shape[-2]
            R_q = operators.comparison_rope_rotation(a_q.shape[-3], r, a_q.device, a_q.dtype)  # (N, r, r)
            R_k = R_q if a_k.shape[-3] == a_q.shape[-3] else operators.comparison_rope_rotation(
                a_k.shape[-3], r, a_k.device, a_k.dtype)
            with ieee_fp32_matmul():
                a_q = torch.einsum("nst,...nht->...nhs", R_q, a_q)
                a_k = a_q if tied else torch.einsum("nst,...nht->...nhs", R_k, a_k)
                if not invariant:
                    G_q = torch.einsum("nst,...nhtu,nvu->...nhsv", R_q, G_q, R_q)
                    G_k = G_q if tied else torch.einsum("nst,...nhtu,nvu->...nhsv", R_k, G_k, R_k)
                if not diagonal:
                    S_q_rot = torch.einsum("nst,htk->nhsk", R_q.to(S_q.dtype), S_q)   # (N, H, r, K)
                    S_k_rot = S_q_rot if tied else torch.einsum("nst,htk->nhsk", R_k.to(S_k.dtype), S_k)
            if not invariant:
                G_q = 0.5 * (G_q + G_q.transpose(-1, -2))
                G_k = G_q if tied else 0.5 * (G_k + G_k.transpose(-1, -2))
            if diagonal:
                projection_q = projection_k = None
            else:
                projection_q = (sigma_h, inverse, S_q_rot, mu_h)
                projection_k = (source_sigma_h, key_inverse, S_k_rot, source_mu_h)
        run_options = dict(cost_options)
        if key_log_noise is not None:
            run_options["key_noise"] = key_log_noise[first_head:first_head + H].exp()   # (H_run,) n_h
        first_head += H
        if want_key_volume:
            key_volumes.append(_key_log_volume(G_k, operators.routing, diagonal=diagonal,
                                               key_noise=run_options.get("key_noise")).to(public))   # (..., H, N)
        if capture is not None and query_shifted:
            with torch.no_grad():
                query_constants.append(_query_row_constant(G_q.detach(), operators.routing, diagonal=diagonal))
        if diagonal:
            run_options["diagonal_covariance"] = True                           # G_q, G_k are (..., N, H, r)
            if capture is not None:
                run_support = (term_support[..., first_head - H:first_head, :, :]   # this run's heads
                               if term_support is not None and term_support.dim() >= 3
                               and term_support.shape[-3] > 1 else term_support)
                term_medians.append(diagonal_kl_term_medians(a_q, G_q, a_k, G_k, run_support))   # (4, H_run + 1)
        elif invariant:
            run_options["commutant_pairs"] = n_pairs
        # G_q, G_k are bitwise symmetric here (projected_moments and the RoPE branch both end in a
        # symmetrization), so the scorer skips its isclose symmetry check.
        scores.append(score(
            a_q, G_q, a_k, G_k,
            projection_q=projection_q,
            projection_k=projection_k,
            symmetric_inputs=True,
            **run_options).to(public))
    if capture is not None:
        # One (..., N, H, r, r) tensor for equal heads; a list with one per run for mixed head sizes.
        capture["projected_covariance_pre"] = captured_q[0] if len(captured_q) == 1 else captured_q
        capture["projected_key_covariance_pre"] = captured_k[0] if len(captured_k) == 1 else captured_k
        if term_medians:
            # (4, H + 1): the heads, then the median pooled over every head (NaN across several runs).
            pooled = (term_medians[0][:, -1:] if len(term_medians) == 1
                      else torch.full_like(term_medians[0][:, -1:], float("nan")))
            capture["score_term_medians_pre"] = torch.cat([*(t[:, :-1] for t in term_medians), pooled], dim=-1)
        if query_constants:
            capture["score_query_constant_pre"] = torch.cat(query_constants, dim=-2)    # (..., H, N) c_i
    if want_key_volume:
        routing_aux["key_log_volume"] = torch.cat(key_volumes, dim=-2)              # (..., H, N) ell_j
    return scores[0] if len(scores) == 1 else torch.cat(scores, dim=-3)      # (..., H, N, N)


def clamped_routing_cost(
    score:     torch.Tensor,                     # (..., H, N, N) routing cost D_ij
    operators: RelationalOperators,
    *,
    kl_max:    float,
    log_prior: Optional[torch.Tensor] = None,    # (..., N, N) or (..., H, N, N), -inf = inadmissible
    key_log_volume: Optional[torch.Tensor] = None,   # (..., H, N) ell_j; required under 'query_shifted'
) -> torch.Tensor:
    r"""Clamp the kernel-router cost at ``kl_max`` relative to ``operators.kl_max_reference``.

    ``absolute`` clamps :math:`D_{ij}` to :math:`[0, \kappa]`. ``query_shifted`` receives the cost without its
    query-only row constant, :math:`\tilde D_{ij} = D_{ij} - c_i` (:func:`relational_score` drops it, audit 2026-10-17
    item 2), subtracts the admissible row mean :math:`m_i` of the key log-volumes :math:`\ell_j = \tfrac12\log\det G_j`
    (:func:`query_shifted_row_reference`, ``key_log_volume`` from :func:`relational_score`) and clamps from above only,

    .. math::
        \bar D_{ij} = \min(D_{ij} - c_i - m_i, \kappa) = \min(t_{ij} + q_{ij} + \ell_j - m_i, \kappa),

    with :math:`t_{ij}` the trace and :math:`q_{ij}` the Mahalanobis term and NaN and :math:`+\infty` mapped to
    :math:`\kappa`. Beta is the softmax of the unclamped cost wherever nothing is capped (:math:`c_i + m_i` is a row
    constant). The tested quantity is invariant under a joint rescaling :math:`(S_Q, S_K) \to c(S_Q, S_K)` of a head's
    comparison maps, which leaves every KL and beta unchanged: :math:`t` and :math:`q` are invariant and every
    :math:`\ell_j` moves by :math:`r\ln c`, as does :math:`m_i`. More generally a common map :math:`A` of the
    comparison space that commutes with the RoPE rotations (:math:`G \to AGA^\top`, :math:`a \to Aa`) moves every
    :math:`\ell_j` and :math:`m_i` by :math:`\log|\det A|` and leaves :math:`\bar D` unchanged. The audit 2026-10-17
    form ``query_shifted_v1``, :math:`\min(D_{ij} - c_i, \kappa)`, kept the key log-determinant and so moved by
    :math:`r\ln c` along that flat direction; heads crossed the cap by scale alone and stayed fully clamped with zero
    router gradient (audit 2026-10-20 item 1). It is retained for configs saved before the fix. ``row_min`` clamps the gap
    :math:`D_{ij}-\min_{j' \in \mathcal{A}_i}D_{ij'}` over the admissible sources :math:`\mathcal{A}_i`
    (audit 2026-10-08 P1-4): the softmax is
    row-shift invariant, so beta is unchanged wherever a row keeps an entry below the cap, and a row
    whose every cost exceeds ``kl_max`` still routes by its cost order. A row with no admissible finite
    cost is not shifted.

    The row minimum carries gradient (audit 2026-10-09 round 2): ``amin``'s subgradient, which sends
    :math:`-\sum_{j'} \partial L/\partial \tilde D_{ij'}` to the argmin entry and is the exact derivative of the
    clamped cost wherever the argmin is unique (``amin`` splits it evenly over tied minima). A detached minimum
    gave the wrong router gradient once the cap binds. Every row of the cost gradient now sums to zero, so the
    row-constant query log-determinant skip (``row_invariant``) is exact also when the cap binds. Where no gap
    exceeds the cap the softmax gradient already sums to zero over the row, and the added term vanishes up to
    roundoff.
    """
    from vfe3.families.base import safe_kl_clamp
    if operators.kl_max_reference == "query_shifted_v1":
        return score.clamp(max=kl_max).nan_to_num(nan=kl_max, posinf=kl_max)
    if operators.kl_max_reference == "query_shifted":
        if key_log_volume is None:
            raise ValueError("routing_kl_max_reference='query_shifted' clamps relative to the row mean of the key "
                             "log-volumes; pass key_log_volume (relational_score's routing_aux)")
        shifted = score - query_shifted_row_reference(score, key_log_volume, log_prior)   # (..., H, N, N)
        return shifted.clamp(max=kl_max).nan_to_num(nan=kl_max, posinf=kl_max)
    if operators.kl_max_reference == "row_min":
        admissible = torch.isfinite(score)
        if log_prior is not None:
            admissible = admissible & torch.isfinite(log_prior)
        row_min = score.masked_fill(~admissible, float("inf")).amin(dim=-1, keepdim=True)   # (..., H, N, 1)
        row_min = torch.where(torch.isfinite(row_min), row_min, torch.zeros_like(row_min))
        score = score - row_min                                                 # (..., H, N, N)
    return safe_kl_clamp(score, kl_max=kl_max)


def relational_predictive_readout(
    mu:        torch.Tensor,
    sigma:     torch.Tensor,
    frame:     CanonicalFrameContext,
    operators: RelationalOperators,
    *,
    tau:       'float | torch.Tensor' = 1.0,
    kl_max:    float = 100.0,
    log_prior:   Optional[torch.Tensor] = None,
    value_frame: Optional[CanonicalFrameContext] = None,
    capture:     Optional[Dict[str, Any]] = None,
    return_covariance: bool = False,
    link_state:  Optional[dict] = None,
    floor_anchor: Optional[torch.Tensor] = None,   # return_covariance: the local prior covariance Sigma_p
) -> 'torch.Tensor | Tuple[torch.Tensor, torch.Tensor]':
    r"""Return the conditional predictive mean with the configured projected routing cost and live source graphs.

    This is a separate content mixture, not a Fisher step on the comparison belief. With
    ``return_covariance`` it also returns ``(mean, sigma_plus)``, where sigma_plus is the
    moment-matched covariance of the same predictive kernel
    (``RelationalOperators.predictive_covariance_update``), computed from the same routing weights
    and value frame as the mean. No source, router, or map is detached. ``tau`` is the configured
    temperature; under ``learn_attention_temperature`` the routing uses
    :meth:`RelationalOperators.routing_temperature` of this layer.
    Under ``content_link_edges`` the edge context (factors, :math:`\Phi_{ij}` and the float64 chart
    stacks) is computed once per layer and shared by the mean, the covariance and the capture.
    Under ``content_link_latent`` its receiver factors are inferred from the source covariances of
    ``sigma`` (whatever ``predictive_covariance`` is), and a positive loop weight reads and writes
    ``link_state`` (``{"previous": ..., "current": ...}``, threaded across layers by ``vfe_stack``).
    ``floor_anchor`` is the local prior covariance :math:`\Sigma_p` whose multiple :math:`f\,\Sigma_p` floors the
    returned covariance (:meth:`RelationalOperators.floor_term`); it is required with ``return_covariance``.
    ``capture`` receives the router ``beta_pre``, the clamped cost ``score_energy_pre`` and the boolean map
    ``score_nonfinite_pre`` of raw costs that were NaN or infinite before the clamp mapped them to ``kl_max``
    (audit 2026-10-16 F10), and, from the raw cost before the reference and the clamp (audit 2026-10-17 item 9),
    ``score_raw_row_min_pre`` and ``score_raw_row_max_pre``, (..., H, N) float64 minimum and maximum over each row's
    admissible finite costs (NaN for a row with none), and on the diagonal route ``score_term_medians_pre``
    (:func:`relational_score`). Under ``query_shifted`` and ``query_shifted_v1`` the raw cost is the one without its
    query constant, :math:`D_{ij} - c_i`, and ``score_row_shift_pre`` ((..., H, N) float64) is the whole row
    constant removed before the clamp, :math:`c_i + m_i` (:math:`c_i` under ``query_shifted_v1``), so that
    ``score_energy_pre`` plus it is the absolute cost as the router read it, :math:`\min(D_{ij}, \kappa + c_i + m_i)`
    (audit 2026-10-20 item 39).

    Under ``predictive_readout='minimal_agent'`` the routing, the content and the residual are replaced by the
    exact Gaussian conditioning of :meth:`RelationalOperators.minimal_agent_update`, which reads the ROUTING frame
    for query, key, content and write-back (a distinct ``value_frame`` with different matrices is refused: the tied
    key is the content). ``tau`` and ``kl_max`` are not read (the posterior has temperature 1 and no clamp); the
    result is the posterior mean, with no residual added, and ``return_covariance`` gives the posterior covariance
    plus the prior-anchored floor.
    """
    if getattr(operators, "predictive_readout", "kernel") == "minimal_agent":
        if value_frame is not None and value_frame is not frame and not (
                value_frame.block_diagonal == frame.block_diagonal
                and torch.equal(value_frame.forward, frame.forward)
                and torch.equal(value_frame.inverse, frame.inverse)):
            raise ValueError("predictive_readout='minimal_agent' reads key and content in one frame; the value frame "
                             "differs from the routing frame (rope_mode='frame')")
        mean, sigma_plus = operators.minimal_agent_update(
            mu, sigma, frame, log_prior, return_covariance=return_covariance, capture=capture,
            floor_anchor=floor_anchor)
        return (mean, sigma_plus) if return_covariance else mean
    from vfe3.free_energy import attention_weights
    dtype = torch.float64 if mu.dtype == torch.float64 else torch.float32
    edges = None
    latent = getattr(operators, "content_link_latent", False)
    latent_out: Optional[Dict[str, Any]] = {} if (latent and capture is not None) else None
    with torch.autocast(device_type=mu.device.type, enabled=False):
        mu_work = mu.to(dtype)
        sigma_work = sigma.to(dtype)
        # Under the row_min reference the clamp and softmax read only row gaps, so the query
        # log-determinant (a row constant of the Gaussian KL) needs no autograd.
        row_invariant = operators.kl_max_reference == "row_min"
        support = (None if capture is None or log_prior is None
                   else torch.isfinite(log_prior.detach()))                  # (..., N, N) or (..., H, N, N)
        routing_aux: Dict[str, torch.Tensor] = {}
        raw_score = relational_score(mu_work, sigma_work, mu_work, sigma_work, frame, operators, capture,
                                     **({"row_invariant": True} if row_invariant else {}),
                                     **({"term_support": support} if support is not None else {}),
                                     routing_aux=routing_aux)                   # (..., H, N, N)
        # The clamp maps NaN and +inf to kl_max, so a capped captured cost cannot tell a failed cost from a
        # finite one at the cap; the capture keeps which raw costs were non-finite (audit 2026-10-16 F10).
        score_nonfinite = None if capture is None else ~torch.isfinite(raw_score.detach())   # (..., H, N, N)
        if capture is not None:
            with torch.no_grad():
                # Pre-clamp row extremes over admissible finite costs (audit 2026-10-17 item 9).
                raw = raw_score.detach().double()                               # (..., H, N, N)
                admissible = ~score_nonfinite if support is None else ~score_nonfinite & support
                capture["score_raw_row_min_pre"] = raw.masked_fill(~admissible, float("inf")).amin(dim=-1).where(
                    admissible.any(dim=-1), torch.tensor(float("nan"), dtype=raw.dtype, device=raw.device))
                capture["score_raw_row_max_pre"] = raw.masked_fill(~admissible, float("-inf")).amax(dim=-1).where(
                    admissible.any(dim=-1), torch.tensor(float("nan"), dtype=raw.dtype, device=raw.device))
        key_log_volume = routing_aux.get("key_log_volume")                     # (..., H, N) or None
        score = clamped_routing_cost(
            raw_score, operators, kl_max=kl_max, log_prior=None if log_prior is None else log_prior.to(dtype),
            key_log_volume=key_log_volume)
        if capture is not None and "score_query_constant_pre" in capture:
            # The row constant the shifted references removed before the clamp (audit 2026-10-20 item 39):
            # c_i, plus the key log-volume row mean m_i under 'query_shifted'.
            with torch.no_grad():
                shift = capture["score_query_constant_pre"].double()            # (..., H, N)
                if key_log_volume is not None:
                    shift = shift + query_shifted_row_reference(
                        raw_score.detach(), key_log_volume.detach(),
                        None if log_prior is None else log_prior.to(dtype)).squeeze(-1).double()
                capture["score_row_shift_pre"] = shift
        beta = attention_weights(score, tau=operators.routing_temperature(tau, dtype),
                                 log_prior=None if log_prior is None else log_prior.to(dtype))
        if getattr(operators, "content_link_edges", "off") != "off":
            inference = ({} if not latent else dict(
                sigma=sigma_work, log_prior=None if log_prior is None else log_prior.to(dtype),
                link_state=link_state, latent_out=latent_out))
            edges = operators.edge_context(mu_work, beta, value_frame or frame, **inference)
            receive = [edge_receive_heads(edges, len(operators.irrep_dims))]
        else:
            _, receive = operators.content_factor_runs()         # O' once per layer for mean and covariance
        shared = {} if edges is None else {"edges": edges}
        result = operators.predictive_content(mu_work, beta, value_frame or frame, receive,
                                              **shared).to(mu.dtype)
        if operators.attention_residual:
            result = mu + result
        # beta_ij = 0 exactly wherever the prior is -inf, so a strictly causal prior lets the routed spreads skip the
        # sources after each query tile (audit 2026-10-20 item 18); read from the prior object the routing used.
        sigma_plus = (operators.predictive_covariance_update(mu_work, sigma_work, beta, value_frame or frame,
                                                             receive, floor_anchor=floor_anchor,
                                                             causal_sources=strictly_causal_routing_prior(log_prior),
                                                             **shared).to(sigma.dtype)
                      if return_covariance else None)
    if capture is not None:
        capture.update(beta_pre=beta.detach(), score_energy_pre=score.detach(), score_nonfinite_pre=score_nonfinite,
                       relational_mean_displacement_norm=(result-mu).norm().detach())
        if edges is not None:
            capture.update(edge_diagnostic_capture(edges, beta))
        if latent_out is not None:
            capture.update(latent_out)
    if return_covariance:
        return result, sigma_plus
    return result
