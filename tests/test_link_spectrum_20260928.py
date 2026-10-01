r"""Content-link spectrum diagnostic (design doc 2026-09-28, Track 3).

The content link of head :math:`h` is :math:`L^{(h)}_{ij} = U_i M_h U_j^{-1}` with
:math:`M_h = \hat O_h \hat T_h` (:math:`K \times K`) and :math:`W_h = \hat T_h \hat O_h` (:math:`d \times d`).
These tests check, on tiny float64 CPU objects (H=2, d=2, K=4, N=4), that the logged scalars equal an
independent :math:`K \times K` computation, that they are invariant under the passive left action, the
common reference change and the per-head value basis while :math:`\sigma_{\max}(M_h)` is not, the
Sylvester identity :math:`\det(zI_K - M_h) = z^{K-d}\det(zI_d - W_h)`, the composition-defect identity,
that the self-loop maximum is the spectral radius of the whole-sequence fixed-routing map under a causal
prior, and that the relational diagnostics emit every key at every layer.
"""

from __future__ import annotations

from tests.profiles import test_config

import math
import warnings
from typing import Any, Dict, Tuple

import numpy as np
import pytest
import torch

from vfe3 import metrics
from vfe3.config import VFE3Config
from vfe3.contracts import CanonicalFrameContext
from vfe3.model.model import VFEModel
from vfe3.model.relational_operators import RelationalOperators
from vfe3.posthoc import maps as mp
from vfe3.train import _VAL_RELATIONAL_DIAG_KEYS

H, D, N = 2, 2, 4
K = H * D
SCOPES = [("head", "head"), ("full", "head"), ("head", "full"), ("full", "full")]


@pytest.fixture(autouse=True)
def _one_thread() -> None:
    torch.set_num_threads(1)


def _ops(input_scope: str, output_scope: str, residual: bool = False, scale: float = 1.0,
         perturb: bool = True, seed: int = 0) -> RelationalOperators:
    ops = RelationalOperators([D] * H, 3, input_scope=input_scope, readout_mode="predictive_content",
                              output_scope=output_scope, attention_residual=residual,
                              output_init_scale=scale).double()
    if perturb:
        gen = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            ops.T.add_(0.7 * torch.randn(ops.T.shape, generator=gen, dtype=torch.float64))
            ops.O.add_(0.7 * torch.randn(ops.O.shape, generator=gen, dtype=torch.float64))
    return ops


def _beta(seed: int = 1) -> torch.Tensor:                  # (1, H, N, N) causal row-stochastic
    gen = torch.Generator().manual_seed(seed)
    logits = torch.randn(1, H, N, N, generator=gen, dtype=torch.float64)
    causal = torch.ones(N, N, dtype=torch.bool).tril()
    return torch.softmax(logits.masked_fill(~causal, float("-inf")), dim=-1)


def _spectrum(ops: RelationalOperators, beta: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        return metrics.content_link_spectrum(*ops._ambient_content_maps(torch.float64),
                                             beta.diagonal(dim1=-2, dim2=-1), ops.attention_residual)


def _ambient_np(ops: RelationalOperators) -> Tuple[np.ndarray, np.ndarray]:
    r"""Independent block placement of the stored maps: (T_hat (H, d, K), O_hat (H, K, d))."""
    T, O = ops.T.detach().numpy(), ops.O.detach().numpy()
    if ops.input_scope == "head":
        T_hat = np.zeros((H, D, K))
        for h in range(H):
            T_hat[h, :, h*D:(h+1)*D] = T[h]
    else:
        T_hat = T
    if ops.output_scope == "head":
        O_hat = np.zeros((H, K, D))
        for h in range(H):
            O_hat[h, h*D:(h+1)*D, :] = O[h]
    else:
        O_hat = O
    return T_hat, O_hat


def _reference(ops: RelationalOperators, beta: torch.Tensor) -> Dict[str, float]:
    r"""The ten scalars from K x K matrices: eig of M_h with its K-d structural zeros dropped."""
    T_hat, O_hat = _ambient_np(ops)
    M = [O_hat[h] @ T_hat[h] for h in range(H)]                                  # H x (K, K)
    lams = []
    for h in range(H):
        lam = np.linalg.eigvals(M[h])
        lams.append(lam[np.argsort(-np.abs(lam))][:D])                           # drop the K-d smallest
    lam_all = np.concatenate(lams)
    r = 1.0 if ops.attention_residual else 0.0
    b = beta.diagonal(dim1=-2, dim2=-1)[0].numpy()                               # (H, N)
    rho_self = [np.abs(np.linalg.eigvals(r * np.eye(K) + sum(b[h, i] * M[h] for h in range(H)))).max()
                for i in range(N)]
    return {
        "relational_link_spectral_radius_max": float(np.abs(lam_all).max()),
        "relational_link_min_abs_eigenvalue": float(np.abs(lam_all).min()),
        "relational_link_mean_log_abs_eigenvalue": float(np.log(np.abs(lam_all)).mean()),
        "relational_link_unit_distance_max": float(np.abs(lam_all - 1.0).max()),
        "relational_link_idempotency_defect_max": float(np.abs(lam_all * (lam_all - 1.0)).max()),
        "relational_link_trace": float(sum(np.trace(m) for m in M)),
        "relational_link_trace_sq": float(sum(np.trace(m @ m) for m in M)),
        "relational_link_cross_trace": float(sum(np.trace(M[h] @ M[g])
                                                 for h in range(H) for g in range(h + 1, H))),
        "relational_link_selfloop_spectral_radius_max": float(max(rho_self)),
        "relational_link_selfloop_spectral_radius_mean": float(np.mean(rho_self)),
    }


@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("input_scope,output_scope", SCOPES)
def test_link_spectrum_matches_independent_kxk_reference(input_scope: str, output_scope: str,
                                                         residual: bool) -> None:
    ops, beta = _ops(input_scope, output_scope, residual), _beta()
    got, want = _spectrum(ops, beta), _reference(ops, beta)
    assert tuple(got) == metrics.LINK_SPECTRUM_KEYS
    for key in metrics.LINK_SPECTRUM_KEYS:
        assert math.isclose(got[key], want[key], rel_tol=1e-9, abs_tol=1e-12), key


def _block_diag(blocks: torch.Tensor) -> torch.Tensor:        # (H, d, d) -> (K, K)
    return torch.block_diag(*blocks.unbind(0))


@pytest.mark.parametrize("input_scope,output_scope", SCOPES)
def test_link_spectrum_is_invariant_where_sigma_max_is_not(input_scope: str, output_scope: str) -> None:
    ops, beta = _ops(input_scope, output_scope), _beta()
    before = _spectrum(ops, beta)
    T0, O0 = (x.detach().clone() for x in ops.content_factors())
    sigma_before = float(torch.linalg.svdvals(O0 @ T0).max())
    with torch.no_grad():
        T0_hat, O0_hat = ops._ambient_content_maps(torch.float64)
        M0 = O0_hat @ T0_hat                                                     # (H, K, K)
    gen = torch.Generator().manual_seed(5)
    eye = torch.eye(D, dtype=torch.float64)
    a_blocks = eye + 0.5 * torch.randn(H, D, D, generator=gen, dtype=torch.float64)   # common reference change
    B = eye + 0.5 * torch.randn(H, D, D, generator=gen, dtype=torch.float64)          # per-head value basis
    a = _block_diag(a_blocks)
    with torch.no_grad():
        ops.T.copy_(B @ (T0 @ (a if input_scope == "full" else a_blocks)))
        ops.O.copy_((torch.linalg.inv(a if output_scope == "full" else a_blocks) @ O0) @ torch.linalg.inv(B))
        T_hat, O_hat = ops._ambient_content_maps(torch.float64)
        np.testing.assert_allclose((O_hat @ T_hat).numpy(), (torch.linalg.inv(a) @ M0 @ a).numpy(),
                                   rtol=1e-10, atol=1e-12)                             # M_h -> a^{-1} M_h a
    after = _spectrum(ops, beta)
    for key in metrics.LINK_SPECTRUM_KEYS:
        assert math.isclose(before[key], after[key], rel_tol=1e-9, abs_tol=1e-12), key
    sigma_after = float(torch.linalg.svdvals(ops.O.detach() @ ops.T.detach()).max())
    assert abs(sigma_after - sigma_before) > 1e-3 * sigma_before


def test_link_spectrum_is_invariant_under_the_passive_left_action() -> None:
    r"""Self links and closed loops of L_ij = U_i M_h U_j^{-1} keep their spectra under U_i -> g_i U_i."""
    ops = _ops("full", "full")
    T_hat, O_hat = (x.detach() for x in ops._ambient_content_maps(torch.float64))
    M = O_hat @ T_hat                                                            # (H, K, K)
    gen = torch.Generator().manual_seed(7)
    eye = torch.eye(K, dtype=torch.float64)
    U = eye + 0.4 * torch.randn(N, K, K, generator=gen, dtype=torch.float64)
    g = eye + 0.4 * torch.randn(N, K, K, generator=gen, dtype=torch.float64)

    def link(frames: torch.Tensor, h: int, i: int, j: int) -> torch.Tensor:
        return frames[i] @ M[h] @ torch.linalg.inv(frames[j])

    def top(x: torch.Tensor) -> np.ndarray:                                     # nonzero spectrum, sorted
        lam = torch.linalg.eigvals(x).numpy()
        lam = lam[np.argsort(-np.abs(lam))][:D]
        return np.sort_complex(np.round(lam, 9))

    lam_W = metrics.content_link_eigenvalues(T_hat, O_hat).numpy()              # (H, d)
    for frames in (U, g @ U):
        for h in range(H):
            for i in range(N):
                np.testing.assert_allclose(top(link(frames, h, i, i)), np.sort_complex(np.round(lam_W[h], 9)),
                                           atol=1e-7)
        loop = link(frames, 0, 0, 1) @ link(frames, 1, 1, 2) @ link(frames, 0, 2, 0)
        np.testing.assert_allclose(top(loop), top(M[0] @ M[1] @ M[0]), atol=1e-7)
    open_before = float(torch.linalg.svdvals(link(U, 0, 0, 1)).max())
    open_after = float(torch.linalg.svdvals(link(g @ U, 0, 0, 1)).max())
    assert abs(open_after - open_before) > 1e-3 * open_before               # an open link is not invariant


@pytest.mark.parametrize("input_scope,output_scope", SCOPES)
def test_sylvester_identity_between_W_and_M(input_scope: str, output_scope: str) -> None:
    r"""det(z I_K - M_h) = z^{K-d} det(z I_d - W_h) at complex z; the eigenvalue helper returns spec W_h."""
    ops = _ops(input_scope, output_scope)
    T_hat, O_hat = (x.detach() for x in ops._ambient_content_maps(torch.float64))
    W = T_hat @ O_hat                                                            # (H, d, d)
    M = O_hat @ T_hat                                                            # (H, K, K)
    for z in (0.3 + 0.7j, -1.1 + 0.2j, 2.5 - 1.0j):
        lhs = torch.linalg.det(z * torch.eye(K, dtype=torch.complex128) - M.to(torch.complex128))
        rhs = z ** (K - D) * torch.linalg.det(z * torch.eye(D, dtype=torch.complex128) - W.to(torch.complex128))
        np.testing.assert_allclose(lhs.numpy(), rhs.numpy(), rtol=1e-10, atol=1e-12)
    lam = metrics.content_link_eigenvalues(T_hat, O_hat).numpy()
    for h in range(H):
        ref = np.linalg.eigvals(M[h].numpy())
        ref = ref[np.argsort(-np.abs(ref))]
        np.testing.assert_allclose(np.sort_complex(lam[h]), np.sort_complex(ref[:D]), atol=1e-12)
        assert np.abs(ref[D:]).max() < 1e-12                                     # the K-d structural zeros


def test_composition_defect_identity_and_its_spectral_surrogates() -> None:
    r"""L_ij L_jk - L_ik = U_i (M^2 - M) U_k^{-1} = U_i O_hat (W - I) T_hat U_k^{-1}; rho(M^2 - M) and
    rho(W - I) are the logged idempotency defect and unit distance."""
    ops = _ops("full", "full")
    T_hat, O_hat = (x.detach() for x in ops._ambient_content_maps(torch.float64))
    M, W = O_hat @ T_hat, T_hat @ O_hat
    gen = torch.Generator().manual_seed(11)
    U = torch.eye(K, dtype=torch.float64) + 0.4 * torch.randn(N, K, K, generator=gen, dtype=torch.float64)
    U_inv = torch.linalg.inv(U)
    heads = metrics.content_link_head_invariants(T_hat, O_hat)
    for h in range(H):
        L = U[:, None] @ M[h] @ U_inv[None, :]                                  # (N, N, K, K)
        via_W = O_hat[h] @ (W[h] - torch.eye(D, dtype=torch.float64)) @ T_hat[h]
        np.testing.assert_allclose((M[h] @ M[h] - M[h]).numpy(), via_W.numpy(), rtol=1e-12, atol=1e-12)
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    defect = L[i, j] @ L[j, k] - L[i, k]
                    want = U[i] @ via_W @ U_inv[k]
                    np.testing.assert_allclose(defect.numpy(), want.numpy(), rtol=1e-10,
                                               atol=1e-10 * float(want.abs().max()))
        rho_defect = float(torch.linalg.eigvals(M[h] @ M[h] - M[h]).abs().max())
        rho_unit = float(torch.linalg.eigvals(W[h] - torch.eye(D, dtype=torch.float64)).abs().max())
        assert math.isclose(heads[h]["link_idempotency_defect"], rho_defect, rel_tol=1e-9)
        assert math.isclose(heads[h]["link_unit_distance"], rho_unit, rel_tol=1e-9)


@pytest.mark.parametrize("residual", [False, True])
def test_selfloop_max_is_the_whole_sequence_spectral_radius(residual: bool) -> None:
    r"""Under causal routing the fixed-routing map mu -> m (+ mu) is block lower triangular, so its
    spectral radius is the maximum self-loop spectral radius."""
    ops, beta = _ops("full", "full", residual), _beta()
    gen = torch.Generator().manual_seed(13)
    blocks = torch.eye(D, dtype=torch.float64) + 0.4 * torch.randn(N, H, D, D, generator=gen, dtype=torch.float64)
    frame = CanonicalFrameContext(forward=blocks, inverse=torch.linalg.inv(blocks), block_diagonal=True)
    NK = N * K
    basis = torch.eye(NK, dtype=torch.float64).reshape(NK, N, K)                # (NK, N, K) unit inputs
    with torch.no_grad():
        image = ops.predictive_content(basis, beta.expand(NK, H, N, N), frame)   # (NK, N, K)
    local = image.reshape(NK, NK).T                                             # column m = image of e_m
    if residual:
        local = local + torch.eye(NK, dtype=torch.float64)
    upper = local.reshape(N, K, N, K).permute(0, 2, 1, 3)                       # (N, N, K, K) blocks
    assert all(float(upper[i, j].abs().max()) == 0.0 for i in range(N) for j in range(i + 1, N))
    rho_whole = float(torch.linalg.eigvals(local).abs().max())
    got = _spectrum(ops, beta)["relational_link_selfloop_spectral_radius_max"]
    assert math.isclose(got, rho_whole, rel_tol=1e-10)


@pytest.mark.parametrize("scale", [1.0, 0.5])
def test_identity_start_closed_forms(scale: float) -> None:
    ops, beta = _ops("full", "full", scale=scale, perturb=False), _beta()
    got = _spectrum(ops, beta)
    b_self = beta.diagonal(dim1=-2, dim2=-1)[0]                                  # (H, N)
    want = {
        "relational_link_spectral_radius_max": scale,
        "relational_link_min_abs_eigenvalue": scale,
        "relational_link_mean_log_abs_eigenvalue": math.log(scale),
        "relational_link_unit_distance_max": 1.0 - scale,
        "relational_link_idempotency_defect_max": scale * (1.0 - scale),
        "relational_link_trace": scale * K,
        "relational_link_trace_sq": scale * scale * K,
        "relational_link_cross_trace": 0.0,
        "relational_link_selfloop_spectral_radius_max": scale,                   # beta_00^h = 1
        "relational_link_selfloop_spectral_radius_mean": float((scale * b_self).amax(dim=0).mean()),
    }
    for key in metrics.LINK_SPECTRUM_KEYS:
        assert math.isclose(got[key], want[key], rel_tol=1e-12, abs_tol=1e-12), key


def test_eigenvalue_order_convention() -> None:
    W = torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, -2.0], [0.0, 2.0, 0.0]], dtype=torch.float64)
    lam = metrics.content_link_eigenvalues(W[None], torch.eye(3, dtype=torch.float64)[None])[0]
    np.testing.assert_allclose(lam.numpy(), np.array([2.0, 2.0j, -2.0j]), atol=1e-12)


def test_singular_head_gives_zero_min_modulus_and_nan_mean_log() -> None:
    r"""A rank-deficient head has min|lambda| = 0; its log|det W_h| = -inf is written as NaN, never -inf."""
    ops, beta = _ops("full", "full"), _beta()
    with torch.no_grad():
        ops.O[0, :, 0] = 0.0                                                     # W_0 = T_0 O_0 has a zero column
    got = _spectrum(ops, beta)
    assert got["relational_link_min_abs_eigenvalue"] < 1e-12
    assert math.isnan(got["relational_link_mean_log_abs_eigenvalue"])
    assert all(math.isfinite(got[key]) for key in metrics.LINK_SPECTRUM_KEYS
               if key != "relational_link_mean_log_abs_eigenvalue")


def test_selfloop_keys_are_nan_without_routing_weights() -> None:
    ops = _ops("head", "head")
    with torch.no_grad():
        got = metrics.content_link_spectrum(*ops._ambient_content_maps(torch.float64))
    assert math.isnan(got["relational_link_selfloop_spectral_radius_max"])
    assert math.isnan(got["relational_link_selfloop_spectral_radius_mean"])
    assert math.isfinite(got["relational_link_spectral_radius_max"])


def _model_cfg(**over: Any) -> VFE3Config:
    values = dict(vocab_size=7, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=2, n_e_steps=1,
                  family="gaussian_diagonal", gauge_group="block_glk", skip_belief_sigma_update=False,
                  e_phi_lr=0.0, oracle_unroll_grad=True, warmup_steps=0, max_steps=2, log_interval=0,
                  eval_interval=0, attention_mode="projected_predictive", decode_mode="linear",
                  relational_input_scope="full", relational_output_scope="full",
                  predictive_covariance="moment_matched", beta_attention_prior="causal")
    values.update(over)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return test_config(**values)


def test_relational_diagnostics_emit_link_keys_at_every_layer() -> None:
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(_model_cfg(relational_input_scope="head", relational_output_scope="head")).eval()
    tokens = torch.tensor([[0, 1, 2, 3]])
    d = model.diagnostics(tokens)
    for key in metrics.LINK_SPECTRUM_KEYS:
        assert key in d and math.isfinite(d[key]), key
    assert d["relational_link_spectral_radius_max"] == pytest.approx(1.0, abs=1e-12)     # W = I at init
    assert d["relational_link_unit_distance_max"] == pytest.approx(0.0, abs=1e-12)
    assert d["relational_link_selfloop_spectral_radius_max"] == pytest.approx(1.0, abs=1e-6)   # beta_00 = 1
    per_layer = model.diagnostics_per_layer(tokens)
    for key in metrics.LINK_SPECTRUM_KEYS:
        assert len(per_layer[key]) == 2 and all(math.isfinite(v) for v in per_layer[key]), key


@pytest.mark.parametrize("identity_content", [True, False])
def test_conditional_update_route_emits_nan_where_undefined(identity_content: bool) -> None:
    r"""Identity content has no content maps (all ten keys NaN); conditional_update has no predictive
    routing weights to form self loops (self-loop keys NaN, head keys finite)."""
    cfg = _model_cfg(n_layers=1, attention_mode="projected_conditional", relational_input_scope="head",
                     relational_output_scope="head", predictive_covariance="passthrough",
                     relational_identity_content=identity_content)
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = VFEModel(cfg).eval().diagnostics(torch.tensor([[0, 1, 2, 3]]))
    selfloop = ("relational_link_selfloop_spectral_radius_max", "relational_link_selfloop_spectral_radius_mean")
    for key in metrics.LINK_SPECTRUM_KEYS:
        assert math.isnan(d[key]) == (identity_content or key in selfloop), key


def test_val_relational_keys_cover_link_keys() -> None:
    assert {f"val_{key}" for key in metrics.LINK_SPECTRUM_KEYS} <= set(_VAL_RELATIONAL_DIAG_KEYS)


@pytest.mark.parametrize("input_scope,output_scope", SCOPES)
def test_posthoc_link_columns_are_invariant_under_the_reference_change(input_scope: str,
                                                                       output_scope: str) -> None:
    cfg = _model_cfg(relational_input_scope=input_scope, relational_output_scope=output_scope)
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(cfg)
    state = {k: v.detach().double().clone() for k, v in model.state_dict().items()}
    gen = torch.Generator().manual_seed(17)
    prefixes = ("relational_operators", "relational_operators_extra.0")
    for prefix in prefixes:                                                      # move off the identity start
        for leaf in ("T", "O"):
            state[f"{prefix}.{leaf}"] += 0.5 * torch.randn(state[f"{prefix}.{leaf}"].shape, generator=gen,
                                                           dtype=torch.float64)
    rows = mp.map_invariants(state, cfg)
    eye = torch.eye(D, dtype=torch.float64)
    moved = dict(state)
    for prefix in prefixes:
        a_blocks = eye + 0.5 * torch.randn(H, D, D, generator=gen, dtype=torch.float64)
        B = eye + 0.5 * torch.randn(H, D, D, generator=gen, dtype=torch.float64)
        a_in = _block_diag(a_blocks) if input_scope == "full" else a_blocks
        a_out = _block_diag(a_blocks) if output_scope == "full" else a_blocks
        for leaf in ("raw_score_q", "raw_score_k"):
            moved[f"{prefix}.{leaf}"] = state[f"{prefix}.{leaf}"] @ a_in
        moved[f"{prefix}.T"] = B @ (state[f"{prefix}.T"] @ a_in)
        moved[f"{prefix}.O"] = (torch.linalg.inv(a_out) @ state[f"{prefix}.O"]) @ torch.linalg.inv(B)
    after = mp.map_invariants(moved, cfg)
    assert len(rows) == cfg.n_layers * cfg.n_heads
    for a, b in zip(rows, after):
        for key in mp.LINK_COLUMNS:
            if key != "link_eigvec_condition":
                assert math.isclose(a[key], b[key], rel_tol=1e-9, abs_tol=1e-12), key
    assert any(abs(a["M_sigma_max"] - b["M_sigma_max"]) > 1e-3 * a["M_sigma_max"] for a, b in zip(rows, after))
    eig_rows = mp.link_eigenvalue_rows(state, cfg)
    assert len(eig_rows) == cfg.n_layers * H * D
    for row in eig_rows:
        assert row["abs"] == pytest.approx(abs(complex(row["re"], row["im"])))
    rho = {(r["layer"], r["head"]): r["link_spectral_radius"] for r in rows}
    for (layer, head), value in rho.items():
        assert value == pytest.approx(max(r["abs"] for r in eig_rows if r["layer"] == layer and r["head"] == head))
