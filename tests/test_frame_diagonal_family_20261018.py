r"""The revived frame-diagonal family ``gaussian_frame_diagonal`` (2026-10-18; audit 2026-10-17 item 3).

:math:`\Sigma_i = U_i\operatorname{diag}(c_i)U_i^\top` with the reference variances :math:`c_i` stored. Tiny CPU cases only
(K <= 16, at most two layers except the six-layer depth recursion at K = 8, two heads, vocabulary <= 64, N <= 16).
The passive local gauge invariance of the live route is pinned in the harness of
``tests/test_audit_20261016_a1_passive_gauge_model.py`` (its ``frame_diagonal*`` cases); here are the closed forms,
the readout and router against a dense full-covariance evaluation, the depth recursion, the decoders, the block, the
diagnostics, a float64 finite-difference gradient check, a few training steps, and the configuration envelope.
"""
from __future__ import annotations

import math
import warnings
from typing import Any, Dict, Tuple

import pytest
import torch

from tests.profiles import config_values, live_config, make_relational_operators
from tests.test_audit_20261017_l2_covariance import _conditioned_frames, _dense
from tests.test_blockwise_moment_matched_20261004 import _case
from vfe3.belief import BeliefState
from vfe3.config import VFE3Config, config_from_serialized, serialize_config
from vfe3.contracts import CanonicalFrameContext
from vfe3.families import get_family, kl
from vfe3.families.frame_gaussian import FrameDiagonalGaussian, frame_diagonal_kl
from vfe3.families.gaussian import FullGaussian
from vfe3.gauge_covariant import fully_gauge_covariant_overrides, gauge_covariance_failures
from vfe3.gradients.relational import relational_score
from vfe3.model.frame_actions import frame_diagonal_local_covariance, reference_variances
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import _pure_path_report

F64 = torch.float64
FAMILY = "gaussian_frame_diagonal"


@pytest.fixture(autouse=True)
def _quiet_two_threads():
    torch.set_num_threads(2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _spd(*shape: int, generator: torch.Generator) -> torch.Tensor:          # (..., K, K) well-conditioned SPD
    A = torch.randn(*shape, shape[-1], generator=generator, dtype=F64)
    return A @ A.mT + shape[-1] * torch.eye(shape[-1], dtype=F64)


def _live(**over: Any) -> VFE3Config:
    r"""The launcher's live route at tiny scale with the frame-diagonal family on the kernel readout and a coordinate
    block MLP behind the LayerNorm (the family's envelope; the launcher's minimal agent and prototype MLP are outside
    it)."""
    values = dict(family=FAMILY, predictive_readout="kernel", block_mlp_mode="coordinate",
                  block_mlp_covariance="passthrough", decode_mode="diagonal_chunked", norm_type_block="layernorm")
    return live_config(**{**values, **over})


# 4. KL closed form against a dense float64 reference.

def test_kl_in_a_shared_frame_and_across_frames_matches_the_dense_gaussian_kl() -> None:
    g = torch.Generator().manual_seed(0)
    K, n = 6, 5
    U_q = torch.linalg.matrix_exp(0.4 * torch.randn(n, K, K, generator=g, dtype=F64))     # (n, K, K) frames
    U_p = torch.linalg.matrix_exp(0.4 * torch.randn(n, K, K, generator=g, dtype=F64))
    mu_q, mu_p = torch.randn(n, K, generator=g, dtype=F64), torch.randn(n, K, generator=g, dtype=F64)
    c_q = torch.rand(n, K, generator=g, dtype=F64) + 0.1
    c_p = torch.rand(n, K, generator=g, dtype=F64) + 0.1

    def dense(mu_a, c_a, U_a, mu_b, c_b, U_b):                                # (n,) KL of the local Gaussians
        S_a = U_a @ torch.diag_embed(c_a) @ U_a.mT
        S_b = U_b @ torch.diag_embed(c_b) @ U_b.mT
        return kl(FullGaussian(mu_a, S_a), FullGaussian(mu_b, S_b), kl_max=math.inf)

    across = frame_diagonal_kl(mu_q, c_q, U_q, mu_p, c_p, U_p)
    torch.testing.assert_close(across, dense(mu_q, c_q, U_q, mu_p, c_p, U_p), rtol=1e-11, atol=1e-12)
    shared = frame_diagonal_kl(mu_q, c_q, U_q, mu_p, c_p, U_q)
    torch.testing.assert_close(shared, dense(mu_q, c_q, U_q, mu_p, c_p, U_q), rtol=1e-11, atol=1e-12)
    # In a shared frame the family's own closed form on the reference moments (U^-1 mu, c) is the same KL, and the
    # per-coordinate terms sum to it.
    xi_q = torch.linalg.solve(U_q, mu_q.unsqueeze(-1)).squeeze(-1)
    xi_p = torch.linalg.solve(U_q, mu_p.unsqueeze(-1)).squeeze(-1)
    q, p = FrameDiagonalGaussian(xi_q, c_q), FrameDiagonalGaussian(xi_p, c_p)
    torch.testing.assert_close(kl(q, p, kl_max=math.inf, eps=1e-12), shared, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(q.renyi_per_coord(p, kl_max=math.inf, eps=1e-12).sum(-1), shared,
                               rtol=1e-11, atol=1e-12)
    # A transported key Omega_ij q_j (Omega_ij = U_i U_j^-1) has the receiver's frame: its KL is the reference one.
    omega = U_q @ torch.linalg.inv(U_p)
    transported = kl(FullGaussian(mu_q, U_q @ torch.diag_embed(c_q) @ U_q.mT),
                     FullGaussian((omega @ mu_p.unsqueeze(-1)).squeeze(-1),
                                  omega @ (U_p @ torch.diag_embed(c_p) @ U_p.mT) @ omega.mT), kl_max=math.inf)
    xi_j = torch.linalg.solve(U_p, mu_p.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(kl(q, FrameDiagonalGaussian(xi_j, c_p), kl_max=math.inf, eps=1e-12), transported,
                               rtol=1e-11, atol=1e-12)
    assert get_family(FAMILY) is FrameDiagonalGaussian and FrameDiagonalGaussian.frame_diagonal_covariance


# The readout and the router read the reference moments: equal to the full family evaluated on U diag(c) U^T.

def _ops_pair(H: int, d: int, residual: bool, floor_mode: str, seed: int) -> tuple:
    ops, mu, _, beta, frame, receive = _case(H, d, residual, (2,), F64, seed=seed)
    fd = make_relational_operators([d] * H, 11, readout_mode="predictive_content", attention_residual=residual,
                                   predictive_covariance="moment_matched", predictive_covariance_floor=1e-3,
                                   predictive_covariance_floor_mode=floor_mode, frame_diagonal=True).double()
    full = make_relational_operators([d] * H, 11, readout_mode="predictive_content", attention_residual=residual,
                                     predictive_covariance="moment_matched", predictive_covariance_floor=1e-3,
                                     predictive_covariance_floor_mode=floor_mode).double()
    state = ops.double().state_dict()
    fd.load_state_dict(state)
    full.load_state_dict(state)
    return fd, full, mu, beta, frame, receive


@pytest.mark.parametrize("floor_mode", ["prior", "relative", "max"])
@pytest.mark.parametrize("residual", [False, True])
def test_moment_matched_update_is_the_reference_diagonal_of_the_full_update(floor_mode: str, residual: bool) -> None:
    fd, full, mu, beta, frame, receive = _ops_pair(2, 3, residual, floor_mode, seed=3)
    g = torch.Generator().manual_seed(5)
    c = torch.rand(mu.shape, generator=g, dtype=F64) + 0.2                     # (2, N, K) reference variances
    c_p = torch.rand(mu.shape, generator=g, dtype=F64) + 0.2
    local = frame_diagonal_local_covariance(c, frame, diagonal=False)          # (2, N, K, K) U diag(c) U^T
    local_p = frame_diagonal_local_covariance(c_p, frame, diagonal=False)
    for maps in (None, receive):
        got = fd.predictive_covariance_update(mu, c, beta, frame, maps, floor_anchor=c_p)            # (2, N, K)
        dense = full.predictive_covariance_update(mu, local, beta, frame, maps, floor_anchor=local_p)  # (2, N, K, K)
        torch.testing.assert_close(got, reference_variances(mu, dense, frame), rtol=1e-10, atol=1e-12)
    # The update depends on the frames only through the means: the stored c is read as diag(c) in reference.
    other = CanonicalFrameContext(frame.forward @ frame.forward, frame.inverse @ frame.inverse, block_diagonal=True)
    xi = (frame.inverse @ mu.reshape(*mu.shape[:-1], 2, 3).unsqueeze(-1)).flatten(-3)
    moved_mu = (other.forward @ xi.reshape(*xi.shape[:-1], 2, 3).unsqueeze(-1)).flatten(-3)        # same xi, new U
    torch.testing.assert_close(fd.predictive_covariance_update(moved_mu, c, beta, other, receive, floor_anchor=c_p),
                               fd.predictive_covariance_update(mu, c, beta, frame, receive, floor_anchor=c_p),
                               rtol=1e-10, atol=1e-12)


def test_router_scores_the_reference_moments() -> None:
    fd, full, mu, beta, frame, _ = _ops_pair(2, 3, False, "prior", seed=7)
    c = torch.rand(mu.shape, generator=torch.Generator().manual_seed(1), dtype=F64) + 0.2
    local = frame_diagonal_local_covariance(c, frame, diagonal=False)
    got = relational_score(mu, c, mu, c, frame, fd)                             # (2, H, N, N)
    expected = relational_score(mu, local, mu, local, frame, full)
    torch.testing.assert_close(got, expected, rtol=1e-9, atol=1e-10)


# 2. Depth recursion at kappa(U) about 20 (audit 2026-10-17 item 3).

def _depth_traces(kappa: float, layers: int = 6) -> Dict[str, list]:
    r"""Max reference trace per layer for the local-diagonal, head-block and frame-diagonal families: identity content,
    uniform causal beta, no residual, prior floor, K = 8, H = 2, N = 8, float64 (the probe of the L2 test, with the
    frame-diagonal family started from the reference diagonal of the same initial Gaussian)."""
    g = torch.Generator().manual_seed(0)
    H, d, N = 2, 4, 8
    K = H * d
    frame = _conditioned_frames(N, H, d, kappa, g)
    mu = torch.randn(1, N, K, generator=g, dtype=F64)
    sigma0 = torch.rand(1, N, K, generator=g, dtype=F64) + 0.5
    causal = torch.ones(N, N, dtype=torch.bool).tril()
    beta = (causal / causal.sum(-1, keepdim=True)).to(F64).expand(1, H, N, N)
    eye = torch.eye(K, dtype=F64)
    receive = eye.reshape(K, H, d).permute(1, 0, 2).contiguous()               # (H, K, d) identity O'_h
    U_inv = _dense(frame.inverse)
    out: Dict[str, list] = {}
    for family in ("diagonal", "block", FAMILY):
        ops = make_relational_operators([d] * H, 11, readout_mode="predictive_content", attention_residual=False,
                                        predictive_covariance="moment_matched", predictive_covariance_floor=1e-3,
                                        block_covariance=family == "block",
                                        frame_diagonal=family == FAMILY).double()
        with torch.no_grad():
            ops.T.copy_(eye.reshape(H, d, K))                                   # identity T_h
        sigma = (torch.diag_embed(sigma0) if family == "block"
                 else reference_variances(mu, sigma0, frame) if family == FAMILY else sigma0)
        anchor, traces = sigma.clone(), []
        for _ in range(layers):
            with torch.no_grad():
                sigma = ops.predictive_covariance_update(mu, sigma, beta, frame, receive, floor_anchor=anchor)
            if family == FAMILY:
                traces.append(float(sigma.sum(-1).max()))                       # tr C = sum c
            else:
                local = sigma if family == "block" else torch.diag_embed(sigma)
                traces.append(float(torch.diagonal(U_inv @ local @ U_inv.mT, dim1=-2, dim2=-1).sum(-1).max()))
        out[family] = traces
    return out


def test_frame_diagonal_family_stays_bounded_with_depth_where_the_local_diagonal_family_grows() -> None:
    traces = _depth_traces(20.0)
    diagonal, block, frame_diagonal = traces["diagonal"], traces["block"], traces[FAMILY]
    assert all(t < 2.0 * frame_diagonal[0] for t in frame_diagonal)             # bounded, like the block family
    assert all(b < 2.0 * block[0] for b in block)
    assert frame_diagonal[-1] < 2.0 * block[-1]
    assert diagonal[-1] > 1e3 * frame_diagonal[-1]                              # the local diagonal inflates
    assert all(later > earlier for earlier, later in zip(diagonal, diagonal[1:]))
    # With identity content the trace of C^att is the routed average of the reference traces plus the routed spread
    # of the means and the floor, and the frame-diagonal projection keeps it (tr diag C = tr C); measured 3.67 -> 6.87
    # here, equal to the head-block family, against 114 -> 7.2e5 on the local diagonal family.
    assert frame_diagonal == pytest.approx(block, rel=1e-10)
    flat = _depth_traces(1.0)                                                   # kappa = 1: all three agree
    torch.testing.assert_close(torch.tensor(flat[FAMILY]), torch.tensor(flat["diagonal"]), rtol=1e-10, atol=0)


# The decoders and the block against the full family evaluated on U diag(c) U^T.

def _paired_models(over: Dict[str, Any], full_over: Dict[str, Any]) -> Tuple[VFEModel, VFEModel]:
    torch.manual_seed(0)
    fd = VFEModel(_live(n_layers=1, **over)).double()
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in fd.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=gen, dtype=p.dtype))
    full = VFEModel(_live(n_layers=1, **{**over, **full_over})).double()
    full.load_state_dict(fd.state_dict(), strict=True)
    return fd, full


def _frame_and_patch(fd: VFEModel, full: VFEModel, ids: torch.Tensor) -> CanonicalFrameContext:
    r"""The realized frame of ``ids``, and ``full``'s encode patched to the local covariance U diag(c) U^T."""
    beliefs = fd.prior_bank.encode(ids)
    beliefs = beliefs._replace(phi=fd._apply_pos_phi(beliefs.phi), right_phi=fd._pos_phi_right(beliefs.phi))
    _, frame, _ = fd._relational_frames(beliefs, None)
    original = full.prior_bank.encode

    def encode(token_ids: torch.Tensor) -> BeliefState:
        prior = original(token_ids)
        c = fd.prior_bank.encode(token_ids).sigma
        return prior._replace(sigma=frame_diagonal_local_covariance(c, frame, diagonal=False))
    full.prior_bank.encode = encode
    return frame


@pytest.mark.parametrize("decode_coordinates", ["local", "reference"])
@pytest.mark.parametrize("block_coordinates, norm_covariance, mlp_covariance", [
    ("local", "passthrough", "passthrough"),
    ("local", "pushforward", "passthrough"),
    ("local", "pushforward_moment", "passthrough"),
    ("local", "passthrough", "delta_full"),
    ("reference", "pushforward", "passthrough"),
    ("reference", "pushforward_moment", "passthrough"),
    ("reference", "passthrough", "delta_full"),
])
def test_block_and_decoder_equal_the_projection_of_the_full_family(
        decode_coordinates: str, block_coordinates: str, norm_covariance: str, mlp_covariance: str) -> None:
    r"""With the passthrough readout the stack carries Sigma = U diag(c) U^T unchanged, so after one block the frame-
    diagonal belief is the M-projection diag(U^-1 S U^-T) of the full family's S, and the diagonal KL decoder's softmax
    equals the full decoder's (exact up to the per-position constant in local coordinates)."""
    over = dict(predictive_covariance="passthrough", block_coordinates=block_coordinates,
                block_norm_covariance=norm_covariance, block_mlp_covariance=mlp_covariance,
                decode_coordinates=decode_coordinates)
    fd, full = _paired_models(over, dict(family="gaussian_full", decode_mode="full_chunked"))
    ids = torch.randint(0, fd.cfg.vocab_size, (2, 4), generator=torch.Generator().manual_seed(3))
    frame = _frame_and_patch(fd, full, ids)
    with torch.no_grad():
        q_fd, logits_fd = fd.forward_beliefs(ids, return_logits=True)
        q_full, logits_full = full.forward_beliefs(ids, return_logits=True)
    torch.testing.assert_close(q_fd.mu, q_full.mu, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(q_fd.sigma, reference_variances(q_full.mu, q_full.sigma, frame), rtol=1e-9, atol=1e-12)
    if norm_covariance == "passthrough" and mlp_covariance == "passthrough":
        torch.testing.assert_close(torch.log_softmax(logits_fd, -1), torch.log_softmax(logits_full, -1),
                                   rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("decode_coordinates", ["local", "reference"])
def test_the_constant_and_encoded_decode_covariances_are_family_members(decode_coordinates: str) -> None:
    for mode in ("constant", "encoded"):
        model = VFEModel(_live(decode_covariance=mode, decode_coordinates=decode_coordinates)).double()
        ids = torch.randint(0, model.cfg.vocab_size, (2, 4), generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            belief, logits = model.forward_beliefs(ids, return_logits=True)
            _, loss, _ = model(ids, ids)
        assert torch.isfinite(logits).all() and torch.isfinite(loss)
        covariance = model._decode_covariance(belief.mu, belief.sigma, model.prior_bank.encode(ids).sigma)
        assert covariance.shape == belief.sigma.shape                          # (B, N, K): reference variances
        if mode == "constant":
            assert bool((covariance == model.cfg.sigma_init).all())


# The self term and the canonical diagnostic read the reference moments.

def test_mstep_self_term_and_canonical_diagnostic_are_invariant_under_a_local_frame_change() -> None:
    import tests.test_audit_20261016_a1_passive_gauge_model as harness
    raw = harness._raw("frame_diagonal")
    cfg = harness._switched({**raw, "mstep_self_coupling_weight": 0.5, "diagnose_canonical_free_energy": True})
    model = harness._model(cfg)
    g = harness._gauge_draw(model)
    ids = torch.randint(0, cfg.vocab_size, (harness.B, harness.N), generator=torch.Generator().manual_seed(3))
    original = model.prior_bank.encode
    values = []
    for encode in (lambda token_ids: original(token_ids), harness._frame_change(model, g, frames=True)):
        model.prior_bank.encode = encode
        try:
            with torch.no_grad():
                _, loss, ce = model(ids, ids)
                d = model.diagnostics(ids)
        finally:
            del model.prior_bank.encode
        values.append((float(loss - ce), d["self_divergence"], d["total"], d["canonical_total_output"]))
    (self_term, self_div, total, total_out), moved = values
    assert self_term > 0.0 and math.isfinite(self_div) and math.isfinite(total)
    for a, b in zip(values[0], moved):
        assert abs(a - b) <= 1e-9 * max(abs(a), 1.0)


def test_the_projected_equivariance_witness_reads_the_stored_reference_variances() -> None:
    from vfe3.viz.extract import projected_gauge_equivariance
    torch.manual_seed(0)
    model = VFEModel(_live(predictive_covariance_floor_mode="relative")).double()
    ids = torch.randint(0, model.cfg.vocab_size, (1, 4), generator=torch.Generator().manual_seed(0))
    residuals = projected_gauge_equivariance(model, ids)["residuals"]
    assert {"readout_mean", "readout_covariance", "routing_score", "routing_weights"} <= set(residuals)
    assert max(max(values) for values in residuals.values()) < 1e-9


def test_diagnostics_are_finite_per_layer() -> None:
    model = VFEModel(_live(diagnose_canonical_free_energy=True, predictive_covariance_floor_mode="relative")).double()
    ids = torch.randint(0, model.cfg.vocab_size, (2, 4), generator=torch.Generator().manual_seed(0))
    per_layer = model.diagnostics_per_layer(ids)
    for key in ("belief_lam_min", "belief_lam_max", "self_divergence", "total"):
        assert all(math.isfinite(v) for v in per_layer[key]), key


# 3. Float64 central differences of the live route, and a few training steps.

@pytest.mark.parametrize("over", [
    dict(block_mlp_covariance="delta_full", predictive_covariance_floor_mode="relative"),
    dict(block_coordinates="reference", decode_coordinates="reference", block_norm_covariance="pushforward_moment",
         predictive_covariance_floor_mode="max"),
], ids=["local_delta_full_relative", "reference_pushforward_moment_max"])
def test_live_route_gradient_matches_central_differences_in_float64(over: Dict[str, Any]) -> None:
    r"""Autograd against a Richardson central difference along one random direction of every parameter.

    The M-step self term is left off: its gradient holds the state-dependent alpha fixed (the envelope argument of
    ``VFEModel.forward``) while its value alpha D omits R(alpha), so it is not the derivative of the logged loss on any
    family (measured on gaussian_diagonal at a relative gap of 0.8, the same as here).
    """
    torch.manual_seed(0)
    model = VFEModel(_live(**over)).double()
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=gen, dtype=p.dtype))
    ids = torch.randint(0, model.cfg.vocab_size, (2, 4), generator=torch.Generator().manual_seed(2))
    targets = torch.roll(ids, -1, dims=1)
    params = [p for p in model.parameters() if p.requires_grad]
    direction = [torch.randn(p.shape, generator=gen, dtype=p.dtype) for p in params]
    norm = math.sqrt(sum(float((v * v).sum()) for v in direction))
    direction = [v / norm for v in direction]
    model.zero_grad()
    _, loss, _ = model(ids, targets)
    loss.backward()
    analytic = sum(float((p.grad * v).sum()) for p, v in zip(params, direction) if p.grad is not None)

    def at(t: float) -> float:
        with torch.no_grad():
            for p, v in zip(params, direction):
                p.add_(t * v)
            try:
                return float(model(ids, targets)[1])
            finally:
                for p, v in zip(params, direction):
                    p.sub_(t * v)

    def central(h: float) -> float:
        return (at(h) - at(-h)) / (2 * h)

    h = 1e-4
    richardson = (4 * central(h / 2) - central(h)) / 3                          # O(h^4) (audit 2026-10-17 item 37)
    assert abs(richardson - analytic) <= 1e-6 * max(abs(richardson), abs(analytic)) + 1e-8
    assert abs(analytic) > 1e-6                                                 # the direction is not degenerate


@pytest.mark.parametrize("over", [
    dict(),
    dict(block_coordinates="reference", decode_coordinates="reference", block_norm_covariance="pushforward"),
    dict(gauge_transport="off", m_table_update_mode="fisher", untie_decode_bank=True),
], ids=["local", "reference", "identity_frames_fisher_table"])
def test_a_few_training_steps_keep_the_loss_finite(over: Dict[str, Any]) -> None:
    from vfe3.train import build_optimizer, train_step
    cfg = _live(**over)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    gen = torch.Generator().manual_seed(4)
    losses = []
    for _ in range(4):
        tokens = torch.randint(0, cfg.vocab_size, (2, 4), generator=gen)
        losses.append(float(train_step(model, optimizer, scheduler, tokens, torch.roll(tokens, -1, dims=1),
                                       grad_clip=1.0)))
    assert all(math.isfinite(x) for x in losses)
    with torch.no_grad():
        belief, _ = model.forward_beliefs(tokens)
    assert bool((belief.sigma > 0).all()) and belief.sigma.shape == belief.mu.shape


# Configuration envelope.

def test_the_family_is_accepted_and_serializes() -> None:
    cfg = _live()
    assert cfg.family == FAMILY and cfg.diagonal_covariance
    assert config_from_serialized(serialize_config(cfg), source="test").family == FAMILY
    with pytest.raises(ValueError, match="retired family='laplace_diagonal'"):
        config_from_serialized({**serialize_config(cfg), "family": "laplace_diagonal"}, source="test")


@pytest.mark.parametrize("over, message", [
    (dict(attention_mode="projected_gradient"), "projected_predictive"),
    (dict(predictive_readout="minimal_agent"), "predictive_readout='minimal_agent'"),
    (dict(content_link_chart="exponential", content_link_edges="lowrank"), "content_link_edges"),
    (dict(rope_mode="frame"), "rope_mode='frame'"),
    (dict(norm_type_block="mahalanobis"), "norm_type_block='mahalanobis'"),
    (dict(block_mlp_mode="prototype_agents"), "block_mlp_mode='prototype_agents'"),
    (dict(block_mlp_mode="gaussian_template"), "block_mlp_mode='gaussian_template'"),
    (dict(decode_mode="full_chunked"), "decode_mode='full_chunked'"),
    (dict(decode_mode="expected_likelihood_chunked"), "decode_mode='expected_likelihood_chunked'"),
    (dict(decode_mode="linear", decode_second_order=True), "decode_second_order"),
    (dict(m_table_update_mode="fisher", untie_decode_bank=True, pos_phi="learned"), "m_table_update_mode=.fisher. with pos_phi=.learned."),
])
def test_routes_outside_the_envelope_are_refused_with_the_reason(over: Dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _live(**over)


def test_the_gauge_switch_keeps_the_family_and_certifies_the_route() -> None:
    raw = config_values(attention_mode="projected_predictive", predictive_readout="kernel",
                        predictive_covariance="moment_matched", family=FAMILY, decode_mode="diagonal_chunked",
                        vocab_size=7, embed_dim=4, n_heads=2, max_seq_len=4, n_layers=2, block_mlp_expansion=2)
    overrides = fully_gauge_covariant_overrides(raw)
    assert "family" not in overrides and "decode_mode" not in overrides
    cfg = VFE3Config(**{**raw, **overrides})
    assert gauge_covariance_failures(cfg) == []
    report = _pure_path_report(cfg, [])
    assert report["gauge_flags"]["family_group_invariant"] is True and report["on_gauge_pure_path"] is True
    local = VFE3Config(**{**raw, **overrides, "block_coordinates": "local"})
    assert "block_norm_gauge_equivariant" in gauge_covariance_failures(local)


def test_local_block_coordinates_break_the_invariance_the_reference_block_keeps() -> None:
    r"""Control: the frame-diagonal family alone does not make the model covariant; the LayerNorm in local coordinates
    still moves the logits under the local frame change."""
    import tests.test_audit_20261016_a1_passive_gauge_model as harness
    raw = harness._raw("frame_diagonal")
    overrides = fully_gauge_covariant_overrides(raw)
    model = harness._model(harness._switched(raw))
    assert harness._invariance(model)[0] < 1e-9
    reverted = VFE3Config(**{**raw, **overrides, "block_coordinates": "local"})
    assert _pure_path_report(reverted, [])["on_gauge_pure_path"] is False
    assert harness._invariance(harness._model(reverted, state=model.state_dict()))[0] > 1e-3
