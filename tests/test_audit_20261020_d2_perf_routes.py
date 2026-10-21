r"""Audit 2026-10-20 lane D2 (performance): items 46, 50 and 51, each against the route it replaces as the oracle.

Item 46: under ``gaussian_frame_diagonal`` the diagonal comparison moments read the identity frame without multiplying
it, :math:`a = S\mu` and :math:`\operatorname{diag}(S\operatorname{diag}(c)S^\top) = (S\circ S)c`. Item 50:
``noisy_query`` and ``noisy_key`` on comparison-invariant RoPE take the diagonal route when :math:`P(G)` is
structurally diagonal. Item 51: the ``pure_gauge`` edge coefficient stays per sender. The oracle is the previous
route, selected by monkeypatching the route choice (items 46 and 50) or by broadcasting the per-sender coefficient to
the per-edge tensor the previous code built (item 51). Tiny CPU models only: B = 2, N = 5, three heads of width 4.
"""
import math
import warnings
from typing import Any, Callable, Dict, Tuple

import pytest
import torch

import vfe3.gradients.relational as gr
import vfe3.model.edge_links as el
import vfe3.relational_scoring as rs
from tests.profiles import tiny_config
from vfe3.model.model import VFEModel

F64 = torch.float64
B, N, VOCAB = 2, 5, 9
TOKENS = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 1]])
TARGETS = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 0, 2]])
REFERENCES = ("absolute", "row_min", "query_shifted", "query_shifted_v1")
PRIORS = ("causal", "causal_alibi")


def _model(dtype: torch.dtype = F64, **overrides: Any) -> VFEModel:
    r"""A one-layer kernel-router model, three heads of width 4 (a diagonal P(G) at rope_fraction 1)."""
    values: Dict[str, Any] = dict(vocab_size=VOCAB, embed_dim=12, n_heads=3, max_seq_len=N, n_layers=1,
                                  phi_scale=0.4, predictive_readout="kernel", rope_mode="comparison_invariant",
                                  rope_fraction=1.0, projected_factorization="gram", kl_max=12.0)
    values.update(overrides)
    torch.manual_seed(7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(tiny_config(**values))
    return model.to(dtype)


def _loss_and_grads(model: VFEModel) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    model.zero_grad(set_to_none=True)
    _, loss, _ = model(TOKENS, TARGETS)
    loss.backward()
    return loss.detach(), {name: p.grad.detach().clone() for name, p in model.named_parameters()
                           if p.grad is not None}


def _assert_same(new: Tuple[torch.Tensor, Dict[str, torch.Tensor]],
                 old: Tuple[torch.Tensor, Dict[str, torch.Tensor]], rtol: float, atol: float) -> float:
    r"""Loss and every parameter gradient agree; returns the largest gradient difference relative to its norm."""
    torch.testing.assert_close(new[0], old[0], rtol=rtol, atol=atol)
    assert new[1].keys() == old[1].keys()
    worst = 0.0
    for name in old[1]:
        torch.testing.assert_close(new[1][name], old[1][name], rtol=rtol, atol=atol, msg=name)
        scale = float(old[1][name].abs().max()) or 1.0
        worst = max(worst, float((new[1][name] - old[1][name]).abs().max()) / scale)
    return worst


def _counting(monkeypatch: pytest.MonkeyPatch, module: Any, name: str) -> Dict[str, int]:
    calls = {"n": 0}
    original = getattr(module, name)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        return original(*args, **kwargs)
    monkeypatch.setattr(module, name, wrapped)
    return calls


# ------------------------------------------------------------------------------------- item 46

def _moment_operands(per_head: bool, dtype: torch.dtype = F64) -> Tuple[torch.Tensor, ...]:
    r"""Means and variances (B, N, h, K), h = 1 (head-shared, expanded by the caller) or H, and maps (2, H, r, K)."""
    g = torch.Generator().manual_seed(11)
    H, r, K = 3, 4, 12
    mu = torch.randn(B, N, 1 if not per_head else H, K, generator=g, dtype=dtype)
    sigma = 0.2 + torch.rand(B, N, 1 if not per_head else H, K, generator=g, dtype=dtype)
    S = torch.randn(2, H, r, K, generator=g, dtype=dtype)
    return mu, sigma, S


@pytest.mark.parametrize("per_head", [False, True])
@pytest.mark.parametrize("roles", [1, 2])
@pytest.mark.parametrize("compact", [True, False])
@pytest.mark.parametrize("diagonal_only", [True, False])
def test_identity_frame_moments_match_the_materialized_identity(per_head: bool, roles: int, compact: bool,
                                                                diagonal_only: bool) -> None:
    mu0, sigma0, S0 = _moment_operands(per_head)
    S0 = S0[0] if roles == 1 else S0
    shape = (B, N, 3, 4, 4) if compact else (B, N, 1, 12, 12)                 # compact blocks or one dense block
    eye = torch.eye(shape[-1], dtype=F64).expand(shape)

    def run(identity: bool) -> Tuple[torch.Tensor, ...]:
        mu = mu0.detach().clone().requires_grad_(True)
        sigma = sigma0.detach().clone().requires_grad_(True)
        S = S0.detach().clone().requires_grad_(True)
        a, G = rs.projected_moments(mu.expand(B, N, 3, 12), sigma.expand(B, N, 3, 12), eye, S,
                                    diagonal_only=diagonal_only, identity_frame=identity)
        (a.sin().sum() + G.cos().sum()).backward()
        return a.detach(), G.detach(), mu.grad, sigma.grad, S.grad

    new, old = run(True), run(False)
    for x, y in zip(new, old):
        torch.testing.assert_close(x, y, rtol=1e-12, atol=1e-13)


def test_identity_frame_moments_in_float32_are_within_rounding() -> None:
    mu0, sigma0, S0 = _moment_operands(False, torch.float32)
    mu0, sigma0 = mu0.expand(B, N, 3, 12), sigma0.expand(B, N, 3, 12)
    eye = torch.eye(4).expand(B, N, 3, 4, 4)
    a_new, g_new = rs.projected_moments(mu0, sigma0, eye, S0, diagonal_only=True, identity_frame=True)
    a_old, g_old = rs.projected_moments(mu0, sigma0, eye, S0, diagonal_only=True)
    torch.testing.assert_close(a_new, a_old, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(g_new, g_old, rtol=1e-5, atol=1e-6)


def test_identity_frame_keeps_the_zero_and_invalid_variance_conventions() -> None:
    mu0, sigma0, S0 = _moment_operands(False)
    sigma0 = sigma0.clone()
    sigma0[0, 1, 0, 3] = 0.0                                                    # a zero variance: zero gradient
    sigma0[1, 2, 0, 5] = -0.5                                                   # a negative one: NaN token
    sigma0[1, 4, 0, 0] = float("nan")
    eye = torch.eye(4, dtype=F64).expand(B, N, 3, 4, 4)
    results = []
    for identity in (True, False):
        sigma = sigma0.detach().clone().requires_grad_(True)
        a, g = rs.projected_moments(mu0.expand(B, N, 3, 12), sigma.expand(B, N, 3, 12), eye, S0[0],
                                    diagonal_only=True, identity_frame=identity)
        g.nan_to_num(0.0).sum().backward()
        results.append((g.detach(), sigma.grad))
    (g_new, grad_new), (g_old, grad_old) = results
    assert torch.equal(g_new.isnan(), g_old.isnan())
    assert bool(g_new[1, 2].isnan().all()) and bool(g_new[1, 4].isnan().all())
    finite = ~g_old.isnan()
    torch.testing.assert_close(g_new[finite], g_old[finite], rtol=1e-12, atol=1e-13)
    assert float(grad_new[0, 1, 0, 3]) == 0.0 == float(grad_old[0, 1, 0, 3])
    healthy = torch.ones_like(grad_old, dtype=torch.bool)
    healthy[1, 2], healthy[1, 4] = False, False
    torch.testing.assert_close(grad_new[healthy], grad_old[healthy], rtol=1e-12, atol=1e-13)


def _old_identity_route(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""The previous frame-diagonal route: the moments multiply the materialized identity frame."""
    original = rs.projected_moments

    def without_flag(*args: Any, identity_frame: bool = False, **kwargs: Any) -> Any:
        return original(*args, **kwargs)
    monkeypatch.setattr(gr, "projected_moments", without_flag)


@pytest.mark.parametrize("reference", REFERENCES)
@pytest.mark.parametrize("prior", PRIORS)
@pytest.mark.parametrize("rope_fraction", [1.0, 0.5])
def test_frame_diagonal_model_matches_the_materialized_identity_route(monkeypatch: pytest.MonkeyPatch,
                                                                      reference: str, prior: str,
                                                                      rope_fraction: float) -> None:
    overrides = dict(family="gaussian_frame_diagonal", routing_kl_max_reference=reference,
                     beta_attention_prior=prior, rope_fraction=rope_fraction)
    fast = _counting(monkeypatch, rs, "_identity_frame_diagonal")
    new = _loss_and_grads(_model(**overrides))
    assert (fast["n"] > 0) == (rope_fraction == 1.0)                          # 0.5: the dense P(G) route
    with monkeypatch.context() as patch:
        _old_identity_route(patch)
        old = _loss_and_grads(_model(**overrides))
    _assert_same(new, old, rtol=1e-11, atol=1e-12)


def test_frame_diagonal_model_in_float32_is_within_rounding(monkeypatch: pytest.MonkeyPatch) -> None:
    overrides = dict(family="gaussian_frame_diagonal", routing_kl_max_reference="query_shifted",
                     beta_attention_prior="causal_alibi")
    new = _loss_and_grads(_model(torch.float32, **overrides))
    _old_identity_route(monkeypatch)
    old = _loss_and_grads(_model(torch.float32, **overrides))
    _assert_same(new, old, rtol=2e-4, atol=2e-5)


def test_frame_diagonal_family_never_reaches_the_fused_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rs, "_diagonal_moments_triton_active", lambda *tensors: True)   # CUDA emulated
    fused = _counting(monkeypatch, rs, "_fused_diagonal_moments")
    rs.reset_diagonal_moments_triton_calls()
    _loss_and_grads(_model(family="gaussian_frame_diagonal"))
    assert fused["n"] == 0 and rs.realized_diagonal_moments_kernel() == "torch"


# ------------------------------------------------------------------------------------- item 50

def _old_dense_noisy_route(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""The previous route choice: only the Gaussian KL takes the diagonal route."""
    from vfe3.geometry.rope import rope_commutant_is_diagonal

    def gaussian_kl_only(operators: Any, r: int, n_pairs: int) -> bool:
        return (rope_commutant_is_diagonal(r, n_pairs) and operators.routing == "gaussian_kl"
                and not rs.projected_condition_recovery())
    monkeypatch.setattr(gr, "_diagonal_comparison_route", gaussian_kl_only)


def _perturb_key_noise(model: VFEModel) -> None:
    r"""Distinct learned noises per head, so a head mix-up cannot pass."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "key_log_noise" in name:
                p.copy_(torch.linspace(-0.7, 0.4, p.numel(), dtype=p.dtype).reshape(p.shape))


@pytest.mark.parametrize("routing", ["noisy_query", "noisy_key"])
@pytest.mark.parametrize("reference", REFERENCES)
@pytest.mark.parametrize("prior", PRIORS)
def test_noisy_costs_on_the_diagonal_route_match_the_dense_route(monkeypatch: pytest.MonkeyPatch, routing: str,
                                                                 reference: str, prior: str) -> None:
    overrides = dict(relational_routing=routing, routing_kl_max_reference=reference, beta_attention_prior=prior)
    target = "_projected_diagonal_noisy_query" if routing == "noisy_query" else "_projected_diagonal_kl"
    diagonal = _counting(monkeypatch, rs, target)
    model = _model(**overrides)
    _perturb_key_noise(model)
    new = _loss_and_grads(model)
    assert diagonal["n"] > 0
    with monkeypatch.context() as patch:
        _old_dense_noisy_route(patch)
        dense = _counting(patch, rs, target)
        model = _model(**overrides)
        _perturb_key_noise(model)
        old = _loss_and_grads(model)
        assert dense["n"] == 0
    _assert_same(new, old, rtol=1e-10, atol=1e-11)
    if routing == "noisy_key":
        assert any("key_log_noise" in name and float(g.abs().max()) > 0 for name, g in new[1].items())


@pytest.mark.parametrize("routing", ["noisy_query", "noisy_key"])
def test_noisy_costs_in_float32_are_within_rounding(monkeypatch: pytest.MonkeyPatch, routing: str) -> None:
    overrides = dict(relational_routing=routing, routing_kl_max_reference="query_shifted",
                     beta_attention_prior="causal_alibi")
    new = _loss_and_grads(_model(torch.float32, **overrides))
    _old_dense_noisy_route(monkeypatch)
    old = _loss_and_grads(_model(torch.float32, **overrides))
    _assert_same(new, old, rtol=2e-4, atol=2e-5)


def _score_call(model: VFEModel) -> Tuple[tuple, Dict[str, Any]]:
    r"""The arguments of the model's first relational_score call."""
    saved: Dict[str, Any] = {}
    original = gr.relational_score

    class _Stop(Exception):
        pass

    def hook(*args: Any, **kwargs: Any) -> Any:
        saved["args"], saved["kwargs"] = args, kwargs
        raise _Stop
    gr.relational_score = hook
    try:
        model(TOKENS, TARGETS)
    except _Stop:
        pass
    finally:
        gr.relational_score = original
    return saved["args"], saved["kwargs"]


@pytest.mark.parametrize("routing", ["noisy_query", "noisy_key"])
@pytest.mark.parametrize("reference", ["query_shifted", "query_shifted_v1", "absolute"])
def test_noisy_score_key_volume_and_capture_match_the_dense_route(monkeypatch: pytest.MonkeyPatch, routing: str,
                                                                  reference: str) -> None:
    model = _model(relational_routing=routing, routing_kl_max_reference=reference)
    _perturb_key_noise(model)
    args, _ = _score_call(model)

    def score() -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, torch.Tensor]]:
        capture: Dict[str, Any] = {}
        aux: Dict[str, torch.Tensor] = {}
        s = gr.relational_score(*args[:6], capture, routing_aux=aux)
        return s.detach(), capture, aux

    new = score()
    _old_dense_noisy_route(monkeypatch)
    old = score()
    torch.testing.assert_close(new[0], old[0], rtol=1e-11, atol=1e-12)
    assert new[2].keys() == old[2].keys() == ({"key_log_volume"} if reference == "query_shifted" else set())
    for key in new[2]:
        torch.testing.assert_close(new[2][key], old[2][key], rtol=1e-12, atol=1e-12)
    assert "score_term_medians_pre" not in new[1] and "score_term_medians_pre" not in old[1]
    for key in ("projected_covariance_pre", "projected_key_covariance_pre", "score_query_constant_pre"):
        assert (key in new[1]) == (key in old[1])
        if key in new[1]:
            torch.testing.assert_close(new[1][key], old[1][key], rtol=1e-12, atol=1e-12)


def test_diagonal_noisy_query_tiles_match_one_tile(monkeypatch: pytest.MonkeyPatch) -> None:
    g = torch.Generator().manual_seed(3)
    aq, ak = torch.randn(2, 7, 3, 4, generator=g, dtype=F64), torch.randn(2, 6, 3, 4, generator=g, dtype=F64)
    gq, gk = (0.1 + torch.rand(2, 7, 3, 4, generator=g, dtype=F64) for _ in range(2))
    gk = gk[:, :6]
    leaves = [x.clone().requires_grad_(True) for x in (aq, gq, ak, gk)]
    whole = rs._projected_diagonal_noisy_query(leaves[0], leaves[1], leaves[2], leaves[3])
    whole.sum().backward()
    monkeypatch.setattr(rs, "_PROJECTION_TILE_BYTES", 1)                       # one query row per checkpointed tile
    tiled_leaves = [x.clone().requires_grad_(True) for x in (aq, gq, ak, gk)]
    tiled = rs._projected_diagonal_noisy_query(*tiled_leaves)
    tiled.sum().backward()
    torch.testing.assert_close(tiled, whole, rtol=0, atol=0)
    for x, y in zip(tiled_leaves, leaves):
        torch.testing.assert_close(x.grad, y.grad, rtol=1e-14, atol=1e-14)


def test_diagonal_noisy_query_refuses_a_nonpositive_variance() -> None:
    a = torch.zeros(1, 3, 1, 2, dtype=F64)
    good = torch.ones(1, 3, 1, 2, dtype=F64)
    bad = good.clone()
    bad[0, 1, 0, 0] = 0.0
    with pytest.raises(ValueError, match="not positive"):
        rs._projected_diagonal_noisy_query(a, bad, a, good)


def test_noisy_routes_activate_the_moments_kernel_field_on_the_diagonal_family() -> None:
    base = dict(attention_mode="projected_predictive", predictive_readout="kernel",
                rope_mode="comparison_invariant", rope_fraction=1.0, projected_factorization="gram")
    for routing in ("gaussian_kl", "noisy_query", "noisy_key"):
        assert "relational_diag_moments_kernel" not in tiny_config(**base, relational_routing=routing,
                                                                  embed_dim=8).inactive_fields()
    assert "relational_diag_moments_kernel" in tiny_config(**base, relational_routing="mean_distance",
                                                           embed_dim=8).inactive_fields()


# ------------------------------------------------------------------------------------- item 51

def _per_edge_oracle(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""The previous context: the sender coefficient broadcast to every edge and zeroed on unrouted pairs."""
    original = el.edge_coefficients

    def broadcast(A: torch.Tensor, B_: torch.Tensor, beta: torch.Tensor, mode: str) -> Any:
        Phi, g_phi, mask = original(A, B_, beta, mode)
        if mode == "pure_gauge":
            Phi = torch.where(mask[..., None, None], Phi.unsqueeze(-4), Phi.new_zeros(()))   # (..., N, N, r, r)
        return Phi, g_phi, mask
    monkeypatch.setattr(el, "edge_coefficients", broadcast)


def _edge_model(dtype: torch.dtype = F64, **overrides: Any) -> VFEModel:
    model = _model(dtype, content_link_chart="exponential", content_link_edges="pure_gauge",
                   content_link_edge_rank=2, rope_mode="none", **overrides)
    with torch.no_grad():                                                       # nonzero edge factors
        for name, p in model.named_parameters():
            if "edge_" in name:
                p.copy_(0.3 * torch.randn(p.shape, generator=torch.Generator().manual_seed(1), dtype=F64).to(dtype))
    return model


@pytest.mark.parametrize("family", ["gaussian_diagonal", "gaussian_full"])
@pytest.mark.parametrize("prior", PRIORS)
@pytest.mark.parametrize("reference", ["absolute", "query_shifted"])
def test_pure_gauge_per_sender_coefficients_match_the_per_edge_tensor(monkeypatch: pytest.MonkeyPatch, family: str,
                                                                      prior: str, reference: str) -> None:
    overrides = dict(family=family, beta_attention_prior=prior, routing_kl_max_reference=reference)
    sender = _counting(monkeypatch, el, "_sender_covariance_factors")
    new = _loss_and_grads(_edge_model(**overrides))
    assert sender["n"] > 0
    with monkeypatch.context() as patch:
        _per_edge_oracle(patch)
        old = _loss_and_grads(_edge_model(**overrides))
    _assert_same(new, old, rtol=1e-11, atol=1e-12)
    assert any("edge_" in name and float(g.abs().max()) > 0 for name, g in new[1].items())


def test_pure_gauge_in_float32_is_within_rounding(monkeypatch: pytest.MonkeyPatch) -> None:
    new = _loss_and_grads(_edge_model(torch.float32, beta_attention_prior="causal_alibi"))
    _per_edge_oracle(monkeypatch)
    old = _loss_and_grads(_edge_model(torch.float32, beta_attention_prior="causal_alibi"))
    _assert_same(new, old, rtol=2e-4, atol=2e-5)


def _edge_operands(nonfinite_sender: int) -> Tuple[el.EdgeContext, torch.Tensor, torch.Tensor, tuple]:
    r"""A pure-gauge context under a causal mask, B = 2, N = 5, H = 2, d = 2, r = 2, optionally with a NaN sender."""
    g = torch.Generator().manual_seed(4)
    H, d, r = 2, 2, 2
    K = H * d
    xi = torch.randn(B, N, K, generator=g, dtype=F64)
    logits = torch.randn(B, H, N, N, generator=g, dtype=F64)
    causal = torch.ones(N, N, dtype=torch.bool).tril()
    beta = logits.masked_fill(~causal, -math.inf).softmax(-1)                   # (B, H, N, N)
    T = torch.eye(K, dtype=F64) + 0.1 * torch.randn(K, K, generator=g, dtype=F64)
    R = torch.linalg.inv(T)
    F = torch.linalg.matrix_exp(0.1 * torch.randn(K, K, generator=g, dtype=F64))
    ctx = el.build_edge_context(xi, beta, T, R, F @ R, F, "pure_gauge",
                                0.3 * torch.randn(K * r, K, generator=g, dtype=F64),
                                0.3 * torch.randn(K, r, generator=g, dtype=F64))
    if nonfinite_sender >= 0:
        Phi = ctx.Phi.clone()
        Phi[:, nonfinite_sender, 0, 1] = float("nan")
        ctx = ctx._replace(Phi=Phi)
    eye_h = torch.eye(H, dtype=F64)
    c = torch.einsum("...hij,...gij->...hgij", beta, beta)
    c = c * (1.0 - eye_h)[..., None, None] + torch.einsum("...hij,hg->...hgij", beta, eye_h)
    C = torch.randn(B, N, K, K, generator=g, dtype=F64)
    C = C @ C.transpose(-1, -2)
    T_heads = T.reshape(H, d, K)
    source = torch.einsum("hsk,...nkm,gtm->...nhgst", T_heads, C, T_heads)
    values = torch.einsum("hsk,...nk->...nhs", T_heads, xi - xi[..., :1, :])
    mean = torch.einsum("...hij,...jhs->...ihs", beta, values)
    z = torch.randn(B, N, K, generator=g, dtype=F64)
    return ctx, beta, z, (c, source, values, mean)


def _broadcast(ctx: el.EdgeContext) -> el.EdgeContext:
    return ctx._replace(Phi=torch.where(ctx.mask[..., None, None], ctx.Phi.unsqueeze(-4), ctx.Phi.new_zeros(())))


@pytest.mark.parametrize("nonfinite_sender", [-1, 0, N - 1])
def test_edge_functions_match_the_per_edge_form_including_nonfinite_senders(nonfinite_sender: int) -> None:
    ctx, beta, z, (c, source, values, mean) = _edge_operands(nonfinite_sender)
    assert ctx.Phi.shape == (B, N, 2, 2)
    old_ctx = _broadcast(ctx)
    pairs = [
        (el.edge_reference_message(ctx, beta, z), el.edge_reference_message(old_ctx, beta, z)),
        *zip(el.edge_covariance_factors(ctx, beta, c, source, values, mean),
             el.edge_covariance_factors(old_ctx, beta, c, source, values, mean)),
        (el.edge_self_coefficients(ctx), el.edge_self_coefficients(old_ctx)),
        (el.edge_part_norms(ctx), el.edge_part_norms(old_ctx)),
    ]
    for new, old in pairs:
        assert new.shape == old.shape
        if new.dim() >= 3:                                                      # per receiver (A_hat is shared)
            # The same receivers are finite; a receiver routing to a non-finite sender is all NaN on the new route.
            assert torch.equal(torch.isfinite(new).flatten(2).all(-1), torch.isfinite(old).flatten(2).all(-1))
        finite = torch.isfinite(old) & torch.isfinite(new)
        torch.testing.assert_close(new[finite], old[finite], rtol=1e-12, atol=1e-13)
    if nonfinite_sender == N - 1:
        # Under the causal mask only the last receiver routes to the last sender: every other receiver is finite.
        message = pairs[0][0]
        assert bool(torch.isfinite(message[:, :-1]).all()) and not bool(torch.isfinite(message[:, -1]).all())
        assert bool(torch.isfinite(pairs[0][1][:, :-1]).all())


def test_edge_gradients_on_zero_weights_match_the_per_edge_form() -> None:
    ctx, beta, z, (c, source, values, mean) = _edge_operands(-1)
    grads = []
    for build in (lambda x: x, _broadcast):
        b = beta.detach().clone().requires_grad_(True)
        cc = torch.einsum("...hij,...gij->...hgij", b, b)
        eye_h = torch.eye(2, dtype=F64)
        cc = cc * (1.0 - eye_h)[..., None, None] + torch.einsum("...hij,hg->...hgij", b, eye_h)
        Phi = ctx.Phi.detach().clone().requires_grad_(True)
        p = ctx.p.detach().clone().requires_grad_(True)
        local = build(ctx._replace(Phi=Phi, p=p))
        message = el.edge_reference_message(local, b, z)
        _, X_hat, Y = el.edge_covariance_factors(local, b, cc, source, values, mean)
        (message.sin().sum() + X_hat.cos().sum() + Y.square().sum()).backward()
        grads.append((b.grad, Phi.grad, p.grad))
    for new, old in zip(*grads):
        torch.testing.assert_close(new, old, rtol=1e-11, atol=1e-12)
