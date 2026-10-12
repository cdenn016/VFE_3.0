"""Remediation of the 2026-10-08 audit critical and high items (docs/audits/audit-2026-10-08.md).

Tiny models on CPU (V=11, K=4, H=2, N=6), reusing the configuration helpers of the 2026-10-07 tests.
"""
import types
from pathlib import Path
from typing import Iterator

import pytest
import torch

import vfe3.relational_scoring as scoring
from tests.test_audit_20261007_opus_fixes import B, H, K, N, V, _cfg, _minimal_agent, _rope_cfg
from vfe3.contracts import CanonicalFrameContext
from vfe3.model.model import VFEModel


@pytest.fixture(autouse=True)
def _restore_scorer_policies() -> Iterator[None]:
    saved = (scoring._CONDITION_RECOVERY, scoring._FACTOR_ROUTE, scoring._GRAM_CONDITION_TOL,
             scoring._FACTOR_CONDITION_TOL, scoring._RECOVERY_PRECISION)
    yield
    (scoring._CONDITION_RECOVERY, scoring._FACTOR_ROUTE, scoring._GRAM_CONDITION_TOL,
     scoring._FACTOR_CONDITION_TOL, scoring._RECOVERY_PRECISION) = saved


def _identity_frame(dtype: torch.dtype) -> CanonicalFrameContext:
    d = K // H
    eye = torch.eye(d, dtype=dtype).expand(1, N, H, d, d).clone()
    return CanonicalFrameContext(forward=eye, inverse=eye.clone(), block_diagonal=True)


def _rank_ill_maps(dtype: torch.dtype) -> tuple:
    r"""Per-head (S_q, S_k), each (H, 2, K); S_q has singular values (1, 1e-4), so its float32 Gram loses SPD."""
    g = torch.Generator().manual_seed(4)
    d = K // H
    Q = torch.linalg.qr(torch.randn(H, d, d, generator=g, dtype=torch.float64))[0]
    W = torch.linalg.qr(torch.randn(H, K, d, generator=g, dtype=torch.float64))[0]   # orthonormal columns
    S_q = Q @ torch.diag(torch.tensor([1.0, 1e-4], dtype=torch.float64)) @ W.transpose(-1, -2)
    S_k = torch.randn(H, d, K, generator=g, dtype=torch.float64)
    return S_q.to(dtype), S_k.to(dtype)


@pytest.mark.parametrize("route", ["gram", "cholesky_qr2"])
@pytest.mark.parametrize("precision, condition", [("fp64", True), ("fp32", False)])   # fp32/off: the launcher
def test_comparison_rope_recovers_a_failed_float32_factor_from_the_rotated_source(
        route: str, precision: str, condition: bool) -> None:
    """P1-2: the rotated source tuple R_i S rebuilds the rotated Gram instead of raising."""
    import vfe3.gradients.relational as rel

    model = VFEModel(_rope_cfg("comparison"))
    ops = model.relational_operators
    g = torch.Generator().manual_seed(1)
    mu = torch.randn(1, N, K, generator=g, dtype=torch.float64)
    sigma = torch.ones(1, N, K, dtype=torch.float64)

    def cost(dtype: torch.dtype) -> torch.Tensor:
        ops.comparison_runs = lambda: [_rank_ill_maps(dtype)]
        return rel.relational_score(mu.to(dtype), sigma.to(dtype), mu.to(dtype), sigma.to(dtype),
                                    _identity_frame(dtype), ops)

    reference = cost(torch.float64)
    scoring.set_projected_factor_route(route)
    scoring.set_projected_recovery_precision(precision)
    scoring.set_projected_condition_recovery(condition)
    scoring.reset_source_recovery_elements()
    recovered = cost(torch.float32)
    assert scoring.source_recovery_elements() > 0
    assert torch.isfinite(recovered).all()
    if condition:
        # Without the certificate only outright failures are rebuilt (as on the unrotated route), so the
        # accuracy bound needs it; the launcher setting checks the no-raise recovery only.
        assert torch.allclose(recovered.double(), reference, rtol=1e-3, atol=1e-3)


def test_rotated_source_root_reproduces_the_rotated_gram() -> None:
    """P1-2: X_i = R_i S U^{-1} Sigma^{1/2} satisfies X X^T = R_i G R_i^T."""
    from vfe3.geometry.rope import build_rope_rotation

    S, _ = _rank_ill_maps(torch.float64)
    g = torch.Generator().manual_seed(2)
    sigma = 0.5 + torch.rand(1, N, H, K, generator=g, dtype=torch.float64)
    inverse = torch.eye(K, dtype=torch.float64).expand(1, N, H, K, K)
    R = build_rope_rotation(torch.arange(N), [2], base=100.0, fraction=1.0, dtype=torch.float64)   # (N, 2, 2)
    G = S @ torch.diag_embed(sigma) @ S.transpose(-1, -2)                                         # (1, N, H, 2, 2)
    G_rot = R[:, None] @ G @ R[:, None].transpose(-1, -2)
    S_rot = torch.einsum("nst,htk->nhsk", R, S)
    mask = torch.ones(1, N, H, dtype=torch.bool)
    root = scoring._source_root(G_rot, (sigma, inverse, S_rot), mask)                             # (n, 2, K)
    assert torch.allclose(root @ root.transpose(-1, -2), G_rot[mask], atol=1e-12)


def test_eval_forward_replays_at_a_stricter_policy() -> None:
    """P0-1: an evaluation forward that raises under 'high' completes at 'highest'."""
    from vfe3.train import _eval_forward_with_precision_retry

    seen = []

    def forward(*_args, **_kwargs):
        seen.append(torch.get_float32_matmul_precision())
        if seen[-1] != "highest":
            raise ValueError("projected covariance is not positive definite")
        return "ok"

    class Fake:
        cfg = types.SimpleNamespace(matmul_precision_retry=True)

        def __call__(self, *args, **kwargs):
            return forward(*args, **kwargs)

    model = Fake()
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        assert _eval_forward_with_precision_retry(model) == "ok"
        assert seen == ["high", "highest"]
        assert torch.get_float32_matmul_precision() == "high"
        model.cfg.matmul_precision_retry = False
        with pytest.raises(ValueError):
            _eval_forward_with_precision_retry(model)
    finally:
        torch.set_float32_matmul_precision(previous)


def test_blocked_metrics_publish_keeps_the_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """P1-6: a PermissionError from a reader lock does not raise, and the next publish carries the row."""
    import vfe3.run_artifacts as run_artifacts

    real = run_artifacts._atomic_replace
    calls = []

    def blocked_once(final, tmp):
        calls.append(final)
        if len(calls) == 1:
            tmp.unlink()
            raise PermissionError("locked by a reader")
        return real(final, tmp)

    monkeypatch.setattr(run_artifacts, "_atomic_replace", blocked_once)
    artifacts = types.SimpleNamespace(csv_path=tmp_path / "metrics.csv", history=[], _fieldnames=None)
    run_artifacts.RunArtifacts.log_metrics(artifacts, {"step": 1, "loss": 2.0})
    assert not artifacts.csv_path.exists() and len(artifacts.history) == 1
    run_artifacts.RunArtifacts.log_metrics(artifacts, {"step": 2, "loss": 1.5})
    assert artifacts.csv_path.read_text(encoding="utf-8").splitlines() == ["step,loss", "1,2.0", "2,1.5"]


def _readout_beta(reference: str, kl_max: float) -> tuple:
    import vfe3.gradients.relational as rel

    model = VFEModel(_rope_cfg("comparison", routing_kl_max_reference=reference)).double()
    ops = model.relational_operators
    assert ops.kl_max_reference == reference
    g = torch.Generator().manual_seed(3)
    mu = (3.0 * torch.randn(1, N, K, generator=g, dtype=torch.float64)).requires_grad_(True)
    sigma = torch.ones(1, N, K, dtype=torch.float64)
    causal = torch.full((N, N), float("-inf"), dtype=torch.float64).triu(1)
    capture = {}
    out = rel.relational_predictive_readout(mu, sigma, _identity_frame(torch.float64), ops, tau=1.0,
                                            kl_max=kl_max, log_prior=causal, capture=capture)
    (grad,) = torch.autograd.grad(out.square().sum(), mu)
    return capture["beta_pre"], capture["score_energy_pre"], grad


def test_row_min_clamp_keeps_the_cost_order_of_a_saturated_row() -> None:
    """P1-4: with every cost at or above kl_max the absolute clamp routes uniformly with zero routing
    gradient; the row_min reference keeps the cost order and equals the exact softmax below the cap."""
    beta_exact, raw, _ = _readout_beta("absolute", kl_max=float("inf"))
    causal = torch.ones(N, N, dtype=torch.bool).tril()
    floor = float(raw[..., causal].min())
    assert floor > 0
    beta_abs, _, _ = _readout_beta("absolute", kl_max=floor)
    uniform = causal / causal.sum(-1, keepdim=True)
    assert torch.allclose(beta_abs[0], uniform.expand(H, N, N).to(beta_abs.dtype))
    beta_row, _, _ = _readout_beta("row_min", kl_max=floor)
    assert not torch.allclose(beta_row[0, :, 1:], uniform[1:].expand(H, N - 1, N).to(beta_row.dtype))
    beta_row_open, _, _ = _readout_beta("row_min", kl_max=1e9)
    assert torch.allclose(beta_row_open, beta_exact, atol=1e-12)


def test_row_min_is_inactive_off_the_kernel_router() -> None:
    cfg = _cfg(_minimal_agent(routing_kl_max_reference="row_min"))
    assert "routing_kl_max_reference" in cfg.inactive_fields()
    with pytest.raises(ValueError, match="routing_kl_max_reference"):
        _cfg(_minimal_agent(predictive_readout="kernel", block_mlp_mode="coordinate",
                            routing_kl_max_reference="row"))


def test_routing_decomposition_refuses_comparison_rope() -> None:
    """P1-3: M4 does not decompose the unrotated router of a comparison-RoPE model."""
    from vfe3.posthoc.routing import not_applicable_reason

    assert "comparison rope_mode" in not_applicable_reason(VFEModel(_rope_cfg("comparison")))
    assert not_applicable_reason(VFEModel(_rope_cfg("frame"))) is None


def test_paper_arms_carry_the_manuscript_m2_rows() -> None:
    """P1-8: the frozen-frames falsifier and three more M2 rows are trainable arms."""
    import paper_arms

    for label, field, value in (("Frames frozen at their random initialization", "gauge_transport", "frozen"),
                                ("Affine content chart", "content_link_chart", "affine"),
                                ("First-order content chart", "content_link_chart", "first_order"),
                                ("Covariance passthrough instead of moment matching",
                                 "predictive_covariance", "passthrough")):
        arm = paper_arms.ARMS[label]
        assert arm["vfe"] == {field: value}
        assert arm["inapplicable"](types.SimpleNamespace(**{field: value})) is not None
