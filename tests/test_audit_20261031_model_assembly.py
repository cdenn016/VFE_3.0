r"""Regression tests of the October 31, 2026 remediation package ``model-assembly``.

The CPU tests use tiny models only (K <= 8, two heads, at most three layers, vocabulary <= 9; the fused-decode
interpreter case scores 13 positions); the one CUDA test, which skips outside the CUDA lane, mirrors the shapes of
the existing fused-decode CUDA test. Each test names the finding it pins: the canonical free-energy diagnostic on the frame-diagonal family under comparison RoPE (A017) and its clamp
share (A018), ``readout_state`` on groups without equal compact head blocks (A053), the private diagnostics snapshot and
the diagonal-route capture layout (A142), the M-step self term with an infinite clamp (A080) and its logged value
(A081), the process-global numeric policy (A118), the diagonal-route eigenvalue statistics and the content-frame
rebuilds (A157), the M-step capture's frame reuse and the handoff identity blends (A183), the canonical-frame block
MLP's dense frame context (A196), the frozen-temperature specialization of the fused KL decode CE (A221), and the
``generate`` memory estimate and the capture contract (A291).
"""
from __future__ import annotations

import gc
import json
import math
import warnings
from typing import Any, Dict, List, Tuple

import pytest
import torch

from tests.profiles import reference_k8_config, tiny_config
from vfe3.config import VFE3Config
from vfe3.contracts import CanonicalFrameContext, MStepCapture
from vfe3.families import get_family
from vfe3.families.frame_gaussian import frame_diagonal_kl
from vfe3.free_energy import attention_tau, canonical_routing_rows, self_divergence_for_alpha
from vfe3.model.block import _as_coeff
from vfe3.model.model import VFEModel

F64 = torch.float64


@pytest.fixture(autouse=True)
def _quiet_two_threads():
    torch.set_num_threads(2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _ids(vocab: int, B: int, N: int, seed: int = 0) -> torch.Tensor:
    return torch.randint(0, vocab, (B, N), generator=torch.Generator().manual_seed(seed))


# The diagonal comparison route of the live recipe at K = 8: comparison_invariant RoPE at rope_fraction 1 (every
# coordinate of a width-4 head rotates, so P(G) is diagonal), noisy-key routing, the gram factorization and a
# passthrough covariance, on the frozen reference route (no launcher read).
def _diagonal_route_config(**over: Any) -> VFE3Config:
    values = dict(relational_routing="noisy_key", predictive_covariance="passthrough", projected_factorization="gram",
                  rope_mode="comparison_invariant", rope_fraction=1.0, phi_type_dropout=0.0)
    return reference_k8_config(**{**values, **over})


def _model(cfg: VFE3Config, seed: int = 0) -> VFEModel:
    torch.manual_seed(seed)
    return VFEModel(cfg)


# ---------------------------------------------------------------------------------------------------- A017

_FRAME_DIAGONAL = dict(family="gaussian_frame_diagonal", predictive_readout="kernel", block_mlp_mode="coordinate",
                       block_mlp_covariance="passthrough", decode_mode="diagonal_chunked", norm_type_block="layernorm",
                       pos_phi="none", untie_decode_bank=True, diagnose_canonical_free_energy=True,
                       predictive_covariance_floor_mode="relative", prior_handoff_rho=0.0)


def _frame_diagonal_model(rope_mode: str) -> VFEModel:
    r"""Frame-diagonal family at K = 8 in float64, with nontrivial frames and log reference variances perturbed by
    N(0, 1), so the variances are anisotropic within every RoPE plane."""
    model = _model(reference_k8_config(**_FRAME_DIAGONAL, rope_mode=rope_mode)).double()
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():
        pb = model.prior_bank
        pb.phi_embed.copy_(0.3 * torch.randn(pb.phi_embed.shape, generator=g, dtype=F64))
        pb.sigma_log_embed.add_(torch.randn(pb.sigma_log_embed.shape, generator=g, dtype=F64))
    return model.eval()


@pytest.mark.parametrize("rope_mode", ["comparison", "comparison_invariant", "none"])
def test_a017_canonical_frame_diagonal_block_rotates_mean_and_reference_variances(rope_mode: str) -> None:
    r"""On ``gaussian_frame_diagonal`` the canonical block reads :math:`(R_i\xi_i, R_i\operatorname{diag}(c_i)R_i^\top)`
    and the self term the frame-free KL of :math:`(\xi, c)`: ``self_divergence`` equals the dense oracle
    :func:`frame_diagonal_kl` on the layer input and its prior, and ``belief_coupling`` equals
    :func:`canonical_routing_rows` on the lifted local covariance :math:`U\operatorname{diag}(c)U^\top` with the rotated
    inverse :math:`R_iU_i^{-1}`."""
    model = _frame_diagonal_model(rope_mode)
    cfg, group = model.cfg, model.group
    ids = _ids(cfg.vocab_size, 2, 4)
    snapshot = model.build_diagnostic_snapshot(ids)
    d = model.diagnostics(ids, snapshot=snapshot)

    layer = cfg.n_layers - 1
    q = snapshot.layer_outputs[layer - 1]
    mu_p, c_p = snapshot.layer_priors[layer]
    A = group.embed_algebra(snapshot.initial_belief.phi.double(), dtype=F64)     # (B, N, K, K)
    U, U_inv = torch.linalg.matrix_exp(A), torch.linalg.matrix_exp(-A)
    self_divergence = frame_diagonal_kl(q.mu.double(), q.sigma.double(), U, mu_p.double(), c_p.double(), U).mean()
    assert float(d["self_divergence"]) == pytest.approx(float(self_divergence), rel=1e-10, abs=1e-12)

    ops = model.relational_operators_by_layer[layer]
    N, K = ids.shape[-1], U.shape[-1]
    rotation = torch.eye(K, dtype=F64).expand(N, K, K).clone()                   # (N, K, K) block-diagonal R_i
    if getattr(ops, "comparison_rope_base", None) is not None:
        offset = 0
        for width in group.irrep_dims:
            rotation[:, offset:offset + width, offset:offset + width] = ops.comparison_rope_rotation(
                N, width, U.device, F64)
            offset += width
    local = U @ torch.diag_embed(q.sigma.double()) @ U.transpose(-1, -2)          # (B, N, K, K) U diag(c) U^T
    tau = ops.routing_temperature(attention_tau(_as_coeff(cfg.kappa_beta, U.device), group.irrep_dims,
                                                device=U.device), F64)
    noise = getattr(ops, "routing_key_log_noise", None)
    rows = canonical_routing_rows(q.mu.double(), local, rotation @ U_inv, snapshot.log_prior, tau=tau,
                                  irrep_dims=list(group.irrep_dims),
                                  key_noise=None if noise is None else noise.detach().double().exp())
    belief_coupling = rows.belief_coupling.sum(dim=1).mean()
    assert float(d["belief_coupling"]) == pytest.approx(float(belief_coupling), rel=1e-9, abs=1e-12)


# ---------------------------------------------------------------------------------------------------- A018

def _mstep_pair_divergence(model: VFEModel, ids: torch.Tensor) -> torch.Tensor:
    r"""The divergence :math:`D(q^*_L\|p_L)` the M-step self term reads, in float64 and unclamped."""
    cfg = model.cfg
    capture: MStepCapture = {}
    with torch.no_grad():
        model.forward_beliefs(ids, capture=capture)
    converged = capture["converged"]
    mu_p, sigma_p = capture["final_block_prior"]
    fam = get_family(cfg.family)
    return self_divergence_for_alpha(fam(converged.mu.double(), converged.sigma.double()),
                                     fam(mu_p.double(), sigma_p.double()), kl_max=math.inf, eps=cfg.eps,
                                     lambda_alpha_mode=cfg.lambda_alpha_mode)


def test_a018_canonical_clamped_share_reads_the_pair_the_mstep_term_clamps() -> None:
    r"""Two layers, passthrough covariance, ``prior_handoff_rho = 1``: the last layer's input has zero self
    divergence, while the M-step pair (``capture['converged']`` against ``capture['final_block_prior']``) does not.
    ``canonical_self_clamped_frac`` must be that pair's share at or above ``self_kl_max``."""
    cfg = tiny_config(embed_dim=8, n_layers=2, predictive_covariance="passthrough", prior_handoff_rho=1.0,
                      mstep_self_coupling_weight=1.0, diagnose_canonical_free_energy=True, phi_scale=0.3)
    model = _model(cfg).eval()
    ids = _ids(cfg.vocab_size, 2, 3, seed=4)
    D = _mstep_pair_divergence(model, ids).reshape(-1)
    distinct = torch.unique(D)
    assert distinct.numel() >= 2 and float(distinct[-1]) > 0.0
    for cap in (0.5 * float(distinct[0]) + 1e-30, 0.5 * float(distinct[-2] + distinct[-1])):
        model.cfg.self_kl_max = cap
        expected = float((D >= cap).double().mean())
        assert expected > 0.0
        assert model.diagnostics(ids)["canonical_self_clamped_frac"] == expected


# ---------------------------------------------------------------------------------------------------- A053

@pytest.mark.parametrize("spec, extra", [([("l1", 1), ("l2", 1)], {}),
                                         ([("l1", 1), ("l2", 1)], {"pos_phi": "none"}),
                                         ([("l1", 1), ("l0", 1)], {"use_cg_coupling": True})],
                         ids=["so_n_l1_l2", "so_n_l1_l2_no_pos", "cg_so_n_l1_l0"])
def test_a053_readout_state_on_a_group_without_equal_compact_blocks(spec: list, extra: Dict[str, Any]) -> None:
    from vfe3.viz import extract
    K = sum({"l0": 1, "l1": 3, "l2": 5}[name] * count for name, count in spec)
    cfg = tiny_config(gauge_group="so_n", group_n=3, irrep_spec=spec, embed_dim=K, n_heads=len(spec), **extra)
    model = _model(cfg)
    with torch.no_grad():
        model.prior_bank.phi_embed.normal_(mean=0.0, std=0.3, generator=torch.Generator().manual_seed(2))
    assert not model._compact_phi_blocks_enabled()
    tokens = torch.tensor([[0, 1, 2]])
    snapshot = model.build_diagnostic_snapshot(tokens)
    state = extract.readout_state(model, tokens, snapshot=snapshot)
    N = tokens.shape[-1]
    assert state["exp_phi"].shape == (N, K, K)
    assert torch.is_tensor(state["omega"]) and state["omega"].shape == (N, N, K, K)
    belief = snapshot.layer_converged[-1]
    U = torch.linalg.matrix_exp(model.group.embed_algebra(belief.phi[0].double(), dtype=F64))
    if belief.right_phi is not None:
        right = belief.right_phi[0] if belief.right_phi.dim() == belief.phi.dim() else belief.right_phi
        U = U @ torch.linalg.matrix_exp(model.group.embed_algebra(right.double(), dtype=F64))
    torch.testing.assert_close(state["exp_phi"].double(), U, rtol=1e-5, atol=1e-5)
    omega = torch.einsum("ikl,jlm->ijkm", U, torch.linalg.inv(U))                 # (N, N, K, K) U_i U_j^{-1}
    torch.testing.assert_close(state["omega"].double(), omega, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------------------------------- A142

def _phi_storages(snapshot: Any) -> set:
    beliefs = [snapshot.encoded_belief, snapshot.initial_belief, *snapshot.layer_converged, *snapshot.layer_outputs,
               *snapshot.layer_block_mlp_inputs, *snapshot.layer_block_mlp_outputs, snapshot.stack_output,
               snapshot.final_belief, *snapshot.trace_states]
    return {belief.phi.untyped_storage().data_ptr() for belief in beliefs}


def _record_private_snapshots(monkeypatch: pytest.MonkeyPatch) -> List[Any]:
    built: List[Any] = []
    original = VFEModel.build_diagnostic_snapshot

    def recording(self: VFEModel, *args: Any, **kwargs: Any) -> Any:
        snapshot = original(self, *args, **kwargs)
        built.append(snapshot)
        return snapshot

    monkeypatch.setattr(VFEModel, "build_diagnostic_snapshot", recording)
    return built


def _same(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    if a.keys() != b.keys():
        return False
    for key in a:
        x, y = a[key], b[key]
        if isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y):
            continue
        if x != y:
            return False
    return True


def test_a142_private_diagnostics_snapshot_holds_one_frame_copy_and_no_clones(
        monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _diagonal_route_config(n_layers=3)
    model = _model(cfg)
    ids = _ids(cfg.vocab_size, 2, 4)
    public = model.build_diagnostic_snapshot(ids, return_logits=False)
    assert len(_phi_storages(public)) == 4 * cfg.n_layers + 6                    # public calls own their copies
    built = _record_private_snapshots(monkeypatch)
    private_values = model.diagnostics(ids)
    (private,) = built
    assert len(_phi_storages(private)) == 1
    last = private.layer_conditional_captures[-1]
    assert (last["projected_covariance_pre"].untyped_storage().data_ptr()
            == private.relational_capture["projected_covariance_pre"].untyped_storage().data_ptr())
    assert _same(private_values, model.diagnostics(ids, snapshot=public))


@pytest.mark.parametrize("rope_mode", ["comparison_invariant", "none"], ids=["diagonal_route", "dense_route"])
def test_a142_capture_layout_of_the_projected_covariances(rope_mode: str) -> None:
    cfg = _diagonal_route_config(rope_mode=rope_mode)
    model = _model(cfg)
    ids = _ids(cfg.vocab_size, 2, 4)
    snapshot = model.build_diagnostic_snapshot(ids, return_logits=False)
    H, r = cfg.n_heads, cfg.embed_dim // cfg.n_heads
    for cap in snapshot.layer_conditional_captures:
        diagonal = rope_mode == "comparison_invariant"
        assert cap["projected_covariance_diagonal_pre"] is diagonal
        shape = (2, 4, H, r) if diagonal else (2, 4, H, r, r)
        assert cap["projected_covariance_pre"].shape == cap["projected_key_covariance_pre"].shape == shape


def test_a142_numerics_counts_read_either_capture_layout() -> None:
    r"""The factorization denominator of :func:`vfe3.train._capture_numerics_counts` is the same whether the capture
    holds the diagonal of :math:`P(G)` or the dense matrices (the post-hoc and block-variance captures take the same
    route; posthoc-routing-5). It is the router's own count ``projected_factorized_entries_pre``, zero on this
    diagonal comparison route, which factors nothing (training-diagnostics-1)."""
    from vfe3.train import _capture_numerics_counts
    cfg = _diagonal_route_config(n_layers=2)
    model = _model(cfg)
    B, N = 2, 4
    capture: Dict[str, Any] = {"diagnostic": {}}
    with torch.no_grad():
        model.forward_beliefs(_ids(cfg.vocab_size, B, N), capture=capture)
    diagnostic = capture["diagnostic"]
    dense = {**diagnostic, "layer_conditional_captures": [
        {**cap, "projected_covariance_diagonal_pre": False,
         "projected_covariance_pre": torch.diag_embed(cap["projected_covariance_pre"]),
         "projected_key_covariance_pre": torch.diag_embed(cap["projected_key_covariance_pre"])}
        for cap in diagnostic["layer_conditional_captures"]]}
    assert all(cap["projected_covariance_diagonal_pre"] is True for cap in diagnostic["layer_conditional_captures"])
    counts = _capture_numerics_counts(diagnostic, cfg.kl_max)
    assert counts == _capture_numerics_counts(dense, cfg.kl_max)
    assert [attempted for _, attempted in counts[1]] == [0] * cfg.n_layers


# ---------------------------------------------------------------------------------------------------- A157

def test_a157_diagonal_layout_statistics_match_the_dense_eigen_solves() -> None:
    cfg = _diagonal_route_config(n_layers=2)
    model = _model(cfg)
    ids = _ids(cfg.vocab_size, 2, 4, seed=3)
    snapshot = model.build_diagnostic_snapshot(ids, return_logits=False)
    for layer in range(cfg.n_layers):
        cap = snapshot.layer_conditional_captures[layer]
        assert cap["projected_covariance_diagonal_pre"] is True
        dense = {**cap, "projected_covariance_diagonal_pre": False,
                 "projected_covariance_pre": torch.diag_embed(cap["projected_covariance_pre"]),
                 "projected_key_covariance_pre": torch.diag_embed(cap["projected_key_covariance_pre"])}
        for fn in (VFEModel._projected_cov_extremes, VFEModel._projected_cov_positional_minimum):
            assert fn(cap) == pytest.approx(fn(dense), rel=1e-6, abs=1e-12)
        got = model._covariance_scale_statistics(snapshot, layer, cap)
        want = model._covariance_scale_statistics(snapshot, layer, dense)
        assert got.keys() == want.keys()
        for key in got:
            assert got[key] == pytest.approx(want[key], rel=1e-9, abs=1e-12), key


def test_a157_diagnostics_builds_the_content_frame_once(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _diagonal_route_config(n_layers=3)
    model = _model(cfg)
    ids = _ids(cfg.vocab_size, 2, 4)
    calls: List[int] = []
    original = VFEModel._relational_frames

    def counting(self: VFEModel, *args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(VFEModel, "_relational_frames", counting)
    model.diagnostics(ids)
    assert len(calls) == 2                                                     # the forward, then one content frame
    calls.clear()
    model.diagnostics_per_layer(ids)
    assert len(calls) == 2


# ---------------------------------------------------------------------------------------------------- A080, A081

def _profiled(D: torch.Tensor, b0: float, c0: float) -> torch.Tensor:
    r""":math:`c_0\log(1 + D/b_0) = \min_\alpha[\alpha D + R(\alpha)] - R(c_0/b_0)`, the profiled self-coupling."""
    return c0 * torch.log1p(D / b0)


@pytest.mark.parametrize("mode", ["state_dependent", "state_dependent_per_coord"])
def test_a080_infinite_clamp_drops_a_non_finite_self_divergence(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    import vfe3.free_energy as free_energy
    w, b0, c0 = 0.5, 1.0, 1.0
    cfg = tiny_config(embed_dim=8, n_layers=2, mstep_self_coupling_weight=w, self_kl_max=math.inf,
                      lambda_alpha_mode=mode, b0=b0, c0=c0)
    model = _model(cfg).eval()
    ids, targets = _ids(cfg.vocab_size, 2, 3, seed=1), _ids(cfg.vocab_size, 2, 3, seed=2)
    D = _mstep_pair_divergence(model, ids)                                    # (B, N) or (B, N, K)
    original = free_energy.self_divergence_for_alpha

    def poisoned(*args: Any, **kwargs: Any) -> torch.Tensor:
        out = original(*args, **kwargs).clone()
        out[(0, 1) if out.dim() == 2 else (0, 1, 0)] = float("inf")           # one degenerate token
        return out

    monkeypatch.setattr(free_energy, "self_divergence_for_alpha", poisoned)
    _, loss, ce = model(ids, targets)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(bool(torch.isfinite(g).all()) for g in grads)
    with_term = model.prior_bank.mu_embed.grad.clone()
    model.zero_grad(set_to_none=True)
    model.cfg.mstep_self_coupling_weight = 0.0
    model(ids, targets)[1].backward()
    model.cfg.mstep_self_coupling_weight = w
    self_term_grad = with_term - model.prior_bank.mu_embed.grad                 # the other tokens' self-term gradient
    assert bool(torch.isfinite(self_term_grad).all()) and float(self_term_grad.abs().sum()) > 0.0
    term = _profiled(D, b0, c0)
    if term.dim() == 3:
        term = term.sum(dim=-1)
    keep = torch.ones_like(term, dtype=torch.bool)
    keep[0, 1] = False
    expected = float(ce) + w * float(term[keep].mean())
    assert float(loss) == pytest.approx(expected, rel=1e-5, abs=1e-6)


def test_a080_full_family_non_pd_token_is_dropped_with_finite_gradients() -> None:
    r"""The failure path the lane names, on ``gaussian_full`` under ``self_kl_max = inf``: an indefinite
    :math:`\Sigma_q` fails every jittered Cholesky round, so its closed-form KL is NaN and the clamp maps it to
    :math:`D = \infty`, where :math:`\alpha^*D = 0\cdot\infty`. The term drops that token: its value is the profiled
    mean over the other tokens, every gradient is finite, the dropped token's gradient is exactly zero and the other
    tokens receive the gradient of the mean over them alone."""
    from vfe3.model.model import _mstep_self_coupling_term
    fam = get_family("gaussian_full")
    g = torch.Generator().manual_seed(3)
    N, K, bad, b0, c0 = 4, 3, 2, 0.5, 1.5
    a = torch.randn(1, N, K, K, generator=g, dtype=F64)
    sigma_q = a @ a.transpose(-1, -2) + 0.5 * torch.eye(K, dtype=F64)        # (1, N, K, K) SPD
    sigma_q[0, bad] = torch.diag(torch.tensor([1.0, -1.0, 1.0], dtype=F64))   # indefinite: eigenvalue -1
    mu_q = torch.randn(1, N, K, generator=g, dtype=F64)
    mu_p = torch.randn(1, N, K, generator=g, dtype=F64)
    c = torch.randn(1, N, K, K, generator=g, dtype=F64)
    sigma_p = c @ c.transpose(-1, -2) + torch.eye(K, dtype=F64)

    def term(keep: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        m, s = mu_q[:, keep].clone().requires_grad_(), sigma_q[:, keep].clone().requires_grad_()
        D = self_divergence_for_alpha(fam(m, s), fam(mu_p[:, keep], sigma_p[:, keep]), kl_max=math.inf,
                                      eps=1e-6, lambda_alpha_mode="state_dependent")   # (1, n)
        value = _mstep_self_coupling_term(D, mode="state_dependent", b0=b0, c0=c0)
        value.backward()
        return value.detach(), D.detach(), m.grad, s.grad

    every, others = torch.arange(N), torch.tensor([i for i in range(N) if i != bad])
    value, D, grad_mu, grad_sigma = term(every)
    assert math.isinf(float(D[0, bad])) and bool(torch.isfinite(D[0, others]).all())
    assert torch.isfinite(value) and bool(torch.isfinite(grad_mu).all()) and bool(torch.isfinite(grad_sigma).all())
    assert float(grad_mu[0, bad].abs().max()) == 0.0 and float(grad_sigma[0, bad].abs().max()) == 0.0
    value_ref, D_ref, grad_mu_ref, grad_sigma_ref = term(others)
    torch.testing.assert_close(value, value_ref, rtol=1e-14, atol=0.0)
    torch.testing.assert_close(value, (c0 * torch.log1p(D_ref / b0)).mean(), rtol=1e-12, atol=0.0)
    torch.testing.assert_close(grad_mu[:, others], grad_mu_ref, rtol=1e-12, atol=0.0)
    torch.testing.assert_close(grad_sigma[:, others], grad_sigma_ref, rtol=1e-12, atol=0.0)


def test_a081_logged_self_term_is_the_function_its_gradient_descends() -> None:
    r"""Float64 central difference of the returned loss against autograd with :math:`D \gg b_0`: the value is the
    profiled term :math:`c_0\log(1 + D/b_0)` and the gradient :math:`\alpha^*\,dD`."""
    w, b0 = 10.0, 0.1
    cfg = tiny_config(embed_dim=8, n_layers=1, predictive_covariance="moment_matched", mstep_self_coupling_weight=w,
                      b0=b0, c0=1.0)
    model = _model(cfg).double().eval()
    ids, targets = _ids(cfg.vocab_size, 2, 3, seed=5), _ids(cfg.vocab_size, 2, 3, seed=6)
    assert float(_mstep_pair_divergence(model, ids).mean()) > 10.0 * b0
    table = model.prior_bank.mu_embed
    direction = torch.randn(table.shape, generator=torch.Generator().manual_seed(7), dtype=F64)
    _, loss, _ = model(ids, targets)
    loss.backward()
    autograd = float((table.grad * direction).sum())
    h = 1e-6
    values = []
    with torch.no_grad():
        base = table.detach().clone()
        for sign in (1.0, -1.0):
            table.copy_(base + sign * h * direction)
            values.append(float(model(ids, targets)[1]))
        table.copy_(base)
    central = (values[0] - values[1]) / (2.0 * h)
    assert abs(autograd) > 1e-3
    assert central == pytest.approx(autograd, rel=1e-6)


def test_a081_self_term_gradient_is_unchanged_by_the_profiled_value(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""The detached profile correction leaves the gradient bitwise :math:`\alpha^*\,dD` (the pre-remediation term)."""
    import vfe3.model.model as model_module
    cfg = tiny_config(embed_dim=8, n_layers=2, mstep_self_coupling_weight=0.7, b0=0.5, c0=2.0)
    ids, targets = _ids(cfg.vocab_size, 2, 3, seed=8), _ids(cfg.vocab_size, 2, 3, seed=9)

    def grads(model: VFEModel) -> List[torch.Tensor]:
        _, loss, _ = model(ids, targets)
        loss.backward()
        return [p.grad.clone() for p in model.parameters() if p.grad is not None]

    new = grads(_model(cfg).eval())

    def alpha_weighted(self_div: torch.Tensor, *, mode: str, b0: Any, c0: Any) -> torch.Tensor:
        from vfe3.alpha_i import alpha_is_per_coord, self_coupling_alpha
        alpha, _ = self_coupling_alpha(self_div, mode=mode, b0=b0, c0=c0)
        coupling = alpha.detach() * self_div
        return (coupling.sum(dim=-1) if alpha_is_per_coord(mode) else coupling).mean()

    monkeypatch.setattr(model_module, "_mstep_self_coupling_term", alpha_weighted)
    old = grads(_model(cfg).eval())
    assert len(new) == len(old) and all(torch.equal(a, b) for a, b in zip(new, old))


# ---------------------------------------------------------------------------------------------------- A118

def test_a118_construction_warns_when_it_changes_a_live_models_numeric_policy(
        monkeypatch: pytest.MonkeyPatch) -> None:
    import weakref
    import vfe3.model.model as model_module
    from vfe3.model.model import NumericPolicyWarning
    monkeypatch.setattr(model_module, "_LIVE_POLICY_MODELS", weakref.WeakSet())   # no other test's live models
    monkeypatch.setattr(model_module, "_INSTALLED_POLICY", {})
    a = _model(_diagonal_route_config())
    ids = _ids(a.cfg.vocab_size, 2, 4)
    with torch.no_grad():
        before = a.forward_beliefs(ids, return_logits=True)[1]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        b = _model(_diagonal_route_config(projected_factorization="gram_certified"))
        same = _model(_diagonal_route_config(projected_factorization="gram_certified"))
    policy = [w for w in caught if issubclass(w.category, NumericPolicyWarning)]
    assert len(policy) == 1 and "projected_factorization" in str(policy[0].message)
    a.apply_numeric_policy()
    with torch.no_grad():
        after = a.forward_beliefs(ids, return_logits=True)[1]
    assert torch.equal(before, after)
    del a, b, same
    gc.collect()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _model(_diagonal_route_config(projected_factorization="cholesky_qr2_certified"))
    assert not [w for w in caught if issubclass(w.category, NumericPolicyWarning)]


# ---------------------------------------------------------------------------------------------------- A183

@pytest.mark.parametrize("profile", ["diagonal_route", "live"])
def test_a183_zero_weight_handoff_blends_are_bitwise_neutral(monkeypatch: pytest.MonkeyPatch, profile: str) -> None:
    r"""The live handoff (``prior_handoff_rho = 1``, ``prior_handoff_sigma = 0``) runs on a log-step forward
    (``estep_grad_out`` set); skipping its zero-weight variance blend leaves the loss and every gradient bitwise, on
    the K = 8 diagonal route and on the tiny live profile with the two handoff weights pinned."""
    import vfe3.model.stack as stack
    from tests.profiles import live_config
    handoff = dict(n_layers=3, prior_handoff_rho=1.0, prior_handoff_sigma=0.0)
    cfg = _diagonal_route_config(**handoff) if profile == "diagonal_route" else live_config(**handoff)
    ids, targets = _ids(cfg.vocab_size, 2, 4, seed=1), _ids(cfg.vocab_size, 2, 4, seed=2)

    def run() -> Tuple[torch.Tensor, List[torch.Tensor]]:
        model = _model(cfg)
        _, loss, _ = model(ids, targets, estep_grad_out={})
        loss.backward()
        return loss.detach(), [p.grad.clone() for p in model.parameters() if p.grad is not None]

    loss_new, grads_new = run()
    monkeypatch.setattr(stack, "_handoff_blend", lambda prior, belief, weight: (1.0 - weight) * prior + weight * belief)
    loss_old, grads_old = run()
    assert torch.equal(loss_new, loss_old)
    assert len(grads_new) == len(grads_old) and all(torch.equal(a, b) for a, b in zip(grads_new, grads_old))


def test_a183_zero_weight_variance_blend_keeps_a_non_finite_belief_out_of_the_prior() -> None:
    from vfe3.model.stack import _handoff_blend
    prior = torch.tensor([1.0, 2.0])
    belief = torch.tensor([float("inf"), 3.0])
    assert _handoff_blend(prior, belief, 0.0) is prior
    assert torch.equal(_handoff_blend(prior, belief, 0.25), 0.75 * prior + 0.25 * belief)


class _NoReuse(VFEModel):
    r"""Fails the content-row gate's exact-type check, so its frames are built per occurrence."""


def test_a183_mstep_capture_admits_the_content_factor_reuse(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _diagonal_route_config(n_layers=2, mstep_self_coupling_weight=0.5)
    ids = torch.tensor([[1, 2, 1, 3], [2, 1, 4, 1]])                           # repeated ids within and across rows
    targets = _ids(cfg.vocab_size, 2, 4, seed=3)
    seen: List[Any] = []
    original = VFEModel._relational_frames

    def spy(self: VFEModel, *args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs.get("_content_token_ids"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(VFEModel, "_relational_frames", spy)
    results = []
    for cls in (VFEModel, _NoReuse):
        torch.manual_seed(0)
        model = cls(cfg)
        _, loss, _ = model(ids, targets)
        loss.backward()
        results.append((loss.detach(), [p.grad.clone() for p in model.parameters() if p.grad is not None]))
    assert seen[0] is not None and torch.equal(seen[0], ids) and seen[1] is None
    (loss_reuse, grads_reuse), (loss_occurrence, grads_occurrence) = results
    assert torch.equal(loss_reuse, loss_occurrence)
    assert len(grads_reuse) == len(grads_occurrence)
    for a, b in zip(grads_reuse, grads_occurrence):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)


# ---------------------------------------------------------------------------------------------------- A196

def test_a196_canonical_frame_block_mlp_reuses_the_validated_compact_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tiny_config(embed_dim=8, use_block_mlp=True, block_mlp_mode="canonical_frame", phi_scale=0.3)
    ids, targets = _ids(cfg.vocab_size, 2, 3, seed=1), _ids(cfg.vocab_size, 2, 3, seed=2)
    validated: List[Tuple[bool, Tuple[int, ...]]] = []
    original = CanonicalFrameContext.__post_init__

    def counting(self: CanonicalFrameContext) -> None:
        validated.append((self.block_diagonal, tuple(self.forward.shape)))
        original(self)

    monkeypatch.setattr(CanonicalFrameContext, "__post_init__", counting)
    model = _model(cfg)
    _, loss, _ = model(ids, targets)
    K = cfg.embed_dim
    assert [entry for entry in validated if not entry[0] and entry[1][-2:] == (K, K)] == []
    assert any(entry[0] for entry in validated)
    loss.backward()
    grads = [p.grad.clone() for p in model.parameters() if p.grad is not None]

    # The pre-remediation construction: the dense pair validated again at the occurrence shape.
    from vfe3.geometry.lie_ops import CompactBlockElement

    def validated_dense(frame: CanonicalFrameContext) -> CanonicalFrameContext:
        if not frame.block_diagonal:
            return frame
        return CanonicalFrameContext(forward=CompactBlockElement(frame.forward, K).to_dense(),
                                     inverse=CompactBlockElement(frame.inverse, K).to_dense())

    monkeypatch.setattr(VFEModel, "_dense_frame_context", staticmethod(validated_dense))
    reference = _model(cfg)
    _, loss_ref, _ = reference(ids, targets)
    loss_ref.backward()
    assert torch.equal(loss.detach(), loss_ref.detach())
    grads_ref = [p.grad for p in reference.parameters() if p.grad is not None]
    assert len(grads) == len(grads_ref) and all(torch.equal(a, b) for a, b in zip(grads, grads_ref))


def test_a196_trusted_dense_context_checks_the_layout() -> None:
    from vfe3.contracts import trusted_dense_frame_context
    blocks = torch.eye(2).expand(1, 3, 2, 2, 2).clone()                        # (B, N, H, d, d)
    compact = CanonicalFrameContext(blocks, blocks.clone(), block_diagonal=True)
    dense = torch.eye(4).expand(1, 3, 4, 4).clone()
    context = trusted_dense_frame_context(dense, dense.clone(), validated=compact)
    assert not context.block_diagonal and context.forward.shape == (1, 3, 4, 4)
    with pytest.raises(ValueError):
        trusted_dense_frame_context(dense[..., :3, :3], dense[..., :3, :3], validated=compact)
    with pytest.raises(ValueError):
        trusted_dense_frame_context(dense, dense, validated=CanonicalFrameContext(dense, dense))


# ---------------------------------------------------------------------------------------------------- A291

def test_a291_generate_memory_estimate_counts_last_position_logits(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""``decode_last`` decodes (B, 1, V): at V = 50000, N = 512, two heads and B = 24 the old (B, N, V) count,
    :math:`4B(HN^2 + NV + NK)` = 2.34 GiB, passed the 2 GiB budget, the last-position count
    :math:`4B(HN^2 + V + NK)` = 0.05 GiB does not, and B = 4000 (8.6 GiB) still warns. The estimate reads only the
    configuration, so a tiny model with those fields set stands in for the large one (``max_new_tokens = 0`` runs
    no forward)."""
    model = _model(tiny_config())
    for name, value in (("vocab_size", 50000), ("max_seq_len", 512), ("n_heads", 2), ("embed_dim", 4)):
        monkeypatch.setattr(model.cfg, name, value)

    def warns(batch: int) -> bool:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.generate(torch.zeros(batch, 1, dtype=torch.long), max_new_tokens=0, greedy=True)
        return any("estimated per-forward peak" in str(w.message) for w in caught)

    assert not warns(24)
    assert warns(4000)


def test_a291_mstep_capture_declares_the_diagnostic_key() -> None:
    assert "diagnostic" in MStepCapture.__annotations__


# ---------------------------------------------------------------------------------------------------- A221

_FROZEN_TAU_SCRIPT = """
import json, sys
try:
    import triton  # noqa: F401
except Exception as exc:
    print("triton import failed:", exc)
    sys.exit(77)
import copy
import torch
from tests.profiles import tiny_config
from tests.test_kl_decode_triton_20261012 import _bank, _inputs
from vfe3.kernels import _kl_decode_triton as tk
from vfe3.kernels import kl_decode
from vfe3.model.model import VFEModel
from vfe3.model.prior_bank import PriorBank
assert tk.INTERPRETED

def forced(self, lhs, u_all):                       # the interpreter runs the fused route on CPU tensors
    self.realized_decode_ce_kernel = "triton" if self.decode_ce_triton else "torch"
    return self.decode_ce_triton
PriorBank._decode_ce_triton_active = forced

need_tau = []
launch = kl_decode._launch
def recording(kernel, n_rows, cfg, *args, **constexprs):
    if "NEED_TAU" in constexprs:
        need_tau.append(bool(constexprs["NEED_TAU"]))
    return launch(kernel, n_rows, cfg, *args, **constexprs)
kl_decode._launch = recording

def ce_and_grads(pb, mu_q, sigma_q, targets):
    mu_q = mu_q.detach().clone().requires_grad_()
    sigma_q = sigma_q.detach().clone().requires_grad_()
    named = [("mu_q", mu_q), ("sigma_q", sigma_q)] + [(n, p) for n, p in pb.named_parameters() if p.requires_grad]
    ce = pb.decode_ce_diagonal_chunked(mu_q, sigma_q, targets)
    grads = torch.autograd.grad(ce, [t for _, t in named], allow_unused=True)
    return ce, {n: g for (n, _), g in zip(named, grads) if g is not None}

def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-300))

def compare(label, banks, inputs):
    mu_q, sigma_q, targets = inputs
    reference, routes = banks
    ce_ref, g_ref = ce_and_grads(reference, mu_q, sigma_q, targets)
    out = {"label": label, "rows": int((targets != -100).sum()), "frozen": "decode_log_scale" not in g_ref}
    for name, pb in routes.items():
        dtype = next(pb.parameters()).dtype
        ce, g = ce_and_grads(pb, mu_q.to(dtype), sigma_q.to(dtype), targets)
        assert pb.realized_decode_ce_kernel == name and set(g) == set(g_ref), (name, sorted(g), sorted(g_ref))
        out[name] = {"ce": abs(float(ce) - float(ce_ref)) / max(abs(float(ce_ref)), 1e-300),
                     **{n: rel(g[n], g_ref[n]) for n in g_ref if g_ref[n].norm() > 0}}
        out["dtype"] = str(dtype)
    return out

results = []
B, N, V, K = 1, 13, 9, 5                           # 13 - 4 ignored = 9 rows (odd); chunks of 4, 4, 1
for dtype, policy in ((torch.float32, "never"), (torch.float32, "always"), (torch.float64, "never")):
    inputs = _inputs("diagonal_chunked", torch.float64, B=B, N=N, V=V, K=K, sigma_init=15.0)
    banks = {}
    for name, triton in (("reference", False), ("torch", False), ("triton", True)):
        pb = _bank("diagonal_chunked", torch.float64 if name == "reference" else dtype, V=V, K=K, tau=0.1,
                   sigma_init=15.0, untie=True, unigram=True, triton=triton, chunk=4)
        pb.decode_log_scale.requires_grad_(False)   # frozen as VFEModel freezes it: NEED_TAU=False
        pb.decode_ce_checkpoint = policy
        banks[name] = pb
    reference = banks.pop("reference")
    results.append(compare(f"bank_{dtype}_{policy}", (reference, banks), inputs))

torch.manual_seed(0)                                 # a bank built through VFEModel, as the training step uses it
cfg = tiny_config(vocab_size=9, embed_dim=4, decode_mode="diagonal_chunked", decode_ce_triton=True,
                  decode_unigram_prior=True, untie_decode_bank=True, decode_chunk_size=4, decode_tau=0.1,
                  sigma_init=15.0)
model = VFEModel(cfg)
with torch.no_grad():                                # distinct decode variances: at init they are all equal, the
    for name, p in model.prior_bank.named_parameters():   # trace term cancels and d CE / d sigma_q is exactly zero
        if "sigma_log" in name:
            p.add_(0.3 * torch.randn(p.shape, generator=torch.Generator().manual_seed(5)))
model.prior_bank.set_unigram_log_prior(torch.randint(1, 50, (9,), generator=torch.Generator().manual_seed(4)).float())
inputs = _inputs("diagonal_chunked", torch.float64, B=1, N=13, V=9, K=4, sigma_init=15.0, seed=3)
for dtype in (torch.float32, torch.float64):
    reference = copy.deepcopy(model.prior_bank).double()
    reference.decode_ce_triton = False
    routes = {"torch": copy.deepcopy(model.prior_bank).to(dtype), "triton": copy.deepcopy(model.prior_bank).to(dtype)}
    routes["torch"].decode_ce_triton = False
    results.append(compare(f"model_bank_{dtype}", (reference, routes), inputs))
print(json.dumps({"results": results, "need_tau": need_tau}))
"""


def test_a221_fused_kl_decode_frozen_temperature_matches_the_torch_route_under_the_interpreter(
        record_property: Any) -> None:
    from tests.test_kl_decode_triton_20261012 import _run_script, _subprocess_env
    report = _run_script(_FROZEN_TAU_SCRIPT, _subprocess_env(TRITON_INTERPRET="1"))
    assert report["need_tau"] and not any(report["need_tau"])                  # every backward ran NEED_TAU=False
    for r in report["results"]:
        record_property(f"frozen_tau_{r['label']}", json.dumps({"torch": r["torch"], "triton": r["triton"]}))
        assert r["frozen"] and r["rows"] % 2 == 1, r
        assert {"mu_q", "sigma_q"} <= set(r["triton"])
        for name, err in r["triton"].items():
            if r["dtype"] == "torch.float64":
                assert err <= 1e-11, (name, r)
            else:                                     # as close to float64 as the torch route itself
                assert err <= 4.0 * r["torch"][name] + 5e-7, (name, r)


_FROZEN_TAU_COMPILE_SCRIPT = """
import json, os, re, subprocess, sys, tempfile
try:
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.backends.nvidia.compiler import get_ptxas
    from triton.compiler import ASTSource
    ptxas = get_ptxas().path
except Exception as exc:
    print("triton/ptxas unavailable:", exc)
    sys.exit(77)
from vfe3.kernels import _kl_decode_triton as tk
assert not tk.INTERPRETED
cfg = tk.launch_config()["bwd"]
kernel = tk.kl_ce_bwd_kernel
out = {}
for dt in ("fp32", "fp64"):
    consts = {"BI": cfg["BI"], "BV": cfg["BV"], "HAS_U": True, "PRECISE_DIV": dt == "fp32", "ACCURATE_EXP": True,
              "NEED_TAU": False}
    sig = {n: ("constexpr" if n in consts else "i32" if n in ("S", "VC", "V0") else "*i64" if n == "TGT"
               else "*" + dt) for n in kernel.arg_names}
    compiled = triton.compile(ASTSource(fn=kernel, signature=sig, constexprs=consts),
                              target=GPUTarget("cuda", 120, 32),
                              options={"num_warps": cfg["num_warps"], "num_stages": cfg["num_stages"]})
    ptx = compiled.asm["ptx"]
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "k.ptx")
        with open(src, "w") as f:
            f.write(ptx)
        log = subprocess.run([ptxas, "-v", "--gpu-name=sm_120a", src, "-o", src + ".o"],
                             capture_output=True, text=True)
    text = log.stdout + log.stderr
    spill = re.search(r"(\\d+) bytes spill stores, (\\d+) bytes spill loads", text)
    out["bwd_" + dt] = {"registers": int(re.search(r"Used (\\d+) registers", text).group(1)),
                        "spill_stores": int(spill.group(1)), "spill_loads": int(spill.group(2))}
print(json.dumps(out))
"""


def test_a221_frozen_temperature_backward_compiles_for_sm120_without_spills(tmp_path: Any,
                                                                            record_property: Any) -> None:
    from tests.test_kl_decode_triton_20261012 import _run_script, _subprocess_env
    report = _run_script(_FROZEN_TAU_COMPILE_SCRIPT,
                         _subprocess_env(TRITON_CACHE_DIR=str(tmp_path / "triton_cache")))
    assert set(report) == {"bwd_fp32", "bwd_fp64"}
    for key, r in report.items():
        record_property(f"sm120_need_tau_false_{key}", json.dumps(r))
        assert r["spill_stores"] == 0 and r["spill_loads"] == 0 and r["registers"] <= 255, (key, r)


@pytest.mark.skipif(__import__("os").environ.get("VFE3_TEST_DEVICE") == "cpu" or not torch.cuda.is_available(),
                    reason="requires the CUDA lane (VFE3_TEST_DEVICE=cuda) and a CUDA device")
def test_a221_cuda_fused_route_with_a_frozen_temperature_matches_float64_like_the_torch_route(
        record_property: Any) -> None:
    pytest.importorskip("triton")
    from tests.test_kl_decode_triton_20261012 import _bank, _inputs

    def ce_and_grads(pb: Any, mu_q: torch.Tensor, sigma_q: torch.Tensor,
                     targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        mu_q = mu_q.detach().clone().requires_grad_()
        sigma_q = sigma_q.detach().clone().requires_grad_()
        named = [("mu_q", mu_q), ("sigma_q", sigma_q)] + [(n, p) for n, p in pb.named_parameters()
                                                           if p.requires_grad]
        ce = pb.decode_ce_diagonal_chunked(mu_q, sigma_q, targets)
        grads = torch.autograd.grad(ce, [t for _, t in named], allow_unused=True)
        return ce, {n: g for (n, _), g in zip(named, grads) if g is not None}

    device = torch.device("cuda")
    B, N, V, K, chunk = 3, 63, 3001, 20, 1000                                  # odd row count, padded tiles
    mu_q, sigma_q, targets = _inputs("diagonal_chunked", torch.float64, B=B, N=N, V=V, K=K)
    reference = _bank("diagonal_chunked", torch.float64, V=V, K=K, tau=0.05, triton=False, chunk=chunk)
    reference.decode_log_scale.requires_grad_(False)
    ce_ref, g_ref = ce_and_grads(reference, mu_q, sigma_q, targets)
    errors = {}
    for name, triton in (("torch", False), ("triton", True)):
        pb = _bank("diagonal_chunked", torch.float32, V=V, K=K, tau=0.05, triton=triton, chunk=chunk).to(device)
        pb.decode_log_scale.requires_grad_(False)
        ce, g = ce_and_grads(pb, mu_q.float().to(device), sigma_q.float().to(device), targets.to(device))
        assert pb.realized_decode_ce_kernel == name and "decode_log_scale" not in g
        errors[name] = {"ce": abs(float(ce) - float(ce_ref)) / abs(float(ce_ref)),
                        **{n: float((g[n].double().cpu() - g_ref[n]).norm() / g_ref[n].norm()) for n in g_ref}}
    record_property("cuda_frozen_tau_rel_err_vs_fp64", json.dumps(errors))
    for n, err in errors["triton"].items():
        assert err <= 4.0 * errors["torch"][n] + 5e-7, (n, errors)
