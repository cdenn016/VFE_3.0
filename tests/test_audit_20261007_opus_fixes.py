"""Remediation of the 2026-10-07 audit items assigned to Opus (docs/audits/audit-2026-10-07-verification.md).

Tiny models on CPU (V=11, K=4, H=2, N=6). Inertness is checked as in tests/test_remediation_20260923_A.py:
the loss and every parameter gradient stay ``torch.equal`` when only the marked knob changes.
"""
from tests.profiles import config_values, test_config
import warnings
from typing import Any, Dict

import pytest
import torch

from vfe3.model.model import VFEModel

V, K, H, N, B = 11, 4, 2, 6, 2


def _minimal_agent(**overrides: Any) -> Dict[str, Any]:
    cfg = config_values()
    cfg.update(vocab_size=V, embed_dim=K, n_heads=H, max_seq_len=N, batch_size=B, eval_stride=None,
               attention_window=N, n_layers=2, attention_mode="projected_predictive",
               predictive_readout="minimal_agent", decode_mode="linear", use_block_mlp=True,
               block_mlp_mode="prototype_agents", norm_type_block="layernorm", norm_type_final="none",
               amp_dtype=None, kl_max=8 * K, mstep_self_coupling_weight=0.0, cg_energy_weight=0.0,
               use_cg_coupling=False, projected_factorization="gram")
    cfg.update(overrides)
    return cfg


def _cfg(values: Dict[str, Any]):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return test_config(**values)


def _loss_and_grads(values: Dict[str, Any]):
    cfg = _cfg(values)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    g = torch.Generator().manual_seed(7)
    ids = torch.randint(0, V, (B, N), generator=g)
    tgt = torch.randint(0, V, (B, N), generator=g)
    _, loss, _ = model(ids, tgt, estep_grad_out={})
    loss.backward()
    return loss.detach(), {name: p.grad for name, p in model.named_parameters()}


def _assert_bit_inert(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    loss_a, grads_a = _loss_and_grads(base)
    loss_b, grads_b = _loss_and_grads({**base, **override})
    assert torch.equal(loss_a, loss_b), override
    for name, grad in grads_a.items():
        other = grads_b[name]
        assert (grad is None) == (other is None), name
        if grad is not None:
            assert torch.equal(grad, other), (override, name)


@pytest.mark.parametrize("inference", ["mean_field", "exact"])
@pytest.mark.parametrize("override", [{"kl_max": 1.0}, {"amp_dtype": "bf16"}])
def test_minimal_agent_marks_and_is_bit_inert_in_the_reported_knobs(inference: str,
                                                                    override: Dict[str, Any]) -> None:
    base = _minimal_agent(minimal_agent_inference=inference, block_mlp_inference=inference)
    marked = _cfg({**base, "amp_dtype": "bf16"}).inactive_fields()
    assert next(iter(override)) in marked
    _assert_bit_inert(base, override)


@pytest.mark.parametrize("mlp", [{"use_block_mlp": False}, {"norm_type_block": "none"}])
def test_amp_rule_covers_the_other_listed_combinations(mlp: Dict[str, Any]) -> None:
    base = _minimal_agent(**mlp)
    assert "amp_dtype" in _cfg({**base, "amp_dtype": "bf16"}).inactive_fields()
    _assert_bit_inert(base, {"amp_dtype": "bf16"})


def test_amp_and_kl_max_stay_live_where_a_consumer_runs() -> None:
    coordinate = _cfg(_minimal_agent(block_mlp_mode="coordinate", amp_dtype="bf16")).inactive_fields()
    assert "amp_dtype" not in coordinate
    kernel = _cfg(_minimal_agent(predictive_readout="kernel", block_mlp_mode="coordinate",
                                 amp_dtype="bf16")).inactive_fields()
    assert "kl_max" not in kernel and "amp_dtype" not in kernel
    coupled = _cfg(_minimal_agent(mstep_self_coupling_weight=0.1)).inactive_fields()
    assert "kl_max" not in coupled


def test_mean_field_reason_and_record_do_not_claim_the_exact_posterior() -> None:
    from vfe3.run_artifacts import _relational_artifact_fields

    cfg = _cfg(_minimal_agent(minimal_agent_inference="mean_field"))
    assert "exact" not in cfg.inactive_fields()["kappa_beta"]
    record = _relational_artifact_fields(cfg)["relational_operators"]
    assert record["temperature_convention"] == "unit_temperature_mean_field_posterior_no_clamp"


def test_amp_probe_detects_a_bf16_consumer() -> None:
    """The bit-inertness probe has power: the coordinate MLP does change under CPU bf16 autocast."""
    base = _minimal_agent(block_mlp_mode="coordinate")
    loss_a, _ = _loss_and_grads(base)
    loss_b, _ = _loss_and_grads({**base, "amp_dtype": "bf16"})
    assert not torch.equal(loss_a, loss_b)


def test_minimal_agent_map_rows_drop_the_gl_r_quotient_columns() -> None:
    """Item 49: on the minimal agent the common scale and S-drift are not symmetry coordinates."""
    import math

    from vfe3.posthoc.maps import map_invariants

    cfg = _cfg(_minimal_agent())
    torch.manual_seed(0)
    model = VFEModel(cfg)
    rows = map_invariants(model.state_dict(), cfg)
    assert rows and all(row["key_tied_to_content"] == 1.0 for row in rows)
    for row in rows:
        assert math.isnan(row["scale_rms"]) and math.isnan(row["conserved_S_drift"])
        assert math.isfinite(row["scale_rms_noise_units"]) and row["T_sv_noise_units_max"] > 0
        assert math.isfinite(row["gain_median"]) and math.isfinite(row["angle_median_deg"])
    # (S_Q, T, n) -> (c S_Q, c T, c^2 n) leaves the noise-unit columns unchanged.
    state = {k: v.clone() for k, v in model.state_dict().items()}
    for key in state:
        if key.endswith("raw_score_q") or key.endswith(".T") or key == "relational_operators.T":
            state[key] = 3.0 * state[key]
        if key.endswith("channel_log_noise"):
            state[key] = state[key] + 2.0 * math.log(3.0)
    scaled = map_invariants(state, cfg)
    for a, b in zip(rows, scaled):
        assert math.isclose(a["scale_rms_noise_units"], b["scale_rms_noise_units"], rel_tol=1e-6)  # float32 log-noise


def test_mean_field_diagnostics_do_not_raise_on_a_failed_float32_utterance() -> None:
    """Item 13 (F57): a singular float32 utterance covariance is a failure flag, not a diagnostics exception."""
    import math

    from vfe3.model.minimal_agent import minimal_agent_readout

    torch.manual_seed(0)
    n, k = 3, 2
    xi = torch.randn(n, k)
    C = torch.eye(k).expand(n, k, k).clone()
    S = torch.eye(k)[None]                               # (H=1, d=2, K=2)
    T = torch.eye(k)[None].clone()
    T[0, 1] = 0.0                                        # second utterance row zero
    log_noise = torch.tensor([-200.0])                   # n_h underflows to 0 in float32
    out = minimal_agent_readout(xi, C, S, T, log_noise, inference="mean_field", diagnostics=True, check=False)
    assert bool(out.failed)
    assert all(math.isnan(v) for v in out.label_staleness.flatten().tolist())
    ok = minimal_agent_readout(xi, C, S, T, torch.zeros(1),                                 # healthy control
                               inference="mean_field", diagnostics=True, check=False)
    assert not bool(ok.failed) and torch.isfinite(ok.label_staleness).all()


@pytest.mark.parametrize("tile_bytes", [None, 1])
def test_exact_spread_has_no_float32_cancellation_at_large_offsets(monkeypatch: pytest.MonkeyPatch,
                                                                   tile_bytes) -> None:
    """Item 33 (F01/F58): R is accumulated centered, so a common offset of r_ij does not cancel."""
    import vfe3.model.minimal_agent as ma

    if tile_bytes is not None:
        monkeypatch.setattr(ma, "PAIR_TILE_BYTES", tile_bytes)     # one pair per tile: exercises Chan's merge
    outs = {}
    for dt in (torch.float32, torch.float64):
        a = torch.zeros(1, 1, dtype=dt)
        A = torch.ones(1, 1, 1, dtype=dt)
        b = torch.tensor([[300.0], [300.001], [300.002]], dtype=dt)
        B = torch.full((3, 1, 1), 1e-3, dtype=dt)
        outs[dt] = ma.exact_record_posterior(a, A, b, B, check=False)
    R32, R64 = outs[torch.float32].R.double(), outs[torch.float64].R
    assert float(R32.min()) >= 0.0
    # The uncentered form gave R = -7.8e-3 here; the residual ~2% is the float32 rounding of r_ij ~ 300 itself.
    assert torch.allclose(R32, R64, rtol=5e-2, atol=1e-9)


@pytest.mark.parametrize("kind,lower,upper", [("sqrt_floor", 1e-6, None), ("inv_sqrt_floor", 1e-6, None),
                                              ("exp_bounded", -50.0, 50.0), ("log_floor", 1e-6, None)])
def test_loewner_divided_differences_are_cancellation_free_in_float32(kind: str, lower: float, upper) -> None:
    """Item 35 (F33/F67): near-degenerate spectra keep float32 divided differences at roundoff accuracy."""
    from vfe3.geometry.retraction import _loewner_adjoint

    torch.manual_seed(0)
    k = 5
    lam64 = 1.0 + 1e-4 * torch.arange(k, dtype=torch.float64)          # resolvable, nearly isotropic
    Q64, _ = torch.linalg.qr(torch.randn(k, k, dtype=torch.float64))
    G64 = torch.randn(k, k, dtype=torch.float64)
    ref = _loewner_adjoint(lam64, Q64, kind, lower, upper, G64)
    got = _loewner_adjoint(lam64.float(), Q64.float(), kind, lower, upper, G64.float()).double()
    assert float((got - ref).norm() / ref.norm()) < 1e-5


def test_public_spd_certificate_repairs_are_counted() -> None:
    """Item 27 (F34/F64): a row the certificate rebuilds is counted; an interior row is not."""
    from vfe3.geometry.retraction import (_certify_public_spd, public_spd_repair_elements,
                                          reset_public_spd_repair_elements)

    reset_public_spd_repair_elements()
    healthy = torch.eye(3).expand(2, 3, 3).clone()
    _certify_public_spd(healthy, eps=1e-6, sigma_max=1e6)
    assert public_spd_repair_elements() == 0
    bad = healthy.clone()
    bad[1] = torch.diag(torch.tensor([1.0, 1.0, 1e-9]))                     # below the eps floor
    out = _certify_public_spd(bad, eps=1e-6, sigma_max=1e6)
    assert public_spd_repair_elements() == 1
    assert float(torch.linalg.eigvalsh(out[1].double()).min()) > 1e-6


def test_certified_numerics_ignore_ambient_bf16_autocast() -> None:
    """Item 26 (F32): the fp32-certified routines compute identically inside a bf16 autocast region."""
    from vfe3.numerics import apply_mu_trust_region, cholesky_qr2, validated_cholesky_solve

    torch.manual_seed(0)
    X = torch.randn(4, 6, 8)
    A = X @ X.transpose(-1, -2) + 0.1 * torch.eye(6)
    rhs = torch.randn(4, 6, 2)
    delta = torch.randn(4, 6) * 10.0
    plain = (validated_cholesky_solve(A, rhs), cholesky_qr2(X), apply_mu_trust_region(delta, A, is_diagonal=False))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        amp = (validated_cholesky_solve(A, rhs), cholesky_qr2(X), apply_mu_trust_region(delta, A, is_diagonal=False))
    for p, q in zip(plain[0], amp[0]):
        if torch.is_tensor(p):
            assert torch.equal(p, q)
    for p, q in zip(plain[1], amp[1]):
        assert torch.equal(p, q)
    assert torch.equal(plain[2], amp[2])


@pytest.mark.parametrize("diagonal", [True, False])
def test_saturated_mean_step_pins_at_the_trust_radius(diagonal: bool) -> None:
    """Item 53 (F35): an infinite step over a variance below 1/(4K) is pinned, not zeroed."""
    from vfe3.numerics import apply_mu_trust_region

    k, trust = 4, 5.0
    var = torch.full((k,), 1e-3)                                  # 1e-3 < 1/(4K) = 0.0625
    delta = torch.tensor([float("inf"), 0.0, 0.0, 0.0])
    sigma = var if diagonal else torch.diag(var)
    out = apply_mu_trust_region(delta, sigma, trust=trust, is_diagonal=diagonal)
    assert torch.isfinite(out).all() and float(out[0]) > 0
    whitened_norm = float((out / var.sqrt()).norm())
    assert abs(whitened_norm - trust) < 1e-3 * trust


def _gaussian_kl(mu_q, s_q, mu_p, s_p):
    from vfe3.families.gaussian import diag_kl_unclamped_per_coord

    return diag_kl_unclamped_per_coord(mu_q, s_q, mu_p, s_p, eps=1e-12)


@pytest.mark.parametrize("kl", [_gaussian_kl])
def test_float32_diagonal_kl_keeps_relative_accuracy_near_equality(kl) -> None:
    """Item 54 (F36): just outside the series band, the float32 KL matches float64 to roundoff."""
    torch.manual_seed(0)
    n = 512
    p = 0.5 + torch.rand(n, dtype=torch.float64) * 4.0
    rel = (1.5e-3 + 8e-3 * torch.rand(n, dtype=torch.float64)) * torch.where(torch.rand(n) < 0.5, -1.0, 1.0)
    q = p * (1.0 + rel)
    mu_p = torch.randn(n, dtype=torch.float64)
    mu_q = mu_p + 1e-3 * p.sqrt() * torch.randn(n, dtype=torch.float64)
    ref = kl(mu_q, q, mu_p, p)
    got = kl(mu_q.float(), q.float(), mu_p.float(), p.float()).double()
    # Reference from the float64 inputs rounded to float32, so only the evaluation error is measured.
    ref = kl(mu_q.float().double(), q.float().double(), mu_p.float().double(), p.float().double())
    rel_err = ((got - ref).abs() / ref.abs()).median()
    assert float(rel_err) < 1e-4                    # the two-log form gave about 2e-3 here


@pytest.mark.parametrize("kl", [_gaussian_kl])
def test_regrouped_diagonal_kl_matches_the_textbook_form_and_gradcheck(kl) -> None:
    import math

    torch.manual_seed(1)
    mu_q, mu_p = torch.randn(6, dtype=torch.float64), torch.randn(6, dtype=torch.float64)
    a = (0.5 + torch.rand(6, dtype=torch.float64)).requires_grad_()
    b = (0.5 + torch.rand(6, dtype=torch.float64)).requires_grad_()
    if kl is _gaussian_kl:
        textbook = 0.5 * (a / b - 1 + torch.log(b / a) + (mu_p - mu_q) ** 2 / b)
    else:
        s = (mu_q - mu_p).abs()
        textbook = torch.log(b / a) + s / b + (a / b) * torch.exp(-s / a) - 1.0
    assert torch.allclose(kl(mu_q, a, mu_p, b), textbook, rtol=1e-12, atol=1e-14)
    assert torch.autograd.gradcheck(lambda x, y: kl(mu_q, x, mu_p, y), (a, b))


def _manifold_optimizer(phi, other, fused: bool):
    from torch import nn  # noqa: F401
    from vfe3.gauge_optim import GaugeManifoldAdamW
    from vfe3.geometry.groups import get_group

    return GaugeManifoldAdamW(
        [{"params": [phi], "lr": 0.05, "pullback_group": True, "weight_decay": 0.0},
         {"params": [other], "lr": 0.01, "weight_decay": 0.0}],
        get_group("block_glk")(n_heads=1, K=2), phi_group_trust_radius=0.1, phi_chart_max_norm=5.0,
        phi_bch_residual_max=1e-6, weight_decay=0.0, fused=fused)


def test_fused_grad_scale_contract_reaches_the_manifold_phi_step() -> None:
    """Item 30 (F43): on GradScaler's fused path the phi step uses unscaled gradients and skips on overflow."""
    from torch import nn

    def run(scale, found_inf, fused):
        phi = nn.Parameter(torch.zeros(5, 4))
        other = nn.Parameter(torch.ones(3))
        opt = _manifold_optimizer(phi, other, fused)
        g = torch.zeros(5, 4)
        g[1, 0], g[3, 3] = 1.0, -2.0
        phi.grad = g * (1.0 if scale is None else scale)
        other.grad = torch.full((3,), 0.5) * (1.0 if scale is None else scale)
        if scale is not None:
            opt.grad_scale = torch.tensor(scale)
            opt.found_inf = torch.tensor(float(found_inf))
        opt.step()
        return phi.detach().clone(), other.detach().clone()

    ref_phi, ref_other = run(None, False, fused=True)              # gradients already unscaled
    phi, other = run(1024.0, False, fused=True)                     # fused hand-off: loss-scaled gradients
    assert torch.allclose(phi, ref_phi, rtol=0, atol=1e-6) and torch.allclose(other, ref_other, atol=1e-6)
    phi, other = run(1024.0, True, fused=True)                      # overflow step: nothing moves
    assert torch.equal(phi, torch.zeros(5, 4)) and torch.equal(other, torch.ones(3))


@pytest.mark.parametrize("untie", [True, False])
def test_finalize_fd_check_probes_a_live_prior_bank_decode_table(untie: bool) -> None:
    """Item 18 (F86): prior-bank decode probes the untied decode table, or says why it cannot."""
    import math

    from vfe3.run_artifacts import _fd_gradient_check

    cfg = _cfg(_minimal_agent(predictive_readout="kernel", block_mlp_mode="coordinate",
                              decode_mode="diagonal_chunked", untie_decode_bank=untie))
    torch.manual_seed(0)
    model = VFEModel(cfg).double()
    g = torch.Generator().manual_seed(7)
    batch = (torch.randint(0, V, (B, N), generator=g), torch.randint(0, V, (B, N), generator=g))
    record = {}
    worst = _fd_gradient_check(model, [batch], torch.device("cpu"), fd_eps=1e-5, record=record)
    if untie:
        assert record["fd_gradient_probe"] == "prior_bank.decode_mu_embed"
        assert math.isfinite(worst) and worst < 1e-4
    else:
        assert math.isnan(worst) and "tied" in record["fd_gradient_nan_reason"]


def test_ablation_reuse_survives_a_schema_only_config_change(tmp_path) -> None:
    """Item 21 (F07): a cell whose stored config differs only by a dropped retired field is still current."""
    import json
    from dataclasses import asdict

    import ablation
    from vfe3.config import serialize_config
    from vfe3.run_artifacts import semantic_config_fingerprint

    cfg = _cfg(_minimal_agent())
    current = semantic_config_fingerprint(asdict(cfg))
    stored_cfg = {**asdict(cfg), "mm_damping": 0.5}                 # a retired inert field, as an old build saved it
    stored = {"schema_version": 1, "semantic_config_fingerprint": semantic_config_fingerprint(stored_cfg),
              "dataset": "x"}
    expected = {**stored, "semantic_config_fingerprint": current}
    (tmp_path / "config.json").write_text(json.dumps({"config": {**serialize_config(cfg), "mm_damping": 0.5}}))
    migrated = ablation._schema_migrated_contract(tmp_path, stored, expected)
    assert migrated == expected
    # A changed effective value is not migrated.
    other = _cfg(_minimal_agent(kl_max=7.0))
    (tmp_path / "config.json").write_text(json.dumps({"config": serialize_config(other)}))
    assert ablation._schema_migrated_contract(tmp_path, stored, expected) is None


def test_phi_numerics_fp32_inverse_matches_the_live_pair_rule() -> None:
    """Item 46 (F65): the fp32 inverse residual is that of the float64 inverse of the float32 forward."""
    from vfe3.geometry.groups import get_group
    from vfe3.metrics import flatness_reference_statistics

    group = get_group("block_glk")(n_heads=1, K=4)
    torch.manual_seed(0)
    phi = 1.5 * torch.randn(6, group.generators.shape[0], dtype=torch.float64)
    stats = flatness_reference_statistics(phi, group)
    M = torch.einsum("na,akl->nkl", phi, group.generators.double())
    U32 = torch.linalg.matrix_exp(M.float())
    live = torch.linalg.inv(U32.double()).float()
    expected = torch.linalg.matrix_norm(U32 @ live - torch.eye(4), ord="fro").mean()
    separate = torch.linalg.matrix_norm(U32 @ torch.linalg.matrix_exp(-M.float()) - torch.eye(4), ord="fro").mean()
    assert abs(stats["inverse_consistency_fp32_abs"] - float(expected)) <= 1e-6 * max(float(expected), 1e-12) + 1e-12
    assert stats["inverse_consistency_fp32_abs"] < float(separate)


@pytest.mark.parametrize("inference", ["mean_field", "exact"])
def test_noself_prior_forced_entry_is_not_a_self_record(inference: str) -> None:
    """Item 51 (F29): under a self-masking prior, token 0 receives no record; plain causal is unchanged."""
    from vfe3.attention_prior import get_prior
    from vfe3.model.minimal_agent import minimal_agent_readout

    torch.manual_seed(0)
    n, k, h = 4, 4, 2
    xi = torch.randn(n, k, dtype=torch.float64)
    C = torch.eye(k, dtype=torch.float64).expand(n, k, k).clone()
    S = torch.randn(h, k // h, k, dtype=torch.float64)
    T = torch.randn(h, k // h, k, dtype=torch.float64)
    noise = torch.zeros(h, dtype=torch.float64)
    noself = get_prior("causal_noself")(n, n, dtype=torch.float64)
    causal = get_prior("causal")(n, n, dtype=torch.float64)
    out = minimal_agent_readout(xi, C, S, T, noise, noself, inference=inference)
    assert torch.equal(out.xi[0], xi[0]) and torch.equal(out.M[0], torch.zeros_like(out.M[0]))
    assert not torch.allclose(out.xi[1:], xi[1:])
    plain = minimal_agent_readout(xi, C, S, T, noise, causal, inference=inference)
    assert not torch.allclose(plain.xi[0], xi[0])


@pytest.mark.parametrize("kl", [_gaussian_kl])
def test_diagonal_kl_stays_finite_for_extreme_variance_ratios(kl) -> None:
    """A ratio below float32 resolution (x rounds to -1) keeps the KL and its gradient finite."""
    p = torch.tensor([5.5e34, 1e-20, 1.0], requires_grad=True)
    q = torch.tensor([0.6, 1.0, 1.0 + 2e-3], requires_grad=True)
    mu = torch.zeros(3)
    out = kl(mu, q, mu, p)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert torch.isfinite(p.grad).all() and torch.isfinite(q.grad).all()


@pytest.mark.parametrize("value", ["pullback_per_block", "none", "killing"])
def test_retired_phi_precond_mode_is_dropped_on_load(value: str) -> None:
    """Item 1 (F06): configs saved before PR #69 load strictly again; the key is dropped."""
    from vfe3.config import config_from_serialized, serialize_config

    payload = {**serialize_config(_cfg(_minimal_agent())), "phi_precond_mode": value}
    cfg = config_from_serialized(payload, source="pre-PR69 run")
    assert not hasattr(cfg, "phi_precond_mode")


def _prototype_cfg(**overrides):
    return _cfg(_minimal_agent(predictive_readout="kernel", block_mlp_mode="prototype_agents",
                               sigma_init=15.0, block_mlp_expansion=4, **overrides))


def _block_inputs(model, ids):
    from vfe3.model.block_mlp import PrototypeAgentBlockMLP

    seen = []
    for mlp in model.block_mlps:
        if isinstance(mlp, PrototypeAgentBlockMLP):
            original = mlp.forward_moments

            def rec(mu, sigma, *a, _o=original, **k):
                seen.append((mu.detach(), sigma.detach()))
                return _o(mu, sigma, *a, **k)
            mlp.forward_moments = rec
    with torch.no_grad():
        model.forward_beliefs(ids, return_logits=False)
    for mlp in model.block_mlps:
        mlp.__dict__.pop("forward_moments", None)
    return seen


def test_working_variance_template_init_sets_the_target_pull_per_layer() -> None:
    """Items 14 and 15 (F77/F78): lambda_0 follows the measured working variance; nu sits at input scale."""
    import math

    cfg = _prototype_cfg(block_mlp_template_init="working_variance", block_mlp_template_init_pull=0.09)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    g = torch.Generator().manual_seed(3)
    ids = torch.randint(0, V, (B, N), generator=g)
    stats = model.calibrate_prototype_templates(ids, pull=0.09, seed=1)
    assert len(stats) == cfg.n_layers
    inputs = _block_inputs(model, ids)
    for (mu, sigma), mlp, st in zip(inputs, model.block_mlps, stats):
        c = sigma.double()
        lam = float(mlp.log_lambda.detach().exp().mean())
        assert math.isclose(lam, st["lambda0"], rel_tol=1e-5)
        assert math.isclose(st["lambda0"], st["working_variance"] * 0.91 / 0.09, rel_tol=1e-6)
        pull = float((c / (c + lam)).mean())                      # gain at uniform routing, per coordinate
        assert abs(pull - 0.09) < 0.03
        assert float(mlp.nu.detach().norm(dim=-1).mean()) > 0.5 * float(mu.norm(dim=-1).mean())
    # The absolute default is untouched: no calibration, lambda = block_mlp_template_init_variance.
    absolute = VFEModel(_prototype_cfg())
    assert torch.allclose(absolute.block_mlps[0].log_lambda.exp(), torch.tensor(10.0))


@pytest.mark.parametrize("init,calls", [("working_variance", 1), ("absolute", 0)])
def test_train_calibrates_only_on_a_fresh_working_variance_start(monkeypatch: pytest.MonkeyPatch,
                                                                 init: str, calls: int) -> None:
    import logging

    from torch.utils.data import DataLoader

    import vfe3.train as T
    from vfe3.data.datasets import TokenWindows

    cfg = _prototype_cfg(block_mlp_template_init=init, max_steps=1, eval_stride=None)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    seen = []
    original = VFEModel.calibrate_prototype_templates
    monkeypatch.setattr(VFEModel, "calibrate_prototype_templates",
                        lambda self, *a, **k: seen.append(1) or original(self, *a, **k))
    loader = DataLoader(TokenWindows(torch.arange(41) % V, N), batch_size=B, shuffle=False)
    T.train(model, loader, cfg, n_steps=1, log_interval=0, eval_interval=0, device=torch.device("cpu"),
            logger=logging.getLogger("calibration-test"), generate_samples=False)
    assert len(seen) == calls


@pytest.mark.parametrize("theta", [2.95, 3.10, 3.5, 4.5])
def test_offstrip_frame_rows_recover_the_exact_covector_and_chart_velocity(theta: float) -> None:
    """Item 24 (F21): blocks inside the strip margin or beyond it are solved exactly, not frozen."""
    from vfe3.geometry.phi_preconditioner import uspace_ambient_gradient, uspace_chart_velocity

    torch.manual_seed(0)
    X = torch.tensor([[0.0, -theta], [theta, 0.0]], dtype=torch.float64) + 0.05 * torch.randn(2, 2, dtype=torch.float64)
    X = X[None]
    G_true = torch.randn(1, 2, 2, dtype=torch.float64)
    X_req = X.clone().requires_grad_()
    (G_true * torch.linalg.matrix_exp(X_req)).sum().backward()
    M = X_req.grad                                                     # chart covector of <G, exp(X)>
    U = torch.linalg.matrix_exp(X)
    certified = torch.zeros(1, dtype=torch.bool)                       # as the strip certificate rejects it
    G_bar, ok = uspace_ambient_gradient(X, M, U=U, certified=certified, offstrip=True)
    assert bool(ok.all())
    assert torch.allclose(G_bar, G_true, rtol=1e-8, atol=1e-10)
    _, ok_default = uspace_ambient_gradient(X, M, U=U, certified=certified)
    assert not bool(ok_default.any())                                  # pullback callers keep their behavior
    W = torch.randn(1, 2, 2, dtype=torch.float64)
    v, v_ok = uspace_chart_velocity(X, W, U=U, certified=certified, offstrip=True)
    assert bool(v_ok.all())
    _, dU = torch.func.jvp(torch.linalg.matrix_exp, (X,), (v,))
    assert torch.allclose(dU, W @ U, rtol=1e-8, atol=1e-10)


def _rope_cfg(space: str, **overrides):
    return _cfg(_minimal_agent(predictive_readout="kernel", block_mlp_mode="coordinate", rope_mode=space,
                               **overrides))


def test_comparison_space_rope_routes_on_relative_position() -> None:
    """Item 25 (F28): under rope_mode='comparison' the routing cost depends on positions only via i - j."""
    import vfe3.gradients.relational as rel

    cfg = _rope_cfg("comparison")
    torch.manual_seed(0)
    model = VFEModel(cfg).double()
    ops = model.relational_operators
    assert ops.comparison_rope_base == cfg.rope_base
    # A sequence of identical beliefs: every pair cost must then depend only on j - i.
    n, k = 6, cfg.embed_dim
    mu = torch.randn(1, 1, k, dtype=torch.float64).expand(1, n, k)
    sigma = (0.5 + torch.rand(1, 1, k, dtype=torch.float64)).expand(1, n, k)
    from vfe3.contracts import CanonicalFrameContext
    d = k // cfg.n_heads
    eye = torch.eye(d, dtype=torch.float64).expand(1, n, cfg.n_heads, d, d).clone()
    frame = CanonicalFrameContext(forward=eye, inverse=eye.clone(), block_diagonal=True)
    cost = rel.relational_score(mu, sigma, mu, sigma, frame, ops)              # (1, H, n, n)
    for offset in range(-n + 1, n):
        diag = torch.diagonal(cost[0], offset=offset, dim1=-2, dim2=-1)        # (H, n - |offset|)
        assert torch.allclose(diag, diag[..., :1].expand_as(diag), rtol=1e-9, atol=1e-10), offset
    assert not torch.allclose(cost[0, :, 1, 0], cost[0, :, 2, 0])            # the rotation does act
    assert VFEModel(_rope_cfg("frame")).relational_operators.comparison_rope_base is None


def test_comparison_space_rope_trains_a_tiny_step_and_refuses_unsupported_routes() -> None:
    _loss_and_grads({**_minimal_agent(predictive_readout="kernel", block_mlp_mode="coordinate",
                                      rope_mode="comparison")})
    with pytest.raises(ValueError, match="rope_mode='comparison'"):
        _cfg(_minimal_agent(predictive_readout="kernel", block_mlp_mode="coordinate", rope_mode="comparison",
                            attention_mode="projected_gradient", attention_residual=False))


@pytest.mark.parametrize("group_cfg", [
    dict(gauge_group="block_so"), dict(gauge_group="block_co"), dict(gauge_group="block_glmc"),
    dict(gauge_group="block_glk", gauge_tying="shared"),
    dict(gauge_group="so_n", embed_dim=6, n_heads=2, group_n=3, irrep_spec=[("l1", 2)]),
])
def test_pullback_group_frame_step_reaches_every_registered_group(group_cfg) -> None:
    """Item 55 (F47): an opt-in geometric frame M-step exists for every group and tying."""
    from vfe3.gauge_optim import GaugeManifoldAdamW
    from vfe3.train import build_optimizer, train_step

    values = dict(vocab_size=12, embed_dim=4, n_heads=2, max_seq_len=8, n_layers=1, pos_phi="none",
                  m_phi_update_mode="pullback_group", m_phi_group_retraction="chart",
                  transport_chart_max_norm=6.0, m_phi_lr=0.1, rope_mode="none")
    values.update(group_cfg)
    cfg = _cfg(values)
    torch.manual_seed(0)
    model = VFEModel(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    phi_before = model.prior_bank.phi_embed.detach().clone()
    tokens = torch.randint(0, 12, (4, 8))
    targets = torch.randint(0, 12, (4, 8))
    loss = train_step(model, optimizer, scheduler, tokens, targets, grad_clip=1.0)
    assert isinstance(optimizer, GaugeManifoldAdamW)
    assert torch.isfinite(torch.tensor(float(loss)))
    assert not torch.equal(model.prior_bank.phi_embed.detach(), phi_before)


@pytest.mark.parametrize("group_name", ["block_so", "block_glmc"])
def test_pullback_group_diagnostic_step_on_a_non_elementary_basis(group_name: str) -> None:
    """The collect path compares dense elements for groups staged densely with a per-block direction."""
    from vfe3.gauge_optim import GaugeManifoldAdamW
    from vfe3.geometry.groups import get_group
    from torch import nn

    group = get_group(group_name)(K=4, n_heads=2)
    torch.manual_seed(0)
    phi = nn.Parameter(0.2 * torch.randn(5, group.generators.shape[0]))
    optimizer = GaugeManifoldAdamW([{"params": [phi], "lr": 0.05, "pullback_group": True, "weight_decay": 0.0}],
                                   group, phi_group_trust_radius=0.1, phi_chart_max_norm=5.0,
                                   phi_bch_residual_max=1e-6, weight_decay=0.0)
    before = phi.detach().clone()
    phi.grad = torch.randn_like(phi)
    optimizer._collect_gauge_diag = True
    optimizer.step()
    assert not torch.equal(phi.detach(), before)
    assert optimizer._gauge_diag["phi_frame_displacement_max"] > 0.0
