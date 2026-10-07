from tests.profiles import test_config
import inspect

import pytest
import torch
from torch import nn

from vfe3.geometry.groups import get_group
from vfe3.geometry.phi_preconditioner import PullbackGroupDirectionResult
from vfe3.gauge_optim import (
    GaugeManifoldAdamW,
    PhiUpdatePolicy,
    _PHI_UPDATE_POLICIES,
    get_phi_update_policy,
    stage_pullback_group_candidate,
)


def _make_optimizer(
    param_groups,
    group,
    *,
    trust_radius=0.1,
    chart_max_norm=5.0,
    bch_residual_max=1e-6,
    **kwargs,
):
    return GaugeManifoldAdamW(
        param_groups,
        group,
        phi_group_trust_radius=trust_radius,
        phi_chart_max_norm=chart_max_norm,
        phi_bch_residual_max=bch_residual_max,
        weight_decay=0.0,
        **kwargs,
    )


def _fixed_direction(xi: torch.Tensor) -> PullbackGroupDirectionResult:
    rows = xi.shape[:-1]
    ones = torch.ones(rows, dtype=torch.float64, device=xi.device)
    zeros = torch.zeros(rows, dtype=torch.float64, device=xi.device)
    return PullbackGroupDirectionResult(
        v_phi=xi.double(),
        xi=xi.double(),
        min_undamped_generalized_eigenvalue=ones,
        undamped_generalized_condition=ones,
        damped_generalized_condition=ones,
        scaled_solve_residual=zeros,
        series_order=40,
    )


def test_phi_update_policy_registry_contract_is_exact_and_immutable():
    assert set(_PHI_UPDATE_POLICIES) == {"adamw", "pullback_group", "fisher_group"}

    adamw = get_phi_update_policy("adamw")
    pullback = get_phi_update_policy("pullback_group")
    assert isinstance(adamw, PhiUpdatePolicy)
    assert dict(adamw.optimizer_group_metadata) == {}
    assert adamw.requires_manifold_optimizer is False
    assert adamw.requires_pullback_geometry is False
    assert dict(pullback.optimizer_group_metadata) == {
        "pullback_group": True,
        "weight_decay": 0.0,
    }
    assert pullback.requires_manifold_optimizer is True
    assert pullback.requires_pullback_geometry is True
    with pytest.raises(TypeError):
        pullback.optimizer_group_metadata["weight_decay"] = 1.0
    assert dict(get_phi_update_policy("pullback_group").optimizer_group_metadata) == {
        "pullback_group": True,
        "weight_decay": 0.0,
    }


def test_default_adamw_one_step_matches_direct_torch_optimizer() -> None:
    from vfe3.model.model import VFEModel
    from vfe3.train import build_optimizer, train_step

    torch.manual_seed(0)
    cfg = test_config(vocab_size=7, embed_dim=4, n_heads=2, max_seq_len=4,
                      n_layers=1, n_e_steps=1, pos_phi="none")
    production = VFEModel(cfg)
    oracle = VFEModel(cfg)
    oracle.load_state_dict(production.state_dict())
    optimizer = build_optimizer(production, cfg)
    assert type(optimizer) is torch.optim.AdamW
    parameter_names = {id(p): name for name, p in production.named_parameters()}
    oracle_parameters = dict(oracle.named_parameters())
    groups = [dict(group, params=[oracle_parameters[parameter_names[id(p)]]
                                 for p in group["params"]])
              for group in optimizer.param_groups]
    accepted = inspect.signature(torch.optim.AdamW).parameters
    defaults = {name: value for name, value in optimizer.defaults.items() if name in accepted}
    reference = torch.optim.AdamW(groups, **defaults)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    tokens = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
    targets = torch.tensor([[1, 2, 3, 4], [2, 1, 0, 6]])

    _, reference_loss, _ = oracle(tokens, targets)
    reference_loss.backward()
    roles: dict[str, list[torch.Tensor]] = {}
    for group in reference.param_groups:
        roles.setdefault(group["role"], []).extend(group["params"])
    for parameters in roles.values():
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    reference.step()
    loss = train_step(production, optimizer, scheduler, tokens, targets, grad_clip=1.0)

    assert loss == float(reference_loss.detach())
    trainable = {id(p) for p in production.parameters() if p.requires_grad}
    assert trainable == {id(p) for g in optimizer.param_groups for p in g["params"]}
    for name, value in production.named_parameters():
        torch.testing.assert_close(value, oracle_parameters[name], rtol=0.0, atol=0.0)
    for state in optimizer.state.values():
        assert set(state) == {"step", "exp_avg", "exp_avg_sq"}


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("trust_radius", 0.0),
        ("trust_radius", float("nan")),
        ("chart_max_norm", -1.0),
        ("chart_max_norm", float("inf")),
        ("bch_residual_max", 0.0),
        ("bch_residual_max", float("nan")),
    ],
)
def test_direct_optimizer_rejects_invalid_phi_bounds(name, value):
    group = get_group("block_glk")(n_heads=1, K=2)
    phi = nn.Parameter(torch.zeros(1, 4))
    kwargs = {name: value}
    with pytest.raises(ValueError, match="finite and positive"):
        _make_optimizer(
            [{"params": [phi], "lr": 0.1, "pullback_group": True, "weight_decay": 0.0}],
            group,
            **kwargs,
        )


def test_pullback_group_step_moves_only_active_rows_and_creates_no_phi_state():
    group = get_group("block_glk")(n_heads=1, K=2)
    phi = nn.Parameter(torch.zeros(5, 4))
    optimizer = _make_optimizer(
        [{"params": [phi], "lr": 0.05, "pullback_group": True, "weight_decay": 0.0}],
        group,
    )
    before = phi.detach().clone()
    phi.grad = torch.zeros_like(phi)
    phi.grad[1, 0] = 1.0
    phi.grad[3, 3] = -2.0

    optimizer.step()

    assert torch.equal(phi.detach()[[0, 2, 4]], before[[0, 2, 4]])
    assert not torch.equal(phi.detach()[1], before[1])
    assert not torch.equal(phi.detach()[3], before[3])
    assert phi.grad is None
    assert phi not in optimizer.state
    assert optimizer.state_dict()["state"] == {}


def test_build_optimizer_selects_pullback_group_route_and_moves_phi_embed():
    from vfe3.model.model import VFEModel
    from vfe3.train import build_optimizer, train_step

    cfg = test_config(
        vocab_size=12,
        embed_dim=4,
        n_heads=2,
        max_seq_len=8,
        n_layers=1,
        gauge_group="block_glk",
        pos_phi="none",
        m_phi_update_mode="pullback_group",
        transport_chart_max_norm=6.0,
        m_phi_lr=0.1,
    )
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
    assert model.prior_bank.phi_embed not in optimizer.state


def test_pullback_group_trust_scales_the_learning_rate_weighted_right_factor(monkeypatch):
    import vfe3.gauge_optim as gauge_optim

    group = get_group("block_glk")(n_heads=1, K=2, dtype=torch.float64)
    xi = torch.tensor([[4.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    monkeypatch.setattr(
        gauge_optim,
        "pullback_group_direction",
        lambda *args, **kwargs: _fixed_direction(xi),
    )

    candidate = stage_pullback_group_candidate(
        torch.ones_like(xi),
        torch.zeros_like(xi),
        group,
        learning_rate=0.5,
        trust_radius=0.25,
        chart_max_norm=5.0,
        bch_residual_max=1e-6,
    )

    assert candidate.trust_scale.item() == pytest.approx(0.125)
    assert candidate.backtracking_reductions.item() == 0
    assert candidate.candidate_chart_norm.item() == pytest.approx(0.25)
    assert torch.allclose(
        candidate.candidate_phi,
        torch.tensor([[-0.25, 0.0, 0.0, 0.0]], dtype=torch.float64),
        atol=1e-15,
        rtol=0.0,
    )


def test_pullback_group_stage_passes_source_basis_and_registered_coordinate_layout(monkeypatch, device):
    import vfe3.gauge_optim as gauge_optim

    group = get_group("block_glk")(K=4, n_heads=2, dtype=torch.float32, device=device)
    seen = {}

    def record_direction(grad, phi, generators, **kwargs):
        seen["generators"] = generators
        seen["coordinate_layout"] = kwargs.get("coordinate_layout")
        return _fixed_direction(torch.zeros_like(grad))

    monkeypatch.setattr(gauge_optim, "pullback_group_direction", record_direction)
    stage_pullback_group_candidate(
        torch.ones(1, group.generators.shape[0], device=device),
        torch.zeros(1, group.generators.shape[0], device=device),
        group,
        learning_rate=0.0,
        trust_radius=0.25,
        chart_max_norm=5.0,
        bch_residual_max=1e-6,
    )

    assert seen["generators"] is group.generators
    assert seen["coordinate_layout"] == "block_head_row_major"


def test_pullback_group_stage_normalizes_basis_for_registered_extension(device):
    import vfe3.geometry.phi_preconditioner as phi_preconditioner

    name = "test_stage_normalized_registered_extension"
    group = get_group("block_glk")(n_heads=1, K=2, dtype=torch.float32, device=device)
    group.pullback_capability = name
    seen = {}

    @phi_preconditioner.register_phi_group_direction(name)
    def registered_direction(grad, phi, generators, *, irrep_dims=None, **kwargs):
        del phi, irrep_dims, kwargs
        seen["generators"] = generators
        return _fixed_direction(torch.zeros_like(grad))

    try:
        stage_pullback_group_candidate(
            torch.ones(1, group.generators.shape[0], device=device),
            torch.zeros(1, group.generators.shape[0], device=device),
            group,
            learning_rate=0.0,
            trust_radius=0.25,
            chart_max_norm=5.0,
            bch_residual_max=1e-6,
        )
    finally:
        for support_name in (
            "_PHI_GROUP_DIRECTIONS",
            "_PHI_GROUP_DIRECTION_ACCEPTS_COORDINATE_LAYOUT",
            "_PHI_GROUP_DIRECTION_REQUIRES_SOURCE_BASIS",
        ):
            support = getattr(phi_preconditioner, support_name, None)
            if support is not None:
                support.pop(name, None)

    assert seen["generators"].dtype == torch.float64
    assert seen["generators"].device == group.generators.device
    assert seen["generators"] is not group.generators


@pytest.mark.parametrize("chart_norm", [3.0, 5.0])
def test_pullback_group_chart_retraction_certifies_right_product_order(
    record_property,
    chart_norm: float,
) -> None:
    r"""A real pullback direction retracts toward exp(phi) exp(-eta xi)."""
    case = _REAL_COMMIT_CASES[chart_norm]
    group = get_group("block_glk")(n_heads=1, K=2, dtype=torch.float64)
    phi = torch.tensor([case["phi"]], dtype=torch.float64, requires_grad=True)
    objective_weights = torch.tensor(case["delta"], dtype=torch.float64).reshape(2, 2)
    objective = (torch.linalg.matrix_exp(phi.reshape(2, 2)) * objective_weights).sum()
    grad_phi, = torch.autograd.grad(objective, phi)
    candidate = stage_pullback_group_candidate(
        grad_phi,
        phi.detach(),
        group,
        learning_rate=1.0,
        trust_radius=1.0,
        chart_max_norm=6.0,
        bch_residual_max=1e-6,
    )

    factor_scale = candidate.trust_scale / torch.exp2(candidate.backtracking_reductions.double())
    accepted_delta = -candidate.direction.xi * factor_scale.unsqueeze(-1)
    exp_candidate = torch.linalg.matrix_exp(candidate.candidate_phi.reshape(1, 2, 2))
    exp_phi = torch.linalg.matrix_exp(phi.detach().reshape(1, 2, 2))
    exp_delta = torch.linalg.matrix_exp(accepted_delta.reshape(1, 2, 2))
    right = exp_phi @ exp_delta
    reversed_product = exp_delta @ exp_phi
    actual_right = torch.linalg.matrix_norm(exp_candidate - right) / torch.linalg.matrix_norm(right)
    actual_reversed = (
        torch.linalg.matrix_norm(exp_candidate - reversed_product)
        / torch.linalg.matrix_norm(reversed_product)
    )
    assert phi.detach().norm().item() == pytest.approx(chart_norm, abs=1e-12)
    assert (grad_phi * candidate.direction.v_phi).sum().item() > 0.0
    assert candidate.group_product_residual.item() <= 1.0e-6
    assert candidate.group_product_residual.item() == pytest.approx(actual_right.item(), abs=1e-14)
    # The reversed product has a first-order commutator error; compare above roundoff only.
    assert actual_reversed.item() > 1e-10
    assert actual_right.item() < actual_reversed.item()
    torch.testing.assert_close(
        candidate.candidate_phi,
        phi.detach() - factor_scale.unsqueeze(-1) * candidate.direction.v_phi,
        rtol=0.0, atol=0.0,
    )
    record_property("current_chart_norm", chart_norm)
    record_property("float64_staging_residual", float(actual_right))


_REAL_COMMIT_CASES = {
    3.0: {
        "phi": [
            0.4461370124222645,
            -0.37866876575372294,
            1.1203006449654396,
            -2.7207532407183694,
        ],
        "delta": [
            0.0075751275839514645,
            -0.002453059541722277,
            -0.003516904099976695,
            -0.002529014357447788,
        ],
    },
    5.0: {
        "phi": [
            0.19176642751290957,
            -1.411275433700944,
            3.477796870171524,
            -3.297947273280199,
        ],
        "delta": [
            0.008133453045635446,
            0.0015903479064029997,
            -0.0025065446684178387,
            -0.001781968201968022,
        ],
    },
}


def _real_optimizer_commit_case(
    monkeypatch,
    chart_norm: float,
) -> dict[str, object]:
    import vfe3.gauge_optim as gauge_optim

    case = _REAL_COMMIT_CASES[chart_norm]
    group = get_group("block_glk")(n_heads=1, K=2, dtype=torch.float32)
    initial = torch.zeros(2, 4, dtype=torch.float32)
    initial[0] = torch.tensor(case["phi"], dtype=torch.float32)
    parameter = nn.Parameter(initial.clone())
    inactive_before = parameter.detach()[1].clone()
    objective_weights = torch.tensor(case["delta"], dtype=torch.float32).reshape(2, 2)
    objective = (torch.linalg.matrix_exp(parameter[0].reshape(2, 2)) * objective_weights).sum()
    objective.backward()
    staged = []
    original_stage = gauge_optim.stage_pullback_group_candidate

    def _stage_spy(*args, **kwargs) -> object:
        candidate = original_stage(*args, **kwargs)
        staged.append(candidate)
        return candidate

    monkeypatch.setattr(gauge_optim, "stage_pullback_group_candidate", _stage_spy)
    optimizer = _make_optimizer(
        [{"params": [parameter], "lr": 1.0, "pullback_group": True, "weight_decay": 0.0}],
        group,
        trust_radius=1.0,
        chart_max_norm=6.0,
    )
    optimizer.step()

    assert len(staged) == 1
    candidate = staged[0]
    assert candidate.candidate_phi.dtype == torch.float64
    stored = parameter.detach()[0]
    staged_cast = candidate.candidate_phi[0].to(dtype=parameter.dtype)
    factor_scale = candidate.trust_scale / torch.exp2(candidate.backtracking_reductions.double())
    right_coordinates = -candidate.direction.xi * factor_scale.unsqueeze(-1)
    stored_exp = torch.linalg.matrix_exp(stored.double().reshape(1, 2, 2))
    initial_exp = torch.linalg.matrix_exp(initial[0].double().reshape(1, 2, 2))
    right_factor_exp = torch.linalg.matrix_exp(right_coordinates.reshape(1, 2, 2))
    right_product = initial_exp @ right_factor_exp
    reversed_product = right_factor_exp @ initial_exp
    right_residual = (
        torch.linalg.matrix_norm(stored_exp - right_product)
        / torch.linalg.matrix_norm(right_product)
    )
    reversed_residual = (
        torch.linalg.matrix_norm(stored_exp - reversed_product)
        / torch.linalg.matrix_norm(reversed_product)
    )
    return {
        "stored_matches_staged_cast": torch.equal(stored, staged_cast),
        "stored_matches_certified": torch.equal(stored.double(), candidate.candidate_phi[0]),
        "inactive_unchanged": torch.equal(parameter.detach()[1], inactive_before),
        "right_residual": float(right_residual),
        "staged_residual": candidate.group_product_residual.item(),
        "reversed_residual": float(reversed_residual),
        "observed_chart_norm": float(torch.linalg.vector_norm(initial[0].double())),
    }


@pytest.mark.parametrize("chart_norm", [3.0, 5.0])
def test_optimizer_commit_is_single_float32_cast_with_certified_right_product(
    monkeypatch,
    record_property,
    chart_norm: float,
) -> None:
    committed = _real_optimizer_commit_case(monkeypatch, chart_norm)
    record_property("current_chart_norm", committed["observed_chart_norm"])
    record_property("float32_committed_residual", committed["right_residual"])
    record_property("float32_committed_reversed_residual", committed["reversed_residual"])
    assert committed["observed_chart_norm"] == pytest.approx(chart_norm, abs=2.0e-7)
    assert committed["stored_matches_staged_cast"]
    assert committed["stored_matches_certified"]
    assert committed["inactive_unchanged"]
    assert committed["right_residual"] <= 1.0e-6
    assert committed["staged_residual"] == pytest.approx(committed["right_residual"], abs=1e-14)
    assert committed["reversed_residual"] > 1e-10
    assert committed["right_residual"] < committed["reversed_residual"]


def test_pullback_group_candidate_rejects_chart_bound(monkeypatch):
    import vfe3.gauge_optim as gauge_optim

    group = get_group("block_glk")(n_heads=1, K=2, dtype=torch.float64)
    phi = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    xi = torch.tensor([[-0.2, 0.0, 0.0, 0.0]], dtype=torch.float64)
    monkeypatch.setattr(
        gauge_optim,
        "pullback_group_direction",
        lambda *args, **kwargs: _fixed_direction(xi),
    )

    # Audit 2026-09-23 #10: a row whose every backtracked candidate leaves the chart is rejected
    # (it keeps its chart) instead of aborting the whole step.
    candidate = stage_pullback_group_candidate(
        -torch.ones_like(phi),
        phi,
        group,
        learning_rate=1.0,
        trust_radius=1.0,
        chart_max_norm=1.0,
        bch_residual_max=1e-6,
    )
    assert candidate.rejected_rows.tolist() == [True]
    assert torch.equal(candidate.candidate_phi, phi)


def test_pullback_group_step_rejects_two_parameter_groups_atomically(monkeypatch):
    import vfe3.gauge_optim as gauge_optim

    group = get_group("block_glk")(n_heads=1, K=2, dtype=torch.float64)
    good = nn.Parameter(torch.zeros(1, 4, dtype=torch.float64))
    bad = nn.Parameter(torch.tensor([[0.99, 0.0, 0.0, 0.0]], dtype=torch.float64))
    xi = torch.tensor([[-0.2, 0.0, 0.0, 0.0]], dtype=torch.float64)
    monkeypatch.setattr(
        gauge_optim,
        "pullback_group_direction",
        lambda *args, **kwargs: _fixed_direction(xi),
    )
    optimizer = _make_optimizer(
        [
            {"params": [good], "lr": 1.0, "pullback_group": True, "weight_decay": 0.0},
            {"params": [bad], "lr": 1.0, "pullback_group": True, "weight_decay": 0.0},
        ],
        group,
        trust_radius=1.0,
        chart_max_norm=1.0,
    )
    good_before = good.detach().clone()
    bad_before = bad.detach().clone()
    good.grad = torch.ones_like(good)
    bad.grad = torch.ones_like(bad)

    # The atomicity guard reports the failing product residual / descent check rather than the
    # chart norm it is reached through; both are the same stored-chart retraction failure.
    with pytest.raises(FloatingPointError, match="stored chart retraction failed"):
        optimizer.step()

    assert torch.equal(good, good_before)
    assert torch.equal(bad, bad_before)
    assert good.grad is not None and bad.grad is not None
    assert optimizer.state_dict()["state"] == {}


@pytest.mark.parametrize(
    "bad_value",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "positive_inf", "negative_inf"],
)
def test_pullback_group_step_rejects_nonfinite_covector_before_any_staging_or_mutation(
    monkeypatch,
    bad_value,
):
    import vfe3.gauge_optim as gauge_optim

    group = get_group("block_glk")(n_heads=1, K=2, dtype=torch.float64)
    finite = nn.Parameter(torch.zeros(1, 4, dtype=torch.float64))
    mixed = nn.Parameter(torch.zeros(2, 4, dtype=torch.float64))
    optimizer = _make_optimizer(
        [
            {"params": [finite], "lr": 0.1, "pullback_group": True, "weight_decay": 0.0},
            {"params": [mixed], "lr": 0.1, "pullback_group": True, "weight_decay": 0.0},
        ],
        group,
    )
    finite.grad = torch.ones_like(finite)
    mixed.grad = torch.zeros_like(mixed)
    mixed.grad[0] = 1.0
    mixed.grad[1, 0] = bad_value
    parameter_before = [parameter.detach().clone() for parameter in (finite, mixed)]
    gradient_before = [parameter.grad.clone() for parameter in (finite, mixed)]
    staging_calls = []
    original_stage = gauge_optim.stage_pullback_group_candidate

    def _stage_spy(*args, **kwargs):
        staging_calls.append(1)
        return original_stage(*args, **kwargs)

    monkeypatch.setattr(gauge_optim, "stage_pullback_group_candidate", _stage_spy)

    with pytest.raises(FloatingPointError, match="nonfinite phi covector"):
        optimizer.step()

    assert staging_calls == []
    for parameter, expected_parameter, expected_gradient in zip(
        (finite, mixed),
        parameter_before,
        gradient_before,
    ):
        assert torch.equal(parameter, expected_parameter)
        assert parameter.grad is not None
        torch.testing.assert_close(
            parameter.grad,
            expected_gradient,
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )
    assert optimizer.state_dict()["state"] == {}


def test_zero_phi_gradient_is_consumed_without_empty_diagnostics():
    from vfe3.geometry.groups import get_group

    group = get_group("block_glk")(n_heads=1, K=2)
    phi = nn.Parameter(torch.zeros(5, 4))
    optimizer = _make_optimizer(
        [{"params": [phi], "lr": 0.05, "pullback_group": True, "weight_decay": 0.0}],
        group,
    )
    optimizer._collect_gauge_diag = True
    optimizer._gauge_diag = {"phi_pullback_damped_gen_cond_max": 123.0}
    before = phi.detach().clone()
    phi.grad = torch.zeros_like(phi)

    optimizer.step()

    assert torch.equal(phi, before)
    assert phi.grad is None
    assert optimizer._gauge_diag == {}
    assert not optimizer.state


def test_attempted_diagnostic_step_clears_stale_gauge_health():
    group = get_group("block_glk")(K=4, n_heads=2)
    phi = nn.Parameter(torch.zeros(3, group.generators.shape[0]))
    opt = _make_optimizer(
        [{"params": [phi], "lr": 0.0, "pullback_group": True, "weight_decay": 0.0}],
        group,
    )
    opt._collect_gauge_diag = True
    opt._gauge_diag = {"phi_pullback_damped_gen_cond_max": 123.0}
    phi.grad = None

    opt.step()

    assert opt._gauge_diag == {}
