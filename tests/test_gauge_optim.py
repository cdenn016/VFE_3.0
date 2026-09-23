import hashlib
from collections.abc import Callable

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
    phi_precond_mode="pullback",
    **kwargs,
):
    return GaugeManifoldAdamW(
        param_groups,
        group,
        phi_group_trust_radius=trust_radius,
        phi_chart_max_norm=chart_max_norm,
        phi_bch_residual_max=bch_residual_max,
        phi_precond_mode=phi_precond_mode,
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
    assert set(_PHI_UPDATE_POLICIES) == {"adamw", "pullback_group"}

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


def test_default_adamw_one_step_is_byte_identical_to_golden():
    from vfe3.config import VFE3Config
    from vfe3.model.model import VFEModel
    from vfe3.train import build_optimizer, train_step

    torch.manual_seed(0)
    # The golden was captured under the library defaults in force before the imported config WIP
    # (docs/audits/audit-2026-09-22.md flip table); pin them so the parameter groups, optimizer
    # state and loss are those of the captured step. The block MLP, decode bias and affine block
    # LayerNorm each add a parameter group under the current defaults.
    cfg = VFE3Config(
        vocab_size=8,
        embed_dim=4,
        n_heads=2,
        max_seq_len=4,
        n_layers=1,
        n_e_steps=1,
        pos_phi="none",
        use_block_mlp=False,
        norm_type_block="none",
        layernorm_affine=False,
        decode_bias=False,
        untie_decode_bank=False,
        decode_unigram_prior=False,
        skip_belief_sigma_update=False,
        beta_attention_prior="causal",
        decode_av_precision="fp32",
    )
    model = VFEModel(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    tokens = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=torch.long)
    targets = torch.tensor([[1, 2, 3, 4], [2, 1, 0, 7]], dtype=torch.long)

    loss = train_step(model, optimizer, scheduler, tokens, targets, grad_clip=1.0)

    assert type(optimizer) is torch.optim.AdamW
    assert [group["role"] for group in optimizer.param_groups] == ["mu", "sigma", "phi", "mu"]
    state = optimizer.state_dict()["state"]
    assert sorted(state) == [0, 1, 3, 4]
    assert all(set(slot) == {"step", "exp_avg", "exp_avg_sq"} for slot in state.values())
    assert 2 not in state
    assert loss == 2.0770487785339355

    def digest(tensor: torch.Tensor) -> str:
        data = tensor.detach().cpu().contiguous().numpy().tobytes()
        return hashlib.sha256(data).hexdigest()

    # These digests are UNCHANGED by audit 2026-07-25 F3, and that is a deliberate property worth
    # stating. F3 makes the factored diagonal congruence escalate to float64 when its (mixed-sign,
    # non-manifestly-nonnegative) regrouping actually loses nonnegativity. A well-conditioned step
    # like this one never triggers the escalation, so it still runs entirely in float32 and remains
    # bit-identical. An UNCONDITIONAL float64 upcast would have moved every digest here while leaving
    # `loss` bit-identical -- same forward value, different low-order gradient bits -- which is
    # precisely the golden churn the data-driven escalation avoids.
    assert digest(model.prior_bank.mu_embed) == "75c40d1b09a080c392223ff5160794c886e371193b7ba79a18fcab05febfd288"
    assert digest(model.prior_bank.sigma_log_embed) == "6f5f7fe389ec74bc1a27b0a162defce8b50aedc1ebbcd601bcfea12442c1cb33"
    assert digest(model.prior_bank.phi_embed) == "0526f6ab4e6fc547285a40cdabb36444f16e3b6523d79c5e5766861fcc661adb"
    assert digest(model.prior_bank.decode_log_scale) == "df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119"
    assert digest(model.prior_bank.output_proj_weight) == "a200e956283737dd6aaba9de0194aecf2d1613bdb76b71feb12f8c641fd0b708"


def test_default_adamw_one_step_matches_recompute_dense_low_level_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vfe3.config import VFE3Config
    from vfe3.model import block as block_module
    from vfe3.model.model import VFEModel
    from vfe3.train import build_optimizer, train_step

    torch.manual_seed(0)
    cfg = VFE3Config(
        vocab_size=8,
        embed_dim=4,
        n_heads=2,
        max_seq_len=4,
        n_layers=1,
        n_e_steps=1,
        pos_phi="none",
    )
    production = VFEModel(cfg)
    oracle = VFEModel(cfg)
    oracle.load_state_dict(production.state_dict())
    tokens = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=torch.long)
    targets = torch.tensor([[1, 2, 3, 4], [2, 1, 0, 7]], dtype=torch.long)

    def one_step(model: VFEModel) -> tuple[float, dict[str, torch.Tensor]]:
        optimizer = build_optimizer(model, cfg)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        loss = train_step(model, optimizer, scheduler, tokens, targets, grad_clip=1.0)
        return loss, {name: value.detach().clone() for name, value in model.named_parameters()}

    production_loss, production_parameters = one_step(production)
    original_e_step = block_module.e_step

    def recompute_dense_e_step(*args: object, **kwargs: object) -> object:
        kwargs["reuse_pairwise_kl_stats"] = False
        kwargs["compact_phi_block_transport"] = False
        kwargs["transport_mean_per_head"] = False
        return original_e_step(*args, **kwargs)

    monkeypatch.setattr(block_module, "e_step", recompute_dense_e_step)
    oracle_loss, oracle_parameters = one_step(oracle)

    assert production_loss == oracle_loss
    assert production_parameters.keys() == oracle_parameters.keys()
    for name in production_parameters:
        torch.testing.assert_close(
            production_parameters[name],
            oracle_parameters[name],
            atol=1e-6,
            rtol=1e-6,
            msg=lambda message, name=name: f"{name}: {message}",
        )


def _pair_route_models(
    monkeypatch:   pytest.MonkeyPatch,
    n_heads:       int,
    e_step_update: str,
    pos_rotation:  str,
) -> tuple[torch.Tensor, list[torch.Tensor], list[None], Callable[[], torch.Tensor]]:
    r"""Run production's forward while recording the E-step's pair-energy route.

    Returns production logits, every generic ``pairwise_energy`` tensor, one entry per
    ``diagonal_kl_pair_stats`` reuse call, and a thunk that runs the SAME weights through a
    recompute oracle: no pair-statistics reuse, no compact/per-head transport, and no model-level
    prebuilt transport (the E-step builds its own). The recorders stay live for the oracle run.
    """
    from vfe3.config import VFE3Config
    from vfe3.gradients import kernels as kernels_module
    from vfe3.model import block as block_module
    from vfe3.model.model import VFEModel

    torch.manual_seed(0)
    cfg = VFE3Config(
        vocab_size=8,
        embed_dim=4,
        n_heads=n_heads,
        max_seq_len=4,
        n_layers=1,
        n_e_steps=1,
        pos_phi="none",
        pos_rotation=pos_rotation,

        e_step_update=e_step_update,
    )
    production = VFEModel(cfg).eval()
    oracle = VFEModel(cfg).eval()
    oracle.load_state_dict(production.state_dict())
    tokens = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=torch.long)
    energies: list[torch.Tensor] = []
    stats_calls: list[None] = []
    original_pairwise_energy = kernels_module.pairwise_energy
    original_pair_stats = kernels_module.diagonal_kl_pair_stats

    def _capture_pairwise_energy(*args: object, **kwargs: object) -> torch.Tensor:
        energy = original_pairwise_energy(*args, **kwargs)
        energies.append(energy.detach().clone())
        return energy

    def _capture_pair_stats(*args: object, **kwargs: object) -> object:
        stats_calls.append(None)
        return original_pair_stats(*args, **kwargs)

    monkeypatch.setattr(kernels_module, "pairwise_energy", _capture_pairwise_energy)
    monkeypatch.setattr(kernels_module, "diagonal_kl_pair_stats", _capture_pair_stats)
    with torch.no_grad():
        production_logits = production(tokens)

    def run_oracle() -> torch.Tensor:
        original_e_step = block_module.e_step

        def recompute_dense_e_step(*args: object, **kwargs: object) -> object:
            kwargs["reuse_pairwise_kl_stats"] = False
            kwargs["compact_phi_block_transport"] = False
            kwargs["transport_mean_per_head"] = False
            kwargs["prebuilt_transport"] = None
            return original_e_step(*args, **kwargs)

        monkeypatch.setattr(block_module, "e_step", recompute_dense_e_step)
        with torch.no_grad():
            return oracle(tokens)

    return production_logits, energies, stats_calls, run_oracle


@pytest.mark.parametrize(
    "e_step_update",
    [
        pytest.param("gradient", id="plain-gradient"),
        pytest.param("mm_exact", id="plain-mm-exact"),
    ],
)
def test_single_head_forward_matches_recompute_dense_low_level_oracle(
    monkeypatch:   pytest.MonkeyPatch,
    e_step_update: str,
) -> None:
    r"""One head is one [K] block: a dense transport on which pair-statistics reuse fails closed.

    Production must take the generic pair-energy route (no ``diagonal_kl_pair_stats`` call, a zero
    self-energy diagonal) and be byte-identical to the recompute oracle. Single-block RoPE is
    rejected at construction (RoPE needs at least two irrep blocks), so RoPE is covered by the
    multi-head test below.
    """
    production_logits, energies, stats_calls, run_oracle = _pair_route_models(
        monkeypatch, 1, e_step_update, "none")

    assert stats_calls == []
    assert len(energies) == 1
    self_energy = energies[0].diagonal(dim1=-2, dim2=-1)
    self_mask = ((self_energy > 0.0) & (self_energy < 100.0)).to(self_energy.dtype)
    assert torch.equal(self_energy, torch.zeros_like(self_energy))
    assert torch.equal(self_mask, torch.zeros_like(self_mask))

    energies.clear()
    oracle_logits = run_oracle()

    assert stats_calls == []
    assert len(energies) == 1
    assert torch.equal(production_logits, oracle_logits)


@pytest.mark.parametrize(
    "e_step_update",
    [
        pytest.param("gradient", id="rope-gradient"),
        pytest.param("mm_exact", id="rope-mm-exact"),
    ],
)
def test_multi_head_rope_forward_reuses_pair_stats_and_matches_recompute_oracle(
    monkeypatch:   pytest.MonkeyPatch,
    e_step_update: str,
) -> None:
    r"""Two heads give a factored [2, 2] transport, on which production reuses pair statistics.

    ``vfe_block`` forwards ``reuse_pairwise_kl_stats=True`` and the shared factored transport admits
    it, so production makes exactly one ``diagonal_kl_pair_stats`` call and no generic pair-energy
    call. The recompute oracle takes the generic route instead; the two evaluation orders round
    differently, so they agree to a float32 tolerance rather than bitwise (as in
    tests/test_p3_pairwise_stats_reuse_20260711.py).
    """
    production_logits, energies, stats_calls, run_oracle = _pair_route_models(
        monkeypatch, 2, e_step_update, "rope")

    assert stats_calls == [None]
    assert energies == []

    oracle_logits = run_oracle()

    assert stats_calls == [None]                       # the oracle made no reuse call
    assert len(energies) == 1                          # it recomputed the generic pair energy
    torch.testing.assert_close(production_logits, oracle_logits, atol=1e-5, rtol=1e-6)


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
    group = get_group("glk")(K=2)
    phi = nn.Parameter(torch.zeros(1, 4))
    kwargs = {name: value}
    with pytest.raises(ValueError, match="finite and positive"):
        _make_optimizer(
            [{"params": [phi], "lr": 0.1, "pullback_group": True, "weight_decay": 0.0}],
            group,
            **kwargs,
        )


def test_pullback_group_step_moves_only_active_rows_and_creates_no_phi_state():
    group = get_group("glk")(K=2)
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
    from vfe3.config import VFE3Config
    from vfe3.model.model import VFEModel
    from vfe3.train import build_optimizer, train_step

    cfg = VFE3Config(
        vocab_size=12,
        embed_dim=4,
        n_heads=2,
        max_seq_len=8,
        n_layers=1,
        gauge_group="block_glk",
        pos_phi="none",
        m_phi_update_mode="pullback_group",
        phi_precond_mode="pullback_per_block",
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

    group = get_group("glk")(K=2, dtype=torch.float64)
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
        phi_precond_mode="pullback",
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
        phi_precond_mode="pullback_per_block",
    )

    assert seen["generators"] is group.generators
    assert seen["coordinate_layout"] == "block_head_row_major"


def test_pullback_group_stage_normalizes_basis_for_registered_extension(device):
    import vfe3.geometry.phi_preconditioner as phi_preconditioner

    name = "test_stage_normalized_registered_extension"
    group = get_group("glk")(K=2, dtype=torch.float32, device=device)
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
            phi_precond_mode=name,
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
    group = get_group("glk")(K=2, dtype=torch.float64)
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
        phi_precond_mode="pullback",
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
    group = get_group("glk")(K=2, dtype=torch.float32)
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

    group = get_group("glk")(K=2, dtype=torch.float64)
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
        phi_precond_mode="pullback",
    )
    assert candidate.rejected_rows.tolist() == [True]
    assert torch.equal(candidate.candidate_phi, phi)


def test_pullback_group_step_rejects_two_parameter_groups_atomically(monkeypatch):
    import vfe3.gauge_optim as gauge_optim

    group = get_group("glk")(K=2, dtype=torch.float64)
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

    group = get_group("glk")(K=2, dtype=torch.float64)
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

    group = get_group("glk")(K=2)
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
        phi_precond_mode="pullback_per_block",
    )
    opt._collect_gauge_diag = True
    opt._gauge_diag = {"phi_pullback_damped_gen_cond_max": 123.0}
    phi.grad = None

    opt.step()

    assert opt._gauge_diag == {}
