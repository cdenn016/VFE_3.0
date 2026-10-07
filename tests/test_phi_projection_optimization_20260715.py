from tests.profiles import test_config
import warnings

import pytest
import torch

import vfe3.gauge_optim as gauge_optim
from vfe3.geometry.groups import GaugeGroup, get_group
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import RunArtifacts
from vfe3.train import build_optimizer, train


def _phi_table_model() -> VFEModel:
    cfg = test_config(
        vocab_size=8,
        embed_dim=4,
        n_heads=2,
        max_seq_len=4,
        n_layers=1,
        pos_phi="none",
    )
    return VFEModel(cfg)


def _phi_tables(model: VFEModel) -> list[torch.Tensor]:
    return [
        model.prior_bank.phi_embed,
    ]


def _dense_norm(phi: torch.Tensor, generators: torch.Tensor) -> torch.Tensor:
    embedded = torch.einsum("...a,aij->...ij", phi, generators)
    return torch.linalg.matrix_norm(embedded, ord="fro", dim=(-2, -1))


def _projection_train_case() -> tuple:
    cfg = test_config(
        vocab_size=8,
        embed_dim=4,
        n_heads=2,
        max_seq_len=4,
        n_layers=1,
        n_e_steps=1,
        pos_phi="none",
        phi_mstep_max_matrix_norm=2.0,
    )
    model = VFEModel(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    tokens = torch.tensor([[0, 1, 2, 3]])
    targets = torch.tensor([[1, 2, 3, 4]])
    return model, optimizer, scheduler, tokens, targets


@pytest.mark.parametrize(
    "name, kwargs, expected_uniform",
    [
        ("block_glk", {"n_heads": 1}, 1.0),
        ("block_glk", {"n_heads": 2}, 1.0),
    ],
)
def test_certified_gram_diagonal_matches_dense_gram(
    name:             str,
    kwargs:           dict,
    expected_uniform: float | None,
) -> None:
    group = get_group(name)(K=4, dtype=torch.float64, **kwargs)
    gram = torch.einsum("aij,bij->ab", group.generators, group.generators)
    diagonal = gram.diagonal()

    torch.testing.assert_close(group.gram_diagonal(), diagonal)
    torch.testing.assert_close(
        gram - torch.diag(diagonal),
        torch.zeros_like(gram),
        rtol=0.0,
        atol=0.0,
    )
    assert group.gram_diagonal_uniform() == expected_uniform
    assert group.phi_norm_route() == "diagonal_gram"


def test_custom_group_defaults_to_dense_route() -> None:
    generators = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[1.0, 1.0], [0.0, 0.0]],
        ]
    )
    group = GaugeGroup(
        name="custom_nonorthogonal",
        generators=generators,
        irrep_dims=[2],
        skew_symmetric=False,
    )

    assert group.gram_diagonal() is None
    assert group.gram_diagonal_uniform() is None
    assert group.phi_norm_route() == "dense_fallback"


def test_gram_diagonal_cache_refreshes_after_in_place_generator_change() -> None:
    # A dense certified basis (the lazy block_glk basis withdraws its certificate on mutation).
    from vfe3.geometry.generators import generate_glk_multihead
    group = GaugeGroup(name="dense_elementary", generators=generate_glk_multihead(2, 1, dtype=torch.float64),
                       irrep_dims=[2], skew_symmetric=False, frobenius_gram_diagonal=True,
                       frobenius_gram_uniform=1.0)
    first = group.gram_diagonal()
    assert first is group.gram_diagonal()

    group.generators.mul_(2.0)
    second = group.gram_diagonal()

    assert second is not first
    torch.testing.assert_close(second, torch.full_like(second, 4.0))
    assert group.gram_diagonal_uniform() == 4.0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_diagonal_norm_kernel_matches_dense_embedding(dtype: torch.dtype) -> None:
    assert hasattr(gauge_optim, "embedded_phi_frobenius_norm")
    group = get_group("block_glk")(K=8, n_heads=2, dtype=dtype)
    generator = torch.Generator().manual_seed(17)
    phi = torch.randn(13, group.generators.shape[0], dtype=dtype, generator=generator)
    expected = _dense_norm(phi, group.generators)

    actual = gauge_optim.embedded_phi_frobenius_norm(phi, group)

    torch.testing.assert_close(actual, expected)


def test_nonuniform_diagonal_norm_kernel_matches_dense_embedding() -> None:
    assert hasattr(gauge_optim, "embedded_phi_frobenius_norm")
    # A certified-orthogonal basis with unequal generator norms (diagonal but nonuniform Gram).
    from vfe3.geometry.generators import generate_glk_multihead
    scales = torch.tensor([0.5, 1.0, 2.0, 3.0], dtype=torch.float64).view(-1, 1, 1)
    group = GaugeGroup(name="custom_scaled", generators=generate_glk_multihead(2, 1, dtype=torch.float64) * scales,
                       irrep_dims=[2], skew_symmetric=False, frobenius_gram_diagonal=True)
    assert group.gram_diagonal_uniform() is None
    phi = torch.randn(11, group.generators.shape[0], dtype=torch.float64)

    torch.testing.assert_close(
        gauge_optim.embedded_phi_frobenius_norm(phi, group),
        _dense_norm(phi, group.generators),
    )


def test_uncertified_nonorthogonal_group_uses_exact_dense_fallback() -> None:
    assert hasattr(gauge_optim, "embedded_phi_frobenius_norm")
    generators = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[1.0, 1.0], [0.0, 0.0]],
        ]
    )
    group = GaugeGroup(
        name="custom_nonorthogonal",
        generators=generators,
        irrep_dims=[2],
        skew_symmetric=False,
    )
    phi = torch.randn(7, 2)

    with pytest.warns(RuntimeWarning, match="dense_fallback"):
        actual = gauge_optim.embedded_phi_frobenius_norm(phi, group)

    torch.testing.assert_close(actual, _dense_norm(phi, generators))


def test_silent_projection_matches_dense_oracle_and_returns_no_stats() -> None:
    model = _phi_table_model()
    generator = torch.Generator().manual_seed(23)
    expected_tables = []
    with torch.no_grad():
        for table in _phi_tables(model):
            table.copy_(torch.randn(table.shape, generator=generator) * 3.0)
            expected = table.detach().clone()
            norm = _dense_norm(expected, model.group.generators)
            scale = (2.0 / norm.clamp(min=1e-12)).clamp(max=1.0)
            expected.mul_(scale.unsqueeze(-1))
            expected_tables.append(expected)

    stats = gauge_optim.project_phi_parameter_rows_(
        model,
        2.0,
        chunk_rows=3,
        collect_stats=False,
    )

    assert stats == {}
    for table, expected in zip(_phi_tables(model), expected_tables):
        torch.testing.assert_close(table, expected)


def test_projection_metrics_match_dense_oracle() -> None:
    model = _phi_table_model()
    with torch.no_grad():
        for table in _phi_tables(model):
            table.fill_(5.0)
    preproject_max = max(
        float(_dense_norm(table, model.group.generators).max().detach())
        for table in _phi_tables(model)
    )
    total_rows = sum(table.numel() // table.shape[-1] for table in _phi_tables(model))

    stats = gauge_optim.project_phi_parameter_rows_(
        model,
        2.0,
        chunk_rows=3,
        collect_stats=True,
    )

    assert stats["phi_chart_projected_rows"] == total_rows
    assert stats["phi_chart_total_rows"] == total_rows
    assert stats["phi_chart_projected_fraction"] == 1.0
    assert stats["phi_chart_preproject_max"] == pytest.approx(preproject_max)
    assert 0.0 < stats["phi_chart_projection_scale_min"] < 1.0


def test_automatic_projection_chunks_preserve_below_bound_rows_exactly() -> None:
    model = _phi_table_model()
    with torch.no_grad():
        for table in _phi_tables(model):
            table.fill_(0.1)
    before = [table.detach().clone() for table in _phi_tables(model)]

    gauge_optim.project_phi_parameter_rows_(model, 2.0, collect_stats=False)

    for table, expected in zip(_phi_tables(model), before):
        torch.testing.assert_close(table, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("radius", [0.0, -1.0, float("inf"), float("nan")])
def test_projection_rejects_invalid_radius(radius: float) -> None:
    with pytest.raises(ValueError, match="max_matrix_norm"):
        gauge_optim.project_phi_parameter_rows_(_phi_table_model(), radius)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_rows": 0},
        {"chunk_rows": 1.5},
        {"temporary_bytes": 0},
        {"temporary_bytes": True},
    ],
)
def test_projection_rejects_invalid_chunk_controls(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        gauge_optim.project_phi_parameter_rows_(_phi_table_model(), 2.0, **kwargs)


def test_projection_deduplicates_aliased_phi_tables() -> None:
    model = _phi_table_model()
    model.pos_phi_free = model.prior_bank.phi_embed
    unique_tables = {id(table): table for table in _phi_tables(model)}
    expected_rows = sum(table.numel() // table.shape[-1] for table in unique_tables.values())

    stats = gauge_optim.project_phi_parameter_rows_(model, 2.0)

    assert stats["phi_chart_total_rows"] == expected_rows


def test_silent_train_step_projects_without_collecting_stats(monkeypatch) -> None:
    import vfe3.train as train_module

    model, optimizer, scheduler, tokens, targets = _projection_train_case()
    collect_stats = []

    def _project(_model, _radius, **kwargs):
        collect_stats.append(kwargs.get("collect_stats"))
        return {}

    monkeypatch.setattr(train_module, "project_phi_parameter_rows_", _project)

    train_module.train_step(model, optimizer, scheduler, tokens, targets)

    assert collect_stats == [False]


def test_logged_train_step_records_projection_stats_and_cpu_timing(monkeypatch) -> None:
    import vfe3.train as train_module

    model, optimizer, scheduler, tokens, targets = _projection_train_case()
    collect_stats = []

    def _project(_model, _radius, **kwargs):
        collect_stats.append(kwargs.get("collect_stats"))
        return {
            "phi_chart_projected_rows": 1.0,
            "phi_chart_total_rows": 8.0,
            "phi_chart_projected_fraction": 0.125,
            "phi_chart_preproject_max": 2.5,
            "phi_chart_projection_scale_min": 0.8,
        }

    monkeypatch.setattr(train_module, "project_phi_parameter_rows_", _project)
    metrics = {}

    train_module.train_step(
        model,
        optimizer,
        scheduler,
        tokens,
        targets,
        metrics_out=metrics,
    )

    assert collect_stats == [True]
    assert metrics["phi_chart_projected_rows"] == 1.0
    assert metrics["phi_chart_projection_stats_collected"] == 1.0
    assert metrics["phi_chart_projection_ms"] >= 0.0


def test_nonfinite_step_does_not_project(monkeypatch) -> None:
    import vfe3.train as train_module

    model, optimizer, scheduler, tokens, targets = _projection_train_case()
    model.prior_bank.mu_embed.register_hook(
        lambda grad: torch.full_like(grad, float("nan"))
    )

    def _unexpected(*args, **kwargs):
        raise AssertionError("skipped optimizer step projected phi")

    monkeypatch.setattr(train_module, "project_phi_parameter_rows_", _unexpected)
    metrics = {}

    scheduler_epoch = scheduler.last_epoch
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        train_module.train_step(
            model,
            optimizer,
            scheduler,
            tokens,
            targets,
            metrics_out=metrics,
        )

    assert not any("lr_scheduler.step" in str(item.message) for item in caught)
    assert scheduler.last_epoch == scheduler_epoch
    assert metrics["step_skipped"] == 1.0


def test_phi_chart_summary_route_is_none_when_disabled_and_named_when_enabled() -> None:
    import vfe3.run_artifacts as run_artifacts

    assert hasattr(run_artifacts, "_phi_chart_norm_route")
    disabled_cfg = test_config(vocab_size=8, embed_dim=4, n_heads=2)
    disabled_model = VFEModel(disabled_cfg)
    enabled_cfg = test_config(
        vocab_size=8,
        embed_dim=4,
        n_heads=2,
        phi_mstep_max_matrix_norm=2.0,
    )
    enabled_model = VFEModel(enabled_cfg)
    pullback_cfg = test_config(
        vocab_size=8,
        embed_dim=4,
        n_heads=2,
        gauge_group="block_glk",
        m_phi_update_mode="pullback_group",
        transport_chart_max_norm=6.0,
    )
    pullback_model = VFEModel(pullback_cfg)

    assert run_artifacts._phi_chart_norm_route(disabled_model, disabled_cfg) is None
    assert (
        run_artifacts._phi_chart_norm_route(enabled_model, enabled_cfg)
        == "diagonal_gram"
    )
    assert (
        run_artifacts._phi_chart_norm_route(pullback_model, pullback_cfg)
        == "diagonal_gram:factor_radius=5.0"
    )


def test_geometry_health_adds_projection_timing_panel_only_when_present() -> None:
    from matplotlib import pyplot as plt
    from vfe3.viz.figures import plot_geometry_health

    old_history = {
        "step": [1, 2, 3],
        "phi_chart_projected_fraction": [0.1, 0.2, 0.1],
    }
    old_figure = plot_geometry_health(old_history)
    assert "Projected M-step cost" not in [axis.get_title() for axis in old_figure.axes]

    timed_history = dict(old_history)
    timed_history["phi_chart_projection_ms"] = [0.4, 0.5, 0.3]
    timed_figure = plot_geometry_health(timed_history)

    assert "Projected M-step cost" in [axis.get_title() for axis in timed_figure.axes]
    plt.close(old_figure)
    plt.close(timed_figure)


def test_enabled_projection_persists_timing_metrics_in_training_row(tmp_path) -> None:
    cfg = test_config(
        vocab_size=8,
        embed_dim=4,
        n_heads=2,
        max_seq_len=4,
        n_layers=1,
        n_e_steps=1,
        pos_phi="none",
        phi_mstep_max_matrix_norm=2.0,
        max_steps=1,
    )
    model = VFEModel(cfg)
    artifacts = RunArtifacts(tmp_path, cfg, model)
    loader = [(torch.tensor([[0, 1, 2, 3]]), torch.tensor([[1, 2, 3, 4]]))]

    train(
        model,
        loader,
        cfg,
        n_steps=1,
        log_interval=1,
        artifacts=artifacts,
        generate_samples=False,
    )

    assert artifacts.history[0]["phi_chart_projection_stats_collected"] == 1.0
    assert artifacts.history[0]["phi_chart_projection_ms"] >= 0.0
    header = (tmp_path / "metrics.csv").read_text().splitlines()[0]
    assert "phi_chart_projection_ms" in header
