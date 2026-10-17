"""Tiny contract and numerical regressions for lazy elementary block generators."""

from tests.profiles import test_config
from collections.abc import Iterator
import copy
import pickle

import pytest
import torch

from vfe3 import metrics
from vfe3.gauge_optim import embedded_phi_frobenius_norm, project_phi_parameter_rows_
from vfe3.geometry import groups
from vfe3.geometry.generators import generate_glk_multihead
from vfe3.geometry.groups import GaugeGroup, get_group
from vfe3.geometry.transport import build_factored_transport, CompactFactoredTransport
from vfe3.model.model import VFEModel
from vfe3.run_artifacts import collect_phi_numerics, _cost_model_fields
from vfe3.train import build_optimizer, lr_lambda, train_step


@pytest.fixture(autouse=True)
def one_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _forbid_dense(*args: object, **kwargs: object) -> torch.Tensor:
    raise AssertionError("active route materialized the dense generator basis")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_lazy_metadata_moves_gram_and_public_tensor(dtype: torch.dtype) -> None:
    group = get_group("block_glk")(6, 3)
    assert isinstance(group._generator_source, groups._ElementaryBlockBasis)
    group.gram_diagonal()
    group.apply_generators(lambda x: x.to(dtype=dtype))
    assert group.generator_shape == (12, 6, 6)
    assert group.generator_dtype == dtype
    assert group.gram_diagonal_uniform() == 1.0
    assert group.gram_diagonal().dtype == dtype
    assert isinstance(group._generator_source, groups._ElementaryBlockBasis)
    actual = group.generators
    assert actual is group.generators
    assert torch.equal(actual, generate_glk_multihead(6, 3, dtype=dtype))
    group.apply_generators(lambda x: x.double())
    assert group.elementary_block_shape == (3, 2)
    assert group.generators.dtype == torch.float64


@pytest.mark.parametrize("kind", ["deepcopy", "pickle", "custom"])
def test_copy_and_custom_tensor_contract(kind: str) -> None:
    group = get_group("block_glk")(4, 2)
    if kind == "deepcopy":
        cloned = copy.deepcopy(group)
    elif kind == "pickle":
        cloned = pickle.loads(pickle.dumps(group))
    else:
        source = generate_glk_multihead(4, 2).roll(1, dims=0)
        cloned = GaugeGroup("custom", source, [2, 2], False)
        assert cloned.generators is source
        assert cloned.elementary_block_shape is None
        return
    assert isinstance(cloned._generator_source, groups._ElementaryBlockBasis)
    assert torch.equal(cloned.generators, group.generators)
    cloned.generators.mul_(2)
    assert not torch.equal(cloned.generators, group.generators)


@pytest.mark.parametrize("mutation", ["in_place", "replacement", "trainable"])
def test_public_basis_changes_withdraw_shortcuts(mutation: str) -> None:
    group = get_group("block_glk")(4, 2, dtype=torch.float64)
    group.gram_diagonal()
    if mutation == "in_place":
        group.generators[0].add_(group.generators[1], alpha=0.5)
    elif mutation == "replacement":
        group.generators = group.generators.roll(1, dims=0)
    else:
        group.generators.requires_grad_()
    phi = torch.randn(1, 2, 8, dtype=torch.float64) * 0.02
    assert group.elementary_block_shape is None
    with torch.no_grad():
        expected = torch.einsum("...a,aij->...ij", phi, group.generators)
        torch.testing.assert_close(group.embed_algebra(phi), expected, rtol=0, atol=0)
        torch.testing.assert_close(embedded_phi_frobenius_norm(phi, group, warn_fallback=False),
                                   torch.linalg.matrix_norm(expected), rtol=0, atol=0)
    transport = build_factored_transport(phi, group, compact_blocks=True)
    if mutation == "trainable":
        assert not isinstance(transport, CompactFactoredTransport)        # the dense embedding keeps the graph
    else:
        # The packed-coordinate reshape is withdrawn. The mutated basis still lives on the head blocks, so the
        # compact route embeds through its blocks (audit 2026-10-10, item 26) and equals the dense factors.
        assert isinstance(transport, CompactFactoredTransport)
        from vfe3.geometry.lie_ops import CompactBlockElement
        dense = build_factored_transport(phi, group, compact_blocks=False)
        torch.testing.assert_close(CompactBlockElement(transport.exp_blocks, 4).to_dense(), dense.exp_phi,
                                   rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_diagnostic_embedding_and_statistics_keep_dense_reductions(dtype: torch.dtype,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    group = get_group("block_glk")(4, 2, dtype=dtype)
    basis = generate_glk_multihead(4, 2, dtype=dtype)
    phi = torch.randn(3, 8, dtype=dtype) * 0.05
    # Include the chart boundary; keep the same dense matrix-norm reduction.
    phi[0].zero_()
    phi[0, 0] = 20.0
    monkeypatch.setattr(groups, "generate_glk_multihead", _forbid_dense)
    with torch.no_grad():
        actual = group.embed_algebra(phi)
        expected = torch.einsum("na,aij->nij", phi, basis)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(torch.linalg.matrix_norm(actual), torch.linalg.matrix_norm(expected), rtol=0, atol=0)
        assert metrics.phi_chart_statistics(phi, group, block_dims=[2, 2]) == metrics.phi_chart_statistics(phi, basis, block_dims=[2, 2])
        small_phi = phi * 0.002
        assert metrics.flatness_reference_statistics(small_phi, group, block_dims=[2, 2]) == metrics.flatness_reference_statistics(small_phi, basis, block_dims=[2, 2])
    assert isinstance(group._generator_source, groups._ElementaryBlockBasis)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_embedding_preserves_dense_provenance(value: float) -> None:
    group = get_group("block_glk")(4, 2)
    phi = torch.zeros(2, 8)
    phi[0, 0] = value
    actual = group.embed_algebra(phi)
    expected = torch.einsum("na,aij->nij", phi, group.generators)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


def test_dense_autograd_and_compact_transport_finite_difference() -> None:
    torch.manual_seed(14)
    group = get_group("block_glk")(4, 2, dtype=torch.float64)
    phi = (torch.randn(1, 2, 8, dtype=torch.float64) * 0.02).requires_grad_()
    weights = torch.randn(1, 2, 2, 2, 2, dtype=torch.float64)
    def objective(coords: torch.Tensor) -> torch.Tensor:
        return (build_factored_transport(
            coords, group, compact_blocks=True).exp_blocks * weights).sum()
    gradient, = torch.autograd.grad(objective(phi), phi)
    direction = torch.randn_like(phi)
    delta = 1e-5
    difference = (objective(phi.detach() + delta * direction) - objective(phi.detach() - delta * direction)) / (2 * delta)
    torch.testing.assert_close((gradient * direction).sum(), difference, rtol=1e-7, atol=1e-9)
    assert isinstance(group._generator_source, groups._ElementaryBlockBasis)
    dense = group.embed_algebra(phi)
    dense_weight = torch.randn_like(dense)
    dense_weight[..., 0, 2] = float("inf")
    actual_grad, = torch.autograd.grad(dense, phi, dense_weight)
    reference = torch.einsum("...a,aij->...ij", phi, group.generators)
    expected_grad, = torch.autograd.grad(reference, phi, dense_weight)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("layers", [1, 2])
def test_complete_training_projection_and_reports_remain_lazy(layers: int,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = test_config(vocab_size=7, embed_dim=4, n_heads=2, max_seq_len=4,
                    n_layers=layers, n_e_steps=1, gauge_group="block_glk",
                    decode_mode="linear", pos_phi="learned",
                    phi_mstep_max_matrix_norm=12.0,
                    use_block_mlp=True, block_mlp_expansion=2, warmup_steps=1,
                    max_steps=4, batch_size=1, amp_dtype=None)
    monkeypatch.setattr(groups, "generate_glk_multihead", _forbid_dense)
    torch.manual_seed(23)
    model = VFEModel(cfg).cpu()
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_lambda(step, cfg))
    tokens = torch.tensor([[0, 1, 2, 1]])
    targets = torch.tensor([[-100, -100, 1, 3]])
    status: dict = {}
    for _ in range(2):
        train_step(model, optimizer, scheduler, tokens, targets, metrics_out={}, status_out=status)
        assert status["did_step"]
    project_phi_parameter_rows_(model, 12.0)
    diagnostics = model.diagnostics(tokens)
    assert "phi_matrix_norm_max" in diagnostics
    model.diagnostics_per_layer(tokens)
    report = collect_phi_numerics(model, tokens)
    assert "chart" in report and "flatness" in report
    _cost_model_fields(model, cfg, sum(p.numel() for p in model.parameters()), 4)
    assert isinstance(model.group._generator_source, groups._ElementaryBlockBasis)
    restored = VFEModel(cfg).double()
    restored.load_state_dict(model.state_dict())
    assert restored.group.generator_dtype == torch.float64
    assert isinstance(restored.group._generator_source, groups._ElementaryBlockBasis)


