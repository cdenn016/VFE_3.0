"""PHI-reachable compact / rope / graph-metric machinery.

Provenance: these tests were relocated verbatim from ``tests/test_omega_direct.py`` when the
``gauge_parameterization == "omega_direct"`` path was removed (schema 25). Their subjects are the
surviving shared machinery -- ``CompactFactoredTransport`` / ``CompactBlockElement`` storage and
arithmetic, ``RopeTransport`` validation and covariance, and the sampled graph metrics -- all of
which the phi route reaches (``build_factored_transport`` with ``compact_blocks=True``,
``CompactFactoredTransport.to_dense_omega``). None of them depends on a ``gauge_parameterization``
selector, so they carry over
unchanged; the file name records the surviving chart they now belong to.
"""
import pytest
import torch

from vfe3.geometry.lie_ops import CompactBlockElement
from vfe3.geometry.transport import (
    RopeTransport,
    CompactFactoredTransport,
    FactoredTransport,
    _checked_group_inverse,
)


def test_tied_compact_element_rejects_partial_logical_matrix_slice():
    blocks = torch.eye(2).expand(2, 3, 2, 2).clone()
    element = CompactBlockElement(blocks, K=4, tied=True)

    leading = element[:, :1]
    explicit_full_matrix = element[..., :, :]
    assert leading.shape == (2, 1, 4, 4)
    assert torch.equal(leading.blocks, blocks[:, :1])
    assert torch.equal(explicit_full_matrix.blocks, blocks)
    with pytest.raises(ValueError, match="matrix axes intact.*to_dense"):
        element[..., :1, :1]


def test_tied_compact_element_rejects_multiaxis_boolean_mask():
    blocks = torch.eye(2).expand(2, 3, 2, 2).clone()
    element = CompactBlockElement(blocks, K=4, tied=True)
    mask_BN = torch.tensor([[True, False, True], [False, True, False]])

    with pytest.raises(ValueError, match="boolean advanced indexing.*blocks.*to_dense"):
        element[mask_BN, :2]


def test_compact_factored_unsqueeze_negative_alias_and_matrix_rejection():
    H, d, K, L, M = 2, 2, 4, 2, 3
    exp_blocks = torch.eye(d).expand(L, H, d, d).clone()
    inv_blocks = torch.eye(d).expand(M, H, d, d).clone()
    factored = CompactFactoredTransport(exp_blocks, inv_blocks, K)

    positive = factored.unsqueeze(0)
    negative = factored.unsqueeze(-5)                         # logical (L,M,K,K) rank -> leading alias
    assert torch.equal(positive.exp_blocks, negative.exp_blocks)
    assert torch.equal(positive.inv_blocks, negative.inv_blocks)
    for dim in (1, 2, 3, 4, -1, -2, -3, -4):
        with pytest.raises(ValueError, match="leading axis"):
            factored.unsqueeze(dim)
    assert torch.equal(
        positive.unsqueeze(0).exp_blocks, positive.unsqueeze(-6).exp_blocks)
    assert torch.equal(
        positive.unsqueeze(1).exp_blocks, positive.unsqueeze(-5).exp_blocks)
    with pytest.raises(ValueError, match="leading axis"):
        positive.unsqueeze(-4)


@pytest.mark.parametrize(
    "representation,L,M",
    [("compact", 2, 3), ("dense", 3, 2)],
)
def test_rope_transport_rejects_nonsquare_query_key_base(representation, L, M):
    H, d, K = 2, 2, 4
    exp_blocks = torch.eye(d).expand(L, H, d, d).clone()
    inv_blocks = torch.eye(d).expand(M, H, d, d).clone()
    compact = CompactFactoredTransport(exp_blocks, inv_blocks, K)
    base = compact if representation == "compact" else compact.to_dense_omega()
    rope = torch.eye(K).expand(L, K, K).clone()

    with pytest.raises(ValueError, match="square token transport"):
        RopeTransport(base, rope, insertion="left")


@pytest.mark.parametrize("representation", ["compact", "dense"])
def test_rope_transport_rejects_mismatched_rotation_length(representation):
    N, H, d, K = 3, 2, 2, 4
    blocks = torch.eye(d).expand(N, H, d, d).clone()
    compact = CompactFactoredTransport(blocks, blocks.clone(), K)
    base = compact if representation == "compact" else compact.to_dense_omega()
    rope = torch.eye(K).expand(N - 1, K, K).clone()

    with pytest.raises(ValueError, match="rope token length"):
        RopeTransport(base, rope, insertion="left")


def test_rope_transport_rejects_dense_singleton_matrix_axis():
    N, K = 3, 4
    base = torch.zeros(N, N, 1, K)
    rope = torch.eye(K).expand(N, K, K).clone()

    with pytest.raises(ValueError, match="dense base.*square K x K"):
        RopeTransport(base, rope, insertion="left")


@pytest.mark.parametrize(
    "exp_shape,inv_shape,match",
    [
        ((3, 1, 4), (3, 4, 4), "each end in square K x K"),
        ((3, 4, 4), (3, 5, 5), "same K"),
        ((1, 3, 4, 4), (2, 3, 4, 4), "matching leading batch shapes"),
    ],
    ids=["singleton_matrix_axis", "different_K", "different_batch_shape"],
)
def test_rope_transport_rejects_malformed_factored_base(exp_shape, inv_shape, match):
    base = FactoredTransport(
        torch.zeros(exp_shape), torch.zeros(inv_shape), irrep_dims=[2, 2])
    rope = torch.eye(4).expand(3, 4, 4).clone()

    with pytest.raises(ValueError, match=match):
        RopeTransport(base, rope, insertion="left")


def test_compact_sampled_metrics_match_canonical_dense_operator():
    import vfe3.metrics as metrics_module

    N, H, d, K = 5, 2, 2, 4
    gen = torch.Generator().manual_seed(113)
    exp_blocks = torch.eye(d).expand(N, H, d, d).clone()
    inv_blocks = torch.eye(d).expand(N, H, d, d).clone()
    exp_blocks += 0.08 * torch.randn(exp_blocks.shape, generator=gen)
    inv_blocks += 0.07 * torch.randn(inv_blocks.shape, generator=gen)
    compact = CompactFactoredTransport(exp_blocks, inv_blocks, K)
    dense = compact.to_dense_omega()

    assert torch.allclose(
        metrics_module.transport_asymmetry(compact),
        metrics_module.transport_asymmetry(dense), atol=2e-6, rtol=1e-5)
    compact_h = metrics_module.holonomy_deviation_sampled(
        compact, n_triples=24, n_boot=16, seed=7)
    dense_h = metrics_module.holonomy_deviation_sampled(
        dense, n_triples=24, n_boot=16, seed=7)
    assert compact_h.keys() == dense_h.keys()
    for key in compact_h:
        assert torch.allclose(compact_h[key], dense_h[key], atol=2e-5, rtol=1e-5), key
    compact_w = metrics_module.holonomy_wilson_sampled(
        compact, n_heads=H, irrep_dims=[d] * H, n_triples=24, n_boot=16, seed=7)
    dense_w = metrics_module.holonomy_wilson_sampled(
        dense, n_heads=H, irrep_dims=[d] * H, n_triples=24, n_boot=16, seed=7)
    assert compact_w.keys() == dense_w.keys()
    for key in compact_w:
        assert torch.allclose(compact_w[key], dense_w[key], atol=2e-5, rtol=1e-5), key
    assert torch.allclose(
        metrics_module.cocycle_residual_sampled(compact, n_triples=24, seed=7),
        metrics_module.cocycle_residual_sampled(dense, n_triples=24, seed=7),
        atol=2e-5, rtol=1e-5)


@pytest.mark.parametrize("L,M", [(2, 3), (3, 2)])
@pytest.mark.parametrize("representation", ["compact", "dense"])
@pytest.mark.parametrize(
    "metric_name",
    [
        "transport_asymmetry",
        "holonomy_deviation_sampled",
        "holonomy_wilson_sampled",
        "cocycle_residual_sampled",
    ],
)
def test_square_graph_metrics_reject_rectangular_transport(
    metric_name, representation, L, M,
):
    import vfe3.metrics as metrics_module

    H, d, K = 2, 2, 4
    exp_blocks = torch.eye(d).expand(L, H, d, d).clone()
    inv_blocks = torch.eye(d).expand(M, H, d, d).clone()
    compact = CompactFactoredTransport(exp_blocks, inv_blocks, K)
    omega = compact if representation == "compact" else compact.to_dense_omega()
    metric = getattr(metrics_module, metric_name)

    with pytest.raises(ValueError, match=r"square token transport \(Nq == Nk\)"):
        metric(omega)


@pytest.mark.parametrize("representation", ["compact", "dense"])
@pytest.mark.parametrize(
    "metric_name",
    [
        "holonomy_deviation_sampled",
        "holonomy_wilson_sampled",
        "cocycle_residual_sampled",
    ],
)
def test_sampled_graph_metrics_reject_batched_square_transport_before_empty_return(
    metric_name, representation,
):
    import vfe3.metrics as metrics_module

    B, N, H, d, K = 1, 2, 2, 2, 4
    blocks = torch.eye(d).expand(B, N, H, d, d).clone()
    compact = CompactFactoredTransport(blocks, blocks.clone(), K)
    omega = compact if representation == "compact" else compact.to_dense_omega()
    metric = getattr(metrics_module, metric_name)

    with pytest.raises(ValueError, match=r"requires (?:an )?unbatched"):
        metric(omega)


@pytest.mark.parametrize(
    "metric_name",
    [
        "transport_asymmetry",
        "holonomy_deviation_sampled",
        "holonomy_wilson_sampled",
        "cocycle_residual_sampled",
    ],
)
def test_graph_metrics_reject_dense_nonsquare_matrix_axes(metric_name):
    import vfe3.metrics as metrics_module

    omega = torch.zeros(3, 3, 1, 4)
    metric = getattr(metrics_module, metric_name)

    with pytest.raises(ValueError, match="trailing square K x K"):
        metric(omega)


def test_checked_group_inverse_skew_rows_match_true_inverse_and_gradient():
    clean_0 = torch.eye(3)
    clean_1 = torch.tensor([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
    ])
    drifted_0 = torch.tensor([
        [1.10, 0.20, 0.00],
        [0.00, 0.90, 0.10],
        [0.00, 0.00, 1.05],
    ])
    drifted_1 = torch.tensor([
        [0.95, 0.00, 0.15],
        [0.05, 1.20, 0.00],
        [0.00, 0.10, 0.85],
    ])
    omega = torch.stack([clean_0, drifted_0, drifted_1, clean_1]).reshape(2, 2, 3, 3)
    omega = omega.requires_grad_(True)
    got = _checked_group_inverse(omega, residual_tol=1e-4)
    true_inverse = torch.linalg.inv(omega.double()).to(omega.dtype)

    assert got.shape == omega.shape
    assert got.dtype == omega.dtype
    assert got.device == omega.device
    assert torch.allclose(got, true_inverse, atol=1e-7, rtol=1e-6)

    weights = torch.arange(got.numel(), device=got.device, dtype=got.dtype).reshape_as(got) / got.numel()
    grad = torch.autograd.grad((got * weights).sum(), omega)[0]

    omega_ref = omega.detach().clone().requires_grad_(True)
    reference = torch.linalg.inv(omega_ref.double()).to(omega_ref.dtype)
    grad_reference = torch.autograd.grad((reference * weights).sum(), omega_ref)[0]

    assert torch.isfinite(grad).all()
    assert torch.isfinite(grad_reference).all()
    assert torch.allclose(grad, grad_reference, atol=1e-6, rtol=1e-5)
