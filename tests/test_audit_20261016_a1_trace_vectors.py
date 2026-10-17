r"""Audit 2026-10-16 F87: the canonical block_glk basis gives its trace functionals without the dense basis.

``project_phi_to_slk`` (pos_phi_project_slk) reads ``GaugeGroup.block_trace_vectors`` on every forward. For the
lazy elementary basis the functional is :math:`V_{h,(h,i,j)} = \operatorname{tr}E_{ij} = \delta_{ij}`, so the
row of head h is ``eye(d).reshape(-1)`` on that head's coordinate run; reading it from the dense
``(H d^2, K, K)`` basis materialized 524 MB of float32 at K = 160, H = 5.
"""
import pytest
import torch

from vfe3.geometry.groups import _ElementaryBlockBasis, get_group
from vfe3.geometry.lie_ops import _block_trace_vectors, project_phi_to_slk


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_elementary_block_trace_vectors_stay_lazy_and_match_the_dense_basis(dtype: torch.dtype) -> None:
    lazy = get_group("block_glk")(6, 2, dtype=dtype)
    assert isinstance(lazy._generator_source, _ElementaryBlockBasis)
    V = lazy.block_trace_vectors()
    phi = torch.randn(3, 4, lazy.generator_shape[0], generator=torch.Generator().manual_seed(0), dtype=dtype)
    projected = project_phi_to_slk(phi, lazy, lazy.irrep_dims)
    assert isinstance(lazy._generator_source, _ElementaryBlockBasis)        # neither call built the dense basis

    dense = get_group("block_glk")(6, 2, dtype=dtype)
    G = dense.generators                                                    # (H d^2, K, K)
    reference = _block_trace_vectors(G, dense.irrep_dims)
    assert V.dtype == reference.dtype and V.device == reference.device
    torch.testing.assert_close(V, reference, rtol=0, atol=0)
    torch.testing.assert_close(projected, project_phi_to_slk(phi, G, dense.irrep_dims), rtol=0, atol=0)
    blocks = projected.reshape(3, 4, 2, 3, 3)
    assert float(blocks.diagonal(dim1=-2, dim2=-1).sum(-1).abs().max()) < 1e-6   # tr M_h = 0 on each head


def test_train_step_with_the_trace_projection_keeps_the_elementary_basis_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    import collections
    import warnings

    import vfe3.geometry.groups as groups_module
    from tests.profiles import tiny_config
    from vfe3.model.model import VFEModel
    from vfe3.train import build_optimizer, train_step

    reads = collections.Counter()
    original = groups_module._GeneratorTensor.__get__

    def spy(self, instance, owner=None):
        if instance is not None and isinstance(instance._generator_source, _ElementaryBlockBasis):
            reads[instance.name] += 1
        return original(self, instance, owner)

    monkeypatch.setattr(groups_module._GeneratorTensor, "__get__", spy)
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(tiny_config(gauge_group="block_glk", embed_dim=8, n_heads=2, pos_phi="learned",
                                     pos_phi_project_slk=True))
    assert isinstance(model.group._generator_source, _ElementaryBlockBasis)
    optimizer = build_optimizer(model, model.cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    train_step(model, optimizer, scheduler, torch.tensor([[1, 4, 2, 3]]), torch.tensor([[4, 2, 3, 1]]))
    assert not reads, dict(reads)
    assert isinstance(model.group._generator_source, _ElementaryBlockBasis)
