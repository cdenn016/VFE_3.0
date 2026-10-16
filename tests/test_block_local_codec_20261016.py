r"""Block-local groups keep only their local basis (gauge-group-buildouts-2026-10-05.md, section 1, M2).

``block_so``, ``block_co`` and ``block_glmc`` store the (m, d, d) local algebra basis as a lazy ``_BlockLocalBasis``;
the dense (H m, K, K) basis is built only when ``GaugeGroup.generators`` is read. Pins, at tiny size in float64 and
float32: no dense basis at construction; the local-basis paths (``embed_algebra``, ``head_block_generators``,
``generator_traces``, ``block_trace_vectors``, the trace projection, the compact transport) equal the dense basis
entry for entry; replacing or moving the basis withdraws or keeps the certificate; and a training step plus the
evaluation diagnostics of a tiny model never read the dense basis.
"""
import collections
import warnings

import pytest
import torch

import vfe3.geometry.groups as groups_module
from tests.profiles import tiny_config
from vfe3.geometry.groups import get_group
from vfe3.geometry.lie_ops import _block_trace_vectors, project_phi_to_slk
from vfe3.geometry.transport import build_factored_transport
from vfe3.model.model import VFEModel

GROUPS = ("block_so", "block_co", "block_glmc")
F64 = torch.float64


def _pair(name: str, dtype: torch.dtype = F64):
    r"""A lazy group and a dense twin whose basis is an ordinary tensor (certificate withdrawn)."""
    lazy = get_group(name)(8, 2, dtype=dtype)
    dense = get_group(name)(8, 2, dtype=dtype)
    dense.generators = dense.generators.clone()                 # replacement withdraws the local certificate
    assert dense.certified_block_local_basis() is None
    return lazy, dense


@pytest.mark.parametrize("name", GROUPS)
def test_no_dense_basis_at_construction(name: str) -> None:
    group = get_group(name)(256, 8)
    assert isinstance(group._generator_source, groups_module._BlockLocalBasis)
    m = group.block_local_basis.shape[0]
    assert group.generator_shape == (8 * m, 256, 256) and group.generator_dtype == torch.float32
    assert group.certified_block_local_basis() is not None


@pytest.mark.parametrize("name", GROUPS)
@pytest.mark.parametrize("dtype", [F64, torch.float32])
def test_local_paths_equal_the_dense_basis(name: str, dtype: torch.dtype) -> None:
    lazy, dense = _pair(name, dtype)
    G = dense.generators                                                        # (n_gen, K, K)
    phi = torch.randn(3, 5, G.shape[0], generator=torch.Generator().manual_seed(0), dtype=dtype)
    torch.testing.assert_close(lazy.embed_algebra(phi), torch.einsum("...a,aij->...ij", phi, G), rtol=0, atol=1e-6)
    blocks = lazy.head_block_generators()                                       # (n_gen, H, d, d)
    torch.testing.assert_close(blocks, dense.head_block_generators(), rtol=0, atol=0)
    torch.testing.assert_close(lazy.generator_traces(), dense.generator_traces(), rtol=0, atol=0)
    torch.testing.assert_close(lazy.block_trace_vectors(), _block_trace_vectors(G, dense.irrep_dims), rtol=0, atol=0)
    torch.testing.assert_close(project_phi_to_slk(phi, lazy, lazy.irrep_dims),
                               project_phi_to_slk(phi, G, dense.irrep_dims), rtol=1e-6, atol=1e-6)
    assert isinstance(lazy._generator_source, groups_module._BlockLocalBasis)   # none of the above built it


@pytest.mark.parametrize("name", GROUPS)
def test_compact_transport_matches_the_dense_basis(name: str) -> None:
    lazy, dense = _pair(name)
    n_gen = lazy.generator_shape[0]
    phi = 0.3 * torch.randn(2, 4, n_gen, generator=torch.Generator().manual_seed(1), dtype=F64)
    right = 0.2 * torch.randn(2, 4, n_gen, generator=torch.Generator().manual_seed(2), dtype=F64)
    a = build_factored_transport(phi, lazy, compact_blocks=True, right_phi=right)
    b = build_factored_transport(phi, dense, compact_blocks=True, right_phi=right)
    torch.testing.assert_close(a.exp_blocks, b.exp_blocks, rtol=1e-12, atol=1e-13)
    torch.testing.assert_close(a.inv_blocks, b.inv_blocks, rtol=1e-12, atol=1e-13)
    assert isinstance(lazy._generator_source, groups_module._BlockLocalBasis)


def test_reading_moving_and_replacing_the_basis() -> None:
    group = get_group("block_glmc")(8, 2)
    G = group.generators                                                        # materializes once
    assert isinstance(G, torch.Tensor) and group.generators is G
    assert group.certified_block_local_basis() is not None                       # the materialized copy keeps it
    group.apply_generators(lambda t: t.to(F64))
    assert group.generator_dtype == F64 and group.certified_block_local_basis() is not None
    lazy = get_group("block_glmc")(8, 2)
    lazy.apply_generators(lambda t: t.to(F64))
    assert isinstance(lazy._generator_source, groups_module._BlockLocalBasis) and lazy.generator_dtype == F64
    with torch.no_grad():
        group.generators.mul_(2.0)                                              # in-place mutation withdraws it
    assert group.certified_block_local_basis() is None
    phi = torch.randn(4, group.generator_shape[0], dtype=F64)
    torch.testing.assert_close(group.embed_algebra(phi), torch.einsum("na,aij->nij", phi, group.generators))


@pytest.mark.parametrize("name", GROUPS)
def test_train_step_and_diagnostics_never_read_the_dense_basis(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from vfe3.train import build_optimizer, train_step
    reads = collections.Counter()
    original = groups_module._GeneratorTensor.__get__

    def spy(self, instance, owner=None):
        if instance is not None and isinstance(instance._generator_source, groups_module._BlockLocalBasis):
            reads[instance.name] += 1
        return original(self, instance, owner)

    monkeypatch.setattr(groups_module._GeneratorTensor, "__get__", spy)
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VFEModel(tiny_config(gauge_group=name, embed_dim=8, n_heads=2, pos_phi="learned",
                                     pos_phi_project_slk=True))
    optimizer = build_optimizer(model, model.cfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    train_step(model, optimizer, scheduler, torch.tensor([[1, 4, 2, 3]]), torch.tensor([[4, 2, 3, 1]]))
    model.eval()
    with torch.no_grad():
        model.diagnostics(torch.tensor([[1, 4, 2, 3]]))
    assert not reads, dict(reads)
    assert isinstance(model.group._generator_source, groups_module._BlockLocalBasis)
