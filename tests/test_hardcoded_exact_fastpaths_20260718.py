"""Architectural contract for exact fast paths that are mandatory in production."""

from tests.profiles import test_config

import ast
from dataclasses import fields
from pathlib import Path

import pytest

from vfe3.config import VFE3Config
from vfe3.model.model import VFEModel


ROOT = Path(__file__).resolve().parents[1]
FORMER_CONFIG_FIELDS = frozenset({
    "compact_phi_block_transport",
    "reuse_pairwise_kl_stats",
    "transport_mean_per_head",
})
PRODUCTION_MODULES = (
    "vfe3/model/block.py",
    "vfe3/model/model.py",
    "vfe3/viz/extract.py",
)


def test_public_configuration_and_drivers_have_no_fastpath_controls() -> None:
    assert FORMER_CONFIG_FIELDS.isdisjoint(field.name for field in fields(VFE3Config))
    for relative in ("vfe3/config.py", "train_vfe3.py", "scaling.py", "ablation.py"):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert all(name not in source for name in FORMER_CONFIG_FIELDS), relative


def test_production_does_not_read_removed_fastpath_attributes() -> None:
    for relative in PRODUCTION_MODULES:
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        reads = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr in FORMER_CONFIG_FIELDS
        }
        assert reads == set(), relative


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, True),
    ],
)
def test_compact_phi_route_is_automatic(
    overrides: dict[str, object],
    expected:  bool,
) -> None:
    values: dict[str, object] = {
        "vocab_size": 9,
        "embed_dim": 4,
        "n_heads": 2,
        "max_seq_len": 3,
        "n_layers": 1,
        "n_e_steps": 1,
    }
    values.update(overrides)
    model = VFEModel(test_config(**values)).eval()
    assert model._compact_phi_blocks_enabled() is expected


