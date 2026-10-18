r"""The exponential-family parameter layer for VFE_3.0.

Importing this package registers the built-in concrete families (``vfe3.families.gaussian``)
so the family registry is populated for any consumer that does
``import vfe3.families`` and then ``get_family(...)`` -- without having to import the
concrete-family module by hand.
"""

from vfe3.families.base import (
    BeliefParams,
    divergence_families,
    family_cov_kind,
    get_family,
    get_functional,
    kl,
    register_family,
    register_functional,
    renyi,
    safe_kl_clamp,
)
from vfe3.families import gaussian as _gaussian  # noqa: F401  (registers the Gaussian families)

__all__ = [
    "BeliefParams",
    "divergence_families",
    "family_cov_kind",
    "get_family",
    "get_functional",
    "kl",
    "register_family",
    "register_functional",
    "renyi",
    "safe_kl_clamp",
]
