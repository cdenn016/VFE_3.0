r"""The exponential-family parameter layer for VFE_3.0.

Importing this package registers the built-in concrete families (``vfe3.families.gaussian``,
``vfe3.families.laplace``, ``vfe3.families.frame_gaussian``,
``vfe3.families.exact_congruence``) so the family registry is populated
for any consumer that does
``import vfe3.families`` and then ``get_family(...)`` -- without having to import the
concrete-family module by hand.
"""

from vfe3.families.base import (
    BeliefParams,
    FunctionalDisplayMetadata,
    divergence_families,
    family_cov_kind,
    get_family,
    get_functional,
    get_functional_display,
    kl,
    register_family,
    register_functional,
    renyi,
    safe_kl_clamp,
)
from vfe3.families import gaussian as _gaussian  # noqa: F401  (registers the Gaussian families)
from vfe3.families import laplace as _laplace    # noqa: F401  (registers the Laplace family)
from vfe3.families import frame_gaussian as _frame_gaussian  # noqa: F401  (frame-intrinsic Gaussian)
from vfe3.families import exact_congruence as _exact_congruence  # noqa: F401  (exact-congruence Gaussian)

__all__ = [
    "BeliefParams",
    "FunctionalDisplayMetadata",
    "divergence_families",
    "family_cov_kind",
    "get_family",
    "get_functional",
    "get_functional_display",
    "kl",
    "register_family",
    "register_functional",
    "renyi",
    "safe_kl_clamp",
]
