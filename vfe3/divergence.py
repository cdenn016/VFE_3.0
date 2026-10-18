r"""The divergence seam for VFE_3.0 (parameter-object API over the families layer).

The divergence is the KL, the alpha = 1 Renyi divergence; the Renyi names and the ``alpha`` keyword
are kept for signature stability, and any order other than one raises ``ValueError``. The retired
``divergence_family`` / ``renyi_order`` fields left KL as the only configured divergence, so the
squared Hellinger, Bhattacharyya and Jeffreys functionals and the alpha != 1 kernels were removed
(audit 2026-10-16 F93). The closed forms and the exponential-family abstraction live in
``vfe3.families``; this module re-exports the parameter-typed divergence functionals (``renyi``/``kl``,
which take two ``BeliefParams``) and the families-registry helpers, so existing callers keep importing
them from ``vfe3.divergence`` while operating on parameter objects.

Diagonal Gaussian KL:
    KL(q || p) = 1/2 ( sum_k s_k/t_k + sum_k (mu_t^k - mu_q^k)^2/t_k
                       - K + sum_k log(t_k/s_k) )
"""

from vfe3.families.base import (
    safe_kl_clamp,
    family_cov_kind,
    divergence_families,
    divergence_functionals,
    register_functional,
    get_functional,
    get_family,
    register_family,
    renyi,
    kl,
    renyi_per_coord,
    register_functional_per_coord,
    get_functional_per_coord,
    has_per_coord_functional,
    divergence_functionals_per_coord,
)
from vfe3.families import gaussian as _gaussian     # noqa: F401  (registers the Gaussian families)
from vfe3.families import frame_gaussian as _frame_gaussian   # noqa: F401  (registers gaussian_frame_diagonal)

__all__ = [
    "renyi",
    "kl",
    "renyi_per_coord",
    "safe_kl_clamp",
    "family_cov_kind",
    "divergence_families",
    "divergence_functionals",
    "register_functional",
    "get_functional",
    "register_functional_per_coord",
    "get_functional_per_coord",
    "has_per_coord_functional",
    "divergence_functionals_per_coord",
    "get_family",
    "register_family",
]
