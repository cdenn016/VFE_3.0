r"""The divergence seam for VFE_3.0 (parameter-object API over the families layer).

Renyi alpha-divergence is the primitive; KL is its alpha = 1 special case. The closed
forms and the exponential-family abstraction live in ``vfe3.families``; this module
re-exports the parameter-typed divergence functionals (``renyi``/``kl``, which take two
``BeliefParams``) and the families-registry helpers, so existing callers keep importing
them from ``vfe3.divergence`` while operating on parameter objects.

Diagonal Gaussian KL:
    KL(q || p) = 1/2 ( sum_k s_k/t_k + sum_k (mu_t^k - mu_q^k)^2/t_k
                       - K + sum_k log(t_k/s_k) )
Diagonal Gaussian Renyi (blend sigma_b = (1-a) s + a t):
    D_a(q || p) = 1/2 [ a sum_k (mu_t-mu_q)^2/sigma_b
                        + 1/(a-1) sum_k ((1-a) log s + a log t - log sigma_b) ]
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
    squared_hellinger,
    bhattacharyya,
    jeffreys,
    renyi_per_coord,
    bhattacharyya_per_coord,
    jeffreys_per_coord,
    register_functional_per_coord,
    get_functional_per_coord,
    has_per_coord_functional,
    divergence_functionals_per_coord,
)
from vfe3.families import gaussian as _gaussian     # noqa: F401  (registers the Gaussian families)

__all__ = [
    "renyi",
    "kl",
    "squared_hellinger",
    "bhattacharyya",
    "jeffreys",
    "renyi_per_coord",
    "bhattacharyya_per_coord",
    "jeffreys_per_coord",
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
