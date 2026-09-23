"""Model-owned Gaussian policy adapters; numerical equations stay in their callers."""
from typing import Any, Optional, TypeVar

import torch

from vfe3.families.base import BeliefParams
from vfe3.families.gaussian import FullGaussian

_Family = TypeVar("_Family", bound=BeliefParams)


def family_instance(
    mu:               torch.Tensor,
    sigma:            torch.Tensor,
    family:           type[_Family],
    precision_policy: Optional[str],
) -> _Family:
    """Apply an explicit policy only to the exact built-in full Gaussian."""
    if family is FullGaussian and precision_policy is not None:
        return family(mu, sigma, _precision_policy=precision_policy)
    return family(mu, sigma)


def coupling_energy(
    family:           type[BeliefParams],
    *args:            Any,
    score_backend:    str                    = "pairwise",
    log_prior:        Optional[torch.Tensor] = None,
    precision_policy: Optional[str]          = None,
    **kwargs:         Any,
) -> torch.Tensor:
    """Pass full-Gaussian policy/support keywords without extending custom-family APIs."""
    if family is FullGaussian:
        kwargs["score_backend"] = score_backend
        if precision_policy is not None:
            kwargs["precision_policy"] = precision_policy
        if log_prior is not None:
            kwargs["pair_support"] = ~torch.isneginf(log_prior)
    return family.coupling_energy(*args, **kwargs)
