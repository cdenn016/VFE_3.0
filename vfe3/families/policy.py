"""Model-owned Gaussian policy adapters; numerical equations stay in their callers."""
from typing import Optional, TypeVar

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


