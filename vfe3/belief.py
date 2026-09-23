r"""The Gaussian belief tuple for VFE_3.0.

``BeliefState`` is a ``typing.NamedTuple`` carrying the three defining belief tensors
(``mu``, ``sigma``, ``phi``) followed by optional trailing fields that only some
configurations populate: the exact positional right factor. Every construction uses keyword
arguments and every read uses attribute access, so trailing defaulted fields leave all of
construction, attribute access, ``_replace``, indexing and iteration intact.
"""

from typing import NamedTuple, Optional

import torch


class BeliefState(NamedTuple):
    """A per-token Gaussian belief q_i = N(mu_i, Sigma_i) with gauge frame phi_i."""

    mu:    torch.Tensor                       # (..., N, K) means
    sigma: torch.Tensor                       # (..., N, K) diagonal variances (or (..., N, K, K) full)
    phi:   torch.Tensor                       # (..., N, n_gen) gauge-frame coordinates

    right_phi: Optional[torch.Tensor] = None  # (..., N, n_gen) exact right positional factor exp(right_phi.G)
