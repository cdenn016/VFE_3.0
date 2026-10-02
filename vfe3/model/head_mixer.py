r"""Schur-commutant head mixer for VFE_3.0 (opt-in).

Without labels (the legacy equal-dims form) mixes ``n`` equal-size gauge-irrep blocks with one
learned invertible matrix :math:`A = \exp(\Delta) \in \mathrm{GL}(n)` embedded as
:math:`\mathrm{kron}(A, I_d)`, where :math:`n` is the number of blocks and :math:`d` the
(shared) block dimension. With ``irrep_labels`` the class builds one :math:`A_t = \exp(\Delta_t)`
per maximal run of equal-labeled blocks -- the full linear commutant of a mixed-irrep tower for
real-type irreps -- so unequal block dimensions no longer raise a dim-collision hazard provided
they carry different labels. Under ``block_glk`` the blocks are the ``n_heads`` heads, so the
mixer mixes heads. Applied to the mean and through the configured family's dispersion hook:

.. math::
    M    = \mathrm{blockdiag}_t(A_t \otimes I_{d_t}) \in \mathbb{R}^{K \times K}, \qquad
    \mu' = M\mu, \qquad
    \Sigma' = M\Sigma M^{\top},

with the diagonal-Gaussian closed form (the diagonal-of-sandwich approximation already used
throughout V3 when ``diagonal_covariance=True``)

.. math::
    \sigma'[m, c] = \sum_n A_t[m, n]^2 \sigma[n, c]

applied block-by-block for each component :math:`t`. Factorized Laplace instead uses the
moment-matched scale :math:`b'[m,c] = \sqrt{\sum_n A_t[m,n]^2 b[n,c]^2}`.

Initialization is exactly the identity (:math:`\Delta_t = 0`, stored as a matrix-log coordinate
with an exact no-grad passthrough), so a model with the mixer enabled is bitwise indistinguishable
from the mixer-disabled path at step 0.

Gauge equivariance: :math:`\mathrm{kron}(A, I_d)` commutes with a block-diagonal gauge
:math:`\mathrm{diag}(h_1, \ldots, h_n)` ONLY when the gauge is TIED (:math:`h_k = h_0` for all
:math:`k`). V3's ``block_glk`` generators (``generate_glk_multihead``) give each head its OWN
independent ``gl(d_head)`` sub-algebra -- an UNTIED gauge -- so the mixer does NOT commute with
the per-head gauge action and breaks strict gauge equivariance there. The deviation is zero at
the identity init and grows as :math:`A` drifts from :math:`I` during training. This is an
accepted, opt-in departure (the no-mixer path is the default and stays equivariant).

The ``tied_block_glk`` group (generators ``kron(I_n, gl(d))``, one shared frame across heads)
restores exact equivariance: under a tied gauge :math:`\Omega = \mathrm{kron}(I_n, h)`,
:math:`M = \mathrm{kron}(A, I_d)` commutes with :math:`\Omega`, so the FULL-COVARIANCE mixer is
exactly equivariant -- :math:`\mathrm{mix}(\Omega\mu, \Omega\Sigma\Omega^\top) = (\Omega M\mu,
\Omega M\Sigma M^\top \Omega^\top)` (pinned by ``test_head_mixer_equivariant_under_tied_gauge_full_cov``).
This is a statement about the MIXER OPERATION, not a claim that the whole model is gauge-equivariant.
CAVEAT: the diagonal closed form :math:`\sigma'[m] = \sum_n A[m,n]^2 \sigma[n]` is equivariant only
under DIAGONAL gauges (the diagonal-of-sandwich approximation V3 already uses when
``diagonal_covariance=True``), not under a general tied gauge.
"""

from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn

from vfe3.families.base import get_family


def _component_runs(
    irrep_dims:   Sequence[int],
    irrep_labels: Optional[Sequence[str]],
) -> List[Tuple[int, int]]:
    """Validate the mixer layout and return maximal equal-label block runs."""
    if len(irrep_dims) < 2:
        raise ValueError(
            f"HeadMixer needs >= 2 blocks to mix, got irrep_dims={list(irrep_dims)}; a single-block "
            f"group (glk / so_k) has nothing to mix. Use block_glk (n_heads >= 2). (If using "
            f"block_glk with cross_couplings, the off-block basis collapses irrep_dims to [K]; "
            f"remove cross_couplings or disable the mixer.)"
        )
    if irrep_labels is None:
        if len(set(irrep_dims)) != 1:
            raise ValueError(
                f"HeadMixer needs equal-size blocks for kron(A, I_d), got "
                f"irrep_dims={list(irrep_dims)}. A labeled irrep tower (so_n) mixes per "
                f"isotypic component instead."
            )
        return [(0, len(irrep_dims))]
    if len(irrep_labels) != len(irrep_dims):
        raise ValueError(
            f"irrep_labels has {len(irrep_labels)} entries but there are "
            f"{len(irrep_dims)} irrep blocks"
        )
    runs, i = [], 0
    while i < len(irrep_dims):
        j = i
        while j < len(irrep_dims) and irrep_labels[j] == irrep_labels[i]:
            j += 1
        runs.append((i, j))
        i = j
    for i, j in runs:
        if len(set(irrep_dims[i:j])) != 1:
            raise ValueError(
                f"blocks {i}:{j} share label {irrep_labels[i]!r} but have unequal dims "
                f"{list(irrep_dims[i:j])}; copies of one irrep must share its dimension"
            )
    return runs


class HeadMixer(nn.Module):
    r"""Isotypic per-component mixer: one :math:`A_t = \exp(\Delta_t)` per maximal run of
    equal-labeled blocks, embedded as :math:`\mathrm{blockdiag}_t(A_t \otimes I_{d_t})` --
    the full linear commutant of the tower for real-type irreps. Without labels the whole
    group must be one equal-dims component (the legacy behavior, byte-identical)."""

    @staticmethod
    def parameter_count(
        irrep_dims:   Sequence[int],
        irrep_labels: Optional[Sequence[str]] = None,
    ) -> int:
        """Exact trainable scalar count for the validated component layout."""
        return sum((j - i) ** 2 for i, j in _component_runs(irrep_dims, irrep_labels))

    def __init__(
        self,
        irrep_dims:   List[int],                      # gauge block sizes
        irrep_labels: Optional[List[str]] = None,     # per-block labels; None -> legacy equal-dims

        *,
        family:       str = "gaussian_diagonal",     # registered belief-family key
    ) -> None:
        super().__init__()
        self.family_name = family
        self._family = get_family(family)
        runs = _component_runs(irrep_dims, irrep_labels)
        # components: (coordinate start, copies m, block dim d); spec layout makes runs contiguous
        starts = [0]
        for d in irrep_dims:
            starts.append(starts[-1] + d)
        self.components = [(starts[i], j - i, irrep_dims[i]) for i, j in runs]
        self.mixer_deltas = nn.ParameterList(
            nn.Parameter(torch.zeros(m, m)) for _, m, _ in self.components
        )
        self._identity_states = tuple(
            (id(delta), delta._version) for delta in self.mixer_deltas
        )
        self._identity_for_forward = True
        if irrep_labels is not None:
            # Silent-degeneracy warnings (audit 2026-06-09 overnight PP6/DB3). Runs are
            # CONTIGUOUS by construction, so (a) a label appearing in two non-adjacent runs
            # gets NO cross-copy mixing (each run mixes only within itself -- a proper
            # subspace of the tower's true commutant), and (b) an all-length-1 tower
            # degenerates to per-block scalar gains (Schur-forced for inequivalent irreps,
            # but worth a heads-up when the user enabled the mixer expecting mixing).
            import warnings
            run_labels = [irrep_labels[i] for i, _j in runs]
            if len(set(run_labels)) != len(run_labels):
                dup = sorted({lab for lab in run_labels if run_labels.count(lab) > 1})
                warnings.warn(
                    f"HeadMixer: label(s) {dup} appear in NON-ADJACENT blocks of the irrep "
                    f"spec, so their copies are mixed per contiguous run only (no cross-copy "
                    f"mixing between the separated runs). List equal-label copies adjacently "
                    f"in irrep_spec to mix them jointly.",
                    stacklevel=2,
                )
            if len(set(irrep_labels)) == len(irrep_labels):
                warnings.warn(
                    "HeadMixer: every irrep label in this tower is distinct, so the isotypic "
                    "mixer degenerates to one scalar gain per block (Schur: no linear "
                    "equivariant map exists between inequivalent irreps). Use multiplicity "
                    "> 1 (or use_cg_coupling=True for bilinear cross-type flow) if you "
                    "intended inter-block mixing.",
                    stacklevel=2,
                )

    @property
    def mixer_delta(self) -> nn.Parameter:
        r"""Back-compat accessor for the single-component (legacy equal-dims) mixer."""
        if len(self.mixer_deltas) != 1:
            raise AttributeError("mixer_delta is single-component only; use mixer_deltas")
        return self.mixer_deltas[0]

    def _A(self, t: int) -> torch.Tensor:
        return torch.linalg.matrix_exp(self.mixer_deltas[t])

    def is_identity(self) -> bool:
        r"""Exact diagnostic predicate; forward uses the synchronization-free version cache."""
        return all(bool((d.detach() == 0).all().item()) for d in self.mixer_deltas)

    def _identity_shortcut_available(self) -> bool:
        r"""Whether identity passthrough is still certified without reading tensor values.

        Parameters initialize exactly to zero. Replacement or a version-visible in-place write
        invalidates this certificate before training, while the first grad-enabled forward
        invalidates it unconditionally. This preserves the no-grad object-identity contract at
        initialization without relying on optimizer-specific version-counter behavior.
        """
        states = tuple((id(delta), delta._version) for delta in self.mixer_deltas)
        if states != self._identity_states:
            self._identity_states = states
            self._identity_for_forward = False
        return self._identity_for_forward

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        r"""Remap the pre-isotypic key ``mixer_delta`` -> ``mixer_deltas.0`` so strict
        state-dict key loading for single-component mixers written before the ParameterList
        refactor continues to work."""
        old = prefix + "mixer_delta"
        if old in state_dict and len(self.mixer_deltas) == 1:
            state_dict[prefix + "mixer_deltas.0"] = state_dict.pop(old)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._identity_states = tuple(
            (id(delta), delta._version) for delta in self.mixer_deltas
        )
        self._identity_for_forward = False

    def forward(
        self,
        mu:    torch.Tensor,             # (..., K) belief means
        sigma: torch.Tensor,             # (..., K) diagonal variances OR (..., K, K) full covariance

        *,
        _need_covariance: bool = True,    # private final-loss consumer contract; public calls mix both
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Grad-free identity short-circuit only: under autograd a zero-delta short-circuit
        # would sever mixer_deltas from the graph and the zero-init deltas could never train
        # (audit 2026-06-09 overnight CR3 verifier correction).
        if torch.is_grad_enabled():
            self._identity_for_forward = False
        elif self._identity_shortcut_available():
            return mu, sigma
        mu_parts, sig_parts = [], []
        for t, (s, m, d) in enumerate(self.components):
            # Cast to the INPUT dtype (audit 2026-06-09 overnight DB1): the parameter dtype
            # follows the module (.double()/.float()), and an uncast A crashes the einsum
            # on any module/input dtype mixture; a same-dtype .to is a free no-op.
            A = self._A(t).to(dtype=mu.dtype, device=mu.device)
            blk = mu[..., s:s + m * d].reshape(*mu.shape[:-1], m, d)
            mu_parts.append(torch.einsum("mn,...nd->...md", A, blk)
                            .reshape(*mu.shape[:-1], m * d))
            if _need_covariance and sigma.dim() == mu.dim():     # diagonal closed form
                sblk = sigma[..., s:s + m * d].reshape(*sigma.shape[:-1], m, d)
                sig_parts.append(self._family.mix_dispersion(sblk, A)
                                 .reshape(*sigma.shape[:-1], m * d))
        mu_out = torch.cat(mu_parts, dim=-1)
        if not _need_covariance:
            return mu_out, sigma                                # internal placeholder, never a public mixed belief
        if sigma.dim() == mu.dim():
            return mu_out, torch.cat(sig_parts, dim=-1)
        # full covariance: exact sandwich M Sigma M^T with the block-diagonal commutant M
        M = self._dense_m(sigma.device, sigma.dtype)             # (K, K)
        full_family = self._family if self._family.cov_kind == "full" else get_family("gaussian_full")
        return mu_out, full_family.mix_dispersion(sigma, M)

    def _dense_m(self, device, dtype) -> torch.Tensor:
        r"""blockdiag_t(A_t kron I_d) materialized once per call (K x K, full-cov path only).

        Deliberately NOT cached: M is differentiable through the trainable deltas, so a
        (device, dtype)-keyed cache would serve stale gradients during training (audit
        2026-06-09 overnight CR4); the rebuild is O(K^2) against the O(K^3) sandwich it feeds."""
        K = sum(m * d for _, m, d in self.components)
        M = torch.zeros(K, K, device=device, dtype=dtype)
        for t, (s, m, d) in enumerate(self.components):
            M[s:s + m * d, s:s + m * d] = torch.kron(
                self._A(t).to(device=device, dtype=dtype), torch.eye(d, device=device, dtype=dtype))
        return M
