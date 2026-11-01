r"""Clebsch-Gordan between-block coupling for irrep towers (opt-in; default off).

The lowest-order exactly equivariant cross-type coupling (linear maps vanish by Schur): an
equivariant linear map between inequivalent irreps is zero, so the coupling is BILINEAR
through the numerically solved CG intertwiners,

    mu'^(c,r) = mu^(c,r) + sum_p w_p C_p( mu^(a,i), mu^(b,j) ),

one learned scalar w_p per path (source copy pair x target copy x multiplicity slot),
zero-initialized so step 0 is byte-identical to the coupling-off path. Equivariance holds
for ANY weights because the weights multiply intertwiners. Covariance either passes through
untouched or uses the explicitly approximate first-order delta closure
sym(J Sigma J^T) + floor Sigma; a bilinear map of Gaussians has no exact Gaussian pushforward.
NEURAL-NETWORK EXCEPTION (sanctioned, default-off).
"""

import math
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
from torch import nn

from vfe3.geometry.cg import cg_intertwiners, cg_selection
from vfe3.runtime import ieee_fp32_matmul


class CGMomentResult(NamedTuple):
    r"""Post-CG belief moments plus the exact analytic Jacobian of the mean map.

    ``mu`` is mu + delta (the bilinear CG update); ``sigma`` is the pushed-forward covariance
    (untouched under ``cg_covariance_mode='passthrough'``, the symmetrized delta-method image
    sym(J Sigma J^T) + floor Sigma under 'delta_full'); ``jacobian`` is J = d mu_out / d mu at
    the current mean, (..., K, K). At zero path weights J is exactly the identity and mu equals
    the input; delta-full covariance is then (1 + floor) Sigma while passthrough covariance is
    exactly the input Sigma."""
    mu:       torch.Tensor
    sigma:    torch.Tensor
    jacobian: torch.Tensor


def cg_moment_energy_rows(
    pre_mu:     torch.Tensor,             # (..., N, K) pre-CG belief means
    pre_sigma:  torch.Tensor,             # (..., N, K) or (..., N, K, K) pre-CG covariance
    post_mu:    torch.Tensor,             # (..., N, K) post-CG belief means
    post_sigma: torch.Tensor,             # (..., N, K) or (..., N, K, K) post-CG covariance

    *,
    kl_max:            float = 100.0,
    eps:               float = 1e-6,
    family:            str   = "gaussian_diagonal",
) -> torch.Tensor:                        # (..., N) KL(q_post || q_pre)
    r"""Per-token post-CG moment KL(q_post || q_pre) via the active-family divergence seam.

    The q-only CG moment regularizer: the full-K self-divergence of the post-CG belief against its
    pre-CG value (means-only under 'passthrough', mean-and-covariance under 'delta_full'). NOT a
    second hierarchical total -- it reads only the belief (q) channel, never s/p/h. Family-agnostic:
    ``family`` selects the covariance kernel; the divergence is the KL (audit 2026-10-16 F93; the
    retired ``divergence_family`` and ``renyi_order`` keywords were removed, audit 2026-10-17 item 72).
    """
    from vfe3.families.base import get_family
    from vfe3.free_energy import self_divergence
    fam = get_family(family)
    return self_divergence(
        fam(post_mu, post_sigma), fam(pre_mu, pre_sigma),
        kl_max=kl_max, eps=eps,
    )


class CGPaths(NamedTuple):
    r"""The admissible CG triples of a labeled tower, their intertwiners and the coupling paths they span."""
    triples:      List[Tuple[str, str, str, int]]                  # (a, b, c, n_mult), a <= b
    intertwiners: List[torch.Tensor]                               # per triple (n_mult, d_c, d_a * d_b) float64
    paths:        List[Tuple[int, int, int, int, int, int, int, int]]   # (s_a, d_a, s_b, d_b, s_c, d_c, triple, slot)
    path_types:   List[Tuple[str, str, str]]                       # (a, b, c) per path


def cg_coupling_paths(
    group_n:      int,                       # N of SO(N) or SU(N)
    algebra:      str,                       # 'so' | 'su'
    irrep_dims:   List[int],                 # per-block dims
    irrep_labels: List[str],                 # per-block labels

    *,
    atol:         float = 1e-8,              # CG null-space solve tolerance (shared with the prune)
) -> CGPaths:
    r"""Enumerate the coupling paths of :class:`CGCoupling` (an empty ``paths`` when the labels admit none).

    One path per source copy pair (unordered for equal labels), target copy and multiplicity slot of every admissible
    triple :math:`V_a\otimes V_b\to V_c`. A self pair (one source copy twice) through a swap-antisymmetric slot is
    pruned: :math:`C_m(x, x) = 0` for every :math:`x`, so its weight would never receive a gradient (audit 2026-06-09
    overnight F10/F13/F20). The prune threshold follows the solve's ``atol`` (audit 2026-06-13 L20). The intertwiners
    are cached by :func:`vfe3.geometry.cg.cg_intertwiners`, so ``VFE3Config`` validation and the module build share
    one solve.
    """
    starts = [0]
    for d in irrep_dims:
        starts.append(starts[-1] + d)
    blocks = list(zip(irrep_labels, starts[:-1], irrep_dims))     # (label, start, d)
    triples = cg_selection(group_n, algebra=algebra, labels=list(irrep_labels), atol=atol)
    intertwiners = [cg_intertwiners(group_n, algebra=algebra, label_a=a, label_b=b, label_c=c, atol=atol)
                    for (a, b, c, _n) in triples]                  # (n_mult, dc, da*db) each
    paths: List[Tuple[int, int, int, int, int, int, int, int]] = []
    path_types: List[Tuple[str, str, str]] = []
    for t, ((a, b, c, n_mult), C64) in enumerate(zip(triples, intertwiners)):
        srcs_a = [(s, d) for lab, s, d in blocks if lab == a]
        srcs_b = [(s, d) for lab, s, d in blocks if lab == b]
        tgts_c = [(s, d) for lab, s, d in blocks if lab == c]
        for ia, (sa, da) in enumerate(srcs_a):
            for jb, (sb, db) in enumerate(srcs_b):
                if a == b and jb < ia:                            # unordered copies
                    continue
                for (sc, dc) in tgts_c:
                    for m in range(n_mult):
                        # Slots that are symmetric or mixed under copy swap stay (mixed can only arise from an eigh
                        # basis rotation inside an n_mult > 1 null space and is live). For the shipped towers the
                        # split is clean (antisymmetric ~1e-12 in float64 vs symmetric ~O(1)).
                        if sa == sb:
                            Cm = C64[m].reshape(dc, da, db)
                            if (Cm + Cm.transpose(-1, -2)).abs().max() < max(1e-10, 10.0 * atol):
                                continue
                        paths.append((sa, da, sb, db, sc, dc, t, m))
                        path_types.append((a, b, c))
    return CGPaths(triples, intertwiners, paths, path_types)


class CGCoupling(nn.Module):
    r"""Bilinear CG coupling over the blocks of a labeled irrep tower."""

    def __init__(
        self,
        group_n:      int,                       # N of SO(N) or SU(N)
        algebra:      str,                       # 'so' | 'su'
        irrep_dims:   List[int],                 # per-block dims
        irrep_labels: Optional[List[str]],       # per-block labels (REQUIRED non-None)

        *,
        atol:               float = 1e-8,        # CG null-space solve tolerance (shared with the prune)
        cg_covariance_mode: str   = "passthrough",  # 'passthrough' | 'delta_full' (delta-method J Sigma J^T)
        cg_covariance_floor: float = 1e-6,       # positive covariant floor: floor * input Sigma
    ) -> None:
        super().__init__()
        if not (math.isfinite(cg_covariance_floor) and cg_covariance_floor > 0.0):
            raise ValueError(
                "cg_covariance_floor must be finite and strictly positive, got "
                f"{cg_covariance_floor!r}")
        self.cg_covariance_mode = cg_covariance_mode
        self.cg_covariance_floor = float(cg_covariance_floor)
        if irrep_labels is None:
            raise ValueError(
                "CGCoupling requires a labeled irrep tower (gauge_group 'so_n' or 'su_n')"
            )

        # one stacked intertwiner buffer per admissible type triple
        self._group_n, self._algebra, self._atol = group_n, algebra, atol
        enumerated = cg_coupling_paths(group_n, algebra, list(irrep_dims), list(irrep_labels), atol=atol)
        self._triple_index = {}
        for t, ((a, b, c, _n), C) in enumerate(zip(enumerated.triples, enumerated.intertwiners)):
            # Stored float64 (the solver's construction precision). A model-wide .float()/
            # .double() converts buffers like everything else; forward re-casts to mu's dtype
            # from the cast cache either way, so numerics follow the RUNTIME dtype -- float64
            # storage just avoids a lossy fp32 round-trip before the cast.
            self.register_buffer(f"cg_{t}", C)                        # (n_mult, dc, da*db)
            self._triple_index[(a, b, c)] = t

        # paths: source copy pair (i <= j for equal labels) x target copy x multiplicity slot, with the
        # swap-antisymmetric self pairs pruned (cg_coupling_paths). path_types mirrors paths 1:1
        # (introspection/debug only; no production consumer).
        self.paths: List[Tuple[int, int, int, int, int, int, int, int]] = enumerated.paths
        self.path_types: List[Tuple[str, str, str]] = enumerated.path_types
        if not self.paths:
            raise ValueError(
                f"CGCoupling found no admissible CG paths for labels {irrep_labels} "
                f"(algebra {algebra!r}, N={group_n}); disable use_cg_coupling"
            )
        self.path_weights = nn.Parameter(torch.zeros(len(self.paths)))
        self._cast_cache: Dict[Tuple[torch.dtype, torch.device], List[torch.Tensor]] = {}
        # maps (dtype, device) -> list of cast intertwiner stacks, lazily built, cleared on _apply

        # Forward batching (audit 2026-06-09 overnight F2/DB2): group the paths that share one
        # intertwiner slot AND one target block -- they differ only in the source copy pair, and
        # delta_c = C_m @ sum_p w_p vec(x_p (x) y_p) is linear in the weighted source outer
        # products -- so the per-path Python loop collapses to one stacked outer product, one
        # weighted sum, and one intertwiner einsum per group. Same arithmetic up to float
        # summation order.
        grouped: Dict[Tuple[int, int, int, int], List[int]] = {}
        for p, (sa, da, sb, db, sc, dc, t, m) in enumerate(self.paths):
            grouped.setdefault((t, m, sc, dc), []).append(p)
        self._groups: List[Tuple[int, int, int, int, List[int],
                                 List[Tuple[int, int, int, int]]]] = []
        for (t, m, sc, dc), idxs in grouped.items():
            srcs = [self.paths[p][:4] for p in idxs]                  # (sa, da, sb, db) per path
            self._groups.append((t, m, sc, dc, idxs, srcs))
        # The path indices of every group as device tensors, built once per device (_group_indices).
        self._index_cache: Dict[torch.device, List[torch.Tensor]] = {}

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs) -> None:
        r"""Load the persisted intertwiners, then re-verify them against this build's irreps.

        The block bases are rebuilt at every load while the ``cg_*`` buffers come from the
        checkpoint, so a basis that changed since the save would pair old intertwiners with new
        generators. The build-time equivariance check ran on the rebuilt buffers; this repeats it
        on the loaded ones, C (rho_a (x) rho_b)(X) = rho_c(X) C for every generator X.
        """
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        from vfe3.geometry.cg import _defining
        from vfe3.geometry.irreps import irrep_generators
        G_def = _defining(self._group_n, self._algebra)
        reps: dict = {}

        def rho(label: str) -> torch.Tensor:
            if label not in reps:
                reps[label] = irrep_generators(G_def, algebra=self._algebra, label=label)
            return reps[label]

        gate = max(1e-6, 10.0 * float(self._atol))
        for (a, b, c), t in self._triple_index.items():
            C = getattr(self, f"cg_{t}").detach().to(device="cpu", dtype=torch.float64)
            if C.shape[0] == 0:
                continue
            ra, rb, rc = rho(a), rho(b), rho(c)
            I_a = torch.eye(ra.shape[-1], dtype=torch.float64)
            I_b = torch.eye(rb.shape[-1], dtype=torch.float64)
            for k in range(G_def.shape[0]):
                rab = torch.kron(ra[k], I_b) + torch.kron(I_a, rb[k])
                residual = float((C @ rab - torch.einsum("ij,mjk->mik", rc[k], C)).abs().max())
                if residual > gate:
                    error_msgs.append(
                        f"{prefix}cg_{t}: the loaded ({a}, {b}) -> {c} intertwiner is not equivariant "
                        f"for this build's irrep basis (residual {residual:.3e} > {gate:.1e}); the "
                        f"checkpoint was saved in a different basis (such as the retired 'eigh' basis)")
                    break

    def _apply(self, fn, recurse=True):
        # .to()/.float()/.cuda() move or convert the buffers; drop the stale cast and index caches.
        self._cast_cache.clear()
        self._index_cache.clear()
        return super()._apply(fn, recurse)

    def _group_indices(self, device: torch.device) -> List[torch.Tensor]:
        r"""The path indices of every forward group as long tensors on ``device``, cached outside inference mode.

        Indexing the weights with a Python list builds a host index tensor and copies it to the device on every
        group of every call (audit 2026-10-31 A191); these are built once per device, as the intertwiner casts
        are (:meth:`_cast_buffers`). Built under ``torch.inference_mode()`` they are returned uncached, since a later
        gradient-enabled call cannot save an inference tensor for backward.
        """
        cached = self._index_cache.get(device)
        if cached is None:
            cached = [torch.tensor(idxs, dtype=torch.long, device=device)
                      for (_t, _m, _sc, _dc, idxs, _srcs) in self._groups]
            if not torch.is_inference_mode_enabled():
                self._index_cache[device] = cached
        return cached

    def _cast_buffers(self, dtype: torch.dtype, device: torch.device) -> List[torch.Tensor]:
        r"""The intertwiner buffers cast to ``(dtype, device)``, cached outside inference mode.

        A cast built under ``torch.inference_mode()`` is an inference tensor that a later
        gradient-enabled call cannot save for backward, so it is returned uncached (audit 2026-10-25
        F13); a cached ordinary cast is still reused inside inference mode.
        """
        key = (dtype, device)
        cached = self._cast_cache.get(key)
        if cached is None:
            n_triples = len(self._triple_index)
            cached = [getattr(self, f"cg_{t}").to(dtype=dtype, device=device)
                      for t in range(n_triples)]
            if not torch.is_inference_mode_enabled():
                self._cast_cache[key] = cached
        return cached

    def is_identity(self) -> bool:
        return bool((self.path_weights.detach() == 0).all().item())

    def forward(
        self,
        mu:    torch.Tensor,             # (..., K) belief means
        sigma: torch.Tensor,             # (..., K) or (..., K, K); passes through UNTOUCHED
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # NOTE the post-coupling tuple (mu + delta, sigma) is NOT a congruence image of the
        # input belief -- sigma is deliberately untouched (means-only phase; see module
        # docstring). Grad-free identity short-circuit only: under autograd a zero-weight
        # short-circuit would sever path_weights from the graph and the zero-init weights
        # could never train (audit 2026-06-09 overnight CR2 verifier correction). The shortcut is
        # restricted to passthrough covariance: at zero weights 'delta_full' still adds its
        # covariant floor, sym(Sigma) + floor * Sigma, so it takes the ordinary moment path in every
        # grad mode (audit 2026-10-25 F14).
        if (self.cg_covariance_mode != "delta_full" and not torch.is_grad_enabled()
                and self.is_identity()):
            return mu, sigma
        if self.cg_covariance_mode == "delta_full":
            res = self.forward_moments(mu, sigma)                 # delta-method covariance pushforward
            return res.mu, res.sigma
        cg = self._cast_buffers(mu.dtype, mu.device)              # per-(dtype, device), cached
        w = self.path_weights.to(dtype=mu.dtype)                  # one cast, differentiable
        delta = torch.zeros_like(mu)
        for (t, m, sc, dc, idxs, srcs), path_index in zip(self._groups, self._group_indices(mu.device)):
            x = torch.stack([mu[..., sa:sa + da] for (sa, da, _sb, _db) in srcs], dim=0)
            y = torch.stack([mu[..., sb:sb + db] for (_sa, _da, sb, db) in srcs], dim=0)
            xy = (x.unsqueeze(-1) * y.unsqueeze(-2)) \
                .reshape(len(srcs), *x.shape[1:-1], -1)           # (P, ..., da*db)
            wsum = torch.einsum("p,p...d->...d", w.index_select(0, path_index), xy)   # weighted source moment
            delta[..., sc:sc + dc] = delta[..., sc:sc + dc] \
                + torch.einsum("cd,...d->...c", cg[t][m], wsum)
        return mu + delta, sigma

    def forward_moments(
        self,
        mu:    torch.Tensor,             # (..., K) belief means
        sigma: torch.Tensor,             # (..., K) or (..., K, K)
    ) -> CGMomentResult:
        r"""Exact analytic moment closure of the bilinear CG map at the current mean.

        The CG update delta^(c) = sum_p w_p C_p(x_p (x) y_p) is BILINEAR in the two source slices
        of ``mu``, so its Jacobian is exact and analytic (no autograd): for a path C(x (x) y) the two
        contractions C(. (x) y) and C(x (x) .) are the mean-derivative blocks landing in the
        (target, source_a) and (target, source_b) sub-blocks of J. J starts at the identity, so at
        zero path weights J == I and mu_out == mu. The covariance is pushed forward by the delta
        method sigma_out = sym(J Sigma J^T) + floor Sigma under 'delta_full'; the covariant floor
        preserves input covariance directions even when J is singular. This is a FIRST-ORDER
        Gaussian moment closure, explicitly not an exact distributional pushforward. Covariance is
        passed through untouched under 'passthrough'.

        Precision: the mean delta keeps the ambient autocast (the coupling's amp_dtype consumer), while the
        Jacobian and the covariance pushforward run with autocast off and IEEE fp32 GEMMs at
        promote_types(mu, sigma, float32), as every other covariance pushforward does. Under bf16 autocast the
        rounding of J Sigma J^T (about 2^-8 of its entries) exceeded the floor's margin floor * lambda_min(Sigma)
        and returned indefinite covariances (audit 2026-10-31 A016).
        """
        cg = self._cast_buffers(mu.dtype, mu.device)              # per-(dtype, device), cached
        w = self.path_weights.to(dtype=mu.dtype)                  # one cast, differentiable
        dtype = torch.promote_types(torch.promote_types(mu.dtype, sigma.dtype), torch.float32)
        cg_work = cg if dtype == mu.dtype else self._cast_buffers(dtype, mu.device)
        w_work = w if dtype == mu.dtype else self.path_weights.to(dtype=dtype)
        K = mu.shape[-1]
        delta = torch.zeros_like(mu)
        eye = torch.eye(K, dtype=dtype, device=mu.device)
        jac = eye.expand(*mu.shape[:-1], K, K).clone()            # (..., K, K), starts at identity
        for (t, m, sc, dc, idxs, srcs), path_index in zip(self._groups, self._group_indices(mu.device)):
            _sa0, da0, _sb0, db0 = srcs[0]                        # da/db/dc constant within a group (fixed triple)
            # Mean delta (same grouped arithmetic as forward).
            x = torch.stack([mu[..., sa:sa + da] for (sa, da, _sb, _db) in srcs], dim=0)   # (P, ..., da)
            y = torch.stack([mu[..., sb:sb + db] for (_sa, _da, sb, db) in srcs], dim=0)   # (P, ..., db)
            xy = (x.unsqueeze(-1) * y.unsqueeze(-2)) \
                .reshape(len(srcs), *x.shape[1:-1], -1)           # (P, ..., da*db)
            wsum = torch.einsum("p,p...d->...d", w.index_select(0, path_index), xy)   # (..., da*db)
            delta[..., sc:sc + dc] = delta[..., sc:sc + dc] \
                + torch.einsum("cd,...d->...c", cg[t][m], wsum)
            # Analytic Jacobian: the two bilinear contractions of every path of the group in one stacked einsum
            # each, then accumulated path by path (source copies differ within the group; the target block sc is
            # shared). A self-pair (sa == sb) adds BOTH contractions into the same source columns, the
            # derivative of C(x (x) x).
            with torch.autocast(device_type=mu.device.type, enabled=False), ieee_fp32_matmul():
                Cm = cg_work[t][m].reshape(dc, da0, db0)          # (dc, da, db)
                wp = w_work.index_select(0, path_index).reshape(-1, *([1] * (mu.dim() + 1)))   # (P, 1, ..., 1)
                j_dx = wp * torch.einsum("cij,p...j->p...ci", Cm, y.to(dtype))   # (P, ..., dc, da) = C(. (x) y_p)
                j_dy = wp * torch.einsum("cij,p...i->p...cj", Cm, x.to(dtype))   # (P, ..., dc, db) = C(x_p (x) .)
                for local_p, (sa, da, sb, db) in enumerate(srcs):
                    jac[..., sc:sc + dc, sa:sa + da] = \
                        jac[..., sc:sc + dc, sa:sa + da] + j_dx[local_p]
                    jac[..., sc:sc + dc, sb:sb + db] = \
                        jac[..., sc:sc + dc, sb:sb + db] + j_dy[local_p]
        mu_out = mu + delta
        if self.cg_covariance_mode == "delta_full":
            with torch.autocast(device_type=mu.device.type, enabled=False), ieee_fp32_matmul():
                work = sigma.to(dtype)                                   # (..., K, K)
                js = torch.einsum("...ij,...jk->...ik", jac, work)       # J Sigma
                jsj = torch.einsum("...ik,...lk->...il", js, jac)        # (J Sigma) J^T
                sigma_out = (0.5 * (jsj + jsj.transpose(-1, -2))
                             + self.cg_covariance_floor * work)
        else:
            sigma_out = sigma
        return CGMomentResult(mu_out, sigma_out, jac)
