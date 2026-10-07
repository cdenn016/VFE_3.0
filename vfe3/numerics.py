r"""Numerical conditioning fallbacks + runtime monitors for VFE_3.0.

Two concerns, both modular (registry-backed):
  CONDITIONING FALLBACKS keep the SPD-manifold math finite under ill-conditioning:
    safe_spd_inverse (escalating-jitter Cholesky -> pinv), floor_eigenvalues,
    condition_number.
  RUNTIME MONITORS report numerical health during a run (nan/inf fraction, condition
    number, ...) as plain scalars, via a register_monitor registry so a new probe slots
    in without editing call sites. ``run_monitors`` emits a CSV/JSON-friendly record.

A theoretically pure path is always available (the unregularized op); the fallbacks are
guards that activate only when the pure path fails, and they are documented as such.
"""

import contextlib
import math
import weakref
from collections import OrderedDict
from typing import Callable, Dict, List, Literal, NamedTuple, Optional, Tuple

import torch

from vfe3.runtime import ieee_fp32_matmul


@contextlib.contextmanager
def _autocast_disabled():
    r"""Disable any active autocast for the enclosed region (also usable as a decorator).

    ``ieee_fp32_matmul`` pins the float32 matmul policy but not autocast, which still lowers the
    enclosed GEMMs to bf16/fp16 under ``amp_dtype``. The certified routines below assume IEEE fp32
    products, so they run with autocast off as well (audit 2026-10-07, F32).
    """
    with contextlib.ExitStack() as stack:
        for device_type in ("cpu", "cuda"):
            if torch.is_autocast_enabled(device_type):
                stack.enter_context(torch.autocast(device_type=device_type, enabled=False))
        yield


def _symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    """Average a matrix with its transpose (kills asymmetric round-off)."""
    return 0.5 * (matrix + matrix.transpose(-1, -2))


# Weak, mutation-aware cache for the bounded_variance_from_log overflow check (audit 2026-07-12
# N13; the lie_ops/killing-cache identity+version pattern). The hot-path callers (prior_bank's
# decode/encode table reads) re-read the SAME parameter tables several times per forward, and the
# check's bool((...).any()) is a device->host sync per call -- key the RESULT on
# (id, _version, ...) with a weakref liveness guard so an unchanged table syncs once, an in-place
# optimizer step (version bump) triggers exactly one recheck, and a dead/recycled id can never
# serve a stale verdict.
_MAX_LOG_CHECK_CACHE: 'OrderedDict[tuple, tuple]' = OrderedDict()
_MAX_LOG_CHECK_CACHE_MAXSIZE = 32


def _max_log_exceeded(
    log_sigma: torch.Tensor,
    max_log:   float,
) -> bool:
    """The one host-syncing overflow check (the cached slow path)."""
    return bool((log_sigma.detach() > max_log).any())


def _cached_max_log_exceeded(
    log_sigma: torch.Tensor,
    max_log:   float,
) -> bool:
    """Resolve the overflow check through the weak, mutation-aware identity/version cache."""
    if log_sigma.is_inference():
        # Inference tensors track NO _version counter (reading it raises), so there is no
        # mutation signal to key on -- fall back to the direct uncached check (the pre-cache
        # behavior; one sync per call, exactly as before the N13 cache).
        return _max_log_exceeded(log_sigma, max_log)
    key = (id(log_sigma), log_sigma._version, tuple(log_sigma.shape), log_sigma.dtype,
           log_sigma.device, max_log)
    cached = _MAX_LOG_CHECK_CACHE.get(key)
    if cached is not None:
        tensor_ref, exceeded = cached
        if tensor_ref() is log_sigma:
            _MAX_LOG_CHECK_CACHE.move_to_end(key)
            return exceeded
        del _MAX_LOG_CHECK_CACHE[key]

    exceeded = _max_log_exceeded(log_sigma, max_log)

    def _drop_dead_entry(tensor_ref: weakref.ReferenceType) -> None:
        current = _MAX_LOG_CHECK_CACHE.get(key)
        if current is not None and current[0] is tensor_ref:
            del _MAX_LOG_CHECK_CACHE[key]

    _MAX_LOG_CHECK_CACHE[key] = (weakref.ref(log_sigma, _drop_dead_entry), exceeded)
    _MAX_LOG_CHECK_CACHE.move_to_end(key)
    while len(_MAX_LOG_CHECK_CACHE) > _MAX_LOG_CHECK_CACHE_MAXSIZE:
        _MAX_LOG_CHECK_CACHE.popitem(last=False)
    return exceeded


def bounded_variance_from_log(
    log_sigma: torch.Tensor,

    *,
    eps:     float = 1e-6,
    max_log: float = 80.0,
) -> torch.Tensor:
    r"""Exponentiate a trainable log-variance without overflowing float32.

    Values in the normal ``[log(eps), max_log]`` range retain the ordinary ``exp`` map. Larger
    detached parameter values emit the numerical warning and are capped only for exponentiation;
    ``sigma_max`` is a separate belief-state retraction policy and is deliberately not used here.
    The overflow check is identity/version-cached (audit 2026-07-12 N13): one device->host sync
    per table mutation instead of per call; the warning still fires on every call while the table
    stays above ``max_log`` (from the cached host bool).
    """
    if _cached_max_log_exceeded(log_sigma, max_log):
        import warnings
        warnings.warn(
            f"trainable log-variance exceeds max_log={max_log:g}; clamping before exponentiation",
            RuntimeWarning,
            stacklevel=2,
        )
    return torch.exp(log_sigma.clamp(max=max_log)).clamp(min=eps)


def _overflow_safe_norm(x: torch.Tensor) -> torch.Tensor:   # (..., K) -> (..., 1)
    r"""Euclidean norm over the last axis; where the plain norm overflows, the max-scaled form
    :math:`m\|x/m\|_2` with :math:`m = \max_k|x_k|` (detached; the value is :math:`\|x\|_2` exactly in
    real arithmetic). Finite norms are returned unchanged, so the healthy path is bit-identical."""
    plain = x.norm(dim=-1, keepdim=True)
    scale = x.detach().abs().amax(dim=-1, keepdim=True)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return torch.where(torch.isfinite(plain), plain, scale * (x / scale).norm(dim=-1, keepdim=True))


@_autocast_disabled()
def apply_mu_trust_region(
    delta_mu: torch.Tensor,              # (..., K) proposed mean step (e_q_mu_lr * nat_grad_mu)
    sigma_q:  torch.Tensor,              # (..., K) diagonal variances OR (..., K, K) covariance

    *,
    trust:       float = 5.0,
    is_diagonal: bool  = True,
    eps:         float = 1e-8,
    family:      Optional[str] = None,
) -> torch.Tensor:                       # (..., K) clamped step, same shape/dtype as delta_mu
    r"""Whitened E-step mean trust region: the Mahalanobis ball.

    Bounds the per-iteration mean update in covariance-whitened units so a large VFE mean gradient
    cannot overshoot the belief by more than ``trust`` standard deviations. Let ``L`` be
    ``diag(sqrt(sigma_q))`` for diagonal covariance or the Cholesky factor of a full covariance.
    Then

    .. math::
        w = L^{-1}\delta\mu,\qquad
        \delta\mu' = L\, w \min(\mathrm{trust} / \|w\|_2, 1),

    which is GL-covariant because :math:`\|w\|_2^2 = \delta\mu^\top \Sigma^{-1} \delta\mu`. A failed
    full-covariance Cholesky uses the marginal-variance path for that batch element only.
    """
    # Local import breaks numerics -> families.gaussian -> numerics during a fresh import.
    from vfe3.families.base import get_family
    family_cls = get_family(
        family or ("gaussian_diagonal" if is_diagonal else "gaussian_full")
    )

    # Sanitize BEFORE whitening (audit 2026-07-27). This guard exists to bound an exploding update,
    # so a non-finite delta_mu is precisely its design case -- yet every route below propagated it
    # instead of catching it. Two distinct failures, both reproduced:
    #   full     : torch.linalg.solve_triangular(eye(3), [1, inf, 2]) returns [1, inf, nan]. LAPACK
    #              spreads one inf across UNRELATED coordinates, so the clamp on the next line -- the
    #              guard's entire purpose -- ran after the corruption and returned all-NaN.
    #   diagonal : in 'ball' mode norm2 is a per-ROW reduction, so one inf makes the whole ratio 0;
    #              inf*0 = nan at the exploded entry and 0 elsewhere, silently discarding the step
    #              and ZEROING coordinates that were perfectly finite.
    # NaN -> 0; +-inf -> +-sentinel, where the sentinel is sized so that summing K of its squares
    # cannot itself overflow (finfo.max would: ``max**2`` is already out of range, which made the
    # 'ball' norm inf, the ratio 0, and the whole step vanish). A saturated coordinate then simply
    # pins at the trust bound in 'box' and dominates the projected direction in 'ball', which is the
    # intended response to an overshoot, while its finite neighbors keep their values.
    #
    # UNCONDITIONAL (audit 2026-08-06 B3/F15). The guard used to be behind
    # ``if not bool(torch.isfinite(delta_mu).all())``, which is a device->host sync on every call --
    # and the branch it guards is idempotent on finite input, so the sync bought nothing. ``nan_to_num``
    # replaces only non-finite entries, and its backward is ``grad * isfinite(input)``, so on the
    # healthy path both the value and the gradient are BIT-IDENTICAL to the guarded form.
    finfo = torch.finfo(delta_mu.dtype)
    sentinel = (finfo.max / max(delta_mu.shape[-1], 1)) ** 0.5 / 2.0
    delta_mu = torch.nan_to_num(delta_mu, nan=0.0, posinf=sentinel, neginf=-sentinel)

    if is_diagonal:
        scale = family_cls.trust_region_scale(sigma_q, eps=eps)
        whitened = delta_mu / scale
        # A saturated coordinate over a small variance overflows on whitening; clamp it to the largest
        # finite value and take an overflow-safe norm, so the step pins at the trust radius instead of
        # vanishing (audit 2026-10-07, F35).
        finfo_w = torch.finfo(whitened.dtype)
        whitened = torch.nan_to_num(whitened, nan=0.0, posinf=finfo_w.max, neginf=-finfo_w.max)
        norm2 = _overflow_safe_norm(whitened)
        return delta_mu * (trust / norm2.clamp(min=eps)).clamp(max=1.0)

    # rounds is a POLICY, not a constant (audit 2026-08-06 C6/F29). At rounds=0 there is zero
    # jitter escalation, so one marginally non-PD sigma_q routes that element to the diagonal
    # whitening below -- which is NOT GL-equivariant in either mode. Raising it lets the ladder
    # rescue the element and keep it on the equivariant path instead. Default 0 keeps every run on
    # disk bit-reproducible; the counter below makes the fallback visible either way.
    factor, ok = safe_cholesky(sigma_q, eps=eps, rounds=_MU_TRUST_CHOLESKY_ROUNDS)
    # rounds=0 means ZERO jitter escalation here, so one marginally non-PD sigma_q routes that batch
    # element to the diagonal-whitening fallback below -- which is NOT GL-equivariant in either mode
    # (measured relative equivariance error 9.3e-2 / 6.6e-1 / 1.1e-1 at gauge scales 0.06/0.5/1.5,
    # against the ball path's 1.1e-15 / 1.1e-13 / 9.8e-11). It is masked in by torch.where with no
    # signal, so a run could not tell it happened (audit 2026-08-06 F29). Counted on-device, no sync.
    _count_mu_trust_fallback(ok)
    eye = torch.eye(sigma_q.shape[-1], device=sigma_q.device, dtype=sigma_q.dtype)
    safe_factor = torch.where(ok.unsqueeze(-1).unsqueeze(-1), factor, eye.expand_as(factor))
    whitened = torch.linalg.solve_triangular(
        safe_factor,
        delta_mu.unsqueeze(-1),
        upper=False,
    ).squeeze(-1)
    # A near-singular safe_factor can still emit a non-finite whitened coordinate from finite input.
    # Unconditional for the same reason as the delta_mu guard above (audit 2026-08-06 B3/F15).
    whitened_finfo = torch.finfo(whitened.dtype)
    whitened = torch.nan_to_num(
        whitened, nan=0.0, posinf=whitened_finfo.max, neginf=-whitened_finfo.max)
    norm2 = _overflow_safe_norm(whitened)                   # finite even when whitened holds finfo.max
    bounded = whitened * (trust / norm2.clamp(min=eps)).clamp(max=1.0)
    full_out = (safe_factor @ bounded.unsqueeze(-1)).squeeze(-1)
    # The fallback is computed UNCONDITIONALLY (audit 2026-08-06 B3/F15). It used to sit behind an
    # ``if bool(ok.all()): return full_out`` early exit, which is a device->host sync on every call
    # to save a handful of elementwise ops on a (..., K) tensor -- far less than the triangular solve
    # and (K, K) matmul already spent above. The trailing ``torch.where`` returns exactly ``full_out``
    # wherever ``ok``, so the value is BIT-IDENTICAL; so is the gradient, because ``torch.where``
    # routes grad by SELECTION (the unselected branch receives an exact 0, not 0 * something).
    sigma_diag = family_cls.covariance_diagonal(sigma_q, eps=eps)
    scale = sigma_diag.clamp(min=eps).sqrt()
    fallback_norm = _overflow_safe_norm(torch.nan_to_num(
        delta_mu / scale, nan=0.0, posinf=whitened_finfo.max, neginf=-whitened_finfo.max))
    fallback = delta_mu * (trust / fallback_norm.clamp(min=eps)).clamp(max=1.0)
    return torch.where(ok.unsqueeze(-1), full_out, fallback)


_MU_TRUST_CHOLESKY_ROUNDS: int = 0


def set_mu_trust_cholesky_rounds(rounds: int) -> int:
    r"""Set the process-wide jitter-escalation rounds for the mu-trust-region whitening Cholesky.

    Returns the previous value. Default 0 reproduces the historical behavior exactly; a positive
    value trades a small ridge for staying on the GL-EQUIVARIANT whitening path instead of dropping
    to the non-equivariant diagonal fallback (measured relative equivariance error 9.3e-2 / 6.6e-1 /
    1.1e-1 at gauge scales 0.06/0.5/1.5, against the ball path's 1.1e-15 / 1.1e-13 / 9.8e-11)."""
    global _MU_TRUST_CHOLESKY_ROUNDS
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 0:
        raise ValueError(f"mu_trust_cholesky_rounds must be a nonnegative int, got {rounds!r}")
    previous = _MU_TRUST_CHOLESKY_ROUNDS
    _MU_TRUST_CHOLESKY_ROUNDS = rounds
    return previous


def mu_trust_cholesky_rounds() -> int:
    r"""Return the active mu-trust-region Cholesky jitter rounds."""
    return _MU_TRUST_CHOLESKY_ROUNDS


_MU_TRUST_FALLBACK_COUNTS: Dict[str, torch.Tensor] = {}


def _count_mu_trust_fallback(ok: torch.Tensor) -> None:
    r"""Accumulate mu-trust-region Cholesky failures per device (async add, no host sync)."""
    key = str(ok.device)
    counter = _MU_TRUST_FALLBACK_COUNTS.get(key)
    if counter is None:
        counter = torch.zeros((), dtype=torch.int64, device=ok.device)
        _MU_TRUST_FALLBACK_COUNTS[key] = counter
    counter += (~ok).sum()


_DECODE_LOGDET_FALLBACK_COUNTS: Dict[str, torch.Tensor] = {}


def _count_decode_logdet_fallback(ok: torch.Tensor) -> None:
    r"""Accumulate decode log-det Cholesky failures per device (async add, no host sync)."""
    key = str(ok.device)
    counter = _DECODE_LOGDET_FALLBACK_COUNTS.get(key)
    if counter is None:
        counter = torch.zeros((), dtype=torch.int64, device=ok.device)
        _DECODE_LOGDET_FALLBACK_COUNTS[key] = counter
    counter += (~ok).sum()


def decode_logdet_fallback_elements() -> int:
    r"""Decode positions whose Sigma_q failed every jitter round since the last reset. Host-syncs."""
    return int(sum(int(c) for c in _DECODE_LOGDET_FALLBACK_COUNTS.values()))


def reset_decode_logdet_fallback_elements() -> None:
    r"""Zero the decode log-det fallback counters (per-run accounting)."""
    for c in _DECODE_LOGDET_FALLBACK_COUNTS.values():
        c.zero_()


def mu_trust_fallback_elements() -> int:
    r"""Belief elements whitened by the NON-equivariant diagonal fallback since the last reset.
    Reads the on-device counters, so it host-syncs: call it at an existing logging cadence."""
    return int(sum(int(count) for count in _MU_TRUST_FALLBACK_COUNTS.values()))


def reset_mu_trust_fallback_elements() -> None:
    r"""Zero the mu-trust-region fallback counters (per-run accounting)."""
    for count in _MU_TRUST_FALLBACK_COUNTS.values():
        count.zero_()


def safe_cholesky(
    matrix: torch.Tensor,                # (..., K, K) symmetric ~PD (per-element factored)

    *,
    eps:    float = 1e-6,
    rounds: int   = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:  # (factor (..., K, K), ok mask (...))
    r"""Per-element Cholesky that never raises, with optional per-element jitter escalation.

    Uses ``torch.linalg.cholesky_ex`` (returns a per-batch-element ``info``, does NOT raise)
    so that a single non-PD element cannot kill the whole batched call. Round 0 adds ZERO
    extra jitter, so on already-SPD inputs the returned factor is byte-identical to
    ``torch.linalg.cholesky`` (the pure path). Elements that fail (``info != 0``) are retried
    with an escalating ridge ``eps * 10^t`` for t = 0..rounds-1, applied ONLY to the failed
    elements so good elements keep their round-0 factor unperturbed.

    Returns the factor ``L`` together with a boolean ``ok`` mask (True where a PD factor was
    obtained). Callers MUST drive masking off ``ok`` (not finiteness): on failure ``cholesky_ex``
    returns a finite *partial* factor, not NaN, so a downstream ``logdet`` would otherwise be a
    finite-but-wrong value rather than NaN. The mask lets the caller inject NaN for failed
    elements so a ``safe_kl_clamp`` maps them to ``kl_max``.

    JITTER SCALE (audit 2026-08-06 C3/F18; relative hardcoded 2026-10-02). The ridge is RELATIVE:
    round ``t`` adds :math:`\epsilon 10^t s(M) I` with :math:`s(M) = \max(1, |\operatorname{tr}(M)/K|)`,
    so every element receives the same relative perturbation and the ladder is covariant under
    :math:`M \to c M` for diagonal means above one. An absolute ridge (the retired alternative)
    doubles an eigenvalue sitting on the ``eps=1e-6`` SPD floor while being a no-op at
    ``sigma_max=100``. The ladder only fires above cond ~1e8.
    """
    M = _symmetrize(matrix)
    L, info = torch.linalg.cholesky_ex(M)
    ok = (info == 0) & torch.isfinite(L).all(dim=(-2, -1))
    if bool(ok.all()):
        return L, ok
    K = M.shape[-1]
    eye = torch.eye(K, device=M.device, dtype=M.dtype)
    scale = M.diagonal(dim1=-2, dim2=-1).mean(-1).abs().clamp_min(1.)
    # Select a ridge without retaining any failed factorization in autograd.
    ridge = torch.zeros_like(scale)
    with torch.no_grad():
        for t in range(rounds):
            candidate = eps * (10. ** t)
            L_t, info_t = torch.linalg.cholesky_ex(
                M + (candidate * scale)[..., None, None] * eye)
            newly = (~ok) & (info_t == 0) & torch.isfinite(L_t).all(dim=(-2, -1))
            ridge = torch.where(newly, candidate, ridge)
            ok = ok | newly
            if bool(ok.all()):
                break
    repaired = M + (ridge * scale)[..., None, None] * eye
    safe = torch.where(ok[..., None, None], repaired, eye)
    # Failed rows expose an identity placeholder; callers must still honor ok.
    L, _ = torch.linalg.cholesky_ex(safe)
    return L, ok


def safe_spd_inverse(
    matrix:    torch.Tensor,             # (..., K, K) symmetric ~PD

    *,
    eps:       float = 1e-6,
    max_tries: int   = 5,
) -> torch.Tensor:                       # (..., K, K) inverse
    r"""SPD inverse via Cholesky with escalating jitter, falling back to the pseudo-inverse.

    Per element (via ``cholesky_ex``, which never raises): tries ``cholesky_inverse`` on
    ``M + (eps * 10^t) I`` for t = 0..max_tries-1, escalating the ridge ONLY on the elements that
    still fail; an element where every jitter level fails falls back to ``pinv``. The per-element
    retry mirrors ``safe_cholesky`` so one non-PD batch element cannot poison the exact inverse of
    its well-conditioned siblings. The pure path is ``t=0`` with the documented default ridge.
    """
    # float64 stays float64 (audit 2026-07-12 N4/N12 dtype policy); half promotes to fp32.
    compute_dtype = torch.float64 if matrix.dtype == torch.float64 else torch.float32
    M = _symmetrize(matrix.to(compute_dtype))
    K = M.shape[-1]
    eye = torch.eye(K, device=M.device, dtype=M.dtype)
    L, ok = safe_cholesky(M + eps * eye)                    # round 0: documented eps ridge
    if bool(ok.all()):
        return torch.cholesky_inverse(L).to(matrix.dtype)
    out = torch.cholesky_inverse(L)                          # ok elements keep their good inverse
    for t in range(1, max_tries):                            # retry ONLY the still-failed elements
        if bool(ok.all()):
            break
        L_t, ok_t = safe_cholesky(M + (eps * (10.0 ** t)) * eye)
        newly = (~ok) & ok_t
        if bool(newly.any()):
            inv_t = torch.cholesky_inverse(L_t)
            out = torch.where(newly.unsqueeze(-1).unsqueeze(-1), inv_t, out)
            ok = ok | ok_t
    if not bool(ok.all()):                                   # pinv ONLY the still-failed elements
        out = torch.where(ok.unsqueeze(-1).unsqueeze(-1), out, torch.linalg.pinv(M))
    return out.to(matrix.dtype)


# cuSOLVER's batched symmetric eigensolver -- the routine torch dispatches to for n <= 32 -- rejects
# the CALL past a fixed batch count, with CUSOLVER_STATUS_INVALID_VALUE out of
# cusolverDnXsyevBatched_bufferSize. Measured last-OK / first-FAIL, IDENTICAL in float32 and float64
# at each K (which is what rules out a workspace-byte or int32 story): K=2 32016/32017,
# K=8 29915/29916, K=20 26305/26306, K=32 23325/23326. At K >= 33 torch takes the non-batched path
# and there is no ceiling at all. PyTorch 2.8.0 regression syevjBatched_bufferSize ->
# xsyevBatched_bufferSize (PR #155695), tracked as pytorch/pytorch#166004.
#
# 16384 is a power of two comfortably under the smallest measured ceiling (23325 at K=32).
_EIG_MAX_BATCH: int = 16_384


def _eig_needs_chunking(matrix: torch.Tensor, max_batch: int) -> bool:
    r"""True when this call would cross the cuSOLVER batched-eigensolver ceiling."""
    if matrix.dim() < 3 or not matrix.is_cuda:
        return False                                  # unbatched, or CPU/MAGMA: no ceiling
    if matrix.shape[-1] > 32:
        return False                                  # n > 32 takes the non-batched path
    return matrix.numel() // (matrix.shape[-1] * matrix.shape[-2]) > max_batch


def safe_eigvalsh(
    matrix:    torch.Tensor,             # (..., K, K) symmetric
    *,
    max_batch: int = _EIG_MAX_BATCH,
) -> torch.Tensor:                       # (..., K) ascending eigenvalues
    r"""``torch.linalg.eigvalsh`` that cannot trip the cuSOLVER batch ceiling (see ``_EIG_MAX_BATCH``).

    Slices the flattened batch and concatenates. Value-identical -- the decomposition is independent
    per matrix, so slicing changes nothing but the launch geometry -- and autograd flows through the
    slice/cat unchanged. Below the ceiling this is exactly ``torch.linalg.eigvalsh``, with no extra
    allocation and no host sync, so the healthy path is untouched.
    """
    if not _eig_needs_chunking(matrix, max_batch):
        return torch.linalg.eigvalsh(matrix)
    K = matrix.shape[-1]
    flat = matrix.reshape(-1, K, K)
    parts = [torch.linalg.eigvalsh(flat[i:i + max_batch])
             for i in range(0, flat.shape[0], max_batch)]
    return torch.cat(parts, dim=0).reshape(*matrix.shape[:-1])


def safe_eigh(
    matrix:    torch.Tensor,             # (..., K, K) symmetric
    *,
    max_batch: int = _EIG_MAX_BATCH,
) -> Tuple[torch.Tensor, torch.Tensor]:  # (..., K) eigenvalues, (..., K, K) eigenvectors
    r"""``torch.linalg.eigh`` twin of ``safe_eigvalsh``; same ceiling, same routine, same fix."""
    if not _eig_needs_chunking(matrix, max_batch):
        return torch.linalg.eigh(matrix)
    K = matrix.shape[-1]
    flat = matrix.reshape(-1, K, K)
    parts = [torch.linalg.eigh(flat[i:i + max_batch])
             for i in range(0, flat.shape[0], max_batch)]
    evals = torch.cat([p.eigenvalues for p in parts], dim=0).reshape(*matrix.shape[:-1])
    evecs = torch.cat([p.eigenvectors for p in parts], dim=0).reshape(matrix.shape)
    return evals, evecs


class ValidatedCholeskySolve(NamedTuple):
    r"""Zero-jitter Cholesky result plus explicit numerical-validity evidence."""

    factor: torch.Tensor
    solution: Optional[torch.Tensor]
    certified: torch.Tensor
    condition: torch.Tensor
    symmetry_residual: torch.Tensor
    factor_residual: torch.Tensor
    solve_residual: torch.Tensor


@torch.no_grad()
def _recheck_spd_condition_on_cpu(
    symmetric:       torch.Tensor,       # (..., K, K), the matrix used by Cholesky
    condition:       torch.Tensor,       # (...) original detached spectral estimate
    eligible:        torch.Tensor,       # (...) all nonspectral checks already passed
    condition_limit: float,
) -> torch.Tensor:
    r"""Independently recheck rejected spectral estimates, without altering the matrix.

    Recompute kappa_2(A) = lambda_max(A) / lambda_min(A) on CPU in float64 for
    eligible failures only. A nonpositive spectrum or failed CPU eigensolve still
    rejects. The caller retains its original dtype-dependent acceptance bound.
    """
    suspect = eligible & (~torch.isfinite(condition) | (condition > condition_limit))
    if not bool(suspect.any()):
        return condition
    flat_mask = suspect.reshape(-1)
    selected = symmetric.detach().reshape(-1, symmetric.shape[-1], symmetric.shape[-1])[flat_mask]
    try:
        spectrum = torch.linalg.eigvalsh(selected.to(device="cpu", dtype=torch.float64))
    except torch.linalg.LinAlgError:
        return condition
    positive = (spectrum[..., 0] > 0.) & torch.isfinite(spectrum).all(dim=-1)
    checked = torch.where(
        positive, spectrum[..., -1] / spectrum[..., 0],
        spectrum.new_full((), float("inf")))
    result = condition.detach().reshape(-1).clone()
    result[flat_mask] = checked.to(device=condition.device, dtype=condition.dtype)
    return result.reshape(condition.shape)


@_autocast_disabled()
@ieee_fp32_matmul()
def validated_cholesky_solve(
    matrix: torch.Tensor,                 # (..., K, K) candidate SPD system
    rhs: Optional[torch.Tensor] = None,   # (..., K, R), when a checked solve is required

    *,
    residual_tol: Optional[float] = None,
    condition_limit: Optional[float] = None,
) -> ValidatedCholeskySolve:
    r"""Factor and optionally solve an SPD system, certifying the unjittered result per row.

    The shared defaults are dtype- and dimension-aware: residuals must be at most ``64 K eps`` and
    the spectral condition number at most ``1 / (64 K eps)`` for matrix dimension ``K``. The
    dimension factor accounts for accumulated inner-product error. Matrix symmetry and Cholesky
    reconstruction use max-entry residuals normalized by ``max(abs(A))``. The solve uses the
    normwise backward error ``max(abs(A x - b)) / (||A||_inf max(abs(x)) + max(abs(b)))``. No ridge,
    symmetrizing projection, or eigenvalue floor changes the returned semantics; symmetrization is
    used only after the original asymmetry has been measured and bounded. A CUDA float64 spectral
    rejection with otherwise valid Cholesky/solve evidence is independently rechecked on CPU;
    the original bounds and differentiable factor/solution remain unchanged.
    """
    if matrix.dim() < 2 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError(
            "validated_cholesky_solve requires square trailing matrix axes, got "
            f"shape {tuple(matrix.shape)}")
    if matrix.dtype not in (torch.float32, torch.float64):
        raise ValueError(
            "validated_cholesky_solve requires float32 or float64 input, got "
            f"{matrix.dtype}")
    if rhs is not None and (
        rhs.dim() < 2
        or rhs.shape[:-2] != matrix.shape[:-2]
        or rhs.shape[-2] != matrix.shape[-1]
    ):
        raise ValueError(
            "validated_cholesky_solve rhs must have shape (..., K, R) matching matrix "
            f"shape {tuple(matrix.shape)}, got {tuple(rhs.shape)}")

    eps = torch.finfo(matrix.dtype).eps
    dimension_factor = 64.0 * float(matrix.shape[-1])
    residual_tol = dimension_factor * eps if residual_tol is None else float(residual_tol)
    condition_limit = 1.0 / (dimension_factor * eps) \
        if condition_limit is None else float(condition_limit)
    if not math.isfinite(residual_tol) or not math.isfinite(condition_limit):
        raise ValueError(
            "validated_cholesky_solve requires finite residual_tol and condition_limit, "
            f"got residual_tol={residual_tol!r}, condition_limit={condition_limit!r}")
    if residual_tol < 0.0 or condition_limit < 1.0:
        raise ValueError(
            "validated_cholesky_solve requires residual_tol >= 0 and condition_limit >= 1, "
            f"got residual_tol={residual_tol!r}, condition_limit={condition_limit!r}")

    symmetric = _symmetrize(matrix)
    factor, info = torch.linalg.cholesky_ex(symmetric)
    factor_ok = info == 0
    solution: Optional[torch.Tensor] = None
    if rhs is not None:
        eye = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
        usable_factor = torch.where(
            factor_ok.unsqueeze(-1).unsqueeze(-1), factor, eye)
        solution = torch.cholesky_solve(rhs, usable_factor)

    with torch.no_grad():
        detached = matrix.detach()
        detached_symmetric = symmetric.detach()
        matrix_finite = torch.isfinite(detached).all(dim=(-2, -1))
        eye = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
        spectral_input = torch.where(
            matrix_finite.unsqueeze(-1).unsqueeze(-1), detached_symmetric, eye)
        eigenvalues = safe_eigvalsh(spectral_input)
        lam_min = eigenvalues[..., 0]
        condition = eigenvalues[..., -1] / lam_min.clamp_min(torch.finfo(matrix.dtype).tiny)
        condition = torch.where(
            lam_min > 0.0, condition, condition.new_full((), float("inf")))

        matrix_scale = detached.abs().amax(dim=(-2, -1)).clamp_min(
            torch.finfo(matrix.dtype).tiny)
        symmetry_residual = (
            detached - detached.transpose(-1, -2)
        ).abs().amax(dim=(-2, -1)) / matrix_scale
        factor_residual = (
            factor.detach() @ factor.detach().transpose(-1, -2) - detached_symmetric
        ).abs().amax(dim=(-2, -1)) / matrix_scale

        solve_residual = torch.zeros_like(condition)
        solution_finite = torch.ones_like(factor_ok)
        if rhs is not None and solution is not None:
            detached_rhs = rhs.detach()
            detached_solution = solution.detach()
            solution_finite = torch.isfinite(detached_solution).all(dim=(-2, -1))
            numerator = (
                detached_symmetric @ detached_solution - detached_rhs
            ).abs().amax(dim=(-2, -1))
            denominator = (
                detached_symmetric.abs().sum(dim=-1).amax(dim=-1)
                * detached_solution.abs().amax(dim=(-2, -1))
                + detached_rhs.abs().amax(dim=(-2, -1))
            ).clamp_min(torch.finfo(matrix.dtype).tiny)
            solve_residual = numerator / denominator

        nonspectral_certified = (
            matrix_finite
            & factor_ok
            & torch.isfinite(factor.detach()).all(dim=(-2, -1))
            & solution_finite
            & torch.isfinite(symmetry_residual)
            & (symmetry_residual <= residual_tol)
            & torch.isfinite(factor_residual)
            & (factor_residual <= residual_tol)
            & torch.isfinite(solve_residual)
            & (solve_residual <= residual_tol)
        )
        if matrix.is_cuda and matrix.dtype == torch.float64:
            # The step-5310 CUDA spectrum rejected a covariance whose retained
            # Cholesky and independent CPU spectrum certify condition 351.
            condition = _recheck_spd_condition_on_cpu(
                detached_symmetric, condition, nonspectral_certified, condition_limit)
        certified = nonspectral_certified & torch.isfinite(condition) & (condition <= condition_limit)

    return ValidatedCholeskySolve(
        factor, solution, certified, condition, symmetry_residual,
        factor_residual, solve_residual)


# Certificate tolerance for square-root (CholeskyQR2) factors, eps * kappa_1-estimate(C) <= this.
# Shared by the projected scorer and the factored common-frame KL (optimization_plan.md WS2/WS3).
# Set just above the observed maximum eps * kappa_1 = 1.3e-4 on the saved K=64/K=128 runs; see
# relational_scoring._factor_status for the error model and why the P2 max-norm reading is not used.
FACTOR_CONDITION_TOL: float = 3e-4


def guarded_cholesky(
    W: torch.Tensor,   # (..., n, n) symmetric
) -> Tuple[torch.Tensor, torch.Tensor]:   # (..., n, n) lower factor, (...) failed
    r"""Zero-jitter Cholesky whose failed entries factor the identity instead.

    Autograd through a failed Cholesky is NaN even under a zero cotangent, and a batched NaN
    reaches every shared parameter. A detached pass finds the failures, and the differentiable pass
    factors ``I`` in their place, so ``failed`` entries return the identity with finite (zero)
    gradients. Healthy entries repeat the same factorization of the same input, bit for bit.
    """
    with torch.no_grad():
        failed = torch.linalg.cholesky_ex(W.detach())[1] != 0
    eye = torch.eye(W.shape[-1], dtype=W.dtype, device=W.device)
    return torch.linalg.cholesky_ex(torch.where(failed[..., None, None], eye, W))[0], failed


@_autocast_disabled()
@ieee_fp32_matmul()
def cholesky_qr2(
    X: torch.Tensor,   # (..., r, m) source root with r <= m

    *,
    shift: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:   # (..., r, r) lower factor, (...) failed
    r"""Lower Cholesky factor C of G = X X^T by CholeskyQR2, error O(eps kappa(X)).

    .. math::
        C_1 C_1^T = \mathrm{fl}(X X^T), \qquad Y = C_1^{-1} X, \qquad
        C_2 C_2^T = \mathrm{fl}(Y Y^T), \qquad C = C_1 C_2 .

    In exact arithmetic :math:`C C^T = C_1 Y Y^T C_1^T = X X^T`, so C is the unique Cholesky factor
    of G and its log-determinant and inverse are the same functions of X. The second pass corrects
    the first pass's squared-conditioning error: CholeskyQR2 is backward stable in X for
    :math:`\kappa(X) \lesssim \varepsilon^{-1/2}` (Fukaya et al. 2014; Yamamoto et al. 2015), so C
    carries relative error of order :math:`\varepsilon \kappa(X)`, not
    :math:`\varepsilon \kappa(X)^2`. ``shift=True`` is shifted CholeskyQR3 (Fukaya et al. 2020):
    the first pass factors :math:`\mathrm{fl}(X X^T) + s I` with
    :math:`s = 11 (m r + r (r + 1)) \varepsilon \lVert X \rVert_F^2`, and CholeskyQR2 then runs on
    :math:`C_1^{-1} X`. The shift is a preconditioner only: the product still factors
    :math:`X X^T` exactly, and validity extends to :math:`\kappa(X) \approx \varepsilon^{-1}`.
    No jitter enters the result.

    ``failed`` marks entries whose factorizations failed or went non-finite. Their factor is
    meaningless but finite with finite gradients (failed passes factor the identity, see
    :func:`guarded_cholesky`), so a caller may overwrite them and still differentiate the rest.
    """
    r, m = X.shape[-2:]
    W1 = X @ X.transpose(-1, -2)
    W1 = .5 * W1 + .5 * W1.transpose(-1, -2)
    if shift:
        scale = 11 * (m * r + r * (r + 1)) * torch.finfo(X.dtype).eps
        W1 = W1 + (scale * X.square().sum(dim=(-2, -1)))[..., None, None] * torch.eye(
            r, dtype=X.dtype, device=X.device)
    C1, failed1 = guarded_cholesky(W1)
    Y = torch.linalg.solve_triangular(C1, X, upper=False)
    if shift:
        C2, failed2 = cholesky_qr2(Y)
    else:
        W2 = Y @ Y.transpose(-1, -2)
        C2, failed2 = guarded_cholesky(.5 * W2 + .5 * W2.transpose(-1, -2))
    C = C1 @ C2
    with torch.no_grad():
        failed = failed1 | failed2 | ~torch.isfinite(C).all(dim=(-2, -1))
    return C, failed


def triangular_condition_estimate(
    C: torch.Tensor,   # (..., n, n) lower triangular, nonsingular

    *,
    iterations: int = 3,
) -> torch.Tensor:     # (...) estimate of kappa_1(C)
    r"""Hager-Higham estimate of :math:`\kappa_1(C) = \lVert C \rVert_1 \lVert C^{-1} \rVert_1`.

    :math:`\lVert C^{-1} \rVert_1` is estimated by Hager's power iteration on the unit 1-norm
    ball (Hager 1984) with Higham's extra alternating-sign test vector (Higham 1988; the algorithm
    behind LAPACK xLACN2/xTRCON), using only triangular solves: O(n^2) per matrix. Every candidate
    is :math:`\lVert C^{-1} v \rVert_1 / \lVert v \rVert_1` for an explicit v, so the result is a
    LOWER bound on :math:`\kappa_1(C)` whatever the iteration count; it is typically within a
    factor of about 3 of the true value. The iteration count is fixed, so batched calls run with no
    data-dependent control flow and no host synchronization.
    """
    n = C.shape[-1]
    with torch.no_grad():
        C = C.detach()
        norm = C.abs().sum(dim=-2).amax(dim=-1)                                # ||C||_1
        if n == 1:
            return torch.ones_like(norm)
        x = torch.full((*C.shape[:-1], 1), 1. / n, dtype=C.dtype, device=C.device)
        estimate = torch.zeros_like(norm)
        for _ in range(iterations):
            y = torch.linalg.solve_triangular(C, x, upper=False)                # C^{-1} x
            estimate = torch.maximum(estimate, y.abs().sum(dim=(-2, -1)))
            signs = torch.where(y >= 0, torch.ones_like(y), -torch.ones_like(y))
            z = torch.linalg.solve_triangular(C.transpose(-1, -2), signs, upper=True)  # C^{-T} sign(y)
            x = torch.zeros_like(x).scatter_(-2, z.abs().argmax(dim=-2, keepdim=True), 1.)
        index = torch.arange(n, dtype=C.dtype, device=C.device)
        alternating = ((-1.) ** index * (1. + index / (n - 1))).expand(*C.shape[:-1]).unsqueeze(-1)
        y = torch.linalg.solve_triangular(C, alternating, upper=False)
        estimate = torch.maximum(estimate, 2. * y.abs().sum(dim=(-2, -1)) / (3. * n))
        return norm * estimate


def floor_eigenvalues(
    matrix: torch.Tensor,                # (..., K, K) symmetric
    *,
    floor:  float = 1e-6,
) -> torch.Tensor:                       # (..., K, K) SPD with eigenvalues >= floor
    r"""Project a symmetric matrix to SPD by clamping its eigenvalues up to ``floor``.

    float64 stays float64 (the module dtype policy); every other dtype computes in float32.
    """
    compute_dtype = torch.float64 if matrix.dtype == torch.float64 else torch.float32
    M = _symmetrize(matrix.to(compute_dtype))
    evals, evecs = safe_eigh(M)
    evals = evals.clamp(min=floor)
    out = (evecs * evals.unsqueeze(-2)) @ evecs.transpose(-1, -2)
    return _symmetrize(out).to(matrix.dtype)


def condition_number(
    matrix: torch.Tensor,                # (..., K, K) symmetric PD OR (..., K) diagonal variances

    *,
    eps:    float                               = 1e-12,
    kind:   Literal["auto", "full", "diagonal"] = "auto",
) -> torch.Tensor:                       # (...) lambda_max / lambda_min
    r"""Spectral condition number lambda_max / lambda_min (clamped at ``eps``).

    ``kind='diagonal'`` always treats the last axis as a variance spectrum, including a square
    ``(N, K)`` table with ``N == K``. ``kind='full'`` requires square trailing dimensions and uses
    ``eigvalsh``. ``kind='auto'`` preserves the legacy shape inference: square trailing dimensions
    select the full-matrix path and every other non-scalar shape selects the diagonal path.
    """
    if kind not in ("auto", "full", "diagonal"):
        raise ValueError(
            f"condition_number kind must be 'auto', 'full', or 'diagonal', got {kind!r}")

    square = matrix.dim() >= 2 and matrix.shape[-1] == matrix.shape[-2]
    if kind == "full" and not square:
        raise ValueError(
            "condition_number kind='full' requires square trailing dimensions (..., K, K), "
            f"got shape {tuple(matrix.shape)}")
    full = kind == "full" or (kind == "auto" and square)
    # float64 stays float64 (the module dtype policy); every other dtype computes in float32.
    compute_dtype = torch.float64 if matrix.dtype == torch.float64 else torch.float32
    if not full:
        if matrix.dim() == 0:
            raise ValueError("condition_number diagonal input must have at least one dimension")
        spec = matrix.to(compute_dtype)
        lam_min = spec.min(dim=-1).values
        cond = spec.max(dim=-1).values / lam_min.clamp(min=eps)
        # non-positive variance -> no condition number; surface +inf, mirroring the full-matrix branch
        # (audit 2026-06-17 round 2 id1), not a large positive value from clamping a zero/negative up to eps.
        return torch.where(lam_min > 0, cond, cond.new_tensor(float("inf"))).to(matrix.dtype)
    evals = safe_eigvalsh(_symmetrize(matrix.to(compute_dtype)))
    lam_min = evals[..., 0]
    cond = evals[..., -1] / lam_min.clamp(min=eps)
    # A non-PD matrix (lambda_min <= 0) has no condition number; surface +inf rather than the large
    # positive value clamping a negative lambda_min up to eps would give (which reads as a merely
    # ill-conditioned SPD matrix and hides the loss of positive-definiteness). (audit 2026-06-17)
    cond = torch.where(lam_min > 0, cond, cond.new_tensor(float("inf")))
    return cond.to(matrix.dtype)


def nan_inf_fraction(
    tensor: torch.Tensor,
) -> float:                              # fraction of non-finite entries in [0, 1]
    r"""Fraction of NaN/Inf entries (0.0 = all finite)."""
    if tensor.numel() == 0:
        return 0.0
    return float((~torch.isfinite(tensor)).float().mean())


def check_finite(
    tensor: torch.Tensor,
    name:   str = "tensor",

    *,
    raise_on_nonfinite: bool = False,
) -> bool:                               # True if all-finite
    r"""Report (and optionally raise on) non-finite entries; returns finiteness."""
    frac = nan_inf_fraction(tensor)
    if frac > 0.0:
        msg = f"{name}: {frac:.3%} non-finite entries"
        if raise_on_nonfinite:
            raise FloatingPointError(msg)
        import warnings
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return False
    return True


# ---------------------------------------------------------------------------
# Monitor registry: name -> (tensor -> scalar). New probes slot in by name.
# ---------------------------------------------------------------------------
_MONITORS: Dict[str, Callable[[torch.Tensor], float]] = {}


def register_monitor(name: str, *, override: bool = False) -> Callable:
    """Decorator registering a scalar numerical monitor under ``name``.

    Duplicate keys fail closed (audit 2026-07-01 round-3): a second registration under an
    existing name silently shadowed the first. Pass ``override=True`` to replace deliberately.
    """
    def _wrap(fn: Callable[[torch.Tensor], float]) -> Callable[[torch.Tensor], float]:
        if name in _MONITORS and not override:
            raise KeyError(f"monitor {name!r} already registered; pass override=True to replace")
        _MONITORS[name] = fn
        return fn
    return _wrap


def get_monitor(name: str) -> Callable[[torch.Tensor], float]:
    """Return the registered monitor (KeyError if absent)."""
    if name not in _MONITORS:
        raise KeyError(f"no monitor {name!r}; available: {sorted(_MONITORS)}")
    return _MONITORS[name]


@register_monitor("nan_fraction")
def _mon_nan(tensor: torch.Tensor) -> float:
    """Fraction of non-finite entries."""
    return nan_inf_fraction(tensor)


@register_monitor("abs_max")
def _mon_absmax(tensor: torch.Tensor) -> float:
    """Largest absolute (finite) entry magnitude."""
    finite = tensor[torch.isfinite(tensor)]
    return float(finite.abs().max()) if finite.numel() else float("nan")


@register_monitor("condition_number")
def _mon_cond(matrix: torch.Tensor) -> float:
    """Spectral condition number (max over any leading batch)."""
    return float(condition_number(matrix).max())


def run_monitors(
    tensor:   torch.Tensor,
    monitors: Optional[List[str]] = None,
) -> Dict[str, float]:
    r"""Apply the named monitors to ``tensor``; returns a CSV/JSON-friendly record.

    ``monitors=None`` runs the family-agnostic probes (nan_fraction, abs_max); pass an
    explicit list to include matrix probes (e.g. condition_number) on SPD inputs.
    """
    names = ["nan_fraction", "abs_max"] if monitors is None else monitors
    return {n: get_monitor(n)(tensor) for n in names}
