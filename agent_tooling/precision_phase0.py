r"""Phase 0 precision measurements for ``optimization_plan.md`` (CUDA, saved checkpoints).

Captures the live projected-predictive scorer inputs of a saved run on real validation batches and
measures, per layer call:

1. certificate trigger rates (float32 Cholesky failure, the Gram certificate
   eps ||G||_F ||G^{-1}||_F > 1e-4, the frame-pair and congruence triggers);
2. the per-token conditioning of X = S U^{-1} Sigma^{1/2}, U, Sigma and G, and how much of it is
   common-mode (removed by the per-(b, h) whitening A = Gbar_k^{-1/2});
3. the error of candidate KL routes against a float64 recomputation from the float64 sources, in
   units of the attention temperature tau, on the causal support after the kl_max clamp;
4. the pair-stage cancellation ratio sum |P_ab G_ab| / tr(P G).

``--profile`` additionally profiles forward+backward steps with each escalation toggle.
Run with the CUDA interpreter: ``C:/anaconda/python.exe agent_tooling/precision_phase0.py RUN_DIR``.
"""

import argparse
import dataclasses
import json
import math
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from vfe3.quiet import silence_toolchain_warnings  # noqa: E402
silence_toolchain_warnings()

import vfe3.gradients.relational as relational_mod  # noqa: E402
import vfe3.inference.e_step as e_step_mod  # noqa: E402
from vfe3.config import VFE3Config, config_from_serialized  # noqa: E402
from vfe3.data.datasets import make_dataloader  # noqa: E402
from vfe3.geometry.transport import (  # noqa: E402
    frame_pair_fp64_escalations, reset_frame_pair_fp64_escalations,
)
from vfe3.model.model import VFEModel  # noqa: E402
from vfe3.numerics import safe_eigvalsh, triangular_condition_estimate  # noqa: E402
from vfe3.relational_scoring import (  # noqa: E402
    _projected_kl_tile, _projected_linear_map, projected_gaussian_kl, projected_moments,
    reset_source_recovery_elements, set_projected_condition_recovery, set_projected_factor_route,
    source_recovery_elements,
)

EPS32 = torch.finfo(torch.float32).eps


def load_run(run_dir: Path, device: torch.device) -> Tuple[VFE3Config, VFEModel, str, List[str]]:
    r"""Rebuild a saved run, dropping retired config fields, and load its final weights strictly."""
    data = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    raw = dict(data["config"])
    known = {f.name for f in dataclasses.fields(VFE3Config)} | {"relational_mode", "relational_readout_mode"}
    dropped = sorted(k for k in raw if k not in known)
    for key in dropped:
        raw.pop(key)
    cfg = config_from_serialized(raw, source=str(run_dir))
    payload = torch.load(run_dir / "final_model.pt", map_location="cpu", weights_only=True)
    model = VFEModel(cfg)
    model.load_state_dict(payload["model_state"], strict=True)
    # Measure float32 arithmetic, not TF32: the run may have trained under "high".
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return cfg, model.to(device), data.get("dataset", ""), dropped


def capture_scorer_inputs(model: VFEModel, token_ids: torch.Tensor) -> List[Dict[str, object]]:
    r"""Run one forward and return the float32 inputs of every projected Gaussian-KL scorer call."""
    calls: List[Dict[str, object]] = []
    context: Dict[str, object] = {}
    readout, scorer = e_step_mod.relational_predictive_readout, relational_mod.projected_gaussian_kl

    def readout_wrapper(mu, sigma, frame, operators, **kwargs):
        context.update(tau=kwargs.get("tau"), kl_max=kwargs.get("kl_max"), log_prior=kwargs.get("log_prior"))
        return readout(mu, sigma, frame, operators, **kwargs)

    def scorer_wrapper(a_q, G_q, a_k, G_k, *, projection_q=None, projection_k=None):
        sigma_h, inverse, S_q, mu_h = projection_q
        tau = context["tau"]
        calls.append(dict(
            mu_h=mu_h.detach().clone(), sigma_h=sigma_h.detach().clone(), inverse=inverse.detach().clone(),
            S=torch.stack((S_q.detach(), projection_k[2].detach())).clone(),
            tau=float(tau) if not torch.is_tensor(tau) else float(tau.detach().float().mean()),
            kl_max=float(context["kl_max"]),
            log_prior=None if context["log_prior"] is None else context["log_prior"].detach().float()))
        return scorer(a_q, G_q, a_k, G_k, projection_q=projection_q, projection_k=projection_k)

    e_step_mod.relational_predictive_readout = readout_wrapper
    relational_mod.projected_gaussian_kl = scorer_wrapper
    try:
        with torch.no_grad():
            model(token_ids)
    finally:
        e_step_mod.relational_predictive_readout = readout
        relational_mod.projected_gaussian_kl = scorer
    return calls


def _sym(M: torch.Tensor) -> torch.Tensor:
    return .5 * M + .5 * M.transpose(-1, -2)


def cholesky_qr2(X: torch.Tensor, shift: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Lower factor C with C C^T = X X^T for X (..., r, m), by (shifted) CholeskyQR2/3.

    Unshifted: C1 = chol(X X^T), Y = C1^{-1} X, C = C1 chol(Y Y^T). Shifted (Fukaya et al. 2020):
    C1 = chol(X X^T + s I) with s = 11 (m r + r (r + 1)) u ||X||_F^2 as a preconditioner only, then
    CholeskyQR2 on C1^{-1} X. Returns (C, failed).
    """
    r, m = X.shape[-2:]
    W1 = _sym(X @ X.transpose(-1, -2))
    if shift:
        s = 11 * (m * r + r * (r + 1)) * torch.finfo(X.dtype).eps * X.square().sum(dim=(-2, -1))
        W1 = W1 + s[..., None, None] * torch.eye(r, device=X.device, dtype=X.dtype)
    C1, info1 = torch.linalg.cholesky_ex(W1)
    Y = torch.linalg.solve_triangular(C1, X, upper=False)
    if shift:
        C2, failed = cholesky_qr2(Y)
        return C1 @ C2, (info1 != 0) | failed
    C2, info2 = torch.linalg.cholesky_ex(_sym(Y @ Y.transpose(-1, -2)))
    return C1 @ C2, (info1 != 0) | (info2 != 0)


def source_root(mu_h, sigma_h, inverse, S, dtype, whitening=None):
    r"""Return (a, X) with a = S U^{-1} mu and X = S U^{-1} Sigma^{1/2} for diagonal Sigma.

    ``whitening`` (R, ..., H, r, r) left-multiplies both, per role, as a common congruence.
    Shapes: a (R, B, N, H, r), X (R, B, N, H, r, K).
    """
    mu, sigma, inv = mu_h.to(dtype), sigma_h.to(dtype), inverse.to(dtype)
    L = torch.stack([_projected_linear_map(S[role].to(dtype), inv) for role in range(S.shape[0])])
    if whitening is not None:
        L = whitening.to(dtype).unsqueeze(-4) @ L
    a = (L @ mu.unsqueeze(-1).unsqueeze(0)).squeeze(-1)
    X = L * sigma.sqrt().unsqueeze(0).unsqueeze(-2)
    return a, X


def kl_from_parts(aq, ak, q_matrix, k_matrix, logdet_q, logdet_k, factor_trace: bool, q_block: int = 8):
    r"""Pair KL grid (B, H, Nq, Nk) from token-major parts, tiled over query rows."""
    aq, ak = aq.movedim(-2, -3), ak.movedim(-2, -3)
    q_matrix, k_matrix = q_matrix.movedim(-3, -4), k_matrix.movedim(-3, -4)
    logdet_q, logdet_k = logdet_q.transpose(-1, -2), logdet_k.transpose(-1, -2)
    rows = []
    for i in range(0, aq.shape[-2], q_block):
        rows.append(_projected_kl_tile(aq[..., i:i+q_block, :], ak, q_matrix[..., i:i+q_block, :, :], k_matrix,
                                       logdet_q[..., i:i+q_block], logdet_k, factor_trace))
    return torch.cat(rows, dim=-2)


def _logdet(C: torch.Tensor) -> torch.Tensor:
    return 2 * C.diagonal(dim1=-2, dim2=-1).log().sum(-1)


def _quantiles(values: torch.Tensor, qs=(0.5, 0.9, 0.99, 0.999)) -> Dict[str, float]:
    v = values.detach().double().flatten()
    v = v[torch.isfinite(v)]
    if v.numel() == 0:
        return {}
    if v.numel() > 8_000_000:
        v = v[torch.randperm(v.numel(), device=v.device)[:8_000_000]]
    out = {f"p{int(q * 1000) / 10:g}": float(torch.quantile(v, q)) for q in qs}
    out["max"] = float(v.max())
    out["mean"] = float(v.mean())
    return out


def measure_call(call: Dict[str, object]) -> Dict[str, object]:
    r"""All Phase 0 per-call measurements for one captured scorer call."""
    mu_h, sigma_h, inverse, S = call["mu_h"], call["sigma_h"], call["inverse"], call["S"]
    tau, kl_max, log_prior = call["tau"], call["kl_max"], call["log_prior"]
    out: Dict[str, object] = {"tau": tau, "kl_max": kl_max, "shape": list(mu_h.shape), "rank": S.shape[-2]}
    marks: Dict[str, float] = {}
    last = [time.perf_counter()]

    def mark(name: str) -> None:
        torch.cuda.synchronize()
        now = time.perf_counter()
        marks[name] = round(now - last[0], 2)
        last[0] = now
    out["section_seconds"] = marks

    # Oracle: float64 from the float64 sources.
    a64, G64 = projected_moments(mu_h.double(), sigma_h.double(), inverse.double(), S.double())
    kl64 = projected_gaussian_kl(a64[0], G64[0], a64[1], G64[1])                     # (B, H, N, N)
    support = torch.isfinite(log_prior).expand_as(kl64) if log_prior is not None else torch.ones_like(kl64, dtype=torch.bool)
    prior = torch.zeros_like(kl64) if log_prior is None else log_prior.double().expand_as(kl64)

    def clamp(kl):
        return kl.double().clamp(0., kl_max).nan_to_num(nan=kl_max, posinf=kl_max, neginf=0.)

    ref = clamp(kl64)
    beta_ref = torch.softmax(-ref / tau + prior, dim=-1)

    def error(kl) -> Dict[str, float]:
        value = clamp(kl)
        diff = (value - ref).abs()[support] / tau
        beta = torch.softmax(-value / tau + prior, dim=-1)
        return {"kl_err_over_tau": _quantiles(diff),
                "beta_err_max": float((beta - beta_ref).abs().max()),
                "nonfinite_raw": int((~torch.isfinite(kl)).sum())}

    errors: Dict[str, object] = {}
    # Today's float32 explicit route, recovery off and on.
    a32, G32 = projected_moments(mu_h, sigma_h, inverse, S)
    projection = [(sigma_h, inverse, S[role], mu_h) for role in range(2)]
    for name, recovery in (("fp32_gram_today", False), ("fp32_gram_fp64_recovery", True)):
        previous = set_projected_condition_recovery(recovery)
        reset_source_recovery_elements()
        try:
            kl = projected_gaussian_kl(a32[0], G32[0], a32[1], G32[1],
                                       projection_q=projection[0], projection_k=projection[1])
            errors[name] = error(kl) | {"recovered_entries": source_recovery_elements()}
        except ValueError as exc:
            errors[name] = {"error": str(exc)}
        finally:
            set_projected_condition_recovery(previous)
        mark(name)

    # The implemented WS1 route (projected_factor_route="cholesky_qr2"), recovery off and on.
    for name, recovery in (("ws1_impl", False), ("ws1_impl_condition_recovery", True)):
        previous_route = set_projected_factor_route("cholesky_qr2")
        previous = set_projected_condition_recovery(recovery)
        reset_source_recovery_elements()
        try:
            kl = projected_gaussian_kl(a32[0], G32[0], a32[1], G32[1],
                                       projection_q=projection[0], projection_k=projection[1])
            errors[name] = error(kl) | {"recovered_entries": source_recovery_elements()}
        except ValueError as exc:
            errors[name] = {"error": str(exc)}
        finally:
            set_projected_condition_recovery(previous)
            set_projected_factor_route(previous_route)
    mark("oracle_and_today")
    # Trigger rates of today's certificates on the float32 Grams.
    chol32, info32 = torch.linalg.cholesky_ex(G32)
    failed32 = (info32 != 0) | ~torch.isfinite(chol32).all(dim=(-2, -1))
    eye = torch.eye(G32.shape[-1], device=G32.device)
    inv_norm = torch.cholesky_inverse(torch.where(failed32[..., None, None], eye, chol32)).norm(dim=(-2, -1))
    gram_cert = (~failed32) & (EPS32 * G32.norm(dim=(-2, -1)) * inv_norm > 1e-4)
    out["trigger_fraction"] = {
        "fp32_cholesky_failed": [float(failed32[role].float().mean()) for role in range(2)],
        "gram_certificate_1e-4": [float(gram_cert[role].float().mean()) for role in range(2)],
    }

    # Conditioning of the sources, from float64 Gram spectra (kappa(X)^2 = kappa(G)).
    ev = safe_eigvalsh(G64)                                                   # (2, B, N, H, r)
    kappa_X = (ev[..., -1] / ev[..., 0]).sqrt()
    U_inv = inverse.double()
    ev_U = safe_eigvalsh(U_inv.transpose(-1, -2) @ U_inv)
    kappa_U = (ev_U[..., -1] / ev_U[..., 0]).sqrt()
    U = torch.linalg.inv(U_inv)
    kappa_F_U = torch.linalg.matrix_norm(U) * torch.linalg.matrix_norm(U_inv)
    sigma_tok = sigma_h[..., 0, :].double()
    kappa_sigma = sigma_tok.amax(-1) / sigma_tok.amin(-1)
    sv_S = torch.linalg.svdvals(S.double())
    out["kappa"] = {
        "X_query": _quantiles(kappa_X[0]), "X_key": _quantiles(kappa_X[1]),
        "G_query": _quantiles(kappa_X[0].square()), "G_key": _quantiles(kappa_X[1].square()),
        "U_block": _quantiles(kappa_U), "Sigma_token": _quantiles(kappa_sigma),
        "S_query_per_head": (sv_S[0, :, 0] / sv_S[0, :, -1]).tolist(),
        "S_key_per_head": (sv_S[1, :, 0] / sv_S[1, :, -1]).tolist(),
    }
    out["trigger_fraction"].update({
        "frame_pair_kappaF_eps_1e-4_blocks": float((kappa_F_U * EPS32 > 1e-4).float().mean()),
        "frame_pair_call_escalates": bool((kappa_F_U.amax() * EPS32 > 1e-4)),
        "congruence_cond4_eps_1e-3_blocks": float((kappa_U.pow(4) * EPS32 > 1e-3).float().mean()),
        "eps_kappaX_gt_1e-4": [float((kappa_X[role] * EPS32 > 1e-4).float().mean()) for role in range(2)],
    })
    mark("certificates_and_svd")
    # WS1 CholeskyQR2 factors in float32 from the float32 sources.
    a32s, X32 = source_root(mu_h, sigma_h, inverse, S, torch.float32)
    C, qr_failed = cholesky_qr2(X32)
    C_shift, shift_failed = cholesky_qr2(X32, shift=True)
    out["trigger_fraction"]["cholqr2_failed"] = [float(qr_failed[role].float().mean()) for role in range(2)]
    out["trigger_fraction"]["shifted_cholqr3_failed"] = [float(shift_failed[role].float().mean()) for role in range(2)]
    C64 = torch.linalg.cholesky(G64)
    R_diag = C.diagonal(dim1=-2, dim2=-1).abs()
    out["kappa"]["R_diag_ratio_query"] = _quantiles((R_diag.amax(-1) / R_diag.amin(-1))[0])
    out["kappa"]["R_diag_ratio_key"] = _quantiles((R_diag.amax(-1) / R_diag.amin(-1))[1])
    C1n = torch.linalg.matrix_norm(C.double(), ord=1) * torch.linalg.matrix_norm(
        torch.linalg.inv(C.double()), ord=1)
    out["kappa"]["R_1norm_query"] = _quantiles(C1n[0])
    out["kappa"]["R_1norm_key"] = _quantiles(C1n[1])
    eye64 = torch.eye(C64.shape[-1], dtype=torch.float64, device=C64.device)

    def whitened(Cfac):
        return (torch.linalg.solve_triangular(C64, Cfac.double(), upper=False) - eye64).norm(dim=(-2, -1))

    white_qr2 = whitened(C)
    white_gram = whitened(torch.where(failed32[..., None, None], eye, chol32))
    kappa1_est = triangular_condition_estimate(C)
    out["kappa"]["R_hager_1norm_est_query"] = _quantiles(kappa1_est[0])
    out["kappa"]["R_hager_1norm_est_key"] = _quantiles(kappa1_est[1])
    out["whitened_factor_err"] = {
        "cholqr2_over_eps_kappaX": _quantiles(white_qr2 / (EPS32 * kappa_X)),
        "gram_over_eps_kappaX2": _quantiles(white_gram / (EPS32 * kappa_X.square())),
        "cholqr2": _quantiles(white_qr2), "gram": _quantiles(white_gram)}
    for tol in (1e-5, 3e-5, 1e-4):
        out["trigger_fraction"][f"eps_hager1_gt_{tol:g}"] = [
            float((EPS32 * kappa1_est[role] > tol).float().mean()) for role in range(2)]
    out["factor_rel_err"] = {
        "cholqr2": _quantiles(((C.double() - C64).norm(dim=(-2, -1)) / C64.norm(dim=(-2, -1)))),
        "shifted_cholqr3": _quantiles(((C_shift.double() - C64).norm(dim=(-2, -1)) / C64.norm(dim=(-2, -1)))),
        "gram_cholesky": _quantiles(((torch.where(failed32[..., None, None], eye, chol32).double() - C64).norm(dim=(-2, -1))
                                     / C64.norm(dim=(-2, -1)))[~failed32]),
        "eps_kappaX": _quantiles(EPS32 * kappa_X),
    }

    def factor_variants(prefix: str, a, Cfac, Gq_explicit):
        ld = _logdet(Cfac)
        Pk = torch.cholesky_inverse(Cfac[1])
        errors[f"{prefix}_gemm_explicitGq"] = error(kl_from_parts(a[0], a[1], Gq_explicit, Pk, ld[0], ld[1], False))
        Gq_ref = _sym(Cfac[0] @ Cfac[0].transpose(-1, -2))
        errors[f"{prefix}_gemm_reformedGq"] = error(kl_from_parts(a[0], a[1], Gq_ref, Pk, ld[0], ld[1], False))
        errors[f"{prefix}_sqrt_pair"] = error(kl_from_parts(a[0], a[1], Cfac[0], Cfac[1], ld[0], ld[1], True, q_block=4))

    factor_variants("ws1_cholqr2", a32s, C, G32[0])
    # Per-pair error model: |dKL_ij| <= c eps (kappa_q,i + kappa_k,j) S_ij, S = tr + quad + r.
    ld64 = _logdet(C64)                                                               # (2, B, N, H)
    kl_ws1 = kl_from_parts(a32s[0], a32s[1], G32[0], torch.cholesky_inverse(C[1]),
                           _logdet(C)[0], _logdet(C)[1], False).double()
    r_dim = S.shape[-2]
    S_pair = (2 * kl64 + 2 * r_dim - ld64[1].transpose(-1, -2).unsqueeze(-2)
              + ld64[0].transpose(-1, -2).unsqueeze(-1))                             # (B, H, Nq, Nk)
    kq, kk = kappa_X[0].transpose(-1, -2), kappa_X[1].transpose(-1, -2)            # (B, H, N)
    model = EPS32 * (kq.unsqueeze(-1) + kk.unsqueeze(-2)) * S_pair
    in_range = support & (kl64 <= kl_max)
    out["pair_error_model_ratio_ws1"] = _quantiles(((kl_ws1 - kl64).abs() / model)[in_range])
    out["pair_S_over_tau"] = _quantiles((S_pair / tau)[in_range])

    mark("ws1")
    # Common whitening A = Gbar_k^{-1/2} per (b, h), from float64 key Grams (detached, exact invariance).
    Gbar = G64[1].mean(dim=-4)                                                        # (B, H, r, r)
    evals, evecs = torch.linalg.eigh(Gbar)
    A = (evecs * evals.rsqrt().unsqueeze(-2)) @ evecs.transpose(-1, -2)               # (B, H, r, r)
    aw, Xw = source_root(mu_h, sigma_h, inverse, S, torch.float32, whitening=torch.stack((A, A)).float())
    Gw = _sym(Xw @ Xw.transpose(-1, -2))
    try:
        errors["whiten_gram"] = error(projected_gaussian_kl(aw[0], Gw[0], aw[1], Gw[1]))
    except ValueError as exc:
        errors["whiten_gram"] = {"error": str(exc)}
    Cw, w_failed = cholesky_qr2(Xw)
    factor_variants("whiten_cholqr2", aw, Cw, Gw[0])
    Gw64 = Xw.double() @ Xw.double().transpose(-1, -2)
    ev_w = safe_eigvalsh(.5 * Gw64 + .5 * Gw64.transpose(-1, -2))
    kappa_w = (ev_w[..., -1] / ev_w[..., 0]).sqrt()
    out["kappa"]["X_whitened_query"] = _quantiles(kappa_w[0])
    out["kappa"]["X_whitened_key"] = _quantiles(kappa_w[1])
    out["errors"] = errors

    mark("whitening")
    # Pair-stage cancellation: sum |P_ab G_ab| / tr(P G), float64, causal support.
    Pk64 = torch.cholesky_inverse(C64[1]).movedim(-3, -4)                             # (B, H, N, r, r)
    Gq64 = G64[0].movedim(-3, -4)
    num = torch.einsum("zhixy,zhjxy->zhij", Gq64.abs(), Pk64.abs())
    den = torch.einsum("zhixy,zhjxy->zhij", Gq64, Pk64)
    out["pair_cancellation_ratio"] = _quantiles((num / den)[support])
    out["pair_gemm_rounding_over_tau"] = _quantiles((S.shape[-2] ** 2 * EPS32 * num / tau)[support])
    mark("pair_stage")
    return out


def _is_fp64_kernel(name: str) -> bool:
    # Profiler keys are usually Itanium-mangled: a double template argument is "Id"/"IKd" followed by
    # the next argument or "E" (larft_T_32IdLi1024E, CUDAFunctor_addIdE), and cutlass names the dtype
    # in the tile tag (tensorop_d884gemm).
    lowered = name.lower()
    return ("double" in lowered or "<d" in lowered or "complex<double" in lowered
            or re.search(r"(^|[_ ])d(gemm|trsm|potrf|syrk|geqrf|trsv|gemv|getrf|orgqr|ormqr)", lowered) is not None
            or re.search(r"I(K?d)+[EILNPRSdfij0-9_]", name) is not None
            or re.search(r"_d\d*gemm", lowered) is not None
            or "_f64" in lowered or "fp64" in lowered)


def profile_steps(model: VFEModel, batches: List[Tuple[torch.Tensor, torch.Tensor]], label: str,
                  steps: int = 10, traced: int = 3) -> Dict[str, object]:
    r"""Wall time over ``steps``; kernel-time shares and host syncs over ``traced`` forward+backward steps.

    CPU-activity tracing of full training steps is memory-heavy, so it covers only ``traced`` steps.
    """
    from torch.profiler import ProfilerActivity, profile

    def step(x, y):
        model.zero_grad(set_to_none=True)
        _, loss, _ = model(x, y)
        loss.backward()

    model.train()
    for x, y in batches[:2]:
        step(x, y)
    torch.cuda.synchronize()
    # Host syncs, counted on an unprofiled pass.
    torch.cuda.set_sync_debug_mode("warn")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for index in range(traced):
            step(*batches[index % len(batches)])
        torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode(0)
    syncs = sum("synchroniz" in str(w.message).lower() for w in caught)
    del caught
    start = time.perf_counter()
    for index in range(steps):
        step(*batches[index % len(batches)])
    torch.cuda.synchronize()
    wall = (time.perf_counter() - start) / steps
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for index in range(traced):
            step(*batches[index % len(batches)])
        torch.cuda.synchronize()
    events = prof.key_averages()
    kernel_total, fp64_total, linalg, kernels = 0.0, 0.0, {}, []
    for event in events:
        if event.device_type == torch.autograd.DeviceType.CUDA:
            kernel_total += event.self_device_time_total
            kernels.append((event.self_device_time_total, event.key))
            if _is_fp64_kernel(event.key):
                fp64_total += event.self_device_time_total
    for event in events:
        if any(key in event.key for key in ("cholesky", "eigh", "eigvalsh", "linalg_qr", "solve_triangular",
                                             "svd", "linalg_inv", "matrix_exp", "cholesky_inverse")):
            linalg[event.key] = round(event.device_time_total / 1e3 / traced, 3)
    return {"label": label, "wall_ms_per_step": 1e3 * wall, "cuda_kernel_ms_per_step": kernel_total / 1e3 / traced,
            "fp64_kernel_share": fp64_total / max(kernel_total, 1e-9), "host_syncs_per_step": syncs / traced,
            "linalg_cuda_ms_per_step": linalg,
            "top_kernels": [(round(t / 1e3 / traced, 3), _is_fp64_kernel(k), k[:120])
                            for t, k in sorted(kernels, reverse=True)[:15]]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batches", type=int, nargs="+", default=[0, 10, 20])
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--skip-measure", action="store_true", help="profile only")
    parser.add_argument("--profile-batch-size", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    cfg, model, dataset, dropped = load_run(args.run_dir, device)
    model.eval()
    loader = make_dataloader(dataset or "wikitext-103", "validation", cfg.max_seq_len, args.batch_size,
                             shuffle=False, drop_last=True, vocab_size=cfg.vocab_size)
    wanted = set(args.batches)
    batches = []
    for index, (x, y) in enumerate(loader):
        if index in wanted:
            batches.append((index, x.to(device), y.to(device)))
        if index >= max(wanted):
            break
    result: Dict[str, object] = {
        "run_dir": str(args.run_dir), "dropped_retired_fields": dropped,
        "shape": {"K": cfg.embed_dim, "H": cfg.n_heads, "N": cfg.max_seq_len, "B": args.batch_size,
                  "L": cfg.n_layers, "family": cfg.family, "attention_mode": cfg.attention_mode},
        "torch": torch.__version__, "device": torch.cuda.get_device_name(), "calls": []}
    for index, x, _ in ([] if args.skip_measure else batches):
        calls = capture_scorer_inputs(model, x)
        for layer, call in enumerate(calls):
            started = time.perf_counter()
            with torch.no_grad():
                measured = measure_call(call)
            measured.update(batch=index, layer=layer, seconds=round(time.perf_counter() - started, 2))
            result["calls"].append(measured)
            print(f"batch {index} layer {layer}: done in {measured['seconds']} s", flush=True)
            del call
        torch.cuda.empty_cache()
    if args.profile:
        profile_bs = args.profile_batch_size or args.batch_size
        train_loader = make_dataloader(dataset or "wikitext-103", "validation", cfg.max_seq_len, profile_bs,
                                       shuffle=False, drop_last=True, vocab_size=cfg.vocab_size)
        prof_batches = []
        for x, y in train_loader:
            prof_batches.append((x.to(device), y.to(device)))
            if len(prof_batches) == 4:
                break
        settings: List[Tuple[str, Dict[Callable, object]]] = [
            ("projected_cov_condition_recovery=True", {set_projected_condition_recovery: True}),
            ("projected_factor_route=cholesky_qr2", {set_projected_factor_route: "cholesky_qr2"}),
            ("projected_factor_route=cholesky_qr2+condition_recovery",
             {set_projected_factor_route: "cholesky_qr2", set_projected_condition_recovery: True})]
        profiles = [profile_steps(model, prof_batches, "all_off")]
        for label, values in settings:
            previous = {setter: setter(value) for setter, value in values.items()}
            reset_source_recovery_elements()
            reset_frame_pair_fp64_escalations()
            try:
                entry = profile_steps(model, prof_batches, label)
                entry["source_recovered_entries"] = source_recovery_elements()
                entry["frame_pair_escalated_calls"] = frame_pair_fp64_escalations()
            except Exception as exc:  # report, do not hide
                entry = {"label": label, "error": repr(exc)}
            finally:
                for setter, value in previous.items():
                    setter(value)
            profiles.append(entry)
        result["profile"] = {"batch_size": profile_bs, "wall_steps": 10, "traced_steps": 3, "entries": profiles}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
