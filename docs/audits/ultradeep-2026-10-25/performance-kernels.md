# Performance, kernels, and custom autograd investigation

Audited the live working tree at `92a7cdfb5bf02172feae22d698195c0abc09e738` on October 25, 2026. This lane read source only and changed no implementation. The coordinator separately executed the tiny CPU Triton-interpreter reproduction cited below. No GPU query, training, profiler, CUDA timing, or allocator measurement was performed for this lane. The performance opportunities below have exact source counts; their wall-clock benefit is unmeasured. None of the findings has been independently adjudicated.

The resolved launcher in `live-config.json` has B = 32, N = 64, K = 32, H = 2, four layers, one inference iteration, `gaussian_frame_diagonal`, exact minimal-agent attention, and tied rank-2 cue/result memory. It uses the diagonal KL decoder with `decode_ce_triton=True` on CUDA. The expected-likelihood defect is on another supported decoder, outside this launcher. Cue/result posterior normalization, low-rank variance cancellation, and dense-memory tiling belong to another investigator and are excluded here.

## Unmeasured performance opportunities

### PK1. Exact attention rebuilds and synchronizes the same causal tile plan eight times per live forward

**Location:** `vfe3/model/minimal_agent.py:445-457`, `vfe3/model/minimal_agent.py:1382-1388`; prior lifetime at `vfe3/model/model.py:669-694`.

**Severity:** medium (P2).

**Trigger and scale:** `predictive_readout="minimal_agent"`, `minimal_agent_inference="exact"`, and a pair grid containing more than one tile. This includes the resolved launcher. For the live B = 32, N = 64, K = 32, H = 2, comparison rank r = 16, the 32 MiB pair budget gives `budget = floor(33554432 / (4 * 32 * 3 * 16**2)) = 341`, `q_step = 32`, `k_step = 10`, and a 2-by-7 tile grid.

**Evidence:** Every `exact_record_posterior` call executes:

```python
grid = torch.zeros(n_q * q_step, n_k * k_step, dtype=torch.bool, device=a.device)
grid[:I, :J] = torch.isfinite(lp).reshape(-1, I, J).any(dim=0)
live = grid.view(n_q, q_step, n_k, k_step).any(dim=3).any(dim=1).tolist()
```

The exact head update calls this function once per head. Four layers times two heads times one inference iteration therefore make eight `.tolist()` transfers per ordinary live forward, in addition to the separate per-layer factorization/covariance verdict. The prior itself comes from `_attention_log_prior`, whose model cache preserves it at fixed sequence length/device/dtype. ALiBi changes finite prior values across heads but keeps the same causal support. The tile plan reads only that support and shape/precision/budget metadata; it reads no learned values, posterior moments, or current token values. A healthy forward therefore repeats the same scheduling decision across heads, layers, and steps. On CUDA each `.tolist()` requires the host to wait for the produced mask. The temporary finite mask also has B N squared logical entries after broadcasting, although the source prior is batch independent.

**Consequence:** Eight avoidable launch-queue drains in the active forward. This is a count, not an eight-times slowdown claim. Increasing H or layer depth increases the transfer count linearly; an escalated/recomputed body can repeat the work again.

**Fix:** Cache the tile-support plan at the owner of the stable prior, keyed by prior identity/version, dimensions, batch layout, compute dtype, device, and pair budget, and pass it through to the exact head updates.

**Numerics:** Reusing the identical Boolean plan changes no floating-point arithmetic or reduction order. Preserve the current plan for arbitrary user-supplied or changed priors, and invalidate on relevant shape/precision changes. Caching a newly created broadcast/view object's identity inside `exact_record_posterior` alone would miss the reuse.

**Test/reproduction suggestion:** Use a tiny exact model with N greater than 32 to force multiple query tiles, intercept only the scheduling `.tolist()` or extract a plan helper, and count first versus second forward builds. Compare output moments and all parameter gradients bitwise before/after cache reuse; mutate the prior and verify rebuilding. A later CUDA sync-debug/profiler pass should measure the benefit when the device is free.

**Falsifies:** A live-call trace showing that these eight calls are already cached or not reached, or that the source mask depends on learned/batch state and cannot be reused under the proposed key. A CUDA profile may show negligible wall-clock benefit without changing the counted redundancy.

**Prior work:** The October 17 audit discussed other synchronization sites, and the October 20 covariance remediation caches causal-support checks for the kernel readout. Neither supplies a plan cache for this exact minimal-agent loop. This is not a request to remove its required numerical failure checks.

### PK2. The new frame-diagonal minimal-agent route still materializes block covariances and multiplies its gains by identity frames

**Location:** `vfe3/model/relational_operators.py:1412-1417`, `vfe3/model/relational_operators.py:1495-1523`, `vfe3/model/relational_operators.py:1222-1233`, `vfe3/model/minimal_agent.py:638-655`, `vfe3/model/minimal_agent.py:1634-1642`.

**Severity:** medium (P2) at K around 128 to 300 and sequence lengths in the hundreds; low (P3) at the current K = 32 launcher.

**Trigger and scale:** `family="gaussian_frame_diagonal"` with the minimal-agent readout and posterior covariance. The new dispatch obtains `(xi, identity_frame)` and enters the generic local-frame implementation. The input covariance is exactly the stored reference diagonal c, and the remaining frame in that body is an identity view.

**Evidence:** `_minimal_agent_body` calls `_reference_covariance`, whose diagonal block path performs

```python
C = (inverse * sigma_h.unsqueeze(-2)) @ inverse.transpose(-1, -2)
return 0.5 * (C + C.transpose(-1, -2))
```

Here `inverse` is identity. The two calls of `gain_basis(C, S/T, blocks=True)` then contract those dense diagonal blocks against the maps. Finally `local_posterior_covariance` forms `Z = local(gains)` with

```python
torch.einsum("...nmab,...nhmbs->...nhmas", forward, X.reshape(...))
```

although `forward` is identity. Each resulting Z has B N H K d scalars and equals its input gains in exact arithmetic. The same body also applies identity to the input/output means.

The source allocation counts for fp32 are:

| Shape regime | Stored c | Materialized C, B N H d squared | Identity-applied Z, B N H K d |
| --- | ---: | ---: | ---: |
| Live: B32, N64, K32, H2, d16 | 0.25 MiB | 4 MiB | 8 MiB |
| Wider supported regime: B32, N256, K192, H6, d32 | 6 MiB | 192 MiB | 1,152 MiB |

These are individual tensors, not measured peaks or a sum of simultaneously live allocations. C construction also creates intermediates and is repeated at every layer. The gain bases themselves are still required by the current sequential-head algorithm; the proposal does not claim to eliminate them.

**Consequence:** The reference-diagonal family avoids dense local covariances but this generic adapter still pays block-diagonal storage and identity contractions. The cost grows with d or K even though the input law has only K covariance coordinates. At larger supported widths, the identity-applied gain copy alone is a substantial tensor.

**Fix:** Add a reference-diagonal minimal-agent entry that reads c directly, forms `C S_h^T` and `C T_h^T` by coordinatewise scaling, and evaluates the diagonal posterior downdate directly from gains without applying the identity frame.

**Numerics:** The mathematical result is unchanged, but replacing contractions and their backwards can change floating-point reduction order. Treat the specialized route as numerics-changing until forward/backward parity establishes a stronger contract; gate it under the project's numerical-toggle practice. Retain the generic route as the reference.

**Test/reproduction suggestion:** Extend `tests/test_exact_route_20261025.py` with direct generic-versus-specialized moment and parameter-gradient comparisons at tiny float64/fp32 shapes, including nonuniform variances and comparison RoPE. Count allocations or inspect saved tensors for C and Z removal; only a subsequent CUDA-event/profile run can establish throughput and peak-memory effects.

**Falsifies:** A dispatch trace showing the active frame-diagonal body already bypasses `_reference_covariance` and `local(gains)`, or a required mathematical dependency on nonidentity/off-diagonal input within that specialized entry. A CUDA profile can show the remaining pairwise inference dominates runtime, reducing the practical priority.

**Prior work:** October 20 item 46 was fixed for `projected_moments` by `_identity_frame_diagonal`; its tests are in `tests/test_audit_20261020_d2_perf_routes.py`. The October 25 minimal-agent extension enters a different body and does not use that specialization. This is new remaining headroom, not a claim that item 46's fix failed.

## Correctness defect outside the active launcher

### PK3. Expected-likelihood Triton CE becomes NaN after an all-negative-infinite prefix even when a later class gives a finite loss

**Location:** `vfe3/kernels/_expected_likelihood_triton.py:120-128`; CUDA dispatch at `vfe3/model/prior_bank.py:2027-2039`.

**Severity:** low (P3). Pathological numerical edge case on a supported optional decoder, not a demonstrated current-training failure.

**Trigger and scale:** `decode_mode="expected_likelihood_chunked"` on CUDA, with V greater than the forward tile width BV = 64 and the first vocabulary tile entirely at negative-infinite logits. This can arise from finite but extreme non-target means whose squared differences overflow float32. A later class must remain finite. K and sequence length may be one; no large model is needed.

**Evidence:** The online reduction starts at `m = -inf`, `l = 0`, then executes

```python
m_new = tl.maximum(m, tl.max(logit, 1))
l = l * tl.exp(m - m_new) + tl.sum(tl.exp(logit - m_new[:, None]), 1)
```

An all-negative-infinite first tile gives `m_new = -inf`; both exponent arguments involve `-inf - (-inf)`, so l becomes NaN and remains NaN even when a finite class appears in a later tile. The KL CE kernel already avoids this with `m_ref = tl.where(m_new == -inf, 0.0, m_new)` in `_kl_decode_triton.py:78-79`.

The coordinator executed `probes_operators.py:29-38` with `TRITON_INTERPRET=1`, hidden CUDA, one CPU thread, and `C:/anaconda/python.exe`. The coordinator's environment check reported torch `2.10.0.dev20251210+cu128`, CUDA build `12.8`, and `torch.cuda.is_available() == False` with `CUDA_VISIBLE_DEVICES=-1`; the same interpreter ran this probe. Inputs were S = 1, K = 1, V = 65, float32, query mean 0, both variances 0.5, means 3e19 for the first 64 classes and 0 for class 64, target 64, tau = 1, and no unigram bias. All input means/variances were finite. `probes_operators.json:148-153` records `cpu_interpreted=true`, reference CE `0.0`, fused CE `NaN`, and `fused_finite=false`. This establishes the interpreter path and arithmetic failure; it is not a hardware GPU run or a speed measurement.

**Consequence:** A vocabulary ordering that puts overflowing non-target scores before a finite target changes a defined finite forward CE into NaN. The training finite-loss gate then cannot use that step. The active diagonal KL decoder has the prefix guard and is not affected by this finding.

**Fix:** Use the KL kernel's finite reference-maximum convention for all-negative-infinite prefixes in the expected-likelihood forward reduction.

**Numerics:** This changes only the formerly NaN prefix case; ordinary finite tiles can retain their exact operation order. It does not by itself prove backward finiteness when energy terms overflow: zero-weight times infinite energy/derivative terms require a separate explicit policy, and ordinary eager autograd can also be nonfinite there.

**Test/reproduction suggestion:** Preserve the coordinator's 65-class interpreter case as a forward regression, plus an all-negative-infinite first tile produced by explicit zero-prior logits while keeping energies finite to test finite backward behavior separately. Add compiled CUDA parity when GPU access is authorized and the device is free.

**Falsifies:** The same supported forward call produces a finite loss equal to the reference before any fix, or validation demonstrably excludes all routes to the trigger. Existing normal-range parity tests do not cover an all-negative-infinite vocabulary prefix.

**Prior work:** The October 16/17 expected-likelihood fixes address small-temperature reduction accuracy, unsupported double backward, and spill-test budgets. The current small-temperature and normal-range parity tests do not supply this initial-prefix guard. No matching defect was found in the October 20 report.

## Coverage, exclusions, and residual checks

Read all six implementation files in `vfe3/kernels/`: the KL CE, expected-likelihood CE, and diagonal-moment wrappers and Triton bodies. Compared their explicit first-order gradients, clamp boundaries, reduction state, dtype/device guards, contiguous layouts, optional gradients, zero-row behavior, saved tensors, and `once_differentiable` boundaries with the torch call sites. No additional first-order derivative defect was established by source inspection. Existing parity tests are evidence of test coverage; they were not rerun by this lane.

Read the diagonal/full/expected-likelihood CE dispatch and factors in `prior_bank.py`, target validation, centered chunk summaries, fixed decode-bank cache, and decode-frame routing. The host-target validation consumes a one-use host-validation token when supplied; its fallback `.tolist()` is therefore not automatically charged to the loader's live path. The legacy `auto` checkpoint policy and newer `auto_aggregate` policy are distinct intentional choices; this report does not call the selected policy a defect. No generic family-decoder or retired V3_Transformer path was executed.

Read `relational_scoring.py` projected-moment routing, identity specialization, fused moment fallback, `_GramPrecision`, and `_IEEEFP32Recompute`; read the transport fixed-schedule exponential/adjoint, inverse validation, compact frame contexts, and forward reuse gates. The matrix-exponential custom Function retains its input and static bound, and its adjoint preserves the protected precision policy; no new failure was established. The complete general spectral-retraction and gauge-optimizer implementations belong to the geometry lane.

Read the model's per-forward transport sharing, cached attention prior/RoPE, scoped decode-bank reuse, shared content charts, and the main training-step transfer/gradient gates. `shared_content_charts` excludes minimal-agent operators because they store no O, so there is no unused chart exponential to report on this route. The canonical-frame block's dense vertex expansion applies to a different registered block mode; cue/result memory uses the reference block's compact frame, so it is not a live-route finding. Normal per-step loss/finiteness/norm transfers are batched, and verbose diagnostic work is gated; no new synchronization defect was inferred merely from a grep hit.

For the new exact route, read the sequential-head assembly, exact record-posterior tiles/checkpoint boundaries, identity-frame adapter, and diagonal posterior readout. The B N squared pairwise inference and the head-conditioned Cholesky operations are required by the selected model, not mislabeled as auxiliary inefficiency. Performance proposals here target only repeated support decisions and exact identity/diagonal structure.

Prior context consulted: `docs/audit-results_2026-10-20.md`, the relevant October 17 items, `docs/2026-10-25.md`, remediation tests, Research's index and transformer hub, and `Research/manuscripts/Attention/minimal_agent_model.tex` sections `ma:sec:model` and `ma:sec:readings`. The manuscript supplies the declared model and approximation boundary, not authority for a runtime claim. No theory adjudication was performed.

Residual limits are compiled sm_120 numerical parity, GPU synchronization duration, allocator peaks, and end-to-end throughput under the new launcher. The small CPU-interpreter counterexample is sufficient for PK3's scoped arithmetic finding, while PK1/PK2 remain optimization candidates pending CUDA measurement. No new evidence is claimed about live GPU contention because GPU access was explicitly excluded.
