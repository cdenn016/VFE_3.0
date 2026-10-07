# Precision optimization plan: removing the float64 fallbacks

Date: 2026-09-24. Scope: the float64 escalation paths (`projected_cov_condition_recovery`,
`frame_pair_fp64_escalation`, `congruence_cond_escalation`, `decode_ranking_fp64_escalation`, the
full-covariance congruence certificate) and the pure-path `"fp64"` precision settings. Target shape:
K >= 128, head/irrep block d = 32 (H = K/32), comparison rank r = 32, B = 64, N = 128, L = 4.

Status (updated 2026-10-07): this is the historical plan; `docs/2026-09-24.md` records what was built.
WS1 and WS2 shipped as `projected_factor_route="cholesky_qr2"` with its certificate (default `"gram"`),
and WS5 shipped as the decode ranking guard by candidate refinement. WS3 targeted the transported
route, which has since been retired together with `congruence_cond_escalation` and the full-covariance
congruence certificate, so those selectors no longer exist and `config_from_serialized` migrates or
refuses them. WS4 was not started; the parked and dropped options stand. Phase 0 measurements are in
`docs/2026-09-24.md`. The text below is the plan as written on 2026-09-24 and is not updated.

Original status (2026-09-24): plan only. No code had changed. No GPU measurement was possible: during
this investigation the RTX 5090 was at 90% utilization under another `python.exe` process, and
`AGENTS.md` forbids GPU workloads during the cleanup. Every cost figure below is an estimate from FLOP
and byte counts against spec-sheet rates (about 105 TFLOP/s float32 and 1.6 TFLOP/s float64). Phase 0
replaces them with measurements.

## 1. Thesis

The escalations fire because the float32 algorithms **square** condition numbers, not because the
problems are ill-conditioned. Three places do this:

- The projected route forms the Gram matrix G = X X^T, with X = S U^{-1} Sigma^{1/2}, and then
  factors and inverts it. Its accuracy is governed by kappa(G) = kappa(X)^2.
- The transported route forms the explicit congruence Omega Sigma Omega^T. Its conditioning squares
  kappa(Omega).
- The certificates are written as bounds on those squared quantities, some loosened further by
  Frobenius norms or fourth powers.

The saved runs show that the inputs themselves are mild. Fix the algorithms so they work with X, U
and Sigma^{1/2} directly. Then the same certificates, restated for the new algorithm, no longer
trigger at the observed conditioning, and float64 becomes a genuine last resort instead of a
per-step event.

## 2. Evidence from D:\DATA_KEEP

Conditioning telemetry from the thirteen saved wikitext-103 runs (`metrics.csv`, `phi_numerics.json`,
checkpoint weights) separates cleanly into four quantities.

The belief covariances are nearly isotropic: `belief_cond_median` is 1.25 to 2.0 in every run except
the 8-layer K=20 run, where it reaches 52 at its maximum. Sigma contributes almost nothing to the
ill-conditioning.

The frames are mild. In the K=128/N=128 run, `vertex_cond` (cond of exp(phi)) has median 7.8 and p99
10.6, with ||phi|| at about 6 and no clamp activation.

The learned comparison maps are mild but asymmetric. Across the K=64, K=128 and K=256 checkpoints,
cond(S_k) is 1.6 to 2.4. cond(S_q) is 3 to 25. The output map T ranges from 1.5 to 74.

The projected Gram matrices are where conditioning becomes large, because the other three multiply
and then square. `relational_projected_cov_max_condition` (batch maximum) has medians of 44 to 250 in
most runs, but 2338 (K=64) and 2652 (K=20 baseline), with maxima of 7505 and 5866. This is consistent
with kappa(G) of roughly (kappa(S) kappa(U))^2 kappa(Sigma): for example (8 x 10)^2 x 1.5, which is
about 10^4.

The current triggers fire against this background as follows. The projected certificate
eps ||G||_F ||G^{-1}||_F > 1e-4 trips at a Frobenius condition of about 840. Since
kappa_F <= r kappa_2 with r = 32, it can trip at kappa_2 of only a few hundred, which is inside the
**median** of most runs. The congruence trigger cond(exp phi)^4 eps > 1e-3 trips at cond(exp phi) of
about 9.6, below the observed p95 of 10.6, so it fires on essentially every call. The
`validated_cholesky_solve` condition limit 1/(64 d eps) is about 4096 at d = 32, below the observed
maxima. The decode ranking guard fires whenever any of the 50257 vocabulary intervals overlaps the
winner, which near ties is routine. The cost after a trigger is the real problem.
`_strict_cholesky` promotes the whole factor tensor to float64. Every pair tile touching a recovered
token then runs float64 triangular solves, which is O(N^2 r^3) at 1/64 rate. The ranking guard
recomputes the full (B, N, V) decode, about 3.1 GiB of float64 logits.

## 3. Principles

P1, square-root form. Never form a Gram matrix or congruence and then factor or invert it. Carry a
factor and compute the factor of the product directly. This is the square-root-filter argument
(Bierman 1977) and the normal-equations-versus-QR argument (Higham 2002, ch. 19 and 20). Error then
scales with kappa, not kappa^2.

P2, certify what is consumed. A certificate should bound the error of the quantity downstream code
reads: the KL entering a logit, measured in units of the temperature. It should not bound a relative
inverse error of an intermediate matrix. Certificates must describe the algorithm that actually ran.

P3, exact invariances as preconditioning. KL between Gaussians is invariant under any common
invertible affine map applied to both arguments. Two maps are free: the gauge pullback for flat
transports, and a common per-(b, h) whitening. Both change only rounding.

P4, no hidden syncs on the healthy path. Precision decisions accumulate as device flags and resolve
at most once per step.

P5, pure path always exists. Every new route is a toggle whose off state is today's behavior. The
existing float64 settings stay as the reference oracle.

## 4. Workstreams

### WS1 (option 1): square-root projected factors via CholeskyQR2

**Is this principled, and do we lose anything?** Yes, and nothing is lost in exact arithmetic. The
projected route consumes the Gaussian (a_i, G_i) with

    X_i = S U_i^{-1} C_i,   C_i C_i^T = Sigma_i,   G_i = X_i X_i^T,

where X_i is r x K and C_i = diag(sqrt(sigma_i)) for diagonal covariance. If X_i^T = Q_i R_i is a
reduced QR factorization with the signs of diag(R_i) fixed positive, then

    G_i = R_i^T Q_i^T Q_i R_i = R_i^T R_i,

so R_i^T is **exactly** the Cholesky factor of G_i. The Gaussian, the KL, the log-determinant
(2 sum log r_jj) and the precision P_i = R_i^{-1} R_i^{-T} are identical functions of the
parameters. The gradients are therefore identical in exact arithmetic. The forward value is the
same map, so no approximation, jitter, floor or projection enters.

What changes is rounding. The explicit route commits an absolute error of order eps ||X||^2 when it
forms G. Relative to lambda_min(G) = sigma_min(X)^2, that is eps kappa(X)^2, and it propagates into
the factor, the inverse and the log-determinant. A backward-stable QR of X^T commits backward error
eps ||X|| in X itself, so R, its inverse and its diagonal are accurate to about eps kappa(X), which is
eps kappa(G)^{1/2}. At the worst observed kappa(G) of 7505, that is about 1.0e-5 in float32 instead
of 8.9e-4. At the medians it is below 2e-6.

Three conditions attach to "nothing lost". First, X must be formed accurately. X = S U^{-1} C is a
product of three matrices with conditions of about 25, 10 and 1.5, so forming it in float32 costs
about eps times their product, roughly 4e-5 in the worst case. U^{-1} must come from the frame pair
as now; at cond(U) of about 10 its float32 inverse is accurate to about 1e-6. Second, the
pair-stage Frobenius product tr(P_k G_q) is a separate rounding source that WS1 does not touch
(Section 6). Third, sign fixing of diag(R) is non-smooth only where a pivot is exactly zero, which is
already the "lost numerical row rank" error case.

**Algorithm.** Use CholeskyQR2 (Fukaya et al. 2014; error analysis in Yamamoto et al. 2015) instead
of Householder QR, because it is all GEMM plus r x r Cholesky and batches well on the GPU:

    W1 = X X^T;          R1 = chol(W1)^T         (the current healthy path, kept as step 1)
    Y  = R1^{-T} X;      W2 = Y Y^T  (close to I)
    R2 = chol(W2)^T;     R  = R2 R1.

Step 2 corrects the squared-conditioning error of step 1. Yamamoto et al. show CholeskyQR2 is
backward stable (orthogonality and residual of order eps) whenever kappa(X) is at most about
eps^{-1/2}, which is about 2900 in float32. The observed worst case is about 90 (the square root of
7505), a margin of 30. Beyond that range, shifted CholeskyQR3 (Fukaya et al. 2020) extends validity
to kappa(X) of about eps^{-1} with one more pass. That should be the documented fallback rather than
float64.

**Cost.** This is an estimate: about 1 MFLOP of float32 per (token, head) at r = 32, K = 128, which
is 3.3e10 FLOP per call at the target shape. That is small next to the step, and it is O(N). For
comparison, the same stage in float64 is at least 20 ms per call at peak rate and several times that
in practice. This is why "run the per-token stage in float64 unconditionally" is **not** adopted.

**Where.** `vfe3/relational_scoring.py`: `_projected_moments_tile` returns X, or its pieces,
alongside G. `_cholesky_status` and `_strict_cholesky` gain a CholeskyQR2 factor route, with the
source tuple already threaded as `projection_q` / `projection_k`. `projected_gaussian_kl` then never
needs the float64 factor tiles on the healthy path. The existing float64 QR recovery (lines 276-311)
stays as the terminal fallback.

**Toggle.** Add a new projected factor-route toggle, default off. With it off, today's arithmetic
must be bit-identical. `projected_cov_condition_recovery` keeps its meaning. Its certificate is
restated for the factor route as eps kappa_est(R) > tol (WS2), which does not trip at the observed
conditioning, so the toggle can be turned on again.

### WS2 (option 2, revised): cheap certificates for the new algorithms

Replacing `eigvalsh` with the Frobenius bound ||A||_F ||A^{-1}||_F, as I suggested earlier, is
**rejected**. You were right to doubt it. That bound can exceed kappa_2 by a factor of up to d = 32.
With observed conditions already between the 4096 limit and 7505, it would escalate more often, not
less.

The revised plan certifies the **factor**. For a triangular R, a Hager-Higham 1-norm condition
estimate (Hager 1984; Higham and Tisseur 2000; the algorithm behind LAPACK xTRCON) costs O(r^2) per
matrix via a few triangular solves, typically within a factor of about 3 of the true value. The
cheaper lower bound max r_jj / min r_jj is available for free. The certificate becomes
eps kappa_est(R) > tol, which is valid because WS1's error scales with kappa(R). For the transported
full-covariance route, the `eigvalsh` in `validated_cholesky_solve` (`numerics.py:623`) is replaced
by the same estimator on its Cholesky factor, once that route consumes factors (WS3).

The tolerance should follow P2. Every consumer of the relational KL divides by tau before a
softmax, and a softmax is 1-Lipschitz in the max norm of its logits. So an absolute KL error
tolerance delta_KL gives logit error at most delta_KL / tau. The plan replaces the dimensionless 1e-4
with an absolute KL tolerance, with a proposed default of 1e-3 nats scaled by tau. The bound must be
derived per consumer before adoption. Decisions become device-side flags gathered into the single
`.tolist()` the projected path already performs (P4). The `bool()` syncs at `transport.py:707,716`
are folded the same way.

### WS3 (option 3): gauge-invariant pullback on the transported route

`_compact_pair_blocks` (`transport.py:2035`) builds Omega_ij = U_i U_j^{-1} explicitly for all N^2
pairs. `_compact_head_block_congruence` then forms Omega_ij Sigma_j Omega_ij^T, which is
(B, N, N, H, d, d): 4.2M blocks and about 0.8 TFLOP per call, and 34 GB if float64 were
untiled. The projected route already avoids this. The transported route can too, exactly:

    KL(Omega_ij q_j || q_i) = KL(U_j^{-1} q_j || U_i^{-1} q_i),

because KL is invariant under the common map U_i^{-1}. Define per-token pulled-back moments
m_i = U_i^{-1} mu_i and C~_i = U_i^{-1} Sigma_i U_i^{-T}, held as factors per WS1. Then the trace term
is a Frobenius-product GEMM with inner dimension d^2, the Mahalanobis term uses per-token precisions,
and the log-determinants are per-token. The cost drops from O(N^2 d^3) to O(N d^3 + N^2 d^2), and the
pair covariance is never materialized. Mixture consumers factor the same way:
sum_j beta_ij Omega_ij Sigma_j Omega_ij^T = U_i (sum_j beta_ij C~_j) U_i^T, so the mixing happens in
the pulled-back frame as a beta-weighted GEMM, followed by one congruence per query.

Mean-only RoPE (`rope_full_gauge=False`) needs one extra step, because it rotates means but not
covariances. Standard RoPE factors as R_{i-j} = R_i R_j^T. Put nu_j = R_j^T U_j^{-1} mu_j and
m_i = R_i^T U_i^{-1} mu_i. The Mahalanobis term is then (m_i - nu_j)^T R_i^T P~_i R_i (m_i - nu_j),
which still involves only per-token matrices. This derivation must be written out and checked
symbolically before implementation. Clamp surrogates are per token, and `pair_support` and `kl_max`
act after the divergence, so none of them break the identity.

The conditioning trade is real and must be checked, not assumed. The explicit pair product can be
better conditioned than its factors, for example when phi_i is close to phi_j. The diagonal family
already hit this cancellation (audit F3, `transport.py:2055-2070`). At the observed cond(U) of about
10, the pulled-back factors cost about kappa(U_i) kappa(U_j) eps, roughly 1e-5 in float32 with WS1's
square-root handling. That is acceptable, but Phase 0 must compare it with a float64 oracle on real
checkpoints. The inventory of consumers comes first: `coupling_energy` via `e_step.py:481,643`,
`gradients/kernels.py:802`, the oracle, and `families/gaussian.py:783`. This workstream matters only
for the transported and projected-conditional routes. The live `projected_predictive` route already
works this way.

**Common whitening, an addition under P3.** KL is also invariant under a common per-(b, h) map A
applied to both roles. Taking A = G_k_bar^{-1/2}, the inverse square root of the batch-mean key Gram
matrix, removes the common-mode part of the conditioning, dominated by cond(S_q)^2 up to about 600.
It also shrinks the Frobenius-product cancellation in Section 6. A can be detached exactly, since
d KL / d A = 0 identically, and it can be folded into S. The cost is one r x r eigendecomposition per
(b, h), 256 matrices. How much conditioning is common-mode rather than per-token is an empirical
question for Phase 0.

### WS4 (option 4): double-float kernels for the float64 that remains

After WS1 through WS3, float64 remains in the pure-path settings (`full_cov_kl_precision="fp64"`,
`full_cov_congruence_precision="fp64"`, `decode_av_precision="fp64"`), the oracles, the terminal
fallbacks, and small unconditional sites such as the M-step `matrix_exp` in
`gauge_optim.py:176,194`. Double-float arithmetic (Dekker 1971) represents each value as an
unevaluated sum of two float32 numbers: about 48 significand bits, with float32 exponent range. A
multiply-add costs about 10 to 20 float32 operations using FMA-based TwoProd and TwoSum. Against the
64:1 rate ratio, I estimate a 3 to 6 times gain for compute-bound kernels (unmeasured). There is no
gain where the kernel is limited by memory, because double-float moves the same bytes as float64.

The targets are batched Triton kernels for 32 x 32 GEMM, Cholesky, triangular solve and the
Taylor/Pade core of `matrix_exp`. These are the factorization-shaped operations that no GEMM
emulation covers. Two things must be established first. The exponent range must suffice: the
float32 range of 1e±38 covers the observed quantities, which include `kl_max` = 8K and variances of
order 1 to 100. The pure-path contract must also be settled: double-float is not IEEE float64, so the
existing `"fp64"` settings stay as the oracle. Double-float enters as a separate setting, for
example `"df64"`. This workstream is gated on Phase 0 showing float64 time that is still material
after WS1 through WS3.

### WS5: decode ranking certificate by candidate refinement

The ranking guard currently recomputes the whole (B, N, V) decode in float64 when **any** row is
uncertain. The ranking depends only on the candidates whose float32 intervals overlap the winner's
interval, which `_final_ranking_uncertain` (`prior_bank.py:477`) already computes. The exact
refinement recomputes, in float64, only the logits of those candidates for only the uncertain rows.
That costs O(B N k K) with k typically single digits, instead of O(B N V K). The result is the same
argmax as the full float64 recompute, because every non-candidate is certified below the winner. It
can be applied in the streamed chunk summaries (`_chunk_ranking_summary`) without materializing the
vocabulary axis. The guard stays opt-in. The CE loss path is unchanged.

### Parked (option 5): Ozaki / cuBLAS FP64 emulation

This is recorded for later, not pursued. The installed PyTorch
(`2.10.0.dev20251210+cu128`) bundles cuBLAS 12.8. Fixed-point FP64 emulation needs cuBLAS 13.0
Update 2 or later, enabled by `CUBLAS_EMULATE_DOUBLE_PRECISION=1` or the
`CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH` math mode, with compute capability 12.x listed. That requires
a separate cu13 environment, since the shared one is off-limits. It applies only to GEMMs. The two
GEMM-shaped sites are the decode a_v product (B N) x 2K x V, about 2e11 FLOP, and the projected trace
product (N x d^2) x (d^2 x N) per (b, h). Both run in float32 today, so emulation matters only if
their pure-path float64 settings become a bottleneck. Relevant work: Ozaki Scheme II
(arXiv:2504.08009) and FP64 emulation on RTX Blackwell (arXiv:2508.00441).

### Dropped (option 6): mixed-precision iterative refinement

This is dropped. The float64 residual it needs costs d^3, the same order as the float64
factorization it would replace.

## 5. Phase 0: measurement (when the GPU is free)

Nothing in Section 4 is implemented before these numbers exist. Run them on the K=64 and K=128
checkpoints in `D:\DATA_KEEP`, which contain the worst and a typical conditioning regime, at the
target shape. Use `$CUDA_PYTHON`.

1. **Profile.** Run `torch.profiler` over ten steps with the four escalation toggles off, then with
   each one on. Record the kernel-time share of float64 kernels, of `cholesky_ex`/`eigvalsh`, and of
   host syncs (`torch.cuda.set_sync_debug_mode("warn")`).
2. **Trigger rates.** Record the per-step fraction of (token, head) entries tripping each
   certificate, together with the per-token kappa distribution of X, U, Sigma and G. This splits the
   conditioning into common-mode and per-token parts for the whitening decision.
3. **Error against an oracle.** For the projected KL grid, measure the maximum absolute error in
   units of tau, relative to a full float64 recomputation from the float64 sources, under four
   variants: the float32 explicit route (today), WS1 CholeskyQR2, WS1 plus common whitening, and the
   float64 recovery. The same comparison runs for the transported route under WS3. Acceptance is an
   error at most the WS2 tolerance on every entry of every measured batch.
4. **Pair-stage cancellation.** Measure the ratio sum |P_ab G_ab| / tr(P G) on real grids. This
   decides whether the float32 Frobenius GEMM is adequate or whether the pair stage needs its own fix
   (Section 6).

## 6. Open risks

The pair-stage Frobenius product carries rounding of order r^2 eps sum |P_ab||G_ab|. That can exceed
eps tr(PG) when the off-diagonal entries cancel. WS1 does not change this term. Common whitening
reduces it. If Phase 0 shows it matters, there are two exact options. One is the square-root pair
form ||R_k^{-T} R_q^T||_F^2, a sum of squares with no cancellation, which costs O(N^2 r^3) in float32
(about 30 times the GEMM form; at the target shape about 2.7e11 FLOP, which is a few ms). The other
is a split-float32 Frobenius GEMM. The first is simpler and should be tried first.

The CholeskyQR2 validity bound of kappa(X) at most about 2900 in float32 has a margin of about 30 at
the observed maximum. Runs that drift far beyond it should fall to shifted CholeskyQR3, then to the
existing float64 recovery. Recovery counters must stay live so this is visible.

Autograd through CholeskyQR2 is a composition of GEMMs, Cholesky and triangular solves, all with
standard backward rules. It still needs a finite-difference smoke test at tiny dimensions, and a
gradient-agreement test against the float64 explicit route.

The literature citations here are from memory and must be checked against the primary sources
before any of this enters a manuscript: Bierman 1977; Higham 2002, *Accuracy and Stability of
Numerical Algorithms*, 2nd ed.; Fukaya et al. 2014 (ScalA); Yamamoto et al. 2015 (ETNA 44);
Fukaya et al. 2020 (SIAM J. Sci. Comput. 42); Hager 1984; Higham and Tisseur 2000; Dekker 1971.

## 7. Order of work and acceptance

**Phase 0** (measurement) gates everything.

**WS1 and WS2** go together first. They address the trigger that fires most and the slowest fallback
(per-pair float64 solves). They touch only `relational_scoring.py` and new estimator helpers in
`numerics.py`. Accepted when:

- the toggle-off path is bit-identical on the existing relational test suite;
- CholeskyQR2 factors agree with float64 Cholesky of G to eps kappa(X) on tiny CPU cases, including
  constructed kappa(X) of 10^3;
- the Phase 0 oracle error is within tolerance;
- recovery fires on zero entries across the measured checkpoint batches.

**WS5** is independent and small.

**WS3** follows WS1, because it reuses the factor machinery. It starts with the consumer inventory
and the symbolic check of the RoPE case.

**WS4** runs only if Phase 0 or later profiling shows material float64 time after WS1 through WS3.

Every implementation step records its changes in the dated docs log, keeps checkpoint parameter
names, and runs only tiny CPU tests (single-digit dimensions, float64 oracle comparisons,
finite-difference gradients) until the GPU is available.
