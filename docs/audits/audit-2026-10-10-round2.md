# Deep audit, October 10, 2026 (round 2)

## Scope

The audit covered the whole VFE_3.0 repository (430 tracked Python files, 112 of them in `vfe3/`) at `main` = `b65b6df`, the merge of PR #91. That commit follows the first deep audit of the day (`docs/audits/audit-2026-10-10.md`, taken at `7bfdf27`) and its three remediation merges: PR #89 (the remediation batch and the approved model options), PR #90 (compact head-block frames) and PR #91 (the fused Triton expected-likelihood cross-entropy). Together they add about 6,300 lines in 70 files, and that diff was the first priority of every investigator. The user's uncommitted work was included as found: `train_vfe3.py`, `vfe3/config.py`, `ablation.py`, `train_baseline.py`, `paper_arms.py`, `multiseed_analysis.py`, `zzzzz.py`, `vfe3/viz/figure_worker.py`, three test files and several docs.

The active configuration is the `train_vfe3.py` launcher dictionary: K = 128 with four heads of 32, N = 128, batch 32 and six layers; `gaussian_diagonal` with a diagonal prior table; the `diagonal_chunked` decoder with an untied decode bank, `decode_tau = 0.1` and the unigram prior; `projected_predictive` with the kernel readout and moment-matched predictive covariance (floor 1e-5); the exponential content chart; untied comparison maps; Gaussian-KL routing with `routing_kl_max_reference = "row_min"` and `kl_max = 8K`; the `causal_alibi` prior; comparison-space RoPE at base 10,000 with `rope_comparison_covariance = "rotation_invariant"` and `rope_fraction = 1`; no attention residual; the coordinate block MLP with an affine LayerNorm in local coordinates and passthrough covariance; block_glk per-head frames under AdamW; bf16 autocast with `fp32_matmul_precision = "high"`; and `diagnose_canonical_free_energy = True`.

At the user's instruction the audit was static. No test was run on either the CPU or the GPU, and no investigator imported torch or `vfe3` or loaded a checkpoint. Investigators read code, docs and plain-text run artifacts (`metrics.csv`, `config.json`, `summary.json`). They checked derivations with standalone numpy, sympy or mpmath scripts in the session scratchpad and deleted those scripts afterward. Every numerical statement about GPU behavior below is therefore an emulation or a derivation, not a device measurement. No source was edited.

## Investigators dispatched

The first wave ran ten agents in parallel: all nine research-domain experts (gauge theorist, geometer, information geometer, variational, transformer-ML, numerical analyst, ML engineer, implementation engineer and philosophy of science), since the request covered the whole codebase and theory, and one general-purpose sweep of the core hot path (`vfe3/model`, `vfe3/gradients`, `vfe3/relational_scoring.py`, `vfe3/geometry`, `vfe3/families`, `vfe3/inference`, `vfe3/numerics.py`, `vfe3/kernels`). The second wave ran two further general-purpose sweeps, launched as first-wave agents finished so that no more than about ten ran at once. One covered training, configuration, data, artifacts and the entrypoints, including the new `vfe3/table_optim.py`. The other covered the analysis layers: `vfe3/viz`, `vfe3/semantics`, `vfe3/posthoc`, `vfe3/paper` and their drivers.

Each investigator received the active configuration, the project invariants, both punch lists of the day and the remediation log `docs/2026-10-10.md`. It was told not to re-report remediated items and to tag reopened ones. Because the user asked for improvements, investigators could also return concrete proposals, which are marked "improvement" below. All agents ran on the session model (Opus 5.5).

The eight most severe findings went through the JEV advisory screen (`jev_screen_findings`, jev-1.13.0). It flagged no duplicates. It judged the quoted evidence supportive for items 1, 2, 4, 5, 6 and 14, at confidences 0.37 to 0.86, and insufficient for items 3 and 15. No finding was dropped or downgraded on that basis.

## Findings by investigator

### ML engineer

The earlier run that reached about 59,000 train perplexity has no folder under `vfe3_runs/`. The restarted run `vfe3_runs/20261010-220438_wikitext-103_K128_block_glk_mlp_s6`, which uses the active launcher configuration, has one metrics row, at step 2500, and that row already shows the failure. Against the healthy K = 128 runs at the same step (122.3, 116.3 and 103.6 for 32.65, 36.41 and 34.48):

| Quantity at step 2500 | Live run | Healthy runs |
|---|---|---|
| Train perplexity | 784.5 | 103.6 to 122.3 |
| Pre-clip gradient norm | 16,847 (content maps 16,008; phi 5,163) | 4.01 in 32.65 |
| Smallest projected-covariance eigenvalue | 2.48e-7 | 0.52 to 0.93 |
| Largest layer fraction of capped scores | 0.936 | 0.0 in 32.65 and 36.41 |

The guards recorded nothing: `step_skipped` was 0, `matmul_precision_replays` 0, and `grad_finite` 1. The per-role pre-clip norms are computed on the device at every step, but only their finiteness reaches the host, so the first evidence of trouble arrived after 881 seconds (item 1).

The investigator reads the same row as a collapse of covariance scale rather than of conditioning. The last layer's largest condition number is 358.7, below the healthy runs' 1,706 to 7,681, so λ_max is at most about 8.9e-5. The trace-relative floor `floor · tr(C_i)/K` cannot bound an absolute scale in a stack with no residual and a norm that touches only the mean (item 3).

Under the active configuration the "mu" clipping role lumps together mu_embed, the untied decode means, the block MLP, the LayerNorm affine and both relational map groups. At step 2500 the content maps alone set a clip coefficient of 6.2e-5 for all of them. Since Adam's step is invariant to gradient rescaling, the clip does not bound the exploding group's update anyway (item 8). The same shared clip rescales the table natural-gradient step (item 9).

The dead-router warning missed this row: the capped fraction was 0.939, but the self weight was only 0.379 (item 7). The table Fisher diagnostics go stale on skipped steps and overstate the realized step (item 37). The decode-mean decay is grouped correctly, but its description as a Gaussian prior misdescribes decoupled decay, and its effect on the rare decode rows is still unmeasured (item 43). `matmul_precision_retry` replays only steps that raise, never steps skipped for non-finite gradients (improvement, item 58).

### Numerical analyst

The pair-KL GEMM expansion `cross = query @ key^T` is pinned to IEEE fp32 in the forward, but its backward runs under the caller's TF32 policy, as the docstring at `vfe3/relational_scoring.py:1154-1155` states. That backward rebuilds ∂L/∂P_j from terms that cancel to Σ_i g'_ij(G_i + δ_ij δ_ijᵀ). In a TF32 emulation at r = 32 and N = 128 with the active diagonal P(G), the relative error of that gradient grows as the variance shrinks against the squared spread: 1.1e-2 at a ratio of 0.1, 0.11 at 1e-2, 0.95 at 1e-3, and 4.3 at 1e-4. In the regime of the live run the router gradient would therefore be mostly rounding noise (item 2, reopening item 44 of the first audit, whose "acceptable" verdict was measured on a healthy run).

Without source tuples, the rotation-invariant path cannot recover a failed factorization (item 12). Under the active `rope_fraction = 1`, P(G) is exactly diagonal, so the active run is unaffected.

The bf16 block-MLP residual is the only reduced-precision input to the otherwise fp32 router. The emulation gives median shifts in the routing gap of 0.58 nats at a comparison variance of 1e-2 and 5.1 nats at 1e-4 (improvement, item 29). Four further findings are low:
- the deferred chart verdict lets `matrix_exp` run on infinite blocks, which can loop indefinitely on CUDA (item 41);
- the new source-root projection has an infinite gradient at zero variance (item 40);
- the Triton kernel accumulates its sum over k sequentially in fp32, an error of 6.9e-3 per row at K = 512 (improvement, item 56);
- the opt-in paths add host syncs (improvement, item 57).

The fused Triton cross-entropy, the closed-form key-Cholesky backward and the product-form chart were checked and found correct.

### Transformer-ML

With the attention residual off, a block passes token i's own content forward only through β_ii. At β = I and the exponential chart's initial O = 0, the layer reduces exactly to (μ, Σ) ↦ (μ, Σ + floor) followed by the norm and the MLP. Both positional mechanisms favor the self pair: ALiBi sets B_ii = 0, the row maximum, and comparison RoPE has R_0 = I. A numpy check shows that a self weight of 0.998 over 128 causal keys needs only a 42 to 58 nat gap per head. That is far below `kl_max`, and at that point the un-lock gradient is about 3e-5 per key. The investigator concludes that a self-locked identity head is the stable attractor of this architecture, and that no code path pushes back out of it (item 4).

The canonical router β* has its maximum on the diagonal in every row, so `canonical_beta_tv` falls as the model locks (item 18). The dead-router warning checks self-lock only under `row_min` or a tied map, and words the cause as capping rather than softmax saturation (item 7).

At `rope_fraction = 1`, `rotation_invariant` makes each head's 32 by 32 comparison covariance diagonal. At base 10,000, six of the sixteen planes rotate by less than 0.5 radians over the whole context (improvement, item 48). `inactive_fields` reads the wrong certificate tolerance under the rotation-invariant path (item 32). Two docs drift from the code: the README says the row minimum is detached, but it carries a gradient, and the user's config comment credits a log-determinant term that cancels (item 45).

### Gauge theorist

The per-coordinate self-coupling `state_dependent_per_coord` is built from ordered conditional KLs, which depend on the frame for the full and block families. The validator admits it there. It enters the loss whenever `mstep_self_coupling_weight > 0` and feeds the canonical diagnostic, yet the gauge-pure report has no flag for it, and the executed witness sets the prior equal to the belief, so it cannot see the dependence. In numpy, Σ_k c₀D_k/(b₀ + D_k) moved by 3.4e-2 under a per-head GL⁺(2) element, while Σ_k D_k moved by 2e-15 (item 16).

The pure-path report labels both RoPE covariance modes alike and does not record `rope_comparison_covariance` (item 33). The canonical β* carries no comparison RoPE (item 18). The remaining four are improvements:
- a covariant covariance pushforward for reference-coordinate blocks (item 51, reopening item 54 of the first audit);
- an exact-zero assertion in `_act_on_frame` (item 52);
- per-head instead of dense K by K float64 exponentials in the canonical diagnostic (item 53).

The compact head-block route was checked and is exactly the same group element as the dense one, with a sound cache key. The dual measurement is covariant. A gauge-pure path exists for `gaussian_full` and `gaussian_block` in reference coordinates.

### Geometer

The LayerNorm Jacobian has a second near-null direction besides the constant vector: on the normalized vector z_i its eigenvalue is ε/(v + ε). Under the pushforward on `gaussian_full`, the token-dependent radial direction D⁻¹z_i therefore carries only the floor. In a two-token fp64 KL at K = 16, the KL moved as 1/f: 1.45e4, 1.42e6 and 1.42e8 at f = 1e-2, 1e-4 and 1e-6. A Monte Carlo estimate puts the true radial variance at 1.59e-3, against 2.31e-6 from first order plus the floor (item 11, reopening item 10). The head-block and diagonal families are unaffected.

Under `rotation_invariant` a failed fp32 factorization raises, and refactoring the stored fp32 matrix in fp64 recovers none of the failures, whereas rebuilding from the source does (item 12, merged with the numerical analyst). Two low findings concern the table Fisher step:
- it runs before the frame step, so `fisher_group` evaluates its metric at the already-updated rows (item 34);
- its docstring calls the prior Fisher the metric of the step's objective (item 35).

Two improvements follow: a final-norm covariance pushforward (item 49), and an O(K²) rank-two form of the full-layout pushforward that matches the dense form to 5.9e-16 (item 50). The LayerNorm Jacobian, the diagonal closed form, the diagonal delta_full M-projection, the product-form chart, the table Fisher metric and the canonical energy formula were all re-derived and found correct.

### Information geometer

Under `relational_receiver_measurement = "inverse"`, `inv_ex` returns finite but wrong measurements at the condition numbers trained O' reaches. In fp64 at K = 128, the relative error of the minimizer is 6.9e-2 at a condition number of 1e8 and 1.0 at 1e12, which contradicts the docstring's promise of non-finite output. An inverse-free form exists, ξ* = O'[W̃ + αO'ᵀC_p⁻¹O']⁻¹(w̃ + αO'ᵀC_p⁻¹ξ_p) (item 14, merged with the variational expert's factor form P = T e^{−OT}).

Under comparison RoPE, the routing cost is exactly invariant under (S_Q, S_K) ↦ (A S_Q, A S_K) for A in the commutant of the RoPE torus. Nothing fixes that orbit, since S reaches the loss only through D and its group has no decay. The λ_min and condition diagnostics, the fp32 GEMM error estimate and the score-map gradients all move along it. In numpy, A = diag(1e-3 R, 30 R) changed the costs by 2.7e-15 but the GEMM error estimate from 1.1e-6 to 90 nats (item 13, merged with the core sweep).

The canonical router prefers self by construction (item 18). The trust-radius claim "KL ≤ r²/2" holds only to second order: the exact ratio is 1.049 at r = 0.1 and 1.158 at 0.3 (item 36). The table Fisher docstring overstates what its metric is (item 35).

The table step's wiring, the Triton logits and all six of its gradients (agreeing with fp64 finite differences to 3.2e-9), the pair-KL backward and the row_min gradient were checked and found correct. So was the claim that `kl_max` never changes β at τ = √32, because fp32 softmax underflows beyond 588 nats.

### Variational

The EM template term pairs the responsibilities of whichever posterior ran (exact, or the last CAVI sweep) with expectations under the block-input moments. With mean field and one sweep, it is the gradient of the template router's reduced free energy, to 1.7e-9. Under exact inference it misses ∇ log p(o | θ) by a relative 1.99. At three CAVI sweeps it matches neither objective. Its fixed point puts λ at V + c instead of the marginal-likelihood value V − c, and the test pins this hybrid gradient (item 15).

The canonical F is evaluated only at the beliefs each router read, so the decoded output never enters a reported F. Evaluating it also at `layer_converged` and logging ΔF_l per layer would test whether the readout lowers it (improvement, item 27). `inactive_fields` marks the diagnostic's inputs inert, and under the minimal agent the diagnostic uses a temperature the router does not (item 31). The prototype mean field has no label-staleness diagnostic (improvement, item 54). The figures switched on by the diagnostic are mislabeled (item 17).

Both CAVI implementations are true coordinate ascent. Their ELBOs rise at every sweep in numpy reimplementations. The moment-matched covariance is the exact moment of the kernel.

### Implementation engineer

Every new field is validated, has a test baseline and a launcher key, is serialized, and leaves default behavior unchanged. The `no_rope` arm of `beta_channel_decomposition` clears `comparison_rope_base`, which also gates the rotation-invariant projection. Under the live configuration it therefore removes P(G) as well as the rotation, while the interpretation string calls it "removing only the rotation". The live run will write this record at finalization (item 19, reopening item 5).

Under `m_table_update_mode = "fisher"`, `m_p_mu_lr` and `m_p_sigma_lr` set only the schedule shape and the decay. Yet they are logged as the table rates, printed in the banner, and swept by `ablation.py` and `parameter_optimization.py` without an inactive flag (item 20).

The user's working-tree flip of `decode_logits_amp` to True has no historical pin. `34.48`, the paper base `k128_exp`, has no such key in its `config.json`, so a reload, resume or `precision = "run"` posthoc evaluation would decode its linear logits in bf16, which it never trained with (item 21, user WIP, reopening item 25).

`expected_likelihood_kernel = "triton"` silently runs the torch kernel on CPU, and no artifact records which kernel ran (item 42). The final LayerNorm never pushes Σ (improvement, item 49). The compact head-block route has no remaining dense-frame consumer, and the Fisher refusals are complete.

### Philosophy of science

Since the first audit, the third seed of the total-parameter-matched transformer has finished. All three seeds reach test cross-entropy 3.362 to 3.368 (SD 0.0036) with 231.3M parameters in 7,550 to 7,681 s, against the 36.41 base's 3.592 with 232.9M parameters in 33,428 s. The gap is 0.224 to 0.230 nats on test and 0.157 to 0.161 on validation. Repricing the 32 rare test targets at their unigram cost leaves 0.155 to 0.161, so the reversal is not the rare-target artifact. README lines 694 to 696 still say the read-table comparison "removes the parameter confound" (item 5, reopening item 1).

The paper's decision rule iterates only over clauses (a), (b) and (c). Neither the total-matched arm, which the user's working tree moves into TIER_A, nor the mean-distance arm can change a verdict. A clause written now for the total-matched arm would be post-result accommodation. Conditions can still be registered in advance for the arms without results (item 6, reopening items 21 and 7).

The 0.03-nat margin is smaller than the 0.071-nat between-run component that 32 rare test targets carry between 36.41 and 32.65 (item 25, reopening item 6). The K = 128 instability is neither recorded in the day's log nor attributed between configuration and code. Meanwhile `base_protocol` reuses the 36.41 replicate trained on the old code next to arms trained on the current tree (item 26). The new options are tied to no prediction or ablation arm (improvement, item 28). The fused kernel's no-spill claim covers K = 20 only, and its CUDA test has no timing bound (item 44). Several README and manuscript statements no longer match the code (item 45).

The dual-measurement, compact-versus-dense and CAVI claims are each pinned by a test that the alternative fails, and are stated within their conditions.

### General sweep: core hot path

The projected-covariance extremes are absolute eigenvalues of a quantity that moves along the free (S_Q, S_K) orbit, so "λ_min ≈ 2e-7" is not by itself evidence of degenerate routing (item 13).

Under `rotation_invariant` at `rope_fraction = 1`, P(G) is diagonal. The scorer still forms G = X Xᵀ, runs two batched Cholesky factorizations over 16,384 blocks of 32 by 32, and evaluates a checkpointed pair GEMM with inner dimension 1,056. A direct-difference diagonal KL grid would remove about 200 GFLOP of IEEE fp32 work per step, together with the expansion's cancellation (improvement, item 30; this is also the numerical analyst's alternative fix for item 2). On the diagonal family the predictive floor guarantees up to d_head times less than its docstring states, because only diag(U C Uᵀ) survives the projection (item 39). The per-layer host read in the scorer can be deferred under this configuration (improvement, item 55). The source-root projection has a latent NaN gradient at zero variance (item 40).

### General sweep: training, configuration, data and entrypoints

The training loop runs its steps in the right order: unscale, divide for accumulation, test finiteness, take the per-role norms, clip, step. It skips non-finite steps without advancing the scheduler, but it has no guard against a divergence that stays finite. Per-step losses reach the host and are discarded except at log rows, and nothing aborts. At 351 ms per step, the remaining 117,500 steps of the live run cost about 11.4 GPU-hours unwatched (item 1, merged with the ML engineer).

If the first log row's diagnostics fail, the nine-key fallback set is latched for the whole run, which drops 52 columns and silences the dead-router warning. `model.diagnostics` also runs outside the precision ladder that `evaluate` uses (item 10, reopening item 16). `canonical_routing_rows` treats a NaN log partition as an empty row and reports zeros instead of NaN (item 38). The table Fisher diagnostics go stale on skipped steps (item 37). Checkpoint resume, the 11 new fields, `grad_clip` wiring, column rectangularity and the decode-decay wiring were found correct.

### General sweep: analysis, visualization, semantics and posthoc

With `diagnose_canonical_free_energy = True`, `free_energy_relationship.png` and `free_energy_decomposition.png` are now drawn on the projected route. They are titled and described as an inner objective "the E-step descends", but nothing on this route optimizes F. On 102.97 the plotted energy rises from 34.8 to 37.0 while validation cross-entropy falls. The self-coupling bar is mostly the regularizer R(α), 24.7 against a divergence of 6.85 (item 17, merged with the philosophy and variational experts).

The new `routing_heads` heatmap draws holonomy for heads with path mass 1.5e-35, and that value, 27.31, sets the color scale (item 22). Paper Figure S1 panel A still labels the displacement ratio as ‖m_i‖/‖μ_i‖ (item 23, reopening item 34). The per-layer canonical F never reaches a figure or a multi-seed aggregate, because one text column withholds the whole channel (item 24, reopening item 19).

The semantics layer and the interventions use two different rules for when frames are inactive (item 46), and `vfe3/posthoc/holonomy.py` keeps a positional CLI (item 47, reopening item 49). No dense-frame assumption remains after PR #90.

## The live K = 128 divergence

Five investigators reached the live run independently. Its folder holds a single row, at step 2500, so every reading below is an inference from that row and from code, not a measurement of the run's history. Three mechanisms are proposed:

| Mechanism | Item | Proposed by | What it explains |
|---|---|---|---|
| Self-locked identity heads | 4 | Transformer-ML | The only identity map a residual-free block has |
| Depth-geometric collapse of covariance scale | 3 | ML engineer | Contraction a trace-relative floor cannot stop |
| TF32 rounding noise in the pair-KL backward | 2 | Numerical analyst | Gradient error once variances are small, which would feed the pre-clip explosion |

Two further findings bear on how the row should be read:
- **The covariance diagnostic is partly free.** The λ_min diagnostic is partly a free gauge coordinate (item 13). On its own it cannot separate genuine collapse from drift along the (S_Q, S_K) orbit.
- **The factorization is not the problem.** Under the active diagonal P(G) the router's Cholesky factorization is numerically inert (item 12 and the numerical analyst's checks).

The guards did not engage, for three reasons:
- the loop has no finite-divergence detection (item 1);
- the dead-router warning checks the wrong signal (item 7);
- a shared clip role lets one exploding group throttle the others without bounding its own Adam step (item 8).

The investigators name experiments that separate the mechanisms, all requiring the GPU and therefore not run:
- gradients of one batch under "high" and "highest" on a saved checkpoint (item 2);
- per-layer projected-covariance scale and per-plane joint map norms on that checkpoint (items 3 and 13);
- a short run with the existing `attention_residual` or `block_norm_covariance = "pushforward"` toggle (items 3 and 4);
- the 36.41 configuration rerun on the current tree for 2,500 steps, to separate code from configuration (item 26).

## Punch list

Findings reported by more than one investigator are merged and credited to each. "(user WIP)" marks defects in uncommitted files, "improvement" marks proposals rather than defects, and "reopens #N" refers to the first punch list of the day.

| # | Severity | Finding | Location | Source |
|---|---|---|---|---|
| 1 | high | No detection or abort for a finite divergence; per-step role norms are computed on device and discarded, and the live K = 128 run reached grad norm 16,847 at step 2500 with no warning | `vfe3/train.py:933-948, 3942-3944, 4033, 4065-4078` | ML engineer, training sweep |
| 2 | high | TF32 backward of the IEEE-forward pair-KL GEMM: O(1) relative gradient error once comparison variance falls below about 1e-2 of the squared spread (emulated; reopens #44) | `vfe3/relational_scoring.py:1131-1167`; `vfe3/runtime.py:49-50` | numerical |
| 3 | high | Residual-free stack with passthrough LayerNorm covariance: covariance scale can contract with depth, and the trace-relative floor cannot bound it; only the last layer's λ_min is logged | `vfe3/model/relational_operators.py:1272-1277`; `vfe3/model/block.py:196-197` | ML engineer |
| 4 | high | A self-locked router is the only identity map of a residual-free block; ALiBi and comparison RoPE favor self and nothing pushes a saturated head back | `vfe3/gradients/relational.py:288-291`; `relational_operators.py:279-281`; `vfe3/attention_prior.py:226-232` | transformer-ML |
| 5 | high | README says the read-table comparison removes the parameter confound; three seeds of the total-matched transformer are 0.224 to 0.230 nats better at 0.23 times the wall time (reopens #1) | `README.md:661, 694-696, 698-718` | philosophy |
| 6 | high | Paper decision rule reads only clauses (a), (b), (c); total-matched and mean-distance arms can change no verdict (reopens #21, #7) | `vfe3/paper/tables.py:512`; `paper_arms.py:491` (user WIP) | philosophy |
| 7 | medium | Dead-router warning checks only self-lock under `row_min` or a tie and only clamped rows otherwise; missed the live row (capped fraction 0.939, self weight 0.379) and names capping where softmax saturates | `vfe3/train.py:1621-1641` | transformer-ML, ML engineer |
| 8 | medium | The catch-all "mu" clip role lets the content maps set one coefficient (6.2e-5) for six groups, and clipping does not bound the Adam step of the exploding group | `vfe3/train.py:309-347, 921-925, 1042-1043` | ML engineer, training sweep |
| 9 | medium | Table Fisher step consumes the shared-role clipped gradient, so its size scales with unrelated groups and its trust radius never engages | `vfe3/table_optim.py:123-132`; `vfe3/train.py:1036-1043` | ML engineer, training sweep |
| 10 | medium | A failed first log-row diagnostic latches the fallback key set for the run, dropping 52 columns and the dead-router warning; diagnostics bypass the precision ladder (reopens #16) | `vfe3/train.py:3972, 3986-3990, 4338-4341` | training sweep |
| 11 | medium | LayerNorm pushforward on `gaussian_full`: the token-dependent radial direction carries only the floor, so pair KLs scale as 1/f (reopens #10) | `vfe3/model/block_mlp.py:192, 335-336, 563-565` | geometer |
| 12 | medium | `rotation_invariant` passes no source tuples: a failed fp32 factorization raises and condition recovery is a silent no-op; inert at the active `rope_fraction = 1`, live below 1 | `vfe3/gradients/relational.py:159-160`; `vfe3/relational_scoring.py:763-767` | geometer, numerical |
| 13 | medium | Projected-covariance λ_min, condition and GEMM-error diagnostics move along an unfixed (S_Q, S_K) commutant orbit that leaves every cost invariant; gradients scale as 1/c along it | `vfe3/model/model.py:1767-1776`; `vfe3/gradients/relational.py:118-173`; `vfe3/train.py:347` | info-geometer, core sweep |
| 14 | medium | Inverse measurement uses `inv_ex(O')`, finite and wrong at trained condition numbers (relative error 1.0 at 1e12); inverse-free and factor forms exist | `vfe3/gradients/full_link.py:349-371` | info-geometer, variational |
| 15 | medium | EM template term pairs posterior responsibilities with block-input moments: not an M-step under exact inference or several sweeps, fixed point at V + c | `vfe3/model/block_mlp.py:1129-1131, 1163-1168` | variational |
| 16 | medium | Per-coordinate self-coupling is frame-dependent for full and block families; the pure-path facet has no flag and the witness cannot detect it | `vfe3/run_artifacts.py:5090-5115`; `vfe3/families/gaussian.py:150-157`; `vfe3/viz/extract.py:753-761` | gauge-theorist |
| 17 | medium | Canonical F drawn as an inner objective the E-step descends; the self-coupling bar is mostly R(α); `free_energy_codescent` alias | `vfe3/viz/figures.py:1513-1558, 1619-1629`; `vfe3/run_artifacts.py:5444-5458` | philosophy, variational, analysis sweep |
| 18 | medium | Canonical β* peaks on the diagonal in every row and omits comparison RoPE, so `canonical_beta_tv` shrinks as the model self-locks and mixes position with the maps | `vfe3/free_energy.py:604-614`; `vfe3/model/model.py:710-711, 1882-1906` | transformer-ML, info-geometer, gauge-theorist, implementation, training sweep |
| 19 | medium | `no_rope` arm also removes the rotation-invariant projection, mislabeled as removing only the rotation (reopens #5) | `vfe3/run_artifacts.py:4432-4451, 4483` | implementation |
| 20 | medium | Under the table Fisher mode, `m_p_mu_lr` and `m_p_sigma_lr` set no step size but are logged, printed and swept as live | `vfe3/table_optim.py:74-78, 142-150`; `vfe3/config.py:1990-1991` | implementation |
| 21 | medium | `decode_logits_amp` default flipped to True with no historical pin; 34.48 reloads with bf16 linear logits (user WIP; reopens #25) | `vfe3/config.py:250, 2648-2655`; `vfe3/model/model.py:386` | implementation, analysis sweep |
| 22 | medium | `routing_heads` heatmap plots holonomy of heads with path mass 1.5e-35, whose value sets the color scale | `vfe3/viz/figures.py:3270-3322` | analysis sweep |
| 23 | medium | Paper Figure S1 panel A labels the displacement ratio as ‖m_i‖/‖μ_i‖ (reopens #34) | `vfe3/paper/figures.py:628-629` | analysis sweep |
| 24 | medium | Per-layer canonical F never reaches a figure or multi-seed aggregate; one text column withholds the channel (reopens #19) | `multiseed_analysis.py:843-853, 1000-1003`; `vfe3/viz/figures.py:3821-3848` | analysis sweep |
| 25 | medium | 0.03-nat margin is smaller than the 0.071-nat between-run component from 32 rare test targets (reopens #6) | `vfe3/paper/tables.py:384-417, 430` | philosophy |
| 26 | medium | K = 128 instability unrecorded and unattributed; `base_protocol` reuses an old-code replicate beside current-tree arms | `vfe3/paper/tables.py:458-467`; `paper_arms.py:111-116, 1013-1018` | philosophy |
| 27 | medium | Improvement: evaluate canonical F also at the readout output and log ΔF per layer | `vfe3/model/model.py:1877`; `vfe3/train.py:3068-3070` | variational |
| 28 | medium | Improvement: tie each new option to a stated prediction and one single-change ablation arm | `ablation.py:546` (user WIP); `paper_arms.py:343-460` | philosophy |
| 29 | medium | Improvement: exclude the block MLP from bf16 autocast; its residual is the only reduced-precision router input (0.58 to 5.1 nat median gap shifts in emulation) | `vfe3/model/block_mlp.py:515, 559-561`; `vfe3/model/model.py:748` | numerical |
| 30 | medium | Improvement: direct-difference diagonal KL for `rotation_invariant` at `rope_fraction = 1` (about 200 GFLOP per step, no expansion cancellation) | `vfe3/geometry/rope.py:150-156`; `vfe3/relational_scoring.py:319, 939, 951, 1162-1165` | core sweep, numerical |
| 31 | low | `inactive_fields` marks the canonical diagnostic's inputs (`lambda_alpha*`, `b0`, `c0`, `lambda_beta`) inert; minimal-agent diagnostic uses τ = κ√d where the router uses 1 | `vfe3/config.py:2068-2072, 2133-2135, 2270-2271` | variational, gauge-theorist |
| 32 | low | `inactive_fields` Gram/factor certificate status ignores `rotation_invariant` | `vfe3/config.py:2028-2035` | transformer-ML |
| 33 | low | Pure-path report gives both RoPE covariance modes one label and never records the toggle | `vfe3/run_artifacts.py:5161` | gauge-theorist |
| 34 | low | Table Fisher step runs before the frame step, so `fisher_group` reads updated rows | `vfe3/table_optim.py:164-170`; `vfe3/gauge_optim.py:1310, 1351` | geometer |
| 35 | low | Table Fisher docstring calls the prior Fisher "the metric of the step's objective" | `vfe3/table_optim.py:4-12`; `vfe3/config.py:656-657` | geometer, info-geometer |
| 36 | low | "KL ≤ r²/2" holds to second order only (1.049 at r = 0.1); decay sits outside the bound | `vfe3/config.py:663`; `vfe3/table_optim.py:15-18, 140-150` | info-geometer, ML engineer |
| 37 | low | Table Fisher diagnostics stale on skipped steps; step norm reported before the schedule factor; `lr_mu` shows the AdamW rate | `vfe3/train.py:3950-3957`; `vfe3/table_optim.py:152-162` | ML engineer, training sweep |
| 38 | low | `canonical_routing_rows` reports zeros, not NaN, on rows with a failed factor | `vfe3/free_energy.py:595-596, 611-619` | training sweep |
| 39 | low | Diagonal-family predictive floor guarantees up to d_head times less than its docstring | `vfe3/model/relational_operators.py:1011-1013, 1181-1189` | core sweep |
| 40 | low | Source-root projection has an infinite or NaN gradient at zero variance (latent) | `vfe3/relational_scoring.py:377-390` | core sweep, numerical |
| 41 | low | Deferred chart verdict lets CUDA `matrix_exp` run on infinite blocks (possible endless squaring loop) | `vfe3/geometry/transport.py:1121-1139, 1156-1157` | numerical |
| 42 | low | `expected_likelihood_kernel = "triton"` silently runs torch on CPU; no artifact records the kernel; absent from `PRECISION_KEYS` | `vfe3/model/prior_bank.py:1992-2001`; `vfe3/paper/tables.py:80-83` | implementation |
| 43 | low | `decode_mu_weight_decay` described as a Gaussian prior, which decoupled decay is not; effect on rare rows unmeasured | `vfe3/config.py:695-699` | ML engineer |
| 44 | low | Triton no-spill claim covers K = 20 only; CUDA test runs at a shape where the spill cannot occur and has no timing bound | `docs/2026-10-10.md:41`; `tests/test_expected_likelihood_kernel_20261010.py:213, 268-309` | philosophy |
| 45 | low | Doc drift: README equivalence condition omits the pushforward, calls "absolute" the default, omits the inverse measurement, says the row minimum is detached; config comment credits a cancelling log-det; launcher comment "prevents self-lock" (user WIP); manuscript says no mean-only control exists | `README.md:495, 504-514, 599-605`; `vfe3/config.py:359-360`; `train_vfe3.py:235` | philosophy, transformer-ML |
| 46 | low | Semantics and interventions use different "frames inactive" rules | `vfe3/semantics/run.py:613`; `vfe3/posthoc/interventions.py:135-163` | analysis sweep |
| 47 | low | `vfe3/posthoc/holonomy.py` keeps a positional argparse CLI (reopens #49) | `vfe3/posthoc/holonomy.py:274-305` | analysis sweep |
| 48 | low | Improvement: warn that `rotation_invariant` with `rope_fraction = 1` makes the router covariance diagonal, and count slow planes | `vfe3/geometry/rope.py:150-156` | transformer-ML |
| 49 | low | Improvement: covariance pushforward at the final LayerNorm | `vfe3/model/model.py:1127-1137` | geometer, implementation |
| 50 | low | Improvement: O(K²) rank-two form of the full-layout LayerNorm pushforward (agrees to 5.9e-16) | `vfe3/model/block_mlp.py:335-336, 458-472` | geometer |
| 51 | low | Improvement: covariant covariance pushforward for reference-coordinate blocks (reopens #54) | `vfe3/model/block.py:273-292` | gauge-theorist |
| 52 | low | Improvement: assert exact off-block zeros in `_act_on_frame` | `vfe3/viz/extract.py:675` | gauge-theorist |
| 53 | low | Improvement: per-head float64 exponentials in the canonical diagnostic instead of dense K by K | `vfe3/model/model.py:1882` | gauge-theorist |
| 54 | low | Improvement: prototype label-staleness diagnostic | `vfe3/model/block_mlp.py:1191-1204` | variational |
| 55 | low | Improvement: defer the scorer's raise-only per-layer host read under this configuration | `vfe3/relational_scoring.py:971-978` | core sweep |
| 56 | low | Improvement: two-level accumulation over k in the Triton kernel (6.9e-3 per row at K = 512) | `vfe3/kernels/_expected_likelihood_triton.py:32-39` | numerical |
| 57 | low | Improvement: remove host syncs on the opt-in pushforward and table Fisher paths | `vfe3/model/block_mlp.py:251-252`; `vfe3/table_optim.py:69-71, 110, 118` | numerical, training sweep |
| 58 | low | Improvement: replay non-finite skipped steps at the next precision rung | `vfe3/train.py:3458-3471` | ML engineer |

Counts: 6 high, 24 medium (20 defects and 4 improvements) and 28 low (17 defects and 11 improvements), 58 in all. The suggested order for the user is item 1, item 2's GPU discrimination test, item 3's per-layer diagnostics, items 7 to 9, then item 21 before any reuse of the 34.48 base.

## Test results

No tests were run. The user asked that no CPU or GPU tests be run for this audit, and every investigator was held to that. For context only: the last recorded run of the CPU lane, on the working tree that became `b65b6df` and before this audit, passed 4,449 tests and skipped 3, with no failures. The GPU lane has not been run on the PR #91 code, so the fused kernel's speed is still unmeasured.

## Adjudication

No adjudication ran. The natural candidates for the `adjudicate` skill are items 2, 3 and 4, the three proposed mechanisms of the live divergence. Their evidence is emulation and inference from one metrics row, and items 3 and 13 read the same λ_min in incompatible ways. Item 15 is a derivation whose numerical confirmation was a numpy reimplementation, not the code, and item 5 is a reading of a README claim against recorded summaries.
