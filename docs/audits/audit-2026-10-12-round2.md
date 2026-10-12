# Deep audit, October 12, 2026 (round 2)

## Scope

The second audit of the day covered the tree at `main` = `822f3d1`, the merge of PR #98. Since the first audit (`docs/audits/audit-2026-10-12.md`, taken at `6b872b4`), 37 commits in PRs #93 to #98 changed 223 files (+8,729 / −7,679 lines). They contain:
- the remediation of every round-1 defect;
- the new options T1 (`diagnose_agent_elbo`), T2 (`predictive_covariance = "spread_corrected"`), T3 (`decode_second_order`) and T6 (`relational_routing = "noisy_key"` and `channel_noise_mstep`);
- `relational_receiver_update = "closed_form_profiled"`;
- the prior-anchored predictive floor, the LayerNorm pushforward trace cap and `phi_frame_condition_max`;
- two configuration-removal passes;
- a live-route test profile;
- eleven new and seven repaired ablation sweeps.

That diff was the first priority of every investigator. The theory reference was again `GLK_attention_v2.tex`.

The active configuration is the `train_vfe3.py` dictionary at this commit. It is the round-1 configuration less the fields removed since:
- K = 20 with two heads of 10, N = 128, four layers;
- `gaussian_diagonal` with the linear decoder;
- `projected_predictive` with the minimal-agent readout (mean field) and the prototype-agent block MLP (mean field);
- moment-matched covariance with floor 1e-4, the exponential chart, the causal prior;
- block_glk per-head frames with learned positional frames, AdamW, `amp_dtype = "bf16"`.

`VFE3Config` now has 162 fields. `inactive_fields()` marks 61 of them inert on the launcher, including 34 of the launcher's 120 keys.

## Method

Seven investigators ran in parallel on disjoint areas:

| Investigator | Area |
|---|---|
| agents | T1, T2, the prior-anchored floor, the T6 channel-noise M-step, the deferred prototype and minimal-agent checks |
| routing | T6 noisy-key routing, T3, `closed_form_profiled`, the inverse receiver measurement and its historical pin |
| geometry | `phi_frame_condition_max`, the reference-change and site-covariance reruns, the trace cap, the geometry diff |
| config | migrations of retired keys, default pins, fingerprints, stale readers, the sweeps, round-2 hardcode candidates |
| training | the round-1 training fixes, resume exactness under the per-epoch origin, deferred-check timing, new logging |
| model | old-tree against new-tree regression comparisons, causality of every new option, hot-path performance |
| hygiene | the analysis fixes, analysis code against the removals, new artifacts, dead code, tests and docs |

All investigators ran repository code on CPU torch 2.14.1 in a session scratch environment. Old payloads and old forwards came from copies of the `6b872b4`, `a74fc8e` and `389413b` trees extracted into the scratchpad; one investigator used a temporary git worktree, removed afterwards. Independent oracles included numpy and mpmath reimplementations, Gauss-Hermite quadrature, exact enumeration, dense Hessian references and Monte Carlo.

The coordinator checked the following against the source:
- the M-step diagnostic slowdown;
- the trace cap's detached bound;
- the unconditional refusal of constant α;
- the live-route profile;
- the ablation baseline's missing measurement key;
- the terminal-callback ordering;
- the noisy-key posthoc call;
- the strict config loaders in the multi-seed driver and figure worker.

No GPU was used, and no source file was edited.

## Round-1 remediation: status

| Round-1 item | Status | Evidence |
|---|---|---|
| 1, LayerNorm pushforward explodes | fixed. The trace cap keeps loss and every gradient finite at `sigma_init` 15 and 1 across ten route and family variants; with the cap disabled the same models go NaN. New defect in the cap's gradient (item 3 below). | run |
| 2, prototype diagonal path never flags non-finite values | fixed for the mean field; small gaps remain (item 9) | run |
| 3, frame conditioning unbounded | fixed by `phi_frame_condition_max`, details below; opt-in, None on the launcher | run |
| 4, floor breaks reference-change invariance | fixed. A model-level reference change by a non-orthogonal a moves loss, logits, μ and Σ by 2e-15 to 6e-15 relative at cond(a) 11 to 20, on the live, kernel, exact, `spread_corrected`, `gaussian_full` and `noisy_key` routes | run |
| 5, 6, baseline data and clip scope | fixed. Mismatched dataset, tokenizer, data seed or token cap are refused; `grad_clip_scope` is matched | run |
| 7, stale scored-target key | fixed. Both finalize writers emit the counts and the reproduction check compares them | run |
| 8, 2·SD noise floor | fixed. The floor is t(n−1, 0.975)·√2·s, used consistently in CSV, Markdown and console; it now needs scipy (item 16) | run |
| 11 to 16 | fixed | run |
| 17, readers of never-written keys | partly done (item 13) | run |
| 21, label-step duplicate work | partly done (item 15) | run |
| 29, finalize contract test | present, but covers few readers (item 13) | code trace |
| C11, live-route test profile | added, but it is not the launcher route (item 5) | code trace |

## What was verified correct

- **New mathematics.**
  - T1: the ELBO matches a closed-form oracle built from its definition to 1.4e-14 over 60 instances. It never exceeds the exact evidence, never decreases over sweeps, is causal, and leaves loss, gradients and RNG state bitwise unchanged.
  - T2: the implementation matches an independent re-derivation to 1.8e-10, passes float64 gradcheck, and is invariant under a non-orthogonal reference change. With two coupled heads it lowers the KL to the exact label-tuple posterior from a median of 0.046 to 0.021.
  - T6 M-step: its optimum matches a brute-force maximization of the ELBO over n_h to 2.7e-7 relative.
  - T6 noisy-key routing:
    - It matches a dense oracle through the whole readout to 3e-16, with and without the kl_max clamp.
    - The noise sits on the key covariance only, and the temperature is exactly 1 on every path.
    - It passes a gradient check against the log-noise and gives each head its own noise with mixed head sizes.
    - Its parameter is in the relational-score group with no weight decay.
  - T3: the chunked kernel matches a dense Hessian reference exactly in float64 at chunk widths 1 to 100, including a chunk that dominates the softmax. That exercises the online renormalization across chunks. Gradients agree to 1e-15, and ignored targets get zero covariance gradient.
  - `closed_form_profiled` matches 3,000 iterated closed-form steps to 7e-12 (diagonal) and 6e-11 (full). Its implicit gradients match central differences to about 1e-9, and on a 6,000-curve stress test its root agrees with bisection to 1.1e-13.
- **Inverse measurement and pin.** The profiled receiver output is invariant under (Uh, Th, Sh, h⁻¹O') to 5e-13 on all three charts, where the adjoint moves by 0.9 to 2.3. Saved configs that lack the measurement key load as adjoint and give bitwise-identical float64 loss.
- **Frame bound.**
  - The logarithmic-norm bound and the per-factor budget hold, and the window shift is the exact Frobenius-nearest projection.
  - The √2 screen bound holds for any spectrum with d ≥ 2.
  - After projection the composed condition number over all tokens, positions and heads is 0.54 to 0.97 × κ_max, where it was 5e3 to 1.2e6 × before.
  - Skew and trace parts change only by rounding, and rows that pass the screen are bitwise unchanged.
  - Twenty real AdamW steps keep the bound, and checkpoint resume is bitwise.
- **Training.**
  - Resume under the hardcoded per-epoch window origin is bitwise exact in parameters and every step's loss. This covers epoch-end and mid-epoch checkpoints, shuffled loaders with a generator, sequential loaders on the global RNG, dropout, and a binding frame bound.
  - The deferred host checks are drained before backward, so a failure raises before the optimizer, the scheduler or the frame bound can act. A transient failure at "high" precision is replayed at "highest".
  - The frame bound runs only on accepted steps, under no_grad.
- **Regression comparison.** On the same weights and batch, old and new trees agree wherever the log says they should:
  - Live and kernel routes: bitwise once the floor is set to 1e-30; the remaining difference is the T5 floor change.
  - Exact inference, all chunked decodes, and `gaussian_full` with the fixed full-covariance KL: bitwise.
  - Migrated `family_chunked` configs: agree to rounding.
  - bf16 against fp32: bitwise.
- **Causality.** Perturbing a later token leaves every earlier position's logits, the masked loss and every parameter gradient bitwise unchanged. This held for `noisy_key`, `spread_corrected`, `decode_second_order`, `closed_form_profiled`, `diagnose_agent_elbo` and `phi_frame_condition_max`.
- **Migrations.**
  - All 26 removed fields are in `RETIRED_CONFIG_KEYS`, and 40 migration cases behave as specified, apart from item 2.
  - A payload without the measurement key gets "adjoint", and one without `kl_max` gets 100. A scan of the 56 commits in the clone found no other unpinned default flip.
  - The `6b872b4` and `a74fc8e` launcher payloads migrate to exactly the new launcher configuration, with the same fingerprint.
  - `test_config_isolation` covers all 162 fields.

## Findings

Severity applies to the effect on results under the configuration where the defect is live. "Active" means it is live on the launcher.

### Defects

1. **Medium.** Finished ablation and scaling cells are no longer reused.
   - **Cause:** the default of `relational_receiver_measurement` changed from adjoint to inverse (`vfe3/config.py:563`). `train_vfe3.py:141` sets the key, but `ablation.BASELINE_CONFIG` (`ablation.py:288-509`) and the `scaling.py` configuration do not.
   - **Effect on reuse:** a cell saved before the change carries "adjoint", so its migrated config differs from the new one in that field, and the reuse contract rejects it. The field is inert on these routes. The real `ablation._schema_migrated_contract` rejected cells from both `6b872b4` and `a74fc8e`; changing only that field made them reusable. `scaling.py:886-893` behaves the same way.
   - **Effect on new arms:** the baseline's projected_gradient arms now run the inverse measurement while the launcher pins adjoint. These include the damping, `sigma_max`, `e_q_*` and receiver-noise sweeps, and the arm at `parameter_optimization.py:333`. On one payload the loss moved from 2.37836 to 2.37731.
   - **Fix:** set the key in both baselines, together with item 4 (run).
2. **Medium.** Runs from October 4 to 7 no longer load. `config_from_serialized` refuses `lambda_alpha_mode = "constant"` unconditionally (`vfe3/config.py:3003-3006`). The committed launcher ran constant α on projected_predictive with `mstep_self_coupling_weight = 0` until October 7, where the old build marked α inert. Those payloads (for example the one at `389413b`) are refused by both the strict and the lenient loader, so the paper, posthoc, multi-seed and figure loaders cannot open them. This breaks the removal pass's own rule that inert values load silently. Refuse only where α was read, and otherwise rewrite to a state-dependent form (run, verified in source).
3. **Medium-low (opt-in).** The trace cap's gradient with respect to gamma is wrong whenever the cap binds.
   - **Cause:** the bound is `K * weight.detach().square().amax()` (`vfe3/model/block_mlp.py:492`). When the cap binds, the forward value is K·max(γ²)·P/tr P, but backward treats the bound as a constant.
   - **Measured:** a finite-difference check at K = 20 and Σ = 15 found autograd missing 41.9 of 42.8 on the largest-γ coordinate. The predicted missing term matched to 1e-11.
   - **Scope:** under `block_norm_covariance = "pushforward"` the cap binds on every token at the launcher's initialization scale.
   - **Fix:** drop `.detach()` (run, verified in source).
4. **Medium-low.** Old fixed-window cells are reused silently. `train_window_offset` is dropped on load (`vfe3/config.py:2587`), and every cell saved before October 12 ran with "fixed". Once item 1 is fixed, `ablation._schema_migrated_contract` (`ablation.py:3197`) and `scaling._cell_resume_status` (`scaling.py:870-905`) reuse those cells beside new ones. New runs draw a per-epoch origin from the loader generator before the sampler shuffles, so the same data seed gives different windows and a different batch order. A sweep then mixes unpaired arms, which acts like a reseed. Resuming a fixed-window checkpoint is refused with an uninformative `data identity mismatch for field(s) ['iterator']`. Treat a raw saved "fixed" as non-matching in both reuse paths (run).
5. **Medium-low.** The C11 "live-route" test profile is not the launcher route.
   - **Profile:** `tests/profiles.py:212-270` uses the kernel readout, the coordinate MLP, `causal_alibi`, `row_min`, an attention residual and no amp. Its comment names it the launcher's route.
   - **Launcher:** the minimal agent, prototype agents, the causal prior, the absolute clamp, no residual, bf16.
   - **Why tests miss it:** `test_live_profile_is_the_launcher_route` compares the profile with itself, and only two test files build from it.
   - **Fix:** pointed at the real launcher route, all four live-route checks pass, so the fix is a drop-in (run, verified in source).
6. **Medium (performance).** A training step with an M-step capture runs every minimal-agent diagnostic.
   - **Cause:** when `block_mlp_em_weight`, `mstep_self_coupling_weight` or `cg_energy_weight` is above 0, `model.py:1203-1207` creates the M-step capture. `model.py:1065` then creates the relational capture, and `relational_operators.py:926` passes `capture is not None` as the diagnostics flag.
   - **Wasted work, each step:** the float64 staleness relabel, the T6 channel-noise M-step (a float64 eigendecomposition, a 64-point grid and 40 Newton steps), coupling and contraction. With `diagnose_agent_elbo` on, it adds the ELBO and a float64 exact readout.
   - **Measured** on the live route at K = 20 with EM weight 0.1:
     - operations: 8,577 per step, against 4,037 with diagnostics forced off;
     - time: 136 ms, against about 90 ms; 252 ms with the ELBO on;
     - loss: bitwise identical.
   - **Affected:** the new `block_mlp_em_weight` sweep and every self-coupling run. The log's "about 1.5%" ELBO cost holds only when these weights are 0.
   - **Fix:** create the relational capture only for diagnostic captures.

   Two investigators found this independently (run, verified in source).
7. **Medium.** Ablation cells record no accepted or skipped update counts. `train()` calls `terminal_callback` at `vfe3/train.py:5105`, before `artifacts.realized_updates` is set at `:5173`. `finalize_validation_run` (`vfe3/run_artifacts.py:4904`) never publishes `diagnostics_failures` either. In a run, a cell that rejected 5 of 6 updates had no update, skip or failure key in `summary.json`; only a NaN final training loss hinted at it. With `divergence_abort` removed, these counts are the only record of a diverged cell, and arms with different skip rates train on different learning-rate schedules. Build the realized block before the callback, as `numerics_counters` already is (`:5076`) (run, verified in source).
8. **Medium (inactive at `prior_handoff_sigma = 0`).** The prior-anchored floor reads a layer prior that plain forwards never hand off.
   - **Cause:** the stack skips the prior handoff on any predictive stack without a capture (`vfe3/model/stack.py:34-52`, `_can_omit_prior_handoff`), on the old assumption that predictive content never reads the prior. The floor now does.
   - **Effect with `prior_handoff_sigma > 0`:** training and evaluation forwards floor every layer with the embedding prior. Captured forwards floor with the handed-off prior; captures happen on diagnostic forwards, and in training whenever an M-step weight is above 0. The config also marks `prior_handoff_sigma` inert (`vfe3/config.py:2329-2336`).
   - **Measured** on the live route at `prior_handoff_sigma = 0.5`: with floor 0.3, the final covariance differs by up to 0.47 (minimal agent) and 7.4 (kernel); at the live floor of 1e-4 with three layers, logits differ by 6.3e-5. At 0, as on the launcher, the two forwards are identical.
   - **Fix:** keep the handoff when `prior_handoff_sigma > 0` and the readout returns a covariance, and narrow the inert rule (run).
9. **Low.** Prototype failure flags still have gaps (`vfe3/model/prototype_agents.py:267, 636`). The exact diagonal tile tests only W > 0, so an infinite variance or NaN mean returns non-finite output with no flag. On both paths, a slightly negative working variance with 1 + cD > 0 returns C⁺ < 0, although the error message claims a positive-semidefiniteness check. The training-step finite gate still prevents a corrupted update (run).
10. **Low-medium.** Old runs with a removed value cannot be aggregated or re-plotted. `multiseed_analysis.py:100-107` and `vfe3/viz/figure_worker.py:143-149` load configs strictly, and neither replays a forward. A run that saved any of the following is refused:
    - `norm_type_final = "layernorm"`;
    - constant α;
    - the ambient law;
    - a non-causal prior;
    - `decode_bias = False`.

    `vfe3.paper` falls back to raw values, and these two should do the same (run).
11. **Low.** Two posthoc intervention rows raise a TypeError under `noisy_key` routing. `vfe3/posthoc/interventions.py:362` and `:630` call the routing cost without `key_noise`, so the report records the row as failed rather than not applicable (run, verified in source).
12. **Low.** Two logged values misdescribe themselves:
    - With `decode_second_order` on, `train_ce` and `train_ppl` include the second-order term but are logged as plain cross-entropy (`vfe3/train.py:834, 1092, 4821-4822`). Validation stays plug-in, so the train-to-validation gap is biased.
    - Under `spread_corrected`, the logged ELBO uses N(ξ⁺, P), not the covariance the readout returns. The `..._evidence` metric is an assumed-density value for coupled heads: median error 1.4e-3 nats against exact enumeration, maximum 0.10, and the gap went negative for 1 of 720 receivers.

    The test at `tests/test_t1_t2_minimal_agent_elbo_20261012.py:338` asserts a non-negative gap with coupled heads, which is not guaranteed (run).
13. **Low.** Several artifact readers still read keys nothing writes, and the contract test misses most readers.
    - **Never-written diagnostics:** 30 of the 45 diagnostic keys copied at `vfe3/train.py:4895-4913` are produced by `model.diagnostics` on none of six routes. The geometry-health figure reads 18 of them (`vfe3/run_artifacts.py:5566`), and `CURVE_SPECS` reads `fisher_trace_mean` (`multiseed_analysis.py:991`).
    - **Config schema key:** the paper table reads `__vfe3_config_schema__` (`vfe3/paper/tables.py:816`), which nothing writes.
    - **Contract test:** the item-29 test calls `train()` without log or eval intervals, so no metrics or diagnostics are written. It also stubs the figures, uses the legacy profile and checks 6 of 19 scalar keys. Run with intervals, it would have caught this item (run).
14. **Low.** The frame bound leaves no trace in the artifacts. `vfe3/train.py:1218` discards the projected-row count, so the `frame_projection_*` columns stay blank and paper panel S1-D shows nothing. The pure-path flag `no_phi_mstep_chart_cap` (`vfe3/run_artifacts.py:5188`) stays True while the bound rewrites frames. The paper Methods table omits `phi_frame_condition_max` and `decode_second_order`, both of which change training. With `pos_phi = "frozen"` the bound is accepted but does not cover the frozen factor, so the composed condition reaches 12.7 × κ_max at N = 128 (`vfe3/gauge_optim.py:1195`). The float32 projection is not quite idempotent at the boundary: about ten re-projections, with drift near 2e-6 (run).
15. **Low-medium (performance, active).** Round-1 item 21 is only partly done.
    - **Done:** the symmetry check and the query log-determinant autograd were dropped. Operations fell from 999 to 973 forward and from 1,206 to 1,165 backward.
    - **Still open:** each layer still runs 11 Cholesky factorizations. B_j is factored three times (`vfe3/model/minimal_agent.py:478, 1174` and inside the KL), and V_Q two or three times.
    - **New:** `label_spread` factors B a fourth time under `spread_corrected` (run).
16. **Low.** Miscellaneous items found by one investigator each:
    - The multi-seed noise floor imports scipy (`multiseed_analysis.py:223`), which is only in the `viz` extra, so aggregation aborts without it. `flag_noise_dominated` is still never called by the driver.
    - A training stream now needs at least 2N tokens (`vfe3/data/datasets.py:797-806`), so a smoke run with a small `MAX_TOKENS` fails inside `train()`.
    - `pos_loss_ratio` disagrees with last_q / first_q when `val_diagnostic_batches > 1` and `eval_stride < N` (ratio 1.146 against 1.603; `vfe3/train.py:3283-3286`).
    - The "auto" decode checkpoint undercounts second-order memory by about 2× (`vfe3/model/prior_bank.py:1843`).
    - The profiled root solve's residual is captured but never read, and its slope floor of 1e-6 (`vfe3/gradients/full_link.py:896`) can amplify gradients up to 1e6 near a fold, uncounted.
    - The canonical free-energy diagnostic compares the noise-broadened router with a noise-free canonical router under `noisy_key` (`vfe3/model/model.py:2118-2123`).
    - `RelationalOperators` defaults `receiver_measurement` to "adjoint" (`vfe3/model/relational_operators.py:395`), unlike the config.
    - Three commented-out sweeps (`rope_base`, `decode_tau`, `lambda_beta`) fail the validator on both baselines, so uncommenting one aborts the session at startup.
    - Several sweep `requires` entries are stricter than needed, so no arm reproduces the baseline run.
    - C12 is partly done: `warmup_steps > max_steps` is still accepted.
    - C7 is incomplete in `paper_arms.py:216, 223, 610`, which read `sigma_weight_decay == 0` directly.
    - `VFEModel.__init__` turns off the NaN fill of uninitialized memory for the whole process, so the deterministic test lane no longer detects uninitialized reads.
17. **Low (docs).**
    - `README.md:287` still describes the old floor f·tr(C)/K·I, and manuscript lines 1164 and 1587 do too. Old checkpoints replay with the new floor, and nothing records this.
    - The README never mentions T1, T2, T3, T6, `phi_frame_condition_max`, `projected_factorization` or `frame_pair_escalation_tol`, or the prototype-agent route.
    - `README.md:545-548` says the linear decoder discards the covariance, which second-order training contradicts.
    - The launcher comments at `train_vfe3.py:128, 131` still list `laplace_diagonal` and `family_chunked`, and `ablation.py:330` describes the old floor.

### Performance and precision

18. **Medium (active).** `amp_dtype` does nothing on any route tested. Under CPU bf16 autocast, no operation ran in reduced precision, forward or backward, on these routes:
    - the live route;
    - the kernel readout with every MLP mode;
    - `delta_full` and the Mahalanobis norm;
    - reference coordinates and projected_gradient;
    - every chunked decode.

    bf16 and fp32 loss and gradients were bitwise equal. The only remaining consumer is `decode_logits_amp`, and the inert rule at `vfe3/config.py:2152` covers only the minimal agent. Set `amp_dtype = None` on the launcher, or fold the field into `decode_logits_amp`. Not checked on CUDA (run).
19. **Low (active).** A silent training step on the live route makes seven host synchronizations:
    - four per-layer `host_read` calls;
    - one `torch.unique` (`vfe3/geometry/transport.py:1154`);
    - the deferred-scope close (`vfe3/train.py:820`);
    - the finite gate (`:1007`).

    `matrix_exp` is called four times per step (two forward at `transport.py:1163`, two in backward). On CUDA, torch reads its scaling exponents on the host. Stacking the two forward calls halves those reads. Because `transport_chart_max_norm` bounds the norm, a fixed-schedule scale-and-square with about five masked squarings needs no host read at all. `minimal_agent_elbo` uses `torch.linalg.cholesky` (`vfe3/model/minimal_agent.py:978`) where `cholesky_ex` would avoid a sync. The frame bound adds two host reads per accepted step plus one per flagged chunk, where its docstring says one per table (run).

### Configuration: round-2 hardcode candidates

The user's recorded decisions to keep C2 (latent links), C3 (CG coupling), C5 (Euler receiver) and C9 (template initializations) were respected. Candidates are ranked by payoff times confidence. Line counts are approximate.

| # | Candidate | Proposal | Source / test lines | Checkpoint handling | Payoff × confidence |
|---|---|---|---|---|---|
| H1 | explicit `kl_max = 8K` in `train_vfe3.py`, `ablation.py:509` (and its grid special case at 1238-1244), `scaling.py:387, 412-417` and per-cell overrides | delete; None already resolves to 8K | 35 / 0 | none; resolved value saved, same fingerprints | high × high |
| H2 | `amp_dtype` (item 18) | set None on the launcher, or fold into `decode_logits_amp` | ~30 / some | drop when inert | high × medium-high |
| H3 | family-consistency machinery left by C1 (`vfe3/config.py:1641-1648, 2260-2263`, the prior-bank registration flag, `vfe3/families/base.py:127`) | delete | 45 / 0 | none | medium × high |
| H4 | `decode_av_precision` | fix at fp32 | 25 / 27 | drop; rounding only | medium × medium-high |
| H5 | `predictive_covariance_checkpoint` and `decode_ce_checkpoint` | one automatic memory policy | 40 / 30 | drop; bit-identical | medium × medium |
| H6 | `t5_relative_bias` and its three fields | retire; unused | 140 / 7 | refuse the prior | medium × medium |
| H7 | `mu_trust_cholesky_rounds` | constant 3 | 12 / 5 | drop | low-medium × high |
| H8 | `decode_ranking_fp64_escalation` | always on; evaluation only | 10 / 21 | drop | low-medium × medium |
| H9 | `transport_chart_max_norm` | constant 50 | 35 / 29 | drop; it only decides refuse against pass | medium × low-medium |
| H10 | `per_type_hidden_bias` and its rate | remove after its paper arm reports | 35 / 15 | refuse True | medium × low-medium |
| H11 | `mass_phi` | retire | 10 / 9 | refuse > 0 | low × medium |
| H12 | `decode_unigram_prior` | merge into `unigram_kappa` (0 = off) | 40 / 28 | keep the buffer | low × low-medium |
| H13 | `routing_kl_max_reference` | always `row_min` | 15 / 20 | refuse absolute where the clamp bound | low × low |

Smaller removal leftovers can go with H3:
- the projected-scoring setters for recovery precision and tolerances, which now only ever receive constants (the module default is still fp64 for callers that never build a model, so scorer-primitive tests run a policy production never uses);
- the BCH-gate argument threaded through `gauge_optim.py` although `train.py:450` fixes it at 1e-6;
- an unreachable branch at `vfe3/posthoc/maps.py:217`;
- `families/base.py:54` `expm1_neg_plus`, a Laplace leftover.

### Theoretical and mathematical improvements

All of these refine the options added since round 1; none is a defect.

**M1. Name and fix the evidence for coupled heads.**
- With head coupling zero, report the exact evidence.
- Otherwise, name the value `adf_evidence`, or add an importance-sampled estimate under q, which gives an unbiased gap.
- Under `spread_corrected`, also log the ELBO at the covariance the readout actually returns.

**M2. The channel-noise M-step is uninformative at the launcher's initialization.** In the hygiene investigator's real run, `channel_noise_mstep` was 0.0 on every head and layer: the optimum sits at the n → 0 bracket at `sigma_init = 15`. The round-1 theory note explains why the declared record's likelihood is unbounded as n → 0. Report the bracket hit explicitly, and log n* only once the record term has moved off that boundary. Otherwise the column reads as a learned value.

**M3. A tighter valid LayerNorm trace cap.** K·λ_max(P Γ² P), with P = I − 11ᵀ/K the centering projector, is still a valid bound. It is up to 2× tighter than K·max γ² when γ is non-uniform (Monte Carlo trace against the current cap: 21.8 against 45).

**M4. Share the condition budget between factors adaptively.** The frame bound splits log κ_max evenly between the token and positional factors. Positional rows rarely use their half, so giving the slack to the token factor bounds the same composed condition with less distortion. Applying the bound once at construction and on resume would also give step 0 and evaluation bounded frames.

**M5. A sync-free matrix exponential.** Because the chart norm is bounded, a fixed-schedule scale-and-square needs no host read (item 19). It removes the main remaining obstacle to capturing a step in a CUDA graph.

**M6. Instrument the profiled root solve.** Log the residual and a count of floored slopes, so that a run near a fold of the profiled energy is visible.

### Dead code

- **New this round:** about 50 lines left by the removals (H3 and the leftovers listed under it).
- **Round-1 deletion candidates, none removed:** about 1.7k lines unreached from production in `vfe3/` and the drivers:
  - the metric registry and 9 test-only functions (431 lines);
  - `vfe3/posthoc/holonomy.py` (327);
  - the geometry test-only helpers with the retraction registry (about 243);
  - `vocab_comparison_figures` (126);
  - the numerics monitor registry (125);
  - test-only prior-bank decode helpers (92);
  - the E-step stubs (about 65);
  - test-only multi-seed helpers (62).
- **Root:** `agent_tooling/` (79 files, 8,384 lines; 29 of its files pass removed fields).
- **Tests:** 16 `importorskip` calls on in-repo modules would turn an import regression into a silent skip.

## Punch list

| # | Severity | Finding | Location | Source |
|---|---|---|---|---|
| 1 | medium | Default flip to inverse makes every finished ablation and scaling cell non-reusable; projected_gradient arms no longer match the launcher's adjoint | `vfe3/config.py:563`; `ablation.py:288-509`; `scaling.py:886-893` | config, routing |
| 2 | medium | Constant α refused unconditionally; October 4 to 7 launcher runs cannot be loaded | `vfe3/config.py:3003-3006` | config |
| 3 | medium-low | Trace-cap bound detached from gamma; gamma gradient wrong when the cap binds | `vfe3/model/block_mlp.py:492` | geometry |
| 4 | medium-low | Old fixed-window cells reused silently beside per-epoch-origin cells | `vfe3/config.py:2587`; `ablation.py:3197`; `scaling.py:870-905` | config, training |
| 5 | medium-low | C11 live-route profile is the kernel route, and its test is self-referential | `tests/profiles.py:212-270`; `tests/test_live_route_profile.py:30` | config, model, hygiene |
| 6 | medium | M-step captures switch on every minimal-agent diagnostic: about 2× operations per step | `vfe3/model/model.py:1065, 1203-1207`; `vfe3/model/relational_operators.py:926` | agents, model |
| 7 | medium | Ablation cells record no realized-update or diagnostics-failure counts | `vfe3/train.py:5105, 5173`; `vfe3/run_artifacts.py:4904` | training |
| 8 | medium (latent) | Prior-anchored floor reads an un-handed-off layer prior when `prior_handoff_sigma > 0` | `vfe3/model/stack.py:34-52`; `vfe3/config.py:2329-2336` | agents |
| 9 | low | Prototype exact-tile and negative-variance flag gaps | `vfe3/model/prototype_agents.py:267, 636` | agents |
| 10 | low-medium | Multi-seed driver and figure worker refuse old runs with removed values | `multiseed_analysis.py:100-107`; `vfe3/viz/figure_worker.py:143-149` | hygiene |
| 11 | low | Posthoc intervention rows raise under `noisy_key` | `vfe3/posthoc/interventions.py:362, 630` | hygiene |
| 12 | low | Train CE under second-order decode, and ELBO and evidence labels, misdescribe their values | `vfe3/train.py:834, 1092, 4821-4822`; `vfe3/model/minimal_agent.py:1066` | routing, model, agents |
| 13 | low | 30 never-written diagnostic keys still read; contract test misses most readers | `vfe3/train.py:4895-4913`; `vfe3/run_artifacts.py:5566`; `vfe3/paper/tables.py:816` | hygiene |
| 14 | low | Frame bound unrecorded; frozen positional factor not covered; boundary chatter | `vfe3/train.py:1218`; `vfe3/gauge_optim.py:1195, 1317` | geometry, training, hygiene |
| 15 | low-medium | Label step still factors B_j three times (four under `spread_corrected`) | `vfe3/model/minimal_agent.py:478, 866, 1174` | model |
| 16 | low | Miscellaneous: scipy import, 2N-token minimum, `pos_loss_ratio`, memory estimate, root-solve floor, sweeps, C7 and C12 leftovers, NaN fill | see item 16 | various |
| 17 | low | README, manuscript and launcher comments describe the old floor and omit the new options | `README.md:287, 545-548`; `train_vfe3.py:128, 131`; `ablation.py:330` | agents, routing, hygiene |
| 18 | medium | Improvement: `amp_dtype` inert on every route tested | `train_vfe3.py`; `vfe3/config.py:2152` | model |
| 19 | low | Improvement: seven syncs per step; sync-free matrix exponential and stacked calls | `vfe3/geometry/transport.py:1154, 1163` | model |
| 20 | improvement | Round-2 hardcode candidates H1 to H13 | `vfe3/config.py` and launchers | config |
| 21 | improvement | M1 to M6 | see Theoretical and mathematical improvements | agents, geometry, routing, model |
| 22 | improvement | Remove the round-1 dead code (about 1.7k lines and `agent_tooling/`) | see Dead code | hygiene |

## Test results

The full CPU suite was not run for this audit; a run was started and stopped at the user's instruction. Investigators ran the test files relevant to their areas, and all of them passed except the platform-sensitive equal-head goldens in `tests/test_son_phase0_20261005.py`. Those fail on torch 2.14.1 Linux on both the `6b872b4` tree and this one, so the failure is not a regression. The relevant files that passed:
- the T1, T2 and T6 tests, the deferred-readout tests and the floor test;
- the closed-form-profiled, noisy-key and second-order decode tests;
- the config, migration and live-route files (193 tests);
- the training files (143 tests);
- four analysis files.

The user's last recorded CPU lane on this tree passed 4,717 tests and skipped 14. The GPU lane was not run.
