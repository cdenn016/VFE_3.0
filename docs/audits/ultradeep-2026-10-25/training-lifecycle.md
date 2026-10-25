# Training and runtime lifecycle audit, October 25, 2026

This lane inspected the live working-tree source at the baseline recorded in `brief.md` (`92a7cdfb5bf02172feae22d698195c0abc09e738`). It found two conditional correctness defects, one unmeasured memory/copy optimization, and one EMA evaluation gap. The coordinator executed small CPU reproductions of both defects. No investigator ran training, tests, profiling, or GPU work, and no finding was independently adjudicated.

The resolved launcher in `live-config.json` has `principled_exact_route=True`, K=32, two heads, four layers, tied cue/result memory, working-variance initialization, table Fisher updates, AdamW frames, no autocast, and `fp32_matmul_precision="high"`. It has no resume request, no weight EMA, and no loss-spike rejection. Those choices are intentional. Findings below name the additional conditions that reach them; a disabled option is not a defect. The selected memory is statistical cue/result inference; the separately implemented conventional baseline has its own architecture and optimizer settings. The source, rather than the older invariant descriptions of retired routes, determines the behavior reported here.

## Correctness defects

### TL1. Resuming without artifact output drops state that controls update acceptance

**Location:** `vfe3/run_artifacts.py:2986-2996`; `vfe3/train.py:5766-5796`, `4905-4913`, `6355-6369`, `6370-6376`; launcher `train_vfe3.py:633-644`, `679-690`.

**Severity:** P2.

**Trigger and scope:** Resume a normal checkpoint through `train(..., artifacts=None, resume_from=...)`, including the supported launcher setting `RUN_ROOT=None`, after the loss monitor has warmed up, with `loss_spike_skip_nats` enabled. The same missing restoration also resets `max_failed_step_skips` and `max_consecutive_skips` accounting. The checked launcher is not currently resuming and these rejection options are off.

**Evidence and causal trace:** `load_checkpoint` puts the saved divergence record only on `artifacts.resumed_divergence_state`, inside `if artifacts is not None`. The loop initializes a fresh `_DivergenceMonitor`, `consecutive_skips=0`, and `failed_step_skips=0`; its sole restoration source is `getattr(artifacts, "resumed_divergence_state", None)`. Without artifacts that source is absent, although model weights, optimizer, RNG, and the exact data cursor are restored. The next step's rejection threshold comes from `divergence.loss_skip_threshold(loss_spike_margin)`, which returns `None` before the newly reset warmup completes. `failed_step_skips` also gates further exception skips at line 6374.

The coordinator ran `probe-resume-monitor.py` with `C:/anaconda/python.exe`, hidden CUDA, one CPU thread, K=4, H=2, V=5, B=1, N=2. It saved an ordinary checkpoint after one actual step and resumed it twice. The fixture shortened `DIVERGENCE_EMA_WARMUP` to 1 and added a constant 10-nat loss term only on the resumed forward. It did not edit the checkpoint. `probe-resume-monitor.log` records:

```text
saved monitor: loss_seen=1, loss_ema=1.2419383525848389
with artifacts: threshold=2.241938352584839, loss=12.472694396972656,
                did_step=False, loss_spike_skip=True
without artifacts: threshold=None, loss=12.472694396972656,
                   did_step=True, loss_spike_skip=False
```

**Consequence:** Changing only output persistence changes the training trajectory after resume: an update rejected by the saved policy is accepted. Disabling output also grants a fresh failed-step budget. The checkpoint contains the needed information, so this is a lost integration path rather than an unavoidable legacy-checkpoint limitation.

**Fix:** Return the divergence/rejection state through an artifact-independent resume-state object and restore it in the loop regardless of output persistence.

**Reproduction/test:** Retain the tiny probe as the evidence; an implementation test should compare artifacts-enabled and artifacts-free resumes from the same normally produced checkpoint, including thresholds and failure budgets. Its initial K=2,H=1 fixture was rejected by configuration validation before a model ran; the corrected K=4,H=2 fixture passed.

**Falsifies:** An artifact-free resume restores the saved loss EMA/warmup and both skip counters before constructing the next step's threshold, and the paired probe yields the same acceptance decision.

**Prior work:** October 16 F77 added divergence-state persistence, and `tests/test_audit_20261016_c1_resume_artifacts.py:88-114` compares resumed monitor outputs with artifacts present in every leg. That fix covers persisted continuations; it does not cover this output-disabled leg. The October 17 skip-policy tests exercise fresh runs.

### TL2. A locked best-model file lets validation finalization rank weights that it did not publish as selected

**Location:** `vfe3/run_artifacts.py:2295-2305`, `5233-5262`, `5302-5346`; consumer `ablation.py:4985`, `5154-5155`.

**Severity:** P2.

**Trigger and scope:** A validation-only run already has a finite earlier best and a reachable `best_model.pt`. Terminal validation improves on it, but the ordinary Windows open-file lock survives the existing replacement retries and raises `PermissionError` during publication of the new best. This requires persistence enabled. A first-best publication failure is already caught by the finite-best guard and is not this finding.

**Evidence and causal trace:** `_publish_best` catches `PermissionError` and returns `False` before advancing `best_val_ppl` or `best_step`, preserving the earlier file. `finalize_validation_run` saves `prior_best`, calls `artifacts.maybe_save_best(completed_step, model, final_ppl)` without using its result, and only checks whether some finite best and file remain. That check passes for the earlier selection. It then computes:

```python
primary_val_ppl = (min(prior_best, final_ppl) if math.isfinite(prior_best) else final_ppl)
best_val_ppl = artifacts.best_val_ppl if artifacts.best_val_ppl != float("inf") else None
```

For an earlier selected PPL of 20 and terminal PPL of 18, the failed publication leaves `best_val_ppl=20` and the old `best_step`, but returns/publishes `primary_val_ppl=18`. The terminal checkpoint then embeds the earlier selected bundle through `save_checkpoint`, and the summary is still written as a successful result. The ablation runner reads `primary_val_ppl` for its headline and derives comparison CE from its logarithm.

**Consequence:** The validation score used to rank the arm no longer identifies its published selected-model artifact. The raw terminal weights can still be recovered from the terminal checkpoint; the defect is the successful selection handoff and contradictory primary/best record, not loss of every copy of the terminal iterate.

**Fix:** Treat a failed publication of an improving terminal selection as a finalization failure, or explicitly return the retained selected score as primary and label the better raw terminal score as unpublished.

**Reproduction/test:** The coordinator executed `probe-locked-terminal-selection.py` on CPU with a tiny model and a normally saved earlier best. Only terminal validation scoring and the ordinary publication failure were simulated; no training or existing file-byte changes were required. `probe-locked-terminal-selection.log` records `primary_val_ppl=18.0`, `best_val_ppl=20.0`, `best_step=1`, and `summary_written=true`. The fixture's initial configuration omitted a compatible warmup value and was rejected before model construction; setting its audit-only `warmup_steps=0` allowed the completed check. A future regression should require finalization to refuse this contradictory selection handoff or report the retained selected score.

**Falsifies:** The failed-publication branch either stops successful finalization or guarantees `primary_val_ppl`, selected step, and selected bundle describe the same state. A documented separate meaning for primary as an unsaved observed score would require corresponding downstream labels and selection behavior; the current successful selected-artifact handoff does not provide that separation.

**Prior work:** Existing best-save behavior deliberately retains the prior best on a transient Windows lock, and October 17 tests cover successful background publication. The terminal-validation tests cover absent/stale selection, ordinary publication, completed-step accounting, and raw-state restoration; the retained-earlier-best plus blocked improving publication combination is not covered in the inspected tests.

## Unmeasured performance opportunity

### TL3. Host snapshots duplicate each tied memory bank once per layer

**Location:** `vfe3/run_artifacts.py:2269-2270`, `4232-4233`; `vfe3/train.py:7233-7235`; alias contract `vfe3/model/model.py:632-645` and `tests/test_tied_block_mlp_20261025.py:58-68`.

**Severity:** P3, optimization opportunity, not a correctness defect.

**Trigger and scope:** `block_mlp_tie_layers=True` with more than one layer, followed by a background best save, terminal-callback snapshot, or the host backup used to evaluate an earlier best. The resolved launcher has four tied layers and reaches background best saving. Its present K=32 memory bank is small; no material wall-time claim is made for that shape.

**Evidence and causal trace:** The tied model retains every per-layer state-dict name as aliases of one bank. The snapshot comprehensions independently execute `tensor.detach().to("cpu", copy=True)` for every name. Consequently the snapshot has one independent copy per aliased key, and background `best_model.pt` serialization no longer benefits from the original shared storage for those tensors. This does not affect the optimizer, which deduplicates shared parameters, or the synchronous `save_checkpoint` state mapping, which does not make these per-key copies.

**Consequence:** A bank represented once during training takes L host copies in these snapshots and, on the background-best path, in the saved bundle. The added memory and copy traffic scale with tied depth and bank width. Total run-memory or latency savings have not been measured.

**Fix:** Clone each unique state tensor/storage once and retain all existing checkpoint names as aliases to that cloned storage.

**Reproduction/test:** On a tiny tied model, capture the background-save bundle before writing and compare the storage pointers for `block_mlps.0.*` and `block_mlps.1.*`; verify all key names and values remain unchanged after an alias-preserving snapshot. Compare unique storage bytes, not the sum of tensor sizes over keys.

**Falsifies:** The snapshot preserves shared storage for tied keys, or the selected route never creates the repeated state-dict aliases. Existing tied-bank tests explicitly establish the aliases' key layout but do not inspect the snapshot bundle's storage sharing.

**Prior work:** The October 25 tied-bank implementation deduplicates optimizer grouping, calibration, and parameter counts. This concerns the existing host-copy consumers of that new alias layout.

## Design and test improvement

### TL4. Validation-only terminal runs do not evaluate their accumulated EMA

**Location:** `vfe3/train.py:7142-7146`; `vfe3/run_artifacts.py:4112-4160`, `4471-4474`, `5192-5194`, `5314-5369`.

**Severity:** P3, evaluation-coverage gap.

**Trigger and scope:** Enable weight EMA for a run that ends through `finalize_validation_run`, especially an ablation with no periodic evaluation before its terminal step. The checked launcher has EMA disabled, so this is not a finding about its current training behavior.

**Evidence and causal trace:** The training loop exposes `artifacts.weight_ema` and its checkpoint state before invoking the terminal callback. `finalize_run` calls `_weight_ema_evaluation`, which supports `test_loader=None`, and publishes the EMA scores/status. `finalize_validation_run` evaluates only the raw model and emits no EMA score or status in its terminal row, results, summary, or return mapping. A sweep cell with `eval_interval > n_steps` therefore accumulates and checkpoints the shadow but never measures it.

**Consequence:** That supported experiment path cannot evaluate the effect of its EMA option from its own output. Raw model selection is internally consistent; this gap does not show that EMA changes raw training or should become the selection policy.

**Fix:** Reuse `_weight_ema_evaluation` in the validation-only finalizer with `test_loader=None`, publishing terminal EMA validation and status alongside the existing raw selection.

**Reproduction/test:** A tiny EMA-enabled terminal callback with periodic eval disabled should report finite `val_ema_ce`/`val_ema_ppl`, preserve raw parameters and RNG, and never open/evaluate a test loader.

**Falsifies:** The validation-only finalizer records an EMA evaluation or an explicit intentional-not-evaluated status for the completed shadow, including when no periodic eval ran.

**Prior work:** `tests/test_weight_ema_20261021.py:152-223` covers raw-trajectory invariance, rejected updates, shadow resume, and direct `_weight_ema_evaluation`; it does not cover the ablation terminal callback.

## Coverage and negative findings

The lane traced `train_vfe3.py` and `train_baseline.py` construction/reseeding, the baseline's native optimizer construction, `build_optimizer`, the scheduler/floor and accepted-update clock, single-batch and token-count-weighted accumulated gradients, scaler/unscale/clip and nonfinite handling, precision retries, failed-step and consecutive-skip controls, crash checkpoints, per-epoch window origins and generator replay, periodic and terminal validation, raw/best weight selection, `WeightEMA`, and checkpoint/model/optimizer/data restoration in `run_artifacts.py`. Optimizer-internal geometry and custom kernels were assigned to other lanes. Relevant source was cross-checked against the October 20 audit and focused resume, accumulation, monitoring, terminal-selection, and EMA test definitions; these tests were read, not run by this investigator.

The following older issues or suspicions were not re-reported: the scheduler advances only after accepted updates and resumes from the saved successful-update clock (`train.py:1364-1367`, `5683-5694`); the explicit zero-scored target gate skips updates; microbatch gradients use decoder counts and a single normalization; a precision retry rewinds RNG and refuses to retry once an optimizer update begins (`4456-4515`); the crash record is written before slow recovery stages; final/raw evaluation remains separate from validation-selected weights; and the stored data cursor includes per-epoch window origin plus generator state. The template calibration/EMA ordering is correct: constructing `WeightEMA` before working-variance calibration retains references only; its shadow is first copied after an accepted training update (`weight_ema.py:62-72`, `93-100`).

Research continuity included `Research/index.md`, the VFE Transformer Program hub, and the inference-machinery theme. Their older optimizer descriptions were treated as historical context, not evidence of present runtime behavior. Prior memory about schedule-horizon versus stopping-step comparisons was used only to maintain scope; current schedule behavior was read directly.

No training-quality, GPU-performance, convergence, or empirical hyperparameter recommendation is made. The CPU reproduction is a control-flow check with a synthetic added loss and shortened monitor warmup, not evidence of an observed corpus-training divergence. The width/depth-dependent snapshot cost and the EMA terminal gap remain unmeasured.
