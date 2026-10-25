# Evaluation, artifacts, and measurement investigation

This lane inspected the live checkout at `92a7cdfb5bf02172feae22d698195c0abc09e738` on October 25, 2026. Its scope was cache-only corpus loading and window construction, post-hoc replay and feature probes, checkpoint averaging, paper metric selection and parameter accounting, and selected multiseed semantic-analysis contracts. All investigator work was source-only. The coordinator ran the two small reproductions identified below; no corpus replay, training, GPU access, or implementation changes were performed by this lane.

## Correctness defects

### EA1. Serialized parameter accounting counts a tied memory bank once per layer

**Location:** `vfe3/run_resources.py:235-250`; callers `vfe3/posthoc/report.py:1210-1212` and `vfe3/paper/runs.py:873-878`; shared module construction `vfe3/model/model.py:447-449`.

**Severity:** medium (P2), measurement defect. The resolved live launcher uses a tied cue/result bank, so its post-hoc resource report reaches the affected case.

**Evidence:** `parameter_breakdown` executes `for name, tensor in named_params.items()` and increments `sections[section][role]` by each entry's `numel`. A tied bank is built with `nn.ModuleList([shared] * cfg.n_layers)`. Consequently, its state dict deliberately contains each layer's alias, while `dict(model.named_parameters())` contains the shared parameter only once. Training resource reporting passes named parameters at `vfe3/run_artifacts.py:4305`; post-hoc reporting passes the checkpoint state directly. The two supported inputs to the same accounting function therefore disagree on unique trainable capacity.

The coordinator reproduced this with a tiny three-layer, K=4, V=7 cue/result model. The shared bank has 264 parameters. Untied named-parameter and state-dict totals both equal 1,086. With tying enabled, the named-parameter total is 558 while the state-dict total remains 1,086; the bank subtotal is respectively 264 and 792. The extra 528 is exactly two unnecessary copies of the bank. Receipt: `probes_contracts_followup.json`, entry `parameter_counts`.

The consequence is an incorrect capacity and capacity-share report for tied models. Paper assembly uses the bad count when it falls back to post-hoc resources or the final checkpoint; it preferentially uses the correct training `resources.json` when present. This finding concerns unique trainable parameter counts. The separate definition of parameters read per token needs an explicit decision about repeated access to a shared bank and is not adjudicated here.

**Fix:** Count shared parameters once using a saved named-parameter/alias manifest or the serialized tying topology; do not rely solely on storage pointers because existing CPU snapshots copy the aliases independently.

**Test:** Compare named-parameter, in-memory state-dict, and save/load checkpoint counts under every tied bank mode, with two or three tiny layers.

**Falsifies:** A current checkpoint-to-accounting path that removes repeated `block_mlps.<layer>` aliases before totaling them, or equal unique counts in the reproduced tied fixture.

**Prior coverage:** The current tied-bank topology tests check the live model and executable build count, but do not call this state-dict accounting path. The October 20 report predates the tied-bank addition. This is distinct from the training lane's storage-duplication opportunity.

### EA2. The prototype feature probe fails when the bank is shared across layers

**Location:** `vfe3/posthoc/feature_probes.py:145-153`, `:201-213`; end-of-run caller `vfe3/viz/report.py:690-698`.

**Severity:** medium (P2), diagnostic failure on a supported optional route. Trigger: `use_block_mlp=True`, `block_mlp_mode='prototype_agents'`, `block_mlp_tie_layers=True`, and `n_layers > 1`. The current exact-route launcher uses cue/result memory, so this particular prototype failure is outside its active route.

**Evidence:** Targets are enumerated per layer: `[(index, mlp) for index, mlp in enumerate(blocks) ...]`. Each target assigns `mlp.forward_moments = recording`, even when two targets are the same object. The second assignment wraps the first wrapper; each real layer invocation is consequently recorded under multiple layer indices. Cleanup then executes `del mlp.forward_moments` once per target. The first deletion succeeds, and the second deletion raises `AttributeError` because the shared object no longer has an instance override.

The coordinator reproduced the cleanup failure on a tiny two-layer tied prototype model with an empty token iterable and `gap_tokens=0`, isolating it from all numerical inference. The exception is `'PrototypeAgentBlockMLP' object has no attribute 'forward_moments'` at line 213. Receipt: `probes_contracts_artifacts.json`, entry `tied_prototype_probe`. The same repeated cleanup follows a nonempty successful replay. The end-of-run wrapper catches the error, logs it, and skips the prototype measurements and figures. Simply deduplicating cleanup would still leave the per-layer measurements wrong because the wrappers are nested.

**Fix:** Install one wrapper per distinct bank and attribute calls to their actual layer invocation, then restore each distinct method once.

**Test:** Run a tiny tied prototype model through the probe with real tokens, verify the expected per-layer token counts and distinct layer inputs, and verify that the original method remains usable afterward.

**Falsifies:** Distinct bank objects despite `block_mlp_tie_layers=True`, or a current wrapper/cleanup implementation that tracks unique objects and actual layer calls.

**Prior coverage:** Existing feature-diagnostic tests cover untied probes; the new tied-bank tests do not exercise feature probes. No matching defect appeared in the October 20 report.

### EA3. A multiseed checkpoint override relabels old semantic metrics as the new checkpoint

**Location:** `vfe3/semantics/multiseed.py:206-213`, `:956`, `:965`, `:1001-1004`; rendered checkpoint label `:815-819`; public option `multiseed_semantics.py:29`.

**Severity:** medium (P2), source-traced result-labeling defect under an explicit supported override. No existing user analysis was checked for this condition.

**Evidence:** `load_seed` reads the existing `metrics.json`, obtains its recorded checkpoint, and then chooses `name = checkpoint or ...recorded...`. It loads the selected model but retains the previously read metrics. Later, `table = headline_metrics([r.metrics for r in records], labels)` uses those old measurements, whereas `stability_analysis(records, ...)` uses the newly loaded weights. The output records `"checkpoint": r.run.checkpoint_path.name`, and that is the checkpoint shown in the Runs table above the headline metrics.

For example, semantic analyses recorded at `best_model.pt` followed by aggregation with `checkpoint='final_model.pt'` produce headline scores from best weights and stability scores from final weights under a single displayed final-checkpoint label. Choosing a different supported checkpoint is sufficient; no file damage or abnormal input is required.

**Fix:** Reject a checkpoint override that differs from the checkpoint of the saved metrics, or retain separate, explicit `analysis_checkpoint` and `stability_checkpoint` provenance throughout the output.

**Test:** Build two tiny analysis records whose saved checkpoint is best and request final; assert either an explicit refusal or unambiguous labels for both measurement populations. Model loading can be stubbed because the defect is the provenance/data-flow contract.

**Falsifies:** Recomputing the headline metrics at the override before aggregation, or separately naming the old and new checkpoints in the emitted artifact.

**Prior coverage:** No matching finding in the inspected October 20 report; this source path had previously been listed as a coverage gap. This finding is source-traced, not a claim of an executed semantic-battery failure.

### EA4. Checkpoint parameter counts include several persistent nonparameter buffers

**Location:** `vfe3/run_resources.py:64`, `:235-250`; buffer producers `vfe3/model/prior_bank.py:858-863`, `vfe3/model/cg_coupling.py:111`, `vfe3/model/model.py:299-300`; consumers as in EA1.

**Severity:** low (P3), optional-route measurement defect, source-traced.

**Evidence:** The only excluded leaf in `BUFFER_LEAVES` is `unigram_log_prior`. The loop counts every other checkpoint tensor as a parameter unless listed as frozen. `additive_R` is registered as a buffer under `encode_mode='per_token_additive'`; `cg_<t>` intertwiners and `gauge_basis_fingerprint` are also persistent buffers. None has a corresponding exclusion here. The additive control's fixed K-by-n_gen projection therefore increases the reported checkpoint parameter count even though it contributes no trainable capacity; CG and basis metadata cause analogous overcounts on their optional paths. Training counts derived from `named_parameters()` do not have this error.

**Fix:** Use an authoritative saved parameter manifest when accounting from a state dict, or extend the known-buffer handling to include these exact registered buffers without treating trainable tensors as buffers.

**Test:** Compare live named-parameter and checkpoint counts for a tiny additive-control model and an optional tiny CG model. These cases were not executed in this lane.

**Falsifies:** These buffers being nonpersistent, absent from the affected checkpoint, or explicitly removed before `parameter_breakdown` is called.

**Prior coverage:** Existing post-hoc parameter-count tests exercise the unigram buffer, not the named optional buffers. Kept separate from EA1 because aliasing and parameter-versus-buffer identity are independent accounting errors.

## Unmeasured performance opportunity

### EA5. Per-target post-hoc NLL materializes the complete vocabulary tensor

**Location:** `vfe3/posthoc/evaluate.py:231-241`.

**Severity:** low (P3), performance opportunity; no timing or peak-memory claim.

**Evidence:** The pass requests `forward_beliefs(..., return_logits=True)` and then computes cross-entropy on the complete `(B,N,V)` logits. For the current resolved B=32, N=64, V=50,257, one float32 tensor of that shape is 392.63 MiB before decoder temporaries. The training decoder already has chunked target/log-normalizer reductions in `vfe3/model/prior_bank.py`; the post-hoc pass needs only per-target NLL, validity, and observations of the belief stack.

**Fix:** Expose a per-target NLL result from the existing chunked decoder and use it in the post-hoc pass, preserving the separate full-logit path for consumers that need it.

**Test and falsifier:** Compare per-target NLL and excluded-target masks against the current dense pass on tiny models, including ragged vocabulary chunks and reference-coordinate decoding, then measure a representative post-hoc replay if performance work is authorized. The benefit is falsified if the decoder already avoids the full logits allocation at this call, or if measured post-hoc peaks and time are unchanged.

**Prior coverage:** No matching item in the inspected October 20 report. This is an analytic allocation count, not a GPU benchmark.

## Design/test improvement

### EA6. Preserve failed replay cross-checks when paper assembly consumes post-hoc scores

**Location:** `vfe3/posthoc/report.py:825-835`; `vfe3/paper/runs.py:690-697` (`posthoc_pass`).

**Severity:** low (P3), source-backed improvement with no reproduced inconsistent replay in this lane.

**Evidence:** The report marks an NLL pass `status='ok'`, then records a separate `cross_check.ok` flag against `train.evaluate`. A false flag does not change the pass status. Paper `posthoc_pass` requires only the checkpoint/split match, successful status, full split, and finite CE, so a flagged NLL value remains eligible as a fallback selected-test score. The check is visible in `posthoc.json`, but that warning is not carried into the paper selection result.

**Fix:** Keep the measurement but propagate the failed consistency check into the selected score's eligibility or displayed qualification. Intentional precision changes should remain distinguishable from an internal NLL-versus-evaluate mismatch.

**Test and falsifier:** Supply a normal in-memory pass record with `cross_check.ok=False` and verify that selection exposes or rejects the discrepancy. Retract this improvement if the downstream artifact already carries the cross-check qualification. No claim is made that current ordinary inputs trigger the mismatch.

## Coverage and limits

Detailed source reading covered the complete checkpoint-average module, post-hoc model loading and NLL/protocol/bootstrap code, stream identity resolution and evaluation/training-sample windows, the `TokenWindows` and loader bodies, parameter role/counting code, paper checkpoint selection and decision-rule consumers, prototype/minimal-agent feature-probe bodies, selected intervention applicability guards, calibration coverage/conversion, and multiseed loading/configuration/statistical aggregation. Semantic CKA, Spearman/bootstrap, Holm adjustment, and crossed-unit bootstrap implementations were sampled; their full callers and mathematical inference conditions were not rederived. Paper figure rendering, all semantic-battery implementations, and all external resource ingestion were not exhaustively inspected.

The inspected normal data path raises when a real cache is absent; no synthetic fallback was found. Padded evaluation windows retain exactly-once target masking, train sampling keeps its distinct protocol, checkpoint averaging checks serialized config and tensor compatibility, and paper selection distinguishes best and final weights. These are source observations, not a full execution certificate. Existing dataset correctness tests were not rerun. No software was changed and no finding was independently adjudicated in this lane.
