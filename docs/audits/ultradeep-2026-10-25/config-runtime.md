# Configuration and runtime wiring audit, October 25, 2026

This lane inspected the live working tree identified in `brief.md`, at HEAD `92a7cdfb5bf02172feae22d698195c0abc09e738`, including the intentional `ablation.py` WIP. It made no implementation changes and ran no Python, tests, model, training, GPU query, benchmark, or subagent. The coordinator separately constructed the live config without constructing a model and saved `live-config.json`; that receipt covers configuration resolution only. The coordinator also executed the two bounded reproductions below in `probes_config_cg.py`, with results in `probes_config_cg.json`. The findings have not been independently adjudicated.

## Active routes and override trace

The active training route is the resolved exact-route configuration, not the coordinate-MLP route written in the raw dictionary. `train_vfe3.py:116` sets `principled_exact_route=True`; `train_vfe3.py:589-598` applies `principled_exact_route_overrides(config)` before constructing `VFE3Config` and overriding the model seed. There is no current launcher multiplication of the literal `kl_max=128`. `VFE3Config.__post_init__` changes `kl_max` only when it is `None`, at `vfe3/config.py:1258-1259`.

| Setting | Raw training dictionary | Resolved training config |
|---|---|---|
| Shape | K=32, H=2, L=4, context=64, batch=32 | Same |
| Attention route | `projected_predictive`, kernel readout | `projected_predictive`, `minimal_agent` readout |
| Minimal-agent inference | `mean_field` | `exact` |
| Belief family | `gaussian_diagonal` | `gaussian_frame_diagonal` |
| Predictive covariance | `passthrough` | `moment_matched` |
| Block | Coordinate MLP, local coordinates, LayerNorm | Cue/result memory, reference coordinates, no block norm |
| Memory posterior | `mean_field`, covariance passthrough | `exact`, posterior covariance |
| Bank sharing / regime | Untied, regime off | Tied, sticky regime |
| Decode | `diagonal_chunked`, local, tau=0.1, untied bank | `diagonal_chunked`, reference, tau=1, untied bank |
| RoPE | `comparison_invariant` | `comparison` |
| `sigma_max` | Omitted, dataclass default 100 | Override `None` |
| Frame condition bound | 20 | 20, deliberately preserved |
| Frame learning | AdamW, rate 0.002 | Same |
| Memory regression rank | 2 | Same |

The fifteen overrides appear in `live-config.json:170-186`, the full resolved object begins at line 187, and `exact_route_failures=[]` is recorded at line 381. The frame bound is an explicit exception implemented by `vfe3/exact_route.py:69,86-87,119-123`; retaining it is not a finding. The route check validates the configured contract, not the mathematics of every operator or execution at production size.

The active ablation is different. `ablation.py:252` selects `frame_capacity`, with both global route switches false at lines 320 and 329. Its baseline is K=128, H=4, L=4, context=128, batch=64, 15,000 steps, and evaluation stride 32 (`ablation.py:338-350`). The sweep requirements at lines 1351-1366 select the kernel readout, coordinate MLP, comparison-invariant RoPE, ALiBi, no positional frames, and an attention residual. The three arms are learned frames, identity frames, and identity frames plus the per-type hidden bias. They use model/data seed pairs (6,6), (23,23), and (54,54), because the sweep seeds are explicit and `DATA_SEED=None`. The standalone `CONFIG` run knobs are not `VFE3Config` fields.

For the hidden-bias arm, `m_per_type_hidden_bias_lr=None` is supplied by the sweep rather than the baseline's explicit 0.0025. `vfe3/config.py:1338` retains the declared frame rate 0.002, the gauge-off branch zeros `m_phi_lr` at line 1353, and lines 2315-2316 resolve the hidden-bias rate from the retained value. `vfe3/train.py:400-410` places this table in an AdamW group with its resolved rate, `phi_weight_decay`, and the phi clipping role. `vfe3/model/model.py:576-580` freezes frame parameters for both `off` and `frozen`. These paths preserve the intended hidden-bias learning and frozen-frame controls on source reading.

The exact-route sweeps carry their requirements in `ablation.py:1383-1446`, including when the global exact-route switch is off. Arm requirements merge before the arm values at lines 2055-2066. The resolved baseline precedes those overrides at lines 2162-2166 and 2243-2244. `_require_exact_route` enforces the route unless the sweep explicitly permits leaving it; the component sweep is the declared exception. The cell runner also forces `checkpoint_interval=0`, sets the requested seed, and optionally replaces `max_steps` at lines 2254-2257. No finding below treats these disclosed regime choices as defects.

## Correctness finding

### C1. Single-layer models bypass mode-specific inactive-knob detection and admit no-op activation sweeps

**Location:** `vfe3/config.py:2747-2761`; consumers at `ablation.py:1056-1060,1894-1914`; unused-argument factories at `vfe3/model/block_mlp.py:2264-2279`.

**Severity:** medium / P2.

**Trigger:** A legal single-layer projected model with `use_block_mlp=True` and `block_mlp_mode='prototype_agents'`, `'gaussian_template'`, or `'cue_result_memory'`, followed by the existing `block_mlp_activation` sweep. It also affects the covariance-floor classification for single-layer passthrough coordinate MLPs. The shipped four-layer training and frame-capacity configurations do not trigger this branch; setting `n_layers=1` does.

**Evidence:** The executable branch sequence is:

```python
if not self.use_block_mlp:
    mark(...)
elif self.n_layers == 1:
    mark(("block_mlp_tie_layers",), "a single layer has one block MLP either way")
elif self.block_mlp_mode == "gaussian_template":
    mark(("block_mlp_activation", *norm_floor), ...)
elif self.block_mlp_mode in ("prototype_agents", "cue_result_memory"):
    mark(("block_mlp_activation", *norm_floor), ...)
elif self.block_mlp_covariance != "delta_full":
    mark(norm_floor, ...)
```

The `n_layers == 1` arm prevents every later mode-specific branch from running. This is not merely a missing display warning. The three relevant registered factories remove `activation` and `covariance_floor` using `kwargs.pop(unused)` before constructing their modules (`block_mlp.py:2264-2279`), so changing the activation cannot affect those blocks. Model construction forwards the configured activation into that registry path. The existing activation sweep varies `gelu`, `silu`, and `relu` and requires only `use_block_mlp=True` (`ablation.py:1056-1060`). At validation, lines 1900-1903 intersect each resolved config's `inactive_fields()` with the varying keys; lines 1908-1914 reject only keys classified inactive on every arm. Under the singleton branch, activation is absent from this set and the no-op sweep is admitted.

The coordinator's `probes_config_cg.json` confirms the oracle mismatch: `inactive_activation` is false at depth 1 and true at depth 2 for `prototype_agents`, using otherwise matching test-owned configurations. Sweep execution itself was not run. Config tracing performed for this finding: procedure steps 1-4 and 6-7 by source inspection, step 5 by reading the current `inactive_fields` oracle and its sweep consumer, step 9 by reading its serialized reconstruction path, and step 10 by checking relevant live-config mutation sites. The coordinator's bounded probe supplies step 8 for the oracle branch, not for a complete training run. The selected execution path is projected predictive attention; this finding does not involve the retired analytic-pair kernel.

**Consequence:** The sweep can train and label several identical experiments as an activation comparison. The launcher's inactive-setting report also omits a setting that the selected factory discards. No previously published result was shown to be affected.

**Fix:** Mark the singleton-only tying knob in an independent conditional, then run the existing block-mode inactivity chain regardless of depth.

**Reproduction proposal:** With CUDA hidden and one CPU thread, compare `test_config(n_layers=1, block_mlp_mode='prototype_agents').inactive_fields()` with the corresponding two-layer config. Activation is missing from the former and present in the latter. For the admission consequence, use a complete tiny profile as `ablation.BASELINE_CONFIG`, set both global switches false, and call `validate_sweeps(['block_mlp_activation'])`: the single-layer prototype route currently passes and the two-layer route should reject the same inert sweep. This requires no model or dataset.

```python
from unittest.mock import patch
from tests.profiles import config_values, test_config
import ablation

for layers in (1, 2):
    cfg = test_config(n_layers=layers, use_block_mlp=True,
                      block_mlp_mode="prototype_agents")
    print(layers, "block_mlp_activation" in cfg.inactive_fields())

tiny = config_values(vocab_size=7, embed_dim=4, n_heads=2,
                     n_layers=1, max_seq_len=4, batch_size=2,
                     use_block_mlp=True, block_mlp_expansion=2,
                     block_mlp_mode="prototype_agents")
with patch.object(ablation, "BASELINE_CONFIG", tiny), \
     patch.object(ablation, "fully_gauge_covariant", False), \
     patch.object(ablation, "principled_exact_route", False):
    ablation.validate_sweeps(["block_mlp_activation"])
```

**Falsifies:** An additional executed branch that still marks activation inactive for those single-layer modes, an actual factory use of activation on those modes, or a sweep admission check that rejects this specific setup before training.

**Prior coverage:** The October 25 tied-bank tests check that `block_mlp_tie_layers` becomes inactive for one layer (`tests/test_tied_block_mlp_20261025.py:123-131`), but do not check that existing mode-specific inactive fields survive that new branch. The October 20 report predates this conditional. This is a new interaction introduced with the singleton tying marker.

## Design and test improvement

### C2. Include the per-type hidden-bias table in scaling previews

**Location:** `scaling.py:639-668,1420-1426,1576-1587`; actual allocation at `vfe3/model/model.py:459-472`.

**Severity:** low / P3.

**Trigger:** A valid scaling configuration or route override with `per_type_hidden_bias=True`, a supported coordinate block, and enough hidden width for its per-layer share of the table. This is an opt-in scaling configuration; the current `frame_capacity` ablation does not use this estimator.

**Evidence:** `predict_n_params` sums token means, variances, frames, decode parameters, positional frames, the registered block count, normalization, and four K-by-K maps per layer, then returns at line 668. It has no branch or delegated argument for `per_type_hidden_bias`. The model independently creates `nn.Parameter(torch.zeros(cfg.vocab_size, n_gen))` at `model.py:472`. Thus changing only this toggle changes the real count by `V*n_gen` and leaves the preview estimate unchanged. `preview_routes` prints that estimate at lines 1584-1587. At model construction, `run_cell` compares against the actual parameter count but labels every discrepancy `small modules` at line 1424. For large vocabularies the omitted table is the same size as the whole frame table and is not a small scalar-module correction.

The trace is baseline plus route override in `scaling._cell_cfg_dict:742-744`, fresh validation in the preview at line 1583, the estimator above, then the actual allocation on model construction in `run_cell:1422`. Fresh and serialized configs both retain this Boolean; its introduction value is false. Relevant procedure steps 1-4, 6, and 9 were inspected. The coordinator's tiny CPU probe supplies the construction check: both estimates are `(460, 8)`, while actual total parameters are 460 with the bias off and 516 with it on (`probes_config_cg.json`, `hidden_bias_parameter_count`). The measured difference is exactly `7 * 8 = 56`.

**Consequence:** The pre-run memory/capacity estimate undercounts the bias arm. Completed scaling results continue to save `actual_n`, so this finding does not assert wrong final parameter counts or wrong scaling exponents in saved results.

**Fix:** Add `cfg.vocab_size * n_gen` when the hidden-bias toggle is enabled, and avoid describing arbitrary unmodeled parameter differences as small modules.

**Reproduction proposal:** Build two complete tiny configs differing only in `per_type_hidden_bias`, with V=7, K=4, H=2, L=2, and expansion=2. `predict_n_params` currently gives the same result. Actual model parameter counts should differ by 56. A focused test can check the estimator delta first, without a forward pass, then compare with the tiny model census.

**Falsifies:** An estimator path outside the inspected body already adds the table before every preview call, or actual model construction does not allocate the table for this legal configuration. The actual-count record in completed runs limits the consequence but does not correct the preview.

**Prior coverage:** The October 25 tied-bank changes updated this estimator to count shared banks once, but did not add the separate per-type table. The earlier report's parameter census addressed a different active configuration. No matching existing regression test was found in the files inspected.

## Other source-backed improvement and residual question

Sweep validation should consume the same effective cell constructor as execution. At present `validate_sweeps` builds `_resolved_sweep_baseline() + overrides` directly (`ablation.py:1896-1900`), while `run_single` uses `_cell_cfg_dict` (`ablation.py:2514`) with extra run-knob merges and an `eval_stride > max_seq_len` repair (`2243-2257`). A user-defined context arm with `max_seq_len=16` against the live baseline's `eval_stride=32` is rejected during validation even though the runner has an explicit, logged disjoint-window fallback. This discrepancy is source-visible; it is not a claimed failure of the currently selected frame-capacity sweep. A config-only regression should compare both resolution paths for a context-shortening arm before deciding whether to unify them or consistently reject the repair.

No new measured performance finding is reported. The source inspection of `contracts.py` found the deferred-check mechanism combines host verdicts before a mandatory read and clears its queue in `finally`; it does not establish its runtime overhead. The unconditional float64 frame-pair certificate and Python stack walk are possible profiling targets, but no timing or throughput claim is made here.

## Coverage and limits

The lane read the full active training dictionary and its run globals, the live ablation baseline/run knobs and selected frame-capacity and exact-route declarations, the scaling baseline and relevant override/preview path, relevant portions of `paper_arms.py`, `config_history.py`, `legacy_config.py`, `contracts.py`, `gauge_covariant.py`, and `exact_route.py`. Within the large config module it traced the exact-type gate, gauge coercions, family restrictions, hidden-bias inheritance, memory restrictions, inactive-field oracle, derived routing properties, historical pins, removed-selector migrations, and fresh/serialized construction. It followed the model/optimizer seams needed for frozen frames and the hidden-bias table. The historical October 20 audit and the relevant October 25 tests were read as prior coverage, not as proof of current execution.

The research index, the VFE Transformer Program hub, the minimal-agent manuscript's model/readings sections, and the local prototype generative-model investigation supplied continuity. No mathematical or novelty claim in this lane relies on those documents as executable evidence. The older partial-buildout note named in the role brief was not found by its filename; the program hub itself documents a later replacement of those routes.

The old role brief's `uses_kernel_route`, `effective_e_step_gradient`, and permissive default serialized loader describe a retired implementation. In this checkout, `vfe3/gradients/kernels.py` does not exist, `attention_mode` has the two projected routes, `relational_readout_mode` is a derived property (`config.py:2983-2988`), and the live forward carries gradients from its caller through the stack. This lane therefore did not claim to execute the obsolete ten-step procedure literally. It applied the construction, override, route, serialization, and mutation checks to their current equivalents.

The main `config_from_serialized` now rejects unknown fields at `config.py:3792-3796`. Introduction values are filled after explicit pins and migrations, before Boolean normalization and fresh construction (`3782-3817`). `legacy_config.lenient_config_from_serialized` explicitly returns its dropped fields; the inspected `paper_arms.load_base` stores that disclosure. The resume preflight compares resolved configs before state mutation (`run_artifacts.py:2641-2662`), with only resume bookkeeping excluded. No new silent-drop or resume-config defect was established in those inspected paths.

The former mid-diagnostic `setattr(model.cfg, ...)` pattern was not found in the current `run_artifacts.py` by the relevant assignment searches. This is a scoped source-search result, not an exhaustive proof that no caller mutates a config anywhere. Ordinary snapshots and output fields were not treated as runtime overrides.

The coordinator's `live-config.json` executes construction and the current inactive-field oracle, but does not establish model execution. Its later `probes_config_cg.py/json` executes C1's oracle comparison and C2's tiny model parameter census; the full sweep-admission reproduction and the context-shortening comparison remain source traces. No fix is claimed complete, and no training result, CUDA behavior, numerical exactness claim, or end-to-end pure-path execution is certified by this report.
