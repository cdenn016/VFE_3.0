# Gauge and geometric-attention investigation, October 25, 2026

This lane inspected the live working-tree source on the baseline specified in `brief.md`. It made no implementation or configuration edits and ran no scripts, tests, training, or GPU work. The coordinator executed the three bounded CPU reproductions cited below, serially. These are investigator findings, without independent adversarial adjudication.

Three correctness findings were supported: two conditional CG execution defects and one rounding-sized violation of prefix independence in the minimal-agent precision fallback. No new structural gauge-transformation, irrep-homomorphism, or link-holonomy defect was established in the examined paths. That negative result is source coverage, not an executed certification of the complete architecture.

## Correctness findings

### CG inference caches can make the next training call fail

**Location:** `vfe3/model/cg_coupling.py:216-224`, consumed at `:244-255` and `:273-310`.

**Severity:** medium

**Evidence:** `_cast_buffers` executes `getattr(self, f"cg_{t}").to(dtype=dtype, device=device)` and stores the resulting tensors in `_cast_cache` without an inference-mode guard. The coordinator's `probes_config_cg.py`, result `probes_config_cg.json` under `cg_inference_cache`, constructed the normal float64-buffer module, set its learned path weights to 0.1, called it on float32 means inside `torch.inference_mode()`, and then called it with gradients enabled. The subsequent call raised `RuntimeError: Inference tensors cannot be saved for backward`. The module's `_apply` clears this cache at lines 211-214, but no conversion or device move occurs between these two forwards. The analogous comparison-RoPE cache explicitly excludes inference-mode builds in `relational_operators.py:2195-2198`.

**Fix:** Build cached constants with inference mode disabled, or avoid retaining casts created inside inference mode so that a later training call creates ordinary tensors.

**Falsifies:** The exact inference-then-training probe completes backward on a fresh module with nonzero path weights and a genuine dtype conversion, without manually clearing the cache or calling `.float()`/`.to()` between forwards.

**Priority:** P2.

**Trigger and reachability:** `CGCoupling(3, "so", [1, 3], ["l0", "l1"])`, float32 inputs, nonzero `path_weights`, and an inference-mode call before the first gradient-enabled call for that dtype/device. The model constructs this same module at `vfe3/model/model.py:520-526`; its registered intertwiners begin in float64. A reachable CG model uses `use_cg_coupling=True`, `gauge_group="so_n"`, `group_n=3`, `irrep_spec=[("l0", 1), ("l1", 1)]`, `embed_dim=4`, `n_heads=2`, and `gauge_tying="shared"`. This is a conditional public-API failure. Ordinary repository evaluation uses `no_grad`; this finding does not claim that the active training launcher crashes.

**Consequence:** Sampling or checking a loaded CG model under inference mode can make subsequent fine-tuning fail despite unchanged weights and valid inputs. A module-wide conversion happens to hide the issue by clearing or avoiding the conversion cache.

**Reproduction suggestion:** Retain the coordinator's four-dimensional test as a mode-transition regression. Cover both `passthrough` and `delta_full`, and require a backward pass after the inference call. Do not convert the module's buffers to the input dtype before the first call, because that removes the triggering conversion.

**Prior coverage:** No corresponding item was located in the October 20 audit or the inspected CG tests. The existing `_apply` invalidation covers device/dtype moves, not inference-created cache entries.

### The zero-weight CG fast path drops the configured covariance floor during evaluation

**Location:** `vfe3/model/cg_coupling.py:239-243`, `:305-309`; model dispatch at `vfe3/model/block.py:122-137`.

**Severity:** medium

**Evidence:** The public forward executes `if not torch.is_grad_enabled() and self.is_identity(): return mu, sigma` before checking `cg_covariance_mode`. In `delta_full`, the other branch calculates `sigma_out = sym(J Sigma J^T) + self.cg_covariance_floor * sigma`. With every path weight zero, its executed Jacobian is the identity. The coordinator's `probes_initial.json`, key `cg_train_eval`, records covariance diagonals `[1.5, 1.5, 1.5, 1.5]` with gradients enabled and `[1.0, 1.0, 1.0, 1.0]` under `no_grad`, with maximum difference `0.5`. The input covariance was the identity and the permitted configured floor was `0.5`. The existing `tests/test_cg.py:223-232` checks the zero-weight floor through `forward_moments`, so it does not exercise this early return.

**Fix:** Restrict the identity shortcut to passthrough covariance, or return `(mu, (1 + cg_covariance_floor) * sigma)` for the zero-weight `delta_full` case.

**Falsifies:** For a zero-weight module with a positive `cg_covariance_floor`, public `forward` returns identical moments with gradients enabled and disabled and agrees with `forward_moments` in both modes.

**Priority:** P2.

**Trigger and reachability:** Set `cg_covariance_mode="delta_full"`, `family="gaussian_full"`, `use_cg_coupling=True`, and a valid shared irrep tower. This is the initialized state, because `path_weights` starts at zero. The constructor accepts every finite positive floor, so the discrepancy is not restricted to the small default. At `cg_energy_weight=0`, `vfe_block` uses public forward. A capture containing `cg_moment_energy_rows` instead calls `forward_moments` directly, introducing a second route to different zero-weight covariance under no-grad execution.

**Consequence:** The same model state evaluates a different covariance update in training and evaluation at zero weights. This can change later attention routing and Gaussian decoding. With the default `1e-6` floor the difference is small; the reproduced larger floor establishes the general branch mismatch without implying a measured perplexity effect.

**Reproduction suggestion:** Construct `CGCoupling(3, "so", [1, 3], ["l0", "l1"], cg_covariance_mode="delta_full", cg_covariance_floor=0.5).double()`, apply it to a four-vector and identity covariance, and compare public forward inside and outside `no_grad`. Also compare the block's capture and captureless routes at zero path weights.

**Prior coverage:** The existing zero-weight test explicitly expects the documented floor, but only on `forward_moments`; no duplicate was located in the October 20 audit. This is not a complaint about leaving CG disabled by default.

### A future token can switch the entire minimal-agent call to float64 and alter earlier outputs

**Location:** `vfe3/model/relational_operators.py:1425-1449`; aggregated query status in `vfe3/model/minimal_agent.py:1369-1380`, `:1546-1551`.

**Severity:** low

**Evidence:** `minimal_agent_update` reads three call-wide flags and, if any is true, replaces all mean/covariance outputs with a float64 recomputation. No per-receiver selection preserves healthy prefix outputs. The coordinator executed `gauge-probe-suffix.py`; `gauge-probe-suffix.json` records a strictly causal three-token, one-head, width-two example. Only token 2's diagonal covariance changed from `[0.9, 1.1]` to `[10000, 1]`. The raw float32 means and covariances at tokens 0 and 1 remained bitwise equal. The call-wide status changed from `[false, false, false]` to `[false, true, false]`, and both returned prefix means and covariances changed by a maximum of `5.960464477539063e-08`.

**Fix:** Retain receiver-level failure/certificate masks and replace only receivers that need escalation, including affected receivers of a failed source, or use a fixed precision for this route independent of future inputs.

**Falsifies:** Re-running the attached probe shows bitwise-equal returned prefix moments when the suffix alone changes the escalation status, and analogous tests cover source failures and covariance-screen failures.

**Priority:** P3.

**Trigger and reachability:** `predictive_readout="minimal_agent"`, `minimal_agent_inference="exact"`, float32 input, a causal routing prior, and a certified projected-factorization policy. The probe uses valid finite positive variances, a healthy identity frame, full-rank two-dimensional query/content maps, and a positive channel noise. It changes no production configuration. The same call-wide replacement branch also handles factorization and output-covariance failures, but those triggers were not separately reproduced in this lane.

**Consequence:** Prefix independence is not bitwise across precision-branch changes. The observed effect is rounding-sized, about half a float32 epsilon at these values. No material information leakage, loss change, perplexity effect, or amplified operating-point error was demonstrated, so this must remain a low-severity numerical-contract issue.

**Reproduction suggestion:** Keep the attached single-case CPU probe, comparing the raw body and public return separately. It isolates precision selection from the causal attention formula: the raw prefix is unchanged, while the wrapper changes it.

**Prior coverage:** Related to the October 16 F41 covariance-escalation issue, whose kernel-readout fix remains visible in `_escalate_failed_sequences`: it uses per-receiver write-back at `relational_operators.py:1748-1754` and `:1761-1764`. This is the analogous behavior in the minimal-agent wrapper. The minimal-agent fallback tests currently require a whole-call float64 result, while their ordinary causality tests do not force the suffix to change the precision branch.

## Performance opportunities

No new performance finding is asserted by this lane. No profiling was performed. The October 20 pure-gauge per-edge broadcast finding was followed to the current per-sender `_sender_terms` contractions; the current code no longer stores a per-edge pure-gauge coefficient tensor. The low-rank and latent-link routes still have explicit pair/loop costs, but those costs alone are not a defect, and no new measured optimization is claimed here.

## Design and test improvements

The three reproductions identify focused missing coverage: CG mode transitions after inference-created dtype casts; zero-initialized public forward under no-grad and diagnostic capture; and prefix independence when an ill-conditioned suffix triggers minimal-agent escalation. These are regression tests for the findings above, not an additional audit panel or a request for training-scale experiments.

## Coverage and retained boundaries

Read and traced `vfe3/model/relational_operators.py` constructor/dispatch, chart and canonical-mean helpers, edge context, floor/ceiling selection, minimal-agent wrapper, and covariance recovery; `edge_links.py` factor generation, masking, per-sender pure-gauge contractions, mean/covariance corrections, receiver inference, and depth-loop code; `frame_actions.py`; `positional_phi.py`; and `cg_coupling.py`. Related reads covered `gradients/relational.py` score/readout dispatch, `model/block.py` CG/reference-coordinate dispatch, the relevant `model.py` construction, `gauge_covariant.py`, `exact_route.py`, and config guards.

Geometry coverage included the group representation/tying builders and compact-block certificates in `groups.py`; direct-sum construction and the live bracket-homomorphism check in `irreps.py`; the construction, residual checks, and admissible-path enumeration in `cg.py`; RoPE generation and covariance projection in `rope.py`; and dense/compact frame products, right-inserted RoPE, inverse-pair handling, and positional composition in `transport.py`. This was not a line-by-line proof of every high-rank SO/SU representation constructor, every numerical exponential/inverse kernel, or every mixed-head covariance contraction. The old BCH composition path is absent from the current model: `right_phi` is composed by the group-product builder, so no defect was filed against the retired BCH guard.

The strict route is present in source. `fully_gauge_covariant_overrides` selects reference-coordinate block/readout paths where needed and removes incompatible fixed-coordinate operations; the exact-route override selects frame-diagonal reference covariance, minimal-agent attention, and reference cue/result memory. Its retained optional frame-condition bound is an intentional training constraint, already distinguished in the route code. This lane does not reclassify that choice as a defect. Full-covariance and reference-diagonal paths were kept distinct from local diagonal projection throughout the review.

The live manuscripts read were `Research/manuscripts/Attention/GL(K)_attention_v2.tex`, especially `sec:framed_states`, `sec:attention_operator`, and the link construction; `GL(K)_supplementary_v2.tex` at `sec:heads_irreps`, `sec:latent_links`, and nonlinear/causal scope; `edge_latent_links.tex` at edge fields, covariance/causality, receiver profiling, and depth loops; and `minimal_agent_model.tex` at `ma:sec:model` and `ma:sec:readings`. The Research frame-coverage report, standing gauge theme/concept pages, and prior local audit/remediation tests supplied continuity only. No manuscript or wiki statement was used as proof that an executable branch works.

The source distinguishes bare vertex-frame transport from reference content links. Edge factors are computed from reference means, and the low-rank coefficient argument is masked before its matrix function. The pure-gauge control retains the shared content link; it is not required to have identity triangle holonomy. The latent mode changes receiver factors while sender factors remain amortized, and its depth-loop indices remain within the receiving prefix. No false defect was filed merely because these content links are not flat. No new whole-model equivariance measurement, primary-literature adjudication, or GPU result is claimed.

## Artifacts and execution provenance

The coordinator's CPU evidence is in `probes_initial.json` (`cg_train_eval`), `probes_config_cg.json` (`cg_inference_cache`), and `gauge-probe-suffix.json`. The first artifact records PyTorch `2.10.0.dev20251210+cu128`, CPU execution, and one thread. The suffix probe was written by this investigator at the coordinator's request and executed only by the coordinator. Its two-dimensional example and the four-dimensional CG examples have no training-scale workload.
