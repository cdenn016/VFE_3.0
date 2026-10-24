# Generative Memory Implementation Plan

> For agentic workers: execute the bounded tasks below using `superpowers:executing-plans`. Follow the repository's rules for batching implementation and focused verification. This plan does not require a fresh investigator or reviewer for each task, a full test suite, or concurrent numerical work. Checkboxes record future implementation work; none is marked complete by drafting this document.

**Goal:** Rename `prototype_agents` to `generative_memory` without changing existing saved models, then add a statistically defined cue/result memory with an explicit posterior and projection interface.

**Architecture:** A memory bank and its record kernel define the local generative model. An inference kernel computes a joint content/component posterior or a declared mean-field approximation; a separate projection supplies the Gaussian carrier consumed by existing attention. The first release keeps the current outer next-token training objective, while exposing the interfaces needed for later persistent model beliefs and observation-driven local learning.

**Tech stack:** Existing Python/PyTorch implementation, `C:/anaconda/python.exe`, current covariance families and block registry, tiny CPU float64 references, and the existing click-to-run configuration system. No new dependency is required.

**Specification:** The design contract in Sections 1–6 implements the architecture discussed with the author on October 24. Its mathematical constraints come from the [current investigation](../research/2026-10-24-mlp-generative-models/report.md), the [existing cue/result construction](../research/2026-10-20-neural-free-continuation/prototypes.md), and the [local-learning analysis](../research/2026-10-20-neural-free-continuation/local-learning.md). Later research-dependent phases are separated in Section 10.

**Status:** Draft plan only. Source inspection used HEAD `80c662445aabe44e7aa1529ab67de2d91dc93a0c` together with the current working files. The checkout contains intentional WIP, including launcher/configuration and local-learner changes. This document neither implements the rename nor changes those files.

## Global constraints

Preserve existing checkpoint parameter names, parameter registration order, and optimizer groups for the same-state model. Preserve all intentional WIP. Add newly introduced configuration fields to `VFE3Config`, `train_vfe3.py`, and `tests/profiles.py`, with historical defaults recorded where serialized configuration loading requires them. Existing launcher values remain the author's choices.

Use type annotations on every new function and mathematical docstrings for nontrivial formulas. Use the paper's notation for numerical objects. Tests use single-digit dimensions, layers, heads, and vocabulary sizes. CPU validation is serial with one numerical thread; no GPU workload is launched as part of drafting or ordinary CPU verification. GPU execution requires the existing availability/authorization check.

Do not edit the Research vault, the retired V3_Transformer repository, historical experiment artifacts, or the film WIP. Do not turn this implementation into a proof project or a new audit. Record implementation changes in the daily `docs/<date>.md` log. Use one isolated checkout only if needed to protect overlapping WIP at execution time; creating this plan does not require one.

## 1. Scope and release sequence

The executable first release has two independently useful parts. The rename and diagnostic migration preserve the current model's computation. The subsequent feature adds the cue/result kernel, structured inference results, and explicit Gaussian projection. These parts must remain distinguishable in tests and commits so that a naming migration cannot conceal a numerical change.

The first release makes structured posteriors available through an explicit API. It does not claim that current attention propagates them: the existing stack continues to consume a projected Gaussian carrier. Full component output is opt-in for small cases. Retained mixtures across layers, bounded component selection, and local predictive learning follow only when their consuming operations and probabilistic semantics have been specified.

The first feature supports a dense full-covariance reference path, then a scalable diagonal-in-reference carrier path. It does not require a dense covariance for every token/template pair to be resident simultaneously. Same-state behavior retains its existing diagonal, block, and dense contracts throughout.

## 2. Naming and compatibility contract

Use **generative memory** for the persistent family of content models and its inference operation. A bank entry is a **memory component** or **model component**. A token remains an agent. The name does not imply that a component is itself an independently updating agent, a semantic concept, or a posterior over learned parameters.

| Current public surface | Canonical surface | Compatibility behavior |
|---|---|---|
| `block_mlp_mode="prototype_agents"` | `block_mlp_mode="generative_memory"` | Accept the old value as an alias with identical same-state behavior. |
| `vfe3.model.prototype_agents` | `vfe3.model.generative_memory` | Keep the old module as an explicit re-export shim. |
| `PrototypeAgentBlockMLP` | `GenerativeMemoryBlockMLP` | Old class name aliases the same class object. |
| `prototype_exact_posterior` | `same_state_exact_posterior` | Old function name remains an alias with its original signature and result fields. |
| `prototype_mean_field_posterior` | `same_state_mean_field_posterior` | Preserve coordinate order, sweep count, and returned factor pair. |
| `prototype_elbo` | `same_state_product_elbo` | Preserve the old function's numerical meaning; the new name identifies the product recognition law. |
| `ExactPrototypePosterior` / `MeanFieldPrototypePosterior` | `SameStateExactPosterior` / `SameStateMeanFieldPosterior` | Preserve existing NamedTuple fields and arity; legacy names alias them. |
| `PROTOTYPE_*` numerical constants | `MEMORY_*` | Export legacy constant names from the shim; internal code uses the canonical module. |
| New `prototype_*` report labels | `memory_*` labels | New writers use canonical terminology; readers continue to accept historical files and schemas. |

Retain generic `block_mlp_*` configuration keys because they configure the existing architectural slot and several other implementations. `template` remains an accurate term for a component's stored distribution; this plan does not mechanically rename every occurrence of that word. The separate `gaussian_template` mode keeps its identity and behavior.

The additional model selector is `block_memory_model`, with values `"same_state"` and `"cue_result"`. Its default and historical introduction value are both `"same_state"`. An old selector combined explicitly with `block_memory_model="cue_result"` is rejected as contradictory; it must not silently reinterpret a saved prototype model. Missing new fields in an old configuration recover the historical same-state model.

Canonicalization belongs at configuration boundaries. Both direct `VFE3Config(...)` construction and `config_from_serialized` must apply the same narrow rule before mode-dependent validation. Direct block-factory calls also accept the old selector. Runtime branches should compare canonical values rather than maintain two competing implementations.

Do not rewrite stored configuration dictionaries on disk. New serialization uses the canonical name and records the resolved model kind. Saved-run comparisons normalize only these specified aliases and introduction defaults. All unrelated consequential configuration differences remain differences. A cue/result model is never checkpoint-compatible with a same-state model merely because their carriers have the same shape.

For same-state checkpoints, registered parameters remain `block_mlps.<layer>.nu`, `.log_lambda`, and `.log_pi`, in their current registration order. Do not nest the existing module under a new registered `bank` or `kernel` child. Preserve the no-weight-decay treatment of `log_lambda` and `log_pi`, calibration behavior, EM activation, and optimizer parameter-name groups.

## 3. Statistical and numerical contract

### 3.1 Preserve the same-state model

The renamed model retains the current exact and mean-field calculations. Its local record completion is

$$
A\sim\operatorname{Cat}(\pi),\qquad X\mid A=a\sim\mathcal N(\nu_a,\Lambda_a),\qquad O\mid X,C\sim\mathcal N(X,C).
$$

Here the incoming mean is the record value and the covariance is declared side information. This completion is a local modeling choice. It does not establish that the upstream network generated an independent observation, or that repeated depth operations optimize a single global ELBO.

Renaming must preserve exact output values, gradients, RNG consumption, covariance passthrough identity, dropout behavior, finite-CAVI ordering, and existing initialization. In particular, do not refresh the final CAVI labels merely to make the result look more converged: that is another coordinate update and changes the legacy computation.

### 3.2 Add the cue/result model

Use the following parameterization of the already developed joint model:

$$
X\mid A=a\sim\mathcal N(\nu_a,L_a),\qquad
Y\mid X,A=a\sim\mathcal N\bigl(\omega_a+F_a(X-\nu_a),Q_a\bigr),\qquad
O\mid X,C\sim\mathcal N(X,C).
$$

The first implementation has equal cue/result dimension K, diagonal positive `L_a` and `Q_a` in the declared reference chart, and a normalized categorical prior. It represents the joint covariance through

$$
B_a=F_aL_a,\qquad V_a=F_aL_aF_a^\top+Q_a.
$$

Thus the Schur complement is Q_a and is positive by construction. F_a is the conditional regression coefficient of this joint model; its induced receiving gain is F_a L_a(L_a+C)^{-1}. The first release still learns these parameters through the task loss. It does not claim that this is already a local sufficient-statistic learner.

Conditioning on O=xi gives

$$
\log w_a=\log\pi_a+\log\mathcal N(\xi;\nu_a,L_a+C)-\log Z,
$$

$$
m_{X,a}=\nu_a+L_a(L_a+C)^{-1}(\xi-\nu_a),\qquad
P_{X,a}=(L_a^{-1}+C^{-1})^{-1},
$$

$$
m_{Y,a}=\omega_a+F_a(m_{X,a}-\nu_a),\qquad
P_{Y,a}=Q_a+F_aP_{X,a}F_a^\top.
$$

Compute the dense posterior through Cholesky solves/information form, and the diagonal posterior elementwise. Do not evaluate output covariance by subtracting two nearly equal dense matrices. Marginal covariance includes the complete between-component term in whichever output projection was requested. An additional residual addition is not part of this model.

The independent reference test forms the joint distribution of `(O,Y)` and conditions it directly. This tests a different algebraic route from the implementation, including noncommuting matrices and negative/rotating regression maps.

### 3.3 Regression structure and initialization

`block_memory_regression_rank=None` selects a free dense F_a, an explicit expensive reference option. A positive rank r selects `F_a = I + U_a V_a^T`, with r at most K. This restricts the departure from identity while allowing an initial pass-through regression. It differs from a rank-r F_a, which would discard directions even at initialization.

For the dense form, initialize F_a to identity. For the structured form, initialize U_a to zero and V_a with a small seeded normal draw of standard deviation `0.01 / sqrt(K)`. Do not initialize both factors to zero, which blocks their initial gradients. Initialize omega_a to nu_a. Initialize the diagonal Q_a to `block_memory_result_noise_ratio * L_a`, with the proposed ratio default `0.01`; this is an initialization choice to evaluate, not a derived optimum. Existing working-variance calibration initializes nu_a and L_a first, then these new parameters. Old same-state initialization consumes exactly the same random draws as before.

The dense coefficient is `regression`; the structured coefficients are `regression_left` and `regression_right`. New parameters are registered directly on the new cue/result block, with `omega` and `log_result_noise`. Its checkpoint schema is distinct and strict. Existing `nu`, `log_lambda`, and `log_pi` names describe the cue bank, but an old checkpoint is not automatically promoted into this larger model.

### 3.4 Compute and allocation policy

Retain disabled autocast around the statistical solves, IEEE fp32 matrix multiplication, float64 when an input is float64, and float32 otherwise. Preserve the existing 32 MiB pair-tile setting initially, while documenting that it bounds individual pair tensors rather than total peak memory.

Normal inference streams components using stable log-sum-exp and centered second-moment accumulation. It may retain O(number_of_tokens * M) label weights, but must not retain O(number_of_tokens * M * K^2) conditional covariance storage by default. Recomputed tiles remain pure functions; do not add mutable callbacks or graph-bearing module caches inside checkpointed computations.

Full component output is an explicit diagnostic/API request, guarded by a proposed 64 MiB returned-component allocation cap. Estimate the requested means, conditional covariance layout, labels/identifiers, and any additional dense representation before allocating. This cap is not a promise about autograd or process peak memory. Larger requests must be explicit. Top-k selection, residual buckets, and persistent mixture compression are later work, not hidden inside exact inference.

## 4. Interfaces and ownership

Keep the legacy low-level result structures unchanged through aliases. Introduce separate new result types in `vfe3/model/memory_types.py` so new semantics do not change existing tuple unpacking.

| Type | Fields and meaning |
|---|---|
| `MemoryMoments` | `mean: Tensor`, `covariance: Tensor | None`, `covariance_layout: str`; the selected marginal moments, not a claim that the marginal is Gaussian. |
| `MemoryComponents` | `component_ids: Tensor`, `mean: Tensor`, `covariance: Tensor`, `covariance_layout: str`; conditional moments with shapes `(M,)`, `(..., M, K)`, and diagonal/dense/compact-block covariance as explicitly tagged. |
| `ExactMemoryResult` | `moments: MemoryMoments`, `log_weights: Tensor`, `log_evidence: Tensor`, `components: MemoryComponents | None`, `model_kind: str`, `coordinates: str`, `marginal_projection: str`. |
| `MeanFieldMemoryResult` | `moments: MemoryMoments`, `log_responsibilities: Tensor`, `sweeps: int`, `model_kind: str`, `coordinates: str`; describes the actual product recognition law returned by CAVI. |

The result types are frozen dataclasses. Accepted marginal-projection labels are `"full"`, `"diagonal"`, and `"blocks"`; a separate carrier policy records covariance passthrough. Conditional covariance layout and marginal projection are separate because diagonal conditional components can produce dense marginal covariance. `components=None` means conditional tensors were not retained, not that the inferred posterior was factorized.

When `component_output="all"` is requested, retain the actual conditional covariance needed to represent the exact local posterior. In cue/result mode this generally means dense conditional matrices even when the requested marginal output is diagonal; include that cost in the allocation check. Do not store only their diagonals and label the resulting components exact. Compute new log-weight fields directly from stabilized logits, rather than taking the logarithm of probabilities that may already have underflowed; the compatibility probability fields remain unchanged.

The coordinate tag is supplied by the caller that knows the chart. Assigning the tag performs no transform. The cue/result adapter accepts only `"reference"` in the first release. The legacy same-state wrapper can describe its existing local or reference evaluation.

The public block API is:

```python
def forward_posterior(
    self,
    mu: Tensor,
    sigma: Tensor,
    *,
    coordinates: str,
    component_output: str = "none",
    component_max_bytes: int = 64 * 1024**2,
) -> ExactMemoryResult | MeanFieldMemoryResult: ...
```

`component_output` accepts `"none"` or `"all"`. Requesting component output from mean-field inference raises a clear error; the product law is represented without M duplicate Gaussian tensors. `forward_posterior` is pre-dropout, has no learning side effect, and returns no reference to a mutable future parameter snapshot. The compatibility `posterior(...)` method remains available with its original result API.

The existing `forward_moments(...) -> BlockMLPMomentResult` remains the adapter consumed by `block.py`. For same-state mode, it preserves the original dropout and covariance contract. For cue/result mode, it uses exact moments, posterior covariance, and zero dropout. A small helper in `memory_projection.py` projects retained `MemoryComponents` for reference checks and explicit API users; the production streamed kernel computes the same sufficient statistics without materializing all components.

The new kernel in `cue_result_memory.py` exposes:

```python
def cue_result_exact_posterior(
    xi: Tensor,
    C: Tensor,
    nu: Tensor,
    omega: Tensor,
    log_lambda: Tensor,
    log_result_noise: Tensor,
    log_pi: Tensor,
    *,
    regression: Tensor | None,
    regression_left: Tensor | None,
    regression_right: Tensor | None,
    output_covariance: str,
    component_output: str = "none",
    component_max_bytes: int = 64 * 1024**2,
    tile_bytes: int = 32 * 1024**2,
) -> ExactMemoryResult: ...
```

Exactly one regression representation is provided. C is diagonal or dense; the initial cue/result kernel rejects compact blocks. Output covariance is `"dense"` or `"diagonal"`. The dense case preserves full local posterior moments; the diagonal case explicitly projects. A dense C can still be supplied to the diagonal-output reference function. This does not make local-diagonal storage frame-covariant.

Keep `GenerativeMemoryBlockMLP` as the renamed existing same-state implementation in `block_mlp.py`. Place the new `CueResultMemoryBlockMLP` in `generative_memory_block.py`, importing `BlockMLPMomentResult` from `block_mlp_types.py`. Move only that NamedTuple into the small shared module and re-export it from `block_mlp.py`; its fields and arity are unchanged. This prevents a circular import without reorganizing unrelated MLP implementations. The existing factory chooses the class from the canonical mode and `block_memory_model`.

Extend `build_block_mlp` with keyword arguments `memory_model: str = "same_state"`, `memory_regression_rank: int | None = None`, and `memory_result_noise_ratio: float = 0.01`. Add them to the canonical memory registration's `factory_options`; other factories do not receive them. Pass them explicitly from `model.py`. Extend `block_mlp_build_metadata` with `memory_model` and `memory_regression_rank` using the same defaults, and pass the resolved values at every call site, including `run_artifacts.py`. Use a narrow memory-specific resource calculation there instead of reusing the current registration's same-state FLOP formula for cue/result. Add optional metadata fields for model kind, regression rank, and whether the FLOP estimate covers the selected inference/layout; readers of older metadata supply same-state defaults. Same-state counts remain unchanged, and new counts are checked against the built module's trainable parameter count.

Do not extend `BeliefState` in the first release. Structured results are explicit transient API values, not an unused state channel. The current stack's projection boundary is documented in `block.py`. A future wrapper for persistent state is described in Section 10.

## 5. Configuration and reachable paths

| New field | Default | Contract |
|---|---|---|
| `block_memory_model` | `"same_state"` | `"same_state"` or `"cue_result"`; historical introduction value is same-state. |
| `block_memory_regression_rank` | `None` | Dense F when None; integer 1 through K for identity plus low-rank correction. Relevant only to cue/result. |
| `block_memory_result_noise_ratio` | `0.01` | Finite positive initialization ratio Q/L; relevant only to cue/result. |

All existing mode-specific fields keep their established defaults. Do not silently switch `block_mlp_inference` from mean field to exact when cue/result is selected. Validate the requested combination instead.

Reject booleans as rank values, nonintegral ranks, ranks outside 1 through K, and nonfinite or nonpositive noise ratios when cue/result is active. The covariance and regression shapes are validated against the same K and M. Populate `INTRODUCED_AFTER_CAA0114` for the new fields; do not change the historical snapshot table.

For the first cue/result integration require `attention_mode="projected_predictive"`, `block_coordinates="reference"`, `block_mlp_inference="exact"`, `block_mlp_covariance="posterior"`, `block_mlp_dropout=0.0`, and `block_mlp_em_weight=0.0`. Initially admit `family="gaussian_full"`; add `gaussian_frame_diagonal` in its own task. Retain that family's other existing restrictions. No cue/result mean-field or EM option is advertised until its objective and updates are implemented.

LayerNorm, attention, decoder, and outer learning choices remain independently configurable within their existing supported paths. A subsequent norm changes the propagated law and does not preserve an exact mixture interpretation. This is stated in diagnostics and documentation rather than hidden by the memory module.

The frame-diagonal integration must implement the conversion and validation path, not simply delete the existing prohibition. Its carrier stores diagonal reference covariance, so project the memory result in the reference chart and return through the existing adapter. The full-covariance route remains available as the local reference calculation. Arbitrary local-diagonal projection is not the scalable covariant route proposed here.

The new rank and noise fields are reported inactive outside cue/result. Resource estimates depend on model kind and regression representation: same-state counts remain unchanged; cue/result counts include omega, output noise, and either M*K*K dense regression entries or 2*M*K*r factor entries. Report the identity term as structure, not a learned parameter.

## 6. Diagnostics and statistical claims

The rename and corrected interpretation require a versioned diagnostic schema. New writers use `memory_*` terminology. Historical `prototype_*` files remain readable and are identified as historical schemas; do not append new meanings under old CSV headers.

Report local `log_evidence`; analytic exact-joint free energy `-log_evidence` when applicable; the separately evaluated product-law ELBO/free energy; and its gap relative to local evidence. Exact-joint free energy describes the inferred joint law even when only its moments were emitted. It is not the free energy of the moment-matched Gaussian carrier.

For mean field, evaluate the actual returned pair of Gaussian and categorical factors. For legacy passthrough or dropout, state that the inference diagnostic precedes the output transform. Do not recompute a different posterior and call it the emitted state. Retain historical product-law numbers under reader aliases with their original meaning.

Compute population entropy, mean conditional entropy, and their difference using the same valid-token mask and normalized weights. Call the difference empirical position/component mutual information. Also report within-component and between-component covariance trace in the evaluated chart. These are diagnostics, not objectives that automatically reward useful specialization. Do not claim a closed-form KL from a moment-matched Gaussian to a general Gaussian mixture; any future numerical estimate must be labeled separately.

Use `memory_calibration.json` for new calibration output and a new canonical diagnostic CSV schema. Readers prefer the new artifact when present and otherwise read the legacy filename. Old files are never rewritten. The cue/result model receives its own resource metadata and calibration description.

## 7. Implementation tasks

### Task 1: Rename without changing the same-state computation

**Files:** `vfe3/model/prototype_agents.py`, new `vfe3/model/generative_memory.py`, `vfe3/model/block_mlp.py`, `vfe3/config.py`, `vfe3/config_history.py`, `vfe3/model/block.py`, `vfe3/model/model.py`, `vfe3/train.py`, `vfe3/posthoc/feature_probes.py`, `vfe3/posthoc/interventions.py`, `ablation.py`, `train_vfe3.py`, and `tests/profiles.py`. Add `tests/test_generative_memory_compatibility.py`.

**Produces:** The canonical names and aliases in Section 2, plus `block_memory_model="same_state"` resolved consistently at construction/load boundaries. The old mode still runs the same algorithms.

- [ ] Capture tiny legacy outputs, gradients, initialization state, and optimizer group names before moving definitions. Use a deterministic test-owned configuration and compare within one interpreter/build.
- [ ] Add `block_memory_model` with its same-state default to configuration, the launcher dictionary, explicit test baselines, and the introduction-value table. Add tests for direct configuration construction, strict/lenient serialized loading, direct factory lookup, legacy imports, contradictory old-selector/cue-result input, and historical missing-field resolution.
- [ ] Move and rename definitions with explicit re-exports. Migrate internal imports and string guards, including EM activation, working-variance calibration, diagnostic applicability, and ablation requirements. Update tests that monkeypatch module globals to patch the canonical module.
- [ ] Preserve state-dict key order and exact same-state initialization. Verify that old and new selectors produce identical outputs/gradients with the same state and RNG, including the existing dropout option.
- [ ] Check a tiny saved optimizer state: resume each spelling, take one identical step, and compare parameters and optimizer state. Verify calibration and nonzero EM actually execute, rather than merely accepting their settings.

Suggested commit after focused verification: `refactor: rename prototype inference to generative memory`.

### Task 2: Preserve saved-run and analysis compatibility

**Files:** `vfe3/checkpoint_average.py`, `vfe3/run_artifacts.py`, `vfe3/legacy_config.py`, `vfe3/paper/runs.py`, `paper_arms.py`, `vfe3/viz/feature_figures.py`, `vfe3/viz/report.py`, and affected feature/logging readers. Extend `tests/test_checkpoint_average_20261021.py`, `tests/test_prototype_train_logging_20261009.py`, and the new compatibility tests.

**Consumes:** Task 1 canonicalization. **Produces:** Narrow semantic comparison of old/new equivalent configurations and readers for both artifact-name generations.

- [ ] Test that otherwise identical old/new selector snapshots compare compatible, while different model kind, inference mode, covariance policy, parameter shapes, or regression rank remain incompatible.
- [ ] Apply the narrow normalization to raw-config comparisons in checkpoint averaging; retain the raw provenance. Route other loaders through the existing serialized-config path rather than inventing a second general migration system.
- [ ] Add legacy-reader fixtures for calibration and diagnostic files. Update new report/probe names and experiment applicability predicates. Preserve the meaning and location of historical rows.
- [ ] Verify that a same-state checkpoint still loads through training, posthoc, and paper-run readers. Do not rewrite old run directories, copied scratch configurations such as `zzzzz.py`, or historical documents.

Suggested commit: `fix: preserve saved-run compatibility for memory naming`.

### Task 3: Expose structured posterior and explicit projection

**Files:** New `vfe3/model/memory_types.py`, `vfe3/model/memory_projection.py`, and `vfe3/model/block_mlp_types.py`; modify canonical same-state kernels and block adapters. Add `tests/test_memory_posterior_contracts.py`.

**Consumes:** Renamed numerical kernels. **Produces:** Section 4 dataclasses and `forward_posterior`, while preserving the original `posterior` and `forward_moments` contracts.

- [ ] Test the symmetric two-component case xi=0, C=1, nu=(-3,3), Lambda=0.25, equal priors. Exact conditional means are (-2.4,2.4), conditional variance is 0.2, and marginal variance is 5.96. Symmetric mean field has variance 0.2 and is represented as a product law.
- [ ] Add opt-in component extraction without altering normal streamed accumulation. Verify selected-layout moment projection against independent direct component summation and test covariance passthrough separately.
- [ ] Test that `component_output="none"` returns no component arrays; that `"all"` checks allocation before materialization; and that mean-field component requests are rejected. Test dense between-component spread even when conditional covariance is diagonal, and retain dense cue/result conditional components when only the marginal output is projected to diagonal.
- [ ] Keep exact/mean-field result types distinct and label their chart and projection. Reject invalid layout tags and tensor shapes at the API boundary.
- [ ] Verify that requesting diagnostic components does not change the normal output, gradients, or next-call state, and that the block does not retain an autograd graph in a module attribute.

Suggested commit: `feat: expose memory posterior and projection contracts`.

### Task 4: Implement the cue/result kernel and independent reference

**Files:** New `vfe3/model/cue_result_memory.py`; add `tests/test_cue_result_memory_reference.py` and `tests/test_cue_result_memory_gradients.py`. Reuse only the small numerical helpers that the implementation actually needs.

**Consumes:** Section 4 types. **Produces:** `cue_result_exact_posterior` with diagonal/dense record covariance, dense/structured regression, and explicit output covariance projection.

- [ ] Build a K=2, M=3 independent joint-Gaussian reference using the full `(O,Y)` covariance. Include correlated C, unequal component covariances, nonuniform weights, a rotation, and a negative response direction.
- [ ] Implement the conditioning formulas and positive-sum covariance, with shape/type validation and normalized evidence. Use stable streamed label normalization and centered second moments; retain all mixture spread.
- [ ] Compare dense F with `I+UV^T` for matched matrices. Compare diagonal and dense C on the same diagonal inputs. Verify that a single component reduces to ordinary Gaussian conditioning and that B=0 gives input-independent component means with still input-dependent component weights.
- [ ] Test covariance positivity on the declared finite fixtures and gradients with respect to xi, C, nu, omega, variance parameters, prior logits, and regression parameters using float64 finite differences. Check zero/nonzero initial factor gradients for the chosen identity-plus-low-rank initialization.
- [ ] Check tiled/untiled values and gradients with several tile boundaries, noncontiguous inputs, and extreme but finite variances in the existing supported numeric range. No silent jitter or covariance repair is introduced.

Suggested commit: `feat: add joint cue-result generative memory`.

### Task 5: Integrate the dense reference path and training groups

**Files:** New `vfe3/model/generative_memory_block.py`; modify `block_mlp.py`, `block.py`, `model.py`, `train.py`, `config.py`, `config_history.py`, `train_vfe3.py`, and `tests/profiles.py`. Add `tests/test_generative_memory_integration.py` and extend launcher/config isolation tests.

**Consumes:** Tasks 1–4 APIs. **Produces:** A reachable cue/result block on `gaussian_full` in reference coordinates, with strict configuration and checkpoint behavior.

- [ ] Add the two remaining new fields with Section 5 defaults and historical values, then wire the factory and metadata arguments from Section 4 without changing same-state registered prefixes. Check exact parameter counts and the declared coverage of FLOP estimates for both model kinds and regression representations.
- [ ] Implement initialization/calibration from Section 3.3. Test that old same-state RNG consumption is unchanged and that cue/result calibration observes only the permitted input batch.
- [ ] Place cue means, result means, and regression coefficients in the block-learning-rate group. Give `log_lambda`, `log_pi`, and `log_result_noise` zero weight decay. Verify each new trainable parameter is present once and receives the intended task gradient.
- [ ] Enforce the initial combination restrictions. In particular, reject cue/result mean field, nonzero EM weight, nonzero dropout, passthrough covariance, and unsupported layouts instead of silently changing or ignoring them.
- [ ] Run a tiny two-layer forward/backward and strict save/load round trip. Confirm the next attention operation consumes the projected posterior moments and that no unused component state is carried as if it were persistent.
- [ ] Test passive frame covariance of the full-reference block under a nonorthogonal invertible change of local frame. Hold the reference model and records fixed; compare transformed means/covariances and invariant weights/evidence.

Suggested commit: `feat: integrate cue-result memory in reference blocks`.

### Task 6: Add the scalable reference-diagonal carrier path

**Files:** `block.py`, `config.py`, relevant frame-diagonal family helpers, and `generative_memory_block.py`; extend `tests/test_generative_memory_integration.py` with frame-diagonal cases.

**Consumes:** Dense reference path. **Produces:** `gaussian_frame_diagonal` support for cue/result memory under the existing family's remaining restrictions.

- [ ] Trace the actual frame-diagonal adapter and pass its stored reference variances directly to the kernel. Compute the exact diagonal of within-component covariance plus the exact diagonal of between-component spread.
- [ ] For structured F, calculate the requested diagonal without constructing all dense component covariances. Verify it against the dense K=3 reference, including the identity and cross terms in `(I+UV^T) P (I+VU^T)`.
- [ ] Extend only the validation combinations now supported; retain edge-link, positional, decoder, and other family restrictions already enforced.
- [ ] Compare the scalable block with the dense block followed by explicit reference-diagonal projection. Check invariance under passive local frame changes and confirm that two successive steps match the explicitly projected reference computation, not an unprojected mixture computation.
- [ ] Keep arbitrary `gaussian_diagonal` local storage and compact-block cue/result integration outside this new validated combination. They remain separate extensions if later needed.

Suggested commit: `feat: support generative memory with reference-diagonal carriers`.

### Task 7: Correct and extend diagnostics

**Files:** `generative_memory.py`, `generative_memory_block.py`, `block_mlp.py`, `train.py`, posthoc/viz readers, and the existing diagnostic tests. Add `tests/test_memory_diagnostic_semantics.py`.

**Consumes:** Explicit inferred law and projection metadata. **Produces:** Canonical diagnostic schema with the meanings in Section 6.

- [ ] Retain the legacy product ELBO function as an alias and introduce explicit canonical labels. For the symmetric case in Task 3, verify exact-joint free energy equals negative log evidence while the product-law gap is approximately 27.102745803244325 nats.
- [ ] Test that covariance passthrough and dropout do not make an inference diagnostic masquerade as a score of the emitted carrier. Test the actual finite-CAVI factor pair without an extra update.
- [ ] Compute conditional/population entropy and empirical mutual information with masking. Identical per-position weights give zero mutual information; distinct concentrated assignments can give positive mutual information even when conditional entropy is low.
- [ ] Report within/between covariance contributions consistently with the selected chart/projection and identify the local observation model. Do not invent a global network ELBO metric.
- [ ] Test new writing and old reading, including resumed runs that retain their historical files and begin a distinct new diagnostic schema.

Suggested commit: `fix: distinguish memory posterior and carrier diagnostics`.

### Task 8: Document and prepare the matched comparison

**Files:** `README.md`, current configuration/registry descriptions, the implementation result note, daily log, and a new click-to-run experiment configuration under `docs/experiments/` using existing training/evaluation entrypoints.

**Produces:** Accurate user-facing terminology, a reproducible comparison specification, and a bounded completion record. This task prepares experiments; it does not launch training under the plan-drafting request.

- [ ] Document the rename, aliases, two model kinds, statistical meaning of the incoming covariance, posterior/projection distinction, numerical cost, supported paths, and the absence of a whole-stack ELBO claim.
- [ ] Freeze the actual baseline configuration at experiment preparation time. Keep tokenizer/data split, attention/readout, normalization, decoder, LR schedule horizon, stopping point, and evaluation procedure explicit. Do not copy the stale K=256 value from earlier notes into the current experiment.
- [ ] Define a parameter-matched comparison of coordinate MLP, same-state memory, and cue/result memory. Use a common supported covariance/coordinate route for the primary comparison. A covariance-family comparison is a separate intervention. Record any unavoidable mismatch instead of attributing its effect to the memory alone.
- [ ] Record model/data seeds and parameter counts, held-out CE/PPL, actual runtime and peak memory when measured, mixture uncertainty, calibration, and predictive contribution. Predefine the training budget and seed pairs before collecting results; no numerical superiority threshold or sample-size guarantee is invented here.
- [ ] Summarize exactly which CPU checks passed, which GPU/training checks were not run, and the remaining research-dependent phases. Publication or canonical checkout synchronization follows only the user's corresponding Git instruction.

Suggested commit: `docs: describe generative memory and matched evaluation`.

## 8. Verification commands and execution budget

Tests are added beside their owning implementation tasks. Run the affected tiny subset when its code is ready, and batch related fixes before the next run. Do not rerun a passing subset unless a later change affects it. The combined final subset below is the one integration pass; substitute a narrower list if the earlier results remain valid and there is no cross-module concern.

```powershell
$env:VFE3_TEST_DEVICE = 'cpu'
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
& 'C:/anaconda/python.exe' -m pytest -q `
  tests/test_generative_memory_compatibility.py `
  tests/test_memory_posterior_contracts.py `
  tests/test_cue_result_memory_reference.py `
  tests/test_cue_result_memory_gradients.py `
  tests/test_generative_memory_integration.py `
  tests/test_memory_diagnostic_semantics.py `
  tests/test_prototype_agents_mlp_20261006.py `
  tests/test_prototype_agents_reference_20261006.py `
  tests/test_cavi_em_20261010.py `
  tests/test_t1_t2_minimal_agent_elbo_20261012.py `
  tests/test_prototype_train_logging_20261009.py `
  tests/test_checkpoint_average_20261021.py `
  tests/test_gauge_block_mlp_integration.py `
  tests/test_block_mlp_launchers.py `
  tests/test_config_isolation.py
```

Keep existing test filenames during the initial migration; their historical names do not change the public model name. New tests set `torch.set_num_threads(1)` and use the test-owned profiles. A CPU-only command does not establish GPU behavior. GPU parity/performance checks, when scheduled, use the CUDA interpreter and device environment explicitly and report collected/executed counts.

The full `run_cpu_tests.py` lane is not required automatically. The rename affects several consumers, so the targeted checkpoint, configuration, optimizer, and logging coverage is necessary; unrelated numerical suites are not. This is a multi-module feature, larger than a small remediation. Deliver the compatibility migration first, then the inference feature, rather than extending the scope with an additional audit or unplanned benchmark campaign.

## 9. Review focus and acceptance criteria

| Failure condition | Required behavior | Owning task |
|---|---|---|
| Old saved config omits new fields or uses the old selector | Reconstruct same-state behavior and optimizer groups exactly. | 1–2 |
| A flag is accepted but its EM/calibration/reporting branch no longer runs | Canonicalization preserves reachability; explicit execution tests catch this. | 1, 5 |
| Structured output changes allocation or checkpointed autograd behavior | Normal path remains streamed; full component output is explicit and bounded. | 3–4 |
| Diagonal conditional covariances hide dense mixture spread | Full reference retains the spread; projection is explicit and numerically checked. | 3, 6 |
| Diagnostic labels refer to a different law from the one evaluated | Schema identifies exact joint, mean field/product, and emitted-carrier transformations. | 7 |

The first release is complete when legacy same-state behavior and saved-run loading remain usable; cue/result inference agrees with independent tiny Gaussian conditioning; its gradients pass the focused checks; both full-reference and reference-diagonal paths are reachable under documented settings; no structured state is advertised as persistent when a consumer ignores it; and the documentation identifies the remaining modeling and performance obligations.

## 10. Dependent phases toward the full agent implementation

These phases are part of the architectural roadmap, but are not disguised as completed derivations. Each has a concrete entry condition and deliverable. The executable first release does not wait for them.

### Persistent joint model/content state

Introduce an `AgentState` wrapper containing the existing carrier, a structured model/content law, and its model-space identity. Before wiring it through the stack, choose and specify either one shared model space or a normalized joint transition `K(a_next, x_next | a, x, context)` between identified spaces. A categorical transition alone is insufficient. Specify whether depth is a generative transition or an inference iteration and how the incoming record is used.

The delivery includes all reconstruction points in `stack.py` and `model.py`, and consumers that use the retained state. The decisive small test compares two rounds against explicit joint enumeration and demonstrates an input where component correlation changes the second-round result relative to moment collapse. Copying unused weights through the stack does not meet this gate.

### Component-aware communication and bounded mixtures

Extend the selected communication kernel to consume joint component/content state, or expose an explicit projection before communication. Peer/source labels and memory/model labels have separate meanings. For a retained mixture path, define component growth and compression before optimizing it. A bounded version may keep selected components plus a residual bucket with its mass and first two moments; report that this coarsens labels and loses higher-order structure.

Maintain an untruncated tiny reference path for the declared finite model. Compare bounded results to it at every approximation boundary. Exact local one-head conditioning or sequential moment matching must not be labeled unrestricted exact multi-head inference.

### A normalized predictive model and local learning

Specify a directed causal model of latent carriers, memory choices, communication, and normalized token observations. Give every node one normalized conditional and state the recognition family. Before using local updates in place of outer backpropagation, show that observed-token evidence reaches each parameter group through the relevant posterior factors, including links and cross-covariances.

Use prefix-only forecast, frozen predictive scoring, then token assimilation. The existing contextual-emission responsibility contrast and probit-sensor work are continuation sources. A detached reconstruction update is not a substitute for predictive credit. The deliverable includes one tiny identifiable example with nonzero token-to-link and token-to-memory learning signals, plus the exact objective optimized by the deployed finite inference schedule. Do not claim exact conditional-likelihood learning from approximate posterior messages without accounting for their approximation.

### Statistical scale control

Integrate the existing reference-relative KL-budget candidate only after identifying the local coordinate target and a fixed co-transforming reference within a declared objective. Its geometric interpolation of densities is a constrained inference result, not a post-hoc theorem for clipping arbitrary block outputs. Preserve the distinction between a fixed reference that bounds absolute relative scale and a moving reference that only bounds individual steps.

The delivery includes independent tiny constrained-update checks and a separate predictive comparison with the current normalization. No LayerNorm equivalence or PPL benefit is assumed. Gaussian formulas are an implementation specialization; the underlying interfaces remain probability laws and normalized kernels.

## 11. Sources and continuity

The [October 24 probabilistic memo](../research/2026-10-24-mlp-generative-models/probabilistic-model.md) gives the posterior/diagnostic distinction and evidence-accounting qualification. The [geometry memo](../research/2026-10-24-mlp-generative-models/gauge-and-species.md) gives the record-kernel requirement and the failure of arbitrary local-diagonal projection. The [October 20 prototype analysis](../research/2026-10-20-neural-free-continuation/prototypes.md) already formulates the cue/result model and its expressive advantage; this plan implements that existing construction. The [normalization analysis](../research/2026-10-20-neural-free-continuation/normalization.md) and [local-learning analysis](../research/2026-10-20-neural-free-continuation/local-learning.md) provide the later-phase obligations.

Live manuscript definitions are in [MAgent foundations](<C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/chapters/01_foundations.tex>), `def:generative-model` and `def:shared-model`, and [minimal_agent_model.tex](<C:/Users/chris and christine/Desktop/Research/manuscripts/Attention/minimal_agent_model.tex>), its model definition, mean-field proposition, and implementation scope. They are research context with claim-specific assumptions. No vault changes are included in this plan.

The classical joint-density conditioning precedent is [Ghahramani and Jordan, Supervised learning from incomplete data via an EM approach](https://proceedings.neurips.cc/paper/1993/file/f2201f5191c4e92cc5af043eebfd0946-Paper.pdf). It supports the statistical construction, not a claim about this implementation's language-model performance.
