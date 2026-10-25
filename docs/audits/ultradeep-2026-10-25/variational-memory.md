# Variational inference and generative memory audit

Three findings: two numerical correctness defects in cue/result posterior moments and one failure of the dense posterior's tile-budget contract. All three have small CPU reproductions executed by the coordinator in [probes_initial.py](probes_initial.py), with results in [probes_initial.json](probes_initial.json). This investigator performed source inspection only and did not run scripts, tests, training, or GPU work. The findings have not undergone independent adjudication.

The inspected baseline is the live source at `92a7cdfb5bf02172feae22d698195c0abc09e738`. No implementation file was changed. Existing toggles are intentional. The audit distinguishes exact local conditioning, Gaussian moment projection, finite mean-field sweeps, the depth-carried regime heuristic, and the outer task loss. It does not recommend neural replacements or treat an optional approximation as a defect merely because it is selected.

## Correctness findings

### VM1. Absolute log-evidence subtraction produces component weights with total mass two and corrupts both posterior moments

**Location:** `vfe3/model/cue_result_memory.py:185-189`, `193-204`, `257`, `296-312`, `380-382`.

**Severity:** P2.

**Trigger and reachability:** Float32 cue/result inference with large finite negative component log densities. Both the dense and diagonal record kernels are affected. A one-dimensional record with `xi=10000`, `C=1`, two identical components with `nu=omega=0`, `lambda=Q=1`, equal priors, and `F=I` suffices. The input and all covariances are finite and admissible. `CueResultMemoryBlockMLP.posterior` dispatches directly to these kernels at `block_mlp.py:2097-2100`. The active `principled_exact_route=True` at `train_vfe3.py:116` is applied at `590-598`; `exact_route.py:45-65` selects this block and the frame-diagonal carrier while retaining the launcher's rank and numerical settings. The production route normally computes these kernels in float32 (`cue_result_memory.py:82-83`, `339-345`). The probe uses dense regression for a simple independent reference; the normalization code is shared with the active low-rank representation.

**Evidence and causal trace:** Both tiles compute `log_mass = torch.logsumexp(log_w, dim=-1)` followed by `w = torch.exp(log_w - log_mass.unsqueeze(-1))`. At log weights near minus 25 million, float32 spacing is two: the `log(2)` increment in the absolute log normalizer is rounded away. Both normalized weights become one. The code then uses them in the mean and in deviations about that incorrect mean. The merge functions likewise recover coefficients by subtracting rounded absolute log masses. Returned `log_weights` repeats that unstable subtraction against the absolute evidence.

The coordinator's `cue_large_logmass` probe returned, in both dense and diagonal float32 kernels:

```text
weight_sum = 2.0
mean = 10000.0
covariance = 50000004.0
```

The two components are identical, so the correct weights are `0.5, 0.5`, the mean is `5000`, and the covariance is `1.5`. The float64 reference returned mean `5000.000009103181`, covariance `1.500000002813822`, and total mass `1.000000001820636`; its remaining error also comes from subtraction at the large absolute log scale. The executed fixture is `probes_initial.py:43-55`, result key `cue_large_logmass`.

**Consequence:** The block can silently emit a finite but grossly wrong Gaussian, and `regime_state['current']` receives log weights that no longer describe a probability vector (`block_mlp.py:2152-2155`). The next layer exponentiates those values in `regime_log_prior` at `2074`, so the error reaches both the continuous carrier and the regime update. No claim is made that a recorded training run reached this input. The coordinator also executed the tile comparison in [probes_config_cg.py](probes_config_cg.py), lines 40-49; [probes_config_cg.json](probes_config_cg.json), key `cue_logmass_tiling`, records covariance `3.0` at `tile_bytes=1` and `50000004.0` at `tile_bytes=1000000` for both kernels, with mean `10000` and weight sum `2` in every case. Thus a knob intended only to change resource use changes the inferred covariance by more than seven orders of magnitude. One-component tiles have zero within-tile spread, whereas one joint tile forms deviations about the wrong doubled mean.

**Fix:** Carry a detached row maximum and a scaled positive mass separately through tile accumulation and merging, as the existing prototype kernel does, and form returned log weights with stable `log_softmax`; construct the absolute log evidence only as an output.

**Test/reproduction:** Turn `cue_large_logmass` into a float32 regression test checking normalization, moments, finite gradients, and agreement across a whole tile and one-component tiles. Include an ordinary-scale reference and a large common negative log-density offset. Merely replacing within-tile weights by `softmax` leaves the mass-based merge problem unresolved, especially for unequal final tile sizes.

**Falsifies:** Retract if the displayed admissible fixture yields normalized weights and the analytical moments in float32 on the present source, or if an upstream enforced bound excludes these log-density magnitudes from every supported call. No such bound was found.

**Prior coverage:** New cue/result defect. The October 20 audit predates this module. The older exact prototype path already uses maximum-shifted statistics (`prototype_agents.py:236-255`, `329-344`). Existing cue/result parity tests use moderate float64 inputs (`test_cue_result_tile_budget_20261025.py:25-26`, `40-83`) and do not cover this failure.

### VM2. The low-rank diagonal covariance can cancel away strictly positive result noise

**Location:** `vfe3/model/cue_result_memory.py:292-299`; direct emission at `vfe3/model/block_mlp.py:2152-2155` and `vfe3/model/block.py:481-482`.

**Severity:** P2.

**Trigger and reachability:** The frame-diagonal cue/result route with a low-rank regression close to canceling the identity on a coordinate, and result noise small compared with the intermediate cue-posterior variance. The coordinator used `K=M=rank=1`, `c=lambda=1`, `U=1`, `V=-1`, and `log Q=-25`, hence `F=I+UV^T=0`, cue-posterior variance `p=0.5`, and strictly positive result variance `Q=exp(-25)`. The same construction embeds in one coordinate at larger K and rank, including the launcher's rank two. `block_memory_regression_rank` and positive result-noise ratios are accepted by `config.py:2182-2192`; the independent trainable `log_result_noise` is registered at `block_mlp.py:2048` and is not bounded away from this value. The active exact-route switch preserves the rank and result noise.

**Evidence and causal trace:** The low-rank branch computes

```python
cross = 2.0 * p * (left * right).sum(-1)
G = torch.einsum("akr,iak,aks->iars", right, p, right)
quad = torch.einsum("akr,iars,aks->iak", left, G, left)
v_Y = q + p + cross + quad
```

For this fixture the last expression is evaluated as `q + 0.5 - 1 + 0.5`. The first addition loses q in float32, and the remaining terms cancel exactly. The result is zero despite a representable positive result-noise variance. The dense-regression branch forms `F` and its squared entries before adding q at `286-288` and does not suffer this instance.

The coordinator's `cue_lowrank_cancellation` probe returned:

```text
expected_noise = 1.3887943746404563e-11
lowrank covariance = 0.0
dense_regression covariance = 1.3887943746404563e-11
dense_record covariance = 1.3887943746404563e-11
```

See `probes_initial.py:31-40` and result key `cue_lowrank_cancellation`. At `p=0.5`, float32 spacing is about `5.96e-8`; the tested q is far below that scale, but it is itself representable. This is cancellation of a valid positive term, not underflow of its exponential.

**Consequence:** The diagonal memory emits a singular carrier where the declared model has positive result noise. The reference-coordinate block writes the returned variance directly into the belief; it applies no posterior screen or recovery. A subsequent consumer may reject the singular state or operate on altered uncertainty. The probe demonstrates zero variance, not a training crash or a measured negative variance.

**Fix:** Add result noise after computing the regression variance, and screen cancellation-prone low-rank rows for recomputation using a positive sum of squares or higher precision; preserve Q explicitly rather than repairing the posterior with an unrelated floor.

**Test/reproduction:** Add the displayed float32 F-zero case to the low-rank diagonal tests, embed it in one coordinate at `K=2`, and compare near-cancellation cases and their gradients against the explicit dense F reference. Adding q last fixes the exact fixture but by itself does not establish accuracy of the remaining `p + cross + quad` cancellation.

**Falsifies:** Retract if the current low-rank diagonal call preserves the positive Q for this fixture, or if validated model parameters make a canceled coordinate impossible. The regression factors are trainable and unconstrained, so the latter condition is absent.

**Prior coverage:** New diagonal cue/result defect. `test_zero_regression_gives_input_independent_component_means` at `test_cue_result_memory_20261024.py:170-180` uses float64, dense regression, and checks the mean. The newer diagonal parity fixtures also use float64 and do not approach a noise-dominated canceled coordinate.

### VM3. Dense cue/result inference cannot honor its tile budget when one component across all tokens exceeds it

**Location:** `vfe3/model/cue_result_memory.py:243-255`; field propagation at `vfe3/model/model.py:443-444`.

**Severity:** P2, resource-contract defect on the supported dense-family path.

**Trigger and reachability:** `family='gaussian_full'` with cue/result memory on the validated reference route. `config.py:2202-2229` explicitly admits this family; `block_mlp.py:2097-2100` selects the dense kernel. It is not the family selected by today's principled exact-route switch. The trigger is `6 * tokens * K^2 * element_bytes > tile_bytes`, the dense kernel's own estimate for one component. A positive MiB budget is validated at `config.py:2200-2201` and passed to the module at `model.py:443-444`.

**Evidence and causal trace:** The dense kernel flattens all tokens into `xi_f` and `C_f`, then chooses only a component chunk:

```python
per_component = max(1, T) * K * K * torch.finfo(dtype).bits // 8 * 6
chunk = max(1, min(M, int(tile_bytes) // max(1, per_component)))
```

Every `_chunk_moments` call still receives the entire `xi_f, C_f`. Once the estimate exceeds the budget, `chunk` bottoms out at one and no token subdivision exists. The coordinator's `cue_dense_budget` instrumented tiny call requested 96 bytes, enough for the kernel's estimate of one float32 `K=2` token/component set; each of its three calls retained all eight tokens and had an estimated size of 768 bytes. These are tensor-shape estimates, not allocator peak measurements (`probes_initial.py:58-75`, result key `cue_dense_budget`).

At the inspected launcher shape `batch_size=32`, `max_seq_len=64`, `K=32`, a dense-family alternative already requires an estimated 48 MiB for one component across all tokens, exceeding the supported 32 MiB default. This is a source arithmetic example, not a run at that shape. Today's launcher sets 1024 MiB and selects the diagonal family, so this example does not assert a live-run memory failure.

**Consequence:** Decreasing the dense memory budget eventually stops controlling transient size. Separately, the dense component loop calls `_chunk_moments` directly with autograd enabled and contains no checkpoint, so normal autograd retains tile intermediates across components. Tiling alone therefore does not bound backward retention. The diagonal kernel has both token tiling and conditional checkpointing at `353-373`; the older exact prototype kernel has the same two protections at `404-419`. Total retained bytes and GPU throughput were not measured here.

**Fix:** Give the dense kernel independent token and component tile axes and non-reentrant per-tile checkpointing, matching the diagonal/prototype structure while retaining stable mixture accumulation.

**Test/reproduction:** Extend tile tests to inspect the actual dimensions passed to `_chunk_moments` and require the estimator to stay within the budget whenever at least one token/component pair fits. Check that reducing the budget reduces token rows when M is already one, and use tiny saved-tensor hooks to distinguish retained inputs from per-component dense factors. Existing parity tests cannot establish a resource bound.

**Falsifies:** Retract the transient finding if dense calls actually subdivide token rows under the displayed budget. Retract the backward-retention observation if a caller or kernel checkpoint suppresses those intermediates on the production dense route. Source inspection found neither mechanism.

**Prior coverage:** The new tile-budget tests cover values and gradients across budgets but do not assert tile dimensions or checkpoint behavior (`test_cue_result_tile_budget_20261025.py:57-83`). This resembles earlier resource-bound findings but concerns the new dense cue/result kernel, not the previously audited routed-spread kernels.

## Unmeasured performance opportunities

The dense checkpointing change in VM3 is a concrete opportunity to reduce backward retention; no timing benefit is claimed. This lane found no separate performance recommendation that needs to be added to the already documented diagonal-kernel fusion work. No benchmark or training inference follows from the tiny probes.

## Design and test improvements

Promote the numerical identities actually needed by this memory interface to float32 regression tests: component mass is one, identical components reduce to the single-component result, a canceled regression retains its positive result noise, and tile boundaries do not change the posterior. Retain the existing float64 differentiation and independent conditioning tests. Add a structural resource test for the dense kernel; changing a tile count while comparing only final values leaves VM3 invisible.

## Coverage and conclusions that are not findings

The source continuity reads included the Research index and VFE program/inference theme pages; `Research/manuscripts/Attention/minimal_agent_model.tex:126-260`; `docs/research/2026-10-24-mlp-generative-models/probabilistic-model.md`; `docs/2026-10-25.md`; and the relevant October 20 audit results. The older wiki route is explicitly marked superseded and was not treated as the live implementation specification.

The executable paths inspected were `minimal_agent.py` one-head exact statistics, mean-field label updates and information fusion, sequential head assembly, multi-sweep routing, ELBO/evidence accounting, and local covariance assembly; `prototype_agents.py` exact tiled statistics, CAVI, exact EM expectations, and product-law diagnostic; the complete cue/result kernel implementation; the prototype and cue/result block wrappers; reference-coordinate block dispatch; tied module construction and calibration; regime-state handoff in `stack.py`; and exact-route/config validation. Matching source tests reviewed included the cue/result dense, diagonal, regime, tying, tile-budget, and exact-route files. No existing pytest result was independently rerun by this lane.

The older exact prototype path retains between-component spread and normalizes the prior in its evidence. Its CAVI implementation recomputes labels from the current Gaussian but always fuses against the original record (`prototype_agents.py:641-674`), so it does not assimilate the same record anew on every sweep. The returned labels precede the final Gaussian coordinate update; the prior investigation already records this finite-sweep limitation. The exact-mode product-law diagnostic is not the evidence of the full correlated label/state posterior, also already recorded. Neither is a new defect.

The minimal-agent readout uses per-head source snapshots and sequential Gaussian moment collapse. Code at `minimal_agent.py:678-694` and `1381-1456` implements that local ADF contract. It does not enumerate the full multi-head label mixture. The exact-route module explicitly identifies those collapses and the family projection. Calling the complete stack an exact joint posterior would exceed its contract, but the code paths inspected do not supply a new unacknowledged exactness claim to report as a defect.

For regime carry, `model.py:449-451` constructs one module shared by depth, `stack.py:124-125` creates state anew for each forward call, and `162-163` promotes the current log weights after each block. `block_mlp.py:2073-2081` computes the declared shared-prior-times-predicted-regime law; both posterior kernels normalize that per-token log prior before their likelihood calculation. No detach cuts the carried regime gradient. Tying is validated when regime carry is enabled. This establishes the implemented local recurrence, not the likelihood of a single external-record model across depth; the prior investigation and the current documentation already delimit that interpretation.

Residual questions are finite-precision behavior outside the displayed tiny cases, total dense backward storage, and actual training exposure to large evidence offsets or small-noise canceled regressions. They require bounded follow-up measurements; no GPU availability, runtime, convergence, whole-stack ELBO, or language-model performance claim is made here.
