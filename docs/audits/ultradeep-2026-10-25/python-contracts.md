# Python and tensor contract investigation, October 25, 2026

This lane examined the live VFE_3.0 checkout at `92a7cdfb5bf02172feae22d698195c0abc09e738` and preserved its working changes. It inspected source only; the coordinator executed the cited tiny CPU reproductions serially with `C:/anaconda/python.exe`, hidden CUDA, and one CPU thread. No implementation was changed, and no independent adjudication was performed.

## Correctness finding

### PC1. Repeated-token loss evaluation under inference mode reads an unavailable tensor version counter

**Location:** `vfe3/contracts.py:179-195`, especially line 193; producer `vfe3/geometry/transport.py:1355-1364,1397-1403`; public reachability `vfe3/model/model.py:1156-1176,1314-1322`.

**Severity:** medium / P2. This is a supported model loss API failure under `torch.inference_mode()`. It does not affect the repository's ordinary `torch.no_grad()` evaluation path, and no failed production run was observed.

**Trigger and causal trace:** Call an evaluated model with both token IDs and targets inside `torch.inference_mode()`, with repeated input IDs and the compact per-head `block_glk` frame route. The private loss path passes `_reuse_content_factors=True` at model line 1319. The eligibility gate allows the built-in per-token encoder with no frame-space RoPE, no M-step capture, and unchanged encoded token frames. `_content_rows` gathers one copy of each repeated token, so `_compact_pair_from_exp` gathers the resulting frame factors back to occurrences and records their provenance. The provenance helper assumes that every tensor has a version counter:

```python
# vfe3/contracts.py:193
record = (unique_forward, unique_inverse, forward._version, inverse._version)
```

Tensors constructed in inference mode do not expose that counter. The bookkeeping operation raises even though the frame tensors and model inputs are valid. A batch with no repeated token bypasses the gather record and succeeds. The live exact-route settings do not themselves disable this optimization: they use per-token encoding, compact `block_glk` frames, comparison RoPE, and no M-step capture. The repository evaluator uses `no_grad`, so reproducing this against the live settings would require an API caller to choose inference mode.

**Executed evidence:** The coordinator's `probes_contracts_followup.py` builds a tiny `VFEModel` with vocabulary 7, dimension 4, two heads, one layer, sequence length 3, and disabled block MLP. It calls `model(x, targets=y)` under both evaluation contexts. `probes_contracts_followup.json` records:

```text
no_grad,   input [1, 1, 2]: completed
no_grad,   input [1, 2, 3]: completed
inference, input [1, 1, 2]: RuntimeError: Inference tensors do not track version counter.
inference, input [1, 2, 3]: completed
```

The saved traceback ends at `contracts.py:193`, through `transport.py:1403`. The separate logits-only controls in `probes_contracts_artifacts.json` complete under both modes, including repeated IDs. This distinction matters: `targets=None` uses `forward_beliefs` without the private reuse flag, so the claim is not that every inference call fails.

**Fix:** Skip the gathered-row certificate when either occurrence tensor is an inference tensor, letting `CanonicalFrameContext` perform its existing full validation, or implement an inference-compatible immutable certificate without reading `_version`. The former is the smaller change and retains the current computation.

**Regression check:** Keep repeated and distinct token batches, both `no_grad` and inference mode, and both logits and target-scored calls. Assert that all complete with finite results and that the loss values agree within the existing numerical tolerance. A target-scored inference test is necessary because a logits test alone misses this path.

**Falsifies:** A current run of the same tiny target-scored, repeated-token inference case that never reaches the unguarded `_version` read, or a supported tensor implementation that supplies the counter, would refute this scope. Neither holds in the recorded reproduction.

**Prior coverage:** The gathered-row optimization and its invalidation tests already exist in `tests/test_round2_projection_numerics_20261009.py`; the inspected cases do not create the factors under inference mode. `vfe3/numerics.py:83-88` already handles the same PyTorch distinction for its variance cache, and `VFEModel._freeze_frame_copy` also avoids version-based caching for inference tensors. This finding is distinct from the CG cache reuse failure in the gauge-attention report: it fails during evaluation itself, before any later backward pass.

## Design improvements and unresolved checks

The frame-space RoPE cache at `vfe3/model/model.py:710-718` stores a tensor built under inference mode without the guard already used by comparison RoPE at `vfe3/model/relational_operators.py:2196`. The coordinator reproduced an inference-tensor error when directly reusing that cached matrix in a grad-enabled matmul (`probes_contracts_artifacts.json`, `rope_cache`). However, the complete tiny model inference-to-training transition passed (`probes_contracts_followup.json`, `rope_model_transition`). On the compact path, `fold_rope_into_frame` stacks the diagonal rotation blocks during the later grad-enabled call (`vfe3/geometry/transport.py:286-289`), which constructs an ordinary tensor before the differentiable products. Consequently this lane does not report a production frame-RoPE failure. Aligning its cache policy with comparison RoPE would make the private helper's lifetime contract clearer; a dense-frame producer and consumer must be demonstrated before elevating the candidate.

Two other cache sites deserve a bounded mode-transition regression if that policy is standardized: `GaugeGroup._local_basis_as` caches cast basis tensors at `vfe3/geometry/groups.py:195-207`, and `head_block_generators` caches constructed block tensors at lines 282-302. `build_factored_transport` consumes these in differentiable algebra contractions at `vfe3/geometry/transport.py:1559-1565`. These optional local-basis and equal-block irrep routes were source-traced but not executed here; they are residual checks, not reproduced defects. The current elementary `block_glk` path does not use those caches.

No general registry signature mismatch or loss of a populated `BeliefState.right_phi` field was established in the inspected paths. The current tuple has only `mu`, `sigma`, `phi`, and `right_phi`; obsolete descriptions of additional fields were not treated as current contracts. The existing `_replace` calls preserve the positional field. The decoder callable alias agrees with the registered four-argument logits implementations, while fused CE uses its separate protocols. The serializer explicitly normalizes serialized booleans, applies historical field migrations, and preserves introduction-time defaults; no additional serialization defect was established by this lane.

## Coverage and limits

The inspected contract paths include `vfe3/contracts.py`, the current `BeliefState`, model construction and `_apply`, target-scored versus logits-only forward dispatch, canonical-frame capture, transport factor reuse, frame actions, generator dtype/device movement and caches, positional builders, decoder registration and callable protocols, configuration migration tails and historical defaults, selected runtime policy restoration, and the existing focused tests around frame provenance and inference tensors. The Research project page was consulted for route continuity; historical architecture statements were treated as dated context.

This was a selected source inspection, not an exhaustive type check of every module or every registry variant. It did not run static type checkers, a full suite, CUDA, or performance profiling. Other investigator reports own cue/result numerical behavior, Fisher geometry, CG coupling, training and checkpoint lifecycle, measurement artifacts, and full-link variational formulas. Broad exception handlers in optional plotting and external data code were not recast as tensor-contract findings without a concrete wrong-result path.
