# TODO

## Triton kernels for the edge and latent link routes

Profiled on October 12, 2026 on the RTX 5090 under the live `train_vfe3.py` configuration (K=20, two heads, four layers, N=128, `decode_ce_triton=True`), one training step on a random token batch, with only the overrides shown changed. Caveat: these runs shared the GPU with another job. The same live-config step measured 64.5 ms later on an idle GPU, against 135 ms here, so the absolute times and the edge-to-baseline ratios below are inflated by an unknown amount. Re-measure on an idle GPU before using them. The φ₁ share of GPU time comes from the kernel breakdown and should survive, but that is also unverified.

| route | batch | ms/step | peak memory |
|---|---|---|---|
| live config, `content_link_edges="off"` | 64 | 135 | 3.6 GB |
| `content_link_edges="lowrank"` | 64 | 2,167 | 19.1 GB |
| `"lowrank"` + `content_link_latent=True` | 64 | out of memory (over 45 GB allocated) | |
| live config | 16 | 41 | 1.3 GB |
| `"lowrank"` | 16 | 178 | 5.1 GB |
| `"lowrank"` + latent | 16 | 627 | 17.8 GB |

On both edge routes, 75 to 90 percent of GPU time is the per-edge coefficient $\Phi_{ij} = \varphi_1(B_j^\top A_i)$. `vfe3/model/edge_links.py::phi1` evaluates it through `relational_operators._phi1`, a `torch.linalg.matrix_exp` of the augmented $2r\times 2r$ block (8×8 at r=4), over a (B, N, N) batch that holds about $10^6$ matrices at batch 64. cuBLAS runs these as batched gemv kernels (`gemmSN_NN`) that do almost no useful work. The autograd backward exponentiates a 16×16 block per edge. The latent route repeats both inside `torch.autograd.grad(create_graph=True)` on every fixed-point step of `infer_latent_receivers`. The step time also grows faster than the batch: 6.5 ms per `matrix_exp` call at batch 16 against 137 ms at batch 64.

### 1. Batched small-matrix φ₁ kernel (first)

One program per matrix computes $\varphi_1(X)$ for r×r inputs entirely in registers. It uses Taylor series in Horner form on $X/2^s$, followed by $s$ doublings $\varphi_1(2Y) = \tfrac12\varphi_1(Y)(I + e^Y)$ with $e^Y = I + Y\varphi_1(Y)$, on a fixed schedule with no host syncs.

The backward is the adjoint Fréchet derivative. For a power series with real coefficients, $L_{\varphi_1}(X)^*[G] = L_{\varphi_1}(X^\top)[G]$, so the backward kernel is the forward-mode Fréchet kernel evaluated at $X^\top$. The latent route differentiates through the gradient (`create_graph=True`), so it also needs the second Fréchet derivative $L^{(2)}_{\varphi_1}(X)[E, F]$. That can be done with second-order dual numbers carried through the same fixed schedule, plus its adjoint.

The kernel is a drop-in replacement for `edge_links.phi1`. It also serves `loop_energy`, which runs in float64, and the `"pure_gauge"` mode.

Evidence so far: written as plain elementwise PyTorch on a (64, 128, 128, 4, 4) batch, the same Taylor schedule (degree 14, three doublings) takes 34 ms for forward plus backward, against 111 ms for `_phi1`, with a maximum difference of 1.9e-6. A register kernel reads and writes 16 floats per matrix, so it should take well under a millisecond; that is an estimate, not a measurement.

Expected effect: lowrank steps fall from about 2.2 s toward a few hundred milliseconds at batch 64. Memory does not change, because the per-edge tensors are still materialized.

Tests: finite-difference gradients in float64 on tiny cases; gradcheck and gradgradcheck against `_phi1`; agreement across $\|X\|$ ranges that force different doubling counts; the existing `tests/test_edge_links_20261005.py` and `tests/test_latent_links_20261005.py` on CUDA.

### 2. Fused edge-contraction kernel (second, fixes memory)

`edge_reference_message`, `edge_covariance_factors` and `latent_transport_energy` build $\Phi_{ij}$ as (B, N, N, r, r) and the per-head weighted products as (B, H, N, N, r, r). The proposed kernel follows the fused decoder's structure. Each program owns a block of receivers $i$ and loops over senders $j$, recomputing $\Phi_{ij}$ in registers from $A_i$ and $B_j$. It accumulates $y_i$, $\Xi_i^g$, $Y_i$ and $q_i^h$ directly, so no per-edge tensor is stored. This is what would let the latent route run at batch 64.

The backward needs hand-derived adjoints of the $\Xi$, $Y$ and $q$ accumulations. The latent route would additionally need these to be twice differentiable. Ship the plain `"lowrank"` path first, and extend to latent only after kernel 1 shows what time remains.

## CUDA graph for the training step (live route, no edges)

Measured October 12, 2026 on an idle RTX 5090 under the live configuration, synthetic batch (B=64, N=128).

The step is CPU-bound, not sync-bound. A step takes 64.5 ms of wall time. Under the profiler, its GPU kernels are busy for 45.9 ms of a 95 ms span. There are 2,984 kernel launches per step: 1,083 from the forward, 1,432 from the backward, 59 from the optimizer, and the rest from the decode kernels and bookkeeping. 2,079 of them run under 5 µs. The forward takes 34 ms of CPU time to issue kernels that keep the GPU busy for 8.5 ms. The backward takes 52 ms of CPU time for 16 ms of GPU time. The 12 host syncs per step wait only 1.8 ms in total. Removing them would therefore not speed anything up by itself; they matter only because they block graph capture. Two cheap variants were within noise: disabling the `opt_einsum` path search (63.3 ms against 64.5 ms) and the sync-free exponential for the shared content link (62.5 ms).

Evidence for the gain: in a probe with the in-step sync sites patched out (listed below), capturing forward plus backward in one `torch.cuda.CUDAGraph` and replaying it took 44.6 ms, against 53.4 ms eager. Loss and every gradient were bitwise identical between replay and eager (maximum gradient difference 0.0). The expected step time falls from about 64.5 ms to about 56 ms, roughly 14 percent. Beyond that the step is close to GPU-bound, and further gains have to come from GPU time in the roughly 180 kernels longer than 50 µs, not from launches.

Sync sites inside forward and backward that block capture, with the fix each needs:

1. `relational_operators._content_link_exp` calls `torch.linalg.matrix_exp`, which reads its squaring counts on the host: three syncs in the forward and three in the backward per step. Route it through `transport._FixedScheduleMatrixExp`. That needs a static norm bound for the generator $OT$, which is unbounded today, so either choose a bound or cap the squaring count with a fail-closed check.
2. `transport._content_rows` uses `torch.unique`, whose output size is data-dependent. Either deduplicate on the host from the loader's CPU token batch and copy the indices non-blocking, or use the static per-occurrence path. The static path was measured 2x slower at K=128; the K=20 probe used it and still got the 44.6 ms above.
3. `numerics.safe_cholesky` branches on `bool(ok.all())` in `prior_bank._prepare_full_covariance_for_decode`. Replay optimistically, carry `ok` to the end-of-step gate, and redo the step eagerly when it is false.
4. `deferred_host_checks` resolves its raise-only verdicts with `host_read` before backward. Move them to the existing end-of-step gate. Raising before `optimizer.step` still keeps uncertified values out of every parameter update.
5. `PriorBank._validate_fused_ce_targets` consumes a loader stamp keyed on the target tensor's `id` and `_version`. In a graph the targets are copied into a static buffer, so the stamp has to be checked at the copy.

Scope: graph only silent steps, meaning `metrics_out is None`, `grad_accum_steps == 1`, and fixed B and N. Logging steps, evaluation and precision-retry replays stay eager. The optimizer, clipping and scheduler stay outside the graph, reading the static `.grad` buffers. The end-of-step gate in `train_step` (loss, gradient finiteness, target partition, clip norms) remains the one host read per step; it gates the precision-retry ladder, so overlapping CPU and GPU across steps would need that gate redesigned. Tests: graph replay against eager, bitwise, for loss and every gradient over several steps with different batches; the fallback when a captured `ok` or host-check flag fails.

## Concurrent training runs in the multi-run launchers

A single run leaves the GPU idle for roughly 30 percent of each step while Python issues kernels (see the CUDA graph section). A second training process on the same GPU fills those gaps with its own kernels. Each run needs one to two cores and 3.6 GB at the live configuration, so two concurrent runs leave the 9900X free for other work. The expected total throughput is 1.3 to 1.4 times a single run, an estimate from the 46 ms of GPU work in a 64.5 ms step, not a measurement. Pinning the process to one CCD, raising its priority or capping torch's intra-op threads would gain at most a few percent and is not worth doing.

Step 1 is to measure. Time two concurrent live-config runs against one on an idle GPU (about 10 minutes). Below about 1.2 times total throughput, drop the item.

Step 2, in `ablation.py` first. Add `max_concurrent_runs` to `CONFIG`, with a default of 1 that leaves current behavior unchanged. Cells run in worker subprocesses that receive the resolved baseline (`BASELINE_CONFIG` through `_resolved_sweep_baseline`), the sweep definition and the invocation's code and source identities as arguments, since `run_single` reads module globals and `parameter_optimization.py` monkeypatches `ablation.BASELINE_CONFIG` and `ablation.SWEEPS` in-process. Only the parent writes the shared files: `sweep_meta.json`, `sweep_results.csv` (rebuilt from per-cell markers by `_collect_sweep_results`) and `ablation_run_status.json`. It also keeps the resume and ownership checks and runs `analyze_sweep` and the figure workers. Per-cell `wall_time_s` is recorded with a flag saying it ran concurrently.

Step 3, the same runner in `scaling.py` (`run_cell`; the parent alone writes `scaling_design.json`) and in `train_vfe3.py` seeds (`_run_once`; the parent alone writes `multiseed_request.json`; importing the module in a child has module-level side effects). `paper_arms.py` comes last. Its calibration runs (`_calibration/`, `width_rule.json`) and the deferred compute-matched arm measure wall-clock throughput (`vfe3/baseline.py` `measured_train_device_hours`), so they must stay serial, and the deferred arm still waits on every evaluated seed. `run_vfe` and `train_baseline.run_baseline` already take every input as arguments. `write_manifest` uses the fixed temporary name `m2_manifest.json.tmp`, so it stays parent-only.

Until this exists, two `ablation.py` invocations on different sweeps can already run side by side if each gets its own `output_dir`; `ablation_run_status.json` in the output root is the only file they would share.
