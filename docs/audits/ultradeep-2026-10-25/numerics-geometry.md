# Numerical geometry and optimizer audit, October 25, 2026

The two findings below concern supported optional update routes. Neither is demonstrated on the resolved active route in `live-config.json`, which selects the exact predictive route, `gaussian_frame_diagonal`, the table Fisher update, and AdamW frames. The covariance Euler retraction and the frame Fisher optimizer are inactive there. No claim of an active-run saturation rate or perplexity effect is made.

The baseline is the live working tree at `92a7cdfb5bf02172feae22d698195c0abc09e738`. This investigator read source and supplied small probes; the coordinator executed the probes serially on CPU. The shared runtime receipt is `probes_initial.json`: `C:/anaconda/python.exe`, `torch 2.10.0.dev20251210+cu128`, one torch thread, CPU tensors. CUDA was deliberately hidden with `CUDA_VISIBLE_DEVICES=-1`, and the coordinator reported `torch.cuda.is_available() == False` in that environment. These are CPU results, not GPU measurements. No finite-difference or gradient-correctness claim is made here.

## Correctness findings

### N1. Stabilizer Fisher chart updates crash near zero mean because the spectrum formula cancels

**Location:** `vfe3/geometry/phi_fisher.py:151-160`, `vfe3/geometry/phi_fisher.py:419-423`; consumer `vfe3/gauge_optim.py:875-878`, `vfe3/gauge_optim.py:928-931`, `vfe3/gauge_optim.py:517-534`.

**Severity:** medium (P2).

**Evidence:** The supported configuration is `m_phi_update_mode="fisher_group"`, `m_phi_fisher_completion="stabilizer"`, `m_phi_group_retraction="chart"`, an untied `block_glk` chart, and a family with local diagonal prior covariance. Config validation permits the stabilizer completion for type frames with `pos_phi="none"` or `"frozen"` (`config.py:1754-1791`). For positive `a = mu^T S^{-1} mu`, the executable spectrum formula is `minus = 0.5 * (2.0 + a - sqrt(a.square() + 4.0))`. At small positive `a`, both large operands round to 2 before subtraction. `type_row_spectrum` then computes `lam_max / lam_min`, and the chart stager treats the resulting infinity as a fatal nonfinite certificate.

The coordinator's `probe_fisher.py` constructs the actual `GaugeManifoldAdamW`, with one 2-by-2 frame initialized at the identity, `S=I`, `mu=(m,0)`, gradient covector `I`, learning rate 0.01, and trust radius 0.1. `probe_fisher.json` records:

| Mean scale `m` | Computed minimum eigenvalue | Actual optimizer result |
|---|---:|---|
| 0 | 2 | Both diagonal chart entries become approximately -0.005 |
| 1e-6 | 5.000444502911705e-13 | Both diagonal chart entries become approximately -0.005 |
| 1e-8 | 0 | `FloatingPointError: pullback group direction returned a nonfinite certificate` |
| 1e-9 | 0 | Same exception |

For this chosen covector the solved direction is finite: its diagonal entries are `1/(2+a)` and `1/2`; the covector has no component in the weak rotation direction. The small spectral eigenvalue is also a positive representable float64 number, approximately `a/2`. This is a cancellation error in the diagnostic consumed by the chart stager, rather than overflow of the requested update. The spectrum is genuinely ill-conditioned near zero mean; that does not make the numerical zero or this whole-step exception correct. The observed crossover lies between mean scales 1e-6 and 1e-8 in this fixture. The algebra predicts cancellation once `a` is at machine precision relative to 2.

The practical consequence is that a finite, small frame update can abort an optional Fisher-chart training step. The direct optimizer reproduction establishes the consumer behavior. It does not establish how frequently a training trajectory reaches this mean scale. The current active AdamW route is unaffected. The October 20 report recorded the opt-in frame Fisher route as largely unexercised; no matching current fix was found in that report or the inspected Fisher tests.

**Fix:** Evaluate the positive root by the rationalized expression `2*a / (2+a+sqrt(a*a+4))` for `a>0`, preserve the distinct `a==0` completion branch, and keep deliberate conditioning rejection separate from a numerically corrupted diagnostic.

**Falsifies:** Re-run the retained actual-optimizer fixture and show a finite correct update, or a documented per-row conditioning rejection without a whole-step exception, for positive `m=1e-8` and `1e-9`; a spectrum-only test is insufficient. Add this regression beside the existing spectrum and optimizer checks, covering the exact zero case separately.

### N2. Finite covariance tangents silently become zero updates when their float32 norm overflows

**Location:** `vfe3/geometry/retraction.py:660-663`, `vfe3/geometry/retraction.py:754-756`; finite-entry gate `vfe3/geometry/retraction.py:604-614`; production caller `vfe3/inference/e_step.py:308-324`.

**Severity:** low (P3).

**Evidence:** Both covariance retractions form a plain float32 Euclidean/Frobenius norm before the trust scaling. `_neutralize_nonfinite_tangent` checks entries, not overflow of their norm. When every entry is finite but the sum of squares overflows, `trust_region / norm` becomes zero, so the tangent is replaced by zero. The trust-ball projection should retain its direction at the prescribed radius.

The coordinator's `probes_operators.py` checks `sigma=ones(2)` and `delta=m*ones(2)` for the diagonal path, and `Sigma=I_2`, `Delta=m*I_2` for the full path, with unit step size and trust radius 1. Both paths should give `exp(1/sqrt(2))*I`, or its diagonal, independently of positive `m`. The retained `probes_operators.json` records:

| Tangent magnitude `m` | Float32 diagonal and full diagonal entries | Float64 reference entries |
|---|---:|---:|
| 1e18 | 2.0281150341033936 | 2.028114981647472 |
| 1e19 | 2.0281150341033936 | 2.028114981647472 |
| 2e19 | 1 | 2.028114981647472 |
| 1e20 | 1 | 2.028114981647472 |

The observed crossover is between 1e19 and 2e19. Squaring two equal float32 coordinates predicts a threshold of approximately `sqrt(finfo(float32).max / 2) = 1.30e19`. The failed covariance differs from the bounded reference by approximately 50.7% relative norm. This is a compute-precision failure: the normalized tangent and final covariance are both comfortably representable in float32. Promoting storage is unnecessary.

The live production caller exists when `attention_mode="projected_gradient"`, `relational_receiver_update="euler"`, and covariance updates are enabled: the E-step forms the Gaussian natural gradient and passes `-e_q_sigma_lr * nat_sigma` to `retract_spd_affine`. Config validation accepts `"euler"` at `config.py:1212-1213`. The active exact predictive route does not enter this branch. No normal-training input distribution or saved run was shown to reach these extreme finite tangents, which is why this is P3. The nonfinite-tangent counter cannot reveal this case because every input entry passes its finite check. The October 20 audit's table-Fisher overflow work concerns a different optimizer and does not repair these two retraction norms.

**Fix:** Compute the trust scaling from max-absolute-value-normalized tangents, or use an overflow-safe norm with a finite working range before applying the scaling, while preserving the existing normal-range result.

**Falsifies:** On the retained two-dimensional fixture, both float32 retractions agree with the float64 bounded-step reference at 2e19 and 1e20; explicitly rejecting and reporting an unsupported finite tangent would also remove the silent-zero behavior. A test that checks only finiteness would miss the defect.

## Performance opportunities and test improvements

No new performance finding is promoted from this lane. No timing or GPU profiling was performed. The two focused regressions described above are the proposed test improvements; neither requires a model or production-sized tensors.

## Coverage and limits

Source inspection covered the guard and solve paths in `numerics.py` and `cpu_linalg.py`; covariance trust scaling, spectral-map adjoints, and the output-SPD certification flow in `geometry/retraction.py`; principal logarithm and paired Frechet-log evaluation in `geometry/matrix_log.py`; coordinate extraction and trace projection in `geometry/lie_ops.py`; frame pullback directions, series/spectral fallback gates, and chart/body conversions in `geometry/phi_preconditioner.py`; frame Fisher completion, type solves, pooled positional solves, and diagnostics in `geometry/phi_fisher.py`; and the group staging/commit paths in `gauge_optim.py`. The table Fisher and row-lazy optimizer update bodies, family KL and natural-gradient seams, and alpha/divergence dispatch were read. This was targeted mechanism inspection, not a line-by-line certification of every function in these modules.

The current table Fisher route snapshots frames before the base frame optimizer steps, transforms mean covectors through those frames, and uses the configured count-normalized metric. Its overflow twin only rescues subsequent arithmetic; finite-precision frame conversions themselves were inspected but not numerically stressed here. No additional supported-input failure was established for that active path. The sigma and divergence guard measurements in `metrics.py` and `model/model.py` were located; no saved-run guard fractions were measured, so no saturation-incidence finding is made.

The old retained-float64/public-dtype comments in `FullGaussian` suggested a covariance-rounding issue, but current producer searches found no production caller creating a narrower `_public_dtype` than the stored covariance dtype. That candidate is excluded for lack of a supported live trigger. Stock eigendecompositions found in the inspected optimizer paths run under their no-gradient update flow; no differentiable repeated-spectrum failure is claimed. The retraction's spectral maps use explicit Loewner adjoints in the current source; the older audit-role description of a Lorentzian-damped eigenvector backward is not an accurate description of this implementation.

Research continuity reads included `Research/index.md`, the wiki pages `Natural gradient`, `Divergence clamp saturation`, and `Diagonal truncation as gauge regularization`, the frame consolidation record at `Research/docs/attention-unification-2026-10-03/frame-coverage.md`, and its source note `Research/sources/manuscripts/dennis-2026-phi-natgrad.md`. The old `GL(K)_gradients.tex` and September 28 frame-note paths named in those records are absent from the current manuscript directory. The current `Attention/GL(K)_attention_v2.tex` and `GL(K)_supplementary_v2.tex` were consulted for the Gaussian Fisher, frame, and receiver conventions. The historical notes were used as context, not numerical evidence. No new theoretical claim depends on the missing files.

No source, configuration, test, or vault files were edited by this investigator. No additional agents or adjudication were launched. The coordinator-owned probe files are the executed evidence. The findings remain investigator findings, not independently adjudicated conclusions.
