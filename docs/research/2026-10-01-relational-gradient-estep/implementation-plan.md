# Full-link gradient E-step implementation plan

**Goal:** Implement the accepted full-state L-based receiver filter as projected_gradient.
**Architecture:** Freeze the incoming Q/K router and source target values once per layer episode while retaining their outer graph. Pool noisy learned receiving laws into precision statistics, then apply the existing Gaussian Fisher directions and SPD retraction over the requested fast iterations.
**Specification:** C:/Users/chris and christine/Desktop/VFE_3.0/docs/research/2026-10-01-relational-gradient-estep/report.md.
**Baseline:** f4e38715def1d03129086ea7c952fc2908ab5736, after predictive full/full publication.

The initial implementation supports local gaussian_diagonal and gaussian_full recognition, constant prior weighting, filtering, full scopes, fixed frames, and unrolled gradient inference. Independent Theta, dynamic alpha, other recognition families, MM, and training comparisons remain outside this implementation. Receiving noise is a separate positive reference variance, default 1.0, not a numerical jitter. Its value requires experimental calibration.

- [x] Kernel owner: add gradients/full_link.py and independent finite-difference, literal-pair and graph tests in test_full_link_gradients.py. API build_full_link_context returns Lambda, eta, c, beta and score with a live outer graph; full_link_gradients differentiates only the receiver variables.
- [x] Integration owner: extend config, relational operator construction, and E-step dispatch. Build one context per episode. Keep existing parameter names, old modes, CPU/GPU implementation portability, causal routing, and covariance skip/trust behavior.
- [x] Integration-test owner: add test_projected_gradient.py using test-owned tiny configurations. Exercise multiple layers/iterations, every map's outer gradient, off-head outputs, causal interventions, covariance skip, evaluation/autocast, diagnostics and serialization.
- [x] Run focused tests once after batched implementation, then one broader projected/transported/config compatibility pass if focused tests pass. Use C:/anaconda/python.exe, explicit CPU and one thread; save JUnit evidence here.
- [x] Save usage, assumptions, exact checks and remaining limits in implementation.md; append the daily edit log. Preserve canonical launcher/config WIP and keep the finished changes reviewable in the attached worktree.

No production training, GPU work, environment changes, vault edits, commits, pushes or merges are part of this implementation request.
