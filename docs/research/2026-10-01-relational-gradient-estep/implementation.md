# Full-link gradient attention implementation

October 1, 2026. Implemented in the attached isolated checkout at C:/Users/chris and christine/.codex/worktrees/relational-gradient-estep/VFE_3.0, based on f4e38715def1d03129086ea7c952fc2908ab5736 after predictive PR #29 merged. Publication follows the tested implementation; the canonical launcher and configuration WIP remain intentional local state.

This implements the first route specified in the [research report](report.md). Select attention_mode="projected_gradient" with e_step_update="gradient". Every head uses full S_Q/S_K and T/O interfaces. The incoming Q/K router and source laws supply one fixed context per layer episode, reused over n_e_steps. Their outer graph remains live so supervised CE can train the source moments, frames, and all four learned maps.

The receiving law uses L_ij^h=U_i O_h T_h U_j^{-1} and covariance L_ij^h Sigma_j L_ij^{hT}+Q_i, with Q_i=relational_receiver_noise U_i U_i^T. The new scalar is a positive reference variance, default 1.0. It is model noise, not numerical jitter or the predictive covariance floor. Rank-deficient head maps therefore define full receiving Gaussian laws. Source covariances and precisions are built once per source/head and pooled using lambda_beta beta_ij^h; the implementation never builds an N-by-N grid of K-by-K receiving covariances.

For the frozen receiver energy, the mean force is Lambda_i mu_i-eta_i. Full covariance uses (Lambda_i-c_i Sigma_i^{-1})/2; diagonal recognition uses (diag Lambda_i-c_i/sigma_i)/2 while retaining the dense mean force. The existing Gaussian Fisher preconditioner, mean trust region, SPD retraction, covariance trust, and sigma ceiling apply. skip_belief_sigma_update remains live. Diagonal recognition is a restriction in the numerical chart rather than an arbitrary-GL-covariant Gaussian family.

## Configuration

Apply these changes to a configuration when ready to run the new mode; the active launcher has not been edited:

~~~python
attention_mode = "projected_gradient"
e_step_update = "gradient"
relational_input_scope = "full"   # omission also selects full in this mode
relational_output_scope = "full"
relational_receiver_noise = 1.0
predictive_covariance = "passthrough"
lambda_alpha_mode = "constant"
~~~

The initial envelope supports gaussian_diagonal or gaussian_full, filtering, unroll, Fisher mean preconditioning, fixed frames (e_phi_lr=0), flat block_glk transport, gaussian_kl routing, and multiple layers/fast iterations. Historical transported, conditional, predictive, and transported mm_exact behavior remains available. Independent Theta, dynamic alpha, other recognition families, MM on the new links, and a common comparison/content routing objective are outside this initial route.

The fast learning rates were inactive on predictive attention and become active here. Receiving-noise variance, mean/covariance rates, and iteration count require training calibration. Finite discrete steps are not guaranteed to decrease the frozen energy; the diagnostic delta reports the executed step. No PPL improvement or production-scale throughput result has been established.

The same per-layer checkpoint names raw_score_q, raw_score_k, T and O are retained. No checkpoint conversion is performed. Resume still checks the saved configuration and optimizer topology; changing the attention mode is a new inference model rather than an exact predictive-run resume.

## Runtime and files

Configuration validation and deserialization admit the new selector and receiving-noise field. VFEModel constructs one relational operator bank per layer with the existing slow score/content learning-rate groups. e_step builds a FullLinkContext once at the incoming state. Each e_step_iteration then differentiates only the current receiver against that context, applies the existing family geometry, and reports its receiver cost. The outer training loss still uses supervised cross entropy and backpropagation.

The [kernel](../../../vfe3/gradients/full_link.py) contains the source-factorized laws and analytic derivatives. Integration touches config.py, inference/e_step.py, model/relational_operators.py and model/model.py. metrics.py, run_artifacts.py and train.py now report executed covariance inference, live episode sources, receiving noise, and receiver-energy diagnostics. The scalar receiver diagnostic omits receiver-independent normalizers and router entropy; it is not a normalized VFE or a slow-map training objective.

tests/profiles.py explicitly supplies the new neutral field. New tests cover the mathematical kernel and end-to-end inference. One preexisting scope test was corrected because PR #29 made None select mode defaults. A separate regression demonstrated and fixed an argument-position compatibility issue by adding receiver_noise as keyword-only.

## Checks and limits

Fresh JUnit evidence records 299 distinct passing cases and four GPU-only skips after replacing the stale scope-file results with its targeted rerun. There was one broader compatibility pass: initially 239 passed, one failed, four skipped; the affected scope file then passed all 13 cases. New kernel checks passed 25 cases, integration passed 30, and the constructor-position regression passed two. The initial collection failure and constructor red results are preserved as development diagnostics, not successful verification.

The checks compare analytic derivatives against literal Gaussian KLs and finite differences, test rank-deficient content maps with positive noise, preserve source/router/map credit, reuse fixed contexts while receiver states change, and verify causal forward behavior over multiple layers and iterations. They also exercise covariance skipping, bf16 CPU autocast, no-grad evaluation, configuration/state round trips, and artifact reporting. Final changed Python files passed syntax inspection and git diff --check.

A read-only AST trace of the live canonical launcher preserved its operational settings, including mean rate 0.9, covariance rate 0.01, sigma_init=15, the diagonal_chunked decoder, and slow relational rates 0.005. Its new-mode conversion changed only the mode, predictive covariance policy, prior-weight policy, and receiving-noise declaration before disclosed size reductions to K=4, H=2, N=3, V=7, batch=1, L=2. Tiny forward/backward and one outer optimizer step were finite; CE was 2.7340106964 on synthetic tokens. That is an implementation smoke check, not a training result. The script initially used the wrong public forward signature; the corrected public call passed.

All execution used C:/anaconda/python.exe, explicit CPU, and one thread. No GPU workload, production training, environment modification, vault modification, or publication was run. The receiver target uses source-only dense Cholesky; extreme conditioning and production-scale memory/throughput remain unmeasured. Canonical train_vfe3.py, vfe3/config.py, ablation.py, scaling.py and zzzzz.py fingerprints were unchanged during the launcher smoke.

Evidence is saved alongside this document in implementation-checks.json, launcher-smoke.json, integration.xml, compatibility.xml, compatibility-scope-fix.xml and constructor-green.xml. Kernel JUnit is copied here as kernel.xml. The source manifest records the exact checked file bytes and baseline.
