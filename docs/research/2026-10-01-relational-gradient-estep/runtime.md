# Runtime trace and refactor map for a full relational gradient E-step

October 1, 2026. Research-only trace. No implementation, test, active configuration, environment, checkpoint, or Research-vault file was changed. The predictive full/full implementation was changing this checkout concurrently; the accompanying source-manifest-runtime.json records observed bytes. HEAD was 3720337f0da6263b8bad4dad8e88c625ba4e8586 with dirty implementation/launcher files. Line references can shift as that work lands. The manifest was captured at 19:16:27 UTC after the trace; config references were refreshed for the final two-line insertion observed at that time.

## Finding

Start from projected_conditional's receiver solver and projected_predictive's full interfaces. Full input already exists in conditional filtering. Full output requires a new receiver objective, separately retained ambient targets for each head, an explicit covariance/noise construction, and an outer source-credit contract. The existing loop, per-layer map construction, family Fisher preconditioner, SPD retraction, and optimizer groups are reusable. Removing guards alone implements none of these new semantics.

The user's slow/fast distinction is already compatible with the training seam: S_Q/S_K/T/O are optimizer parameters fixed throughout the forward E-step, while q changes in its inner loop. They are updated by the outer optimizer afterward. Keeping parameters fixed during inference does not require detaching them from the graph used by outer CE.

No new numerical/training test was run. The user's PPL observations motivate the work; this trace does not establish their causal explanation.

## Existing continuity and geometry

Research/manuscripts/Attention/GL(K)_attention_v2.tex, sec:role_measurements and sec:attention_operator, already defines whole-reference-state S_Q/S_K/T and ambient O. Lines 216-240 define L_ij,h = U_i O_h T_h U_j^{-1}, its endpoint law, and the routed sum. Its table distinguishes L, flat Omega_ij=U_i U_j^{-1}, and a separately specified invertible connection Theta. L can be rank deficient; it cannot be substituted blindly for an invertible Gaussian transport.

Research/manuscripts/Attention/GL(K)_gradients.tex, sec:filtering_free_energy, lines 243-280, distinguishes target-blind own-row recognition, smoothing with later-row key terms, and predictive content. The latter is not identified with the former dynamics. The prior reports docs/investigations/full-relational-2026-10-01/{report,estep-routes}.md already identified the conditional output-force boundary; this trace extends their code map.

## Active settings and actual route

train_vfe3.py:108-118 sets K=20, H=2, N=128, batch=64, L=2, and n_e_steps=1. At :124-143 it selects projected_predictive, gaussian_kl routing, identity O initialization, moment_matched covariance, centering=False, attention_residual=False, KL order 1, gaussian_diagonal. The concurrent agent removed predictive scope fields from the launcher; config.py:371-372 defaults full/full, :894-898 rejects alternate predictive scopes, and model.py:377-380 passes literal full/full. The launcher sets gradient, mean rate 0.9, sigma rate 0.01, and phi rate 0 at :260-265. Construction is VFE3Config(**{**config, "seed": seed}) at :642; :433 overrides kl_max to 8*K.

Defaults are unroll, oracle_unroll_grad=True, fisher mean preconditioning, and flat transport. Active launcher values also include head_mixer=False, block_mlp=True, block_coordinates=local, mass_phi=0, M-step self-coupling weight=0, and diagonal_chunked decoding. These are intentional values.

Model creates one independent RelationalOperators per layer at model.py:368-391, builds realized comparison/value frames at :1271-1279, and passes the operators/frame/backward selector to stack at :1343-1370. stack.py:148-171 selects the layer module. block.py:154-181 forwards it into e_step; shared kwargs at :60/:68 carry gradient_mode/update. e_step.py:1317-1335 iterates.

The active execution enters e_step.py:922-935, computes relational_predictive_readout, and assigns both predictive moments. It bypasses belief gradients, conditional forces, Fisher stepping, SPD retraction, and mm_exact. Thus gradient is nominal admission policy for predictive mode, and e_q_mu_lr/e_q_sigma_lr do not set predictive gain. Source/router/map graphs remain live at relational.py:167-175.

Simply changing attention_mode is rejected independently by the moment_matched guard at config.py:909-911, the full-output guard at :901-902, and conditional's L=1 guard at :978-979. A declared new route must resolve these policies without rewriting active predictive configuration.

## Current conditional receiver objective

relational.py:95-137 calls qkv_filtering_gradients. Full-input projected comparison is supported at relational.py:56-63. S_Q/S_K retain exact reduced covariance, with each score still d=K/H dimensional.

At qkv.py:239 source_mu/source_sigma are detached. At :258 content is applied to sources and then flat transport supplies the head-output version of U_i O_h T_h U_j^{-1} mu_j. apply_content at relational_operators.py:288-309 assembles one K-vector before routing, and rejects full output at :294-295. The constructor separately rejects it at :69-70. Ambient O_h gives a different K-vector for every head, routed by a different beta_ij,h, so these targets need their own head axis.

Target covariance at qkv.py:259-277 is the source family covariance under value Omega, with full transported head marginals for full/exact/frame families or diagonal transport for legacy diagonal Gaussian. T/O do not push it forward. Current conditional filtering therefore combines projected comparison routing, mapped mean templates, and an independently transported covariance metric. It is not KL to L_*q_j.

The analytic diagonal force at qkv.py:293-308 contracts routing heads into corresponding coordinate blocks and adds self-prior attraction. It excludes inner router/template derivatives. The alternate/full-family branch differentiates an independent receiver probe at :315-322. Both return (grad_mu,None) at :335. Conditional inference never changes covariance: e_step.py:1036-1042 freezes it even with skip_belief_sigma_update=False. This differs from predictive moment matching and transported covariance descent.

create_graph at qkv.py:227 and e_step.py:951 keeps outer derivatives through beta, comparison/content maps, and frames. Source detachment at qkv.py:239 additionally removes outer CE credit through sender moments. It does not freeze S_K/T themselves. The slow M-step can still learn them from their appearance in receiver forces.

An independent receiver variable z permits a stronger separation: compute partial_z of the receiver objective while treating templates/router as fixed with respect to z, but build those templates from outer-live source snapshots and fixed optimizer parameters. Then fast own-row inference and slow CE-through-source credit coexist. This is a proposed graph contract; current source snapshots are detached. Fixed inference coefficients, fixed optimizer parameters, and detached outer graphs must not be conflated.

## Transported gradients and geometry

Transported mode creates no relational modules. e_step.py:955-975 calls kernels.py:441 onward. The analytic capability predicate at kernels.py:360-380 requires a registered eligible family, filtering, KL order 1, entropy, belief-independent transport, and coupled value gauge. Other cases use the autograd oracle. Belief-dependent transport is rebuilt from receiver/source differentiation leaves through omega_builder.

oracle.py:193-196 detaches source-role leaves for filtering and aliases receiver/source leaves for smoothing. It assembles transported free_energy at :241-246 and differentiates at :247-254. The live path additionally requires both mu.requires_grad and sigma.requires_grad at :186. A new solver must not drop a live mean chain merely because covariance is intentionally fixed.

Smoothing changes the forward update; it is not an outer-backprop toggle. Later rows using an earlier token as key contribute to that token's force, explicitly described by GL(K)_gradients.tex:252-256. Triangular routing alone cannot preserve prefix-only earlier beliefs under this full-sequence gradient. Begin deployed autoregressive inference with own-row filtering and choose outer source credit independently.

The reusable update is e_step.py:992-1049. Gaussian geometry/retraction.py:733-774 applies nat_mu=Sigma*grad_mu and nat_Sigma=2*Sigma*grad_Sigma*Sigma; mean descends additively with optional whitened trust, covariance uses affine SPD retraction and bounds. The new objective supplies derivatives; existing family geometry supplies the tangent. New Euclidean Lie-algebra stepping is unnecessary.

| Estimator | Current behavior and new-solver implication |
| --- | --- |
| unroll | Finite-trajectory differentiation; conditional force needs create_graph. This should be the first supported estimator with complete source/map/frame credit. |
| straight_through | e_step.py:1007-1010 detaches the preconditioned tangent; identity state chain remains. MM has separate carriers at :918-921. Maps entering only in the tangent lose outer credit; reject or specify a separate estimator. |
| detach | model.py:1211 uses no_grad over the belief pipeline; solver maps lose CE credit. Current block MLP also rejects it at config.py:2272-2275. |
| implicit | Not executable: config.py:76 supports only unroll/straight_through/detach. Requires a fixed-point equation, residual criterion, Jacobian/VJP solve, and chosen role-graph contract. It does not reproduce an arbitrary one-step trajectory. |

Current projected support at config.py:925-989 requires filtering, gradient, unroll, oracle=True, one iteration, phi rate zero, flat, block_glk, supported Gaussian families, equal heads, and no cross couplings. Conditional additionally requires one layer. Infrastructure already constructs independent layer operators and updates layer priors at stack.py:181-183. The inner loop already recomputes scores on updated q. Extending layers/iterations requires objective/covariance/capture/causality checks, not replacement of the stack. Inner phi updates would require refreshed relational frames as well as transport.

## Outer learning and EM terminology

The held-out target is supplied to the decoder after forward_beliefs at model.py:1463-1472; the belief routine receives no target. loss=ce at :1575, followed by optional frame/self/CG terms. Active mass_phi and self-coupling weights are zero. train.py:856-862 calls loss.backward through inference. This is target-blind recognition plus supervised outer learning; this code path establishes no common observation-conditioned ELBO.

train.py:409-418 already gathers raw_score_q/raw_score_k from every layer at m_relational_score_lr and T/O at m_relational_content_lr, zero weight decay. Parameters stay fixed inside the q iterations; optimizer stepping occurs after backward. Reuse this seam. Comparison rows are initialized orthonormally, but comparison_pair at relational_operators.py:261-263 directly returns learned raw parameters; it does not maintain orthonormality after M-steps.

## Minimum module changes and unresolved requirements

| Module | Change |
| --- | --- |
| config.py | Declare the receiver objective/solver, full/full contract, covariance/noise and outer source-credit policies. Separate objective, derivative support, solver, and backward estimator; admit only implemented combinations. |
| model/relational_operators.py | Preserve raw_score_q/raw_score_k/T/O. Reuse full endpoints; add separately represented ambient targets/channels per head. One-vector apply_content is insufficient. |
| gradients/{relational,qkv}.py | Implement chosen receiver mean/covariance derivatives with independent receiver variables and explicit source snapshots/outer graphs. Replace block-coordinate contractions. |
| inference/e_step.py | Dispatch new derivatives into existing Fisher/retraction geometry. Declare frozen/recomputed quantities each iteration and record matching objective/residual diagnostics. |
| model/{model,stack,block}.py | Reuse per-layer maps/frame threading/recurrence. Validate depth/iterations; refresh frames only if inner frame updates are enabled. |
| gradients/kernels.py | Preserve transported kernels; projected MM needs its own derived capability and cannot use old Omega fusion unchanged. |
| train.py/run_artifacts.py | Reuse slow optimizer groups; report the actual receiver objective, covariance, source credit, and estimator. |

The central unresolved specification is the receiver objective: comparison-space divergence, ambient content channel, residual synthesis consistency, or a defined combination. Comparison-only inference contains no T/O and cannot by itself learn them through the fast q solver. If ambient targets use L Sigma L^T, rank-d links with d<K make that covariance singular; a full-rank channel/noise law or differently scoped construction is needed. Predictive mixture moment matching is not automatically a reverse-KL solver covariance step. Optional Theta also requires a declared role: comparison transport, content middle factor, channel covariance, or independently regularized connection.

Existing mm_exact at e_step.py:873-884 receives no relational operators. Natural-coordinate damping at :893-917 is reusable only after deriving the new frozen optimizer. Projected comparison precision pullback generally differs from ambient content attraction. No exactness, monotonicity, or convergence assertion follows from reusing the damping code.

## Checkpoint boundary

Keep vfe3, layer-zero relational_operators.*, and deeper relational_operators_extra.* prefixes at model.py:365-391. Conditional head maps have smaller shapes; full maps require explicit embedding and compatible optimizer-state handling. Transported checkpoints contain no relational parameters, so adding maps changes key sets and optimizer topology. A warm start with new interfaces is not exact resume. Even identity/block content initialization does not guarantee that new projected scores equal legacy diagonal transported scores.

Strict serialized loading translates historical attention selectors at config.py:2890-2905, rejects unknown fields at :2906-2911, and validates the constructed config at :2930. run_artifacts.py requires matching model keys/shapes at :1124-1134, optimizer topology at :1190-1209, and saved-config preflight around :2285-2296. Preserve old transported execution for historical continuation or require an explicit architecture conversion; do not silently reinterpret saved attention_mode.

## Evidence limits

Only this report and its source manifest were written. No new tests, GPU work, environment changes, Lean work, or vault writes. Earlier CPU evidence remains bound to its earlier snapshot. JEV ranking aided discovery; required sources were read directly. Memory was orientation only (MEMORY.md:55-56) and supplied no current-behavior evidence.
