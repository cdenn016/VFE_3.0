# Full-link gradient E-step investigation

October 1, 2026. Research and design proposal; no production implementation or training result.

## Recommendation and scope

The user's proposed replacement of the flat transport Omega_ij by the learned matrix link L_ij in the E-step energies is a mathematically viable direction. The first refactor should use full-state sending and receiving interfaces, a declared positive-definite receiving covariance, and receiver-side natural-gradient inference. Keep S_Q, S_K, T, and O fixed during each fast inference episode and learn them in the slow outer step. Keeping their values fixed during inference must not disconnect the outer learning graph.

For the first engineering comparison, retain the successful projected S_Q/S_K router, compute it from the incoming state snapshot, and freeze its weights throughout the receiver episode. Replace the receiver's Omega-based templates by full L-based Gaussian channels. This preserves all four learned interfaces and gives a specified receiver objective. A second, mathematically distinct variant routes by the new L-based energies themselves. If a common objective for comparison and content is required, its categorical update must use both costs. These alternatives are specified below; they should not be silently conflated.

The reported predictive test PPL of 20–30 versus transported E-step PPL of 40–50 motivates this investigation. Those are user-reported observations, not newly reproduced results or a matched causal comparison. No claim that this refactor will close that gap follows from the derivation. The project scope remains supervised language modeling with a target-blind inner filter. A transformer trained exclusively by a generative VFE without backpropagation is outside this task.

The separate agent's predictive full/full changes were preserved. The initial checkout HEAD was 3720337f0da6263b8bad4dad8e88c625ba4e8586 with dirty files; source fingerprints accompany this report. Sources continued changing during the investigation.

## What the live sources already establish

The live [attention manuscript](<C:/Users/chris and christine/Desktop/Research/manuscripts/Attention/GL(K)_attention_v2.tex>), sections sec:role_measurements and sec:attention_operator, already defines full-state comparison maps S_Q,h and S_K,h, sending maps T_h, and receiving maps O_h. With reference moments

$$
\xi_i=U_i^{-1}\mu_i,\qquad C_i=U_i^{-1}\Sigma_iU_i^{-\top},
$$

the content link is

$$
L_{ij}^{(h)}=U_iM_hU_j^{-1},\qquad M_h=O_hT_h.
$$

Here T_h has shape d by K and O_h has shape K by d. Every head reads and writes the full state, even though each head's link has rank at most d. Full coordinate access does not require a K-dimensional comparison space. Communication heads and frame blocks have different roles.

The same manuscript, section sec:flat_identification, distinguishes Omega_ij=U_i U_j^{-1}, whose products telescope, from L, whose products need not telescope, and from an independently specified group connection Theta. Its sec:probabilistic_status derives local categorical routing and an additive predictive kernel. It explicitly separates those constructions from gradient-based belief revision. The [predictive note](<C:/Users/chris and christine/Desktop/Research/manuscripts/Attention/projected_predictive.tex>) supplies the current moment-matched covariance convention.

The [gradient companion](<C:/Users/chris and christine/Desktop/Research/manuscripts/Attention/GL(K)_gradients.tex>), sections sec:final_free_energy, sec:filtering_free_energy, and app:variational_descent, already states the target-blind observation problem, receiver filtering, later-row smoothing derivatives, and Gaussian Fisher geometry. The [extension manuscript](<C:/Users/chris and christine/Desktop/Research/manuscripts/Attention/GL(K)_extension.tex>), sections sec:population, sec:coordinates, sec:scorecomparison, and sec:causality, supplies a normalized observation/interaction-record model. That extension has its own recognition restrictions, expected residual costs, and computational obligations. Its objective cannot be imported while leaving the existing Gaussian-KL scores unchanged by assertion.

The wiki pages [GL(K) gauge-equivariant attention](<C:/Users/chris and christine/Desktop/Research/wiki/concepts/GL(K) gauge-equivariant attention.md>), [Projected predictive attention](<C:/Users/chris and christine/Desktop/Research/wiki/methods/Projected predictive attention.md>), and [Inference machinery](<C:/Users/chris and christine/Desktop/Research/wiki/themes/Inference machinery — variational EM and filtering.md>) provide continuity. Their dated benchmark summaries were not used as current-run evidence. The earlier [full-scope investigation](../../investigations/full-relational-2026-10-01/report.md) and its [MM derivation](../../investigations/full-relational-2026-10-01/mm-exact.md) already identified the missing full-output receiver law. This report advances that obligation with a concrete L-based law and gradients.

## Direct replacement: the receiver law and energy

At the start of one receiver episode, snapshot source laws bar_q_j and frames. Hold these values, the slow interfaces, and the selected routing weights fixed with respect to the fast receiver variables. The snapshots may retain an outer parameter graph.

Define a normalized full-width receiving channel

$$
X_i^{(h,j)}=L_{ij}^{(h)}X_j+\epsilon_i^{(h)},\qquad
\epsilon_i^{(h)}\sim\mathcal N(0,Q_i^{(h)}),
$$

with independent channel noise and

$$
Q_i^{(h)}=U_iQ_hU_i^\top,\qquad Q_h\succ0.
$$

The resulting target law and pair energy are

$$
p_{ij}^{(h)}
=\mathcal N(m_{ij}^{(h)},R_{ij}^{(h)}),\qquad
m_{ij}^{(h)}=L_{ij}^{(h)}\bar\mu_j,\qquad
R_{ij}^{(h)}=L_{ij}^{(h)}\bar\Sigma_jL_{ij}^{(h)\top}+Q_i^{(h)},
$$

$$
E_{ij}^{(h)}(q_i)=\mathrm{KL}(q_i\Vert p_{ij}^{(h)}).
$$

This is the direct Omega-to-L proposal with an explicit receiving law. Without Q, a head with d<K has a singular pushed covariance. A nondegenerate K-dimensional receiver Gaussian then has infinite KL to that lower-dimensional target. Receiving noise is part of the model; numerical jitter is a separate approximation. A fixed SPD Q_h in reference coordinates is a simple initial choice, not a new learned subsystem.

Under passive site changes, L transforms as g_i L g_j^{-1}, Q_i as g_i Q_i g_i^T, and each target as a receiver-law pushforward. The full-Gaussian KL is invariant. A diagonal recognition family in a fixed numerical chart is a restricted approximation and is not closed under arbitrary GL(K) changes. The full-law construction and the diagonal implementation restriction must both be reported.

For fixed nonnegative weights w_ijh and prior coefficient a_i, use

$$
F_i(q_i)=a_i\mathrm{KL}(q_i\Vert p_i)
+\sum_{h,j}w_{ijh}\mathrm{KL}(q_i\Vert p_{ij}^{(h)}).
$$

Assume c_i=a_i+sum_hj w_ijh>0. Every Gaussian target and the prior are SPD. Write

$$
\Lambda_i=a_i\Sigma_{p,i}^{-1}
+\sum_{h,j}w_{ijh}R_{ij}^{(h)-1},\qquad
b_i=a_i\Sigma_{p,i}^{-1}\mu_{p,i}
+\sum_{h,j}w_{ijh}R_{ij}^{(h)-1}m_{ij}^{(h)}.
$$

Expanding Gaussian KL gives, up to receiver-independent terms,

$$
F_i=\frac12\mu_i^\top\Lambda_i\mu_i-b_i^\top\mu_i
+\frac12\operatorname{tr}(\Lambda_i\Sigma_i)
-\frac{c_i}{2}\log\det\Sigma_i+\mathrm{constant}.
$$

Consequently the analytic receiver derivatives are

$$
g_{\mu_i}=\Lambda_i\mu_i-b_i,\qquad
G_{\Sigma_i}=\frac12(\Lambda_i-c_i\Sigma_i^{-1}).
$$

The Gaussian Fisher descent directions are

$$
\dot\mu_i=-\Sigma_i(\Lambda_i\mu_i-b_i),\qquad
\dot\Sigma_i=c_i\Sigma_i-\Sigma_i\Lambda_i\Sigma_i.
$$

These formulas supply a gradient implementation using the existing family geometry. The unique full-Gaussian frozen optimum is

$$
\mu_i^*=\Lambda_i^{-1}b_i,\qquad
\Sigma_i^*=c_i\Lambda_i^{-1}.
$$

For J_i=Sigma_i^{-1} and h_i=J_i mu_i, the full-Gaussian flow satisfies Jdot=Lambda_i-c_i J_i and hdot=b_i-c_i h_i. Thus natural-coordinate interpolation with rho=1-exp(-c_i t) is its exact finite-time solution for these frozen coefficients. This provides a derived damping/closed-form lane alongside gradient stepping. It does not establish descent when sources, coefficients, or templates are refreshed during the step, or when bounds and clipping change the map.

Within a diagonal receiver family, the frozen optimum instead has diagonal variance v_k=c_i/(Lambda_i)_kk, while its mean still solves the full precision system. Taking diag(c_i Lambda_i^{-1}) computes an inclusive moment projection of the dense optimum and generally gives another answer. The diagonal restricted Fisher flow also differs from the full Fisher flow followed by projection. Retain a declared family-specific solver rather than calling these approximations interchangeable. If receiver covariance is held fixed at Sigma_0, the mean-only Fisher flow is $\mu(t)=\mu^*+\exp(-t\Sigma_0\Lambda_i)(\mu(0)-\mu^*)$. Applying joint information-coordinate damping while freezing covariance does not implement that mean-only flow.

## Routing with S_Q/S_K: three distinct contracts

The existing projected comparison cost is

$$
D_{ij}^{(h)}
=\mathrm{KL}\!\left(
\mathcal N(S_{Q,h}\xi_i,S_{Q,h}C_iS_{Q,h}^\top)
\Vert
\mathcal N(S_{K,h}\xi_j,S_{K,h}C_jS_{K,h}^\top)
\right).
$$

The first contract computes beta from D on incoming states and freezes it for F_i, with w_ijh=lambda_h beta_ijh. T/O define the full receiving targets; S_Q/S_K determine which targets receive weight. This is a separately routed receiver inference operator. Its receiver optimization is well-defined, while the router is not claimed to minimize that receiver energy.

The direct shared-objective contract instead uses

$$
F_i(q_i,\beta_i)=a_i\mathrm{KL}(q_i\Vert p_i)
+\sum_h\left[
\lambda_h\sum_j\beta_{ij}^{(h)}E_{ij}^{(h)}(q_i)
+\tau_h\mathrm{KL}(\beta_i^{(h)}\Vert\pi_i^{(h)})
\right].
$$

The exact label coordinate is beta_ijh proportional to pi_ijh exp[-lambda_h E_ijh/tau_h]. It routes by linked-content agreement. It no longer contains independent S_Q/S_K role projections. This is a useful direct test of the user's energy replacement, but changes the successful predictive router as well as the receiver dynamics.

A combined objective retains all four maps by replacing the row cost with gamma_h D_ijh+lambda_h E_ijh. Its exact beta must use that combined cost. A Q/K-only beta is not that coordinate minimizer. The derivative of one comparison cost with respect to reference covariance C_i is

$$
\frac12 S_{Q,h}^\top
\left[V_{K,j}^{-1}-(S_{Q,h}C_iS_{Q,h}^\top)^{-1}\right]S_{Q,h}.
$$

For the local Sigma_i derivative, replace S_Q,h by A_Q,i,h=S_Q,h U_i^{-1} and form the query covariance as A_Q,i,h Sigma_i A_Q,i,h^T. The projected entropy contribution prevents the simple scalar-mass covariance fusion above. A mean derivative at frozen covariance remains a quadratic precision pullback, but joint covariance inference needs the actual projected-KL gradient. This is an additional model and solver choice.

These distinctions are consistent with probabilistic-attention precedents: Movellan and Gabbur derive Gaussian-mixture responsibilities and a value update under their specific likelihood and prior assumptions, then separately allow supervised differentiation through the inference map. Their construction does not prove that arbitrary independent comparison/content maps have one common E-step objective. [Probabilistic Transformers](https://arxiv.org/html/2010.15583v2), equations 3–20 and “Off-line Supervised Parameter Learning.”

## Why this does not automatically reproduce predictive attention

The predictive mean is an additive synthesis,

$$
\mu_i^{\mathrm{pred}}
=r\mu_i+U_i\sum_hO_h\sum_j\beta_{ij}^{(h)}T_h\xi_j.
$$

The L-based KL optimum is a precision-weighted geometric pooling of full receiving laws. With no prior and equal target precision, its mean averages over all head/source weights. Because each head's beta row sums to one, H heads yield a factor 1/H relative to the unscaled predictive head sum. Unequal precisions further change both amplitude and direction. The prior introduces another attraction term. Mean update rates, covariance evolution, and the fast episode schedule therefore remain substantive architectural differences.

Predictive moment matching includes between-source spread and cross-head shared-source covariance. The KL pooling covariance c Lambda^{-1} has another statistical meaning and does not contain those mixture-spread terms. Preserving full/full shapes alone does not preserve the empirical mechanism supplied by predictive uncertainty.

A useful parity control is to construct the existing predictive Gaussian target and relax a receiver toward it by KL. Its exact dense Gaussian optimum reproduces that target, but this is relaxation toward a declared predictor approximation. It does not create a new generative VFE or reproduce a reverse-KL optimum in a diagonal family without an explicit projection choice. Use it as a diagnostic if needed, not as evidence that the direct L energy equals predictive attention.

## Fast beliefs, slow maps, and the outer graph

The slow/fast assignment is explicit: within each fast episode update q_i and, if selected, beta_i; hold S_Q/S_K/T/O and frames fixed in value. After supervised loss evaluation, update their parameters in the slow optimizer. The existing trainer already provides score-map and content-map optimizer groups.

An inner partial derivative and an outer graph detachment are separate decisions. Introduce independent receiver probes z_i and C_probe,i. Build source snapshots, routing, and targets from the incoming layer state with their outer graphs live. Differentiate F_i with respect to the probes only; then differentiate the resulting finite inference map through the source bank, router, and slow maps for supervised learning. This retains own-row filtering without adding later-row source forces to the forward update.

The current conditional code detaches source_mu/source_sigma at vfe3/gradients/qkv.py:239, uses T/O in means at :258, keeps Omega-based covariance at :259–277, and returns no covariance derivative. The source detachment also removes outer cross-token belief credit. Restoring that credit with receiver probes is a concrete proposed improvement, not a demonstrated explanation of the PPL gap.

Smoothing differentiates later receiver rows back into earlier source states during the forward inference computation. Over a complete teacher-forced sequence that can make earlier beliefs depend on future tokens. A triangular attention mask does not remove those later-row terms. Keep causal filtering for the initial deployed route and test prefix invariance. Ordinary outer differentiation through a causal receiver computation is compatible with that choice.

Initial learning should use finite unroll with differentiable analytic forces. Straight-through and detached estimators can remove task gradients for maps that appear only inside the force. True shared-objective EM would require an M-step on the same declared objective with beliefs held fixed. Slow CE-through-inference learning remains supervised bilevel training. A different label or cadence does not turn it into textbook EM. [Dempster, Laird, and Rubin](https://www.maths.usyd.edu.au/u/jchan/GLM/Dempster1977EM.pdf), sections 2–3.

Tokens are observations for the outer categorical language-model loss. The missing ingredient in the current inner objective is a declared observation likelihood for the latent beliefs and a matching recognition law. Evaluating decoder CE at a computed mean is generally not an expected negative log-likelihood under q. The negative-ELBO identity requires an explicit normalized latent/observed joint. [Blei, Kucukelbir, and McAuliffe](https://www.cs.columbia.edu/~blei/papers/BleiKucukelbirMcAuliffe2017.pdf), section 2.2. Adding the held-out next token to the deployed receiver computation would also change its prediction-time information. No such change is proposed here.

Implicit differentiation belongs later, if an equilibrium formulation is actually selected and solved with a measured residual and an admissible Jacobian system. It differentiates an equilibrium, not an arbitrary one-step trace. [Bai, Kolter, and Koltun](https://proceedings.neurips.cc/paper/8358-deep-equilibrium-models.pdf), section 3.

## Scale and optional Theta

In reference coordinates the full target is

$$
\widetilde R_{hj}=Q_h+O_h B_{hj}O_h^\top,\qquad
B_{hj}=T_hC_jT_h^\top.
$$

It depends on head/source, not receiver. A dense implementation can cache H N targets rather than an H N N K K pair field. A low-rank implementation can use Woodbury and the determinant lemma with a d-dimensional core; singular B is permitted because Q is SPD. Receiver/source trace and quadratic contractions still require work, and the frozen precision solve must be implemented for the chosen family. This is an algebraic cost route, not a measured speedup.

The [geometry contribution](geometry.md) derives the PSD-safe Woodbury form and source-factorized score contractions. For N>=512, retain compact frames, stream pair scores, and avoid dense per-edge K by K storage. For illustration, a batch-one H=8, N=512, K=512 fp32 pair field would occupy 2 TiB; that calculation is not a benchmark. Receiving noise shared in reference coordinates preserves the factorization; receiver-dependent model noise or general edge-dependent transports can change it.

The live manuscript's independent connection extension is L_ijh=F_ih Theta_ij, F_ih=U_i M_h U_i^{-1}. Writing Theta_ij=U_i A_ij U_j^{-1} gives L_ijh=U_i M_h A_ij U_j^{-1}. Edge A_ij may carry nontrivial connection information only when it is not merely another telescoping vertex factor. Reverse transport must be defined as the inverse connection edge; independently computed reverse attention need not equal it.

Theta can be a later opt-in extension with sparse or generated edge actions. Its role in comparison, content, and covariance must be declared separately, and its conditioning and optimization need an appropriate group metric. The first L-based refactor needs no independent Theta. Nonidentity matrix-link products do not establish connection curvature. Wetterich supplies a precedent for general matrix links under compact site groups, not a GL(K) VFE-transformer action or a universality result for this model. [Linear lattice gauge theory](https://arxiv.org/html/1307.0722v2), section II. The current implementation is real block_glk; complex/Hermitian Gaussian conventions and SU(n) representations require additional implementation rather than changing transpose notation alone.

## Concrete refactoring boundary and validation

The [runtime contribution](runtime.md) supplies source locations and checkpoint constraints. Reuse the projected operator bank, existing per-layer threading, optimizer groups, and family geometry. Add full per-head receiving targets and their mean/covariance derivatives; the current conditional apply_content collapses the wrong head axis for arbitrary ambient O. Define the episode snapshot contract before lifting conditional depth/iteration guards. Reuse the existing gradient/mm solver selector only for a newly derived full-L capability; the legacy transported MM routine receives no relational operators.

Preserve the vfe3 import package and raw_score_q/raw_score_k/T/O parameter names. Preserve historical transported checkpoint semantics. Transported checkpoints lack these relational maps, while old head-scope conditional maps have different shapes. Warm starts or explicit map embeddings require a declared conversion and optimizer handling; they are not exact resume. New inference semantics should be identifiable in saved configuration rather than silently reinterpreting existing attention_mode values.

| Comparison | Purpose |
| --- | --- |
| Predictive full/full versus full-L receiver with the same frozen projected router and live outer source credit | Test receiver dynamics while retaining all four interfaces |
| Full-L receiver with outer source credit enabled versus detached, otherwise identical | Isolate the graph-contract change |
| Frozen projected router versus direct L-energy or combined-cost router | Isolate routing-objective changes after the receiver route works |
| Gradient versus the derived frozen full-L optimum/damping | Validate the solver and assess finite-step effects |

Production studies should use matched tokenizer, WikiText-103 split, evaluation context/stride, dimensions, depth, training tokens, seed set, decoder, MLP/norm, covariance policy, and parameter counts. Report both token-budget and measured-compute comparisons; increased E-step work changes throughput. Use at least 512 positions/agents for the requested scale study when resources are available. No production study, benchmark, or GPU work was launched here.

Meaningful implementation checks are tiny CPU gradient finite differences, a dense oracle for low-rank scores/targets, covariance positivity, exact frozen-solver stationarity, complete map/source gradient reachability, and prefix-invariance checks. Test every relevant configuration through the real loader and overrides. Do not use production-size models in unit tests.

## Checks performed and remaining obligations

Three specialists examined the variational construction, geometry/scaling, and executable route. An independent mathematical reconstruction is saved in [math-check.md](math-check.md). The [variational contribution](variational.md) contains the detailed objectives, derivatives, slow-step scope, and alternatives.

The reproducible [CPU script](check_derivations.py) used K=4, H=2, J=3 solely to validate the formulas. This small system sets no direction for the theory. Its [executed output](derivation-checks.json) records mean/covariance formula errors below 1.2e-15, directional finite-difference errors below 1.8e-10, envelope-force agreement, and frozen full-content optimum residuals below 3.1e-16. The example also exhibits a 0.44945 diagonal exclusive-versus-inclusive variance difference and a 0.02713 difference between combined-cost and Q/K-only routing. These numbers are witnesses for those formulas and distinctions, not population or training claims. All four proposed map groups received finite nonzero outer gradients; detaching the entire receiver result removed that synthetic readout's parameter graph. The script uses explicit CPU tensors and one thread under C:/anaconda/python.exe.

No production tests or training runs were executed by this investigation. Only research artifacts and the daily log were written; implementation/configuration changes visible in this checkout belong to the separate task. No Lean, shared-environment changes, or Research-vault edits were performed.

The implementation-ready core is the full-L receiving law, frozen receiver objective, analytic forces, and explicit graph/time-scale contract. The consequential design choice still to select for implementation is whether the first route keeps the existing projected router frozen, routes by L agreement, or changes to the combined cost. The recommendation is the first option for an interpretable engineering comparison, with the direct-L routing variant retained as a separate test of the user's broader hypothesis. Independent Theta, equilibrium learning, a population ELBO, and removal of backpropagation remain separate work.

## Retrieval scope

This was a targeted source/literature investigation, not a systematic literature review. JEV ranked local source candidates; the mandatory configuration, exact live manuscript sections, and linked prior investigations were read directly. External searches covered Dempster/Laird/Rubin EM, Movellan/Gabbur probabilistic transformers, Wetterich matrix links, Blei/Kucukelbir/McAuliffe variational inference, and Bai/Kolter/Koltun equilibrium differentiation. Only the primary papers cited above support external claims. Agent agreement and retrieval scores were not used as mathematical evidence.
