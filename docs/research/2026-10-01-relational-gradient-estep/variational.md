# Full-coordinate relational gradient E-step

October 1, 2026. Research-only contribution. All source/configuration and Research vault files were read-only. No numerical calculation, GPU workload, environment change, or Lean work ran here. The parent's independent numerical probes are separate evidence. The user reports predictive PPL 20-30 and transported-gradient PPL 40-50; this motivates the question without establishing causality for individual maps.

## Result, target, and assumptions

The direct energy replacement is
$$
g_{ij}^h=N(v_{ij}^h,R_{ij}^h),\qquad
v_{ij}^h=L_{ij}^h\mu_j^-,\qquad
R_{ij}^h=Q_i^h+L_{ij}^h\Sigma_j^-L_{ij}^{h\top},\qquad
L_{ij}^h=U_iO_hT_hU_j^{-1}.
$$
A declared $Q_i^h\succ0$ makes this a nondegenerate full-coordinate receiving law for arbitrary finite learned maps, including rank-deficient heads. Replacing $\Omega_{ij}$ by $L_{ij}^h$ in both the target mean and covariance gives an explicit receiver energy and Gaussian Fisher update. Changing only the mean gives a different receiving model.

The supported first refactor retains the incoming full-input $S_Q/S_K$ router, freezes its row and all source snapshots throughout the fast receiver episode, and uses these full-output $L$ receiving targets. The outer CE unroll can retain a live graph through all four maps. This is a coherent receiver energy with a separately computed router, not joint categorical/receiver descent on that energy. Coupled routing instead requires content cost, or comparison plus content costs.

The supplied target is fast Gaussian-belief inference with slow $S_Q,S_K,T,O$ learning; frames follow the declared schedule, with no fast frame update required initially. The user specifically proposes $\Omega\mapsto L$ in E-step energies. Equivalence here means equality of a declared objective, derivative, stationary law, or endpoint operator. Shared routing or covariance laws alone do not establish equivalence.

The derivation holds for every finite $N,K,H$, arbitrary invertible frames, finite maps of the stated dimensions, and finite nonempty allowed source sets. Full input/output means each head can read/write all $K$ coordinates; its intermediate dimensions may remain smaller. It does not require a head partition or commuting maps and does not establish large-population convergence or training quality.

Introduce a fixed prefix context $c$, source snapshots $(\mu_j^-,\Sigma_j^-)$, an SPD prior $p_i=N(\mu_{p,i},\Sigma_{p,i})$, fixed scalar weight $a_i>0$, nonnegative head weights, SPD receiving noise, positive source probabilities on allowed support, and positive temperatures. Recognition is $q_i=N(\mu_i,\Sigma_i)$ with $\Sigma_i\in SPD(K)$, or an explicitly diagonal family in a declared chart. Continuous densities use Lebesgue measure and labels/records counting measure. Nondegenerate Gaussians have positive full support and finite quadratic/log-density expectations, giving the absolute continuity and integrability required below. Singular receiving laws require a separate support-compatible family.

The source snapshots, maps, targets, routing, and weights are constants for the receiver partial derivative. This does not require deleting their slow-parameter dependence from outer unrolling.

## Source continuity

The live attention manuscript already supplies full-input comparison maps, independent content maps, $U_iOTU_j^{-1}$ links, receiving noise for rank-deficient links, and the shared-latent key/value correction. Its comparison tensor needs a receiver metric before becoming a vector link and is distinct from the learned content link. The earlier full-scope investigation already distinguishes head-local conditional inference from full predictive readout. The gap here is a full-output receiver energy and its routing semantics.

Sources read directly: [GL(K)_attention_v2.tex](<C:/Users/chris and christine/Desktop/Research/manuscripts/Attention/GL(K)_attention_v2.tex:114>), role measurements, attention operator, probabilistic status, matrix links, normalized-source appendix; [GL(K)_gradients.tex](<C:/Users/chris and christine/Desktop/Research/manuscripts/Attention/GL(K)_gradients.tex:131>), final free energy, filtering free energy, variational descent; [GL(K)_extension.tex](<C:/Users/chris and christine/Desktop/Research/manuscripts/Attention/GL(K)_extension.tex:48>), population, coordinates, score comparison, causality; [Projected predictive attention.md](<C:/Users/chris and christine/Desktop/Research/wiki/methods/Projected predictive attention.md:17>); [Inference machinery - variational EM and filtering](<C:/Users/chris and christine/Desktop/Research/wiki/themes/Inference machinery — variational EM and filtering.md>); [mm-exact.md](../../investigations/full-relational-2026-10-01/mm-exact.md); and [report.md](../../investigations/full-relational-2026-10-01/report.md).

## Normalized receiving channel

Let $Z_j\sim N(\mu_j^-,\Sigma_j^-)$ and $\varepsilon_i^h\sim N(0,Q_i^h)$ independently, and set $X_i=L_{ij}^hZ_j+\varepsilon_i^h$. The normalized joint is
$$
p(z_j,x_i\mid c,j,h)=N(z_j;\mu_j^-,\Sigma_j^-)N(x_i;L_{ij}^hz_j,Q_i^h).
$$
Integrating $x_i$ first proves normalization, and its receiver marginal is $g_{ij}^h$. Define
$$
E_{ij}^h(q_i)=E_{X\sim q_i}[\log q_i(X)-\log g_{ij}^h(X)].
$$
For $P_{ij}^h=(R_{ij}^h)^{-1}$ and $d_{ij}^h=\mu_i-v_{ij}^h$, expanding the expectation yields
$$
E_{ij}^h=\frac12[\operatorname{tr}(P_{ij}^h\Sigma_i)
+d_{ij}^{h\top}P_{ij}^hd_{ij}^h-K+\log\det R_{ij}^h-\log\det\Sigma_i].
$$
A covariant noise convention is $Q_i^h=U_i\widetilde Q_hU_i^\top$, $\widetilde Q_h\succ0$. Passive changes give $v'=g_iv$, $R'=g_iRg_i^\top$, hence invariant KL. With no noise, a head narrower than $K$ generally gives infinite KL against a full-support receiver. This rank/support fact applies at arbitrary finite $K$, not only a small-network example.

## Candidate 1: full content KL with a clamped Q/K router

Fix the incoming row $\bar\beta_{ij}^h$ and $w_{ij}^h=\lambda_h\bar\beta_{ij}^h$, $\lambda_h\ge0$. Define the scalar receiver objective before simplification:
$$
F_i=a_iE_{q_i}[\log q_i-\log p_i]
+\sum_{h,j}w_{ij}^hE_{q_i}[\log q_i-\log g_{ij}^h].
$$
Put
$$
c_i=a_i+\sum_{h,j}w_{ij}^h,\qquad
\Lambda_i=a_i\Sigma_{p,i}^{-1}+\sum_{h,j}w_{ij}^hP_{ij}^h,\qquad
\eta_i=a_i\Sigma_{p,i}^{-1}\mu_{p,i}+\sum_{h,j}w_{ij}^hP_{ij}^hv_{ij}^h.
$$
Normalized head rows give $c_i=a_i+\sum_h\lambda_h$, and the prior guarantees $\Lambda_i\succ0$. The objective is
$$
F_i=\frac12\mu_i^\top\Lambda_i\mu_i-\eta_i^\top\mu_i
+\frac12\operatorname{tr}(\Lambda_i\Sigma_i)-\frac{c_i}{2}\log\det\Sigma_i+\text{constant}.
$$
The constant is independent of the receiver only in the fixed-target block. Differentiating gives
$$
g_\mu=\Lambda_i\mu_i-\eta_i,\qquad
G_\Sigma=\frac12(\Lambda_i-c_i\Sigma_i^{-1}),
$$
with the unique full-Gaussian minimizer
$$
\mu_i^*=\Lambda_i^{-1}\eta_i,\qquad
\Sigma_i^*=c_i\Lambda_i^{-1}.
$$
Positive mean precision and strict convexity of the negative log determinant prove uniqueness. Every KL supplies receiver entropy, which requires the factor $c_i$. This precision fusion generally differs from the additive predictive mean $r\mu_i^-+\sum\bar\beta L\mu_j^-$. Arbitrary learned $T/O$ do not force these endpoints to coincide.

For the diagonal chart $\Sigma_i=\operatorname{diag}(s_i)$, the exact exclusive-KL optimum is $s_{i,k}^*=c_i/(\Lambda_i)_{kk}$. It generally differs from the moment projection $c_i(\Lambda_i^{-1})_{kk}$. A fixed-local-chart diagonal family is not closed under arbitrary GL coordinate changes; a declared reference diagonal family transported with its frame is a different restriction.

### Explicit free-energy realization and observation status

Index the prior and targets as densities $g_k$ with weights $u_k$, and set $\alpha_k=u_k/c_i$. Let $f_i(x)=\prod_k g_k(x)^{\alpha_k}$ and $Z_i=\int f_i(x)dx$. Weighted arithmetic-geometric inequality gives $f_i\le\sum_k\alpha_kg_k$, hence $0<Z_i\le1$. For any proper positive density $r_i$, declare
$$
p_i(x,e=1\mid c)=f_i(x),\qquad p_i(x,e=0\mid c)=(1-Z_i)r_i(x).
$$
This is normalized on $\mathbb R^K\times\{0,1\}$. On the record slice $e=1$ its negative ELBO is
$$
E_{q_i}[\log q_i(X)-\log p_i(X,e=1\mid c)]
=F_i/c_i=KL(q_i\Vert f_i/Z_i)-\log Z_i.
$$
The record is an internal modeled compatibility event, not an observed token or a derived language likelihood. This construction also does not make the external router a variational coordinate. Declaring only the normalized conditional $f_i/Z_i$ instead produces $KL(q_i\Vert f_i/Z_i)$, whose slow parameter derivative differs by $\partial\log Z_i$. A normalizer is constant only in the receiver block and cannot be silently omitted in that model's M-step.

### Natural dynamics and the exact fixed-episode flow

For tangents $(u,V)$ with $V$ symmetric, use the Gaussian Fisher metric
$$
g_{(\mu,\Sigma)}((u,V),(u',V'))
=u^\top\Sigma^{-1}u'+\frac12\operatorname{tr}(\Sigma^{-1}V\Sigma^{-1}V').
$$
The inverse metric gives
$$
\dot\mu_i=-\Sigma_i(\Lambda_i\mu_i-\eta_i),\qquad
\dot\Sigma_i=c_i\Sigma_i-\Sigma_i\Lambda_i\Sigma_i.
$$
Its objective derivative is $-g_\mu^\top\Sigma_i g_\mu-2\operatorname{tr}(\Sigma_iG_\Sigma\Sigma_iG_\Sigma)\le0$. Set $J_i=\Sigma_i^{-1}$ and $h_i=J_i\mu_i$. Direct differentiation gives
$$
\dot J_i=\Lambda_i-c_iJ_i,\qquad \dot h_i=\eta_i-c_ih_i.
$$
The exact finite-time flow with $\rho=1-e^{-c_it}$ is
$$
J_i(t)=(1-\rho)J_i(0)+\rho\Lambda_i/c_i,\qquad
h_i(t)=(1-\rho)h_i(0)+\rho\eta_i/c_i.
$$
SPD holds for $0\le\rho\le1$ and the infinite-time limit is the unique stationary law. Thus gradient, exact-flow damping, and MM can share one frozen objective. Euler steps, floors, clipping, and other damping rules are separately declared approximations.

If covariance is frozen, the proper mean flow is $\mu_i(t)=\mu_i^*+\exp[-t\Sigma_i\Lambda_i](\mu_i(0)-\mu_i^*)$. Joint information-coordinate damping is not this flow with covariance frozen. In the diagonal family the Fisher directions are $\dot\mu=-\operatorname{diag}(s)\nabla_\mu F$ and $\dot s_k=-2s_k^2\partial_{s_k}F$.

Exactness and descent require the external router and all targets fixed throughout the episode. Recomputing that Q/K router each iteration changes $F_i$, yielding an update-map iteration without the common-objective monotonicity guarantee. Adaptive weights need their own variational stationarity/envelope block or their ordinary through-weight derivatives.

## Candidate 2: coupled comparison and content objective

Define $A_h=S_{Q,h}U_i^{-1}$, $b_{hj}=S_{K,h}U_j^{-1}\mu_j^-$, and $V_{hj}=S_{K,h}U_j^{-1}\Sigma_j^-U_j^{-\top}S_{K,h}^\top$. Require full row rank of comparison maps; full input scope alone is not a rank condition. Define
$$
D_{ij}^h=E_{Y\sim(A_h)_*q_i}[\log((A_h)_*q_i)(Y)-\log N(Y;b_{hj},V_{hj})].
$$
Take
$$
F_i^c=a_iKL(q_i\Vert p_i)+\sum_h[
\sum_j\beta_{ij}^h(\gamma_hD_{ij}^h+\lambda_hE_{ij}^h)
+\tau_hKL(\beta_i^h\Vert\pi_i^h)].
$$
The constraints are $\beta^h\ge0$, $\sum_j\beta_{ij}^h=1$, $\gamma_h,\lambda_h\ge0$, and $\tau_h>0$. Content-only direct replacement uses $\gamma_h=0$; positive $\gamma_h$ retains independent comparison maps in one scalar energy.

The multiplier condition is $\gamma_hD_j+\lambda_hE_j+\tau_h(\log(\beta_j/\pi_j)+1)+\zeta_h=0$. Its Hessian is positive on nonzero simplex tangents, giving
$$
\beta_{ij}^{h*}=\operatorname{softmax}_{j\in S_i^h}
(\log\pi_{ij}^h-(\gamma_hD_{ij}^h+\lambda_hE_{ij}^h)/\tau_h).
$$
With $\gamma_h=1$, the current Gaussian-KL row is recovered only if $\lambda_hE_j$ is source-independent or absent. This follows exactly from probability ratios. The reduced value is a negative log partition; stationarity cancels explicit first-order row-response terms. Deleting entropy and differentiating through the same row changes the force by $-\tau_h^{-1}\operatorname{Cov}_{\beta^*}(C_j,\partial C_j)$ for row cost $C_j$.

For $B_h=A_h\Sigma_iA_h^\top$ and $P_{K,hj}=V_{hj}^{-1}$,
$$
\nabla_\mu D_{ij}^h=A_h^\top P_{K,hj}(A_h\mu_i-b_{hj}),\qquad
\partial_\Sigma D_{ij}^h=\frac12A_h^\top(P_{K,hj}-B_h^{-1})A_h.
$$
For fixed rows put
$$
\begin{aligned}
\Lambda_i^c&=a_i\Sigma_{p,i}^{-1}+\sum_{h,j}\beta_{ij}^h
(\gamma_hA_h^\top P_{K,hj}A_h+\lambda_hP_{ij}^h),\\
\eta_i^c&=a_i\Sigma_{p,i}^{-1}\mu_{p,i}+\sum_{h,j}\beta_{ij}^h
(\gamma_hA_h^\top P_{K,hj}b_{hj}+\lambda_hP_{ij}^hv_{ij}^h),\\
c_i^{content}&=a_i+\sum_h\lambda_h.
\end{aligned}
$$
Then
$$
\nabla_\mu F_i^c=\Lambda_i^c\mu_i-\eta_i^c,\qquad
2\partial_\Sigma F_i^c=\Lambda_i^c-c_i^{content}\Sigma_i^{-1}
-\sum_h\gamma_hA_h^\top(A_h\Sigma_iA_h^\top)^{-1}A_h.
$$
The mean uses a full precision solve, but covariance stationarity is nonlinear for arbitrary rectangular comparison maps. Its fixed-row covariance objective is
$$
\operatorname{tr}(\Lambda_i^c\Sigma_i)/2-c_i^{content}\log\det\Sigma_i/2
-\sum_h\gamma_h\log\det(A_h\Sigma_iA_h^\top)/2.
$$
It has a unique SPD minimizer: the full negative log determinant is strictly convex, projected ones are convex, positive linear growth dominates logarithms at infinity, and the full entropy diverges at the SPD boundary on bounded sets. Joint convexity after routing refresh is not asserted. A stationary check is
$$
\operatorname{tr}(\Lambda_i^c\Sigma_i^*)=c_i^{content}K+\sum_h\gamma_hr_h.
$$
A scalar-mass formula $\Sigma^*=c(\Lambda_i^c)^{-1}$ is generally wrong. Square invertible comparison maps give the special mass $c_i^{content}+\sum_h\gamma_h$ because each pullback becomes $\Sigma^{-1}$. Full-input maps need not be square. A diagonal restriction retains the diagonal of every projected-entropy pullback. The same Fisher metric supplies natural updates; SPD preservation alone does not imply finite-step descent.

At unit coefficients/temperature, a normalized augmented model realizes this exact constrained objective. Introduce an anchor $X_0$, head receiver copies $X_h$, comparison copies $Y_h$, and labels $J_h$:
$$
P=p_i(x_0)\prod_h\pi_{ij_h}^hg_{ij_h}^h(x_h)p_{K,j_h}^h(y_h),\qquad
Q=q_i(x_0)\prod_hq_i(x_h)((A_h)_*q_i)(y_h)\beta_{ij_h}^h.
$$
All model factors normalize. The recognition copies are independent with tied parameters. Expanding $E_Q[\log Q-\log P]$ gives the unit-coefficient coupled objective. It does not identify these copies as one token latent draw and has no observed language likelihood. General real weights and independently chosen temperatures define the displayed generalized energy, not the negative ELBO of this same joint.

## Shared key/value latent correction

For $Z_j\sim N(\xi_j,C_j)$, $Y=S_KZ_j+\epsilon$, $\epsilon\sim N(0,R_K)$ independent, $V=TZ_j$, and $V_K=S_KC_jS_K^\top+R_K$, Gaussian conditioning gives
$$
E[V\mid Y=y,J=j]=T\xi_j+TC_jS_K^\top V_K^{-1}(y-S_K\xi_j),
$$
$$
\operatorname{Cov}(V\mid Y,J=j)=TC_jT^\top-TC_jS_K^\top V_K^{-1}S_KC_jT^\top.
$$
Completing the square, or separating the Gaussian residual with zero cross covariance, proves these identities. Receiving by $U_iO$ propagates both. Averaging under a query covariance $V_Q$ adds $TC_jS_K^\top V_K^{-1}V_QV_K^{-1}S_KC_jT^\top$. Uncorrected values give this mean for every query exactly when $TC_jS_K^\top=0$. Full input permits overlap and does not justify deleting this term. Independent content laws or a predictive operation are different declared models.

## Candidate 3: predictive Gaussian relaxation control

Retain incoming projected routing and the kernel $X_i^+=rX_i^-+\sum_hL_{iJ_h}^hX_{J_h}^-+\epsilon_i$, with independent head labels conditioned on the snapshot. Its exact mean is $m_i^{pred}=r\mu_i^-+\sum\bar\beta L\mu_j^-$. The existing covariance includes source uncertainty, source spread, cross-head covariance from shared source draws, and residual/self-edge terms. Independent source laws are an additional assumption; correlated sources add cross-agent covariance blocks.

Let $P_i^{pred}$ be that mixture and $G_i^{pred}=N(m_i^{pred},C_i^{pred})$ its Gaussian moment projection, with $C_i^{pred}\succ0$. Expanding Gaussian cross entropy proves this is the unique inclusive projection $\arg\min_G KL(P_i^{pred}\Vert G)$, generally different from exclusive Gaussian inference against the mixture.

Define $F_i^{relax}(q_i)=E_{q_i}[\log q_i-\log G_i^{pred}]$. Its unique optimum is $G_i^{pred}$ and its exact Fisher flow is Candidate 1 with $c=1$, $\Lambda=(C_i^{pred})^{-1}$, $\eta=\Lambda m_i^{pred}$. A solved relaxation exactly preserves the predictive mean/moment-covariance endpoint and all maps/router. Finite settling changes the operator. A unit mean step reaches the target only when its fixed preconditioner equals $C_i^{pred}$.

The omitted mixture term is
$$
KL(q_i\Vert P_i^{pred})-KL(q_i\Vert G_i^{pred})
=E_{q_i}\log[G_i^{pred}(X)/p_i^{pred}(X)].
$$
It has no universal sign for arbitrary $q_i$. At $q_i=G_i^{pred}$ it is nonnegative, vanishing only for equal laws. Positive receiving noise and finite labels make these log expectations finite. This is a parity control, not a new generative VFE for the whole transformer.

## Fast partials, slow maps, and observations

Holding sources fixed in the inner receiver partial does not remove their outer contribution. For $q_i^+=\Phi_i(q_i^-,s^-,\theta)$,
$$
dq_i^+/d\theta=\partial_q\Phi_idq_i^-/d\theta
+\partial_s\Phi_ids^-/d\theta+\partial_\theta\Phi_i.
$$
Analytic partial forces can retain the full source/template graph. An independent receiver probe does the same with automatic differentiation. Detaching sources from the outer graph deletes cross-token credit and is an additional choice. Fixed per-episode routing can remain live to slow parameters.

A true shared-functional M-step holds recognition coordinates fixed and differentiates the same objective, including determinants, covariance dependence, and model normalizers. For one content target with $v=L\mu_j$, $R=Q+L\Sigma_jL^\top$, $P=R^{-1}$,
$$
\partial_LE=P(v-\mu)\mu_j^\top
+[P-P(\Sigma+(\mu-v)(\mu-v)^\top)P]L\Sigma_j.
$$
Freezing content covariance removes the second term and changes the M-step. Derivatives of $T/O$ follow from $dL=U_i(dOT+OdT)U_j^{-1}$, with source/receiver beliefs fixed.

If slow learning instead minimizes outer next-token CE through the fast map, call it discriminative end-to-end learning with iterative belief refinement, or disclose the structural-EM analogy. It is not coordinate variational EM on one functional. Target tokens are observations in the outer normalized autoregressive likelihood; the target-blind inner energy has no next-token likelihood. CE can be an observation term of a genuine VFE: the missing shared latent joint/objective is the issue. CE at the Gaussian mean also generally differs from expected latent negative log likelihood.

The causal fast step excludes the held-out target. Candidate-dependent fast states do not automatically define one normalized predictive conditional. A training posterior may use observed targets in a declared latent model, but its prediction kernel and training bound then need a separate consistent specification.

## Recommendation and unresolved obligation

Use fixed incoming Q/K routing plus full $L$ receiving targets as the first research refactor. Preserve the slow four-map graph, define SPD receiving noise, choose full/diagonal covariance chart and fixed per-episode weights, and implement analytic receiver forces. Gradient/exact-flow/MM alternatives should name the same frozen energy. Do not claim joint routing descent or predictive endpoint equality.

Content routing and coupled comparison/content routing are separate alternatives. Their displayed rows are exact coordinates of their declared scalar energy; the coupled case changes routing and covariance stationarity. Predictive Gaussian relaxation is a control for finite settling with the predictive target fixed.

The minimal unresolved model choice is the SPD receiving law and the chosen router/covariance combination. The empirical obligation is a matched full-link E-step comparison with the predictive route. The next action that materially changes the answer is to select one declared route, implement its analytic receiver force with live outer graph, and validate its fixed-snapshot objective/gradient/endpoint before training.
