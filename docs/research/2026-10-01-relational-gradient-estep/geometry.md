# Full-state matrix links for a transported gradient E-step

Date: October 1, 2026. Research-only assignment. This note changes no implementation, configuration, checkpoint, environment, training process, or Research vault file. The concurrently edited predictive full/full route is outside its ownership.

The direct replacement Omega_ij -> L_ij is mathematically valid after defining a nonsingular receiving distribution. A low-rank content map cannot use the noiseless ambient Gaussian density supplied by invertible frame transport. The first refactor can use full-input/full-output maps with fixed positive model noise, without introducing an independent group connection. Its Gaussian target remains source-factorized. A genuine connection is a separate extension.

## Target, assumptions, and sources

For every finite agent count N, every dimension K, every invertible frame collection U_i, every collection of full Gaussian states q_i, and every full-input/full-output collection of head maps, construct an own-row inference energy using L_ij that is finite, normalized at the channel level, and covariant under independent endpoint coordinate reexpressions. The equivalence sought is passive covariance of the declared mathematical inference problem. It is not equality with predictive attention, standard Transformer scores, a language-model ELBO, or a perplexity claim. The algebra applies to arbitrary finite N, including N >= 512; it does not establish target-scale throughput.

The user supplies full-state projection support, the primary direct-link hypothesis, and slow-M-step learned maps. The reported perplexity gap motivates this investigation but is not a matched causal result here. I introduce fixed frames, frozen sender snapshots, fixed model noise, and optionally frozen connection links during one fast own-row solve. These assumptions permit the exact frozen-target result below. Updating senders, frames, noise, or links inside that solve requires their derivatives and retains a separate coupled obligation. Holding slow maps fixed during fast inference does not determine whether outer training differentiates through its solution.

The live [attention manuscript](<C:/Users/chris and christine/Desktop/Research/manuscripts/Attention/GL(K)_attention_v2.tex:118>) already separates communication heads from frame blocks and gives full-state role interfaces. Its [matrix-link section](<C:/Users/chris and christine/Desktop/Research/manuscripts/Attention/GL(K)_attention_v2.tex:712>) establishes rank bounds and normalized noisy receiving laws. Its [comparison lift](<C:/Users/chris and christine/Desktop/Research/manuscripts/Attention/GL(K)_attention_v2.tex:775>) differs from learned T/O content maps. Its [decomposition](<C:/Users/chris and christine/Desktop/Research/manuscripts/Attention/GL(K)_attention_v2.tex:820>) gives L=F Omega and permits an independent F Theta. The [gradients manuscript](<C:/Users/chris and christine/Desktop/Research/manuscripts/Attention/GL(K)_gradients.tex:283>) fixes the connection sign; its [GL(K) preconditioning section](<C:/Users/chris and christine/Desktop/Research/manuscripts/Attention/GL(K)_gradients.tex:735>) states the limitations of constant Cartan and exponential-coordinate metrics.

The consulted wiki pages were [GL(K) gauge-equivariant attention](<C:/Users/chris and christine/Desktop/Research/wiki/concepts/GL(K) gauge-equivariant attention.md>), [Holonomy](<C:/Users/chris and christine/Desktop/Research/wiki/concepts/Holonomy.md>), and [Lattice gauge theory](<C:/Users/chris and christine/Desktop/Research/wiki/concepts/Lattice gauge theory.md>), with their pertinent current manuscript, Wetterich, Wilson, Milnor, and Pennec source notes. Historical descriptions are not current software specifications. The prior [architecture investigation](<C:/Users/chris and christine/Desktop/VFE_3.0/docs/investigations/full-relational-2026-10-01/architecture.md>) establishes the full/full projection support and current route limits; this note resumes at the remaining inference-law choice.

Consulted live manuscript SHA-256: GL(K)_attention_v2.tex = B162C4EEB38ABBDCFEC4D7A6A46BD798FFA901465E4A18852F629D915C8F0ABD; GL(K)_gradients.tex = DF56FB4E4451F96C5F7FF3B34F748786F6124FDE3AA2CCE64FC986E007FCB46E. These identify source bytes, without assigning a verification closure.

## Typed dictionary

Layer indices are suppressed. ij means sender j to receiver i. Matrices act on columns and compose right to left.

| Object | Type and convention |
|---|---|
| Token base | I={0,...,N-1}, with a separately declared graph. Causal support has j <= i. |
| Optional contextual base | Smooth C; the current token construction pulls back along the constant map i -> c_0 and sees no contextual connection or topology. |
| Principal bundle | P -> C, with free principal right action n -> n dot g. Mathematical real structure group GL(K,R); current projected code uses real block_glk represented block diagonally on V. |
| Associated fibers | E_i is K-dimensional; its statistical fiber consists of full Gaussians N(mu_i,Sigma_i), Sigma_i positive definite. Covariance is E_i^* -> E_i. |
| Quotient and local section | (n dot g,v) ~ (n,gv). Rechoice u_i -> u_i dot g_i^-1 gives local coordinates v_i -> g_i v_i. |
| Reference frame | V=R^K; reference section sigma_0 and local section satisfy u_i=sigma_0 dot U_i^-1. U_i:V -> E_i. |
| Reference moments | xi_i=U_i^-1 mu_i, C_i=U_i^-1 Sigma_i U_i^-T. |
| Representation | Left defining action g:v -> gv; Gaussian action (mu,Sigma) -> (g mu,g Sigma g^T); covectors transform by g^-T. |
| Comparison interfaces | S_Q,h,S_K,h:V -> H_cmp,h=R^(r_h), full row rank for nonsingular projected Gaussians. |
| Content interfaces | T_h:V -> H_val,h=R^(s_h), O_h:H_val,h -> V; M_h=O_h T_h has rank at most s_h. Full scope specifies support, not rank. |
| Local endpoint maps | S_Q,h U_i^-1 and S_K,h U_i^-1 measure; T_h U_j^-1 sends; U_i O_h receives. |
| Content link | L_ij^h=U_i M_h U_j^-1 in Hom(E_j,E_i), possibly singular. |
| Flat identification | Omega_ij=U_i U_j^-1, invertible, with inverse reversal and Omega_ij Omega_jk=Omega_ik. |
| Optional connection | Theta_ij:E_j -> E_i, group-valued in chosen frames, independently declared; Theta_ji=Theta_ij^-1. |
| Metrics | Reference inner product H=I in one chosen basis; local metric G_i=U_i^-T H U_i^-1; Gaussian Fisher metric below. |
| Model noise | Fixed Q_h positive definite on V; Q_i,h=U_i Q_h U_i^T on E_i. This is probability-model noise, distinct from numerical jitter. |

Passive endpoint coordinates transform as

$$
\mu_i'=g_i\mu_i,\quad \Sigma_i'=g_i\Sigma_i g_i^T,\quad U_i'=g_iU_i,\quad
G_i'=g_i^{-T}G_i g_i^{-1}.
$$

Then xi_i'=xi_i, C_i'=C_i, and reference maps/noise stay fixed. Composition gives

$$
L_{ij}^{h\prime}=g_iL_{ij}^h g_j^{-1},\qquad Q_{i,h}'=g_iQ_{i,h}g_i^T.
$$

These are passive tensor laws. Holding a numerical metric fixed instead restricts active metric-preserving maps to its orthogonal group. Arbitrary GL(K) congruence also leaves a diagonal-covariance model family, so the full-Gaussian statement does not certify diagonal storage or whole-stack covariance of coordinate normalization, MLP, and decoder.

A common reference-basis change U_i -> U_i a requires xi_i -> a^-1 xi_i, C_i -> a^-1 C_i a^-T, S -> S a, T -> T a, O -> a^-1 O, Q_h -> a^-1 Q_h a^-T, and H -> a^T H a. Resetting noise to the same epsilon I after a nonorthogonal reference change defines a different model. Choosing epsilon I in one fixed reference basis remains compatible with passive site covariance.

## Direct Omega-to-L receiving law and energy

Fix sender X_j ~ q_j and independent receiver noise epsilon_i,h ~ N(0,Q_i,h). Define

$$
Y_{ij}^h=L_{ij}^hX_j+\epsilon_{i,h},\qquad
p_{ij}^h=\mathcal N(m_{ij}^h,V_{ij}^h),
$$

$$
m_{ij}^h=L_{ij}^h\mu_j,\qquad
V_{ij}^h=Q_{i,h}+L_{ij}^h\Sigma_jL_{ij}^{hT}.
$$

The conditional law integrates to one for every source value and every rank of L. Independence gives the marginal mean and covariance. For nonzero v, v^T V v >= v^T Q v > 0, so every full receiver Gaussian has finite cost

$$
E_{ij}^h=\frac12\left[
\log\frac{\det V_{ij}^h}{\det\Sigma_i}-K+
\operatorname{Tr}((V_{ij}^h)^{-1}\Sigma_i)+
(\mu_i-m_{ij}^h)^T(V_{ij}^h)^{-1}(\mu_i-m_{ij}^h)
\right].
$$

The endpoint laws give m'=g_i m and V'=g_i V g_i^T. Inverse congruence and cyclic trace leave the trace and quadratic terms unchanged. Both determinants acquire det(g_i)^2, which cancels. This proves E_ij^{h prime}=E_ij^h for every stated input, independently of link rank.

Applying U_i^-1 gives the equivalent reference target

$$
\widetilde p_{hj}=\mathcal N(M_h\xi_j,R_{hj}),\qquad
R_{hj}=Q_h+M_hC_jM_h^T,
$$

and E_ij^h=KL(N(xi_i,C_i) || ptilde_hj). The target depends on h,j, without reference-coordinate dependence on receiver i.

The minimal counterexample to bare ambient substitution is K=2, identity frames, T=[1,0], O=[1,0]^T, and Sigma_j=I. Then L=diag(1,0) and L Sigma_j L^T=diag(1,0). The noiseless pushforward is supported on a line; a full receiver Gaussian assigns probability one off that line, so it is not absolutely continuous with respect to that target and KL is infinite. Positive epsilon I gives diag(1+epsilon,epsilon). This exact low-dimensional example validates the rank/support condition; it does not set a large-population research direction.

For q_i=N(0,I), that noisy cost is

$$
E_\epsilon=\frac12[\log((1+\epsilon)\epsilon)-2+(1+\epsilon)^{-1}+\epsilon^{-1}]
\longrightarrow+\infty,\qquad \epsilon\downarrow0.
$$

Thus vanishing noise does not recover a finite legacy cost for singular L with a fixed full receiver. When M_h=I_K, L=Omega is invertible and Q_h=0 is permitted, recovering the old target exactly. For full-rank M, the zero-noise limit is regular for fixed full states. Keeping positive noise at M=I defines a noisy legacy target.

Noise magnitude controls how strongly an incoming content law constrains private/null directions. Making Q depend on the current receiver covariance changes the objective derivative. A floor used only inside a numerical factorization cannot silently count as fixed model noise.

## Fast own-row inference and distinct routing objectives

At fixed maps, frames, sender snapshots, noise, and nonnegative weights w_ijh, let

$$
\mathcal F_i(q_i)=\alpha_i\operatorname{KL}(q_i\|p_i)+
\sum_{j,h}w_{ijh}\operatorname{KL}(q_i\|p_{ij}^h),
\qquad \alpha_i\ge0,\quad s_i=\alpha_i+\sum_{j,h}w_{ijh}>0.
$$

Define

$$
P_i=\alpha_i\Sigma_{p,i}^{-1}+\sum_{j,h}w_{ijh}(V_{ij}^h)^{-1},\qquad
b_i=\alpha_i\Sigma_{p,i}^{-1}\mu_{p,i}
+\sum_{j,h}w_{ijh}(V_{ij}^h)^{-1}m_{ij}^h.
$$

All targets have positive precision; positive total mass makes P_i positive definite. Gaussian differentiation gives

$$
\partial_{\mu_i}\mathcal F_i=P_i\mu_i-b_i,\qquad
\partial_{\Sigma_i}\mathcal F_i=\frac12(P_i-s_i\Sigma_i^{-1}).
$$

Hence the unique full-Gaussian minimizer is

$$
\mu_i^*=P_i^{-1}b_i,\qquad \Sigma_i^*=s_iP_i^{-1}.
$$

For a global check, the mean Hessian is P_i positive definite. The covariance objective is one half of Tr(P_i Sigma_i)-s_i logdet Sigma_i plus constants, with second variation s_i Tr(Sigma_i^-1 Delta Sigma_i^-1 Delta)/2 > 0 for nonzero symmetric Delta. Completing the mean square and minimizing the covariance proves uniqueness. These are frozen-target results. They differ from predictive moment matching and do not claim execution by an existing mm_exact kernel.

Direct replacement uses E_ij^h in both the fast row energy and the categorical source update:

$$
\mathcal J_i^h(q_i,\beta)=\sum_j\beta_{ij}^hE_{ij}^h(q_i)
+\tau_h\operatorname{KL}(\beta_i^h\|\pi_i^h),\qquad
\beta_{ij}^{h*}\propto\pi_{ij}^h\exp(-E_{ij}^h/\tau_h).
$$

Finite costs, nonempty support, and tau_h>0 give the unique categorical minimizer. At categorical stationarity its reduced derivative is sum_j beta_ij^{h*} dE_ij^h; the first-order beta response cancels. Exact alternating minimizations or accepted descent steps descend this frozen own-row objective. This is not an unconditional finite-step, coupled-population, or language-model descent theorem.

Retaining S_Q/S_K gives a different routing cost,

$$
D_{ij}^h=\operatorname{KL}\left(
\mathcal N(S_{Q,h}\xi_i,S_{Q,h}C_iS_{Q,h}^T)
\|\mathcal N(S_{K,h}\xi_j,S_{K,h}C_jS_{K,h}^T)
\right).
$$

Beta minimizing D is generally not a coordinate minimizer for a free energy with pair costs E. Freezing this separately routed beta during q inference gives a valid conditional surrogate. Differentiating sum beta(D) E gives an extra sum E d beta term, which is not the envelope derivative for E. A combined source objective using D+lambda E is also valid and yields exp(-(D+lambda E)/tau) routing, while changing the predictive router. Geometry does not choose among these objectives; the refactor must.

The comparison-induced lift already in the manuscript is

$$
R_{Q,i}=C_iS_Q^T(S_QC_iS_Q^T)^{-1},\qquad S_QR_{Q,i}=I.
$$

It uniquely minimizes delta^T C_i^-1 delta subject to S_Q delta=y. Write delta=R_Q,i y+w with S_Q w=0; its cross term is y^T(S_Q C_i S_Q^T)^-1 S_Q w=0, leaving the strictly positive w penalty. The Gaussian conditional mean is xi_i+R_Q,i(y-S_Q xi_i). This is a full-coordinate update without identifying R_Q,i with O or S_K with T. The induced link U_i R_Q,i S_K U_j^-1 equals U_i O T U_j^-1 only under additional map identities. A comparison bilinear tensor first maps E_j into E_i^*; raising its first index by a declared metric is a lift, not an inverse rectangular query map.

## Source-factorized full-state KL without dense pair covariances

Put B_hj=T_h C_j T_h^T, which is positive semidefinite even if T is rank deficient. Then R_hj=Q_h+O_h B_hj O_h^T. Set

$$
A_h=O_h^TQ_h^{-1}O_h,\qquad
J_{hj}=B_{hj}^{1/2}
(I+B_{hj}^{1/2}A_hB_{hj}^{1/2})^{-1}B_{hj}^{1/2}.
$$

The small core is positive definite, and no inverse of B is required. With W=O B^(1/2), the identities are

$$
R_{hj}^{-1}=Q_h^{-1}-Q_h^{-1}O_hJ_{hj}O_h^TQ_h^{-1},
$$

$$
\log\det R_{hj}=\log\det Q_h+
\log\det(I+B_{hj}^{1/2}A_hB_{hj}^{1/2}).
$$

For an explicit inverse check, (Q+WW^T)Q^-1 W=W(I+W^TQ^-1W). Substituting this identity makes multiplication of the proposed inverse by Q+WW^T equal I. The determinant identity follows by eliminating Q and then I in the block matrix [[Q,W],[-W^T,I]]. Thus singular B is an examined, allowed case.

Precompute receiver and sender quantities

$$
D_{hi}=O_h^TQ_h^{-1}C_iQ_h^{-1}O_h,\qquad
u_{hi}=O_h^TQ_h^{-1}\xi_i,\qquad v_{hj}=T_h\xi_j.
$$

For delta_ij=xi_i-O_h v_hj,

$$
\operatorname{Tr}(R_{hj}^{-1}C_i)
=\operatorname{Tr}(Q_h^{-1}C_i)-\operatorname{Tr}(J_{hj}D_{hi}),
$$

$$
\delta_{ij}^TR_{hj}^{-1}\delta_{ij}
=\xi_i^TQ_h^{-1}\xi_i-2u_{hi}^Tv_{hj}+v_{hj}^TA_hv_{hj}
-(u_{hi}-A_hv_{hj})^TJ_{hj}(u_{hi}-A_hv_{hj}).
$$

Pair-specific contractions are scalar or value-dimensional. Source cores and log determinants are cached per h,j; no N by N by K by K target tensor is necessary. Full receiver covariances, source projections, and scalar routing scores still have costs. These are exact algebraic reductions, not measured speed claims or guarantees for a diagonal-family approximation.

For the frozen solve, the precision sum can aggregate corrections by receiver:

$$
\sum_jw_{ijh}R_{hj}^{-1}
=(\sum_jw_{ijh})Q_h^{-1}-
Q_h^{-1}O_h(\sum_jw_{ijh}J_{hj})O_h^TQ_h^{-1}.
$$

The final K-dimensional inversion may remain expensive, but individual pair precisions need not be materialized. Fixed shared reference noise preserves source factorization. General pair-dependent transport or receiver-dependent reference noise loses some of this caching.

## Independent Theta extension, ordering, and comparison

Let F_i,h=U_i M_h U_i^-1. Then L_ij^h=F_i,h Omega_ij. A separately specified connection allows

$$
\mathcal L_{ij}^h=F_{i,h}\Theta_{ij},\qquad
\Theta_{ij}'=g_i\Theta_{ij}g_j^{-1}.
$$

Since F_i,h'=g_i F_i,h g_i^-1, the extended link obeys the same endpoint law. Define reference A_ij=U_i^-1 Theta_ij U_j. It is site-invariant, A_ji=A_ij^-1, and

$$
\mathcal L_{ij}^h=U_iM_hA_{ij}U_j^{-1}.
$$

This transports the full source first, then applies receiver processing. Processing before transport instead gives U_i A_ij M_h U_j^-1; equality requires commutation. A head-space link O_h a_ij,h T_h is a third construction. General rectangular T/O do not induce a full-state group representation on the head space without intertwining conditions.

A content-only Theta keeps D(S_Q xi_i,S_K xi_j) routing while changing content to M_h A_ij. A connection-aware comparison instead compares the query against key moments S_K A_ij xi_j and S_K A_ij C_j A_ij^T S_K^T. Both are passively invariant and are different models. Direct full-state Theta content targets use covariance Q_h+M_h A_ij C_j A_ij^T M_h^T, generally depending on i and j and losing source factorization.

A Theta generated from current q depends on fast variables unless explicitly frozen as a snapshot; its derivatives cannot then be omitted. Headwise independent Theta fields define H connections, rather than automatically one geometric connection. Shared Theta with headwise F is the smaller extension.

## Reversal, loops, and metric conditions

Connection reversal is inversion. A separately processed reverse content link L_ji=U_j M_h U_i^-1 equals L_ij^-1, when M is invertible, only when M_h=M_h^-1. Reverse attention weights introduce another difference. No reverse attention row gives inverse transport automatically.

The endpoint metric adjoint is L_ij^dagger=G_j^-1 L_ij^T G_i. It transforms as g_j L_ij^dagger g_i^-1 but is generally not an inverse. Wetterich's compact-group matrix links use an adjoint reverse convention; its Eqs. (2), (11), and (20)-(21) give a primary precedent for general matrix links and a processing/transport decomposition. This is not a GL(K) inverse-reversal construction for singular attention links. [Linear lattice gauge theory](https://arxiv.org/html/1307.0722v2).

Omega paths telescope; every closed product is I. Content links compose as L_ij^h L_jk^h=U_i M_h^2 U_k^-1, so nonidentity communication products may express gain or rank loss without Omega holonomy. For declared Theta loops the endpoint factors cancel to conjugation at the base site, giving invariant trace and spectrum. The graph and reverse links must be declared first. Causal source indices are nonincreasing around a directed walk, so there is no distinct-position directed loop. A reversible structural graph can coexist with causal message use; its extra reverse geometry is additional data.

On a connected graph with inverse reversal, all closed products equal I if and only if Theta_ij=W_i W_j^-1 for some site frames W_i. For the converse, choose a root and let W_i be transport from the root along a path. Trivial closed products make this path-independent; appending an edge gives the factorization. Thus learned noncommuting site factors remain flat.

An exact minimal independent nonabelian example takes Theta_12=exp(a E_12), Theta_23=exp(b E_21), Theta_31=I in GL(2,R), with inverse reverse edges. The closed product is

$$
\Theta_{12}\Theta_{23}\Theta_{31}
=\begin{pmatrix}1+ab&a\\b&1\end{pmatrix}.
$$

It is nonidentity when a or b is nonzero, with explicit noncommutativity when ab != 0. This validates independent connection data; it does not demonstrate a large-N mechanism.

Fixed Frobenius edge norms are not GL invariant: L=E_11, g_i=diag(2,1), g_j=I changes Tr(L^T L) from 1 to 4. For K >= 2, no positive Lie-algebra inner product is invariant under all GL(K) conjugations: Ad_diag(exp(t),1,...)(E_12)=exp(t)E_12 would force a nonzero norm to equal exp(2t) times itself for every t. K=1 is abelian; noncompactness alone is insufficient for this obstruction. This direct proof avoids relying on inaccessible Milnor full text.

A valid positive edge metric, with transforming endpoint metrics, is

$$
\langle\Delta_1,\Delta_2\rangle
=\operatorname{Tr}(G_j^{-1}\Delta_1^TG_i\Delta_2).
$$

For dF=Tr(P^T Delta), its metric gradient is G_i^-1 P G_j. The first-order retraction

$$
\mathcal R_\Theta(\eta\Delta)=\Theta\exp(\eta\Theta^{-1}\Delta)
$$

stays in the connected group component and obeys endpoint covariance by similarity covariance of exp. It is a Lie-group retraction, not the Riemannian exponential of this extrinsic metric. It offers metric-preconditioned unrestricted-GL group-element updates; for a block or SU subgroup, the gradient must first be projected to that subgroup's tangent using the declared metric. A constant Cartan/Frobenius coordinate preconditioner is not automatically a full-GL natural gradient.

For full Gaussian beliefs, the Fisher metric is

$$
\langle(a,A),(b,B)\rangle_{(\mu,\Sigma)}
=a^T\Sigma^{-1}b+\frac12\operatorname{Tr}(\Sigma^{-1}A\Sigma^{-1}B),
$$

giving delta_mu=-Sigma partial_mu F and delta_Sigma=-2 Sigma(partial_Sigma F)Sigma. Contragredient derivative laws give delta_mu'=g delta_mu and delta_Sigma'=g delta_Sigma g^T.

For the covariance affine-invariant metric Tr(Sigma^-1 A Sigma^-1 B), or its constant one-half multiple, the exponential is

$$
\operatorname{Exp}_{\Sigma}(A)=\Sigma^{1/2}
\exp(\Sigma^{-1/2}A\Sigma^{-1/2})\Sigma^{1/2},
$$

and Log_Sigma(S)=Sigma^(1/2) log(Sigma^(-1/2) S Sigma^(-1/2)) Sigma^(1/2). These are actual covariance manifold maps, as in Section 3.4 of [Pennec, Fillard, and Ayache's primary report](https://www-sop.inria.fr/asclepios/Publications/Xavier.Pennec/Pennec_RR_Tensors.pdf). Congruence covariance follows from Exp_Sigma(A)=Sigma exp(Sigma^-1 A) and matrix-exp similarity. The vector-fiber pushforward Sigma -> Theta Sigma Theta^T is not SPD Levi-Civita tangent transport along a covariance geodesic. A covariance exponential alone introduces no model SPD holonomy.

## Target-scale connection families and group domains

Independent dense K by K maps on every edge require |E| K^2 parameters before optimizer state. At N=K=512, one orientation for each distinct pair has 130,816 edges, about 34.3 billion entries or 137 GB in float32. The general formula, rather than N=K specifically, is the constraint. Dense all-pairs connection storage is not the proposal.

| Family | Exact construction and limitations |
|---|---|
| Sparse graph | O(N m) structural edges for bounded neighbor count m; store one orientation and obtain its reverse by inversion. A tree has no independent cycle holonomy; chords add it. Missing direct edges need declared paths, with path dependence in a nonflat field. |
| Generated small algebra | Shared p generators B_a and on-demand Z_ij=sum_a c_ij,a B_a, A_ij=exp(Z_ij). Z_ji=-Z_ij gives exact reversal for a single exponential. This constrains edge freedom; p dense generators cost p K^2 once, while dense pair exponentiation may remain expensive. |
| Low-rank algebra action | Z=P R, shapes K by r and r by K; exp(Z)=I+P phi_1(RP)R, phi_1(X)=sum_(n>=0) X^n/(n+1)!. The power identity proves this without inverses. It gives small-core action evaluation but restricts active subspaces. Noncommuting generators must be included if intended. |
| Ordered simple factors | Sparse/block/shear factors give invertible actions. Reverse order and invert each factor for reversal; coefficient negation in unchanged order is not inversion when factors do not commute. Coverage and conditioning are separate obligations. |
| Site factors only | A_ij=A_i A_j^-1 is cheap and flat; it does not create the desired independent connection field. |

For GL(K,R), exp(Z) has determinant exp(Tr Z)>0. A single real exponential therefore supplies a subfamily of GL^+(K), not negative components or every positive-determinant element. Negative components require an explicit component choice; multiplicative exponential updates cannot cross them.

For GL(K,C), complex-linear links use arbitrary complex generators and Hermitian covariance congruence. Proper complex Gaussians require Hermitian positive covariance and zero pseudo-covariance; their KL uses complex normalization rather than the real one-half formula. Alternatively, realification gives a 2K-dimensional real model with complex-linear subgroup [[Re A,-Im A],[Im A,Re A]]. These are extensions, not existing projected executors.

For SU(m), use V=C^m and skew-Hermitian traceless Z. Then exp(Z) is unitary with determinant one; the Euclidean Hermitian metric is preserved. The positive pairing -Re Tr(XY) on su(m) is Ad invariant. For its bi-invariant metric, the group exponential is the Riemannian exponential. Content maps remain arbitrary and may be singular. Here m is group dimension and N counts agents; the current real projected executor does not demonstrate a complex/SU(m) route.

Current [projected admission](<C:/Users/chris and christine/Desktop/VFE_3.0/vfe3/config.py:923>) requires block_glk, flat transport, fixed frames (e_phi_lr=0), filtering, gradient/unroll, and one E-step. The [frame checker](<C:/Users/chris and christine/Desktop/VFE_3.0/vfe3/gradients/relational.py:42>) rejects genuinely dense off-block multihead inverse frames. Full O/T communication does not enlarge the frame group. A block-diagonal reference A keeps Theta in block_glk; a block-mixing A needs a larger structure group or an explicitly external transport representation.

## Contextual overlaps and global compatibility

The finite constant token pullback admits the working reference trivialization. A graph connection can have cycle holonomy on that trivial carrier; this is different from nontrivial topology of a contextual principal bundle.

For local reference sections sigma_b=sigma_a dot t_ab, transitions satisfy t_ab t_bc=t_ac and coordinates xi_a=t_ab xi_b. The same communication rule on overlaps requires

$$
U_{i,b}=U_{i,a}t_{ab},\quad S_b=S_a t_{ab},\quad
T_b=T_a t_{ab},\quad O_b=t_{ab}^{-1}O_a,\quad
Q_b=t_{ab}^{-1}Q_a t_{ab}^{-T}.
$$

These laws make links and target costs agree. Holding the same numerical S,T,O in every chart instead imposes intertwining/descent constraints. A nontrivial bundle need not admit global sigma_0. Rectangular endpoints would then be sections of appropriate Hom bundles, with a declared head bundle and its transitions; globally stored constant matrices do not supply that descent.

For local columns with D=d+A, the overlap law is A_a=t_ab A_b t_ab^-1-(d t_ab)t_ab^-1. With v_i=U_i v_0, A_i=-(dU_i)U_i^-1 is compatible and pure gauge. Independent connection transports along declared paths obey the same endpoint law but may depend on paths. These conditions are required before a global contextual extension is claimed.

## Supported result and next obligation

The supported result is a normalized, passively covariant full-state direct-L receiving law for all finite N and all link ranks, with exact frozen Gaussian energy/minimizer and source-factorized low-rank evaluation. The rank counterexample explains why generic low-rank heads need positive receiving noise for ambient full-Gaussian KL. It proves neither a perplexity improvement nor a common population VFE or observation model for the outer cross-entropy objective.

The minimal first-refactor obligation is to choose the fast objective: L energies routing themselves, separately routed S_Q/S_K weights held fixed during content inference, or a combined categorical cost. Then declare fixed model noise, source freezing and self-edge treatment, covariance family, and whether frames also update. These change the gradients and available closed form. Slow-M-step map ownership is already supplied.

Theta should be later independent work unless connection/path information is itself the immediate hypothesis. It adds pair-dependent geometry, reversal/path requirements, metric choices, and potential loss of source factorization while being unnecessary for direct Omega-to-L replacement. The next action that changes the answer is an explicit noisy-L objective specification, followed by tiny exact algebra/gradient checks against its reference-coordinate oracle and a target-scale script after implementation and GPU authorization.

No numerical experiments or tests ran for this note. Retrieval read the live sources, Wetterich's primary Section II/III, and Pennec's primary exp/log formulas. Wilson's publisher abstract was accessible but its PDF returned 401; Milnor's publisher full text was unavailable. Assertions depending on those unavailable full texts were proved directly here or not used. Literature retrieval was bounded source following, not a systematic novelty review.
