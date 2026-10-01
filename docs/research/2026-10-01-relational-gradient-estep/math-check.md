# Independent check of the full-link receiver construction

October 1, 2026. This is a bounded mathematical reconstruction, not an implementation audit or a perplexity judgment. The independent result below was recorded before comparison with the specialist draft.

## Exact conditional target

For every finite \(K\ge1\) and every finite collection of pairs \((h,j)\), fix Gaussian source moments \((\mu_j,\Sigma_j)\), with \(\Sigma_j\succeq0\), invertible frames \(U_i,U_j\), arbitrary linear maps \(T_h:\mathbb R^K\to\mathbb R^{d_h}\), \(O_h:\mathbb R^{d_h}\to\mathbb R^K\), and receiving noise \(Q_h\succ0\). Define

$$
L_{ij}^{(h)}=U_iO_hT_hU_j^{-1},\quad
Q_{i,h}=U_iQ_hU_i^\top,\quad
m_{hj}=L_{ij}^{(h)}\mu_j,\quad
R_{hj}=L_{ij}^{(h)}\Sigma_jL_{ij}^{(h)\top}+Q_{i,h}.
$$

Fix a full-width prior \(p_i=\mathcal N(\mu_p,\Sigma_p)\), \(\Sigma_p\succ0\), and scalar weights \(a,w_{hj}\ge0\), with \(c=a+\sum_{hj}w_{hj}>0\). Minimize

$$
F(q)=aD_{\rm KL}(q\Vert p_i)+\sum_{hj}w_{hj}D_{\rm KL}(q\Vert\mathcal N(m_{hj},R_{hj})),
\quad q=\mathcal N(\mu,\Sigma),\quad\Sigma\succ0.
$$

These are the supplied conditional assumptions. All sources, frames, maps, noise, weights, masks and prior are fixed during receiver optimization. Receiving noise is an additional modeling choice; T/O alone do not specify it. Equivalence means equality of this scalar functional up to a receiver-independent constant and equality of its minimizer/flow. It does not mean equality to the predictive residual, the outer cross-entropy loss or a population VFE.

Live source continuity: [GL(K)_attention_v2.tex](<C:/Users/chris and christine/Desktop/Research/manuscripts/Attention/GL(K)_attention_v2.tex:127>) defines the comparison/content maps and full input, lines 216–240 define L and its endpoint law, and lines 731–752 explicitly give the noisy Gaussian receiving marginal at arbitrary link rank. [GL(K)_gradients.tex](<C:/Users/chris and christine/Desktop/Research/manuscripts/Attention/GL(K)_gradients.tex:798>) gives the Gaussian Fisher tangent convention. The prior [mm-exact.md](../../investigations/full-relational-2026-10-01/mm-exact.md) already distinguishes full-content geometric pooling from projected-comparison KL. These sources establish continuity, not correctness by authority.

## Positivity and minimizer

For nonzero x, \(x^\top R_{hj}x>0\): its source term is nonnegative and its noise term \((U_i^\top x)^\top Q_h(U_i^\top x)\) is strictly positive. This handles rectangular or deficient-rank content links. Set

$$
\Lambda=a\Sigma_p^{-1}+\sum_{hj}w_{hj}R_{hj}^{-1},\qquad
\eta=a\Sigma_p^{-1}\mu_p+\sum_{hj}w_{hj}R_{hj}^{-1}m_{hj}.
$$

At least one positive weight multiplies a positive-definite precision, so \(\Lambda\succ0\). Gaussian KL expansion gives

$$
F(\mu,\Sigma)=\frac12[
\operatorname{tr}(\Lambda\Sigma)+\mu^\top\Lambda\mu-2\eta^\top\mu-c\log\det\Sigma]+C,
$$

where C is receiver-independent. Therefore

$$
\nabla_\mu F=\Lambda\mu-\eta,\qquad
\partial_\Sigma F=\frac12(\Lambda-c\Sigma^{-1}),
$$

and

$$
\mu_*=\Lambda^{-1}\eta,\qquad \Sigma_*=c\Lambda^{-1}.
$$

Global uniqueness follows without relying on stationary-point reasoning. Omitting zero-weight factors, define \(g=p_i^{a/c}\prod_{hj}p_{hj}^{w_{hj}/c}\) and \(Z=\int g(x) dx\). Completing the square gives precision \(\Lambda/c\succ0\), hence \(0<Z<\infty\) and \(r=g/Z=\mathcal N(\mu_*,\Sigma_*)\). Direct substitution gives

$$
F(q)=cD_{\rm KL}(q\Vert r)-c\log Z.
$$

For Gaussians the excess is the sum of \((\mu-\mu_*)^\top\Lambda(\mu-\mu_*)/2\) and \(c\sum_k(\lambda_k-1-\log\lambda_k)/2\), where the positive \(\lambda_k\) are eigenvalues of \((\Lambda/c)^{1/2}\Sigma(\Lambda/c)^{1/2}\). The scalar inequality \(x-1-\log x\ge0\), with equality only at x=1, proves the unique minimum. No joint-law or observation interpretation is required for this local identity.

## Fisher flow and exact finite step

The Gaussian Fisher metric is

$$
g((u,A),(v,B))=u^\top\Sigma^{-1}v+
\frac12\operatorname{tr}(\Sigma^{-1}A\Sigma^{-1}B).
$$

Its inverse gives the manuscript's tangents \(-\Sigma\nabla_\mu F\) and \(-2\Sigma(\partial_\Sigma F)\Sigma\). Thus

$$
\dot\mu=-\Sigma(\Lambda\mu-\eta),\qquad
\dot\Sigma=c\Sigma-\Sigma\Lambda\Sigma.
$$

Let \(J=\Sigma^{-1},h=J\mu\). Keeping matrix order gives

$$
\dot J=-J\dot\Sigma J=\Lambda-cJ,\qquad
\dot h=\dot J\mu+J\dot\mu=(\Lambda-cJ)\mu-(\Lambda\mu-\eta)=\eta-ch.
$$

For every \(\Sigma_0\succ0,t\ge0\), let \(\rho=1-e^{-ct}\). The exact solution is

$$
J_t=(1-\rho)J_0+\rho\Lambda/c,\quad
h_t=(1-\rho)h_0+\rho\eta/c,\quad
\Sigma_t=J_t^{-1},\quad\mu_t=J_t^{-1}h_t.
$$

Positive definiteness follows from convexity of the SPD cone. This proves existence at every finite nonnegative time and convergence to the stated minimizer. The endpoint \(\rho=1\) is the infinite-time limit. Direct mean/covariance interpolation is a different finite update in general.

Writing \(g_\mu=\Lambda\mu-\eta\), \(A=\Lambda-cJ\), the exact dissipation is

$$
\frac{dF}{dt}
=-g_\mu^\top\Sigma g_\mu
-\frac12\operatorname{tr}(A\Sigma A\Sigma)
=-g_\mu^\top\Sigma g_\mu-\frac12\|\Sigma^{1/2}A\Sigma^{1/2}\|_F^2\le0.
$$

Hence every exact finite update decreases or preserves the same frozen F. Equality over a nonzero time interval requires the optimum. Euler/retraction steps, independent mean/covariance rates, floors, clipping and live weight/frame changes do not inherit this guarantee automatically.

If covariance is intentionally frozen at \(\Sigma_0\), the mean optimum is unchanged but its flow is

$$
\mu_t=\mu_*+\exp(-\Sigma_0\Lambda t)(\mu_0-\mu_*).
$$

Similarity to \(\Sigma_0^{1/2}\Lambda\Sigma_0^{1/2}\succ0\) proves convergence. This is generally a matrix-exponential update, not simultaneous J/h damping. Its dissipation is \(-g_\mu^\top\Sigma_0g_\mu\).

## Diagonal receiver restriction

In the fixed chart, \(\Sigma=D=\operatorname{diag}(v)\) gives

$$
F(\mu,v)=\frac12[\mu^\top\Lambda\mu-2\eta^\top\mu+
\sum_k(\Lambda_{kk}v_k-c\log v_k)]+C.
$$

Thus the exact restricted optimum is

$$
\mu_*=\Lambda^{-1}\eta,\qquad v_{*,k}=c/\Lambda_{kk}.
$$

The mean solve remains dense. The covariance differs from retaining \(\operatorname{diag}(c\Lambda^{-1})\), the marginals of the dense optimum.

The restricted Fisher flow is

$$
\dot\mu=-D(\Lambda\mu-\eta),\qquad
\dot v_k=cv_k-\Lambda_{kk}v_k^2.
$$

For \(p_k=1/v_k,b_k=p_k\mu_k\),

$$
\dot p_k=\Lambda_{kk}-cp_k,\qquad
\dot b=\eta-cb-(\Lambda-\operatorname{diag}(\Lambda_{kk}))\mu.
$$

The dense h equation therefore does not survive diagonal restriction. At a diagonal initial state the mean and diagonal covariance tangents agree with the projected dense tangents. Finite dense-flow-then-project and restricted-flow updates nevertheless differ.

An exact two-dimensional example checks this algorithmic distinction, without choosing a large-network mechanism. Take one head/source, all frames and T/O equal I, \(a=0,w=c=1,\mu_j=0\), and

$$
\Lambda=\begin{pmatrix}2&1\\1&2\end{pmatrix},\quad
R=\Lambda^{-1}=\frac13\begin{pmatrix}2&-1\\-1&2\end{pmatrix},\quad
\Sigma_j=Q=R/2.
$$

This satisfies the stated link/noise model. From \(\mu_0=0,\Sigma_0=I\), write \(z=e^{-t}\). Dense flow has

$$
J_t=\begin{pmatrix}2-z&1-z\\1-z&2-z\end{pmatrix},\quad
\operatorname{diag}(\Sigma_t)=\frac{2-z}{3-2z}(1,1).
$$

Diagonal restricted flow has \(v_t=(2-z)^{-1}(1,1)\). Their per-coordinate difference is \((1-z)^2/[(3-2z)(2-z)]>0\) for t>0. Limits are 2/3 versus 1/2. This block embeds in every K>=2. It proves algorithmic inequality, not a performance effect.

## Router boundary

On a nonempty finite source support, let \(\pi_j>0,\tau>0\). For finite costs, the joint block

$$
F_{\rm joint}(q,\beta)=aD_{\rm KL}(q\Vert p_i)+
\sum_j\beta_j[\gamma D_{{\rm cmp},j}(q)+\lambda D_{{\rm content},j}(q)]
+\tau\sum_j\beta_j\log(\beta_j/\pi_j)
$$

has the unique fixed-q optimizer

$$
\beta_j^*=\frac{\pi_j\exp[-(\gamma D_{{\rm cmp},j}+\lambda D_{{\rm content},j})/\tau]}
{\sum_k\pi_k\exp[-(\gamma D_{{\rm cmp},k}+\lambda D_{{\rm content},k})/\tau]}.
$$

This follows from a normalization multiplier; the interior Hessian is \(\operatorname{diag}(\tau/\beta_j)\), positive on simplex tangents. The QK-only router with the same gamma, tau and pi agrees exactly when \(\lambda D_{{\rm content},j}\) is constant on the support, including lambda=0.

An admissible counterexample has K=2, one head, all frames I, receiver N(0,I), source covariances I, and source means \(0,(1,-1)^\top\). Choose

$$
S_Q=S_K=\begin{pmatrix}1&1\end{pmatrix},\quad
T=\begin{pmatrix}1&-1\end{pmatrix},\quad
O=\frac12\begin{pmatrix}1\\-1\end{pmatrix},\quad Q=I.
$$

All maps use both input coordinates, O writes both outputs, and comparison maps have full row rank. Both comparison laws equal N(0,2), so both comparison costs vanish. The content link is \(L=\tfrac12\begin{pmatrix}1&-1\\-1&1\end{pmatrix}\), with receiving covariance \(R=I+L\). It preserves the second source mean, and its content cost exceeds the first by

$$
\frac12\mu_2^\top R^{-1}\mu_2=\frac12.
$$

For uniform pi and lambda=tau=1, QK routing is uniform whereas combined routing is \((e^{1/2},1)/(1+e^{1/2})\). This smallest-dimensional example isolates the variational mismatch; it embeds in larger spaces and makes no PPL claim.

Adjusting beta is not sufficient to reuse content fusion as the joint q-coordinate solution. If gamma D_cmp(q) is live, freezing beta does not remove its receiver derivative. Projected comparison KL contributes entropies such as \(-\log\det(A_Q\Sigma A_Q^\top)\). The content theorem therefore applies only to its declared content energy, or when comparison terms are explicitly frozen with respect to q. A QK router feeding fixed weights into content optimization is a coherent conditional construction, without a shared-coordinate-minimization claim.

## Covariance and final scope

Under time-independent passive changes \(x_i'=g_ix_i,x_j'=g_jx_j\), transforming all data gives \(L'=g_iLg_j^{-1},Q_i'=g_iQ_ig_i^\top,R'=g_iRg_i^\top,m'=g_im\), and

$$
\Lambda'=g_i^{-\top}\Lambda g_i^{-1},\quad
\eta'=g_i^{-\top}\eta,\quad
\mu_*'=g_i\mu_*,\quad\Sigma_*'=g_i\Sigma_*g_i^\top.
$$

The full Fisher flow transforms accordingly. A fixed-chart diagonal family is not invariant under arbitrary GL(K) changes. No invertibility of L is required; adding a separately defined Theta to a new fixed link preserves the noisy receiving-law algebra but introduces additional geometric/model data.

The strongest supported result is the exact full-width frozen-link content minimizer and globally defined energy-decreasing Fisher flow for arbitrary finite dimensions/head/source counts under the hypotheses. The minimal open obligation for a shared receiver/router energy is whether comparison costs vary during receiver optimization and whether routing minimizes that same declared energy. Specifying that choice materially changes the receiver equations and implementation. Population VFE, smoothing, outer-loss monotonicity and performance remain outside this result.

Execution record: read-only PowerShell source reads and analytic calculations. No numerical or symbolic computation was executed, and no conclusion relies on the parent's numerical examples. No GPU workload, code change, vault change, environment change or Lean work was performed.


## Comparison after independent reconstruction

After recording this result, the parent report and variational contribution were read. The frozen-content minimizer, Fisher dynamics, diagonal restricted optimum, and categorical routing boundary match the independent reconstruction. No load-bearing mismatch was found in that bounded core. This is a derivation check, not closure by agreement. One coordinate clarification was returned to the parent: the comparison covariance derivative written with S_Q and reference C_i is with respect to C_i; with respect to local Sigma_i it uses A_Q = S_Q U_i^{-1} on both sides. Other geometric, runtime, complexity and empirical claims in those reports are outside this assigned mathematical check.
