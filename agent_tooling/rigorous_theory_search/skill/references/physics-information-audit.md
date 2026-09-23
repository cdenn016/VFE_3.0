# Physics, gauge, information, and ELBO audit

## Gauge and associated statistical bundles

Declare a principal bundle $\pi:P\to B$, a right $G$ action, representations $\rho_q$ and $\rho_p$, and associated statistical bundles $E_q=P\times_G\mathcal S_q$ and $E_p=P\times_G\mathcal S_p$. State the gauge convention. Under the common convention,

$$
A^g=g^{-1}Ag+g^{-1}dg,
\qquad
F^g=g^{-1}Fg.
$$

Gauge fixing is not invariance. Show that an observable descends to the quotient or that an equivariant object yields a gauge-independent scalar or tensor. Check transition cocycles, topology, singular strata, boundaries, and anomalies.

For a regular dominated smooth statistical family, require the directional score derivatives to exist and be square-integrable, the Fisher tensor below to be finite and smooth on the declared parameter domain, and every differentiation-under-the-integral step to satisfy an explicit domination or other valid interchange theorem. Then use

$$
I_\theta(u,v)=\mathbb E_\theta[(u\log p_\theta)(v\log p_\theta)].
$$

In a trivial product bundle, $h_x^s(X,Y)=I_{s(x)}(ds_xX,ds_xY)$ is positive semidefinite and has rank at most $\operatorname{rank}ds_x$. In a nontrivial associated bundle, the Fisher tensor is fiberwise and acts on vertical tangents. Choose a connection and define the covariant vertical first jet by

$$
D^\omega s=\operatorname{ver}^\omega\circ Ts,
$$

then set

$$
h_x^\omega(X,Y)=I_{s(x)}(D^\omega sX,D^\omega sY).
$$

Prove that the group action preserves Fisher information and that $D^\omega s$ transforms equivariantly before calling the pullback gauge-invariant. The tensor remains connection-relative. If the fiberwise Fisher tensor is positive definite, then

$$
\operatorname{rad}h^\omega=\ker D^\omega s,
\qquad
\operatorname{rank}h^\omega=\operatorname{rank}D^\omega s.
$$

Without injectivity it is a semimetric. A vector-bundle quotient needs constant rank. A quotient manifold metric additionally needs an involutive radical distribution, a regular leaf space, and a basic tensor along the leaves.

## Curves, duration, and clocks

Write $\pi_E:E\to B$ for the associated-bundle projection and $E_x=\pi_E^{-1}(x)$ for its fiber. Before interpreting a curve, distinguish these cases. A vertical curve in one fiber is typed

$$
\gamma:J\to E_x,
\qquad
\pi_E\circ\gamma=x,
\qquad
\dot\gamma\in VE.
$$

If $c:J\to B$, the section-induced curve

$$
\eta=s\circ c:J\to E,
\qquad
\pi_E\circ\eta=c,
\qquad
T\pi_E(\dot\eta)=\dot c
$$

is not vertical where $\dot c\ne0$. After choosing a connection $\omega$, its covariant belief velocity is

$$
\xi^\omega
=\operatorname{ver}^\omega(\dot\eta)
=D^\omega s(\dot c).
$$

A horizontal lift instead satisfies $\operatorname{ver}^\omega(\dot\eta^{\mathrm{hor}})=0$. Thus horizontal transport and comparison between different fibers are connection-dependent and are not themselves vertical belief change. A one-parameter family of sections $s_t$ gives a vertical curve only after fixing $x$, namely $t\mapsto s_t(x)\in E_x$. Endpoints do not specify a path, orientation, or regularity.

For a duration claim, use an absolutely continuous curve contained in a smooth statistical stratum where the Fisher tensor is finite and nondegenerate. A fixed-fiber vertical curve has connection-independent Fisher speed

$$
v_F^{\mathrm{vert}}(\lambda)
=\sqrt{I_{\gamma(\lambda)}(\dot\gamma(\lambda),\dot\gamma(\lambda))},
$$

whereas a section-induced curve has the connection-relative speed

$$
v_F^\omega(\lambda)
=\sqrt{I_{\eta(\lambda)}(\xi^\omega(\lambda),\xi^\omega(\lambda))}.
$$

For either case, write $\star=\mathrm{vert}$ or $\omega$ and set

$$
\tau_F^\star(\lambda)
=\tau_0+\int_{\lambda_0}^{\lambda}v_F^\star(u)\,du.
$$

This cumulative length is nondecreasing and invariant under regular orientation-preserving reparameterization. It is strictly increasing exactly when there is positive accumulated length on every nontrivial subinterval. If the curve is $C^1$ and $v_F^\star>0$ everywhere, then $\tau_F^\star$ is a regular arc-length coordinate; an isolated zero can preserve strict monotonicity while destroying regular invertibility there. A zero-speed interval destroys strict monotonicity, and a Fisher-null tangent destroys regular invertibility at that point. A singular-stratum crossing falls outside this smooth-stratum sufficient theorem but need not destroy a duration coordinate: either work on separate regular intervals or give a separate pathwise proof that the metric speed extends finitely and positively and that the accumulated length remains regular across the crossing. The vertical duration is intrinsic to the chosen oriented fixed-fiber curve up to its origin. The section-induced duration $\tau_F^\omega$ is connection-relative as well as curve-, orientation-, and origin-dependent. Neither is a unique physical time coordinate or a global function on $B$. Before asserting a regional clock potential, define a closed clock one-form $\alpha$, prove zero periods on the region, and only then conclude

$$
\alpha=d\tau.
$$

Also state the operational bridge connecting $\tau$ to physical clock readings, orientation, synchronization, completeness or causality where relevant, and the treatment of zero-speed and closed histories. Do not smuggle the path parameter in as external time.

## Density gluing and exact ELBO bookkeeping

For local log-density corrections on overlaps, define $c_{\alpha\beta}$ with orientation and require the triple-overlap cocycle

$$
c_{\alpha\beta}+c_{\beta\gamma}+c_{\gamma\alpha}=0.
$$

Global gluing requires the cocycle to be a coboundary,

$$
c_{\alpha\beta}=f_\beta-f_\alpha,
$$

on the chosen cover, followed by normalization and uniqueness under the declared equivalence. If $(f_\alpha)$ is one solution, then $(f_\alpha+h|_{U_\alpha})$ is another for every global function $h$ of the admitted regularity. State what additional data or equivalence removes this global-function freedom. Only when those conditions reduce it to a global additive constant can scalar normalization remove the remaining ambiguity.

Declare ELBO/VFE signs, base measures, and Radon-Nikodym derivatives. Under absolute continuity and finite terms,

$$
\log p(y)=\mathcal L(q;y)+\operatorname{KL}(q\|p(\cdot\mid y)),
$$

$$
\mathcal L(q;y)=\mathbb E_q[\log p(z,y)-\log q(z)],
\qquad
\mathcal F(q;y)=-\mathcal L(q;y).
$$

Require regular conditional laws, support conditions, normalization, KL and entropy finiteness, and justified integration. KL disintegration separates marginal and conditional contributions. For compatible joint laws $Q,P$ with marginals $Q_i,P_i$, the exact correction identity is

```text
KL(Q\|P)=\sum_i KL(Q_i\|P_i)+TC(Q)+E_Q log(\prod_i p_i/p).
```

$$
KL(Q\|P)=\sum_i KL(Q_i\|P_i)+TC(Q)+\mathbb E_Q\log\left(\frac{\prod_i p_i}{p}\right),
$$

where $TC(Q)=KL(Q\|\prod_iQ_i)$ and all densities and finite terms must exist. The global objective is not generally a naive sum of local objectives: account for shared factors, double counting, boundaries, mutual information, total correlation, shared latents, and interaction corrections.

For a factor graph and mean-field family, an agent's coordinate-local objective includes its entropy and every incident factor. With the other coordinates fixed, prove that its variation matches the global variation up to an additive constant; do not infer equality of the full local and global functionals.

### Stochastic-channel replacement and ontology

For every proposed replacement of observations, latents, or environment variables by stochastic channels, state a conditional replacement theorem even when a counterexample refutes the stronger claim. Let $R$ contain every retained variable, $E$ every removed variable, and $Y=(Y_1,\ldots,Y_m)$. Work on standard Borel spaces or explicitly supply every required disintegration and regular conditional law. Marginal identities such as $P(dy_i\mid x_j)=K_{ij}(dy_i\mid x_j)$ are insufficient. Declare typed, measurable, normalized kernels and prove that they assemble into a joint kernel $K$ preserving the full joint law on retained variables:

$$
P_{R,Y}:=(\pi_{R,Y})_\#P=P_R(dr)K(dy\mid r).
$$

If a product channel is claimed, prove both

$$
K(dy\mid r)=\prod_{i=1}^mK_{i\,j(i)}(dy_i\mid x_{j(i)})
$$

and the corresponding conditional independences, for an ordering compatible with the claimed factorization,

$$
P(dy_i\mid r,y_{<i})
=K_{i\,j(i)}(dy_i\mid x_{j(i)})
\quad P_{R,Y_{<i}}\text{-a.s.}
$$

Otherwise retain a joint channel and its shared-noise or interaction factors. Declare sigma-finite base measures $\mu_R$, $\mu_E$, and $\nu_Y$, together with jointly measurable densities $p(r,e,y)$ and

$$
p_{R,Y}(r,y)=\int p(r,e,y)\,\mu_E(de).
$$

Require equality almost everywhere of all likelihood factors, shared factors, and normalization terms. That law-level equality proves replacement equivalence only for almost every observation; it does not identify likelihood values on a chosen null slice. If a claim includes every specified observation or a fixed-$y$ ELBO, every compared model and replacement must declare chosen pointwise density representatives $p$ and $\widetilde p$ and prove slice-wise equality at that $y$:

$$
\widetilde p(r,e,y)=p(r,e,y)
\quad q\text{-a.e.}
$$

for each admitted $q$ used in the full ELBO, and at minimum

$$
\widetilde p_{R,Y}(r,y)=p_{R,Y}(r,y)
\quad q_R\text{-a.e.}
$$

for equality of collapsed ELBOs. Prove equality of the slice zero sets and support conditions as well. A claim uniform over a variational family must state a common dominating measure and prove these equalities on the union of its admitted supports. Otherwise restrict the equivalence theorem to almost every observation and call any selected-null-slice likelihood or ELBO version-relative.

State the required absolute continuity of every admissible variational law and finiteness of all logarithms. Account for the entire ELBO rather than one likelihood term. For a specified fixed observation $y$, do not use an arbitrary regular-conditional version on a null slice. On the $q_R$-supported set where $0<p_{R,Y}(r,y)<\infty$, define the pointwise density ratio

$$
p_y(e\mid r):=\frac{p(r,e,y)}{p_{R,Y}(r,y)},
\qquad
P_y(de\mid r):=p_y(e\mid r)\,\mu_E(de),
$$

and require $\int p_y(e\mid r)\,\mu_E(de)=1$ there. This fixes the representative needed by the slice-wise algebra. Without that declaration, an abstract regular conditional supports the identity only for almost every observation under the corresponding marginal law, not at an arbitrarily selected null value. Now let $q(dr,de)=q_R(dr)q_E(de\mid r)$, require $q_E(\cdot\mid r)\ll P_y(\cdot\mid r)$ for $q_R$-almost every $r$, and require every displayed expectation and logarithm to be finite. Then

$$
\mathcal L_P(q;y)
=\mathcal L_{P_{R,Y}}(q_R;y)
-\mathbb E_{q_R}\operatorname{KL}\!\left(
q_E(\cdot\mid R)\,\|\,P_y(\cdot\mid R)
\right).
$$

The correction is nonnegative and vanishes exactly under the conditional-posterior equality

$$
q_E(\cdot\mid r)=P_y(\cdot\mid r)
\quad q_R\text{-a.e.}
$$

Thus marginalization preserves the collapsed ELBO exactly only under that condition; otherwise retain the correction. Equality of induced laws or ELBOs is representational equivalence on the declared observables, not ontological identity. Exogenous noise, latent environment state, interventions, and nonunique factorizations remain unless separately excluded. Label an agent-only ontology as a distinct **modeling or operational bridge postulate**. If $\mathscr A_{\mathrm{adm}}$ is a previously declared ambient observable sigma-algebra, an observational completeness postulate may require equality of the P-completed sigma-algebras,

$$
\overline{\mathscr A_{\mathrm{adm}}}^{\,P}
=\overline{\sigma(X_1,\ldots,X_n)}^{\,P}.
$$

State interventional completeness separately: every admitted intervention and its induced law or transition kernel must be represented by typed operations on agent variables. Require every admitted noise source and latent state to be encoded as well. Neither observational completion, channel factorization, nor ELBO equality proves these postulates.
