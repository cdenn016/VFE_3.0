# Renormalization and effective-theory audit

## Exact coarse theory before truncation

Start from a fine law $P(dz)$, or $P(dz)=Z^{-1}e^{-S(z)}\mu(dz)$, and a declared coarse-graining Markov kernel $C(dZ\mid z)$. Define the exact coarse law before projecting to an ansatz:

$$
P'=C_\#P,
\qquad
P'(dZ)=\int C(dZ\mid z)P(dz).
$$

Relative to a declared coarse reference measure $\mu'$, define $S'(Z)=-\log(dP'/d\mu')(Z)$ up to the stated normalization only after proving $P'\ll\mu'$. If absolute continuity fails, retain the exact measure $P'$ or declare a Lebesgue-decomposition and extended-valued convention instead of writing an undefined density. For a declared fine functional $\mathcal F$, define the contracted functional by

$$
\mathcal F'_C[Q]:=\inf_{q:C_\#q=Q}\mathcal F[q].
$$

This is a definition, not an automatic identification with a coarse KL divergence. If

$$
\mathcal F[q]=\operatorname{KL}(q\|P)+c,
$$

where $c$ is a stated Q-independent constant, then one may prove

$$
\mathcal F'_C[Q]=\operatorname{KL}(Q\|C_\#P)+c
$$

only after establishing the measurable-space hypotheses, absolute continuity, extended-value convention, existence of a regular disintegration of $P$ over the deterministic coarse map, and an admissible lift attaining or approximating the infimum. For a stochastic kernel, formulate and prove the corresponding lifted joint problem rather than copying the deterministic contraction identity.

Define the joint law $J(dz,dZ):=P(dz)C(dZ\mid z)$. For $f\in L^2(C_\#P)$, define observable lifting by $(Uf)(z)=\int f(Z)C(dZ\mid z)$. For $a\in L^2(P)$, define restriction by conditional expectation $(Ra)(Z)=\mathbb E_J[a(z)\mid Z]$ when a regular conditional law exists. Then the typed adjoint pairing is

$$
\langle a,Uf\rangle_{L^2(P)}=\langle Ra,f\rangle_{L^2(C_\#P)}.
$$

Define block-boundary effective interactions by integrating the block microstates at fixed coarse and boundary data, with reference densities and normalizers made explicit. Obtain cross-scale conditional kernels by disintegrating the exact coarse joint law; pairwise conditional kernels do not prove factorization or uniqueness.

Track the full generated operator space, including higher-body and nonlocal interactions, memory, constraints, entropy and normalization, Jacobian or log-determinant terms, boundary operators, and every symmetry-allowed term. Declare an ambient normed vector space $\mathcal A$ of normalized operator representatives, or a normed vector quotient by additive constants and other redundant directions. Declare an RG map $\mathcal R:\mathcal A\to\mathcal A$, a finite ansatz space $\mathcal T$, an embedding $\iota:\mathcal T\hookrightarrow\mathcal A$, and a retraction or projection $\Pi:\mathcal A\to\mathcal T$ satisfying $\Pi\iota=\operatorname{Id}_{\mathcal T}$. For $S\in\mathcal T$, the projected update is $\Pi\mathcal R(\iota S)$ and its ambient truncation residual is the typed element

$$
\epsilon_S:=\mathcal R(\iota S)-\iota\Pi\mathcal R(\iota S)\in\mathcal A.
$$

The projection is exact on $\mathcal T$ only if a closure theorem proves $\mathcal R(\iota\mathcal T)\subseteq\iota\mathcal T$, equivalently $\epsilon_S=0$ for every $S\in\mathcal T$ under the stated retraction assumptions. Otherwise record an error norm or bound and a proved stability estimate before propagating the residual.

For attention-like couplings, declare whether each coefficient is a Markov weight, energy coupling, or inverse temperature. Matrix aggregation requires an identifiable exact decomposition and a proved lumpability or intertwining condition. Otherwise it is a projection scheme, not exact coarse attention.

## Cross-scale evolution and beta data

Cross-scale coherence has distinct typed cases. Let $L$ be an ordered scale set, let $\mathcal X_n$ be the state space at level $n$, and for $\ell\preceq n$ let $C_{\ell\leftarrow n}:\mathcal X_n\rightsquigarrow\mathcal X_\ell$ be a measurable map or Markov kernel of one fixed declared kind. Without such maps, the levels are unrelated. The identity and ordered composition laws

$$
C_{n\leftarrow n}=\operatorname{Id}_{\mathcal X_n},
\qquad
C_{\ell\leftarrow n}=C_{\ell\leftarrow m}\circ C_{m\leftarrow n}
\quad(\ell\preceq m\preceq n)
$$

define a contravariant functor from the thin category of $L$, equivalently a covariant functor from $L^{\mathrm{op}}$, to the declared category of spaces and maps or kernels. When $L$ carries a suitable scale parameter, this may be called a two-parameter evolution family; add the appropriate continuity or measurability in the scale variables when limits, generators, or beta functions are required. A cocycle additionally requires a declared base flow $\theta$ and a typed law such as $\Phi(t+s,\omega)=\Phi(t,\theta_s\omega)\circ\Phi(s,\omega)$. Only after identifying all level spaces with a common space, taking the scale indices in an additive ordered abelian group, and imposing scale-translation invariance, $C_{s\leftarrow t}=T_{t-s}$, does the evolution reduce to a one-parameter semigroup $T_{a+b}=T_a\circ T_b$ with $T_0=\operatorname{Id}$ on the declared nonnegative scale differences; regularity is still needed for an infinitesimal generator. A bundle-valued RG step requires a bundle morphism covering a base map, compatible sections, and compatible connections; otherwise record the vertical or horizontal mismatch term.

Before comparing scale-dependent operators, declare reference embeddings or isomorphisms

$$
i_\ell:\mathcal A_\ell\to\mathcal A_*.
$$

Require each $i_\ell$ to be injective on the compared operator classes and declare the common linear or affine structure used for subtraction. If an $i_\ell$ is lossy, expose its kernel or fibers and propagate a reconstruction or identification error instead of calling it an identification. Only after this transport to $\mathcal A_*$ may a finite-difference beta compare levels. Declare the scale coordinate $t$ before writing $\beta^a=dg^a/dt$. Under a scheme change or coupling-coordinate change $g'=f(g)$, beta components transform as $\beta'^a=(\partial f^a/\partial g^b)\beta^b$. An autonomous beta function needs scale homogeneity and semigroup assumptions; an exact flow may instead be functional and infinite dimensional.

An autonomous fixed point satisfies $\mathcal R(S_*)\sim S_*$ modulo declared rescaling, additive constants, gauge, and reparameterization. If a family $\mathcal R_b$ of finite-scale maps is used instead, declare its scale parameter and state whether the fixed-point condition is required for one generating scale or every $b$. Quotient redundant directions before classifying the stability spectrum. For a cocycle or nonautonomous system, the relevant fixed objects may be invariant sections, invariant measures, pullback or random attractors, or other equivariant families, not ordinary zeros of a vector field. State which object replaces a fixed point and prove its invariance law.

The final audit exposes the coarse kernel, reference measures, exact effective functional, restriction and lifting, generated operators, closure theorem or truncation residual, attention identifiability and lumpability, coherence law, reference identifications, scale and scheme, fixed points modulo symmetry or the appropriate nonautonomous invariant object, stability data, and every approximation error.
