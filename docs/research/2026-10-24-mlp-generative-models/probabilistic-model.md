# The prototype block as an agent generative model

The prototype bank admits an exact normalized generative-model interpretation for one declared internal record. The implemented block computes either posterior moments or finitely many mean-field updates. It does not propagate the exact joint posterior, and its ELBO diagnostic need not equal the evidence even when inference is named `exact`. These distinctions correct the scope of [the supplied note](<C:/Users/chris and christine/Desktop/mlp_as_generative_models.md>) without rejecting its identification of templates with model presentations.

The target is to establish, for every admissible fixed input and bank, which probabilistic law produces the implemented update and what equivalence survives its compression. The note supplies the proposed interpretation. The following assumptions make it precise: finite dimensions $K,M\geq1$, positive weights $\sum_a\pi_a=1$, finite means, and $C,\Lambda_a\succ0$. The implementation restricts $\Lambda_a$ to diagonal matrices; the derivation permits full matrices. Bank parameters and $C$ are held fixed as conditioning information during inference. Singular limits require a separate choice of supporting measures and are outside this statement.

[MAgent foundations, lines 165–197](<C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/chapters/01_foundations.tex:165>) defines a model by a reference law on presentations, an evaluator over contents, and a normalized record kernel. Let $A\in\{1,\ldots,M\}$, $X,O\in\mathbb R^K$, with counting measure for $A$ and Lebesgue measures for $X,O$. Write $\varphi_S(y)=\mathcal N(y;0,S)$. A completion satisfying that definition is

\[
p(a,x,o\mid C,\theta)=\pi_a\varphi_{\Lambda_a}(x-\nu_a)\varphi_C(o-x),
\qquad \theta=(\pi,\nu,\Lambda).
\]

Every conditional integrates to one, so the joint is normalized. Here the presentation reference is $\pi$, the evaluator is $\mathcal N(\nu_a,\Lambda_a)$, and the label-independent record kernel is $O\mid X\sim\mathcal N(X,C)$. Observe $O=\xi$. This is also the model explicitly stated by the [template EM objective](<C:/Users/chris and christine/Desktop/VFE_3.0/vfe3/model/block_mlp.py:1785>).

The original [minimal-agent model](<C:/Users/chris and christine/Desktop/Research/manuscripts/Attention/minimal_agent_model.tex:126>) instead starts from a receiver prior and a comparison record. After marginalizing the selected template state and specializing to identity maps, its joint is

\[
\widetilde p(a,x,o\mid\xi,C,\theta)
=\pi_a\varphi_C(x-\xi)\varphi_{\Lambda_a}(o-x+\nu_a),
\qquad O=0.
\]

Gaussian symmetry gives $p(a,x,\xi)=\widetilde p(a,x,0)$. Thus both declared models yield the same unnormalized posterior slice, posterior, and evidence density at these respective observations. Their complete generative joints differ: one makes the incoming mean an observation, while the other makes it a prior parameter. The product called a “joint” in [prototype_elbo](<C:/Users/chris and christine/Desktop/VFE_3.0/vfe3/model/prototype_agents.py:717>) is this observed slice; its integral over $a,x$ is the evidence, generally not one. Equality of slices does not identify the two record-generating mechanisms.

For completeness, expand the exponent of the slice as a quadratic in $x$. Its precision is $J_a=C^{-1}+\Lambda_a^{-1}$ and its linear coefficient is $b_a=C^{-1}\xi+\Lambda_a^{-1}\nu_a$. Substituting $x=m_a+y$, with $m_a=J_a^{-1}b_a$, leaves $y^\top J_ay$ plus the constant $(\xi-\nu_a)^\top(C+\Lambda_a)^{-1}(\xi-\nu_a)$. The determinant relation $\det P_a=\det C\det\Lambda_a/\det(C+\Lambda_a)$ matches the Gaussian prefactors. This proves the complete density identity, including its normalizer:

\[
\varphi_C(x-\xi)\varphi_{\Lambda_a}(x-\nu_a)
=\varphi_{C+\Lambda_a}(\xi-\nu_a)\varphi_{P_a}(x-m_a),
\]
\[
P_a=(C^{-1}+\Lambda_a^{-1})^{-1},\qquad
m_a=P_a(C^{-1}\xi+\Lambda_a^{-1}\nu_a)
=\xi+C(C+\Lambda_a)^{-1}(\nu_a-\xi).
\]

Consequently the finite, strictly positive evidence and exact posterior are

\[
Z=\sum_a\pi_a\varphi_{C+\Lambda_a}(\xi-\nu_a),\qquad
w_a=\frac{\pi_a\varphi_{C+\Lambda_a}(\xi-\nu_a)}Z,
\qquad p_*(a,x)=w_a\varphi_{P_a}(x-m_a).
\]

In MAgent notation, $s(a)=w_a$ and $q(dx\mid a)=\mathcal N(m_a,P_a)(dx)$. Bayesian model averaging gives

\[
\mu_* =\sum_aw_am_a,\qquad
V_* =\sum_aw_a\{P_a+(m_a-\mu_*)(m_a-\mu_*)^\top\}.
\]

The second term is uncertainty about the selected presentation. It can increase the marginal variance beyond $C$. These are the quantities calculated by the [exact helper](<C:/Users/chris and christine/Desktop/VFE_3.0/vfe3/model/prototype_agents.py:348>), subject to its covariance layout.

For a normalized recognition density $\zeta(a,x)$, define the ELBO before simplifying:

\[
\mathcal L(\zeta)=\sum_a\int\zeta(a,x)
\log\frac{\pi_a\varphi_{\Lambda_a}(x-\nu_a)\varphi_C(\xi-x)}{\zeta(a,x)}dx.
\]

Assume absolute continuity with respect to $p_*$ and finite KL; use the expectation formula when its terms are integrable. All Gaussian/product families below satisfy these requirements. Since the numerator is $Zp_*$,

\[
\mathcal F(\zeta)=-\mathcal L(\zeta)
=\mathrm{KL}(\zeta\Vert p_*)-\log Z.
\]

Therefore $\mathcal F=-\log Z$ holds exactly for the full correlated posterior, and only for that posterior up to null sets. Generally $A$ and $X$ remain dependent after conditioning. Positive-definite Gaussian factors make the posterior density positive everywhere, so any absolutely continuous recognition density has the required support. Finite second moments control the expected quadratic energies, while a finite entropy controls the remaining recognition term. These sufficient conditions hold for the finite Gaussian mixtures and nonsingular Gaussians used here; a point-mass state would require a different base-measure treatment.

The code's diagnostic uses $\zeta(a,x)=r_aQ(x)$, with $Q=\mathcal N(m,S)$, $S\succ0$. Its expectation is

\[
\mathcal L(Qr)=\mathbb E_Q\log\varphi_C(\xi-X)
+\sum_ar_a\mathbb E_Q\log\varphi_{\Lambda_a}(X-\nu_a)
+\sum_ar_a\log\pi_a+H(Q)+H(r).
\]

Equivalently,

\[
\mathcal L(Qr)=-\mathrm{KL}(Q\Vert\mathcal N(\xi,C))
+\sum_ar_a\mathbb E_Q\log\varphi_{\Lambda_a}(X-\nu_a)
-\mathrm{KL}(r\Vert\pi),
\]
\[
\mathbb E_Q\log\varphi_{\Lambda_a}(X-\nu_a)
=-\frac12\left[K\log(2\pi)+\log\det\Lambda_a
+\operatorname{tr}(\Lambda_a^{-1}S)
+(m-\nu_a)^\top\Lambda_a^{-1}(m-\nu_a)\right].
\]

These expressions reproduce `prototype_elbo`, including its normalization constants. They apply to any probability vector $r$, with $0\log0=0$.

The distinction survives exact mixture moments. If $p_X^*=\sum_aw_a\mathcal N(m_a,P_a)$, disintegration gives the exact residual

\[
\log Z-\mathcal L(Qr)
=\mathrm{KL}(Q\Vert p_X^*)
+\mathbb E_Q\mathrm{KL}(r\Vert p_*(A\mid X)).
\]

The first term concerns the state marginal; the second concerns discarded label-state dependence. Matching moments minimizes the opposite-direction divergence $\mathrm{KL}(p_X^*\Vert Q)$ over Gaussian $Q$: its parameter-dependent terms are

\[
\frac12\log\det S+\frac12\operatorname{tr}(S^{-1}V_*)
+\frac12(\mu_*-m)^\top S^{-1}(\mu_*-m).
\]

They are minimized at $m=\mu_*,S=V_*$, as follows by diagonalizing $S^{-1/2}V_*S^{-1/2}$ and using $t-\log t-1\geq0$. This does not minimize the reverse-KL ELBO gap. Under a diagonal or block covariance restriction, the same calculation retains only the corresponding diagonal or blocks of $V_*$. Even diagonal component covariances can produce off-diagonal mixture covariance through the shared categorical label.

The one-template case checks the boundary of this qualification. With $M=1$, the exact posterior is one Gaussian, its label is deterministic, and the product diagnostic reaches the evidence whenever its layout represents that Gaussian. Identical template laws provide another such case: the observation leaves the label law equal to $\pi$, independent of the common conditional state law. For distinguishable templates, neither exact labels nor exact first and second moments remove the dependence term. The large gap below is therefore a property of an admissible heterogeneous bank, not an inevitable positive gap for every bank.

The [block diagnostic](<C:/Users/chris and christine/Desktop/VFE_3.0/vfe3/model/block_mlp.py:1863>) passes $r=w$ and this compressed Gaussian to `prototype_elbo` in exact mode. It therefore measures the displayed product-law gap. The [forward method](<C:/Users/chris and christine/Desktop/VFE_3.0/vfe3/model/block_mlp.py:1756>) emits moments rather than a joint law; its passthrough option retains the input covariance, and dropout changes the posterior mean increment. The diagnostic recomputes the posterior before these output modifications. Its ELBO is consequently not, in general, an ELBO evaluated at the Gaussian encoded by the emitted moments.

Mean-field updates follow directly from constrained variation. At fixed $Q$, imposing $r_a\geq0$ and $\sum_ar_a=1$, stationarity gives

\[
\log\pi_a+\mathbb E_Q\log\varphi_{\Lambda_a}(X-\nu_a)
-\log r_a-1+\lambda=0,
\]
\[
r_a^+=\operatorname{softmax}_a
\left(\log\pi_a-\mathrm{KL}(Q\Vert\mathcal N(\nu_a,\Lambda_a))\right).
\]

Strictly positive finite inputs make this the unique interior label optimum. The omitted $H(Q)$ is constant only in this label block. The normalization axis is the bank index for each token, and the categorical entropy coefficient, hence temperature, is one. Another temperature requires a modified objective.

At fixed $r$, varying a normalized state density gives

\[
Q^+(x)\propto\varphi_C(x-\xi)
\prod_a\varphi_{\Lambda_a}(x-\nu_a)^{r_a},
\]
\[
P=(C^{-1}+\sum_ar_a\Lambda_a^{-1})^{-1},\qquad
m=P(C^{-1}\xi+\sum_ar_a\Lambda_a^{-1}\nu_a).
\]

This is the exact Gaussian coordinate optimum; $P\prec C$. Each coordinate update cannot decrease the fixed-model ELBO, because the difference from its optimum is a KL divergence. Neither statement implies a global joint optimum. The [implemented CAVI loop](<C:/Users/chris and christine/Desktop/VFE_3.0/vfe3/model/prototype_agents.py:584>) starts labels from $Q_0=\mathcal N(\xi,C)$, always uses the original $\xi,C$ in the state coordinate, and stops after a state update. Its returned labels were computed from the preceding Gaussian. Finite sweeps need not satisfy the label stationarity equation at the returned Gaussian; repeated sweeps optimize one record rather than assimilating repeated copies.

A scalar counterexample quantifies both issues. Take $\xi=0,C=1,\pi_\pm=1/2,\nu_\pm=\pm3,\Lambda_\pm=1/4$. The posterior components have means $\pm b$, $b=2.4$, and variance $v=0.2$. Exact weights are $1/2$, so $V_*=v+b^2=5.96$. For the diagnostic product $Q=\mathcal N(0,V_*),r=(1/2,1/2)$,

\[
\mathrm{KL}(Qr\Vert p_*)
=\frac12\sum_{a=\pm}\mathrm{KL}(Q\Vert\mathcal N(ab,v))
=\frac{b^2}{v}-\frac12\log(1+b^2/v)
=28.8-\tfrac12\log29.8\approx27.1027458032.
\]

Symmetric CAVI instead returns $Q=\mathcal N(0,v)$, $r=(1/2,1/2)$, on every positive sweep count, with gap $b^2/(2v)=14.4$. The feasible product concentrated on one label, with that label's conditional Gaussian, has gap $-\log(1/2)=\log2$; hence the symmetric fixed point is nonglobal. More strongly, fixing $S=v$ and optimizing labels gives

\[
\mathcal L(m)=\mathrm{constant}-\frac{m^2}{2v}
+\log\cosh(3m/(1/4)),\qquad \mathcal L''(0)=-5+144=139.
\]

Thus it is not even a local maximum along this jointly varied path. Zero label staleness alone does not certify a good variational optimum.

The existing [check-results.json](<C:/Users/chris and christine/Desktop/VFE_3.0/docs/research/2026-10-24-mlp-generative-models/check-results.json>) records these examples and the earlier execution of [check_cases.py](<C:/Users/chris and christine/Desktop/VFE_3.0/docs/research/2026-10-24-mlp-generative-models/check_cases.py>): CPU float64, one thread, $K\leq2,M\leq3$, no training or GPU execution. Independent exact mean, covariance, weight, and evidence discrepancies were at most $2.23\times10^{-16}$. Those finite checks support their instances; the derivations establish the general identities. No computation was rerun while finishing this memo.

The internal-record semantics require a separate decision. A normalized kernel for an invented record does not establish that a working belief is fresh evidence about external contents. If the incoming law already obeys $q_{\rm in}(x)=p_0(x)L_D(x)/Z_D$, replacing its prior by the bank $h(x)=\sum_a\pi_a\varphi_{\Lambda_a}(x-\nu_a)$ requires

\[
p_h(x\mid D)\propto h(x)L_D(x)
\propto h(x)\frac{q_{\rm in}(x)}{p_0(x)},
\]

on the relevant support, with a finite positive new normalizer. Multiplying $h q_{\rm in}$ retains the old prior factor. That multiplication is valid for the newly declared comparison-record model; it is not automatically prior replacement under the original data likelihood. If $O=f(D)$ deterministically, then conditioning on $D,O$ equals conditioning on $D$; reprocessing the summary supplies no additional independent observation.

Holding $C$ as declared side information makes the displayed conditional Gaussian kernel normalized. Substituting an observation-dependent covariance into an unconditional density needs another normalization check. In one dimension, $C(o)=1$ for $o\leq0$ and $C(o)=4$ otherwise gives

\[
\int\varphi_{C(o)}(o-x)do=\Phi(-x)+\Phi(x/2),
\]

which is below one for $x>0$. Restoring normalization introduces an $x$-dependent factor and changes inference. This example identifies the distinction between conditioning on an externally specified covariance and silently making it a function of the variable being generated.

Likewise, preserving a model belief across layers requires a declared alignment or stochastic transition between their label spaces. With transition $T_\ell(b\mid a)$, the predicted label law is $s_\ell^-(b)=\sum_aT_\ell(b\mid a)s_{\ell-1}(a)$; the next update requires a likelihood conditional on the previous records. A correlated model must also propagate or approximate $q(X\mid A)$. Independent per-layer banks and reused deterministic summaries do not supply these objects. Summing their local free energies does not establish one latent-variable ELBO. A separately normalized causal token model can nevertheless define a valid autoregressive product over tokens.

A further check concerns what the replacement can express. The [prior investigation](<C:/Users/chris and christine/Desktop/VFE_3.0/docs/research/2026-10-20-neural-free-continuation/prototypes.md:47>) derives a restriction for a fixed bank and covariance. Write the marginal posterior as an exponential family with natural parameter $\eta=C^{-1}\xi$ and base factor $h(x)\exp(-x^\top C^{-1}x/2)$. Differentiating its finite log normalizer once gives its mean and twice gives its covariance. These derivatives exist because a finite Gaussian mixture has all the needed locally dominated moments. The chain rule yields $D_\xi\mu_*=V_*C^{-1}$. In coordinates whitened by $C$, the Jacobian is the symmetric positive-definite matrix $C^{-1/2}V_*C^{-1/2}$. In one dimension its derivative is strictly positive, whereas $f(x)=x-4\operatorname{GELU}(x)$ has $f'(0)=-1$. No increase in bank size makes this exact same-state posterior mean equal that residual map near zero. Input-dependent covariance or a different joint cue/result law changes the assumptions.

This restriction does not obstruct calling the bank a generative model: it locates the model's particular inductive bias. It also does not establish a language-model performance disadvantage. The local inference calculation supplies a normalized statistical mechanism, and the retained moments supply an approximation interface for the surrounding program. Neither supplies predictive credit from the next observed token by itself. In particular, fitting the likelihood of internal carrier summaries can differ from improving the normalized next-token conditional.

The strongest supported conclusion is a normalized local perception mechanism with explicitly characterized posterior approximations. The minimal remaining obligation for the stronger claim is an explicit normalized joint connecting external token records, content states, and persistent presentations across depth. Writing that joint and its actual recognition family is the next action that could establish a whole-stack ELBO; the local identities alone do not.
