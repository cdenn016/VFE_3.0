# The MLP slot as inference in an agent's generative model

October 24, 2026. Investigation of [mlp_as_generative_models.md](<C:/Users/chris and christine/Desktop/mlp_as_generative_models.md>) against the live `prototype_agents` implementation, the Research manuscripts, prior investigations, and primary literature.

The central perspective is mathematically sound for `prototype_agents`: a learned template bank specifies a distribution over possible contents, and the block uses an incoming record to infer both the applicable template and the content state. This supplies an explicit probabilistic mechanism for local nonlinear processing. Three distinctions determine how far the statement reaches. The bank and observation kernel define the generative model; the block performs inference in it. Exact posterior statistics do not mean the propagated state is the full posterior. Finally, neither a shared bank nor a collection of locally valid inference steps establishes the manuscript's complete species condition or one latent-variable ELBO for the network.

The most consequential corrections to the supplied note are its identification of the implemented exact-mode free energy with negative log evidence, and its claim that reference-coordinate evaluators alone establish a species. The ordinary GELU comparison also needs a narrower claim: an arbitrary GELU MLP is not established to approximate this particular Bayesian model. It has other exact probabilistic representations, and its learned input and output maps are less constrained than same-state prototype conditioning.

This investigation retained the earlier results on monotone prototype retrieval, joint cue/result memory, and predictive local learning. It did not implement the note's suggested next steps, alter the launcher, edit the film or Research vault, run training, or make a performance claim. Three parallel investigations supplied the probabilistic, geometric, and implementation analyses. After an application interruption, their captured findings and the completed CPU calculation were retained; the unfinished reports were completed without rerunning that calculation.

## 1. What is established, and what needs qualification

| Claim in the supplied perspective | Result of this investigation |
|---|---|
| Prototype labels can be model presentations. | Exact for a finite model family with a specified evaluator and normalized observation kernel. |
| Template weights are beliefs over models. | Exact for Bayesian component probabilities in exact mode; mean-field responsibilities are variational label coordinates. This is uncertainty over component identity, not over bank parameters. |
| Exact mode has free energy equal to negative log evidence. | True for the full mathematical joint posterior. False in general for the product law evaluated by the current diagnostic or the Gaussian propagated by the block. |
| The block returns the moment-matched posterior. | Requires the posterior covariance contract, no active dropout, and a representation qualification. Passthrough covariance does not return the posterior moments. |
| Mean field performs inference in the same local model. | Correct. Finite sweeps are coordinate updates, not proof of convergence; even a fixed point can be a poor stationary point. |
| One shared bank makes a layer one species on the reference path. | Insufficient. The manuscript also requires compatible record kernels on the declared record space. Token-specific noise matters. |
| Encoder, prototype, and decoder share a template pattern. | Useful and partly exact at the level of kernels and label updates. They are distinct banks and operations, with no established common internal joint. |
| A GELU MLP is an approximate generative model of the same kind. | A memory analogy is supported. Approximation to this specific posterior has not been defined or demonstrated. |
| Shared templates are hyperpriors. | They are point parameters of a shared model. A hyperprior over them would be an additional probability law. |
| The implementation is a non-neural MLP replacement. | It replaces the affine/GELU/affine computation with distribution-based inference. It remains a parameterized nonlinear layer trained through the current outer gradient computation. |

The detailed source trace is [implementation.md](<C:/Users/chris and christine/Desktop/VFE_3.0/docs/research/2026-10-24-mlp-generative-models/implementation.md>). The mathematical qualifications are developed in [probabilistic-model.md](<C:/Users/chris and christine/Desktop/VFE_3.0/docs/research/2026-10-24-mlp-generative-models/probabilistic-model.md>) and [gauge-and-species.md](<C:/Users/chris and christine/Desktop/VFE_3.0/docs/research/2026-10-24-mlp-generative-models/gauge-and-species.md>).

## 2. The normalized generative model behind the block

Fix one token, one layer, and a positive-definite incoming covariance $C$. Let $A\in\{1,\ldots,M\}$ be the presentation label, $X\in\mathbb R^K$ the represented content, and $O\in\mathbb R^K$ the record. Declare

$$
p(a,x,o\mid C)=\pi_a\mathcal N(x;\nu_a,\Lambda_a)\mathcal N(o;x,C),
$$

with positive normalized weights and positive-definite component covariances. The implemented component covariances are diagonal in the coordinates where the bank is defined. The joint integrates to one by integrating the record, content, and label in order. Conditioning on $O=\xi$ gives

$$
Z_C(\xi)=\sum_a\pi_a\mathcal N(\xi;\nu_a,C+\Lambda_a),
$$

$$
w_a=\frac{\pi_a\mathcal N(\xi;\nu_a,C+\Lambda_a)}{Z_C(\xi)},\qquad
P_a=(C^{-1}+\Lambda_a^{-1})^{-1},
$$

$$
m_a=\xi+C(C+\Lambda_a)^{-1}(\nu_a-\xi),\qquad
p(a,x\mid\xi,C)=w_a\mathcal N(x;m_a,P_a).
$$

This precisely instantiates the manuscript's reference law, evaluator, and record kernel in [Definition `def:generative-model`](<C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/chapters/01_foundations.tex:165>). The categorical posterior is a belief over presentations, and averaging the conditional means is Bayesian model averaging. A soft distribution over components is maintained during the calculation; there is no obligatory hard model choice.

The bank alone specifies $r(a)G_a(dx)$, a prior over presentations and contents. The complete generative model also includes the record kernel. Calling a template a candidate model is defensible because, together with that common kernel, it predicts a normalized record distribution. Calling the forward computation itself the generative model omits the distinction between a model and an inference procedure.

There is a close classical precedent. The measurement model and component posterior are the identity-projection special case of the model used by Bovy, Hogg, and Roweis for Gaussian-mixture inference with heterogeneous observational errors. Their equations 1–7 define the noisy mixture, and equations 13–16 give its component statistics. The correspondence is algebraic; internally computed neural carriers do not automatically possess the empirical noise semantics of measured stellar velocities. [Extreme deconvolution](https://arxiv.org/pdf/0905.2979).

The minimal-agent manuscript originally declares a different full generative experiment: the receiver has prior $X\sim\mathcal N(\xi,C)$, a selected source produces a comparison record with likelihood $\mathcal N(o;X-\nu_a,\Lambda_a)$, and the record is observed at zero. At that observation its unnormalized posterior is the same Gaussian product as above. This gives the same evidence value and posterior for the declared inputs. It does not identify the two full laws over all possible records: one places the incoming Gaussian in the prior, while the other places its mean in an observed carrier. See [the minimal-agent model](<C:/Users/chris and christine/Desktop/Research/manuscripts/Attention/minimal_agent_model.tex:126>). This is a legitimate alternative probabilistic interpretation with a precise scope.

The general construction does not require Gaussians. A proper presentation prior, normalized evaluator $G_a(dx)$, and normalized record kernel $H(x,do)$ suffice. Gaussian laws make this realization analytically tractable. Limitations of its same-state observation model should not be promoted into limitations of all statistical agents.

## 3. Inference, posterior compression, and the actual free energy

For any normalized recognition law $\zeta(a,x)$, the local free energy is

$$
\mathcal F(\zeta;\xi,C)
=\mathbb E_\zeta[\log\zeta(A,X)-\log p(A,X,\xi\mid C)]
=D_{\mathrm{KL}}(\zeta\Vert p(\cdot\mid\xi,C))-\log Z_C(\xi).
$$

It equals negative log evidence at the full correlated posterior. The exact helper computes its label weights and moments without retaining that whole joint as the next layer's state. Its content marginal is a Gaussian mixture. The full moment-matched covariance is

$$
\bar m=\sum_a w_a m_a,\qquad
\bar S=\sum_a w_a\left[P_a+(m_a-\bar m)(m_a-\bar m)^\top\right].
$$

Replacing the mixture by $Q=\mathcal N(\bar m,\bar S)$ preserves its first two moments. It generally removes multimodality. It is the Gaussian minimizer of $D_{\mathrm{KL}}(p_X\Vert Q)$, which is a different optimization direction from the joint free energy. A diagonal or block representation discards additional covariance entries. Passthrough covariance retains the original covariance instead of these posterior moments; nonzero training dropout modifies the mean increment after inference.

The current [ELBO function](<C:/Users/chris and christine/Desktop/VFE_3.0/vfe3/model/prototype_agents.py:717>) evaluates $Q(X)r(A)$. Its [exact-mode caller](<C:/Users/chris and christine/Desktop/VFE_3.0/vfe3/model/block_mlp.py:1863>) inserts $Q=\mathcal N(\bar m,\bar S)$ and $r=w$. The exact gap of this diagnostic is

$$
\log Z_C(\xi)-\mathcal L(Qr)
=D_{\mathrm{KL}}(Q\Vert p_X)
+\mathbb E_Q D_{\mathrm{KL}}\left(r\Vert p(A\mid X,\xi,C)\right).
$$

This separates the error of the Gaussian marginal from the error of making the label independent of content. Both can be present. A positive gap in exact mode is therefore compatible with accurate exact posterior moments. Conversely, the diagnostic does not necessarily measure the state propagated by a passthrough or dropout configuration, because it reconstructs a posterior covariance and omits dropout.

For the product family $Q(X)r(A)$, coordinate ascent alternates

$$
r_a\propto\pi_a\exp\left[-D_{\mathrm{KL}}(Q\Vert\mathcal N(\nu_a,\Lambda_a))\right],
$$

$$
S^{-1}=C^{-1}+\sum_a r_a\Lambda_a^{-1},\qquad
m=S\left(C^{-1}\xi+\sum_a r_a\Lambda_a^{-1}\nu_a\right).
$$

The Gaussian coordinate always uses the original incoming $C,\xi$. It is not repeated multiplication of the same likelihood into successive Gaussian posteriors. Each exact coordinate update improves or preserves the same local ELBO, but a finite schedule is not a converged optimizer, and convergence need not select a global optimum. This is the usual local-coordinate variational structure; [Winn and Bishop](https://www.jmlr.org/papers/v6/winn05a.html) provide its general message-passing treatment.

### A concrete example

Take $K=1$, $\xi=0$, $C=1$, equal component weights, template means $\pm3$, and template variances $0.25$. The posterior components have means $\pm2.4$ and variance $0.2$. The exact marginal has mean zero and variance $5.96$. The symmetric CAVI initialization stays at equal labels and $Q=\mathcal N(0,0.2)$, however many sweeps are taken. Both means are zero; their uncertainty statements differ sharply.

![Exact mixture, moment compression, and mean-field uncertainty](<C:/Users/chris and christine/Desktop/VFE_3.0/docs/research/2026-10-24-mlp-generative-models/posterior-comparison.png>)

For the exact moments inserted into the product diagnostic, the evidence gap is

$$
28.8-\tfrac12\log(29.8)=27.1027458032\text{ nats}.
$$

The symmetric CAVI gap is $14.4$ nats. A recognition law that selects one label and uses its conditional Gaussian has gap $\log2$, proving that this symmetric fixed point is not the global variational optimum. These numbers were reproduced by the actual helper and ELBO functions in [check_cases.py](<C:/Users/chris and christine/Desktop/VFE_3.0/docs/research/2026-10-24-mlp-generative-models/check_cases.py>); the complete output is [check-results.json](<C:/Users/chris and christine/Desktop/VFE_3.0/docs/research/2026-10-24-mlp-generative-models/check-results.json>). A smaller product-law gap does not imply a more faithful multimodal marginal: the two objectives reward different approximations.

## 4. What the incoming record means

The note's communication-then-perception interpretation is coherent as a local computational organization. Attention and the configured normalization produce a carrier; the template model interprets that carrier. A normalized conditional model for any held $\xi,C$ is enough to define the local operation. Whether that carrier is an empirically justified observation of an underlying content is a further modeling question.

In particular, if an incoming belief already equals $q_{\mathrm{in}}(x)\propto p_0(x)L_D(x)$ after some data $D$, replacing its prior by the bank density $h(x)$ gives

$$
p_h(x\mid D)\propto h(x)L_D(x)\propto h(x)\frac{q_{\mathrm{in}}(x)}{p_0(x)}.
$$

The prototype product $h(x)q_{\mathrm{in}}(x)$ is different. It can be justified by the separately declared record model, or by adding another legitimate factor, but it is not automatically prior replacement for the same data. Reusing an internal summary of already incorporated evidence does not create a fresh independent observation. This does not invalidate the code's local model; it prevents a local identity from silently acquiring stronger evidence semantics.

The covariance also needs a role. $p(o\mid x,C)$ is a normalized kernel for each fixed $C\succ0$. If a proposed unconditional record model instead substitutes a record-dependent $C(o)$ inside a Gaussian density, normalization must be checked again. Merely treating the numerical value as a covariance argument does not establish a law for its generation. A conditional model with covariance as declared side information is a defensible local scope.

Depth adds the same issue. If independent per-layer factors are declared, they need a common latent typing, conditioning graph, and appropriate normalization. If layers are inference iterations for one model, they must update coordinates of that fixed objective. The present succession of separately learned banks and transformed carriers establishes neither interpretation automatically. The language model does define a normalized autoregressive distribution through its categorical next-token predictions. What remains unestablished is a single internal latent-variable ELBO whose coordinates are all these operations.

## 5. Sharing a generative model requires the record kernel

The manuscript's species condition is stronger than parameter sharing. It requires, on a fixed record space,

$$
r_j=\phi_\#r_i,\qquad G_j(\phi(a))=\psi_\#G_i(a),\qquad
\ell_j(\psi(x),\phi(a);\cdot)=\ell_i(x,a;\cdot).
$$

For the note's literal kernel $\ell_i(x;do)=\mathcal N(do;x,C_i)$, equality of the Gaussian laws on the two sides forces $\psi(x)=x$ and $C_j=C_i$. Transforming evaluator means and covariances through token frames does not satisfy the record-kernel identity by itself. The definition has no record-coordinate transformation that can be silently added.

A complete reference-coordinate construction can satisfy it. Put $G_i(a)=(U_i)_\#\mathcal N(\nu_a,\Lambda_a)$, use one common reference record space, and define

$$
\ell_i(x;do)=\mathcal N(do;U_i^{-1}x,\bar C).
$$

With common $\bar C$, the map $\psi_{ji}=U_jU_i^{-1}$ establishes all three identities. If observation covariances differ, one can specify a common kernel conditioned on an observed covariance design variable. The geometry memo also gives a normalized augmented-record construction using a common proper law for that variable. These are explicit model completions, not evidence that upstream attention actually samples their records.

Conversely, numerically identical local banks and equal local noise covariances can satisfy the formal species definition using identity maps, even if they do not represent one physically aligned model under the intended token-frame interpretation. The note's claim that different frames necessarily make the local path fail is therefore too strong. Formal equivalence of kernels and passive covariance of an architecture are separate properties.

The reference block gives the latter property when reference inputs remain invariant and the output family preserves the required congruence. Full covariance allows this. Local diagonal storage generally does not: removing off-diagonal posterior covariance does not commute with frame changes. Diagonal template covariances in a fixed reference chart are a different restriction and do not themselves prevent passive local-frame covariance. The [geometry memo](<C:/Users/chris and christine/Desktop/VFE_3.0/docs/research/2026-10-24-mlp-generative-models/gauge-and-species.md>) supplies an explicit two-dimensional counterexample and the current configuration boundaries.

A finite label set can carry a trivial group action or a compatible permutation action. A continuous action of a connected group on a finite discrete set is trivial. This still permits a nontrivial categorical belief law and an equivariant evaluator; it does not supply the continuous parameter-space model frame contemplated elsewhere in the theory.

## 6. How the relation to an ordinary MLP should be stated

An ordinary residual MLP computes

$$
F_{\mathrm{MLP}}(x)=x+W_2\operatorname{GELU}(W_1x+b_1)+b_2.
$$

Interpreting the first matrix as pattern detectors and the second as associated output directions has empirical support. Geva and colleagues study this structure in trained language models, using ReLU coefficients in their reported model. Their results support an associative-memory interpretation; they do not establish Bayesian component inference, normalized GELU responsibilities, or a generative covariance for every unit. [Transformer Feed-Forward Layers Are Key-Value Memories](https://aclanthology.org/2021.emnlp-main.446/).

The precursor `gaussian_template` block has free value vectors, so its common-covariance case resembles a softmax-gated memory directly. `prototype_agents` adds a stronger constraint: its content displacement and uncertainty follow from conditioning on the same template that supplies the score. With $C=cI$, $\Lambda_a=\lambda I$, its exact mean is

$$
F(\xi)=\frac{\lambda}{c+\lambda}\xi+
\frac{c}{c+\lambda}\sum_a w_a(\xi)\nu_a,
$$

$$
w_a(\xi)=\operatorname{softmax}_a\left[
\log\pi_a+\frac{\nu_a^\top\xi}{c+\lambda}
-\frac{\|\nu_a\|^2}{2(c+\lambda)}\right].
$$

The detector directions and retrieved template means are tied, and the residual coefficient is constrained by uncertainty. The first mean-field label coordinate instead has scale $1/\lambda$. Thus “replace GELU by softmax” does not describe all the differences between an arbitrary MLP and the prototype block.

GELU itself has an exact probabilistic identity. If $h=W_1x+b_1$ and independent $B_j\mid x\sim\operatorname{Bernoulli}(\Phi(h_j))$, then

$$
\mathbb E\left[x+W_2(h\odot B)+b_2\mid x\right]=F_{\mathrm{MLP}}(x).
$$

The original GELU paper explicitly derives the activation as an expected input-dependent stochastic gate. The Bernoulli probability is $\Phi(h_j)$, while $h_j\Phi(h_j)$ is a signed amplitude. This is a normalized conditional gating interpretation with multiple simultaneous gates; it is not a one-label posterior over fixed content models. Adding a proper output-noise kernel supplies a conditional generative distribution, but does not derive the chosen matrices or their learning objective. [Hendrycks and Gimpel, Section 2](https://arxiv.org/html/1606.08415v5).

More generally, any deterministic block can be placed inside a normalized conditional law such as $Y\mid x\sim\mathcal N(F(x),Q)$. Existence of some generative representation is therefore a weak discriminator. The substantive property here is that specified statistical objects determine the prototype update and its uncertainty. The supplied note should call the GELU connection an associative-memory analogy unless it supplies a particular target generative model, an inference approximation, and a criterion for approximation error.

This distinction also prevents an inflated novelty claim. Persistent learned key/value memories have already replaced feed-forward sublayers in language-model experiments by Sukhbaatar and colleagues. Their construction uses attention over persistent vectors and does not carry the present Gaussian posterior covariance or establish backpropagation-free learning. It supports the plausibility of replacing the MLP's computational role, not performance parity for this implementation. [Augmenting Self-attention with Persistent Memory, Section 4](https://arxiv.org/pdf/1907.01470).

## 7. The precise expressive restriction and the existing extension

The [October 20 investigation](<C:/Users/chris and christine/Desktop/VFE_3.0/docs/research/2026-10-20-neural-free-continuation/prototypes.md>) already establishes that exact same-state retrieval has a constrained response. For fixed $C$ and bank density $h$, write

$$
p_\xi(x)\propto h(x)\exp\left[-\tfrac12x^\top C^{-1}x+x^\top C^{-1}\xi\right].
$$

Differentiating the log normalizer of this exponential family gives

$$
F(\xi)=\xi+C\nabla_\xi\log Z_C(\xi),\qquad
D_\xi F=\operatorname{Cov}_{p_\xi}(X)C^{-1}.
$$

The whitened Jacobian is symmetric positive definite. In one dimension the exact posterior mean is strictly increasing. The residual GELU map $x-4\operatorname{GELU}(x)$ has derivative $-1$ at zero, so this prototype model cannot equal every GELU residual map on a neighborhood, regardless of bank size. The covariance example above also shows why this must not be described as contraction: $F'(0)=5.96$ there. The fresh two-dimensional calculation reproduced the response identity to about $9\times10^{-12}$, but the displayed argument establishes its stated general scope.

The restriction assumes fixed covariance and bank, the same-state measurement model, and exact posterior means. It does not automatically apply to an input-dependent covariance, finite CAVI, an additional output map, or the whole network. It also does not show that the excluded local map is required for language modeling.

The existing joint cue/result extension addresses this specific restriction. Let a component model a pair of contents,

$$
\begin{pmatrix}X\\Y\end{pmatrix}\mid A=a
\sim\mathcal N\left[
\begin{pmatrix}\nu_a\\\omega_a\end{pmatrix},
\begin{pmatrix}L_a&B_a^\top\\B_a&V_a\end{pmatrix}\right],
\qquad V_a-B_aL_a^{-1}B_a^\top\succ0,
$$

and observe $O\mid X\sim\mathcal N(X,R)$. Then

$$
w_a(o)\propto\pi_a\mathcal N(o;\nu_a,L_a+R),\qquad
\mathbb E[Y\mid o,a]=\omega_a+B_a(L_a+R)^{-1}(o-\nu_a),
$$

$$
\operatorname{Cov}(Y\mid o,a)=V_a-B_a(L_a+R)^{-1}B_a^\top.
$$

The receiving map is determined by cross-covariance. It can rotate or reverse a response without violating positive definiteness. For example, with $L=I$, $R=rI$, $B=(1+r)A_0$, and $V=BB^\top+Q$, $Q\succ0$, the conditional mean has arbitrary chosen linear part $A_0$. The component covariance is valid by its Schur complement. This is a concrete statistical alternative already present in the project, not a new architecture proposed here. Learning joint densities and conditioning them to form regression maps has a classical precedent in [Ghahramani and Jordan](https://proceedings.neurips.cc/paper/1993/file/f2201f5191c4e92cc5af043eebfd0946-Paper.pdf).

The extension still needs predictive training and a practical representation of its covariance. Merely treating cross-covariance parameters as another unconstrained layer optimized by the current downstream loss would not establish the program's separate backpropagation-free objective. Likewise, factorial latent-feature models already discussed in the earlier report address simultaneous explanations rather than one-of-$M$ competition; they are separate modeling choices with different inference costs.

## 8. Fast model belief, slow parameters, and prediction

The distinction between fast inference and slow learning is useful. $w_{ia}$ or $r_{ia}$ expresses which component explains the current record. The means, widths, and prior weights describe the component family and are updated during training. Shared parameters do not imply identical model beliefs; identical model beliefs do not imply identical predictions without compatible evaluators and record kernels.

There is currently no posterior over bank parameters. Calling the bank a repertoire of models or a shared statistical memory is accurate. Calling it a hyperprior would require a declared law over those parameters. Calling its components semantic concepts or social norms is an interpretation to test through their predictive behavior, not a consequence of Gaussian normalization.

The optional EM term fits incoming internal records using detached posterior statistics and a gradient update on the bank. It is not a closed-form M-step, and the rest of the network still uses backward. A model that explains its incoming carriers need not preserve distinctions useful for predicting words. In the already developed contextual-emission construction, a token emission $\psi_{av}$ and pre-observation component weights $\rho_a(m)$ give

$$
p(v\mid m)=\sum_a\rho_a(m)\psi_{av},\qquad
\gamma_a^v=\frac{\rho_a(m)\psi_{av}}{p(v\mid m)}.
$$

For Gaussian-overlap weights with covariance $R_a$, the local score contains the normalization correction

$$
\nabla_m\log p(v\mid m)
=\sum_a(\gamma_a^v-\rho_a(m))[-R_a^{-1}(m-\nu_a)].
$$

This prior result identifies an actual word-dependent learning signal. It should not be replaced by target-blind template reconstruction and then described as the same learning mechanism. Its causality requirement is forecast from the prefix, score the next word, and only then assimilate that word for learning. See the [prior local-learning analysis](<C:/Users/chris and christine/Desktop/VFE_3.0/docs/research/2026-10-20-neural-free-continuation/local-learning.md>).

Carrying $s_i$ through depth is also not a purely mechanical addition. Separate layers have independently learned banks, so label index $a$ has no guaranteed persistent meaning. A transition $T_{ba}=P(A^{\ell+1}=b\mid A^\ell=a)$ can map one label law into another, but its semantics and the evidence used in the next update must be specified. Carrying only the label marginal also loses its dependence on content. If preservation of the theory's joint $\zeta_i$ is the aim, the relevant obligation is a joint state/model transition, not just an extra vector of weights.

## 9. Encoder and decoder: a shared vocabulary, different operations

The encoder's observed word selects an evaluator-like law over content. The prototype block infers an unobserved template label and revises content. The decoder holds a final content belief fixed and computes a categorical score over word templates. This is a useful common language for explaining the architecture.

At unit effective temperature, the Gaussian-KL decoder rule

$$
r_v\propto\pi_v\exp[-D_{\mathrm{KL}}(q\Vert p_v)]
\propto\pi_v\exp[\mathbb E_q\log p_v(X)],
$$

is exactly a mean-field label coordinate. It is generally different from the exact noisy-record posterior, which integrates the likelihood before taking its logarithm, and from averaging a classifier $p(v\mid X)$ over $q$. Other temperatures define a tempered categorical rule; an equivalent ordinary Bayesian representation would require explicit changes to the model rather than retaining all original meanings unchanged. The live decoder temperature is `0.1`.

The encoder lookup alone supplies an evaluator, not all three pieces of the manuscript's generative model. The decoder bank is untied from the encoder in the live launcher, and neither is tied to the intermediate prototype banks. Consequently “three uses of Gaussian template families” is supported. “Three manifestations of one established generative joint” is not.

## 10. Evidence, remaining obligations, and usable wording

The numerical evidence is deliberately small. One CPU float64 process used one thread, $K\le2$, and $M\le3$, with CUDA hidden. Independent precision-form Gaussian conditioning agreed with the exact helper to at most $2.23\times10^{-16}$ for means, covariance, labels, and evidence. Finite differences checked the posterior-response and evidence-score identities to below $2\times10^{-11}$. A diagonal-layout example reproduced the diagonal of the dense result while discarding an off-diagonal covariance of about $-0.2745$. A nonsymmetric example improved its ELBO from one through eight CAVI sweeps, while label staleness did not decrease monotonically. The symmetric example reproduced the larger gaps discussed above. These calculations establish those instances and illustrate the derivations; they are not language-model or performance measurements.

The live implementation uses a coordinate GELU block at `K=128`; the prototype path is selectable and currently inactive. Earlier collapse results depend on their training regime. The short lower-learning-rate runs reported in the project retained diffuse responsibilities, but their effective-template count does not establish semantic specialization or performance parity. No checkpoint or training comparison was rerun here.

The remaining obligations depend on the intended claim. To claim that a layer expresses each agent's complete generative model, specify its record space and the role of the incoming covariance. To claim persistent beliefs over models, specify the label and content transition across layers. To claim a single internal free energy, supply one normalized joint and a recognition family whose coordinates match the executed operations. To claim a useful MLP replacement, establish predictive contribution under a matched training comparison. These are different obligations; solving one does not settle the others.

A technically accurate statement of the perspective is:

> The prototype block gives each token access to a shared family of probabilistic content models. Given its current carrier and uncertainty, the token infers which models apply and how they revise its content belief. Exact inference produces a correlated mixture over model identity and content; the implementation compresses that result, or uses mean-field coordinates, before continuing. This replaces the local GELU computation with a specified inference mechanism. The model of internal records, the persistence of model beliefs, and predictive learning across the complete stack require additional structure.

This wording preserves the intended generative-model interpretation while matching the current implementation and the manuscript's definitions.

## Source and artifact notes

The live source anchors are linked where each claim is used. The starting note, Research manuscripts, wiki, and previous investigations remained read-only. The [Template routing wiki page](<C:/Users/chris and christine/Desktop/Research/wiki/methods/Template routing.md>) was consulted for continuity; its implementation-status statements predate the implemented reference and prototype paths. No vault update was made.

The supporting artifacts are the three analysis memos linked above, the executable numerical example and JSON output, and the [editable plotting script](<C:/Users/chris and christine/Desktop/VFE_3.0/docs/research/2026-10-24-mlp-generative-models/plot_posterior.py>). The figure is also available as [SVG](<C:/Users/chris and christine/Desktop/VFE_3.0/docs/research/2026-10-24-mlp-generative-models/posterior-comparison.svg>). The artifact directory is covered by the repository's existing ignore rules; these local files were saved but not committed or published.
