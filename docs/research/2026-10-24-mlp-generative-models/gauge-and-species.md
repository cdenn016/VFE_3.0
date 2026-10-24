# Gauge covariance and species for prototype generative models

October 24, 2026. The species claim in [the supplied note](<C:/Users/chris and christine/Desktop/mlp_as_generative_models.md>) is too strong: transforming the evaluator does not establish the record-kernel equality required by `def:shared-model`. Conversely, identical local-coordinate mechanisms can satisfy that definition even when they do not represent one common reference-space model. These are distinct questions.

This memo checks the note against the live [MAgent definitions](<C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/chapters/01_foundations.tex:165>), the [attention frame convention](<C:/Users/chris and christine/Desktop/Research/manuscripts/Attention/GL(K)_attention_v2.tex:347>), and the implementation cited below. The [October 20 investigation](<C:/Users/chris and christine/Desktop/VFE_3.0/docs/research/2026-10-20-neural-free-continuation/prototypes.md>) already identifies the reference/local covariance distinction. The wiki's [Template routing](<C:/Users/chris and christine/Desktop/Research/wiki/methods/Template routing.md>) still describes the reference and structured-posterior variants as unimplemented; those status statements are stale relative to the inspected code. No scripts, tests, training, or GPU calculations were run for this memo. All counterexamples below are algebraic.

## Target and conventions

The target is to decide, for every pair of token agents in a layer, whether their normalized generative mechanisms satisfy the manuscript's exact equivalence, and separately whether the prototype operation commutes with independent passive changes of local coordinates. The note supplies finite labels, Gaussian templates, and a Gaussian record law. Here the Gaussian calculations assume finite dimension, strictly positive definite covariances, and positive template probabilities. The corrected common-record constructions below are additional assumptions, not discoveries about the upstream attention law.

| Object | Type and convention |
|---|---|
| Base | One context, \(\mathcal C=\{c_*\}\); token positions index agents above it. |
| Principal fiber | A right \(G=\mathrm{GL}(K,\mathbb R)\) torsor; choose reference \(\sigma\), with \(u_i=\sigma U_i^{-1}\). |
| Content fibers | Reference \(V=\mathbb R^K\), local copy \(E_i\), frame \(U_i:V\to E_i\); vectors are columns. |
| Representation | Left action \(k\mapsto g_i k\); laws transform by pushforward. The implemented exponential frames have positive determinant, with further block restrictions when configured. |
| Model presentations | Finite set \(\mathsf M=\{1,\ldots,M\}\), usually with trivial gauge action; optional relabelings are permutations. |
| Statistical section | At the singleton base, \(\zeta_i(da,dk)=s_i(a)q_i(dk\mid a)\), a law on presentations and contents. |
| Record space | A declared common measurable space \(\mathsf O\); the species definition does not transform records. |
| Endpoint identification | \(T_{ji}=U_jU_i^{-1}:E_i\to E_j\). It is not independently specified connection transport. |
| Geometry | Gaussian Fisher metric and relative entropy; covariance matrices are SPD tensors. Moment matching uses ordinary moments, not a Riemannian exponential or retraction. |

The manuscript's [passive convention](<C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/chapters/01_foundations.tex:76>) is

\[
(\mu_i,\Sigma_i,U_i)\mapsto(g_i\mu_i,g_i\Sigma_i g_i^\top,g_iU_i).
\]

Consequently \(\xi_i=U_i^{-1}\mu_i\) and \(C_i=U_i^{-1}\Sigma_iU_i^{-\top}\) are unchanged. For tangent pairs \((v,H),(w,L)\), the Gaussian Fisher metric is

\[
\langle(v,H),(w,L)\rangle_{(\mu,\Sigma)}
=v^\top\Sigma^{-1}w+\tfrac12\operatorname{tr}(\Sigma^{-1}H\Sigma^{-1}L).
\]

Substitution of \((g\Sigma g^\top)^{-1}=g^{-\top}\Sigma^{-1}g^{-1}\) and cyclicity of trace prove its invariance under simultaneous congruence. This does not preserve a fixed Euclidean metric unless \(g\) is orthogonal. Lebesgue densities acquire \(|\det g|^{-1}\), rather than remaining numerically unchanged; that common factor cancels in likelihood ratios and relative entropy.

## The species condition includes the record law

[Definition `def:shared-model`](<C:/Users/chris and christine/Desktop/Research/manuscripts/MAgent_Theory_2.0/chapters/01_foundations.tex:185>) requires bimeasurable bijections \(\phi:\mathsf M\to\mathsf M\) and \(\psi:\mathsf K\to\mathsf K\) satisfying, for every presentation and content,

\[
r_j=\phi_\#r_i,\qquad
G_j(\phi(a),\cdot)=\psi_\#G_i(a,\cdot),\qquad
\ell_j(\psi(k),\phi(a);\cdot)=\ell_i(k,a;\cdot).
\]

There is no pushforward on the final record argument. For the note's identity-record model \(\ell_i(k;do)=\mathcal N(do;k,C_i)\), the last equality becomes

\[
\mathcal N(\psi(k),C_j)=\mathcal N(k,C_i).
\]

Equality of these probability measures implies equality of their means and covariances. Thus \(\psi(k)=k\) for every \(k\) and \(C_j=C_i\). This holds even when \(\psi\) is allowed to be nonlinear. Merely choosing \(\psi=U_jU_i^{-1}\) does not work unless that map is the identity. A one-dimensional witness is \(U_i=1,U_j=2,C_i=C_j=1\): at \(k=1\), the proposed record equality would identify \(\mathcal N(2,1)\) with \(\mathcal N(1,1)\).

The converse qualification matters. If all agents have the same numerical local bank and the same numerical \(C_i\), identity maps satisfy the species definition. Different frames alone do not negate this. Such a bank generally changes its reference-space interpretation with the frame, so formal species membership does not establish passive covariance or a common physical interpretation of local coordinates. The definition permits bijections beyond the designated frame identifications.

A useful complete criterion within linear-Gaussian equivalences is obtained by writing

\[
G_i(a)=\mathcal N(\nu_{ia},\Lambda_{ia}),\qquad
\ell_i(k)=\mathcal N(B_i k+b_i,R_i),\qquad \psi(k)=Ak,
\]

where \(A\) is invertible and \(\phi\) is a permutation. All three conditions hold if and only if

\[
\pi_{j,\phi(a)}=\pi_{ia},\quad
\nu_{j,\phi(a)}=A\nu_{ia},\quad
\Lambda_{j,\phi(a)}=A\Lambda_{ia}A^\top,
\]

\[
B_jA=B_i,\qquad b_j=b_i,\qquad R_j=R_i.
\]

Necessity follows by comparing categorical masses and Gaussian moments, with the record means compared for all \(k\); substitution proves sufficiency. This characterizes the specified linear witnesses, not every possible nonlinear equivalence.

## A normalized construction that does establish species

Choose a common reference bank \(\pi_a,\mathcal N(\nu_a,\Lambda_a)\), a fixed common record space \(V\), and one \(\bar C\succ0\). Define

\[
G_i(a)=(U_i)_\#\mathcal N(\nu_a,\Lambda_a),\qquad
\ell_i(k;do)=\mathcal N(do;U_i^{-1}k,\bar C).
\]

Every factor is normalized, so \(\pi_aG_i(a,dk)\ell_i(k;do)\) defines a proper joint. For \(\phi=\mathrm{id}\), \(\psi=U_jU_i^{-1}\), the evaluator equality is composition of pushforwards and the record equality follows from \(U_j^{-1}\psi(k)=U_i^{-1}k\). Hence every pair shares a model exactly. Conditioning on a reference record \(o=\xi_i\) gives the usual Gaussian prototype posterior with record-noise covariance \(\bar C\). The observed values may differ between agents without changing their mechanism.

The actual block accepts heterogeneous reference covariances \(C_i\). Declaring a common design-conditioned kernel \(\ell_i(k;do\mid D)=\mathcal N(do;U_i^{-1}k,D)\) describes one mechanism indexed by design. Equality holds at the same design; fixing different designs does not prove the original fixed-context species assertion.

Alternatively, introduce an augmented record \((D,o)\) with a common proper law \(\eta\) on \(\mathrm{SPD}(K)\):

\[
\ell_i(k;dD,do)=\eta(dD)\mathcal N(do;U_i^{-1}k,D).
\]

Integrating first over \(o\), then over \(D\), gives total mass one. The same frame witness establishes species on this common augmented record space. Conditioning on \((C_i,\xi_i)\) recovers the desired posterior wherever the declared conditional law is defined; a positive density for \(\eta\) on the admitted SPD domain supplies a regular version there. This is an additional normalized model. It does not prove that the upstream covariance actually follows \(\eta\), or justify treating an arbitrary state-dependent covariance as independent generative noise. Allowing record pushforwards would be another possible extension, but changes the manuscript's stated equivalence.

## Kernel covariance and the storage obstruction

Let reference inference return moments \((F(\xi,C),S(\xi,C))\). Then the full-covariance output

\[
\mu_i^+=U_iF(\xi_i,C_i),\qquad
\Sigma_i^+=U_iS(\xi_i,C_i)U_i^\top
\]

obeys \((\mu_i^+,\Sigma_i^+)\mapsto(g_i\mu_i^+,g_i\Sigma_i^+g_i^\top)\), because its reference arguments are unchanged. This establishes a passive coordinate identity, independently of the species condition. It also holds for a fixed finite reference-coordinate approximation; it does not certify that approximation as an exact posterior.

The live [reference wrapper](<C:/Users/chris and christine/Desktop/VFE_3.0/vfe3/model/block.py:427>) performs the inverse-frame read, passes reference covariance to prototypes, and returns posterior covariance through the frame. [The prototype forward method](<C:/Users/chris and christine/Desktop/VFE_3.0/vfe3/model/block_mlp.py:1756>) additionally distinguishes posterior from passthrough covariance and applies residual dropout. The posterior statement here concerns the deterministic inference moments; it is not an assertion that every configured forward output is the posterior.

For local diagonal storage, [the return operation](<C:/Users/chris and christine/Desktop/VFE_3.0/vfe3/model/frame_actions.py:214>) keeps only \(\operatorname{diag}(USU^\top)\). Write \(D(M)\) for the matrix obtained by zeroing its off-diagonal entries. This fails covariance even for admissible rotated inputs. Take \(K=2\), \(U=I\), \(\xi=0\), \(C=I\), equal weights, \(\nu_\pm=\pm(2,2)^\top\), and \(\Lambda_\pm=I\). Gaussian conditioning gives

\[
w_\pm=\tfrac12,\quad m_\pm=\pm(1,1)^\top,\quad
P_\pm=\tfrac12I,\quad
S=\begin{pmatrix}3/2&1\\1&3/2\end{pmatrix}.
\]

The stored covariance is \(\tfrac32I\). Choose

\[
g=\frac1{\sqrt2}\begin{pmatrix}1&-1\\1&1\end{pmatrix}.
\]

The transformed input still has \(\mu'=0,\Sigma'=I\), while \(U'=g\); hence reference inference returns the same \(S\). Its stored output is \(D(gSg^\top)=\operatorname{diag}(1/2,5/2)\), whereas transforming the original stored output gives \(g(3I/2)g^\top=3I/2\). The two differ. Thus diagonal local storage defeats general \(\mathrm{GL}(K)\) covariance, including a rotation subgroup, even though diagonal reference templates do not obstruct the passive identity of the full kernel.

Storing a diagonal approximation in a fixed reference frame could preserve that passive identity in principle. It is not an available prototype route here: [configuration validation](<C:/Users/chris and christine/Desktop/VFE_3.0/vfe3/config.py:1564>) restricts the frame-diagonal family to the coordinate MLP. Arbitrary common reference rebasing is also stronger than passive local re-description: \(U_i\mapsto U_i a\) requires transforming reference templates by \(a^{-1}\), and generally takes diagonal template covariances outside their model family.

## Model labels and global scope

A finite model-label fiber is permitted by the general MAgent definition; the linear model-sector realization is optional. Labels may transform by permutations, with \(s'(\phi(a))=s(a)\). A continuous action of connected \(\mathrm{GL}^+(K)\) on a finite discrete set is trivial: each orbit map has connected image, hence one point. Therefore the prototype label posterior does not supply a nontrivial continuous model frame or uncertainty over template parameters. Carrying label beliefs between independently learned layer banks requires an explicit identification or stochastic transition \(Q(b\mid a)\), giving \(s_{\mathrm{next}}(b)=\sum_aQ(b\mid a)s(a)\); equal array indices do not establish semantic alignment.

At the singleton base, \(T_{kj}T_{ji}=T_{ki}\), so endpoint identifications have identity products around loops. No connection, path dependence, or global topological result follows. Over a nontrivial contextual base, local templates would need overlap compatibility under transition functions satisfying their cocycle identities; a globally constant reference bank cannot simply be assumed without a suitable trivialization or equivariant descent construction.

The corrected wording is: **Reference-coordinate prototypes provide a passively covariant inference kernel when their full returned covariance is retained. Tokens form one species under the stated definition only when their presentation priors, evaluators, and fixed-record kernels satisfy all three equivalence conditions. A common reference bank alone is insufficient.** The remaining modeling obligation is to specify the record space and the status of heterogeneous incoming covariance. Choosing a common design law or an explicit normalized augmented record is the next action that would settle that obligation; it is separate from implementing persistent model beliefs or measuring language-model performance.
