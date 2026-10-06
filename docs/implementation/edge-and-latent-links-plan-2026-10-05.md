# Edge-dependent and latent content links: implementation plan (2026-10-05)

Binding specification for the implementation and review agents. Author of the specification: Claude
Opus 5.5 (the orchestrating session). Source theory: `Research/manuscripts/Attention/GL(K)_attention_v2.tex`
Section 3 (`thm:routed_transport`, `prop:lie_split`, `lem:lie_lowrank`) and the latent-link supplement
`GL(K)_supplementary_v2.tex` Section S6 (`sec:latent_links`, Eq. `eq:lie_latent_model`, `prop:lie_field_equation`).

## 0. Ground rules

* Repository: `C:/Users/chris and christine/Desktop/VFE_3.0`, branch `main`, working tree. Do NOT commit,
  branch, stash, reset or revert anything. The tree carries the user's uncommitted work in
  `ablation.py`, `train_vfe3.py`, `vfe3/config.py`, `vfe3/viz/figure_worker.py`, `zzzzz.py`,
  `multiseed_analysis.py` and several `docs/*.md`. Never change an existing line of those files except
  where this plan says so. In `vfe3/config.py` you may ADD new fields, ADD validation, ADD entries to
  `inactive_fields()`; never change an existing default value or comment. Do not edit `train_vfe3.py`
  or `ablation.py` at all (the user adds the new keys there).
* No GPU workloads. CPU tests only, tiny models (single-digit or low-double-digit K, N), one thread per
  process. Torch tests run with `C:/anaconda/python.exe`. Command template (Git Bash):
  `cd "C:/Users/chris and christine/Desktop/VFE_3.0" && VFE3_TEST_DEVICE=cpu CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "C:/anaconda/python.exe" -m pytest <files> -q -p no:cacheprovider`
  (add `-n 8` if `pytest-xdist` is importable; each worker stays single-threaded). Do not install
  packages or modify the Python environment. No Lean.
* Conventions (user global rules): type hints on every signature; docstrings with LaTeX for every
  nontrivial formula (`r"""..."""`); variable names in paper notation; tensor shape comments at
  critical points; American spelling; no banned words in docs (key insight, crucially, critically,
  notably, importantly, worth noting, fundamentally, leverages, underscores). Match the surrounding
  code style of `vfe3/model/relational_operators.py`.
* Every new behavior is opt-in: with the new config fields at their defaults the model, its state-dict
  keys, its numerics and every existing test are unchanged.
* Record the change in `docs/2026-10-05.md` (append a new section at the end; that file has user WIP,
  keep it) in the house style of the existing entries: flowing prose, what was added, the exact
  formulas, the config keys and defaults, the tests and their results, costs, and what is not done.

## 1. What exists

Per layer, `RelationalOperators` holds per-head comparison maps `raw_score_q`, `raw_score_k`
(`(H, d, K)`), sending maps `T` (`(H, d, K)`) and receiving factors `O` (`(H, K, d)`); `K = H d`. In the
exponential chart (`content_link_chart="exponential"`) the effective receiving map is
`O' = e^{L} T^{-1} = T^{-1} + O phi_1(T O)` with the shared generator `L = O T` (stacked K x K maps,
evaluated in float64 by `chart_receive_runs`). With reference means `xi_j = U_j^{-1} mu_j` (value
frame), values `v_j^h = T_h xi_j`, pooled values `z_i^h = sum_j beta_ij^h v_j^h`, the message is
`m_i = U_i sum_h O'_h z_i^h` (`predictive_content`), the residual adds `mu_i` when
`attention_residual=True` (c_res = r = 1), and the moment-matched covariance
(`predictive_covariance_update`, dense and blockwise bodies) is the exact covariance of the kernel
`k_i' | k, J ~ N(r k_i + sum_h M_h k_{J^h}, Q_i)` with `M_h = O'_h T_h`. The call site is
`vfe3/gradients/relational.py::relational_predictive_readout`, which computes `receive = O'` once per
layer and passes it to both the mean and the covariance. Read that module and
`vfe3/model/relational_operators.py` completely before writing code.

In reference coordinates with `F = e^{L}`, `R = T^{-1}` (K x K) and `R_h = R[:, block h]` (K x d),
`O'_h = F R_h`, the head projector is `Pi_h = R_h T_h` and `Pi(beta_ij) = sum_h beta_ij^h Pi_h`, and the
message is `m~_i = sum_j F Pi(beta_ij) xi_j` (Theorem `thm:routed_transport`).

## 2. Level 1: edge-dependent links (amortized)

### 2.1 Unified link form

Every edge `j -> i` gets a group-valued reference link
\[
G_{ij} = g_i\,F\,E_{ij},\qquad E_{ij} = I_K + A_i\,\Phi_{ij}\,B_j^\top ,
\]
with per-token factors `A_i, B_j in R^{K x r}` and an `r x r` coefficient `Phi_ij`. The local link is
`Theta_ij = U_i G_ij U_j^{-1}`, the head edge map is `M_h^{ij} = G_ij Pi_h`, and the message is
`m~_i = sum_j G_ij Pi(beta_ij) xi_j`. Two modes (config `content_link_edges`):

* `"lowrank"` (curved): `g_i = I`, `A_i = a(xi_i)`, `B_j = b(xi_j)`, `Phi_ij = phi_1(B_j^T A_i)`, so that
  `E_ij = exp(A_i B_j^T)` by the low-rank identity `exp(A B^T) = I + A phi_1(B^T A) B^T`
  (`lem:lie_lowrank`). `G_ij = e^{L} exp(A_i B_j^T)` is in GL^+(K); its triangle holonomies vary from
  triangle to triangle.
* `"pure_gauge"` (flat control): one shared basis `A in R^{K x r}` per layer (a parameter),
  `B_j = b(xi_j)`, `g_i = exp(A B_i^T) = I + A phi_1(B_i^T A) B_i^T`, `A_i = A` for every i, and
  `Phi_ij = -phi_1(-B_j^T A)` (depends on the sender only), so that
  `E_ij = exp(-A B_j^T) = g_j^{-1}` and `G_ij = g_i e^{L} g_j^{-1}`. Every triangle holonomy
  `G_ik^{-1} G_ij G_jk = g_k e^{L} g_k^{-1}` is conjugate to `e^{L}`: the link field is a gauge transform
  of the shared generator, with the head projector kept in the original reference frame. This is the
  control that has the same content dependence and no curvature beyond the shared class.

The factor maps act on the RMS-normalized reference mean `xi^_i = xi_i / sqrt(mean_k xi_{ik}^2 + 1e-12)`
(frame invariant, so `G_ij` is reference data and `Theta_ij` transforms covariantly under the passive
action `mu -> g mu, Sigma -> g Sigma g^T, U -> g U`):

* `"lowrank"`: parameters `edge_receiver_map` `W_A in R^{(K r) x K}`, initialized `N(0, K^{-2})` from the
  module's seeded generator (so A_i has columns of expected unit norm), and `edge_sender_map`
  `W_B in R^{(K r) x K}`, initialized to zero. `A_i = reshape(W_A xi^_i, (K, r))`, `B_j = reshape(W_B xi^_j, (K, r))`.
* `"pure_gauge"`: parameters `edge_site_basis` `A in R^{K x r}` (orthonormal columns from the seeded
  generator, via the existing `_orthonormal_columns`) and `edge_sender_map` `W_B` as above, zero.

With `W_B = 0` both modes reduce EXACTLY (to roundoff) to the exponential chart: `B = 0` gives
`E_ij = g_i = I`. That is the step-0 state and a mandatory regression test.

### 2.2 Mean

With `c_ij = sum_h beta_ij^h p_j^h`, `p_j^h = (B~_j)_h^T v_j^h` (r-vector), `B~_j = R^T B_j = T^{-T} B_j`
(K x r; `(B~_j)_h` its d x r row block of head h), and `y_i = sum_j Phi_ij c_ij`,
\[
\tilde m_i = g_i\,O'\bigl(z_i + T A_i\,y_i\bigr),\qquad m_i = U_i\,\tilde m_i ,
\]
where `z_i` stacks `z_i^h` and `T A_i` stacks `T_h A_i`. Derivation: `Pi(beta_ij) xi_j = R y_ij` with
`y_ij = stack_h beta_ij^h v_j^h`, `F R = O'`, `F A_i = O' T A_i`, `B_j^T R y_ij = c_ij`. In
`"pure_gauge"` this is `g_i O'(z_i - T A sum_j phi_1(-B_j^T A) c_ij)`, with no per-edge matrix function.

Causality and finiteness: masked pairs (`beta_ij^h = 0` for every h, the future) contribute nothing, but
`phi_1` of a large argument overflows and `inf * 0 = nan`. Zero the argument `B_j^T A_i` on pairs whose
routing row is zero in every head BEFORE `phi_1` (then `phi_1(0) = I`). Use the existing `_phi1`
(augmented `matrix_exp`), batched; compute every `Phi_ij` once per layer and share it between the mean,
the covariance and the diagnostics. Run the edge algebra with autocast disabled at the compute dtype of
the existing code (`_compute_dtype`, float64 under an escalated frame); `F = O' T`, `R = T^{-1}` come
from the float64 chart evaluation (add a helper that returns the float64 stacks `T`, `R`, `O'`, `F` once
per layer and use it in place of a second inversion).

### 2.3 Moment-matched covariance

The kernel is unchanged except `M_h -> M_h^{ij} = G_ij Pi_h`. With `s_j^{hh'} = T_h C_j T_{h'}^T` (the
existing head-pair source blocks), `c_ij^{hh'}` the existing coefficients (`beta^h` on the diagonal,
`beta^h beta^{h'}` off it), and `v`, `z`, `Sigma~_i^h` the existing values, pooled values and head spread:
\[
C_i^{\mathrm{att}} = g_i\,F\bigl(\hat P_i + \textstyle\sum_h R_h \tilde\Sigma_i^h R_h^\top
 + A_i X_i + X_i^\top A_i^\top + A_i Y_i A_i^\top\bigr)F^\top g_i^\top
 + r\bigl[C_i + \mathrm{cross}_i + \mathrm{cross}_i^\top\bigr] + \tilde Q_i ,
\]
\[
X_i = \sum_j \Phi_{ij}\sum_{h,h'} c_{ij}^{hh'} (\tilde B_j)_h^\top s_j^{hh'} R_{h'}^\top
 + \sum_h\Bigl[\sum_j \beta_{ij}^h\,\Phi_{ij} p_j^h\, v_j^{h\top} - q_i^h z_i^{h\top}\Bigr] R_h^\top ,
\qquad q_i^h = \sum_j \beta_{ij}^h \Phi_{ij} p_j^h ,
\]
\[
Y_i = \sum_j \Phi_{ij}\Bigl[\sum_{h,h'} c_{ij}^{hh'} (\tilde B_j)_h^\top s_j^{hh'} (\tilde B_j)_{h'}\Bigr]\Phi_{ij}^\top
 + \sum_h\Bigl[\sum_j \beta_{ij}^h (\Phi_{ij}p_j^h)(\Phi_{ij}p_j^h)^\top - q_i^h q_i^{h\top}\Bigr],
\]
\[
\mathrm{cross}_i = \sum_h \beta_{ii}^h\, g_i F\bigl(I + A_i\Phi_{ii}B_i^\top\bigr)\Pi_h C_i ,
\]
where `F(P^ + sum_h R_h Sigma~ R_h^T)F^T = sum_{h,g} O'_h P_i^{hg} O'^T_g` is exactly the existing lifted
term, `F A_i = O' T A_i =: A^_i`, and `F X_i^T = sum_{h'} O'_{h'} Xi_i^{h'T}` when `X_i = sum_{h'} Xi_i^{h'} R_{h'}^T`
(r x d blocks), so the correction is `A^_i X^_i^T + X^_i A^_i^T + A^_i Y_i A^_i^T` with `X^_i = F X_i^T`.

The shift trap: the existing code subtracts token 0's reference mean before the spread, which is valid
only because the spread is invariant under a common shift of the values. With edge-dependent maps the
shift-invariant parts are `Sigma~` and the cross covariance between `v` and `Phi p`; `p_j^h` itself MUST
be computed from the unshifted `v_j^h` (the shifted `v` may be used in `Sigma~` and in the `v` slot of the
cross term). A test must catch a violation (a nonzero token-0 mean with nonzero edge factors).

Both covariance bodies must be extended: the dense body (full family or dense frames) and the blockwise
body (diagonal beliefs on block-diagonal frames), which needs only the diagonal reference blocks
`[C^att]_ll`: add `A^_i^{(l)} X^_i^{(l)T} + X^_i^{(l)} A^_i^{(l)T} + A^_i^{(l)} Y_i A^_i^{(l)T}` and the residual
correction block. In `"pure_gauge"` the outer `g_i = I + A phi_1(B_i^T A) B_i^T` is full K x K; for the
blockwise output use `(g Inner g^T)_{ll} = Inner_ll + A^{(l)} phi_i (B_i^T Inner)_{:,l} + transpose +
A^{(l)} phi_i (B_i^T Inner B_i) phi_i^T A^{(l)T}` with `B_i^T Inner` formed through `B_i^T O'` (r x K) without a
dense `Inner`, or route `"pure_gauge"` to the dense body if that is simpler; state the choice and its cost.
The dense body is the oracle for the blockwise body in the tests (as `test_blockwise_moment_matched_20261004.py` does).

### 2.4 Scope and refusals

Edge links require `content_link_chart="exponential"`, `attention_mode="projected_predictive"`
(predictive_content readout) and equal heads (one head run); refuse mixed head sizes (`so_n` towers)
and `projected_gradient` with clear `ValueError`s in `VFE3Config.__post_init__` validation and in the
`RelationalOperators` constructor. They work with dense and block-diagonal frames, diagonal and full
families, residual on or off, `predictive_covariance` passthrough or moment_matched, and
`predictive_covariance_checkpoint` (values and gradients bit-identical with and without it).

Audit every other consumer of the content maps (`grep` for `predictive_content(`, `content_factor_runs(`,
`content_factors(`, `_ambient_content_maps(`, `chart_receive`): `vfe3/model/model.py` diagnostics,
`vfe3/posthoc/*` (maps, interventions, routing, covariance), `vfe3/viz/extract.py`, `vfe3/gradients/full_link.py`.
A consumer that would silently ignore the edge links on an edge-enabled model must either be made
edge-aware or raise a clear `NotImplementedError`/`ValueError`; never return wrong numbers silently.

### 2.5 Wiring

* `vfe3/config.py`: new fields (place them right after `content_link_chart`, each with a comment in the
  existing style):
  `content_link_edges: str = "off"` (`"off" | "lowrank" | "pure_gauge"`),
  `content_link_edge_rank: int = 4`,
  `content_link_latent: bool = False`, `content_link_latent_steps: int = 1`,
  `content_link_latent_sigma: float = 0.1`, `content_link_latent_damping: float = 1.0`,
  `content_link_loop_weight: float = 0.0`, `content_link_loop_window: Optional[int] = 1`
  (the Level 2 fields are specified in Section 3; Level 1 may add them as validated-but-unused only if
  Level 2 is implemented in the same pass, otherwise add them in Level 2). Validation as in 2.4 plus:
  rank is an int with `1 <= r <= embed_dim`; `inactive_fields()` reports the rank when edges are off.
  Add every new field to `tests/profiles.py` (`_CONFIG_VALUES`) at its default.
* `vfe3/model/model.py`: pass the new settings to every layer's `RelationalOperators`.
* `vfe3/train.py::build_optimizer`: the new parameters join the existing relational content group
  (`m_relational_content_lr`, weight decay 0, aux role `relational_content`); the
  ungrouped-parameter guard must stay green. `_parameter_group`: names starting with `edge_` under
  `relational_operators` map to `relational_content`. `vfe3/run_resources.py`: the same mapping into
  `context_processing / relational_content`. Keep checkpoint compatibility: no new parameter or buffer
  exists when the edges are off.
* Diagnostics (`model.py` per-layer metrics and the relational capture): when edges are on, record the
  beta-weighted mean and the maximum over causal edges of `||A_i Phi_ij B_j^T||_F` (the edge part of
  the link, `E_ij - I`), and `||B||` statistics; keep them cheap. Map snapshots (`vfe3/posthoc/maps.py`)
  should save the new parameters if they save T and O.
* Put the edge algebra in a new module `vfe3/model/edge_links.py` (pure functions with docstrings and
  shape comments) and keep `RelationalOperators` changes to construction, parameter storage and calls.

### 2.6 Tests (new file `tests/test_edge_links_20261005.py`)

1. Reduction: with `W_B = 0` both modes equal the exponential chart for the mean and for the
   moment-matched covariance (dense and blockwise bodies, diagonal and full families, block and dense
   frames, residual on and off) to 1e-12 in float64.
2. Brute-force reference (float64, K <= 6, N <= 4, H = 2, r in {1, 2}): build every `G_ij` explicitly
   with `torch.linalg.matrix_exp` (and the pure-gauge `g_i e^L g_j^{-1}`), compute
   `m~_i = sum_j G_ij Pi(beta_ij) xi_j` and the generic covariance with explicit loops over j, h, h'
   (source term with c^{hh'}, spread with the UNshifted `n_ij^h = G_ij Pi_h xi_j`, residual cross), and
   compare with the implementation for random nonzero `W_A`, `W_B`, random causal routing, random
   frames, nonzero token-0 mean; tolerance 1e-10.
3. Blockwise equals dense with edges on.
4. Passive gauge covariance: `mu -> g mu`, `Sigma -> g Sigma g^T`, `U -> g U` maps outputs covariantly.
5. Holonomy: pure-gauge triangle holonomies `G_ik^{-1} G_ij G_jk` all have the eigenvalues of `e^L`;
   low-rank ones generically do not.
6. Causality: changing `mu_j` for `j > i` leaves outputs at i unchanged (with large factor norms, to
   exercise the masking before `phi_1`).
7. Finite-difference gradients (float64, central differences) of a scalar of the outputs with respect
   to `W_A`, `W_B`, `A`, `mu`, `sigma`.
8. Config: validation errors and `inactive_fields`; optimizer grouping (new parameters present, guard
   green); `_parameter_group` and resource accounting; state-dict round trip; a default-config model has
   no new state-dict keys.
9. Model smoke (tiny `tests/profiles.tiny_config`-style config, CPU): a few training steps with each
   mode give finite loss and nonzero gradients on `W_B` after the first step.

## 3. Level 2: latent receiver factors (per-context MAP links)

### 3.1 Model

Supplement S6 promotes the edge generators to latent variables with a normalized prior and estimates
them by MAP inside the variational free energy. Two facts shape the causal-LM version:

* Causality: a latent variable may only be inferred from information available at its receiver. A
  sender factor `B_j` is shared by all receivers `i >= j`, so inferring it would leak later positions
  into earlier ones. The latent variable is therefore the RECEIVER factor `A_i` of `"lowrank"` (it
  enters only receiver i's links); `B_j` stays amortized. Each receiver's problem is independent and
  causal. Latent inference is defined for `content_link_edges="lowrank"` only.
* The receiver's next-layer state is not observed in the forward pass. Restricted to one layer
  transition (the filtering approximation a causal forward pass imposes: later layers and the
  observation are not yet available), the optimal recognition law of `zeta_i^{l+1}` in S6's MAP free
  energy `F(L, q)` is `N(r xi_i + m~_i, Q)`; profiling it out leaves the expected transport energy,
  which is exactly half the Q^{-1}-weighted trace of the moment-matched predictive covariance of
  Section 2.3 without the floor (the variance of the kernel with the residual term included). So the
  per-context link problem minimizes the predictive uncertainty of the receiver (agreement of the
  transported sources plus transported source uncertainty) against the prior; the current of
  `prop:lie_field_equation` with the profiled `zeta^{l+1}` is the gradient of this energy. Without the
  prior this energy is minimized by contracting links (it is a variance), which is why the anchor is
  required (S6 makes the same point for its unregularized problem).

With `Q = c_i I`, `c_i = tr(C_i)/K` (the receiver's mean reference variance, frame invariant), and the
prior `A_i ~ N(A_i^0, sigma_L^2 I)` around the amortized factor `A_i^0 = a(xi^_i)` (the S6 anchor moved
from the shared generator to the amortized edge field; `sigma_L -> 0` is the static limit, Remark
`rem:lie_static_limit`), the per-receiver MAP energy is
\[
\mathcal F_i(A_i) = \frac{1}{2c_i}\operatorname{tr} C_i^{\mathrm{att}}(A_i)\Big|_{\text{no floor}}
 + \frac{1}{2\sigma_L^2}\|A_i - A_i^0\|_F^2 + \lambda_W \sum_{(i,j,k)\in\mathcal W_i} w_{ijk}\,\Psi(W_{ijk}) .
\]

### 3.2 Inference

`T = content_link_latent_steps` fixed-point steps of the stationarity condition
`A_i = A_i^0 - sigma_L^2 grad_{A_i}[E_i + lambda_W Loop_i]` (the natural-gradient step in the Fisher
metric of the prior, unit step), damped by `rho = content_link_latent_damping`:
\[
A_i^{(t)} = (1-\rho)A_i^{(t-1)} + \rho\bigl(A_i^0 - \sigma_L^2\nabla_{A_i}[E_i + \lambda_W \mathrm{Loop}_i](A_i^{(t-1)})\bigr),\qquad A_i^{(0)} = A_i^0 .
\]
The gradient is computed with `torch.autograd.grad` of the summed per-receiver energies (they decouple
across receivers), with `create_graph=True` while training so the outer parameters (comparison and
content maps, `W_A`, `W_B`, frames, tables) train through the unrolled steps; in eval or under
`torch.no_grad` use `torch.enable_grad()` on a detached `A^{(0)}` and `create_graph=False`. The final
message and covariance use `A^{(T)}`. The energy is computed whatever `predictive_covariance` is (it is
part of the link inference), but only its trace is needed: compute the reference-coordinate trace
directly from the Section 2.3 terms (`tr(g F M F^T g^T)` with `g = I` here) rather than materializing the
local covariance; the blockwise diagonal reference blocks suffice for diagonal beliefs on block frames.

### 3.3 Loop prior (S6 Wilson-type term, optional)

Loops `W_{ijk}^l = (G_{ik}^{l+1} G_{kk}^l)^{-1} G_{ij}^{l+1} G_{jk}^l` for `k < j <= i` (time-like when
`j = i`), with `G^l_{ab} = e^{L_l} E^l_{ab}`. The shared generators cancel at layer l+1:
\[
W_{ijk}^l = (E_{kk}^l)^{-1} e^{-L_l} (E_{ik}^{l+1})^{-1} E_{ij}^{l+1} e^{L_l} E_{jk}^l ,
\]
so every loop is the identity for the static shared-generator model. At layer l+1, receiver i's loops
depend on its own latent `A_i^{l+1}` (through `E_ik^{l+1}`, `E_ij^{l+1}`) and on layer l's already inferred
links (fixed): causal and per receiver. `Psi(W) = 1/2 ||W - I||_F^2` (vanishes exactly at `W = I`; see
`prop:lie_blind`). Weights `w_ijk = pibar_ij pibar_jk` from the head-averaged routing PRIOR
(`softmax(log_prior)` rows, causal-uniform when no prior is given). The window
`content_link_loop_window = w` keeps loops with `i - w <= k < j <= i` (w = 1: the time-like loop with
k = i - 1 only; `None`: every loop, O(N^3 K^3) per layer, an extreme opt-in path to document).
Layer 0 has no loops. Evaluate loops in float64 (`e^{-L}` can be very large for trained generators; say
so in the docstring). Thread the previous layer's link state (its `F`, `e^{-L}`, inferred `A`, `B`) from
layer l to layer l+1 through an optional `link_state` dict carried by `vfe_stack -> vfe_block -> e_step ->
relational_predictive_readout` (or an equivalent explicit mechanism); with the loop weight at 0 nothing
is threaded and nothing changes. Loops require `content_link_latent=True`.

### 3.4 Config (validation)

`content_link_latent` requires `content_link_edges="lowrank"`; `content_link_latent_steps` int >= 1;
`content_link_latent_sigma` finite > 0; `content_link_latent_damping` in (0, 1];
`content_link_loop_weight` finite >= 0 and > 0 only with latent on; `content_link_loop_window` None or
int >= 1. `inactive_fields()` reports the latent fields when latent is off and the window when the loop
weight is 0.

### 3.5 Diagnostics

When latent is on, record per layer: the energy at `A^0` and at `A^{(T)}`, the loop energy, and
`||A^{(T)} - A^0||_F / ||A^0||_F` (means over receivers).

### 3.6 Tests (new file `tests/test_latent_links_20261005.py`)

1. Static limit: `sigma_L -> 0` (e.g. 1e-12) reproduces Level 1 to roundoff; `rho` irrelevant there.
2. Descent: for a small `sigma_L` one step lowers `F_i` for every receiver (float64, tiny dims).
3. Fixed point: with many steps and a contracting `sigma_L`, the iterate satisfies the stationarity
   condition to tolerance; compare `grad` with a finite-difference gradient of `F_i`.
4. Causality: future perturbations leave receiver i's inferred `A_i` and outputs unchanged.
5. Gauge covariance of the outputs with latent inference on.
6. Outer gradients: training-mode backward reaches `W_A`, `W_B`, `T`, `O`, `raw_score_q`; eval under
   `torch.no_grad` runs and matches training-mode forward values.
7. Loops: brute-force `W_ijk` from explicit matrices equals the implementation; all loops are identity
   when every edge part vanishes; window semantics; layer 0 has none; `lambda_W = 0` reproduces the
   loop-free latent model exactly.
8. Config validation and `inactive_fields`; a tiny model trains a few CPU steps with latent and loops on.

## 4. Costs to report

For the user's launcher shape (K = 192, H = 6, d = 32, N = 256, batch 8, 6 layers, r = 4) report the
dominant per-layer costs and memory of: the per-edge `phi_1` (N^2 matrices of size 2r), the edge
covariance terms, the latent inner step (with double backward), and the loop terms per window size.
Do not run them on the GPU; estimate from the formulas and, if useful, time a CPU instance at reduced N.
