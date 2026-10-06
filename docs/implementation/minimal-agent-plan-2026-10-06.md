# Minimal-agent attention readout and prototype-agent MLP block: implementation plan (2026-10-06)

Binding specification for the implementation, reference-test, review and integration agents. Author of the
specification: Claude Opus 5.5 (the orchestrating session). Source theory:
`C:/Users/chris and christine/Desktop/Research/manuscripts/Attention/minimal_agent_model.tex` (read Sections 2 to 4:
Definition `ma:def:model`, Proposition `ma:prop:minimal` items (a) to (e), Proposition `ma:prop:mean_field`, Eq.
`ma:eq:derived_output`). The predictive-kernel readout it replaces is `GL(K)_attention_v2.tex` Theorem
`thm:probabilistic_status` item 2 as implemented in `vfe3/model/relational_operators.py`.

## 0. Ground rules

* Work ONLY in the git worktree you are assigned (Section 4). Never edit `C:/Users/chris and christine/Desktop/VFE_3.0`
  itself (the canonical checkout carries the user's uncommitted work) and never edit `../V3_Transformer`. Do not commit,
  branch, stash, reset, rebase or push. Leave every change uncommitted in your worktree.
* No GPU workloads. CPU tests only, tiny models (single-digit or low-double-digit K and N, one or two layers), one
  thread per process. Torch runs with `C:/anaconda/python.exe`; `pytest-xdist` is not installed. Command template
  (Git Bash), run from YOUR worktree root so that its own `vfe3` is imported (the package is not installed):
  `cd "<worktree>" && VFE3_TEST_DEVICE=cpu CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "C:/anaconda/python.exe" -m pytest <files> -q -p no:cacheprovider`.
  Do not install packages or modify the Python environment. No Lean.
* Conventions (user global rules): type hints on every signature; docstrings with LaTeX for every nontrivial formula
  (`r"""..."""`); variable names in paper notation (`xi`, `C`, `S_Q`, `T`, `G`, `w`, `Lambda_bar`, `nu`, `lambda_`);
  tensor shape comments at critical points; American spelling; none of the words key insight, crucially, critically,
  notably, importantly, worth noting, fundamentally, leverages, underscores in docs or comments. Match the style of
  `vfe3/model/relational_operators.py` and `vfe3/model/block_mlp.py`.
* Every new behavior is opt-in. With every new config field at its default, the model, its state-dict keys, its numerics
  and every existing test are unchanged. A default-config model has no new parameter or buffer.
* Keep checkpoint parameter names and the `vfe3` import package. Never change an existing default value in
  `vfe3/config.py`; you may add fields, add validation and add entries to `inactive_fields()`.
* A consumer of the content maps or comparison maps that would silently ignore the new modes must either be made aware of
  them or raise a clear `ValueError`/`NotImplementedError`. Never return wrong numbers silently.
* Small-network tests validate the implementation against exact results; they do not set research direction. Report
  what was executed, with counts, and say plainly what was not run.

## 1. Notation shared by both features

Reference moments of token `i` in frame `U_i`: `xi_i = U_i^{-1} mu_i`, `C_i = U_i^{-1} Sigma_i U_i^{-T}` (C_i is
symmetric positive definite). Heads: `H` heads of equal width `d`, `K = H d`. The record model of the note: a receiver with
state `X_i ~ N(xi_i, C_i)`, sources with states `X_j ~ N(xi_j, C_j)` (snapshots of the input beliefs, the self edge
included), head-`h` label `J^h` with prior `pi^h_ij`, record `o_h = S_h X_i - T_h X_{J^h} + eps_h`, `eps_h ~ N(0, N_h)`,
observed at `o_h = 0`. `log N(x; m, W)` is the Gaussian log density.

For one head (index suppressed), a receiver with query mean `a_i = S xi_i`, query covariance `A_i = S C_i S^T`, sources
with noisy-utterance means `b_j = T xi_j` and covariances `B_j = T C_j T^T + N`, the record covariance is
`W_ij = A_i + B_j` (positive definite because `N` is), and the EXACT posterior (Proposition ma:prop:minimal (a)) is

```
w_ij       = softmax_j( log pi_ij + log N(b_j; a_i, W_ij) )                    # exact source posterior, temperature 1
r_ij       = W_ij^{-1} (b_j - a_i)                                              # (r,)
u_i        = sum_j w_ij r_ij                                                    # (r,)
Lambda_bar = sum_j w_ij W_ij^{-1}                                               # (r, r)
R_i        = sum_j w_ij r_ij r_ij^T - u_i u_i^T                                 # (r, r), between-source spread
Kmat_i     = C_i S^T                                                            # (K, r) gain basis
xi_i^+     = xi_i + Kmat_i u_i
C_i^+      = C_i - Kmat_i (Lambda_bar_i - R_i) Kmat_i^T
```

These equal `xi^+ = sum_j w_ij m_ij`, `C^+ = sum_j w_ij [P_ij + (m_ij - xi^+)(m_ij - xi^+)^T]` with
`m_ij = xi_i + G_ij (b_j - a_i)`, `G_ij = C_i S^T W_ij^{-1}`, `P_ij = C_i - G_ij S C_i`; `C_i^+` is a mixture covariance,
hence positive semidefinite (positive definite when every `P_ij` is). Verify this equivalence in a test.

The MEAN-FIELD step (Proposition ma:prop:mean_field) uses the label update at the receiver's prior and the power
posterior at fixed labels:

```
beta_ij  = softmax_j( log pi_ij - KL( N(a_i, A_i) || N(b_j, B_j) ) )           # temperature 1
Lambda_i = sum_j beta_ij B_j^{-1}                                               # (r, r)
eta_i    = sum_j beta_ij B_j^{-1} b_j                                           # (r,)
```

and, summed over heads, `P_i^{-1} = C_i^{-1} + sum_h S_h^T Lambda_i^h S_h`, `xi_i^+ = P_i (C_i^{-1} xi_i + sum_h S_h^T eta_i^h)`,
`C_i^+ = P_i` (the mean-field covariance omits the between-source spread; say so in docstrings). The KL is separable
(per-receiver and per-sender factors, no per-pair inverse); the `log det A_i` term is a row constant and may be dropped.

## 2. Feature A: the minimal-agent attention readout

### 2.1 Model and configuration

A new readout for `attention_mode="projected_predictive"`, selected by a new field

```
predictive_readout:          str   = "kernel"        # "kernel" (predictive-kernel content, existing) | "minimal_agent"
minimal_agent_inference:     str   = "exact"         # "exact" (exact single-head posterior, heads by ADF) | "mean_field"
minimal_agent_channel_noise: float = 1.0             # initial isotropic channel noise n_h of every head (learned), > 0
```

placed directly after `relational_routing` in `vfe3/config.py`, each with a comment in the existing style. Under
`"minimal_agent"` every head `h` has a query map `S_h` (the existing `raw_score_q[h]`, `(d, K)`), a sending map `T_h` (the
existing `T[h]`, `(d, K)`) that is ALSO the key map (tied key and content, `S_K = T`), and a learned isotropic channel noise
`N_h = n_h I_d`, `n_h = exp(channel_log_noise[h])`, stored as a new parameter `channel_log_noise` of shape `(H,)` initialized
to `log(minimal_agent_channel_noise)`. The parameters `raw_score_k` and `O` are NOT created in this mode (the derived
receiver replaces the receiving map; the key is tied). Initialization: `T` keeps its block-projector initialization and
`raw_score_q` is initialized EQUAL to `T`'s initial block projector (S_Q = T at step 0, so the record compares like
coordinates); state why in a comment. The routing temperature is 1 and no `kl_max` clamp is applied (the exact posterior;
softmax is evaluated with log-sum-exp). The routing log-prior (`log_prior`, causal mask, ALiBi or other registered prior) is
the label prior `pi`.

Validation (`VFE3Config.__post_init__` and the `RelationalOperators` constructor, clear `ValueError`s):
`predictive_readout="minimal_agent"` requires `attention_mode="projected_predictive"`, equal heads (one head run; refuse
mixed `so_n` towers), `content_link_edges="off"`, and NOT (`pos_rotation="rope"` with `rope_on_value=False`), because the
tied key is the content and both must be read in one frame. `minimal_agent_inference` in {"exact", "mean_field"};
`minimal_agent_channel_noise` finite and > 0. `inactive_fields()` reports, under `"minimal_agent"`:
`relational_routing`, `content_link_chart`, `attention_residual` (the residual is derived and precision weighted),
`relational_output_init`, and `kappa_beta` ONLY IF nothing else in the projected-predictive route reads it (check);
and it reports `minimal_agent_inference` and `minimal_agent_channel_noise` when `predictive_readout="kernel"`.
`predictive_covariance` keeps its meaning: `"passthrough"` returns the incoming covariance unchanged (the mean still uses
`C_i` in its gain), `"moment_matched"` returns the posterior covariance plus the existing covariant floor
`Q~_i = predictive_covariance_floor * (tr C_i / K) I` (tr of the INPUT `C_i`), mapped to local coordinates. Under
`predictive_covariance_checkpoint=True` the whole minimal-agent readout of a layer is recomputed in backward
(non-reentrant checkpoint, no random numbers), with values and gradients bit-identical to the uncheckpointed call.

### 2.2 Heads: exact inference by assumed-density filtering (ADF)

The exact joint posterior over all heads' labels costs `N^H` components. The implementation declares that each head hears
an independent snapshot of the sender's utterance state (the per-head snapshot convention, an extension of the note's
self-edge snapshot; it coincides with the note's model when `T_h C_j T_g^T = 0`) and conditions the receiver on the heads in
index order `h = 1..H` by assumed-density filtering: head `h` applies the exact single-head posterior of Section 1 to the
Gaussian `(xi_i^{(h-1)}, C_i^{(h-1)})` left by heads `1..h-1` and collapses the result to its first two moments:

```
a_i^h = S_h xi_i^{(h-1)},   A_i^h = S_h C_i^{(h-1)} S_h^T,   b_j^h = T_h xi_j,   B_j^h = T_h C_j T_h^T + n_h I   (input C_j)
w^h, u^h, Lambda_bar^h, R^h   from Section 1 with (a^h, A^h, b^h, B^h, log pi^h)
Kmat_i^h = C_i^{(h-1)} S_h^T
xi_i^{(h)} = xi_i^{(h-1)} + Kmat_i^h u_i^h,    C_i^{(h)} = C_i^{(h-1)} - Kmat_i^h M_i^h Kmat_i^{hT},   M_i^h = Lambda_bar_i^h - R_i^h
```

with `(xi^{(0)}, C^{(0)}) = (xi_i, C_i)` and outputs `xi^+ = xi^{(H)}`, `C^+ = C^{(H)}`. Never materialize `C^{(h-1)}` densely
when the input is blockwise: keep `C^{(h-1)} = C - sum_{g<h} Kmat^g M^g Kmat^{gT}`, so
`Kmat^h = C S_h^T - sum_{g<h} Kmat^g M^g (S_g C^{(g-1)} S_h^T)` (form the needed `(d, d)` cross blocks), and
`A^h = S_h C S_h^T - sum_{g<h} (S_h Kmat^g) M^g (S_h Kmat^g)^T`. Prove in the docstring (short argument, as in the plan
below) and test that ADF equals Proposition ma:prop:minimal (b) whenever `S_h C_i S_g^T = 0` for `h != g`: then
`S_g C^{(h-1)} = S_g C`, so every head sees the input moments on its own subspace and the label posteriors factor. When the
condition fails ADF is a declared, order-dependent approximation that always returns a valid Gaussian (each step is an
exact mixture moment match). Local outputs: `mu_i^+ = U_i xi_i^+`; for a full local covariance
`Sigma_i^+ = Sigma_i - sum_h Z_i^h M_i^h Z_i^{hT} + floor (tr C_i / K) U_i U_i^T` with `Z_i^h = U_i Kmat_i^h` (K, d), and for a
diagonal local covariance its diagonal, `diag(Sigma_i) - sum_h rowquad(Z_i^h, M_i^h) + floor (tr C_i/K) diag(U_i U_i^T)`
(`rowquad(Z, M)_k = Z_k M Z_k^T`), which never forms a `K x K` matrix. (`Sigma_i = U_i C_i U_i^T` exactly, so it need not
be recomputed.)

The mean-field variant (`minimal_agent_inference="mean_field"`) uses Section 1's mean-field step for all heads at once,
with every head's label update evaluated at the INPUT receiver moments (the first structured-mean-field step) and the
information-form fusion `P_i^{-1} = C_i^{-1} + sum_h S_h^T Lambda_i^h S_h`. Reuse
`vfe3.relational_scoring.projected_gaussian_kl` (with the noisy key covariance `B_j^h`) for the KL if its contract fits.

### 2.3 Per-pair exact routing: cost and memory

`W_ij^h` differs for every pair, so the exact path needs one `d x d` Cholesky factor per `(b, h, i, j)`. Follow
`vfe3.relational_scoring.projected_noisy_query_cost` and `_noisy_query_tile`: zero jitter, no covariance repair, bounded
pair tiles (about 32 MB of pair matrices per tile), non-reentrant `torch.utils.checkpoint` of each tile when gradients are
on and the grid has more than one tile, autocast disabled, compute dtype from `vfe3.relational_scoring._compute_dtype`
(float64 under an escalated frame), one host check for factorization failure (raise `ValueError`, never repair). The
softmax needs every `j` before the weighted sums, so either run two passes over the tiles (pass 1: costs, then softmax;
pass 2: `u`, `Lambda_bar`, `R` accumulated tile by tile) or one pass with an online log-sum-exp accumulation; state which.
Masked pairs (prior `-inf`) get weight exactly 0 and must not produce non-finite values (their `W_ij` is still positive
definite). Put this kernel in a new module `vfe3/model/minimal_agent.py` as pure functions with docstrings and shape
comments, together with the ADF and mean-field assemblies; keep `RelationalOperators` changes to construction, parameter
storage and calls. Suggested API (adapt names if needed, but keep them documented):

```
exact_record_posterior(a, A, b, B, log_prior) -> RecordPosterior(weights, cost, u, Lambda_bar, R)
    a (..., I, r), A (..., I, r, r), b (..., J, r), B (..., J, r, r), log_prior broadcastable to (..., I, J) or None
mean_field_record_posterior(a, A, b, B, log_prior) -> (weights, cost, Lambda, eta)
minimal_agent_readout(xi, C, S, T, log_noise, log_prior, inference, ...) -> MinimalAgentResult
```

### 2.4 Wiring

* `vfe3/config.py`: the three fields of 2.1, validation, `inactive_fields()`. Add every new field to
  `tests/profiles.py` (`_CONFIG_VALUES`) at its default.
* `vfe3/model/relational_operators.py`: constructor arguments `predictive_readout`, `minimal_agent_inference`,
  `minimal_agent_channel_noise`; conditional parameter creation (2.1); `comparison_runs()` returns `(S_q, T)` under the
  minimal agent (tied key) so that score-reading diagnostics stay correct; `content_factors()`/`content_factor_runs()` and
  every other O-reading method raise a clear error in this mode; a method `minimal_agent_update(mu, sigma, frame, log_prior,
  *, return_covariance, capture)` returning local `(mu_plus, sigma_plus or None)` in the input layout.
* `vfe3/gradients/relational.py::relational_predictive_readout`: when `operators.predictive_readout == "minimal_agent"`,
  skip `relational_score`/`attention_weights`/`predictive_content` and call the minimal-agent update with the ROUTING frame
  for query, key, content and write-back (refuse a distinct `value_frame` object whose matrices differ, as 2.1 requires).
  The result is the posterior mean (no extra residual is added: the residual is the precision-weighted
  `(I - G S) xi_i`). Capture: `beta_pre` (the per-head weights actually used, `(..., H, N, N)`), `score_energy_pre` (the
  per-pair costs), `relational_mean_displacement_norm`, and the new diagnostics of 2.5.
* `vfe3/model/model.py`: pass the new settings to every layer's `RelationalOperators`; per-layer metrics for 2.5.
* `vfe3/train.py::build_optimizer` and `_parameter_group`, `vfe3/run_resources.py`: `channel_log_noise` joins the
  relational SCORE group (`m_relational_score_lr`, weight decay 0, aux role `relational_score`); the ungrouped-parameter
  guard stays green; resource accounting counts the mode's actual parameters (score `H d K + H`, content `H d K`).
* Audit every consumer of `raw_score_k`, `O`, `content_factors(`, `content_factor_runs(`, `_ambient_content_maps(`,
  `chart_receive`, `comparison_runs(` (grep): `vfe3/model/model.py`, `vfe3/posthoc/*`, `vfe3/viz/extract.py`,
  `vfe3/gradients/full_link.py`, `vfe3/run_artifacts.py`, `vfe3/metrics.py`. Make each aware or make it refuse clearly.

### 2.5 Diagnostics (captured only when a capture dict is requested)

Per layer and head, means over receivers: the gain fraction `tr(A_i^h Lambda_bar_i^h) / d` in [0, 1] (how much of the
probed subspace the head overwrites; mean-field: `tr(A_i^h Lambda_i^h)/d` with its own definition documented), the channel
noise `n_h`, and the head coupling `max_{h != g} ||S_h C_i S_g^T||_F / sqrt(||S_h C_i S_h^T||_F ||S_g C_i S_g^T||_F)` at the
input moments (0 means ADF is exact). Name them `relational_minimal_agent_gain_fraction`,
`relational_minimal_agent_channel_noise`, `relational_minimal_agent_head_coupling` and wire them like the edge diagnostics.

### 2.6 Tests (new file `tests/test_minimal_agent_attention_20261006.py`)

1. Single head, float64, K <= 6, N <= 4: the implementation equals an explicit per-pair Gaussian conditioning (component
   means `m_ij`, covariances `P_ij`, weights from explicit densities, mixture moments), random frames, random `S`, `T`,
   noise, causal prior, nonzero means; tolerance 1e-10. The `(u, Lambda_bar, R)` form equals the mixture form.
2. Two heads with `S_1 C_i S_2^T = 0` built in (block-local maps on block-diagonal `C`): ADF equals the exact posterior by
   enumerating all `N^2` label pairs of the per-head-snapshot model and moment-matching the joint mixture.
3. Two heads without the condition: ADF equals an explicit sequential reference (head 1 then head 2 with full conditioning
   and collapse); report the difference from the enumerated exact posterior (it need not vanish).
4. Limits: `n -> 0` with point-mass senders (`C_j -> 0` for sources) gives `G -> R_Q = C S^T (S C S^T)^{-1}` and the
   minimal-change projection of item (c); full rank (`d = K`, invertible `S`, `T`) gives item (d)'s weights and means.
5. Mean-field variant equals an explicit power posterior with KL labels. Separately, for one head with equal `B_j`, the
   power posterior evaluated at the EXACT weights `w` (not the KL labels, which differ) has the exact mean and the exact
   covariance minus the spread (Proposition ma:prop:mean_field (iii)); test that identity on the assembly function.
6. Passive gauge covariance: `mu -> g mu`, `Sigma -> g Sigma g^T`, `U -> g U` maps the outputs covariantly.
7. Causality: changing `mu_j`, `Sigma_j` for `j > i` leaves outputs at `i` unchanged.
8. Blockwise (diagonal family on block frames) equals dense (full family or dense frames) on the same beliefs.
9. Finite-difference gradients (float64, central differences) of a scalar of the outputs with respect to `S`, `T`,
   `channel_log_noise`, `mu`, `sigma`; a tile-boundary case (force tiny tiles) gives identical values and gradients.
10. Config: validation errors, `inactive_fields`, optimizer grouping (new parameter present, guard green), resource
    accounting, state-dict round trip, a default-config model has no new state-dict keys and `minimal_agent` models have no
    `raw_score_k`/`O`.
11. Model smoke (tiny `tests/profiles` config, CPU): a few training steps in each inference mode with
    `predictive_covariance` in {"passthrough", "moment_matched"} give finite loss and nonzero gradients on `raw_score_q`,
    `T` and `channel_log_noise`; `predictive_covariance_checkpoint` gives identical values and gradients.

## 3. Feature B: the prototype-agent block MLP

### 3.1 Model

The block MLP becomes the minimal agent model with a static library of `M = block_mlp_expansion * K` prototype agents:
template `a` has belief `p_a = N(nu_a, Lambda_a)`, `Lambda_a = diag(lambda_a)`, `lambda_a = exp(log_lambda_a) > 0`, and prior
weight `pi_a = softmax(log_pi)_a`; the token's working belief is `N(xi_i, C_i)` (see 3.3 for coordinates); query and sending
maps are the identity (full rank, `S = T = I`) and the channel noise is absorbed into `Lambda_a`. Then, with Section 1 at
`a_i = xi_i`, `A_i = C_i`, `b_a = nu_a`, `B_a = Lambda_a`:

* exact (`block_mlp_inference="exact"`): `w_ia ∝ pi_a N(nu_a; xi_i, C_i + Lambda_a)`, `xi^+ = xi_i + C_i u_i`,
  `C^+ = C_i - C_i (Lambda_bar_i - R_i) C_i` (Proposition ma:prop:minimal (a), (d));
* mean-field (`block_mlp_inference="mean_field"`): `r_ia ∝ pi_a exp(-KL(N(xi_i, C_i) || p_a))`, which are exactly the
  responsibilities of the existing `GaussianTemplateBlockMLP.responsibilities` (test this), and the power posterior
  `P_i = (C_i^{-1} + diag(D_i))^{-1}`, `D_i = sum_a r_ia / lambda_a`, `xi^+ = P_i (C_i^{-1} xi_i + sum_a r_ia nu_a / lambda_a)`,
  `C^+ = P_i` (no spread).

The update is a precision-weighted combination of the token's own mean and the template means: the values are tied to the
template means (there is no free value matrix), and the residual is derived. Document in the class docstring that the
mean-field form is the first structured-mean-field step of the record model started at the token's prior (Proposition
ma:prop:mean_field), that `gaussian_template`'s responsibilities are its label step, and that the update is a
precision-weighted modern-Hopfield-type retrieval with tied keys and values. Dropout (the block's `dropout`) applies to the
mean increment `xi^+ - xi` only.

### 3.2 Configuration

* `block_mlp_mode`: add `"prototype_agents"` to the registry (`vfe3/model/block_mlp.py::BLOCK_MLP_REGISTRATIONS`) with
  `covariance_kinds = {"passthrough", "posterior"}`, the correct `parameter_count_per_layer` (`2 e K^2 + e K`: `nu`,
  `log_lambda`, `log_pi`), `matmul_flops_per_layer` for the mean-field path (about `10 e K^2`), and a report label.
* `block_mlp_covariance`: the new value `"posterior"` (valid only for `prototype_agents`, through the registry's
  `covariance_kinds`) returns `C^+` mapped to local coordinates; `"passthrough"` keeps the incoming covariance.
* New fields after `block_mlp_expansion`:
  `block_mlp_inference: str = "mean_field"` ("mean_field" | "exact"; read only by `prototype_agents`) and
  `block_mlp_template_init_variance: float = 10.0` (initial `lambda_a` of `prototype_agents`; finite, > 0). At
  `lambda = 10` and token variances near 1 the initial pull toward the templates is about 9 percent per layer; say so.
* Initialization: `nu` uniform in `(-K^{-1/2}, K^{-1/2})` from the module's existing initialization stream (mirror
  `GaussianTemplateBlockMLP`), `log_lambda = log(block_mlp_template_init_variance)`, `log_pi = 0`.
* `hidden_bias` (`per_type_hidden_bias`) is rejected (no hidden layer), as for `gaussian_template`.
* `inactive_fields()` reports `block_mlp_inference` and `block_mlp_template_init_variance` unless the mode is
  `prototype_agents`, and `block_mlp_activation` under `prototype_agents`. Add the new fields to `tests/profiles.py`.
* Optimizer: the parameters stay in the block-MLP group; follow `gaussian_template`'s weight-decay policy and state it.

### 3.3 Coordinates

* `block_coordinates="local"`: the block reads the local `(mu_i, Sigma_i)` (diagonal `(..., K)` or full `(..., K, K)`),
  like the coordinate MLP (`gauge_class` coordinate, nonintertwiner).
* `block_coordinates="reference"`: extend `vfe3/model/block.py::_reference_coordinate_block` so that for
  `prototype_agents` it passes the reference covariance `C_i` (blocks `(..., H, d, d)` for diagonal beliefs on block frames,
  else dense `(..., K, K)`; add a helper in `vfe3/model/frame_actions.py` or reuse the construction of
  `RelationalOperators._reference_covariance`) instead of only its diagonal, and, under `"posterior"`, maps the returned
  `C^+` back as `Sigma^+ = U C^+ U^T` (its diagonal for diagonal beliefs). This makes the block invariant under the passive
  action. `GaussianTemplateBlockMLP` and the other modes keep their current behavior exactly.
* The class therefore accepts `sigma` with `ndim == mu.ndim` (diagonal), `mu.ndim + 1` (dense), or `mu.ndim + 2`
  (blocks `(..., H, d, d)`), and returns the covariance in the same layout. Use block-batched linear algebra with block size
  1 (elementwise), `d`, or `K` for the three layouts; the template covariances are diagonal, so `C_i + Lambda_a` and
  `C_i^{-1} + diag(D_i)` keep the block structure. Note: the exact spread `C R C` couples blocks; for a diagonal input only
  `diag(R)` is needed, `diag(C R C)_k = c_k^2 R_kk`; for block or dense inputs the exact path forms the dense spread,
  which is `O(N M K^2)` per layer, an extreme opt-in path to document. Chunk the exact path over templates (online
  log-sum-exp accumulation or two passes) so that no `(..., N, M, K)` tensor larger than about 32 MB exists at once, with
  checkpointed chunks when gradients are on.

### 3.4 Tests (new file `tests/test_prototype_agents_mlp_20261006.py`)

1. Exact path, float64, small K and M: equals explicit Gaussian conditioning on each template and mixture moment matching,
   for diagonal, block and dense working covariances.
2. Mean-field path equals an explicit power posterior; its responsibilities equal `GaussianTemplateBlockMLP`'s for the same
   `(nu, log_lambda, log_pi)`.
3. One template (`M = 1`): exact and mean-field means agree (Proposition ma:prop:mean_field (iii) analog) and the exact
   covariance exceeds the mean-field one by the (zero) spread.
4. Broad templates (`lambda -> infinity`): the block tends to the identity; sharp templates (`lambda -> 0`, one template
   dominant): the mean tends to that template's mean.
5. Covariance contract: `"passthrough"` returns the input covariance object unchanged; `"posterior"` returns a positive
   definite covariance in the input layout.
6. Reference coordinates: passive gauge invariance (`mu -> g mu`, `Sigma -> g Sigma g^T`, `U -> g U` maps outputs
   covariantly); the block path through `_reference_coordinate_block` equals a hand-built `U F(U^{-1} .)` reference.
7. Finite-difference gradients (float64) with respect to `nu`, `log_lambda`, `log_pi`, `mu`, `sigma`; a chunk-boundary case
   gives identical values and gradients.
8. Config validation, `inactive_fields`, registry parameter count against the module, optimizer grouping, state-dict
   round trip; existing `gaussian_template` behavior and tests unchanged.
9. Model smoke (tiny config, CPU): a few training steps with `block_mlp_mode="prototype_agents"` in both inference modes
   and both covariance contracts, in local and reference coordinates, give finite loss and nonzero template gradients.

## 4. Worktrees and documentation

* Feature A works in `C:/Users/chris and christine/Desktop/VFE_3.0/.claude/worktrees/ma-attention-20261006`
  (branch `claude/minimal-agent-attention-20261006`), Feature B in
  `C:/Users/chris and christine/Desktop/VFE_3.0/.claude/worktrees/ma-prototypes-20261006`
  (branch `claude/prototype-agents-mlp-20261006`). Both start at `origin/main` (`1c70c87`). Feature B must not touch the
  relational files; Feature A must not touch `block_mlp.py`, `block.py` or `frame_actions.py`. Both touch `config.py`,
  `tests/profiles.py`, `train.py` and `run_resources.py`: keep those edits minimal and localized (new fields in the place
  this plan names) so the integration merge is mechanical.
* Each implementer records its change in a feature note in its worktree:
  `docs/implementation/minimal-agent-attention-results-2026-10-06.md` (A) or
  `docs/implementation/prototype-agents-results-2026-10-06.md` (B): flowing prose in the house style of
  `docs/2026-10-05.md`; what was added, the exact formulas, the config keys and defaults, the tests and their counts, costs,
  deviations from this plan with reasons, and what is not done. The integration agent writes the summary entry in
  `docs/2026-10-06.md` (append a new section at the end; keep the existing content).

## 5. Costs to report

For the user's K = 128 launcher shape (H = 4, d = 32, N = 128, batch 32, 4 layers, M = 8 K = 1024, float32) estimate from
the formulas, and time a CPU instance at reduced N (one thread, batch 1): per layer, the exact per-pair routing (pairs,
`d x d` factors, FLOPs, tile memory, passes), the ADF assembly, the mean-field variant, and for Feature B the mean-field and
exact paths in local diagonal coordinates and the exact path in reference coordinates. Compare with the existing kernel
readout (`relational_routing="noisy_query"` is the closest existing per-pair cost) and with the coordinate MLP. GPU cost is
not measured; say so.
