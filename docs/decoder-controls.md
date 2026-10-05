# Decoder controls

Nonlinear chunked decoders expose memory/compute and numerical-accuracy choices.

| Setting | Behavior |
|---|---|
| decode_ce_checkpoint="auto" | Historical per-chunk estimate for analytic diagonal/full, expected-likelihood, and linear CE; generic family retains its existing aggregate estimate. |
| decode_ce_checkpoint="auto_aggregate" | Whole-vocabulary estimate including analytic fp64 storage and expected-likelihood worksets; may recompute in backward to save memory. |

The default is auto. Both automatic policies use a 2 GiB estimate threshold. Evaluation never checkpoints. The forced policies "always" (checkpoint every chunk with live gradients) and "never" (retain activations without recomputation) remain on `PriorBank` for tests only; a saved config carrying either loads as "auto" (2026-10-02). Nonlinear fused CE always centers each chunk's scores on their detached maximum, which avoids large-score cancellation; the retired `decode_ce_stable_reduction` selector loads with either value and is dropped. Linear CE keeps its own reduction. Public target validation, ignored-row handling, covariance exclusions, and their gradient contracts remain active. Centering follows the configured score clamp, temperature, head evidence, and class prior.

At B=64, N=128, V=50257, chunk=8192, fp32, the analytic estimate is 0.5 GiB per chunk versus about 3.07 GiB across the vocabulary. Only the latter crosses the threshold. These are heuristic estimates, not measured peak GPU memory. A tiny CPU trace records nine versus twelve forward/backward GEMMs when aggregate checkpointing activates, with identical loss. Production CUDA throughput was not measured.

## Independent decoder learning rates

m_decode_mu_lr controls untied decoder means or linear weight/bias; m_decode_sigma_lr controls untied decoder log variances. Each defaults to None and resolves independently: explicit split rate, then legacy m_decode_lr if supplied, then the corresponding encoder rate. Zero remains zero under scheduling.

Tied tables retain encoder rates. Linear decoding has no independent variance table. Learned temperature retains m_decode_lr when supplied, otherwise m_p_sigma_lr; split variance does not change temperature. Fixed temperature has no optimizer group. Mean/variance clipping and decay policies remain attached, and linear bias keeps zero decay. Omitting split fields preserves prior optimizer topology and updates.

Both launchers expose None defaults. Independent split-rate sweep entries are optional; existing selected arms and grids are preserved. Older saved combined-rate configurations remain accepted.

## Other learning-rate inheritance

| Component | Rate and inheritance |
|---|---|
| Encoder mean table | Independent `m_p_mu_lr` |
| Encoder log-variance table | Independent `m_p_sigma_lr` |
| Token gauge frame | Independent `m_phi_lr`; gauge-off/frozen policies freeze it |
| Untied decoder means / linear projection | `m_decode_mu_lr` -> legacy `m_decode_lr` -> `m_p_mu_lr` |
| Untied decoder log variances | `m_decode_sigma_lr` -> legacy `m_decode_lr` -> `m_p_sigma_lr` |
| Learned decoder temperature | Legacy `m_decode_lr` -> `m_p_sigma_lr`; absent when fixed |
| Block MLP | `m_block_mlp_lr=None` inherits `m_p_mu_lr`; layers share the chosen rate |
| LayerNorm affine | `m_layernorm_lr=None` inherits `m_p_mu_lr`; both seams share the chosen rate |
| Relational Q/K | `m_relational_score_lr=None` inherits `m_p_mu_lr`; Q and K share the chosen rate |
| Relational T/O | `m_relational_content_lr=None` inherits `m_p_mu_lr`; T and O share the chosen rate |
| Learned positional frame | `m_pos_phi_lr=None` inherits resolved `m_phi_lr` |
| Per-type MLP hidden bias | `m_per_type_hidden_bias_lr=None` inherits declared `m_phi_lr` before gauge-off/frozen coercion |
| CG path weights | Directly use `m_p_mu_lr`; no separate toggle |
| Learned T5 bucket bias | Directly uses `m_p_mu_lr`; no separate toggle |
| Conventional baseline parameters | Independent baseline `lr`; `per_type_table_lr=None` inherits it |

E-step rates `e_q_mu_lr`, `e_q_sigma_lr`, and `e_phi_lr` are independent of M-step rates and each other; execution policies determine whether they are consumed. The recorded projected-predictive route does not consume the mean/covariance gradient-step rates. The existing custom trainable transport-registration grouping gap remains outside this request: the optimizer docstring describes `m_phi_lr`, but trainable registration-owned state reaches the missing-parameter coverage assertion. Built-in flat transport is stateless.

## Validation scope

Tiny CPU regressions cover decoder tradeoffs, ordinary value/gradient parity, the extreme-score correction, full-family exclusions, canonical delegation, evaluation, serialization, rate precedence, inactive fields, optimizer topology, certificate cost, and generation. No production GPU timing or peak-memory measurement was performed.
