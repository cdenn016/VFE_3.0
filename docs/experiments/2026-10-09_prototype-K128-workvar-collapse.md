# Prototype-agent block at K=128 with working-variance init: template collapse

Runs compared (one seed each, so every comparison here is [underpowered]):

New: `vfe3_runs/52.03_wikitext-103_K128_MinAgent_Proto_WorkVar_MeanField` (K=128, 4 layers, M=1024 templates per layer, 180k steps, val ppl 52.03).
Old: `D:/DATA_KEEP/10-7-2026_Experiments/136.46_wikitext-103_K20_MF_Proto_MF_post_diag-chunk` (K=20, M=160, 30k steps, val ppl 136.46).

## What happened

The prototype block in the new run collapsed to a single template in every layer before the first diagnostic at step 10k and never recovered. From `diagnostics/prototype_templates.csv`, effective templates are 1.00 to 1.02, the dead fraction is 0.998 to 0.999, the prior entropy is below 1e-3 nats, and the gain fraction (removed working variance per dimension) is 0.0 to 2.7 percent. The old run at its end had 9 to 49 effective templates, dead fraction 0.43 to 0.91, prior entropy 0.3 to 4.3 nats, and gain fraction 5 to 50 percent. The final checkpoints agree: the new run's `log_pi` maxima are 47, 122, 107 and 189 (prior mass on the winner 0.9993 to 1.0000), against 2.6 to 7.4 in the old run.

The block also switched itself off. The median template log variance is 5.0 to 7.8 (variance of roughly 150 to 2400) against a working variance near `sigma_init=5`, so the gain `c/(c+lambda)` is on the order of 1e-3 to 3e-2. The template means never moved off their initialization: the median `|nu|` is 0.00 to 0.09, against 3 to 5 in the old run. Because the posterior weight is concentrated on one template, the other 1023 receive essentially no gradient, which freezes the collapse in place. The reported 52.03 is therefore the score of what is effectively an attention-only model with an inert MLP slot; it says nothing about prototype agents at K=128.

## Mechanism (interpretation, partly measured)

With `block_mlp_template_init='working_variance'` the template means are copied from the fresh model's block-input means. At `mu_init_std=0.03`, K=128, those have norm about 0.03 * sqrt(128) = 0.34, while `lambda_0 = c(1-p)/p` with c near 5 and p = 0.09 gives lambda_0 near 50. The squared Mahalanobis separation between templates is then of order 0.2/55, so the data term of the routing logits is nearly identical across templates and the routing is symmetric. The only symmetry breaker left is `log_pi`, and nothing restores it: in an EM fit the prior's gradient is `rbar_a - pi_a`, which is mean-reverting, but here `log_pi` is trained by the downstream cross-entropy, which carries no such restoring term. With `m_block_mlp_lr=0.05` (ten times the old run's 0.005, and `log_pi` sits in a group at that LR with no weight decay, `vfe3/train.py:314`), Adam can move `log_pi` by about 0.05 per step, so a winner can open a gap of tens of nats within a few thousand steps. That is consistent with the observed `log_pi` maxima, but the run logs no prototype statistics before step 10k, so the onset is unobserved. The calibrated `working_variance` and `lambda_0` were sent to the logger only and are not in the run directory, so the c of about 5 above is inferred from `sigma_init`, not read.

## Confounds

The old and new runs differ in K (128 vs 20), template init (working_variance vs absolute), block LR (0.05 vs 0.005), covariance contract (posterior vs passthrough), `sigma_init` (5 vs 15), `mu_init_std` (0.03 vs 0.065), batch size (16 vs 32), positional scheme (rope vs learned phi), `decode_tau` (0.1 vs 1) and `kl_max`. The collapse cannot be attributed to the working-variance init alone from these two runs. The two most likely causes, given the mechanism, are the 10x block LR and the near-degenerate template means; K enters through the routing temperature.

## Next experiments

Each is a short K=128 run (5k steps, with the prototype diagnostic interval reduced to 500 steps) that changes one knob from the new run's config. Collapse is visible in under 10k steps, so these are cheap.

1. `m_block_mlp_lr=0.005`, everything else as the new run. If effective templates stay above about 10 and prior entropy above 1 nat at 5k steps, the LR is sufficient to explain the collapse. If it still collapses, the LR is not the cause.
2. `block_mlp_template_init='absolute'` at the new run's LR. Separates the init from the LR.
3. If (1) alone fixes it, consider a lower LR (or weight decay toward zero) for `log_pi` specifically, since it is the unrestored direction.

Two instrumentation gaps surfaced here and were closed the same day (`docs/2026-10-09.md`): the calibration statistics (`working_variance`, `lambda0`) were not written to the run directory, and prototype diagnostics were only recorded at `eval_interval`, which missed the entire collapse. Runs now write `prototype_calibration.json` and `diagnostics/prototype_templates_train.csv` (every log step), so the short runs above only need `log_interval` set to 500 or less.

## Follow-up: 2x2 of template init and block LR (5k steps)

Four runs in `vfe3_runs/prototype-testing/`, identical to the 52.03 config except `block_mlp_template_init` and `m_block_mlp_lr` (and the shorter schedule with `log_interval=100`, `eval_interval=1000`). One seed each.

| run | final val PPL | prior entropy at 5k (layers 0 to 3, nats; uniform is 6.93) | gain fraction at 5k |
|---|---|---|---|
| working_variance, 0.005 | 101.55 | 6.92, 6.92, 6.91, 6.66 | 0.14 to 0.22 |
| working_variance, 0.05 | 102.04 | 0.0 in every layer by step 3000 | 0.04 to 0.10 |
| absolute, 0.05 | 102.25 | 0.0 in every layer by step 3000 | 0.06 to 0.11 |
| absolute, 0.005 | 103.79 | 4.50, 4.38, 4.37, 6.52 | 0.01 to 0.08 |

The block LR is what causes the collapse: at 0.05 both inits drive the prior entropy to zero in every layer between steps 500 and 3000, and at 0.005 neither does. The init decides whether the block stays active at the lower LR. Under `absolute` at 0.005 the template variances grow from log lambda of about 2.4 to about 6 in layers 0 to 2, and the gain fraction falls to 1 to 2 percent, so the block switches itself off much as it did in 52.03. Under `working_variance` at 0.005 log lambda moves only from 3.3 to 3.4 (layer 0) and the gain fraction holds at 14 to 22 percent with hundreds of templates in use in layers 0 to 2.

This corrects two statements above. The measured working variance at the block input is 2.7, 1.44, 0.84 and 0.53 for layers 0 to 3 (`prototype_calibration.json`), not about 5, so lambda_0 is 27, 15, 8.5 and 5.4. And `log_pi` is not the first thing to give way. At LR 0.05 the training-batch routing is already concentrated on one to three templates by step 100 to 300, while the prior entropy is still near 6.9. The prior collapses afterwards. The high LR on the whole block (nu and lambda as well as log_pi) concentrates the routing, and the prior follows it.

The PPL ordering is not established. The spread is 2.2 PPL across single-seed 5k-step runs. The only measured noise floor is at K=30, about 0.9 PPL at two standard deviations for a single-seed difference, and none has been measured at K=128. The diagnostics separate the arms clearly. The perplexities do not yet.
