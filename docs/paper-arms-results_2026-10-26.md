# Paper arms on the 35.06 base: results, October 26, 2026

## Setup

The base is `vfe3_runs/35.06_wikitext-103_K128_paper-arms`: the gaussian_diagonal family with learned GL(K) frames, K = 128, four heads, six layers, the ALiBi prior without RoPE, the attention residual, the coordinate block MLP, the chunked diagonal decoder and no stored-covariance decay, trained for 15,000 steps of 32 by 256 tokens (122.9M tokens, about 1.05 epochs of WikiText-103). Every arm is a declared change of that config at the same budget, run at the seed pairs (6, 3), (23, 23) and (54, 54). The base itself is the evaluated arm's (6, 3) replicate. All 33 runs completed, and the manifest is `vfe3_runs/paper_arms/k128_paper_arms/base_protocol/m2_manifest.json`. Statistics are test CE at the best-validation checkpoint, paired by seed pair, as computed by `vfe3.paper.tables.decision_rule` and `paired_difference`; intervals are 95% Student-t on two degrees of freedom.

## Results

| arm | test PPL (mean ± SD) | paired test CE minus evaluated (nats) | 95% interval | params | wall time per run |
|---|---|---|---|---|---|
| Evaluated: learned frames, stored covariance without decay | 35.16 ± 0.18 | 0 | | 233.6M | 83 min |
| Covariance-blind routing (mean distance) | 35.23 ± 0.31 | +0.002 | [−0.010, +0.015] | 233.6M | 51 min |
| Identity frames, per-type additive table of equal size | 41.00 ± 0.18 | +0.154 | [+0.142, +0.166] | 439.4M allocated (233.6M trainable) | 77 min |
| Frames frozen at their random initialization | 43.24 ± 0.13 | +0.207 | [+0.190, +0.224] | 233.6M allocated (27.7M trainable) | 75 min |
| Identity frames, stored covariance without decay | 43.47 ± 0.39 | +0.212 | [+0.178, +0.247] | 233.6M allocated (27.7M trainable) | 74 min |
| Matched total parameters (conventional, width 1024) | 29.96 ± 0.04 | −0.160 | [−0.171, −0.149] | 228.9M | 39 min |
| Matched context-processing parameters (conventional) | 57.99 ± 0.54 | +0.500 | [+0.482, +0.519] | 14.9M | 8 min |
| Per-type read table, ALiBi (as run) | 50.43 ± 0.17 | +0.361 | [+0.346, +0.375] | 220.7M | 11 min |
| Per-type read table, ALiBi, realization optimizer | 39.61 ± 0.26 | +0.119 | [+0.091, +0.147] | 220.7M | 11 min |
| Per-type read-and-write table, ALiBi | 50.03 ± 0.16 | +0.353 | [+0.336, +0.370] | 220.7M | 28 min |
| Per-type read table, ALiBi, shared post-norm | 50.96 ± 0.08 | +0.371 | [+0.362, +0.380] | 220.7M | 11 min |

The registered decision rule returns **stands**. Clause (a), the as-run ALiBi read table, exceeds the evaluated arm by 0.361 nats (SE 0.0034). Clause (b), frozen frames, costs 0.207 nats (SE 0.0039). Clause (c), the equal-size capacity control, costs 0.154 nats (SE 0.0028). Every margin is the 0.03-nat floor, since 2 SE is below it in each clause. No clause is undetermined, no run is stale or failed, and the homogeneity checks pass.

## Reading

Learned frames carry the model's advantage over the frame-free variants. Removing them costs 0.212 nats. Freezing them at a random initialization recovers only 0.005 of that, so random per-type GL transforms do almost nothing on their own. An additive per-type hidden-layer table of the frame table's size recovers 0.058 nats, about 27% of the gap. The capacity explanation therefore fails for this form of capacity. A per-type multiplicative map without gauge structure, acting where the frames act, has not been tested and is the nearest remaining capacity alternative.

The comparison against conventional transformers splits. The VFE model beats every per-type read-table transformer and the transformer with matched context-processing parameters. It loses by 0.160 nats to the conventional transformer of matched total parameters, which spends about 126M parameters on context processing against the VFE model's 2.0M, trains in 39 minutes against 83, and was run at an untuned learning rate (the base width's peak rate with the hidden-matrix rate scaled by 128/1024). The manuscript's frame claim is the per-type-table comparison, which this run supports; any claim of competitiveness with a conventional transformer at equal total parameters is contradicted at this operating point.

The comparator is under-tuned. Changing only the optimizer to the realization's (Adam β₂ = 0.999 instead of 0.95, no decay on matrices instead of 0.1, decay 0.1 on the table) moves the read-table transformer from 50.43 to 39.61 and cuts its gap from 0.361 to 0.119 nats. Clause (a) still holds against that arm with a large margin, but 0.119 nats, not 0.361, is the defensible size of the frame model's advantage over the read table. The read-and-write and shared post-norm arms ran under the as-run optimizer, so their structural changes are measured against the weaker optimizer: read-and-write gains 0.008 nats and post-norm loses 0.010, both small beside the optimizer's 0.242.

Covariance-blind routing matches the Gaussian-KL routing: +0.002 nats, with an interval that contains zero, at 61% of the wall time. At this operating point the routing reads nothing measurable from the covariance. This does not test the covariance elsewhere in the model (the decoder reads it), only its role in the routing cost.

## Training trajectories per step

Every arm uses the same batch (32 by 256 tokens), 250 warmup steps and a cosine decay to zero over 15,000 steps, so loss per step is also loss per token. The tables below are seed means over the three replicates. Train CE is the interval mean over each 500-step logging window, and validation CE is measured every 3,000 steps. Each entry is the arm's value minus the evaluated arm's at the same step, so positive means the VFE model is ahead.

| arm | train, 500 | 1000 | 2000 | 3000 | 6000 | 9000 | 12000 | 15000 |
|---|---|---|---|---|---|---|---|---|
| Matched total parameters | −0.064 | +0.153 | +0.115 | +0.083 | +0.011 | −0.025 | −0.078 | −0.099 |
| Read table, realization optimizer | +0.163 | +0.250 | +0.156 | +0.109 | +0.069 | +0.082 | +0.103 | +0.136 |
| Read table, as run | +0.225 | +0.401 | +0.382 | +0.382 | +0.410 | +0.419 | +0.397 | +0.447 |
| Frozen frames | +0.139 | +0.131 | +0.139 | +0.148 | +0.179 | +0.194 | +0.216 | +0.294 |
| Identity frames, capacity table | +0.123 | +0.116 | +0.113 | +0.117 | +0.134 | +0.143 | +0.162 | +0.227 |
| Mean distance | −0.025 | +0.040 | +0.034 | +0.031 | +0.027 | +0.026 | +0.024 | +0.032 |

| arm | val, 3000 | 6000 | 9000 | 12000 | 15000 |
|---|---|---|---|---|---|
| Matched total parameters | +0.044 | −0.034 | −0.077 | −0.131 | −0.160 |
| Read table, realization optimizer | +0.093 | +0.066 | +0.082 | +0.103 | +0.113 |
| Read table, as run | +0.369 | +0.391 | +0.399 | +0.371 | +0.355 |
| Frozen frames | +0.145 | +0.179 | +0.189 | +0.206 | +0.212 |
| Identity frames, capacity table | +0.115 | +0.131 | +0.138 | +0.151 | +0.155 |
| Mean distance | +0.020 | +0.013 | +0.014 | +0.010 | +0.009 |

Per step, the VFE model learns faster early than every conventional arm. Against the matched-total transformer its lead peaks at 0.153 nats of train CE at step 1,000 and then shrinks steadily. The validation curves cross between steps 3,000 and 6,000 (about step 4,700 by linear interpolation) and the train curves near step 7,000. The transformer then gains about 0.04 nats per 3,000 steps until the end. Against the read table with the realization optimizer, the lead is also largest at step 1,000 (0.250); it falls to a minimum of 0.069 near step 6,000 and widens again to 0.136 in the cosine tail. Measured as steps, the evaluated arm reaches the as-run read table's final validation CE by step 6,800, the tuned read table's by step 10,600, and the frozen-frame arm's by step 8,800. It never reaches the matched-total transformer's 3.381 within the budget.

The early lead over the matched-total transformer is confounded with that arm's learning rate. Its hidden matrices train at 0.003 × 0.125 = 3.75e-4, a scale set by the width ratio and not tuned, and that rate governs how fast a width-1024 model moves during the first few thousand steps. A transformer with a better-tuned rate could close the early gap as well. The run supports this statement: at equal steps and this optimizer setting, the VFE model is ahead for roughly the first third of training and behind thereafter. It does not establish that the VFE model has a higher early sample efficiency than a tuned transformer.

The frame arms behave in the opposite way. Their gap is already 0.12 to 0.14 nats at step 500 and grows over training, and in train CE it jumps sharply in the last interval: +0.294 against +0.212 in validation for frozen frames. The evaluated arm's final train CE (3.485) sits 0.056 below its validation CE (3.541), while the frozen-frame arm's train CE sits 0.026 above its validation CE. Learned frames therefore buy faster fitting throughout and also a larger train-validation gap in the last part of the schedule; part of their late train advantage does not transfer to held-out text. Mean-distance routing tracks the Gaussian-KL routing within 0.04 nats at every step after step 500.

## Caveats

These results hold at one operating point: K = 128, six layers, one epoch. The 25.12 configuration (K = 192, eight layers, RoPE, 120,000 steps) operates in a memorization regime (a train-sample gap of 0.545 nats) that one epoch excludes, and the frame table and the capacity table are both trained about one-eighth as long per type here. The `frame_capacity` ablation sweep on the eight-layer RoPE configuration at one epoch gave a similar capacity gap (+0.165 nats in validation CE against +0.154 in test CE here), so the result does not depend on depth or RoPE at that budget.

The manifest flags code-identity differences: the runs were trained on three source-tree hashes. Comparing the archived sources of the base, a frozen-frame run, a mean-distance run and a baseline run shows the `vfe3` package identical across all of them; only the root scripts `paper_arms.py` (the base selection committed on October 25) and `train_vfe3.py` (the launcher's working-copy edits) differ, and neither is on the training path of `vfe3.experiment.run_vfe` or the baseline trainer.

Three seeds give a paired SE of 0.003 to 0.008 nats. The 2 SE part of the margin then has a one-sided level of about 0.092 on two degrees of freedom, as the rule records; every observed difference except the mean-distance arm's exceeds its margin by more than a factor of three.
