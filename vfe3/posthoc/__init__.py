r"""Post-hoc analyses of a trained vfe3 run (no training): the evidence GL(K)_attention_v2 reserves.

Modules (imported lazily by the caller; this package imports nothing heavy on its own):

* :mod:`~vfe3.posthoc.streams`: the run's pinned evaluation streams (validation, test, a fixed
  training-split sample), identity-checked by SHA-256 and token count, and article segmentation.
* :mod:`~vfe3.posthoc.evaluate`: model loading, the per-target NLL pass under the run's own evaluation
  protocol, cross-checked against :func:`vfe3.train.evaluate`, and the article-level bootstrap.
* :mod:`~vfe3.posthoc.interventions`: the M5 frozen interventions (rows 1-12) as reversible patches.
* :mod:`~vfe3.posthoc.klcap`: beta-weighted activation of the KL cost cap per layer and head.
* :mod:`~vfe3.posthoc.attention`: per-head attention statistics (S2 panel B).
* :mod:`~vfe3.posthoc.maps`: invariants of the relational maps (S1 panel B, S2 panel A).
* :mod:`~vfe3.posthoc.residual`: per-layer residual ratios and LayerNorm change (v2:899, S1 panel A
  endpoints), with the in-training probe's definitions.
* :mod:`~vfe3.posthoc.covariance`: realized-frame routing covariance gap and the hidden temperature of
  the stored variances (v2:872).
* :mod:`~vfe3.posthoc.calibration`: BPE/word text audit and word-normalized test perplexity
  (``tab:placeholder_calibration``).
* :mod:`~vfe3.posthoc.routing`: the M4 routing-score decomposition with its own writer; the report
  imports it lazily and records a reason when it is missing or fails.
* :mod:`~vfe3.posthoc.figures`: the S1 panels A (endpoints) and B, S2 panels A/B, M5, v2:872 and
  CE-by-context figures.
* :mod:`~vfe3.posthoc.report`: the orchestrating :func:`~vfe3.posthoc.report.run_posthoc` and the
  markdown tables (driver: ``posthoc_eval.py``).

Parameter and resource accounting lives in :mod:`vfe3.run_resources`, shared with finalize.
"""
