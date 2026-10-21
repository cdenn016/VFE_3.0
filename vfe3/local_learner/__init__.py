r"""Backprop-free local-learner test (design: ``docs/research/2026-10-21-local-learner-test/design.md``).

A linear-Gaussian latent state with a probit vocabulary tree, filtered by one causal assumed-density
filter, learned either from token-conditioned posterior statistics (stepwise EM) or by backpropagation
through the same filter. Independent of :class:`vfe3.config.VFE3Config` and of the transformer model.
"""

from vfe3.local_learner.core import (
    Params,
    TreeSensor,
    assimilate,
    evaluate_filter,
    forecast,
    init_params,
    sample_sequences,
)
from vfe3.local_learner.learners import train_backprop, train_local
from vfe3.local_learner.baselines import ngram_nll, particle_filter_nll

__all__ = [
    "Params", "TreeSensor", "assimilate", "evaluate_filter", "forecast", "init_params", "sample_sequences",
    "train_backprop", "train_local", "ngram_nll", "particle_filter_nll",
]
