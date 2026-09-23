"""Pure serialized-metric contracts shared by producers and artifact consumers."""
from __future__ import annotations

import math
from numbers import Real

# A scaling power law is reported with R^2, an exponent CI, or a lack-of-fit verdict only from this
# many distinct sizes. Two sizes give R^2 = 1 exactly, and three leave at most one residual degree of
# freedom for the two-parameter law (none for the offset law), so any monotone curve passes.
SCALING_MIN_TESTABLE_SIZES = 4


def perplexity_from_ce(ce: float) -> float:
    """Return the exact image ``exp(CE)``; finite-CE overflow is positive infinity."""
    value = float(ce)
    try:
        return math.exp(value)
    except OverflowError:
        return float("inf")


def perplexity_matches_ce(
    ce: object,
    perplexity: object,
    *,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-12,
) -> bool:
    """Whether a serialized PPL is exactly consistent with a finite serialized CE.

    Positive infinity is accepted only when evaluating ``exp(CE)`` overflows. A finite
    exponential must be represented by its finite value; caps and arbitrary infinities fail.
    """
    if (isinstance(ce, bool) or isinstance(perplexity, bool)
            or not isinstance(ce, Real) or not isinstance(perplexity, Real)):
        return False
    ce_value, ppl_value = float(ce), float(perplexity)
    if not math.isfinite(ce_value) or math.isnan(ppl_value) or ppl_value <= 0.0:
        return False
    expected = perplexity_from_ce(ce_value)
    if math.isinf(expected):
        return math.isinf(ppl_value) and ppl_value > 0.0
    return math.isfinite(ppl_value) and math.isclose(
        ppl_value, expected, rel_tol=rel_tol, abs_tol=abs_tol)


def target_blind_objective_interpretation() -> str:
    """State only what target blindness implies about held-out prediction."""
    return ("The target-blind latent objective and held-out prediction are separate objectives; "
            "target blindness alone implies no Pearson sign.")
