"""Pure serialized-metric contracts shared by producers and artifact consumers."""
from __future__ import annotations

import math
from numbers import Real

# A scaling power law is reported with R^2, an exponent CI, or a lack-of-fit verdict only from this
# many distinct sizes. Two sizes give R^2 = 1 exactly, and three leave at most one residual degree of
# freedom for the two-parameter law (none for the offset law), so any monotone curve passes.
SCALING_MIN_TESTABLE_SIZES = 4
# The offset law E + A N^-alpha has three parameters, so it needs one more size than the plain law to
# keep two residual degrees of freedom (audit 2026-10-08 gap-stats: four sizes leave it one).
SCALING_MIN_TESTABLE_SIZES_OFFSET = 5


def scaling_min_testable_sizes(with_offset: bool) -> int:
    r"""Distinct sizes a scaling fit needs to be reported as testable: 5 for the offset law, 4 otherwise."""
    return SCALING_MIN_TESTABLE_SIZES_OFFSET if with_offset else SCALING_MIN_TESTABLE_SIZES


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
