"""Regression tests for E-step naming and fixed-point reporting buildout."""

from tests.profiles import test_config

import pytest





def test_group_product_accepts_the_flat_tied_phi_route() -> None:
    assert test_config(pos_phi="learned").pos_phi == "learned"
    assert test_config(pos_phi="none").pos_phi == "none"


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_phi_mstep_max_matrix_norm_must_be_positive_or_none(value: float) -> None:
    assert test_config(phi_mstep_max_matrix_norm=None).phi_mstep_max_matrix_norm is None
    with pytest.raises(ValueError, match="phi_mstep_max_matrix_norm"):
        test_config(phi_mstep_max_matrix_norm=value)
