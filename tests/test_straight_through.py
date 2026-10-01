"""Projected attention requires live unrolled inference gradients."""
from tests.profiles import test_config

import pytest


@pytest.mark.parametrize("mode", ["straight_through", "detach", "not_a_mode"])
def test_unsupported_estimator_modes_are_rejected(mode: str) -> None:
    with pytest.raises(ValueError):
        test_config(e_step_gradient=mode, use_block_mlp=False)


def test_config_explicit_unroll_is_preserved() -> None:
    assert test_config(e_step_gradient="unroll").e_step_gradient == "unroll"
