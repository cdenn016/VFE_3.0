"""2026-10-08 audit items fixed outside the per-area groups: window-clustered Spearman and int/float homogeneity."""
import numpy as np


def test_grouped_spearman_resamples_windows_and_drops_the_iid_p_value() -> None:
    """gap-stats#7: tokens of one window share a window effect, so the interval resamples windows."""
    from vfe3.semantics import stats

    rng = np.random.default_rng(0)
    windows = np.repeat(np.arange(6), 40)                        # 6 windows of 40 dependent tokens
    shift = rng.standard_normal(6)[windows]
    x = shift + 0.1 * rng.standard_normal(windows.size)
    y = shift + 0.1 * rng.standard_normal(windows.size)
    iid = stats.spearman_ci(x, y, np.random.default_rng(1), n_boot=400)
    clustered = stats.spearman_ci(x, y, np.random.default_rng(1), n_boot=400, groups=windows)
    assert clustered["p"] is None and clustered["n_groups"] == 6 and iid["p"] is not None
    assert clustered["rho"] == iid["rho"]
    assert clustered["ci_hi"] - clustered["ci_lo"] > 2 * (iid["ci_hi"] - iid["ci_lo"])


def test_homogeneity_canonical_form_ignores_int_float_spelling() -> None:
    """general-C#7: rope_fraction=1 and 1.0 are one configuration."""
    from vfe3.paper.runs import _canonical

    assert _canonical({"rope_fraction": 1, "kl_max": [160]}) == _canonical({"rope_fraction": 1.0, "kl_max": [160.0]})
    assert _canonical({"rope_fraction": 0.5}) != _canonical({"rope_fraction": 1})
    assert _canonical({"flag": True}) != _canonical({"flag": 1.0})
