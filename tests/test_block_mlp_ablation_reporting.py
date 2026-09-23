"""End-to-end ablation disclosure for the optional block MLP."""

import ablation

from vfe3.config import VFE3Config


def _cfg(enabled: bool) -> VFE3Config:
    return VFE3Config(
        vocab_size=17,
        embed_dim=4,
        n_heads=2,
        max_seq_len=8,
        gauge_group="block_glk",
        use_block_mlp=enabled,
    )


def test_ablation_rows_and_csv_preserve_mlp_gauge_contract():
    off = ablation._gauge_reporting_fields(_cfg(False))
    on = ablation._gauge_reporting_fields(_cfg(True))

    assert off["block_mlp_structural_mode"] == "disabled"
    assert off["block_mlp_covariance_contract"] == "not_applicable"
    assert off["block_mlp_intertwiner_compatible"] is True
    assert on["block_mlp_structural_mode"] == "coordinate_mean_only_nonintertwiner"
    assert on["block_mlp_covariance_contract"] == "passthrough"
    assert on["block_mlp_intertwiner_compatible"] is False
    assert on["on_gauge_pure_path"] is False

    for key in (
        "block_mlp_structural_mode",
        "block_mlp_covariance_contract",
        "block_mlp_intertwiner_compatible",
    ):
        assert key in ablation._CSV_COLUMNS


def test_ablation_summary_and_figure_disclose_nonpushforward_mlp():
    rows = [
        {"label": "off", **ablation._gauge_reporting_fields(_cfg(False))},
        {"label": "on", **ablation._gauge_reporting_fields(_cfg(True))},
    ]
    summary = ablation._gauge_purity_summary(rows)
    assert summary["contains_coordinate_mean_only_nonintertwiner"] is True
    assert summary["block_mlp_intertwiner_compatible_by_label"] == {
        "off": True,
        "on": False,
    }
    assert summary["all_rows_on_gauge_pure_path"] is False
    text = ablation._gauge_disclosure_text(summary).lower()
    assert "not gauge-pure" in text
    assert "mean-only" in text
    assert "covariance passthrough" in text
    assert "not a gaussian pushforward" in text


def test_cross_sweep_bar_label_includes_active_mlp_classification():
    label = ablation._sensitivity_gauge_label({
        "head_mixer_compatibility": "disabled",
        "block_mlp_structural_mode": "coordinate_mean_only_nonintertwiner",
    })
    assert label == (
        "head_mixer=disabled; block_mlp=coordinate_mean_only_nonintertwiner"
    )
