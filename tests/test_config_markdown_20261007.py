"""config.md: the launcher's config dict verbatim, plus the values it does not show as literals."""

from pathlib import Path

from vfe3.config_markdown import render_config_markdown, write_config_markdown

SOURCE = '''import os

DATASET = "toy"

config = dict(
    #####  Training  #####
    vocab_size  = len(DATASET),     # from the tokenizer
    embed_dim   = 4,                # K
    lr          = 1e-3,

    family      = "gaussian_diagonal",
)

# scale with K
config["kl_max"] = 8 * config["embed_dim"]


def main():
    config["never"] = 1
'''


def test_block_is_verbatim_and_unshown_values_are_listed() -> None:
    resolved = {"vocab_size": 3, "embed_dim": 4, "lr": 2e-3, "family": "gaussian_diagonal",
                "kl_max": 32, "seed": 6}
    md = render_config_markdown(SOURCE, resolved, header={"DATASET": "toy"})
    block = SOURCE[SOURCE.index("config = dict("):SOURCE.index('config["embed_dim"]') + len('config["embed_dim"]')]
    assert block in md
    assert "def main" not in md and "never" not in md
    assert "| `seed` | `6` | from SEEDS |" in md
    assert "| `vocab_size` | `3` | from `len(DATASET)` |" in md
    assert "| `kl_max` | `32` | set after the block |" in md
    assert "| `lr` | `0.002` | source literal `0.001` |" in md
    assert "`embed_dim`" not in md.split("## Values")[1] and "`family`" not in md.split("## Values")[1]
    assert md.startswith("# Run configuration (train_vfe3.py)\n\n- DATASET: `'toy'`")


def test_write_skips_a_launcher_without_a_block(tmp_path: Path) -> None:
    launcher = tmp_path / "launch.py"
    launcher.write_text("x = 1\n", encoding="utf-8")
    assert write_config_markdown(tmp_path, launcher, {}) is None
    launcher.write_text(SOURCE, encoding="utf-8")
    path = write_config_markdown(tmp_path, launcher, {"embed_dim": 4})
    assert path == tmp_path / "config.md" and "# Run configuration (launch.py)" in path.read_text(encoding="utf-8")
