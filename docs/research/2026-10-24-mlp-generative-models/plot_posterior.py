"""Render the analytical two-template example from check-results.json."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def density(x: np.ndarray, mean: float, variance: float) -> np.ndarray:
    r"""Evaluate the normalized scalar Gaussian density."""
    return np.exp(-0.5 * (x - mean) ** 2 / variance) / np.sqrt(2 * np.pi * variance)


def main() -> None:
    """Save an explanatory figure; no model or training run is performed."""
    folder = Path(__file__).resolve().parent
    example = json.loads((folder / "check-results.json").read_text(encoding="utf-8"))["symmetric_example"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none", "savefig.dpi": 240})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), layout="constrained")
    x = np.linspace(-6, 6, 1201)
    exact = 0.5 * density(x, -2.4, 0.2) + 0.5 * density(x, 2.4, 0.2)
    axes[0].plot(x, exact, color="#0072B2", lw=2.2, label="Exact mixture marginal")
    axes[0].plot(x, density(x, 0, example["exact_output_variance"]), color="#D55E00", lw=2,
                 ls="--", label="Gaussian with exact moments")
    axes[0].plot(x, density(x, 0, example["mean_field_output_variance"]), color="#009E73", lw=2,
                 ls=":", label="Symmetric CAVI fixed point")
    axes[0].set(xlabel=r"Latent content $x$", ylabel="Density", xlim=(-6, 6), ylim=(0, 0.96),
                title=r"(a) Same mean, different beliefs at $o=0$")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    observation = np.linspace(-2.5, 2.5, 501)
    w_plus = 1 / (1 + np.exp(-4.8 * observation))
    posterior_variance = 0.2 + 23.04 * w_plus * (1 - w_plus)
    axes[1].plot(observation, posterior_variance, color="#0072B2", lw=2.2, label="Exact mixture variance")
    axes[1].axhline(1, color="#777777", lw=1.5, ls="--", label="Incoming variance / passthrough")
    axes[1].axhline(0.2, color="#009E73", lw=2, ls=":", label="CAVI Gaussian variance")
    axes[1].set(xlabel=r"Record value $o=\xi$", ylabel="Variance", xlim=(-2.5, 2.5), ylim=(0, 6.4),
                title="(b) Ambiguity can increase posterior variance")
    axes[1].legend(frameon=False, fontsize=8.5, loc="upper right")
    for ax in axes:
        ax.grid(axis="y", color="#DDDDDD", lw=0.5, alpha=0.6)
        ax.set_axisbelow(True)
    fig.suptitle(r"Two equally weighted templates: $\nu=\pm3$, $\Lambda=0.25$, $C=1$", fontsize=12)
    for extension in ("svg", "png"):
        fig.savefig(folder / f"posterior-comparison.{extension}", facecolor="white")
    plt.close(fig)
    print("Saved posterior-comparison.svg and posterior-comparison.png")


if __name__ == "__main__":
    main()
