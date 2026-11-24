"""Draw the frozen paired final-set comparison from saved evaluation CSVs."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUNS = Path("outputs/experiments")
FIGURES = Path("figures")
SEEDS = range(42, 47)
DOMAINS = ("night", "all", "daytime", "rainy")


def metrics(method, seed):
    path = RUNS / f"benchmark_night_{method}_seed{seed}_steps7500" / "domain_metrics_final_evaluation.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["domain"]: float(row["map_50_95"]) for row in csv.DictReader(handle)}


def main():
    # The figure is only produced once all five pairs are available.
    generic = {seed: metrics("generic_coverage", seed) for seed in SEEDS}
    adaptive = {seed: metrics("adaptive_target_aware", seed) for seed in SEEDS}
    FIGURES.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    colors = {"night": "#17558a", "all": "#27916f", "daytime": "#b77723", "rainy": "#7b58a3"}

    ax = axes[0]
    for i, seed in enumerate(SEEDS):
        left, right = generic[seed]["night"], adaptive[seed]["night"]
        ax.plot([0, 1], [left, right], color="#a6b8c6", lw=1.4, zorder=1)
        ax.scatter([0, 1], [left, right], color=["#8a9aa8", colors["night"]], s=48, zorder=2)
    ax.set_xticks([0, 1], ["Generic coverage", "Adaptive"])
    ax.set_xlim(-0.2, 1.3)
    ax.set_ylim(0.18, 0.23)
    ax.set_ylabel("Night COCO mAP@0.50:0.95")
    ax.set_title("A. Paired night performance")
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1]
    for j, domain in enumerate(DOMAINS):
        values = [adaptive[seed][domain] - generic[seed][domain] for seed in SEEDS]
        y = 3 - j
        ax.scatter(values, [y] * len(values), color=colors[domain], s=34, alpha=0.75)
        mean = sum(values) / len(values)
        ax.scatter([mean], [y], color=colors[domain], s=110, marker="D", edgecolor="white", linewidth=0.7)
    ax.axvline(0, color="#555", lw=1, ls="--")
    ax.set_yticks(range(4), ["Rain", "Day", "Overall", "Night"])
    ax.set_xlabel("Adaptive − generic mAP (each dot = one seed; diamond = mean)")
    ax.set_xlim(-0.012, 0.03)
    ax.set_title("B. Paired differences by domain")
    ax.grid(axis="x", alpha=0.2)

    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"final_paired_comparison.{suffix}", dpi=200)
    plt.close(fig)
    print("Wrote figures/final_paired_comparison.png and .pdf")


if __name__ == "__main__":
    main()
