"""Summarize the frozen five-seed final evaluation after every run completes."""

import csv
import math
import statistics
from pathlib import Path


ROOT = Path("outputs/experiments")
SEEDS = range(42, 47)
METHODS = ("generic_coverage", "adaptive_target_aware")
DOMAINS = ("all", "daytime", "night", "rainy")
T_CRITICAL_DF4 = 2.7764451051977987


def read_metrics(method, seed):
    path = (ROOT / f"benchmark_night_{method}_seed{seed}_steps7500"
            / "domain_metrics_final_evaluation.csv")
    if not path.is_file():
        raise FileNotFoundError(f"Final evaluation is incomplete: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = {row["domain"]: row for row in csv.DictReader(handle)}
    if set(rows) != set(DOMAINS):
        raise ValueError(f"Unexpected domains in {path}: {sorted(rows)}")
    return {domain: float(rows[domain]["map_50_95"]) for domain in DOMAINS}


def fmt(values):
    return " | ".join(f"{value:.4f}" for value in values)


def main():
    # Read all inputs before writing a report, so a partial run cannot be
    # mistaken for the frozen five-pair comparison.
    records = {(method, seed): read_metrics(method, seed)
               for method in METHODS for seed in SEEDS}
    lines = ["# Frozen final evaluation", "",
             "Five paired seeds (42–46), 7,500 optimizer updates and 7,500 selected training images per run.",
             "Metrics are COCO mAP@0.50:0.95 on the same 8,500 held-out images.",
             "The primary endpoint is the mean paired night difference (adaptive minus generic).",
             "", "| Method | Seed | Overall | Day | Night | Rain |",
             "|---|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        for seed in SEEDS:
            lines.append(f"| {method} | {seed} | {fmt(records[method, seed][domain] for domain in DOMAINS)} |")
        lines.append(f"| **{method} mean** | — | " + fmt(
            statistics.mean(records[method, seed][domain] for seed in SEEDS)
            for domain in DOMAINS) + " |")
    lines += ["", "## Paired differences", "",
              "| Seed | Overall | Day | Night | Rain |",
              "|---:|---:|---:|---:|---:|"]
    diffs = {domain: [records[METHODS[1], seed][domain] - records[METHODS[0], seed][domain]
                      for seed in SEEDS] for domain in DOMAINS}
    for index, seed in enumerate(SEEDS):
        lines.append(f"| {seed} | " + " | ".join(
            f"{diffs[domain][index]:+.4f}" for domain in DOMAINS) + " |")
    lines.append("| **Mean** | " + " | ".join(
        f"**{statistics.mean(diffs[domain]):+.4f}**" for domain in DOMAINS) + " |")
    lines += ["", "## Exploratory uncertainty", "",
              "Two-sided 95% paired t intervals across five seeds (df=4). These describe run-to-run variation;",
              "five seeds are too few for strong distributional claims. Images within a seed are shared,",
              "so this interval does not estimate uncertainty from sampling new images or datasets.", "",
              "| Domain | Mean difference | 95% paired t interval | Positive pairs |",
              "|---|---:|---:|---:|"]
    for domain in ("night", "all", "daytime", "rainy"):
        values = diffs[domain]
        mean = statistics.mean(values)
        halfwidth = T_CRITICAL_DF4 * statistics.stdev(values) / math.sqrt(len(values))
        lines.append(f"| {domain} | {mean:+.4f} | [{mean-halfwidth:+.4f}, {mean+halfwidth:+.4f}] | "
                     f"{sum(value > 0 for value in values)}/5 |")
    lines += ["", "Rain is a secondary slice (532 held-out images) and was not the selection target.",
              "The protocol and endpoint were frozen before final scoring; see `FINAL_EVALUATION_PROTOCOL.md`."]
    output = Path("FINAL_EVALUATION_RESULTS.md")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
