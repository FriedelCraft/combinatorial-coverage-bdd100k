"""Summarize repeated development-domain mAP without touching the final set."""

import argparse
import csv
import statistics
from pathlib import Path


def read_metrics(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["domain"]: float(row["map_50_95"]) for row in csv.DictReader(handle)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("outputs/experiments"))
    parser.add_argument("--output", type=Path, default=Path("DEVELOPMENT_RESULTS.md"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--methods", nargs="+",
                        default=["generic_coverage", "visual_cluster_aware"])
    parser.add_argument("--steps", type=int, default=7500)
    args = parser.parse_args()
    records = {}
    for method in args.methods:
        for seed in args.seeds:
            path = args.runs_dir / f"benchmark_night_{method}_seed{seed}_steps{args.steps}" / "domain_metrics_development.csv"
            if not path.is_file():
                raise FileNotFoundError(f"Incomplete benchmark: {path}")
            records[method, seed] = read_metrics(path)
    domains = ("all", "daytime", "night", "rainy")
    lines = ["# Repeated development benchmark", "",
             f"{args.steps:,} optimizer updates per run; seeds {', '.join(map(str, args.seeds))}.",
             "All metrics use the same 1,000-image development set. The final set is untouched.",
             "", "| Method | Seed | Overall | Day | Night | Rain |",
             "|---|---:|---:|---:|---:|---:|"]
    for method in args.methods:
        for seed in args.seeds:
            values = records[method, seed]
            lines.append(f"| {method} | {seed} | " + " | ".join(
                f"{values[domain]:.4f}" for domain in domains) + " |")
        means = [statistics.mean(records[method, seed][domain] for seed in args.seeds)
                 for domain in domains]
        lines.append(f"| **{method} mean** | — | " + " | ".join(
            f"**{value:.4f}**" for value in means) + " |")
    if len(args.methods) == 2:
        first, second = args.methods
        lines.extend(["", f"Paired difference ({second} minus {first}):", "",
                      "| Seed | Overall | Day | Night | Rain |",
                      "|---:|---:|---:|---:|---:|"])
        for seed in args.seeds:
            lines.append(f"| {seed} | " + " | ".join(
                f"{records[second, seed][domain] - records[first, seed][domain]:+.4f}"
                for domain in domains) + " |")
        lines.append("| **Mean difference** | " + " | ".join(
            f"**{statistics.mean(records[second, seed][domain] - records[first, seed][domain] for seed in args.seeds):+.4f}**"
            for domain in domains) + " |")
    lines.extend(["", f"{len(args.seeds)} seeds support descriptive comparisons; uncertainty is limited. The rainy",
                  "development domain contains 78 images; uncertainty is not estimated.",
                  "Final evaluation must wait until the method and analysis plan are frozen."])
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
