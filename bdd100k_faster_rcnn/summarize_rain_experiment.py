"""Report the second-target rain comparison only after all runs finish."""

import csv
import math
import statistics
from pathlib import Path


ROOT = Path("outputs/experiments")
SEEDS = range(42, 47)
DOMAINS = ("rainy", "all", "daytime", "night", "nonrain")
T4 = 2.7764451051977987


def read(method, seed):
    target = "night" if method == "generic_coverage" else "rain"
    path = (ROOT / f"benchmark_{target}_{method}_seed{seed}_steps7500"
            / "domain_metrics_final_evaluation.csv")
    if not path.is_file():
        raise FileNotFoundError(f"Run incomplete: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        values = {row["domain"]: float(row["map_50_95"]) for row in csv.DictReader(handle)}
    if any(domain not in values for domain in DOMAINS if domain != "nonrain"):
        raise ValueError(f"Missing expected domain in {path}")
    return values


def main():
    methods = ("generic_coverage", "rain_count_control", "adaptive_target_aware")
    values = {(method, seed): read(method, seed)
              for method in methods for seed in SEEDS}
    lines = ["# Rain target experiment: exploratory results", "",
             "This second-target experiment reuses a final partition whose rainy baseline",
             "scores were already viewed during the night study. It is exploratory.",
             "The protocol is in `RAIN_EXPERIMENT_PROTOCOL.md`.", "",
             "| Method | Seed | Rain mAP | Overall mAP | Day mAP | Night mAP |",
             "|---|---:|---:|---:|---:|---:|"]
    for method in methods:
        for seed in SEEDS:
            x = values[method, seed]
            lines.append(f"| {method} | {seed} | {x['rainy']:.4f} | {x['all']:.4f} | "
                         f"{x['daytime']:.4f} | {x['night']:.4f} |")
        means = [statistics.mean(values[method, seed][d] for seed in SEEDS)
                 for d in DOMAINS[:4]]
        lines.append(f"| **{method} mean** | — | " + " | ".join(f"**{m:.4f}**" for m in means) + " |")
    for method, baseline in (("adaptive_target_aware", "generic_coverage"),
                             ("rain_count_control", "generic_coverage"),
                             ("adaptive_target_aware", "rain_count_control")):
        lines += ["", f"## {method} minus {baseline}", "",
                  "| Domain | Mean paired difference | Exploratory 95% paired t interval | Positive pairs |",
                  "|---|---:|---:|---:|"]
        for domain in DOMAINS[:4]:
            diffs = [values[method, seed][domain] - values[baseline, seed][domain]
                     for seed in SEEDS]
            mean = statistics.mean(diffs)
            half = T4 * statistics.stdev(diffs) / math.sqrt(5)
            lines.append(f"| {domain} | {mean:+.4f} | [{mean-half:+.4f}, {mean+half:+.4f}] | "
                         f"{sum(d > 0 for d in diffs)}/5 |")
        lines += ["", "Rainy paired differences by seed: " + ", ".join(
            f"{seed}: {values[method, seed]['rainy'] - values[baseline, seed]['rainy']:+.4f}"
            for seed in SEEDS) + "."]
    lines += ["", "The intervals measure training-seed variation on a fixed evaluation set.",
              "They do not account for a new sample of rainy driving scenes. The rainy",
              "development slice contains 78 images and the final rainy slice 532 images."]
    out = Path("RAIN_EXPERIMENT_RESULTS.md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
