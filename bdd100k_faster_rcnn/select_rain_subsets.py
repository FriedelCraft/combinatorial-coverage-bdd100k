"""Build rain-adaptive and rain-count control manifests for one paired seed."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from select_night_count_control import select as quota_select
from select_subsets import lazy_greedy, tokens, write_manifest
from src.config import CATEGORIES


def rain_class_gaps(paths):
    gaps = {category: [] for category in CATEGORIES}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            data = {row["domain"]: row for row in csv.DictReader(handle)}
        if "rainy" not in data or "nonrain" not in data:
            raise ValueError(f"Rain/nonrain development AP missing: {path}")
        for category in CATEGORIES:
            key = f"ap_{category.replace(' ', '_')}"
            rain, nonrain = data["rainy"][key], data["nonrain"][key]
            if rain and nonrain:
                gaps[category].append(float(nonrain) - float(rain))
    return {category: max(0.0, sum(values) / len(values)) if values else 0.0
            for category, values in gaps.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--master", type=Path, default=Path("outputs/metadata/master_train.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("research_subsets"))
    parser.add_argument("--runs-root", type=Path, default=Path("outputs/experiments"))
    parser.add_argument("--budget", type=int, default=7500)
    args = parser.parse_args()
    if args.budget != 7500:
        raise ValueError("This protocol freezes the training-image budget at 7500")
    with args.master.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 70000 or len({row["image_name"] for row in rows}) != len(rows):
        raise ValueError("Unexpected training pool or duplicate image names")
    refs = [args.runs_root / f"benchmark_night_generic_coverage_seed{s}_steps7500"
            / "domain_metrics_development_with_nonrain.csv" for s in (42, 43, 44)]
    gaps = rain_class_gaps(refs)
    token_sets = [tokens(row) for row in rows]
    augmented = []
    for row, item in zip(rows, token_sets):
        updated = set(item)
        if row["weather"] == "rainy":
            for category in CATEGORIES:
                if int(row[f"count_{category.replace(' ', '_')}"]) > 0:
                    updated.add(f"target_class={category}")
        augmented.append(updated)
    adaptive = lazy_greedy(rows, augmented, args.budget, "rainy",
                           "adaptive_target_aware", args.seed, class_gaps=gaps)
    rainy_count = sum(rows[i]["weather"] == "rainy" for i in adaptive)
    control = quota_select(rows, args.budget, rainy_count, args.seed,
                           target_field="weather", target_value="rainy")
    out = args.output_root / f"rain_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    names = {"adaptive_target_aware": adaptive, "rain_count_control": control}
    summary = {"target": "rainy", "target_field": "weather", "seed": args.seed,
               "budget": args.budget, "rainy_quota": rainy_count,
               "master_sha256": hashlib.sha256(args.master.read_bytes()).hexdigest(),
               "reference_development_metrics": [
                   {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for path in refs],
               "reference_class_gaps": gaps, "gap_coefficient": 12.0,
               "methods": {}}
    for method, indices in names.items():
        path = out / f"rain_{method}_{args.budget}.csv"
        digest = write_manifest(path, rows, indices)
        selected = [rows[i] for i in indices]
        summary["methods"][method] = {
            "images": len(indices), "sha256": digest,
            "weather": dict(Counter(row["weather"] for row in selected)),
            "timeofday": dict(Counter(row["timeofday"] for row in selected))}
    if (summary["methods"]["adaptive_target_aware"]["weather"].get("rainy", 0)
            != summary["methods"]["rain_count_control"]["weather"].get("rainy", 0)):
        raise AssertionError("Rain-count control does not match adaptive count")
    (out / "rain_selection_summary_7500.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Seed {args.seed}: {rainy_count} rainy images in both subsets")


if __name__ == "__main__":
    main()
