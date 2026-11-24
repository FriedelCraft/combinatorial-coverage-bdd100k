"""Freeze disjoint target-query, development, and final-evaluation manifests."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def ranked(rows, seed, tag):
    return sorted(rows, key=lambda row: hashlib.sha256(
        f"{seed}:{tag}:{row['image_name']}".encode()).digest())


def write_manifest(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_name"])
        writer.writerows([[row["image_name"]] for row in sorted(rows, key=lambda r: r["image_name"])])
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, default=Path("outputs/metadata/master_val.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("research_splits"))
    parser.add_argument("--night-query", type=int, default=400)
    parser.add_argument("--rain-query", type=int, default=100)
    parser.add_argument("--development", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()
    with args.master.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    names = [row["image_name"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("Master table has duplicate image names")
    night = ranked([row for row in rows if row["timeofday"] == "night"], args.seed, "night")
    if len(night) < args.night_query:
        raise ValueError("Not enough night images")
    night_query = night[:args.night_query]
    used = {row["image_name"] for row in night_query}
    rain = ranked([row for row in rows if row["weather"] == "rainy"
                   and row["image_name"] not in used], args.seed, "rain")
    if len(rain) < args.rain_query:
        raise ValueError("Not enough disjoint rainy images")
    rain_query = rain[:args.rain_query]
    used.update(row["image_name"] for row in rain_query)
    remaining = ranked([row for row in rows if row["image_name"] not in used],
                       args.seed, "development")
    if len(remaining) <= args.development:
        raise ValueError("Development set would leave no final evaluation images")
    groups = {"query_night": night_query, "query_rain": rain_query,
              "development": remaining[:args.development],
              "final_evaluation": remaining[args.development:]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"seed": args.seed, "master_sha256": hashlib.sha256(args.master.read_bytes()).hexdigest(),
               "groups": {}}
    for name, group in groups.items():
        path = args.output_dir / f"{name}.csv"
        digest = write_manifest(path, group)
        summary["groups"][name] = {
            "images": len(group), "sha256": digest,
            "timeofday": dict(Counter(row["timeofday"] for row in group)),
            "weather": dict(Counter(row["weather"] for row in group)),
        }
    all_names = [row["image_name"] for group in groups.values() for row in group]
    if len(all_names) != len(set(all_names)) or set(all_names) != set(names):
        raise AssertionError("Research partitions are not disjoint and exhaustive")
    (args.output_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["groups"], indent=2))


if __name__ == "__main__":
    main()
