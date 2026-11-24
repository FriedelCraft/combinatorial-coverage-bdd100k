"""Select generic metadata coverage with the adaptive method's night count.

This is a post-final, exploratory ablation. It controls the night-image count
without using development class-gap weights or final-evaluation labels.
"""

import argparse
import csv
import heapq
import json
import random
from collections import Counter
from pathlib import Path

from select_subsets import marginal, tokens, weights_for, write_manifest


def select(rows, budget, night_quota, seed, target_field="timeofday", target_value="night"):
    token_sets = [tokens(row) for row in rows]
    weights = weights_for(rows, token_sets, "night", "generic_coverage")
    counts = Counter()
    used = set()
    chosen = []
    rng = random.Random(seed)
    ties = list(range(len(rows)))
    rng.shuffle(ties)
    tie_rank = {index: rank for rank, index in enumerate(ties)}
    heap = [(-marginal(item, counts, weights), tie_rank[i], i)
            for i, item in enumerate(token_sets)]
    heapq.heapify(heap)
    selected_night = 0
    non_night_quota = budget - night_quota
    while len(chosen) < budget:
        if not heap:
            raise RuntimeError("Candidate heap exhausted before meeting quota")
        _, rank, index = heapq.heappop(heap)
        if index in used:
            continue
        is_night = rows[index][target_field] == target_value
        if (is_night and selected_night >= night_quota) or (
                not is_night and len(chosen) - selected_night >= non_night_quota):
            continue
        actual = marginal(token_sets[index], counts, weights)
        if heap and actual < -heap[0][0] - 1e-12:
            heapq.heappush(heap, (-actual, rank, index))
            continue
        chosen.append(index)
        used.add(index)
        counts.update(token_sets[index])
        selected_night += is_night
    if selected_night != night_quota:
        raise AssertionError(f"Night quota mismatch: {selected_night} != {night_quota}")
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--master", type=Path, default=Path("outputs/metadata/master_train.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("research_subsets"))
    args = parser.parse_args()
    adaptive_summary = (args.output_root / f"adaptive_seed{args.seed}"
                        / "night_selection_summary_7500.json")
    summary = json.loads(adaptive_summary.read_text(encoding="utf-8"))
    budget = int(summary["budget"])
    night_quota = int(summary["methods"]["adaptive_target_aware"]["timeofday"]["night"])
    with args.master.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if sum(row["timeofday"] == "night" for row in rows) < night_quota:
        raise ValueError("Not enough night candidates")
    if sum(row["timeofday"] != "night" for row in rows) < budget - night_quota:
        raise ValueError("Not enough non-night candidates")
    indices = select(rows, budget, night_quota, args.seed)
    output_dir = args.output_root / f"night_count_control_seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / f"night_night_count_control_{budget}.csv"
    digest = write_manifest(manifest, rows, indices)
    metadata = {"seed": args.seed, "budget": budget, "night_quota": night_quota,
                "night_selected": sum(rows[i]["timeofday"] == "night" for i in indices),
                "sha256": digest, "method": "generic coverage with exact night count",
                "status": "post-final exploratory ablation"}
    (output_dir / "control_summary.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest} ({night_quota} night, SHA-256 {digest})")


if __name__ == "__main__":
    main()
