"""Build equal-budget random, target-only, and coverage-based training subsets."""

import argparse
import csv
import hashlib
import heapq
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np

from src.config import CATEGORIES


def tokens(row):
    weather, scene, time = row["weather"], row["scene"], row["timeofday"]
    result = set()
    for key, value in (("weather", weather), ("scene", scene), ("time", time)):
        if value != "undefined":
            result.add(f"{key}={value}")
    for name, first, second in (("weather|scene", weather, scene),
                                ("weather|time", weather, time),
                                ("scene|time", scene, time)):
        if first != "undefined" and second != "undefined":
            result.add(f"{name}={first}|{second}")
    for category in CATEGORIES:
        if int(row[f"count_{category.replace(' ', '_')}"]) > 0:
            result.add(f"class={category}")
    for size in ("small", "medium", "large"):
        if int(row[f"{size}_box_count"]) > 0:
            result.add(f"box_size={size}")
    return result


def weights_for(rows, token_sets, target, mode, class_gaps=None):
    frequencies = Counter(token for item in token_sets for token in item)
    weights = {}
    for token, frequency in frequencies.items():
        # Rarity normalization prevents ubiquitous tokens from dominating.
        weight = 1.0 / (frequency ** 0.25)
        if mode in ("target_aware", "visual_target_aware", "visual_cluster_aware") and (token == f"time={target}" or
            token.endswith(f"|{target}")):
            weight *= (1.15 if mode == "visual_cluster_aware" else
                       1.3 if mode == "visual_target_aware" else 2.0)
        if mode == "visual_cluster_aware" and token.startswith("query_cluster="):
            weight *= 1.5
        if mode == "adaptive_target_aware" and token.startswith("target_class="):
            category = token.split("=", 1)[1]
            weight *= 1.0 + 12.0 * max(0.0, class_gaps.get(category, 0.0))
        weights[token] = weight
    return weights


def marginal(item_tokens, counts, weights, relevance=0.0):
    return relevance + sum(weights[token] *
                           (math.sqrt(counts[token] + 1) - math.sqrt(counts[token]))
                           for token in item_tokens)


def lazy_greedy(rows, token_sets, budget, target, mode, seed, visual_scores=None,
                class_gaps=None):
    weights = weights_for(rows, token_sets, target, mode, class_gaps)
    counts = Counter()
    chosen = []
    used = set()
    # Target relevance is modular; concave token coverage is submodular.
    relevance_scale = sum(weights.values()) / max(len(weights), 1)
    def relevance(index):
        if mode == "visual_target_aware":
            return relevance_scale * (
                0.01 * (rows[index]["timeofday"] == target) +
                0.04 * float(visual_scores[index]))
        if mode == "visual_cluster_aware":
            return relevance_scale * 0.025 * float(visual_scores[index])
        return 0.05 * relevance_scale if mode == "target_aware" and \
            rows[index]["timeofday"] == target else 0.0
    rng = random.Random(seed)
    tie = list(range(len(rows)))
    rng.shuffle(tie)
    tie_rank = {index: rank for rank, index in enumerate(tie)}
    heap = [(-marginal(item, counts, weights, relevance(i)), tie_rank[i], i)
            for i, item in enumerate(token_sets)]
    heapq.heapify(heap)
    while len(chosen) < budget:
        neg_gain, rank, index = heapq.heappop(heap)
        if index in used:
            continue
        actual = marginal(token_sets[index], counts, weights, relevance(index))
        if heap and actual < -heap[0][0] - 1e-12:
            heapq.heappush(heap, (-actual, rank, index))
            continue
        chosen.append(index)
        used.add(index)
        counts.update(token_sets[index])
    return chosen


def load_visual_features(rows, train_prefix, query_prefix):
    train_meta = json.loads(Path(str(train_prefix) + ".json").read_text(encoding="utf-8"))
    query_meta = json.loads(Path(str(query_prefix) + ".json").read_text(encoding="utf-8"))
    if train_meta["names"] != [row["image_name"] for row in rows]:
        raise ValueError("Training feature order does not match master table")
    if train_meta["weights"] != query_meta["weights"]:
        raise ValueError("Training/query features use different encoders")
    train_vectors = np.load(str(train_prefix) + ".npy", mmap_mode="r")
    query_vectors = np.load(str(query_prefix) + ".npy", mmap_mode="r")
    if train_vectors.shape != (len(rows), 2048) or query_vectors.shape[1] != 2048:
        raise ValueError("Unexpected feature shape")
    return train_vectors, np.asarray(query_vectors, dtype="float32")


def visual_percentiles(rows, train_prefix, query_prefix):
    train_vectors, query_vectors = load_visual_features(rows, train_prefix, query_prefix)
    prototype = query_vectors.mean(axis=0)
    prototype /= np.linalg.norm(prototype)
    scores = np.empty(len(rows), dtype="float32")
    for start in range(0, len(rows), 2048):
        stop = min(start + 2048, len(rows))
        scores[start:stop] = np.asarray(train_vectors[start:stop], dtype="float32") @ prototype
    order = np.argsort(scores, kind="stable")
    percentiles = np.empty(len(rows), dtype="float32")
    percentiles[order] = np.arange(len(rows), dtype="float32") / max(len(rows) - 1, 1)
    return percentiles


def cluster_visual_features(rows, train_prefix, query_prefix, clusters=16):
    train_vectors, query_vectors = load_visual_features(rows, train_prefix, query_prefix)
    # Deterministic spherical k-means. The first center is the query image
    # closest to the mean; subsequent centers use farthest-point seeding.
    query_vectors /= np.linalg.norm(query_vectors, axis=1, keepdims=True)
    center_ids = [int(np.argmax(query_vectors @ query_vectors.mean(axis=0)))]
    while len(center_ids) < clusters:
        current = query_vectors @ query_vectors[center_ids].T
        center_ids.append(int(np.argmin(current.max(axis=1))))
    centers = query_vectors[center_ids].copy()
    for _ in range(20):
        assignments = (query_vectors @ centers.T).argmax(axis=1)
        new_centers = centers.copy()
        for index in range(clusters):
            members = query_vectors[assignments == index]
            if len(members):
                new_centers[index] = members.mean(axis=0)
        new_centers /= np.linalg.norm(new_centers, axis=1, keepdims=True)
        if np.allclose(new_centers, centers, atol=1e-5):
            break
        centers = new_centers
    scores = np.empty(len(rows), dtype="float32")
    closest = np.empty(len(rows), dtype="int16")
    for start in range(0, len(rows), 1024):
        stop = min(start + 1024, len(rows))
        similarities = np.asarray(train_vectors[start:stop], dtype="float32") @ centers.T
        closest[start:stop] = similarities.argmax(axis=1)
        scores[start:stop] = similarities.max(axis=1)
    order = np.argsort(scores, kind="stable")
    percentiles = np.empty(len(rows), dtype="float32")
    percentiles[order] = np.arange(len(rows), dtype="float32") / max(len(rows) - 1, 1)
    return percentiles, closest


def write_manifest(path, rows, indices):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_name"])
        writer.writerows([[rows[i]["image_name"]] for i in sorted(indices,
                         key=lambda i: rows[i]["image_name"])])
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_class_gaps(paths):
    gaps = {category: [] for category in CATEGORIES}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            metrics = {row["domain"]: row for row in csv.DictReader(handle)}
        for category in CATEGORIES:
            key = f"ap_{category.replace(' ', '_')}"
            day, night = metrics["daytime"][key], metrics["night"][key]
            if day and night:
                gaps[category].append(float(day) - float(night))
    return {category: max(0.0, sum(values) / len(values)) if values else 0.0
            for category, values in gaps.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, default=Path("outputs/metadata/master_train.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("research_subsets"))
    parser.add_argument("--target", choices=("night", "daytime"), default="night")
    parser.add_argument("--budget", type=int, default=7500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-features", type=Path, default=None,
                        help="Feature prefix (.npy/.json) for visual target selection.")
    parser.add_argument("--query-features", type=Path, default=None)
    parser.add_argument("--reference-metrics", type=Path, nargs="*", default=None,
                        help="Development-domain metric CSVs from a frozen reference detector.")
    args = parser.parse_args()
    with args.master.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args.budget <= 0 or args.budget > len(rows):
        raise ValueError("Budget must be between 1 and the training pool size")
    names = [row["image_name"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate training image names")
    token_sets = [tokens(row) for row in rows]
    target_indices = [i for i, row in enumerate(rows) if row["timeofday"] == args.target]
    if len(target_indices) < args.budget:
        raise ValueError("Target-only pool is smaller than budget")
    rng = random.Random(args.seed)
    manual_target_count = round(0.8 * args.budget)
    non_target_indices = [i for i, row in enumerate(rows)
                          if row["timeofday"] == "daytime"]
    if args.target == "night" and (len(target_indices) < manual_target_count or
                                  len(non_target_indices) < args.budget - manual_target_count):
        raise ValueError("Not enough day/night images for manual 80% night baseline")
    selections = {
        "random": rng.sample(range(len(rows)), args.budget),
        "target_only": rng.sample(target_indices, args.budget),
    }
    if args.target == "night":
        selections["manual_80pct_night"] = (
            rng.sample(target_indices, manual_target_count) +
            rng.sample(non_target_indices, args.budget - manual_target_count))
    for mode in ("generic_coverage", "target_aware"):
        print(f"Selecting {mode}...", flush=True)
        selections[mode] = lazy_greedy(rows, token_sets, args.budget,
                                        args.target, mode, args.seed)
    if (args.train_features is None) != (args.query_features is None):
        raise ValueError("Provide both train and query features")
    if args.train_features is not None:
        scores = visual_percentiles(rows, args.train_features, args.query_features)
        print("Selecting visual_target_aware...", flush=True)
        selections["visual_target_aware"] = lazy_greedy(
            rows, token_sets, args.budget, args.target, "visual_target_aware",
            args.seed, visual_scores=scores)
        cluster_scores, cluster_ids = cluster_visual_features(
            rows, args.train_features, args.query_features)
        cluster_tokens = [set(item) | {f"query_cluster={int(cluster_ids[i])}"}
                          for i, item in enumerate(token_sets)]
        print("Selecting visual_cluster_aware...", flush=True)
        selections["visual_cluster_aware"] = lazy_greedy(
            rows, cluster_tokens, args.budget, args.target, "visual_cluster_aware",
            args.seed, visual_scores=cluster_scores)
    class_gaps = None
    if args.reference_metrics:
        class_gaps = load_class_gaps(args.reference_metrics)
        interaction_tokens = []
        for row, item in zip(rows, token_sets):
            updated = set(item)
            if row["timeofday"] == args.target:
                for category in CATEGORIES:
                    if int(row[f"count_{category.replace(' ', '_')}"]) > 0:
                        updated.add(f"target_class={category}")
            interaction_tokens.append(updated)
        print("Selecting adaptive_target_aware...", flush=True)
        selections["adaptive_target_aware"] = lazy_greedy(
            rows, interaction_tokens, args.budget, args.target,
            "adaptive_target_aware", args.seed, class_gaps=class_gaps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"target": args.target, "budget": args.budget, "seed": args.seed,
               "master_sha256": hashlib.sha256(args.master.read_bytes()).hexdigest(),
               "train_features": str(args.train_features) if args.train_features else None,
               "query_features": str(args.query_features) if args.query_features else None,
               "reference_metric_files": [str(path) for path in args.reference_metrics] if args.reference_metrics else [],
               "reference_class_gaps": class_gaps,
               "methods": {}}
    for method, indices in selections.items():
        path = args.output_dir / f"{args.target}_{method}_{args.budget}.csv"
        digest = write_manifest(path, rows, indices)
        group = [rows[i] for i in indices]
        summary["methods"][method] = {
            "images": len(group), "sha256": digest,
            "timeofday": dict(Counter(row["timeofday"] for row in group)),
            "weather": dict(Counter(row["weather"] for row in group)),
            "class_image_counts": {category: sum(int(row[f"count_{category.replace(' ', '_')}"]) > 0
                                                 for row in group) for category in CATEGORIES},
        }
    (args.output_dir / f"{args.target}_selection_summary_{args.budget}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({name: item["timeofday"] for name, item in summary["methods"].items()},
                     indent=2))


if __name__ == "__main__":
    main()
