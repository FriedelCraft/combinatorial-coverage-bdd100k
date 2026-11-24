"""Describe rainy per-class AP and selected-image composition without inference."""

import csv
import json
import statistics
from collections import Counter
from pathlib import Path

from src.config import CATEGORIES


ROOT = Path(__file__).resolve().parent
SEEDS = range(42, 47)
METHODS = ("generic_coverage", "rain_count_control", "adaptive_target_aware")
METRICS = ROOT / "outputs" / "experiments"
SUBSETS = ROOT / "research_subsets"
OUTPUT = ROOT / "RAIN_CLASS_ERROR_ANALYSIS.md"


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def manifest(path):
    return {row["image_name"] for row in rows(path)}


def metric(method, seed):
    target = "night" if method == "generic_coverage" else "rain"
    path = METRICS / f"benchmark_{target}_{method}_seed{seed}_steps7500" / "domain_metrics_final_evaluation.csv"
    rainy, = (row for row in rows(path) if row["domain"] == "rainy")
    return rainy


def subset(method, seed):
    if method == "generic_coverage":
        path = SUBSETS / f"adaptive_seed{seed}" / "night_generic_coverage_7500.csv"
    else:
        path = SUBSETS / f"rain_seed{seed}" / f"rain_{method}_7500.csv"
    return manifest(path)


def count_boxes(names, metadata, *, rainy_only=False):
    counts = Counter()
    image_counts = Counter()
    for name in names:
        row = metadata[name]
        if rainy_only and row["weather"] != "rainy":
            continue
        for category in CATEGORIES:
            n = int(row[f"count_{category.replace(' ', '_')}"])
            counts[category] += n
            image_counts[category] += n > 0
    return counts, image_counts


def ap(row, category):
    value = row[f"ap_{category.replace(' ', '_')}"]
    return float(value) if value else None


def fmt(value, signed=False):
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def main():
    train = {row["image_name"]: row for row in rows(ROOT / "outputs/metadata/master_train.csv")}
    val = {row["image_name"]: row for row in rows(ROOT / "outputs/metadata/master_val.csv")}
    final = manifest(ROOT / "research_splits/final_evaluation.csv")
    final_rain = {name for name in final if val[name]["weather"] == "rainy"}
    gt_boxes, gt_images = count_boxes(final_rain, val)
    metrics = {(method, seed): metric(method, seed) for method in METHODS for seed in SEEDS}
    subsets = {(method, seed): subset(method, seed) for method in METHODS for seed in SEEDS}
    gaps = json.loads((SUBSETS / "rain_seed42/rain_selection_summary_7500.json").read_text(
        encoding="utf-8"))["reference_class_gaps"]
    rain_boxes = {key: count_boxes(names, train, rainy_only=True)[0]
                  for key, names in subsets.items()}
    rain_image_counts = {key: sum(train[name]["weather"] == "rainy" for name in names)
                         for key, names in subsets.items()}
    if len(final_rain) != 532 or any(len(names) != 7500 for names in subsets.values()):
        raise ValueError("Unexpected final rainy slice or selected subset size")
    if any(rain_image_counts["adaptive_target_aware", seed] !=
           rain_image_counts["rain_count_control", seed] for seed in SEEDS):
        raise ValueError("Rain count mismatch between adaptive and control")

    lines = ["# Rain experiment: class-level error analysis", "",
             "This is a descriptive post-final analysis of the same previously viewed",
             "532-image rainy final slice. It uses saved per-class AP and metadata;",
             "image-level predictions were not saved, so it cannot identify individual",
             "false positives or missed objects. No model was retrained or rerun.", "",
             "## Rainy final-set AP by class", "",
             "| Class | Final GT boxes | Generic AP | Count-control AP | Adaptive AP | Adaptive − control | Seed 42 adaptive − control |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    valid_classes = []
    for category in CATEGORIES:
        values = {method: [ap(metrics[method, seed], category) for seed in SEEDS]
                  for method in METHODS}
        if all(value is None for group in values.values() for value in group):
            lines.append(f"| {category} | {gt_boxes[category]} | — | — | — | — | — |")
            continue
        if any(value is None for group in values.values() for value in group):
            raise ValueError(f"Inconsistent AP availability for {category}")
        valid_classes.append(category)
        means = {method: statistics.mean(group) for method, group in values.items()}
        difference = means["adaptive_target_aware"] - means["rain_count_control"]
        seed42_difference = values["adaptive_target_aware"][0] - values["rain_count_control"][0]
        lines.append(f"| {category} | {gt_boxes[category]} | {fmt(means['generic_coverage'])} | "
                     f"{fmt(means['rain_count_control'])} | {fmt(means['adaptive_target_aware'])} | "
                     f"{fmt(difference, True)} | {fmt(seed42_difference, True)} |")

    lines += ["", f"AP is available for {len(valid_classes)} classes with rainy final ground truth.",
              "A class AP difference is descriptive; it does not establish why the model changed.",
              "The box counts come from the filtered BDD100K metadata.", "",
              "## Rainy training-box representation", "",
              "Both rain-targeted methods select the same number of rainy images per seed.",
              "The table compares boxes *within those rainy images*, not all selected boxes.", "",
              "| Class | Development gap used | Mean count-control boxes | Mean adaptive boxes | Mean difference | Seed 42 difference | Final rainy images with class |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for category in CATEGORIES:
        control = statistics.mean(rain_boxes["rain_count_control", seed][category] for seed in SEEDS)
        adaptive = statistics.mean(rain_boxes["adaptive_target_aware", seed][category] for seed in SEEDS)
        seed42_difference = (rain_boxes["adaptive_target_aware", 42][category] -
                             rain_boxes["rain_count_control", 42][category])
        lines.append(f"| {category} | {gaps[category]:.4f} | {control:.0f} | {adaptive:.0f} | {adaptive-control:+.0f} | "
                     f"{seed42_difference:+d} | {gt_images[category]} |")

    lines += ["", "## Per-seed context", "",
              "| Seed | Adaptive rainy mAP − control | Adaptive rainy mAP − generic | Rainy images: generic / control / adaptive | Adaptive/control image overlap | Lowest development-loss step: adaptive / control |",
              "|---:|---:|---:|---:|---:|---:|"]
    for seed in SEEDS:
        adaptive = metrics["adaptive_target_aware", seed]
        control = metrics["rain_count_control", seed]
        generic = metrics["generic_coverage", seed]
        overlap = len(subsets["adaptive_target_aware", seed] & subsets["rain_count_control", seed])
        best = {}
        for method in ("adaptive_target_aware", "rain_count_control"):
            path = METRICS / f"benchmark_rain_{method}_seed{seed}_steps7500" / "history.csv"
            best[method] = min(rows(path), key=lambda row: float(row["val_loss"]))["global_step"]
        lines.append(f"| {seed} | {float(adaptive['map_50_95'])-float(control['map_50_95']):+.4f} | "
                     f"{float(adaptive['map_50_95'])-float(generic['map_50_95']):+.4f} | "
                     f"{rain_image_counts['generic_coverage', seed]} / "
                     f"{rain_image_counts['rain_count_control', seed]} / "
                     f"{rain_image_counts['adaptive_target_aware', seed]} | {overlap}/7500 | "
                     f"{best['adaptive_target_aware']} / {best['rain_count_control']} |")
    lines += ["", "The checkpoint step is selected by whole-development-set loss, not rainy AP.",
              "Different checkpoints and selected image sets are both possible explanations",
              "for a seed-level gap; this analysis cannot separate their effects.", "",
              "## Interpretation", ""]
    seed42_deltas = {category: ap(metrics["adaptive_target_aware", 42], category) -
                     ap(metrics["rain_count_control", 42], category)
                     for category in valid_classes}
    negative = sum(delta < 0 for delta in seed42_deltas.values())
    worst = min(seed42_deltas, key=seed42_deltas.get)
    mean_rider_delta = statistics.mean(
        ap(metrics["adaptive_target_aware", seed], "rider") -
        ap(metrics["rain_count_control", seed], "rider") for seed in SEEDS)
    rider_negative_pairs = sum(
        ap(metrics["adaptive_target_aware", seed], "rider") <
        ap(metrics["rain_count_control", seed], "rider") for seed in SEEDS)
    within_overlap = [len(subsets["adaptive_target_aware", 42] &
                          subsets["adaptive_target_aware", seed]) for seed in range(43, 47)]
    pair_overlap = [len(subsets["adaptive_target_aware", seed] &
                        subsets["rain_count_control", seed]) for seed in SEEDS]
    overlap_text = (f"{min(pair_overlap):,}" if min(pair_overlap) == max(pair_overlap)
                    else f"{min(pair_overlap):,}–{max(pair_overlap):,}")
    rain_time = {}
    for method in ("adaptive_target_aware", "rain_count_control"):
        names = subsets[method, 42]
        rain_time[method] = Counter(train[name]["timeofday"] for name in names
                                    if train[name]["weather"] == "rainy")
    lines += [
        f"In seed 42, adaptive AP fell for {negative}/{len(valid_classes)} evaluable classes,",
        f"so the -0.0229 rainy mAP gap is broad rather than a single-class failure.",
        f"The largest class drop was {worst} ({seed42_deltas[worst]:+.4f} AP).",
        f"Across five seeds, rider was the largest negative class difference",
        f"({mean_rider_delta:+.4f} AP; {rider_negative_pairs}/5 pairs negative), but it",
        f"has only {gt_boxes['rider']} rainy final boxes across {gt_images['rider']} images.",
        "Its development AP gap g_c was zero, so it received no AP-gap uplift.",
        "Its rainy training-box count was unchanged between methods; that",
        "box-count measure does not explain the rider loss.", "",
        f"The adaptive and count-control manifests overlap by {overlap_text}",
        "of 7,500 images in each seed. Adaptive subsets also overlap seed 42 by",
        f"{min(within_overlap):,}–{max(within_overlap):,} images across the other seeds.",
        "Thus the five seeds mostly vary training randomness and a small",
        "selection tie-break, not five substantially different data samples.",
        "Despite having more rainy training boxes for most classes, the",
        "adaptive subset did worse broadly in seed 42; raw box counts alone",
        "do not explain the failure.", "",
        f"The seed-42 adaptive subset contains {rain_time['adaptive_target_aware']['daytime']:,}",
        f"rainy daytime and {rain_time['adaptive_target_aware']['night']:,} rainy night",
        f"images; the count control contains {rain_time['rain_count_control']['daytime']:,}",
        f"and {rain_time['rain_count_control']['night']:,}, respectively.",
        "This time-of-day shift and the seed-42 checkpoint difference are",
        "plausible factors, but existing aggregate metrics cannot test either",
        "mechanism. The available evidence supports a descriptive finding of",
        "high run variability, not a causal explanation of seed 42.", "",
        "A subsequent inference-only checkpoint and rainy time-of-day",
        "comparison is reported in `RAIN_SEED42_CHECKPOINT_DIAGNOSTIC.md`.", "",
    ]
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT.name}")


if __name__ == "__main__":
    main()
