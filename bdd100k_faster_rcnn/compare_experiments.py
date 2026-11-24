"""Compare the BDD100K day-heavy, balanced, and night-heavy experiments.

The main comparison uses a fixed four-class metric in every validation slice:
person, car, traffic light, and traffic sign AP averaged together. This avoids
comparing slice mAP values that may have been calculated over different class
sets.
"""

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COMMON_CLASSES = ("person", "car", "traffic_light", "traffic_sign")

# These are the five weather/scene strata used to build the controlled subsets.
MATCHED_STRATA = (
    ("clear", "city street"),
    ("clear", "highway"),
    ("clear", "residential"),
    ("rainy", "city street"),
    ("snowy", "city street"),
)

CONTROLLED_MODELS = (
    ("day_heavy", "Day-heavy", 0.20),
    ("balanced", "Balanced", 0.50),
    ("night_heavy", "Night-heavy", 0.80),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=Path("outputs") / "combination_metrics_val.csv",
    )
    parser.add_argument(
        "--day-heavy-csv",
        type=Path,
        default=Path("outputs")
        / "experiments"
        / "day_heavy"
        / "combination_metrics_val.csv",
    )
    parser.add_argument(
        "--balanced-csv",
        type=Path,
        default=Path("outputs")
        / "experiments"
        / "balanced"
        / "combination_metrics_val.csv",
    )
    parser.add_argument(
        "--night-heavy-csv",
        type=Path,
        default=Path("outputs")
        / "experiments"
        / "night_heavy"
        / "combination_metrics_val.csv",
    )
    parser.add_argument(
        "--baseline-metrics",
        type=Path,
        default=Path("outputs") / "validation_metrics.txt",
    )
    parser.add_argument(
        "--day-heavy-metrics",
        type=Path,
        default=Path("outputs")
        / "experiments"
        / "day_heavy"
        / "validation_metrics.txt",
    )
    parser.add_argument(
        "--balanced-metrics",
        type=Path,
        default=Path("outputs")
        / "experiments"
        / "balanced"
        / "validation_metrics.txt",
    )
    parser.add_argument(
        "--night-heavy-metrics",
        type=Path,
        default=Path("outputs")
        / "experiments"
        / "night_heavy"
        / "validation_metrics.txt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "experiment_comparison",
    )
    return parser.parse_args()


def read_combination_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f"Combination CSV not found: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    required = {
        "weather",
        "scene",
        "timeofday",
        "image_count",
        "map_50_95",
        *(f"ap_{class_name}" for class_name in COMMON_CLASSES),
    }
    if not rows:
        raise ValueError(f"Combination CSV contains no rows: {path}")

    missing = required.difference(rows[0])
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"{path} is missing required columns: {names}")

    indexed = {}
    for row in rows:
        key = (
            row["weather"].strip().lower(),
            row["scene"].strip().lower(),
            row["timeofday"].strip().lower(),
        )
        if key in indexed:
            raise ValueError(f"Duplicate metadata combination {key} in {path}")
        indexed[key] = row

    return indexed


def required_float(row, column, description):
    value = (row.get(column) or "").strip()
    if not value:
        raise ValueError(f"Missing {column} for {description}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite {column} for {description}")
    return number


def common4_map(row, description):
    values = [
        required_float(row, f"ap_{class_name}", description)
        for class_name in COMMON_CLASSES
    ]
    return sum(values) / len(values)


def sign_test(differences):
    """Return counts and an exact two-sided sign-test p-value."""
    positive = sum(value > 1e-12 for value in differences)
    negative = sum(value < -1e-12 for value in differences)
    ties = len(differences) - positive - negative
    n = positive + negative

    if n == 0:
        return positive, negative, ties, 1.0

    smaller = min(positive, negative)
    tail = sum(math.comb(n, k) for k in range(smaller + 1)) / (2**n)
    return positive, negative, ties, min(1.0, 2 * tail)


def mean(values):
    return sum(values) / len(values)


def read_matched_rows(model_key, display_name, night_fraction, indexed_rows):
    records = []
    for weather, scene in MATCHED_STRATA:
        day_key = (weather, scene, "daytime")
        night_key = (weather, scene, "night")

        if day_key not in indexed_rows or night_key not in indexed_rows:
            raise ValueError(
                f"{display_name} is missing a required pair: {weather}, {scene}"
            )

        day_row = indexed_rows[day_key]
        night_row = indexed_rows[night_key]
        day_description = f"{display_name}: {weather}, {scene}, daytime"
        night_description = f"{display_name}: {weather}, {scene}, night"
        day_map = common4_map(day_row, day_description)
        night_map = common4_map(night_row, night_description)

        record = {
            "model": model_key,
            "model_name": display_name,
            "training_night_fraction": night_fraction,
            "weather": weather,
            "scene": scene,
            "day_images": int(day_row["image_count"]),
            "night_images": int(night_row["image_count"]),
            "day_common4_map_50_95": day_map,
            "night_common4_map_50_95": night_map,
            "night_minus_day": night_map - day_map,
        }

        for class_name in COMMON_CLASSES:
            record[f"day_ap_{class_name}"] = required_float(
                day_row, f"ap_{class_name}", day_description
            )
            record[f"night_ap_{class_name}"] = required_float(
                night_row, f"ap_{class_name}", night_description
            )
            record[f"night_minus_day_ap_{class_name}"] = (
                record[f"night_ap_{class_name}"]
                - record[f"day_ap_{class_name}"]
            )

        records.append(record)

    return records


def summarize_model(records):
    day_values = [row["day_common4_map_50_95"] for row in records]
    night_values = [row["night_common4_map_50_95"] for row in records]
    differences = [row["night_minus_day"] for row in records]
    positive, negative, ties, p_value = sign_test(differences)

    day_mean = mean(day_values)
    night_mean = mean(night_values)
    gap = night_mean - day_mean

    summary = {
        "model": records[0]["model"],
        "model_name": records[0]["model_name"],
        "training_night_fraction": records[0]["training_night_fraction"],
        "matched_strata": len(records),
        "mean_day_common4_map_50_95": day_mean,
        "mean_night_common4_map_50_95": night_mean,
        "night_minus_day": gap,
        "relative_night_minus_day": gap / day_mean if day_mean else float("nan"),
        "night_higher_pairs": positive,
        "day_higher_pairs": negative,
        "tied_pairs": ties,
        "day_night_sign_test_p": p_value,
    }

    for class_name in COMMON_CLASSES:
        summary[f"mean_day_ap_{class_name}"] = mean(
            [row[f"day_ap_{class_name}"] for row in records]
        )
        summary[f"mean_night_ap_{class_name}"] = mean(
            [row[f"night_ap_{class_name}"] for row in records]
        )
        summary[f"night_minus_day_ap_{class_name}"] = mean(
            [row[f"night_minus_day_ap_{class_name}"] for row in records]
        )

    return summary


def compare_models(records_by_model, first_key, second_key, condition):
    """Compare two models on the same five validation strata."""
    value_column = f"{condition}_common4_map_50_95"
    first_rows = records_by_model[first_key]
    second_rows = records_by_model[second_key]
    differences = [
        first[value_column] - second[value_column]
        for first, second in zip(first_rows, second_rows)
    ]
    positive, negative, ties, p_value = sign_test(differences)

    return {
        "first_model": first_key,
        "second_model": second_key,
        "validation_condition": condition,
        "mean_first_minus_second": mean(differences),
        "first_higher_pairs": positive,
        "second_higher_pairs": negative,
        "tied_pairs": ties,
        "two_sided_sign_test_p": p_value,
    }


def parse_overall_metrics(path):
    if not path.is_file():
        return None

    # Windows PowerShell 5 often writes Tee-Object output as UTF-16, whereas
    # PowerShell 7 normally uses UTF-8. Detect both so the same analysis works
    # in either shell.
    data = path.read_bytes()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = data.decode("utf-16", errors="replace")
    elif b"\x00" in data[:200]:
        text = data.decode("utf-16-le", errors="replace")
    else:
        text = data.decode("utf-8-sig", errors="replace")
    patterns = {
        "map_50_95": r"(?m)^[ \t]*mAP@0\.50:0\.95:[ \t]+([0-9.]+)",
        "map_50": r"(?m)^[ \t]*mAP@0\.50:[ \t]+([0-9.]+)",
        "map_75": r"(?m)^[ \t]*mAP@0\.75:[ \t]+([0-9.]+)",
        "mar_100": r"(?m)^[ \t]*mAR@100:[ \t]+([0-9.]+)",
        "micro": (
            r"Overall \(micro\)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)"
        ),
    }

    result = {}
    for key in ("map_50_95", "map_50", "map_75", "mar_100"):
        match = re.search(patterns[key], text)
        if match:
            result[key] = float(match.group(1))

    micro_match = re.search(patterns["micro"], text)
    if micro_match:
        result["micro_precision"] = float(micro_match.group(1))
        result["micro_recall"] = float(micro_match.group(2))
        result["micro_f1"] = float(micro_match.group(3))

    return result or None


def write_csv(path, rows, fieldnames=None):
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_day_night(summary_rows, output_dir):
    controlled = [row for row in summary_rows if row["model"] != "full_baseline"]
    labels = [row["model_name"] for row in controlled]
    day_values = [row["mean_day_common4_map_50_95"] for row in controlled]
    night_values = [row["mean_night_common4_map_50_95"] for row in controlled]

    positions = list(range(len(labels)))
    width = 0.34
    fig, axis = plt.subplots(figsize=(8.2, 5.2))
    axis.bar(
        [value - width / 2 for value in positions],
        day_values,
        width,
        label="Daytime validation",
        color="#e69f00",
    )
    axis.bar(
        [value + width / 2 for value in positions],
        night_values,
        width,
        label="Night validation",
        color="#3572a5",
    )
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Mean common-four mAP@0.50:0.95")
    axis.set_title("Matched Daytime and Night-time Performance")
    axis.set_ylim(0, max(day_values + night_values) * 1.25)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    for x, value in zip(positions, day_values):
        axis.text(x - width / 2, value + 0.003, f"{value:.3f}", ha="center")
    for x, value in zip(positions, night_values):
        axis.text(x + width / 2, value + 0.003, f"{value:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(output_dir / "day_night_common4.png", dpi=180)
    plt.close(fig)


def plot_coverage_tradeoff(summary_rows, output_dir):
    controlled = sorted(
        (row for row in summary_rows if row["model"] != "full_baseline"),
        key=lambda row: row["training_night_fraction"],
    )
    fractions = [row["training_night_fraction"] * 100 for row in controlled]
    day_values = [row["mean_day_common4_map_50_95"] for row in controlled]
    night_values = [row["mean_night_common4_map_50_95"] for row in controlled]

    fig, axis = plt.subplots(figsize=(8.2, 5.2))
    axis.plot(
        fractions,
        day_values,
        marker="o",
        linewidth=2.4,
        label="Daytime validation",
        color="#e69f00",
    )
    axis.plot(
        fractions,
        night_values,
        marker="o",
        linewidth=2.4,
        label="Night validation",
        color="#3572a5",
    )
    axis.set_xlabel("Night images in controlled training set (%)")
    axis.set_ylabel("Mean common-four mAP@0.50:0.95")
    axis.set_title("Training Coverage Trade-off")
    axis.set_xticks(fractions)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "coverage_tradeoff.png", dpi=180)
    plt.close(fig)


def plot_overall_metrics(overall_rows, output_dir):
    usable = [row for row in overall_rows if row.get("map_50_95") is not None]
    if not usable:
        return

    labels = [row["model_name"] for row in usable]
    values = [row["map_50_95"] for row in usable]
    colors = ["#555555", "#e69f00", "#5a9f68", "#3572a5"][: len(values)]

    fig, axis = plt.subplots(figsize=(8.5, 5.2))
    bars = axis.bar(labels, values, color=colors)
    axis.set_ylabel("Overall mAP@0.50:0.95")
    axis.set_title("Overall Validation Performance")
    axis.set_ylim(0, max(values) * 1.25)
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.004,
            f"{value:.3f}",
            ha="center",
        )
    fig.tight_layout()
    fig.savefig(output_dir / "overall_map.png", dpi=180)
    plt.close(fig)


def format_summary(summary_rows, comparison_rows, overall_rows, baseline_used):
    by_model = {row["model"]: row for row in summary_rows}
    lines = [
        "BDD100K CONTROLLED COVERAGE EXPERIMENT",
        "======================================",
        "",
        "Primary metric",
        "--------------",
        "Mean AP across person, car, traffic light, and traffic sign, calculated",
        "within the same five matched weather/scene strata for daytime and night.",
        "",
        "Controlled-model summary",
        "------------------------",
    ]

    for key, display_name, _ in CONTROLLED_MODELS:
        row = by_model[key]
        lines.append(
            f"{display_name:<12} day={row['mean_day_common4_map_50_95']:.4f}, "
            f"night={row['mean_night_common4_map_50_95']:.4f}, "
            f"night-day={row['night_minus_day']:+.4f} "
            f"({row['relative_night_minus_day']:+.2%})"
        )

    lines.extend(
        [
            "",
            "Directional comparisons across the five matched strata",
            "--------------------------------------------------------",
        ]
    )
    for row in comparison_rows:
        lines.append(
            f"{row['first_model']} minus {row['second_model']} on "
            f"{row['validation_condition']}: mean={row['mean_first_minus_second']:+.4f}; "
            f"first higher in {row['first_higher_pairs']}/5; "
            f"two-sided sign-test p={row['two_sided_sign_test_p']:.4f}."
        )

    if baseline_used:
        baseline = by_model["full_baseline"]
        lines.extend(
            [
                "",
                "Full-data reference",
                "-------------------",
                f"Day mean common-four mAP:   {baseline['mean_day_common4_map_50_95']:.4f}",
                f"Night mean common-four mAP: {baseline['mean_night_common4_map_50_95']:.4f}",
                f"Night-minus-day gap:        {baseline['night_minus_day']:+.4f}",
            ]
        )

    if overall_rows:
        lines.extend(["", "Overall validation mAP", "----------------------"])
        for row in overall_rows:
            if row.get("map_50_95") is not None:
                lines.append(f"{row['model_name']:<18} {row['map_50_95']:.4f}")

    lines.extend(
        [
            "",
            "Interpretation",
            "--------------",
            "Increasing night-time training coverage produced an ordered trade-off:",
            "daytime performance decreased while night-time performance increased.",
            "The night-heavy model therefore reduced the day/night performance gap,",
            "but the day-heavy model retained higher daytime and overall accuracy.",
            "",
            "Statistical caution",
            "-------------------",
            "The sign tests use only five matched weather/scene strata. A result of",
            "5/5 in one direction has two-sided p=0.0625, so it is strong descriptive",
            "evidence but does not cross the conventional 0.05 threshold. There was",
            "also only one training run per condition, so random-seed variability has",
            "not been estimated. Object-class frequencies differ between the real",
            "day/night subsets and should be reported as a secondary coverage factor.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = {
        "day_heavy": args.day_heavy_csv,
        "balanced": args.balanced_csv,
        "night_heavy": args.night_heavy_csv,
    }
    records_by_model = {}
    all_matched_rows = []
    summary_rows = []

    for model_key, display_name, night_fraction in CONTROLLED_MODELS:
        indexed = read_combination_csv(csv_paths[model_key])
        matched = read_matched_rows(
            model_key, display_name, night_fraction, indexed
        )
        records_by_model[model_key] = matched
        all_matched_rows.extend(matched)
        summary_rows.append(summarize_model(matched))

    baseline_used = args.baseline_csv.is_file()
    if baseline_used:
        baseline_indexed = read_combination_csv(args.baseline_csv)
        baseline_records = read_matched_rows(
            "full_baseline", "Full 70k baseline", float("nan"), baseline_indexed
        )
        records_by_model["full_baseline"] = baseline_records
        all_matched_rows.extend(baseline_records)
        summary_rows.insert(0, summarize_model(baseline_records))

    comparison_rows = [
        compare_models(records_by_model, "day_heavy", "balanced", "day"),
        compare_models(records_by_model, "day_heavy", "night_heavy", "day"),
        compare_models(records_by_model, "night_heavy", "balanced", "night"),
        compare_models(records_by_model, "night_heavy", "day_heavy", "night"),
    ]

    metric_specs = [
        ("full_baseline", "Full 70k baseline", args.baseline_metrics),
        ("day_heavy", "Day-heavy", args.day_heavy_metrics),
        ("balanced", "Balanced", args.balanced_metrics),
        ("night_heavy", "Night-heavy", args.night_heavy_metrics),
    ]
    overall_rows = []
    missing_metric_files = []
    unparsed_metric_files = []
    for model_key, display_name, path in metric_specs:
        metrics = parse_overall_metrics(path)
        if metrics is None:
            if path.is_file():
                unparsed_metric_files.append(path)
            else:
                missing_metric_files.append(path)
            continue
        overall_rows.append(
            {"model": model_key, "model_name": display_name, **metrics}
        )

    write_csv(args.output_dir / "matched_strata.csv", all_matched_rows)
    write_csv(args.output_dir / "model_summary.csv", summary_rows)
    write_csv(args.output_dir / "direction_tests.csv", comparison_rows)
    if overall_rows:
        write_csv(args.output_dir / "overall_metrics.csv", overall_rows)

    plot_day_night(summary_rows, args.output_dir)
    plot_coverage_tradeoff(summary_rows, args.output_dir)
    plot_overall_metrics(overall_rows, args.output_dir)

    summary_text = format_summary(
        summary_rows, comparison_rows, overall_rows, baseline_used
    )
    (args.output_dir / "comparison_summary.txt").write_text(
        summary_text, encoding="utf-8"
    )

    print(summary_text)
    print("Saved analysis outputs:")
    for path in sorted(args.output_dir.iterdir()):
        print(f"  {path.resolve()}")

    if missing_metric_files:
        print("\nOptional overall-metric files not found:")
        for path in missing_metric_files:
            print(f"  {path}")
        print("Slice-level analysis was still completed.")

    if unparsed_metric_files:
        print("\nOptional overall-metric files could not be parsed:")
        for path in unparsed_metric_files:
            print(f"  {path}")
        print("Check that each file contains the printed COCO-style metrics.")


if __name__ == "__main__":
    main()