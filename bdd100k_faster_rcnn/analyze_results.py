"""Relate BDD100K training coverage to validation detection performance."""

import argparse
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.lines import Line2D
    from scipy.stats import binomtest, pearsonr, spearmanr
except ImportError as error:
    raise SystemExit(
        "Missing analysis dependency. Install them with:\n"
        "python -m pip install pandas numpy scipy matplotlib"
    ) from error


KEY_COLUMNS = ["weather", "scene", "timeofday"]

CATEGORIES = (
    "person",
    "car",
    "rider",
    "truck",
    "bus",
    "train",
    "motor",
    "bike",
    "traffic light",
    "traffic sign",
)

# These common classes occur often enough to support fairer comparisons across
# metadata slices. Overall slice mAP can otherwise change merely because a rare
# class is absent from one slice.
COMMON_CLASSES = (
    "person",
    "car",
    "traffic light",
    "traffic sign",
)


def category_slug(category):
    return category.replace(" ", "_").replace("/", "_")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-coverage",
        type=Path,
        default=Path("outputs") / "coverage" / "combinations_train.csv",
        help="CSV produced by analyze_coverage.py for the training split.",
    )
    parser.add_argument(
        "--val-metrics",
        type=Path,
        default=Path("outputs") / "combination_metrics_val.csv",
        help="CSV produced by evaluate_combinations.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "analysis",
        help="Directory for processed CSVs, plots, and the text summary.",
    )
    parser.add_argument(
        "--primary-min-images",
        type=int,
        default=100,
        help="Minimum validation images for a primary combination.",
    )
    parser.add_argument(
        "--exploratory-min-images",
        type=int,
        default=25,
        help="Minimum validation images for an exploratory combination.",
    )
    return parser.parse_args()


def require_columns(frame, required, filename):
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{filename} is missing columns: {missing}")


def load_inputs(train_file, val_file):
    train = pd.read_csv(train_file)
    val = pd.read_csv(val_file)

    train_required = KEY_COLUMNS + ["image_count", "total_boxes", *CATEGORIES]
    val_required = KEY_COLUMNS + [
        "image_count",
        "box_count",
        "map_50_95",
        "map_50",
        "map_75",
        "mar_100",
    ]
    val_required += [f"gt_{category_slug(c)}" for c in CATEGORIES]
    val_required += [f"ap_{category_slug(c)}" for c in CATEGORIES]

    require_columns(train, train_required, train_file)
    require_columns(val, val_required, val_file)

    if train.duplicated(KEY_COLUMNS).any():
        raise ValueError(f"Duplicate metadata combinations found in {train_file}")
    if val.duplicated(KEY_COLUMNS).any():
        raise ValueError(f"Duplicate metadata combinations found in {val_file}")

    return train, val


def reliability_label(image_count, primary_min, exploratory_min):
    if pd.isna(image_count):
        return "no_validation"
    if image_count >= primary_min:
        return "primary"
    if image_count >= exploratory_min:
        return "exploratory"
    return "insufficient"


def join_coverage_and_performance(train, val, primary_min, exploratory_min):
    train_rename = {
        "image_count": "train_image_count",
        "total_boxes": "train_box_count",
    }
    for category in CATEGORIES:
        train_rename[category] = f"train_gt_{category_slug(category)}"

    val_rename = {
        "image_count": "val_image_count",
        "box_count": "val_box_count",
    }
    if "reliability" in val.columns:
        val_rename["reliability"] = "reported_reliability"

    train = train.rename(columns=train_rename)
    val = val.rename(columns=val_rename)

    joined = train.merge(
        val,
        on=KEY_COLUMNS,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    joined["split_presence"] = joined["_merge"].map(
        {
            "both": "both",
            "left_only": "train_only",
            "right_only": "val_only",
        }
    )
    joined = joined.drop(columns="_merge")

    joined["reliability"] = joined["val_image_count"].apply(
        reliability_label,
        args=(primary_min, exploratory_min),
    )

    ap_columns = [f"ap_{category_slug(c)}" for c in CATEGORIES]
    common_ap_columns = [f"ap_{category_slug(c)}" for c in COMMON_CLASSES]
    joined["represented_classes"] = joined[ap_columns].notna().sum(axis=1)
    joined.loc[joined["val_image_count"].isna(), "represented_classes"] = np.nan

    # skipna=False guarantees that the fixed-class metric is calculated only
    # when all four common-class AP values exist.
    joined["common4_map_50_95"] = joined[common_ap_columns].mean(
        axis=1,
        skipna=False,
    )

    presence_order = pd.Categorical(
        joined["split_presence"],
        categories=["both", "val_only", "train_only"],
        ordered=True,
    )
    joined = (
        joined.assign(_presence_order=presence_order)
        .sort_values(
            ["_presence_order", "val_image_count", "train_image_count"],
            ascending=[True, False, False],
            na_position="last",
        )
        .drop(columns="_presence_order")
        .reset_index(drop=True)
    )
    return joined


def correlation_record(scope, predictor, outcome, data):
    columns = [predictor, outcome]
    usable = data[columns].dropna()
    usable = usable[(usable[predictor] > 0) & np.isfinite(usable[outcome])]

    record = {
        "scope": scope,
        "predictor": predictor,
        "outcome": outcome,
        "n": len(usable),
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
        "pearson_log_r": np.nan,
        "pearson_log_p": np.nan,
    }

    if len(usable) < 3 or usable[predictor].nunique() < 2:
        return record

    spearman = spearmanr(usable[predictor], usable[outcome])
    pearson = pearsonr(np.log10(usable[predictor]), usable[outcome])
    record.update(
        {
            "spearman_rho": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
            "pearson_log_r": float(pearson.statistic),
            "pearson_log_p": float(pearson.pvalue),
        }
    )
    return record


def calculate_correlations(joined, primary_min, exploratory_min):
    matched = joined[joined["split_presence"] == "both"]
    scopes = {
        "matched_all": matched,
        "matched_at_least_exploratory": matched[
            matched["val_image_count"] >= exploratory_min
        ],
        "matched_primary": matched[matched["val_image_count"] >= primary_min],
    }
    predictors = ("train_image_count", "train_box_count")
    outcomes = ("map_50_95", "common4_map_50_95")

    records = []
    for scope_name, scope_data in scopes.items():
        for predictor in predictors:
            for outcome in outcomes:
                records.append(
                    correlation_record(scope_name, predictor, outcome, scope_data)
                )
    return pd.DataFrame(records)


def build_day_night_table(joined, primary_min):
    primary = joined[
        (joined["split_presence"] == "both")
        & (joined["val_image_count"] >= primary_min)
    ]

    rows = []
    for (weather, scene), group in primary.groupby(["weather", "scene"]):
        daytime = group[group["timeofday"] == "daytime"]
        night = group[group["timeofday"] == "night"]
        if len(daytime) != 1 or len(night) != 1:
            continue

        day = daytime.iloc[0]
        dark = night.iloc[0]
        if pd.isna(day["common4_map_50_95"]) or pd.isna(
            dark["common4_map_50_95"]
        ):
            continue

        row = {
            "weather": weather,
            "scene": scene,
            "day_val_images": int(day["val_image_count"]),
            "night_val_images": int(dark["val_image_count"]),
            "day_raw_map_50_95": day["map_50_95"],
            "night_raw_map_50_95": dark["map_50_95"],
            "raw_night_minus_day": dark["map_50_95"] - day["map_50_95"],
            "day_common4_map_50_95": day["common4_map_50_95"],
            "night_common4_map_50_95": dark["common4_map_50_95"],
            "common4_night_minus_day": (
                dark["common4_map_50_95"] - day["common4_map_50_95"]
            ),
        }

        for category in COMMON_CLASSES:
            slug = category_slug(category)
            day_ap = day[f"ap_{slug}"]
            night_ap = dark[f"ap_{slug}"]
            row[f"day_ap_{slug}"] = day_ap
            row[f"night_ap_{slug}"] = night_ap
            row[f"night_minus_day_ap_{slug}"] = night_ap - day_ap

        rows.append(row)

    return pd.DataFrame(rows)


def find_correlation(correlations, scope, predictor, outcome):
    selected = correlations[
        (correlations["scope"] == scope)
        & (correlations["predictor"] == predictor)
        & (correlations["outcome"] == outcome)
    ]
    if selected.empty:
        return None
    return selected.iloc[0]


def plot_coverage_vs_performance(joined, correlations, output_file, exploratory_min):
    data = joined[
        (joined["split_presence"] == "both")
        & (joined["val_image_count"] >= exploratory_min)
        & joined["common4_map_50_95"].notna()
    ].copy()

    colors = {
        "daytime": "#E69F00",
        "night": "#1F4E79",
        "dawn/dusk": "#8E44AD",
        "undefined": "#7F8C8D",
    }
    markers = {"primary": "o", "exploratory": "^"}

    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in data.iterrows():
        ax.scatter(
            row["train_image_count"],
            row["common4_map_50_95"],
            color=colors.get(row["timeofday"], "#555555"),
            marker=markers.get(row["reliability"], "o"),
            s=65,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.6,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Training images in metadata combination (log scale)")
    ax.set_ylabel("Validation mAP@0.50:0.95 (fixed four classes)")
    ax.set_title("Training Coverage vs Validation Detection Performance")
    ax.grid(True, which="both", alpha=0.22)

    correlation = find_correlation(
        correlations,
        "matched_at_least_exploratory",
        "train_image_count",
        "common4_map_50_95",
    )
    if correlation is not None:
        ax.text(
            0.02,
            0.97,
            (
                f"Spearman rho = {correlation['spearman_rho']:.3f}\n"
                f"p = {correlation['spearman_p']:.3f}, "
                f"n = {int(correlation['n'])}"
            ),
            transform=ax.transAxes,
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
        )

    time_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=color,
            markeredgecolor="white",
            label=time,
            markersize=8,
        )
        for time, color in colors.items()
        if time in set(data["timeofday"])
    ]
    reliability_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="none",
            color="#555555",
            label=label,
            markersize=8,
        )
        for label, marker in markers.items()
        if label in set(data["reliability"])
    ]
    ax.legend(handles=time_handles + reliability_handles, loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_day_night(day_night, output_file):
    if day_night.empty:
        return False

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(day_night)))

    for (_, row), color in zip(day_night.iterrows(), colors):
        label = f"{row['weather']} + {row['scene']}"
        values = [
            row["day_common4_map_50_95"],
            row["night_common4_map_50_95"],
        ]
        ax.plot(
            [0, 1],
            values,
            marker="o",
            linewidth=2,
            markersize=7,
            color=color,
            label=label,
        )

    ax.set_xticks([0, 1], ["Daytime", "Night"])
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylabel("mAP@0.50:0.95 (fixed four classes)")
    ax.set_title("Matched Daytime vs Night-Time Performance")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def write_summary(joined, correlations, day_night, output_file, primary_min):
    presence = joined["split_presence"].value_counts()
    val_rows = joined[joined["val_image_count"].notna()]
    reliability = val_rows["reliability"].value_counts()
    primary = joined[
        (joined["split_presence"] == "both")
        & (joined["val_image_count"] >= primary_min)
        & joined["common4_map_50_95"].notna()
    ]

    correlation = find_correlation(
        correlations,
        "matched_primary",
        "train_image_count",
        "common4_map_50_95",
    )

    lines = [
        "BDD100K COVERAGE–PERFORMANCE ANALYSIS",
        "====================================",
        "",
        "Combination coverage",
        "--------------------",
        f"Combinations present in both splits: {presence.get('both', 0)}",
        f"Training-only combinations:          {presence.get('train_only', 0)}",
        f"Validation-only combinations:        {presence.get('val_only', 0)}",
        "",
        "Validation reliability",
        "----------------------",
        f"Primary combinations:      {reliability.get('primary', 0)}",
        f"Exploratory combinations:  {reliability.get('exploratory', 0)}",
        f"Insufficient combinations: {reliability.get('insufficient', 0)}",
        "",
        "Fixed-class coverage relationship",
        "---------------------------------",
    ]

    if correlation is not None:
        lines.extend(
            [
                (
                    "Primary training-image coverage vs common-four-class mAP: "
                    f"Spearman rho={correlation['spearman_rho']:.4f}, "
                    f"p={correlation['spearman_p']:.4f}, "
                    f"n={int(correlation['n'])}."
                ),
                "The common-four metric averages person, car, traffic light, "
                "and traffic sign AP in every slice.",
            ]
        )

    lines.extend(
        [
            "",
            "Primary fixed-class performance range",
            "-------------------------------------",
        ]
    )
    if not primary.empty:
        lines.extend(
            [
                f"Mean:   {primary['common4_map_50_95'].mean():.4f}",
                f"Median: {primary['common4_map_50_95'].median():.4f}",
                f"Minimum: {primary['common4_map_50_95'].min():.4f}",
                f"Maximum: {primary['common4_map_50_95'].max():.4f}",
            ]
        )

    lines.extend(["", "Matched daytime vs night-time comparison", "----------------------------------------"])
    if day_night.empty:
        lines.append("No primary daytime/night-time pairs were available.")
    else:
        differences = day_night["common4_night_minus_day"]
        day_mean = day_night["day_common4_map_50_95"].mean()
        night_mean = day_night["night_common4_map_50_95"].mean()
        nonzero = differences[differences != 0]
        negative_count = int((nonzero < 0).sum())
        sign_result = binomtest(negative_count, len(nonzero), 0.5)

        lines.extend(
            [
                f"Matched pairs:                  {len(day_night)}",
                f"Daytime mean common-four mAP:  {day_mean:.4f}",
                f"Night mean common-four mAP:    {night_mean:.4f}",
                f"Mean night-minus-day change:  {differences.mean():.4f}",
                f"Relative mean change:          {(night_mean / day_mean - 1):.2%}",
                f"Pairs lower at night:          {negative_count}/{len(nonzero)}",
                f"Two-sided sign-test p-value:   {sign_result.pvalue:.4f}",
                "",
                "Interpret the paired result descriptively because the number of "
                "matched primary combinations is small.",
            ]
        )

        lines.extend(["", "Average night-minus-day AP by class", "-----------------------------------"])
        for category in COMMON_CLASSES:
            slug = category_slug(category)
            difference = day_night[f"night_minus_day_ap_{slug}"].mean()
            lines.append(f"{category:<16} {difference:+.4f}")

    lines.extend(
        [
            "",
            "Important interpretation note",
            "-----------------------------",
            "Raw slice mAP is not directly comparable when slices contain different "
            "sets of classes. Use common4_map_50_95 or compare the same class across "
            "slices. Treat combinations with fewer than 25 validation images as "
            "coverage gaps rather than stable performance estimates.",
        ]
    )

    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()

    if args.exploratory_min_images > args.primary_min_images:
        raise ValueError(
            "--exploratory-min-images cannot exceed --primary-min-images"
        )

    train, val = load_inputs(args.train_coverage, args.val_metrics)
    joined = join_coverage_and_performance(
        train,
        val,
        args.primary_min_images,
        args.exploratory_min_images,
    )
    correlations = calculate_correlations(
        joined,
        args.primary_min_images,
        args.exploratory_min_images,
    )
    day_night = build_day_night_table(joined, args.primary_min_images)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joined_file = args.output_dir / "joined_coverage_performance.csv"
    correlation_file = args.output_dir / "correlations.csv"
    gaps_file = args.output_dir / "coverage_gaps.csv"
    day_night_file = args.output_dir / "day_night_comparison.csv"
    scatter_file = args.output_dir / "coverage_vs_performance.png"
    day_night_plot_file = args.output_dir / "day_vs_night.png"
    summary_file = args.output_dir / "analysis_summary.txt"

    joined.to_csv(joined_file, index=False)
    correlations.to_csv(correlation_file, index=False)
    joined[joined["split_presence"] != "both"].to_csv(gaps_file, index=False)
    day_night.to_csv(day_night_file, index=False)

    plot_coverage_vs_performance(
        joined,
        correlations,
        scatter_file,
        args.exploratory_min_images,
    )
    day_plot_created = plot_day_night(day_night, day_night_plot_file)
    write_summary(
        joined,
        correlations,
        day_night,
        summary_file,
        args.primary_min_images,
    )

    print(summary_file.read_text(encoding="utf-8"))
    print("Saved analysis outputs:")
    for path in (
        joined_file,
        correlation_file,
        gaps_file,
        day_night_file,
        scatter_file,
        summary_file,
    ):
        print(f"  {path.resolve()}")
    if day_plot_created:
        print(f"  {day_night_plot_file.resolve()}")


if __name__ == "__main__":
    main()