"""Summarize the exploratory seed-42 rain/time checkpoint inference."""

import csv
from pathlib import Path

from src.config import CATEGORIES


ROOT = Path(__file__).resolve().parent
METRICS = ROOT / "outputs/analysis/rain_seed42_checkpoint_diagnostic/rain_time_checkpoint_metrics.csv"
OUTPUT = ROOT / "RAIN_SEED42_CHECKPOINT_DIAGNOSTIC.md"
GROUPS = ("rainy_all", "rainy_daytime", "rainy_night", "rainy_dawn_dusk")
COUNT = {"rainy_all": 532, "rainy_daytime": 292, "rainy_night": 198,
         "rainy_dawn_dusk": 42}


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def value(row, key="map_50_95"):
    return float(row[key]) if row[key] else None


def main():
    records = rows(METRICS)
    data = {(row["method"], int(row["step"]), row["group"]): row for row in records}
    if len(records) != 12 or len(data) != 12:
        raise ValueError("Expected three checkpoints and four groups each")
    for group in GROUPS:
        for method, step in (("adaptive_target_aware", 3750),
                             ("adaptive_target_aware", 7500),
                             ("rain_count_control", 7500)):
            if int(data[method, step, group]["images"]) != COUNT[group]:
                raise ValueError(f"Unexpected image count: {method} {step} {group}")
    for method in ("adaptive_target_aware", "rain_count_control"):
        original = next(row for row in rows(
            ROOT / "outputs/experiments" /
            f"benchmark_rain_{method}_seed42_steps7500" /
            "domain_metrics_final_evaluation.csv") if row["domain"] == "rainy")
        actual = data[method, 3750 if method == "adaptive_target_aware" else 7500,
                      "rainy_all"]
        if abs(value(original) - value(actual)) > 0.00001:
            raise ValueError(f"Recomputed rainy AP disagrees with frozen result: {method}")

    lines = ["# Seed 42 rain/time and checkpoint diagnostic", "",
             "This is an exploratory **post-final** diagnostic on the already viewed",
             "532-image rainy final slice. The original result used the checkpoint",
             "with lower whole-development-set loss. This comparison does not change",
             "that frozen checkpoint rule or any result in `RAIN_EXPERIMENT_RESULTS.md`.", "",
             "| Rainy final slice | Images | Adaptive step 3,750 | Adaptive step 7,500 | Count control step 7,500 | Adaptive checkpoint gain | Adaptive step 7,500 − control |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for group in GROUPS:
        early = value(data["adaptive_target_aware", 3750, group])
        late = value(data["adaptive_target_aware", 7500, group])
        control = value(data["rain_count_control", 7500, group])
        name = {"rainy_all": "All rainy", "rainy_daytime": "Rainy daytime",
                "rainy_night": "Rainy night", "rainy_dawn_dusk": "Rainy dawn/dusk"}[group]
        lines.append(f"| {name} | {COUNT[group]} | {early:.4f} | {late:.4f} | "
                     f"{control:.4f} | {late-early:+.4f} | {late-control:+.4f} |")

    early_all = value(data["adaptive_target_aware", 3750, "rainy_all"])
    late_all = value(data["adaptive_target_aware", 7500, "rainy_all"])
    control_all = value(data["rain_count_control", 7500, "rainy_all"])
    gap_closed = (late_all - early_all) / (control_all - early_all)
    rider_key = "ap_rider"
    lines += ["", "## What the comparison shows", "",
              f"The saved step-7,500 adaptive checkpoint raises seed-42 rainy mAP by {late_all-early_all:+.4f}",
              "compared with its selected step-3,750 checkpoint.",
              f"This closes {gap_closed:.0%} of the original {control_all-early_all:.4f} gap to the",
              f"count control; {control_all-late_all:.4f} remains.",
              "The selected step-3,750 checkpoint had lower whole-development",
              "loss (1.1121 versus 1.1243), which is why the frozen rule chose it.",
              "Selecting step 7,500 now because of these final-set scores would",
              "invalidate the original evaluation protocol.", "",
              "At step 7,500, adaptive slightly exceeds the control on rainy",
              "daytime images but remains lower on rainy night images. The",
              "adaptive seed-42 training subset has 1,011 rainy night images",
              "versus 1,146 in the count control, and 1,513 rainy daytime",
              "images versus 1,371. This composition shift is compatible with",
              "the nighttime gap, but these results cannot show that it caused it.",
              "The rainy dawn/dusk slice has only 42 images.", "",
              f"Rider AP on all rainy images rises from {value(data['adaptive_target_aware', 3750, 'rainy_all'], rider_key):.4f}",
              f"to {value(data['adaptive_target_aware', 7500, 'rainy_all'], rider_key):.4f} between adaptive checkpoints,",
              f"but the count control remains at {value(data['rain_count_control', 7500, 'rainy_all'], rider_key):.4f}.",
              "The class has only 27 ground-truth boxes in this rainy slice.", "",
              "## Per-class rainy time-of-day AP", "",
              "These are separate COCO AP calculations for each slice. Overall",
              "rainy mAP is not the image-count-weighted average of the slices.", "",
              "| Class | Day: adaptive 3,750 | Day: adaptive 7,500 | Day: control | Night: adaptive 3,750 | Night: adaptive 7,500 | Night: control |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for category in CATEGORIES:
        key = f"ap_{category.replace(' ', '_')}"
        entries = [value(data[method, step, group], key)
                   for group in ("rainy_daytime", "rainy_night")
                   for method, step in (("adaptive_target_aware", 3750),
                                        ("adaptive_target_aware", 7500),
                                        ("rain_count_control", 7500))]
        cells = ["—" if entry is None else f"{entry:.4f}" for entry in entries]
        lines.append(f"| {category} | " + " | ".join(cells) + " |")
    lines += ["", "The first and third `All rainy` scores reproduce the stored",
              "final-evaluation mAP within 0.00001. Evaluation code:",
              "`evaluate_rain_time_diagnostic.py`; metric CSV:",
              "`outputs/analysis/rain_seed42_checkpoint_diagnostic/rain_time_checkpoint_metrics.csv`.",
              "No training was performed.", ""]
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT.name}")


if __name__ == "__main__":
    main()
