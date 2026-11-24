

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

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

# The same weather/scene groups have reliable daytime and night-time validation
# slices. Keeping these groups fixed isolates the effect of time-of-day coverage.
STRATA = (
    ("clear", "city street"),
    ("clear", "highway"),
    ("clear", "residential"),
    ("rainy", "city street"),
    ("snowy", "city street"),
)

# Only this ratio changes between experiments. With the default 1,500 images
# per stratum, these become 1,200/300, 750/750, and 300/1,200.
EXPERIMENT_DAY_FRACTIONS = {
    "day_heavy": 0.80,
    "balanced": 0.50,
    "night_heavy": 0.20,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Folder containing bdd100k_images_100k and bdd100k_labels.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "subsets",
        help="Directory for the manifests and audit tables.",
    )
    parser.add_argument(
        "--per-stratum",
        type=int,
        default=1500,
        help="Total images selected from each weather/scene stratum.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic sampling.",
    )
    return parser.parse_args()


def find_existing_directory(candidates, description):
    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    tried = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Could not find {description}. Tried:\n{tried}")


def find_training_directories(project_root):
    image_dir = find_existing_directory(
        (
            project_root / "bdd100k_images_100k" / "100k" / "train",
            project_root
            / "bdd100k_images_100k"
            / "bdd100k"
            / "images"
            / "100k"
            / "train",
            project_root / "images" / "100k" / "train",
        ),
        "the training image directory",
    )
    label_dir = find_existing_directory(
        (
            project_root / "bdd100k_labels" / "100k" / "train",
            project_root / "bdd100k_labels" / "train",
            project_root / "labels" / "100k" / "train",
        ),
        "the training label directory",
    )
    return image_dir, label_dir


def read_record(label_file):
    with label_file.open("r", encoding="utf-8") as file:
        record = json.load(file)

    if isinstance(record, list):
        if len(record) != 1 or not isinstance(record[0], dict):
            raise ValueError(f"Unexpected JSON structure in {label_file}")
        record = record[0]

    if not isinstance(record, dict):
        raise ValueError(f"Expected a JSON object in {label_file}")
    return record


def metadata_value(attributes, key):
    value = attributes.get(key, "undefined")
    if value is None or str(value).strip() == "":
        return "undefined"
    return str(value).strip().lower()


def read_metadata(record):
    attributes = record.get("attributes", {})
    if not isinstance(attributes, dict):
        attributes = {}
    return (
        metadata_value(attributes, "weather"),
        metadata_value(attributes, "scene"),
        metadata_value(attributes, "timeofday"),
    )


def get_objects(record):
    """Support standard labels and the per-image frames/objects layout."""
    labels = record.get("labels")
    if isinstance(labels, list):
        return labels

    objects = []
    frames = record.get("frames", [])
    if isinstance(frames, list):
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            frame_objects = frame.get("objects", frame.get("labels", []))
            if isinstance(frame_objects, list):
                objects.extend(frame_objects)
    return objects


def has_valid_box(annotation):
    box = annotation.get("box2d")
    if not isinstance(box, dict):
        return False

    try:
        x1 = float(box["x1"])
        y1 = float(box["y1"])
        x2 = float(box["x2"])
        y2 = float(box["y2"])
    except (KeyError, TypeError, ValueError):
        return False

    return x2 > x1 and y2 > y1


def collect_candidate_pools(image_dir, label_dir):
    """Collect image records for the five strata and two time conditions."""
    pool_keys = {
        (weather, scene, timeofday)
        for weather, scene in STRATA
        for timeofday in ("daytime", "night")
    }
    pools = {key: [] for key in pool_keys}

    image_by_stem = {
        image_path.stem: image_path for image_path in image_dir.glob("*.jpg")
    }
    label_files = sorted(label_dir.glob("*.json"))
    if not label_files:
        raise RuntimeError(f"No JSON files found in {label_dir}")

    unreadable = 0
    missing_images = []

    for file_index, label_file in enumerate(label_files, start=1):
        if file_index % 10_000 == 0 or file_index == len(label_files):
            print(
                f"Scanning training metadata: "
                f"{file_index:,}/{len(label_files):,}"
            )
        try:
            record = read_record(label_file)
        except (OSError, json.JSONDecodeError, ValueError):
            unreadable += 1
            continue

        combination = read_metadata(record)
        if combination not in pools:
            continue

        image_path = image_by_stem.get(label_file.stem)
        if image_path is None:
            missing_images.append(label_file.stem)
            continue

        weather, scene, timeofday = combination
        pools[combination].append(
            {
                "image_name": image_path.name,
                "label_name": label_file.name,
                "weather": weather,
                "scene": scene,
                "timeofday": timeofday,
            }
        )

    if unreadable:
        raise RuntimeError(f"Could not read {unreadable} training JSON files")
    if missing_images:
        preview = ", ".join(missing_images[:5])
        raise RuntimeError(
            f"Found {len(missing_images)} selected labels without images: {preview}"
        )

    return pools


def deterministically_shuffle_pools(pools, seed):
    """Shuffle each pool once so experiments use nested/shared samples."""
    shuffled = {}
    for key, records in pools.items():
        records = sorted(records, key=lambda record: record["image_name"])
        weather, scene, timeofday = key
        pool_seed = f"{seed}|{weather}|{scene}|{timeofday}"
        random.Random(pool_seed).shuffle(records)
        shuffled[key] = records
    return shuffled


def required_counts(per_stratum, day_fraction):
    day_count = round(per_stratum * day_fraction)
    night_count = per_stratum - day_count
    return day_count, night_count


def verify_pool_sizes(pools, per_stratum):
    shortages = []
    maximum_needed = max(
        max(required_counts(per_stratum, fraction))
        for fraction in EXPERIMENT_DAY_FRACTIONS.values()
    )

    for weather, scene in STRATA:
        for timeofday in ("daytime", "night"):
            available = len(pools[(weather, scene, timeofday)])
            if available < maximum_needed:
                shortages.append(
                    (weather, scene, timeofday, available, maximum_needed)
                )

    if shortages:
        lines = [
            f"{weather} | {scene} | {timeofday}: "
            f"available={available}, required={required}"
            for weather, scene, timeofday, available, required in shortages
        ]
        raise RuntimeError("Insufficient images for the design:\n" + "\n".join(lines))


def build_subsets(pools, per_stratum):
    subsets = {}

    for experiment, day_fraction in EXPERIMENT_DAY_FRACTIONS.items():
        day_count, night_count = required_counts(per_stratum, day_fraction)
        selected = []

        for weather, scene in STRATA:
            selected.extend(pools[(weather, scene, "daytime")][:day_count])
            selected.extend(pools[(weather, scene, "night")][:night_count])

        selected = sorted(
            selected,
            key=lambda record: (
                record["weather"],
                record["scene"],
                record["timeofday"],
                record["image_name"],
            ),
        )

        names = [record["image_name"] for record in selected]
        if len(names) != len(set(names)):
            raise RuntimeError(f"Duplicate images found inside {experiment}")

        expected_total = per_stratum * len(STRATA)
        if len(selected) != expected_total:
            raise RuntimeError(
                f"{experiment} contains {len(selected)} images; "
                f"expected {expected_total}"
            )
        subsets[experiment] = selected

    return subsets


def write_manifests(subsets, output_dir):
    fieldnames = ("image_name", "label_name", "weather", "scene", "timeofday")
    manifest_paths = {}

    for experiment, records in subsets.items():
        output_file = output_dir / f"{experiment}.csv"
        with output_file.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        manifest_paths[experiment] = output_file

    return manifest_paths


def count_classes_for_selected_images(subsets, label_dir):
    """Read each selected image once and cache its object-class counts."""
    label_names = {
        record["label_name"]
        for records in subsets.values()
        for record in records
    }
    counts_by_label = {}

    sorted_label_names = sorted(label_names)
    for file_index, label_name in enumerate(sorted_label_names, start=1):
        if file_index % 5_000 == 0 or file_index == len(sorted_label_names):
            print(
                f"Auditing selected classes: "
                f"{file_index:,}/{len(sorted_label_names):,}"
            )
        record = read_record(label_dir / label_name)
        counts = Counter()
        for annotation in get_objects(record):
            if not isinstance(annotation, dict):
                continue
            category = annotation.get("category")
            if category in CATEGORIES and has_valid_box(annotation):
                counts[category] += 1
        counts_by_label[label_name] = counts

    return counts_by_label


def write_subset_summary(subsets, counts_by_label, output_file):
    fieldnames = [
        "subset",
        "total_images",
        "daytime_images",
        "night_images",
        "total_boxes",
        *CATEGORIES,
    ]

    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for experiment, records in subsets.items():
            class_counts = Counter()
            for record in records:
                class_counts.update(counts_by_label[record["label_name"]])

            time_counts = Counter(record["timeofday"] for record in records)
            row = {
                "subset": experiment,
                "total_images": len(records),
                "daytime_images": time_counts["daytime"],
                "night_images": time_counts["night"],
                "total_boxes": sum(class_counts.values()),
            }
            row.update({category: class_counts[category] for category in CATEGORIES})
            writer.writerow(row)


def write_stratum_summary(subsets, output_file):
    fieldnames = ("subset", "weather", "scene", "timeofday", "image_count")

    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for experiment, records in subsets.items():
            counts = Counter(
                (record["weather"], record["scene"], record["timeofday"])
                for record in records
            )
            for (weather, scene, timeofday), image_count in sorted(counts.items()):
                writer.writerow(
                    {
                        "subset": experiment,
                        "weather": weather,
                        "scene": scene,
                        "timeofday": timeofday,
                        "image_count": image_count,
                    }
                )


def write_overlap_summary(subsets, output_file):
    names_by_subset = {
        experiment: {record["image_name"] for record in records}
        for experiment, records in subsets.items()
    }

    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ("subset_a", "subset_b", "shared_images", "shared_fraction")
        )

        experiments = list(EXPERIMENT_DAY_FRACTIONS)
        for first_index, first in enumerate(experiments):
            for second in experiments[first_index + 1 :]:
                shared = len(names_by_subset[first] & names_by_subset[second])
                fraction = shared / len(names_by_subset[first])
                writer.writerow((first, second, shared, fraction))


def print_pool_counts(pools):
    print("\nAVAILABLE TRAINING IMAGES")
    print("-" * 72)
    print(f"{'Weather':<12} {'Scene':<18} {'Daytime':>10} {'Night':>10}")
    print("-" * 72)
    for weather, scene in STRATA:
        day_count = len(pools[(weather, scene, "daytime")])
        night_count = len(pools[(weather, scene, "night")])
        print(f"{weather:<12} {scene:<18} {day_count:>10,} {night_count:>10,}")


def print_subset_counts(subsets):
    print("\nCREATED SUBSETS")
    print("-" * 62)
    print(f"{'Subset':<16} {'Total':>10} {'Daytime':>10} {'Night':>10}")
    print("-" * 62)
    for experiment, records in subsets.items():
        times = Counter(record["timeofday"] for record in records)
        print(
            f"{experiment:<16} {len(records):>10,} "
            f"{times['daytime']:>10,} {times['night']:>10,}"
        )


def main():
    args = parse_args()
    if args.per_stratum <= 0:
        raise ValueError("--per-stratum must be positive")

    project_root = args.project_root.resolve()
    image_dir, label_dir = find_training_directories(project_root)
    print(f"Training images: {image_dir}")
    print(f"Training labels: {label_dir}")
    print(f"Sampling seed:   {args.seed}")

    pools = collect_candidate_pools(image_dir, label_dir)
    print_pool_counts(pools)
    verify_pool_sizes(pools, args.per_stratum)
    shuffled_pools = deterministically_shuffle_pools(pools, args.seed)
    subsets = build_subsets(shuffled_pools, args.per_stratum)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_paths = write_manifests(subsets, args.output_dir)

    counts_by_label = count_classes_for_selected_images(subsets, label_dir)
    subset_summary = args.output_dir / "subset_summary.csv"
    stratum_summary = args.output_dir / "subset_stratum_summary.csv"
    overlap_summary = args.output_dir / "subset_overlap.csv"
    write_subset_summary(subsets, counts_by_label, subset_summary)
    write_stratum_summary(subsets, stratum_summary)
    write_overlap_summary(subsets, overlap_summary)

    print_subset_counts(subsets)
    print("\nSaved files:")
    for experiment in EXPERIMENT_DAY_FRACTIONS:
        print(f"  {manifest_paths[experiment].resolve()}")
    for output_file in (subset_summary, stratum_summary, overlap_summary):
        print(f"  {output_file.resolve()}")


if __name__ == "__main__":
    main()
