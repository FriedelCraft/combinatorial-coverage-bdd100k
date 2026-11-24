"""Analyse metadata combinations and object coverage in BDD100K labels."""

import argparse
import csv
import json
from collections import Counter, defaultdict
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Folder containing bdd100k_labels.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        required=True,
        help="Dataset split to analyse.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Optional exact path to the split's label directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "coverage",
        help="Directory in which CSV reports will be saved.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of most frequent combinations to print.",
    )
    return parser.parse_args()


def find_label_dir(project_root, split, labels):
    """Find the directory containing one JSON label file per image."""
    if labels is not None:
        label_dir = labels
        if not label_dir.is_dir():
            raise FileNotFoundError(f"Label directory does not exist: {label_dir}")
        return label_dir

    candidates = (
        project_root / "bdd100k_labels" / "100k" / split,
        project_root / "bdd100k_labels" / split,
        project_root / "labels" / "100k" / split,
    )

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    tried = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"Could not find the {split} label directory. Tried:\n{tried}"
    )


def read_record(label_file):
    """Read one JSON file and return its frame dictionary."""
    with label_file.open("r", encoding="utf-8") as file:
        record = json.load(file)

    # Some converted datasets wrap a single frame dictionary in a list.
    if isinstance(record, list):
        if len(record) != 1 or not isinstance(record[0], dict):
            raise ValueError("Expected one frame dictionary")
        record = record[0]

    if not isinstance(record, dict):
        raise ValueError("Expected a JSON object")

    return record


def get_objects(record):
    """Support both standard BDD labels and the per-image frames/objects layout."""
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


def metadata_value(attributes, key):
    """Return a clean metadata value, using 'undefined' when it is missing."""
    value = attributes.get(key, "undefined")
    if value is None or str(value).strip() == "":
        return "undefined"
    return str(value).strip().lower()


def has_valid_box(annotation):
    """Check that an annotation contains a positive-area box2d."""
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


def write_combination_csv(
    output_file,
    combination_image_counts,
    combination_class_counts,
):
    fieldnames = [
        "weather",
        "scene",
        "timeofday",
        "image_count",
        "total_boxes",
        *CATEGORIES,
    ]

    combinations = sorted(
        combination_image_counts,
        key=lambda combination: (
            -combination_image_counts[combination],
            combination,
        ),
    )

    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for weather, scene, timeofday in combinations:
            combination = (weather, scene, timeofday)
            class_counts = combination_class_counts[combination]
            row = {
                "weather": weather,
                "scene": scene,
                "timeofday": timeofday,
                "image_count": combination_image_counts[combination],
                "total_boxes": sum(class_counts.values()),
            }
            row.update({category: class_counts[category] for category in CATEGORIES})
            writer.writerow(row)


def write_class_csv(output_file, class_counts):
    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("class", "box_count"))
        for category in CATEGORIES:
            writer.writerow((category, class_counts[category]))


def write_metadata_csv(output_file, metadata_counts):
    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("attribute", "value", "image_count"))
        for attribute in ("weather", "scene", "timeofday"):
            for value, count in metadata_counts[attribute].most_common():
                writer.writerow((attribute, value, count))


def print_counter(title, counter):
    print(f"\n{title}")
    print("-" * len(title))
    for value, count in counter.most_common():
        print(f"{value:<24} {count:>8,}")


def main():
    args = parse_args()
    project_root = args.project_root.resolve()
    label_dir = find_label_dir(project_root, args.split, args.labels)
    label_files = sorted(label_dir.glob("*.json"))

    if not label_files:
        raise RuntimeError(f"No JSON files found in {label_dir}")

    metadata_counts = {
        "weather": Counter(),
        "scene": Counter(),
        "timeofday": Counter(),
    }
    class_counts = Counter()
    combination_image_counts = Counter()
    combination_class_counts = defaultdict(Counter)

    readable_files = 0
    unreadable_files = 0
    images_without_detection_boxes = 0
    invalid_detection_boxes = 0

    print(f"Reading {len(label_files):,} JSON files from {label_dir} ...")

    for label_file in label_files:
        try:
            record = read_record(label_file)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            unreadable_files += 1
            print(f"Warning: could not read {label_file.name}: {error}")
            continue

        readable_files += 1
        attributes = record.get("attributes", {})
        if not isinstance(attributes, dict):
            attributes = {}

        weather = metadata_value(attributes, "weather")
        scene = metadata_value(attributes, "scene")
        timeofday = metadata_value(attributes, "timeofday")
        combination = (weather, scene, timeofday)

        metadata_counts["weather"][weather] += 1
        metadata_counts["scene"][scene] += 1
        metadata_counts["timeofday"][timeofday] += 1
        combination_image_counts[combination] += 1

        valid_boxes_in_image = 0
        for annotation in get_objects(record):
            if not isinstance(annotation, dict):
                continue

            category = annotation.get("category")
            if category not in CATEGORIES:
                continue

            if not has_valid_box(annotation):
                invalid_detection_boxes += 1
                continue

            class_counts[category] += 1
            combination_class_counts[combination][category] += 1
            valid_boxes_in_image += 1

        if valid_boxes_in_image == 0:
            images_without_detection_boxes += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combination_file = args.output_dir / f"combinations_{args.split}.csv"
    class_file = args.output_dir / f"class_counts_{args.split}.csv"
    metadata_file = args.output_dir / f"metadata_counts_{args.split}.csv"

    write_combination_csv(
        combination_file,
        combination_image_counts,
        combination_class_counts,
    )
    write_class_csv(class_file, class_counts)
    write_metadata_csv(metadata_file, metadata_counts)

    print("\nSUMMARY")
    print("-------")
    print(f"Split:                          {args.split}")
    print(f"JSON files found:               {len(label_files):,}")
    print(f"Readable JSON files:            {readable_files:,}")
    print(f"Unreadable JSON files:          {unreadable_files:,}")
    print(f"Unique metadata combinations:   {len(combination_image_counts):,}")
    print(f"Valid detection boxes:          {sum(class_counts.values()):,}")
    print(f"Invalid detection boxes:        {invalid_detection_boxes:,}")
    print(f"Images without detection boxes: {images_without_detection_boxes:,}")

    print_counter("WEATHER", metadata_counts["weather"])
    print_counter("SCENE", metadata_counts["scene"])
    print_counter("TIME OF DAY", metadata_counts["timeofday"])
    print_counter("OBJECT CLASSES", class_counts)

    print(f"\nTOP {args.top} METADATA COMBINATIONS")
    print("-" * 88)
    print(
        f"{'Weather':<18} {'Scene':<22} {'Time of day':<16} "
        f"{'Images':>8} {'Boxes':>10}"
    )
    print("-" * 88)

    for combination, image_count in combination_image_counts.most_common(args.top):
        weather, scene, timeofday = combination
        box_count = sum(combination_class_counts[combination].values())
        print(
            f"{weather:<18} {scene:<22} {timeofday:<16} "
            f"{image_count:>8,} {box_count:>10,}"
        )

    print("\nSaved reports:")
    print(f"  {combination_file.resolve()}")
    print(f"  {class_file.resolve()}")
    print(f"  {metadata_file.resolve()}")


if __name__ == "__main__":
    main()
