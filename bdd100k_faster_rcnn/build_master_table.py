"""Build a per-image BDD100K detection and metadata table for selection."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from analyze_coverage import CATEGORIES, get_objects, metadata_value, read_record


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(".."))
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/metadata"))
    parser.add_argument("--limit", type=int, default=None, help="Small preflight only")
    return parser.parse_args()


def valid_area(annotation, width, height):
    box = annotation.get("box2d")
    if not isinstance(box, dict):
        return None
    try:
        x1 = max(0.0, min(float(box["x1"]), width))
        y1 = max(0.0, min(float(box["y1"]), height))
        x2 = max(0.0, min(float(box["x2"]), width))
        y2 = max(0.0, min(float(box["y2"]), height))
    except (KeyError, TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (x2 - x1) * (y2 - y1) / (width * height)


def main():
    args = parse_args()
    image_dir = args.project_root / "bdd100k_images_100k" / "100k" / args.split
    label_dir = args.project_root / "bdd100k_labels" / "100k" / args.split
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError("Expected BDD100K images and per-image labels under project root")
    images = sorted(image_dir.glob("*.jpg"))
    if args.limit is not None:
        images = images[:args.limit]
    if not images:
        raise RuntimeError("No images found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    columns = ["image_name", "weather", "scene", "timeofday", "width", "height",
               "object_count", "small_box_count", "medium_box_count", "large_box_count",
               "mean_relative_box_area", *[f"count_{name.replace(' ', '_')}" for name in CATEGORIES]]
    metadata_counts = {key: Counter() for key in ("weather", "scene", "timeofday")}
    class_counts = Counter()
    missing_labels = []
    output = args.output_dir / f"master_{args.split}.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for index, image_path in enumerate(images, 1):
            label_path = label_dir / f"{image_path.stem}.json"
            if not label_path.is_file():
                missing_labels.append(image_path.name)
                continue
            record = read_record(label_path)
            with Image.open(image_path) as image:
                width, height = image.size
            attributes = record.get("attributes") or {}
            row = {"image_name": image_path.name, "width": width, "height": height}
            for key in metadata_counts:
                row[key] = metadata_value(attributes, key)
                metadata_counts[key][row[key]] += 1
            counts = Counter()
            areas = []
            for annotation in get_objects(record):
                category = annotation.get("category")
                if category not in CATEGORIES:
                    continue
                area = valid_area(annotation, width, height)
                if area is None:
                    continue
                counts[category] += 1
                areas.append(area)
            row["object_count"] = len(areas)
            row["small_box_count"] = sum(area < 0.01 for area in areas)
            row["medium_box_count"] = sum(0.01 <= area < 0.1 for area in areas)
            row["large_box_count"] = sum(area >= 0.1 for area in areas)
            row["mean_relative_box_area"] = f"{sum(areas) / len(areas):.8f}" if areas else ""
            for category in CATEGORIES:
                row[f"count_{category.replace(' ', '_')}"] = counts[category]
            class_counts.update(counts)
            writer.writerow(row)
            if index % 10000 == 0:
                print(f"Processed {index:,}/{len(images):,} images")
    summary = {"split": args.split, "images_found": len(images),
               "images_written": len(images) - len(missing_labels),
               "missing_label_count": len(missing_labels),
               "missing_label_examples": missing_labels[:10],
               "metadata_counts": {key: dict(value) for key, value in metadata_counts.items()},
               "class_box_counts": dict(class_counts),
               "total_valid_boxes": sum(class_counts.values())}
    (args.output_dir / f"summary_{args.split}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in
                      ("images_written", "missing_label_count", "total_valid_boxes")}, indent=2))


if __name__ == "__main__":
    main()
