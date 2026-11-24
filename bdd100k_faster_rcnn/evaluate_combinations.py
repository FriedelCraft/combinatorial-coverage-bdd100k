"""Evaluate Faster R-CNN for each BDD100K metadata combination."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm

from src.bdd_dataset import BDD100KDataset
from src.config import CATEGORIES, ID_TO_CLASS
from src.model import create_model
from src.utils import choose_device, collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--val-manifest", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--min-size", type=int, default=640)
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument(
        "--primary-min-images",
        type=int,
        default=100,
        help="Minimum images for a primary/reliable combination.",
    )
    parser.add_argument(
        "--exploratory-min-images",
        type=int,
        default=25,
        help="Minimum images for an exploratory combination.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / "combination_metrics_val.csv",
    )
    return parser.parse_args()


def clean_metadata_value(attributes, key):
    """Read one metadata value and normalize missing values."""
    value = attributes.get(key, "undefined")
    if value is None or str(value).strip() == "":
        return "undefined"
    return str(value).strip().lower()


def read_metadata(label_file):
    """Read weather, scene, and time of day from one per-image JSON file."""
    with label_file.open("r", encoding="utf-8") as file:
        record = json.load(file)

    if isinstance(record, list):
        if len(record) != 1 or not isinstance(record[0], dict):
            raise ValueError(f"Unexpected JSON structure in {label_file}")
        record = record[0]

    if not isinstance(record, dict):
        raise ValueError(f"Expected a JSON object in {label_file}")

    attributes = record.get("attributes", {})
    if not isinstance(attributes, dict):
        attributes = {}

    return (
        clean_metadata_value(attributes, "weather"),
        clean_metadata_value(attributes, "scene"),
        clean_metadata_value(attributes, "timeofday"),
    )


def load_all_metadata(dataset):
    """Load metadata in the same order as dataset.image_paths."""
    if not dataset.labels_path.is_dir():
        raise ValueError(
            "This script expects one JSON label file per image. "
            f"Received: {dataset.labels_path}"
        )

    metadata = []
    for image_path in tqdm(dataset.image_paths, desc="Reading metadata"):
        label_file = dataset.labels_path / f"{image_path.stem}.json"
        if not label_file.exists():
            raise FileNotFoundError(f"Missing label file: {label_file}")
        metadata.append(read_metadata(label_file))

    return metadata


def prediction_to_cpu(prediction):
    """Keep only the values required for bounding-box mAP."""
    return {
        "boxes": prediction["boxes"].detach().cpu(),
        "scores": prediction["scores"].detach().cpu(),
        "labels": prediction["labels"].detach().cpu(),
    }


def target_to_cpu(target):
    """Keep target fields understood by TorchMetrics."""
    converted = {
        "boxes": target["boxes"].detach().cpu(),
        "labels": target["labels"].detach().cpu(),
    }

    # These fields are optional, but including them preserves the dataset's
    # area and crowd information for COCO-style evaluation.
    if "area" in target:
        converted["area"] = target["area"].detach().cpu()
    if "iscrowd" in target:
        converted["iscrowd"] = target["iscrowd"].detach().cpu()

    return converted


def create_map_metric():
    return MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        class_metrics=True,
    )


def reliability_label(image_count, primary_min, exploratory_min):
    if image_count >= primary_min:
        return "primary"
    if image_count >= exploratory_min:
        return "exploratory"
    return "insufficient"


def tensor_value(result, key):
    """Convert a one-value result tensor into a normal Python float."""
    return float(result[key].item())


def category_slug(category):
    return category.replace(" ", "_").replace("/", "_")


def build_result_rows(
    metrics,
    image_counts,
    box_counts,
    class_counts,
    primary_min,
    exploratory_min,
):
    rows = []

    combinations = sorted(
        metrics,
        key=lambda combination: (-image_counts[combination], combination),
    )

    for combination in tqdm(combinations, desc="Computing combination mAP"):
        result = metrics[combination].compute()
        weather, scene, timeofday = combination
        image_count = image_counts[combination]

        row = {
            "weather": weather,
            "scene": scene,
            "timeofday": timeofday,
            "reliability": reliability_label(
                image_count,
                primary_min,
                exploratory_min,
            ),
            "image_count": image_count,
            "box_count": box_counts[combination],
            "map_50_95": tensor_value(result, "map"),
            "map_50": tensor_value(result, "map_50"),
            "map_75": tensor_value(result, "map_75"),
            "mar_100": tensor_value(result, "mar_100"),
        }

        ap_by_class = {
            int(class_id): float(class_ap)
            for class_id, class_ap in zip(
                result["classes"].tolist(),
                result["map_per_class"].tolist(),
            )
        }

        for class_id, category in ID_TO_CLASS.items():
            slug = category_slug(category)
            row[f"gt_{slug}"] = class_counts[combination][class_id]

            class_ap = ap_by_class.get(class_id, -1.0)
            # A blank cell means that AP could not be calculated because the
            # combination contained no ground-truth examples of that class.
            row[f"ap_{slug}"] = class_ap if class_ap >= 0 else ""

        rows.append(row)

    return rows


def write_results(output_file, rows):
    output_file.parent.mkdir(parents=True, exist_ok=True)

    metric_fields = [
        "weather",
        "scene",
        "timeofday",
        "reliability",
        "image_count",
        "box_count",
        "map_50_95",
        "map_50",
        "map_75",
        "mar_100",
    ]
    support_fields = [f"gt_{category_slug(category)}" for category in CATEGORIES]
    ap_fields = [f"ap_{category_slug(category)}" for category in CATEGORIES]
    fieldnames = metric_fields + support_fields + ap_fields

    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows, output_file):
    reliability_counts = Counter(row["reliability"] for row in rows)

    print("\nCOMBINATION EVALUATION SUMMARY")
    print("------------------------------")
    print(f"Combinations evaluated: {len(rows):,}")
    print(f"Primary:                {reliability_counts['primary']:,}")
    print(f"Exploratory:            {reliability_counts['exploratory']:,}")
    print(f"Insufficient:           {reliability_counts['insufficient']:,}")

    primary_rows = [row for row in rows if row["reliability"] == "primary"]

    print("\nPRIMARY COMBINATIONS")
    print("-" * 96)
    print(
        f"{'Weather':<18} {'Scene':<22} {'Time':<14} "
        f"{'Images':>8} {'Boxes':>10} {'mAP':>8} {'mAP50':>8}"
    )
    print("-" * 96)

    for row in primary_rows:
        print(
            f"{row['weather']:<18} {row['scene']:<22} "
            f"{row['timeofday']:<14} {row['image_count']:>8,} "
            f"{row['box_count']:>10,} {row['map_50_95']:>8.4f} "
            f"{row['map_50']:>8.4f}"
        )

    print(f"\nSaved complete results to {output_file.resolve()}")


def main():
    args = parse_args()

    if args.exploratory_min_images > args.primary_min_images:
        raise ValueError(
            "--exploratory-min-images cannot be greater than "
            "--primary-min-images"
        )

    device = choose_device()
    print(f"Using device: {device}")

    dataset = BDD100KDataset(
        args.project_root,
        "val",
        labels_path=args.labels,
        manifest_path=args.val_manifest,
        augment=False,
        max_images=args.max_images,
    )
    print(f"Validation images: {len(dataset):,}")

    metadata_by_index = load_all_metadata(dataset)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    model = create_model(
        pretrained=False,
        min_size=args.min_size,
        max_size=args.max_size,
    ).to(device)
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    metrics = {}
    image_counts = Counter()
    box_counts = Counter()
    class_counts = defaultdict(Counter)

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Running inference"):
            predictions = model([image.to(device) for image in images])

            for prediction, target in zip(predictions, targets):
                image_index = int(target["image_id"].item())
                combination = metadata_by_index[image_index]

                if combination not in metrics:
                    metrics[combination] = create_map_metric()

                prediction_cpu = prediction_to_cpu(prediction)
                target_cpu = target_to_cpu(target)
                metrics[combination].update([prediction_cpu], [target_cpu])

                image_counts[combination] += 1
                box_counts[combination] += len(target_cpu["boxes"])
                class_counts[combination].update(
                    int(class_id) for class_id in target_cpu["labels"].tolist()
                )

    rows = build_result_rows(
        metrics,
        image_counts,
        box_counts,
        class_counts,
        args.primary_min_images,
        args.exploratory_min_images,
    )
    write_results(args.output, rows)
    print_summary(rows, args.output)


if __name__ == "__main__":
    main()
