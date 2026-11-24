"""Evaluate a checkpoint using precision, recall, and F1 at a chosen IoU."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from tqdm import tqdm

from src.bdd_dataset import BDD100KDataset
from src.config import CATEGORIES, ID_TO_CLASS
from src.model import create_model
from src.utils import choose_device, collate_fn
from torchmetrics.detection import MeanAveragePrecision

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--val-manifest", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--min-size", type=int, default=640)
    parser.add_argument("--max-size", type=int, default=1024)
    return parser.parse_args()


def match_one_image(prediction, target, score_threshold, iou_threshold):
    """Greedily match predictions to same-class ground-truth boxes."""
    keep = prediction["scores"] >= score_threshold
    pred_boxes = prediction["boxes"][keep].cpu()
    pred_labels = prediction["labels"][keep].cpu()
    pred_scores = prediction["scores"][keep].cpu()

    true_boxes = target["boxes"].cpu()
    true_labels = target["labels"].cpu()

    order = pred_scores.argsort(descending=True)
    matched_true = set()
    counts = {
        class_id: {"tp": 0, "fp": 0, "fn": 0}
        for class_id in ID_TO_CLASS
    }

    for prediction_index in order.tolist():
        class_id = int(pred_labels[prediction_index])
        if class_id not in counts:
            continue

        candidates = [
            index
            for index, label in enumerate(true_labels.tolist())
            if label == class_id and index not in matched_true
        ]

        if not candidates:
            counts[class_id]["fp"] += 1
            continue

        ious = box_iou(
            pred_boxes[prediction_index].unsqueeze(0),
            true_boxes[candidates],
        )[0]
        best_position = int(ious.argmax())

        if float(ious[best_position]) >= iou_threshold:
            matched_true.add(candidates[best_position])
            counts[class_id]["tp"] += 1
        else:
            counts[class_id]["fp"] += 1

    for true_index, class_id in enumerate(true_labels.tolist()):
        if class_id in counts and true_index not in matched_true:
            counts[class_id]["fn"] += 1

    return counts


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def main():
    args = parse_args()
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
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    totals = {
        class_id: {"tp": 0, "fp": 0, "fn": 0}
        for class_id in ID_TO_CLASS
    }

    map_metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        class_metrics=True,
    )

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluating"):
            predictions = model([image.to(device) for image in images])
            map_predictions = [
                {
                    "boxes": prediction["boxes"].cpu(),
                    "scores": prediction["scores"].cpu(),
                    "labels": prediction["labels"].cpu(),
                }
                for prediction in predictions
            ]

            map_targets = [
                {
                    "boxes": target["boxes"].cpu(),
                    "labels": target["labels"].cpu(),
                }
                for target in targets
            ]

            map_metric.update(map_predictions, map_targets)

            for prediction, target in zip(predictions, targets):
                counts = match_one_image(
                    prediction,
                    target,
                    args.score_threshold,
                    args.iou_threshold,
                )
                for class_id in totals:
                    for key in ("tp", "fp", "fn"):
                        totals[class_id][key] += counts[class_id][key]

    print(
        f"\nMetrics at score >= {args.score_threshold} "
        f"and IoU >= {args.iou_threshold}"
    )
    print(f"{'Class':<16} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 49)

    total_tp = total_fp = total_fn = 0
    for class_id, class_name in enumerate(CATEGORIES, start=1):
        tp = totals[class_id]["tp"]
        fp = totals[class_id]["fp"]
        fn = totals[class_id]["fn"]
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        print(f"{class_name:<16} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f}")
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = safe_divide(total_tp, total_tp + total_fp)
    recall = safe_divide(total_tp, total_tp + total_fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    print("-" * 49)
    print(f"{'Overall (micro)':<16} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f}")
    map_results = map_metric.compute()

    print("\nCOCO-style object detection metrics")
    print("-" * 49)
    print(f"mAP@0.50:0.95: {map_results['map'].item():.4f}")
    print(f"mAP@0.50:      {map_results['map_50'].item():.4f}")
    print(f"mAP@0.75:      {map_results['map_75'].item():.4f}")
    print(f"mAR@100:       {map_results['mar_100'].item():.4f}")
    print("\nPer-class mAP@0.50:0.95")
    print("-" * 32)

    for class_id, class_ap in zip(
        map_results["classes"].tolist(),
        map_results["map_per_class"].tolist(),
    ):
        class_name = ID_TO_CLASS.get(class_id, f"class {class_id}")
        print(f"{class_name:<16} {class_ap:>10.4f}")


if __name__ == "__main__":
    main()

