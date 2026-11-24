"""Evaluate COCO mAP for whole, day, night, and rainy validation domains."""

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate_combinations import (create_map_metric, load_all_metadata,
                                   prediction_to_cpu, target_to_cpu)
from src.bdd_dataset import BDD100KDataset
from src.config import ID_TO_CLASS
from src.model import create_model
from src.utils import choose_device, collate_fn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(".."))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    device = choose_device()
    dataset = BDD100KDataset(args.project_root, "val", manifest_path=args.val_manifest)
    metadata = load_all_metadata(dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=collate_fn,
                        pin_memory=device.type == "cuda")
    model = create_model(pretrained=False).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    groups = {name: create_map_metric() for name in ("all", "daytime", "night", "rainy", "nonrain")}
    counts = {name: 0 for name in groups}
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluating domains"):
            predictions = model([image.to(device) for image in images])
            for prediction, target in zip(predictions, targets):
                index = int(target["image_id"])
                weather, _, timeofday = metadata[index]
                active = ["all"]
                if timeofday in ("daytime", "night"):
                    active.append(timeofday)
                if weather == "rainy":
                    active.append("rainy")
                else:
                    active.append("nonrain")
                pred, truth = prediction_to_cpu(prediction), target_to_cpu(target)
                for name in active:
                    groups[name].update([pred], [truth])
                    counts[name] += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["domain", "images", "map_50_95", "map_50", "map_75",
              *[f"ap_{name.replace(' ', '_')}" for name in ID_TO_CLASS.values()]]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, metric in groups.items():
            result = metric.compute()
            per_class = {int(class_id): float(ap) for class_id, ap in zip(
                result["classes"].tolist(), result["map_per_class"].tolist())}
            row = {"domain": name, "images": counts[name],
                   "map_50_95": float(result["map"]),
                   "map_50": float(result["map_50"]),
                   "map_75": float(result["map_75"])}
            for class_id, category in ID_TO_CLASS.items():
                ap = per_class.get(class_id, -1.0)
                row[f"ap_{category.replace(' ', '_')}"] = ap if ap >= 0 else ""
            writer.writerow(row)
            print(f"{name}: n={counts[name]}, mAP={row['map_50_95']:.4f}")


if __name__ == "__main__":
    main()
