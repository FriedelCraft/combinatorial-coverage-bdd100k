"""Exploratory seed-42 rainy time-of-day and checkpoint evaluation."""

import csv
import gc
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate_combinations import create_map_metric, prediction_to_cpu, target_to_cpu
from src.bdd_dataset import BDD100KDataset
from src.config import ID_TO_CLASS
from src.model import create_model
from src.utils import choose_device, collate_fn


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/analysis/rain_seed42_checkpoint_diagnostic"
RUNS = ROOT / "outputs/experiments"
SPECIFICATIONS = (
    ("adaptive_target_aware", "best_checkpoint.pth", 3750),
    ("adaptive_target_aware", "last_checkpoint.pth", 7500),
    ("rain_count_control", "best_checkpoint.pth", 7500),
)
GROUPS = ("rainy_all", "rainy_daytime", "rainy_night", "rainy_dawn_dusk")
FIELDS = ("method", "checkpoint", "step", "group", "images", "map_50_95",
          "map_50", "map_75",
          *[f"ap_{name.replace(' ', '_')}" for name in ID_TO_CLASS.values()])


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_rain_manifest():
    metadata = {row["image_name"]: row for row in
                read_csv(ROOT / "outputs/metadata/master_val.csv")}
    final = [row["image_name"] for row in
             read_csv(ROOT / "research_splits/final_evaluation.csv")]
    rainy = [name for name in final if metadata[name]["weather"] == "rainy"]
    if len(rainy) != 532 or len(rainy) != len(set(rainy)):
        raise ValueError("Unexpected rainy final set")
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = OUT / "rainy_final_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_name"])
        writer.writerows((name,) for name in rainy)
    return manifest, metadata


def metrics_for(checkpoint_path, expected_step, dataset, metadata, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    step = int(checkpoint["global_step"])
    if step != expected_step:
        raise ValueError(f"Expected step {expected_step}, found {step}: {checkpoint_path}")
    model = create_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device).eval()
    del checkpoint
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0,
                        collate_fn=collate_fn, pin_memory=device.type == "cuda")
    metrics = {group: create_map_metric() for group in GROUPS}
    counts = {group: 0 for group in GROUPS}
    with torch.inference_mode():
        for images, targets in tqdm(loader, desc=f"Step {step} ({checkpoint_path.parent.name})"):
            predictions = model([image.to(device) for image in images])
            for prediction, target in zip(predictions, targets):
                name = dataset.image_paths[int(target["image_id"])].name
                time = metadata[name]["timeofday"]
                group = {"daytime": "rainy_daytime", "night": "rainy_night",
                         "dawn/dusk": "rainy_dawn_dusk"}.get(time)
                pred, truth = prediction_to_cpu(prediction), target_to_cpu(target)
                active = ("rainy_all", group) if group else ("rainy_all",)
                for target_group in active:
                    metrics[target_group].update([pred], [truth])
                    counts[target_group] += 1
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    result_rows = []
    for group, metric in metrics.items():
        result = metric.compute()
        per_class = {int(class_id): float(ap) for class_id, ap in zip(
            result["classes"].tolist(), result["map_per_class"].tolist())}
        row = {"step": step, "group": group, "images": counts[group],
               "map_50_95": float(result["map"]),
               "map_50": float(result["map_50"]),
               "map_75": float(result["map_75"])}
        for class_id, category in ID_TO_CLASS.items():
            ap = per_class.get(class_id, -1.0)
            row[f"ap_{category.replace(' ', '_')}"] = ap if ap >= 0 else ""
        result_rows.append(row)
    return result_rows


def main():
    manifest, metadata = build_rain_manifest()
    dataset = BDD100KDataset(ROOT.parent, "val", manifest_path=manifest)
    if len(dataset) != 532:
        raise ValueError(f"Expected 532 rainy images, got {len(dataset)}")
    device = choose_device()
    print(f"Using {device}; {len(dataset)} rainy final images", flush=True)
    output = OUT / "rain_time_checkpoint_metrics.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for method, filename, expected_step in SPECIFICATIONS:
            checkpoint = RUNS / f"benchmark_rain_{method}_seed42_steps7500" / filename
            for row in metrics_for(checkpoint, expected_step, dataset, metadata, device):
                writer.writerow({"method": method, "checkpoint": filename, **row})
                print(f"{method} step {row['step']} {row['group']}: "
                      f"n={row['images']} mAP={row['map_50_95']:.4f}", flush=True)
            handle.flush()
    print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
