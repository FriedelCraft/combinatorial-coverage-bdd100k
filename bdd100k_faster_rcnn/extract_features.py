"""Extract L2-normalized ResNet-50 image vectors without reading labels."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm


class Images(Dataset):
    def __init__(self, directory, names, transform):
        self.directory, self.names, self.transform = directory, names, transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        with Image.open(self.directory / self.names[index]) as image:
            return self.transform(image.convert("RGB"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(".."))
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    image_dir = args.project_root / "bdd100k_images_100k" / "100k" / args.split
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)
    if args.manifest is None:
        names = sorted(path.name for path in image_dir.glob("*.jpg"))
    else:
        with args.manifest.open(newline="", encoding="utf-8-sig") as handle:
            names = [row["image_name"] for row in csv.DictReader(handle)]
    if not names or len(names) != len(set(names)):
        raise ValueError("Image list is empty or has duplicates")
    missing = [name for name in names if not (image_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images; first: {missing[0]}")
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    loader = DataLoader(Images(image_dir, names, weights.transforms()),
                        batch_size=args.batch_size, num_workers=args.workers,
                        shuffle=False, pin_memory=device.type == "cuda")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    vectors = np.lib.format.open_memmap(str(args.output_prefix) + ".npy", mode="w+",
                                         dtype="float16", shape=(len(names), 2048))
    offset = 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Extracting {args.split} features"):
            batch = batch.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                features = model(batch)
            features = torch.nn.functional.normalize(features.float(), dim=1)
            count = len(batch)
            vectors[offset:offset + count] = features.cpu().numpy().astype("float16")
            offset += count
    vectors.flush()
    meta = {"split": args.split, "images": len(names), "dimension": 2048,
            "weights": str(weights), "names": names}
    (Path(str(args.output_prefix) + ".json")).write_text(
        json.dumps(meta), encoding="utf-8")
    print(f"Saved {len(names):,} vectors to {args.output_prefix}.npy")


if __name__ == "__main__":
    main()
