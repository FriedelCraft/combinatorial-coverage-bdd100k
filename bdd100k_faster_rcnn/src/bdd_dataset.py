"""BDD100K dataset class for TorchVision object-detection models."""

import csv
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

from src.config import CLASS_TO_ID


class BDD100KDataset(Dataset):
    """Read BDD100K images and their 2D bounding boxes.

    This class supports both common label layouts:

    1. One large file such as ``bdd100k_labels_images_train.json``.
    2. One JSON file per image inside ``bdd100k_labels/100k/train``.
    """

    def __init__(
        self,
        project_root,
        split,
        labels_path=None,
        manifest_path=None,
        augment=False,
        max_images=None,
    ):
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")

        self.project_root = Path(project_root)
        self.split = split
        self.augment = augment
        self.image_dir = self._find_image_dir()
        self.labels_path = (
            Path(labels_path)
            if labels_path is not None
            else self._find_labels_path()
        )

        # The source is either a dictionary held in memory or a directory from
        # which one small JSON file is read for each requested image.
        self.annotations = None

        if self.labels_path.is_file():
            self.annotations = self._read_combined_labels(self.labels_path)
            self.image_paths = []

            for image_name in self.annotations:
                image_path = self.image_dir / image_name
                if image_path.exists():
                    self.image_paths.append(image_path)
        else:
            self.image_paths = sorted(self.image_dir.glob("*.jpg"))

        if manifest_path is not None:
            self.image_paths = self._select_manifest_images(manifest_path)

        if max_images is not None:
            self.image_paths = self.image_paths[:max_images]

        if not self.image_paths:
            raise RuntimeError(f"No labeled images were found in {self.image_dir}")

    def _find_image_dir(self):
        candidates = [
            self.project_root / "bdd100k_images_100k" / "100k" / self.split,
            self.project_root
            / "bdd100k_images_100k"
            / "bdd100k"
            / "images"
            / "100k"
            / self.split,
            self.project_root / "images" / "100k" / self.split,
        ]

        for path in candidates:
            if path.is_dir():
                return path

        tried = "\n".join(f"  - {path}" for path in candidates)
        raise FileNotFoundError(
            f"Could not find the BDD100K {self.split} image folder. Tried:\n{tried}"
        )

    def _find_labels_path(self):
        label_root = self.project_root / "bdd100k_labels"

        file_names = (
            f"bdd100k_labels_images_{self.split}.json",
            f"det_{self.split}.json",
        )
        for file_name in file_names:
            matches = sorted(label_root.rglob(file_name)) if label_root.exists() else []
            if matches:
                return matches[0]

        directory_candidates = [
            label_root / "100k" / self.split,
            label_root / self.split,
        ]
        for path in directory_candidates:
            if path.is_dir():
                return path

        raise FileNotFoundError(
            f"Could not find labels for split '{self.split}' inside {label_root}. "
            "You can pass the exact path with labels_path."
        )

    def _select_manifest_images(self, manifest_path):
        """Return image paths listed in a CSV manifest, in manifest order."""
        manifest_path = Path(manifest_path)
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Training manifest not found: {manifest_path}")

        available_images = {path.name: path for path in self.image_paths}
        selected_paths = []
        seen_names = set()

        with manifest_path.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None or "image_name" not in reader.fieldnames:
                raise ValueError(
                    f"Manifest {manifest_path} must contain an 'image_name' column"
                )

            for row_number, row in enumerate(reader, start=2):
                image_name = (row.get("image_name") or "").strip()
                if not image_name:
                    raise ValueError(
                        f"Manifest {manifest_path} has an empty image_name "
                        f"at row {row_number}"
                    )
                if image_name in seen_names:
                    raise ValueError(
                        f"Manifest {manifest_path} contains duplicate image "
                        f"'{image_name}' at row {row_number}"
                    )

                image_path = available_images.get(image_name)
                if image_path is None:
                    raise FileNotFoundError(
                        f"Manifest image '{image_name}' was not found among the "
                        f"labeled {self.split} images in {self.image_dir}"
                    )

                seen_names.add(image_name)
                selected_paths.append(image_path)

        if not selected_paths:
            raise ValueError(f"Manifest {manifest_path} contains no images")

        return selected_paths

    @staticmethod
    def _read_combined_labels(labels_file):
        print(f"Loading annotations from {labels_file} ...")
        with labels_file.open("r", encoding="utf-8") as file:
            frames = json.load(file)

        if not isinstance(frames, list):
            raise ValueError(f"Expected a list of frames in {labels_file}")

        # Keep only the small part needed for object detection. This releases
        # attributes and polygon points instead of retaining the complete,
        # much larger BDD100K JSON structure during every epoch.
        annotations = {}
        for frame in frames:
            if not isinstance(frame, dict) or "name" not in frame:
                continue

            detection_labels = []
            for annotation in frame.get("labels", []):
                if (
                    annotation.get("category") in CLASS_TO_ID
                    and annotation.get("box2d") is not None
                ):
                    detection_labels.append(
                        {
                            "category": annotation["category"],
                            "box2d": annotation["box2d"],
                        }
                    )

            annotations[frame["name"]] = {"labels": detection_labels}

        return annotations

    def _read_frame(self, image_path):
        if self.annotations is not None:
            return self.annotations[image_path.name]

        label_file = self.labels_path / f"{image_path.stem}.json"
        if not label_file.exists():
            # Some exports keep the image extension in the label filename.
            label_file = self.labels_path / f"{image_path.name}.json"

        if not label_file.exists():
            raise FileNotFoundError(f"Missing label file for {image_path.name}")

        with label_file.open("r", encoding="utf-8") as file:
            frame = json.load(file)

        # A few BDD100K exports wrap one frame dictionary in a list.
        if isinstance(frame, list):
            if len(frame) != 1:
                raise ValueError(f"Expected one frame in {label_file}")
            frame = frame[0]

        if "frames" in frame:
            frame = {
                "name": frame.get("name", image_path.name),
                "attributes": frame.get("attributes", {}),
                "labels": frame["frames"][0].get("objects", []),
            }
        return frame

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        frame = self._read_frame(image_path)

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        boxes = []
        labels = []

        for annotation in frame.get("labels", []):
            category = annotation.get("category")
            box = annotation.get("box2d")

            # Ignore lane/polygon labels and categories outside our 10 classes.
            if category not in CLASS_TO_ID or box is None:
                continue

            x1 = max(0.0, min(float(box["x1"]), width))
            y1 = max(0.0, min(float(box["y1"]), height))
            x2 = max(0.0, min(float(box["x2"]), width))
            y2 = max(0.0, min(float(box["y2"]), height))

            # Faster R-CNN cannot train on zero-area or malformed boxes.
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(CLASS_TO_ID[category])

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image = F.pil_to_tensor(image)
        image = F.convert_image_dtype(image, torch.float32)

        # A horizontal flip is the only augmentation in this baseline.
        if self.augment and torch.rand(1).item() < 0.5:
            image = F.hflip(image)
            if len(boxes) > 0:
                old_x1 = boxes[:, 0].clone()
                old_x2 = boxes[:, 2].clone()
                boxes[:, 0] = width - old_x2
                boxes[:, 2] = width - old_x1

        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(index, dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
        }

        return image, target
