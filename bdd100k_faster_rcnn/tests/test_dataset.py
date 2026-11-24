import json

import torch
from PIL import Image

from src.bdd_dataset import BDD100KDataset
from src.config import CLASS_TO_ID


def test_dataset_reads_and_filters_boxes(tmp_path):
    image_dir = tmp_path / "bdd100k_images_100k" / "100k" / "train"
    label_dir = tmp_path / "bdd100k_labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    Image.new("RGB", (100, 50), "white").save(image_dir / "sample.jpg")

    frames = [
        {
            "name": "sample.jpg",
            "labels": [
                {
                    "category": "car",
                    "box2d": {"x1": 10, "y1": 5, "x2": 80, "y2": 40},
                },
                {
                    "category": "lane",
                    "box2d": {"x1": 1, "y1": 1, "x2": 20, "y2": 20},
                },
                {
                    "category": "person",
                    "box2d": {"x1": 30, "y1": 10, "x2": 20, "y2": 25},
                },
            ],
        }
    ]

    labels_file = label_dir / "bdd100k_labels_images_train.json"
    labels_file.write_text(json.dumps(frames), encoding="utf-8")

    dataset = BDD100KDataset(tmp_path, "train", augment=False)
    image, target = dataset[0]

    assert image.shape == (3, 50, 100)
    assert image.dtype == torch.float32
    assert target["boxes"].shape == (1, 4)
    assert target["labels"].tolist() == [CLASS_TO_ID["car"]]


def test_horizontal_flip_changes_box_coordinates(tmp_path, monkeypatch):
    image_dir = tmp_path / "bdd100k_images_100k" / "100k" / "train"
    label_dir = tmp_path / "bdd100k_labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    Image.new("RGB", (100, 50), "white").save(image_dir / "sample.jpg")
    frames = [
        {
            "name": "sample.jpg",
            "labels": [
                {
                    "category": "car",
                    "box2d": {"x1": 10, "y1": 5, "x2": 30, "y2": 40},
                }
            ],
        }
    ]
    (label_dir / "bdd100k_labels_images_train.json").write_text(
        json.dumps(frames), encoding="utf-8"
    )

    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor([0.0]))
    dataset = BDD100KDataset(tmp_path, "train", augment=True)
    _, target = dataset[0]

    assert target["boxes"].tolist() == [[70.0, 5.0, 90.0, 40.0]]

