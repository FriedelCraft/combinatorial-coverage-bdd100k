"""Load a few items before committing to a long training run."""

import argparse
from pathlib import Path

from src.bdd_dataset import BDD100KDataset
from src.config import ID_TO_CLASS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args()

    dataset = BDD100KDataset(
        args.project_root,
        args.split,
        labels_path=args.labels,
        max_images=args.count,
    )

    print(f"Dataset contains {len(dataset):,} selected images")
    for index in range(len(dataset)):
        image, target = dataset[index]
        names = [ID_TO_CLASS[int(label)] for label in target["labels"]]
        print(
            f"[{index}] image shape={tuple(image.shape)}, "
            f"boxes={tuple(target['boxes'].shape)}, classes={names}"
        )


if __name__ == "__main__":
    main()

