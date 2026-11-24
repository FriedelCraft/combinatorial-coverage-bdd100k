"""Draw detections from a trained checkpoint on one image."""

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as F

from src.config import ID_TO_CLASS
from src.model import create_model
from src.utils import choose_device


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("prediction.jpg"))
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--min-size", type=int, default=640)
    parser.add_argument("--max-size", type=int, default=1024)
    return parser.parse_args()


def main():
    args = parse_args()
    device = choose_device()

    model = create_model(
        pretrained=False,
        min_size=args.min_size,
        max_size=args.max_size,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    image = Image.open(args.image).convert("RGB")
    image_tensor = F.convert_image_dtype(F.pil_to_tensor(image), torch.float32)

    with torch.no_grad():
        prediction = model([image_tensor.to(device)])[0]

    draw = ImageDraw.Draw(image)
    for box, class_id, score in zip(
        prediction["boxes"].cpu(),
        prediction["labels"].cpu(),
        prediction["scores"].cpu(),
    ):
        if float(score) < args.score_threshold:
            continue

        x1, y1, x2, y2 = box.tolist()
        class_name = ID_TO_CLASS.get(int(class_id), "unknown")
        text = f"{class_name} {float(score):.2f}"
        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        draw.text((x1 + 4, y1 + 4), text, fill="yellow", stroke_width=2, stroke_fill="black")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(f"Saved prediction to {args.output.resolve()}")


if __name__ == "__main__":
    main()

