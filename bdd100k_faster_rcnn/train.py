"""Train Faster R-CNN on the BDD100K training split."""

import argparse
import csv
import json
import hashlib
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.bdd_dataset import BDD100KDataset
from src.model import create_model
from src.utils import (
    choose_device,
    collate_fn,
    move_targets_to_device,
    save_checkpoint,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--train-labels", type=Path, default=None)
    parser.add_argument("--val-labels", type=Path, default=None)
    parser.add_argument("--val-manifest", type=Path, default=None,
                        help="CSV of development images used for checkpoint selection.")
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=None,
        help="Optional CSV containing an image_name column for the training subset.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Total optimizer steps for budget-matched research runs.")
    parser.add_argument("--lr-step-steps", type=int, default=None,
                        help="LR decay interval in optimizer steps; required with --max-steps.")
    parser.add_argument("--eval-every-steps", type=int, default=None,
                        help="Use fixed validation intervals for budget-matched runs.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--step-size", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--min-size", type=int, default=640)
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--no-amp", action="store_true")

    # These two options are only for quick experiments. Their default, None,
    # means the entire train/validation split is used.
    parser.add_argument("--max-train-images", type=int, default=None)
    parser.add_argument("--max-val-images", type=int, default=None)
    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, print_every,
                    step_limit=None, scheduler=None, amp_enabled=False):
    model.train()
    running_loss = 0.0
    updates = 0

    progress = tqdm(loader, desc=f"Epoch {epoch} - training")
    for step, (images, targets) in enumerate(progress, start=1):
        images = [image.to(device) for image in images]
        targets = move_targets_to_device(targets, device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=amp_enabled):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

        scaler.scale(loss).backward()
        scale_before_step = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer_stepped = scaler.get_scale() >= scale_before_step
        if optimizer_stepped:
            updates += 1
        if scheduler is not None and optimizer_stepped:
            scheduler.step()

        running_loss += loss.item()

        if step % print_every == 0 or step == len(loader):
            progress.set_postfix(loss=f"{running_loss / step:.4f}")
        if step_limit is not None and updates >= step_limit:
            break

    return running_loss / step, updates


@torch.no_grad()
def calculate_validation_loss(model, loader, device):
    # Detection models return losses only in training mode. No gradients are
    # created here, so this does not update the network.
    model.train()
    running_loss = 0.0

    progress = tqdm(loader, desc="Validation")
    for images, targets in progress:
        images = [image.to(device) for image in images]
        targets = move_targets_to_device(targets, device)

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        running_loss += loss.item()
        progress.set_postfix(loss=f"{running_loss / (progress.n + 1):.4f}")

    return running_loss / len(loader)


def main():
    args = parse_args()
    if args.max_steps is not None:
        if args.max_steps <= 0 or not args.lr_step_steps or args.lr_step_steps <= 0:
            raise ValueError("--max-steps requires positive --max-steps and --lr-step-steps")
        if args.eval_every_steps is not None and args.eval_every_steps <= 0:
            raise ValueError("--eval-every-steps must be positive")
    set_seed(args.seed)
    device = choose_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")

    train_dataset = BDD100KDataset(
        args.project_root,
        "train",
        labels_path=args.train_labels,
        manifest_path=args.train_manifest,
        augment=not args.no_augmentation,
        max_images=args.max_train_images,
    )
    val_dataset = BDD100KDataset(
        args.project_root,
        "val",
        labels_path=args.val_labels,
        manifest_path=args.val_manifest,
        augment=False,
        max_images=args.max_val_images,
    )

    print(f"Training images:   {len(train_dataset):,}")
    print(f"Validation images: {len(val_dataset):,}")
    if args.train_manifest is not None:
        print(f"Training manifest: {args.train_manifest.resolve()}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )

    model = create_model(
        pretrained=not args.no_pretrained,
        min_size=args.min_size,
        max_size=args.max_size,
    ).to(device)

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.SGD(
        trainable_parameters,
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step_steps if args.max_steps is not None else args.step_size,
        gamma=args.gamma,
    )

    use_amp = device.type == "cuda" and not args.no_amp
    if use_amp and not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is required for AMP; use --no-amp on this GPU")
    # BF16 has sufficient exponent range without gradient scaling. FP16
    # overflow skipped all updates on the RTX 5050 smoke run.
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    start_epoch = 1
    global_step = 0
    best_val_loss = float("inf")

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint.get("global_step", 0)
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
            torch.set_rng_state(checkpoint["torch_rng_state"])
            if device.type == "cuda" and checkpoint.get("cuda_rng_states") is not None:
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
        old_args = checkpoint.get("args", {})
        if args.max_steps is not None and (
            old_args.get("max_steps") != args.max_steps
            or old_args.get("lr_step_steps") != args.lr_step_steps
            or old_args.get("eval_every_steps") != args.eval_every_steps
        ):
            raise ValueError("Resume checkpoint uses a different fixed-step budget or schedule")
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"Resuming after epoch {checkpoint['epoch']}")

    manifest_hash = None
    if args.train_manifest is not None:
        manifest_hash = hashlib.sha256(args.train_manifest.read_bytes()).hexdigest()
    val_manifest_hash = None
    if args.val_manifest is not None:
        val_manifest_hash = hashlib.sha256(args.val_manifest.read_bytes()).hexdigest()
    run_config = {**{key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(args).items()},
                  "train_images": len(train_dataset), "val_images": len(val_dataset),
                  "manifest_sha256": manifest_hash,
                  "val_manifest_sha256": val_manifest_hash,
                  "torch_version": torch.__version__,
                  "torchvision_version": __import__("torchvision").__version__}
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, default=str), encoding="utf-8")

    epoch = start_epoch
    while (global_step < args.max_steps if args.max_steps is not None
           else epoch <= args.epochs):
        remaining = args.max_steps - global_step if args.max_steps is not None else None
        if remaining is not None and args.eval_every_steps is not None:
            remaining = min(remaining, args.eval_every_steps)
        train_loss, steps_done = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            epoch,
            args.print_every,
            step_limit=remaining,
            scheduler=scheduler if args.max_steps is not None else None,
            amp_enabled=use_amp,
        )
        global_step += steps_done
        if steps_done == 0:
            raise RuntimeError("No optimizer updates completed in this pass")
        val_loss = calculate_validation_loss(model, val_loader, device)
        current_learning_rate = optimizer.param_groups[0]["lr"]
        if args.max_steps is None:
            scheduler.step()

        print(
            f"Epoch {epoch:02d}, global_step={global_step}: "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"lr={current_learning_rate:.6f}"
        )

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        save_checkpoint(
            args.output_dir / "last_checkpoint.pth",
            model,
            optimizer,
            scheduler,
            epoch,
            best_val_loss,
            args,
            global_step=global_step,
        )

        history_file = args.output_dir / "history.csv"
        needs_header = not history_file.exists()
        with history_file.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if needs_header:
                writer.writerow(["epoch", "global_step", "train_loss", "val_loss", "learning_rate"])
            writer.writerow([epoch, global_step, train_loss, val_loss, current_learning_rate])

        if is_best:
            save_checkpoint(
                args.output_dir / "best_checkpoint.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_loss,
                args,
                global_step=global_step,
            )
            print("Saved a new best checkpoint.")
        epoch += 1

    print(f"Training finished. Checkpoints are in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
