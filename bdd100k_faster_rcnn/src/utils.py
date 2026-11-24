"""Small helpers shared by the command-line programs."""

import random
from pathlib import Path

import torch


def collate_fn(batch):
    """Keep images and targets as lists because boxes have different lengths."""
    return tuple(zip(*batch))


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_targets_to_device(targets, device):
    return [
        {key: value.to(device) for key, value in target.items()}
        for target in targets
    ]


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_loss, args,
                    global_step=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "args": vars(args),
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        path,
    )

