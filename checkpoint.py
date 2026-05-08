"""Checkpoint save/load utilities."""

from pathlib import Path

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_loss: float,
    adaptive_weighter: torch.nn.Module | None = None,
) -> None:
    """Save training checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_loss": best_loss,
    }
    if adaptive_weighter is not None:
        data["adaptive_weighter_state_dict"] = adaptive_weighter.state_dict()
    torch.save(data, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    adaptive_weighter: torch.nn.Module | None = None,
) -> tuple[int, float]:
    """Load training checkpoint. Returns (start_epoch, best_loss)."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if adaptive_weighter is not None and "adaptive_weighter_state_dict" in ckpt:
        adaptive_weighter.load_state_dict(ckpt["adaptive_weighter_state_dict"])
    return ckpt["epoch"], ckpt["best_loss"]
