"""Plotting utilities for w(x, y) displacement field."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from data import PLATE_LENGTH, PLATE_WIDTH, device


def plot_displacement(
    model: torch.nn.Module,
    epoch: int,
    save_dir: str | Path = "plots",
    n_points: int = 100,
) -> Path:
    """Plot w(x, y) and save to file. Returns the saved path."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.no_grad():
        x = torch.linspace(0, PLATE_LENGTH, n_points, device=device)
        y = torch.linspace(0, PLATE_WIDTH, n_points, device=device)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        xy = torch.stack([X.flatten(), Y.flatten()], dim=1)
        W = model(xy).reshape(n_points, n_points).cpu().numpy()

    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    c = ax.contourf(X_np, Y_np, W, levels=50, cmap="viridis")
    fig.colorbar(c, ax=ax, label="w (m)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Displacement w(x, y) — Epoch {epoch}")
    ax.set_aspect("equal")

    path = save_dir / f"w_epoch_{epoch:06d}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
