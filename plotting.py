"""Plotting utilities for w(x, y) displacement field."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from data import PLATE_LENGTH, PLATE_WIDTH, device


def plot_displacement_3d(
    model: torch.nn.Module,
    epoch: int,
    save_dir: str | Path = "plots",
    n_points: int = 80,
    elev: float = 30,
    azim: float = -60,
) -> Path:
    """Create and save a 3D surface plot of w(x,y) for the given epoch.

    Returns the saved file path.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.inference_mode():
        x = torch.linspace(0, PLATE_LENGTH, n_points, device=device)
        y = torch.linspace(0, PLATE_WIDTH, n_points, device=device)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        xy = torch.stack([X.flatten(), Y.flatten()], dim=1)
        W = model(xy).reshape(n_points, n_points).cpu().numpy()

    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        X_np,
        Y_np,
        W,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        rcount=n_points,
        ccount=n_points,
    )
    fig.colorbar(surf, ax=ax, shrink=0.6, label="w (m)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("w (m)")
    ax.set_title(f"Displacement w(x, y) — Epoch {epoch} (3D)")
    ax.view_init(elev=elev, azim=azim)

    path = save_dir / f"w_epoch_{epoch:06d}_3d.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
