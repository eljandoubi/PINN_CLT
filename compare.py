"""Compare a trained PINN checkpoint with the Ritz reference solution.

    uv run compare.py --checkpoint runs/<run_id>/checkpoints/best.pt

Pass the same architecture options as in training (`--hidden_layers`,
`--hidden_units`, `--activation`, `--use_residual`, ...). The script writes
`comparison.json` and `comparison.png` into `--output`.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MaxNLocator

from data import PLATE_LENGTH, PLATE_WIDTH
from model import ACTIVATIONS, PINN
from reference import RitzSolution, boundary_residuals, ritz_solve


@dataclass(frozen=True)
class Comparison:
    """Metrics plus the two fields on the evaluation grid (indexed [x, y])."""

    metrics: dict
    x: np.ndarray
    y: np.ndarray
    network: np.ndarray
    reference: np.ndarray


def load_model(
    checkpoint: str | Path,
    hidden_layers: int = 4,
    hidden_units: int = 128,
    activation: str = "tanh",
    use_residual: bool = False,
    use_norm: bool = False,
    use_ffmlp: bool = False,
) -> PINN:
    """Rebuild the network of `train.py` on the CPU and load the checkpoint."""
    model = PINN(
        hidden_layers=hidden_layers,
        hidden_units=hidden_units,
        activation=ACTIVATIONS[activation],
        use_residual=use_residual,
        use_norm=use_norm,
        use_ffmlp=use_ffmlp,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    return model.eval()


def network_fields(model: torch.nn.Module):
    """Return `fields(x, y)` giving w, w_x, w_xx and w_yy of the network."""

    def fields(x, y) -> dict[str, np.ndarray]:
        xy = torch.tensor(
            np.stack([x, y], axis=1), dtype=torch.float32, requires_grad=True
        )
        w = model(xy).squeeze(1)
        gradient = torch.autograd.grad(w.sum(), xy, create_graph=True)[0]
        w_x, w_y = gradient[:, 0], gradient[:, 1]
        w_xx = torch.autograd.grad(w_x.sum(), xy, retain_graph=True)[0][:, 0]
        w_yy = torch.autograd.grad(w_y.sum(), xy)[0][:, 1]
        values = {"w": w, "w_x": w_x, "w_xx": w_xx, "w_yy": w_yy}
        return {
            name: tensor.detach().numpy().astype(float)
            for name, tensor in values.items()
        }

    return fields


def compare(
    model: torch.nn.Module,
    reference: RitzSolution,
    points_x: int = 101,
    points_y: int = 51,
) -> Comparison:
    """Evaluate the network and the reference on a uniform grid and compare."""
    x = np.linspace(0.0, reference.length, points_x)
    y = np.linspace(0.0, reference.width, points_y)
    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
    fields = network_fields(model)

    network = fields(grid_x.ravel(), grid_y.ravel())["w"].reshape(grid_x.shape)
    exact = reference.evaluate(grid_x, grid_y)
    scale = float(np.abs(exact).max())
    difference = network - exact
    center = fields(
        np.array([0.5 * reference.length]), np.array([0.5 * reference.width])
    )["w"][0]

    metrics = {
        "relative_l2_error": float(np.linalg.norm(difference) / np.linalg.norm(exact)),
        "max_error_over_max_deflection": float(np.abs(difference).max() / scale),
        "reference_center_deflection": reference.center_deflection,
        "network_center_deflection": float(center),
        "reference_max_deflection": float(exact.max()),
        "network_max_deflection": float(network.max()),
        "reference_min_deflection": float(exact.min()),
        "network_min_deflection": float(network.min()),
        "network_boundary_rms": boundary_residuals(
            fields, scale, length=reference.length, width=reference.width
        ),
        "reference_boundary_rms": boundary_residuals(
            reference.fields, scale, length=reference.length, width=reference.width
        ),
    }
    return Comparison(metrics, x, y, network, exact)


def plot_comparison(result: Comparison, path: str | Path) -> Path:
    """Save the network field, the reference and their difference side by side."""
    figure, axes = plt.subplots(1, 3, figsize=(15, 3.6), constrained_layout=True)
    grid_x, grid_y = np.meshgrid(result.x, result.y, indexing="ij")
    difference = result.network - result.reference
    low = min(result.network.min(), result.reference.min())
    high = max(result.network.max(), result.reference.max())
    limit = float(np.abs(difference).max()) or 1.0
    panels = [
        ("PINN w (m)", result.network, {"vmin": low, "vmax": high}),
        ("Ritz reference w (m)", result.reference, {"vmin": low, "vmax": high}),
        (
            "PINN - reference (m)",
            difference,
            {"vmin": -limit, "vmax": limit, "cmap": "coolwarm"},
        ),
    ]
    for axis, (title, values, options) in zip(axes, panels, strict=True):
        image = axis.pcolormesh(grid_x, grid_y, values, shading="auto", **options)
        axis.set_title(title)
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.set_aspect("equal")
        colorbar = figure.colorbar(image, ax=axis, orientation="horizontal", shrink=0.9)
        colorbar.locator = MaxNLocator(nbins=4)
        colorbar.update_ticks()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _flag(text: str) -> bool:
    return text.lower() in ("true", "1", "yes")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True, help="path to best.pt")
    parser.add_argument("--output", default="comparison", help="output directory")
    parser.add_argument("--hidden_layers", type=int, default=4)
    parser.add_argument("--hidden_units", type=int, default=128)
    parser.add_argument("--activation", choices=sorted(ACTIVATIONS), default="tanh")
    parser.add_argument("--use_residual", type=_flag, default=False)
    parser.add_argument("--use_norm", type=_flag, default=False)
    parser.add_argument("--use_ffmlp", type=_flag, default=False)
    parser.add_argument("--basis_x", type=int, default=12, help="Ritz terms in x")
    parser.add_argument("--basis_y", type=int, default=8, help="Ritz terms in y")
    parser.add_argument("--points_x", type=int, default=101)
    parser.add_argument("--points_y", type=int, default=51)
    return parser


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    model = load_model(
        args.checkpoint,
        hidden_layers=args.hidden_layers,
        hidden_units=args.hidden_units,
        activation=args.activation,
        use_residual=args.use_residual,
        use_norm=args.use_norm,
        use_ffmlp=args.use_ffmlp,
    )
    reference = ritz_solve(
        args.basis_x, args.basis_y, length=PLATE_LENGTH, width=PLATE_WIDTH
    )
    result = compare(model, reference, args.points_x, args.points_y)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": Path(args.checkpoint).name,
        "ritz_basis": [args.basis_x, args.basis_y],
        "ritz_condition_number": reference.condition_number,
        **result.metrics,
    }
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_comparison(result, output / "comparison.png")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
