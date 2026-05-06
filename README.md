# PINN-CLT: Physics-Informed Neural Network for Classical Lamination Theory

A Physics-Informed Neural Network (PINN) for solving the orthotropic plate bending problem governed by Classical Lamination Theory (CLT).

The network learns the transverse displacement field `w(x, y)` by minimizing the residual of the governing PDE:

```
D11·∂⁴w/∂x⁴ + 2(D12 + 2·D66)·∂⁴w/∂x²∂y² + D22·∂⁴w/∂y⁴ = q
```

## Features

- **Orthotropic material model** — T300/5208 carbon fiber with full [Q] and [D] stiffness matrices
- **Automatic differentiation** — 4th-order derivatives computed via PyTorch autograd
- **Mini-batch collocation** — fresh random domain points resampled each epoch for memory efficiency
- **Checkpointing & resume** — save/load full training state
- **Early stopping** — configurable patience-based stopping
- **W&B logging** — losses, learning rate, displacement plots, and training video
- **CLI configuration** — all hyperparameters configurable via command-line arguments

## Project Structure

```
├── data.py             # Material properties, geometry, boundary data generation
├── model.py            # PINN architecture + PDE residual + boundary loss
├── train.py            # Training loop with checkpointing, early stopping, W&B
├── checkpoint.py       # Save/load checkpoint utilities
├── early_stopping.py   # Early stopping class
├── plotting.py         # Displacement field w(x,y) visualization
├── video.py            # Generate MP4 video from plot frames
├── pyproject.toml      # Project dependencies
└── LICENSE             # Apache 2.0
```

## Installation

```bash
# Clone the repository
git clone https://github.com/eljandoubi/PINN_CLT.git && cd PINN_CLT

# Install dependencies (requires uv)
uv sync
```

### System FFmpeg (required for MP4 video export)

The `video.py` script uses `imageio-ffmpeg` which bundles its own ffmpeg binary. If you prefer using a system-installed ffmpeg or encounter issues, install it natively:

**macOS (Homebrew):**
```bash
brew install ffmpeg
```

**Ubuntu / Debian:**
```bash
sudo apt update && sudo apt install ffmpeg
```

**Fedora / RHEL:**
```bash
sudo dnf install ffmpeg
```

**Windows (winget):**
```powershell
winget install FFmpeg
```

**Windows (Chocolatey):**
```powershell
choco install ffmpeg
```

## Usage

### Train from scratch

```bash
uv run train.py
```

### Custom hyperparameters

```bash
uv run train.py --epochs 100000 --learning_rate 5e-4 --batch_size 4096 --patience 3000
```

### Resume from checkpoint

```bash
uv run train.py --resume checkpoints/best.pt --epochs 100000
```

### Generate video from existing plots

```bash
uv run video.py
```

### All available options

```bash
uv run train.py --help
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_layers` | 4 | Number of hidden layers |
| `hidden_units` | 128 | Neurons per hidden layer |
| `activation` | `tanh` | Activation function (`tanh`, `silu`, `gelu`, `softplus`, `mish`) |
| `learning_rate` | 1e-3 | Adam learning rate |
| `epochs` | 100000 | Maximum training epochs |
| `batch_size` | 16384 | Collocation points per epoch |
| `lambda_physics` | 1.0 | PDE loss weight |
| `lambda_boundary` | 1.0 | Boundary loss weight |
| `scheduler_step` | 10000 | LR decay step |
| `scheduler_gamma` | 0.5 | LR decay factor |
| `patience` | 10 | Early stopping patience |
| `checkpoint_every` | 1000 | Checkpoint frequency (epochs) |
| `resume` | `""` | Path to checkpoint for resuming |

## Boundary Conditions

- **Fixed edge** (x = 0): `w = 0`, `∂w/∂x = 0`
- **Simply supported edge** (x = L): `w = 0`

## Material Properties (T300/5208)

| Property | Value |
|----------|-------|
| E1 | 181 GPa |
| E2 | 10.3 GPa |
| G12 | 7.17 GPa |
| ν12 | 0.28 |
| Thickness h | 5 mm |

## License

Apache 2.0 — see [LICENSE](LICENSE).
Physics Informed Neural Networks applied to Classical Lamination Theory
