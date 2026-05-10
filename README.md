# PINN-CLT: Physics-Informed Neural Network for Classical Lamination Theory

A Physics-Informed Neural Network (PINN) for solving the orthotropic plate bending problem governed by Classical Lamination Theory (CLT).

The network learns the transverse displacement field `w(x, y)` by minimizing the residual of the governing PDE:
```
D11·∂⁴w/∂x⁴ + 2(D12 + 2·D66)·∂⁴w/∂x²∂y² + D22·∂⁴w/∂y⁴ = q
```

## Features

- **Orthotropic material model** — T300/5208 carbon fiber with full [Q] and [D] stiffness matrices
- **Residual blocks** — optional ResNet-style MLP with LayerNorm for stable training
- **Adaptive loss weighting** — learnable task weights via homoscedastic uncertainty (Kendall et al., 2018)
- **Multiple loss functions** — MSE, Huber, Reverse Huber, and L1
- **Automatic differentiation** — 4th-order derivatives computed via PyTorch autograd
- **Mini-batch collocation** — fresh random domain points resampled each epoch
- **Checkpointing & resume** — save/load full training state (model, optimizer, scheduler, L-BFGS)
- **Early stopping** — configurable patience-based stopping on averaged loss
- **L-BFGS fine-tuning** — optional switch to L-BFGS optimizer after Adam warmup for sharper convergence
- **W&B logging** — losses, learning rate, 3D displacement plots, training video, and model artifact upload
- **CLI configuration** — all hyperparameters configurable via `simple-parsing`

## Project Structure

```
├── data.py             # Material properties, geometry, boundary data generation
├── model.py            # PINN architecture (network, residual blocks, adaptive weights)
├── losses.py           # PDE residual, boundary/natural BC losses, gradient clipping
├── train.py            # Training loop with checkpointing, early stopping, W&B
├── checkpoint.py       # Save/load checkpoint utilities
├── early_stopping.py   # Early stopping class
├── plotting.py         # 3D displacement field w(x,y) visualization
├── video.py            # Generate MP4/GIF video from plot frames
├── tests/              # Test suite (pytest)
│   ├── test_checkpoint.py
│   ├── test_data.py
│   ├── test_early_stopping.py
│   ├── test_losses.py
│   ├── test_model.py
│   └── test_train.py
├── pyproject.toml      # Project metadata & dependencies (managed by uv)
└── LICENSE             # Apache 2.0
```

## Installation

```bash
# Clone the repository
git clone https://github.com/eljandoubi/PINN_CLT.git && cd PINN_CLT

# Install dependencies (requires uv)
uv sync
```

### System FFmpeg (optional, for MP4 video export)

`imageio-ffmpeg` bundles its own ffmpeg binary. If you prefer system ffmpeg or encounter issues:

<details>
<summary>Platform-specific install instructions</summary>

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
</details>

## Usage

### Train from scratch

```bash
uv run train.py
```

### Custom hyperparameters

```bash
uv run train.py --epochs 50000 --learning_rate 5e-4 --batch_size 4096 --patience 20
```

### With residual blocks and LayerNorm

```bash
uv run train.py --use_residual true --use_norm true --hidden_layers 6 --hidden_units 128
```

### With adaptive loss weighting

```bash
uv run train.py --adaptive_weights true --lambda_physics 10.0 --lambda_boundary 1.0 --lambda_natural 1.0
```

### With L-BFGS after Adam warmup

```bash
uv run train.py --use_lbfgs true --lbfgs_warmup 10000 --lbfgs_lr 1.0
```

### Resume from checkpoint

```bash
uv run train.py --resume runs/<run_id>/checkpoints/best.pt --run_id <run_id>
```

### Generate video from existing plots

```bash
uv run video.py
```

### Run tests

```bash
uv run pytest
```

### All available options

```bash
uv run train.py --help
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_layers` | 4 | Number of hidden layers / residual blocks |
| `hidden_units` | 128 | Neurons per hidden layer |
| `activation` | `tanh` | Activation function (`tanh`, `silu`, `gelu`, `softplus`, `mish`) |
| `loss_fn` | `mse` | Loss function (`mse`, `huber`, `reverse_huber`, `l1`) |
| `learning_rate` | 1e-3 | Adam learning rate |
| `max_grad_norm` | 1.0 | Maximum gradient norm for clipping |
| `epochs` | 100000 | Maximum training epochs |
| `batch_size` | 4096 | Collocation points per epoch |
| `lambda_physics` | 1.0 | PDE loss weight |
| `lambda_boundary` | 1.0 | Boundary condition loss weight |
| `lambda_natural` | 1.0 | Natural BC loss weight (Mx, My, Vy) |
| `scheduler_step` | 10000 | LR decay step interval |
| `scheduler_gamma` | 0.5 | LR decay factor |
| `patience` | 10 | Early stopping patience (checked every `log_every` epochs) |
| `log_every` | 1000 | Logging frequency (epochs) |
| `checkpoint_every` | 1000 | Checkpoint & plot frequency (epochs) |
| `use_residual` | `false` | Use ResNet-like residual blocks |
| `use_norm` | `false` | Apply LayerNorm inside residual blocks |
| `use_ffmlp` | `false` | Use gated feed-forward MLP blocks (transformer-style) |
| `normalize` | `false` | Normalize PDE/natural BC losses by D11 for stability |
| `adaptive_weights` | `false` | Learnable adaptive loss weighting (Kendall et al.) |
| `use_lbfgs` | `false` | Switch to L-BFGS optimizer after Adam warmup |
| `lbfgs_warmup` | 10000 | Number of Adam epochs before switching to L-BFGS |
| `lbfgs_max_iter` | 20 | Max iterations per L-BFGS step |
| `lbfgs_history_size` | 50 | L-BFGS history size |
| `lbfgs_lr` | 1.0 | L-BFGS learning rate |
| `reset_period` | `None` | Reset adaptive weights every N epochs (must be multiple of `log_every`) |
| `runs_dir` | `runs` | Base directory for all run outputs |
| `run_id` | `None` | W&B run ID (auto-generated if omitted) |
| `resume` | `""` | Path to checkpoint for resuming |

## Boundary Conditions

| Edge | Type | Conditions |
|------|------|------------|
| x = 0 | Fixed (clamped) | w = 0, ∂w/∂x = 0 |
| x = L | Simply supported | w = 0, Mx = 0 |
| y = 0 | Free | My = 0, Vy = 0 |
| y = W | Free | My = 0, Vy = 0 |

## Material Properties (T300/5208 Carbon Fiber)

| Property | Value |
|----------|-------|
| E₁ | 181 GPa |
| E₂ | 10.3 GPa |
| G₁₂ | 7.17 GPa |
| ν₁₂ | 0.28 |
| Thickness h | 5 mm |
| Plate L × W | 1.0 m × 0.5 m |
| Pressure q | 10 kPa |

## License

Apache 2.0 — see [LICENSE](LICENSE).
