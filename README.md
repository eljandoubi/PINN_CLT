# PINN-CLT: Physics-Informed Neural Network for Classical Lamination Theory

[![CI](https://github.com/eljandoubi/PINN_CLT/actions/workflows/ci.yml/badge.svg)](https://github.com/eljandoubi/PINN_CLT/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.14+](https://img.shields.io/badge/python-3.14%2B-blue)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A Physics-Informed Neural Network (PINN) for solving the orthotropic plate bending problem governed by Classical Lamination Theory (CLT).

The network learns the transverse displacement field `w(x, y)` by minimizing the residual of the governing PDE:
```
D11·∂⁴w/∂x⁴ + 2(D12 + 2·D66)·∂⁴w/∂x²∂y² + D22·∂⁴w/∂y⁴ = q
```

## My Solution
![displacement field](assets/1779082278160.png)

An earlier checkpoint (epoch 81,000). Its colour scale spans about -0.007 m to 0.034 m, whereas the [reference solution](#reference-solution) peaks at 0.0295 m and vanishes on the clamped and simply supported edges, so this checkpoint is not converged. [`compare.py`](#comparing-a-trained-network) measures any checkpoint against the reference; a small run that reaches a relative L2 error of 2.1 percent is described under [Results](#results).

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Boundary Conditions](#boundary-conditions)
- [Material Properties](#material-properties-t3005208-carbon-fiber)
- [Reference Solution](#reference-solution)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Orthotropic material model**: T300/5208 carbon fiber with full [Q] and [D] stiffness matrices
- **Residual blocks**: optional ResNet-style MLP with LayerNorm for stable training
- **Adaptive loss weighting**: learnable task weights via homoscedastic uncertainty (Kendall et al., 2018)
- **Multiple loss functions**: MSE, Huber, Reverse Huber, and L1
- **Smooth activation functions**: all activations are C∞ (infinitely differentiable), suitable for 4th-order PDE: `tanh`, `silu`, `gelu`, `softplus`, `mish`, `sigmoid`, `logsigmoid`, `tanhshrink`, `gaussian` (exp(−x²)), `sin`, `cos`
- **Automatic differentiation**: 4th-order derivatives computed via PyTorch autograd
- **Mini-batch collocation**: fresh random domain points resampled each epoch
- **Checkpointing & resume**: save/load full training state; best model tracked by averaged physics loss
- **Early stopping**: configurable patience-based stopping on averaged physics loss
- **L-BFGS fine-tuning**: optional switch to L-BFGS optimizer after Adam warmup for sharper convergence
- **Optimizer state reset**: clears Adam/L-BFGS momentum for adaptive weight params on periodic reset
- **W&B logging**: losses, learning rate, 3D displacement plots, training video, and model artifact upload
- **CLI configuration**: all hyperparameters configurable via `simple-parsing`
- **Fully tested**: 160+ pytest cases (unit + end-to-end training smoke tests) with >90% coverage, enforced in CI

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
├── reference.py        # Ritz reference solution of the same plate problem
├── compare.py          # Compare a trained checkpoint with the reference
├── scripts/            # mutate_reference.py: hand-made faults the tests must catch
├── results/            # comparisons with the reference, mutation run
├── tests/              # Test suite (pytest)
│   ├── conftest.py          # Shared fixtures (headless plotting, offline W&B)
│   ├── test_checkpoint.py
│   ├── test_compare.py
│   ├── test_data.py
│   ├── test_early_stopping.py
│   ├── test_integration.py  # End-to-end training smoke tests
│   ├── test_losses.py
│   ├── test_model.py
│   ├── test_plotting.py
│   ├── test_reference.py
│   ├── test_train.py
│   └── test_video.py
├── .github/workflows/  # CI (lint + format + test on every push/PR)
├── justfile            # Convenience commands (`just test`, `just lint`, ...)
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
# or, with the justfile:
just test
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
| `activation` | `tanh` | Activation function (`tanh`, `silu`, `gelu`, `softplus`, `mish`, `sigmoid`, `logsigmoid`, `tanhshrink`, `gaussian`, `sin`, `cos`) |
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

## Reference Solution

The plate has no closed-form solution: the clamped, simply supported and free edges meet at corners, and the coupling term D12 bends the free edges across the width. [`reference.py`](reference.py) therefore solves the same problem with the Rayleigh-Ritz method, so a network can be measured against numbers instead of a picture.

The solution minimises the potential energy

$$U(w) = \frac12 \iint \left(D_{11} w_{xx}^2 + 2 D_{12} w_{xx} w_{yy} + D_{22} w_{yy}^2 + 4 D_{66} w_{xy}^2\right) dA - q \iint w \, dA$$

over products of $\xi^2 (1 - \xi) P_i(2\xi - 1)$ in $x = L\xi$, which already meets the clamped and simply supported conditions, and $P_j(2\eta - 1)$ in $y = W\eta$, which has no condition at the free edges ($P_n$ is the Legendre polynomial of degree $n$). The stiffness matrix is a sum of Kronecker products of one-dimensional Gauss-Legendre integrals, solved in float64. The free-edge conditions and the moment condition at $x = L$ are natural conditions of the energy, so they hold only in the limit of a large basis.

| basis (x by y) | centre deflection (m) |
| --- | --- |
| 4 x 3 | 2.7090e-2 |
| 8 x 6 | 2.7204e-2 |
| 12 x 8 (default) | 2.7205e-2 |
| 20 x 14 | 2.7205e-2 |

The peak deflection is 2.953e-2 m near x = 0.58 m, almost uniform across the width. The free-edge moment residual falls from 0.13 (4 x 3) to 0.03 (20 x 14) of D11 w_max / L^2, as a root mean square along the edges. A maximum would only measure the clamped corner, where `My` is nonzero because `w_xx` is nonzero on a clamped edge, so the two conditions cannot both hold there.

### Comparing a trained network

```bash
uv run compare.py --checkpoint runs/<run_id>/checkpoints/best.pt
```

Pass the same architecture options as in training (`--hidden_layers`, `--hidden_units`, `--activation`, `--use_residual`, `--use_norm`, `--use_ffmlp`). The script prints and saves `comparison/comparison.json`, plus `comparison/comparison.png` with the network, the reference and their difference side by side.

| key | meaning |
| --- | --- |
| `relative_l2_error` | norm of (network - reference) over the norm of the reference, on a 101 x 51 grid |
| `max_error_over_max_deflection` | largest pointwise error over the largest reference deflection |
| `*_center_deflection`, `*_max_deflection`, `*_min_deflection` | for the network and for the reference |
| `network_boundary_rms`, `reference_boundary_rms` | root mean square residual of each boundary condition: deflection and slope relative to w_max, moments relative to D11 w_max / L^2. The reference gives the yardstick. The free-edge shear condition is not measured. |

### Results

A small network trained on a laptop CPU and measured against the 12 x 8 Ritz solution. The checkpoint is in [`results/cpu_run/final.pt`](results/cpu_run/final.pt), so `uv run compare.py --checkpoint results/cpu_run/final.pt --hidden_units 64` reproduces the tables below.

```bash
uv run train.py --epochs 10000 --batch_size 512 --hidden_units 64 --normalize true \
  --scheduler_step 3000 --scheduler_gamma 0.5 --run_id small
```

The network has 4 hidden layers of 64 tanh units and sees 512 collocation points per epoch, with Adam at 1e-3 halved every 3,000 epochs. The default early stopping (patience of 10 checks, one every 100 epochs) ended the run at epoch 5,700 with a relative L2 error of 0.096. The run was then resumed from that checkpoint with early stopping switched off, up to the planned 10,000 epochs (the same options plus `--patience 1000 --resume runs/small/checkpoints/final.pt`). Training took about 31 minutes on 3 CPU threads.

| metric | network | reference |
| --- | --- | --- |
| relative L2 error | 0.0214 | |
| max error over max deflection | 0.0288 | |
| centre deflection (m) | 2.78e-2 | 2.72e-2 |
| max deflection (m) | 2.96e-2 | 2.95e-2 |

Boundary conditions, as the root mean square residuals defined above:

| condition | network | reference |
| --- | --- | --- |
| clamped deflection | 0.0120 | 0 |
| clamped slope | 0.0420 | 0 |
| supported deflection | 0.0092 | 0 |
| supported moment | 0.092 | 0.0015 |
| free-edge moment | 0.149 | 0.063 |

![PINN, Ritz reference and their difference](results/cpu_run/comparison.png)

The largest pointwise error is 2.9 percent of the peak deflection. The network meets the deflection conditions better than the moment conditions: its free-edge moment residual is more than twice the reference's.

The loss weights mattered more than the length of training. [`results/cpu_run/pilots.json`](results/cpu_run/pilots.json) holds four 600-epoch runs of the same network. With the default weights the PDE term is about 7e4 and the boundary term about 13 at the end, and the relative error is 84. With `--normalize true` it is 1.47, with adaptive weights alone 57, and with both 1.71. This is one seed and one configuration, chosen after those four short runs by looking at the same reference, so the numbers show that the pipeline works, not a held-out accuracy.

### How the reference is verified

[`tests/test_reference.py`](tests/test_reference.py) checks the solver against facts that do not come from the solver itself:

- with D12 = 0 the free edges carry no moment and the plate bends as a propped cantilever, which has a closed form. The solver reproduces it to 1e-12 on a 1.3 m x 0.8 m plate, not 1 m x 0.5 m, so a missing length or width factor cannot hide.
- the stored energy equals a direct quadrature of the energy functional, equals minus half the work of the load, and grows quadratically when the coefficients are perturbed, so the solution is the minimiser.
- in the middle of the plate (0.3 L to 0.7 L, 0.2 W to 0.8 W) the solution satisfies the governing equation to 0.17 percent of q (basis 16 x 12). Doubling D12 or D66 in the solver, or halving D66, raises this to 0.6 percent or more.
- derivatives match finite differences, the essential conditions hold to rounding, and the free-edge moment shrinks as the basis grows.

Twenty hand-made faults in the solver (wrong factors in the stiffness terms, missing length scaling, a swapped derivative, and so on) are all caught by the test suite. `uv run python scripts/mutate_reference.py` applies them one at a time to a temporary copy of the repository, and its output is in [`results/mutants.txt`](results/mutants.txt).

## Development

This project uses [uv](https://github.com/astral-sh/uv) for dependency management and [ruff](https://github.com/astral-sh/ruff) for linting/formatting. A [`justfile`](justfile) wraps the common commands:

```bash
just install       # uv sync --locked (main deps + dev tools)
just test          # run the full test suite (unit + slow smoke tests)
just test-fast     # run only the fast unit tests
just cov           # run tests with terminal + HTML coverage report
just lint          # ruff check
just fix           # ruff check --fix
just format        # ruff format
just format-check  # ruff format --check (no changes; used in CI)
just check         # lint + format-check + cov (what CI runs)
just train         # uv run train.py (extra args are forwarded, e.g. `just train --epochs 5000`)
just video         # uv run video.py
just clean         # remove caches, coverage, runs/, wandb/, checkpoints/, plots/
```

Don't have [`just`](https://github.com/casey/just) installed? Every recipe is a thin wrapper around a `uv run ...` command, see the [justfile](justfile) and run the equivalent command directly.

### Testing

The test suite (`tests/`) covers material/geometry data, model architecture, loss/PDE-residual math, checkpointing, early stopping, plotting, video generation, CLI config validation, and full end-to-end training runs (offline W&B, tiny models), with over 160 tests at >90% coverage.

Slow, end-to-end tests are marked `@pytest.mark.slow`; skip them for a fast inner-loop with `just test-fast` or `uv run pytest -m "not slow"`.

### Continuous Integration

Every push and pull request to `main` runs [`.github/workflows/ci.yml`](.github/workflows/ci.yml): `ruff check`, `ruff format --check`, and the full test suite with a coverage gate (fails below 90%).

## Contributing

1. Fork and clone the repo, then run `uv sync` (or `just install`).
2. Make your changes, adding/updating tests as needed.
3. Run `just check` locally (mirrors CI) before opening a PR.
4. Open a pull request describing the change and its motivation.

## License

Apache 2.0, see [LICENSE](LICENSE).
