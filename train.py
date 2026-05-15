from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import wandb
from dotenv import load_dotenv
from simple_parsing import ArgumentParser
from tqdm import trange

from checkpoint import load_checkpoint, save_checkpoint
from data import (
    PLATE_LENGTH,
    PLATE_WIDTH,
    boundary_data,
    device,
    material_props,
)
from early_stopping import EarlyStopping
from losses import clip_grads, compute_losses
from model import (
    PINN,
    AdaptiveLossWeights,
    CosActivation,
    GaussianActivation,
    ReverseHuberLoss,
    SinActivation,
)
from plotting import plot_displacement_3d
from video import make_video

print("Loading environment variables...", load_dotenv())

# --- DATA SUMMARY ---
print(f"\nUsing device: {device}")
print("\n--- Orthotropic Bending Stiffnesses ---")
print(f"D11 = {material_props['D11']:.4f} N·m")
print(f"D22 = {material_props['D22']:.4f} N·m")
print(f"D12 = {material_props['D12']:.4f} N·m")
print(f"D66 = {material_props['D66']:.4f} N·m")
print("\n--- Boundary Data Summary ---")
print(f"Fixed edge points: {boundary_data['fixed_edge']['x'].shape}")
print(f"Simply supported points: {boundary_data['simply_supported']['xy'].shape}")
print(f"Free edge y=0 points: {boundary_data['free_edge_y0']['x'].shape}")
print(f"Free edge y=W points: {boundary_data['free_edge_yW']['x'].shape}")
print("\nGoverning PDE: D11·∂⁴w/∂x⁴ + 2(D12+2D66)·∂⁴w/∂x²∂y² + D22·∂⁴w/∂y⁴ = q")


@dataclass
class TrainingConfig:
    """PINN training configuration."""

    hidden_layers: int = 4
    hidden_units: int = 128
    activation: Literal["tanh", "silu", "gelu", "softplus", "mish", "sigmoid", "logsigmoid", "tanhshrink", "gaussian", "sin", "cos"] = "tanh"
    loss_fn: Literal["mse", "huber", "reverse_huber", "l1"] = "mse"
    learning_rate: float = 1e-3
    epochs: int = 100000
    lambda_physics: float = 1.0
    lambda_boundary: float = 1.0
    lambda_natural: float = 1.0
    scheduler_step: int = 10000
    scheduler_gamma: float = 0.5
    max_grad_norm: float = 1.0
    batch_size: int = int(2**12)
    reset_period: int | None = (
        None  # If set, resets adaptive weights every N epochs (for long runs)
    )
    log_every: int = 1000
    checkpoint_every: int = 1000
    runs_dir: str | Path = "runs"
    checkpoint_dir: Path = Path("checkpoints")
    plot_dir: Path = Path("plots")
    resume: str = ""  # Path to checkpoint to resume from
    patience: int = 10  # Early stopping patience
    use_residual: bool = False  # Use ResNet-like residual blocks
    use_norm: bool = False  # Apply LayerNorm inside residual blocks
    use_ffmlp: bool = False  # Use Feed-Forward MLP
    normalize: bool = False  # Normalize PDE/natural BC losses by D11
    adaptive_weights: bool = False  # Use learnable adaptive loss weighting
    use_lbfgs: bool = False  # Switch to L-BFGS after warmup
    only_counter: bool = True  # Only reset early stopping counter (not best_loss) when resetting adaptive weights
    lbfgs_warmup: int = 10000  # Number of Adam warmup epochs before switching to L-BFGS
    lbfgs_max_iter: int = 20  # Max iterations per L-BFGS step
    lbfgs_history_size: int = 50  # L-BFGS history size
    lbfgs_lr: float = 1.0  # L-BFGS learning rate
    run_id: str | None = None  # Optional run ID for logging (overrides auto-generated)

    def __post_init__(self) -> None:
        assert self.hidden_layers > 0, "hidden_layers must be > 0"
        assert self.hidden_units > 0, "hidden_units must be > 0"
        assert self.activation in ("tanh", "silu", "gelu", "softplus", "mish", "sigmoid", "logsigmoid", "tanhshrink", "gaussian", "sin", "cos"), (
            f"activation must be one of tanh, silu, gelu, softplus, mish, sigmoid, logsigmoid, tanhshrink, gaussian, sin, cos; got {self.activation}"
        )
        assert self.loss_fn in ("mse", "huber", "reverse_huber", "l1"), (
            f"loss_fn must be one of mse, huber, reverse_huber, l1; got {self.loss_fn}"
        )
        assert self.learning_rate > 0, "learning_rate must be > 0"
        assert self.epochs > 0, "epochs must be > 0"
        assert self.lambda_physics >= 0, "lambda_physics must be >= 0"
        assert self.lambda_boundary >= 0, "lambda_boundary must be >= 0"
        assert self.lambda_natural >= 0, "lambda_natural must be >= 0"
        assert self.scheduler_step > 0, "scheduler_step must be > 0"
        assert self.scheduler_gamma > 0 and self.scheduler_gamma < 1, (
            "scheduler_gamma must be in (0,1)"
        )
        assert self.max_grad_norm > 0, "max_grad_norm must be > 0"
        assert self.batch_size > 0, "batch_size must be > 0"
        assert self.log_every > 0, "log_every must be > 0"
        assert self.checkpoint_every > 0, "checkpoint_every must be > 0"
        assert self.patience >= 0, "patience must be >= 0"
        if self.use_lbfgs:
            assert self.lbfgs_warmup > 0, "lbfgs_warmup must be > 0"
            assert self.lbfgs_max_iter > 0, "lbfgs_max_iter must be > 0"
        if self.resume:
            assert Path(self.resume).is_file(), (
                f"Checkpoint file {self.resume} does not exist"
            )
        assert self.checkpoint_every % self.log_every == 0, (
            "checkpoint_every must be a multiple of log_every"
        )
        if self.reset_period is not None:
            assert self.reset_period % self.log_every == 0, (
                "reset_period must be a multiple of log_every"
            )

    def set_id(self, run_id: str) -> None:
        """Set run ID (for logging) after initialization."""
        self.run_id = run_id

    def update_paths(self) -> None:
        """Update checkpoint and plot directories based on base run directory."""
        assert self.run_id is not None, "run_id must be set before updating paths"

        self.run_dir = Path(self.runs_dir) / self.run_id
        self.checkpoint_dir = self.run_dir / Path(self.checkpoint_dir)
        self.plot_dir = self.run_dir / Path(self.plot_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        assert isinstance(self.checkpoint_dir, Path) and isinstance(
            self.plot_dir, Path
        ), "checkpoint_dir and plot_dir must be Path objects after update_paths()"


def main(config: TrainingConfig) -> None:
    # --- WANDB INIT ---
    run = wandb.init(
        project="PINN_CLT",
        config=vars(config),
        id=config.run_id,
        resume="allow" if config.run_id else None,
    )
    config.set_id(run.id)
    config.update_paths()

    # Update wandb config with run paths
    wandb.config.update(
        {
            "run_dir": str(config.run_dir),
            "run_id": run.id,
            "checkpoint_dir": str(config.checkpoint_dir),
            "plot_dir": str(config.plot_dir),
        },
        allow_val_change=True,
    )

    # --- MODEL SETUP ---
    activation_map = {
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
        "gelu": nn.GELU,
        "softplus": nn.Softplus,
        "mish": nn.Mish,
        "sigmoid": nn.Sigmoid,
        "logsigmoid": nn.LogSigmoid,
        "tanhshrink": nn.Tanhshrink,
        "gaussian": GaussianActivation,
        "sin": SinActivation,
        "cos": CosActivation,
    }
    loss_fn_map = {
        "mse": nn.MSELoss(),
        "huber": nn.HuberLoss(),
        "reverse_huber": ReverseHuberLoss(),
        "l1": nn.L1Loss(),
    }
    criterion = loss_fn_map[config.loss_fn]
    model = PINN(
        hidden_layers=config.hidden_layers,
        hidden_units=config.hidden_units,
        activation=activation_map[config.activation],
        use_residual=config.use_residual,
        use_norm=config.use_norm,
        use_ffmlp=config.use_ffmlp,
    ).to(device)

    # --- ADAPTIVE LOSS WEIGHTING ---
    adaptive_weighter = (
        AdaptiveLossWeights(
            num_losses=3,
            initial_weights=[
                config.lambda_physics,
                config.lambda_boundary,
                config.lambda_natural,
            ],
        ).to(device)
        if config.adaptive_weights
        else None
    )
    eff_weights = [
        config.lambda_physics,
        config.lambda_boundary,
        config.lambda_natural,
    ]
    params = list(model.parameters())
    if adaptive_weighter is not None:
        params += list(adaptive_weighter.parameters())
    optimizer = torch.optim.Adam(params, lr=config.learning_rate)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.scheduler_step,
        gamma=config.scheduler_gamma,
    )

    lbfgs_optimizer = None  # Will be created after warmup
    using_lbfgs = False

    # --- RESUME FROM CHECKPOINT ---
    start_epoch = 1
    best_loss = float("inf")
    if config.resume:
        # Pre-create L-BFGS optimizer so its state can be restored
        if config.use_lbfgs:
            lbfgs_params = list(model.parameters())
            if adaptive_weighter is not None:
                lbfgs_params += list(adaptive_weighter.parameters())
            lbfgs_optimizer = torch.optim.LBFGS(
                lbfgs_params,
                lr=config.lbfgs_lr,
                max_iter=config.lbfgs_max_iter,
                history_size=config.lbfgs_history_size,
                line_search_fn="strong_wolfe",
            )
        start_epoch, best_loss, using_lbfgs = load_checkpoint(
            config.resume,
            model,
            optimizer,
            scheduler,
            device,
            adaptive_weighter,
            lbfgs_optimizer,
        )
        start_epoch += 1  # Start from next epoch
        print(f"Resumed from epoch {start_epoch - 1}, best_loss={best_loss:.6e}")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    wandb.watch(model, log="all", log_freq=500)
    if adaptive_weighter is not None:
        wandb.watch(adaptive_weighter, log="all", log_freq=500)

    # --- EARLY STOPPING ---
    early_stop = EarlyStopping(patience=config.patience)

    # --- TRAINING LOOP ---
    pbar = trange(start_epoch, config.epochs + 1, desc="Training")
    loss_accumulator = 0.0
    physics_accumulator = 0.0
    loss_count = 0
    epoch = start_epoch  # Default in case loop never runs
    for epoch in pbar:
        model.train()

        # Switch to L-BFGS after warmup
        if (
            config.use_lbfgs
            and not using_lbfgs
            and epoch >= config.lbfgs_warmup + start_epoch
        ):
            print(f"\nSwitching to L-BFGS optimizer at epoch {epoch}")
            lbfgs_params = list(model.parameters())
            if adaptive_weighter is not None:
                lbfgs_params += list(adaptive_weighter.parameters())
            lbfgs_optimizer = torch.optim.LBFGS(
                lbfgs_params,
                lr=config.lbfgs_lr,
                max_iter=config.lbfgs_max_iter,
                history_size=config.lbfgs_history_size,
                line_search_fn="strong_wolfe",
            )
            using_lbfgs = True

        # Physics loss - resample fresh collocation points each epoch
        xy_batch = torch.cat(
            [
                torch.rand(config.batch_size, 1, device=device) * PLATE_LENGTH,
                torch.rand(config.batch_size, 1, device=device) * PLATE_WIDTH,
            ],
            dim=1,
        )

        if using_lbfgs:
            # L-BFGS needs a closure because it re-evaluates the loss
            # multiple times per step during its line search.
            total_loss = loss_physics = loss_boundary = loss_natural = torch.tensor(0.0)

            def closure():
                nonlocal \
                    total_loss, \
                    loss_physics, \
                    loss_boundary, \
                    loss_natural, \
                    eff_weights
                lbfgs_optimizer.zero_grad(set_to_none=True)  # type: ignore[union-attr]
                total_loss, loss_physics, loss_boundary, loss_natural, eff_weights = (
                    compute_losses(
                        model,
                        xy_batch,
                        material_props,
                        boundary_data,
                        criterion,
                        config.lambda_physics,
                        config.lambda_boundary,
                        config.lambda_natural,
                        adaptive_weighter,
                        config.normalize,
                    )
                )
                total_loss.backward()
                clip_grads(model, config.max_grad_norm, adaptive_weighter)
                return total_loss

            lbfgs_optimizer.step(closure)  # type: ignore[union-attr]
        else:
            optimizer.zero_grad(set_to_none=True)
            total_loss, loss_physics, loss_boundary, loss_natural, eff_weights = (
                compute_losses(
                    model,
                    xy_batch,
                    material_props,
                    boundary_data,
                    criterion,
                    config.lambda_physics,
                    config.lambda_boundary,
                    config.lambda_natural,
                    adaptive_weighter,
                    config.normalize,
                )
            )
            total_loss.backward()
            clip_grads(model, config.max_grad_norm, adaptive_weighter)
            optimizer.step()
            scheduler.step()

        current_loss = total_loss.item()
        loss_accumulator += current_loss
        physics_accumulator += loss_physics.item()
        loss_count += 1

        # Logging
        if epoch % config.log_every == 0:
            avg_loss = loss_accumulator / loss_count
            avg_physics = physics_accumulator / loss_count
            log_dict = {
                "loss/total": current_loss,
                "loss/avg": avg_loss,
                "loss/physics": loss_physics.item(),
                "loss/avg_physics": avg_physics,
                "loss/boundary": loss_boundary.item(),
                "loss/natural": loss_natural.item(),
                "loss/ratio_phys_bc": loss_physics.item()
                / (loss_boundary.item() + 1e-12),
                "loss/ratio_phys_nat": loss_physics.item()
                / (loss_natural.item() + 1e-12),
                "weights/physics": eff_weights[0],
                "weights/boundary": eff_weights[1],
                "weights/natural": eff_weights[2],
                "lr": (lbfgs_optimizer or optimizer).param_groups[0]["lr"],
            }
            wandb.log(log_dict, step=epoch)
            pbar.set_postfix(
                avg=f"{avg_loss:.2e}",
                phys=f"{loss_physics.item():.2e}",
                bc=f"{loss_boundary.item():.2e}",
                nat=f"{loss_natural.item():.2e}",
            )

            # Periodically reset adaptive weights to initial values
            if (
                adaptive_weighter is not None
                and config.reset_period is not None
                and epoch % config.reset_period == 0
            ):
                adaptive_weighter.reset()
                # Clear optimizer state for adaptive weight params
                for p in adaptive_weighter.parameters():
                    if p in optimizer.state:
                        del optimizer.state[p]
                    if lbfgs_optimizer is not None and p in lbfgs_optimizer.state:
                        del lbfgs_optimizer.state[p]
                early_stop.reset(only_counter=config.only_counter)

            # Early stopping on avg physics loss
            if early_stop.step(avg_physics):
                print(
                    f"\nEarly stopping at epoch {epoch} (avg_physics={avg_physics:.6e})"
                )
                break

            # Checkpoint + plot
            if epoch % config.checkpoint_every == 0:
                if avg_physics < best_loss:
                    best_loss = avg_physics
                    # Save best model separately
                    save_checkpoint(
                        config.checkpoint_dir / "best.pt",
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        best_loss,
                        adaptive_weighter,
                        lbfgs_optimizer,
                        using_lbfgs,
                    )
                save_checkpoint(
                    config.checkpoint_dir / f"ckpt_epoch_{epoch:06d}.pt",
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_loss,
                    adaptive_weighter,
                    lbfgs_optimizer,
                    using_lbfgs,
                )
                # Plot 3D displacement
                fig_path = plot_displacement_3d(model, epoch, save_dir=config.plot_dir)
                wandb.log({"displacement": wandb.Image(str(fig_path))}, step=epoch)

            # Reset accumulator
            loss_accumulator = 0.0
            physics_accumulator = 0.0
            loss_count = 0

    # --- FINAL SAVE & PLOT ---
    final_epoch = epoch
    save_checkpoint(
        config.checkpoint_dir / "final.pt",
        model,
        optimizer,
        scheduler,
        final_epoch,
        best_loss,
        adaptive_weighter,
        lbfgs_optimizer,
        using_lbfgs,
    )
    plot_displacement_3d(model, final_epoch, save_dir=config.plot_dir)

    # --- GENERATE VIDEO ---
    video_path = config.run_dir / "displacement_evolution.mp4"
    make_video(plot_dir=config.plot_dir, output_path=video_path)
    wandb.log({"video": wandb.Video(str(video_path))}, step=final_epoch)

    # --- UPLOAD CHECKPOINTS TO WANDB ---
    artifact = wandb.Artifact(
        f"model-{run.id}", type="model", description="Model checkpoints"
    )
    artifact.add_dir(str(config.checkpoint_dir))
    run.log_artifact(artifact)

    wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(TrainingConfig, dest="training_config")
    args = parser.parse_args()
    config = getattr(args, "training_config")
    main(config)
