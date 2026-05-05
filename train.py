from dataclasses import dataclass
from pathlib import Path

import torch
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
from model import PINN, compute_boundary_loss, compute_pde_residual
from plotting import plot_displacement
from video import make_video

print("Loading environment variables...", load_dotenv())


@dataclass
class TrainingConfig:
    """PINN training configuration."""

    hidden_layers: int = 4
    hidden_units: int = 64
    learning_rate: float = 1e-3
    epochs: int = 50000
    lambda_physics: float = 1.0
    lambda_boundary: float = 10.0
    scheduler_step: int = 5000
    scheduler_gamma: float = 0.5
    batch_size: int = 2048
    log_every: int = 100
    checkpoint_every: int = 1000
    checkpoint_dir: str = "checkpoints"
    plot_dir: str = "plots"
    resume: str = ""  # Path to checkpoint to resume from
    patience: int = 10  # Early stopping patience

    def __post_init__(self):
        assert self.hidden_layers > 0, "hidden_layers must be > 0"
        assert self.hidden_units > 0, "hidden_units must be > 0"
        assert self.learning_rate > 0, "learning_rate must be > 0"
        assert self.epochs > 0, "epochs must be > 0"
        assert self.lambda_physics >= 0, "lambda_physics must be >= 0"
        assert self.lambda_boundary >= 0, "lambda_boundary must be >= 0"
        assert self.scheduler_step > 0, "scheduler_step must be > 0"
        assert self.scheduler_gamma > 0 and self.scheduler_gamma < 1, (
            "scheduler_gamma must be in (0,1)"
        )
        assert self.batch_size > 0, "batch_size must be > 0"
        assert self.log_every > 0, "log_every must be > 0"
        assert self.checkpoint_every > 0, "checkpoint_every must be > 0"
        assert self.patience >= 0, "patience must be >= 0"
        if self.resume:
            assert Path(self.resume).is_file(), (
                f"Checkpoint file {self.resume} does not exist"
            )
        assert self.checkpoint_every % self.log_every == 0, (
            "checkpoint_every must be a multiple of log_every"
        )
        assert self.log_every % self.patience == 0, (
            "log_every must be a multiple of patience"
        )


def main(config: TrainingConfig):
    # --- WANDB INIT ---
    wandb.init(project="PINN_CLT", config=vars(config))

    # --- MODEL SETUP ---
    model = PINN(
        hidden_layers=config.hidden_layers,
        hidden_units=config.hidden_units,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.scheduler_step,
        gamma=config.scheduler_gamma,
    )

    # --- RESUME FROM CHECKPOINT ---
    start_epoch = 1
    best_loss = float("inf")
    if config.resume:
        start_epoch, best_loss = load_checkpoint(
            config.resume, model, optimizer, scheduler, device
        )
        start_epoch += 1  # Start from next epoch
        print(f"Resumed from epoch {start_epoch - 1}, best_loss={best_loss:.6e}")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    wandb.watch(model, log="all", log_freq=500)

    # --- EARLY STOPPING ---
    early_stop = EarlyStopping(patience=config.patience)

    # --- TRAINING LOOP ---
    pbar = trange(start_epoch, config.epochs + 1, desc="Training")
    loss_accumulator = 0.0
    loss_count = 0
    for epoch in pbar:
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # Physics loss - resample fresh collocation points each epoch
        xy_batch = torch.cat(
            [
                torch.rand(config.batch_size, 1, device=device) * PLATE_LENGTH,
                torch.rand(config.batch_size, 1, device=device) * PLATE_WIDTH,
            ],
            dim=1,
        )
        residual = compute_pde_residual(model, xy_batch, material_props)
        loss_physics = torch.mean(residual**2)

        # Boundary loss
        loss_boundary = compute_boundary_loss(model, boundary_data)

        # Total loss
        total_loss = (
            config.lambda_physics * loss_physics
            + config.lambda_boundary * loss_boundary
        )

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        current_loss = total_loss.item()
        loss_accumulator += current_loss
        loss_count += 1

        # Logging
        if epoch % config.log_every == 0 or epoch == 1:
            avg_loss = loss_accumulator / loss_count
            wandb.log(
                {
                    "loss/total": current_loss,
                    "loss/avg": avg_loss,
                    "loss/physics": loss_physics.item(),
                    "loss/boundary": loss_boundary.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                },
                step=epoch,
            )
            pbar.set_postfix(
                avg=f"{avg_loss:.2e}",
                phys=f"{loss_physics.item():.2e}",
                bc=f"{loss_boundary.item():.2e}",
            )

            # Early stopping on avg loss
            if early_stop.step(avg_loss):
                print(f"\nEarly stopping at epoch {epoch} (avg_loss={avg_loss:.6e})")
                break

            # Checkpoint + plot
            if epoch % config.checkpoint_every == 0:
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    # Save best model separately
                    save_checkpoint(
                        Path(config.checkpoint_dir) / "best.pt",
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        best_loss,
                    )
                save_checkpoint(
                    Path(config.checkpoint_dir) / f"ckpt_epoch_{epoch:06d}.pt",
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_loss,
                )
                # Plot displacement
                fig_path = plot_displacement(model, epoch, save_dir=config.plot_dir)
                wandb.log({"displacement": wandb.Image(str(fig_path))}, step=epoch)

            # Reset accumulator
            loss_accumulator = 0.0
            loss_count = 0

    # --- FINAL SAVE & PLOT ---
    save_checkpoint(
        Path(config.checkpoint_dir) / "final.pt",
        model,
        optimizer,
        scheduler,
        epoch,
        best_loss,
    )
    plot_displacement(model, epoch, save_dir=config.plot_dir)

    # --- GENERATE VIDEO ---
    video_path = "displacement_evolution.mp4"
    make_video(plot_dir=config.plot_dir, output_path=video_path)
    wandb.log({"video": wandb.Video(video_path)}, step=epoch)

    wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(TrainingConfig, dest="training_config")
    args = parser.parse_args()
    config = getattr(args, "training_config")
    main(config)
