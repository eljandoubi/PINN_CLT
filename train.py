from dataclasses import dataclass

import torch
import wandb
from simple_parsing import ArgumentParser
from tqdm import trange

from data import (
    PLATE_LENGTH,
    PLATE_WIDTH,
    boundary_data,
    device,
    material_props,
)
from model import PINN, compute_boundary_loss, compute_pde_residual


@dataclass
class TrainingConfig:
    """PINN training configuration."""

    hidden_layers: int = 4
    hidden_units: int = 64
    learning_rate: float = 1e-3
    epochs: int = 10000
    lambda_physics: float = 1.0
    lambda_boundary: float = 10.0
    scheduler_step: int = 2000
    scheduler_gamma: float = 0.5
    batch_size: int = 2048
    log_every: int = 100


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

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    wandb.watch(model, log="all", log_freq=500)

    # --- TRAINING LOOP ---
    pbar = trange(1, config.epochs + 1, desc="Training")
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

        # Logging
        if epoch % config.log_every == 0 or epoch == 1:
            wandb.log(
                {
                    "epoch": epoch,
                    "loss/total": total_loss.item(),
                    "loss/physics": loss_physics.item(),
                    "loss/boundary": loss_boundary.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )
            pbar.set_postfix(
                total=f"{total_loss.item():.2e}",
                phys=f"{loss_physics.item():.2e}",
                bc=f"{loss_boundary.item():.2e}",
            )

    # --- SAVE MODEL ---
    torch.save(model.state_dict(), "pinn_clt_model.pth")
    wandb.save("pinn_clt_model.pth")
    wandb.finish()
    print("Training complete. Model saved to pinn_clt_model.pth")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(TrainingConfig, dest="training_config")
    args = parser.parse_args()
    config = getattr(args, "training_config")
    main(config)
