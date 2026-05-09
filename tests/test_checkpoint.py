"""Tests for checkpoint.py — save/load round-trip."""

import torch

from checkpoint import load_checkpoint, save_checkpoint
from model import PINN, AdaptiveLossWeights


def _make_training_state(use_adaptive=False, use_lbfgs=False):
    """Create a minimal training state for testing."""
    model = PINN(hidden_layers=2, hidden_units=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    adaptive = (
        AdaptiveLossWeights(num_losses=3, initial_weights=[1.0, 1.0, 1.0])
        if use_adaptive
        else None
    )

    lbfgs = (
        torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=5) if use_lbfgs else None
    )

    # Step once to populate optimizer state
    xy = torch.rand(8, 2)
    loss = model(xy).pow(2).mean()
    loss.backward()
    optimizer.step()
    scheduler.step()

    return model, optimizer, scheduler, adaptive, lbfgs


class TestCheckpoint:
    def test_save_load_basic(self, tmp_path):
        model, optimizer, scheduler, _, _ = _make_training_state()
        path = tmp_path / "ckpt.pt"

        save_checkpoint(path, model, optimizer, scheduler, epoch=42, best_loss=0.123)
        assert path.exists()

        model2, optimizer2, scheduler2, _, _ = _make_training_state()
        epoch, best_loss, using_lbfgs = load_checkpoint(
            path, model2, optimizer2, scheduler2, torch.device("cpu")
        )

        assert epoch == 42
        assert best_loss == 0.123
        assert using_lbfgs is False

        # Model weights should match
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            torch.testing.assert_close(p1, p2)

    def test_save_load_with_adaptive(self, tmp_path):
        model, optimizer, scheduler, adaptive, _ = _make_training_state(
            use_adaptive=True
        )
        path = tmp_path / "ckpt_adapt.pt"

        # Perturb adaptive weights
        with torch.no_grad():
            adaptive.log_vars.fill_(5.0)

        save_checkpoint(
            path,
            model,
            optimizer,
            scheduler,
            epoch=10,
            best_loss=0.5,
            adaptive_weighter=adaptive,
        )

        _, _, _, adaptive2, _ = _make_training_state(use_adaptive=True)
        load_checkpoint(
            path,
            model,
            optimizer,
            scheduler,
            torch.device("cpu"),
            adaptive_weighter=adaptive2,
        )

        torch.testing.assert_close(adaptive2.log_vars.data, adaptive.log_vars.data)

    def test_save_load_lbfgs_flag(self, tmp_path):
        model, optimizer, scheduler, _, lbfgs = _make_training_state(use_lbfgs=True)
        path = tmp_path / "ckpt_lbfgs.pt"

        save_checkpoint(
            path,
            model,
            optimizer,
            scheduler,
            epoch=100,
            best_loss=0.01,
            lbfgs_optimizer=lbfgs,
            using_lbfgs=True,
        )

        model2, optimizer2, scheduler2, _, lbfgs2 = _make_training_state(use_lbfgs=True)
        epoch, best_loss, using_lbfgs = load_checkpoint(
            path,
            model2,
            optimizer2,
            scheduler2,
            torch.device("cpu"),
            lbfgs_optimizer=lbfgs2,
        )

        assert using_lbfgs is True
        assert epoch == 100

    def test_load_missing_lbfgs_key_defaults_false(self, tmp_path):
        """Checkpoints saved before the lbfgs feature should default using_lbfgs=False."""
        model, optimizer, scheduler, _, _ = _make_training_state()
        path = tmp_path / "old_ckpt.pt"

        # Simulate old checkpoint without using_lbfgs key
        data = {
            "epoch": 5,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_loss": 1.0,
        }
        torch.save(data, path)

        model2, optimizer2, scheduler2, _, _ = _make_training_state()
        _, _, using_lbfgs = load_checkpoint(
            path, model2, optimizer2, scheduler2, torch.device("cpu")
        )
        assert using_lbfgs is False
