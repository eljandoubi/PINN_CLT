"""End-to-end smoke tests for the full training pipeline (train.main).

These exercise the real training loop, checkpointing, plotting, and video
generation together. W&B runs offline (see tests/conftest.py) so no network
access or credentials are required.
"""

import pytest
import torch

from train import TrainingConfig, main


def _tiny_config(tmp_path, **overrides):
    defaults = {
        "hidden_layers": 1,
        "hidden_units": 4,
        "batch_size": 8,
        "log_every": 1,
        "checkpoint_every": 1,
        "patience": 100,
        "runs_dir": str(tmp_path / "runs"),
    }
    defaults.update(overrides)
    return TrainingConfig(**defaults)


@pytest.fixture(autouse=True)
def _offline_wandb_dir(tmp_path, monkeypatch):
    """Keep W&B's local run files inside tmp_path instead of the repo root."""
    wandb_dir = tmp_path / "wandb"
    wandb_dir.mkdir()
    monkeypatch.setenv("WANDB_DIR", str(wandb_dir))


@pytest.mark.slow
class TestMainSmoke:
    def test_full_training_run_end_to_end(self, tmp_path):
        # log_every < checkpoint_every so some epochs log-without-checkpointing
        # and some skip logging entirely, exercising both loop branches.
        config = _tiny_config(
            tmp_path,
            epochs=8,
            log_every=2,
            checkpoint_every=4,
            run_id="pytest-smoke",
        )
        main(config)

        run_dir = tmp_path / "runs" / "pytest-smoke"
        assert (run_dir / "checkpoints" / "final.pt").exists()
        assert (run_dir / "checkpoints" / "best.pt").exists()
        assert (run_dir / "displacement_evolution.mp4").exists()
        assert any((run_dir / "plots").glob("w_epoch_*_3d.png"))

    def test_adaptive_weights_and_lbfgs_run(self, tmp_path):
        """Cover the adaptive-loss-weighting, periodic-reset, and L-BFGS paths."""
        config = _tiny_config(
            tmp_path,
            epochs=2,
            run_id="pytest-smoke-lbfgs",
            adaptive_weights=True,
            reset_period=1,
            use_lbfgs=True,
            lbfgs_warmup=1,
            lbfgs_max_iter=2,
        )
        main(config)

        run_dir = tmp_path / "runs" / "pytest-smoke-lbfgs"
        assert (run_dir / "checkpoints" / "final.pt").exists()


@pytest.mark.slow
class TestTrainResume:
    def test_resume_continues_from_checkpoint(self, tmp_path):
        # adaptive_weights must match across runs: the optimizer's saved
        # param groups have to line up with the resumed optimizer's groups.
        first = _tiny_config(
            tmp_path, epochs=1, run_id="pytest-resume-a", adaptive_weights=True
        )
        main(first)
        best_ckpt = tmp_path / "runs" / "pytest-resume-a" / "checkpoints" / "best.pt"
        assert best_ckpt.exists()

        second = _tiny_config(
            tmp_path,
            epochs=2,
            run_id="pytest-resume-b",
            resume=str(best_ckpt),
            adaptive_weights=True,
            use_lbfgs=True,
            lbfgs_warmup=1,
            lbfgs_max_iter=2,
        )
        main(second)

        final_ckpt = tmp_path / "runs" / "pytest-resume-b" / "checkpoints" / "final.pt"
        assert final_ckpt.exists()
        state = torch.load(final_ckpt, map_location="cpu", weights_only=True)
        assert state["epoch"] == 2
