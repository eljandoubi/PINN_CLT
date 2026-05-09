"""Tests for TrainingConfig validation and helper methods in train.py."""

import pytest

from train import TrainingConfig


class TestTrainingConfigDefaults:
    def test_default_construction(self):
        config = TrainingConfig()
        assert config.hidden_layers == 4
        assert config.hidden_units == 128
        assert config.activation == "tanh"
        assert config.loss_fn == "mse"
        assert config.learning_rate == 1e-3
        assert config.epochs == 100000
        assert config.patience == 10
        assert config.use_lbfgs is False
        assert config.adaptive_weights is False
        assert config.run_id is None

    def test_custom_values(self):
        config = TrainingConfig(
            hidden_layers=8,
            hidden_units=256,
            activation="silu",
            loss_fn="huber",
            learning_rate=5e-4,
            epochs=50000,
        )
        assert config.hidden_layers == 8
        assert config.hidden_units == 256
        assert config.activation == "silu"
        assert config.loss_fn == "huber"


class TestTrainingConfigValidation:
    def test_hidden_layers_zero(self):
        with pytest.raises(AssertionError, match="hidden_layers must be > 0"):
            TrainingConfig(hidden_layers=0)

    def test_hidden_layers_negative(self):
        with pytest.raises(AssertionError, match="hidden_layers must be > 0"):
            TrainingConfig(hidden_layers=-1)

    def test_hidden_units_zero(self):
        with pytest.raises(AssertionError, match="hidden_units must be > 0"):
            TrainingConfig(hidden_units=0)

    def test_invalid_activation(self):
        with pytest.raises(AssertionError, match="activation must be one of"):
            TrainingConfig(activation="relu")  # type: ignore[arg-type]

    @pytest.mark.parametrize("act", ["tanh", "silu", "gelu", "softplus", "mish"])
    def test_valid_activations(self, act):
        config = TrainingConfig(activation=act)  # type: ignore[arg-type]
        assert config.activation == act

    def test_invalid_loss_fn(self):
        with pytest.raises(AssertionError, match="loss_fn must be one of"):
            TrainingConfig(loss_fn="bce")  # type: ignore[arg-type]

    @pytest.mark.parametrize("fn", ["mse", "huber", "reverse_huber", "l1"])
    def test_valid_loss_fns(self, fn):
        config = TrainingConfig(loss_fn=fn)  # type: ignore[arg-type]
        assert config.loss_fn == fn

    def test_learning_rate_zero(self):
        with pytest.raises(AssertionError, match="learning_rate must be > 0"):
            TrainingConfig(learning_rate=0)

    def test_learning_rate_negative(self):
        with pytest.raises(AssertionError, match="learning_rate must be > 0"):
            TrainingConfig(learning_rate=-1e-3)

    def test_epochs_zero(self):
        with pytest.raises(AssertionError, match="epochs must be > 0"):
            TrainingConfig(epochs=0)

    def test_lambda_physics_negative(self):
        with pytest.raises(AssertionError, match="lambda_physics must be >= 0"):
            TrainingConfig(lambda_physics=-1.0)

    def test_lambda_physics_zero_ok(self):
        config = TrainingConfig(lambda_physics=0.0)
        assert config.lambda_physics == 0.0

    def test_lambda_boundary_negative(self):
        with pytest.raises(AssertionError, match="lambda_boundary must be >= 0"):
            TrainingConfig(lambda_boundary=-0.1)

    def test_lambda_natural_negative(self):
        with pytest.raises(AssertionError, match="lambda_natural must be >= 0"):
            TrainingConfig(lambda_natural=-0.1)

    def test_scheduler_step_zero(self):
        with pytest.raises(AssertionError, match="scheduler_step must be > 0"):
            TrainingConfig(scheduler_step=0)

    def test_scheduler_gamma_zero(self):
        with pytest.raises(AssertionError, match="scheduler_gamma must be in"):
            TrainingConfig(scheduler_gamma=0.0)

    def test_scheduler_gamma_one(self):
        with pytest.raises(AssertionError, match="scheduler_gamma must be in"):
            TrainingConfig(scheduler_gamma=1.0)

    def test_scheduler_gamma_valid(self):
        config = TrainingConfig(scheduler_gamma=0.9)
        assert config.scheduler_gamma == 0.9

    def test_max_grad_norm_zero(self):
        with pytest.raises(AssertionError, match="max_grad_norm must be > 0"):
            TrainingConfig(max_grad_norm=0)

    def test_batch_size_zero(self):
        with pytest.raises(AssertionError, match="batch_size must be > 0"):
            TrainingConfig(batch_size=0)

    def test_log_every_zero(self):
        with pytest.raises(AssertionError, match="log_every must be > 0"):
            TrainingConfig(log_every=0)

    def test_checkpoint_every_zero(self):
        with pytest.raises(AssertionError, match="checkpoint_every must be > 0"):
            TrainingConfig(checkpoint_every=0)

    def test_patience_negative(self):
        with pytest.raises(AssertionError, match="patience must be >= 0"):
            TrainingConfig(patience=-1)

    def test_patience_zero_ok(self):
        config = TrainingConfig(patience=0)
        assert config.patience == 0

    def test_checkpoint_every_not_multiple_of_log_every(self):
        with pytest.raises(
            AssertionError, match="checkpoint_every must be a multiple of log_every"
        ):
            TrainingConfig(log_every=1000, checkpoint_every=1500)

    def test_checkpoint_every_multiple_of_log_every_ok(self):
        config = TrainingConfig(log_every=500, checkpoint_every=2000)
        assert config.checkpoint_every == 2000

    def test_reset_period_not_multiple_of_log_every(self):
        with pytest.raises(
            AssertionError, match="reset_period must be a multiple of log_every"
        ):
            TrainingConfig(log_every=1000, reset_period=1500)

    def test_reset_period_multiple_ok(self):
        config = TrainingConfig(log_every=1000, reset_period=5000)
        assert config.reset_period == 5000

    def test_reset_period_none_ok(self):
        config = TrainingConfig(reset_period=None)
        assert config.reset_period is None

    def test_resume_nonexistent_file(self):
        with pytest.raises(AssertionError, match="does not exist"):
            TrainingConfig(resume="/nonexistent/path/checkpoint.pt")

    def test_resume_empty_ok(self):
        config = TrainingConfig(resume="")
        assert config.resume == ""


class TestTrainingConfigLBFGS:
    def test_lbfgs_warmup_zero_rejected(self):
        with pytest.raises(AssertionError, match="lbfgs_warmup must be > 0"):
            TrainingConfig(use_lbfgs=True, lbfgs_warmup=0)

    def test_lbfgs_max_iter_zero_rejected(self):
        with pytest.raises(AssertionError, match="lbfgs_max_iter must be > 0"):
            TrainingConfig(use_lbfgs=True, lbfgs_max_iter=0)

    def test_lbfgs_disabled_skips_validation(self):
        # When use_lbfgs=False, lbfgs_warmup=0 should not raise
        config = TrainingConfig(use_lbfgs=False, lbfgs_warmup=0, lbfgs_max_iter=0)
        assert config.use_lbfgs is False

    def test_lbfgs_valid_config(self):
        config = TrainingConfig(
            use_lbfgs=True,
            lbfgs_warmup=5000,
            lbfgs_max_iter=10,
            lbfgs_history_size=100,
            lbfgs_lr=0.5,
        )
        assert config.lbfgs_warmup == 5000
        assert config.lbfgs_lr == 0.5


class TestTrainingConfigMethods:
    def test_set_id(self):
        config = TrainingConfig()
        assert config.run_id is None
        config.set_id("test-run-123")
        assert config.run_id == "test-run-123"

    def test_update_paths_without_id_raises(self):
        config = TrainingConfig()
        with pytest.raises(AssertionError, match="run_id must be set"):
            config.update_paths()

    def test_update_paths_creates_dirs(self, tmp_path):
        config = TrainingConfig(runs_dir=str(tmp_path))
        config.set_id("test-run")
        config.update_paths()

        assert config.checkpoint_dir.exists()
        assert config.plot_dir.exists()
        assert config.run_dir == tmp_path / "test-run"
        assert "checkpoints" in str(config.checkpoint_dir)
        assert "plots" in str(config.plot_dir)

    def test_update_paths_nested_structure(self, tmp_path):
        config = TrainingConfig(runs_dir=str(tmp_path))
        config.set_id("my-run")
        config.update_paths()

        assert config.checkpoint_dir == tmp_path / "my-run" / "checkpoints"
        assert config.plot_dir == tmp_path / "my-run" / "plots"
