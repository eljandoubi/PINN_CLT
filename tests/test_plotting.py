"""Tests for plotting.py — 3D displacement field visualization."""

from pathlib import Path

from PIL import Image

from model import PINN
from plotting import plot_displacement_3d


class TestPlotDisplacement3D:
    def test_returns_existing_png_path(self, tmp_path):
        model = PINN(hidden_layers=1, hidden_units=8)
        path = plot_displacement_3d(model, epoch=1, save_dir=tmp_path, n_points=6)
        assert isinstance(path, Path)
        assert path.exists()
        assert path.suffix == ".png"

    def test_filename_zero_padded_epoch(self, tmp_path):
        model = PINN(hidden_layers=1, hidden_units=8)
        path = plot_displacement_3d(model, epoch=42, save_dir=tmp_path, n_points=6)
        assert path.name == "w_epoch_000042_3d.png"

    def test_creates_missing_save_dir(self, tmp_path):
        save_dir = tmp_path / "nested" / "plots"
        assert not save_dir.exists()
        model = PINN(hidden_layers=1, hidden_units=8)
        plot_displacement_3d(model, epoch=1, save_dir=save_dir, n_points=6)
        assert save_dir.is_dir()

    def test_output_is_valid_image(self, tmp_path):
        model = PINN(hidden_layers=1, hidden_units=8)
        path = plot_displacement_3d(model, epoch=1, save_dir=tmp_path, n_points=6)
        with Image.open(path) as img:
            img.verify()

    def test_leaves_model_in_eval_mode(self, tmp_path):
        model = PINN(hidden_layers=1, hidden_units=8)
        model.train()
        plot_displacement_3d(model, epoch=1, save_dir=tmp_path, n_points=6)
        assert model.training is False

    def test_custom_view_angles(self, tmp_path):
        model = PINN(hidden_layers=1, hidden_units=8)
        path = plot_displacement_3d(
            model, epoch=1, save_dir=tmp_path, n_points=6, elev=10, azim=45
        )
        assert path.exists()
