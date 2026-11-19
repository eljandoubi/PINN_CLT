"""Tests of the PINN versus Ritz comparison script (compare.py)."""

import json
import typing

import numpy as np
import pytest
import torch

from checkpoint import save_checkpoint
from compare import (
    build_parser,
    compare,
    load_model,
    main,
    network_fields,
    plot_comparison,
)
from data import PLATE_LENGTH, material_props
from model import ACTIVATIONS, PINN
from reference import ritz_solve
from train import TrainingConfig


class Beam(torch.nn.Module):
    """Closed-form propped cantilever: the Ritz solution when D12 = 0."""

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        x = xy[:, :1]
        length = PLATE_LENGTH
        beam = (
            material_props["pressure"]
            * x**2
            * (3 * length**2 - 5 * length * x + 2 * x**2)
            / (48 * material_props["D11"])
        )
        return self.scale * beam


@pytest.fixture
def beam_reference():
    return ritz_solve(4, 2, {**material_props, "D12": 0.0})


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(0)
    model = PINN(hidden_layers=2, hidden_units=8)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    path = tmp_path / "best.pt"
    save_checkpoint(path, model, optimizer, scheduler, epoch=1, best_loss=0.5)
    return path


def test_network_fields_are_the_autograd_derivatives():
    class Polynomial(torch.nn.Module):
        def forward(self, xy):
            return (xy[:, :1] ** 3) * xy[:, 1:]  # w = x^3 y

    x, y = np.array([0.3, 0.7]), np.array([0.2, 0.4])
    fields = network_fields(Polynomial())(x, y)
    assert np.allclose(fields["w"], x**3 * y, atol=1e-6)
    assert np.allclose(fields["w_x"], 3 * x**2 * y, atol=1e-6)
    assert np.allclose(fields["w_xx"], 6 * x * y, atol=1e-5)
    assert np.allclose(fields["w_yy"], 0.0, atol=1e-6)


def test_exact_field_has_no_error(beam_reference):
    result = compare(Beam(), beam_reference, points_x=21, points_y=11)
    assert result.metrics["relative_l2_error"] < 1e-5
    assert result.metrics["max_error_over_max_deflection"] < 1e-5
    assert result.metrics["network_center_deflection"] == pytest.approx(
        result.metrics["reference_center_deflection"], rel=1e-5
    )
    assert result.metrics["network_min_deflection"] >= -1e-9
    assert result.network.shape == (21, 11)


def test_half_the_field_has_half_the_error(beam_reference):
    result = compare(Beam(scale=0.5), beam_reference, points_x=21, points_y=11)
    assert result.metrics["relative_l2_error"] == pytest.approx(0.5, rel=1e-4)
    assert result.metrics["max_error_over_max_deflection"] == pytest.approx(
        0.5, rel=1e-4
    )


def test_boundary_metrics_cover_both_fields(beam_reference):
    result = compare(Beam(), beam_reference, points_x=11, points_y=6)
    expected = {
        "clamped_deflection",
        "clamped_slope",
        "supported_deflection",
        "supported_moment",
        "free_edge_moment",
    }
    assert set(result.metrics["network_boundary_rms"]) == expected
    assert set(result.metrics["reference_boundary_rms"]) == expected
    assert result.metrics["network_boundary_rms"]["clamped_deflection"] < 1e-6


def test_load_model_round_trips_the_checkpoint(checkpoint):
    model = load_model(checkpoint, hidden_layers=2, hidden_units=8)
    torch.manual_seed(0)
    expected = PINN(hidden_layers=2, hidden_units=8)
    xy = torch.rand(5, 2)
    assert torch.allclose(model(xy), expected(xy))
    assert not model.training


def test_main_writes_the_report_and_the_figure(checkpoint, tmp_path):
    output = tmp_path / "out"
    options = {
        "--checkpoint": checkpoint,
        "--output": output,
        "--hidden_layers": 2,
        "--hidden_units": 8,
        "--basis_x": 6,
        "--basis_y": 4,
        "--points_x": 21,
        "--points_y": 11,
    }
    report = main([part for pair in options.items() for part in map(str, pair)])
    saved = json.loads((output / "comparison.json").read_text())
    assert saved == report
    assert saved["checkpoint"] == checkpoint.name
    assert saved["ritz_basis"] == [6, 4]
    assert saved["relative_l2_error"] > 0.0
    assert (output / "comparison.png").stat().st_size > 0


def test_plot_comparison_handles_identical_fields(beam_reference, tmp_path):
    result = compare(Beam(), beam_reference, points_x=11, points_y=6)
    path = plot_comparison(result, tmp_path / "nested" / "figure.png")
    assert path.exists()


def test_architecture_options_match_the_training_options():
    training = typing.get_args(TrainingConfig.__dataclass_fields__["activation"].type)
    assert set(training) == set(ACTIVATIONS)
    parser = build_parser()
    assert parser.parse_args(["--checkpoint", "x", "--use_norm", "true"]).use_norm
    assert not parser.parse_args(["--checkpoint", "x", "--use_norm", "no"]).use_norm
