"""Tests for data.py — material properties and boundary data shapes."""

import torch

from data import (
    N_BOUNDARY,
    PLATE_LENGTH,
    PLATE_WIDTH,
    THICKNESS,
    boundary_data,
    device,
    material_props,
)


class TestConstants:
    def test_plate_dimensions_positive(self):
        assert PLATE_LENGTH > 0
        assert PLATE_WIDTH > 0
        assert THICKNESS > 0

    def test_material_props_keys(self):
        for key in ("D11", "D22", "D12", "D66", "pressure"):
            assert key in material_props

    def test_stiffnesses_positive(self):
        assert material_props["D11"] > 0
        assert material_props["D22"] > 0
        assert material_props["D66"] > 0

    def test_pressure_positive(self):
        assert material_props["pressure"] > 0


class TestBoundaryData:
    def test_fixed_edge_shapes(self):
        fe = boundary_data["fixed_edge"]
        assert fe["x"].shape == (N_BOUNDARY, 1)
        assert fe["y"].shape == (N_BOUNDARY, 1)
        assert fe["w"].shape == (N_BOUNDARY, 1)

    def test_fixed_edge_x_is_zero(self):
        assert (boundary_data["fixed_edge"]["x"] == 0).all()

    def test_simply_supported_shapes(self):
        ss = boundary_data["simply_supported"]
        assert ss["x"].shape == (N_BOUNDARY, 1)
        assert ss["y"].shape == (N_BOUNDARY, 1)
        assert ss["xy"].shape == (N_BOUNDARY, 2)
        assert ss["w"].shape == (N_BOUNDARY, 1)

    def test_simply_supported_x_is_L(self):
        torch.testing.assert_close(
            boundary_data["simply_supported"]["x"],
            torch.full((N_BOUNDARY, 1), PLATE_LENGTH, device=device),
        )

    def test_free_edge_shapes(self):
        for key in ("free_edge_y0", "free_edge_yW"):
            assert boundary_data[key]["x"].shape == (N_BOUNDARY, 1)
            assert boundary_data[key]["y"].shape == (N_BOUNDARY, 1)

    def test_free_edge_y0_is_zero(self):
        assert (boundary_data["free_edge_y0"]["y"] == 0).all()

    def test_free_edge_yW_is_W(self):
        torch.testing.assert_close(
            boundary_data["free_edge_yW"]["y"],
            torch.full((N_BOUNDARY, 1), PLATE_WIDTH, device=device),
        )

    def test_all_tensors_on_correct_device(self):
        for group in boundary_data.values():
            for tensor in group.values():
                assert tensor.device == device
