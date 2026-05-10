"""Tests for losses.py — compute_losses and clip_grads."""

import torch
import torch.nn as nn

from data import boundary_data, material_props
from losses import clip_grads, compute_losses
from model import PINN, AdaptiveLossWeights


class TestComputeLosses:
    def test_returns_five_elements(self):
        model = PINN(hidden_layers=2, hidden_units=32)
        xy = torch.rand(64, 2)
        result = compute_losses(
            model,
            xy,
            material_props,
            boundary_data,
            nn.MSELoss(),
            1.0,
            1.0,
            1.0,
        )
        assert len(result) == 5

    def test_output_shapes(self):
        model = PINN(hidden_layers=2, hidden_units=32)
        xy = torch.rand(32, 2)
        total, phys, bc, nat, weights = compute_losses(
            model,
            xy,
            material_props,
            boundary_data,
            nn.MSELoss(),
            1.0,
            1.0,
            1.0,
        )
        assert total.shape == ()
        assert phys.shape == ()
        assert bc.shape == ()
        assert nat.shape == ()
        assert len(weights) == 3

    def test_losses_non_negative(self):
        model = PINN(hidden_layers=2, hidden_units=32)
        xy = torch.rand(32, 2)
        total, phys, bc, nat, _ = compute_losses(
            model,
            xy,
            material_props,
            boundary_data,
            nn.MSELoss(),
            1.0,
            1.0,
            1.0,
        )
        assert total.item() >= 0
        assert phys.item() >= 0
        assert bc.item() >= 0
        assert nat.item() >= 0

    def test_differentiable(self):
        model = PINN(hidden_layers=2, hidden_units=32)
        xy = torch.rand(32, 2)
        total, _, _, _, _ = compute_losses(
            model,
            xy,
            material_props,
            boundary_data,
            nn.MSELoss(),
            1.0,
            1.0,
            1.0,
        )
        total.backward()
        grads = [p.grad is not None for p in model.parameters()]
        assert sum(grads) >= len(grads) - 1

    def test_manual_weights_returned(self):
        model = PINN(hidden_layers=2, hidden_units=32)
        xy = torch.rand(16, 2)
        _, _, _, _, weights = compute_losses(
            model,
            xy,
            material_props,
            boundary_data,
            nn.MSELoss(),
            2.0,
            3.0,
            4.0,
        )
        assert weights == [2.0, 3.0, 4.0]

    def test_with_adaptive_weighter(self):
        model = PINN(hidden_layers=2, hidden_units=32)
        aw = AdaptiveLossWeights(num_losses=3, initial_weights=[1.0, 1.0, 1.0])
        xy = torch.rand(16, 2)
        total, phys, bc, nat, weights = compute_losses(
            model,
            xy,
            material_props,
            boundary_data,
            nn.MSELoss(),
            1.0,
            1.0,
            1.0,
            adaptive_weighter=aw,
        )
        assert total.shape == ()
        assert len(weights) == 3
        # Adaptive weights should be close to 1.0 initially
        for w in weights:
            assert abs(w - 1.0) < 0.1

    def test_with_normalize(self):
        model = PINN(hidden_layers=2, hidden_units=32)
        xy = torch.rand(16, 2)
        total_raw, _, _, _, _ = compute_losses(
            model,
            xy,
            material_props,
            boundary_data,
            nn.MSELoss(),
            1.0,
            1.0,
            1.0,
            normalize=False,
        )
        total_norm, _, _, _, _ = compute_losses(
            model,
            xy,
            material_props,
            boundary_data,
            nn.MSELoss(),
            1.0,
            1.0,
            1.0,
            normalize=True,
        )
        # Both should be valid scalars (exact comparison not meaningful)
        assert total_raw.shape == ()
        assert total_norm.shape == ()

    def test_zero_lambda_zeroes_component(self):
        model = PINN(hidden_layers=2, hidden_units=32)
        xy = torch.rand(16, 2)
        total, phys, _, _, _ = compute_losses(
            model,
            xy,
            material_props,
            boundary_data,
            nn.MSELoss(),
            0.0,
            0.0,
            0.0,
        )
        # With all lambdas=0, total should be 0
        assert total.item() == 0.0

    def test_different_criteria(self):
        model = PINN(hidden_layers=2, hidden_units=32)
        xy = torch.rand(16, 2)
        for crit in [nn.MSELoss(), nn.L1Loss(), nn.HuberLoss()]:
            total, _, _, _, _ = compute_losses(
                model,
                xy,
                material_props,
                boundary_data,
                crit,
                1.0,
                1.0,
                1.0,
            )
            assert total.shape == ()


class TestClipGrads:
    def test_clips_model_gradients(self):
        model = PINN(hidden_layers=2, hidden_units=32)
        xy = torch.rand(16, 2)
        loss = model(xy).pow(2).mean()
        loss.backward()

        clip_grads(model, max_grad_norm=0.01)

        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        assert total_norm.item() <= 0.01 + 1e-6

    def test_clips_adaptive_weighter_gradients(self):
        model = PINN(hidden_layers=2, hidden_units=32)
        aw = AdaptiveLossWeights(num_losses=2)

        # Create a loss involving both model and aw
        xy = torch.rand(8, 2)
        pred = model(xy).pow(2).mean()
        total, _ = aw(pred, pred)
        total.backward()

        clip_grads(model, max_grad_norm=0.01, adaptive_weighter=aw)

        model_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        aw_norm = torch.nn.utils.clip_grad_norm_(aw.parameters(), float("inf"))
        assert model_norm.item() <= 0.01 + 1e-6
        assert aw_norm.item() <= 0.01 + 1e-6

    def test_no_error_without_adaptive(self):
        model = PINN(hidden_layers=2, hidden_units=32)
        xy = torch.rand(8, 2)
        loss = model(xy).pow(2).mean()
        loss.backward()
        # Should not raise
        clip_grads(model, max_grad_norm=1.0, adaptive_weighter=None)
