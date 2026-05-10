"""Tests for model.py — architecture, losses, and PDE residual."""

import pytest
import torch
import torch.nn as nn

from data import boundary_data, material_props
from losses import (
    compute_boundary_loss,
    compute_natural_bc_loss,
    compute_pde_residual,
    zero_loss,
)
from model import (
    FFMLP,
    PINN,
    AdaptiveLossWeights,
    ResidualBlock,
    ReverseHuberLoss,
)


# ---------------------------------------------------------------------------
# ReverseHuberLoss
# ---------------------------------------------------------------------------
class TestReverseHuberLoss:
    def test_output_is_scalar(self):
        loss = ReverseHuberLoss(delta=1.0)
        pred = torch.randn(16, 1)
        target = torch.randn(16, 1)
        assert loss(pred, target).shape == ()

    def test_small_errors_use_l1(self):
        loss_fn = ReverseHuberLoss(delta=10.0, reduction="none")
        pred = torch.tensor([0.1])
        target = torch.tensor([0.0])
        # |error| = 0.1 < delta=10 → should equal L1
        result = loss_fn(pred, target)
        expected = torch.tensor([0.1])
        torch.testing.assert_close(result, expected)

    def test_large_errors_use_mse(self):
        loss_fn = ReverseHuberLoss(delta=0.01, reduction="none")
        pred = torch.tensor([2.0])
        target = torch.tensor([0.0])
        # |error| = 2.0 > delta=0.01 → should equal MSE = 4.0
        result = loss_fn(pred, target)
        expected = torch.tensor([4.0])
        torch.testing.assert_close(result, expected)

    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    def test_reductions(self, reduction):
        loss_fn = ReverseHuberLoss(delta=1.0, reduction=reduction)
        out = loss_fn(torch.randn(8, 1), torch.randn(8, 1))
        if reduction == "none":
            assert out.shape == (8, 1)
        else:
            assert out.shape == ()


# ---------------------------------------------------------------------------
# AdaptiveLossWeights
# ---------------------------------------------------------------------------
class TestAdaptiveLossWeights:
    def test_initial_effective_weights(self):
        aw = AdaptiveLossWeights(num_losses=3, initial_weights=[1.0, 2.0, 3.0])
        l1, l2, l3 = torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.0)
        _, weights = aw(l1, l2, l3)
        torch.testing.assert_close(
            torch.tensor(weights), torch.tensor([1.0, 2.0, 3.0]), atol=1e-5, rtol=1e-5
        )

    def test_output_is_differentiable(self):
        aw = AdaptiveLossWeights(num_losses=2)
        total, _ = aw(torch.tensor(1.0), torch.tensor(2.0))
        total.backward()
        assert aw.log_vars.grad is not None

    def test_reset(self):
        aw = AdaptiveLossWeights(num_losses=2, initial_weights=[1.0, 1.0])
        # Manually perturb
        with torch.no_grad():
            aw.log_vars.fill_(99.0)
        aw.reset()
        torch.testing.assert_close(aw.log_vars.data, aw._initial_log_vars)


# ---------------------------------------------------------------------------
# FFMLP
# ---------------------------------------------------------------------------
class TestFFMLP:
    def test_output_shape(self):
        m = FFMLP(in_features=8, hidden_features=16, out_features=4)
        x = torch.randn(32, 8)
        assert m(x).shape == (32, 4)

    def test_defaults_preserve_dim(self):
        m = FFMLP(in_features=8)
        assert m(torch.randn(4, 8)).shape == (4, 8)


# ---------------------------------------------------------------------------
# ResidualBlock
# ---------------------------------------------------------------------------
class TestResidualBlock:
    def test_same_dim(self):
        rb = ResidualBlock(16, 16)
        assert rb(torch.randn(4, 16)).shape == (4, 16)

    def test_dim_change_uses_shortcut(self):
        rb = ResidualBlock(8, 16)
        assert rb.shortcut is not None
        assert rb(torch.randn(4, 8)).shape == (4, 16)

    def test_with_norm(self):
        rb = ResidualBlock(16, 16, use_norm=True)
        assert hasattr(rb, "norm1")
        assert rb(torch.randn(4, 16)).shape == (4, 16)

    def test_with_ffmlp(self):
        rb = ResidualBlock(16, 16, use_ffmlp=True)
        assert isinstance(rb.fc1, FFMLP)
        assert rb(torch.randn(4, 16)).shape == (4, 16)


# ---------------------------------------------------------------------------
# PINN
# ---------------------------------------------------------------------------
class TestPINN:
    def test_basic_forward(self):
        model = PINN(hidden_layers=2, hidden_units=32)
        xy = torch.randn(16, 2)
        assert model(xy).shape == (16, 1)

    def test_with_residual(self):
        model = PINN(hidden_layers=2, hidden_units=32, use_residual=True)
        assert model(torch.randn(8, 2)).shape == (8, 1)

    def test_with_residual_and_norm(self):
        model = PINN(hidden_layers=2, hidden_units=32, use_residual=True, use_norm=True)
        assert model(torch.randn(8, 2)).shape == (8, 1)

    def test_with_ffmlp(self):
        model = PINN(hidden_layers=2, hidden_units=32, use_ffmlp=True)
        assert model(torch.randn(8, 2)).shape == (8, 1)

    @pytest.mark.parametrize("act", [nn.Tanh, nn.SiLU, nn.GELU, nn.Softplus, nn.Mish])
    def test_activations(self, act):
        model = PINN(hidden_layers=2, hidden_units=32, activation=act)
        assert model(torch.randn(4, 2)).shape == (4, 1)

    def test_xavier_init(self):
        model = PINN(hidden_layers=2, hidden_units=32)
        for m in model.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                assert (m.bias == 0).all()


# ---------------------------------------------------------------------------
# zero_loss
# ---------------------------------------------------------------------------
class TestZeroLoss:
    def test_mse(self):
        x = torch.tensor([2.0, 3.0])
        result = zero_loss(nn.MSELoss(), x)
        expected = x.pow(2).mean()
        torch.testing.assert_close(result, expected)

    def test_l1(self):
        x = torch.tensor([-2.0, 3.0])
        result = zero_loss(nn.L1Loss(), x)
        expected = x.abs().mean()
        torch.testing.assert_close(result, expected)

    def test_huber(self):
        x = torch.tensor([0.5, 1.5])
        result = zero_loss(nn.HuberLoss(), x)
        expected = nn.HuberLoss()(x, torch.zeros_like(x))
        torch.testing.assert_close(result, expected)

    def test_reverse_huber(self):
        x = torch.tensor([0.5, 1.5])
        result = zero_loss(ReverseHuberLoss(), x)
        expected = ReverseHuberLoss()(x, torch.zeros_like(x))
        torch.testing.assert_close(result, expected)


# ---------------------------------------------------------------------------
# PDE residual
# ---------------------------------------------------------------------------
class TestPDEResidual:
    def test_output_shape(self):
        model = PINN(hidden_layers=2, hidden_units=32)
        xy = torch.rand(64, 2)
        res = compute_pde_residual(model, xy, material_props)
        assert res.shape == (64, 1)

    def test_residual_is_differentiable(self):
        model = PINN(hidden_layers=2, hidden_units=32)
        xy = torch.rand(16, 2)
        res = compute_pde_residual(model, xy, material_props)
        loss = res.pow(2).mean()
        loss.backward()
        # The output layer bias has zero gradient through 4th-order derivatives,
        # so we only check that most parameters received gradients.
        grads = [p.grad is not None for p in model.parameters()]
        assert sum(grads) >= len(grads) - 1

    def test_normalize_flag(self):
        model = PINN(hidden_layers=2, hidden_units=32)
        xy = torch.rand(16, 2)
        res_raw = compute_pde_residual(model, xy, material_props, normalize=False)
        res_norm = compute_pde_residual(model, xy, material_props, normalize=True)
        # Normalized residual should have smaller magnitude
        assert res_norm.abs().mean() < res_raw.abs().mean() + 1e-6


# ---------------------------------------------------------------------------
# Boundary & Natural BC losses
# ---------------------------------------------------------------------------
class TestBoundaryLoss:
    def test_scalar_output(self):
        model = PINN(hidden_layers=2, hidden_units=32)
        loss = compute_boundary_loss(model, boundary_data)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_differentiable(self):
        model = PINN(hidden_layers=2, hidden_units=32)
        loss = compute_boundary_loss(model, boundary_data)
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None


class TestNaturalBCLoss:
    def test_scalar_output(self):
        model = PINN(hidden_layers=2, hidden_units=32)
        loss = compute_natural_bc_loss(model, boundary_data, material_props)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_differentiable(self):
        model = PINN(hidden_layers=2, hidden_units=32)
        loss = compute_natural_bc_loss(model, boundary_data, material_props)
        loss.backward()
        grads = [p.grad is not None for p in model.parameters()]
        assert sum(grads) >= len(grads) - 1
