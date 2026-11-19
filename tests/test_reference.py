"""Tests of the Ritz reference solver (reference.py)."""

from dataclasses import replace

import numpy as np
import pytest

from data import PLATE_LENGTH, PLATE_WIDTH, material_props
from reference import (
    boundary_residuals,
    clamped_basis,
    free_basis,
    ritz_solve,
)

L, W = PLATE_LENGTH, PLATE_WIDTH
D11, D22, D12, D66 = (material_props[k] for k in ("D11", "D22", "D12", "D66"))
Q = material_props["pressure"]


def _grid_integral(solution, integrand, points=30):
    """Integrate `integrand(solution, x, y)` over the plate by Gauss-Legendre."""
    nodes, weights = np.polynomial.legendre.leggauss(points)
    s, weights = 0.5 * (nodes + 1.0), 0.5 * weights
    x, y = np.meshgrid(solution.length * s, solution.width * s, indexing="ij")
    area = np.outer(weights, weights) * solution.length * solution.width
    return float(np.sum(area * integrand(solution, x, y)))


def _energy_density(solution, x, y):
    w_xx = solution.evaluate(x, y, dx=2)
    w_yy = solution.evaluate(x, y, dy=2)
    w_xy = solution.evaluate(x, y, dx=1, dy=1)
    bending = D11 * w_xx**2 + 2 * D12 * w_xx * w_yy + D22 * w_yy**2 + 4 * D66 * w_xy**2
    return 0.5 * bending - Q * solution.evaluate(x, y)


class TestBasis:
    def test_clamped_basis_meets_the_essential_conditions(self):
        basis = clamped_basis(6, np.array([0.0, 1.0]))
        assert np.all(basis[0, :, 0] == 0.0)  # w = 0 at x = 0
        assert np.all(basis[1, :, 0] == 0.0)  # w_x = 0 at x = 0
        assert np.all(basis[0, :, 1] == 0.0)  # w = 0 at x = L

    def test_free_basis_is_the_shifted_legendre_family(self):
        values = free_basis(5, np.array([0.0, 1.0]))
        degrees = np.arange(5)
        assert np.allclose(values[0, :, 1], 1.0)  # P_n(1) = 1
        assert np.allclose(values[0, :, 0], (-1.0) ** degrees)  # P_n(-1)
        assert np.allclose(values[1, :, 1], degrees * (degrees + 1))  # 2 P_n'(1)


class TestSolution:
    def test_beam_limit_matches_the_closed_form(self):
        # Without D12 the free edges carry no moment, so w depends on x only
        # and solves the propped cantilever D11 w'''' = q.
        props = {**material_props, "D12": 0.0}
        length, width = 1.3, 0.8  # not 1, so a missing length factor shows
        solution = ritz_solve(4, 2, props, length=length, width=width)
        x = np.linspace(0.0, length, 11)
        y = np.linspace(0.0, width, 11)
        beam = Q * x**2 * (3 * length**2 - 5 * length * x + 2 * x**2) / (48 * D11)
        assert np.allclose(solution.evaluate(x, y), beam, rtol=0, atol=1e-12)

    def test_essential_boundary_conditions_hold(self):
        solution = ritz_solve(8, 6)
        y = np.linspace(0.0, W, 21)
        assert np.allclose(solution.evaluate(0.0, y), 0.0, atol=1e-14)
        assert np.allclose(solution.evaluate(0.0, y, dx=1), 0.0, atol=1e-14)
        assert np.allclose(solution.evaluate(L, y), 0.0, atol=1e-14)

    def test_solution_is_symmetric_about_the_plate_axis(self):
        solution = ritz_solve(8, 6)
        x = np.linspace(0.0, L, 9)
        y = np.linspace(0.0, 0.5 * W, 9)
        assert np.allclose(solution.evaluate(x, y), solution.evaluate(x, W - y))

    def test_potential_energy_matches_direct_quadrature(self):
        solution = ritz_solve(6, 4)
        direct = _grid_integral(solution, _energy_density)
        assert solution.potential_energy == pytest.approx(direct, rel=1e-9)

    def test_energy_is_minus_half_the_work_of_the_load(self):
        solution = ritz_solve(6, 4)
        work = _grid_integral(solution, lambda s, x, y: Q * s.evaluate(x, y))
        assert solution.potential_energy == pytest.approx(-0.5 * work, rel=1e-9)

    def test_solution_minimises_the_energy(self):
        solution = ritz_solve(6, 4)
        rng = np.random.default_rng(0)
        base = _grid_integral(solution, _energy_density)

        def increase(direction, step):
            moved = replace(
                solution, coefficients=solution.coefficients + step * direction
            )
            return _grid_integral(moved, _energy_density) - base

        for _ in range(3):
            direction = rng.standard_normal(solution.coefficients.shape)
            direction *= np.abs(solution.coefficients).max()
            small, large = increase(direction, 1e-2), increase(direction, 2e-2)
            assert small > 0.0
            assert large / small == pytest.approx(4.0, rel=1e-4)  # quadratic, no slope

    def test_solution_satisfies_the_governing_equation_inside(self):
        solution = ritz_solve(16, 12)
        rng = np.random.default_rng(0)
        x = rng.uniform(0.3 * L, 0.7 * L, 40)
        y = rng.uniform(0.2 * W, 0.8 * W, 40)
        h = 2e-3

        def second_difference(dx, dy, along_x):
            ahead = solution.evaluate(x + h * along_x, y + h * (1 - along_x), dx, dy)
            here = solution.evaluate(x, y, dx, dy)
            behind = solution.evaluate(x - h * along_x, y - h * (1 - along_x), dx, dy)
            return (ahead - 2 * here + behind) / h**2

        residual = (
            D11 * second_difference(2, 0, 1)
            + 2 * (D12 + 2 * D66) * second_difference(2, 0, 0)
            + D22 * second_difference(0, 2, 0)
            - Q
        )
        # D11 w_xxxx alone is about q: a wrong D12 or D66 raises this to 6e-3 or more.
        assert np.sqrt(np.mean((residual / Q) ** 2)) < 3.5e-3

    @pytest.mark.parametrize("length, width", [(L, W), (1.3, 0.8)])
    def test_derivatives_match_finite_differences(self, length, width):
        solution = ritz_solve(5, 4, length=length, width=width)
        rng = np.random.default_rng(1)
        x = rng.uniform(0.1 * length, 0.9 * length, 10)
        y = rng.uniform(0.1 * width, 0.9 * width, 10)
        h = 1e-4

        def value(dx=0, dy=0, shift_x=0.0, shift_y=0.0):
            return solution.evaluate(x + shift_x, y + shift_y, dx, dy)

        centred_x = (value(shift_x=h) - value(shift_x=-h)) / (2 * h)
        centred_y = (value(shift_y=h) - value(shift_y=-h)) / (2 * h)
        assert np.allclose(value(dx=1), centred_x, atol=1e-7)
        assert np.allclose(value(dy=1), centred_y, atol=1e-7)
        second_x = (value(shift_x=h) - 2 * value() + value(shift_x=-h)) / h**2
        second_y = (value(shift_y=h) - 2 * value() + value(shift_y=-h)) / h**2
        assert np.allclose(value(dx=2), second_x, atol=1e-5)
        assert np.allclose(value(dy=2), second_y, atol=1e-5)
        mixed = (value(dx=1, shift_y=h) - value(dx=1, shift_y=-h)) / (2 * h)
        assert np.allclose(value(dx=1, dy=1), mixed, atol=1e-6)

    def test_center_deflection_converges_with_the_basis(self):
        coarse = ritz_solve(8, 6).center_deflection
        fine = ritz_solve(12, 8).center_deflection
        assert fine == pytest.approx(coarse, rel=1e-4)
        assert fine == pytest.approx(2.7205e-2, rel=1e-3)

    def test_evaluate_broadcasts_and_rejects_high_orders(self):
        solution = ritz_solve(4, 3)
        assert solution.evaluate(
            np.ones((5, 1)) * 0.3, np.ones((1, 4)) * 0.2
        ).shape == (5, 4)
        assert solution.evaluate(0.3, 0.2).shape == ()
        with pytest.raises(ValueError, match="derivative orders"):
            solution.evaluate(0.3, 0.2, dx=3)
        with pytest.raises(ValueError, match="derivative orders"):
            solution.evaluate(0.3, 0.2, dy=-1)

    def test_condition_number_is_reported(self):
        assert 1.0 <= ritz_solve(6, 4).condition_number < 1e6

    def test_fields_returns_the_named_derivatives(self):
        solution = ritz_solve(5, 4)
        x, y = np.array([0.3, 0.6]), np.array([0.1, 0.4])
        fields = solution.fields(x, y)
        assert np.array_equal(fields["w"], solution.evaluate(x, y))
        assert np.array_equal(fields["w_x"], solution.evaluate(x, y, dx=1))
        assert np.array_equal(fields["w_xx"], solution.evaluate(x, y, dx=2))
        assert np.array_equal(fields["w_yy"], solution.evaluate(x, y, dy=2))
        assert not np.allclose(fields["w_x"], solution.evaluate(x, y, dy=1))


class TestBoundaryResiduals:
    @staticmethod
    def _residuals(count_x, count_y):
        solution = ritz_solve(count_x, count_y)
        x, y = np.meshgrid(np.linspace(0, L, 101), np.linspace(0, W, 51), indexing="ij")
        scale = float(np.abs(solution.evaluate(x, y)).max())
        return boundary_residuals(solution.fields, scale)

    def test_essential_conditions_are_exact(self):
        residuals = self._residuals(8, 6)
        for name in ("clamped_deflection", "clamped_slope", "supported_deflection"):
            assert residuals[name] < 1e-12

    def test_natural_conditions_are_met_as_the_basis_grows(self):
        coarse = self._residuals(4, 3)
        fine = self._residuals(16, 12)
        assert fine["free_edge_moment"] < 0.5 * coarse["free_edge_moment"]
        # a solver that ignores the D12 coupling leaves 0.17 here, this one 0.04
        assert fine["free_edge_moment"] < 0.05
        assert fine["supported_moment"] < 1e-2

    def test_a_wrong_field_is_flagged(self):
        length, width = 2.0, 0.7

        def uniform(x, y):
            ones = np.ones_like(x)
            return {"w": ones, "w_x": ones, "w_xx": 2 * ones, "w_yy": 3 * ones}

        residuals = boundary_residuals(uniform, 1.0, length=length, width=width)
        moment_scale = D11 / length**2
        assert residuals["clamped_deflection"] == pytest.approx(1.0)
        assert residuals["clamped_slope"] == pytest.approx(length)
        assert residuals["supported_deflection"] == pytest.approx(1.0)
        assert residuals["supported_moment"] == pytest.approx(
            (2 * D11 + 3 * D12) / moment_scale
        )
        assert residuals["free_edge_moment"] == pytest.approx(
            (2 * D12 + 3 * D22) / moment_scale
        )
