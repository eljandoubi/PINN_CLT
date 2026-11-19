"""Rayleigh-Ritz reference solution of the plate problem defined in `data.py`.

The plate is clamped at x = 0, simply supported at x = L, free at y = 0 and
y = W, and loaded by a uniform pressure q. With the orthotropic bending
stiffnesses D11, D22, D12 and D66 (D16 = D26 = 0) it has no closed-form
solution, so the PINN is checked against the minimiser of the potential energy

    U(w) = 1/2 * integral(D11 w_xx^2 + 2 D12 w_xx w_yy + D22 w_yy^2
                          + 4 D66 w_xy^2) dA - q * integral(w) dA

over the span of the products phi_i(x) * psi_j(y), with

    phi_i(x) = s^2 (1 - s) P_i(2 s - 1),  s = x / L,  i = 0 .. nx - 1
    psi_j(y) = P_j(2 t - 1),              t = y / W,  j = 0 .. ny - 1

where P_n is the Legendre polynomial of degree n. Every phi_i satisfies the
essential conditions w = w_x = 0 at x = 0 and w = 0 at x = L. The free edges and
the moment condition at x = L are natural conditions of the energy, so they are
met only in the limit of a large basis (`boundary_report` measures how well).
The energy is a separable sum, hence the stiffness matrix is a sum of Kronecker
products of one-dimensional Gauss-Legendre integrals.
"""

from dataclasses import dataclass

import numpy as np
from numpy.polynomial import legendre

from data import PLATE_LENGTH, PLATE_WIDTH, material_props

MAX_DERIVATIVE = 2


def _legendre_derivative(degree: int, s: np.ndarray, order: int) -> np.ndarray:
    """Return the `order`-th derivative of P_degree(2 s - 1) with respect to s."""
    coefficients = np.zeros(degree + 1)
    coefficients[degree] = 1.0
    if order:
        coefficients = legendre.legder(coefficients, order)
    return 2.0**order * legendre.legval(2.0 * s - 1.0, coefficients)


def clamped_basis(count: int, s: np.ndarray) -> np.ndarray:
    """Return phi_i and its first two derivatives; shape (3, count, len(s))."""
    g = s**2 * (1.0 - s)
    g1 = 2.0 * s - 3.0 * s**2
    g2 = 2.0 - 6.0 * s
    basis = np.empty((MAX_DERIVATIVE + 1, count, s.size))
    for i in range(count):
        p0 = _legendre_derivative(i, s, 0)
        p1 = _legendre_derivative(i, s, 1)
        p2 = _legendre_derivative(i, s, 2)
        basis[0, i] = g * p0
        basis[1, i] = g1 * p0 + g * p1
        basis[2, i] = g2 * p0 + 2.0 * g1 * p1 + g * p2
    return basis


def free_basis(count: int, s: np.ndarray) -> np.ndarray:
    """Return psi_j and its first two derivatives; shape (3, count, len(s))."""
    basis = np.empty((MAX_DERIVATIVE + 1, count, s.size))
    for j in range(count):
        for order in range(MAX_DERIVATIVE + 1):
            basis[order, j] = _legendre_derivative(j, s, order)
    return basis


@dataclass(frozen=True)
class RitzSolution:
    """Coefficients c_ij of w(x, y) = sum c_ij phi_i(x) psi_j(y), plus diagnostics."""

    coefficients: np.ndarray
    length: float
    width: float
    potential_energy: float
    condition_number: float

    def evaluate(self, x, y, dx: int = 0, dy: int = 0) -> np.ndarray:
        """Return d^(dx+dy) w / dx^dx dy^dy at the points (x, y) (broadcast)."""
        if not (0 <= dx <= MAX_DERIVATIVE and 0 <= dy <= MAX_DERIVATIVE):
            raise ValueError(f"derivative orders must be in 0..{MAX_DERIVATIVE}")
        shape = np.broadcast(np.asarray(x), np.asarray(y)).shape
        s = np.broadcast_to(np.asarray(x, dtype=float), shape).ravel() / self.length
        t = np.broadcast_to(np.asarray(y, dtype=float), shape).ravel() / self.width
        count_x, count_y = self.coefficients.shape
        phi = clamped_basis(count_x, s)[dx] / self.length**dx
        psi = free_basis(count_y, t)[dy] / self.width**dy
        values = np.einsum("ij,iq,jq->q", self.coefficients, phi, psi)
        return values.reshape(shape)

    @property
    def center_deflection(self) -> float:
        return float(self.evaluate(0.5 * self.length, 0.5 * self.width))

    def fields(self, x, y) -> dict[str, np.ndarray]:
        """Return w, w_x, w_xx and w_yy at the points (x, y)."""
        return {
            "w": self.evaluate(x, y),
            "w_x": self.evaluate(x, y, dx=1),
            "w_xx": self.evaluate(x, y, dx=2),
            "w_yy": self.evaluate(x, y, dy=2),
        }


def boundary_residuals(
    fields,
    w_scale: float,
    props: dict[str, float] | None = None,
    length: float = PLATE_LENGTH,
    width: float = PLATE_WIDTH,
    points: int = 201,
) -> dict[str, float]:
    """Measure how well a field meets the boundary conditions of the problem.

    `fields(x, y)` returns a dict with w, w_x, w_xx and w_yy at the points, so
    the same check applies to a `RitzSolution.fields` and to a network. Each
    value is the root mean square along the edge. The deflection and slope
    residuals are divided by `w_scale` (a typical |w|) and the moment residuals
    by D11 * w_scale / L^2. A root mean square is used because the clamped edge
    and the free edges meet at corners where the two conditions cannot both hold
    (the moment My at x = 0 is nonzero), so a maximum would only measure the
    corner. The free-edge shear condition needs third derivatives and is not
    measured.
    """
    props = material_props if props is None else props
    d11, d22, d12 = props["D11"], props["D22"], props["D12"]
    moment_scale = d11 * w_scale / length**2
    along_y = np.linspace(0.0, width, points)
    along_x = np.linspace(0.0, length, points)

    clamped = fields(np.zeros(points), along_y)
    supported = fields(np.full(points, length), along_y)
    free_0 = fields(along_x, np.zeros(points))
    free_w = fields(along_x, np.full(points, width))

    def rms(values: np.ndarray, scale: float) -> float:
        return float(np.sqrt(np.mean(np.square(values))) / scale)

    free_moments = np.concatenate(
        [d12 * edge["w_xx"] + d22 * edge["w_yy"] for edge in (free_0, free_w)]
    )
    return {
        "clamped_deflection": rms(clamped["w"], w_scale),
        "clamped_slope": rms(clamped["w_x"] * length, w_scale),
        "supported_deflection": rms(supported["w"], w_scale),
        "supported_moment": rms(
            d11 * supported["w_xx"] + d12 * supported["w_yy"], moment_scale
        ),
        "free_edge_moment": rms(free_moments, moment_scale),
    }


def ritz_solve(
    count_x: int = 12,
    count_y: int = 8,
    props: dict[str, float] | None = None,
    length: float = PLATE_LENGTH,
    width: float = PLATE_WIDTH,
) -> RitzSolution:
    """Solve K c = f for the energy above with a (count_x x count_y) basis."""
    props = material_props if props is None else props
    d11, d22, d12, d66 = props["D11"], props["D22"], props["D12"], props["D66"]
    pressure = props["pressure"]

    # Gauss-Legendre nodes on [0, 1]: exact for the polynomial integrands here.
    nodes, weights = legendre.leggauss(max(count_x, count_y) + 6)
    s = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    bx = clamped_basis(count_x, s)
    by = free_basis(count_y, s)

    def gram(basis: np.ndarray, scale: float, a: int, b: int) -> np.ndarray:
        # integral over [0, scale] of the a-th and b-th derivatives, in s units
        return scale ** (1 - a - b) * np.einsum(
            "iq,kq,q->ik", basis[a], basis[b], weights
        )

    stiffness = (
        d11 * np.kron(gram(bx, length, 2, 2), gram(by, width, 0, 0))
        + d12
        * (
            np.kron(gram(bx, length, 2, 0), gram(by, width, 0, 2))
            + np.kron(gram(bx, length, 0, 2), gram(by, width, 2, 0))
        )
        + d22 * np.kron(gram(bx, length, 0, 0), gram(by, width, 2, 2))
        + 4.0 * d66 * np.kron(gram(bx, length, 1, 1), gram(by, width, 1, 1))
    )
    load = pressure * np.kron(length * (bx[0] @ weights), width * (by[0] @ weights))

    # Jacobi scaling makes the solve insensitive to the units of the basis.
    scale = 1.0 / np.sqrt(np.diag(stiffness))
    scaled = scale[:, None] * stiffness * scale[None, :]
    solution = scale * np.linalg.solve(scaled, scale * load)

    return RitzSolution(
        coefficients=solution.reshape(count_x, count_y),
        length=length,
        width=width,
        potential_energy=float(-0.5 * load @ solution),
        condition_number=float(np.linalg.cond(scaled)),
    )
