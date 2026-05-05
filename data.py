import torch

# --- DEVICE CONFIGURATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. DEFINE THE PHYSICAL & GEOMETRIC PARAMETERS ---

# Plate dimensions (in meters)
PLATE_LENGTH = 1.0  # L
PLATE_WIDTH = 0.5  # W

# Orthotropic Material Properties for Carbon Fiber (T300/5208)
E1 = 181e9  # Longitudinal Young's Modulus (Pa)
E2 = 10.3e9  # Transverse Young's Modulus (Pa)
G12 = 7.17e9  # In-plane Shear Modulus (Pa)
NU12 = 0.28  # Major Poisson's ratio
NU21 = NU12 * E2 / E1  # Minor Poisson's ratio (from reciprocity)

THICKNESS = 0.005  # Total laminate thickness h (m)

# Reduced stiffness matrix components [Q] for a single orthotropic lamina
denom = 1 - NU12 * NU21
Q11 = E1 / denom
Q22 = E2 / denom
Q12 = NU12 * E2 / denom
Q66 = G12

# Bending stiffness matrix [D] (symmetric single-layer simplification)
# D_ij = Q_ij * h^3 / 12
D11 = Q11 * THICKNESS**3 / 12
D22 = Q22 * THICKNESS**3 / 12
D12 = Q12 * THICKNESS**3 / 12
D66 = Q66 * THICKNESS**3 / 12

print("\n--- Orthotropic Bending Stiffnesses ---")
print(f"D11 = {D11:.4f} N·m")
print(f"D22 = {D22:.4f} N·m")
print(f"D12 = {D12:.4f} N·m")
print(f"D66 = {D66:.4f} N·m")

# Load Conditions
PRESSURE = 10000.0  # Uniform transverse pressure (Pa)

# --- 2. GENERATE THE DATA POINTS AS TORCH TENSORS ---

N_BOUNDARY = 200  # Points on boundaries
N_PHYSICS = 10000  # Collocation points inside the domain

# a) Boundary Condition Data (fixed edge at x=0: w=0, dw/dx=0)
y_boundary = torch.linspace(0, PLATE_WIDTH, N_BOUNDARY, device=device).unsqueeze(1)
x_boundary = torch.zeros_like(y_boundary)
w_boundary = torch.zeros_like(y_boundary)

# b) Free edges boundary data (y=0 and y=W: moment My=0, shear Vy=0)
x_free = torch.linspace(0, PLATE_LENGTH, N_BOUNDARY, device=device).unsqueeze(1)
y_free_bottom = torch.zeros_like(x_free)
y_free_top = torch.full_like(x_free, PLATE_WIDTH)

# c) Simply supported edge at x=L (w=0, Mx=0)
y_ss = torch.linspace(0, PLATE_WIDTH, N_BOUNDARY, device=device).unsqueeze(1)
x_ss = torch.full_like(y_ss, PLATE_LENGTH)
w_ss = torch.zeros_like(y_ss)

# d) Collocation Points for Physics Loss (interior domain)
x_collocation = (torch.rand(N_PHYSICS, 1, device=device) * PLATE_LENGTH).requires_grad_(
    True
)
y_collocation = (torch.rand(N_PHYSICS, 1, device=device) * PLATE_WIDTH).requires_grad_(
    True
)

# --- 3. PACKAGE DATA INTO DICTIONARIES ---

boundary_data = {
    "fixed_edge": {
        "xy": torch.cat([x_boundary, y_boundary], dim=1),
        "w": w_boundary,
    },
    "free_edge_bottom": {
        "xy": torch.cat([x_free, y_free_bottom], dim=1),
    },
    "free_edge_top": {
        "xy": torch.cat([x_free, y_free_top], dim=1),
    },
    "simply_supported": {
        "xy": torch.cat([x_ss, y_ss], dim=1),
        "w": w_ss,
    },
}

collocation_data = {
    "xy": torch.cat([x_collocation, y_collocation], dim=1),
}

material_props = {
    "D11": D11,
    "D22": D22,
    "D12": D12,
    "D66": D66,
    "pressure": PRESSURE,
}

# --- 4. SUMMARY ---

print(f"\n--- Data Summary (device: {device}) ---")
print(f"Fixed edge points: {boundary_data['fixed_edge']['xy'].shape}")
print(f"Free edge (bottom) points: {boundary_data['free_edge_bottom']['xy'].shape}")
print(f"Free edge (top) points: {boundary_data['free_edge_top']['xy'].shape}")
print(f"Simply supported points: {boundary_data['simply_supported']['xy'].shape}")
print(f"Collocation points: {collocation_data['xy'].shape}")
print("\nGoverning PDE (orthotropic plate):")
print("  D11·∂⁴w/∂x⁴ + 2(D12+2D66)·∂⁴w/∂x²∂y² + D22·∂⁴w/∂y⁴ = q")
