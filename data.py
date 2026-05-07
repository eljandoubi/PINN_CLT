import torch

# --- DEVICE CONFIGURATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Load Conditions
PRESSURE = 10000.0  # Uniform transverse pressure (Pa)

# --- 2. GENERATE THE DATA POINTS AS TORCH TENSORS ---

N_BOUNDARY = 256  # Points on boundaries

# a) Boundary Condition Data (fixed edge at x=0: w=0, dw/dx=0)
y_boundary = torch.linspace(0, PLATE_WIDTH, N_BOUNDARY, device=device).unsqueeze(1)
x_boundary = torch.zeros_like(y_boundary)
w_boundary = torch.zeros_like(y_boundary)

# b) Simply supported edge at x=L (w=0, Mx=0)
y_ss = torch.linspace(0, PLATE_WIDTH, N_BOUNDARY, device=device).unsqueeze(1)
x_ss = torch.full_like(y_ss, PLATE_LENGTH)
w_ss = torch.zeros_like(y_ss)

# c) Free edges at y=0 and y=W (My=0, Vy=0)
x_free_y0 = torch.linspace(0, PLATE_LENGTH, N_BOUNDARY, device=device).unsqueeze(1)
y_free_y0 = torch.zeros_like(x_free_y0)

x_free_yW = torch.linspace(0, PLATE_LENGTH, N_BOUNDARY, device=device).unsqueeze(1)
y_free_yW = torch.full_like(x_free_yW, PLATE_WIDTH)

# --- 3. PACKAGE DATA INTO DICTIONARIES ---

boundary_data = {
    "fixed_edge": {
        "x": x_boundary,
        "y": y_boundary,
        "w": w_boundary,
    },
    "simply_supported": {
        "x": x_ss,
        "y": y_ss,
        "xy": torch.cat([x_ss, y_ss], dim=1),
        "w": w_ss,
    },
    "free_edge_y0": {
        "x": x_free_y0,
        "y": y_free_y0,
    },
    "free_edge_yW": {
        "x": x_free_yW,
        "y": y_free_yW,
    },
}

material_props = {
    "D11": D11,
    "D22": D22,
    "D12": D12,
    "D66": D66,
    "pressure": PRESSURE,
}
