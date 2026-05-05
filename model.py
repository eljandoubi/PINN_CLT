import torch
import torch.nn as nn


class PINN(nn.Module):
    """Physics-Informed Neural Network for orthotropic plate bending (CLT)."""

    def __init__(self, hidden_layers=4, hidden_units=64, activation=nn.Tanh):
        super().__init__()
        layers = []
        # Input: (x, y) -> 2 features
        in_features = 2
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_features, hidden_units))
            layers.append(activation())
            in_features = hidden_units
        # Output: w (transverse displacement) -> 1 feature
        layers.append(nn.Linear(in_features, 1))
        self.net = nn.Sequential(*layers)

        # Xavier initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xy):
        """Forward pass: predict displacement w given (x, y) coordinates."""
        return self.net(xy)


def compute_pde_residual(model, xy, material_props, normalize=False):
    """
    Compute the residual of the orthotropic plate PDE:
        D11·∂⁴w/∂x⁴ + 2(D12+2D66)·∂⁴w/∂x²∂y² + D22·∂⁴w/∂y⁴ - q = 0

    Args:
        model: PINN model
        xy: collocation points (N, 2) with requires_grad=True
        material_props: dict with D11, D22, D12, D66, pressure

    Returns:
        PDE residual tensor (N, 1)
    """
    x = xy[:, 0:1]
    y = xy[:, 1:2]

    # Ensure gradients
    x.requires_grad_(True)
    y.requires_grad_(True)
    xy_input = torch.cat([x, y], dim=1)

    w = model(xy_input)

    # Pre-allocate grad_outputs (avoids repeated allocation)
    ones = torch.ones_like(w)

    # First derivatives
    dw_dx = torch.autograd.grad(w, x, grad_outputs=ones, create_graph=True)[0]
    dw_dy = torch.autograd.grad(w, y, grad_outputs=ones, create_graph=True)[0]

    # Second derivatives
    d2w_dx2 = torch.autograd.grad(dw_dx, x, grad_outputs=ones, create_graph=True)[0]
    d2w_dy2 = torch.autograd.grad(dw_dy, y, grad_outputs=ones, create_graph=True)[0]

    # Third derivatives
    d3w_dx3 = torch.autograd.grad(d2w_dx2, x, grad_outputs=ones, create_graph=True)[0]
    d3w_dy3 = torch.autograd.grad(d2w_dy2, y, grad_outputs=ones, create_graph=True)[0]
    # For mixed: d²(d²w/dx²)/dy² → differentiate d2w_dx2 w.r.t. y twice
    d3w_dx2dy = torch.autograd.grad(d2w_dx2, y, grad_outputs=ones, create_graph=True)[0]

    # Fourth derivatives
    d4w_dx4 = torch.autograd.grad(d3w_dx3, x, grad_outputs=ones, create_graph=True)[0]
    d4w_dy4 = torch.autograd.grad(d3w_dy3, y, grad_outputs=ones, create_graph=True)[0]
    d4w_dx2dy2 = torch.autograd.grad(
        d3w_dx2dy, y, grad_outputs=ones, create_graph=True
    )[0]

    # Extract material properties (convert to tensors on same device/dtype)
    D11 = torch.as_tensor(material_props["D11"], device=w.device, dtype=w.dtype)
    D22 = torch.as_tensor(material_props["D22"], device=w.device, dtype=w.dtype)
    D12 = torch.as_tensor(material_props["D12"], device=w.device, dtype=w.dtype)
    D66 = torch.as_tensor(material_props["D66"], device=w.device, dtype=w.dtype)
    q = torch.as_tensor(material_props["pressure"], device=w.device, dtype=w.dtype)

    # PDE residual (normalize by D11 to stabilize magnitudes)
    numerator = D11 * d4w_dx4 + 2 * (D12 + 2 * D66) * d4w_dx2dy2 + D22 * d4w_dy4 - q
    if normalize:
        residual = numerator / (D11 + 1e-12)
    else:
        residual = numerator

    return residual


def compute_boundary_loss(model, boundary_data):
    """
    Compute boundary condition losses:
    - Fixed edge (x=0): w=0, dw/dx=0
    - Simply supported edge (x=L): w=0

    Returns:
        Total boundary loss (scalar)
    """
    # Fixed edge: w = 0
    xy_fixed = boundary_data["fixed_edge"]["xy"].requires_grad_(True)
    w_pred = model(xy_fixed)
    w_target = boundary_data["fixed_edge"]["w"]
    loss_w_fixed = torch.mean((w_pred - w_target) ** 2)

    # Fixed edge: dw/dx = 0
    dw_dx = torch.autograd.grad(
        w_pred, xy_fixed, grad_outputs=torch.ones_like(w_pred), create_graph=True
    )[0][:, 0:1]
    loss_slope_fixed = torch.mean(dw_dx**2)

    # Simply supported: w = 0
    xy_ss = boundary_data["simply_supported"]["xy"]
    w_pred_ss = model(xy_ss)
    w_target_ss = boundary_data["simply_supported"]["w"]
    loss_w_ss = torch.mean((w_pred_ss - w_target_ss) ** 2)

    return loss_w_fixed + loss_slope_fixed + loss_w_ss
