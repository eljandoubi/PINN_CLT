import torch
import torch.nn as nn


class ReverseHuberLoss(nn.Module):
    """Reverse Huber loss: L1 for |error| <= delta, MSE for |error| > delta."""

    def __init__(self, delta: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.delta = delta
        self.mse = nn.MSELoss(reduction="none")
        self.l1 = nn.L1Loss(reduction="none")
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1 = self.l1(input, target)
        mse = self.mse(input, target)
        loss = torch.where(l1 > self.delta, mse, l1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def zero_loss(criterion: nn.Module, input: torch.Tensor) -> torch.Tensor:
    """Compute criterion(input, 0) without allocating a zero tensor.

    For standard losses against a zero target, this avoids the memory
    overhead of torch.zeros_like(input) by computing directly:
      MSE -> mean(input²)
      L1  -> mean(|input|)
      Huber/ReverseHuber -> use scalar zero broadcast
    """
    if isinstance(criterion, nn.MSELoss):
        return input.pow(2).mean()
    elif isinstance(criterion, nn.L1Loss):
        return input.abs().mean()
    else:
        # Huber, ReverseHuber, etc.: use scalar zero (no allocation)
        return criterion(input, input.new_zeros(1).expand_as(input))


class ResidualBlock(nn.Module):
    """A simple MLP-style residual block: x -> Linear -> Act -> Linear -> +x

    If the input and output dimensions differ, a linear projection is applied
    to the shortcut path.
    """

    def __init__(self, in_features, out_features, activation=nn.Tanh, use_norm=False):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.act = activation()
        self.fc2 = nn.Linear(out_features, out_features)

        self.use_norm = use_norm
        if use_norm:
            # LayerNorm is friendly for small batches common in PINNs
            self.norm1 = nn.LayerNorm(out_features)
            self.norm2 = nn.LayerNorm(out_features)

        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
        else:
            self.shortcut = None

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        if self.use_norm:
            out = self.norm1(out)
        out = self.act(out)
        out = self.fc2(out)
        if self.use_norm:
            out = self.norm2(out)
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        return self.act(out + identity)


class PINN(nn.Module):
    """Physics-Informed Neural Network for orthotropic plate bending (CLT).

    Options:
    - `use_residual`: when True, builds the hidden layers as `ResidualBlock`s.
    - `use_norm`: applies `LayerNorm` inside residual blocks (recommended
      for small batches).
    """

    def __init__(
        self,
        hidden_layers=4,
        hidden_units=64,
        activation=nn.Tanh,
        use_residual: bool = False,
        use_norm: bool = False,
    ):
        super().__init__()
        layers = []
        in_features = 2

        if use_residual:
            # Initial projection to hidden_units
            layers.append(nn.Linear(in_features, hidden_units))
            layers.append(activation())
            in_features = hidden_units
            # Stack residual blocks
            for _ in range(hidden_layers):
                layers.append(
                    ResidualBlock(in_features, hidden_units, activation, use_norm)
                )
            # Final projection to output
            layers.append(nn.Linear(hidden_units, 1))
        else:
            for _ in range(hidden_layers):
                layers.append(nn.Linear(in_features, hidden_units))
                layers.append(activation())
                in_features = hidden_units
            layers.append(nn.Linear(in_features, 1))

        # Use Sequential to keep backward compatibility
        self.net = nn.Sequential(*layers)

        # Initialize weights for all Linear modules
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
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


def compute_boundary_loss(model, boundary_data, criterion=None):
    """
    Compute boundary condition losses:
    - Fixed edge (x=0): w=0, dw/dx=0
    - Simply supported edge (x=L): w=0

    Args:
        model: PINN model
        boundary_data: dict with boundary condition data
        criterion: loss function (default: MSELoss)

    Returns:
        Total boundary loss (scalar)
    """
    if criterion is None:
        criterion = torch.nn.MSELoss()

    # Fixed edge: w = 0
    xy_fixed = boundary_data["fixed_edge"]["xy"].requires_grad_(True)
    w_pred = model(xy_fixed)
    w_target = boundary_data["fixed_edge"]["w"]
    loss_w_fixed = criterion(w_pred, w_target)

    # Fixed edge: dw/dx = 0
    dw_dx = torch.autograd.grad(
        w_pred, xy_fixed, grad_outputs=torch.ones_like(w_pred), create_graph=True
    )[0][:, 0:1]
    loss_slope_fixed = zero_loss(criterion, dw_dx)

    # Simply supported: w = 0
    xy_ss = boundary_data["simply_supported"]["xy"]
    w_pred_ss = model(xy_ss)
    w_target_ss = boundary_data["simply_supported"]["w"]
    loss_w_ss = criterion(w_pred_ss, w_target_ss)

    return loss_w_fixed + loss_slope_fixed + loss_w_ss
