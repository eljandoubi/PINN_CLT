"""Loss computation and gradient utilities for training."""

import torch
import torch.nn as nn

from model import (
    PINN,
    AdaptiveLossWeights,
)


def compute_pde_residual(
    model: PINN,
    xy: torch.Tensor,
    material_props: dict[str, float],
    normalize: bool = False,
) -> torch.Tensor:
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


def zero_loss(criterion: nn.Module, residual: torch.Tensor) -> torch.Tensor:
    """Compute criterion(residual, 0) without allocating a zero tensor.

    For standard losses against a zero target, this avoids the memory
    overhead of torch.zeros_like(residual) by computing directly:
      MSE -> mean(residual²)
      L1  -> mean(|residual|)
      Huber/ReverseHuber -> use scalar zero broadcast
    """
    if isinstance(criterion, nn.MSELoss):
        sq = residual.pow(2)
        if criterion.reduction == "mean":
            return sq.mean()
        elif criterion.reduction == "sum":
            return sq.sum()
        return sq
    elif isinstance(criterion, nn.L1Loss):
        ab = residual.abs()
        if criterion.reduction == "mean":
            return ab.mean()
        elif criterion.reduction == "sum":
            return ab.sum()
        return ab
    else:
        # Huber, ReverseHuber, etc.: use scalar zero (no allocation)
        return criterion(residual, residual.new_zeros(1).expand_as(residual))


def compute_boundary_loss(
    model: PINN,
    boundary_data: dict[str, dict[str, torch.Tensor]],
    criterion: nn.Module | None = None,
) -> torch.Tensor:
    """
    Compute essential boundary condition losses:
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
    x_fixed = boundary_data["fixed_edge"]["x"].requires_grad_(True)
    y_fixed = boundary_data["fixed_edge"]["y"].requires_grad_(True)
    xy_fixed = torch.cat([x_fixed, y_fixed], dim=1)

    w_pred = model(xy_fixed)
    w_target = boundary_data["fixed_edge"]["w"]
    loss_w_fixed = criterion(w_pred, w_target)

    # Fixed edge: dw/dx = 0
    dw_dx = torch.autograd.grad(
        w_pred, x_fixed, grad_outputs=torch.ones_like(w_pred), create_graph=True
    )[0]
    loss_slope_fixed = zero_loss(criterion, dw_dx)

    # Simply supported: w = 0
    xy_ss = boundary_data["simply_supported"]["xy"]
    w_pred_ss = model(xy_ss)
    w_target_ss = boundary_data["simply_supported"]["w"]
    loss_w_ss = criterion(w_pred_ss, w_target_ss)

    return loss_w_fixed + loss_slope_fixed + loss_w_ss


def compute_natural_bc_loss(
    model: PINN,
    boundary_data: dict[str, dict[str, torch.Tensor]],
    material_props: dict[str, float],
    criterion: nn.Module | None = None,
    normalize: bool = False,
) -> torch.Tensor:
    """
    Compute natural boundary condition losses:
    - Simply supported edge (x=L): Mx = -(D11·∂²w/∂x² + D12·∂²w/∂y²) = 0
    - Free edges (y=0, y=W): My = -(D12·∂²w/∂x² + D22·∂²w/∂y²) = 0
    - Free edges (y=0, y=W): Vy = -(D22·∂³w/∂y³ + (D12+2D66)·∂³w/∂x²∂y) = 0

    Args:
        model: PINN model
        boundary_data: dict with boundary condition data
        material_props: dict with D11, D22, D12, D66
        criterion: loss function (default: MSELoss)

    Returns:
        Total natural BC loss (scalar)
    """
    if criterion is None:
        criterion = torch.nn.MSELoss()

    D11 = material_props["D11"]
    D22 = material_props["D22"]
    D12 = material_props["D12"]
    D66 = material_props["D66"]

    # --- Simply Supported Edge (x=L): Mx = 0 ---
    x_ss = boundary_data["simply_supported"]["x"].requires_grad_(True)
    y_ss = boundary_data["simply_supported"]["y"].requires_grad_(True)
    xy_ss_input = torch.cat([x_ss, y_ss], dim=1)

    w_ss = model(xy_ss_input)
    ones = torch.ones_like(w_ss)

    dw_dx_ss = torch.autograd.grad(w_ss, x_ss, grad_outputs=ones, create_graph=True)[0]
    dw_dy_ss = torch.autograd.grad(w_ss, y_ss, grad_outputs=ones, create_graph=True)[0]

    d2w_dx2_ss = torch.autograd.grad(
        dw_dx_ss, x_ss, grad_outputs=ones, create_graph=True
    )[0]
    d2w_dy2_ss = torch.autograd.grad(
        dw_dy_ss, y_ss, grad_outputs=ones, create_graph=True
    )[0]

    Mx_ss = -(D11 * d2w_dx2_ss + D12 * d2w_dy2_ss)
    if normalize:
        Mx_ss = Mx_ss / (D11 + 1e-12)
    loss_Mx_ss = zero_loss(criterion, Mx_ss)
    total_loss = loss_Mx_ss

    # --- Free Edge y=0: My = 0, Vy = 0 ---
    x_y0 = boundary_data["free_edge_y0"]["x"].requires_grad_(True)
    y_y0 = boundary_data["free_edge_y0"]["y"].requires_grad_(True)
    xy_y0_input = torch.cat([x_y0, y_y0], dim=1)

    w_y0 = model(xy_y0_input)
    ones_y0 = torch.ones_like(w_y0)

    dw_dx_y0 = torch.autograd.grad(w_y0, x_y0, grad_outputs=ones_y0, create_graph=True)[
        0
    ]
    dw_dy_y0 = torch.autograd.grad(w_y0, y_y0, grad_outputs=ones_y0, create_graph=True)[
        0
    ]

    d2w_dx2_y0 = torch.autograd.grad(
        dw_dx_y0, x_y0, grad_outputs=ones_y0, create_graph=True
    )[0]
    d2w_dy2_y0 = torch.autograd.grad(
        dw_dy_y0, y_y0, grad_outputs=ones_y0, create_graph=True
    )[0]

    My_y0 = -(D12 * d2w_dx2_y0 + D22 * d2w_dy2_y0)
    if normalize:
        My_y0 = My_y0 / (D11 + 1e-12)
    loss_My_y0 = zero_loss(criterion, My_y0)
    total_loss = total_loss + loss_My_y0

    d3w_dy3_y0 = torch.autograd.grad(
        d2w_dy2_y0, y_y0, grad_outputs=ones_y0, create_graph=True
    )[0]
    d3w_dx2dy_y0 = torch.autograd.grad(
        d2w_dx2_y0, y_y0, grad_outputs=ones_y0, create_graph=True
    )[0]

    Vy_y0 = -(D22 * d3w_dy3_y0 + (D12 + 2 * D66) * d3w_dx2dy_y0)
    if normalize:
        Vy_y0 = Vy_y0 / (D11 + 1e-12)
    loss_Vy_y0 = zero_loss(criterion, Vy_y0)
    total_loss = total_loss + loss_Vy_y0

    # --- Free Edge y=W: My = 0, Vy = 0 ---
    x_yW = boundary_data["free_edge_yW"]["x"].requires_grad_(True)
    y_yW = boundary_data["free_edge_yW"]["y"].requires_grad_(True)
    xy_yW_input = torch.cat([x_yW, y_yW], dim=1)

    w_yW = model(xy_yW_input)
    ones_yW = torch.ones_like(w_yW)

    dw_dx_yW = torch.autograd.grad(w_yW, x_yW, grad_outputs=ones_yW, create_graph=True)[
        0
    ]
    dw_dy_yW = torch.autograd.grad(w_yW, y_yW, grad_outputs=ones_yW, create_graph=True)[
        0
    ]

    d2w_dx2_yW = torch.autograd.grad(
        dw_dx_yW, x_yW, grad_outputs=ones_yW, create_graph=True
    )[0]
    d2w_dy2_yW = torch.autograd.grad(
        dw_dy_yW, y_yW, grad_outputs=ones_yW, create_graph=True
    )[0]

    My_yW = -(D12 * d2w_dx2_yW + D22 * d2w_dy2_yW)
    if normalize:
        My_yW = My_yW / (D11 + 1e-12)
    loss_My_yW = zero_loss(criterion, My_yW)
    total_loss = total_loss + loss_My_yW

    d3w_dy3_yW = torch.autograd.grad(
        d2w_dy2_yW, y_yW, grad_outputs=ones_yW, create_graph=True
    )[0]
    d3w_dx2dy_yW = torch.autograd.grad(
        d2w_dx2_yW, y_yW, grad_outputs=ones_yW, create_graph=True
    )[0]

    Vy_yW = -(D22 * d3w_dy3_yW + (D12 + 2 * D66) * d3w_dx2dy_yW)
    if normalize:
        Vy_yW = Vy_yW / (D11 + 1e-12)
    loss_Vy_yW = zero_loss(criterion, Vy_yW)
    total_loss = total_loss + loss_Vy_yW

    return total_loss


def compute_losses(
    model: PINN,
    xy_batch: torch.Tensor,
    material_props: dict[str, float],
    boundary_data: dict[str, dict[str, torch.Tensor]],
    criterion: nn.Module,
    lambda_physics: float,
    lambda_boundary: float,
    lambda_natural: float,
    adaptive_weighter: AdaptiveLossWeights | None = None,
    normalize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[float]]:
    """Forward pass: compute all loss components.

    Args:
        model: PINN model.
        xy_batch: collocation points (N, 2).
        material_props: dict with D11, D22, D12, D66, pressure.
        boundary_data: dict with boundary condition data.
        criterion: loss function (MSE, Huber, etc.).
        lambda_physics: PDE loss weight (used when adaptive_weighter is None).
        lambda_boundary: boundary loss weight.
        lambda_natural: natural BC loss weight.
        adaptive_weighter: optional learnable loss weighter.
        normalize: whether to normalize PDE/natural BC losses by D11.

    Returns:
        (total_loss, physics_loss, boundary_loss, natural_loss, effective_weights)
    """
    residual = compute_pde_residual(
        model, xy_batch, material_props, normalize=normalize
    )
    l_phys = zero_loss(criterion, residual)
    l_bc = compute_boundary_loss(model, boundary_data, criterion)
    l_nat = compute_natural_bc_loss(
        model,
        boundary_data,
        material_props,
        criterion,
        normalize=normalize,
    )
    if adaptive_weighter is not None:
        total, w = adaptive_weighter(l_phys, l_bc, l_nat)
    else:
        total = (
            lambda_physics * l_phys + lambda_boundary * l_bc + lambda_natural * l_nat
        )
        w = [lambda_physics, lambda_boundary, lambda_natural]
    return total, l_phys, l_bc, l_nat, w


def clip_grads(
    model: PINN,
    max_grad_norm: float,
    adaptive_weighter: AdaptiveLossWeights | None = None,
) -> None:
    """Clip gradients for model (and optional adaptive weighter).

    Args:
        model: PINN model.
        max_grad_norm: maximum gradient norm for clipping.
        adaptive_weighter: optional learnable loss weighter.
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
    if adaptive_weighter is not None:
        torch.nn.utils.clip_grad_norm_(
            adaptive_weighter.parameters(), max_norm=max_grad_norm
        )
