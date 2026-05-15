import torch
import torch.nn as nn


class GaussianActivation(nn.Module):
    """Gaussian activation: f(x) = exp(-x²). Infinitely differentiable (C∞)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-(x**2))


class SinActivation(nn.Module):
    """Sinusoidal activation: f(x) = sin(x). Infinitely differentiable (C∞).

    Used in SIREN (Sinusoidal Representation Networks) for learning
    high-frequency details in PDE solutions.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class CosActivation(nn.Module):
    """Cosine activation: f(x) = cos(x). Infinitely differentiable (C∞)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cos(x)


class ReverseHuberLoss(nn.Module):
    """Reverse Huber loss: L1 for |error| <= delta, MSE for |error| > delta."""

    def __init__(self, delta: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.delta = delta
        self.mse = nn.MSELoss(reduction="none")
        self.l1 = nn.L1Loss(reduction="none")
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1 = self.l1(pred, target)
        mse = self.mse(pred, target)
        loss = torch.where(l1 > self.delta, mse, l1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class AdaptiveLossWeights(nn.Module):
    """Learnable loss weights via homoscedastic uncertainty (Kendall et al., 2018).

    Each task loss is weighted as: weighted_loss = exp(-log_var) * loss + log_var
    This automatically balances losses of different magnitudes during training.
    """

    def __init__(
        self, num_losses: int = 3, initial_weights: list[float] | None = None
    ) -> None:
        super().__init__()
        if initial_weights is not None:
            # log_var = -log(weight) so that exp(-log_var) = weight
            init_vals = torch.tensor(
                [-torch.tensor(w).log().item() for w in initial_weights]
            )
        else:
            init_vals = torch.zeros(num_losses)
        self.register_buffer("_initial_log_vars", init_vals.clone())
        self.log_vars = nn.Parameter(init_vals)

    def forward(self, *losses: torch.Tensor) -> tuple[torch.Tensor, list[float]]:
        """Compute adaptively weighted total loss.

        Args:
            *losses: individual loss tensors (physics, boundary, natural)

        Returns:
            (total_weighted_loss, list of effective weights)
        """
        total = torch.zeros_like(losses[0])
        weights = []
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total = total + precision * loss + self.log_vars[i]
            weights.append(precision.item())
        return total, weights

    _initial_log_vars: torch.Tensor

    def reset(self) -> None:
        """Reset log_vars to their initial values."""
        with torch.no_grad():
            self.log_vars.copy_(self._initial_log_vars.to(self.log_vars.device))


class FFMLP(nn.Module):
    """Gated feed-forward MLP block (SwiGLU variant).

    Computes: up_proj(act(down_proj(x)) * gate_proj(x))
    See: Shazeer, "GLU Variants Improve Transformer", 2020.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.down_proj = nn.Linear(in_features, hidden_features)
        self.act = activation()
        self.up_proj = nn.Linear(hidden_features, out_features)
        self.gate_proj = nn.Linear(in_features, hidden_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down = self.down_proj(x)
        gate = self.gate_proj(x)
        return self.up_proj(self.act(down) * gate)


class ResidualBlock(nn.Module):
    """A simple MLP-style residual block: x -> Linear -> Act -> Linear -> +x

    If the input and output dimensions differ, a linear projection is applied
    to the shortcut path.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: type[nn.Module] = nn.Tanh,
        use_norm: bool = False,
        use_ffmlp: bool = False,
    ) -> None:
        super().__init__()

        self.act = activation()
        if use_ffmlp:
            self.fc1 = FFMLP(
                in_features,
                hidden_features=out_features,
                out_features=out_features,
                activation=activation,
            )
            self.fc2 = FFMLP(
                out_features,
                hidden_features=out_features,
                out_features=out_features,
                activation=activation,
            )
        else:
            self.fc1 = nn.Linear(in_features, out_features)
            self.fc2 = nn.Linear(out_features, out_features)

        self.use_norm = use_norm
        if use_norm:
            # LayerNorm is friendly for small batches common in PINNs
            self.norm1 = nn.LayerNorm(out_features)
            self.norm2 = nn.LayerNorm(out_features)

        if in_features != out_features:
            if use_ffmlp:
                self.shortcut = FFMLP(
                    in_features,
                    hidden_features=out_features,
                    out_features=out_features,
                    activation=activation,
                )
            else:
                self.shortcut = nn.Linear(in_features, out_features)
        else:
            self.shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        hidden_layers: int = 4,
        hidden_units: int = 128,
        activation: type[nn.Module] = nn.Tanh,
        use_residual: bool = False,
        use_norm: bool = False,
        use_ffmlp: bool = False,
    ) -> None:
        super().__init__()
        layers = []
        in_features = 2

        if use_residual:
            # Initial projection to hidden_units
            layers.append(nn.Linear(in_features, hidden_units))
            layers.append(activation())
            # Stack residual blocks
            for _ in range(hidden_layers):
                layers.append(
                    ResidualBlock(
                        hidden_units, hidden_units, activation, use_norm, use_ffmlp
                    )
                )

        else:
            for _ in range(hidden_layers):
                if use_ffmlp:
                    layers.append(
                        FFMLP(
                            in_features,
                            hidden_features=hidden_units,
                            out_features=hidden_units,
                            activation=activation,
                        )
                    )
                else:
                    layers.append(nn.Linear(in_features, hidden_units))
                layers.append(activation())
                in_features = hidden_units

        # Final projection to output
        layers.append(nn.Linear(hidden_units, 1))

        # Use Sequential to keep backward compatibility
        self.net = nn.Sequential(*layers)

        # Initialize weights for all Linear modules
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """Forward pass: predict displacement w given (x, y) coordinates."""
        return self.net(xy)
