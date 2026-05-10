"""Early stopping utility."""


class EarlyStopping:
    """Stop training when loss stops improving."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-8) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: float | None = None

    def step(self, loss: float) -> bool:
        """Returns True if training should stop."""
        if self.best_loss is None or loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

    def reset(self) -> None:
        """Reset the early stopping state."""
        self.counter = 0
        self.best_loss = None
