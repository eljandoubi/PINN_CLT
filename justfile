# Run `just` with no arguments to list all available recipes.
default:
    @just --list

# Install/sync all dependencies, including dev tools (ruff, pytest-cov)
install:
    uv sync --locked

# Run the full test suite (including slow end-to-end training smoke tests)
test:
    uv run pytest

# Run only the fast unit tests, skipping slow end-to-end smoke tests
test-fast:
    uv run pytest -m "not slow"

# Run tests with terminal + HTML coverage reports (see htmlcov/index.html)
cov:
    uv run pytest --cov --cov-report=term-missing --cov-report=html

# Lint the codebase with ruff
lint:
    uv run ruff check .

# Auto-fix lint issues where possible
fix:
    uv run ruff check --fix .

# Format the codebase with ruff
format:
    uv run ruff format .

# Check formatting without modifying any files (used in CI)
format-check:
    uv run ruff format --check .

# Run everything CI runs: lint, format check, and tests with coverage
check: lint format-check cov

# Train the PINN, forwarding any extra CLI args, e.g. `just train --epochs 5000`
train *args:
    uv run train.py {{ args }}

# Regenerate the training video from saved plot frames
video:
    uv run video.py

# Remove caches, build artifacts, and local run outputs
clean:
    rm -rf .pytest_cache .ruff_cache htmlcov .coverage coverage.xml runs wandb checkpoints plots
