"""Shared pytest configuration and fixtures."""

import os

import matplotlib

# Force a non-interactive backend so plotting tests work on headless CI runners.
matplotlib.use("Agg")

# Never let a test hit the network / W&B servers unless explicitly overridden.
os.environ.setdefault("WANDB_MODE", "offline")
