"""Seed setting and deterministic mode utilities for reproducible experiments."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set seeds for all relevant RNGs to ensure reproducibility.

    Sets seeds for the Python ``random`` module, NumPy, PyTorch CPU, and
    PyTorch CUDA (all devices).

    Parameters
    ----------
    seed:
        Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def enable_deterministic_mode() -> None:
    """Enable fully deterministic behaviour in PyTorch.

    This function:
    * Calls ``torch.use_deterministic_algorithms(True)``.
    * Sets the ``CUBLAS_WORKSPACE_CONFIG`` environment variable required by
      CUDA >= 10.2 to avoid non-deterministic cuBLAS calls.
    * Disables cuDNN benchmarking and enables cuDNN deterministic mode.

    .. warning::
        Deterministic mode may significantly reduce performance.
    """
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_rng(seed: int) -> np.random.Generator:
    """Return a seeded NumPy random ``Generator``.

    Parameters
    ----------
    seed:
        Integer seed for the generator.

    Returns
    -------
    np.random.Generator
        A new NumPy random generator seeded with *seed*.
    """
    return np.random.default_rng(seed)
