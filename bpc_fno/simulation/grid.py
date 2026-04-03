"""Utilities for working with regular Cartesian tissue grids."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

GridShape = tuple[int, int, int]


def coerce_grid_shape(value: Any) -> GridShape:
    """Normalize a scalar or 3-sequence grid specification to ``(Nx, Ny, Nz)``."""
    if isinstance(value, (int, np.integer)):
        size = int(value)
        if size <= 0:
            raise ValueError(f"grid_size must be positive, got {size}.")
        return (size, size, size)

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 3:
            raise ValueError(
                "grid_shape must have exactly 3 entries (Nx, Ny, Nz)."
            )
        shape = tuple(int(v) for v in value)
        if any(v <= 0 for v in shape):
            raise ValueError(f"grid_shape must be positive, got {shape}.")
        return shape

    raise TypeError(
        f"Unsupported grid specification {value!r}; expected int or 3-sequence."
    )


def resolve_grid_shape(config_or_section: Any) -> GridShape:
    """Resolve ``(Nx, Ny, Nz)`` from a config object or config subsection."""
    section = getattr(config_or_section, "simulation", config_or_section)

    if hasattr(section, "grid_shape"):
        return coerce_grid_shape(getattr(section, "grid_shape"))
    if hasattr(section, "grid_size"):
        return coerce_grid_shape(getattr(section, "grid_size"))

    if isinstance(section, dict):
        if "grid_shape" in section:
            return coerce_grid_shape(section["grid_shape"])
        if "grid_size" in section:
            return coerce_grid_shape(section["grid_size"])

    raise AttributeError(
        "Could not resolve grid shape; expected 'grid_shape' or 'grid_size'."
    )


def build_voxel_centers(
    grid_shape: GridShape,
    voxel_size_cm: float,
) -> np.ndarray:
    """Return voxel-center coordinates in cm for a regular Cartesian grid."""
    nx, ny, nz = grid_shape
    xs = (np.arange(nx, dtype=np.float64) + 0.5) * voxel_size_cm
    ys = (np.arange(ny, dtype=np.float64) + 0.5) * voxel_size_cm
    zs = (np.arange(nz, dtype=np.float64) + 0.5) * voxel_size_cm
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)


def build_output_times(total_time_ms: float, n_timesteps: int) -> np.ndarray:
    """Return evenly spaced snapshot times without duplicating the endpoint."""
    if n_timesteps <= 0:
        raise ValueError(f"n_timesteps must be positive, got {n_timesteps}.")
    dt_out = float(total_time_ms) / float(n_timesteps)
    return np.arange(n_timesteps, dtype=np.float64) * dt_out


def select_time_indices(total_steps: int, n_outputs: int) -> np.ndarray:
    """Return evenly spaced integer indices across ``total_steps`` snapshots."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive.")
    if n_outputs <= 0:
        raise ValueError("n_outputs must be positive.")
    return np.linspace(0, total_steps - 1, n_outputs, dtype=int)
