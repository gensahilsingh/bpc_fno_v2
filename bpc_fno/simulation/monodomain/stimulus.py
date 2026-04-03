"""Stimulus helpers for regular-grid tissue simulations."""

from __future__ import annotations

import numpy as np


class SphericalStimulus:
    """Stimulus applied to a spherical region around a pacing voxel."""

    def __init__(
        self,
        grid_shape: tuple[int, int, int],
        voxel_size_cm: float,
        pacing_site_voxel: tuple[int, int, int],
        magnitude_uA_cm2: float,
        duration_ms: float,
        start_ms: float,
        radius_cm: float,
    ) -> None:
        nx, ny, nz = grid_shape
        h = float(voxel_size_cm)
        px, py, pz = pacing_site_voxel
        cx = (px + 0.5) * h
        cy = (py + 0.5) * h
        cz = (pz + 0.5) * h

        coords = np.stack(
            np.meshgrid(
                (np.arange(nx, dtype=np.float64) + 0.5) * h,
                (np.arange(ny, dtype=np.float64) + 0.5) * h,
                (np.arange(nz, dtype=np.float64) + 0.5) * h,
                indexing="ij",
            ),
            axis=-1,
        )
        dist = np.sqrt(
            (coords[..., 0] - cx) ** 2
            + (coords[..., 1] - cy) ** 2
            + (coords[..., 2] - cz) ** 2
        )
        self.mask_3d: np.ndarray = dist <= float(radius_cm)
        self._mask_flat = self.mask_3d.ravel()
        self._magnitude = float(magnitude_uA_cm2)
        self._start_ms = float(start_ms)
        self._end_ms = float(start_ms + duration_ms)

    def __call__(self, t_ms: float) -> np.ndarray:
        current = np.zeros(self._mask_flat.shape[0], dtype=np.float64)
        if self._start_ms <= t_ms < self._end_ms:
            current[self._mask_flat] = self._magnitude
        return current
