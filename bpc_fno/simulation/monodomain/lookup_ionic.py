"""Approximate ionic-current lookup model for local smoke monodomain runs."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


class LookupTableIonicCallback:
    """Approximate voxelwise ionic current using cell-type lookup tables.

    This is intentionally only for the local Windows smoke path. The lookup
    tables are driven by per-cell-type TT2006 traces and eikonal activation
    delays rather than full voxelwise ionic-state integration.
    """

    def __init__(
        self,
        cell_type_map: np.ndarray,
        activation_times_ms: np.ndarray,
        ap_waveforms: dict[int, dict[str, np.ndarray]],
    ) -> None:
        self.cell_type_map = np.asarray(cell_type_map, dtype=np.int8).ravel()
        self.activation_times_ms = np.asarray(
            activation_times_ms, dtype=np.float64
        ).ravel()
        self.n_voxels = self.cell_type_map.shape[0]

        self._current_interps: dict[int, interp1d] = {}
        for cell_type_idx, result in ap_waveforms.items():
            t_ms = np.asarray(result["t_ms"], dtype=np.float64)
            i_ion = np.asarray(result["I_ion_total"], dtype=np.float64)
            self._current_interps[cell_type_idx] = interp1d(
                t_ms,
                i_ion,
                kind="linear",
                bounds_error=False,
                fill_value=(float(i_ion[0]), float(i_ion[-1])),
            )

    def __call__(
        self,
        V_flat: np.ndarray,
        dt_ms: float,
        t_global_ms: float,
    ) -> np.ndarray:
        _ = (V_flat, dt_ms)  # reserved for a future fully coupled callback API
        local_t = np.maximum(0.0, t_global_ms - self.activation_times_ms)
        i_ion = np.zeros(self.n_voxels, dtype=np.float64)

        for cell_type_idx, interp_fn in self._current_interps.items():
            mask = self.cell_type_map == cell_type_idx
            if np.any(mask):
                i_ion[mask] = interp_fn(local_t[mask])

        return i_ion
