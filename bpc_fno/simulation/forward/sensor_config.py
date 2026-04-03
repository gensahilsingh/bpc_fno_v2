"""Sensor position definitions for KCD 4-sensor array and virtual sensor grids.

Provides helper methods to obtain sensor coordinates in centimetres for
both the physical KCD magnetometer array and the virtual sensor grid used
during synthetic training-data generation.
"""

from __future__ import annotations

import numpy as np
from omegaconf import DictConfig

from bpc_fno.simulation.grid import resolve_grid_shape


# KCD 4-sensor array layout (cm) — fixed hardware geometry.
_KCD_SENSOR_OFFSETS_CM: np.ndarray = np.array(
    [
        [0.0, 0.0, 0.0],   # Sensor 0
        [-3.0, 0.0, 3.0],  # Sensor 1
        [-3.0, 0.0, 0.0],  # Sensor 2
        [0.0, 0.0, 3.0],   # Sensor 3
    ],
    dtype=np.float64,
)


class SensorConfig:
    """Manage sensor positions for forward-model computations.

    Parameters
    ----------
    config : DictConfig
        Hydra/OmegaConf configuration.  Expected keys under
        ``config.simulation.sensor``:

        - ``virtual_sensor_grid_n`` — number of virtual sensors per axis
        - ``virtual_sensor_spacing_cm`` — spacing between virtual sensors
        - ``sensor_height_cm`` — height of sensor plane above tissue (y-axis)

        Also reads ``config.simulation.grid_size`` and
        ``config.simulation.voxel_size_cm`` for grid geometry.
    """

    def __init__(self, config: DictConfig) -> None:
        # sensor config may be at config.sensor or config.simulation.sensor
        if hasattr(config, "sensor"):
            sensor_cfg = config.sensor
        else:
            sensor_cfg = config.simulation.sensor

        self.virtual_sensor_grid_n: int = int(sensor_cfg.virtual_sensor_grid_n)
        self.virtual_sensor_spacing_cm: float = float(sensor_cfg.virtual_sensor_spacing_cm)
        self.sensor_height_cm: float = float(
            getattr(sensor_cfg, "sensor_height_cm", 14.0)
        )

        # Grid geometry for centering the virtual sensor array.
        self.grid_shape: tuple[int, int, int] = resolve_grid_shape(config)
        self.voxel_size_cm: float = float(config.simulation.voxel_size_cm)

    # ------------------------------------------------------------------
    # KCD physical array
    # ------------------------------------------------------------------

    def get_kcd_sensor_positions(
        self,
        array_offset_cm: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return KCD 4-sensor positions in centimetres.

        Parameters
        ----------
        array_offset_cm : np.ndarray | None
            Optional shape ``(3,)`` offset added to every sensor position.

        Returns
        -------
        np.ndarray
            Shape ``(4, 3)`` float64 — sensor positions in cm.
        """
        positions = _KCD_SENSOR_OFFSETS_CM.copy()
        if array_offset_cm is not None:
            positions += np.asarray(array_offset_cm, dtype=np.float64).reshape(3)
        return positions

    # ------------------------------------------------------------------
    # Virtual sensor grid
    # ------------------------------------------------------------------

    def get_virtual_sensor_positions(self) -> np.ndarray:
        """Return virtual sensor positions on a regular grid.

        The grid is ``virtual_sensor_grid_n x virtual_sensor_grid_n``
        positions in the x-z plane, centred over the tissue slab.  The
        y-coordinate is set to ``sensor_height_cm``.

        Returns
        -------
        np.ndarray
            Shape ``(N_virtual, 3)`` float64 — sensor positions in cm,
            where ``N_virtual = virtual_sensor_grid_n ** 2``.
        """
        n = self.virtual_sensor_grid_n
        spacing = self.virtual_sensor_spacing_cm

        # Centre of the tissue slab in the x-z plane.
        tissue_centre_x = self.grid_shape[0] * self.voxel_size_cm / 2.0
        tissue_centre_z = self.grid_shape[2] * self.voxel_size_cm / 2.0

        # 1-D sensor coordinates centred at tissue_centre.
        offsets = np.arange(n, dtype=np.float64) - (n - 1) / 2.0
        x_coords = tissue_centre_x + offsets * spacing
        z_coords = tissue_centre_z + offsets * spacing

        # Meshgrid in x-z; y is constant at sensor_height_cm.
        xg, zg = np.meshgrid(x_coords, z_coords, indexing="ij")
        positions = np.stack(
            [
                xg.ravel(),
                np.full(n * n, self.sensor_height_cm, dtype=np.float64),
                zg.ravel(),
            ],
            axis=-1,
        )
        return positions

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_n_sensors(self, mode: str = "virtual") -> int:
        """Return the number of sensors for the given mode.

        Parameters
        ----------
        mode : str
            ``'kcd'`` for the physical 4-sensor array, ``'virtual'`` for
            the virtual training grid.

        Returns
        -------
        int
            Number of sensor positions.

        Raises
        ------
        ValueError
            If *mode* is not recognised.
        """
        if mode == "kcd":
            return 4
        if mode == "virtual":
            return self.virtual_sensor_grid_n ** 2
        raise ValueError(f"Unknown sensor mode '{mode}'; expected 'kcd' or 'virtual'.")
