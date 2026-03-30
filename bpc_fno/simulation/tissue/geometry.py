"""Ventricular slab geometry: voxel grid, fiber orientation (Streeter rule), and SDF.

This module provides the spatial discretisation for a transmural ventricular
slab used in monodomain tissue-level simulations.  The slab is a regular
Cartesian grid with configurable resolution and voxel spacing.

Physics references
------------------
* Streeter DD et al., "Fiber orientation in the canine left ventricle during
  diastole and systole", Circ Res 24(3):339-347, 1969.
* Bishop MJ et al., "Development of an anatomically detailed MRI-derived
  rabbit ventricular model and assessment of its impact on simulations of
  electrophysiological function", Am J Physiol Heart Circ Physiol, 2010.
* Clayton RH & Panfilov AV, "A guide to modelling cardiac electrical
  activity in anatomically detailed ventricles", Prog Biophys Mol Biol,
  96(1-3):19-43, 2008.
"""

from __future__ import annotations

import numpy as np


class VentricularSlab:
    """Rectangular slab of ventricular myocardium on a regular Cartesian grid.

    The transmural axis is aligned with the x-axis (index 0): endocardium at
    ``x = 0``, epicardium at ``x = N - 1``.  Fibre orientation follows the
    Streeter rule with a linear rotation from -60 deg (endo) to +60 deg (epi).

    Parameters
    ----------
    grid_size : int
        Number of voxels along each axis (*N*).  Default 32.
    voxel_size_cm : float
        Edge length of each cubic voxel in centimetres.  Default 0.05 cm
        (500 um).
    layer_fractions : list[float]
        Fractional thickness of endocardial, midmyocardial and epicardial
        layers.  Must sum to 1.  Default ``[0.33, 0.33, 0.34]``.
    """

    # Streeter rule endpoint angles (degrees)
    _ENDO_ANGLE_DEG: float = -60.0
    _EPI_ANGLE_DEG: float = 60.0

    def __init__(
        self,
        grid_size: int = 32,
        voxel_size_cm: float = 0.05,
        layer_fractions: list[float] | None = None,
    ) -> None:
        self.grid_size: int = grid_size
        self.voxel_size_cm: float = voxel_size_cm
        self.layer_fractions: list[float] = (
            layer_fractions if layer_fractions is not None else [0.33, 0.33, 0.34]
        )

        if len(self.layer_fractions) != 3:
            raise ValueError("layer_fractions must have exactly 3 elements (endo/mid/epi).")
        if not np.isclose(sum(self.layer_fractions), 1.0):
            raise ValueError(f"layer_fractions must sum to 1.0, got {sum(self.layer_fractions):.6f}.")

        # Physical dimensions (cm)
        self.physical_size_cm: float = grid_size * voxel_size_cm

    # ------------------------------------------------------------------
    # Fiber orientation
    # ------------------------------------------------------------------

    def get_fiber_field(self) -> np.ndarray:
        """Compute the fibre-orientation vector field using the Streeter rule.

        The fibre angle interpolates linearly across the transmural axis:
        ``angle(x) = -60 + 120 * x / (N - 1)`` degrees, where *x* is the
        voxel index along the transmural axis.

        The fibre direction at each voxel is
        ``(0, sin(angle), cos(angle))``, lying in the y-z plane.

        Returns
        -------
        np.ndarray
            Shape ``(N, N, N, 3)`` float64 — unit fibre vectors.
        """
        N = self.grid_size

        # Transmural fractional coordinate for each x-index
        x_idx = np.arange(N, dtype=np.float64)
        angle_deg = self._ENDO_ANGLE_DEG + (self._EPI_ANGLE_DEG - self._ENDO_ANGLE_DEG) * x_idx / max(N - 1, 1)
        angle_rad = np.deg2rad(angle_deg)  # shape (N,)

        fiber = np.zeros((N, N, N, 3), dtype=np.float64)
        # Broadcast angle across y and z axes
        fiber[:, :, :, 0] = 0.0
        fiber[:, :, :, 1] = np.sin(angle_rad)[:, None, None]
        fiber[:, :, :, 2] = np.cos(angle_rad)[:, None, None]

        return fiber

    # ------------------------------------------------------------------
    # Signed Distance Function
    # ------------------------------------------------------------------

    def get_sdf(self) -> np.ndarray:
        """Compute a signed-distance function for the slab geometry.

        ``SDF[x, y, z] = min(x, N - 1 - x) * voxel_size_cm``

        The SDF is positive inside the myocardium and reaches zero at the
        endocardial (``x = 0``) and epicardial (``x = N - 1``) surfaces.

        Returns
        -------
        np.ndarray
            Shape ``(N, N, N)`` float64 — signed distance in cm.
        """
        N = self.grid_size
        x_idx = np.arange(N, dtype=np.float64)
        dist_from_wall = np.minimum(x_idx, (N - 1) - x_idx) * self.voxel_size_cm  # (N,)
        sdf = np.broadcast_to(dist_from_wall[:, None, None], (N, N, N)).copy()
        return sdf

    # ------------------------------------------------------------------
    # Cell-type map
    # ------------------------------------------------------------------

    def get_cell_type_map(self) -> np.ndarray:
        """Assign cell types based on transmural position.

        * 0 — endocardial  (``x < N * layer_fractions[0]``)
        * 1 — midmyocardial (``x < N * (layer_fractions[0] + layer_fractions[1])``)
        * 2 — epicardial    (remainder)

        Returns
        -------
        np.ndarray
            Shape ``(N, N, N)`` int8.
        """
        N = self.grid_size
        endo_end = N * self.layer_fractions[0]
        mid_end = N * (self.layer_fractions[0] + self.layer_fractions[1])

        x_idx = np.arange(N, dtype=np.float64)
        layer_1d = np.where(x_idx < endo_end, 0, np.where(x_idx < mid_end, 1, 2)).astype(np.int8)

        cell_map = np.broadcast_to(layer_1d[:, None, None], (N, N, N)).copy()
        return cell_map

    # ------------------------------------------------------------------
    # Combined geometry tensor
    # ------------------------------------------------------------------

    def get_geometry_tensor(self) -> np.ndarray:
        """Stack SDF and fibre vectors into a single geometry tensor.

        Channel ordering: ``[SDF, fiber_x, fiber_y, fiber_z]``.

        Returns
        -------
        np.ndarray
            Shape ``(N, N, N, 4)`` float64.
        """
        sdf = self.get_sdf()
        fiber = self.get_fiber_field()
        geom = np.concatenate([sdf[..., None], fiber], axis=-1)
        return geom

    # ------------------------------------------------------------------
    # Fibrosis generation
    # ------------------------------------------------------------------

    def add_fibrosis(
        self,
        rng: np.random.Generator,
        density: float,
        blob_sigma_voxels: float,
    ) -> np.ndarray:
        """Generate a binary fibrosis mask using additive Gaussian blobs.

        The algorithm:
        1. Sample ``n_centres = int(density * N)`` random centres within
           the myocardium.
        2. For each centre, add a Gaussian blob
           ``exp(-|r - centre|^2 / (2 * sigma^2))`` to an accumulator field.
        3. Normalise the accumulator to [0, 1].
        4. Threshold at the ``(1 - density)`` percentile to obtain a binary
           mask that is *True* at fibrotic voxels.

        Parameters
        ----------
        rng : np.random.Generator
            Seeded random number generator for reproducibility.
        density : float
            Fibrosis density in (0, 1).  Controls both the number of blob
            centres and the thresholding percentile.
        blob_sigma_voxels : float
            Standard deviation of each Gaussian blob in voxel units.

        Returns
        -------
        np.ndarray
            Shape ``(N, N, N)`` bool — *True* where tissue is fibrotic.
        """
        N = self.grid_size
        n_centres = max(1, int(density * N))

        # Coordinate grids (voxel indices)
        coords = np.mgrid[0:N, 0:N, 0:N].astype(np.float64)  # (3, N, N, N)

        accumulator = np.zeros((N, N, N), dtype=np.float64)

        for _ in range(n_centres):
            cx = rng.integers(0, N)
            cy = rng.integers(0, N)
            cz = rng.integers(0, N)
            dist_sq = (
                (coords[0] - cx) ** 2
                + (coords[1] - cy) ** 2
                + (coords[2] - cz) ** 2
            )
            accumulator += np.exp(-dist_sq / (2.0 * blob_sigma_voxels**2))

        # Normalise to [0, 1]
        a_min, a_max = accumulator.min(), accumulator.max()
        if a_max - a_min > 0:
            accumulator = (accumulator - a_min) / (a_max - a_min)
        else:
            accumulator[:] = 0.0

        # Threshold: voxels with normalised value above the (1-density) percentile
        # are marked as fibrotic, so roughly *density* fraction is fibrotic.
        threshold = np.percentile(accumulator, (1.0 - density) * 100.0)
        fibrosis_mask: np.ndarray = accumulator >= threshold

        return fibrosis_mask
