"""Biot-Savart lead-field matrix and forward magnetic-field computation.

Implements the quasi-static magnetostatic forward problem: given an
intracellular current-density field J_i(r) [uA/cm^2] on a voxelised grid,
compute the magnetic flux density B(r_s) [Tesla] at a set of sensor
positions using a precomputed lead-field matrix.

Physics
-------
    B(r_s) = (mu_0 / 4 pi) * integral_V  J(r') x (r_s - r') / |r_s - r'|^3  dV'

where mu_0 = 4 pi * 1e-7  T m / A.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Permeability of free space  [T m / A].
_MU_0: float = 4.0 * np.pi * 1e-7


class BiotSavartOperator:
    """Precompute and apply the Biot-Savart lead-field matrix.

    Parameters
    ----------
    voxel_centers_cm : np.ndarray
        Shape ``(N_voxels, 3)`` — centre coordinates of every active voxel
        in centimetres.
    sensor_positions_cm : np.ndarray
        Shape ``(N_sensors, 3)`` — sensor positions in centimetres.
    voxel_size_cm : float
        Edge length of each cubic voxel in centimetres.
    """

    def __init__(
        self,
        voxel_centers_cm: np.ndarray,
        sensor_positions_cm: np.ndarray,
        voxel_size_cm: float,
    ) -> None:
        self.voxel_centers_cm = np.asarray(voxel_centers_cm, dtype=np.float64)
        self.sensor_positions_cm = np.asarray(sensor_positions_cm, dtype=np.float64)
        self.voxel_size_cm = float(voxel_size_cm)

        if self.voxel_centers_cm.ndim != 2 or self.voxel_centers_cm.shape[1] != 3:
            raise ValueError(
                f"voxel_centers_cm must have shape (N_voxels, 3), "
                f"got {self.voxel_centers_cm.shape}"
            )
        if self.sensor_positions_cm.ndim != 2 or self.sensor_positions_cm.shape[1] != 3:
            raise ValueError(
                f"sensor_positions_cm must have shape (N_sensors, 3), "
                f"got {self.sensor_positions_cm.shape}"
            )

        self.n_voxels: int = self.voxel_centers_cm.shape[0]
        self.n_sensors: int = self.sensor_positions_cm.shape[0]
        self.lead_field: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Lead-field computation
    # ------------------------------------------------------------------

    def precompute_lead_field(
        self, cache_path: str | Path | None = None
    ) -> None:
        """Compute (or load) the lead-field matrix L.

        The lead-field matrix has shape ``(N_sensors * 3, N_voxels * 3)``
        and dtype float32.  It maps a flattened current-density vector
        ``J_flat`` (in A/m^2) to a flattened B-field vector (in Tesla):

            ``B_flat = L @ J_flat``

        When *cache_path* is provided and the file exists on disk the
        matrix is loaded directly; otherwise it is computed and saved.

        Parameters
        ----------
        cache_path : str | Path | None
            Optional ``.npy`` file for caching the lead-field matrix.
        """
        if cache_path is not None:
            cache_path = Path(cache_path)
            if cache_path.exists():
                logger.info("Loading cached lead-field from %s", cache_path)
                self.lead_field = np.load(cache_path).astype(np.float32)
                return

        logger.info(
            "Computing lead-field matrix: %d sensors x %d voxels ...",
            self.n_sensors,
            self.n_voxels,
        )

        # Displacement vectors  r_s - r_v  in metres.
        # sensor_positions_cm: (Ns, 3)   voxel_centers_cm: (Nv, 3)
        # d: (Ns, Nv, 3)  in metres
        d = (
            self.sensor_positions_cm[:, np.newaxis, :]
            - self.voxel_centers_cm[np.newaxis, :, :]
        ) * 1e-2  # cm -> m

        # Distance |d|, clamped to avoid singularity.
        dist = np.linalg.norm(d, axis=-1)  # (Ns, Nv)
        min_dist = self.voxel_size_cm * 1e-2 / 2.0
        dist = np.maximum(dist, min_dist)

        # Volume element in m^3.
        dV = (self.voxel_size_cm * 1e-2) ** 3

        # Prefactor  (mu_0 / 4 pi) * dV / |d|^3   shape (Ns, Nv)
        prefactor = (_MU_0 / (4.0 * np.pi)) * dV / (dist ** 3)

        # Extract displacement components: (Ns, Nv) each.
        dx = d[..., 0]
        dy = d[..., 1]
        dz = d[..., 2]

        # Build lead-field matrix  L: (Ns*3, Nv*3)  float32.
        # Cross-product  J x d  gives B components:
        #   B_x = J_y * d_z - J_z * d_y
        #   B_y = J_z * d_x - J_x * d_z
        #   B_z = J_x * d_y - J_y * d_x
        #
        # L is sparse-structured but stored dense for simplicity / speed.

        Ns = self.n_sensors
        Nv = self.n_voxels
        L = np.zeros((Ns * 3, Nv * 3), dtype=np.float32)

        # B_x row-block (sensor s, component 0)
        # B_x from J_y:  prefactor * d_z
        L[0::3, 1::3] = (prefactor * dz).astype(np.float32)
        # B_x from J_z: -prefactor * d_y
        L[0::3, 2::3] = (prefactor * (-dy)).astype(np.float32)

        # B_y row-block (sensor s, component 1)
        # B_y from J_x: -prefactor * d_z
        L[1::3, 0::3] = (prefactor * (-dz)).astype(np.float32)
        # B_y from J_z:  prefactor * d_x
        L[1::3, 2::3] = (prefactor * dx).astype(np.float32)

        # B_z row-block (sensor s, component 2)
        # B_z from J_x:  prefactor * d_y
        L[2::3, 0::3] = (prefactor * dy).astype(np.float32)
        # B_z from J_y: -prefactor * d_x
        L[2::3, 1::3] = (prefactor * (-dx)).astype(np.float32)

        self.lead_field = L
        logger.info("Lead-field matrix shape: %s, dtype: %s", L.shape, L.dtype)

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, L)
            logger.info("Saved lead-field cache to %s", cache_path)

    # ------------------------------------------------------------------
    # Forward operators
    # ------------------------------------------------------------------

    def forward(self, J_i_volume: np.ndarray) -> np.ndarray:
        """Compute B-field at all sensors for a single time snapshot.

        Parameters
        ----------
        J_i_volume : np.ndarray
            Shape ``(N, N, N, 3)`` — intracellular current density in
            uA/cm^2.

        Returns
        -------
        np.ndarray
            Shape ``(N_sensors, 3)`` — magnetic flux density in Tesla.

        Raises
        ------
        RuntimeError
            If :meth:`precompute_lead_field` has not been called.
        """
        if self.lead_field is None:
            raise RuntimeError(
                "Lead-field matrix has not been computed.  "
                "Call precompute_lead_field() first."
            )

        # Flatten spatial dimensions: (N*N*N*3,)
        # Unit conversion:  uA/cm^2 -> A/m^2  (factor 1e-2)
        #   1 uA/cm^2 = 1e-6 A / (1e-2 m)^2 = 1e-6 / 1e-4 = 1e-2 A/m^2
        J_flat = J_i_volume.reshape(-1).astype(np.float32) * 1e-2

        B_flat = self.lead_field @ J_flat  # (Ns*3,)
        return B_flat.reshape(self.n_sensors, 3)

    def forward_batch(self, J_i_sequence: np.ndarray) -> np.ndarray:
        """Compute B-field time series for a batch of snapshots.

        Parameters
        ----------
        J_i_sequence : np.ndarray
            Shape ``(T, N, N, N, 3)`` — intracellular current density
            sequence in uA/cm^2.

        Returns
        -------
        np.ndarray
            Shape ``(T, N_sensors, 3)`` — magnetic flux density in Tesla.
        """
        if self.lead_field is None:
            raise RuntimeError(
                "Lead-field matrix has not been computed.  "
                "Call precompute_lead_field() first."
            )

        T = J_i_sequence.shape[0]
        # Flatten spatial dims: (T, N*N*N*3)
        J_flat = J_i_sequence.reshape(T, -1).astype(np.float32) * 1e-2

        # Vectorised matmul:  (T, Nv*3) @ (Nv*3, Ns*3)^T  = (T, Ns*3)
        B_flat = (J_flat @ self.lead_field.T)  # (T, Ns*3)
        return B_flat.reshape(T, self.n_sensors, 3)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_dipole(self) -> float:
        """Validate the lead-field against the analytic magnetic dipole formula.

        Places a unit current dipole (moment ``m = (0, 0, 1)`` A m^2
        equivalent) at the grid centre and compares the B-field from the
        lead-field to the analytic far-field expression:

            B(r) = (mu_0 / 4 pi) * [3 (m . r_hat) r_hat - m] / |r|^3

        Only sensors at distance > ``5 * voxel_size_cm`` are included in
        the comparison.

        Returns
        -------
        float
            Maximum relative error across included sensors.  Should be
            < 0.01 (1 %) for a well-resolved grid.
        """
        if self.lead_field is None:
            raise RuntimeError(
                "Lead-field matrix has not been computed.  "
                "Call precompute_lead_field() first."
            )

        # Determine grid dimensions from voxel count (assume cubic grid).
        N = round(self.n_voxels ** (1.0 / 3.0))
        if N ** 3 != self.n_voxels:
            raise RuntimeError(
                f"validate_dipole requires a cubic grid; n_voxels={self.n_voxels} "
                f"is not a perfect cube."
            )

        # Place a unit z-dipole at the grid centre.
        J_volume = np.zeros((N, N, N, 3), dtype=np.float64)
        mid = N // 2
        dV_cm3 = self.voxel_size_cm ** 3

        # We want the dipole moment  m = p * dV  = 1 A m^2.
        # J is in uA/cm^2, forward() converts to A/m^2 (*1e-2) and
        # lead-field already includes dV in m^3.
        # So the effective moment per voxel is:
        #   m = J_uA_cm2 * 1e-2 [A/m^2] * dV_m3 [m^3]
        # We need m_z = 1 A m^2, so:
        dV_m3 = (self.voxel_size_cm * 1e-2) ** 3
        J_z_value = 1.0 / (1e-2 * dV_m3)  # uA/cm^2
        J_volume[mid, mid, mid, 2] = J_z_value

        B_numerical = self.forward(J_volume)  # (Ns, 3)

        # Analytic dipole field.
        dipole_pos_cm = self.voxel_centers_cm.reshape(N, N, N, 3)[mid, mid, mid]
        m_vec = np.array([0.0, 0.0, 1.0])  # A m^2

        min_dist_cm = 5.0 * self.voxel_size_cm
        B_analytic = np.zeros_like(B_numerical)
        mask = np.ones(self.n_sensors, dtype=bool)

        for s in range(self.n_sensors):
            r_vec = (self.sensor_positions_cm[s] - dipole_pos_cm) * 1e-2  # m
            r_mag = np.linalg.norm(r_vec)
            if r_mag < min_dist_cm * 1e-2:
                mask[s] = False
                continue
            r_hat = r_vec / r_mag
            B_analytic[s] = (
                (_MU_0 / (4.0 * np.pi))
                * (3.0 * np.dot(m_vec, r_hat) * r_hat - m_vec)
                / (r_mag ** 3)
            )

        if not np.any(mask):
            logger.warning(
                "No sensors far enough from dipole for validation "
                "(min_dist=%.3f cm).",
                min_dist_cm,
            )
            return float("inf")

        # Relative error.
        B_num_sel = B_numerical[mask]
        B_ana_sel = B_analytic[mask]
        denom = np.linalg.norm(B_ana_sel, axis=-1, keepdims=True)
        denom = np.maximum(denom, 1e-30)
        rel_err = np.linalg.norm(B_num_sel - B_ana_sel, axis=-1, keepdims=True) / denom
        max_rel_err = float(rel_err.max())

        logger.info(
            "Dipole validation: max relative error = %.6e across %d sensors.",
            max_rel_err,
            int(mask.sum()),
        )
        return max_rel_err
