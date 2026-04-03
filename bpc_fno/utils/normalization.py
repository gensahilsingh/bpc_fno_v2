"""Per-channel z-score normalization for J_i, B, and geometry tensors."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

logger = logging.getLogger(__name__)


class Normalizer:
    """Per-channel z-score normalizer for BPC-FNO data.

    Statistics are computed over the *training* split only (sample indices
    where ``idx % 10 in {0, 1, ..., 7}``).

    Attributes
    ----------
    stats : dict[str, list[float]]
        Dictionary with keys ``'J_i_mean'``, ``'J_i_std'``,
        ``'B_mean'``, ``'B_std'``, each holding a list of per-channel
        values.
    """

    _EPS: float = 1e-30  # guard against division by zero (must be tiny for Tesla-scale data)

    def __init__(self) -> None:
        self.stats: dict[str, list[float]] = {}

    @staticmethod
    def _sample_id_from_path(path: Path) -> int | None:
        stem = path.stem
        if not stem.startswith("sample_"):
            return None
        try:
            return int(stem.split("_", 1)[1])
        except (IndexError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, data_dir: str | Path) -> None:
        """Compute per-channel mean and std from training-split HDF5 files.

        **J_i statistics** are computed from *active voxels only* (top 5%
        by magnitude at the peak activation timestep).  This avoids the
        problem where ~95% resting-tissue voxels (near zero) dominate
        the statistics and produce a tiny std, causing normalized values
        to explode to [-50, +50].

        **B statistics** use all values (no masking needed).

        The method walks *data_dir* for ``*.h5`` / ``*.hdf5`` files.  A
        sample belongs to the training split when its parsed sample ID from
        ``sample_XXXXX`` satisfies ``sample_id % 10 in {0..7}``.

        Expected HDF5 dataset layout per file:
        * ``J_i``  -- shape ``(T, N, N, N, 3)``
        * ``B_mig`` -- shape ``(T, S, 3)``

        Parameters
        ----------
        data_dir:
            Path to the directory containing HDF5 sample files.
        """
        data_path = Path(data_dir)
        h5_files = sorted(
            list(data_path.glob("*.h5")) + list(data_path.glob("*.hdf5"))
        )
        if not h5_files:
            raise FileNotFoundError(
                f"No HDF5 files found in {data_path}"
            )

        train_files: list[Path] = []
        skipped_files = 0
        for fpath in h5_files:
            sample_id = self._sample_id_from_path(fpath)
            if sample_id is None:
                skipped_files += 1
                logger.warning(
                    "Skipping file without parseable sample ID for normalization split: %s",
                    fpath,
                )
                continue
            if sample_id % 10 in range(8):
                train_files.append(fpath)
        if skipped_files:
            logger.info(
                "Skipped %d file(s) without parseable sample IDs during normalization split selection.",
                skipped_files,
            )
        if not train_files:
            raise RuntimeError(
                f"No training-split sample_XXXXX files found in {data_path}"
            )
        logger.info(
            "Fitting normalizer on %d / %d files (training split).",
            len(train_files),
            len(h5_files),
        )

        # -- J_i statistics (Welford online per-channel, ACTIVE VOXELS ONLY) --
        # We use the peak activation timestep per file (argmax of total
        # |J_i|), then mask to top 5% by magnitude.  This matches what
        # __getitem__ returns and avoids resting-tissue contamination.
        j_n: int = 0
        j_mean = np.zeros(3, dtype=np.float64)
        j_m2 = np.zeros(3, dtype=np.float64)

        # -- B statistics (all values, 10 subsampled timesteps) --
        b_n: int = 0
        b_mean: np.ndarray | None = None
        b_m2: np.ndarray | None = None
        b_channels: int | None = None

        n_time_samples = 10  # for B subsampling

        for file_idx, fpath in enumerate(train_files):
            if file_idx % 200 == 0:
                logger.info(
                    "  Normalizer progress: %d / %d files...",
                    file_idx, len(train_files),
                )
            try:
                hf = h5py.File(fpath, "r")
            except OSError:
                logger.warning("Skipping corrupted file: %s", fpath)
                continue
            with hf:
                # --- J_i (peak timestep, active voxels only) ---
                if "J_i" in hf:
                    ds = hf["J_i"]
                    j_all = np.asarray(ds, dtype=np.float64)  # (T, N, N, N, 3)

                    # Select peak activation timestep
                    activity = np.abs(j_all).sum(axis=(1, 2, 3, 4))  # (T,)
                    t_peak = int(np.argmax(activity))
                    j_peak = j_all[t_peak]  # (N, N, N, 3)

                    # Compute magnitude per voxel: (N, N, N)
                    j_mag = np.linalg.norm(j_peak, axis=-1)

                    # Find 95th percentile threshold (top 5% captures
                    # the true wavefront without resting-tissue dilution)
                    threshold = np.percentile(j_mag, 95)

                    # Mask: active voxels = top 5% by magnitude
                    active_mask = j_mag > threshold  # (N, N, N)

                    # Collect J values at masked positions
                    active_j = j_peak[active_mask]  # (n_active, 3)

                    if active_j.shape[0] > 0:
                        for c in range(3):
                            col = active_j[:, c]
                            count = col.shape[0]
                            batch_mean = col.mean()
                            batch_var = col.var()
                            new_n = j_n + count
                            delta = batch_mean - j_mean[c]
                            j_mean[c] += delta * count / new_n
                            j_m2[c] += (
                                batch_var * count
                                + delta ** 2 * j_n * count / new_n
                            )
                        j_n += active_j.shape[0]

                # --- B_mig (all values) ---
                if "B_mig" in hf:
                    ds_b = hf["B_mig"]
                    T_total_b = ds_b.shape[0]
                    if T_total_b <= n_time_samples:
                        b_data = np.asarray(ds_b, dtype=np.float64)
                    else:
                        t_indices_b = np.unique(
                            np.linspace(0, T_total_b - 1, n_time_samples, dtype=int)
                        )
                        b_data = np.asarray(ds_b[t_indices_b], dtype=np.float64)
                    b_flat = b_data.reshape(-1, b_data.shape[-1])
                    n_ch = b_flat.shape[1]

                    if b_mean is None:
                        b_channels = n_ch
                        b_mean = np.zeros(n_ch, dtype=np.float64)
                        b_m2 = np.zeros(n_ch, dtype=np.float64)

                    if n_ch != b_channels:
                        raise ValueError(
                            f"Inconsistent B channel count: expected "
                            f"{b_channels}, got {n_ch} in {fpath}"
                        )

                    for c in range(n_ch):
                        col = b_flat[:, c]
                        count = col.shape[0]
                        batch_mean = col.mean()
                        batch_var = col.var()
                        new_n = b_n + count
                        delta = batch_mean - b_mean[c]
                        b_mean[c] += delta * count / new_n
                        b_m2[c] += (
                            batch_var * count
                            + delta ** 2 * b_n * count / new_n
                        )
                    b_n += b_flat.shape[0]

        # Finalise
        if j_n == 0:
            raise RuntimeError("No J_i data found in training files.")

        j_std_raw = np.sqrt(j_m2 / j_n)

        # J_i is a 3D vector field: components with smaller dynamic range
        # (e.g. J_x) should still use the same normalisation scale so that
        # the model treats the vector uniformly.  We enforce a floor equal
        # to the largest per-component std.
        j_std_floor = j_std_raw.max()
        j_std = np.maximum(j_std_raw, j_std_floor)
        j_std = np.where(j_std < self._EPS, 1.0, j_std)

        self.stats["J_i_mean"] = j_mean.tolist()
        self.stats["J_i_std"] = j_std.tolist()

        logger.info(
            "J_i active-voxel stats: mean=%s, raw_std=%s, shared_std=%s "
            "(from %d active voxels at peak timestep)",
            [f"{v:.6e}" for v in j_mean],
            [f"{v:.6e}" for v in j_std_raw],
            [f"{v:.6e}" for v in j_std],
            j_n,
        )

        if b_n > 0 and b_mean is not None and b_m2 is not None:
            b_std = np.sqrt(b_m2 / b_n)
            b_std = np.where(b_std < self._EPS, 1.0, b_std)
            self.stats["B_mean"] = b_mean.tolist()
            self.stats["B_std"] = b_std.tolist()
        else:
            logger.warning("No B_mig data found; B statistics not computed.")

        logger.info("Normalizer fitted. J_i channels=%d, B channels=%s",
                     len(self.stats["J_i_mean"]),
                     len(self.stats.get("B_mean", [])) or "N/A")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_tensor(self, key: str, device: torch.device | None = None) -> torch.Tensor:
        if key not in self.stats:
            raise RuntimeError(
                f"Statistics key '{key}' not available. "
                "Call fit() or load() first."
            )
        t = torch.tensor(self.stats[key], dtype=torch.float32)
        if device is not None:
            t = t.to(device)
        return t

    # ------------------------------------------------------------------
    # J_i normalisation
    # ------------------------------------------------------------------

    def normalize_J_i(self, J_i: torch.Tensor) -> torch.Tensor:
        """Per-channel z-score normalisation of ``J_i``.

        Handles both channels-last ``(... x C)`` and channels-first
        ``(C, ...)`` layouts by reshaping the stats to broadcast correctly.
        """
        mean = self._to_tensor("J_i_mean", device=J_i.device)
        std = self._to_tensor("J_i_std", device=J_i.device)
        # Reshape for broadcasting: if J_i is (C, N, N, N), reshape to (C, 1, 1, 1)
        if J_i.ndim >= 4 and J_i.shape[0] == len(mean):
            shape = [len(mean)] + [1] * (J_i.ndim - 1)
            mean = mean.view(shape)
            std = std.view(shape)
        return (J_i - mean) / std

    def denormalize_J_i(self, J_i: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`normalize_J_i`."""
        mean = self._to_tensor("J_i_mean", device=J_i.device)
        std = self._to_tensor("J_i_std", device=J_i.device)
        if J_i.ndim >= 4 and J_i.shape[0] == len(mean):
            shape = [len(mean)] + [1] * (J_i.ndim - 1)
            mean = mean.view(shape)
            std = std.view(shape)
        return J_i * std + mean

    # ------------------------------------------------------------------
    # B normalisation
    # ------------------------------------------------------------------

    def normalize_B(self, B: torch.Tensor) -> torch.Tensor:
        """Per-component z-score normalisation of ``B`` (magnetic field).

        Handles flat ``(Ns*3,)`` layout by tiling 3-component stats across
        sensors, as well as ``(... x 3)`` layout.
        """
        mean = self._to_tensor("B_mean", device=B.device)  # (3,)
        std = self._to_tensor("B_std", device=B.device)    # (3,)
        # If B is flat (Ns*3,) or (batch, Ns*3), tile stats across sensors
        n_components = len(mean)
        if B.shape[-1] > n_components and B.shape[-1] % n_components == 0:
            n_sensors = B.shape[-1] // n_components
            mean = mean.repeat(n_sensors)   # (Ns*3,)
            std = std.repeat(n_sensors)
        return (B - mean) / std

    def denormalize_B(self, B: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`normalize_B`."""
        mean = self._to_tensor("B_mean", device=B.device)
        std = self._to_tensor("B_std", device=B.device)
        n_components = len(mean)
        if B.shape[-1] > n_components and B.shape[-1] % n_components == 0:
            n_sensors = B.shape[-1] // n_components
            mean = mean.repeat(n_sensors)
            std = std.repeat(n_sensors)
        return B * std + mean

    # ------------------------------------------------------------------
    # Geometry normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_geometry(geometry: torch.Tensor) -> torch.Tensor:
        """Normalise a geometry tensor.

        The first channel (SDF) is clipped to [-5, 5] and divided by 5.
        Fiber components are left unchanged.

        Handles both channels-first ``(C, ...)`` and channels-last
        ``(..., C)`` layouts.
        """
        out = geometry.clone()
        # Channels-first: (4, N, N, N) or (B, 4, N, N, N)
        if out.ndim >= 4 and out.shape[-4] == 4:
            out[..., 0, :, :, :] = torch.clamp(
                out[..., 0, :, :, :], min=-5.0, max=5.0
            ) / 5.0
        else:
            # Channels-last fallback
            out[..., 0] = torch.clamp(out[..., 0], min=-5.0, max=5.0) / 5.0
        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save normalisation statistics to a JSON file.

        Parameters
        ----------
        path:
            Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2)
        logger.info("Normalizer stats saved to %s", path)

    def load(self, path: str | Path) -> None:
        """Load normalisation statistics from a JSON file.

        Parameters
        ----------
        path:
            Source file path.
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            self.stats = json.load(f)
        logger.info("Normalizer stats loaded from %s", path)
