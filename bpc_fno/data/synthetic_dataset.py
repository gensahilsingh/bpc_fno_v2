"""PyTorch Dataset backed by HDF5 files for synthetic MIG data.

Each sample is stored as a separate ``sample_*.h5`` file.  Samples are
deterministically assigned to train / val / test splits based on index
modulo 10.

HDF5 schema per file:
  J_i:           (T, N, N, N, 3)  float32
  B_mig:         (T, Ns, 3)       float32   <- clean
  B_mig_noisy:   (T, Ns, 3)       float32   <- with noise
  V_m:           (T, N, N, N)     float32    (optional)
  geometry/sdf:  (N, N, N)        float32
  geometry/fiber:(N, N, N, 3)     float32
  sensor_positions: (Ns, 3)       float32
  t_ms:          (T,)             float32

For each access, we select the PEAK ACTIVATION timestep (argmax of
sum(|J_i|) over spatial dims) and return a single-timestep slice with
proper normalization.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from bpc_fno.utils.normalization import Normalizer

logger = logging.getLogger(__name__)

_SPLIT_MAP: dict[str, set[int]] = {
    "train": {0, 1, 2, 3, 4, 5, 6, 7},
    "val": {8},
    "test": {9},
}


class SyntheticMIGDataset(Dataset):
    """PyTorch Dataset over synthetic MIG HDF5 samples.

    For each access, the peak-activation timestep is selected from the
    time series (argmax of total |J_i| over spatial dims).  This ensures
    the model always sees the most informative snapshot.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        normalizer: Normalizer,
        config: DictConfig,
    ) -> None:
        if split not in _SPLIT_MAP:
            raise ValueError(
                f"Invalid split '{split}'. Must be one of {list(_SPLIT_MAP.keys())}."
            )

        self.data_dir = Path(data_dir)
        self.split = split
        self.normalizer = normalizer
        self.config = config

        all_h5 = sorted(self.data_dir.glob("sample_*.h5"))
        self._files: list[Path] = []
        self._sample_ids: list[int] = []

        for h5_path in all_h5:
            stem = h5_path.stem
            try:
                idx = int(stem.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            if idx % 10 in _SPLIT_MAP[split]:
                self._files.append(h5_path)
                self._sample_ids.append(idx)

        logger.info(
            "SyntheticMIGDataset [%s]: %d samples from %s",
            split, len(self._files), self.data_dir,
        )

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        h5_path = self._files[idx]
        sample_id = self._sample_ids[idx]

        with h5py.File(h5_path, "r") as f:
            # ----- Select PEAK ACTIVATION timestep -----
            # Compute sum(|J_i|) over spatial+component dims for each timestep
            # J_i shape: (T, N, N, N, 3)
            j_all = np.asarray(f["J_i"], dtype=np.float32)  # (T, N, N, N, 3)
            # sum of absolute values over spatial (N,N,N) and component (3) dims
            activity = np.abs(j_all).sum(axis=(1, 2, 3, 4))  # (T,)
            t_idx = int(np.argmax(activity))

            # ----- Load single timestep -----
            # J_i: (N, N, N, 3) -> (3, N, N, N)
            j_i = torch.from_numpy(j_all[t_idx]).permute(3, 0, 1, 2)  # (3, N, N, N)

            # B_mig (clean): (Ns, 3) -> flatten to (Ns*3,)
            b_clean = np.asarray(f["B_mig"][t_idx], dtype=np.float32)
            b_true = torch.from_numpy(b_clean.reshape(-1))  # (Ns*3,)

            # B_mig_noisy: (Ns, 3) -> flatten to (Ns*3,)
            b_noisy = np.asarray(f["B_mig_noisy"][t_idx], dtype=np.float32)
            b_noisy_flat = torch.from_numpy(b_noisy.reshape(-1))  # (Ns*3,)

            # ----- Geometry: SDF (N,N,N) + fiber (N,N,N,3) -> (4, N, N, N) -----
            sdf = np.asarray(f["geometry/sdf"], dtype=np.float32)
            fiber = np.asarray(f["geometry/fiber"], dtype=np.float32)
            geometry = np.concatenate(
                [sdf[..., None], fiber], axis=-1
            )  # (N, N, N, 4)
            geometry = torch.from_numpy(geometry).permute(3, 0, 1, 2)  # (4, N, N, N)

        # ----- Normalize SDF: clamp [-5,5] / 5.0 -----
        geometry[0] = torch.clamp(geometry[0], min=-5.0, max=5.0) / 5.0

        # ----- Normalize J_i using active-voxel stats -----
        # J_i is channels-first: (3, N, N, N)
        j_mean = torch.tensor(
            self.normalizer.stats["J_i_mean"], dtype=torch.float32
        ).view(3, 1, 1, 1)
        j_std = torch.tensor(
            self.normalizer.stats["J_i_std"], dtype=torch.float32
        ).view(3, 1, 1, 1)
        j_i = (j_i - j_mean) / j_std

        # ----- Normalize B fields -----
        # B stats are per-component (3,); tile across Ns sensors
        b_comp_mean = torch.tensor(
            self.normalizer.stats["B_mean"], dtype=torch.float32
        )  # (3,)
        b_comp_std = torch.tensor(
            self.normalizer.stats["B_std"], dtype=torch.float32
        )  # (3,)
        n_sensors = b_true.shape[0] // 3
        b_mean_tiled = b_comp_mean.repeat(n_sensors)  # (Ns*3,)
        b_std_tiled = b_comp_std.repeat(n_sensors)    # (Ns*3,)

        b_true = (b_true - b_mean_tiled) / b_std_tiled
        b_noisy_flat = (b_noisy_flat - b_mean_tiled) / b_std_tiled

        return {
            "J_i": j_i,                    # (3, N, N, N)
            "B_true": b_true,               # (Ns*3,) — clean, for loss
            "B_noisy": b_noisy_flat,         # (Ns*3,) — model input
            "geometry": geometry,            # (4, N, N, N)
        }
