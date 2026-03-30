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

For training, we select a single random timestep (during the active
wavefront period) from each sample per access.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import h5py
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

_SPLIT_MAP: dict[str, set[int]] = {
    "train": {0, 1, 2, 3, 4, 5, 6, 7},
    "val": {8},
    "test": {9},
}


@runtime_checkable
class Normalizer(Protocol):
    def normalize(self, key: str, tensor: torch.Tensor) -> torch.Tensor: ...
    def denormalize(self, key: str, tensor: torch.Tensor) -> torch.Tensor: ...


class SyntheticMIGDataset(Dataset):
    """PyTorch Dataset over synthetic MIG HDF5 samples.

    For each access, a single timestep with significant activity is
    selected from the time series.  This provides data augmentation
    (different timesteps on different epochs) while keeping memory
    manageable.
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
            T = f["J_i"].shape[0]

            # Pick a timestep with activity (middle 60% of the time series
            # is most likely to contain wavefront activity)
            t_lo = max(int(T * 0.1), 0)
            t_hi = max(int(T * 0.7), t_lo + 1)
            t_hi = min(t_hi, T)  # clamp to available timesteps
            if self.split == "train":
                t_idx = np.random.randint(t_lo, t_hi)
            else:
                # Deterministic for val/test: pick the midpoint
                t_idx = (t_lo + t_hi) // 2

            # J_i: (T, N, N, N, 3) -> pick timestep -> (N, N, N, 3) -> (3, N, N, N)
            j_i = np.asarray(f["J_i"][t_idx], dtype=np.float32)
            j_i = torch.from_numpy(j_i).permute(3, 0, 1, 2)  # (3, N, N, N)

            # B_mig_noisy: (T, Ns, 3) -> flatten to (Ns*3,) per timestep
            # We take a window of timesteps around t_idx for the B signal
            b_noisy = np.asarray(f["B_mig_noisy"][t_idx], dtype=np.float32)
            b_noisy_flat = torch.from_numpy(b_noisy.reshape(-1))  # (Ns*3,)

            b_clean = np.asarray(f["B_mig"][t_idx], dtype=np.float32)
            b_clean_flat = torch.from_numpy(b_clean.reshape(-1))  # (Ns*3,)

            # Geometry: SDF (N,N,N) + fiber (N,N,N,3) -> (4, N, N, N)
            sdf = np.asarray(f["geometry/sdf"], dtype=np.float32)
            fiber = np.asarray(f["geometry/fiber"], dtype=np.float32)
            geometry = np.concatenate(
                [sdf[..., None], fiber], axis=-1
            )  # (N, N, N, 4)
            geometry = torch.from_numpy(geometry).permute(3, 0, 1, 2)  # (4, N, N, N)

            sensor_pos = torch.from_numpy(
                np.asarray(f["sensor_positions"], dtype=np.float32)
            )

        # Normalize
        j_i = self.normalizer.normalize("J_i", j_i)
        b_noisy_flat = self.normalizer.normalize("B_mig", b_noisy_flat)
        b_clean_flat = self.normalizer.normalize("B_mig_clean", b_clean_flat)
        geometry = self.normalizer.normalize("geometry", geometry)

        return {
            "J_i": j_i,                    # (3, N, N, N)
            "B_mig": b_noisy_flat,          # (Ns*3,) — model input
            "B_mig_clean": b_clean_flat,    # (Ns*3,) — for loss
            "geometry": geometry,            # (4, N, N, N)
            "sensor_pos": sensor_pos,        # (Ns, 3)
            "sample_id": sample_id,
        }
