"""PyTorch Dataset backed by HDF5 files for synthetic MIG data.

Each sample is stored as a separate ``sample_*.h5`` file.  Samples are
deterministically assigned to train / val / test splits based on index
modulo 10.

Full time-series mode: subsamples T timesteps from the stored 100,
returning J_i (3, T, N, N, N) and B (S, T).
"""

from __future__ import annotations

import logging
import time as _time
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

    Returns full time-series tensors subsampled to ``n_output_timesteps``
    from config. Supports optional RAM preloading.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        normalizer: Normalizer,
        config: DictConfig,
    ) -> None:
        if split not in _SPLIT_MAP:
            raise ValueError(f"Invalid split '{split}'.")

        self.data_dir = Path(data_dir)
        self.split = split
        self.normalizer = normalizer
        self.config = config
        self.n_output_timesteps: int = int(
            config.model.get("n_output_timesteps", 10)
        )

        all_h5 = sorted(self.data_dir.glob("sample_*.h5"))
        self._files: list[Path] = []
        self._sample_ids: list[int] = []

        for h5_path in all_h5:
            try:
                idx = int(h5_path.stem.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            if idx % 10 in _SPLIT_MAP[split]:
                self._files.append(h5_path)
                self._sample_ids.append(idx)

        logger.info(
            "SyntheticMIGDataset [%s]: %d samples, T=%d from %s",
            split, len(self._files), self.n_output_timesteps, self.data_dir,
        )

        # Optional RAM preloading
        self._cache: dict[int, dict[str, torch.Tensor]] | None = None
        if getattr(config.data, "preload_to_ram", False):
            self._preload_to_ram()

    def _preload_to_ram(self) -> None:
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(it: Any, **kw: Any) -> Any:
                return it

        est_gb = len(self._files) * 3.0 * self.n_output_timesteps / 1024
        logger.info(
            "Preloading %d samples to RAM (~%.1f GB)...",
            len(self._files), est_gb,
        )
        t0 = _time.monotonic()
        self._cache = {}
        failed = 0
        for i, fp in enumerate(tqdm(self._files, desc=f"Preloading [{self.split}]")):
            try:
                self._cache[i] = self._load_sample(fp)
            except Exception as e:
                logger.warning("Failed to preload %d: %s", i, e)
                failed += 1
        elapsed = _time.monotonic() - t0
        logger.info(
            "Preloaded %d/%d in %.1fs. Failed: %d",
            len(self._cache), len(self._files), elapsed, failed,
        )
        if failed > len(self._files) * 0.05:
            raise RuntimeError(f"Too many preload failures: {failed}/{len(self._files)}")

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self._cache is not None:
            return self._cache[idx]
        return self._load_sample(self._files[idx])

    def _load_sample(self, h5_path: Path) -> dict[str, torch.Tensor]:
        T = self.n_output_timesteps

        with h5py.File(h5_path, "r") as f:
            j_all = np.asarray(f["J_i"], dtype=np.float32)       # (T_stored, N,N,N,3)
            b_all = np.asarray(f["B_mig"], dtype=np.float32)     # (T_stored, Ns, 3)
            if "B_mig_noisy" in f:
                b_noisy = np.asarray(f["B_mig_noisy"], dtype=np.float32)
            else:
                b_noisy = b_all.copy()
            sdf = np.asarray(f["geometry/sdf"], dtype=np.float32)
            fib = np.asarray(f["geometry/fiber"], dtype=np.float32)
            t_ms_arr = np.asarray(f["t_ms"], dtype=np.float32)
            cell_type_map = None
            if "geometry/cell_type_map" in f:
                cell_type_map = np.asarray(
                    f["geometry/cell_type_map"], dtype=np.int8
                )
            elif "geometry/cell_type" in f:
                cell_type_map = np.asarray(
                    f["geometry/cell_type"], dtype=np.int8
                )
            activation_times = None
            if "activation_times_ms" in f:
                activation_times = np.asarray(
                    f["activation_times_ms"], dtype=np.float32
                )
            stimulus_mask = None
            if "stimulus_mask" in f:
                stimulus_mask = np.asarray(f["stimulus_mask"], dtype=np.float32)
            fibrosis_mask = None
            if "geometry/fibrosis_mask" in f:
                fibrosis_mask = np.asarray(
                    f["geometry/fibrosis_mask"], dtype=np.float32
                )
            elif "geometry/fibrosis" in f:
                fibrosis_mask = np.asarray(
                    f["geometry/fibrosis"], dtype=np.float32
                )

        T_stored = j_all.shape[0]

        # Deterministic temporal subsampling
        t_indices = np.linspace(0, T_stored - 1, T, dtype=int)
        j_sub = j_all[t_indices]           # (T, N, N, N, 3)
        b_sub = b_all[t_indices]           # (T, Ns, 3)
        b_noisy_sub = b_noisy[t_indices]   # (T, Ns, 3)
        t_sub = t_ms_arr[t_indices]        # (T,)

        Ns = b_sub.shape[1]

        # J_i: (T, N, N, N, 3) -> (3, T, N, N, N)
        j_i = torch.from_numpy(j_sub).permute(4, 0, 1, 2, 3)

        # B: (T, Ns, 3) -> (Ns*3, T) = (S, T)
        b_true = torch.from_numpy(b_sub).permute(1, 2, 0).reshape(Ns * 3, -1)
        b_obs = torch.from_numpy(b_noisy_sub).permute(1, 2, 0).reshape(Ns * 3, -1)

        # Geometry: (4, N, N, N) — static
        geometry = torch.from_numpy(
            np.concatenate([sdf[..., None], fib], axis=-1)
        ).permute(3, 0, 1, 2)
        geometry[0] = torch.clamp(geometry[0], -5.0, 5.0) / 5.0

        # Normalize J_i: (3, T, N, N, N) — stats shape (3, 1, 1, 1, 1)
        j_mean = torch.tensor(
            self.normalizer.stats["J_i_mean"], dtype=torch.float32
        ).view(3, 1, 1, 1, 1)
        j_std = torch.tensor(
            self.normalizer.stats["J_i_std"], dtype=torch.float32
        ).view(3, 1, 1, 1, 1)
        j_i = (j_i - j_mean) / j_std

        # Normalize B: (S, T) — tile 3-component stats across sensors
        b_comp_mean = torch.tensor(
            self.normalizer.stats["B_mean"], dtype=torch.float32
        )
        b_comp_std = torch.tensor(
            self.normalizer.stats["B_std"], dtype=torch.float32
        )
        b_mean_tiled = b_comp_mean.repeat(Ns).unsqueeze(-1)   # (S, 1)
        b_std_tiled = b_comp_std.repeat(Ns).unsqueeze(-1)     # (S, 1)
        b_true = (b_true - b_mean_tiled) / b_std_tiled
        b_obs = (b_obs - b_mean_tiled) / b_std_tiled

        sample = {
            "J_i": j_i,            # (3, T, N, N, N)
            "B_true": b_true,       # (S, T)
            "B_obs": b_obs,         # (S, T)
            "geometry": geometry,   # (4, N, N, N)
            "t_ms": torch.from_numpy(t_sub),  # (T,)
        }

        if cell_type_map is not None:
            sample["cell_type_map"] = torch.from_numpy(cell_type_map.astype(np.int64))
        if activation_times is not None:
            sample["activation_times_ms"] = torch.from_numpy(activation_times)
        if stimulus_mask is not None:
            sample["stimulus_mask"] = torch.from_numpy(stimulus_mask)
        if fibrosis_mask is not None:
            sample["fibrosis_mask"] = torch.from_numpy(fibrosis_mask)

        return sample
