from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from omegaconf import OmegaConf

from bpc_fno.data.synthetic_dataset import SyntheticMIGDataset
from bpc_fno.utils.normalization import Normalizer


def _write_sample(path: Path) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("J_i", data=np.zeros((5, 4, 4, 4, 3), dtype=np.float32))
        f.create_dataset("B_mig", data=np.zeros((5, 16, 3), dtype=np.float32))
        f.create_dataset("B_mig_noisy", data=np.zeros((5, 16, 3), dtype=np.float32))
        f.create_dataset("t_ms", data=np.arange(5, dtype=np.float32))
        f.create_dataset("activation_times_ms", data=np.zeros((4, 4, 4), dtype=np.float32))
        f.create_dataset("stimulus_mask", data=np.zeros((4, 4, 4), dtype=np.uint8))
        f.create_dataset("sensor_positions", data=np.zeros((16, 3), dtype=np.float32))
        geo = f.create_group("geometry")
        geo.create_dataset("sdf", data=np.zeros((4, 4, 4), dtype=np.float32))
        geo.create_dataset("fiber", data=np.zeros((4, 4, 4, 3), dtype=np.float32))
        geo.create_dataset("cell_type_map", data=np.zeros((4, 4, 4), dtype=np.int8))
        geo.create_dataset("fibrosis_mask", data=np.zeros((4, 4, 4), dtype=np.uint8))


def test_dataset_loads_new_schema_optional_metadata(tmp_path: Path) -> None:
    _write_sample(tmp_path / "sample_00000.h5")

    normalizer = Normalizer()
    normalizer.stats = {
        "J_i_mean": [0.0, 0.0, 0.0],
        "J_i_std": [1.0, 1.0, 1.0],
        "B_mean": [0.0, 0.0, 0.0],
        "B_std": [1.0, 1.0, 1.0],
    }
    cfg = OmegaConf.create({
        "model": {"n_output_timesteps": 3},
        "data": {"preload_to_ram": False},
    })

    ds = SyntheticMIGDataset(tmp_path, "train", normalizer, cfg)
    sample = ds[0]

    assert "cell_type_map" in sample
    assert "activation_times_ms" in sample
    assert "stimulus_mask" in sample
    assert "fibrosis_mask" in sample
    assert sample["J_i"].shape == (3, 3, 4, 4, 4)
