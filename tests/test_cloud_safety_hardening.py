from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from bpc_fno.data.data_module import BPCFNODataModule
from bpc_fno.simulation.backends.opencarp import (
    _box_from_pacing_site,
    _stimulus_mask_from_box,
)
from bpc_fno.simulation.pipeline import SimulationPipeline
from bpc_fno.utils.normalization import Normalizer


def _write_sample(path: Path, value: float) -> None:
    grid_shape = (4, 4, 4)
    n_timesteps = 5
    n_sensors = 16

    j_i = np.zeros((n_timesteps, *grid_shape, 3), dtype=np.float32)
    j_i[:, 0, 0, 0, :] = value
    b = np.full((n_timesteps, n_sensors, 3), value, dtype=np.float32)

    with h5py.File(path, "w") as f:
        f.create_dataset("J_i", data=j_i)
        f.create_dataset("B_mig", data=b)
        f.create_dataset("B_mig_noisy", data=b)
        f.create_dataset("t_ms", data=np.arange(n_timesteps, dtype=np.float32))
        f.create_dataset(
            "activation_times_ms", data=np.zeros(grid_shape, dtype=np.float32)
        )
        f.create_dataset("stimulus_mask", data=np.zeros(grid_shape, dtype=np.uint8))
        f.create_dataset(
            "sensor_positions", data=np.zeros((n_sensors, 3), dtype=np.float32)
        )
        geo = f.create_group("geometry")
        geo.create_dataset("sdf", data=np.zeros(grid_shape, dtype=np.float32))
        geo.create_dataset(
            "fiber", data=np.zeros((*grid_shape, 3), dtype=np.float32)
        )
        geo.create_dataset("cell_type_map", data=np.zeros(grid_shape, dtype=np.int8))
        geo.create_dataset("fibrosis_mask", data=np.zeros(grid_shape, dtype=np.uint8))


def _minimal_generation_config(output_dir: Path, n_samples: int = 4):
    return OmegaConf.create(
        {
            "simulation": {
                "pipeline": "eikonal",
                "backend": "eikonal",
                "grid_size": 4,
                "voxel_size_cm": 0.1,
                "n_samples": n_samples,
                "output_dir": str(output_dir),
                "stored_timesteps": 5,
                "master_seed": 42,
            },
            "ionic": {
                "pacing_cycle_length_ms_range": [1000.0, 1000.0],
                "ko_range_mM": [5.4, 5.4],
                "conductance_scaling_range": [1.0, 1.0],
            },
            "tissue": {
                "layer_fractions": [0.33, 0.33, 0.34],
                "sigma_il": 3.0e-3,
                "sigma_it": 3.0e-4,
                "conduction_velocity_scale_range": [1.0, 1.0],
                "fibrosis_max_density": 0.0,
                "fibrosis_blob_sigma_voxels": 1.0,
            },
            "monodomain": {
                "total_time_ms": 10.0,
            },
        }
    )


def test_normalizer_uses_sample_id_split_with_missing_ids(tmp_path: Path) -> None:
    _write_sample(tmp_path / "sample_00008.h5", value=9.0)
    _write_sample(tmp_path / "sample_00009.h5", value=10.0)
    _write_sample(tmp_path / "sample_00010.h5", value=11.0)

    normalizer = Normalizer()
    normalizer.fit(tmp_path)

    assert normalizer.stats["B_mean"] == pytest.approx([11.0, 11.0, 11.0])
    assert normalizer.stats["J_i_mean"] == pytest.approx([11.0, 11.0, 11.0])


def test_real_normalizer_supports_test_preloading_without_proxy(tmp_path: Path) -> None:
    _write_sample(tmp_path / "sample_00009.h5", value=1.0)

    normalizer = Normalizer()
    normalizer.stats = {
        "J_i_mean": [0.0, 0.0, 0.0],
        "J_i_std": [1.0, 1.0, 1.0],
        "B_mean": [0.0, 0.0, 0.0],
        "B_std": [1.0, 1.0, 1.0],
    }
    config = OmegaConf.create(
        {
            "model": {"n_output_timesteps": 3},
            "data": {
                "data_dir": str(tmp_path),
                "preload_to_ram": True,
                "batch_size": 1,
                "num_workers": 0,
                "pin_memory": False,
            },
            "training": {"batch_size": 1},
        }
    )

    data_module = BPCFNODataModule(config, normalizer=normalizer)
    data_module.setup(stage="test")
    batch = next(iter(data_module.test_dataloader()))

    assert batch["J_i"].shape == (1, 3, 3, 4, 4, 4)
    assert batch["B_true"].shape == (1, 48, 3)


def test_pipeline_run_writes_manifest_and_raises_on_sample_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "generated"
    config = _minimal_generation_config(output_dir=output_dir, n_samples=2)
    pipeline = SimulationPipeline(config)

    def fake_generate_sample(
        params, noise_model, sample_idx, output_dir=None
    ):  # noqa: ANN001
        target_dir = Path(output_dir or config.simulation.output_dir)
        if sample_idx == 0:
            out_path = target_dir / f"sample_{sample_idx:05d}.h5"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.touch()
            return out_path
        return None

    monkeypatch.setattr(pipeline, "generate_sample", fake_generate_sample)

    with pytest.raises(RuntimeError, match="Synthetic generation failed"):
        pipeline.run(
            noise_model=object(),
            output_dir=output_dir,
            sample_start=0,
            sample_count=2,
            shard_id=0,
            num_shards=1,
            seed_offset=0,
        )

    manifest_path = output_dir / "MANIFEST.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["n_pass"] == 1
    assert manifest["n_fail"] == 1
    assert manifest["failed_sample_ids"] == [1]
    assert manifest["strict_failures"] is True


def test_stimulus_mask_matches_opencarp_box_geometry() -> None:
    p0_um, p1_um = _box_from_pacing_site((2, 2, 2), voxel_size_cm=0.1, radius_cm=0.15)
    mask = _stimulus_mask_from_box(
        grid_shape=(5, 5, 5),
        voxel_size_cm=0.1,
        p0_um=p0_um,
        p1_um=p1_um,
    )

    assert mask[2, 2, 2] == 1
    assert int(mask.sum()) == 27


def test_generate_synthetic_resume_only_regenerates_missing_selected_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.generate_synthetic_resume as resume_script

    output_dir = tmp_path / "resume_out"
    output_dir.mkdir()
    (output_dir / "sample_00000.h5").touch()
    (output_dir / "sample_00002.h5").touch()

    config_path = tmp_path / "resume_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "simulation:",
                "  pipeline: eikonal",
                "  backend: eikonal",
                "  grid_size: 4",
                "  voxel_size_cm: 0.1",
                "  n_samples: 4",
                f"  output_dir: {output_dir.as_posix()}",
                "  stored_timesteps: 5",
                "  master_seed: 42",
                "ionic:",
                "  pacing_cycle_length_ms_range: [1000.0, 1000.0]",
                "  ko_range_mM: [5.4, 5.4]",
                "  conductance_scaling_range: [1.0, 1.0]",
                "tissue:",
                "  layer_fractions: [0.33, 0.33, 0.34]",
                "  sigma_il: 0.003",
                "  sigma_it: 0.0003",
                "  conduction_velocity_scale_range: [1.0, 1.0]",
                "  fibrosis_max_density: 0.0",
                "  fibrosis_blob_sigma_voxels: 1.0",
                "monodomain:",
                "  total_time_ms: 10.0",
            ]
        ),
        encoding="utf-8",
    )
    noise_model_path = tmp_path / "noise_model.json"
    noise_model_path.write_text("{}", encoding="utf-8")

    called_ids: list[int] = []

    def fake_load(self, path):  # noqa: ANN001
        _ = path

    def fake_generate_sample(self, params, noise_model, sample_idx, output_dir=None):  # noqa: ANN001
        _ = params, noise_model
        called_ids.append(sample_idx)
        out_path = Path(output_dir or self._config.simulation.output_dir) / f"sample_{sample_idx:05d}.h5"
        out_path.touch()
        return out_path

    monkeypatch.setattr(resume_script.OPMNoiseModel, "load", fake_load)
    monkeypatch.setattr(resume_script.SimulationPipeline, "generate_sample", fake_generate_sample)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_synthetic_resume.py",
            "--config",
            str(config_path),
            "--noise-model",
            str(noise_model_path),
            "--output-dir",
            str(output_dir),
            "--batch-size",
            "1",
        ],
    )

    resume_script.main()

    assert called_ids == [1, 3]
    resume_manifest = json.loads(
        (output_dir / "RESUME_MANIFEST.json").read_text(encoding="utf-8")
    )
    assert resume_manifest["n_fail"] == 0
    assert resume_manifest["remaining_ids"] == [1, 3]


def test_phase2_training_exits_when_physics_loss_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.train_joint as train_joint

    config_path = tmp_path / "arch.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model:",
                "  n_output_timesteps: 1",
                "training:",
                "  phase2_epochs: 1",
                "  lr_init: 1.0e-3",
                "  lr_final: 1.0e-4",
                "  lr_warmup_steps: 0",
                "  grad_clip_norm: 1.0",
                "  lambda_physics_init: 0.1",
                "  lambda_physics_final: 0.1",
                "  lambda_physics_doubling_epochs: 1",
                "  lambda_kl_init: 0.0",
                "  lambda_consistency_start_epoch: 999",
                "  lambda_consistency: 0.0",
                "  batch_size: 1",
                "data:",
                f"  data_dir: {tmp_path.as_posix()}",
                "  batch_size: 1",
                "  num_workers: 0",
                "  pin_memory: false",
                "simulation:",
                "  voxel_size_cm: 0.1",
            ]
        ),
        encoding="utf-8",
    )

    norm_path = tmp_path / "normalization.json"
    Normalizer().save(norm_path)  # create valid file structure first
    norm_data = {
        "J_i_mean": [0.0, 0.0, 0.0],
        "J_i_std": [1.0, 1.0, 1.0],
        "B_mean": [0.0, 0.0, 0.0],
        "B_std": [1.0, 1.0, 1.0],
    }
    norm_path.write_text(json.dumps(norm_data), encoding="utf-8")

    phase1_path = tmp_path / "phase1_best.pt"
    phase1_path.touch()

    batch = {
        "J_i": torch.ones((1, 3, 1, 2, 2, 2), dtype=torch.float32),
        "geometry": torch.ones((1, 4, 2, 2, 2), dtype=torch.float32),
        "B_obs": torch.ones((1, 6, 1), dtype=torch.float32),
        "B_true": torch.ones((1, 6, 1), dtype=torch.float32),
    }

    class FakeDataModule:
        def __init__(self, config, normalizer):  # noqa: ANN001
            _ = config, normalizer

        def setup(self, stage=None):  # noqa: ANN001
            _ = stage

        def train_dataloader(self):
            return [batch]

        def val_dataloader(self):
            return [batch]

    class FakeModel(torch.nn.Module):
        def __init__(self, config):  # noqa: ANN001
            super().__init__()
            self.p_shared = torch.nn.Parameter(torch.tensor(1.0))
            self.p_forward = torch.nn.Parameter(torch.tensor(1.0))
            self.p_inverse = torch.nn.Parameter(torch.tensor(1.0))
            self.p_decoder = torch.nn.Parameter(torch.tensor(1.0))

        def get_parameter_groups(self):
            return {
                "fno_shared": [self.p_shared],
                "forward_head": [self.p_forward],
                "inverse_head": [self.p_inverse],
                "decoder": [self.p_decoder],
            }

        def forward(self, batch):  # noqa: ANN001
            scale = self.p_shared + self.p_forward + self.p_inverse + self.p_decoder
            mu = torch.zeros((batch["J_i"].shape[0], 2), device=batch["J_i"].device)
            log_var = torch.zeros_like(mu)
            return {
                "J_i_hat": batch["J_i"] * scale,
                "B_pred": batch["B_true"] * scale,
                "mu": mu,
                "log_var": log_var,
            }

        __call__ = torch.nn.Module.__call__

        def to(self, device):  # noqa: ANN001
            return super().to(device)

    class FakeLossManager:
        def __init__(self, config):  # noqa: ANN001
            _ = config

        def physics_residual_loss(self, J_i_hat, voxel_size_cm):  # noqa: ANN001
            _ = J_i_hat, voxel_size_cm
            raise ValueError("boom")

    def fake_load_checkpoint(*args, **kwargs):  # noqa: ANN001
        return {"epoch": 1, "phase": "forward", "metrics": {}, "extra_state": {}}

    monkeypatch.setattr(train_joint, "BPCFNODataModule", FakeDataModule)
    monkeypatch.setattr(train_joint, "BPC_FNO_A", FakeModel)
    monkeypatch.setattr(train_joint, "LossManager", FakeLossManager)
    monkeypatch.setattr(train_joint, "load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(train_joint, "save_checkpoint", lambda **kwargs: None)
    monkeypatch.setattr(train_joint.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_joint.py",
            "--config",
            str(config_path),
            "--phase1-checkpoint",
            str(phase1_path),
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--normalization",
            str(norm_path),
            "--data-dir",
            str(tmp_path),
            "--device",
            "cpu",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        train_joint.main()

    assert exc.value.code == 1
