"""Backend-aware synthetic data generation pipeline."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from omegaconf import DictConfig, OmegaConf

from bpc_fno import __version__ as _bpc_fno_version
from bpc_fno.simulation.backends import BACKEND_REGISTRY
from bpc_fno.simulation.backends.base import SimulationContext, SimulationResult
from bpc_fno.simulation.forward.biot_savart import BiotSavartOperator
from bpc_fno.simulation.forward.sensor_config import SensorConfig
from bpc_fno.simulation.grid import build_output_times, build_voxel_centers
from bpc_fno.simulation.tissue.conductivity import ConductivityTensor
from bpc_fno.simulation.tissue.geometry import VentricularSlab

logger = logging.getLogger(__name__)


_CONDUCTANCE_MAP: dict[str, str] = {
    "I_Na": "fast_sodium_current.g_Na",
    "I_CaL": "L_type_Ca_current.g_CaL",
    "I_Kr": "rapid_time_dependent_potassium_current.g_Kr",
    "I_Ks": "slow_time_dependent_potassium_current.g_Ks",
}


class SimulationPipeline:
    """End-to-end synthetic data generation pipeline."""

    def __init__(self, config: DictConfig) -> None:
        self._config = config
        self._sensor_config: SensorConfig | None = None
        self._biot_savart_cache: dict[str, BiotSavartOperator] = {}
        self._cellml_loader: Any = None
        self._runner: Any = None

        backend_name = self._resolve_backend_name()
        backend_cls = BACKEND_REGISTRY[backend_name]
        self._backend = backend_cls()
        logger.info("Simulation backend resolved to '%s'.", backend_name)

    def _resolve_backend_name(self) -> str:
        sim_cfg = self._config.simulation
        backend_name = str(sim_cfg.get("backend", "auto"))
        pipeline_name = str(sim_cfg.get("pipeline", "eikonal"))

        if backend_name != "auto":
            if backend_name not in BACKEND_REGISTRY:
                raise KeyError(f"Unknown backend '{backend_name}'.")
            return backend_name

        if pipeline_name == "eikonal":
            return "eikonal"
        if os.name == "nt":
            return "windows_hybrid"
        return "opencarp"

    def _ensure_runner(self) -> None:
        if self._runner is not None:
            return

        from bpc_fno.simulation.ionic.cellml_loader import CellMLLoader
        from bpc_fno.simulation.ionic.tt2006_runner import TT2006Runner

        loader_config = OmegaConf.create({
            "simulation": {"ionic": self._config.ionic}
        })
        self._cellml_loader = CellMLLoader(loader_config)
        for ct in ("endo", "mid", "epi"):
            self._cellml_loader.get_model(ct)
        self._runner = TT2006Runner(self._cellml_loader, loader_config)

    @property
    def sensor_config(self) -> SensorConfig:
        if self._sensor_config is None:
            self._sensor_config = SensorConfig(self._config)
        return self._sensor_config

    def _get_biot_savart(
        self,
        voxel_centers_cm: np.ndarray,
        sensor_positions_cm: np.ndarray,
    ) -> BiotSavartOperator:
        key = (
            f"{voxel_centers_cm.shape[0]}_"
            f"{sensor_positions_cm.shape[0]}_"
            f"{float(self._config.simulation.voxel_size_cm):.4f}"
        )
        if key not in self._biot_savart_cache:
            op = BiotSavartOperator(
                voxel_centers_cm=voxel_centers_cm,
                sensor_positions_cm=sensor_positions_cm,
                voxel_size_cm=float(self._config.simulation.voxel_size_cm),
            )
            cache_dir = Path("data") / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            op.precompute_lead_field(
                cache_path=cache_dir / f"lead_field_{key}.npy"
            )
            self._biot_savart_cache[key] = op
        return self._biot_savart_cache[key]

    def build_sample_params(
        self,
        sample_idx: int,
        seed_offset: int = 0,
    ) -> dict[str, Any]:
        """Build deterministic sample parameters for a global sample index."""
        sim_cfg = self._config.simulation
        ionic_cfg = self._config.ionic
        tissue_cfg = self._config.tissue
        slab = VentricularSlab(
            grid_size=sim_cfg.get("grid_shape", sim_cfg.get("grid_size", 32)),
            voxel_size_cm=float(sim_cfg.voxel_size_cm),
            layer_fractions=list(tissue_cfg.layer_fractions),
        )
        ss = np.random.SeedSequence(
            [int(sim_cfg.get("master_seed", 42)), int(seed_offset), sample_idx]
        )
        rng = np.random.default_rng(ss)

        cl_lo, cl_hi = ionic_cfg.pacing_cycle_length_ms_range
        cv_lo, cv_hi = tissue_cfg.conduction_velocity_scale_range
        ko_lo, ko_hi = ionic_cfg.ko_range_mM
        gs_lo, gs_hi = ionic_cfg.conductance_scaling_range

        pacing_site = tuple(
            int(rng.integers(0, dim)) for dim in slab.grid_shape
        )
        sample_seed = int(ss.generate_state(1, dtype=np.uint32)[0])

        return {
            "pacing_site_voxel": pacing_site,
            "pacing_cl_ms": float(rng.uniform(cl_lo, cl_hi)),
            "cv_scale": float(rng.uniform(cv_lo, cv_hi)),
            "fibrosis_density": float(
                rng.uniform(0.0, tissue_cfg.fibrosis_max_density)
            ),
            "ko_mM": float(rng.uniform(ko_lo, ko_hi)),
            "conductance_scales": {
                "I_Na": float(rng.uniform(gs_lo, gs_hi)),
                "I_CaL": float(rng.uniform(gs_lo, gs_hi)),
                "I_Kr": float(rng.uniform(gs_lo, gs_hi)),
                "I_Ks": float(rng.uniform(gs_lo, gs_hi)),
            },
            "sample_seed": sample_seed,
        }

    def select_sample_ids(
        self,
        sample_start: int = 0,
        sample_count: int | None = None,
        shard_id: int = 0,
        num_shards: int = 1,
    ) -> list[int]:
        """Return the selected global sample IDs for the current run."""
        if int(num_shards) <= 0:
            raise ValueError("num_shards must be >= 1")
        if not 0 <= int(shard_id) < int(num_shards):
            raise ValueError(
                f"shard_id must satisfy 0 <= shard_id < num_shards; "
                f"got shard_id={shard_id}, num_shards={num_shards}"
            )

        total_samples = int(self._config.simulation.n_samples)
        start = int(sample_start)
        count = int(sample_count) if sample_count is not None else total_samples
        stop = min(start + count, total_samples)
        return [
            idx for idx in range(start, stop)
            if idx % int(num_shards) == int(shard_id)
        ]

    def _build_context(self, params: dict[str, Any]) -> SimulationContext:
        cfg = self._config
        sim_cfg = cfg.simulation
        tissue_cfg = cfg.tissue
        slab = VentricularSlab(
            grid_size=sim_cfg.get("grid_shape", sim_cfg.get("grid_size", 32)),
            voxel_size_cm=float(sim_cfg.voxel_size_cm),
            layer_fractions=list(tissue_cfg.layer_fractions),
        )
        sdf = slab.get_sdf()
        fiber = slab.get_fiber_field()
        cell_type_map = slab.get_cell_type_map()

        rng = np.random.default_rng(int(params["sample_seed"]))
        fibrosis_density = float(params.get("fibrosis_density", 0.0))
        if fibrosis_density > 0.0:
            fibrosis_mask = slab.add_fibrosis(
                rng=rng,
                density=fibrosis_density,
                blob_sigma_voxels=float(tissue_cfg.fibrosis_blob_sigma_voxels),
            )
        else:
            fibrosis_mask = np.zeros(slab.grid_shape, dtype=bool)

        conductivity = ConductivityTensor(
            sigma_il=float(tissue_cfg.sigma_il),
            sigma_it=float(tissue_cfg.sigma_it),
            fiber_field=fiber,
            fibrosis_mask=fibrosis_mask,
        )

        stored_timesteps = int(sim_cfg.stored_timesteps)
        output_times_ms = build_output_times(
            total_time_ms=float(cfg.monodomain.total_time_ms),
            n_timesteps=stored_timesteps,
        )

        return SimulationContext(
            config=cfg,
            slab=slab,
            conductivity=conductivity,
            sdf=sdf,
            fiber=fiber,
            cell_type_map=cell_type_map,
            fibrosis_mask=fibrosis_mask,
            pacing_site_voxel=tuple(params["pacing_site_voxel"]),
            params=params,
            output_times_ms=output_times_ms,
            save_vm=bool(sim_cfg.get("save_vm", False)),
        )

    def _build_lookup_waveforms(
        self,
        context: SimulationContext,
    ) -> dict[int, dict[str, np.ndarray]]:
        self._ensure_runner()

        raw_scales = dict(context.params.get("conductance_scales", {}))
        conductance_scales = {
            _CONDUCTANCE_MAP.get(k, k): v for k, v in raw_scales.items()
        }
        ko_mM = float(context.params.get("ko_mM", 5.4))
        pacing_cl_ms = float(context.params["pacing_cl_ms"])

        waveforms: dict[int, dict[str, np.ndarray]] = {}
        for cell_type_idx, cell_type_name in {0: "endo", 1: "mid", 2: "epi"}.items():
            if not np.any(context.cell_type_map == cell_type_idx):
                continue
            result = self._runner.run_single(
                cell_type=cell_type_name,
                params=conductance_scales,
                pacing_cl_ms=pacing_cl_ms,
                n_beats=1,
                absolute_params={"Ko": ko_mM},
            )
            waveforms[cell_type_idx] = result
        return waveforms

    def _save_sample(
        self,
        out_path: Path,
        context: SimulationContext,
        sim_result: SimulationResult,
        B_mig: np.ndarray,
        B_mig_noisy: np.ndarray,
        sim_time_seconds: float,
    ) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.parent / (
            f".{out_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        try:
            with h5py.File(tmp_path, "w") as f:
                f.create_dataset(
                    "J_i",
                    data=sim_result.J_i.astype(np.float32),
                    compression="gzip",
                    compression_opts=4,
                )
                f.create_dataset(
                    "B_mig",
                    data=B_mig.astype(np.float32),
                    compression="gzip",
                    compression_opts=4,
                )
                f.create_dataset(
                    "B_mig_noisy",
                    data=B_mig_noisy.astype(np.float32),
                    compression="gzip",
                    compression_opts=4,
                )
                if context.save_vm and sim_result.V_m is not None:
                    f.create_dataset(
                        "V_m",
                        data=sim_result.V_m.astype(np.float32),
                        compression="gzip",
                        compression_opts=4,
                    )

                geo = f.create_group("geometry")
                geo.create_dataset("sdf", data=context.sdf.astype(np.float32))
                geo.create_dataset("fiber", data=context.fiber.astype(np.float32))
                geo.create_dataset(
                    "cell_type_map", data=context.cell_type_map.astype(np.int8)
                )
                geo.create_dataset(
                    "fibrosis_mask", data=context.fibrosis_mask.astype(np.uint8)
                )

                f.create_dataset(
                    "sensor_positions",
                    data=self.sensor_config.get_virtual_sensor_positions().astype(
                        np.float32
                    ),
                )
                f.create_dataset("t_ms", data=sim_result.t_ms.astype(np.float32))
                f.create_dataset(
                    "activation_times_ms",
                    data=sim_result.activation_times_ms.astype(np.float32),
                    compression="gzip",
                    compression_opts=4,
                )
                f.create_dataset(
                    "stimulus_mask",
                    data=sim_result.stimulus_mask.astype(np.uint8),
                    compression="gzip",
                    compression_opts=4,
                )

                f.attrs["pacing_site_voxel"] = list(context.pacing_site_voxel)
                f.attrs["pacing_cl_ms"] = float(context.params["pacing_cl_ms"])
                f.attrs["cv_scale"] = float(context.params.get("cv_scale", 1.0))
                f.attrs["fibrosis_density"] = float(
                    context.params.get("fibrosis_density", 0.0)
                )
                f.attrs["ko_mM"] = float(context.params.get("ko_mM", 5.4))
                f.attrs["conductance_scales"] = json.dumps(
                    context.params.get("conductance_scales", {})
                )
                f.attrs["sample_seed"] = int(context.params["sample_seed"])
                f.attrs["backend"] = self._backend.name
                f.attrs["sim_time_seconds"] = round(float(sim_time_seconds), 3)
                for key, value in sim_result.metadata.items():
                    if isinstance(
                        value, (str, int, float, bool, np.integer, np.floating)
                    ):
                        f.attrs[key] = value
                f.attrs["grid_shape"] = list(context.slab.grid_shape)
                f.attrs["voxel_size_cm"] = float(context.slab.voxel_size_cm)
                f.attrs["bpc_fno_version"] = _bpc_fno_version
                f.attrs["generation_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            tmp_path.replace(out_path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise

    def generate_sample(
        self,
        params: dict[str, Any],
        noise_model: Any,
        sample_idx: int,
        output_dir: str | Path | None = None,
    ) -> Path | None:
        try:
            context = self._build_context(params)
            ap_waveforms = None
            if self._backend.requires_lookup_ionics:
                ap_waveforms = self._build_lookup_waveforms(context)

            t_sim_start = time.monotonic()
            sim_result = self._backend.simulate(context, ap_waveforms=ap_waveforms)
            t_sim_elapsed = time.monotonic() - t_sim_start

            if not np.all(np.isfinite(sim_result.J_i)):
                logger.warning("Sample %d: non-finite J_i. Skipping.", sample_idx)
                return None
            if sim_result.V_m is not None and not np.all(np.isfinite(sim_result.V_m)):
                logger.warning("Sample %d: non-finite V_m. Skipping.", sample_idx)
                return None

            voxel_centers = build_voxel_centers(
                context.slab.grid_shape, context.slab.voxel_size_cm
            )
            sensor_positions = self.sensor_config.get_virtual_sensor_positions()
            biot_savart = self._get_biot_savart(voxel_centers, sensor_positions)
            B_mig = biot_savart.forward_batch(sim_result.J_i)

            if not np.all(np.isfinite(B_mig)) or np.max(np.abs(B_mig)) == 0.0:
                logger.warning("Sample %d: invalid or zero B_mig. Skipping.", sample_idx)
                return None

            T_out, n_sensors, _ = B_mig.shape
            noise_flat = noise_model.sample(
                n_channels=n_sensors * 3,
                n_timepoints=T_out,
                fs=1000.0 / max(float(sim_result.t_ms[1] - sim_result.t_ms[0]), 1e-6),
                rng=np.random.default_rng(int(params["sample_seed"])),
            )
            B_mig_noisy = B_mig + noise_flat.T.reshape(T_out, n_sensors, 3)

            out_dir = Path(output_dir or self._config.simulation.output_dir)
            out_path = out_dir / f"sample_{sample_idx:05d}.h5"
            self._save_sample(
                out_path,
                context,
                sim_result,
                B_mig,
                B_mig_noisy,
                sim_time_seconds=t_sim_elapsed,
            )

            logger.info(
                "Sample %d [%s]: |B|=%.2e T, |J|=%.2e, %.2fs",
                sample_idx,
                self._backend.name,
                float(np.max(np.abs(B_mig))),
                float(np.max(np.abs(sim_result.J_i))),
                t_sim_elapsed,
            )
            return out_path

        except Exception:
            logger.exception("Failed to generate sample %d.", sample_idx)
            return None

    def simulate_fields(
        self,
        params: dict[str, Any],
    ) -> tuple[SimulationContext, SimulationResult]:
        """Run only the tissue-field backend without forward/noise/HDF5 steps."""
        context = self._build_context(params)
        ap_waveforms = None
        if self._backend.requires_lookup_ionics:
            ap_waveforms = self._build_lookup_waveforms(context)
        result = self._backend.simulate(context, ap_waveforms=ap_waveforms)
        return context, result

    def run(
        self,
        noise_model: Any,
        n_workers: int = 1,
        output_dir: str | Path | None = None,
        sample_start: int = 0,
        sample_count: int | None = None,
        shard_id: int = 0,
        num_shards: int = 1,
        seed_offset: int = 0,
    ) -> dict[str, Any]:
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(it: Any, **kw: Any) -> Any:
                return it

        if n_workers != 1:
            logger.info(
                "Generation remains sequential at the Python layer; "
                "scale-out is expected via backend parallelism and sharding."
            )

        cfg = self._config
        total_samples = int(cfg.simulation.n_samples)
        start = int(sample_start)
        count = int(sample_count) if sample_count is not None else total_samples
        stop = min(start + count, total_samples)
        selected_ids = self.select_sample_ids(
            sample_start=sample_start,
            sample_count=sample_count,
            shard_id=shard_id,
            num_shards=num_shards,
        )

        logger.info(
            "Starting generation: %d selected samples (range=%d:%d, shard=%d/%d).",
            len(selected_ids), start, stop, shard_id, num_shards,
        )
        t0 = time.monotonic()

        results: list[Path | None] = []
        n_pass = 0
        n_fail = 0
        out_dir = Path(output_dir or cfg.simulation.output_dir)

        for sample_idx in tqdm(selected_ids, desc="Generating"):
            params = self.build_sample_params(sample_idx, seed_offset=seed_offset)
            path = self.generate_sample(
                params=params,
                noise_model=noise_model,
                sample_idx=sample_idx,
                output_dir=out_dir,
            )
            results.append(path)
            if path is None:
                n_fail += 1
            else:
                n_pass += 1

        elapsed = time.monotonic() - t0
        failed_sample_ids = [
            sample_idx
            for sample_idx, path in zip(selected_ids, results)
            if path is None
        ]
        manifest = {
            "backend": self._backend.name,
            "n_samples_requested": len(selected_ids),
            "selected_ids": selected_ids,
            "failed_sample_ids": failed_sample_ids,
            "sample_start": start,
            "sample_stop": stop,
            "shard_id": int(shard_id),
            "num_shards": int(num_shards),
            "seed_offset": int(seed_offset),
            "n_pass": n_pass,
            "n_fail": n_fail,
            "runtime_seconds": round(elapsed, 2),
            "bpc_fno_version": _bpc_fno_version,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "strict_failures": True,
            "samples": [str(p) if p is not None else None for p in results],
        }
        manifest_path = out_dir / "MANIFEST.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as fp:
            json.dump(manifest, fp, indent=2)

        logger.info(
            "Generation complete: %d/%d passed in %.1fs.",
            n_pass, len(selected_ids), elapsed,
        )
        if n_fail > 0:
            raise RuntimeError(
                f"Synthetic generation failed for {n_fail}/{len(selected_ids)} "
                f"sample(s). See {manifest_path}."
            )
        return manifest
