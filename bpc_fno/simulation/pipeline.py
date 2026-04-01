"""Orchestrates the full synthetic-data simulation pipeline.

Generates training samples by composing tissue geometry, TT2006 ionic
models (via Myokit), activation-time propagation, and Biot-Savart forward
model.  Results are stored as HDF5 files with a standardised schema.

Physics chain per sample:
  1. VentricularSlab → geometry (SDF, fiber, cell_type, fibrosis)
  2. ConductivityTensor → D_i anisotropic conductivity
  3. TT2006 via Myokit → per-sample AP waveforms V_m(t) for endo/mid/epi
     with unique conductance scales and pacing CL
  4. Activation-time propagation → V_m(x,t) on 3D grid
  5. J_i = -D_i * grad(V_m) at each output timestep
  6. BiotSavartOperator → B_mig(t) at sensor positions
  7. OPMNoiseModel → B_mig_noisy(t)

Myokit compilation strategy:
  Myokit compiles a C extension the first time a model is simulated.
  Subsequent simulations with the SAME model object reuse the compiled
  .pyd (no os.add_dll_directory calls, no WinError 206).
  We cache the base myokit.Model per cell type (3 total).
  TT2006Runner.run_single() calls model.clone() internally to apply
  per-sample parameter modifications without recompiling.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy.interpolate import interp1d

from bpc_fno import __version__ as _bpc_fno_version
from bpc_fno.simulation.forward.biot_savart import BiotSavartOperator
from bpc_fno.simulation.forward.sensor_config import SensorConfig
from bpc_fno.simulation.tissue.conductivity import ConductivityTensor
from bpc_fno.simulation.tissue.geometry import VentricularSlab

logger = logging.getLogger(__name__)


def _build_voxel_centers(grid_size: int, voxel_size_cm: float) -> np.ndarray:
    """Return shape ``(grid_size**3, 3)`` voxel centres in cm."""
    coords_1d = (np.arange(grid_size) + 0.5) * voxel_size_cm
    gx, gy, gz = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)


def _build_tissue_fields(
    slab: VentricularSlab,
    conductivity: ConductivityTensor,
    cell_type_map: np.ndarray,
    ap_waveforms: dict[int, tuple[np.ndarray, np.ndarray]],
    pacing_site: tuple[int, int, int],
    cv_scale: float,
    config: DictConfig,
    dt_ms: float,
    total_ms: float,
    output_stride: int,
) -> dict[str, np.ndarray]:
    """Build V_m and J_i fields from per-sample TT2006 AP waveforms.

    Each cell type gets its own AP waveform (simulated with the sample's
    unique conductance scales and pacing CL).  Activation times are
    computed via anisotropic eikonal propagation from the pacing site.
    """
    N = slab.grid_size
    h = slab.voxel_size_cm

    # Anisotropic conduction velocity
    cv_base = 0.06 * cv_scale  # cm/ms (60 cm/s base along fiber)
    sigma_ratio = float(config.tissue.sigma_il) / float(config.tissue.sigma_it)
    cv_ratio = np.sqrt(sigma_ratio)

    fiber = slab.get_fiber_field()  # (N, N, N, 3)

    # Displacement from pacing site
    coords = np.mgrid[0:N, 0:N, 0:N].astype(np.float64)
    pace_pos = np.array(pacing_site, dtype=np.float64)
    disp = np.stack([
        coords[0] - pace_pos[0],
        coords[1] - pace_pos[1],
        coords[2] - pace_pos[2],
    ], axis=-1) * h  # (N, N, N, 3) in cm

    # Anisotropic effective distance
    fiber_proj = np.sum(disp * fiber, axis=-1)
    cross_proj = np.sqrt(np.maximum(np.sum(disp**2, axis=-1) - fiber_proj**2, 0.0))
    eff_dist = np.sqrt(fiber_proj**2 + (cross_proj * cv_ratio)**2)
    activation_time_ms = eff_dist / cv_base  # (N, N, N)

    # Output timesteps
    n_steps = int(total_ms / dt_ms)
    T_out = n_steps // output_stride
    t_output = np.arange(T_out, dtype=np.float64) * dt_ms * output_stride

    # Build interpolators per cell type
    ap_interps: dict[int, interp1d] = {}
    for ct_idx, (t_ap, v_ap) in ap_waveforms.items():
        ap_interps[ct_idx] = interp1d(
            t_ap, v_ap, kind="linear", bounds_error=False,
            fill_value=(v_ap[0], v_ap[-1]),
        )

    D_tensor = conductivity.get_tensor_field()  # (N, N, N, 3, 3)

    V_snapshots: list[np.ndarray] = []
    J_snapshots: list[np.ndarray] = []

    for t in t_output:
        V_m = np.full((N, N, N), -85.23, dtype=np.float64)
        for ct_idx, interp_fn in ap_interps.items():
            mask = cell_type_map == ct_idx
            V_m[mask] = interp_fn(t - activation_time_ms[mask])

        V_snapshots.append(V_m.astype(np.float32))

        grad_V = np.stack(np.gradient(V_m, h, edge_order=2), axis=-1)
        J_i = -np.einsum("...ab,...b->...a", D_tensor, grad_V)
        J_snapshots.append(J_i.astype(np.float32))

    return {
        "V_m": np.stack(V_snapshots, axis=0),
        "J_i": np.stack(J_snapshots, axis=0),
        "t_ms": t_output,
    }


# Conductance short-name → Myokit qualified variable name mapping
_CONDUCTANCE_MAP: dict[str, str] = {
    "I_Na": "fast_sodium_current.g_Na",
    "I_CaL": "L_type_Ca_current.g_CaL",
    "I_Kr": "rapid_time_dependent_potassium_current.g_Kr",
    "I_Ks": "slow_time_dependent_potassium_current.g_Ks",
}


class SimulationPipeline:
    """End-to-end synthetic data generation pipeline.

    Uses real TT2006 ionic model (via Myokit) with per-sample parameter
    variation.  Compiled Myokit models are cached per cell type to avoid
    recompilation (which triggers WinError 206 on Windows).
    """

    def __init__(self, config: DictConfig) -> None:
        self._config = config
        self._sensor_config: SensorConfig | None = None
        self._biot_savart: BiotSavartOperator | None = None

        # Level 1 cache: compiled Myokit model objects (one per cell type).
        # These are loaded once and reused for ALL samples.
        # TT2006Runner.run_single() calls model.clone() to apply per-sample
        # params without modifying the cached base model.
        self._cellml_loader: Any = None
        self._runner: Any = None  # TT2006Runner instance (reused)

    def _ensure_runner(self) -> None:
        """Lazily create the CellML loader and TT2006 runner."""
        if self._runner is not None:
            return

        from bpc_fno.simulation.ionic.cellml_loader import CellMLLoader
        from bpc_fno.simulation.ionic.tt2006_runner import TT2006Runner

        loader_config = OmegaConf.create({
            "simulation": {"ionic": self._config.ionic}
        })
        self._cellml_loader = CellMLLoader(loader_config)

        # Pre-load all 3 cell type models to warm the compilation cache
        for ct in ("endo", "mid", "epi"):
            self._cellml_loader.get_model(ct)
        logger.info("All 3 TT2006 cell type models loaded and compiled.")

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
        if self._biot_savart is None:
            self._biot_savart = BiotSavartOperator(
                voxel_centers_cm=voxel_centers_cm,
                sensor_positions_cm=sensor_positions_cm,
                voxel_size_cm=float(self._config.simulation.voxel_size_cm),
            )
            cache_dir = Path("data") / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._biot_savart.precompute_lead_field(
                cache_path=cache_dir / "lead_field.npy"
            )
        return self._biot_savart

    # ------------------------------------------------------------------
    # Single-sample generation
    # ------------------------------------------------------------------

    def generate_sample(
        self,
        params: dict[str, Any],
        noise_model: Any,
        sample_idx: int,
    ) -> Path | None:
        """Generate one training sample with per-sample TT2006 parameter variation."""
        try:
            rng = np.random.default_rng(params.get("sample_seed", sample_idx))
            cfg = self._config
            grid_size: int = int(cfg.simulation.grid_size)
            voxel_size_cm: float = float(cfg.simulation.voxel_size_cm)
            dt_ms: float = float(cfg.monodomain.dt_ms)
            total_ms: float = float(cfg.monodomain.total_time_ms)
            output_stride: int = int(cfg.monodomain.output_stride)

            # ---- 1. Tissue geometry ----
            slab = VentricularSlab(
                grid_size=grid_size,
                voxel_size_cm=voxel_size_cm,
                layer_fractions=list(cfg.tissue.layer_fractions),
            )
            sdf = slab.get_sdf()
            fiber = slab.get_fiber_field()
            cell_type_map = slab.get_cell_type_map()

            # ---- 2. Fibrosis ----
            fibrosis_density: float = float(params.get("fibrosis_density", 0.0))
            if fibrosis_density > 0.0:
                fibrosis_mask = slab.add_fibrosis(
                    rng=rng, density=fibrosis_density,
                    blob_sigma_voxels=float(cfg.tissue.fibrosis_blob_sigma_voxels),
                )
            else:
                fibrosis_mask = np.zeros((grid_size,) * 3, dtype=bool)

            # ---- 3. Conductivity tensor ----
            cv_scale = float(params.get("cv_scale", 1.0))
            conductivity = ConductivityTensor(
                sigma_il=float(cfg.tissue.sigma_il) * cv_scale,
                sigma_it=float(cfg.tissue.sigma_it) * cv_scale,
                fiber_field=fiber,
                fibrosis_mask=fibrosis_mask if fibrosis_density > 0 else None,
            )

            # ---- 4. Run TT2006 per cell type with THIS sample's params ----
            self._ensure_runner()

            raw_scales = params.get("conductance_scales", {})
            conductance_scales = {
                _CONDUCTANCE_MAP.get(k, k): v for k, v in raw_scales.items()
            }
            pacing_cl_ms = float(params["pacing_cl_ms"])
            pacing_site = tuple(params.get("pacing_site_voxel", (0, 0, 0)))

            # Build absolute-value params (Ko) for the ionic model.
            # Ko is the extracellular potassium concentration in mM.
            ko_mM = float(params.get("ko_mM", 5.4))
            absolute_params: dict[str, float] = {"Ko": ko_mM}

            t_sim_start = time.monotonic()

            # Run INDEPENDENT Myokit simulation for each cell type present,
            # with this sample's unique conductance_scales, ko_mM, and pacing_cl_ms.
            cell_types_present = {0: "endo", 1: "mid", 2: "epi"}
            ap_waveforms: dict[int, tuple[np.ndarray, np.ndarray]] = {}

            for ct_idx, ct_name in cell_types_present.items():
                if not np.any(cell_type_map == ct_idx):
                    continue
                result = self._runner.run_single(
                    cell_type=ct_name,
                    params=conductance_scales,
                    pacing_cl_ms=pacing_cl_ms,
                    n_beats=1,
                    absolute_params=absolute_params,
                )
                ap_waveforms[ct_idx] = (result["t_ms"], result["V_m"])

            # ---- 5. Build V_m and J_i fields ----
            sim_result = _build_tissue_fields(
                slab=slab,
                conductivity=conductivity,
                cell_type_map=cell_type_map,
                ap_waveforms=ap_waveforms,
                pacing_site=pacing_site,
                cv_scale=cv_scale,
                config=cfg,
                dt_ms=dt_ms,
                total_ms=total_ms,
                output_stride=output_stride,
            )

            t_sim_elapsed = time.monotonic() - t_sim_start

            V_m = sim_result["V_m"]
            J_i = sim_result["J_i"]
            t_ms = sim_result["t_ms"]

            # ---- 6. Sanity checks ----
            V_m_max = float(np.max(V_m))
            V_m_min = float(np.min(V_m))

            if V_m_max < 0.0:
                logger.warning(
                    "Sample %d: max(V_m)=%.1f mV — no activation. Skipping.",
                    sample_idx, V_m_max,
                )
                return None

            if not (np.all(np.isfinite(V_m)) and np.all(np.isfinite(J_i))):
                logger.warning("Sample %d: NaN/Inf. Skipping.", sample_idx)
                return None

            # ---- 7. Forward model ----
            voxel_centers = _build_voxel_centers(grid_size, voxel_size_cm)
            sensor_positions = self.sensor_config.get_virtual_sensor_positions()
            biot_savart = self._get_biot_savart(voxel_centers, sensor_positions)
            B_mig = biot_savart.forward_batch(J_i)

            # ---- 8. Add noise ----
            n_sensors = sensor_positions.shape[0]
            fs_output = 1000.0 / (dt_ms * output_stride)
            T_out = B_mig.shape[0]
            noise_flat = noise_model.sample(
                n_channels=n_sensors * 3, n_timepoints=T_out,
                fs=fs_output, rng=rng,
            )
            noise_3d = noise_flat.T.reshape(T_out, n_sensors, 3)
            B_mig_noisy = B_mig + noise_3d

            max_B = float(np.max(np.abs(B_mig)))
            max_J = float(np.max(np.abs(J_i)))

            # ---- 9. Save to HDF5 ----
            out_dir = Path("data") / "synthetic"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"sample_{sample_idx:05d}.h5"

            with h5py.File(out_path, "w") as f:
                f.create_dataset("J_i", data=J_i.astype(np.float32),
                                 compression="gzip", compression_opts=4)
                f.create_dataset("B_mig", data=B_mig.astype(np.float32),
                                 compression="gzip", compression_opts=4)
                f.create_dataset("B_mig_noisy", data=B_mig_noisy.astype(np.float32),
                                 compression="gzip", compression_opts=4)
                if bool(cfg.simulation.get("save_vm", False)):
                    f.create_dataset("V_m", data=V_m.astype(np.float32),
                                     compression="gzip", compression_opts=4)

                geo = f.create_group("geometry")
                geo.create_dataset("sdf", data=sdf.astype(np.float32))
                geo.create_dataset("fiber", data=fiber.astype(np.float32))
                geo.create_dataset("cell_type", data=cell_type_map)
                geo.create_dataset("fibrosis", data=fibrosis_mask)

                f.create_dataset("sensor_positions", data=sensor_positions.astype(np.float32))
                f.create_dataset("t_ms", data=t_ms.astype(np.float32))

                f.attrs["pacing_site_voxel"] = list(pacing_site)
                f.attrs["cell_type"] = str(params.get("cell_type", "endo"))
                f.attrs["pacing_cl_ms"] = pacing_cl_ms
                f.attrs["cv_scale"] = cv_scale
                f.attrs["fibrosis_density"] = fibrosis_density
                f.attrs["ko_mM"] = float(params.get("ko_mM", 5.4))
                f.attrs["conductance_scales"] = json.dumps(
                    params.get("conductance_scales", {}))
                f.attrs["sample_seed"] = int(params.get("sample_seed", sample_idx))
                f.attrs["bpc_fno_version"] = _bpc_fno_version
                f.attrs["generation_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                f.attrs["sim_time_seconds"] = round(t_sim_elapsed, 1)

            logger.info(
                "Sample %d: |B|=%.2e T, |J|=%.2e, V_m=[%.0f,%.0f], %.1fs",
                sample_idx, max_B, max_J, V_m_min, V_m_max, t_sim_elapsed,
            )
            return out_path

        except Exception:
            logger.exception("Failed to generate sample %d.", sample_idx)
            return None

    # ------------------------------------------------------------------
    # Batch run
    # ------------------------------------------------------------------

    def run(self, noise_model: Any, n_workers: int = 8) -> None:
        """Generate all samples sequentially."""
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(it: Any, **kw: Any) -> Any:
                return it

        cfg = self._config
        n_samples: int = int(cfg.simulation.n_samples)

        logger.info("Starting pipeline: %d samples.", n_samples)
        t0 = time.monotonic()

        rng_master = np.random.default_rng(42)
        params_list: list[dict[str, Any]] = []
        for i in range(n_samples):
            seed_i = int(rng_master.integers(0, 2**31))
            rng_i = np.random.default_rng(seed_i)
            ionic_cfg = cfg.ionic
            tissue_cfg = cfg.tissue
            cl_lo, cl_hi = ionic_cfg.pacing_cycle_length_ms_range
            cv_lo, cv_hi = tissue_cfg.conduction_velocity_scale_range
            ko_lo, ko_hi = ionic_cfg.ko_range_mM
            gs_lo, gs_hi = ionic_cfg.conductance_scaling_range
            grid_size = int(cfg.simulation.grid_size)

            params_list.append({
                "pacing_site_voxel": tuple(rng_i.integers(0, grid_size, size=3).tolist()),
                "cell_type": str(rng_i.choice(["endo", "mid", "epi"])),
                "pacing_cl_ms": float(rng_i.uniform(cl_lo, cl_hi)),
                "cv_scale": float(rng_i.uniform(cv_lo, cv_hi)),
                "fibrosis_density": float(rng_i.uniform(0.0, tissue_cfg.fibrosis_max_density)),
                "ko_mM": float(rng_i.uniform(ko_lo, ko_hi)),
                "conductance_scales": {
                    "I_Na": float(rng_i.uniform(gs_lo, gs_hi)),
                    "I_CaL": float(rng_i.uniform(gs_lo, gs_hi)),
                    "I_Kr": float(rng_i.uniform(gs_lo, gs_hi)),
                    "I_Ks": float(rng_i.uniform(gs_lo, gs_hi)),
                },
                "sample_seed": seed_i,
            })

        n_pass = 0
        n_fail = 0
        results: list[Path | None] = []

        for i, p in tqdm(enumerate(params_list), total=n_samples, desc="Generating"):
            path = self.generate_sample(p, noise_model, i)
            results.append(path)
            if path is not None:
                n_pass += 1
            else:
                n_fail += 1

        elapsed = time.monotonic() - t0

        manifest_path = Path("data") / "synthetic" / "MANIFEST.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as fp:
            json.dump({
                "n_samples_requested": n_samples,
                "n_pass": n_pass, "n_fail": n_fail,
                "runtime_seconds": round(elapsed, 2),
                "bpc_fno_version": _bpc_fno_version,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "samples": [str(p) if p is not None else None for p in results],
            }, fp, indent=2)

        logger.info(
            "Pipeline complete: %d/%d passed in %.1f s.",
            n_pass, n_samples, elapsed,
        )
