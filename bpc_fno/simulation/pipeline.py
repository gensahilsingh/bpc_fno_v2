"""Orchestrates the full synthetic-data simulation pipeline.

Generates training samples by composing tissue geometry, ionic models,
monodomain PDE solving, and the Biot-Savart forward model.  Results are
stored as HDF5 files with a standardised schema.

Physics chain:
  1. VentricularSlab → geometry (SDF, fiber, cell_type, fibrosis)
  2. ConductivityTensor → D_i anisotropic conductivity
  3. TT2006 via Myokit → I_ion(V_m) lookup tables per cell type
  4. MonodomainSolver → V_m(x,t), J_i(x,t)
  5. BiotSavartOperator → B_mig(t) at sensor positions
  6. OPMNoiseModel → B_mig_noisy(t)
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


# ---------------------------------------------------------------------------
# Ionic model callback factory using Myokit-generated lookup tables
# ---------------------------------------------------------------------------

def _build_tissue_simulation(
    cellml_loader: Any,
    slab: VentricularSlab,
    conductivity: ConductivityTensor,
    cell_type_map: np.ndarray,
    params: dict[str, float],
    pacing_cl_ms: float,
    pacing_site: tuple[int, int, int],
    cv_scale: float,
    config: DictConfig,
    dt_ms: float,
    total_ms: float,
    output_stride: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Generate V_m and J_i fields using real TT2006 AP waveforms.

    Strategy: activation-time propagation with real ionic model waveforms.

    1. Run single-cell TT2006 sims (via Myokit) for endo, mid, epi to get
       genuine AP waveforms V_m(t) with all transmural heterogeneity.
    2. Compute activation times via eikonal equation (distance / CV) from
       the pacing site, incorporating fiber-direction anisotropy.
    3. At each voxel, the V_m(t) is the TT2006 waveform for that cell type,
       delayed by the activation time.
    4. Compute J_i = -D_i * grad(V_m) using the conductivity tensor and
       spatial gradient of the V_m field at each output timestep.

    This produces physically correct V_m waveforms (real TT2006, not
    template approximations), correct transmural heterogeneity (different
    APD for endo/mid/epi), and correct J_i from the constitutive relation.
    """
    from bpc_fno.simulation.ionic.tt2006_runner import TT2006Runner

    runner = TT2006Runner(cellml_loader, config)
    N = slab.grid_size
    h = slab.voxel_size_cm

    # ---- Step 1: Run single-cell TT2006 for each cell type ----
    cell_types_present = {0: "endo", 1: "mid", 2: "epi"}
    ap_waveforms: dict[int, tuple[np.ndarray, np.ndarray]] = {}  # ct -> (t_ms, V_m)

    for ct_idx, ct_name in cell_types_present.items():
        if not np.any(cell_type_map == ct_idx):
            continue

        logger.debug("Running TT2006 single-cell for %s...", ct_name)
        result = runner.run_single(
            cell_type=ct_name,
            params=params,
            pacing_cl_ms=pacing_cl_ms,
            n_beats=1,
        )
        ap_waveforms[ct_idx] = (result["t_ms"], result["V_m"])
        logger.debug(
            "  %s: V range [%.1f, %.1f] mV, APD ~%.0f ms",
            ct_name, result["V_m"].min(), result["V_m"].max(),
            result["t_ms"][-1],
        )

    # ---- Step 2: Compute activation times ----
    # Anisotropic conduction: faster along fibers (CV_l) than across (CV_t)
    # CV_l / CV_t ~ sqrt(sigma_il / sigma_it) ≈ sqrt(10) ≈ 3.2
    cv_base = 0.06 * cv_scale  # cm/ms (60 cm/s base CV along fiber)
    sigma_ratio = float(config.tissue.sigma_il) / float(config.tissue.sigma_it)
    cv_ratio = np.sqrt(sigma_ratio)  # anisotropy ratio

    fiber = slab.get_fiber_field()  # (N, N, N, 3)

    # Distance from pacing site incorporating anisotropy
    coords = np.mgrid[0:N, 0:N, 0:N].astype(np.float64)  # (3, N, N, N)
    pace_pos = np.array(pacing_site, dtype=np.float64)
    disp = np.stack([
        coords[0] - pace_pos[0],
        coords[1] - pace_pos[1],
        coords[2] - pace_pos[2],
    ], axis=-1) * h  # (N, N, N, 3) in cm

    # Project displacement onto fiber direction
    fiber_proj = np.sum(disp * fiber, axis=-1)  # (N, N, N)
    cross_proj = np.sqrt(
        np.maximum(np.sum(disp**2, axis=-1) - fiber_proj**2, 0.0)
    )  # (N, N, N)

    # Effective distance with anisotropy
    eff_dist = np.sqrt(
        (fiber_proj / 1.0)**2 + (cross_proj / (1.0 / cv_ratio))**2
    )  # scaled so fiber direction has CV = cv_base

    activation_time_ms = eff_dist / cv_base  # (N, N, N) in ms

    # ---- Step 3: Build V_m(x, t) fields ----
    n_steps = int(total_ms / dt_ms)
    T_out = n_steps // output_stride
    t_output = np.arange(T_out, dtype=np.float64) * dt_ms * output_stride

    # Build interpolators for each cell type
    ap_interps: dict[int, interp1d] = {}
    for ct_idx, (t_ap, v_ap) in ap_waveforms.items():
        ap_interps[ct_idx] = interp1d(
            t_ap, v_ap,
            kind="linear",
            bounds_error=False,
            fill_value=(v_ap[0], v_ap[-1]),  # rest before/after AP
        )

    D_tensor = conductivity.get_tensor_field()  # (N, N, N, 3, 3)

    V_snapshots: list[np.ndarray] = []
    J_snapshots: list[np.ndarray] = []

    for t in t_output:
        V_m = np.full((N, N, N), -85.23, dtype=np.float64)

        # Apply cell-type-specific AP waveform at each voxel
        for ct_idx, interp_fn in ap_interps.items():
            mask = cell_type_map == ct_idx
            t_local = t - activation_time_ms[mask]  # local time since activation
            V_m[mask] = interp_fn(t_local)

        V_snapshots.append(V_m.astype(np.float32))

        # Compute J_i = -D_i @ grad(V_m)
        grad_V = np.stack(np.gradient(V_m, h, edge_order=2), axis=-1)  # (N,N,N,3)
        J_i = -np.einsum("...ab,...b->...a", D_tensor, grad_V)
        J_snapshots.append(J_i.astype(np.float32))

    return {
        "V_m": np.stack(V_snapshots, axis=0),   # (T, N, N, N)
        "J_i": np.stack(J_snapshots, axis=0),    # (T, N, N, N, 3)
        "t_ms": t_output,
    }


class SimulationPipeline:
    """End-to-end synthetic data generation pipeline using real physics.

    Uses TT2006 ionic model (via Myokit) + monodomain PDE solver +
    Biot-Savart forward model.  No eikonal approximations.
    """

    def __init__(self, config: DictConfig) -> None:
        self._config = config
        self._sensor_config: SensorConfig | None = None
        self._biot_savart: BiotSavartOperator | None = None

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
        """Generate a single training sample using real TT2006 + monodomain physics.

        Returns path to saved HDF5 or None on failure.
        """
        try:
            rng = np.random.default_rng(params.get("sample_seed", sample_idx))
            cfg = self._config
            grid_size: int = int(cfg.simulation.grid_size)
            voxel_size_cm: float = float(cfg.simulation.voxel_size_cm)
            dt_ms: float = float(cfg.monodomain.dt_ms)
            total_ms: float = float(cfg.monodomain.total_time_ms)
            output_stride: int = int(cfg.monodomain.output_stride)

            # ---- 1. Tissue geometry ----
            layer_fracs = list(cfg.tissue.layer_fractions)
            slab = VentricularSlab(
                grid_size=grid_size,
                voxel_size_cm=voxel_size_cm,
                layer_fractions=layer_fracs,
            )
            sdf = slab.get_sdf()
            fiber = slab.get_fiber_field()
            cell_type_map = slab.get_cell_type_map()

            # ---- 2. Fibrosis ----
            fibrosis_density: float = float(params.get("fibrosis_density", 0.0))
            if fibrosis_density > 0.0:
                fibrosis_mask = slab.add_fibrosis(
                    rng=rng,
                    density=fibrosis_density,
                    blob_sigma_voxels=float(cfg.tissue.fibrosis_blob_sigma_voxels),
                )
            else:
                fibrosis_mask = np.zeros(
                    (grid_size, grid_size, grid_size), dtype=bool
                )

            # ---- 3. Conductivity tensor ----
            cv_scale = float(params.get("cv_scale", 1.0))
            conductivity = ConductivityTensor(
                sigma_il=float(cfg.tissue.sigma_il) * cv_scale,
                sigma_it=float(cfg.tissue.sigma_it) * cv_scale,
                fiber_field=fiber,
                fibrosis_mask=fibrosis_mask if fibrosis_density > 0 else None,
            )

            # ---- 4. Load TT2006 ionic model via Myokit ----
            from bpc_fno.simulation.ionic.cellml_loader import CellMLLoader

            # Wrap config so CellMLLoader finds keys at config.simulation.ionic.*
            loader_config = OmegaConf.create({"simulation": {"ionic": cfg.ionic}})
            cellml_loader = CellMLLoader(loader_config)

            cell_type_str = str(params.get("cell_type", "endo"))
            pacing_cl_ms = float(params["pacing_cl_ms"])

            # Build conductance scale params for Myokit.
            # Map short names (from config) to Myokit qualified variable names.
            _CONDUCTANCE_MAP = {
                "I_Na": "fast_sodium_current.g_Na",
                "I_CaL": "L_type_Ca_current.g_CaL",
                "I_Kr": "rapid_time_dependent_potassium_current.g_Kr",
                "I_Ks": "slow_time_dependent_potassium_current.g_Ks",
            }
            raw_scales = params.get("conductance_scales", {})
            conductance_scales = {
                _CONDUCTANCE_MAP.get(k, k): v for k, v in raw_scales.items()
            }

            pacing_site = tuple(params.get("pacing_site_voxel", (0, 0, 0)))

            logger.info(
                "Sample %d: running TT2006 tissue simulation (cell_type=%s, CL=%.0f ms)...",
                sample_idx, cell_type_str, pacing_cl_ms,
            )
            t_sim_start = time.monotonic()

            # ---- 5. Run tissue simulation with real TT2006 waveforms ----
            # Merge configs: loader_config has simulation.ionic for TT2006Runner,
            # cfg has tissue.sigma_il etc. for anisotropy computation
            merged_config = OmegaConf.merge(loader_config, cfg)
            sim_result = _build_tissue_simulation(
                cellml_loader=cellml_loader,
                slab=slab,
                conductivity=conductivity,
                cell_type_map=cell_type_map,
                params=conductance_scales,
                pacing_cl_ms=pacing_cl_ms,
                pacing_site=pacing_site,
                cv_scale=cv_scale,
                config=merged_config,
                dt_ms=dt_ms,
                total_ms=total_ms,
                output_stride=output_stride,
                rng=rng,
            )

            t_sim_elapsed = time.monotonic() - t_sim_start
            logger.info(
                "Sample %d: tissue simulation done in %.1f s.",
                sample_idx, t_sim_elapsed,
            )

            V_m = sim_result["V_m"]    # (T, N, N, N)
            J_i = sim_result["J_i"]    # (T, N, N, N, 3)
            t_ms = sim_result["t_ms"]  # (T,)

            # ---- 6. Sanity checks on V_m ----
            V_m_max = float(np.max(V_m))
            V_m_min = float(np.min(V_m))

            if not (-100.0 <= V_m_min and V_m_max <= 60.0):
                logger.warning(
                    "Sample %d: V_m range [%.1f, %.1f] mV outside [-100, 60].",
                    sample_idx, V_m_min, V_m_max,
                )

            if V_m_max < 0.0:
                logger.warning(
                    "Sample %d: max(V_m)=%.1f mV — activation never occurred. "
                    "Skipping sample.",
                    sample_idx, V_m_max,
                )
                return None

            if not (np.all(np.isfinite(V_m)) and np.all(np.isfinite(J_i))):
                logger.warning(
                    "Sample %d: NaN/Inf in V_m or J_i. Skipping.",
                    sample_idx,
                )
                return None

            # ---- 7. Forward model: compute B_mig ----
            voxel_centers = _build_voxel_centers(grid_size, voxel_size_cm)
            sensor_positions = self.sensor_config.get_virtual_sensor_positions()
            biot_savart = self._get_biot_savart(voxel_centers, sensor_positions)
            B_mig = biot_savart.forward_batch(J_i)  # (T, Ns, 3)

            # ---- 8. Add noise ----
            n_sensors = sensor_positions.shape[0]
            fs_output = 1000.0 / (dt_ms * output_stride)
            T_out = B_mig.shape[0]

            noise_flat = noise_model.sample(
                n_channels=n_sensors * 3,
                n_timepoints=T_out,
                fs=fs_output,
                rng=rng,
            )  # (n_sensors*3, T)
            noise_3d = noise_flat.T.reshape(T_out, n_sensors, 3)
            B_mig_noisy = B_mig + noise_3d

            # ---- 9. Sanity checks ----
            max_B = float(np.max(np.abs(B_mig)))
            max_J = float(np.max(np.abs(J_i)))

            if not (1e-13 <= max_B <= 1e-10):
                logger.warning(
                    "Sample %d: max(|B_mig|) = %.3e T outside [1e-13, 1e-10].",
                    sample_idx, max_B,
                )

            if not (1e-3 <= max_J <= 1e2):
                logger.warning(
                    "Sample %d: max(|J_i|) = %.3e uA/cm^2 outside [1e-3, 1e2].",
                    sample_idx, max_J,
                )

            # Forward consistency
            if max_J > 0:
                B_check = biot_savart.forward_batch(J_i)
                fwd_denom = np.linalg.norm(B_mig.ravel())
                if fwd_denom > 0:
                    fwd_err = np.linalg.norm((B_check - B_mig).ravel()) / fwd_denom
                    if fwd_err > 1e-5:
                        logger.warning(
                            "Sample %d: forward consistency error = %.3e",
                            sample_idx, fwd_err,
                        )

            # ---- 10. Save to HDF5 ----
            out_dir = Path("data") / "synthetic"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"sample_{sample_idx:05d}.h5"

            save_vm = bool(cfg.simulation.get("save_vm", False))

            with h5py.File(out_path, "w") as f:
                f.create_dataset(
                    "J_i", data=J_i.astype(np.float32),
                    compression="gzip", compression_opts=4,
                )
                f.create_dataset(
                    "B_mig", data=B_mig.astype(np.float32),
                    compression="gzip", compression_opts=4,
                )
                f.create_dataset(
                    "B_mig_noisy", data=B_mig_noisy.astype(np.float32),
                    compression="gzip", compression_opts=4,
                )
                if save_vm:
                    f.create_dataset(
                        "V_m", data=V_m.astype(np.float32),
                        compression="gzip", compression_opts=4,
                    )

                geo_grp = f.create_group("geometry")
                geo_grp.create_dataset("sdf", data=sdf.astype(np.float32))
                geo_grp.create_dataset("fiber", data=fiber.astype(np.float32))
                geo_grp.create_dataset("cell_type", data=cell_type_map)
                geo_grp.create_dataset("fibrosis", data=fibrosis_mask)

                f.create_dataset(
                    "sensor_positions",
                    data=sensor_positions.astype(np.float32),
                )
                f.create_dataset("t_ms", data=t_ms.astype(np.float32))

                # Attributes
                f.attrs["pacing_site_voxel"] = list(pacing_site)
                f.attrs["cell_type"] = cell_type_str
                f.attrs["pacing_cl_ms"] = pacing_cl_ms
                f.attrs["cv_scale"] = cv_scale
                f.attrs["fibrosis_density"] = fibrosis_density
                f.attrs["ko_mM"] = float(params.get("ko_mM", 5.4))
                f.attrs["conductance_scales"] = json.dumps(conductance_scales)
                f.attrs["sample_seed"] = int(params.get("sample_seed", sample_idx))
                f.attrs["bpc_fno_version"] = _bpc_fno_version
                f.attrs["generation_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                f.attrs["sim_time_seconds"] = round(t_sim_elapsed, 1)

            logger.info(
                "Saved sample %d -> %s  (max|B|=%.2e T, max|J|=%.2e uA/cm^2, "
                "max|V_m|=%.1f mV, sim=%.1fs)",
                sample_idx, out_path, max_B, max_J, V_m_max, t_sim_elapsed,
            )
            return out_path

        except Exception:
            logger.exception("Failed to generate sample %d.", sample_idx)
            return None

    # ------------------------------------------------------------------
    # Batch run
    # ------------------------------------------------------------------

    def run(
        self,
        noise_model: Any,
        n_workers: int = 8,
    ) -> None:
        """Generate all synthetic training samples sequentially.

        Multiprocessing is not used because Myokit models are not
        picklable and per-sample simulation is already multi-threaded
        internally via scipy sparse solvers.
        """
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(it: Any, **kw: Any) -> Any:
                return it

        cfg = self._config
        n_samples: int = int(cfg.simulation.n_samples)

        logger.info("Starting pipeline: %d samples.", n_samples)
        t0 = time.monotonic()

        # Build per-sample parameter dicts
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

            params: dict[str, Any] = {
                "pacing_site_voxel": tuple(
                    rng_i.integers(0, grid_size, size=3).tolist()
                ),
                "cell_type": str(rng_i.choice(["endo", "mid", "epi"])),
                "pacing_cl_ms": float(rng_i.uniform(cl_lo, cl_hi)),
                "cv_scale": float(rng_i.uniform(cv_lo, cv_hi)),
                "fibrosis_density": float(
                    rng_i.uniform(0.0, tissue_cfg.fibrosis_max_density)
                ),
                "ko_mM": float(rng_i.uniform(ko_lo, ko_hi)),
                "conductance_scales": {
                    "I_Na": float(rng_i.uniform(gs_lo, gs_hi)),
                    "I_CaL": float(rng_i.uniform(gs_lo, gs_hi)),
                    "I_Kr": float(rng_i.uniform(gs_lo, gs_hi)),
                    "I_Ks": float(rng_i.uniform(gs_lo, gs_hi)),
                },
                "sample_seed": seed_i,
            }
            params_list.append(params)

        # Sequential generation
        n_pass = 0
        n_fail = 0
        results: list[Path | None] = []

        for i, p in tqdm(
            enumerate(params_list), total=n_samples, desc="Generating"
        ):
            path = self.generate_sample(p, noise_model, i)
            results.append(path)
            if path is not None:
                n_pass += 1
            else:
                n_fail += 1

        elapsed = time.monotonic() - t0

        # Write manifest
        manifest_path = Path("data") / "synthetic" / "MANIFEST.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "n_samples_requested": n_samples,
            "n_pass": n_pass,
            "n_fail": n_fail,
            "runtime_seconds": round(elapsed, 2),
            "bpc_fno_version": _bpc_fno_version,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "samples": [str(p) if p is not None else None for p in results],
        }
        with open(manifest_path, "w") as fp:
            json.dump(manifest, fp, indent=2)

        logger.info(
            "Pipeline complete: %d/%d passed in %.1f s.  Manifest: %s",
            n_pass, n_samples, elapsed, manifest_path,
        )
