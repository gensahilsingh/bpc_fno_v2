"""Verify synthetic HDF5 data integrity and backend-specific plausibility."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import h5py
import numpy as np

_LOG_DIR = Path("logs")
_LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_LOG_DIR / "verify_data.log"),
    ],
)
logger = logging.getLogger(__name__)


def _pick_evenly_spaced(files: list[Path], n: int) -> list[Path]:
    if len(files) <= n:
        return files
    indices = np.linspace(0, len(files) - 1, n, dtype=int)
    return [files[i] for i in indices]


def _infer_grid_shape(hf: h5py.File) -> tuple[int, int, int]:
    if "geometry/cell_type_map" in hf:
        return tuple(int(v) for v in hf["geometry/cell_type_map"].shape)
    return tuple(int(v) for v in hf["geometry/sdf"].shape)


def _check_required_dataset(hf: h5py.File, key: str) -> bool:
    if key in hf:
        return True
    logger.error("    missing dataset: %s", key)
    return False


def _vm_expectation_mode(value: str) -> str:
    value = value.lower()
    if value not in {"yes", "no", "auto"}:
        raise ValueError("--expect-vm must be one of yes/no/auto")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify synthetic data integrity and plausibility"
    )
    parser.add_argument(
        "--data-dir",
        default="data/synthetic_eikonal",
        help="Directory containing sample_*.h5 files",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=20,
        help="Number of evenly-spaced samples to check",
    )
    parser.add_argument(
        "--n-sensors",
        type=int,
        default=16,
        help="Expected number of sensors",
    )
    parser.add_argument(
        "--expect-vm",
        default="auto",
        help="Whether V_m must be present: yes, no, or auto",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Enable smoke-only acceptance checks",
    )
    parser.add_argument(
        "--time-budget-hours",
        type=float,
        default=None,
        help="Optional runtime budget check against MANIFEST.json",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error("Data directory not found: %s", data_dir)
        sys.exit(1)

    all_h5 = sorted(data_dir.glob("sample_*.h5"))
    if not all_h5:
        logger.error("No sample_*.h5 files found in %s", data_dir)
        sys.exit(1)

    selected = _pick_evenly_spaced(all_h5, args.n_samples)
    logger.info(
        "Checking %d / %d samples from %s",
        len(selected), len(all_h5), data_dir,
    )

    required = [
        "J_i",
        "B_mig",
        "B_mig_noisy",
        "geometry/sdf",
        "geometry/fiber",
        "geometry/cell_type_map",
        "sensor_positions",
        "t_ms",
        "activation_times_ms",
        "stimulus_mask",
    ]
    expect_vm = _vm_expectation_mode(args.expect_vm)

    results: dict[str, bool] = {}
    any_missing = False
    any_bad_shape = False
    any_nan_inf = False
    any_bad_b = False
    any_bad_j = False
    any_bad_vm = False
    cell_types_seen: set[int] = set()
    adjacent_corrs: list[float] = []
    activation_ok = True

    for fpath in selected:
        logger.info("  Checking %s ...", fpath.name)
        with h5py.File(fpath, "r") as hf:
            for key in required:
                if not _check_required_dataset(hf, key):
                    any_missing = True

            if any_missing:
                continue

            grid_shape = _infer_grid_shape(hf)
            t_len = hf["t_ms"].shape[0]

            if hf["J_i"].shape != (t_len, *grid_shape, 3):
                logger.error("    J_i shape mismatch: %s", hf["J_i"].shape)
                any_bad_shape = True
            if hf["B_mig"].shape != (t_len, args.n_sensors, 3):
                logger.error("    B_mig shape mismatch: %s", hf["B_mig"].shape)
                any_bad_shape = True
            if hf["B_mig_noisy"].shape != (t_len, args.n_sensors, 3):
                logger.error("    B_mig_noisy shape mismatch: %s", hf["B_mig_noisy"].shape)
                any_bad_shape = True
            if hf["geometry/fiber"].shape != (*grid_shape, 3):
                logger.error("    geometry/fiber shape mismatch: %s", hf["geometry/fiber"].shape)
                any_bad_shape = True
            if hf["sensor_positions"].shape != (args.n_sensors, 3):
                logger.error("    sensor_positions shape mismatch: %s", hf["sensor_positions"].shape)
                any_bad_shape = True
            if hf["activation_times_ms"].shape != grid_shape:
                logger.error("    activation_times_ms shape mismatch: %s", hf["activation_times_ms"].shape)
                any_bad_shape = True
            if hf["stimulus_mask"].shape != grid_shape:
                logger.error("    stimulus_mask shape mismatch: %s", hf["stimulus_mask"].shape)
                any_bad_shape = True

            has_vm = "V_m" in hf
            if expect_vm == "yes" and not has_vm:
                logger.error("    V_m missing but required")
                any_bad_vm = True
            if expect_vm == "no" and has_vm:
                logger.error("    V_m present but should be omitted")
                any_bad_vm = True

            sample_indices = [0, max(0, t_len // 2), t_len - 1]
            for t_idx in sample_indices:
                for key in ("J_i", "B_mig", "B_mig_noisy"):
                    data = np.asarray(hf[key][t_idx])
                    if not np.all(np.isfinite(data)):
                        logger.error("    non-finite data in %s at t=%d", key, t_idx)
                        any_nan_inf = True
                if has_vm:
                    vm = np.asarray(hf["V_m"][t_idx])
                    if not np.all(np.isfinite(vm)):
                        logger.error("    non-finite data in V_m at t=%d", t_idx)
                        any_nan_inf = True

            b_abs = float(np.max(np.abs(hf["B_mig"][:])))
            j_abs = float(np.max(np.abs(hf["J_i"][:])))
            if b_abs <= 0.0:
                logger.error("    B_mig is identically zero")
                any_bad_b = True
            if j_abs <= 0.0:
                logger.error("    J_i is identically zero")
                any_bad_j = True

            cell_types = np.unique(np.asarray(hf["geometry/cell_type_map"]))
            cell_types_seen.update(int(v) for v in cell_types.tolist())

            if args.smoke_only:
                act = np.asarray(hf["activation_times_ms"])
                finite_act = act[np.isfinite(act)]
                if finite_act.size == 0 or float(np.max(finite_act)) <= 0.0:
                    logger.error("    activation_times_ms does not contain valid activations")
                    activation_ok = False
                if has_vm:
                    vm_all = np.asarray(hf["V_m"])
                    if float(np.max(vm_all)) < 0.0:
                        logger.error("    V_m never depolarizes above 0 mV")
                        activation_ok = False

                j_mid = np.asarray(hf["J_i"][:, grid_shape[0] // 2, grid_shape[1] // 2, grid_shape[2] // 2, :]).ravel()
                j_adj = np.asarray(hf["J_i"][:, min(grid_shape[0] // 2 + 1, grid_shape[0] - 1), grid_shape[1] // 2, grid_shape[2] // 2, :]).ravel()
                if np.std(j_mid) > 0 and np.std(j_adj) > 0:
                    adjacent_corrs.append(float(np.corrcoef(j_mid, j_adj)[0, 1]))

    results["required_datasets"] = not any_missing
    results["shapes"] = not any_bad_shape
    results["no_nan_inf"] = not any_nan_inf
    results["nonzero_B_mig"] = not any_bad_b
    results["nonzero_J_i"] = not any_bad_j
    results["vm_policy"] = not any_bad_vm
    results["cell_type_map"] = len(cell_types_seen) >= 2

    if args.smoke_only:
        results["plausible_activation"] = activation_ok
        if adjacent_corrs:
            max_corr = max(adjacent_corrs)
            results["adjacent_voxel_variation"] = max_corr < 0.999
        else:
            max_corr = float("nan")
            results["adjacent_voxel_variation"] = False
        logger.info(
            "Smoke-only adjacent voxel correlation max: %.4f",
            max_corr,
        )

    if args.time_budget_hours is not None:
        manifest_path = data_dir / "MANIFEST.json"
        if not manifest_path.exists():
            results["time_budget"] = False
        else:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            runtime_hours = float(manifest.get("runtime_seconds", 0.0)) / 3600.0
            results["time_budget"] = runtime_hours <= float(args.time_budget_hours)
            logger.info(
                "Manifest runtime: %.2f hours (budget %.2f hours)",
                runtime_hours,
                float(args.time_budget_hours),
            )

    logger.info("")
    logger.info("=" * 60)
    logger.info("VERIFICATION RESULTS")
    logger.info("=" * 60)
    for key, passed in results.items():
        logger.info("[%s] %s", "PASS" if passed else "FAIL", key)

    n_pass = sum(results.values())
    n_total = len(results)
    logger.info("=" * 60)
    logger.info("Overall: %d/%d checks passed", n_pass, n_total)

    if n_pass < n_total:
        logger.error("Data verification FAILED.")
        sys.exit(1)

    logger.info("All data verification checks PASSED.")


if __name__ == "__main__":
    main()
