"""Verify synthetic HDF5 data integrity and physical plausibility.

Loads 20 evenly-spaced samples from data/synthetic and checks:
  - B_mig range: 1e-14 < max(|B|) < 1e-10 T
  - J_i range: 0.1 < max(|J|) < 10.0 uA/cm^2
  - J_i std across samples > 0.05
  - At least 2 cell types present
  - V_m NOT stored
  - No NaN/Inf
  - Shapes correct: J_i (T, 32, 32, 32, 3), B_mig (T, 16, 3)

Prints PASS/FAIL for each check.
"""

from __future__ import annotations

import argparse
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
    """Select n evenly-spaced files from a sorted list."""
    if len(files) <= n:
        return files
    indices = np.linspace(0, len(files) - 1, n, dtype=int)
    return [files[i] for i in indices]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify synthetic data integrity and plausibility"
    )
    parser.add_argument(
        "--data-dir", default="data/synthetic",
        help="Directory containing sample_*.h5 files",
    )
    parser.add_argument(
        "--n-samples", type=int, default=20,
        help="Number of evenly-spaced samples to check (default: 20)",
    )
    parser.add_argument(
        "--grid-size", type=int, default=32,
        help="Expected spatial grid size N (default: 32)",
    )
    parser.add_argument(
        "--n-sensors", type=int, default=16,
        help="Expected number of sensors (default: 16)",
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

    N = args.grid_size
    Ns = args.n_sensors

    # Accumulators for cross-sample checks
    all_J_maxabs: list[float] = []
    cell_types_seen: set[str] = set()
    results: dict[str, bool] = {}

    # Per-sample checks
    any_nan_inf = False
    any_bad_b_range = False
    any_bad_j_range = False
    any_bad_shape = False
    any_vm_stored = False

    try:
        for fpath in selected:
            logger.info("  Checking %s ...", fpath.name)
            try:
                with h5py.File(fpath, "r") as f:
                    # --- Shape checks ---
                    if "J_i" not in f:
                        logger.error("    MISSING J_i dataset")
                        any_bad_shape = True
                        continue

                    j_shape = f["J_i"].shape
                    # Expected: (T, N, N, N, 3)
                    if len(j_shape) != 5:
                        logger.error(
                            "    J_i ndim=%d, expected 5", len(j_shape)
                        )
                        any_bad_shape = True
                    elif j_shape[1:] != (N, N, N, 3):
                        logger.error(
                            "    J_i shape=%s, expected (T, %d, %d, %d, 3)",
                            j_shape, N, N, N,
                        )
                        any_bad_shape = True

                    if "B_mig" not in f:
                        logger.error("    MISSING B_mig dataset")
                        any_bad_shape = True
                        continue

                    b_shape = f["B_mig"].shape
                    # Expected: (T, Ns, 3)
                    if len(b_shape) != 3:
                        logger.error(
                            "    B_mig ndim=%d, expected 3", len(b_shape)
                        )
                        any_bad_shape = True
                    elif b_shape[1:] != (Ns, 3):
                        logger.error(
                            "    B_mig shape=%s, expected (T, %d, 3)",
                            b_shape, Ns,
                        )
                        any_bad_shape = True

                    # --- V_m NOT stored ---
                    if "V_m" in f:
                        logger.warning("    V_m dataset found (should not be stored)")
                        any_vm_stored = True

                    # --- NaN/Inf checks ---
                    # Sample a few timesteps to avoid reading full arrays
                    T = j_shape[0]
                    t_check = [0, T // 2, T - 1] if T >= 3 else list(range(T))

                    for t in t_check:
                        j_data = np.asarray(f["J_i"][t])
                        b_data = np.asarray(f["B_mig"][t])

                        if not np.all(np.isfinite(j_data)):
                            logger.error("    NaN/Inf in J_i at t=%d", t)
                            any_nan_inf = True
                        if not np.all(np.isfinite(b_data)):
                            logger.error("    NaN/Inf in B_mig at t=%d", t)
                            any_nan_inf = True

                    # --- B_mig range check ---
                    # Read all B_mig to get true max
                    b_all = np.asarray(f["B_mig"])
                    b_maxabs = float(np.max(np.abs(b_all)))
                    if not (1e-14 < b_maxabs < 1e-10):
                        logger.warning(
                            "    B_mig max(|B|)=%.2e (expected 1e-14..1e-10 T)",
                            b_maxabs,
                        )
                        any_bad_b_range = True
                    else:
                        logger.info("    B_mig max(|B|)=%.2e T -- OK", b_maxabs)

                    # --- J_i range check ---
                    # Read all J_i to get true max (may be large)
                    j_all = np.asarray(f["J_i"])
                    j_maxabs = float(np.max(np.abs(j_all)))
                    all_J_maxabs.append(j_maxabs)
                    if not (0.1 < j_maxabs < 10.0):
                        logger.warning(
                            "    J_i max(|J|)=%.4f (expected 0.1..10.0 uA/cm^2)",
                            j_maxabs,
                        )
                        any_bad_j_range = True
                    else:
                        logger.info(
                            "    J_i max(|J|)=%.4f uA/cm^2 -- OK", j_maxabs
                        )

                    # --- Cell type ---
                    if "cell_type" in f.attrs:
                        ct = str(f.attrs["cell_type"])
                        cell_types_seen.add(ct)
                    elif "metadata" in f and "cell_type" in f["metadata"].attrs:
                        ct = str(f["metadata"].attrs["cell_type"])
                        cell_types_seen.add(ct)

            except Exception as exc:
                logger.error("    Error reading %s: %s", fpath.name, exc)
                any_nan_inf = True  # treat read errors as data issues

        # ---- Aggregate checks ----
        logger.info("")
        logger.info("=" * 60)
        logger.info("VERIFICATION RESULTS")
        logger.info("=" * 60)

        # 1. B_mig range
        results["B_mig_range"] = not any_bad_b_range
        status = "PASS" if results["B_mig_range"] else "FAIL"
        logger.info("[%s] B_mig range: 1e-14 < max(|B|) < 1e-10 T", status)

        # 2. J_i range
        results["J_i_range"] = not any_bad_j_range
        status = "PASS" if results["J_i_range"] else "FAIL"
        logger.info("[%s] J_i range: 0.1 < max(|J|) < 10.0 uA/cm^2", status)

        # 3. J_i std across samples
        if len(all_J_maxabs) >= 2:
            j_std = float(np.std(all_J_maxabs))
            results["J_i_std"] = j_std > 0.05
        else:
            j_std = 0.0
            results["J_i_std"] = False
        status = "PASS" if results["J_i_std"] else "FAIL"
        logger.info(
            "[%s] J_i std across samples: %.4f (threshold > 0.05)",
            status, j_std,
        )

        # 4. At least 2 cell types
        results["cell_types"] = len(cell_types_seen) >= 2
        status = "PASS" if results["cell_types"] else "FAIL"
        logger.info(
            "[%s] Cell types present: %s (need >= 2)",
            status, sorted(cell_types_seen) if cell_types_seen else "none found",
        )

        # 5. V_m NOT stored
        results["no_Vm"] = not any_vm_stored
        status = "PASS" if results["no_Vm"] else "FAIL"
        logger.info("[%s] V_m NOT stored in HDF5 files", status)

        # 6. No NaN/Inf
        results["no_nan_inf"] = not any_nan_inf
        status = "PASS" if results["no_nan_inf"] else "FAIL"
        logger.info("[%s] No NaN/Inf values", status)

        # 7. Shapes correct
        results["shapes"] = not any_bad_shape
        status = "PASS" if results["shapes"] else "FAIL"
        logger.info(
            "[%s] Shapes correct: J_i (T,%d,%d,%d,3), B_mig (T,%d,3)",
            status, N, N, N, Ns,
        )

        logger.info("=" * 60)
        n_pass = sum(results.values())
        n_total = len(results)
        logger.info(
            "Overall: %d/%d checks passed", n_pass, n_total
        )

        if n_pass < n_total:
            logger.error("Data verification FAILED.")
            sys.exit(1)
        else:
            logger.info("All data verification checks PASSED.")

    except Exception as exc:
        logger.error("Data verification failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
