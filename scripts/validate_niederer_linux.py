"""Validate the Linux openCARP backend against the Niederer 2011 benchmark."""

from __future__ import annotations

import argparse
import shutil
import sys

import numpy as np

from bpc_fno.simulation.pipeline import SimulationPipeline
from bpc_fno.utils.config_loading import load_config_with_extends


NIEDERER_POINTS_MM = [
    (0.0, 0.0, 0.0),
    (3.0, 0.0, 0.0),
    (7.0, 0.0, 0.0),
    (10.0, 0.0, 0.0),
    (0.0, 3.0, 0.0),
    (0.0, 0.0, 1.5),
    (3.0, 3.0, 0.0),
    (3.0, 0.0, 1.5),
    (10.0, 3.0, 1.5),
]

NIEDERER_EXPECTED_MS = [
    1.2, 5.5, 11.5, 16.7,
    5.6, 2.3, 9.2, 6.7, 22.1,
]


def _activation_time(trace: np.ndarray, t_ms: np.ndarray, threshold: float) -> float:
    idx = np.flatnonzero(trace >= threshold)
    if idx.size == 0:
        return float("inf")
    return float(t_ms[int(idx[0])])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the strict Linux/openCARP Niederer benchmark"
    )
    parser.add_argument(
        "--config",
        default="configs/benchmark_niederer.yaml",
        help="Benchmark config file",
    )
    parser.add_argument(
        "--tolerance-ms",
        type=float,
        default=1.5,
        help="Allowed activation-time error tolerance",
    )
    args = parser.parse_args()

    if shutil.which("openCARP") is None and shutil.which("carp.pt") is None:
        print(
            "openCARP benchmark cannot run: neither 'openCARP' nor 'carp.pt' "
            "was found on PATH.",
            file=sys.stderr,
        )
        sys.exit(2)

    config = load_config_with_extends(args.config)
    pipeline = SimulationPipeline(config)
    if pipeline._backend.name != "opencarp":
        print(
            f"Benchmark config resolved to backend '{pipeline._backend.name}', "
            "but 'opencarp' is required.",
            file=sys.stderr,
        )
        sys.exit(2)

    params = {
        "pacing_site_voxel": (0, 0, 0),
        "pacing_cl_ms": 1000.0,
        "cv_scale": 1.0,
        "fibrosis_density": 0.0,
        "ko_mM": 5.4,
        "conductance_scales": {
            "I_Na": 1.0,
            "I_CaL": 1.0,
            "I_Kr": 1.0,
            "I_Ks": 1.0,
        },
        "sample_seed": 0,
    }

    _, result = pipeline.simulate_fields(params)
    if result.V_m is None:
        print(
            "Benchmark config must set simulation.save_vm=true to evaluate activation times.",
            file=sys.stderr,
        )
        sys.exit(2)

    voxel_size_mm = float(config.simulation.voxel_size_cm) * 10.0
    t_ms = np.asarray(result.t_ms, dtype=np.float64)
    vm = np.asarray(result.V_m, dtype=np.float64)
    threshold = float(
        config.monodomain.get("activation_threshold_mV", -40.0)
    )

    print("=== NIEDERER 2011 BENCHMARK (openCARP) ===")
    all_pass = True

    for point_mm, expected_ms in zip(NIEDERER_POINTS_MM, NIEDERER_EXPECTED_MS):
        ix = int(round(point_mm[0] / voxel_size_mm))
        iy = int(round(point_mm[1] / voxel_size_mm))
        iz = int(round(point_mm[2] / voxel_size_mm))
        trace = vm[:, ix, iy, iz]
        observed_ms = _activation_time(trace, t_ms, threshold)
        error_ms = abs(observed_ms - expected_ms)
        passed = error_ms <= float(args.tolerance_ms)
        all_pass = all_pass and passed
        print(
            f"  {point_mm}: got={observed_ms:.2f}ms "
            f"expected={expected_ms:.2f}ms err={error_ms:.2f}ms "
            f"[{'PASS' if passed else 'FAIL'}]"
        )

    if not all_pass:
        print("NIEDERER BENCHMARK: FAILED", file=sys.stderr)
        sys.exit(1)

    print("NIEDERER BENCHMARK: ALL PASS")


if __name__ == "__main__":
    main()
