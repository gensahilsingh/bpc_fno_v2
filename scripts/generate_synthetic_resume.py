"""Resume synthetic data generation from where it left off.

Generates samples in batches to avoid memory buildup that can cause
segfaults in long-running processes.
"""

from __future__ import annotations

import argparse
import gc
import glob
import logging
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

from bpc_fno.data.kcd_noise_model import OPMNoiseModel

_LOG_DIR = Path("logs")
_LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_LOG_DIR / "data_generation.log"),
    ],
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume synthetic data generation in batches"
    )
    parser.add_argument(
        "--config", default="configs/data_gen.yaml",
    )
    parser.add_argument(
        "--noise-model", default="data/processed/noise_model.json",
    )
    parser.add_argument(
        "--n-samples", type=int, default=4000,
    )
    parser.add_argument(
        "--batch-size", type=int, default=200,
        help="Samples per batch (process restarts between batches)",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    noise_model = OPMNoiseModel()
    noise_model.load(args.noise_model)

    # Find already-generated samples
    existing = set()
    for f in glob.glob("data/synthetic/sample_*.h5"):
        idx = int(Path(f).stem.split("_")[1])
        existing.add(idx)

    remaining = [i for i in range(args.n_samples) if i not in existing]
    logger.info(
        "%d samples already exist, %d remaining.",
        len(existing), len(remaining),
    )

    if not remaining:
        logger.info("All samples already generated!")
        return

    # Process in batches
    for batch_start in range(0, len(remaining), args.batch_size):
        batch_indices = remaining[batch_start:batch_start + args.batch_size]
        logger.info(
            "=== Batch: samples %d-%d (indices %d-%d) ===",
            batch_start, batch_start + len(batch_indices) - 1,
            batch_indices[0], batch_indices[-1],
        )

        # Fresh pipeline per batch to avoid memory accumulation
        from bpc_fno.simulation.pipeline import SimulationPipeline
        import numpy as np

        # Recreate parameter list with same RNG for reproducibility
        rng_master = np.random.default_rng(42)
        all_params: list[dict] = []
        for i in range(args.n_samples):
            seed_i = int(rng_master.integers(0, 2**31))
            rng_i = np.random.default_rng(seed_i)

            ionic_cfg = config.ionic
            tissue_cfg = config.tissue
            cl_lo, cl_hi = ionic_cfg.pacing_cycle_length_ms_range
            cv_lo, cv_hi = tissue_cfg.conduction_velocity_scale_range
            ko_lo, ko_hi = ionic_cfg.ko_range_mM
            gs_lo, gs_hi = ionic_cfg.conductance_scaling_range
            grid_size = int(config.simulation.grid_size)

            params = {
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
            all_params.append(params)

        pipeline = SimulationPipeline(config)
        n_pass = 0
        n_fail = 0

        for idx in batch_indices:
            t0 = time.monotonic()
            path = pipeline.generate_sample(all_params[idx], noise_model, idx)
            elapsed = time.monotonic() - t0
            if path is not None:
                n_pass += 1
            else:
                n_fail += 1

            if (n_pass + n_fail) % 50 == 0:
                logger.info(
                    "  Progress: %d/%d (pass=%d, fail=%d, %.1fs/sample)",
                    n_pass + n_fail, len(batch_indices),
                    n_pass, n_fail, elapsed,
                )

        logger.info(
            "Batch done: %d pass, %d fail.", n_pass, n_fail,
        )

        # Force cleanup
        del pipeline
        gc.collect()

    # Final count
    final_count = len(glob.glob("data/synthetic/sample_*.h5"))
    logger.info("Total samples on disk: %d / %d requested.", final_count, args.n_samples)


if __name__ == "__main__":
    main()
