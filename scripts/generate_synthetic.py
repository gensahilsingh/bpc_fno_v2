"""Run the full synthetic data generation pipeline.

Loads the data generation config, loads a previously fitted noise model,
and generates all n_samples synthetic MIG training pairs using multiprocessing
with a progress bar.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

from bpc_fno.data.kcd_noise_model import OPMNoiseModel
from bpc_fno.simulation.pipeline import SimulationPipeline
from bpc_fno.utils.config_loading import load_config_with_extends

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic MIG training data"
    )
    parser.add_argument(
        "--config", default="configs/data_gen_eikonal.yaml",
        help="Path to data generation config file",
    )
    parser.add_argument(
        "--noise-model", default="data/processed/noise_model.json",
        help="Path to fitted noise model JSON",
    )
    parser.add_argument(
        "--n-workers", type=int, default=None,
        help="Override number of worker processes (default: from config)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=None,
        help="Override number of samples to generate (default: from config)",
    )
    parser.add_argument(
        "--pipeline", choices=["eikonal", "monodomain"], default=None,
        help="Override simulation pipeline type",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "eikonal", "windows_hybrid", "opencarp", "chaste"],
        default=None,
        help="Override simulation backend",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--save-vm", action="store_true",
        help="Force saving V_m into generated HDF5 files",
    )
    parser.add_argument(
        "--sample-start", type=int, default=0,
        help="Global sample index to start from before sharding",
    )
    parser.add_argument(
        "--sample-count", type=int, default=None,
        help="Maximum number of global sample indices to consider",
    )
    parser.add_argument(
        "--shard-id", type=int, default=0,
        help="Shard index for distributed generation",
    )
    parser.add_argument(
        "--num-shards", type=int, default=1,
        help="Total number of shards for distributed generation",
    )
    parser.add_argument(
        "--seed-offset", type=int, default=0,
        help="Extra seed offset for distributed generation",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    noise_model_path = Path(args.noise_model)
    if not noise_model_path.exists():
        logger.error(
            "Noise model file not found: %s. "
            "Run scripts/fit_noise_model.py first.",
            noise_model_path,
        )
        sys.exit(1)

    try:
        config = load_config_with_extends(args.config)

        # Apply overrides.
        if args.n_workers is not None:
            OmegaConf.update(config, "simulation.n_workers", args.n_workers)
        if args.n_samples is not None:
            OmegaConf.update(config, "simulation.n_samples", args.n_samples)
        if args.pipeline is not None:
            OmegaConf.update(config, "simulation.pipeline", args.pipeline)
        if args.backend is not None:
            OmegaConf.update(config, "simulation.backend", args.backend)
        if args.output_dir is not None:
            OmegaConf.update(config, "simulation.output_dir", args.output_dir)
        if args.save_vm:
            OmegaConf.update(config, "simulation.save_vm", True)

        n_samples: int = int(config.simulation.n_samples)
        n_workers: int = int(config.simulation.n_workers)

        # Load noise model.
        logger.info("Loading noise model from %s", noise_model_path)
        noise_model = OPMNoiseModel()
        noise_model.load(noise_model_path)
        logger.info(
            "Noise model loaded (%d channels).",
            len(noise_model.channel_params),
        )

        # Create and run pipeline.
        logger.info(
            "Starting pipeline: n_samples=%d, n_workers=%d",
            n_samples,
            n_workers,
        )
        t0 = time.monotonic()

        pipeline = SimulationPipeline(config)
        manifest = pipeline.run(
            noise_model=noise_model,
            n_workers=n_workers,
            output_dir=config.simulation.output_dir,
            sample_start=args.sample_start,
            sample_count=args.sample_count,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
            seed_offset=args.seed_offset,
        )

        elapsed = time.monotonic() - t0
        logger.info(
            "Generation complete in %.1f s (%.2f s/sample).",
            elapsed,
            elapsed / max(int(manifest.get("n_samples_requested", n_samples)), 1),
        )

        # Report statistics from manifest.
        manifest_path = Path(config.simulation.output_dir) / "MANIFEST.json"
        if manifest_path.exists():
            import json

            with open(manifest_path) as fp:
                manifest = json.load(fp)
            logger.info(
                "Results: %d passed, %d failed out of %d requested.",
                manifest.get("n_pass", 0),
                manifest.get("n_fail", 0),
                manifest.get("n_samples_requested", n_samples),
            )
        else:
            logger.warning("Manifest file not found at %s", manifest_path)

    except Exception as exc:
        logger.error("Synthetic generation failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
