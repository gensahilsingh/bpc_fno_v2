"""Resume synthetic data generation for missing sample IDs only."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

from bpc_fno.data.kcd_noise_model import OPMNoiseModel
from bpc_fno.simulation.pipeline import SimulationPipeline
from bpc_fno.utils.config_loading import load_config_with_extends

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


def _sample_id_from_path(path: Path) -> int | None:
    if not path.stem.startswith("sample_"):
        return None
    try:
        return int(path.stem.split("_", 1)[1])
    except (IndexError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume synthetic data generation for missing samples"
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
        "--output-dir", default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--n-samples", type=int, default=None,
        help="Override number of samples to generate (default: from config)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=200,
        help="Number of missing sample IDs to regenerate per fresh pipeline process",
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
        if args.output_dir is not None:
            OmegaConf.update(config, "simulation.output_dir", args.output_dir)
        if args.n_samples is not None:
            OmegaConf.update(config, "simulation.n_samples", args.n_samples)

        out_dir = Path(config.simulation.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        noise_model = OPMNoiseModel()
        noise_model.load(noise_model_path)

        selector = SimulationPipeline(config)
        selected_ids = selector.select_sample_ids(
            sample_start=args.sample_start,
            sample_count=args.sample_count,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
        )

        existing_ids = {
            sample_id
            for path in out_dir.glob("sample_*.h5")
            for sample_id in [_sample_id_from_path(path)]
            if sample_id is not None
        }
        remaining_ids = [sample_id for sample_id in selected_ids if sample_id not in existing_ids]

        logger.info(
            "Resume selection: %d requested, %d already present, %d missing.",
            len(selected_ids),
            len(selected_ids) - len(remaining_ids),
            len(remaining_ids),
        )

        if not remaining_ids:
            logger.info("No missing sample IDs remain for this selection.")
            return

        total_pass = 0
        total_fail = 0
        t0 = time.monotonic()

        for batch_start in range(0, len(remaining_ids), args.batch_size):
            batch_ids = remaining_ids[batch_start: batch_start + args.batch_size]
            logger.info(
                "Resume batch %d-%d / %d: sample IDs %d..%d",
                batch_start + 1,
                batch_start + len(batch_ids),
                len(remaining_ids),
                batch_ids[0],
                batch_ids[-1],
            )

            pipeline = SimulationPipeline(config)
            batch_pass = 0
            batch_fail = 0
            for local_index, sample_idx in enumerate(batch_ids, start=1):
                params = pipeline.build_sample_params(
                    sample_idx, seed_offset=args.seed_offset
                )
                path = pipeline.generate_sample(
                    params=params,
                    noise_model=noise_model,
                    sample_idx=sample_idx,
                    output_dir=out_dir,
                )
                if path is None:
                    batch_fail += 1
                    total_fail += 1
                else:
                    batch_pass += 1
                    total_pass += 1

                if local_index % 25 == 0 or local_index == len(batch_ids):
                    logger.info(
                        "  Resume progress: %d/%d in batch (pass=%d, fail=%d)",
                        local_index,
                        len(batch_ids),
                        batch_pass,
                        batch_fail,
                    )

            logger.info(
                "Resume batch complete: %d passed, %d failed.",
                batch_pass,
                batch_fail,
            )
            del pipeline
            gc.collect()

        elapsed = time.monotonic() - t0
        resume_manifest = {
            "output_dir": str(out_dir),
            "selected_ids": selected_ids,
            "remaining_ids": remaining_ids,
            "n_requested_missing": len(remaining_ids),
            "n_pass": total_pass,
            "n_fail": total_fail,
            "sample_start": int(args.sample_start),
            "sample_count": (
                int(args.sample_count) if args.sample_count is not None else None
            ),
            "shard_id": int(args.shard_id),
            "num_shards": int(args.num_shards),
            "seed_offset": int(args.seed_offset),
            "runtime_seconds": round(elapsed, 2),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        resume_manifest_path = out_dir / "RESUME_MANIFEST.json"
        with open(resume_manifest_path, "w", encoding="utf-8") as fp:
            json.dump(resume_manifest, fp, indent=2)

        logger.info(
            "Resume complete in %.1fs: %d passed, %d failed.",
            elapsed,
            total_pass,
            total_fail,
        )
        logger.info("Resume manifest written to %s", resume_manifest_path)

        if total_fail > 0:
            raise RuntimeError(
                f"Resume generation failed for {total_fail}/{len(remaining_ids)} "
                f"missing sample(s)."
            )

    except Exception as exc:
        logger.error("Synthetic resume generation failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
