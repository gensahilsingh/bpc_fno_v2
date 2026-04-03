"""Compute per-channel normalization statistics from synthetic training data.

Walks the synthetic data directory, fits a z-score normalizer on the training
split, and saves the statistics to JSON.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf

from bpc_fno.utils.data_paths import resolve_required_data_dir, validate_sample_data_dir
from bpc_fno.utils.normalization import Normalizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute normalization statistics for BPC-FNO training"
    )
    parser.add_argument(
        "--config", default="configs/arch_a.yaml",
        help="Path to architecture config file",
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Override synthetic data directory (required unless config.data.data_dir is set)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Override output path for normalization JSON",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    try:
        config = OmegaConf.load(args.config)

        data_path = resolve_required_data_dir(config, args.data_dir)
        output_path = args.output or str(
            Path(config.data.processed_dir) / "normalization.json"
        )

        validate_sample_data_dir(data_path)

        # Create normalizer and fit on training data.
        logger.info("Fitting normalizer on data in %s", data_path)
        normalizer = Normalizer()
        normalizer.fit(data_path)

        # Save statistics.
        normalizer.save(output_path)
        logger.info("Normalization statistics saved to %s", output_path)

        # Report per-channel statistics.
        logger.info("=== Per-channel statistics ===")

        if "J_i_mean" in normalizer.stats and "J_i_std" in normalizer.stats:
            j_mean = normalizer.stats["J_i_mean"]
            j_std = normalizer.stats["J_i_std"]
            for i, (m, s) in enumerate(zip(j_mean, j_std)):
                logger.info("  J_i channel %d: mean=%.6e  std=%.6e", i, m, s)

        if "B_mean" in normalizer.stats and "B_std" in normalizer.stats:
            b_mean = normalizer.stats["B_mean"]
            b_std = normalizer.stats["B_std"]
            for i, (m, s) in enumerate(zip(b_mean, b_std)):
                logger.info("  B   channel %d: mean=%.6e  std=%.6e", i, m, s)

        logger.info("Normalization computation complete.")

    except Exception as exc:
        logger.error(
            "Normalization computation failed: %s", exc, exc_info=True
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
