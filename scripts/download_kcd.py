"""Download Kiel Cardio Database from PhysioNet."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf

from bpc_fno.data.kcd_loader import KCDLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download KCD from PhysioNet")
    parser.add_argument("--config", default="configs/kcd.yaml")
    parser.add_argument("--output-dir", default="data/kcd")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    try:
        config = OmegaConf.load(args.config)
        loader = KCDLoader(config.kcd)
        logger.info("Downloading KCD to %s", args.output_dir)
        loader.download(args.output_dir)
        logger.info("Download complete.")
    except Exception as exc:
        logger.error("Download failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
