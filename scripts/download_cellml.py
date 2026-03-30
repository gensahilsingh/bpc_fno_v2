"""Download TT2006 CellML model files from the Physiome Model Repository.

Downloads all three cell-type variants (endo, mid, epi), caches them locally,
and verifies each model loads correctly via Myokit.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from bpc_fno.simulation.ionic.cellml_loader import CellMLLoader, VALID_CELL_TYPES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _build_simulation_config(config: DictConfig) -> DictConfig:
    """Wrap raw data_gen.yaml config into the structure CellMLLoader expects.

    CellMLLoader expects ``config.simulation.ionic.*`` but data_gen.yaml has
    ``ionic.*`` at the top level.  This helper bridges the two layouts.
    """
    if OmegaConf.is_missing(config, "simulation"):
        # Wrap: put ionic config under simulation.ionic
        wrapped = OmegaConf.create({"simulation": {"ionic": config.ionic}})
        return wrapped
    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download TT2006 CellML models from PMR"
    )
    parser.add_argument(
        "--config", default="configs/data_gen.yaml",
        help="Path to data generation config file",
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help="Override local cache directory for CellML files",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    try:
        config = OmegaConf.load(args.config)
        sim_config = _build_simulation_config(config)

        # Optionally override cache directory.
        if args.cache_dir is not None:
            OmegaConf.update(
                sim_config, "simulation.ionic.cache_dir", args.cache_dir
            )

        loader = CellMLLoader(sim_config.ionic if hasattr(sim_config, 'ionic') else sim_config)

        n_success = 0
        n_fail = 0

        for cell_type in VALID_CELL_TYPES:
            logger.info("Processing cell type: %s", cell_type)
            try:
                model = loader.get_model(cell_type)
                logger.info(
                    "  OK: %s model loaded and validated (%d variables)",
                    cell_type,
                    model.count_variables(),
                )
                n_success += 1
            except Exception as exc:
                logger.error(
                    "  FAIL: %s model download/validation failed: %s",
                    cell_type,
                    exc,
                )
                n_fail += 1

        logger.info(
            "Summary: %d/%d cell types downloaded and verified successfully.",
            n_success,
            len(VALID_CELL_TYPES),
        )

        if n_fail > 0:
            logger.warning("%d cell type(s) failed.", n_fail)
            sys.exit(1)

    except Exception as exc:
        logger.error("CellML download failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
