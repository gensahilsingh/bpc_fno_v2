"""Fit OPM noise model from local KCD preprocessed recordings.

Loads all KCD records from local wav files, extracts inter-beat baseline
noise segments, fits a parametric white + 1/f^alpha PSD model per channel,
saves the result to JSON, and optionally generates a validation plot.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from bpc_fno.data.kcd_noise_model import OPMNoiseModel
from bpc_fno.data.local_kcd_loader import LocalKCDLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit OPM noise model from local KCD data"
    )
    parser.add_argument(
        "--data-dir",
        default="kcd_preprocessed/data/preprocessed",
        help="Path to the preprocessed KCD data root",
    )
    parser.add_argument(
        "--output",
        default="data/processed/noise_model.json",
        help="Output path for fitted noise model JSON",
    )
    parser.add_argument(
        "--plot-dir",
        default="data/processed",
        help="Directory to save the validation plot",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip validation plot generation",
    )
    parser.add_argument(
        "--n-segments",
        type=int,
        default=20,
        help="Max noise segments to extract per record",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error("Data directory not found: %s", data_dir)
        sys.exit(1)

    try:
        # Load all records from local wav files.
        loader = LocalKCDLoader(data_dir)
        records = loader.load_all()
        logger.info("Loaded %d records.", len(records))

        # Create and fit noise model.
        noise_model = OPMNoiseModel()
        logger.info(
            "Fitting noise model (n_noise_segments_per_record=%d)...",
            args.n_segments,
        )
        noise_model.fit(records, n_noise_segments_per_record=args.n_segments)

        # Save fitted model.
        output_path = Path(args.output)
        noise_model.save(output_path)
        logger.info("Noise model saved to %s", output_path)

        # Report per-channel statistics.
        for i, cp in enumerate(noise_model.channel_params):
            logger.info(
                "  Channel %d: sigma_white=%.4e  sigma_1f=%.4e  alpha=%.3f",
                i,
                cp.sigma_white,
                cp.sigma_1f,
                cp.alpha,
            )

        # Generate validation plot.
        if not args.no_plot:
            logger.info("Generating validation plot...")
            fig = noise_model.validate()
            plot_dir = Path(args.plot_dir)
            plot_dir.mkdir(parents=True, exist_ok=True)
            plot_path = plot_dir / "noise_model_validation.png"
            fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
            logger.info("Validation plot saved to %s", plot_path)

        logger.info("Noise model fitting complete.")

    except Exception as exc:
        logger.error("Noise model fitting failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
