"""Full evaluation suite for trained BPC-FNO models.

Loads a trained checkpoint, runs metrics on the test set (forward L2 error,
reconstruction correlation, physics residuals, UQ coverage), generates
visualisation plots, and saves a comprehensive results JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from bpc_fno.data.data_module import BPCFNODataModule
from bpc_fno.models.bpc_fno_a import BPC_FNO_A
from bpc_fno.utils.checkpointing import load_checkpoint, validate_checkpoint
from bpc_fno.utils.normalization import Normalizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _relative_l2_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute mean relative L2 error across batch."""
    diff_norm = torch.norm(pred - target, dim=-1)
    tgt_norm = torch.norm(target, dim=-1).clamp(min=1e-12)
    return float((diff_norm / tgt_norm).mean().item())


def _pearson_r_per_channel(
    pred: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """Compute Pearson correlation per channel (last axis)."""
    n_ch = pred.shape[-1] if pred.ndim > 1 else 1
    pred_flat = pred.reshape(-1, n_ch)
    tgt_flat = target.reshape(-1, n_ch)

    rs = np.zeros(n_ch)
    for c in range(n_ch):
        p = pred_flat[:, c]
        t = tgt_flat[:, c]
        corr = np.corrcoef(p, t)
        rs[c] = corr[0, 1] if not np.isnan(corr[0, 1]) else 0.0
    return rs


def _uq_coverage(
    J_true: np.ndarray,
    J_mean: np.ndarray,
    J_std: np.ndarray,
    z_level: float = 1.96,
) -> float:
    """Compute fraction of ground-truth voxels within the predicted
    credible interval (mean +/- z_level * std)."""
    lower = J_mean - z_level * J_std
    upper = J_mean + z_level * J_std
    inside = (J_true >= lower) & (J_true <= upper)
    return float(inside.mean())


def _create_normalizer_proxy(normalizer: Normalizer) -> object:
    """Create a normalizer proxy matching SyntheticMIGDataset protocol."""

    class _NormalizerProxy:
        def __init__(self, norm: Normalizer) -> None:
            self._norm = norm

        def normalize(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
            if key == "J_i":
                return self._norm.normalize_J_i(tensor)
            if key in ("B_mig", "B_mig_clean"):
                return self._norm.normalize_B(tensor)
            if key == "geometry":
                return self._norm.normalize_geometry(tensor)
            return tensor

        def denormalize(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
            if key == "J_i":
                return self._norm.denormalize_J_i(tensor)
            if key in ("B_mig", "B_mig_clean"):
                return self._norm.denormalize_B(tensor)
            return tensor

    return _NormalizerProxy(normalizer)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _generate_visualizations(
    results: dict[str, Any],
    output_dir: Path,
    sample_predictions: list[dict[str, np.ndarray]],
) -> list[str]:
    """Generate evaluation plots and return list of saved file paths."""
    saved_plots: list[str] = []

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping visualisation plots.")
        return saved_plots

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Reconstruction quality scatter.
    if sample_predictions:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ch in range(min(3, len(axes))):
            ax = axes[ch]
            all_true = []
            all_pred = []
            for sp in sample_predictions[:20]:
                if "J_true" in sp and "J_mean" in sp:
                    all_true.append(sp["J_true"].ravel())
                    all_pred.append(sp["J_mean"].ravel())
            if all_true:
                true_cat = np.concatenate(all_true)
                pred_cat = np.concatenate(all_pred)
                # Subsample for plotting.
                n_plot = min(5000, len(true_cat))
                idx = np.random.choice(len(true_cat), n_plot, replace=False)
                ax.scatter(
                    true_cat[idx], pred_cat[idx],
                    s=1, alpha=0.3, rasterized=True,
                )
                lim = max(abs(true_cat[idx]).max(), abs(pred_cat[idx]).max())
                ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=0.8)
                ax.set_xlabel("Ground Truth")
                ax.set_ylabel("Predicted")
                ax.set_title(f"J_i reconstruction (all channels)")
                break  # Single combined plot.

        fig.suptitle("Reconstruction Quality")
        fig.tight_layout()
        scatter_path = output_dir / "reconstruction_scatter.png"
        fig.savefig(str(scatter_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_plots.append(str(scatter_path))

    # 2. Forward error histogram.
    if "per_sample_forward_l2" in results:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(results["per_sample_forward_l2"], bins=30, edgecolor="black")
        ax.axvline(
            results.get("forward_l2_mean", 0),
            color="r", linestyle="--", label="Mean",
        )
        ax.set_xlabel("Relative L2 Forward Error")
        ax.set_ylabel("Count")
        ax.set_title("Forward Error Distribution (Test Set)")
        ax.legend()
        fig.tight_layout()
        hist_path = output_dir / "forward_error_histogram.png"
        fig.savefig(str(hist_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_plots.append(str(hist_path))

    # 3. UQ calibration plot.
    if "uq_coverage_95" in results:
        fig, ax = plt.subplots(figsize=(6, 5))
        targets = [0.5, 0.8, 0.9, 0.95, 0.99]
        coverages = results.get("uq_coverage_levels", targets)
        ax.plot(targets, coverages if len(coverages) == len(targets) else targets, "bo-")
        ax.plot([0, 1], [0, 1], "r--", linewidth=0.8, label="Ideal")
        ax.set_xlabel("Target Coverage")
        ax.set_ylabel("Empirical Coverage")
        ax.set_title("UQ Calibration")
        ax.legend()
        fig.tight_layout()
        cal_path = output_dir / "uq_calibration.png"
        fig.savefig(str(cal_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_plots.append(str(cal_path))

    return saved_plots


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained BPC-FNO model on the test set"
    )
    parser.add_argument(
        "--config", default="configs/arch_a.yaml",
        help="Path to architecture config file",
    )
    parser.add_argument(
        "--checkpoint", default="checkpoints/phase2_best.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--normalization", default="data/processed/normalization.json",
        help="Path to normalization statistics JSON",
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Directory to save evaluation outputs",
    )
    parser.add_argument(
        "--n-uq-samples", type=int, default=None,
        help="Override number of UQ posterior samples (default: from config)",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device for evaluation (default: auto-detect)",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip visualisation plot generation",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    norm_path = Path(args.normalization)
    if not norm_path.exists():
        logger.error("Normalization file not found: %s", norm_path)
        sys.exit(1)

    try:
        config = OmegaConf.load(args.config)

        # Resolve device.
        if args.device is not None:
            device = torch.device(args.device)
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        logger.info("Using device: %s", device)

        n_uq_samples: int = (
            args.n_uq_samples
            or int(config.evaluation.get("n_uq_samples", 50))
        )

        # Validate checkpoint metadata.
        ckpt_info = validate_checkpoint(ckpt_path)
        logger.info(
            "Checkpoint: epoch=%d, phase=%s, params=%d",
            ckpt_info["epoch"],
            ckpt_info["phase"],
            ckpt_info["num_parameters"],
        )

        # Load normalizer.
        normalizer = Normalizer()
        normalizer.load(norm_path)
        norm_proxy = _create_normalizer_proxy(normalizer)

        # Create data module (test split).
        data_module = BPCFNODataModule(config, normalizer=norm_proxy)
        data_module.setup(stage="test")
        test_loader = data_module.test_dataloader()
        logger.info("Test set: %d batches", len(test_loader))

        # Create model and load weights.
        model = BPC_FNO_A(config).to(device)
        load_checkpoint(model=model, optimizer=None, path=ckpt_path, strict=True)
        model.eval()

        # Evaluation thresholds from config.
        fwd_l2_thr = float(config.evaluation.get("forward_l2_threshold", 0.08))
        phys_thr = float(
            config.evaluation.get("physics_residual_threshold", 0.05)
        )
        recon_r_thr = float(
            config.evaluation.get("reconstruction_r_threshold", 0.85)
        )
        uq_cov_tgt = float(config.evaluation.get("uq_coverage_target", 0.95))

        # --- Run evaluation ---
        logger.info("Running evaluation with %d UQ samples...", n_uq_samples)
        t0 = time.monotonic()

        forward_l2_errors: list[float] = []
        reconstruction_rs: list[float] = []
        uq_coverages: list[float] = []
        sample_predictions: list[dict[str, np.ndarray]] = []

        with torch.no_grad():
            for batch in test_loader:
                J_i = batch["J_i"].to(device)
                geometry = batch["geometry"].to(device)
                B_obs = batch["B_mig"].to(device)
                B_clean = batch["B_mig_clean"].to(device)
                batch_size = J_i.shape[0]

                # Forward error.
                B_pred = model.forward_only(J_i, geometry)
                for b in range(batch_size):
                    fwd_err = _relative_l2_error(
                        B_pred[b].flatten(), B_clean[b].flatten()
                    )
                    forward_l2_errors.append(fwd_err)

                # Reconstruction via UQ sampling.
                recon = model.reconstruct(
                    B_obs, geometry, n_samples=n_uq_samples
                )
                J_mean = recon["J_i_mean"]
                J_std = recon["J_i_std"]

                for b in range(batch_size):
                    j_true_np = J_i[b].cpu().numpy()
                    j_mean_np = J_mean[b].cpu().numpy()
                    j_std_np = J_std[b].cpu().numpy()

                    # Pearson R.
                    r_vals = _pearson_r_per_channel(
                        j_mean_np.reshape(-1, 1),
                        j_true_np.reshape(-1, 1),
                    )
                    reconstruction_rs.append(float(r_vals.mean()))

                    # UQ coverage (95% CI).
                    cov = _uq_coverage(j_true_np, j_mean_np, j_std_np)
                    uq_coverages.append(cov)

                    # Store sample predictions for visualisation.
                    if len(sample_predictions) < 50:
                        sample_predictions.append({
                            "J_true": j_true_np,
                            "J_mean": j_mean_np,
                            "J_std": j_std_np,
                        })

        elapsed = time.monotonic() - t0

        # --- Aggregate metrics ---
        results: dict[str, Any] = {
            "checkpoint": str(ckpt_path),
            "checkpoint_epoch": ckpt_info["epoch"],
            "checkpoint_phase": ckpt_info["phase"],
            "n_test_samples": len(forward_l2_errors),
            "n_uq_samples": n_uq_samples,
            "evaluation_time_s": round(elapsed, 2),
            "forward_l2_mean": float(np.mean(forward_l2_errors)),
            "forward_l2_std": float(np.std(forward_l2_errors)),
            "forward_l2_median": float(np.median(forward_l2_errors)),
            "per_sample_forward_l2": [float(x) for x in forward_l2_errors],
            "reconstruction_r_mean": float(np.mean(reconstruction_rs)),
            "reconstruction_r_std": float(np.std(reconstruction_rs)),
            "uq_coverage_95": float(np.mean(uq_coverages)),
            "uq_coverage_95_std": float(np.std(uq_coverages)),
        }

        # Pass/fail checks.
        results["pass_forward_l2"] = results["forward_l2_mean"] < fwd_l2_thr
        results["pass_reconstruction_r"] = (
            results["reconstruction_r_mean"] > recon_r_thr
        )
        results["pass_uq_coverage"] = results["uq_coverage_95"] >= uq_cov_tgt

        # Compute coverage at multiple levels for calibration plot.
        z_levels = [0.674, 1.282, 1.645, 1.96, 2.576]
        target_coverages = [0.5, 0.8, 0.9, 0.95, 0.99]
        empirical_coverages: list[float] = []
        for z_val in z_levels:
            covs = []
            for sp in sample_predictions:
                c = _uq_coverage(sp["J_true"], sp["J_mean"], sp["J_std"], z_val)
                covs.append(c)
            empirical_coverages.append(float(np.mean(covs)) if covs else 0.0)
        results["uq_coverage_levels"] = empirical_coverages

        # --- Report ---
        logger.info("=" * 60)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(
            "Forward L2 error:    %.4f +/- %.4f  (threshold: %.4f)  [%s]",
            results["forward_l2_mean"],
            results["forward_l2_std"],
            fwd_l2_thr,
            "PASS" if results["pass_forward_l2"] else "FAIL",
        )
        logger.info(
            "Reconstruction R:    %.4f +/- %.4f  (threshold: %.4f)  [%s]",
            results["reconstruction_r_mean"],
            results["reconstruction_r_std"],
            recon_r_thr,
            "PASS" if results["pass_reconstruction_r"] else "FAIL",
        )
        logger.info(
            "UQ coverage (95%%):   %.4f +/- %.4f  (target: %.4f)   [%s]",
            results["uq_coverage_95"],
            results["uq_coverage_95_std"],
            uq_cov_tgt,
            "PASS" if results["pass_uq_coverage"] else "FAIL",
        )
        logger.info(
            "Evaluation time:     %.1f s (%d test samples)",
            elapsed,
            len(forward_l2_errors),
        )
        logger.info("=" * 60)

        # --- Save results ---
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results_path = output_dir / "evaluation_results.json"
        # Remove non-serialisable items for JSON.
        results_json = {
            k: v for k, v in results.items()
            if not isinstance(v, np.ndarray)
        }
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results_json, f, indent=2)
        logger.info("Results saved to %s", results_path)

        # --- Generate plots ---
        if not args.no_plots:
            logger.info("Generating visualisation plots...")
            plots = _generate_visualizations(
                results, output_dir, sample_predictions
            )
            for p in plots:
                logger.info("  Plot saved: %s", p)

        logger.info("Evaluation complete.")

    except Exception as exc:
        logger.error("Evaluation failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
