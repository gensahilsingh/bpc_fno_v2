"""Evaluation metrics for the BPC-FNO inverse cardiac imaging pipeline.

Includes volumetric error metrics, physics-consistency checks,
uncertainty-quantification diagnostics, and activation-time analysis.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from omegaconf import DictConfig

from bpc_fno.models.bpc_fno_a import BPC_FNO_A

logger = logging.getLogger(__name__)


# ======================================================================
# Standalone metric functions
# ======================================================================


def relative_l2_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample relative L2 error: ||pred - target||_2 / ||target||_2.

    Args:
        pred: Predicted tensor of arbitrary shape with batch dimension first.
        target: Ground-truth tensor of the same shape.

    Returns:
        Tensor of shape ``(B,)`` with per-sample relative errors.
    """
    diff = (pred - target).reshape(pred.shape[0], -1)
    tgt_flat = target.reshape(target.shape[0], -1)
    numer = torch.linalg.norm(diff, dim=1)
    denom = torch.linalg.norm(tgt_flat, dim=1).clamp(min=1e-12)
    return numer / denom


def pearson_correlation(
    pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Per-channel Pearson correlation between predicted and true volumes.

    Args:
        pred: ``(B, C, N, N, N)``
        target: ``(B, C, N, N, N)``

    Returns:
        Tensor of shape ``(B, C)`` with Pearson R values.
    """
    B, C = pred.shape[:2]
    # Flatten spatial dims
    p = pred.reshape(B, C, -1)   # (B, C, M)
    t = target.reshape(B, C, -1)

    # Centre
    p_mean = p.mean(dim=2, keepdim=True)
    t_mean = t.mean(dim=2, keepdim=True)
    p_c = p - p_mean
    t_c = t - t_mean

    # Covariance and standard deviations
    cov = (p_c * t_c).sum(dim=2)  # (B, C)
    p_std = torch.linalg.norm(p_c, dim=2).clamp(min=1e-12)
    t_std = torch.linalg.norm(t_c, dim=2).clamp(min=1e-12)

    return cov / (p_std * t_std)


def physics_residual_metric(
    J_i: torch.Tensor, voxel_size_cm: float
) -> torch.Tensor:
    """Mean absolute divergence of J_i (should be ~0 in resting tissue).

    Uses second-order central finite differences, consistent with
    :class:`bpc_fno.physics.monodomain_loss.MonodomainPDELoss`.

    Args:
        J_i: ``(B, 3, N, N, N)`` current density field.
        voxel_size_cm: Spatial voxel size in centimetres.

    Returns:
        Scalar tensor — mean |div(J)| over all voxels and batch elements.
    """
    inv_2h = 1.0 / (2.0 * voxel_size_cm)

    dJx_dx = torch.zeros_like(J_i[:, 0:1])
    dJx_dx[:, :, 1:-1, :, :] = (
        J_i[:, 0:1, 2:, :, :] - J_i[:, 0:1, :-2, :, :]
    ) * inv_2h

    dJy_dy = torch.zeros_like(J_i[:, 0:1])
    dJy_dy[:, :, :, 1:-1, :] = (
        J_i[:, 1:2, :, 2:, :] - J_i[:, 1:2, :, :-2, :]
    ) * inv_2h

    dJz_dz = torch.zeros_like(J_i[:, 0:1])
    dJz_dz[:, :, :, :, 1:-1] = (
        J_i[:, 2:3, :, :, 2:] - J_i[:, 2:3, :, :, :-2]
    ) * inv_2h

    div_J = dJx_dx + dJy_dy + dJz_dz  # (B, 1, N, N, N)
    return div_J.abs().mean()


def uq_coverage(
    J_i_samples: torch.Tensor,
    J_i_true: torch.Tensor,
    alpha: float = 0.95,
) -> float:
    """Fraction of true voxel values within the alpha credible interval.

    Args:
        J_i_samples: ``(N_samples, B, 3, N, N, N)`` posterior draws.
        J_i_true: ``(B, 3, N, N, N)`` ground truth.
        alpha: Credible-interval probability (default 0.95).

    Returns:
        Empirical coverage as a float in ``[0, 1]``.
    """
    tail = (1.0 - alpha) / 2.0
    n_samples = J_i_samples.shape[0]

    # Quantile indices (clamp to valid range)
    lo_idx = max(int(tail * n_samples), 0)
    hi_idx = min(int((1.0 - tail) * n_samples), n_samples - 1)

    sorted_samples, _ = J_i_samples.sort(dim=0)  # sort along sample axis
    lo = sorted_samples[lo_idx]   # (B, 3, N, N, N)
    hi = sorted_samples[hi_idx]   # (B, 3, N, N, N)

    within = (J_i_true >= lo) & (J_i_true <= hi)
    return float(within.float().mean().item())


def uq_sharpness(J_i_samples: torch.Tensor) -> torch.Tensor:
    """Mean width of the 95 % credible interval across all voxels.

    Smaller is better (sharper posteriors), given adequate coverage.

    Args:
        J_i_samples: ``(N_samples, B, 3, N, N, N)`` posterior draws.

    Returns:
        Scalar tensor — mean interval width.
    """
    sorted_samples, _ = J_i_samples.sort(dim=0)
    n = J_i_samples.shape[0]
    lo_idx = max(int(0.025 * n), 0)
    hi_idx = min(int(0.975 * n), n - 1)
    widths = sorted_samples[hi_idx] - sorted_samples[lo_idx]
    return widths.mean()


def activation_time_error(
    V_m_pred: torch.Tensor,
    V_m_true: torch.Tensor,
    threshold_mV: float = -20.0,
) -> torch.Tensor:
    """Mean absolute error of activation times.

    Activation time is defined as the first temporal index at which V_m
    crosses *threshold_mV* (from below).  Voxels that never activate are
    excluded from the error.

    Args:
        V_m_pred: ``(B, T, N, N, N)`` predicted transmembrane potentials.
        V_m_true: ``(B, T, N, N, N)`` ground-truth potentials.
        threshold_mV: Activation threshold in millivolts.

    Returns:
        Scalar tensor — MAE of activation times (in time-step units).
    """

    def _first_crossing(V: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (activation_time, mask) tensors.

        activation_time: ``(B, N, N, N)``  index of first threshold crossing.
        mask: ``(B, N, N, N)``  True where voxel activates at some timestep.
        """
        above = V >= threshold_mV  # (B, T, N, N, N)
        # Argmax along T returns the first True (maximum in a bool tensor)
        # but gives 0 when no crossing exists; we differentiate with `any`.
        activated = above.any(dim=1)               # (B, N, N, N)
        at = above.to(torch.uint8).argmax(dim=1)   # (B, N, N, N)
        return at.float(), activated

    at_pred, mask_pred = _first_crossing(V_m_pred)
    at_true, mask_true = _first_crossing(V_m_true)

    # Only evaluate where *both* predictions and ground truth activate
    valid = mask_pred & mask_true
    if valid.sum() == 0:
        return torch.tensor(0.0, device=V_m_pred.device)

    errors = (at_pred - at_true).abs()
    return errors[valid].mean()


# ======================================================================
# MetricsComputer — convenience wrapper
# ======================================================================


class MetricsComputer:
    """Compute and aggregate all evaluation metrics for a BPC-FNO model.

    Parameters
    ----------
    config:
        Hydra/OmegaConf configuration.  Reads:
        - ``evaluation.voxel_size_cm``  (default 0.1)
        - ``evaluation.uq_alpha``       (default 0.95)
        - ``evaluation.uq_n_samples``   (default 50)
    """

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        eval_cfg = config.get("evaluation", {})
        self.voxel_size_cm: float = float(eval_cfg.get("voxel_size_cm", 0.1))
        self.uq_alpha: float = float(eval_cfg.get("uq_alpha", 0.95))
        self.uq_n_samples: int = int(eval_cfg.get("uq_n_samples", 50))

    @torch.no_grad()
    def compute_all(
        self,
        model_output: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        model: BPC_FNO_A,
    ) -> dict[str, float]:
        """Compute the full suite of metrics.

        Args:
            model_output: Dict from ``model.forward(batch)`` containing at
                least ``J_i_hat``, ``B_pred``.
            batch: Original data batch with keys ``J_i``, ``B_mig``,
                ``geometry``.
            model: The model instance (used for posterior sampling).

        Returns:
            Flat dict of metric names to scalar values, suitable for
            logger / wandb.
        """
        J_i_true: torch.Tensor = batch["J_i"]
        B_true: torch.Tensor = batch["B_mig"]
        geometry: torch.Tensor = batch["geometry"]
        J_i_hat: torch.Tensor = model_output["J_i_hat"]
        B_pred: torch.Tensor = model_output["B_pred"]

        metrics: dict[str, float] = {}

        # -- Relative L2 errors --
        metrics["rel_l2_J_i"] = float(
            relative_l2_error(J_i_hat, J_i_true).mean().item()
        )
        metrics["rel_l2_B"] = float(
            relative_l2_error(B_pred, B_true).mean().item()
        )

        # -- Pearson R for J_i (mean over channels and batch) --
        metrics["pearson_r_J_i"] = float(
            pearson_correlation(J_i_hat, J_i_true).mean().item()
        )

        # -- Physics residual --
        metrics["physics_residual"] = float(
            physics_residual_metric(J_i_hat, self.voxel_size_cm).item()
        )

        # -- UQ metrics via posterior sampling --
        try:
            recon = model.reconstruct(
                B_true, geometry, n_samples=self.uq_n_samples
            )
            # Collect individual samples for UQ analysis
            mu = recon["mu"]
            log_var = recon["log_var"]
            std = (0.5 * log_var).exp()

            samples_list: list[torch.Tensor] = []
            for _ in range(self.uq_n_samples):
                eps = torch.randn_like(std)
                z = mu + eps * std
                J_sample = model.vae_decoder.decode(z, geometry)
                samples_list.append(J_sample)

            J_i_samples = torch.stack(samples_list, dim=0)  # (S, B, 3, N, N, N)

            metrics["uq_coverage"] = uq_coverage(
                J_i_samples, J_i_true, alpha=self.uq_alpha
            )
            metrics["uq_sharpness"] = float(
                uq_sharpness(J_i_samples).item()
            )
        except Exception:
            logger.warning(
                "UQ metric computation failed; skipping.", exc_info=True
            )
            metrics["uq_coverage"] = float("nan")
            metrics["uq_sharpness"] = float("nan")

        return metrics
