"""Weighted multi-term loss manager for BPC-FNO Architecture A training.

Handles both Phase 1 (forward-only) and Phase 2 (joint forward + inverse)
losses with configurable weighting and scheduling of the physics loss term.

Self-contained: all loss computations (MSE, KL, physics residual,
consistency) are implemented directly here without external physics module
dependencies.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class LossManager:
    """Compute and schedule all loss terms for Architecture A.

    Parameters
    ----------
    config:
        OmegaConf configuration.  Expected keys under ``config.training``:

        - ``lambda_kl_init`` (float): weight for the VAE KL-divergence term.
        - ``lambda_physics_init`` (float): initial weight for the physics loss.
        - ``lambda_physics_final`` (float): cap for the physics loss weight.
        - ``lambda_physics_doubling_epochs`` (int): epochs between doublings.
        - ``lambda_consistency`` (float): weight for the forward-consistency loss.
        - ``lambda_consistency_start_epoch`` (int): epoch at which L_consistency turns on.
        - ``voxel_size_cm`` (float): voxel spacing for central-FD divergence.
    """

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        t = config.training

        # Lambda weights
        self.lambda_kl: float = float(t.get("lambda_kl_init", 1e-4))
        self.lambda_physics_init: float = float(
            t.get("lambda_physics_init", 1e-3)
        )
        self.lambda_physics_final: float = float(
            t.get("lambda_physics_final", 0.05)
        )
        self.lambda_physics_doubling_epochs: int = int(
            t.get("lambda_physics_doubling_epochs", 20)
        )
        self.lambda_consistency: float = float(
            t.get("lambda_consistency", 0.01)
        )
        self.consistency_start_epoch: int = int(
            t.get("lambda_consistency_start_epoch", 30)
        )
        self.voxel_size_cm: float = float(t.get("voxel_size_cm", 0.1))

        # Current epoch (updated externally by the trainer)
        self.current_epoch: int = 0

    # ------------------------------------------------------------------
    # Individual loss terms
    # ------------------------------------------------------------------

    @staticmethod
    def _get_forward_target(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the clean B-field target when available, else fall back safely."""
        for key in ("B_true", "B_mig_clean", "B_mig", "B_obs"):
            if key in batch:
                return batch[key]
        raise KeyError(
            "Batch must contain one of: 'B_true', 'B_mig_clean', 'B_mig', or 'B_obs'."
        )

    @staticmethod
    def _get_observed_B(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the observed/noisy B-field used for inverse inference paths."""
        for key in ("B_obs", "B_mig", "B_true", "B_mig_clean"):
            if key in batch:
                return batch[key]
        raise KeyError(
            "Batch must contain one of: 'B_obs', 'B_mig', 'B_true', or 'B_mig_clean'."
        )

    @staticmethod
    def forward_loss(B_pred: torch.Tensor, B_true: torch.Tensor) -> torch.Tensor:
        """MSE between predicted and true B-field.

        Args:
            B_pred: (B, N_sensors) predicted sensor measurements.
            B_true: (B, N_sensors) ground-truth sensor measurements.

        Returns:
            Scalar MSE loss.
        """
        return F.mse_loss(B_pred, B_true)

    @staticmethod
    def recon_loss(J_i_hat: torch.Tensor, J_i_true: torch.Tensor) -> torch.Tensor:
        """MSE between reconstructed and true current density.

        Args:
            J_i_hat: (B, 3, N, N, N) reconstructed current density.
            J_i_true: (B, 3, N, N, N) ground-truth current density.

        Returns:
            Scalar MSE loss.
        """
        return F.mse_loss(J_i_hat, J_i_true)

    @staticmethod
    def kl_loss(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """KL divergence between N(mu, sigma^2) and N(0, 1).

        KL = -0.5 * mean(1 + log_var - mu^2 - exp(log_var))

        Args:
            mu: (B, D) latent mean.
            log_var: (B, D) log-variance.

        Returns:
            Scalar KL loss.
        """
        return -0.5 * torch.mean(1.0 + log_var - mu.pow(2) - log_var.exp())

    @staticmethod
    def physics_residual_loss(
        J_i_hat: torch.Tensor, voxel_size_cm: float
    ) -> torch.Tensor:
        """Physics residual: mean(div(J_i)^2) via central finite differences.

        Current-conservation loss (div J ~ 0 in quasi-static tissue).
        Handles both (B, 3, N, N, N) and (B, 3, T, N, N, N).

        Args:
            J_i_hat: (B, 3, N, N, N) or (B, 3, T, N, N, N).
            voxel_size_cm: Spatial step in cm.
        """
        if J_i_hat.ndim == 6:
            # Time-series: merge B and T dims
            B, C, T, N1, N2, N3 = J_i_hat.shape
            J_flat = J_i_hat.permute(0, 2, 1, 3, 4, 5).reshape(B * T, C, N1, N2, N3)
            return LossManager._physics_residual_3d(J_flat, voxel_size_cm)
        return LossManager._physics_residual_3d(J_i_hat, voxel_size_cm)

    @staticmethod
    def _physics_residual_3d(
        J_i: torch.Tensor, voxel_size_cm: float
    ) -> torch.Tensor:
        """Compute div(J)^2 loss on (B, 3, N, N, N) tensors."""
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

        div_J = dJx_dx + dJy_dy + dJz_dz
        return (div_J ** 2).mean()

    @staticmethod
    def consistency_loss(
        B_check: torch.Tensor, B_obs: torch.Tensor
    ) -> torch.Tensor:
        """MSE between re-predicted B-field and observed B-field.

        Args:
            B_check: (B, N_sensors) B-field predicted from J_i_hat.
            B_obs: (B, N_sensors) observed B-field measurements.

        Returns:
            Scalar MSE loss.
        """
        return F.mse_loss(B_check, B_obs)

    # ------------------------------------------------------------------
    # Phase 1: forward-only
    # ------------------------------------------------------------------

    def compute_phase1(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor | float]:
        """Compute Phase 1 (forward-only) losses.

        Parameters
        ----------
        outputs:
            Must contain ``'B_pred'`` — predicted B-field.
        batch:
            Must contain ``'B_obs'`` — ground-truth B-field.

        Returns
        -------
        dict
            Contains individual loss values and ``'total'`` scalar tensor.
        """
        B_target = self._get_forward_target(batch)
        L_forward = self.forward_loss(outputs["B_pred"], B_target)

        return {
            "L_data_forward": L_forward,
            "total": L_forward,
        }

    # ------------------------------------------------------------------
    # Phase 2: joint forward + inverse
    # ------------------------------------------------------------------

    def compute_phase2(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        model: Any,
        epoch: int,
        cfg: DictConfig,
    ) -> dict[str, torch.Tensor | float]:
        """Compute Phase 2 (joint) losses with all terms.

        The consistency loss uses ``torch.no_grad()`` on the forward model
        call (so the forward model's weights do not receive gradients from
        this path) but does NOT detach ``J_i_hat`` (so the decoder that
        produced it still gets gradients).

        Parameters
        ----------
        outputs:
            Expected keys: ``'B_pred'``, ``'J_i_hat'``, ``'mu'``,
            ``'log_var'``.
        batch:
            Expected keys: ``'B_obs'``, ``'J_i'``, ``'geometry'``.
        model:
            The full BPC-FNO model.  Its ``forward_pino.predict_B`` is
            called with forward-model parameters temporarily frozen for
            the consistency term.
        epoch:
            Current training epoch.
        cfg:
            OmegaConf configuration (same as ``self.config``; passed
            explicitly so the caller can override).

        Returns
        -------
        dict
            Contains individual loss values (as tensors or floats) and
            ``'total'`` scalar tensor for backpropagation.
        """
        result: dict[str, torch.Tensor | float] = {}

        # --- L_data_forward: MSE(B_pred, B_true) ---
        B_target = self._get_forward_target(batch)
        B_obs = self._get_observed_B(batch)
        L_forward = self.forward_loss(outputs["B_pred"], B_target)
        result["L_data_forward"] = L_forward

        # --- L_data_recon: MSE(J_i_hat, J_i_true) ---
        L_recon = self.recon_loss(outputs["J_i_hat"], batch["J_i"])
        result["L_data_recon"] = L_recon

        # --- L_KL: KL divergence ---
        L_kl = self.kl_loss(outputs["mu"], outputs["log_var"])
        result["L_KL"] = L_kl

        # --- L_physics: div(J_i)^2 residual ---
        lambda_physics = self.get_lambda_physics(epoch, cfg)
        L_physics = self.physics_residual_loss(
            outputs["J_i_hat"], self.voxel_size_cm
        )
        result["L_physics"] = L_physics
        result["lambda_physics"] = lambda_physics

        # --- L_consistency: forward-consistency (delayed start) ---
        if epoch >= self.consistency_start_epoch:
            J_i_hat = outputs["J_i_hat"]  # keep in computation graph
            geometry = batch["geometry"]

            # Temporarily freeze forward-model parameters so they receive
            # no gradients from this path, but the computation graph still
            # connects through J_i_hat so the decoder gets gradients.
            fwd_params_grad_state: list[tuple[torch.nn.Parameter, bool]] = []
            for p in model.forward_pino.parameters():
                fwd_params_grad_state.append((p, p.requires_grad))
                p.requires_grad_(False)

            B_check = model.forward_pino.predict_B(J_i_hat, geometry)

            # Restore requires_grad on forward-model parameters
            for p, grad_flag in fwd_params_grad_state:
                p.requires_grad_(grad_flag)

            L_consistency = self.consistency_loss(B_check, B_obs)
        else:
            L_consistency = torch.tensor(
                0.0,
                device=outputs["B_pred"].device,
                dtype=outputs["B_pred"].dtype,
            )
        result["L_consistency"] = L_consistency

        # --- Total loss ---
        L_total = (
            L_forward
            + L_recon
            + self.lambda_kl * L_kl
            + lambda_physics * L_physics
            + self.lambda_consistency * L_consistency
        )
        result["total"] = L_total

        return result

    # ------------------------------------------------------------------
    # Physics weight schedule
    # ------------------------------------------------------------------

    @staticmethod
    def get_lambda_physics(epoch: int, cfg: DictConfig) -> float:
        """Compute the physics loss weight for a given epoch.

        The weight starts at ``lambda_physics_init`` and doubles every
        ``lambda_physics_doubling_epochs``, capped at
        ``lambda_physics_final``.

        Parameters
        ----------
        epoch:
            Current training epoch.
        cfg:
            OmegaConf configuration with ``training`` sub-key.

        Returns
        -------
        float
            Physics loss weight for this epoch.
        """
        t = cfg.training
        init = float(t.get("lambda_physics_init", 1e-3))
        final = float(t.get("lambda_physics_final", 0.05))
        doubling = int(t.get("lambda_physics_doubling_epochs", 20))

        if doubling <= 0:
            return init

        n_doublings = epoch // doubling
        weight = init * (2.0 ** n_doublings)
        return min(weight, final)
