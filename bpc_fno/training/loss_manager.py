"""Weighted multi-term loss manager for BPC-FNO Architecture A training.

Handles both Phase 1 (forward-only) and Phase 2 (joint forward + inverse)
losses with configurable weighting and scheduling of the physics loss term.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from bpc_fno.models.interfaces import ForwardOperatorInterface
from bpc_fno.physics.consistency_loss import ForwardConsistencyLoss
from bpc_fno.physics.monodomain_loss import MonodomainPDELoss

logger = logging.getLogger(__name__)


class LossManager:
    """Compute and schedule all loss terms for Architecture A.

    Parameters
    ----------
    config:
        OmegaConf configuration.  Expected keys under ``config.loss``:

        - ``lambda_KL`` (float): weight for the VAE KL-divergence term.
        - ``lambda_physics_init`` (float): initial weight for the physics loss.
        - ``lambda_physics_final`` (float): cap for the physics loss weight.
        - ``lambda_physics_doubling_epochs`` (int): epochs between doublings.
        - ``lambda_consistency`` (float): weight for the forward-consistency loss.
        - ``consistency_start_epoch`` (int): epoch at which L_consistency turns on.
        - ``voxel_size_cm`` (float): passed through to MonodomainPDELoss.
        - ``n_collocation_points`` (int): collocation points for physics residual.
    """

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        loss_cfg = config.loss

        # Lambda weights
        self.lambda_KL: float = float(loss_cfg.get("lambda_KL", 1e-4))
        self.lambda_physics_init: float = float(
            loss_cfg.get("lambda_physics_init", 1e-3)
        )
        self.lambda_physics_final: float = float(
            loss_cfg.get("lambda_physics_final", 1.0)
        )
        self.lambda_physics_doubling_epochs: int = int(
            loss_cfg.get("lambda_physics_doubling_epochs", 10)
        )
        self.lambda_consistency: float = float(
            loss_cfg.get("lambda_consistency", 0.1)
        )
        self.consistency_start_epoch: int = int(
            loss_cfg.get("consistency_start_epoch", 5)
        )

        # Current epoch (updated externally by the trainer)
        self.current_epoch: int = 0

        # Loss function instances
        physics_cfg = DictConfig(
            {
                "voxel_size_cm": float(loss_cfg.get("voxel_size_cm", 0.1)),
                "n_collocation_points": int(
                    loss_cfg.get("n_collocation_points", 1024)
                ),
            }
        )
        self.monodomain_loss = MonodomainPDELoss(physics_cfg)
        self.consistency_loss = ForwardConsistencyLoss()

    # ------------------------------------------------------------------
    # Phase 1: forward-only
    # ------------------------------------------------------------------

    def compute_phase1_loss(
        self,
        model_output: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute Phase 1 (forward-only) loss.

        Parameters
        ----------
        model_output:
            Must contain ``'B_pred'`` — predicted B-field ``(B, S, T)``.
        batch:
            Must contain ``'B_obs'`` — ground-truth B-field ``(B, S, T)``.

        Returns
        -------
        (total_loss, loss_dict)
            Scalar loss tensor and a dictionary of named loss values for logging.
        """
        L_data_forward = F.mse_loss(model_output["B_pred"], batch["B_obs"])

        loss_dict: dict[str, float] = {
            "L_data_forward": L_data_forward.item(),
            "total_loss": L_data_forward.item(),
        }
        return L_data_forward, loss_dict

    # ------------------------------------------------------------------
    # Phase 2: joint forward + inverse
    # ------------------------------------------------------------------

    def compute_phase2_loss(
        self,
        model_output: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        forward_model: ForwardOperatorInterface,
        epoch: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute Phase 2 (joint) loss with all terms.

        Parameters
        ----------
        model_output:
            Expected keys: ``'B_pred'``, ``'J_i_hat'``, ``'mu'``,
            ``'log_var'``.
        batch:
            Expected keys: ``'B_obs'``, ``'J_i'``, ``'geometry'``.
        forward_model:
            The forward PINO model, used for the consistency loss.
            Its weights must NOT be updated by the consistency loss --
            the trainer is responsible for zeroing forward-model gradients
            after backward.
        epoch:
            Current training epoch.

        Returns
        -------
        (total_loss, loss_dict)
            Scalar loss tensor and a dictionary of named loss values for logging.
        """
        loss_dict: dict[str, float] = {}

        # --- L_data_forward: MSE(B_pred, B_true) ---
        L_data_forward = F.mse_loss(model_output["B_pred"], batch["B_obs"])
        loss_dict["L_data_forward"] = L_data_forward.item()

        # --- L_data_recon: MSE(J_i_hat, J_i_true) ---
        L_data_recon = F.mse_loss(model_output["J_i_hat"], batch["J_i"])
        loss_dict["L_data_recon"] = L_data_recon.item()

        # --- L_KL: KL divergence ---
        mu = model_output["mu"]
        log_var = model_output["log_var"]
        L_KL = -0.5 * torch.mean(
            1.0 + log_var - mu.pow(2) - log_var.exp()
        )
        loss_dict["L_KL"] = L_KL.item()

        # --- L_physics: monodomain PDE residual ---
        lambda_physics = self.get_lambda_physics(epoch)
        geometry = batch["geometry"]
        myocardium_mask = geometry[:, 3:4, :, :, :]
        L_physics = self.monodomain_loss(
            model_output["J_i_hat"], geometry, myocardium_mask
        )
        loss_dict["L_physics"] = L_physics.item()
        loss_dict["lambda_physics"] = lambda_physics

        # --- L_consistency: forward-consistency (delayed start) ---
        L_consistency: torch.Tensor
        if epoch >= self.consistency_start_epoch:
            L_consistency = self.consistency_loss(
                forward_model,
                model_output["J_i_hat"],
                geometry,
                batch["B_obs"],
            )
        else:
            L_consistency = torch.tensor(
                0.0,
                device=model_output["B_pred"].device,
                dtype=model_output["B_pred"].dtype,
            )
        loss_dict["L_consistency"] = L_consistency.item()

        # --- Total loss ---
        L_total = (
            L_data_forward
            + L_data_recon
            + self.lambda_KL * L_KL
            + lambda_physics * L_physics
            + self.lambda_consistency * L_consistency
        )
        loss_dict["total_loss"] = L_total.item()

        return L_total, loss_dict

    # ------------------------------------------------------------------
    # Physics weight schedule
    # ------------------------------------------------------------------

    def get_lambda_physics(self, epoch: int) -> float:
        """Compute the physics loss weight for a given epoch.

        The weight starts at ``lambda_physics_init`` and doubles every
        ``lambda_physics_doubling_epochs``, capped at ``lambda_physics_final``.

        Parameters
        ----------
        epoch:
            Current training epoch.

        Returns
        -------
        float
            Physics loss weight for this epoch.
        """
        if self.lambda_physics_doubling_epochs <= 0:
            return self.lambda_physics_init

        n_doublings = epoch // self.lambda_physics_doubling_epochs
        weight = self.lambda_physics_init * (2.0 ** n_doublings)
        return min(weight, self.lambda_physics_final)
