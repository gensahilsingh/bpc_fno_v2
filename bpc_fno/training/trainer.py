"""PyTorch Lightning module for BPC-FNO Architecture A two-phase training.

Phase 1 ("forward"):
    Train only the forward PINO operator (J_i -> B_pred) using supervised
    MSE loss against ground-truth B-field measurements.

Phase 2 ("joint"):
    Train the full pipeline (inverse encoder + decoder + forward PINO)
    with data-fidelity, KL, physics, and forward-consistency losses.
    The forward model's weights are protected from consistency-loss
    gradients by zeroing those gradients after backward.
"""

from __future__ import annotations

import logging
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from bpc_fno.training.loss_manager import LossManager
from bpc_fno.training.lr_schedule import CosineWarmupScheduler

logger = logging.getLogger(__name__)


class BPCFNOTrainer(pl.LightningModule):
    """Lightning module orchestrating two-phase Architecture A training.

    Parameters
    ----------
    model:
        The full BPC-FNO Architecture A model.  Must expose:

        - ``forward_pino`` — the forward PINO sub-module.
        - ``forward_only(J_i, geometry) -> dict`` — Phase 1 forward pass.
        - ``forward(batch) -> dict`` — Phase 2 full forward pass.
        - ``get_parameter_groups(lr) -> list[dict]`` — optimizer param groups.
        - ``get_forward_param_names() -> set[str]`` — names of forward-PINO
          parameters (used to block consistency-loss gradients).
    config:
        OmegaConf configuration.
    """

    def __init__(self, model: Any, config: DictConfig) -> None:
        super().__init__()
        self.model = model
        self.config = config
        self.loss_manager = LossManager(config)

        # Current training phase: 'forward' or 'joint'
        self.phase: str = "forward"

        # Cache forward-model parameter names for gradient blocking
        self._forward_param_names: set[str] | None = None

        # Store last output for diagnostics callback
        self._last_output: dict[str, torch.Tensor] = {}

        # Store last validation physics residual for the callback
        self._last_val_physics_residual: float | None = None

        # Disable automatic optimisation so we can manually handle
        # gradient blocking for the consistency loss.
        self.automatic_optimization = False

    # ------------------------------------------------------------------
    # Phase management
    # ------------------------------------------------------------------

    def set_phase(self, phase: str) -> None:
        """Switch between ``'forward'`` and ``'joint'`` training phases.

        In the ``'forward'`` phase, only the forward PINO parameters
        require gradients.  In the ``'joint'`` phase, all parameters
        are trainable.

        Parameters
        ----------
        phase:
            ``'forward'`` or ``'joint'``.
        """
        if phase not in ("forward", "joint"):
            raise ValueError(f"Unknown phase '{phase}'; expected 'forward' or 'joint'.")

        self.phase = phase
        logger.info("Training phase set to '%s'.", phase)

        if phase == "forward":
            # Freeze everything except forward_pino
            forward_param_names = self._get_forward_param_names()
            for name, param in self.model.named_parameters():
                param.requires_grad = name in forward_param_names
        else:
            # Unfreeze all parameters
            for param in self.model.parameters():
                param.requires_grad = True

    def _get_forward_param_names(self) -> set[str]:
        """Return the set of parameter names belonging to the forward PINO."""
        if self._forward_param_names is None:
            if hasattr(self.model, "get_forward_param_names"):
                self._forward_param_names = self.model.get_forward_param_names()
            else:
                # Fallback: walk forward_pino sub-module
                prefix = "forward_pino."
                self._forward_param_names = {
                    name
                    for name, _ in self.model.named_parameters()
                    if name.startswith(prefix)
                }
        return self._forward_param_names

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Execute one training step (manual optimisation).

        Parameters
        ----------
        batch:
            Dictionary with at least ``'J_i'``, ``'geometry'``, ``'B_obs'``.
        batch_idx:
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            The total loss for this step.
        """
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()

        if self.phase == "forward":
            B_pred = self.model.forward_only(batch["J_i"], batch["geometry"])
            output = {"B_pred": B_pred}
            result = self.loss_manager.compute_phase1(output, batch)
            loss = result["total"]
            loss_dict = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in result.items()}
        else:
            output = self.model(batch)
            result = self.loss_manager.compute_phase2(
                output, batch, self.model, self.current_epoch, self.config
            )
            loss = result["total"]
            loss_dict = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in result.items()}

        # Store output for diagnostics callback
        self._last_output = output

        # Manual backward
        optimizer.zero_grad()
        self.manual_backward(loss)

        # --- Gradient blocking for consistency loss ---
        # In joint phase, zero out gradients on forward-model params that
        # were contributed by the consistency loss path.  Since
        # consistency loss flows through forward_pino.predict_B, those
        # gradients accumulate on forward_pino params.  We block them
        # to prevent the consistency loss from updating forward weights.
        if self.phase == "joint" and self.current_epoch >= self.loss_manager.consistency_start_epoch:
            forward_param_names = self._get_forward_param_names()
            for name, param in self.model.named_parameters():
                if name in forward_param_names and param.grad is not None:
                    param.grad.zero_()

        # Gradient clipping
        grad_clip_norm: float | None = self.config.training.get(
            "grad_clip_norm", None
        )
        if grad_clip_norm is not None and grad_clip_norm > 0:
            self.clip_gradients(
                optimizer, gradient_clip_val=grad_clip_norm, gradient_clip_algorithm="norm"
            )

        optimizer.step()

        # Step the LR scheduler
        if scheduler is not None:
            scheduler.step()

        # Logging
        for key, value in loss_dict.items():
            self.log(
                f"train/{key}",
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=(key == "total_loss"),
                batch_size=batch["B_obs"].shape[0],
            )

        # Log learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        self.log("train/lr", current_lr, on_step=True, on_epoch=False)

        return loss

    # ------------------------------------------------------------------
    # Validation step
    # ------------------------------------------------------------------

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> None:
        """Compute and log validation metrics for the current phase.

        Parameters
        ----------
        batch:
            Dictionary with at least ``'J_i'``, ``'geometry'``, ``'B_obs'``.
        batch_idx:
            Index of the current batch.
        """
        batch_size = batch["B_obs"].shape[0]

        if self.phase == "forward":
            B_pred = self.model.forward_only(batch["J_i"], batch["geometry"])
            output = {"B_pred": B_pred}
            result = self.loss_manager.compute_phase1(output, batch)
            loss_dict = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in result.items()}

            # Relative L2 error for forward B-field
            B_true = batch["B_obs"]
            l2_error = torch.norm(B_pred - B_true) / (torch.norm(B_true) + 1e-8)
            self.log(
                "val/forward_l2",
                l2_error.item(),
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size,
            )

        else:
            output = self.model(batch)
            result = self.loss_manager.compute_phase2(
                output, batch, self.model, self.current_epoch, self.config
            )
            loss_dict = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in result.items()}

            # Reconstruction loss for J_i
            recon_loss = F.mse_loss(output["J_i_hat"], batch["J_i"])
            self.log(
                "val/reconstruction_loss",
                recon_loss.item(),
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size,
            )

            # Relative L2 for J_i
            J_pred = output["J_i_hat"]
            J_true = batch["J_i"]
            l2_J = torch.norm(J_pred - J_true) / (torch.norm(J_true) + 1e-8)
            self.log(
                "val/J_i_l2",
                l2_J.item(),
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
            )

            # Physics residual for the callback
            voxel_size = float(self.config.training.get("voxel_size_cm", 0.1))
            residual = LossManager.physics_residual_loss(
                output["J_i_hat"], voxel_size
            )
            self._last_val_physics_residual = residual.item()

        # Log all loss terms
        for key, value in loss_dict.items():
            self.log(
                f"val/{key}",
                value,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
            )

    # ------------------------------------------------------------------
    # Optimizer and scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> dict[str, Any]:
        """Set up AdamW optimizer with CosineWarmupScheduler.

        Returns
        -------
        dict
            Dictionary with ``'optimizer'`` and ``'lr_scheduler'`` entries
            compatible with Lightning's manual optimisation mode.
        """
        train_cfg = self.config.training

        # Learning rate
        lr: float = float(train_cfg.get("lr", 1e-3))
        weight_decay: float = float(train_cfg.get("weight_decay", 1e-4))

        # Parameter groups (potentially different LRs per component)
        if hasattr(self.model, "get_parameter_groups"):
            param_groups = self.model.get_parameter_groups(lr)
        else:
            param_groups = [{"params": self.model.parameters(), "lr": lr}]

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
        )

        # LR schedule
        warmup_steps: int = int(train_cfg.get("warmup_steps", 500))
        total_steps: int = int(train_cfg.get("total_steps", 50_000))
        lr_final: float = float(train_cfg.get("lr_final", 1e-6))

        scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            lr_init=lr,
            lr_final=lr_final,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    # ------------------------------------------------------------------
    # Epoch-end hooks
    # ------------------------------------------------------------------

    def on_train_epoch_end(self) -> None:
        """Log epoch-level metrics and update loss schedules."""
        # Update physics loss weight in the loss manager
        self.loss_manager.current_epoch = self.current_epoch

        # Log the current physics weight
        lambda_phys = LossManager.get_lambda_physics(self.current_epoch, self.config)
        self.log("train/lambda_physics", lambda_phys, on_epoch=True)

        logger.info(
            "Epoch %d complete | phase=%s | lambda_physics=%.6e",
            self.current_epoch,
            self.phase,
            lambda_phys,
        )
