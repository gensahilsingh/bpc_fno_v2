"""Custom PyTorch Lightning callbacks for BPC-FNO training.

Provides diagnostics logging, physics residual tracking, phase-aware
checkpointing, and phase-aware early stopping.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from bpc_fno.physics.monodomain_loss import MonodomainPDELoss
from bpc_fno.utils.checkpointing import (
    load_checkpoint,
    save_checkpoint,
)
from bpc_fno.utils.diagnostics import DiagnosticsLogger

logger = logging.getLogger(__name__)


# ====================================================================== #
# DiagnosticsCallback
# ====================================================================== #


class DiagnosticsCallback(pl.Callback):
    """Log stage-boundary tensor statistics every *log_interval* steps.

    Uses :class:`~bpc_fno.utils.diagnostics.DiagnosticsLogger` under the
    hood to compute and (optionally) push tensor statistics to W&B.

    Parameters
    ----------
    log_interval:
        Log diagnostics every *log_interval* training steps.
    use_wandb:
        Forward to :class:`DiagnosticsLogger`.
    """

    def __init__(
        self,
        log_interval: int = 100,
        use_wandb: bool = False,
    ) -> None:
        super().__init__()
        self.log_interval = log_interval
        self.diag_logger = DiagnosticsLogger(use_wandb=use_wandb)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Log intermediate tensor statistics at the configured interval."""
        step = trainer.global_step
        if step % self.log_interval != 0:
            return

        # Collect tensors from the most recent model output if available.
        # The trainer module should store them in `self._last_output`.
        last_output: dict[str, torch.Tensor] | None = getattr(
            pl_module, "_last_output", None
        )
        if last_output is None:
            return

        tensors_to_log: dict[str, torch.Tensor] = {}
        for key in ("B_pred", "J_i_hat", "mu", "log_var"):
            if key in last_output and isinstance(last_output[key], torch.Tensor):
                tensors_to_log[key] = last_output[key]

        if tensors_to_log:
            self.diag_logger.log_stage_boundary(
                stage_name="train_batch_end",
                tensors=tensors_to_log,
                step=step,
            )


# ====================================================================== #
# PhysicsResidualCallback
# ====================================================================== #


class PhysicsResidualCallback(pl.Callback):
    """Compute and log the physics residual on the validation set.

    Evaluates the monodomain divergence residual every *eval_every_n_epochs*
    epochs and tracks whether it is converging.

    Parameters
    ----------
    config:
        OmegaConf configuration (forwarded to :class:`MonodomainPDELoss`).
    eval_every_n_epochs:
        Frequency (in epochs) at which to evaluate the residual.
    convergence_threshold:
        If the mean residual drops below this value the model is considered
        converged from a physics standpoint.
    """

    def __init__(
        self,
        config: DictConfig,
        eval_every_n_epochs: int = 5,
        convergence_threshold: float = 1e-4,
    ) -> None:
        super().__init__()
        self.eval_every_n_epochs = eval_every_n_epochs
        self.convergence_threshold = convergence_threshold

        physics_cfg = DictConfig(
            {
                "voxel_size_cm": float(config.loss.get("voxel_size_cm", 0.1)),
                "n_collocation_points": int(
                    config.loss.get("n_collocation_points", 1024)
                ),
            }
        )
        self.physics_loss = MonodomainPDELoss(physics_cfg)
        self.residual_history: list[float] = []
        self.converged: bool = False

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Evaluate and log the physics residual at the configured interval."""
        epoch = trainer.current_epoch
        if epoch % self.eval_every_n_epochs != 0:
            return

        # Accumulate residuals from the last validation outputs stored on the module.
        last_val_residual: float | None = getattr(
            pl_module, "_last_val_physics_residual", None
        )
        if last_val_residual is not None:
            self.residual_history.append(last_val_residual)
            self.converged = last_val_residual < self.convergence_threshold

            pl_module.log(
                "val/physics_residual",
                last_val_residual,
                prog_bar=False,
            )
            pl_module.log(
                "val/physics_converged",
                float(self.converged),
                prog_bar=False,
            )
            logger.info(
                "Physics residual at epoch %d: %.6e (converged=%s)",
                epoch,
                last_val_residual,
                self.converged,
            )


# ====================================================================== #
# CheckpointCallback
# ====================================================================== #


class CheckpointCallback(pl.Callback):
    """Phase-aware checkpointing wrapper.

    Saves the best model per training phase (``'forward'`` / ``'joint'``)
    using the :mod:`bpc_fno.utils.checkpointing` utilities.

    Parameters
    ----------
    checkpoint_dir:
        Directory in which checkpoints are stored.
    monitor_forward:
        Metric to monitor during the forward phase.
    monitor_joint:
        Metric to monitor during the joint phase.
    mode:
        ``'min'`` or ``'max'`` — how to determine the "best" model.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        monitor_forward: str = "val/forward_l2",
        monitor_joint: str = "val/reconstruction_loss",
        mode: str = "min",
    ) -> None:
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.monitor_forward = monitor_forward
        self.monitor_joint = monitor_joint
        self.mode = mode

        self.best_forward_score: float | None = None
        self.best_joint_score: float | None = None

    # ------------------------------------------------------------------

    def _is_better(self, current: float, best: float | None) -> bool:
        if best is None:
            return True
        if self.mode == "min":
            return current < best
        return current > best

    # ------------------------------------------------------------------

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Save a checkpoint if the monitored metric improved."""
        phase: str = getattr(pl_module, "phase", "forward")
        metrics = trainer.callback_metrics

        if phase == "forward":
            monitor_key = self.monitor_forward
            best_ref = self.best_forward_score
        else:
            monitor_key = self.monitor_joint
            best_ref = self.best_joint_score

        if monitor_key not in metrics:
            return

        current = float(metrics[monitor_key].item())
        if not self._is_better(current, best_ref):
            return

        # Update best score
        if phase == "forward":
            self.best_forward_score = current
        else:
            self.best_joint_score = current

        # Persist checkpoint
        ckpt_path = self.checkpoint_dir / f"best_{phase}.ckpt"
        metric_dict = {k: float(v.item()) for k, v in metrics.items() if isinstance(v, torch.Tensor)}
        optimizers = trainer.optimizers
        optimizer = optimizers[0] if optimizers else None

        save_checkpoint(
            model=pl_module,
            optimizer=optimizer,
            epoch=trainer.current_epoch,
            phase=phase,
            metrics=metric_dict,
            path=ckpt_path,
        )
        logger.info(
            "Best %s checkpoint saved: %s=%.6f -> %s",
            phase,
            monitor_key,
            current,
            ckpt_path,
        )

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def save_checkpoint_manual(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        path: str | Path,
    ) -> None:
        """Manually trigger a checkpoint save."""
        phase: str = getattr(pl_module, "phase", "forward")
        metrics = {
            k: float(v.item())
            for k, v in trainer.callback_metrics.items()
            if isinstance(v, torch.Tensor)
        }
        optimizers = trainer.optimizers
        optimizer = optimizers[0] if optimizers else None

        save_checkpoint(
            model=pl_module,
            optimizer=optimizer,
            epoch=trainer.current_epoch,
            phase=phase,
            metrics=metrics,
            path=path,
        )

    def load_checkpoint_into(
        self,
        pl_module: pl.LightningModule,
        path: str | Path,
        optimizer: torch.optim.Optimizer | None = None,
        strict: bool = True,
    ) -> dict[str, Any]:
        """Load a checkpoint into the given Lightning module."""
        return load_checkpoint(
            model=pl_module,
            optimizer=optimizer,
            path=path,
            strict=strict,
        )


# ====================================================================== #
# EarlyStoppingWithPhase
# ====================================================================== #


class EarlyStoppingWithPhase(pl.Callback):
    """Phase-aware early stopping for two-phase BPC-FNO training.

    Monitors different metrics and uses different patience values depending
    on whether the trainer is in the ``'forward'`` or ``'joint'`` phase.

    Parameters
    ----------
    monitor_forward:
        Metric to watch during Phase 1.
    patience_forward:
        Number of validation epochs with no improvement before stopping Phase 1.
    monitor_joint:
        Metric to watch during Phase 2.
    patience_joint:
        Number of validation epochs with no improvement before stopping Phase 2.
    mode:
        ``'min'`` or ``'max'``.
    min_delta:
        Minimum change to qualify as an improvement.
    """

    def __init__(
        self,
        monitor_forward: str = "val/forward_l2",
        patience_forward: int = 10,
        monitor_joint: str = "val/reconstruction_loss",
        patience_joint: int = 15,
        mode: str = "min",
        min_delta: float = 0.0,
    ) -> None:
        super().__init__()
        self.monitor_forward = monitor_forward
        self.patience_forward = patience_forward
        self.monitor_joint = monitor_joint
        self.patience_joint = patience_joint
        self.mode = mode
        self.min_delta = min_delta

        # Internal state
        self._best_forward: float | None = None
        self._best_joint: float | None = None
        self._wait_forward: int = 0
        self._wait_joint: int = 0
        self.stopped: bool = False

    # ------------------------------------------------------------------

    def _is_improvement(self, current: float, best: float | None) -> bool:
        if best is None:
            return True
        if self.mode == "min":
            return current < best - self.min_delta
        return current > best + self.min_delta

    def reset_phase_state(self, phase: str) -> None:
        """Reset early-stopping counters when transitioning between phases."""
        if phase == "forward":
            self._best_forward = None
            self._wait_forward = 0
        else:
            self._best_joint = None
            self._wait_joint = 0
        self.stopped = False
        logger.info("Early stopping state reset for phase '%s'.", phase)

    # ------------------------------------------------------------------

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Check the monitored metric and signal the trainer to stop if
        patience is exhausted."""
        phase: str = getattr(pl_module, "phase", "forward")
        metrics = trainer.callback_metrics

        if phase == "forward":
            monitor_key = self.monitor_forward
            patience = self.patience_forward
        else:
            monitor_key = self.monitor_joint
            patience = self.patience_joint

        if monitor_key not in metrics:
            return

        current = float(metrics[monitor_key].item())

        if phase == "forward":
            if self._is_improvement(current, self._best_forward):
                self._best_forward = current
                self._wait_forward = 0
            else:
                self._wait_forward += 1
                if self._wait_forward >= patience:
                    logger.info(
                        "Early stopping triggered for phase '%s' after %d "
                        "epochs without improvement on '%s'.",
                        phase,
                        patience,
                        monitor_key,
                    )
                    self.stopped = True
                    trainer.should_stop = True
        else:
            if self._is_improvement(current, self._best_joint):
                self._best_joint = current
                self._wait_joint = 0
            else:
                self._wait_joint += 1
                if self._wait_joint >= patience:
                    logger.info(
                        "Early stopping triggered for phase '%s' after %d "
                        "epochs without improvement on '%s'.",
                        phase,
                        patience,
                        monitor_key,
                    )
                    self.stopped = True
                    trainer.should_stop = True
