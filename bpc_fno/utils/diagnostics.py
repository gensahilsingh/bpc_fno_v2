"""Stage-boundary tensor diagnostics logging."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Attempt to import wandb; gracefully degrade if unavailable.
try:
    import wandb  # type: ignore[import-untyped]

    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None  # type: ignore[assignment]
    _WANDB_AVAILABLE = False


@dataclass(frozen=True)
class TensorStats:
    """Immutable snapshot of scalar tensor statistics."""

    shape: tuple[int, ...]
    mean: float
    std: float
    min: float
    max: float
    has_nan: bool
    has_inf: bool

    def as_dict(self, prefix: str = "") -> dict[str, Any]:
        """Return statistics as a flat dictionary suitable for logging."""
        p = f"{prefix}/" if prefix else ""
        return {
            f"{p}shape": list(self.shape),
            f"{p}mean": self.mean,
            f"{p}std": self.std,
            f"{p}min": self.min,
            f"{p}max": self.max,
            f"{p}has_nan": self.has_nan,
            f"{p}has_inf": self.has_inf,
        }


def _compute_stats(tensor: torch.Tensor) -> TensorStats:
    """Compute scalar statistics for *tensor* (detached, on CPU)."""
    t = tensor.detach().float()
    return TensorStats(
        shape=tuple(tensor.shape),
        mean=t.mean().item(),
        std=t.std().item() if t.numel() > 1 else 0.0,
        min=t.min().item(),
        max=t.max().item(),
        has_nan=bool(torch.isnan(t).any().item()),
        has_inf=bool(torch.isinf(t).any().item()),
    )


class DiagnosticsLogger:
    """Log tensor statistics at stage boundaries.

    Parameters
    ----------
    use_wandb:
        If ``True`` **and** W&B is available and a run is active, tensor
        statistics are also logged to Weights & Biases.
    log_level:
        Python logging level used for the text log messages.
    """

    def __init__(
        self,
        use_wandb: bool = False,
        log_level: int = logging.DEBUG,
    ) -> None:
        self.use_wandb = use_wandb and _WANDB_AVAILABLE
        self.log_level = log_level

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_tensor(
        self,
        name: str,
        tensor: torch.Tensor,
        step: int,
    ) -> TensorStats:
        """Log statistics for a single tensor.

        Parameters
        ----------
        name:
            Human-readable name of the tensor (e.g. ``"J_i_pred"``).
        tensor:
            The tensor to inspect.
        step:
            Global training / evaluation step counter.

        Returns
        -------
        TensorStats
            The computed statistics.
        """
        stats = _compute_stats(tensor)
        logger.log(
            self.log_level,
            "[step=%d] %s | shape=%s mean=%.6g std=%.6g min=%.6g max=%.6g nan=%s inf=%s",
            step,
            name,
            list(stats.shape),
            stats.mean,
            stats.std,
            stats.min,
            stats.max,
            stats.has_nan,
            stats.has_inf,
        )

        if self.use_wandb and wandb is not None and wandb.run is not None:
            wandb.log(stats.as_dict(prefix=name), step=step)

        return stats

    def log_stage_boundary(
        self,
        stage_name: str,
        tensors: dict[str, torch.Tensor],
        step: int,
    ) -> dict[str, TensorStats]:
        """Log statistics for every tensor at a stage boundary.

        Parameters
        ----------
        stage_name:
            Descriptive label for the boundary (e.g. ``"after_forward_pass"``).
        tensors:
            Mapping from tensor name to tensor.
        step:
            Global step counter.

        Returns
        -------
        dict[str, TensorStats]
            Mapping from tensor name to computed statistics.
        """
        logger.log(
            self.log_level,
            "===== Stage boundary: %s (step=%d) =====",
            stage_name,
            step,
        )

        all_stats: dict[str, TensorStats] = {}
        wandb_payload: dict[str, Any] = {}

        for tensor_name, tensor in tensors.items():
            full_name = f"{stage_name}/{tensor_name}"
            stats = _compute_stats(tensor)
            all_stats[tensor_name] = stats

            logger.log(
                self.log_level,
                "  %s | shape=%s mean=%.6g std=%.6g min=%.6g max=%.6g nan=%s inf=%s",
                full_name,
                list(stats.shape),
                stats.mean,
                stats.std,
                stats.min,
                stats.max,
                stats.has_nan,
                stats.has_inf,
            )

            if self.use_wandb:
                wandb_payload.update(stats.as_dict(prefix=full_name))

        if (
            self.use_wandb
            and wandb is not None
            and wandb.run is not None
            and wandb_payload
        ):
            wandb.log(wandb_payload, step=step)

        return all_stats
