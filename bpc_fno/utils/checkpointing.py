"""Save, load, and validate training checkpoints with stage metadata."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    phase: str,
    metrics: dict[str, Any],
    path: str | Path,
    *,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    extra_state: dict[str, Any] | None = None,
) -> None:
    """Save a training checkpoint to disk.

    Parameters
    ----------
    model:
        The model whose ``state_dict`` will be saved.
    optimizer:
        Optional optimizer whose ``state_dict`` will be saved.
    epoch:
        Current epoch number.
    phase:
        Training phase identifier (e.g. ``'forward'`` or ``'joint'``).
    metrics:
        Dictionary of scalar metrics to record alongside the checkpoint.
    path:
        Destination file path.
    """
    if phase not in ("forward", "joint"):
        logger.warning(
            "Unexpected phase '%s'; expected 'forward' or 'joint'.", phase
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "phase": phase,
        "metrics": metrics,
        "parameter_names": sorted(model.state_dict().keys()),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if extra_state is not None:
        payload["extra_state"] = extra_state

    tmp_path = path.with_name(f"{path.name}.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)
    logger.info(
        "Checkpoint saved: epoch=%d phase=%s path=%s", epoch, phase, path
    )


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    path: str | Path,
    *,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint and restore model / optimizer state.

    Parameters
    ----------
    model:
        Target model to load weights into.
    optimizer:
        Target optimizer to restore state into.  Pass ``None`` to skip
        optimizer restoration (useful for fine-tuning).
    path:
        Path to the checkpoint file.
    strict:
        If ``True`` (default), validate that the checkpoint's parameter
        names match the model's parameter names exactly before loading.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys ``'epoch'``, ``'phase'``, and ``'metrics'``.

    Raises
    ------
    RuntimeError
        If *strict* is ``True`` and the checkpoint's parameter names do
        not match the model's current parameter names.
    """
    path = Path(path)
    checkpoint: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)

    # --- Architecture validation ---
    if strict:
        saved_names = set(checkpoint.get("parameter_names", []))
        model_names = set(model.state_dict().keys())

        missing = model_names - saved_names
        unexpected = saved_names - model_names

        if missing or unexpected:
            parts: list[str] = ["Checkpoint architecture mismatch."]
            if missing:
                parts.append(f"  Missing in checkpoint: {sorted(missing)}")
            if unexpected:
                parts.append(f"  Unexpected in checkpoint: {sorted(unexpected)}")
            raise RuntimeError("\n".join(parts))

    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    elif optimizer is not None:
        logger.warning("Checkpoint has no optimizer state: %s", path)

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    elif scheduler is not None:
        logger.warning("Checkpoint has no scheduler state: %s", path)

    meta = {
        "epoch": checkpoint.get("epoch", 0),
        "phase": checkpoint.get("phase", "unknown"),
        "metrics": checkpoint.get("metrics", {}),
        "extra_state": checkpoint.get("extra_state", {}),
    }
    logger.info(
        "Checkpoint loaded: epoch=%d phase=%s path=%s",
        meta["epoch"],
        meta["phase"],
        path,
    )
    return meta


def validate_checkpoint(path: str | Path) -> dict[str, Any]:
    """Read checkpoint metadata without loading weights into a model.

    Parameters
    ----------
    path:
        Path to the checkpoint file.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys ``'epoch'``, ``'phase'``, ``'metrics'``,
        ``'parameter_names'``, and ``'num_parameters'``.
    """
    path = Path(path)
    checkpoint: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)

    param_names: list[str] = checkpoint.get("parameter_names", [])
    model_sd: dict[str, torch.Tensor] = checkpoint.get("model_state_dict", {})

    total_params = sum(v.numel() for v in model_sd.values())

    meta = {
        "epoch": checkpoint.get("epoch", 0),
        "phase": checkpoint.get("phase", "unknown"),
        "metrics": checkpoint.get("metrics", {}),
        "parameter_names": param_names,
        "num_parameters": total_params,
        "has_optimizer_state": "optimizer_state_dict" in checkpoint,
        "has_scheduler_state": "scheduler_state_dict" in checkpoint,
        "extra_state": checkpoint.get("extra_state", {}),
    }
    logger.info(
        "Checkpoint validated: epoch=%d phase=%s params=%d path=%s",
        meta["epoch"],
        meta["phase"],
        total_params,
        path,
    )
    return meta
