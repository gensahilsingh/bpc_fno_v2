"""Phase 1 training: train the forward operator (Stage 2) only.

Trains the shared FNO backbone and the forward-PINO head to predict
magnetic field measurements B from current density J_i and geometry.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from bpc_fno.data.data_module import BPCFNODataModule
from bpc_fno.models.bpc_fno_a import BPC_FNO_A
from bpc_fno.utils.checkpointing import save_checkpoint
from bpc_fno.utils.normalization import Normalizer

_LOG_DIR = Path("logs")
_LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_LOG_DIR / "train_phase1.log"),
    ],
)
logger = logging.getLogger(__name__)


def _build_optimizer(
    model: BPC_FNO_A, config: DictConfig
) -> torch.optim.Optimizer:
    """Build an AdamW optimizer for Phase 1 (forward-only) training.

    Only the shared FNO backbone and forward head parameters are trained.
    """
    param_groups = model.get_parameter_groups()
    lr: float = float(config.training.lr_init)
    return torch.optim.AdamW(
        [
            {"params": param_groups["fno_shared"], "lr": lr},
            {"params": param_groups["forward_head"], "lr": lr},
        ],
        weight_decay=1e-4,
    )


def _build_scheduler(
    optimizer: torch.optim.Optimizer, config: DictConfig, steps_per_epoch: int
) -> torch.optim.lr_scheduler._LRScheduler:
    """Build a cosine-annealing LR scheduler with optional warmup."""
    total_epochs: int = int(config.training.phase1_epochs)
    warmup_steps: int = int(config.training.get("lr_warmup_steps", 0))
    lr_final: float = float(config.training.lr_final)

    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(step / max(warmup_steps, 1), 1e-6)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine_decay = 0.5 * (1.0 + __import__("math").cos(
            __import__("math").pi * progress
        ))
        lr_init = float(config.training.lr_init)
        return max((lr_final / lr_init) + (1.0 - lr_final / lr_init) * cosine_decay, lr_final / lr_init)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _create_normalizer_proxy(normalizer: Normalizer) -> object:
    """Create a normalizer proxy that implements the protocol expected by
    SyntheticMIGDataset (normalize/denormalize with string key)."""

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
            return tensor  # sensor_pos and others: passthrough

        def denormalize(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
            if key == "J_i":
                return self._norm.denormalize_J_i(tensor)
            if key in ("B_mig", "B_mig_clean"):
                return self._norm.denormalize_B(tensor)
            return tensor

    return _NormalizerProxy(normalizer)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1: train forward operator (Stage 2)"
    )
    parser.add_argument(
        "--config", default="configs/arch_a.yaml",
        help="Path to architecture config file",
    )
    parser.add_argument(
        "--checkpoint-dir", default="checkpoints",
        help="Directory to save training checkpoints",
    )
    parser.add_argument(
        "--normalization", default="data/processed/normalization.json",
        help="Path to normalization statistics JSON",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device to train on (default: auto-detect)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    norm_path = Path(args.normalization)
    if not norm_path.exists():
        logger.error(
            "Normalization file not found: %s. "
            "Run scripts/compute_normalization.py first.",
            norm_path,
        )
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

        # Load normalizer.
        normalizer = Normalizer()
        normalizer.load(norm_path)
        norm_proxy = _create_normalizer_proxy(normalizer)

        # Create data module.
        data_module = BPCFNODataModule(config, normalizer=norm_proxy)
        data_module.setup(stage="fit")
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        steps_per_epoch = len(train_loader)
        logger.info(
            "Data: %d train batches, %d val batches per epoch",
            steps_per_epoch,
            len(val_loader),
        )

        # Create model.
        model = BPC_FNO_A(config).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Model created: %d total parameters", total_params)

        # Create optimizer and scheduler.
        optimizer = _build_optimizer(model, config)
        scheduler = _build_scheduler(optimizer, config, steps_per_epoch)

        # Training loop.
        n_epochs: int = int(config.training.phase1_epochs)
        grad_clip: float = float(config.training.get("grad_clip_norm", 1.0))
        best_val_loss = float("inf")

        logger.info("Starting Phase 1 training for %d epochs...", n_epochs)
        t0 = time.monotonic()

        for epoch in range(1, n_epochs + 1):
            # --- Train ---
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                J_i = batch["J_i"].to(device)
                geometry = batch["geometry"].to(device)
                B_target = batch["B_mig_clean"].to(device)

                B_pred = model.forward_only(J_i, geometry)
                loss = F.mse_loss(B_pred, B_target)

                optimizer.zero_grad()
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)

            # --- Validate ---
            model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for batch in val_loader:
                    J_i = batch["J_i"].to(device)
                    geometry = batch["geometry"].to(device)
                    B_target = batch["B_mig_clean"].to(device)

                    B_pred = model.forward_only(J_i, geometry)
                    loss = F.mse_loss(B_pred, B_target)
                    val_loss += loss.item()
                    n_val += 1

            avg_val_loss = val_loss / max(n_val, 1)
            current_lr = optimizer.param_groups[0]["lr"]

            logger.info(
                "Epoch %d/%d  train_loss=%.6f  val_loss=%.6f  lr=%.2e",
                epoch, n_epochs, avg_train_loss, avg_val_loss, current_lr,
            )

            # Save best checkpoint.
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                ckpt_path = Path(args.checkpoint_dir) / "phase1_best.pt"
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    phase="forward",
                    metrics={
                        "train_loss": avg_train_loss,
                        "val_loss": avg_val_loss,
                    },
                    path=ckpt_path,
                )

        # Save final checkpoint.
        final_path = Path(args.checkpoint_dir) / "phase1_final.pt"
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=n_epochs,
            phase="forward",
            metrics={
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "best_val_loss": best_val_loss,
            },
            path=final_path,
        )

        elapsed = time.monotonic() - t0
        logger.info(
            "Phase 1 training complete in %.1f s. "
            "Best val_loss=%.6f. Checkpoints saved to %s",
            elapsed, best_val_loss, args.checkpoint_dir,
        )

    except Exception as exc:
        logger.error("Phase 1 training failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
