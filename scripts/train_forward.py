"""Phase 1 training: train the forward operator (Stage 2) only.

Trains the shared FNO backbone and the forward-PINO head to predict
magnetic field measurements B from current density J_i and geometry.

Plain PyTorch training loop (no Lightning).
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from bpc_fno.data.data_module import BPCFNODataModule
from bpc_fno.models.bpc_fno_a import BPC_FNO_A
from bpc_fno.utils.checkpointing import load_checkpoint, save_checkpoint
from bpc_fno.utils.data_paths import resolve_required_data_dir, validate_sample_data_dir
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
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build a cosine-annealing LR scheduler with linear warmup."""
    total_epochs: int = int(config.training.phase1_epochs)
    warmup_steps: int = int(config.training.get("lr_warmup_steps", 0))
    lr_init: float = float(config.training.lr_init)
    lr_final: float = float(config.training.lr_final)

    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        # Linear warmup
        if step < warmup_steps:
            return max(step / max(warmup_steps, 1), 1e-6)
        # Cosine decay from lr_init to lr_final
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_ratio = lr_final / lr_init
        return max(min_ratio + (1.0 - min_ratio) * cosine_decay, min_ratio)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _resolve_resume_path(
    resume_arg: str | None, checkpoint_dir: Path, default_name: str
) -> Path | None:
    """Resolve ``--resume`` to an on-disk checkpoint path."""
    if resume_arg is None:
        return None
    if resume_arg == "auto":
        return checkpoint_dir / default_name
    return Path(resume_arg)


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
        "--data-dir", default=None,
        help="Synthetic data directory (required unless config.data.data_dir is set)",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device to train on (default: auto-detect CUDA)",
    )
    parser.add_argument(
        "--resume", nargs="?", const="auto", default=None,
        help=(
            "Resume from a checkpoint path. Pass without a value to use "
            "<checkpoint-dir>/phase1_last.pt."
        ),
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
        checkpoint_dir = Path(args.checkpoint_dir)
        data_dir = resolve_required_data_dir(config, args.data_dir)
        validate_sample_data_dir(data_dir)
        resume_path = _resolve_resume_path(
            args.resume, checkpoint_dir, "phase1_last.pt"
        )

        # ---- Device auto-detection ----
        if args.device is not None:
            device = torch.device(args.device)
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        logger.info("Using device: %s", device)

        # ---- Load normalizer ----
        normalizer = Normalizer()
        normalizer.load(norm_path)

        # ---- Create data module (num_workers=0 for Windows) ----
        # Dataset accesses normalizer.stats directly (no proxy needed)
        data_module = BPCFNODataModule(config, normalizer=normalizer)
        data_module.setup(stage="fit")
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        steps_per_epoch = len(train_loader)
        logger.info(
            "Data: %d train batches, %d val batches per epoch",
            steps_per_epoch,
            len(val_loader),
        )
        logger.info("Training data directory: %s", data_dir)

        # ---- Create model ----
        model = BPC_FNO_A(config).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Model created: %d total parameters", total_params)

        # ---- Optimizer + cosine LR schedule with warmup ----
        optimizer = _build_optimizer(model, config)
        scheduler = _build_scheduler(optimizer, config, steps_per_epoch)
        start_epoch = 1

        # ---- Training loop ----
        n_epochs: int = int(config.training.phase1_epochs)
        grad_clip: float = float(config.training.get("grad_clip_norm", 1.0))
        best_val_loss = float("inf")
        run_meta = {
            "n_output_timesteps": int(config.model.get("n_output_timesteps", 1)),
            "batch_size": int(config.training.get("batch_size", 16)),
        }

        if resume_path is not None:
            if not resume_path.exists():
                logger.error("Resume checkpoint not found: %s", resume_path)
                sys.exit(1)
            ckpt_meta = load_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                path=resume_path,
                strict=True,
            )
            if ckpt_meta["phase"] != "forward":
                logger.error(
                    "Resume checkpoint phase mismatch: expected 'forward', got '%s'",
                    ckpt_meta["phase"],
                )
                sys.exit(1)
            start_epoch = int(ckpt_meta["epoch"]) + 1
            best_val_loss = float(
                ckpt_meta["extra_state"].get(
                    "best_val_loss",
                    ckpt_meta["metrics"].get("best_val_loss", float("inf")),
                )
            )
            logger.info(
                "Resuming Phase 1 from %s at epoch %d (best_val_loss=%.6f)",
                resume_path,
                start_epoch,
                best_val_loss,
            )
        else:
            existing_phase_ckpts = sorted(checkpoint_dir.glob("phase1_*.pt"))
            if existing_phase_ckpts:
                logger.warning(
                    "Starting a fresh Phase 1 run in %s, which already contains "
                    "%d phase1 checkpoint(s). Use --resume or a different "
                    "--checkpoint-dir to avoid overwriting prior results.",
                    checkpoint_dir,
                    len(existing_phase_ckpts),
                )

        logger.info("Starting Phase 1 training for %d epochs...", n_epochs)
        t0 = time.monotonic()

        if start_epoch > n_epochs:
            logger.info(
                "Checkpoint epoch exceeds configured phase1_epochs "
                "(start_epoch=%d, phase1_epochs=%d); nothing to do.",
                start_epoch,
                n_epochs,
            )
            return

        for epoch in range(start_epoch, n_epochs + 1):
            # --- Train ---
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                J_i = batch["J_i"].to(device)
                geometry = batch["geometry"].to(device)
                B_target = batch["B_true"].to(device)

                B_pred = model.forward_only(J_i, geometry)
                loss = F.mse_loss(B_pred, B_target)
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite Phase 1 loss at epoch {epoch}, "
                        f"train batch {n_batches + 1}: {float(loss.detach().cpu())}"
                    )

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
                    B_target = batch["B_true"].to(device)

                    B_pred = model.forward_only(J_i, geometry)
                    loss = F.mse_loss(B_pred, B_target)
                    if not torch.isfinite(loss):
                        raise RuntimeError(
                            f"Non-finite Phase 1 validation loss at epoch {epoch}, "
                            f"val batch {n_val + 1}: {float(loss.detach().cpu())}"
                        )
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
                ckpt_path = checkpoint_dir / "phase1_best.pt"
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    phase="forward",
                    metrics={
                        "train_loss": avg_train_loss,
                        "val_loss": avg_val_loss,
                        "best_val_loss": best_val_loss,
                        **run_meta,
                    },
                    extra_state={"best_val_loss": best_val_loss},
                    path=ckpt_path,
                )

            last_path = checkpoint_dir / "phase1_last.pt"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                phase="forward",
                metrics={
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "best_val_loss": best_val_loss,
                    **run_meta,
                },
                extra_state={"best_val_loss": best_val_loss},
                path=last_path,
            )

        # Save final checkpoint.
        final_path = checkpoint_dir / "phase1_final.pt"
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=n_epochs,
            phase="forward",
            metrics={
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "best_val_loss": best_val_loss,
                **run_meta,
            },
            extra_state={"best_val_loss": best_val_loss},
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
