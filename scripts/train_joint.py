"""Phase 2 training: train all stages jointly (forward + inverse + decoder).

Loads a Phase 1 checkpoint and fine-tunes all model components end-to-end,
including the forward consistency loss and physics-informed losses.

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
from bpc_fno.physics.consistency_loss import ForwardConsistencyLoss
from bpc_fno.physics.monodomain_loss import MonodomainPDELoss
from bpc_fno.utils.checkpointing import load_checkpoint, save_checkpoint
from bpc_fno.utils.normalization import Normalizer

_LOG_DIR = Path("logs")
_LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_LOG_DIR / "train_phase2.log"),
    ],
)
logger = logging.getLogger(__name__)


def _build_optimizer(
    model: BPC_FNO_A, config: DictConfig
) -> torch.optim.Optimizer:
    """Build AdamW optimizer with differential learning rates per group.

    0.1x for shared FNO + forward head (pre-trained in Phase 1),
    1x for inverse head + decoder (new components).
    """
    param_groups = model.get_parameter_groups()
    lr: float = float(config.training.lr_init)
    return torch.optim.AdamW(
        [
            {"params": param_groups["fno_shared"], "lr": lr * 0.1},
            {"params": param_groups["forward_head"], "lr": lr * 0.1},
            {"params": param_groups["inverse_head"], "lr": lr},
            {"params": param_groups["decoder"], "lr": lr},
        ],
        weight_decay=1e-4,
    )


def _build_scheduler(
    optimizer: torch.optim.Optimizer, config: DictConfig, steps_per_epoch: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build a cosine-annealing LR scheduler with linear warmup for Phase 2."""
    total_epochs: int = int(config.training.phase2_epochs)
    warmup_steps: int = int(config.training.get("lr_warmup_steps", 0))
    lr_init: float = float(config.training.lr_init)
    lr_final: float = float(config.training.lr_final)

    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(step / max(warmup_steps, 1), 1e-6)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_ratio = lr_final / lr_init
        return max(min_ratio + (1.0 - min_ratio) * cosine_decay, min_ratio)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _get_lambda_physics(epoch: int, config: DictConfig) -> float:
    """Compute physics loss weight with exponential warm-up."""
    lam_init: float = float(config.training.lambda_physics_init)
    lam_final: float = float(config.training.lambda_physics_final)
    doubling: float = float(config.training.lambda_physics_doubling_epochs)

    lam = lam_init * (2.0 ** (epoch / doubling))
    return min(lam, lam_final)


def _get_lambda_kl(epoch: int, config: DictConfig) -> float:
    """KL divergence weight (constant after initial value)."""
    return float(config.training.lambda_kl_init)


def _get_lambda_consistency(epoch: int, config: DictConfig) -> float:
    """Consistency loss weight (activated after a start epoch)."""
    start_epoch: int = int(config.training.lambda_consistency_start_epoch)
    if epoch < start_epoch:
        return 0.0
    return float(config.training.lambda_consistency)


def _kl_divergence(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence D_KL(q(z|x) || N(0, I))."""
    return -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2: train all stages jointly"
    )
    parser.add_argument(
        "--config", default="configs/arch_a.yaml",
        help="Path to architecture config file",
    )
    parser.add_argument(
        "--phase1-checkpoint", default="checkpoints/phase1_best.pt",
        help="Path to Phase 1 checkpoint to load",
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
        help="Device to train on (default: auto-detect CUDA)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    phase1_path = Path(args.phase1_checkpoint)
    if not phase1_path.exists():
        logger.error(
            "Phase 1 checkpoint not found: %s. "
            "Run scripts/train_forward.py first.",
            phase1_path,
        )
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

        # ---- Create model and load Phase 1 weights ----
        model = BPC_FNO_A(config).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Model created: %d total parameters", total_params)

        # Train ALL model parameters (differential LR applied in optimizer).
        optimizer = _build_optimizer(model, config)
        scheduler = _build_scheduler(optimizer, config, steps_per_epoch)

        logger.info("Loading Phase 1 checkpoint from %s", phase1_path)
        ckpt_meta = load_checkpoint(
            model=model, optimizer=None, path=phase1_path, strict=True
        )
        logger.info(
            "Phase 1 checkpoint loaded (epoch=%d, phase=%s)",
            ckpt_meta["epoch"],
            ckpt_meta["phase"],
        )

        # ---- Loss modules ----
        consistency_loss_fn = ForwardConsistencyLoss().to(device)
        physics_cfg = OmegaConf.create({
            "voxel_size_cm": config.model.get("grid_size", 32) * 0.5 / config.model.get("grid_size", 32),
            "n_collocation_points": config.training.get("n_collocation_points", 1024),
        })
        physics_loss_fn = MonodomainPDELoss(physics_cfg).to(device)

        # ---- Training loop ----
        n_epochs: int = int(config.training.phase2_epochs)
        grad_clip: float = float(config.training.get("grad_clip_norm", 1.0))
        best_val_loss = float("inf")

        logger.info("Starting Phase 2 training for %d epochs...", n_epochs)
        t0 = time.monotonic()

        for epoch in range(1, n_epochs + 1):
            lam_physics = _get_lambda_physics(epoch, config)
            lam_kl = _get_lambda_kl(epoch, config)
            lam_consistency = _get_lambda_consistency(epoch, config)

            # --- Train ---
            model.train()
            epoch_losses: dict[str, float] = {
                "total": 0.0,
                "recon": 0.0,
                "forward": 0.0,
                "kl": 0.0,
                "consistency": 0.0,
                "physics": 0.0,
            }
            n_batches = 0

            for batch in train_loader:
                J_i = batch["J_i"].to(device)
                geometry = batch["geometry"].to(device)
                B_obs = batch["B_noisy"].to(device)
                B_clean = batch["B_true"].to(device)

                outputs = model(batch={
                    "J_i": J_i, "geometry": geometry, "B_mig": B_obs,
                })

                # Reconstruction loss: predicted J_i_hat vs ground truth J_i.
                loss_recon = F.mse_loss(outputs["J_i_hat"], J_i)

                # Forward loss: predicted B vs clean B.
                loss_forward = F.mse_loss(outputs["B_pred"], B_clean)

                # KL divergence.
                loss_kl = _kl_divergence(outputs["mu"], outputs["log_var"])

                # Forward consistency: re-predict B from reconstructed J_i_hat.
                loss_consistency = torch.tensor(0.0, device=device)
                if lam_consistency > 0:
                    B_re_pred = model.forward_only(outputs["J_i_hat"], geometry)
                    loss_consistency = F.mse_loss(B_re_pred, B_obs)

                # Physics residual loss.
                loss_physics = torch.tensor(0.0, device=device)
                if lam_physics > 0:
                    try:
                        loss_physics = physics_loss_fn(
                            outputs["J_i_hat"], geometry
                        )
                    except Exception:
                        pass  # Gracefully skip if physics loss fails.

                # Total loss.
                total_loss = (
                    loss_recon
                    + loss_forward
                    + lam_kl * loss_kl
                    + lam_consistency * loss_consistency
                    + lam_physics * loss_physics
                )

                optimizer.zero_grad()
                total_loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()

                epoch_losses["total"] += total_loss.item()
                epoch_losses["recon"] += loss_recon.item()
                epoch_losses["forward"] += loss_forward.item()
                epoch_losses["kl"] += loss_kl.item()
                epoch_losses["consistency"] += loss_consistency.item()
                epoch_losses["physics"] += loss_physics.item()
                n_batches += 1

            # Average losses.
            for k in epoch_losses:
                epoch_losses[k] /= max(n_batches, 1)

            # --- Validate ---
            model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for batch in val_loader:
                    J_i = batch["J_i"].to(device)
                    geometry = batch["geometry"].to(device)
                    B_obs = batch["B_noisy"].to(device)
                    B_clean = batch["B_true"].to(device)

                    outputs = model(batch={
                        "J_i": J_i, "geometry": geometry, "B_mig": B_obs,
                    })

                    loss = (
                        F.mse_loss(outputs["J_i_hat"], J_i)
                        + F.mse_loss(outputs["B_pred"], B_clean)
                    )
                    val_loss += loss.item()
                    n_val += 1

            avg_val_loss = val_loss / max(n_val, 1)
            current_lr = optimizer.param_groups[0]["lr"]

            # Log ALL loss terms every epoch.
            logger.info(
                "Epoch %d/%d  total=%.6f  recon=%.6f  fwd=%.6f  "
                "kl=%.6f  cons=%.6f  phys=%.6f  val=%.6f  "
                "lam_p=%.4f  lam_c=%.4f  lr=%.2e",
                epoch, n_epochs,
                epoch_losses["total"],
                epoch_losses["recon"],
                epoch_losses["forward"],
                epoch_losses["kl"],
                epoch_losses["consistency"],
                epoch_losses["physics"],
                avg_val_loss,
                lam_physics,
                lam_consistency,
                current_lr,
            )

            # Save best checkpoint by val_loss.
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                ckpt_path = Path(args.checkpoint_dir) / "phase2_best.pt"
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    phase="joint",
                    metrics={
                        **epoch_losses,
                        "val_loss": avg_val_loss,
                    },
                    path=ckpt_path,
                )

        # Save final checkpoint.
        final_path = Path(args.checkpoint_dir) / "phase2_final.pt"
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=n_epochs,
            phase="joint",
            metrics={
                **epoch_losses,
                "val_loss": avg_val_loss,
                "best_val_loss": best_val_loss,
            },
            path=final_path,
        )

        elapsed = time.monotonic() - t0
        logger.info(
            "Phase 2 training complete in %.1f s. "
            "Best val_loss=%.6f. Checkpoints saved to %s",
            elapsed, best_val_loss, args.checkpoint_dir,
        )

    except Exception as exc:
        logger.error("Phase 2 training failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
