"""Simple Conv3D baseline to verify synthetic data is learnable.

Trains a small 3D CNN to predict flattened B_mig from J_i for 30 epochs
on the first 200 samples, validates on next 40. If val_loss < 0.5, the
data is considered learnable (the signal-to-noise ratio is sufficient
for a simple model to pick up the mapping).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_LOG_DIR = Path("logs")
_LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_LOG_DIR / "sanity_check_conv.log"),
    ],
)
logger = logging.getLogger(__name__)


# ---- Model ----

class ConvBaseline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(3, 32, 3, padding=1), nn.GELU(),
            nn.Conv3d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv3d(64, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool3d(4), nn.Flatten(),
            nn.Linear(64 * 64, 48),
        )

    def forward(self, J: torch.Tensor) -> torch.Tensor:
        return self.net(J)


# ---- Dataset ----

class SimpleH5Dataset(Dataset):
    """Minimal dataset: loads J_i and B_mig from HDF5 files.

    Takes a single mid-activity timestep per sample. Normalizes both
    inputs and targets to zero-mean unit-variance using training stats.
    """

    def __init__(
        self,
        files: list[Path],
        j_mean: float = 0.0,
        j_std: float = 1.0,
        b_mean: float = 0.0,
        b_std: float = 1.0,
    ) -> None:
        self.files = files
        self.j_mean = j_mean
        self.j_std = j_std
        self.b_mean = b_mean
        self.b_std = b_std

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        fpath = self.files[idx]
        with h5py.File(fpath, "r") as f:
            T = f["J_i"].shape[0]
            # Pick midpoint of active window
            t_lo = max(int(T * 0.1), 0)
            t_hi = max(int(T * 0.7), t_lo + 1)
            t_hi = min(t_hi, T)
            t_idx = (t_lo + t_hi) // 2

            # J_i: (N, N, N, 3) -> (3, N, N, N)
            j_i = np.asarray(f["J_i"][t_idx], dtype=np.float32)
            j_i = torch.from_numpy(j_i).permute(3, 0, 1, 2)

            # B_mig: (Ns, 3) -> (Ns*3,)
            b_mig = np.asarray(f["B_mig"][t_idx], dtype=np.float32)
            b_mig = torch.from_numpy(b_mig.reshape(-1))

        # Normalize
        j_i = (j_i - self.j_mean) / self.j_std
        b_mig = (b_mig - self.b_mean) / self.b_std

        return j_i, b_mig


def _compute_stats(files: list[Path]) -> tuple[float, float, float, float]:
    """Compute simple global mean/std for J_i and B_mig over given files."""
    j_vals: list[float] = []
    b_vals: list[float] = []

    n_files = min(len(files), 50)  # subsample for speed
    indices = np.linspace(0, len(files) - 1, n_files, dtype=int)

    for i in indices:
        fpath = files[i]
        try:
            with h5py.File(fpath, "r") as f:
                T = f["J_i"].shape[0]
                t_idx = max(int(T * 0.1), 0)
                t_idx = (t_idx + min(int(T * 0.7), T)) // 2
                j_data = np.asarray(f["J_i"][t_idx], dtype=np.float64)
                b_data = np.asarray(f["B_mig"][t_idx], dtype=np.float64)
                j_vals.append(j_data.mean())
                j_vals.append(j_data.std())
                b_vals.append(b_data.mean())
                b_vals.append(b_data.std())
        except Exception:
            continue

    if not j_vals:
        return 0.0, 1.0, 0.0, 1.0

    # Simple approach: use median of stds as global std
    j_means = j_vals[0::2]
    j_stds = j_vals[1::2]
    b_means = b_vals[0::2]
    b_stds = b_vals[1::2]

    j_mean = float(np.mean(j_means))
    j_std = float(np.median(j_stds)) if j_stds else 1.0
    b_mean = float(np.mean(b_means))
    b_std = float(np.median(b_stds)) if b_stds else 1.0

    # Guard against zero std
    j_std = max(j_std, 1e-30)
    b_std = max(b_std, 1e-30)

    return j_mean, j_std, b_mean, b_std


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Conv3D baseline sanity check for data learnability"
    )
    parser.add_argument(
        "--data-dir", default="data/synthetic",
        help="Directory containing sample_*.h5 files",
    )
    parser.add_argument(
        "--n-train", type=int, default=200,
        help="Number of training samples (default: 200)",
    )
    parser.add_argument(
        "--n-val", type=int, default=40,
        help="Number of validation samples (default: 40)",
    )
    parser.add_argument(
        "--epochs", type=int, default=30,
        help="Number of training epochs (default: 30)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size (default: 8)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Val loss threshold for convergence (default: 0.5)",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device to train on (default: auto-detect CUDA)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error("Data directory not found: %s", data_dir)
        sys.exit(1)

    all_h5 = sorted(data_dir.glob("sample_*.h5"))
    n_available = len(all_h5)
    if n_available < args.n_train + args.n_val:
        logger.error(
            "Not enough samples: found %d, need %d (train) + %d (val)",
            n_available, args.n_train, args.n_val,
        )
        sys.exit(1)

    train_files = all_h5[: args.n_train]
    val_files = all_h5[args.n_train : args.n_train + args.n_val]
    logger.info("Train: %d samples, Val: %d samples", len(train_files), len(val_files))

    try:
        # ---- Device ----
        if args.device is not None:
            device = torch.device(args.device)
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        logger.info("Using device: %s", device)

        # ---- Compute normalization stats from training set ----
        logger.info("Computing normalization statistics...")
        j_mean, j_std, b_mean, b_std = _compute_stats(train_files)
        logger.info("  J_i: mean=%.4e  std=%.4e", j_mean, j_std)
        logger.info("  B_mig: mean=%.4e  std=%.4e", b_mean, b_std)

        # ---- Datasets ----
        train_ds = SimpleH5Dataset(train_files, j_mean, j_std, b_mean, b_std)
        val_ds = SimpleH5Dataset(val_files, j_mean, j_std, b_mean, b_std)

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=0, pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=0, pin_memory=True,
        )

        # ---- Model ----
        model = ConvBaseline().to(device)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info("ConvBaseline: %d parameters", n_params)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        # ---- Training loop ----
        logger.info("Training for %d epochs...", args.epochs)
        t0 = time.monotonic()
        best_val_loss = float("inf")

        for epoch in range(1, args.epochs + 1):
            # Train
            model.train()
            train_loss = 0.0
            n_batches = 0
            for j_i, b_target in train_loader:
                j_i = j_i.to(device)
                b_target = b_target.to(device)

                b_pred = model(j_i)
                loss = F.mse_loss(b_pred, b_target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            avg_train = train_loss / max(n_batches, 1)

            # Validate
            model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for j_i, b_target in val_loader:
                    j_i = j_i.to(device)
                    b_target = b_target.to(device)

                    b_pred = model(j_i)
                    loss = F.mse_loss(b_pred, b_target)
                    val_loss += loss.item()
                    n_val += 1

            avg_val = val_loss / max(n_val, 1)
            best_val_loss = min(best_val_loss, avg_val)

            logger.info(
                "Epoch %d/%d  train_loss=%.6f  val_loss=%.6f",
                epoch, args.epochs, avg_train, avg_val,
            )

        elapsed = time.monotonic() - t0
        logger.info("Training complete in %.1f s", elapsed)
        logger.info("Best val_loss: %.6f  (threshold: %.2f)", best_val_loss, args.threshold)

        # ---- Verdict ----
        if best_val_loss < args.threshold:
            logger.info("CONV BASELINE CONVERGES")
            print("CONV BASELINE CONVERGES")
        else:
            logger.warning("CONV BASELINE FAILS")
            print("CONV BASELINE FAILS")
            sys.exit(1)

    except Exception as exc:
        logger.error("Sanity check failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
