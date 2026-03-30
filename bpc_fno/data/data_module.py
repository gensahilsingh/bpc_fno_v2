"""PyTorch Lightning DataModule for the BPC-FNO pipeline.

Wraps :class:`~bpc_fno.data.synthetic_dataset.SyntheticMIGDataset` instances
for train / val / test splits behind a single :class:`LightningDataModule`.
"""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from bpc_fno.data.synthetic_dataset import Normalizer, SyntheticMIGDataset


class BPCFNODataModule(pl.LightningDataModule):
    """Lightning DataModule for BPC-FNO synthetic MIG data.

    Parameters
    ----------
    config:
        OmegaConf configuration.  Expected keys:

        - ``data.data_dir`` (str): path to the directory with HDF5 samples.
        - ``data.batch_size`` (int): batch size for all dataloaders.
        - ``data.num_workers`` (int): number of dataloader workers.
        - ``data.pin_memory`` (bool): whether to pin memory for GPU transfer.
    normalizer:
        Object implementing the :class:`Normalizer` protocol.
    """

    def __init__(self, config: DictConfig, normalizer: Normalizer) -> None:
        super().__init__()
        self.config = config
        self.normalizer = normalizer

        self._train_ds: SyntheticMIGDataset | None = None
        self._val_ds: SyntheticMIGDataset | None = None
        self._test_ds: SyntheticMIGDataset | None = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, stage: str | None = None) -> None:
        """Create dataset instances for the requested stage.

        Parameters
        ----------
        stage:
            One of ``'fit'``, ``'validate'``, ``'test'``, or ``None``
            (creates all).
        """
        data_dir: str = self.config.data.get(
            "data_dir", self.config.data.get("synthetic_dir", "data/synthetic")
        )

        if stage in ("fit", None):
            self._train_ds = SyntheticMIGDataset(
                data_dir=data_dir,
                split="train",
                normalizer=self.normalizer,
                config=self.config,
            )
            self._val_ds = SyntheticMIGDataset(
                data_dir=data_dir,
                split="val",
                normalizer=self.normalizer,
                config=self.config,
            )

        if stage in ("validate", None):
            if self._val_ds is None:
                self._val_ds = SyntheticMIGDataset(
                    data_dir=data_dir,
                    split="val",
                    normalizer=self.normalizer,
                    config=self.config,
                )

        if stage in ("test", None):
            self._test_ds = SyntheticMIGDataset(
                data_dir=data_dir,
                split="test",
                normalizer=self.normalizer,
                config=self.config,
            )

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------

    def _common_loader_kwargs(self) -> dict[str, Any]:
        """Return shared DataLoader keyword arguments from config."""
        return {
            "batch_size": int(self.config.data.get(
                "batch_size", self.config.training.get("batch_size", 16)
            )),
            "num_workers": int(self.config.data.get("num_workers", 0)),
            "pin_memory": bool(self.config.data.get("pin_memory", True)),
        }

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader."""
        assert self._train_ds is not None, "Call setup('fit') first."
        return DataLoader(
            self._train_ds,
            shuffle=True,
            drop_last=True,
            **self._common_loader_kwargs(),
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader."""
        assert self._val_ds is not None, "Call setup('fit') or setup('validate') first."
        return DataLoader(
            self._val_ds,
            shuffle=False,
            drop_last=False,
            **self._common_loader_kwargs(),
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test DataLoader."""
        assert self._test_ds is not None, "Call setup('test') first."
        return DataLoader(
            self._test_ds,
            shuffle=False,
            drop_last=False,
            **self._common_loader_kwargs(),
        )
