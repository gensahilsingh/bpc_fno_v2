"""Stage 2: Forward physics-informed neural operator — J_i -> B_pred."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor

from bpc_fno.models.fno_layers import FNOBackbone
from bpc_fno.models.geometry_encoder import VoxelGeometryEncoder
from bpc_fno.models.interfaces import ForwardOperatorInterface


class ForwardPINO(nn.Module, ForwardOperatorInterface):
    """Forward operator mapping impressed current density J_i and geometry
    to predicted magnetic field measurements B_pred at sensor locations.

    The geometry encoder and FNO backbone are **shared** with the inverse
    encoder — they are passed in by reference and must not be re-instantiated.

    Parameters
    ----------
    geometry_encoder:
        Shared :class:`VoxelGeometryEncoder` instance.
    fno_backbone:
        Shared :class:`FNOBackbone` instance.
    config:
        Hydra/OmegaConf configuration.
    """

    def __init__(
        self,
        geometry_encoder: VoxelGeometryEncoder,
        fno_backbone: FNOBackbone,
        config: DictConfig,
    ) -> None:
        super().__init__()
        # Shared modules — stored as plain attributes so they are NOT registered
        # as sub-modules here (the parent BPC_FNO_A owns them).
        self.geometry_encoder = geometry_encoder
        self.fno_backbone = fno_backbone

        c_hidden: int = config.model.n_fno_hidden
        n_sensors_total: int = config.model.get("n_sensors_total", 16 * 3)

        self.n_sensors_total = n_sensors_total

        # Input adapter: project concatenated [J_i, geometry_features] to the
        # FNO backbone's expected input channel count.
        self.input_adapter = nn.Conv3d(3 + c_hidden, c_hidden, kernel_size=1)

        # Sensor projection head using adaptive spatial pooling to keep the
        # parameter count tractable. Outputs (B, n_sensors_total) — one
        # prediction per sensor component for the given input timestep.
        self.sensor_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(4),   # (B, c_hidden, 4, 4, 4)
            nn.Flatten(1),             # (B, c_hidden * 64)
            nn.Linear(c_hidden * 64, 1024),
            nn.GELU(),
            nn.Linear(1024, n_sensors_total),
        )

    def predict_B(self, J_i: Tensor, geometry: Tensor) -> Tensor:
        """Predict sensor B-field from J_i and geometry.

        Args:
            J_i: (B, 3, N, N, N)
            geometry: (B, 4, N, N, N)

        Returns:
            B_pred: (B, n_sensors_total)
        """
        B = J_i.shape[0]

        # Encode geometry (shared encoder)
        geo_features = self.geometry_encoder.encode(geometry)  # (B, C, N, N, N)

        # Concatenate and adapt channels for backbone
        x = torch.cat([J_i, geo_features], dim=1)  # (B, 3+C, N, N, N)
        x = self.input_adapter(x)                   # (B, C, N, N, N)

        # FNO backbone (shared)
        features = self.fno_backbone(x)  # (B, C_out, N, N, N)

        # Project to sensor predictions
        return self.sensor_head(features)  # (B, n_sensors_total)
