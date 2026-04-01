"""Stage 2: Forward physics-informed neural operator — J_i -> B_pred.

Geometry conditioning uses adaLN-Zero (Adaptive Layer Normalization with
Zero initialisation, from the DiT paper) instead of channel concatenation.
The geometry encoder output is global-average-pooled to a (B, C) vector,
then projected to per-channel scale and shift parameters that modulate
the lifted J_i features *before* the FNO backbone.
"""

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

    Geometry conditioning is applied via adaLN-Zero: the geometry encoder
    output is globally pooled to a conditioning vector, projected to
    per-channel (scale, shift) pairs, and used to modulate the lifted
    input features.

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

        # Input adapter: lift 3-channel J_i to c_hidden channels (no geometry
        # channels concatenated — geometry conditioning is via adaLN-Zero).
        self.input_adapter = nn.Conv3d(3, c_hidden, kernel_size=1)

        # adaLN-Zero conditioning layers.
        # geo_embed_dim == c_hidden (output of VoxelGeometryEncoder).
        self.adaLN_proj = nn.Linear(c_hidden, 2 * c_hidden)
        nn.init.zeros_(self.adaLN_proj.weight)
        nn.init.zeros_(self.adaLN_proj.bias)
        self.norm = nn.GroupNorm(
            num_groups=min(8, c_hidden), num_channels=c_hidden
        )

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
        C = self.input_adapter.out_channels  # c_hidden

        # Encode geometry (shared encoder) and global-average-pool
        geo_features = self.geometry_encoder.encode(geometry)  # (B, C, N, N, N)
        geo_cond = geo_features.mean(dim=[-3, -2, -1])        # (B, C)

        # Lift J_i to c_hidden channels
        x = self.input_adapter(J_i)                            # (B, C, N, N, N)

        # adaLN-Zero conditioning
        cond = self.adaLN_proj(geo_cond)                       # (B, 2*C)
        scale, shift = cond.chunk(2, dim=1)
        scale = scale.view(B, C, 1, 1, 1)
        shift = shift.view(B, C, 1, 1, 1)
        x = self.norm(x)
        x = (1 + scale) * x + shift                           # (B, C, N, N, N)

        # FNO backbone (shared)
        features = self.fno_backbone(x)  # (B, C_out, N, N, N)

        # Project to sensor predictions
        return self.sensor_head(features)  # (B, n_sensors_total)
