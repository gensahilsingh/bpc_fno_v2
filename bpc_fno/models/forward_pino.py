"""Stage 2: Forward physics-informed neural operator — J_i -> B_pred.

Geometry conditioning uses adaLN-Zero. Supports both single-timestep
(B, 3, N, N, N) and full time-series (B, 3, T, N, N, N) inputs.
Time is handled by merging into the batch dimension for the 3D FNO.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from bpc_fno.models.fno_layers import FNOBackbone
from bpc_fno.models.geometry_encoder import VoxelGeometryEncoder
from bpc_fno.models.interfaces import ForwardOperatorInterface


class ForwardPINO(nn.Module, ForwardOperatorInterface):
    """Forward operator: J_i + geometry -> B_pred.

    Shared geometry encoder and FNO backbone with InverseEncoder.
    adaLN-Zero geometry conditioning.
    """

    def __init__(
        self,
        geometry_encoder: VoxelGeometryEncoder,
        fno_backbone: FNOBackbone,
        config: DictConfig,
    ) -> None:
        super().__init__()
        self.geometry_encoder = geometry_encoder
        self.fno_backbone = fno_backbone

        c_hidden: int = config.model.n_fno_hidden
        n_sensors_total: int = config.model.get("n_sensors_total", 48)

        self.n_sensors_total = n_sensors_total

        # Lift J_i to c_hidden channels
        self.input_adapter = nn.Conv3d(3, c_hidden, kernel_size=1)

        # adaLN-Zero conditioning
        self.adaLN_proj = nn.Linear(c_hidden, 2 * c_hidden)
        nn.init.zeros_(self.adaLN_proj.weight)
        nn.init.zeros_(self.adaLN_proj.bias)
        self.norm = nn.GroupNorm(min(8, c_hidden), c_hidden)

        # Sensor head: (B, c_hidden, N, N, N) -> (B, S)
        self.sensor_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(4),
            nn.Flatten(1),
            nn.Linear(c_hidden * 64, 1024),
            nn.GELU(),
            nn.Linear(1024, n_sensors_total),
        )

    def _encode_geometry_condition(self, geometry: Tensor) -> Tensor:
        """Encode geometry once and pool it to a per-sample conditioning vector."""
        geo_features = self.geometry_encoder.encode(geometry)  # (B, C, N, N, N)
        return geo_features.mean(dim=[-3, -2, -1])             # (B, C)

    def _predict_B_single(self, J_i: Tensor, geo_cond: Tensor) -> Tensor:
        """Single-timestep forward: (B, 3, N, N, N) -> (B, S)."""
        B = J_i.shape[0]
        C = self.input_adapter.out_channels

        x = self.input_adapter(J_i)                            # (B, C, N, N, N)

        # adaLN-Zero
        cond = self.adaLN_proj(geo_cond)
        scale, shift = cond.chunk(2, dim=1)
        scale = scale.view(B, C, 1, 1, 1)
        shift = shift.view(B, C, 1, 1, 1)
        x = (1 + scale) * self.norm(x) + shift

        features = self.fno_backbone(x)
        return self.sensor_head(features)  # (B, S)

    def predict_B(self, J_i: Tensor, geometry: Tensor) -> Tensor:
        """Predict B-field. Handles both single-timestep and time-series.

        Args:
            J_i: (B, 3, N, N, N) or (B, 3, T, N, N, N)
            geometry: (B, 4, N, N, N)

        Returns:
            B_pred: (B, S) or (B, S, T)
        """
        if J_i.ndim == 5:
            geo_cond = self._encode_geometry_condition(geometry)
            return self._predict_B_single(J_i, geo_cond)

        # Time-series: (B, 3, T, N, N, N)
        B, C, T, N1, N2, N3 = J_i.shape
        geo_cond = self._encode_geometry_condition(geometry)   # (B, C_hidden)
        # Merge batch and time: (B*T, 3, N, N, N)
        J_flat = J_i.permute(0, 2, 1, 3, 4, 5).reshape(B * T, C, N1, N2, N3)
        # Expand the pooled conditioning vector instead of re-encoding geometry T times
        geo_cond_flat = geo_cond.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)

        B_pred_flat = self._predict_B_single(J_flat, geo_cond_flat)  # (B*T, S)
        S = B_pred_flat.shape[1]
        return B_pred_flat.reshape(B, T, S).permute(0, 2, 1)  # (B, S, T)
