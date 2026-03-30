"""Stage 1: Voxel geometry encoder — pointwise Conv3d network."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from bpc_fno.models.interfaces import GeometryEncoderInterface


class VoxelGeometryEncoder(nn.Module, GeometryEncoderInterface):
    """Encode per-voxel geometry channels (SDF, mu_r, sigma, mask) into a
    dense feature volume using two 1x1x1 convolutions with GELU activation.

    Parameters
    ----------
    n_geometry_channels:
        Number of input geometry channels (default 4).
    hidden_channels:
        Number of output feature channels (default 64).
    """

    def __init__(
        self,
        n_geometry_channels: int = 4,
        hidden_channels: int = 64,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(n_geometry_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=1),
        )

    def encode(self, geometry: Tensor) -> Tensor:
        """geometry: (B, 4, N, N, N) -> encoded: (B, hidden_channels, N, N, N)"""
        return self.net(geometry)
