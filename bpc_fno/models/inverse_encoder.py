"""Stage 3: Inverse encoder — B_obs -> latent z via VAE."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor

from bpc_fno.models.fno_layers import FNOBackbone
from bpc_fno.models.geometry_encoder import VoxelGeometryEncoder
from bpc_fno.models.interfaces import InverseEncoderInterface


class InverseEncoder(nn.Module, InverseEncoderInterface):
    """Inverse encoder that maps observed B-field sensor measurements and
    geometry to a variational latent space (mu, log_var).

    The geometry encoder and FNO backbone are **shared** with the forward
    operator — they are passed in by reference.

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
        # Shared modules (owned by parent BPC_FNO_A)
        self.geometry_encoder = geometry_encoder
        self.fno_backbone = fno_backbone

        c_hidden: int = config.model.n_fno_hidden
        n_sensors_total: int = config.model.get("n_sensors_total", 16 * 3)
        grid_size: int = config.model.get("grid_size", 32)
        latent_dim: int = config.model.get("latent_dim", 512)

        self.c_hidden = c_hidden
        self.grid_size = grid_size
        self.latent_dim = latent_dim

        # B-field projection: flatten sensor readings into a 3-D feature volume
        # via an MLP that outputs a small 4^3 volume, then trilinear-upsample.
        self.b_projection = nn.Sequential(
            nn.Linear(n_sensors_total, 1024),
            nn.GELU(),
            nn.Linear(1024, c_hidden * 4 * 4 * 4),
        )

        # Input adapter: project concatenated [B_projected, geometry_features]
        # (2 * c_hidden channels) down to the FNO backbone's expected c_in.
        self.input_adapter = nn.Conv3d(2 * c_hidden, c_hidden, kernel_size=1)

        # VAE heads (parallel linear layers for mu and log_var)
        self.mu_head = nn.Linear(c_hidden, latent_dim)
        self.log_var_head = nn.Linear(c_hidden, latent_dim)

    def encode_to_latent(
        self, B_obs: Tensor, geometry: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Encode observed B-field and geometry into VAE latent parameters.

        Args:
            B_obs: (B, N_sensors*3)
            geometry: (B, 4, N, N, N)

        Returns:
            (mu, log_var): each (B, latent_dim)
        """
        B = B_obs.shape[0]
        N = self.grid_size

        # Project B_obs into a volumetric feature map
        b_flat = B_obs.reshape(B, -1)                            # (B, n_sensors*3)
        b_vol = self.b_projection(b_flat)                        # (B, c_hidden*64)
        b_vol = b_vol.view(B, self.c_hidden, 4, 4, 4)           # (B, C, 4, 4, 4)
        b_vol = F.interpolate(
            b_vol, size=(N, N, N), mode="trilinear", align_corners=False
        )  # (B, C, N, N, N)

        # Encode geometry (shared encoder)
        geo_features = self.geometry_encoder.encode(geometry)     # (B, C, N, N, N)

        # Concatenate and adapt channels
        x = torch.cat([b_vol, geo_features], dim=1)              # (B, 2C, N, N, N)
        x = self.input_adapter(x)                                # (B, C, N, N, N)

        # FNO backbone (shared)
        features = self.fno_backbone(x)                          # (B, C_out, N, N, N)

        # Global average pool -> latent parameters
        pooled = features.mean(dim=[-3, -2, -1])                 # (B, C_out)
        mu = self.mu_head(pooled)                                # (B, D)
        log_var = self.log_var_head(pooled).clamp(-10.0, 2.0)   # (B, D)
        return mu, log_var

    def sample_z(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """Reparametrisation trick.  Returns mu during eval.

        Args:
            mu: (B, D)
            log_var: (B, D)

        Returns:
            z: (B, D)
        """
        if self.training:
            std = (0.5 * log_var).exp()
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
