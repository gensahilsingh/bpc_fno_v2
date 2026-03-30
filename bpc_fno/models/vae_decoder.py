"""Stage 4: VAE decoder — latent z -> reconstructed J_i volume."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor

from bpc_fno.models.interfaces import DecoderInterface


class _UpsampleBlock(nn.Module):
    """Single spatial-doubling block: either transposed convolution or
    trilinear interpolation + 3x3 conv, followed by InstanceNorm + GELU."""

    def __init__(
        self, c_in: int, c_out: int, use_bilinear: bool = False
    ) -> None:
        super().__init__()
        if use_bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(c_in, c_out, kernel_size=3, padding=1),
            )
        else:
            self.up = nn.ConvTranspose3d(
                c_in, c_out, kernel_size=4, stride=2, padding=1
            )
        self.norm = nn.InstanceNorm3d(c_out)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.up(x)))


class VAEDecoder(nn.Module, DecoderInterface):
    """Convolutional decoder that maps a latent vector *z* and the raw
    geometry volume to a reconstructed impressed-current density J_i.

    Architecture
    ------------
    1. Linear projection from latent to a (decoder_base_channels, 4, 4, 4)
       volume.
    2. Three upsampling blocks: 4 -> 8 -> 16 -> 32, halving channels at each
       step (512 -> 256 -> 128 -> 64).
    3. Concatenation with the raw geometry (4 channels), then a 1x1x1 conv to
       produce 3-channel output (J_ix, J_iy, J_iz).

    Parameters
    ----------
    config:
        Hydra/OmegaConf configuration.  Relevant keys under ``config.model``:
        - ``latent_dim`` (int, default 512)
        - ``decoder_base_channels`` (int, default 512)
        - ``n_decoder_upsamples`` (int, default 3)
        - ``use_bilinear_decoder`` (bool, default False)
        - ``n_geometry_channels`` (int, default 4)
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        latent_dim: int = config.model.get("latent_dim", 512)
        base_ch: int = config.model.get("decoder_base_channels", 512)
        n_upsamples: int = config.model.get("n_decoder_upsamples", 3)
        use_bilinear: bool = config.model.get("use_bilinear_decoder", False)
        n_geometry_channels: int = config.model.get("n_geometry_channels", 4)

        self.base_ch = base_ch

        # Latent -> initial volume
        self.fc = nn.Linear(latent_dim, base_ch * 4 * 4 * 4)

        # Progressive upsampling blocks: 512->256->128->64
        channels = [base_ch]
        for _ in range(n_upsamples):
            channels.append(channels[-1] // 2)
        # channels = [512, 256, 128, 64]

        self.upsample_blocks = nn.ModuleList()
        for i in range(n_upsamples):
            self.upsample_blocks.append(
                _UpsampleBlock(channels[i], channels[i + 1], use_bilinear)
            )

        # Final 1x1 conv: (last_ch + geometry_channels) -> 3
        self.final_conv = nn.Conv3d(
            channels[-1] + n_geometry_channels, 3, kernel_size=1
        )

    def decode(self, z: Tensor, geometry: Tensor) -> Tensor:
        """Decode latent vector into a current-density volume.

        Args:
            z: (B, D)
            geometry: (B, 4, N, N, N)

        Returns:
            J_i_hat: (B, 3, N, N, N) — no output activation (J_i is signed).
        """
        B = z.shape[0]

        # Project and reshape to initial volume
        x = self.fc(z)                            # (B, base_ch * 64)
        x = x.view(B, self.base_ch, 4, 4, 4)     # (B, 512, 4, 4, 4)

        # Progressive upsampling: 4 -> 8 -> 16 -> 32
        for block in self.upsample_blocks:
            x = block(x)                           # (B, 64, 32, 32, 32) after 3 blocks

        # Concatenate geometry and produce output
        x = torch.cat([x, geometry], dim=1)        # (B, 64+4, N, N, N)
        return self.final_conv(x)                  # (B, 3, N, N, N)
