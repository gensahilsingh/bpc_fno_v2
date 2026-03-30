"""Core Fourier Neural Operator layers — spectral convolution blocks and backbone."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from bpc_fno.models.interfaces import FNOBackboneInterface


class FNOSpectralBlock(nn.Module):
    """Single FNO spectral convolution block with residual bypass.

    Performs a truncated spectral linear transform in Fourier space and adds
    a pointwise (1x1x1) residual connection, followed by GELU activation.

    Parameters
    ----------
    channels:
        Number of input/output channels (must be equal).
    n_modes:
        Number of Fourier modes to retain along each spatial axis.
    """

    def __init__(self, channels: int, n_modes: int) -> None:
        super().__init__()
        self.channels = channels
        self.n_modes = n_modes
        # Number of modes kept along the last (rfft) axis
        self.n_modes_half = n_modes // 2 + 1

        # Complex spectral weights — stored as a real Parameter pair inside a
        # complex-valued buffer built on-the-fly, or more conveniently as two
        # real Parameters that we combine.  We use nn.Parameter with a complex
        # tensor directly (PyTorch >=1.9 supports complex parameters).
        scale = (2.0 / channels) ** 0.5
        weight_real = torch.empty(
            channels, channels, n_modes, n_modes, self.n_modes_half
        )
        weight_imag = torch.empty_like(weight_real)
        nn.init.xavier_uniform_(weight_real.view(channels, -1)).view_as(weight_real)
        nn.init.xavier_uniform_(weight_imag.view(channels, -1)).view_as(weight_imag)
        self.weight = nn.Parameter(
            torch.complex(weight_real, weight_imag)
        )  # (C, C, m1, m2, m3)

        # Pointwise residual connection
        self.conv = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, C, N, N, N) -> (B, C, N, N, N)"""
        B, C, Nx, Ny, Nz = x.shape

        # --- spectral path ---
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])  # (B, C, Nx, Ny, Nz//2+1)

        # Truncate to low-frequency modes
        m1, m2, m3 = self.n_modes, self.n_modes, self.n_modes_half
        x_ft_trunc = x_ft[:, :, :m1, :m2, :m3]  # (B, C, m1, m2, m3)

        # Complex linear mixing over the channel dimension
        out_ft_trunc = torch.einsum(
            "bicde,oicde->bocde", x_ft_trunc, self.weight
        )  # (B, C, m1, m2, m3)

        # Pad back to full rfft shape
        out_ft = x_ft.new_zeros(B, C, Nx, Ny, Nz // 2 + 1)
        out_ft[:, :, :m1, :m2, :m3] = out_ft_trunc

        x_spectral = torch.fft.irfftn(out_ft, s=(Nx, Ny, Nz))

        # --- residual path ---
        x_local = self.conv(x)

        return F.gelu(x_spectral + x_local)


class FNOBackbone(nn.Module, FNOBackboneInterface):
    """Stack of FNO spectral blocks with input/output projection layers.

    Parameters
    ----------
    c_in:
        Input channels.
    c_out:
        Output channels.
    n_layers:
        Number of spectral blocks.
    n_modes:
        Fourier modes retained per spatial axis in each block.
    c_hidden:
        Hidden channel width inside the spectral blocks.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        n_layers: int,
        n_modes: int,
        c_hidden: int,
    ) -> None:
        super().__init__()
        self.proj_in = nn.Conv3d(c_in, c_hidden, kernel_size=1)
        self.blocks = nn.ModuleList(
            [FNOSpectralBlock(c_hidden, n_modes) for _ in range(n_layers)]
        )
        self.proj_out = nn.Sequential(
            nn.Conv3d(c_hidden, c_out, kernel_size=1),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, C_in, N, N, N) -> (B, C_out, N, N, N)"""
        x = self.proj_in(x)
        for block in self.blocks:
            x = block(x)
        return self.proj_out(x)
