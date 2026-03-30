"""Tests for FNO (Fourier Neural Operator) layers: spectral blocks and backbone."""

from __future__ import annotations

import torch
import pytest

from bpc_fno.models.fno_layers import FNOBackbone, FNOSpectralBlock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_GRID_SIZE: int = 8
_CHANNELS: int = 16
_N_MODES: int = 4
_BATCH_SIZE: int = 2


@pytest.fixture()
def spectral_block() -> FNOSpectralBlock:
    torch.manual_seed(0)
    return FNOSpectralBlock(channels=_CHANNELS, n_modes=_N_MODES)


@pytest.fixture()
def fno_backbone() -> FNOBackbone:
    torch.manual_seed(0)
    return FNOBackbone(
        c_in=3,
        c_out=3,
        n_layers=2,
        n_modes=_N_MODES,
        c_hidden=_CHANNELS,
    )


@pytest.fixture()
def random_input() -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(_BATCH_SIZE, _CHANNELS, _GRID_SIZE, _GRID_SIZE, _GRID_SIZE)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSpectralBlockShape:
    """Verify input/output shapes match for FNOSpectralBlock."""

    def test_spectral_block_shape(
        self, spectral_block: FNOSpectralBlock, random_input: torch.Tensor
    ) -> None:
        output = spectral_block(random_input)
        assert output.shape == random_input.shape, (
            f"Expected shape {random_input.shape}, got {output.shape}"
        )

    def test_spectral_block_different_grid_sizes(
        self, spectral_block: FNOSpectralBlock
    ) -> None:
        """Block should work with grids >= n_modes."""
        for N in [8, 12, 16]:
            x = torch.randn(1, _CHANNELS, N, N, N)
            out = spectral_block(x)
            assert out.shape == x.shape


class TestFNOBackboneShape:
    """Verify end-to-end shape of the FNO backbone."""

    def test_fno_backbone_shape(self, fno_backbone: FNOBackbone) -> None:
        B, C_in, N = _BATCH_SIZE, 3, _GRID_SIZE
        x = torch.randn(B, C_in, N, N, N)
        output = fno_backbone(x)
        assert output.shape == (B, 3, N, N, N), (
            f"Expected (B, 3, N, N, N), got {output.shape}"
        )


class TestSpectralBlockGradient:
    """Verify gradients flow through the spectral path."""

    def test_spectral_block_gradient(
        self, spectral_block: FNOSpectralBlock
    ) -> None:
        x = torch.randn(
            1, _CHANNELS, _GRID_SIZE, _GRID_SIZE, _GRID_SIZE, requires_grad=True
        )
        output = spectral_block(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None, "No gradient computed for input."
        assert x.grad.abs().sum() > 0, "Gradient is all zeros."

        # Verify spectral weight has gradient
        assert spectral_block.weight.grad is not None, (
            "Spectral weight has no gradient."
        )
        assert spectral_block.weight.grad.abs().sum() > 0, (
            "Spectral weight gradient is all zeros."
        )

    def test_backbone_gradient(self, fno_backbone: FNOBackbone) -> None:
        x = torch.randn(
            1, 3, _GRID_SIZE, _GRID_SIZE, _GRID_SIZE, requires_grad=True
        )
        output = fno_backbone(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        n_params_with_grad = sum(
            1 for p in fno_backbone.parameters() if p.grad is not None
        )
        n_params_total = sum(1 for _ in fno_backbone.parameters())
        assert n_params_with_grad == n_params_total, (
            f"Only {n_params_with_grad}/{n_params_total} params received gradients."
        )


class TestModeTruncation:
    """Verify only low-frequency modes are used."""

    def test_mode_truncation(self) -> None:
        """When spectral weights approximate identity, high-frequency output
        should be near zero (since those modes are zeroed out in the spectral path).
        """
        C = 8
        n_modes = 4
        N = 16  # larger grid to clearly separate low/high freq

        block = FNOSpectralBlock(channels=C, n_modes=n_modes)

        # Set spectral weights to identity-like (channel mixing = I)
        with torch.no_grad():
            block.weight.zero_()
            n_modes_half = n_modes // 2 + 1
            for c in range(C):
                block.weight[c, c, :, :, :] = 1.0 + 0.0j

            # Zero out the residual conv path
            block.conv.weight.zero_()
            block.conv.bias.zero_()

        # Create input with only high-frequency content
        x_high = torch.zeros(1, C, N, N, N)
        # Add a checkerboard pattern (Nyquist frequency)
        indices = torch.arange(N)
        checker = ((-1.0) ** indices).unsqueeze(0).unsqueeze(0)  # (1, 1, N)
        x_high[:, :, :, :, :] = (
            checker[:, :, :, None, None]
            * checker[:, :, None, :, None]
            * checker[:, :, None, None, :]
        )

        output = block(x_high)

        # The spectral path should truncate high frequencies, and the residual
        # path is zeroed, so output should be close to zero after GELU
        # (GELU(~0) ~ 0 for small inputs).
        # We check that the high-frequency content is substantially reduced.
        x_high_energy = x_high.pow(2).sum().item()
        out_energy = output.pow(2).sum().item()

        assert out_energy < 0.1 * x_high_energy, (
            f"High-freq energy not sufficiently attenuated: "
            f"input={x_high_energy:.2f}, output={out_energy:.2f}"
        )


class TestResidualPath:
    """Verify local convolution (residual) path works independently."""

    def test_residual_path(self) -> None:
        C = 8
        block = FNOSpectralBlock(channels=C, n_modes=4)

        # Zero out spectral weights to isolate residual path
        with torch.no_grad():
            block.weight.zero_()

        x = torch.randn(1, C, _GRID_SIZE, _GRID_SIZE, _GRID_SIZE)
        output = block(x)

        # Output should be non-trivial (from the conv residual path + GELU)
        assert output.shape == x.shape
        # The 1x1 conv with default init should produce non-zero output
        # through GELU(conv(x))
        assert output.abs().sum() > 0, "Residual path produced all zeros."
