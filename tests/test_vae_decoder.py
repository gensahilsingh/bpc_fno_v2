"""Tests for the VAE decoder: latent z -> reconstructed J_i volume."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import pytest

from bpc_fno.models.vae_decoder import VAEDecoder, _UpsampleBlock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_BATCH_SIZE: int = 2
_LATENT_DIM: int = 64
_GRID_SIZE: int = 32
_BASE_CH: int = 64
_N_GEO_CH: int = 4


def _make_config(use_bilinear: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(
            latent_dim=_LATENT_DIM,
            decoder_base_channels=_BASE_CH,
            n_decoder_upsamples=3,
            use_bilinear_decoder=use_bilinear,
            n_geometry_channels=_N_GEO_CH,
            # SimpleNamespace needs a .get method to mimic DictConfig
        )
    )


def _add_get_method(ns: SimpleNamespace) -> SimpleNamespace:
    """Add a DictConfig-compatible .get method to SimpleNamespace."""

    def _get(self: SimpleNamespace, key: str, default: object = None) -> object:
        return getattr(self, key, default)

    ns.model.get = _get.__get__(ns.model, type(ns.model))
    return ns


@pytest.fixture()
def config_default() -> SimpleNamespace:
    return _add_get_method(_make_config(use_bilinear=False))


@pytest.fixture()
def config_bilinear() -> SimpleNamespace:
    return _add_get_method(_make_config(use_bilinear=True))


@pytest.fixture()
def decoder_default(config_default: SimpleNamespace) -> VAEDecoder:
    torch.manual_seed(0)
    return VAEDecoder(config_default)


@pytest.fixture()
def decoder_bilinear(config_bilinear: SimpleNamespace) -> VAEDecoder:
    torch.manual_seed(0)
    return VAEDecoder(config_bilinear)


@pytest.fixture()
def sample_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    z = torch.randn(_BATCH_SIZE, _LATENT_DIM)
    geometry = torch.randn(_BATCH_SIZE, _N_GEO_CH, _GRID_SIZE, _GRID_SIZE, _GRID_SIZE)
    return z, geometry


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOutputShape:
    """Verify (B, 3, 32, 32, 32) output from latent."""

    def test_output_shape(
        self,
        decoder_default: VAEDecoder,
        sample_inputs: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        z, geometry = sample_inputs
        output = decoder_default.decode(z, geometry)
        assert output.shape == (_BATCH_SIZE, 3, _GRID_SIZE, _GRID_SIZE, _GRID_SIZE), (
            f"Expected ({_BATCH_SIZE}, 3, {_GRID_SIZE}, {_GRID_SIZE}, {_GRID_SIZE}), "
            f"got {output.shape}"
        )


class TestNoFinalActivation:
    """Verify output can be negative (no ReLU/sigmoid at output)."""

    def test_no_final_activation(
        self,
        decoder_default: VAEDecoder,
        sample_inputs: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        z, geometry = sample_inputs
        output = decoder_default.decode(z, geometry)
        has_negative = (output < 0).any().item()
        assert has_negative, (
            "Output has no negative values — final activation may be clipping."
        )


class TestGeometryConcatenation:
    """Verify geometry is concatenated before final layer."""

    def test_geometry_concatenation(
        self, decoder_default: VAEDecoder
    ) -> None:
        """The final_conv should accept (last_ch + n_geometry_channels) input
        channels, confirming geometry is concatenated."""
        final_in_channels = decoder_default.final_conv.in_channels
        # Last upsampled channels: base_ch // 2^n_upsamples
        last_ch = _BASE_CH
        for _ in range(3):
            last_ch = last_ch // 2
        expected_in = last_ch + _N_GEO_CH

        assert final_in_channels == expected_in, (
            f"final_conv in_channels = {final_in_channels}, "
            f"expected {expected_in} = {last_ch} + {_N_GEO_CH}"
        )

    def test_geometry_affects_output(
        self,
        decoder_default: VAEDecoder,
    ) -> None:
        """Changing geometry should change the output."""
        torch.manual_seed(42)
        z = torch.randn(1, _LATENT_DIM)
        geo1 = torch.randn(1, _N_GEO_CH, _GRID_SIZE, _GRID_SIZE, _GRID_SIZE)
        geo2 = torch.randn(1, _N_GEO_CH, _GRID_SIZE, _GRID_SIZE, _GRID_SIZE)

        decoder_default.eval()
        out1 = decoder_default.decode(z, geo1)
        out2 = decoder_default.decode(z, geo2)

        assert not torch.allclose(out1, out2, atol=1e-6), (
            "Different geometries should produce different outputs."
        )


class TestBilinearMode:
    """If use_bilinear=True, verify no ConvTranspose3d layers."""

    def test_bilinear_mode(self, decoder_bilinear: VAEDecoder) -> None:
        for i, block in enumerate(decoder_bilinear.upsample_blocks):
            has_conv_transpose = any(
                isinstance(m, torch.nn.ConvTranspose3d)
                for m in block.modules()
            )
            assert not has_conv_transpose, (
                f"Upsample block {i} contains ConvTranspose3d "
                f"despite use_bilinear=True."
            )

    def test_bilinear_output_shape(
        self,
        decoder_bilinear: VAEDecoder,
        sample_inputs: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        z, geometry = sample_inputs
        output = decoder_bilinear.decode(z, geometry)
        assert output.shape == (_BATCH_SIZE, 3, _GRID_SIZE, _GRID_SIZE, _GRID_SIZE)

    def test_non_bilinear_has_conv_transpose(
        self, decoder_default: VAEDecoder
    ) -> None:
        has_conv_transpose = any(
            isinstance(m, torch.nn.ConvTranspose3d)
            for m in decoder_default.modules()
        )
        assert has_conv_transpose, (
            "Default decoder should use ConvTranspose3d for upsampling."
        )


class TestDecoderDeterministic:
    """Same input produces same output in eval mode."""

    def test_decoder_deterministic(
        self, decoder_default: VAEDecoder
    ) -> None:
        decoder_default.eval()
        torch.manual_seed(99)
        z = torch.randn(1, _LATENT_DIM)
        geo = torch.randn(1, _N_GEO_CH, _GRID_SIZE, _GRID_SIZE, _GRID_SIZE)

        out1 = decoder_default.decode(z, geo)
        out2 = decoder_default.decode(z, geo)

        torch.testing.assert_close(out1, out2), (
            "Decoder should be deterministic in eval mode."
        )
