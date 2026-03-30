"""Tests for the KCD (Kiel Cardio Dataset) WFDB loader and preprocessor."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bpc_fno.data.kcd_loader import KCDLoader


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_config() -> SimpleNamespace:
    """Minimal OmegaConf-like config for KCDLoader."""
    return SimpleNamespace(
        data_dir="/tmp/fake_kcd",
        sensor_positions_cm=None,
        x_offsets=None,
        z_offsets=None,
    )


@pytest.fixture()
def loader(mock_config: SimpleNamespace) -> KCDLoader:
    return KCDLoader(mock_config)


def _make_record_dict(
    fs: float = 1000.0,
    n_samples: int = 10_000,
    n_channels: int = 8,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Create a synthetic record dict mimicking KCDLoader.load_record output."""
    if rng is None:
        rng = np.random.default_rng(42)
    signal = rng.standard_normal((n_samples, n_channels)).astype(np.float64)
    return {
        "signal": signal,
        "fs": fs,
        "sensor_positions_cm": np.zeros((4, 3), dtype=np.float64),
        "array_offset_cm": np.array([0.0, 0.0, 0.0]),
        "subject_id": 1,
        "trial_id": 1,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadRecordShape:
    """Mock a WFDB record and verify output dict keys and shapes."""

    @patch("bpc_fno.data.kcd_loader.wfdb.rdrecord")
    def test_load_record_shape(
        self, mock_rdrecord: MagicMock, loader: KCDLoader
    ) -> None:
        n_samples, n_channels = 5000, 8
        mock_record = MagicMock()
        mock_record.p_signal = np.random.default_rng(0).standard_normal(
            (n_samples, n_channels)
        )
        mock_record.fs = 1000.0
        mock_record.comments = []
        mock_rdrecord.return_value = mock_record

        result = loader.load_record(subject_id=1, trial_id=1)

        assert set(result.keys()) == {
            "signal",
            "fs",
            "sensor_positions_cm",
            "array_offset_cm",
            "subject_id",
            "trial_id",
        }
        assert result["signal"].shape == (n_samples, n_channels)
        assert result["signal"].dtype == np.float64
        assert isinstance(result["fs"], float)
        assert result["sensor_positions_cm"].shape[1] == 3
        assert result["array_offset_cm"].shape == (3,)


class TestExtractBeatsShape:
    """Mock signal with known R-peaks, verify beat extraction."""

    def test_extract_beats_shape(self, loader: KCDLoader) -> None:
        fs = 1000.0
        n_samples = 10_000
        n_channels = 4
        signal = np.zeros((n_samples, n_channels), dtype=np.float64)

        # Place synthetic R-peaks at regular 800 ms intervals
        r_peak_interval = int(0.8 * fs)  # 800 samples
        peak_positions = list(range(1000, n_samples - 1000, r_peak_interval))
        for pos in peak_positions:
            signal[pos, :] = 10.0  # sharp spike

        record_dict: dict[str, Any] = {
            "signal": signal,
            "fs": fs,
            "sensor_positions_cm": np.zeros((4, 3)),
            "array_offset_cm": np.zeros(3),
            "subject_id": 1,
            "trial_id": 1,
        }

        beats = loader.extract_beats(record_dict)

        # beat_len = pre_samples + post_samples = 200 + 400 = 600 at 1 kHz
        expected_beat_len = int(0.2 * fs) + int(0.4 * fs)
        assert beats.ndim == 3
        assert beats.shape[1] == n_channels
        assert beats.shape[2] == expected_beat_len
        assert beats.shape[0] > 0  # at least some beats detected

    def test_extract_beats_n_beats_limit(self, loader: KCDLoader) -> None:
        fs = 1000.0
        n_samples = 10_000
        signal = np.zeros((n_samples, 4), dtype=np.float64)
        for pos in range(1000, n_samples - 1000, 800):
            signal[pos, :] = 10.0

        record_dict: dict[str, Any] = {
            "signal": signal,
            "fs": fs,
            "sensor_positions_cm": np.zeros((4, 3)),
            "array_offset_cm": np.zeros(3),
            "subject_id": 1,
            "trial_id": 1,
        }

        beats = loader.extract_beats(record_dict, n_beats=3)
        assert beats.shape[0] <= 3


class TestBeatRejection:
    """Verify beats with <400ms or >2000ms intervals are rejected."""

    def test_short_interval_rejected(self, loader: KCDLoader) -> None:
        """Beats separated by less than 400ms should be rejected."""
        fs = 1000.0
        n_samples = 10_000
        signal = np.zeros((n_samples, 4), dtype=np.float64)

        # First peak at 500, second at 700 (200ms apart = too short)
        # Third at 1600 (900ms from second = OK)
        # Fourth at 2500 (900ms from third = OK)
        peaks = [500, 700, 1600, 2500]
        for pos in peaks:
            signal[pos, :] = 10.0

        record_dict: dict[str, Any] = {
            "signal": signal,
            "fs": fs,
            "sensor_positions_cm": np.zeros((4, 3)),
            "array_offset_cm": np.zeros(3),
            "subject_id": 1,
            "trial_id": 1,
        }

        beats = loader.extract_beats(record_dict)
        # The 200ms interval beat should be rejected; we should get fewer
        # beats than total peaks.
        assert beats.shape[0] < len(peaks)

    def test_long_interval_rejected(self, loader: KCDLoader) -> None:
        """Beats separated by more than 2000ms should be rejected."""
        fs = 1000.0
        n_samples = 20_000
        signal = np.zeros((n_samples, 4), dtype=np.float64)

        # Two peaks separated by 3000ms (too long)
        peaks = [1000, 4000, 5000]
        for pos in peaks:
            signal[pos, :] = 10.0

        record_dict: dict[str, Any] = {
            "signal": signal,
            "fs": fs,
            "sensor_positions_cm": np.zeros((4, 3)),
            "array_offset_cm": np.zeros(3),
            "subject_id": 1,
            "trial_id": 1,
        }

        beats = loader.extract_beats(record_dict)
        # The 3000ms interval beat (at 4000) should be rejected
        # Only the first peak (no prior) and the 1000ms-apart peak should pass
        assert beats.shape[0] <= 2


class TestSignalStatistics:
    """Verify stats computation on mock data."""

    @patch.object(KCDLoader, "load_all")
    def test_signal_statistics(
        self, mock_load_all: MagicMock, loader: KCDLoader
    ) -> None:
        rng = np.random.default_rng(42)
        n_channels = 4

        # Create two mock records with known statistics
        sig1 = rng.standard_normal((1000, n_channels))
        sig2 = rng.standard_normal((2000, n_channels))

        mock_load_all.return_value = [
            {"signal": sig1},
            {"signal": sig2},
        ]

        stats = loader.get_signal_statistics()

        assert set(stats.keys()) == {"mean", "std", "min", "max"}
        assert stats["mean"].shape == (n_channels,)
        assert stats["std"].shape == (n_channels,)
        assert stats["min"].shape == (n_channels,)
        assert stats["max"].shape == (n_channels,)

        # Verify against manual computation
        all_sig = np.concatenate([sig1, sig2], axis=0)
        np.testing.assert_allclose(stats["mean"], all_sig.mean(axis=0), atol=1e-10)
        np.testing.assert_allclose(
            stats["min"], np.minimum(sig1.min(axis=0), sig2.min(axis=0)), atol=1e-10
        )
        np.testing.assert_allclose(
            stats["max"], np.maximum(sig1.max(axis=0), sig2.max(axis=0)), atol=1e-10
        )
