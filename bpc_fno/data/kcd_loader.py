"""KCD (Kiel Cardio Dataset) WFDB reader and preprocessor.

Provides utilities to download, load, and preprocess magnetocardiography
recordings from the Kiel Cardio Dataset hosted on PhysioNet.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import wfdb
from omegaconf import DictConfig
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default sensor geometry (cm, relative to array centre)
# ---------------------------------------------------------------------------
DEFAULT_SENSOR_POSITIONS_CM: np.ndarray = np.array(
    [
        [0.0, 0.0, 0.0],   # Sensor 0
        [-3.0, 0.0, 3.0],   # Sensor 1
        [-3.0, 0.0, 0.0],   # Sensor 2
        [0.0, 0.0, 3.0],    # Sensor 3
    ],
    dtype=np.float64,
)

# 5x5 measurement grid offsets (cm)
DEFAULT_X_OFFSETS: list[float] = [-12.0, -9.0, -6.0, -3.0, 0.0]
DEFAULT_Z_OFFSETS: list[float] = [0.0, 3.0, 6.0, 9.0, 12.0]

N_SUBJECTS: int = 7
N_TRIALS: int = 25
N_CHANNELS: int = 8  # 4 sensors x 2 components (typically)


class KCDLoader:
    """Reader and preprocessor for the Kiel Cardio Dataset (KCD).

    Parameters
    ----------
    config:
        An OmegaConf ``DictConfig`` expected to contain at least:

        - ``data_dir`` (str): root directory for raw / downloaded data.
        - ``sensor_positions_cm`` (optional list[list[float]]): 4x3 sensor
          positions in centimetres, relative to the array centre.
        - ``x_offsets`` (optional list[float]): 5 x-offsets for the
          measurement grid.
        - ``z_offsets`` (optional list[float]): 5 z-offsets for the
          measurement grid.
    """

    # PhysioNet database slug
    _PHYSIONET_DB: str = "kiel-cardio/1.0.0"

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.data_dir: Path = Path(config.data_dir)

        # Sensor positions within the array (cm)
        if hasattr(config, "sensor_positions_cm") and config.sensor_positions_cm is not None:
            self.sensor_positions_cm: np.ndarray = np.asarray(
                config.sensor_positions_cm, dtype=np.float64
            )
        else:
            self.sensor_positions_cm = DEFAULT_SENSOR_POSITIONS_CM.copy()

        # Grid offsets
        self.x_offsets: list[float] = list(
            getattr(config, "x_offsets", None) or DEFAULT_X_OFFSETS
        )
        self.z_offsets: list[float] = list(
            getattr(config, "z_offsets", None) or DEFAULT_Z_OFFSETS
        )

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download(self, output_dir: str | Path) -> None:
        """Download all KCD records from PhysioNet.

        Parameters
        ----------
        output_dir:
            Local directory to write the downloaded WFDB files into.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading KCD from PhysioNet to %s ...", output_dir)
        wfdb.dl_database(self._PHYSIONET_DB, dl_dir=str(output_dir))
        logger.info("Download complete.")

    # ------------------------------------------------------------------
    # Record loading
    # ------------------------------------------------------------------

    @staticmethod
    def _record_name(subject_id: int, trial_id: int, variant: str) -> str:
        """Build the WFDB record name for a given subject / trial."""
        return f"subject{subject_id}_{variant}_trial{trial_id}"

    @staticmethod
    def _trial_to_grid_index(trial_id: int) -> tuple[int, int]:
        """Map a 1-based trial id to (row=z_idx, col=x_idx).

        trial = row * 5 + col + 1
        """
        idx0 = trial_id - 1
        row = idx0 // 5
        col = idx0 % 5
        return row, col

    def _array_offset_for_trial(self, trial_id: int) -> np.ndarray:
        """Return the (x, y, z) array-centre offset in cm for *trial_id*."""
        row, col = self._trial_to_grid_index(trial_id)
        x = self.x_offsets[col]
        z = self.z_offsets[row]
        return np.array([x, 0.0, z], dtype=np.float64)

    @staticmethod
    def _parse_sensor_positions_from_comments(
        comments: list[str],
    ) -> np.ndarray | None:
        """Attempt to extract sensor positions from WFDB header comments.

        Expected comment format (one per sensor):
            ``sensor_pos <idx> <x> <y> <z>``

        Returns ``None`` when no matching comments are found.
        """
        pattern = re.compile(
            r"sensor_pos\s+(\d+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)"
        )
        positions: dict[int, np.ndarray] = {}
        for line in comments:
            m = pattern.search(line)
            if m:
                idx = int(m.group(1))
                pos = np.array(
                    [float(m.group(2)), float(m.group(3)), float(m.group(4))],
                    dtype=np.float64,
                )
                positions[idx] = pos

        if not positions:
            return None

        n_sensors = max(positions.keys()) + 1
        arr = np.zeros((n_sensors, 3), dtype=np.float64)
        for idx, pos in positions.items():
            arr[idx] = pos
        return arr

    def load_record(
        self,
        subject_id: int,
        trial_id: int,
        variant: str = "preprocessed",
    ) -> dict[str, Any]:
        """Load a single KCD WFDB record.

        Parameters
        ----------
        subject_id:
            Subject number (1-7).
        trial_id:
            Trial number (1-25).
        variant:
            ``'preprocessed'`` or ``'raw'``.

        Returns
        -------
        dict
            Dictionary with keys ``'signal'``, ``'fs'``,
            ``'sensor_positions_cm'``, ``'array_offset_cm'``,
            ``'subject_id'``, ``'trial_id'``.
        """
        rec_name = self._record_name(subject_id, trial_id, variant)
        rec_path = str(self.data_dir / rec_name)
        record: wfdb.Record = wfdb.rdrecord(rec_path)

        signal: np.ndarray = record.p_signal.astype(np.float64)  # (n_samples, n_ch)
        fs: float = float(record.fs)

        # Try to get sensor positions from header comments
        sensor_pos = self._parse_sensor_positions_from_comments(
            record.comments if record.comments else []
        )
        if sensor_pos is None:
            sensor_pos = self.sensor_positions_cm.copy()

        array_offset = self._array_offset_for_trial(trial_id)

        return {
            "signal": signal,
            "fs": fs,
            "sensor_positions_cm": sensor_pos,
            "array_offset_cm": array_offset,
            "subject_id": subject_id,
            "trial_id": trial_id,
        }

    def load_all(self, variant: str = "preprocessed") -> list[dict[str, Any]]:
        """Load all 7 subjects x 25 trials.

        Parameters
        ----------
        variant:
            ``'preprocessed'`` or ``'raw'``.

        Returns
        -------
        list[dict]
            List of record dictionaries (see :meth:`load_record`).
        """
        records: list[dict[str, Any]] = []
        for subj in range(1, N_SUBJECTS + 1):
            for trial in range(1, N_TRIALS + 1):
                try:
                    rec = self.load_record(subj, trial, variant=variant)
                    records.append(rec)
                except Exception:
                    logger.warning(
                        "Failed to load subject=%d trial=%d variant=%s",
                        subj,
                        trial,
                        variant,
                        exc_info=True,
                    )
        logger.info("Loaded %d records (variant=%s).", len(records), variant)
        return records

    # ------------------------------------------------------------------
    # Beat extraction
    # ------------------------------------------------------------------

    def extract_beats(
        self,
        record_dict: dict[str, Any],
        n_beats: int | None = None,
    ) -> np.ndarray:
        """Detect R-peaks and extract individual heartbeat windows.

        R-peak detection is performed on the channel with the highest QRS
        amplitude.  Each beat is windowed from -200 ms to +400 ms around the
        R-peak.  Beats whose preceding inter-beat interval falls outside
        [400 ms, 2000 ms] are rejected.

        Parameters
        ----------
        record_dict:
            Record dictionary as returned by :meth:`load_record`.
        n_beats:
            If given, return at most *n_beats* beats.

        Returns
        -------
        np.ndarray
            Shape ``(n_beats, n_channels, beat_samples)``.
        """
        signal: np.ndarray = record_dict["signal"]  # (n_samples, n_ch)
        fs: float = record_dict["fs"]
        n_samples, n_channels = signal.shape

        # Select channel with largest peak-to-peak (proxy for QRS amplitude)
        ptp = np.ptp(signal, axis=0)
        best_ch: int = int(np.argmax(ptp))
        ref_signal = signal[:, best_ch]

        # R-peak detection
        min_distance = int(0.4 * fs)
        # Use absolute value — R-peaks can be positive or negative
        abs_ref = np.abs(ref_signal)
        height_threshold = np.mean(abs_ref) + 0.5 * np.std(abs_ref)
        peaks, _ = find_peaks(abs_ref, distance=min_distance, height=height_threshold)

        if len(peaks) == 0:
            logger.warning("No R-peaks detected.")
            return np.empty((0, n_channels, 0), dtype=np.float64)

        # Beat window boundaries (samples)
        pre_samples = int(round(0.200 * fs))   # 200 ms before R
        post_samples = int(round(0.400 * fs))   # 400 ms after R
        beat_len = pre_samples + post_samples

        # Inter-beat interval limits (samples)
        ibi_min = int(0.400 * fs)
        ibi_max = int(2.000 * fs)

        beats: list[np.ndarray] = []
        for i, pk in enumerate(peaks):
            # Reject based on inter-beat interval (skip first peak — no prior)
            if i > 0:
                ibi = pk - peaks[i - 1]
                if ibi < ibi_min or ibi > ibi_max:
                    continue

            start = pk - pre_samples
            end = pk + post_samples
            if start < 0 or end > n_samples:
                continue

            # (n_ch, beat_len)
            beat = signal[start:end, :].T.copy()
            beats.append(beat)

            if n_beats is not None and len(beats) >= n_beats:
                break

        if len(beats) == 0:
            return np.empty((0, n_channels, beat_len), dtype=np.float64)

        return np.stack(beats, axis=0)  # (n_beats, n_ch, beat_len)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_signal_statistics(
        self, variant: str = "preprocessed"
    ) -> dict[str, np.ndarray]:
        """Compute per-channel signal statistics across all records.

        Parameters
        ----------
        variant:
            ``'preprocessed'`` or ``'raw'``.

        Returns
        -------
        dict
            Keys: ``'mean'``, ``'std'``, ``'min'``, ``'max'`` — each an
            ndarray of shape ``(n_channels,)``.
        """
        records = self.load_all(variant=variant)
        if len(records) == 0:
            raise RuntimeError("No records loaded; cannot compute statistics.")

        # Determine channel count from first record
        n_ch = records[0]["signal"].shape[1]
        running_sum = np.zeros(n_ch, dtype=np.float64)
        running_sq = np.zeros(n_ch, dtype=np.float64)
        running_min = np.full(n_ch, np.inf)
        running_max = np.full(n_ch, -np.inf)
        total_samples: int = 0

        for rec in records:
            sig = rec["signal"]  # (n_samples, n_ch)
            n = sig.shape[0]
            running_sum += sig.sum(axis=0)
            running_sq += (sig ** 2).sum(axis=0)
            running_min = np.minimum(running_min, sig.min(axis=0))
            running_max = np.maximum(running_max, sig.max(axis=0))
            total_samples += n

        mean = running_sum / total_samples
        std = np.sqrt(running_sq / total_samples - mean ** 2)

        return {
            "mean": mean,
            "std": std,
            "min": running_min,
            "max": running_max,
        }
