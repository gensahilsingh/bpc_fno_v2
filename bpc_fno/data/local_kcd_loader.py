"""Local KCD data loader for wav-format preprocessed recordings.

The local KCD data is stored as individual WAV files per channel at:
  kcd_preprocessed/data/preprocessed/patient_N/mcg_200_channel/trial_N/channel_N.wav

Each wav file contains a single-channel float64 signal at 200 Hz.
Amplitudes are in femtotesla (fT).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile

logger = logging.getLogger(__name__)

# Conversion factor: femtotesla -> Tesla
_FT_TO_TESLA: float = 1e-15


class LocalKCDLoader:
    """Loads KCD recordings from local wav-format preprocessed files.

    Drop-in replacement for :class:`KCDLoader` when fitting noise models
    from local data.

    Parameters
    ----------
    base_dir:
        Path to the preprocessed data root, e.g.
        ``"kcd_preprocessed/data/preprocessed"``.
    """

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)

    def load_record(
        self, patient_id: int, trial_id: int
    ) -> dict[str, np.ndarray | float | int]:
        """Load a single trial's 8-channel recording.

        Parameters
        ----------
        patient_id:
            Patient number (1-7).
        trial_id:
            Trial number (1-25).

        Returns
        -------
        dict
            Keys: ``signal`` (n_samples, 8) in Tesla, ``fs``, ``subject_id``,
            ``trial_id``.
        """
        trial_dir = (
            self.base_dir
            / f"patient_{patient_id}"
            / "mcg_200_channel"
            / f"trial_{trial_id}"
        )
        if not trial_dir.exists():
            raise FileNotFoundError(f"Trial directory not found: {trial_dir}")

        channels: list[np.ndarray] = []
        for ch_idx in range(8):
            wav_path = trial_dir / f"channel_{ch_idx}.wav"
            if not wav_path.exists():
                raise FileNotFoundError(f"Channel file not found: {wav_path}")
            fs, data = wavfile.read(str(wav_path))
            channels.append(data.astype(np.float64))

        # Stack into (n_samples, 8) and convert fT -> Tesla
        signal = np.column_stack(channels) * _FT_TO_TESLA

        return {
            "signal": signal,
            "fs": float(fs),
            "subject_id": patient_id,
            "trial_id": trial_id,
        }

    def load_all(self) -> list[dict[str, np.ndarray | float | int]]:
        """Load all available patient/trial recordings.

        Scans the base directory for patient_N/mcg_200_channel/trial_N
        directories and loads each one.

        Returns
        -------
        list[dict]
            One dict per trial with keys ``signal``, ``fs``, ``subject_id``,
            ``trial_id``.
        """
        records: list[dict] = []
        patient_dirs = sorted(self.base_dir.glob("patient_*"))
        if not patient_dirs:
            raise FileNotFoundError(
                f"No patient directories found in {self.base_dir}"
            )

        for patient_dir in patient_dirs:
            patient_id = int(patient_dir.name.split("_")[1])
            trials_root = patient_dir / "mcg_200_channel"
            if not trials_root.exists():
                logger.warning("No mcg_200_channel in %s, skipping.", patient_dir)
                continue

            trial_dirs = sorted(trials_root.glob("trial_*"))
            for trial_dir in trial_dirs:
                trial_id = int(trial_dir.name.split("_")[1])
                try:
                    rec = self.load_record(patient_id, trial_id)
                    records.append(rec)
                except Exception as exc:
                    logger.warning(
                        "Failed to load patient_%d/trial_%d: %s",
                        patient_id, trial_id, exc,
                    )

        logger.info(
            "Loaded %d records from %d patients.", len(records), len(patient_dirs)
        )
        return records
