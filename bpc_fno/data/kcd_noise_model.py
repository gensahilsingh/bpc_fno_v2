"""OPM noise PSD fitting for realistic synthetic noise generation.

Fits a parametric noise model (white + 1/f^alpha) to inter-beat baseline
segments of KCD magnetocardiography recordings and provides a sampler for
generating noise with the same spectral characteristics.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import welch

logger = logging.getLogger(__name__)


@dataclass
class _ChannelParams:
    """Fitted noise parameters for a single channel."""

    sigma_white: float = 0.0
    sigma_1f: float = 0.0
    alpha: float = 1.0


class OPMNoiseModel:
    """Parametric OPM sensor noise model.

    The power spectral density is modelled as:

    .. math::
        S(f) = \\sigma_{\\mathrm{white}}^2
               + \\frac{\\sigma_{1/f}^2}{f^{\\alpha} + \\epsilon}

    where :math:`\\epsilon` is a small constant to avoid division by zero.
    """

    _EPSILON: float = 1e-12

    def __init__(self) -> None:
        self.channel_params: list[_ChannelParams] = []
        self._measured_psds: list[np.ndarray] | None = None
        self._psd_freqs: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    @staticmethod
    def _psd_model(
        f: np.ndarray,
        sigma_white: float,
        sigma_1f: float,
        alpha: float,
    ) -> np.ndarray:
        """Parametric PSD: white + 1/f^alpha."""
        eps = 1e-12
        return sigma_white ** 2 + sigma_1f ** 2 / (np.abs(f) ** alpha + eps)

    @staticmethod
    def _extract_noise_segments(
        signal: np.ndarray,
        r_peaks: np.ndarray,
        fs: float,
        n_segments: int,
    ) -> list[np.ndarray]:
        """Extract inter-beat baseline segments between T-wave end and P-wave.

        Heuristic timing (from R-peak):
            - T-wave end ~ R + 400 ms
            - Next P-wave start ~ next_R - 200 ms

        Parameters
        ----------
        signal:
            Shape ``(n_samples, n_channels)``.
        r_peaks:
            1-D array of R-peak sample indices.
        fs:
            Sampling frequency.
        n_segments:
            Maximum number of segments to extract.

        Returns
        -------
        list[np.ndarray]
            Each element has shape ``(seg_len, n_channels)``.
        """
        t_end_offset = int(round(0.400 * fs))
        p_start_offset = int(round(0.200 * fs))
        n_samples = signal.shape[0]

        segments: list[np.ndarray] = []
        for i in range(len(r_peaks) - 1):
            seg_start = r_peaks[i] + t_end_offset
            seg_end = r_peaks[i + 1] - p_start_offset

            if seg_start >= seg_end or seg_start < 0 or seg_end > n_samples:
                continue
            # Require at least 64 samples for a useful PSD estimate
            if (seg_end - seg_start) < 64:
                continue

            segments.append(signal[seg_start:seg_end, :].copy())
            if len(segments) >= n_segments:
                break

        return segments

    def fit(
        self,
        source: Any,
        n_noise_segments_per_record: int = 20,
    ) -> None:
        """Fit the noise model from KCD recordings.

        Steps:
        1. Load all records.
        2. Extract inter-beat baseline (noise) segments using R-peak timing.
        3. Compute Welch PSD per channel across all noise segments.
        4. Fit the parametric model to each channel's PSD.

        Parameters
        ----------
        source:
            Either a list of record dicts (each with 'signal', 'fs' keys),
            or a loader object with a ``load_all()`` method.
        n_noise_segments_per_record:
            Max noise segments to extract per record.
        """
        from scipy.signal import find_peaks as _find_peaks

        # Accept either a pre-loaded list of records or a loader object
        if isinstance(source, list):
            records = source
        elif hasattr(source, "load_all"):
            # Try calling with variant kwarg (KCDLoader), fall back to no args (LocalKCDLoader)
            try:
                records = source.load_all(variant="preprocessed")
            except TypeError:
                records = source.load_all()
        else:
            raise TypeError(
                f"Expected list of records or loader with load_all(), got {type(source)}"
            )
        if len(records) == 0:
            raise RuntimeError("No records available for noise model fitting.")

        fs: float = records[0]["fs"]
        n_channels: int = records[0]["signal"].shape[1]

        # Accumulate PSD estimates across all segments
        all_psds: list[np.ndarray] = []  # each (n_freqs, n_channels)
        ref_freqs: np.ndarray | None = None

        for rec in records:
            sig = rec["signal"]
            rec_fs = rec["fs"]

            # R-peak detection (same logic as KCDLoader.extract_beats)
            ptp = np.ptp(sig, axis=0)
            best_ch = int(np.argmax(ptp))
            abs_ref = np.abs(sig[:, best_ch])
            height_thr = np.mean(abs_ref) + 0.5 * np.std(abs_ref)
            peaks, _ = _find_peaks(
                abs_ref, distance=int(0.4 * rec_fs), height=height_thr
            )
            if len(peaks) < 3:
                continue

            segments = self._extract_noise_segments(
                sig, peaks, rec_fs, n_noise_segments_per_record
            )

            for seg in segments:
                # Use fixed nperseg so all segments produce identical
                # frequency grids regardless of segment length.
                fixed_nperseg = 64
                if seg.shape[0] < fixed_nperseg:
                    continue
                freqs, pxx = welch(
                    seg, fs=rec_fs, nperseg=fixed_nperseg, axis=0
                )  # pxx: (n_freqs, n_ch)
                if ref_freqs is None:
                    ref_freqs = freqs
                if len(freqs) == len(ref_freqs):
                    all_psds.append(pxx)

        if not all_psds or ref_freqs is None:
            raise RuntimeError(
                "Could not compute any noise PSDs — check data availability."
            )

        logger.info("Accumulated %d PSD segments across all records.", len(all_psds))
        mean_psd = np.mean(np.stack(all_psds, axis=0), axis=0)  # (n_freqs, n_ch)
        self._psd_freqs = ref_freqs
        self._measured_psds = [mean_psd[:, ch] for ch in range(n_channels)]

        # Fit parametric model per channel.
        # PSD values can be extremely small (e.g. 1e-30) when signals are in
        # Tesla.  We normalise the PSD before curve fitting and rescale back
        # to avoid numerical issues in the optimiser.
        self.channel_params = []
        for ch in range(n_channels):
            psd_ch = mean_psd[:, ch]
            # Skip DC bin (f=0) for fitting
            mask = ref_freqs > 0
            f_fit = ref_freqs[mask]
            psd_fit = psd_ch[mask]

            # Normalise PSD to order-1 for stable curve fitting
            psd_scale = float(np.median(psd_fit)) if np.median(psd_fit) > 0 else 1.0
            psd_norm = psd_fit / psd_scale

            try:
                popt, _ = curve_fit(
                    self._psd_model,
                    f_fit,
                    psd_norm,
                    p0=[np.sqrt(np.median(psd_norm)), np.sqrt(psd_norm[0]), 1.0],
                    bounds=([0.0, 0.0, 0.1], [np.inf, np.inf, 4.0]),
                    maxfev=10_000,
                )
                # Rescale sigma parameters back: PSD = sigma^2, so sigma *= sqrt(scale)
                scale_factor = np.sqrt(psd_scale)
                params = _ChannelParams(
                    sigma_white=float(popt[0] * scale_factor),
                    sigma_1f=float(popt[1] * scale_factor),
                    alpha=float(popt[2]),
                )
            except RuntimeError:
                logger.warning(
                    "Curve fit failed for channel %d; using fallback params.", ch
                )
                params = _ChannelParams(
                    sigma_white=float(np.sqrt(np.median(psd_fit))),
                    sigma_1f=float(np.sqrt(psd_fit[0])),
                    alpha=1.0,
                )

            self.channel_params.append(params)
            logger.info(
                "Channel %d: sigma_white=%.4e, sigma_1f=%.4e, alpha=%.3f",
                ch,
                params.sigma_white,
                params.sigma_1f,
                params.alpha,
            )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        n_channels: int,
        n_timepoints: int,
        fs: float,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Generate synthetic noise with the fitted spectral characteristics.

        Parameters
        ----------
        n_channels:
            Number of channels to generate.
        n_timepoints:
            Number of time-domain samples.
        fs:
            Sampling frequency in Hz.
        rng:
            Optional NumPy random generator for reproducibility.

        Returns
        -------
        np.ndarray
            Shape ``(n_channels, n_timepoints)``.
        """
        if not self.channel_params:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        if rng is None:
            rng = np.random.default_rng()

        n_fft = n_timepoints
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)

        noise = np.zeros((n_channels, n_timepoints), dtype=np.float64)

        for ch in range(n_channels):
            # Cycle through fitted params if n_channels > fitted channels
            params = self.channel_params[ch % len(self.channel_params)]

            # Build target amplitude spectrum
            psd_target = self._psd_model(
                freqs, params.sigma_white, params.sigma_1f, params.alpha
            )
            amplitude = np.sqrt(np.maximum(psd_target, 0.0))

            # Generate white noise in frequency domain
            white = rng.standard_normal(n_fft)
            white_fft = np.fft.rfft(white)

            # Shape the spectrum
            shaped_fft = white_fft * amplitude

            # Inverse FFT to time domain
            noise[ch, :] = np.fft.irfft(shaped_fft, n=n_fft)

        return noise

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save fitted model parameters to a JSON file.

        Parameters
        ----------
        path:
            Output file path (should end in ``.json``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {
            "channels": [
                {
                    "sigma_white": p.sigma_white,
                    "sigma_1f": p.sigma_1f,
                    "alpha": p.alpha,
                }
                for p in self.channel_params
            ]
        }
        path.write_text(json.dumps(data, indent=2))
        logger.info("Noise model saved to %s", path)

    def load(self, path: str | Path) -> None:
        """Load fitted model parameters from a JSON file.

        Parameters
        ----------
        path:
            Path to a previously saved JSON file.
        """
        path = Path(path)
        data = json.loads(path.read_text())
        self.channel_params = [
            _ChannelParams(
                sigma_white=ch["sigma_white"],
                sigma_1f=ch["sigma_1f"],
                alpha=ch["alpha"],
            )
            for ch in data["channels"]
        ]
        logger.info("Noise model loaded from %s (%d channels)", path, len(self.channel_params))

    # ------------------------------------------------------------------
    # Validation plot
    # ------------------------------------------------------------------

    def validate(self, source: Any = None) -> matplotlib.figure.Figure:
        """Plot fitted vs. measured PSD for visual validation.

        If the model was fitted in this session, the stored measured PSD is
        reused.  Otherwise a fresh PSD is estimated from the data.

        Parameters
        ----------
        source:
            A loader object or list of records. Only needed if measured PSD
            is not cached from a prior ``fit()`` call.

        Returns
        -------
        matplotlib.figure.Figure
            A matplotlib figure with one subplot per channel.
        """
        if not self.channel_params:
            raise RuntimeError("Model not fitted / loaded. Nothing to validate.")

        # Re-estimate measured PSD if not cached
        if self._measured_psds is None or self._psd_freqs is None:
            if source is None:
                raise RuntimeError(
                    "No cached PSD data and no source provided for re-estimation."
                )
            self.fit(source)

        assert self._psd_freqs is not None
        assert self._measured_psds is not None

        n_channels = len(self.channel_params)
        n_cols = min(4, n_channels)
        n_rows = (n_channels + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False
        )

        freqs = self._psd_freqs
        mask = freqs > 0

        for ch in range(n_channels):
            ax = axes[ch // n_cols, ch % n_cols]
            measured = self._measured_psds[ch]
            params = self.channel_params[ch]

            fitted = self._psd_model(
                freqs[mask], params.sigma_white, params.sigma_1f, params.alpha
            )

            ax.semilogy(freqs[mask], measured[mask], label="Measured", alpha=0.7)
            ax.semilogy(freqs[mask], fitted, "--", label="Fitted", alpha=0.9)
            ax.set_title(f"Channel {ch}")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("PSD")
            ax.legend(fontsize="small")

        # Hide unused axes
        for idx in range(n_channels, n_rows * n_cols):
            axes[idx // n_cols, idx % n_cols].set_visible(False)

        fig.suptitle("OPM Noise Model: Fitted vs. Measured PSD")
        fig.tight_layout()
        return fig
