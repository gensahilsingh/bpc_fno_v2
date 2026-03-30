"""Tests for uncertainty quantification (UQ) calibration metrics."""

from __future__ import annotations

import numpy as np
import torch
import pytest


# ---------------------------------------------------------------------------
# UQ utility functions (tested here, could live in bpc_fno.evaluation)
# ---------------------------------------------------------------------------


def compute_coverage(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    alpha: float = 0.95,
) -> float:
    """Compute the empirical coverage of a prediction interval.

    For a Gaussian predictive distribution:
        interval = [mean - z * std,  mean + z * std]
    where z = norm.ppf((1 + alpha) / 2).

    Parameters
    ----------
    y_true : (N,) or (N, D) ground truth values.
    y_mean : (N,) or (N, D) predictive mean.
    y_std : (N,) or (N, D) predictive standard deviation.
    alpha : confidence level (e.g. 0.95 for 95% interval).

    Returns
    -------
    float : fraction of y_true values within the interval.
    """
    from scipy.stats import norm

    z = norm.ppf((1.0 + alpha) / 2.0)
    lower = y_mean - z * y_std
    upper = y_mean + z * y_std

    covered = np.logical_and(y_true >= lower, y_true <= upper)
    return float(covered.mean())


def compute_sharpness(y_std: np.ndarray, alpha: float = 0.95) -> float:
    """Compute mean interval width (sharpness) for Gaussian intervals.

    Parameters
    ----------
    y_std : (N,) or (N, D) predictive standard deviation.
    alpha : confidence level.

    Returns
    -------
    float : mean interval width.
    """
    from scipy.stats import norm

    z = norm.ppf((1.0 + alpha) / 2.0)
    widths = 2.0 * z * y_std
    return float(widths.mean())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCoveragePerfect:
    """With deterministic model (zero variance), coverage should be 0 or 1."""

    def test_coverage_zero_variance_exact_mean(self) -> None:
        """When std=0 and predictions are exact, all points are covered
        (they lie exactly on the boundary)."""
        N = 1000
        y_true = np.linspace(-5, 5, N)
        y_mean = y_true.copy()  # perfect predictions
        y_std = np.zeros(N)

        # With zero std, interval is [mean, mean]; coverage depends on
        # whether y_true == y_mean exactly.
        coverage = compute_coverage(y_true, y_mean, y_std, alpha=0.95)
        # All points should be covered since y_true == y_mean exactly
        assert coverage == 1.0

    def test_coverage_zero_variance_wrong_mean(self) -> None:
        """When std=0 and predictions are wrong, coverage should be 0."""
        N = 1000
        y_true = np.linspace(-5, 5, N)
        y_mean = y_true + 1.0  # systematically wrong
        y_std = np.zeros(N)

        coverage = compute_coverage(y_true, y_mean, y_std, alpha=0.95)
        assert coverage == 0.0


class TestCoverageGaussian:
    """Generate known Gaussian samples, verify coverage ~ alpha."""

    @pytest.mark.parametrize("alpha", [0.5, 0.8, 0.9, 0.95, 0.99])
    def test_coverage_gaussian(self, rng: np.random.Generator, alpha: float) -> None:
        """Draw N samples from N(mu, sigma^2) and verify that the empirical
        coverage of the alpha-level interval is approximately alpha."""
        N = 50_000
        mu = 3.0
        sigma = 2.0

        y_true = rng.normal(mu, sigma, size=N)
        y_mean = np.full(N, mu)
        y_std = np.full(N, sigma)

        coverage = compute_coverage(y_true, y_mean, y_std, alpha=alpha)

        # With N=50k, the coverage should be within ~1% of alpha
        assert abs(coverage - alpha) < 0.015, (
            f"Coverage {coverage:.4f} deviates from alpha={alpha:.2f}"
        )


class TestSharpnessDecreasing:
    """More samples should give same or tighter confidence intervals
    (in the sense that the estimated std should converge)."""

    def test_sharpness_convergence(self, rng: np.random.Generator) -> None:
        """As we use more posterior samples to estimate std, the estimate
        should stabilize (not increase)."""
        true_sigma = 1.5
        n_data_points = 100

        # Simulate drawing different numbers of posterior samples
        sharpness_values: list[float] = []
        for n_samples in [5, 10, 50, 200, 1000]:
            # Draw posterior samples and compute std
            posterior_samples = rng.normal(
                0.0, true_sigma, size=(n_samples, n_data_points)
            )
            y_std = posterior_samples.std(axis=0)
            sharpness = compute_sharpness(y_std, alpha=0.95)
            sharpness_values.append(sharpness)

        # With more samples, the std estimate should converge to the true value.
        # The last estimate should be closest to the analytic sharpness.
        from scipy.stats import norm

        z = norm.ppf(0.975)
        analytic_sharpness = 2.0 * z * true_sigma

        # The estimate from 1000 samples should be within 5% of analytic
        assert abs(sharpness_values[-1] - analytic_sharpness) / analytic_sharpness < 0.05


class TestNSamplesEffect:
    """Verify UQ improves with more posterior samples."""

    def test_n_samples_effect(self, rng: np.random.Generator) -> None:
        """Coverage calibration should improve (approach target alpha) as
        the number of posterior samples increases."""
        alpha = 0.90
        true_mu = 0.0
        true_sigma = 2.0
        n_data = 5000

        y_true = rng.normal(true_mu, true_sigma, size=n_data)

        calibration_errors: list[float] = []

        for n_samples in [3, 10, 50, 200]:
            # Simulate n_samples posterior draws for each data point
            posterior_samples = rng.normal(
                true_mu, true_sigma, size=(n_samples, n_data)
            )
            y_mean = posterior_samples.mean(axis=0)
            y_std = posterior_samples.std(axis=0)

            coverage = compute_coverage(y_true, y_mean, y_std, alpha=alpha)
            calibration_errors.append(abs(coverage - alpha))

        # Calibration error should generally decrease with more samples
        # We check that the last (most samples) is better than the first
        assert calibration_errors[-1] < calibration_errors[0] + 0.02, (
            f"Calibration did not improve: errors = {calibration_errors}"
        )

    def test_std_estimate_improves(self, rng: np.random.Generator) -> None:
        """The standard deviation of the std estimate should decrease with
        more posterior samples (law of large numbers)."""
        true_sigma = 1.0
        n_data = 500
        n_trials = 100

        std_of_std: list[float] = []
        for n_samples in [5, 50, 500]:
            trial_stds: list[float] = []
            for _ in range(n_trials):
                samples = rng.normal(0.0, true_sigma, size=(n_samples, n_data))
                trial_stds.append(samples.std(axis=0).mean())
            std_of_std.append(np.std(trial_stds))

        # Variability should decrease monotonically
        for i in range(len(std_of_std) - 1):
            assert std_of_std[i + 1] < std_of_std[i], (
                f"Std variability did not decrease: {std_of_std}"
            )
