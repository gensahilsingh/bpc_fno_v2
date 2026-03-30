"""Visualization utilities for BPC-FNO evaluation and diagnostics.

Provides publication-ready figures for current-density volumes,
magnetic-field time series, UQ calibration, and training curves.
"""

from __future__ import annotations

from typing import Any

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from bpc_fno.utils.normalization import Normalizer


class MIGVisualizer:
    """Magnetocardiographic Inverse-problem/Geometry Visualizer.

    Parameters
    ----------
    normalizer:
        Optional :class:`Normalizer` instance.  When provided, data arrays
        are assumed to be in normalised units and will be denormalised
        before plotting (labels adjusted accordingly).
    """

    # Shared style defaults
    _CMAP_DIVERGING: str = "RdBu_r"
    _CMAP_SEQUENTIAL: str = "viridis"
    _COMPONENT_LABELS: list[str] = ["$J_x$", "$J_y$", "$J_z$"]

    def __init__(self, normalizer: Normalizer | None = None) -> None:
        self.normalizer = normalizer

    # ------------------------------------------------------------------
    # J_i slice plots
    # ------------------------------------------------------------------

    def plot_J_i_slices(
        self,
        J_i: np.ndarray,
        title: str = "",
        slice_axis: int = 0,
        n_slices: int = 4,
    ) -> matplotlib.figure.Figure:
        """Plot evenly-spaced slices of a current-density volume.

        Args:
            J_i: ``(3, N, N, N)`` single-sample, channels-first.
            title: Optional super-title.
            slice_axis: Spatial axis along which to slice (0, 1, or 2,
                corresponding to the first, second, or third spatial
                dimension of the volume).
            n_slices: Number of equidistant slices to show.

        Returns:
            Matplotlib :class:`~matplotlib.figure.Figure`.
        """
        n_components = J_i.shape[0]
        spatial_size = J_i.shape[1 + slice_axis]  # size along chosen axis
        slice_indices = np.linspace(0, spatial_size - 1, n_slices, dtype=int)

        fig, axes = plt.subplots(
            n_components,
            n_slices,
            figsize=(3 * n_slices, 3 * n_components),
            squeeze=False,
        )

        vmax = np.abs(J_i).max() or 1.0  # guard against all-zero fields

        for row, comp in enumerate(range(n_components)):
            for col, si in enumerate(slice_indices):
                # Extract the 2-D slice
                slicer: list[Any] = [comp, slice(None), slice(None), slice(None)]
                slicer[1 + slice_axis] = si
                img = J_i[tuple(slicer)]

                ax = axes[row, col]
                im = ax.imshow(
                    img,
                    cmap=self._CMAP_DIVERGING,
                    vmin=-vmax,
                    vmax=vmax,
                    origin="lower",
                    interpolation="nearest",
                )
                ax.set_title(
                    f"{self._COMPONENT_LABELS[comp]}  slice {si}",
                    fontsize=9,
                )
                ax.axis("off")

        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="A/cm$^2$")
        if title:
            fig.suptitle(title, fontsize=12, y=1.02)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # J_i comparison (true / pred / error / uncertainty)
    # ------------------------------------------------------------------

    def plot_J_i_comparison(
        self,
        J_i_true: np.ndarray,
        J_i_pred: np.ndarray,
        J_i_std: np.ndarray | None = None,
    ) -> matplotlib.figure.Figure:
        """Side-by-side comparison of true, predicted, error, and uncertainty.

        Displays the mid-slice along the transmural (first spatial) axis.

        Args:
            J_i_true: ``(3, N, N, N)`` ground truth.
            J_i_pred: ``(3, N, N, N)`` model prediction.
            J_i_std: ``(3, N, N, N)`` optional posterior std-dev.

        Returns:
            Matplotlib :class:`~matplotlib.figure.Figure`.
        """
        n_components = J_i_true.shape[0]
        mid = J_i_true.shape[1] // 2  # transmural mid-slice

        has_std = J_i_std is not None
        n_cols = 4 if has_std else 3
        col_titles = ["True", "Predicted", "Absolute Error"]
        if has_std:
            col_titles.append("Posterior Std")

        fig, axes = plt.subplots(
            n_components,
            n_cols,
            figsize=(3.5 * n_cols, 3 * n_components),
            squeeze=False,
        )

        error = np.abs(J_i_true - J_i_pred)
        vmax_field = max(np.abs(J_i_true).max(), np.abs(J_i_pred).max()) or 1.0
        vmax_error = error.max() or 1.0

        for row in range(n_components):
            true_slice = J_i_true[row, mid]
            pred_slice = J_i_pred[row, mid]
            err_slice = error[row, mid]

            # True
            axes[row, 0].imshow(
                true_slice,
                cmap=self._CMAP_DIVERGING,
                vmin=-vmax_field,
                vmax=vmax_field,
                origin="lower",
            )
            # Predicted
            axes[row, 1].imshow(
                pred_slice,
                cmap=self._CMAP_DIVERGING,
                vmin=-vmax_field,
                vmax=vmax_field,
                origin="lower",
            )
            # Error
            axes[row, 2].imshow(
                err_slice,
                cmap="magma",
                vmin=0,
                vmax=vmax_error,
                origin="lower",
            )
            # Uncertainty
            if has_std:
                std_slice = J_i_std[row, mid]  # type: ignore[index]
                axes[row, 3].imshow(
                    std_slice,
                    cmap="magma",
                    vmin=0,
                    vmax=std_slice.max() or 1.0,
                    origin="lower",
                )

            for c in range(n_cols):
                axes[row, c].axis("off")
                if row == 0:
                    axes[row, c].set_title(col_titles[c], fontsize=10)

            axes[row, 0].set_ylabel(
                self._COMPONENT_LABELS[row], fontsize=11, rotation=0, labelpad=30
            )

        fig.suptitle("J$_i$ Reconstruction Comparison (mid-transmural slice)", fontsize=12)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # B-field time-series overlay
    # ------------------------------------------------------------------

    def plot_B_reconstruction(
        self,
        B_true: np.ndarray,
        B_pred: np.ndarray,
    ) -> matplotlib.figure.Figure:
        """Overlay true and predicted magnetic-field sensor signals.

        Args:
            B_true: ``(N_sensors*3, T)`` ground-truth sensor data.
            B_pred: ``(N_sensors*3, T)`` predicted sensor data.

        Returns:
            Matplotlib :class:`~matplotlib.figure.Figure`.
        """
        n_channels, T = B_true.shape
        # Show up to 16 channels in a 4x4 grid; subsample if more
        max_panels = min(n_channels, 16)
        ncols = 4
        nrows = int(np.ceil(max_panels / ncols))

        channel_indices = np.linspace(0, n_channels - 1, max_panels, dtype=int)
        time_axis = np.arange(T)

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(4 * ncols, 2.5 * nrows), squeeze=False
        )

        for idx, ch in enumerate(channel_indices):
            r, c = divmod(idx, ncols)
            ax = axes[r, c]
            ax.plot(time_axis, B_true[ch], "k-", linewidth=1.0, label="True")
            ax.plot(time_axis, B_pred[ch], "r--", linewidth=1.0, label="Pred")
            ax.set_title(f"Ch {ch}", fontsize=9)
            ax.tick_params(labelsize=7)
            if idx == 0:
                ax.legend(fontsize=7)

        # Hide unused panels
        for idx in range(max_panels, nrows * ncols):
            r, c = divmod(idx, ncols)
            axes[r, c].set_visible(False)

        fig.suptitle("B-field Sensor Reconstruction", fontsize=12)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # UQ calibration plot
    # ------------------------------------------------------------------

    def plot_uq_calibration(
        self,
        coverages: list[float],
        alphas: list[float],
    ) -> matplotlib.figure.Figure:
        """Expected-vs-observed coverage calibration plot.

        Args:
            coverages: Observed (empirical) coverage values.
            alphas: Nominal credible-interval levels.

        Returns:
            Matplotlib :class:`~matplotlib.figure.Figure`.
        """
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect calibration")
        ax.plot(alphas, coverages, "o-", color="tab:blue", markersize=5, label="Model")
        ax.set_xlabel("Nominal coverage (alpha)")
        ax.set_ylabel("Observed coverage")
        ax.set_title("UQ Calibration")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Loss curves
    # ------------------------------------------------------------------

    def plot_loss_curves(
        self,
        loss_history: dict[str, list[float]],
    ) -> matplotlib.figure.Figure:
        """Multi-panel plot of all loss components over training.

        Args:
            loss_history: Mapping from loss name (e.g.
                ``"recon_loss"``, ``"kl_loss"``, ``"physics_loss"``) to a
                list of per-epoch (or per-step) scalar values.

        Returns:
            Matplotlib :class:`~matplotlib.figure.Figure`.
        """
        n_losses = len(loss_history)
        if n_losses == 0:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No loss data", ha="center", va="center")
            return fig

        ncols = min(n_losses, 3)
        nrows = int(np.ceil(n_losses / ncols))

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False
        )

        for idx, (name, values) in enumerate(loss_history.items()):
            r, c = divmod(idx, ncols)
            ax = axes[r, c]
            ax.plot(values, linewidth=1.0)
            ax.set_title(name, fontsize=10)
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)

        # Hide unused panels
        for idx in range(n_losses, nrows * ncols):
            r, c = divmod(idx, ncols)
            axes[r, c].set_visible(False)

        fig.suptitle("Training Loss Curves", fontsize=13, y=1.01)
        fig.tight_layout()
        return fig
