"""Helpers for resolving synthetic data directories in CLI entrypoints."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def resolve_required_data_dir(
    config: DictConfig,
    override: str | None = None,
) -> Path:
    """Resolve the synthetic data directory for CLI scripts.

    The cloud-safe policy is to require an explicit monodomain data path via
    ``--data-dir`` or ``config.data.data_dir``. We intentionally do not fall
    back to the legacy ``config.data.synthetic_dir`` field here.
    """
    raw_value = override
    if raw_value is None:
        raw_value = OmegaConf.select(config, "data.data_dir", default=None)
    if raw_value is None or str(raw_value).strip() == "":
        raise ValueError(
            "Synthetic data directory must be provided via --data-dir or "
            "config.data.data_dir. Refusing to fall back to legacy "
            "config.data.synthetic_dir."
        )

    data_dir = Path(str(raw_value))
    OmegaConf.update(config, "data.data_dir", str(data_dir), force_add=True)
    return data_dir


def validate_sample_data_dir(data_dir: str | Path) -> Path:
    """Validate that a synthetic sample directory exists and is non-empty."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Synthetic data directory does not exist: {data_path}"
        )
    if not any(data_path.glob("sample_*.h5")):
        raise FileNotFoundError(
            f"No sample_*.h5 files found in synthetic data directory: {data_path}"
        )
    return data_path
