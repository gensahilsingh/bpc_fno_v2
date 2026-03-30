"""CellML model loader for TT2006 ionic models.

Downloads, caches, and validates CellML files using Myokit.
Supports endocardial, midmyocardial, and epicardial cell variants.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Any

import myokit
import myokit.formats.cellml
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

# Required membrane currents that must be present in a valid TT2006 model.
# NEW - matches actual TT2006 CellML variable naming
REQUIRED_CURRENTS: tuple[str, ...] = (
    "i_Na",
    "i_CaL",
    "i_Kr", 
    "i_Ks",
    "i_K1",
    "i_to",
)

VALID_CELL_TYPES: tuple[str, ...] = ("endo", "mid", "epi")


class CellMLLoader:
    """Downloads and caches TT2006 CellML models via Myokit.

    Parameters
    ----------
    config : DictConfig
        Hydra/OmegaConf configuration.  Expected keys under
        ``config.simulation.ionic``:

        - ``cellml_url_endo`` – URL for endocardial CellML file.
        - ``cellml_url_mid``  – URL for midmyocardial CellML file.
        - ``cellml_url_epi``  – URL for epicardial CellML file.
        - ``cache_dir``       – (optional) local cache directory,
          defaults to ``data/cellml/``.
    """

    def __init__(self, config: DictConfig) -> None:
        # Handle both full config and pre-sliced ionic config
        if hasattr(config, 'simulation'):
            ionic_cfg = config.simulation.ionic
        else:
            ionic_cfg = config

        self._urls: dict[str, str] = {
            "endo": ionic_cfg.cellml_url_endo,
            "mid": ionic_cfg.cellml_url_mid,
            "epi": ionic_cfg.cellml_url_epi,
        }

        cache_dir_str: str = getattr(ionic_cfg, "cache_dir", "data/cellml")
        self._cache_dir = Path(cache_dir_str)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory model cache: cell_type -> myokit.Model
        self._model_cache: dict[str, myokit.Model] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _download_if_needed(self, url: str, cell_type: str) -> Path:
        """Download a CellML file if it is not already cached locally.

        Parameters
        ----------
        url : str
            Remote URL of the CellML file.
        cell_type : str
            One of ``'endo'``, ``'mid'``, ``'epi'``.

        Returns
        -------
        Path
            Local filesystem path to the cached CellML file.
        """
        filename = f"tt2006_{cell_type}.cellml"
        local_path = self._cache_dir / filename

        if local_path.exists():
            logger.debug("CellML file already cached: %s", local_path)
            return local_path

        logger.info("Downloading CellML for '%s' from %s ...", cell_type, url)
        try:
            urllib.request.urlretrieve(url, str(local_path))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download CellML file for cell_type='{cell_type}' "
                f"from {url}: {exc}"
            ) from exc

        logger.info("Saved CellML file to %s", local_path)
        return local_path

    @staticmethod
    def _validate_model(model: myokit.Model, cell_type: str) -> None:
        """Validate that a loaded Myokit model contains expected variables.

        Checks for the membrane voltage variable ``V`` and a set of
        required ionic currents.

        Raises
        ------
        ValueError
            If any required variables are missing.
        """
        # Collect all variable names across all components.
        all_var_names: set[str] = set()
        for var in model.variables(deep=True):
            all_var_names.add(var.name())
            all_var_names.add(var.qname())

        missing: list[str] = []

        # Check membrane voltage -----------------------------------------
        # NEW
        has_v = any(
            v.name() in ("V", "v") or v.qname().endswith(".V") or v.qname().endswith(".v")
            for v in model.variables(deep=True)
        )
        if not has_v:
            missing.append("membrane.V (membrane voltage)")

        # Check required currents ----------------------------------------
        for current_name in REQUIRED_CURRENTS:
            has_current = any(
                v.name() == current_name or v.qname().endswith(f".{current_name}")
                for v in model.variables(deep=True)
            )
            if not has_current:
                missing.append(current_name)

        if missing:
            raise ValueError(
                f"CellML model for cell_type='{cell_type}' is missing "
                f"required variables: {', '.join(missing)}. "
                f"Available top-level variable names (sample): "
                f"{sorted(list(all_var_names))[:30]}"
            )

        logger.debug(
            "Model validation passed for cell_type='%s'.", cell_type
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_model(self, cell_type: str) -> myokit.Model:
        """Load (or retrieve from cache) a TT2006 Myokit model.

        Parameters
        ----------
        cell_type : str
            One of ``'endo'``, ``'mid'``, ``'epi'``.

        Returns
        -------
        myokit.Model
            The loaded and validated Myokit model instance.

        Raises
        ------
        ValueError
            If *cell_type* is invalid or the model fails validation.
        """
        if cell_type not in VALID_CELL_TYPES:
            raise ValueError(
                f"Invalid cell_type='{cell_type}'. "
                f"Must be one of {VALID_CELL_TYPES}."
            )

        # Return from in-memory cache if available.
        if cell_type in self._model_cache:
            logger.debug("Returning cached model for cell_type='%s'.", cell_type)
            return self._model_cache[cell_type]

        url = self._urls[cell_type]
        local_path = self._download_if_needed(url, cell_type)

        logger.info(
            "Loading CellML model for cell_type='%s' from %s ...",
            cell_type,
            local_path,
        )

        importer = myokit.formats.cellml.CellMLImporter()
        model: myokit.Model = importer.model(str(local_path))

        self._validate_model(model, cell_type)

        self._model_cache[cell_type] = model
        logger.info("Model for cell_type='%s' loaded and validated.", cell_type)
        return model
