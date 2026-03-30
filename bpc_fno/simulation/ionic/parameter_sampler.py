"""Parameter sampler for TT2006 ionic model simulations.

Generates reproducible sets of physiological parameter samples
spanning endocardial, midmyocardial, and epicardial cell types
with randomised conductance scaling and pacing rates.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

# Default parameter ranges.
_CELL_TYPES: list[str] = ["endo", "mid", "epi"]
_PACING_CL_RANGE: tuple[float, float] = (600.0, 1200.0)
_CONDUCTANCE_SCALE_RANGE: tuple[float, float] = (0.7, 1.3)
_KO_RANGE: tuple[float, float] = (3.5, 6.5)
_CONDUCTANCE_KEYS: list[str] = ["I_Na", "I_CaL", "I_Kr", "I_Ks"]


class ParameterSampler:
    """Sample physiological parameter sets for TT2006 simulations.

    Parameters
    ----------
    config : DictConfig
        Hydra/OmegaConf configuration.  Optional overrides under
        ``config.simulation.ionic.sampling``:

        - ``pacing_cl_min``, ``pacing_cl_max``
        - ``conductance_scale_min``, ``conductance_scale_max``
        - ``ko_min``, ``ko_max``
        - ``cell_types``  (list of str)
    """

    def __init__(self, config: DictConfig) -> None:
        self._config = config

        # Allow optional overrides from config; fall back to defaults.
        sampling_cfg: Any = getattr(
            getattr(config.simulation, "ionic", None), "sampling", None
        )

        if sampling_cfg is not None:
            self._pacing_cl_min: float = float(
                getattr(sampling_cfg, "pacing_cl_min", _PACING_CL_RANGE[0])
            )
            self._pacing_cl_max: float = float(
                getattr(sampling_cfg, "pacing_cl_max", _PACING_CL_RANGE[1])
            )
            self._cond_min: float = float(
                getattr(
                    sampling_cfg,
                    "conductance_scale_min",
                    _CONDUCTANCE_SCALE_RANGE[0],
                )
            )
            self._cond_max: float = float(
                getattr(
                    sampling_cfg,
                    "conductance_scale_max",
                    _CONDUCTANCE_SCALE_RANGE[1],
                )
            )
            self._ko_min: float = float(
                getattr(sampling_cfg, "ko_min", _KO_RANGE[0])
            )
            self._ko_max: float = float(
                getattr(sampling_cfg, "ko_max", _KO_RANGE[1])
            )
            self._cell_types: list[str] = list(
                getattr(sampling_cfg, "cell_types", _CELL_TYPES)
            )
        else:
            self._pacing_cl_min, self._pacing_cl_max = _PACING_CL_RANGE
            self._cond_min, self._cond_max = _CONDUCTANCE_SCALE_RANGE
            self._ko_min, self._ko_max = _KO_RANGE
            self._cell_types = list(_CELL_TYPES)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def sample_single(self, rng: np.random.Generator, idx: int) -> dict[str, Any]:
        """Generate a single parameter sample.

        Parameters
        ----------
        rng : np.random.Generator
            Numpy random generator for reproducibility.
        idx : int
            Sample index, used to derive a unique per-sample seed.

        Returns
        -------
        dict[str, Any]
            A parameter dictionary with keys ``'cell_type'``,
            ``'pacing_cl_ms'``, ``'conductance_scales'``, ``'ko_mM'``,
            and ``'seed'``.
        """
        cell_type: str = rng.choice(self._cell_types)

        pacing_cl_ms: float = float(
            rng.uniform(self._pacing_cl_min, self._pacing_cl_max)
        )

        conductance_scales: dict[str, float] = {
            key: float(rng.uniform(self._cond_min, self._cond_max))
            for key in _CONDUCTANCE_KEYS
        }

        ko_mM: float = float(rng.uniform(self._ko_min, self._ko_max))

        # Unique per-sample seed derived from the RNG state.
        seed: int = int(rng.integers(0, 2**31))

        return {
            "cell_type": cell_type,
            "pacing_cl_ms": pacing_cl_ms,
            "conductance_scales": conductance_scales,
            "ko_mM": ko_mM,
            "seed": seed,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(
        self, n_samples: int, rng_seed: int = 42
    ) -> list[dict[str, Any]]:
        """Generate *n_samples* parameter sets.

        Parameters
        ----------
        n_samples : int
            Number of parameter samples to generate.
        rng_seed : int
            Seed for the underlying ``np.random.Generator`` to ensure
            full reproducibility.

        Returns
        -------
        list[dict[str, Any]]
            List of parameter dictionaries ready for
            :meth:`TT2006Runner.run_single` or batch submission.
        """
        rng = np.random.default_rng(rng_seed)
        samples: list[dict[str, Any]] = [
            self.sample_single(rng, idx) for idx in range(n_samples)
        ]

        # ---- Log distribution statistics ----
        pacing_vals = np.array([s["pacing_cl_ms"] for s in samples])
        ko_vals = np.array([s["ko_mM"] for s in samples])

        logger.info(
            "Sampled %d parameter sets (seed=%d).", n_samples, rng_seed
        )
        logger.info(
            "  pacing_cl_ms  — mean=%.1f  std=%.1f  min=%.1f  max=%.1f",
            pacing_vals.mean(),
            pacing_vals.std(),
            pacing_vals.min(),
            pacing_vals.max(),
        )
        logger.info(
            "  ko_mM         — mean=%.2f  std=%.2f  min=%.2f  max=%.2f",
            ko_vals.mean(),
            ko_vals.std(),
            ko_vals.min(),
            ko_vals.max(),
        )

        for cond_key in _CONDUCTANCE_KEYS:
            vals = np.array(
                [s["conductance_scales"][cond_key] for s in samples]
            )
            logger.info(
                "  %s scale  — mean=%.3f  std=%.3f  min=%.3f  max=%.3f",
                cond_key.ljust(5),
                vals.mean(),
                vals.std(),
                vals.min(),
                vals.max(),
            )

        # Cell-type distribution.
        from collections import Counter

        ct_counts = Counter(s["cell_type"] for s in samples)
        logger.info(
            "  cell_type distribution: %s",
            {k: ct_counts[k] for k in sorted(ct_counts)},
        )

        return samples
