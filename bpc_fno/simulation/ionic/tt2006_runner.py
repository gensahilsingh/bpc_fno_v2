"""Batch ODE integration runner for TT2006 ionic models.

Provides single-cell and batch simulation with Myokit,
including parameter modification, pre-pacing, and parallel execution.
"""

from __future__ import annotations

# Fix for WinError 206: os.add_dll_directory fails on long paths.
# Redirect Myokit compilation temp dir to a short path BEFORE any myokit import.
import os
import tempfile

_MYOKIT_TMP = os.path.join("C:\\", "tmp", "mk")
os.makedirs(_MYOKIT_TMP, exist_ok=True)
os.environ["TEMP"] = _MYOKIT_TMP
os.environ["TMP"] = _MYOKIT_TMP
os.environ["TMPDIR"] = _MYOKIT_TMP
tempfile.tempdir = _MYOKIT_TMP

import logging
import multiprocessing
from functools import partial
from typing import Any

import numpy as np
from omegaconf import DictConfig

from bpc_fno.simulation.ionic.cellml_loader import CellMLLoader

logger = logging.getLogger(__name__)

# Default stimulus parameters (rectangular pulse).
_STIM_DURATION_MS: float = 2.0
_STIM_LEVEL: float = -52.0  # uA/cm^2


def _find_variable_by_suffix(
    model: Any, suffix: str
) -> str | None:
    """Search model variables for one whose qualified name ends with *suffix*.

    Returns the qualified name if found, else ``None``.
    """
    for var in model.variables(deep=True):
        qname: str = var.qname()
        if qname == suffix or qname.endswith(f".{suffix}"):
            return qname
    return None


def _worker_run_single(
    args: tuple[dict[str, Any], int],
    config: DictConfig,
) -> dict[str, np.ndarray] | None:
    """Worker function executed in a child process.

    Each worker creates its own :class:`CellMLLoader` because Myokit
    model objects are not picklable.

    Parameters
    ----------
    args : tuple
        ``(params_dict, index)`` where *params_dict* contains all the
        information needed for a single simulation run.
    config : DictConfig
        Full configuration (must be picklable).

    Returns
    -------
    dict or None
        Simulation result dictionary, or ``None`` on failure.
    """
    params_dict, idx = args
    try:
        loader = CellMLLoader(config)
        runner = TT2006Runner(loader, config)
        result = runner.run_single(
            cell_type=params_dict["cell_type"],
            params=params_dict.get("conductance_scales", {}),
            pacing_cl_ms=params_dict["pacing_cl_ms"],
            n_beats=params_dict.get("n_beats", 1),
        )
        return result
    except Exception:
        logger.exception("Worker %d failed for params: %s", idx, params_dict)
        return None


class TT2006Runner:
    """Run single or batch TT2006 ionic-model simulations via Myokit.

    Parameters
    ----------
    cellml_loader : CellMLLoader
        Loader instance providing Myokit models.
    config : DictConfig
        Hydra/OmegaConf configuration.  Expected key:
        ``config.simulation.ionic.n_prepacing_beats`` (default 10).
    """

    def __init__(self, cellml_loader: CellMLLoader, config: DictConfig) -> None:
        self._loader = cellml_loader
        self._config = config

        ionic_cfg = config.simulation.ionic
        self._n_prepacing_beats: int = int(
            getattr(ionic_cfg, "n_prepacing_beats", 10)
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_param_name(model: Any, key: str) -> str:
        """Resolve a user-supplied parameter key to a model-qualified name.

        The key may already be a fully-qualified name (``component.var``)
        or a short name.  We search the model to find the best match.

        Raises
        ------
        KeyError
            If the parameter cannot be resolved to any model variable.
        """
        # Direct match first.
        for var in model.variables(deep=True):
            if var.qname() == key:
                return key

        # Suffix match.
        for var in model.variables(deep=True):
            if var.name() == key or var.qname().endswith(f".{key}"):
                return var.qname()

        raise KeyError(
            f"Parameter '{key}' could not be resolved to any variable in "
            f"the model.  Available variables (sample): "
            f"{[v.qname() for v in list(model.variables(deep=True))[:30]]}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_single(
        self,
        cell_type: str,
        params: dict[str, float],
        pacing_cl_ms: float,
        n_beats: int = 1,
    ) -> dict[str, np.ndarray]:
        """Run a single TT2006 simulation.

        Parameters
        ----------
        cell_type : str
            One of ``'endo'``, ``'mid'``, ``'epi'``.
        params : dict[str, float]
            Conductance scale factors.  Keys are Myokit variable names
            (qualified or short); values are multiplicative scale factors
            applied to the model's default constant value.
        pacing_cl_ms : float
            Pacing cycle length in milliseconds.
        n_beats : int
            Number of recorded beats (the last *n_beats* are returned).
            Typically 1.

        Returns
        -------
        dict[str, np.ndarray]
            Keys: ``'V_m'``, ``'I_Na'``, ``'I_CaL'``, ``'I_Kr'``,
            ``'I_Ks'``, ``'I_ion_total'``, ``'t_ms'``.

        Raises
        ------
        RuntimeError
            If the Myokit simulation encounters an error.
        """
        import myokit  # local import so workers can load independently

        model = self._loader.get_model(cell_type)

        # Clone the model so parameter changes don't mutate the cached copy.
        model = model.clone()

        # ---- Apply parameter modifications (scale factors) ----
        for key, scale_factor in params.items():
            qname = self._resolve_param_name(model, key)
            var = model.get(qname)
            default_value = var.value()
            if not isinstance(default_value, myokit.Number):
                # The variable might be an expression; try evaluating.
                try:
                    default_value = float(var.eval())
                except Exception:
                    logger.warning(
                        "Cannot evaluate default for '%s'; skipping.", qname
                    )
                    continue
            else:
                default_value = float(default_value)
            var.set_rhs(default_value * scale_factor)
            logger.debug(
                "Set %s = %.6g (default %.6g x %.4f)",
                qname,
                default_value * scale_factor,
                default_value,
                scale_factor,
            )

        # ---- Bind stimulus to Myokit pace protocol ----
        # CellML models have an internal piecewise i_Stim that doesn't
        # respond to Myokit's protocol.  We override it by binding i_Stim
        # to 'pace' so the protocol controls stimulus timing directly.
        stim_name = _find_variable_by_suffix(model, "i_Stim")
        if stim_name is not None:
            stim_var = model.get(stim_name)
            stim_var.set_binding("pace")
            # RHS is replaced by protocol value when bound to pace

        protocol = myokit.pacing.blocktrain(
            period=pacing_cl_ms,
            duration=_STIM_DURATION_MS,
            offset=10,  # small delay before first stimulus
            level=_STIM_LEVEL,
        )

        # ---- Create simulation (with retry for WinError 206) ----
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                sim = myokit.Simulation(model, protocol)
                break
            except (FileNotFoundError, OSError) as e:
                if "WinError 206" in str(e) or "too long" in str(e).lower():
                    last_err = e
                    import time as _time
                    _time.sleep(0.5 * (attempt + 1))
                    logger.warning(
                        "WinError 206 on attempt %d for %s, retrying...",
                        attempt + 1, cell_type,
                    )
                    continue
                raise
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to create Myokit simulation for "
                    f"cell_type='{cell_type}': {exc}"
                ) from exc
        else:
            raise RuntimeError(
                f"Failed to create Myokit simulation for "
                f"cell_type={cell_type!r} after 3 attempts: {last_err}"
            )

        # ---- Pre-pace ----
        total_prepace_ms = pacing_cl_ms * self._n_prepacing_beats
        try:
            sim.pre(total_prepace_ms)
        except myokit.SimulationError as exc:
            raise RuntimeError(
                f"Pre-pacing failed after {self._n_prepacing_beats} beats "
                f"(CL={pacing_cl_ms} ms) for cell_type='{cell_type}': {exc}"
            ) from exc

        # ---- Resolve variable names by inspecting the model ----
        # Look up actual qualified names — do NOT hardcode them.
        v_name = _find_variable_by_suffix(model, "V")
        time_name = _find_variable_by_suffix(model, "time")
        i_na_name = _find_variable_by_suffix(model, "i_Na")
        i_cal_name = _find_variable_by_suffix(model, "i_CaL")
        i_kr_name = _find_variable_by_suffix(model, "i_Kr")
        i_ks_name = _find_variable_by_suffix(model, "i_Ks")

        if v_name is None:
            raise RuntimeError(
                "Could not find membrane voltage variable 'V' in model."
            )

        # Build log variable list from actual model qnames only.
        log_vars: list[str] = []
        if time_name is not None:
            log_vars.append(time_name)
        log_vars.append(v_name)
        for name in (i_na_name, i_cal_name, i_kr_name, i_ks_name):
            if name is not None:
                log_vars.append(name)

        # ---- Run recorded beats ----
        record_duration_ms = pacing_cl_ms * n_beats
        try:
            log = sim.run(record_duration_ms, log=log_vars, log_interval=0.1)
        except myokit.SimulationError as exc:
            raise RuntimeError(
                f"Simulation failed during recording phase "
                f"(CL={pacing_cl_ms} ms, n_beats={n_beats}) for "
                f"cell_type='{cell_type}': {exc}"
            ) from exc

        # ---- Extract arrays ----
        # Time: use the model's time variable if logged, else construct
        if time_name is not None and time_name in log:
            t_raw = np.asarray(log[time_name])
            t_ms = t_raw - t_raw[0]
        else:
            # Fallback: construct from log length and duration
            n_log_points = len(log[v_name])
            t_ms = np.linspace(0, record_duration_ms, n_log_points)

        v_m = np.asarray(log[v_name])

        def _safe_extract(name: str | None) -> np.ndarray:
            if name is not None and name in log:
                return np.asarray(log[name])
            return np.zeros_like(t_ms)

        i_na = _safe_extract(i_na_name)
        i_cal = _safe_extract(i_cal_name)
        i_kr = _safe_extract(i_kr_name)
        i_ks = _safe_extract(i_ks_name)

        # Total ionic current: sum of all available recorded currents.
        i_ion_total = i_na + i_cal + i_kr + i_ks

        return {
            "V_m": v_m,
            "I_Na": i_na,
            "I_CaL": i_cal,
            "I_Kr": i_kr,
            "I_Ks": i_ks,
            "I_ion_total": i_ion_total,
            "t_ms": t_ms,
        }

    def run_batch(
        self,
        params_list: list[dict[str, Any]],
        n_workers: int = 4,
    ) -> list[dict[str, np.ndarray] | None]:
        """Run multiple simulations in parallel.

        Parameters
        ----------
        params_list : list[dict]
            Each dict must contain at least ``'cell_type'``,
            ``'pacing_cl_ms'``, and optionally ``'conductance_scales'``
            and ``'n_beats'``.
        n_workers : int
            Number of parallel worker processes.

        Returns
        -------
        list[dict | None]
            Ordered list of result dicts; entries are ``None`` for any
            simulation that failed.
        """
        indexed_args = [(p, i) for i, p in enumerate(params_list)]
        worker_fn = partial(_worker_run_single, config=self._config)

        logger.info(
            "Starting batch of %d simulations with %d workers.",
            len(params_list),
            n_workers,
        )

        # Use 'spawn' context to avoid fork-related issues on all platforms.
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            results: list[dict[str, np.ndarray] | None] = pool.map(
                worker_fn, indexed_args
            )

        n_success = sum(1 for r in results if r is not None)
        logger.info(
            "Batch complete: %d/%d succeeded.", n_success, len(params_list)
        )
        return results
