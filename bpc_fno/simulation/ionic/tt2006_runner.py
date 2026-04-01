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

        # Persistent simulation cache: one compiled Simulation per cell type.
        # Reused across run_single() calls via sim.reset() + set_constant().
        # This avoids Myokit recompilation (and WinError 206 on Windows).
        self._sim_cache: dict[str, Any] = {}  # cell_type -> myokit.Simulation
        self._default_constants: dict[str, dict[str, float]] = {}  # cell_type -> {qname: default_value}

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

    def _get_or_create_sim(self, cell_type: str) -> Any:
        """Get a persistent Simulation for this cell type, creating once.

        The Simulation is compiled once and then reused via reset() +
        set_constant() for each new parameter set.  This avoids
        recompilation (and WinError 206 on Windows).
        """
        import myokit

        if cell_type in self._sim_cache:
            return self._sim_cache[cell_type]

        model = self._loader.get_model(cell_type)
        model = model.clone()

        # Bind stimulus to pace protocol
        stim_name = _find_variable_by_suffix(model, "i_Stim")
        if stim_name is not None:
            model.get(stim_name).set_binding("pace")

        # Use a standard protocol — CL will be changed via set_constant
        protocol = myokit.pacing.blocktrain(
            period=1000, duration=_STIM_DURATION_MS,
            offset=10, level=_STIM_LEVEL,
        )

        sim = myokit.Simulation(model, protocol)

        # Cache the default constant values for parameter scaling
        defaults: dict[str, float] = {}
        for var in model.variables(deep=True):
            try:
                val = float(var.eval())
                defaults[var.qname()] = val
            except Exception:
                pass

        self._sim_cache[cell_type] = sim
        self._default_constants[cell_type] = defaults
        logger.info("Compiled and cached Simulation for %s", cell_type)
        return sim

    def run_single(
        self,
        cell_type: str,
        params: dict[str, float],
        pacing_cl_ms: float,
        n_beats: int = 1,
        absolute_params: dict[str, float] | None = None,
    ) -> dict[str, np.ndarray]:
        """Run a single TT2006 simulation, reusing compiled Simulation.

        Parameters
        ----------
        cell_type : str
            One of ``'endo'``, ``'mid'``, ``'epi'``.
        params : dict[str, float]
            Conductance scale factors.  Keys are Myokit variable names;
            values are multiplicative scale factors.
        pacing_cl_ms : float
            Pacing cycle length in milliseconds.
        n_beats : int
            Number of recorded beats (last *n_beats* returned).
        absolute_params : dict[str, float] | None
            Absolute-value parameter overrides.  Keys are Myokit variable
            names; values are set directly (not multiplied by the default).
            Use this for parameters like Ko where a physical value (e.g.
            5.4 mM) should be set directly rather than scaled.

        Returns
        -------
        dict[str, np.ndarray]
            Keys: V_m, I_Na, I_CaL, I_Kr, I_Ks, I_ion_total, t_ms.
        """
        import myokit

        sim = self._get_or_create_sim(cell_type)
        defaults = self._default_constants[cell_type]
        model = sim._model  # access the model for variable name resolution

        # ---- Reset simulation to initial state ----
        sim.reset()

        # ---- Apply parameter modifications via set_constant ----
        for key, scale_factor in params.items():
            qname = self._resolve_param_name(model, key)
            if qname in defaults:
                sim.set_constant(qname, defaults[qname] * scale_factor)

        # ---- Apply absolute-value parameter overrides ----
        if absolute_params:
            for key, value in absolute_params.items():
                qname = self._resolve_param_name(model, key)
                sim.set_constant(qname, value)

        # ---- Set pacing CL ----
        protocol = myokit.pacing.blocktrain(
            period=pacing_cl_ms,
            duration=_STIM_DURATION_MS,
            offset=10,
            level=_STIM_LEVEL,
        )
        sim.set_protocol(protocol)

        # ---- Pre-pace ----
        total_prepace_ms = pacing_cl_ms * self._n_prepacing_beats
        try:
            sim.pre(total_prepace_ms)
        except myokit.SimulationError as exc:
            raise RuntimeError(
                f"Pre-pacing failed ({cell_type}, CL={pacing_cl_ms}): {exc}"
            ) from exc

        # ---- Resolve log variable names ----
        v_name = _find_variable_by_suffix(model, "V")
        time_name = _find_variable_by_suffix(model, "time")
        i_na_name = _find_variable_by_suffix(model, "i_Na")
        i_cal_name = _find_variable_by_suffix(model, "i_CaL")
        i_kr_name = _find_variable_by_suffix(model, "i_Kr")
        i_ks_name = _find_variable_by_suffix(model, "i_Ks")

        if v_name is None:
            raise RuntimeError("Could not find 'V' in model.")

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
                f"Simulation failed ({cell_type}, CL={pacing_cl_ms}): {exc}"
            ) from exc

        # ---- Extract arrays ----
        if time_name is not None and time_name in log:
            t_raw = np.asarray(log[time_name])
            t_ms = t_raw - t_raw[0]
        else:
            n_pts = len(log[v_name])
            t_ms = np.linspace(0, record_duration_ms, n_pts)

        v_m = np.asarray(log[v_name])

        def _safe(name: str | None) -> np.ndarray:
            if name and name in log:
                return np.asarray(log[name])
            return np.zeros_like(t_ms)

        i_na = _safe(i_na_name)
        i_cal = _safe(i_cal_name)
        i_kr = _safe(i_kr_name)
        i_ks = _safe(i_ks_name)

        return {
            "V_m": v_m,
            "I_Na": i_na,
            "I_CaL": i_cal,
            "I_Kr": i_kr,
            "I_Ks": i_ks,
            "I_ion_total": i_na + i_cal + i_kr + i_ks,
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
