"""Smoke-only Windows monodomain backend with lookup-table ionics."""

from __future__ import annotations

import numpy as np

from bpc_fno.simulation.backends.base import (
    SimulationBackend,
    SimulationContext,
    SimulationResult,
)
from bpc_fno.simulation.backends.eikonal import compute_eikonal_activation_times
from bpc_fno.simulation.monodomain.hybrid_solver import HybridMonodomainSolver
from bpc_fno.simulation.monodomain.lookup_ionic import LookupTableIonicCallback
from bpc_fno.simulation.monodomain.stimulus import SphericalStimulus


class WindowsHybridMonodomainBackend(SimulationBackend):
    """Smoke-only hybrid monodomain backend for local Windows validation."""

    name = "windows_hybrid"
    smoke_only = True
    requires_lookup_ionics = True
    coupling_description = "cn_diffusion_lookup_ionics"

    def simulate(
        self,
        context: SimulationContext,
        ap_waveforms: dict[int, dict[str, np.ndarray]] | None = None,
    ) -> SimulationResult:
        if ap_waveforms is None:
            raise ValueError(
                "WindowsHybridMonodomainBackend requires lookup ionic traces."
            )

        mono_cfg = context.config.monodomain
        activation_times_ms = compute_eikonal_activation_times(
            fiber=context.fiber,
            pacing_site_voxel=context.pacing_site_voxel,
            voxel_size_cm=context.slab.voxel_size_cm,
            sigma_il=float(context.config.tissue.sigma_il),
            sigma_it=float(context.config.tissue.sigma_it),
            cv_scale=float(context.params.get("cv_scale", 1.0)),
        )

        ionic_callback = LookupTableIonicCallback(
            cell_type_map=context.cell_type_map,
            activation_times_ms=activation_times_ms,
            ap_waveforms=ap_waveforms,
        )
        stimulus = SphericalStimulus(
            grid_shape=context.slab.grid_shape,
            voxel_size_cm=context.slab.voxel_size_cm,
            pacing_site_voxel=context.pacing_site_voxel,
            magnitude_uA_cm2=float(mono_cfg.stimulus_magnitude_uA_cm2),
            duration_ms=float(mono_cfg.stimulus_duration_ms),
            start_ms=float(mono_cfg.stimulus_start_ms),
            radius_cm=float(mono_cfg.stimulus_radius_cm),
        )
        solver = HybridMonodomainSolver(
            conductivity_tensor=context.conductivity.get_tensor_field(),
            grid_shape=context.slab.grid_shape,
            voxel_size_cm=context.slab.voxel_size_cm,
            chi_cm_inv=float(mono_cfg.chi_cm_inv),
            Cm_uF_per_cm2=float(mono_cfg.Cm_uF_per_cm2),
            method=str(mono_cfg.solver_method),
        )

        sim = solver.solve(
            V_init=np.full(
                context.slab.grid_shape,
                float(mono_cfg.resting_potential_mV),
                dtype=np.float64,
            ),
            ionic_callback=ionic_callback,
            stimulus_fn=stimulus,
            dt_ms=float(mono_cfg.dt_ms),
            total_time_ms=float(mono_cfg.total_time_ms),
            output_times_ms=context.output_times_ms,
        )

        metadata = {
            "monodomain_exactness": "smoke_only_hybrid",
            "simulation_method": "monodomain_operator_splitting_lookup_ionics",
            "smoke_only": True,
        }
        return SimulationResult(
            J_i=sim["J_i"],
            V_m=sim["V_m"] if context.save_vm else None,
            t_ms=sim["t_ms"],
            activation_times_ms=activation_times_ms.astype(np.float32),
            stimulus_mask=stimulus.mask_3d.astype(np.uint8),
            metadata=metadata,
        )
