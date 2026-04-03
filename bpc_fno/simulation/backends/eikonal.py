"""Legacy eikonal stamping backend kept for backward-compatible smoke generation."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

from bpc_fno.simulation.backends.base import (
    SimulationBackend,
    SimulationContext,
    SimulationResult,
)
from bpc_fno.simulation.monodomain.stimulus import SphericalStimulus


def compute_eikonal_activation_times(
    fiber: np.ndarray,
    pacing_site_voxel: tuple[int, int, int],
    voxel_size_cm: float,
    sigma_il: float,
    sigma_it: float,
    cv_scale: float,
) -> np.ndarray:
    """Approximate anisotropic activation delays from fibre geometry."""
    nx, ny, nz, _ = fiber.shape
    h = float(voxel_size_cm)

    coords = np.mgrid[0:nx, 0:ny, 0:nz].astype(np.float64)
    pace = np.array(pacing_site_voxel, dtype=np.float64)
    disp = np.stack(
        [
            coords[0] - pace[0],
            coords[1] - pace[1],
            coords[2] - pace[2],
        ],
        axis=-1,
    ) * h

    cv_base = 0.06 * float(cv_scale)
    sigma_ratio = float(sigma_il) / max(float(sigma_it), 1e-12)
    cv_ratio = np.sqrt(sigma_ratio)

    fiber_proj = np.sum(disp * fiber, axis=-1)
    cross_proj = np.sqrt(
        np.maximum(np.sum(disp**2, axis=-1) - fiber_proj**2, 0.0)
    )
    eff_dist = np.sqrt(fiber_proj**2 + (cross_proj * cv_ratio) ** 2)
    return eff_dist / max(cv_base, 1e-8)


class EikonalBackend(SimulationBackend):
    """Legacy AP-stamping backend."""

    name = "eikonal"
    requires_lookup_ionics = True
    coupling_description = "eikonal_ap_stamping"

    def simulate(
        self,
        context: SimulationContext,
        ap_waveforms: dict[int, dict[str, np.ndarray]] | None = None,
    ) -> SimulationResult:
        if ap_waveforms is None:
            raise ValueError("EikonalBackend requires precomputed AP waveforms.")

        params = context.params
        activation_times_ms = compute_eikonal_activation_times(
            fiber=context.fiber,
            pacing_site_voxel=context.pacing_site_voxel,
            voxel_size_cm=context.slab.voxel_size_cm,
            sigma_il=float(context.config.tissue.sigma_il),
            sigma_it=float(context.config.tissue.sigma_it),
            cv_scale=float(params.get("cv_scale", 1.0)),
        )

        mono_cfg = context.config.monodomain
        stimulus = SphericalStimulus(
            grid_shape=context.slab.grid_shape,
            voxel_size_cm=context.slab.voxel_size_cm,
            pacing_site_voxel=context.pacing_site_voxel,
            magnitude_uA_cm2=float(mono_cfg.stimulus_magnitude_uA_cm2),
            duration_ms=float(mono_cfg.stimulus_duration_ms),
            start_ms=float(mono_cfg.stimulus_start_ms),
            radius_cm=float(mono_cfg.stimulus_radius_cm),
        )

        D_tensor = context.conductivity.get_tensor_field()
        rest_voltage = float(mono_cfg.resting_potential_mV)

        vm_interps: dict[int, interp1d] = {}
        for cell_type_idx, result in ap_waveforms.items():
            vm_interps[cell_type_idx] = interp1d(
                np.asarray(result["t_ms"], dtype=np.float64),
                np.asarray(result["V_m"], dtype=np.float64),
                kind="linear",
                bounds_error=False,
                fill_value=(
                    float(result["V_m"][0]),
                    float(result["V_m"][-1]),
                ),
            )

        V_snapshots: list[np.ndarray] = []
        J_snapshots: list[np.ndarray] = []
        for t_ms in context.output_times_ms:
            V_m = np.full(context.slab.grid_shape, rest_voltage, dtype=np.float64)
            for cell_type_idx, interp_fn in vm_interps.items():
                mask = context.cell_type_map == cell_type_idx
                if np.any(mask):
                    V_m[mask] = interp_fn(t_ms - activation_times_ms[mask])
            edge_order = 2 if min(context.slab.grid_shape) >= 3 else 1
            grad_V = np.stack(
                np.gradient(V_m, context.slab.voxel_size_cm, edge_order=edge_order),
                axis=-1,
            )
            J_i = -np.einsum("...ab,...b->...a", D_tensor, grad_V)
            V_snapshots.append(V_m.astype(np.float32))
            J_snapshots.append(J_i.astype(np.float32))

        metadata = {
            "monodomain_exactness": "none",
            "simulation_method": "eikonal_ap_stamping",
            "smoke_only": False,
        }
        return SimulationResult(
            J_i=np.stack(J_snapshots, axis=0),
            V_m=np.stack(V_snapshots, axis=0) if context.save_vm else None,
            t_ms=context.output_times_ms.astype(np.float32),
            activation_times_ms=activation_times_ms.astype(np.float32),
            stimulus_mask=stimulus.mask_3d.astype(np.uint8),
            metadata=metadata,
        )
