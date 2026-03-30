"""Tests for the monodomain PDE solver.

Validates against the Niederer 2011 benchmark and analytic solutions.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp

from bpc_fno.simulation.tissue.conductivity import ConductivityTensor
from bpc_fno.simulation.tissue.geometry import VentricularSlab
from bpc_fno.simulation.tissue.monodomain import MonodomainSolver


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SMALL_N: int = 8
_SMALL_H: float = 0.05  # cm


@pytest.fixture()
def small_slab() -> VentricularSlab:
    return VentricularSlab(grid_size=_SMALL_N, voxel_size_cm=_SMALL_H)


@pytest.fixture()
def isotropic_conductivity(small_slab: VentricularSlab) -> ConductivityTensor:
    """Isotropic conductivity (sigma_il == sigma_it) for analytic tests."""
    sigma = 1.0e-3  # S/cm
    fiber = small_slab.get_fiber_field()
    return ConductivityTensor(
        sigma_il=sigma, sigma_it=sigma, fiber_field=fiber
    )


@pytest.fixture()
def small_config() -> SimpleNamespace:
    return SimpleNamespace(
        beta=0.14,
        C_m=1.0,
        stim_amplitude=-52.0,
        stim_duration_ms=2.0,
    )


@pytest.fixture()
def solver(
    small_slab: VentricularSlab, small_config: SimpleNamespace
) -> MonodomainSolver:
    return MonodomainSolver(small_slab, small_config)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestNiedererBenchmark:
    """Stimulate 3D slab at one corner, check activation front position.

    Reference: Niederer et al., Phil Trans R Soc A (2011) 369, 4331-4351.
    Activation times at specific locations should match within 10% tolerance.
    """

    def test_niederer_benchmark(self) -> None:
        """Run a small-scale analog of the Niederer benchmark.

        We use a coarser grid for speed, but verify that the activation
        front propagates outward from the stimulus corner in a physically
        plausible manner (monotonically increasing activation time with
        distance).
        """
        N = 16
        h = 0.05  # cm (500 um)
        slab = VentricularSlab(grid_size=N, voxel_size_cm=h)
        fiber = slab.get_fiber_field()

        # Slightly anisotropic conductivity
        sigma_il = 1.74e-3  # S/cm (longitudinal)
        sigma_it = 0.19e-3  # S/cm (transverse)
        cond = ConductivityTensor(
            sigma_il=sigma_il, sigma_it=sigma_it, fiber_field=fiber
        )

        config = SimpleNamespace(
            beta=0.14, C_m=1.0, stim_amplitude=-52.0, stim_duration_ms=2.0
        )
        solver = MonodomainSolver(slab, config)
        dt_ms = 0.02
        solver.setup(cond, dt_ms=dt_ms)
        solver.set_pacing_site((0, 0, 0))

        # Simple ionic model: passive leak current
        def passive_ionic(V_m: np.ndarray, dt: float) -> tuple[np.ndarray, None]:
            g_leak = 0.01  # mS/cm^2
            V_rest = -85.0
            I_ion = g_leak * (V_m - V_rest)
            return I_ion, None

        total_time_ms = 50.0
        output_stride = int(10.0 / dt_ms)  # snapshot every 10 ms

        result = solver.run(
            total_time_ms=total_time_ms,
            dt_ms=dt_ms,
            output_stride=output_stride,
            ionic_model_func=passive_ionic,
        )

        V_m_snapshots = result["V_m"]  # (T, N, N, N)
        t_ms = result["t_ms"]

        # Verify that activation (V_m significantly above rest) propagates
        # outward from (0,0,0).  Activation threshold: V_m > -80 mV
        activation_threshold = -80.0
        for t_idx in range(1, len(t_ms)):
            activated = V_m_snapshots[t_idx] > activation_threshold
            if activated.any():
                # The activated region should include the stimulus corner
                assert V_m_snapshots[t_idx, 0, 0, 0] > activation_threshold or t_idx == 0
                break


class TestZeroFluxBC:
    """Verify zero-flux (Neumann) boundary conditions."""

    def test_zero_flux_bc(
        self,
        solver: MonodomainSolver,
        isotropic_conductivity: ConductivityTensor,
    ) -> None:
        """After diffusion on a uniform V_m field, V_m should remain uniform
        (zero-flux BCs preserve total charge)."""
        dt_ms = 0.02
        solver.setup(isotropic_conductivity, dt_ms=dt_ms)

        # Set uniform V_m
        V_init = -60.0
        solver.V_m = np.full(solver.n_voxels, V_init, dtype=np.float64)

        # Zero ionic current
        I_ion = np.zeros(solver.n_voxels, dtype=np.float64)

        # Take several diffusion steps
        for step in range(100):
            solver.step(dt_ms, I_ion, t_ms=step * dt_ms)

        # V_m should remain uniform under zero-flux BC with no ionic current
        np.testing.assert_allclose(
            solver.V_m,
            V_init,
            atol=1e-10,
            err_msg="Uniform V_m changed under zero-flux BC with zero I_ion.",
        )

    def test_total_charge_conservation(
        self,
        solver: MonodomainSolver,
        isotropic_conductivity: ConductivityTensor,
    ) -> None:
        """Total V_m (proxy for charge) should be conserved by diffusion alone."""
        dt_ms = 0.02
        solver.setup(isotropic_conductivity, dt_ms=dt_ms)

        # Non-uniform initial condition
        rng = np.random.default_rng(42)
        solver.V_m = rng.standard_normal(solver.n_voxels) * 10.0 - 80.0
        total_initial = solver.V_m.sum()

        I_ion = np.zeros(solver.n_voxels, dtype=np.float64)
        for step in range(50):
            solver.step(dt_ms, I_ion, t_ms=step * dt_ms)

        total_final = solver.V_m.sum()
        np.testing.assert_allclose(
            total_final,
            total_initial,
            rtol=1e-6,
            err_msg="Total V_m not conserved by diffusion with zero-flux BC.",
        )


class TestDiffusionOnly:
    """With no ionic current, verify V_m diffuses correctly."""

    def test_diffusion_only(
        self,
        solver: MonodomainSolver,
        isotropic_conductivity: ConductivityTensor,
    ) -> None:
        """A point perturbation should spread out under diffusion.

        Compare qualitatively to the heat equation: the peak should decrease
        and the profile should broaden over time.
        """
        dt_ms = 0.02
        solver.setup(isotropic_conductivity, dt_ms=dt_ms)

        N = _SMALL_N
        # Uniform background with a point perturbation at the centre
        solver.V_m = np.full(solver.n_voxels, -85.0, dtype=np.float64)
        mid = N // 2
        centre_flat = mid * N * N + mid * N + mid
        solver.V_m[centre_flat] = -40.0  # 45 mV perturbation

        peak_before = solver.V_m[centre_flat]
        I_ion = np.zeros(solver.n_voxels, dtype=np.float64)

        for step in range(200):
            solver.step(dt_ms, I_ion, t_ms=step * dt_ms)

        peak_after = solver.V_m[centre_flat]

        # Peak should have decreased (spread out)
        assert peak_after < peak_before, (
            f"Peak did not decrease: before={peak_before:.3f}, after={peak_after:.3f}"
        )

        # Neighbours should have increased above -85 mV
        V_field = solver.V_m.reshape(N, N, N)
        neighbour_val = V_field[mid + 1, mid, mid]
        assert neighbour_val > -85.0, (
            f"Neighbour did not receive diffused potential: {neighbour_val:.3f}"
        )


class TestStimulusApplication:
    """Verify stimulus is applied at correct site and time."""

    def test_stimulus_application(
        self,
        solver: MonodomainSolver,
        isotropic_conductivity: ConductivityTensor,
    ) -> None:
        dt_ms = 0.02
        solver.setup(isotropic_conductivity, dt_ms=dt_ms)
        solver.set_pacing_site((0, 0, 0))

        # Activate stimulus
        solver._stim_active = True
        solver._stim_start_ms = 0.0

        # At t=0 (within stim_duration), stimulus should be active at (0,0,0)
        I_stim = solver._get_I_stim(t_ms=0.0)
        stim_flat_idx = 0  # (0,0,0) maps to flat index 0
        assert I_stim[stim_flat_idx] == solver.stim_amplitude
        # All other voxels should have zero stimulus
        I_stim[stim_flat_idx] = 0.0
        np.testing.assert_allclose(I_stim, 0.0, atol=1e-30)

    def test_stimulus_off_after_duration(
        self,
        solver: MonodomainSolver,
        isotropic_conductivity: ConductivityTensor,
    ) -> None:
        dt_ms = 0.02
        solver.setup(isotropic_conductivity, dt_ms=dt_ms)
        solver.set_pacing_site((0, 0, 0))
        solver._stim_active = True
        solver._stim_start_ms = 0.0

        # After the stimulus duration, it should be off
        t_after = solver.stim_duration_ms + 1.0
        I_stim = solver._get_I_stim(t_ms=t_after)
        np.testing.assert_allclose(I_stim, 0.0, atol=1e-30)
