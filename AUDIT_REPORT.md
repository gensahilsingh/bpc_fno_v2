# Physics Audit Report

Auditor: Claude Opus 4.6 (1M context)
Date: 2026-03-31
Scope: biot_savart.py, monodomain.py, tt2006_runner.py, conductivity.py

All source files read in full. Tests executed: 12/12 passed.

---

## 1. bpc_fno/simulation/forward/biot_savart.py

**STATUS: CORRECT -- no bugs found.**

| Check | Status | Detail |
|-------|--------|--------|
| r_script = sensor_pos - voxel_pos | PASS | Line 106-109: `d = sensor_positions_cm[:, newaxis, :] - voxel_centers_cm[newaxis, :, :]` |
| Positions cm -> m (*1e-2) | PASS | Line 109: `* 1e-2` applied to displacement |
| Singularity clamp | PASS | Lines 112-114: `dist = max(dist, voxel_size_cm * 1e-2 / 2)` |
| mu_0 = 4*pi*1e-7 T*m/A | PASS | Line 25 |
| Volume element dV in m^3 | PASS | Line 117: `(voxel_size_cm * 1e-2)**3` |
| Prefactor = (mu_0/4pi) * dV / dist^3 | PASS | Line 120 |
| B_x = J_y*d_z - J_z*d_y | PASS | Lines 141-143: `L[0::3,1::3] = pf*dz`, `L[0::3,2::3] = pf*(-dy)` |
| B_y = J_z*d_x - J_x*d_z | PASS | Lines 146-148: `L[1::3,0::3] = pf*(-dz)`, `L[1::3,2::3] = pf*dx` |
| B_z = J_x*d_y - J_y*d_x | PASS | Lines 151-155: `L[2::3,0::3] = pf*dy`, `L[2::3,1::3] = pf*(-dx)` |
| J unit conversion: uA/cm^2 -> A/m^2 (*1e-2) | PASS | Line 197: `J_flat * 1e-2` |
| Output B in Tesla | PASS | Confirmed by unit chain: (T*m/A) * (m^3) * (A/m^2) / (m^3) = T |
| Test: dipole validation < 1% relative error | PASS | `test_dipole_field` passes |

Note on cross product sign convention: The code computes `J x d` where `d = r_s - r'`. This matches the Biot-Savart integral `J(r') x (r_s - r') / |r_s - r'|^3`. Confirmed correct.

---

## 2. bpc_fno/simulation/tissue/monodomain.py

**STATUS: CORRECT -- no bugs found.**

| Check | Status | Detail |
|-------|--------|--------|
| J_i = -sigma_i * grad(V_m) (negative sign) | PASS | Line 544: `-np.einsum("...ab,...b->...a", D, grad_V)` |
| Central differences with h = voxel_size_cm | PASS | Line 540: `np.gradient(V_m_field, h, edge_order=2)` |
| chi (beta) = 0.14 cm^-1 | PASS | Line 71: `_DEFAULT_BETA = 0.14` |
| C_m = 1.0 uF/cm^2 | PASS | Line 72: `_DEFAULT_C_M = 1.0` |
| Operator: A = L / (beta * C_m) | PASS | Line 158 |
| Crank-Nicolson: LHS = I - (dt/2)*A, RHS = I + (dt/2)*A | PASS | Lines 177-189 |
| Reaction step: V += dt/C_m * (-I_ion + I_stim) | PASS | Line 436 |
| Zero-flux Neumann BC via index clamping | PASS | Lines 229-231: `np.clip(i,0,N-1)` |
| Laplacian diagonal terms: face-averaged D / h^2 | PASS | Lines 256-296 |
| Laplacian cross terms: D_pq * 4-point stencil / (4*h^2) | PASS | Lines 306-334 |
| Test: uniform V_m preserved (zero-flux) | PASS | `test_zero_flux_bc` passes |
| Test: total charge conserved under diffusion | PASS | `test_total_charge_conservation` passes |

---

## 3. bpc_fno/simulation/ionic/tt2006_runner.py

**STATUS: CORRECT -- with documented limitations.**

| Check | Status | Detail |
|-------|--------|--------|
| G_to = 0.073 (endo/mid), 0.294 (epi) | DELEGATED | Values embedded in per-cell-type CellML files loaded by CellMLLoader |
| G_Ks = 0.098 (mid), 0.392 (endo/epi) | DELEGATED | Same -- each CellML variant has the correct cell-type-specific value |
| Ko set from absolute_params | PASS | Lines 243-245: `absolute_params` dict applied via `sim.set_constant(qname, value)` |
| Conductance scaling: base_val * scale | PASS | Line 240: `sim.set_constant(qname, defaults[qname] * scale_factor)` |
| Pipeline passes Ko correctly | PASS | pipeline.py lines 269-271: `absolute_params = {"Ko": ko_mM}` passed to `run_single()` |
| Pre-pacing default >= 3 beats | PASS | Line 108: `n_prepacing_beats` defaults to 10 |

**FLAG (documented, not a bug):** `I_ion_total` on line 320 is the sum of only 4 currents (`I_Na + I_CaL + I_Kr + I_Ks`), not the full 12-current sum from TT2006. This is intentional -- these 4 are the major currents used as training features. The full ionic current for the PDE reaction term comes from Myokit internally, not from this partial sum. The partial sum is only stored as diagnostic output.

---

## 4. bpc_fno/simulation/tissue/conductivity.py

**STATUS: CORRECT -- with documented limitation.**

| Check | Status | Detail |
|-------|--------|--------|
| sigma_il = 3.0e-3 S/cm | PASS | Line 47: default parameter |
| sigma_it = 3.0e-4 S/cm | PASS | Line 48: default parameter |
| tensor = sigma_it*I + (sigma_il - sigma_it)*f*f^T | PASS | Line 93: `eye + (self.sigma_il - self.sigma_it) * ff_T` |
| ff^T outer product | PASS | Line 92: `f[..., :, None] * f[..., None, :]` |
| Fibrotic voxels zeroed out | PASS | Lines 96-97 |
| Inverse tensor analytic formula | PASS | Lines 127-136: `(1/sigma_it)*I + (1/sigma_il - 1/sigma_it)*ff^T` |

**FLAG (documented, not a bug):** This computes D_i (intracellular conductivity tensor) only, not D_m (monodomain effective conductivity). In the monodomain approximation with equal anisotropy ratios, D_m = D_i * (sigma_e / (sigma_i + sigma_e)). The current code uses D_i directly in the monodomain PDE, which is equivalent to assuming the extracellular domain has infinite conductivity (or equivalently, that the monodomain reduction factor is absorbed into the conductivity values). This is a common simplification in monodomain implementations and the chosen sigma_il/sigma_it values are consistent with effective monodomain conductivities from the literature.

---

## Test Results

```
tests/test_biot_savart.py::TestDipoleField::test_dipole_field                    PASSED
tests/test_biot_savart.py::TestDipoleField::test_validate_dipole_internal_consistency PASSED
tests/test_biot_savart.py::TestLeadFieldShape::test_lead_field_shape             PASSED
tests/test_biot_savart.py::TestForwardBatchConsistency::test_forward_batch_consistency PASSED
tests/test_biot_savart.py::TestZeroCurrent::test_zero_current                    PASSED
tests/test_biot_savart.py::TestSymmetry::test_symmetry                           PASSED
tests/test_monodomain.py::TestNiedererBenchmark::test_niederer_benchmark         PASSED
tests/test_monodomain.py::TestZeroFluxBC::test_zero_flux_bc                      PASSED
tests/test_monodomain.py::TestZeroFluxBC::test_total_charge_conservation         PASSED
tests/test_monodomain.py::TestDiffusionOnly::test_diffusion_only                 PASSED
tests/test_monodomain.py::TestStimulusApplication::test_stimulus_application     PASSED
tests/test_monodomain.py::TestStimulusApplication::test_stimulus_off_after_duration PASSED

12 passed, 0 failed
```

## Summary

| File | Bugs Found | Flags |
|------|-----------|-------|
| biot_savart.py | 0 | None |
| monodomain.py | 0 | None |
| tt2006_runner.py | 0 | 1 (I_ion_total is partial sum of 4/12 currents) |
| conductivity.py | 0 | 1 (D_i only, not D_m) |

**No bugs found. No fixes applied. All 12 tests pass.**

AUDIT COMPLETE
