# Physics Audit Report

Auditor: Sub-agent A (Physics Audit)
Date: 2026-03-30

## File-by-file audit results

---

### 1. bpc_fno/simulation/forward/biot_savart.py

**REFERENCE CHECKS:**

| Check | Status |
|-------|--------|
| r_script = r_s - r_v (sensor minus voxel) | PASS (line 106-109) |
| dist = max(\|r_script\|, voxel_size_m / 2) | PASS (lines 112-114) |
| prefactor = (mu_0 / 4*pi) * dV / dist^3 | PASS (line 120) |
| mu_0 = 4*pi*1e-7 exactly | PASS (line 25) |
| Cross product signs (all 6 entries) | PASS (lines 141-155) |
| Positions: cm -> m (multiply 1e-2) | PASS (line 109) |
| J_i: uA/cm^2 -> A/m^2 (multiply 1e-2) | PASS (line 197) |
| Output B in Tesla | PASS |

**NO ISSUES FOUND**

---

### 2. bpc_fno/simulation/tissue/monodomain.py

**REFERENCE CHECKS:**

| Check | Status |
|-------|--------|
| J_i = -sigma_i * grad(V_m) (negative sign) | PASS (line 544) |
| grad uses central differences: (V[i+1]-V[i-1])/(2*h), h in cm | PASS (line 540, np.gradient with spacing h) |
| chi (beta) = 0.14 cm^-1 | PASS (line 71) |
| C_m = 1.0 uF/cm^2 | PASS (line 72) |
| Operator splitting: A = L / (beta * C_m) | PASS (line 158) |
| Reaction step: V += dt/C_m * (-I_ion + I_stim) | PASS (line 436) |

**NO ISSUES FOUND**

---

### 3. bpc_fno/simulation/tissue/conductivity.py

**REFERENCE CHECKS:**

| Check | Status |
|-------|--------|
| sigma_i = sigma_it*I + (sigma_il - sigma_it)*f*f^T | PASS (line 93) |
| sigma_il = 3.0e-3 S/cm | PASS (line 47) |
| sigma_it = 3.0e-4 S/cm | PASS (line 48) |

**NO ISSUES FOUND**

---

### 4. bpc_fno/simulation/tissue/geometry.py

**REFERENCE CHECKS:**

| Check | Status |
|-------|--------|
| Streeter rule: -60 deg (endo) to +60 deg (epi) | PASS (lines 43-45) |
| Cell type map: endo/mid/epi based on transmural position | PASS (lines 128-148) |

**NO ISSUES FOUND**

---

### 5. bpc_fno/simulation/ionic/tt2006_runner.py

**REFERENCE CHECKS:**

| Check | Status |
|-------|--------|
| G_Ks = 0.098 for mid, 0.392 for epi/endo | DELEGATED to CellML files (correct by design -- each cell type variant has its own CellML) |
| G_to = 0.073 for endo/mid, 0.294 for epi | DELEGATED to CellML files (correct by design) |
| Conductance scaling: sim.set_constant(var, base_val * scale) | PASS (line 234) |
| Pre-pacing >= 3 beats | PASS (default n_prepacing_beats = 10, line 108) |
| Ko set correctly | SEE ISSUE BELOW |

FILE: pipeline.py
LINE: 277-283, 370, 415
ISSUE: Ko (extracellular potassium concentration) is sampled per-sample from ko_range_mM (line 415) and stored as an HDF5 attribute (line 370), but is NEVER passed to the TT2006 runner. The ionic model simulation always uses the CellML default Ko value (5.4 mM), regardless of the sampled ko_mM value. This means the stated Ko variation in the dataset metadata does not match the actual simulation.
EXPECTED: ko_mM should be passed to run_single() and applied via sim.set_constant() before simulation, so that the ionic model actually uses the varied Ko value.
CURRENT: ko_mM is generated and stored as metadata but never applied to the Myokit simulation.

**1 ISSUE FOUND** (see pipeline.py fix below)

---

### 6. bpc_fno/simulation/ionic/cellml_loader.py

**NO ISSUES FOUND**

Model loading, caching, and validation logic is correct. The CellML importer properly loads cell-type-specific variants which embed the correct G_Ks and G_to values per the TT2006 paper.

---

### 7. bpc_fno/simulation/pipeline.py

FILE: pipeline.py
LINE: 261-263
ISSUE: The conductance_scales mapping only maps short names (I_Na, I_CaL, I_Kr, I_Ks) to Myokit qualified names. There is no mapping for Ko (extracellular potassium), and ko_mM is never included in the params dict passed to run_single().
EXPECTED: ko_mM should be included as an absolute value (not a scale factor) passed to run_single() and applied to the Myokit model constant for extracellular potassium.
CURRENT: ko_mM is only stored as HDF5 attribute metadata and never used in the actual simulation.

**1 ISSUE FOUND** (fix applied -- see below)

---

### 8. bpc_fno/simulation/forward/sensor_config.py

**NO ISSUES FOUND**

Sensor positioning logic is correct. Virtual grid is centred over the tissue slab. KCD 4-sensor positions match hardware layout.

---

## Summary of fixes applied

### Fix 1: Apply Ko to TT2006 ionic model simulations (pipeline.py)

Added Ko (extracellular potassium) as an absolute-value parameter passed through to the TT2006Runner. The runner already supports arbitrary Myokit variable names via `_resolve_param_name()`, so we pass the Ko qualified name with its absolute value (not a scale factor). Modified `run_single()` to accept an `absolute_params` dict for parameters that should be set directly (not scaled).

Files modified:
- `bpc_fno/simulation/pipeline.py` -- pass ko_mM to runner
- `bpc_fno/simulation/ionic/tt2006_runner.py` -- accept absolute_params in run_single()

---

PHYSICS AUDIT COMPLETE
