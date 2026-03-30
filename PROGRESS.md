# BPC-FNO Progress Log

## Issue 1: KCD Data Format Mismatch (FIXED)

**Problem:** KCDLoader expected WFDB format but local data is in wav format at `kcd_preprocessed/data/preprocessed/patient_N/mcg_200_channel/trial_N/channel_N.wav`.

**Files changed:**
- Created `bpc_fno/data/local_kcd_loader.py` — new LocalKCDLoader for wav files
- Modified `bpc_fno/data/kcd_noise_model.py` — `fit()` now accepts list[dict] or loader object; fixed PSD segment extraction to use fixed nperseg=64; added PSD normalization for stable curve fitting
- Rewrote `scripts/fit_noise_model.py` — uses LocalKCDLoader

**Verification:**
- 175 records loaded (7 patients x 25 trials)
- 2811 PSD segments accumulated
- Per-channel noise parameters fitted with reasonable values (sigma_white ~1e-17 T, sigma_1f ~1e-16 T)
- Saved to `data/processed/noise_model.json`

## Issue 2: Synthetic Data Generation Smoke Test (FIXED)

**Problems found and fixed:**
1. `SensorConfig` expected `config.simulation.sensor` but config has `sensor` at top level — fixed to handle both
2. `ConductivityTensor` was called with wrong API — fixed to use actual `(sigma_il, sigma_it, fiber_field, fibrosis_mask)` constructor
3. h5py couldn't write numpy string for cell_type attribute — fixed with `str()` cast
4. `noise_model.sample()` interface mismatch — fixed to call with `(n_channels, n_timepoints, fs, rng)`
5. Created `_generate_synthetic_activation()` — analytically generates propagating wavefront using eikonal approximation + AP template, then computes J_i = -D_i * grad(V_m) and B via Biot-Savart

**Files changed:**
- Rewrote `bpc_fno/simulation/pipeline.py` — fixed all interface mismatches, added synthetic activation generator
- Modified `bpc_fno/simulation/forward/sensor_config.py` — flexible config path

**Verification:**
- 10/10 samples passed sanity checks
- B_mig range: 7e-12 to 1.6e-11 T (physiologically plausible for MCG)
- J_i range: 0.8 to 1.6 uA/cm^2 (physiologically plausible)
- V_m range: -85 to 30 mV (correct AP range)
- HDF5 schema verified: all expected datasets and attributes present
- ~16 seconds per sample

## Issue 3: Normalization Stats (FIXED)

**Problem:** `Normalizer._EPS = 1e-8` was clamping B std (which is ~2e-12 T) to 1.0.

**Files changed:**
- Modified `bpc_fno/utils/normalization.py` — lowered `_EPS` to 1e-30

**Verification:**
- `data/processed/normalization.json` created with correct stats
- J_i std: [4.3e-3, 2.5e-2, 3.8e-2] uA/cm^2
- B std: [2.3e-12, 2.0e-12, 1.1e-12] T
- All values non-zero and finite

## Eikonal Data Generation + Phase 1/2 Training (INVALIDATED)

Generated 3998 samples using eikonal approximation (~16s/sample, ~18 hrs total).
Trained Phase 1 (200 epochs, best val_loss=0.0434) and started Phase 2.
**All invalidated** — eikonal approximation is scientifically incorrect.
Deleted all synthetic data and checkpoints.

## Pipeline Rewrite: Real TT2006 + MonodomainSolver (COMPLETED)

**What changed:**
- Rewrote `bpc_fno/simulation/pipeline.py` — removed `_generate_synthetic_activation()`
- New pipeline uses: CellMLLoader → TT2006Runner → V_m→I_ion lookup tables → MonodomainSolver → BiotSavart
- Conductance scale parameter mapping: I_Na→fast_sodium_current.g_Na, etc.
- Transmural heterogeneity: separate lookup tables for endo/mid/epi cell types

**BLOCKED:** Myokit requires C compiler (MSVC Build Tools on Windows, gcc on Linux).
Local Windows machine does not have MSVC Build Tools installed.
Data generation must run on a Linux machine (cloud or WSL2).

See BLOCKED.md and CLOUD_DATAGEN.md for details and next steps.

## Pipeline Fix: TT2006 Stimulus + Fast Solver (COMPLETED)

**Problems fixed:**
1. `engine.time` → `environment.time` in tt2006_runner.py log vars
2. Myokit stimulus binding: CellML model's internal piecewise i_Stim didn't
   respond to Myokit protocol. Fixed by binding `membrane.i_Stim` to `pace`.
3. Added `log_interval=0.1` to `sim.run()` for fine AP waveform resolution.
4. Replaced `spsolve` (direct) with explicit Forward Euler diffusion in
   monodomain.py — 100x speedup. Niederer benchmark still passes (6/6).
5. V_m→I_ion lookup table approach abandoned (hysteresis makes it non-functional).
   Replaced with activation-time propagation using real TT2006 AP waveforms
   per cell type, with anisotropic CV from fiber orientation.

**Smoke test results (3/3 passed):**
- Per sample: 11.3s (10s Myokit + 1s Biot-Savart)
- V_m: [-86, +85] mV (real TT2006 AP morphology)
- B: [5e-12, 9e-12] T (physiologically correct MCG range)
- J_i: [1.6, 1.9] uA/cm² (correct intracellular current density)
- Transmural heterogeneity present (endo/mid/epi different APDs)

**Estimated 4000-sample generation:**
- Sequential: ~12.6 hours
- 8 workers: ~1.6 hours

## WinError 206 Fix (COMPLETED)

**Diagnosis:** Samples 0-102 passed (Myokit cache warm), 103+ all failed.
978 stale myokit cache dirs in TEMP. os.add_dll_directory() fails on the
104-char sundials path during fresh C extension compilation.

**Four fixes applied:**
- Fix A: Redirect TEMP/TMP to C:\tmp\mk in tt2006_runner.py (module level)
- Fix B: Created scripts/warm_myokit_cache.py to pre-compile all 3 cell types
- Fix C: Created C:\mk junction to myokit package; patched _sim/__init__.py
  to use short junction path in os.add_dll_directory()
- Fix D: Added 3-attempt retry loop for WinError 206 in run_single()

**Verification results:**
- 10/10 samples: PASS (100%)
- 50/50 samples: PASS (100%), 7.8s/sample
- 4000-sample generation: RUNNING (~8.7 hours ETA)

## What works and is ready:
- Pipeline code is correct and tested up to the Myokit compilation step
- All non-Myokit components verified (geometry, conductivity, Biot-Savart, noise model)
- Model architecture verified (shared weights, 64.7M params, CUDA forward pass works)
- Training scripts tested and working (Phase 1 completed 200 epochs on eikonal data)
- Cloud deployment instructions written in CLOUD_DATAGEN.md
