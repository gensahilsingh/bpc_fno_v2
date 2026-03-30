# Cloud Data Generation Instructions

The TT2006 ionic model simulation via Myokit requires a C compiler.
This works out of the box on Linux (gcc) but requires MSVC Build Tools on Windows.
Since data generation is CPU-intensive, a cloud Linux instance is recommended.

## Requirements
- CPU instance: 8+ cores recommended (e.g. Vast.ai, Lambda, AWS EC2)
- RAM: 16GB minimum (32GB recommended for safety)
- Storage: 50GB free (actual output ~15GB for 4000 samples)
- OS: Ubuntu 22.04
- gcc/g++ installed (default on Ubuntu)

## Setup

```bash
# Clone the repo
git clone <repo-url> bpc_fno_v2
cd bpc_fno_v2

# Install Python dependencies
pip install -e .

# Download CellML model files (small, ~1MB total)
python scripts/download_cellml.py

# Copy noise_model.json from local machine (or refit from KCD data)
# scp local:bpc_fno_v2/data/processed/noise_model.json data/processed/
# OR if KCD data is available:
# python scripts/fit_noise_model.py
```

## Verify physics before full run

```bash
# Run physics tests
pytest tests/test_biot_savart.py tests/test_monodomain.py -v

# Generate 3 test samples
python scripts/generate_synthetic.py --n_samples 3

# Verify output
python -c "
import h5py, glob, numpy as np
files = sorted(glob.glob('data/synthetic/*.h5'))
print(f'Files: {len(files)}')
for f in files:
    with h5py.File(f, 'r') as hf:
        J = hf['J_i'][:]
        B = hf['B_mig'][:]
        t = hf['t_ms'][:]
        print(f'{f}: J={J.shape} B={B.shape} t=[{t[0]:.0f},{t[-1]:.0f}]ms '
              f'|B|=[{abs(B).min():.1e},{abs(B).max():.1e}] '
              f'|J|=[{abs(J).min():.1e},{abs(J).max():.1e}]')
"
```

Expected output per sample:
- J_i shape: (100, 32, 32, 32, 3) — 100 timesteps, 32^3 grid, 3 components
- B_mig shape: (100, 16, 3) — 100 timesteps, 16 sensors, 3 components
- t_ms: 0 to ~500ms
- |B| in [1e-13, 1e-10] Tesla
- |J| in [1e-3, 1e2] uA/cm^2
- V_m NOT stored (save_vm=false)

## Full generation

```bash
# Generate all 4000 samples (sequential — Myokit is not multiprocess-safe)
python scripts/generate_synthetic.py --n_samples 4000

# Or use the resume script if the process gets interrupted:
python scripts/generate_synthetic_resume.py --n-samples 4000 --batch-size 200
```

Estimated time: depends on hardware.
- Single-cell TT2006 sim: ~10-30s per sample (pre-pacing + recording)
- Monodomain 32^3 solver: ~60-300s per sample (500ms at dt=0.05ms)
- Biot-Savart: ~1s per sample
- Total: ~2-5 min per sample → 4000 samples ≈ 6-14 hours on 8-core machine

## Download results

```bash
# Compress output for download
tar -czf synthetic_data.tar.gz data/synthetic/

# Download to local machine
# scp cloud:bpc_fno_v2/synthetic_data.tar.gz .

# Extract on local machine
# cd bpc_fno_v2
# tar -xzf synthetic_data.tar.gz
```

## After downloading data locally

```bash
# Recompute normalization
python scripts/compute_normalization.py

# Resume training pipeline
python scripts/train_forward.py --config configs/arch_a.yaml
python scripts/train_joint.py --config configs/arch_a.yaml --phase1-checkpoint checkpoints/phase1_best.pt
python scripts/evaluate.py --config configs/arch_a.yaml --checkpoint checkpoints/phase2_best.pt
```
