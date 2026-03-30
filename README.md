# BPC-FNO: Biophysical Cardiac Fourier Neural Operator

Reconstructs 3D intracellular cardiac current density J_i(x,t) from
magnetoionography (MIG) / magnetocardiography (MCG) surface field measurements B(r,t).

## Architecture A

- Stage 1: Voxel geometry encoding (SDF + fiber orientation)
- Stage 2: Physics-informed FNO forward operator (J_i -> B)
- Stage 3: Inverse encoder with shared FNO weights (B -> latent z)
- Stage 4: VAE decoder (z -> J_i reconstruction)
- Stage 5: Uncertainty quantification via posterior sampling

## Setup

```bash
pip install -e ".[test]"
```

## Data Generation

```bash
python scripts/download_cellml.py
python scripts/download_kcd.py
python scripts/fit_noise_model.py
python scripts/generate_synthetic.py
python scripts/compute_normalization.py
```

## Training

```bash
python scripts/train_forward.py    # Phase 1: forward model
python scripts/train_joint.py      # Phase 2: joint training
python scripts/evaluate.py         # Evaluation
```
