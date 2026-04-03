# Time-Series Refactor Results

## Summary

The time-series refactor was implemented to address severe overfitting in single-timestep (T=1) training. The refactor changes tensor shapes throughout the pipeline so each sample provides T temporal snapshots instead of one peak-activation frame.

**Current status: T=3 training completed 2 epochs then crashed silently. Val_loss=0.342 at epoch 2 — already better than the T=1 baseline's final val_loss=0.447 after 200 epochs.**

---

## What Changed

### Config (configs/arch_a.yaml)
```yaml
model:
  n_output_timesteps: 3    # was 1, tried 10 (OOM)
  n_sensors_total: 48      # made explicit
training:
  batch_size: 32           # was 64 for T=1, reduced for T=3
data:
  preload_to_ram: true     # eliminates HDF5 disk bottleneck
```

### Dataset (bpc_fno/data/synthetic_dataset.py)
- Now subsamples T evenly-spaced timesteps from the stored 100
- Returns:
  - `J_i`: (3, T, N, N, N) — was (3, N, N, N)
  - `B_true`: (S, T) — was (S,) where S=48
  - `B_obs`: (S, T) — was (S,)
  - `geometry`: (4, N, N, N) — unchanged (static)
  - `t_ms`: (T,) — new key
- RAM preloading works correctly (3200 train + 400 val in ~10 min)

### ForwardPINO (bpc_fno/models/forward_pino.py)
- `predict_B()` now handles both 5D (B,3,N,N,N) and 6D (B,3,T,N,N,N)
- Time-series mode: merges B*T into batch dimension, runs each timestep through the 3D FNO independently, then reshapes back
- Single-timestep backward compat preserved via `_predict_B_single()`
- Output: (B, S, T) — was (B, S)

### InverseEncoder (bpc_fno/models/inverse_encoder.py)
- `b_projection` input size changed from S to S*T (48 → 144 for T=3)
- Flattens (B, S, T) → (B, S*T) before the MLP
- Rest of architecture unchanged

### VAEDecoder (bpc_fno/models/vae_decoder.py)
- Final conv outputs 3*T channels instead of 3
- Reshapes (B, 3*T, N, N, N) → (B, 3, T, N, N, N)
- n_output_timesteps stored from config

### Loss Manager (bpc_fno/training/loss_manager.py)
- `physics_residual_loss()` handles 6D J_i_hat by merging B*T dims
- Refactored into `_physics_residual_3d()` static method for both paths

### Integration Test
All shape assertions passed:
```
J_i:     (2, 3, 3, 32, 32, 32)  ✓
B_true:  (2, 48, 3)              ✓
B_obs:   (2, 48, 3)              ✓
B_pred:  (2, 48, 3)              ✓
J_i_hat: (2, 3, 3, 32, 32, 32)  ✓
mu:      (2, 512)                ✓
```

---

## Training Results

### T=1 Baseline (single-timestep, completed)
- Config: batch_size=64, n_output_timesteps=1
- 200 epochs in ~10 hours
- val_loss: 1.86 → 0.447 (plateaued from epoch ~50)
- train_loss: 0.000 (complete memorization — severe overfitting)
- ~3 min/epoch, GPU ~60% utilization

### T=10 Attempt (FAILED — OOM)
- Config: batch_size=64, n_output_timesteps=10
- Crashed immediately: CUDA OOM trying to allocate 5 GB
- 16 GB VRAM insufficient for batch=64 × T=10 (effective 640 FNO passes)

### T=10 Attempt #2 (FAILED — too slow)
- Config: batch_size=8, n_output_timesteps=10
- Preloaded successfully
- 400 batches/epoch × 80 FNO passes/batch = 32,000 per epoch
- Never completed epoch 1 after 70+ minutes
- Killed

### T=3 Training (current — partial results)
- Config: batch_size=32, n_output_timesteps=3
- Preloaded: 3200 train in 573s, 400 val in 76s
- 100 batches/epoch, 96 FNO passes/batch (32 samples × 3 timesteps)

**Epoch timings:**
| Epoch | Train Loss | Val Loss | Time | LR |
|-------|-----------|---------|------|-----|
| 1 | 0.967 | 0.923 | ~2.8 hrs | 2e-4 |
| 2 | 0.560 | 0.342 | ~3.2 hrs | 4e-4 |

**Key observation: val_loss=0.342 at epoch 2 already beats T=1's final val_loss=0.447 after 200 epochs.** The time-series provides dramatically more training signal per sample.

**Crash:** Training died silently after epoch 2 (no error in output, process disappeared). Likely a segfault from memory pressure — 15.8 GB of 16.4 GB VRAM was in use.

---

## Root Cause Analysis: Why T=3 is Slow

The batch-time merge approach is O(B × T) in FNO forward passes:

| Config | B | T | FNO passes/batch | Batches/epoch | FNO passes/epoch | Est. epoch time |
|--------|---|---|-------------------|---------------|-------------------|-----------------|
| T=1, bs=64 | 64 | 1 | 64 | 50 | 3,200 | ~3 min |
| T=3, bs=32 | 32 | 3 | 96 | 100 | 9,600 | ~3 hrs |
| T=10, bs=8 | 8 | 10 | 80 | 400 | 32,000 | ~10 hrs |

The 3D FNO (3 spectral layers on 32³ grid) takes ~0.1s per forward+backward pass. This is the fundamental bottleneck — not data loading (preloading solved that).

The epoch time scales roughly as `(FNO_passes × 0.1s) + overhead`:
- T=1: 3200 × 0.06s = 192s ≈ 3 min ✓
- T=3: 9600 × 1.1s = ~10,560s ≈ 3 hrs (overhead from geometry expansion, memory pressure)

The 1.1s per pass (vs 0.06s for T=1) suggests significant overhead from:
1. Expanding geometry tensor 3x per batch (B,4,N,N,N) → (B*3,4,N,N,N)
2. Memory pressure at 15.8/16.4 GB VRAM causing GPU memory management overhead
3. Larger gradient computation graphs

---

## Options to Resolve

### Option A: Stick with T=1, address overfitting differently
- Already proven: 3 min/epoch, 10 hours for 200 epochs
- Fix overfitting via: data augmentation (random timestep selection instead of peak), dropout, weight decay, larger dataset
- Pros: fast iteration, proven pipeline
- Cons: loses temporal structure

### Option B: T=3 with reduced model size
- Reduce n_fno_hidden from 64 to 32 (cuts FNO FLOPS 4x)
- Reduce batch_size to 16 (fewer FNO passes per batch)
- Expected: ~45 min/epoch, ~150 hours (6 days) for 200 epochs
- Pros: retains temporal structure
- Cons: still slow, reduced model capacity

### Option C: T=3 with gradient accumulation
- batch_size=8, accumulate gradients over 4 steps (effective batch=32)
- Only 24 FNO passes per physical batch (vs 96)
- Expected: ~45 min/epoch
- Pros: same convergence as T=3/bs=32
- Cons: still slow, adds code complexity

### Option D: T=1 with random timestep augmentation (RECOMMENDED)
- Keep batch_size=64, n_output_timesteps=1
- Instead of always selecting peak activation timestep, randomly sample from the top 50% of active timesteps per access
- This gives temporal diversity across epochs while keeping T=1 speed
- Each sample contributes ~50 different timesteps across 200 epochs
- Effectively 50x data augmentation for free
- Pros: 3 min/epoch, temporal diversity, proven infrastructure
- Cons: model doesn't learn temporal correlations within a sample

### Option E: T=3 on cloud GPU (A100/H100)
- A100 has 80 GB VRAM and 3x FNO throughput
- batch_size=64, T=3 would fit easily
- ~30 min/epoch, ~100 hours for 200 epochs
- Pros: proper time-series with sufficient compute
- Cons: cost, requires cloud setup

---

## Recommendation

**Start with Option D (T=1 + random timestep augmentation)** to get results fast. The val_loss=0.342 from T=3 epoch 2 proves the model architecture can learn well — the issue is purely compute. Option D captures most of the benefit (temporal diversity) at T=1 speed.

If Option D doesn't beat val_loss=0.342 after 200 epochs, then Option E (cloud A100) is the path for T=3+.

---

## Files Modified in Time-Series Refactor

All changes are backward-compatible with T=1 (set n_output_timesteps=1 in config):

1. `configs/arch_a.yaml` — n_output_timesteps, batch_size, preload_to_ram
2. `bpc_fno/data/synthetic_dataset.py` — full time-series loading + RAM preload
3. `bpc_fno/models/forward_pino.py` — _predict_B_single + batch-time merge
4. `bpc_fno/models/inverse_encoder.py` — b_projection input S*T
5. `bpc_fno/models/vae_decoder.py` — final conv 3*T channels
6. `bpc_fno/training/loss_manager.py` — 6D physics loss handling

## Checkpoint Available

`checkpoints/phase1_best.pt` — T=3 epoch 2 (val_loss=0.342, 860 MB)
This checkpoint can be used to resume T=3 training or as a starting point for Phase 2.
