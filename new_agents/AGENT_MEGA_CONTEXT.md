# Agent Mega Context

Last updated: 2026-04-04
Repo commit at time of writing: `b456bea`

This file is the high-context handoff for a fresh coding agent on a new machine.
It is meant to prevent the next agent from wasting time rediscovering the
project state, repeating invalidated work, or making scientifically wrong
assumptions.

## 1. Project Purpose

This repository implements `BPC-FNO`, a biophysical cardiac Fourier Neural
Operator pipeline for reconstructing 3D intracellular current density
`J_i(x, t)` from surface magnetic measurements `B(r, t)` such as
magnetocardiography / magnetoionography.

The intended workflow is:

1. Generate synthetic training data from cardiac electrophysiology simulations.
2. Compute normalization statistics on the final production dataset.
3. Train Architecture A in two phases:
   - Phase 1: forward operator only
   - Phase 2: full joint model
4. Evaluate reconstruction quality, forward accuracy, uncertainty, and physics
   consistency.

The project started with an eikonal-based synthetic pipeline. That path is now
scientifically invalid for the paper-grade dataset and has been superseded by a
monodomain-focused roadmap.

## 2. Scientific Ground Truth And Current Strategy

### What is scientifically acceptable now

- Production data generation target:
  fully coupled `openCARP` monodomain on Linux/cloud.
- Ionic model target:
  `TT2006` / Ten Tusscher-Panfilov 2006.
- Production paper wording target:
  synthetic data generated with `openCARP` solving the monodomain equation on a
  `32^3` grid at `1 mm` resolution over `150 ms`, with native ionic integration.

### What is not scientifically acceptable as production

- The old eikonal waveform-stamping dataset as final training data.
- The Windows hybrid solver as production data.

### What is acceptable for local smoke testing

- Windows hybrid backend:
  diffusion solved on-grid, ionics approximated, explicitly `SMOKE_ONLY`.

## 3. Important History

### Phase A: Early synthetic pipeline and training

- A synthetic pipeline was built and made operational.
- A `3998`-sample eikonal dataset was generated.
- Phase 1 training was run successfully on that old dataset.
- That whole dataset was later invalidated because the eikonal propagation
  approach is not appropriate for the intended scientific claim.

### Phase B: Monodomain / backend refactor

The repo was refactored toward a backend-aware generation system with:

- shared pipeline core
- backend abstraction
- legacy eikonal backend kept for fallback
- Windows hybrid backend for smoke tests
- openCARP backend for production
- Chaste stub for architectural completeness

New schema and config layering were added.

### Phase C: Time-series refactor

The training stack was refactored from single-timestep `T=1` to support
time-series training. The dataset now stores and loads `100` timesteps and the
model can subsample `T` outputs for training.

This was done because `T=1` overfit badly.

### Phase D: Hardening pass before paid cloud runs

Several cloud-cost and reproducibility bugs were found and fixed:

- evaluation crash due to fake normalizer proxy
- generation succeeding even when some samples failed
- normalization split mismatch vs actual dataset split
- stale resume generation script
- physics loss being silently disabled in Phase 2
- evaluation not actually computing `pass_physics`
- incorrect stored `stimulus_mask`
- benchmark threshold drift (`0.0 mV` vs intended `-40.0 mV`)

This hardening pass was completed and regression-tested.

## 4. Current Repo Status

At commit `b456bea`:

- tests: `84 passed, 6 skipped`
- compile check passed:
  `python -m compileall -q bpc_fno scripts tests`
- latest push to remote was completed
- large generated data was intentionally not committed

The repo now contains code, configs, docs, tests, and scripts needed to resume
work on a new machine.

## 5. High-Level Architecture

### Model

Main config: [configs/arch_a.yaml](C:/Users/wishi/OneDrive/Desktop/bpc_fno_v2/configs/arch_a.yaml)

Architecture A has:

1. geometry encoder
2. shared 3D FNO backbone
3. forward operator `J_i -> B`
4. inverse encoder `B -> z`
5. VAE decoder `z -> J_i_hat`
6. posterior sampling for uncertainty

Important implementation files:

- [bpc_fno/models/bpc_fno_a.py](C:/Users/wishi/OneDrive/Desktop/bpc_fno_v2/bpc_fno/models/bpc_fno_a.py)
- [bpc_fno/models/forward_pino.py](C:/Users/wishi/OneDrive/Desktop/bpc_fno_v2/bpc_fno/models/forward_pino.py)
- [bpc_fno/models/inverse_encoder.py](C:/Users/wishi/OneDrive/Desktop/bpc_fno_v2/bpc_fno/models/inverse_encoder.py)
- [bpc_fno/models/vae_decoder.py](C:/Users/wishi/OneDrive/Desktop/bpc_fno_v2/bpc_fno/models/vae_decoder.py)

### Data generation

Core simulation orchestration:

- [bpc_fno/simulation/pipeline_core.py](C:/Users/wishi/OneDrive/Desktop/bpc_fno_v2/bpc_fno/simulation/pipeline_core.py)
- [bpc_fno/simulation/pipeline.py](C:/Users/wishi/OneDrive/Desktop/bpc_fno_v2/bpc_fno/simulation/pipeline.py)

Backends:

- [bpc_fno/simulation/backends/eikonal.py](C:/Users/wishi/OneDrive/Desktop/bpc_fno_v2/bpc_fno/simulation/backends/eikonal.py)
- [bpc_fno/simulation/backends/windows_hybrid.py](C:/Users/wishi/OneDrive/Desktop/bpc_fno_v2/bpc_fno/simulation/backends/windows_hybrid.py)
- [bpc_fno/simulation/backends/opencarp.py](C:/Users/wishi/OneDrive/Desktop/bpc_fno_v2/bpc_fno/simulation/backends/opencarp.py)
- [bpc_fno/simulation/backends/chaste.py](C:/Users/wishi/OneDrive/Desktop/bpc_fno_v2/bpc_fno/simulation/backends/chaste.py)

Monodomain helper modules:

- [bpc_fno/simulation/monodomain/hybrid_solver.py](C:/Users/wishi/OneDrive/Desktop/bpc_fno_v2/bpc_fno/simulation/monodomain/hybrid_solver.py)
- [bpc_fno/simulation/monodomain/lookup_ionic.py](C:/Users/wishi/OneDrive/Desktop/bpc_fno_v2/bpc_fno/simulation/monodomain/lookup_ionic.py)
- [bpc_fno/simulation/monodomain/stimulus.py](C:/Users/wishi/OneDrive/Desktop/bpc_fno_v2/bpc_fno/simulation/monodomain/stimulus.py)

### Data loading and normalization

- [bpc_fno/data/synthetic_dataset.py](C:/Users/wishi/OneDrive/Desktop/bpc_fno_v2/bpc_fno/data/synthetic_dataset.py)
- [bpc_fno/data/data_module.py](C:/Users/wishi/OneDrive/Desktop/bpc_fno_v2/bpc_fno/data/data_module.py)
- [bpc_fno/utils/normalization.py](C:/Users/wishi/OneDrive/Desktop/bpc_fno_v2/bpc_fno/utils/normalization.py)
- [bpc_fno/utils/data_paths.py](C:/Users/wishi/OneDrive/Desktop/bpc_fno_v2/bpc_fno/utils/data_paths.py)

### Training and evaluation

- [scripts/train_forward.py](C:/Users/wishi/OneDrive/Desktop/bpc_fno_v2/scripts/train_forward.py)
- [scripts/train_joint.py](C:/Users/wishi/OneDrive/Desktop/bpc_fno_v2/scripts/train_joint.py)
- [scripts/evaluate.py](C:/Users/wishi/OneDrive/Desktop/bpc_fno_v2/scripts/evaluate.py)
- [bpc_fno/training/loss_manager.py](C:/Users/wishi/OneDrive/Desktop/bpc_fno_v2/bpc_fno/training/loss_manager.py)
- [bpc_fno/utils/checkpointing.py](C:/Users/wishi/OneDrive/Desktop/bpc_fno_v2/bpc_fno/utils/checkpointing.py)

## 6. Current Config Truths

### Architecture A current defaults

From [configs/arch_a.yaml](C:/Users/wishi/OneDrive/Desktop/bpc_fno_v2/configs/arch_a.yaml):

- grid size: `32`
- hidden width: `64`
- FNO layers: `3`
- latent dim: `512`
- total sensors: `48`
- output timesteps: `3`
- batch size: `32`
- `preload_to_ram: true`

Important note:
`preload_to_ram: true` is fine for smaller local datasets, but it becomes a
system RAM bottleneck on large datasets. Do not blindly keep it enabled on the
full production dataset.

### Data-generation config layout

- [configs/data_gen.yaml](C:/Users/wishi/OneDrive/Desktop/bpc_fno_v2/configs/data_gen.yaml)
- [configs/data_gen_eikonal.yaml](C:/Users/wishi/OneDrive/Desktop/bpc_fno_v2/configs/data_gen_eikonal.yaml)
- [configs/data_gen_smoke_windows.yaml](C:/Users/wishi/OneDrive/Desktop/bpc_fno_v2/configs/data_gen_smoke_windows.yaml)
- [configs/data_gen_opencarp_prod.yaml](C:/Users/wishi/OneDrive/Desktop/bpc_fno_v2/configs/data_gen_opencarp_prod.yaml)
- [configs/data_gen_opencarp_cloud_shard.yaml](C:/Users/wishi/OneDrive/Desktop/bpc_fno_v2/configs/data_gen_opencarp_cloud_shard.yaml)
- [configs/benchmark_niederer.yaml](C:/Users/wishi/OneDrive/Desktop/bpc_fno_v2/configs/benchmark_niederer.yaml)

## 7. Data Schema Truths

The new monodomain-oriented schema is the intended direction.

Required datasets:

- `J_i`
- `B_mig`
- `B_mig_noisy`
- `geometry/sdf`
- `geometry/fiber`
- `geometry/cell_type_map`
- `sensor_positions`
- `t_ms`
- `activation_times_ms`
- `stimulus_mask`

Optional:

- `V_m` when `save_vm: true`
- `geometry/fibrosis_mask`

The dataset loader is backward-compatible with legacy files where possible, but
new work should target the new schema.

## 8. What Has Been Fixed

### Training / model fixes

- time-series shapes are supported end-to-end
- legacy `B_mig` vs new `B_obs` input mismatch was fixed
- evaluation stack was updated to the new batch contract
- Lightning/path mismatches that broke optimizer grouping were fixed
- time-series physics metric bug was fixed
- geometry encoder memory bug was fixed in forward time-series mode
- checkpoint saving and `--resume` were added to both training scripts

### Cloud-safety fixes

- evaluation uses the real normalizer, not a fake proxy
- training/evaluation/normalization scripts support explicit `--data-dir`
- generation now fails loudly on sample failures
- generation writes samples atomically
- resume generation was rewritten around current configs and sharding
- physics-loss failures in Phase 2 are now fatal
- evaluation now reports physics residual and `pass_physics`
- openCARP `stimulus_mask` now matches the actual stimulated region
- benchmark threshold semantics were standardized to `-40.0 mV`

## 9. What Is Still Not Fully Proven

This section matters. Do not let a new agent claim the project is already fully
validated end-to-end.

### Not yet proven locally

- the real Linux `openCARP` production backend has not been run end-to-end from
  this Windows environment
- the strict Linux Niederer benchmark has not been executed on the final target
  Linux/cloud machine yet
- the full 15k monodomain dataset has not been generated yet
- normalization has not yet been recomputed on the final monodomain dataset
- the final production training run has not yet been executed

### Meaning

The codebase is much safer than before, but the first Linux preflight is still
mandatory before paying for the full generation/training run.

## 10. Time-Series Refactor Status

See [TIME-SERIES-RESULTS.md](C:/Users/wishi/OneDrive/Desktop/bpc_fno_v2/TIME-SERIES-RESULTS.md).

Important facts:

- T=1 completed historically, but overfit badly.
- T=10 was too large for the old 4060 Ti setup.
- T=3 showed much better validation behavior early.
- The old T=3 crash on the 4060 Ti was strongly tied to VRAM pressure and the
  old geometry-conditioning inefficiency.
- The geometry-conditioning memory bug has since been fixed in
  [forward_pino.py](C:/Users/wishi/OneDrive/Desktop/bpc_fno_v2/bpc_fno/models/forward_pino.py).

Do not trust the old T=3 crash as evidence that T=3 is still impossible. That
result predates later fixes and also came from much weaker hardware.

## 11. New Workstation Context

User has moved to a new workstation with:

- GPU: `RTX 5090`
- system RAM: `64 GB`
- CPU: `Ryzen 9 9950X3D`

Previous workstation was:

- GPU: `RTX 4060 Ti`
- system RAM: `32 GB`
- CPU: `i9-10900K`

### Practical implication

The new workstation is a real local training machine for this repo.

It should be able to handle:

- serious Arch A development
- full T=1 training locally
- T=3 local experiments and substantial runs
- local benchmarking and hyperparameter work

But the system RAM still constrains the largest dataset workflows.

### RAM bottleneck

From the dataset loader's own estimate, with `T=3`:

- `4000` total samples preload to about `31.6 GB` train+val
- `6000` total samples preload to about `47.5 GB`
- `8000` total samples preload to about `63.3 GB`
- `15000` total samples preload to about `118.7 GB`

So:

- `preload_to_ram: true` is acceptable for moderate local datasets
- `preload_to_ram: true` is not acceptable for the full `15k` production set on
  a `64 GB` machine
- for full-size local experiments, disable preloading and rely on fast NVMe

### Recommended split of work

- local 5090 machine:
  model iteration, T=3 experiments, smoke training, debugging, partial/full
  local runs if desired
- cloud Linux CPU:
  production openCARP data generation
- cloud high-memory GPU:
  final locked production training run

## 12. Existing Docs Worth Reading

These are the main docs a new agent should read after this file:

- [README.md](C:/Users/wishi/OneDrive/Desktop/bpc_fno_v2/README.md)
- [PROGRESS.md](C:/Users/wishi/OneDrive/Desktop/bpc_fno_v2/PROGRESS.md)
- [AUDIT_REPORT.md](C:/Users/wishi/OneDrive/Desktop/bpc_fno_v2/AUDIT_REPORT.md)
- [TIME-SERIES-RESULTS.md](C:/Users/wishi/OneDrive/Desktop/bpc_fno_v2/TIME-SERIES-RESULTS.md)
- [docs/MONODOMAIN_IMPLEMENTATION_SPEC.md](C:/Users/wishi/OneDrive/Desktop/bpc_fno_v2/docs/MONODOMAIN_IMPLEMENTATION_SPEC.md)
- [docs/OPENCARP_CLOUD_RUNBOOK.md](C:/Users/wishi/OneDrive/Desktop/bpc_fno_v2/docs/OPENCARP_CLOUD_RUNBOOK.md)
- [docs/CLOUD_CPU_DATAGEN_INSTALL.md](C:/Users/wishi/OneDrive/Desktop/bpc_fno_v2/docs/CLOUD_CPU_DATAGEN_INSTALL.md)
- [docs/CLOUD_GPU_TRAINING_INSTALL.md](C:/Users/wishi/OneDrive/Desktop/bpc_fno_v2/docs/CLOUD_GPU_TRAINING_INSTALL.md)

## 13. Recommended Immediate Next Steps

These are the highest-value actions now.

### A. Bring up the new workstation cleanly

1. Clone/pull the repo on the new PC.
2. Create a fresh Python environment.
3. Install dependencies.
4. Run:
   - `pytest -q`
   - `python -m compileall -q bpc_fno scripts tests`
5. Confirm CUDA + PyTorch work on the 5090.

### B. Benchmark local training on the 5090

Run a short local benchmark, ideally:

- `2-5` epochs of Phase 1
- current Arch A config
- current T=3 path

Capture:

- epoch time
- VRAM usage
- system RAM usage
- whether `preload_to_ram: true` is still appropriate

This will give a much better basis for deciding what work stays local.

### C. Decide local training mode

Likely options:

- small/medium local dataset with `preload_to_ram: true`
- larger local dataset with `preload_to_ram: false`

Do not assume the old 4060 Ti timings still apply.

### D. Prepare Linux validation path

Before any paid production run:

1. get a Linux environment with `openCARP`
2. run:
   `python scripts/validate_niederer_linux.py --config configs/benchmark_niederer.yaml`
3. run a one-sample monodomain smoke generation
4. run `verify_data.py` on that output

Only after that should the full 15k generation run start.

### E. Production data generation

Intended target:

- `15000` samples
- `6` CPU shards
- `openCARP`
- `save_vm: false`
- recompute normalization only after final merged dataset is ready

### F. Final training

After production dataset + normalization are ready:

1. Phase 1 on final dataset
2. verify resume behavior once
3. Phase 2 on final dataset
4. final evaluation

## 14. Known Practical Pitfalls

### Data path confusion

Always be explicit about `--data-dir` and normalization paths. Do not let a new
agent accidentally train on stale or legacy data.

### Old eikonal assumptions

Do not let a new agent assume the old eikonal data path is the scientific end
goal. It is now fallback/smoke-only territory.

### Preloading on large datasets

Blindly leaving `preload_to_ram: true` on the full production dataset will
cause avoidable RAM pressure.

### Declaring production readiness too early

Do not claim the openCARP production path is fully validated until the strict
Linux benchmark and Linux smoke generation actually run successfully.

### Confusing smoke-only with production

The Windows hybrid path is for local validation and development only.

## 15. Recommended Command Checklist For The Next Agent

### Verify the repo

```bash
pytest -q
python -m compileall -q bpc_fno scripts tests
```

### Verify the GPU

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### Local training benchmark

```bash
python scripts/train_forward.py \
  --config configs/arch_a.yaml \
  --data-dir <local_dataset_dir> \
  --normalization <normalization_json> \
  --checkpoint-dir checkpoints/local_5090_benchmark
```

### Resume sanity check

```bash
python scripts/train_forward.py \
  --config configs/arch_a.yaml \
  --data-dir <local_dataset_dir> \
  --normalization <normalization_json> \
  --checkpoint-dir checkpoints/local_5090_benchmark \
  --resume
```

### Linux production preflight

```bash
python scripts/validate_niederer_linux.py --config configs/benchmark_niederer.yaml
```

### Production shard generation pattern

```bash
python scripts/generate_synthetic.py \
  --config configs/data_gen_opencarp_cloud_shard.yaml \
  --pipeline monodomain \
  --backend opencarp \
  --shard-id 0 \
  --num-shards 6 \
  --sample-start 0 \
  --sample-count 15000 \
  --seed-offset 0
```

## 16. What The Next Agent Should Probably Do First

If a new agent opens this repo on the 5090 machine, the first smart move is:

1. verify environment and tests
2. benchmark T=3 Phase 1 locally on the 5090
3. decide whether to keep `preload_to_ram: true` for the local working dataset
4. prepare Linux/openCARP benchmark execution

That is the shortest path to useful new information.

## 17. Bottom Line

This repo is no longer in the "barely working prototype" state.

It now has:

- a backend-aware simulation stack
- a modernized time-series training path
- resume-capable training scripts
- explicit cloud-safety hardening
- cloud runbooks and install guides
- regression tests for the major failures already found

But the real production milestone has still not happened yet:

- strict Linux/openCARP validation
- full 15k monodomain generation
- production normalization
- final full training run

That is the work now in front of the next agent.
