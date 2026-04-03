# Monodomain Data Generation Spec

## Scope

- Linux/cloud production path: fully coupled openCARP monodomain backend.
- Windows local path: smoke-only hybrid backend.
- Legacy eikonal path remains available and is still the default until cloud production is confirmed.

## Backend Contract

All simulation backends implement the same contract:

- Input: `SimulationContext`
- Output: `SimulationResult`

`SimulationContext` contains:

- slab geometry and voxel spacing
- conductivity tensor
- SDF, fiber field, cell-type map, fibrosis mask
- pacing site and sampled parameters
- requested output times
- `save_vm` flag

`SimulationResult` must return:

- `J_i`
- `t_ms`
- `activation_times_ms`
- `stimulus_mask`
- optional `V_m`
- metadata describing exactness and backend provenance

## Production Path

- Backend: `opencarp`
- Grid: `32 x 32 x 32`
- Voxel size: `0.1 cm`
- Duration: `150 ms`
- PDE timestep: `0.05 ms`
- Stored timesteps in HDF5: `100`
- Conductivities for data generation:
  - `sigma_il = 3.0e-3 S/cm`
  - `sigma_it = 3.0e-4 S/cm`
- Fibrosis handling: conductivity reduction only
- `save_vm: false`

Paper-facing wording for the production dataset:

> Synthetic training data were generated using the openCARP simulator solving the monodomain equation with the TenTusscher-Panfilov 2006 human ventricular ionic model integrated via Rush-Larsen operator splitting at 0.05ms timestep on a 32^3 voxel grid at 1mm resolution, capturing the full ventricular depolarization sequence.

## Windows Smoke Path

- Backend: `windows_hybrid`
- Grid: `16 x 16 x 16`
- Voxel size: `0.1 cm`
- Duration: `150 ms`
- PDE timestep: `0.1 ms`
- Stored timesteps in HDF5: `100`
- `save_vm: true`
- `SMOKE_ONLY` semantics:
  - diffusion is solved on-grid with Crank-Nicolson
  - ionics are approximated with TT2006 lookup tables plus eikonal timing
  - data are not suitable for production training

## HDF5 Schema

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

Optional datasets:

- `V_m` when `save_vm: true`
- `geometry/fibrosis_mask`

Required attrs:

- `pacing_site_voxel`
- `pacing_cl_ms`
- `cv_scale`
- `fibrosis_density`
- `ko_mM`
- `conductance_scales`
- `sample_seed`
- `backend`
- `grid_shape`
- `voxel_size_cm`
- `bpc_fno_version`
- `generation_timestamp`

## Config Layout

- `configs/data_gen_eikonal.yaml`
- `configs/data_gen_smoke_windows.yaml`
- `configs/data_gen_opencarp_prod.yaml`
- `configs/data_gen_opencarp_cloud_shard.yaml`
- `configs/benchmark_niederer.yaml`

Config inheritance uses a lightweight `extends:` key resolved relative to the child file.

## Sharding Contract

Generation supports:

- `--sample-start`
- `--sample-count`
- `--shard-id`
- `--num-shards`
- `--seed-offset`

Sharding is deterministic at the sample-id level, so shards can be run independently without duplicate samples.

## Validation Gates

### Windows smoke

All of these must pass:

- no NaNs/Infs
- nonzero `J_i`
- nonzero `B_mig`
- plausible activation/depolarization
- nontrivial adjacent-voxel behavior
- completion within the configured local time budget

### Linux/openCARP production

- strict Niederer benchmark must pass before full cloud generation
- production normalization is computed only after the full monodomain dataset is generated
