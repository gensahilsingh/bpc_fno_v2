# Start Here For Agents

This file is the fast-start brief for a fresh Codex or Claude session on a new
machine.

Read this first, then read
[AGENT_MEGA_CONTEXT.md](C:/Users/wishi/OneDrive/Desktop/bpc_fno_v2/new_agents/AGENT_MEGA_CONTEXT.md).

## Goal

Do not treat this repo as a blank slate.

The project already has:

- a backend-aware simulation stack
- a time-series-capable Arch A training path
- resume-capable training scripts
- cloud-safety hardening for generation/training/evaluation
- cloud docs and runbooks

The next real milestones are:

1. benchmark local training on the new `RTX 5090` machine
2. validate the Linux `openCARP` path
3. generate the final monodomain dataset
4. recompute normalization
5. run final training

## Read These 5 Files First

1. [new_agents/AGENT_MEGA_CONTEXT.md](C:/Users/wishi/OneDrive/Desktop/bpc_fno_v2/new_agents/AGENT_MEGA_CONTEXT.md)
2. [configs/arch_a.yaml](C:/Users/wishi/OneDrive/Desktop/bpc_fno_v2/configs/arch_a.yaml)
3. [TIME-SERIES-RESULTS.md](C:/Users/wishi/OneDrive/Desktop/bpc_fno_v2/TIME-SERIES-RESULTS.md)
4. [docs/MONODOMAIN_IMPLEMENTATION_SPEC.md](C:/Users/wishi/OneDrive/Desktop/bpc_fno_v2/docs/MONODOMAIN_IMPLEMENTATION_SPEC.md)
5. [docs/OPENCARP_CLOUD_RUNBOOK.md](C:/Users/wishi/OneDrive/Desktop/bpc_fno_v2/docs/OPENCARP_CLOUD_RUNBOOK.md)

## Core Truths

- Production data target is Linux/cloud `openCARP`, not the old eikonal path.
- Windows hybrid backend is `SMOKE_ONLY`.
- Current training config is time-series `T=3`.
- The old 4060 Ti T=3 crash should not be treated as definitive anymore.
- The 5090 workstation is strong enough for serious local Arch A training.
- Full `15k` dataset with `preload_to_ram: true` will not fit comfortably in
  `64 GB` RAM.

## First 30 Minutes

### 1. Verify the repo and environment

Run:

```bash
pytest -q
python -m compileall -q bpc_fno scripts tests
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### 2. Confirm local training inputs

Figure out:

- which dataset directory is available on the new machine
- which normalization JSON goes with it
- whether this is a small local dataset or a large dataset

Do not assume defaults blindly.

### 3. Benchmark the 5090

Run a short Phase 1 training benchmark with explicit paths and a dedicated
checkpoint directory:

```bash
python scripts/train_forward.py \
  --config configs/arch_a.yaml \
  --data-dir <dataset_dir> \
  --normalization <normalization_json> \
  --checkpoint-dir checkpoints/local_5090_benchmark
```

Capture:

- epoch time
- VRAM usage
- RAM usage
- whether `preload_to_ram: true` is still the right local setting

### 4. Test resume once

```bash
python scripts/train_forward.py \
  --config configs/arch_a.yaml \
  --data-dir <dataset_dir> \
  --normalization <normalization_json> \
  --checkpoint-dir checkpoints/local_5090_benchmark \
  --resume
```

## What Not To Do

- Do not claim the repo is fully production-validated before Linux
  `openCARP` benchmark + smoke test run.
- Do not treat the Windows hybrid backend as production data generation.
- Do not accidentally train on stale legacy data just because a path happens to
  exist.
- Do not keep `preload_to_ram: true` on the final full dataset without checking
  RAM headroom.

## Likely Best Next Step

After the local 5090 benchmark, decide between:

- local serious experimentation on a smaller dataset
- cloud Linux `openCARP` validation and production generation

That choice should be driven by measured local performance, not guesses.
