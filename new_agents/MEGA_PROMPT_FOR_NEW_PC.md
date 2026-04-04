# Mega Prompt For New PC

Use the text below as the starting prompt for Codex, Claude Code, or another
coding agent on the new workstation.

```text
You are taking over development of the BPC-FNO project on a new workstation.

Repository:
- GitHub repo: https://github.com/gensahilsingh/bpc_fno_v2.git
- Branch: main
- Latest known handoff commit: 65f0376
- If you need your own branch, create one from main after syncing.

New workstation specs:
- GPU: NVIDIA RTX 5090
- RAM: 64 GB
- CPU: Ryzen 9 9950X3D

Old workstation for context:
- GPU: RTX 4060 Ti
- RAM: 32 GB
- CPU: i9-10900K

Your job is to continue development safely and efficiently without rediscovering
project history from scratch.

HIGH-PRIORITY INSTRUCTIONS

1. Do not treat this repo as a blank slate.
2. Read the codebase first before making assumptions.
3. Specifically read the entire `new_agents/` folder after your initial repo scan.
4. Do not assume the old eikonal synthetic pipeline is acceptable for final
   production data.
5. Do not claim the Linux/openCARP path is fully validated unless you actually
   run the strict Linux benchmark and smoke test.
6. Be careful with dataset paths and normalization paths. Do not accidentally
   train on stale legacy data.
7. Be aware that `preload_to_ram: true` will become a system RAM bottleneck on
   large datasets on this 64 GB machine.

PROJECT SUMMARY

This repo implements BPC-FNO, a biophysical cardiac Fourier Neural Operator
pipeline that reconstructs 3D intracellular current density `J_i(x,t)` from
surface magnetic measurements `B(r,t)`.

Current scientific direction:
- Production data generation target: Linux/cloud `openCARP` monodomain backend
- Windows local path: smoke-only hybrid backend
- Legacy eikonal path: fallback only, not scientifically acceptable as final
  training data

Training target:
- Architecture A
- Time-series training path currently active
- Current config uses `n_output_timesteps: 3`

IMPORTANT CURRENT STATE

The repo already contains:
- backend-aware simulation stack
- time-series-capable model/data/training stack
- resume-capable training scripts
- cloud-safety hardening
- cloud data-generation and training docs
- agent handoff docs

The repo is not “done.” The major remaining milestones are:
1. benchmark local training on the RTX 5090
2. validate the Linux/openCARP production path
3. generate the final monodomain dataset
4. recompute normalization on that final dataset
5. run the final production training

FILES YOU MUST READ EARLY

Read these first, in this order if possible:

1. `new_agents/START_HERE_FOR_AGENTS.md`
2. `new_agents/AGENT_MEGA_CONTEXT.md`
3. `configs/arch_a.yaml`
4. `TIME-SERIES-RESULTS.md`
5. `docs/MONODOMAIN_IMPLEMENTATION_SPEC.md`
6. `docs/OPENCARP_CLOUD_RUNBOOK.md`
7. `docs/CLOUD_CPU_DATAGEN_INSTALL.md`
8. `docs/CLOUD_GPU_TRAINING_INSTALL.md`
9. `README.md`
10. `PROGRESS.md`
11. `AUDIT_REPORT.md`

After that, inspect these code areas carefully:
- `bpc_fno/models/`
- `bpc_fno/data/`
- `bpc_fno/simulation/`
- `bpc_fno/training/`
- `bpc_fno/utils/`
- `scripts/`

KEY TECHNICAL TRUTHS

1. Production data generation
- Final intended production backend is `openCARP` on Linux/cloud.
- Windows hybrid backend is explicitly smoke-only.
- Do not confuse those two.

2. Time-series training
- The repo was refactored from `T=1` to time-series support.
- Current config in `configs/arch_a.yaml` is `T=3`.
- The old 4060 Ti T=3 crash happened before later fixes and on much weaker hardware.
- In particular, the geometry-conditioning memory bug in `forward_pino.py`
  has already been fixed.

3. RAM constraints on the new PC
- This machine has 64 GB RAM.
- Current config still has `preload_to_ram: true`.
- That is okay for moderate local datasets, but not okay for the full 15k
  production dataset.
- For full production-size local experiments, you will likely need
  `preload_to_ram: false`.

4. Cloud-safety hardening already done
These classes of bugs were already fixed:
- evaluation normalizer crash
- generation silently succeeding despite failed samples
- normalization split mismatch
- stale resume generator
- silent disabling of physics loss
- evaluation not actually reporting physics threshold results
- incorrect openCARP `stimulus_mask`
- benchmark threshold semantics drift

5. Resume support exists
- `scripts/train_forward.py` supports `--resume`
- `scripts/train_joint.py` supports `--resume`
- checkpoint saving has been hardened for long runs

DO NOT MAKE THESE MISTAKES

- Do not start by editing major model logic before understanding the current repo state.
- Do not assume the repo’s old synthetic data is the right dataset for the final run.
- Do not assume Linux/openCARP has been fully proven just because the code exists.
- Do not assume `preload_to_ram: true` is safe on the full dataset.
- Do not accidentally use stale legacy normalization or stale legacy data paths.
- Do not report “production ready” unless Linux benchmark + smoke generation have actually run.

INITIAL SETUP TASKS

If the repo is not already on disk:

```bash
git clone https://github.com/gensahilsingh/bpc_fno_v2.git
cd bpc_fno_v2
git checkout main
git pull origin main
```

Create the environment, install dependencies, and verify the machine:

```bash
python -m venv .venv
# activate it for your shell
pip install --upgrade pip wheel setuptools
pip install -e ".[test]"
pytest -q
python -m compileall -q bpc_fno scripts tests
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

FIRST DEVELOPMENT TASKS

After reading the required files and verifying the environment, do the
following:

1. Read over the codebase broadly.
- Build a high-level map of the repo.
- Understand how data generation, loading, normalization, training, and
  evaluation connect.

2. Specifically read everything in `new_agents/`.
- Treat that folder as the authoritative agent handoff context.
- Cross-check the claims in those docs against the actual codebase.

3. Verify the current training path on the new PC.
- Inspect `configs/arch_a.yaml`
- Inspect `scripts/train_forward.py`
- Inspect `scripts/train_joint.py`
- Inspect `bpc_fno/data/synthetic_dataset.py`
- Inspect `bpc_fno/models/forward_pino.py`
- Inspect `bpc_fno/models/bpc_fno_a.py`

4. Benchmark local training on the RTX 5090.
Run a short Phase 1 benchmark with explicit paths:

```bash
python scripts/train_forward.py \
  --config configs/arch_a.yaml \
  --data-dir <dataset_dir> \
  --normalization <normalization_json> \
  --checkpoint-dir checkpoints/local_5090_benchmark
```

Collect:
- epoch time
- VRAM usage
- RAM usage
- whether `preload_to_ram: true` is appropriate
- whether T=3 is now practical locally

5. Test resume behavior once.
```bash
python scripts/train_forward.py \
  --config configs/arch_a.yaml \
  --data-dir <dataset_dir> \
  --normalization <normalization_json> \
  --checkpoint-dir checkpoints/local_5090_benchmark \
  --resume
```

6. Report the local training feasibility.
Specifically answer:
- Can Architecture A be trained locally on this 5090?
- Can T=3 be trained locally?
- What dataset scale is practical locally?
- Should `preload_to_ram` stay on or be disabled?
- Which work should stay local vs move to cloud?

AFTER THE LOCAL BENCHMARK

Your next recommendation should be one of:
- local serious experimentation on the 5090
- Linux/openCARP validation next
- full cloud production data generation next

That recommendation must be based on actual measured local behavior, not
generic GPU assumptions.

WHEN THE TIME COMES FOR PRODUCTION

Before any full paid data-generation or training job:
1. Linux/openCARP benchmark must be run
2. Linux/openCARP smoke generation must be run
3. production monodomain dataset must be generated
4. normalization must be recomputed on the final dataset
5. only then should final training begin

FINAL EXPECTATION

Do not just skim the docs and start coding randomly.
Understand the repo, read the `new_agents/` folder carefully, verify the
environment, benchmark the 5090 properly, and then make the next technical
recommendation from evidence.
```
