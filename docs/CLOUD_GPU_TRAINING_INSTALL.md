# Cloud GPU Training Install Guide

## Purpose

Use this guide for the long `T=3` ML training run after the full monodomain dataset is generated and normalization has been recomputed.

This guide assumes:

- the production dataset is already generated
- the new normalization file has already been computed
- training will run on Linux, not Windows

## Recommended Baseline

- OS: Ubuntu `22.04` LTS, `x86_64`
- GPU:
  - preferred: `NVIDIA H100 80 GB`
  - good fallback: `NVIDIA A100 80 GB`
  - possible but less safe: `NVIDIA A100 40 GB`
- System RAM:
  - minimum: `192 GB`
  - preferred: `256 GB`
- Storage: `500 GB` fast SSD or NVMe
- CUDA-ready VM image with NVIDIA driver already installed

## Why The RAM Requirement Is High

Your current training config uses `preload_to_ram: true`.

From the local `T=3` logs:

- `3200` train samples at `T=3` used about `28.1 GB`
- `400` val samples at `T=3` used about `3.5 GB`

Scaling that to the production split (`12,000` train, `1,500` val) puts the preload requirement in roughly the `130+ GB` range before OS, cache, Python, and worker overhead.

That is why `192-256 GB` system RAM is the safe range.

## Best GPU Choices

### Best overall

- `NVIDIA H100 80 GB`
  - best throughput
  - most headroom for batch-size stability
  - best choice for minimizing wall-clock risk

### Good fallback

- `NVIDIA A100 80 GB`
  - slower than H100, but still viable
  - safer than `40 GB` for long unattended runs

### Only if cost forces it

- `NVIDIA A100 40 GB`
  - may need retuning if memory is tighter than expected
  - not my first choice for the first production run

## Suggested Providers

### Lambda

- Website: <https://lambda.ai/service/gpu-cloud>
- Hardware reference: <https://docs.lambda.ai/assets/docs/Lambda_Support_Hardware.pdf>

Good choices:

- `1x H100 80 GB`
- `1x A100 80 GB`

### Runpod

- Website: <https://www.runpod.io/>
- H100 PCIe page: <https://www.runpod.io/gpu-models/h100-pcie>

Good choices:

- `1x H100 PCIe 80 GB`
- `1x A100 80 GB`

### AWS

- P5 overview: <https://aws.amazon.com/ec2/instance-types/p5/>
- P4 overview: <https://aws.amazon.com/ec2/instance-types/p4/>

Relevant instance families:

- `p5` for `H100`
- `p4de` for `A100 80 GB`
- `p4d` for `A100 40 GB`

### Google Cloud

- A3: <https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3_machines>
- A2: <https://cloud.google.com/compute/docs/gpus#a2_vms>

Relevant machine families:

- `A3` for `H100`
- `A2 Ultra` / `A2` for `A100`

## Practical Recommendation

If you want the safest single-node choice, use:

- `1x H100 80 GB`
- `192-256 GB RAM`
- `500 GB` SSD
- Ubuntu `22.04`

If you need a lower-cost fallback, use:

- `1x A100 80 GB`
- `192-256 GB RAM`
- `500 GB` SSD

## Provision The VM

Choose a CUDA-ready Ubuntu image if your provider offers one.

After boot, confirm the GPU exists:

```bash
nvidia-smi
```

If `nvidia-smi` fails, stop and fix the driver/image first.

## System Packages

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  git \
  curl \
  wget \
  unzip \
  htop \
  python3 \
  python3-venv \
  python3-dev
```

## Clone The Repo

```bash
git clone <your-repo-url> bpc_fno_v2
cd bpc_fno_v2
```

## Create The Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
```

Install PyTorch using the current official Linux CUDA command from:

- <https://pytorch.org/get-started/locally/>

Then install this repo:

```bash
pip install -e ".[test]"
```

## Verify PyTorch Sees The GPU

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected:

- `True` for CUDA availability
- your rented GPU name printed correctly

## Copy In The Production Inputs

You need:

- the merged production monodomain dataset
- the recomputed normalization file for that dataset

Place them where your config expects them, or override paths in config.

## Recommended Environment Variables

These are worth setting for long runs:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_ORDER=PCI_BUS_ID
```

If you are not using W&B live logging:

```bash
export WANDB_MODE=offline
```

## Phase 1 Training

Use a dedicated checkpoint directory per run.

```bash
python scripts/train_forward.py \
  --config configs/arch_a.yaml \
  --checkpoint-dir checkpoints/run_t3_prod \
  --normalization data/processed/normalization.json
```

Resume after interruption:

```bash
python scripts/train_forward.py \
  --config configs/arch_a.yaml \
  --checkpoint-dir checkpoints/run_t3_prod \
  --normalization data/processed/normalization.json \
  --resume
```

The script now writes:

- `phase1_last.pt` every epoch
- `phase1_best.pt` on improvement
- `phase1_final.pt` at completion

## Phase 2 Training

```bash
python scripts/train_joint.py \
  --config configs/arch_a.yaml \
  --checkpoint-dir checkpoints/run_t3_prod \
  --normalization data/processed/normalization.json \
  --phase1-checkpoint checkpoints/run_t3_prod/phase1_best.pt
```

Resume after interruption:

```bash
python scripts/train_joint.py \
  --config configs/arch_a.yaml \
  --checkpoint-dir checkpoints/run_t3_prod \
  --normalization data/processed/normalization.json \
  --resume
```

The script now writes:

- `phase2_last.pt` every epoch
- `phase2_best.pt` on improvement
- `phase2_final.pt` at completion

## Time Planning

These are planning estimates inferred from your local logs plus the expected gap between a `4060 Ti` and datacenter GPUs. They are not direct benchmarks on rented hardware.

### Full run estimate: Phase 1 + Phase 2

- `H100 80 GB`: about `57-78 hours`
- `A100 80 GB`: about `87-109 hours`
- `A100 40 GB`: about `105-130 hours`

## First-Run Strategy

Before committing to the full run:

1. Run `2-5` epochs of Phase 1.
2. Confirm VRAM headroom with `nvidia-smi`.
3. Confirm `phase1_last.pt` is updating every epoch.
4. Kill and resume once on purpose.
5. Only then start the unattended long run.

## Recommended Websites

- PyTorch install selector: <https://pytorch.org/get-started/locally/>
- Lambda GPU Cloud: <https://lambda.ai/service/gpu-cloud>
- Lambda hardware reference: <https://docs.lambda.ai/assets/docs/Lambda_Support_Hardware.pdf>
- Runpod H100 PCIe: <https://www.runpod.io/gpu-models/h100-pcie>
- AWS P5: <https://aws.amazon.com/ec2/instance-types/p5/>
- AWS P4: <https://aws.amazon.com/ec2/instance-types/p4/>
- Google A3: <https://cloud.google.com/compute/docs/accelerator-optimized-machines#a3_machines>
- Google A2: <https://cloud.google.com/compute/docs/gpus#a2_vms>

## Bottom Line

If you want the least painful path:

- generate data on Linux CPU shards with openCARP
- train on a single `H100 80 GB` node with `192-256 GB RAM`
- use the new `--resume` flow and a dedicated checkpoint directory

## Example Cost Calculation

These are planning estimates, not binding quotes.

Assumptions:

- one GPU node
- uninterrupted single-run billing
- excludes storage, bandwidth, taxes, and any provider-specific fees
- training time ranges come from the current repo's measured local logs, scaled to datacenter GPUs

### AWS example costs

Using current public AWS pricing references:

- `H100 80 GB` via `p5.4xlarge`: `$3.933/hour`
- `A100 80 GB` via `p4de.24xlarge`: `$40.96/hour` for `8` GPUs, or `$5.12/hour` per GPU
- `A100 40 GB` via `p4d.24xlarge`: `$32.77/hour` for `8` GPUs, or `$4.096/hour` per GPU

Estimated total training cost:

- `H100 80 GB` for `57-78h`: `$224.18 - $306.77`
- `A100 80 GB` for `87-109h`: `$445.44 - $558.08`
- `A100 40 GB` for `105-130h`: `$430.08 - $532.48`

### Runpod budget example

Runpod currently lists `H100 PCIe 80 GB` from `$2.39/hour` on Secure Cloud.

At that rate:

- `H100 PCIe 80 GB` for `57-78h`: `$136.23 - $186.42`

### Practical reading of the numbers

- cheapest plausible path: `Runpod H100 PCIe`
- safest first production run: `AWS p5.4xlarge` or an equivalent managed `H100 80 GB` node
- `A100 80 GB` remains viable, but it is usually slower enough that total spend can approach or exceed an H100 run

### Pricing sources

- AWS P5 Capacity Blocks: <https://aws.amazon.com/ec2/capacityblocks/pricing/>
- AWS P4/P4de family page: <https://aws.amazon.com/ec2/instance-types/p4/>
- Runpod H100 PCIe: <https://www.runpod.io/gpu-models/h100-pcie>
