# Cloud CPU Data Generation Install Guide

## Purpose

Use this guide for the production monodomain data-generation run on Linux CPU instances with the `opencarp` backend.

This is the recommended path for the `15,000`-sample dataset.

## Recommended Baseline

- OS: Ubuntu `22.04` or `24.04` LTS, `x86_64`
- Per-shard CPU: `16-32 vCPU`
- Per-shard RAM: `64-128 GB`
- Per-shard storage: `250-500 GB` NVMe SSD
- Network: `>= 5 Gbps` helpful, not mandatory
- openCARP: installed and available as `openCARP` or `carp.pt`
- Python: `3.10` or `3.11`

## Suggested Instance Types

These are good fits for one shard per machine.

### Best default

- AWS EC2 `c7a.8xlarge`
  - CPU family: `4th Gen AMD EPYC (Genoa)`
  - Specs: `32 vCPU`, `64 GiB RAM`
  - Website: <https://aws.amazon.com/ec2/instance-types/c7a/>

- AWS EC2 `c7i.8xlarge`
  - CPU family: `4th Gen Intel Xeon Scalable (Sapphire Rapids)`
  - Specs: `32 vCPU`, `64 GiB RAM`
  - Website: <https://aws.amazon.com/ec2/instance-types/c7i/>

### Lower-cost baseline

- AWS EC2 `c6i.4xlarge`
  - CPU family: `3rd Gen Intel Xeon Scalable (Ice Lake)`
  - Specs: `16 vCPU`, `32 GiB RAM`
  - Website: <https://aws.amazon.com/ec2/instance-types/c6i/>

### Good alternative

- Google Cloud `C3`
  - CPU family: `4th Gen Intel Xeon Scalable (Sapphire Rapids)`
  - Website: <https://cloud.google.com/compute/docs/general-purpose-machines#c3_machine_types>

- Google Cloud `C3D`
  - CPU family: `4th Gen AMD EPYC (Genoa)`
  - Website: <https://cloud.google.com/compute/docs/general-purpose-machines#c3d_machine_types>

## Sizing Notes

- Production generation is sharded, so size for `2,500` samples per machine if you use `6` shards.
- Production files are written with gzip compression, but you should still assume roughly `40-100+ GB` per shard depending on data content and retention.
- `save_vm` must stay `false` for production. Turning it on will materially increase storage use.

## Recommended Websites

- openCARP: <https://opencarp.org/>
- openCARP manual: <https://opencarp.org/manual/opencarp-manual-latest.pdf>
- AWS EC2 C7a: <https://aws.amazon.com/ec2/instance-types/c7a/>
- AWS EC2 C7i: <https://aws.amazon.com/ec2/instance-types/c7i/>
- Google Cloud C3/C3D: <https://cloud.google.com/compute/docs/general-purpose-machines>

## Provision The VM

Choose:

- Ubuntu `22.04` or `24.04`
- x86_64 image
- attached NVMe or fast SSD storage
- SSH access enabled

## System Packages

Run this first after SSH login:

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  git \
  curl \
  wget \
  unzip \
  htop \
  pkg-config \
  cmake \
  ninja-build \
  python3 \
  python3-venv \
  python3-dev \
  openmpi-bin \
  libopenmpi-dev
```

## Install openCARP

Use the current Linux installation instructions from the official openCARP site/manual.

After install, verify one of these works:

```bash
which openCARP || which carp.pt
openCARP -h || carp.pt -h
```

If neither command works, do not continue.

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
pip install -e ".[test]"
```

## Fetch Required Inputs

```bash
python scripts/download_cellml.py
```

You also need a fitted noise model:

- preferred: copy `data/processed/noise_model.json` from your validated local repo
- alternative: fit it on the cloud node if the KCD source data is available

Example:

```bash
mkdir -p data/processed
# scp local-machine:/path/to/bpc_fno_v2/data/processed/noise_model.json data/processed/
```

## Preflight Verification

Run the strict Linux benchmark before the production run:

```bash
python scripts/validate_niederer_linux.py --config configs/benchmark_niederer.yaml
```

Only continue if it passes.

Then run a one-sample smoke generation:

```bash
python scripts/generate_synthetic.py \
  --config configs/data_gen_opencarp_prod.yaml \
  --pipeline monodomain \
  --backend opencarp \
  --sample-count 1 \
  --output-dir data/synthetic_smoke
```

Verify the smoke sample:

```bash
python scripts/verify_data.py --data-dir data/synthetic_smoke --n-samples 1
```

## Production Sharded Command

Run one shard per VM:

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

Change `--shard-id` and `--seed-offset` per machine.

Suggested mapping:

- shard 0: `--shard-id 0 --seed-offset 0`
- shard 1: `--shard-id 1 --seed-offset 1`
- shard 2: `--shard-id 2 --seed-offset 2`
- shard 3: `--shard-id 3 --seed-offset 3`
- shard 4: `--shard-id 4 --seed-offset 4`
- shard 5: `--shard-id 5 --seed-offset 5`

## Post-Generation

On the merged dataset:

```bash
python scripts/verify_data.py --data-dir <merged_dir> --n-samples 50
```

Then recompute normalization on the final monodomain dataset.

Do not start training against the old eikonal normalization stats.

## Practical Recommendation

For this project, the safest production path is:

- `6 x c7a.8xlarge` or `6 x c7i.8xlarge`
- one shard per machine
- Ubuntu `22.04/24.04`
- openCARP installed directly on Linux

Do not plan the main data-generation run around GPU acceleration. openCARP has some GPU-related infrastructure, but the low-risk production plan here is still CPU + sharding.

## Example Cost Calculation

These are planning estimates, not binding quotes.

Assumptions:

- region: AWS `us-east-2` style Linux on-demand example pricing
- fleet shape: `6` CPU instances
- run duration: `36-48` hours
- excludes storage, bandwidth, public IPv4 charges, taxes, and any failed/retried shards

### Lower-cost baseline example

Using AWS `c6i.4xlarge` at `$0.904/hour` from AWS's public GameLift pricing example page:

- `1` node for `48h`: `48 x 0.904 = $43.39`
- `6` nodes for `36h`: `6 x 36 x 0.904 = $195.26`
- `6` nodes for `48h`: `6 x 48 x 0.904 = $260.35`

That means your earlier target of roughly `$196` corresponds almost exactly to:

- `6 x c6i.4xlarge` for about `36 hours`

### Important note

The recommended `c7a.8xlarge` / `c7i.8xlarge` machines are stronger than `c6i.4xlarge`, so their real total cost will usually be higher than the `c6i` example above. Use the example as a budget floor, not a ceiling.

### Pricing sources

- AWS C6i example with public hourly rate: <https://aws.amazon.com/gamelift/pricing/>
- AWS C7a family page: <https://aws.amazon.com/ec2/instance-types/c7a/>
- AWS C7i family page: <https://aws.amazon.com/ec2/instance-types/c7i/>
