# openCARP Cloud Runbook

## Purpose

Generate the production `15,000`-sample monodomain dataset on Linux using the `opencarp` backend with deterministic sharding.

## Expected Environment

- Linux
- `openCARP` or `carp.pt` on `PATH`
- Python environment with this repository installed
- fitted noise model available at `data/processed/noise_model.json`

## Preflight

1. Verify the binary:

```bash
which openCARP || which carp.pt
```

2. Run the strict benchmark:

```bash
python scripts/validate_niederer_linux.py --config configs/benchmark_niederer.yaml
```

3. Only continue if the benchmark passes.

## Production Command Pattern

Each cloud instance should run one shard:

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

Change `--shard-id` and `--seed-offset` per instance.

## Recommended Shard Assignment

- instance 0: `--shard-id 0 --num-shards 6 --seed-offset 0`
- instance 1: `--shard-id 1 --num-shards 6 --seed-offset 1`
- instance 2: `--shard-id 2 --num-shards 6 --seed-offset 2`
- instance 3: `--shard-id 3 --num-shards 6 --seed-offset 3`
- instance 4: `--shard-id 4 --num-shards 6 --seed-offset 4`
- instance 5: `--shard-id 5 --num-shards 6 --seed-offset 5`

## Output

Each shard writes:

- `sample_*.h5`
- `MANIFEST.json`

into its configured output directory.

## Post-Generation

1. Merge shard outputs into a single production data directory.
2. Run:

```bash
python scripts/verify_data.py --data-dir <merged_dir> --n-samples 50
```

3. Recompute normalization on the final merged monodomain dataset.
4. Only then start ML training.
