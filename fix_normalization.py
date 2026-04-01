import h5py, glob, numpy as np, json
from tqdm import tqdm

files = sorted(glob.glob('data/synthetic/sample_*.h5'))
train_files = [f for i,f in enumerate(files) if i % 10 in range(8)]
print(f'Computing stats from {len(train_files)} training files')

J_vals = []
B_vals = []

for fp in tqdm(train_files[:200]):  # use 200 samples for speed
    with h5py.File(fp) as f:
        J = f['J_i'][:]   # (T, N, N, N, 3)
        B = f['B_mig'][:] # (T, Ns, 3)

    # For J: only include voxels where magnitude > threshold
    # This excludes resting tissue from the stats
    J_mag = np.abs(J).sum(axis=-1)  # (T, N, N, N)
    threshold = np.percentile(J_mag[J_mag > 0], 10)  # bottom 10% of active voxels
    active_mask = J_mag > threshold  # (T, N, N, N)

    for c in range(3):
        J_c = J[..., c]  # (T, N, N, N)
        active_vals = J_c[active_mask]
        if len(active_vals) > 0:
            J_vals.append(active_vals)

    # For B: use all values (B is already spatially compact)
    B_vals.append(B.reshape(-1, 3))

J_all = np.concatenate(J_vals, axis=0)
B_all = np.concatenate(B_vals, axis=0)

stats = {
    'J_i_mean': [float(np.mean(J_all))] * 3,
    'J_i_std':  [float(np.std(J_all))]  * 3,
    'B_mean':   [float(np.mean(B_all[:,c])) for c in range(3)],
    'B_std':    [float(np.std(B_all[:,c]))  for c in range(3)],
}

print('New J_i_std:', stats['J_i_std'])
print('New B_std:',   stats['B_std'])
print('Expected J_norm range with new stats:',
      (-1.5 - stats['J_i_mean'][0]) / stats['J_i_std'][0],
      (1.5  - stats['J_i_mean'][0]) / stats['J_i_std'][0])

with open('data/processed/normalization.json', 'w') as f:
    json.dump(stats, f, indent=2)
print('Saved to data/processed/normalization.json')