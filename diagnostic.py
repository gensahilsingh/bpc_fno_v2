import torch, glob, h5py, numpy as np, json
from omegaconf import OmegaConf
from bpc_fno.models.bpc_fno_a import BPC_FNO_A

config = OmegaConf.load('configs/arch_a.yaml')
model = BPC_FNO_A(config)
ckpt = torch.load('checkpoints/phase1_best.pt', map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

with open('data/processed/normalization.json') as f:
    stats = json.load(f)

J_mean = torch.FloatTensor(stats['J_i_mean']).view(1,3,1,1,1)
J_std  = torch.FloatTensor(stats['J_i_std']).view(1,3,1,1,1)
B_mean_r = torch.FloatTensor(stats['B_mean']).view(1,1,3)
B_std_r  = torch.FloatTensor(stats['B_std']).view(1,1,3)

files = sorted(glob.glob('data/synthetic/sample_*.h5'))
fp = [f for i,f in enumerate(files) if i%10==9][0]

with h5py.File(fp) as f:
    J_raw = f['J_i'][:]
    B_raw = f['B_mig'][:]
    sdf   = f['geometry/sdf'][:]
    fib   = f['geometry/fiber'][:]

t_idx = int(np.argmax(np.abs(J_raw).sum(axis=(1,2,3,4))))
J = torch.FloatTensor(J_raw[t_idx]).permute(3,0,1,2).unsqueeze(0)
B_t = torch.FloatTensor(B_raw[t_idx]).reshape(1,16,3)
sdf_t = torch.FloatTensor(sdf).unsqueeze(0).unsqueeze(0)
fib_t = torch.FloatTensor(fib).permute(3,0,1,2).unsqueeze(0)
geo   = torch.cat([sdf_t, fib_t], dim=1)
geo_norm = geo.clone()
geo_norm[:,0:1] = geo[:,0:1].clamp(-5,5) / 5.0
J_norm = (J - J_mean) / (J_std + 1e-8)
B_norm = ((B_t - B_mean_r) / (B_std_r + 1e-8)).reshape(1,48)

with torch.no_grad():
    B_pred = model.forward_only(J_norm, geo_norm)

print('--- Diagnosis ---')
print('J_norm range: [{:.3f}, {:.3f}]'.format(J_norm.min().item(), J_norm.max().item()))
print('B_norm range: [{:.3f}, {:.3f}]'.format(B_norm.min().item(), B_norm.max().item()))
print('B_pred range: [{:.3f}, {:.3f}]'.format(B_pred.min().item(), B_pred.max().item()))
print('B_pred mean:  {:.6f}'.format(B_pred.mean().item()))
print('B_norm mean:  {:.6f}'.format(B_norm.mean().item()))
print('B_pred std:   {:.6f}'.format(B_pred.std().item()))
print('B_norm std:   {:.6f}'.format(B_norm.std().item()))
print()
print('First 6 B_pred:', B_pred[0,:6].tolist())
print('First 6 B_norm:', B_norm[0,:6].tolist())
print()
print('--- Normalization sanity ---')
B_phys = B_raw[t_idx]
print('Raw B range: [{:.3e}, {:.3e}] Tesla'.format(float(B_phys.min()), float(B_phys.max())))
print('B_std used:', stats['B_std'])
B_manual_norm = (B_phys - np.array(stats['B_mean'])) / np.array(stats['B_std'])
print('Manual B_norm range: [{:.3f}, {:.3f}]'.format(float(B_manual_norm.min()), float(B_manual_norm.max())))
print()
J_phys = J_raw[t_idx]
print('Raw J range: [{:.3e}, {:.3e}] uA/cm2'.format(float(J_phys.min()), float(J_phys.max())))
print('J_std used:', stats['J_i_std'])
J_manual_norm = (J_phys - np.array(stats['J_i_mean'])) / np.array(stats['J_i_std'])
print('Manual J_norm range: [{:.3f}, {:.3f}]'.format(float(J_manual_norm.min()), float(J_manual_norm.max())))