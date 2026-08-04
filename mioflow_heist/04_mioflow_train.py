"""
Step 4: Train the MIOFlow ODE on the 2D HEIST latent across bp542's time_bins,
then integrate 500 trajectories -> ptrajs.npy (500, 100, 2).

Run: .venv/bin/python mioflow_heist/04_mioflow_train.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import scanpy as sc
from torchdiffeq import odeint

from src.mioflow import ODEFunc, TimeSeriesDataset, train_mioflow

SUFFIX = os.environ.get("HEIST_SUFFIX", "")
H5AD = os.environ.get("HEIST_H5AD", paths.h5ad("axolotl_heist.h5ad"))
PTRAJS = paths.traj(f"ptrajs{SUFFIX}.npy")
MODEL_OUT = paths.ckpt(f"mioflow_ode{SUFFIX}.pth")
N_TRAJ = 500
N_BINS = 100
SEED = 0

torch.manual_seed(SEED); np.random.seed(SEED)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[step4] device={device}")

adata = sc.read_h5ad(H5AD)
Z = np.asarray(adata.obsm['X_gaga_2stage_heist'], dtype=np.float32)   # (N,2)
tb = np.asarray(adata.obs['time_bin'], dtype=np.float32)
times = np.sort(np.unique(tb))
print(f"[step4] latent {Z.shape} | time bins {times} | counts {[int((tb==t).sum()) for t in times]}")

time_series_data = [(Z[tb == t], float(t)) for t in times]
dataset = TimeSeriesDataset(time_series_data)

model = ODEFunc(input_dim=2, hidden_dim=64).to(device)
train_mioflow(
    model=model, dataset=dataset, num_epochs=500, mode='local', batch_size=256,
    learning_rate=1e-2, lambda_ot=1.0, lambda_density=1e-4, lambda_energy=0.01,
    energy_time_steps=10, device=device,
)

# --- Integrate trajectories from 500 cells sampled at t=0 ---
model.eval()
X0 = dataset.get_initial_condition(0)          # cells at earliest time
idx = torch.randperm(X0.size(0))[:N_TRAJ]
X0s = X0[idx].to(device)
t_bins = torch.linspace(float(times.min()), float(times.max()), N_BINS, device=device)
with torch.no_grad():
    traj = odeint(model, X0s, t_bins)          # (N_BINS, N_TRAJ, 2)
ptrajs = traj.permute(1, 0, 2).cpu().numpy()   # (N_TRAJ, N_BINS, 2)
print(f"[step4] ptrajs {ptrajs.shape}")
assert ptrajs.shape == (N_TRAJ, N_BINS, 2)
assert np.isfinite(ptrajs).all(), "non-finite ptrajs (ODE diverged?)"

np.save(PTRAJS, ptrajs)
torch.save(model.state_dict(), MODEL_OUT)
print(f"[step4] saved {PTRAJS} and {MODEL_OUT}")
