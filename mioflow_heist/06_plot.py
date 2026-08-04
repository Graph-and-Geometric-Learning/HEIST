"""
Step 6: Render Fig 5 (HEIST single-embedding version) -> fig5_heist.png

Panels:
  Row 1: HEIST embedding (X_gaga_2stage_heist) colored by (a) cell type, (b) pseudotime, (c) NCAN:SDC3 signaling
  Row 2: Trajectories over the HEIST embedding, colored by NCAN:SDC3 signaling (from strajs)
  Row 3: Spatial tissue at 2/15/60 DPI (full dataset) colored by LR_feats[:,72]; + signaling vs pseudotime
  Row 4: Gene-expression trends over pseudotime (VIM, SLC1A3, STMN4, S100A10) from gtrajs

Run: .venv/bin/python mioflow_heist/06_plot.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.collections import LineCollection

SUFFIX = os.environ.get("HEIST_SUFFIX", "")
HEIST_H5AD = os.environ.get("HEIST_H5AD", paths.h5ad("axolotl_heist.h5ad"))
FULL_H5AD = paths.AXOLOTL_FULL
OUT_PNG = paths.fig(f"fig5_heist{SUFFIX}.png")

LR = 72                                   # NCAN:SDC3
GOIS = [6009, 11395, 8976, 4243]
GNAMES = ['VIM', 'SLC1A3', 'STMN4', 'S100A10']
EMB_KEY = 'X_gaga_2stage_heist'
CT_KEY = 'Annotation_grouped'
SQUARE_TRAJ = os.environ.get("SQUARE_TRAJ", "0") == "1"   # force the trajectory panel to a square box

# ---------- helpers (from bp542 plot.ipynb) ----------
def llcenter(X):
    return X - np.min(X, axis=0)

def notick(ax):
    ax.set_xticks([]); ax.set_yticks([])

def subto(data, val, key):
    return data[data.obs[key] == val]

def oneof(data, i=0, key='Batch'):
    lock = np.unique(data.obs[key])[i]
    return data[data.obs[key] == lock]

def gradline(ax, trajs, c, cmap='magma', alpha=1, lw=1):
    """trajs: N x T x 2 ; c: N x T signaling used to color segments."""
    x0s = trajs[:, 0:-1, 0].reshape(-1, 1); x1s = trajs[:, 1:, 0].reshape(-1, 1)
    y0s = trajs[:, 0:-1, 1].reshape(-1, 1); y1s = trajs[:, 1:, 1].reshape(-1, 1)
    pts0 = np.concatenate((x0s, y0s), axis=1); pts1 = np.concatenate((x1s, y1s), axis=1)
    pts = np.stack((pts0, pts1), axis=1)
    colors = ((c[..., 1:] + c[..., 0:-1]) / 2).flatten()
    lc = LineCollection(pts, array=colors, cmap=cmap, alpha=alpha, lw=lw, capstyle='round')
    ax.add_collection(lc)
    return lc

# ---------- load ----------
print("[step6] loading data...")
data = sc.read_h5ad(HEIST_H5AD)
data.obs_names_make_unique()
ptrajs = np.load(paths.traj(f"ptrajs{SUFFIX}.npy"))[:100]
gtrajs = np.load(paths.traj(f"gtrajs{SUFFIX}.npy"))[:100]
strajs = np.load(paths.traj(f"strajs{SUFFIX}.npy"))[:100]
print(f"[step6] ptrajs{ptrajs.shape} gtrajs{gtrajs.shape} strajs{strajs.shape}")

emb = np.asarray(data.obsm[EMB_KEY])
lr_static = np.asarray(data.obsm['LR_feats'])[:, LR]
tb = np.asarray(data.obs['time_bin'], dtype=float)
tx = np.linspace(tb.min(), tb.max(), ptrajs.shape[1])
cts = list(np.unique(data.obs[CT_KEY]))

smean = strajs[..., LR].mean(0); sdev = strajs[..., LR].std(0)

# 3-row layout on a 20-col grid: row1 = 4 panels (spans of 5), row2 = 5 panels (spans of 4),
# row3 = 4 panels (spans of 5).
fig = plt.figure(figsize=(28, 18))
gs = fig.add_gridspec(3, 20, hspace=0.32, wspace=2.2)
C4 = [(0, 5), (5, 10), (10, 15), (15, 20)]           # 4-panel rows
C5 = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20)]   # 5-panel row
def cell(r, span): return fig.add_subplot(gs[r, span[0]:span[1]])

# ---------- Row 1: embedding (PHATE) scatters + trajectories ----------
ax = cell(0, C4[0])
for ct in cts:
    m = (data.obs[CT_KEY] == ct).to_numpy()
    ax.scatter(*emb[m].T, s=3, label=ct)
ax.legend(markerscale=4, title='Cell Type', fontsize=8)
ax.set_title('HEIST Embedding — Cell Type'); notick(ax)

ax = cell(0, C4[1])
sctp = ax.scatter(*emb.T, s=3, c=tb, cmap='coolwarm', vmin=0, vmax=1)
fig.colorbar(sctp, ax=ax, fraction=0.046); ax.set_title('HEIST Embedding — Pseudotime'); notick(ax)

ax = cell(0, C4[2])
scls = ax.scatter(*emb.T, s=3, c=lr_static, cmap='magma', vmin=0)
fig.colorbar(scls, ax=ax, fraction=0.046); ax.set_title('HEIST Embedding — NCAN:SDC3'); notick(ax)

ax = cell(0, C4[3])
ax.scatter(*emb.T, s=3, c='lightgrey')
lc = gradline(ax, ptrajs[:, :, 0:2], strajs[..., LR], cmap='magma', lw=2)
fig.colorbar(lc, ax=ax, fraction=0.046)
ax.set_title('Trajectories (NCAN:SDC3 signaling)')
ax.set_xlabel('HEIST-latent 1'); ax.set_ylabel('HEIST-latent 2')
if SQUARE_TRAJ:
    ax.set_box_aspect(1)

# ---------- Row 2: spatial DPI panels + both NCAN:SDC3 curves ----------
try:
    alldata = sc.read_h5ad(FULL_H5AD, backed='r')
    obs = alldata.obs
    sp_all = np.asarray(alldata.obsm['spatial'])
    lr_all = np.asarray(alldata.obsm['LR_feats'])[:, LR]
    inj = (obs['inj_uninj'].astype(str) == 'inj').to_numpy()
    dpi = obs['dpi'].astype(str).to_numpy()
    batch = obs['Batch'].astype(str).to_numpy()
    vmax_sp = float(np.max(np.asarray(data.obsm['LR_feats'])[:, LR]))
    sp_plot = None
    for i, t in enumerate([2, 15, 60]):
        ax = cell(1, C5[i])
        m = inj & (dpi == str(t))
        if m.sum() == 0:
            ax.set_title(f'{t} DPI (no cells)'); notick(ax); continue
        m = m & (batch == np.unique(batch[m])[0])
        sp_plot = ax.scatter(*(llcenter(sp_all[m]) / 2).T, s=4, c=lr_all[m], cmap='magma', vmin=0, vmax=vmax_sp)
        ax.set_title(f'{t} DPI'); ax.set_aspect('equal'); notick(ax)
    if sp_plot is not None:
        fig.colorbar(sp_plot, ax=ax, fraction=0.046)
except Exception as e:
    print(f"[step6] WARNING: spatial DPI panels skipped ({type(e).__name__}: {e})")

ax = cell(1, C5[3])
ax.plot(tx, smean, lw=3, c='k'); ax.fill_between(tx, smean - sdev, smean + sdev, color='grey', alpha=0.3)
ax.set_xlabel('Pseudotime'); ax.set_ylabel('Signaling Level'); ax.set_title('NCAN:SDC3 Signaling')

ax = cell(1, C5[4])
ax.plot(tx, smean, lw=3, c='k'); ax.fill_between(tx, smean - sdev, smean + sdev, color='grey', alpha=0.3)
ax.set_xlabel('Pseudotime'); ax.set_ylabel('Signaling'); ax.set_title('NCAN:SDC3 over Pseudotime')

# ---------- Row 3: gene trends (unchanged) ----------
for j, (g, nm) in enumerate(zip(GOIS, GNAMES)):
    ax = cell(2, C4[j])
    gmean = gtrajs[..., g].mean(0); gstd = gtrajs[..., g].std(0)
    ax.plot(tx, gmean, c='k'); ax.fill_between(tx, gmean - gstd, gmean + gstd, color='grey', alpha=0.3)
    ax.set_title(nm); ax.set_xlabel('Pseudotime')
    if j == 0:
        ax.set_ylabel('Expression Level')

fig.suptitle('Fig 5 (HEIST single-embedding) — Axolotl regeneration trajectories', fontsize=16, y=0.995)
fig.savefig(OUT_PNG, dpi=130, bbox_inches='tight')
print(f"[step6] saved {OUT_PNG}")
