"""
Central path config for the mioflow_heist pipeline. Import this in every step script
instead of hard-coding output locations, so the folder stays organized.

Layout:
  mioflow_heist/
    <NN>_*.py            step scripts (entry points)
    heist_preprocess.py, aux_model.py, paths.py, run_sweep.sh, README.md
    src/                 vendored MIOFlow libs (gaga.py, mioflow.py)
    ortho_model/         first-commit HEIST model (for the ortho checkpoint)
    data/                preprocessed PyG graphs (*.pt)
    outputs/
      h5ad/              axolotl_heist*.h5ad
      trajectories/      *trajs*.npy, phate_gene.npy
      checkpoints/       *.pth
      figures/           fig5_*.png, diag_*.png
    plot.ipynb
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
OUTPUTS = os.path.join(ROOT, "outputs")
H5AD_DIR = os.path.join(OUTPUTS, "h5ad")
TRAJ_DIR = os.path.join(OUTPUTS, "trajectories")
CKPT_DIR = os.path.join(OUTPUTS, "checkpoints")
FIG_DIR = os.path.join(OUTPUTS, "figures")

for _d in (DATA_DIR, H5AD_DIR, TRAJ_DIR, CKPT_DIR, FIG_DIR):
    os.makedirs(_d, exist_ok=True)

# External reference data (read-only, bp542)
AXOLOTL_RAN = "/nfs/roberts/project/pi_sk2433/bp542/MIOFlow_lite/notebooks/axolotl_ran.h5ad"
AXOLOTL_FULL = "/nfs/roberts/project/pi_sk2433/bp542/Axolotl_Spatial/Axolotl_processed.h5ad"


def h5ad(name):   return os.path.join(H5AD_DIR, name)
def traj(name):   return os.path.join(TRAJ_DIR, name)
def ckpt(name):   return os.path.join(CKPT_DIR, name)
def fig(name):    return os.path.join(FIG_DIR, name)
def data(name):   return os.path.join(DATA_DIR, name)
