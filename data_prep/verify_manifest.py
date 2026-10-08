"""Check that manifest cell ids point at the SAME cells in the downloaded raw data as in v1.

For every v1 sample, map each archived v1 chunk's cell_id through the raw file the v2 loader uses and
compare the chunk-relative coordinates with the v1 center_coord. A wrong release (e.g. 1.0.2 vs a
re-segmented 1.3.0) can give ids that are all in range but point at different cells, so an
in-range check is not enough. Prints one line per sample; any max error > 0.01 is a mismatch.
"""
import os
import sys
from glob import glob

import numpy as np
import pandas as pd
import torch

RAW = os.path.expanduser('~/scratch_pi_sk2433/hm638/HEIST_raw')
ARCHIVE = sys.argv[1] if len(sys.argv) > 1 else 'archive/pretraining'


def raw_xy(source, sample):
    if source == '10x':
        c = pd.read_parquet(os.path.join(RAW, 'xenium', sample, 'cells.parquet'),
                            columns=['x_centroid', 'y_centroid'])
        return c.to_numpy(), None
    meta = pd.read_csv(os.path.join(RAW, 'vizgen', sample, 'cell_metadata.csv'), index_col=0,
                       usecols=[0, 3, 4])
    return meta.to_numpy(), meta.index


bad = []
for source, pat in [('10x', '_outs_'), ('vizgen', '_')]:
    files = glob(os.path.join(ARCHIVE, f'{source}_preprocessed', '*.pt'))
    samples = sorted({os.path.basename(f).rsplit(pat, 1)[0] for f in files})
    for s in samples:
        xy, entity = raw_xy(source, s)
        errs, out_of_range = [], 0
        for f in sorted(glob(os.path.join(ARCHIVE, f'{source}_preprocessed', f'{s}{pat}*.pt')))[:4]:
            g = torch.load(f, weights_only=False)[0]
            ids = np.array(g.cell_id, dtype=np.int64)
            if entity is not None and ids.max() >= len(xy):      # v1 used EntityIDs for this sample
                pos = pd.Index(entity).get_indexer(ids)
                out_of_range += int((pos < 0).sum()); ids = pos
            out_of_range += int((ids >= len(xy)).sum())
            if out_of_range:
                break
            p = xy[ids]
            errs.append(float(np.abs((p - p.min(0)) - g.center_coord.numpy()).max()))
        ok = not out_of_range and max(errs) < 0.01
        if not ok:
            bad.append(s)
        print(f'{"OK " if ok else "BAD"} {source:6s} {s:65s} raw cells {len(xy):8d}  '
              f'max coord err {max(errs) if errs else float("nan"):10.3f}  ids out of range {out_of_range}',
              flush=True)
print('MISMATCHED:', bad)
