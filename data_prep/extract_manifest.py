"""Record which raw cells went into each old pretraining chunk.

The v1 chunks (data/pretraining/{10x,vizgen,sea}_preprocessed/*.pt) have the shared-gene-graph bug,
so their per-cell expression is unusable, but G_cell.cell_id still identifies every cell. For 10x
it is the row index into the sample's cells.parquet (verified: coordinates match exactly), so
rebuilding from this manifest reproduces the v1 chunk membership without the lost tiling script.

Output: one parquet with columns source, sample, chunk, cell_id, cell_type.
"""
import argparse
import os
import re
from glob import glob

import pandas as pd
import torch
from tqdm import tqdm


def sample_of(stem, source):
    # 10x: <sample>_outs_<quadtree digits>; vizgen/sea: <sample>_<digits>. Two heart/brain files are
    # untiled and end in "_outs_" with no digits.
    if source == '10x':
        return re.sub(r'_outs_\d*$', '', stem)
    return re.sub(r'_\d+$', '', stem)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='data/pretraining')
    p.add_argument('--out', default='data/pretraining_v2/manifest.parquet')
    args = p.parse_args()

    frames = []
    for source in ['10x', 'vizgen', 'sea']:
        files = sorted(glob(os.path.join(args.root, f'{source}_preprocessed', '*.pt')))
        for f in tqdm(files, desc=source):
            stem = os.path.basename(f)[:-3]
            g = torch.load(f, weights_only=False)[0]
            types = [str(g.cell_types[i]) for i in g.cell_type.tolist()]
            frames.append(pd.DataFrame({
                'source': source,
                'sample': sample_of(stem, source),
                'chunk': stem,
                'cell_id': [str(c) for c in g.cell_id],
                'cell_type': types,
            }))
    df = pd.concat(frames, ignore_index=True)
    for c in ['source', 'sample', 'chunk', 'cell_type']:
        df[c] = df[c].astype('category')
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_parquet(args.out)
    print(df.groupby('source', observed=True).agg(chunks=('chunk', 'nunique'), cells=('cell_id', 'size')))


if __name__ == '__main__':
    main()
