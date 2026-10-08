"""Build the global gene vocabulary and per-sample gene panels for the v2 pretraining data.

gene_panels.json : {"<source>/<sample>": [genes measured, as named in the raw data]}
gene_vocab.json  : {"<UPPER-CASE SYMBOL>": index}, the sorted union of all panels. preprocess()
                   uses it to store G_cell.gene_ids so gene-graph node i maps to a global gene.

Symbols are upper-cased only; no alias/Ensembl harmonization (Vizgen and SEA ship symbols only).
Must run before build_chunks.py, and must be rerun (and chunks rebuilt) if samples are added.
"""
import argparse
import json
import os
from glob import glob

import h5py
import pandas as pd
import scanpy as sc

RAW = os.path.expanduser('~/scratch_pi_sk2433/hm638/HEIST_raw')


def panels():
    out = {}
    for d in sorted(glob(os.path.join(RAW, 'xenium', '*'))):
        a = sc.read_10x_h5(os.path.join(d, 'cell_feature_matrix.h5'))  # gene expression only
        out[f'10x/{os.path.basename(d)}'] = a.var_names.tolist()
    for d in sorted(glob(os.path.join(RAW, 'vizgen', '*'))):
        cols = pd.read_csv(os.path.join(d, 'cell_by_gene.csv'), nrows=0).columns[1:]
        out[f'vizgen/{os.path.basename(d)}'] = [c for c in cols if not c.upper().startswith('BLANK')]
    with h5py.File(os.path.join(RAW, 'sea', 'SEAAD_MTG_MERFISH.2024-12-11.h5ad'), 'r') as f:
        v = f['var']
        out['sea/SEAAD_MTG_MERFISH'] = [g for g in v[v.attrs['_index']].asstr()[:].tolist()
                                        if not g.upper().startswith('BLANK')]  # 40 control probes
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_root', default='data/pretraining_v2')
    args = p.parse_args()
    pan = panels()
    vocab = {g: i for i, g in enumerate(sorted({g.upper() for genes in pan.values() for g in genes}))}
    os.makedirs(args.out_root, exist_ok=True)
    with open(os.path.join(args.out_root, 'gene_panels.json'), 'w') as f:
        json.dump(pan, f, indent=1)
    with open(os.path.join(args.out_root, 'gene_vocab.json'), 'w') as f:
        json.dump(vocab, f, indent=1)
    by_src = {}
    for k, genes in pan.items():
        by_src.setdefault(k.split('/')[0], set()).update(g.upper() for g in genes)
    print(f'{len(pan)} samples, vocab size {len(vocab)}',
          {s: len(g) for s, g in by_src.items()})


if __name__ == '__main__':
    main()
