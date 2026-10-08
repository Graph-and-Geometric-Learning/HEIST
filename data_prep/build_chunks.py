"""Rebuild the pretraining chunks from raw data + the v1 cell manifest, using the fixed preprocess().

One invocation handles one sample (SLURM array index -> sample), writing every chunk of that sample
to <out_root>/<source>_preprocessed/<chunk>.pt. Existing outputs are skipped (preprocess() returns
the cached file), so a failed array task can simply be resubmitted.

Cell identity per source (see extract_manifest.py):
  10x    cell_id = row index into <sample>/cells.parquet (== row order of cell_feature_matrix.h5)
  sea    NOT from the manifest: the v1 366k-cell SEA-AD release is gone and its ids do not map to
         the 2024-12-11 release. v2 uses all cells of that release, one chunk per Section
         (<donor>_<i>, sections sorted), cell_type = Subclass.
  vizgen cell_id = row index into cell_metadata.csv (verified; cell_by_gene.csv is reordered to match)
"""
import argparse
import os
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.preprocess import preprocess, spatial_chunks  # noqa: E402

RAW = os.path.expanduser('~/scratch_pi_sk2433/hm638/HEIST_raw')


def load_10x(sample):
    d = os.path.join(RAW, 'xenium', sample)
    a = sc.read_10x_h5(os.path.join(d, 'cell_feature_matrix.h5'))
    cells = pd.read_parquet(os.path.join(d, 'cells.parquet'))
    ids = cells['cell_id'].astype(str).to_numpy()
    if not np.array_equal(a.obs_names.to_numpy().astype(str), ids):
        raise ValueError(f'{sample}: matrix barcodes and cells.parquet rows are not in the same order')
    a.obsm['spatial'] = cells[['x_centroid', 'y_centroid']].to_numpy()
    a.obs_names = np.arange(a.n_obs).astype(str)  # manifest ids are row indices
    return a


def load_vizgen(sample):
    d = os.path.join(RAW, 'vizgen', sample)
    X = pd.read_csv(os.path.join(d, 'cell_by_gene.csv'), index_col=0)
    X = X.loc[:, ~X.columns.str.upper().str.startswith('BLANK')]
    # Manifest ids are row indices into cell_metadata.csv (verified: coordinates match exactly).
    # cell_by_gene.csv holds the same EntityIDs in a DIFFERENT order, so align X to metadata order.
    meta = pd.read_csv(os.path.join(d, 'cell_metadata.csv'), index_col=0)
    X = X.loc[meta.index]
    a = ad.AnnData(X.to_numpy(dtype=np.float32), var=pd.DataFrame(index=X.columns))
    a.obsm['spatial'] = meta[['center_x', 'center_y']].to_numpy()
    a.obs_names = np.arange(a.n_obs).astype(str)
    a.uns['entity_ids'] = meta.index.to_numpy()   # see to_row_index()
    return a


def to_row_index(ids, full):
    """Manifest cell ids -> row positions in `full`.

    v1 stored row indices for every sample EXCEPT the two HumanUterineCancerPatient2-*Costain Vizgen
    samples, which stored Vizgen EntityIDs (verified with data_prep/verify_manifest.py: all present,
    coordinates match exactly). Row indices are < n_obs; EntityIDs are ~1e16, so the range decides.
    """
    ids = np.asarray(ids, dtype=np.int64)
    if ids.max() < full.n_obs:
        return ids
    pos = pd.Index(full.uns['entity_ids']).get_indexer(ids)
    if (pos < 0).any():
        raise ValueError(f'{(pos < 0).sum()} manifest EntityIDs are missing from the raw metadata')
    return pos


SEA_H5AD = os.path.join(RAW, 'sea', 'SEAAD_MTG_MERFISH.2024-12-11.h5ad')


def sea_manifest():
    """Build the SEA part of the manifest from the 2024 release's obs (read via h5py: fast)."""
    import h5py
    with h5py.File(SEA_H5AD, 'r') as f:
        def col(n):
            g = f['obs'][n]
            if isinstance(g, h5py.Group):
                return np.asarray(g['categories'].asstr()[:])[g['codes'][:]]
            return g.asstr()[:]
        donor, section, sub = col('Donor ID'), col('Section'), col('Subclass')
    df = pd.DataFrame({'sample': donor, 'section': section, 'cell_type': sub,
                       'cell_id': np.arange(len(donor)).astype(str)})
    df = df[df.groupby('section').cell_id.transform('size') > 0]
    sec_idx = {s: i for d, g in df.groupby('sample') for i, s in enumerate(sorted(g.section.unique()))}
    df['chunk'] = df['sample'] + '_' + df.section.map(sec_idx).astype(str)
    df['source'] = 'sea'
    return df


def load_sea(sample):
    a = ad.read_h5ad(SEA_H5AD, backed='r')
    a.obs_names = np.arange(a.n_obs).astype(str)  # sea_manifest ids are row indices
    a.obsm['spatial'] = np.asarray(a.obsm['spatial'])
    return a  # BLANK-* probes are dropped in main(): a backed AnnData cannot be sliced twice


LOADERS = {'10x': load_10x, 'vizgen': load_vizgen, 'sea': load_sea}

MIN_COUNTS = 10       # cells with fewer raw transcripts carry ~no signal (~2.7% of 10x/Vizgen cells)
MIN_KEPT_COUNTS = 1   # ...and not ALL outside the <=200 genes kept (see preprocess; 2 would drop 11%)
MIN_CLUSTER_SIZE = 50 # Leiden clusters smaller than this are merged into a neighbour (one GRN each)
# Spatial chunking (partition, MAX_CHUNK, SPLIT_ABOVE) lives in utils/preprocess.py, shared with inference.


def new_sample_manifest(source, sample, full):
    """Chunks for a sample with no v1 manifest (e.g. Xenium Prime 5K): v1 partition() rules."""
    xy = np.asarray(full.obsm['spatial'])
    labels = spatial_chunks(xy)
    sep = '_outs_' if source == '10x' else '_'
    return pd.DataFrame({'source': source, 'sample': sample, 'chunk': [f'{sample}{sep}{l}' for l in labels],
                         'cell_id': np.arange(len(xy)).astype(str), 'cell_type': ''})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True, choices=list(LOADERS))
    p.add_argument('--index', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)),
                   help='which sample (sorted) of this source to build')
    p.add_argument('--manifest', default='data/pretraining_v2/manifest.parquet')
    p.add_argument('--out_root', default='data/pretraining_v2')
    p.add_argument('--vocab', default='data/pretraining_v2/gene_vocab.json',
                   help='from build_gene_vocab.py; stored per chunk as G_cell.gene_ids')
    p.add_argument('--only_chunk', default=None, help='build just this chunk (for timing)')
    args = p.parse_args()

    if args.source == 'sea':
        man = sea_manifest()
    else:
        man = pd.read_parquet(args.manifest)
        man = man[man.source == args.source]
    raw_dir = {'10x': 'xenium', 'vizgen': 'vizgen', 'sea': None}[args.source]
    raw_samples = set(os.listdir(os.path.join(RAW, raw_dir))) if raw_dir else set()
    samples = sorted(set(man['sample'].unique()) | raw_samples)
    sample = samples[args.index]
    man = man[man['sample'] == sample]
    full = LOADERS[args.source](sample)
    if man.empty:
        man = new_sample_manifest(args.source, sample, full)
    out_dir = os.path.join(args.out_root, f'{args.source}_preprocessed')
    print(f'[{args.source} {args.index}/{len(samples)}] {sample}: {man.chunk.nunique()} chunks', flush=True)

    import json
    with open(args.vocab) as f:
        vocab = json.load(f)
    for chunk, rows in man.groupby('chunk', observed=True):
        if args.only_chunk and chunk != args.only_chunk:
            continue
        if os.path.exists(os.path.join(out_dir, chunk + '.pt')):
            continue
        t0 = time.time()
        rows = rows.assign(row=to_row_index(rows.cell_id.to_numpy(), full))
        idx = np.sort(rows.row.to_numpy())
        rows = rows.set_index('row').loc[idx]
        a = full[idx]
        a = a.to_memory() if a.isbacked else a.copy()
        # Control probes are not genes: 40 of the 180 SEA-AD "genes" are BLANK-* (v1 kept them as
        # gene nodes). Vizgen Blank-* are already dropped in load_vizgen; 10x h5 has none.
        a = a[:, ~a.var_names.str.upper().str.startswith('BLANK')].copy()
        a.X = a.X.toarray() if hasattr(a.X, 'toarray') else np.asarray(a.X)
        a.X = a.X.astype(np.float32)
        keep = a.X.sum(1) >= MIN_COUNTS   # raw counts here; preprocess() normalizes later
        a, rows = a[keep].copy(), rows[keep]
        use_types = args.source == 'sea'
        if use_types:
            a.obs['cell_type'] = rows.cell_type.astype(str).to_numpy()
        preprocess(a, out_dir, chunk, cell_type=use_types, gene_vocab=vocab, leiden_resolution='auto',
                   min_kept_counts=MIN_KEPT_COUNTS,
                   min_cluster_size=MIN_CLUSTER_SIZE)
        print(f'  {chunk}: {a.n_obs} cells, {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
