#!/bin/bash
# Download the per-sample count matrix + cell table (not the multi-GB outs.zip) for every Xenium
# sample in the pretraining set. Versions were probed from the 10x CDN (xenium_versions.txt).
OUT=${1:-$HOME/scratch_pi_sk2433/hm638/HEIST_raw/xenium}
HERE=$(dirname "$0")
mkdir -p "$OUT"
fetch() {
    s=$1; v=$2; out=$3
    base=https://cf.10xgenomics.com/samples/xenium/$v/$s/$s
    mkdir -p "$out/$s"
    # Skip finished files: resuming a complete file gets HTTP 416, which -f turns into a failure.
    [ -s "$out/$s/cell_feature_matrix.h5" ] || curl -sfL -o "$out/$s/cell_feature_matrix.h5" "${base}_cell_feature_matrix.h5" || echo "FAIL $s matrix"
    [ -s "$out/$s/cells.parquet" ] || curl -sfL -o "$out/$s/cells.parquet" "${base}_cells.parquet" \
        || curl -sfL -o "$out/$s/cells.csv.gz" "${base}_cells.csv.gz" || echo "FAIL $s cells"
    echo "done $s"
}
export -f fetch
xargs -P 6 -L 1 bash -c 'fetch "$0" "$1" "'"$OUT"'"' < "$HERE/xenium_versions.txt"
