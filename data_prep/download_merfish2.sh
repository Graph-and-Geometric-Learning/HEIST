#!/bin/bash
# MERFISH 2.0 showcase (gs://vz-merfish2-showcase, needs the same Vizgen Google access as
# download_vizgen.sh). One sample per region: <short experiment>_<region>. Human regions go under
# vizgen/ (same cell_by_gene/cell_metadata format); the mouse brain goes under vizgen_mouse/ and is
# NOT part of the pretraining corpus unless moved.
RAW=${1:-$HOME/scratch_pi_sk2433/hm638/HEIST_raw}
module load gcloud 2>/dev/null
gcloud storage ls "gs://vz-merfish2-showcase/**/cell_by_gene.csv" | while read -r url; do
    dir=${url%/cell_by_gene.csv}; region=$(basename "$dir"); exp=$(basename "$(dirname "$dir")")
    short=$(echo "$exp" | sed -E 's/^[0-9]+_//; s/-D2M[0-9]+.*//; s/^240916JHHUBC0005XQ-V2V-//; s/-V2.*//')
    name="MERFISH2_${short}_${region#region_}"
    case "$short" in Ms*) dest="$RAW/vizgen_mouse/$name" ;; *) dest="$RAW/vizgen/$name" ;; esac
    mkdir -p "$dest"
    for f in cell_by_gene.csv cell_metadata.csv; do
        [ -s "$dest/$f" ] || gcloud storage cp "$dir/$f" "$dest/$f" || echo "FAIL $name/$f"
    done
    echo "done $name"
done
