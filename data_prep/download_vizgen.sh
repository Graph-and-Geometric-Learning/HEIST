#!/bin/bash
# Download cell_by_gene.csv + cell_metadata.csv for the 17 Vizgen FFPE showcase samples.
# The bucket is not public: access is granted to the Google account that submitted Vizgen's
# showcase form. Authenticate once first:
#   module load gcloud && gcloud auth login --no-launch-browser
OUT=${1:-$HOME/scratch_pi_sk2433/hm638/HEIST_raw/vizgen}
BUCKET=gs://vz-ffpe-showcase
module load gcloud 2>/dev/null
mkdir -p "$OUT"
for s in HumanBreastCancerPatient1 HumanColonCancerPatient1 HumanColonCancerPatient2 \
         HumanLiverCancerPatient1 HumanLiverCancerPatient2 HumanLungCancerPatient1 \
         HumanLungCancerPatient2 HumanMelanomaPatient1 HumanMelanomaPatient2 \
         HumanOvarianCancerPatient1 HumanOvarianCancerPatient2Slice1 HumanOvarianCancerPatient2Slice2 \
         HumanOvarianCancerPatient2Slice3 HumanProstateCancerPatient1 HumanProstateCancerPatient2 HumanUterineCancerPatient1 \
         HumanUterineCancerPatient2-RACostain HumanUterineCancerPatient2-ROCostain; do
    mkdir -p "$OUT/$s"
    for f in cell_by_gene.csv cell_metadata.csv; do
        gcloud storage cp -n "$BUCKET/$s/$f" "$OUT/$s/$f" || echo "FAIL $s/$f"
    done
done
