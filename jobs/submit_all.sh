#!/bin/bash
# Submit every dataset job. Pass names to submit a subset:
#   ./submit_all.sh Adam Zeisel
set -euo pipefail
cd "$(dirname "$0")"

if [ $# -gt 0 ]; then
  DATASETS=("$@")
else
  DATASETS=(10X_PBMC human_kidney_counts Mouse mouse_ES Quake_10x_Bladder Quake_Smart-seq2_Limb_Muscle Quake_Smart-seq2_Trachea worm_neuron Zeisel Adam Bach Baron_human Baron_mouse Campbell Cao_2020_Spleen Muraro Quake_10x_Limb_Muscle_raw Quake_Smart-seq2_Diaphragm Shekhar Tosches_turtle Wang_Large_Intestine Young)
fi

for ds in "${DATASETS[@]}"; do
  f="job_${ds}.sh"
  if [ ! -f "$f" ]; then echo "missing $f, skipping" >&2; continue; fi
  jid=$(sbatch --parsable "$f")
  echo "$jid  $ds"
done
