#!/bin/bash
set -e
cd /mnt/task_runtime/sort-llm

SCRIPT=mechanistic-interpretability/role-of-position/plot_hijack_per_i.py
DATADIR=mechanistic-interpretability/role-of-position/data_allI_v2
MB=5000

declare -A CKPTS
CKPTS[seed1]="new-grid/k32_N512/checkpoints/std0p01_iseed1__ckpt100000.pt"
CKPTS[seed2]="new-grid-multiple/k32_N512/seed2/checkpoints/std0p01_iseed2__ckpt100000.pt"
CKPTS[seed3]="new-grid-multiple/k32_N512/seed3/checkpoints/std0p01_iseed3__ckpt100000.pt"
CKPTS[seed4]="new-grid-multiple/k32_N512/seed4/checkpoints/std0p01_iseed4__ckpt100000.pt"
CKPTS[seed5]="new-grid-multiple/k32_N512/seed5/checkpoints/std0p01_iseed5__ckpt100000.pt"

for SEED in seed1 seed2 seed3 seed4 seed5; do
  CK="${CKPTS[$SEED]}"
  echo "===== $SEED ====="

  for GAP in 5 60; do
    echo "  gap=$GAP ..."
    if [ "$GAP" -le 5 ]; then
      OFFARGS="--fine-offsets"
    elif [ "$GAP" -eq 60 ]; then
      OFFARGS="--offsets 61,65,70,80,90,100,120,150"
    fi
    python $SCRIPT --gap $GAP $OFFARGS --group-avg "0-497" \
      --ckpt "$CK" --out-tag "${SEED}_allI_v2" --max-batches $MB \
      --save-data "$DATADIR/${SEED}_gap${GAP}.json"
  done
done

echo "All done."
