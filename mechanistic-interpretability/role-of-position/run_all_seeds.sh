#!/bin/bash
# Run hijack experiments for all 5 seeds × 4 gaps, saving JSON data
set -e
cd /mnt/task_runtime/sort-llm

SCRIPT=mechanistic-interpretability/role-of-position/plot_hijack_per_i.py
DATADIR=mechanistic-interpretability/role-of-position/data_allI
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

  echo "  gap=1 ..."
  python $SCRIPT --gap 1 --fine-offsets --group-avg "0-497" \
    --ckpt "$CK" --out-tag "${SEED}_allI" --max-batches $MB \
    --save-data "$DATADIR/${SEED}_gap1.json"

  echo "  gap=10 ..."
  python $SCRIPT --gap 10 --offsets "11,13,15,17,19,21,25,30" --group-avg "0-497" \
    --ckpt "$CK" --out-tag "${SEED}_allI" --max-batches $MB \
    --save-data "$DATADIR/${SEED}_gap10.json"

  echo "  gap=20 ..."
  python $SCRIPT --gap 20 --offsets "21,25,30,35,40,45,50" --group-avg "0-497" \
    --ckpt "$CK" --out-tag "${SEED}_allI" --max-batches $MB \
    --save-data "$DATADIR/${SEED}_gap20.json"

  echo "  gap=40 ..."
  python $SCRIPT --gap 40 --offsets "41,45,50,55,60,70,80,100,120,150" --group-avg "0-497" \
    --ckpt "$CK" --out-tag "${SEED}_allI" --max-batches $MB \
    --save-data "$DATADIR/${SEED}_gap40.json"
done

echo "All done. JSON files in $DATADIR/"
