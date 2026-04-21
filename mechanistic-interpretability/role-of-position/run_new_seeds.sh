#!/bin/bash
set -euo pipefail
cd /mnt/task_runtime/sort-llm

SCRIPT=mechanistic-interpretability/role-of-position/plot_hijack_per_i.py
DATADIR=mechanistic-interpretability/role-of-position/data_allI_v2
MB=5000
NUM_GPUS=8
GAPS=(1 5 10 20 40 60)

declare -A CKPTS
for s in $(seq 6 15); do
  CKPTS[seed${s}]="new-grid-multiple-2/k32_N512/seed${s}/checkpoints/std0p01_iseed${s}__ckpt100000.pt"
done
for s in $(seq 16 25); do
  CKPTS[seed${s}]="new-grid-multiple-3/k32_N512/seed${s}/checkpoints/std0p01_iseed${s}__ckpt100000.pt"
done

SEEDS=($(for s in $(seq 6 25); do echo "seed${s}"; done))

declare -a JOB_QUEUE=()
for SEED in "${SEEDS[@]}"; do
  for GAP in "${GAPS[@]}"; do
    OUT="$DATADIR/${SEED}_gap${GAP}.json"
    if [ -f "$OUT" ]; then
      echo "SKIP (exists): $OUT"
      continue
    fi
    JOB_QUEUE+=("${SEED}:${GAP}")
  done
done

TOTAL=${#JOB_QUEUE[@]}
echo "$(date '+%H:%M:%S') | Total jobs: $TOTAL across $NUM_GPUS GPUs"

declare -A GPU_PID
for ((g=0; g<NUM_GPUS; g++)); do
  GPU_PID[$g]=0
done

find_free_gpu() {
  while true; do
    for ((g=0; g<NUM_GPUS; g++)); do
      local pid=${GPU_PID[$g]}
      if [ "$pid" -eq 0 ]; then
        echo "$g"; return
      fi
      if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid" 2>/dev/null || true
        GPU_PID[$g]=0
        echo "$g"; return
      fi
    done
    sleep 2
  done
}

JOB_IDX=0
for job in "${JOB_QUEUE[@]}"; do
  IFS=':' read -r SEED GAP <<< "$job"
  JOB_IDX=$((JOB_IDX + 1))
  gpu=$(find_free_gpu)

  CK="${CKPTS[$SEED]}"
  OUT="$DATADIR/${SEED}_gap${GAP}.json"

  if [ "$GAP" -le 5 ]; then
    OFFARGS="--fine-offsets"
  elif [ "$GAP" -eq 60 ]; then
    OFFARGS="--offsets 61,65,70,80,90,100,120,150"
  else
    OFFARGS=""
  fi

  echo "$(date '+%H:%M:%S') | [$JOB_IDX/$TOTAL] $SEED gap=$GAP on GPU $gpu"

  CUDA_VISIBLE_DEVICES=$gpu python $SCRIPT --gap $GAP $OFFARGS \
    --group-avg "0-497" --ckpt "$CK" --out-tag "${SEED}_allI_v2" \
    --max-batches $MB --save-data "$OUT" \
    > /dev/null 2>&1 &

  GPU_PID[$gpu]=$!
done

echo "$(date '+%H:%M:%S') | All $TOTAL jobs launched, waiting..."
for ((g=0; g<NUM_GPUS; g++)); do
  pid=${GPU_PID[$g]}
  if [ "$pid" -ne 0 ]; then
    wait "$pid" 2>/dev/null || true
  fi
done
wait

echo "$(date '+%H:%M:%S') | ===== ALL DONE ====="
