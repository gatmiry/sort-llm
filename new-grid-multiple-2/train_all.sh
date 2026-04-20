#!/usr/bin/env bash
# Train 10 new seeds (6-15) × 8 configs on 8 GPUs with a job queue.
# Usage: nohup bash new-grid-multiple-2/train_all.sh > new-grid-multiple-2/master.log 2>&1 &
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
TRAIN_PY="$REPO_DIR/sortgpt_toolkit/train.py"
BASE_OUT="$SCRIPT_DIR"

NUM_GPUS=8
SEEDS=(6 7 8 9 10 11 12 13 14 15)
BLOCK_SIZES=(16 32)
VOCAB_NS=(128 256 512 1024)

INIT_STD=0.01
LR=0.03
BATCH_SIZE=4096
MAX_ITERS=100000
CHECKPOINT_EVERY=5000

declare -a JOB_QUEUE=()
for SEED in "${SEEDS[@]}"; do
  for K in "${BLOCK_SIZES[@]}"; do
    for N in "${VOCAB_NS[@]}"; do
      JOB_QUEUE+=("${SEED}:${K}:${N}")
    done
  done
done

TOTAL_JOBS=${#JOB_QUEUE[@]}
echo "$(date '+%Y-%m-%d %H:%M:%S') | Total jobs: $TOTAL_JOBS (${#SEEDS[@]} seeds × ${#BLOCK_SIZES[@]} k × ${#VOCAB_NS[@]} N)"
echo "$(date '+%Y-%m-%d %H:%M:%S') | GPUs: $NUM_GPUS | Seeds: ${SEEDS[*]}"
echo ""

# Create all directories upfront
for SEED in "${SEEDS[@]}"; do
  for K in "${BLOCK_SIZES[@]}"; do
    for N in "${VOCAB_NS[@]}"; do
      mkdir -p "$BASE_OUT/k${K}_N${N}/seed${SEED}/checkpoints"
      mkdir -p "$BASE_OUT/k${K}_N${N}/seed${SEED}/logs"
    done
  done
done

declare -A GPU_PID
for ((g=0; g<NUM_GPUS; g++)); do
  GPU_PID[$g]=0
done

wait_for_gpu() {
  local gpu=$1
  local pid=${GPU_PID[$gpu]}
  if [ "$pid" -ne 0 ]; then
    wait "$pid" 2>/dev/null || true
    GPU_PID[$gpu]=0
  fi
}

find_free_gpu() {
  while true; do
    for ((g=0; g<NUM_GPUS; g++)); do
      local pid=${GPU_PID[$g]}
      if [ "$pid" -eq 0 ]; then
        echo "$g"
        return
      fi
      if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid" 2>/dev/null || true
        GPU_PID[$g]=0
        echo "$g"
        return
      fi
    done
    sleep 5
  done
}

launch_job() {
  local job_str=$1
  IFS=':' read -r SEED K N <<< "$job_str"

  local gpu
  gpu=$(find_free_gpu)

  local run_dir="$BASE_OUT/k${K}_N${N}/seed${SEED}"
  local log_file="$run_dir/logs/train.log"
  local tag="seed${SEED}_k${K}_N${N}"

  echo "$(date '+%Y-%m-%d %H:%M:%S') | START $tag on GPU $gpu"

  CUDA_VISIBLE_DEVICES=$gpu python3 "$TRAIN_PY" \
    --init-seed "$SEED" \
    --init-std "$INIT_STD" \
    --run-dir "$run_dir" \
    --max-iters "$MAX_ITERS" \
    --checkpoint-every "$CHECKPOINT_EVERY" \
    --lr "$LR" \
    --batch-size "$BATCH_SIZE" \
    --block-size "$K" \
    --vocab-n "$N" \
    > "$log_file" 2>&1 &

  GPU_PID[$gpu]=$!
}

JOB_IDX=0
for job_str in "${JOB_QUEUE[@]}"; do
  JOB_IDX=$((JOB_IDX + 1))
  echo "$(date '+%Y-%m-%d %H:%M:%S') | Queuing job $JOB_IDX/$TOTAL_JOBS: $job_str"
  launch_job "$job_str"
done

echo ""
echo "$(date '+%Y-%m-%d %H:%M:%S') | All $TOTAL_JOBS jobs launched/queued. Waiting for completion..."

for ((g=0; g<NUM_GPUS; g++)); do
  wait_for_gpu "$g"
done

wait

echo ""
echo "$(date '+%Y-%m-%d %H:%M:%S') | ===== ALL DONE ====="
echo "$(date '+%Y-%m-%d %H:%M:%S') | Checkpoints in: $BASE_OUT/k{16,32}_N{128,256,512,1024}/seed{6..15}/checkpoints/"
