#!/bin/bash
# Fine-grained hijack sweeps for all 25 k32_N512 seeds.
#
# Unlike run_new_seeds.sh, every job passes an explicit --offsets list. Without
# it the script falls back to generate_gap_batch(), which cannot build a batch
# when GAP * block_size >= vocab_n (true for gap 20 and 40 at k32_N512) and
# silently produces empty results.
#
# Large gaps use targeted batches, which must fit cval, cval+GAP and one token
# per offset into block_size=32 tokens, so their sweeps are split into chunks
# of <=28 offsets. Merge the chunks with merge_hijack_chunks.py afterwards.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"

SCRIPT=mechanistic-interpretability/role-of-position/plot_hijack_per_i.py
DATADIR="${DATADIR:-mechanistic-interpretability/role-of-position/data_allI_v3}"
NUM_GPUS="${NUM_GPUS:-8}"
MB_RANDOM="${MB_RANDOM:-60000}"
MB_TARGETED="${MB_TARGETED:-40000}"
SEED_LIST="${SEED_LIST:-$(seq 1 25)}"
GAPS=(1 5 10 20 40 60)

# gap -> space-separated "lo-hi" offset chunks. Every range starts at gap+1
# because an offset equal to the gap collides with the true next value and is
# discarded by collect_data().
declare -A CHUNKS
CHUNKS[1]="2-20"
CHUNKS[5]="6-30"
CHUNKS[10]="11-45"
CHUNKS[20]="21-40 41-60 61-80"
CHUNKS[40]="41-64 65-88 89-112 113-136 137-160"
CHUNKS[60]="61-88 89-116 117-144 145-172 173-200"

declare -A CKPTS
CKPTS[seed1]="new-grid/k32_N512/checkpoints/std0p01_iseed1__ckpt100000.pt"
for s in 2 3 4 5; do
  CKPTS[seed${s}]="new-grid-multiple/k32_N512/seed${s}/checkpoints/std0p01_iseed${s}__ckpt100000.pt"
done
for s in $(seq 6 15); do
  CKPTS[seed${s}]="new-grid-multiple-2/k32_N512/seed${s}/checkpoints/std0p01_iseed${s}__ckpt100000.pt"
done
for s in $(seq 16 25); do
  CKPTS[seed${s}]="new-grid-multiple-3/k32_N512/seed${s}/checkpoints/std0p01_iseed${s}__ckpt100000.pt"
done

mkdir -p "$DATADIR"

declare -a JOB_QUEUE=()
for s in $SEED_LIST; do
  SEED="seed${s}"
  CK="${CKPTS[$SEED]:-}"
  if [ -z "$CK" ]; then
    echo "WARNING: no checkpoint mapping for $SEED, skipping" >&2
    continue
  fi
  if [ ! -f "$CK" ]; then
    echo "WARNING: missing checkpoint $CK, skipping $SEED" >&2
    continue
  fi
  for GAP in "${GAPS[@]}"; do
    for CHUNK in ${CHUNKS[$GAP]}; do
      OUT="$DATADIR/${SEED}_gap${GAP}__${CHUNK}.json"
      if [ -f "$OUT" ]; then
        echo "SKIP (exists): $OUT"
        continue
      fi
      JOB_QUEUE+=("${SEED}:${GAP}:${CHUNK}")
    done
  done
done

TOTAL=${#JOB_QUEUE[@]}
if [ "$TOTAL" -eq 0 ]; then
  echo "Nothing to do."
  exit 0
fi
echo "$(date '+%H:%M:%S') | Total jobs: $TOTAL across $NUM_GPUS GPUs"

declare -A GPU_PID
for ((g = 0; g < NUM_GPUS; g++)); do
  GPU_PID[$g]=0
done

find_free_gpu() {
  while true; do
    for ((g = 0; g < NUM_GPUS; g++)); do
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
  IFS=':' read -r SEED GAP CHUNK <<< "$job"
  IFS='-' read -r LO HI <<< "$CHUNK"
  JOB_IDX=$((JOB_IDX + 1))
  gpu=$(find_free_gpu)

  OFFSETS=$(seq "$LO" "$HI" | paste -sd, -)
  OUT="$DATADIR/${SEED}_gap${GAP}__${CHUNK}.json"
  LOG="$DATADIR/logs/${SEED}_gap${GAP}__${CHUNK}.log"
  mkdir -p "$(dirname "$LOG")"

  if [ "$GAP" -ge 20 ]; then
    MB=$MB_TARGETED
  else
    MB=$MB_RANDOM
  fi

  echo "$(date '+%H:%M:%S') | [$JOB_IDX/$TOTAL] $SEED gap=$GAP offsets=$LO..$HI on GPU $gpu"

  CUDA_VISIBLE_DEVICES=$gpu python $SCRIPT --gap "$GAP" --offsets "$OFFSETS" \
    --group-avg "0-497" --ckpt "${CKPTS[$SEED]}" \
    --out-tag "${SEED}_allI_v3_${CHUNK}" \
    --max-batches "$MB" --save-data "$OUT" \
    > "$LOG" 2>&1 &

  GPU_PID[$gpu]=$!
done

echo "$(date '+%H:%M:%S') | All $TOTAL jobs launched, waiting..."
for ((g = 0; g < NUM_GPUS; g++)); do
  pid=${GPU_PID[$g]}
  if [ "$pid" -ne 0 ]; then
    wait "$pid" 2>/dev/null || true
  fi
done
wait

echo "$(date '+%H:%M:%S') | ===== ALL DONE ====="
echo "Now merge chunks:"
echo "  python mechanistic-interpretability/role-of-position/merge_hijack_chunks.py --datadir $DATADIR"
