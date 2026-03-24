#!/usr/bin/env bash
#
# run_experiment.sh — Launch a multi-seed SortGPT training experiment.
#
# Edit the CONFIGURATION section below, then run:
#     bash run_experiment.sh
#
# To stop a single seed:   kill $(cat $RUN_DIR/pids/seed_XXXX.pid)
# To stop all:             kill $(cat $RUN_DIR/pids/*.pid)
# To check status:         cat $RUN_DIR/status.txt
# To view monitor log:     tail -f $RUN_DIR/logs/monitor.log

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these for your experiment
# ═══════════════════════════════════════════════════════════════════════════════

# Seeds and their init_std values (parallel arrays, same length)
SEEDS=(1501 1502 1503 1504 1505)
INIT_STDS=(0.02 0.02 0.02 0.02 0.02)

# Training settings
MAX_ITERS=1000000
CHECKPOINT_EVERY=50000
LEARNING_RATE=0.03
BATCH_SIZE=4096

# GPU assignment: which GPUs to use for training, and which for the monitor
TRAIN_GPUS=(0 1 2 3 4)   # one per seed (must match length of SEEDS)
MONITOR_GPU=5             # GPU for the eval monitor

# Experiment name (used in directory name)
EXPERIMENT_NAME="my_experiment"

# ═══════════════════════════════════════════════════════════════════════════════
# END CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

NUM_SEEDS=${#SEEDS[@]}
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$SCRIPT_DIR/runs/${EXPERIMENT_NAME}_${TIMESTAMP}"

mkdir -p "$RUN_DIR/checkpoints" "$RUN_DIR/logs" "$RUN_DIR/pids" "$RUN_DIR/plots"

echo "============================================================"
echo "  SortGPT Experiment: $EXPERIMENT_NAME"
echo "  RUN_DIR:    $RUN_DIR"
echo "  SEEDS:      ${SEEDS[*]}"
echo "  INIT_STDS:  ${INIT_STDS[*]}"
echo "  MAX_ITERS:  $MAX_ITERS"
echo "  CKPT_EVERY: $CHECKPOINT_EVERY"
echo "  LR:         $LEARNING_RATE"
echo "============================================================"
echo ""

# Launch training workers
for i in "${!SEEDS[@]}"; do
    seed=${SEEDS[$i]}
    std=${INIT_STDS[$i]}
    gpu=${TRAIN_GPUS[$i]}
    log_file="$RUN_DIR/logs/seed_${seed}_gpu${gpu}.log"
    pid_file="$RUN_DIR/pids/seed_${seed}.pid"

    echo "  Starting seed=$seed (std=$std) on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python3 "$SCRIPT_DIR/train.py" \
        --init-seed "$seed" \
        --init-std "$std" \
        --run-dir "$RUN_DIR" \
        --max-iters "$MAX_ITERS" \
        --checkpoint-every "$CHECKPOINT_EVERY" \
        --lr "$LEARNING_RATE" \
        --batch-size "$BATCH_SIZE" \
        > "$log_file" 2>&1 &
    echo $! > "$pid_file"
    echo "    PID=$(cat $pid_file) → $log_file"
done

echo ""

# Launch monitor
echo "  Starting monitor on GPU $MONITOR_GPU..."
CUDA_VISIBLE_DEVICES=$MONITOR_GPU python3 "$SCRIPT_DIR/monitor.py" \
    --run-dir "$RUN_DIR" \
    --seeds ${SEEDS[@]} \
    --init-stds ${INIT_STDS[@]} \
    --checkpoint-every "$CHECKPOINT_EVERY" \
    --max-iters "$MAX_ITERS" \
    > "$RUN_DIR/logs/monitor.log" 2>&1 &
echo $! > "$RUN_DIR/pids/monitor.pid"
echo "    Monitor PID=$(cat $RUN_DIR/pids/monitor.pid)"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  All launched."
echo ""
echo "  Check status:    cat $RUN_DIR/status.txt"
echo "  Monitor log:     tail -f $RUN_DIR/logs/monitor.log"
echo "  Stop one seed:   kill \$(cat $RUN_DIR/pids/seed_XXXX.pid)"
echo "  Stop all:        kill \$(cat $RUN_DIR/pids/*.pid)"
echo "════════════════════════════════════════════════════════════"
