#!/bin/bash
# SLURM script for Huang-style length ranges with LN ablations.
#
# Experiments:
#   0) no LN + RMS qk-norm (no gating), train K<=50
#   1) no LN + element-wise attention output gate (no qk-norm), train K<=50
#   2) keep LN, train only K=50
#
# Common eval buckets:
#   in-dist:   K=2..50
#   ood-mid:   K=51..100
#   ood-long:  K=101..150

#SBATCH --job-name=huang_ln_ablate_v2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1  
#SBATCH --cpus-per-task=24
#SBATCH --mem=100G
#SBATCH --account=kempner_grads
#SBATCH --partition=kempner
#SBATCH --time=72:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=csu@g.harvard.edu
#SBATCH --output=logs/huang_ln_ablate-%A_%a.out
#SBATCH --array=0-71

set -euo pipefail

# source "$HOME/miniconda3/etc/profile.d/conda.sh"
source ~/.bashrc
conda deactivate
conda activate vla

# export HF_HOME=/temp_work/ch225816/hf

# ROOT_DIR=/temp_work/ch225816/sort-llm-repo/length_generalization/code
# cd "$ROOT_DIR"

# fixed grid dims in this launcher
LAYER_COUNTS=(1 2)
MLP_FLAGS=(true false)
SEEDS=(1337 2337 3337)
POS_FLAGS=(true false)
GRID_SIZE=$(( ${#LAYER_COUNTS[@]} * ${#MLP_FLAGS[@]} * ${#POS_FLAGS[@]} ))   # 8 tasks per seed
TASKS_PER_VARIANT=$(( GRID_SIZE * ${#SEEDS[@]} ))                            # 24

TASK_ID=${SLURM_ARRAY_TASK_ID}
VARIANT_INDEX=$(( TASK_ID / TASKS_PER_VARIANT ))     # 0..2
REM=$(( TASK_ID % TASKS_PER_VARIANT ))
SEED_INDEX=$(( REM / GRID_SIZE ))
INNER_ID=$(( REM % GRID_SIZE ))

if (( VARIANT_INDEX < 0 || VARIANT_INDEX > 2 )); then
  echo "Invalid variant index $VARIANT_INDEX from task $TASK_ID"
  exit 1
fi

SEED=${SEEDS[$SEED_INDEX]}

# defaults
TRAIN_MIN_K=2
TRAIN_MAX_K=50
USE_LN_FLAGS=(false)
USE_QK_NORM_FLAG=""
USE_RMS_QK_NORM_FLAG=""
USE_GATED_ATTN_FLAG=""
USE_ELEMWISE_ATTN_GATE_FLAG=""
VARIANT_NAME=""

case "$VARIANT_INDEX" in
  0)
    VARIANT_NAME="noLN_rmsQkNorm_trainLe50"
    USE_QK_NORM_FLAG="--use-qk-norm"
    USE_RMS_QK_NORM_FLAG="--use-rms-qk-norm"
    ;;
  1)
    VARIANT_NAME="noLN_elemGate_trainLe50"
    USE_ELEMWISE_ATTN_GATE_FLAG="--use-elementwise-attn-output-gate"
    ;;
  2)
    VARIANT_NAME="LN_fixed50"
    TRAIN_MIN_K=50
    TRAIN_MAX_K=50
    USE_LN_FLAGS=(true)
    ;;
esac

GROUP_NAME="huang_${VARIANT_NAME}"

python -u length_generalization/code/sortGPT_len_generalization.py \
  --project sortgpt-huang-ln-ablations \
  --root /n/holylfs06/LABS/sham_lab/Users/chloe00/sort-llm/grid_outputs \
  --group "$GROUP_NAME" \
  --task-id "$INNER_ID" \
  --layer-counts "${LAYER_COUNTS[@]}" \
  --without-pos-flags "${POS_FLAGS[@]}" \
  --use-mlp-flags "${MLP_FLAGS[@]}" \
  --use-layernorm-flags "${USE_LN_FLAGS[@]}" \
  --length-modes mix \
  --allow-duplicates-flags false \
  --vocab-n 256 \
  --n-embd 64 \
  --n-heads 1 \
  --train-min-k "$TRAIN_MIN_K" \
  --train-max-k "$TRAIN_MAX_K" \
  --test-min-k 2 \
  --test-max-k 150 \
  --eval-id-min 2 \
  --eval-id-max 50 \
  --eval-mid-min 51 \
  --eval-mid-max 100 \
  --eval-long-min 101 \
  --eval-long-max 150 \
  --eval-samples-per-length 100 \
  --eval-batch-size 100 \
  --max-iters 30000 \
  --warmup-iters 400 \
  --learning-rate 1e-3 \
  --min-lr 1e-6 \
  --micro-batch-size 4096 \
  --effective-batch-size 4096 \
  --log-interval 250 \
  --eval-interval 250 \
  --ckpt-interval 15000 \
  --seed "$SEED" \
  $USE_QK_NORM_FLAG \
  $USE_RMS_QK_NORM_FLAG \
  $USE_GATED_ATTN_FLAG \
  $USE_ELEMWISE_ATTN_GATE_FLAG

