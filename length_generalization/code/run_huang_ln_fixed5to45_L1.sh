#!/bin/bash
# LN-only fixed-length sweep for K=5,10,...,45 (L1 only, MLP on/off).
#
# Eval buckets:
#   in-dist:   K=2..50
#   ood-mid:   K=51..100
#   ood-long:  K=101..150

#SBATCH --job-name=huang_ln_k5to45_L1
#SBATCH --account=chip
#SBATCH --partition=chip-gpu
#SBATCH --gres=gpu:large:1
#SBATCH --mem=48GB
#SBATCH --time=72:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=shan.chen@childrens.harvard.edu
#SBATCH --output=logs/huang_ln_k5to45_L1-%A_%a.out
#SBATCH --array=0-53

set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate verl

export HF_HOME=/temp_work/ch225816/hf

ROOT_DIR=/temp_work/ch225816/sort-llm-repo/length_generalization/code
cd "$ROOT_DIR"

TRAIN_K_LIST=(5 10 15 20 25 30 35 40 45)
LAYER_COUNTS=(1)
MLP_FLAGS=(true false)
SEEDS=(1337 2337 3337)

GRID_SIZE=$(( ${#LAYER_COUNTS[@]} * ${#MLP_FLAGS[@]} ))    # 2
TASKS_PER_K=$(( GRID_SIZE * ${#SEEDS[@]} ))                # 6

TASK_ID=${SLURM_ARRAY_TASK_ID}
K_INDEX=$(( TASK_ID / TASKS_PER_K ))                       # 0..8
REM=$(( TASK_ID % TASKS_PER_K ))
SEED_INDEX=$(( REM / GRID_SIZE ))
INNER_ID=$(( REM % GRID_SIZE ))

if (( K_INDEX < 0 || K_INDEX >= ${#TRAIN_K_LIST[@]} )); then
  echo "Invalid K index $K_INDEX from task $TASK_ID"
  exit 1
fi

TRAIN_K=${TRAIN_K_LIST[$K_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}

GROUP_NAME="huang_LN_fixedK5to45_L1"

python -u sortGPT_len_generalization.py \
  --project sortgpt \
  --root /temp_work/ch225816/sort-llm/grid_outputs \
  --group "$GROUP_NAME" \
  --task-id "$INNER_ID" \
  --layer-counts "${LAYER_COUNTS[@]}" \
  --without-pos-flags true \
  --use-mlp-flags "${MLP_FLAGS[@]}" \
  --use-layernorm-flags true \
  --norm-type layernorm \
  --length-modes mix \
  --allow-duplicates-flags false \
  --vocab-n 256 \
  --n-embd 128 \
  --n-heads 1 \
  --train-min-k "$TRAIN_K" \
  --train-max-k "$TRAIN_K" \
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
  --learning-rate 1e-4 \
  --min-lr 1e-6 \
  --micro-batch-size 4096 \
  --effective-batch-size 4096 \
  --log-interval 250 \
  --eval-interval 250 \
  --ckpt-interval 15000 \
  --seed "$SEED"

