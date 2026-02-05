#!/bin/bash
# SLURM batch script for length generalization experiments without layer norm (SortGPT)

#SBATCH --job-name=sortgpt_len_gen_no_ln
#SBATCH --account=chip
#SBATCH --partition=chip-gpu
#SBATCH --gres=gpu:large:1
#SBATCH --mem=48GB
#SBATCH --time=99:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=shan.chen@childrens.harvard.edu
#SBATCH --output=logs/len_gen_no_ln-%A_%a.out
#SBATCH --array=0-83

set -euo pipefail

# Activate environment
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate verl

export HF_HOME=/temp_work/ch225816/hf

ROOT_DIR=/temp_work/ch225816/ranking
cd "$ROOT_DIR"

# Experiment grid:
# - layer_counts: 1,2
# - length_modes (data loading): mix, curriculum
# - train_max_k: 4,6,8,10,12,14,16
# - seeds: 3
# - mlp: on, off
# - layernorm: OFF (key difference from previous runs)
# - fixed: no pos embedding, test lengths 17-32

TRAIN_MAX_K_LIST=(4 6 8 10 12 14 16)
LAYER_COUNTS=(1 2)
LENGTH_MODES=(mix curriculum)
MLP_FLAGS=(true false)
SEEDS=(1337 2337 3337)

GRID_PER_K=$(( ${#LAYER_COUNTS[@]} * ${#LENGTH_MODES[@]} * ${#MLP_FLAGS[@]} ))
GRID_PER_K_WITH_SEEDS=$(( GRID_PER_K * ${#SEEDS[@]} ))
TASK_ID=${SLURM_ARRAY_TASK_ID}
K_INDEX=$(( TASK_ID / GRID_PER_K_WITH_SEEDS ))
REM=$(( TASK_ID % GRID_PER_K_WITH_SEEDS ))
SEED_INDEX=$(( REM / GRID_PER_K ))
INNER_ID=$(( REM % GRID_PER_K ))

if (( K_INDEX < 0 || K_INDEX >= ${#TRAIN_MAX_K_LIST[@]} )); then
  echo "Invalid task id $TASK_ID for TRAIN_MAX_K_LIST size ${#TRAIN_MAX_K_LIST[@]}"
  exit 1
fi

TRAIN_MAX_K=${TRAIN_MAX_K_LIST[$K_INDEX]}
GROUP_NAME="len_gen_no_ln_k${TRAIN_MAX_K}"
SEED=${SEEDS[$SEED_INDEX]}

python -u sortGPT_len_generalization.py \
  --project sortgpt \
  --root /temp_work/ch225816/sort-llm/grid_outputs \
  --group "$GROUP_NAME" \
  --task-id "$INNER_ID" \
  --layer-counts "${LAYER_COUNTS[@]}" \
  --without-pos-flags true \
  --use-mlp-flags "${MLP_FLAGS[@]}" \
  --use-layernorm-flags false \
  --length-modes "${LENGTH_MODES[@]}" \
  --allow-duplicates-flags false \
  --vocab-n 128 \
  --n-embd 128 \
  --n-heads 1 \
  --train-min-k 2 \
  --train-max-k "$TRAIN_MAX_K" \
  --test-min-k 17 \
  --test-max-k 32 \
  --eval-samples-per-length 100 \
  --eval-batch-size 100 \
  --max-iters 40000 \
  --warmup-iters 400 \
  --learning-rate 1e-4 \
  --min-lr 1e-6 \
  --micro-batch-size 4096 \
  --effective-batch-size 4096 \
  --log-interval 250 \
  --eval-interval 250 \
  --ckpt-interval 20000 \
  --seed "$SEED"
