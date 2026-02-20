#!/bin/bash
# LN-only mixed-length comparison with explicit train K choices {25, 50}.
#
# Goal:
#   Compare in-domain and OOD behavior, especially ID performance,
#   under Huang-style eval ranges.
#
# Eval buckets:
#   in-dist:   K=2..50
#   ood-mid:   K=51..100
#   ood-long:  K=101..150

#SBATCH --job-name=huang_ln_mix25_50
#SBATCH --account=chip
#SBATCH --partition=chip-gpu
#SBATCH --gres=gpu:large:1
#SBATCH --mem=48GB
#SBATCH --time=72:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=shan.chen@childrens.harvard.edu
#SBATCH --output=logs/huang_ln_mix25_50-%A_%a.out
#SBATCH --array=0-11

set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate verl

export HF_HOME=/temp_work/ch225816/hf

ROOT_DIR=/temp_work/ch225816/sort-llm-repo/length_generalization/code
cd "$ROOT_DIR"

LAYER_COUNTS=(1 2)
MLP_FLAGS=(true false)
SEEDS=(1337 2337 3337)

GRID_SIZE=$(( ${#LAYER_COUNTS[@]} * ${#MLP_FLAGS[@]} ))   # 4 tasks per seed
TASK_ID=${SLURM_ARRAY_TASK_ID}
SEED_INDEX=$(( TASK_ID / GRID_SIZE ))
INNER_ID=$(( TASK_ID % GRID_SIZE ))

if (( TASK_ID < 0 || TASK_ID > 11 )); then
  echo "Invalid task id $TASK_ID"
  exit 1
fi

SEED=${SEEDS[$SEED_INDEX]}

GROUP_NAME="huang_LN_mixK25_50_cmp"

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
  --train-min-k 25 \
  --train-max-k 50 \
  --train-k-choices 25 50 \
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

