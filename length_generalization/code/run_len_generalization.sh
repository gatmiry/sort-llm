#!/bin/bash
# SLURM batch script for length generalization experiments (SortGPT)

#SBATCH --job-name=sortgpt_len_gen
#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1  
#SBATCH --cpus-per-task=24
#SBATCH --mem=100G
#SBATCH --time=1-23:59:00
#SBATCH --account=kempner_grads
#SBATCH --partition=kempner
#SBATCH --mail-user=csu@g.harvard.edu
#SBATCH --output=logs/len_gen-%A_%a.out
#SBATCH --array=0-287

set -euo pipefail

# Activate environment
# source "$HOME/miniconda3/etc/profile.d/conda.sh"
source ~/.bashrc
conda deactivate
conda activate vla

# export HF_HOME=/temp_work/ch225816/hf

# ROOT_DIR=/temp_work/ch225816/ranking
# cd "$ROOT_DIR"

# Experiment grid:
# - vocab_n: 64, 128, 256
# - n_embd: N/2, N  (relative to vocab)
# - train_max_k: 16, 32
# - layer_counts: 1, 2
# - length_modes: mix, curriculum
# - use_layernorm: true, false
# - mlp: off (we already have mlp-on results)
# - seeds: 3
# - fixed: no pos embedding
# Total: 3 × 2 × 2 × 3 × (2 layers × 2 modes × 1 mlp × 2 layernorm) = 288 tasks

VOCAB_LIST=(64 128 256)
TRAIN_MAX_K_LIST=(16 32)
LAYER_COUNTS=(1 2)
LENGTH_MODES=(mix curriculum)
MLP_FLAGS=(false)
LAYERNORM_FLAGS=(true false)
SEEDS=(1337 2337 3337)

# Inner grid size (handled by Python's build_grid):
# layers(2) × length_modes(2) × mlp(1) × layernorm(2) = 8
GRID_INNER=$(( ${#LAYER_COUNTS[@]} * ${#LENGTH_MODES[@]} * ${#MLP_FLAGS[@]} * ${#LAYERNORM_FLAGS[@]} ))

TASK_ID=${SLURM_ARRAY_TASK_ID}
INNER_ID=$(( TASK_ID % GRID_INNER ))
REM=$(( TASK_ID / GRID_INNER ))
SEED_INDEX=$(( REM % ${#SEEDS[@]} ))
REM=$(( REM / ${#SEEDS[@]} ))
K_INDEX=$(( REM % ${#TRAIN_MAX_K_LIST[@]} ))
REM=$(( REM / ${#TRAIN_MAX_K_LIST[@]} ))
EMBD_INDEX=$(( REM % 2 ))
VOCAB_INDEX=$(( REM / 2 ))

VOCAB=${VOCAB_LIST[$VOCAB_INDEX]}
TRAIN_MAX_K=${TRAIN_MAX_K_LIST[$K_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}
if [ "$EMBD_INDEX" -eq 0 ]; then
  N_EMBD=$(( VOCAB / 2 ))
else
  N_EMBD=$VOCAB
fi
TEST_MIN_K=$TRAIN_MAX_K
TEST_MAX_K=$TRAIN_MAX_K
GROUP_NAME="len_gen_v${VOCAB}_e${N_EMBD}_k${TRAIN_MAX_K}"

python -u length_generalization/code/sortGPT_len_generalization.py \
  --project sortgpt \
  --root ./grid_outputs \
  --group "$GROUP_NAME" \
  --task-id "$INNER_ID" \
  --layer-counts "${LAYER_COUNTS[@]}" \
  --without-pos-flags true \
  --use-mlp-flags "${MLP_FLAGS[@]}" \
  --use-layernorm-flags "${LAYERNORM_FLAGS[@]}" \
  --length-modes "${LENGTH_MODES[@]}" \
  --allow-duplicates-flags false \
  --vocab-n "$VOCAB" \
  --n-embd "$N_EMBD" \
  --n-heads 1 \
  --train-min-k "$TRAIN_MAX_K" \
  --train-max-k "$TRAIN_MAX_K" \
  --test-min-k "$TEST_MIN_K" \
  --test-max-k "$TEST_MAX_K" \
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
