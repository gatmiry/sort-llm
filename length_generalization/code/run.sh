#!/bin/bash
#SBATCH --job-name=train-sortgpt

#SBATCH --output=/n/holylfs06/LABS/sham_lab/Users/chloe00/sort-llm/vocab_size_and_layer_num_ablations/logs/%A_%a.log
#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1  
#SBATCH --cpus-per-task=24
#SBATCH --mem=100G
#SBATCH --time=3:00:00

#SBATCH --account=kempner_grads
#SBATCH --partition=kempner
#SBATCH --mail-user=csu@g.harvard.edu
#SBATCH --mail-type=END
#SBATCH --exclusive
#SBATCH --array=0-11


# Custom environment
source ~/.bashrc
conda deactivate
conda activate vla

# sleep $(( SLURM_ARRAY_TASK_ID * 60 ))
# module load cuda/12.4.1-fasrc01
# export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:${HOME}/cuda-12.0/targets/x86_64-linux/include
# module load gcc/12.2.0-fasrc01


# python -u grid_train_sortGPT.py \
#   --project sortgpt \
#   --root /n/holylfs06/LABS/sham_lab/Users/chloe00/sort-llm/grid_outputs \
#   --vocab-sizes 256  \
#   --layer-counts 1 2 3 \
#   --block-sizes 16 \
#   --without-pos-flags false true \
#   --vocab-over-embd-list 2 4 8 \
#   --allow-duplicates-flags true \
#   --use-mlp-flags true false \
#   --max-iters 25000 \
#   --warmup-iters 200 \
#   --learning-rate 1e-4 \
#   --min-lr 1e-6 \
#   --micro-batch-size 4096 \
#   --effective-batch-size 4096 \
#   --log-interval 250 \
#   --ckpt-interval 20000



python -u grid_train_sortGPT.py \
  --project sortgpt \
  --root /n/holylfs06/LABS/sham_lab/Users/chloe00/sort-llm/grid_outputs \
  --vocab-sizes 128  \
  --layer-counts 1 2 3 \
  --block-sizes 64 \
  --test-block-sizes 64 \
  --without-pos-flags true \
  --vocab-over-embd-list 2 \
  --allow-duplicates-flags true \
  --use-mlp-flags true \
  --max-iters 25000 \
  --warmup-iters 200 \
  --learning-rate 1e-4 \
  --min-lr 1e-6 \
  --micro-batch-size 4096 \
  --effective-batch-size 4096 \
  --log-interval 250 \
  --ckpt-interval 20000



# length generalization experiments
# python -u sortGPT_len_generalization.py \
#   --project sortgpt \
#   --root /n/holylfs06/LABS/sham_lab/Users/chloe00/sort-llm/grid_outputs \
#   --layer-counts 1 2 3 \
#   --without-pos-flags false true \
#   --use-mlp-flags true false \
#   --length-modes mix curriculum \
#   --allow-duplicates-flags false \
#   --vocab-n 128 \
#   --n-embd 128 \
#   --n-heads 1 \
#   --train-min-k 2 --train-max-k 16 \
#   --test-min-k 2 --test-max-k 32 \
#   --eval-samples-per-length 100 \
#   --eval-batch-size 100 \
#   --max-iters 25000 \
#   --micro-batch-size 4096 \
#   --effective-batch-size 4096 \
#   --log-interval 250 \
#   --eval-interval 250 \
#   --ckpt-interval 20000

