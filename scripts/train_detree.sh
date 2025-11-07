#!/usr/bin/env bash
#
# DETree training shortcut.
#
# The command below expands to ``python -m detree.cli.train``
# with all options spelled out so you never have to hunt for CLI flags.
#
DATA_ROOT="/path/to/RealBench"
TREE_TXT="/path/to/RealBench/trees/pcl_tree.txt"  # Tree definition file produced by the HAT pipeline.
OUTPUT_DIR="/path/to/RealBench/runs"            # Directory where experiment folders are saved.
EXPERIMENT_NAME="detree_stage1"                 # Base name for the run directory.
MODEL_NAME="FacebookAI/roberta-large"
DATASET_NAME="all"                              # Dataset alias understood by detree.utils.dataset.load_datapath.
DATASET_SPLIT="train"                           # Split to consume (train/test/extra).
DEVICE_NUM=4                                     # Number of CUDA devices available to Fabric.
PER_GPU_BATCH_SIZE=64
PER_GPU_EVAL_BATCH_SIZE=16
NUM_WORKERS=8
TOTAL_EPOCH=10                                   # Number of training epochs.
WARMUP_STEPS=2000                                # Scheduler warm-up steps.
LEARNING_RATE=3e-5                               # Peak learning rate.
MIN_LR=5e-6                                      # Minimum learning rate for cosine schedule.
WEIGHT_DECAY=1e-4                                # Optimiser weight decay.
BETA1=0.9
BETA2=0.99
EPS=1e-6
TEMPERATURE=0.07                                 # Contrastive loss temperature.
PROJECTION_SIZE=1024                             # Projection head dimension.
ADV_P=0.5                                        # Probability of sampling adversarial data.
POOLING="max"                                    # Embedding pooling strategy.
FREEZE_LAYER=0                                   # Number of initial encoder layers to freeze.
SEED=42                                          # Random seed for reproducibility.
LORA_R=128                                       # LoRA rank.
LORA_ALPHA=256                                   # LoRA scaling factor.
LORA_DROPOUT=0.0                                 # LoRA dropout probability.
MAX_LENGTH=512                                   # Maximum tokenised sequence length.

# Toggle optional behaviour by adding/removing flags from EXTRA_FLAGS. Useful
# examples: --resume, --deepspeed, --no-lora, --no-adversarial,
# --no-include-attack, --no-has-mix, --no-freeze-embedding-layer.
# If deepspeed is not used, DDP will be used by default.
EXTRA_FLAGS=(
  # --resume
  --deepspeed
  # --no-lora
)

set -euo pipefail

python -m detree.cli.train \
  --path "$DATA_ROOT" \
  --tree-txt "$TREE_TXT" \
  --output-dir "$OUTPUT_DIR" \
  --experiment-name "$EXPERIMENT_NAME" \
  --model-name "$MODEL_NAME" \
  --dataset-name "$DATASET_NAME" \
  --dataset "$DATASET_SPLIT" \
  --device-num "$DEVICE_NUM" \
  --per-gpu-batch-size "$PER_GPU_BATCH_SIZE" \
  --per-gpu-eval-batch-size "$PER_GPU_EVAL_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --max-length "$MAX_LENGTH" \
  --total-epoch "$TOTAL_EPOCH" \
  --warmup-steps "$WARMUP_STEPS" \
  --lr "$LEARNING_RATE" \
  --min-lr "$MIN_LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --beta1 "$BETA1" \
  --beta2 "$BETA2" \
  --eps "$EPS" \
  --temperature "$TEMPERATURE" \
  --projection-size "$PROJECTION_SIZE" \
  --adv-p "$ADV_P" \
  --pooling "$POOLING" \
  --freeze-layer "$FREEZE_LAYER" \
  --seed "$SEED" \
  --lora-r "$LORA_R" \
  --lora-alpha "$LORA_ALPHA" \
  --lora-dropout "$LORA_DROPOUT" \
  "${EXTRA_FLAGS[@]}"
