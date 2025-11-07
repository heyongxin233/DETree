#!/usr/bin/env bash
#
# Export DETree embeddings for RealBench.
#
DATA_ROOT="/path/to/RealBench"                  # Root directory that contains the RealBench datasets.
DATABASE_NAME="all"                              # Dataset alias understood by detree.utils.dataset.load_datapath.
MODEL_NAME="FacebookAI/roberta-large"          # Backbone encoder identifier or local Hugging Face-style directory.
SAVE_DIR="/path/to/RealBench/embeddings"       # Directory where embedding databases will be stored.
SAVE_NAME="detree_stage1"                       # Filename (without extension) for the saved embeddings.
DEVICE_NUM=4                                     # Number of CUDA devices available to Fabric.
BATCH_SIZE=64                                    # Inference batch size per device.
NUM_WORKERS=8                                    # DataLoader workers for reading datasets.
MAX_LENGTH=512                                   # Maximum tokenised sequence length.
POOLING="max"                                    # Embedding pooling strategy.
NEED_LAYER=(16 17 18 19 22 23)                   # Hidden layers to export from the encoder.
SPLIT="train"                                   # Dataset split to encode (train/test/extra).

# Extra CLI switches can be appended here, for example:
#   EXTRA_FLAGS=(--no-adversarial)
EXTRA_FLAGS=(
  # --no-adversarial
  # --has-mix
)

set -euo pipefail

python -m detree.cli.embeddings \
  --path "$DATA_ROOT" \
  --database-name "$DATABASE_NAME" \
  --model-name "$MODEL_NAME" \
  --savedir "$SAVE_DIR" \
  --name "$SAVE_NAME" \
  --device-num "$DEVICE_NUM" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --max-length "$MAX_LENGTH" \
  --pooling "$POOLING" \
  --need-layer "${NEED_LAYER[@]}" \
  --split "$SPLIT" \
  "${EXTRA_FLAGS[@]}"
