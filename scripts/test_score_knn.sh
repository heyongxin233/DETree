#!/usr/bin/env bash
#
# Evaluate DETree checkpoints directly on JSONL corpora.
#
TRAIN_JSONL="/path/to/RealBench/jsonl/train.jsonl"  # JSONL file used to build the in-memory database.
TEST_JSONL="/path/to/RealBench/jsonl/test.jsonl"    # JSONL corpus to evaluate against the database.
MODEL_NAME="/path/to/RealBench/models/detree_final"  # DETree checkpoint (merged model or adapter directory).
LOG_FILE="/path/to/RealBench/logs/raw_knn_eval.txt"  # File where evaluation metrics will be appended.
DEVICE_NUM=4                                          # Number of CUDA devices available to Fabric.
BATCH_SIZE=32                                         # Inference batch size per device.
NUM_WORKERS=8                                         # DataLoader workers for streaming JSONL data.
MAX_LENGTH=512                                        # Maximum tokenised sequence length.
TEMPERATURE=0.05                                      # Softmax temperature for vote aggregation.
MAX_K=50                                              # Highest k to probe when evaluating.
MIN_LAYER=15                                          # Minimum encoder layer to include in the sweep.
POOLING="max"                                        # Embedding pooling strategy.
EMBEDDING_DIM=1024                                    # Dimension of each embedding vector.
N_SUBQUANTIZERS=1                                     # IVF-PQ subquantizers (for FAISS compatibility).
N_BITS=8                                              # Bits per subquantizer.
SAVE_DIR="/path/to/RealBench/database_snapshots"     # Directory for optional embedding dumps.
SAVE_NAME="raw_eval_snapshot"                        # Filename (without extension) when saving embeddings.
POOL_WORKERS=16                                       # CPU workers for multiprocessing kNN scoring.
SAVE_EMBEDDINGS=false                                 # Toggle to store the generated embeddings to disk.

# Extra CLI switches can be appended here, for example:
#   EXTRA_FLAGS=(--normalize)
EXTRA_FLAGS=()
if [[ "$SAVE_EMBEDDINGS" == true ]]; then
  EXTRA_FLAGS+=(--save-embeddings)
fi

set -euo pipefail

python -m detree.cli.test_score_knn \
  --database-path "$TRAIN_JSONL" \
  --test-dataset-path "$TEST_JSONL" \
  --model-name-or-path "$MODEL_NAME" \
  --log-file "$LOG_FILE" \
  --device-num "$DEVICE_NUM" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --max-length "$MAX_LENGTH" \
  --temperature "$TEMPERATURE" \
  --max-k "$MAX_K" \
  --min-layer "$MIN_LAYER" \
  --pooling "$POOLING" \
  --embedding-dim "$EMBEDDING_DIM" \
  --n-subquantizers "$N_SUBQUANTIZERS" \
  --n-bits "$N_BITS" \
  --savedir "$SAVE_DIR" \
  --name "$SAVE_NAME" \
  --pool-workers "$POOL_WORKERS" \
  "${EXTRA_FLAGS[@]}"
