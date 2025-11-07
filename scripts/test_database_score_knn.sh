#!/usr/bin/env bash
#
# RealBench kNN evaluation helper.
#
# Edit the variables below so they point to your exported embedding database,
# evaluation JSONL, and merged DETree checkpoint.  Every option maps directly to
# ``python -m detree.cli.test_database_score_knn``; keep them here so you do not
# have to remember CLI flags when rerunning experiments.
#
# Required paths -------------------------------------------------------------
DATABASE_PATH="/path/to/RealBench/embeddings/final_database.pt"
TEST_JSONL="/path/to/RealBench/jsonl/test.jsonl"
MODEL_NAME="/path/to/RealBench/models/detree_final"
LOG_FILE="/path/to/RealBench/logs/knn_eval.txt"
#
# Hardware configuration -----------------------------------------------------
DEVICE_NUM=1          # Number of GPUs visible to Lightning Fabric
BATCH_SIZE=32         # Evaluation batch size
NUM_WORKERS=8         # DataLoader workers for reading JSONL
MAX_LENGTH=512        # Tokenisation max length
#
# Scoring controls -----------------------------------------------------------
TEMPERATURE=0.05      # Softmax temperature for vote aggregation
MAX_K=51              # Highest k to probe when evaluating
POOLING="max"          # Options: max, average, cls
EMBEDDING_DIM=1024    # Dimension of each embedding vector
POOL_WORKERS=16       # CPU workers for multiprocessing kNN scoring
# ---------------------------------------------------------------------------
# Uncomment the line below to forward extra CLI switches, for example:
#   EXTRA_FLAGS=(--normalize --some-flag value)
EXTRA_FLAGS=()

set -euo pipefail

python -m detree.cli.test_database_score_knn \
  --database-path "$DATABASE_PATH" \
  --test-dataset-path "$TEST_JSONL" \
  --model-name-or-path "$MODEL_NAME" \
  --log-file "$LOG_FILE" \
  --device-num "$DEVICE_NUM" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --max-length "$MAX_LENGTH" \
  --temperature "$TEMPERATURE" \
  --max-k "$MAX_K" \
  --pooling "$POOLING" \
  --embedding-dim "$EMBEDDING_DIM" \
  --pool-workers "$POOL_WORKERS" \
  "${EXTRA_FLAGS[@]}"
