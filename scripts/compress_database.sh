#!/usr/bin/env bash
#
# Compress an embedding database into clustered prototypes.
#
# The command below expands to ``python -m detree.cli.database`` with
# descriptive variables so you can tweak the run directly in this file.
#
# Update the paths and hyper-parameters before executing the script.
#
DATABASE_PATH="/path/to/RealBench/pipeline/database/detree_embeddings.pt"   # Input embedding database (.pt).
OUTPUT_PATH="/path/to/RealBench/pipeline/database/detree_embeddings_comp.pt"  # Destination for the clustered database.
NUM_CLUSTERS=10000
EMBED_DIM=1024
NUM_ITERATIONS=100
NUM_GPUS=1
HUMAN_CLASS_NAME="human"

set -euo pipefail

python -m detree.cli.database \
  --database "$DATABASE_PATH" \
  --output "$OUTPUT_PATH" \
  --clusters "$NUM_CLUSTERS" \
  --dimension "$EMBED_DIM" \
  --iterations "$NUM_ITERATIONS" \
  --gpus "$NUM_GPUS" \
  --human-class-name "$HUMAN_CLASS_NAME"
