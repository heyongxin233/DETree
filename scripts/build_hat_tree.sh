#!/usr/bin/env bash
#
# Build a HAT tree from warm-up embeddings.
#
# This script first generates similarity matrices with
# ``python -m detree.cli.similarity_matrix`` and then runs
# ``python -m detree.cli.hierarchical_clustering`` to export the tree.
#
# Adjust the variables below to match your workspace before executing it.
#
DATABASE_PATH="/path/to/RealBench/pipeline/database/stage1_embeddings.pt"  # Warm-up embedding database (.pt).
WORK_DIR="/path/to/RealBench/pipeline/hat"                                # Directory to store similarity and tree artefacts.
LAYERS=(18)                                                               # Encoder layers to process.
TREE_PRIORI=1                                                             # Prior selection passed to hierarchical_clustering.
SAVE_MAX_DEPTH=5                                                          # Maximum depth to render in the table PDF.
END_SCORE=0.1                                                             # Early-stop threshold for tree merging.

# Optional extras appended to the hierarchical clustering invocation.
EXTRA_CLUSTER_FLAGS=(
  # --randmo-filter
)

set -euo pipefail

SIM_DIR="$WORK_DIR/similarity"
TREE_DIR="$WORK_DIR/tree"
mkdir -p "$SIM_DIR" "$TREE_DIR"

python -m detree.cli.similarity_matrix \
  --database "$DATABASE_PATH" \
  --output-dir "$SIM_DIR" \
  --layers "${LAYERS[@]}"

for layer in "${LAYERS[@]}"; do
  SIM_TXT="$SIM_DIR/similarity_layer_${layer}.txt"
  TREE_TXT="$TREE_DIR/tree_layer_${layer}.txt"
  TREE_TABLE="$TREE_DIR/tree_layer_${layer}.pdf"

  python -m detree.cli.hierarchical_clustering \
    --file-path "$SIM_TXT" \
    --priori "$TREE_PRIORI" \
    --save-txt-path "$TREE_TXT" \
    --save-table-path "$TREE_TABLE" \
    --save-max-dep "$SAVE_MAX_DEPTH" \
    --end-score "$END_SCORE" \
    "${EXTRA_CLUSTER_FLAGS[@]}"
done

echo "HAT tree artefacts saved under $TREE_DIR"
