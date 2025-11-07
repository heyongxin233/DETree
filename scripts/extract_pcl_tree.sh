#!/usr/bin/env bash
#
# Produce the two-level PCL tree required by the warm-up DETree stage.
#
DATA_ROOT="/path/to/RealBench"                  # Root directory that contains the RealBench datasets.
DATASET_NAME="all"                              # Dataset alias understood by detree.utils.dataset.load_datapath.
SPLIT="train"                                   # Split to scan (train/test).
OUTPUT_TXT="/path/to/RealBench/trees/pcl_tree.txt"  # Destination for the generated tree specification.
INCLUDE_ADVERSARIAL=true                        # Whether to scan adversarial sub-folders.
HAS_MIX=true                                    # Keep mixed human-machine sources when building the tree.

# Extra CLI switches can be appended here, for example:
#   EXTRA_FLAGS=(--some-flag value)
EXTRA_FLAGS=()

set -euo pipefail

python -m detree.cli.gen_tree \
  --path "$DATA_ROOT" \
  --dataset_name "$DATASET_NAME" \
  --mode "$SPLIT" \
  --tree_txt "$OUTPUT_TXT" \
  --adversarial "$INCLUDE_ADVERSARIAL" \
  --has_mix "$HAS_MIX" \
  "${EXTRA_FLAGS[@]}"
