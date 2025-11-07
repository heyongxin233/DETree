#!/usr/bin/env bash
#
# Merge a LoRA adapter into its backbone and save a standalone checkpoint.
#
BASE_MODEL="FacebookAI/roberta-large"          # Backbone encoder identifier.
ADAPTER_PATH="/path/to/RealBench/runs/detree_stage1/last"  # Directory containing the LoRA adapter weights.
OUTPUT_DIR="/path/to/RealBench/models/detree_stage1"       # Destination for the merged checkpoint.
USE_SAFE_TENSORS=true                           # Save weights using safetensors (set false to disable).

set -euo pipefail

# Extra CLI switches can be appended here, for example:
#   EXTRA_FLAGS=(--no-safe-serialization)
EXTRA_FLAGS=()
if [[ "$USE_SAFE_TENSORS" != true ]]; then
  EXTRA_FLAGS+=(--no-safe-serialization)
fi

python -m detree.cli.merge_lora \
  --base-model "$BASE_MODEL" \
  --adapter-path "$ADAPTER_PATH" \
  --output-dir "$OUTPUT_DIR" \
  "${EXTRA_FLAGS[@]}"
