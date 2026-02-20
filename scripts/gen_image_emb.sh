#!/bin/bash
# Generate image embedding database using the trained CLIP projector.
#
# Projects all CLIP embeddings through the trained MLP and saves them
# in the DeTree database format (.pt).
#
# Prerequisites:
#   1. A trained CLIPProjector checkpoint (from train_clip_projector.sh)
#   2. Pre-computed CLIP embeddings (.npy or .npz files)
#
# Usage:
#   bash scripts/gen_image_emb.sh

set -e

# ──────────────────────────────────────────────────────────────────────
# Configuration - adjust these paths as needed
# ──────────────────────────────────────────────────────────────────────

# Root directory containing CLIP embeddings
EMBEDDINGS_DIR="Embeddings/Embeddings/Embeddings"

# Path to the trained projector checkpoint
PROJECTOR_PATH="runs/clip_projector/clip_align/best"

# Target layer (must match what you used during training)
TARGET_LAYER=23

# Output path for the image embedding database
OUTPUT="databases/image_embeddings.pt"

# ──────────────────────────────────────────────────────────────────────
# Run embedding generation
# ──────────────────────────────────────────────────────────────────────

python -m detree.cli.gen_image_embeddings \
    --embeddings-dir "$EMBEDDINGS_DIR" \
    --projector-path "$PROJECTOR_PATH" \
    --target-layer "$TARGET_LAYER" \
    --output "$OUTPUT" \
    --batch-size 512 \
    --normalize-input

echo ""
echo "Image embedding database saved to: $OUTPUT"
