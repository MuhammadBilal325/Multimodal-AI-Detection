#!/bin/bash
# Train the CLIP → DeTree alignment projector.
#
# This script trains an MLP to project CLIP image embeddings into the
# DeTree text embedding space, enabling unified kNN-based AI detection
# across both modalities.
#
# Prerequisites:
#   1. Pre-computed CLIP embeddings (.npy or .npz files) organised in
#      a directory structure with AI/ and Real/ folders.
#   2. A frozen DeTree text database (.pt file).
#
# Usage:
#   bash scripts/train_clip_projector.sh

set -e

# ──────────────────────────────────────────────────────────────────────
# Configuration - adjust these paths as needed
# ──────────────────────────────────────────────────────────────────────

# Root directory containing CLIP embeddings (should have AI/ and Real/ subfolders)
EMBEDDINGS_DIR="Embeddings/Embeddings/Embeddings"

# Path to the frozen DeTree text database
TEXT_DATABASE="databases/text_compressed.pt"

# Layer in the text database to align to (check your database keys)
TARGET_LAYER=23

# CLIP embedding dimension (512 for ViT-B/32, 768 for ViT-L/14)
CLIP_DIM=512

# Output directory
OUTPUT_DIR="runs/clip_projector"
EXPERIMENT_NAME="clip_align"

# Training hyperparameters
BATCH_SIZE=256
EPOCHS=50
LR=1e-3
TEMPERATURE=0.07
NUM_CENTROIDS=1024

# ──────────────────────────────────────────────────────────────────────
# Run training
# ──────────────────────────────────────────────────────────────────────

python -m detree.cli.train_clip_projector \
    --embeddings-dir "$EMBEDDINGS_DIR" \
    --text-database "$TEXT_DATABASE" \
    --target-layer "$TARGET_LAYER" \
    --clip-dim "$CLIP_DIM" \
    --output-dir "$OUTPUT_DIR" \
    --experiment-name "$EXPERIMENT_NAME" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --temperature "$TEMPERATURE" \
    --num-centroids "$NUM_CENTROIDS" \
    --normalize-input

echo ""
echo "Training complete!"
echo "Best checkpoint saved to: $OUTPUT_DIR/$EXPERIMENT_NAME/best"
