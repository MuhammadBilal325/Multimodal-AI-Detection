@echo off
REM Train the CLIP → DeTree alignment projector (Windows version)
REM
REM This script trains an MLP to project CLIP image embeddings into the
REM DeTree text embedding space, enabling unified kNN-based AI detection
REM across both modalities.

setlocal

REM ──────────────────────────────────────────────────────────────────────
REM Configuration - adjust these paths as needed
REM ──────────────────────────────────────────────────────────────────────

REM Root directory containing CLIP embeddings (should have AI/ and Real/ subfolders)
set EMBEDDINGS_DIR=Embeddings\Embeddings\Embeddings

REM Path to the frozen DeTree text database
set TEXT_DATABASE=databases\text_compressed.pt

REM Layer in the text database to align to (check your database keys)
set TARGET_LAYER=23

REM CLIP embedding dimension (512 for ViT-B/32, 768 for ViT-L/14)
set CLIP_DIM=512

REM Output directory
set OUTPUT_DIR=runs\clip_projector
set EXPERIMENT_NAME=clip_align

REM Training hyperparameters
set BATCH_SIZE=256
set EPOCHS=50
set LR=1e-3
set TEMPERATURE=0.07
set NUM_CENTROIDS=1024

REM ──────────────────────────────────────────────────────────────────────
REM Run training
REM ──────────────────────────────────────────────────────────────────────

python -m detree.cli.train_clip_projector ^
    --embeddings-dir "%EMBEDDINGS_DIR%" ^
    --text-database "%TEXT_DATABASE%" ^
    --target-layer %TARGET_LAYER% ^
    --clip-dim %CLIP_DIM% ^
    --output-dir "%OUTPUT_DIR%" ^
    --experiment-name "%EXPERIMENT_NAME%" ^
    --batch-size %BATCH_SIZE% ^
    --epochs %EPOCHS% ^
    --lr %LR% ^
    --temperature %TEMPERATURE% ^
    --num-centroids %NUM_CENTROIDS% ^
    --normalize-input

echo.
echo Training complete!
echo Best checkpoint saved to: %OUTPUT_DIR%\%EXPERIMENT_NAME%\best

endlocal
