@echo off
REM Generate image embedding database using the trained CLIP projector (Windows version)

setlocal

REM ──────────────────────────────────────────────────────────────────────
REM Configuration - adjust these paths as needed
REM ──────────────────────────────────────────────────────────────────────

REM Root directory containing CLIP embeddings
set EMBEDDINGS_DIR=Embeddings\Embeddings\Embeddings

REM Path to the trained projector checkpoint
set PROJECTOR_PATH=runs\clip_projector\clip_align\best

REM Target layer (must match what you used during training)
set TARGET_LAYER=23

REM Output path for the image embedding database
set OUTPUT=databases\image_embeddings.pt

REM ──────────────────────────────────────────────────────────────────────
REM Run embedding generation
REM ──────────────────────────────────────────────────────────────────────

python -m detree.cli.gen_image_embeddings ^
    --embeddings-dir "%EMBEDDINGS_DIR%" ^
    --projector-path "%PROJECTOR_PATH%" ^
    --target-layer %TARGET_LAYER% ^
    --output "%OUTPUT%" ^
    --batch-size 512 ^
    --normalize-input

echo.
echo Image embedding database saved to: %OUTPUT%

endlocal
