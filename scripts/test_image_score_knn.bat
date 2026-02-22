@echo off
REM Evaluate image detection accuracy on pre-computed CLIP embeddings.
REM
REM Runs kNN search against the merged multimodal database and reports
REM overall accuracy + per-folder breakdown.
REM
REM Prerequisites:
REM   1. Merged database (from merge_db.bat)
REM   2. Trained CLIPProjector (from train_clip_projector.bat)
REM   3. Pre-computed CLIP embeddings (from gen_image_emb.bat)

setlocal

REM ──────────────────────────────────────────────────────────────────────
REM Configuration - adjust these paths as needed
REM ──────────────────────────────────────────────────────────────────────

REM Merged multimodal database (text + image)
set DATABASE=databases\merged_multimodal.pt

REM Trained CLIPProjector checkpoint directory
set PROJECTOR=Models\clip_projector\clip_align\best

REM Root directory of pre-computed CLIP .npy embeddings
set EMBEDDINGS_DIR=Embeddings


REM kNN neighbours to retrieve
set TOP_K=10

REM P(Real) decision threshold (0.5 = balanced)
set THRESHOLD=0.5

REM Batch size for projection
set BATCH_SIZE=256

REM Optional: save a JSON report (leave empty to skip)
set OUTPUT=results\image_score_knn_report.json

REM ──────────────────────────────────────────────────────────────────────
REM Run evaluation
REM ──────────────────────────────────────────────────────────────────────

set OUTPUT_ARG=
if not "%OUTPUT%"=="" set OUTPUT_ARG=--output "%OUTPUT%"

python -m detree.cli.test_image_score_knn ^
    --database "%DATABASE%" ^
    --projector "%PROJECTOR%" ^
    --embeddings-dir "%EMBEDDINGS_DIR%" ^
    --top-k %TOP_K% ^
    --threshold %THRESHOLD% ^
    --batch-size %BATCH_SIZE% ^
    %OUTPUT_ARG%

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Evaluation failed. See output above.
    exit /b %ERRORLEVEL%
)

echo.
if not "%OUTPUT%"=="" (
    echo Evaluation complete. Report saved to: %OUTPUT%
) else (
    echo Evaluation complete.
)

endlocal
