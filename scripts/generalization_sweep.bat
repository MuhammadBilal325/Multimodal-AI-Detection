@echo off
REM Generalization sweep for CLIPProjector.
REM
REM Trains one CLIPProjector per non-empty combination of AI embedding
REM datasets, then evaluates every trained model on every combination.
REM Produces a full train x eval accuracy matrix saved as JSON.
REM
REM With 4 AI datasets this runs 15 training jobs x 15 eval combos = 225
REM evaluations.  Expect this to take several hours depending on GPU and
REM epoch count.  Reduce --epochs for a quick smoke-test.
REM
REM Prerequisites:
REM   1. Pre-computed CLIP .npy embeddings (from gen_image_emb.bat)
REM   2. Frozen DeTree text database (from the DeTree training pipeline)
REM
REM The val split (--val-split 0.2) uses a fixed seed (--seed 42), so the
REM same 20%% of files from every directory are always held out for eval,
REM regardless of which training combo ran.  No training data contaminates
REM any evaluation.

setlocal

REM ──────────────────────────────────────────────────────────────────────
REM Paths  --  adjust as needed
REM ──────────────────────────────────────────────────────────────────────

REM Root directory that contains the AI_* and Real_* embedding subfolders
set EMBEDDINGS_ROOT=Embeddings\Embeddings\Embeddings

REM Frozen DeTree text database used as training centroids
set TEXT_DATABASE=databases\text_compressed.pt

REM Layer key in the text database the projector is trained to align to
set TARGET_LAYER=23

REM Database used for kNN evaluation.
REM Leave blank to use TEXT_DATABASE (evaluates against pure text space).
REM Set to a merged multimodal database to include image entries in the index.
set EVAL_DATABASE=

REM Output root for all model checkpoints and result files
set OUTPUT_DIR=runs\generalization_sweep

REM ──────────────────────────────────────────────────────────────────────
REM Training hyper-parameters
REM ──────────────────────────────────────────────────────────────────────

set EPOCHS=30
set BATCH_SIZE=256
set LR=1e-3
set VAL_SPLIT=0.2
set SEED=42
set TOP_K=10
set THRESHOLD=0.5
set NUM_WORKERS=4

REM Metric shown in the printed summary matrix after the sweep completes.
REM Choices: accuracy  f1  roc_auc  recall_AI  recall_Real
set REPORT_METRIC=roc_auc

REM ──────────────────────────────────────────────────────────────────────
REM Build the command
REM ──────────────────────────────────────────────────────────────────────

set BASE_CMD=python -m detree.cli.generalization_sweep ^
    --embeddings-root "%EMBEDDINGS_ROOT%" ^
    --text-database   "%TEXT_DATABASE%"   ^
    --target-layer    %TARGET_LAYER%      ^
    --output-dir      "%OUTPUT_DIR%"      ^
    --epochs          %EPOCHS%            ^
    --batch-size      %BATCH_SIZE%        ^
    --lr              %LR%                ^
    --val-split       %VAL_SPLIT%         ^
    --seed            %SEED%              ^
    --top-k           %TOP_K%             ^
    --threshold       %THRESHOLD%         ^
    --num-workers     %NUM_WORKERS%        ^
    --report-metric   %REPORT_METRIC%

REM Append --eval-database only when explicitly set
if not "%EVAL_DATABASE%"=="" (
    set BASE_CMD=%BASE_CMD% ^
    --eval-database "%EVAL_DATABASE%"
)

%BASE_CMD%

echo.
echo Sweep complete.  Matrix saved to: %OUTPUT_DIR%\generalization_matrix.json

endlocal
