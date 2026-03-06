@echo off
REM
REM Compress the image embedding database into clustered prototypes.
REM
REM The command below runs ``python -m detree.cli.database`` with
REM descriptive variables so you can tweak the run directly in this file.
REM
REM Update the paths and hyper-parameters before executing the script.
REM

SET DATABASE_PATH=databases\image_embeddings.pt
SET OUTPUT_PATH=databases\image_embeddings_comp.pt
SET NUM_CLUSTERS=1156
SET EMBED_DIM=1024
SET NUM_ITERATIONS=100
SET NUM_GPUS=1
SET HUMAN_CLASS_NAME=human

python -m detree.cli.database ^
  --database "%DATABASE_PATH%" ^
  --output "%OUTPUT_PATH%" ^
  --clusters "%NUM_CLUSTERS%" ^
  --dimension "%EMBED_DIM%" ^
  --iterations "%NUM_ITERATIONS%" ^
  --gpus "%NUM_GPUS%" ^
  --human-class-name "%HUMAN_CLASS_NAME%"
