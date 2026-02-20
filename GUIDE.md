# DeTree System Guide

A comprehensive guide for the DeTree AI content detection system, adapted to work with both text and image embeddings.

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Step-by-Step Setup](#step-by-step-setup)
5. [Training Guide](#training-guide)
6. [Inference](#inference)
7. [Quick Start: Training on Your Data](#quick-start-training-on-your-data)

---

## System Overview

DeTree is a **representation learning-based detection framework** for identifying AI-generated content. Originally designed for text detection, it has been extended to support **image detection** using CLIP embeddings.

### Key Concepts

| Component | Description |
|-----------|-------------|
| **Text Embeddings** | Generated from RoBERTa-large encoder, 1024 dimensions |
| **Image Embeddings** | CLIP embeddings (512 dimensions), projected to 1024 dimensions via CLIPProjector |
| **kNN Detection** | Uses k-nearest neighbors on embedding database to classify Real vs AI |
| **Unified Database** | Text and image embeddings can be merged for multi-modal detection |

### Detection Flow

```
Input (Text/Image)
       │
       ▼
┌─────────────────────────────────────────┐
│  Text Path              Image Path      │
│  ─────────              ──────────      │
│  RoBERTa Encoder        CLIP Encoder    │
│       │                      │          │
│       ▼                      ▼          │
│  Text Embedding         CLIP Embedding  │
│  (1024 dim)             (512 dim)       │
│       │                      │          │
│       │              CLIPProjector      │
│       │                      │          │
│       ▼                      ▼          │
│       └──────→ kNN Search ←──────┘      │
│                    │                    │
│                    ▼                    │
│            Real / AI Decision           │
└─────────────────────────────────────────┘
```

---

## Architecture

### Directory Structure

```
DeTree Test/
├── detree/                          # Main package
│   ├── cli/                         # Command-line tools
│   │   ├── train.py                 # Train text model (original)
│   │   ├── train_clip_projector.py  # Train CLIP→DeTree projector
│   │   ├── embeddings.py            # Generate text embeddings
│   │   ├── gen_image_embeddings.py  # Generate image embedding database
│   │   └── merge_databases.py       # Merge text + image databases
│   ├── model/
│   │   ├── clip_projector.py        # CLIP projection MLP
│   │   └── simclr.py                # Tree-based contrastive model
│   └── inference/
│       └── detector.py              # Unified inference (text + images)
│
├── Embeddings/Embeddings/Embeddings/ # Pre-computed CLIP embeddings
│   ├── AI_Diffusion/AI/Raw/         # AI-generated image embeddings
│   │   ├── Flux_1/                  # Flux model embeddings
│   │   ├── SDXL/                    # Stable Diffusion XL
│   │   └── ...                      # Other AI generators
│   ├── Real_VISION/Real/Raw/        # Real image embeddings
│   └── Real_ImageNet/Real/Raw/      # ImageNet real images
│
├── RealBench/embeddings/            # Pre-computed text databases
│   ├── mage_center10k.pt            # MAGE text embeddings (10k)
│   └── priori1_center10k.pt         # RealBench text embeddings (10k)
│
└── scripts/                         # Helper scripts
    ├── train_clip_projector.bat     # Windows training script
    ├── gen_image_emb.bat            # Generate image embeddings
    └── merge_db.bat                 # Merge databases
```

### Data Format

#### CLIP Embeddings (`.npy` files)
- Location: `Embeddings/Embeddings/Embeddings/`
- Format: NumPy array of shape `(512,)` (CLIP ViT-B/32 embeddings)
- Labels: Inferred from directory structure
  - `AI` in path → **AI-generated** (label 0)
  - `Real` in path → **Real/Human** (label 1)

#### Embedding Databases (`.pt` files)
- Format: PyTorch dictionary with keys:
  ```python
  {
      "embeddings": {layer_key: tensor(N, 1024)},  # L2-normalized
      "labels": tensor(N,),                        # 0=AI, 1=Human
      "ids": tensor(N,),                           # Unique identifiers
      "classes": ["llm", "human"]                  # Class names
  }
  ```

---

## Prerequisites

### 1. Install Dependencies

```bash
# Create environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Set PYTHONPATH

```bash
# Windows (PowerShell)
$env:PYTHONPATH = "$(pwd);$env:PYTHONPATH"

# Windows (CMD)
set PYTHONPATH=%CD%;%PYTHONPATH%

# Linux/Mac
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

### 3. Verify GPU Availability

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

---

## Step-by-Step Setup

### Step 1: Understand Your Data

Your current data layout:

```
Image Embeddings (CLIP, 512-dim):
  └── Embeddings/Embeddings/Embeddings/
      ├── AI_Diffusion/AI/Raw/     → AI images (Flux, SDXL, SD, etc.)
      ├── AI_Firefly/AI/Raw/       → Adobe Firefly AI images
      ├── AI_GAN/AI/Raw/           → GAN-generated images
      ├── AI_Midjourney/AI/Raw/    → Midjourney images
      ├── Real_ImageNet/Real/Raw/  → Real images (ImageNet)
      └── Real_VISION/Real/Raw/    → Real images (VISION dataset)

Text Embeddings (already processed, 1024-dim):
  └── RealBench/embeddings/
      ├── mage_center10k.pt        → MAGE text database
      └── priori1_center10k.pt     → RealBench text database
```

### Step 2: Check Available Text Database

The text databases in `RealBench/embeddings/` are already compressed to 10k entries. You can use either:
- `priori1_center10k.pt` - Recommended for training the CLIP projector
- `mage_center10k.pt` - Alternative database

To inspect a database:
```python
import torch
db = torch.load("RealBench/embeddings/priori1_center10k.pt")
print(f"Keys: {db.keys()}")
print(f"Embeddings shape: {db['embeddings'][23].shape}")  # (N, 1024)
print(f"Classes: {db['classes']}")
print(f"Available layers: {list(db['embeddings'].keys())}")
```

---

## Training Guide

### Overview: Three-Stage Pipeline

```
Stage 1: Train CLIP Projector
    Image embeddings (512d) → Aligned to text space (1024d)
    
Stage 2: Generate Image Database
    Project all images through trained projector
    
Stage 3: Merge Databases (Optional)
    Combine text + image for multi-modal detection
```

### Stage 1: Train the CLIP Projector

The CLIP projector is an MLP that maps 512-dimensional CLIP embeddings into the 1024-dimensional DeTree text embedding space.

**Architecture:**
```
Linear(512 → 1024) → GELU → LayerNorm(1024) → Linear(1024 → 1024)
```

**Training Objective:** Proxy Supervised Contrastive Loss
- Pulls projected image embeddings toward same-class text centroids
- Pushes away from opposite-class centroids

**Command:**
```bash
python -m detree.cli.train_clip_projector ^
    --embeddings-dir "Embeddings\Embeddings\Embeddings" ^
    --text-database "RealBench\embeddings\priori1_center10k.pt" ^
    --target-layer 23 ^
    --clip-dim 512 ^
    --output-dir "runs\clip_projector" ^
    --experiment-name "clip_align" ^
    --batch-size 256 ^
    --epochs 50 ^
    --lr 1e-3 ^
    --temperature 0.07 ^
    --num-centroids 1024 ^
    --normalize-input
```

**Key Parameters:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| `--embeddings-dir` | `Embeddings\Embeddings\Embeddings` | Root directory with `AI/` and `Real/` folders |
| `--text-database` | `RealBench\embeddings\priori1_center10k.pt` | Frozen text embedding database |
| `--target-layer` | `23` | Layer key in text database (usually 23 for RoBERTa-large) |
| `--clip-dim` | `512` | CLIP embedding dimension |
| `--epochs` | `50` | Training epochs |
| `--num-centroids` | `1024` | Text centroids sampled per batch |

**Output:**
```
runs/clip_projector/clip_align/
├── best/                  # Best checkpoint
│   └── clip_projector.pt
├── last/                  # Latest checkpoint
│   └── clip_projector.pt
├── config.yaml            # Training configuration
└── runs/                  # TensorBoard logs
```

### Stage 2: Generate Image Embedding Database

After training, project all CLIP embeddings through the projector:

```bash
python -m detree.cli.gen_image_embeddings ^
    --embeddings-dir "Embeddings\Embeddings\Embeddings" ^
    --projector-path "runs\clip_projector\clip_align\best" ^
    --target-layer 23 ^
    --output "databases\image_embeddings.pt" ^
    --batch-size 512 ^
    --normalize-input
```

**Output:** `databases/image_embeddings.pt`
- Contains all image embeddings projected to 1024 dimensions
- Same format as text databases

### Stage 3: Merge Databases (Optional)

For multi-modal detection, merge text and image databases:

```bash
python -m detree.cli.merge_databases ^
    --databases "RealBench\embeddings\priori1_center10k.pt" "databases\image_embeddings.pt" ^
    --target-layer 23 ^
    --output "databases\merged_multimodal.pt"
```

---

## Inference

### Using the Detector Class

```python
from detree.inference import Detector

# For text-only detection
detector = Detector(
    database_path="databases/image_embeddings.pt",
    model_name_or_path="FacebookAI/roberta-large",
    layer=23,
    top_k=10,
    threshold=0.5,
)

# Text prediction
results = detector.predict(["This is some text to analyze."])
for r in results:
    print(f"Label: {r.label}, P(Human): {r.probability_human:.3f}")
```

### For Image Detection

```python
from detree.inference import Detector
from pathlib import Path

detector = Detector(
    database_path="databases/image_embeddings.pt",
    model_name_or_path="FacebookAI/roberta-large",
    projector_path=Path("runs/clip_projector/clip_align/best"),
    layer=23,
    top_k=10,
    threshold=0.5,
)

# Predict on CLIP embedding files
image_results = detector.predict_images([
    "path/to/image1.npy",
    "path/to/image2.npy",
])
for r in image_results:
    print(f"{r.image_path}: {r.label} (Real: {r.probability_human:.3f})")
```

### Command-Line Inference

```bash
# Text inference
python example/infer.py ^
    --database-path "databases\image_embeddings.pt" ^
    --model-name-or-path "FacebookAI/roberta-large" ^
    --text "Your text here" ^
    --layer 23

# Image inference
python example/infer.py ^
    --database-path "databases\image_embeddings.pt" ^
    --model-name-or-path "FacebookAI/roberta-large" ^
    --projector-path "runs\clip_projector\clip_align\best" ^
    --image-dir "path\to\clip\embeddings" ^
    --layer 23
```

---

## Quick Start: Training on Your Data

### Your Current Setup

You have:
1. **Image embeddings**: `Embeddings/Embeddings/Embeddings/` (CLIP 512-dim `.npy` files)
2. **Text embeddings**: `RealBench/embeddings/priori1_center10k.pt` (pre-processed)

### Complete Training Pipeline

Run these commands in order:

#### 1. Create Output Directory
```bash
mkdir databases
```

#### 2. Train CLIP Projector
```bash
python -m detree.cli.train_clip_projector ^
    --embeddings-dir "Embeddings\Embeddings\Embeddings" ^
    --text-database "RealBench\embeddings\priori1_center10k.pt" ^
    --target-layer 23 ^
    --clip-dim 512 ^
    --output-dir "runs\clip_projector" ^
    --experiment-name "my_training" ^
    --batch-size 256 ^
    --epochs 50 ^
    --lr 1e-3 ^
    --temperature 0.07 ^
    --num-centroids 1024 ^
    --normalize-input
```

Expected training time: ~30-60 minutes on GPU, depending on dataset size.

#### 3. Generate Image Database
```bash
python -m detree.cli.gen_image_embeddings ^
    --embeddings-dir "Embeddings\Embeddings\Embeddings" ^
    --projector-path "runs\clip_projector\my_training\best" ^
    --target-layer 23 ^
    --output "databases\image_embeddings.pt" ^
    --batch-size 512 ^
    --normalize-input
```

#### 4. (Optional) Merge with Text Database
```bash
python -m detree.cli.merge_databases ^
    --databases "RealBench\embeddings\priori1_center10k.pt" "databases\image_embeddings.pt" ^
    --target-layer 23 ^
    --output "databases\merged_multimodal.pt"
```

#### 5. Test Inference
```python
# Quick test script
from detree.inference import Detector
from pathlib import Path

detector = Detector(
    database_path=Path("databases/image_embeddings.pt"),
    model_name_or_path="FacebookAI/roberta-large",
    projector_path=Path("runs/clip_projector/my_training/best"),
    layer=23,
    top_k=10,
    threshold=0.5,
)

# Test on some images
test_images = list(Path("Embeddings/Embeddings/Embeddings/AI_Diffusion/AI/Raw/Flux_1").glob("*.npy"))[:5]
results = detector.predict_images(test_images)
for r in results:
    print(f"{Path(r.image_path).name}: {r.label} (AI: {r.probability_ai:.3f})")
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `No .npy files found` | Check that embeddings directory has `AI/` and `Real/` folders in path |
| `Layer not found in database` | Use `--target-layer 23` (default for RoBERTa-large) |
| `CUDA out of memory` | Reduce `--batch-size` (try 128 or 64) |
| `Module not found` | Set `PYTHONPATH` to include project root |

### Verifying Data Labels

The system infers labels from directory structure. Ensure your embeddings follow this pattern:
```
.../AI/.../*.npy     → Labeled as AI (0)
.../Real/.../*.npy   → Labeled as Real (1)
```

### Checking Database Contents
```python
import torch
db = torch.load("databases/image_embeddings.pt")
labels = db['labels']
print(f"Total: {len(labels)}")
print(f"AI samples: {(labels == 0).sum()}")
print(f"Real samples: {(labels == 1).sum()}")
```

---

## Advanced Configuration

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 50 | Number of training epochs |
| `--lr` | 1e-3 | Initial learning rate |
| `--min-lr` | 1e-5 | Minimum learning rate (cosine decay target) |
| `--batch-size` | 256 | Batch size for training |
| `--temperature` | 0.07 | Contrastive loss temperature |
| `--num-centroids` | 1024 | Text centroids per batch |
| `--val-split` | 0.1 | Validation split ratio |
| `--warmup-steps` | 500 | Learning rate warmup steps |

### Inference Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--top-k` | 10 | Number of nearest neighbors for kNN |
| `--threshold` | 0.5 | Classification threshold (P(Human) ≥ threshold → Human) |
| `--layer` | 23 | Embedding layer to use from database |

---

## Summary

1. **System Purpose**: Detect AI-generated content (text and images) using kNN on embedding databases
2. **Image Support**: CLIP embeddings projected to DeTree text space via CLIPProjector
3. **Training Flow**: CLIP Projector → Image Database → (Optional) Merge
4. **Your Data**: Ready to train with existing embeddings in `Embeddings/` and text database in `RealBench/`
