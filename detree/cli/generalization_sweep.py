"""Generalization sweep for CLIPProjector training.

Trains a ``CLIPProjector`` on every non-empty combination of AI embedding
datasets, then evaluates each trained model against every combination of AI
datasets.  The result is a full train-combo × eval-combo accuracy matrix that
reveals how well representations learned on a subset of AI sources transfer to
unseen sources.

------------------------------------------------------------------------------
Directory layout expected under ``--embeddings-root``::

    <root>/
        AI_Diffusion/AI/Raw/**/*.npy
        AI_Firefly/AI/Raw/**/*.npy
        AI_GAN/AI/Raw/**/*.npy
        AI_Midjourney/AI/Raw/**/*.npy
        Real_ImageNet/Real/Raw/**/*.npy
        Real_VISION/Real/Raw/**/*.npy

Any subfolder whose name starts with ``AI_`` is treated as an AI dataset;
any subfolder starting with ``Real_`` is treated as a Real dataset.

------------------------------------------------------------------------------
Train / val split strategy (mirrors the ``collect_files`` notebook function):

* For each AI dir in the training combo: collect all ``.npy`` files under
  ``<dir>/AI/<raw-subdir>/``, then do a ``train_test_split`` with the given
  ``--val-split`` fraction and fixed ``--seed``.
* Collect Real files from every Real dir, then for each Real dir take
  ``len(ai_train_files) // len(real_dirs)`` files from the train split and
  ``len(ai_val_files) // len(real_dirs)`` files from the val split, so that
  the overall AI:Real ratio stays 1:1.

Because the seed is held constant across the whole sweep, the val files for
any given directory are *always the same set*, regardless of which training
combo included that directory.  This guarantees the model is never evaluated
on files it was trained on.

------------------------------------------------------------------------------
Typical usage::

    python -m detree.cli.generalization_sweep \\
        --embeddings-root  Embeddings/Embeddings/Embeddings \\
        --text-database    databases/text_compressed.pt \\
        --target-layer     23 \\
        --output-dir       runs/generalization_sweep \\
        --epochs           30
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from detree.model.clip_projector import CLIPProjector
from detree.cli.train_clip_projector import (
    proxy_contrastive_loss,
    _load_text_centroids,
)
from detree.utils.index import Indexer


# ======================================================================
# File-list dataset  (replaces directory-scanning CLIPEmbeddingDataset)
# ======================================================================

class FileListDataset(Dataset):
    """Dataset built from an explicit list of (path, label_int) pairs.

    Args:
        samples:         List of ``(path, label)`` where label is 0=AI, 1=Real.
        normalize_input: L2-normalise each loaded embedding before returning.
    """

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        normalize_input: bool = True,
    ) -> None:
        self.samples = samples
        self.normalize_input = normalize_input

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        emb = _load_npy(path)
        t = torch.from_numpy(emb)
        if self.normalize_input:
            t = F.normalize(t, dim=0)
        return t, label


# ======================================================================
# Low-level helpers
# ======================================================================

def _load_npy(path: str) -> np.ndarray:
    if path.endswith(".npz"):
        data = np.load(path)
        for key in ("embedding", "emb", "arr_0"):
            if key in data:
                return data[key].astype(np.float32).flatten()
        return data[list(data.keys())[0]].astype(np.float32).flatten()
    return np.load(path).astype(np.float32).flatten()


def _walk_files(folder: str) -> List[str]:
    """Collect all .npy/.npz files under *folder* (sorted for reproducibility)."""
    found: List[str] = []
    for dirpath, _, filenames in os.walk(folder):
        for fname in sorted(filenames):
            if fname.endswith(".npy") or fname.endswith(".npz"):
                found.append(os.path.join(dirpath, fname))
    return found


def _combo_name(dirs: Sequence[Path]) -> str:
    """Short identifier for a combination of dataset dirs."""
    return "+".join(sorted(d.name for d in dirs))


# ======================================================================
# collect_files – adapted from notebook function for .npy embeddings
# ======================================================================

def collect_files(
    ai_dirs: Sequence[Path],
    real_dirs: Sequence[Path],
    seed: int,
    val_split: float,
    raw_subdir: str = "Raw",
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """Build train / val file lists with balanced AI : Real ratio.

    Mirrors the ``collect_files`` notebook helper exactly:

    * For each AI dir: walk ``<dir>/AI/<raw_subdir>/``, split 80/20 (or
      whatever ``val_split`` is), add to train/val pools.
    * For each Real dir: walk ``<dir>/Real/<raw_subdir>/``, split the same
      way, then take only ``n_ai_train // n_real_dirs`` train files and the
      same proportion for val, so that AI:Real stays balanced.
    * Shuffle both resulting lists using *seed*.

    Returns:
        ``(train_samples, val_samples)`` where each sample is
        ``(absolute_path, label_int)`` with label 0=AI, 1=Real.
    """
    train_files: List[str] = []
    val_files: List[str] = []
    train_labels: List[int] = []
    val_labels: List[int] = []

    # ── AI dirs ──────────────────────────────────────────────────────
    for ai_dir in ai_dirs:
        folder = os.path.join(str(ai_dir), "AI", raw_subdir)
        if not os.path.isdir(folder):
            print(f"  [skip] {folder} does not exist")
            continue
        files = _walk_files(folder)
        if not files:
            print(f"  [skip] no .npy files under {folder}")
            continue
        t_f, v_f = train_test_split(
            files, test_size=val_split, random_state=seed
        )
        train_files.extend(t_f)
        val_files.extend(v_f)
        train_labels.extend([0] * len(t_f))
        val_labels.extend([0] * len(v_f))
        print(f"    AI  {ai_dir.name:30s} train={len(t_f):>6,}  val={len(v_f):>6,}")

    n_real_dirs = max(len(real_dirs), 1)
    needed_train = len(train_files) // n_real_dirs
    needed_val   = len(val_files)   // n_real_dirs

    # ── Real dirs ─────────────────────────────────────────────────────
    for real_dir in real_dirs:
        folder = os.path.join(str(real_dir), "Real", raw_subdir)
        if not os.path.isdir(folder):
            print(f"  [skip] {folder} does not exist")
            continue
        files = _walk_files(folder)
        if not files:
            print(f"  [skip] no .npy files under {folder}")
            continue
        t_f, v_f = train_test_split(
            files, test_size=val_split, random_state=seed
        )
        # Balance: take only as many as needed
        t_f = t_f[:needed_train]
        v_f = v_f[:needed_val]
        train_files.extend(t_f)
        val_files.extend(v_f)
        train_labels.extend([1] * len(t_f))
        val_labels.extend([1] * len(v_f))
        print(f"    Real {real_dir.name:30s} train={len(t_f):>6,}  val={len(v_f):>6,}")

    # Shuffle preserving label correspondence
    rng = random.Random(seed)
    train_pairs = list(zip(train_files, train_labels))
    val_pairs   = list(zip(val_files, val_labels))
    rng.shuffle(train_pairs)
    rng.shuffle(val_pairs)

    return train_pairs, val_pairs


# ======================================================================
# Training
# ======================================================================

def _train_projector(
    train_pairs: List[Tuple[str, int]],
    val_pairs: List[Tuple[str, int]],
    centroids: torch.Tensor,
    centroid_labels: torch.Tensor,
    output_dir: Path,
    *,
    clip_dim: int,
    target_dim: int,
    hidden_dim: Optional[int],
    batch_size: int,
    num_workers: int,
    epochs: int,
    lr: float,
    min_lr: float,
    weight_decay: float,
    warmup_steps: int,
    temperature: float,
    num_centroids: int,
    seed: int,
    normalize_input: bool,
    device: torch.device,
) -> Path:
    """Train a CLIPProjector on *train_pairs*, validate on *val_pairs*.

    Saves checkpoints under *output_dir*; returns the path to the ``best``
    checkpoint (lowest validation loss).
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    train_dataset = FileListDataset(train_pairs, normalize_input=normalize_input)
    val_dataset   = FileListDataset(val_pairs,   normalize_input=normalize_input)

    # Auto-detect clip_dim
    sample_emb, _ = train_dataset[0]
    detected_clip_dim = sample_emb.shape[0]
    if clip_dim != detected_clip_dim:
        clip_dim = detected_clip_dim

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = CLIPProjector(
        clip_dim=clip_dim,
        target_dim=target_dim,
        hidden_dim=hidden_dim,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = max(epochs * len(train_loader) - warmup_steps, 1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, total_steps, eta_min=min_lr,
    )

    # Pre-compute centroid indices for balanced sampling
    ai_indices   = (centroid_labels == 0).nonzero(as_tuple=True)[0]
    real_indices = (centroid_labels == 1).nonzero(as_tuple=True)[0]

    # Class prototypes for val accuracy
    ai_proto   = F.normalize(centroids[ai_indices].mean(dim=0, keepdim=True), dim=-1)
    real_proto = F.normalize(centroids[real_indices].mean(dim=0, keepdim=True), dim=-1)
    prototypes = torch.cat([ai_proto, real_proto], dim=0)  # (2, D)

    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    global_step = 0
    history: List[Dict] = []

    for epoch in range(epochs):
        # ── train ────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for i, (clip_embs, labels) in enumerate(
            tqdm(train_loader, desc=f"  Epoch {epoch+1}/{epochs}", leave=False)
        ):
            global_step += 1
            if global_step <= warmup_steps:
                warmup_lr = lr * global_step / max(warmup_steps, 1)
                for pg in optimizer.param_groups:
                    pg["lr"] = warmup_lr

            clip_embs = clip_embs.to(device)
            labels    = labels.to(device)
            projected = model(clip_embs, normalize=True)

            n_per_class = min(num_centroids // 2, len(ai_indices), len(real_indices))
            sampled_ai   = ai_indices[torch.randperm(len(ai_indices), device=device)[:n_per_class]]
            sampled_real = real_indices[torch.randperm(len(real_indices), device=device)[:n_per_class]]
            sampled_idx  = torch.cat([sampled_ai, sampled_real])
            loss = proxy_contrastive_loss(
                projected, labels,
                centroids[sampled_idx], centroid_labels[sampled_idx],
                temperature=temperature,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if global_step > warmup_steps:
                scheduler.step()

            epoch_loss = (epoch_loss * i + loss.item()) / (i + 1)

        # ── val ──────────────────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for clip_embs, labels in val_loader:
                clip_embs = clip_embs.to(device)
                labels    = labels.to(device)
                projected = model(clip_embs, normalize=True)
                loss = proxy_contrastive_loss(
                    projected, labels, centroids, centroid_labels,
                    temperature=temperature,
                )
                val_loss_sum += loss.item() * clip_embs.size(0)
                sim = projected @ prototypes.T
                correct += (sim.argmax(dim=1) == labels).sum().item()
                total   += labels.size(0)

        val_loss = val_loss_sum / max(total, 1)
        val_acc  = correct / max(total, 1) * 100.0
        history.append({"epoch": epoch, "train_loss": epoch_loss,
                        "val_loss": val_loss, "val_acc": val_acc})
        print(f"    epoch {epoch+1:>3}  train={epoch_loss:.4f}  val={val_loss:.4f}  acc={val_acc:.1f}%")

        model.save_pretrained(str(output_dir / "last"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(str(output_dir / "best"))

    with open(output_dir / "history.json", "w") as fh:
        json.dump(history, fh, indent=2)

    print(f"    ✓ Best val_loss={best_val_loss:.4f}  saved → {output_dir/'best'}")
    return output_dir / "best"


# ======================================================================
# Evaluation
# ======================================================================

def _evaluate_on_samples(
    samples: List[Tuple[str, int]],
    projector: CLIPProjector,
    index: Indexer,
    *,
    top_k: int,
    threshold: float,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    """Run kNN prediction on *samples* using a pre-built *index*."""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    all_true:  List[int]   = []
    all_pred:  List[int]   = []
    all_score: List[float] = []

    projector.eval()
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            paths  = [s[0] for s in batch]
            labels = [s[1] for s in batch]

            raw = [_load_npy(p) for p in paths]
            t   = torch.from_numpy(np.stack(raw, axis=0))
            t   = F.normalize(t, dim=-1).to(device)
            proj = projector(t, normalize=True).cpu().numpy().astype(np.float32)

            results = index.search_knn(
                proj, top_k,
                index_batch_size=max(1, min(top_k, 128)),
            )
            for true_label, (_ids, scores, nn_labels) in zip(labels, results):
                scores_t = torch.from_numpy(np.asarray(scores))
                weights  = torch.softmax(scores_t, dim=0)
                label_t  = torch.tensor(nn_labels, dtype=torch.float32)
                p_real   = float(torch.dot(weights, label_t).clamp(0.0, 1.0))
                pred     = 1 if p_real >= threshold else 0
                all_true.append(true_label)
                all_pred.append(pred)
                all_score.append(p_real)

    y_true  = np.array(all_true,  dtype=np.int64)
    y_pred  = np.array(all_pred,  dtype=np.int64)
    y_score = np.array(all_score, dtype=np.float32)

    acc = float(accuracy_score(y_true, y_pred))
    f1  = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auc = float("nan")

    ai_mask   = y_true == 0
    real_mask = y_true == 1
    recall_ai   = float(np.mean(y_pred[ai_mask]   == 0)) if ai_mask.any()   else float("nan")
    recall_real = float(np.mean(y_pred[real_mask] == 1)) if real_mask.any() else float("nan")

    return {
        "n": len(y_true),
        "n_ai": int(ai_mask.sum()),
        "n_real": int(real_mask.sum()),
        "accuracy": acc,
        "f1": f1,
        "roc_auc": auc,
        "recall_AI": recall_ai,
        "recall_Real": recall_real,
    }


# ======================================================================
# Discover dataset dirs
# ======================================================================

def _discover_dirs(
    embeddings_root: Path,
    ai_names: Optional[List[str]],
    real_names: Optional[List[str]],
) -> Tuple[List[Path], List[Path]]:
    """Auto-discover AI_* and Real_* subdirs unless names are provided."""
    all_subdirs = [d for d in sorted(embeddings_root.iterdir()) if d.is_dir()]

    if ai_names:
        ai_dirs = [embeddings_root / n for n in ai_names]
    else:
        ai_dirs = [d for d in all_subdirs if d.name.startswith("AI_")]

    if real_names:
        real_dirs = [embeddings_root / n for n in real_names]
    else:
        real_dirs = [d for d in all_subdirs if d.name.startswith("Real_")]

    missing = [str(d) for d in ai_dirs + real_dirs if not d.is_dir()]
    if missing:
        raise FileNotFoundError(f"Dataset directories not found: {missing}")

    return ai_dirs, real_dirs


# ======================================================================
# Main sweep
# ======================================================================

def run_sweep(args: argparse.Namespace) -> None:
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Discover dataset directories ─────────────────────────────────
    ai_dirs, real_dirs = _discover_dirs(
        args.embeddings_root,
        args.ai_datasets,
        args.real_datasets,
    )
    print(f"\nAI datasets  ({len(ai_dirs)}): {[d.name for d in ai_dirs]}")
    print(f"Real datasets({len(real_dirs)}): {[d.name for d in real_dirs]}")

    # ── All non-empty subsets of AI dirs ─────────────────────────────
    all_combos: List[Tuple[Path, ...]] = []
    for r in range(1, len(ai_dirs) + 1):
        for combo in combinations(ai_dirs, r):
            all_combos.append(combo)
    print(f"\n{len(all_combos)} training combinations × "
          f"{len(all_combos)} eval combinations = "
          f"{len(all_combos)**2} total evaluations\n")

    # ── Load frozen text database ─────────────────────────────────────
    print(f"Loading text database from {args.text_database} …")
    centroids, centroid_labels = _load_text_centroids(
        args.text_database, args.target_layer
    )
    target_dim = centroids.shape[1]
    centroids       = F.normalize(centroids, dim=-1).to(device)
    centroid_labels = centroid_labels.to(device)
    print(f"  dim={target_dim}  layer={args.target_layer}  "
          f"n={centroids.shape[0]} ({(centroid_labels==0).sum()} AI, "
          f"{(centroid_labels==1).sum()} Real)")

    # ── Build eval database for kNN (resolve layer) ───────────────────
    print(f"\nLoading kNN database from {args.eval_database or args.text_database} …")
    eval_db_path = Path(args.eval_database) if args.eval_database else args.text_database
    raw_db = torch.load(eval_db_path, map_location="cpu")
    db_embs_raw  = raw_db["embeddings"]
    db_labels_raw = raw_db["labels"]
    db_ids_raw    = raw_db["ids"]
    db_classes    = raw_db.get("classes", [])

    if isinstance(db_embs_raw, dict):
        available_layers = sorted(int(k) for k in db_embs_raw)
        eval_layer = args.target_layer if args.target_layer in db_embs_raw else available_layers[-1]
        db_embs = db_embs_raw[eval_layer].float().numpy().astype(np.float32)
    else:
        eval_layer = args.target_layer
        db_embs = db_embs_raw.float().numpy().astype(np.float32)

    if isinstance(db_labels_raw, dict):
        db_labels_t = db_labels_raw.get(eval_layer, db_labels_raw[min(int(k) for k in db_labels_raw)])
    else:
        db_labels_t = db_labels_raw

    if isinstance(db_ids_raw, dict):
        db_ids_t = db_ids_raw.get(eval_layer, db_ids_raw[min(int(k) for k in db_ids_raw)])
    else:
        db_ids_t = db_ids_raw

    if "human" in db_classes:
        human_idx = db_classes.index("human")
        db_labels = (db_labels_t == human_idx).long().numpy().astype(np.int64)
    else:
        db_labels = db_labels_t.long().numpy().astype(np.int64)

    db_ids = db_ids_t.long().numpy().astype(np.int64)
    print(f"  kNN layer={eval_layer}  n={len(db_ids):,}  dim={db_embs.shape[1]}")

    # Build the shared kNN index once — the DB never changes during the sweep
    print("  Building kNN index …")
    shared_index = Indexer(db_embs.shape[1])
    shared_index.label_dict = {int(i): int(l) for i, l in zip(db_ids, db_labels)}
    shared_index.index_data(db_ids.tolist(), db_embs)
    print(f"  kNN index built (top_k={args.top_k})")

    # ── Pre-compute val file lists for ALL combos (cached) ────────────
    # val files are always identical for a given dir+seed regardless of
    # which training combo included it → no contamination.
    print("\nPre-computing file splits for all AI dirs …")
    # Compute once per individual AI dir
    per_dir_val: Dict[str, List[Tuple[str, int]]] = {}
    for ai_dir in ai_dirs:
        _, val_pairs = collect_files(
            [ai_dir], real_dirs,
            seed=args.seed,
            val_split=args.val_split,
            raw_subdir=args.raw_subdir,
        )
        per_dir_val[ai_dir.name] = val_pairs

    # ── Main sweep ───────────────────────────────────────────────────
    results_matrix: Dict[str, Dict[str, Dict]] = {}
    combo_names = [_combo_name(c) for c in all_combos]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for train_combo in all_combos:
        train_name = _combo_name(train_combo)
        print(f"\n{'='*70}")
        print(f"TRAINING COMBO: {train_name}")
        print(f"{'='*70}")

        # ─ collect training data ──────────────────────────────────────
        train_pairs, val_pairs = collect_files(
            list(train_combo), real_dirs,
            seed=args.seed,
            val_split=args.val_split,
            raw_subdir=args.raw_subdir,
        )
        n_ai_tr  = sum(1 for _, l in train_pairs if l == 0)
        n_real_tr = sum(1 for _, l in train_pairs if l == 1)
        n_ai_v   = sum(1 for _, l in val_pairs if l == 0)
        n_real_v  = sum(1 for _, l in val_pairs if l == 1)
        print(f"  Train: {len(train_pairs):,} ({n_ai_tr:,} AI, {n_real_tr:,} Real)  "
              f"Val: {len(val_pairs):,} ({n_ai_v:,} AI, {n_real_v:,} Real)")

        if len(train_pairs) < args.batch_size:
            print(f"  [skip] too few training samples ({len(train_pairs)} < batch_size {args.batch_size})")
            continue

        # ─ train ─────────────────────────────────────────────────────
        combo_output_dir = args.output_dir / "models" / train_name
        best_ckpt = _train_projector(
            train_pairs, val_pairs,
            centroids, centroid_labels,
            combo_output_dir,
            clip_dim=args.clip_dim,
            target_dim=target_dim,
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            epochs=args.epochs,
            lr=args.lr,
            min_lr=args.min_lr,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            temperature=args.temperature,
            num_centroids=args.num_centroids,
            seed=args.seed,
            normalize_input=args.normalize_input,
            device=device,
        )

        # ─ load trained projector ─────────────────────────────────────
        projector = CLIPProjector.from_pretrained(
            str(best_ckpt), device=str(device)
        ).to(device)

        # ─ evaluate on every AI combo ─────────────────────────────────
        results_matrix[train_name] = {}

        for eval_combo in all_combos:
            eval_name = _combo_name(eval_combo)

            # Merge val files from each AI dir in this eval combo
            # (same seed → never overlaps with training data)
            eval_samples: List[Tuple[str, int]] = []
            for ai_dir in eval_combo:
                eval_samples.extend(per_dir_val[ai_dir.name])

            if not eval_samples:
                print(f"  [eval skip] {eval_name}: no val samples")
                continue

            metrics = _evaluate_on_samples(
                eval_samples, projector, shared_index,
                top_k=args.top_k,
                threshold=args.threshold,
                batch_size=args.batch_size,
                device=device,
            )
            results_matrix[train_name][eval_name] = metrics

            marker = "  ←train" if eval_name == train_name else ""
            print(
                f"  eval [{eval_name:40s}]  "
                f"acc={metrics['accuracy']:.3f}  "
                f"f1={metrics['f1']:.3f}  "
                f"auc={metrics['roc_auc']:.3f}{marker}"
            )

        # ─ save per-training-combo results ────────────────────────────
        results_path = combo_output_dir / "eval_results.json"
        results_path.write_text(
            json.dumps(results_matrix[train_name], indent=2)
        )

    # ── Summary matrix ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  GENERALIZATION MATRIX  (rows=trained-on, cols=evaluated-on)")
    print(f"  metric: {args.report_metric}")
    print(f"{'='*70}")

    # Header row
    col_w = 8
    row_label_w = max(len(n) for n in combo_names) + 2
    header = f"  {'Train \\ Eval':<{row_label_w}}" + "".join(
        f"  {n[:col_w]:>{col_w}}" for n in combo_names
    )
    print(header)
    print("  " + "─" * (len(header) - 2))

    for tr_name in combo_names:
        if tr_name not in results_matrix:
            continue
        row = f"  {tr_name:<{row_label_w}}"
        for ev_name in combo_names:
            if ev_name in results_matrix[tr_name]:
                val = results_matrix[tr_name][ev_name].get(args.report_metric, float("nan"))
                cell = f"{val:.3f}"
            else:
                cell = "  n/a "
            row += f"  {cell:>{col_w}}"
        print(row)

    # ── Save full matrix ──────────────────────────────────────────────
    matrix_path = args.output_dir / "generalization_matrix.json"
    matrix_path.write_text(json.dumps(results_matrix, indent=2))
    print(f"\nFull results saved to {matrix_path}")
    print(f"Total time: {(time.time() - t0) / 60:.1f} min")


# ======================================================================
# Argument parser
# ======================================================================

def build_argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Train CLIPProjector on every combination of AI datasets and "
            "evaluate each model on every combination.  Produces a "
            "generalization matrix."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data ----------------------------------------------------------------
    p.add_argument("--embeddings-root", type=Path, required=True,
                   help="Root directory that contains the AI_* and Real_* embedding folders.")
    p.add_argument("--text-database", type=Path, required=True,
                   help="Frozen DeTree text database (.pt) used as training centroids.")
    p.add_argument("--target-layer", type=int, required=True,
                   help="Layer key in the text database to align image embeddings to.")
    p.add_argument("--eval-database", type=str, default=None,
                   help=(
                       "Database (.pt) used for kNN evaluation.  "
                       "Defaults to --text-database if omitted."
                   ))
    p.add_argument("--ai-datasets", nargs="+", default=None,
                   help="Override auto-discovery: explicit AI dataset folder names.")
    p.add_argument("--real-datasets", nargs="+", default=None,
                   help="Override auto-discovery: explicit Real dataset folder names.")
    p.add_argument("--raw-subdir", type=str, default="Raw",
                   help="Subdirectory under <dataset>/AI/ and <dataset>/Real/ that holds embeddings.")

    # Output --------------------------------------------------------------
    p.add_argument("--output-dir", type=Path, default=Path("runs/generalization_sweep"),
                   help="Root directory for model checkpoints and result files.")
    p.add_argument("--report-metric", type=str, default="roc_auc",
                   choices=["accuracy", "f1", "roc_auc", "recall_AI", "recall_Real"],
                   help="Metric shown in the printed summary matrix.")

    # Architecture --------------------------------------------------------
    p.add_argument("--clip-dim", type=int, default=512,
                   help="Dimensionality of input CLIP embeddings (auto-detected if wrong).")
    p.add_argument("--hidden-dim", type=int, default=None,
                   help="Hidden layer width for the projector MLP (default: target_dim).")

    # Training hyper-parameters -------------------------------------------
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--num-centroids", type=int, default=1024)
    p.add_argument("--val-split", type=float, default=0.2,
                   help="Fraction held out for validation (and for eval). Default matches collect_files.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed used for all splits. Fixed seed guarantees no train/eval overlap.")

    # Evaluation ----------------------------------------------------------
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--threshold", type=float, default=0.5,
                   help="P(Real) threshold for binary prediction.")

    # Input normalisation -------------------------------------------------
    p.add_argument("--normalize-input", dest="normalize_input", action="store_true")
    p.add_argument("--no-normalize-input", dest="normalize_input", action="store_false")
    p.set_defaults(normalize_input=True)

    return p


# ======================================================================
# Entry-point
# ======================================================================

def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    run_sweep(args)


if __name__ == "__main__":
    main()

__all__ = ["build_argument_parser", "run_sweep", "main", "collect_files"]
