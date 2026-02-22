"""Evaluate image detection accuracy on pre-computed CLIP embeddings.

Projects every ``.npy``/``.npz`` file in the given embeddings directory through
the trained ``CLIPProjector``, runs kNN search against the merged multimodal
database, and reports detection accuracy.

Results are broken down by:
  * Overall binary accuracy, precision, recall, F1, ROC-AUC
  * Per top-level folder (e.g. AI_Diffusion, AI_GAN, Real_VISION …)

Typical usage::

    python -m detree.cli.test_image_score_knn \\
        --database   databases/merged_multimodal.pt \\
        --projector  runs/clip_projector/clip_align/best \\
        --embeddings-dir Embeddings/Embeddings/Embeddings \\
        --target-layer 23 \\
        --top-k 10
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from detree.model.clip_projector import CLIPProjector
from detree.utils.index import Indexer


# ======================================================================
# Helpers
# ======================================================================

def _infer_label(full_path: str, root_str: str) -> Optional[int]:
    """Return 0 (AI) or 1 (Real) by scanning path components."""
    rel = os.path.relpath(full_path, root_str).replace("\\", "/")
    for part in rel.split("/"):
        if part == "AI":
            return 0
        if part == "Real":
            return 1
    return None


def _top_folder(full_path: str, root_str: str) -> str:
    """Return the top-level folder name relative to root."""
    rel = os.path.relpath(full_path, root_str).replace("\\", "/")
    return rel.split("/")[0]


def _load_embedding(path: str) -> np.ndarray:
    if path.endswith(".npz"):
        data = np.load(path)
        for key in ("embedding", "emb", "arr_0"):
            if key in data:
                return data[key].astype(np.float32).flatten()
        return data[list(data.keys())[0]].astype(np.float32).flatten()
    return np.load(path).astype(np.float32).flatten()


def _scan_embeddings(
    root_dirs: Sequence[Path],
) -> List[Tuple[str, int, str]]:
    """Walk root_dirs and return (path, true_label, folder_name) triples."""
    samples: List[Tuple[str, int, str]] = []
    for root in root_dirs:
        root_str = str(root)
        for dirpath, _, filenames in os.walk(root_str):
            for fname in sorted(filenames):
                if not (fname.endswith(".npy") or fname.endswith(".npz")):
                    continue
                full = os.path.join(dirpath, fname)
                label = _infer_label(full, root_str)
                if label is None:
                    continue
                folder = _top_folder(full, root_str)
                samples.append((full, label, folder))
    return samples


# ======================================================================
# Database loading
# ======================================================================

def _load_database(
    db_path: Path,
    target_layer: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return (embeddings, labels, ids, resolved_layer) arrays for *target_layer*.

    If *target_layer* is None, the highest layer present in the database is used.
    """
    data = torch.load(db_path, map_location="cpu")
    embs = data["embeddings"]
    labs = data["labels"]
    ids = data["ids"]

    # Resolve layer
    if isinstance(embs, dict):
        available = sorted(int(k) for k in embs)
        if target_layer is None:
            target_layer = available[-1]
            print(f"  No --target-layer specified, using highest layer in database: {target_layer}")
        elif target_layer not in available:
            raise ValueError(
                f"Layer {target_layer} not found in database. "
                f"Available: {available}"
            )
        layer_emb = embs[target_layer].float()
    else:
        if target_layer is None:
            target_layer = 0
        layer_emb = embs.float()

    # Resolve labels (may be per-layer dict or flat tensor)
    if isinstance(labs, dict):
        layer_labs = labs.get(target_layer, labs[min(int(k) for k in labs)])
    else:
        layer_labs = labs

    # Resolve ids
    if isinstance(ids, dict):
        layer_ids = ids.get(target_layer, ids[min(int(k) for k in ids)])
    else:
        layer_ids = ids

    # Remap to binary 0=AI, 1=Real
    classes = data.get("classes", [])
    if "human" in classes:
        human_idx = classes.index("human")
        bin_labels = (layer_labs == human_idx).long()
    else:
        bin_labels = layer_labs.long()

    return (
        layer_emb.numpy().astype(np.float32),
        bin_labels.numpy().astype(np.int64),
        layer_ids.numpy().astype(np.int64),
        target_layer,
    )


# ======================================================================
# Metrics helpers
# ======================================================================

def _binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> Dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
    )

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float("nan")

    # Per-class recall (balanced accuracy)
    ai_mask = y_true == 0
    real_mask = y_true == 1
    ai_recall = float(np.mean(y_pred[ai_mask] == 0)) if ai_mask.any() else float("nan")
    real_recall = float(np.mean(y_pred[real_mask] == 1)) if real_mask.any() else float("nan")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "recall_AI": ai_recall,
        "recall_Real": real_recall,
    }


def _print_metrics(title: str, metrics: Dict[str, float], n: int) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}  (n={n:,})")
    print(f"{'─' * 60}")
    print(f"  Accuracy   : {metrics['accuracy']:.4f}")
    print(f"  Precision  : {metrics['precision']:.4f}")
    print(f"  Recall     : {metrics['recall']:.4f}")
    print(f"  F1 Score   : {metrics['f1']:.4f}")
    print(f"  ROC-AUC    : {metrics['roc_auc']:.4f}")
    print(f"  Recall(AI) : {metrics['recall_AI']:.4f}")
    print(f"  Recall(Real): {metrics['recall_Real']:.4f}")


# ======================================================================
# Main evaluation
# ======================================================================

def evaluate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # 1. Load projector
    # ------------------------------------------------------------------
    print(f"Loading projector from {args.projector} …")
    projector = CLIPProjector.from_pretrained(str(args.projector), device=str(device))
    projector = projector.to(device).eval()
    print(f"  clip_dim={projector.clip_dim}  target_dim={projector.target_dim}")

    # ------------------------------------------------------------------
    # 2. Load database and build kNN index
    # ------------------------------------------------------------------
    print(f"\nLoading database from {args.database} …")
    db_embs, db_labels, db_ids, resolved_layer = _load_database(args.database, args.target_layer)
    n_ai_db = int((db_labels == 0).sum())
    n_real_db = int((db_labels == 1).sum())
    print(
        f"  Layer {resolved_layer}: {len(db_ids):,} entries "
        f"({n_ai_db:,} AI, {n_real_db:,} Real), dim={db_embs.shape[1]}"
    )

    index = Indexer(db_embs.shape[1])
    label_dict = {int(i): int(l) for i, l in zip(db_ids, db_labels)}
    index.label_dict = label_dict
    index.index_data(db_ids.tolist(), db_embs)
    print(f"  kNN index built (k={args.top_k})")

    # ------------------------------------------------------------------
    # 3. Scan embedding files
    # ------------------------------------------------------------------
    print(f"\nScanning {args.embeddings_dir} …")
    samples = _scan_embeddings(args.embeddings_dir)
    if not samples:
        raise ValueError(f"No labelled .npy/.npz files found under {args.embeddings_dir}")
    n_ai = sum(1 for _, l, _ in samples if l == 0)
    n_real = sum(1 for _, l, _ in samples if l == 1)
    print(f"  {len(samples):,} samples ({n_ai:,} AI, {n_real:,} Real)")

    # ------------------------------------------------------------------
    # 4. Project embeddings and run kNN
    # ------------------------------------------------------------------
    all_true: List[int] = []
    all_pred: List[int] = []
    all_score: List[float] = []   # P(Real) for ROC-AUC
    folder_data: Dict[str, Dict] = defaultdict(lambda: {"true": [], "pred": [], "score": []})

    paths = [s[0] for s in samples]
    true_labels = [s[1] for s in samples]
    folders = [s[2] for s in samples]

    # Process in batches
    with torch.no_grad():
        for i in tqdm(range(0, len(paths), args.batch_size), desc="Evaluating"):
            batch_paths = paths[i : i + args.batch_size]
            batch_true = true_labels[i : i + args.batch_size]
            batch_folders = folders[i : i + args.batch_size]

            # Load + normalise CLIP embeddings
            raw = [_load_embedding(p) for p in batch_paths]
            batch_tensor = torch.from_numpy(np.stack(raw, axis=0))
            batch_tensor = F.normalize(batch_tensor, dim=-1).to(device)

            # Project
            projected = projector(batch_tensor, normalize=True)
            projected_np = projected.cpu().numpy().astype(np.float32)

            # kNN search
            results = index.search_knn(
                projected_np,
                args.top_k,
                index_batch_size=max(1, min(args.top_k, 128)),
            )

            for j, (_ids, scores, nn_labels) in enumerate(results):
                scores_t = torch.from_numpy(np.asarray(scores))
                weights = torch.softmax(scores_t, dim=0)
                label_t = torch.tensor(nn_labels, dtype=torch.float32)
                p_real = float(torch.dot(weights, label_t).clamp(0.0, 1.0))
                pred = 1 if p_real >= args.threshold else 0

                all_true.append(batch_true[j])
                all_pred.append(pred)
                all_score.append(p_real)

                f = batch_folders[j]
                folder_data[f]["true"].append(batch_true[j])
                folder_data[f]["pred"].append(pred)
                folder_data[f]["score"].append(p_real)

    # ------------------------------------------------------------------
    # 5. Compute and print metrics
    # ------------------------------------------------------------------
    y_true = np.array(all_true, dtype=np.int64)
    y_pred = np.array(all_pred, dtype=np.int64)
    y_score = np.array(all_score, dtype=np.float32)

    overall = _binary_metrics(y_true, y_pred, y_score)
    _print_metrics("OVERALL", overall, len(y_true))

    print(f"\n{'─' * 60}")
    print("  PER-FOLDER BREAKDOWN")
    print(f"{'─' * 60}")
    # Table header
    header = f"  {'Folder':<28}  {'n':>6}  {'Acc':>7}  {'F1':>7}  {'AUC':>7}  {'R(AI)':>7}  {'R(Real)':>7}"
    print(header)
    print(f"  {'─'*28}  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}")

    for folder in sorted(folder_data):
        fd = folder_data[folder]
        ft = np.array(fd["true"], dtype=np.int64)
        fp = np.array(fd["pred"], dtype=np.int64)
        fs = np.array(fd["score"], dtype=np.float32)
        m = _binary_metrics(ft, fp, fs)
        print(
            f"  {folder:<28}  {len(ft):>6,}  "
            f"{m['accuracy']:>7.4f}  {m['f1']:>7.4f}  {m['roc_auc']:>7.4f}  "
            f"{m['recall_AI']:>7.4f}  {m['recall_Real']:>7.4f}"
        )

    print(f"\n{'─' * 60}\n")

    # ------------------------------------------------------------------
    # 6. Save results (optional)
    # ------------------------------------------------------------------
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        import json

        report = {
            "overall": {k: float(v) for k, v in overall.items()},
            "per_folder": {
                folder: {
                    k: float(v)
                    for k, v in _binary_metrics(
                        np.array(fd["true"], dtype=np.int64),
                        np.array(fd["pred"], dtype=np.int64),
                        np.array(fd["score"], dtype=np.float32),
                    ).items()
                }
                for folder, fd in folder_data.items()
            },
        }
        args.output.write_text(json.dumps(report, indent=2))
        print(f"Results saved to {args.output}")


# ======================================================================
# Argument parser
# ======================================================================

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate image detection accuracy using a merged multimodal database.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--database",
        type=Path,
        required=True,
        help="Path to the merged .pt database (output of merge_databases).",
    )
    parser.add_argument(
        "--projector",
        type=Path,
        required=True,
        help="Path to the trained CLIPProjector checkpoint directory.",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        nargs="+",
        required=True,
        help="Root director(ies) containing pre-computed CLIP .npy files.",
    )
    parser.add_argument(
        "--target-layer",
        type=int,
        default=None,
        help=(
            "Layer key in the database to run kNN against. "
            "Defaults to the highest layer present in the database "
            "(same behaviour as the Detector class)."
        ),
    )
    parser.add_argument("--top-k", type=int, default=10, help="Number of nearest neighbours.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="P(Real) threshold above which a sample is classified as Real.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save a JSON results report.",
    )
    return parser


# ======================================================================
# Entry-point
# ======================================================================

def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    evaluate(args)


if __name__ == "__main__":
    main()

__all__ = ["build_argument_parser", "evaluate", "main"]
