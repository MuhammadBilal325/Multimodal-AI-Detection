"""Training CLI for the CLIP → DeTree alignment projector.

This script trains a small MLP (``CLIPProjector``) so that pre-computed CLIP
image embeddings are mapped into the **same** vector space as the frozen DeTree
text embedding database.  After training, projected image embeddings can be
merged with the text database for unified kNN-based AI-vs-Real detection.

Training objective
------------------
*Proxy Supervised Contrastive Loss* – for every projected image embedding the
loss pulls it toward same-class text centroids (proxies) sampled from the
frozen database and pushes it away from opposite-class centroids.

Typical usage::

    python -m detree.cli.train_clip_projector \\
        --embeddings-dir  Embeddings/Embeddings/Embeddings \\
        --text-database   databases/text_compressed.pt \\
        --target-layer    23 \\
        --clip-dim        768 \\
        --epochs          50 \\
        --output-dir      runs/clip_projector
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from detree.model.clip_projector import CLIPProjector


# ======================================================================
# Dataset
# ======================================================================

class CLIPEmbeddingDataset(Dataset):
    """Dataset of pre-computed CLIP embeddings stored as individual ``.npy``
    files.

    Labels are inferred from the directory hierarchy:

    * A path component named ``AI``   → label **0** (AI-generated)
    * A path component named ``Real`` → label **1** (Real / Human)

    This matches the convention used by the ``Embeddings/`` tree shipped
    with this project (e.g.
    ``Embeddings/AI_Diffusion/AI/Raw/Flash_SD/001.npy``).

    Args:
        root_dirs:       One or more root directories to scan recursively.
        normalize_input: L2-normalise each loaded embedding before returning.
    """

    def __init__(
        self,
        root_dirs: Sequence[Path],
        normalize_input: bool = True,
    ) -> None:
        self.normalize_input = normalize_input
        self.samples: List[Tuple[str, int]] = []
        for root in root_dirs:
            self._scan(root)
        if not self.samples:
            raise ValueError(f"No .npy files found under {root_dirs}")
        labels = [s[1] for s in self.samples]
        print(
            f"CLIPEmbeddingDataset: {len(self.samples)} samples "
            f"({labels.count(0)} AI, {labels.count(1)} Real)"
        )

    # ------------------------------------------------------------------
    def _scan(self, root_dir: Path) -> None:
        root_str = str(root_dir)
        for dirpath, _, filenames in os.walk(root_str):
            for fname in filenames:
                if not fname.endswith(".npy"):
                    continue
                full_path = os.path.join(dirpath, fname)
                rel_parts = (
                    os.path.relpath(full_path, root_str)
                    .replace("\\", "/")
                    .split("/")
                )
                label: Optional[int] = None
                for part in rel_parts:
                    if part == "AI":
                        label = 0
                        break
                    elif part == "Real":
                        label = 1
                        break
                if label is not None:
                    self.samples.append((full_path, label))

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        emb = np.load(path).astype(np.float32).flatten()
        emb = torch.from_numpy(emb)
        if self.normalize_input:
            emb = F.normalize(emb, dim=0)
        return emb, label


# ======================================================================
# Loss
# ======================================================================

def proxy_contrastive_loss(
    projected: torch.Tensor,
    labels: torch.Tensor,
    centroids: torch.Tensor,
    centroid_labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Supervised contrastive loss against database centroids (proxies).

    For each projected image embedding the loss maximises cosine similarity
    with same-class text centroids and minimises it with opposite-class
    centroids.  This is the standard SupCon formulation applied to *proxy*
    anchors rather than in-batch anchors.

    Args:
        projected:       ``(B, D)`` L2-normalised projected image embeddings.
        labels:          ``(B,)`` binary labels (0 = AI, 1 = Real).
        centroids:       ``(M, D)`` L2-normalised text centroids.
        centroid_labels: ``(M,)`` binary labels for the centroids.
        temperature:     Softmax temperature.

    Returns:
        Scalar loss averaged over the batch.
    """
    # Cosine similarity (both sides are L2-normalised)
    sim = projected @ centroids.T / temperature  # (B, M)

    # Positive mask: True where the centroid shares the image's label
    pos_mask = (
        labels.unsqueeze(1) == centroid_labels.unsqueeze(0)
    ).float()  # (B, M)

    # Numerical stability
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim_stable = sim - sim_max.detach()

    log_sum_exp = torch.log(
        torch.exp(sim_stable).sum(dim=1, keepdim=True) + 1e-8
    )  # (B, 1)
    log_prob = sim_stable - log_sum_exp  # (B, M)

    # Average log-probability over positive centroids
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (
        pos_mask.sum(dim=1) + 1e-8
    )  # (B,)

    return -mean_log_prob_pos.mean()


# ======================================================================
# Helpers
# ======================================================================

def _load_text_centroids(
    database_path: Path,
    target_layer: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load centroids and binary labels from a frozen DeTree text database.

    Returns:
        centroids:       ``(N, D)`` float32 tensor.
        binary_labels:   ``(N,)`` long tensor with 0 = AI, 1 = Real/Human.
    """
    data = torch.load(database_path, map_location="cpu")
    embeddings = data["embeddings"]
    labels = data["labels"]
    classes = data["classes"]

    # --- resolve layer ---------------------------------------------------
    if isinstance(embeddings, dict):
        if target_layer not in embeddings:
            available = sorted(embeddings.keys())
            raise ValueError(
                f"Layer {target_layer} not found in database. "
                f"Available layers: {available}"
            )
        centroids = embeddings[target_layer].float()
    else:
        centroids = embeddings.float()

    # --- binary label mapping --------------------------------------------
    if "human" in classes:
        human_idx = classes.index("human")
    else:
        human_idx = None

    if human_idx is not None:
        binary_labels = (labels == human_idx).long()
    else:
        binary_labels = labels.long()

    return centroids, binary_labels


# ======================================================================
# Argument parser
# ======================================================================

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the CLIP → DeTree alignment projector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data ----------------------------------------------------------------
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        nargs="+",
        required=True,
        help="Root director(ies) containing pre-computed CLIP .npy files.",
    )
    parser.add_argument(
        "--text-database",
        type=Path,
        required=True,
        help="Path to the frozen DeTree text embedding database (.pt).",
    )
    parser.add_argument(
        "--target-layer",
        type=int,
        required=True,
        help="Layer key in the text database to align image embeddings to.",
    )

    # Architecture --------------------------------------------------------
    parser.add_argument(
        "--clip-dim",
        type=int,
        default=768,
        help="Dimensionality of the input CLIP embeddings.",
    )
    parser.add_argument(
        "--target-dim",
        type=int,
        default=1024,
        help=(
            "Dimensionality of the DeTree text embedding space.  "
            "Auto-detected from the database if the provided value differs."
        ),
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Hidden layer width for the projector MLP (default: target_dim).",
    )

    # Output --------------------------------------------------------------
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/clip_projector"),
        help="Root directory for experiment outputs.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="clip_align",
        help="Sub-folder name inside --output-dir.",
    )

    # Training hyper-parameters -------------------------------------------
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument(
        "--num-centroids",
        type=int,
        default=1024,
        help=(
            "Total number of text centroids to sample per batch for the "
            "contrastive loss (balanced across classes)."
        ),
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of data to hold out for validation.",
    )
    parser.add_argument("--seed", type=int, default=42)

    # Input normalisation -------------------------------------------------
    parser.add_argument(
        "--normalize-input",
        dest="normalize_input",
        action="store_true",
        help="L2-normalise each loaded CLIP embedding.",
    )
    parser.add_argument(
        "--no-normalize-input",
        dest="normalize_input",
        action="store_false",
    )
    parser.set_defaults(normalize_input=True)

    return parser


# ======================================================================
# Training loop
# ======================================================================

def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # 1. Load frozen text database centroids
    # ------------------------------------------------------------------
    print(f"Loading text database from {args.text_database} …")
    centroids, centroid_labels = _load_text_centroids(
        args.text_database, args.target_layer,
    )

    # Auto-detect target dimension from the database
    target_dim = centroids.shape[1]
    if args.target_dim != target_dim:
        print(
            f"  Auto-detected target_dim={target_dim} from database "
            f"(overriding --target-dim={args.target_dim})"
        )
        args.target_dim = target_dim

    # L2-normalise centroids for loss computation
    centroids = F.normalize(centroids, dim=-1).to(device)
    centroid_labels = centroid_labels.to(device)

    n_ai = int((centroid_labels == 0).sum().item())
    n_real = int((centroid_labels == 1).sum().item())
    print(
        f"  Database: {centroids.shape[0]} centroids "
        f"({n_ai} AI, {n_real} Real), dim={target_dim}"
    )

    # Pre-compute index tensors for balanced sampling
    ai_indices = (centroid_labels == 0).nonzero(as_tuple=True)[0]
    real_indices = (centroid_labels == 1).nonzero(as_tuple=True)[0]

    # ------------------------------------------------------------------
    # 2. Load CLIP embedding dataset
    # ------------------------------------------------------------------
    print(f"Scanning embedding directories: {args.embeddings_dir}")
    full_dataset = CLIPEmbeddingDataset(
        args.embeddings_dir,
        normalize_input=args.normalize_input,
    )

    # Auto-detect CLIP dimension from the first sample
    sample_emb, _ = full_dataset[0]
    detected_clip_dim = sample_emb.shape[0]
    if args.clip_dim != detected_clip_dim:
        print(
            f"  Auto-detected clip_dim={detected_clip_dim} "
            f"(overriding --clip-dim={args.clip_dim})"
        )
        args.clip_dim = detected_clip_dim

    # Train / validation split
    n_val = int(len(full_dataset) * args.val_split)
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"  Split: {n_train} train, {n_val} val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # 3. Create model, optimiser, scheduler
    # ------------------------------------------------------------------
    model = CLIPProjector(
        clip_dim=args.clip_dim,
        target_dim=args.target_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Projector: {n_params:,} trainable parameters")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = max(args.epochs * len(train_loader) - args.warmup_steps, 1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, total_steps, eta_min=args.min_lr,
    )

    # ------------------------------------------------------------------
    # 4. Output directory & logging
    # ------------------------------------------------------------------
    experiment_dir = args.output_dir / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(experiment_dir / "runs"))

    # Persist config
    config_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    with open(experiment_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, sort_keys=False, default_flow_style=False)

    # ------------------------------------------------------------------
    # 5. Training
    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(args.epochs):
        # --- train -------------------------------------------------------
        model.train()
        epoch_loss = 0.0
        iterator = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            leave=True,
        )

        for i, (clip_embs, labels) in enumerate(iterator):
            global_step += 1

            # Warmup
            if global_step <= args.warmup_steps:
                warmup_lr = args.lr * global_step / max(args.warmup_steps, 1)
                for pg in optimizer.param_groups:
                    pg["lr"] = warmup_lr

            clip_embs = clip_embs.to(device)
            labels = labels.to(device)

            # Forward
            projected = model(clip_embs, normalize=True)

            # Sample a balanced subset of centroids for this batch
            n_per_class = min(
                args.num_centroids // 2,
                len(ai_indices),
                len(real_indices),
            )
            sampled_ai = ai_indices[
                torch.randperm(len(ai_indices), device=device)[:n_per_class]
            ]
            sampled_real = real_indices[
                torch.randperm(len(real_indices), device=device)[:n_per_class]
            ]
            sampled_idx = torch.cat([sampled_ai, sampled_real])
            sampled_centroids = centroids[sampled_idx]
            sampled_labels = centroid_labels[sampled_idx]

            loss = proxy_contrastive_loss(
                projected, labels, sampled_centroids, sampled_labels,
                temperature=args.temperature,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if global_step > args.warmup_steps:
                scheduler.step()

            epoch_loss = (epoch_loss * i + loss.item()) / (i + 1)
            current_lr = optimizer.param_groups[0]["lr"]
            iterator.set_postfix(
                loss=f"{loss.item():.4f}",
                avg=f"{epoch_loss:.4f}",
                lr=f"{current_lr:.2e}",
            )

            if global_step % 10 == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr", current_lr, global_step)

        # --- validation --------------------------------------------------
        model.eval()
        val_loss_sum = 0.0
        correct = 0
        total = 0

        # Class prototypes for a quick accuracy check
        ai_proto = F.normalize(
            centroids[ai_indices].mean(dim=0, keepdim=True), dim=-1,
        )
        real_proto = F.normalize(
            centroids[real_indices].mean(dim=0, keepdim=True), dim=-1,
        )
        prototypes = torch.cat([ai_proto, real_proto], dim=0)  # (2, D)

        with torch.no_grad():
            for clip_embs, labels in val_loader:
                clip_embs = clip_embs.to(device)
                labels = labels.to(device)

                projected = model(clip_embs, normalize=True)

                # Loss against ALL centroids (no sampling for stable metric)
                loss = proxy_contrastive_loss(
                    projected, labels, centroids, centroid_labels,
                    temperature=args.temperature,
                )
                val_loss_sum += loss.item() * clip_embs.size(0)

                # Prototype-nearest accuracy
                sim = projected @ prototypes.T  # (B, 2)
                preds = sim.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss = val_loss_sum / max(total, 1)
        val_acc = correct / max(total, 1) * 100.0

        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/accuracy", val_acc, epoch)
        print(f"  Val loss: {val_loss:.4f}  |  Val acc: {val_acc:.1f}%")

        # --- checkpointing -----------------------------------------------
        ckpt_dir = experiment_dir / f"epoch_{epoch:02d}"
        model.save_pretrained(str(ckpt_dir))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(str(experiment_dir / "best"))
            print(f"  ★ New best model (val_loss={val_loss:.4f})")

        model.save_pretrained(str(experiment_dir / "last"))
        writer.flush()

    writer.close()
    print(f"\nTraining complete.  Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {experiment_dir}")


# ======================================================================
# Entry-point
# ======================================================================

def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()

__all__ = ["build_argument_parser", "train", "main", "CLIPEmbeddingDataset"]
