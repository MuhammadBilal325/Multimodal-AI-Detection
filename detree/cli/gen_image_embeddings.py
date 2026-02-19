"""Generate an image embedding database using the trained CLIP projector.

This script loads pre-computed CLIP embeddings (``.npy`` files), projects them
through a trained ``CLIPProjector``, L2-normalises the results, and saves
everything in the same ``.pt`` database format that DeTree uses for text
embeddings.  The resulting database can then be merged with a text database
via ``merge_databases.py`` for unified kNN detection.

Typical usage::

    python -m detree.cli.gen_image_embeddings \\
        --embeddings-dir  Embeddings/Embeddings/Embeddings \\
        --projector-path  runs/clip_projector/clip_align/best \\
        --target-layer    23 \\
        --output          databases/image_embeddings.pt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from detree.model.clip_projector import CLIPProjector


# ======================================================================
# Dataset
# ======================================================================

class CLIPEmbeddingDataset(Dataset):
    """Loads pre-computed CLIP embeddings from ``.npy`` files.

    Labels are inferred from directory names:
        * ``AI``   in path → label **0** (AI-generated)
        * ``Real`` in path → label **1** (Real / Human)

    Each sample returns ``(embedding, label, id)`` where *id* is the
    sequential index (used as a unique identifier in the database).
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

    def _scan(self, root_dir: Path) -> None:
        root_str = str(root_dir)
        for dirpath, _, filenames in os.walk(root_str):
            for fname in sorted(filenames):
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

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        path, label = self.samples[idx]
        emb = np.load(path).astype(np.float32).flatten()
        emb = torch.from_numpy(emb)
        if self.normalize_input:
            emb = F.normalize(emb, dim=0)
        return emb, label, idx


# ======================================================================
# Argument parser
# ======================================================================

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Project pre-computed CLIP embeddings through a trained "
            "CLIPProjector and save as a DeTree-compatible .pt database."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        nargs="+",
        required=True,
        help="Root director(ies) containing pre-computed CLIP .npy files.",
    )
    parser.add_argument(
        "--projector-path",
        type=Path,
        required=True,
        help=(
            "Path to the directory containing a trained clip_projector.pt "
            "checkpoint (e.g. runs/clip_projector/clip_align/best)."
        ),
    )
    parser.add_argument(
        "--target-layer",
        type=int,
        required=True,
        help=(
            "Layer key under which to store the projected embeddings.  "
            "Must match the layer used in the text database for merging."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for the image embedding database (.pt).",
    )

    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)

    parser.add_argument(
        "--normalize-input",
        dest="normalize_input",
        action="store_true",
        help="L2-normalise each loaded CLIP embedding before projection.",
    )
    parser.add_argument(
        "--no-normalize-input",
        dest="normalize_input",
        action="store_false",
    )
    parser.set_defaults(normalize_input=True)

    return parser


# ======================================================================
# Main logic
# ======================================================================

def generate_image_embeddings(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained projector
    print(f"Loading projector from {args.projector_path} …")
    projector = CLIPProjector.from_pretrained(
        str(args.projector_path), device=str(device),
    )
    projector = projector.to(device)
    projector.eval()
    print(
        f"  clip_dim={projector.clip_dim}  target_dim={projector.target_dim}  "
        f"hidden_dim={projector.hidden_dim}"
    )

    # Load dataset
    dataset = CLIPEmbeddingDataset(
        args.embeddings_dir, normalize_input=args.normalize_input,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    all_embeddings: List[torch.Tensor] = []
    all_labels: List[int] = []
    all_ids: List[int] = []

    with torch.no_grad():
        for clip_embs, labels, ids in tqdm(dataloader, desc="Projecting"):
            clip_embs = clip_embs.to(device)
            projected = projector(clip_embs, normalize=True)

            all_embeddings.append(projected.cpu().to(torch.bfloat16))
            all_labels.extend(labels.tolist())
            all_ids.extend(ids.tolist())

    all_embeddings_t = torch.cat(all_embeddings, dim=0)
    all_labels_t = torch.tensor(all_labels, dtype=torch.long)
    all_ids_t = torch.tensor(all_ids, dtype=torch.long)

    # Labels: 0 = AI → "llm", 1 = Real → "human"  (DeTree convention)
    emb_dict = {
        "embeddings": {args.target_layer: all_embeddings_t},
        "labels": all_labels_t,
        "ids": all_ids_t,
        "classes": ["llm", "human"],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(emb_dict, args.output)

    n_ai = int((all_labels_t == 0).sum().item())
    n_real = int((all_labels_t == 1).sum().item())
    print(
        f"Saved image embedding database to {args.output}\n"
        f"  {len(all_ids_t)} entries ({n_ai} AI, {n_real} Real), "
        f"dim={all_embeddings_t.shape[1]}, layer={args.target_layer}"
    )


# ======================================================================
# Entry-point
# ======================================================================

def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    generate_image_embeddings(args)


if __name__ == "__main__":
    main()

__all__ = ["build_argument_parser", "generate_image_embeddings", "main"]
