"""Merge multiple embedding databases into a single unified database.

Combines text and image (or any number of) ``.pt`` embedding databases into
one file that can be loaded by the DeTree kNN evaluator or ``Detector``.
All databases are aligned to a single *target layer* and labels are remapped
to a binary scheme: **0 = AI / LLM**, **1 = Real / Human**.

Typical usage::

    python -m detree.cli.merge_databases \\
        --databases  databases/text_compressed.pt  databases/image_embeddings.pt \\
        --target-layer 23 \\
        --output databases/merged.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import torch


# ======================================================================
# Argument parser
# ======================================================================

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge two or more .pt embedding databases (e.g. text + image) "
            "into a single unified database for kNN evaluation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--databases",
        type=Path,
        nargs="+",
        required=True,
        help="Paths to the .pt databases to merge (order does not matter).",
    )
    parser.add_argument(
        "--target-layer",
        type=int,
        required=True,
        help=(
            "The layer key to extract from each database.  All databases "
            "must contain embeddings at this layer."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for the merged database (.pt).",
    )
    return parser


# ======================================================================
# Merge logic
# ======================================================================

def merge_databases(args: argparse.Namespace) -> None:
    all_embeddings: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    total_count = 0

    print(f"Merging {len(args.databases)} database(s) at layer {args.target_layer}\n")

    for db_path in args.databases:
        data = torch.load(db_path, map_location="cpu")
        embeddings = data["embeddings"]
        labels = data["labels"]
        classes = data["classes"]

        # --- resolve target layer ----------------------------------------
        if isinstance(embeddings, dict):
            if args.target_layer not in embeddings:
                available = sorted(embeddings.keys())
                raise ValueError(
                    f"Layer {args.target_layer} not found in {db_path}. "
                    f"Available layers: {available}"
                )
            layer_emb = embeddings[args.target_layer]
        else:
            layer_emb = embeddings

        # --- remap labels to binary 0=AI, 1=Human -----------------------
        if "human" in classes:
            human_idx = classes.index("human")
            binary_labels = (labels == human_idx).long()
        else:
            # Assume labels are already binary (0=AI, 1=Real)
            binary_labels = labels.long()

        n_entries = binary_labels.shape[0]
        n_ai = int((binary_labels == 0).sum().item())
        n_real = int((binary_labels == 1).sum().item())
        dim = layer_emb.shape[1]
        print(
            f"  {db_path.name}: {n_entries} entries "
            f"({n_ai} AI, {n_real} Real), dim={dim}"
        )

        all_embeddings.append(layer_emb)
        all_labels.append(binary_labels)
        total_count += n_entries

    # --- concatenate -----------------------------------------------------
    merged_embeddings = torch.cat(all_embeddings, dim=0)
    merged_labels = torch.cat(all_labels, dim=0)

    # Sequential IDs for the merged database (avoids collisions)
    merged_ids = torch.arange(total_count, dtype=torch.long)

    emb_dict = {
        "embeddings": {args.target_layer: merged_embeddings},
        "labels": merged_labels,
        "ids": merged_ids,
        "classes": ["llm", "human"],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(emb_dict, args.output)

    n_ai_total = int((merged_labels == 0).sum().item())
    n_real_total = int((merged_labels == 1).sum().item())
    print(
        f"\nMerged database saved to {args.output}\n"
        f"  {total_count} total entries ({n_ai_total} AI, {n_real_total} Real)\n"
        f"  dim={merged_embeddings.shape[1]}, layer={args.target_layer}"
    )


# ======================================================================
# Entry-point
# ======================================================================

def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    merge_databases(args)


if __name__ == "__main__":
    main()

__all__ = ["build_argument_parser", "merge_databases", "main"]
