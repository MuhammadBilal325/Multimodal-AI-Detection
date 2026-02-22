"""Merge multiple embedding databases into a single unified database.

Combines text and image (or any number of) ``.pt`` embedding databases into
one file that can be loaded by the DeTree kNN evaluator or ``Detector``.
Labels are remapped to a binary scheme: **0 = AI / LLM**, **1 = Real / Human**.

All layers present across every database are preserved in the output.  When
two databases both contain the same layer (e.g. the layer the image projector
was trained on), their embeddings are concatenated at that layer.  Layers
that exist in only one database are passed through unchanged.

Because different layers can contain different numbers of entries, the output
stores ``labels`` and ``ids`` as dicts keyed by layer index rather than as
flat tensors.  The ``Detector`` already supports this format.

Typical usage::

    python -m detree.cli.merge_databases \\
        --databases  databases/text_compressed.pt  databases/image_embeddings.pt \\
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
            "into a single unified database for kNN evaluation.  "
            "All layers present in any input database are preserved."
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
    # Accumulate per-layer embeddings and labels across all input databases.
    # Structure: {layer_int: {"embeddings": [...], "labels": [...]}}
    layer_data: dict = {}

    print(f"Merging {len(args.databases)} database(s) — keeping all layers\n")

    for db_path in args.databases:
        data = torch.load(db_path, map_location="cpu")
        embeddings = data["embeddings"]
        labels = data["labels"]
        classes = data["classes"]

        # Normalise embeddings to always be a layer-keyed dict
        if not isinstance(embeddings, dict):
            raise ValueError(
                f"{db_path}: 'embeddings' must be a dict keyed by layer index."
            )

        # Normalise labels: may already be a per-layer dict or a flat tensor
        if isinstance(labels, dict):
            label_per_layer: dict = {int(k): v for k, v in labels.items()}
        else:
            # Flat tensor — shared across all layers in this database
            label_per_layer = {int(k): labels for k in embeddings}

        available_layers = sorted(int(k) for k in embeddings)
        print(f"  {db_path.name}: layers {available_layers}")

        for layer in available_layers:
            layer_emb = embeddings[layer]
            layer_labels_raw = label_per_layer.get(layer, label_per_layer.get(min(label_per_layer)))

            # Remap labels to binary 0=AI, 1=Human
            if "human" in classes:
                human_idx = classes.index("human")
                binary_labels = (layer_labels_raw == human_idx).long()
            else:
                binary_labels = layer_labels_raw.long()

            n_entries = binary_labels.shape[0]
            n_ai = int((binary_labels == 0).sum().item())
            n_real = int((binary_labels == 1).sum().item())
            print(
                f"    layer {layer}: {n_entries} entries "
                f"({n_ai} AI, {n_real} Real), dim={layer_emb.shape[1]}"
            )

            if layer not in layer_data:
                layer_data[layer] = {"embeddings": [], "labels": []}
            layer_data[layer]["embeddings"].append(layer_emb)
            layer_data[layer]["labels"].append(binary_labels)

    # --- build merged tensors per layer ----------------------------------
    merged_emb_dict: dict = {}
    merged_label_dict: dict = {}
    merged_id_dict: dict = {}

    print()
    total_unique = 0
    id_offset = 0
    for layer in sorted(layer_data):
        embs = torch.cat(layer_data[layer]["embeddings"], dim=0)
        labs = torch.cat(layer_data[layer]["labels"], dim=0)
        ids = torch.arange(id_offset, id_offset + embs.shape[0], dtype=torch.long)
        id_offset += embs.shape[0]
        total_unique += embs.shape[0]

        merged_emb_dict[layer] = embs
        merged_label_dict[layer] = labs
        merged_id_dict[layer] = ids

        n_ai = int((labs == 0).sum().item())
        n_real = int((labs == 1).sum().item())
        print(
            f"  Merged layer {layer}: {embs.shape[0]} entries "
            f"({n_ai} AI, {n_real} Real), dim={embs.shape[1]}"
        )

    emb_dict = {
        "embeddings": merged_emb_dict,
        "labels": merged_label_dict,
        "ids": merged_id_dict,
        "classes": ["llm", "human"],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(emb_dict, args.output)

    print(
        f"\nMerged database saved to {args.output}\n"
        f"  {len(merged_emb_dict)} layers: {sorted(merged_emb_dict)}"
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
