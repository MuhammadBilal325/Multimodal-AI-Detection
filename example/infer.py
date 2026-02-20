"""Utility script to run DETree kNN inference against a saved database.

Supports both text and image inputs.  For images, provide a ``--projector-path``
pointing to a trained CLIPProjector checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional

# Ensure the ``detree`` package is importable when running from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from detree.inference import Detector  # noqa: E402


def _load_text_inputs(args: argparse.Namespace) -> List[str]:
    texts: List[str] = []
    if args.text:
        texts.extend(args.text)
    if args.input_file:
        input_path = Path(args.input_file)
        if input_path.suffix == ".jsonl":
            with input_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if "text" not in record:
                        raise ValueError("JSONL entries must contain a 'text' field")
                    texts.append(record["text"])
        else:
            with input_path.open("r", encoding="utf-8") as handle:
                texts.extend([line.strip() for line in handle if line.strip()])
    return texts


def _load_image_inputs(args: argparse.Namespace) -> List[str]:
    """Collect image embedding paths from --image arguments or --image-dir."""
    paths: List[str] = []
    if args.image:
        paths.extend(args.image)
    if args.image_dir:
        image_dir = Path(args.image_dir)
        for ext in ("*.npy", "*.npz"):
            paths.extend(str(p) for p in image_dir.rglob(ext))
    return sorted(paths)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DETree kNN inference.")
    parser.add_argument("--database-path", type=Path, required=True)
    parser.add_argument("--model-name-or-path", type=str, required=True)

    # Text inputs
    parser.add_argument("--text", action="append", help="Direct text input (repeatable).")
    parser.add_argument("--input-file", type=str, help="File with one example per line or JSONL with a 'text' field.")

    # Image inputs
    parser.add_argument("--image", action="append", help="Path to a .npy/.npz CLIP embedding file (repeatable).")
    parser.add_argument("--image-dir", type=str, help="Directory to scan recursively for .npy/.npz embedding files.")
    parser.add_argument("--projector-path", type=Path, help="Path to a trained CLIPProjector checkpoint (required for images).")

    # Output
    parser.add_argument("--output", type=Path, help="Optional JSON file to store predictions.")

    # Model settings
    parser.add_argument("--pooling", type=str, default="max")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.97)
    parser.add_argument("--layer", type=int, help="Layer index to use from the database.")
    parser.add_argument("--device", type=str, help="Override torch device (e.g. 'cpu').")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    texts = _load_text_inputs(args)
    images = _load_image_inputs(args)

    if not texts and not images:
        raise ValueError("No input provided.  Use --text, --input-file, --image, or --image-dir.")

    if images and not args.projector_path:
        raise ValueError("Image inference requires --projector-path to be set.")

    detector = Detector(
        database_path=args.database_path,
        model_name_or_path=args.model_name_or_path,
        pooling=args.pooling,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        top_k=args.top_k,
        threshold=args.threshold,
        layer=args.layer,
        device=args.device,
        projector_path=args.projector_path,
    )

    all_results = []

    # --- Text predictions ------------------------------------------------
    if texts:
        print(f"\n{'='*60}")
        print(f"TEXT PREDICTIONS ({len(texts)} items)")
        print(f"{'='*60}")
        predictions = detector.predict(texts)
        for prediction in predictions:
            snippet = prediction.text[:80] + "..." if len(prediction.text) > 80 else prediction.text
            print(f"\nText: {snippet}")
            print(
                f"  -> {prediction.label}  (Human: {prediction.probability_human:.4f}, AI: {prediction.probability_ai:.4f})"
            )
            all_results.append({
                "type": "text",
                "text": prediction.text,
                "label": prediction.label,
                "probability_human": prediction.probability_human,
                "probability_ai": prediction.probability_ai,
            })

    # --- Image predictions -----------------------------------------------
    if images:
        print(f"\n{'='*60}")
        print(f"IMAGE PREDICTIONS ({len(images)} items)")
        print(f"{'='*60}")
        img_predictions = detector.predict_images(images)
        for prediction in img_predictions:
            basename = os.path.basename(prediction.image_path)
            print(f"\nImage: {basename}")
            print(
                f"  -> {prediction.label}  (Real: {prediction.probability_human:.4f}, AI: {prediction.probability_ai:.4f})"
            )
            all_results.append({
                "type": "image",
                "image_path": prediction.image_path,
                "label": prediction.label,
                "probability_human": prediction.probability_human,
                "probability_ai": prediction.probability_ai,
            })

    # --- Save output -----------------------------------------------------
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(all_results, handle, ensure_ascii=False, indent=2)
        print(f"\nSaved {len(all_results)} predictions to {args.output}")


if __name__ == "__main__":
    main()
