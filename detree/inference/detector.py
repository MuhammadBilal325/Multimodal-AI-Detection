"""High-level detector interface for running DETree inference.

Supports both text and image inputs.  For images, a trained ``CLIPProjector``
maps CLIP embeddings into the DeTree embedding space before kNN lookup.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from detree.model.text_embedding import TextEmbeddingModel
from detree.model.clip_projector import CLIPProjector
from detree.utils.index import Indexer

__all__ = ["Detector", "Prediction", "ImagePrediction"]


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _load_database(path: Path):
    data = torch.load(path, map_location="cpu")
    embeddings = data["embeddings"]
    labels = data["labels"]
    ids = data["ids"]
    classes = data["classes"]
    if not isinstance(embeddings, dict):
        raise ValueError("Expected embeddings to be a dict keyed by layer index")
    return embeddings, labels, ids, classes


class TextDataset(Dataset):
    def __init__(self, texts: Sequence[str]):
        self._texts = [str(text) for text in texts]

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, idx: int):
        return self._texts[idx], idx


@dataclass
class Prediction:
    text: str
    probability_ai: float
    probability_human: float
    label: str


@dataclass
class ImagePrediction:
    """Prediction result for an image input."""
    image_path: str
    probability_ai: float
    probability_human: float
    label: str


class Detector:
    """Wraps model + database logic for kNN predictions.
    
    Supports both text and image inputs.  For images, provide a trained
    ``projector_path`` pointing to a ``CLIPProjector`` checkpoint.
    """

    def __init__(
        self,
        database_path: Path,
        model_name_or_path: str,
        *,
        pooling: str = "max",
        max_length: int = 512,
        batch_size: int = 8,
        num_workers: int = 0,
        top_k: int = 10,
        threshold: float = 0.97,
        layer: Optional[int] = None,
        device: Optional[str] = None,
        projector_path: Optional[Path] = None,
    ) -> None:
        self.database_path = database_path
        self.model_name_or_path = model_name_or_path
        self.pooling = pooling
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.top_k = top_k
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "threshold must be a probability between 0 and 1 (inclusive)."
            )
        self.threshold = threshold
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        embeddings, labels, ids, classes = _load_database(database_path)
        self.classes = list(classes)
        self.human_index = None
        if "human" in self.classes:
            self.human_index = self.classes.index("human")

        self._raw_labels = labels
        self._raw_ids = ids

        self.layer_embeddings = {
            int(layer): tensor.float() for layer, tensor in embeddings.items()
        }

        if isinstance(labels, dict):
            self.layer_labels = {int(layer): tensor for layer, tensor in labels.items()}
        else:
            self.layer_labels = None
        if isinstance(ids, dict):
            self.layer_ids = {int(layer): tensor for layer, tensor in ids.items()}
        else:
            self.layer_ids = None

        self.available_layers = sorted(self.layer_embeddings.keys())
        if not self.available_layers:
            raise ValueError("No layers found in the embedding database")
        requested_layer = layer if layer is not None else self.available_layers[-1]
        if requested_layer not in self.available_layers:
            raise ValueError(f"Requested layer {layer} not present in database")

        self.model = TextEmbeddingModel(
            model_name_or_path,
            output_hidden_states=True,
            infer=True,
            use_pooling=self.pooling,
        ).to(self.device)
        self.model.eval()
        self.tokenizer = self.model.tokenizer

        if self.human_index is None:
            raise ValueError(
                "Database must include a 'human' entry in its classes list to compute probabilities."
            )

        # --- Image projector (optional) -----------------------------------
        self.projector: Optional[CLIPProjector] = None
        self.projector_path = projector_path
        if projector_path is not None:
            self.projector = CLIPProjector.from_pretrained(
                str(projector_path), device=str(self.device),
            ).to(self.device)
            self.projector.eval()

        self._configure_layer(requested_layer)

    def _configure_layer(self, layer: int) -> None:
        if layer not in self.layer_embeddings:
            raise ValueError(f"Requested layer {layer} not present in database")

        layer_embeddings = self.layer_embeddings[layer]
        self.embedding_dim = layer_embeddings.shape[-1]

        if self.layer_labels is not None:
            layer_labels = self.layer_labels[layer]
        else:
            # Fall back to shared labels tensor when per-layer labels are unavailable.
            layer_labels = self._raw_labels

        if self.layer_ids is not None:
            layer_ids = self.layer_ids[layer]
        else:
            layer_ids = self._raw_ids

        train_embeddings = _to_numpy(layer_embeddings)
        train_labels = _to_numpy(layer_labels).astype(np.int64)
        train_ids = _to_numpy(layer_ids).astype(np.int64)

        self.index = Indexer(self.embedding_dim)
        label_dict = {}
        for idx, label in zip(train_ids.tolist(), train_labels.tolist()):
            label_dict[int(idx)] = 1 if int(label) == int(self.human_index) else 0
        self.index.label_dict = label_dict
        self.index.index_data(train_ids.tolist(), train_embeddings.astype(np.float32))

        self.layer = layer

    def set_layer(self, layer: int) -> None:
        """Switch the active database layer used for inference."""
        if layer == self.layer:
            return
        self._configure_layer(layer)

    def get_available_layers(self) -> List[int]:
        return list(self.available_layers)

    @torch.no_grad()
    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        dataset = TextDataset(texts)
        if len(dataset) == 0:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=lambda batch: tuple(zip(*batch)),
        )

        all_embeddings: List[torch.Tensor] = []
        all_indices: List[int] = []
        for texts_batch, indices_batch in tqdm(
            dataloader, desc="Encoding", leave=False
        ):
            encoded_batch = self.tokenizer.batch_encode_plus(
                list(texts_batch),
                return_tensors="pt",
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
            )
            encoded_batch = {k: v.to(self.device) for k, v in encoded_batch.items()}
            embeddings = self.model(encoded_batch, hidden_states=True)
            embeddings = F.normalize(embeddings, dim=-1)
            all_embeddings.append(embeddings.cpu())
            all_indices.extend(indices_batch)

        stacked = torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty(0)
        if stacked.numel() == 0:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        order = torch.tensor(all_indices, dtype=torch.long)
        if order.numel() != stacked.shape[0]:
            raise RuntimeError("Index and embedding counts do not match.")
        sorted_indices = torch.argsort(order)
        stacked = stacked[sorted_indices]
        stacked = stacked.permute(1, 0, 2)
        selected_layer = stacked[self.layer]
        return selected_layer.numpy().astype(np.float32)

    def predict(self, texts: Sequence[str]) -> List[Prediction]:
        texts_list = [str(text) for text in texts]
        embeddings = self._encode(texts_list)
        if embeddings.shape[0] == 0:
            return []

        results = self.index.search_knn(
            embeddings,
            self.top_k,
            index_batch_size=max(1, min(self.top_k, 128)),
        )

        predictions: List[Prediction] = []
        for text, (_ids, scores, labels) in zip(texts_list, results):
            scores_tensor = torch.from_numpy(np.asarray(scores))
            weights = torch.softmax(scores_tensor, dim=0)
            label_tensor = torch.tensor(labels, dtype=torch.float32)
            probability_human = float(torch.dot(weights, label_tensor).item())
            probability_human = max(0.0, min(1.0, probability_human))
            probability_ai = float(max(0.0, min(1.0, 1.0 - probability_human)))
            label = "Human" if probability_human >= self.threshold else "AI"
            predictions.append(
                Prediction(
                    text=text,
                    probability_ai=probability_ai,
                    probability_human=probability_human,
                    label=label,
                )
            )
        return predictions

    # ======================================================================
    # IMAGE INFERENCE
    # ======================================================================

    def _load_clip_embedding(self, path: Union[str, Path]) -> np.ndarray:
        """Load a single pre-computed CLIP embedding from .npy or .npz file."""
        path = Path(path)
        if path.suffix == ".npz":
            data = np.load(path)
            # Attempt common key names; fall back to first array
            for key in ("embedding", "emb", "arr_0"):
                if key in data:
                    return data[key].astype(np.float32).flatten()
            # Default to first array if no matching key
            return data[list(data.keys())[0]].astype(np.float32).flatten()
        else:
            return np.load(path).astype(np.float32).flatten()

    @torch.no_grad()
    def _encode_images(
        self,
        embedding_paths: Sequence[Union[str, Path]],
    ) -> np.ndarray:
        """Project pre-computed CLIP embeddings through the trained projector.

        Args:
            embedding_paths: Paths to ``.npy`` or ``.npz`` files containing
                             pre-computed CLIP embeddings.

        Returns:
            ``(N, embedding_dim)`` float32 array of projected, L2-normalised
            embeddings ready for kNN search.
        """
        if self.projector is None:
            raise RuntimeError(
                "Image inference requires a projector.  "
                "Initialise Detector with projector_path=..."
            )

        all_embeddings: List[torch.Tensor] = []

        # Process in batches
        for i in range(0, len(embedding_paths), self.batch_size):
            batch_paths = embedding_paths[i : i + self.batch_size]
            batch_embs = [self._load_clip_embedding(p) for p in batch_paths]
            batch_tensor = torch.from_numpy(np.stack(batch_embs, axis=0))

            # L2-normalise input (same as training)
            batch_tensor = F.normalize(batch_tensor, dim=-1).to(self.device)

            # Project
            projected = self.projector(batch_tensor, normalize=True)
            all_embeddings.append(projected.cpu())

        if not all_embeddings:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        stacked = torch.cat(all_embeddings, dim=0)
        return stacked.numpy().astype(np.float32)

    def predict_images(
        self,
        embedding_paths: Sequence[Union[str, Path]],
    ) -> List[ImagePrediction]:
        """Run kNN predictions on image (CLIP) embeddings.

        Args:
            embedding_paths: Paths to ``.npy`` or ``.npz`` files containing
                             pre-computed CLIP embeddings.

        Returns:
            List of ``ImagePrediction`` results.
        """
        paths_list = [str(p) for p in embedding_paths]
        embeddings = self._encode_images(paths_list)
        if embeddings.shape[0] == 0:
            return []

        results = self.index.search_knn(
            embeddings,
            self.top_k,
            index_batch_size=max(1, min(self.top_k, 128)),
        )

        predictions: List[ImagePrediction] = []
        for path, (_ids, scores, labels) in zip(paths_list, results):
            scores_tensor = torch.from_numpy(np.asarray(scores))
            weights = torch.softmax(scores_tensor, dim=0)
            label_tensor = torch.tensor(labels, dtype=torch.float32)
            probability_human = float(torch.dot(weights, label_tensor).item())
            probability_human = max(0.0, min(1.0, probability_human))
            probability_ai = float(max(0.0, min(1.0, 1.0 - probability_human)))
            label = "Real" if probability_human >= self.threshold else "AI"
            predictions.append(
                ImagePrediction(
                    image_path=path,
                    probability_ai=probability_ai,
                    probability_human=probability_human,
                    label=label,
                )
            )
        return predictions

