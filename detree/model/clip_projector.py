"""CLIP-to-DeTree projection head for aligning image embeddings to the text
embedding space.

The ``CLIPProjector`` learns a lightweight mapping from CLIP's representation
space into the *same* vector space that DeTree uses for its kNN text-based
detector.  Once trained, CLIP embeddings projected through this head can be
stored alongside (or merged with) the existing DeTree text embedding database
so that a **single** FAISS index handles both modalities.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPProjector(nn.Module):
    """Small trainable MLP that projects CLIP image embeddings into the DeTree
    text embedding space.

    Architecture::

        Linear(clip_dim → hidden_dim) → GELU → LayerNorm(hidden_dim)
        → Linear(hidden_dim → target_dim)

    The output is L2-normalised by default so that inner-product search in
    FAISS is equivalent to cosine similarity – matching the convention used
    by ``detree.cli.embeddings`` when it stores text embeddings.

    Args:
        clip_dim:   Dimensionality of the incoming CLIP embeddings (e.g. 512
                    for ViT-B/32, 768 for ViT-L/14).
        target_dim: Dimensionality of the DeTree text embedding space (e.g.
                    1024 for RoBERTa-large).
        hidden_dim: Width of the hidden layer.  Defaults to *target_dim*.
    """

    def __init__(
        self,
        clip_dim: int = 768,
        target_dim: int = 1024,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = target_dim

        self.clip_dim = clip_dim
        self.target_dim = target_dim
        self.hidden_dim = hidden_dim

        self.projector = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, target_dim),
        )
        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for module in self.projector:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, *, normalize: bool = True) -> torch.Tensor:
        """Project CLIP embeddings into the DeTree target space.

        Args:
            x:         Tensor of shape ``(batch, clip_dim)``.
            normalize: If ``True``, L2-normalise the output vectors.  This
                       should always be ``True`` when the embeddings are
                       destined for FAISS inner-product search.

        Returns:
            Tensor of shape ``(batch, target_dim)``.
        """
        out = self.projector(x)
        if normalize:
            out = F.normalize(out, dim=-1)
        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_pretrained(self, save_directory: str) -> None:
        """Save the projector weights and architecture hyper-parameters."""
        os.makedirs(save_directory, exist_ok=True)
        payload = {
            "state_dict": self.state_dict(),
            "clip_dim": self.clip_dim,
            "target_dim": self.target_dim,
            "hidden_dim": self.hidden_dim,
        }
        torch.save(payload, os.path.join(save_directory, "clip_projector.pt"))

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        *,
        device: str = "cpu",
    ) -> "CLIPProjector":
        """Load a previously saved ``CLIPProjector``.

        The hyper-parameters (``clip_dim``, ``target_dim``, ``hidden_dim``)
        are restored automatically from the checkpoint.
        """
        ckpt_path = os.path.join(path, "clip_projector.pt")
        state = torch.load(ckpt_path, map_location=device)
        model = cls(
            clip_dim=state["clip_dim"],
            target_dim=state["target_dim"],
            hidden_dim=state["hidden_dim"],
        )
        model.load_state_dict(state["state_dict"])
        return model
