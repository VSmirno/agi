"""Stage 66: Decode head — codebook index → object type lookup table.

Instead of training a neural network, builds a direct mapping from
VQ codebook indices to object types using semantic map ground truth.
Zero-shot at test time: look up patch index → object type.
"""

from __future__ import annotations

import math
from collections import Counter

import torch
import torch.nn as nn
import numpy as np

from snks.agent.crafter_pixel_env import NEAR_OBJECTS, INVENTORY_ITEMS, SEMANTIC_NAMES


# Include "empty" as a class for near/standing
NEAR_CLASSES = ["empty"] + NEAR_OBJECTS
NEAR_TO_IDX = {name: i for i, name in enumerate(NEAR_CLASSES)}

# Terrain types that are NOT interesting for "near" detection
TERRAIN_TYPES = {"grass", "path", "sand", "water", "lava", "unknown"}


class DecodeHead:
    """Lookup-table decoder: codebook index → object type.

    Built from (semantic_map, codebook_indices) pairs during training.
    At test time: look up agent-adjacent patch indices → detect near object.

    No neural network. No gradient. Pure counting.
    """

    def __init__(self, codebook_size: int = 4096):
        self.codebook_size = codebook_size
        # For each codebook index: Counter of object types seen
        self.index_to_votes: list[Counter] = [Counter() for _ in range(codebook_size)]
        # Resolved mapping after build()
        self.index_to_type: dict[int, str] = {}
        self._built = False

    def observe(
        self,
        indices: np.ndarray | torch.Tensor,
        semantic: np.ndarray,
        patch_size: int = 8,
    ) -> None:
        """Record one frame's codebook indices + semantic map.

        For each patch, find the majority object type from the semantic map
        and vote for that type on the codebook index.

        Args:
            indices: (64,) codebook indices for each patch.
            semantic: (64, 64) semantic map from Crafter info.
            patch_size: patch size (8).
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()

        grid = 64 // patch_size  # 8
        for r in range(grid):
            for c in range(grid):
                patch_idx = r * grid + c
                cb_idx = int(indices[patch_idx])

                # Get majority object type in this patch from semantic map
                sem_patch = semantic[
                    r * patch_size:(r + 1) * patch_size,
                    c * patch_size:(c + 1) * patch_size,
                ]
                types = Counter()
                for val in sem_patch.flat:
                    name = SEMANTIC_NAMES.get(int(val), "unknown")
                    types[name] += 1
                majority_type = types.most_common(1)[0][0]

                self.index_to_votes[cb_idx][majority_type] += 1

    def build(self) -> dict[str, int]:
        """Resolve lookup table from accumulated votes.

        Returns:
            Stats: {object_type: count_of_indices_mapped}.
        """
        self.index_to_type = {}
        stats: dict[str, int] = Counter()

        for idx in range(self.codebook_size):
            votes = self.index_to_votes[idx]
            if not votes:
                continue
            majority = votes.most_common(1)[0][0]
            self.index_to_type[idx] = majority
            stats[majority] += 1

        self._built = True
        return dict(stats)

    def decode_near(self, agent_indices: torch.Tensor | np.ndarray) -> str:
        """Detect nearest interesting object from agent-adjacent patch indices.

        Args:
            agent_indices: (9,) codebook indices for agent-adjacent patches.

        Returns:
            Name of nearest non-terrain object, or "empty".
        """
        if isinstance(agent_indices, torch.Tensor):
            agent_indices = agent_indices.cpu().numpy()

        for idx in agent_indices:
            obj_type = self.index_to_type.get(int(idx), "unknown")
            if obj_type in NEAR_OBJECTS:
                return obj_type

        return "empty"

    def decode_situation_key(
        self,
        agent_indices: torch.Tensor,
        z_local: torch.Tensor | None = None,
    ) -> tuple[str, float]:
        """Decode into (situation_key_string, decode_certainty).

        Args:
            agent_indices: (9,) codebook indices for agent-adjacent patches.
            z_local: unused, kept for API compatibility.

        Returns:
            (key, certainty) where certainty ∈ [0, 1].
        """
        near = self.decode_near(agent_indices)
        certainty = 0.9 if near != "empty" else 0.5

        situation: dict[str, str] = {"domain": "crafter", "near": near}

        from snks.agent.crafter_encoder import make_crafter_key
        key = make_crafter_key(situation, "")

        return key, certainty

    def parameters(self):
        """API compatibility — no parameters."""
        return []

    def to(self, device):
        """API compatibility — nothing to move."""
        return self

    def train_step(self, *args, **kwargs):
        """API compatibility — no training needed."""
        return {"near_acc": 0.0, "near_loss": 0.0, "inv_loss": 0.0, "total_loss": 0.0}


def symbolic_to_gt_tensors(
    symbolic_obs: dict[str, str],
) -> tuple[int, list[float]]:
    """Convert symbolic observation to ground truth tensors for training.

    Returns:
        (near_idx, inventory_vec) where:
        - near_idx: int index into NEAR_CLASSES
        - inventory_vec: list of 0/1 floats for each INVENTORY_ITEM
    """
    near = symbolic_obs.get("near", "empty")
    near_idx = NEAR_TO_IDX.get(near, 0)  # default to "empty"

    inventory_vec = []
    for item in INVENTORY_ITEMS:
        count = int(symbolic_obs.get(f"has_{item}", "0"))
        inventory_vec.append(1.0 if count > 0 else 0.0)

    return near_idx, inventory_vec
