"""Stage 66: Decode head — z_real → situation key for neocortex.

Supervised classification heads that decode the VQ encoder's real-valued
latent into symbolic situation components (near object, inventory).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from snks.agent.crafter_pixel_env import NEAR_OBJECTS, INVENTORY_ITEMS


# Include "empty" as a class for near/standing
NEAR_CLASSES = ["empty"] + NEAR_OBJECTS
NEAR_TO_IDX = {name: i for i, name in enumerate(NEAR_CLASSES)}


class DecodeHead(nn.Module):
    """Decode agent-adjacent patch indices into symbolic situation.

    Uses codebook indices of 9 patches around agent, NOT z_real/z_local.
    Each codebook index maps to a learned embedding → classify near object.
    Scene-invariant: same object type → same codebook index → same prediction.

    Heads:
    - near_object: what's adjacent to agent (softmax over NEAR_CLASSES)
    - inventory: which items agent has (sigmoid per item, multi-label)
    """

    def __init__(
        self,
        codebook_size: int = 4096,
        n_agent_patches: int = 9,
        patch_embed_dim: int = 32,
        hidden: int = 128,
        z_dim: int = 2048,
    ):
        super().__init__()
        self.n_near = len(NEAR_CLASSES)
        self.n_items = len(INVENTORY_ITEMS)
        self.n_agent_patches = n_agent_patches

        # Small embedding for each codebook entry (not the large 2048-dim one)
        self.patch_embed = nn.Embedding(codebook_size, patch_embed_dim)

        # Near head: from 9 patch embeddings
        self.near_backbone = nn.Sequential(
            nn.Linear(n_agent_patches * patch_embed_dim, hidden),
            nn.ReLU(),
        )
        self.head_near = nn.Linear(hidden, self.n_near)

        # Inventory head: from z_local (needs wider context than 9 patches)
        self.inv_backbone = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ReLU(),
        )
        self.head_inventory = nn.Linear(hidden, self.n_items)

    def forward(
        self,
        agent_indices: torch.Tensor,
        z_local: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            agent_indices: (B, 9) or (9,) codebook indices for agent-adjacent patches.
            z_local: (B, z_dim) or (z_dim,) for inventory prediction. Optional.

        Returns:
            {"near_logits": (B, n_near), "inventory_logits": (B, n_items)}
        """
        single = agent_indices.dim() == 1
        if single:
            agent_indices = agent_indices.unsqueeze(0)
            if z_local is not None:
                z_local = z_local.unsqueeze(0)

        # Near: from patch codebook indices
        patch_embs = self.patch_embed(agent_indices)  # (B, 9, patch_embed_dim)
        patch_flat = patch_embs.view(patch_embs.shape[0], -1)  # (B, 9*32)
        h_near = self.near_backbone(patch_flat)
        near_logits = self.head_near(h_near)

        # Inventory: from z_local if available
        if z_local is not None:
            h_inv = self.inv_backbone(z_local)
            inv_logits = self.head_inventory(h_inv)
        else:
            inv_logits = torch.zeros(
                agent_indices.shape[0], self.n_items,
                device=agent_indices.device,
            )

        result = {"near_logits": near_logits, "inventory_logits": inv_logits}

        if single:
            result = {k: v.squeeze(0) for k, v in result.items()}
        return result

    def decode_situation_key(
        self,
        agent_indices: torch.Tensor,
        z_local: torch.Tensor | None = None,
    ) -> tuple[str, float]:
        """Decode into (situation_key_string, decode_certainty).

        Args:
            agent_indices: (9,) codebook indices for agent-adjacent patches.
            z_local: (z_dim,) for inventory. Optional.

        Returns:
            (key, certainty) where certainty ∈ [0, 1].
        """
        with torch.no_grad():
            out = self.forward(agent_indices, z_local)

        # Near object
        near_probs = torch.softmax(out["near_logits"], dim=-1)
        near_idx = near_probs.argmax().item()
        near_name = NEAR_CLASSES[near_idx]

        # Inventory (high threshold to reduce false positives)
        inv_probs = torch.sigmoid(out["inventory_logits"])
        inv_items: dict[str, int] = {}
        for i, item in enumerate(INVENTORY_ITEMS):
            if inv_probs[i].item() > 0.8:
                inv_items[item] = 1  # binary presence (count not decoded)

        # Build situation key (compatible with make_crafter_key format)
        situation: dict[str, str] = {"domain": "crafter", "near": near_name}
        for item, count in sorted(inv_items.items()):
            situation[f"has_{item}"] = str(count)

        from snks.agent.crafter_encoder import make_crafter_key
        key = make_crafter_key(situation, "")  # action added externally

        # Decode certainty: 1 - normalized entropy of near head
        entropy = -(near_probs * near_probs.clamp(min=1e-8).log()).sum()
        max_entropy = math.log(self.n_near)
        certainty = 1.0 - (entropy.item() / max_entropy)

        return key, certainty

    def train_step(
        self,
        agent_indices: torch.Tensor,
        gt_near: torch.Tensor,
        gt_inventory: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        z_local: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Supervised training step.

        Args:
            agent_indices: (B, 9) codebook indices for agent-adjacent patches.
            gt_near: (B,) int indices into NEAR_CLASSES.
            gt_inventory: (B, n_items) float binary labels.
            optimizer: optimizer for decode head params.
            z_local: (B, z_dim) for inventory. Optional.

        Returns:
            Dict with losses.
        """
        out = self.forward(agent_indices, z_local)

        near_loss = nn.functional.cross_entropy(out["near_logits"], gt_near)
        inv_loss = nn.functional.binary_cross_entropy_with_logits(
            out["inventory_logits"], gt_inventory,
        )

        total = near_loss + inv_loss

        optimizer.zero_grad()
        total.backward()
        optimizer.step()

        # Accuracy
        near_acc = (out["near_logits"].argmax(dim=1) == gt_near).float().mean()

        return {
            "near_loss": near_loss.item(),
            "inv_loss": inv_loss.item(),
            "total_loss": total.item(),
            "near_acc": near_acc.item(),
        }


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
