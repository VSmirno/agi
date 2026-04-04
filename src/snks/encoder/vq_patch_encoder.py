"""Stage 66: VQ Patch Codebook Encoder — pixels to latent + binary VSA.

Memory-heavy, GPU-light design:
- Patchify 64×64 RGB → 64 patches of 8×8
- Codebook lookup (cosine similarity) → quantized indices
- Embedding table → z_real (2048-dim float, for predictor + decode head)
- VSA bind/bundle from indices → z_vsa (2048-dim binary, for SDM hippocampus)
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn


class EncoderOutput(NamedTuple):
    z_real: torch.Tensor    # (B, embed_dim) float — mean of ALL patches, for predictor
    z_vsa: torch.Tensor     # (B, vsa_dim) binary {0,1} — for SDM hippocampus
    indices: torch.Tensor   # (B, n_patches) int — codebook indices
    z_local: torch.Tensor   # (B, embed_dim) float — mean of AGENT-ADJACENT patches, for decode head


class VQPatchEncoder(nn.Module):
    """VQ Patch Codebook encoder: (3, 64, 64) → z_real + z_vsa.

    Minimal GPU compute: cosine similarity lookup + mean pooling.
    Heavy on RAM: codebook + embedding table + VSA vectors.
    """

    def __init__(
        self,
        patch_size: int = 8,
        img_size: int = 64,
        codebook_size: int = 4096,
        embed_dim: int = 2048,
        vsa_dim: int = 2048,
        seed: int = 42,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        self.vsa_dim = vsa_dim
        self.n_patches = (img_size // patch_size) ** 2  # 64
        self.patch_dim = 3 * patch_size * patch_size     # 192

        dev = torch.device(device) if device else torch.device("cpu")

        # Codebook prototypes — NOT a parameter, updated via EMA
        self.register_buffer(
            "codebook",
            torch.randn(codebook_size, self.patch_dim, device=dev),
        )
        # Normalize codebook for cosine similarity
        with torch.no_grad():
            self.codebook.div_(self.codebook.norm(dim=1, keepdim=True).clamp(min=1e-8))

        # Embedding table — trainable via straight-through
        self.embeddings = nn.Embedding(codebook_size, embed_dim)

        # Usage tracking for dead entry reset
        self.register_buffer(
            "usage_counts",
            torch.zeros(codebook_size, dtype=torch.long, device=dev),
        )

        # Binary VSA vectors for codebook entries and patch positions
        rng = torch.Generator(device="cpu")
        rng.manual_seed(seed)
        codebook_vsa = torch.randint(
            0, 2, (codebook_size, vsa_dim), dtype=torch.float32, generator=rng,
        )
        position_roles = torch.randint(
            0, 2, (self.n_patches, vsa_dim), dtype=torch.float32, generator=rng,
        )
        self.register_buffer("codebook_vsa", codebook_vsa.to(dev))
        self.register_buffer("position_roles", position_roles.to(dev))

        self._initialized = False

    def _patchify(self, pixels: torch.Tensor) -> torch.Tensor:
        """(B, 3, 64, 64) → (B, n_patches, patch_dim)."""
        B = pixels.shape[0]
        ps = self.patch_size
        # (B, 3, 8, 8, 8, 8) → (B, 8, 8, 3, 8, 8) → (B, 64, 192)
        x = pixels.unfold(2, ps, ps).unfold(3, ps, ps)  # (B, 3, H//ps, W//ps, ps, ps)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()     # (B, H//ps, W//ps, 3, ps, ps)
        return x.view(B, self.n_patches, self.patch_dim)

    def forward(self, pixels: torch.Tensor) -> EncoderOutput:
        """Encode pixel observations.

        Args:
            pixels: (B, 3, 64, 64) or (3, 64, 64) RGB float32 in [0, 1].

        Returns:
            EncoderOutput(z_real, z_vsa, indices).
        """
        single = pixels.dim() == 3
        if single:
            pixels = pixels.unsqueeze(0)

        B = pixels.shape[0]
        patches = self._patchify(pixels)  # (B, 64, 192)

        # Cosine similarity with codebook
        patches_norm = patches / patches.norm(dim=2, keepdim=True).clamp(min=1e-8)
        sim = patches_norm @ self.codebook.T  # (B, 64, 4096)

        # Hard quantization
        indices = sim.argmax(dim=2)  # (B, 64)

        # Embedding lookup + straight-through estimator
        embeds = self.embeddings(indices)  # (B, 64, 2048)
        z_real = embeds.mean(dim=1)        # (B, 2048)

        # z_local: mean of agent-adjacent patches only (scene-invariant)
        # Agent is at center (32,32) → patch grid (4,4). Adjacent = 3×3 around center.
        z_local = embeds[:, self.AGENT_PATCHES, :].mean(dim=1)  # (B, 2048)

        # Binary VSA address from codebook indices
        z_vsa = self._build_vsa(indices)  # (B, 2048)

        if single:
            return EncoderOutput(
                z_real.squeeze(0), z_vsa.squeeze(0),
                indices.squeeze(0), z_local.squeeze(0),
            )
        return EncoderOutput(z_real, z_vsa, indices, z_local)

    # Agent-adjacent patches: 3×3 around center (row 3-5, col 3-5 in 8×8 grid)
    # Agent at pixel (32,32) = patch (4,4). "Near" objects are in adjacent patches.
    # These patches are scene-invariant — same object type → same embedding.
    AGENT_PATCHES: list[int] = [
        r * 8 + c for r in range(3, 6) for c in range(3, 6)
    ]  # 9 patches around agent

    # Central 4×4 patch indices (rows 2-5, cols 2-5 in the 8×8 grid)
    # Used for z_vsa — wider context than AGENT_PATCHES but still local.
    CENTRAL_PATCHES: list[int] = [
        r * 8 + c for r in range(2, 6) for c in range(2, 6)
    ]  # 16 patches

    def _build_vsa(self, indices: torch.Tensor) -> torch.Tensor:
        """Build binary VSA vector from CENTRAL codebook indices via bind + bundle.

        Uses only central 4×4 patches (16 out of 64) for scene-invariant encoding.
        Peripheral terrain is ignored — only nearby objects matter for SDM matching.
        """
        B = indices.shape[0]
        bound_vecs = []

        for p in self.CENTRAL_PATCHES:
            cb_vecs = self.codebook_vsa[indices[:, p]]  # (B, vsa_dim)
            pos_vec = self.position_roles[p]             # (vsa_dim,)
            bound = (cb_vecs + pos_vec.unsqueeze(0)) % 2  # XOR bind
            bound_vecs.append(bound)

        n_central = len(self.CENTRAL_PATCHES)
        stacked = torch.stack(bound_vecs, dim=1)  # (B, 16, vsa_dim)
        summed = stacked.sum(dim=1)                 # (B, vsa_dim)
        z_vsa = (summed > n_central / 2).float()    # majority vote
        return z_vsa

    @torch.no_grad()
    def update_codebook_ema(
        self,
        patches: torch.Tensor,
        indices: torch.Tensor,
        momentum: float = 0.99,
    ) -> None:
        """EMA update of codebook prototypes. Call after each train step."""
        # patches: (B, n_patches, patch_dim), indices: (B, n_patches)
        flat_patches = patches.view(-1, self.patch_dim)   # (B*64, 192)
        flat_indices = indices.view(-1)                     # (B*64,)

        # Accumulate per-entry sums
        for idx in range(self.codebook_size):
            mask = flat_indices == idx
            count = mask.sum().item()
            if count == 0:
                continue
            self.usage_counts[idx] += count
            centroid = flat_patches[mask].mean(dim=0)
            self.codebook[idx] = momentum * self.codebook[idx] + (1 - momentum) * centroid

        # Re-normalize
        self.codebook.div_(self.codebook.norm(dim=1, keepdim=True).clamp(min=1e-8))

    @torch.no_grad()
    def reset_dead_entries(self, patches: torch.Tensor, min_usage: int = 2) -> int:
        """Reset codebook entries with usage < min_usage to random batch patches.

        Call after each epoch. Returns number of entries reset.
        """
        dead = (self.usage_counts < min_usage).nonzero(as_tuple=True)[0]
        if len(dead) == 0:
            return 0

        flat = patches.view(-1, self.patch_dim)
        n_available = flat.shape[0]

        for i, idx in enumerate(dead):
            donor = flat[i % n_available]
            self.codebook[idx] = donor / donor.norm().clamp(min=1e-8)

        # Reset usage counts for new epoch
        self.usage_counts.zero_()
        return len(dead)

    @torch.no_grad()
    def init_codebook_kmeans(self, patches: torch.Tensor, n_iter: int = 10) -> None:
        """Initialize codebook via k-means on a batch of patches.

        Args:
            patches: (B, n_patches, patch_dim) or (N, patch_dim).
        """
        if self._initialized:
            return

        if patches.dim() == 3:
            patches = patches.view(-1, self.patch_dim)

        # Normalize
        patches = patches / patches.norm(dim=1, keepdim=True).clamp(min=1e-8)

        # Random init: sample from patches
        n = patches.shape[0]
        perm = torch.randperm(n, device=patches.device)[:self.codebook_size]
        if len(perm) < self.codebook_size:
            # Repeat if not enough patches
            repeats = (self.codebook_size // len(perm)) + 1
            perm = perm.repeat(repeats)[:self.codebook_size]
        self.codebook.copy_(patches[perm])

        # K-means iterations
        for _ in range(n_iter):
            sim = patches @ self.codebook.T  # (N, codebook_size)
            assignments = sim.argmax(dim=1)
            for k in range(self.codebook_size):
                mask = assignments == k
                if mask.any():
                    centroid = patches[mask].mean(dim=0)
                    self.codebook[k] = centroid / centroid.norm().clamp(min=1e-8)

        self._initialized = True

    @property
    def codebook_utilization(self) -> float:
        """Fraction of codebook entries with nonzero usage."""
        return (self.usage_counts > 0).float().mean().item()
