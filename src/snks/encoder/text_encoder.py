"""Text encoder: str → SDR → DAF external currents."""

import torch

from snks.daf.types import EncoderConfig
from snks.encoder.sdr import kwta


class TextEncoder:
    """Encodes text into sparse distributed representations.

    Pipeline: text → sentence-transformers (384-dim) → RandomProjection (4096-dim) → k-WTA → SDR.

    The sentence-transformer model is frozen (no gradient updates).
    The random projection matrix is fixed (seed=42, normalized columns).
    """

    def __init__(self, config: EncoderConfig, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.config = config
        self.device = device or "cpu"

        # Frozen sentence-transformer
        self._st_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
        self._st_model.eval()

        # Fixed random projection matrix: (384, sdr_size), columns normalized to unit L2 norm
        g = torch.Generator()
        g.manual_seed(42)
        proj = torch.randn(384, config.sdr_size, generator=g)
        proj = proj / proj.norm(dim=0, keepdim=True)  # normalize columns
        self._proj = proj.to(self.device)

        self.k = round(config.sdr_size * config.sdr_sparsity)

    def encode(self, text: str) -> torch.Tensor:
        """Encode text to binary SDR.

        Args:
            text: input string.

        Returns:
            (sdr_size,) binary SDR, float32.
        """
        with torch.no_grad():
            emb = self._st_model.encode(text, convert_to_tensor=True, show_progress_bar=False)
            emb = emb.to(self.device).float()  # (384,)
            projected = emb @ self._proj        # (sdr_size,)
            sdr = kwta(projected, self.k)       # binary (sdr_size,)
        return sdr

    def sdr_to_currents(self, sdr: torch.Tensor, n_nodes: int) -> torch.Tensor:
        """Map SDR to external currents for DAF engine.

        Same mapping as VisualEncoder: multiplicative hash distributes nodes
        uniformly across SDR bits. Active bits inject current_strength into channel 0.

        Args:
            sdr: (sdr_size,) binary SDR.
            n_nodes: number of DAF nodes.

        Returns:
            (n_nodes, 8) external currents tensor.
        """
        PRIME = 2654435761
        node_sdr_idx = (torch.arange(n_nodes, device=sdr.device) * PRIME) % self.config.sdr_size

        currents = torch.zeros(n_nodes, 8, device=sdr.device)
        currents[:, 0] = sdr[node_sdr_idx] * self.config.sdr_current_strength
        return currents
