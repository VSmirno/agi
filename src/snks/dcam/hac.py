"""HAC (Holographic Associative Codes) engine.

Implements HRR (Holographic Reduced Representations) operations:
bind/unbind via circular convolution/correlation in FFT domain,
bundle (superposition), permute (cyclic shift), fractional power encoding.
"""

from __future__ import annotations

import torch
from torch import Tensor


class HACEngine:
    """Holographic Associative Codes engine using FFT-based operations."""

    def __init__(self, dim: int = 2048, device: torch.device | None = None) -> None:
        self.dim = dim
        self.device = device or torch.device("cpu")
        # Base vector for fractional power encoding (encode_scalar)
        self._scalar_base = self.random_vector()

    def random_vector(self) -> Tensor:
        """Generate a random unit-norm vector of dimension D."""
        v = torch.randn(self.dim, device=self.device, dtype=torch.float32)
        return v / v.norm().clamp(min=1e-8)

    def bind(self, a: Tensor, b: Tensor) -> Tensor:
        """Circular convolution via FFT: encodes a role-filler pair."""
        fa = torch.fft.rfft(a.float())
        fb = torch.fft.rfft(b.float())
        return torch.fft.irfft(fa * fb, n=self.dim)

    def unbind(self, a: Tensor, bound: Tensor) -> Tensor:
        """Circular correlation with power-spectrum normalization."""
        fa = torch.fft.rfft(a.float())
        fb = torch.fft.rfft(bound.float())
        power = (fa * fa.conj()).real.clamp(min=1e-10)
        return torch.fft.irfft(fa.conj() * fb / power, n=self.dim)

    def bundle(self, vectors: list[Tensor]) -> Tensor:
        """Superposition: sum + normalize."""
        s = torch.stack(vectors).sum(dim=0)
        norm = s.norm().clamp(min=1e-8)
        return s / norm

    def permute(self, v: Tensor, k: int) -> Tensor:
        """Cyclic shift by k positions (for positional encoding)."""
        return torch.roll(v, shifts=k)

    def encode_scalar(self, value: float) -> Tensor:
        """Fractional power encoding: continuous scalar → HAC vector.

        sim(encode(x), encode(y)) decreases monotonically with |x - y|.
        """
        base_fft = torch.fft.rfft(self._scalar_base.float())
        angles = torch.angle(base_fft)
        magnitudes = torch.abs(base_fft)
        result_fft = magnitudes * torch.exp(1j * angles * value)
        return torch.fft.irfft(result_fft, n=self.dim)

    def similarity(self, a: Tensor, b: Tensor) -> float:
        """Cosine similarity between two vectors."""
        a_f = a.float()
        b_f = b.float()
        return torch.nn.functional.cosine_similarity(
            a_f.unsqueeze(0), b_f.unsqueeze(0)
        ).item()

    def batch_bind(self, A: Tensor, B: Tensor) -> Tensor:
        """Batch circular convolution: (batch, D) × (batch, D) → (batch, D)."""
        fA = torch.fft.rfft(A.float(), n=self.dim)
        fB = torch.fft.rfft(B.float(), n=self.dim)
        return torch.fft.irfft(fA * fB, n=self.dim)

    def batch_similarity(self, query: Tensor, keys: Tensor) -> Tensor:
        """Cosine similarity of query (D,) against keys (M, D) → (M,)."""
        q = query.float().unsqueeze(0)  # (1, D)
        k = keys.float()                # (M, D)
        return torch.nn.functional.cosine_similarity(q, k, dim=1)
