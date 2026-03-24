"""Experiment 11: Семантическая близость.

STS-B dev set (первые 200 пар).
Метрика: Spearman ρ между SDR overlap (Jaccard) и cosine similarity из разметки.
Gate: Spearman ρ > 0.7.
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from scipy.stats import spearmanr

from snks.daf.types import EncoderConfig
from snks.encoder.text_encoder import TextEncoder


def run() -> float:
    from datasets import load_dataset

    config = EncoderConfig()
    encoder = TextEncoder(config)

    dataset = load_dataset("stsb_multi_mt", "en", split="dev", trust_remote_code=True)
    pairs = list(dataset)[:200]

    sdr_overlaps = []
    cosine_sims = []

    for pair in pairs:
        sent1 = pair["sentence1"]
        sent2 = pair["sentence2"]
        # STS-B scores are 0–5; normalize to 0–1
        score = float(pair["similarity_score"]) / 5.0

        sdr1 = encoder.encode(sent1)
        sdr2 = encoder.encode(sent2)

        # Jaccard: |A∩B| / |A∪B|
        intersection = (sdr1 * sdr2).sum().item()
        union = ((sdr1 + sdr2) > 0).float().sum().item()
        jaccard = intersection / union if union > 0 else 0.0

        sdr_overlaps.append(jaccard)
        cosine_sims.append(score)

    rho, pvalue = spearmanr(sdr_overlaps, cosine_sims)

    print("Exp 11: Семантическая близость")
    print(f"  Пар: {len(pairs)}")
    print(f"  Spearman ρ: {rho:.4f}  (p={pvalue:.2e})")
    print(f"  Gate (ρ > 0.7): {'PASS' if rho > 0.7 else 'FAIL'}")
    return float(rho)


if __name__ == "__main__":
    run()
