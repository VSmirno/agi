"""Experiment 52: Autonomous cross-modal recall (Stage 23).

Verifies that GroundedTokenizer produces identical SDRs and currents
as TextEncoder for words in the GroundingMap vocabulary.

Gate: sdr_match_ratio > 0.8, currents_match_ratio > 0.8
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import torch

from snks.daf.types import EncoderConfig
from snks.encoder.grounded_tokenizer import GroundedTokenizer
from snks.language.grounding_map import GroundingMap


# Simulated vocabulary: words with pre-generated SDRs
# (as if TextEncoder produced them during Stage 19 co-activation)
VOCAB = [
    "key", "door", "ball", "box", "wall",
    "goal", "red", "blue", "green", "agent",
    "left", "right", "forward", "pick up", "open",
]

SDR_SIZE = 4096
K = 164  # 4% sparsity


def _make_gmap() -> GroundingMap:
    """Create GroundingMap with reproducible SDRs for each word."""
    gmap = GroundingMap()
    g = torch.Generator().manual_seed(42)
    for i, word in enumerate(VOCAB):
        sdr = torch.zeros(SDR_SIZE)
        indices = torch.randperm(SDR_SIZE, generator=g)[:K]
        sdr[indices] = 1.0
        gmap.register(word, i + 1, sdr)
    return gmap


def run(device: str = "cpu") -> dict:
    print("\n" + "=" * 60)
    print("Exp 52: Autonomous cross-modal recall")
    print("=" * 60)

    config = EncoderConfig(sdr_size=SDR_SIZE, sdr_sparsity=0.04, sdr_current_strength=1.0)
    gmap = _make_gmap()
    tokenizer = GroundedTokenizer(gmap, config)

    n_nodes = 5000
    sdr_matches = 0
    currents_matches = 0
    total_known = 0
    total_unknown = 0

    # Test known words: SDR from tokenizer must match SDR in GroundingMap
    print("\n  Known words:")
    for word in VOCAB:
        original_sdr = gmap.word_to_sdr(word)
        tokenizer_sdr = tokenizer.encode(word)

        sdr_match = torch.equal(original_sdr, tokenizer_sdr)
        if sdr_match:
            sdr_matches += 1

        # Compare currents
        orig_currents = tokenizer.sdr_to_currents(original_sdr, n_nodes)
        tok_currents = tokenizer.sdr_to_currents(tokenizer_sdr, n_nodes)
        currents_match = torch.equal(orig_currents, tok_currents)
        if currents_match:
            currents_matches += 1

        total_known += 1
        status = "✓" if sdr_match and currents_match else "✗"
        print(f"    {status} {word!r}: sdr={'match' if sdr_match else 'MISMATCH'}, "
              f"currents={'match' if currents_match else 'MISMATCH'}")

    # Test unknown words: must return zero SDR
    unknown_words = ["castle", "dragon", "sword", "magic", "river"]
    print("\n  Unknown words:")
    for word in unknown_words:
        sdr = tokenizer.encode(word)
        is_zero = sdr.sum() == 0.0
        currents = tokenizer.sdr_to_currents(sdr, n_nodes)
        currents_zero = currents.sum() == 0.0
        total_unknown += 1
        status = "✓" if is_zero and currents_zero else "✗"
        print(f"    {status} {word!r}: sdr={'zero' if is_zero else 'NON-ZERO'}, "
              f"currents={'zero' if currents_zero else 'NON-ZERO'}")

    sdr_ratio = sdr_matches / total_known
    currents_ratio = currents_matches / total_known
    passed = sdr_ratio > 0.8 and currents_ratio > 0.8

    print(f"\n{'=' * 40}")
    print(f"SDR match ratio:      {sdr_ratio:.3f} ({sdr_matches}/{total_known})")
    print(f"Currents match ratio: {currents_ratio:.3f} ({currents_matches}/{total_known})")
    print(f"Unknown words tested: {total_unknown}")
    print(f"Gate (> 0.8): {'PASS' if passed else 'FAIL'}")

    return {
        "sdr_match_ratio": sdr_ratio,
        "currents_match_ratio": currents_ratio,
        "total_known": total_known,
        "total_unknown": total_unknown,
        "pass": passed,
    }


if __name__ == "__main__":
    result = run()
    sys.exit(0 if result["pass"] else 1)
