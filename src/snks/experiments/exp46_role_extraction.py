"""Experiment 46: Role extraction accuracy (Stage 20).

Tests that RoleFillerParser correctly encodes and decodes role-filler
structures across three grammar levels:
  Level 1 — SVO(L): "cat sits on mat"
  Level 2 — ATTR+SVO(L): "red cat sits on mat"
  Level 3 — MiniGrid: "pick up the red key"

Method: for each sentence, parse → sentence_hac, then extract each role
and find best match among all known embeddings (argmax cosine similarity).
Correct = best match is the ground truth filler.

Gate: accuracy > 0.8
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import torch

from snks.dcam.hac import HACEngine
from snks.language.chunker import Chunk, RuleBasedChunker
from snks.language.parser import RoleFillerParser
from snks.language.roles import get_roles, random_hac_vector


# --- Vocabulary: each word gets a fixed random HAC embedding ---

NOUNS = ["cat", "dog", "key", "ball", "door", "box"]
VERBS = ["sits", "runs", "opens", "hits", "sees"]
LOCATIONS = ["mat", "room", "table", "floor"]
ADJECTIVES = ["red", "blue", "green"]
MINIGRID_OBJECTS = ["key", "door", "ball", "box", "wall"]
MINIGRID_ACTIONS = ["pick up", "go to", "open", "toggle"]

# Level 1: SVO(L)
LEVEL1_SENTENCES = [
    ("cat sits on mat", {"AGENT": "cat", "ACTION": "sits", "LOCATION": "mat"}),
    ("dog runs on floor", {"AGENT": "dog", "ACTION": "runs", "LOCATION": "floor"}),
    ("cat sees ball", {"AGENT": "cat", "ACTION": "sees", "OBJECT": "ball"}),
    ("dog opens door", {"AGENT": "dog", "ACTION": "opens", "OBJECT": "door"}),
    ("cat hits box", {"AGENT": "cat", "ACTION": "hits", "OBJECT": "box"}),
    ("dog sits on table", {"AGENT": "dog", "ACTION": "sits", "LOCATION": "table"}),
    ("key sits on mat", {"AGENT": "key", "ACTION": "sits", "LOCATION": "mat"}),
    ("ball runs on floor", {"AGENT": "ball", "ACTION": "runs", "LOCATION": "floor"}),
    ("cat runs on room", {"AGENT": "cat", "ACTION": "runs", "LOCATION": "room"}),
    ("dog sees key", {"AGENT": "dog", "ACTION": "sees", "OBJECT": "key"}),
]

# Level 2: ATTR + SVO(L)
LEVEL2_SENTENCES = [
    ("red cat sits on mat", {"ATTR": "red", "AGENT": "cat", "ACTION": "sits", "LOCATION": "mat"}),
    ("blue dog runs on floor", {"ATTR": "blue", "AGENT": "dog", "ACTION": "runs", "LOCATION": "floor"}),
    ("green ball sits on table", {"ATTR": "green", "AGENT": "ball", "ACTION": "sits", "LOCATION": "table"}),
    ("red key sits on room", {"ATTR": "red", "AGENT": "key", "ACTION": "sits", "LOCATION": "room"}),
    ("blue cat sees ball", {"ATTR": "blue", "AGENT": "cat", "ACTION": "sees", "OBJECT": "ball"}),
]

# Level 3: MiniGrid imperative
LEVEL3_SENTENCES = [
    ("pick up the red key", {"ACTION": "pick up", "ATTR": "red", "OBJECT": "key"}),
    ("go to the blue door", {"ACTION": "go to", "OBJECT": "door"}),
    ("open the green box", {"ACTION": "open", "ATTR": "green", "OBJECT": "box"}),
    ("toggle the red door", {"ACTION": "toggle", "ATTR": "red", "OBJECT": "door"}),
    ("pick up the blue ball", {"ACTION": "pick up", "ATTR": "blue", "OBJECT": "ball"}),
]


def build_vocab_embeddings(hac_dim: int = 2048) -> dict[str, torch.Tensor]:
    """Generate fixed HAC embedding for each vocabulary word."""
    all_words = set(NOUNS + VERBS + LOCATIONS + ADJECTIVES +
                    MINIGRID_OBJECTS + MINIGRID_ACTIONS)
    embeddings: dict[str, torch.Tensor] = {}
    for i, word in enumerate(sorted(all_words)):
        embeddings[word] = random_hac_vector(hac_dim, seed=1000 + i)
    return embeddings


def best_match(
    query: torch.Tensor, candidates: dict[str, torch.Tensor], hac: HACEngine,
) -> tuple[str, float]:
    """Find the candidate with highest cosine similarity to query."""
    best_word = ""
    best_sim = -1.0
    for word, emb in candidates.items():
        sim = hac.similarity(query, emb)
        if sim > best_sim:
            best_sim = sim
            best_word = word
    return best_word, best_sim


def evaluate_level(
    sentences: list[tuple[str, dict[str, str]]],
    level_name: str,
    chunker: RuleBasedChunker,
    parser: RoleFillerParser,
    hac: HACEngine,
    vocab_emb: dict[str, torch.Tensor],
) -> tuple[int, int]:
    """Evaluate role extraction on a set of sentences. Returns (correct, total)."""
    correct = 0
    total = 0

    for sentence, expected_roles in sentences:
        chunks = chunker.chunk(sentence)
        # Build embeddings for this sentence's chunks.
        emb = {}
        for c in chunks:
            if c.text in vocab_emb:
                emb[c.text] = vocab_emb[c.text]
            else:
                print(f"  WARNING: '{c.text}' not in vocab, skipping")
                continue

        if not emb:
            continue

        sentence_hac = parser.parse(chunks, emb)

        # For each expected role, extract and find best match.
        for role, expected_word in expected_roles.items():
            recovered = parser.extract(role, sentence_hac)
            match_word, sim = best_match(recovered, vocab_emb, hac)
            is_correct = match_word == expected_word
            total += 1
            if is_correct:
                correct += 1
            else:
                print(f"  MISS: '{sentence}' role={role}: "
                      f"expected='{expected_word}', got='{match_word}' (sim={sim:.3f})")

    acc = correct / total if total > 0 else 0
    print(f"  {level_name}: {correct}/{total} = {acc:.3f}")
    return correct, total


def run(device: str = "cpu") -> dict:
    """Run experiment 46: role extraction accuracy."""
    print("\n" + "=" * 60)
    print("Exp 46: Role extraction accuracy")
    print("=" * 60)

    hac_dim = 2048
    hac = HACEngine(dim=hac_dim)
    roles = get_roles(hac_dim=hac_dim)
    chunker = RuleBasedChunker()
    parser = RoleFillerParser(hac, roles)
    vocab_emb = build_vocab_embeddings(hac_dim)

    print(f"\nVocabulary: {len(vocab_emb)} words")
    print(f"HAC dim: {hac_dim}")
    print()

    total_correct = 0
    total_count = 0

    c1, t1 = evaluate_level(LEVEL1_SENTENCES, "Level 1 (SVO)", chunker, parser, hac, vocab_emb)
    total_correct += c1
    total_count += t1

    c2, t2 = evaluate_level(LEVEL2_SENTENCES, "Level 2 (ATTR+SVO)", chunker, parser, hac, vocab_emb)
    total_correct += c2
    total_count += t2

    c3, t3 = evaluate_level(LEVEL3_SENTENCES, "Level 3 (MiniGrid)", chunker, parser, hac, vocab_emb)
    total_correct += c3
    total_count += t3

    overall_acc = total_correct / total_count if total_count > 0 else 0

    print(f"\nOverall: {total_correct}/{total_count} = {overall_acc:.3f}")
    print(f"Gate (> 0.8): {'PASS' if overall_acc > 0.8 else 'FAIL'}")

    return {
        "level1_acc": c1 / t1 if t1 else 0,
        "level2_acc": c2 / t2 if t2 else 0,
        "level3_acc": c3 / t3 if t3 else 0,
        "overall_acc": overall_acc,
        "correct": total_correct,
        "total": total_count,
        "pass": overall_acc > 0.8,
    }


if __name__ == "__main__":
    result = run()
    sys.exit(0 if result["pass"] else 1)
