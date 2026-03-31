"""Experiment 47: Compositional generalization (Stage 20).

Tests that HAC role-filler parsing works on NOVEL combinations of known words.
All individual words are seen during a "grounding" phase, but specific
word combinations in test sentences were never seen together.

This validates the fundamental algebraic property of circular convolution:
bind/unbind is compositional — it doesn't memorize specific combinations.

Split: train sentences (seen combinations) vs test sentences (novel combinations).
All words appear in both splits; only the combinations differ.

Gate: unbind accuracy on novel combinations > 0.7
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import torch

from snks.dcam.hac import HACEngine
from snks.language.chunker import RuleBasedChunker
from snks.language.parser import RoleFillerParser
from snks.language.roles import get_roles, random_hac_vector


# --- Vocabulary ---

NOUNS = ["cat", "dog", "key", "ball", "door", "box"]
VERBS = ["sits", "runs", "opens", "hits", "sees"]
LOCATIONS = ["mat", "room", "table", "floor"]

# Train: these specific combinations are "seen".
TRAIN_SENTENCES = [
    ("cat sits on mat", {"AGENT": "cat", "ACTION": "sits", "LOCATION": "mat"}),
    ("dog runs on floor", {"AGENT": "dog", "ACTION": "runs", "LOCATION": "floor"}),
    ("key sits on table", {"AGENT": "key", "ACTION": "sits", "LOCATION": "table"}),
    ("ball runs on mat", {"AGENT": "ball", "ACTION": "runs", "LOCATION": "mat"}),
    ("cat opens door", {"AGENT": "cat", "ACTION": "opens", "OBJECT": "door"}),
    ("dog hits ball", {"AGENT": "dog", "ACTION": "hits", "OBJECT": "ball"}),
    ("box sits on room", {"AGENT": "box", "ACTION": "sits", "LOCATION": "room"}),
    ("door sits on floor", {"AGENT": "door", "ACTION": "sits", "LOCATION": "floor"}),
    ("cat sees key", {"AGENT": "cat", "ACTION": "sees", "OBJECT": "key"}),
    ("dog opens box", {"AGENT": "dog", "ACTION": "opens", "OBJECT": "box"}),
]

# Test: NOVEL combinations of the same words.
TEST_SENTENCES = [
    ("cat runs on floor", {"AGENT": "cat", "ACTION": "runs", "LOCATION": "floor"}),
    ("dog sits on mat", {"AGENT": "dog", "ACTION": "sits", "LOCATION": "mat"}),
    ("key runs on room", {"AGENT": "key", "ACTION": "runs", "LOCATION": "room"}),
    ("ball sits on table", {"AGENT": "ball", "ACTION": "sits", "LOCATION": "table"}),
    ("cat hits ball", {"AGENT": "cat", "ACTION": "hits", "OBJECT": "ball"}),
    ("dog sees key", {"AGENT": "dog", "ACTION": "sees", "OBJECT": "key"}),
    ("box runs on floor", {"AGENT": "box", "ACTION": "runs", "LOCATION": "floor"}),
    ("door runs on mat", {"AGENT": "door", "ACTION": "runs", "LOCATION": "mat"}),
    ("cat opens box", {"AGENT": "cat", "ACTION": "opens", "OBJECT": "box"}),
    ("dog opens door", {"AGENT": "dog", "ACTION": "opens", "OBJECT": "door"}),
]


def build_vocab_embeddings(hac_dim: int = 2048) -> dict[str, torch.Tensor]:
    """Generate fixed HAC embedding for each vocabulary word."""
    all_words = set(NOUNS + VERBS + LOCATIONS)
    embeddings: dict[str, torch.Tensor] = {}
    for i, word in enumerate(sorted(all_words)):
        embeddings[word] = random_hac_vector(hac_dim, seed=1000 + i)
    return embeddings


def best_match(
    query: torch.Tensor, candidates: dict[str, torch.Tensor], hac: HACEngine,
) -> str:
    """Find the candidate with highest cosine similarity to query."""
    best_word = ""
    best_sim = -1.0
    for word, emb in candidates.items():
        sim = hac.similarity(query, emb)
        if sim > best_sim:
            best_sim = sim
            best_word = word
    return best_word


def evaluate(
    sentences: list[tuple[str, dict[str, str]]],
    label: str,
    chunker: RuleBasedChunker,
    parser: RoleFillerParser,
    hac: HACEngine,
    vocab_emb: dict[str, torch.Tensor],
) -> tuple[int, int]:
    """Evaluate role extraction. Returns (correct, total)."""
    correct = 0
    total = 0

    for sentence, expected_roles in sentences:
        chunks = chunker.chunk(sentence)
        emb = {c.text: vocab_emb[c.text] for c in chunks if c.text in vocab_emb}
        if not emb:
            continue

        sentence_hac = parser.parse(chunks, emb)

        for role, expected_word in expected_roles.items():
            recovered = parser.extract(role, sentence_hac)
            match_word = best_match(recovered, vocab_emb, hac)
            total += 1
            if match_word == expected_word:
                correct += 1
            else:
                print(f"  MISS: '{sentence}' role={role}: "
                      f"expected='{expected_word}', got='{match_word}'")

    acc = correct / total if total > 0 else 0
    print(f"  {label}: {correct}/{total} = {acc:.3f}")
    return correct, total


def run(device: str = "cpu") -> dict:
    """Run experiment 47: compositional generalization."""
    print("\n" + "=" * 60)
    print("Exp 47: Compositional generalization")
    print("=" * 60)

    hac_dim = 2048
    hac = HACEngine(dim=hac_dim)
    roles = get_roles(hac_dim=hac_dim)
    chunker = RuleBasedChunker()
    parser = RoleFillerParser(hac, roles)
    vocab_emb = build_vocab_embeddings(hac_dim)

    print(f"\nVocabulary: {len(vocab_emb)} words")
    print(f"Train sentences: {len(TRAIN_SENTENCES)}")
    print(f"Test sentences (novel combos): {len(TEST_SENTENCES)}")
    print()

    # Evaluate on train (sanity check).
    c_train, t_train = evaluate(
        TRAIN_SENTENCES, "Train (seen combos)",
        chunker, parser, hac, vocab_emb,
    )

    # Evaluate on test (novel combinations).
    c_test, t_test = evaluate(
        TEST_SENTENCES, "Test (novel combos)",
        chunker, parser, hac, vocab_emb,
    )

    train_acc = c_train / t_train if t_train else 0
    test_acc = c_test / t_test if t_test else 0

    print(f"\nTrain accuracy: {train_acc:.3f}")
    print(f"Test accuracy:  {test_acc:.3f}")
    print(f"Gate (test > 0.7): {'PASS' if test_acc > 0.7 else 'FAIL'}")

    return {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "train_correct": c_train,
        "train_total": t_train,
        "test_correct": c_test,
        "test_total": t_test,
        "pass": test_acc > 0.7,
    }


if __name__ == "__main__":
    result = run()
    sys.exit(0 if result["pass"] else 1)
