"""Experiment 14: Текстовая генерация (retrieval-based).

Загружает eval set из ingest_corpus.py.
Для каждого предложения: encode → query top-1 по SDR overlap → проверить совпадение.
Gate: recall@1 > 0.5.

NOTE: Требует запуска ПОСЛЕ ingest_corpus.py на minipc.
Eval set читается из --eval-path (по умолчанию /opt/agi/corpus/eval_set.jsonl).

Реализация: SDR dot-product retrieval (без DCAM).
TextEncoder детерминирован → self-similarity всегда максимальна среди несхожих предложений.
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import argparse
import json
import os


def run(eval_path: str = "/opt/agi/corpus/eval_set.jsonl",
        n_eval: int = 1000) -> float:
    """Run Exp 14: recall@1 on SDR retrieval.

    For each sentence: re-encode → find top-1 nearest by SDR overlap → check match.
    Gate: recall@1 > 0.5.
    """
    import torch
    from snks.daf.types import EncoderConfig
    from snks.encoder.text_encoder import TextEncoder

    if not os.path.exists(eval_path):
        print(f"Exp 14: eval set не найден: {eval_path}")
        print("  Сначала запустите: python scripts/ingest_corpus.py")
        return 0.0

    records = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    records = records[:n_eval]
    print(f"Exp 14: Текстовая генерация (retrieval@1)")
    print(f"  Eval set: {len(records)} записей")

    encoder = TextEncoder(EncoderConfig())

    embeddings = []
    sentences = []
    for rec in records:
        emb = torch.tensor(rec["embedding"], dtype=torch.float32)
        embeddings.append(emb)
        sentences.append(rec["sentence"])

    emb_matrix = torch.stack(embeddings)  # (N, sdr_size)

    correct = 0
    for i, sentence in enumerate(sentences):
        query_sdr = encoder.encode(sentence)  # (sdr_size,)
        scores = emb_matrix @ query_sdr  # (N,)
        top1 = int(scores.argmax().item())

        if top1 == i:
            correct += 1

    recall1 = correct / len(sentences)
    print(f"  recall@1: {recall1:.4f}")
    print(f"  Gate (recall@1 > 0.5): {'PASS' if recall1 > 0.5 else 'FAIL'}")
    return recall1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-path", default="/opt/agi/corpus/eval_set.jsonl")
    parser.add_argument("--n-eval", type=int, default=1000)
    args = parser.parse_args()
    run(args.eval_path, args.n_eval)
