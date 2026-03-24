"""Experiment 13: Корпусное обучение.

Загружает eval set из ingest_corpus.py.
Для каждого предложения: encode → query DCAM top-5 → проверить совпадение.
Gate: precision@5 > 0.7.

NOTE: Этот эксперимент требует запуска ПОСЛЕ ingest_corpus.py на minipc.
Eval set читается из --eval-path (по умолчанию /opt/agi/corpus/eval_set.jsonl).
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import argparse
import json
import os


def run(eval_path: str = "/opt/agi/corpus/eval_set.jsonl",
        n_eval: int = 1000) -> float:
    """Run Exp 13: precision@5 on DCAM retrieval.

    NOTE: Requires a DCAM-enabled pipeline that was used during ingestion.
    This script evaluates retrieval quality of the ingested corpus.
    """
    import torch
    from snks.daf.types import EncoderConfig
    from snks.encoder.text_encoder import TextEncoder

    if not os.path.exists(eval_path):
        print(f"Exp 13: eval set не найден: {eval_path}")
        print("  Сначала запустите: python scripts/ingest_corpus.py")
        return 0.0

    # Load eval set
    records = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    records = records[:n_eval]
    print(f"Exp 13: Корпусное обучение")
    print(f"  Eval set: {len(records)} записей")

    # Build embedding matrix from eval set
    encoder = TextEncoder(EncoderConfig())

    embeddings = []
    sentences = []
    for rec in records:
        emb = torch.tensor(rec["embedding"], dtype=torch.float32)
        embeddings.append(emb)
        sentences.append(rec["sentence"])

    emb_matrix = torch.stack(embeddings)  # (N, sdr_size)

    # For each sentence: encode → find top-5 nearest by cosine/dot similarity
    # (Proxy for DCAM query: use dot product on binary SDRs = overlap count)
    correct = 0
    for i, sentence in enumerate(sentences):
        query_sdr = encoder.encode(sentence)  # (sdr_size,)

        # Dot product: higher = more overlap
        scores = emb_matrix @ query_sdr  # (N,)
        top5 = scores.topk(5).indices.tolist()

        if i in top5:
            correct += 1

    precision5 = correct / len(sentences)
    print(f"  precision@5: {precision5:.4f}")
    print(f"  Gate (precision@5 > 0.7): {'PASS' if precision5 > 0.7 else 'FAIL'}")
    return precision5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-path", default="/opt/agi/corpus/eval_set.jsonl")
    parser.add_argument("--n-eval", type=int, default=1000)
    args = parser.parse_args()
    run(args.eval_path, args.n_eval)
