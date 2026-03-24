"""Corpus ingestion: Simple English Wikipedia → DCAM.

Usage:
    python scripts/ingest_corpus.py --n-train 90000 --n-eval 10000 --output-dir /opt/agi/corpus/

Requires: datasets, sentence-transformers, tqdm
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using simple rules."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=90000)
    parser.add_argument("--n-eval", type=int, default=10000)
    parser.add_argument("--output-dir", type=str, default="/app/results/corpus/")
    parser.add_argument("--num-nodes", type=int, default=10000)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick smoke test: n-train=100, n-eval=20")
    args = parser.parse_args()

    if args.smoke_test:
        args.n_train = 100
        args.n_eval = 20

    os.makedirs(args.output_dir, exist_ok=True)

    from datasets import load_dataset
    from tqdm import tqdm

    import torch
    from snks.daf.types import DafConfig, EncoderConfig, PipelineConfig, PredictionConfig, SKSConfig
    from snks.pipeline.runner import Pipeline

    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        try:
            import torch_directml
            device = "privateuseone"
        except ImportError:
            device = "cpu"
    log.info(f"Device: {device}")

    # Pipeline config (small DAF for text-only ingestion)
    config = PipelineConfig(
        daf=DafConfig(
            num_nodes=args.num_nodes,
            avg_degree=20,
            oscillator_model="fhn",
            coupling_strength=0.05,
            dt=0.01,
            noise_sigma=0.005,
            fhn_I_base=0.0,
            device=device,
        ),
        encoder=EncoderConfig(sdr_current_strength=1.0),
        sks=SKSConfig(
            top_k=min(args.num_nodes // 2, 5000),
            dbscan_eps=0.3,
            dbscan_min_samples=5,
            min_cluster_size=5,
            coherence_mode="rate",
        ),
        prediction=PredictionConfig(),
        steps_per_cycle=200,
        device=device,
    )
    pipeline = Pipeline(config)

    # Check if DCAM is available
    has_dcam = hasattr(pipeline, "dcam") and pipeline.dcam is not None
    if not has_dcam:
        log.warning("Pipeline has no DCAM — будет только text encoding (Exp 13 потребует DCAM).")

    log.info("Загружаем Simple English Wikipedia...")
    dataset = load_dataset("wikipedia", "20220301.simple", split="train", trust_remote_code=False)

    # Collect all sentences
    log.info("Разбиваем на предложения...")
    all_sentences: list[str] = []
    for article in dataset:
        sentences = split_sentences(article["text"])
        all_sentences.extend(sentences)
        if len(all_sentences) >= args.n_train + args.n_eval + 10000:
            break

    log.info(f"Всего предложений: {len(all_sentences)}")

    # Split
    train_sentences = all_sentences[:args.n_train]
    eval_sentences = all_sentences[args.n_train: args.n_train + args.n_eval]

    log.info(f"Train: {len(train_sentences)}, Eval: {len(eval_sentences)}")

    # Train: ingest into pipeline
    log.info("Начинаем ingestion (train)...")
    for i, sentence in enumerate(tqdm(train_sentences, desc="Ingestion")):
        result = pipeline.perception_cycle(text=sentence)

        if has_dcam:
            # Store in DCAM with text metadata
            # (DCAM integration depends on pipeline having a world_model)
            pass

        if (i + 1) % 1000 == 0:
            log.info(f"  [{i+1}/{len(train_sentences)}] n_sks={result.n_sks}")

    # Save eval set
    eval_path = os.path.join(args.output_dir, "eval_set.jsonl")
    log.info(f"Сохраняем eval set → {eval_path}")

    text_encoder = pipeline.text_encoder
    with open(eval_path, "w", encoding="utf-8") as f:
        for sentence in tqdm(eval_sentences, desc="Eval encoding"):
            sdr = text_encoder.encode(sentence)
            embedding = sdr.cpu().tolist()
            record = {"sentence": sentence, "embedding": embedding}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    log.info(f"Eval set сохранён: {len(eval_sentences)} записей → {eval_path}")
    log.info("Ingestion завершён.")


if __name__ == "__main__":
    main()
