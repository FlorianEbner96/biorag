"""Offline end-to-end demo: retrieval comparison + grounded answering + metrics.

Runs with no network and no API key — uses a hashing embedder and a scripted LLM
when the real ones are unavailable, so the full shape of the system is always
demonstrable.

    python examples/demo_evaluation.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from biorag.data.clean_corpus import clean_corpus
from biorag.data.synthetic import make_corpus
from biorag.evaluation.gold import gold_set_health, synthetic_gold_set
from biorag.evaluation.metrics import compare_retrievers, evaluate_answers, evaluate_retrieval
from biorag.generation.answerer import Answerer, RAGPipeline
from biorag.ingestion.chunker import chunk_corpus
from biorag.ingestion.store import NumpyStore
from biorag.retrieval.retriever import BM25Retriever, DenseRetriever, HybridRetriever


def main() -> int:
    logging.basicConfig(level=logging.ERROR)
    from conftest import FakeEmbedder, FakeLLM

    print("=" * 70)
    print("BIORAG — evaluation demo (offline: hashing embedder, scripted LLM)")
    print("=" * 70)

    cleaned, _ = clean_corpus(make_corpus(n=60, seed=0))
    chunks = chunk_corpus(cleaned)
    embedder = FakeEmbedder()
    store = NumpyStore()
    store.add(chunks, embedder.embed_chunks(chunks).vectors)
    print(f"\nCorpus: {len(cleaned)} abstracts -> {len(chunks)} chunks indexed")

    gold = synthetic_gold_set()
    health = gold_set_health(gold)
    print(f"\nGold set: {health['n']} questions, "
          f"{health['unanswerable_fraction']:.0%} unanswerable")
    for warning in health["warnings"]:
        print(f"  ! {warning}")

    dense = DenseRetriever(store, embedder)
    bm25 = BM25Retriever(chunks)
    hybrid = HybridRetriever([dense, bm25])

    print("\n" + compare_retrievers({"dense": dense, "bm25": bm25, "hybrid": hybrid}, gold, k=5))
    print("\n  NOTE: with the hashing embedder these numbers are meaningless as")
    print("  quality signals — they demonstrate the harness. Real numbers need")
    print("  the real model and a hand-written gold set on the real corpus.")

    print("\n" + "=" * 70)
    print("Grounded answering")
    print("=" * 70)
    pipeline = RAGPipeline(hybrid, Answerer(FakeLLM(mode="cite_all")), top_k=3)
    answers = [pipeline.ask(q.question) for q in gold]

    for question, answer in list(zip(gold, answers))[:2]:
        print(f"\nQ: {question.question}")
        print(f"A: {answer.text[:150]}")
        print(f"   grounded={answer.is_grounded}  citation_rate={answer.citation_rate:.2f}")

    answer_metrics = evaluate_answers(answers, gold)
    print("\nAnswer metrics\n" + answer_metrics.summary())
    print("\nRetrieval metrics (hybrid)\n" + evaluate_retrieval(hybrid, gold, k=5).summary())

    if answer_metrics.refusal_recall < 1.0:
        print("\n" + "!" * 70)
        print("refusal recall is below 1.0 — the model answered questions it should")
        print("have refused. This scripted LLM answers everything by design, so the")
        print("metric is doing its job: it is the ONLY number here that detects a")
        print("system which sounds confident on questions its corpus cannot answer.")
        print("A gold set without unanswerable questions cannot compute it at all.")
        print("!" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
