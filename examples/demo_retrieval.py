"""Offline end-to-end demo: synthetic corpus -> clean -> chunk -> index -> retrieve.

Runs with no network and no API key. Uses the real embedding model if
sentence-transformers is installed, otherwise falls back to a hashing embedder
so the pipeline is always demonstrable.

    python examples/demo_retrieval.py
    python examples/demo_retrieval.py --real-model     # force the real model
"""
from __future__ import annotations

import argparse
import logging
import sys

from biorag.data.clean_corpus import clean_corpus
from biorag.data.synthetic import make_corpus
from biorag.ingestion.chunker import chunk_corpus
from biorag.ingestion.store import NumpyStore
from biorag.retrieval.retriever import (
    BM25Retriever, DenseRetriever, HybridRetriever, deduplicate_by_pmid,
)

QUESTIONS = [
    "How do allosteric inhibitors overcome gatekeeper resistance mutations?",
    "What is the IC50 of SHP099?",
    "Which compounds target PI3K alpha?",
]


def _get_embedder(force_real: bool):
    try:
        from biorag.ingestion.embedder import Embedder
        embedder = Embedder()
        _ = embedder.model  # trigger the load now, not mid-demo
        print("Using real embedding model:", embedder.cfg.model_name)
        return embedder
    except Exception as exc:
        if force_real:
            raise
        print(f"sentence-transformers unavailable ({type(exc).__name__}); "
              "falling back to hashing embedder.\n")
        sys.path.insert(0, "tests")
        from conftest import FakeEmbedder
        return FakeEmbedder()


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-model", action="store_true")
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()

    print("=" * 70)
    print("BIORAG — offline retrieval demo")
    print("=" * 70)

    raw = make_corpus(n=args.n, seed=7)
    print(f"\n1. Generated {len(raw)} synthetic PubMed records")

    cleaned, report = clean_corpus(raw)
    print("\n2. Cleaning\n")
    print("   " + report.summary().replace("\n", "\n   "))

    chunks = chunk_corpus(cleaned)
    print(f"\n3. Chunking: {len(cleaned)} abstracts -> {len(chunks)} chunks "
          f"({len(chunks)/len(cleaned):.2f} per abstract — near 1.0 is correct)")

    embedder = _get_embedder(args.real_model)
    store = NumpyStore()
    store.add(chunks, embedder.embed_chunks(chunks, show_progress=False).vectors)
    print(f"\n4. Indexed {len(store)} chunks")

    dense = DenseRetriever(store, embedder)
    bm25 = BM25Retriever(chunks)
    hybrid = HybridRetriever([dense, bm25])

    print("\n5. Retrieval comparison\n")
    for question in QUESTIONS:
        print("-" * 70)
        print(f"Q: {question}\n")
        for name, retriever in (("dense", dense), ("bm25", bm25), ("hybrid", hybrid)):
            hits = deduplicate_by_pmid(retriever.retrieve(question, k=6), max_per_pmid=1)[:2]
            print(f"  [{name}]")
            for hit in hits:
                print(f"    {hit.rank}. ({hit.score:.3f}) PMID {hit.chunk.pmid} — "
                      f"{hit.chunk.title[:58]}")
            print()

    print("=" * 70)
    print("Every hit carries a PMID. Generation layer (next) may cite only these.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
