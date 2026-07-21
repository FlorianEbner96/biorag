"""Evaluate retrieval quality on the REAL corpus against a hand-written gold set.

Prerequisite: the index must be built (data/index/) and a gold-question file
must exist (see gold/allosteric_kinase_gold.json).

    python examples/evaluate.py
    python examples/evaluate.py --gold gold/allosteric_kinase_gold.json --k 5
    python examples/evaluate.py --out results/eval_metrics.json    # also save numbers

Compares three retrievers (dense, BM25, hybrid) on the same questions. Prints a
table and, with --out, writes the metrics to JSON so they can go into the README.

Retrieval metrics only — no API key needed. Answer/refusal metrics require
generation (an ANTHROPIC_API_KEY) and are out of scope for this quick runner;
demo_evaluation.py covers those against synthetic data.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from biorag.evaluation.gold import gold_set_health, load_gold
from biorag.evaluation.metrics import compare_retrievers, evaluate_retrieval
from biorag.ingestion.embedder import Embedder
from biorag.ingestion.store import NumpyStore
from biorag.retrieval.retriever import BM25Retriever, DenseRetriever, HybridRetriever


def main() -> int:
    logging.basicConfig(level=logging.ERROR)
    parser = argparse.ArgumentParser(description="Evaluate retrieval on the real corpus.")
    parser.add_argument("--gold", default="gold/allosteric_kinase_gold.json",
                        help="Path to the gold-question JSON file.")
    parser.add_argument("--k", type=int, default=5, help="Cutoff for recall@k / precision@k.")
    parser.add_argument("--out", default=None,
                        help="Optional: write the metrics to this JSON file.")
    args = parser.parse_args()

    # 1. Load the built index.
    try:
        store = NumpyStore.load()
    except FileNotFoundError:
        print("No index found. Build it first (fetch -> clean -> ingestion.pipeline).")
        return 1

    # 2. Load and sanity-check the gold set.
    gold = load_gold(args.gold)
    health = gold_set_health(gold)
    print(f"Corpus: {len(store)} chunks")
    print(f"Gold set: {health['n']} questions "
          f"({health['unanswerable_fraction']:.0%} unanswerable)  {health['by_type']}")
    for warning in health["warnings"]:
        print(f"  ! {warning}")
    print()

    # 3. Build the three retrievers over the SAME index.
    embedder = Embedder()
    dense = DenseRetriever(store, embedder)
    bm25 = BM25Retriever(store.chunks)
    hybrid = HybridRetriever([dense, bm25])

    # 4. Head-to-head comparison table.
    table = compare_retrievers({"dense": dense, "bm25": bm25, "hybrid": hybrid}, gold, k=args.k)
    print(table)

    # 5. Optional: persist the numbers for the README.
    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"k": args.k, "n_questions": health["n"],
                   "unanswerable_fraction": health["unanswerable_fraction"],
                   "retrievers": {}}
        for name, r in {"dense": dense, "bm25": bm25, "hybrid": hybrid}.items():
            m = evaluate_retrieval(r, gold, k=args.k)
            payload["retrievers"][name] = {
                "recall_at_k": round(m.recall_at_k, 4),
                "precision_at_k": round(m.precision_at_k, 4),
                "mrr": round(m.mrr, 4),
                "hit_rate": round(m.hit_rate, 4),
                "n_scored": m.n_questions,
            }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nMetrics saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
