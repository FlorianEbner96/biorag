"""Query the built index with real data.

Prerequisite: build the index first
    python -m biorag.data.fetch_pubmed --email you@example.com
    python -m biorag.data.clean_corpus
    python -m biorag.ingestion.pipeline

Then:
    python examples/ask.py "Which compounds inhibit PI3K-alpha allosterically?"
    python examples/ask.py                              # interactive, several questions
    python examples/ask.py "..." --out results.json     # also save results to a file

Without ANTHROPIC_API_KEY only retrieval runs (free): it lists the matching
abstracts with their PMIDs. With the key set, a cited answer is generated too.

Saving is optional. Without --out nothing is written and the tool behaves exactly
as before. With --out, every answered question is appended to a JSON file.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from biorag.ingestion.embedder import Embedder
from biorag.ingestion.store import NumpyStore
from biorag.retrieval.retriever import build_retriever


def _append_record(out_path: Path, record: dict) -> None:
    """Append one result record to a JSON array file (created if missing).

    Loads the existing array, appends, and writes it back. Fine for the small
    number of queries this tool is used for; a high-volume logger would use JSONL.
    """
    records: list = []
    if out_path.exists():
        try:
            records = json.loads(out_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            records = []
    records.append(record)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Query the biorag index.")
    parser.add_argument("question", nargs="*",
                        help="Question to ask (omit for interactive mode).")
    parser.add_argument("--out", default=None,
                        help="Optional: append each question's results to this JSON "
                             "file. Without --out nothing is saved.")
    parser.add_argument("--k", type=int, default=5, help="Number of results to retrieve.")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else None

    # 1. Load the built index (from data/index/).
    try:
        store = NumpyStore.load()
    except FileNotFoundError:
        print("No index found. Build it first:")
        print("  python -m biorag.data.fetch_pubmed --email you@example.com")
        print("  python -m biorag.data.clean_corpus")
        print("  python -m biorag.ingestion.pipeline")
        return 1

    print(f"Index loaded: {len(store)} chunks.\n")

    # 2. Hybrid retriever (semantic + keyword) with the real embedding model.
    embedder = Embedder()
    retriever = build_retriever(store, mode="hybrid", embedder=embedder)

    # 3. Optional answer generation, if an API key is present.
    pipeline = None
    if os.environ.get("ANTHROPIC_API_KEY"):
        from biorag.generation.answerer import AnthropicClient, Answerer, RAGPipeline
        pipeline = RAGPipeline(retriever, Answerer(AnthropicClient()))
        print("ANTHROPIC_API_KEY found -> full answer generation enabled.")
    else:
        print("No ANTHROPIC_API_KEY -> retrieval only (free). "
              "Set the key for generated answers.")
    if out_path is not None:
        print(f"Saving results to {out_path}")
    print()

    def handle(question: str) -> None:
        if pipeline is not None:
            answer = pipeline.ask(question)
            print(answer)
            record = {
                "question": question,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "mode": "rag",
                "answer": answer.text,
                "cited_pmids": answer.cited_pmids,
                "retrieved_pmids": answer.retrieved_pmids,
                "is_grounded": answer.is_grounded,
            }
        else:
            hits = retriever.retrieve(question, k=args.k)
            for h in hits:
                print(f"  [{h.rank}] PMID {h.chunk.pmid}  ({h.score:.3f})")
                print(f"      {h.chunk.title}")
                print(f"      https://pubmed.ncbi.nlm.nih.gov/{h.chunk.pmid}/\n")
            record = {
                "question": question,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "mode": "retrieval",
                "hits": [
                    {"rank": h.rank, "pmid": h.chunk.pmid, "score": round(h.score, 4),
                     "title": h.chunk.title,
                     "url": f"https://pubmed.ncbi.nlm.nih.gov/{h.chunk.pmid}/"}
                    for h in hits
                ],
            }
        if out_path is not None:
            _append_record(out_path, record)

    # 4. One-shot question from the command line, or interactive loop.
    if args.question:
        handle(" ".join(args.question))
    else:
        print("Enter a question (empty line to quit):")
        while True:
            try:
                q = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break
            handle(q)

    if out_path is not None:
        print(f"\nResults saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
