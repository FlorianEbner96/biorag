"""End-to-end ingestion: clean JSON -> chunks -> embeddings -> persisted index.

    python -m biorag.ingestion.pipeline --in data/clean/allosteric_kinase_inhibitors.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from ..config import DEFAULT, INDEX_DIR
from ..schema import CleanAbstract
from .chunker import chunk_corpus
from .embedder import Embedder
from .store import NumpyStore

logger = logging.getLogger(__name__)


def load_clean(path: Path) -> list[CleanAbstract]:
    with Path(path).open(encoding="utf-8") as f:
        return [CleanAbstract.from_dict(d) for d in json.load(f)]


def build_index(
    abstracts: list[CleanAbstract],
    *,
    index_dir: Path | None = None,
    embedder: Embedder | None = None,
) -> NumpyStore:
    chunks = chunk_corpus(abstracts, DEFAULT.chunking)
    if not chunks:
        raise ValueError("No chunks produced — is the clean corpus empty?")

    embedder = embedder or Embedder()
    result = embedder.embed_chunks(chunks)

    store = NumpyStore()
    store.add(chunks, result.vectors)
    store.persist(index_dir or INDEX_DIR)

    # Record what produced this index. Embeddings from different models are not
    # comparable, and an index silently queried with the wrong model returns
    # plausible nonsense rather than an error.
    manifest = {
        "model_name": result.model_name,
        "dim": result.dim,
        "n_abstracts": len(abstracts),
        "n_chunks": len(chunks),
        "normalize": DEFAULT.embedding.normalize,
        "max_chunk_chars": DEFAULT.chunking.max_chunk_chars,
    }
    Path(index_dir or INDEX_DIR).mkdir(parents=True, exist_ok=True)
    (Path(index_dir or INDEX_DIR) / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info("Index built: %d abstracts -> %d chunks, dim %d", len(abstracts), len(chunks), result.dim)
    return store


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description="Build the vector index from a clean corpus.")
    ap.add_argument("--in", dest="in_path", default="data/clean/allosteric_kinase_inhibitors.json")
    ap.add_argument("--index-dir", default=str(INDEX_DIR))
    args = ap.parse_args()

    abstracts = load_clean(Path(args.in_path))
    store = build_index(abstracts, index_dir=Path(args.index_dir))
    print(f"\nIndexed {len(store)} chunks from {len(abstracts)} abstracts -> {args.index_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
