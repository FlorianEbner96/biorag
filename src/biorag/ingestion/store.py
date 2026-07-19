"""Vector storage.

Two backends, and the default is the boring one:

  - `NumpyStore` (default) — exact brute-force cosine search over a float32
    matrix, persisted as .npz + JSON.
  - `ChromaStore` (optional) — persistent client, for when the API layer wants a
    server-ish component.

**Why exact search is the right default here.** This corpus is a few thousand
chunks. A 3,000 x 384 float32 matrix is 4.6 MB; one matrix-vector product against
it takes single-digit milliseconds. Approximate nearest-neighbour indexes (HNSW,
IVF) exist to trade recall for speed at 10^5-10^8 vectors — at 10^3 they add a
dependency, a build step, and a recall loss below 100%, in exchange for speed you
cannot perceive. Reaching for a vector database at this scale is the most common
over-engineering mistake in RAG projects.

The Chroma backend is here because persistence is genuinely useful once an API
serves the index, and because being able to say "I benchmarked both and the
simpler one was faster at my scale" is a stronger position in an interview than
having only ever used the fashionable one.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from ..config import DEFAULT, INDEX_DIR, RetrievalConfig
from ..schema import Chunk

logger = logging.getLogger(__name__)


@dataclass
class SearchHit:
    chunk: Chunk
    score: float


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...
    def search(self, query_vector: np.ndarray, k: int) -> list[SearchHit]: ...
    def persist(self, path: Path) -> None: ...
    def __len__(self) -> int: ...


def _chunk_to_dict(c: Chunk) -> dict:
    return {
        "chunk_id": c.chunk_id, "pmid": c.pmid, "text": c.text,
        "section": c.section, "ordinal": c.ordinal, "title": c.title,
        "pub_year": c.pub_year, "journal": c.journal,
    }


def _chunk_from_dict(d: dict) -> Chunk:
    return Chunk(
        chunk_id=d["chunk_id"], pmid=d["pmid"], text=d["text"],
        section=d.get("section"), ordinal=int(d.get("ordinal", 0)),
        title=d.get("title", ""), pub_year=d.get("pub_year", ""),
        journal=d.get("journal", ""),
    )


class NumpyStore:
    """Exact cosine search over an in-memory float32 matrix.

    Assumes vectors are L2-normalised (EmbeddingConfig.normalize=True), so the
    dot product IS cosine similarity and no per-query normalisation is needed.
    If you ever set normalize=False, this silently returns dot-product rankings
    instead of cosine ones — so it is asserted at add() time rather than trusted.
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks vs {len(vectors)} vectors")
        if len(chunks) == 0:
            return

        vectors = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError(
                "Vectors are not L2-normalised; NumpyStore assumes unit vectors so "
                "that dot product equals cosine similarity. Set "
                "EmbeddingConfig.normalize=True."
            )

        self._chunks.extend(chunks)
        self._matrix = vectors if self._matrix is None else np.vstack([self._matrix, vectors])

    def search(self, query_vector: np.ndarray, k: int) -> list[SearchHit]:
        if self._matrix is None or not self._chunks:
            return []
        q = np.asarray(query_vector, dtype=np.float32).ravel()
        q = q / (np.linalg.norm(q) or 1.0)
        scores = self._matrix @ q
        k = min(k, len(scores))
        # argpartition for the top-k, then sort only those k.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [SearchHit(chunk=self._chunks[i], score=float(scores[i])) for i in top]

    def persist(self, path: Path | None = None) -> None:
        path = Path(path or INDEX_DIR)
        path.mkdir(parents=True, exist_ok=True)
        if self._matrix is not None:
            np.savez_compressed(path / "vectors.npz", matrix=self._matrix)
        (path / "chunks.json").write_text(
            json.dumps([_chunk_to_dict(c) for c in self._chunks], ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Persisted %d chunks to %s", len(self._chunks), path)

    @classmethod
    def load(cls, path: Path | None = None) -> "NumpyStore":
        path = Path(path or INDEX_DIR)
        store = cls()
        chunks_file = path / "chunks.json"
        vectors_file = path / "vectors.npz"
        if not chunks_file.exists() or not vectors_file.exists():
            raise FileNotFoundError(f"No index at {path}. Run the ingestion pipeline first.")
        store._chunks = [_chunk_from_dict(d) for d in json.loads(chunks_file.read_text(encoding="utf-8"))]
        store._matrix = np.load(vectors_file)["matrix"].astype(np.float32)
        logger.info("Loaded %d chunks from %s", len(store._chunks), path)
        return store

    @property
    def chunks(self) -> list[Chunk]:
        return self._chunks

    def __len__(self) -> int:
        return len(self._chunks)


class ChromaStore:
    """Persistent Chroma backend. Optional — import only if chromadb is installed."""

    def __init__(self, cfg: RetrievalConfig | None = None, path: Path | None = None) -> None:
        import chromadb  # optional dependency, imported late on purpose

        self.cfg = cfg or DEFAULT.retrieval
        path = Path(path or INDEX_DIR / "chroma")
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        # Cosine space to match the normalised embeddings we produce.
        self._collection = self._client.get_or_create_collection(
            name=self.cfg.collection_name, metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if not chunks:
            return
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=[v.tolist() for v in np.asarray(vectors, dtype=np.float32)],
            documents=[c.text for c in chunks],
            metadatas=[c.to_metadata() for c in chunks],
        )

    def search(self, query_vector: np.ndarray, k: int) -> list[SearchHit]:
        res = self._collection.query(
            query_embeddings=[np.asarray(query_vector, dtype=np.float32).ravel().tolist()],
            n_results=k,
        )
        hits: list[SearchHit] = []
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                        res["metadatas"][0], res["distances"][0]):
            chunk = Chunk(
                chunk_id=cid, pmid=meta.get("pmid", ""), text=doc,
                section=meta.get("section") or None, ordinal=int(meta.get("ordinal", 0)),
                title=meta.get("title", ""), pub_year=meta.get("pub_year", ""),
                journal=meta.get("journal", ""),
            )
            # Chroma returns cosine DISTANCE; convert to similarity so both
            # backends return comparable scores.
            hits.append(SearchHit(chunk=chunk, score=1.0 - float(dist)))
        return hits

    def persist(self, path: Path | None = None) -> None:
        """No-op: PersistentClient writes on add()."""

    def __len__(self) -> int:
        return int(self._collection.count())
