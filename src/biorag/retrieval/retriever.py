"""Retrieval: dense, keyword, and hybrid.

Three retrievers, because the evaluation layer's whole purpose is to show that
RAG beats keyword search *on this corpus* — which requires actually implementing
the keyword baseline rather than asserting it loses.

**Hybrid fusion uses Reciprocal Rank Fusion, not score interpolation.**
Dense cosine similarities live in roughly [0.2, 0.9] and BM25 scores are unbounded
and corpus-dependent; a weighted sum of the two is arithmetic on incommensurable
quantities, and its behaviour changes as the corpus grows. RRF discards the
magnitudes and combines RANKS:

    score(d) = sum over retrievers of  1 / (k + rank_r(d))

with k=60 as in the original paper. It has no tuning parameters that depend on
score scale, and it is what most production hybrid search actually uses.

**Why hybrid matters specifically here.** Biomedical queries carry exact
identifiers — "MK-2206", "PI3Kα", "IC50", "SHP099" — where lexical match is
precisely right and embeddings are weak, because a compound code is a rare token
the model has barely seen. Meanwhile "how do allosteric inhibitors avoid ATP-site
resistance mutations?" is a semantic question no keyword search will answer.
Real questions are a mix, so the retrievers are complementary rather than
competing.
"""
from __future__ import annotations

import logging
import re
from typing import Protocol

from ..config import DEFAULT, RetrievalConfig
from ..ingestion.embedder import Embedder
from ..ingestion.store import NumpyStore, SearchHit, VectorStore
from ..schema import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)

RRF_K = 60  # smoothing constant from the Reciprocal Rank Fusion paper

# Keep alphanumerics, hyphens, and Greek letters: "MK-2206" and "PI3Kα" must
# survive tokenisation as single tokens. A plain \w+ split would shatter both.
_TOKEN_RE = re.compile(r"[A-Za-z0-9Ͱ-Ͽ][A-Za-z0-9Ͱ-Ͽ\-]*")


def tokenize(text: str) -> list[str]:
    """Lowercase tokens, preserving compound codes and Greek letters."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class Retriever(Protocol):
    name: str
    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]: ...


class DenseRetriever:
    """Semantic search over embedded chunks."""

    name = "dense"

    def __init__(self, store: VectorStore, embedder: Embedder | None = None) -> None:
        self.store = store
        self.embedder = embedder or Embedder()

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        k = k or DEFAULT.retrieval.top_k
        hits = self.store.search(self.embedder.embed_query(query), k)
        return [
            RetrievedChunk(chunk=h.chunk, score=h.score, rank=i + 1, retriever=self.name)
            for i, h in enumerate(hits)
        ]


class BM25Retriever:
    """Okapi BM25 keyword baseline.

    Exists to be BEATEN, measurably, in the evaluation layer — and to be honest
    about the cases where it wins. On exact compound identifiers it frequently
    does.

    **Expect negative scores on this corpus, and do not treat them as a bug.**
    BM25's IDF term goes negative for any word appearing in more than half the
    documents. Because the corpus is deliberately scoped to one topic, the words
    "allosteric", "kinase", and "inhibitor" are in nearly every abstract and
    therefore carry negative weight — a query built from those words scores every
    document negatively.

    This is not a defect in BM25; it is BM25 correctly reporting that those terms
    carry no discriminative information HERE. It is also a concrete argument for
    hybrid retrieval: on a narrow corpus, the terms that define the domain are
    exactly the terms keyword search cannot use, and only the dense retriever can
    tell two abstracts about allosteric kinase inhibition apart. Ranking within
    the negative scores is still meaningful, and RRF discards magnitudes anyway.
    """

    name = "bm25"

    def __init__(self, chunks: list[Chunk]) -> None:
        from rank_bm25 import BM25Okapi  # optional dependency, imported late

        if not chunks:
            raise ValueError("BM25Retriever needs a non-empty chunk list.")
        self.chunks = chunks
        self._bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        k = k or DEFAULT.retrieval.top_k
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [
            RetrievedChunk(chunk=self.chunks[i], score=float(scores[i]),
                           rank=rank + 1, retriever=self.name)
            for rank, i in enumerate(order)
        ]


class HybridRetriever:
    """Reciprocal Rank Fusion over several retrievers."""

    name = "hybrid"

    def __init__(
        self,
        retrievers: list[Retriever],
        weights: list[float] | None = None,
        rrf_k: int = RRF_K,
    ) -> None:
        if not retrievers:
            raise ValueError("HybridRetriever needs at least one retriever.")
        if weights is not None and len(weights) != len(retrievers):
            raise ValueError("weights must align with retrievers")
        self.retrievers = retrievers
        self.weights = weights or [1.0] * len(retrievers)
        self.rrf_k = rrf_k

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        k = k or DEFAULT.retrieval.top_k
        pool = DEFAULT.retrieval.candidate_k

        fused: dict[str, float] = {}
        best: dict[str, RetrievedChunk] = {}

        for retriever, weight in zip(self.retrievers, self.weights):
            for hit in retriever.retrieve(query, pool):
                cid = hit.chunk.chunk_id
                fused[cid] = fused.get(cid, 0.0) + weight / (self.rrf_k + hit.rank)
                # Keep one representative object per chunk for its text/metadata.
                if cid not in best:
                    best[cid] = hit

        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [
            RetrievedChunk(chunk=best[cid].chunk, score=score, rank=i + 1, retriever=self.name)
            for i, (cid, score) in enumerate(ranked)
        ]


def deduplicate_by_pmid(hits: list[RetrievedChunk], max_per_pmid: int = 1) -> list[RetrievedChunk]:
    """Collapse multiple chunks from the same paper.

    Matters for the generation layer: five chunks from one abstract look like
    five independent sources to an LLM, and it will happily present them as
    corroborating evidence. They are one paper. Capping per-PMID chunks keeps the
    apparent weight of evidence honest.
    """
    seen: dict[str, int] = {}
    out: list[RetrievedChunk] = []
    for hit in hits:
        count = seen.get(hit.chunk.pmid, 0)
        if count >= max_per_pmid:
            continue
        seen[hit.chunk.pmid] = count + 1
        out.append(hit)
    for i, hit in enumerate(out):
        hit.rank = i + 1
    return out


def build_retriever(
    store: NumpyStore,
    *,
    mode: str = "hybrid",
    embedder: Embedder | None = None,
    cfg: RetrievalConfig | None = None,
) -> Retriever:
    """Convenience factory. `mode` is one of dense | bm25 | hybrid."""
    cfg = cfg or DEFAULT.retrieval
    if mode == "dense":
        return DenseRetriever(store, embedder)
    if mode == "bm25":
        return BM25Retriever(store.chunks)
    if mode == "hybrid":
        alpha = cfg.hybrid_alpha
        return HybridRetriever(
            [DenseRetriever(store, embedder), BM25Retriever(store.chunks)],
            weights=[alpha, 1.0 - alpha],
        )
    raise ValueError(f"Unknown retrieval mode: {mode!r}")
