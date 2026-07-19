"""Shared fixtures. Nothing here touches the network or downloads a model."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from biorag.data.synthetic import make_corpus
from biorag.data.clean_corpus import clean_corpus
from biorag.ingestion.chunker import chunk_corpus


class FakeEmbedder:
    """Deterministic hashing embedder — no model download, no network.

    Hashes tokens into a fixed-dimensional bag-of-words vector and L2-normalises.
    Crude, but it produces *meaningful* similarity for lexical overlap, so the
    retrieval plumbing can be tested end to end and even ranked sensibly. Real
    semantic quality is a separate question, measured by the evaluation layer
    against the real model.
    """

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _encode_one(self, text: str) -> np.ndarray:
        from biorag.retrieval.retriever import tokenize
        vec = np.zeros(self._dim, dtype=np.float32)
        for token in tokenize(text):
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            vec[h % self._dim] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def embed_passages(self, texts, *, show_progress: bool = False):
        from biorag.ingestion.embedder import EmbeddingResult
        vectors = np.vstack([self._encode_one(t) for t in texts]) if texts \
            else np.zeros((0, self._dim), dtype=np.float32)
        return EmbeddingResult(vectors=vectors, model_name="fake", dim=self._dim)

    def embed_chunks(self, chunks, *, show_progress: bool = False):
        return self.embed_passages([c.text for c in chunks], show_progress=show_progress)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode_one(text)


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def raw_corpus():
    return make_corpus(n=40, seed=0, with_pathologies=True)


@pytest.fixture
def clean_abstracts(raw_corpus):
    cleaned, _ = clean_corpus(raw_corpus)
    return cleaned


@pytest.fixture
def chunks(clean_abstracts):
    return chunk_corpus(clean_abstracts)


@pytest.fixture
def store(chunks, fake_embedder):
    from biorag.ingestion.store import NumpyStore
    s = NumpyStore()
    s.add(chunks, fake_embedder.embed_chunks(chunks).vectors)
    return s


class FakeLLM:
    """Scripted LLM client — no API key, no cost, deterministic.

    Behaviour is chosen per test via `responses` or `mode`, so generation logic
    (citation extraction, hallucination detection, refusal handling) is tested
    without ever depending on what a real model happens to say. Testing against a
    live model would make these tests slow, expensive, and flaky in exchange for
    no additional coverage of OUR code.
    """

    def __init__(self, responses=None, mode: str = "cite_first"):
        self.responses = list(responses) if responses else None
        self.mode = mode
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, *, max_tokens: int, temperature: float) -> str:
        self.calls.append((system, user))
        if self.responses:
            return self.responses.pop(0)

        import re
        pmids = re.findall(r"\[PMID (\d+)\]", user)
        if self.mode == "cite_first" and pmids:
            return f"The compound acts allosterically [{pmids[0]}]."
        if self.mode == "cite_all" and pmids:
            return " ".join(f"Finding {i} is reported [{p}]." for i, p in enumerate(pmids, 1))
        if self.mode == "hallucinate":
            return "This is well established [99999999]."
        if self.mode == "refuse":
            return "INSUFFICIENT_CONTEXT: the abstracts do not address this question."
        if self.mode == "uncited":
            return "Allosteric inhibitors bind outside the ATP pocket."
        return "No answer."


@pytest.fixture
def fake_llm():
    return FakeLLM()
