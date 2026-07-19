"""Vector store: exact search, persistence, and the normalisation guard."""
from __future__ import annotations

import numpy as np
import pytest

from biorag.ingestion.store import NumpyStore


def test_search_returns_k_hits(store):
    assert len(store.search(np.random.rand(64).astype(np.float32), k=5)) == 5


def test_search_is_ordered_by_descending_score(store, fake_embedder):
    hits = store.search(fake_embedder.embed_query("allosteric PI3Kα inhibitor"), k=10)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_identical_text_retrieves_itself_first(store, fake_embedder, chunks):
    """Sanity: querying with a chunk's own text should return that chunk top-1."""
    target = chunks[3]
    hits = store.search(fake_embedder.embed_query(target.text), k=1)
    assert hits[0].chunk.chunk_id == target.chunk_id


def test_k_larger_than_corpus_is_safe(store):
    hits = store.search(np.random.rand(64).astype(np.float32), k=10_000)
    assert len(hits) == len(store)


def test_empty_store_returns_nothing():
    assert NumpyStore().search(np.random.rand(64).astype(np.float32), k=5) == []


def test_unnormalized_vectors_are_rejected(chunks):
    """Guards a silent correctness bug: unnormalised vectors turn cosine
    ranking into dot-product ranking, which favours long chunks and never errors."""
    store = NumpyStore()
    with pytest.raises(ValueError, match="normalised"):
        store.add(chunks[:3], np.full((3, 64), 5.0, dtype=np.float32))


def test_length_mismatch_is_rejected(chunks):
    with pytest.raises(ValueError):
        NumpyStore().add(chunks[:3], np.eye(2, 64, dtype=np.float32))


def test_roundtrip_persistence(store, tmp_path, fake_embedder):
    store.persist(tmp_path)
    loaded = NumpyStore.load(tmp_path)
    assert len(loaded) == len(store)

    q = fake_embedder.embed_query("SHP2 allosteric inhibitor resistance")
    before = [h.chunk.chunk_id for h in store.search(q, k=5)]
    after = [h.chunk.chunk_id for h in loaded.search(q, k=5)]
    assert before == after


def test_load_missing_index_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        NumpyStore.load(tmp_path / "nonexistent")


def test_provenance_survives_persistence(store, tmp_path):
    store.persist(tmp_path)
    assert all(c.pmid for c in NumpyStore.load(tmp_path).chunks)
