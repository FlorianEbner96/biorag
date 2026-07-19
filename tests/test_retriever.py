"""Retrieval: dense, BM25, hybrid fusion, and PMID deduplication."""
from __future__ import annotations

import pytest

from biorag.retrieval.retriever import (
    BM25Retriever, DenseRetriever, HybridRetriever,
    deduplicate_by_pmid, tokenize,
)


class TestTokenization:
    def test_compound_codes_stay_intact(self):
        """'MK-2206' must not shatter into 'mk' and '2206' — the hyphenated code
        is the identifier, and splitting it destroys exact-match retrieval."""
        assert "mk-2206" in tokenize("Treatment with MK-2206 reduced signalling")

    def test_greek_letters_tokenize(self):
        tokens = tokenize("inhibition of PI3Kα")
        assert any("α" in t for t in tokens)

    def test_lowercased(self):
        assert tokenize("PI3K AKT") == ["pi3k", "akt"]


class TestDenseRetriever:
    def test_returns_requested_k(self, store, fake_embedder):
        r = DenseRetriever(store, fake_embedder)
        assert len(r.retrieve("allosteric kinase inhibitor", k=5)) == 5

    def test_ranks_are_sequential(self, store, fake_embedder):
        hits = DenseRetriever(store, fake_embedder).retrieve("SHP2 inhibitor", k=5)
        assert [h.rank for h in hits] == [1, 2, 3, 4, 5]

    def test_provenance_present(self, store, fake_embedder):
        hits = DenseRetriever(store, fake_embedder).retrieve("PI3Kα", k=3)
        assert all(h.chunk.pmid and h.url.endswith(f"{h.chunk.pmid}/") for h in hits)

    def test_retriever_name_recorded(self, store, fake_embedder):
        hits = DenseRetriever(store, fake_embedder).retrieve("kinase", k=2)
        assert all(h.retriever == "dense" for h in hits)


class TestBM25Retriever:
    def test_exact_compound_code_retrieved(self, chunks):
        """Where BM25 genuinely beats dense retrieval: rare exact identifiers."""
        r = BM25Retriever(chunks)
        hits = r.retrieve("SHP099", k=5)
        assert any("SHP099" in h.chunk.text for h in hits)

    def test_returns_requested_k(self, chunks):
        assert len(BM25Retriever(chunks).retrieve("allosteric inhibitor", k=5)) == 5

    def test_empty_chunk_list_rejected(self):
        with pytest.raises(ValueError):
            BM25Retriever([])


class TestHybridRetriever:
    def test_fuses_both_retrievers(self, store, chunks, fake_embedder):
        hybrid = HybridRetriever([DenseRetriever(store, fake_embedder), BM25Retriever(chunks)])
        hits = hybrid.retrieve("allosteric PI3Kα inhibitor resistance", k=5)
        assert len(hits) == 5
        assert all(h.retriever == "hybrid" for h in hits)

    def test_rrf_scores_are_descending(self, store, chunks, fake_embedder):
        hybrid = HybridRetriever([DenseRetriever(store, fake_embedder), BM25Retriever(chunks)])
        scores = [h.score for h in hybrid.retrieve("kinase inhibitor", k=8)]
        assert scores == sorted(scores, reverse=True)

    def test_no_duplicate_chunks_in_fused_output(self, store, chunks, fake_embedder):
        """A chunk found by both retrievers must appear once, ranked higher —
        not twice."""
        hybrid = HybridRetriever([DenseRetriever(store, fake_embedder), BM25Retriever(chunks)])
        ids = [h.chunk.chunk_id for h in hybrid.retrieve("allosteric inhibitor", k=10)]
        assert len(ids) == len(set(ids))

    def test_requires_at_least_one_retriever(self):
        with pytest.raises(ValueError):
            HybridRetriever([])

    def test_weight_length_validated(self, store, fake_embedder):
        with pytest.raises(ValueError):
            HybridRetriever([DenseRetriever(store, fake_embedder)], weights=[0.5, 0.5])


class TestDeduplication:
    def test_collapses_chunks_from_same_paper(self, store, fake_embedder):
        """Five chunks from one abstract look like five sources to an LLM.
        They are one paper."""
        hits = DenseRetriever(store, fake_embedder).retrieve("allosteric", k=20)
        deduped = deduplicate_by_pmid(hits, max_per_pmid=1)
        pmids = [h.chunk.pmid for h in deduped]
        assert len(pmids) == len(set(pmids))

    def test_ranks_are_renumbered(self, store, fake_embedder):
        hits = DenseRetriever(store, fake_embedder).retrieve("kinase", k=10)
        deduped = deduplicate_by_pmid(hits)
        assert [h.rank for h in deduped] == list(range(1, len(deduped) + 1))
