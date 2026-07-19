"""Generation: citation extraction, hallucination detection, refusal."""
from __future__ import annotations

import pytest

from biorag.generation.answerer import Answerer, RAGPipeline, extract_citations
from biorag.generation.prompts import format_context
from biorag.retrieval.retriever import DenseRetriever
from conftest import FakeLLM


@pytest.fixture
def hits(store, fake_embedder):
    return DenseRetriever(store, fake_embedder).retrieve("allosteric inhibitor", k=4)


class TestCitationExtraction:
    def test_plain_bracket_pmid(self):
        assert extract_citations("Claim [30000001].") == ["30000001"]

    def test_pmid_prefix_form(self):
        assert extract_citations("Claim [PMID 30000001].") == ["30000001"]

    def test_multiple_in_order_deduplicated(self):
        text = "A [30000001]. B [30000002]. C [30000001]."
        assert extract_citations(text) == ["30000001", "30000002"]

    def test_no_citations(self):
        assert extract_citations("An entirely uncited claim.") == []

    def test_ignores_short_numbers(self):
        """[5] is a reference marker, not a PMID. PMIDs are 6-9 digits."""
        assert extract_citations("See [5] and [12].") == []


class TestGroundedness:
    def test_valid_citation_is_grounded(self, hits):
        answer = Answerer(FakeLLM(mode="cite_first")).answer("Q?", hits)
        assert answer.is_grounded and not answer.hallucinated_citations

    def test_fabricated_pmid_is_caught(self, hits):
        """*** The most damaging failure mode. ***

        A model citing a plausible PMID that was never retrieved produces output
        that looks perfectly credible and is unverifiable.
        """
        answer = Answerer(FakeLLM(mode="hallucinate")).answer("Q?", hits)
        assert not answer.is_grounded
        assert "99999999" in answer.hallucinated_citations

    def test_retrieved_pmids_recorded(self, hits):
        answer = Answerer(FakeLLM()).answer("Q?", hits)
        assert set(answer.retrieved_pmids) == {h.chunk.pmid for h in hits}

    def test_warning_appears_in_output(self, hits):
        answer = Answerer(FakeLLM(mode="hallucinate")).answer("Q?", hits)
        assert "WARNING" in str(answer)


class TestRefusal:
    def test_refusal_detected(self, hits):
        assert Answerer(FakeLLM(mode="refuse")).answer("Q?", hits).insufficient_context

    def test_empty_retrieval_refuses_without_calling_llm(self):
        """Deterministic refusal is free; hoping the model refuses on empty
        context is a coin flip."""
        llm = FakeLLM(mode="cite_first")
        answer = Answerer(llm).answer("Q?", [])
        assert answer.insufficient_context
        assert llm.calls == []

    def test_normal_answer_not_flagged_as_refusal(self, hits):
        assert not Answerer(FakeLLM(mode="cite_first")).answer("Q?", hits).insufficient_context


class TestCitationRate:
    def test_fully_cited_answer(self, hits):
        assert Answerer(FakeLLM(mode="cite_all")).answer("Q?", hits).citation_rate == 1.0

    def test_uncited_answer(self, hits):
        assert Answerer(FakeLLM(mode="uncited")).answer("Q?", hits).citation_rate == 0.0


class TestPromptConstruction:
    def test_pmid_appears_before_abstract_text(self, hits):
        """Models copy identifiers more reliably when the identifier is adjacent
        to the text it labels."""
        context = format_context(hits)
        first = context.split("\n")[0]
        assert first.startswith("[PMID ")

    def test_every_chunk_present_in_context(self, hits):
        context = format_context(hits)
        assert all(h.chunk.pmid in context for h in hits)

    def test_system_prompt_forbids_parametric_knowledge(self):
        from biorag.generation.prompts import SYSTEM_PROMPT
        assert "do not use it" in SYSTEM_PROMPT.lower()


class TestRAGPipeline:
    def test_end_to_end(self, store, fake_embedder):
        pipeline = RAGPipeline(DenseRetriever(store, fake_embedder),
                               Answerer(FakeLLM(mode="cite_first")), top_k=3)
        answer = pipeline.ask("How do allosteric inhibitors work?")
        assert answer.text and answer.is_grounded
        assert len(answer.retrieved_pmids) <= 3

    def test_deduplicates_sources_by_pmid(self, store, fake_embedder):
        pipeline = RAGPipeline(DenseRetriever(store, fake_embedder),
                               Answerer(FakeLLM(mode="cite_all")), top_k=5)
        answer = pipeline.ask("allosteric kinase inhibition")
        assert len(answer.retrieved_pmids) == len(set(answer.retrieved_pmids))

    def test_sources_render_with_urls(self, store, fake_embedder):
        pipeline = RAGPipeline(DenseRetriever(store, fake_embedder),
                               Answerer(FakeLLM(mode="cite_first")), top_k=3)
        assert "pubmed.ncbi.nlm.nih.gov" in pipeline.ask("kinase").format_sources()
