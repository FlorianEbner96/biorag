"""HTTP layer: contract, degradation modes, and readiness.

Every test here runs fully offline. That is not a nicety — it is the same
property the other 102 tests have, and the reason it holds is that the app takes
its dependencies through `get_deps()` instead of constructing them. Overriding
that one function swaps the real index and the real embedding model for the
in-memory fakes the rest of the suite already uses.

What is deliberately NOT tested here: whether retrieval is any *good*. That is
the evaluation layer's job and it is measured, not asserted. These tests check
that the HTTP surface does what it says — status codes, response shape, and the
two service modes.
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="API extra not installed: pip install -e '.[api]'")

from conftest import FakeLLM
from fastapi.testclient import TestClient

from biorag.api.app import Deps, app, get_deps
from biorag.generation.answerer import Answerer, RAGPipeline
from biorag.retrieval.retriever import build_retriever


@pytest.fixture
def retriever(store, fake_embedder):
    return build_retriever(store, mode="hybrid", embedder=fake_embedder)


@pytest.fixture
def retrieval_client(retriever, store):
    """Retrieval-only mode: no LLM configured, as when ANTHROPIC_API_KEY is unset."""
    deps = Deps(retriever=retriever, pipeline=None, n_chunks=len(store))
    app.dependency_overrides[get_deps] = lambda: deps
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def rag_client(retriever, store):
    """Full RAG mode, with a scripted LLM standing in for Claude."""
    pipeline = RAGPipeline(retriever, Answerer(FakeLLM(mode="cite_first")))
    deps = Deps(retriever=retriever, pipeline=pipeline, n_chunks=len(store))
    app.dependency_overrides[get_deps] = lambda: deps
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def empty_client():
    """No index — the state a container is in before its volume is mounted."""
    app.dependency_overrides[get_deps] = lambda: Deps()
    yield TestClient(app)
    app.dependency_overrides.clear()


class TestHealth:
    def test_reports_ready_with_chunk_count(self, retrieval_client, store):
        body = retrieval_client.get("/health").json()
        assert body["status"] == "ok"
        assert body["chunks"] == len(store)

    def test_reports_no_index_rather_than_failing(self, empty_client):
        """Liveness and readiness are different questions.

        The process is up, so /health must answer — but it has to say it cannot
        serve, otherwise an orchestrator routes traffic to a container that will
        503 every request.
        """
        resp = empty_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "no_index"
        assert resp.json()["chunks"] == 0

    def test_generation_flag_false_without_a_key(self, retrieval_client):
        assert retrieval_client.get("/health").json()["generation_enabled"] is False

    def test_generation_flag_true_with_a_pipeline(self, rag_client):
        """Split from the case above on purpose.

        Both client fixtures override `get_deps` on the same module-level `app`,
        so requesting two of them in one test silently leaves only the last
        override in place. Keeping one client per test makes that impossible
        rather than merely unlikely.
        """
        assert rag_client.get("/health").json()["generation_enabled"] is True


class TestAskRetrievalMode:
    def test_returns_sources_without_an_answer(self, retrieval_client):
        body = retrieval_client.post("/ask", json={"question": "allosteric inhibitor"}).json()
        assert body["mode"] == "retrieval"
        assert body["answer"] is None
        assert body["sources"]

    def test_grounded_is_null_not_false(self, retrieval_client):
        """Nothing was generated, so nothing can be fabricated.

        Reporting `is_grounded: false` here would be wrong in a way that matters:
        it implies a verification ran and failed.
        """
        body = retrieval_client.post("/ask", json={"question": "allosteric inhibitor"}).json()
        assert body["is_grounded"] is None

    def test_every_source_carries_its_pmid_and_url(self, retrieval_client):
        body = retrieval_client.post("/ask", json={"question": "kinase"}).json()
        for s in body["sources"]:
            assert s["pmid"]
            assert s["url"].endswith(f"/{s['pmid']}/")


class TestAskRagMode:
    def test_returns_generated_answer_with_citations(self, rag_client):
        body = rag_client.post("/ask", json={"question": "allosteric inhibitor"}).json()
        assert body["mode"] == "rag"
        assert body["answer"]
        assert body["cited_pmids"]

    def test_grounded_answer_reports_true(self, rag_client):
        body = rag_client.post("/ask", json={"question": "allosteric inhibitor"}).json()
        assert body["is_grounded"] is True
        assert body["hallucinated_citations"] == []

    def test_hallucinated_citation_surfaces_in_the_response(self, retriever, store):
        """The failure mode this whole project exists to catch must reach the client.

        A fabricated PMID that is logged server-side but returned as a clean 200
        is worse than useless: the caller has no way to know the answer is
        unverifiable.
        """
        pipeline = RAGPipeline(retriever, Answerer(FakeLLM(mode="hallucinate")))
        deps = Deps(retriever=retriever, pipeline=pipeline, n_chunks=len(store))
        app.dependency_overrides[get_deps] = lambda: deps
        try:
            body = TestClient(app).post("/ask", json={"question": "anything"}).json()
            assert body["is_grounded"] is False
            assert "99999999" in body["hallucinated_citations"]
        finally:
            app.dependency_overrides.clear()

    def test_refusal_is_reported_as_insufficient_context(self, retriever, store):
        pipeline = RAGPipeline(retriever, Answerer(FakeLLM(mode="refuse")))
        deps = Deps(retriever=retriever, pipeline=pipeline, n_chunks=len(store))
        app.dependency_overrides[get_deps] = lambda: deps
        try:
            body = TestClient(app).post("/ask", json={"question": "anything"}).json()
            assert body["insufficient_context"] is True
        finally:
            app.dependency_overrides.clear()


class TestSearchEndpoint:
    def test_never_generates_even_when_a_pipeline_exists(self, rag_client):
        """/search is the guaranteed-free path.

        Making that a property of the endpoint rather than of the environment
        means a caller can avoid paid calls without inspecting server config.
        """
        body = rag_client.get("/search", params={"q": "allosteric inhibitor"}).json()
        assert body["mode"] == "retrieval"
        assert body["answer"] is None

    def test_respects_k(self, retrieval_client):
        body = retrieval_client.get("/search", params={"q": "kinase", "k": 2}).json()
        assert len(body["sources"]) <= 2


class TestValidationAndReadiness:
    def test_missing_index_returns_503_not_500(self, empty_client):
        """Not ready is not the same as broken."""
        assert empty_client.post("/ask", json={"question": "anything"}).status_code == 503

    def test_too_short_question_is_rejected(self, retrieval_client):
        assert retrieval_client.post("/ask", json={"question": "a"}).status_code == 422

    def test_k_out_of_range_is_rejected(self, retrieval_client):
        """Bounded because retrieval cost and prompt size both grow with k."""
        assert retrieval_client.post(
            "/ask", json={"question": "allosteric inhibitor", "k": 500}
        ).status_code == 422

    def test_openapi_schema_is_served(self, retrieval_client):
        """The schema is the contract; if it does not render, the API is undocumented."""
        schema = retrieval_client.get("/openapi.json").json()
        assert "/ask" in schema["paths"]
        assert "/health" in schema["paths"]
