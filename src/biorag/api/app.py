"""FastAPI service exposing retrieval and grounded QA over HTTP.

Run it:
    pip install -e ".[api,generation]"
    uvicorn biorag.api.app:app --reload
    # -> http://127.0.0.1:8000/docs

Three design decisions shape this module:

**1. The index and embedding model load once, at startup — via `lifespan`.**
Loading them at import time would make the module unimportable without a built
index, which breaks the test suite and even `--help`. Loading them per request
would add a model load (~80 MB) and a disk read to every call. FastAPI's
`lifespan` context manager does it exactly once and is the documented
replacement for the deprecated `@app.on_event("startup")`.

**2. Dependencies are injected, never constructed inside a handler.**
The rest of this codebase already works this way — `Answerer` takes an injected
`LLMClient`, `RAGPipeline` an injected retriever — and that is precisely why the
existing 102 tests need no API key, no network and no model download. The API
must not be the layer that breaks that property, so handlers depend on
`get_deps()`, which tests override with a fake-backed instance.

**3. Missing index returns 503, not 500.**
"I have no index" is not a server error, it is *not ready to serve*. Orchestrators
and load balancers treat those differently: 503 keeps the container out of
rotation until the volume is there, 500 looks like a crash loop.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import Depends, FastAPI, HTTPException, Request

from ..config import DEFAULT
from ..generation.answerer import Answer
from ..schema import RetrievedChunk
from .schemas import AskRequest, AskResponse, HealthResponse, SourceOut

logger = logging.getLogger(__name__)


@dataclass
class Deps:
    """Everything a request handler needs, assembled once at startup.

    `pipeline` is None when no ANTHROPIC_API_KEY is configured. That is a
    supported mode, not a degraded error state: retrieval alone is free, needs no
    key, and is genuinely useful — it returns the abstracts that answer the
    question, just without the prose on top.
    """
    retriever: object | None = None
    pipeline: object | None = None
    n_chunks: int = 0

    @property
    def ready(self) -> bool:
        return self.retriever is not None and self.n_chunks > 0

    @property
    def generation_enabled(self) -> bool:
        return self.pipeline is not None


def build_deps() -> Deps:
    """Load the index, build the retriever, and wire up generation if a key exists.

    Failure to find an index is logged and returns an empty `Deps` rather than
    raising. A container that exits on startup because a volume is missing is
    harder to debug than one that starts and reports `status: "no_index"` on
    /health.
    """
    from ..config import INDEX_DIR
    from ..ingestion.embedder import Embedder
    from ..ingestion.store import NumpyStore
    from ..retrieval.retriever import build_retriever

    try:
        store = NumpyStore.load()
    except FileNotFoundError:
        logger.warning(
            "No index found at %s. The service will start but /ask returns 503 "
            "until the index is built (python -m biorag.ingestion.pipeline).",
            INDEX_DIR,
        )
        return Deps()

    retriever = build_retriever(store, mode="hybrid", embedder=Embedder())

    pipeline = None
    if os.environ.get("ANTHROPIC_API_KEY"):
        from ..generation.answerer import Answerer, AnthropicClient, RAGPipeline
        pipeline = RAGPipeline(retriever, Answerer(AnthropicClient()))
        logger.info("ANTHROPIC_API_KEY present -> answer generation enabled.")
    else:
        logger.info("No ANTHROPIC_API_KEY -> retrieval-only mode.")

    return Deps(retriever=retriever, pipeline=pipeline, n_chunks=len(store))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.deps = build_deps()
    yield
    app.state.deps = None


app = FastAPI(
    title="biorag",
    version="0.1.0",
    summary="Retrieval-augmented QA over PubMed literature on allosteric kinase inhibitors.",
    description=(
        "Every generated claim is traced to a PMID, and every citation is "
        "verified against what was actually retrieved — `is_grounded` is false "
        "whenever the model cited an abstract that was never in its context."
    ),
    lifespan=lifespan,
)


def get_deps(request: Request) -> Deps:
    """Dependency handle. Tests override this to inject fake-backed deps."""
    deps = getattr(request.app.state, "deps", None)
    return deps if deps is not None else Deps()


def _require_ready(deps: Deps) -> None:
    if not deps.ready:
        raise HTTPException(
            status_code=503,
            detail="No index loaded. Build one with: python -m biorag.ingestion.pipeline",
        )


def _to_sources(hits: list[RetrievedChunk]) -> list[SourceOut]:
    return [
        SourceOut(
            pmid=h.chunk.pmid,
            title=h.chunk.title,
            journal=h.chunk.journal or "",
            pub_year=h.chunk.pub_year or "",
            score=round(float(h.score), 4),
            rank=h.rank,
            url=h.url,
        )
        for h in hits
    ]


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health(deps: Deps = Depends(get_deps)) -> HealthResponse:
    return HealthResponse(
        status="ok" if deps.ready else "no_index",
        chunks=deps.n_chunks,
        generation_enabled=deps.generation_enabled,
        embedding_model=DEFAULT.embedding.model_name,
    )


@app.post("/ask", response_model=AskResponse, tags=["qa"])
def ask(req: AskRequest, deps: Deps = Depends(get_deps)) -> AskResponse:
    """Answer a question from the indexed corpus.

    With a key configured this generates a cited answer; without one it returns
    the retrieved abstracts. Both paths return the same response model, so a
    client can handle them uniformly by reading `mode`.
    """
    _require_ready(deps)

    if deps.pipeline is not None:
        answer: Answer = deps.pipeline.ask(req.question)
        return AskResponse(
            question=req.question,
            mode="rag",
            answer=answer.text,
            cited_pmids=answer.cited_pmids,
            retrieved_pmids=answer.retrieved_pmids,
            is_grounded=answer.is_grounded,
            hallucinated_citations=answer.hallucinated_citations,
            insufficient_context=answer.insufficient_context,
            sources=_to_sources(answer.sources),
        )

    hits = deps.retriever.retrieve(req.question, req.k)
    return AskResponse(
        question=req.question,
        mode="retrieval",
        answer=None,
        retrieved_pmids=list(dict.fromkeys(h.chunk.pmid for h in hits)),
        is_grounded=None,
        sources=_to_sources(hits),
    )


@app.get("/search", response_model=AskResponse, tags=["qa"])
def search(q: str, k: int = 5, deps: Deps = Depends(get_deps)) -> AskResponse:
    """Retrieval only, never generation — even when a key is configured.

    Separate from /ask on purpose: /ask is allowed to cost money, /search never
    is. Making that a property of the endpoint rather than of the environment
    means a caller can guarantee a free path without inspecting the server's
    configuration.
    """
    _require_ready(deps)
    validated = AskRequest(question=q, k=k)
    hits = deps.retriever.retrieve(validated.question, validated.k)
    return AskResponse(
        question=validated.question,
        mode="retrieval",
        answer=None,
        retrieved_pmids=list(dict.fromkeys(h.chunk.pmid for h in hits)),
        is_grounded=None,
        sources=_to_sources(hits),
    )
