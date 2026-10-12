"""Wire format for the HTTP API.

These Pydantic models are deliberately NOT the internal dataclasses from
`schema.py`. Two reasons:

  - **The OpenAPI schema is the contract.** Whatever is declared here is what
    appears at /docs, and that page is the first thing anyone evaluating this
    service will open. Leaking `RetrievedChunk` — a chunk object wrapped in a
    score, a rank and a retriever name — would document the internals rather than
    the interface.
  - **Internal refactors should not break clients.** The retrieval layer is the
    part most likely to change (a reranker, a different fusion strategy). Pinning
    the response shape here means those changes stay internal.

`is_grounded` and `hallucinated_citations` are first-class response fields
rather than diagnostics hidden in a log. Verified provenance is the core promise
of this system; an API that returns the answer text but drops the verification
result gives back exactly the unverifiable output the project exists to avoid.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Natural-language question about allosteric kinase inhibitors.",
        examples=["Which compounds inhibit PI3K-alpha allosterically?"],
    )
    k: int = Field(
        5, ge=1, le=20,
        description="Number of abstracts to retrieve. Bounded because retrieval "
                    "cost and prompt size both grow with k.",
    )


class SourceOut(BaseModel):
    """One retrieved abstract, with provenance intact."""
    pmid: str
    title: str
    journal: str
    pub_year: str
    score: float
    rank: int
    url: str


class AskResponse(BaseModel):
    """Answer to a question.

    `mode` distinguishes the two service levels, because they are genuinely
    different products and a client must be able to tell them apart:
      - "rag"       — an LLM produced a cited answer (requires ANTHROPIC_API_KEY)
      - "retrieval" — no key configured, so only the matching abstracts are
                      returned. Free, and still useful.
    """
    question: str
    mode: str = Field(..., description='"rag" or "retrieval"')
    answer: str | None = Field(
        None, description="Generated answer with inline [PMID] citations. Null in retrieval mode."
    )
    cited_pmids: list[str] = Field(default_factory=list)
    retrieved_pmids: list[str] = Field(default_factory=list)
    is_grounded: bool | None = Field(
        None,
        description="True if every cited PMID was actually retrieved. Null in "
                    "retrieval mode, where nothing is generated and so nothing "
                    "can be fabricated.",
    )
    hallucinated_citations: list[str] = Field(
        default_factory=list,
        description="PMIDs the model cited that were never in the context. "
                    "Non-empty means the answer must not be trusted.",
    )
    insufficient_context: bool = Field(
        False, description="The system declined to answer rather than guess."
    )
    sources: list[SourceOut] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Readiness, not just liveness.

    A health check that returns 200 as soon as the process is up is worthless
    here: the failure that actually happens in a container is that the process
    starts fine but the index volume was never mounted. `chunks` surfaces that
    immediately.
    """
    status: str
    chunks: int = Field(..., description="Number of indexed chunks. 0 means no usable index.")
    generation_enabled: bool = Field(
        ..., description="Whether ANTHROPIC_API_KEY was present at startup."
    )
    embedding_model: str
