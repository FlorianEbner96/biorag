"""Grounded answer generation.

The only component that costs money, and the only one that can fabricate. Both
facts shape the design:

  - The LLM client is injected, not constructed internally, so the whole layer is
    testable with a fake and the test suite never needs an API key.
  - Citations are VERIFIED after generation, not trusted. The prompt asks for
    [PMID] markers; this module checks that every PMID cited was actually in the
    retrieved context. A model citing a plausible-looking PMID that was never
    retrieved is the single most damaging failure mode this system has, because
    the output looks perfectly credible.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Protocol

from ..config import DEFAULT, GenerationConfig
from ..schema import RetrievedChunk
from .prompts import ANSWER_TEMPLATE, SYSTEM_PROMPT, format_context

logger = logging.getLogger(__name__)

PMID_CITATION_RE = re.compile(r"\[(?:PMID[:\s]*)?(\d{6,9})\]")
INSUFFICIENT_MARKER = "INSUFFICIENT_CONTEXT"


class LLMClient(Protocol):
    """Minimal surface an LLM backend must expose."""
    def complete(self, system: str, user: str, *, max_tokens: int, temperature: float) -> str: ...


class AnthropicClient:
    """Claude backend. Imported lazily so the package works without the SDK."""

    def __init__(self, cfg: GenerationConfig | None = None, api_key: str | None = None) -> None:
        import anthropic  # optional dependency

        self.cfg = cfg or DEFAULT.generation
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "No API key. Set ANTHROPIC_API_KEY or pass api_key=. "
                "The retrieval pipeline runs fine without this; only generation needs it."
            )
        self._client = anthropic.Anthropic(api_key=key)

    def complete(self, system: str, user: str, *, max_tokens: int, temperature: float) -> str:
        response = self._client.messages.create(
            model=self.cfg.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text


@dataclass
class Answer:
    """A generated answer with its provenance and verification status."""
    question: str
    text: str
    cited_pmids: list[str]
    retrieved_pmids: list[str]
    sources: list[RetrievedChunk] = field(default_factory=list)
    insufficient_context: bool = False
    hallucinated_citations: list[str] = field(default_factory=list)

    @property
    def is_grounded(self) -> bool:
        """True if every cited PMID was actually retrieved.

        Necessary but not sufficient: a model can cite a real retrieved PMID for
        a claim that abstract does not make. Claim-level checking lives in
        evaluation/groundedness.py.
        """
        return not self.hallucinated_citations

    @property
    def citation_rate(self) -> float:
        """Fraction of sentences carrying at least one citation."""
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", self.text.strip()) if s]
        if not sentences:
            return 0.0
        return sum(bool(PMID_CITATION_RE.search(s)) for s in sentences) / len(sentences)

    def format_sources(self) -> str:
        """Reference list, deduplicated by PMID, in citation order."""
        seen, lines = set(), []
        for pmid in self.cited_pmids:
            if pmid in seen:
                continue
            seen.add(pmid)
            match = next((s for s in self.sources if s.chunk.pmid == pmid), None)
            if match:
                c = match.chunk
                lines.append(f"  [{pmid}] {c.title} ({c.journal}, {c.pub_year})\n"
                             f"          https://pubmed.ncbi.nlm.nih.gov/{pmid}/")
        return "\n".join(lines)

    def __str__(self) -> str:
        out = [self.text]
        if self.cited_pmids:
            out += ["", "Sources:", self.format_sources()]
        if self.hallucinated_citations:
            out += ["", f"WARNING — uncited-in-context PMIDs: {self.hallucinated_citations}"]
        return "\n".join(out)


def extract_citations(text: str) -> list[str]:
    """PMIDs cited in the answer, in order of first appearance."""
    seen, out = set(), []
    for pmid in PMID_CITATION_RE.findall(text):
        if pmid not in seen:
            seen.add(pmid)
            out.append(pmid)
    return out


class Answerer:
    """Turns a question plus retrieved chunks into a cited answer."""

    def __init__(self, client: LLMClient, cfg: GenerationConfig | None = None) -> None:
        self.client = client
        self.cfg = cfg or DEFAULT.generation

    def answer(self, question: str, chunks: list[RetrievedChunk]) -> Answer:
        retrieved_pmids = list(dict.fromkeys(c.chunk.pmid for c in chunks))

        # Short-circuit on empty retrieval. Sending an empty context and hoping
        # the model refuses is a coin flip; refusing deterministically is free.
        if not chunks:
            return Answer(
                question=question,
                text=f"{INSUFFICIENT_MARKER}: no abstracts were retrieved for this question.",
                cited_pmids=[], retrieved_pmids=[], sources=[], insufficient_context=True,
            )

        raw = self.client.complete(
            SYSTEM_PROMPT,
            ANSWER_TEMPLATE.format(context=format_context(chunks), question=question),
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
        ).strip()

        cited = extract_citations(raw)
        hallucinated = [p for p in cited if p not in retrieved_pmids]
        if hallucinated:
            logger.warning("Answer cited PMIDs absent from context: %s", hallucinated)

        return Answer(
            question=question,
            text=raw,
            cited_pmids=cited,
            retrieved_pmids=retrieved_pmids,
            sources=chunks,
            insufficient_context=raw.startswith(INSUFFICIENT_MARKER),
            hallucinated_citations=hallucinated,
        )


class RAGPipeline:
    """Retriever + Answerer. The user-facing entry point."""

    def __init__(self, retriever, answerer: Answerer, *, top_k: int | None = None,
                 max_per_pmid: int = 1) -> None:
        self.retriever = retriever
        self.answerer = answerer
        self.top_k = top_k or DEFAULT.retrieval.top_k
        self.max_per_pmid = max_per_pmid

    def ask(self, question: str) -> Answer:
        from ..retrieval.retriever import deduplicate_by_pmid

        hits = self.retriever.retrieve(question, self.top_k * 3)
        hits = deduplicate_by_pmid(hits, max_per_pmid=self.max_per_pmid)[: self.top_k]
        return self.answerer.answer(question, hits)
