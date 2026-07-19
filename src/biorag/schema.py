"""Canonical record types passed between layers.

The invariant that matters: **every object downstream of the fetcher carries its
PMID**. The core promise of this system is that any generated claim can be traced
to a specific abstract, and provenance that is reconstructed later is provenance
that is wrong sometimes. Carry it, don't rebuild it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PubMedAbstract:
    """Raw record as parsed from EFetch XML. Mirrors data/fetch_pubmed.py."""
    pmid: str
    title: str
    abstract: str
    journal: str
    pub_year: str
    authors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PubMedAbstract":
        return cls(
            pmid=str(d.get("pmid", "")),
            title=d.get("title", "") or "",
            abstract=d.get("abstract", "") or "",
            journal=d.get("journal", "") or "",
            pub_year=str(d.get("pub_year", "") or ""),
            authors=list(d.get("authors", []) or []),
        )


@dataclass
class CleanAbstract:
    """Post-cleaning record.

    `sections` preserves the structured labels PubMed provides (BACKGROUND,
    METHODS, RESULTS, CONCLUSIONS). Keeping them is what lets the chunker split
    on semantic boundaries instead of arbitrary character counts, and lets the
    retriever prefer RESULTS/CONCLUSIONS text when a question asks what was found.
    """
    pmid: str
    title: str
    abstract: str
    journal: str
    pub_year: str
    authors: list[str]
    sections: dict[str, str] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        """Short human-readable citation for display next to a generated answer."""
        first = self.authors[0].split()[-1] if self.authors else "Unknown"
        etal = " et al." if len(self.authors) > 1 else ""
        year = self.pub_year or "n.d."
        return f"{first}{etal} ({year}), {self.journal}. PMID {self.pmid}"

    @property
    def url(self) -> str:
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CleanAbstract":
        return cls(
            pmid=str(d["pmid"]), title=d["title"], abstract=d["abstract"],
            journal=d.get("journal", ""), pub_year=str(d.get("pub_year", "")),
            authors=list(d.get("authors", [])), sections=dict(d.get("sections", {})),
        )


@dataclass
class Chunk:
    """An indexed unit of text, with provenance back to its source abstract."""
    chunk_id: str           # f"{pmid}::{ordinal}"
    pmid: str
    text: str
    section: str | None     # e.g. "RESULTS", or None for whole-abstract chunks
    ordinal: int
    title: str
    pub_year: str
    journal: str

    def to_metadata(self) -> dict[str, Any]:
        """Flat metadata dict for the vector store.

        Chroma metadata values must be str/int/float/bool — no None, no lists.
        Anything nullable is coerced to a string here rather than at the call
        site, so the constraint lives in one place.
        """
        return {
            "pmid": self.pmid,
            "section": self.section or "",
            "ordinal": self.ordinal,
            "title": self.title,
            "pub_year": self.pub_year or "",
            "journal": self.journal or "",
        }


@dataclass
class RetrievedChunk:
    """A chunk returned by a retriever, with its score and provenance intact."""
    chunk: Chunk
    score: float
    rank: int
    retriever: str          # "dense" | "bm25" | "hybrid"

    @property
    def url(self) -> str:
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.chunk.pmid}/"
