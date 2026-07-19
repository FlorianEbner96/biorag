"""Chunking clean abstracts into indexable units.

**The default is not to split.** This is the most important decision in the file
and it runs against most RAG tutorials, so it is worth stating plainly:

A PubMed abstract averages ~250 words and is *already* a coherent, self-contained
retrieval unit written by a domain expert to summarise one paper. Splitting it
into 200-token windows — the reflexive tutorial default, inherited from
chunking 40-page PDFs — actively destroys retrieval quality here:

  - A window containing "IC50 of 12 nM" without the sentence naming the compound
    is unusable in an answer, and worse, it retrieves well because it is dense
    with the query's numeric terms.
  - Splitting separates a finding from the experimental condition that qualifies
    it, which in this domain frequently inverts the meaning.
  - Embedding quality degrades on fragments: models are trained on coherent
    passages, not clauses.

So: keep the whole abstract as one chunk unless it exceeds `max_chunk_chars`
(default 2000, which retains ~95% of PubMed abstracts intact). Only then split,
and split on **section boundaries** — BACKGROUND / METHODS / RESULTS /
CONCLUSIONS — because those are semantic joints the author already marked.
Character-window splitting is the last resort, not the first move.

Title is prepended to every chunk. It costs ~15 tokens and reliably improves
retrieval, because the title names the compound and target that the abstract body
often refers to only by pronoun or abbreviation.
"""
from __future__ import annotations

import logging
import re

from ..config import DEFAULT, ChunkingConfig
from ..schema import Chunk, CleanAbstract

logger = logging.getLogger(__name__)

# Sections in the order a reader would encounter them. Used to keep split chunks
# in document order rather than dict-insertion order.
SECTION_ORDER = [
    "BACKGROUND", "OBJECTIVE", "OBJECTIVES", "INTRODUCTION", "PURPOSE", "AIM",
    "METHODS", "MATERIALS AND METHODS", "DESIGN",
    "RESULTS", "FINDINGS",
    "DISCUSSION", "CONCLUSION", "CONCLUSIONS", "SIGNIFICANCE",
]
_SECTION_RANK = {name: i for i, name in enumerate(SECTION_ORDER)}

_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


def _with_title(title: str, body: str) -> str:
    """Prepend the title so every chunk carries its own topical anchor."""
    title = title.strip().rstrip(".")
    return f"{title}. {body.strip()}" if title else body.strip()


def _split_on_sentences(text: str, max_chars: int, overlap: int, min_chars: int) -> list[str]:
    """Last-resort splitter. Breaks on sentence boundaries, never mid-sentence.

    A regex sentence splitter is wrong on biomedical text roughly 2% of the time
    ("approx. 5 nM", "Fig. 3", "S. aureus", "et al."). That error rate is
    acceptable HERE because this path handles only the ~5% of abstracts too long
    to keep whole — but it would not be acceptable as the primary code path,
    which is another reason not to make splitting the default.
    """
    sentences = _SENTENCE_END_RE.split(text)
    chunks: list[str] = []
    buffer: list[str] = []
    length = 0

    for sentence in sentences:
        s_len = len(sentence) + 1
        if length + s_len > max_chars and buffer:
            chunks.append(" ".join(buffer).strip())
            if overlap > 0:
                tail, tail_len = [], 0
                for prev in reversed(buffer):
                    if tail_len + len(prev) > overlap:
                        break
                    tail.insert(0, prev)
                    tail_len += len(prev) + 1
                buffer, length = tail, tail_len
            else:
                buffer, length = [], 0
        buffer.append(sentence)
        length += s_len

    if buffer:
        tail = " ".join(buffer).strip()
        # Don't emit a runt chunk; fold it into the previous one.
        if chunks and len(tail) < min_chars:
            chunks[-1] = f"{chunks[-1]} {tail}".strip()
        elif tail:
            chunks.append(tail)

    return chunks


def _ordered_sections(sections: dict[str, str]) -> list[tuple[str, str]]:
    """Sections in reading order; unknown labels keep their original position."""
    items = list(sections.items())
    return sorted(items, key=lambda kv: _SECTION_RANK.get(kv[0], len(SECTION_ORDER)))


def chunk_abstract(
    abstract: CleanAbstract,
    cfg: ChunkingConfig | None = None,
) -> list[Chunk]:
    """Chunk one abstract. Usually returns exactly one chunk, by design."""
    cfg = cfg or DEFAULT.chunking
    body = abstract.abstract.strip()
    if not body:
        return []

    full = _with_title(abstract.title, body)

    def make(text: str, section: str | None, ordinal: int) -> Chunk:
        return Chunk(
            chunk_id=f"{abstract.pmid}::{ordinal}",
            pmid=abstract.pmid,
            text=text,
            section=section,
            ordinal=ordinal,
            title=abstract.title,
            pub_year=abstract.pub_year,
            journal=abstract.journal,
        )

    # The common case: the whole abstract fits, so keep it intact.
    if len(full) <= cfg.max_chunk_chars:
        return [make(full, None, 0)]

    # Too long. Prefer the author's own section boundaries.
    if cfg.split_on_sections and abstract.sections:
        chunks: list[Chunk] = []
        for label, text in _ordered_sections(abstract.sections):
            text = text.strip()
            if len(text) < cfg.min_chunk_chars:
                continue
            piece = _with_title(abstract.title, f"{label}: {text}")
            if len(piece) <= cfg.max_chunk_chars:
                chunks.append(make(piece, label, len(chunks)))
            else:
                for part in _split_on_sentences(text, cfg.max_chunk_chars,
                                                cfg.overlap_chars, cfg.min_chunk_chars):
                    chunks.append(make(_with_title(abstract.title, f"{label}: {part}"),
                                       label, len(chunks)))
        if chunks:
            return chunks

    # Unstructured and long: sentence-boundary split.
    return [
        make(_with_title(abstract.title, part), None, i)
        for i, part in enumerate(
            _split_on_sentences(body, cfg.max_chunk_chars, cfg.overlap_chars, cfg.min_chunk_chars)
        )
    ]


def chunk_corpus(
    abstracts: list[CleanAbstract],
    cfg: ChunkingConfig | None = None,
) -> list[Chunk]:
    """Chunk a whole corpus, logging the split ratio.

    Watch that ratio. If it climbs much above ~1.15 chunks per abstract, either
    `max_chunk_chars` is set too low or the corpus is unusual — both worth
    knowing before you index a few thousand fragments.
    """
    cfg = cfg or DEFAULT.chunking
    chunks: list[Chunk] = []
    for abstract in abstracts:
        chunks.extend(chunk_abstract(abstract, cfg))

    if abstracts:
        ratio = len(chunks) / len(abstracts)
        logger.info("Chunked %d abstracts into %d chunks (%.2f per abstract)",
                    len(abstracts), len(chunks), ratio)
        if ratio > 1.15:
            logger.warning(
                "Split ratio %.2f is high — most abstracts should stay intact. "
                "Check ChunkingConfig.max_chunk_chars (currently %d).",
                ratio, cfg.max_chunk_chars,
            )
    return chunks
