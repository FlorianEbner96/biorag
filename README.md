# biorag — Retrieval-Augmented QA over PubMed

A RAG system that answers questions about **allosteric kinase inhibitors** by
retrieving and citing PubMed abstracts, rather than relying on an LLM's
parametric knowledge.

**Core promise:** every generated claim traces to a specific abstract (PMID +
citation). Ungrounded generation is a credibility failure in a scientific domain,
so provenance is carried through every layer rather than reconstructed at the end.

## Status

| Layer | Module | State |
|---|---|---|
| Data — fetch | `data/fetch_pubmed.py` | complete |
| Data — clean | `data/clean_corpus.py` | complete |
| Ingestion — chunk | `ingestion/chunker.py` | complete |
| Ingestion — embed | `ingestion/embedder.py` | complete |
| Ingestion — store | `ingestion/store.py` | complete |
| Retrieval | `retrieval/retriever.py` | complete |
| Generation | `generation/` | not started |
| Evaluation | `evaluation/` | not started |
| API | `api/` | not started |

64 tests, all passing, none requiring network or a model download.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e . && pip install -r requirements.txt

# See the whole pipeline run on synthetic data — no API key, no network
python examples/demo_retrieval.py

# Real corpus
python -m biorag.data.fetch_pubmed --email you@example.com --retmax 1000
python -m biorag.data.clean_corpus
python -m biorag.ingestion.pipeline

pytest -q
```

## Design decisions worth knowing

**Abstracts are not chunked by default.** A PubMed abstract is ~250 words and is
already a coherent retrieval unit written by an expert to summarise one paper.
Splitting it into 200-token windows — the reflexive tutorial default, inherited
from chunking long PDFs — separates findings from the conditions that qualify
them. Whole abstracts are kept intact unless they exceed 2000 characters (~95% do
not); only then do we split, and only on the author's own section boundaries.

**Unicode is normalised but not ASCII-folded.** PI3Kα and PI3Kβ are different
proteins with different inhibitors. The standard
`normalize('NFKD').encode('ascii','ignore')` cleaning idiom collapses both to
"PI3K" and destroys the distinction a question may turn on.

**Exact search, not a vector database.** At a few thousand chunks, a 3000×384
float32 matrix is 4.6 MB and one matrix-vector product takes single-digit
milliseconds. ANN indexes trade recall for speed at 10⁵–10⁸ vectors; at 10³ they
add a dependency and lose recall in exchange for imperceptible speed. A Chroma
backend is included for persistence, behind the same interface.

**Hybrid retrieval fuses ranks, not scores.** Cosine similarities and BM25 scores
are not commensurable, and BM25's scale is corpus-dependent. Reciprocal Rank
Fusion combines ranks, so it needs no score-scale tuning.

**BM25 scores are negative on this corpus, correctly.** "Allosteric", "kinase",
and "inhibitor" appear in nearly every abstract, so their IDF is negative. That
is BM25 reporting that the domain's defining terms carry no discriminative power
here — and a concrete argument for the dense retriever rather than an argument
against the baseline.

**Fuzzy title deduplication is tuned tight (0.97).** A group publishing serially
on related compounds produces titles sharing >90% of their characters while
describing different work. At 0.92 this discarded ~15% of a corpus of genuinely
distinct papers.

**Everything is testable offline.** `data/synthetic.py` generates PubMed-shaped
records with deliberately planted pathologies — retractions, boilerplate,
duplicate titles, one-line abstracts — so the cleaner is asserted against known
ground truth. Tests that need the network get skipped, and skipped tests protect
nothing.

## Next

Generation layer (Claude, constrained to answer only from retrieved context, with
inline PMID citations), then the evaluation layer — hand-written questions with
known answers, measuring RAG against the keyword baseline for a quantitative
results section rather than a demo.
