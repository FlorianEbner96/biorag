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
| Generation | `generation/` | complete |
| Evaluation | `evaluation/` | complete |
| Query CLI | `examples/ask.py` | complete |
| Evaluation runner | `examples/evaluate.py` + `gold/` | complete |
| API | `api/` | complete |
| Container | `Dockerfile` + CI smoke test | complete |

122 tests, all passing, none requiring network, an API key, or a model download.

## Results

Evaluated on a real corpus of **409 PubMed abstracts** (494 chunks) against a
hand-written gold set of **18 questions** (`gold/allosteric_kinase_gold.json`):
factual, identifier, conceptual, multi-hop, and 22% deliberately unanswerable.
Reproduce with `python examples/evaluate.py --out results/eval_metrics.json`.

Retrieval quality over the 14 answerable questions (cutoff k=5):

| Retriever | recall@5 | precision@5 | MRR | hit rate |
|---|---|---|---|---|
| Dense (semantic, MiniLM) | 0.27 | 0.10 | 0.34 | 0.50 |
| BM25 (keyword baseline)  | 0.41 | 0.14 | 0.39 | 0.57 |
| **Hybrid (RRF)**         | **0.43** | **0.16** | 0.39 | **0.64** |

**Hybrid wins — but not for the textbook reason.** The naive expectation is that
semantic retrieval beats keyword search. Here it does not: dense alone is the
*weakest* of the three. Two honest limitations explain that:

1. **The gold set leans on exact identifiers.** Nine of the 14 answerable
   questions hinge on a specific code — EAI045, ABL001, MK-2206, IACS-13909,
   SHP099 — which is exactly what BM25 is built for and what a general embedding
   model handles poorly, since a compound code is a rare token it has barely seen.
2. **The embedding model is general-purpose, not biomedical.**
   `all-MiniLM-L6-v2` was never trained on this vocabulary. A domain model
   (`S-PubMedBert-MS-MARCO`) should lift the dense numbers materially — the
   evaluation harness exists precisely to *measure* that swap rather than assume it.

The result that does hold is the one that matters: **fusing the two beats either
alone** on recall (0.43) and hit rate (0.64). Neither retriever is redundant —
BM25 nails the identifiers, dense contributes on the conceptual questions BM25
misses, and Reciprocal Rank Fusion combines them with no score-scale tuning.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[generation,dev]" # installs torch/sentence-transformers too (a few hundred MB)
```

## Try it without any data (offline, no API key)

The demos run on synthetic PubMed-shaped data — no network, no model download,
no key. Good for seeing the whole pipeline end to end in seconds.

```bash
python examples/demo_retrieval.py    # clean -> chunk -> index -> retrieve
python examples/demo_evaluation.py   # retrieval + generation + metrics
pytest -q                            # 119 tests, all offline
```

## Run it on a real corpus

**1. Fetch abstracts from PubMed.** NCBI requires an email to identify callers;
a free [API key](https://www.ncbi.nlm.nih.gov/account/) is optional and only
raises the rate limit.

```bash
python -m biorag.data.fetch_pubmed --email you@example.com --retmax 1000
```

**2. Clean the corpus** (dedup, boilerplate stripping, retraction/quality
filters — prints a cleaning report).

```bash
python -m biorag.data.clean_corpus
```

**3. Build the index** (downloads the embedding model, ~80 MB, on first run).

```bash
python -m biorag.ingestion.pipeline
```

**4. Ask questions.**

```bash
python examples/ask.py "Which compounds inhibit PI3K-alpha allosterically?"
python examples/ask.py                 # interactive: several questions in a row
python examples/ask.py "..." --out results.json   # optional: also save results to a file
```

Saving is opt-in: without `--out` nothing is written; with `--out` each question
and its results are appended to the given JSON file. The evaluation demo takes the
same flag (`python examples/demo_evaluation.py --out metrics.json`).

Without an API key this returns the retrieved abstracts with their PMIDs — free.
For generated answers **with inline citations**, set a key (costs a few cents,
requires credits):

```bash
export ANTHROPIC_API_KEY=sk-ant-...    # Windows: $env:ANTHROPIC_API_KEY = "sk-ant-..."
python examples/ask.py
```

## Serve it over HTTP

```
pip install -e ".[api,generation]"
uvicorn biorag.api.app:app --reload
```

Then open <http://127.0.0.1:8000/docs> for the interactive OpenAPI schema.

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/health` | GET | Readiness, not just liveness — reports the number of indexed chunks and whether generation is configured. |
| `/ask` | POST | Answer a question. Generates a cited answer when `ANTHROPIC_API_KEY` is set, otherwise returns the retrieved abstracts. |
| `/search` | GET | Retrieval only, **never** generation — the guaranteed-free path, even when a key is configured. |

```
curl -s localhost:8000/health
curl -s -X POST localhost:8000/ask \
  -H 'content-type: application/json' \
  -d '{"question": "Which compounds inhibit PI3K-alpha allosterically?"}'
```

**The verification result is part of the response, not a server-side log line.**
`is_grounded` is false and `hallucinated_citations` is non-empty whenever the
model cited a PMID that was never retrieved. An API that returned the answer text
but dropped that flag would hand back exactly the unverifiable output this
project exists to avoid.

**Three decisions worth knowing about the API layer.** The index and embedding
model load once in a `lifespan` handler — at import time the module would be
unusable without an index, and per request it would pay an ~80 MB model load on
every call. Handlers take their dependencies through an injected `get_deps()`,
which is what keeps the API tests offline like the rest of the suite. And a
missing index returns **503, not 500**: "not ready to serve" and "crashed" mean
different things to an orchestrator.

## Run it in Docker

```
docker build -t biorag .
docker run --rm -p 8000:8000 biorag
```

The image serves immediately: the prebuilt index (`data/index/`, 494 chunks)
and the embedding model are baked in, so it starts with no setup and no network
access. Same endpoints as above, at <http://127.0.0.1:8000/docs>.

```
# Enable answer generation (retrieval-only without it). The key is passed at
# runtime and never written into the image.
docker run --rm -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-... biorag

# Serve a different corpus without rebuilding: mount your own index over the
# default one.
docker run --rm -p 8000:8000 -v "$PWD/data/index:/app/data/index:ro" biorag
```

**CPU-only PyTorch.** The default PyPI wheel bundles ~4 GB of CUDA libraries
that a MiniLM service on CPU never touches; installing torch from PyTorch's CPU
index first keeps the image around 1 GB instead of 5+. **Multi-stage build,
non-root user, allowlist `.dockerignore`:** build tooling never ships, the
process can't modify its own image, and a secret added to the repo later is
excluded from the build context by default. **The `HEALTHCHECK` requires
`status: "ok"`**, not merely a 200, so a container whose index is missing reports
unhealthy instead of quietly serving 503s.

CI ([`.github/workflows/docker.yml`](.github/workflows/docker.yml)) runs lint
and the offline test suite, then builds the image and smoke-tests it: it starts
the container, waits for `/health` to report all 494 chunks, and runs a real
`/search`. A green build means the image serves, not just that it built.

`BIORAG_DATA_DIR` overrides where the data directory lives (default: `data/` in
the repo checkout). The image sets it to `/app/data`.

## Data flow

```
fetch_pubmed  ->  clean_corpus  ->  ingestion.pipeline  ->  ask
 raw JSON          clean JSON        chunks + vector index    retrieve (+ answer)
 data/raw/         data/clean/       data/index/              PMID-cited output
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

**BM25's weakness here is paraphrase, not vocabulary.** On the real 409-abstract
corpus the domain terms ("allosteric", "kinase", "inhibitor") all carry positive
IDF, so BM25 ranks well on questions built from exact identifiers (EAI045, ABL001,
MK-2206). Where it fails is conceptual and multi-hop questions, whose wording
shares no tokens with the abstract that answers them (see Results) — the concrete
case for dense retrieval. (On a tiny corpus those same terms can take *negative*
IDF by appearing in >50% of documents; the real corpus is diverse enough that
they do not, which is itself a reason to evaluate on real rather than toy data.)

**Fuzzy title deduplication is tuned tight (0.97).** A group publishing serially
on related compounds produces titles sharing >90% of their characters while
describing different work, so a loose threshold throws away genuinely distinct
papers. On the real corpus the 0.97 threshold removed 4 near-duplicate titles out
of 413 fetched; a synthetic stress test showed that dropping to 0.92 would discard
~15% of genuinely distinct papers.

**Everything is testable offline.** `data/synthetic.py` generates PubMed-shaped
records with deliberately planted pathologies — retractions, boilerplate,
duplicate titles, one-line abstracts — so the cleaner is asserted against known
ground truth. Tests that need the network get skipped, and skipped tests protect
nothing.

**Citations are verified, not trusted.** The prompt asks for inline `[PMID]`
markers; the answerer then checks every cited PMID against what was actually
retrieved. A model citing a plausible PMID that was never in the context is the
most damaging failure this system has, because the output looks entirely
credible. `Answer.is_grounded` is False whenever that happens.

**~20% of the gold set is deliberately unanswerable.** This is the category most
evaluation sets omit, and the only one that detects a system which answers
everything confidently. A RAG system scoring perfectly on factual questions while
never refusing is useless in a scientific setting — `refusal_recall` is the metric
that catches it, and it cannot be computed at all without such questions.

**Retrieval and generation are scored separately, in that order.** If recall@5 is
0.4, no prompt engineering will fix the answers — the relevant abstract was never
in the context. Debugging answers when the problem is upstream in retrieval is the
most common time sink in these projects.

## Next

1. Fetch a real corpus and hand-write 20-30 gold questions against it, **before**
   tuning anything. Questions written after seeing what the system does well
   measure your own selection bias.
2. Run the retriever comparison on real data and paste the table here.
3. Swap in a biomedical embedding model (`S-PubMedBert-MS-MARCO`) and measure
   whether it beats MiniLM on this corpus, rather than assuming it does.
