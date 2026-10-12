"""Central configuration.

Paths and model choices live here so no module hard-codes them. The stack is
deliberately cheap to run: embeddings are local and free, and the only paid API
call in the whole system is answer generation.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# PROJECT_ROOT is only meaningful in a repo checkout. Installed into
# site-packages (as in the Docker image) it points nowhere useful, so
# BIORAG_DATA_DIR lets the deployment say where the data actually lives.
DATA_DIR = Path(os.environ.get("BIORAG_DATA_DIR") or PROJECT_ROOT / "data")
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
INDEX_DIR = DATA_DIR / "index"


@dataclass(frozen=True)
class CleaningConfig:
    """Thresholds for corpus cleaning.

    `min_abstract_chars` at 200 is a deliberate choice: shorter records are
    almost always editorials, errata, or conference blurbs whose 'abstract' is a
    single sentence. They retrieve badly and pollute the index.
    """
    min_abstract_chars: int = 200
    max_abstract_chars: int = 12_000
    min_title_chars: int = 10
    drop_retracted: bool = True
    dedup_on_title: bool = True
    # 0.97, not the more intuitive 0.90. Fuzzy title matching is dangerous in a
    # narrow corpus: a group publishing serially on related compounds produces
    # titles like "Allosteric inhibition of PI3Ka by MK-2206" and "... of BRAF by
    # asciminib", which share >90% of their characters while describing entirely
    # different work. At 0.92 this silently discarded ~15% of a synthetic corpus
    # of genuinely distinct papers. Duplicates in PubMed are near-exact (errata,
    # reprints, corrected records), so a tight threshold catches what matters and
    # a loose one destroys real data.
    title_similarity_threshold: float = 0.97


@dataclass(frozen=True)
class ChunkingConfig:
    """Chunking parameters.

    NOTE the default `max_chunk_chars` of 2000 is large on purpose. See the
    docstring in ingestion/chunker.py — a PubMed abstract is already a coherent
    retrieval unit, and splitting it by default is a mistake most RAG tutorials
    make.
    """
    max_chunk_chars: int = 2000
    overlap_chars: int = 150
    split_on_sections: bool = True
    min_chunk_chars: int = 100


@dataclass(frozen=True)
class EmbeddingConfig:
    """Local embedding model.

    Default is a general-purpose MiniLM: small, fast, and free. For biomedical
    text a domain model (e.g. `pritamdeka/S-PubMedBert-MS-MARCO`) measurably
    beats it on retrieval, at ~5x the size and inference cost. The evaluation
    layer exists to let you MEASURE that trade-off on your own corpus rather
    than guess — swap `model_name` and re-run the eval.
    """
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    normalize: bool = True          # unit vectors, so dot product == cosine
    device: str | None = None       # None -> auto-detect cuda/mps/cpu


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 5
    candidate_k: int = 20           # pre-rerank / hybrid fusion pool
    hybrid_alpha: float = 0.5       # 1.0 = pure dense, 0.0 = pure BM25
    collection_name: str = "allosteric_kinase"


@dataclass(frozen=True)
class GenerationConfig:
    """The only layer that costs money."""
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 1024
    temperature: float = 0.0        # deterministic: this is a factual QA task


@dataclass(frozen=True)
class Config:
    cleaning: CleaningConfig = CleaningConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    generation: GenerationConfig = GenerationConfig()


DEFAULT = Config()
