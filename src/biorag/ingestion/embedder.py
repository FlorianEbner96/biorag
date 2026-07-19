"""Local sentence-transformer embeddings.

Runs on your own machine, costs nothing, needs no API key. The model is ~80MB
and downloads once to the HuggingFace cache.

Asymmetric retrieval note: a short question and a 250-word abstract are different
kinds of text, and some retrieval models expect an explicit prefix to distinguish
them (E5 wants "query: " / "passage: ", BGE wants an instruction on the query
only). MiniLM is symmetric and needs neither, but `query_prefix`/`passage_prefix`
exist here so that swapping to a stronger asymmetric model is a config change
rather than a rewrite. Getting this wrong is a silent 10-20% retrieval loss that
looks like nothing at all.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..config import DEFAULT, EmbeddingConfig
from ..schema import Chunk

logger = logging.getLogger(__name__)

# Prefixes for known asymmetric models, applied automatically by model name.
_MODEL_PREFIXES = {
    "intfloat/e5": ("query: ", "passage: "),
    "intfloat/multilingual-e5": ("query: ", "passage: "),
    "BAAI/bge": ("Represent this sentence for searching relevant passages: ", ""),
}


@dataclass
class EmbeddingResult:
    vectors: np.ndarray          # (n, dim), float32
    model_name: str
    dim: int


class Embedder:
    """Wraps a sentence-transformers model with batching and prefix handling.

    The model loads lazily on first use so that importing this module — which the
    API layer and the test suite both do — doesn't pull ~80MB into memory or
    trigger a download.
    """

    def __init__(self, cfg: EmbeddingConfig | None = None) -> None:
        self.cfg = cfg or DEFAULT.embedding
        self._model = None
        self._query_prefix, self._passage_prefix = self._resolve_prefixes(self.cfg.model_name)

    @staticmethod
    def _resolve_prefixes(model_name: str) -> tuple[str, str]:
        for stem, prefixes in _MODEL_PREFIXES.items():
            if model_name.lower().startswith(stem.lower()):
                logger.info("Model %s is asymmetric; applying retrieval prefixes.", model_name)
                return prefixes
        return "", ""

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # heavy; import late
            logger.info("Loading embedding model %s ...", self.cfg.model_name)
            self._model = SentenceTransformer(self.cfg.model_name, device=self.cfg.device)
        return self._model

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def _encode(self, texts: list[str], prefix: str, show_progress: bool) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        payload = [prefix + t for t in texts] if prefix else texts
        vectors = self.model.encode(
            payload,
            batch_size=self.cfg.batch_size,
            normalize_embeddings=self.cfg.normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_passages(self, texts: list[str], *, show_progress: bool = True) -> EmbeddingResult:
        """Embed documents for indexing."""
        vectors = self._encode(texts, self._passage_prefix, show_progress)
        return EmbeddingResult(vectors=vectors, model_name=self.cfg.model_name,
                               dim=vectors.shape[1] if len(vectors) else self.dim)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single question. Returns a 1-D vector."""
        return self._encode([text], self._query_prefix, show_progress=False)[0]

    def embed_chunks(self, chunks: list[Chunk], *, show_progress: bool = True) -> EmbeddingResult:
        return self.embed_passages([c.text for c in chunks], show_progress=show_progress)
