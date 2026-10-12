"""Chunking. The headline assertion: abstracts are NOT split by default."""
from __future__ import annotations

from biorag.config import ChunkingConfig
from biorag.data.clean_corpus import clean_corpus
from biorag.data.synthetic import make_abstract
from biorag.ingestion.chunker import chunk_abstract, chunk_corpus


class TestNoUnnecessarySplitting:
    def test_typical_abstract_stays_intact(self, clean_abstracts):
        """*** The most important test in this file. ***

        A PubMed abstract is already a coherent retrieval unit. Splitting it is
        the reflexive tutorial default and it is wrong here.
        """
        for abstract in clean_abstracts[:20]:
            assert len(chunk_abstract(abstract)) == 1

    def test_corpus_split_ratio_is_near_one(self, clean_abstracts):
        chunks = chunk_corpus(clean_abstracts)
        assert len(chunks) / len(clean_abstracts) < 1.15

    def test_long_abstract_is_split(self, clean_abstracts):
        """When splitting IS warranted, it happens."""
        long_one = clean_abstracts[0]
        cfg = ChunkingConfig(max_chunk_chars=200, min_chunk_chars=50)
        assert len(chunk_abstract(long_one, cfg)) > 1


class TestProvenance:
    def test_every_chunk_carries_its_pmid(self, chunks):
        assert all(c.pmid and c.pmid.isdigit() for c in chunks)

    def test_chunk_ids_are_unique(self, chunks):
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_id_encodes_pmid(self, chunks):
        for c in chunks:
            assert c.chunk_id.startswith(f"{c.pmid}::")

    def test_metadata_has_no_none_values(self, chunks):
        """Chroma rejects None in metadata. Coercion lives in to_metadata()."""
        for c in chunks:
            assert all(v is not None for v in c.to_metadata().values())


class TestTitlePrepending:
    def test_title_is_prepended(self, clean_abstracts):
        abstract = clean_abstracts[0]
        chunk = chunk_abstract(abstract)[0]
        assert chunk.text.startswith(abstract.title.rstrip("."))

    def test_title_appears_in_every_split_chunk(self, clean_abstracts):
        """Each chunk must be independently interpretable once retrieved."""
        cfg = ChunkingConfig(max_chunk_chars=250, min_chunk_chars=50)
        abstract = clean_abstracts[0]
        for chunk in chunk_abstract(abstract, cfg):
            assert abstract.title.rstrip(".")[:30] in chunk.text


class TestSectionAwareSplitting:
    def test_structured_abstract_splits_on_sections(self):
        raw = make_abstract("1", structured=True)
        cleaned, _ = clean_corpus([raw])
        cfg = ChunkingConfig(max_chunk_chars=300, min_chunk_chars=40, split_on_sections=True)
        chunks = chunk_abstract(cleaned[0], cfg)
        assert len(chunks) > 1
        assert any(c.section in {"BACKGROUND", "METHODS", "RESULTS", "CONCLUSIONS"} for c in chunks)

    def test_sections_kept_in_reading_order(self):
        raw = make_abstract("1", structured=True)
        cleaned, _ = clean_corpus([raw])
        cfg = ChunkingConfig(max_chunk_chars=300, min_chunk_chars=40)
        sections = [c.section for c in chunk_abstract(cleaned[0], cfg) if c.section]
        if "BACKGROUND" in sections and "CONCLUSIONS" in sections:
            assert sections.index("BACKGROUND") < sections.index("CONCLUSIONS")

    def test_no_chunk_exceeds_limit_by_much(self, clean_abstracts):
        cfg = ChunkingConfig(max_chunk_chars=400, min_chunk_chars=50)
        for abstract in clean_abstracts[:10]:
            for chunk in chunk_abstract(abstract, cfg):
                # Title prefix and sentence boundaries allow modest overshoot.
                assert len(chunk.text) < cfg.max_chunk_chars * 2


def test_empty_corpus():
    assert chunk_corpus([]) == []
