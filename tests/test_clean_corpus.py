"""Corpus cleaning."""
from __future__ import annotations

from biorag.data.clean_corpus import (
    clean_corpus, clean_record, is_retracted, normalize_text,
    parse_sections, strip_boilerplate,
)
from biorag.data.synthetic import make_abstract, make_corpus


class TestNormalization:
    def test_greek_letters_survive(self):
        """*** The domain-critical assertion. ***

        PI3Kα and PI3Kβ are different proteins. Standard ASCII-folding text
        cleanup collapses both to 'PI3K' and destroys the distinction the whole
        question might turn on.
        """
        assert "α" in normalize_text("Inhibition of PI3Kα by alpelisib")
        assert normalize_text("PI3Kα") != normalize_text("PI3Kβ")

    def test_subscripts_and_symbols_survive(self):
        assert "β" in normalize_text("TGF-β signalling")
        assert "μ" in normalize_text("5 μM concentration")

    def test_ligatures_are_expanded(self):
        assert "ﬁ" not in normalize_text("the ﬁrst inhibitor")
        assert "first" in normalize_text("the ﬁrst inhibitor")

    def test_whitespace_collapsed(self):
        assert normalize_text("a   b\n\n\n\nc") == "a b\n\nc"

    def test_empty_input(self):
        assert normalize_text("") == ""


class TestBoilerplate:
    def test_copyright_line_removed(self):
        text = "Real findings here.\nCopyright © 2023 Elsevier Ltd. All rights reserved."
        cleaned, modified = strip_boilerplate(text)
        assert modified
        assert "Copyright" not in cleaned
        assert "Real findings here." in cleaned

    def test_clean_text_untouched(self):
        text = "A perfectly normal abstract with no boilerplate."
        cleaned, modified = strip_boilerplate(text)
        assert not modified and cleaned == text

    def test_trial_registration_tail_removed(self):
        text = "Findings.\nTRIAL REGISTRATION: NCT01234567."
        cleaned, _ = strip_boilerplate(text)
        assert "NCT01234567" not in cleaned


class TestSectionParsing:
    def test_structured_abstract_parsed(self):
        record = make_abstract("1", structured=True)
        sections = parse_sections(record.abstract)
        assert set(sections) >= {"BACKGROUND", "METHODS", "RESULTS", "CONCLUSIONS"}
        assert all(v.strip() for v in sections.values())

    def test_unstructured_returns_empty(self):
        """Roughly half of PubMed abstracts are unstructured. Not an error."""
        assert parse_sections(make_abstract("1", structured=False).abstract) == {}

    def test_section_content_preserved(self):
        sections = parse_sections(make_abstract("1", structured=True).abstract)
        assert "IC50" in " ".join(sections.values())


class TestRetraction:
    def test_retracted_detected(self):
        assert is_retracted(make_abstract("1", retracted=True))

    def test_normal_record_not_flagged(self):
        assert not is_retracted(make_abstract("1"))


class TestCleanCorpus:
    def test_planted_pathologies_are_dropped(self, raw_corpus):
        """The generator plants exactly one of each. Assert each is caught."""
        _, report = clean_corpus(raw_corpus)
        assert report.dropped_retracted == 1
        assert report.dropped_too_short == 1
        assert report.dropped_duplicate_title == 1

    def test_good_records_retained(self, raw_corpus):
        cleaned, report = clean_corpus(raw_corpus)
        assert len(cleaned) == 40
        assert report.n_input == 43 and report.n_output == 40

    def test_no_duplicate_pmids_in_output(self, clean_abstracts):
        pmids = [a.pmid for a in clean_abstracts]
        assert len(pmids) == len(set(pmids))

    def test_boilerplate_gone_from_output(self, clean_abstracts):
        assert not any("Copyright" in a.abstract for a in clean_abstracts)

    def test_provenance_survives_cleaning(self, clean_abstracts):
        """Every record keeps a PMID. This is the system's core promise."""
        assert all(a.pmid and a.pmid.isdigit() for a in clean_abstracts)

    def test_report_accounts_for_every_record(self, raw_corpus):
        """Inputs must equal outputs plus drops — no records vanish silently."""
        _, r = clean_corpus(raw_corpus)
        total_dropped = (r.dropped_no_abstract + r.dropped_too_short + r.dropped_too_long
                         + r.dropped_no_title + r.dropped_retracted
                         + r.dropped_duplicate_pmid + r.dropped_duplicate_title)
        assert r.n_input == r.n_output + total_dropped

    def test_exact_duplicate_pmid_dropped(self):
        record = make_abstract("30000001")
        _, report = clean_corpus([record, record])
        assert report.dropped_duplicate_pmid == 1

    def test_citation_format(self, clean_abstracts):
        citation = clean_abstracts[0].citation
        assert "PMID" in citation and clean_abstracts[0].pmid in citation

    def test_url_is_valid(self, clean_abstracts):
        assert clean_abstracts[0].url.startswith("https://pubmed.ncbi.nlm.nih.gov/")

    def test_empty_corpus(self):
        cleaned, report = clean_corpus([])
        assert cleaned == [] and report.n_input == 0

    def test_report_summary_renders(self, raw_corpus):
        _, report = clean_corpus(raw_corpus)
        assert "Corpus cleaning report" in report.summary()
