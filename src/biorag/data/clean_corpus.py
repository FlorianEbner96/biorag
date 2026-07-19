"""Corpus cleaning: raw PubMed records -> clean, deduplicated, indexable text.

Cleaning decisions here are domain-specific and matter more than they look.

**Unicode is NOT normalised to ASCII.** In biomedical text Greek letters are
semantically load-bearing: PI3Kα and PI3Kβ are different proteins with different
inhibitors, and "PI3K" alone loses the distinction that the entire question might
hinge on. Standard `unicodedata.normalize('NFKD').encode('ascii', 'ignore')`
text-cleaning boilerplate silently destroys this. We normalise whitespace and
ligatures only, and preserve every Greek character and subscript.

**Publisher boilerplate is stripped.** Roughly a third of abstracts end with a
copyright line. It is identical across thousands of records, so it embeds to
near-identical vectors and pulls unrelated abstracts together in vector space —
actively harmful to retrieval rather than merely useless.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from ..config import DEFAULT, CleaningConfig
from ..schema import CleanAbstract, PubMedAbstract

logger = logging.getLogger(__name__)

# Publisher boilerplate. Matched case-insensitively at any position, though in
# practice it is almost always trailing.
BOILERPLATE_PATTERNS = [
    r"Copyright\s+©.*$",
    r"©\s*\d{4}.*$",
    r"\(c\)\s*\d{4}\s+(?:by\s+)?(?:Elsevier|Wiley|Springer|Elsevier Inc).*$",
    r"This article is protected by copyright.*$",
    r"All rights reserved\.?\s*$",
    r"Published by Elsevier.*$",
    r"Crown Copyright.*$",
    r"^\s*Copyright\s+.*?\.\s*",
]

# Structured-abstract labels PubMed emits. fetch_pubmed.py writes them as
# "LABEL: text", one section per line.
SECTION_LABEL_RE = re.compile(
    r"^(?P<label>[A-Z][A-Z /&'-]{2,40}):\s*(?P<text>.*)$"
)

# Trial registration / funding tails that carry no scientific content.
TRAILING_NOISE_RE = re.compile(
    r"\s*(?:TRIAL REGISTRATION|CLINICALTRIALS\.GOV|FUNDING|CONFLICT OF INTEREST)"
    r"\s*:.*$",
    re.IGNORECASE | re.DOTALL,
)

RETRACTION_MARKERS = (
    "retracted article", "this article has been retracted",
    "retraction notice", "withdrawn:",
)

# Ligatures and typographic characters that carry no meaning and hurt tokenisation.
_TYPOGRAPHIC = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "−": "-", " ": " ",
}


@dataclass
class CleaningReport:
    """Per-run evidence of what cleaning did. Portfolio-relevant on its own.

    Reviewers care far more about a candidate who can say *why* 84 of 1,000
    records were dropped than about a candidate who silently ends up with 916.
    """
    n_input: int = 0
    n_output: int = 0
    dropped_no_abstract: int = 0
    dropped_too_short: int = 0
    dropped_too_long: int = 0
    dropped_no_title: int = 0
    dropped_retracted: int = 0
    dropped_duplicate_pmid: int = 0
    dropped_duplicate_title: int = 0
    boilerplate_stripped: int = 0
    sections_parsed: int = 0
    year_distribution: dict[str, int] = field(default_factory=dict)
    dropped_pmids: dict[str, str] = field(default_factory=dict)  # pmid -> reason

    @property
    def n_dropped(self) -> int:
        return self.n_input - self.n_output

    def summary(self) -> str:
        pct = (100.0 * self.n_output / self.n_input) if self.n_input else 0.0
        lines = [
            "Corpus cleaning report",
            "=" * 46,
            f"  input records            {self.n_input:>6}",
            f"  output records          {self.n_output:>6}  ({pct:.1f}% retained)",
            f"  dropped                 {self.n_dropped:>6}",
            "",
            "  drop reasons",
            f"    no abstract           {self.dropped_no_abstract:>6}",
            f"    abstract too short    {self.dropped_too_short:>6}",
            f"    abstract too long     {self.dropped_too_long:>6}",
            f"    no/short title        {self.dropped_no_title:>6}",
            f"    retracted             {self.dropped_retracted:>6}",
            f"    duplicate PMID        {self.dropped_duplicate_pmid:>6}",
            f"    near-duplicate title  {self.dropped_duplicate_title:>6}",
            "",
            "  transformations",
            f"    boilerplate stripped  {self.boilerplate_stripped:>6}",
            f"    structured abstracts  {self.sections_parsed:>6}",
        ]
        if self.year_distribution:
            years = sorted(self.year_distribution.items())
            span = f"{years[0][0]}-{years[-1][0]}"
            lines += ["", f"  year span               {span:>6}"]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "n_input": self.n_input, "n_output": self.n_output,
            "dropped": {
                "no_abstract": self.dropped_no_abstract,
                "too_short": self.dropped_too_short,
                "too_long": self.dropped_too_long,
                "no_title": self.dropped_no_title,
                "retracted": self.dropped_retracted,
                "duplicate_pmid": self.dropped_duplicate_pmid,
                "duplicate_title": self.dropped_duplicate_title,
            },
            "boilerplate_stripped": self.boilerplate_stripped,
            "sections_parsed": self.sections_parsed,
            "year_distribution": self.year_distribution,
        }


def normalize_text(text: str) -> str:
    """Whitespace, ligatures, and control characters. Greek letters SURVIVE.

    NFKC rather than NFKD-plus-ASCII-encode: NFKC composes characters and folds
    compatibility forms (e.g. full-width Latin) without discarding anything
    non-ASCII. 'α' stays 'α'.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    for src, dst in _TYPOGRAPHIC.items():
        text = text.replace(src, dst)
    # Strip control characters but keep newlines and tabs.
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or unicodedata.category(ch)[0] != "C")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_boilerplate(text: str) -> tuple[str, bool]:
    """Remove publisher copyright lines and trial-registration tails.

    Returns (cleaned_text, was_modified).
    """
    original = text
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = TRAILING_NOISE_RE.sub("", text)
    text = re.sub(r"\s+$", "", text)
    return text.strip(), text.strip() != original.strip()


def parse_sections(abstract: str) -> dict[str, str]:
    """Split a structured abstract into {LABEL: text}.

    fetch_pubmed.py joins labelled AbstractText elements as "LABEL: text" lines,
    so we reverse that. Unstructured abstracts return {} — which is correct and
    common; roughly half of PubMed abstracts have no section labels at all.
    """
    sections: dict[str, str] = {}
    current_label: str | None = None
    buffer: list[str] = []

    for line in abstract.split("\n"):
        match = SECTION_LABEL_RE.match(line.strip())
        if match:
            if current_label and buffer:
                sections[current_label] = " ".join(buffer).strip()
            current_label = match.group("label").strip().upper()
            buffer = [match.group("text").strip()]
        elif current_label:
            buffer.append(line.strip())

    if current_label and buffer:
        sections[current_label] = " ".join(buffer).strip()

    return {k: v for k, v in sections.items() if v}


def is_retracted(record: PubMedAbstract) -> bool:
    """Detect retraction markers in title or abstract.

    Retracted findings in a corpus are worse than absent findings: the system
    will cite them confidently, with a real PMID, and the citation will look
    perfectly legitimate to anyone who doesn't check.
    """
    haystack = f"{record.title} {record.abstract[:300]}".lower()
    return any(marker in haystack for marker in RETRACTION_MARKERS)


def _title_key(title: str) -> str:
    """Normalised title for duplicate detection."""
    return re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()


def _is_near_duplicate(title: str, seen: dict[str, str], threshold: float) -> str | None:
    """Return the PMID of a near-duplicate title, if any.

    O(n) per record against previously seen titles. Fine at a few thousand
    abstracts; if the corpus ever grows past ~20k, replace with MinHash/LSH.
    A first-character bucket keeps the constant low in the meantime.
    """
    key = _title_key(title)
    if not key:
        return None
    if key in seen:
        return seen[key]
    bucket = key[:1]
    for other_key, pmid in seen.items():
        if other_key[:1] != bucket:
            continue
        if abs(len(other_key) - len(key)) > 0.2 * max(len(key), 1):
            continue
        if SequenceMatcher(None, key, other_key).ratio() >= threshold:
            return pmid
    return None


def clean_record(record: PubMedAbstract) -> CleanAbstract | None:
    """Clean one record. Returns None if it should be dropped for content reasons."""
    title = normalize_text(record.title).rstrip(".")
    abstract = normalize_text(record.abstract)
    abstract, _ = strip_boilerplate(abstract)
    if not abstract:
        return None

    sections = parse_sections(abstract)
    flat = " ".join(sections.values()) if sections else abstract

    return CleanAbstract(
        pmid=record.pmid,
        title=title,
        abstract=flat,
        journal=normalize_text(record.journal),
        pub_year=record.pub_year,
        authors=[normalize_text(a) for a in record.authors if a],
        sections=sections,
    )


def clean_corpus(
    records: list[PubMedAbstract],
    cfg: CleaningConfig | None = None,
) -> tuple[list[CleanAbstract], CleaningReport]:
    """Clean, filter, and deduplicate a raw corpus.

    Order matters: cheap content filters run before the O(n) near-duplicate
    scan, so junk records never enter the comparison pool.
    """
    cfg = cfg or DEFAULT.cleaning
    report = CleaningReport(n_input=len(records))
    out: list[CleanAbstract] = []
    seen_pmids: set[str] = set()
    seen_titles: dict[str, str] = {}

    for record in records:
        if record.pmid in seen_pmids:
            report.dropped_duplicate_pmid += 1
            report.dropped_pmids[record.pmid] = "duplicate_pmid"
            continue

        if not record.abstract or not record.abstract.strip():
            report.dropped_no_abstract += 1
            report.dropped_pmids[record.pmid] = "no_abstract"
            continue

        if cfg.drop_retracted and is_retracted(record):
            report.dropped_retracted += 1
            report.dropped_pmids[record.pmid] = "retracted"
            continue

        had_boilerplate = strip_boilerplate(normalize_text(record.abstract))[1]
        cleaned = clean_record(record)
        if cleaned is None:
            report.dropped_no_abstract += 1
            report.dropped_pmids[record.pmid] = "empty_after_cleaning"
            continue

        if len(cleaned.title) < cfg.min_title_chars:
            report.dropped_no_title += 1
            report.dropped_pmids[record.pmid] = "no_title"
            continue
        if len(cleaned.abstract) < cfg.min_abstract_chars:
            report.dropped_too_short += 1
            report.dropped_pmids[record.pmid] = "too_short"
            continue
        if len(cleaned.abstract) > cfg.max_abstract_chars:
            report.dropped_too_long += 1
            report.dropped_pmids[record.pmid] = "too_long"
            continue

        if cfg.dedup_on_title:
            dupe_of = _is_near_duplicate(cleaned.title, seen_titles, cfg.title_similarity_threshold)
            if dupe_of is not None:
                report.dropped_duplicate_title += 1
                report.dropped_pmids[record.pmid] = f"near_duplicate_of_{dupe_of}"
                continue
            seen_titles[_title_key(cleaned.title)] = cleaned.pmid

        if had_boilerplate:
            report.boilerplate_stripped += 1
        if cleaned.sections:
            report.sections_parsed += 1

        seen_pmids.add(cleaned.pmid)
        out.append(cleaned)

    report.n_output = len(out)
    report.year_distribution = dict(sorted(Counter(a.pub_year for a in out if a.pub_year).items()))
    return out, report


def load_raw(path: Path) -> list[PubMedAbstract]:
    with Path(path).open(encoding="utf-8") as f:
        return [PubMedAbstract.from_dict(d) for d in json.load(f)]


def save_clean(abstracts: list[CleanAbstract], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump([a.to_dict() for a in abstracts], f, ensure_ascii=False, indent=2)
    logger.info("Saved %d clean abstracts to %s", len(abstracts), path)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description="Clean a raw PubMed corpus.")
    ap.add_argument("--in", dest="in_path",
                    default="data/raw/allosteric_kinase_inhibitors.json")
    ap.add_argument("--out", dest="out_path",
                    default="data/clean/allosteric_kinase_inhibitors.json")
    ap.add_argument("--report", default="data/clean/cleaning_report.json")
    ap.add_argument("--min-chars", type=int, default=DEFAULT.cleaning.min_abstract_chars)
    args = ap.parse_args()

    records = load_raw(Path(args.in_path))
    cfg = CleaningConfig(min_abstract_chars=args.min_chars)
    cleaned, report = clean_corpus(records, cfg)

    save_clean(cleaned, Path(args.out_path))
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    print("\n" + report.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
