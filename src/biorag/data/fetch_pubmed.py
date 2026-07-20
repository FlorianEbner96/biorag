"""
fetch_pubmed.py

Fetches abstracts from PubMed via NCBI's E-utilities API for a given
search topic and caches the raw results as JSON.

This is the first module in the pipeline:
    fetch_pubmed.py -> clean_corpus.py -> ingestion/chunker.py -> ...

Project scope: allosteric kinase inhibitors.

NCBI E-utilities usage policy:
    - Without an API key: max 3 requests/second.
    - With a free NCBI API key: max 10 requests/second.
    - Always identify yourself via `email` (and ideally `tool`) params.
    - Get a free API key at: https://www.ncbi.nlm.nih.gov/account/

Docs: https://www.ncbi.nlm.nih.gov/books/NBK25501/
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ESEARCH_URL = f"{EUTILS_BASE}/esearch.fcgi"
EFETCH_URL = f"{EUTILS_BASE}/efetch.fcgi"

# Default project scope. Kept narrow on purpose so the resulting corpus
# stays in the low thousands of abstracts rather than "all of oncology".
DEFAULT_QUERY = (
    '("allosteric inhibitor"[tiab] OR "allosteric modulator"[tiab]) '
    'AND ("kinase"[tiab] OR "kinase inhibitor"[tiab]) '
    'AND (drug discovery[tiab] OR medicinal chemistry[tiab])'
)


@dataclass
class PubMedAbstract:
    pmid: str
    title: str
    abstract: str
    journal: str
    pub_year: str
    authors: list[str]


class PubMedFetcher:
    """Thin client around NCBI E-utilities (esearch + efetch)."""

    def __init__(
        self,
        email: str,
        api_key: str | None = None,
        tool_name: str = "biomed-rag-project",
        requests_per_second: float | None = None,
    ) -> None:
        if not email:
            raise ValueError(
                "NCBI requires an email address to identify API callers. "
                "Pass --email you@example.com"
            )
        self.email = email
        self.api_key = api_key
        self.tool_name = tool_name
        # Stay one notch under NCBI's published limits (3/s or 10/s with a key)
        self.rps = requests_per_second or (9.0 if api_key else 2.5)
        self._min_interval = 1.0 / self.rps
        self._last_request_ts = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_ts
        wait = self._min_interval - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_ts = time.monotonic()

    def _base_params(self) -> dict:
        params = {"email": self.email, "tool": self.tool_name}
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def search_pmids(self, query: str, retmax: int = 1000) -> list[str]:
        """Run an ESearch query and return matching PubMed IDs."""
        params = {
            **self._base_params(),
            "db": "pubmed",
            "term": query,
            "retmax": retmax,
            "retmode": "json",
            "sort": "relevance",
        }
        self._throttle()
        resp = requests.get(ESEARCH_URL, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        pmids = payload.get("esearchresult", {}).get("idlist", [])
        logger.info("ESearch matched %d PMIDs (requested up to %d)", len(pmids), retmax)
        return pmids

    def fetch_abstracts(self, pmids: list[str], batch_size: int = 150) -> list[PubMedAbstract]:
        """Fetch title/abstract/metadata for a list of PMIDs via EFetch, batched.

        Uses HTTP POST, not GET. NCBI recommends POST whenever the ID list is
        large: with a few hundred PMIDs a GET URL exceeds the server's length
        limit and comes back as "400 Bad Request". POST puts the IDs in the
        request body, so there is no length limit.

        A batch that still fails (e.g. a single malformed or withdrawn PMID) is
        logged and skipped rather than aborting the whole run -- losing one batch
        of 150 is far better than losing the entire fetch.
        """
        results: list[PubMedAbstract] = []
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i : i + batch_size]
            logger.info(
                "Fetching batch %d-%d of %d", i + 1, min(i + batch_size, len(pmids)), len(pmids)
            )
            data = {
                **self._base_params(),
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "xml",
                "rettype": "abstract",
            }
            self._throttle()
            try:
                resp = requests.post(EFETCH_URL, data=data, timeout=60)
                resp.raise_for_status()
                results.extend(self._parse_efetch_xml(resp.text))
            except (requests.HTTPError, requests.RequestException) as exc:
                logger.warning("Batch %d-%d failed (%s) -- skipping it and continuing.",
                               i + 1, min(i + batch_size, len(pmids)), exc)
                continue
        return results

    @staticmethod
    def _parse_efetch_xml(xml_text: str) -> list[PubMedAbstract]:
        root = ET.fromstring(xml_text)
        parsed: list[PubMedAbstract] = []

        for article in root.findall(".//PubmedArticle"):
            pmid_el = article.find(".//PMID")
            pmid = pmid_el.text if pmid_el is not None else ""

            title_el = article.find(".//ArticleTitle")
            title = "".join(title_el.itertext()).strip() if title_el is not None else ""

            # Abstracts can have multiple labeled sections (Background, Methods, ...)
            abstract_parts = []
            for ab_text_el in article.findall(".//Abstract/AbstractText"):
                label = ab_text_el.get("Label")
                text = "".join(ab_text_el.itertext()).strip()
                if not text:
                    continue
                abstract_parts.append(f"{label}: {text}" if label else text)
            abstract = "\n".join(abstract_parts)

            journal_el = article.find(".//Journal/Title")
            journal = journal_el.text if journal_el is not None else ""

            year_el = article.find(".//JournalIssue/PubDate/Year")
            if year_el is None:
                # Some records only have MedlineDate, e.g. "2023 Jan-Feb"
                medline_date_el = article.find(".//JournalIssue/PubDate/MedlineDate")
                pub_year = (medline_date_el.text or "")[:4] if medline_date_el is not None else ""
            else:
                pub_year = year_el.text or ""

            authors = []
            for author_el in article.findall(".//AuthorList/Author"):
                last = author_el.find("LastName")
                fore = author_el.find("ForeName")
                if last is not None and fore is not None:
                    authors.append(f"{fore.text} {last.text}")
                elif last is not None:
                    authors.append(last.text or "")

            if not abstract:
                # Skip records with no abstract text (e.g. editorials, letters)
                continue

            parsed.append(
                PubMedAbstract(
                    pmid=pmid,
                    title=title,
                    abstract=abstract,
                    journal=journal or "",
                    pub_year=pub_year,
                    authors=authors,
                )
            )

        return parsed


def save_raw(abstracts: list[PubMedAbstract], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(a) for a in abstracts], f, ensure_ascii=False, indent=2)
    logger.info("Saved %d abstracts to %s", len(abstracts), out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch PubMed abstracts for the biomed-rag project.")
    parser.add_argument("--email", required=True, help="Your email address (required by NCBI).")
    parser.add_argument("--api-key", default=None, help="Optional free NCBI API key (raises rate limit to 10 req/s).")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="PubMed search query.")
    parser.add_argument("--retmax", type=int, default=1000, help="Max number of PMIDs to retrieve.")
    parser.add_argument(
        "--out",
        default="data/raw/allosteric_kinase_inhibitors.json",
        help="Output path for the raw JSON cache.",
    )
    args = parser.parse_args()

    fetcher = PubMedFetcher(email=args.email, api_key=args.api_key)
    pmids = fetcher.search_pmids(args.query, retmax=args.retmax)
    if not pmids:
        logger.warning("No PMIDs found for query: %s", args.query)
        return

    abstracts = fetcher.fetch_abstracts(pmids)
    save_raw(abstracts, Path(args.out))


if __name__ == "__main__":
    main()
