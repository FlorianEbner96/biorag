"""Synthetic PubMed-shaped records for offline testing and demos.

Every layer of this system must be testable without hitting NCBI and without
downloading a model. Tests that need the network are tests that get skipped, and
tests that get skipped stop protecting anything.

The generator deliberately plants the exact pathologies the cleaner exists to
handle — copyright boilerplate, retraction notices, duplicate titles, one-line
abstracts — so `test_clean_corpus.py` asserts against known ground truth rather
than whatever the real corpus happens to contain today.
"""
from __future__ import annotations

import random

from ..schema import PubMedAbstract

TARGETS = ["PI3Kα", "SHP2", "AKT1", "MEK1", "ABL1", "TRPA1", "BRAF", "JAK2"]
COMPOUNDS = ["MK-2206", "SHP099", "TNO155", "GDC-0994", "asciminib", "alpelisib"]
JOURNALS = ["J Med Chem", "Nat Chem Biol", "ACS Med Chem Lett", "Bioorg Med Chem"]

# Title stems. Length 7 is co-prime with the target (8) and compound (6)
# cycles, so LCM(8, 6, 7) = 168 distinct titles — comfortably more than any
# test corpus, meaning duplicate titles only appear where deliberately planted.
DESCRIPTORS = [
    "{compound} is a selective allosteric inhibitor of {target}",
    "Structural basis for conformational blockade of {target}",
    "Kinetic characterisation of non-ATP-competitive {target} binding by {compound}",
    "Overcoming gatekeeper resistance mutations with {compound}",
    "Fragment-based discovery of a cryptic pocket in {target}",
    "Cellular target engagement of {compound} and downstream signalling",
    "Pharmacokinetic profiling of {compound} across species",
]

_BODY = (
    "Allosteric inhibition of {target} offers a route to selectivity that "
    "ATP-competitive agents cannot achieve, because the allosteric pocket is "
    "less conserved across the kinome. We report that {compound} binds a pocket "
    "distal to the active site and stabilises the autoinhibited conformation. "
    "Biochemical assays gave an IC50 of {ic50} nM against {target}, with greater "
    "than 100-fold selectivity over closely related kinases. Cellular assays "
    "confirmed target engagement and showed reduced downstream signalling. "
    "Crucially, {compound} retained potency against the gatekeeper mutant that "
    "confers resistance to ATP-site inhibitors, supporting the hypothesis that "
    "allosteric mechanisms circumvent common resistance pathways."
)


def make_abstract(
    pmid: str,
    *,
    descriptor: str = "{compound} is a selective allosteric inhibitor of {target}",
    target: str = "PI3Kα",
    compound: str = "MK-2206",
    ic50: int = 12,
    year: str = "2023",
    structured: bool = False,
    boilerplate: bool = False,
    retracted: bool = False,
    short: bool = False,
    title: str | None = None,
) -> PubMedAbstract:
    """Build one synthetic record with controllable pathologies."""
    body = _BODY.format(target=target, compound=compound, ic50=ic50)

    if short:
        body = "A brief commentary."
    elif structured:
        sentences = [s.strip() for s in body.split(". ") if s.strip()]
        # Split proportionally so all four sections are guaranteed non-empty,
        # regardless of how many sentences the template happens to produce.
        n = len(sentences)
        cuts = [0, max(1, n // 4), max(2, n // 2), max(3, 3 * n // 4), n]
        labels = ["BACKGROUND", "METHODS", "RESULTS", "CONCLUSIONS"]
        parts = []
        for label, lo, hi in zip(labels, cuts[:-1], cuts[1:]):
            text = ". ".join(sentences[lo:hi]).strip()
            if not text.endswith("."):
                text += "."
            parts.append(f"{label}: {text}")
        body = "\n".join(parts)

    if boilerplate:
        body += "\nCopyright © 2023 Elsevier Ltd. All rights reserved."

    resolved_title = title or descriptor.format(target=target, compound=compound)
    if retracted:
        resolved_title = f"RETRACTED ARTICLE: {resolved_title}"

    return PubMedAbstract(
        pmid=pmid,
        title=resolved_title,
        abstract=body,
        journal=random.choice(JOURNALS),
        pub_year=year,
        authors=["Jane Doe", "Richard Roe"],
    )


def make_corpus(n: int = 40, *, seed: int = 0, with_pathologies: bool = True) -> list[PubMedAbstract]:
    """Generate a corpus. With `with_pathologies`, plants known bad records.

    Planted when `with_pathologies=True` and n >= 10:
      1 retracted · 1 too-short · 1 exact-duplicate title · boilerplate on ~1/3.
    """
    rng = random.Random(seed)
    random.seed(seed)
    records: list[PubMedAbstract] = []

    for i in range(n):
        records.append(make_abstract(
            pmid=str(30000000 + i),
            descriptor=DESCRIPTORS[i % len(DESCRIPTORS)],
            target=TARGETS[i % len(TARGETS)],
            compound=COMPOUNDS[i % len(COMPOUNDS)],
            ic50=rng.choice([3, 12, 47, 130, 890]),
            year=str(rng.choice([2019, 2020, 2021, 2022, 2023, 2024])),
            structured=(i % 4 == 0),
            boilerplate=(i % 3 == 0),
        ))

    if with_pathologies and n >= 10:
        records.append(make_abstract(str(30009001), retracted=True))
        records.append(make_abstract(str(30009002), short=True))
        # Near-duplicate of record 0 — same title, different PMID.
        records.append(make_abstract(str(30009003), title=records[0].title))

    return records
