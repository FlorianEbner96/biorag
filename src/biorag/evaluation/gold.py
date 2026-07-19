"""The gold question set.

This is the most valuable and least glamorous artifact in the project. A RAG
system without an evaluation set is a demo: you ask it three questions you
already know it handles, it looks impressive, and you learn nothing.

**Write these against the real corpus, by hand, before tuning anything.** Twenty
to thirty questions is enough. Writing them after seeing what the system does
well is how you end up with an eval set that measures your own selection bias
instead of the system.

Question types to cover deliberately, because they fail differently:

  - `factual`      — one abstract has the answer. Tests basic retrieval.
  - `identifier`   — hinges on an exact code ("SHP099"). Where BM25 beats dense.
  - `conceptual`   — mechanism questions. Where dense beats BM25.
  - `multi_hop`    — needs two or more abstracts combined.
  - `unanswerable` — deliberately NOT in the corpus. **Do not skip these.**

That last category is the one people leave out, and it is the one that matters
most. A system that answers everything confidently scores perfectly on the other
four types while being useless in a scientific setting. Roughly 20% of the set
should be unanswerable, and the target behaviour is an explicit refusal.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

QuestionType = str  # factual | identifier | conceptual | multi_hop | unanswerable


@dataclass
class GoldQuestion:
    """One evaluation question with known ground truth.

    `relevant_pmids` drives retrieval metrics — it is the set of abstracts that
    SHOULD be retrieved. For unanswerable questions it is empty and
    `expect_refusal` is True.
    """
    question: str
    question_type: QuestionType
    relevant_pmids: list[str] = field(default_factory=list)
    expected_answer: str | None = None
    expect_refusal: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if self.expect_refusal and self.relevant_pmids:
            raise ValueError(
                f"{self.question!r} expects refusal but lists relevant PMIDs. "
                "If relevant abstracts exist, the system should answer, not refuse."
            )
        if not self.expect_refusal and not self.relevant_pmids:
            raise ValueError(
                f"{self.question!r} has no relevant PMIDs and does not expect refusal. "
                "Every answerable question needs ground truth to score against."
            )


def load_gold(path: str | Path) -> list[GoldQuestion]:
    with Path(path).open(encoding="utf-8") as f:
        return [GoldQuestion(**d) for d in json.load(f)]


def save_gold(questions: list[GoldQuestion], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(q) for q in questions], indent=2), encoding="utf-8")


def gold_set_health(questions: list[GoldQuestion]) -> dict:
    """Check the gold set is balanced before trusting any metric computed on it.

    Warns when unanswerable questions are under-represented — the failure mode
    that makes every other number look better than it is.
    """
    from collections import Counter

    counts = Counter(q.question_type for q in questions)
    n = len(questions) or 1
    unanswerable_frac = sum(q.expect_refusal for q in questions) / n

    warnings = []
    if unanswerable_frac < 0.10:
        warnings.append(
            f"Only {unanswerable_frac:.0%} unanswerable questions. Aim for ~20% — "
            "without them you cannot detect a system that answers everything confidently."
        )
    if n < 15:
        warnings.append(f"Only {n} questions; metrics will be noisy. Aim for 20-30.")
    missing = {"factual", "identifier", "conceptual", "unanswerable"} - set(counts)
    if missing:
        warnings.append(f"No questions of type(s): {sorted(missing)}.")

    return {"n": len(questions), "by_type": dict(counts),
            "unanswerable_fraction": unanswerable_frac, "warnings": warnings}


def synthetic_gold_set() -> list[GoldQuestion]:
    """Gold questions matching `data.synthetic.make_corpus(seed=0)`.

    For testing the evaluation machinery itself. NOT a substitute for a
    hand-written set against the real corpus — these questions were written with
    knowledge of how the generator works, which is precisely the bias a real gold
    set exists to avoid.
    """
    return [
        GoldQuestion("What is MK-2206 an inhibitor of?", "factual",
                     relevant_pmids=["30000000"]),
        GoldQuestion("Which compound targets SHP2 allosterically?", "identifier",
                     relevant_pmids=["30000001"]),
        GoldQuestion("How do allosteric inhibitors overcome gatekeeper resistance mutations?",
                     "conceptual", relevant_pmids=["30000003", "30000010", "30000017"]),
        GoldQuestion("What is the reported IC50 for TNO155?", "identifier",
                     relevant_pmids=["30000002"]),
        GoldQuestion("What is the crystal structure of the SARS-CoV-2 spike protein?",
                     "unanswerable", expect_refusal=True,
                     notes="Entirely outside the corpus scope — must refuse."),
        GoldQuestion("Which allosteric inhibitor is approved for paediatric leukaemia?",
                     "unanswerable", expect_refusal=True,
                     notes="Plausible-sounding and in-domain, but not in the corpus. "
                           "Harder refusal than the off-topic case."),
    ]
