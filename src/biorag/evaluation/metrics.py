"""Retrieval and answer metrics.

Retrieval and generation are scored SEPARATELY and in that order, because they
fail independently: if recall@5 is 0.4, no prompt engineering will fix the
answers, since the relevant abstract was never in the context. Debugging a RAG
system by staring at its answers, when the problem is upstream in retrieval, is
the most common time sink in these projects.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    recall_at_k: float          # fraction of relevant abstracts retrieved
    precision_at_k: float
    mrr: float                  # mean reciprocal rank of the first relevant hit
    hit_rate: float             # fraction of questions with >=1 relevant hit
    k: int
    n_questions: int

    def summary(self) -> str:
        return (f"  recall@{self.k}     {self.recall_at_k:.3f}\n"
                f"  precision@{self.k}  {self.precision_at_k:.3f}\n"
                f"  MRR            {self.mrr:.3f}\n"
                f"  hit rate       {self.hit_rate:.3f}   (n={self.n_questions})")


@dataclass
class AnswerMetrics:
    citation_rate: float            # sentences carrying a citation
    grounded_rate: float            # answers citing only retrieved PMIDs
    refusal_precision: float        # of refusals, how many SHOULD have refused
    refusal_recall: float           # of should-refuse, how many did
    n_answers: int

    def summary(self) -> str:
        return (f"  citation rate      {self.citation_rate:.3f}\n"
                f"  grounded rate      {self.grounded_rate:.3f}\n"
                f"  refusal precision  {self.refusal_precision:.3f}\n"
                f"  refusal recall     {self.refusal_recall:.3f}   (n={self.n_answers})")


def recall_at_k(retrieved_pmids: list[str], relevant_pmids: list[str], k: int) -> float:
    if not relevant_pmids:
        return 1.0                      # vacuously satisfied for unanswerable questions
    top = set(retrieved_pmids[:k])
    return len(top & set(relevant_pmids)) / len(set(relevant_pmids))


def precision_at_k(retrieved_pmids: list[str], relevant_pmids: list[str], k: int) -> float:
    if not relevant_pmids or k == 0:
        return 0.0
    top = retrieved_pmids[:k]
    return len(set(top) & set(relevant_pmids)) / len(top) if top else 0.0


def reciprocal_rank(retrieved_pmids: list[str], relevant_pmids: list[str]) -> float:
    """1/rank of the first relevant hit; 0 if none.

    Complements recall: recall asks whether the relevant abstract was retrieved,
    MRR asks whether it was retrieved NEAR THE TOP. With a top_k of 5 feeding a
    prompt, position matters — models weight earlier context more heavily.
    """
    relevant = set(relevant_pmids)
    for i, pmid in enumerate(retrieved_pmids, start=1):
        if pmid in relevant:
            return 1.0 / i
    return 0.0


def evaluate_retrieval(retriever, gold_questions, k: int = 5) -> RetrievalMetrics:
    """Score a retriever against the gold set. Unanswerable questions excluded.

    They are excluded because retrieval cannot be right or wrong when nothing
    relevant exists — including them would inflate recall with free 1.0s. They
    are scored in `evaluate_answers` instead, where refusal is the thing measured.
    """
    scored = [q for q in gold_questions if q.relevant_pmids]
    if not scored:
        return RetrievalMetrics(0.0, 0.0, 0.0, 0.0, k, 0)

    recalls, precisions, rrs, hits = [], [], [], []
    for q in scored:
        retrieved = list(dict.fromkeys(h.chunk.pmid for h in retriever.retrieve(q.question, k * 3)))
        recalls.append(recall_at_k(retrieved, q.relevant_pmids, k))
        precisions.append(precision_at_k(retrieved, q.relevant_pmids, k))
        rrs.append(reciprocal_rank(retrieved[:k], q.relevant_pmids))
        hits.append(bool(set(retrieved[:k]) & set(q.relevant_pmids)))

    n = len(scored)
    return RetrievalMetrics(
        recall_at_k=sum(recalls) / n, precision_at_k=sum(precisions) / n,
        mrr=sum(rrs) / n, hit_rate=sum(hits) / n, k=k, n_questions=n,
    )


def evaluate_answers(answers, gold_questions) -> AnswerMetrics:
    """Score generated answers. `answers` must align with `gold_questions`."""
    if len(answers) != len(gold_questions):
        raise ValueError("answers and gold_questions must align")
    if not answers:
        return AnswerMetrics(0.0, 0.0, 0.0, 0.0, 0)

    citation_rates = [a.citation_rate for a in answers if not a.insufficient_context]
    grounded = [a.is_grounded for a in answers]

    refused = [a.insufficient_context for a in answers]
    should_refuse = [q.expect_refusal for q in gold_questions]
    true_pos = sum(r and s for r, s in zip(refused, should_refuse))

    n_refused = sum(refused)
    n_should = sum(should_refuse)

    return AnswerMetrics(
        citation_rate=sum(citation_rates) / len(citation_rates) if citation_rates else 0.0,
        grounded_rate=sum(grounded) / len(grounded),
        refusal_precision=true_pos / n_refused if n_refused else 1.0,
        refusal_recall=true_pos / n_should if n_should else 1.0,
        n_answers=len(answers),
    )


def compare_retrievers(retrievers: dict, gold_questions, k: int = 5) -> str:
    """Head-to-head table. The quantitative results section.

    "RAG beats keyword search" is an assertion until this table exists — and
    occasionally the table disagrees, which is itself worth reporting honestly.
    """
    lines = [f"Retrieval comparison (k={k}, n={sum(1 for q in gold_questions if q.relevant_pmids)})",
             "=" * 62,
             f"{'retriever':<14}{'recall':>10}{'precision':>12}{'MRR':>10}{'hit rate':>12}",
             "-" * 62]
    for name, retriever in retrievers.items():
        m = evaluate_retrieval(retriever, gold_questions, k)
        lines.append(f"{name:<14}{m.recall_at_k:>10.3f}{m.precision_at_k:>12.3f}"
                     f"{m.mrr:>10.3f}{m.hit_rate:>12.3f}")
    return "\n".join(lines)
