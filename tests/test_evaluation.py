"""Evaluation: gold-set integrity and metric correctness."""
from __future__ import annotations

import pytest

from biorag.evaluation.gold import GoldQuestion, gold_set_health, synthetic_gold_set
from biorag.evaluation.metrics import (
    compare_retrievers, evaluate_answers, evaluate_retrieval,
    precision_at_k, recall_at_k, reciprocal_rank,
)
from biorag.generation.answerer import Answerer
from biorag.retrieval.retriever import BM25Retriever, DenseRetriever, HybridRetriever
from conftest import FakeLLM


class TestGoldQuestionValidation:
    def test_refusal_question_cannot_have_relevant_pmids(self):
        with pytest.raises(ValueError):
            GoldQuestion("Q?", "unanswerable", relevant_pmids=["123"], expect_refusal=True)

    def test_answerable_question_needs_ground_truth(self):
        with pytest.raises(ValueError):
            GoldQuestion("Q?", "factual")

    def test_valid_questions_construct(self):
        assert GoldQuestion("Q?", "factual", relevant_pmids=["30000001"])
        assert GoldQuestion("Q?", "unanswerable", expect_refusal=True)


class TestGoldSetHealth:
    def test_warns_when_no_unanswerable_questions(self):
        """The omission that makes every other metric look better than it is."""
        qs = [GoldQuestion(f"Q{i}?", "factual", relevant_pmids=["1"]) for i in range(20)]
        warnings = gold_set_health(qs)["warnings"]
        assert any("unanswerable" in w for w in warnings)

    def test_warns_on_small_set(self):
        assert any("noisy" in w for w in gold_set_health(synthetic_gold_set())["warnings"])

    def test_reports_type_distribution(self):
        health = gold_set_health(synthetic_gold_set())
        assert health["by_type"]["unanswerable"] == 2
        assert health["unanswerable_fraction"] > 0.3


class TestRetrievalMetrics:
    def test_perfect_recall(self):
        assert recall_at_k(["1", "2", "3"], ["1", "2"], k=3) == 1.0

    def test_partial_recall(self):
        assert recall_at_k(["1", "9"], ["1", "2"], k=2) == 0.5

    def test_recall_respects_k(self):
        """Relevant doc at position 3 must not count for recall@2."""
        assert recall_at_k(["9", "8", "1"], ["1"], k=2) == 0.0

    def test_precision(self):
        assert precision_at_k(["1", "9"], ["1"], k=2) == 0.5

    def test_reciprocal_rank_first_position(self):
        assert reciprocal_rank(["1", "9"], ["1"]) == 1.0

    def test_reciprocal_rank_third_position(self):
        assert reciprocal_rank(["9", "8", "1"], ["1"]) == pytest.approx(1 / 3)

    def test_reciprocal_rank_no_hit(self):
        assert reciprocal_rank(["9", "8"], ["1"]) == 0.0

    def test_unanswerable_excluded_from_retrieval_scoring(self, store, fake_embedder):
        """Including them would inflate recall with free 1.0s."""
        m = evaluate_retrieval(DenseRetriever(store, fake_embedder), synthetic_gold_set(), k=5)
        assert m.n_questions == 4          # 6 total, 2 unanswerable


class TestAnswerMetrics:
    def test_refusal_recall_perfect_when_all_refused(self):
        gold = [GoldQuestion("Q?", "unanswerable", expect_refusal=True)]
        answers = [Answerer(FakeLLM(mode="refuse")).answer("Q?", [])]
        assert evaluate_answers(answers, gold).refusal_recall == 1.0

    def test_grounded_rate_penalises_hallucination(self, store, fake_embedder):
        hits = DenseRetriever(store, fake_embedder).retrieve("kinase", k=3)
        gold = [GoldQuestion("Q?", "factual", relevant_pmids=["30000000"])]
        answers = [Answerer(FakeLLM(mode="hallucinate")).answer("Q?", hits)]
        assert evaluate_answers(answers, gold).grounded_rate == 0.0

    def test_misaligned_inputs_rejected(self):
        with pytest.raises(ValueError):
            evaluate_answers([], [GoldQuestion("Q?", "factual", relevant_pmids=["1"])])


class TestComparison:
    def test_comparison_table_renders_all_retrievers(self, store, chunks, fake_embedder):
        dense = DenseRetriever(store, fake_embedder)
        bm25 = BM25Retriever(chunks)
        table = compare_retrievers(
            {"dense": dense, "bm25": bm25, "hybrid": HybridRetriever([dense, bm25])},
            synthetic_gold_set(), k=5,
        )
        assert all(name in table for name in ("dense", "bm25", "hybrid"))
        assert "recall" in table
