from __future__ import annotations

from pathlib import Path
from unittest import TestCase

from src.evaluation import (
    EvaluationCase,
    JudgeResult,
    evaluate_case_deterministically,
    load_evaluation_cases,
    run_evaluation,
    summarize_evaluation_results,
)
from src.qa import QAEngineResult
from src.schemas import AnalyticsResult, GroundedAnswer, QueryPlan, RetrievedEvidence


def make_result(
    *,
    question: str,
    intent: str = "segment_summary",
    branch: str | None = "Disneyland_HongKong",
    reviewer_location: str | None = "Australia",
    comparison_dimension: str | None = None,
    compare_values: list[str] | None = None,
    external_context_requested: list[str] | None = None,
    analytics_route: str = "analyze_segment",
    evidence: list[RetrievedEvidence] | None = None,
    answer: str = "Answer text.\n\nLimitations: historical dataset only.",
    limitations: list[str] | None = None,
    supporting_metrics: list[str] | None = None,
    error: str | None = None,
) -> QAEngineResult:
    plan = QueryPlan(
        intent=intent,
        branch=branch,
        reviewer_location=reviewer_location,
        comparison_dimension=comparison_dimension,
        compare_values=compare_values or [],
        external_context_requested=external_context_requested or [],
    )
    analytics_result = AnalyticsResult(
        intent=intent,
        selected_analytics_function=analytics_route,
        applied_filters={},
        sample_size=100,
        metrics={
            "review_count": 100,
            "average_rating": 4.2,
            "positive_share": 80.0,
            "negative_share": 5.0,
            "aspect_mention_rate": 30.0,
        },
        baseline_metrics={"average_rating": 4.1},
        dataset_date_coverage={"start": "2010-03", "end": "2019-05", "rows_with_missing_review_date": 0},
        warnings=[],
    )
    evidence = evidence or [
        RetrievedEvidence(
            review_id="1",
            text="Helpful staff and fun rides.",
            branch=branch or "Disneyland_HongKong",
            reviewer_location=reviewer_location or "Australia",
            rating=5,
            rating_sentiment="positive",
            year=2018,
            month=5,
            season="Spring",
            has_date=True,
        )
    ]
    structured_answer = GroundedAnswer(
        direct_answer="Direct answer.",
        supporting_metrics=supporting_metrics or ["Sample size: 100; Average rating: 4.2; Positive share: 80.0%"],
        evidence=["Helpful staff and fun rides."],
        limitations=limitations or ["Historical dataset only."],
    )
    return QAEngineResult(
        question=question,
        answer=answer,
        plan=plan,
        analytics_result=analytics_result,
        evidence=evidence,
        error=error,
        structured_answer=structured_answer,
    )


class FakeEngine:
    def __init__(self, mapping):
        self.mapping = mapping
        self.settings = None

    def ask(self, question: str):
        return self.mapping[question]


class EvaluationTests(TestCase):
    def test_load_evaluation_cases(self):
        cases = load_evaluation_cases(Path("evals/evaluation_cases.json"))
        self.assertGreaterEqual(len(cases), 10)
        self.assertEqual(cases[0].id, "segment_australia_hong_kong")

    def test_deterministic_evaluation_detects_incorrect_intent(self):
        case = EvaluationCase(
            id="case",
            category="segment",
            question="q",
            expected_intent="segment_summary",
            expected_branch="Disneyland_HongKong",
            expected_reviewer_location="Australia",
        )
        result = make_result(question="q", intent="general_search")
        evaluation = evaluate_case_deterministically(case, result)
        self.assertFalse(evaluation["intent_correct"])

    def test_retrieval_filter_violation_fails(self):
        case = EvaluationCase(
            id="case",
            category="segment",
            question="q",
            expected_intent="segment_summary",
            expected_branch="Disneyland_HongKong",
            expected_reviewer_location="Australia",
        )
        bad_evidence = [
            RetrievedEvidence(
                review_id="1",
                text="Wrong branch.",
                branch="Disneyland_Paris",
                reviewer_location="Australia",
                rating=4,
                rating_sentiment="positive",
                year=2018,
                month=5,
                season="Spring",
                has_date=True,
            )
        ]
        result = make_result(question="q", evidence=bad_evidence)
        evaluation = evaluate_case_deterministically(case, result)
        self.assertFalse(evaluation["retrieval_filters_valid"])

    def test_external_context_case_requires_warning(self):
        case = EvaluationCase(
            id="case",
            category="unsupported",
            question="q",
            expected_intent="time_comparison",
            expected_branch="Disneyland_HongKong",
            expected_comparison_dimension="month",
            must_flag_external_context=True,
        )
        result = make_result(
            question="q",
            intent="time_comparison",
            branch="Disneyland_HongKong",
            reviewer_location=None,
            comparison_dimension="month",
            external_context_requested=["weather"],
            answer="Dataset does not include weather. Limitations: historical dataset only.",
        )
        evaluation = evaluate_case_deterministically(case, result)
        self.assertTrue(evaluation["external_context_behavior_correct"])

    def test_empty_expected_compare_values_does_not_fail_branch_comparison(self):
        case = EvaluationCase(
            id="case",
            category="branch",
            question="q",
            expected_intent="branch_comparison",
            expected_branch=None,
            expected_comparison_dimension="branch",
            expected_compare_values=[],
        )
        result = make_result(
            question="q",
            intent="branch_comparison",
            branch=None,
            reviewer_location=None,
            comparison_dimension="branch",
            compare_values=[
                "Disneyland_California",
                "Disneyland_HongKong",
                "Disneyland_Paris",
            ],
            analytics_route="analyze_branch_comparison",
        )
        evaluation = evaluate_case_deterministically(case, result)
        self.assertTrue(evaluation["filters_correct"])

    def test_numeric_supporting_metrics_validation_tolerates_rounding(self):
        case = EvaluationCase(
            id="case",
            category="segment",
            question="q",
            expected_intent="segment_summary",
            expected_branch="Disneyland_HongKong",
            expected_reviewer_location="Australia",
        )
        result = make_result(
            question="q",
            supporting_metrics=[
                "Sample size: 100; Average rating: 4.20; Positive share: 80.00%",
            ],
        )
        evaluation = evaluate_case_deterministically(case, result)
        self.assertTrue(evaluation["numeric_supporting_metrics_valid"])

    def test_judge_output_is_parsed_correctly(self):
        case = EvaluationCase(
            id="case",
            category="segment",
            question="q",
            expected_intent="segment_summary",
        )
        result = make_result(question="q")
        engine = FakeEngine({"q": result})

        def judge(_payload):
            return {
                "relevance": 5,
                "faithfulness": 4,
                "completeness": 4,
                "clarity": 5,
                "limitation_handling": 4,
                "external_knowledge_leak": False,
                "unsupported_claims": [],
                "explanation": "Solid answer.",
            }

        df = run_evaluation(engine, [case], use_llm_judge=True, judge_evaluator=judge)
        self.assertEqual(df.loc[0, "judge_relevance"], 5)
        self.assertEqual(df.loc[0, "final_score"], 4.4)

    def test_judge_api_failure_returns_clear_error(self):
        case = EvaluationCase(
            id="case",
            category="segment",
            question="q",
            expected_intent="segment_summary",
        )
        result = make_result(question="q")
        engine = FakeEngine({"q": result})

        def failing_judge(_payload):
            raise RuntimeError("judge down")

        df = run_evaluation(engine, [case], use_llm_judge=True, judge_evaluator=failing_judge)
        self.assertIn("Judge error:", df.loc[0, "failure_reason"])

    def test_evaluation_works_without_llm_judge(self):
        case = EvaluationCase(
            id="case",
            category="segment",
            question="q",
            expected_intent="segment_summary",
            expected_branch="Disneyland_HongKong",
            expected_reviewer_location="Australia",
        )
        result = make_result(question="q")
        engine = FakeEngine({"q": result})
        df = run_evaluation(engine, [case], use_llm_judge=False)
        summary = summarize_evaluation_results(df)
        self.assertEqual(len(df), 1)
        self.assertIsNone(df.loc[0, "judge_relevance"])
        self.assertEqual(summary["num_questions"], 1)
