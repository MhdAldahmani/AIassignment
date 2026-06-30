from __future__ import annotations

from pathlib import Path
from unittest import TestCase

from src.data import load_clean_reviews
from src.planner import AdaptivePlanner, normalize_and_validate_plan
from src.schemas import QueryPlan


REPO_ROOT = Path(__file__).resolve().parents[1]


class PlannerTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clean_reviews_df = load_clean_reviews(REPO_ROOT / "DisneylandReviews.csv")
        cls.planner = AdaptivePlanner(use_llm=False)

    def test_best_time_question_becomes_time_comparison(self):
        result = self.planner.plan(
            "What is the best time of the year to visit Disneyland Hong Kong?",
            self.clean_reviews_df,
        )

        plan = result.normalized_plan
        self.assertEqual(plan.intent, "time_comparison")
        self.assertEqual(plan.branch, "Disneyland_HongKong")
        self.assertEqual(plan.comparison_dimension, "month")
        self.assertEqual(plan.ranking_direction, "best")
        self.assertIn("average_rating", plan.required_metrics)
        self.assertIn("crowding_complaint_rate", plan.required_metrics)

    def test_segment_question_becomes_segment_summary(self):
        result = self.planner.plan(
            "What do visitors from Australia say about Hong Kong Disneyland?",
            self.clean_reviews_df,
        )

        plan = result.normalized_plan
        self.assertEqual(plan.intent, "segment_summary")
        self.assertEqual(plan.branch, "Disneyland_HongKong")
        self.assertEqual(plan.reviewer_location, "Australia")

    def test_paris_summer_vs_winter_preserves_both_seasons(self):
        result = self.planner.plan(
            "How does Paris perform in summer compared with winter?",
            self.clean_reviews_df,
        )

        plan = result.normalized_plan
        self.assertEqual(plan.intent, "time_comparison")
        self.assertEqual(plan.branch, "Disneyland_Paris")
        self.assertEqual(plan.comparison_dimension, "season")
        self.assertEqual(plan.compare_values, ["Summer", "Winter"])
        self.assertIsNone(plan.season)

    def test_strongest_park_satisfaction_becomes_branch_comparison(self):
        result = self.planner.plan(
            "Which park has the strongest customer satisfaction?",
            self.clean_reviews_df,
        )

        plan = result.normalized_plan
        self.assertEqual(plan.intent, "branch_comparison")
        self.assertEqual(plan.comparison_dimension, "branch")
        self.assertIsNone(plan.branch)
        self.assertEqual(plan.ranking_direction, "best")

    def test_fewest_crowding_complaints_becomes_time_comparison(self):
        result = self.planner.plan(
            "Which month has the fewest crowding complaints in California?",
            self.clean_reviews_df,
        )

        plan = result.normalized_plan
        self.assertEqual(plan.intent, "time_comparison")
        self.assertEqual(plan.branch, "Disneyland_California")
        self.assertEqual(plan.comparison_dimension, "month")
        self.assertEqual(plan.ranking_direction, "lowest")
        self.assertIn("crowding", plan.requested_aspects)
        self.assertIn("crowding_complaint_rate", plan.required_metrics)

    def test_external_context_request_is_flagged(self):
        result = self.planner.plan(
            "Considering Hong Kong weather and public holidays, what is the best month to visit?",
            self.clean_reviews_df,
        )

        plan = result.normalized_plan
        self.assertEqual(plan.intent, "time_comparison")
        self.assertEqual(plan.branch, "Disneyland_HongKong")
        self.assertGreaterEqual(set(plan.external_context_requested), {"weather", "holidays"})

    def test_generic_best_time_normalization_defaults_to_month(self):
        raw_plan = QueryPlan(
            intent="time_comparison",
            branch="Disneyland_HongKong",
            comparison_dimension="season",
            ranking_direction="best",
            required_metrics=["review_count", "average_rating"],
        )
        heuristic_plan = self.planner.fallback.plan(
            "What is the best time of year to visit Hong Kong Disneyland?",
            self.clean_reviews_df,
        )
        normalized_plan, _ = normalize_and_validate_plan(
            raw_plan,
            self.clean_reviews_df,
            heuristic_plan=heuristic_plan,
        )
        self.assertEqual(normalized_plan.comparison_dimension, "month")
