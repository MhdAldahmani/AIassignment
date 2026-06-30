from __future__ import annotations

from pathlib import Path
from unittest import TestCase

from src.analytics import run_analytics
from src.config import get_settings
from src.data import load_clean_reviews
from src.planner import AdaptivePlanner


REPO_ROOT = Path(__file__).resolve().parents[1]


class AnalyticsTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clean_reviews_df = load_clean_reviews(REPO_ROOT / "DisneylandReviews.csv")
        cls.planner = AdaptivePlanner(use_llm=False)

    def test_best_time_analytics_returns_ranked_month_rows(self):
        plan_result = self.planner.plan(
            "What is the best time of the year to visit Disneyland Hong Kong?",
            self.clean_reviews_df,
        )
        analytics_result = run_analytics(plan_result.normalized_plan, self.clean_reviews_df)

        self.assertEqual(analytics_result.selected_analytics_function, "analyze_time_comparison")
        self.assertEqual(analytics_result.ranking_criterion, "visit_score")
        self.assertIsNotNone(analytics_result.score_definition)
        self.assertTrue(analytics_result.selected_candidates)
        self.assertTrue(analytics_result.comparison_rows)
        self.assertTrue(any(row.eligible for row in analytics_result.comparison_rows))
        for row in analytics_result.comparison_rows:
            threshold = get_settings().min_month_reviews
            self.assertEqual(row.eligible, row.review_count >= threshold)

    def test_paris_summer_vs_winter_keeps_requested_seasons_only(self):
        plan_result = self.planner.plan(
            "How does Paris perform in summer compared with winter?",
            self.clean_reviews_df,
        )
        analytics_result = run_analytics(plan_result.normalized_plan, self.clean_reviews_df)

        groups = [row.group_key for row in analytics_result.comparison_rows]
        self.assertEqual(analytics_result.selected_analytics_function, "analyze_time_comparison")
        self.assertEqual(groups, ["Summer", "Winter"])
        self.assertEqual(analytics_result.selected_candidates, [])

    def test_branch_comparison_returns_all_three_parks(self):
        plan_result = self.planner.plan(
            "Which park has the strongest customer satisfaction?",
            self.clean_reviews_df,
        )
        analytics_result = run_analytics(plan_result.normalized_plan, self.clean_reviews_df)

        branches = [row.branch for row in analytics_result.comparison_rows]
        self.assertEqual(analytics_result.selected_analytics_function, "analyze_branch_comparison")
        self.assertEqual(
            set(branches),
            {
                "Disneyland_California",
                "Disneyland_Paris",
                "Disneyland_HongKong",
            },
        )
        self.assertTrue(analytics_result.selected_candidates)

    def test_crowding_question_ranks_by_crowding_rate(self):
        plan_result = self.planner.plan(
            "Which month has the fewest crowding complaints in California?",
            self.clean_reviews_df,
        )
        analytics_result = run_analytics(plan_result.normalized_plan, self.clean_reviews_df)

        self.assertEqual(analytics_result.selected_analytics_function, "analyze_time_comparison")
        self.assertEqual(analytics_result.ranking_criterion, "crowding_complaint_rate")
        self.assertTrue(analytics_result.selected_candidates)
        self.assertTrue(
            any(row.crowding_complaint_rate is not None for row in analytics_result.comparison_rows)
        )
