from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import TestCase

import pandas as pd
from langchain_core.embeddings import Embeddings

from src.analytics import run_analytics
from src.data import load_clean_reviews
from src.planner import AdaptivePlanner
from src.retrieval import execute_multi_retrieval, resolve_retrieval_tasks
from src.vectorstore import build_or_load_vectorstore


class DummyEmbeddings(Embeddings):
    def __init__(self) -> None:
        self.tokens = [
            "crowd",
            "queue",
            "wait",
            "staff",
            "friendly",
            "food",
            "price",
            "ride",
            "family",
            "paris",
            "hong",
            "california",
            "australia",
            "summer",
            "winter",
            "september",
            "november",
            "may",
            "march",
        ]

    def _embed(self, text: str) -> list[float]:
        lowered = text.lower()
        return [float(lowered.count(token)) for token in self.tokens]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


REPO_ROOT = Path(__file__).resolve().parents[1]


class RetrievalTests(TestCase):
    @classmethod
    def setUpClass(cls):
        full_df = load_clean_reviews(REPO_ROOT / "DisneylandReviews.csv")
        subset_parts = [
            full_df.loc[
                (full_df["Branch"] == "Disneyland_HongKong")
                & (full_df["Reviewer_Location"] == "Australia")
            ].head(80),
            full_df.loc[
                (full_df["Branch"] == "Disneyland_HongKong")
                & (full_df["Month"].isin([3, 5]))
            ].head(260),
            full_df.loc[
                (full_df["Branch"] == "Disneyland_California")
                & (full_df["Month"].isin([9, 11]))
            ].head(260),
            full_df.loc[
                (full_df["Branch"] == "Disneyland_Paris")
                & (full_df["Season"].isin(["Summer", "Winter"]))
            ].head(260),
            full_df.head(80),
        ]
        cls.test_df = pd.concat(subset_parts, ignore_index=True).drop_duplicates(
            subset=["Review_ID"]
        ).reset_index(drop=True)
        cls.planner = AdaptivePlanner(use_llm=False)
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.vectorstore_result = build_or_load_vectorstore(
            cls.test_df,
            persist_directory=Path(cls.temp_dir.name) / "chroma",
            embeddings=DummyEmbeddings(),
            force_rebuild=True,
            verbose=False,
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_time_comparison_tasks_are_resolved_to_candidate_months(self):
        plan_result = self.planner.plan(
            "What is the best time of the year to visit Disneyland Hong Kong?",
            self.test_df,
        )
        analytics_result = run_analytics(plan_result.normalized_plan, self.test_df)
        resolved_tasks = resolve_retrieval_tasks(
            "What is the best time of the year to visit Disneyland Hong Kong?",
            plan_result.normalized_plan,
            analytics_result,
        )

        self.assertTrue(resolved_tasks)
        self.assertTrue(all(task.branch == "Disneyland_HongKong" for task in resolved_tasks))
        self.assertIsNotNone(resolved_tasks[0].month)

    def test_metadata_filtered_retrieval_obeys_branch_and_location_filters(self):
        question = "What do Australians say about Hong Kong Disneyland?"
        plan_result = self.planner.plan(question, self.test_df)
        analytics_result = run_analytics(plan_result.normalized_plan, self.test_df)
        retrieval_result = execute_multi_retrieval(
            question,
            plan_result.normalized_plan,
            analytics_result,
            self.test_df,
            self.vectorstore_result.vectorstore,
        )

        self.assertTrue(retrieval_result.final_evidence)
        self.assertEqual(retrieval_result.validation.invalid_count, 0)
        for evidence in retrieval_result.final_evidence:
            self.assertEqual(evidence.branch, "Disneyland_HongKong")
            self.assertEqual(evidence.reviewer_location, "Australia")

    def test_crowding_retrieval_stays_inside_candidate_month(self):
        question = "Which month has the fewest crowding complaints in California?"
        plan_result = self.planner.plan(question, self.test_df)
        analytics_result = run_analytics(plan_result.normalized_plan, self.test_df)
        retrieval_result = execute_multi_retrieval(
            question,
            plan_result.normalized_plan,
            analytics_result,
            self.test_df,
            self.vectorstore_result.vectorstore,
        )

        candidate_months = {
            candidate.month
            for candidate in analytics_result.selected_candidates
            if candidate.month is not None
        }
        self.assertTrue(retrieval_result.final_evidence)
        self.assertEqual(retrieval_result.validation.invalid_count, 0)
        self.assertTrue(candidate_months)
        self.assertTrue(
            all(evidence.branch == "Disneyland_California" for evidence in retrieval_result.final_evidence)
        )
        self.assertTrue(
            all(evidence.month in candidate_months for evidence in retrieval_result.final_evidence)
        )
