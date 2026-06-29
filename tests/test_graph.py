from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import TestCase

import pandas as pd
from langchain_core.embeddings import Embeddings

from src.prompts import GROUNDING_SYSTEM_PROMPT
from src.qa import build_qa_engine
from src.schemas import GroundedAnswer
from src.vectorstore import build_or_load_vectorstore
from src.data import load_clean_reviews


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


def successful_answer_generator(_payload):
    return GroundedAnswer(
        direct_answer="Stub grounded answer for testing.",
        supporting_metrics=["sample size mentioned"],
        evidence=["stub evidence"],
        limitations=["stub limitation"],
    )


def failing_answer_generator(_payload):
    raise RuntimeError("mocked answer failure")


REPO_ROOT = Path(__file__).resolve().parents[1]


class GraphTests(TestCase):
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
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.vectorstore = build_or_load_vectorstore(
            cls.test_df,
            persist_directory=Path(cls.temp_dir.name) / "chroma",
            embeddings=DummyEmbeddings(),
            force_rebuild=True,
            verbose=False,
        ).vectorstore

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def build_engine(self, answer_generator=successful_answer_generator):
        return build_qa_engine(
            review_df=self.test_df,
            vectorstore=self.vectorstore,
            answer_generator=answer_generator,
            use_planner_llm=False,
        )

    def test_time_comparison_routes_to_time_analytics(self):
        engine = self.build_engine()
        result = engine.ask("What is the best time of year to visit Hong Kong Disneyland?")
        self.assertIsNone(result.error)
        self.assertEqual(result.plan.intent, "time_comparison")
        self.assertEqual(result.analytics_result.selected_analytics_function, "analyze_time_comparison")

    def test_summer_and_winter_are_both_preserved(self):
        engine = self.build_engine()
        result = engine.ask("How does Paris perform in summer compared with winter?")
        self.assertIsNone(result.error)
        self.assertEqual(result.plan.compare_values, ["Summer", "Winter"])

    def test_branch_comparison_routes_correctly(self):
        engine = self.build_engine()
        result = engine.ask("Which park has the strongest customer satisfaction?")
        self.assertIsNone(result.error)
        self.assertEqual(result.plan.intent, "branch_comparison")
        self.assertEqual(result.analytics_result.selected_analytics_function, "analyze_branch_comparison")

    def test_retrieved_evidence_respects_metadata_filters(self):
        engine = self.build_engine()
        result = engine.ask("What do visitors from Australia say about Hong Kong Disneyland?")
        self.assertIsNone(result.error)
        self.assertTrue(result.evidence)
        for evidence in result.evidence:
            self.assertEqual(evidence.branch, "Disneyland_HongKong")
            self.assertEqual(evidence.reviewer_location, "Australia")

    def test_grounding_prompt_contains_outside_knowledge_restriction(self):
        self.assertIn("Do not use outside knowledge about:", GROUNDING_SYSTEM_PROMPT)
        self.assertIn("Weather", GROUNDING_SYSTEM_PROMPT)
        self.assertIn("Ticket prices", GROUNDING_SYSTEM_PROMPT)

    def test_engine_returns_clear_error_when_answer_generation_fails(self):
        engine = self.build_engine(answer_generator=failing_answer_generator)
        result = engine.ask("Which park has the strongest customer satisfaction?")
        self.assertIsNotNone(result.error)
        self.assertTrue(result.error.startswith("Answer generation error:"))
