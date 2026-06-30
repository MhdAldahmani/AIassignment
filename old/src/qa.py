from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Settings, get_settings
from .data import load_clean_reviews
from .graph import build_qa_graph
from .planner import AdaptivePlanner
from .prompts import build_answer_generator, render_grounded_answer
from .schemas import (
    AnalyticsResult,
    EvidenceValidationSummary,
    GroundedAnswer,
    QueryPlan,
    ResolvedRetrievalTask,
    RetrievedEvidence,
)
from .vectorstore import build_or_load_vectorstore


@dataclass
class QAEngineResult:
    question: str
    answer: str | None
    raw_plan: QueryPlan | None = None
    plan: QueryPlan | None = None
    analytics_result: AnalyticsResult | None = None
    retrieval_tasks: list[ResolvedRetrievalTask] = field(default_factory=list)
    evidence: list[RetrievedEvidence] = field(default_factory=list)
    error: str | None = None
    structured_answer: GroundedAnswer | None = None
    retrieval_validation: EvidenceValidationSummary | None = None
    retrieval_warnings: list[str] | None = field(default_factory=list)
    answer_payload: dict[str, str] | None = None
    trace: dict[str, Any] | None = None


class QAEngine:
    def __init__(
        self,
        *,
        review_df: pd.DataFrame,
        vectorstore,
        settings: Settings | None = None,
        answer_generator=None,
        use_planner_llm: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.review_df = review_df
        self.vectorstore = vectorstore
        self.planner = AdaptivePlanner(settings=self.settings, use_llm=use_planner_llm)
        self.answer_generator = answer_generator
        if self.answer_generator is None and self.settings.openai_api_key:
            self.answer_generator = build_answer_generator(self.settings)
        self.graph = build_qa_graph(
            review_df=self.review_df,
            vectorstore=self.vectorstore,
            planner=self.planner,
            answer_generator=self.answer_generator,
            settings=self.settings,
        )

    def ask(self, question: str, *, return_trace: bool = False) -> QAEngineResult:
        state = self.graph.invoke({"question": question})
        structured_answer = state.get("structured_answer")
        answer = render_grounded_answer(structured_answer) if structured_answer else None
        trace = None
        if return_trace:
            trace = {
                "question": question,
                "raw_plan": state.get("raw_plan"),
                "normalized_plan": state.get("normalized_plan"),
                "analytics_result": state.get("analytics_result"),
                "retrieval_tasks": state.get("retrieval_tasks", []),
                "retrieval_validation": state.get("retrieval_validation"),
                "retrieval_warnings": state.get("retrieval_warnings", []),
                "answer_payload": state.get("answer_payload"),
                "structured_answer": structured_answer,
                "error": state.get("error"),
            }
        return QAEngineResult(
            question=question,
            answer=answer,
            raw_plan=state.get("raw_plan"),
            plan=state.get("normalized_plan"),
            analytics_result=state.get("analytics_result"),
            retrieval_tasks=state.get("retrieval_tasks", []),
            evidence=state.get("evidence", []),
            error=state.get("error"),
            structured_answer=structured_answer,
            retrieval_validation=state.get("retrieval_validation"),
            retrieval_warnings=state.get("retrieval_warnings", []),
            answer_payload=state.get("answer_payload"),
            trace=trace,
        )


def build_qa_engine(
    *,
    dataset_path: str | Path = "DisneylandReviews.csv",
    review_df: pd.DataFrame | None = None,
    vectorstore=None,
    settings: Settings | None = None,
    answer_generator=None,
    use_planner_llm: bool = True,
    force_rebuild_index: bool = False,
    verbose_index: bool = False,
) -> QAEngine:
    settings = settings or get_settings()
    review_df = review_df if review_df is not None else load_clean_reviews(dataset_path)
    vectorstore = vectorstore or build_or_load_vectorstore(
        review_df,
        settings=settings,
        force_rebuild=force_rebuild_index,
        verbose=verbose_index,
    ).vectorstore
    return QAEngine(
        review_df=review_df,
        vectorstore=vectorstore,
        settings=settings,
        answer_generator=answer_generator,
        use_planner_llm=use_planner_llm,
    )
