from __future__ import annotations

from typing import TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from .analytics import (
    analyze_aspect,
    analyze_branch_comparison,
    analyze_general,
    analyze_segment,
    analyze_time_comparison,
)
from .config import Settings, get_settings
from .planner import AdaptivePlanner
from .prompts import build_answer_payload, invoke_answer_generator
from .retrieval import execute_multi_retrieval, resolve_retrieval_tasks
from .schemas import (
    AnalyticsResult,
    EvidenceValidationSummary,
    GroundedAnswer,
    QueryPlan,
    ResolvedRetrievalTask,
    RetrievedEvidence,
)


class QAState(TypedDict, total=False):
    question: str
    raw_plan: QueryPlan
    normalized_plan: QueryPlan
    analytics_result: AnalyticsResult
    retrieval_tasks: list[ResolvedRetrievalTask]
    evidence: list[RetrievedEvidence]
    retrieval_validation: EvidenceValidationSummary
    retrieval_warnings: list[str]
    answer_payload: dict[str, str]
    structured_answer: GroundedAnswer
    error: str


def build_qa_graph(
    *,
    review_df: pd.DataFrame,
    vectorstore,
    planner: AdaptivePlanner,
    answer_generator,
    settings: Settings | None = None,
):
    settings = settings or get_settings()
    graph = StateGraph(QAState)

    def parse_question_node(state: QAState) -> QAState:
        try:
            raw_plan = planner.plan_without_fallback(state["question"], review_df)
            return {"raw_plan": raw_plan}
        except Exception as exc:
            return {"error": f"Planner error: {exc}"}

    def normalize_plan_node(state: QAState) -> QAState:
        try:
            normalized_plan, _ = planner.normalize_plan_with_context(
                state["raw_plan"],
                review_df,
                question=state["question"],
            )
            return {"normalized_plan": normalized_plan}
        except Exception as exc:
            return {"error": f"Planner error: {exc}"}

    def analyze_segment_node(state: QAState) -> QAState:
        try:
            return {"analytics_result": analyze_segment(state["normalized_plan"], review_df, settings)}
        except Exception as exc:
            return {"error": f"Analytics error: {exc}"}

    def analyze_aspect_node(state: QAState) -> QAState:
        try:
            return {"analytics_result": analyze_aspect(state["normalized_plan"], review_df, settings)}
        except Exception as exc:
            return {"error": f"Analytics error: {exc}"}

    def analyze_time_node(state: QAState) -> QAState:
        try:
            return {"analytics_result": analyze_time_comparison(state["normalized_plan"], review_df, settings)}
        except Exception as exc:
            return {"error": f"Analytics error: {exc}"}

    def analyze_branch_node(state: QAState) -> QAState:
        try:
            return {"analytics_result": analyze_branch_comparison(state["normalized_plan"], review_df, settings)}
        except Exception as exc:
            return {"error": f"Analytics error: {exc}"}

    def analyze_general_node(state: QAState) -> QAState:
        try:
            return {"analytics_result": analyze_general(state["normalized_plan"], review_df, settings)}
        except Exception as exc:
            return {"error": f"Analytics error: {exc}"}

    def resolve_retrieval_tasks_node(state: QAState) -> QAState:
        try:
            tasks = resolve_retrieval_tasks(
                state["question"],
                state["normalized_plan"],
                state["analytics_result"],
                settings=settings,
            )
            return {"retrieval_tasks": tasks}
        except Exception as exc:
            return {"error": f"Retrieval error: {exc}"}

    def retrieve_evidence_node(state: QAState) -> QAState:
        try:
            retrieval_result = execute_multi_retrieval(
                state["question"],
                state["normalized_plan"],
                state["analytics_result"],
                review_df,
                vectorstore,
                resolved_tasks=state["retrieval_tasks"],
                settings=settings,
            )
            if retrieval_result.validation.invalid_count:
                return {
                    "evidence": retrieval_result.final_evidence,
                    "retrieval_validation": retrieval_result.validation,
                    "retrieval_warnings": retrieval_result.warnings,
                    "error": (
                        "Retrieval error: "
                        f"{retrieval_result.validation.invalid_count} evidence items failed metadata validation."
                    ),
                }
            return {
                "evidence": retrieval_result.final_evidence,
                "retrieval_validation": retrieval_result.validation,
                "retrieval_warnings": retrieval_result.warnings,
            }
        except Exception as exc:
            return {"error": f"Retrieval error: {exc}"}

    def generate_answer_node(state: QAState) -> QAState:
        try:
            if answer_generator is None:
                raise ValueError("OPENAI_API_KEY is not configured.")
            payload = build_answer_payload(
                question=state["question"],
                plan=state["normalized_plan"],
                analytics_result=state["analytics_result"],
                evidence=state.get("evidence", []),
            )
            structured_answer = invoke_answer_generator(answer_generator, payload)
            return {
                "answer_payload": payload,
                "structured_answer": structured_answer,
            }
        except Exception as exc:
            return {"error": f"Answer generation error: {exc}"}

    def route_if_error(state: QAState) -> str:
        return END if state.get("error") else "continue"

    def route_intent(state: QAState) -> str:
        intent = state["normalized_plan"].intent
        routes = {
            "segment_summary": "analyze_segment",
            "aspect_assessment": "analyze_aspect",
            "time_comparison": "analyze_time",
            "branch_comparison": "analyze_branch",
            "general_search": "analyze_general",
        }
        return routes[intent]

    graph.add_node("parse_question", parse_question_node)
    graph.add_node("normalize_plan", normalize_plan_node)
    graph.add_node("analyze_segment", analyze_segment_node)
    graph.add_node("analyze_aspect", analyze_aspect_node)
    graph.add_node("analyze_time", analyze_time_node)
    graph.add_node("analyze_branch", analyze_branch_node)
    graph.add_node("analyze_general", analyze_general_node)
    graph.add_node("resolve_retrieval_tasks", resolve_retrieval_tasks_node)
    graph.add_node("retrieve_evidence", retrieve_evidence_node)
    graph.add_node("generate_answer", generate_answer_node)

    graph.add_edge(START, "parse_question")
    graph.add_conditional_edges("parse_question", route_if_error, {"continue": "normalize_plan", END: END})
    graph.add_conditional_edges(
        "normalize_plan",
        lambda state: END if state.get("error") else route_intent(state),
        {
            "analyze_segment": "analyze_segment",
            "analyze_aspect": "analyze_aspect",
            "analyze_time": "analyze_time",
            "analyze_branch": "analyze_branch",
            "analyze_general": "analyze_general",
            END: END,
        },
    )
    for node_name in [
        "analyze_segment",
        "analyze_aspect",
        "analyze_time",
        "analyze_branch",
        "analyze_general",
    ]:
        graph.add_conditional_edges(
            node_name,
            route_if_error,
            {"continue": "resolve_retrieval_tasks", END: END},
        )
    graph.add_conditional_edges(
        "resolve_retrieval_tasks",
        route_if_error,
        {"continue": "retrieve_evidence", END: END},
    )
    graph.add_conditional_edges(
        "retrieve_evidence",
        route_if_error,
        {"continue": "generate_answer", END: END},
    )
    graph.add_edge("generate_answer", END)
    return graph.compile()
