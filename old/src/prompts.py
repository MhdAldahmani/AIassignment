from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from .config import Settings, get_settings
from .schemas import AnalyticsResult, GroundedAnswer, QueryPlan, RetrievedEvidence

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - optional dependency
    ChatOpenAI = None


GROUNDING_SYSTEM_PROMPT = """You answer questions only from the supplied Disneyland review dataset evidence.

Use only:
1. Deterministic statistics calculated from the dataset
2. Retrieved review excerpts

Do not use outside knowledge about:
- Disney parks
- Tourism
- Weather
- Public holidays
- Ticket prices
- Current operations
- Current crowd levels

Rules:
- Never invent or modify numbers.
- Mention the sample size.
- For best/worst questions, state the criteria used.
- Describe the result as the strongest candidate in this historical dataset.
- Do not claim universal truth.
- Mention at least one limitation.
- If deterministic warnings mention excluded missing dates or small samples, include that limitation explicitly.
- If requested information is unavailable in the dataset, say so clearly.
- Do not make causal claims.
"""


def _truncate(text: str, limit: int = 220) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def format_analytics_result_for_prompt(analytics_result: AnalyticsResult) -> str:
    payload = analytics_result.model_dump(mode="json")
    return json.dumps(payload, indent=2, ensure_ascii=False)


def format_evidence_for_prompt(evidence: list[RetrievedEvidence], *, max_items: int = 8) -> str:
    if not evidence:
        return "No retrieved evidence was provided."
    lines: list[str] = []
    for index, item in enumerate(evidence[:max_items], start=1):
        lines.append(
            (
                f"[{index}] review_id={item.review_id}; branch={item.branch}; "
                f"reviewer_location={item.reviewer_location}; rating={item.rating}; "
                f"month={item.month}; season={item.season}\n{_truncate(item.text)}"
            )
        )
    return "\n\n".join(lines)


def build_answer_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", GROUNDING_SYSTEM_PROMPT),
            (
                "human",
                (
                    "Question: {question}\n\n"
                    "Normalized query plan:\n{plan_json}\n\n"
                    "AUTHORITATIVE DETERMINISTIC SUMMARY\n"
                    "{deterministic_summary}\n\n"
                    "AUTHORITATIVE DATASET STATISTICS\n"
                    "These values were calculated deterministically. Never modify or recompute them.\n"
                    "{analytics_json}\n\n"
                    "RETRIEVED QUALITATIVE EVIDENCE\n"
                    "{evidence_block}\n\n"
                    "Dataset date range for this answer subset: {date_range}\n"
                    "Unsupported or unavailable external context requested: {unsupported_context}\n\n"
                    "Return a concise GroundedAnswer."
                ),
            ),
        ]
    )


def build_answer_generator(settings: Settings | None = None):
    settings = settings or get_settings()
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not configured.")
    if ChatOpenAI is None:
        raise ImportError("langchain_openai is not installed.")
    prompt = build_answer_prompt()
    llm = ChatOpenAI(
        model=settings.answer_model,
        temperature=settings.answer_temperature,
        use_responses_api=True,
        reasoning_effort="low",
        verbosity="low",
    )
    return prompt | llm.with_structured_output(GroundedAnswer)


def build_deterministic_summary(plan: QueryPlan, analytics_result: AnalyticsResult) -> str:
    lines = [
        f"Intent: {plan.intent}",
        f"Analytics route: {analytics_result.selected_analytics_function}",
        f"Sample size: {analytics_result.sample_size}",
    ]
    if analytics_result.ranking_criterion:
        lines.append(f"Ranking criterion: {analytics_result.ranking_criterion}")
    if analytics_result.score_definition:
        lines.append(f"Score definition: {analytics_result.score_definition}")

    if analytics_result.selected_candidates:
        for index, candidate in enumerate(analytics_result.selected_candidates, start=1):
            label = "Top candidate" if index == 1 else "Runner-up"
            lines.append(
                f"{label}: {candidate.value} "
                f"(metric={candidate.metric_name}, metric_value={candidate.metric_value})"
            )
    if analytics_result.warnings:
        for warning in analytics_result.warnings:
            lines.append(f"Deterministic warning: {warning}")
    if plan.intent == "aspect_assessment":
        lines.append(
            "Aspect summary: "
            f"total_filtered_review_count={analytics_result.metrics.get('total_filtered_review_count')}, "
            f"aspect_review_count={analytics_result.metrics.get('aspect_review_count')}, "
            f"aspect_mention_rate={analytics_result.metrics.get('aspect_mention_rate')}, "
            f"aspect_subset_average_rating={analytics_result.metrics.get('average_rating')}, "
            f"baseline_average_rating={analytics_result.baseline_metrics.get('average_rating')}"
        )

    if plan.intent == "time_comparison" and analytics_result.comparison_rows:
        rows = analytics_result.comparison_rows[:4]
        for row in rows:
            if getattr(row, "row_type", None) == "time":
                lines.append(
                    f"Time row {row.group_key}: review_count={row.review_count}, "
                    f"average_rating={row.average_rating}, positive_share={row.positive_share}, "
                    f"negative_share={row.negative_share}, crowding_complaint_rate={row.crowding_complaint_rate}, "
                    f"visit_score={row.visit_score}"
                )
    if plan.intent == "branch_comparison" and analytics_result.comparison_rows:
        for row in analytics_result.comparison_rows[:3]:
            lines.append(
                f"Branch row {row.branch}: review_count={row.review_count}, "
                f"average_rating={row.average_rating}, positive_share={row.positive_share}, "
                f"negative_share={row.negative_share}"
            )
    return "\n".join(lines)


def build_answer_payload(
    *,
    question: str,
    plan: QueryPlan,
    analytics_result: AnalyticsResult,
    evidence: list[RetrievedEvidence],
) -> dict[str, str]:
    date_range = (
        f"{analytics_result.dataset_date_coverage.get('start')} to "
        f"{analytics_result.dataset_date_coverage.get('end')}"
    )
    unsupported_context = (
        ", ".join(plan.external_context_requested) if plan.external_context_requested else "None"
    )
    return {
        "question": question,
        "plan_json": json.dumps(plan.model_dump(mode="json"), indent=2, ensure_ascii=False),
        "deterministic_summary": build_deterministic_summary(plan, analytics_result),
        "analytics_json": format_analytics_result_for_prompt(analytics_result),
        "evidence_block": format_evidence_for_prompt(evidence),
        "date_range": date_range,
        "unsupported_context": unsupported_context,
    }


def invoke_answer_generator(answer_generator: Any, payload: dict[str, Any]) -> GroundedAnswer:
    if hasattr(answer_generator, "invoke"):
        result = answer_generator.invoke(payload)
    elif isinstance(answer_generator, Callable):
        result = answer_generator(payload)
    else:
        raise TypeError("Answer generator must be callable or expose invoke().")

    if isinstance(result, GroundedAnswer):
        return result
    if isinstance(result, dict):
        return GroundedAnswer(**result)
    raise TypeError("Answer generator did not return a GroundedAnswer-compatible result.")


def render_grounded_answer(answer: GroundedAnswer) -> str:
    sections = [answer.direct_answer.strip()]
    if answer.supporting_metrics:
        sections.append("Supporting metrics: " + "; ".join(answer.supporting_metrics))
    if answer.evidence:
        sections.append("Evidence: " + " | ".join(answer.evidence))
    if answer.limitations:
        sections.append("Limitations: " + "; ".join(answer.limitations))
    return "\n\n".join(section for section in sections if section)
