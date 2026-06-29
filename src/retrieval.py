from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

from .config import Settings, get_settings
from .planner import BRANCH_LABELS
from .schemas import (
    AnalyticsResult,
    EvidenceValidationSummary,
    QueryPlan,
    ResolvedRetrievalTask,
    RetrievalExecutionResult,
    RetrievedEvidence,
    RetrievalTask,
)


def _label_for_branch(branch: str | None) -> str | None:
    if branch is None:
        return None
    return BRANCH_LABELS.get(branch, branch)


def _as_filter_dict(task: RetrievalTask) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    if task.branch:
        filters["branch"] = task.branch
    if task.reviewer_location:
        filters["reviewer_location"] = task.reviewer_location
    if task.month is not None:
        filters["month"] = int(task.month)
    if task.season:
        filters["season"] = task.season
    if task.rating_filter != "all":
        filters["rating_sentiment"] = task.rating_filter
    return filters


def _to_chroma_where(filters: dict[str, Any]) -> dict[str, Any] | None:
    if not filters:
        return None
    clauses = [{key: value} for key, value in filters.items()]
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _comparison_label(candidate_filters: dict[str, Any]) -> str:
    if candidate_filters.get("month") is not None:
        return f"month={candidate_filters['month']}"
    if candidate_filters.get("season"):
        return f"season={candidate_filters['season']}"
    if candidate_filters.get("branch"):
        return str(candidate_filters["branch"])
    return "subset"


def resolve_retrieval_tasks(
    question: str,
    plan: QueryPlan,
    analytics_result: AnalyticsResult,
    *,
    settings: Settings | None = None,
) -> list[ResolvedRetrievalTask]:
    settings = settings or get_settings()
    resolved_tasks: list[ResolvedRetrievalTask] = []
    candidate_filters = analytics_result.recommended_retrieval_filters or [
        analytics_result.applied_filters
    ]

    if analytics_result.selected_candidates:
        top_candidate = analytics_result.selected_candidates[0]
        top_filters = {
            key: value
            for key, value in top_candidate.retrieval_filters.items()
            if value is not None
        }
        planner_tasks = plan.retrieval_tasks[: settings.max_planner_retrieval_tasks]
        for index, task in enumerate(planner_tasks):
            if len(resolved_tasks) >= settings.max_planner_retrieval_tasks:
                break
            merged = task.model_copy(update=top_filters)
            resolved_tasks.append(
                ResolvedRetrievalTask(
                    **merged.model_dump(),
                    task_id=f"task_{len(resolved_tasks) + 1}",
                    source="top_candidate",
                    candidate_value=top_candidate.value,
                    expected_filters=_as_filter_dict(merged),
                )
            )

        if len(analytics_result.selected_candidates) > 1 and len(resolved_tasks) < settings.max_planner_retrieval_tasks:
            runner_up = analytics_result.selected_candidates[1]
            comparison_filters = {
                key: value
                for key, value in runner_up.retrieval_filters.items()
                if value is not None
            }
            comparison_query = f"general experience for comparison in {_comparison_label(comparison_filters)}"
            comparison_task = RetrievalTask(
                query=comparison_query,
                purpose="general_experience",
                branch=comparison_filters.get("branch"),
                reviewer_location=comparison_filters.get("reviewer_location"),
                month=comparison_filters.get("month"),
                season=comparison_filters.get("season"),
                rating_filter="all",
                top_k=4,
            )
            resolved_tasks.append(
                ResolvedRetrievalTask(
                    **comparison_task.model_dump(),
                    task_id=f"task_{len(resolved_tasks) + 1}",
                    source="runner_up",
                    candidate_value=runner_up.value,
                    expected_filters=_as_filter_dict(comparison_task),
                )
            )
    else:
        for task in plan.retrieval_tasks[: settings.max_planner_retrieval_tasks]:
            resolved_tasks.append(
                ResolvedRetrievalTask(
                    **task.model_dump(),
                    task_id=f"task_{len(resolved_tasks) + 1}",
                    source="planner",
                    candidate_value=None,
                    expected_filters=_as_filter_dict(task),
                )
            )

    if not resolved_tasks:
        fallback_filters = candidate_filters[0] if candidate_filters else {}
        fallback_task = RetrievalTask(
            query=question,
            purpose="general_experience",
            branch=fallback_filters.get("branch"),
            reviewer_location=fallback_filters.get("reviewer_location"),
            month=fallback_filters.get("month"),
            season=fallback_filters.get("season"),
            rating_filter="all",
            top_k=4,
        )
        resolved_tasks.append(
            ResolvedRetrievalTask(
                **fallback_task.model_dump(),
                task_id="task_1",
                source="planner",
                candidate_value=None,
                expected_filters=_as_filter_dict(fallback_task),
            )
        )

    return resolved_tasks[: settings.max_planner_retrieval_tasks]


def _validate_hit_metadata(
    evidence: RetrievedEvidence,
    expected_filters: dict[str, Any],
    review_row: pd.Series | None,
) -> list[str]:
    errors: list[str] = []
    if expected_filters.get("branch") and evidence.branch != expected_filters["branch"]:
        errors.append("branch_mismatch")
    if expected_filters.get("reviewer_location") and evidence.reviewer_location != expected_filters["reviewer_location"]:
        errors.append("reviewer_location_mismatch")
    if expected_filters.get("month") is not None and evidence.month != int(expected_filters["month"]):
        errors.append("month_mismatch")
    if expected_filters.get("season") and evidence.season != expected_filters["season"]:
        errors.append("season_mismatch")
    if expected_filters.get("rating_sentiment") and evidence.rating_sentiment != expected_filters["rating_sentiment"]:
        errors.append("rating_sentiment_mismatch")
    if review_row is None:
        errors.append("review_id_missing_from_dataframe")
    return errors


def _build_evidence_record(
    *,
    doc,
    score: float,
    task: ResolvedRetrievalTask,
    review_lookup: dict[str, pd.Series],
) -> RetrievedEvidence:
    metadata = doc.metadata
    review_id = str(metadata["review_id"])
    review_row = review_lookup.get(review_id)
    relevance_score = round(float(1.0 / (1.0 + max(score, 0.0))), 6)
    evidence = RetrievedEvidence(
        review_id=review_id,
        text=doc.page_content,
        branch=str(metadata["branch"]),
        reviewer_location=str(metadata["reviewer_location"]),
        rating=int(metadata["rating"]),
        rating_sentiment=str(metadata["rating_sentiment"]),
        year=int(metadata["year"]),
        month=int(metadata["month"]),
        season=str(metadata["season"]),
        has_date=bool(metadata["has_date"]),
        relevance_score=relevance_score,
        matched_task_ids=[task.task_id],
        matched_queries=[task.query],
        matched_purposes=[task.purpose],
    )
    validation_errors = _validate_hit_metadata(evidence, task.expected_filters, review_row)
    if validation_errors:
        evidence.validation_passed = False
        evidence.validation_errors = validation_errors
    return evidence


def _merge_evidence_records(
    raw_hits: list[RetrievedEvidence],
    *,
    final_limit: int,
) -> list[RetrievedEvidence]:
    merged: dict[str, RetrievedEvidence] = {}
    for hit in raw_hits:
        existing = merged.get(hit.review_id)
        if existing is None:
            merged[hit.review_id] = hit
            continue
        existing.relevance_score = max(existing.relevance_score or 0.0, hit.relevance_score or 0.0)
        for task_id in hit.matched_task_ids:
            if task_id not in existing.matched_task_ids:
                existing.matched_task_ids.append(task_id)
        for query in hit.matched_queries:
            if query not in existing.matched_queries:
                existing.matched_queries.append(query)
        for purpose in hit.matched_purposes:
            if purpose not in existing.matched_purposes:
                existing.matched_purposes.append(purpose)
        if not hit.validation_passed:
            existing.validation_passed = False
            for error in hit.validation_errors:
                if error not in existing.validation_errors:
                    existing.validation_errors.append(error)

    task_buckets: dict[str, list[RetrievedEvidence]] = defaultdict(list)
    for evidence in merged.values():
        first_task_id = evidence.matched_task_ids[0]
        task_buckets[first_task_id].append(evidence)
    for bucket in task_buckets.values():
        bucket.sort(key=lambda item: item.relevance_score or 0.0, reverse=True)

    ordered: list[RetrievedEvidence] = []
    seen_ids: set[str] = set()
    while len(ordered) < final_limit:
        made_progress = False
        for task_id in sorted(task_buckets):
            bucket = task_buckets[task_id]
            while bucket and bucket[0].review_id in seen_ids:
                bucket.pop(0)
            if not bucket:
                continue
            evidence = bucket.pop(0)
            ordered.append(evidence)
            seen_ids.add(evidence.review_id)
            made_progress = True
            if len(ordered) >= final_limit:
                break
        if not made_progress:
            break
    return ordered[:final_limit]


def execute_multi_retrieval(
    question: str,
    plan: QueryPlan,
    analytics_result: AnalyticsResult,
    review_df: pd.DataFrame,
    vectorstore,
    *,
    resolved_tasks: list[ResolvedRetrievalTask] | None = None,
    settings: Settings | None = None,
) -> RetrievalExecutionResult:
    settings = settings or get_settings()
    resolved_tasks = resolved_tasks or resolve_retrieval_tasks(
        question,
        plan,
        analytics_result,
        settings=settings,
    )
    review_lookup = {
        str(row["Review_ID"]): row
        for _, row in review_df[
            [
                "Review_ID",
                "Branch",
                "Reviewer_Location",
                "Month",
                "Season",
                "Rating_Sentiment",
            ]
        ].iterrows()
    }
    raw_hits: list[RetrievedEvidence] = []
    warnings: list[str] = []

    for task in resolved_tasks:
        search_filter = _to_chroma_where(task.expected_filters)
        docs_with_scores = vectorstore.similarity_search_with_score(
            task.query,
            k=task.top_k,
            filter=search_filter,
        )
        if not docs_with_scores:
            warnings.append(f"No evidence returned for {task.task_id} ({task.purpose}).")
            continue
        for doc, score in docs_with_scores:
            raw_hits.append(
                _build_evidence_record(
                    doc=doc,
                    score=score,
                    task=task,
                    review_lookup=review_lookup,
                )
            )

    final_evidence = _merge_evidence_records(
        raw_hits,
        final_limit=settings.final_evidence_max,
    )
    invalid_hits = [hit for hit in final_evidence if not hit.validation_passed]
    validation = EvidenceValidationSummary(
        valid_count=len(final_evidence) - len(invalid_hits),
        invalid_count=len(invalid_hits),
        warnings=[f"{len(invalid_hits)} evidence items failed metadata validation."] if invalid_hits else [],
    )

    return RetrievalExecutionResult(
        resolved_tasks=resolved_tasks,
        raw_hits=raw_hits,
        final_evidence=final_evidence,
        validation=validation,
        warnings=warnings,
    )
