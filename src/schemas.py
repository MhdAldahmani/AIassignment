from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


IntentName = Literal[
    "segment_summary",
    "aspect_assessment",
    "time_comparison",
    "branch_comparison",
    "general_search",
]
ComparisonDimension = Literal["month", "season", "branch"]
RankingDirection = Literal["best", "worst", "highest", "lowest"]
RatingFilter = Literal["positive", "neutral", "negative", "all"]
ExternalContext = Literal[
    "weather",
    "holidays",
    "ticket_prices",
    "current_operations",
    "live_crowds",
    "other",
]
RequiredMetricName = Literal[
    "review_count",
    "average_rating",
    "median_rating",
    "positive_share",
    "neutral_share",
    "negative_share",
    "crowding_complaint_rate",
    "aspect_mention_rate",
    "rating_difference_from_baseline",
]
RetrievalPurpose = Literal[
    "supporting_praise",
    "supporting_complaints",
    "crowding",
    "staff",
    "food",
    "value",
    "rides",
    "maintenance",
    "cleanliness",
    "family",
    "general_experience",
    "tradeoffs",
]


class RetrievalTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    purpose: RetrievalPurpose
    branch: str | None = None
    reviewer_location: str | None = None
    month: int | None = None
    season: str | None = None
    rating_filter: RatingFilter = "all"
    top_k: int = 5


class QueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: IntentName
    branch: str | None = None
    reviewer_location: str | None = None
    month: int | None = None
    season: str | None = None
    comparison_dimension: ComparisonDimension | None = None
    compare_values: list[str] = Field(default_factory=list)
    requested_aspects: list[str] = Field(default_factory=list)
    required_metrics: list[RequiredMetricName] = Field(default_factory=list)
    ranking_direction: RankingDirection | None = None
    retrieval_tasks: list[RetrievalTask] = Field(default_factory=list)
    external_context_requested: list[ExternalContext] = Field(default_factory=list)
    clarification_needed: bool = False
    clarification_reason: str | None = None


class PlanValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    planner_source: Literal["llm", "deterministic_fallback"]
    raw_plan: QueryPlan
    normalized_plan: QueryPlan
    warnings: list[str] = Field(default_factory=list)


class TimeComparisonRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row_type: Literal["time"] = "time"
    comparison_dimension: Literal["month", "season"]
    group_key: str
    month: int | None = None
    month_name: str | None = None
    season: str | None = None
    review_count: int
    eligible: bool
    average_rating: float | None = None
    positive_share: float | None = None
    negative_share: float | None = None
    crowding_complaint_rate: float | None = None
    rating_difference_from_baseline: float | None = None
    aspect_mention_rate: float | None = None
    ranking_metric_value: float | None = None
    normalized_average_rating: float | None = None
    normalized_low_negative_share: float | None = None
    normalized_low_crowding_complaint_rate: float | None = None
    visit_score: float | None = None


class BranchComparisonRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row_type: Literal["branch"] = "branch"
    branch: str
    review_count: int
    average_rating: float | None = None
    positive_share: float | None = None
    negative_share: float | None = None
    crowding_complaint_rate: float | None = None
    aspect_mention_rate: float | None = None
    aspect_negative_share: float | None = None
    ranking_metric_value: float | None = None


ComparisonRow = Annotated[TimeComparisonRow | BranchComparisonRow, Field(discriminator="row_type")]


class CandidateGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    comparison_dimension: ComparisonDimension
    value: str
    branch: str | None = None
    month: int | None = None
    season: str | None = None
    metric_name: str
    metric_value: float | None = None
    retrieval_filters: dict[str, Any] = Field(default_factory=dict)


class AnalyticsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: IntentName
    selected_analytics_function: str
    applied_filters: dict[str, Any] = Field(default_factory=dict)
    sample_size: int
    metrics: dict[str, Any] = Field(default_factory=dict)
    baseline_metrics: dict[str, Any] = Field(default_factory=dict)
    dataset_date_coverage: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    comparison_rows: list[ComparisonRow] = Field(default_factory=list)
    selected_candidates: list[CandidateGroup] = Field(default_factory=list)
    recommended_retrieval_filters: list[dict[str, Any]] = Field(default_factory=list)
    ranking_criterion: str | None = None
    score_definition: str | None = None


class ResolvedRetrievalTask(RetrievalTask):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    source: Literal["planner", "top_candidate", "runner_up", "comparison_support"]
    candidate_value: str | None = None
    expected_filters: dict[str, Any] = Field(default_factory=dict)


class RetrievedEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_id: str
    text: str
    branch: str
    reviewer_location: str
    rating: int
    rating_sentiment: str
    year: int
    month: int
    season: str
    has_date: bool
    relevance_score: float | None = None
    matched_task_ids: list[str] = Field(default_factory=list)
    matched_queries: list[str] = Field(default_factory=list)
    matched_purposes: list[str] = Field(default_factory=list)
    validation_passed: bool = True
    validation_errors: list[str] = Field(default_factory=list)


class EvidenceValidationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid_count: int = 0
    invalid_count: int = 0
    warnings: list[str] = Field(default_factory=list)


class RetrievalExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resolved_tasks: list[ResolvedRetrievalTask] = Field(default_factory=list)
    raw_hits: list[RetrievedEvidence] = Field(default_factory=list)
    final_evidence: list[RetrievedEvidence] = Field(default_factory=list)
    validation: EvidenceValidationSummary = Field(default_factory=EvidenceValidationSummary)
    warnings: list[str] = Field(default_factory=list)


class GroundedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    direct_answer: str
    supporting_metrics: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
