from __future__ import annotations

import re
from typing import Any

import pandas as pd

from .config import Settings, get_settings
from .planner import ASPECT_KEYWORDS, BRANCH_LABELS
from .schemas import AnalyticsResult, BranchComparisonRow, CandidateGroup, QueryPlan, TimeComparisonRow


CROWDING_COMPLAINT_TERMS = [
    "long queue",
    "long queues",
    "long line",
    "long lines",
    "waited",
    "waiting",
    "too crowded",
    "overcrowded",
    "packed",
    "busy",
]


def _safe_round(value: float | None, digits: int = 3) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def _percentage(series: pd.Series, value: str) -> float:
    if series.empty:
        return 0.0
    return round(float(series.eq(value).mean() * 100), 3)


def _date_coverage(df: pd.DataFrame) -> dict[str, Any]:
    review_dates = df["Review_Date"]
    return {
        "start": review_dates.min().strftime("%Y-%m") if review_dates.notna().any() else None,
        "end": review_dates.max().strftime("%Y-%m") if review_dates.notna().any() else None,
        "rows_with_missing_review_date": int(review_dates.isna().sum()),
    }


def _core_metrics(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "review_count": 0,
            "average_rating": None,
            "median_rating": None,
            "positive_share": None,
            "neutral_share": None,
            "negative_share": None,
        }
    return {
        "review_count": int(len(df)),
        "average_rating": _safe_round(df["Rating"].mean(), 3),
        "median_rating": _safe_round(df["Rating"].median(), 3),
        "positive_share": _percentage(df["Rating_Sentiment"], "positive"),
        "neutral_share": _percentage(df["Rating_Sentiment"], "neutral"),
        "negative_share": _percentage(df["Rating_Sentiment"], "negative"),
    }


def _apply_plan_filters(
    df: pd.DataFrame,
    plan: QueryPlan,
    *,
    ignore_dimension_filter: bool = False,
) -> pd.DataFrame:
    filtered_df = df.copy()
    if plan.branch and not (ignore_dimension_filter and plan.comparison_dimension == "branch"):
        filtered_df = filtered_df.loc[filtered_df["Branch"].eq(plan.branch)]
    if plan.reviewer_location:
        filtered_df = filtered_df.loc[filtered_df["Reviewer_Location"].eq(plan.reviewer_location)]
    if plan.month and not (ignore_dimension_filter and plan.comparison_dimension == "month"):
        filtered_df = filtered_df.loc[filtered_df["Month"].eq(plan.month)]
    if plan.season and not (ignore_dimension_filter and plan.comparison_dimension == "season"):
        filtered_df = filtered_df.loc[filtered_df["Season"].eq(plan.season)]
    return filtered_df.copy()


def _baseline_subset(df: pd.DataFrame, plan: QueryPlan) -> pd.DataFrame:
    baseline_df = df.copy()
    if plan.branch:
        baseline_df = baseline_df.loc[baseline_df["Branch"].eq(plan.branch)]
    return baseline_df.copy()


def _pattern_for_aspect(aspect: str) -> str:
    keywords = ASPECT_KEYWORDS.get(aspect, [])
    return "|".join(re.escape(keyword) for keyword in keywords)


def _aspect_mask(df: pd.DataFrame, aspect: str) -> pd.Series:
    if aspect not in ASPECT_KEYWORDS:
        return pd.Series(False, index=df.index)
    pattern = _pattern_for_aspect(aspect)
    return df["Review_Text"].str.contains(pattern, case=False, regex=True, na=False)


def _crowding_complaint_mask(df: pd.DataFrame) -> pd.Series:
    crowding_mask = _aspect_mask(df, "crowding")
    complaint_pattern = "|".join(re.escape(term) for term in CROWDING_COMPLAINT_TERMS)
    complaint_language = df["Review_Text"].str.contains(
        complaint_pattern,
        case=False,
        regex=True,
        na=False,
    )
    lower_rating = df["Rating_Sentiment"].isin(["negative", "neutral"])
    relief_pattern = r"\b(?:no queues?|not crowded|hardly any queues?)\b"
    relief_language = df["Review_Text"].str.contains(relief_pattern, case=False, regex=True, na=False)
    return crowding_mask & ((complaint_language | lower_rating) & ~relief_language)


def _add_sample_warnings(warnings: list[str], sample_size: int) -> None:
    if sample_size == 0:
        warnings.append("No reviews matched the requested filters.")
    elif sample_size < 30:
        warnings.append("Small sample size: treat the result as directional rather than stable.")
    elif sample_size < 100:
        warnings.append("Moderate sample size: useful, but still worth interpreting cautiously.")


def _normalize_series(series: pd.Series, *, higher_is_better: bool) -> pd.Series:
    numeric = series.astype(float)
    min_value = numeric.min()
    max_value = numeric.max()
    if pd.isna(min_value) or pd.isna(max_value):
        return pd.Series(0.0, index=series.index)
    if max_value == min_value:
        return pd.Series(1.0, index=series.index)
    scaled = (numeric - min_value) / (max_value - min_value)
    if not higher_is_better:
        scaled = 1 - scaled
    return scaled


def analyze_segment(plan: QueryPlan, df: pd.DataFrame, settings: Settings | None = None) -> AnalyticsResult:
    filtered_df = _apply_plan_filters(df, plan)
    baseline_df = _baseline_subset(df, plan)
    warnings: list[str] = []
    _add_sample_warnings(warnings, len(filtered_df))

    metrics = _core_metrics(filtered_df)
    baseline_metrics = _core_metrics(baseline_df)
    if plan.branch and baseline_metrics["review_count"] and metrics["average_rating"] is not None:
        baseline_avg = baseline_metrics["average_rating"]
        if baseline_avg is not None:
            metrics["rating_difference_from_baseline"] = _safe_round(
                metrics["average_rating"] - baseline_avg,
                3,
            )

    return AnalyticsResult(
        intent=plan.intent,
        selected_analytics_function="analyze_segment",
        applied_filters={key: value for key, value in plan.model_dump().items() if key in {"branch", "reviewer_location", "month", "season"} and value is not None},
        sample_size=int(len(filtered_df)),
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        dataset_date_coverage=_date_coverage(filtered_df),
        warnings=warnings,
        recommended_retrieval_filters=[
            {key: value for key, value in plan.model_dump().items() if key in {"branch", "reviewer_location", "month", "season"} and value is not None}
        ],
    )


def analyze_aspect(plan: QueryPlan, df: pd.DataFrame, settings: Settings | None = None) -> AnalyticsResult:
    filtered_df = _apply_plan_filters(df, plan)
    baseline_df = filtered_df.copy()
    warnings: list[str] = []
    primary_aspect = plan.requested_aspects[0] if plan.requested_aspects else "general_experience"
    aspect_df = filtered_df.loc[_aspect_mask(filtered_df, primary_aspect)] if primary_aspect in ASPECT_KEYWORDS else filtered_df.copy()

    _add_sample_warnings(warnings, len(aspect_df))
    metrics = _core_metrics(aspect_df)
    baseline_metrics = _core_metrics(baseline_df)
    metrics["total_filtered_review_count"] = int(len(filtered_df))
    metrics["aspect_review_count"] = int(len(aspect_df))
    if len(filtered_df) > 0:
        metrics["aspect_mention_rate"] = _safe_round((len(aspect_df) / len(filtered_df)) * 100, 3)
    else:
        metrics["aspect_mention_rate"] = None
    if metrics["average_rating"] is not None and baseline_metrics["average_rating"] is not None:
        metrics["rating_difference_from_baseline"] = _safe_round(
            metrics["average_rating"] - baseline_metrics["average_rating"],
            3,
        )

    return AnalyticsResult(
        intent=plan.intent,
        selected_analytics_function="analyze_aspect",
        applied_filters={key: value for key, value in plan.model_dump().items() if key in {"branch", "reviewer_location", "month", "season"} and value is not None},
        sample_size=int(len(filtered_df)),
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        dataset_date_coverage=_date_coverage(filtered_df),
        warnings=warnings,
        recommended_retrieval_filters=[
            {
                **{key: value for key, value in plan.model_dump().items() if key in {"branch", "reviewer_location", "month", "season"} and value is not None},
                "aspect": primary_aspect,
            }
        ],
    )


def _build_time_rows(plan: QueryPlan, dated_df: pd.DataFrame, baseline_average_rating: float | None, settings: Settings) -> pd.DataFrame:
    primary_aspect = plan.requested_aspects[0] if plan.requested_aspects else None
    crowding_mask = _crowding_complaint_mask(dated_df)
    aspect_mask = _aspect_mask(dated_df, primary_aspect) if primary_aspect in ASPECT_KEYWORDS else pd.Series(False, index=dated_df.index)
    working_df = dated_df.assign(
        is_positive=dated_df["Rating_Sentiment"].eq("positive").astype(int),
        is_negative=dated_df["Rating_Sentiment"].eq("negative").astype(int),
        crowding_complaint=crowding_mask.astype(int),
        aspect_match=aspect_mask.astype(int),
    )

    if plan.comparison_dimension == "season":
        grouped = (
            working_df.groupby("Season", observed=True)
            .agg(
                review_count=("Review_ID", "size"),
                average_rating=("Rating", "mean"),
                positive_share=("is_positive", "mean"),
                negative_share=("is_negative", "mean"),
                crowding_complaint_rate=("crowding_complaint", "mean"),
                aspect_mention_rate=("aspect_match", "mean"),
            )
            .reset_index()
            .rename(columns={"Season": "group_key"})
        )
        grouped["comparison_dimension"] = "season"
        grouped["season"] = grouped["group_key"]
        grouped["month"] = None
        grouped["month_name"] = None
        threshold = settings.min_season_reviews
    else:
        grouped = (
            working_df.groupby(["Month", "Month_Name"], observed=True)
            .agg(
                review_count=("Review_ID", "size"),
                average_rating=("Rating", "mean"),
                positive_share=("is_positive", "mean"),
                negative_share=("is_negative", "mean"),
                crowding_complaint_rate=("crowding_complaint", "mean"),
                aspect_mention_rate=("aspect_match", "mean"),
            )
            .reset_index()
            .rename(columns={"Month": "month", "Month_Name": "group_key"})
        )
        grouped["comparison_dimension"] = "month"
        grouped["month_name"] = grouped["group_key"]
        grouped["season"] = None
        threshold = settings.min_month_reviews

    grouped["positive_share"] = grouped["positive_share"] * 100
    grouped["negative_share"] = grouped["negative_share"] * 100
    grouped["crowding_complaint_rate"] = grouped["crowding_complaint_rate"] * 100
    grouped["aspect_mention_rate"] = grouped["aspect_mention_rate"] * 100
    grouped["rating_difference_from_baseline"] = grouped["average_rating"] - (baseline_average_rating or 0.0)
    grouped["eligible"] = grouped["review_count"] >= threshold
    if plan.comparison_dimension == "season":
        grouped = grouped.sort_values("group_key").reset_index(drop=True)
    else:
        grouped = grouped.sort_values("month").reset_index(drop=True)
    return grouped


def _filter_time_rows_for_compare_values(grouped: pd.DataFrame, plan: QueryPlan) -> pd.DataFrame:
    if not plan.compare_values:
        return grouped
    if plan.comparison_dimension == "season":
        filtered = grouped.loc[grouped["group_key"].isin(plan.compare_values)].copy()
        filtered["sort_order"] = filtered["group_key"].map({value: index for index, value in enumerate(plan.compare_values)})
        return filtered.sort_values("sort_order").drop(columns="sort_order")
    if plan.comparison_dimension == "month":
        filtered = grouped.loc[grouped["group_key"].isin(plan.compare_values)].copy()
        filtered["sort_order"] = filtered["group_key"].map({value: index for index, value in enumerate(plan.compare_values)})
        return filtered.sort_values("sort_order").drop(columns="sort_order")
    return grouped


def _select_time_ranking_criterion(plan: QueryPlan) -> tuple[str, bool]:
    requested_metrics = set(plan.required_metrics)
    if "crowding_complaint_rate" in requested_metrics and requested_metrics.issubset(
        {"review_count", "crowding_complaint_rate", "average_rating", "negative_share"}
    ):
        return "crowding_complaint_rate", False
    if requested_metrics.issubset({"review_count", "average_rating", "positive_share", "negative_share"}):
        return "average_rating", plan.ranking_direction not in {"lowest", "worst"}
    return "visit_score", True


def analyze_time_comparison(plan: QueryPlan, df: pd.DataFrame, settings: Settings | None = None) -> AnalyticsResult:
    settings = settings or get_settings()
    warnings: list[str] = []

    base_df = _apply_plan_filters(df, plan, ignore_dimension_filter=True)
    dated_df = base_df.dropna(subset=["Review_Date"]).copy()
    excluded_missing_dates = int(len(base_df) - len(dated_df))
    if excluded_missing_dates:
        warnings.append(
            f"Excluded {excluded_missing_dates:,} reviews with missing dates from the time comparison."
        )
    _add_sample_warnings(warnings, len(dated_df))

    baseline_metrics = _core_metrics(dated_df)
    grouped = _build_time_rows(plan, dated_df, baseline_metrics["average_rating"], settings)
    grouped = _filter_time_rows_for_compare_values(grouped, plan)

    ranking_criterion = None
    score_definition = None
    selected_candidates: list[CandidateGroup] = []
    grouped["visit_score"] = None
    grouped["normalized_average_rating"] = None
    grouped["normalized_low_negative_share"] = None
    grouped["normalized_low_crowding_complaint_rate"] = None

    if plan.ranking_direction and not plan.compare_values and not grouped.empty:
        ranking_criterion, higher_is_better = _select_time_ranking_criterion(plan)
        eligible_rows = grouped.loc[grouped["eligible"]].copy()
        if eligible_rows.empty:
            warnings.append("No time periods met the minimum review threshold for ranking.")
        else:
            if ranking_criterion == "visit_score":
                eligible_rows["normalized_average_rating"] = _normalize_series(
                    eligible_rows["average_rating"],
                    higher_is_better=True,
                )
                eligible_rows["normalized_low_negative_share"] = _normalize_series(
                    eligible_rows["negative_share"],
                    higher_is_better=False,
                )
                eligible_rows["normalized_low_crowding_complaint_rate"] = _normalize_series(
                    eligible_rows["crowding_complaint_rate"],
                    higher_is_better=False,
                )
                eligible_rows["visit_score"] = (
                    eligible_rows["normalized_average_rating"] * settings.visit_score_rating_weight
                    + eligible_rows["normalized_low_negative_share"] * settings.visit_score_low_negative_weight
                    + eligible_rows["normalized_low_crowding_complaint_rate"] * settings.visit_score_low_crowding_weight
                )
                metrics_by_group = eligible_rows.set_index("group_key")
                for column_name in [
                    "normalized_average_rating",
                    "normalized_low_negative_share",
                    "normalized_low_crowding_complaint_rate",
                    "visit_score",
                ]:
                    grouped[column_name] = grouped["group_key"].map(metrics_by_group[column_name])
                metric_column = "visit_score"
                score_definition = (
                    "visit_score = "
                    f"{settings.visit_score_rating_weight:.2f} x normalized average rating + "
                    f"{settings.visit_score_low_negative_weight:.2f} x normalized low negative-review share + "
                    f"{settings.visit_score_low_crowding_weight:.2f} x normalized low crowding-complaint rate"
                )
            else:
                metric_column = ranking_criterion

            sort_ascending = not higher_is_better
            ranked = eligible_rows.sort_values(
                by=[metric_column, "review_count"],
                ascending=[sort_ascending, False],
            ).head(2)
            for _, row in ranked.iterrows():
                selected_candidates.append(
                    CandidateGroup(
                        comparison_dimension=plan.comparison_dimension or "month",
                        value=row["group_key"],
                        branch=plan.branch,
                        month=int(row["month"]) if pd.notna(row.get("month")) else None,
                        season=row.get("season"),
                        metric_name=metric_column,
                        metric_value=_safe_round(row[metric_column], 4),
                        retrieval_filters={
                            "branch": plan.branch,
                            "reviewer_location": plan.reviewer_location,
                            "month": int(row["month"]) if pd.notna(row.get("month")) else None,
                            "season": row.get("season"),
                        },
                    )
                )
    metrics = _core_metrics(base_df)
    recommended_retrieval_filters = [candidate.retrieval_filters for candidate in selected_candidates if candidate.retrieval_filters]

    comparison_rows = [
        TimeComparisonRow(
            comparison_dimension=row["comparison_dimension"],
            group_key=row["group_key"],
            month=int(row["month"]) if pd.notna(row.get("month")) else None,
            month_name=row.get("month_name"),
            season=row.get("season"),
            review_count=int(row["review_count"]),
            eligible=bool(row["eligible"]),
            average_rating=_safe_round(row["average_rating"], 4),
            positive_share=_safe_round(row["positive_share"], 4),
            negative_share=_safe_round(row["negative_share"], 4),
            crowding_complaint_rate=_safe_round(row["crowding_complaint_rate"], 4),
            rating_difference_from_baseline=_safe_round(row["rating_difference_from_baseline"], 4),
            aspect_mention_rate=_safe_round(row["aspect_mention_rate"], 4),
            ranking_metric_value=_safe_round(
                row["visit_score"] if ranking_criterion == "visit_score" else row.get(ranking_criterion),
                4,
            )
            if ranking_criterion
            else None,
            normalized_average_rating=_safe_round(row.get("normalized_average_rating"), 4),
            normalized_low_negative_share=_safe_round(row.get("normalized_low_negative_share"), 4),
            normalized_low_crowding_complaint_rate=_safe_round(
                row.get("normalized_low_crowding_complaint_rate"),
                4,
            ),
            visit_score=_safe_round(row.get("visit_score"), 4),
        )
        for _, row in grouped.iterrows()
    ]

    return AnalyticsResult(
        intent=plan.intent,
        selected_analytics_function="analyze_time_comparison",
        applied_filters={key: value for key, value in plan.model_dump().items() if key in {"branch", "reviewer_location", "month", "season"} and value is not None},
        sample_size=int(len(base_df)),
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        dataset_date_coverage=_date_coverage(base_df),
        warnings=warnings,
        comparison_rows=comparison_rows,
        selected_candidates=selected_candidates,
        recommended_retrieval_filters=recommended_retrieval_filters,
        ranking_criterion=ranking_criterion,
        score_definition=score_definition,
    )


def analyze_branch_comparison(plan: QueryPlan, df: pd.DataFrame, settings: Settings | None = None) -> AnalyticsResult:
    warnings: list[str] = []
    filtered_df = _apply_plan_filters(df, plan, ignore_dimension_filter=True)
    primary_aspect = plan.requested_aspects[0] if plan.requested_aspects else None
    aspect_mask = _aspect_mask(filtered_df, primary_aspect) if primary_aspect in ASPECT_KEYWORDS else pd.Series(False, index=filtered_df.index)
    crowding_mask = _crowding_complaint_mask(filtered_df)
    branch_working_df = filtered_df.assign(
        is_positive=filtered_df["Rating_Sentiment"].eq("positive").astype(int),
        is_negative=filtered_df["Rating_Sentiment"].eq("negative").astype(int),
        crowding_complaint=crowding_mask.astype(int),
        aspect_match=aspect_mask.astype(int),
        aspect_negative=(aspect_mask & filtered_df["Rating_Sentiment"].isin(["negative", "neutral"])).astype(int),
    )
    grouped = (
        branch_working_df.groupby("Branch", observed=True)
        .agg(
            review_count=("Review_ID", "size"),
            average_rating=("Rating", "mean"),
            positive_share=("is_positive", "mean"),
            negative_share=("is_negative", "mean"),
            crowding_complaint_rate=("crowding_complaint", "mean"),
            aspect_mention_rate=("aspect_match", "mean"),
            aspect_negative_share=("aspect_negative", "mean"),
        )
        .reset_index()
    )
    grouped["positive_share"] = grouped["positive_share"] * 100
    grouped["negative_share"] = grouped["negative_share"] * 100
    grouped["crowding_complaint_rate"] = grouped["crowding_complaint_rate"] * 100
    grouped["aspect_mention_rate"] = grouped["aspect_mention_rate"] * 100
    grouped["aspect_negative_share"] = grouped["aspect_negative_share"] * 100

    _add_sample_warnings(warnings, len(filtered_df))

    ranking_criterion = None
    selected_candidates: list[CandidateGroup] = []
    if plan.ranking_direction and not grouped.empty:
        if primary_aspect and "complaint" in " ".join(plan.required_metrics):
            ranking_criterion = "aspect_negative_share"
            ascending = plan.ranking_direction not in {"best", "highest"}
        elif primary_aspect and plan.ranking_direction in {"highest", "lowest"}:
            ranking_criterion = "aspect_mention_rate"
            ascending = plan.ranking_direction in {"lowest"}
        else:
            ranking_criterion = "average_rating"
            ascending = plan.ranking_direction in {"lowest", "worst"}
        ranked = grouped.sort_values(
            by=[ranking_criterion, "negative_share", "positive_share"],
            ascending=[ascending, True, False],
        ).head(1)
        for _, row in ranked.iterrows():
            selected_candidates.append(
                CandidateGroup(
                    comparison_dimension="branch",
                    value=row["Branch"],
                    branch=row["Branch"],
                    metric_name=ranking_criterion,
                    metric_value=_safe_round(row[ranking_criterion], 4),
                    retrieval_filters={
                        "branch": row["Branch"],
                        "reviewer_location": plan.reviewer_location,
                        "month": plan.month,
                        "season": plan.season,
                    },
                )
            )

    comparison_rows = [
        BranchComparisonRow(
            branch=row["Branch"],
            review_count=int(row["review_count"]),
            average_rating=_safe_round(row["average_rating"], 4),
            positive_share=_safe_round(row["positive_share"], 4),
            negative_share=_safe_round(row["negative_share"], 4),
            crowding_complaint_rate=_safe_round(row["crowding_complaint_rate"], 4),
            aspect_mention_rate=_safe_round(row["aspect_mention_rate"], 4),
            aspect_negative_share=_safe_round(row["aspect_negative_share"], 4),
            ranking_metric_value=_safe_round(row[ranking_criterion], 4) if ranking_criterion else None,
        )
        for _, row in grouped.sort_values("average_rating", ascending=False).iterrows()
    ]

    return AnalyticsResult(
        intent=plan.intent,
        selected_analytics_function="analyze_branch_comparison",
        applied_filters={key: value for key, value in plan.model_dump().items() if key in {"reviewer_location", "month", "season"} and value is not None},
        sample_size=int(len(filtered_df)),
        metrics=_core_metrics(filtered_df),
        baseline_metrics={},
        dataset_date_coverage=_date_coverage(filtered_df),
        warnings=warnings,
        comparison_rows=comparison_rows,
        selected_candidates=selected_candidates,
        recommended_retrieval_filters=[candidate.retrieval_filters for candidate in selected_candidates],
        ranking_criterion=ranking_criterion,
    )


def analyze_general(plan: QueryPlan, df: pd.DataFrame, settings: Settings | None = None) -> AnalyticsResult:
    filtered_df = _apply_plan_filters(df, plan)
    warnings: list[str] = []
    _add_sample_warnings(warnings, len(filtered_df))

    return AnalyticsResult(
        intent=plan.intent,
        selected_analytics_function="analyze_general",
        applied_filters={key: value for key, value in plan.model_dump().items() if key in {"branch", "reviewer_location", "month", "season"} and value is not None},
        sample_size=int(len(filtered_df)),
        metrics=_core_metrics(filtered_df),
        baseline_metrics={},
        dataset_date_coverage=_date_coverage(filtered_df),
        warnings=warnings,
        recommended_retrieval_filters=[
            {key: value for key, value in plan.model_dump().items() if key in {"branch", "reviewer_location", "month", "season"} and value is not None}
        ],
    )


def run_analytics(plan: QueryPlan, df: pd.DataFrame, settings: Settings | None = None) -> AnalyticsResult:
    settings = settings or get_settings()
    if plan.intent == "segment_summary":
        return analyze_segment(plan, df, settings)
    if plan.intent == "aspect_assessment":
        return analyze_aspect(plan, df, settings)
    if plan.intent == "time_comparison":
        return analyze_time_comparison(plan, df, settings)
    if plan.intent == "branch_comparison":
        return analyze_branch_comparison(plan, df, settings)
    return analyze_general(plan, df, settings)
