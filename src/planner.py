from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate

from .config import Settings, get_settings
from .schemas import PlanValidationResult, QueryPlan, RetrievalTask

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - optional dependency
    ChatOpenAI = None


BRANCH_LABELS = {
    "Disneyland_California": "California",
    "Disneyland_Paris": "Paris",
    "Disneyland_HongKong": "Hong Kong",
}

BRANCH_ALIASES = {
    "Disneyland_California": [
        "disneyland california",
        "california disneyland",
        "disneyland anaheim",
        "anaheim disneyland",
        "la disneyland",
        "disneyland la",
        "california",
        "anaheim",
    ],
    "Disneyland_Paris": [
        "disneyland paris",
        "paris disneyland",
        "disney paris",
        "paris",
    ],
    "Disneyland_HongKong": [
        "disneyland hong kong",
        "hong kong disneyland",
        "hong kong park",
        "hk disneyland",
        "hong kong",
        "hk",
    ],
}

MONTH_NAME_TO_NUMBER = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
NUMBER_TO_MONTH_NAME = {value: key.title() for key, value in MONTH_NAME_TO_NUMBER.items()}
SEASON_ALIASES = {
    "winter": "Winter",
    "spring": "Spring",
    "summer": "Summer",
    "autumn": "Autumn",
    "fall": "Autumn",
}

ASPECT_KEYWORDS = {
    "crowding": [
        "crowd",
        "crowded",
        "queue",
        "queues",
        "line",
        "lines",
        "wait",
        "waiting",
        "busy",
        "packed",
    ],
    "staff": [
        "staff",
        "service",
        "cast member",
        "cast members",
        "employee",
        "employees",
        "friendly",
        "helpful",
        "rude",
    ],
    "cleanliness": ["clean", "cleanliness", "dirty", "bathroom", "restroom", "toilet"],
    "food": ["food", "restaurant", "meal", "meals", "snack", "dining", "eat", "coffee"],
    "value": ["price", "prices", "expensive", "cheap", "value", "cost", "overpriced", "worth"],
    "rides": [
        "ride",
        "rides",
        "attraction",
        "attractions",
        "roller coaster",
        "fastpass",
        "lightning lane",
        "closure",
        "closed rides",
    ],
    "maintenance": ["maintenance", "renovation", "construction", "closed", "closure", "broken"],
    "cleanliness": ["clean", "cleanliness", "dirty", "toilet", "bathroom", "restroom"],
    "family": ["family", "families", "kids", "children", "child", "daughter", "son", "toddler"],
    "general_experience": ["experience", "trip", "visit", "overall"],
}

SUPPORTED_ASPECTS = set(ASPECT_KEYWORDS)

COMMON_LOCATION_ALIASES = {
    "Australia": ["australia", "australian", "australians"],
    "United Kingdom": ["united kingdom", "uk", "britain", "british", "england"],
    "United States": ["united states", "us", "usa", "american", "americans"],
    "Canada": ["canada", "canadian", "canadians"],
}

EXTERNAL_CONTEXT_KEYWORDS = {
    "weather": ["weather", "temperature", "rain", "rainy", "humid", "climate"],
    "holidays": ["holiday", "holidays", "public holiday", "public holidays"],
    "ticket_prices": ["ticket price", "ticket prices", "price", "prices", "cheapest", "cheap tickets"],
    "current_operations": ["current operations", "currently open", "open now", "operating now"],
    "live_crowds": ["current crowd", "current crowds", "live crowd", "live crowds", "today crowd"],
}

INTENT_METRICS = {
    "segment_summary": [
        "review_count",
        "average_rating",
        "median_rating",
        "positive_share",
        "neutral_share",
        "negative_share",
    ],
    "aspect_assessment": [
        "review_count",
        "average_rating",
        "positive_share",
        "negative_share",
        "aspect_mention_rate",
        "rating_difference_from_baseline",
    ],
    "time_comparison": [
        "review_count",
        "average_rating",
        "positive_share",
        "negative_share",
        "crowding_complaint_rate",
        "rating_difference_from_baseline",
    ],
    "branch_comparison": [
        "review_count",
        "average_rating",
        "positive_share",
        "negative_share",
        "crowding_complaint_rate",
    ],
    "general_search": [
        "review_count",
        "average_rating",
        "positive_share",
        "negative_share",
    ],
}


def normalize_text(text: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


@dataclass
class DatasetLookups:
    locations: list[str]
    location_aliases: list[tuple[str, str]]


def build_dataset_lookups(df: pd.DataFrame) -> DatasetLookups:
    locations = (
        df["Reviewer_Location"]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda values: values.ne("")]
        .drop_duplicates()
        .tolist()
    )
    alias_map: dict[str, str] = {}
    for location in locations:
        normalized = normalize_text(location)
        if normalized:
            alias_map[normalized] = location
        for canonical_location, aliases in COMMON_LOCATION_ALIASES.items():
            if normalize_text(canonical_location) == normalized:
                for alias in aliases:
                    alias_map[normalize_text(alias)] = location
    ordered_aliases = sorted(alias_map.items(), key=lambda item: len(item[0]), reverse=True)
    return DatasetLookups(locations=locations, location_aliases=ordered_aliases)


def normalize_branch_value(value: object) -> str | None:
    if value is None:
        return None
    raw_value = str(value).strip()
    if raw_value in BRANCH_LABELS:
        return raw_value
    normalized_value = normalize_text(raw_value)
    for branch, aliases in BRANCH_ALIASES.items():
        if normalized_value == normalize_text(BRANCH_LABELS[branch]):
            return branch
        if normalized_value in {normalize_text(alias) for alias in aliases}:
            return branch
    return None


def normalize_location_value(value: object, lookups: DatasetLookups) -> str | None:
    if value is None:
        return None
    normalized_value = normalize_text(value)
    if not normalized_value:
        return None
    for alias, location in lookups.location_aliases:
        if alias == normalized_value:
            return location
    for alias, location in lookups.location_aliases:
        if f" {alias} " in f" {normalized_value} ":
            return location
    return None


def normalize_month_value(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value if 1 <= value <= 12 else None
    raw_value = str(value).strip()
    if raw_value.isdigit():
        month = int(raw_value)
        return month if 1 <= month <= 12 else None
    normalized_value = normalize_text(raw_value)
    return MONTH_NAME_TO_NUMBER.get(normalized_value)


def normalize_season_value(value: object) -> str | None:
    if value is None:
        return None
    normalized_value = normalize_text(value)
    return SEASON_ALIASES.get(normalized_value)


def extract_month_numbers(question: str) -> list[int]:
    matches: list[tuple[int, int]] = []
    for name, number in MONTH_NAME_TO_NUMBER.items():
        for match in re.finditer(rf"\b{name}\b", question.lower()):
            matches.append((match.start(), number))
    return [number for _, number in sorted(matches)]


def extract_seasons(question: str) -> list[str]:
    matches: list[tuple[int, str]] = []
    for alias, season in SEASON_ALIASES.items():
        for match in re.finditer(rf"\b{alias}\b", question.lower()):
            matches.append((match.start(), season))
    seen: list[str] = []
    for _, season in sorted(matches):
        if season not in seen:
            seen.append(season)
    return seen


def detect_requested_aspects(question: str) -> list[str]:
    normalized_question = normalize_text(question)
    scores: list[tuple[int, str]] = []
    for aspect, keywords in ASPECT_KEYWORDS.items():
        if aspect == "general_experience":
            continue
        score = sum(1 for keyword in keywords if normalize_text(keyword) in normalized_question)
        if score > 0:
            scores.append((score, aspect))
    scores.sort(key=lambda item: (-item[0], item[1]))
    return [aspect for _, aspect in scores]


def detect_external_context(question: str) -> list[str]:
    normalized_question = normalize_text(question)
    detected: list[str] = []
    for context_name, keywords in EXTERNAL_CONTEXT_KEYWORDS.items():
        if any(normalize_text(keyword) in normalized_question for keyword in keywords):
            detected.append(context_name)
    return detected


def detect_ranking_direction(question: str) -> str | None:
    normalized_question = normalize_text(question)
    if any(token in normalized_question for token in ["best", "strongest", "highest", "most satisfied"]):
        return "best" if "best" in normalized_question or "strongest" in normalized_question else "highest"
    if any(token in normalized_question for token in ["worst", "lowest", "weakest", "fewest"]):
        return "lowest" if any(token in normalized_question for token in ["lowest", "fewest"]) else "worst"
    return None


def infer_intent(
    question: str,
    *,
    branch: str | None,
    reviewer_location: str | None,
    months: list[int],
    seasons: list[str],
    requested_aspects: list[str],
    ranking_direction: str | None,
) -> str:
    normalized_question = normalize_text(question)
    mentions_compare = any(
        phrase in normalized_question
        for phrase in ["compare", "compared", "versus", "vs", "better than", "worse than"]
    )
    mentions_time = bool(months or seasons) or any(
        phrase in normalized_question
        for phrase in [
            "time of year",
            "time of the year",
            "time to visit",
            "month",
            "season",
            "when should",
            "when is",
        ]
    )
    mentions_branch_comparison = any(
        phrase in normalized_question
        for phrase in ["which park", "which branch", "which disneyland", "across parks"]
    )

    if mentions_branch_comparison:
        return "branch_comparison"
    if mentions_time and (mentions_compare or ranking_direction or len(seasons) > 1 or len(months) > 1):
        return "time_comparison"
    if mentions_time and ranking_direction:
        return "time_comparison"
    if mentions_time and any(phrase in normalized_question for phrase in ["good time", "best time", "visit"]):
        return "time_comparison"
    if requested_aspects and any(
        phrase in normalized_question
        for phrase in ["is the", "is ", "are the", "what do visitors dislike", "complain", "friendly", "crowded"]
    ):
        return "aspect_assessment"
    if reviewer_location or any(
        phrase in normalized_question
        for phrase in ["what do", "how satisfied", "what are visitors saying", "say about"]
    ):
        return "segment_summary"
    if branch and requested_aspects:
        return "aspect_assessment"
    return "general_search"


def infer_required_metrics(
    intent: str,
    *,
    question: str,
    requested_aspects: list[str],
) -> list[str]:
    normalized_question = normalize_text(question)
    metrics = list(INTENT_METRICS[intent])
    if "fewest crowding complaints" in normalized_question or "least crowded" in normalized_question:
        metrics = ["review_count", "crowding_complaint_rate", "average_rating", "negative_share"]
    elif "highest rating" in normalized_question or "highest-rated" in normalized_question:
        metrics = ["review_count", "average_rating", "positive_share", "negative_share"]
    elif requested_aspects and intent == "aspect_assessment":
        metrics = [
            "review_count",
            "average_rating",
            "positive_share",
            "negative_share",
            "aspect_mention_rate",
            "rating_difference_from_baseline",
        ]
    return metrics


def build_retrieval_tasks(
    question: str,
    *,
    intent: str,
    branch: str | None,
    reviewer_location: str | None,
    month: int | None,
    season: str | None,
    requested_aspects: list[str],
) -> list[RetrievalTask]:
    tasks: list[RetrievalTask] = []

    def add_task(query: str, purpose: str, rating_filter: str = "all") -> None:
        if len(tasks) >= get_settings().max_planner_retrieval_tasks:
            return
        tasks.append(
            RetrievalTask(
                query=query,
                purpose=purpose,
                branch=branch,
                reviewer_location=reviewer_location,
                month=month,
                season=season,
                rating_filter=rating_filter,
                top_k=4,
            )
        )

    if intent == "segment_summary":
        add_task(question, "general_experience")
        add_task("positive highlights and praise", "supporting_praise", "positive")
        add_task("complaints, frustrations, and tradeoffs", "supporting_complaints", "negative")
    elif intent == "aspect_assessment":
        primary_aspect = requested_aspects[0] if requested_aspects else "general_experience"
        purpose = primary_aspect if primary_aspect in SUPPORTED_ASPECTS else "general_experience"
        add_task(question, purpose)
        add_task(f"{primary_aspect} praise and positive comments", "supporting_praise", "positive")
        add_task(f"{primary_aspect} complaints and tradeoffs", "tradeoffs", "negative")
    elif intent == "time_comparison":
        add_task("positive experiences in the strongest period", "supporting_praise", "positive")
        add_task("complaints and tradeoffs in the strongest period", "tradeoffs", "negative")
        if "crowding" in requested_aspects or "crowded" in normalize_text(question):
            add_task("crowding, lines, queues, and waiting", "crowding")
    elif intent == "branch_comparison":
        add_task("positive experiences that represent the strongest park", "supporting_praise", "positive")
        add_task("complaints that explain weaker performance", "supporting_complaints", "negative")
        if requested_aspects:
            primary_aspect = requested_aspects[0]
            purpose = primary_aspect if primary_aspect in SUPPORTED_ASPECTS else "general_experience"
            add_task(f"{primary_aspect} examples across parks", purpose)
    else:
        add_task(question, "general_experience")
    return tasks[: get_settings().max_planner_retrieval_tasks]


class DeterministicFallbackPlanner:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def plan(self, question: str, df: pd.DataFrame) -> QueryPlan:
        lookups = build_dataset_lookups(df)
        normalized_question = f" {normalize_text(question)} "

        branch = None
        matched_branch_aliases: list[str] = []
        for candidate_branch, aliases in BRANCH_ALIASES.items():
            normalized_aliases = [normalize_text(alias) for alias in aliases]
            if any(f" {alias} " in normalized_question for alias in normalized_aliases):
                branch = candidate_branch
                matched_branch_aliases.extend(normalized_aliases)
                break

        location_search_text = normalized_question
        for alias in matched_branch_aliases:
            location_search_text = location_search_text.replace(f" {alias} ", " ")

        reviewer_location = None
        for alias, location in lookups.location_aliases:
            if alias and f" {alias} " in location_search_text:
                reviewer_location = location
                break

        months = extract_month_numbers(question)
        seasons = extract_seasons(question)
        requested_aspects = detect_requested_aspects(question)
        ranking_direction = detect_ranking_direction(question)
        intent = infer_intent(
            question,
            branch=branch,
            reviewer_location=reviewer_location,
            months=months,
            seasons=seasons,
            requested_aspects=requested_aspects,
            ranking_direction=ranking_direction,
        )

        comparison_dimension = None
        compare_values: list[str] = []
        month = months[0] if len(months) == 1 else None
        season = seasons[0] if len(seasons) == 1 else None

        if intent == "time_comparison":
            if len(seasons) > 1 or "summer" in normalize_text(question) and "winter" in normalize_text(question):
                comparison_dimension = "season"
                compare_values = seasons
                season = None
            elif len(months) > 1:
                comparison_dimension = "month"
                compare_values = [NUMBER_TO_MONTH_NAME[number] for number in months]
                month = None
            elif any(phrase in normalize_text(question) for phrase in ["season", "spring", "summer", "autumn", "fall", "winter"]):
                comparison_dimension = "season"
                if seasons:
                    compare_values = seasons
                    season = None
            else:
                comparison_dimension = "month"

        if intent == "branch_comparison":
            comparison_dimension = "branch"

        required_metrics = infer_required_metrics(
            intent,
            question=question,
            requested_aspects=requested_aspects,
        )
        external_context_requested = detect_external_context(question)
        retrieval_tasks = build_retrieval_tasks(
            question,
            intent=intent,
            branch=branch,
            reviewer_location=reviewer_location,
            month=month,
            season=season,
            requested_aspects=requested_aspects,
        )

        return QueryPlan(
            intent=intent,
            branch=branch,
            reviewer_location=reviewer_location,
            month=month,
            season=season,
            comparison_dimension=comparison_dimension,
            compare_values=compare_values,
            requested_aspects=requested_aspects,
            required_metrics=required_metrics,
            ranking_direction=ranking_direction,
            retrieval_tasks=retrieval_tasks,
            external_context_requested=external_context_requested,
            clarification_needed=False,
            clarification_reason=None,
        )


class AdaptivePlanner:
    def __init__(self, settings: Settings | None = None, *, use_llm: bool = True) -> None:
        self.settings = settings or get_settings()
        self.use_llm = use_llm
        self.fallback = DeterministicFallbackPlanner(self.settings)
        self._llm_chain = self._build_llm_chain()

    def _build_llm_chain(self):
        if (
            not self.use_llm
            or not self.settings.openai_api_key
            or ChatOpenAI is None
        ):
            return None

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You plan Disneyland review questions using only the allowed schema values.\n"
                        "Return a structured QueryPlan.\n"
                        "Rules:\n"
                        "- Use only the allowed intent values.\n"
                        "- Treat best/worst/which month/compare questions as analytical, not generic retrieval.\n"
                        "- Preserve explicit comparisons like Summer vs Winter in compare_values.\n"
                        "- Use month comparison by default for generic best-time questions.\n"
                        "- Keep retrieval_tasks to at most four non-redundant tasks.\n"
                        "- Flag unsupported requests like weather, holidays, prices, current operations, and live crowds in external_context_requested.\n"
                        "- Do not invent metrics, branches, reviewer locations, or results.\n"
                    ),
                ),
                (
                    "human",
                    (
                        "Question: {question}\n"
                        "Known branches: {branches}\n"
                        "Known seasons: Winter, Spring, Summer, Autumn\n"
                        "Important note: reviewer_location should be a best-effort normalized country/location phrase if clearly present.\n"
                    ),
                ),
            ]
        )
        llm = ChatOpenAI(
            model=self.settings.planner_model,
            temperature=self.settings.planner_temperature,
            use_responses_api=True,
            reasoning_effort="low",
            verbosity="low",
        )
        return prompt | llm.with_structured_output(QueryPlan)

    def _plan_with_llm(self, question: str, df: pd.DataFrame) -> QueryPlan:
        if self._llm_chain is None:
            raise RuntimeError("LLM planner is unavailable")
        branches = ", ".join(BRANCH_LABELS)
        return self._llm_chain.invoke({"question": question, "branches": branches})

    def plan_without_fallback(self, question: str, df: pd.DataFrame) -> QueryPlan:
        if self.use_llm:
            return self._plan_with_llm(question, df)
        return self.fallback.plan(question, df)

    def plan_raw(self, question: str, df: pd.DataFrame) -> tuple[QueryPlan, str, list[str]]:
        planner_source = "deterministic_fallback"
        warnings: list[str] = []
        try:
            raw_plan = self._plan_with_llm(question, df)
            planner_source = "llm"
        except Exception:
            raw_plan = self.fallback.plan(question, df)
            warnings.append("LLM planner unavailable or failed, so deterministic fallback planning was used.")
        return raw_plan, planner_source, warnings

    def normalize_plan(self, raw_plan: QueryPlan, df: pd.DataFrame) -> tuple[QueryPlan, list[str]]:
        return self.normalize_plan_with_context(raw_plan, df)

    def normalize_plan_with_context(
        self,
        raw_plan: QueryPlan,
        df: pd.DataFrame,
        *,
        question: str | None = None,
    ) -> tuple[QueryPlan, list[str]]:
        heuristic_plan = self.fallback.plan(question, df) if question else None
        return normalize_and_validate_plan(
            raw_plan,
            df,
            settings=self.settings,
            heuristic_plan=heuristic_plan,
            question=question,
        )

    def plan(self, question: str, df: pd.DataFrame) -> PlanValidationResult:
        raw_plan, planner_source, warnings = self.plan_raw(question, df)

        normalized_plan, normalization_warnings = self.normalize_plan_with_context(
            raw_plan,
            df,
            question=question,
        )
        warnings.extend(normalization_warnings)
        return PlanValidationResult(
            question=question,
            planner_source=planner_source,
            raw_plan=raw_plan,
            normalized_plan=normalized_plan,
            warnings=warnings,
        )


def normalize_and_validate_plan(
    plan: QueryPlan,
    df: pd.DataFrame,
    *,
    settings: Settings | None = None,
    heuristic_plan: QueryPlan | None = None,
    question: str | None = None,
) -> tuple[QueryPlan, list[str]]:
    settings = settings or get_settings()
    lookups = build_dataset_lookups(df)
    warnings: list[str] = []

    fallback_branch = heuristic_plan.branch if heuristic_plan else None
    fallback_location = heuristic_plan.reviewer_location if heuristic_plan else None
    fallback_month = heuristic_plan.month if heuristic_plan else None
    fallback_season = heuristic_plan.season if heuristic_plan else None
    fallback_comparison_dimension = heuristic_plan.comparison_dimension if heuristic_plan else None
    fallback_compare_values = heuristic_plan.compare_values if heuristic_plan else []
    fallback_requested_aspects = heuristic_plan.requested_aspects if heuristic_plan else []
    fallback_ranking_direction = heuristic_plan.ranking_direction if heuristic_plan else None

    branch = normalize_branch_value(plan.branch) or normalize_branch_value(fallback_branch)
    if plan.branch and branch is None:
        warnings.append(f"Dropped unknown branch value: {plan.branch}")

    reviewer_location = normalize_location_value(plan.reviewer_location, lookups) or normalize_location_value(
        fallback_location,
        lookups,
    )
    if plan.reviewer_location and reviewer_location is None:
        warnings.append(f"Dropped unknown reviewer location: {plan.reviewer_location}")

    month = normalize_month_value(plan.month)
    if month is None:
        month = normalize_month_value(fallback_month)
    if plan.month is not None and month is None:
        warnings.append(f"Dropped invalid month value: {plan.month}")

    season = normalize_season_value(plan.season)
    if season is None:
        season = normalize_season_value(fallback_season)
    if plan.season and season is None:
        warnings.append(f"Dropped invalid season value: {plan.season}")

    comparison_dimension = plan.comparison_dimension or fallback_comparison_dimension
    if comparison_dimension not in {None, "month", "season", "branch"}:
        comparison_dimension = None
        warnings.append("Dropped invalid comparison_dimension value.")
    normalized_question = normalize_text(question) if question else ""
    explicit_season_request = any(
        token in normalized_question for token in ["season", "spring", "summer", "autumn", "fall", "winter"]
    )
    force_month_default = (
        heuristic_plan is not None
        and comparison_dimension == "season"
        and fallback_comparison_dimension == "month"
        and month is None
        and season is None
        and (plan.ranking_direction or fallback_ranking_direction) is not None
        and not explicit_season_request
    )
    if force_month_default:
        comparison_dimension = "month"

    normalized_intent = plan.intent
    if (
        normalized_intent == "general_search"
        and reviewer_location is not None
        and comparison_dimension is None
        and month is None
        and season is None
        and not plan.requested_aspects
        and plan.ranking_direction is None
    ):
        normalized_intent = "segment_summary"
    if normalized_intent == "general_search" and heuristic_plan and heuristic_plan.intent != "general_search":
        normalized_intent = heuristic_plan.intent

    compare_values: list[str] = []
    source_compare_values = fallback_compare_values if force_month_default else (plan.compare_values or fallback_compare_values)
    for value in source_compare_values:
        normalized_value: str | None = None
        if comparison_dimension == "month":
            month_number = normalize_month_value(value)
            if month_number is not None:
                normalized_value = NUMBER_TO_MONTH_NAME[month_number]
        elif comparison_dimension == "season":
            normalized_value = normalize_season_value(value)
        elif comparison_dimension == "branch":
            normalized_branch = normalize_branch_value(value)
            if normalized_branch is not None:
                normalized_value = normalized_branch
        elif season is not None:
            normalized_value = normalize_season_value(value)
        elif month is not None:
            month_number = normalize_month_value(value)
            if month_number is not None:
                normalized_value = NUMBER_TO_MONTH_NAME[month_number]

        if normalized_value is None:
            warnings.append(f"Dropped invalid compare value: {value}")
            continue
        if normalized_value not in compare_values:
            compare_values.append(normalized_value)

    requested_aspect_source = plan.requested_aspects or fallback_requested_aspects
    requested_aspects = [aspect for aspect in requested_aspect_source if aspect in SUPPORTED_ASPECTS]
    dropped_aspects = sorted(set(plan.requested_aspects).difference(requested_aspects))
    if dropped_aspects:
        warnings.append(f"Dropped unsupported aspects: {', '.join(dropped_aspects)}")

    required_metrics = list(dict.fromkeys(plan.required_metrics))
    if not required_metrics:
        required_metrics = infer_required_metrics(
            plan.intent,
            question="",
            requested_aspects=requested_aspects,
        )
    ranking_direction = plan.ranking_direction or fallback_ranking_direction

    task_source = heuristic_plan.retrieval_tasks if (force_month_default and heuristic_plan) else plan.retrieval_tasks
    normalized_tasks: list[RetrievalTask] = []
    seen_tasks: set[tuple[Any, ...]] = set()
    for task in task_source[: settings.max_planner_retrieval_tasks]:
        task_branch = normalize_branch_value(task.branch) if task.branch else branch
        task_location = normalize_location_value(task.reviewer_location, lookups) if task.reviewer_location else reviewer_location
        task_month = normalize_month_value(task.month)
        task_season = normalize_season_value(task.season)
        top_k = min(max(task.top_k, settings.min_retrieval_top_k), settings.max_retrieval_top_k)
        normalized_task = RetrievalTask(
            query=task.query.strip() or "general experience",
            purpose=task.purpose,
            branch=task_branch,
            reviewer_location=task_location,
            month=task_month,
            season=task_season,
            rating_filter=task.rating_filter,
            top_k=top_k,
        )
        dedupe_key = (
            normalize_text(normalized_task.query),
            normalized_task.purpose,
            normalized_task.branch,
            normalized_task.reviewer_location,
            normalized_task.month,
            normalized_task.season,
            normalized_task.rating_filter,
        )
        if dedupe_key in seen_tasks:
            continue
        seen_tasks.add(dedupe_key)
        normalized_tasks.append(normalized_task)

    external_context_requested = []
    for value in plan.external_context_requested:
        if value not in external_context_requested:
            external_context_requested.append(value)

    normalized_plan = QueryPlan(
        intent=normalized_intent,
        branch=branch,
        reviewer_location=reviewer_location,
        month=month,
        season=season,
        comparison_dimension=comparison_dimension,
        compare_values=compare_values,
        requested_aspects=requested_aspects,
        required_metrics=required_metrics,
        ranking_direction=ranking_direction,
        retrieval_tasks=normalized_tasks,
        external_context_requested=external_context_requested,
        clarification_needed=plan.clarification_needed,
        clarification_reason=plan.clarification_reason,
    )
    return normalized_plan, warnings
