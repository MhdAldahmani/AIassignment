from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ConfigDict, Field

from .config import Settings, get_settings
from .planner import MONTH_NAME_TO_NUMBER
from .qa import QAEngine, QAEngineResult
from .schemas import AnalyticsResult, QueryPlan, RetrievedEvidence

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - optional dependency
    ChatOpenAI = None


EXPECTED_ANALYTICS_ROUTE = {
    "segment_summary": "analyze_segment",
    "aspect_assessment": "analyze_aspect",
    "time_comparison": "analyze_time_comparison",
    "branch_comparison": "analyze_branch_comparison",
    "general_search": "analyze_general",
}


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    category: str
    question: str
    expected_intent: str
    expected_branch: str | None = None
    expected_reviewer_location: str | None = None
    expected_comparison_dimension: str | None = None
    expected_compare_values: list[str] = Field(default_factory=list)
    must_use_analytics: bool = True
    must_flag_external_context: bool = False
    must_not_claim_external_facts: bool = True


class JudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relevance: int
    faithfulness: int
    completeness: int
    clarity: int
    limitation_handling: int
    external_knowledge_leak: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    explanation: str


JUDGE_SYSTEM_PROMPT = """Evaluate the answer only against the supplied dataset statistics,
retrieved evidence, and expected test behavior.

Do not use your own knowledge about Disneyland, tourism, weather,
holidays, prices, or park operations.

A plausible or fluent answer is not necessarily correct.

Flag:
- Unsupported numbers
- Claims not supported by evidence
- External knowledge
- Missing limitations
- Excessive certainty
- Failure to answer the actual question
"""


def load_evaluation_cases(path: str | Path = "evals/evaluation_cases.json") -> list[EvaluationCase]:
    raw = json.loads(Path(path).read_text())
    return [EvaluationCase(**item) for item in raw]


def _answer_text(result: QAEngineResult) -> str:
    return result.answer or ""


def _normalize_number_token(token: str) -> str:
    return token.replace(",", "").strip()


def _number_tokens(text: str) -> list[str]:
    return [_normalize_number_token(token) for token in re.findall(r"(?<!/)-?\d[\d,]*(?:\.\d+)?", text)]


def _compare_values_match(plan: QueryPlan | None, expected_values: list[str]) -> bool:
    if plan is None:
        return False
    if not expected_values:
        return True
    return plan.compare_values == expected_values


def _filters_correct(case: EvaluationCase, plan: QueryPlan | None) -> tuple[bool, dict[str, bool]]:
    checks = {
        "branch_correct": case.expected_branch == (plan.branch if plan else None),
        "reviewer_location_correct": case.expected_reviewer_location == (plan.reviewer_location if plan else None),
        "comparison_dimension_correct": case.expected_comparison_dimension == (plan.comparison_dimension if plan else None),
        "compare_values_correct": _compare_values_match(plan, case.expected_compare_values),
    }
    return all(checks.values()), checks


def _allowed_months_from_plan(plan: QueryPlan | None) -> set[int]:
    if plan is None or plan.comparison_dimension != "month":
        return set()
    return {
        MONTH_NAME_TO_NUMBER[value.lower()]
        for value in plan.compare_values
        if value.lower() in MONTH_NAME_TO_NUMBER
    }


def _retrieval_filters_valid(
    plan: QueryPlan | None,
    analytics_result: AnalyticsResult | None,
    evidence: list[RetrievedEvidence],
) -> bool:
    if plan is None or analytics_result is None:
        return False
    if not evidence:
        return True

    allowed_candidate_branches = {
        candidate.branch for candidate in analytics_result.selected_candidates if candidate.branch
    }
    allowed_candidate_months = {
        candidate.month for candidate in analytics_result.selected_candidates if candidate.month is not None
    }
    allowed_candidate_seasons = {
        candidate.season for candidate in analytics_result.selected_candidates if candidate.season
    }

    allowed_compare_months = _allowed_months_from_plan(plan)
    allowed_compare_seasons = set(plan.compare_values) if plan.comparison_dimension == "season" else set()

    for item in evidence:
        if plan.branch and not allowed_candidate_branches and item.branch != plan.branch:
            return False
        if allowed_candidate_branches and item.branch not in allowed_candidate_branches:
            return False
        if plan.reviewer_location and item.reviewer_location != plan.reviewer_location:
            return False
        if plan.month is not None and item.month != plan.month:
            return False
        if plan.season and item.season != plan.season:
            return False
        if allowed_candidate_months and item.month not in allowed_candidate_months:
            return False
        if allowed_candidate_seasons and item.season not in allowed_candidate_seasons:
            return False
        if allowed_compare_months and item.month not in allowed_compare_months:
            return False
        if allowed_compare_seasons and item.season not in allowed_compare_seasons:
            return False
    return True


def _external_context_behavior_correct(case: EvaluationCase, result: QAEngineResult) -> bool:
    if result.plan is None:
        return False
    flagged = bool(result.plan.external_context_requested)
    limitations = " ".join(result.structured_answer.limitations).lower() if result.structured_answer else ""
    answer = f"{_answer_text(result).lower()} {limitations}"
    mentions_unavailable = any(
        token in answer
        for token in [
            "not available",
            "unavailable",
            "unsupported",
            "does not include",
            "cannot",
            "not contain",
        ]
    )
    if case.must_flag_external_context:
        return flagged and mentions_unavailable
    return not flagged


def _limitations_present(result: QAEngineResult) -> bool:
    structured = result.structured_answer
    if structured is None or not structured.limitations:
        return False
    joined = " ".join(structured.limitations).lower()
    analytics_warnings = " ".join(result.analytics_result.warnings).lower() if result.analytics_result else ""
    answer = _answer_text(result).lower()

    limitation_blob = f"{joined} {answer}"
    if "missing" in analytics_warnings and "missing" not in limitation_blob and "excluded" not in limitation_blob:
        return False
    if "small sample" in analytics_warnings and "sample" not in limitation_blob:
        return False
    return True


def _numeric_supporting_metrics_valid(result: QAEngineResult) -> bool:
    if result.structured_answer is None:
        return False
    supporting_text = " ".join(result.structured_answer.supporting_metrics)
    supporting_numbers = [token for token in _number_tokens(supporting_text) if token not in {"5"}]
    if not supporting_numbers:
        return True
    analytics_blob = json.dumps(
        {
            "plan": result.plan.model_dump(mode="json") if result.plan else {},
            "analytics": result.analytics_result.model_dump(mode="json") if result.analytics_result else {},
            "evidence": [item.model_dump(mode="json") for item in result.evidence],
        },
        ensure_ascii=False,
    )
    analytics_numbers = [_normalize_number_token(token) for token in _number_tokens(analytics_blob)]

    def is_supported(token: str) -> bool:
        if token in analytics_numbers:
            return True
        try:
            target = float(token)
        except ValueError:
            return False
        for candidate in analytics_numbers:
            try:
                value = float(candidate)
            except ValueError:
                continue
            if abs(target - value) <= 0.01:
                return True
        return False

    matches = sum(1 for token in supporting_numbers if is_supported(token))
    return matches / len(supporting_numbers) >= 0.75


def evaluate_case_deterministically(case: EvaluationCase, result: QAEngineResult) -> dict[str, Any]:
    if result.error:
        return {
            "runtime_error": True,
            "intent_correct": False,
            "filters_correct": False,
            "analytics_route_correct": False,
            "retrieval_filters_valid": False,
            "retrieved_review_ids_unique": False,
            "external_context_behavior_correct": False,
            "limitations_present": False,
            "numeric_supporting_metrics_valid": False,
            "branch_correct": False,
            "reviewer_location_correct": False,
            "comparison_dimension_correct": False,
            "compare_values_correct": False,
            "deterministic_pass_rate": 0.0,
            "failure_reason": result.error,
        }

    filters_correct, filter_detail = _filters_correct(case, result.plan)
    analytics_route_correct = (
        result.analytics_result is not None
        and result.analytics_result.selected_analytics_function
        == EXPECTED_ANALYTICS_ROUTE.get(case.expected_intent)
    )
    retrieved_ids = [item.review_id for item in result.evidence]
    retrieved_review_ids_unique = len(retrieved_ids) == len(set(retrieved_ids))
    retrieval_filters_valid = _retrieval_filters_valid(result.plan, result.analytics_result, result.evidence)
    external_context_behavior_correct = _external_context_behavior_correct(case, result)
    limitations_present = _limitations_present(result)
    numeric_supporting_metrics_valid = _numeric_supporting_metrics_valid(result)
    intent_correct = result.plan is not None and result.plan.intent == case.expected_intent

    checks = {
        "intent_correct": intent_correct,
        "filters_correct": filters_correct,
        "analytics_route_correct": analytics_route_correct,
        "retrieval_filters_valid": retrieval_filters_valid,
        "retrieved_review_ids_unique": retrieved_review_ids_unique,
        "external_context_behavior_correct": external_context_behavior_correct,
        "limitations_present": limitations_present,
        "numeric_supporting_metrics_valid": numeric_supporting_metrics_valid,
    }
    failure_reasons = [name for name, passed in checks.items() if not passed]
    failure_reasons.extend(name for name, passed in filter_detail.items() if not passed)
    pass_rate = sum(1 for passed in checks.values() if passed) / len(checks)
    return {
        "runtime_error": False,
        **checks,
        **filter_detail,
        "deterministic_pass_rate": round(pass_rate, 3),
        "failure_reason": "; ".join(dict.fromkeys(failure_reasons)),
    }


def build_judge_evaluator(settings: Settings | None = None):
    settings = settings or get_settings()
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not configured.")
    if ChatOpenAI is None:
        raise ImportError("langchain_openai is not installed.")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", JUDGE_SYSTEM_PROMPT),
            (
                "human",
                (
                    "Question: {question}\n\n"
                    "Normalized query plan:\n{plan_json}\n\n"
                    "Deterministic analytics result:\n{analytics_json}\n\n"
                    "Retrieved evidence:\n{evidence_json}\n\n"
                    "Generated answer:\n{answer}\n\n"
                    "Expected test behavior:\n{expected_behavior}\n\n"
                    "Return a JudgeResult."
                ),
            ),
        ]
    )
    llm = ChatOpenAI(
        model=settings.judge_model,
        temperature=settings.judge_temperature,
        use_responses_api=True,
        reasoning_effort="low",
        verbosity="low",
    )
    return prompt | llm.with_structured_output(JudgeResult)


def invoke_judge(judge_evaluator: Any, payload: dict[str, Any]) -> JudgeResult:
    if hasattr(judge_evaluator, "invoke"):
        result = judge_evaluator.invoke(payload)
    else:
        result = judge_evaluator(payload)
    if isinstance(result, JudgeResult):
        return result
    if isinstance(result, dict):
        return JudgeResult(**result)
    raise TypeError("Judge evaluator did not return a JudgeResult-compatible object.")


def evaluate_single_result(
    *,
    question: str,
    result: QAEngineResult,
    expected_behavior: dict[str, Any] | EvaluationCase | None = None,
    judge_evaluator=None,
    settings: Settings | None = None,
) -> JudgeResult:
    settings = settings or get_settings()
    if result.error:
        raise ValueError(f"Cannot judge a failed answer: {result.error}")
    if result.plan is None:
        raise ValueError("Cannot judge a result without a normalized plan.")
    if judge_evaluator is None:
        judge_evaluator = build_judge_evaluator(settings)

    if isinstance(expected_behavior, EvaluationCase):
        case = expected_behavior
    else:
        expected_payload = expected_behavior or {
            "id": "interactive_question",
            "category": "interactive",
            "question": question,
            "expected_intent": result.plan.intent,
            "expected_branch": result.plan.branch,
            "expected_reviewer_location": result.plan.reviewer_location,
            "expected_comparison_dimension": result.plan.comparison_dimension,
            "expected_compare_values": result.plan.compare_values,
            "must_use_analytics": True,
            "must_flag_external_context": bool(result.plan.external_context_requested),
            "must_not_claim_external_facts": True,
        }
        case = EvaluationCase(**expected_payload)
    return invoke_judge(judge_evaluator, _judge_payload(case, result))


def _judge_payload(case: EvaluationCase, result: QAEngineResult) -> dict[str, Any]:
    return {
        "question": case.question,
        "plan_json": json.dumps(result.plan.model_dump(mode="json") if result.plan else {}, indent=2, ensure_ascii=False),
        "analytics_json": json.dumps(
            result.analytics_result.model_dump(mode="json") if result.analytics_result else {},
            indent=2,
            ensure_ascii=False,
        ),
        "evidence_json": json.dumps(
            [item.model_dump(mode="json") for item in result.evidence],
            indent=2,
            ensure_ascii=False,
        ),
        "answer": result.answer or "",
        "expected_behavior": json.dumps(case.model_dump(mode="json"), indent=2, ensure_ascii=False),
    }


def _judge_score(judge_result: JudgeResult) -> float:
    base = (
        judge_result.relevance
        + judge_result.faithfulness
        + judge_result.completeness
        + judge_result.clarity
        + judge_result.limitation_handling
    ) / 5
    if judge_result.external_knowledge_leak:
        base -= 1.0
    return round(max(base, 0.0), 3)


def run_evaluation(
    engine: QAEngine,
    cases: list[EvaluationCase],
    *,
    use_llm_judge: bool = True,
    judge_case_ids: set[str] | None = None,
    judge_evaluator=None,
    export: bool = False,
    csv_path: str | Path = "artifacts/evaluation_results.csv",
    json_path: str | Path = "artifacts/evaluation_results.json",
) -> pd.DataFrame:
    if use_llm_judge and judge_evaluator is None:
        judge_evaluator = build_judge_evaluator(engine.settings)

    rows: list[dict[str, Any]] = []
    for case in cases:
        result = engine.ask(case.question)
        deterministic = evaluate_case_deterministically(case, result)
        row: dict[str, Any] = {
            "case_id": case.id,
            "category": case.category,
            "question": case.question,
            "intent_correct": deterministic["intent_correct"],
            "filters_correct": deterministic["filters_correct"],
            "analytics_route_correct": deterministic["analytics_route_correct"],
            "retrieval_filters_valid": deterministic["retrieval_filters_valid"],
            "retrieved_review_ids_unique": deterministic["retrieved_review_ids_unique"],
            "external_context_behavior_correct": deterministic["external_context_behavior_correct"],
            "limitations_present": deterministic["limitations_present"],
            "numeric_supporting_metrics_valid": deterministic["numeric_supporting_metrics_valid"],
            "deterministic_pass_rate": deterministic["deterministic_pass_rate"],
            "runtime_error": deterministic["runtime_error"],
            "failure_reason": deterministic["failure_reason"],
        }

        should_judge = use_llm_judge and (judge_case_ids is None or case.id in judge_case_ids)
        if should_judge:
            try:
                judge_result = invoke_judge(judge_evaluator, _judge_payload(case, result))
                row.update(
                    {
                        "judge_relevance": judge_result.relevance,
                        "judge_faithfulness": judge_result.faithfulness,
                        "judge_completeness": judge_result.completeness,
                        "judge_clarity": judge_result.clarity,
                        "judge_limitation_handling": judge_result.limitation_handling,
                        "external_knowledge_leak": judge_result.external_knowledge_leak,
                        "unsupported_claims": json.dumps(judge_result.unsupported_claims, ensure_ascii=False),
                        "judge_explanation": judge_result.explanation,
                        "final_score": _judge_score(judge_result),
                    }
                )
                if judge_result.unsupported_claims:
                    prefix = row["failure_reason"] + "; " if row["failure_reason"] else ""
                    row["failure_reason"] = prefix + "unsupported_claims"
            except Exception as exc:
                row.update(
                    {
                        "judge_relevance": None,
                        "judge_faithfulness": None,
                        "judge_completeness": None,
                        "judge_clarity": None,
                        "judge_limitation_handling": None,
                        "external_knowledge_leak": None,
                        "unsupported_claims": None,
                        "judge_explanation": None,
                        "final_score": None,
                    }
                )
                prefix = row["failure_reason"] + "; " if row["failure_reason"] else ""
                row["failure_reason"] = prefix + f"Judge error: {exc}"
        else:
            row.update(
                {
                    "judge_relevance": None,
                    "judge_faithfulness": None,
                    "judge_completeness": None,
                    "judge_clarity": None,
                    "judge_limitation_handling": None,
                    "external_knowledge_leak": None,
                    "unsupported_claims": None,
                    "judge_explanation": None,
                    "final_score": round(deterministic["deterministic_pass_rate"] * 5, 3),
                }
            )
        rows.append(row)

    results_df = pd.DataFrame(rows)
    if export:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(csv_path, index=False)
        results_df.to_json(json_path, orient="records", indent=2)
    return results_df


def summarize_evaluation_results(results_df: pd.DataFrame) -> dict[str, Any]:
    deterministic_pass_rate = round(float(results_df["deterministic_pass_rate"].mean()), 3)
    judge_columns = [
        "judge_relevance",
        "judge_faithfulness",
        "judge_completeness",
        "judge_clarity",
        "judge_limitation_handling",
    ]
    judge_means = {
        column: round(float(results_df[column].dropna().mean()), 3)
        if results_df[column].dropna().any()
        else None
        for column in judge_columns
    }
    leak_count = int(results_df["external_knowledge_leak"].fillna(False).astype(bool).sum())
    failed_mask = (
        (results_df["deterministic_pass_rate"] < 1.0)
        | results_df["external_knowledge_leak"].fillna(False).astype(bool)
        | results_df["failure_reason"].fillna("").ne("")
    )
    failed_case_ids = results_df.loc[failed_mask, "case_id"].tolist()
    category_scores = results_df.groupby("category")["deterministic_pass_rate"].mean()
    weakest_category = category_scores.idxmin() if not category_scores.empty else None

    improvements: list[str] = []
    if (results_df["filters_correct"] == False).any():
        improvements.append("Tighten metadata normalization and planner backfilling for difficult questions.")
    if (results_df["retrieval_filters_valid"] == False).any():
        improvements.append("Harden retrieval filter resolution for comparison-style evidence tasks.")
    if leak_count:
        improvements.append("Strengthen answer grounding so unsupported external context is always stated clearly.")
    if (results_df["limitations_present"] == False).any():
        improvements.append("Make limitation wording more explicit for date exclusions and small samples.")
    if not improvements:
        improvements.append("Expand the evaluation set with more adversarial metadata and aspect questions.")

    return {
        "num_questions": int(len(results_df)),
        "deterministic_pass_rate": deterministic_pass_rate,
        "judge_means": judge_means,
        "external_knowledge_leaks": leak_count,
        "failed_case_ids": failed_case_ids,
        "weakest_question_category": weakest_category,
        "recommended_improvements": improvements[:3],
    }
