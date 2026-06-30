from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    planner_model: str
    answer_model: str
    judge_model: str
    embedding_provider: str
    openai_embedding_model: str
    local_embedding_model: str
    planner_temperature: float
    answer_temperature: float
    judge_temperature: float
    max_planner_retrieval_tasks: int
    min_retrieval_top_k: int
    max_retrieval_top_k: int
    min_month_reviews: int
    min_season_reviews: int
    visit_score_rating_weight: float
    visit_score_low_negative_weight: float
    visit_score_low_crowding_weight: float
    chroma_collection_name: str
    chroma_persist_directory: Path
    vectorstore_batch_size: int
    final_evidence_target: int
    final_evidence_max: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        planner_model=os.getenv("PLANNER_MODEL", "gpt-5.4-mini"),
        answer_model=os.getenv("ANSWER_MODEL", "gpt-5.4-mini"),
        judge_model=os.getenv("JUDGE_MODEL", "gpt-5.5"),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "local"),
        openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        local_embedding_model=os.getenv(
            "LOCAL_EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        ),
        planner_temperature=_get_float("PLANNER_TEMPERATURE", 0.0),
        answer_temperature=_get_float("ANSWER_TEMPERATURE", 0.0),
        judge_temperature=_get_float("JUDGE_TEMPERATURE", 0.0),
        max_planner_retrieval_tasks=_get_int("MAX_PLANNER_RETRIEVAL_TASKS", 4),
        min_retrieval_top_k=_get_int("MIN_RETRIEVAL_TOP_K", 1),
        max_retrieval_top_k=_get_int("MAX_RETRIEVAL_TOP_K", 8),
        min_month_reviews=_get_int("MIN_MONTH_REVIEWS", 100),
        min_season_reviews=_get_int("MIN_SEASON_REVIEWS", 200),
        visit_score_rating_weight=_get_float("VISIT_SCORE_RATING_WEIGHT", 0.45),
        visit_score_low_negative_weight=_get_float("VISIT_SCORE_LOW_NEGATIVE_WEIGHT", 0.30),
        visit_score_low_crowding_weight=_get_float("VISIT_SCORE_LOW_CROWDING_WEIGHT", 0.25),
        chroma_collection_name=os.getenv("CHROMA_COLLECTION_NAME", "disney_reviews"),
        chroma_persist_directory=Path(
            os.getenv("CHROMA_PERSIST_DIRECTORY", "artifacts/chroma/disney_reviews")
        ),
        vectorstore_batch_size=_get_int("VECTORSTORE_BATCH_SIZE", 256),
        final_evidence_target=_get_int("FINAL_EVIDENCE_TARGET", 8),
        final_evidence_max=_get_int("FINAL_EVIDENCE_MAX", 12),
    )
