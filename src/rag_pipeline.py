from typing import Literal, Optional

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from rapidfuzz import process as rf_process

from config import ANSWER_MODEL, ASPECTS, BRANCHES, GROQ_API_KEY, MIN_SAMPLE_SIZE

SEASONS = ["Winter", "Spring", "Summer", "Fall"]

RELAXATION_ORDER = ["month", "rating", "aspects", "season", "year", "country"]


class QueryFilter(BaseModel):
    branch: Optional[Literal[tuple(BRANCHES)]] = None
    country: Optional[str] = Field(
        default=None, description="Reviewer's country/region of origin, in plain English"
    )
    year: Optional[int] = None
    month: Optional[int] = Field(default=None, ge=1, le=12)
    season: Optional[Literal[tuple(SEASONS)]] = None
    rating_min: Optional[int] = Field(default=None, ge=1, le=5)
    rating_max: Optional[int] = Field(default=None, ge=1, le=5)
    aspects: Optional[list[Literal[tuple(ASPECTS)]]] = Field(default=None)
    semantic_query: str = Field(
        description="A clean rephrasing of the question's topical intent, stripped of the filter "
        "conditions captured above, for use in semantic similarity search over review text"
    )


_ASPECT_LIST = ", ".join(ASPECTS)
_BRANCH_LIST = ", ".join(BRANCHES)

_FILTER_SYSTEM_PROMPT = f"""You turn a customer-experience question about Disneyland park reviews into a \
structured filter + semantic search query. The dataset has reviews for these branches: {_BRANCH_LIST}. \
Valid aspect tags are: {_ASPECT_LIST}. Only set a field if the question clearly implies it; leave others null/empty. \
Country must be a real country/region name in plain English (e.g. "Australia", "United States", "Hong Kong"), \
inferred from demonyms if needed (e.g. "Aussies" -> "Australia", "Brits"/"the UK" -> "United Kingdom").

Examples:
Q: "What do visitors from Australia say about Disneyland in HongKong?"
-> branch=Disneyland_HongKong, country=Australia, semantic_query="general opinions and experiences visiting"

Q: "Is spring a good time to visit Disneyland?"
-> season=Spring, semantic_query="crowd levels, weather, wait times during spring visits"

Q: "Is Disneyland California usually crowded in June?"
-> branch=Disneyland_California, month=6, aspects=[crowding], semantic_query="how crowded the park is in June"

Q: "Is the staff in Paris friendly?"
-> branch=Disneyland_Paris, aspects=[staff_service], semantic_query="staff and cast member friendliness"
"""

_filter_prompt = ChatPromptTemplate.from_messages(
    [("system", _FILTER_SYSTEM_PROMPT), ("human", "{question}")]
)


def build_filter_chain():
    llm = ChatGroq(model=ANSWER_MODEL, temperature=0, api_key=GROQ_API_KEY, max_retries=10)
    return _filter_prompt | llm.with_structured_output(QueryFilter)


_ANSWER_SYSTEM_PROMPT = """You are a customer-experience analyst answering questions about Disneyland park \
reviews. You are given: (1) aggregate statistics computed directly from the dataset for the question's \
matching subset of reviews, and (2) a handful of representative review snippets with their Review_ID. \
Answer using ONLY this information - never invent numbers or facts not present in it. \
Always state the sample size (number of matching reviews) the answer is based on. \
Cite Review_IDs in parentheses when quoting or referencing a specific snippet. \
If any filter conditions were relaxed/dropped to find enough data, mention this transparently. \
If the sample size is below {min_sample}, say there is insufficient data for a confident answer."""

_answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _ANSWER_SYSTEM_PROMPT),
        (
            "human",
            "Question: {question}\n\n"
            "Aggregate stats (JSON): {aggregates}\n\n"
            "Relaxation log (filters dropped due to insufficient data, in order): {relaxation_log}\n\n"
            "Country match note: {country_note}\n\n"
            "Representative snippets:\n{snippets}",
        ),
    ]
)


def build_answer_chain():
    llm = ChatGroq(model=ANSWER_MODEL, temperature=0.2, api_key=GROQ_API_KEY, max_retries=10)
    return _answer_prompt | llm


def apply_filter_to_df(df: pd.DataFrame, filt: QueryFilter) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if filt.branch:
        mask &= df["Branch"] == filt.branch
    if filt.country:
        mask &= df["Reviewer_Location"] == filt.country
    if filt.year:
        mask &= df["year"] == filt.year
    if filt.month:
        mask &= df["month"] == filt.month
    if filt.season:
        mask &= df["season"] == filt.season
    if filt.rating_min:
        mask &= df["Rating"] >= filt.rating_min
    if filt.rating_max:
        mask &= df["Rating"] <= filt.rating_max
    if filt.aspects:  # None or [] both falsy — treated as "no aspect filter"
        aspect_mask = pd.Series(False, index=df.index)
        for aspect in filt.aspects:
            col = f"mentions_{aspect}"
            if col in df.columns:
                aspect_mask |= df[col].fillna(False)
        mask &= aspect_mask
    return df[mask]


def filter_to_chroma_where(filt: QueryFilter) -> Optional[dict]:
    conditions = []
    if filt.branch:
        conditions.append({"branch": filt.branch})
    if filt.country:
        conditions.append({"country": filt.country})
    if filt.year:
        conditions.append({"year": filt.year})
    if filt.month:
        conditions.append({"month": filt.month})
    if filt.season:
        conditions.append({"season": filt.season})
    if filt.rating_min:
        conditions.append({"rating": {"$gte": filt.rating_min}})
    if filt.rating_max:
        conditions.append({"rating": {"$lte": filt.rating_max}})
    if filt.aspects:
        aspect_conditions = [{f"mentions_{a}": True} for a in filt.aspects]
        conditions.append(aspect_conditions[0] if len(aspect_conditions) == 1 else {"$or": aspect_conditions})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def drop_field(filt: QueryFilter, field: str) -> QueryFilter:
    if field == "month":
        return filt.model_copy(update={"month": None})
    if field == "rating":
        return filt.model_copy(update={"rating_min": None, "rating_max": None})
    if field == "aspects":
        return filt.model_copy(update={"aspects": None})
    if field == "season":
        return filt.model_copy(update={"season": None})
    if field == "year":
        return filt.model_copy(update={"year": None})
    if field == "country":
        return filt.model_copy(update={"country": None})
    return filt


def compute_aggregates(subset: pd.DataFrame) -> dict:
    count = len(subset)
    if count == 0:
        return {"count": 0}
    agg = {
        "count": count,
        "avg_rating": round(float(subset["Rating"].mean()), 2),
        "rating_histogram": subset["Rating"].value_counts().sort_index().to_dict(),
        "aspects": {},
    }
    for aspect in ASPECTS:
        col = f"mentions_{aspect}"
        if col not in subset.columns:
            continue
        mentioned = subset[subset[col].fillna(False)]
        if len(mentioned) == 0:
            continue
        agg["aspects"][aspect] = {
            "mention_rate": round(len(mentioned) / count, 3),
            "sentiment_breakdown": mentioned[f"{aspect}_sentiment"].value_counts().to_dict(),
        }
    return agg


def _format_snippets(snippets: list[Document]) -> str:
    lines = []
    for doc in snippets:
        rid = doc.metadata.get("Review_ID")
        rating = doc.metadata.get("rating")
        text = doc.page_content[:500]
        lines.append(f"[Review_ID={rid}, rating={rating}] {text}")
    return "\n\n".join(lines) if lines else "(no snippets retrieved)"


class RagPipeline:
    def __init__(self, df: pd.DataFrame, store: Chroma):
        self.df = df
        self.store = store
        self.countries = sorted(df["Reviewer_Location"].dropna().unique().tolist())
        self.filter_chain = build_filter_chain()
        self.answer_chain = build_answer_chain()

    def _match_country(self, raw_country: Optional[str]) -> tuple[Optional[str], Optional[str]]:
        if not raw_country:
            return None, None
        match = rf_process.extractOne(raw_country, self.countries, score_cutoff=85)
        if match is None:
            return None, f"Could not match '{raw_country}' to a known country in the dataset; ignoring this filter."
        return match[0], None

    def ask(self, question: str, k: int = 8, return_trace: bool = False):
        filt = self.filter_chain.invoke({"question": question})
        matched_country, country_note = self._match_country(filt.country)
        filt = filt.model_copy(update={"country": matched_country})

        relaxation_log: list[str] = []
        subset = apply_filter_to_df(self.df, filt)
        for field in RELAXATION_ORDER:
            if len(subset) >= MIN_SAMPLE_SIZE:
                break
            filt = drop_field(filt, field)
            relaxation_log.append(field)
            subset = apply_filter_to_df(self.df, filt)

        aggregates = compute_aggregates(subset)
        where = filter_to_chroma_where(filt)
        snippets = self.store.similarity_search(filt.semantic_query, k=k, filter=where)

        answer = self.answer_chain.invoke(
            {
                "question": question,
                "aggregates": aggregates,
                "relaxation_log": relaxation_log or "none",
                "country_note": country_note or "none",
                "snippets": _format_snippets(snippets),
                "min_sample": MIN_SAMPLE_SIZE,
            }
        ).content

        if not return_trace:
            return answer
        return {
            "question": question,
            "filter": filt,
            "aggregates": aggregates,
            "relaxation_log": relaxation_log,
            "country_note": country_note,
            "snippets": snippets,
            "answer": answer,
        }
