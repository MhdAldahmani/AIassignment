import json
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from config import ASPECTS, ENRICHMENT_MODEL, GROQ_API_KEY

SENTIMENT = Literal["positive", "negative", "neutral", "mixed"]

ASPECT_DESCRIPTIONS = {
    "queues_wait_times": "waiting in line for rides/attractions/security/parking, FastPass/Genie+/standby times",
    "staff_service": "cast members / staff helpfulness, friendliness, professionalism",
    "price_value": "ticket price, value for money, cost of food/merchandise/hotels",
    "food_beverage": "restaurants, snacks, food quality or variety inside the park",
    "rides_attractions": "specific rides, shows, attractions, parades themselves (not the wait to get on them)",
    "cleanliness": "park/restroom/facility cleanliness and maintenance",
    "crowding": "how busy/crowded the park felt, density of visitors",
    "weather": "heat, rain, humidity, cold and its effect on the visit",
    "accessibility_mobility": "ease of access for strollers, wheelchairs, elderly, disabled visitors",
}


class AspectSignal(BaseModel):
    mentioned: bool = Field(description="Whether this aspect is discussed in the review")
    sentiment: SENTIMENT = Field(
        description="Sentiment toward this aspect; if mentioned=False, use 'neutral' as a placeholder"
    )


class ReviewEnrichment(BaseModel):
    overall_sentiment: SENTIMENT
    queues_wait_times: AspectSignal
    staff_service: AspectSignal
    price_value: AspectSignal
    food_beverage: AspectSignal
    rides_attractions: AspectSignal
    cleanliness: AspectSignal
    crowding: AspectSignal
    weather: AspectSignal
    accessibility_mobility: AspectSignal
    primary_complaint: Optional[str] = Field(
        default=None, description="One short phrase naming the single biggest complaint, or null if none"
    )
    primary_delight: Optional[str] = Field(
        default=None, description="One short phrase naming the single biggest highlight, or null if none"
    )


_ASPECT_LINES = "\n".join(f"- {a}: {d}" for a, d in ASPECT_DESCRIPTIONS.items())

_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are tagging Disneyland park reviews for a customer-experience analytics pipeline. "
            "For the given review, determine the overall sentiment, and for EACH of the following aspects "
            "decide whether it is mentioned and (if mentioned) its sentiment:\n"
            f"{_ASPECT_LINES}\n\n"
            "Only mark an aspect as mentioned if the review actually discusses it. "
            "primary_complaint/primary_delight should be short (under 8 words) or null.",
        ),
        ("human", "Review:\n{review_text}"),
    ]
)


def build_enrichment_chain():
    llm = ChatGroq(model=ENRICHMENT_MODEL, temperature=0, api_key=GROQ_API_KEY, max_retries=5)
    structured_llm = llm.with_structured_output(ReviewEnrichment)
    return _PROMPT | structured_llm


def load_checkpoint(path: Path) -> dict[str, dict]:
    if not Path(path).exists():
        return {}
    out = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["Review_ID"]] = rec
    return out


def append_checkpoint(path: Path, records: list[dict]) -> None:
    with open(path, "a") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def enrich_all(
    df: pd.DataFrame,
    checkpoint_path: Path,
    batch_size: int = 200,
    max_concurrency: int = 10,
    limit: Optional[int] = None,
    verbose: bool = True,
) -> None:
    chain = build_enrichment_chain()
    done_ids = set(load_checkpoint(checkpoint_path).keys())

    todo = df[~df["Review_ID"].astype(str).isin(done_ids)]
    if limit is not None:
        todo = todo.head(limit)

    if verbose:
        print(f"{len(done_ids)} already done, {len(todo)} to process")

    retry_queue: list[dict] = []

    for start in range(0, len(todo), batch_size):
        batch_df = todo.iloc[start : start + batch_size]
        inputs = [{"review_text": t} for t in batch_df["Review_Text"]]
        results = chain.batch(inputs, config={"max_concurrency": max_concurrency}, return_exceptions=True)

        successes = []
        for (_, row), result in zip(batch_df.iterrows(), results):
            if isinstance(result, Exception):
                retry_queue.append({"Review_ID": str(row["Review_ID"]), "review_text": row["Review_Text"]})
                continue
            rec = result.model_dump()
            rec["Review_ID"] = str(row["Review_ID"])
            successes.append(rec)

        append_checkpoint(checkpoint_path, successes)
        if verbose:
            done_so_far = start + len(batch_df)
            print(f"  {done_so_far}/{len(todo)} processed ({len(retry_queue)} failed so far)")

    if retry_queue:
        if verbose:
            print(f"Retrying {len(retry_queue)} failed rows at lower concurrency")
        inputs = [{"review_text": r["review_text"]} for r in retry_queue]
        results = chain.batch(inputs, config={"max_concurrency": 3}, return_exceptions=True)
        successes = []
        still_failed = 0
        for r, result in zip(retry_queue, results):
            if isinstance(result, Exception):
                still_failed += 1
                continue
            rec = result.model_dump()
            rec["Review_ID"] = r["Review_ID"]
            successes.append(rec)
        append_checkpoint(checkpoint_path, successes)
        if verbose:
            print(f"  retry done, {still_failed} permanently failed")


def _flatten_record(rec: dict) -> dict:
    flat = {"overall_sentiment": rec["overall_sentiment"]}
    for aspect in ASPECTS:
        signal = rec[aspect]
        flat[f"mentions_{aspect}"] = bool(signal["mentioned"])
        flat[f"{aspect}_sentiment"] = signal["sentiment"] if signal["mentioned"] else None
    flat["primary_complaint"] = rec.get("primary_complaint")
    flat["primary_delight"] = rec.get("primary_delight")
    return flat


def load_enriched(df: pd.DataFrame, checkpoint_path: Path) -> pd.DataFrame:
    checkpoint = load_checkpoint(checkpoint_path)
    df = df.copy()
    df["Review_ID"] = df["Review_ID"].astype(str)

    flat_records = []
    for rid in df["Review_ID"]:
        rec = checkpoint.get(rid)
        flat_records.append(_flatten_record(rec) if rec is not None else _flatten_record_empty())

    flat_df = pd.DataFrame(flat_records, index=df.index)
    return pd.concat([df, flat_df], axis=1)


def _flatten_record_empty() -> dict:
    flat = {"overall_sentiment": None}
    for aspect in ASPECTS:
        flat[f"mentions_{aspect}"] = False
        flat[f"{aspect}_sentiment"] = None
    flat["primary_complaint"] = None
    flat["primary_delight"] = None
    return flat
