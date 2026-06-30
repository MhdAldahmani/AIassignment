import re

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from config import GROQ_API_KEY, JUDGE_MODEL
from rag_pipeline import RagPipeline

GOLDEN_SET = [
    {
        "question": "What do visitors from Australia say about Disneyland in HongKong?",
        "expected_filter": {"branch": "Disneyland_HongKong", "country": "Australia"},
    },
    {
        "question": "Is spring a good time to visit Disneyland?",
        "expected_filter": {"season": "Spring"},
    },
    {
        "question": "Is Disneyland California usually crowded in June?",
        "expected_filter": {"branch": "Disneyland_California", "month": 6, "aspects": ["crowding"]},
    },
    {
        "question": "Is the staff in Paris friendly?",
        "expected_filter": {"branch": "Disneyland_Paris", "aspects": ["staff_service"]},
    },
    {
        "question": "I hate waiting in queues. What is the best time of year to visit Disneyland Hong Kong?",
        "expected_filter": {"branch": "Disneyland_HongKong", "aspects": ["queues_wait_times"]},
    },
    {
        "question": "How do reviewers from the United States rate Disneyland Paris?",
        "expected_filter": {"branch": "Disneyland_Paris", "country": "United States"},
    },
    {
        "question": "What do people complain about most at Disneyland California?",
        "expected_filter": {"branch": "Disneyland_California"},
    },
    {
        "question": "Are the rides at Hong Kong Disneyland worth it?",
        "expected_filter": {"branch": "Disneyland_HongKong", "aspects": ["rides_attractions"]},
    },
    {
        "question": "Is food expensive at Disneyland Paris?",
        "expected_filter": {"branch": "Disneyland_Paris", "aspects": ["price_value", "food_beverage"]},
    },
    {
        "question": "How was Disneyland California rated by visitors in December?",
        "expected_filter": {"branch": "Disneyland_California", "month": 12},
    },
    {
        "question": "Do visitors from Singapore enjoy Disneyland Hong Kong?",
        "expected_filter": {"branch": "Disneyland_HongKong", "country": "Singapore"},
    },
    {
        "question": "Is the weather an issue for visitors to Disneyland Hong Kong in summer?",
        "expected_filter": {"branch": "Disneyland_HongKong", "season": "Summer", "aspects": ["weather"]},
    },
    {
        "question": "What do 1-star reviews of Disneyland Paris complain about?",
        "expected_filter": {"branch": "Disneyland_Paris", "rating_max": 1},
    },
    {
        "question": "Are cleanliness standards good at Disneyland California?",
        "expected_filter": {"branch": "Disneyland_California", "aspects": ["cleanliness"]},
    },
    {
        "question": "What do British visitors think about Disneyland Paris staff?",
        "expected_filter": {"branch": "Disneyland_Paris", "country": "United Kingdom", "aspects": ["staff_service"]},
    },
    {
        "question": "Is autumn a quiet time to visit Disneyland California?",
        "expected_filter": {"branch": "Disneyland_California", "season": "Fall", "aspects": ["crowding"]},
    },
    {
        "question": "Do visitors from India rate Disneyland Hong Kong highly?",
        "expected_filter": {"branch": "Disneyland_HongKong", "country": "India"},
    },
    {
        "question": "How accessible is Disneyland Paris for visitors with mobility issues?",
        "expected_filter": {"branch": "Disneyland_Paris", "aspects": ["accessibility_mobility"]},
    },
    {
        "question": "What's the best month to visit Disneyland California to avoid crowds?",
        "expected_filter": {"branch": "Disneyland_California", "aspects": ["crowding"]},
    },
    {
        "question": "Do 5-star reviews of Disneyland Hong Kong mention the rides specifically?",
        "expected_filter": {"branch": "Disneyland_HongKong", "rating_min": 5, "aspects": ["rides_attractions"]},
    },
    {
        "question": "What do visitors from Canada say about ticket prices at Disneyland California?",
        "expected_filter": {"branch": "Disneyland_California", "country": "Canada", "aspects": ["price_value"]},
    },
    {
        "question": "Is winter a bad time to visit Disneyland Paris because of weather?",
        "expected_filter": {"branch": "Disneyland_Paris", "season": "Winter", "aspects": ["weather"]},
    },
    {
        "question": "What do reviewers from the Philippines say about Hong Kong Disneyland's food?",
        "expected_filter": {"branch": "Disneyland_HongKong", "country": "Philippines", "aspects": ["food_beverage"]},
    },
    {
        "question": "What's the overall sentiment toward Disneyland Paris compared to its rating?",
        "expected_filter": {"branch": "Disneyland_Paris"},
    },
    {
        "question": "Do New Zealand visitors find Disneyland California overcrowded in summer?",
        "expected_filter": {
            "branch": "Disneyland_California",
            "country": "New Zealand",
            "season": "Summer",
            "aspects": ["crowding"],
        },
    },
]


def score_filter_extraction(predicted, expected: dict) -> dict:
    pred = predicted.model_dump()
    fields = ["branch", "country", "year", "month", "season", "rating_min", "rating_max"]
    matches, total = 0, 0
    for f in fields:
        if f in expected:
            total += 1
            if pred.get(f) == expected[f]:
                matches += 1
    if "aspects" in expected:
        total += 1
        if set(pred.get("aspects") or []) & set(expected["aspects"]):
            matches += 1
    return {"matches": matches, "total": total, "accuracy": matches / total if total else None}


def check_aggregate_accuracy(answer: str, aggregates: dict) -> bool:
    """Programmatic check: does the answer state the correct sample count (no LLM needed)?"""
    count = aggregates.get("count")
    if not count:
        return None
    numbers_in_answer = {int(m.replace(",", "")) for m in re.findall(r"\b[\d,]{2,}\b", answer)}
    return count in numbers_in_answer


class JudgeScore(BaseModel):
    score: int = Field(ge=1, le=5, description="1=poor, 5=excellent")
    reasoning: str = Field(description="One short sentence explaining the score")


_faithfulness_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You score how faithfully an AI answer is grounded in the provided evidence (aggregate stats + review snippets). "
            "Use this rubric:\n"
            "5 — every factual/numeric claim is directly traceable to the evidence; Review_IDs cited appear in the snippets.\n"
            "4 — one minor rounding or phrasing difference from the evidence, no invented facts.\n"
            "3 — answer correctly summarises the evidence but adds one plausible-sounding detail not present in it.\n"
            "2 — multiple claims that extrapolate significantly beyond what the evidence shows.\n"
            "1 — key numbers or quotes are invented / contradict the evidence.\n"
            "Do NOT penalise for citing Review_IDs that appear in the snippets — those are expected citations.",
        ),
        (
            "human",
            "Evidence (aggregates):\n{aggregates}\n\nEvidence (snippets):\n{snippets}\n\nAnswer:\n{answer}",
        ),
    ]
)

_answer_quality_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You score the overall quality of an AI answer to a customer-experience question about Disneyland. "
            "Evaluate on three sub-criteria simultaneously and give a single 1-5 score:\n"
            "A) DIRECTNESS — does it give a clear verdict or takeaway, not just raw data?\n"
            "B) SPECIFICITY — does it back the verdict with concrete numbers or examples from the data?\n"
            "C) HONESTY — does it transparently state the sample size and any data limitations?\n\n"
            "5 — excels on all three (clear verdict, hard numbers, honest caveats).\n"
            "4 — strong on two, adequate on the third.\n"
            "3 — meets all three minimally, or excels on two while missing one entirely.\n"
            "2 — clear on one, weak on the others.\n"
            "1 — vague, no supporting data, or overly hedged without reason.",
        ),
        ("human", "Question:\n{question}\n\nAnswer:\n{answer}"),
    ]
)


def _judge_llm():
    return ChatGroq(model=JUDGE_MODEL, temperature=0, api_key=GROQ_API_KEY, max_retries=10).with_structured_output(
        JudgeScore
    )


def score_faithfulness(answer: str, aggregates: dict, snippets: list = None) -> JudgeScore:
    snippets_text = "\n\n".join(
        f"[Review_ID={d.metadata.get('Review_ID')}] {d.page_content[:300]}" for d in (snippets or [])
    )
    chain = _faithfulness_prompt | _judge_llm()
    return chain.invoke({"aggregates": aggregates, "snippets": snippets_text or "(none)", "answer": answer})


def score_answer_quality(question: str, answer: str) -> JudgeScore:
    chain = _answer_quality_prompt | _judge_llm()
    return chain.invoke({"question": question, "answer": answer})


def run_eval(pipeline: RagPipeline, golden_set: list[dict] = GOLDEN_SET) -> pd.DataFrame:
    rows = []
    for item in golden_set:
        trace = pipeline.ask(item["question"], return_trace=True)
        filter_score = score_filter_extraction(trace["filter"], item["expected_filter"])
        faithfulness = score_faithfulness(trace["answer"], trace["aggregates"], trace["snippets"])
        answer_quality = score_answer_quality(item["question"], trace["answer"])
        count_correct = check_aggregate_accuracy(trace["answer"], trace["aggregates"])
        rows.append(
            {
                "question": item["question"],
                "sample_size": trace["aggregates"].get("count"),
                "filter_accuracy": filter_score["accuracy"],
                "faithfulness": faithfulness.score,
                "faithfulness_reason": faithfulness.reasoning,
                "answer_quality": answer_quality.score,
                "answer_quality_reason": answer_quality.reasoning,
                "count_in_answer": count_correct,
                "answer": trace["answer"],
            }
        )
    return pd.DataFrame(rows)
