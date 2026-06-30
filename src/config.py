from pathlib import Path

from dotenv import load_dotenv
import os

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env", override=True)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Switched from OpenAI to Groq for all chat calls: this OpenAI account's gpt-4o-mini
# usage tier caps out at 10,000 requests/day (hit while enriching ~3.7K rows), which
# can't finish 42.6K rows quickly. Groq's openai/gpt-oss-20b has a 500K req/day,
# 250K tokens/min limit on this key, and is fast enough to finish in a few hours.
ENRICHMENT_MODEL = "openai/gpt-oss-20b"
ANSWER_MODEL = "openai/gpt-oss-20b"
JUDGE_MODEL = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "BAAI/bge-m3"

RAW_CSV_PATH = ROOT_DIR / "DisneylandReviews.csv"
DATA_DIR = ROOT_DIR / "data" / "clean"
CLEAN_PARQUET_PATH = DATA_DIR / "reviews_clean.parquet"
ENRICHED_PARQUET_PATH = DATA_DIR / "reviews_enriched.parquet"
ENRICHMENT_CHECKPOINT_PATH = DATA_DIR / "enrichment_checkpoint.jsonl"
CHROMA_PERSIST_DIR = DATA_DIR / "chroma_db"

DATA_DIR.mkdir(parents=True, exist_ok=True)

ASPECTS = [
    "queues_wait_times",
    "staff_service",
    "price_value",
    "food_beverage",
    "rides_attractions",
    "cleanliness",
    "crowding",
    "weather",
    "accessibility_mobility",
]

BRANCHES = ["Disneyland_California", "Disneyland_Paris", "Disneyland_HongKong"]

MIN_SAMPLE_SIZE = 5
