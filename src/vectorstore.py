from pathlib import Path

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import ASPECTS, EMBEDDING_MODEL


def build_embeddings() -> HuggingFaceEmbeddings:
    emb = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "mps"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 8},
        query_encode_kwargs={"normalize_embeddings": True},
    )
    emb._client.max_seq_length = 1024
    return emb


def flatten_for_metadata(row: pd.Series) -> dict:
    meta = {
        "Review_ID": str(row["Review_ID"]),
        "branch": row["Branch"],
        "country": row["Reviewer_Location"],
        "rating": int(row["Rating"]),
    }
    if pd.notna(row.get("year")):
        meta["year"] = int(row["year"])
    if pd.notna(row.get("month")):
        meta["month"] = int(row["month"])
    if pd.notna(row.get("season")):
        meta["season"] = row["season"]
    if pd.notna(row.get("overall_sentiment")):
        meta["overall_sentiment"] = row["overall_sentiment"]
    for aspect in ASPECTS:
        mentions_col = f"mentions_{aspect}"
        sentiment_col = f"{aspect}_sentiment"
        if mentions_col in row and pd.notna(row[mentions_col]):
            meta[mentions_col] = bool(row[mentions_col])
            if row[mentions_col] and pd.notna(row.get(sentiment_col)):
                meta[sentiment_col] = row[sentiment_col]
    if pd.notna(row.get("primary_complaint")):
        meta["primary_complaint"] = row["primary_complaint"]
    if pd.notna(row.get("primary_delight")):
        meta["primary_delight"] = row["primary_delight"]
    return meta


def build_documents(df: pd.DataFrame) -> list[Document]:
    docs = []
    for _, row in df.iterrows():
        docs.append(Document(page_content=row["Review_Text"], metadata=flatten_for_metadata(row)))
    return docs


def build_or_load_vectorstore(
    documents: list[Document],
    persist_dir: Path,
    embedding: HuggingFaceEmbeddings | None = None,
    batch_size: int = 500,
    verbose: bool = True,
) -> Chroma:
    embedding = embedding or build_embeddings()
    store = Chroma(
        collection_name="disney_reviews",
        embedding_function=embedding,
        persist_directory=str(persist_dir),
    )

    existing_ids = set(store.get(include=[])["ids"])
    new_docs = [d for d in documents if d.metadata["Review_ID"] not in existing_ids]

    if verbose:
        print(f"{len(existing_ids)} already indexed, {len(new_docs)} to add")

    for start in range(0, len(new_docs), batch_size):
        batch = new_docs[start : start + batch_size]
        ids = [d.metadata["Review_ID"] for d in batch]
        store.add_documents(batch, ids=ids)
        if verbose:
            print(f"  indexed {start + len(batch)}/{len(new_docs)}")

    return store
