from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .config import Settings, get_settings
from .embeddings import build_embedding_model

try:
    from langchain_chroma import Chroma
except ImportError:  # pragma: no cover - optional dependency
    Chroma = None


MANIFEST_FILE_NAME = "manifest.json"


@dataclass
class VectorStoreBuildResult:
    vectorstore: Chroma
    persist_directory: Path
    manifest: dict[str, object]
    reused_existing_index: bool


def _manifest_path(persist_directory: Path) -> Path:
    return persist_directory / MANIFEST_FILE_NAME


def _read_manifest(persist_directory: Path) -> dict[str, object] | None:
    manifest_path = _manifest_path(persist_directory)
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text())


def _write_manifest(persist_directory: Path, manifest: dict[str, object]) -> None:
    manifest_path = _manifest_path(persist_directory)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def dataset_fingerprint(df: pd.DataFrame, settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    subset = (
        df[
        [
            "Review_ID",
            "Branch",
            "Reviewer_Location",
            "Rating",
            "Rating_Sentiment",
            "Year",
            "Month",
            "Season",
            "Review_Text",
        ]
        ]
        .astype("string")
        .fillna("")
    )
    hashed_rows = pd.util.hash_pandas_object(subset, index=False).astype(str)
    digest = hashlib.sha256()
    digest.update("|".join(hashed_rows.tolist()).encode("utf-8"))
    digest.update(f"|provider={settings.embedding_provider}".encode("utf-8"))
    digest.update(f"|model={settings.openai_embedding_model}|{settings.local_embedding_model}".encode("utf-8"))
    return digest.hexdigest()


def review_documents(df: pd.DataFrame) -> tuple[list[Document], list[str]]:
    documents: list[Document] = []
    ids: list[str] = []
    for row in df.itertuples(index=False):
        review_id = str(getattr(row, "Review_ID"))
        has_date = pd.notna(getattr(row, "Review_Date"))
        documents.append(
            Document(
                page_content=str(getattr(row, "Review_Text")),
                metadata={
                    "review_id": review_id,
                    "branch": str(getattr(row, "Branch")),
                    "reviewer_location": str(getattr(row, "Reviewer_Location")),
                    "rating": int(getattr(row, "Rating")),
                    "rating_sentiment": str(getattr(row, "Rating_Sentiment")),
                    "year": int(getattr(row, "Year")) if pd.notna(getattr(row, "Year")) else -1,
                    "month": int(getattr(row, "Month")) if pd.notna(getattr(row, "Month")) else -1,
                    "season": str(getattr(row, "Season")) if pd.notna(getattr(row, "Season")) else "Unknown",
                    "has_date": bool(has_date),
                },
            )
        )
        ids.append(review_id)
    return documents, ids


def build_or_load_vectorstore(
    df: pd.DataFrame,
    *,
    settings: Settings | None = None,
    persist_directory: str | Path | None = None,
    collection_name: str | None = None,
    embeddings: Embeddings | None = None,
    force_rebuild: bool = False,
    verbose: bool = True,
) -> VectorStoreBuildResult:
    if Chroma is None:
        raise ImportError("langchain_chroma is not installed.")

    settings = settings or get_settings()
    persist_path = Path(persist_directory or settings.chroma_persist_directory)
    collection = collection_name or settings.chroma_collection_name
    persist_path.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset_fingerprint": dataset_fingerprint(df, settings),
        "embedding_provider": settings.embedding_provider,
        "openai_embedding_model": settings.openai_embedding_model,
        "local_embedding_model": settings.local_embedding_model,
        "row_count": int(len(df)),
        "collection_name": collection,
    }
    existing_manifest = _read_manifest(persist_path)
    embeddings = embeddings or build_embedding_model(settings)

    if not force_rebuild and existing_manifest == manifest:
        if verbose:
            print(f"Reusing existing Chroma index at {persist_path}")
        vectorstore = Chroma(
            collection_name=collection,
            persist_directory=str(persist_path),
            embedding_function=embeddings,
        )
        return VectorStoreBuildResult(
            vectorstore=vectorstore,
            persist_directory=persist_path,
            manifest=manifest,
            reused_existing_index=True,
        )

    if persist_path.exists():
        shutil.rmtree(persist_path)
    persist_path.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma(
        collection_name=collection,
        persist_directory=str(persist_path),
        embedding_function=embeddings,
    )
    documents, ids = review_documents(df)
    batch_size = settings.vectorstore_batch_size
    total = len(documents)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_documents = documents[start:end]
        batch_ids = ids[start:end]
        vectorstore.add_documents(batch_documents, ids=batch_ids)
        if verbose:
            print(f"Indexed reviews {start + 1:,}-{end:,} of {total:,}")

    _write_manifest(persist_path, manifest)
    return VectorStoreBuildResult(
        vectorstore=vectorstore,
        persist_directory=persist_path,
        manifest=manifest,
        reused_existing_index=False,
    )
