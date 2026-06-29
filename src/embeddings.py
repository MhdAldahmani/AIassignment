from __future__ import annotations

from langchain_core.embeddings import Embeddings

from .config import Settings, get_settings

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:  # pragma: no cover - optional dependency
    HuggingFaceEmbeddings = None

try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:  # pragma: no cover - optional dependency
    OpenAIEmbeddings = None


def build_embedding_model(
    settings: Settings | None = None,
    *,
    provider: str | None = None,
) -> Embeddings:
    settings = settings or get_settings()
    embedding_provider = (provider or settings.embedding_provider).lower()

    if embedding_provider == "openai":
        if OpenAIEmbeddings is None:
            raise ImportError("langchain_openai is not installed.")
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings.")
        return OpenAIEmbeddings(model=settings.openai_embedding_model)

    if HuggingFaceEmbeddings is None:
        raise ImportError("langchain_huggingface is not installed.")
    return HuggingFaceEmbeddings(
        model_name=settings.local_embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
