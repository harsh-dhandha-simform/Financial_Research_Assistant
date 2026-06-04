"""
Embedding model with automatic fallback chain.

Priority order (tries each until one succeeds):
    1. OpenAI text-embedding-3-large (3072 dims, best quality)
    2. OpenAI text-embedding-3-small (1536 dims, cheaper fallback)
    3. BAAI/bge-base-en-v1.5 (768 dims, local HF model)
    4. nomic-ai/nomic-embed-text-v1.5 (768 dims, local HF model)

Uses the API_KEY and BASE_URL from .env for OpenAI-compatible endpoints.

Usage:
    from retrieval.embedding import get_embedding_model, embed_texts, embed_query

    model = get_embedding_model()   # auto-selects best available
    vectors = embed_texts(["Apple revenue was $394.3B"], model)
"""

import logging
from functools import lru_cache

from langchain_core.embeddings import Embeddings

from config import settings

logger = logging.getLogger(__name__)


# ── Fallback chain configuration ─────────────────────────────────────────────
# Each entry: (name, provider, model_id, dimensions)
EMBEDDING_CHAIN: list[tuple[str, str, str, int]] = [
    ("OpenAI text-embedding-3-large", "openai", "text-embedding-3-large", 3072),
    ("OpenAI text-embedding-3-small", "openai", "text-embedding-3-small", 1536),
    ("BAAI/bge-base-en-v1.5", "huggingface", "BAAI/bge-base-en-v1.5", 768),
    ("nomic-ai/nomic-embed-text-v1.5", "huggingface", "nomic-ai/nomic-embed-text-v1.5", 768),
]

# Cached model info after successful initialization
_active_model_name: str = ""
_active_dimensions: int = 0


def _try_openai_embeddings(model_id: str) -> Embeddings | None:
    """Try to create an OpenAI-compatible embedding model."""
    if not settings.openai_api_key:
        logger.info("Skipping OpenAI embeddings — API_KEY not set")
        return None

    try:
        from langchain_openai import OpenAIEmbeddings

        model = OpenAIEmbeddings(
            model=model_id,
            openai_api_key=settings.openai_api_key,
            openai_api_base=settings.openai_base_url or None,
        )
        # Test with a quick embed to verify it works
        test_vec = model.embed_query("test")
        logger.info(
            "OpenAI embeddings ready: %s (%d dims)",
            model_id,
            len(test_vec),
        )
        return model
    except Exception as exc:
        logger.warning("OpenAI embeddings failed for %s: %s", model_id, exc)
        return None


def _try_huggingface_embeddings(model_id: str) -> Embeddings | None:
    """Try to create a HuggingFace embedding model."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        model = HuggingFaceEmbeddings(
            model_name=model_id,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
        )
        # Test with a quick embed to verify it works
        test_vec = model.embed_query("test")
        logger.info(
            "HuggingFace embeddings ready: %s (%d dims)",
            model_id,
            len(test_vec),
        )
        return model
    except Exception as exc:
        logger.warning("HuggingFace embeddings failed for %s: %s", model_id, exc)
        return None


@lru_cache(maxsize=1)
def get_embedding_model() -> Embeddings:
    """Get the best available embedding model via fallback chain.

    Tries each model in EMBEDDING_CHAIN order. First successful model
    is cached and reused for all subsequent calls.

    Returns:
        A LangChain Embeddings instance.

    Raises:
        RuntimeError: If all models in the fallback chain fail.
    """
    global _active_model_name, _active_dimensions

    for name, provider, model_id, dimensions in EMBEDDING_CHAIN:
        logger.info("Trying embedding model: %s (%s)", name, provider)

        if provider == "openai":
            model = _try_openai_embeddings(model_id)
        elif provider == "huggingface":
            model = _try_huggingface_embeddings(model_id)
        else:
            continue

        if model is not None:
            _active_model_name = name
            _active_dimensions = dimensions
            logger.info("✅ Active embedding model: %s (%d dims)", name, dimensions)
            return model

    raise RuntimeError(
        "All embedding models failed. Check your API_KEY/BASE_URL in .env "
        "or install sentence-transformers for local models."
    )


def embed_texts(
    texts: list[str],
    model: Embeddings | None = None,
) -> list[list[float]]:
    """Embed a list of texts into dense vectors.

    Args:
        texts: List of text strings to embed.
        model: Embedding model instance. Uses auto-selected if None.

    Returns:
        List of embedding vectors (each is a list of floats).
    """
    if not texts:
        return []
    if model is None:
        model = get_embedding_model()
    vectors = model.embed_documents(texts)
    logger.info("Embedded %d texts → %d-dim vectors", len(texts), len(vectors[0]))
    return vectors


def embed_query(
    query: str,
    model: Embeddings | None = None,
) -> list[float]:
    """Embed a single query for retrieval.

    Args:
        query: The search query to embed.
        model: Embedding model instance. Uses auto-selected if None.

    Returns:
        Embedding vector as a list of floats.
    """
    if model is None:
        model = get_embedding_model()
    return model.embed_query(query)


def get_embedding_dimensions() -> int:
    """Return the embedding dimensions of the active model.

    Triggers model initialization if not yet done.
    """
    if _active_dimensions == 0:
        get_embedding_model()
    return _active_dimensions


def get_active_model_name() -> str:
    """Return the name of the active embedding model."""
    if not _active_model_name:
        get_embedding_model()
    return _active_model_name
