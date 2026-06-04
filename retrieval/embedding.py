"""
Embedding model with automatic fallback chain.

Uses inference APIs where possible (no downloads) with local fallback.

Priority order (tries each until one succeeds):
    1. Voyage voyage-4        (1024 dims, Voyage API)
    2. BAAI/bge-m3            (1024 dims, HuggingFace Inference API)
    3. Qwen3-Embedding-0.6B   (1024 dims, local sentence-transformers)
    4. nomic-embed-text-v1.5  (768 dims, Nomic Atlas API)
    5. OpenAI text-embedding-3-large (3072 dims, OpenAI-compat API)
    6. OpenAI text-embedding-3-small (1536 dims, OpenAI-compat API)

First 3 models all produce 1024-dim embeddings — ideal for Qdrant consistency.
Models 4-6 have different dimensions and serve as last-resort fallbacks.

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
# Primary models (1-3) all produce 1024-dim for Qdrant consistency.
# Fallback models (4-6) have different dims — will need a separate collection.
# Each entry: (name, provider, model_id, dimensions)
EMBEDDING_CHAIN: list[tuple[str, str, str, int]] = [
    # ── Primary: 1024-dim models ──
    ("Voyage voyage-4", "voyage", "voyage-4", 1024),
    ("BAAI/bge-m3", "hf_inference", "BAAI/bge-m3", 1024),
    ("Qwen3-Embedding-0.6B", "hf_local", "Qwen/Qwen3-Embedding-0.6B", 1024),
    # ── Fallback: different dimensions ──
    ("nomic-embed-text-v1.5", "nomic", "nomic-embed-text-v1.5", 768),
    ("OpenAI text-embedding-3-large", "openai", "text-embedding-3-large", 3072),
    ("OpenAI text-embedding-3-small", "openai", "text-embedding-3-small", 1536),
]

# Cached model info after successful initialization
_active_model_name: str = ""
_active_dimensions: int = 0


# ═════════════════════════════════════════════════════════════════════════════
# Custom LangChain Embeddings wrappers for API-based providers
# ═════════════════════════════════════════════════════════════════════════════


class HuggingFaceInferenceEmbeddings(Embeddings):
    """LangChain-compatible wrapper for HuggingFace Inference API.

    Uses the HuggingFace serverless Inference API — no local model download.
    Requires HF_TOKEN with access to inference providers.
    """

    def __init__(self, model_id: str, api_key: str) -> None:
        from huggingface_hub import InferenceClient

        self.model_id = model_id
        self.client = InferenceClient(model=model_id, token=api_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents via HF Inference API."""
        result = self.client.feature_extraction(texts)
        # HF returns numpy arrays or nested lists
        return [list(vec) for vec in result]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query via HF Inference API."""
        result = self.client.feature_extraction(text)
        return list(result)


class VoyageEmbeddings(Embeddings):
    """LangChain-compatible wrapper for Voyage AI embedding API."""

    def __init__(self, model: str = "voyage-4", api_key: str = "") -> None:
        import voyageai

        self.model = model
        self.client = voyageai.Client(api_key=api_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents via Voyage API."""
        result = self.client.embed(texts, model=self.model, input_type="document")
        return result.embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query via Voyage API."""
        result = self.client.embed([text], model=self.model, input_type="query")
        return result.embeddings[0]


class NomicEmbeddings(Embeddings):
    """LangChain-compatible wrapper for Nomic Atlas embedding API.

    Uses the `nomic` package to call the Atlas embedding endpoint.
    """

    def __init__(self, model: str = "nomic-embed-text-v1.5", api_key: str = "") -> None:
        import nomic

        self.model = model
        nomic.login(api_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents via Nomic Atlas API."""
        from nomic import embed

        result = embed.text(
            texts=texts,
            model=self.model,
            task_type="search_document",
        )
        return [list(vec) for vec in result["embeddings"]]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query via Nomic Atlas API."""
        from nomic import embed

        result = embed.text(
            texts=[text],
            model=self.model,
            task_type="search_query",
        )
        return list(result["embeddings"][0])


# ═════════════════════════════════════════════════════════════════════════════
# Provider-specific initialization functions
# ═════════════════════════════════════════════════════════════════════════════


def _try_hf_inference_embeddings(model_id: str) -> Embeddings | None:
    """Try to create a HuggingFace Inference API embedding model."""
    if not settings.hf_token:
        logger.info("Skipping HF Inference — HF_TOKEN not set")
        return None

    try:
        model = HuggingFaceInferenceEmbeddings(
            model_id=model_id,
            api_key=settings.hf_token,
        )
        # Verify with a quick test embed
        test_vec = model.embed_query("test")
        logger.info(
            "HF Inference embeddings ready: %s (%d dims)",
            model_id,
            len(test_vec),
        )
        return model
    except Exception as exc:
        logger.warning("HF Inference embeddings failed for %s: %s", model_id, exc)
        return None


def _try_hf_local_embeddings(model_id: str) -> Embeddings | None:
    """Try to create a local HuggingFace embedding model via sentence-transformers.

    Downloads the model on first use (~1.2GB for Qwen3-0.6B).
    Cached locally after first download.
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        model = HuggingFaceEmbeddings(
            model_name=model_id,
            model_kwargs={"device": "cpu", "trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
        )
        # Verify with a quick test embed
        test_vec = model.embed_query("test")
        logger.info(
            "HF Local embeddings ready: %s (%d dims)",
            model_id,
            len(test_vec),
        )
        return model
    except Exception as exc:
        logger.warning("HF Local embeddings failed for %s: %s", model_id, exc)
        return None

def _try_voyage_embeddings(model_id: str) -> Embeddings | None:
    """Try to create a Voyage AI embedding model."""
    if not settings.voyage_api_key:
        logger.info("Skipping Voyage embeddings — VOYAGE_API_KEY not set")
        return None

    try:
        model = VoyageEmbeddings(model=model_id, api_key=settings.voyage_api_key)
        # Verify with a quick test embed
        test_vec = model.embed_query("test")
        logger.info(
            "Voyage embeddings ready: %s (%d dims)",
            model_id,
            len(test_vec),
        )
        return model
    except Exception as exc:
        logger.warning("Voyage embeddings failed for %s: %s", model_id, exc)
        return None


def _try_nomic_embeddings(model_id: str) -> Embeddings | None:
    """Try to create a Nomic Atlas embedding model."""
    if not settings.nomic_api_key:
        logger.info("Skipping Nomic embeddings — NOMIC_API_KEY not set")
        return None

    try:
        model = NomicEmbeddings(model=model_id, api_key=settings.nomic_api_key)
        # Verify with a quick test embed
        test_vec = model.embed_query("test")
        logger.info(
            "Nomic embeddings ready: %s (%d dims)",
            model_id,
            len(test_vec),
        )
        return model
    except Exception as exc:
        logger.warning("Nomic embeddings failed for %s: %s", model_id, exc)
        return None


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
        # Verify with a quick test embed
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


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════


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

        if provider == "hf_inference":
            model = _try_hf_inference_embeddings(model_id)
        elif provider == "hf_local":
            model = _try_hf_local_embeddings(model_id)
        elif provider == "voyage":
            model = _try_voyage_embeddings(model_id)
        elif provider == "nomic":
            model = _try_nomic_embeddings(model_id)
        elif provider == "openai":
            model = _try_openai_embeddings(model_id)
        else:
            continue

        if model is not None:
            _active_model_name = name
            _active_dimensions = dimensions
            logger.info("✅ Active embedding model: %s (%d dims)", name, dimensions)
            return model

    raise RuntimeError(
        "All embedding models failed. Ensure HF_TOKEN, VOYAGE_API_KEY, "
        "NOMIC_API_KEY, or API_KEY is set in .env."
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
