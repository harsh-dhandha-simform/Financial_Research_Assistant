"""
rag_retriever — hybrid retrieval tool for agents.

Uses Qdrant dense + BM25 sparse → RRF fusion → parent chunk expansion.
Supports section-based filtering for targeted retrieval.

Used by: Metrics Agent, Risk Agent, Synthesis Agent, Chat Agent.
"""

import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_core.tools import tool

from retrieval import QdrantStore, BM25Retriever, HybridRetriever

logger = logging.getLogger(__name__)

# We keep a cache of initialized hybrid retrievers keyed by collection_name
# to avoid recreating Qdrant/BM25 clients on every single tool call within a session.
_retriever_cache: dict[str, HybridRetriever] = {}

def _get_hybrid(collection_name: str) -> HybridRetriever:
    """Lazy-init the hybrid retriever for a specific collection.

    On first call for a collection, also builds the BM25 index from existing
    Qdrant data so both dense AND sparse retrieval are active.
    """
    if collection_name not in _retriever_cache:
        store = QdrantStore(collection_name=collection_name)
        bm25 = BM25Retriever()

        # Auto-build BM25 index from existing Qdrant data
        try:
            count = bm25.build_from_qdrant(store.client, collection_name)
            if count > 0:
                logger.info(
                    "BM25 index built from Qdrant for '%s': %d chunks indexed",
                    collection_name, count
                )
            else:
                logger.info("BM25 index empty — no existing data in Qdrant for '%s'", collection_name)
        except Exception as exc:
            logger.warning("Failed to build BM25 from Qdrant for '%s': %s", collection_name, exc)

        hybrid = HybridRetriever(store, bm25)
        _retriever_cache[collection_name] = hybrid
        logger.info(
            "Initialized hybrid retriever for collection '%s' (BM25 ready: %s)",
            collection_name,
            bm25.is_ready,
        )
        
    return _retriever_cache[collection_name]


def rebuild_bm25_index(collection_name: str):
    """Rebuild the BM25 index from current Qdrant data for a specific session collection.

    Called after new documents are ingested to keep BM25 in sync.
    """
    try:
        # Force initialization if not exists
        hybrid = _get_hybrid(collection_name)
        count = hybrid.bm25_retriever.build_from_qdrant(
            hybrid.qdrant_store.client, collection_name
        )
        logger.info("BM25 index rebuilt for '%s': %d chunks re-indexed", collection_name, count)
    except Exception as exc:
        logger.warning("Failed to rebuild BM25 index for '%s': %s", collection_name, exc)


def prewarm_bm25_for_user(user_id: str) -> None:
    """Eagerly initialize the hybrid retriever (including BM25) for a user's collection.

    Called at session start so BM25 is ready before the first query.
    Safe to call multiple times — subsequent calls are no-ops (cached).

    Args:
        user_id: The user identifier used to derive the collection name.
    """
    if not user_id:
        return
    try:
        from sessions.session_store import session_store
        collection_name = session_store.get_user_collection(user_id)
        hybrid = _get_hybrid(collection_name)
        if not hybrid.bm25_retriever.is_ready:
            count = hybrid.bm25_retriever.build_from_qdrant(
                hybrid.qdrant_store.client, collection_name
            )
            logger.info(
                "BM25 pre-warmed for user '%s' (collection '%s'): %d chunks",
                user_id, collection_name, count,
            )
        else:
            logger.debug("BM25 already ready for collection '%s' — skipping prewarm", collection_name)
    except Exception as exc:
        logger.warning("BM25 prewarm failed for user '%s': %s", user_id, exc)



from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import InjectedToolArg
from typing import Annotated

@tool
def rag_retriever(
    query: str,
    section_filter: str = "",
    top_k: int = 6,
    config: RunnableConfig = None,
) -> list[Document]:
    """Retrieve relevant document chunks using hybrid search (dense + BM25 + RRF).

    Uses Qdrant vector search combined with BM25 keyword search for
    high-quality financial document retrieval. Returns parent chunks
    (~1500-2000 tokens) for full context.

    Args:
        query: Search query, e.g. "Apple revenue growth FY2024".
        section_filter: Filter by section: "mda", "risk_factors", or "" for all.
        top_k: Number of results to return (default: 6).

    Returns:
        List of LangChain Document objects with metadata (page, section, company, year).
    """
    configurable = config.get("configurable", {}) if config else {}
    session_id = configurable.get("session_id", "")
    user_id = configurable.get("user_id", "")

    # Determine the user's collection name
    # Priority: user_id → session lookup → fallback
    if user_id:
        collection_name = f"fin_{user_id}"
    elif session_id:
        # Look up the session to find the user's collection
        try:
            from sessions.session_store import session_store
            session = session_store.get(session_id)
            if session and session.user_identifier:
                collection_name = session_store.get_user_collection(session.user_identifier)
            elif session and session.collection_name:
                collection_name = session.collection_name
            else:
                logger.warning("rag_retriever: session %s has no user_id, using collection_name field", session_id[:8])
                collection_name = f"fin_{session_id[:8]}"
        except Exception:
            collection_name = f"fin_{session_id[:8]}"
    else:
        logger.warning("rag_retriever called without session_id or user_id. This may cause isolation issues.")
        collection_name = "financial_chunks"

    logger.info("[Read] rag_retriever reading from collection '%s' for session '%s'", collection_name, session_id or user_id)
    hybrid = _get_hybrid(collection_name)

    # Map section names to our internal filter values
    section_map = {
        "md&a": "mda",
        "mda": "mda",
        "risk factors": "risk_factors",
        "risk_factors": "risk_factors",
        "business overview": "business_overview",
        "notes": "notes",
    }
    normalized_section = section_map.get(section_filter.lower(), section_filter) if section_filter else None

    result = hybrid.search(
        query=query,
        top_k=top_k,
        section_filter=normalized_section,
    )

    # Convert to LangChain Documents
    documents = []
    for chunk in result.chunks:
        # Merge chunk-level metadata with retrieval scores
        doc_metadata = {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "section": chunk.section,
            "score": chunk.rrf_score,
            "dense_score": chunk.dense_score,
            "sparse_score": chunk.sparse_score,
            # Child content (the actual retrieval unit, ~500 tokens) is more
            # precise for image crops than the larger parent_content.
            "child_content": chunk.content,
        }
        # Include stored metadata (source_document, page, company_name, etc.)
        if chunk.metadata:
            doc_metadata.update(chunk.metadata)

        doc = Document(
            page_content=chunk.parent_content or chunk.content,
            metadata=doc_metadata,
        )
        documents.append(doc)


    logger.info(
        "rag_retriever: query='%s' section=%s top_k=%d → %d docs",
        query[:50],
        normalized_section or "all",
        top_k,
        len(documents),
    )
    return documents
