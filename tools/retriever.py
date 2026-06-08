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
from retrieval.qdrant_store import DEFAULT_COLLECTION

logger = logging.getLogger(__name__)

# Use the same collection name as QdrantStore
COLLECTION_NAME = DEFAULT_COLLECTION

# Lazy-initialized singletons
_store: QdrantStore | None = None
_bm25: BM25Retriever | None = None
_hybrid: HybridRetriever | None = None


def _get_hybrid() -> HybridRetriever:
    """Lazy-init the hybrid retriever (singleton).

    On first call, also builds the BM25 index from existing Qdrant data
    so both dense AND sparse retrieval are active from the start.
    """
    global _store, _bm25, _hybrid
    if _hybrid is None:
        _store = QdrantStore(collection_name=COLLECTION_NAME)
        _bm25 = BM25Retriever()

        # Auto-build BM25 index from existing Qdrant data
        try:
            count = _bm25.build_from_qdrant(_store.client, COLLECTION_NAME)
            if count > 0:
                logger.info(
                    "BM25 index built from Qdrant: %d chunks indexed", count
                )
            else:
                logger.info("BM25 index empty — no existing data in Qdrant")
        except Exception as exc:
            logger.warning("Failed to build BM25 from Qdrant: %s", exc)

        _hybrid = HybridRetriever(_store, _bm25)
        logger.info(
            "Initialized hybrid retriever for collection '%s' (BM25 ready: %s)",
            COLLECTION_NAME,
            _bm25.is_ready,
        )
    return _hybrid


def rebuild_bm25_index():
    """Rebuild the BM25 index from current Qdrant data.

    Called after new documents are ingested to keep BM25 in sync.
    """
    global _store, _bm25
    if _store is None or _bm25 is None:
        # Force initialization
        _get_hybrid()
    try:
        count = _bm25.build_from_qdrant(_store.client, COLLECTION_NAME)
        logger.info("BM25 index rebuilt: %d chunks re-indexed", count)
    except Exception as exc:
        logger.warning("Failed to rebuild BM25 index: %s", exc)


@tool
def rag_retriever(
    query: str,
    section_filter: str = "",
    top_k: int = 6,
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
    hybrid = _get_hybrid()

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
        doc = Document(
            page_content=chunk.parent_content or chunk.content,
            metadata={
                "section": chunk.section,
                "score": chunk.rrf_score,
                "dense_score": chunk.dense_score,
                "sparse_score": chunk.sparse_score,
            },
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
