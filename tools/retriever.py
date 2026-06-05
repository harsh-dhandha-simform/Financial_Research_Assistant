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

# Default collection name
COLLECTION_NAME = "financial_docs"

# Lazy-initialized singletons
_store: QdrantStore | None = None
_bm25: BM25Retriever | None = None
_hybrid: HybridRetriever | None = None


def _get_hybrid() -> HybridRetriever:
    """Lazy-init the hybrid retriever (singleton)."""
    global _store, _bm25, _hybrid
    if _hybrid is None:
        _store = QdrantStore(collection_name=COLLECTION_NAME)
        _bm25 = BM25Retriever()
        _hybrid = HybridRetriever(_store, _bm25)
        logger.info("Initialized hybrid retriever for collection '%s'", COLLECTION_NAME)
    return _hybrid


@tool
def rag_retriever(
    query: str,
    section_filter: str = "",
    top_k: int = 8,
) -> list[Document]:
    """Retrieve relevant document chunks using hybrid search (dense + BM25 + RRF).

    Uses Qdrant vector search combined with BM25 keyword search for
    high-quality financial document retrieval. Returns parent chunks
    (~1500-2000 tokens) for full context.

    Args:
        query: Search query, e.g. "Apple revenue growth FY2024".
        section_filter: Filter by section: "mda", "risk_factors", or "" for all.
        top_k: Number of results to return (default: 8).

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
