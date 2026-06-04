"""
BM25 sparse keyword retriever for child chunks.

BM25 (Best Matching 25) excels at exact keyword matching — critical for
financial terms that dense embeddings may miss:
  - Ticker symbols (AAPL, MSFT)
  - Accounting terms (EBITDA, GAAP, non-GAAP)
  - SEC form types (10-K, 10-Q, DEF 14A)
  - Specific dollar amounts ($394.3 billion)

The BM25 index is built in-memory from child chunks and can be rebuilt
from Qdrant via scroll if needed.

Usage:
    from retrieval.bm25_retriever import BM25Retriever

    bm25 = BM25Retriever()
    bm25.build_index(child_chunks, parent_lookup)
    results = bm25.search("AAPL revenue EBITDA", top_k=10)
"""

import logging
import re

from rank_bm25 import BM25Okapi

from schemas.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for BM25.

    Lowercases, splits on non-alphanumeric chars, filters short tokens.
    Keeps financial tokens like $ amounts and ticker-like terms intact.
    """
    # Lowercase and split on whitespace/punctuation (keep $, %, .)
    tokens = re.findall(r"[a-z0-9$%][a-z0-9$%.,'/-]*", text.lower())
    # Filter out very short noise tokens
    return [t for t in tokens if len(t) >= 2]


class BM25Retriever:
    """In-memory BM25 keyword retriever over child chunks.

    Stores chunk metadata alongside the BM25 index so search results
    include parent_content for context expansion.
    """

    def __init__(self) -> None:
        self._index: BM25Okapi | None = None
        self._chunks: list[dict] = []  # Stored chunk payloads

    @property
    def is_ready(self) -> bool:
        """True if the index has been built."""
        return self._index is not None and len(self._chunks) > 0

    def build_index(
        self,
        chunks: list[dict],
    ) -> int:
        """Build the BM25 index from chunk payloads.

        Each payload dict must have at least: chunk_id, content, doc_id, section.
        Optionally: parent_content, parent_id, metadata.

        This accepts raw dicts (not ChildChunk) so it can also be built from
        Qdrant scroll payloads directly.

        Args:
            chunks: List of chunk payload dicts.

        Returns:
            Number of chunks indexed.
        """
        if not chunks:
            logger.warning("No chunks provided for BM25 index")
            return 0

        self._chunks = chunks
        corpus = [_tokenize(chunk.get("content", "")) for chunk in chunks]

        self._index = BM25Okapi(corpus)
        logger.info("Built BM25 index: %d documents", len(chunks))
        return len(chunks)

    def build_from_child_chunks(
        self,
        children: list,
        parent_lookup: dict | None = None,
    ) -> int:
        """Build the BM25 index from ChildChunk schema objects.

        Convenience method that converts ChildChunks to payload dicts
        and includes parent_content for context expansion.

        Args:
            children: List of ChildChunk objects.
            parent_lookup: Mapping of parent_id → ParentChunk.

        Returns:
            Number of chunks indexed.
        """
        parent_lookup = parent_lookup or {}
        payloads = []
        for child in children:
            parent = parent_lookup.get(child.parent_id)
            payloads.append({
                "chunk_id": child.chunk_id,
                "parent_id": child.parent_id,
                "doc_id": child.doc_id,
                "section": child.section,
                "content": child.content,
                "parent_content": parent.content if parent else "",
                "chunk_index": child.chunk_index,
                "token_count": child.token_count,
                "metadata": child.metadata,
            })
        return self.build_index(payloads)

    def build_from_qdrant(self, qdrant_client, collection_name: str) -> int:
        """Rebuild the BM25 index by scrolling all points in a Qdrant collection.

        Used to sync the BM25 index with whatever's stored in Qdrant.

        Args:
            qdrant_client: A QdrantClient instance.
            collection_name: Name of the Qdrant collection.

        Returns:
            Number of chunks indexed.
        """
        all_payloads: list[dict] = []
        offset = None

        while True:
            results, next_offset = qdrant_client.scroll(
                collection_name=collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in results:
                if point.payload:
                    all_payloads.append(point.payload)
            if next_offset is None:
                break
            offset = next_offset

        logger.info(
            "Scrolled %d points from Qdrant collection '%s'",
            len(all_payloads),
            collection_name,
        )
        return self.build_index(all_payloads)

    def search(
        self,
        query: str,
        top_k: int = 10,
        section_filter: str | None = None,
        doc_id_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """Search the BM25 index with a keyword query.

        Args:
            query: Natural language search query.
            top_k: Number of results to return.
            section_filter: Only return chunks from this section.
            doc_id_filter: Only return chunks from this document.

        Returns:
            List of RetrievedChunk sorted by BM25 score (descending).
        """
        if not self.is_ready:
            logger.warning("BM25 index not built — returning empty results")
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        # Get BM25 scores for all documents
        scores = self._index.get_scores(query_tokens)

        # Build (index, score) pairs and apply filters
        scored: list[tuple[int, float]] = []
        for idx, score in enumerate(scores):
            if score <= 0:
                continue
            chunk = self._chunks[idx]
            if section_filter and chunk.get("section") != section_filter:
                continue
            if doc_id_filter and chunk.get("doc_id") != doc_id_filter:
                continue
            scored.append((idx, float(score)))

        # Sort by score descending, take top_k
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]

        # Convert to RetrievedChunk objects
        results: list[RetrievedChunk] = []
        for idx, score in top:
            chunk = self._chunks[idx]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.get("chunk_id", ""),
                    content=chunk.get("content", ""),
                    parent_content=chunk.get("parent_content", ""),
                    doc_id=chunk.get("doc_id", ""),
                    section=chunk.get("section", ""),
                    score=score,
                    sparse_score=score,
                    retrieval_method="sparse",
                )
            )

        logger.info(
            "BM25 search '%s' → %d results (section=%s)",
            query[:50],
            len(results),
            section_filter or "all",
        )
        return results
