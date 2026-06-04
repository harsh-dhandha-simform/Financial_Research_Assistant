"""
Hybrid retriever — fuses Dense (Qdrant) + Sparse (BM25) via RRF.

Reciprocal Rank Fusion (RRF) formula:
    RRF_score(doc) = Σ 1 / (k + rank_i(doc))
where k=60 (standard constant) and rank_i is the position in each ranker.

Why hybrid?
  - Dense search (semantic) understands meaning: "company earnings" matches
    "net income" and "revenue growth"
  - Sparse search (BM25) catches exact terms: "EBITDA", "AAPL", "$394.3B",
    "10-K" — terms that dense embeddings often miss or conflate

The fusion ensures both signals contribute to the final ranking.

Pipeline:
    Query → Dense search (Qdrant) → ranked list
    Query → Sparse search (BM25)  → ranked list
    Both  → RRF fusion → final RetrievalResult

Usage:
    from retrieval.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever(qdrant_store, bm25_retriever)
    result = retriever.search("Apple revenue growth FY2024", top_k=8)
    print(result.context_text)
"""

import logging
from collections import defaultdict

from retrieval.bm25_retriever import BM25Retriever
from retrieval.qdrant_store import QdrantStore
from schemas.retrieval import RetrievedChunk, RetrievalResult

logger = logging.getLogger(__name__)

# RRF constant — standard value from the original paper (Cormack et al., 2009)
RRF_K = 60


def reciprocal_rank_fusion(
    dense_results: list[RetrievedChunk],
    sparse_results: list[RetrievedChunk],
    k: int = RRF_K,
) -> list[RetrievedChunk]:
    """Fuse two ranked lists using Reciprocal Rank Fusion.

    Each chunk's RRF score = sum of 1/(k + rank) across all lists it appears in.
    Chunks that appear in both lists get boosted; chunks in only one list still
    contribute.

    Args:
        dense_results: Ranked results from dense vector search.
        sparse_results: Ranked results from BM25 keyword search.
        k: RRF smoothing constant (default 60).

    Returns:
        Fused list of RetrievedChunks sorted by RRF score (descending).
    """
    # Track scores and best chunk data by chunk_id
    rrf_scores: dict[str, float] = defaultdict(float)
    dense_scores: dict[str, float] = {}
    sparse_scores: dict[str, float] = {}
    chunk_data: dict[str, RetrievedChunk] = {}

    # Score dense results by rank position
    for rank, chunk in enumerate(dense_results, start=1):
        rrf_scores[chunk.chunk_id] += 1.0 / (k + rank)
        dense_scores[chunk.chunk_id] = chunk.score
        chunk_data[chunk.chunk_id] = chunk

    # Score sparse results by rank position
    for rank, chunk in enumerate(sparse_results, start=1):
        rrf_scores[chunk.chunk_id] += 1.0 / (k + rank)
        sparse_scores[chunk.chunk_id] = chunk.score
        # Keep the chunk data from sparse if not already from dense
        if chunk.chunk_id not in chunk_data:
            chunk_data[chunk.chunk_id] = chunk

    # Build fused results
    fused: list[RetrievedChunk] = []
    for chunk_id, rrf_score in rrf_scores.items():
        base = chunk_data[chunk_id]
        fused_chunk = RetrievedChunk(
            chunk_id=chunk_id,
            content=base.content,
            parent_content=base.parent_content,
            doc_id=base.doc_id,
            section=base.section,
            score=rrf_score,
            rrf_score=rrf_score,
            dense_score=dense_scores.get(chunk_id, 0.0),
            sparse_score=sparse_scores.get(chunk_id, 0.0),
            retrieval_method="hybrid",
            metadata=base.metadata,
        )
        fused.append(fused_chunk)

    # Sort by RRF score descending
    fused.sort(key=lambda c: c.rrf_score, reverse=True)
    return fused


class HybridRetriever:
    """Orchestrates dense + sparse retrieval with RRF fusion.

    Runs both retrievers in sequence (dense first, then sparse),
    fuses via RRF, and returns a RetrievalResult ready for agents.
    """

    def __init__(
        self,
        qdrant_store: QdrantStore,
        bm25_retriever: BM25Retriever,
        rrf_k: int = RRF_K,
    ):
        self.qdrant_store = qdrant_store
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 8,
        dense_top_k: int | None = None,
        sparse_top_k: int | None = None,
        section_filter: str | None = None,
        doc_id_filter: str | None = None,
    ) -> RetrievalResult:
        """Run hybrid retrieval: dense + sparse → RRF fusion.

        Fetches more candidates from each retriever than the final top_k
        to improve fusion quality (default: 2x top_k per retriever).

        Args:
            query: Natural language search query.
            top_k: Number of final fused results to return.
            dense_top_k: Candidates from dense search (default: 2*top_k).
            sparse_top_k: Candidates from sparse search (default: 2*top_k).
            section_filter: Filter by document section.
            doc_id_filter: Filter by document ID.

        Returns:
            RetrievalResult with fused chunks and context_text ready for LLM.
        """
        candidate_k = top_k * 2
        dense_k = dense_top_k or candidate_k
        sparse_k = sparse_top_k or candidate_k

        # ── Dense retrieval (Qdrant) ─────────────────────────────────────────
        dense_results = self.qdrant_store.search(
            query=query,
            top_k=dense_k,
            section_filter=section_filter,
            doc_id_filter=doc_id_filter,
        )

        # ── Sparse retrieval (BM25) ──────────────────────────────────────────
        if self.bm25_retriever.is_ready:
            sparse_results = self.bm25_retriever.search(
                query=query,
                top_k=sparse_k,
                section_filter=section_filter,
                doc_id_filter=doc_id_filter,
            )
        else:
            logger.warning("BM25 index not ready — using dense-only retrieval")
            sparse_results = []

        # ── RRF Fusion ───────────────────────────────────────────────────────
        fused = reciprocal_rank_fusion(dense_results, sparse_results, k=self.rrf_k)

        # Trim to final top_k
        final_chunks = fused[:top_k]

        total_candidates = len(set(
            [c.chunk_id for c in dense_results] +
            [c.chunk_id for c in sparse_results]
        ))

        logger.info(
            "Hybrid search '%s' → dense=%d, sparse=%d, fused=%d, final=%d",
            query[:50],
            len(dense_results),
            len(sparse_results),
            len(fused),
            len(final_chunks),
        )

        return RetrievalResult(
            query=query,
            chunks=final_chunks,
            total_candidates=total_candidates,
            top_k=top_k,
        )
