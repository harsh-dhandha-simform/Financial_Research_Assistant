"""
Pydantic models for hybrid retrieval results.

The retrieval pipeline:
  1. Query hits both Dense Vector (Qdrant semantic) and BM25 (sparse keyword)
  2. Results are fused via Reciprocal Rank Fusion (RRF)
  3. Top-k RetrievedChunks are returned, each with parent context expanded
"""

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """A single chunk returned by the hybrid retriever after RRF fusion."""

    chunk_id: str = Field(..., description="ID of the matched child chunk")
    content: str = Field(..., description="Child chunk text (retrieval unit)")
    parent_content: str = Field(
        default="",
        description="Parent chunk text for expanded context in LLM prompts",
    )
    doc_id: str = Field(..., description="Source document ID")
    section: str = Field(default="", description="Document section name")

    # ── Retrieval scores ─────────────────────────────────────────────────────
    rrf_score: float = Field(..., description="Final Reciprocal Rank Fusion score")
    dense_score: float = Field(
        default=0.0, description="Dense (semantic) similarity score"
    )
    sparse_score: float = Field(default=0.0, description="BM25 sparse match score")

    metadata: dict = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """Complete result set from a hybrid retrieval query.

    Used by all agents to access document context.
    The context_text property pre-formats chunks for LLM prompts.
    """

    query: str = Field(..., description="The search query")
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    total_candidates: int = Field(
        default=0, description="Total chunks scored before top-k cutoff"
    )
    top_k: int = Field(default=8, description="Number of chunks returned")

    @property
    def context_text(self) -> str:
        """Concatenate retrieved chunks into a formatted context string.

        Uses parent_content when available (expanded context),
        falls back to child content otherwise.
        """
        parts: list[str] = []
        for i, chunk in enumerate(self.chunks, 1):
            header = f"--- Chunk {i}"
            if chunk.section:
                header += f" [Section: {chunk.section}]"
            header += " ---"
            text = chunk.parent_content or chunk.content
            parts.append(f"{header}\n{text}")
        return "\n\n".join(parts)

    @property
    def is_empty(self) -> bool:
        """True if no chunks were retrieved."""
        return len(self.chunks) == 0
