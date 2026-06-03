"""
Pydantic models for the parent-child chunk hierarchy.

Architecture:
  ParentChunk (~2000 tokens)  — broad context window, NOT directly retrieved.
  ChildChunk  (~500 tokens)   — retrieval unit, embedded and stored in Qdrant.

Each ChildChunk links back to its ParentChunk via parent_id.
When a ChildChunk is retrieved, the system expands context by loading
the parent chunk text for the LLM prompt.
"""

from uuid import uuid4

from pydantic import BaseModel, Field


# ── Standardised section labels ──────────────────────────────────────────────
# Use these values in the `section` field for consistent retrieval filtering.
# Keys are the canonical values; descriptions explain what they map to.
STANDARD_SECTIONS: dict[str, str] = {
    "business": "Business overview, products, segments (10-K Item 1)",
    "risk_factors": "Key risk factors (10-K Item 1A, 10-Q Part II Item 1A)",
    "mda": "Management Discussion & Analysis (10-K Item 7, 10-Q Item 2)",
    "financial_statements": "Financial statements & notes (10-K Item 8, 10-Q Item 1)",
    "market_risk": "Quantitative disclosures about market risk (10-K Item 7A)",
    "legal_proceedings": "Legal proceedings (10-K Item 3)",
    "executive_compensation": "Executive compensation (Proxy / 10-K Part III)",
    "corporate_governance": "Board & governance information (Proxy)",
    "shareholder_proposals": "Shareholder proposals (Proxy)",
    "material_event": "Material event disclosure (8-K)",
    "exhibits": "Exhibits and schedules",
    "other": "Sections not matching standard categories",
}


class ParentChunk(BaseModel):
    """Large chunk (~2000 tokens) providing broad context.

    Parent chunks are NOT directly retrieved — they exist so that
    when a child is matched, the LLM gets a wider context window.
    """

    chunk_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique chunk identifier",
    )
    doc_id: str = Field(..., description="ID of the source RawDocument")
    content: str = Field(..., description="Chunk text (~2000 tokens)")
    chunk_index: int = Field(
        ..., ge=0, description="Sequential position in the document"
    )
    token_count: int = Field(default=0, ge=0, description="Actual token count")
    section: str = Field(
        default="",
        description=(
            "Standardised section label from STANDARD_SECTIONS, "
            "e.g. 'risk_factors', 'mda', 'financial_statements'"
        ),
    )
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class ChildChunk(BaseModel):
    """Small chunk (~500 tokens) used as the retrieval unit.

    Child chunks are embedded and stored in Qdrant for dense retrieval,
    and indexed for BM25 sparse retrieval. Each child links back to
    its parent for context expansion during answer generation.
    """

    chunk_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique chunk identifier",
    )
    parent_id: str = Field(..., description="ID of the parent ParentChunk")
    doc_id: str = Field(..., description="ID of the source RawDocument")
    content: str = Field(..., description="Chunk text (~500 tokens)")
    chunk_index: int = Field(
        ..., ge=0, description="Position within the parent chunk"
    )
    token_count: int = Field(default=0, ge=0, description="Actual token count")
    section: str = Field(
        default="",
        description=(
            "Standardised section label from STANDARD_SECTIONS, "
            "e.g. 'risk_factors', 'mda', 'financial_statements'"
        ),
    )
    metadata: dict = Field(default_factory=dict, description="Additional metadata")
