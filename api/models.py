"""
Request/response Pydantic models for the FastAPI layer.

These are API-specific models — they wrap the core pipeline schemas
into HTTP-friendly request/response shapes.
"""

from pydantic import BaseModel, Field


# ── Research pipeline ────────────────────────────────────────────────────────


class AnalyzeURLRequest(BaseModel):
    """Request body for POST /analyze/url."""

    edgar_url: str = Field(..., description="EDGAR filing URL to analyse")
    company_name: str = Field(..., description="Company name, e.g. 'Apple Inc.'")
    ticker: str = Field(default="", description="Stock ticker, e.g. 'AAPL'")
    query: str = Field(
        default="",
        description="Research query. If empty, auto-generated from company name.",
    )


class AnalyzeUploadRequest(BaseModel):
    """Metadata for POST /analyze/upload (multipart form)."""

    company_name: str = Field(..., description="Company name")
    ticker: str = Field(default="", description="Stock ticker")
    query: str = Field(default="", description="Research query")


class AnalyzeResponse(BaseModel):
    """Response for /analyze endpoints."""

    session_id: str = Field(..., description="Pipeline session ID for tracking")
    status: str = Field(default="complete", description="Pipeline status")
    rating: str | None = Field(default=None, description="Investment rating")
    confidence: float | None = Field(default=None, description="Confidence score")
    memo: dict | None = Field(default=None, description="InvestmentMemo as JSON")
    report: dict | None = Field(default=None, description="ResearchReport as JSON")
    errors: list[str] = Field(default_factory=list, description="Any pipeline errors")


# ── Ingestion ────────────────────────────────────────────────────────────────


class IngestURLRequest(BaseModel):
    """Request body for POST /ingest/url."""

    url: str = Field(..., description="URL to ingest (EDGAR filing page)")
    company_name: str = Field(default="", description="Company name for metadata")
    ticker: str = Field(default="", description="Stock ticker for metadata")


class IngestResponse(BaseModel):
    """Response for /ingest endpoints."""

    status: str = Field(default="success", description="Ingestion status")
    document_id: str = Field(default="", description="Ingested document identifier")
    pages_extracted: int = Field(default=0, description="Number of pages extracted")
    chunks_created: int = Field(default=0, description="Number of chunks created")
    chunks_stored: int = Field(default=0, description="Number of chunks stored in Qdrant")
    message: str = Field(default="", description="Status message")


# ── Chat ─────────────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    """Request body for POST /chat."""

    question: str = Field(..., description="User question about analysed company")
    session_id: str = Field(
        default="", description="Session ID to scope context (from /analyze response)"
    )


class ChatResponse(BaseModel):
    """Response for POST /chat."""

    answer: str = Field(..., description="Agent's answer")
    citations: list[str] = Field(
        default_factory=list, description="Source citations"
    )


# ── Export ───────────────────────────────────────────────────────────────────


class ExportRequest(BaseModel):
    """Request body for POST /export."""

    format: str = Field(
        default="markdown",
        description="Export format: markdown, pdf, docx, json",
    )
    document_type: str = Field(
        default="report",
        description="Which document to export: memo or report",
    )


# ── Agent testing ────────────────────────────────────────────────────────────


class AgentTestRequest(BaseModel):
    """Request body for testing individual agents."""

    company_name: str = Field(..., description="Company name")
    ticker: str = Field(default="", description="Stock ticker")
    session_id: str = Field(default="", description="Langfuse session ID")


# ── Status ───────────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str = "healthy"
    version: str = "0.1.0"
    providers: dict[str, bool] = Field(
        default_factory=dict,
        description="Provider API key availability",
    )
    qdrant: bool = Field(default=False, description="Qdrant connection status")
