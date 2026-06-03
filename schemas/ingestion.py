"""
Pydantic models for the document ingestion layer.

Covers both ingestion paths:
  - URL → Jina Reader API (SEC filings: 10-K, 10-Q, 8-K, Proxy)
  - PDF → PyMuPDF extractor (uploaded documents)

Filing types supported: 10-K, 10-Q, 8-K, DEF 14A (Proxy), DEF 14C, annual reports.
"""

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


class IngestionSource(str, Enum):
    """How the document entered the system."""

    URL = "url"  # Jina Reader API path
    PDF = "pdf"  # PyMuPDF upload path


class FilingType(str, Enum):
    """SEC filing type — determines expected sections and parsing strategy."""

    FORM_10K = "10-K"       # Annual report (most comprehensive)
    FORM_10Q = "10-Q"       # Quarterly report
    FORM_8K = "8-K"         # Current report (event-driven)
    PROXY = "DEF 14A"       # Proxy statement (shareholder meeting)
    INFO_STMT = "DEF 14C"   # Information statement
    ANNUAL_REPORT = "annual_report"  # Non-SEC annual report
    EARNINGS = "earnings"   # Earnings call transcript / press release
    OTHER = "other"         # Any other document type


class DocumentMetadata(BaseModel):
    """Metadata attached to every ingested document."""

    source_type: IngestionSource
    company_name: str = Field(..., description="Company name, e.g. 'Apple Inc.'")
    ticker: str = Field(default="", description="Stock ticker, e.g. 'AAPL'")
    filing_type: FilingType = Field(
        default=FilingType.OTHER,
        description="SEC filing type — drives section detection and agent routing",
    )
    fiscal_year: str = Field(default="", description="Fiscal year or period covered")
    source_url: str = Field(default="", description="Original URL (URL ingestion only)")
    file_name: str = Field(
        default="", description="Original filename (PDF ingestion only)"
    )
    ingested_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of ingestion",
    )
    page_count: int = Field(default=0, ge=0, description="Number of pages (PDF only)")

    # ── SEC-specific identifiers (populated when available) ──────────────────
    cik_number: str = Field(
        default="", description="SEC Central Index Key, e.g. '0000320193' (Apple)"
    )
    accession_number: str = Field(
        default="",
        description="SEC accession number, e.g. '0000320193-24-000123'",
    )
    filing_date: str = Field(
        default="", description="Date filed with SEC, e.g. '2024-11-01'"
    )
    period_of_report: str = Field(
        default="",
        description="Period covered, e.g. '2024-09-28' (differs from fiscal_year for 10-Q)",
    )
    industry: str = Field(
        default="", description="Industry classification, e.g. 'Technology'"
    )


class DocumentPage(BaseModel):
    """A single page extracted from a PDF document."""

    page_number: int = Field(..., ge=1, description="1-indexed page number")
    content: str = Field(..., description="Extracted text content of the page")


class RawDocument(BaseModel):
    """Complete document after extraction, before chunking.

    This is the merge point — both the Jina Reader and PyMuPDF paths
    produce a RawDocument that feeds into the Chonkie chunker.
    """

    doc_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique document identifier",
    )
    content: str = Field(..., description="Full extracted text (all pages combined)")
    metadata: DocumentMetadata
    pages: list[DocumentPage] = Field(
        default_factory=list,
        description="Per-page content (populated for PDF ingestion only)",
    )
