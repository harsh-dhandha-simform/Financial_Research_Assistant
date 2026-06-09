"""
Jina Reader API ingestion — fetches and extracts text content from URLs.

Used for SEC filings (10-K, 10-Q, 8-K, DEF 14A) and any financial web page.
The Jina Reader API converts web pages into clean markdown text, stripping
navigation, ads, and boilerplate.

Data flow:
    URL → Jina Reader API (r.jina.ai) → RawDocument

Usage:
    from ingestion.jina_reader import ingest_url

    doc = ingest_url(
        url="https://www.sec.gov/Archives/edgar/data/...",
        company_name="Apple Inc.",
        ticker="AAPL",
        filing_type=FilingType.FORM_10K,
    )
"""

import logging
from urllib.parse import quote

import requests

from config import settings
from schemas.ingestion import (
    DocumentMetadata,
    FilingType,
    IngestionSource,
    RawDocument,
)

logger = logging.getLogger(__name__)

JINA_READER_BASE = "https://r.jina.ai/"


def _detect_filing_type(url: str) -> FilingType:
    """Best-effort filing type detection from URL patterns.

    Handles common EDGAR URL patterns. Returns FilingType.OTHER
    if the filing type cannot be determined.
    """
    url_lower = url.lower()

    # SEC EDGAR filing patterns
    filing_keywords: dict[str, FilingType] = {
        "10-k": FilingType.FORM_10K,
        "10k": FilingType.FORM_10K,
        "10-q": FilingType.FORM_10Q,
        "10q": FilingType.FORM_10Q,
        "8-k": FilingType.FORM_8K,
        "8k": FilingType.FORM_8K,
        "def14a": FilingType.PROXY,
        "def-14a": FilingType.PROXY,
        "proxy": FilingType.PROXY,
        "def14c": FilingType.INFO_STMT,
        "annual-report": FilingType.ANNUAL_REPORT,
        "annual_report": FilingType.ANNUAL_REPORT,
        "earnings": FilingType.EARNINGS,
    }

    for keyword, ftype in filing_keywords.items():
        if keyword in url_lower:
            return ftype

    return FilingType.OTHER


def ingest_url(
    url: str,
    company_name: str,
    ticker: str = "",
    filing_type: FilingType | None = None,
    fiscal_year: str = "",
    cik_number: str = "",
    accession_number: str = "",
    filing_date: str = "",
    period_of_report: str = "",
    industry: str = "",
) -> RawDocument:
    """Fetch a URL via Jina Reader API and return a RawDocument.

    Args:
        url: The URL to fetch (SEC filing, earnings report, any financial page).
        company_name: Company name (required for metadata).
        ticker: Stock ticker symbol, e.g. 'AAPL'.
        filing_type: SEC filing type. Auto-detected from URL if None.
        fiscal_year: Fiscal year or period, e.g. 'FY2024'.
        cik_number: SEC Central Index Key.
        accession_number: SEC accession number.
        filing_date: Date filed with SEC.
        period_of_report: Period covered by the filing.
        industry: Industry classification.

    Returns:
        RawDocument with the extracted text and populated metadata.

    Raises:
        ValueError: If URL is empty or JINA_API_KEY is not configured.
        requests.HTTPError: If the Jina Reader API returns a non-2xx status.
    """
    if not url.strip():
        raise ValueError("URL cannot be empty")
    if not settings.jina_api_key:
        raise ValueError(
            "JINA_API_KEY not configured. Add it to your .env file."
        )

    # Auto-detect filing type if not provided
    resolved_filing_type = filing_type or _detect_filing_type(url)

    logger.info("Ingesting URL via Jina Reader: %s", url)

    # ── Call Jina Reader API ─────────────────────────────────────────────────
    # Required headers — image + links extraction enabled per Fixes.md
    headers = {
        "Authorization": f"Bearer {settings.jina_api_key}",
        "Accept": "application/json",
        "X-Keep-Img-Data-Url": "true",
        "X-With-Generated-Alt": "true",
        "X-With-Images-Summary": "all",
        "X-With-Links-Summary": "all",
    }

    response = requests.get(
        f"{JINA_READER_BASE}{quote(url, safe='')}",
        headers=headers,
        timeout=120,
    )
    response.raise_for_status()

    payload = response.json()

    # ── Extract content ──────────────────────────────────────────────────────
    content_data = payload.get("data", {})
    content: str = content_data.get("content", "")
    title: str = content_data.get("title", "")
    final_url: str = content_data.get("url", url)
    usage: dict = content_data.get("usage", {})

    # ── Extract image and link metadata ──────────────────────────────────
    images_data: dict = content_data.get("images", {})
    # images_data = {image_url: {"alt": "...", "description": "..."}, ...}
    images_summary: list = content_data.get("imagesData", [])
    links_summary: list = content_data.get("linksData", [])

    if not content.strip():
        raise ValueError(
            f"Jina Reader returned empty content for URL: {url}"
        )

    # ── Build metadata ───────────────────────────────────────────────────────
    metadata = DocumentMetadata(
        source_type=IngestionSource.URL,
        company_name=company_name,
        ticker=ticker,
        filing_type=resolved_filing_type,
        fiscal_year=fiscal_year,
        source_url=final_url,
        cik_number=cik_number,
        accession_number=accession_number,
        filing_date=filing_date,
        period_of_report=period_of_report,
        industry=industry,
    )

    # ── Build RawDocument ────────────────────────────────────────────────────
    doc = RawDocument(content=content, metadata=metadata)

    # Attach image and links data as extra attributes for downstream processing
    doc._jina_images = images_data          # type: ignore[attr-defined]
    doc._jina_images_summary = images_summary  # type: ignore[attr-defined]
    doc._jina_links_summary = links_summary    # type: ignore[attr-defined]

    token_count = usage.get("tokens", 0)
    logger.info(
        "Ingested %d chars (%d tokens) from %s — title: %s, images: %d",
        len(content),
        token_count,
        url,
        title[:80] if title else "(no title)",
        len(images_data),
    )

    return doc
