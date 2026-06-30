"""
PyMuPDF PDF ingestion — extracts text page-by-page from uploaded PDFs.

Used for uploaded SEC filings, annual reports, and any financial PDF document.
PyMuPDF extracts clean text from each page, preserving reading order.

Data flow:
    PDF file/bytes → PyMuPDF extractor → RawDocument (with DocumentPages)

Usage:
    from ingestion.pdf_reader import ingest_pdf

    # From file path
    doc = ingest_pdf(file_path="apple_10k.pdf", company_name="Apple Inc.")

    # From bytes (e.g. FastAPI UploadFile)
    doc = ingest_pdf(file_bytes=upload.read(), file_name="apple_10k.pdf",
                     company_name="Apple Inc.", filing_type=FilingType.FORM_10K)
"""

import logging
from pathlib import Path

import pymupdf

from schemas.ingestion import (
    DocumentMetadata,
    DocumentPage,
    FilingType,
    IngestionSource,
    RawDocument,
)

logger = logging.getLogger(__name__)


def _detect_filing_type_from_name(file_name: str) -> FilingType:
    """Best-effort filing type detection from the PDF filename."""
    name_lower = file_name.lower()

    patterns: dict[str, FilingType] = {
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
        "annual": FilingType.ANNUAL_REPORT,
        "earnings": FilingType.EARNINGS,
    }

    for pattern, ftype in patterns.items():
        if pattern in name_lower:
            return ftype

    return FilingType.OTHER


def ingest_pdf(
    file_path: str | None = None,
    file_bytes: bytes | None = None,
    file_name: str = "",
    company_name: str = "",
    ticker: str = "",
    filing_type: FilingType | None = None,
    fiscal_year: str = "",
    cik_number: str = "",
    accession_number: str = "",
    filing_date: str = "",
    period_of_report: str = "",
    industry: str = "",
) -> RawDocument:
    """Extract text from a PDF and return a RawDocument.

    Provide either file_path or file_bytes (not both).

    Args:
        file_path: Path to a PDF file on disk.
        file_bytes: Raw PDF bytes (e.g. from a FastAPI UploadFile).
        file_name: Original filename (used for filing type detection and metadata).
        company_name: Company name (required for metadata).
        ticker: Stock ticker symbol.
        filing_type: SEC filing type. Auto-detected from filename if None.
        fiscal_year: Fiscal year or period.
        cik_number: SEC Central Index Key.
        accession_number: SEC accession number.
        filing_date: Date filed with SEC.
        period_of_report: Period covered by the filing.
        industry: Industry classification.

    Returns:
        RawDocument with per-page text in the `pages` field and
        concatenated full text in the `content` field.

    Raises:
        ValueError: If neither file_path nor file_bytes is provided.
        FileNotFoundError: If file_path does not exist.
        RuntimeError: If PyMuPDF fails to extract text.
    """
    if not file_path and not file_bytes:
        raise ValueError("Provide either file_path or file_bytes")
    if file_path and file_bytes:
        raise ValueError("Provide file_path or file_bytes, not both")

    # ── Resolve file name ────────────────────────────────────────────────────
    if file_path:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")
        if not file_name:
            file_name = path.name

    # Auto-detect filing type from filename
    resolved_filing_type = filing_type or _detect_filing_type_from_name(file_name)

    logger.info("Ingesting PDF via PyMuPDF: %s", file_name or "(bytes)")

    # ── Open and extract ─────────────────────────────────────────────────────
    try:
        if file_path:
            pdf = pymupdf.open(file_path)
        else:
            pdf = pymupdf.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise RuntimeError(f"Failed to open PDF: {exc}") from exc

    pages: list[DocumentPage] = []
    all_text_parts: list[str] = []

    for page_num in range(len(pdf)):
        page = pdf[page_num]
        text = page.get_text("text")

        # ── Extract structured tables from the page ──────────────────────
        # PyMuPDF find_tables() detects tabular structures and returns them
        # as structured objects. We convert them to markdown tables and
        # append them after the page text so agents can parse them properly.
        table_md_parts = []
        try:
            tables = page.find_tables()
            for table in tables:
                rows = table.extract()
                if not rows or len(rows) < 2:
                    continue
                # Build markdown table from extracted rows
                # First row is header
                header = rows[0]
                header_clean = [str(h).strip() if h else "" for h in header]
                md_lines = [
                    "| " + " | ".join(header_clean) + " |",
                    "| " + " | ".join(["---"] * len(header_clean)) + " |",
                ]
                for row in rows[1:]:
                    cells = [str(c).strip() if c else "" for c in row]
                    md_lines.append("| " + " | ".join(cells) + " |")
                table_md_parts.append("\n".join(md_lines))
        except Exception:
            pass  # Some pages may not have tables; silently continue

        # Combine text + any extracted tables
        if table_md_parts:
            tables_block = (
                "\n\n[FINANCIAL TABLE START]\n"
                + "\n\n".join(table_md_parts)
                + "\n[FINANCIAL TABLE END]\n"
            )
            combined_text = text + tables_block
        else:
            combined_text = text

        if combined_text.strip():
            pages.append(
                DocumentPage(
                    page_number=page_num + 1,
                    content=combined_text,
                )
            )
            all_text_parts.append(combined_text)

    pdf.close()

    full_content = "\n\n".join(all_text_parts)

    if not full_content.strip():
        raise RuntimeError(
            f"PyMuPDF extracted no text from {file_name or 'PDF'}. "
            "The PDF may be scanned/image-based (OCR not supported)."
        )

    # ── Build metadata ───────────────────────────────────────────────────────
    metadata = DocumentMetadata(
        source_type=IngestionSource.PDF,
        company_name=company_name,
        ticker=ticker,
        filing_type=resolved_filing_type,
        fiscal_year=fiscal_year,
        file_name=file_name,
        page_count=len(pages),
        cik_number=cik_number,
        accession_number=accession_number,
        filing_date=filing_date,
        period_of_report=period_of_report,
        industry=industry,
    )

    # ── Build RawDocument ────────────────────────────────────────────────────
    doc = RawDocument(
        content=full_content,
        metadata=metadata,
        pages=pages,
    )

    logger.info(
        "Extracted %d chars from %d pages (%s)",
        len(full_content),
        len(pages),
        file_name or "bytes",
    )

    return doc
