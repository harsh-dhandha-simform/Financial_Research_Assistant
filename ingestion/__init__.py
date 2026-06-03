"""
Document ingestion — Jina Reader API (URL) and PyMuPDF (PDF) paths.

Both paths produce a RawDocument that feeds into the Chonkie chunker.

Usage:
    from ingestion import ingest_url   # Module 3 — URL path
    from ingestion import ingest_pdf   # Module 4 — PDF path
"""

from ingestion.jina_reader import ingest_url
from ingestion.pdf_reader import ingest_pdf

__all__ = ["ingest_url", "ingest_pdf"]

