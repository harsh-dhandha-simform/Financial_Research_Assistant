"""
Analyze router — runs the full research pipeline.

POST /analyze/url     → pipeline from EDGAR URL
POST /analyze/upload  → pipeline from uploaded PDF
"""

import logging
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from api.models import AnalyzeResponse
from graph.workflow import run_research
from ingestion.jina_reader import ingest_url
from ingestion.pdf_reader import ingest_pdf
from chunking.chunker import chunk_document
from retrieval.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analyze", tags=["analyze"])


def _ingest_and_store(raw_doc) -> dict:
    """Shared helper: chunk a RawDocument and store in Qdrant.

    Returns:
        Dict with ingestion stats.
    """
    parents, children = chunk_document(raw_doc)
    parent_lookup = {p.chunk_id: p for p in parents}
    store = QdrantStore()
    stored = store.upsert_chunks(children, parent_lookup)
    return {
        "pages": len(raw_doc.pages),
        "parents": len(parents),
        "children": len(children),
        "chunks_stored": stored,
    }


def _run_pipeline(company_name: str, ticker: str, query: str) -> AnalyzeResponse:
    """Run the full research pipeline and return API response."""
    session_id = f"api-{company_name.lower().replace(' ', '-')}-{uuid.uuid4().hex[:6]}"

    if not query:
        query = f"Analyse the latest SEC filings for {company_name} ({ticker})"

    try:
        state = run_research(
            query=query,
            company_name=company_name,
            ticker=ticker,
            session_id=session_id,
        )

        # Cache results for /export endpoints
        if state.memo and state.report:
            from api.routers.export import store_result
            store_result(session_id, state.memo, state.report)

        return AnalyzeResponse(
            session_id=session_id,
            status="complete",
            rating=state.synthesis_output.rating.value if state.synthesis_output else None,
            confidence=state.synthesis_output.confidence if state.synthesis_output else None,
            memo=state.memo.model_dump(mode="json") if state.memo else None,
            report=state.report.model_dump(mode="json") if state.report else None,
            errors=state.errors,
        )
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc)
        return AnalyzeResponse(
            session_id=session_id,
            status="failed",
            errors=[str(exc)],
        )


@router.post("/url", response_model=AnalyzeResponse)
async def analyze_url(
    edgar_url: str,
    company_name: str,
    ticker: str = "",
    query: str = "",
):
    """Run full research pipeline from an EDGAR filing URL.

    1. Ingests the URL via Jina Reader
    2. Chunks and stores in Qdrant
    3. Runs the multi-agent pipeline
    4. Returns InvestmentMemo + ResearchReport
    """
    try:
        raw_doc = ingest_url(url=edgar_url, company_name=company_name)
        stats = _ingest_and_store(raw_doc)
        logger.info(
            "Ingested URL: %d pages, %d parents, %d children, %d stored",
            stats["pages"], stats["parents"], stats["children"], stats["chunks_stored"],
        )
    except Exception as exc:
        logger.warning("URL ingestion failed (pipeline will use search): %s", exc)

    return _run_pipeline(company_name, ticker, query)


@router.post("/upload", response_model=AnalyzeResponse)
async def analyze_upload(
    file: UploadFile = File(...),
    company_name: str = Form(...),
    ticker: str = Form(""),
    query: str = Form(""),
):
    """Run full research pipeline from an uploaded PDF.

    1. Extracts text via PyMuPDF
    2. Chunks and stores in Qdrant
    3. Runs the multi-agent pipeline
    4. Returns InvestmentMemo + ResearchReport
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    content = await file.read()
    try:
        raw_doc = ingest_pdf(
            file_bytes=content,
            file_name=file.filename,
            company_name=company_name,
        )
        stats = _ingest_and_store(raw_doc)
        logger.info(
            "Ingested PDF: %d pages, %d parents, %d children, %d stored",
            stats["pages"], stats["parents"], stats["children"], stats["chunks_stored"],
        )
    except Exception as exc:
        logger.warning("PDF ingestion failed (pipeline will use search): %s", exc)

    return _run_pipeline(company_name, ticker, query)
