"""
Ingest router — ingest documents into Qdrant without running the pipeline.

POST /ingest/url     → ingest from URL (Jina Reader)
POST /ingest/upload  → ingest from uploaded PDF
GET  /ingest/status  → check Qdrant collection info
"""

import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from api.models import IngestResponse
from ingestion.jina_reader import ingest_url
from ingestion.pdf_reader import ingest_pdf
from chunking.chunker import chunk_document
from retrieval.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingest"])


import uuid

def _chunk_and_store(raw_doc, session_id: str) -> dict:
    """Chunk a RawDocument and store in Qdrant."""
    parents, children = chunk_document(raw_doc)
    parent_lookup = {p.chunk_id: p for p in parents}
    
    collection_name = f"fin_{session_id[:8]}"
    store = QdrantStore(collection_name)
    stored = store.upsert_chunks(children, parent_lookup)
    return {
        "pages": len(raw_doc.pages),
        "parents": len(parents),
        "children": len(children),
        "chunks_stored": stored,
    }


@router.post("/url", response_model=IngestResponse)
async def ingest_from_url(
    url: str,
    company_name: str = "",
    ticker: str = "",
    session_id: str = "",
):
    """Ingest a document from URL into Qdrant.

    Uses Jina Reader to extract content, chunks it, and stores
    in the vector database. Does NOT run the research pipeline.
    """
    if not session_id:
        session_id = f"api-{uuid.uuid4().hex[:8]}"

    try:
        raw_doc = ingest_url(url=url, company_name=company_name)
        stats = _chunk_and_store(raw_doc, session_id)

        return IngestResponse(
            status="success",
            document_id=raw_doc.metadata.source_id,
            pages_extracted=stats["pages"],
            chunks_created=stats["children"],
            chunks_stored=stats["chunks_stored"],
            message=(
                f"Ingested {stats['pages']} pages → "
                f"{stats['parents']} parents, {stats['children']} children, "
                f"{stats['chunks_stored']} stored in Qdrant for session {session_id}"
            ),
        )
    except Exception as exc:
        logger.error("URL ingestion failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")


@router.post("/upload", response_model=IngestResponse)
async def ingest_from_upload(
    file: UploadFile = File(...),
    company_name: str = Form(""),
    ticker: str = Form(""),
    session_id: str = Form(""),
):
    """Ingest a PDF into Qdrant.

    Extracts text via PyMuPDF, chunks it, and stores in the
    vector database. Does NOT run the research pipeline.
    """
    if not session_id:
        session_id = f"api-{uuid.uuid4().hex[:8]}"

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        content = await file.read()
        raw_doc = ingest_pdf(
            file_bytes=content,
            file_name=file.filename,
            company_name=company_name,
        )
        stats = _chunk_and_store(raw_doc, session_id)

        return IngestResponse(
            status="success",
            document_id=raw_doc.metadata.source_id,
            pages_extracted=stats["pages"],
            chunks_created=stats["children"],
            chunks_stored=stats["chunks_stored"],
            message=(
                f"Ingested {stats['pages']} pages from '{file.filename}' → "
                f"{stats['parents']} parents, {stats['children']} children, "
                f"{stats['chunks_stored']} stored in Qdrant"
            ),
        )
    except Exception as exc:
        logger.error("PDF ingestion failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")


@router.get("/status")
async def ingest_status():
    """Check Qdrant collection status (document count, vector size, etc.)."""
    try:
        store = QdrantStore()
        info = store.collection_info()
        return {"status": "connected", "collection": info}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}
