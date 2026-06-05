"""
Export router — download reports in various formats.

POST /export  → export memo or report as Markdown, PDF, DOCX, or JSON
"""

import io
import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from api.models import ExportRequest
from schemas.agents import SynthesisOutput, InvestmentRating
from schemas.reports import InvestmentMemo, ResearchReport
from output.markdown_exporter import memo_to_markdown, report_to_markdown
from output.pdf_exporter import memo_to_pdf, report_to_pdf
from output.docx_exporter import memo_to_docx, report_to_docx

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/export", tags=["export"])

# In-memory cache of last pipeline results for export.
# In production, this would be a proper session store.
_last_results: dict[str, dict] = {}


def store_result(session_id: str, memo: InvestmentMemo, report: ResearchReport):
    """Store pipeline results for later export."""
    _last_results[session_id] = {"memo": memo, "report": report}


def _get_result(session_id: str) -> dict:
    """Retrieve stored result by session_id."""
    if session_id not in _last_results:
        raise HTTPException(
            status_code=404,
            detail=f"No results found for session '{session_id}'. Run /analyze first.",
        )
    return _last_results[session_id]


@router.post("/markdown")
async def export_markdown(session_id: str, document_type: str = "report"):
    """Export memo or report as Markdown text."""
    result = _get_result(session_id)

    if document_type == "memo":
        md = memo_to_markdown(result["memo"])
    else:
        md = report_to_markdown(result["report"])

    return {"format": "markdown", "document_type": document_type, "content": md}


@router.post("/pdf")
async def export_pdf(session_id: str, document_type: str = "report"):
    """Export memo or report as PDF file download."""
    result = _get_result(session_id)

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        if document_type == "memo":
            memo_to_pdf(result["memo"], tmp.name)
            filename = f"{result['memo'].company_name}_memo.pdf"
        else:
            report_to_pdf(result["report"], tmp.name)
            filename = f"{result['report'].company_name}_report.pdf"

        return FileResponse(
            path=tmp.name,
            media_type="application/pdf",
            filename=filename,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {exc}")


@router.post("/docx")
async def export_docx(session_id: str, document_type: str = "report"):
    """Export memo or report as DOCX file download."""
    result = _get_result(session_id)

    tmp = tempfile.NamedTemporaryFile(suffix=".docx", delete=False)
    try:
        if document_type == "memo":
            memo_to_docx(result["memo"], tmp.name)
            filename = f"{result['memo'].company_name}_memo.docx"
        else:
            report_to_docx(result["report"], tmp.name)
            filename = f"{result['report'].company_name}_report.docx"

        return FileResponse(
            path=tmp.name,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            filename=filename,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"DOCX generation failed: {exc}")


@router.post("/json")
async def export_json(session_id: str, document_type: str = "report"):
    """Export memo or report as raw JSON."""
    result = _get_result(session_id)

    if document_type == "memo":
        data = result["memo"].model_dump(mode="json")
    else:
        data = result["report"].model_dump(mode="json")

    return {"format": "json", "document_type": document_type, "data": data}
