"""
Export actions — handles DOCX/PDF export button callbacks.

Wires the existing exporters (output/docx_exporter, output/pdf_exporter)
to Chainlit action callbacks. Does NOT rewrite the exporters themselves.
"""

import logging
import os
import tempfile

import chainlit as cl

from output.docx_exporter import memo_to_docx, report_to_docx
from output.pdf_exporter import memo_to_pdf, report_to_pdf
from output.markdown_exporter import memo_to_markdown, report_to_markdown

logger = logging.getLogger(__name__)

# Temp directory for exports
EXPORT_DIR = os.path.join(tempfile.gettempdir(), "financial_assistant_exports")
os.makedirs(EXPORT_DIR, exist_ok=True)


async def export_memo_docx(memo, company_name: str):
    """Export Investment Memo as DOCX file."""
    safe_name = company_name.replace(" ", "_").lower()
    path = os.path.join(EXPORT_DIR, f"{safe_name}_memo.docx")
    memo_to_docx(memo, path)
    elements = [cl.File(name=f"{safe_name}_memo.docx", path=path, display="inline")]
    await cl.Message(content="📥 **Investment Memo — DOCX**", elements=elements).send()


async def export_memo_pdf(memo, company_name: str):
    """Export Investment Memo as PDF file."""
    safe_name = company_name.replace(" ", "_").lower()
    path = os.path.join(EXPORT_DIR, f"{safe_name}_memo.pdf")
    memo_to_pdf(memo, path)
    elements = [cl.File(name=f"{safe_name}_memo.pdf", path=path, display="inline")]
    await cl.Message(content="📥 **Investment Memo — PDF**", elements=elements).send()


async def export_report_docx(report, company_name: str):
    """Export Research Report as DOCX file."""
    safe_name = company_name.replace(" ", "_").lower()
    path = os.path.join(EXPORT_DIR, f"{safe_name}_report.docx")
    report_to_docx(report, path)
    elements = [cl.File(name=f"{safe_name}_report.docx", path=path, display="inline")]
    await cl.Message(content="📥 **Research Report — DOCX**", elements=elements).send()


async def export_report_pdf(report, company_name: str):
    """Export Research Report as PDF file."""
    safe_name = company_name.replace(" ", "_").lower()
    path = os.path.join(EXPORT_DIR, f"{safe_name}_report.pdf")
    report_to_pdf(report, path)
    elements = [cl.File(name=f"{safe_name}_report.pdf", path=path, display="inline")]
    await cl.Message(content="📥 **Research Report — PDF**", elements=elements).send()
