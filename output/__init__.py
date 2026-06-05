"""
Output module — generates and exports InvestmentMemo + ResearchReport.

Public API:
    # Generation (from SynthesisOutput)
    from output import generate_output_fork, generate_memo, generate_report

    # Markdown rendering
    from output import memo_to_markdown, report_to_markdown

    # PDF export
    from output import memo_to_pdf, report_to_pdf

    # DOCX export
    from output import memo_to_docx, report_to_docx
"""

from output.report_generator import (
    generate_output_fork,
    generate_memo,
    generate_report,
)
from output.markdown_exporter import memo_to_markdown, report_to_markdown
from output.pdf_exporter import memo_to_pdf, report_to_pdf
from output.docx_exporter import memo_to_docx, report_to_docx

__all__ = [
    # Generation
    "generate_output_fork",
    "generate_memo",
    "generate_report",
    # Markdown
    "memo_to_markdown",
    "report_to_markdown",
    # PDF
    "memo_to_pdf",
    "report_to_pdf",
    # DOCX
    "memo_to_docx",
    "report_to_docx",
]
