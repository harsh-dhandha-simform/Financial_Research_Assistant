"""
DOCX exporter — renders InvestmentMemo and ResearchReport as Word documents.

Uses python-docx to build structured .docx files with proper heading
hierarchy, styled tables, and professional formatting.

Usage:
    from output.docx_exporter import memo_to_docx, report_to_docx

    memo_to_docx(memo, "/path/to/memo.docx")
    report_to_docx(report, "/path/to/report.docx")
"""

import logging
from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

from schemas.reports import InvestmentMemo, ResearchReport

logger = logging.getLogger(__name__)


# ── Styling helpers ──────────────────────────────────────────────────────────

NAVY = RGBColor(0x1A, 0x36, 0x5D)
ACCENT = RGBColor(0x25, 0x63, 0xEB)
MUTED = RGBColor(0x64, 0x74, 0x8B)


def _set_font(run, size: int = 10, bold: bool = False, color: RGBColor = None):
    """Apply font styling to a run."""
    run.font.size = Pt(size)
    run.font.bold = bold
    if color:
        run.font.color.rgb = color


def _add_heading(doc: Document, text: str, level: int = 1):
    """Add a styled heading."""
    heading = doc.add_heading(text, level=level)
    for run in heading.runs:
        run.font.color.rgb = NAVY
    return heading


def _add_section_content(doc: Document, content: str):
    """Add section content, splitting by newlines into paragraphs.

    Handles bullet points (lines starting with - or •) and
    bold markers (**text**) in a basic way.
    """
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Bullet points
        if line.startswith(("- ", "• ", "* ")):
            p = doc.add_paragraph(style="List Bullet")
            _add_formatted_text(p, line[2:])
        # Sub-bullets
        elif line.startswith(("  - ", "  • ")):
            p = doc.add_paragraph(style="List Bullet 2")
            _add_formatted_text(p, line[4:])
        # Headings (### or ####)
        elif line.startswith("#### "):
            _add_heading(doc, line[5:], level=4)
        elif line.startswith("### "):
            _add_heading(doc, line[4:], level=3)
        elif line.startswith("## "):
            _add_heading(doc, line[3:], level=2)
        elif line.startswith("---"):
            # Horizontal rule → thin paragraph
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(6)
        else:
            p = doc.add_paragraph()
            _add_formatted_text(p, line)


def _add_formatted_text(paragraph, text: str):
    """Add text with basic **bold** and *italic* formatting."""
    import re

    # Split on **bold** patterns
    parts = re.split(r"(\*\*[^*]+\*\*)", text)
    for part in parts:
        if part.startswith("**") and part.endswith("**"):
            run = paragraph.add_run(part[2:-2])
            run.font.bold = True
            run.font.color.rgb = NAVY
        else:
            # Handle *italic*
            sub_parts = re.split(r"(\*[^*]+\*)", part)
            for sp in sub_parts:
                if sp.startswith("*") and sp.endswith("*") and not sp.startswith("**"):
                    run = paragraph.add_run(sp[1:-1])
                    run.font.italic = True
                    run.font.color.rgb = MUTED
                else:
                    paragraph.add_run(sp)


def _add_disclaimer(doc: Document, text: str):
    """Add a styled disclaimer paragraph."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(18)
    run = p.add_run(text)
    run.font.size = Pt(8)
    run.font.italic = True
    run.font.color.rgb = MUTED


# ═════════════════════════════════════════════════════════════════════════════
# Investment Memo → DOCX
# ═════════════════════════════════════════════════════════════════════════════


def memo_to_docx(memo: InvestmentMemo, output_path: str | Path) -> Path:
    """Render an InvestmentMemo as a Word document.

    Args:
        memo: The InvestmentMemo to render.
        output_path: File path for the output .docx.

    Returns:
        The Path to the generated DOCX file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = Document()

    # Title
    title = doc.add_heading(memo.title, level=0)
    for run in title.runs:
        run.font.color.rgb = NAVY

    # Metadata
    date_str = memo.date.strftime("%B %d, %Y")
    meta = doc.add_paragraph()
    run = meta.add_run(f"Date: {date_str}  |  Rating: {memo.rating.value.replace('_', ' ').upper()}")
    _set_font(run, size=10, color=MUTED)

    doc.add_paragraph()  # spacer

    # Executive Summary
    _add_heading(doc, "Executive Summary", level=1)
    _add_section_content(doc, memo.executive_summary)

    # Key Metrics
    _add_heading(doc, "Key Financial Metrics", level=1)
    _add_section_content(doc, memo.key_metrics)

    # Key Risks
    _add_heading(doc, "Key Risks", level=1)
    _add_section_content(doc, memo.key_risks)

    # Recent News
    _add_heading(doc, "Recent News & Sentiment", level=1)
    _add_section_content(doc, memo.recent_news)

    # Conclusion
    _add_heading(doc, "Conclusion", level=1)
    _add_section_content(doc, memo.conclusion)

    doc.save(str(output_path))
    logger.info("Memo DOCX generated: %s (%.1f KB)", output_path, output_path.stat().st_size / 1024)
    return output_path


# ═════════════════════════════════════════════════════════════════════════════
# Research Report → DOCX
# ═════════════════════════════════════════════════════════════════════════════


def report_to_docx(report: ResearchReport, output_path: str | Path) -> Path:
    """Render a ResearchReport as a Word document.

    Args:
        report: The ResearchReport to render.
        output_path: File path for the output .docx.

    Returns:
        The Path to the generated DOCX file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = Document()

    # Title
    title = doc.add_heading(report.title, level=0)
    for run in title.runs:
        run.font.color.rgb = NAVY

    # Metadata
    date_str = report.date.strftime("%B %d, %Y")
    rating_display = report.rating.value.replace("_", " ").upper()
    meta = doc.add_paragraph()
    run = meta.add_run(f"Date: {date_str}  |  Analyst: {report.analyst}  |  Rating: {rating_display}")
    _set_font(run, size=10, color=MUTED)

    doc.add_paragraph()

    # Executive Summary
    _add_heading(doc, "Executive Summary", level=1)
    _add_section_content(doc, report.executive_summary)

    # Company Overview
    _add_heading(doc, "Company Overview", level=1)
    _add_section_content(doc, report.company_overview)

    # Financial Analysis
    _add_heading(doc, "Financial Analysis", level=1)
    _add_section_content(doc, report.financial_analysis)

    # Risk Assessment
    _add_heading(doc, "Risk Assessment", level=1)
    _add_section_content(doc, report.risk_assessment)

    # Market Sentiment
    _add_heading(doc, "Market Sentiment & News", level=1)
    _add_section_content(doc, report.market_sentiment)

    # Investment Thesis
    _add_heading(doc, "Investment Thesis", level=1)
    _add_section_content(doc, report.investment_thesis)

    # Strengths / Weaknesses / Catalysts
    _add_heading(doc, "Strengths", level=2)
    for s in report.strengths:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(s)

    _add_heading(doc, "Weaknesses", level=2)
    for w in report.weaknesses:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(w)

    _add_heading(doc, "Catalysts", level=2)
    for c in report.catalysts:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(c)

    # Additional Sections
    if report.additional_sections:
        _add_heading(doc, "Additional Sections", level=1)
        for section in report.additional_sections:
            _add_heading(doc, section.title, level=2)
            _add_section_content(doc, section.content)

    # Conclusion
    _add_heading(doc, "Conclusion & Recommendation", level=1)
    _add_section_content(doc, report.conclusion)

    # Disclaimer
    _add_disclaimer(doc, report.disclaimer)

    doc.save(str(output_path))
    logger.info("Report DOCX generated: %s (%.1f KB)", output_path, output_path.stat().st_size / 1024)
    return output_path
