"""
PDF exporter — renders InvestmentMemo and ResearchReport as styled PDF files.

Uses Markdown → HTML → PDF pipeline:
  1. markdown_exporter produces Markdown text
  2. `markdown` library converts to HTML
  3. `weasyprint` renders HTML to PDF with embedded CSS styling

Usage:
    from output.pdf_exporter import memo_to_pdf, report_to_pdf

    memo_to_pdf(memo, "/path/to/memo.pdf")
    report_to_pdf(report, "/path/to/report.pdf")
"""

import logging
from pathlib import Path

import markdown
import weasyprint

from schemas.reports import InvestmentMemo, ResearchReport
from output.markdown_exporter import memo_to_markdown, report_to_markdown

logger = logging.getLogger(__name__)


# ── PDF Styling ──────────────────────────────────────────────────────────────
# Professional financial report styling with clean typography.

PDF_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

:root {
    --primary: #1a365d;
    --accent: #2563eb;
    --text: #1e293b;
    --muted: #64748b;
    --border: #e2e8f0;
    --bg: #ffffff;
    --bg-alt: #f8fafc;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 10pt;
    line-height: 1.6;
    color: var(--text);
    background: var(--bg);
}

/* ── Page layout ── */
@page {
    size: A4;
    margin: 2cm 2.5cm;

    @bottom-center {
        content: counter(page) " of " counter(pages);
        font-size: 8pt;
        color: var(--muted);
    }

    @top-right {
        content: "Confidential";
        font-size: 7pt;
        color: var(--muted);
        font-style: italic;
    }
}

@page :first {
    @top-right { content: none; }
}

/* ── Typography ── */
h1 {
    font-size: 20pt;
    font-weight: 700;
    color: var(--primary);
    margin-bottom: 0.5em;
    padding-bottom: 0.3em;
    border-bottom: 3px solid var(--accent);
}

h2 {
    font-size: 14pt;
    font-weight: 600;
    color: var(--primary);
    margin-top: 1.5em;
    margin-bottom: 0.5em;
    padding-bottom: 0.2em;
    border-bottom: 1px solid var(--border);
}

h3 {
    font-size: 12pt;
    font-weight: 600;
    color: var(--text);
    margin-top: 1.2em;
    margin-bottom: 0.4em;
}

h4 {
    font-size: 10pt;
    font-weight: 600;
    color: var(--muted);
    margin-top: 1em;
    margin-bottom: 0.3em;
}

p {
    margin-bottom: 0.8em;
    text-align: justify;
}

/* ── Lists ── */
ul, ol {
    margin-left: 1.5em;
    margin-bottom: 0.8em;
}

li {
    margin-bottom: 0.3em;
}

/* ── Horizontal rules ── */
hr {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.5em 0;
}

/* ── Strong / emphasis ── */
strong {
    font-weight: 600;
    color: var(--primary);
}

em {
    color: var(--muted);
}

/* ── Code blocks (for structured data) ── */
code {
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 9pt;
    background: var(--bg-alt);
    padding: 0.15em 0.3em;
    border-radius: 3px;
}

pre {
    background: var(--bg-alt);
    padding: 1em;
    border-radius: 4px;
    border: 1px solid var(--border);
    overflow-x: auto;
    margin-bottom: 1em;
}

pre code {
    background: none;
    padding: 0;
}

/* ── Tables ── */
table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 1em;
    font-size: 9pt;
}

th {
    background: var(--primary);
    color: white;
    padding: 0.5em 0.8em;
    text-align: left;
    font-weight: 600;
}

td {
    padding: 0.4em 0.8em;
    border-bottom: 1px solid var(--border);
}

tr:nth-child(even) td {
    background: var(--bg-alt);
}

/* ── Disclaimer ── */
.disclaimer {
    margin-top: 2em;
    padding: 1em;
    background: var(--bg-alt);
    border-left: 3px solid var(--muted);
    font-size: 8pt;
    color: var(--muted);
    font-style: italic;
}
"""


def _markdown_to_html(md_text: str) -> str:
    """Convert Markdown text to HTML with extensions."""
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "nl2br"],
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <style>{PDF_CSS}</style>
</head>
<body>
{html_body}
</body>
</html>"""


def memo_to_pdf(memo: InvestmentMemo, output_path: str | Path) -> Path:
    """Render an InvestmentMemo as a styled PDF.

    Args:
        memo: The InvestmentMemo to render.
        output_path: File path for the output PDF.

    Returns:
        The Path to the generated PDF file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    md_text = memo_to_markdown(memo)
    html = _markdown_to_html(md_text)

    weasyprint.HTML(string=html).write_pdf(str(output_path))
    logger.info("Memo PDF generated: %s (%.1f KB)", output_path, output_path.stat().st_size / 1024)
    return output_path


def report_to_pdf(report: ResearchReport, output_path: str | Path) -> Path:
    """Render a ResearchReport as a styled PDF.

    Args:
        report: The ResearchReport to render.
        output_path: File path for the output PDF.

    Returns:
        The Path to the generated PDF file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    md_text = report_to_markdown(report)
    html = _markdown_to_html(md_text)

    weasyprint.HTML(string=html).write_pdf(str(output_path))
    logger.info("Report PDF generated: %s (%.1f KB)", output_path, output_path.stat().st_size / 1024)
    return output_path
