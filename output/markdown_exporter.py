"""
Markdown exporter — renders InvestmentMemo and ResearchReport as Markdown.

Provides clean, readable Markdown output ready for:
  - Display in Streamlit/web UI
  - Export to PDF (via markdown→PDF converters)
  - Saving as .md files
  - Rendering in chat interfaces
"""

import logging
from datetime import datetime

from schemas.reports import InvestmentMemo, ResearchReport

logger = logging.getLogger(__name__)


def memo_to_markdown(memo: InvestmentMemo) -> str:
    """Render an InvestmentMemo as a Markdown string.

    Args:
        memo: The InvestmentMemo to render.

    Returns:
        A formatted Markdown string.
    """
    date_str = memo.date.strftime("%B %d, %Y")

    md = f"""# {memo.title}

**Date:** {date_str}
**Rating:** {memo.rating.value.replace('_', ' ').upper()}

---

## Executive Summary

{memo.executive_summary}

---

## Key Financial Metrics

{memo.key_metrics}

---

## Key Risks

{memo.key_risks}

---

## Recent News & Sentiment

{memo.recent_news}

---

## Conclusion

{memo.conclusion}
"""
    return md.strip()


def report_to_markdown(report: ResearchReport) -> str:
    """Render a ResearchReport as a Markdown string.

    Args:
        report: The ResearchReport to render.

    Returns:
        A formatted Markdown string.
    """
    date_str = report.date.strftime("%B %d, %Y")
    rating_display = report.rating.value.replace("_", " ").upper()

    md = f"""# {report.title}

**Date:** {date_str} | **Analyst:** {report.analyst} | **Rating:** {rating_display}

---

## Executive Summary

{report.executive_summary}

---

## Company Overview

{report.company_overview}

---

## Financial Analysis

{report.financial_analysis}

---

## Risk Assessment

{report.risk_assessment}

---

## Market Sentiment & News

{report.market_sentiment}

---

## Investment Thesis

{report.investment_thesis}

---

## Strengths

{chr(10).join(f'- {s}' for s in report.strengths) if report.strengths else '- No specific strengths identified'}

## Weaknesses

{chr(10).join(f'- {w}' for w in report.weaknesses) if report.weaknesses else '- No specific weaknesses identified'}

## Catalysts

{chr(10).join(f'- {c}' for c in report.catalysts) if report.catalysts else '- No upcoming catalysts identified'}

---

{report.conclusion}

---

### Additional Sections
"""

    if report.additional_sections:
        for section in report.additional_sections:
            md += f"\n#### {section.title}\n\n{section.content}\n"
    else:
        md += "\n*No additional sections.*\n"

    md += f"\n---\n\n*{report.disclaimer}*\n"

    return md.strip()
