"""
Report renderer — formats and sends a ResearchReport to Chainlit.

Uses the existing markdown_exporter to convert the report, then sends
it as a Chainlit message.
"""

import chainlit as cl

from output.markdown_exporter import report_to_markdown
from schemas.reports import ResearchReport


async def render_report(report: ResearchReport):
    """Render a ResearchReport as a formatted Chainlit message."""
    report_md = report_to_markdown(report)
    rating = report.rating.value.replace("_", " ").upper()

    await cl.Message(
        content=(
            f"## 📑 Full Research Report\n\n"
            f"**Rating:** {rating} | **Analyst:** {report.analyst}\n\n"
            f"---\n\n"
            f"{report_md}"
        )
    ).send()
