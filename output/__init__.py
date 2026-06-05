"""
Output module — generates InvestmentMemo + ResearchReport from pipeline outputs.

Public API:
    from output import generate_output_fork, memo_to_markdown, report_to_markdown
"""

from output.report_generator import (
    generate_output_fork,
    generate_memo,
    generate_report,
)
from output.markdown_exporter import memo_to_markdown, report_to_markdown

__all__ = [
    "generate_output_fork",
    "generate_memo",
    "generate_report",
    "memo_to_markdown",
    "report_to_markdown",
]
