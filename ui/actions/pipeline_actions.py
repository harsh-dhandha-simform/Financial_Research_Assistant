"""
Pipeline actions — post-pipeline action button handlers.

Shows styled action buttons after every completed pipeline run
and handles user interaction with them.
"""

import logging

import chainlit as cl

from ui.renderers.memo_renderer import render_memo
from ui.renderers.report_renderer import render_report

logger = logging.getLogger(__name__)


async def show_pipeline_actions(company_name: str):
    """Show post-pipeline action buttons (Change 10)."""
    actions = [
        cl.Action(name="view_memo",     label="📋 Investment Memo",     payload={"value": "memo"}),
        cl.Action(name="view_report",   label="📑 Full Research Report", payload={"value": "report"}),
        cl.Action(name="export_docx",   label="📥 Export DOCX",         payload={"value": "docx"}),
        cl.Action(name="export_pdf",    label="📥 Export PDF",          payload={"value": "pdf"}),
        cl.Action(name="ask_questions", label="💬 Ask Questions",       payload={"value": "chat"}),
    ]

    await cl.Message(
        content=f"✅ **Analysis complete** for **{company_name}**\nChoose your output:",
        actions=actions
    ).send()
