"""
Reusable Chainlit UI components for the Financial Research Assistant.

Contains helpers for:
  - Source rendering (citations + images)
  - Pipeline action buttons
  - Document library / Sources tab
  - Activity feed helpers
"""

import logging
from typing import Optional

import chainlit as cl

from schemas.citation import Citation
from sessions.session_model import UserSession

logger = logging.getLogger(__name__)


# ── Document badge classification ────────────────────────────────────────────

def classify_doc_badge(doc: dict) -> str:
    """Return a display badge for a document type."""
    doc_type = doc.get("type", "unknown")
    source_url = doc.get("source_url", "")

    if "sec.gov" in source_url and "10-K" in doc.get("name", ""):
        return "[EDGAR 10-K]"
    if "sec.gov" in source_url and "10-Q" in doc.get("name", ""):
        return "[EDGAR 10-Q]"
    if "sec.gov" in source_url:
        return "[EDGAR]"
    if doc_type == "pdf":
        return "[PDF]"
    if doc_type == "url":
        return "[Web Article]"
    if doc_type == "image_upload":
        return "[Image Upload]"
    return "[Document]"


# ── Source rendering ─────────────────────────────────────────────────────────

async def send_answer_with_sources(answer: str, citations: list[Citation]):
    """Send a RAG answer followed by expandable source citations."""
    # 1. Stream the main answer
    msg = cl.Message(content=answer)
    await msg.send()

    if not citations:
        return

    # 2. Build source elements
    elements = []
    for i, cit in enumerate(citations):
        # Text source
        elements.append(cl.Text(
            name=f"📄 Source {i+1}: {cit.source_document} p.{cit.page_number}",
            content=(
                f"**Document:** {cit.source_document}\n"
                f"**Page:** {cit.page_number} | **Section:** {cit.section}\n"
                f"**Relevance:** {cit.confidence:.0%}\n\n"
                f"> {cit.chunk_preview}..."
            ),
            display="side"
        ))
        # Image source if available
        if cit.has_image and cit.image_url:
            if cit.image_url.startswith("http"):
                img_el = cl.Image(
                    name=f"🖼️ Visual ({cit.image_type})",
                    url=cit.image_url,
                    display="side"
                )
            else:
                img_el = cl.Image(
                    name=f"🖼️ Visual ({cit.image_type})",
                    path=cit.image_url,
                    display="side"
                )
            elements.append(img_el)

    # 3. Send sources as collapsible panel
    await cl.Message(
        content=f"📎 **{len(citations)} source(s)** — click to expand",
        elements=elements
    ).send()


# ── Pipeline action buttons ──────────────────────────────────────────────────

async def show_pipeline_actions(company_name: str):
    """Show post-pipeline action buttons."""
    actions = [
        cl.Action(name="view_memo",     label="📋 Investment Memo",    payload={"value": "memo"}),
        cl.Action(name="view_report",   label="📑 Full Research Report", payload={"value": "report"}),
        cl.Action(name="export_docx",   label="📥 Export DOCX",        payload={"value": "docx"}),
        cl.Action(name="export_pdf",    label="📥 Export PDF",         payload={"value": "pdf"}),
        cl.Action(name="ask_questions", label="💬 Ask Questions",      payload={"value": "chat"}),
    ]

    await cl.Message(
        content=f"✅ **Analysis complete** for **{company_name}**\nChoose your output:",
        actions=actions
    ).send()


# ── Sources tab / Document library ───────────────────────────────────────────

async def show_sources_tab(session: UserSession):
    """Display the document library for the current session."""
    if not session.ingested_documents:
        await cl.Message(
            content="📂 No documents ingested yet. Upload a PDF or paste an EDGAR URL."
        ).send()
        return

    for doc in session.ingested_documents:
        badge = classify_doc_badge(doc)
        elements = []

        # PDF preview
        if doc.get("type") == "pdf" and doc.get("local_path"):
            elements.append(cl.Pdf(
                name=doc.get("name", "document.pdf"),
                path=doc["local_path"],
                display="inline"
            ))
        elif doc.get("type") == "url":
            elements.append(cl.Text(
                name=doc.get("name", "document"),
                content=f"URL: {doc.get('source_url', 'N/A')}",
                display="side"
            ))

        # Extracted images
        for img in doc.get("images", []):
            elements.append(cl.Image(
                name=f"Page {img.get('page', '?')}",
                path=img.get("path", ""),
                display="inline"
            ))

        await cl.Message(
            content=(
                f"**{doc.get('name', 'Unknown')}** {badge}\n"
                f"Chunks: {doc.get('chunk_count', 0)} | "
                f"Images: {doc.get('image_count', 0)} | "
                f"Ingested: {doc.get('ingested_at', 'N/A')}"
            ),
            elements=elements,
            actions=[
                cl.Action(
                    name="remove_doc",
                    label="🗑 Remove",
                    payload={"doc_id": doc.get("doc_id", "")}
                )
            ]
        ).send()


# ── Welcome message ──────────────────────────────────────────────────────────

async def show_welcome(session: Optional[UserSession] = None):
    """Show the welcome message for a new or returning session."""
    if session and session.has_documents:
        # Returning user with existing data
        doc_list = "\n".join(
            f"  • {d.get('name', '?')}" for d in session.ingested_documents
        )
        await cl.Message(
            content=(
                f"Welcome back! You have a previous analysis for **{session.company_name or 'your documents'}**.\n\n"
                f"**Session:** `{session.session_id[:8]}` | "
                f"**Docs:** {len(session.ingested_documents)} | "
                f"**Chunks:** {session.total_chunks}\n\n"
                f"**📄 Documents:**\n{doc_list}\n\n"
                f"Would you like to continue with it or start fresh?"
            ),
            actions=[
                cl.Action(name="continue_session", label="▶ Continue Session", payload={"value": "continue"}),
                cl.Action(name="start_fresh",      label="🗑 Start Fresh",     payload={"value": "fresh"}),
                cl.Action(name="View Sources",     label="📂 View Sources",    payload={"value": "sources"}),
                cl.Action(name="View Settings",    label="⚙️ View Settings",   payload={"value": "settings"}),
            ]
        ).send()
    else:
        # New session — show session ID context
        session_id_short = session.session_id[:8] if session else "new"
        await cl.Message(
            content=(
                "# 📊 Financial Research Assistant\n\n"
                f"**Session:** `{session_id_short}`\n\n"
                "I can help you with:\n"
                "- **Upload a PDF** — Drag & drop an SEC filing or annual report\n"
                "- **Paste a URL** — `/ingest https://sec.gov/...`\n"
                "- **Run analysis** — `Analyze Apple AAPL` or `Run research on TSLA`\n"
                "- **Ask questions** — `What was their revenue last quarter?`\n\n"
                "Type `/help` for all commands."
            ),
            actions=[
                cl.Action(name="View Sources",  label="📂 View Sources",  payload={"value": "sources"}),
                cl.Action(name="View Settings", label="⚙️ View Settings", payload={"value": "settings"}),
            ]
        ).send()

