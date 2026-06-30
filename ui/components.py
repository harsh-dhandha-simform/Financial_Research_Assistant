"""
Reusable Chainlit UI components for the Financial Research Assistant.

Contains helpers for:
  - Source rendering (citations + images)
  - Pipeline action buttons
  - Document library / Sources tab
  - Activity feed helpers
"""

import logging
import os
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
    message_content = f"📎 **{len(citations)} source(s)**\n"
    for i, cit in enumerate(citations):
        name = f"📄 Source {i+1}: {cit.source_document} p.{cit.page_number}"
        message_content += f"- [[{name}]]\n"

        content = (
            f"**Document:** {cit.source_document}\n"
            f"**Page:** {cit.page_number} | **Section:** {cit.section}\n"
            f"**Relevance:** {cit.confidence:.0%}\n\n"
            f"> {cit.chunk_preview}...\n"
        )
        
        # Add image element and reference it in details if available
        if cit.has_image and cit.image_url:
            content += (
                f"\n<details>\n"
                f"<summary>view source images</summary>\n\n"
                f"[[Source Image {i+1}]]\n"
                f"</details>\n"
            )
            if cit.image_url.startswith("http"):
                img_el = cl.Image(
                    name=f"Source Image {i+1}",
                    url=cit.image_url,
                    display="inline"
                )
            else:
                img_el = cl.Image(
                    name=f"Source Image {i+1}",
                    path=cit.image_url,
                    display="inline"
                )
            elements.append(img_el)

        elements.append(cl.Text(
            name=name,
            content=content,
            display="side"
        ))

    # 3. Send sources as collapsible panel
    await cl.Message(
        content=message_content,
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

def _render_doc_elements(doc: dict) -> list:
    """Build Chainlit elements (PDF viewer / URL text) for a document dict."""
    elements = []
    badge = classify_doc_badge(doc)

    if doc.get("type") == "pdf" and doc.get("local_path"):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        local_path = doc["local_path"]
        if not os.path.isabs(local_path):
            local_path = os.path.join(project_root, local_path)
        if os.path.exists(local_path):
            elements.append(cl.Pdf(
                name=doc.get("name", "document.pdf"),
                path=local_path,
                display="inline"
            ))
    elif doc.get("type") == "url":
        elements.append(cl.Text(
            name=doc.get("name", "document"),
            content=f"URL: {doc.get('source_url', 'N/A')}",
            display="side"
        ))
    return elements


async def show_sources_for_user(user_id: str):
    """Display ALL documents for a user (user-scoped, across all sessions).

    Queries the documents table by user_id — this is the canonical source of truth.
    Used by both the Sources tab button and the /sources command.
    """
    from sessions.session_store import session_store
    logger.debug("[SOURCES-READ] user_id=%s", user_id)

    docs = session_store.get_docs_for_user(user_id)

    if not docs:
        await cl.Message(
            content="📂 No documents ingested yet. Upload a PDF or paste an EDGAR URL."
        ).send()
        return

    await cl.Message(
        content=f"### 📁 Your Document Library ({len(docs)} document{'s' if len(docs) != 1 else ''})"
    ).send()

    for doc in docs:
        elements = _render_doc_elements(doc)
        badge = classify_doc_badge(doc)
        ingested_at = doc.get("ingested_at", "")[:19].replace("T", " ") if doc.get("ingested_at") else "N/A"

        await cl.Message(
            content=(
                f"**{doc.get('name', 'Unknown')}** {badge}\n"
                f"Chunks: {doc.get('chunk_count', 0)} | "
                f"Images: {doc.get('image_count', 0)} | "
                f"Ingested: {ingested_at}"
            ),
            elements=elements,
            actions=[
                cl.Action(
                    name="remove_doc",
                    label="🗑 Remove",
                    payload={"doc_id": doc.get("doc_id", ""), "user_id": user_id}
                )
            ]
        ).send()


# Backward-compat aliases — both now delegate to show_sources_for_user
async def show_scoped_sources(session: "UserSession"):
    """Show sources for the user associated with this session."""
    user_id = getattr(session, "user_identifier", "") or "anonymous"
    await show_sources_for_user(user_id)


async def show_global_sources_tab(user_identifier: str):
    """Show all sources for a user (tab click or /sources command)."""
    await show_sources_for_user(user_identifier)


# ── Welcome message ──────────────────────────────────────────────────────────

async def show_welcome(session: Optional[UserSession] = None, is_fresh: bool = False):
    """Show the welcome message for a new or returning session.

    Args:
        session: The current UserSession.
        is_fresh: If True, always show the new-session UI regardless of
                  whether the user has existing documents. Used when the
                  user clicks "Start Fresh" or opens a brand-new chat tab.
    """
    nav_actions = [
        cl.Action(name="view_sources",  label="📂 View Sources",  payload={"value": "sources"}),
        cl.Action(name="view_settings", label="⚙️ View Settings", payload={"value": "settings"}),
    ]

    user_id = getattr(session, "user_identifier", "") if session else ""

    # Only show "Welcome back" when resuming an existing thread (is_fresh=False)
    if not is_fresh and user_id:
        from sessions.session_store import session_store
        docs = session_store.get_docs_for_user(user_id)
    else:
        docs = []  # Fresh session — pretend no docs for the welcome screen

    if docs:
        # Returning user resuming a thread that already had documents
        doc_list = "\n".join(f"  • {d.get('name', '?')}" for d in docs[:5])
        if len(docs) > 5:
            doc_list += f"\n  _...and {len(docs) - 5} more_"
        total_chunks = sum(d.get("chunk_count", 0) for d in docs)
        await cl.Message(
            content=(
                f"Welcome back! You have **{len(docs)} document{'s' if len(docs) != 1 else ''}** ready for analysis.\n\n"
                f"**Session:** `{session.session_id[:8]}` | "
                f"**Docs:** {len(docs)} | "
                f"**Chunks:** {total_chunks}\n\n"
                f"**📄 Documents:**\n{doc_list}\n\n"
                f"Would you like to continue or start fresh?"
            ),
            actions=[
                cl.Action(name="continue_session", label="▶ Continue Session", payload={"value": "continue"}),
                cl.Action(name="start_fresh",      label="🗑 Start Fresh",     payload={"value": "fresh"}),
            ] + nav_actions
        ).send()
    else:
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
            actions=nav_actions
        ).send()
