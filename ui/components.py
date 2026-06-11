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

async def show_scoped_sources(session: UserSession):
    """Display the document library for the current session ONLY."""
    if not session.ingested_documents:
        await cl.Message(
            content="📂 No documents ingested yet. Upload a PDF or paste an EDGAR URL."
        ).send()
        return

    for doc in session.ingested_documents:
        badge = classify_doc_badge(doc)
        elements = []

        # PDF preview (first page as image)
        if doc.get("type") == "pdf" and doc.get("local_path"):
            preview_img = next((img for img in doc.get("images", []) if img.get("type") == "page_preview"), None)
            if preview_img and os.path.exists(preview_img.get("path", "")):
                elements.append(cl.Image(
                    name=f"📄 Page 1 Preview - {doc.get('name', 'document.pdf')}",
                    path=preview_img.get("path"),
                    display="inline"
                ))
            else:
                # Fallback to rendering page 1 on the fly
                from ingestion.image_extractor import extract_page_image
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                local_path = doc["local_path"]
                if not os.path.isabs(local_path):
                    local_path = os.path.join(project_root, local_path)
                preview_path = extract_page_image(local_path, 1, session.session_id)
                if preview_path:
                    if "images" not in doc:
                        doc["images"] = []
                    # Avoid duplicates
                    if not any(img.get("type") == "page_preview" for img in doc["images"]):
                        doc["images"].append({
                            "path": preview_path,
                            "page": 1,
                            "type": "page_preview"
                        })
                    elements.append(cl.Image(
                        name=f"📄 Page 1 Preview - {doc.get('name', 'document.pdf')}",
                        path=preview_path,
                        display="inline"
                    ))
        elif doc.get("type") == "url":
            elements.append(cl.Text(
                name=doc.get("name", "document"),
                content=f"URL: {doc.get('source_url', 'N/A')}",
                display="side"
            ))

        # Extracted images (excluding page preview)
        for img in doc.get("images", []):
            if img.get("type") == "page_preview":
                continue
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



async def show_global_sources_tab(user_identifier: str):
    """Display all documents ingested across all sessions for a user."""
    from sessions.session_store import session_store
    all_sessions = session_store.get_all_by_user(user_identifier)
    
    if not all_sessions:
        await cl.Message(
            content="📂 No documents ingested yet across any sessions."
        ).send()
        return

    has_docs = any(s.has_documents for s in all_sessions)
    if not has_docs:
        await cl.Message(
            content="📂 No documents ingested yet across any sessions."
        ).send()
        return
        
    for sess in all_sessions:
        if not sess.has_documents:
            continue
            
        # Session header
        date_str = sess.created_at.strftime("%Y-%m-%d %H:%M")
        company = sess.company_name or "Unknown Company"
        await cl.Message(
            content=f"### 🗂️ Session: {sess.session_id[:8]} ({company}) - {date_str}"
        ).send()
        
        # We can reuse the rendering logic from scoped sources
        await show_scoped_sources(sess)
# ── Welcome message ──────────────────────────────────────────────────────────

async def show_welcome(session: Optional[UserSession] = None):
    """Show the welcome message for a new or returning session."""
    # Common navigation actions available in all welcome states
    nav_actions = [
        cl.Action(name="view_sources",  label="📂 View Sources",  payload={"value": "sources"}),
        cl.Action(name="view_settings", label="⚙️ View Settings", payload={"value": "settings"}),
    ]

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
            ] + nav_actions
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
            actions=nav_actions
        ).send()

