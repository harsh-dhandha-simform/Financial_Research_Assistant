"""
Chainlit UI — Financial Research Analyst (Unified Interface).

Single chat window — no tabs. All messages route through the
UnifiedOrchestrator which classifies intent and delegates to:
  - Full research pipeline (LangGraph)
  - Single agent runs
  - RAG chat Q&A
  - Document ingestion
  - Export actions

Run: chainlit run ui/app.py --port 8001
"""

import asyncio
import logging
import os
import sys
import uuid
from typing import Tuple


# Ensure project root is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import chainlit as cl
from chainlit.server import app as chainlit_app
from ui.signup_router import router as signup_router

chainlit_app.include_router(signup_router)
# Move the recently appended signup routes to the very beginning so they intercept before the Chainlit catch-all route `/{full_path:path}`
num_new_routes = len(signup_router.routes)
chainlit_app.router.routes = chainlit_app.router.routes[-num_new_routes:] + chainlit_app.router.routes[:-num_new_routes]

from chat.agent import chat, store_pipeline_context, clear_session
from core.orchestrator import UnifiedOrchestrator, OrchestratorResponse
from core.intent_router import UserIntent
from ingestion.jina_reader import ingest_url
from ingestion.pdf_reader import ingest_pdf
from chunking.chunker import chunk_document
from retrieval.qdrant_store import QdrantStore
from output.markdown_exporter import memo_to_markdown, report_to_markdown
from sessions.session_store import session_store
from sessions.session_model import UserSession
from ui.renderers.memo_renderer import render_memo
from ui.renderers.report_renderer import render_report
from ui.renderers.source_renderer import render_sources
from ui.actions.export_actions import (
    export_memo_docx, export_memo_pdf,
    export_report_docx, export_report_pdf,
)
from ui.components import show_welcome, show_sources_tab, show_pipeline_actions
import ui.auth  # Registers @cl.password_auth_callback

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
)

# Singleton orchestrator
orchestrator = UnifiedOrchestrator()

# ═════════════════════════════════════════════════════════════════════════════
# Data Layer (Sidebar History Persistence)
# ═════════════════════════════════════════════════════════════════════════════

from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from config import settings

@cl.data_layer
def get_data_layer():
    """Mount the SQLAlchemy Data Layer for Supabase persistence."""
    return SQLAlchemyDataLayer(
        conninfo=settings.supabase_uri,
        connect_args={"statement_cache_size": 0}
    )

# ═════════════════════════════════════════════════════════════════════════════
# Navigation & Starters
# ═════════════════════════════════════════════════════════════════════════════



@cl.set_starters
async def starters():
    return [
        cl.Starter(
            label="Upload a PDF",
            message="I'd like to upload a document for analysis.",
            icon="https://api.iconify.design/material-symbols/upload-file-rounded.svg",
        ),
        cl.Starter(
            label="Analyze a Company",
            message="Analyze Apple AAPL",
            icon="https://api.iconify.design/material-symbols/analytics-rounded.svg",
        ),
        cl.Starter(
            label="Ask a Question",
            message="What were the key risk factors mentioned in the filing?",
            icon="https://api.iconify.design/material-symbols/help-outline-rounded.svg",
        ),
        cl.Starter(
            label="Ingest EDGAR Filing",
            message="/ingest https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm",
            icon="https://api.iconify.design/material-symbols/link-rounded.svg",
        ),
    ]


# ═════════════════════════════════════════════════════════════════════════════
# Session info panel helper
# ═════════════════════════════════════════════════════════════════════════════


async def _show_session_info(session: UserSession):
    """Display session info panel (sidebar equivalent)."""
    doc_lines = []
    for doc in session.ingested_documents:
        badge = _classify_badge(doc)
        doc_lines.append(f"  • {doc.get('name', 'Unknown')} {badge}")

    doc_section = "\n".join(doc_lines) if doc_lines else "  _(none)_"

    status_emoji = {
        "idle": "⚪", "running": "🟡", "done": "🟢", "error": "🔴"
    }.get(session.pipeline_status, "⚪")

    await cl.Message(
        content=(
            f"### 🔧 Session Info\n\n"
            f"**Session ID:** `{session.session_id[:8]}`\n"
            f"**Collection:** `{session.collection_name}`\n"
            f"**Status:** {status_emoji} {session.pipeline_status}\n"
            f"**Company:** {session.company_name or '_(not set)_'}\n\n"
            f"---\n"
            f"**📄 Ingested Documents:**\n{doc_section}\n\n"
            f"---\n"
            f"**Memory:** {session.total_chunks} chunks | {session.total_images} images\n"
            f"**Chat history:** {len(session.chat_history)} messages"
        ),
        actions=[
            cl.Action(name="start_fresh", label="🗑 New Session", payload={"value": "fresh"}),
        ]
    ).send()


def _classify_badge(doc: dict) -> str:
    """Return a badge string for a document."""
    source_url = doc.get("source_url", "")
    if "sec.gov" in source_url:
        return "[EDGAR]"
    if doc.get("type") == "pdf":
        return "[PDF]"
    if doc.get("type") == "url":
        return "[Web]"
    return "[Doc]"


# ═════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═════════════════════════════════════════════════════════════════════════════

def _format_error(exc: Exception) -> str:
    """Format exceptions nicely for the UI to avoid dumping raw JSON stack traces."""
    msg = str(exc)
    if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
        return "API Quota Exceeded (429). Please check your API keys or billing limits."
    # If the message is a giant dictionary/json block, truncate it
    if len(msg) > 200:
        return msg[:197] + "..."
    return msg


# def _classify_badge(doc: dict) -> str:


def _chunk_and_store(raw_doc) -> dict:
    """Chunk a RawDocument and store in Qdrant. Returns stats."""
    parents, children = chunk_document(raw_doc)
    parent_lookup = {p.chunk_id: p for p in parents}
    store = QdrantStore()
    stored = store.upsert_chunks(children, parent_lookup)

    # Build BM25 index from the newly ingested chunks so hybrid retrieval works
    from tools.retriever import rebuild_bm25_index
    rebuild_bm25_index()

    return {
        "pages": len(raw_doc.pages),
        "parents": len(parents),
        "children": len(children),
        "stored": stored,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Chainlit lifecycle — on_chat_start
# ═════════════════════════════════════════════════════════════════════════════


@cl.on_chat_start
async def on_start():
    """Initialize or restore session — routes by active profile tab."""
    session_id = cl.user_session.get("id") or f"cl-{uuid.uuid4().hex[:8]}"
    cl.user_session.set("session_id", session_id)

    # Get or create persistent session
    session = session_store.get_or_create(session_id)
    cl.user_session.set("session", session)
    cl.user_session.set("memo_obj", None)
    cl.user_session.set("report_obj", None)

    # Since chat profiles are removed, we default to the Welcome Screen.
    # Users navigate via Action Buttons within the same chat thread.
    await show_welcome(session)

@cl.action_callback("View Sources")
async def on_action_view_sources(action: cl.Action):
    session = cl.user_session.get("session")
    
    # Hide chat input via CSS injection
    css = "<style>#chat-input { display: none !important; }</style>"
    await cl.Message(
        content=f"**📂 Sources Tab**{css}",
        actions=[cl.Action(name="Return to Chat", label="💬 Return to Chat", payload={"value": "return"})]
    ).send()
    
    await show_sources_tab(session)

@cl.action_callback("View Settings")
async def on_action_view_settings(action: cl.Action):
    session = cl.user_session.get("session")
    
    css = "<style>#chat-input { display: none !important; }</style>"
    await cl.Message(
        content=f"**⚙️ Settings Tab**{css}",
        actions=[cl.Action(name="Return to Chat", label="💬 Return to Chat", payload={"value": "return"})]
    ).send()
    
    await _show_session_info(session)

@cl.action_callback("Return to Chat")
async def on_action_return_to_chat(action: cl.Action):
    # This restores the chat input box by overriding the previous CSS
    css = "<style>#chat-input { display: flex !important; }</style>"
    await cl.Message(content=f"**💬 Returned to Chat**{css}").send()


@cl.on_chat_resume
async def on_resume(thread):
    """Re-hydrate session on chat resume (Change 1)."""
    session_id = cl.user_session.get("id") or ""
    session = session_store.get(session_id)
    if session:
        cl.user_session.set("session_id", session_id)
        cl.user_session.set("session", session)
        await cl.Message(
            content=f"▶ Session restored for **{session.company_name or 'your documents'}**."
        ).send()
    else:
        await on_start()


@cl.on_chat_end
async def on_end():
    """Cleanup temp files on session end — keep Qdrant collection."""
    session_id = cl.user_session.get("session_id")
    if session_id:
        session_store.cleanup(session_id)


# ═════════════════════════════════════════════════════════════════════════════
# Main message handler — single entry point, routes through orchestrator
# ═════════════════════════════════════════════════════════════════════════════


@cl.on_message
async def on_message(message: cl.Message):
    """Handle all incoming messages — routes through UnifiedOrchestrator."""
    session_id = cl.user_session.get("session_id", "")
    user_input = message.content.strip()
    session: UserSession = cl.user_session.get("session")

    # ── Global commands ───────────────────────────────────────────────
    if user_input.lower().startswith("/clear"):
        clear_session(session_id)
        cl.user_session.set("memo_obj", None)
        cl.user_session.set("report_obj", None)
        if session:
            session.pipeline_status = "idle"
            session.last_pipeline_result = None
            session.chat_history = []
            session_store.update(session)
        await cl.Message(content="🗑️ Session cleared. Start fresh!").send()
        return

    if user_input.lower().startswith("/help"):
        await _show_help()
        return

    if user_input.lower().startswith("/sources"):
        if session:
            await show_sources_tab(session)
        else:
            await cl.Message(content="📂 No session active.").send()
        return

    if user_input.lower().startswith("/session"):
        if session:
            await _show_session_info(session)
        else:
            await cl.Message(content="No active session.").send()
        return

    # ── Route through orchestrator ────────────────────────────────────
    response: OrchestratorResponse = await orchestrator.process_message(
        user_input=user_input,
        session_id=session_id,
        attachments=message.elements or [],
    )

    # ── Handle orchestrator response ──────────────────────────────────
    action = response.action_required

    if action == "trigger_research_pipeline":
        await _handle_pipeline(response.data, session_id, session)

    elif action == "trigger_single_agent":
        await _handle_pipeline(response.data, session_id, session)

    elif action == "trigger_pdf_ingestion":
        await _handle_pdf_upload_action(response.data, session_id, session)

    elif action == "trigger_url_ingestion":
        await _handle_url_ingestion_action(response.data, session_id, session)

    elif action == "trigger_chat_agent":
        await _handle_chat(user_input, session_id)

    elif action == "show_export_buttons":
        memo_obj = cl.user_session.get("memo_obj")
        report_obj = cl.user_session.get("report_obj")
        if memo_obj or report_obj:
            company = session.company_name if session else "company"
            await show_pipeline_actions(company)
        else:
            await cl.Message(content="⚠️ No analysis available to export. Run a pipeline first.").send()

    elif action == "open_source_panel":
        if session:
            await show_sources_tab(session)
        else:
            await cl.Message(content="📂 No sources available.").send()

    else:
        # Simple text response (chitchat, out_of_domain, etc.)
        if response.content:
            await cl.Message(content=response.content).send()


# ═════════════════════════════════════════════════════════════════════════════
# Pipeline handler — runs LangGraph with live activity feed (Change 13)
# ═════════════════════════════════════════════════════════════════════════════


async def _handle_pipeline(data: dict, session_id: str, session: UserSession):
    """Run the full research pipeline with cl.Step activity feed."""
    company_name = data.get("company_name", "")
    ticker = data.get("ticker", "")
    query = data.get("query", f"Analyse the latest SEC filings for {company_name}")

    if session:
        session.company_name = company_name
        session.ticker = ticker
        session.pipeline_status = "running"
        session_store.update(session)

    await cl.Message(
        content=(
            f"🔬 **Analysing {company_name} ({ticker or 'no ticker'})...**\n\n"
            f"Running agents in parallel: Metrics · Risk · News · Synthesis\n"
            f"This takes 30-60 seconds."
        )
    ).send()

    try:
        from graph.workflow import build_research_graph
        from graph.state import ResearchState
        from agents.base import create_langfuse_config

        graph = build_research_graph()
        initial_state = ResearchState(
            query=query,
            company_name=company_name,
            ticker=ticker,
            session_id=session_id,
        )
        config = create_langfuse_config(
            session_id=session_id,
            trace_name="research-pipeline",
        )

        state_dict = initial_state.model_dump()

        # Stream graph with cl.Step activity feed
        async with cl.Step(name="🔍 Supervisor analyzing query...", type="llm") as supervisor_step:
            pass  # Will be updated by stream

        async for event in graph.astream(state_dict, config=config):
            for node_name, state_update in event.items():
                if node_name == "query_guardrail":
                    rejected = state_update.get("guardrail_rejected", False)
                    if rejected:
                        msg = state_update.get("rejection_message", "Query rejected.")
                        await cl.Message(content=f"⚠️ {msg}").send()
                        if session:
                            session.pipeline_status = "error"
                            session_store.update(session)
                        return

                elif node_name == "supervisor":
                    decision = state_update.get("supervisor_decision")
                    if decision:
                        tasks = getattr(decision, "tasks", [])
                        if not tasks and isinstance(decision, dict):
                            tasks = decision.get("tasks", [])
                        task_names = []
                        for t in tasks:
                            name = t.get("agent_name", "") if isinstance(t, dict) else getattr(t, "agent_name", "")
                            if name:
                                task_names.append(f"`{name}`")
                        await cl.Message(
                            content=f"⚙️ Supervisor assigned: {', '.join(task_names)}. Running agents..."
                        ).send()

                elif node_name == "run_agents":
                    await cl.Message(content="📝 Agents complete. Synthesis combining results...").send()

                elif node_name == "synthesis":
                    await cl.Message(content="📋 Synthesis complete. Formatting output...").send()

                state_dict.update(state_update)

        state = ResearchState(**state_dict)

        # Store results
        if state.memo:
            cl.user_session.set("memo_obj", state.memo)
        if state.report:
            cl.user_session.set("report_obj", state.report)

        # Store pipeline context for chat
        if state.synthesis_output:
            ctx_parts = [f"Company: {company_name} ({ticker})"]
            ctx_parts.append(f"Rating: {state.synthesis_output.rating.value}")
            ctx_parts.append(f"Thesis: {state.synthesis_output.investment_thesis}")
            if state.memo:
                ctx_parts.append(f"\n--- MEMO ---\n{memo_to_markdown(state.memo)}")
            store_pipeline_context(session_id, "\n".join(ctx_parts))

        if session:
            session.pipeline_status = "done"
            session.last_pipeline_result = {
                "company_name": company_name,
                "ticker": ticker,
                "rating": state.synthesis_output.rating.value if state.synthesis_output else None,
            }
            session_store.update(session)

        # Show errors if any
        if state.errors:
            await cl.Message(
                content="⚠️ **Pipeline warnings:**\n" + "\n".join(f"- {e}" for e in state.errors)
            ).send()

        # Show action buttons
        await show_pipeline_actions(company_name)

    except Exception as exc:
        err_msg = _format_error(exc)
        await cl.Message(content=f"❌ **Pipeline failed:** {err_msg}").send()
        logger.error("Pipeline failed: %s", exc)
        if session:
            session.pipeline_status = "error"
            session_store.update(session)


# ═════════════════════════════════════════════════════════════════════════════
# PDF Upload handler
# ═════════════════════════════════════════════════════════════════════════════


async def _handle_pdf_upload_action(data: dict, session_id: str, session: UserSession):
    """Handle PDF file upload — ingest into Qdrant."""
    file_path = data.get("file_path", "")
    filename = data.get("file_name", "document.pdf")
    company_name = data.get("company_name", filename.replace(".pdf", ""))

    msg = cl.Message(content=f"📄 **Processing `{filename}`...**")
    await msg.send()

    try:
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        raw_doc = ingest_pdf(
            file_bytes=file_bytes,
            file_name=filename,
            company_name=company_name,
        )

        await cl.Message(content=f"📄 Extracted **{len(raw_doc.pages)} pages** — chunking & storing...").send()

        stats = _chunk_and_store(raw_doc)

        # Update session
        if session:
            from datetime import datetime
            session.add_document({
                "doc_id": uuid.uuid4().hex[:8],
                "name": filename,
                "type": "pdf",
                "source_url": "",
                "ingested_at": datetime.utcnow().isoformat(),
                "chunk_count": stats["stored"],
                "image_count": 0,
                "local_path": file_path,
            })
            session.company_name = company_name
            session_store.update(session)

        await cl.Message(
            content=(
                f"✅ **Ingested `{filename}`**\n\n"
                f"| Metric | Count |\n|--------|-------|\n"
                f"| Pages extracted | **{stats['pages']}** |\n"
                f"| Parent chunks | **{stats['parents']}** |\n"
                f"| Child chunks | **{stats['children']}** |\n"
                f"| Stored in Qdrant | **{stats['stored']}** |\n\n"
                f"Now type a query like `Analyze {company_name}` to run the pipeline."
            )
        ).send()

    except Exception as exc:
        err_msg = _format_error(exc)
        await cl.Message(content=f"❌ **Ingestion failed:** {err_msg}").send()
        logger.error("PDF upload ingestion failed: %s", exc)


# ═════════════════════════════════════════════════════════════════════════════
# URL Ingestion handler
# ═════════════════════════════════════════════════════════════════════════════


async def _handle_url_ingestion_action(data: dict, session_id: str, session: UserSession):
    """Handle URL ingestion after domain validation."""
    url = data.get("url", "")

    msg = cl.Message(content=f"🔗 **Ingesting from URL...**\n`{url[:80]}`")
    await msg.send()

    try:
        company_name = session.company_name if session else "Unknown"
        raw_doc = ingest_url(url=url, company_name=company_name)

        await cl.Message(content=f"📄 Extracted **{len(raw_doc.pages)} pages** — chunking & storing...").send()

        stats = _chunk_and_store(raw_doc)

        # Update session
        if session:
            from datetime import datetime
            session.add_document({
                "doc_id": uuid.uuid4().hex[:8],
                "name": url.split("/")[-1][:50] or "web_document",
                "type": "url",
                "source_url": url,
                "ingested_at": datetime.utcnow().isoformat(),
                "chunk_count": stats["stored"],
                "image_count": 0,
            })
            session_store.update(session)

        await cl.Message(
            content=(
                f"✅ **URL ingested**\n\n"
                f"| Metric | Count |\n|--------|-------|\n"
                f"| Pages extracted | **{stats['pages']}** |\n"
                f"| Parent chunks | **{stats['parents']}** |\n"
                f"| Child chunks | **{stats['children']}** |\n"
                f"| Stored in Qdrant | **{stats['stored']}** |\n\n"
                f"Now type a query to run analysis."
            )
        ).send()

    except Exception as exc:
        err_msg = _format_error(exc)
        await cl.Message(content=f"❌ **URL ingestion failed:** {err_msg}").send()
        logger.error("URL ingestion failed: %s", exc)


# ═════════════════════════════════════════════════════════════════════════════
# Chat Q&A handler
# ═════════════════════════════════════════════════════════════════════════════


async def _handle_chat(question: str, session_id: str):
    """Handle a chat question with RAG retrieval + source footer (Change 12)."""
    msg = cl.Message(content="")
    await msg.send()

    try:
        response = chat(
            question=question,
            session_id=session_id,
        )

        answer = response.answer

        if response.rewritten_query:
            answer += f"\n\n*🔄 Query rewritten: \"{response.rewritten_query}\"*"

        msg.content = answer
        await msg.update()

        # Render source citations if available (Change 12)
        if response.sources and response.relevant:
            from ui.renderers.source_renderer import render_sources
            await render_sources(response.sources)

        # Per-message footer bar (Change 12)
        await cl.Message(
            content="",
            actions=[
                cl.Action(name="thumbs_up",   label="👍", payload={"value": "up", "question": question[:100]}),
                cl.Action(name="thumbs_down", label="👎", payload={"value": "down", "question": question[:100]}),
                cl.Action(name="copy_msg",    label="📋 Copy", payload={"value": answer[:200]}),
                cl.Action(name="regenerate",  label="🔄 Regenerate", payload={"value": "regen", "question": question}),
            ]
        ).send()

        # Update session chat history
        session: UserSession = cl.user_session.get("session")
        if session:
            session.chat_history.append({"role": "user", "content": question})
            session.chat_history.append({"role": "assistant", "content": answer[:500]})
            session.trim_chat_history()
            session_store.update(session)

    except Exception as exc:
        err_msg = _format_error(exc)
        msg.content = f"❌ Error: {err_msg}"
        await msg.update()
        logger.error("Chat failed: %s", exc)


# ═════════════════════════════════════════════════════════════════════════════
# Action callbacks (Change 10)
# ═════════════════════════════════════════════════════════════════════════════


@cl.action_callback("view_memo")
async def on_view_memo(action):
    memo_obj = cl.user_session.get("memo_obj")
    if memo_obj:
        await render_memo(memo_obj)
    else:
        await cl.Message(content="⚠️ No memo available.").send()


@cl.action_callback("view_report")
async def on_view_report(action):
    report_obj = cl.user_session.get("report_obj")
    if report_obj:
        await render_report(report_obj)
    else:
        await cl.Message(content="⚠️ No report available.").send()


@cl.action_callback("export_docx")
async def on_export_docx(action):
    memo_obj = cl.user_session.get("memo_obj")
    report_obj = cl.user_session.get("report_obj")
    session: UserSession = cl.user_session.get("session")
    company = session.company_name if session else "company"

    if memo_obj:
        await export_memo_docx(memo_obj, company)
    if report_obj:
        await export_report_docx(report_obj, company)
    if not memo_obj and not report_obj:
        await cl.Message(content="⚠️ Nothing to export.").send()


@cl.action_callback("export_pdf")
async def on_export_pdf(action):
    memo_obj = cl.user_session.get("memo_obj")
    report_obj = cl.user_session.get("report_obj")
    session: UserSession = cl.user_session.get("session")
    company = session.company_name if session else "company"

    if memo_obj:
        await export_memo_pdf(memo_obj, company)
    if report_obj:
        await export_report_pdf(report_obj, company)
    if not memo_obj and not report_obj:
        await cl.Message(content="⚠️ Nothing to export.").send()


@cl.action_callback("ask_questions")
async def on_ask_questions(action):
    await cl.Message(
        content="💬 Ask me anything about the analysis. I have full context from the pipeline."
    ).send()


@cl.action_callback("continue_session")
async def on_continue_session(action):
    session: UserSession = cl.user_session.get("session")
    if session:
        await cl.Message(
            content=f"▶ Resuming session for **{session.company_name or 'your documents'}**. What would you like to do?"
        ).send()


@cl.action_callback("start_fresh")
async def on_start_fresh(action):
    session_id = cl.user_session.get("session_id")
    if session_id:
        clear_session(session_id)
        session_store.delete(session_id)

    # Create fresh session
    new_session_id = f"cl-{uuid.uuid4().hex[:8]}"
    cl.user_session.set("session_id", new_session_id)
    session = session_store.get_or_create(new_session_id)
    cl.user_session.set("session", session)
    cl.user_session.set("memo_obj", None)
    cl.user_session.set("report_obj", None)

    await show_welcome()


@cl.action_callback("remove_doc")
async def on_remove_doc(action):
    doc_id = action.payload.get("doc_id", "") if action.payload else ""
    session: UserSession = cl.user_session.get("session")
    if session and doc_id:
        removed = session.remove_document(doc_id)
        if removed:
            session_store.update(session)
            await cl.Message(content=f"🗑️ Document removed.").send()
        else:
            await cl.Message(content="⚠️ Document not found.").send()


# ═════════════════════════════════════════════════════════════════════════════
# Feedback action callbacks (Change 12)
# ═════════════════════════════════════════════════════════════════════════════


@cl.action_callback("thumbs_up")
async def on_thumbs_up(action):
    logger.info("Positive feedback for: %s", action.payload.get("question", "")[:50])
    await cl.Message(content="👍 Thanks for the feedback!").send()


@cl.action_callback("thumbs_down")
async def on_thumbs_down(action):
    logger.info("Negative feedback for: %s", action.payload.get("question", "")[:50])
    await cl.Message(content="👎 Sorry about that. I'll try to improve.").send()


@cl.action_callback("copy_msg")
async def on_copy_msg(action):
    await cl.Message(content="📋 Answer copied to clipboard.").send()


@cl.action_callback("regenerate")
async def on_regenerate(action):
    question = action.payload.get("question", "") if action.payload else ""
    if question:
        session_id = cl.user_session.get("session_id", "")
        await _handle_chat(question, session_id)
    else:
        await cl.Message(content="⚠️ Cannot regenerate — no question found.").send()


# ═════════════════════════════════════════════════════════════════════════════
# Help command
# ═════════════════════════════════════════════════════════════════════════════


async def _show_help():
    """Show unified help."""
    await cl.Message(
        content=(
            "## 📊 Financial Research Assistant — Help\n\n"
            "| Action | How |\n"
            "|--------|-----|\n"
            "| Upload a PDF | Drag & drop a file |\n"
            "| Ingest a URL | `/ingest <EDGAR URL>` |\n"
            "| Run full analysis | `Analyze Apple AAPL` |\n"
            "| Ask questions | Just type your question |\n"
            "| View sources | `/sources` |\n"
            "| Session info | `/session` |\n"
            "| Clear session | `/clear` |\n\n"
            "**Navigation:**\n"
            "- 💬 **Chat** — Main interface (upload, analyze, ask)\n"
            "- 📁 **Sources** — View ingested documents\n"
            "- ⚙️ **Settings** — Session info & management\n\n"
            "**Tips:**\n"
            "- I auto-detect your intent — no need to switch tabs\n"
            "- After analysis, use action buttons to view/export results\n"
            "- I use RAG to search your ingested documents"
        )
    ).send()
