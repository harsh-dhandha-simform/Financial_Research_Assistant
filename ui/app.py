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
from ui.components import show_welcome, show_scoped_sources, show_global_sources_tab, show_pipeline_actions
from graph.workflow import build_research_graph
from graph.state import ResearchState
from agents.base import create_langfuse_config
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
    user_id = session.user_identifier or "anonymous"
    docs = session_store.get_docs_for_user(user_id)

    doc_lines = []
    for doc in docs:
        badge = _classify_badge(doc)
        doc_lines.append(f"  • {doc.get('name', 'Unknown')} {badge}")

    doc_section = "\n".join(doc_lines) if doc_lines else "  _(none)_"
    total_chunks = sum(d.get("chunk_count", 0) for d in docs)

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
            f"**📄 Ingested Documents ({len(docs)} across all sessions):**\n{doc_section}\n\n"
            f"---\n"
            f"**Memory:** {total_chunks} chunks\n"
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


def _chunk_and_store(raw_doc, collection_name: str, doc_id: str = "", user_id: str = "") -> dict:
    """Chunk a RawDocument and store in Qdrant. Returns stats."""
    if not collection_name:
        raise ValueError("Cannot store document without a valid collection_name")

    parents, children = chunk_document(raw_doc)
    parent_lookup = {p.chunk_id: p for p in parents}
    store = QdrantStore(collection_name)
    stored = store.upsert_chunks(children, parent_lookup)

    # Update collection chunk count in user_collections table
    if user_id:
        session_store.update_collection_chunk_count(user_id, stored)

    # Build BM25 index from the newly ingested chunks so hybrid retrieval works
    from tools.retriever import rebuild_bm25_index
    rebuild_bm25_index(collection_name)

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
    """Initialize or restore session using the Chainlit thread_id as the canonical session_id."""
    # Lazy collection cleanup (24h rule) — runs quickly, no-op if nothing to clean
    try:
        session_store.check_and_cleanup_empty_collections()
    except Exception:
        pass

    # Use the actual Chainlit thread_id, NOT a generated UUID
    thread_id_raw = cl.context.session.thread_id or cl.user_session.get("id")
    if isinstance(thread_id_raw, dict):
        thread_id = thread_id_raw.get("id") or thread_id_raw.get("session_id")
    else:
        thread_id = thread_id_raw
        
    # Extra safety check to ensure thread_id is a string and not excessively long
    if not isinstance(thread_id, str) or len(thread_id) > 50:
        logger.warning("Invalid thread_id extracted (%s), falling back to generated UUID", type(thread_id))
        thread_id = f"cl-{uuid.uuid4().hex[:8]}"
        
    if not thread_id:
        thread_id = f"cl-{uuid.uuid4().hex[:8]}"
        
    cl.user_session.set("session_id", thread_id)

    user = cl.user_session.get("user")
    user_id = user.identifier if user else "anonymous"

    # Check if this thread already has a session (e.g. page refresh on same thread)
    session = session_store.get_by_thread(thread_id, user_id)
    if session:
        # Existing thread — re-hydrate, do NOT create new collection
        cl.user_session.set("session", session)
        cl.user_session.set("memo_obj", None)
        cl.user_session.set("report_obj", None)
        logger.info("Loaded existing session %s for user %s", thread_id[:8], user_id)
    else:
        # Genuinely new thread
        session = session_store.create_new(thread_id, user_id)
        cl.user_session.set("session", session)
        cl.user_session.set("memo_obj", None)
        cl.user_session.set("report_obj", None)

    await show_welcome(session)


# ── Action callbacks for Sources / Settings navigation ───────────────────

@cl.action_callback("view_sources")
async def on_view_sources(action):
    """Show the Sources panel inline in the chat thread."""
    session_id = cl.user_session.get("session_id")
    session: UserSession = session_store.get(session_id) if session_id else None
    
    if session:
        user_id = session.user_identifier or "anonymous"
        await show_global_sources_tab(user_id)
    else:
        await show_global_sources_tab("anonymous")

@cl.action_callback("view_settings")
async def on_view_settings(action: cl.Action):
    """Show the Settings/Session info panel inline in the chat thread."""
    session_id = cl.user_session.get("session_id")
    session = session_store.get(session_id) if session_id else None
    if session:
        cl.user_session.set("session", session)
        await _show_session_info(session)
    else:
        await cl.Message(content="⚙️ No session active.").send()


@cl.on_chat_resume
async def on_resume(thread):
    """Re-hydrate session when user navigates back to an existing thread.

    NEVER creates a new session or a new collection here.
    """
    thread_id = thread.id if hasattr(thread, "id") else str(thread)
    cl.user_session.set("session_id", thread_id)

    user = cl.user_session.get("user")
    user_id = user.identifier if user else "anonymous"

    session = session_store.get_by_thread(thread_id, user_id)
    if session:
        cl.user_session.set("session", session)
        cl.user_session.set("memo_obj", None)
        cl.user_session.set("report_obj", None)
        logger.info("Resumed session %s for user %s", thread_id[:8], user_id)
    else:
        # Thread exists in Chainlit but not in our session DB — create it
        session = session_store.create_new(thread_id, user_id)
        cl.user_session.set("session", session)
        cl.user_session.set("memo_obj", None)
        cl.user_session.set("report_obj", None)
        logger.info("Re-created session record for thread %s user %s", thread_id[:8], user_id)


@cl.on_chat_end
async def on_end():
    """Cleanup on session end - persistent documents are kept."""
    pass


# ═════════════════════════════════════════════════════════════════════════════
# Main message handler — single entry point, routes through orchestrator
# ═════════════════════════════════════════════════════════════════════════════


@cl.on_message
async def on_message(message: cl.Message):
    """Main message handler for slash commands, files, and chat Q&A."""
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
        session = cl.user_session.get("session")
        user_id = session.user_identifier if session else "anonymous"
        await show_global_sources_tab(user_id)
        return

    if user_input.lower().startswith("/delete"):
        await _handle_delete_command(session_id)
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
            user_id = session.user_identifier or "anonymous"
            await show_global_sources_tab(user_id)
        else:
            await show_global_sources_tab("anonymous")

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
    query = data.get("query", "")

    if session:
        if company_name == query:
            company_name = session.company_name or "your documents"
            
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
        async with cl.Step(name="🔍 Supervisor analyzing query...", type="run") as pipeline_step:
            async for event in graph.astream(state_dict, config=config):
                for node_name, state_update in event.items():
                    if node_name == "document_gate":
                        if state_update.get("gate_rejected"):
                            msg = state_update.get("rejection_message", "Query rejected.")
                            await cl.Message(content=f"⚠️ {msg}").send()
                            if session:
                                session.pipeline_status = "error"
                                session_store.update(session)
                            pipeline_step.status = "failed"
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
                            pipeline_step.name = "⚙️ Agents running in parallel..."
                            await pipeline_step.update()

                    elif node_name == "run_agents":
                        await cl.Message(content="📝 Agents complete. Synthesis combining results...").send()
                        pipeline_step.name = "📝 Synthesizing results..."
                        await pipeline_step.update()

                    elif node_name == "synthesis":
                        await cl.Message(content="📋 Synthesis complete. Formatting output...").send()
                        pipeline_step.name = "✅ Pipeline complete!"
                        pipeline_step.status = "success"
                        await pipeline_step.update()

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
        import traceback
        err_msg = _format_error(exc)
        await cl.Message(content=f"❌ **Pipeline failed:** {err_msg}").send()
        logger.error("Pipeline failed: %s\n%s", exc, traceback.format_exc())
        if session:
            session.pipeline_status = "error"
            session_store.update(session)


# ═════════════════════════════════════════════════════════════════════════════
# PDF Upload handler
# ═════════════════════════════════════════════════════════════════════════════

async def classify_and_extract_document(text: str) -> dict:
    """Use a fast LLM to classify domain and extract company name/ticker."""
    if not text or len(text.strip()) < 50:
        return {"is_financial_domain": False, "company_name": "", "ticker": "", "document_type": "Unknown"}
        
    try:
        from agents.base import invoke_with_fallback
        from langchain_core.prompts import ChatPromptTemplate
        from pydantic import BaseModel, Field
        
        class DocumentClassification(BaseModel):
            is_financial_domain: bool = Field(description="True if the document pertains to financial or business analysis, SEC filings, corporate reports, news, or earnings.")
            document_type: str = Field(description="Type of document (e.g., '10-K', 'Earnings Transcript', 'News Article', 'Recipe', 'Unknown')")
            company_name: str = Field(description="Primary company name if found, else empty string")
            ticker: str = Field(description="Stock ticker if found, else empty string")
            
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Analyze the provided text. Determine if it belongs to the financial/business domain (e.g. SEC filings, annual reports, earnings transcripts, business news). If it is a recipe, fiction, or unrelated, set is_financial_domain to false. Also extract the document type, company name, and stock ticker if present. Respond strictly with the JSON schema."),
            ("human", "Text:\n{text}")
        ])
        
        # Run synchronously via asyncio.to_thread or just block briefly
        import asyncio
        loop = asyncio.get_event_loop()
        classification = await loop.run_in_executor(None, lambda: invoke_with_fallback(
            agent_name="metrics", # Use a fast configured model
            prompt_chain=prompt,
            input_data={"text": text[:3000]},
            output_schema=DocumentClassification,
        ))
        
        return {
            "is_financial_domain": classification.is_financial_domain,
            "document_type": classification.document_type,
            "company_name": classification.company_name.strip(),
            "ticker": classification.ticker.strip()
        }
    except Exception as exc:
        logger.warning("Failed to classify document: %s", exc)
        # Fail-open: if the LLM call fails, assume it's valid so we don't block the user
        return {"is_financial_domain": True, "company_name": "", "ticker": "", "document_type": "Unknown"}



async def _handle_pdf_upload_action(data: dict, session_id: str, session: UserSession):
    """Handle PDF file upload — ingest into Qdrant."""
    file_path = data.get("file_path", "")
    filename = data.get("file_name", "document.pdf")
    company_name = data.get("company_name", filename.replace(".pdf", ""))

    msg = cl.Message(content=f"📄 **Processing `{filename}`...**")
    await msg.send()

    if not session:
        await cl.Message(content="❌ **Error:** No active session found to ingest documents.").send()
        return

    user_id = session.user_identifier or "anonymous"
    collection_name = session_store.get_user_collection(user_id)

    try:
        import shutil
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        session_docs_dir = os.path.join(project_root, "data", "documents", session_id[:8])
        os.makedirs(session_docs_dir, exist_ok=True)
        permanent_path = os.path.join(session_docs_dir, filename)
        shutil.copy2(file_path, permanent_path)
        file_path = permanent_path

        from ingestion.image_extractor import extract_page_image, extract_images_from_pdf
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        raw_doc = ingest_pdf(
            file_bytes=file_bytes,
            file_name=filename,
            company_name=company_name,
        )

        await cl.Message(content=f"📄 Extracted **{len(raw_doc.pages)} pages**...").send()
        
        # Classify document domain and try to auto-identify company name and ticker
        if len(raw_doc.pages) > 0:
            preview_text = "\n".join(p.content for p in raw_doc.pages[:3] if p.content)
            classification = await classify_and_extract_document(preview_text)
            
            if not classification.get("is_financial_domain", True):
                await cl.Message(content=f"❌ **Document Rejected:** The uploaded document does not appear to be related to the financial or business domain. Please provide a valid SEC filing, annual report, or business analysis document. (Detected type: {classification.get('document_type', 'Unknown')})").send()
                try:
                    os.remove(file_path)
                except Exception:
                    pass
                return
                
            extracted_company = classification.get("company_name")
            extracted_ticker = classification.get("ticker")
            if extracted_company:
                company_name = extracted_company
            if extracted_ticker and session:
                session.ticker = extracted_ticker
                
        await cl.Message(content=f"🧠 Identified Company: **{company_name}** {f'({session.ticker})' if session and session.ticker else ''}. Chunking & storing...").send()

        logger.info("[Ingest PDF] Storing in collection %s for user %s", collection_name, user_id)
        stats = _chunk_and_store(raw_doc, collection_name, user_id=user_id)
        
        # Extract all embedded images from the PDF
        extracted_images = extract_images_from_pdf(file_path, session_id)
        images = extracted_images if extracted_images else []
        
        # Extract preview of page 1
        preview_path = extract_page_image(file_path, 1, session_id)
        if preview_path:
            images.append({
                "path": preview_path,
                "page": 1,
                "type": "page_preview"
            })

        # Calculate relative path for portability
        try:
            rel_path = os.path.relpath(file_path, project_root)
        except ValueError:
            rel_path = file_path

        # Store document metadata in the user-scoped documents table
        doc_id = uuid.uuid4().hex[:8]
        from datetime import datetime
        doc_meta = {
            "doc_id": doc_id,
            "name": filename,
            "type": "pdf",
            "source_url": "",
            "local_path": rel_path,
            "chunk_count": stats["stored"],
            "image_count": len(images),
            "images": images,
            "ingested_at": datetime.utcnow().isoformat(),
        }
        session_store.add_document(doc_meta, user_id=user_id, session_id=session.session_id)
        if session:
            session.company_name = company_name
            session_store.update(session)

        actions = [
            cl.Action(name="ask_questions", label=f"💬 Ask about {filename}", payload={"value": "chat"}),
            cl.Action(name="run_analysis", label=f"🔍 Run Full Analysis", payload={"value": company_name}),
        ]

        await cl.Message(
            content=(
                f"✅ **Ingested `{filename}`**\n\n"
                f"| Metric | Count |\n|--------|-------|\n"
                f"| Pages extracted | **{stats['pages']}** |\n"
                f"| Parent chunks | **{stats['parents']}** |\n"
                f"| Child chunks | **{stats['children']}** |\n"
                f"| Stored in Qdrant | **{stats['stored']}** |\n\n"
                f"What would you like to do next?"
            ),
            actions=actions
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

    if not session:
        await cl.Message(content="❌ **Error:** No active session found to ingest documents.").send()
        return

    user_id = session.user_identifier or "anonymous"
    collection_name = session_store.get_user_collection(user_id)

    try:
        company_name = session.company_name if session else "Unknown"
        raw_doc = ingest_url(url=url, company_name=company_name)

        await cl.Message(content=f"📄 Extracted **{len(raw_doc.pages)} pages**...").send()
        
        # Classify document domain and try to auto-identify company name and ticker
        if len(raw_doc.pages) > 0:
            preview_text = "\n".join(p.text for p in raw_doc.pages[:3] if p.text)
            classification = await classify_and_extract_document(preview_text)
            
            if not classification.get("is_financial_domain", True):
                await cl.Message(content=f"❌ **URL Rejected:** The content at this URL does not appear to be related to the financial or business domain. Please provide a valid SEC filing, annual report, or business analysis URL. (Detected type: {classification.get('document_type', 'Unknown')})").send()
                return
                
            extracted_company = classification.get("company_name")
            extracted_ticker = classification.get("ticker")
            if extracted_company:
                company_name = extracted_company
            if extracted_ticker and session:
                session.ticker = extracted_ticker
                
        await cl.Message(content=f"🧠 Identified Company: **{company_name}** {f'({session.ticker})' if session and session.ticker else ''}. Chunking & storing...").send()

        logger.info("[Ingest URL] Storing in collection %s for user %s", collection_name, user_id)
        stats = _chunk_and_store(raw_doc, collection_name, user_id=user_id)

        # Store document metadata in the user-scoped documents table
        from datetime import datetime
        doc_meta = {
            "doc_id": uuid.uuid4().hex[:8],
            "name": url.split("/")[-1][:50] or "web_document",
            "type": "url",
            "source_url": url,
            "local_path": "",
            "chunk_count": stats["stored"],
            "image_count": 0,
            "images": [],
            "ingested_at": datetime.utcnow().isoformat(),
        }
        session_store.add_document(doc_meta, user_id=user_id, session_id=session.session_id)
        if session:
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
# /delete command handler
# ═════════════════════════════════════════════════════════════════════════════


async def _handle_delete_command(session_id: str):
    """Handle the /delete slash command — list docs with delete action buttons."""
    session: UserSession = cl.user_session.get("session")
    user_id = session.user_identifier if session else "anonymous"

    docs = session_store.get_docs_for_user(user_id)
    if not docs:
        await cl.Message(content="📂 You have no ingested documents to delete.").send()
        return

    from datetime import datetime as _dt
    actions = []
    lines = [f"### 🗑️ Select a document to delete ({len(docs)} total)\n"]
    for i, doc in enumerate(docs, 1):
        ingested_at = doc.get("ingested_at", "")[:10] if doc.get("ingested_at") else "?"
        badge = _classify_badge(doc)
        lines.append(
            f"**{i}.** {doc.get('name', 'Unknown')} {badge} "
            f"({doc.get('chunk_count', 0)} chunks) — ingested {ingested_at}"
        )
        actions.append(cl.Action(
            name="remove_doc",
            label=f"🗑 Delete [{i}]",
            payload={"doc_id": doc["doc_id"], "user_id": user_id}
        ))

    await cl.Message(
        content="\n".join(lines),
        actions=actions
    ).send()


# ═════════════════════════════════════════════════════════════════════════════
# Chat Q&A handler
# ═════════════════════════════════════════════════════════════════════════════


async def _handle_chat(question: str, session_id: str):
    """Handle a chat question with RAG retrieval + source footer (Change 12)."""
    msg = cl.Message(content="")
    await msg.send()

    try:
        _session: UserSession = cl.user_session.get("session")
        _user_id = _session.user_identifier if _session else ""
        response = chat(
            question=question,
            session_id=session_id,
            user_id=_user_id,
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
    """Handle Ask Questions button click."""
    await cl.Message(content="Start typing your questions in the chat!").send()


@cl.action_callback("run_analysis")
async def on_run_analysis(action):
    """Handle Run Full Analysis button click."""
    company_name = action.payload.get("value")
    session_id = cl.user_session.get("session_id")
    session = session_store.get(session_id) if session_id else None
    
    data = {
        "company_name": company_name,
        "ticker": "",
        "query": f"Analyze the latest SEC filings for {company_name}"
    }
    
    await cl.Message(content=f"🚀 Starting full analysis for **{company_name}**...").send()
    await _handle_pipeline(data, session_id, session)


@cl.action_callback("continue_session")
async def on_continue_session(action):
    session: UserSession = cl.user_session.get("session")
    if session:
        await cl.Message(
            content=f"▶ Resuming session for **{session.company_name or 'your documents'}**. What would you like to do?"
        ).send()


@cl.action_callback("start_fresh")
async def on_start_fresh(action):
    # Emit Chainlit's native clear session event to restart the thread correctly
    try:
        await cl.context.emitter.emit("clear_session", {})
    except Exception as exc:
        logger.warning("Failed to emit clear_session: %s", exc)
        # Fallback if emitter fails
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
    
        await show_welcome(session)


@cl.action_callback("remove_doc")
async def on_remove_doc(action):
    """Delete a document from the documents table and remove its Qdrant points."""
    doc_id = action.payload.get("doc_id", "") if action.payload else ""
    user_id = action.payload.get("user_id", "") if action.payload else ""

    # Fallback: get user_id from current session if not in payload
    if not user_id:
        session: UserSession = cl.user_session.get("session")
        user_id = session.user_identifier if session else "anonymous"

    if not doc_id:
        await cl.Message(content="⚠️ No document ID provided.").send()
        return

    # Soft-delete from Postgres and get metadata for Qdrant cleanup
    result = session_store.delete_document(doc_id, user_id)
    if not result:
        await cl.Message(
            content="⚠️ Document not found or you don't have permission to delete it."
        ).send()
        return

    # Delete Qdrant points belonging to this document
    collection_name = result["collection_name"]
    deleted_count = 0
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
        existing = [c.name for c in client.get_collections().collections]
        if collection_name in existing:
            # Scroll to find all point IDs for this doc_id
            offset = None
            point_ids = []
            while True:
                scroll_result = client.scroll(
                    collection_name=collection_name,
                    scroll_filter=Filter(
                        must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                    ),
                    limit=250,
                    offset=offset,
                    with_payload=False,
                )
                points, next_offset = scroll_result
                point_ids.extend(p.id for p in points)
                if next_offset is None:
                    break
                offset = next_offset

            if point_ids:
                from qdrant_client.models import PointIdsList
                client.delete(
                    collection_name=collection_name,
                    points_selector=PointIdsList(points=point_ids)
                )
                deleted_count = len(point_ids)
                logger.info(
                    "Deleted %d Qdrant points for doc %s from collection %s",
                    deleted_count, doc_id, collection_name
                )
    except Exception as exc:
        logger.warning("Qdrant cleanup failed for doc %s: %s", doc_id, exc)

    await cl.Message(
        content=f"✅ Document deleted. {deleted_count} chunk{'s' if deleted_count != 1 else ''} removed from the vector store."
    ).send()


@cl.action_callback("confirm_delete_doc")
async def on_confirm_delete_doc(action):
    """Handle the confirm button from /delete flow — delegates to remove_doc logic."""
    await on_remove_doc(action)


@cl.action_callback("cancel_delete")
async def on_cancel_delete(action):
    await cl.Message(content="❌ Deletion cancelled.").send()



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
