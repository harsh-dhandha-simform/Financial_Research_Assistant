"""
Chainlit UI — Financial Research Analyst.

Two Chat Profiles:
  📊 Research Pipeline — Ingest docs → Analyse → Output Fork (Memo/Report/DOCX/PDF)
  💬 Chat Q&A          — RAG-based conversational agent with memory

Output Fork gives users interactive buttons to choose their preferred format.

Run: chainlit run ui/app.py --port 8001
"""

import logging
import os
import sys
import tempfile
import uuid

# Ensure project root is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import chainlit as cl

from chat.agent import chat, store_pipeline_context, clear_session
from graph.workflow import run_research
from ingestion.jina_reader import ingest_url
from ingestion.pdf_reader import ingest_pdf
from chunking.chunker import chunk_document
from retrieval.qdrant_store import QdrantStore
from output.markdown_exporter import memo_to_markdown, report_to_markdown
from output.docx_exporter import memo_to_docx, report_to_docx
from output.pdf_exporter import memo_to_pdf, report_to_pdf

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
)

# Temp directory for exports
EXPORT_DIR = os.path.join(tempfile.gettempdir(), "financial_assistant_exports")
os.makedirs(EXPORT_DIR, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# Chat Profiles — two modes
# ═════════════════════════════════════════════════════════════════════════════


@cl.set_chat_profiles
async def chat_profiles():
    return [
        cl.ChatProfile(
            name="Research Pipeline",
            markdown_description="📊 **Ingest SEC filings → Run multi-agent analysis → Generate Memo / Report**\n\nUpload PDFs or paste EDGAR URLs, then run the full research pipeline with Output Fork.",
            icon="https://api.iconify.design/material-symbols/analytics-rounded.svg",
        ),
        cl.ChatProfile(
            name="Chat Q&A",
            markdown_description="💬 **Ask questions about analysed companies using RAG retrieval**\n\nConversational agent with memory, source attribution, and query rewriting.",
            icon="https://api.iconify.design/material-symbols/chat-bubble-outline-rounded.svg",
        ),
    ]


# ═════════════════════════════════════════════════════════════════════════════
# Ingestion helpers
# ═════════════════════════════════════════════════════════════════════════════


def _chunk_and_store(raw_doc) -> dict:
    """Chunk a RawDocument and store in Qdrant. Returns stats."""
    parents, children = chunk_document(raw_doc)
    parent_lookup = {p.chunk_id: p for p in parents}
    store = QdrantStore()
    stored = store.upsert_chunks(children, parent_lookup)
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
    """Initialize session based on the selected chat profile."""
    session_id = f"cl-{uuid.uuid4().hex[:8]}"
    cl.user_session.set("session_id", session_id)
    cl.user_session.set("has_analysis", False)
    cl.user_session.set("company_name", "")
    cl.user_session.set("ticker", "")
    cl.user_session.set("ingested", False)
    # Store memo/report objects (not just markdown) for export
    cl.user_session.set("memo_obj", None)
    cl.user_session.set("report_obj", None)

    profile = cl.user_session.get("chat_profile")

    if profile == "Chat Q&A":
        await _start_chat_profile()
    else:
        await _start_research_profile()


async def _start_research_profile():
    """Welcome message for the Research Pipeline profile."""
    await cl.Message(
        content=(
            "# 📊 Research Pipeline\n\n"
            "Welcome to the **Financial Research Analyst** pipeline.\n\n"
            "### How it works:\n\n"
            "**Step 1 — Ingest Documents** (choose one):\n"
            "- 📎 **Upload a PDF** — drag & drop an SEC filing (10-K, 10-Q, etc.)\n"
            "- 🔗 **Paste a URL** — type `/ingest https://www.sec.gov/Archives/...`\n\n"
            "**Step 2 — Run Analysis:**\n"
            "- Type the company name and ticker, e.g.: `Apple Inc. AAPL`\n"
            "- The system runs **4 agents in parallel** (Metrics, Risk, News, Synthesis)\n\n"
            "**Step 3 — Output Fork:**\n"
            "- After analysis completes, you'll get interactive buttons to choose:\n"
            "  - 📋 Investment Memo (concise 1-page summary)\n"
            "  - 📑 Full Research Report (detailed multi-section)\n"
            "  - 📥 Export as DOCX or PDF\n\n"
            "💡 *You can skip Step 1 — agents will use web search if no documents are ingested.*\n\n"
            "---\n"
            "**Commands:** `/ingest <URL>` · `/clear` · `/help`"
        )
    ).send()


async def _start_chat_profile():
    """Welcome message for the Chat Q&A profile."""
    has_analysis = cl.user_session.get("has_analysis")
    company = cl.user_session.get("company_name")

    context_note = ""
    if has_analysis and company:
        context_note = (
            f"\n\n✅ **Pipeline context available** for **{company}**. "
            f"I can answer questions about the analysis results!\n"
        )
    else:
        context_note = (
            "\n\n⚠️ *No analysis has been run yet. Switch to the Research Pipeline "
            "profile to analyse a company first, or just ask general financial questions.*\n"
        )

    await cl.Message(
        content=(
            "# 💬 Chat Q&A\n\n"
            "Ask me anything about companies, SEC filings, financial metrics, "
            "risk factors, or market sentiment.\n\n"
            "I use **RAG retrieval** from ingested documents plus **conversation memory** "
            "to provide accurate, source-backed answers."
            f"{context_note}\n"
            "---\n"
            "**Commands:** `/clear` (reset memory) · `/help`"
        )
    ).send()


# ═════════════════════════════════════════════════════════════════════════════
# Message handler — routes based on active profile
# ═════════════════════════════════════════════════════════════════════════════


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages — route by chat profile."""
    profile = cl.user_session.get("chat_profile")
    session_id = cl.user_session.get("session_id")
    user_input = message.content.strip()

    # ── Global commands ───────────────────────────────────────────────
    if user_input.lower().startswith("/clear"):
        clear_session(session_id)
        cl.user_session.set("has_analysis", False)
        cl.user_session.set("ingested", False)
        cl.user_session.set("memo_obj", None)
        cl.user_session.set("report_obj", None)
        await cl.Message(content="🗑️ Session cleared. Start fresh!").send()
        return

    if user_input.lower().startswith("/help"):
        await _show_help(profile)
        return

    # ── Route by profile ──────────────────────────────────────────────
    if profile == "Chat Q&A":
        await _handle_chat(user_input, session_id)
    else:
        # Research Pipeline
        await _handle_research_input(message, user_input, session_id)


# ═════════════════════════════════════════════════════════════════════════════
# Research Pipeline handlers
# ═════════════════════════════════════════════════════════════════════════════


async def _handle_research_input(message: cl.Message, user_input: str, session_id: str):
    """Handle input in the Research Pipeline profile."""

    # ── File uploads ──────────────────────────────────────────────────
    if message.elements:
        for element in message.elements:
            if hasattr(element, "path") and element.path and element.path.endswith(".pdf"):
                await _handle_pdf_upload(element, session_id, user_input)
                return

    # ── /ingest <URL> ─────────────────────────────────────────────────
    if user_input.lower().startswith("/ingest"):
        await _handle_ingest_url(user_input, session_id)
        return

    # ── Otherwise: treat as company name + ticker → run analysis ─────
    await _handle_analyze(user_input, session_id)


async def _handle_pdf_upload(element, session_id: str, user_input: str):
    """Handle an uploaded PDF file — ingest into Qdrant."""
    filename = getattr(element, "name", "uploaded.pdf")
    msg = cl.Message(content=f"📄 **Processing `{filename}`...**")
    await msg.send()

    try:
        with open(element.path, "rb") as f:
            file_bytes = f.read()

        company_name = user_input.strip() if user_input.strip() else filename.replace(".pdf", "")

        raw_doc = ingest_pdf(
            file_bytes=file_bytes,
            file_name=filename,
            company_name=company_name,
        )
        stats = _chunk_and_store(raw_doc)

        cl.user_session.set("ingested", True)
        cl.user_session.set("company_name", company_name)

        msg.content = (
            f"✅ **Ingested `{filename}`**\n\n"
            f"| Metric | Count |\n|--------|-------|\n"
            f"| Pages extracted | **{stats['pages']}** |\n"
            f"| Parent chunks | **{stats['parents']}** |\n"
            f"| Child chunks | **{stats['children']}** |\n"
            f"| Stored in Qdrant | **{stats['stored']}** |\n\n"
            f"Now type the **company name and ticker** to run analysis.\n"
            f"Example: `{company_name} AAPL`"
        )
        await msg.update()

    except Exception as exc:
        msg.content = f"❌ **Ingestion failed:** {exc}"
        await msg.update()
        logger.error("PDF upload ingestion failed: %s", exc)


async def _handle_ingest_url(user_input: str, session_id: str):
    """Handle /ingest <URL> command."""
    parts = user_input.split(maxsplit=1)
    if len(parts) < 2:
        await cl.Message(
            content="⚠️ Usage: `/ingest <URL>`\nExample: `/ingest https://www.sec.gov/Archives/edgar/data/320193/...`"
        ).send()
        return

    url = parts[1].strip()
    msg = cl.Message(content=f"🔗 **Ingesting from URL...**\n`{url[:80]}`")
    await msg.send()

    try:
        company_name = cl.user_session.get("company_name") or "Unknown"
        raw_doc = ingest_url(url=url, company_name=company_name)
        stats = _chunk_and_store(raw_doc)

        cl.user_session.set("ingested", True)

        msg.content = (
            f"✅ **URL ingested**\n\n"
            f"| Metric | Count |\n|--------|-------|\n"
            f"| Pages extracted | **{stats['pages']}** |\n"
            f"| Parent chunks | **{stats['parents']}** |\n"
            f"| Child chunks | **{stats['children']}** |\n"
            f"| Stored in Qdrant | **{stats['stored']}** |\n\n"
            f"Now type the **company name and ticker** to run analysis."
        )
        await msg.update()

    except Exception as exc:
        msg.content = f"❌ **URL ingestion failed:** {exc}"
        await msg.update()
        logger.error("URL ingestion failed: %s", exc)


async def _handle_analyze(user_input: str, session_id: str):
    """Run the full research pipeline, then present the Output Fork."""
    words = user_input.split()
    ticker = ""
    company_name = user_input

    # Try to extract ticker (last word if uppercase and <= 5 chars)
    if len(words) >= 2 and words[-1].isupper() and len(words[-1]) <= 5:
        ticker = words[-1]
        company_name = " ".join(words[:-1])

    cl.user_session.set("company_name", company_name)
    cl.user_session.set("ticker", ticker)

    ingested = cl.user_session.get("ingested")
    ingest_note = "" if ingested else "\n⚠️ *No documents ingested — agents will rely on web search.*"

    msg = cl.Message(
        content=(
            f"🔬 **Analysing {company_name} ({ticker or 'no ticker'})...**\n\n"
            f"Running 4 agents in parallel: Metrics · Risk · News · Synthesis\n"
            f"This takes 30-60 seconds.{ingest_note}"
        )
    )
    await msg.send()

    try:
        state = run_research(
            query=f"Analyse the latest SEC filings for {company_name} ({ticker})",
            company_name=company_name,
            ticker=ticker,
            session_id=session_id,
        )

        # Store pipeline context for the Chat Q&A profile
        if state.synthesis_output:
            ctx_parts = [f"Company: {company_name} ({ticker})"]
            ctx_parts.append(f"Rating: {state.synthesis_output.rating.value}")
            ctx_parts.append(f"Thesis: {state.synthesis_output.investment_thesis}")
            if state.memo:
                ctx_parts.append(f"\n--- MEMO ---\n{memo_to_markdown(state.memo)}")
            store_pipeline_context(session_id, "\n".join(ctx_parts))
            cl.user_session.set("has_analysis", True)

        # Store memo and report objects for export
        if state.memo:
            cl.user_session.set("memo_obj", state.memo)
        if state.report:
            cl.user_session.set("report_obj", state.report)

        # Show pipeline summary
        if state.synthesis_output:
            rating = state.synthesis_output.rating.value.replace("_", " ").upper()
            confidence = state.synthesis_output.confidence

            await cl.Message(
                content=(
                    f"✅ **Analysis Complete for {company_name} ({ticker})**\n\n"
                    f"**Rating:** {rating} · **Confidence:** {confidence:.0%}\n\n"
                    f"---"
                )
            ).send()

        if state.errors:
            await cl.Message(
                content="⚠️ **Pipeline warnings:**\n" + "\n".join(f"- {e}" for e in state.errors)
            ).send()

        # ── Present the Output Fork ──────────────────────────────────
        await _present_output_fork()

    except Exception as exc:
        await cl.Message(content=f"❌ **Pipeline failed:** {exc}").send()
        logger.error("Pipeline failed in Chainlit: %s", exc)


# ═════════════════════════════════════════════════════════════════════════════
# Output Fork — interactive user choice
# ═════════════════════════════════════════════════════════════════════════════


async def _present_output_fork():
    """Present the Output Fork as interactive buttons.

    The user chooses how they want to view/export the results:
      - View Memo (markdown in chat)
      - View Report (markdown in chat)
      - View Both
      - Export Memo as DOCX
      - Export Memo as PDF
      - Export Report as DOCX
      - Export Report as PDF
    """
    memo_obj = cl.user_session.get("memo_obj")
    report_obj = cl.user_session.get("report_obj")

    if not memo_obj and not report_obj:
        await cl.Message(content="⚠️ No output available — synthesis may have failed.").send()
        return

    actions = []
    if memo_obj:
        actions.append(cl.Action(name="view_memo", label="📋 View Investment Memo", value="view_memo"))
    if report_obj:
        actions.append(cl.Action(name="view_report", label="📑 View Full Report", value="view_report"))
    if memo_obj and report_obj:
        actions.append(cl.Action(name="view_both", label="📋📑 View Both", value="view_both"))

    # Export options
    if memo_obj:
        actions.append(cl.Action(name="export_memo_docx", label="📥 Export Memo (DOCX)", value="export_memo_docx"))
        actions.append(cl.Action(name="export_memo_pdf", label="📥 Export Memo (PDF)", value="export_memo_pdf"))
    if report_obj:
        actions.append(cl.Action(name="export_report_docx", label="📥 Export Report (DOCX)", value="export_report_docx"))
        actions.append(cl.Action(name="export_report_pdf", label="📥 Export Report (PDF)", value="export_report_pdf"))

    res = await cl.AskActionMessage(
        content=(
            "## 📤 Output Fork\n\n"
            "Choose how you'd like to view or export the results:\n\n"
            "**View in chat:**\n"
            "- 📋 **Investment Memo** — concise 1-page executive summary\n"
            "- 📑 **Full Research Report** — detailed multi-section analyst report\n\n"
            "**Export as file:**\n"
            "- 📥 Download as **DOCX** or **PDF** for offline use"
        ),
        actions=actions,
        timeout=300,
    ).send()

    if res:
        await _handle_output_fork_choice(res.get("value"))


async def _handle_output_fork_choice(choice: str):
    """Handle the user's Output Fork selection."""
    memo_obj = cl.user_session.get("memo_obj")
    report_obj = cl.user_session.get("report_obj")
    company = cl.user_session.get("company_name") or "company"
    ticker = cl.user_session.get("ticker") or ""
    safe_name = company.replace(" ", "_").lower()

    if choice == "view_memo" and memo_obj:
        memo_md = memo_to_markdown(memo_obj)
        await cl.Message(content=f"## 📋 Investment Memo\n\n{memo_md}").send()

    elif choice == "view_report" and report_obj:
        report_md = report_to_markdown(report_obj)
        await cl.Message(content=f"## 📑 Full Research Report\n\n{report_md}").send()

    elif choice == "view_both":
        if memo_obj:
            memo_md = memo_to_markdown(memo_obj)
            await cl.Message(content=f"## 📋 Investment Memo\n\n{memo_md}").send()
        if report_obj:
            report_md = report_to_markdown(report_obj)
            await cl.Message(content=f"## 📑 Full Research Report\n\n{report_md}").send()

    elif choice == "export_memo_docx" and memo_obj:
        path = os.path.join(EXPORT_DIR, f"{safe_name}_memo.docx")
        memo_to_docx(memo_obj, path)
        elements = [cl.File(name=f"{safe_name}_memo.docx", path=path, display="inline")]
        await cl.Message(content=f"📥 **Investment Memo — DOCX**", elements=elements).send()

    elif choice == "export_memo_pdf" and memo_obj:
        path = os.path.join(EXPORT_DIR, f"{safe_name}_memo.pdf")
        memo_to_pdf(memo_obj, path)
        elements = [cl.File(name=f"{safe_name}_memo.pdf", path=path, display="inline")]
        await cl.Message(content=f"📥 **Investment Memo — PDF**", elements=elements).send()

    elif choice == "export_report_docx" and report_obj:
        path = os.path.join(EXPORT_DIR, f"{safe_name}_report.docx")
        report_to_docx(report_obj, path)
        elements = [cl.File(name=f"{safe_name}_report.docx", path=path, display="inline")]
        await cl.Message(content=f"📥 **Research Report — DOCX**", elements=elements).send()

    elif choice == "export_report_pdf" and report_obj:
        path = os.path.join(EXPORT_DIR, f"{safe_name}_report.pdf")
        report_to_pdf(report_obj, path)
        elements = [cl.File(name=f"{safe_name}_report.pdf", path=path, display="inline")]
        await cl.Message(content=f"📥 **Research Report — PDF**", elements=elements).send()

    else:
        await cl.Message(content="⚠️ That output is not available.").send()
        return

    # ── Offer follow-up actions ──────────────────────────────────────
    await _offer_followup_actions()


async def _offer_followup_actions():
    """After showing an output, offer follow-up choices."""
    memo_obj = cl.user_session.get("memo_obj")
    report_obj = cl.user_session.get("report_obj")

    actions = []
    if memo_obj:
        actions.append(cl.Action(name="view_memo", label="📋 View Memo", value="view_memo"))
        actions.append(cl.Action(name="export_memo_docx", label="📥 Memo DOCX", value="export_memo_docx"))
        actions.append(cl.Action(name="export_memo_pdf", label="📥 Memo PDF", value="export_memo_pdf"))
    if report_obj:
        actions.append(cl.Action(name="view_report", label="📑 View Report", value="view_report"))
        actions.append(cl.Action(name="export_report_docx", label="📥 Report DOCX", value="export_report_docx"))
        actions.append(cl.Action(name="export_report_pdf", label="📥 Report PDF", value="export_report_pdf"))

    if not actions:
        return

    res = await cl.AskActionMessage(
        content="**Need another format?** Pick an option below, or just type to continue.",
        actions=actions,
        timeout=120,
    ).send()

    if res:
        await _handle_output_fork_choice(res.get("value"))


# ═════════════════════════════════════════════════════════════════════════════
# Chat Q&A handler
# ═════════════════════════════════════════════════════════════════════════════


async def _handle_chat(question: str, session_id: str):
    """Handle a chat question with RAG retrieval."""
    msg = cl.Message(content="")
    await msg.send()

    try:
        response = chat(
            question=question,
            session_id=session_id,
        )

        answer = response.answer

        # Add sources if available
        if response.sources and response.relevant:
            answer += "\n\n---\n📚 **Sources:**\n"
            for src in response.sources[:5]:
                answer += f"- {src}\n"

        if response.rewritten_query:
            answer += f"\n*🔄 Query rewritten: \"{response.rewritten_query}\"*"

        msg.content = answer
        await msg.update()

    except Exception as exc:
        msg.content = f"❌ Error: {exc}"
        await msg.update()
        logger.error("Chat failed: %s", exc)


# ═════════════════════════════════════════════════════════════════════════════
# Help command
# ═════════════════════════════════════════════════════════════════════════════


async def _show_help(profile: str):
    """Show help based on the active profile."""
    if profile == "Chat Q&A":
        await cl.Message(
            content=(
                "## 💬 Chat Q&A — Help\n\n"
                "| Action | How |\n"
                "|--------|-----|\n"
                "| Ask a question | Just type your question |\n"
                "| Clear memory | `/clear` |\n"
                "| Switch to pipeline | Use the profile selector in the header |\n\n"
                "**Tips:**\n"
                "- I use RAG to search through ingested documents\n"
                "- If you've run an analysis, I can answer questions about it\n"
                "- My query rewriter resolves pronouns from conversation context"
            )
        ).send()
    else:
        await cl.Message(
            content=(
                "## 📊 Research Pipeline — Help\n\n"
                "| Step | Action |\n"
                "|------|--------|\n"
                "| 1. Upload PDF | Drag & drop an SEC filing |\n"
                "| 2. Or ingest URL | `/ingest <EDGAR URL>` |\n"
                "| 3. Run analysis | Type `Company Name TICKER` |\n"
                "| 4. Choose output | Click a button from the Output Fork |\n\n"
                "**Commands:**\n"
                "- `/ingest <URL>` — ingest from an EDGAR URL\n"
                "- `/clear` — clear session and start fresh\n"
                "- `/help` — show this message"
            )
        ).send()
