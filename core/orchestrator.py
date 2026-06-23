"""
Unified Orchestrator — single entry point for all user interactions.

Replaces the dual-profile Chainlit UI logic. It takes user input,
validates attachments/URLs, determines intent, and delegates to the
appropriate pipeline (Research Graph, Chat Agent, or pure UI response).

Usage:
    from core.orchestrator import UnifiedOrchestrator

    orchestrator = UnifiedOrchestrator()
    response = await orchestrator.process_message(user_input, session_id, attachments)
"""

import logging
from typing import Any, Optional

from core.intent_router import route_intent, UserIntent, IntentResult
from core.domain_validator import validate_url_domain

logger = logging.getLogger(__name__)


class OrchestratorResponse:
    """Standardized response from the Orchestrator to the UI."""
    def __init__(
        self,
        content: str,
        intent: str,
        success: bool = True,
        action_required: Optional[str] = None,
        data: Optional[dict] = None,
    ):
        self.content = content
        self.intent = intent
        self.success = success
        self.action_required = action_required
        self.data = data or {}


class UnifiedOrchestrator:
    """Manages the full lifecycle of a user request."""

    async def process_message(
        self,
        user_input: str,
        session_id: str,
        attachments: list[Any] = None,
    ) -> OrchestratorResponse:
        """Process a user message and route to the correct sub-system.
        
        Args:
            user_input: The raw text from the user.
            session_id: Unique session identifier.
            attachments: List of file attachments (e.g. Chainlit elements).
            
        Returns:
            OrchestratorResponse indicating what to display to the user.
        """
        attachments = attachments or []
        
        # 1. Handle explicit attachments first
        if attachments:
            IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
            pdf_files = [a for a in attachments if getattr(a, "path", "").lower().endswith(".pdf")]
            img_files = [a for a in attachments if getattr(a, "path", "").lower().endswith(IMAGE_EXTS)]
            if pdf_files:
                return await self._handle_file_upload(pdf_files[0], session_id, user_input)
            if img_files:
                return await self._handle_image_upload(img_files[0], session_id, user_input)
                
        # 2. Intent Routing (returns IntentResult)
        intent_result: IntentResult = route_intent(user_input, session_id)
        intent = intent_result.intent

        # 3. Intercept generic report-related or analysis-related requests for clarification
        input_lower = user_input.lower().strip()
        is_report_related = any(kw in input_lower for kw in ("report", "memo", "analysis", "synthesis", "pdf", "docx"))
        
        # Check if the user is asking about past/existing reports in history
        is_history_query = False
        check_keywords = {"past", "previous", "existing", "history", "stored", "saved", "have any", "any past", "list my", "what report", "see any"}
        if is_report_related and any(kw in input_lower for kw in check_keywords):
            is_history_query = True
        elif any(dm in input_lower for dm in ("do you have a report", "do you have any reports", "are there any reports", "show past reports", "list reports", "what reports do we have")):
            is_history_query = True

        if is_history_query:
            try:
                from sessions.session_store import session_store
                session = session_store.get(session_id)
                user_id = session.user_identifier if session else "anonymous"
                user_sessions = session_store.get_all_by_user(user_id) if user_id else []
                
                reports_found = []
                for s in user_sessions:
                    if s.last_pipeline_result and (s.last_pipeline_result.get("report") or s.last_pipeline_result.get("memo")):
                        comp = s.last_pipeline_result.get("company_name", s.company_name or "Unknown Company")
                        reports_found.append({
                            "company_name": comp,
                            "session_id": s.session_id,
                            "rating": s.last_pipeline_result.get("rating"),
                            "has_memo": bool(s.last_pipeline_result.get("memo")),
                            "has_report": bool(s.last_pipeline_result.get("report")),
                        })
                
                seen = set()
                unique_reports = []
                for r in reports_found:
                    if r["company_name"] not in seen:
                        seen.add(r["company_name"])
                        unique_reports.append(r)
                        
                return OrchestratorResponse(
                    content="",
                    intent=intent,
                    action_required="show_past_reports_list",
                    data={
                        "reports": unique_reports
                    }
                )
            except Exception as exc:
                logger.warning("Error checking history reports in orchestrator: %s", exc)

        is_unclear = False
        if is_report_related:
            generic_words = {"give", "me", "i", "want", "show", "get", "download", "export", "view", "the", "as", "pdf", "docx", "report", "memo", "analysis", "synthesis", "final", "latest"}
            words = set(input_lower.split())
            if words.issubset(generic_words) or not intent_result.extracted_entity:
                is_unclear = True
        elif intent == UserIntent.FULL_PIPELINE.value and not intent_result.extracted_entity:
            is_unclear = True

        if is_unclear:
            try:
                from sessions.session_store import session_store
                session = session_store.get(session_id)
                user_id = session.user_identifier if session else "anonymous"
                user_sessions = session_store.get_all_by_user(user_id) if user_id else []
                
                existing_company = ""
                existing_session_id = ""
                
                # Check current session first
                if session and session.last_pipeline_result and (session.last_pipeline_result.get("report") or session.last_pipeline_result.get("memo")):
                    existing_company = session.last_pipeline_result.get("company_name", session.company_name or "the company")
                    existing_session_id = session.session_id
                else:
                    # Check other sessions
                    for s in user_sessions:
                        if s.last_pipeline_result and (s.last_pipeline_result.get("report") or s.last_pipeline_result.get("memo")):
                            existing_company = s.last_pipeline_result.get("company_name", s.company_name or "the company")
                            existing_session_id = s.session_id
                            break
                            
                if existing_company and existing_session_id:
                    return OrchestratorResponse(
                        content="",
                        intent=intent,
                        action_required="ask_report_clarification",
                        data={
                            "has_existing": True,
                            "company_name": existing_company,
                            "target_session_id": existing_session_id
                        }
                    )
                else:
                    return OrchestratorResponse(
                        content="",
                        intent=intent,
                        action_required="ask_report_clarification",
                        data={
                            "has_existing": False
                        }
                    )
            except Exception as exc:
                logger.warning("Error checking for existing reports in orchestrator: %s", exc)
        
        # 4. Delegate based on intent
        if intent == UserIntent.FULL_PIPELINE.value:
            return await self._handle_full_pipeline(user_input, session_id, intent_result)
            
        elif intent == UserIntent.SINGLE_AGENT.value:
            return await self._handle_single_agent(user_input, session_id, intent_result)
            
        elif intent == UserIntent.INGEST_DOC.value:
            return await self._handle_url_ingestion(user_input, session_id)
            
        elif intent == UserIntent.EXPORT.value:
            return OrchestratorResponse(
                content="Please use the export buttons provided after analysis.",
                intent=intent,
                action_required="show_export_buttons"
            )
            
        elif intent == UserIntent.VIEW_SOURCES.value:
            return OrchestratorResponse(
                content="Here are the sources for the previous response.",
                intent=intent,
                action_required="open_source_panel"
            )
            
        elif intent == UserIntent.CHITCHAT.value:
            # Handle conversational queries using the chat agent but without RAG retrieval
            return await self._handle_rag_chat(user_input, session_id, skip_rag=True)
            
        elif intent == UserIntent.OUT_OF_DOMAIN.value:
            return OrchestratorResponse(
                content=(
                    "I'm a specialized financial research assistant. I can only help "
                    "with investment analysis, document research, company filings, and "
                    "related financial topics. Please rephrase your query accordingly."
                ),
                intent=intent,
                success=False
            )
            
        # Default fallback to RAG Chat
        else:
            return await self._handle_rag_chat(user_input, session_id)

    async def _handle_file_upload(self, file_element, session_id: str, user_input: str) -> OrchestratorResponse:
        """Handle PDF upload."""
        filename = getattr(file_element, "name", "document.pdf")
        
        return OrchestratorResponse(
            content=f"Received `{filename}` for ingestion.",
            intent=UserIntent.INGEST_DOC.value,
            action_required="trigger_pdf_ingestion",
            data={
                "file_path": file_element.path,
                "file_name": filename,
                "company_name": user_input.strip() or filename.replace(".pdf", "")
            }
        )

    async def _handle_image_upload(self, file_element, session_id: str, user_input: str) -> OrchestratorResponse:
        """Handle image file (.png/.jpg/.jpeg) upload."""
        filename = getattr(file_element, "name", "image.png")
        
        return OrchestratorResponse(
            content=f"Received image `{filename}` for ingestion.",
            intent=UserIntent.INGEST_DOC.value,
            action_required="trigger_image_ingestion",
            data={
                "file_path": file_element.path,
                "file_name": filename,
                "company_name": user_input.strip() or filename.rsplit(".", 1)[0],
            }
        )

    async def _handle_url_ingestion(self, user_input: str, session_id: str) -> OrchestratorResponse:
        """Handle /ingest <URL>."""
        parts = user_input.split(maxsplit=1)
        if len(parts) < 2:
            return OrchestratorResponse(
                content="⚠️ Please provide a URL. Example: `/ingest https://sec.gov/...`",
                intent=UserIntent.INGEST_DOC.value,
                success=False
            )
            
        url = parts[1].strip()
        
        # Validate domain before we even try to fetch
        is_valid, reason = validate_url_domain(url)
        if not is_valid:
            return OrchestratorResponse(
                content=f"❌ **Invalid URL:** {reason}",
                intent=UserIntent.INGEST_DOC.value,
                success=False
            )
            
        return OrchestratorResponse(
            content=f"🔗 Ingesting from URL: `{url}`...",
            intent=UserIntent.INGEST_DOC.value,
            action_required="trigger_url_ingestion",
            data={"url": url}
        )

    async def _handle_full_pipeline(self, user_input: str, session_id: str, intent_result: IntentResult) -> OrchestratorResponse:
        """Trigger the main research pipeline."""
        company = intent_result.extracted_entity or user_input
        ticker = ""
        
        return OrchestratorResponse(
            content=f"🔬 Starting full research pipeline for **{company}** {f'({ticker})' if ticker else ''}...",
            intent=UserIntent.FULL_PIPELINE.value,
            action_required="trigger_research_pipeline",
            data={
                "company_name": company,
                "ticker": ticker,
                "query": user_input
            }
        )

    async def _handle_single_agent(self, user_input: str, session_id: str, intent_result: IntentResult) -> OrchestratorResponse:
        """Trigger just one agent."""
        agent = intent_result.extracted_entity or "metrics"
        company = user_input
        
        return OrchestratorResponse(
            content=f"⚙️ Running the **{agent} agent** for **{company}**...",
            intent=UserIntent.SINGLE_AGENT.value,
            action_required="trigger_single_agent",
            data={
                "company_name": company,
                "target_agent": agent,
                "query": user_input
            }
        )

    async def _handle_rag_chat(self, user_input: str, session_id: str, skip_rag: bool = False) -> OrchestratorResponse:
        """Handle a standard Q&A chat turn."""
        return OrchestratorResponse(
            content="",  # Content will be streamed by the UI layer calling chat.agent
            intent=UserIntent.RAG_CHAT.value if not skip_rag else UserIntent.CHITCHAT.value,
            action_required="trigger_chat_agent",
            data={"skip_rag": skip_rag}
        )

