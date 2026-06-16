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
            return OrchestratorResponse(
                content="Hello! I'm your Financial Research Assistant. I can analyze SEC filings, run risk assessments, or answer specific financial questions about companies. How can I help you today?",
                intent=intent
            )
            
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

    async def _handle_rag_chat(self, user_input: str, session_id: str) -> OrchestratorResponse:
        """Handle a standard Q&A chat turn."""
        return OrchestratorResponse(
            content="",  # Content will be streamed by the UI layer calling chat.agent
            intent=UserIntent.RAG_CHAT.value,
            action_required="trigger_chat_agent"
        )

