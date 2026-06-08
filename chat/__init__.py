"""
Chat package — conversational RAG agent for post-research Q&A.

Public API:
    from chat import chat, store_pipeline_context, clear_session
"""

from chat.agent import chat, store_pipeline_context, clear_session

__all__ = ["chat", "store_pipeline_context", "clear_session"]
