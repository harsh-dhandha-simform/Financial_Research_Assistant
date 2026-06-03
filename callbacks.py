"""
Langfuse callback handler factory.

Import get_langfuse_handler() in every module that makes LLM calls.
Pass the handler in config={"callbacks": [handler]} on every
graph.invoke(), llm.invoke(), and chain.invoke().

Usage:
    from callbacks import get_langfuse_handler

    handler = get_langfuse_handler(session_id="abc", trace_name="metrics-agent")
    result = llm.invoke(prompt, config={"callbacks": [handler]})
    result = graph.invoke(state, config={"callbacks": [handler]})
"""

from langfuse.callback import CallbackHandler

from config import settings


def get_langfuse_handler(**kwargs) -> CallbackHandler:
    """Create a fresh Langfuse CallbackHandler for tracing.

    Keyword arguments are forwarded to CallbackHandler for trace metadata:
        session_id  — groups related traces into a session
        user_id     — identifies the end-user
        trace_name  — custom name shown in Langfuse UI
        tags        — list of string tags for filtering

    Returns:
        A configured CallbackHandler ready to pass into LangChain/LangGraph
        config={"callbacks": [handler]}.
    """
    return CallbackHandler(
        secret_key=settings.langfuse_secret_key,
        public_key=settings.langfuse_public_key,
        host=settings.langfuse_base_url,
        **kwargs,
    )
